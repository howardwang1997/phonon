from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read
from e3nn import o3
from mace.modules import ScaleShiftMACE
from mace.modules.blocks import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "smearing_kink"
DATA = ROOT / "data" / "graphene_r2o_taylor_null_core"
sys.path.insert(0, str(SCRIPTS))

import graphene_r2o_routed_tail as routed  # noqa: E402
from graphene_r2o_routed_tail import (  # noqa: E402
    EXPECTED_INVARIANT_WIDTH,
    EXPECTED_RAW_NODE_WIDTH,
    FORMAL_BUNDLE_KIND,
    FORMAL_GATE_STATUS,
    MAX_FROZEN_GRAPH_SOURCE_QUANTIZATION_A,
    OUTER_LOCO_EPOCHS,
    OUTER_LOCO_SELECTION_POLICY,
    LoadedFormalR2O,
    R2ORoutedTailSpecification,
    R2OTaylorNullRoutedTail,
    audit_core_tail_taylor_linearity,
    audit_paired_native_fp64_sensitivity,
    build_order_mic_query,
    center_train8_prediction_and_target,
    frozen_model_state_sha256,
    frozen_graph_semantics,
    held_energy_error_same_train8_gauge,
    load_formal_r2o_bundle,
    make_outer_loco_contract,
    query_raw_invariants,
    raw_tail_energy,
    r2o_invariant_schema,
    r2o_node_invariants,
    r2o_raw_node_features,
    sha256,
    source_order_tail_forces,
    state_dict_sha256,
    tail_taylor_remainder,
    train8_relative_energy_mse,
    validate_outer_loco_contract,
    validate_loaded_formal_r2o,
    validate_r2o_invariant_schema,
    whole_energy_taylor2_null,
)


def _make_formal_shape_mace() -> ScaleShiftMACE:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(83)
            model = ScaleShiftMACE(
                r_max=3.2,
                num_bessel=24,
                num_polynomial_cutoff=5,
                max_ell=2,
                interaction_cls=RealAgnosticResidualInteractionBlock,
                interaction_cls_first=RealAgnosticInteractionBlock,
                num_interactions=2,
                num_elements=1,
                hidden_irreps=o3.Irreps("16x0e+16x1o+16x2e"),
                MLP_irreps=o3.Irreps("16x0e"),
                atomic_energies=np.asarray([0.0]),
                avg_num_neighbors=12.0,
                atomic_numbers=[6],
                correlation=3,
                gate=torch.nn.functional.silu,
                pair_repulsion=False,
                atomic_inter_scale=1.0,
                atomic_inter_shift=0.0,
                heads=["Default"],
            ).double()
    finally:
        torch.set_default_dtype(previous)
    model.requires_grad_(False)
    model.eval()
    return model


def _random_tail(pristine: torch.Tensor) -> R2OTaylorNullRoutedTail:
    model = R2OTaylorNullRoutedTail(
        R2ORoutedTailSpecification(),
        np.zeros(EXPECTED_INVARIANT_WIDTH, dtype=np.float64),
        np.ones(EXPECTED_INVARIANT_WIDTH, dtype=np.float64),
        pristine,
    )
    generator = torch.Generator().manual_seed(83)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                0.08
                * torch.randn(
                    parameter.shape, generator=generator, dtype=torch.float64
                )
            )
        model.router_encoder[-1].bias.fill_(0.5)
    return model


def _hessian_from_force(force: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = [
        -torch.autograd.grad(value, positions, retain_graph=True)[0].reshape(-1)
        for value in force.reshape(-1)
    ]
    return torch.stack(rows)


def test_r2o_schema_is_new_fp64_3p2A_and_bound_to_formal_core(
    formal_shape_loaded: LoadedFormalR2O,
) -> None:
    schema = r2o_invariant_schema(formal_shape_loaded)
    validate_r2o_invariant_schema(schema)
    assert schema["version"] == "graphene_r2o_raw_mace_invariants_v1"
    assert schema["core_r_max_A"] == 3.2
    assert schema["dtype"] == "torch.float64"
    assert schema["raw_node_feats_width"] == 160
    assert schema["output_dimension"] == 64
    assert schema["raw_MACE_energy_policy"] == (
        "descriptor_only_direct_prediction_forbidden"
    )
    assert schema["MIC_policy"] == (
        "integer_image_detached_continuous_displacement_live"
    )
    semantics = schema["graph_semantics"]
    assert semantics == frozen_graph_semantics()
    assert semantics["formal_AtomicData_source_default_dtype"] == "torch.float32"
    assert semantics["native_FP64_comparator_source_default_dtype"] == (
        "torch.float64"
    )
    assert semantics["formal_graph_source_geometry_bound_A"] == 1.0e-6
    assert len(semantics["graph_semantics_sha256"]) == 64


def test_r2o_schema_rejects_rehashed_policy_or_binding_change(
    formal_shape_loaded: LoadedFormalR2O,
) -> None:
    schema = r2o_invariant_schema(formal_shape_loaded)
    changed = copy.deepcopy(schema)
    changed["raw_MACE_energy_policy"] = "direct_prediction_allowed"
    body = dict(changed)
    body.pop("schema_sha256")
    changed["schema_sha256"] = routed.canonical_json_sha256(body)
    with pytest.raises(ValueError, match="raw_MACE_energy_policy"):
        validate_r2o_invariant_schema(changed)
    changed = copy.deepcopy(schema)
    changed["formal_binding"]["model_state_sha256"] = "aa" * 32
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_r2o_invariant_schema(changed)
    original = formal_shape_loaded.binding
    formal_shape_loaded.binding = routed.FormalR2OBinding(
        model_state_sha256="aa" * 32,
        wrapper_sha256=original.wrapper_sha256,
        data_manifest_sha256=original.data_manifest_sha256,
        reference_6x6_sha256=original.reference_6x6_sha256,
        reference_8x8_sha256=original.reference_8x8_sha256,
    )
    try:
        with pytest.raises(ValueError, match="provenance|model state"):
            r2o_invariant_schema(formal_shape_loaded)
    finally:
        formal_shape_loaded.binding = original


def test_tail_and_invariants_fail_closed_on_float32(
    formal_shape_loaded: LoadedFormalR2O,
) -> None:
    with pytest.raises(ValueError, match="FP64"):
        R2OTaylorNullRoutedTail(
            R2ORoutedTailSpecification(),
            np.zeros(64, dtype=np.float32),
            np.ones(64, dtype=np.float64),
            np.zeros((2, 64), dtype=np.float64),
        )
    schema = r2o_invariant_schema(formal_shape_loaded)
    with pytest.raises(ValueError, match="FP64"):
        r2o_node_invariants(torch.zeros(2, 160, dtype=torch.float32), schema)


def test_tail_whole_energy_null_removes_its_own_E_F_H_and_keeps_cubic() -> None:
    dtype = torch.float64
    reference = torch.tensor(
        [[-0.4, 0.1], [0.2, 0.3], [0.1, -0.2]], dtype=dtype
    )

    def invariants(positions: torch.Tensor) -> torch.Tensor:
        centered = positions - positions.mean(dim=0)
        radius2 = centered.square().sum(dim=1, keepdim=True)
        columns = [
            torch.sin((index + 1) * radius2 / 64.0)
            + (index + 1) * centered[:, :1] / 128.0
            for index in range(64)
        ]
        return torch.cat(columns, dim=1)

    pristine = invariants(reference).detach()
    tail = _random_tail(pristine)
    batch = torch.zeros(3, dtype=torch.long)

    def raw_tail(position: torch.Tensor) -> torch.Tensor:
        return tail.raw_graph_energies(invariants(position), batch)[0]

    at_reference = reference.clone().requires_grad_(True)
    result = whole_energy_taylor2_null(
        raw_tail,
        at_reference,
        reference,
        torch.zeros_like(reference),
        create_graph=True,
    )
    assert abs(float(result.energy)) <= 2.0e-14
    assert float(result.forces_reference_order.abs().max()) <= 3.0e-14
    hessian = _hessian_from_force(result.forces_reference_order, at_reference)
    assert float(hessian.abs().max()) <= 5.0e-12

    displacement = torch.tensor(
        [[0.03, -0.01], [-0.02, 0.025], [0.01, -0.015]], dtype=dtype
    )
    current = (reference + displacement).clone().requires_grad_(True)
    away = whole_energy_taylor2_null(
        raw_tail,
        current,
        reference,
        torch.zeros_like(reference),
        create_graph=False,
    )
    assert abs(float(away.energy)) > 1.0e-12
    assert float(away.forces_reference_order.abs().max()) > 1.0e-11


def test_tail_router_and_query_features_are_live() -> None:
    pristine = torch.zeros(3, 64, dtype=torch.float64)
    tail = _random_tail(pristine)
    query = (0.2 * torch.randn(3, 64, dtype=torch.float64)).requires_grad_(True)
    _, gate, _, epsilon = tail.node_terms(query)
    gauge = tail.pristine_carbon_gauge()
    full = torch.sum(gate * (epsilon - gauge))
    full_gradient = torch.autograd.grad(full, query, retain_graph=True)[0]
    detached_router = torch.sum(gate.detach() * (epsilon - gauge))
    shortcut_gradient = torch.autograd.grad(detached_router, query)[0]
    assert float(torch.max(torch.abs(full_gradient - shortcut_gradient))) > 1.0e-9


def test_prediction_and_target_use_the_same_train8_energy_gauge() -> None:
    prediction = torch.linspace(-0.2, 0.3, 8, dtype=torch.float64).requires_grad_(True)
    target = torch.linspace(0.1, -0.4, 8, dtype=torch.float64)
    pred_centered, target_centered = center_train8_prediction_and_target(
        prediction, target
    )
    torch.testing.assert_close(pred_centered.mean(), torch.zeros((), dtype=torch.float64))
    torch.testing.assert_close(target_centered.mean(), torch.zeros((), dtype=torch.float64))
    base = train8_relative_energy_mse(prediction, target)
    shifted = train8_relative_energy_mse(prediction + 17.0, target - 23.0)
    torch.testing.assert_close(base, shifted, atol=2.0e-14, rtol=0.0)
    gradient = torch.autograd.grad(base, prediction)[0]
    assert torch.isfinite(gradient).all()
    held = held_energy_error_same_train8_gauge(
        torch.tensor(0.7, dtype=torch.float64),
        prediction.detach(),
        torch.tensor(-0.2, dtype=torch.float64),
        target,
    )
    shifted_held = held_energy_error_same_train8_gauge(
        torch.tensor(11.7, dtype=torch.float64),
        prediction.detach() + 11.0,
        torch.tensor(-13.2, dtype=torch.float64),
        target - 13.0,
    )
    torch.testing.assert_close(held, shifted_held)
    with pytest.raises(ValueError, match="train8"):
        center_train8_prediction_and_target(prediction[:7], target[:7])


def test_combined_raw_scalar_taylor_null_is_linear_in_energy_and_force() -> None:
    reference = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float64)
    current = (reference + 0.04).clone().requires_grad_(True)

    def core_energy(position: torch.Tensor) -> torch.Tensor:
        return (
            0.7 * position.square().sum()
            + 0.2 * position.pow(3).sum()
        ).reshape(1)

    def tail_energy(position: torch.Tensor) -> torch.Tensor:
        return (
            -0.4 * position.sum()
            + 0.17 * position.square().sum()
            + 0.09 * position.pow(4).sum()
        ).reshape(1)

    arguments = (current, reference, torch.zeros_like(reference))
    core = whole_energy_taylor2_null(core_energy, *arguments, create_graph=False)
    tail = whole_energy_taylor2_null(tail_energy, *arguments, create_graph=False)
    combined = whole_energy_taylor2_null(
        lambda value: core_energy(value) + tail_energy(value),
        *arguments,
        create_graph=False,
    )
    torch.testing.assert_close(
        combined.energy, core.energy + tail.energy, atol=2.0e-14, rtol=0.0
    )
    torch.testing.assert_close(
        combined.forces_reference_order,
        core.forces_reference_order + tail.forces_reference_order,
        atol=3.0e-13,
        rtol=0.0,
    )


def test_fixed_reference_raw_features_are_order_and_mic_safe(
    formal_shape_loaded: LoadedFormalR2O,
) -> None:
    formal_shape_core = formal_shape_loaded.model
    reference = read(DATA / "reference_6x6.xyz", index=0)
    structure = reference.copy()
    structure.positions[4] += np.asarray([0.01, -0.02, 0.015])
    first = build_order_mic_query(formal_shape_loaded, structure)
    raw = r2o_raw_node_features(formal_shape_loaded, first)
    official_data = dict(first.data)
    official_data["positions"] = first.aligned_positions
    official = formal_shape_core(
        official_data, training=False, compute_force=False
    )["node_feats"]
    assert raw.shape == (72, EXPECTED_RAW_NODE_WIDTH)
    torch.testing.assert_close(raw, official, atol=2.0e-12, rtol=0.0)

    generator = np.random.default_rng(83)
    order = generator.permutation(len(structure))
    transformed = structure[order]
    transformed.set_cell(structure.cell, scale_atoms=False)
    transformed.pbc = structure.pbc
    source_index = int(np.flatnonzero(order == 4)[0])
    transformed.positions[source_index] += np.asarray(structure.cell[0])
    second = build_order_mic_query(formal_shape_loaded, transformed)
    assert np.any(second.assignment.image_integer_reference_order != 0)
    assert second.reference_atom_count == 72
    assert second.reference_artifact_sha256 == (
        formal_shape_loaded.binding.reference_6x6_sha256
    )
    assert second.frozen_graph_source_quantization_max_A <= (
        MAX_FROZEN_GRAPH_SOURCE_QUANTIZATION_A
    )
    assert float(
        torch.max(torch.abs(second.aligned_positions - first.aligned_positions))
    ) <= 5.0e-7
    raw_second = r2o_raw_node_features(formal_shape_loaded, second)
    # The frozen formal wrapper's AtomicData construction can quantize the cell
    # at float32 before casting it to FP64.  Its audited image/permutation gate
    # is 1e-6/1e-5; keep this explicit instead of silently changing the wrapper.
    torch.testing.assert_close(raw_second, raw, atol=1.0e-6, rtol=0.0)
    direction = torch.randn_like(second.current_positions_reference_order)
    derivative = torch.autograd.grad(
        torch.sum(second.aligned_positions * direction),
        second.current_positions_reference_order,
    )[0]
    torch.testing.assert_close(derivative, direction)

    schema = r2o_invariant_schema(formal_shape_loaded)
    invariants = query_raw_invariants(formal_shape_loaded, second, schema)
    assert invariants.shape == (72, EXPECTED_INVARIANT_WIDTH)
    sensitivity = audit_paired_native_fp64_sensitivity(
        formal_shape_loaded, second
    )
    assert sensitivity["source_default_dtype_pair"] == [
        "torch.float32",
        "torch.float64",
    ]
    assert sensitivity["graph_semantics_sha256"] == second.graph_semantics_sha256
    assert sensitivity["passes_paired_native_FP64_sensitivity"] is True, sensitivity
    assert sensitivity["energy_size_normalization"]["atom_count"] == 72
    assert sensitivity["energy_size_normalization"][
        "derived_total_energy_cap_eV"
    ] == pytest.approx(1.0e-6)
    tampered_query = copy.copy(second)
    tampered_query.reference_artifact_sha256 = "aa" * 32
    with pytest.raises(ValueError, match="formal 6x6/8x8 reference"):
        query_raw_invariants(formal_shape_loaded, tampered_query, schema)
    tampered_positions = copy.copy(second)
    tampered_positions.reference_positions = second.reference_positions.clone()
    tampered_positions.reference_positions[0, 0] += 1.0e-5
    with pytest.raises(ValueError, match="reference-position tensor changed"):
        query_raw_invariants(formal_shape_loaded, tampered_positions, schema)
    tampered_graph = copy.copy(second)
    tampered_graph.data = dict(second.data)
    tampered_graph.data["shifts"] = second.data["shifts"].clone()
    tampered_graph.data["shifts"][0, 0] += 1.0e-5
    with pytest.raises(ValueError, match="graph content changed"):
        r2o_raw_node_features(formal_shape_loaded, tampered_graph)
    tampered_image_shift = copy.copy(second)
    tampered_image_shift.image_shift = second.image_shift.clone()
    tampered_image_shift.image_shift[0, 0] += 0.1
    with pytest.raises(ValueError, match="frozen-semantics image shift changed"):
        query_raw_invariants(formal_shape_loaded, tampered_image_shift, schema)
    tampered_native_shift = copy.copy(second)
    tampered_native_shift.native_fp64_image_shift = (
        second.native_fp64_image_shift.clone()
    )
    tampered_native_shift.native_fp64_image_shift[0, 0] += 0.1
    with pytest.raises(ValueError, match="native-FP64 image shift changed"):
        query_raw_invariants(formal_shape_loaded, tampered_native_shift, schema)
    tampered_mic = copy.copy(second)
    tampered_mic.assignment = copy.deepcopy(second.assignment)
    tampered_mic.assignment.image_integer_reference_order[0, 0] += 1
    with pytest.raises(ValueError, match="assignment MIC integers changed"):
        query_raw_invariants(formal_shape_loaded, tampered_mic, schema)
    tampered_mapping = copy.copy(second)
    tampered_mapping.assignment = copy.deepcopy(second.assignment)
    reference_to_source = tampered_mapping.assignment.reference_to_source
    reference_to_source[[0, 1]] = reference_to_source[[1, 0]]
    tampered_mapping.assignment.source_to_reference[:] = np.argsort(
        reference_to_source
    )
    tampered_mapping.assignment.validate(72)
    with pytest.raises(ValueError, match="assignment reference_to_source changed"):
        query_raw_invariants(formal_shape_loaded, tampered_mapping, schema)
    tampered_schema = copy.deepcopy(schema)
    tampered_schema["formal_binding"]["wrapper_sha256"] = "aa" * 32
    body = dict(tampered_schema)
    body.pop("schema_sha256")
    tampered_schema["schema_sha256"] = routed.canonical_json_sha256(body)
    with pytest.raises(ValueError, match="another formal bundle"):
        query_raw_invariants(formal_shape_loaded, second, tampered_schema)

    reference_8x8 = formal_shape_loaded.reference_for_count(128)
    query_8x8 = build_order_mic_query(formal_shape_loaded, reference_8x8)
    assert 5.0e-7 < query_8x8.frozen_graph_source_quantization_max_A <= (
        MAX_FROZEN_GRAPH_SOURCE_QUANTIZATION_A
    )
    sensitivity_8x8 = audit_paired_native_fp64_sensitivity(
        formal_shape_loaded, query_8x8
    )
    assert sensitivity_8x8["passes_paired_native_FP64_sensitivity"] is True, (
        sensitivity_8x8
    )
    assert sensitivity_8x8["energy_size_normalization"]["atom_count"] == 128
    assert sensitivity_8x8["energy_size_normalization"][
        "derived_total_energy_cap_eV"
    ] == pytest.approx(128.0e-6 / 72.0)
    assert sensitivity_8x8["energy_size_normalization"][
        "raw_scalar_energy_max_abs_total_eV"
    ] <= sensitivity_8x8["energy_size_normalization"][
        "derived_total_energy_cap_eV"
    ]


def test_public_tail_api_actual_6x6_null_linearity_order_mic_and_force_hessian_fd(
    formal_shape_loaded: LoadedFormalR2O,
) -> None:
    reference = formal_shape_loaded.reference_for_count(72)
    schema = r2o_invariant_schema(formal_shape_loaded)
    pristine_query = build_order_mic_query(formal_shape_loaded, reference)
    pristine = query_raw_invariants(
        formal_shape_loaded, pristine_query, schema
    ).detach()
    tail = _random_tail(pristine)

    displaced = reference.copy()
    displaced.positions[4] += np.asarray([0.012, -0.007, 0.018])
    query = build_order_mic_query(formal_shape_loaded, displaced)
    raw_energy = raw_tail_energy(
        formal_shape_loaded,
        tail,
        query,
        query.aligned_positions,
        schema,
    )
    raw_force = -torch.autograd.grad(raw_energy.sum(), query.current_positions_reference_order)[0]
    assert torch.isfinite(raw_energy).all() and torch.isfinite(raw_force).all()
    assert float(raw_force.abs().max()) > 1.0e-10

    at_reference = tail_taylor_remainder(
        formal_shape_loaded,
        tail,
        schema,
        pristine_query,
        create_graph=True,
    )
    assert abs(float(at_reference.energy.detach())) <= 1.0e-12
    assert float(at_reference.forces_reference_order.detach().abs().max()) <= 1.0e-11
    hessian = _hessian_from_force(
        at_reference.forces_reference_order,
        pristine_query.current_positions_reference_order,
    )
    assert hessian.shape == (216, 216)
    assert float(hessian.abs().max()) <= 1.0e-8
    assert float(torch.max(torch.abs(hessian - hessian.T))) <= 1.0e-8
    asr = hessian.reshape(72, 3, 72, 3).sum(dim=2)
    assert float(asr.abs().max()) <= 1.0e-8

    step = 2.0e-4
    plus = reference.copy()
    minus = reference.copy()
    plus.positions[4, 2] += step
    minus.positions[4, 2] -= step
    plus_query = build_order_mic_query(formal_shape_loaded, plus)
    minus_query = build_order_mic_query(formal_shape_loaded, minus)
    plus_result = tail_taylor_remainder(
        formal_shape_loaded, tail, schema, plus_query, create_graph=False
    )
    minus_result = tail_taylor_remainder(
        formal_shape_loaded, tail, schema, minus_query, create_graph=False
    )
    numerical_force_derivative = (
        plus_result.forces_reference_order - minus_result.forces_reference_order
    ).reshape(-1) / (2.0 * step)
    selected = 3 * 4 + 2
    analytic_force_derivative = -hessian[:, selected]
    assert float(
        torch.max(torch.abs(numerical_force_derivative - analytic_force_derivative))
    ) <= 1.0e-5

    away = tail_taylor_remainder(
        formal_shape_loaded, tail, schema, query, create_graph=False
    )
    linearity = audit_core_tail_taylor_linearity(
        formal_shape_loaded, tail, schema, query
    )
    assert linearity.energy_max_abs_difference_eV <= 1.0e-10
    assert linearity.force_max_abs_difference_eV_A <= 1.0e-9

    order = np.random.default_rng(29).permutation(72)
    transformed = displaced[order]
    transformed.set_cell(displaced.cell, scale_atoms=False)
    transformed.pbc = displaced.pbc
    source_index = int(np.flatnonzero(order == 4)[0])
    transformed.positions[source_index] += np.asarray(displaced.cell[0])
    transformed_query = build_order_mic_query(formal_shape_loaded, transformed)
    transformed_result = tail_taylor_remainder(
        formal_shape_loaded, tail, schema, transformed_query, create_graph=False
    )
    source_force = source_order_tail_forces(
        formal_shape_loaded, transformed_result, transformed_query
    )
    torch.testing.assert_close(
        source_force[transformed_query.assignment.reference_to_source],
        away.forces_reference_order,
        atol=1.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        transformed_result.energy, away.energy, atol=1.0e-6, rtol=0.0
    )
    tampered_source_query = copy.copy(transformed_query)
    tampered_source_query.assignment = copy.deepcopy(transformed_query.assignment)
    reference_to_source = tampered_source_query.assignment.reference_to_source
    reference_to_source[[0, 1]] = reference_to_source[[1, 0]]
    tampered_source_query.assignment.source_to_reference[:] = np.argsort(
        reference_to_source
    )
    tampered_source_query.assignment.validate(72)
    with pytest.raises(ValueError, match="assignment reference_to_source changed"):
        source_order_tail_forces(
            formal_shape_loaded, transformed_result, tampered_source_query
        )
    paired = audit_paired_native_fp64_sensitivity(
        formal_shape_loaded, query, tail=tail, schema=schema
    )
    assert paired["mode"] == "raw_core_plus_tail"
    assert paired["passes_paired_native_FP64_sensitivity"] is True, paired


def _write_formal_portable_tree(
    root: Path,
    *,
    dtype: torch.dtype = torch.float64,
    model: torch.nn.Module | None = None,
) -> dict[str, str]:
    artifacts = root / "formal"
    code = root / "code"
    data = root / "data"
    artifacts.mkdir(parents=True)
    code.mkdir()
    data.mkdir()
    wrapper = SCRIPTS / "graphene_r2o_taylor_null.py"
    shutil.copyfile(wrapper, code / wrapper.name)
    shutil.copyfile(DATA / "manifest.json", data / "manifest.json")
    shutil.copyfile(DATA / "reference_6x6.xyz", data / "reference_6x6.xyz")
    shutil.copyfile(DATA / "reference_8x8.xyz", data / "reference_8x8.xyz")
    for marker in ("TRAINING_DONE", "CORE_GATE_PASSED"):
        (artifacts / marker).write_bytes(b"")
    (artifacts / "DONE").write_text(FORMAL_GATE_STATUS + "\n", encoding="utf-8")
    (artifacts / "EXIT_CODE").write_text("0\n", encoding="utf-8")

    if model is None:
        model = torch.nn.Linear(3, 2).to(dtype=dtype)
    else:
        model = copy.deepcopy(model).to(dtype=dtype)
        model.requires_grad_(False)
        model.eval()
    model_state = state_dict_sha256(model)
    binding_hashes = {
        "wrapper_sha256": sha256(code / wrapper.name),
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "reference_6x6_sha256": sha256(data / "reference_6x6.xyz"),
        "reference_8x8_sha256": sha256(data / "reference_8x8.xyz"),
    }
    metadata = {
        "kind": FORMAL_BUNDLE_KIND,
        "model_state_sha256": model_state,
        **binding_hashes,
        "raw_MACE_direct_deployment_forbidden": True,
        "wrapper_snapshot": "/remote/run/code_snapshots/graphene_r2o_taylor_null.py",
    }
    bundle_path = artifacts / "selected_or_diagnostic_bundle.pt"
    torch.save(
        {
            "format": "graphene_r2o_deployment_bundle_v1",
            "raw_MACE_must_not_be_deployed_without_Taylor_wrapper": True,
            "model": model,
            "metadata": metadata,
        },
        bundle_path,
    )
    bundle_hash = sha256(bundle_path)
    gate = {
        "format": "graphene_r2o_fixed_checkpoint_gate_v1",
        "status": FORMAL_GATE_STATUS,
        "smoke": False,
        "postcore_or_deployment_authorized": True,
        "seed2_or_support_read": False,
        "implementation_pass": True,
        "cutoff_C2_pass": True,
        "reference_null_pass": True,
        "inputs": {"data_manifest_sha256": binding_hashes["data_manifest_sha256"]},
        "bundle": {
            "kind": FORMAL_BUNDLE_KIND,
            "path": "/remote/run/selected_or_diagnostic_bundle.pt",
            "sha256": bundle_hash,
            "model_state_sha256": model_state,
            **binding_hashes,
        },
    }
    gate_path = artifacts / "core_checkpoint_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    paths = {
        "gate": "formal/core_checkpoint_gate.json",
        "bundle": "formal/selected_or_diagnostic_bundle.pt",
        "wrapper": "code/graphene_r2o_taylor_null.py",
        "data_manifest": "data/manifest.json",
        "reference_6x6": "data/reference_6x6.xyz",
        "reference_8x8": "data/reference_8x8.xyz",
        "DONE": "formal/DONE",
        "TRAINING_DONE": "formal/TRAINING_DONE",
        "CORE_GATE_PASSED": "formal/CORE_GATE_PASSED",
        "EXIT_CODE": "formal/EXIT_CODE",
    }
    receipt = {
        "format": "graphene_r2o_formal_portable_bundle_receipt_v1",
        "status": "frozen_R2O_formal_bundle_for_postcore",
        "formal_core": {
            "gate_status": FORMAL_GATE_STATUS,
            "bundle_kind": FORMAL_BUNDLE_KIND,
            "postcore_or_deployment_authorized": True,
            "model_state_sha256": model_state,
        },
        "paths": paths,
        "sha256": {key: sha256(root / value) for key, value in paths.items()},
        "portability": {
            "path_mode": "relative_to_portable_root",
            "symlinks_allowed": False,
            "content_hash_required_for_every_path": True,
        },
        "forbidden": {
            "support9_opened": False,
            "seed2_opened": False,
            "raw_MACE_direct_deployment": False,
        },
    }
    receipt_path = root / "formal_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return {
        "receipt": sha256(receipt_path),
        "gate": sha256(gate_path),
        "bundle": bundle_hash,
    }


@pytest.fixture(scope="module")
def formal_shape_loaded(tmp_path_factory: pytest.TempPathFactory) -> LoadedFormalR2O:
    root = tmp_path_factory.mktemp("r2o_formal_shape")
    hashes = _write_formal_portable_tree(root, model=_make_formal_shape_mace())
    return load_formal_r2o_bundle(
        root,
        "formal_receipt.json",
        expected_receipt_sha256=hashes["receipt"],
        expected_gate_sha256=hashes["gate"],
        expected_bundle_sha256=hashes["bundle"],
        device="cpu",
    )


def test_formal_bundle_loader_is_hash_bound_relocatable_and_freezes_raw_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = tmp_path / "first"
    hashes = _write_formal_portable_tree(first_root)
    monkeypatch.setattr(routed, "validate_mace_architecture", lambda model: {})
    loaded = load_formal_r2o_bundle(
        first_root,
        "formal_receipt.json",
        expected_receipt_sha256=hashes["receipt"],
        expected_gate_sha256=hashes["gate"],
        expected_bundle_sha256=hashes["bundle"],
        device="cpu",
    )
    assert loaded.gate["status"] == FORMAL_GATE_STATUS
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    assert all(parameter.dtype == torch.float64 for parameter in loaded.model.parameters())
    assert set(loaded.reference_paths) == {72, 128}
    validate_loaded_formal_r2o(loaded)

    second_root = tmp_path / "relocated"
    shutil.copytree(first_root, second_root)
    relocated = load_formal_r2o_bundle(
        second_root,
        "formal_receipt.json",
        expected_receipt_sha256=hashes["receipt"],
        expected_gate_sha256=hashes["gate"],
        expected_bundle_sha256=hashes["bundle"],
        device="cpu",
    )
    assert relocated.binding == loaded.binding

    reference_path = loaded.reference_paths[72]
    reference_bytes = reference_path.read_bytes()
    reference_path.write_bytes(reference_bytes + b"\n")
    with pytest.raises(ValueError, match="reference artifact changed"):
        loaded.reference_for_count(72)
    reference_path.write_bytes(reference_bytes)

    before = frozen_model_state_sha256(loaded.model)
    with torch.no_grad():
        loaded.model.weight.add_(0.125)
    after = frozen_model_state_sha256(loaded.model)
    assert before != after
    with pytest.raises(ValueError, match="semantic model state"):
        validate_loaded_formal_r2o(loaded)


def test_bundle_loader_rejects_nonformal_gate_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "portable"
    hashes = _write_formal_portable_tree(root)
    gate_path = root / "formal" / "core_checkpoint_gate.json"
    gate = json.loads(gate_path.read_text())
    gate["status"] = "R2O_two_epoch_smoke_passed"
    gate["smoke"] = True
    gate["postcore_or_deployment_authorized"] = False
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    receipt_path = root / "formal_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["sha256"]["gate"] = sha256(gate_path)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch_load must not run for a diagnostic gate")

    monkeypatch.setattr(routed, "torch_load", forbidden_load)
    with pytest.raises(ValueError, match="not passing/frozen"):
        load_formal_r2o_bundle(
            root,
            "formal_receipt.json",
            expected_receipt_sha256=sha256(receipt_path),
            expected_gate_sha256=sha256(gate_path),
            expected_bundle_sha256=hashes["bundle"],
            device="cpu",
        )
    assert called is False


def test_frozen_model_hash_recomputes_after_copy_and_load_state_dict() -> None:
    model = torch.nn.Linear(3, 2).double()
    model.requires_grad_(False)
    first = frozen_model_state_sha256(model)
    with torch.no_grad():
        model.bias.copy_(model.bias + 0.25)
    second = frozen_model_state_sha256(model)
    assert second != first
    replacement = copy.deepcopy(model.state_dict())
    replacement["weight"] = replacement["weight"] + 0.5
    model.load_state_dict(replacement, strict=True)
    third = frozen_model_state_sha256(model)
    assert third not in {first, second}


def test_bundle_loader_rejects_float32_and_forbidden_run_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "float32"
    hashes = _write_formal_portable_tree(root, dtype=torch.float32)
    monkeypatch.setattr(routed, "validate_mace_architecture", lambda model: {})
    with pytest.raises(ValueError, match="entirely FP64"):
        load_formal_r2o_bundle(
            root,
            "formal_receipt.json",
            expected_receipt_sha256=hashes["receipt"],
            expected_gate_sha256=hashes["gate"],
            expected_bundle_sha256=hashes["bundle"],
            device="cpu",
        )

    clean = tmp_path / "running"
    hashes = _write_formal_portable_tree(clean)
    (clean / "formal" / "RUNNING").write_text("stale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden marker RUNNING"):
        load_formal_r2o_bundle(
            clean,
            "formal_receipt.json",
            expected_receipt_sha256=hashes["receipt"],
            expected_gate_sha256=hashes["gate"],
            expected_bundle_sha256=hashes["bundle"],
            device="cpu",
        )


def test_bundle_receipt_rejects_absolute_traversal_and_symlink_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "portable"
    hashes = _write_formal_portable_tree(root)
    receipt_path = root / "formal_receipt.json"
    original = json.loads(receipt_path.read_text())
    for bad_path in ("/tmp/gate.json", "../gate.json", "formal/../gate.json"):
        receipt = copy.deepcopy(original)
        receipt["paths"]["gate"] = bad_path
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="relative path|traversal|canonical"):
            routed.validate_formal_portable_receipt(
                root,
                "formal_receipt.json",
                expected_receipt_sha256=sha256(receipt_path),
                expected_gate_sha256=hashes["gate"],
                expected_bundle_sha256=hashes["bundle"],
            )
    receipt_path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
    link = root / "gate_link.json"
    link.symlink_to(root / "formal" / "core_checkpoint_gate.json")
    receipt = copy.deepcopy(original)
    receipt["paths"]["gate"] = "gate_link.json"
    receipt["sha256"]["gate"] = hashes["gate"]
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="symlinks"):
        routed.validate_formal_portable_receipt(
            root,
            "formal_receipt.json",
            expected_receipt_sha256=sha256(receipt_path),
            expected_gate_sha256=hashes["gate"],
            expected_bundle_sha256=hashes["bundle"],
        )


@pytest.mark.parametrize(
    ("artifact_key", "filename"),
    [
        ("gate", "core_checkpoint_gate.json"),
        ("bundle", "selected_or_diagnostic_bundle.pt"),
    ],
)
def test_formal_gate_bundle_and_markers_must_share_one_run_root(
    tmp_path: Path, artifact_key: str, filename: str
) -> None:
    root = tmp_path / artifact_key
    hashes = _write_formal_portable_tree(root)
    other = root / "other_run"
    other.mkdir()
    source = root / "formal" / filename
    copied = other / filename
    shutil.copyfile(source, copied)
    receipt_path = root / "formal_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["paths"][artifact_key] = f"other_run/{filename}"
    receipt["sha256"][artifact_key] = sha256(copied)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="share one run root"):
        load_formal_r2o_bundle(
            root,
            "formal_receipt.json",
            expected_receipt_sha256=sha256(receipt_path),
            expected_gate_sha256=(
                sha256(copied) if artifact_key == "gate" else hashes["gate"]
            ),
            expected_bundle_sha256=(
                sha256(copied) if artifact_key == "bundle" else hashes["bundle"]
            ),
            device="cpu",
        )


def test_outer_loco_contract_freezes_full_recipe_and_rejects_leakage_paths(
    formal_shape_loaded: LoadedFormalR2O,
) -> None:
    source_root = formal_shape_loaded.portable_root / "sources"
    source_root.mkdir(exist_ok=True)
    module_path = source_root / "graphene_r2o_routed_tail.py"
    test_path = source_root / "test_graphene_r2o_routed_tail.py"
    shutil.copyfile(SCRIPTS / module_path.name, module_path)
    shutil.copyfile(Path(__file__), test_path)
    wrapper_path = formal_shape_loaded.portable_root / "code" / "graphene_r2o_taylor_null.py"
    sources = {
        "r2o_routed_tail_module": {
            "path": "sources/graphene_r2o_routed_tail.py",
            "sha256": sha256(module_path),
        },
        "r2o_taylor_wrapper": {
            "path": "code/graphene_r2o_taylor_null.py",
            "sha256": sha256(wrapper_path),
        },
        "r2o_routed_tail_tests": {
            "path": "sources/test_graphene_r2o_routed_tail.py",
            "sha256": sha256(test_path),
        },
    }
    contract = make_outer_loco_contract(
        formal=formal_shape_loaded,
        source_artifacts=sources,
    )
    validate_outer_loco_contract(contract, root=formal_shape_loaded.portable_root)
    assert contract["training_policy"]["epochs"] == OUTER_LOCO_EPOCHS == 240
    assert contract["training_policy"]["selection_policy"] == (
        OUTER_LOCO_SELECTION_POLICY
    )
    assert contract["energy_gauge"]["prediction"].startswith("R_tail_i_minus")
    assert contract["energy_gauge"]["target"].startswith("d_i_minus")
    assert contract["formal_R2O_condition"][
        "paired_native_FP64_sensitivity_gate_required"
    ] is True
    assert contract["formal_R2O_condition"]["graph_semantics_sha256"] == (
        frozen_graph_semantics()["graph_semantics_sha256"]
    )
    assert contract["leakage"] == {
        "support9_opened": False,
        "seed2_opened": False,
        "held_in_scaler_gradient_gauge_selection": False,
    }
    recipe = contract["training_recipe"]
    assert recipe["feature_scaler"]["fit_separately_per_fold"] is True
    assert "outer held" in recipe["feature_scaler"]["excluded"]
    assert recipe["pristine_carbon_gauge"]["source_artifact"] == (
        "pristine_reference_6x6"
    )
    assert recipe["optimizer"] == {
        "name": "AdamW",
        "amsgrad": True,
        "learning_rate": 1.0e-3,
        "constant_schedule": True,
        "weight_decay": 1.0e-6,
        "gradient_clip_global_norm": 20.0,
        "seed": 83,
        "EMA_decay": 0.99,
    }
    assert recipe["per_epoch_exposure"]["optimizer_steps"] == 8
    assert sum(recipe["loss"]["force_group_mass"].values()) == pytest.approx(1.0)

    changed = copy.deepcopy(contract)
    changed["training_policy"]["epochs"] = 239
    body = dict(changed)
    body.pop("contract_sha256")
    changed["contract_sha256"] = routed.canonical_json_sha256(body)
    with pytest.raises(ValueError, match="training_policy"):
        validate_outer_loco_contract(changed)
    changed_recipe = copy.deepcopy(contract)
    changed_recipe["training_recipe"]["optimizer"]["seed"] = 84
    body = dict(changed_recipe)
    body.pop("contract_sha256")
    changed_recipe["contract_sha256"] = routed.canonical_json_sha256(body)
    with pytest.raises(ValueError, match="training_recipe"):
        validate_outer_loco_contract(changed_recipe)
    for token in ("support9", "seed2", "reserved"):
        forbidden = copy.deepcopy(sources)
        forbidden["r2o_routed_tail_module"] = {
            "path": f"sources/{token}_module.py",
            "sha256": "11" * 32,
        }
        with pytest.raises(ValueError, match="forbidden unopened token"):
            make_outer_loco_contract(
                formal=formal_shape_loaded,
                source_artifacts=forbidden,
            )
    unknown = copy.deepcopy(sources)
    unknown["seed2_reader"] = sources["r2o_routed_tail_module"]
    with pytest.raises(ValueError, match="not whitelisted|forbidden"):
        make_outer_loco_contract(formal=formal_shape_loaded, source_artifacts=unknown)
