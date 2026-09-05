from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "smearing_kink"
sys.path.insert(0, str(SCRIPTS))

import graphene_r2o_taylor_null as r2o  # noqa: E402
import graphene_r2r_multipolar_background as r2r  # noqa: E402


DATA = ROOT / "data" / "graphene_r2o_taylor_null_core"
RUN = (
    ROOT
    / "results"
    / "graphene_physics_temperature"
    / "post_p4_feasibility"
    / "R2Q_four_step_trust_region"
    / "formal_4step_seed83_rtx"
)


def _allowed_paths() -> dict[str, Path]:
    return {
        "endpoint_checkpoint": RUN / "endpoint.pt",
        "endpoint_receipt": RUN / "endpoint_receipt.json",
        "endpoint_marker": RUN / "ENDPOINT_FROZEN",
        "reference_6x6": DATA / "reference_6x6.xyz",
        "reference_8x8": DATA / "reference_8x8.xyz",
        "thermal92": DATA / "train_thermal.xyz",
        "harmonic_zero32": DATA / "train_harmonic_lambda1_small_zero.xyz",
    }


def _allclose_fields(left: r2r.BackgroundFields, right: r2r.BackgroundFields) -> None:
    for name in (
        "covariance_A2",
        "trace_A2",
        "cross_product_square_A4",
        "multipolar_gate",
        "amplitude_gate",
    ):
        assert torch.allclose(
            getattr(left, name), getattr(right, name), atol=2.0e-13, rtol=2.0e-12
        ), name


def test_canonical_contract_and_train_only_input_hashes() -> None:
    assert r2r.CANONICAL_CONTRACT_SHA256 == (
        "e589497c7b9f6a9d4ff0cdcc434cf07804fa0fd732d791dba811d4c7e1c5aecc"
    )
    assert r2r.SCALE_DERIVATION_SHA256 == (
        "c97e5ea994ac0d034cb2c039daff6c01041b4efd8dd612d1b47dbcd6b8681211"
    )
    assert r2r.CANONICAL_CONTRACT["authorization"] == {
        "may_fit": False,
        "may_train": False,
        "may_open_held": False,
        "may_launch_remote": False,
        "may_authorize_R2R-1": False,
    }
    assert r2r.CANONICAL_CONTRACT["encoder_history"] == {
        "gradient_seen_structures": "thermal92 plus harmonic_zero32 only",
        "endpoint_only_once_after_freeze": (
            "E50 seed1 and actual small-H were evaluated once only after the R2Q "
            "endpoint freeze and never wrote back to parameters, directions, or selection"
        ),
        "never_opened": "seed2 and support",
        "R2R0_encoder_update": False,
    }
    gates = r2r.CANONICAL_CONTRACT["fixed_gates"]
    assert gates["runner_override_allowed"] is False
    assert gates["thermal92"] == {"min_b_A4": 1.0e-6, "min_a": 0.99}
    assert gates["harmonic_zero32"] == {
        "max_b_A4": 1.0e-20,
        "max_a": 1.0e-12,
        "max_Taylor_remainder_energy_eV": 1.0e-10,
        "max_Taylor_remainder_force_eV_A": 1.0e-9,
    }
    assert gates["R2Q_node_parity"] == {
        "energy_sum_abs_eV": 1.0e-10,
        "position_gradient_max_abs_eV_A": 1.0e-9,
    }
    assert gates["corrected_carrier_vs_frozen_R2Q_parity"] == {
        "energy_abs_eV": 1.0e-10,
        "force_max_abs_eV_A": 1.0e-9,
        "Hessian_max_abs_eV_A2": 1.0e-7,
        "probes": "reference_6x6 and thermal92_E50_seed0_global_index0",
    }
    assert gates["parameter_rank0_zero_jet"] == {
        "nodewise_a_value_max_abs": 0.0,
        "nodewise_a_Jacobian_max_abs_A-1": 0.0,
        "nodewise_a_Hessian_max_abs_A-2": 0.0,
        "probes": "reference_6x6 and reference_8x8",
        "nonzero_local_production_value_abs": 1.0e-14,
        "nonzero_local_production_gradient_max_abs_A-1": 1.0e-12,
        "nonzero_probe_b_over_beta_abs_difference": 1.0e-12,
        "nonzero_probe_target_a": pytest.approx(1.0 - math.exp(-1.0)),
    }
    assert gates["O3_proper_and_improper"] == {
        "energy_abs_eV": 1.0e-6,
        "force_covariance_max_abs_eV_A": 1.0e-5,
    }
    assert gates["translation"] == {
        "energy_abs_eV": 1.0e-9,
        "force_max_abs_eV_A": 1.0e-8,
    }
    assert gates["permutation"] == {
        "energy_abs_eV": 1.0e-6,
        "force_max_abs_eV_A": 1.0e-5,
    }
    assert gates["native_cell_wrap_order_MIC_combined_probe"] == {
        "energy_abs_eV": 1.0e-6,
        "force_max_abs_eV_A": 1.0e-5,
        "raw_65_column_difference": "diagnostic_only",
    }
    assert gates["finite_difference"] == {
        "force_abs_eV_A": 1.0e-5,
        "force_Hessian_abs_eV_A2": 1.0e-5,
    }
    assert gates["nonreference_complete_Hessian"] == {
        "antisymmetry_max_abs_eV_A2": 1.0e-7,
        "translation_ASR_max_abs_eV_A2": 1.0e-7,
    }
    assert gates["localized_6x6_to_8x8_full_remainder"] == {
        "total_energy_abs_eV": 1.0e-7,
        "max_force_abs_eV_A": 1.0e-5,
    }
    assert gates["reference_Taylor_remainder"] == {
        "energy_abs_eV": 1.0e-10,
        "force_max_abs_eV_A": 1.0e-9,
        "Hessian_max_abs_eV_A2": 1.0e-7,
        "Hessian_antisymmetry_max_abs_eV_A2": 1.0e-7,
        "Hessian_translation_ASR_max_abs_eV_A2": 1.0e-7,
        "Gamma_K_frequency_drift_cm-1": 2.0,
    }
    assert gates["thermal_force_design"] == {
        "columns": 65,
        "all_columns_active": True,
        "scaled_rank": 65,
        "scaled_condition_number_max": 1.0e8,
    }
    assert gates["quintic_cutoff_inside_limit"] == {
        "value_abs": 1.0e-12,
        "first_derivative_abs_A-1": 1.0e-11,
        "second_derivative_abs_A-2": 1.0e-10,
        "third_derivative_abs_A-3_min": 1.0e-6,
    }
    integration = r2r.CANONICAL_CONTRACT["production_integration"]
    assert integration["only_public_formal_API"] == "production_linear_design_query"
    assert integration["canonical_combination_method"] == (
        "ProductionLinearDesignQuery.canonical_combined_probe"
    )
    assert integration["complete_Hessian_API"] == (
        "production_combined_energy_force_hessian"
    )
    assert integration["linear_design_create_graph"] is False
    assert integration["legacy_66_column_create_graph_mechanics_forbidden"] is True
    assert len(integration["affine_component_names"]) == 66
    assert integration["affine_component_names"][:3] == [
        "fixed_carrier",
        "w0[0]",
        "w0[1]",
    ]
    assert integration["affine_component_names"][33:35] == ["b1", "w1[0]"]
    assert integration["affine_component_names"][-1] == "w1[31]"
    assert integration["manual_primitive_assembly_for_formal_forbidden"] is True
    assert integration["float32_origin_graph_required"] is True
    assert all(
        key in integration["rotated_graph_policy"]
        for key in ("positions", "shifts", "cell", "byte exact")
    )
    assert integration["native_rotated_graph_rebuild_role"].startswith(
        "explicit diagnostic"
    )
    assert "array-exact" in integration["rigid_transform_public_input_covariance"]
    assert integration["rigid_transform_internal_covariance_atol_A"] == 1.0e-12
    assert "exact directed physical edge identity-key set" in integration[
        "native_rebuild_physical_multiset_veto"
    ]
    assert integration["rigid_transform_probe_baseline_templates"][
        "structure_full_semantic_sha256"
    ] == "d3f6eca52753a6407c58d107374e06e80cec470c3668c1c6f1f034ea59488766"
    assert integration["rigid_transform_v4_CPU_regression"]["proper"][
        "derived_graph_sha256"
    ] == "8f40d3565ffbd0e58b5b40dd4a40f35165a2904822b950ef45e7ece3bbcf8f87"
    assert integration["mechanics_probe_coefficients_sha256"] == (
        "656ca438f398dc4373456439322d2c94883789e2f97ac1bbcad66b5bd5713b04"
    )
    assert integration["mechanics_evaluates"] == "actual fixed_offset + X @ p"
    assert integration["node_energy_parity_probes"] == {
        "reference_6x6_semantic_sha256": (
            "2d9d97aa3a994f1bc4db0965a1837d46269b0f588fc19d8c96308848c58ef6ea"
        ),
        "thermal92_E50_seed0_global_index": 0,
        "thermal92_E50_seed0_structure_semantic_sha256": (
            "e64c2c5cc7681f710f01a22e9a374cf900c435387de16edfbb2037b025d941b9"
        ),
    }
    zero_jet_contract = r2r.CANONICAL_CONTRACT["linear_design"][
        "rank0_zero_jet_optimization"
    ]
    assert "direct-pair O(u^4)" in zero_jet_contract["audit_policy"]
    zero_receipts = zero_jet_contract["local_direct_numeric_receipts"]
    assert zero_receipts["reference_6x6"]["node_count"] == 72
    assert zero_receipts["reference_6x6"]["exact_q_signature_count"] == 71
    assert zero_receipts["reference_8x8"]["node_count"] == 128
    assert zero_receipts["reference_8x8"]["exact_q_signature_count"] == 123
    assert zero_receipts["all_node_zero_value_Jacobian_local_full_Hessian_max"] == 0.0
    assert zero_receipts["force_labels_used"] is False
    assert len(integration["mechanics_probe_coefficients"]) == 65
    assert integration["mechanics_probe_coefficients"][0] == -0.2
    assert integration["mechanics_probe_coefficients"][32] == pytest.approx(0.05)
    assert integration["mechanics_probe_coefficients"][64] == 0.3
    assert r2r.CANONICAL_CONTRACT["background"]["no_wrap"] == {
        "6x6_shortest_in_plane_translation_A": 14.76,
        "8x8_shortest_in_plane_translation_A": 19.68,
        "6x6_diameter_margin_A": 1.96,
        "8x8_diameter_margin_A": 6.88,
        "strict_diameter_lt_translation": True,
        "same_source_duplicate_periodic_image_within_6A": False,
    }
    assert r2r.verify_r2r0_input_hashes(_allowed_paths()) == {
        role: item["sha256"] for role, item in r2r.EXPECTED_INPUTS.items()
    }


def test_reference_semantic_payload_canonicalizes_v100_signed_zero_but_rejects_tamper() -> None:
    plus_zero = Atoms(
        "C", positions=np.asarray([[0.0, 0.0, 0.0]]), cell=np.eye(3), pbc=False
    )
    minus_zero = Atoms(
        "C", positions=np.asarray([[-0.0, 0.0, 0.0]]), cell=np.eye(3), pbc=False
    )
    plus_payload = r2r.reference_semantic_payload(plus_zero)
    minus_payload = r2r.reference_semantic_payload(minus_zero)
    assert minus_payload == plus_payload
    assert r2r.reference_semantic_sha256(minus_zero) == r2r.reference_semantic_sha256(plus_zero)
    assert "-0.0" not in r2r._canonical_bytes(minus_payload).decode("ascii")

    reference = read(DATA / "reference_6x6.xyz")
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transformed = reference.copy()
    transformed.positions = np.asarray(reference.positions) @ rotation.T
    transformed.set_cell(np.asarray(reference.cell) @ rotation.T, scale_atoms=False)
    base_payload = r2r.reference_semantic_payload(reference)
    transformed_payload = r2r.reference_semantic_payload(transformed)
    assert transformed_payload == base_payload
    assert "-0.0" not in r2r._canonical_bytes(transformed_payload).decode("ascii")
    assert r2r.validate_reference_semantics(transformed) == (
        r2r.EXPECTED_REFERENCE_SEMANTIC_SHA256["reference_6x6"]
    )

    signed_zero = reference.copy()
    positions = np.asarray(signed_zero.positions, dtype=np.float64).copy()
    positions[positions == 0.0] = -0.0
    signed_zero.positions = positions
    assert r2r.reference_semantic_payload(signed_zero) == base_payload

    tampered = reference.copy()
    tampered.positions[1, 0] += 1.0e-4
    with pytest.raises(ValueError, match="ordered tiling/background origin changed"):
        r2r.validate_reference_semantics(tampered)


def test_path_isolation_rejects_forbidden_lexically_before_file_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _allowed_paths()
    paths["thermal92"] = Path("future_seed1/train_thermal.xyz")

    def forbidden_open(*_args, **_kwargs):
        raise AssertionError("input must be rejected before open")

    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(ValueError, match="forbidden R2R-0 path token"):
        r2r.verify_r2r0_input_hashes(paths)


def test_degree5_reverse_smootherstep_and_center_inclusive_normalization() -> None:
    radius = torch.tensor(
        [0.0, 3.0, 6.0], dtype=torch.float64, requires_grad=True
    )
    weight = r2r.quintic_reference_weight(radius)
    assert torch.equal(weight.detach(), torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64))
    first = torch.autograd.grad(weight.sum(), radius, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), radius)[0]
    assert abs(float(first[0].detach())) < 1.0e-15
    assert abs(float(first[-1].detach())) < 1.0e-15
    assert abs(float(second[0])) < 1.0e-15
    assert abs(float(second[-1])) < 1.0e-15
    metrics = r2r.quintic_inside_cutoff_metrics()
    assert abs(metrics["value"]) <= 1.0e-12
    assert abs(metrics["first_derivative_A-1"]) <= 1.0e-11
    assert abs(metrics["second_derivative_A-2"]) <= 1.0e-10
    assert abs(metrics["third_derivative_A-3"]) > 1.0e-6

    graph = r2r.fixed_reference_neighborhood(read(DATA / "reference_6x6.xyz"))
    degree = torch.bincount(graph.receiver, minlength=graph.atom_count)
    assert torch.equal(degree, torch.full((72,), 39, dtype=torch.long))
    neighbor_sum = torch.zeros(72, dtype=torch.float64)
    neighbor_sum.index_add_(0, graph.receiver, graph.weight)
    assert torch.allclose(graph.normalization, 1.0 + neighbor_sum, atol=2.0e-14, rtol=0.0)
    for atom in range(72):
        sender = graph.sender[graph.receiver == atom]
        assert torch.unique(sender).numel() == 39
    assert r2r.BACKGROUND_INTERACTION_DIAMETER_A < min(
        read(DATA / "reference_6x6.xyz").cell.lengths()[:2]
    )
    graph8 = r2r.fixed_reference_neighborhood(read(DATA / "reference_8x8.xyz"))
    assert graph.shortest_in_plane_translation_A == pytest.approx(14.76)
    assert graph8.shortest_in_plane_translation_A == pytest.approx(19.68)
    assert graph.shortest_in_plane_translation_A - 12.8 == pytest.approx(1.96)
    assert graph8.shortest_in_plane_translation_A - 12.8 == pytest.approx(6.88)
    changed = read(DATA / "reference_6x6.xyz")
    changed.positions[0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="ordered tiling/background origin changed"):
        r2r.fixed_reference_neighborhood(changed)


def test_small_synthetic_all_node_local_direct_zero_jet_audit() -> None:
    receiver = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    sender = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)
    weight = torch.tensor([0.3, 0.7, 0.3, 0.7, 0.3, 0.7], dtype=torch.float64)
    normalization = torch.ones(3, dtype=torch.float64)
    normalization.index_add_(0, receiver, weight)
    graph = SimpleNamespace(
        receiver=receiver,
        sender=sender,
        weight=weight,
        normalization=normalization,
        atom_count=3,
        validate=lambda: None,
    )
    receipt = r2r.background_rank0_zero_jet_audit(graph)
    assert receipt["pass"] is True
    assert receipt["global_all_node_value_and_Jacobian_exact_zero"] is True
    assert receipt["all_nodes_covered_exactly_once"] is True
    assert receipt["all_local_senders_unique"] is True
    assert receipt["local_sample_count"] == 3
    assert receipt["local_Hessian_shape_per_node"] == [3, 3, 3, 3]
    assert receipt["local_all_node_value_max_abs"] == 0.0
    assert receipt["local_all_node_Jacobian_max_abs_A-1"] == 0.0
    assert receipt["local_all_node_Hessian_max_abs_A-2"] == 0.0
    assert receipt["local_all_node_finite"] is True
    for probe in receipt["nonzero_local_production_probe"]:
        assert math.isfinite(probe["base_b_A4"])
        assert probe["base_b_A4"] > 0.0
        assert math.isfinite(probe["scale"])
        assert abs(probe["b_over_beta"] - 1.0) <= 1.0e-12
        assert probe["local_a"] == pytest.approx(1.0 - math.exp(-1.0), abs=1.0e-14)
        assert probe["value_abs_difference"] <= 1.0e-14
        assert probe["mapped_gradient_max_abs_difference_A-1"] <= 1.0e-12
        assert probe["outside_local_global_gradient_max_abs_A-1"] == 0.0


def test_frozen_train_geometry_scale_statistics_and_rank1_boundary() -> None:
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, source, reference_name in (
        ("thermal92", "train_thermal.xyz", "reference_6x6.xyz"),
        (
            "harmonic_zero32",
            "train_harmonic_lambda1_small_zero.xyz",
            "reference_8x8.xyz",
        ),
    ):
        reference = read(DATA / reference_name)
        graph = r2r.fixed_reference_neighborhood(reference)
        traces = []
        cross_squares = []
        for structure in read(DATA / source, ":"):
            ordered = r2r.background_from_structure(
                structure, reference, requires_grad=False
            )
            traces.extend(ordered.fields.trace_A2.detach().numpy().tolist())
            cross_squares.extend(
                ordered.fields.cross_product_square_A4.detach().numpy().tolist()
            )
            assert ordered.graph.atom_count == graph.atom_count
        values[split] = (np.asarray(traces), np.asarray(cross_squares))

    thermal_s, thermal_b = values["thermal92"]
    harmonic_s, harmonic_b = values["harmonic_zero32"]
    assert np.median(thermal_s) == pytest.approx(0.014458039461185433, abs=2.0e-16)
    assert np.min(thermal_b) == pytest.approx(1.8338961456148413e-6, rel=2.0e-12)
    assert np.median(thermal_b) == pytest.approx(2.6026297280e-5, rel=2.0e-10)
    assert np.max(harmonic_b) == pytest.approx(2.2896187649234325e-34, rel=2.0e-12)
    assert np.max(harmonic_b) < r2r.RANK1_NUMERICAL_B_GATE_A4
    assert np.max(-harmonic_b) <= 0.0
    assert np.max(-thermal_b) <= 0.0
    assert np.max(harmonic_s) < 1.0e-3


def test_background_o3_translation_rank1_order_and_mic() -> None:
    reference = read(DATA / "reference_6x6.xyz")
    graph = r2r.fixed_reference_neighborhood(reference)
    generator = torch.Generator().manual_seed(773)
    displacement = 0.03 * torch.randn((72, 3), generator=generator, dtype=torch.float64)
    base = r2r.multipolar_background(displacement, graph)

    translated = r2r.multipolar_background(
        displacement + torch.tensor([0.31, -0.27, 0.19], dtype=torch.float64), graph
    )
    _allclose_fields(base, translated)

    proper = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    improper = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)) @ proper
    for rotation in (proper, improper):
        rotated = r2r.multipolar_background(displacement @ rotation.T, graph)
        assert torch.allclose(rotated.trace_A2, base.trace_A2, atol=2.0e-16, rtol=2.0e-14)
        assert torch.allclose(
            rotated.cross_product_square_A4,
            base.cross_product_square_A4,
            atol=2.0e-19,
            rtol=2.0e-13,
        )
        expected_covariance = torch.einsum(
            "ab,nbc,dc->nad", rotation, base.covariance_A2, rotation
        )
        assert torch.allclose(
            rotated.covariance_A2, expected_covariance, atol=2.0e-16, rtol=2.0e-13
        )

    direction = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64)
    alpha = torch.linspace(-0.04, 0.05, 72, dtype=torch.float64)
    rank1 = r2r.multipolar_background(alpha[:, None] * direction, graph)
    assert torch.max(rank1.cross_product_square_A4).item() < 1.0e-36
    assert torch.max(rank1.multipolar_gate).item() < 1.0e-28

    structure = read(DATA / "train_thermal.xyz", index=0)
    canonical = r2r.background_from_structure(structure, reference, requires_grad=False)
    permutation = np.random.default_rng(44).permutation(len(structure))
    permuted = structure[permutation]
    reordered = r2r.background_from_structure(permuted, reference, requires_grad=False)
    _allclose_fields(canonical.fields, reordered.fields)
    wrapped = structure.copy()
    wrapped.positions[7] += np.asarray(wrapped.cell[0])
    image_changed = r2r.background_from_structure(wrapped, reference, requires_grad=False)
    _allclose_fields(canonical.fields, image_changed.fields)


def test_full_affine_scalar_has_o3_force_covariance_translation_and_locality() -> None:
    reference = read(DATA / "reference_6x6.xyz")
    graph = r2r.fixed_reference_neighborhood(reference)
    coefficients = torch.linspace(-0.2, 0.3, 65, dtype=torch.float64)

    def scalar_energy(displacement: torch.Tensor) -> torch.Tensor:
        fields = r2r.multipolar_background(displacement, graph)
        delta = displacement[graph.sender] - displacement[graph.receiver]
        local = displacement.new_zeros(72)
        local.index_add_(0, graph.receiver, graph.weight * torch.sum(delta * delta, dim=1))
        local = local / graph.normalization
        signed = local[:, None] * torch.linspace(
            -1.0, 1.0, 32, dtype=displacement.dtype
        )[None, :]
        basis = r2r.parameter_energy_basis(
            r2r.R2QNodeObservables(local, signed), fields
        )
        return torch.dot(basis, coefficients)

    generator = torch.Generator().manual_seed(919)
    displacement = (0.025 * torch.randn((72, 3), generator=generator, dtype=torch.float64)).requires_grad_(True)
    energy = scalar_energy(displacement)
    force = -torch.autograd.grad(energy, displacement)[0]
    transformations = (
        torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
    )
    for transformation in transformations:
        transformed = (displacement.detach() @ transformation.T).requires_grad_(True)
        transformed_energy = scalar_energy(transformed)
        transformed_force = -torch.autograd.grad(transformed_energy, transformed)[0]
        assert abs(float((transformed_energy - energy.detach()).detach())) < 2.0e-12
        assert torch.allclose(
            transformed_force, force @ transformation.T, atol=2.0e-11, rtol=2.0e-10
        )
    translated = (
        displacement.detach()
        + torch.tensor([0.8, -0.4, 0.2], dtype=torch.float64)
    ).requires_grad_(True)
    translated_energy = scalar_energy(translated)
    translated_force = -torch.autograd.grad(translated_energy, translated)[0]
    assert abs(float((translated_energy - energy.detach()).detach())) < 2.0e-12
    assert torch.allclose(translated_force, force, atol=2.0e-11, rtol=2.0e-10)

    neighbor_set = set(graph.sender[graph.receiver == 0].tolist()) | {0}
    outside = next(index for index in range(72) if index not in neighbor_set)
    localized = displacement.detach().clone()
    localized[outside] += torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    before = r2r.multipolar_background(displacement.detach(), graph)
    after = r2r.multipolar_background(localized, graph)
    assert torch.equal(before.trace_A2[0], after.trace_A2[0])
    assert torch.equal(
        before.cross_product_square_A4[0], after.cross_product_square_A4[0]
    )


def test_localized_6x6_8x8_parity_and_locality_bounds() -> None:
    reference6 = read(DATA / "reference_6x6.xyz")
    reference8 = read(DATA / "reference_8x8.xyz")
    graph6 = r2r.fixed_reference_neighborhood(reference6)
    graph8 = r2r.fixed_reference_neighborhood(reference8)
    u6 = torch.zeros((72, 3), dtype=torch.float64)
    u8 = torch.zeros((128, 3), dtype=torch.float64)
    u6[0] = u8[0] = torch.tensor([0.02, -0.01, 0.005], dtype=torch.float64)
    u6[1] = u8[1] = torch.tensor([-0.008, 0.017, 0.011], dtype=torch.float64)
    f6 = r2r.multipolar_background(u6, graph6)
    f8 = r2r.multipolar_background(u8, graph8)
    for name in (
        "trace_A2",
        "cross_product_square_A4",
        "multipolar_gate",
        "amplitude_gate",
    ):
        left = getattr(f6, name).detach().numpy()
        right = getattr(f8, name).detach().numpy()
        left = np.sort(left[np.abs(left) > 1.0e-25])
        right = np.sort(right[np.abs(right) > 1.0e-25])
        assert left.shape == right.shape
        # Archived 6x6/8x8 XYZ coordinates differ at the final printed digits;
        # this field-level diagnostic is subordinate to the later E/F gate.
        assert np.allclose(left, right, atol=2.0e-10, rtol=5.0e-8)
    assert float(graph6.reference_distance_A.max()) < 6.0
    assert float(graph8.reference_distance_A.max()) < 6.0
    assert r2r.BACKGROUND_EFFECTIVE_RADIUS_A == 6.4
    assert r2r.BACKGROUND_INTERACTION_DIAMETER_A == 12.8


def test_parameter_basis_has_exact_65_column_order() -> None:
    node_energy = torch.tensor([2.0, -1.0], dtype=torch.float64)
    signed = torch.arange(64, dtype=torch.float64).reshape(2, 32) / 17.0
    a = torch.tensor([0.25, 0.75], dtype=torch.float64)
    c = torch.tensor([0.4, 0.2], dtype=torch.float64)
    fields = r2r.BackgroundFields(
        covariance_A2=torch.zeros((2, 3, 3), dtype=torch.float64),
        trace_A2=torch.zeros(2, dtype=torch.float64),
        cross_product_square_A4=torch.zeros(2, dtype=torch.float64),
        multipolar_gate=a,
        amplitude_gate=c,
    )
    observables = r2r.R2QNodeObservables(node_energy, signed)
    basis = r2r.parameter_energy_basis(observables, fields)
    assert basis.shape == (65,)
    assert torch.equal(basis[:32], torch.sum(a[:, None] * signed, 0))
    assert basis[32] == torch.sum(a * c)
    assert torch.equal(
        basis[33:], torch.sum(a[:, None] * c[:, None] * signed, 0)
    )


def test_weighted_carrier_matches_explicit_node_Taylor2_energy_force_full_Hessian() -> None:
    dtype = torch.float64
    reference = torch.linspace(-0.3, 0.4, 6, dtype=dtype).reshape(2, 3)
    current = (
        reference
        + torch.tensor(
            [[0.08, -0.02, 0.05], [-0.03, 0.07, 0.01]], dtype=dtype
        )
    ).requires_grad_(True)
    image = torch.zeros_like(reference)

    def epsilon(positions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                torch.sin(positions[0]).sum()
                + 0.3 * torch.sum(positions[1] ** 3),
                torch.exp(0.2 * positions[1]).sum()
                + torch.sum(positions[0] * positions[1]),
            )
        )

    live_weight = torch.sigmoid(
        torch.stack((current.sum(), torch.sum(current * current)))
    )
    epsilon_current = epsilon(current)
    weighted = r2r.weighted_node_taylor2_carrier_energy(
        epsilon_current,
        epsilon,
        live_weight,
        current,
        reference,
        image,
    )
    displacement = current - reference
    x0 = reference.clone().requires_grad_(True)
    epsilon0 = epsilon(x0)
    explicit_terms = []
    for node in range(2):
        gradient = torch.autograd.grad(
            epsilon0[node], x0, create_graph=True, retain_graph=True
        )[0]
        hvp = torch.autograd.grad(
            gradient,
            x0,
            grad_outputs=displacement,
            create_graph=True,
            retain_graph=True,
        )[0]
        explicit_terms.append(
            live_weight[node]
            * (
                epsilon_current[node]
                - epsilon0[node]
                - torch.sum(gradient * displacement)
                - 0.5 * torch.sum(displacement * hvp)
            )
        )
    explicit = torch.stack(explicit_terms).sum()

    def force_hessian(energy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        force = -torch.autograd.grad(
            energy, current, create_graph=True, retain_graph=True
        )[0]
        rows = []
        for component in force.reshape(-1):
            rows.append(
                -torch.autograd.grad(
                    component, current, retain_graph=True
                )[0].reshape(-1)
            )
        return force, torch.stack(rows)

    weighted_force, weighted_hessian = force_hessian(weighted)
    explicit_force, explicit_hessian = force_hessian(explicit)
    assert abs(float((weighted - explicit).detach())) < 2.0e-16
    assert torch.max(torch.abs(weighted_force - explicit_force)).item() < 2.0e-15
    assert torch.max(torch.abs(weighted_hessian - explicit_hessian)).item() < 2.0e-15


def test_taylor_null_affine_offset_and_all_65_columns_have_reference_E_F_H_zero() -> None:
    dtype = torch.float64
    reference = torch.zeros((2, 3), dtype=dtype)
    current = reference.clone().requires_grad_(True)
    image = torch.zeros_like(reference)
    coefficients = torch.linspace(0.3, 1.7, 66, dtype=dtype)

    def basis(positions: torch.Tensor) -> torch.Tensor:
        z = positions.reshape(-1)
        scalar = (
            0.7
            + 0.2 * z.sum()
            + 0.3 * torch.sum(z * z)
            + 0.4 * torch.sum(z**3)
            + 0.1 * torch.sum(z**4)
        )
        return coefficients * scalar

    result = r2r.taylor_null_linear_force_design(
        basis, current, reference, image, create_graph=True
    )
    assert result.fixed_offset_energy_eV.shape == (1,)
    assert result.fixed_offset_force_eV_A.shape == (6, 1)
    assert result.energy_design_eV.shape == (65,)
    assert result.force_design_eV_A.shape == (6, 65)
    assert torch.max(torch.abs(result.fixed_offset_energy_eV)).item() < 1.0e-15
    assert torch.max(torch.abs(result.energy_design_eV)).item() < 1.0e-15
    assert torch.max(torch.abs(result.fixed_offset_force_eV_A)).item() < 1.0e-14
    assert torch.max(torch.abs(result.force_design_eV_A)).item() < 1.0e-14
    for column in (0, 1, 32, 65):
        energy = (
            result.fixed_offset_energy_eV[0]
            if column == 0
            else result.energy_design_eV[column - 1]
        )
        gradient = torch.autograd.grad(energy, current, create_graph=True, retain_graph=True)[0]
        rows = []
        for flat in range(current.numel()):
            rows.append(
                torch.autograd.grad(
                    gradient.reshape(-1)[flat], current, retain_graph=True
                )[0].reshape(-1)
            )
        hessian = torch.stack(rows)
        assert torch.max(torch.abs(hessian)).item() < 1.0e-13


def test_r2q_node_energy_matches_formal_energy_and_position_gradient() -> None:
    checkpoint = r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")
    model = checkpoint["model"].to(dtype=torch.float64).eval()
    assert r2o.state_dict_sha256(model) == r2r.R2Q_ENDPOINT_STATE_SHA256
    reference = read(DATA / "reference_6x6.xyz")
    data = r2o.fixed_reference_graph(reference, device="cpu", dtype=torch.float64)
    positions = torch.as_tensor(
        np.asarray(reference.positions), dtype=torch.float64
    ).clone().requires_grad_(True)
    observables = r2r.r2q_node_observables(model, data, positions)
    extracted = observables.scale_shift_node_energy_eV.sum().reshape(1)
    formal = r2o.mace_interaction_energy(model, data, positions)
    assert torch.max(torch.abs(extracted - formal)).item() < 1.0e-12
    extracted_gradient = torch.autograd.grad(
        extracted.sum(), positions, retain_graph=True
    )[0]
    formal_gradient = torch.autograd.grad(formal.sum(), positions)[0]
    assert torch.max(torch.abs(extracted_gradient - formal_gradient)).item() < 1.0e-11
    assert observables.signed_l0.shape == (72, 32)

    thermal = read(DATA / "train_thermal.xyz", index=0)
    thermal_reference = r2o.adapt_reference_cell(reference, thermal)
    assignment = r2o.solve_assignment(thermal, thermal_reference)
    ordered = r2o.reordered_structure(thermal, assignment)
    aligned_positions = (
        np.asarray(ordered.positions)
        - assignment.image_integer_reference_order @ np.asarray(ordered.cell)
    )
    thermal_data = r2o.fixed_reference_graph(
        thermal_reference, device="cpu", dtype=torch.float64
    )
    thermal_positions = torch.as_tensor(
        aligned_positions, dtype=torch.float64
    ).clone().requires_grad_(True)
    thermal_observables = r2r.r2q_node_observables(
        model, thermal_data, thermal_positions
    )
    thermal_extracted = thermal_observables.scale_shift_node_energy_eV.sum().reshape(1)
    thermal_formal = r2o.mace_interaction_energy(model, thermal_data, thermal_positions)
    assert torch.max(torch.abs(thermal_extracted - thermal_formal)).item() < 1.0e-10
    thermal_extracted_gradient = torch.autograd.grad(
        thermal_extracted.sum(), thermal_positions, retain_graph=True
    )[0]
    thermal_formal_gradient = torch.autograd.grad(
        thermal_formal.sum(), thermal_positions
    )[0]
    assert (
        torch.max(torch.abs(thermal_extracted_gradient - thermal_formal_gradient)).item()
        < 1.0e-9
    )
    parameter = next(model.parameters())
    saved = parameter.detach().clone()
    with torch.no_grad():
        parameter.reshape(-1)[0].add_(1.0e-8)
    with pytest.raises(ValueError, match="exact frozen R2Q endpoint state"):
        r2r.r2q_node_observables(model, data, positions)
    with torch.no_grad():
        parameter.copy_(saved)
    assert r2o.state_dict_sha256(model) == r2r.R2Q_ENDPOINT_STATE_SHA256


def test_signed_scalar_selector_rejects_any_0o_channel() -> None:
    class ScalarIrrep:
        l = 0
        dim = 1

        def __init__(self, parity: int) -> None:
            self.p = parity

    valid = torch.arange(32, dtype=torch.float64).reshape(2, 16)
    assert torch.equal(r2r.even_l0_channels(valid, [(16, ScalarIrrep(1))]), valid)
    with pytest.raises(ValueError, match="odd-parity 0o"):
        r2r.even_l0_channels(valid, [(16, ScalarIrrep(-1))])


def test_rank_condition_uses_rms_only_no_centering_and_zero_relative_threshold() -> None:
    rng = np.random.default_rng(935)
    design = rng.normal(size=(400, 65))
    design[:, 7] *= 1.0e-7
    receipt = r2r.rank_condition_precheck(design)
    assert receipt["scaling"] == "RMS_without_mean_centering"
    assert receipt["full_column_rank"] is True
    assert receipt["scaled_rank"] == 65
    assert receipt["can_authorize_fit_or_training"] is False
    deficient = design.copy()
    deficient[:, 12] = 0.0
    deficient_receipt = r2r.rank_condition_precheck(deficient)
    assert deficient_receipt["zero_or_near_zero_columns"] == [12]
    assert deficient_receipt["full_column_rank"] is False

    collector = r2r.R2R0DesignCollector()
    with pytest.raises(ValueError, match="not train-only"):
        collector.add("harmonic_zero32", "h0", np.zeros((384, 65)))
    for index in range(92):
        collector.add("thermal92", f"thermal:{index}", rng.normal(size=(3, 65)))
    collected, collected_receipt = collector.finalize()
    assert collected.shape == (276, 65)
    assert collected_receipt["split_counts"] == {"thermal92": 92}
    assert collected_receipt["R2R0_precheck_pass"] is True


def test_harmonic_gate_is_separate_actual_receipt_not_hand_zeroed_design() -> None:
    receipt = r2r.harmonic_rank1_gate_receipt(
        structure_count=32,
        max_cross_product_square_A4=2.2896187649234325e-34,
        max_multipolar_gate=1.0e-25,
        max_taylor_remainder_energy_eV=1.0e-12,
        max_taylor_remainder_force_eV_A=1.0e-11,
    )
    assert receipt["pass"] is True
    assert receipt["enters_design_scaler_rank_OOF_or_fit"] is False
    assert receipt["can_authorize_fit_or_training"] is False
    failed = r2r.harmonic_rank1_gate_receipt(
        structure_count=32,
        max_cross_product_square_A4=2.0e-20,
        max_multipolar_gate=1.0e-13,
        max_taylor_remainder_energy_eV=0.0,
        max_taylor_remainder_force_eV_A=0.0,
    )
    assert failed["pass"] is False


def test_real_endpoint_production_integration_mapping_mic_graph_and_full_hessian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")
    model = checkpoint["model"].to(dtype=torch.float64).eval()
    reference = read(DATA / "reference_6x6.xyz")
    thermal = read(DATA / "train_thermal.xyz", index=0)

    def lightweight_components(context):
        aligned = (
            context.current_positions_reference_order - context.image_shift.detach()
        )
        displacement_nodes = aligned - context.reference_positions
        scales = torch.linspace(
            0.5, 1.5, 66, dtype=aligned.dtype, device=aligned.device
        )
        norm2 = torch.sum(displacement_nodes * displacement_nodes, dim=1)
        quartic = torch.sum(norm2 * norm2)
        energy = scales * quartic
        return energy[0], energy[1:]

    monkeypatch.setattr(r2r, "_production_energy_components", lightweight_components)
    query = r2r.production_linear_design_query(
        model, thermal, reference, device="cpu"
    )
    assert query.fixed_offset_force_source_order_eV_A.shape == (216, 1)
    assert query.parameter_force_design_source_order_eV_A.shape == (216, 65)
    assert query.receipt["endpoint_state_sha256"] == r2r.R2Q_ENDPOINT_STATE_SHA256
    assert query.receipt["absolute_structure_semantic_sha256"] == (
        "e64c2c5cc7681f710f01a22e9a374cf900c435387de16edfbb2037b025d941b9"
    )
    assert query.receipt["baseline_formal_R2O_graph_hash_match"] is True
    assert query.receipt["formal_R2O_graph_mode"] == "baseline"
    assert query.receipt["graph_construction_default_dtype"] == "torch.float32"
    assert query.receipt["shortest_in_plane_translation_A"] == pytest.approx(14.76)
    assert query.receipt["no_wrap_margin_A"] == pytest.approx(1.96)

    adapted_reference = r2o.adapt_reference_cell(reference, thermal)
    formal_data = r2o.fixed_reference_graph(
        adapted_reference, device="cpu", dtype=torch.float64
    )
    formal_graph_cell = formal_data["cell"].reshape(-1, 3, 3)[0].numpy()

    # Internal exact replay uses the same quantized float32-origin cell as the
    # frozen formal graph and therefore isolates mapping/MIC implementation.
    formal_wrapped = thermal.copy()
    formal_wrapped.positions[5] += formal_graph_cell[0]
    formal_wrapped_query = r2r.production_linear_design_query(
        model, formal_wrapped, reference, device="cpu"
    )
    assert (
        torch.max(
            torch.abs(
                formal_wrapped_query.parameter_force_design_source_order_eV_A
                - query.parameter_force_design_source_order_eV_A
            )
        ).item()
        < 1.0e-12
    )

    # A native-cell wrap intentionally exposes the archive-cell versus frozen
    # float32-origin graph-cell quantization.  Raw columns are diagnostic only;
    # authorization is based on the actual canonical fixed+X@p observable.
    native_wrapped = thermal.copy()
    native_wrapped.positions[5] += np.asarray(native_wrapped.cell[0])
    native_wrapped_query = r2r.production_linear_design_query(
        model, native_wrapped, reference, device="cpu"
    )
    wrap_receipt = r2r.native_cell_wrap_order_mic_receipt(
        query, native_wrapped_query
    )
    assert math.isfinite(wrap_receipt["raw_65_force_design_max_abs_eV_A"])
    assert math.isfinite(wrap_receipt["raw_65_force_design_relative"])
    assert wrap_receipt["raw_65_column_difference_role"] == (
        "diagnostic_only_not_authorization"
    )
    assert wrap_receipt["combined_energy_abs_eV"] <= 1.0e-6
    assert wrap_receipt["combined_force_max_abs_eV_A"] <= 1.0e-5
    assert wrap_receipt["pass"] is True

    permutation = np.random.default_rng(220).permutation(72)
    permuted = thermal[permutation]
    permuted_query = r2r.production_linear_design_query(
        model, permuted, reference, device="cpu"
    )
    expected_permuted = query.parameter_force_design_source_order_eV_A.reshape(
        72, 3, 65
    )[permutation].reshape(216, 65)
    assert torch.allclose(
        permuted_query.parameter_force_design_source_order_eV_A,
        expected_permuted,
        atol=1.0e-14,
        rtol=1.0e-12,
    )

    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_reference = reference.copy()
    rotated_reference.positions = np.asarray(reference.positions) @ rotation.T
    rotated_reference.set_cell(np.asarray(reference.cell) @ rotation.T, scale_atoms=False)
    rotated_thermal = thermal.copy()
    rotated_thermal.positions = np.asarray(thermal.positions) @ rotation.T
    rotated_thermal.set_cell(np.asarray(thermal.cell) @ rotation.T, scale_atoms=False)
    rotated_query = r2r.production_linear_design_query(
        model,
        rotated_thermal,
        rotated_reference,
        device="cpu",
        graph_mode="rigid_transform_probe",
        baseline_reference_template=reference,
        baseline_structure_template=thermal,
        rigid_transform=rotation,
    )
    assert rotated_query.receipt["formal_R2O_graph_mode"] == "rigid_transform_probe"
    assert rotated_query.receipt["rigid_transform_receipt"]["determinant"] == pytest.approx(1.0)
    base_probe = query.canonical_combined_probe()
    rotated_probe = rotated_query.canonical_combined_probe()
    assert abs(
        float((rotated_probe.energy_eV - base_probe.energy_eV).detach())
    ) < 1.0e-10
    assert torch.allclose(
        rotated_probe.force_source_order_eV_A,
        base_probe.force_source_order_eV_A
        @ torch.as_tensor(rotation, dtype=torch.float64).T,
        atol=1.0e-10,
        rtol=1.0e-9,
    )

    tampered = dict(formal_data)
    tampered["edge_index"] = formal_data["edge_index"].clone()
    tampered["edge_index"][0, 0] = (tampered["edge_index"][0, 0] + 1) % 72
    with pytest.raises(ValueError, match="formal R2O graph tensor changed"):
        r2r.production_linear_design_query(
            model,
            thermal,
            reference,
            device="cpu",
            formal_graph_data=tampered,
        )

    original_builder = r2r.fixed_reference_graph

    def wrong_builder(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        result = dict(result)
        result["shifts"] = result["shifts"].clone()
        result["shifts"][0, 0] += 1.0e-7
        return result

    monkeypatch.setattr(r2r, "fixed_reference_graph", wrong_builder)
    with pytest.raises(ValueError, match="baseline formal R2O graph hash"):
        r2r.production_linear_design_query(
            model, thermal, reference, device="cpu"
        )
    monkeypatch.setattr(r2r, "fixed_reference_graph", original_builder)

    wrong_absolute = thermal.copy()
    wrong_absolute.positions[0] += 0.8 * np.asarray(wrong_absolute.cell[0])
    with pytest.raises(ValueError, match="assignment"):
        r2r.production_linear_design_query(
            model, wrong_absolute, reference, device="cpu"
        )

    original_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="float32-origin"):
            r2r.production_linear_design_query(
                model, thermal, reference, device="cpu"
            )
    finally:
        torch.set_default_dtype(original_default)

    with pytest.raises(ValueError, match="66-column create_graph mechanics is forbidden"):
        r2r.production_linear_design_query(
            model, thermal, reference, device="cpu", create_graph=True
        )

    mechanics = r2r.production_combined_energy_force_hessian(
        model, thermal, reference, device="cpu"
    )
    assert mechanics.force_source_order_eV_A.shape == (72, 3)
    assert mechanics.Hessian_source_order_eV_A2.shape == (216, 216)
    assert mechanics.coefficients_sha256 == (
        "656ca438f398dc4373456439322d2c94883789e2f97ac1bbcad66b5bd5713b04"
    )
    assert mechanics.Hessian_antisymmetry_max_abs_eV_A2 < 1.0e-12
    with pytest.raises(ValueError, match="canonical probe"):
        r2r.production_combined_energy_force_hessian(
            model,
            thermal,
            reference,
            device="cpu",
            coefficients=np.zeros(65),
        )
    with pytest.raises(ValueError, match="canonical probe"):
        r2r.production_combined_energy_force(
            model,
            thermal,
            reference,
            device="cpu",
            coefficients=np.zeros(65),
        )


def test_real_endpoint_v4_covariant_identity_proper_improper_and_native_diagnostic() -> None:
    checkpoint = r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")
    model = checkpoint["model"].to(dtype=torch.float64).eval()
    reference = read(DATA / "reference_6x6.xyz")
    thermal = read(DATA / "train_thermal.xyz", index=0)
    baseline = r2r.production_linear_design_query(
        model, thermal, reference, device="cpu"
    )
    coefficients = torch.as_tensor(
        r2r.MECHANICS_PROBE_COEFFICIENTS, dtype=torch.float64
    )
    base_fixed_energy = baseline.fixed_offset_energy_eV[0]
    base_fixed_force = baseline.fixed_offset_force_source_order_eV_A.reshape(72, 3)
    base_parameter_energy = torch.dot(
        baseline.parameter_energy_design_eV, coefficients
    )
    base_parameter_force = (
        baseline.parameter_force_design_source_order_eV_A @ coefficients
    ).reshape(72, 3)

    generator = np.random.default_rng(83)
    proper, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(proper) < 0:
        proper[:, 0] *= -1
    improper = proper.copy()
    improper[:, 0] *= -1
    transformations = (("identity", np.eye(3)), ("proper", proper), ("improper", improper))
    background_hash = None
    transformed_by_name = {}
    for name, transformation in transformations:
        transformed_reference = reference.copy()
        transformed_reference.positions = (
            np.asarray(reference.positions) @ transformation.T
        )
        transformed_reference.set_cell(
            np.asarray(reference.cell) @ transformation.T, scale_atoms=False
        )
        transformed_thermal = thermal.copy()
        transformed_thermal.positions = (
            np.asarray(thermal.positions) @ transformation.T
        )
        transformed_thermal.set_cell(
            np.asarray(thermal.cell) @ transformation.T, scale_atoms=False
        )
        transformed = r2r.production_linear_design_query(
            model,
            transformed_thermal,
            transformed_reference,
            device="cpu",
            graph_mode="rigid_transform_probe",
            baseline_reference_template=reference,
            baseline_structure_template=thermal,
            rigid_transform=transformation,
        )
        transformed_by_name[name] = (transformed_thermal, transformed_reference)
        q = torch.as_tensor(transformation, dtype=torch.float64)
        fixed_energy = transformed.fixed_offset_energy_eV[0]
        fixed_force = transformed.fixed_offset_force_source_order_eV_A.reshape(72, 3)
        parameter_energy = torch.dot(
            transformed.parameter_energy_design_eV, coefficients
        )
        parameter_force = (
            transformed.parameter_force_design_source_order_eV_A @ coefficients
        ).reshape(72, 3)
        assert abs(float((fixed_energy - base_fixed_energy).detach())) <= 1.0e-10
        assert torch.max(torch.abs(fixed_force - base_fixed_force @ q.T)).item() <= 1.0e-9
        assert abs(float((parameter_energy - base_parameter_energy).detach())) <= 1.0e-10
        assert (
            torch.max(torch.abs(parameter_force - base_parameter_force @ q.T)).item()
            <= 1.0e-9
        )
        combined_energy = fixed_energy + parameter_energy
        base_combined_energy = base_fixed_energy + base_parameter_energy
        combined_force = fixed_force + parameter_force
        base_combined_force = base_fixed_force + base_parameter_force
        assert abs(float((combined_energy - base_combined_energy).detach())) <= 1.0e-10
        assert (
            torch.max(torch.abs(combined_force - base_combined_force @ q.T)).item()
            <= 1.0e-9
        )
        receipt = transformed.receipt["rigid_transform_receipt"]
        assert receipt["baseline_graph_frozen_hash_match"] is True
        assert all(receipt["assignment_arrays_byte_equal"].values())
        assert receipt["baseline_structure_full_semantic_sha256"] == (
            r2r.full_structure_semantic_sha256(thermal)
        )
        assert receipt["baseline_reference_full_semantic_sha256"] == (
            r2r.full_structure_semantic_sha256(reference)
        )
        assert len(receipt["baseline_assignment_and_MIC_semantic_sha256"]) == 64
        assert receipt["native_background_physical_edge_multiset_equivalent"] is True
        assert receipt["public_reference_input_covariance_array_exact"] is True
        assert receipt["public_structure_input_covariance_array_exact"] is True
        assert receipt["internal_covariance_atol_A"] == 1.0e-12
        assert all(
            receipt[
                "reordered_and_reference_covariance_within_named_tolerance"
            ].values()
        )
        assert receipt["native_rebuild_used_for_physics"] is False
        assert receipt["native_rebuild_diagnostic"]["physics_gate_authorized"] is False
        assert receipt["native_rebuild_diagnostic"][
            "covariant_edge_vector_max_abs_difference_A"
        ] <= 1.0e-12
        assert receipt["native_rebuild_diagnostic"][
            "covariant_edge_length_max_abs_difference_A"
        ] <= 1.0e-12
        assert receipt["selected_graph_sha256"] == receipt["derived_graph_sha256"]
        assert receipt["native_rebuild_diagnostic"][
            "native_physical_edge_multiset_equivalent"
        ] is True
        assert receipt["native_rebuild_diagnostic"][
            "numeric_differences_are_diagnostic_only"
        ] is True
        if background_hash is None:
            background_hash = receipt["background_graph_sha256"]
        assert receipt["background_graph_sha256"] == background_hash
        if name == "identity":
            assert receipt["derived_graph_sha256"] == (
                r2r.EXPECTED_BASELINE_FORMAL_GRAPH_SHA256["reference_6x6"]
            )

    transformed_thermal, transformed_reference = transformed_by_name["proper"]
    diagnostic = r2r.production_rigid_transform_native_rebuild_diagnostic(
        model,
        transformed_thermal,
        transformed_reference,
        baseline_reference_template=reference,
        baseline_structure_template=thermal,
        rigid_transform=proper,
        device="cpu",
    )
    assert diagnostic["authorization_role"] == "diagnostic_only_not_physics_gate"
    assert diagnostic["physics_gate_authorized"] is False
    assert diagnostic["energy_abs_difference_eV"] == pytest.approx(
        7.0869794e-7, rel=5.0e-6
    )
    assert diagnostic["force_max_abs_difference_eV_A"] == pytest.approx(
        2.67009836e-5, rel=5.0e-7
    )


def test_v4_rigid_probe_tamper_fail_closed_and_6x8_baseline_hashes_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")
    model = checkpoint["model"].to(dtype=torch.float64).eval()
    for size in (6, 8):
        reference = read(DATA / f"reference_{size}x{size}.xyz")
        canonical = r2o.adapt_reference_cell(reference, reference)
        graph = r2o.fixed_reference_graph(
            canonical, device="cpu", dtype=torch.float64
        )
        assert r2r.tensor_mapping_semantic_sha256(
            graph, r2r.FORMAL_GRAPH_TENSOR_KEYS
        ) == r2r.EXPECTED_BASELINE_FORMAL_GRAPH_SHA256[f"reference_{size}x{size}"]

    reference = read(DATA / "reference_6x6.xyz")
    thermal = read(DATA / "train_thermal.xyz", index=0)
    generator = np.random.default_rng(83)
    proper, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(proper) < 0:
        proper[:, 0] *= -1
    transformed_reference = reference.copy()
    transformed_reference.positions = np.asarray(reference.positions) @ proper.T
    transformed_reference.set_cell(
        np.asarray(reference.cell) @ proper.T, scale_atoms=False
    )
    transformed_thermal = thermal.copy()
    transformed_thermal.positions = np.asarray(thermal.positions) @ proper.T
    transformed_thermal.set_cell(np.asarray(thermal.cell) @ proper.T, scale_atoms=False)

    nonorthogonal = proper.copy()
    nonorthogonal[0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="proper/improper orthogonal"):
        r2r.production_combined_energy_force(
            model,
            transformed_thermal,
            transformed_reference,
            device="cpu",
            graph_mode="rigid_transform_probe",
            baseline_reference_template=reference,
            baseline_structure_template=thermal,
            rigid_transform=nonorthogonal,
        )

    live_tamper = transformed_thermal.copy()
    live_tamper.positions[0, 0] += 1.0e-12
    with pytest.raises(ValueError, match="live structure/reference covariance"):
        r2r.production_combined_energy_force(
            model,
            live_tamper,
            transformed_reference,
            device="cpu",
            graph_mode="rigid_transform_probe",
            baseline_reference_template=reference,
            baseline_structure_template=thermal,
            rigid_transform=proper,
        )

    reference_tamper = transformed_reference.copy()
    reference_tamper.positions[1, 0] += 1.0e-12
    with pytest.raises(ValueError, match="reference is not covariant with baseline"):
        r2r.production_combined_energy_force(
            model,
            transformed_thermal,
            reference_tamper,
            device="cpu",
            graph_mode="rigid_transform_probe",
            baseline_reference_template=reference,
            baseline_structure_template=thermal,
            rigid_transform=proper,
        )

    canonical = r2o.adapt_reference_cell(reference, reference)
    baseline_graph = r2o.fixed_reference_graph(
        canonical, device="cpu", dtype=torch.float64
    )
    derived_graph = r2r._covariant_formal_graph(baseline_graph, proper)
    graph_tamper = dict(derived_graph)
    graph_tamper["shifts"] = derived_graph["shifts"].clone()
    graph_tamper["shifts"][0, 0] += 1.0e-7
    with pytest.raises(ValueError, match="formal R2O graph tensor changed"):
        r2r.production_combined_energy_force(
            model,
            transformed_thermal,
            transformed_reference,
            device="cpu",
            formal_graph_data=graph_tamper,
            graph_mode="rigid_transform_probe",
            baseline_reference_template=reference,
            baseline_structure_template=thermal,
            rigid_transform=proper,
        )

    original_assignment = r2r.solve_assignment
    call_count = 0

    def tampered_baseline_assignment(*args, **kwargs):
        nonlocal call_count
        result = original_assignment(*args, **kwargs)
        call_count += 1
        if call_count == 2:
            image = result.image_integer_reference_order.copy()
            image[0, 0] += 1
            return r2o.Assignment(
                reference_to_source=result.reference_to_source,
                source_to_reference=result.source_to_reference,
                image_integer_reference_order=image,
                maximum_distance_A=result.maximum_distance_A,
                minimum_uniqueness_gap_A=result.minimum_uniqueness_gap_A,
            )
        return result

    monkeypatch.setattr(r2r, "solve_assignment", tampered_baseline_assignment)
    with pytest.raises(ValueError, match="assignment or MIC image gauge"):
        r2r.production_combined_energy_force(
            model,
            transformed_thermal,
            transformed_reference,
            device="cpu",
            graph_mode="rigid_transform_probe",
            baseline_reference_template=reference,
            baseline_structure_template=thermal,
            rigid_transform=proper,
        )


def test_real_endpoint_corrected_carrier_matches_frozen_R2Q_EF_and_rejects_legacy() -> None:
    checkpoint = r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")
    model = checkpoint["model"].to(dtype=torch.float64).eval()
    reference = read(DATA / "reference_6x6.xyz")
    probes = (reference.copy(), read(DATA / "train_thermal.xyz", index=0))
    gates = r2r.CANONICAL_CONTRACT["fixed_gates"][
        "corrected_carrier_vs_frozen_R2Q_parity"
    ]
    for probe_index, structure in enumerate(probes):
        design = r2r.production_linear_design_query(
            model, structure, reference, device="cpu"
        )
        column_combined = design.canonical_combined_probe()
        fast = r2r.production_combined_energy_force(
            model, structure, reference, device="cpu"
        )
        assert abs(float((fast.energy_eV - column_combined.energy_eV).detach())) < 1.0e-12
        assert float(
            torch.max(
                torch.abs(
                    fast.force_source_order_eV_A
                    - column_combined.force_source_order_eV_A
                )
            ).detach()
        ) < 1.0e-11
        carrier = r2r.production_fixed_carrier_energy_force(
            model, structure, reference, device="cpu"
        )
        frozen, assignment = r2o.evaluate_structure(
            model, structure, reference, device="cpu", create_graph=False
        )
        frozen_force = r2o.source_order_forces(
            frozen.forces_reference_order, assignment
        )
        assert abs(float((carrier.energy_eV - frozen.energy[0]).detach())) <= gates[
            "energy_abs_eV"
        ]
        assert float(
            torch.max(
                torch.abs(carrier.force_source_order_eV_A - frozen_force)
            ).detach()
        ) <= gates["force_max_abs_eV_A"]
        assert fast.query_receipt["legacy_66_column_create_graph_used"] is False
        assert fast.query_receipt["affine_component_names_sha256"] == (
            r2r.AFFINE_COMPONENT_NAMES_SHA256
        )
        assert fast.query_receipt["mechanics_affine_[1,p]_sha256"] == (
            r2r.MECHANICS_AFFINE_COEFFICIENTS_SHA256
        )
        if probe_index == 1:
            context = r2r._verified_production_context(
                model, structure, reference, device="cpu"
            )
            aligned = (
                context.current_positions_reference_order
                - context.image_shift.detach()
            )
            observables = context.node_observables_fn(aligned)
            fields = context.background_fields_fn(aligned)
            legacy = r2r._legacy_raw_a_epsilon_energy(observables, fields)
            legacy_force_reference = -torch.autograd.grad(
                legacy, context.current_positions_reference_order
            )[0]
            legacy_force = r2r._source_order_force(
                legacy_force_reference, context.assignment
            )
            assert float(torch.max(torch.abs(legacy_force - frozen_force))) > 0.1


def test_real_endpoint_corrected_carrier_rank1_EF_null() -> None:
    checkpoint = r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")
    model = checkpoint["model"].to(dtype=torch.float64).eval()
    reference = read(DATA / "reference_8x8.xyz")
    harmonic = read(DATA / "train_harmonic_lambda1_small_zero.xyz", index=0)
    carrier = r2r.production_fixed_carrier_energy_force(
        model, harmonic, reference, device="cpu"
    )
    gates = r2r.CANONICAL_CONTRACT["fixed_gates"]["harmonic_zero32"]
    assert abs(float(carrier.energy_eV.detach())) <= gates[
        "max_Taylor_remainder_energy_eV"
    ]
    assert torch.max(torch.abs(carrier.force_source_order_eV_A)).item() <= gates[
        "max_Taylor_remainder_force_eV_A"
    ]


@pytest.mark.skipif(
    os.environ.get("R2R_RUN_SLOW_ZERO_JET") != "1",
    reason="explicit real 6x6/8x8 all-node local-direct zero-jet audit",
)
def test_slow_real_6x6_8x8_all_node_local_direct_zero_jet() -> None:
    expected = {
        "reference_6x6.xyz": {
            "nodes": 72,
            "orbits": 71,
            "orbit_sha256": (
                "19ca084d9f54c17864d059bffc2da71e290e87bd8bab9dbea4ac6395441d4440"
            ),
            "probe_nodes_sha256": (
                "35013e78c8691a14ba643ae6e528e387a76a351ce2bf3ccec566101e9ab676a5"
            ),
            "source_order_sha256": (
                "4f0fcf5acf9669588bf7438fdb9eaf0b00dcff5c62c528b8d7637ffac3b744db"
            ),
        },
        "reference_8x8.xyz": {
            "nodes": 128,
            "orbits": 123,
            "orbit_sha256": (
                "b6cec9b3886a71c076c715daeaf2dbbc48a42986a7c0b76c0becc4a144a0fdb5"
            ),
            "probe_nodes_sha256": (
                "5c3946ef8052e49b3193e9baa95fb44168a50ef1eb5bb96939848915f4da66b8"
            ),
            "source_order_sha256": (
                "6e44497aa2b55fc433c8155ce08f9f9b2695d9adcf9b298046dfd291958ecc56"
            ),
        },
    }
    for source, frozen in expected.items():
        graph = r2r.fixed_reference_neighborhood(read(DATA / source))
        receipt = r2r.background_rank0_zero_jet_audit(graph)
        assert receipt["pass"] is True
        assert receipt["node_count"] == frozen["nodes"]
        assert receipt["q_signature_orbit_count"] == frozen["orbits"]
        assert receipt["q_signature_orbits_sha256"] == frozen["orbit_sha256"]
        assert receipt["nonzero_probe_base_sha256"] == (
            "5e9fdb9d11a6cd6573121a00bbc379056bf6bbe3c2c729935de9b8be5ae91d4d"
        )
        assert receipt["nonzero_probe_nodes_sha256"] == frozen[
            "probe_nodes_sha256"
        ]
        assert receipt["production_local_source_order_sha256"] == frozen[
            "source_order_sha256"
        ]
        assert receipt["all_nodes_covered_exactly_once"] is True
        assert receipt["all_local_senders_unique"] is True
        assert receipt["global_all_node_value_and_Jacobian_exact_zero"] is True
        assert receipt["local_sample_count"] == 40
        assert receipt["local_Hessian_shape_per_node"] == [40, 3, 40, 3]
        assert receipt["local_all_node_value_max_abs"] == 0.0
        assert receipt["local_all_node_Jacobian_max_abs_A-1"] == 0.0
        assert receipt["local_all_node_Hessian_max_abs_A-2"] == 0.0
        assert receipt["local_all_node_finite"] is True
        for probe in receipt["nonzero_local_production_probe"]:
            assert math.isfinite(probe["base_b_A4"])
            assert probe["base_b_A4"] > 0.0
            assert math.isfinite(probe["scale"])
            assert abs(probe["b_over_beta"] - 1.0) <= 1.0e-12
            assert probe["local_a"] == pytest.approx(
                1.0 - math.exp(-1.0), abs=1.0e-14
            )
            assert probe["value_abs_difference"] <= 1.0e-14
            assert probe["mapped_gradient_max_abs_difference_A-1"] <= 1.0e-12
            assert probe["outside_local_global_gradient_max_abs_A-1"] == 0.0


@pytest.mark.skipif(
    os.environ.get("R2R_RUN_SLOW_FULL_H_PARITY") != "1",
    reason="explicit ~80 s corrected-carrier full-H audit",
)
def test_slow_real_endpoint_corrected_carrier_matches_frozen_R2Q_full_H() -> None:
    checkpoint = r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")
    model = checkpoint["model"].to(dtype=torch.float64).eval()
    reference = read(DATA / "reference_6x6.xyz")
    probes = (reference.copy(), read(DATA / "train_thermal.xyz", index=0))
    gates = r2r.CANONICAL_CONTRACT["fixed_gates"][
        "corrected_carrier_vs_frozen_R2Q_parity"
    ]
    for structure in probes:
        carrier = r2r.production_fixed_carrier_energy_force_hessian(
            model, structure, reference, device="cpu"
        )
        frozen, assignment = r2o.evaluate_structure(
            model, structure, reference, device="cpu", create_graph=True
        )
        rows = []
        for component in frozen.forces_reference_order.reshape(-1):
            rows.append(
                -torch.autograd.grad(
                    component,
                    frozen.current_positions_reference_order,
                    retain_graph=True,
                )[0].reshape(-1)
            )
        reference_hessian = torch.stack(rows)
        reference_to_source = torch.as_tensor(
            assignment.reference_to_source, dtype=torch.long
        )
        source_components = (
            3 * reference_to_source[:, None] + torch.arange(3)[None, :]
        ).reshape(-1)
        frozen_hessian = torch.empty_like(reference_hessian)
        frozen_hessian[source_components[:, None], source_components[None, :]] = (
            reference_hessian
        )
        frozen_force = r2o.source_order_forces(
            frozen.forces_reference_order, assignment
        )
        assert abs(float((carrier.energy_eV - frozen.energy[0]).detach())) <= gates[
            "energy_abs_eV"
        ]
        assert torch.max(
            torch.abs(carrier.force_source_order_eV_A - frozen_force)
        ).item() <= gates["force_max_abs_eV_A"]
        assert torch.max(
            torch.abs(carrier.Hessian_source_order_eV_A2 - frozen_hessian)
        ).item() <= gates["Hessian_max_abs_eV_A2"]


def test_future_continuous_folds_are_exact_and_do_not_claim_encoder_independence() -> None:
    folds = r2r.continuous_four_fold_assignments()
    assert [len(fold) for fold in folds["E50_seed0"]] == [5] * 4
    assert [len(fold) for fold in folds["T300"]] == [9] * 4
    assert [len(fold) for fold in folds["T600"]] == [9] * 4
    assert "harmonic_zero" not in folds
    for groups, total in ((folds["E50_seed0"], 20), (folds["T300"], 36), (folds["T600"], 36)):
        assert [value for group in groups for value in group] == list(range(total))
    boundary = r2r.CANONICAL_CONTRACT["future_readout_boundary"]
    assert "not encoder-level independence" in boundary["OOF_interpretation"]
