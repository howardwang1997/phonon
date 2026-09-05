from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parents[1] / "scripts" / "smearing_kink"
sys.path.insert(0, str(HERE))

from graphene_r2m_routed_tail import (  # noqa: E402
    EXPECTED_INVARIANT_WIDTH,
    EXPECTED_RAW_NODE_WIDTH,
    RoutedTail,
    RoutedTailSpecification,
    canonical_json_sha256,
    core_invariant_schema,
    correction_energy_and_forces,
    load_routed_tail,
    node_invariants,
    save_routed_tail,
    smootherstep,
    validate_invariant_schema,
)
from aggregate_graphene_r2m_routed_tail_outer_loco import (  # noqa: E402
    FOLD_LOCAL_LIMITS,
    raw_fold_observed,
    recompute_fold_local_gates,
)
from graphene_r2m_run_provenance import (  # noqa: E402
    sha256 as provenance_sha256,
    validate_completion_manifest,
)
from train_graphene_r2m_routed_tail_outer_fold import (  # noqa: E402
    EPOCHS,
    GROUP_MASS,
    SELECTION_POLICY,
    balanced_replay_indices,
    feature_statistics,
)


class _Irrep:
    def __init__(self, angular_momentum: int, parity: int) -> None:
        self.l = angular_momentum
        self.p = parity
        self.dim = 2 * angular_momentum + 1


class _Linear:
    def __init__(self, blocks: list[tuple[int, _Irrep]]) -> None:
        self.irreps_out = blocks


class _Product:
    def __init__(self, blocks: list[tuple[int, _Irrep]]) -> None:
        self.linear = _Linear(blocks)


class _LayoutCore:
    r_max = 2.0
    num_interactions = 2

    def __init__(self) -> None:
        self.products = [
            _Product([(16, _Irrep(0, 1)), (16, _Irrep(1, -1)), (16, _Irrep(2, 1))]),
            _Product([(16, _Irrep(0, 1))]),
        ]


class _DifferentiableFakeCore(torch.nn.Module):
    """Exact-layout fake whose features depend on relative coordinates."""

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
    ) -> dict[str, torch.Tensor]:
        del training, compute_force
        positions = data["positions"]
        batch = data["batch"]
        centered = torch.empty_like(positions)
        for graph in torch.unique(batch):
            mask = batch == graph
            centered[mask] = positions[mask] - positions[mask].mean(dim=0)
        radius2 = torch.sum(centered.square(), dim=1, keepdim=True)
        l0_first = torch.cat([radius2 * (index + 1) / 16.0 for index in range(16)], dim=1)
        l1 = torch.cat(
            [centered[:, None, :] * (index + 1) / 16.0 for index in range(16)],
            dim=1,
        ).reshape(len(positions), -1)
        l2 = positions.new_zeros((len(positions), 16 * 5))
        l0_second = torch.cat(
            [torch.sin(radius2 * (index + 1) / 16.0) for index in range(16)], dim=1
        )
        node_feats = torch.cat([l0_first, l1, l2, l0_second], dim=1)
        assert node_feats.shape[1] == EXPECTED_RAW_NODE_WIDTH
        return {"node_feats": node_feats}


def _layout_core() -> _LayoutCore:
    return _LayoutCore()


def _random_tail(pristine: torch.Tensor) -> RoutedTail:
    generator = torch.Generator().manual_seed(83)
    model = RoutedTail(
        RoutedTailSpecification(),
        np.zeros(EXPECTED_INVARIANT_WIDTH),
        np.ones(EXPECTED_INVARIANT_WIDTH),
        pristine,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(0.08 * torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype))
        model.router_encoder[-1].bias.fill_(0.5)
    return model


def test_fixed_core_schema_is_exactly_64_dimensional() -> None:
    schema = core_invariant_schema(_layout_core())
    validate_invariant_schema(schema)
    assert schema["raw_node_feats_width"] == 160
    assert schema["output_dimension"] == 64
    assert [item["operation"] for item in schema["blocks"]] == [
        "signed_l0_value",
        "mean_m_squared",
        "mean_m_squared",
        "signed_l0_value",
    ]


def test_outer_fold_training_has_one_fixed_endpoint_and_balanced_loss() -> None:
    assert EPOCHS == 240
    assert SELECTION_POLICY == "fixed_epoch_240_no_support_checkpoint_selection"
    assert GROUP_MASS == {
        "support": 0.50,
        "harmonic": 0.25,
        "e50": 0.15,
        "auxiliary": 0.10,
    }
    assert abs(sum(GROUP_MASS.values()) - 1.0) <= 1.0e-15


def test_training_only_scaler_keeps_all_64_columns_with_relative_floor() -> None:
    values = np.zeros((12, EXPECTED_INVARIANT_WIDTH), float)
    values[:, 0] = np.linspace(-1.0, 1.0, len(values))
    values[:, 1] = 1.0e-12 * np.linspace(-1.0, 1.0, len(values))
    mean, scale, record = feature_statistics(values)
    assert mean.shape == scale.shape == (EXPECTED_INVARIANT_WIDTH,)
    assert np.all(scale > 0.0)
    assert abs(scale[1] - 1.0e-8) <= 1.0e-20
    assert record["feature_pruning"] is False
    assert record["input_dimension"] == record["output_dimension"] == 64


def test_replay_schedule_is_balanced_over_all_240_epochs() -> None:
    for length, salt in ((72, 23), (20, 37), (72, 53)):
        counts = np.zeros(length, int)
        for epoch in range(1, EPOCHS + 1):
            indices = balanced_replay_indices(length, epoch, 8, salt)
            counts[indices] += 1
        assert int(np.max(counts) - np.min(counts)) <= 1
        assert int(np.sum(counts)) == EPOCHS * 8


def test_schema_rejects_hash_and_layout_changes() -> None:
    schema = core_invariant_schema(_layout_core())
    changed = copy.deepcopy(schema)
    changed["blocks"][0]["parity"] = -1
    with pytest.raises(ValueError, match="SHA-256"):
        validate_invariant_schema(changed)

    class WrongCore(_LayoutCore):
        r_max = 3.0

    with pytest.raises(ValueError, match="r_max"):
        core_invariant_schema(WrongCore())


def test_schema_rejects_rehashed_slice_operation_and_name_changes() -> None:
    for mutate in (
        lambda value: value["blocks"][1].update(raw_slice=[17, 65]),
        lambda value: value["blocks"][2].update(operation="signed_l0_value"),
        lambda value: value["blocks"][3]["feature_names"].__setitem__(0, "wrong"),
    ):
        schema = core_invariant_schema(_layout_core())
        mutate(schema)
        payload = dict(schema)
        payload.pop("schema_sha256")
        schema["schema_sha256"] = canonical_json_sha256(payload)
        with pytest.raises(ValueError, match="schema"):
            validate_invariant_schema(schema)


def test_routed_tail_rejects_nonfinite_frozen_statistics() -> None:
    mean = np.zeros(EXPECTED_INVARIANT_WIDTH)
    scale = np.ones(EXPECTED_INVARIANT_WIDTH)
    pristine = np.zeros((2, EXPECTED_INVARIANT_WIDTH))
    for target, index in ((mean, 0), (scale, 1), (pristine, (0, 2))):
        changed_mean = mean.copy()
        changed_scale = scale.copy()
        changed_pristine = pristine.copy()
        if target is mean:
            changed_mean[index] = np.nan
        elif target is scale:
            changed_scale[index] = np.inf
        else:
            changed_pristine[index] = np.nan
        with pytest.raises(ValueError):
            RoutedTail(
                RoutedTailSpecification(),
                changed_mean,
                changed_scale,
                changed_pristine,
            )


def test_fold_local_gate_boolean_is_recomputed() -> None:
    gates = {
        name: {"observed": 0.5 * limit, "threshold": limit, "pass": True}
        for name, limit in FOLD_LOCAL_LIMITS.items()
    }
    summary = {"fixed_gates": gates, "passes_fold_local_fixed_gates": True}
    assert recompute_fold_local_gates(summary) is True
    summary["fixed_gates"]["support_force_RMSE_meV_A"]["pass"] = False
    with pytest.raises(ValueError, match="boolean"):
        recompute_fold_local_gates(summary)


def test_raw_fold_gate_inputs_require_exact_shapes_and_are_recomputed() -> None:
    coordinates = np.linspace(0.01, 0.20, 20).astype(complex)
    target = -2.0 * coordinates
    data = {
        "held_force_error_eV_A": np.zeros((72, 3)),
        "held_tail_force_eV_A": np.zeros((72, 3)),
        "held_gate": np.zeros(72),
        "held_energy_error_eV": np.asarray(0.0),
        "support_Aprime_coordinate": np.asarray(0.1 + 0.0j),
        "support_Aprime_predicted_force": np.asarray(-0.2 + 0.0j),
        "support_Aprime_target_force": np.asarray(-0.2 + 0.0j),
        "seed1_force_errors_eV_A": np.zeros((20, 72, 3)),
        "seed1_Aprime_coordinates": coordinates,
        "seed1_Aprime_predicted_forces": target.copy(),
        "seed1_Aprime_target_forces": target,
        "harmonic_force_errors_eV_A": np.zeros((25, 128, 3)),
        "harmonic_gate_means": np.zeros(25),
    }
    summary = {
        "mechanics": {
            "finite_difference": {
                "step_A": 1.0e-4,
                "tail_energy_eV": 0.0,
                "records": [
                    {
                        "analytic_eV_A": 0.0,
                        "numerical_eV_A": 0.0,
                        "absolute_error_eV_A": 0.0,
                    }
                    for _ in range(3)
                ],
                "maximum_absolute_error_eV_A": 0.0,
            },
            "Hessian_symmetry": {
                "matrix_dimension": 216,
                "maximum_absolute_antisymmetry_eV_A2": 0.0,
                "RMS_antisymmetry_eV_A2": 0.0,
            },
            "O3": {
                "records": [
                    {
                        "transformation": "rotation",
                        "determinant": 1.0,
                        "energy_absolute_error_eV": 0.0,
                        "force_equivariance_max_abs_eV_A": 0.0,
                    },
                    {
                        "transformation": "reflection",
                        "determinant": -1.0,
                        "energy_absolute_error_eV": 0.0,
                        "force_equivariance_max_abs_eV_A": 0.0,
                    },
                ],
                "energy_max_abs_eV": 0.0,
                "force_equivariance_max_abs_eV_A": 0.0,
            },
            "size_consistency": {
                "test": "same centered local displacement in pristine 6x6 and 8x8 cells",
                "center_indices": [42, 72],
                "tail_force_max_abs_difference_eV_A": 0.0,
                "tail_node_energy_abs_difference_eV": 0.0,
                "tail_gate_abs_difference": 0.0,
                "core_plus_tail_force_max_abs_difference_eV_A": 0.0,
            },
        }
    }
    observed, _ = raw_fold_observed(summary, data)
    assert set(observed) == set(FOLD_LOCAL_LIMITS)
    assert all(value == 0.0 for value in observed.values())
    wrong = dict(data)
    wrong["harmonic_force_errors_eV_A"] = np.zeros((25, 127, 3))
    with pytest.raises(ValueError, match="shape"):
        raw_fold_observed(summary, wrong)


def test_completion_manifest_closes_disconnect_before_done_window(tmp_path: Path) -> None:
    output = tmp_path / "outer00"
    code = output / "code_snapshots"
    train = output / "train"
    code.mkdir(parents=True)
    train.mkdir()
    provenance = code / "graphene_r2m_run_provenance.py"
    provenance.write_text("frozen provenance\n", encoding="utf-8")
    freeze_path = output / "launcher_freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    freeze = {
        "outer_fold": 0,
        "held_sscha_index": 11,
        "code_snapshots": {
            "graphene_r2m_run_provenance.py": {"path": str(provenance)}
        },
    }
    files = {
        "core": output / "core.model",
        "fold_manifest": output / "fold_manifest.json",
        "tail": train / "r2m_routed_tail_epoch240_ema.pt",
        "arrays": output / "outer_fold_evaluation.npz",
        "training_summary": train / "training_summary.json",
        "training_freeze": train / "training_freeze.json",
        "training_metrics": train / "training_metrics.jsonl",
        "TRAINING_DONE": train / "TRAINING_DONE",
        "launcher_freeze": freeze_path,
    }
    for name, path in files.items():
        if name != "launcher_freeze":
            path.write_bytes(b"" if name == "TRAINING_DONE" else b"frozen\n")
    evaluation_path = output / "outer_fold_evaluation.json"
    evaluation = {
        "status": "R2M_routed_tail_outer_fold_post_freeze_evaluation_complete",
        "outer_fold": 0,
        "held_sscha_index": 11,
        "selection_policy": "fixed_epoch_240_no_support_checkpoint_selection",
        "held_opened_only_after_epoch240_EMA_hash_freeze": True,
        "training_provenance_verified_before_held_read": True,
        "E50_seed2_read": False,
        "full_composite_deployment_authorized": False,
        "artifacts": {
            name: {"path": str(path), "sha256": provenance_sha256(path)}
            for name, path in files.items()
        },
    }
    evaluation_path.write_text(json.dumps(evaluation) + "\n", encoding="utf-8")
    completion_path = output / "completion_manifest.json"
    completion = {
        "status": "R2M_routed_tail_outer_fold_execution_complete",
        "outer_fold": 0,
        "held_sscha_index": 11,
        "launcher_freeze": {
            "path": str(freeze_path),
            "sha256": provenance_sha256(freeze_path),
        },
        "training_summary": {
            "path": str(files["training_summary"]),
            "sha256": provenance_sha256(files["training_summary"]),
        },
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": provenance_sha256(evaluation_path),
        },
    }
    completion_path.write_text(json.dumps(completion) + "\n", encoding="utf-8")
    validate_completion_manifest(completion_path, freeze_path, freeze)
    evaluation_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation changed"):
        validate_completion_manifest(completion_path, freeze_path, freeze)


def test_tensor_power_features_are_orthogonal_and_reflection_invariant() -> None:
    schema = core_invariant_schema(_layout_core())
    generator = torch.Generator().manual_seed(19)
    raw = torch.randn(5, EXPECTED_RAW_NODE_WIDTH, generator=generator, dtype=torch.float64)
    transformed = raw.clone()
    for item in schema["blocks"]:
        if item["l"] == 0:
            continue
        start, stop = item["raw_slice"]
        block = transformed[:, start:stop].reshape(
            len(raw), item["multiplicity"], item["dimension"]
        )
        matrix = torch.randn(
            item["dimension"], item["dimension"], generator=generator, dtype=raw.dtype
        )
        orthogonal, _ = torch.linalg.qr(matrix)
        transformed[:, start:stop] = (block @ orthogonal).reshape(len(raw), -1)
    expected = node_invariants(raw, schema)
    observed = node_invariants(transformed, schema)
    torch.testing.assert_close(observed, expected, rtol=2.0e-12, atol=2.0e-12)


def test_smootherstep_is_c2_at_both_boundaries() -> None:
    for point in (0.0, 1.0):
        value = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        output = smootherstep(value)
        first = torch.autograd.grad(output, value, create_graph=True)[0]
        second = torch.autograd.grad(first, value)[0]
        assert abs(float(first.detach())) <= 1.0e-14
        assert abs(float(second.detach())) <= 1.0e-14


def test_pristine_carbon_gauge_zeros_identical_node_energy() -> None:
    row = torch.linspace(-0.2, 0.2, EXPECTED_INVARIANT_WIDTH, dtype=torch.float64)
    pristine = row.repeat(12, 1)
    model = _random_tail(pristine)
    energy, diagnostics = model.graph_energies(pristine, torch.zeros(12, dtype=torch.long))
    assert abs(float(energy.detach())) <= 1.0e-12
    torch.testing.assert_close(
        diagnostics["epsilon"],
        diagnostics["c_C_eV"].expand_as(diagnostics["epsilon"]),
    )


def test_full_energy_force_contains_router_product_rule() -> None:
    generator = torch.Generator().manual_seed(31)
    pristine = torch.zeros(6, EXPECTED_INVARIANT_WIDTH, dtype=torch.float64)
    model = _random_tail(pristine)
    invariants = (0.25 * torch.randn(
        6, EXPECTED_INVARIANT_WIDTH, generator=generator, dtype=torch.float64
    )).requires_grad_(True)
    _, gate, _, epsilon = model.node_terms(invariants)
    c_c = model.pristine_carbon_gauge()
    full_energy = torch.sum(gate * (epsilon - c_c))
    full_force = -torch.autograd.grad(full_energy, invariants, retain_graph=True)[0]
    shortcut_energy = torch.sum(gate.detach() * (epsilon - c_c))
    shortcut_force = -torch.autograd.grad(shortcut_energy, invariants)[0]
    assert float(torch.max(torch.abs(full_force - shortcut_force))) > 1.0e-9


def test_complete_autograd_force_matches_finite_difference() -> None:
    schema = core_invariant_schema(_layout_core())
    fake_core = _DifferentiableFakeCore()
    pristine_positions = torch.tensor(
        [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.8, 0.0]],
        dtype=torch.float64,
    )
    pristine_raw = fake_core(
        {"positions": pristine_positions, "batch": torch.zeros(3, dtype=torch.long)}
    )["node_feats"]
    pristine = node_invariants(pristine_raw, schema).detach()
    tail = _random_tail(pristine)
    positions = torch.tensor(
        [[-0.62, 0.03, 0.02], [0.73, -0.04, 0.01], [0.01, 0.82, -0.03]],
        dtype=torch.float64,
    )
    data = {"positions": positions, "batch": torch.zeros(3, dtype=torch.long)}
    energy, force, _ = correction_energy_and_forces(
        fake_core, tail, data, schema, create_graph=False
    )
    step = 1.0e-5
    plus = positions.clone()
    minus = positions.clone()
    plus[0, 1] += step
    minus[0, 1] -= step
    energy_plus = correction_energy_and_forces(
        fake_core,
        tail,
        {"positions": plus, "batch": data["batch"]},
        schema,
        create_graph=False,
    )[0]
    energy_minus = correction_energy_and_forces(
        fake_core,
        tail,
        {"positions": minus, "batch": data["batch"]},
        schema,
        create_graph=False,
    )[0]
    numerical = -float(
        ((energy_plus.sum() - energy_minus.sum()) / (2.0 * step)).detach()
    )
    assert abs(float(force[0, 1]) - numerical) <= 2.0e-7
    assert torch.isfinite(energy).all()


def test_complete_autograd_hessian_is_symmetric() -> None:
    schema = core_invariant_schema(_layout_core())
    fake_core = _DifferentiableFakeCore()
    positions = torch.tensor(
        [[-0.62, 0.03, 0.02], [0.73, -0.04, 0.01], [0.01, 0.82, -0.03]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = torch.zeros(3, dtype=torch.long)
    pristine_raw = fake_core(
        {"positions": positions.detach(), "batch": batch}
    )["node_feats"]
    pristine = node_invariants(pristine_raw, schema).detach()
    tail = _random_tail(pristine)

    def force_function(candidate: torch.Tensor) -> torch.Tensor:
        _, force, _ = correction_energy_and_forces(
            fake_core,
            tail,
            {"positions": candidate, "batch": batch},
            schema,
            create_graph=True,
        )
        return force

    jacobian = torch.autograd.functional.jacobian(force_function, positions)
    matrix = jacobian.reshape(positions.numel(), positions.numel())
    assert float(torch.max(torch.abs(matrix - matrix.T))) <= 2.0e-12


def test_fake_core_path_is_translation_rotation_and_reflection_safe() -> None:
    schema = core_invariant_schema(_layout_core())
    fake_core = _DifferentiableFakeCore()
    positions = torch.tensor(
        [[-0.6, 0.1, 0.2], [0.8, -0.2, 0.0], [0.0, 0.7, -0.1]],
        dtype=torch.float64,
    )
    batch = torch.zeros(3, dtype=torch.long)
    pristine_raw = fake_core({"positions": positions, "batch": batch})["node_feats"]
    pristine = node_invariants(pristine_raw, schema).detach()
    tail = _random_tail(torch.zeros_like(pristine))

    def value(candidate: torch.Tensor) -> tuple[float, torch.Tensor]:
        energy, force, _ = correction_energy_and_forces(
            fake_core,
            tail,
            {"positions": candidate, "batch": batch},
            schema,
            create_graph=False,
        )
        return float(energy.sum().detach()), force

    base_energy, base_force = value(positions)
    shifted_energy, shifted_force = value(positions + torch.tensor([1.2, -0.4, 0.7]))
    assert abs(base_energy - shifted_energy) <= 2.0e-12
    torch.testing.assert_close(base_force, shifted_force, rtol=2.0e-11, atol=2.0e-11)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    rotated_energy, rotated_force = value(positions @ rotation.T)
    assert abs(base_energy - rotated_energy) <= 2.0e-12
    torch.testing.assert_close(
        rotated_force, base_force @ rotation.T, rtol=2.0e-10, atol=2.0e-10
    )
    assert float(torch.max(torch.abs(base_force.sum(dim=0)))) <= 2.0e-12


def test_artifact_is_bound_to_exact_core_hash(tmp_path: Path) -> None:
    schema = core_invariant_schema(_layout_core())
    pristine = torch.zeros(4, EXPECTED_INVARIANT_WIDTH, dtype=torch.float64)
    model = _random_tail(pristine)
    artifact = tmp_path / "tail.pt"
    core_hash = "ab" * 32
    save_routed_tail(artifact, model, schema, core_hash, {"outer_fold": 3})
    replay, replay_schema, metadata = load_routed_tail(
        artifact, torch.device("cpu"), core_hash
    )
    assert replay_schema == schema
    assert metadata == {"outer_fold": 3}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(replay.state_dict()[name], value)
    with pytest.raises(ValueError, match="another core"):
        load_routed_tail(artifact, torch.device("cpu"), "cd" * 32)
