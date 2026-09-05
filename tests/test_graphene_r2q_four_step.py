from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "smearing_kink"
sys.path.insert(0, str(SCRIPTS))

import graphene_r2q_four_step as r2q  # noqa: E402


class TinyState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.large = torch.nn.Parameter(
            torch.tensor([3.0, 4.0], dtype=torch.float64)
        )
        self.small = torch.nn.Parameter(torch.tensor([0.01], dtype=torch.float64))
        self.register_buffer("fixed_buffer", torch.tensor([7.0], dtype=torch.float64))


class ZeroTensorState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.active = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.zero = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.register_buffer("fixed_buffer", torch.tensor([2.0], dtype=torch.float64))


def _block_ids() -> list[str]:
    return [f"block:{index:03d}" for index in range(176)]


def test_protocol_freezes_contract_schedule_and_reference_limits() -> None:
    assert r2q.EXPECTED_CONTRACT_DOC_SHA256 == (
        "8b287433b1046d4c229d1d3bb12aeba6bb9d5b0180af2ebf7605802935f614f0"
    )
    assert r2q.EXPECTED_BASE_STATE_SHA256 == (
        "09c29479be6c8f207211614c2b3da20f09aba0702092a2b1bf8edab5a5e6d236"
    )
    assert r2q.STEP_COUNT == 4
    assert r2q.HALVING_COUNT == 6
    assert r2q.GLOBAL_RELATIVE_TRUST_CAP == 1.0e-3
    assert r2q.PER_TENSOR_RELATIVE_TRUST_CAP == 2.0e-2
    assert r2q.ARMIJO_C1 == 0.1
    assert r2q.ARMIJO_NUMERIC_EPS == 1.0e-12
    assert r2q.PROTOCOL["candidate_halving_indices"] == list(range(6))
    assert r2q.PROTOCOL["steps_exact"] == 4
    assert r2q.PROTOCOL["gradient_blocks_each_step"] == {
        "Aprime_seed0": 20,
        "thermal_total_force": 92,
        "small_zero_RMS": 32,
        "small_zero_exact_top": 32,
    }
    assert r2q.PROTOCOL["optimizer_instantiated"] is False
    assert r2q.PROTOCOL["optimizer_step_called"] is False
    assert r2q.PROTOCOL_SHA256 == r2q.r2p.canonical_json_sha256(r2q.PROTOCOL)
    assert r2q.REFERENCE_LIMITS == {
        "energy_abs_eV": 1.0e-10,
        "force_max_abs_eV_A": 1.0e-9,
        "Hessian_max_abs_eV_A2": 1.0e-7,
        "Hessian_symmetry_max_abs_eV_A2": 1.0e-7,
        "Hessian_ASR_row_sum_max_abs_eV_A2": 1.0e-7,
        "Gamma_K_frequency_drift_upper_bound_cm-1": 2.0,
    }
    assert r2q.sha256(
        ROOT / "docs" / "GRAPHENE_R2Q_FINITE_TRUST_REGION_CONTRACT_DRAFT_2026-08-25.md"
    ) == r2q.EXPECTED_CONTRACT_DOC_SHA256


def test_source_has_no_optimizer_backward_step_or_default_dtype_mutation() -> None:
    source_path = Path(r2q.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden_attributes = {
        node.func.attr
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr in {"backward", "step", "set_default_dtype"}
    }
    forbidden_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("torch.optim")
    }
    forbidden_from_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("torch.optim")
    }
    assert forbidden_attributes == set()
    assert forbidden_imports == set()
    assert forbidden_from_imports == set()
    assert "torch.optim" not in source


def test_live_decision_helper_source_hashes_are_frozen() -> None:
    assert r2q._validated_source_hashes() == {
        "R2P_gradient_and_cone": r2q.EXPECTED_R2P_SOURCE_SHA256,
        "R2O_Taylor_wrapper": r2q.EXPECTED_R2O_WRAPPER_SHA256,
        "R2O_endpoint_evaluator": r2q.EXPECTED_R2O_EVALUATOR_SHA256,
    }


def test_fixed_grid_reproduces_step1_eta_and_all_six_halvings() -> None:
    grid = r2q.fixed_candidate_grid(216.54915896171457, 1.0)
    assert len(grid) == 6
    assert grid[0] == pytest.approx(r2q.EXPECTED_STEP1_ETA0, abs=5.0e-17)
    assert grid == pytest.approx(
        [r2q.EXPECTED_STEP1_ETA0 / (2**index) for index in range(6)],
        rel=0.0,
        abs=5.0e-17,
    )
    other = r2q.fixed_candidate_grid(12.5, 2.0)
    assert other == pytest.approx([0.00625 / (2**index) for index in range(6)])


@pytest.mark.parametrize(
    ("theta_norm", "direction_norm"),
    [(0.0, 1.0), (-1.0, 1.0), (math.nan, 1.0), (1.0, 0.0), (1.0, math.inf)],
)
def test_fixed_grid_rejects_nonfinite_or_nonpositive_norms(
    theta_norm: float, direction_norm: float
) -> None:
    with pytest.raises(ValueError):
        r2q.fixed_candidate_grid(theta_norm, direction_norm)


def test_candidate_exact_top_unique_value_and_metadata_are_exact() -> None:
    force = np.asarray(
        [[0.001, -0.004, 0.002], [0.003, 0.0005, -0.001]],
        dtype=np.float64,
    )
    value, metadata = r2q._top_value_metadata(force)
    expected = (
        0.004**2
        / 32.0
        / (r2q.r2p.SMALL_CORRECTION_MARGIN_EV_A**2)
    )
    assert value == pytest.approx(expected)
    assert metadata["top_flat_index"] == 1
    assert metadata["top_atom_source_order_0based"] == 0
    assert metadata["top_cartesian_component"] == "y"
    assert metadata["top_signed_eV_A"] == pytest.approx(-0.004)
    assert metadata["second_flat_index"] == 3
    assert metadata["second_atom_source_order_0based"] == 1
    assert metadata["second_cartesian_component"] == "x"
    assert metadata["second_signed_eV_A"] == pytest.approx(0.003)
    assert metadata["exact_tie_count"] == 1
    assert metadata["exact_tie_indices"] == [1]
    assert metadata["absolute_gap_eV_A"] == pytest.approx(0.001)
    assert metadata["relative_gap"] == pytest.approx(0.25)


def test_candidate_exact_top_allows_and_records_an_exact_tie() -> None:
    force = np.asarray([[0.004, -0.004, 0.001]], dtype=np.float64)
    value, metadata = r2q._top_value_metadata(force)
    assert value == pytest.approx(
        0.004**2 / 32.0 / r2q.r2p.SMALL_CORRECTION_MARGIN_EV_A**2
    )
    assert metadata["top_flat_index"] == 0
    assert metadata["second_flat_index"] == 1
    assert metadata["exact_tie_count"] == 2
    assert metadata["exact_tie_indices"] == [0, 1]
    assert metadata["absolute_gap_eV_A"] == 0.0
    assert metadata["relative_gap"] == 0.0
    assert metadata["candidate_tie_role"] == "record_only_no_candidate_gradient"


def test_candidate_exact_top_allows_a_near_tie_and_reselects_the_candidate_top() -> None:
    force = np.asarray(
        [[0.002, -(0.004 - 5.0e-9), 0.004]], dtype=np.float64
    )
    value, metadata = r2q._top_value_metadata(force)
    assert metadata["top_flat_index"] == 2
    assert metadata["second_flat_index"] == 1
    assert metadata["exact_tie_count"] == 1
    assert 0.0 < metadata["absolute_gap_eV_A"] <= 1.0e-8
    assert value == pytest.approx(
        0.004**2 / 32.0 / r2q.r2p.SMALL_CORRECTION_MARGIN_EV_A**2
    )


def test_candidate_exact_top_all_zero_is_finite_and_records_all_ties() -> None:
    value, metadata = r2q._top_value_metadata(np.zeros((2, 3), dtype=np.float64))
    assert value == 0.0
    assert metadata["top_flat_index"] == 0
    assert metadata["second_flat_index"] == 1
    assert metadata["exact_tie_count"] == 6
    assert metadata["exact_tie_indices"] == list(range(6))
    assert metadata["absolute_gap_eV_A"] == 0.0
    assert metadata["relative_gap"] == 0.0
    assert all(
        math.isfinite(float(metadata[name]))
        for name in ("top_abs_eV_A", "second_abs_eV_A", "relative_gap")
    )


@pytest.mark.parametrize(
    "force",
    [
        np.asarray([math.nan, 0.0]),
        np.asarray([math.inf, 0.0]),
        np.asarray([0.0]),
    ],
)
def test_candidate_exact_top_rejects_nonfinite_or_malformed_force(
    force: np.ndarray,
) -> None:
    with pytest.raises(FloatingPointError):
        r2q._top_value_metadata(force)


@pytest.mark.parametrize(
    "force",
    [
        [[0.004, -0.004, 0.001]],
        [[0.004, -(0.004 - 5.0e-9), 0.001]],
    ],
)
def test_r2p_gradient_base_still_rejects_exact_and_near_ties(
    force: list[list[float]],
) -> None:
    live = torch.tensor(force, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError):
        r2q.r2p._exact_top_objective(
            live,
            scale_eV_A=r2q.r2p.SMALL_CORRECTION_MARGIN_EV_A,
            aggregate_coefficient=1.0 / 32.0,
        )


def test_armijo_is_per_block_uses_raw_slope_and_has_inclusive_boundary() -> None:
    base = np.ones(176, dtype=np.float64)
    slopes = np.ones(176, dtype=np.float64)
    slopes[0] = 3.0
    eta = 0.1
    epsilon = r2q.ARMIJO_NUMERIC_EPS * np.maximum(1.0, np.abs(base))
    upper = base - r2q.ARMIJO_C1 * eta * slopes + epsilon
    receipt = r2q.armijo_receipt(base, slopes, upper, eta, _block_ids())
    assert receipt["all_blocks_pass"] is True
    assert receipt["failed_block_count"] == 0
    assert receipt["per_block"][0]["raw_directional_slope"] == 3.0
    assert receipt["per_block"][0]["required_upper_bound"] == pytest.approx(
        1.0 - 0.1 * eta * 3.0 + 1.0e-12
    )
    assert receipt["per_block"][0]["Armijo_margin"] == pytest.approx(0.0)

    one_block_fails = upper.copy()
    one_block_fails[0] -= 0.5
    one_block_fails[-1] = np.nextafter(upper[-1], np.inf)
    assert float(np.sum(one_block_fails)) < float(np.sum(upper))
    rejected = r2q.armijo_receipt(
        base, slopes, one_block_fails, eta, _block_ids()
    )
    assert rejected["all_blocks_pass"] is False
    assert rejected["failed_block_count"] == 1
    assert rejected["minimum_margin_block_id"] == "block:175"
    assert rejected["per_block"][-1]["pass"] is False


def test_armijo_rejects_misalignment_nonfinite_and_nonpositive_raw_slope() -> None:
    values = np.ones(176, dtype=np.float64)
    with pytest.raises(ValueError, match="exactly 176"):
        r2q.armijo_receipt(values[:-1], values, values, 0.1, _block_ids())
    duplicate_ids = _block_ids()
    duplicate_ids[-1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="block ids"):
        r2q.armijo_receipt(values, values, values, 0.1, duplicate_ids)
    bad_slope = values.copy()
    bad_slope[17] = 0.0
    with pytest.raises(ValueError, match="nonpositive raw"):
        r2q.armijo_receipt(values, bad_slope, values, 0.1, _block_ids())
    nonfinite = values.copy()
    nonfinite[4] = math.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        r2q.armijo_receipt(values, values, nonfinite, 0.1, _block_ids())


def test_held_science_gate_rejects_nonfinite_metrics_or_invalid_thresholds() -> None:
    endpoint = {
        "E50_seed1": {
            "force_error": {"RMSE_meV_A": 1.0, "max_abs_meV_A": 2.0},
            "Aprime_force_error_RMS_meV_A": 3.0,
            "relative_slope_error": 0.01,
        },
        "harmonic_lambda1_small_gate": {
            "force_error": {"RMSE_meV_A": 0.1, "max_abs_meV_A": 1.0}
        },
    }
    thresholds = {
        "e50_seed1_force_RMSE_meV_A": 30.0,
        "e50_seed1_force_max_abs_meV_A": 200.0,
        "e50_seed1_Aprime_RMS_meV_A": 15.0,
        "e50_seed1_Aprime_slope_relative_error_abs": 0.05,
        "harmonic_small_force_RMSE_meV_A": 0.5,
        "harmonic_small_force_max_abs_meV_A": 10.0,
    }
    assert r2q._held_science_gate(endpoint, thresholds)["pass"] is True
    bad_endpoint = copy.deepcopy(endpoint)
    bad_endpoint["E50_seed1"]["force_error"]["RMSE_meV_A"] = math.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        r2q._held_science_gate(bad_endpoint, thresholds)
    bad_thresholds = dict(thresholds)
    bad_thresholds["harmonic_small_force_max_abs_meV_A"] = 0.0
    with pytest.raises(ValueError, match="threshold"):
        r2q._held_science_gate(endpoint, bad_thresholds)


def test_json_safe_diagnostics_never_emit_nonfinite_json_numbers() -> None:
    safe = r2q._json_safe(
        {"nan": math.nan, "inf": np.float64(math.inf), "flag": np.bool_(True)}
    )
    assert safe == {
        "nan": "nonfinite:nan",
        "inf": "nonfinite:inf",
        "flag": True,
    }
    json.dumps(safe, allow_nan=False)


def test_apply_direction_clones_state_preserves_schema_and_buffers_and_audits_trust() -> None:
    base = TinyState()
    base_hash = r2q.state_dict_sha256(base)
    base_schema = r2q._parameter_schema(base)
    base_buffer = base.fixed_buffer.detach().clone()
    direction = np.asarray([math.sqrt(0.99), 0.0, 0.1], dtype=np.float64)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    eta0 = r2q.fixed_candidate_grid(
        r2q.parameter_l2_norm(base), float(np.linalg.norm(direction))
    )[0]

    candidate, receipt = r2q.apply_direction_to_clone(base, direction, eta0)
    assert candidate is not base
    assert r2q.state_dict_sha256(base) == base_hash
    assert r2q._parameter_schema(candidate) == base_schema
    torch.testing.assert_close(base.fixed_buffer, base_buffer, rtol=0.0, atol=0.0)
    torch.testing.assert_close(candidate.fixed_buffer, base_buffer, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        candidate.large,
        base.large - eta0 * torch.tensor(direction[:2], dtype=torch.float64),
    )
    torch.testing.assert_close(
        candidate.small,
        base.small - eta0 * torch.tensor(direction[2:], dtype=torch.float64),
    )
    assert receipt["base_state_sha256"] == base_hash
    assert receipt["candidate_state_sha256"] == r2q.state_dict_sha256(candidate)
    assert receipt["candidate_state_sha256"] != base_hash
    assert receipt["nonparameter_buffers_unchanged"] is True
    assert receipt["update_replay_pass"] is True
    assert receipt["update_replay_epsilon_multiplier"] == 16.0
    assert receipt["update_replay_max_abs"] <= max(
        item["update_replay_tolerance"] for item in receipt["per_tensor"]
    )
    assert receipt["global_relative_update"] == pytest.approx(1.0e-3)
    assert receipt["global_actual_relative_update"] == pytest.approx(1.0e-3)
    assert all(item["update_replay_pass"] for item in receipt["per_tensor"])
    assert receipt["per_tensor_max_relative_update"] == pytest.approx(
        eta0 * 0.1 / 0.01
    )
    assert r2q.trust_receipt_passes(receipt) is False
    assert all(parameter.grad is None for parameter in base.parameters())
    assert all(parameter.grad is None for parameter in candidate.parameters())

    smaller, smaller_receipt = r2q.apply_direction_to_clone(
        base, direction, eta0 / 4.0
    )
    assert r2q.trust_receipt_passes(smaller_receipt) is True
    assert r2q.state_dict_sha256(smaller) == smaller_receipt[
        "candidate_state_sha256"
    ]


def test_zero_norm_tensor_allows_zero_block_but_rejects_nonzero_update() -> None:
    base = ZeroTensorState()
    base_hash = r2q.state_dict_sha256(base)
    candidate, receipt = r2q.apply_direction_to_clone(
        base, np.asarray([0.1, 0.0, 0.0], dtype=np.float64), 0.01
    )
    zero_row = next(item for item in receipt["per_tensor"] if item["name"] == "zero")
    assert zero_row["parameter_norm"] == 0.0
    assert zero_row["delta_norm"] == 0.0
    assert zero_row["relative_update"] == 0.0
    assert torch.equal(candidate.zero, base.zero)

    with pytest.raises(FloatingPointError, match="zero-norm parameter tensor zero"):
        r2q.apply_direction_to_clone(
            base, np.asarray([0.0, 1.0, 0.0], dtype=np.float64), 0.01
        )
    assert r2q.state_dict_sha256(base) == base_hash


def test_trust_receipt_boundary_is_global_tolerant_and_tensor_strict() -> None:
    planned_allowance = (
        r2q.UPDATE_REPLAY_EPS_MULTIPLIER
        * torch.finfo(torch.float64).eps
        * r2q.GLOBAL_RELATIVE_TRUST_CAP
    )

    def receipt(global_value: float, tensor_value: float) -> dict:
        planned = bool(
            math.isfinite(global_value)
            and global_value
            <= r2q.GLOBAL_RELATIVE_TRUST_CAP + planned_allowance
            and tensor_value <= r2q.PER_TENSOR_RELATIVE_TRUST_CAP
        )
        actual = planned
        return {
            "global_relative_update": global_value,
            "per_tensor_max_relative_update": tensor_value,
            "planned_global_cap_roundoff_allowance": planned_allowance,
            "planned_trust_pass": planned,
            "global_actual_relative_update": global_value,
            "actual_global_cap_roundoff_allowance": planned_allowance,
            "actual_trust_pass": actual,
            "per_tensor": [
                {
                    "actual_relative_update": tensor_value,
                    "actual_cap_roundoff_allowance": 0.0,
                    "update_replay_pass": True,
                }
            ],
            "update_replay_pass": True,
            "combined_trust_pass": bool(planned and actual),
        }

    assert r2q.trust_receipt_passes(
        receipt(np.nextafter(1.0e-3, np.inf), 2.0e-2)
    )
    assert not r2q.trust_receipt_passes(
        receipt(1.0e-3 + 2.0 * planned_allowance, 2.0e-2)
    )
    assert not r2q.trust_receipt_passes(
        receipt(1.0e-3, np.nextafter(2.0e-2, np.inf))
    )
    assert not r2q.trust_receipt_passes(receipt(math.nan, 0.0))


def test_actual_trust_uses_replay_derived_roundoff_allowance() -> None:
    base = TinyState()
    direction = np.asarray([math.sqrt(0.99), 0.0, 0.1], dtype=np.float64)
    eta = r2q.fixed_candidate_grid(
        r2q.parameter_l2_norm(base), float(np.linalg.norm(direction))
    )[2]
    _, receipt = r2q.apply_direction_to_clone(base, direction, eta)
    assert receipt["planned_trust_pass"] is True
    assert receipt["actual_trust_pass"] is True
    assert receipt["combined_trust_pass"] is True
    assert r2q.trust_receipt_passes(receipt) is True
    tampered = copy.deepcopy(receipt)
    row = tampered["per_tensor"][0]
    row["actual_relative_update"] = (
        r2q.PER_TENSOR_RELATIVE_TRUST_CAP
        + row["actual_cap_roundoff_allowance"]
        + 1.0e-15
    )
    tampered["actual_trust_pass"] = False
    tampered["combined_trust_pass"] = False
    assert r2q.trust_receipt_passes(tampered) is False


def test_step_artifact_requires_update_to_bind_actual_candidate_hash(
    tmp_path: Path,
) -> None:
    base = TinyState()
    parent_hash = r2q.state_dict_sha256(base)
    direction = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    candidate, update = r2q.apply_direction_to_clone(base, direction, 1.0e-3)
    cone = r2q.CurrentCone(
        blocks=[],
        matrix=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        records=[],
        certificate=SimpleNamespace(gram=np.ones((1, 1), dtype=np.float64)),
        state_sha256=parent_hash,
        wall_seconds=0.0,
    )
    bad_update = dict(update)
    bad_update["candidate_state_sha256"] = "0" * 64
    bad_root = tmp_path / "bad_step"
    with pytest.raises(RuntimeError, match="accepted R2Q candidate semantic state changed"):
        r2q._write_step_artifacts(
            bad_root,
            step=1,
            parent_state_sha256=parent_hash,
            model=candidate,
            cone=cone,
            cone_payload={"status": "synthetic"},
            direction=direction,
            armijo={"all_blocks_pass": True},
            reference={"pass": True},
            update=bad_update,
        )
    assert not bad_root.exists()

    step_root = tmp_path / "step_01"
    receipt = r2q._write_step_artifacts(
        step_root,
        step=1,
        parent_state_sha256=parent_hash,
        model=candidate,
        cone=cone,
        cone_payload={"status": "synthetic"},
        direction=direction,
        armijo={"all_blocks_pass": True},
        reference={"pass": True},
        update=update,
    )
    assert receipt["model_state_sha256"] == update["candidate_state_sha256"]
    assert receipt["update"]["base_state_sha256"] == parent_hash
    assert receipt["update"]["candidate_state_sha256"] == r2q.state_dict_sha256(
        candidate
    )
    assert (step_root / "STEP_ACCEPTED").read_bytes() == (
        receipt["step_receipt_sha256"] + "\n"
    ).encode("ascii")
    assert r2q.sha256(step_root / "accepted_step.pt") == receipt[
        "checkpoint_sha256"
    ]


def test_frozen_endpoint_loader_rejects_marker_before_checkpoint_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "ENDPOINT_FROZEN"
    marker.write_bytes(b"wrong\n")
    opened = []

    def trap(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("endpoint checkpoint must remain unopened")

    monkeypatch.setattr(r2q, "torch_load", trap)
    with pytest.raises(ValueError, match="marker does not bind"):
        r2q._load_frozen_endpoint(
            tmp_path / "endpoint.pt",
            tmp_path / "endpoint_receipt.json",
            marker,
            expected_receipt_hash="a" * 64,
            expected_state="b" * 64,
            expected_run_contract_sha256="c" * 64,
        )
    assert opened == []


def test_held_loader_rejects_endpoint_marker_before_resolving_held_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "ENDPOINT_FROZEN"
    held_started = tmp_path / "HELD_EVALUATION_STARTED"
    marker.write_bytes(b"wrong\n")
    held_started.write_bytes(b"wrong\n")
    resolved = []

    def trap(*args, **kwargs):
        resolved.append((args, kwargs))
        raise AssertionError("held path must not be resolved or opened")

    monkeypatch.setattr(r2q.r2p, "_require_file", trap)
    loaded = SimpleNamespace(data_root=tmp_path / "held_data", data_manifest={})
    with pytest.raises(ValueError, match="endpoint marker changed before held open"):
        r2q._held_structures(loaded, marker, held_started, "a" * 64)
    assert resolved == []


def test_held_loader_rejects_held_start_marker_before_resolving_held_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint_receipt = tmp_path / "endpoint_receipt.json"
    r2q.r2p.strict_json(endpoint_receipt, {"status": "synthetic"})
    receipt_hash = r2q.sha256(endpoint_receipt)
    marker = tmp_path / "ENDPOINT_FROZEN"
    held_started = tmp_path / "HELD_EVALUATION_STARTED"
    marker.write_bytes((receipt_hash + "\n").encode("ascii"))
    held_started.write_bytes(b"wrong\n")
    resolved = []

    def trap(*args, **kwargs):
        resolved.append((args, kwargs))
        raise AssertionError("held path must not be resolved or opened")

    monkeypatch.setattr(r2q.r2p, "_require_file", trap)
    loaded = SimpleNamespace(data_root=tmp_path / "held_data", data_manifest={})
    with pytest.raises(ValueError, match="held-start marker does not bind"):
        r2q._held_structures(loaded, marker, held_started, receipt_hash)
    assert resolved == []


class _FakeFrozenArray:
    def __init__(self, shape: tuple[int, ...], semantic_sha256: str) -> None:
        self.dtype = np.dtype(np.float64)
        self.shape = shape
        self.semantic_sha256 = semantic_sha256


def _synthetic_sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _install_synthetic_four_step_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    """Install lightweight file/array/model readers for the four-step verifier."""

    output = tmp_path / "run"
    output.mkdir()
    hashes: dict[Path, str] = {}
    arrays: dict[Path, _FakeFrozenArray] = {}
    receipts: dict[Path, dict] = {}
    checkpoints: dict[Path, dict] = {}
    checkpoint_loads: list[Path] = []
    chain: list[dict] = []
    states: list[str] = []
    parent = r2q.EXPECTED_BASE_STATE_SHA256

    for step in range(1, 5):
        root = output / "steps" / f"step_{step:02d}"
        root.mkdir(parents=True)
        model = TinyState()
        with torch.no_grad():
            model.large.add_(step * 1.0e-3)
            model.small.add_(step * 1.0e-5)
        state = r2q.state_dict_sha256(model)
        states.append(state)

        gradient_file = _synthetic_sha256(f"step{step}:gradient:file")
        gradient_semantic = _synthetic_sha256(f"step{step}:gradient:semantic")
        gram_file = _synthetic_sha256(f"step{step}:gram:file")
        gram_semantic = _synthetic_sha256(f"step{step}:gram:semantic")
        direction_file = _synthetic_sha256(f"step{step}:direction:file")
        direction_semantic = _synthetic_sha256(f"step{step}:direction:semantic")
        cone_file = _synthetic_sha256(f"step{step}:cone")
        armijo_file = _synthetic_sha256(f"step{step}:armijo")
        reference_file = _synthetic_sha256(f"step{step}:reference")
        checkpoint_file = _synthetic_sha256(f"step{step}:checkpoint")
        receipt_sha = _synthetic_sha256(f"step{step}:receipt")

        receipt = {
            "format": r2q.STEP_FORMAT,
            "step": step,
            "parent_state_sha256": parent,
            "model_state_sha256": state,
            "checkpoint_sha256": checkpoint_file,
            "gradient_matrix_sha256": gradient_file,
            "gradient_matrix_semantic_sha256": gradient_semantic,
            "normalized_gram_file_sha256": gram_file,
            "normalized_gram_semantic_sha256": gram_semantic,
            "direction_file_sha256": direction_file,
            "direction_semantic_sha256": direction_semantic,
            "cone_certificate_sha256": cone_file,
            "armijo_receipt_sha256": armijo_file,
            "reference_receipt_sha256": reference_file,
            "update": {
                "base_state_sha256": parent,
                "candidate_state_sha256": state,
            },
        }
        receipt_path = root / "step_receipt.json"
        receipts[receipt_path] = receipt
        hashes[receipt_path] = receipt_sha
        (root / "STEP_ACCEPTED").write_bytes((receipt_sha + "\n").encode("ascii"))

        array_specs = {
            root / "gradient_matrix.npy": (
                (176, 42096),
                gradient_file,
                gradient_semantic,
            ),
            root / "normalized_gram.npy": (
                (176, 176),
                gram_file,
                gram_semantic,
            ),
            root / "direction.npy": (
                (42096,),
                direction_file,
                direction_semantic,
            ),
        }
        for path, (shape, file_sha, semantic_sha) in array_specs.items():
            hashes[path] = file_sha
            arrays[path] = _FakeFrozenArray(shape, semantic_sha)

        file_specs = {
            root / "cone_certificate.json": cone_file,
            root / "armijo_receipt.json": armijo_file,
            root / "reference_receipt.json": reference_file,
            root / "accepted_step.pt": checkpoint_file,
        }
        hashes.update(file_specs)
        checkpoints[root / "accepted_step.pt"] = {
            "format": r2q.STEP_FORMAT,
            "step": step,
            "parent_state_sha256": parent,
            "model_state_sha256": state,
            "model": model,
        }
        chain.append({**receipt, "step_receipt_sha256": receipt_sha})
        parent = state

    endpoint_receipt_path = output / "endpoint_receipt.json"
    endpoint_path = output / "endpoint.pt"
    endpoint_marker = output / "ENDPOINT_FROZEN"
    endpoint_receipt_sha = _synthetic_sha256("endpoint:receipt")
    endpoint_file_sha = _synthetic_sha256("endpoint:file")
    run_contract_sha = _synthetic_sha256("run:contract:file")
    endpoint_receipt = {
        "format": r2q.ENDPOINT_FORMAT,
        "status": "FOUR_STEP_ENDPOINT_FROZEN",
        "run_contract_sha256": run_contract_sha,
        "run_contract_semantic_sha256": _synthetic_sha256("run:contract:semantic"),
        "protocol_sha256": r2q.PROTOCOL_SHA256,
        "contract_doc_sha256": r2q.EXPECTED_CONTRACT_DOC_SHA256,
        "R2Q_source_snapshot_sha256": _synthetic_sha256("r2q:source"),
        "reviewed_helper_source_sha256": {"helper": _synthetic_sha256("helper")},
        "parameter_schema_sha256": _synthetic_sha256("parameter:schema"),
        "runtime_fingerprint": {"device": "synthetic-cpu"},
        "R2P_input_sha256": {"primary": _synthetic_sha256("r2p:primary")},
        "R2O_input_sha256": {"bundle": _synthetic_sha256("r2o:bundle")},
        "endpoint_checkpoint_sha256": endpoint_file_sha,
        "endpoint_model_state_sha256": states[-1],
        "base_model_state_sha256": r2q.EXPECTED_BASE_STATE_SHA256,
        "step_count": 4,
        "step_chain": chain,
    }
    receipts[endpoint_receipt_path] = endpoint_receipt
    hashes[endpoint_receipt_path] = endpoint_receipt_sha
    hashes[endpoint_path] = endpoint_file_sha
    endpoint_marker.write_bytes((endpoint_receipt_sha + "\n").encode("ascii"))
    checkpoints[endpoint_path] = {
        "format": r2q.ENDPOINT_FORMAT,
        "model_state_sha256": states[-1],
        "model": checkpoints[output / "steps" / "step_04" / "accepted_step.pt"][
            "model"
        ],
    }

    def fake_sha256(path: Path) -> str:
        candidate = Path(path)
        if candidate not in hashes:
            raise AssertionError(f"unexpected SHA-256 read: {candidate}")
        return hashes[candidate]

    def fake_json_load(path: Path) -> dict:
        candidate = Path(path)
        if candidate not in receipts:
            raise AssertionError(f"unexpected JSON read: {candidate}")
        return receipts[candidate]

    def fake_numpy_load(path: Path, *, allow_pickle: bool) -> _FakeFrozenArray:
        assert allow_pickle is False
        candidate = Path(path)
        if candidate not in arrays:
            raise AssertionError(f"unexpected NumPy read: {candidate}")
        return arrays[candidate]

    def fake_torch_load(path: Path, *, map_location: str) -> dict:
        assert map_location == "cpu"
        candidate = Path(path)
        checkpoint_loads.append(candidate)
        if candidate not in checkpoints:
            raise AssertionError(f"unexpected checkpoint read: {candidate}")
        return checkpoints[candidate]

    original_isfinite = np.isfinite
    original_array_sha256 = r2q.r2p._array_sha256
    monkeypatch.setattr(r2q, "sha256", fake_sha256)
    monkeypatch.setattr(r2q.r2p, "strict_json_load", fake_json_load)
    monkeypatch.setattr(r2q.np, "load", fake_numpy_load)
    monkeypatch.setattr(
        r2q.np,
        "isfinite",
        lambda value: True
        if isinstance(value, _FakeFrozenArray)
        else original_isfinite(value),
    )
    monkeypatch.setattr(
        r2q.r2p,
        "_array_sha256",
        lambda value, dtype="<f8": value.semantic_sha256
        if isinstance(value, _FakeFrozenArray)
        else original_array_sha256(value, dtype=dtype),
    )
    monkeypatch.setattr(r2q, "torch_load", fake_torch_load)
    return SimpleNamespace(
        output=output,
        hashes=hashes,
        arrays=arrays,
        receipts=receipts,
        checkpoints=checkpoints,
        checkpoint_loads=checkpoint_loads,
        chain=chain,
        states=states,
        endpoint_path=endpoint_path,
        endpoint_receipt_path=endpoint_receipt_path,
        endpoint_marker=endpoint_marker,
        endpoint_receipt_sha=endpoint_receipt_sha,
        endpoint_receipt=endpoint_receipt,
        run_contract_sha=run_contract_sha,
    )


def test_four_step_chain_closes_receipts_markers_files_semantics_and_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic = _install_synthetic_four_step_chain(tmp_path, monkeypatch)
    r2q._verify_step_chain(
        synthetic.output,
        synthetic.chain,
        expected_base_state=r2q.EXPECTED_BASE_STATE_SHA256,
        expected_endpoint_state=synthetic.states[-1],
    )

    step2 = synthetic.output / "steps" / "step_02"
    marker = step2 / "STEP_ACCEPTED"
    original_marker = marker.read_bytes()
    marker.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="step 2 receipt marker/hash changed"):
        r2q._verify_step_chain(
            synthetic.output,
            synthetic.chain,
            expected_base_state=r2q.EXPECTED_BASE_STATE_SHA256,
            expected_endpoint_state=synthetic.states[-1],
        )
    marker.write_bytes(original_marker)

    receipt_path = step2 / "step_receipt.json"
    original_parent = synthetic.receipts[receipt_path]["parent_state_sha256"]
    synthetic.receipts[receipt_path]["parent_state_sha256"] = "tampered"
    with pytest.raises(ValueError, match="step 2 endpoint receipt copy changed"):
        r2q._verify_step_chain(
            synthetic.output,
            synthetic.chain,
            expected_base_state=r2q.EXPECTED_BASE_STATE_SHA256,
            expected_endpoint_state=synthetic.states[-1],
        )
    synthetic.receipts[receipt_path]["parent_state_sha256"] = original_parent

    direction_path = step2 / "direction.npy"
    original_file_sha = synthetic.hashes[direction_path]
    synthetic.hashes[direction_path] = "tampered"
    with pytest.raises(ValueError, match="step 2 direction.npy file hash changed"):
        r2q._verify_step_chain(
            synthetic.output,
            synthetic.chain,
            expected_base_state=r2q.EXPECTED_BASE_STATE_SHA256,
            expected_endpoint_state=synthetic.states[-1],
        )
    synthetic.hashes[direction_path] = original_file_sha

    original_semantic_sha = synthetic.arrays[direction_path].semantic_sha256
    synthetic.arrays[direction_path].semantic_sha256 = "tampered"
    with pytest.raises(RuntimeError, match="step 2 direction.npy changed while loaded"):
        r2q._verify_step_chain(
            synthetic.output,
            synthetic.chain,
            expected_base_state=r2q.EXPECTED_BASE_STATE_SHA256,
            expected_endpoint_state=synthetic.states[-1],
        )
    synthetic.arrays[direction_path].semantic_sha256 = original_semantic_sha

    checkpoint_path = step2 / "accepted_step.pt"
    original_checkpoint_parent = synthetic.checkpoints[checkpoint_path][
        "parent_state_sha256"
    ]
    synthetic.checkpoints[checkpoint_path]["parent_state_sha256"] = "tampered"
    with pytest.raises(ValueError, match="step 2 checkpoint metadata changed"):
        r2q._verify_step_chain(
            synthetic.output,
            synthetic.chain,
            expected_base_state=r2q.EXPECTED_BASE_STATE_SHA256,
            expected_endpoint_state=synthetic.states[-1],
        )
    synthetic.checkpoints[checkpoint_path][
        "parent_state_sha256"
    ] = original_checkpoint_parent


def test_frozen_endpoint_rejects_receipt_or_checkpoint_chain_tamper_before_endpoint_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic = _install_synthetic_four_step_chain(tmp_path, monkeypatch)
    model, receipt = r2q._load_frozen_endpoint(
        synthetic.endpoint_path,
        synthetic.endpoint_receipt_path,
        synthetic.endpoint_marker,
        expected_receipt_hash=synthetic.endpoint_receipt_sha,
        expected_state=synthetic.states[-1],
        expected_run_contract_sha256=synthetic.run_contract_sha,
    )
    assert r2q.state_dict_sha256(model) == synthetic.states[-1]
    assert receipt == synthetic.endpoint_receipt
    assert synthetic.checkpoint_loads[-1] == synthetic.endpoint_path

    step3_receipt = synthetic.output / "steps" / "step_03" / "step_receipt.json"
    original_parent = synthetic.receipts[step3_receipt]["parent_state_sha256"]
    synthetic.receipts[step3_receipt]["parent_state_sha256"] = "tampered"
    synthetic.checkpoint_loads.clear()
    with pytest.raises(ValueError, match="step 3 endpoint receipt copy changed"):
        r2q._load_frozen_endpoint(
            synthetic.endpoint_path,
            synthetic.endpoint_receipt_path,
            synthetic.endpoint_marker,
            expected_receipt_hash=synthetic.endpoint_receipt_sha,
            expected_state=synthetic.states[-1],
            expected_run_contract_sha256=synthetic.run_contract_sha,
        )
    assert synthetic.endpoint_path not in synthetic.checkpoint_loads
    synthetic.receipts[step3_receipt]["parent_state_sha256"] = original_parent

    step3_checkpoint = synthetic.output / "steps" / "step_03" / "accepted_step.pt"
    original_checkpoint_parent = synthetic.checkpoints[step3_checkpoint][
        "parent_state_sha256"
    ]
    synthetic.checkpoints[step3_checkpoint]["parent_state_sha256"] = "tampered"
    synthetic.checkpoint_loads.clear()
    with pytest.raises(ValueError, match="step 3 checkpoint metadata changed"):
        r2q._load_frozen_endpoint(
            synthetic.endpoint_path,
            synthetic.endpoint_receipt_path,
            synthetic.endpoint_marker,
            expected_receipt_hash=synthetic.endpoint_receipt_sha,
            expected_state=synthetic.states[-1],
            expected_run_contract_sha256=synthetic.run_contract_sha,
        )
    assert synthetic.endpoint_path not in synthetic.checkpoint_loads
    synthetic.checkpoints[step3_checkpoint][
        "parent_state_sha256"
    ] = original_checkpoint_parent


def test_endpoint_receipt_builder_contains_complete_reproducibility_closure() -> None:
    tree = ast.parse(Path(r2q.__file__).read_text(encoding="utf-8"))
    endpoint_dicts = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "endpoint_receipt"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    assert len(endpoint_dicts) == 1
    keys = {
        key.value
        for key in endpoint_dicts[0].keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {
        "run_contract_sha256",
        "run_contract_semantic_sha256",
        "protocol_sha256",
        "contract_doc_sha256",
        "R2Q_source_snapshot_sha256",
        "reviewed_helper_source_sha256",
        "parameter_schema_sha256",
        "runtime_fingerprint",
        "R2P_input_sha256",
        "R2O_input_sha256",
        "endpoint_checkpoint_sha256",
        "endpoint_model_state_sha256",
        "base_model_state_sha256",
        "step_chain",
    } <= keys

    loader_source = inspect.getsource(r2q._load_frozen_endpoint)
    assert loader_source.index("_verify_step_chain(") < loader_source.index(
        "torch_load(endpoint_path"
    )


def test_held_numerical_inconclusive_terminal_semantics_are_internally_consistent() -> None:
    source = inspect.getsource(r2q.run_four_step)
    normal_start = source.index(
        'terminal_exit = 2 if held_status == "NUMERICAL_INCONCLUSIVE_HELD" else 0'
    )
    normal_end = source.index("    except TrainNoGo")
    normal = source[normal_start:normal_end]
    assert '_atomic_bytes(output / "EXIT_CODE", f"{terminal_exit}\\n".encode("ascii"))' in normal
    assert 'else "HELD_NUMERICAL_INCONCLUSIVE"' in normal
    assert '_atomic_bytes(output / "DONE", (held_status + "\\n").encode("ascii"))' in normal
    assert normal.index('output / "EXIT_CODE"') < normal.index(
        '"HELD_NUMERICAL_INCONCLUSIVE"'
    ) < normal.index('output / "DONE"')

    numerical_start = source.index("    except NumericalInconclusive")
    numerical_end = source.index("    except BaseException")
    numerical = source[numerical_start:numerical_end]
    assert '"NUMERICAL_INCONCLUSIVE_HELD"' in numerical
    assert 'held_opened = (output / "HELD_EVALUATION_STARTED").exists()' in numerical
    assert '_atomic_bytes(output / "EXIT_CODE", b"2\\n")' in numerical
    assert '"HELD_NUMERICAL_INCONCLUSIVE"' in numerical
    assert '_atomic_bytes(output / "DONE", (payload["status"] + "\\n").encode("ascii"))' in numerical
    assert numerical.index('output / "EXIT_CODE"') < numerical.index(
        '"HELD_NUMERICAL_INCONCLUSIVE"'
    ) < numerical.index('output / "DONE"')
