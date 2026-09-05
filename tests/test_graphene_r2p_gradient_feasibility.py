from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "smearing_kink"
sys.path.insert(0, str(SCRIPTS))

import graphene_r2p_gradient_feasibility as r2p  # noqa: E402


def _write_synthetic_failed_terminal_receipt(root: Path) -> Path:
    root.mkdir()
    (root / "EXIT_CODE").write_bytes(b"0\n")
    (root / "DONE").write_bytes(r2p.EXPECTED_FORMAL_DONE_BYTES)
    for name in ("PRETRAIN_DONE", "TRAINING_DONE", "CORE_GATE_FAILED"):
        (root / name).write_bytes(b"")
    (root / "COMPLETED_AT").write_bytes(b"2026-08-25T07:17:58+0800\n")
    gate = root / "core_checkpoint_gate.json"
    gate.write_bytes(b'{"status":"synthetic"}\n')
    return gate


def test_protocol_freezes_exact_primary_counts_and_report_only_smooth_bound() -> None:
    assert r2p.PRIMARY_FAMILY_COUNTS == {
        "Aprime_seed0": 20,
        "thermal_total_force": 92,
        "small_zero_RMS": 32,
        "small_zero_exact_top": 32,
    }
    assert sum(r2p.PRIMARY_FAMILY_COUNTS.values()) == 176
    assert r2p.PROTOCOL["primary_block_count"] == 176
    assert r2p.PROTOCOL["smooth_Linf_role"].startswith("report_only")
    assert r2p.PROTOCOL["smooth_Linf_ncomp_small"] == 384
    assert r2p.PROTOCOL["smooth_Linf_max_underestimate_bound_eV_A"] == pytest.approx(
        0.001 * np.log(384)
    )
    assert r2p.SMALL_CORRECTION_MARGIN_EV_A == pytest.approx(0.00358418)
    assert r2p.PROTOCOL["exact_top_tie_tolerance_eV_A"] == 0.0
    assert r2p.PROTOCOL["exact_top_tie_policy"].startswith("fail_closed")
    assert r2p.PROTOCOL["formal_graph_source_semantics"] == {
        **r2p.FORMAL_GRAPH_SOURCE_SEMANTICS
    }
    assert (
        r2p.FORMAL_GRAPH_SOURCE_SEMANTICS_SHA256
        == r2p.canonical_json_sha256(r2p.FORMAL_GRAPH_SOURCE_SEMANTICS)
    )
    assert r2p.PROTOCOL_SHA256 == r2p.canonical_json_sha256(r2p.PROTOCOL)


def test_formal_graph_semantics_require_float32_default_without_mutation(
    tmp_path: Path,
) -> None:
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        r2p.assert_formal_graph_source_semantics()
        assert torch.get_default_dtype() == torch.float32
        torch.set_default_dtype(torch.float64)
        with pytest.raises(RuntimeError, match="AtomicData graph"):
            r2p.assert_formal_graph_source_semantics()
        with pytest.raises(RuntimeError, match="AtomicData graph"):
            r2p.run_formal_audit(
                data_root=tmp_path / "not_opened_data",
                formal_run_root=tmp_path / "not_opened_run",
                output_root=tmp_path / "must_not_be_created",
                device="cpu",
            )
        assert torch.get_default_dtype() == torch.float64
        assert not (tmp_path / "must_not_be_created").exists()
    finally:
        torch.set_default_dtype(original)


def test_contract_hash_binds_formal_graph_semantics() -> None:
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    loaded = SimpleNamespace(
        model=model,
        input_sha256={"train": "a" * 64},
        wrapper_sha256="b" * 64,
        data_manifest={
            "outputs": {
                "valid_e50_seed1.xyz": {"sha256": "c" * 64},
                "harmonic_lambda1_small_gate.xyz": {"sha256": "d" * 64},
            }
        },
        formal_terminal_receipt={"status": "synthetic_test"},
    )
    payload = r2p._contract_payload(
        loaded,
        device=torch.device("cpu"),
        source_snapshot_relative="snapshot.py",
        source_snapshot_sha256="e" * 64,
    )
    assert payload["formal_graph_source_semantics"] == (
        r2p.FORMAL_GRAPH_SOURCE_SEMANTICS
    )
    assert payload["formal_graph_source_semantics_sha256"] == (
        r2p.FORMAL_GRAPH_SOURCE_SEMANTICS_SHA256
    )
    assert payload["runtime"]["default_dtype"] == "torch.float32"


def test_common_descent_orthogonal_certificate_has_exact_primal_dual_and_kkt() -> None:
    gradients = np.eye(2, dtype=np.float64)
    certificate = r2p.solve_max_min_common_descent(gradients)
    expected = 1.0 / np.sqrt(2.0)
    assert certificate.scientific_status == "GO"
    assert certificate.common_descent is True
    assert certificate.primal_min_cosine == pytest.approx(expected, abs=1.0e-12)
    assert certificate.dual_norm == pytest.approx(expected, abs=1.0e-12)
    assert certificate.cosine_duality_gap <= 1.0e-12
    assert certificate.kkt["maximum_violation"] <= 1.0e-12
    assert certificate.kkt["simplex_sum_residual"] <= 1.0e-12
    assert certificate.gram_audit["raw_diagonal_unit_max_abs_error"] <= 1.0e-12
    assert certificate.weights == pytest.approx([0.5, 0.5], abs=1.0e-12)


def test_opposite_gradients_are_reliably_no_go_by_dual_upper_bound() -> None:
    gradients = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    certificate = r2p.solve_max_min_common_descent(gradients)
    assert certificate.scientific_status == "NO_GO"
    assert certificate.common_descent is False
    assert certificate.dual_norm <= r2p.CONE_GO_TOL
    assert certificate.primal_min_cosine == pytest.approx(0.0, abs=1.0e-12)
    assert certificate.weights == pytest.approx([0.5, 0.5], abs=1.0e-12)


def test_per_configuration_cone_detects_conflict_hidden_by_aggregate() -> None:
    gradients = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float64
    )
    certificate = r2p.solve_max_min_common_descent(gradients)
    assert certificate.scientific_status == "NO_GO"
    assert certificate.dual_norm <= 1.0e-9


def test_solver_handles_the_formal_176_row_shape() -> None:
    gradients = np.eye(4, dtype=np.float64)[np.arange(176) % 4]
    certificate = r2p.solve_max_min_common_descent(gradients)
    assert certificate.scientific_status == "GO"
    assert certificate.primal_min_cosine == pytest.approx(0.5, abs=1.0e-10)
    assert certificate.dual_norm == pytest.approx(0.5, abs=1.0e-10)
    assert certificate.cosine_duality_gap <= 1.0e-10
    assert certificate.kkt["maximum_violation"] <= 1.0e-10


def test_solver_is_deterministic_and_crosscheck_is_recorded() -> None:
    gradients = np.asarray(
        [[1.0, 0.2, -0.1], [0.4, 1.0, 0.3], [0.2, 0.5, 1.0]],
        dtype=np.float64,
    )
    first = r2p.solve_max_min_common_descent(gradients)
    second = r2p.solve_max_min_common_descent(gradients)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.direction, second.direction)
    assert first.crosscheck == second.crosscheck
    assert first.crosscheck["SLSQP_feasible"] is True
    assert first.crosscheck["selected_solution"] == first.solution_source


def test_solver_rejects_zero_nonfinite_or_malformed_gradient_rows() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        r2p.solve_max_min_common_descent(np.asarray([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        r2p.solve_max_min_common_descent(np.asarray([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="2D"):
        r2p.solve_max_min_common_descent(np.asarray([1.0, 2.0]))


def test_exact_top_unique_branch_uses_live_selected_component() -> None:
    force = torch.tensor(
        [[0.001, -0.004, 0.002], [0.003, 0.0005, -0.001]],
        dtype=torch.float64,
        requires_grad=True,
    )
    objective, metadata = r2p._exact_top_objective(
        force, scale_eV_A=0.01, aggregate_coefficient=0.5
    )
    gradient = torch.autograd.grad(objective, force)[0]
    assert metadata["top_flat_index"] == 1
    assert metadata["top_atom_source_order_0based"] == 0
    assert metadata["top_cartesian_component"] == "y"
    assert metadata["top_signed_eV_A"] == pytest.approx(-0.004)
    assert metadata["top_sign"] == -1
    assert metadata["second_flat_index"] == 3
    assert metadata["second_atom_source_order_0based"] == 1
    assert metadata["second_cartesian_component"] == "x"
    assert metadata["second_signed_eV_A"] == pytest.approx(0.003)
    assert metadata["second_sign"] == 1
    assert metadata["exact_active_count"] == 1
    expected = torch.zeros_like(force)
    expected[0, 1] = 0.5 * 2.0 * (-0.004) / 0.01**2
    assert torch.equal(gradient, expected)


def test_exact_top_tie_fails_closed_instead_of_using_one_subgradient() -> None:
    force = torch.tensor(
        [[0.004, -0.004, 0.001]], dtype=torch.float64, requires_grad=True
    )
    with pytest.raises(ValueError, match="nonsmooth exact tie"):
        r2p._exact_top_objective(
            force, scale_eV_A=0.01, aggregate_coefficient=1.0
        )


def test_exact_top_near_tie_fails_closed() -> None:
    force = torch.tensor(
        [[0.004, -(0.004 - 5.0e-9), 0.001]],
        dtype=torch.float64,
        requires_grad=True,
    )
    with pytest.raises(ValueError, match="unstable near tie"):
        r2p._exact_top_objective(
            force, scale_eV_A=0.01, aggregate_coefficient=1.0
        )


def test_exact_top_difference_inside_old_active_tolerance_also_fails() -> None:
    force = torch.tensor(
        [[0.004, -(0.004 - 5.0e-11), 0.001]],
        dtype=torch.float64,
        requires_grad=True,
    )
    with pytest.raises(ValueError, match="unstable near tie"):
        r2p._exact_top_objective(
            force, scale_eV_A=0.01, aggregate_coefficient=1.0
        )


def test_shifted_log_mean_exp_is_explicitly_not_an_exact_max_bound() -> None:
    force = torch.zeros((128, 3), dtype=torch.float64)
    force.reshape(-1)[0] = 0.01
    smooth = float(r2p._smooth_log_mean_exp(force))
    assert smooth < 0.01
    assert 0.01 - smooth <= r2p.SMOOTH_TAU_EV_A * np.log(384) + 1.0e-12


def test_autograd_helper_returns_fp64_vector_without_populating_dot_grad() -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    state = copy.deepcopy(model.state_dict())
    x = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    objective = model(x).square().sum()
    vector = r2p._objective_gradient(
        objective, list(model.parameters()), retain_graph=False
    )
    assert vector.dtype == np.float64
    assert vector.shape == (2,)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(state[key], model.state_dict()[key]) for key in state)


def test_certificate_payload_records_every_block_norm_cosine_and_hash() -> None:
    gradients = np.eye(2, dtype=np.float64)
    certificate = r2p.solve_max_min_common_descent(gradients)
    records = [
        {"block_id": "a", "family": "x", "objective": 1.0},
        {"block_id": "b", "family": "y", "objective": 2.0},
    ]
    payload = r2p._certificate_payload(
        certificate, records, gradients, authorization_role="test"
    )
    assert payload["block_count"] == 2
    assert payload["can_authorize_parameter_update_or_training"] is False
    assert payload["can_authorize_postcore_or_tail"] is False
    assert payload["solver"]["scientific_status"] == "GO"
    assert payload["solver"]["KKT"]["maximum_violation"] <= 1.0e-12
    assert len(payload["per_block"]) == 2
    assert all("gradient_norm" in item for item in payload["per_block"])
    assert all("common_direction_cosine" in item for item in payload["per_block"])
    assert payload["gradient_matrix_sha256"] == r2p._array_sha256(gradients)


def test_formal_gate_prediction_parity_is_strict_and_fail_closed() -> None:
    force = {
        "RMSE_meV_A": 1.0,
        "MAE_meV_A": 0.8,
        "max_abs_meV_A": 2.0,
        "n_force_components": 6,
    }
    seed1 = {"force_error": dict(force), "Aprime_RMS_meV_A": 3.0}
    actual = {
        "force_error": dict(force),
        "per_configuration_force": [dict(force), dict(force)],
    }
    choice = {
        "E50_seed1": {
            "force_error": dict(force),
            "Aprime_force_error_RMS_meV_A": 3.0,
        },
        "harmonic_lambda1_small_gate": {
            "force_error": dict(force),
            "per_configuration_force": [dict(force), dict(force)],
        },
    }
    payload = r2p.validate_formal_gate_prediction_parity(seed1, actual, choice)
    assert payload["pass"] is True
    assert payload["maximum_absolute_numeric_difference"] == 0.0
    mismatched = copy.deepcopy(seed1)
    mismatched["force_error"]["RMSE_meV_A"] += 2.0e-7
    with pytest.raises(ValueError, match="does not reproduce"):
        r2p.validate_formal_gate_prediction_parity(mismatched, actual, choice)


def test_forbidden_paths_are_rejected_before_existence_or_open_checks(
    tmp_path: Path,
) -> None:
    trap = tmp_path / "support_trap" / "does_not_exist.json"
    with pytest.raises(ValueError, match="forbidden unopened data"):
        r2p.reject_forbidden_path(trap, "trap", must_exist=True)
    seed2 = tmp_path / "seed2_gate.xyz"
    with pytest.raises(ValueError, match="forbidden unopened data"):
        r2p.reject_forbidden_path(seed2, "seed2", must_exist=False)


def test_formal_terminal_receipt_is_hash_bound_and_marker_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "formal"
    gate = _write_synthetic_failed_terminal_receipt(root)
    with pytest.raises(ValueError, match="gate SHA-256 differs"):
        r2p.validate_formal_failed_run_terminal_receipt(root)
    monkeypatch.setattr(r2p, "EXPECTED_FORMAL_GATE_SHA256", r2p.sha256(gate))
    receipt = r2p.validate_formal_failed_run_terminal_receipt(root)
    assert receipt["gate_sha256"] == r2p.sha256(gate)
    assert receipt["gate_marker_arm"] == "CORE_GATE_FAILED"
    (root / "CORE_GATE_PASSED").write_bytes(b"")
    with pytest.raises(ValueError, match="incompatible terminal markers"):
        r2p.validate_formal_failed_run_terminal_receipt(root)


def test_formal_terminal_receipt_precedes_any_training_artifact_open(
    tmp_path: Path,
) -> None:
    data = tmp_path / "empty_data"
    formal = tmp_path / "unfinished_formal"
    data.mkdir()
    formal.mkdir()
    (formal / "RUNNING").write_bytes(b"")
    with pytest.raises(ValueError, match="incompatible terminal markers"):
        r2p.load_training_inputs(data, formal, device=torch.device("cpu"))


def test_symlink_and_traversal_paths_fail_closed(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        r2p.reject_forbidden_path(link, "linked", must_exist=True)
    with pytest.raises(ValueError, match="traversal"):
        r2p.reject_forbidden_path(
            tmp_path / "real" / ".." / "real", "traversal", must_exist=True
        )


def test_report_only_loader_rejects_wrong_primary_hash_before_report_open(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.json"
    r2p.strict_json(
        primary,
        {
            "format": r2p.PRIMARY_FORMAT,
            "authorization_role": "primary_hard_exact_top",
        },
    )
    loaded = SimpleNamespace(data_root=tmp_path, data_manifest={"outputs": {}})
    with pytest.raises(ValueError, match="changed before report-only open"):
        r2p.load_report_only_structures(
            loaded,
            primary_certificate_path=primary,
            expected_primary_certificate_sha256="0" * 64,
        )


def test_strict_json_refuses_nonfinite_and_writes_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "value.json"
    with pytest.raises(ValueError):
        r2p.strict_json(destination, {"bad": float("nan")})
    assert not destination.exists()
    r2p.strict_json(destination, {"finite": 1.0})
    assert json.loads(destination.read_text()) == {"finite": 1.0}


def test_source_has_no_optimizer_backward_or_parameter_assignment_path() -> None:
    source_path = Path(r2p.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden_attributes = {
        node.func.attr
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "backward",
            "step",
            "set_default_dtype",
            "requires_grad_",
            "load_state_dict",
            "copy_",
        }
    }
    assert forbidden_attributes == set()
    assert "torch.optim" not in source_path.read_text(encoding="utf-8")
