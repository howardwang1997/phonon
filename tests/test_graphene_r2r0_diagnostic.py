from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/smearing_kink"
sys.path.insert(0, str(SCRIPTS))

import graphene_r2r0_diagnostic as diagnostic  # noqa: E402


def _external_authorization(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "freeze_manifest.json"
    diagnostic.prepare_freeze_manifest(manifest)
    marker = tmp_path / "R2R0D_DIAGNOSTIC_AUTH"
    marker.write_bytes((diagnostic.sha256(manifest) + "\n").encode("ascii"))
    return manifest, marker


def _rewrite_manifest_and_marker(
    manifest: Path, marker: Path, payload: dict
) -> None:
    manifest.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    marker.write_bytes((diagnostic.sha256(manifest) + "\n").encode("ascii"))


def _strict_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str) -> Path:
    recommended = tmp_path / "recommended"
    monkeypatch.setattr(diagnostic, "RECOMMENDED_OUTPUT_ROOT", recommended)
    return recommended / name


def _point_provenance(graph: str = "g", assignment: str = "a") -> dict:
    return {
        "formal_R2O_graph_semantic_sha256": graph,
        "assignment_and_MIC_semantic_sha256": assignment,
    }


def _smooth_fd_points(
    *,
    force_target: float = 0.25,
    jacobian_target: float = -1.5,
    force_error_coefficient: float = 100.0,
    jump_coefficient: float = 0.4,
) -> list[dict]:
    points = []
    for h in diagnostic.FD_STEPS_A:
        derivative = jacobian_target + force_error_coefficient * h**2
        jump = jump_coefficient * h
        left = derivative - 0.5 * jump
        right = derivative + 0.5 * jump
        force_minus = force_target - h * left
        force_plus = force_target + h * right
        energy_derivative = force_target + 3.0 * h**2
        points.append(
            {
                "h_A": h,
                "evaluation_order": ["minus", "plus"],
                "energy_minus_eV": h * energy_derivative,
                "energy_plus_eV": -h * energy_derivative,
                "force_minus_eV_A": force_minus,
                "force_plus_eV_A": force_plus,
                "minus_point_provenance": _point_provenance(),
                "plus_point_provenance": _point_provenance(),
            }
        )
    return points


def _zerojet_public() -> dict:
    return {
        "zero_jet_physical_exact": True,
        "zero_jet_actual": {"q_signature_orbits_sha256": "raw-orbit"},
    }


def _q_pair(delta: float = 2.0e-15) -> tuple[dict, dict, dict, dict]:
    receiver = np.asarray([0, 0, 1, 1], dtype="<i8")
    sender = np.asarray([1, 2, 0, 2], dtype="<i8")
    image = np.zeros((4, 3), dtype="<i8")
    offsets = np.asarray([0, 2, 4], dtype="<i8")
    edge_index = np.arange(4, dtype="<i8")
    distance = np.asarray([1.0, 2.0, 1.0, 2.0], dtype="<f8")
    weight = np.asarray([0.5, 0.25, 0.5, 0.25], dtype="<f8")
    normalization = np.asarray([1.75, 1.75], dtype="<f8")
    center = np.asarray([1.0 / 1.75, 1.0 / 1.75], dtype="<f8")
    neighbor = weight / normalization[receiver]
    cpu = {
        "receiver": receiver,
        "sender": sender,
        "image_integer": image,
        "local_source_order_offsets": offsets,
        "local_source_order_sender": sender.copy(),
        "local_source_order_edge_index": edge_index,
        "reference_distance_A": distance,
        "weight": weight,
        "normalization": normalization,
        "center_q": center,
        "neighbor_q": neighbor,
    }
    gpu = {name: value.copy() for name, value in cpu.items()}
    gpu["center_q"][0] += delta
    gpu["neighbor_q"][1] += delta
    cpu_round = {}
    gpu_round = {}
    for decimals in diagnostic.ROUND_DECIMALS:
        for values, destination in ((cpu, cpu_round), (gpu, gpu_round)):
            payload = np.ascontiguousarray(
                np.round(
                    np.concatenate((values["center_q"], values["neighbor_q"])),
                    decimals=decimals,
                ),
                dtype="<f8",
            )
            values[f"round{decimals}_payload"] = payload
            destination[str(decimals)] = {
                "sha256": hashlib.sha256(payload.tobytes()).hexdigest()
            }
    cpu_public = {
        **_zerojet_public(),
        "atom_count": 2,
        "local_source_order_semantic_sha256": "source-order",
        "round_payload": cpu_round,
    }
    gpu_public = {
        **_zerojet_public(),
        "atom_count": 2,
        "local_source_order_semantic_sha256": "source-order",
        "round_payload": gpu_round,
    }
    return cpu_public, cpu, gpu_public, gpu


def test_canonical_contract_digest_is_frozen() -> None:
    assert (
        diagnostic._semantic_sha256(diagnostic.DIAGNOSTIC_CONTRACT)
        == diagnostic.DIAGNOSTIC_CONTRACT_SHA256
        == "bfe73de26b346527b3c2ac7d17c50b8abf9535417fbcff48520ba03c4946204f"
    )
    scope = diagnostic.DIAGNOSTIC_CONTRACT["scope"]
    assert scope["adjudicates_attempt2"] is False
    assert scope["formal_authorization"] is False
    assert scope["fit_or_training_performed"] is False
    assert scope["labels_used"] is False
    assert scope["GO_marker_creation"] is False


def test_six_step_fd_confirms_second_order_and_richardson() -> None:
    result = diagnostic.analyze_fd_points(
        _smooth_fd_points(),
        force_target=0.25,
        jacobian_target=-1.5,
        base_point_provenance=_point_provenance(),
    )
    assert result["FD_TRUNCATION_CONFIRMED"] is True
    assert result["NONSMOOTH_CANDIDATE"] is False
    assert result["FLOAT_FLOOR_CANDIDATE"] is False
    assert len(result["points"]) == 6
    assert all(
        1.8 <= item["empirical_order_force_error"]["value"] <= 2.2
        for item in result["points"][1:]
    )
    assert all(
        0.8 <= item["empirical_order_slope_jump"]["value"] <= 1.2
        for item in result["points"][1:]
    )
    assert max(
        item["Richardson_force_abs_error_eV_A2"]
        for item in result["points"][-3:]
    ) <= 1.0e-7


def test_energy_path_is_explicit_report_only_for_Hessian_attribution() -> None:
    points = _smooth_fd_points()
    for point in points:
        h = point["h_A"]
        derivative = 0.25 + 1.0e-4
        point["energy_minus_eV"] = h * derivative
        point["energy_plus_eV"] = -h * derivative
    result = diagnostic.analyze_fd_points(
        points,
        force_target=0.25,
        jacobian_target=-1.5,
        base_point_provenance=_point_provenance(),
    )
    assert result["FD_TRUNCATION_CONFIRMED"] is True
    assert result["energy_path_consistent_report_only"] is False
    assert result["checks"][
        "energy_path_consistency_physics_gate_authorized"
    ] is False


def test_fd_zero_error_order_is_explicitly_null() -> None:
    points = _smooth_fd_points(
        force_target=0.0, jacobian_target=0.0, force_error_coefficient=0.0
    )
    result = diagnostic.analyze_fd_points(
        points,
        force_target=0.0,
        jacobian_target=0.0,
        base_point_provenance=_point_provenance(),
    )
    orders = [
        item["empirical_order_force_error"] for item in result["points"][1:]
    ]
    assert all(item["value"] is None for item in orders)
    assert all(item["zero_case"] == "both_zero" for item in orders)
    assert result["FD_TRUNCATION_CONFIRMED"] is False


def test_graph_change_is_nonsmooth_candidate() -> None:
    points = _smooth_fd_points()
    points[-1]["plus_point_provenance"] = _point_provenance(graph="changed")
    result = diagnostic.analyze_fd_points(
        points,
        force_target=0.25,
        jacobian_target=-1.5,
        base_point_provenance=_point_provenance(),
    )
    assert result["FD_TRUNCATION_CONFIRMED"] is False
    assert result["NONSMOOTH_CANDIDATE"] is True
    assert (
        diagnostic.classify_diagnostic(
            result, {"Q_FLOAT_SERIALIZATION_ONLY": False}
        )
        == "NONSMOOTH_CANDIDATE"
    )
    assert (
        diagnostic.classify_diagnostic(
            result, {"Q_FLOAT_SERIALIZATION_ONLY": True}
        )
        == "NONSMOOTH_CANDIDATE"
    )


def test_base_graph_is_part_of_stability_check() -> None:
    result = diagnostic.analyze_fd_points(
        _smooth_fd_points(),
        force_target=0.25,
        jacobian_target=-1.5,
        base_point_provenance=_point_provenance(graph="different-base"),
    )
    assert result["checks"][
        "all_point_graph_and_assignment_topology_stable"
    ] is False
    assert result["NONSMOOTH_CANDIDATE"] is True


def test_smallest_step_plateau_is_float_floor_candidate() -> None:
    points = _smooth_fd_points()
    # Make the finest derivative error equal to the preceding error while
    # retaining unchanged graph/assignment provenance.
    previous_h = diagnostic.FD_STEPS_A[-2]
    desired_derivative = -1.5 + 100.0 * previous_h**2
    h = diagnostic.FD_STEPS_A[-1]
    jump = 0.4 * h
    left = desired_derivative - 0.5 * jump
    right = desired_derivative + 0.5 * jump
    points[-1]["force_minus_eV_A"] = 0.25 - h * left
    points[-1]["force_plus_eV_A"] = 0.25 + h * right
    result = diagnostic.analyze_fd_points(
        points,
        force_target=0.25,
        jacobian_target=-1.5,
        base_point_provenance=_point_provenance(),
    )
    assert result["FD_TRUNCATION_CONFIRMED"] is False
    assert result["NONSMOOTH_CANDIDATE"] is False
    assert result["FLOAT_FLOOR_CANDIDATE"] is True


def test_q_float_serialization_allows_raw_fp64_difference() -> None:
    cpu_public, cpu, gpu_public, gpu = _q_pair()
    gpu_public["zero_jet_actual"]["q_signature_orbits_sha256"] = "different-raw"
    result = diagnostic.compare_q_snapshots(cpu_public, cpu, gpu_public, gpu)
    assert result["topology_and_source_order_exact"] is True
    assert 0.0 < result["normalized_q_max_abs"] <= 5.0e-13
    assert result["round12_and_round13_elementwise_and_hash_equal"] is True
    assert result["Q_FLOAT_SERIALIZATION_ONLY"] is True
    assert result["raw_q_orbit_hash_equal"] is False
    assert result["float_arrays_CPU_vs_CUDA"]["center_q"][
        "exact_different_count"
    ] == 1
    assert result["float_arrays_CPU_vs_CUDA"]["center_q"]["ULP"]["max"] > 0


def test_q_is_divided_on_graph_device_not_recomputed_from_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    sender = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    graph = SimpleNamespace(
        receiver=receiver,
        sender=sender,
        image_integer=torch.zeros((4, 3), dtype=torch.long),
        reference_distance_A=torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float64),
        weight=torch.tensor([0.5, 0.25, 0.5, 0.25], dtype=torch.float64),
        normalization=torch.tensor([1.75, 1.75], dtype=torch.float64),
    )
    source_payload = [
        {"receiver": 0, "sender_order": [0, 1]},
        {"receiver": 1, "sender_order": [0, 1]},
    ]
    zerojet = {
        "production_local_source_order_sha256": diagnostic.r2r.semantic_sha256(
            source_payload
        ),
        "full_local_hessian_performed": False,
        "q_signature_orbits_sha256": "diagnostic-only",
    }
    monkeypatch.setattr(
        diagnostic.r2r, "fixed_reference_neighborhood", lambda *args, **kwargs: graph
    )
    monkeypatch.setattr(
        diagnostic.r2r,
        "background_rank0_zero_jet_audit",
        lambda *args, **kwargs: zerojet,
    )

    original = diagnostic._little_endian_array

    class NoHostDivide(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            if ufunc is np.divide:
                raise AssertionError("normalized q was recomputed on the host")
            plain = tuple(
                np.asarray(value) if isinstance(value, NoHostDivide) else value
                for value in inputs
            )
            return getattr(ufunc, method)(*plain, **kwargs)

    def guarded_copy(value: torch.Tensor, dtype: str) -> np.ndarray:
        result = original(value, dtype)
        return result.view(NoHostDivide) if np.dtype(dtype).kind == "f" else result

    monkeypatch.setattr(diagnostic, "_little_endian_array", guarded_copy)
    _, arrays = diagnostic.capture_q_snapshot(
        Atoms("CC"), "reference_2", "cpu", full_local_hessian=False
    )
    np.testing.assert_array_equal(
        np.asarray(arrays["center_q"]),
        np.asarray((1.0 / graph.normalization).numpy(), dtype="<f8"),
    )
    np.testing.assert_array_equal(
        np.asarray(arrays["neighbor_q"]),
        np.asarray(
            (graph.weight / graph.normalization[graph.receiver]).numpy(), dtype="<f8"
        ),
    )


def test_q_discrete_mismatch_stops_fail_closed() -> None:
    cpu_public, cpu, gpu_public, gpu = _q_pair()
    gpu["sender"][0] = 9
    with pytest.raises(diagnostic.DiagnosticStop, match="topology") as caught:
        diagnostic.compare_q_snapshots(cpu_public, cpu, gpu_public, gpu)
    assert caught.value.status == "INCONCLUSIVE"


def test_zerojet_requires_nonzero_local_to_production_parity_pass() -> None:
    summary = {
        "multipolar_gate_value_max_abs": 0.0,
        "global_all_node_Jacobian_max_abs_A-1": 0.0,
        "global_all_node_value_and_Jacobian_exact_zero": True,
        "local_all_node_value_max_abs": 0.0,
        "local_all_node_Jacobian_max_abs_A-1": 0.0,
        "local_all_node_Hessian_max_abs_A-2": 0.0,
        "local_all_node_finite": True,
        "all_nodes_covered_exactly_once": True,
        "all_local_senders_unique": True,
        "full_local_hessian_performed": True,
        "pass": False,
    }
    assert diagnostic._zerojet_physical_exact(summary) is False
    summary["pass"] = True
    assert diagnostic._zerojet_physical_exact(summary) is True


def test_npz_receipt_key_and_raw_hash_are_fail_closed() -> None:
    value = np.arange(6, dtype="<f8").reshape(2, 3)
    schema = diagnostic._array_schema({"force": value})
    receipt = diagnostic._array_receipt("force", value, "force")
    audit = diagnostic._validate_npz_references({"array": receipt}, schema)
    assert audit["reference_count"] == 1
    assert audit["all_references_resolve_exactly"] is True
    tampered = {"array": {**receipt, "raw_sha256": "0" * 64}}
    with pytest.raises(diagnostic.DiagnosticStop, match="NPZ binding"):
        diagnostic._validate_npz_references(tampered, schema)
    missing = {"array": {**receipt, "npz_key": "missing"}}
    with pytest.raises(diagnostic.DiagnosticStop, match="NPZ key is absent"):
        diagnostic._validate_npz_references(missing, schema)
    schema_with_unreferenced = diagnostic._array_schema(
        {"force": value, "unreferenced": value + 1.0}
    )
    with pytest.raises(diagnostic.DiagnosticStop, match="missing=.*unreferenced"):
        diagnostic._validate_npz_references({"array": receipt}, schema_with_unreferenced)
    with pytest.raises(diagnostic.DiagnosticStop, match="duplicates=.*force"):
        diagnostic._validate_npz_references(
            {"first": receipt, "second": receipt}, schema
        )


def test_combined_classification_never_uses_formal_pass() -> None:
    statuses = {
        diagnostic.classify_diagnostic(
            {"FD_TRUNCATION_CONFIRMED": True},
            {"Q_FLOAT_SERIALIZATION_ONLY": True},
        ),
        diagnostic.classify_diagnostic(
            {"FD_TRUNCATION_CONFIRMED": True},
            {"Q_FLOAT_SERIALIZATION_ONLY": False},
        ),
        diagnostic.classify_diagnostic(
            {"FD_TRUNCATION_CONFIRMED": False},
            {"Q_FLOAT_SERIALIZATION_ONLY": True},
        ),
    }
    assert statuses == {
        "ATTRIBUTED_BOTH",
        "DIAGNOSTIC_TRUNCATION",
        "NUMERICAL_IDENTITY",
    }
    assert all("PASS" not in status for status in statuses)


def test_output_must_be_strict_recommended_descendant() -> None:
    with pytest.raises(ValueError, match="strict raw-path descendant"):
        diagnostic._fresh_output(diagnostic.ATTEMPT2_ROOT / "r2r0d")
    with pytest.raises(ValueError, match="strict raw-path descendant"):
        diagnostic._fresh_output(diagnostic.DATA / "r2r0d")
    with pytest.raises(ValueError, match="strict raw-path descendant"):
        diagnostic._fresh_output(diagnostic.RECOMMENDED_OUTPUT_ROOT)


def test_attempt2_DONE_content_is_not_an_optional_marker(tmp_path: Path) -> None:
    (tmp_path / "EXIT_CODE").write_bytes(b"0\n")
    (tmp_path / "DONE").write_bytes(
        (
            diagnostic.ATTEMPT2_SCIENTIFIC_STATUS
            + "\n"
            + diagnostic.EXPECTED_LAUNCH_RECEIPT_SHA256
            + "\n"
        ).encode("ascii")
    )
    receipt = diagnostic._validate_attempt2_terminal(tmp_path)
    assert set(receipt) == {"DONE", "EXIT_CODE"}
    (tmp_path / "DONE").write_bytes(b"tampered\n")
    with pytest.raises(diagnostic.DiagnosticStop, match="DONE binding"):
        diagnostic._validate_attempt2_terminal(tmp_path)


def test_cpu_request_to_full_runner_is_failed_terminal_without_GO(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, marker = _external_authorization(tmp_path)
    output = _strict_output(monkeypatch, tmp_path, "cpu_forbidden")
    receipt = diagnostic.run_diagnostic(
        output,
        device="cpu",
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    assert receipt["status"] == "INCONCLUSIVE"
    assert receipt["adjudicates_attempt2"] is False
    assert receipt["formal_authorization"] is False
    assert diagnostic._terminal_set(output) == {"FAILED"}
    assert (output / "EXIT_CODE").read_bytes() == b"2\n"
    expected_failed = (
        receipt["status"]
        + "\n"
        + diagnostic.sha256(output / "diagnostic_receipt.json")
        + "\n"
    ).encode("ascii")
    assert (output / "FAILED").read_bytes() == expected_failed
    assert not any("GO" in path.name for path in output.iterdir())


def test_requested_cuda_index_controls_sync_and_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostic.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(diagnostic.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(diagnostic.torch.cuda, "current_device", lambda: 1)
    assert diagnostic._normalize_cuda_device("cuda:2") == ("cuda:2", 2)
    assert diagnostic._normalize_cuda_device("cuda") == ("cuda:1", 1)
    observed = []
    monkeypatch.setattr(
        diagnostic.torch.cuda, "synchronize", lambda index: observed.append(index)
    )
    diagnostic._synchronize_cuda(2)
    assert observed == [2]


def test_full_runtime_requires_exact_allowlisted_v100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostic.torch.cuda, "get_device_name", lambda index: "NVIDIA RTX 2060"
    )
    with pytest.raises(diagnostic.DiagnosticStop, match="allowlisted NVIDIA V100"):
        diagnostic._validate_v100_runtime(0)
    monkeypatch.setattr(
        diagnostic.torch.cuda,
        "get_device_name",
        lambda index: diagnostic.ALLOWED_V100_DEVICE_NAMES[0],
    )
    assert (
        diagnostic._validate_v100_runtime(0)
        == diagnostic.ALLOWED_V100_DEVICE_NAMES[0]
    )


def test_float32_origin_is_mandatory() -> None:
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(diagnostic.DiagnosticStop, match="originate as float32"):
            diagnostic._validate_float32_origin()
        torch.set_default_dtype(torch.float32)
        diagnostic._validate_float32_origin()
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_budget_boundary_is_inclusive_and_epsilon_over_stops(kind: str) -> None:
    budget = diagnostic.Budget()
    limit = (
        diagnostic.CPU_BUDGET_SECONDS
        if kind == "cpu"
        else diagnostic.GPU_BUDGET_SECONDS
    )
    budget.add(kind, limit)
    with pytest.raises(
        diagnostic.DiagnosticStop, match=f"{kind.upper()} wall budget exceeded"
    ):
        budget.add(kind, np.finfo(float).eps * limit)


def test_exception_path_reconciles_total_wall_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = diagnostic.Budget()
    monkeypatch.setattr(
        diagnostic.time,
        "perf_counter",
        lambda: diagnostic.CPU_BUDGET_SECONDS + 1.0,
    )
    result = diagnostic._reconcile_exception_budget(budget, 0.0, ValueError("x"))
    assert isinstance(result, diagnostic.DiagnosticStop)
    assert result.status == "DIAGNOSTIC_BUDGET_EXCEEDED"
    assert budget.cpu_seconds == diagnostic.CPU_BUDGET_SECONDS + 1.0


def test_label_contamination_stops_before_calculation() -> None:
    contaminated = Atoms("C")
    contaminated.arrays["forces"] = np.zeros((1, 3))
    with pytest.raises(diagnostic.DiagnosticStop, match="label entered Atoms"):
        diagnostic._validate_label_free_atoms(contaminated, "contaminated")


def test_output_rejects_holdout_dotdot_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recommended = tmp_path / "recommended"
    recommended.mkdir()
    monkeypatch.setattr(diagnostic, "RECOMMENDED_OUTPUT_ROOT", recommended)
    with pytest.raises(ValueError, match="forbidden data token"):
        diagnostic._validated_output_path(recommended / "holdout_probe")
    with pytest.raises(ValueError, match="traversal"):
        diagnostic._validated_output_path(recommended / "safe" / ".." / "escape")
    outside = tmp_path / "outside"
    outside.mkdir()
    (recommended / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        diagnostic._validated_output_path(recommended / "redirect" / "run")


@pytest.mark.parametrize("role", sorted(diagnostic.DIAGNOSTIC_SOURCE_ROLES))
def test_manifest_rejects_each_single_source_role_hash_tamper(
    tmp_path: Path, role: str
) -> None:
    manifest, marker = _external_authorization(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["diagnostic_source_sha256"][role] = "0" * 64
    _rewrite_manifest_and_marker(manifest, marker, payload)
    with pytest.raises(ValueError, match="manifest binding changed"):
        diagnostic.validate_execution_authorization(manifest, marker)


@pytest.mark.parametrize("role", sorted(diagnostic.DIAGNOSTIC_SOURCE_ROLES))
def test_live_source_missing_is_fail_closed_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    manifest, marker = _external_authorization(tmp_path)
    monkeypatch.setitem(
        diagnostic.DIAGNOSTIC_SOURCE_PATHS, role, tmp_path / f"missing_{role}"
    )
    with pytest.raises(diagnostic.DiagnosticStop, match="missing or symlinked"):
        diagnostic.validate_execution_authorization(manifest, marker)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_manifest_rejects_missing_or_extra_source_role(
    tmp_path: Path, change: str
) -> None:
    manifest, marker = _external_authorization(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if change == "missing":
        payload["diagnostic_source_sha256"].pop("diagnostic_tests")
    else:
        payload["diagnostic_source_sha256"]["extra_role"] = "0" * 64
    _rewrite_manifest_and_marker(manifest, marker, payload)
    with pytest.raises(ValueError, match="source role inventory changed"):
        diagnostic.validate_execution_authorization(manifest, marker)


@pytest.mark.parametrize("binding", ["v5", "input", "artifact"])
def test_manifest_rejects_single_v5_input_or_attempt2_artifact_tamper(
    tmp_path: Path, binding: str
) -> None:
    manifest, marker = _external_authorization(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if binding == "v5":
        payload["v5_binding"]["manifest_sha256"] = "1" * 64
    elif binding == "input":
        first = sorted(payload["v5_binding"]["input_sha256"])[0]
        payload["v5_binding"]["input_sha256"][first] = "2" * 64
    else:
        payload["attempt2_binding"]["artifact_sha256"][
            "aggregate_receipt"
        ] = "3" * 64
    _rewrite_manifest_and_marker(manifest, marker, payload)
    with pytest.raises(ValueError, match="manifest binding changed"):
        diagnostic.validate_execution_authorization(manifest, marker)


def test_manifest_marker_missing_tampered_and_manifest_tampered(
    tmp_path: Path,
) -> None:
    manifest, marker = _external_authorization(tmp_path)
    diagnostic.validate_execution_authorization(manifest, marker)
    with pytest.raises(FileNotFoundError):
        diagnostic.validate_execution_authorization(
            tmp_path / "missing_manifest.json", marker
        )
    missing = tmp_path / "missing_auth"
    with pytest.raises(FileNotFoundError):
        diagnostic.validate_execution_authorization(manifest, missing)
    marker.write_bytes(b"0" * 64 + b"\n")
    with pytest.raises(ValueError, match="does not bind"):
        diagnostic.validate_execution_authorization(manifest, marker)
    marker.write_bytes((diagnostic.sha256(manifest) + "\n").encode("ascii"))
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="does not bind"):
        diagnostic.validate_execution_authorization(manifest, marker)


def test_prepare_manifest_never_creates_authorization_or_GO(tmp_path: Path) -> None:
    manifest = tmp_path / "candidate" / "freeze_manifest.json"
    payload = diagnostic.prepare_freeze_manifest(manifest)
    assert payload["authorization_marker_created"] is False
    assert set(payload["diagnostic_source_sha256"]) == (
        diagnostic.DIAGNOSTIC_SOURCE_ROLES
    )
    assert [path.name for path in manifest.parent.iterdir()] == ["freeze_manifest.json"]
    assert not any("GO" in path.name for path in manifest.parent.iterdir())


def test_authorization_is_validated_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, marker = _external_authorization(tmp_path)
    marker.write_bytes(b"0" * 64 + b"\n")
    output = _strict_output(monkeypatch, tmp_path, "must_not_exist")
    with pytest.raises(ValueError, match="does not bind"):
        diagnostic.run_cpu_preflight(
            output,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert not output.exists()


def test_completed_preflight_recovers_without_recompute_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, marker = _external_authorization(tmp_path)
    output = _strict_output(monkeypatch, tmp_path, "preflight_complete")
    first = diagnostic.run_cpu_preflight(
        output,
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    assert first["status"] == "DIAGNOSTIC_PREFLIGHT_ONLY"
    assert diagnostic._terminal_set(output) == {"DONE"}
    assert (output / "EXIT_CODE").read_bytes() == b"0\n"
    monkeypatch.setattr(
        diagnostic,
        "capture_q_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("scientific recompute on DONE recovery")
        ),
    )
    recovered = diagnostic.run_cpu_preflight(
        output,
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    assert recovered["completion_recovered_without_recompute"] is True

    tampered = output.parent / "preflight_tampered"
    shutil.copytree(output, tampered)
    with (tampered / "preflight_arrays.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact hash/schema manifest"):
        diagnostic.run_cpu_preflight(
            tampered,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )

    done_tampered = output.parent / "preflight_DONE_tampered"
    shutil.copytree(output, done_tampered)
    (done_tampered / "DONE").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="DONE does not bind"):
        diagnostic.run_cpu_preflight(
            done_tampered,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )


@pytest.mark.parametrize("terminal", ["RUNNING", "FAILED"])
def test_partial_or_failed_output_is_never_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: str
) -> None:
    manifest, marker = _external_authorization(tmp_path)
    output = _strict_output(monkeypatch, tmp_path, f"not_recoverable_{terminal}")
    output.mkdir(parents=True)
    (output / terminal).write_text("partial\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="recovery is forbidden"):
        diagnostic.run_cpu_preflight(
            output,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )


def test_geometry_loader_retains_no_force_or_energy_labels() -> None:
    thermal, reference6, reference8, audit = diagnostic.load_label_free_geometry()
    assert len(thermal) == 92
    assert len(reference6) == 72
    assert len(reference8) == 128
    assert audit["header_label_values_retained"] is False
    assert audit["numeric_label_columns_converted"] is False
    for item in audit.values():
        if isinstance(item, dict) and "force_or_energy_labels_present" in item:
            assert item["force_or_energy_labels_present"] is False


def test_frozen_attempt2_provenance_and_fail_status_are_exact() -> None:
    receipt = diagnostic.validate_frozen_provenance()
    assert (
        receipt["attempt2_scientific_status_observed"]
        == diagnostic.ATTEMPT2_SCIENTIFIC_STATUS
    )
    assert receipt["attempt2_status_preserved"] is True
    assert receipt["original_extxyz_force_or_energy_columns_converted"] is False
    assert receipt["attempt2_artifact_sha256"] == (
        diagnostic.EXPECTED_ATTEMPT2_ARTIFACT_SHA256
    )
