from __future__ import annotations

import ast
import copy
import errno
import hashlib
import io
import inspect
import json
import os
import pickle
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/smearing_kink"
sys.path.insert(0, str(SCRIPTS))

import graphene_r2r1_linear_readout as linear  # noqa: E402
import aggregate_graphene_r2r1_linear_readout as aggregate  # noqa: E402
import graphene_r2r1_frozen_readout as frozen  # noqa: E402
import run_graphene_r2r1_linear_readout as run_cli  # noqa: E402
import launch_graphene_r2r1_linear_readout as launcher  # noqa: E402


ATTEMPT3 = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/formal_r2r0_three_host_seed83_attempt3"
)


def _rewrite_same_bytes_and_restore_mtime(descriptor: int) -> None:
    """Change ctime while restoring the exact payload and mtime."""
    before = os.fstat(descriptor)
    raw = os.pread(descriptor, before.st_size, 0)
    os.ftruncate(descriptor, 0)
    offset = 0
    while offset < len(raw):
        written = os.pwrite(descriptor, raw[offset:], offset)
        assert written > 0
        offset += written
    os.fsync(descriptor)
    os.utime(
        descriptor,
        ns=(before.st_atime_ns, before.st_mtime_ns),
    )
    after = os.fstat(descriptor)
    assert os.pread(descriptor, after.st_size, 0) == raw
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns


def _rewrite_different_same_size_and_restore_mtime(descriptor: int) -> None:
    """Change bytes while preserving inode, size, and mtime."""
    before = os.fstat(descriptor)
    raw = bytearray(os.pread(descriptor, before.st_size, 0))
    assert raw
    raw[-1] ^= 1
    os.pwrite(descriptor, raw, 0)
    os.fsync(descriptor)
    os.utime(descriptor, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.fstat(descriptor)
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns


def _diagnostic_path_xattrs(path: Path) -> str:
    """Return a non-authoritative exact diagnostic digest for forensic tests."""
    if sys.platform != "darwin":
        return hashlib.sha256(b"").hexdigest()
    completed = subprocess.run(
        ["/usr/bin/xattr", "-l", "-x", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _patch_production_paths(
    monkeypatch, tmp_path: Path
) -> tuple[Path, Path, Path, Path]:
    if linear.RECOMMENDED_THERMAL92.exists():
        monkeypatch.setattr(
            linear,
            "THERMAL92_FILE_SIZE_BYTES",
            os.stat(linear.RECOMMENDED_THERMAL92).st_size,
        )
    control = tmp_path / "control_attempt2_retry1"
    output = tmp_path / "r2r1_release_attempt2_retry1"
    candidate = tmp_path / "candidate_attempt2_retry1"
    release_manifest = candidate / "release_manifest.json"
    monkeypatch.setattr(linear, "RECOMMENDED_CONTROL_ROOT", control)
    monkeypatch.setattr(
        linear, "RECOMMENDED_FREEZE_MANIFEST", control / "freeze_manifest.json"
    )
    monkeypatch.setattr(
        linear, "RECOMMENDED_AUTHORIZATION_MARKER", control / "R2R1_FORMAL_GO"
    )
    monkeypatch.setattr(linear, "RECOMMENDED_FIT_OUTPUT_ROOT", output)
    monkeypatch.setattr(linear, "RECOMMENDED_CANDIDATE_ROOT", candidate)
    monkeypatch.setattr(linear, "RECOMMENDED_RELEASE_MANIFEST", release_manifest)
    return control, output, candidate, release_manifest


def _authorization_stub() -> dict:
    return {
        "fit_authorized": True,
        "thermal92_force_labels_authorized": True,
        "energy_labels_authorized": False,
        "development_or_held_access_authorized": False,
        "freeze_manifest_sha256": "a" * 64,
        "authorization_marker_sha256": "b" * 64,
    }


def _real_test_authorization(tmp_path: Path, monkeypatch):
    safe_root = tmp_path.parent / (
        "r2r1_auth_" + hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()[:12]
    )
    safe_root.mkdir()
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, safe_root
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    return linear._issue_production_fit_authorization(manifest, marker)


def _real_authorization_files(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "freeze_manifest.json"
    marker = tmp_path / "R2R1_FORMAL_GO"
    manifest.write_bytes(linear.canonical_json_bytes(linear.freeze_manifest_payload()))
    marker.write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    return manifest, marker


def _synthetic_label_raw_hashes(atom_count: int = 2) -> dict[str, str]:
    reference = np.empty((92, atom_count, 3), dtype="<f8")
    for index in range(92):
        for atom in range(atom_count):
            reference[index, atom] = [
                float(f"{0.001 * (index + 1):.16g}"),
                float(f"{-0.002 * (atom + 1):.16g}"),
                float(f"{0.003:.16g}"),
            ]
    mode_real = np.zeros((20, atom_count, 3), dtype="<f8")
    mode_real[:, 0, 0] = 1.0
    mode_imag = np.zeros_like(mode_real)
    mode_imag[:, 1, 1] = 1.0
    coordinates = np.column_stack(
        (
            [float(f"{0.01 + index * 1.0e-4:.12f}") for index in range(20)],
            [float(f"{-0.002 + index * 2.0e-5:.12f}") for index in range(20)],
        )
    ).astype("<f8")
    foundation = np.zeros((20, atom_count, 3), dtype="<f8")
    foundation[:, :, 0] = -0.1
    q6 = np.zeros_like(foundation)
    q6[:, :, 0] = -0.02
    return {
        "REF_forces_all92": linear.raw_array_sha256(reference, "<f8"),
        "REF_forces_E50": linear.raw_array_sha256(reference[:20], "<f8"),
        "REF_forces_T300": linear.raw_array_sha256(reference[20:56], "<f8"),
        "REF_forces_T600": linear.raw_array_sha256(reference[56:92], "<f8"),
        "APRIME_mode_real": linear.raw_array_sha256(mode_real, "<f8"),
        "APRIME_mode_imag": linear.raw_array_sha256(mode_imag, "<f8"),
        "APRIME_coordinates": linear.raw_array_sha256(coordinates, "<f8"),
        "FOUNDATION_BASE_forces": linear.raw_array_sha256(foundation, "<f8"),
        "FROZEN_Q6_forces": linear.raw_array_sha256(q6, "<f8"),
    }


def _property_text(schema: tuple[tuple[str, str, int], ...]) -> str:
    return ":".join(
        value
        for name, kind, width in schema
        for value in (name, kind, str(width))
    )


def _synthetic_extxyz(path: Path, atom_count: int = 2) -> None:
    lines: list[str] = []
    for index in range(92):
        group = linear.group_for_global_index(index)
        schema = (
            linear.E50_PROPERTY_SCHEMA
            if group == "E50_seed0"
            else linear.AUXILIARY_PROPERTY_SCHEMA
        )
        header = [
            'Lattice="10 0 0 0 10 0 0 0 10"',
            f"Properties={_property_text(schema)}",
            f"config_type={linear.THERMAL_CONFIG_TYPES[group]}",
            'pbc="T T T"',
            "REF_energy=definitely_not_a_number",
            "SECRET_HELD_LABEL=must_be_discarded",
        ]
        if group == "E50_seed0":
            header.extend(
                [
                    f"APRIME_coordinate_real_A={0.01 + index * 1.0e-4:.12f}",
                    f"APRIME_coordinate_imag_A={-0.002 + index * 2.0e-5:.12f}",
                ]
            )
        lines.extend([str(atom_count), " ".join(header)])
        for atom in range(atom_count):
            row: list[str] = []
            for name, kind, width in schema:
                if kind == "S":
                    row.append("C")
                    continue
                if name == "pos":
                    values = [index * 1.0e-3, atom * 1.0e-2, 5.0]
                elif name == "REF_forces":
                    values = [0.001 * (index + 1), -0.002 * (atom + 1), 0.003]
                elif name == "APRIME_mode_real":
                    values = [1.0 if atom == 0 else 0.0, 0.0, 0.0]
                elif name == "APRIME_mode_imag":
                    values = [0.0, 1.0 if atom == 1 else 0.0, 0.0]
                elif name == "FOUNDATION_BASE_forces":
                    values = [-0.1, 0.0, 0.0]
                elif name == "FROZEN_Q6_forces":
                    values = [-0.02, 0.0, 0.0]
                else:
                    # If the parser accidentally converts an unlisted column,
                    # this deliberately invalid token will raise.
                    row.extend(["opaque_not_numeric"] * width)
                    continue
                row.extend(f"{value:.16g}" for value in values)
            lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _synthetic_problem(seed: int = 83, atom_count: int = 2):
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(92, atom_count, 3, 65))
    fixed = rng.normal(scale=0.005, size=(92, atom_count, 3))
    coefficient = rng.normal(scale=0.003, size=65)
    target = fixed + np.einsum("natk,k->nat", design, coefficient, optimize=False)
    mode_real = np.zeros((20, atom_count, 3))
    mode_real[:, 0, 0] = 1.0
    mode_imag = np.zeros_like(mode_real)
    mode_imag[:, 0, 1] = 1.0
    coordinates = np.column_stack(
        (np.linspace(0.01, 0.03, 20), np.linspace(-0.004, 0.004, 20))
    )
    foundation = np.zeros((20, atom_count, 3))
    q6 = np.zeros_like(foundation)
    complex_coordinate = coordinates[:, 0] + 1.0j * coordinates[:, 1]
    # Make the target total-force projection exactly -coord so k_target=1.
    foundation[:, 0, 0] = -complex_coordinate.real - target[:20, 0, 0]
    foundation[:, 0, 1] = complex_coordinate.imag - target[:20, 0, 1]
    labels = linear.ThermalLabels(
        reference_force_eV_A=target,
        aprime=linear.AprimeData(
            mode_real=mode_real,
            mode_imag=mode_imag,
            coordinates=coordinates,
            foundation_base_force_eV_A=foundation,
            frozen_q6_force_eV_A=q6,
        ),
        structure_identity_sha256="c" * 64,
        raw_sha256={},
        parser_receipt={},
    )
    return design, fixed, coefficient, labels


def test_frozen_fold_partition_counts_and_hashes() -> None:
    receipt = linear.validate_fold_contract()
    assert receipt["partition_exact"] is True
    assert receipt["fold_contract_sha256"] == linear.FOLD_CONTRACT_SHA256
    assert tuple(receipt["fold_index_sha256"]) == linear.FOLD_GLOBAL_INDEX_SHA256
    assert receipt["per_fold_counts"] == [
        {"E50_seed0": 5, "T300": 9, "T600": 9}
    ] * 4


def test_alpha_weight_scaler_and_gate_constants_are_exact() -> None:
    assert linear.ALPHA_GRID == tuple(float(10.0**value) for value in range(-10, 3))
    assert linear.GROUP_MASSES == {
        "E50_seed0": 0.50,
        "T300": 0.25,
        "T600": 0.25,
    }


def test_retry1_paths_and_v2_schema_constants_are_exact() -> None:
    assert linear.RECOMMENDED_FIT_OUTPUT_ROOT.name == (
        "R2R1_conditional_linear_readout_fit_seed83_attempt2_retry1_20260826"
    )
    assert linear.RECOMMENDED_CONTROL_ROOT.name == (
        "R2R1_conditional_linear_readout_control_attempt2_retry1_20260826"
    )
    assert linear.RECOMMENDED_CANDIDATE_ROOT.name == (
        "R2R1_conditional_linear_readout_candidate_attempt2_retry1_20260826"
    )
    assert linear.AGGREGATE_FORMAT == "graphene_r2r1_nested_oof_aggregate_v2"
    assert aggregate.FORMAT == "graphene_r2r1_local_fit_release_v2"
    assert aggregate.FIT_RECEIPT_FORMAT == "graphene_r2r1_frozen_fit_receipt_v2"
    assert aggregate.COMPLETION_FORMAT == "graphene_r2r1_local_completion_v2"
    assert aggregate.RELEASE_MANIFEST_FORMAT == (
        "graphene_r2r1_external_release_manifest_v2"
    )
    assert aggregate.TERMINAL_LEDGER_FORMAT == (
        "graphene_r2r1_terminal_identity_ledger_v2"
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra_ctime", "bool", "float", "hardlink", "directory", "negative"],
)
def test_owned_regular_identity_schema_is_exact_eight_keys(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "owned.bin"
    path.write_bytes(b"owned\n")
    identity = linear._owned_regular_identity(os.stat(path))
    assert set(identity) == aggregate.OWNED_REGULAR_IDENTITY_KEYS
    changed = dict(identity)
    if mutation == "missing":
        changed.pop("st_uid")
    elif mutation == "extra_ctime":
        changed["st_ctime_ns"] = os.stat(path).st_ctime_ns
    elif mutation == "bool":
        changed["st_uid"] = True
    elif mutation == "float":
        changed["st_mtime_ns"] = float(changed["st_mtime_ns"])
    elif mutation == "hardlink":
        changed["st_nlink"] = 2
    elif mutation == "directory":
        changed["st_mode"] = stat.S_IFDIR | 0o700
    elif mutation == "negative":
        changed["st_size"] = -1
    with pytest.raises(ValueError):
        aggregate._validate_owned_regular_identity(changed, "test owned identity")
    assert linear.FORCE_SCALE_EV_A == 0.030
    assert linear.GATE_THRESHOLDS == {
        "force_RMSE_meV_A": 30.0,
        "force_max_abs_meV_A": 200.0,
        "E50_Aprime_RMS_meV_A": 15.0,
        "E50_slope_relative_abs_error": 0.05,
    }


def test_prior_attempt_roots_are_lexical_protection_only_zero_io(
    tmp_path: Path, monkeypatch
) -> None:
    forensic_parent = tmp_path / "forensic"
    attempt1_fit = forensic_parent / "fit_attempt1"
    attempt1_control = forensic_parent / "control_attempt1"
    attempt1_candidate = forensic_parent / "candidate_attempt1"
    failed_attempt2_fit = forensic_parent / "fit_attempt2"
    failed_attempt2_control = forensic_parent / "control_attempt2"
    failed_attempt2_candidate = forensic_parent / "candidate_attempt2"
    for root, basename, raw in (
        (attempt1_fit, "RUNNING", b"forensic-running\n"),
        (attempt1_control, "freeze_manifest.json", b'{"forensic":true}\n'),
        (attempt1_candidate, "release_manifest.json", b"forensic-anchor\n"),
        (failed_attempt2_fit, "RUNNING", b"failed-attempt2-running\n"),
        (
            failed_attempt2_control,
            "forensic_empty_control_sentinel",
            b"failed-attempt2-control\n",
        ),
        (
            failed_attempt2_candidate,
            "release_manifest.json",
            b"failed-attempt2-anchor\n",
        ),
    ):
        root.mkdir(parents=True, exist_ok=True)
        (root / basename).write_bytes(raw)

    monkeypatch.setattr(linear, "ATTEMPT1_FIT_OUTPUT_ROOT", attempt1_fit)
    monkeypatch.setattr(linear, "ATTEMPT1_CONTROL_ROOT", attempt1_control)
    monkeypatch.setattr(
        linear, "ATTEMPT1_FREEZE_MANIFEST", attempt1_control / "freeze_manifest.json"
    )
    monkeypatch.setattr(
        linear, "ATTEMPT1_AUTHORIZATION_MARKER", attempt1_control / "R2R1_FORMAL_GO"
    )
    monkeypatch.setattr(linear, "ATTEMPT1_CANDIDATE_ROOT", attempt1_candidate)
    monkeypatch.setattr(
        linear,
        "ATTEMPT1_RELEASE_MANIFEST",
        attempt1_candidate / "release_manifest.json",
    )
    monkeypatch.setattr(linear, "FAILED_ATTEMPT2_FIT_OUTPUT_ROOT", failed_attempt2_fit)
    monkeypatch.setattr(linear, "FAILED_ATTEMPT2_CONTROL_ROOT", failed_attempt2_control)
    monkeypatch.setattr(
        linear,
        "FAILED_ATTEMPT2_FREEZE_MANIFEST",
        failed_attempt2_control / "freeze_manifest.json",
    )
    monkeypatch.setattr(
        linear,
        "FAILED_ATTEMPT2_AUTHORIZATION_MARKER",
        failed_attempt2_control / "R2R1_FORMAL_GO",
    )
    monkeypatch.setattr(
        linear, "FAILED_ATTEMPT2_CANDIDATE_ROOT", failed_attempt2_candidate
    )
    monkeypatch.setattr(
        linear,
        "FAILED_ATTEMPT2_RELEASE_MANIFEST",
        failed_attempt2_candidate / "release_manifest.json",
    )
    current_parent = tmp_path / "current"
    control, output, _candidate, release_manifest = _patch_production_paths(
        monkeypatch, current_parent
    )

    def forensic_snapshot() -> dict[str, dict[str, object]]:
        snapshot: dict[str, dict[str, object]] = {}
        for root in (
            attempt1_fit,
            attempt1_control,
            attempt1_candidate,
            failed_attempt2_fit,
            failed_attempt2_control,
            failed_attempt2_candidate,
        ):
            for path in (root, *sorted(root.rglob("*"))):
                observed = os.stat(path, follow_symlinks=False)
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                if stat.S_ISDIR(observed.st_mode):
                    flags |= getattr(os, "O_DIRECTORY", 0)
                descriptor = os.open(path, flags)
                try:
                    item: dict[str, object] = {
                        "identity": linear._stat_identity(os.fstat(descriptor)),
                        "xattrs_sha256": _diagnostic_path_xattrs(path),
                    }
                    if stat.S_ISREG(observed.st_mode):
                        item["raw"] = os.pread(descriptor, observed.st_size, 0)
                    snapshot[str(path)] = item
                finally:
                    os.close(descriptor)
        return snapshot

    before = forensic_snapshot()
    protected = tuple(
        Path(os.path.abspath(str(path)))
        for path in (
            attempt1_fit,
            attempt1_control,
            attempt1_candidate,
            failed_attempt2_fit,
            failed_attempt2_control,
            failed_attempt2_candidate,
        )
    )
    accesses: list[tuple[str, str]] = []

    def is_protected(value) -> bool:
        if isinstance(value, int):
            return False
        try:
            lexical = Path(os.path.abspath(os.fsdecode(value)))
        except TypeError:
            return False
        return any(lexical == root or root in lexical.parents for root in protected)

    originals = {
        name: getattr(os, name)
        for name in (
            "stat",
            "open",
            "listdir",
            "scandir",
            "mkdir",
            "unlink",
            "remove",
            "rename",
            "replace",
            "link",
            "symlink",
        )
    }

    def guarded(name):
        original = originals[name]

        def invoke(*args, **kwargs):
            if any(is_protected(value) for value in args[:2]):
                accesses.append((name, "|".join(map(str, args[:2]))))
                raise AssertionError(f"prior-attempt forensic path reached {name}")
            return original(*args, **kwargs)

        return invoke

    class FakeRelease(dict):
        anchor_witness = object()

    status = "R2R1_CONDITIONAL_OOF_FAILED"
    with monkeypatch.context() as guarded_patch:
        for name in originals:
            guarded_patch.setattr(os, name, guarded(name))
        partition = linear._validate_attempt2_path_partition()
        assert partition["attempt1_fit"] == str(attempt1_fit)
        assert partition["failed_attempt2_control"] == str(failed_attempt2_control)
        aggregate._validate_production_path_contract(
            output_root=output,
            attempt3_root=linear.ATTEMPT3_ROOT,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=control / "freeze_manifest.json",
            authorization_marker=control / "R2R1_FORMAL_GO",
            release_manifest=release_manifest,
            input_existence_required=False,
        )
        guarded_patch.setattr(
            aggregate, "inspect_release_state", lambda _root: {"state": "ABSENT"}
        )
        guarded_patch.setattr(
            aggregate,
            "_inspect_fresh_publication_pair",
            lambda _root: {"both_absent": True},
        )
        guarded_patch.setattr(
            linear,
            "label_blind_preflight",
            lambda *_args: {"status": "R2R1_LABEL_BLIND_PREFLIGHT_PASSED"},
        )
        fake_release = FakeRelease(
            status=status,
            mechanics_status="R2R1_MECHANICS_PENDING",
        )
        guarded_patch.setattr(
            aggregate, "materialize_authorized_fit", lambda **_kwargs: fake_release
        )

        def fake_anchor(**kwargs):
            assert kwargs["materialization_witness"] is fake_release.anchor_witness
            return {
                "release_status": status,
                "mechanics_status": "R2R1_MECHANICS_PENDING",
            }

        guarded_patch.setattr(aggregate, "anchor_completed_release", fake_anchor)
        launched = launcher.execute(
            output_root=output,
            attempt3_root=linear.ATTEMPT3_ROOT,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=control / "freeze_manifest.json",
            authorization_marker=control / "R2R1_FORMAL_GO",
            release_manifest=release_manifest,
        )
        assert launched["status"] == status

    assert accesses == []
    assert forensic_snapshot() == before


def test_real_attempt3_completed_recovery_and_design_only_expectations(
    monkeypatch,
) -> None:
    def forbidden_remote(*args, **kwargs):
        pytest.fail("R2R-0 DONE recovery reached remote or staging code")

    monkeypatch.setattr(linear.r2r0_launcher, "_stage_and_verify", forbidden_remote)
    monkeypatch.setattr(linear.r2r0_launcher, "_run_logged", forbidden_remote)
    monkeypatch.setattr(linear.r2r0_launcher.subprocess, "Popen", forbidden_remote)
    recovery, arrays = linear.validate_attempt3_completed_recovery(ATTEMPT3)
    assert recovery["R2R0_launcher_terminal_pure_read_verified"] is True
    assert recovery["R2R0_aggregate_terminal_pure_read_verified"] is True
    assert recovery["R2R0_launcher_completion_recovered_without_recompute"] is True
    assert recovery["R2R0_aggregate_completion_recovered_without_recompute"] is True
    assert recovery["remote_or_staging_code_triggered"] is False
    assert recovery["completed_tree_hash_mtime_identity_unchanged"] is True
    assert recovery["three_shard_arrays_bit_exact_to_aggregate"] is True
    audit = linear.all_split_design_audits(
        arrays["thermal_parameter_force_design_eV_A"]
    )
    assert audit["pass"] is True
    assert audit["frozen_design_only_expectation"]["matches"] is True
    observed_outer = [
        audit["outer_train69"][str(index)]["condition"] for index in range(4)
    ]
    assert np.allclose(
        observed_outer,
        linear.EXPECTED_OUTER_TRAIN69_CONDITION,
        rtol=2.0e-10,
        atol=0.0,
    )


def test_private_attempt3_legacy_full9_transient_gets_one_owned_retry(
    monkeypatch,
) -> None:
    original_validate = linear.r2r0.validate_execution_authorization
    calls = {"count": 0}

    def fail_once_then_validate(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("synthetic private-copy ctime-only legacy failure")
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        linear.r2r0, "validate_execution_authorization", fail_once_then_validate
    )
    recovery, _arrays = linear.validate_attempt3_completed_recovery(ATTEMPT3)
    # One failed outer authorization plus the successful retry's outer and
    # aggregate-internal authorization calls.
    assert calls["count"] == 3
    assert recovery["R2R0_launcher_terminal_pure_read_verified"] is True


def test_private_attempt3_content_change_never_reaches_legacy_retry(
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def tamper_private_copy_then_fail(manifest, marker):
        del marker
        calls["count"] += 1
        copy_root = Path(manifest).parent.parent
        descriptor = os.open(
            copy_root / "launch_receipt.json",
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            _rewrite_different_same_size_and_restore_mtime(descriptor)
        finally:
            os.close(descriptor)
        raise ValueError("synthetic legacy failure after private content change")

    monkeypatch.setattr(
        linear.r2r0, "validate_execution_authorization", tamper_private_copy_then_fail
    )
    with pytest.raises(ValueError, match="synthetic legacy failure"):
        linear.validate_attempt3_completed_recovery(ATTEMPT3)
    assert calls["count"] == 1


def test_label_blind_preflight_never_calls_parser_or_svd(monkeypatch) -> None:
    monkeypatch.setattr(
        linear,
        "_load_thermal92_force_labels_streaming_authorized",
        lambda *args, **kwargs: pytest.fail("label parser called by preflight"),
    )
    monkeypatch.setattr(
        linear,
        "_fit_weighted_ridge",
        lambda *args, **kwargs: pytest.fail("fit called by label-blind preflight"),
    )
    result = linear.label_blind_preflight()
    assert result["status"] == "R2R1_LABEL_BLIND_PREFLIGHT_PASSED"
    assert result["thermal_force_labels_opened"] is False
    assert result["fit_performed"] is False
    assert result["thermal92_file_access"] == "not_opened_before_external_GO"


def test_rejects_live_and_broken_symlink_before_resolve(tmp_path: Path) -> None:
    target = tmp_path / "plain.xyz"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link.xyz"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        linear._reject_path(link, "test")
    broken = tmp_path / "broken.xyz"
    broken.symlink_to(tmp_path / "missing.xyz")
    with pytest.raises(ValueError, match="symlink"):
        linear._reject_path(broken, "test")


def test_same_fd_reader_rejects_hardlink_alias(tmp_path: Path) -> None:
    original = tmp_path / "plain.xyz"
    alias = tmp_path / "alias.xyz"
    original.write_text("x", encoding="utf-8")
    os.link(original, alias)
    with pytest.raises(ValueError, match="exactly one hard link"):
        linear._read_regular_file_once(
            original, "test hardlink", size_limit=1024
        )


def test_label_path_denylist_includes_holdout(tmp_path: Path) -> None:
    path = tmp_path / "future_holdout.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="holdout"):
        linear._reject_path(
            path,
            "labels",
            forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
        )


@pytest.mark.parametrize(
    "basename",
    [
        "seed_1.xyz",
        "seed-2.xyz",
        "seed01.xyz",
        "seed0_1.xyz",
        "s.e.e.d.0.2.xyz",
        "small.xyz",
    ],
)
def test_label_path_denylist_normalizes_forbidden_variants(
    tmp_path: Path, basename: str
) -> None:
    path = tmp_path / basename
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden path token"):
        linear._reject_path(
            path,
            "labels",
            forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
        )


def test_header_parser_discards_every_unlisted_value() -> None:
    fields = linear._header_fields(
        'Properties=species:S:1 Lattice="1 0 0 0 1 0 0 0 1" '
        'pbc="T T T" config_type=x APRIME_coordinate_real_A=0.1 '
        'REF_energy=123.4 SECRET_FORCE_LABEL=999'
    )
    assert set(fields) == {
        "Properties",
        "Lattice",
        "pbc",
        "config_type",
        "APRIME_coordinate_real_A",
    }


def test_streaming_whitelist_keeps_opaque_columns_unconverted(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "thermal_train.xyz"
    _synthetic_extxyz(path, atom_count=72)
    monkeypatch.setattr(linear, "RECOMMENDED_THERMAL92", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(linear, "THERMAL92_FILE_SHA256", digest)
    monkeypatch.setattr(
        linear, "EXPECTED_LABEL_RAW_SHA256", _synthetic_label_raw_hashes(72)
    )
    monkeypatch.setattr(linear.r2r0, "structure_semantic_sha256", lambda atoms: "s")
    identity = linear.r2r.semantic_sha256([(index, "s") for index in range(92)])
    monkeypatch.setattr(linear, "THERMAL_STRUCTURE_IDENTITY_SHA256", identity)
    authorization = _real_test_authorization(tmp_path, monkeypatch)
    labels = linear._load_thermal92_force_labels_streaming_authorized(
        path,
        authorization=authorization,
    )
    assert labels.reference_force_eV_A.shape == (92, 72, 3)
    assert labels.aprime.mode_real.shape == (20, 72, 3)
    assert labels.aprime.coordinates.shape == (20, 2)
    assert labels.parser_receipt["header_label_values_retained"] is False
    assert labels.parser_receipt["same_fd_hash_and_parse"] is True
    assert labels.parser_receipt["streaming_binary_readline"] is True
    assert labels.parser_receipt["full_file_bytes_buffered"] is False
    assert labels.parser_receipt["file_sha256"] == digest
    assert labels.parser_receipt["energy_header_values_converted"] is False
    assert labels.parser_receipt["unlisted_numeric_columns_converted"] is False
    assert labels.parser_receipt["temperature_or_smearing_used_as_feature"] is False


def test_wrong_whole_file_hash_precedes_header_float_and_svd(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "thermal_train.xyz"
    _synthetic_extxyz(path)
    monkeypatch.setattr(linear, "RECOMMENDED_THERMAL92", path)
    calls = {"header": 0, "float": 0, "svd": 0}

    def forbidden(kind: str):
        def fail(*args, **kwargs):
            calls[kind] += 1
            raise AssertionError(f"{kind} must not run before whole-file SHA passes")

        return fail

    monkeypatch.setattr(linear, "THERMAL92_FILE_SHA256", "0" * 64)
    authorization = _real_test_authorization(tmp_path, monkeypatch)
    monkeypatch.setattr(linear, "_header_fields", forbidden("header"))
    monkeypatch.setattr(linear, "_parse_whitelisted_float", forbidden("float"))
    monkeypatch.setattr(linear.np.linalg, "svd", forbidden("svd"))
    with pytest.raises(ValueError, match="before label parsing"):
        linear._load_thermal92_force_labels_streaming_authorized(
            path,
            authorization=authorization,
        )
    assert calls == {"header": 0, "float": 0, "svd": 0}


def test_authorized_parser_rejects_fake_capability_before_float_or_open(
    tmp_path: Path, monkeypatch
) -> None:
    missing = tmp_path / "thermal_train.xyz"
    float_calls = 0
    open_calls = 0

    def forbidden_float(value):
        nonlocal float_calls
        float_calls += 1
        raise AssertionError("float conversion must not run")

    def forbidden_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        raise AssertionError("thermal file must not open")

    monkeypatch.setattr(linear, "_parse_whitelisted_float", forbidden_float)
    monkeypatch.setattr(linear, "_open_regular_file_fd", forbidden_open)
    for authorization in (None, {}, _authorization_stub()):
        with pytest.raises(PermissionError, match="authorization"):
            linear._load_thermal92_force_labels_streaming_authorized(
                missing, authorization=authorization
            )
    assert float_calls == 0
    assert open_calls == 0


def test_streaming_parser_rejects_oversized_line_at_bounded_readline(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "thermal_train.xyz"
    path.write_bytes(b"x" * (4 * 1024 * 1024 + 2))
    monkeypatch.setattr(linear, "RECOMMENDED_THERMAL92", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(linear, "THERMAL92_FILE_SHA256", digest)
    authorization = _real_test_authorization(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="oversized line"):
        linear._load_thermal92_force_labels_streaming_authorized(
            path,
            authorization=authorization,
        )


def test_production_loader_has_no_hash_or_geometry_override() -> None:
    parameters = inspect.signature(
        linear.load_thermal92_force_labels_streaming
    ).parameters
    assert set(parameters) == {
        "path",
        "freeze_manifest",
        "authorization_marker",
        "publication_boundary",
    }


def test_missing_or_tampered_marker_precedes_parser_and_svd(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    payload = linear.freeze_manifest_payload()
    manifest = control / "freeze_manifest.json"
    manifest.write_bytes(linear.canonical_json_bytes(payload))
    marker = control / "R2R1_FORMAL_GO"
    parser_calls = 0
    svd_calls = 0

    def parser(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        raise AssertionError("parser must not run")

    def svd(*args, **kwargs):
        nonlocal svd_calls
        svd_calls += 1
        raise AssertionError("SVD must not run")

    monkeypatch.setattr(
        linear, "_load_thermal92_force_labels_streaming_authorized", parser
    )
    monkeypatch.setattr(linear.np.linalg, "svd", svd)
    with pytest.raises((FileNotFoundError, PermissionError)):
        linear.authorized_fit_pipeline(
            attempt3_root=ATTEMPT3,
            thermal92_path=ROOT
            / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    marker.write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(PermissionError, match="marker bytes"):
        linear.authorized_fit_pipeline(
            attempt3_root=ATTEMPT3,
            thermal92_path=ROOT
            / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert parser_calls == 0
    assert svd_calls == 0


def test_weighted_svd_recovers_linear_coefficients_without_centering(
    tmp_path: Path, monkeypatch
) -> None:
    design, fixed, expected, labels = _synthetic_problem()
    authorization = _real_test_authorization(tmp_path, monkeypatch)
    fit = linear._fit_weighted_ridge(
        design,
        fixed,
        labels.reference_force_eV_A,
        np.arange(92),
        1.0e-10,
        authorization=authorization,
    )
    expected_scale = np.sqrt(np.mean(design.reshape(-1, 65) ** 2, axis=0))
    assert np.allclose(fit.scale_eV_A, expected_scale, rtol=1.0e-14, atol=0.0)
    assert np.max(np.abs(fit.physical_coefficient - expected)) < 2.0e-9
    assert fit.receipt["target_centering"] is False
    assert fit.receipt["fit_intercept"] is False
    assert abs(fit.receipt["weight_sum"] - 1.0) <= 2.0e-15


def test_aprime_complex_convention_and_target_slope_threshold() -> None:
    design, fixed, expected, labels = _synthetic_problem()
    prediction = linear.predict_force(design, fixed, expected, np.arange(92))
    metrics = linear.gate_metrics(
        prediction,
        labels.reference_force_eV_A,
        labels.aprime,
        np.arange(92),
    )
    assert metrics["E50_Aprime"]["RMS_meV_A"] < 1.0e-10
    assert metrics["E50_Aprime"]["target_restoring_slope_eV_A2"] == pytest.approx(1.0)
    assert abs(metrics["E50_Aprime"]["slope_relative_error"]) < 1.0e-12
    zero_base = np.zeros_like(labels.aprime.foundation_base_force_eV_A)
    zero_target = labels.reference_force_eV_A.copy()
    zero_target[:20] = 0.0
    zero_aprime = linear.AprimeData(
        mode_real=labels.aprime.mode_real,
        mode_imag=labels.aprime.mode_imag,
        coordinates=labels.aprime.coordinates,
        foundation_base_force_eV_A=zero_base,
        frozen_q6_force_eV_A=zero_base,
    )
    with pytest.raises(ValueError, match="<=1e-14"):
        linear.gate_metrics(zero_target, zero_target, zero_aprime, np.arange(92))


def test_nested_outer_holdout_never_enters_its_scaler_or_fit(
    tmp_path: Path, monkeypatch
) -> None:
    design, fixed, _expected, labels = _synthetic_problem()
    authorization = _real_test_authorization(tmp_path, monkeypatch)
    original = linear._fit_weighted_ridge
    calls: list[tuple[int, ...]] = []

    def recording(*args, **kwargs):
        indices = args[3] if len(args) > 3 else kwargs["train_indices"]
        calls.append(tuple(int(value) for value in indices))
        return original(*args, **kwargs)

    monkeypatch.setattr(linear, "_fit_weighted_ridge", recording)
    receipt, prediction = linear._nested_conditional_oof(
        design, fixed, labels, authorization=authorization
    )
    assert receipt["pass"] is True
    assert np.all(np.isfinite(prediction))
    # The final call in each outer block is its train69 refit; verify exact
    # exclusion directly from the recorded receipt as well.
    for record in receipt["outer_records"]:
        hold = set(record["outer_hold_global_indices"])
        train = set(record["outer_train_global_indices"])
        assert hold.isdisjoint(train)
        assert len(hold) == 23 and len(train) == 69
    assert calls


def test_private_production_solve_rejects_unsealed_values_before_svd(
    monkeypatch,
) -> None:
    design, fixed, _expected, labels = _synthetic_problem()
    split_audits = linear.all_split_design_audits(design)
    svd_calls = 0

    def forbidden_svd(*args, **kwargs):
        nonlocal svd_calls
        svd_calls += 1
        raise AssertionError("SVD must not run without a sealed capability")

    monkeypatch.setattr(linear.np.linalg, "svd", forbidden_svd)
    for value in (None, {}, _authorization_stub()):
        with pytest.raises(PermissionError, match="authorization"):
            linear._fit_pipeline_from_loaded_labels(
                design,
                fixed,
                labels,
                authorization=value,
                prevalidated_split_audits=split_audits,
            )
    assert svd_calls == 0


def test_round12_score_tie_selects_larger_alpha() -> None:
    candidates = [
        {"alpha": 1.0e-4, "metrics": {"selection_score_rounded_12": 0.5}},
        {"alpha": 1.0e-2, "metrics": {"selection_score_rounded_12": 0.5}},
    ]
    assert min(candidates, key=linear._candidate_order_key)["alpha"] == 1.0e-2


def test_contract_declares_no_energy_or_development_access() -> None:
    assert linear.CONTRACT_PAYLOAD["data"]["energy_labels_used"] is False
    assert (
        linear.CONTRACT_PAYLOAD["data"]["temperature_or_smearing_is_feature"]
        is False
    )
    assert (
        linear.CONTRACT_PAYLOAD["safety"]["seed1_seed2_small_support_held_access"]
        is False
    )
    assert linear.CONTRACT_PAYLOAD["readout"]["fit_intercept"] is False


def _mock_pipeline_result(status: str) -> tuple[dict, dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {
        "OOF_predicted_force_eV_A": np.zeros((92, 72, 3), dtype="<f8")
    }
    receipt = {
        "format": linear.AGGREGATE_FORMAT,
        "status": status,
        "numerically_inconclusive": False,
        "conditional_OOF_pass": status != "R2R1_CONDITIONAL_OOF_FAILED",
        "final_fit_performed": status != "R2R1_CONDITIONAL_OOF_FAILED",
        "energy_labels_used": False,
        "development_or_held_access": False,
        "split_design_audits": {},
        "nested_OOF": {},
        "execution_authorization": {
            "format": "graphene_r2r1_external_execution_authorization_v2"
        },
        "attempt3_completed_recovery": {},
        "label_parser_receipt": {},
        "thermal_label_raw_sha256": {},
    }
    if status != "R2R1_CONDITIONAL_OOF_FAILED":
        final_pass = (
            status
            == "R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING"
        )
        receipt.update(
            {
                "final_train_gate_pass": final_pass,
                "mechanics_pending": final_pass,
                "encoder_updated": False,
                "final_alpha_selection": {},
                "final_fit": {},
                "final_train_metrics": {},
                "final_coefficients": np.linspace(-0.01, 0.01, 65).tolist(),
                "final_coefficients_sha256": "a" * 64,
                "final_scale_raw_sha256": "b" * 64,
            }
        )
        arrays.update(
            {
                "final_predicted_force_eV_A": np.zeros(
                    (92, 72, 3), dtype="<f8"
                ),
                "final_coefficients": np.linspace(
                    -0.01, 0.01, 65, dtype=np.float64
                ).astype("<f8"),
                "final_scale_eV_A": np.ones(65, dtype="<f8"),
            }
        )
    return receipt, arrays


def _private_checkpoint_fixture() -> tuple[dict[str, bytes], str, np.ndarray]:
    physical = np.linspace(-0.01, 0.01, 65, dtype="<f8")
    array_buffer = io.BytesIO()
    np.savez(array_buffer, physical_p=physical)
    array_raw = array_buffer.getvalue()
    receipt = frozen.checkpoint_receipt_payload(
        physical,
        array_sha256=hashlib.sha256(array_raw).hexdigest(),
        fit_manifest_sha256="a" * 64,
        fit_receipt_sha256="b" * 64,
    )
    receipt_raw = linear.canonical_json_bytes(receipt)
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    marker = (frozen.CHECKPOINT_STATUS + "\n" + receipt_sha + "\n").encode(
        "ascii"
    )
    return (
        {
            frozen.CHECKPOINT_ARRAY_BASENAME: array_raw,
            frozen.CHECKPOINT_RECEIPT_BASENAME: receipt_raw,
            frozen.CHECKPOINT_MARKER_BASENAME: marker,
        },
        receipt_sha,
        physical,
    )


def test_atomic_owned_record_allows_same_bytes_rewrite_with_restored_mtime(
    tmp_path: Path, monkeypatch
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_capture = aggregate._capture_held_regular_descriptor
    injected = {"value": False}

    def rewrite_before_initial(descriptor, expected_raw, *, label, **kwargs):
        if "initial private atomic inode" in label and not injected["value"]:
            _rewrite_same_bytes_and_restore_mtime(descriptor)
            injected["value"] = True
        return original_capture(
            descriptor, expected_raw, label=label, **kwargs
        )

    monkeypatch.setattr(
        aggregate, "_capture_held_regular_descriptor", rewrite_before_initial
    )
    try:
        identity = aggregate._atomic_write_bytes_at(
            directory_fd, "owned.bin", b"owned\n"
        )
    finally:
        os.close(directory_fd)
    assert injected["value"] is True
    assert set(identity) == aggregate.OWNED_REGULAR_IDENTITY_KEYS
    assert (tmp_path / "owned.bin").read_bytes() == b"owned\n"
    assert not any(path.name.startswith(".owned.bin.") for path in tmp_path.iterdir())


def test_running_owned_record_allows_same_bytes_rewrite_with_restored_mtime(
    tmp_path: Path, monkeypatch
) -> None:
    root_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_rewrite = aggregate._rewrite_held_file
    injected = {"value": False}

    def rewrite_then_tamper(descriptor, payload):
        identity = original_rewrite(descriptor, payload)
        if payload == (aggregate.FORMAT + "\n").encode("ascii"):
            _rewrite_same_bytes_and_restore_mtime(descriptor)
            injected["value"] = True
        return identity

    monkeypatch.setattr(aggregate, "_rewrite_held_file", rewrite_then_tamper)
    try:
        descriptor, identity = aggregate._create_running_marker(root_fd)
        os.close(descriptor)
    finally:
        os.close(root_fd)
    assert injected["value"] is True
    assert set(identity) == aggregate.OWNED_REGULAR_IDENTITY_KEYS
    assert (tmp_path / "RUNNING").read_bytes() == (
        aggregate.FORMAT + "\n"
    ).encode("ascii")


@pytest.mark.parametrize(
    "name_sequences,accepted",
    [
        (
            [
                b"com.apple.provenance\0",
                b"com.apple.decmpfs\0",
                b"com.apple.provenance\0",
                b"com.apple.decmpfs\0",
            ],
            True,
        ),
        ([b"user.r2r1-tamper\0"], False),
        ([b"com.apple.provenance\0", b"user.r2r1-tamper\0"], False),
    ],
)
def test_owned_capture_ignores_only_managed_xattr_name_changes(
    tmp_path: Path, monkeypatch, name_sequences: list[bytes], accepted: bool
) -> None:
    class FakeCall:
        def __init__(self, values):
            self.values = list(values)
            self.argtypes = None
            self.restype = None

        def __call__(self, _descriptor, buffer, size, _options):
            raw = self.values.pop(0)
            assert len(raw) <= size
            linear.ctypes.memmove(buffer, raw, len(raw))
            return len(raw)

    fake_call = FakeCall(name_sequences)

    class FakeLibc:
        flistxattr = fake_call

    monkeypatch.setattr(linear.sys, "platform", "darwin")
    monkeypatch.setattr(linear.ctypes.util, "find_library", lambda _name: "libc")
    monkeypatch.setattr(linear.ctypes, "CDLL", lambda *_a, **_k: FakeLibc())
    path = tmp_path / "owned.bin"
    path.write_bytes(b"owned\n")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        if accepted:
            record = aggregate._capture_held_regular_descriptor(
                descriptor, b"owned\n", label="managed-xattr capture"
            )
            assert record["xattrs"] == {}
        else:
            with pytest.raises(ValueError, match="unsupported authoritative xattrs"):
                aggregate._capture_held_regular_descriptor(
                    descriptor, b"owned\n", label="unknown-xattr capture"
                )
    finally:
        os.close(descriptor)


def test_held_regular_write_guard_detects_payload_write_with_restored_mtime(
    tmp_path: Path,
) -> None:
    path = tmp_path / "guarded.bin"
    path.write_bytes(b"guarded-payload\n")
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    identity = linear._owned_regular_identity(os.fstat(descriptor))
    try:
        with pytest.raises(ValueError, match="held payload write|portable write"):
            with linear._HeldRegularWriteGuard("R2R-1 test guard") as guard:
                guard.watch_owned_descriptor(
                    descriptor,
                    expected_identity=identity,
                    label="R2R-1 test guarded file",
                )
                _rewrite_different_same_size_and_restore_mtime(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin vnode semantics")
def test_held_regular_write_guard_does_not_treat_attribute_event_as_payload_write(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "guarded-xattr.bin"
    path.write_bytes(b"guarded-payload\n")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    identity = linear._owned_regular_identity(os.fstat(descriptor))
    monkeypatch.setattr(linear, "_authoritative_owned_xattrs", lambda *_a, **_k: {})
    try:
        with linear._HeldRegularWriteGuard("R2R-1 attribute-only guard") as guard:
            guard.watch_owned_descriptor(
                descriptor,
                expected_identity=identity,
                label="R2R-1 managed-attribute simulation",
            )
            subprocess.run(
                ["/usr/bin/xattr", "-w", "user.r2r1_guard", "1", str(path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
    finally:
        subprocess.run(
            ["/usr/bin/xattr", "-d", "user.r2r1_guard", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.close(descriptor)


def test_snapshot_write_guard_closes_reverse_pass_tail(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        aggregate.FIT_MANIFEST_BASENAME: b'{"manifest":1}\n',
        aggregate.FIT_RECEIPT_BASENAME: b'{"receipt":1}\n',
    }
    for name, raw in payloads.items():
        (tmp_path / name).write_bytes(raw)
    snapshot = {
        name: {
            "kind": "file",
            "identity": linear._owned_regular_identity(os.stat(tmp_path / name)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        }
        for name, raw in payloads.items()
    }
    root_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_read = aggregate._read_regular_file_at
    manifest_reads = {"count": 0}

    def final_reader_then_tamper(directory_fd, name, label, **kwargs):
        raw = original_read(directory_fd, name, label, **kwargs)
        if name == aggregate.FIT_MANIFEST_BASENAME:
            manifest_reads["count"] += 1
            if manifest_reads["count"] == 2:
                writable = os.open(
                    aggregate.FIT_RECEIPT_BASENAME,
                    os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    _rewrite_different_same_size_and_restore_mtime(writable)
                finally:
                    os.close(writable)
        return raw

    monkeypatch.setattr(
        aggregate, "_read_regular_file_at", final_reader_then_tamper
    )
    try:
        with pytest.raises(ValueError, match="held payload write|portable write"):
            aggregate._verify_snapshot_bytes_fd(root_fd, snapshot)
    finally:
        os.close(root_fd)
    assert manifest_reads["count"] == 2


def test_manifest_generator_never_creates_marker(tmp_path: Path, monkeypatch) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    manifest = control / "freeze_manifest.json"
    receipt = run_cli.write_freeze_manifest(manifest)
    assert manifest.is_file()
    payload = json.loads(manifest.read_bytes())
    assert payload["format"] == "graphene_r2r1_freeze_manifest_v2"
    assert payload["attempt_id"] == "attempt2_retry1"
    assert payload["failed_attempt2_forensic_partial_must_remain_read_only"] is True
    assert receipt["authorization_marker_created"] is False
    assert not (manifest.parent / "R2R1_FORMAL_GO").exists()


def test_manifest_generator_rereads_owned_bytes_after_final_parent_rebind(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    manifest = control / "freeze_manifest.json"
    original_verify = linear._verify_directory_chain
    injected = {"value": False}

    def rebind_then_tamper(path, label, expected_chain):
        result = original_verify(path, label, expected_chain)
        if label == "R2R-1 control root final binding" and not injected["value"]:
            descriptor = os.open(
                manifest, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                _rewrite_different_same_size_and_restore_mtime(descriptor)
            finally:
                os.close(descriptor)
            injected["value"] = True
        return result

    monkeypatch.setattr(linear, "_verify_directory_chain", rebind_then_tamper)
    with pytest.raises(ValueError, match="freeze manifest.*generator return"):
        run_cli.write_freeze_manifest(manifest)
    assert injected["value"] is True
    assert manifest.is_file()
    assert not (control / "R2R1_FORMAL_GO").exists()


@pytest.mark.parametrize(
    "status,checkpoint_expected",
    [
        ("R2R1_CONDITIONAL_OOF_FAILED", False),
        ("R2R1_FINAL_READOUT_FAILED", False),
        ("R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING", True),
    ],
)
def test_atomic_local_release_three_scientific_terminal_branches(
    tmp_path: Path,
    monkeypatch,
    status: str,
    checkpoint_expected: bool,
) -> None:
    control, output, _candidate, release_manifest = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    pipeline = _mock_pipeline_result(status)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: pipeline,
    )
    result = aggregate.materialize_authorized_fit(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=ROOT
        / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    assert result["status"] == status
    assert result["canonical_root_reserved_before_fit"] is True
    assert result["terminal_committed_by_RUNNING_inode_rename"] is True
    assert (output / "DONE").is_file()
    assert not (output / "RUNNING").exists()
    assert not (output / "FAILED").exists()
    assert (output / "checkpoint").is_dir() is checkpoint_expected
    fit_receipt = json.loads((output / aggregate.FIT_RECEIPT_BASENAME).read_bytes())
    completion = json.loads((output / aggregate.COMPLETION_BASENAME).read_bytes())
    done_ledger = json.loads((output / "DONE").read_bytes())
    assert fit_receipt["format"] == aggregate.FIT_RECEIPT_FORMAT
    assert fit_receipt["pipeline_receipt"]["format"] == linear.AGGREGATE_FORMAT
    assert fit_receipt["pipeline_receipt"]["execution_authorization"]["format"] == (
        "graphene_r2r1_external_execution_authorization_v2"
    )
    assert completion["format"] == aggregate.COMPLETION_FORMAT
    assert done_ledger["format"] == aggregate.TERMINAL_LEDGER_FORMAT
    for item in completion["precompletion_artifact_snapshot"].values():
        if item["kind"] == "file":
            assert set(item["identity"]) == aggregate.OWNED_REGULAR_IDENTITY_KEYS
    assert "terminal_inode_xattrs_sha256" not in done_ledger
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=ROOT
        / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=result.anchor_witness,
    )
    assert json.loads(release_manifest.read_bytes())["format"] == (
        aggregate.RELEASE_MANIFEST_FORMAT
    )
    recovered = aggregate.recover_completed_fit(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=ROOT
        / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        expected_release_manifest_sha256=anchor["sha256"],
    )
    assert recovered["scientific_recomputed"] is True
    assert recovered["status"] == status
    assert recovered["checkpoint_published"] is checkpoint_expected
    if checkpoint_expected:
        checkpoint = frozen.load_frozen_readout_checkpoint(
            output / "checkpoint",
            expected_receipt_sha256=result["checkpoint"]["receipt_sha256"],
        )
        assert checkpoint.physical_p.shape == (65,)


def test_owned_release_directories_allow_time_only_drift(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    pipeline = _mock_pipeline_result(
        "R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING"
    )
    calls = {"count": 0}

    def drift_directories(**kwargs):
        calls["count"] += 1
        for path in (output, output / aggregate.CHECKPOINT_DIRNAME):
            if path.is_dir():
                before = os.stat(path)
                os.utime(
                    path,
                    ns=(before.st_atime_ns, before.st_mtime_ns + calls["count"]),
                )
        return pipeline

    monkeypatch.setattr(
        aggregate.linear, "authorized_fit_pipeline", drift_directories
    )
    receipt = aggregate.materialize_authorized_fit(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    assert calls["count"] == 3
    assert receipt["status"] == pipeline[0]["status"]
    assert (output / "DONE").is_file()


def test_external_completion_pin_rejects_self_consistent_root_rewrite(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, release_manifest = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    pipeline = _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED")
    monkeypatch.setattr(
        aggregate.linear, "authorized_fit_pipeline", lambda **kwargs: pipeline
    )
    result = aggregate.materialize_authorized_fit(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=ROOT
        / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=ROOT
        / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=result.anchor_witness,
    )
    completion_path = output / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    completion["status"] = "R2R1_FINAL_READOUT_FAILED"
    completion_path.write_bytes(linear.canonical_json_bytes(completion))
    rewritten_sha = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    (output / "DONE").write_text(
        completion["status"] + "\n" + rewritten_sha + "\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="DONE identity differs from its external anchor"):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=ROOT
            / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )


def test_missing_marker_leaves_fit_root_absent(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest = control / "freeze_manifest.json"
    manifest.write_bytes(linear.canonical_json_bytes(linear.freeze_manifest_payload()))
    with pytest.raises((FileNotFoundError, PermissionError)):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=control / "R2R1_FORMAL_GO",
        )
    assert not output.exists()


@pytest.mark.parametrize("collision_kind", ["file", "directory", "symlink"])
def test_publication_collision_never_replaces_target_or_leaves_staging(
    tmp_path: Path, monkeypatch, collision_kind: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    if collision_kind == "file":
        output.write_text("preserve", encoding="utf-8")
    elif collision_kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(sentinel)
    fit_calls = {"count": 0}

    def forbidden_fit(**kwargs):
        fit_calls["count"] += 1
        pytest.fail("collision reached fit")

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_fit)
    with pytest.raises(FileExistsError):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    if collision_kind == "file":
        assert output.read_text(encoding="utf-8") == "preserve"
    elif collision_kind == "directory":
        assert output.is_dir() and not any(output.iterdir())
    else:
        assert output.is_symlink() and output.readlink() == sentinel
    assert fit_calls["count"] == 0


def test_post_terminal_commit_fsync_fault_preserves_done_release(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    pipeline = _mock_pipeline_result(
        "R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING"
    )
    monkeypatch.setattr(
        aggregate.linear, "authorized_fit_pipeline", lambda **kwargs: pipeline
    )
    renamed = {"value": False}
    original_rename = aggregate._rename_noreplace_at
    original_fsync = aggregate.os.fsync

    def rename_then_flag(*args, **kwargs):
        result = original_rename(*args, **kwargs)
        renamed["value"] = True
        return result

    def fault_after_rename(descriptor):
        if renamed["value"]:
            raise OSError("injected post-rename fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(aggregate, "_rename_noreplace_at", rename_then_flag)
    monkeypatch.setattr(aggregate.os, "fsync", fault_after_rename)
    with pytest.raises(aggregate._TerminalCommittedError, match="committed"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert (output / "DONE").is_file()
    assert not (output / "FAILED").exists()
    assert (output / "checkpoint" / frozen.CHECKPOINT_MARKER_BASENAME).is_file()


def test_terminal_rename_success_then_move_back_is_committed_uncertain(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED"),
    )
    original_rename = aggregate._rename_noreplace_at
    injected = {"value": False}

    def rename_then_move_back(directory_fd, source, destination, **kwargs):
        result = original_rename(directory_fd, source, destination, **kwargs)
        if destination == "DONE" and not injected["value"]:
            os.rename(
                destination,
                source,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            injected["value"] = True
            raise OSError("injected rename-success then move-back fault")
        return result

    monkeypatch.setattr(aggregate, "_rename_noreplace_at", rename_then_move_back)
    with pytest.raises(aggregate._TerminalCommittedError, match="committed"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert injected["value"] is True
    assert (output / "RUNNING").is_file()
    assert (output / "EXIT_CODE").read_bytes() == b"0\n"
    assert (output / aggregate.FIT_ARRAYS_BASENAME).is_file()
    assert not (output / "DONE").exists()
    assert not (output / "FAILED").exists()


def test_terminal_post_rename_destination_stat_fault_cannot_trigger_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED"),
    )
    original_rename = aggregate._rename_noreplace_at
    original_stat = aggregate.os.stat
    probes = {"done": 0}

    def rename_then_fault(directory_fd, source, destination, **kwargs):
        original_rename(directory_fd, source, destination, **kwargs)
        raise OSError("injected exception after successful terminal rename")

    def destination_stat_eio(path, *args, **kwargs):
        if path == "DONE" and kwargs.get("dir_fd") is not None:
            probes["done"] += 1
            raise OSError("injected DONE stat EIO")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(aggregate, "_rename_noreplace_at", rename_then_fault)
    monkeypatch.setattr(aggregate.os, "stat", destination_stat_eio)
    with pytest.raises(aggregate._TerminalCommittedError, match="committed"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert probes["done"] == 0
    assert (output / "DONE").is_file()
    assert (output / "EXIT_CODE").read_bytes() == b"0\n"
    assert not (output / "FAILED").exists()


@pytest.mark.parametrize("target_name", ["EXIT_CODE", "DONE"])
def test_materializer_final_release_bytes_close_postcommit_terminal_siblings(
    tmp_path: Path, monkeypatch, target_name: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED"),
    )
    original_pair = aggregate._verify_active_fit_publication_pair
    injected = {"value": False}

    def pair_then_tamper(*args, **kwargs):
        result = original_pair(*args, **kwargs)
        if kwargs.get("marker_name") == "DONE" and not injected["value"]:
            descriptor = os.open(
                output / target_name,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                _rewrite_different_same_size_and_restore_mtime(descriptor)
            finally:
                os.close(descriptor)
            injected["value"] = True
        return result

    monkeypatch.setattr(
        aggregate, "_verify_active_fit_publication_pair", pair_then_tamper
    )
    with pytest.raises(ValueError, match="ledger-bound artifact bytes changed"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert injected["value"] is True
    assert (output / "DONE").is_file()
    assert not (output / "RUNNING").exists()
    assert not (output / "FAILED").exists()


def test_recovery_root_ordinary_directory_swap_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, release_manifest = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    pipeline = _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED")
    monkeypatch.setattr(
        aggregate.linear, "authorized_fit_pipeline", lambda **kwargs: pipeline
    )
    result = aggregate.materialize_authorized_fit(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=result.anchor_witness,
    )
    original_rebind = aggregate._verify_snapshot_identities_fd
    swapped = {"value": False}
    backup = tmp_path / "original_release"

    def swap_after_ledger_rebind(root_fd, snapshot):
        original_rebind(root_fd, snapshot)
        if not swapped["value"]:
            output.rename(backup)
            shutil.copytree(backup, output)
            swapped["value"] = True

    monkeypatch.setattr(
        aggregate, "_verify_snapshot_identities_fd", swap_after_ledger_rebind
    )
    with pytest.raises(
        ValueError,
        match="no longer names the held root|held payload write",
    ):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )
    assert swapped["value"] is True
    assert result["status"] == "R2R1_CONDITIONAL_OOF_FAILED"


def _materialize_mock_release(
    tmp_path: Path,
    monkeypatch,
    status: str = "R2R1_CONDITIONAL_OOF_FAILED",
) -> tuple[Path, Path, Path, Path, Path, dict]:
    control, output, _candidate, release_manifest = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    pipeline = _mock_pipeline_result(status)
    monkeypatch.setattr(
        aggregate.linear, "authorized_fit_pipeline", lambda **kwargs: pipeline
    )
    receipt = aggregate.materialize_authorized_fit(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    return output, release_manifest, manifest, marker, control, receipt


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("fit_arrays", "snapshot identity changed|changed before write-event guard"),
        ("completion", "ledger-bound completion"),
        ("exit", "ledger-bound EXIT_CODE"),
        ("root_mode", "root stable binding"),
    ],
)
def test_unanchored_done_persistent_ledger_rejects_postcommit_drift_before_solve(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    expected_message: str,
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    if mutation == "root_mode":
        os.chmod(output, 0o755)
    else:
        basename = {
            "fit_arrays": aggregate.FIT_ARRAYS_BASENAME,
            "completion": aggregate.COMPLETION_BASENAME,
            "exit": "EXIT_CODE",
        }[mutation]
        path = output / basename
        raw = path.read_bytes()
        before = os.stat(path)
        path.write_bytes(raw)
        after = os.stat(path)
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
        assert after.st_ctime_ns != before.st_ctime_ns
    calls = {"pipeline": 0, "parser": 0, "svd": 0}

    def forbidden_pipeline(**kwargs):
        calls["pipeline"] += 1
        raise AssertionError("persistent-ledger drift reached scientific solve")

    def forbidden_parser(value):
        calls["parser"] += 1
        raise AssertionError("persistent-ledger drift reached label parser")

    def forbidden_svd(*args, **kwargs):
        calls["svd"] += 1
        raise AssertionError("persistent-ledger drift reached ridge SVD")

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_pipeline)
    monkeypatch.setattr(aggregate.linear, "_parse_whitelisted_float", forbidden_parser)
    monkeypatch.setattr(aggregate.linear.np.linalg, "svd", forbidden_svd)
    with pytest.raises(ValueError, match=expected_message):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert calls == {"pipeline": 0, "parser": 0, "svd": 0}
    assert not release_manifest.parent.exists()


def test_fresh_witness_rejects_same_inode_done_mtime_rewrite(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    done = output / "DONE"
    raw = done.read_bytes()
    before = os.stat(done)
    done.write_bytes(raw)
    after = os.stat(done)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert after.st_ctime_ns != before.st_ctime_ns
    with pytest.raises(ValueError, match="DONE identity differs from its external anchor"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert not release_manifest.parent.exists()


def test_fresh_witness_rejects_unknown_done_xattr_before_solve(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_xattrs = aggregate._authoritative_owned_xattrs
    calls = {"pipeline": 0, "parser": 0, "svd": 0}

    def changed_done_xattrs(descriptor, *, label):
        if "DONE" in label:
            raise ValueError(
                "R2R-1 owned file has unsupported authoritative xattrs"
            )
        return original_xattrs(descriptor, label=label)

    def forbidden_pipeline(**kwargs):
        calls["pipeline"] += 1
        raise AssertionError("DONE xattr drift reached scientific solve")

    def forbidden_parser(value):
        calls["parser"] += 1
        raise AssertionError("DONE xattr drift reached label parser")

    def forbidden_svd(*args, **kwargs):
        calls["svd"] += 1
        raise AssertionError("DONE xattr drift reached ridge SVD")

    monkeypatch.setattr(aggregate, "_authoritative_owned_xattrs", changed_done_xattrs)
    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_pipeline)
    monkeypatch.setattr(aggregate.linear, "_parse_whitelisted_float", forbidden_parser)
    monkeypatch.setattr(aggregate.linear.np.linalg, "svd", forbidden_svd)
    with pytest.raises(ValueError, match="unsupported authoritative xattrs"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert calls == {"pipeline": 0, "parser": 0, "svd": 0}
    assert not release_manifest.parent.exists()


def test_unwitnessed_unanchored_done_requires_new_external_pin(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, _receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    with pytest.raises(PermissionError, match="materialization witness"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
        )
    assert not release_manifest.parent.exists()


def test_disk_derived_materialization_witness_forgery_is_rejected_before_solve(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    ledger = json.loads((output / "DONE").read_bytes())
    forged_payload = {
        "release_root": str(output),
        "status": receipt["status"],
        "done_sha256": receipt["done_sha256"],
        "done_identity": receipt["done_identity"],
        "expected_release_snapshot": receipt["expected_release_snapshot"],
    }

    class PublicFieldForgery:
        pass

    public_fields = PublicFieldForgery()
    for key, value in forged_payload.items():
        setattr(public_fields, key, value)

    def disk_replay_implementation(**_kwargs):
        return {"status": receipt["status"]}, forged_payload

    alternate_materializer, _alternate_require = (
        aggregate._build_materialization_witness_boundary(
            disk_replay_implementation
        )
    )
    alternate_result = alternate_materializer(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    calls = {"pipeline": 0, "parser": 0, "svd": 0}

    def forbidden_pipeline(**kwargs):
        calls["pipeline"] += 1
        raise AssertionError("forged witness reached scientific solve")

    def forbidden_parser(value):
        calls["parser"] += 1
        raise AssertionError("forged witness reached label parser")

    def forbidden_svd(*args, **kwargs):
        calls["svd"] += 1
        raise AssertionError("forged witness reached ridge SVD")

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_pipeline)
    monkeypatch.setattr(aggregate.linear, "_parse_whitelisted_float", forbidden_parser)
    monkeypatch.setattr(aggregate.linear.np.linalg, "svd", forbidden_svd)
    assert not hasattr(aggregate, "_MaterializationWitness")
    genuine = receipt.anchor_witness
    assert not hasattr(genuine, "_seal")
    assert not hasattr(genuine, "_payload_raw")
    with pytest.raises(AttributeError):
        setattr(genuine, "_payload_raw", linear.canonical_json_bytes(forged_payload))
    with pytest.raises(PermissionError, match="cannot be constructed"):
        type(genuine)(object())
    with pytest.raises(PermissionError, match="non-copyable"):
        copy.copy(genuine)
    with pytest.raises(PermissionError, match="non-copyable"):
        copy.deepcopy(genuine)
    with pytest.raises(TypeError, match="non-serializable"):
        pickle.dumps(genuine)
    unregistered_exact_type = object.__new__(type(genuine))
    for forged in (
        public_fields,
        alternate_result.anchor_witness,
        unregistered_exact_type,
    ):
        with pytest.raises(PermissionError, match="sealed in-process"):
            aggregate.anchor_completed_release(
                output_root=output,
                attempt3_root=ATTEMPT3,
                thermal92_path=linear.RECOMMENDED_THERMAL92,
                freeze_manifest=manifest,
                authorization_marker=marker,
                release_manifest=release_manifest,
                materialization_witness=forged,
            )
        assert not release_manifest.parent.exists()
    assert calls == {"pipeline": 0, "parser": 0, "svd": 0}


def test_unanchored_done_exact_bytes_new_inode_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    done = output / "DONE"
    raw = done.read_bytes()
    held_old = os.open(done, os.O_RDONLY)
    try:
        old = os.fstat(held_old)
        done.unlink()
        done.write_bytes(raw)
        new = os.stat(done)
        assert (new.st_dev, new.st_ino) != (old.st_dev, old.st_ino)
        with pytest.raises(
            ValueError, match="DONE identity differs from its external anchor"
        ):
            aggregate.anchor_completed_release(
                output_root=output,
                attempt3_root=ATTEMPT3,
                thermal92_path=linear.RECOMMENDED_THERMAL92,
                freeze_manifest=manifest,
                authorization_marker=marker,
                release_manifest=release_manifest,
                materialization_witness=receipt.anchor_witness,
            )
    finally:
        os.close(held_old)
    assert not release_manifest.parent.exists()


def test_external_anchor_pins_done_full_identity_after_self_anchor_exception(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    done = output / "DONE"
    raw = done.read_bytes()
    before = os.stat(done)
    done.write_bytes(raw)
    after = os.stat(done)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    with pytest.raises(ValueError, match="DONE identity differs from its external anchor"):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )


def test_pipeline_error_before_artifacts_commits_exact_failed(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected solve error")),
    )
    with pytest.raises(RuntimeError, match="injected solve error"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert {path.name for path in output.iterdir()} == {
        "failure.json",
        "EXIT_CODE",
        "FAILED",
    }
    assert (output / "EXIT_CODE").read_bytes() == b"1\n"
    assert not (output / "RUNNING").exists()
    failed_raw = (output / "FAILED").read_bytes()
    assert len(failed_raw) <= aggregate.ROOT_FILE_SIZE_LIMITS["FAILED"]
    ledger = aggregate._validate_terminal_ledger(
        json.loads(failed_raw), expected_terminal="FAILED"
    )
    failure_raw = (output / "failure.json").read_bytes()
    assert ledger["payload_sha256"] == hashlib.sha256(failure_raw).hexdigest()
    assert ledger["exit_sha256"] == hashlib.sha256(b"1\n").hexdigest()
    assert ledger["root_binding_identity"] == linear._directory_binding_identity(
        os.stat(output)
    )
    failed_fd = os.open(output / "FAILED", os.O_RDONLY)
    try:
        assert ledger["terminal_inode_stable_identity"] == aggregate._file_stable_identity(
            linear._stat_identity(os.fstat(failed_fd))
        )
        assert "terminal_inode_xattrs_sha256" not in ledger
    finally:
        os.close(failed_fd)


@pytest.mark.parametrize("directory_target", ["root", "parent"])
def test_fit_directory_stable_binding_change_precedes_pipeline(
    tmp_path: Path, monkeypatch, directory_target: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    pipeline_calls = {"count": 0}

    def forbidden_pipeline(**kwargs):
        pipeline_calls["count"] += 1
        raise AssertionError("stable-directory drift reached fit pipeline")

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_pipeline)
    original_pair = aggregate._verify_active_fit_publication_pair
    injected = {"value": False}
    changed_path = output if directory_target == "root" else output.parent
    original_mode = stat.S_IMODE(os.stat(changed_path).st_mode) if changed_path.exists() else None

    def mutate_before_pair(*args, **kwargs):
        if kwargs.get("label") == "R2R-1 publication pair before fit pipeline":
            os.chmod(changed_path, 0o755)
            injected["value"] = True
        return original_pair(*args, **kwargs)

    monkeypatch.setattr(
        aggregate, "_verify_active_fit_publication_pair", mutate_before_pair
    )
    try:
        with pytest.raises(ValueError, match="stable directory binding"):
            aggregate.materialize_authorized_fit(
                output_root=output,
                attempt3_root=ATTEMPT3,
                thermal92_path=linear.RECOMMENDED_THERMAL92,
                freeze_manifest=manifest,
                authorization_marker=marker,
            )
    finally:
        if directory_target == "parent" and original_mode is not None:
            os.chmod(changed_path, original_mode)
    assert injected["value"] is True
    assert pipeline_calls["count"] == 0
    assert (output / "RUNNING").is_file()
    assert not (output / "DONE").exists()
    assert not (output / "FAILED").exists()


@pytest.mark.parametrize("tamper", ["extra", "running_changed_bytes"])
def test_fit_initial_inventory_and_running_identity_close_before_pipeline(
    tmp_path: Path, monkeypatch, tamper: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    pipeline_calls = {"count": 0}
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: pipeline_calls.__setitem__(
            "count", pipeline_calls["count"] + 1
        ),
    )
    original_pair = aggregate._verify_active_fit_publication_pair
    injected = {"value": False}

    def mutate_before_pair(*args, **kwargs):
        if kwargs.get("label") == "R2R-1 publication pair before fit pipeline":
            if tamper == "extra":
                (output / "foreign").write_bytes(b"preserve\n")
            else:
                descriptor = os.open(output / "RUNNING", os.O_RDWR)
                try:
                    _rewrite_different_same_size_and_restore_mtime(descriptor)
                finally:
                    os.close(descriptor)
            injected["value"] = True
        return original_pair(*args, **kwargs)

    monkeypatch.setattr(
        aggregate, "_verify_active_fit_publication_pair", mutate_before_pair
    )
    with pytest.raises(ValueError, match="inventory|marker identity|marker payload"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert injected["value"] is True
    assert pipeline_calls["count"] == 0
    assert (output / "RUNNING").is_file()
    assert not (output / "DONE").exists()
    assert not (output / "FAILED").exists()


def test_foreign_entry_on_failure_is_preserved_as_running_partial(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)

    def inject_foreign(**kwargs):
        (output / "foreign_sentinel").write_text("preserve", encoding="ascii")
        raise RuntimeError("late injected error")

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", inject_foreign)
    with pytest.raises(RuntimeError, match="late injected error"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert (output / "RUNNING").is_file()
    assert (output / "foreign_sentinel").read_text(encoding="ascii") == "preserve"
    assert not (output / "FAILED").exists()


def test_in_place_running_mutation_cannot_be_overwritten_by_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED"),
    )
    original_finish = aggregate._finish_success

    def mutate_running(*args, **kwargs):
        with (output / "RUNNING").open("ab") as handle:
            handle.write(b"tamper")
            handle.flush()
            os.fsync(handle.fileno())
        return original_finish(*args, **kwargs)

    monkeypatch.setattr(aggregate, "_finish_success", mutate_running)
    with pytest.raises(ValueError, match="RUNNING"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert (output / "RUNNING").read_bytes().endswith(b"tamper")
    assert not (output / "DONE").exists()
    assert not (output / "FAILED").exists()


@pytest.mark.parametrize("terminal_case", ["DONE", "FAILED"])
def test_terminal_owned_identity_allows_same_bytes_ctime_change(
    tmp_path: Path, monkeypatch, terminal_case: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    if terminal_case == "DONE":
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: _mock_pipeline_result(
                "R2R1_CONDITIONAL_OOF_FAILED"
            ),
        )
    else:
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("injected pre-science failure")
            ),
        )
    original_rewrite = aggregate._rewrite_held_file
    injected = {"value": False}

    def rewrite_terminal_then_tamper(descriptor, payload):
        identity = original_rewrite(descriptor, payload)
        if payload != (aggregate.FORMAT + "\n").encode("ascii"):
            _rewrite_same_bytes_and_restore_mtime(descriptor)
            injected["value"] = True
        return identity

    monkeypatch.setattr(
        aggregate, "_rewrite_held_file", rewrite_terminal_then_tamper
    )
    invoke = lambda: aggregate.materialize_authorized_fit(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    if terminal_case == "DONE":
        invoke()
    else:
        with pytest.raises(RuntimeError, match="pre-science"):
            invoke()
    assert injected["value"] is True
    assert not (output / "RUNNING").exists()
    assert (output / terminal_case).is_file()


@pytest.mark.parametrize("terminal_case", ["DONE", "FAILED"])
def test_terminal_callback_cannot_change_reserved_root_binding(
    tmp_path: Path, monkeypatch, terminal_case: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    if terminal_case == "DONE":
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: _mock_pipeline_result(
                "R2R1_CONDITIONAL_OOF_FAILED"
            ),
        )
    else:
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("injected pre-science failure")
            ),
        )
    original_commit = aggregate._commit_held_terminal
    injected = {"value": False}

    def inject_after_long_validator(*args, **kwargs):
        validator = kwargs["precommit_validator"]

        def validate_then_chmod():
            result = validator()
            os.fchmod(args[0], 0o755)
            injected["value"] = True
            return result

        return original_commit(
            *args, **{**kwargs, "precommit_validator": validate_then_chmod}
        )

    monkeypatch.setattr(
        aggregate, "_commit_held_terminal", inject_after_long_validator
    )
    with pytest.raises((ValueError, RuntimeError), match="root|pre-science"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert injected["value"] is True
    assert (output / "RUNNING").is_file()
    assert not (output / "DONE").exists()
    assert not (output / "FAILED").exists()


@pytest.mark.parametrize("tamper", ["arrays", "done", "extra"])
def test_materializer_return_boundary_detects_committed_tree_tamper(
    tmp_path: Path, monkeypatch, tamper: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED"),
    )
    original_verify = aggregate._verify_root_entry
    injected = {"value": False}

    def inject_after_commit(parent_fd, root_name, root_fd, *, label):
        result = original_verify(parent_fd, root_name, root_fd, label=label)
        if label == "R2R-1 committed root" and not injected["value"]:
            if tamper == "arrays":
                path = output / aggregate.FIT_ARRAYS_BASENAME
                raw = bytearray(path.read_bytes())
                raw[-1] ^= 1
                path.write_bytes(raw)
            elif tamper == "done":
                path = output / "DONE"
                raw = bytearray(path.read_bytes())
                raw[0] ^= 1
                path.write_bytes(raw)
            else:
                (output / "extra").write_bytes(b"x")
            injected["value"] = True
        return result

    monkeypatch.setattr(aggregate, "_verify_root_entry", inject_after_commit)
    with pytest.raises(
        (ValueError, aggregate._TerminalCommittedError), match="changed|inventory|committed"
    ):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert injected["value"] is True
    assert (output / "DONE").exists()
    assert not (output / "FAILED").exists()


def test_candidate_precommit_failure_leaves_canonical_absent_and_retry_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_write = aggregate._atomic_write_bytes_at
    failed = {"value": False}

    def fail_first_candidate_write(directory_fd, name, payload):
        if name == release_manifest.name and not failed["value"]:
            failed["value"] = True
            raise OSError("injected candidate precommit failure")
        return original_write(directory_fd, name, payload)

    monkeypatch.setattr(aggregate, "_atomic_write_bytes_at", fail_first_candidate_write)
    with pytest.raises(OSError, match="candidate precommit"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert not release_manifest.parent.exists()
    monkeypatch.setattr(aggregate, "_atomic_write_bytes_at", original_write)
    recovered = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    assert recovered["newly_materialized"] is True
    assert release_manifest.is_file()


def test_candidate_anchor_owned_identity_allows_same_bytes_rewrite(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_commit = aggregate._commit_candidate_directory_at
    injected = {"value": False}

    def bounded_tamper(*args, **kwargs):
        candidate_fd = args[3]
        descriptor = os.open(
            kwargs["anchor_name"],
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=candidate_fd,
        )
        try:
            _rewrite_same_bytes_and_restore_mtime(descriptor)
        finally:
            os.close(descriptor)
        injected["value"] = True
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(aggregate, "_commit_candidate_directory_at", bounded_tamper)
    aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    assert injected["value"] is True
    assert release_manifest.is_file()


def test_candidate_temporary_directory_time_drift_does_not_change_binding(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_commit = aggregate._commit_candidate_directory_at
    injected = {"value": False}

    def drift_directory(*args, **kwargs):
        candidate_fd = args[3]
        before = os.fstat(candidate_fd)
        os.utime(
            candidate_fd,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1),
        )
        injected["value"] = True
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(aggregate, "_commit_candidate_directory_at", drift_directory)
    receipt = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    assert injected["value"] is True
    assert receipt["newly_materialized"] is True
    assert release_manifest.is_file()


def test_candidate_postcommit_fault_preserves_anchor_and_retry_is_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_rename = aggregate._rename_noreplace_at
    injected = {"value": False}

    def rename_then_fault(directory_fd, source, destination, **kwargs):
        result = original_rename(directory_fd, source, destination, **kwargs)
        if destination == release_manifest.parent.name and not injected["value"]:
            injected["value"] = True
            raise OSError("injected post-candidate-rename failure")
        return result

    monkeypatch.setattr(aggregate, "_rename_noreplace_at", rename_then_fault)
    with pytest.raises(aggregate._AnchorCommittedError, match="committed"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    first = release_manifest.read_bytes()
    monkeypatch.setattr(aggregate, "_rename_noreplace_at", original_rename)
    recovered = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    assert recovered["newly_materialized"] is False
    assert release_manifest.read_bytes() == first


def test_candidate_rename_success_then_move_back_is_committed_uncertain(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_rename = aggregate._rename_noreplace_at
    injected = {"value": False}

    def rename_then_move_back(directory_fd, source, destination, **kwargs):
        result = original_rename(directory_fd, source, destination, **kwargs)
        if destination == release_manifest.parent.name and not injected["value"]:
            os.rename(
                destination,
                source,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            injected["value"] = True
            raise OSError("injected candidate rename-success then move-back fault")
        return result

    monkeypatch.setattr(aggregate, "_rename_noreplace_at", rename_then_move_back)
    with pytest.raises(aggregate._AnchorCommittedError, match="committed"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert injected["value"] is True
    assert not release_manifest.parent.exists()
    temporaries = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(".r2r1-candidate-")
        and path.name.endswith(".tmp")
    ]
    assert len(temporaries) == 1
    assert (temporaries[0] / release_manifest.name).is_file()


def test_candidate_dir_fsync_mutation_is_caught_before_canonical_commit(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_fsync = aggregate.os.fsync
    mutated = {"value": False}
    candidate_dir_fsyncs = {"count": 0}

    def fsync_then_mutate(descriptor):
        result = original_fsync(descriptor)
        if not mutated["value"]:
            try:
                names = set(os.listdir(descriptor))
            except OSError:
                names = set()
            if names == {release_manifest.name}:
                candidate_dir_fsyncs["count"] += 1
            if (
                names == {release_manifest.name}
                and candidate_dir_fsyncs["count"] == 3
            ):
                file_fd = os.open(
                    release_manifest.name,
                    os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    raw = bytearray(
                        aggregate._read_regular_file_at(
                            descriptor,
                            release_manifest.name,
                            "test candidate anchor",
                            size_limit=2 * 1024 * 1024,
                        )
                    )
                    raw[-2] ^= 1
                    os.lseek(file_fd, 0, os.SEEK_SET)
                    os.write(file_fd, raw)
                    original_fsync(file_fd)
                finally:
                    os.close(file_fd)
                mutated["value"] = True
        return result

    monkeypatch.setattr(aggregate.os, "fsync", fsync_then_mutate)
    with pytest.raises(ValueError, match="anchor changed|bounded single-link"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert mutated["value"] is True
    assert not release_manifest.parent.exists()


@pytest.mark.parametrize(
    "artifact,status",
    [
        ("arrays", "R2R1_CONDITIONAL_OOF_FAILED"),
        ("completion", "R2R1_CONDITIONAL_OOF_FAILED"),
        ("done", "R2R1_CONDITIONAL_OOF_FAILED"),
        ("checkpoint", "R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING"),
    ],
)
def test_recovery_detects_in_place_change_during_scientific_resolve(
    tmp_path: Path, monkeypatch, artifact: str, status: str
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch, status)
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    pipeline = _mock_pipeline_result(status)
    mutated = {"value": False}

    def mutate_during_resolve(**kwargs):
        if not mutated["value"]:
            if artifact == "arrays":
                path = output / aggregate.FIT_ARRAYS_BASENAME
            elif artifact == "completion":
                path = output / aggregate.COMPLETION_BASENAME
            elif artifact == "done":
                path = output / "DONE"
            else:
                path = output / "checkpoint" / frozen.CHECKPOINT_MARKER_BASENAME
            raw = bytearray(path.read_bytes())
            raw[-1 if artifact == "arrays" else 0] ^= 1
            path.write_bytes(raw)
            mutated["value"] = True
        return pipeline

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", mutate_during_resolve)
    with pytest.raises(
        ValueError, match="changed during scientific re-solve|checkpoint three-file"
    ):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )
    assert mutated["value"] is True


def test_array_schema_rejects_float_dimension_even_when_numerically_equal() -> None:
    schema = {
        "OOF_predicted_force_eV_A": {
            "shape": [92.0, 72, 3],
            "dtype": "float64",
            "raw_sha256": "a" * 64,
        }
    }
    with pytest.raises(ValueError, match="shape/dtype"):
        aggregate._validate_fit_array_schema_for_status(
            "R2R1_CONDITIONAL_OOF_FAILED", schema
        )


def test_design_audit_failure_precedes_parser_and_fit_svd(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    design = np.zeros((92, 72, 3, 65), dtype="<f8")
    fixed = np.zeros((92, 72, 3), dtype="<f8")
    monkeypatch.setattr(
        linear,
        "validate_attempt3_completed_recovery",
        lambda *args, **kwargs: (
            {"status": "synthetic label-blind recovery"},
            {
                "thermal_parameter_force_design_eV_A": design,
                "thermal_fixed_force_eV_A": fixed,
            },
        ),
    )
    monkeypatch.setattr(linear, "all_split_design_audits", lambda value: {"pass": False})
    calls = {"parser": 0, "fit": 0}

    def forbidden(kind):
        def fail(*args, **kwargs):
            calls[kind] += 1
            raise AssertionError(f"{kind} reached after failed design audit")

        return fail

    monkeypatch.setattr(
        linear, "_load_thermal92_force_labels_streaming_authorized", forbidden("parser")
    )
    monkeypatch.setattr(linear, "_fit_weighted_ridge", forbidden("fit"))
    with pytest.raises(ValueError, match="before label access"):
        linear.authorized_fit_pipeline(
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert calls == {"parser": 0, "fit": 0}


@pytest.mark.parametrize(
    "tamper",
    [
        "write_parent_float",
        "alpha_int",
        "safety_zero",
        "attempt3_float",
        "attempt3_stale",
    ],
)
def test_manifest_type_drift_fails_before_thermal_open_parser_or_svd(
    tmp_path: Path, monkeypatch, tamper: str
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    payload = linear.freeze_manifest_payload()
    if tamper == "write_parent_float":
        binding = payload["result_parent_binding"]
        integer = binding["chain"][-1]["st_dev"]
        binding["chain"][-1]["st_dev"] = float(integer)
        binding["directory_identity"]["st_dev"] = float(integer)
    elif tamper == "alpha_int":
        assert payload["alpha_grid"][-1] == 100.0
        payload["alpha_grid"][-1] = 100
    elif tamper in {"attempt3_float", "attempt3_stale"}:
        binding = payload["attempt3_root_binding"]
        integer = binding["chain"][-1]["st_ino"]
        replacement = float(integer) if tamper == "attempt3_float" else integer + 1
        binding["chain"][-1]["st_ino"] = replacement
        binding["directory_identity"]["st_ino"] = replacement
    else:
        assert payload["energy_labels_authorized"] is False
        payload["energy_labels_authorized"] = 0
    manifest = control / "freeze_manifest.json"
    manifest.write_bytes(linear.canonical_json_bytes(payload))
    marker = control / "R2R1_FORMAL_GO"
    marker.write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    calls = {"thermal_open": 0, "parser": 0, "svd": 0}
    original_open = linear._open_bound_regular_file

    def count_thermal_open(path, *args, **kwargs):
        if Path(path) == linear.RECOMMENDED_THERMAL92:
            calls["thermal_open"] += 1
        return original_open(path, *args, **kwargs)

    def forbidden(kind):
        def fail(*args, **kwargs):
            calls[kind] += 1
            raise AssertionError(f"{kind} reached after manifest type drift")

        return fail

    monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
    monkeypatch.setattr(
        linear, "_load_thermal92_force_labels_streaming_authorized", forbidden("parser")
    )
    monkeypatch.setattr(linear.np.linalg, "svd", forbidden("svd"))
    with pytest.raises((PermissionError, ValueError), match="changed|identity|differs"):
        linear.authorized_fit_pipeline(
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert calls == {"thermal_open": 0, "parser": 0, "svd": 0}


def test_manifest_rejects_cross_snapshot_result_parent_before_label_access(
    tmp_path: Path, monkeypatch
) -> None:
    scope = tmp_path / "scope"
    scope.mkdir()
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, scope
    )
    control.mkdir()
    payload = linear.freeze_manifest_payload()
    backup = tmp_path / "scope_a"
    scope.rename(backup)
    scope.mkdir()
    try:
        snapshot_b = linear.result_parent_binding()
    finally:
        scope.rmdir()
        backup.rename(scope)
    payload["result_parent_binding"] = snapshot_b
    manifest = control / "freeze_manifest.json"
    manifest.write_bytes(linear.canonical_json_bytes(payload))
    marker = control / "R2R1_FORMAL_GO"
    marker.write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    calls = {"thermal_open": 0, "parser": 0, "svd": 0}
    original_open = linear._open_bound_regular_file

    def count_open(path, *args, **kwargs):
        if Path(path) == linear.RECOMMENDED_THERMAL92:
            calls["thermal_open"] += 1
        return original_open(path, *args, **kwargs)

    def forbidden(kind):
        def fail(*args, **kwargs):
            calls[kind] += 1
            raise AssertionError(f"{kind} reached after mixed result-parent snapshot")

        return fail

    monkeypatch.setattr(linear, "_open_bound_regular_file", count_open)
    monkeypatch.setattr(
        linear, "_load_thermal92_force_labels_streaming_authorized", forbidden("parser")
    )
    monkeypatch.setattr(linear.np.linalg, "svd", forbidden("svd"))
    with pytest.raises(PermissionError, match="shared prefix"):
        linear.authorized_fit_pipeline(
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert calls == {"thermal_open": 0, "parser": 0, "svd": 0}


def test_self_consistent_signed_zero_array_rewrite_fails_bit_exact_resolve(
    tmp_path: Path, monkeypatch
) -> None:
    status = "R2R1_CONDITIONAL_OOF_FAILED"
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch, status)
    )
    aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    arrays_path = output / aggregate.FIT_ARRAYS_BASENAME
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    member = arrays["OOF_predicted_force_eV_A"]
    assert not np.signbit(member.reshape(-1)[0])
    member.reshape(-1)[0] = -0.0
    np.savez(arrays_path, **arrays)
    arrays_raw = arrays_path.read_bytes()
    arrays_sha = hashlib.sha256(arrays_raw).hexdigest()

    fit_receipt_path = output / aggregate.FIT_RECEIPT_BASENAME
    fit_receipt = json.loads(fit_receipt_path.read_text(encoding="ascii"))
    fit_receipt["fit_arrays"]["sha256"] = arrays_sha
    fit_receipt["fit_arrays"]["schema"]["OOF_predicted_force_eV_A"][
        "raw_sha256"
    ] = linear.raw_array_sha256(member, "<f8")
    fit_receipt_path.write_bytes(linear.canonical_json_bytes(fit_receipt))
    fit_receipt_sha = hashlib.sha256(fit_receipt_path.read_bytes()).hexdigest()

    completion_path = output / aggregate.COMPLETION_BASENAME
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    completion["fit_arrays_sha256"] = arrays_sha
    completion["fit_receipt_sha256"] = fit_receipt_sha
    without_schema = {
        key: value
        for key, value in completion.items()
        if key != "recursive_schema_sha256"
    }
    completion["recursive_schema_sha256"] = aggregate._recursive_schema_sha256(
        without_schema
    )
    completion_path.write_bytes(linear.canonical_json_bytes(completion))
    completion_sha = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    (output / "DONE").write_text(
        status + "\n" + completion_sha + "\n", encoding="ascii"
    )

    release = json.loads(release_manifest.read_text(encoding="ascii"))
    release["expected_completion_sha256"] = completion_sha
    release["expected_fit_receipt_sha256"] = fit_receipt_sha
    release_manifest.write_bytes(linear.canonical_json_bytes(release))
    release_sha = hashlib.sha256(release_manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="DONE identity differs from its external anchor"):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=release_sha,
        )


def test_existing_empty_candidate_is_incomplete_and_never_adopted(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    release_manifest.parent.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert not release_manifest.exists()


def test_output_parent_swap_after_authorization_cannot_redirect_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    fit_parent = tmp_path / "result_parent"
    control = tmp_path / "control_attempt2_retry1"
    for directory in (fit_parent, control):
        directory.mkdir()
    output = fit_parent / "release_attempt2_retry1"
    candidate = fit_parent / "candidate_attempt2_retry1"
    monkeypatch.setattr(linear, "RECOMMENDED_CONTROL_ROOT", control)
    monkeypatch.setattr(linear, "RECOMMENDED_FREEZE_MANIFEST", control / "freeze_manifest.json")
    monkeypatch.setattr(linear, "RECOMMENDED_AUTHORIZATION_MARKER", control / "R2R1_FORMAL_GO")
    monkeypatch.setattr(linear, "RECOMMENDED_FIT_OUTPUT_ROOT", output)
    monkeypatch.setattr(linear, "RECOMMENDED_CANDIDATE_ROOT", candidate)
    monkeypatch.setattr(linear, "RECOMMENDED_RELEASE_MANIFEST", candidate / "release_manifest.json")
    manifest, marker = _real_authorization_files(control)
    original_auth = linear.validate_execution_authorization
    swapped = {"value": False}
    backup = tmp_path / "fit_parent_original"

    def authorize_then_swap(*args, **kwargs):
        receipt = original_auth(*args, **kwargs)
        if not swapped["value"]:
            fit_parent.rename(backup)
            fit_parent.mkdir()
            swapped["value"] = True
        return receipt

    monkeypatch.setattr(linear, "validate_execution_authorization", authorize_then_swap)
    fit_calls = {"count": 0}

    def forbidden_fit(**kwargs):
        fit_calls["count"] += 1
        raise AssertionError("parent swap reached fit")

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_fit)
    with pytest.raises((PermissionError, ValueError), match="binding|differs"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert fit_calls["count"] == 0
    assert not output.exists()
    assert not (backup / output.name).exists()


@pytest.mark.parametrize(
    "state_entries",
    [
        {"RUNNING": b"x"},
        {"FAILED": b"x"},
        {},
        {"RUNNING": b"x", "DONE": b"x"},
    ],
)
def test_launcher_incomplete_states_fail_before_preflight_or_fit(
    tmp_path: Path, monkeypatch, state_entries: dict[str, bytes]
) -> None:
    control, output, _candidate, release_manifest = _patch_production_paths(
        monkeypatch, tmp_path
    )
    output.mkdir()
    for name, raw in state_entries.items():
        (output / name).write_bytes(raw)
    calls = {"preflight": 0, "fit": 0, "anchor": 0}

    def forbidden(name):
        def fail(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} called for incomplete release")

        return fail

    monkeypatch.setattr(launcher.linear, "label_blind_preflight", forbidden("preflight"))
    monkeypatch.setattr(launcher.aggregate, "materialize_authorized_fit", forbidden("fit"))
    monkeypatch.setattr(launcher.aggregate, "anchor_completed_release", forbidden("anchor"))
    receipt = launcher.execute(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=control / "freeze_manifest.json",
        authorization_marker=control / "R2R1_FORMAL_GO",
        release_manifest=release_manifest,
    )
    assert receipt["fit_started"] is False
    assert calls == {"preflight": 0, "fit": 0, "anchor": 0}


@pytest.mark.parametrize(
    "status,with_anchor,expected",
    [
        ("R2R1_CONDITIONAL_OOF_FAILED", True, 0),
        ("R2R1_FINAL_READOUT_FAILED", True, 0),
        ("R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING", True, 0),
        ("R2R1_EXISTING_RUNNING_INCOMPLETE", False, 1),
        ("R2R1_NUMERICAL_INCONCLUSIVE_PREFLIGHT", False, 1),
    ],
)
def test_launcher_main_exit_code_requires_scientific_done_and_anchor(
    tmp_path: Path, monkeypatch, status: str, with_anchor: bool, expected: int
) -> None:
    receipt = {"status": status}
    if with_anchor:
        receipt["external_release_manifest"] = {"sha256": "a" * 64}
    monkeypatch.setattr(launcher, "execute", lambda **kwargs: receipt)
    rc = launcher.main(
        [
            "--output-root",
            str(tmp_path / "output"),
            "--freeze-manifest",
            str(tmp_path / "manifest"),
            "--authorization-marker",
            str(tmp_path / "GO"),
            "--release-manifest",
            str(tmp_path / "release"),
        ]
    )
    assert rc == expected


@pytest.mark.parametrize(
    "collision_kind", ["file", "empty_directory", "partial_directory", "symlink"]
)
def test_preexisting_candidate_fails_before_preflight_authorization_or_fit(
    tmp_path: Path, monkeypatch, collision_kind: str
) -> None:
    control, output, candidate, release_manifest = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    if collision_kind == "file":
        candidate.write_bytes(b"foreign candidate\n")
    elif collision_kind == "empty_directory":
        candidate.mkdir()
    elif collision_kind == "partial_directory":
        candidate.mkdir()
        (candidate / "foreign").write_bytes(b"do not touch\n")
    else:
        target = tmp_path / "foreign-target"
        target.mkdir()
        candidate.symlink_to(target, target_is_directory=True)
    before = os.lstat(candidate)
    calls = {"preflight": 0, "authorization": 0, "fit": 0}

    def forbidden(kind):
        def fail(*args, **kwargs):
            calls[kind] += 1
            raise AssertionError(f"{kind} reached with preexisting candidate")

        return fail

    monkeypatch.setattr(launcher.linear, "label_blind_preflight", forbidden("preflight"))
    monkeypatch.setattr(
        aggregate.linear, "validate_execution_authorization", forbidden("authorization")
    )
    monkeypatch.setattr(
        aggregate.linear, "authorized_fit_pipeline", forbidden("fit")
    )
    receipt = launcher.execute(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
    )
    assert receipt["status"] == "R2R1_EXISTING_CANDIDATE_COLLISION"
    assert calls == {"preflight": 0, "authorization": 0, "fit": 0}
    with pytest.raises(FileExistsError, match="candidate root"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert calls == {"preflight": 0, "authorization": 0, "fit": 0}
    after = os.lstat(candidate)
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )
    assert not output.exists()


@pytest.mark.parametrize("interface", ["launcher", "run_cli"])
@pytest.mark.parametrize("bad_target", ["wrong", "overlap"])
def test_fit_cli_rejects_release_path_before_preflight_or_materialize(
    tmp_path: Path, monkeypatch, interface: str, bad_target: str
) -> None:
    control, output, _candidate, release_manifest = _patch_production_paths(
        monkeypatch, tmp_path
    )
    bad_release = (
        tmp_path / "wrong-release.json"
        if bad_target == "wrong"
        else output / "release_manifest.json"
    )
    calls = {"preflight": 0, "materialize": 0}

    def forbidden(kind):
        def fail(*args, **kwargs):
            calls[kind] += 1
            raise AssertionError(f"{kind} reached after invalid release path")

        return fail

    monkeypatch.setattr(launcher.linear, "label_blind_preflight", forbidden("preflight"))
    monkeypatch.setattr(
        aggregate, "materialize_authorized_fit", forbidden("materialize")
    )
    arguments = {
        "output_root": output,
        "attempt3_root": ATTEMPT3,
        "thermal92_path": linear.RECOMMENDED_THERMAL92,
        "freeze_manifest": control / "freeze_manifest.json",
        "authorization_marker": control / "R2R1_FORMAL_GO",
        "release_manifest": bad_release,
    }
    with pytest.raises(PermissionError, match="release manifest"):
        if interface == "launcher":
            launcher.execute(**arguments)
        else:
            run_cli.main(
                [
                    "fit",
                    "--output-root",
                    str(output),
                    "--attempt3-root",
                    str(ATTEMPT3),
                    "--thermal92",
                    str(linear.RECOMMENDED_THERMAL92),
                    "--freeze-manifest",
                    str(control / "freeze_manifest.json"),
                    "--authorization-marker",
                    str(control / "R2R1_FORMAL_GO"),
                    "--release-manifest",
                    str(bad_release),
                ]
            )
    assert calls == {"preflight": 0, "materialize": 0}
    assert not output.exists()


@pytest.mark.parametrize("terminal", ["DONE", "FAILED"])
def test_terminal_long_validator_cannot_replace_held_running_inode(
    tmp_path: Path, monkeypatch, terminal: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    swapped = {"value": False}
    if terminal == "DONE":
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED"),
        )
        original_validator = aggregate._validate_success_content_fd

        def validator(*args, **kwargs):
            result = original_validator(*args, **kwargs)
            if kwargs.get("terminal_name") == "RUNNING" and not swapped["value"]:
                raw = (output / "RUNNING").read_bytes()
                (output / "RUNNING").unlink()
                (output / "RUNNING").write_bytes(raw)
                swapped["value"] = True
            return result

        monkeypatch.setattr(aggregate, "_validate_success_content_fd", validator)
        expected_error = "terminal marker at rename boundary changed"
    else:
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic solve")),
        )
        original_validator = aggregate._validate_failure_content_fd

        def validator(*args, **kwargs):
            result = original_validator(*args, **kwargs)
            if kwargs.get("terminal_name") == "RUNNING" and not swapped["value"]:
                raw = (output / "RUNNING").read_bytes()
                (output / "RUNNING").unlink()
                (output / "RUNNING").write_bytes(raw)
                swapped["value"] = True
            return result

        monkeypatch.setattr(aggregate, "_validate_failure_content_fd", validator)
        expected_error = "synthetic solve"
    with pytest.raises((ValueError, RuntimeError), match=expected_error):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert swapped["value"] is True
    assert (output / "RUNNING").is_file()
    assert not (output / "DONE").exists()
    assert not (output / "FAILED").exists()


@pytest.mark.parametrize("extra_kind", ["file", "directory", "symlink"])
def test_candidate_precommit_extra_never_reaches_canonical_name(
    tmp_path: Path, monkeypatch, extra_kind: str
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_commit = aggregate._commit_candidate_directory_at

    def inject_extra(*args, **kwargs):
        candidate_fd = args[3]
        if extra_kind == "file":
            descriptor = os.open(
                "extra", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=candidate_fd
            )
            os.close(descriptor)
        elif extra_kind == "directory":
            os.mkdir("extra", 0o700, dir_fd=candidate_fd)
        else:
            os.symlink("missing", "extra", dir_fd=candidate_fd)
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(aggregate, "_commit_candidate_directory_at", inject_extra)
    with pytest.raises(ValueError, match="inventory"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert not release_manifest.parent.exists()


def test_candidate_final_source_rebind_blocks_exact_replacement_directory(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_read = aggregate._read_regular_file_at
    swapped = {"value": False}

    def swap_after_final_anchor(directory_fd, name, label, **kwargs):
        raw = original_read(directory_fd, name, label, **kwargs)
        if label == "R2R-1 candidate anchor at no-replace boundary":
            temporary = next(
                path
                for path in release_manifest.parent.parent.iterdir()
                if path.name.startswith(".r2r1-candidate-")
                and path.name.endswith(".tmp")
            )
            backup = temporary.with_name(temporary.name + ".old")
            temporary.rename(backup)
            temporary.mkdir()
            (temporary / name).write_bytes(raw)
            swapped["value"] = True
        return raw

    monkeypatch.setattr(aggregate, "_read_regular_file_at", swap_after_final_anchor)
    with pytest.raises(ValueError, match="canonical basename"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert swapped["value"] is True
    assert not release_manifest.parent.exists()


@pytest.mark.parametrize("explicit_binding", [False, True])
def test_attempt3_long_recovery_swap_rejects_before_replacement_open(
    tmp_path: Path, monkeypatch, explicit_binding: bool
) -> None:
    attempt = tmp_path / "attempt3"
    shutil.copytree(ATTEMPT3, attempt)
    monkeypatch.setattr(linear, "ATTEMPT3_ROOT", attempt)
    if explicit_binding:
        observed_binding = linear.directory_metadata_binding(
            attempt, "test attempt3 binding"
        )
        expected = {
            key: observed_binding[key]
            for key in ("path", "directory_identity", "chain")
        }
    else:
        expected = None
    original_validate = linear.r2r0_launcher._validate_launch_recovery
    original_open = linear.os.open
    backup = tmp_path / "attempt3-original"
    swapped = {"value": False, "opens_at_swap": 0, "replacement_opens": 0}
    parent_stat = os.stat(tmp_path)

    def count_open(path, *args, **kwargs):
        name = os.fsdecode(path) if isinstance(path, (bytes, bytearray)) else str(path)
        if swapped["value"]:
            directory_fd = kwargs.get("dir_fd")
            if name == "external_payload":
                swapped["replacement_opens"] += 1
            elif name == attempt.name and directory_fd is not None:
                parent_observed = os.fstat(directory_fd)
                if (parent_observed.st_dev, parent_observed.st_ino) == (
                    parent_stat.st_dev,
                    parent_stat.st_ino,
                ):
                    swapped["replacement_opens"] += 1
        return original_open(path, *args, **kwargs)

    def validate_then_swap(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        if not swapped["value"]:
            attempt.rename(backup)
            attempt.mkdir()
            (attempt / "external_payload").write_bytes(b"must not be opened\n")
            swapped["value"] = True
        return result

    monkeypatch.setattr(linear.os, "open", count_open)
    monkeypatch.setattr(
        linear.r2r0_launcher, "_validate_launch_recovery", validate_then_swap
    )
    try:
        with pytest.raises(ValueError, match="identity changed"):
            linear.validate_attempt3_completed_recovery(
                attempt, expected_root_binding=expected
            )
        assert swapped["value"] is True
        assert swapped["replacement_opens"] == 0
    finally:
        if swapped["value"]:
            (attempt / "external_payload").unlink()
            attempt.rmdir()
            backup.rename(attempt)


def test_recovery_detects_checkpoint_same_bytes_rewrite_identity_change(
    tmp_path: Path, monkeypatch
) -> None:
    status = "R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING"
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch, status)
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    pipeline = _mock_pipeline_result(status)
    rewritten = {"value": False}

    def rewrite_same_bytes(**kwargs):
        if not rewritten["value"]:
            path = output / "checkpoint" / frozen.CHECKPOINT_MARKER_BASENAME
            before = os.stat(path)
            raw = path.read_bytes()
            path.write_bytes(raw)
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            assert path.read_bytes() == raw
            rewritten["value"] = True
        return pipeline

    monkeypatch.setattr(
        aggregate.linear, "authorized_fit_pipeline", rewrite_same_bytes
    )
    with pytest.raises(ValueError, match="release content changed"):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )
    assert rewritten["value"] is True


def test_attempt3_capture_globally_rebinds_closed_sibling_subtree(
    tmp_path: Path, monkeypatch
) -> None:
    tree = tmp_path / "capture_root"
    early = tree / "a" / "deep" / "early.bin"
    late = tree / "z" / "late.bin"
    early.parent.mkdir(parents=True)
    late.parent.mkdir(parents=True)
    early.write_bytes(b"early\n")
    late.write_bytes(b"late\n")
    late_identity = os.stat(late)
    original_read = linear.os.read
    mutated = {"value": False}

    def mutate_early_while_reading_late(descriptor, size):
        current = os.fstat(descriptor)
        if (
            not mutated["value"]
            and current.st_dev == late_identity.st_dev
            and current.st_ino == late_identity.st_ino
        ):
            before = os.stat(early)
            raw = early.read_bytes()
            early.write_bytes(raw)
            os.utime(
                early,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            mutated["value"] = True
        return original_read(descriptor, size)

    monkeypatch.setattr(linear.os, "read", mutate_early_while_reading_late)
    with pytest.raises(ValueError, match="global rebind"):
        linear._fd_bound_tree_capture(tree)
    assert mutated["value"] is True


def test_auth_rechecks_go_after_attempt3_rebind_before_thermal_open(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    original_verify = linear._verify_directory_chain
    original_open = linear._open_bound_regular_file
    tampered = {"value": False}
    thermal_opens = {"count": 0}

    def verify_then_tamper(path, label, expected_chain):
        result = original_verify(path, label, expected_chain)
        if label == "R2R-1 held attempt3 lexical rebind" and not tampered["value"]:
            before = os.stat(marker)
            marker.write_bytes(marker.read_bytes())
            os.utime(
                marker,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            tampered["value"] = True
        return result

    def count_thermal_open(path, label, **kwargs):
        if label == "R2R-1 post-GO thermal92 binding":
            thermal_opens["count"] += 1
        return original_open(path, label, **kwargs)

    monkeypatch.setattr(linear, "_verify_directory_chain", verify_then_tamper)
    monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
    with pytest.raises(PermissionError, match="held|identity|authorization"):
        linear.validate_execution_authorization(manifest, marker)
    assert tampered["value"] is True
    assert thermal_opens["count"] == 0


def test_auth_manifest_owned8_allows_ctime_only_rewrite_with_exact_raw(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    descriptor = os.open(manifest, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        _rewrite_same_bytes_and_restore_mtime(descriptor)
    finally:
        os.close(descriptor)
    receipt = linear.validate_execution_authorization(manifest, marker)
    assert set(receipt["freeze_manifest_file_identity"]) == (
        aggregate.OWNED_REGULAR_IDENTITY_KEYS
    )
    assert "st_ctime_ns" not in receipt["freeze_manifest_file_identity"]


def test_auth_external_go_full9_rejects_ctime_only_rewrite_before_thermal_open(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    original_verify = linear._verify_directory_chain
    original_open = linear._open_bound_regular_file
    injected = {"value": False}
    thermal_opens = {"count": 0}

    def rebind_then_rewrite_marker(path, label, expected_chain):
        result = original_verify(path, label, expected_chain)
        if label == "R2R-1 held attempt3 lexical rebind" and not injected["value"]:
            descriptor = os.open(
                marker, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                _rewrite_same_bytes_and_restore_mtime(descriptor)
            finally:
                os.close(descriptor)
            injected["value"] = True
        return result

    def count_thermal_open(path, label, **kwargs):
        if label == "R2R-1 post-GO thermal92 binding":
            thermal_opens["count"] += 1
        return original_open(path, label, **kwargs)

    monkeypatch.setattr(linear, "_verify_directory_chain", rebind_then_rewrite_marker)
    monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
    with pytest.raises(PermissionError, match="marker|control|identity"):
        linear.validate_execution_authorization(manifest, marker)
    assert injected["value"] is True
    assert thermal_opens["count"] == 0


@pytest.mark.parametrize("child", ["fit", "candidate"])
def test_fresh_authorization_rechecks_result_children_before_thermal_open(
    tmp_path: Path, monkeypatch, child: str
) -> None:
    control, output, candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    injected_path = output if child == "fit" else candidate
    original_verify = linear._verify_directory_chain
    original_open = linear._open_bound_regular_file
    injected = {"value": False}
    thermal_opens = {"count": 0}

    def verify_then_inject(path, label, expected_chain):
        result = original_verify(path, label, expected_chain)
        if label == "R2R-1 held attempt3 lexical rebind" and not injected["value"]:
            injected_path.mkdir()
            injected["value"] = True
        return result

    def count_thermal_open(path, label, **kwargs):
        if label == "R2R-1 post-GO thermal92 binding":
            thermal_opens["count"] += 1
        return original_open(path, label, **kwargs)

    monkeypatch.setattr(linear, "_verify_directory_chain", verify_then_inject)
    monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
    with pytest.raises(FileExistsError, match="result child"):
        linear.validate_execution_authorization(
            manifest,
            marker,
            require_fresh_result_children=True,
        )
    assert injected["value"] is True
    assert thermal_opens["count"] == 0


@pytest.mark.parametrize(
    "status,target_kind",
    [
        ("R2R1_CONDITIONAL_OOF_FAILED", "fit_manifest"),
        (
            "R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING",
            "checkpoint_marker",
        ),
    ],
)
def test_release_snapshot_second_pass_catches_late_earlier_artifact_rewrite(
    tmp_path: Path, monkeypatch, status: str, target_kind: str
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch, status)
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    pipeline = _mock_pipeline_result(status)
    original_read = aggregate._root_file
    armed = {"value": False}
    rewritten = {"value": False}

    def resolve_then_arm(**kwargs):
        armed["value"] = True
        return pipeline

    def rewrite_earlier_when_terminal_is_read(root_fd, name, label):
        raw = original_read(root_fd, name, label)
        if (
            armed["value"]
            and not rewritten["value"]
            and label == "R2R-1 release snapshot DONE"
        ):
            target = (
                output / aggregate.FIT_MANIFEST_BASENAME
                if target_kind == "fit_manifest"
                else output / aggregate.CHECKPOINT_DIRNAME / frozen.CHECKPOINT_MARKER_BASENAME
            )
            descriptor = os.open(
                target, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                _rewrite_different_same_size_and_restore_mtime(descriptor)
            finally:
                os.close(descriptor)
            rewritten["value"] = True
        return raw

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", resolve_then_arm)
    monkeypatch.setattr(aggregate, "_root_file", rewrite_earlier_when_terminal_is_read)
    with pytest.raises(
        ValueError,
        match="bytes changed|release content changed|checkpoint child changed",
    ):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )
    assert rewritten["value"] is True


def test_failure_validator_rebinds_earlier_receipt_after_later_read(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    original_read = aggregate._root_file
    rewritten = {"value": False}

    def rewrite_failure_receipt_after_first_read(root_fd, name, label):
        raw = original_read(root_fd, name, label)
        if label == "R2R-1 failure EXIT_CODE" and not rewritten["value"]:
            path = output / "failure.json"
            before = os.stat(path)
            receipt_raw = path.read_bytes()
            path.write_bytes(receipt_raw)
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            rewritten["value"] = True
        return raw

    monkeypatch.setattr(aggregate, "_root_file", rewrite_failure_receipt_after_first_read)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert rewritten["value"] is True
    assert (output / "RUNNING").is_file()
    assert not (output / "FAILED").exists()


def test_auth_joint_control_sibling_pass_catches_manifest_change_during_marker_read(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    marker_stat = os.stat(marker)
    original_read = linear.os.read
    original_open = linear._open_bound_regular_file
    marker_payload_reads = {"count": 0}
    mutated = {"value": False}
    thermal_opens = {"count": 0}

    def mutate_manifest_during_held_marker_read(descriptor, size):
        block = original_read(descriptor, size)
        observed = os.fstat(descriptor)
        if (
            block
            and observed.st_dev == marker_stat.st_dev
            and observed.st_ino == marker_stat.st_ino
        ):
            marker_payload_reads["count"] += 1
            if marker_payload_reads["count"] == 3 and not mutated["value"]:
                before = os.stat(manifest)
                raw = manifest.read_bytes()
                manifest.write_bytes(raw)
                os.utime(
                    manifest,
                    ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
                )
                mutated["value"] = True
        return block

    def count_thermal_open(path, label, **kwargs):
        if label == "R2R-1 post-GO thermal92 binding":
            thermal_opens["count"] += 1
        return original_open(path, label, **kwargs)

    monkeypatch.setattr(linear.os, "read", mutate_manifest_during_held_marker_read)
    monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
    with pytest.raises(PermissionError, match="control sibling|identity"):
        linear.validate_execution_authorization(manifest, marker)
    assert mutated["value"] is True
    assert thermal_opens["count"] == 0


def test_publication_walk_manifest_tamper_precedes_thermal_metadata_open(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    boundary = object()
    mutated = {"value": False}
    thermal_opens = {"count": 0}
    original_open = linear._open_bound_regular_file

    def publication_walk_then_tamper(value):
        assert value is boundary
        if not mutated["value"]:
            descriptor = os.open(
                manifest, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                _rewrite_different_same_size_and_restore_mtime(descriptor)
            finally:
                os.close(descriptor)
            mutated["value"] = True

    def count_thermal_open(path, label, **kwargs):
        if label == "R2R-1 post-GO thermal92 binding":
            thermal_opens["count"] += 1
        return original_open(path, label, **kwargs)

    monkeypatch.setattr(
        linear, "_require_fit_publication_boundary", publication_walk_then_tamper
    )
    monkeypatch.setattr(linear, "_close_fit_publication_boundary", lambda value: None)
    monkeypatch.setattr(
        linear, "_watch_fit_publication_boundary", lambda value, guard: None
    )
    monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
    with pytest.raises(PermissionError, match="post-publication freeze manifest"):
        linear.validate_execution_authorization(
            manifest, marker, publication_boundary=boundary
        )
    assert mutated["value"] is True
    assert thermal_opens["count"] == 0


def test_authorization_require_publication_tamper_precedes_payload_fd_open(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    boundary = object()
    state = {"armed": False, "mutated": False}

    def publication_walk(value):
        assert value is boundary
        if state["armed"] and not state["mutated"]:
            descriptor = os.open(
                manifest, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                _rewrite_different_same_size_and_restore_mtime(descriptor)
            finally:
                os.close(descriptor)
            state["mutated"] = True

    monkeypatch.setattr(linear, "_require_fit_publication_boundary", publication_walk)
    monkeypatch.setattr(linear, "_close_fit_publication_boundary", lambda value: None)
    monkeypatch.setattr(
        linear, "_watch_fit_publication_boundary", lambda value, guard: None
    )
    authorization = linear._issue_production_fit_authorization(
        manifest, marker, publication_boundary=boundary
    )
    payload_opens = {"count": 0}
    original_payload_open = linear._open_regular_file_fd

    def count_payload_open(path, label, **kwargs):
        payload_opens["count"] += 1
        return original_payload_open(path, label, **kwargs)

    state["armed"] = True
    monkeypatch.setattr(linear, "_open_regular_file_fd", count_payload_open)
    with pytest.raises(PermissionError, match="post-publication freeze manifest"):
        linear._load_thermal92_force_labels_streaming_authorized(
            linear.RECOMMENDED_THERMAL92,
            authorization=authorization,
        )
    assert state["mutated"] is True
    assert payload_opens["count"] == 0


def test_auth_final_source_closure_catches_earlier_basename_replacement(
    tmp_path: Path, monkeypatch
) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    copied_sources = safe_root / "sources"
    copied_sources.mkdir()
    source_paths: dict[str, Path] = {}
    for name, source in linear.R2R1_SOURCE_PATHS.items():
        destination = copied_sources / f"{name}.src"
        shutil.copyfile(source, destination)
        source_paths[name] = destination
    monkeypatch.setattr(linear, "R2R1_SOURCE_PATHS", source_paths)
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, safe_root
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    first_name = sorted(source_paths)[0]
    last_name = sorted(source_paths)[-1]
    earlier = source_paths[first_name]
    original_verify = linear._verify_directory_chain
    original_open = linear._open_bound_regular_file
    replaced = {"value": False}
    thermal_opens = {"count": 0}

    def replace_earlier_at_later_final_parent(path, label, expected_chain):
        result = original_verify(path, label, expected_chain)
        if (
            label == f"R2R-1 final closure parent rebind source.{last_name}"
            and not replaced["value"]
        ):
            raw = earlier.read_bytes()
            backup = earlier.with_suffix(".old")
            earlier.rename(backup)
            earlier.write_bytes(raw)
            replaced["value"] = True
        return result

    def count_thermal_open(path, label, **kwargs):
        if label == "R2R-1 post-GO thermal92 binding":
            thermal_opens["count"] += 1
        return original_open(path, label, **kwargs)

    monkeypatch.setattr(
        linear, "_verify_directory_chain", replace_earlier_at_later_final_parent
    )
    monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
    with pytest.raises(PermissionError, match="closure identity"):
        linear.validate_execution_authorization(manifest, marker)
    assert replaced["value"] is True
    assert thermal_opens["count"] == 0


def test_parser_revalidates_authorization_after_thermal_open_before_payload_read(
    tmp_path: Path, monkeypatch
) -> None:
    thermal = tmp_path / "thermal_train.xyz"
    thermal.write_bytes(b"payload must remain unread\n")
    monkeypatch.setattr(linear, "RECOMMENDED_THERMAL92", thermal)
    monkeypatch.setattr(
        linear, "THERMAL92_FILE_SHA256", hashlib.sha256(thermal.read_bytes()).hexdigest()
    )
    authorization = _real_test_authorization(tmp_path, monkeypatch)
    marker = authorization.authorization_marker
    thermal_stat = os.stat(thermal)
    original_require = linear._require_fit_authorization
    original_read = linear.os.read
    require_calls = {"count": 0}
    thermal_payload_reads = {"count": 0}

    def require_then_invalidate(value):
        require_calls["count"] += 1
        result = original_require(value)
        if require_calls["count"] == 1:
            before = os.stat(marker)
            marker.write_bytes(marker.read_bytes())
            os.utime(
                marker,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
        return result

    def count_payload_read(descriptor, size):
        observed = os.fstat(descriptor)
        if (
            observed.st_dev == thermal_stat.st_dev
            and observed.st_ino == thermal_stat.st_ino
        ):
            thermal_payload_reads["count"] += 1
        return original_read(descriptor, size)

    monkeypatch.setattr(linear, "_require_fit_authorization", require_then_invalidate)
    monkeypatch.setattr(linear.os, "read", count_payload_read)
    with pytest.raises(PermissionError, match="authorization|identity|control"):
        linear._load_thermal92_force_labels_streaming_authorized(
            thermal, authorization=authorization
        )
    assert require_calls["count"] >= 2
    assert thermal_payload_reads["count"] == 0


def test_thermal_metadata_file_identity_schema_fails_before_final_open(
    tmp_path: Path, monkeypatch
) -> None:
    thermal = tmp_path / "thermal_train.xyz"
    thermal.write_bytes(b"not opened\n")
    monkeypatch.setattr(linear, "RECOMMENDED_THERMAL92", thermal)
    monkeypatch.setattr(
        linear, "THERMAL92_FILE_SHA256", hashlib.sha256(thermal.read_bytes()).hexdigest()
    )
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    payload = linear.freeze_manifest_payload()
    payload["thermal92_metadata_binding"]["file_identity"] = None
    manifest = control / "freeze_manifest.json"
    marker = control / "R2R1_FORMAL_GO"
    manifest.write_bytes(linear.canonical_json_bytes(payload))
    marker.write_bytes(hashlib.sha256(manifest.read_bytes()).hexdigest().encode() + b"\n")
    original_open = linear._open_bound_regular_file
    thermal_opens = {"count": 0}

    def count_thermal_open(path, label, **kwargs):
        if label == "R2R-1 post-GO thermal92 binding":
            thermal_opens["count"] += 1
        return original_open(path, label, **kwargs)

    monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
    with pytest.raises(PermissionError, match="thermal92 metadata binding schema"):
        linear.validate_execution_authorization(manifest, marker)
    assert thermal_opens["count"] == 0


@pytest.mark.parametrize("terminal", ["success", "failure"])
def test_terminal_post_callback_identity_closure_blocks_extra_before_rename(
    tmp_path: Path, monkeypatch, terminal: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    injected = {"value": False}
    if terminal == "success":
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED"),
        )
        original_validator = aggregate._validate_success_content_fd

        def validator(*args, **kwargs):
            result = original_validator(*args, **kwargs)
            if kwargs.get("terminal_name") == "RUNNING" and not injected["value"]:
                (output / "post_callback_extra").write_bytes(b"foreign\n")
                injected["value"] = True
            return result

        monkeypatch.setattr(aggregate, "_validate_success_content_fd", validator)
        expected_error = (ValueError,)
    else:
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
        )
        original_validator = aggregate._validate_failure_content_fd

        def validator(*args, **kwargs):
            result = original_validator(*args, **kwargs)
            if kwargs.get("terminal_name") == "RUNNING" and not injected["value"]:
                (output / "post_callback_extra").write_bytes(b"foreign\n")
                injected["value"] = True
            return result

        monkeypatch.setattr(aggregate, "_validate_failure_content_fd", validator)
        expected_error = (RuntimeError,)
    with pytest.raises(expected_error):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert injected["value"] is True
    assert (output / "RUNNING").is_file()
    assert not (output / "DONE").exists()
    assert not (output / "FAILED").exists()


@pytest.mark.parametrize("terminal", ["success", "failure"])
def test_terminal_post_callback_byte_closure_precedes_rename(
    tmp_path: Path, monkeypatch, terminal: str
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    injected = {"value": False}
    rename_calls = {"count": 0}
    original_rename = aggregate._rename_noreplace_at

    def count_rename(*args, **kwargs):
        rename_calls["count"] += 1
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(aggregate, "_rename_noreplace_at", count_rename)
    if terminal == "success":
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED"),
        )
        original_validator = aggregate._validate_success_content_fd
        target_name = aggregate.FIT_MANIFEST_BASENAME

        def validator(*args, **kwargs):
            result = original_validator(*args, **kwargs)
            if kwargs.get("terminal_name") == "RUNNING" and not injected["value"]:
                descriptor = os.open(output / target_name, os.O_RDWR)
                try:
                    _rewrite_different_same_size_and_restore_mtime(descriptor)
                finally:
                    os.close(descriptor)
                injected["value"] = True
            return result

        monkeypatch.setattr(aggregate, "_validate_success_content_fd", validator)
    else:
        monkeypatch.setattr(
            aggregate.linear,
            "authorized_fit_pipeline",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
        )
        original_validator = aggregate._validate_failure_content_fd
        target_name = "failure.json"

        def validator(*args, **kwargs):
            result = original_validator(*args, **kwargs)
            if kwargs.get("terminal_name") == "RUNNING" and not injected["value"]:
                descriptor = os.open(output / target_name, os.O_RDWR)
                try:
                    _rewrite_different_same_size_and_restore_mtime(descriptor)
                finally:
                    os.close(descriptor)
                injected["value"] = True
            return result

        monkeypatch.setattr(aggregate, "_validate_failure_content_fd", validator)
    with pytest.raises((ValueError, RuntimeError)):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert injected["value"] is True
    assert rename_calls["count"] == 0
    assert (output / "RUNNING").is_file()
    assert not (output / "DONE").exists()
    assert not (output / "FAILED").exists()


def test_candidate_callback_extra_is_rejected_before_directory_publication(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_commit = aggregate._commit_candidate_directory_at
    injected = {"value": False}

    def wrap_commit(*args, **kwargs):
        candidate_fd = args[3]
        original_callback = kwargs["precommit_validator"]

        def callback_then_inject():
            original_callback()
            descriptor = os.open(
                "callback_extra",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=candidate_fd,
            )
            os.close(descriptor)
            injected["value"] = True

        kwargs["precommit_validator"] = callback_then_inject
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(aggregate, "_commit_candidate_directory_at", wrap_commit)
    with pytest.raises(ValueError, match="inventory changed"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert injected["value"] is True
    assert not release_manifest.parent.exists()


def _rewrite_same_bytes_with_new_mtime(path: Path) -> None:
    before = os.stat(path)
    raw = path.read_bytes()
    path.write_bytes(raw)
    os.utime(
        path,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )


def test_materializer_does_not_reset_baseline_after_last_internal_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: _mock_pipeline_result("R2R1_CONDITIONAL_OOF_FAILED"),
    )
    original_recovery = aggregate._recover_fit_release_bound
    calls = {"count": 0}

    def rewrite_after_second_recovery(**kwargs):
        result = original_recovery(**kwargs)
        calls["count"] += 1
        if calls["count"] == 2:
            _rewrite_same_bytes_with_new_mtime(
                output / aggregate.FIT_MANIFEST_BASENAME
            )
        return result

    monkeypatch.setattr(
        aggregate, "_recover_fit_release_bound", rewrite_after_second_recovery
    )
    with pytest.raises(ValueError, match="baseline changed"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert calls["count"] == 2
    assert (output / "RUNNING").is_file()
    assert not (output / "DONE").exists()


def test_anchor_does_not_reset_release_baseline_after_inner_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_recovery = aggregate._recover_fit_release_bound
    rewritten = {"value": False}

    def rewrite_after_recovery(**kwargs):
        result = original_recovery(**kwargs)
        if not rewritten["value"]:
            _rewrite_same_bytes_with_new_mtime(
                output / aggregate.FIT_MANIFEST_BASENAME
            )
            rewritten["value"] = True
        return result

    monkeypatch.setattr(aggregate, "_recover_fit_release_bound", rewrite_after_recovery)
    with pytest.raises(ValueError, match="pre-recovery baseline"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert rewritten["value"] is True
    assert not release_manifest.parent.exists()


def test_public_recovery_does_not_reset_baseline_after_inner_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    original_recovery = aggregate._recover_fit_release_bound
    rewritten = {"value": False}

    def rewrite_after_recovery(**kwargs):
        result = original_recovery(**kwargs)
        if not rewritten["value"]:
            _rewrite_same_bytes_with_new_mtime(
                output / aggregate.FIT_MANIFEST_BASENAME
            )
            rewritten["value"] = True
        return result

    monkeypatch.setattr(aggregate, "_recover_fit_release_bound", rewrite_after_recovery)
    with pytest.raises(ValueError, match="frozen baseline"):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )
    assert rewritten["value"] is True


def test_candidate_helper_closes_release_bytes_after_callback_returns(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_commit = aggregate._commit_candidate_directory_at
    rewritten = {"value": False}

    def wrap_commit(*args, **kwargs):
        original_callback = kwargs["precommit_validator"]

        def callback_then_rewrite_release():
            result = original_callback()
            descriptor = os.open(
                output / aggregate.FIT_MANIFEST_BASENAME,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                _rewrite_different_same_size_and_restore_mtime(descriptor)
            finally:
                os.close(descriptor)
            rewritten["value"] = True
            return result

        kwargs["precommit_validator"] = callback_then_rewrite_release
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(aggregate, "_commit_candidate_directory_at", wrap_commit)
    with pytest.raises(ValueError, match="ledger-bound artifact bytes changed"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert rewritten["value"] is True
    assert not release_manifest.parent.exists()


def test_candidate_helper_rereads_anchor_after_final_release_byte_closure(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_verify = aggregate._verify_snapshot_bytes_fd
    original_rename = aggregate._rename_noreplace_at
    injected = {"value": False}
    rename_calls = {"count": 0}

    def verify_then_tamper_anchor(root_fd, snapshot):
        result = original_verify(root_fd, snapshot)
        caller = inspect.currentframe().f_back
        if (
            caller is not None
            and caller.f_code.co_name == "_commit_candidate_directory_at_guarded"
            and not injected["value"]
        ):
            candidate_fd = caller.f_locals["candidate_fd"]
            anchor_name = caller.f_locals["anchor_name"]
            descriptor = os.open(anchor_name, os.O_RDWR, dir_fd=candidate_fd)
            try:
                _rewrite_different_same_size_and_restore_mtime(descriptor)
            finally:
                os.close(descriptor)
            injected["value"] = True
        return result

    def count_rename(*args, **kwargs):
        rename_calls["count"] += 1
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(aggregate, "_verify_snapshot_bytes_fd", verify_then_tamper_anchor)
    monkeypatch.setattr(aggregate, "_rename_noreplace_at", count_rename)
    with pytest.raises(ValueError, match="anchor bytes changed during final release"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert injected["value"] is True
    assert rename_calls["count"] == 0
    assert not release_manifest.parent.exists()


def test_anchor_return_rereads_published_anchor_after_release_byte_closure(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_verify = aggregate._verify_snapshot_bytes_fd
    direct_calls = {"count": 0}
    injected = {"value": False}

    def verify_then_tamper_published_anchor(root_fd, snapshot):
        result = original_verify(root_fd, snapshot)
        caller = inspect.currentframe().f_back
        if (
            caller is not None
            and caller.f_code.co_name == "_verify_snapshot_and_anchor_bytes_fd"
            and caller.f_locals.get("label") == "R2R-1 anchor return"
        ):
            direct_calls["count"] += 1
            if direct_calls["count"] == 1 and not injected["value"]:
                descriptor = os.open(release_manifest, os.O_RDWR)
                try:
                    _rewrite_different_same_size_and_restore_mtime(descriptor)
                finally:
                    os.close(descriptor)
                injected["value"] = True
        return result

    monkeypatch.setattr(
        aggregate, "_verify_snapshot_bytes_fd", verify_then_tamper_published_anchor
    )
    with pytest.raises(
        ValueError,
        match="anchor return candidate anchor bytes changed",
    ):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert injected["value"] is True
    assert release_manifest.is_file()


def test_public_recovery_return_rereads_anchor_after_release_byte_closure(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    original_verify = aggregate._verify_snapshot_bytes_fd
    direct_calls = {"count": 0}
    injected = {"value": False}

    def verify_then_tamper_anchor(root_fd, snapshot):
        result = original_verify(root_fd, snapshot)
        caller = inspect.currentframe().f_back
        if (
            caller is not None
            and caller.f_code.co_name == "_verify_snapshot_and_anchor_bytes_fd"
            and caller.f_locals.get("label") == "R2R-1 recovery return"
        ):
            direct_calls["count"] += 1
            if direct_calls["count"] == 1 and not injected["value"]:
                descriptor = os.open(release_manifest, os.O_RDWR)
                try:
                    _rewrite_different_same_size_and_restore_mtime(descriptor)
                finally:
                    os.close(descriptor)
                injected["value"] = True
        return result

    monkeypatch.setattr(aggregate, "_verify_snapshot_bytes_fd", verify_then_tamper_anchor)
    with pytest.raises(
        ValueError,
        match="recovery return candidate anchor bytes changed",
    ):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )
    assert injected["value"] is True
    assert release_manifest.is_file()


def test_anchor_final_parent_rebind_closes_release_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_verify = aggregate.linear._verify_directory_chain
    injected = {"value": False}

    def verify_then_inject(path, label, expected_chain):
        result = original_verify(path, label, expected_chain)
        if (
            label == "R2R-1 final anchored release-parent binding"
            and not injected["value"]
        ):
            (output / "post_parent_rebind_extra").write_bytes(b"foreign\n")
            injected["value"] = True
        return result

    monkeypatch.setattr(
        aggregate.linear, "_verify_directory_chain", verify_then_inject
    )
    with pytest.raises(ValueError, match="release inventory changed at anchor return"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert injected["value"] is True
    assert (output / "post_parent_rebind_extra").read_bytes() == b"foreign\n"


def test_candidate_precommit_parent_rebind_closes_release_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_verify = aggregate.linear._verify_directory_chain
    injected = {"value": False}

    def verify_then_inject(path, label, expected_chain):
        result = original_verify(path, label, expected_chain)
        if label == "R2R-1 release parent after anchor callback":
            (output / "precommit_extra").write_bytes(b"foreign\n")
            injected["value"] = True
        return result

    monkeypatch.setattr(
        aggregate.linear, "_verify_directory_chain", verify_then_inject
    )
    with pytest.raises(ValueError, match="release inventory changed"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert injected["value"] is True
    assert not release_manifest.parent.exists()


def test_wrong_preexisting_anchor_fails_before_scientific_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    release_manifest.parent.mkdir()
    release_manifest.write_bytes(b"wrong\n")
    calls = {"pipeline": 0, "recovery": 0}

    def forbidden_pipeline(**kwargs):
        calls["pipeline"] += 1
        raise AssertionError("wrong candidate must fail before scientific solve")

    def forbidden_recovery(**kwargs):
        calls["recovery"] += 1
        raise AssertionError("wrong candidate must fail before recovery")

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_pipeline)
    monkeypatch.setattr(aggregate, "_recover_fit_release_bound", forbidden_recovery)
    with pytest.raises(FileExistsError, match="manifest differs"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert calls == {"pipeline": 0, "recovery": 0}
    assert release_manifest.read_bytes() == b"wrong\n"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "special", "oversize"])
def test_invalid_preexisting_anchor_entry_fails_before_scientific_recovery(
    tmp_path: Path, monkeypatch, kind: str
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    release_manifest.parent.mkdir()
    if kind == "symlink":
        release_manifest.symlink_to("missing")
    elif kind == "hardlink":
        source = tmp_path / "anchor_source"
        source.write_bytes(b"wrong\n")
        os.link(source, release_manifest)
    elif kind == "special":
        os.mkfifo(release_manifest)
    else:
        with release_manifest.open("wb") as handle:
            handle.truncate(2 * 1024 * 1024 + 1)
    calls = {"pipeline": 0, "recovery": 0}

    def forbidden_pipeline(**kwargs):
        calls["pipeline"] += 1
        raise AssertionError("invalid candidate must fail before scientific solve")

    def forbidden_recovery(**kwargs):
        calls["recovery"] += 1
        raise AssertionError("invalid candidate must fail before recovery")

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_pipeline)
    monkeypatch.setattr(aggregate, "_recover_fit_release_bound", forbidden_recovery)
    with pytest.raises((OSError, PermissionError, ValueError)):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert calls == {"pipeline": 0, "recovery": 0}


def test_public_recovery_rebinds_shared_parent_after_final_release_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(shared, monkeypatch)
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    original_inner = aggregate._recover_fit_release_bound
    original_snapshot = aggregate._release_snapshot_fd
    phase = {"inner_returned": False, "swapped": False}
    backup = tmp_path / "shared_original"

    def inner_then_arm(**kwargs):
        result = original_inner(**kwargs)
        phase["inner_returned"] = True
        return result

    def snapshot_then_swap(*args, **kwargs):
        result = original_snapshot(*args, **kwargs)
        if phase["inner_returned"] and not phase["swapped"]:
            shared.rename(backup)
            shutil.copytree(backup, shared)
            phase["swapped"] = True
        return result

    monkeypatch.setattr(aggregate, "_recover_fit_release_bound", inner_then_arm)
    monkeypatch.setattr(aggregate, "_release_snapshot_fd", snapshot_then_swap)
    with pytest.raises(ValueError, match="directory identity changed"):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )
    assert phase == {"inner_returned": True, "swapped": True}


@pytest.mark.parametrize("mutation", ["oversize_before_open", "grow_while_read"])
def test_source_closure_size_cap_fails_before_thermal_or_fit(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    source_probe = tmp_path / "source_probe.py"
    source_probe.write_bytes(b"x = 1\n")
    source_paths = dict(linear.R2R1_SOURCE_PATHS)
    source_paths["linear_core"] = source_probe
    monkeypatch.setattr(linear, "R2R1_SOURCE_PATHS", source_paths)
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    probe_stat = os.stat(source_probe)
    if mutation == "oversize_before_open":
        with source_probe.open("r+b") as handle:
            handle.truncate(linear.SOURCE_CLOSURE_SINGLE_FILE_BYTES_LIMIT + 1)

    original_read = linear.os.read
    original_open_bound = linear._open_bound_regular_file
    counts = {
        "thermal_open": 0,
        "parser": 0,
        "svd": 0,
        "probe_bytes": 0,
        "grown": False,
    }

    def open_bound(path, label, **kwargs):
        if os.path.abspath(str(path)) == os.path.abspath(
            str(linear.RECOMMENDED_THERMAL92)
        ):
            counts["thermal_open"] += 1
        return original_open_bound(path, label, **kwargs)

    def bounded_read(descriptor, size):
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == (
            probe_stat.st_dev,
            probe_stat.st_ino,
        ):
            if mutation == "grow_while_read" and not counts["grown"]:
                writer = os.open(source_probe, os.O_WRONLY | os.O_APPEND)
                try:
                    os.write(
                        writer,
                        b"z" * (linear.SOURCE_CLOSURE_SINGLE_FILE_BYTES_LIMIT + 32),
                    )
                finally:
                    os.close(writer)
                counts["grown"] = True
            block = original_read(descriptor, size)
            counts["probe_bytes"] += len(block)
            return block
        return original_read(descriptor, size)

    def forbidden_parser(value):
        counts["parser"] += 1
        raise AssertionError("source-cap failure reached label conversion")

    def forbidden_svd(*args, **kwargs):
        counts["svd"] += 1
        raise AssertionError("source-cap failure reached fit SVD")

    monkeypatch.setattr(linear, "_open_bound_regular_file", open_bound)
    monkeypatch.setattr(linear.os, "read", bounded_read)
    monkeypatch.setattr(linear, "_parse_whitelisted_float", forbidden_parser)
    monkeypatch.setattr(linear.np.linalg, "svd", forbidden_svd)
    with pytest.raises((PermissionError, ValueError)):
        linear.authorized_fit_pipeline(
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert counts["thermal_open"] == 0
    assert counts["parser"] == 0
    assert counts["svd"] == 0
    if mutation == "oversize_before_open":
        assert counts["probe_bytes"] == 0
    else:
        assert counts["grown"] is True
        assert counts["probe_bytes"] <= (
            linear.SOURCE_CLOSURE_SINGLE_FILE_BYTES_LIMIT + 1
        )


def test_unchanged_file_recheck_hashes_in_bounded_streaming_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "streamed.xyz"
    payload = b"a" * (2 * 1024 * 1024 + 17)
    source.write_bytes(payload)
    binding = linear.regular_file_binding(source, "streamed recheck fixture")
    source_stat = os.stat(source)
    original_read = linear.os.read
    requests: list[int] = []
    returned = 0

    def record_read(descriptor, size):
        nonlocal returned
        observed = os.fstat(descriptor)
        block = original_read(descriptor, size)
        if (observed.st_dev, observed.st_ino) == (
            source_stat.st_dev,
            source_stat.st_ino,
        ):
            requests.append(size)
            returned += len(block)
        return block

    monkeypatch.setattr(linear.os, "read", record_read)
    linear._require_bound_file_unchanged(
        source,
        label="streamed unchanged-file recheck",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_identity=binding["file_identity"],
        expected_parent_chain=binding["parent_chain"],
    )
    assert returned == len(payload)
    assert len(requests) >= 3
    assert max(requests) <= 1024 * 1024


def test_npz_huge_shape_header_is_rejected_before_numpy_load(monkeypatch) -> None:
    member = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        member,
        {
            "descr": "<f8",
            "fortran_order": False,
            "shape": (2**40,),
        },
    )
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("member.npy", member.getvalue())
    calls = {"np_load": 0}

    def forbidden_load(*args, **kwargs):
        calls["np_load"] += 1
        raise AssertionError("huge NPY shape reached np.load allocation")

    monkeypatch.setattr(aggregate.np, "load", forbidden_load)
    with pytest.raises(ValueError, match="NPY header"):
        aggregate._load_npz_bytes(
            raw.getvalue(),
            {
                "member": {
                    "shape": [1],
                    "dtype": "float64",
                    "raw_sha256": "0" * 64,
                }
            },
            "bounded NPZ",
        )
    assert calls["np_load"] == 0


def test_npz_many_entry_directory_is_rejected_before_zipfile_or_numpy_load(
    monkeypatch,
) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for index in range(32):
            archive.writestr(f"member_{index}.npy", b"")
    raw = stream.getvalue()
    calls = {"zipfile": 0, "numpy": 0}

    def forbidden_zipfile(*args, **kwargs):
        calls["zipfile"] += 1
        raise AssertionError("ZipFile constructed before raw EOCD prevalidation")

    def forbidden_numpy_load(*args, **kwargs):
        calls["numpy"] += 1
        raise AssertionError("np.load reached before raw EOCD prevalidation")

    monkeypatch.setattr(aggregate.zipfile, "ZipFile", forbidden_zipfile)
    monkeypatch.setattr(aggregate.np, "load", forbidden_numpy_load)
    with pytest.raises(ValueError, match="EOCD/resource contract"):
        aggregate._load_npz_bytes(
            raw,
            {
                "expected": {
                    "shape": [1],
                    "dtype": "float64",
                    "raw_sha256": "0" * 64,
                }
            },
            "many-entry NPZ",
        )
    assert calls == {"zipfile": 0, "numpy": 0}


def test_thermal_first_hash_growth_stops_at_frozen_size_plus_one(
    tmp_path: Path, monkeypatch
) -> None:
    thermal = tmp_path / "thermal_train.xyz"
    thermal.write_bytes(b"payload\n")
    monkeypatch.setattr(linear, "RECOMMENDED_THERMAL92", thermal)
    monkeypatch.setattr(
        linear, "THERMAL92_FILE_SHA256", hashlib.sha256(thermal.read_bytes()).hexdigest()
    )
    authorization = _real_test_authorization(tmp_path, monkeypatch)
    thermal_stat = os.stat(thermal)
    original_read = linear.os.read
    returned = 0

    def endless_after_eof(descriptor, size):
        nonlocal returned
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) != (
            thermal_stat.st_dev,
            thermal_stat.st_ino,
        ):
            return original_read(descriptor, size)
        block = original_read(descriptor, size)
        if not block:
            block = b"x" * size
        returned += len(block)
        return block

    monkeypatch.setattr(linear.os, "read", endless_after_eof)
    with pytest.raises(ValueError, match="grew beyond its frozen size"):
        linear._load_thermal92_force_labels_streaming_authorized(
            thermal, authorization=authorization
        )
    assert returned == linear.THERMAL92_FILE_SIZE_BYTES + 1


def test_thermal_parser_growth_stops_at_frozen_size_plus_one(
    tmp_path: Path, monkeypatch
) -> None:
    thermal = tmp_path / "thermal_train.xyz"
    _synthetic_extxyz(thermal, atom_count=72)
    monkeypatch.setattr(linear, "RECOMMENDED_THERMAL92", thermal)
    digest = hashlib.sha256(thermal.read_bytes()).hexdigest()
    monkeypatch.setattr(linear, "THERMAL92_FILE_SHA256", digest)
    monkeypatch.setattr(
        linear, "EXPECTED_LABEL_RAW_SHA256", _synthetic_label_raw_hashes(72)
    )
    monkeypatch.setattr(linear.r2r0, "structure_semantic_sha256", lambda atoms: "s")
    monkeypatch.setattr(
        linear,
        "THERMAL_STRUCTURE_IDENTITY_SHA256",
        linear.r2r.semantic_sha256([(index, "s") for index in range(92)]),
    )
    authorization = _real_test_authorization(tmp_path, monkeypatch)
    thermal_stat = os.stat(thermal)
    original_read = linear.os.read
    returned = 0

    monkeypatch.setattr(
        linear,
        "_sha256_fd_exact_size",
        lambda descriptor, expected_size, label: digest,
    )

    def endless_after_eof(descriptor, size):
        nonlocal returned
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) != (
            thermal_stat.st_dev,
            thermal_stat.st_ino,
        ):
            return original_read(descriptor, size)
        block = original_read(descriptor, size)
        if not block:
            block = b"x" * size
        returned += len(block)
        return block

    monkeypatch.setattr(linear.os, "read", endless_after_eof)
    with pytest.raises(ValueError, match="grew beyond its frozen size while parsed"):
        linear._load_thermal92_force_labels_streaming_authorized(
            thermal, authorization=authorization
        )
    assert returned == linear.THERMAL92_FILE_SIZE_BYTES + 1


def test_final_thermal_rehash_growth_stops_at_expected_size_plus_one(
    tmp_path: Path, monkeypatch
) -> None:
    thermal = tmp_path / "thermal.xyz"
    thermal.write_bytes(b"frozen\n")
    binding = linear.regular_file_binding(thermal, "final thermal fixture")
    thermal_stat = os.stat(thermal)
    original_read = linear.os.read
    returned = 0

    def endless_after_eof(descriptor, size):
        nonlocal returned
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) != (
            thermal_stat.st_dev,
            thermal_stat.st_ino,
        ):
            return original_read(descriptor, size)
        block = original_read(descriptor, size)
        if not block:
            block = b"x" * size
        returned += len(block)
        return block

    monkeypatch.setattr(linear.os, "read", endless_after_eof)
    with pytest.raises(ValueError, match="grew beyond its frozen size"):
        linear._require_bound_file_unchanged(
            thermal,
            label="final thermal rehash",
            expected_sha256=hashlib.sha256(thermal.read_bytes()).hexdigest(),
            expected_identity=binding["file_identity"],
            expected_parent_chain=binding["parent_chain"],
        )
    assert returned == len(b"frozen\n") + 1


def test_freeze_manifest_growth_stops_at_fixed_limit_plus_one(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "freeze_manifest.json"
    manifest.write_bytes(b"{}")
    binding = linear.regular_file_binding(manifest, "manifest fixture")
    manifest_stat = os.stat(manifest)
    original_read = linear.os.read
    returned = 0

    def endless_after_eof(descriptor, size):
        nonlocal returned
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) != (
            manifest_stat.st_dev,
            manifest_stat.st_ino,
        ):
            return original_read(descriptor, size)
        block = original_read(descriptor, size)
        if not block:
            block = b"x" * size
        returned += len(block)
        return block

    monkeypatch.setattr(linear.os, "read", endless_after_eof)
    with pytest.raises(ValueError, match="exceeds its size limit while read"):
        aggregate._read(
            manifest,
            "growing freeze manifest",
            expected_parent_chain=tuple(binding["parent_chain"]),
            expected_file_identity=binding["file_identity"],
            size_limit=aggregate.FREEZE_MANIFEST_BYTES_LIMIT,
        )
    assert returned == aggregate.FREEZE_MANIFEST_BYTES_LIMIT + 1


def test_every_read_regular_file_once_call_has_an_explicit_cap() -> None:
    for module in (linear, aggregate):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            function = call.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else None
            )
            if name == "_read_regular_file_once":
                assert any(keyword.arg == "size_limit" for keyword in call.keywords)


def test_private_attempt3_copy_ignores_ctime_and_directory_time_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "copy"
    expected = linear._materialize_captured_tree(
        root,
        {
            ".": {"kind": "directory"},
            "sub": {"kind": "directory"},
            "sub/payload.bin": {"kind": "file"},
        },
        {"sub/payload.bin": b"payload\n"},
    )
    before = linear._immutable_tree_snapshot(root, stable_directories=True)
    assert before == expected
    payload_fd = os.open(root / "sub" / "payload.bin", os.O_RDWR)
    try:
        _rewrite_same_bytes_and_restore_mtime(payload_fd)
    finally:
        os.close(payload_fd)
    for directory in (root, root / "sub"):
        observed = os.stat(directory)
        os.utime(
            directory,
            ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000),
        )
    assert linear._immutable_tree_snapshot(
        root, stable_directories=True
    ) == expected


def test_private_attempt3_copy_snapshot_rejects_changed_same_size_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "copy"
    expected = linear._materialize_captured_tree(
        root,
        {
            ".": {"kind": "directory"},
            "payload.bin": {"kind": "file"},
        },
        {"payload.bin": b"payload\n"},
    )
    writable = os.open(root / "payload.bin", os.O_RDWR)
    try:
        _rewrite_different_same_size_and_restore_mtime(writable)
    finally:
        os.close(writable)
    assert linear._immutable_tree_snapshot(
        root, stable_directories=True
    ) != expected


def test_private_checkpoint_copy_ignores_managed_xattr_drift_before_loader(
    monkeypatch,
) -> None:
    files, receipt_sha, physical = _private_checkpoint_fixture()
    calls = {"count": 0}

    def managed_drift(_descriptor: int, *, label: str):
        assert label
        calls["count"] += 1
        return {}

    monkeypatch.setattr(aggregate, "_authoritative_owned_xattrs", managed_drift)
    loaded = aggregate._load_settled_private_checkpoint(
        files,
        expected_receipt_sha256=receipt_sha,
        prefix="graphene_r2r1_test_checkpoint_settle_",
    )
    assert np.array_equal(loaded.physical_p, physical)
    assert calls["count"] > 0


def test_private_checkpoint_copy_allows_same_byte_rewrite_during_loader(
    monkeypatch,
) -> None:
    files, receipt_sha, _physical = _private_checkpoint_fixture()
    original_loader = aggregate.frozen.load_frozen_readout_checkpoint
    injected = {"value": False}

    def loader_then_rewrite(root, **kwargs):
        loaded = original_loader(root, **kwargs)
        target = root / frozen.CHECKPOINT_RECEIPT_BASENAME
        descriptor = os.open(
            target,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            _rewrite_same_bytes_and_restore_mtime(descriptor)
        finally:
            os.close(descriptor)
        injected["value"] = True
        return loaded

    monkeypatch.setattr(
        aggregate.frozen, "load_frozen_readout_checkpoint", loader_then_rewrite
    )
    aggregate._load_settled_private_checkpoint(
        files,
        expected_receipt_sha256=receipt_sha,
        prefix="graphene_r2r1_test_checkpoint_rewrite_",
    )
    assert injected["value"] is True


def test_private_checkpoint_copy_rejects_changed_same_size_bytes_during_loader(
    monkeypatch,
) -> None:
    files, receipt_sha, _physical = _private_checkpoint_fixture()
    original_loader = aggregate.frozen.load_frozen_readout_checkpoint

    def loader_then_rewrite(root, **kwargs):
        loaded = original_loader(root, **kwargs)
        descriptor = os.open(
            root / frozen.CHECKPOINT_RECEIPT_BASENAME,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            _rewrite_different_same_size_and_restore_mtime(descriptor)
        finally:
            os.close(descriptor)
        return loaded

    monkeypatch.setattr(
        aggregate.frozen, "load_frozen_readout_checkpoint", loader_then_rewrite
    )
    with pytest.raises(ValueError, match="post-load|changed during frozen load"):
        aggregate._load_settled_private_checkpoint(
            files,
            expected_receipt_sha256=receipt_sha,
            prefix="graphene_r2r1_test_checkpoint_changed_bytes_",
        )


@pytest.mark.parametrize("failure_point", ["parent_fsync", "final_binding"])
def test_reserve_root_closes_unreturned_fd_on_postopen_failure(
    tmp_path: Path, monkeypatch, failure_point: str
) -> None:
    _parent, parent_fd, parent_chain = linear._open_directory_chain(
        tmp_path, "reserve fd-leak test parent"
    )
    captured: list[int] = []
    original_open = aggregate.os.open

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if path == "fit" and kwargs.get("dir_fd") == parent_fd:
            captured.append(descriptor)
        return descriptor

    monkeypatch.setattr(aggregate.os, "open", tracking_open)
    if failure_point == "parent_fsync":
        original_fsync = aggregate.os.fsync

        def fail_parent_fsync(descriptor):
            if descriptor == parent_fd:
                raise OSError("injected reservation parent fsync failure")
            return original_fsync(descriptor)

        monkeypatch.setattr(aggregate.os, "fsync", fail_parent_fsync)
    else:
        original_verify = aggregate.linear._verify_directory_chain

        def fail_final_binding(path, label, expected_chain):
            if label == "R2R-1 reserved-root parent binding":
                raise OSError("injected reservation final binding failure")
            return original_verify(path, label, expected_chain)

        monkeypatch.setattr(
            aggregate.linear, "_verify_directory_chain", fail_final_binding
        )
    try:
        with pytest.raises(OSError, match="injected reservation"):
            aggregate._reserve_canonical_root(
                tmp_path / "fit",
                parent_fd=parent_fd,
                parent_chain=parent_chain,
            )
        assert captured
        with pytest.raises(OSError) as closed:
            os.fstat(captured[-1])
        assert closed.value.errno == errno.EBADF
        assert (tmp_path / "fit").is_dir()
    finally:
        os.close(parent_fd)


def test_candidate_temporary_closes_unreturned_fd_on_parent_fsync_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _parent, parent_fd, parent_chain = linear._open_directory_chain(
        tmp_path, "candidate fd-leak test parent"
    )
    captured: list[int] = []
    original_open = aggregate.os.open
    original_fsync = aggregate.os.fsync

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if (
            isinstance(path, str)
            and path.startswith(".r2r1-candidate-")
            and kwargs.get("dir_fd") == parent_fd
        ):
            captured.append(descriptor)
        return descriptor

    def fail_parent_fsync(descriptor):
        if descriptor == parent_fd:
            raise OSError("injected candidate parent fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(aggregate.os, "open", tracking_open)
    monkeypatch.setattr(aggregate.os, "fsync", fail_parent_fsync)
    try:
        with pytest.raises(OSError, match="injected candidate"):
            aggregate._create_bound_candidate_temporary(
                parent_path=tmp_path,
                parent_fd=parent_fd,
                parent_chain=parent_chain,
            )
        assert captured
        with pytest.raises(OSError) as closed:
            os.fstat(captured[-1])
        assert closed.value.errno == errno.EBADF
        assert len(list(tmp_path.glob(".r2r1-candidate-*.tmp"))) == 1
    finally:
        os.close(parent_fd)


def test_existing_child_open_closes_unreturned_fd_on_fstat_failure(
    tmp_path: Path, monkeypatch
) -> None:
    child = tmp_path / "existing"
    child.mkdir()
    _parent, parent_fd, parent_chain = linear._open_directory_chain(
        tmp_path, "existing-child fd-leak test parent"
    )
    captured: list[int] = []
    original_open = aggregate.os.open
    original_fstat = aggregate.os.fstat

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if path == child.name and kwargs.get("dir_fd") == parent_fd:
            captured.append(descriptor)
        return descriptor

    def fail_child_fstat(descriptor):
        if captured and descriptor == captured[-1]:
            raise OSError("injected existing-child fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(aggregate.os, "open", tracking_open)
    monkeypatch.setattr(aggregate.os, "fstat", fail_child_fstat)
    try:
        with pytest.raises(OSError, match="injected existing-child"):
            aggregate._open_existing_bound_child_directory(
                parent_path=tmp_path,
                parent_fd=parent_fd,
                parent_chain=parent_chain,
                child_name=child.name,
                label="existing child fstat fault",
            )
        assert captured
        with pytest.raises(OSError) as closed:
            original_fstat(captured[-1])
        assert closed.value.errno == errno.EBADF
        assert child.is_dir()
    finally:
        os.close(parent_fd)


@pytest.mark.parametrize("failure_point", ["anchor_fstat", "child_fstat"])
def test_open_directory_chain_closes_all_fds_on_fstat_failure(
    tmp_path: Path, monkeypatch, failure_point: str
) -> None:
    target = tmp_path / "child"
    target.mkdir()
    opened: list[int] = []
    original_open = linear.os.open
    original_fstat = linear.os.fstat

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def injected_fstat(descriptor):
        if opened and (
            (failure_point == "anchor_fstat" and descriptor == opened[0])
            or (
                failure_point == "child_fstat"
                and len(opened) >= 2
                and descriptor == opened[1]
            )
        ):
            raise OSError(f"injected directory-chain {failure_point}")
        return original_fstat(descriptor)

    monkeypatch.setattr(linear.os, "open", tracking_open)
    monkeypatch.setattr(linear.os, "fstat", injected_fstat)
    with pytest.raises(OSError, match="injected directory-chain"):
        linear._open_directory_chain(target, "directory-chain fd fault")
    assert opened
    for descriptor in set(opened):
        with pytest.raises(OSError) as closed:
            original_fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_reject_path_closes_new_and_old_fds_on_child_fstat_failure(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "child" / "payload"
    target.parent.mkdir()
    target.write_bytes(b"payload\n")
    opened: list[int] = []
    original_open = linear.os.open
    original_fstat = linear.os.fstat

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_first_child_fstat(descriptor):
        if len(opened) >= 2 and descriptor == opened[1]:
            raise OSError("injected reject-path child fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(linear.os, "open", tracking_open)
    monkeypatch.setattr(linear.os, "fstat", fail_first_child_fstat)
    with pytest.raises(OSError, match="injected reject-path"):
        linear._reject_path(target, "reject-path fd fault")
    assert len(opened) >= 2
    for descriptor in set(opened):
        with pytest.raises(OSError) as closed:
            original_fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_authorization_control_child_fstat_failure_closes_unregistered_fd(
    tmp_path: Path, monkeypatch
) -> None:
    control, _output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    captured: list[int] = []
    original_open = linear.os.open
    original_fstat = linear.os.fstat

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if any(frame.function == "hold_control_child" for frame in inspect.stack()):
            captured.append(descriptor)
        return descriptor

    def fail_held_child_fstat(descriptor):
        if captured and descriptor == captured[-1]:
            raise OSError("injected held control-child fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(linear.os, "open", tracking_open)
    monkeypatch.setattr(linear.os, "fstat", fail_held_child_fstat)
    with pytest.raises(OSError, match="injected held control-child"):
        linear.validate_execution_authorization(manifest, marker)
    assert captured
    for descriptor in set(captured):
        with pytest.raises(OSError) as closed:
            original_fstat(descriptor)
        assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("failure_point", ["checkpoint_fstat", "numpy_asarray"])
def test_checkpoint_writer_closes_unreturned_directory_fd(
    tmp_path: Path, monkeypatch, failure_point: str
) -> None:
    root_fd = os.open(
        tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    captured: list[int] = []
    original_open = aggregate.os.open
    original_fstat = aggregate.os.fstat

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if path == aggregate.CHECKPOINT_DIRNAME and kwargs.get("dir_fd") == root_fd:
            captured.append(descriptor)
        return descriptor

    monkeypatch.setattr(aggregate.os, "open", tracking_open)
    if failure_point == "checkpoint_fstat":
        def fail_checkpoint_fstat(descriptor):
            if captured and descriptor == captured[-1]:
                raise OSError("injected checkpoint fstat failure")
            return original_fstat(descriptor)

        monkeypatch.setattr(aggregate.os, "fstat", fail_checkpoint_fstat)
    else:
        monkeypatch.setattr(
            aggregate.np,
            "asarray",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("injected checkpoint np.asarray failure")
            ),
        )
    try:
        with pytest.raises(OSError, match="injected checkpoint"):
            aggregate._write_checkpoint(
                np.zeros(65, dtype="<f8"),
                fit_manifest_sha256="a" * 64,
                fit_receipt_sha256="b" * 64,
                release_root_fd=root_fd,
            )
        assert captured
        with pytest.raises(OSError) as closed:
            original_fstat(captured[-1])
        assert closed.value.errno == errno.EBADF
        assert (tmp_path / aggregate.CHECKPOINT_DIRNAME).is_dir()
    finally:
        os.close(root_fd)


def test_linear_atomic_cleanup_fault_still_closes_file_and_parent_fds(
    tmp_path: Path, monkeypatch
) -> None:
    opened: list[int] = []
    temporary_fd: list[int] = []
    cleanup = {"active": False}
    original_open = linear.os.open
    original_fstat = linear.os.fstat
    original_fsync = linear.os.fsync

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        opened.append(descriptor)
        if isinstance(path, str) and path.startswith(".payload.bin."):
            temporary_fd.append(descriptor)
        return descriptor

    def fail_write_fsync(descriptor):
        if temporary_fd and descriptor == temporary_fd[-1]:
            cleanup["active"] = True
            raise OSError("injected linear atomic write fsync failure")
        return original_fsync(descriptor)

    def fail_cleanup_fstat(descriptor):
        if cleanup["active"] and temporary_fd and descriptor == temporary_fd[-1]:
            raise OSError("injected linear atomic cleanup fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(linear.os, "open", tracking_open)
    monkeypatch.setattr(linear.os, "fsync", fail_write_fsync)
    monkeypatch.setattr(linear.os, "fstat", fail_cleanup_fstat)
    with pytest.raises(OSError, match="cleanup fstat"):
        linear.atomic_write_bytes(tmp_path / "payload.bin", b"payload\n")
    assert temporary_fd
    for descriptor in set(opened):
        with pytest.raises(OSError) as closed:
            original_fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_aggregate_atomic_cleanup_fault_still_closes_private_file_fd(
    tmp_path: Path, monkeypatch
) -> None:
    directory_fd = os.open(
        tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    temporary_fd: list[int] = []
    cleanup = {"active": False}
    original_open = aggregate.os.open
    original_fstat = aggregate.os.fstat
    original_fsync = aggregate.os.fsync

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if isinstance(path, str) and path.startswith(".owned.bin."):
            temporary_fd.append(descriptor)
        return descriptor

    def fail_write_fsync(descriptor):
        if temporary_fd and descriptor == temporary_fd[-1]:
            cleanup["active"] = True
            raise OSError("injected aggregate atomic write fsync failure")
        return original_fsync(descriptor)

    def fail_cleanup_fstat(descriptor):
        if cleanup["active"] and temporary_fd and descriptor == temporary_fd[-1]:
            raise OSError("injected aggregate atomic cleanup fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(aggregate.os, "open", tracking_open)
    monkeypatch.setattr(aggregate.os, "fsync", fail_write_fsync)
    monkeypatch.setattr(aggregate.os, "fstat", fail_cleanup_fstat)
    try:
        with pytest.raises(OSError, match="cleanup fstat"):
            aggregate._atomic_write_bytes_at(directory_fd, "owned.bin", b"owned\n")
        assert temporary_fd
        with pytest.raises(OSError) as closed:
            original_fstat(temporary_fd[-1])
        assert closed.value.errno == errno.EBADF
        original_fstat(directory_fd)
    finally:
        os.close(directory_fd)


def test_running_marker_cleanup_fault_still_closes_private_file_fd(
    tmp_path: Path, monkeypatch
) -> None:
    root_fd = os.open(
        tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    temporary_fd: list[int] = []
    cleanup = {"active": False}
    original_open = aggregate.os.open
    original_fstat = aggregate.os.fstat

    def tracking_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if isinstance(path, str) and path.startswith(".RUNNING."):
            temporary_fd.append(descriptor)
        return descriptor

    def fail_rewrite(*args, **kwargs):
        cleanup["active"] = True
        raise OSError("injected RUNNING rewrite failure")

    def fail_cleanup_fstat(descriptor):
        if cleanup["active"] and temporary_fd and descriptor == temporary_fd[-1]:
            raise OSError("injected RUNNING cleanup fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(aggregate.os, "open", tracking_open)
    monkeypatch.setattr(aggregate, "_rewrite_held_file", fail_rewrite)
    monkeypatch.setattr(aggregate.os, "fstat", fail_cleanup_fstat)
    try:
        with pytest.raises(OSError, match="cleanup fstat"):
            aggregate._create_running_marker(root_fd)
        assert temporary_fd
        with pytest.raises(OSError) as closed:
            original_fstat(temporary_fd[-1])
        assert closed.value.errno == errno.EBADF
        original_fstat(root_fd)
    finally:
        os.close(root_fd)


def test_checkpoint_directory_time_drift_during_success_validation_is_portable(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: _mock_pipeline_result(aggregate.CHECKPOINT_STATUS_SOURCE),
    )
    original_root_file = aggregate._root_file
    changed = {"value": False}

    def root_file_with_directory_time_drift(root_fd, name, label, **kwargs):
        if label == "R2R-1 success EXIT_CODE" and not changed["value"]:
            checkpoint = output / aggregate.CHECKPOINT_DIRNAME
            observed = os.stat(checkpoint)
            os.utime(
                checkpoint,
                ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000),
            )
            changed["value"] = True
        return original_root_file(root_fd, name, label, **kwargs)

    monkeypatch.setattr(aggregate, "_root_file", root_file_with_directory_time_drift)
    result = aggregate.materialize_authorized_fit(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
    )
    assert changed["value"] is True
    assert result["status"] == aggregate.CHECKPOINT_STATUS_SOURCE
    assert (output / "DONE").is_file()
    assert (output / aggregate.CHECKPOINT_DIRNAME).is_dir()


def test_checkpoint_directory_mode_drift_during_success_validation_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    monkeypatch.setattr(
        aggregate.linear,
        "authorized_fit_pipeline",
        lambda **kwargs: _mock_pipeline_result(aggregate.CHECKPOINT_STATUS_SOURCE),
    )
    original_root_file = aggregate._root_file
    changed = {"value": False}

    def root_file_with_directory_mode_drift(root_fd, name, label, **kwargs):
        if label == "R2R-1 success EXIT_CODE" and not changed["value"]:
            os.chmod(output / aggregate.CHECKPOINT_DIRNAME, 0o755)
            changed["value"] = True
        return original_root_file(root_fd, name, label, **kwargs)

    monkeypatch.setattr(aggregate, "_root_file", root_file_with_directory_mode_drift)
    with pytest.raises(ValueError, match="science artifacts changed|identity changed"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert changed["value"] is True
    assert (output / "RUNNING").is_file()
    assert not (output / "DONE").exists()


def test_oversize_failure_receipt_is_rejected_before_artifact_write(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, _candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    oversized_message = "x" * (aggregate.ROOT_FILE_SIZE_LIMITS["failure.json"] + 1)

    def fail_pipeline(**kwargs):
        raise RuntimeError(oversized_message)

    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", fail_pipeline)
    with pytest.raises(RuntimeError) as captured:
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert isinstance(captured.value.__cause__, ValueError)
    assert "failure receipt exceeds" in str(captured.value.__cause__)
    assert (output / "RUNNING").is_file()
    assert not (output / "failure.json").exists()
    assert not (output / "EXIT_CODE").exists()
    assert not (output / "FAILED").exists()


def test_publication_boundary_second_pass_catches_earlier_file_rewrite(
    tmp_path: Path, monkeypatch
) -> None:
    _control, output, candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    output.mkdir()
    first = output / "a.bin"
    later = output / "b.bin"
    first.write_bytes(b"a\n")
    later.write_bytes(b"b\n")
    _parent, parent_fd, parent_chain = linear._open_directory_chain(
        output.parent, "publication-test parent"
    )
    _root, root_fd, root_chain = linear._open_directory_chain(
        output, "publication-test root"
    )
    try:
        snapshot = {
            path.name: {
                "kind": "file",
                "identity": linear._owned_regular_identity(os.stat(path)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
            for path in (first, later)
        }
        boundary = linear._issue_fit_publication_boundary(
            release_parent_fd=parent_fd,
            release_parent_path=output.parent,
            release_parent_chain=parent_chain,
            root_fd=root_fd,
            root_path=output,
            expected_root_identity=root_chain[-1],
            expected_root_snapshot=snapshot,
            candidate_parent_fd=parent_fd,
            candidate_parent_path=candidate.parent,
            candidate_parent_chain=parent_chain,
            candidate_name=candidate.name,
            candidate_state="absent",
        )
        original_stat = linear.os.stat
        injected = {"value": False}

        def rewrite_earlier_at_later(name, *args, **kwargs):
            observed = original_stat(name, *args, **kwargs)
            if (
                name == later.name
                and kwargs.get("dir_fd") == root_fd
                and not injected["value"]
            ):
                descriptor = os.open(first, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
                try:
                    _rewrite_different_same_size_and_restore_mtime(descriptor)
                finally:
                    os.close(descriptor)
                injected["value"] = True
            return observed

        monkeypatch.setattr(linear.os, "stat", rewrite_earlier_at_later)
        with pytest.raises(ValueError, match="guarded release bytes changed"):
            linear._require_fit_publication_boundary(boundary)
        assert injected["value"] is True
    finally:
        os.close(root_fd)
        os.close(parent_fd)


def test_publication_boundary_existing_candidate_closes_anchor_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    _control, output, candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    output.mkdir()
    candidate.mkdir()
    release_file = output / "artifact.bin"
    anchor_file = candidate / "release_manifest.json"
    release_file.write_bytes(b"release\n")
    anchor_raw = b"anchor-v2\n"
    anchor_file.write_bytes(anchor_raw)
    _parent, parent_fd, parent_chain = linear._open_directory_chain(
        tmp_path, "existing-boundary parent"
    )
    _root, root_fd, root_chain = linear._open_directory_chain(
        output, "existing-boundary release"
    )
    _candidate, candidate_fd, candidate_chain = linear._open_directory_chain(
        candidate, "existing-boundary candidate"
    )
    try:
        release_identity = linear._owned_regular_identity(os.stat(release_file))
        anchor_identity = linear._owned_regular_identity(os.stat(anchor_file))
        boundary = linear._issue_fit_publication_boundary(
            release_parent_fd=parent_fd,
            release_parent_path=tmp_path,
            release_parent_chain=parent_chain,
            root_fd=root_fd,
            root_path=output,
            expected_root_identity=root_chain[-1],
            expected_root_snapshot={
                release_file.name: {
                    "kind": "file",
                    "identity": release_identity,
                    "sha256": hashlib.sha256(release_file.read_bytes()).hexdigest(),
                    "size": release_file.stat().st_size,
                }
            },
            candidate_parent_fd=parent_fd,
            candidate_parent_path=tmp_path,
            candidate_parent_chain=parent_chain,
            candidate_name=candidate.name,
            candidate_state="existing",
            candidate_fd=candidate_fd,
            candidate_path=candidate,
            candidate_chain=candidate_chain,
            anchor_name=anchor_file.name,
            anchor_identity=anchor_identity,
            anchor_raw=anchor_raw,
        )
        descriptor = os.open(anchor_file, os.O_RDWR)
        try:
            _rewrite_different_same_size_and_restore_mtime(descriptor)
        finally:
            os.close(descriptor)
        with pytest.raises(ValueError, match="guarded candidate anchor bytes changed"):
            linear._require_fit_publication_boundary(boundary)
    finally:
        os.close(candidate_fd)
        os.close(root_fd)
        os.close(parent_fd)


@pytest.mark.parametrize("tamper", ["release", "anchor"])
def test_control_pass_tamper_is_closed_by_reverse_publication_pass_before_open(
    tmp_path: Path, monkeypatch, tamper: str
) -> None:
    control, output, candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    output.mkdir()
    release_file = output / "RUNNING"
    release_raw = b"running-v2\n"
    release_file.write_bytes(release_raw)
    _parent, parent_fd, parent_chain = linear._open_directory_chain(
        output.parent, "union-boundary parent"
    )
    _root, root_fd, root_chain = linear._open_directory_chain(
        output, "union-boundary release"
    )
    candidate_fd = None
    try:
        release_identity = linear._owned_regular_identity(os.stat(release_file))
        issue_kwargs = {
            "release_parent_fd": parent_fd,
            "release_parent_path": output.parent,
            "release_parent_chain": parent_chain,
            "root_fd": root_fd,
            "root_path": output,
            "expected_root_identity": root_chain[-1],
            "expected_root_snapshot": {
                release_file.name: {
                    "kind": "file",
                    "identity": release_identity,
                    "sha256": hashlib.sha256(release_raw).hexdigest(),
                    "size": len(release_raw),
                }
            },
            "candidate_parent_fd": parent_fd,
            "candidate_parent_path": candidate.parent,
            "candidate_parent_chain": parent_chain,
            "candidate_name": candidate.name,
            "candidate_state": "absent",
        }
        target = release_file
        expected_message = "guarded release bytes changed"
        if tamper == "anchor":
            candidate.mkdir()
            anchor = candidate / "release_manifest.json"
            anchor_raw = b"anchor-v2\n"
            anchor.write_bytes(anchor_raw)
            _candidate, candidate_fd, candidate_chain = linear._open_directory_chain(
                candidate, "union-boundary candidate"
            )
            issue_kwargs.update(
                {
                    "candidate_state": "existing",
                    "candidate_fd": candidate_fd,
                    "candidate_path": candidate,
                    "candidate_chain": candidate_chain,
                    "anchor_name": anchor.name,
                    "anchor_identity": linear._owned_regular_identity(
                        os.stat(anchor)
                    ),
                    "anchor_raw": anchor_raw,
                }
            )
            target = anchor
            expected_message = "guarded candidate anchor bytes changed"
        boundary = linear._issue_fit_publication_boundary(**issue_kwargs)
        original_xattrs = linear._authoritative_owned_xattrs
        original_open = linear._open_bound_regular_file
        injected = {"value": False}
        control_manifest_checks = {"count": 0}
        thermal_opens = {"count": 0}

        def control_capture_then_tamper(descriptor, *, label):
            result = original_xattrs(descriptor, label=label)
            if label == "R2R-1 post-publication freeze manifest before marker":
                control_manifest_checks["count"] += 1
            if control_manifest_checks["count"] == 3 and not injected["value"]:
                writable = os.open(
                    target, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    _rewrite_different_same_size_and_restore_mtime(writable)
                finally:
                    os.close(writable)
                injected["value"] = True
            return result

        def count_thermal_open(path, label, **kwargs):
            if label == "R2R-1 post-GO thermal92 binding":
                thermal_opens["count"] += 1
            return original_open(path, label, **kwargs)

        monkeypatch.setattr(
            linear, "_authoritative_owned_xattrs", control_capture_then_tamper
        )
        monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
        with pytest.raises(ValueError, match=expected_message):
            linear.validate_execution_authorization(
                manifest, marker, publication_boundary=boundary
            )
        assert injected["value"] is True
        assert thermal_opens["count"] == 0
    finally:
        if candidate_fd is not None:
            os.close(candidate_fd)
        os.close(root_fd)
        os.close(parent_fd)


def test_candidate_appearing_during_attempt3_recovery_blocks_thermal_open_and_fit(
    tmp_path: Path, monkeypatch
) -> None:
    control, output, candidate, _release = _patch_production_paths(
        monkeypatch, tmp_path
    )
    control.mkdir()
    manifest, marker = _real_authorization_files(control)
    calls = {"thermal_open": 0, "parser": 0, "ridge": 0}
    injected = {"value": False}
    original_open = linear._open_bound_regular_file

    def recover_then_collide(*args, **kwargs):
        candidate.mkdir()
        (candidate / "wrong").write_bytes(b"collision\n")
        injected["value"] = True
        return (
            {"status": "synthetic attempt3 recovery"},
            {
                "thermal_parameter_force_design_eV_A": np.zeros(
                    (92, 72, 3, 65), dtype="<f8"
                ),
                "thermal_fixed_force_eV_A": np.zeros((92, 72, 3), dtype="<f8"),
            },
        )

    def count_thermal_open(path, label, **kwargs):
        if injected["value"] and Path(path) == linear.RECOMMENDED_THERMAL92:
            calls["thermal_open"] += 1
        return original_open(path, label, **kwargs)

    def forbidden(kind):
        def fail(*args, **kwargs):
            calls[kind] += 1
            raise AssertionError(f"candidate collision reached {kind}")

        return fail

    monkeypatch.setattr(
        linear, "validate_attempt3_completed_recovery", recover_then_collide
    )
    monkeypatch.setattr(linear, "all_split_design_audits", lambda value: {"pass": True})
    monkeypatch.setattr(linear, "_open_bound_regular_file", count_thermal_open)
    monkeypatch.setattr(
        linear, "_load_thermal92_force_labels_streaming_authorized", forbidden("parser")
    )
    monkeypatch.setattr(linear, "_fit_weighted_ridge", forbidden("ridge"))
    with pytest.raises(FileExistsError, match="candidate appeared"):
        aggregate.materialize_authorized_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
        )
    assert calls == {"thermal_open": 0, "parser": 0, "ridge": 0}


def test_anchor_inner_snapshot_candidate_appearance_blocks_scientific_solve(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_rebind = aggregate._verify_snapshot_identities_fd
    calls = {"rebind": 0, "pipeline": 0, "parser": 0, "svd": 0}

    def rebind_then_inject(*args, **kwargs):
        result = original_rebind(*args, **kwargs)
        calls["rebind"] += 1
        if calls["rebind"] == 1:
            release_manifest.parent.mkdir()
            release_manifest.write_bytes(b"wrong\n")
        return result

    def forbidden_pipeline(**kwargs):
        calls["pipeline"] += 1
        raise AssertionError("candidate drift reached scientific solve")

    def forbidden_parser(value):
        calls["parser"] += 1
        raise AssertionError("candidate drift reached label parser")

    def forbidden_svd(*args, **kwargs):
        calls["svd"] += 1
        raise AssertionError("candidate drift reached ridge SVD")

    monkeypatch.setattr(
        aggregate, "_verify_snapshot_identities_fd", rebind_then_inject
    )
    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_pipeline)
    monkeypatch.setattr(aggregate.linear, "_parse_whitelisted_float", forbidden_parser)
    monkeypatch.setattr(aggregate.linear.np.linalg, "svd", forbidden_svd)
    with pytest.raises(FileExistsError, match="candidate root appeared"):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert calls["rebind"] >= 1
    assert calls["pipeline"] == calls["parser"] == calls["svd"] == 0


def test_public_recovery_inner_snapshot_candidate_replacement_blocks_solve(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    anchor = aggregate.anchor_completed_release(
        output_root=output,
        attempt3_root=ATTEMPT3,
        thermal92_path=linear.RECOMMENDED_THERMAL92,
        freeze_manifest=manifest,
        authorization_marker=marker,
        release_manifest=release_manifest,
        materialization_witness=receipt.anchor_witness,
    )
    original_rebind = aggregate._verify_snapshot_identities_fd
    calls = {"rebind": 0, "pipeline": 0, "parser": 0, "svd": 0}
    backup = tmp_path / "candidate_original"

    def rebind_then_replace(*args, **kwargs):
        result = original_rebind(*args, **kwargs)
        calls["rebind"] += 1
        if calls["rebind"] == 1:
            release_manifest.parent.rename(backup)
            shutil.copytree(backup, release_manifest.parent)
        return result

    def forbidden_pipeline(**kwargs):
        calls["pipeline"] += 1
        raise AssertionError("candidate replacement reached scientific solve")

    def forbidden_parser(value):
        calls["parser"] += 1
        raise AssertionError("candidate replacement reached label parser")

    def forbidden_svd(*args, **kwargs):
        calls["svd"] += 1
        raise AssertionError("candidate replacement reached ridge SVD")

    monkeypatch.setattr(
        aggregate, "_verify_snapshot_identities_fd", rebind_then_replace
    )
    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_pipeline)
    monkeypatch.setattr(aggregate.linear, "_parse_whitelisted_float", forbidden_parser)
    monkeypatch.setattr(aggregate.linear.np.linalg, "svd", forbidden_svd)
    with pytest.raises(ValueError, match="directory identity changed"):
        aggregate.recover_completed_fit(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            expected_release_manifest_sha256=anchor["sha256"],
        )
    assert calls["rebind"] >= 1
    assert calls["pipeline"] == calls["parser"] == calls["svd"] == 0


def test_bound_recovery_inner_snapshot_release_root_replacement_blocks_solve(
    tmp_path: Path, monkeypatch
) -> None:
    output, release_manifest, manifest, marker, _control, receipt = (
        _materialize_mock_release(tmp_path, monkeypatch)
    )
    original_rebind = aggregate._verify_snapshot_identities_fd
    calls = {"rebind": 0, "pipeline": 0, "parser": 0, "svd": 0}
    backup = tmp_path / "release_original"

    def rebind_then_replace(*args, **kwargs):
        result = original_rebind(*args, **kwargs)
        calls["rebind"] += 1
        if calls["rebind"] == 1:
            output.rename(backup)
            shutil.copytree(backup, output)
        return result

    def forbidden_pipeline(**kwargs):
        calls["pipeline"] += 1
        raise AssertionError("release replacement reached scientific solve")

    def forbidden_parser(value):
        calls["parser"] += 1
        raise AssertionError("release replacement reached label parser")

    def forbidden_svd(*args, **kwargs):
        calls["svd"] += 1
        raise AssertionError("release replacement reached ridge SVD")

    monkeypatch.setattr(
        aggregate, "_verify_snapshot_identities_fd", rebind_then_replace
    )
    monkeypatch.setattr(aggregate.linear, "authorized_fit_pipeline", forbidden_pipeline)
    monkeypatch.setattr(aggregate.linear, "_parse_whitelisted_float", forbidden_parser)
    monkeypatch.setattr(aggregate.linear.np.linalg, "svd", forbidden_svd)
    with pytest.raises(
        ValueError,
        match="directory identity changed|held payload write",
    ):
        aggregate.anchor_completed_release(
            output_root=output,
            attempt3_root=ATTEMPT3,
            thermal92_path=linear.RECOMMENDED_THERMAL92,
            freeze_manifest=manifest,
            authorization_marker=marker,
            release_manifest=release_manifest,
            materialization_witness=receipt.anchor_witness,
        )
    assert calls["rebind"] >= 1
    assert calls["pipeline"] == calls["parser"] == calls["svd"] == 0
