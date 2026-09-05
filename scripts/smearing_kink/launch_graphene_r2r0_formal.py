#!/usr/bin/env python3
"""Freeze and execute the fixed Tailscale-only R2R-0 formal allocation."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence

from graphene_r2r0_formal import (
    FORMAL_CONTRACT_SHA256,
    FORMAL_SOURCE_PATHS,
    FROZEN_SOURCE_PATHS,
    SHARD_INDICES,
    _atomic_json,
    _atomic_bytes,
    _failure,
    _fresh_run_root,
    _is_suspicious_safety_key,
    _key_path_matches,
    _reject_path,
    recursive_key_path_schema,
    recursive_key_path_schema_sha256,
    _success,
    _validate_receipt_contract,
    prepare_freeze_manifest,
    sha256,
    validate_execution_authorization,
)


ROOT = Path(__file__).resolve().parents[2]
CONTROL_RELATIVE = "results/r2r0_formal/control"
INPUT_RELATIVE = {
    "endpoint_checkpoint": "results/graphene_physics_temperature/post_p4_feasibility/R2Q_four_step_trust_region/formal_4step_seed83_rtx/endpoint.pt",
    "endpoint_receipt": "results/graphene_physics_temperature/post_p4_feasibility/R2Q_four_step_trust_region/formal_4step_seed83_rtx/endpoint_receipt.json",
    "endpoint_marker": "results/graphene_physics_temperature/post_p4_feasibility/R2Q_four_step_trust_region/formal_4step_seed83_rtx/ENDPOINT_FROZEN",
    "reference_6x6": "data/graphene_r2o_taylor_null_core/reference_6x6.xyz",
    "reference_8x8": "data/graphene_r2o_taylor_null_core/reference_8x8.xyz",
    "thermal92": "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
    "harmonic_zero32": "data/graphene_r2o_taylor_null_core/train_harmonic_lambda1_small_zero.xyz",
}
NODE_SPECS = {
    "V100-A": {
        "target": "root@100.80.236.112",
        "bundle_base": "/data/graphene_r2r0/formal_bundle",
        "env": "phonon-mlip",
        "conda": "/root/miniconda3/bin/conda",
        "roles": ("shard0",),
    },
    "V100-B": {
        "target": "root@100.123.220.57",
        "bundle_base": "/data/graphene_r2r0/formal_bundle",
        "env": "phonon-mlip",
        "conda": "/root/miniconda3/bin/conda",
        "roles": ("shard1", "mechanics"),
    },
    "RTX-2060": {
        "target": "howardwang@100.105.21.7",
        "bundle_base": "/home/howardwang/phonon/results/r2r0_formal_bundle_20260825",
        "env": "phonon",
        "conda": "/home/howardwang/miniconda3/bin/conda",
        "roles": ("shard2",),
    },
}
ROLE_NODE = {
    role: name for name, spec in NODE_SPECS.items() for role in spec["roles"]
}
ALLOWED_TARGETS = frozenset(spec["target"] for spec in NODE_SPECS.values())
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ControlPath=none")
REMOTE_SYMLINK_GUARD = (
    "import os\n"
    "import pathlib\n"
    "import stat\n"
    "import sys\n"
    "path = pathlib.Path(sys.argv[1])\n"
    "current = pathlib.Path(path.anchor)\n"
    "for part in path.parts[1:]:\n"
    "    current = current / part\n"
    "    try:\n"
    "        mode = os.lstat(current).st_mode\n"
    "    except FileNotFoundError:\n"
    "        continue\n"
    "    if stat.S_ISLNK(mode):\n"
    "        raise SystemExit(2)\n"
    "if len(sys.argv) == 3 and sys.argv[2] == 'require-absent':\n"
    "    try:\n"
    "        os.lstat(path)\n"
    "    except FileNotFoundError:\n"
    "        pass\n"
    "    else:\n"
    "        raise SystemExit(3)\n"
)
ROLE_OUTPUT_BASES = {
    "shard0": "/data/graphene_r2r0/formal_r2r0_three_host_seed83",
    "shard1": "/data/graphene_r2r0/formal_r2r0_three_host_seed83",
    "mechanics": "/data/graphene_r2r0/formal_r2r0_three_host_seed83",
    "shard2": "/home/howardwang/phonon/results/r2r0_formal_three_host_seed83",
}
FORMAL_ATTEMPT = 3
EXPECTED_SUCCESSFUL_COMMAND_COUNT = 220
EXPECTED_SUCCESSFUL_LOG_FILE_COUNT = 660
LAUNCH_SAFETY_FIELDS = {
    "fit_or_training": False,
    "held_or_support_access": False,
}
LAUNCH_RECEIPT_FORMAT = (
    "graphene_r2r0_formal_launch_receipt_v5_attempt3_recursive_schema"
)
LAUNCH_RECEIPT_TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "status",
        "formal_contract_sha256",
        "execution_authorization",
        "plan",
        "commands",
        "partial_state",
        "run_id",
        "attempt",
        "artifact_manifests",
        "log_manifest",
        "control_manifest",
        "aggregate_receipt",
        *LAUNCH_SAFETY_FIELDS,
    }
)
LAUNCH_AGGREGATE_SUMMARY_KEYS = frozenset(
    {
        "sha256",
        "status",
        "arrays_sha256",
        "representation_precheck_pass",
        "numerically_inconclusive",
    }
)
LAUNCH_SAFETY_FALSE_PATHS = frozenset(
    {(key,) for key in LAUNCH_SAFETY_FIELDS}
)
LAUNCH_BENIGN_SCIENTIFIC_PATHS = frozenset(
    {
        ("commands", "*", "train_gate"),
        ("commands", "*", "training_count"),
        ("commands", "*", "force_design_rank"),
    }
)
LAUNCH_RECURSIVE_WILDCARD_MAPPING_PATHS = frozenset(
    {("log_manifest", "files")}
)
LAUNCH_RECURSIVE_SCHEMA_SHA256 = (
    "a6eb1777c1504987eba4f590f10f428cf2741ce08983587c7f25e40599633145"
)
LAUNCH_RECURSIVE_SCHEMA_PATH_COUNT = 211

PLAN = {
    "format": "graphene_r2r0_formal_launch_plan_v6_attempt3_recursive_schema",
    "formal_contract_sha256": FORMAL_CONTRACT_SHA256,
    "remote_execution_authorized": "only_by_valid_external_GO_marker",
    "transport": "OpenSSH over Tailscale TUN to three exact 100.x targets",
    "targets_allowlist": sorted(ALLOWED_TARGETS),
    "nodes": NODE_SPECS,
    "role_node": ROLE_NODE,
    "shard_indices": SHARD_INDICES,
    "explicit_environment_variable": "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
    "execution_order": "three shards concurrently; mechanics after all shard children join; local aggregate after exact collection",
    "remote_root_is_user_configurable": False,
    "authorization_marker_created_by_launcher": False,
    "remote_output_template": "<fixed-base>/<manifest-sha12>/attempt_0003/<role>",
    "formal_attempt": FORMAL_ATTEMPT,
    "fresh_role_outputs_required": ["shard0", "shard1", "shard2", "mechanics"],
    "attempt2_artifact_reuse": False,
    "successful_command_count": EXPECTED_SUCCESSFUL_COMMAND_COUNT,
    "successful_log_file_count": EXPECTED_SUCCESSFUL_LOG_FILE_COUNT,
    "stdout_and_stderr_required_for_every_command": True,
    "launch_receipt_format": LAUNCH_RECEIPT_FORMAT,
    "launch_receipt_top_level_keys": sorted(LAUNCH_RECEIPT_TOP_LEVEL_KEYS),
    "launch_aggregate_summary_keys": sorted(LAUNCH_AGGREGATE_SUMMARY_KEYS),
    "launch_recursive_schema_sha256": LAUNCH_RECURSIVE_SCHEMA_SHA256,
    "launch_recursive_schema_path_count": LAUNCH_RECURSIVE_SCHEMA_PATH_COUNT,
    "launch_recursive_wildcard_mapping_paths": sorted(
        ".".join(path) for path in LAUNCH_RECURSIVE_WILDCARD_MAPPING_PATHS
    ),
    "receipt_safety_alias_policy": (
        "complete frozen recursive key-path schema first; exact top-level and "
        "aggregate-summary schemas plus the safety-alias scanner are independent "
        "secondary checks"
    ),
}


def _bundle_root(spec: dict[str, Any], run_id: str) -> str:
    if len(run_id) != 12 or any(character not in "0123456789abcdef" for character in run_id):
        raise ValueError("run id must be the manifest SHA-256 prefix")
    return f"{spec['bundle_base']}/{run_id}"


def _remote_path(spec: dict[str, Any], relative: str, run_id: str) -> str:
    if relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError("remote relative path left fixed allowlist")
    return f"{_bundle_root(spec, run_id)}/{relative}"


def _role_output(role: str, run_id: str, attempt: int) -> str:
    if attempt != FORMAL_ATTEMPT:
        raise ValueError("formal launcher is frozen to fresh attempt 3")
    return f"{ROLE_OUTPUT_BASES[role]}/{run_id}/attempt_{attempt:04d}/{role}"


def _remote_run_command(role: str, run_id: str, attempt: int) -> list[str]:
    spec = NODE_SPECS[ROLE_NODE[role]]
    target = spec["target"]
    if target not in ALLOWED_TARGETS:
        raise ValueError("remote target left exact Tailscale allowlist")
    script = _remote_path(spec, "scripts/smearing_kink/run_graphene_r2r0_formal.py", run_id)
    remote_command = [
        "env", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
        spec["conda"], "run", "-n", spec["env"], "python", script,
    ]
    if role.startswith("shard"):
        remote_command += ["shard", "--shard-id", role[-1], "--device", "cuda"]
    elif role == "mechanics":
        remote_command += ["mechanics", "--device", "cuda"]
    else:
        raise ValueError("unknown fixed formal role")
    remote_command += [
        "--output", _role_output(role, run_id, attempt),
        "--freeze-manifest", _remote_path(spec, f"{CONTROL_RELATIVE}/freeze_manifest.json", run_id),
        "--authorization-marker", _remote_path(spec, f"{CONTROL_RELATIVE}/R2R0_FORMAL_GO", run_id),
    ]
    return _ssh(target, *remote_command)


def _ssh(target: str, *remote_argv: str) -> list[str]:
    if target not in ALLOWED_TARGETS:
        raise ValueError("SSH target left exact Tailscale allowlist")
    return ["ssh", *SSH_OPTIONS, target, shlex.join(remote_argv)]


def _scp(local: str, target: str, remote_path: str) -> list[str]:
    if target not in ALLOWED_TARGETS:
        raise ValueError("SCP target left exact Tailscale allowlist")
    return ["scp", *SSH_OPTIONS, local, f"{target}:{remote_path}"]


def _scp_from(target: str, remote_path: str, local: str) -> list[str]:
    if target not in ALLOWED_TARGETS:
        raise ValueError("SCP target left exact Tailscale allowlist")
    return ["scp", *SSH_OPTIONS, "-r", f"{target}:{remote_path}", local]


def _command_receipt(command: Sequence[str]) -> dict[str, Any]:
    encoded = json.dumps(list(command), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {"argv": list(command), "sha256": __import__("hashlib").sha256(encoded).hexdigest()}


def _run_logged(command: Sequence[str], logs: Path, name: str) -> dict[str, Any]:
    record = _command_receipt(command)
    command_path = logs / f"{name}.command.json"
    stdout_path = logs / f"{name}.stdout"
    stderr_path = logs / f"{name}.stderr"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(list(command), stdout=stdout, stderr=stderr, check=False)
    record.update({
        "returncode": completed.returncode,
        "stdout_basename": stdout_path.name,
        "stderr_basename": stderr_path.name,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "command_basename": command_path.name,
        "expected_sha256": None,
        "observed_sha256": None,
    })
    _atomic_json(command_path, record)
    if completed.returncode != 0:
        raise RuntimeError(f"launcher command {name} failed with {completed.returncode}")
    return record


def _stage_and_verify(
    manifest_path: Path,
    marker_path: Path,
    logs: Path,
    run_id: str,
    attempt: int,
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    if resume:
        raise ValueError("attempt3 staging is fresh-only and cannot resume")
    commands = []
    source_relatives = {
        **{name: str(path.relative_to(ROOT)) for name, path in FORMAL_SOURCE_PATHS.items()},
        **{f"primitive_{name}": str(path.relative_to(ROOT)) for name, path in FROZEN_SOURCE_PATHS.items()},
    }
    local_files = {relative: ROOT / relative for relative in source_relatives.values()}
    local_digests = {relative: sha256(path) for relative, path in local_files.items()}
    input_digests = {
        relative: sha256(ROOT / relative) for relative in INPUT_RELATIVE.values()
    }
    for node_name, spec in NODE_SPECS.items():
        target = spec["target"]
        bundle_root = _bundle_root(spec, run_id)
        guarded_paths = [bundle_root] + [
            _role_output(role, run_id, attempt) for role in spec["roles"]
        ] + [
            _remote_path(spec, relative, run_id)
            for relative in (*local_files, *INPUT_RELATIVE.values(), f"{CONTROL_RELATIVE}/freeze_manifest.json", f"{CONTROL_RELATIVE}/R2R0_FORMAL_GO")
        ]
        for index, candidate in enumerate(guarded_paths):
            guard_argv = ["python3", "-c", REMOTE_SYMLINK_GUARD, candidate]
            if candidate == bundle_root:
                guard_argv.append("require-absent")
            commands.append(_run_logged(
                _ssh(target, *guard_argv),
                logs,
                f"{node_name}_prewrite_symlink_guard_{index}",
            ))
        fresh_targets = [
            _role_output(role, run_id, attempt) for role in spec["roles"]
        ]
        for index, candidate in enumerate(fresh_targets):
            try:
                commands.append(
                    _run_logged(
                        _ssh(target, "test", "!", "-e", candidate),
                        logs,
                        f"{node_name}_fresh_{index}",
                    )
                )
            except RuntimeError as exception:
                raise FileExistsError(
                    f"remote formal path is not fresh: {candidate}"
                ) from exception
        directories = sorted({
            str(Path(_remote_path(spec, relative, run_id)).parent)
            for relative in (*local_files, CONTROL_RELATIVE)
        })
        commands.append(_run_logged(_ssh(target, "mkdir", "-p", *directories), logs, f"{node_name}_mkdir"))
        for index, (relative, local_path) in enumerate(sorted(local_files.items())):
            commands.append(_run_logged(
                _scp(str(local_path), target, _remote_path(spec, relative, run_id)),
                logs,
                f"{node_name}_source_{index}",
            ))
        input_directories = sorted({
            str(Path(_remote_path(spec, relative, run_id)).parent)
            for relative in INPUT_RELATIVE.values()
        })
        commands.append(_run_logged(_ssh(target, "mkdir", "-p", *input_directories), logs, f"{node_name}_input_mkdir"))
        for role, relative in INPUT_RELATIVE.items():
            commands.append(_run_logged(
                _scp(str(ROOT / relative), target, _remote_path(spec, relative, run_id)),
                logs,
                f"{node_name}_{role}_stage",
            ))
        control = _remote_path(spec, CONTROL_RELATIVE, run_id)
        commands.append(_run_logged(_ssh(target, "mkdir", "-p", control), logs, f"{node_name}_control_mkdir"))
        commands.append(_run_logged(_scp(str(manifest_path), target, f"{control}/freeze_manifest.json"), logs, f"{node_name}_manifest"))
        commands.append(_run_logged(_scp(str(marker_path), target, f"{control}/R2R0_FORMAL_GO"), logs, f"{node_name}_GO"))
        expected = {
            **{_remote_path(spec, relative, run_id): local_digests[relative] for relative in local_files},
            **{
                _remote_path(spec, relative, run_id): input_digests[relative]
                for relative in INPUT_RELATIVE.values()
            },
            f"{control}/freeze_manifest.json": sha256(manifest_path),
            f"{control}/R2R0_FORMAL_GO": sha256(marker_path),
        }
        for index, (remote_path, digest) in enumerate(sorted(expected.items())):
            command = _ssh(target, "sha256sum", remote_path)
            name = f"{node_name}_verify_{index}"
            record = _run_logged(command, logs, name)
            fields = (logs / record["stdout_basename"]).read_text(
                encoding="utf-8"
            ).strip().split(maxsplit=1)
            observed = fields[0] if fields else ""
            record.update({"expected_sha256": digest, "observed_sha256": observed})
            _atomic_json(logs / record["command_basename"], record)
            if observed != digest:
                raise ValueError(f"remote staged hash mismatch on {node_name}: {remote_path}")
            commands.append(record)
        symlink_command = _ssh(target, "find", bundle_root, "-type", "l", "-print")
        symlink_record = _run_logged(
            symlink_command, logs, f"{node_name}_symlink_audit"
        )
        symlink_stdout = (logs / symlink_record["stdout_basename"]).read_text(
            encoding="utf-8"
        )
        if symlink_stdout.strip():
            raise ValueError(f"remote immutable bundle contains a symlink on {node_name}")
        commands.append(symlink_record)
    if {relative: sha256(path) for relative, path in local_files.items()} != local_digests:
        raise ValueError("local formal source snapshot changed during staging")
    if {relative: sha256(ROOT / relative) for relative in INPUT_RELATIVE.values()} != input_digests:
        raise ValueError("local formal input snapshot changed during staging")
    return commands


def _tree_manifest(root: Path, *, require_terminal: bool) -> dict[str, Any]:
    base = _reject_path(root, "collected formal artifact")
    if not base.is_dir():
        raise ValueError("collected formal artifact is not a directory")
    files = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError("collected formal artifact contains a symlink")
        if path.is_file():
            files[str(path.relative_to(base))] = sha256(path)
    if require_terminal and ("DONE" not in files or "receipt.json" not in files or "EXIT_CODE" not in files):
        raise ValueError("collected formal artifact lacks terminal provenance")
    payload = {"files": files, "file_count": len(files)}
    payload["semantic_sha256"] = __import__("hashlib").sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _artifact_manifest(root: Path) -> dict[str, Any]:
    return _tree_manifest(root, require_terminal=True)


def _validate_command_log_inventory(logs: Path) -> list[dict[str, Any]]:
    root = _reject_path(logs, "launcher command logs")
    entries = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("launcher command logs must be ordinary flat files")
    command_paths = sorted(root.glob("*.command.json"))
    if len(command_paths) != EXPECTED_SUCCESSFUL_COMMAND_COUNT:
        raise ValueError("launcher successful command count changed")
    expected_files = {path.name for path in command_paths}
    stream_basenames = set()
    commands = []
    required = {
        "argv",
        "sha256",
        "returncode",
        "stdout_basename",
        "stderr_basename",
        "stdout_sha256",
        "stderr_sha256",
        "command_basename",
        "expected_sha256",
        "observed_sha256",
    }
    for path in command_paths:
        command = json.loads(path.read_text(encoding="utf-8"))
        if set(command) != required or command["command_basename"] != path.name:
            raise ValueError("launcher command receipt stream schema changed")
        if command["returncode"] != 0:
            raise ValueError("launcher successful receipt contains nonzero command")
        if _command_receipt(command["argv"])["sha256"] != command["sha256"]:
            raise ValueError("launcher command argv hash changed")
        expected_digest = command["expected_sha256"]
        observed_digest = command["observed_sha256"]
        if (expected_digest is None) != (observed_digest is None):
            raise ValueError("launcher staged hash receipt is incomplete")
        if expected_digest is not None and (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
            or observed_digest != expected_digest
        ):
            raise ValueError("launcher staged hash receipt changed")
        for stream in ("stdout", "stderr"):
            basename = command[f"{stream}_basename"]
            if (
                not isinstance(basename, str)
                or not basename
                or Path(basename).name != basename
                or basename in stream_basenames
            ):
                raise ValueError("launcher command stream basename changed or reused")
            stream_basenames.add(basename)
            stream_path = root / basename
            if (
                stream_path.is_symlink()
                or not stream_path.is_file()
                or sha256(stream_path) != command[f"{stream}_sha256"]
            ):
                raise ValueError("launcher command stream hash changed")
            expected_files.add(basename)
        commands.append(command)
    observed_files = {path.name for path in entries}
    if (
        observed_files != expected_files
        or len(observed_files) != EXPECTED_SUCCESSFUL_LOG_FILE_COUNT
    ):
        raise ValueError("launcher command log inventory changed")
    return commands


def _validate_launch_safety_fields(receipt: dict[str, Any]) -> None:
    observed_recursive_paths = recursive_key_path_schema(
        receipt,
        wildcard_mapping_paths=LAUNCH_RECURSIVE_WILDCARD_MAPPING_PATHS,
    )
    if len(observed_recursive_paths) != LAUNCH_RECURSIVE_SCHEMA_PATH_COUNT:
        raise ValueError(
            "launcher recovery recursive key-path schema count changed: "
            f"observed={len(observed_recursive_paths)}, "
            f"expected={LAUNCH_RECURSIVE_SCHEMA_PATH_COUNT}"
        )
    observed_recursive_schema = recursive_key_path_schema_sha256(
        receipt,
        wildcard_mapping_paths=LAUNCH_RECURSIVE_WILDCARD_MAPPING_PATHS,
    )
    if observed_recursive_schema != LAUNCH_RECURSIVE_SCHEMA_SHA256:
        raise ValueError(
            "launcher recovery recursive key-path schema changed: "
            f"observed={observed_recursive_schema}, "
            f"expected={LAUNCH_RECURSIVE_SCHEMA_SHA256}"
        )
    if set(receipt) != LAUNCH_RECEIPT_TOP_LEVEL_KEYS:
        extra = sorted(set(receipt) - LAUNCH_RECEIPT_TOP_LEVEL_KEYS)
        missing = sorted(LAUNCH_RECEIPT_TOP_LEVEL_KEYS - set(receipt))
        raise ValueError(
            "launcher recovery top-level receipt schema changed; "
            f"extra={extra}, missing={missing}"
        )
    if receipt.get("format") != LAUNCH_RECEIPT_FORMAT:
        raise ValueError("launcher recovery receipt format changed")
    if receipt.get("status") not in {
        "R2R0_FORMAL_REPRESENTATION_PRECHECK_PASSED",
        "R2R0_FORMAL_REPRESENTATION_PRECHECK_FAILED",
        "R2R0_FORMAL_NUMERICALLY_INCONCLUSIVE",
    }:
        raise ValueError("launcher recovery scientific aggregate status changed")
    if set(receipt.get("aggregate_receipt") or {}) != LAUNCH_AGGREGATE_SUMMARY_KEYS:
        raise ValueError("launcher recovery aggregate summary schema changed")
    for key, value in LAUNCH_SAFETY_FIELDS.items():
        if receipt.get(key) is not value:
            raise ValueError(f"launcher recovery safety field changed: {key}")
    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = (*path, key)
                if _is_suspicious_safety_key(key):
                    benign = any(
                        _key_path_matches(child_path, pattern)
                        for pattern in LAUNCH_BENIGN_SCIENTIFIC_PATHS
                    )
                    safety_false = any(
                        _key_path_matches(child_path, pattern)
                        for pattern in LAUNCH_SAFETY_FALSE_PATHS
                    )
                    if not benign and not safety_false:
                        raise ValueError(
                            "launcher recovery unexpected safety-like key: "
                            + ".".join(child_path)
                        )
                    if safety_false and child is not False:
                        raise ValueError(
                            "launcher recovery safety-like field is not exact false: "
                            + ".".join(child_path)
                        )
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(receipt, ())


def _validate_launch_recovery(
    root: Path,
    receipt: dict[str, Any],
    authorization: dict[str, Any],
    run_id: str,
    attempt: int,
) -> None:
    present = {name for name in ("RUNNING", "FAILED", "DONE") if (root / name).exists()}
    if present != {"DONE"} or (root / "EXIT_CODE").read_bytes() != b"0\n":
        raise ValueError("launcher recovery terminal XOR/exit code changed")
    if receipt.get("format") != LAUNCH_RECEIPT_FORMAT:
        raise ValueError("launcher recovery receipt format changed")
    if receipt.get("status") not in {
        "R2R0_FORMAL_REPRESENTATION_PRECHECK_PASSED",
        "R2R0_FORMAL_REPRESENTATION_PRECHECK_FAILED",
        "R2R0_FORMAL_NUMERICALLY_INCONCLUSIVE",
    }:
        raise ValueError("launcher recovery scientific aggregate status changed")
    canonical_plan = json.loads(json.dumps(PLAN, sort_keys=True))
    if receipt.get("plan") != canonical_plan or not all(receipt.get("partial_state", {}).values()):
        raise ValueError("launcher recovery plan or completion state changed")
    if receipt.get("run_id") != run_id or receipt.get("attempt") != attempt:
        raise ValueError("launcher recovery run id/attempt changed")
    if receipt.get("formal_contract_sha256") != FORMAL_CONTRACT_SHA256:
        raise ValueError("launcher recovery formal contract is stale")
    if receipt.get("execution_authorization") != authorization:
        raise ValueError("launcher recovery authorization is stale")
    _validate_launch_safety_fields(receipt)
    observed = {
        role: _artifact_manifest(root / "artifacts" / role)
        for role in ("shard0", "shard1", "shard2", "mechanics")
    }
    observed["aggregate"] = _artifact_manifest(root / "aggregate")
    if observed != receipt.get("artifact_manifests"):
        raise ValueError("launcher recovery artifact manifests changed")
    if _tree_manifest(root / "logs", require_terminal=False) != receipt.get("log_manifest"):
        raise ValueError("launcher recovery command logs changed")
    if _tree_manifest(root / "control", require_terminal=False) != receipt.get("control_manifest"):
        raise ValueError("launcher recovery immutable control snapshot changed")
    validated_commands = _validate_command_log_inventory(root / "logs")
    if validated_commands != receipt.get("commands"):
        raise ValueError("launcher recovery command receipt inventory changed")
    aggregate_receipt_path = root / "aggregate" / "receipt.json"
    aggregate_receipt = json.loads(aggregate_receipt_path.read_text(encoding="utf-8"))
    _validate_receipt_contract(
        aggregate_receipt,
        "launcher recovered aggregate",
        receipt_kind="aggregate",
    )
    expected = receipt.get("aggregate_receipt", {})
    if (
        sha256(aggregate_receipt_path) != expected.get("sha256")
        or aggregate_receipt.get("status") != expected.get("status")
        or aggregate_receipt.get("arrays_sha256") != expected.get("arrays_sha256")
        or receipt.get("status") != aggregate_receipt.get("status")
    ):
        raise ValueError("launcher recovery aggregate receipt changed")


def execute(
    *,
    freeze_manifest: Path,
    authorization_marker: Path,
    collect_root: Path,
    attempt: int,
) -> dict[str, Any]:
    authorization = validate_execution_authorization(freeze_manifest, authorization_marker)
    run_id = authorization["freeze_manifest_sha256"][:12]
    if attempt != FORMAL_ATTEMPT:
        raise ValueError("formal execution is frozen to fresh attempt 3")
    existing = _reject_path(collect_root, "launcher collection root", must_exist=False)
    if existing.is_dir() and (existing / "DONE").exists():
        receipt_path = existing / "launch_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = receipt["status"] + "\n" + sha256(receipt_path) + "\n"
        _validate_launch_recovery(existing, receipt, authorization, run_id, attempt)
        if (existing / "DONE").read_text(encoding="ascii") != expected:
            raise ValueError("launcher DONE marker changed")
        return {**receipt, "completion_recovered_without_recompute": True}
    if existing.is_dir() and any((existing / marker).exists() for marker in ("RUNNING", "FAILED")):
        raise FileExistsError("interrupted/failed launcher roots require a new attempt and fresh collection root")
    root = _fresh_run_root(collect_root)
    state = {"staging_complete": False, "shards_complete": False, "mechanics_complete": False, "collection_complete": False}
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    command_receipts = []
    try:
        control = root / "control"
        control.mkdir()
        manifest_snapshot = control / "freeze_manifest.json"
        marker_snapshot = control / "R2R0_FORMAL_GO"
        _atomic_bytes(manifest_snapshot, Path(freeze_manifest).read_bytes())
        _atomic_bytes(marker_snapshot, Path(authorization_marker).read_bytes())
        if validate_execution_authorization(manifest_snapshot, marker_snapshot) != authorization:
            raise ValueError("immutable launcher control snapshot changed authorization")
        state_path = root / "partial_state.json"
        def write_state() -> None:
            _atomic_json(state_path, {**state, "execution_authorization": authorization, "run_id": run_id, "attempt": attempt})
        write_state()
        if not state["staging_complete"]:
            command_receipts.extend(_stage_and_verify(
                manifest_snapshot, marker_snapshot, logs, run_id, attempt, resume=False
            ))
            state["staging_complete"] = True
            write_state()
        if not state["shards_complete"]:
            processes = {}
            streams = {}
            for role in ("shard0", "shard1", "shard2"):
                spec = NODE_SPECS[ROLE_NODE[role]]
                for marker in ("RUNNING", "FAILED"):
                    command_receipts.append(_run_logged(
                        _ssh(spec["target"], "test", "!", "-e", f"{_role_output(role, run_id, attempt)}/{marker}"),
                        logs,
                        f"{role}_precheck_{marker}",
                    ))
                command = _remote_run_command(role, run_id, attempt)
                stdout = (logs / f"{role}.stdout").open("xb")
                stderr = (logs / f"{role}.stderr").open("xb")
                streams[role] = (stdout, stderr)
                try:
                    processes[role] = (
                        command,
                        subprocess.Popen(command, stdout=stdout, stderr=stderr),
                    )
                except BaseException:
                    stdout.close()
                    stderr.close()
                    for started_role, (_started_command, process) in processes.items():
                        process.wait()
                        started_stdout, started_stderr = streams[started_role]
                        started_stdout.close()
                        started_stderr.close()
                    raise
            return_codes = {}
            for role, (command, process) in processes.items():
                return_codes[role] = process.wait()
                stdout, stderr = streams[role]
                stdout.close(); stderr.close()
                record = _command_receipt(command)
                record.update({
                    "returncode": return_codes[role],
                    "stdout_basename": f"{role}.stdout",
                    "stderr_basename": f"{role}.stderr",
                    "stdout_sha256": sha256(logs / f"{role}.stdout"),
                    "stderr_sha256": sha256(logs / f"{role}.stderr"),
                    "command_basename": f"{role}.command.json",
                    "expected_sha256": None,
                    "observed_sha256": None,
                })
                _atomic_json(logs / f"{role}.command.json", record)
                command_receipts.append(record)
            if any(code != 0 for code in return_codes.values()):
                raise RuntimeError(f"formal shards failed after all children joined: {return_codes}")
            state["shards_complete"] = True
            write_state()
        if not state["mechanics_complete"]:
            mechanics_spec = NODE_SPECS[ROLE_NODE["mechanics"]]
            for marker in ("RUNNING", "FAILED"):
                command_receipts.append(_run_logged(
                    _ssh(mechanics_spec["target"], "test", "!", "-e", f"{_role_output('mechanics', run_id, attempt)}/{marker}"),
                    logs,
                    f"mechanics_precheck_{marker}",
                ))
            command_receipts.append(_run_logged(_remote_run_command("mechanics", run_id, attempt), logs, "mechanics"))
            state["mechanics_complete"] = True
            write_state()
        artifacts = root / "artifacts"
        artifacts.mkdir(exist_ok=True)
        if not state["collection_complete"]:
            for role in ("shard0", "shard1", "shard2", "mechanics"):
                if (artifacts / role).exists():
                    raise FileExistsError(f"partial collected role requires a new launcher attempt: {role}")
                spec = NODE_SPECS[ROLE_NODE[role]]
                command_receipts.append(_run_logged(
                    _scp_from(spec["target"], _role_output(role, run_id, attempt), str(artifacts / role)),
                    logs,
                    f"collect_{role}",
                ))
            state["collection_complete"] = True
            write_state()
        aggregate_command = [
            "/Users/howardwang/miniconda3/bin/conda", "run", "-n", "phonon", "python",
            str(ROOT / "scripts/smearing_kink/aggregate_graphene_r2r0_formal.py"),
        ]
        for role in ("shard0", "shard1", "shard2"):
            aggregate_command += ["--shard", str(artifacts / role)]
        aggregate_command += [
            "--mechanics", str(artifacts / "mechanics"),
            "--output", str(root / "aggregate"),
            "--freeze-manifest", str(manifest_snapshot),
            "--authorization-marker", str(marker_snapshot),
        ]
        command_receipts.append(_run_logged(aggregate_command, logs, "aggregate"))
        aggregate_receipt_path = root / "aggregate" / "receipt.json"
        aggregate_receipt = json.loads(aggregate_receipt_path.read_text(encoding="utf-8"))
        _validate_receipt_contract(
            aggregate_receipt,
            "launcher fresh aggregate",
            receipt_kind="aggregate",
        )
        if aggregate_receipt.get("execution_authorization") != authorization:
            raise ValueError("aggregate execution authorization differs from launcher snapshot")
        if validate_execution_authorization(manifest_snapshot, marker_snapshot) != authorization:
            raise ValueError("launcher control snapshot changed before completion")
        if validate_execution_authorization(freeze_manifest, authorization_marker) != authorization:
            raise ValueError("launcher authorization inputs changed during execution")
        aggregate_status = aggregate_receipt["status"]
        artifact_manifests = {
            role: _artifact_manifest(artifacts / role)
            for role in ("shard0", "shard1", "shard2", "mechanics")
        }
        artifact_manifests["aggregate"] = _artifact_manifest(root / "aggregate")
        command_receipts = _validate_command_log_inventory(logs)
        receipt = {
            "format": LAUNCH_RECEIPT_FORMAT,
            "status": aggregate_status,
            "formal_contract_sha256": FORMAL_CONTRACT_SHA256,
            "execution_authorization": authorization,
            "plan": PLAN,
            "commands": command_receipts,
            "partial_state": state,
            "run_id": run_id,
            "attempt": attempt,
            "artifact_manifests": artifact_manifests,
            "log_manifest": _tree_manifest(logs, require_terminal=False),
            "control_manifest": _tree_manifest(control, require_terminal=False),
            "aggregate_receipt": {
                "sha256": sha256(aggregate_receipt_path),
                "status": aggregate_status,
                "arrays_sha256": aggregate_receipt["arrays_sha256"],
                "representation_precheck_pass": aggregate_receipt["representation_precheck_pass"],
                "numerically_inconclusive": aggregate_receipt["numerically_inconclusive"],
            },
            "fit_or_training": False,
            "held_or_support_access": False,
        }
        _validate_launch_safety_fields(receipt)
        receipt_path = root / "launch_receipt.json"
        _atomic_json(receipt_path, receipt)
        _success(root, receipt_path, aggregate_status)
        return receipt
    except BaseException as exception:
        _failure(root, exception)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-freeze-manifest", type=Path)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--authorization-marker", type=Path)
    parser.add_argument("--collect-root", type=Path)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare_freeze_manifest is not None:
        if args.execute:
            raise ValueError("manifest preparation and execution are separate operations")
        prepare_freeze_manifest(args.prepare_freeze_manifest)
        return
    if args.execute:
        if args.freeze_manifest is None or args.authorization_marker is None or args.collect_root is None or args.attempt is None:
            raise ValueError("execute requires manifest, marker, collect root, and positive attempt")
        execute(
            freeze_manifest=args.freeze_manifest,
            authorization_marker=args.authorization_marker,
            collect_root=args.collect_root,
            attempt=args.attempt,
        )
        return
    text = json.dumps(PLAN, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        if args.output.exists():
            raise FileExistsError("launch plan output must be fresh")
        _atomic_json(args.output, PLAN)


if __name__ == "__main__":
    main()
