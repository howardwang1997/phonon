#!/usr/bin/env python3
"""Fail-closed, no-support R2O core+Taylor-null-tail smoke protocol.

The module contains the node runner, dual-node evaluator, and a plan-only
Tailscale launcher.  It deliberately has no support/fold preparation code and
the launcher cannot execute a remote command.  A later independent review must
authorize use with a real formal bundle.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from ase import Atoms

from graphene_r2o_routed_tail import (
    EXPECTED_INVARIANT_WIDTH,
    LoadedFormalR2O,
    R2ORoutedTailSpecification,
    R2OTaylorNullRoutedTail,
    audit_core_tail_taylor_linearity,
    audit_paired_native_fp64_sensitivity,
    build_order_mic_query,
    canonical_json_sha256,
    ensure_fp64_module,
    frozen_graph_semantics,
    load_formal_r2o_bundle,
    query_raw_invariants,
    resolve_relative_artifact,
    r2o_invariant_schema,
    sha256,
    source_order_tail_forces,
    state_dict_sha256,
    strict_json,
    tail_taylor_remainder,
    validate_loaded_formal_r2o,
    validate_r2o_invariant_schema,
    whole_energy_taylor2_null,
)
from graphene_r2o_taylor_null import TaylorResult, mace_interaction_energy


SMOKE_INPUT_FORMAT = "graphene_r2o_no_support_actual_tail_smoke_input_v1"
SMOKE_INPUT_STATUS = "frozen_for_pre_support_dual_node_smoke"
TAIL_CHECKPOINT_FORMAT = "graphene_r2o_actual_taylor_null_tail_checkpoint_v1"
TAIL_CHECKPOINT_PURPOSE = "nonzero_pre_support_mechanics_smoke_only"
RUNTIME_CONTRACT_FORMAT = "graphene_r2o_tail_runtime_contract_v1"
RUNTIME_CONTRACT_STATUS = "frozen_before_dual_node_smoke"
NODE_RESULT_FORMAT = "graphene_r2o_actual_tail_node_smoke_result_v1"
NODE_RESULT_STATUS = "R2O_actual_tail_node_smoke_passed"
DUAL_RESULT_FORMAT = "graphene_r2o_actual_tail_dual_node_smoke_result_v1"
DUAL_RESULT_STATUS = "R2O_actual_tail_dual_node_smoke_passed"
LAUNCH_PLAN_FORMAT = "graphene_r2o_actual_tail_dual_node_launch_plan_v1"
LAUNCH_PLAN_STATUS = "awaiting_independent_review_remote_launch_forbidden"
FORMAL_GO_FORMAT = "graphene_r2o_actual_tail_formal_smoke_independent_go_v1"
FORMAL_GO_STATUS = "independently_reviewed_real_formal_no_support_smoke_GO"

SYNTHETIC_MODE = "synthetic_portable_formal_receipt_test"
FORMAL_ACTUAL_MODE = "formal_no_support_actual_tail_smoke"
ALLOWED_MODES = {SYNTHETIC_MODE, FORMAL_ACTUAL_MODE}

DONE_BYTES = (NODE_RESULT_STATUS + "\n").encode("utf-8")
DUAL_DONE_BYTES = (DUAL_RESULT_STATUS + "\n").encode("utf-8")
SUCCESS_EXIT_BYTES = b"0\n"
FAILURE_EXIT_BYTES = b"1\n"
FORBIDDEN_PATH_TOKENS = ("support", "seed2", "reserved", "outer_fold")
CM1_PER_SQRT_EV_A2_AMU = 521.4708983725066
RUNTIME_DEVICE_POLICY = {
    "synthetic_mode_allowed_device_types": ["cpu", "cuda"],
    "formal_actual_mode_required_device_type": "cuda",
    "GPU_name_and_capability_may_differ_across_nodes": True,
    "semantic_environment_must_match_exactly": True,
}
SMOKE_FORBIDDEN_STATE = {
    "support9_opened": False,
    "seed2_opened": False,
    "outer_folds_prepared": False,
    "remote_launch_authorized": False,
}

SMOKE_THRESHOLDS = {
    "reference_energy_abs_eV": 1.0e-10,
    "reference_force_max_abs_eV_A": 1.0e-9,
    "reference_Hessian_max_abs_eV_A2": 1.0e-7,
    "reference_Hessian_antisymmetry_eV_A2": 1.0e-7,
    "reference_Hessian_ASR_eV_A2": 1.0e-7,
    "reference_frequency_drift_bound_cm-1": 2.0,
    "combined_separate_energy_eV": 1.0e-10,
    "combined_separate_force_eV_A": 1.0e-9,
    "energy_force_FD_eV_A": 1.0e-5,
    "force_Hessian_FD_eV_A2": 1.0e-5,
    "nonreference_Hessian_antisymmetry_eV_A2": 1.0e-7,
    "nonreference_Hessian_translation_ASR_eV_A2": 1.0e-7,
    "O3_energy_eV": 1.0e-6,
    "O3_force_eV_A": 1.0e-5,
    "translation_energy_eV": 1.0e-9,
    "translation_force_eV_A": 1.0e-8,
    "order_MIC_energy_eV": 1.0e-6,
    "order_MIC_force_eV_A": 1.0e-5,
    "size_energy_eV": 1.0e-7,
    "size_force_eV_A": 1.0e-5,
    "graph_source_geometry_A": 1.0e-6,
    "nonzero_tail_energy_or_force": 1.0e-12,
    "cross_node_probe_energy_eV": 1.0e-6,
    "cross_node_probe_force_eV_A": 1.0e-5,
}

SMOKE_SOURCE_FILENAMES = {
    "smoke_module": "graphene_r2o_tail_smoke.py",
    "node_runner": "run_graphene_r2o_tail_smoke.py",
    "dual_evaluator": "evaluate_graphene_r2o_tail_smoke_dual.py",
    "plan_only_launcher": "launch_graphene_r2o_tail_smoke_dual.py",
    "routed_tail_module": "graphene_r2o_routed_tail.py",
    "taylor_wrapper": "graphene_r2o_taylor_null.py",
}


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    if value != value.lower():
        raise ValueError(f"{label} must use lowercase hexadecimal")
    return value


def _strict_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a canonical POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise ValueError(f"{label} must be a canonical relative path")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} contains traversal")
    lowered = path.as_posix().casefold()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"{label} names unopened or reserved data")
    return path


def _artifact(root: Path, relative: Any, label: str) -> Path:
    _strict_relative_path(relative, label)
    return resolve_relative_artifact(root, relative, label)


def _new_relative_artifact(root: Path, relative: Any, label: str) -> Path:
    path = _strict_relative_path(relative, label)
    root = Path(root).resolve(strict=True)
    candidate = root
    for component in path.parts[:-1]:
        candidate = candidate / component
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"{label} may not contain symlinks")
        candidate.mkdir(exist_ok=True)
    target = candidate / path.name
    if target.exists() and target.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    resolved_parent = target.parent.resolve(strict=True)
    if root != resolved_parent and root not in resolved_parent.parents:
        raise ValueError(f"{label} escapes its portable root")
    return target


def _read_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _safe_torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("tail smoke requires torch.load(weights_only=True)") from error
    if not isinstance(payload, dict):
        raise ValueError("tail checkpoint must contain a plain dictionary")
    return payload


def smoke_source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    result = {}
    for label, filename in SMOKE_SOURCE_FILENAMES.items():
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required R2O smoke source is missing: {label}")
        result[label] = sha256(path)
    return result


def smoke_protocol() -> dict:
    payload = {
        "format": "graphene_r2o_no_support_actual_tail_smoke_protocol_v1",
        "dtype": "torch.float64",
        "reference_atom_count": 72,
        "size_and_paired_native_atom_counts": [72, 128],
        "probe_displacement": {"atom": 4, "vector_A": [0.012, -0.007, 0.018]},
        "localized_size_displacement": {"atom": 0, "axis": 2, "value_A": 0.03},
        "finite_difference": {"atom": 4, "axis": 2, "step_A": 1.0e-4},
        "O3_seed": 83,
        "permutation_seed": 83,
        "global_translation_A": [0.031, -0.027, 0.019],
        "whole_energy_policy": "core_Taylor2_plus_tail_Taylor2_and_combined_once_linearity",
        "raw_MACE_direct_deployment": False,
        "thresholds": copy.deepcopy(SMOKE_THRESHOLDS),
    }
    payload["protocol_sha256"] = canonical_json_sha256(payload)
    return payload


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def runtime_fingerprint(
    device: torch.device | str,
    *,
    synthetic_hostname_override: str | None = None,
) -> dict:
    requested = torch.device(device)
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("R2O tail smoke supports only CPU or CUDA")
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA smoke requested but CUDA is unavailable")
    semantic = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "mace_torch_version": _distribution_version("mace-torch"),
        "e3nn_version": _distribution_version("e3nn"),
        "ase_version": _distribution_version("ase"),
        "scipy_version": _distribution_version("scipy"),
        "torch_geometric_version": _distribution_version("torch-geometric"),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if requested.type == "cuda":
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        execution_device = {
            "type": "cuda",
            "index": int(index),
            "name": properties.name,
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        }
    else:
        execution_device = {"type": "cpu", "index": None, "name": platform.processor()}
    if synthetic_hostname_override is not None:
        if not synthetic_hostname_override.replace("-", "").replace("_", "").isalnum():
            raise ValueError("synthetic hostname override is not a simple identifier")
        hostname = synthetic_hostname_override
        hostname_source = "synthetic_test_override"
    else:
        hostname = platform.node()
        hostname_source = "platform.node"
    if not hostname:
        raise ValueError("runtime hostname is empty")
    host_identity = {
        "hostname": hostname,
        "hostname_source": hostname_source,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
    execution = {
        "host_identity": host_identity,
        "host_identity_sha256": canonical_json_sha256(host_identity),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "device": execution_device,
    }
    return {
        "format": "graphene_r2o_tail_runtime_fingerprint_v1",
        "semantic_environment": semantic,
        "semantic_environment_sha256": canonical_json_sha256(semantic),
        "execution_environment": execution,
        "execution_environment_sha256": canonical_json_sha256(execution),
    }


def freeze_runtime_contract(path: Path, *, device: torch.device | str) -> dict:
    fingerprint = runtime_fingerprint(device)
    payload = {
        "format": RUNTIME_CONTRACT_FORMAT,
        "status": RUNTIME_CONTRACT_STATUS,
        "semantic_environment": fingerprint["semantic_environment"],
        "semantic_environment_sha256": fingerprint[
            "semantic_environment_sha256"
        ],
        "device_policy": copy.deepcopy(RUNTIME_DEVICE_POLICY),
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    strict_json(path, payload)
    return payload


def validate_runtime_contract(
    path: Path,
    *,
    expected_sha256: str,
    fingerprint: dict,
    mode: str,
) -> dict:
    if sha256(path) != _validate_sha256(expected_sha256, "runtime contract SHA-256"):
        raise ValueError("runtime contract artifact SHA-256 mismatch")
    payload = _read_json(path, "runtime contract")
    expected_keys = {
        "format",
        "status",
        "semantic_environment",
        "semantic_environment_sha256",
        "device_policy",
        "contract_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("runtime contract fields changed")
    recorded = payload.pop("contract_sha256", None)
    if recorded is None or canonical_json_sha256(payload) != recorded:
        raise ValueError("runtime contract canonical SHA-256 mismatch")
    payload["contract_sha256"] = recorded
    if payload.get("format") != RUNTIME_CONTRACT_FORMAT:
        raise ValueError("wrong R2O runtime contract format")
    if payload.get("status") != RUNTIME_CONTRACT_STATUS:
        raise ValueError("R2O runtime contract is not frozen")
    if payload.get("device_policy") != RUNTIME_DEVICE_POLICY:
        raise ValueError("runtime device policy changed")
    semantic = payload.get("semantic_environment")
    if canonical_json_sha256(semantic) != payload.get("semantic_environment_sha256"):
        raise ValueError("runtime semantic environment hash mismatch")
    if semantic != fingerprint.get("semantic_environment"):
        raise ValueError("runtime semantic environment differs from frozen contract")
    if payload["semantic_environment_sha256"] != fingerprint.get(
        "semantic_environment_sha256"
    ):
        raise ValueError("runtime semantic fingerprint SHA-256 differs")
    device_type = fingerprint["execution_environment"]["device"]["type"]
    if mode == FORMAL_ACTUAL_MODE and device_type != "cuda":
        raise ValueError("formal actual-tail smoke requires CUDA")
    if mode == SYNTHETIC_MODE and device_type not in {"cpu", "cuda"}:
        raise ValueError("synthetic smoke has an unsupported device type")
    return payload


def make_tail_checkpoint_payload(
    formal: LoadedFormalR2O,
    tail: R2OTaylorNullRoutedTail,
    schema: dict,
) -> dict:
    validate_loaded_formal_r2o(formal)
    validate_r2o_invariant_schema(schema)
    if schema["formal_binding"] != asdict(formal.binding):
        raise ValueError("tail checkpoint schema belongs to another formal bundle")
    ensure_fp64_module(tail, "R2O actual smoke tail")
    specification = tail.specification
    specification.validate()
    state = {name: tensor.detach().cpu().clone() for name, tensor in tail.state_dict().items()}
    payload = {
        "format": TAIL_CHECKPOINT_FORMAT,
        "purpose": TAIL_CHECKPOINT_PURPOSE,
        "deployable": False,
        "support_or_seed2_opened": False,
        "specification": asdict(specification),
        "state_dict": state,
        "state_dict_sha256": state_dict_sha256(tail),
        "formal_binding": asdict(formal.binding),
        "formal_receipt_sha256": formal.receipt_sha256,
        "schema": copy.deepcopy(schema),
        "schema_sha256": schema["schema_sha256"],
        "graph_semantics_sha256": frozen_graph_semantics()[
            "graph_semantics_sha256"
        ],
    }
    return payload


def write_tail_checkpoint(
    path: Path,
    formal: LoadedFormalR2O,
    tail: R2OTaylorNullRoutedTail,
    schema: dict,
) -> str:
    _atomic_torch_save(make_tail_checkpoint_payload(formal, tail, schema), path)
    return sha256(path)


def _materialize_tail_checkpoint(
    path: Path,
    *,
    formal: LoadedFormalR2O,
    expected_artifact_sha256: str,
    expected_state_sha256: str,
    expected_schema_sha256: str,
    device: torch.device | str,
) -> tuple[R2OTaylorNullRoutedTail, dict]:
    if sha256(path) != _validate_sha256(
        expected_artifact_sha256, "tail checkpoint artifact SHA-256"
    ):
        raise ValueError("tail checkpoint artifact SHA-256 mismatch")
    checkpoint = _safe_torch_load(path)
    expected_keys = {
        "format",
        "purpose",
        "deployable",
        "support_or_seed2_opened",
        "specification",
        "state_dict",
        "state_dict_sha256",
        "formal_binding",
        "formal_receipt_sha256",
        "schema",
        "schema_sha256",
        "graph_semantics_sha256",
    }
    if set(checkpoint) != expected_keys:
        raise ValueError("tail checkpoint fields changed")
    if checkpoint["format"] != TAIL_CHECKPOINT_FORMAT:
        raise ValueError("wrong R2O actual-tail checkpoint format")
    if checkpoint["purpose"] != TAIL_CHECKPOINT_PURPOSE:
        raise ValueError("tail checkpoint is not the pre-support mechanics adapter")
    if checkpoint["deployable"] is not False:
        raise ValueError("pre-support smoke tail must not be deployable")
    if checkpoint["support_or_seed2_opened"] is not False:
        raise ValueError("tail checkpoint reports forbidden data access")
    if checkpoint["formal_binding"] != asdict(formal.binding):
        raise ValueError("tail checkpoint belongs to another formal bundle")
    if checkpoint["formal_receipt_sha256"] != formal.receipt_sha256:
        raise ValueError("tail checkpoint belongs to another formal receipt")
    if checkpoint["schema_sha256"] != expected_schema_sha256:
        raise ValueError("tail checkpoint schema SHA-256 differs from smoke input")
    schema = checkpoint["schema"]
    validate_r2o_invariant_schema(schema)
    if schema["schema_sha256"] != checkpoint["schema_sha256"]:
        raise ValueError("tail checkpoint embedded schema hash differs")
    if schema["formal_binding"] != asdict(formal.binding):
        raise ValueError("tail checkpoint embedded schema binding differs")
    graph_hash = frozen_graph_semantics()["graph_semantics_sha256"]
    if checkpoint["graph_semantics_sha256"] != graph_hash:
        raise ValueError("tail checkpoint graph semantics differ from runtime")
    specification = R2ORoutedTailSpecification(**checkpoint["specification"])
    specification.validate()
    state = checkpoint["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("tail checkpoint state_dict is not a mapping")
    for required in ("feature_mean", "feature_scale", "pristine_invariants"):
        value = state.get(required)
        if not torch.is_tensor(value) or value.dtype != torch.float64:
            raise ValueError(f"tail checkpoint {required} is missing or not FP64")
    tail = R2OTaylorNullRoutedTail(
        specification,
        state["feature_mean"],
        state["feature_scale"],
        state["pristine_invariants"],
    )
    tail.load_state_dict(state, strict=True)
    tail = tail.to(device=device, dtype=torch.float64)
    tail.requires_grad_(False)
    tail.eval()
    observed = state_dict_sha256(tail)
    if observed != checkpoint["state_dict_sha256"]:
        raise ValueError("tail semantic state SHA-256 differs from checkpoint")
    if observed != _validate_sha256(expected_state_sha256, "tail state SHA-256"):
        raise ValueError("tail semantic state SHA-256 differs from smoke input")
    ensure_fp64_module(tail, "loaded R2O actual smoke tail")
    return tail, schema


@dataclass
class LoadedSmokeInput:
    portable_root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict
    formal: LoadedFormalR2O
    tail: R2OTaylorNullRoutedTail
    schema: dict
    runtime_contract: dict
    runtime_fingerprint: dict


def freeze_smoke_input_manifest(
    portable_root: Path,
    manifest_relative_path: str,
    *,
    mode: str,
    formal_receipt_relative_path: str,
    expected_formal_receipt_sha256: str,
    expected_gate_sha256: str,
    expected_bundle_sha256: str,
    tail_checkpoint_relative_path: str,
    expected_tail_checkpoint_sha256: str,
    expected_tail_state_sha256: str,
    expected_schema_sha256: str,
    runtime_contract_relative_path: str,
    expected_runtime_contract_sha256: str,
) -> dict:
    if mode not in ALLOWED_MODES:
        raise ValueError("unsupported R2O smoke input mode")
    root = Path(portable_root).resolve(strict=True)
    paths = {
        "formal_receipt": formal_receipt_relative_path,
        "tail_checkpoint": tail_checkpoint_relative_path,
        "runtime_contract": runtime_contract_relative_path,
    }
    resolved = {
        key: _artifact(root, value, f"smoke input {key}")
        for key, value in paths.items()
    }
    expected_hashes = {
        "formal_receipt": _validate_sha256(
            expected_formal_receipt_sha256, "formal receipt SHA-256"
        ),
        "tail_checkpoint": _validate_sha256(
            expected_tail_checkpoint_sha256, "tail checkpoint SHA-256"
        ),
        "runtime_contract": _validate_sha256(
            expected_runtime_contract_sha256, "runtime contract SHA-256"
        ),
    }
    for key, path in resolved.items():
        if sha256(path) != expected_hashes[key]:
            raise ValueError(f"smoke input {key} artifact SHA-256 mismatch")
    protocol = smoke_protocol()
    payload = {
        "format": SMOKE_INPUT_FORMAT,
        "status": SMOKE_INPUT_STATUS,
        "mode": mode,
        "paths": paths,
        "sha256": expected_hashes,
        "formal_loader_expected": {
            "gate_sha256": _validate_sha256(expected_gate_sha256, "gate SHA-256"),
            "bundle_sha256": _validate_sha256(
                expected_bundle_sha256, "bundle SHA-256"
            ),
        },
        "tail_expected": {
            "state_dict_sha256": _validate_sha256(
                expected_tail_state_sha256, "tail state SHA-256"
            ),
            "schema_sha256": _validate_sha256(
                expected_schema_sha256, "tail schema SHA-256"
            ),
            "purpose": TAIL_CHECKPOINT_PURPOSE,
        },
        "graph_semantics_sha256": frozen_graph_semantics()[
            "graph_semantics_sha256"
        ],
        "protocol_sha256": protocol["protocol_sha256"],
        "source_sha256": smoke_source_hashes(),
        "forbidden": copy.deepcopy(SMOKE_FORBIDDEN_STATE),
    }
    target = _new_relative_artifact(
        root, manifest_relative_path, "smoke input manifest path"
    )
    strict_json(target, payload)
    return payload


def _validate_smoke_manifest_static(
    portable_root: Path,
    manifest_relative_path: str,
    *,
    expected_manifest_sha256: str,
) -> tuple[dict, dict[str, Path], Path]:
    root = Path(portable_root).resolve(strict=True)
    manifest_path = _artifact(root, manifest_relative_path, "smoke input manifest")
    if sha256(manifest_path) != _validate_sha256(
        expected_manifest_sha256, "smoke input manifest SHA-256"
    ):
        raise ValueError("smoke input manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path, "smoke input manifest")
    expected_keys = {
        "format",
        "status",
        "mode",
        "paths",
        "sha256",
        "formal_loader_expected",
        "tail_expected",
        "graph_semantics_sha256",
        "protocol_sha256",
        "source_sha256",
        "forbidden",
    }
    if set(manifest) != expected_keys:
        raise ValueError("smoke input manifest fields changed")
    if manifest["format"] != SMOKE_INPUT_FORMAT or manifest["status"] != SMOKE_INPUT_STATUS:
        raise ValueError("smoke input manifest is not frozen")
    if manifest["mode"] not in ALLOWED_MODES:
        raise ValueError("unsupported smoke input mode")
    if manifest["forbidden"] != SMOKE_FORBIDDEN_STATE:
        raise ValueError("smoke input permits forbidden data or launch state")
    if manifest["protocol_sha256"] != smoke_protocol()["protocol_sha256"]:
        raise ValueError("smoke protocol SHA-256 changed")
    if manifest["graph_semantics_sha256"] != frozen_graph_semantics()[
        "graph_semantics_sha256"
    ]:
        raise ValueError("smoke input graph semantics changed")
    if manifest["source_sha256"] != smoke_source_hashes():
        raise ValueError("smoke source code SHA-256 changed")
    if set(manifest["paths"]) != {"formal_receipt", "tail_checkpoint", "runtime_contract"}:
        raise ValueError("smoke input artifact path set changed")
    if set(manifest["sha256"]) != set(manifest["paths"]):
        raise ValueError("smoke input artifact hash set changed")
    if set(manifest["formal_loader_expected"]) != {
        "gate_sha256",
        "bundle_sha256",
    }:
        raise ValueError("smoke input formal-loader fields changed")
    if set(manifest["tail_expected"]) != {
        "state_dict_sha256",
        "schema_sha256",
        "purpose",
    }:
        raise ValueError("smoke input tail fields changed")
    resolved = {}
    for key, relative in manifest["paths"].items():
        resolved[key] = _artifact(root, relative, f"smoke input {key}")
        if sha256(resolved[key]) != _validate_sha256(
            manifest["sha256"][key], f"smoke input {key} SHA-256"
        ):
            raise ValueError(f"smoke input {key} artifact changed")
    formal_receipt = _read_json(
        resolved["formal_receipt"], "nested formal portable receipt"
    )
    nested_paths = formal_receipt.get("paths")
    if not isinstance(nested_paths, dict) or not nested_paths:
        raise ValueError("nested formal portable receipt has no artifact paths")
    for key, relative in nested_paths.items():
        _strict_relative_path(relative, f"nested formal receipt {key}")
    for value in manifest["formal_loader_expected"].values():
        _validate_sha256(value, "formal loader expected SHA-256")
    for key in ("state_dict_sha256", "schema_sha256"):
        _validate_sha256(manifest["tail_expected"].get(key), f"tail expected {key}")
    if manifest["tail_expected"].get("purpose") != TAIL_CHECKPOINT_PURPOSE:
        raise ValueError("smoke input tail purpose changed")
    return manifest, resolved, manifest_path


def load_smoke_input(
    portable_root: Path,
    manifest_relative_path: str,
    *,
    expected_manifest_sha256: str,
    device: torch.device | str,
    synthetic_hostname_override: str | None = None,
) -> LoadedSmokeInput:
    manifest, resolved, manifest_path = _validate_smoke_manifest_static(
        portable_root,
        manifest_relative_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if synthetic_hostname_override is not None and manifest["mode"] != SYNTHETIC_MODE:
        raise ValueError("hostname override is forbidden for formal actual smoke")
    fingerprint = runtime_fingerprint(
        device, synthetic_hostname_override=synthetic_hostname_override
    )
    runtime_contract = validate_runtime_contract(
        resolved["runtime_contract"],
        expected_sha256=manifest["sha256"]["runtime_contract"],
        fingerprint=fingerprint,
        mode=manifest["mode"],
    )
    formal = load_formal_r2o_bundle(
        Path(portable_root),
        manifest["paths"]["formal_receipt"],
        expected_receipt_sha256=manifest["sha256"]["formal_receipt"],
        expected_gate_sha256=manifest["formal_loader_expected"]["gate_sha256"],
        expected_bundle_sha256=manifest["formal_loader_expected"]["bundle_sha256"],
        device=device,
    )
    if smoke_source_hashes()["taylor_wrapper"] != formal.binding.wrapper_sha256:
        raise ValueError("executing Taylor wrapper differs from formal receipt")
    tail, schema = _materialize_tail_checkpoint(
        resolved["tail_checkpoint"],
        formal=formal,
        expected_artifact_sha256=manifest["sha256"]["tail_checkpoint"],
        expected_state_sha256=manifest["tail_expected"]["state_dict_sha256"],
        expected_schema_sha256=manifest["tail_expected"]["schema_sha256"],
        device=device,
    )
    return LoadedSmokeInput(
        portable_root=Path(portable_root).resolve(strict=True),
        manifest_path=manifest_path,
        manifest_sha256=expected_manifest_sha256,
        manifest=manifest,
        formal=formal,
        tail=tail,
        schema=schema,
        runtime_contract=runtime_contract,
        runtime_fingerprint=fingerprint,
    )


def validate_formal_smoke_go_marker(
    portable_root: Path,
    marker_relative_path: str,
    *,
    expected_marker_sha256: str,
    input_manifest_sha256: str,
    node_label: str,
) -> dict:
    marker_path = _artifact(
        Path(portable_root).resolve(strict=True),
        marker_relative_path,
        "formal smoke independent-GO marker",
    )
    if sha256(marker_path) != _validate_sha256(
        expected_marker_sha256, "formal smoke GO marker SHA-256"
    ):
        raise ValueError("formal smoke GO marker artifact SHA-256 mismatch")
    payload = _read_json(marker_path, "formal smoke GO marker")
    expected_keys = {
        "format",
        "status",
        "scope",
        "input_manifest_sha256",
        "authorized_node_labels",
        "independent_review_receipt_sha256",
        "source_sha256",
        "protocol_sha256",
        "real_formal_smoke_authorized",
        "outer_prepare_authorized",
        "authorization_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("formal smoke GO marker fields changed")
    recorded = payload.pop("authorization_sha256")
    if canonical_json_sha256(payload) != recorded:
        raise ValueError("formal smoke GO marker canonical SHA-256 mismatch")
    payload["authorization_sha256"] = recorded
    if payload["format"] != FORMAL_GO_FORMAT or payload["status"] != FORMAL_GO_STATUS:
        raise ValueError("formal smoke has no independent GO authorization")
    if payload["scope"] != "real_formal_no_support_actual_tail_smoke_only":
        raise ValueError("formal smoke GO marker scope changed")
    if payload["input_manifest_sha256"] != input_manifest_sha256:
        raise ValueError("formal smoke GO marker belongs to another input manifest")
    labels = payload["authorized_node_labels"]
    if (
        not isinstance(labels, list)
        or len(labels) != 2
        or len(set(labels)) != 2
        or not all(
            isinstance(label, str)
            and label
            and label.replace("-", "").replace("_", "").isalnum()
            for label in labels
        )
    ):
        raise ValueError("formal smoke GO marker must authorize two distinct labels")
    if node_label not in labels:
        raise ValueError("formal smoke GO marker does not authorize this node label")
    _validate_sha256(
        payload["independent_review_receipt_sha256"],
        "independent review receipt SHA-256",
    )
    if payload["source_sha256"] != smoke_source_hashes():
        raise ValueError("formal smoke GO marker source binding changed")
    if payload["protocol_sha256"] != smoke_protocol()["protocol_sha256"]:
        raise ValueError("formal smoke GO marker protocol binding changed")
    if payload["real_formal_smoke_authorized"] is not True:
        raise ValueError("formal smoke GO marker does not authorize execution")
    if payload["outer_prepare_authorized"] is not False:
        raise ValueError("formal smoke GO marker improperly authorizes outer prepare")
    return payload


def _sum_taylor_results(core: TaylorResult, tail: TaylorResult) -> TaylorResult:
    torch.testing.assert_close(
        core.current_positions_reference_order,
        tail.current_positions_reference_order,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(core.displacement, tail.displacement, atol=0.0, rtol=0.0)
    return TaylorResult(
        energy=core.energy + tail.energy,
        forces_reference_order=(
            core.forces_reference_order + tail.forces_reference_order
        ),
        raw_energy=core.raw_energy + tail.raw_energy,
        reference_energy=core.reference_energy + tail.reference_energy,
        linear_term=core.linear_term + tail.linear_term,
        quadratic_term=core.quadratic_term + tail.quadratic_term,
        displacement=core.displacement,
        current_positions_reference_order=core.current_positions_reference_order,
    )


def _evaluate_composite(
    loaded: LoadedSmokeInput,
    structure: Atoms,
    *,
    create_graph: bool,
) -> tuple[TaylorResult, TaylorResult, TaylorResult, Any]:
    query = build_order_mic_query(loaded.formal, structure)
    tail = tail_taylor_remainder(
        loaded.formal,
        loaded.tail,
        loaded.schema,
        query,
        create_graph=create_graph,
    )
    core = whole_energy_taylor2_null(
        lambda positions: mace_interaction_energy(
            loaded.formal.model, query.data, positions
        ),
        query.current_positions_reference_order,
        query.reference_positions,
        query.image_shift,
        create_graph=create_graph,
    )
    return _sum_taylor_results(core, tail), core, tail, query


def _source_forces(
    loaded: LoadedSmokeInput, result: TaylorResult, query: Any
) -> torch.Tensor:
    return source_order_tail_forces(loaded.formal, result, query)


def _hessian_from_force(force: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = [
        -torch.autograd.grad(value, positions, retain_graph=True)[0].reshape(-1)
        for value in force.reshape(-1)
    ]
    return torch.stack(rows)


def _proper_and_improper() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(83)
    proper, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(proper) < 0.0:
        proper[:, 0] *= -1.0
    improper = proper.copy()
    improper[:, 0] *= -1.0
    return proper, improper


def _probe(reference: Atoms) -> Atoms:
    result = reference.copy()
    result.positions[4] += np.asarray([0.012, -0.007, 0.018])
    return result


def _float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def evaluate_actual_tail_smoke(loaded: LoadedSmokeInput) -> dict:
    """Run the complete nonzero core+tail mechanics protocol."""
    validate_loaded_formal_r2o(loaded.formal)
    validate_r2o_invariant_schema(loaded.schema)
    protocol = smoke_protocol()
    reference = loaded.formal.reference_for_count(72)
    reference_result, _, _, reference_query = _evaluate_composite(
        loaded, reference, create_graph=True
    )
    hessian = _hessian_from_force(
        reference_result.forces_reference_order,
        reference_query.current_positions_reference_order,
    )
    hessian_np = hessian.detach().cpu().numpy()
    hessian_sym = 0.5 * (hessian_np + hessian_np.T)
    asr = hessian_np.reshape(72, 3, 72, 3).sum(axis=2)
    mass_weighted = hessian_sym / 12.011
    eigen_bound = float(np.max(np.abs(np.linalg.eigvalsh(mass_weighted))))
    frequency_bound = CM1_PER_SQRT_EV_A2_AMU * math.sqrt(max(0.0, eigen_bound))
    reference_null = {
        "energy_abs_eV": abs(_float(reference_result.energy)),
        "force_max_abs_eV_A": float(
            reference_result.forces_reference_order.detach().abs().max().cpu()
        ),
        "Hessian_max_abs_eV_A2": float(np.max(np.abs(hessian_np))),
        "Hessian_antisymmetry_max_abs_eV_A2": float(
            np.max(np.abs(hessian_np - hessian_np.T))
        ),
        "Hessian_ASR_max_abs_eV_A2": float(np.max(np.abs(asr))),
        "Gamma_K_frequency_drift_upper_bound_cm-1": frequency_bound,
        "Hessian_semantic_sha256": canonical_json_sha256(
            {
                "dtype": str(hessian_np.dtype),
                "shape": list(hessian_np.shape),
                "bytes_sha256": hashlib.sha256(
                    np.ascontiguousarray(hessian_np).tobytes()
                ).hexdigest(),
            }
        ),
    }
    del hessian, reference_result, reference_query
    gc.collect()
    if next(loaded.formal.model.parameters()).device.type == "cuda":
        torch.cuda.empty_cache()

    probe = _probe(reference)
    probe_result, probe_core, probe_tail, probe_query = _evaluate_composite(
        loaded, probe, create_graph=True
    )
    probe_force = _source_forces(loaded, probe_result, probe_query)
    probe_tail_force = _source_forces(loaded, probe_tail, probe_query)
    probe_hessian = _hessian_from_force(
        probe_result.forces_reference_order,
        probe_query.current_positions_reference_order,
    )
    probe_hessian_np = probe_hessian.detach().cpu().numpy()
    probe_asr = probe_hessian_np.reshape(72, 3, 72, 3).sum(axis=2)
    nonreference_hessian = {
        "shape": list(probe_hessian_np.shape),
        "antisymmetry_max_abs_eV_A2": float(
            np.max(np.abs(probe_hessian_np - probe_hessian_np.T))
        ),
        "translation_ASR_max_abs_eV_A2": float(np.max(np.abs(probe_asr))),
        "Hessian_semantic_sha256": canonical_json_sha256(
            {
                "dtype": str(probe_hessian_np.dtype),
                "shape": list(probe_hessian_np.shape),
                "bytes_sha256": hashlib.sha256(
                    np.ascontiguousarray(probe_hessian_np).tobytes()
                ).hexdigest(),
            }
        ),
    }
    linearity = audit_core_tail_taylor_linearity(
        loaded.formal, loaded.tail, loaded.schema, probe_query
    )

    coordinate = (4, 2)
    step = 1.0e-4
    displaced_energy: list[float] = []
    displaced_force: list[float] = []
    for sign in (-1.0, 1.0):
        structure = probe.copy()
        structure.positions[coordinate] += sign * step
        result, _, _, query = _evaluate_composite(
            loaded, structure, create_graph=False
        )
        source_force = _source_forces(loaded, result, query)
        displaced_energy.append(_float(result.energy))
        displaced_force.append(_float(source_force[coordinate]))
    fd_force = -(displaced_energy[1] - displaced_energy[0]) / (2.0 * step)
    reference_atom = int(probe_query.assignment.source_to_reference[coordinate[0]])
    analytic_force_jacobian = torch.autograd.grad(
        probe_force[coordinate],
        probe_query.current_positions_reference_order,
        retain_graph=True,
    )[0][reference_atom, coordinate[1]]
    fd_force_jacobian = (displaced_force[1] - displaced_force[0]) / (2.0 * step)
    finite_difference = {
        "coordinate": list(coordinate),
        "step_A": step,
        "analytic_force_eV_A": _float(probe_force[coordinate]),
        "finite_difference_force_eV_A": fd_force,
        "energy_force_absolute_difference_eV_A": abs(
            fd_force - _float(probe_force[coordinate])
        ),
        "analytic_dF_dx_eV_A2": _float(analytic_force_jacobian),
        "finite_difference_dF_dx_eV_A2": fd_force_jacobian,
        "force_Hessian_absolute_difference_eV_A2": abs(
            fd_force_jacobian - _float(analytic_force_jacobian)
        ),
    }

    original_energy = _float(probe_result.energy)
    original_force = probe_force.detach().cpu().numpy()
    o3 = {}
    for name, matrix in zip(("proper", "improper"), _proper_and_improper(), strict=True):
        transformed = probe.copy()
        transformed.positions = np.asarray(probe.positions) @ matrix.T
        transformed.set_cell(np.asarray(probe.cell) @ matrix.T, scale_atoms=False)
        result, _, _, query = _evaluate_composite(
            loaded, transformed, create_graph=False
        )
        force = _source_forces(loaded, result, query).detach().cpu().numpy()
        o3[name] = {
            "determinant": float(np.linalg.det(matrix)),
            "energy_abs_difference_eV": abs(_float(result.energy) - original_energy),
            "force_max_abs_difference_eV_A": float(
                np.max(np.abs(force - original_force @ matrix.T))
            ),
        }

    translation = np.asarray(protocol["global_translation_A"], float)
    translated = probe.copy()
    translated.positions = np.asarray(probe.positions) + translation
    translated_result, _, _, translated_query = _evaluate_composite(
        loaded, translated, create_graph=False
    )
    translated_force = _source_forces(
        loaded, translated_result, translated_query
    ).detach().cpu().numpy()
    translation_metrics = {
        "translation_A": translation.tolist(),
        "energy_abs_difference_eV": abs(
            _float(translated_result.energy) - original_energy
        ),
        "force_max_abs_difference_eV_A": float(
            np.max(np.abs(translated_force - original_force))
        ),
    }

    permutation = np.random.default_rng(83).permutation(72)
    transformed = probe[permutation]
    transformed.set_cell(probe.cell, scale_atoms=False)
    transformed.pbc = probe.pbc
    transformed.positions[0] += np.asarray(transformed.cell[0])
    order_result, _, _, order_query = _evaluate_composite(
        loaded, transformed, create_graph=False
    )
    order_force = _source_forces(
        loaded, order_result, order_query
    ).detach().cpu().numpy()
    order_mic = {
        "has_nonzero_MIC_integer": bool(
            np.any(order_query.assignment.image_integer_reference_order != 0)
        ),
        "energy_abs_difference_eV": abs(_float(order_result.energy) - original_energy),
        "force_max_abs_difference_eV_A": float(
            np.max(np.abs(order_force - original_force[permutation]))
        ),
    }

    localized = {}
    localized_queries = {}
    for count in (72, 128):
        template = loaded.formal.reference_for_count(count)
        structure = template.copy()
        structure.positions[0, 2] += 0.03
        result, _, _, query = _evaluate_composite(
            loaded, structure, create_graph=False
        )
        force = _source_forces(loaded, result, query).detach().cpu().numpy()
        localized[count] = {
            "energy_eV": _float(result.energy),
            "central_force_eV_A": force[0].tolist(),
            "frozen_graph_source_quantization_max_A": (
                query.frozen_graph_source_quantization_max_A
            ),
        }
        localized_queries[count] = query
    size = {
        "localized_displacement_A": 0.03,
        "energy_abs_difference_eV": abs(
            localized[72]["energy_eV"] - localized[128]["energy_eV"]
        ),
        "central_force_max_abs_difference_eV_A": float(
            np.max(
                np.abs(
                    np.asarray(localized[72]["central_force_eV_A"])
                    - np.asarray(localized[128]["central_force_eV_A"])
                )
            )
        ),
        "per_size": localized,
    }
    paired = {
        "6x6_72_atom": audit_paired_native_fp64_sensitivity(
            loaded.formal,
            probe_query,
            tail=loaded.tail,
            schema=loaded.schema,
        ),
        "8x8_128_atom": audit_paired_native_fp64_sensitivity(
            loaded.formal,
            localized_queries[128],
            tail=loaded.tail,
            schema=loaded.schema,
        ),
    }
    graph_semantics = frozen_graph_semantics()
    graph_source_geometry = {
        "bound_A": graph_semantics["formal_graph_source_geometry_bound_A"],
        "6x6_observed_max_A": localized[72][
            "frozen_graph_source_quantization_max_A"
        ],
        "8x8_observed_max_A": localized[128][
            "frozen_graph_source_quantization_max_A"
        ],
    }
    metrics = {
        "format": "graphene_r2o_actual_tail_mechanics_metrics_v1",
        "protocol": protocol,
        "bindings": {
            "formal_receipt_sha256": loaded.formal.receipt_sha256,
            "formal_model_state_sha256": loaded.formal.binding.model_state_sha256,
            "tail_state_dict_sha256": state_dict_sha256(loaded.tail),
            "schema_sha256": loaded.schema["schema_sha256"],
            "graph_semantics_sha256": graph_semantics["graph_semantics_sha256"],
            "runtime_semantic_environment_sha256": loaded.runtime_fingerprint[
                "semantic_environment_sha256"
            ],
            "runtime_host_identity_sha256": loaded.runtime_fingerprint[
                "execution_environment"
            ]["host_identity_sha256"],
            "source_sha256": smoke_source_hashes(),
        },
        "raw_MACE_direct_deployment": False,
        "actual_tail": {
            "checkpoint_purpose": TAIL_CHECKPOINT_PURPOSE,
            "tail_energy_abs_eV_at_probe": abs(_float(probe_tail.energy)),
            "tail_force_max_abs_eV_A_at_probe": float(
                probe_tail_force.detach().abs().max().cpu()
            ),
            "core_energy_eV_at_probe": _float(probe_core.energy),
            "composite_energy_eV_at_probe": original_energy,
            "composite_forces_source_order_eV_A": original_force.tolist(),
        },
        "reference_null": reference_null,
        "nonreference_complete_Hessian": nonreference_hessian,
        "combined_separate_linearity": {
            "energy_max_abs_difference_eV": (
                linearity.energy_max_abs_difference_eV
            ),
            "force_max_abs_difference_eV_A": (
                linearity.force_max_abs_difference_eV_A
            ),
        },
        "finite_difference": finite_difference,
        "O3": o3,
        "global_translation": translation_metrics,
        "order_MIC": order_mic,
        "size_6x6_8x8": size,
        "paired_native_FP64_sensitivity": paired,
        "graph_source_geometry": graph_source_geometry,
        "graph_semantics": graph_semantics,
    }
    thresholds = SMOKE_THRESHOLDS
    tail_nonzero = max(
        metrics["actual_tail"]["tail_energy_abs_eV_at_probe"],
        metrics["actual_tail"]["tail_force_max_abs_eV_A_at_probe"],
    ) > thresholds["nonzero_tail_energy_or_force"]
    gates = {
        "actual_nonzero_tail": tail_nonzero,
        "reference_energy": reference_null["energy_abs_eV"]
        <= thresholds["reference_energy_abs_eV"],
        "reference_force": reference_null["force_max_abs_eV_A"]
        <= thresholds["reference_force_max_abs_eV_A"],
        "reference_Hessian": reference_null["Hessian_max_abs_eV_A2"]
        <= thresholds["reference_Hessian_max_abs_eV_A2"],
        "reference_Hessian_symmetry": reference_null[
            "Hessian_antisymmetry_max_abs_eV_A2"
        ]
        <= thresholds["reference_Hessian_antisymmetry_eV_A2"],
        "reference_Hessian_ASR": reference_null["Hessian_ASR_max_abs_eV_A2"]
        <= thresholds["reference_Hessian_ASR_eV_A2"],
        "reference_frequency_drift": reference_null[
            "Gamma_K_frequency_drift_upper_bound_cm-1"
        ]
        <= thresholds["reference_frequency_drift_bound_cm-1"],
        "combined_separate_energy": linearity.energy_max_abs_difference_eV
        <= thresholds["combined_separate_energy_eV"],
        "combined_separate_force": linearity.force_max_abs_difference_eV_A
        <= thresholds["combined_separate_force_eV_A"],
        "energy_force_FD": finite_difference[
            "energy_force_absolute_difference_eV_A"
        ]
        <= thresholds["energy_force_FD_eV_A"],
        "force_Hessian_FD": finite_difference[
            "force_Hessian_absolute_difference_eV_A2"
        ]
        <= thresholds["force_Hessian_FD_eV_A2"],
        "nonreference_Hessian_antisymmetry": nonreference_hessian[
            "antisymmetry_max_abs_eV_A2"
        ]
        <= thresholds["nonreference_Hessian_antisymmetry_eV_A2"],
        "nonreference_Hessian_translation_ASR": nonreference_hessian[
            "translation_ASR_max_abs_eV_A2"
        ]
        <= thresholds["nonreference_Hessian_translation_ASR_eV_A2"],
        "O3_proper": o3["proper"]["energy_abs_difference_eV"]
        <= thresholds["O3_energy_eV"]
        and o3["proper"]["force_max_abs_difference_eV_A"]
        <= thresholds["O3_force_eV_A"],
        "O3_improper": o3["improper"]["energy_abs_difference_eV"]
        <= thresholds["O3_energy_eV"]
        and o3["improper"]["force_max_abs_difference_eV_A"]
        <= thresholds["O3_force_eV_A"],
        "translation": translation_metrics["energy_abs_difference_eV"]
        <= thresholds["translation_energy_eV"]
        and translation_metrics["force_max_abs_difference_eV_A"]
        <= thresholds["translation_force_eV_A"],
        "order_MIC": order_mic["has_nonzero_MIC_integer"]
        and order_mic["energy_abs_difference_eV"]
        <= thresholds["order_MIC_energy_eV"]
        and order_mic["force_max_abs_difference_eV_A"]
        <= thresholds["order_MIC_force_eV_A"],
        "size_6x6_8x8": size["energy_abs_difference_eV"]
        <= thresholds["size_energy_eV"]
        and size["central_force_max_abs_difference_eV_A"]
        <= thresholds["size_force_eV_A"],
        "paired_native_FP64_sensitivity_6x6": paired["6x6_72_atom"][
            "passes_paired_native_FP64_sensitivity"
        ]
        is True,
        "paired_native_FP64_sensitivity_8x8": paired["8x8_128_atom"][
            "passes_paired_native_FP64_sensitivity"
        ]
        is True,
        "graph_source_geometry_6x6": graph_source_geometry[
            "6x6_observed_max_A"
        ]
        <= thresholds["graph_source_geometry_A"],
        "graph_source_geometry_8x8": graph_source_geometry[
            "8x8_observed_max_A"
        ]
        <= thresholds["graph_source_geometry_A"],
        "graph_semantics": graph_semantics["graph_semantics_sha256"]
        == loaded.manifest["graph_semantics_sha256"],
        "raw_MACE_not_directly_deployed": True,
    }
    metrics["gates"] = gates
    metrics["passes_all_gates"] = all(gates.values())
    return metrics


def _validate_clean_output_root(path: Path) -> Path:
    path = Path(path)
    lowered = path.as_posix().casefold()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError("smoke output path names unopened or reserved data")
    if path.exists() and path.is_symlink():
        raise ValueError("smoke output root may not be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    if any(item.is_symlink() for item in path.iterdir()):
        raise ValueError("smoke output root contains a symlink")
    return path.resolve(strict=True)


def _publish_failure(run_root: Path, error: BaseException) -> None:
    failure = {
        "format": "graphene_r2o_actual_tail_node_smoke_failure_v1",
        "error_type": type(error).__name__,
        "message": str(error)[:500],
    }
    strict_json(run_root / "FAILED", failure)
    _atomic_bytes(run_root / "EXIT_CODE", FAILURE_EXIT_BYTES)
    running = run_root / "RUNNING"
    if running.exists():
        running.unlink()


def run_node_smoke(
    portable_root: Path,
    manifest_relative_path: str,
    *,
    expected_manifest_sha256: str,
    node_label: str,
    output_root: Path,
    device: torch.device | str,
    formal_go_marker_relative_path: str | None = None,
    expected_formal_go_marker_sha256: str | None = None,
    synthetic_hostname_override: str | None = None,
    evaluator: Callable[[LoadedSmokeInput], dict] = evaluate_actual_tail_smoke,
) -> dict:
    if not isinstance(node_label, str) or not node_label or not node_label.replace(
        "-", ""
    ).replace("_", "").isalnum():
        raise ValueError("node label must be a simple non-empty identifier")
    run_root = _validate_clean_output_root(output_root)
    result_path = run_root / "result.json"
    if (run_root / "DONE").exists():
        observed_hash = sha256(result_path)
        recovered = validate_node_result(run_root, expected_result_sha256=observed_hash)
        if recovered["node_label"] != node_label:
            raise ValueError("completed smoke belongs to another node label")
        if recovered["input_manifest_sha256"] != expected_manifest_sha256:
            raise ValueError("completed smoke belongs to another input manifest")
        return {"result": recovered, "result_sha256": observed_hash, "recovered": True}
    occupied = [
        name
        for name in ("RUNNING", "FAILED", "EXIT_CODE", "result.json")
        if (run_root / name).exists()
    ]
    if occupied:
        raise ValueError(
            "incomplete smoke output refuses in-place recovery; use a new output root: "
            + ",".join(occupied)
        )
    manifest, _, _ = _validate_smoke_manifest_static(
        portable_root,
        manifest_relative_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    formal_go_sha256 = None
    if manifest["mode"] == FORMAL_ACTUAL_MODE:
        if (
            formal_go_marker_relative_path is None
            or expected_formal_go_marker_sha256 is None
        ):
            raise ValueError(
                "formal actual-tail smoke requires an independent GO marker"
            )
        validate_formal_smoke_go_marker(
            portable_root,
            formal_go_marker_relative_path,
            expected_marker_sha256=expected_formal_go_marker_sha256,
            input_manifest_sha256=expected_manifest_sha256,
            node_label=node_label,
        )
        formal_go_sha256 = expected_formal_go_marker_sha256
    elif (
        formal_go_marker_relative_path is not None
        or expected_formal_go_marker_sha256 is not None
    ):
        raise ValueError("synthetic smoke must not consume a formal GO marker")
    loaded = load_smoke_input(
        portable_root,
        manifest_relative_path,
        expected_manifest_sha256=expected_manifest_sha256,
        device=device,
        synthetic_hostname_override=synthetic_hostname_override,
    )
    invocation = {
        "format": "graphene_r2o_actual_tail_node_smoke_invocation_v1",
        "node_label": node_label,
        "input_manifest_sha256": expected_manifest_sha256,
        "mode": loaded.manifest["mode"],
        "device": str(torch.device(device)),
    }
    running_path = run_root / "RUNNING"
    try:
        descriptor = os.open(running_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(invocation, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
        metrics = evaluator(loaded)
        if metrics.get("passes_all_gates") is not True:
            failed = [name for name, value in metrics.get("gates", {}).items() if not value]
            raise ValueError("actual-tail smoke gates failed: " + ",".join(failed))
        metrics_path = run_root / "metrics.json"
        runtime_path = run_root / "runtime_fingerprint.json"
        input_snapshot = run_root / "input_manifest.snapshot.json"
        strict_json(metrics_path, metrics)
        strict_json(runtime_path, loaded.runtime_fingerprint)
        _atomic_copy(loaded.manifest_path, input_snapshot)
        paths = {
            "metrics": "metrics.json",
            "runtime_fingerprint": "runtime_fingerprint.json",
            "input_manifest_snapshot": "input_manifest.snapshot.json",
            "DONE": "DONE",
            "EXIT_CODE": "EXIT_CODE",
        }
        hashes = {
            "metrics": sha256(metrics_path),
            "runtime_fingerprint": sha256(runtime_path),
            "input_manifest_snapshot": sha256(input_snapshot),
            "DONE": _bytes_sha256(DONE_BYTES),
            "EXIT_CODE": _bytes_sha256(SUCCESS_EXIT_BYTES),
        }
        result = {
            "format": NODE_RESULT_FORMAT,
            "status": NODE_RESULT_STATUS,
            "node_label": node_label,
            "mode": loaded.manifest["mode"],
            "input_manifest_sha256": loaded.manifest_sha256,
            "formal_receipt_sha256": loaded.formal.receipt_sha256,
            "formal_model_state_sha256": loaded.formal.binding.model_state_sha256,
            "tail_state_dict_sha256": state_dict_sha256(loaded.tail),
            "schema_sha256": loaded.schema["schema_sha256"],
            "graph_semantics_sha256": loaded.manifest["graph_semantics_sha256"],
            "runtime_semantic_environment_sha256": loaded.runtime_fingerprint[
                "semantic_environment_sha256"
            ],
            "runtime_execution_environment_sha256": loaded.runtime_fingerprint[
                "execution_environment_sha256"
            ],
            "runtime_hostname": loaded.runtime_fingerprint["execution_environment"][
                "host_identity"
            ]["hostname"],
            "runtime_host_identity_sha256": loaded.runtime_fingerprint[
                "execution_environment"
            ]["host_identity_sha256"],
            "formal_go_marker_sha256": formal_go_sha256,
            "protocol_sha256": smoke_protocol()["protocol_sha256"],
            "source_sha256": smoke_source_hashes(),
            "paths": paths,
            "sha256": hashes,
            "passes_all_gates": True,
            "recovery_policy": "validate_completed_or_refuse_incomplete",
            "forbidden": loaded.manifest["forbidden"],
        }
        strict_json(result_path, result)
        result_sha256 = sha256(result_path)
        _atomic_bytes(run_root / "EXIT_CODE", SUCCESS_EXIT_BYTES)
        running_path.unlink()
        _atomic_bytes(run_root / "DONE", DONE_BYTES)
        return {
            "result": result,
            "result_sha256": result_sha256,
            "recovered": False,
        }
    except BaseException as error:
        _publish_failure(run_root, error)
        raise


def validate_node_result(
    run_root: Path, *, expected_result_sha256: str
) -> dict:
    root = Path(run_root).resolve(strict=True)
    for forbidden in ("RUNNING", "FAILED"):
        if (root / forbidden).exists():
            raise ValueError(f"node smoke has forbidden marker {forbidden}")
    result_path = resolve_relative_artifact(root, "result.json", "node result")
    if sha256(result_path) != _validate_sha256(
        expected_result_sha256, "node result SHA-256"
    ):
        raise ValueError("node smoke result SHA-256 mismatch")
    result = _read_json(result_path, "node smoke result")
    expected_keys = {
        "format",
        "status",
        "node_label",
        "mode",
        "input_manifest_sha256",
        "formal_receipt_sha256",
        "formal_model_state_sha256",
        "tail_state_dict_sha256",
        "schema_sha256",
        "graph_semantics_sha256",
        "runtime_semantic_environment_sha256",
        "runtime_execution_environment_sha256",
        "runtime_hostname",
        "runtime_host_identity_sha256",
        "formal_go_marker_sha256",
        "protocol_sha256",
        "source_sha256",
        "paths",
        "sha256",
        "passes_all_gates",
        "recovery_policy",
        "forbidden",
    }
    if set(result) != expected_keys:
        raise ValueError("node smoke result fields changed")
    if result["format"] != NODE_RESULT_FORMAT or result["status"] != NODE_RESULT_STATUS:
        raise ValueError("node smoke result is not passing")
    if result["mode"] not in ALLOWED_MODES or result["passes_all_gates"] is not True:
        raise ValueError("node smoke result did not pass")
    if result["mode"] == FORMAL_ACTUAL_MODE:
        _validate_sha256(
            result["formal_go_marker_sha256"], "formal GO marker SHA-256"
        )
    elif result["formal_go_marker_sha256"] is not None:
        raise ValueError("synthetic node result improperly binds a formal GO marker")
    if result["forbidden"] != SMOKE_FORBIDDEN_STATE:
        raise ValueError("node smoke result permits forbidden state")
    if result["source_sha256"] != smoke_source_hashes():
        raise ValueError("node smoke source code changed after run")
    if result["protocol_sha256"] != smoke_protocol()["protocol_sha256"]:
        raise ValueError("node smoke protocol changed after run")
    if set(result["paths"]) != {
        "metrics",
        "runtime_fingerprint",
        "input_manifest_snapshot",
        "DONE",
        "EXIT_CODE",
    } or set(result["sha256"]) != set(result["paths"]):
        raise ValueError("node smoke artifact receipt fields changed")
    resolved = {}
    for key, relative in result["paths"].items():
        _validate_sha256(result["sha256"].get(key), f"node result {key} SHA-256")
        resolved[key] = resolve_relative_artifact(root, relative, f"node result {key}")
        if sha256(resolved[key]) != result["sha256"][key]:
            raise ValueError(f"node smoke artifact changed: {key}")
    if resolved["DONE"].read_bytes() != DONE_BYTES:
        raise ValueError("node smoke DONE marker content changed")
    if resolved["EXIT_CODE"].read_bytes() != SUCCESS_EXIT_BYTES:
        raise ValueError("node smoke EXIT_CODE is not zero")
    metrics = _read_json(resolved["metrics"], "node metrics")
    runtime = _read_json(resolved["runtime_fingerprint"], "node runtime fingerprint")
    snapshot = _read_json(resolved["input_manifest_snapshot"], "input snapshot")
    if sha256(resolved["input_manifest_snapshot"]) != result[
        "input_manifest_sha256"
    ]:
        raise ValueError("node input snapshot differs from bound manifest")
    if snapshot.get("forbidden") != result["forbidden"]:
        raise ValueError("node result forbidden-data declaration changed")
    bindings = metrics.get("bindings", {})
    for result_key, metric_key in (
        ("formal_receipt_sha256", "formal_receipt_sha256"),
        ("formal_model_state_sha256", "formal_model_state_sha256"),
        ("tail_state_dict_sha256", "tail_state_dict_sha256"),
        ("schema_sha256", "schema_sha256"),
        ("graph_semantics_sha256", "graph_semantics_sha256"),
        ("runtime_semantic_environment_sha256", "runtime_semantic_environment_sha256"),
        ("runtime_host_identity_sha256", "runtime_host_identity_sha256"),
    ):
        if result[result_key] != bindings.get(metric_key):
            raise ValueError(f"node metrics binding differs: {metric_key}")
    if metrics.get("passes_all_gates") is not True or not all(
        metrics.get("gates", {}).values()
    ):
        raise ValueError("node metrics do not pass every gate")
    if runtime.get("semantic_environment_sha256") != result[
        "runtime_semantic_environment_sha256"
    ]:
        raise ValueError("node runtime semantic hash differs from result")
    if runtime.get("execution_environment_sha256") != result[
        "runtime_execution_environment_sha256"
    ]:
        raise ValueError("node runtime execution hash differs from result")
    host_identity = runtime.get("execution_environment", {}).get("host_identity", {})
    if host_identity.get("hostname") != result["runtime_hostname"]:
        raise ValueError("node runtime hostname differs from result")
    if runtime.get("execution_environment", {}).get(
        "host_identity_sha256"
    ) != result["runtime_host_identity_sha256"]:
        raise ValueError("node runtime host identity differs from result")
    return result


def _publish_dual_failure(root: Path, error: BaseException) -> None:
    strict_json(
        root / "FAILED",
        {
            "format": "graphene_r2o_actual_tail_dual_node_failure_v1",
            "error_type": type(error).__name__,
            "message": str(error)[:500],
        },
    )
    _atomic_bytes(root / "EXIT_CODE", FAILURE_EXIT_BYTES)
    if (root / "RUNNING").exists():
        (root / "RUNNING").unlink()


def evaluate_dual_node_results(
    node_runs: Sequence[tuple[Path, str]],
    *,
    output_root: Path,
) -> dict:
    if len(node_runs) != 2:
        raise ValueError("dual-node smoke requires exactly two node results")
    root = _validate_clean_output_root(output_root)
    result_path = root / "dual_result.json"
    if (root / "DONE").exists():
        observed = sha256(result_path)
        recovered = validate_dual_result(root, expected_result_sha256=observed)
        return {"result": recovered, "result_sha256": observed, "recovered": True}
    occupied = [name for name in ("RUNNING", "FAILED", "EXIT_CODE") if (root / name).exists()]
    if occupied:
        raise ValueError("incomplete dual result refuses in-place recovery")
    running = root / "RUNNING"
    try:
        descriptor = os.open(running, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        nodes = [
            validate_node_result(path, expected_result_sha256=expected)
            for path, expected in node_runs
        ]
        if nodes[0]["node_label"] == nodes[1]["node_label"]:
            raise ValueError("dual-node result labels must be distinct")
        if nodes[0]["runtime_hostname"] == nodes[1]["runtime_hostname"]:
            raise ValueError("dual-node smoke requires distinct runtime hostnames")
        if nodes[0]["runtime_host_identity_sha256"] == nodes[1][
            "runtime_host_identity_sha256"
        ]:
            raise ValueError("dual-node smoke requires distinct host fingerprints")
        invariant_keys = (
            "mode",
            "input_manifest_sha256",
            "formal_receipt_sha256",
            "formal_model_state_sha256",
            "tail_state_dict_sha256",
            "schema_sha256",
            "graph_semantics_sha256",
            "runtime_semantic_environment_sha256",
            "formal_go_marker_sha256",
            "protocol_sha256",
            "source_sha256",
            "forbidden",
        )
        for key in invariant_keys:
            if nodes[0][key] != nodes[1][key]:
                raise ValueError(f"dual-node smoke binding differs: {key}")
        metrics = []
        for path, _ in node_runs:
            metrics.append(_read_json(Path(path) / "metrics.json", "node metrics"))
        energy_difference = abs(
            metrics[0]["actual_tail"]["composite_energy_eV_at_probe"]
            - metrics[1]["actual_tail"]["composite_energy_eV_at_probe"]
        )
        force_difference = float(
            np.max(
                np.abs(
                    np.asarray(
                        metrics[0]["actual_tail"][
                            "composite_forces_source_order_eV_A"
                        ],
                        float,
                    )
                    - np.asarray(
                        metrics[1]["actual_tail"][
                            "composite_forces_source_order_eV_A"
                        ],
                        float,
                    )
                )
            )
        )
        cross_gates = {
            "probe_energy": energy_difference
            <= SMOKE_THRESHOLDS["cross_node_probe_energy_eV"],
            "probe_force": force_difference
            <= SMOKE_THRESHOLDS["cross_node_probe_force_eV_A"],
            "both_nodes_pass": all(node["passes_all_gates"] for node in nodes),
            "same_runtime_semantics": nodes[0][
                "runtime_semantic_environment_sha256"
            ]
            == nodes[1]["runtime_semantic_environment_sha256"],
            "distinct_runtime_hostnames": nodes[0]["runtime_hostname"]
            != nodes[1]["runtime_hostname"],
            "distinct_host_fingerprints": nodes[0][
                "runtime_host_identity_sha256"
            ]
            != nodes[1]["runtime_host_identity_sha256"],
            "same_frozen_bindings": True,
        }
        if not all(cross_gates.values()):
            raise ValueError("dual-node cross-result gates failed")
        aggregate = {
            "format": "graphene_r2o_actual_tail_dual_metrics_v1",
            "node_labels": [node["node_label"] for node in nodes],
            "node_hostnames": [node["runtime_hostname"] for node in nodes],
            "node_host_identity_sha256": [
                node["runtime_host_identity_sha256"] for node in nodes
            ],
            "node_result_sha256": [expected for _, expected in node_runs],
            "cross_node_probe_energy_abs_difference_eV": energy_difference,
            "cross_node_probe_force_max_abs_difference_eV_A": force_difference,
            "cross_node_gates": cross_gates,
            "passes_all_gates": True,
        }
        aggregate_path = root / "aggregate.json"
        strict_json(aggregate_path, aggregate)
        first_snapshot = root / "node_0_result.snapshot.json"
        second_snapshot = root / "node_1_result.snapshot.json"
        _atomic_copy(Path(node_runs[0][0]) / "result.json", first_snapshot)
        _atomic_copy(Path(node_runs[1][0]) / "result.json", second_snapshot)
        result = {
            "format": DUAL_RESULT_FORMAT,
            "status": DUAL_RESULT_STATUS,
            "mode": nodes[0]["mode"],
            "node_labels": aggregate["node_labels"],
            "node_hostnames": aggregate["node_hostnames"],
            "node_host_identity_sha256": aggregate[
                "node_host_identity_sha256"
            ],
            "node_result_sha256": aggregate["node_result_sha256"],
            "input_manifest_sha256": nodes[0]["input_manifest_sha256"],
            "formal_receipt_sha256": nodes[0]["formal_receipt_sha256"],
            "formal_model_state_sha256": nodes[0]["formal_model_state_sha256"],
            "tail_state_dict_sha256": nodes[0]["tail_state_dict_sha256"],
            "schema_sha256": nodes[0]["schema_sha256"],
            "graph_semantics_sha256": nodes[0]["graph_semantics_sha256"],
            "runtime_semantic_environment_sha256": nodes[0][
                "runtime_semantic_environment_sha256"
            ],
            "formal_go_marker_sha256": nodes[0]["formal_go_marker_sha256"],
            "protocol_sha256": nodes[0]["protocol_sha256"],
            "source_sha256": nodes[0]["source_sha256"],
            "paths": {
                "aggregate": "aggregate.json",
                "node_0_result_snapshot": "node_0_result.snapshot.json",
                "node_1_result_snapshot": "node_1_result.snapshot.json",
                "DONE": "DONE",
                "EXIT_CODE": "EXIT_CODE",
            },
            "sha256": {
                "aggregate": sha256(aggregate_path),
                "node_0_result_snapshot": sha256(first_snapshot),
                "node_1_result_snapshot": sha256(second_snapshot),
                "DONE": _bytes_sha256(DUAL_DONE_BYTES),
                "EXIT_CODE": _bytes_sha256(SUCCESS_EXIT_BYTES),
            },
            "passes_all_gates": True,
            "remote_launch_or_outer_prepare_authorized": False,
        }
        strict_json(result_path, result)
        result_sha256 = sha256(result_path)
        _atomic_bytes(root / "EXIT_CODE", SUCCESS_EXIT_BYTES)
        running.unlink()
        _atomic_bytes(root / "DONE", DUAL_DONE_BYTES)
        return {"result": result, "result_sha256": result_sha256, "recovered": False}
    except BaseException as error:
        _publish_dual_failure(root, error)
        raise


def validate_dual_result(root: Path, *, expected_result_sha256: str) -> dict:
    root = Path(root).resolve(strict=True)
    for marker in ("RUNNING", "FAILED"):
        if (root / marker).exists():
            raise ValueError(f"dual smoke has forbidden marker {marker}")
    result_path = resolve_relative_artifact(root, "dual_result.json", "dual result")
    if sha256(result_path) != _validate_sha256(
        expected_result_sha256, "dual result SHA-256"
    ):
        raise ValueError("dual smoke result SHA-256 mismatch")
    result = _read_json(result_path, "dual result")
    expected_keys = {
        "format",
        "status",
        "mode",
        "node_labels",
        "node_hostnames",
        "node_host_identity_sha256",
        "node_result_sha256",
        "input_manifest_sha256",
        "formal_receipt_sha256",
        "formal_model_state_sha256",
        "tail_state_dict_sha256",
        "schema_sha256",
        "graph_semantics_sha256",
        "runtime_semantic_environment_sha256",
        "formal_go_marker_sha256",
        "protocol_sha256",
        "source_sha256",
        "paths",
        "sha256",
        "passes_all_gates",
        "remote_launch_or_outer_prepare_authorized",
    }
    if set(result) != expected_keys:
        raise ValueError("dual smoke result fields changed")
    if result.get("format") != DUAL_RESULT_FORMAT or result.get("status") != DUAL_RESULT_STATUS:
        raise ValueError("dual smoke result is not passing")
    if result.get("passes_all_gates") is not True:
        raise ValueError("dual smoke result gates did not pass")
    if result.get("remote_launch_or_outer_prepare_authorized") is not False:
        raise ValueError("dual smoke result improperly authorizes a next phase")
    if (
        not isinstance(result["node_labels"], list)
        or len(result["node_labels"]) != 2
        or len(set(result["node_labels"])) != 2
        or not isinstance(result["node_hostnames"], list)
        or len(result["node_hostnames"]) != 2
        or len(set(result["node_hostnames"])) != 2
        or not isinstance(result["node_host_identity_sha256"], list)
        or len(result["node_host_identity_sha256"]) != 2
        or len(set(result["node_host_identity_sha256"])) != 2
    ):
        raise ValueError("dual smoke does not prove two distinct hosts")
    for value in result["node_host_identity_sha256"]:
        _validate_sha256(value, "dual node host-identity SHA-256")
    if result["mode"] == FORMAL_ACTUAL_MODE:
        _validate_sha256(
            result["formal_go_marker_sha256"], "dual formal GO marker SHA-256"
        )
    elif result["mode"] == SYNTHETIC_MODE:
        if result["formal_go_marker_sha256"] is not None:
            raise ValueError("synthetic dual result binds a formal GO marker")
    else:
        raise ValueError("dual smoke mode changed")
    if result.get("source_sha256") != smoke_source_hashes():
        raise ValueError("dual smoke source code changed")
    if set(result.get("paths", {})) != {
        "aggregate",
        "node_0_result_snapshot",
        "node_1_result_snapshot",
        "DONE",
        "EXIT_CODE",
    }:
        raise ValueError("dual smoke artifact paths changed")
    if set(result.get("sha256", {})) != set(result["paths"]):
        raise ValueError("dual smoke artifact hashes changed")
    resolved = {}
    for key, relative in result["paths"].items():
        _validate_sha256(result["sha256"].get(key), f"dual {key} SHA-256")
        resolved[key] = resolve_relative_artifact(root, relative, f"dual {key}")
        if sha256(resolved[key]) != result["sha256"][key]:
            raise ValueError(f"dual smoke artifact changed: {key}")
    if resolved["DONE"].read_bytes() != DUAL_DONE_BYTES:
        raise ValueError("dual smoke DONE marker changed")
    if resolved["EXIT_CODE"].read_bytes() != SUCCESS_EXIT_BYTES:
        raise ValueError("dual smoke EXIT_CODE changed")
    aggregate = _read_json(resolved["aggregate"], "dual aggregate")
    if aggregate.get("passes_all_gates") is not True or not all(
        aggregate.get("cross_node_gates", {}).values()
    ):
        raise ValueError("dual aggregate gates did not pass")
    if aggregate.get("node_result_sha256") != result["node_result_sha256"]:
        raise ValueError("dual aggregate node-result chain changed")
    for index, key in enumerate(
        ("node_0_result_snapshot", "node_1_result_snapshot")
    ):
        snapshot = _read_json(resolved[key], f"dual node {index} snapshot")
        if sha256(resolved[key]) != result["node_result_sha256"][index]:
            raise ValueError("dual node result snapshot hash differs from receipt")
        if snapshot.get("node_label") != result["node_labels"][index]:
            raise ValueError("dual node result snapshot label differs")
        if snapshot.get("runtime_hostname") != result["node_hostnames"][index]:
            raise ValueError("dual node result snapshot hostname differs")
        if snapshot.get("runtime_host_identity_sha256") != result[
            "node_host_identity_sha256"
        ][index]:
            raise ValueError("dual node result snapshot host fingerprint differs")
        if snapshot.get("input_manifest_sha256") != result[
            "input_manifest_sha256"
        ]:
            raise ValueError("dual node result snapshot input binding differs")
    return result


def _tailscale_target(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise ValueError("launch-plan target must be a numeric Tailscale address") from error
    network = ipaddress.ip_network("100.64.0.0/10")
    if address.version != 4 or address not in network:
        raise ValueError("launch-plan target is not in Tailscale CGNAT space")
    return value


def _safe_remote_absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value:
        raise ValueError(f"{label} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if value != path.as_posix() or any(part in ("", ".", "..") for part in path.parts[1:]):
        raise ValueError(f"{label} is not canonical")
    lowered = value.casefold()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"{label} names unopened or reserved data")
    return value


def build_dual_node_launch_plan(
    portable_root: Path,
    manifest_relative_path: str,
    *,
    expected_manifest_sha256: str,
    nodes: Sequence[Mapping[str, str]],
    output_path: Path,
) -> dict:
    manifest, _, _ = _validate_smoke_manifest_static(
        portable_root,
        manifest_relative_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if len(nodes) != 2:
        raise ValueError("launch plan requires exactly two nodes")
    records = []
    for node in nodes:
        if set(node) != {
            "label",
            "target",
            "transport",
            "remote_portable_root",
            "remote_output_root",
            "device",
        }:
            raise ValueError("launch-plan node fields changed")
        label = node["label"]
        if not label.replace("-", "").replace("_", "").isalnum():
            raise ValueError("launch-plan node label is not safe")
        target = _tailscale_target(node["target"])
        if node["transport"] not in {"tailscale_ssh", "openssh_over_tailscale_tun"}:
            raise ValueError("launch-plan transport must stay on Tailscale")
        remote_root = _safe_remote_absolute_path(
            node["remote_portable_root"], "remote portable root"
        )
        remote_output = _safe_remote_absolute_path(
            node["remote_output_root"], "remote output root"
        )
        if node["device"] != "cuda:0":
            raise ValueError("formal dual-node plan freezes device=cuda:0")
        runner = f"{remote_root}/scripts/smearing_kink/run_graphene_r2o_tail_smoke.py"
        remote_argv = [
            "conda",
            "run",
            "-n",
            "phonon",
            "python",
            runner,
            "--portable-root",
            remote_root,
            "--manifest",
            manifest_relative_path,
            "--expected-manifest-sha256",
            expected_manifest_sha256,
            "--node-label",
            label,
            "--output-root",
            remote_output,
            "--device",
            "cuda:0",
        ]
        prefix = [
            "tailscale" if node["transport"] == "tailscale_ssh" else "ssh",
            "ssh" if node["transport"] == "tailscale_ssh" else target,
        ]
        if node["transport"] == "tailscale_ssh":
            prefix.append(target)
        records.append(
            {
                "label": label,
                "target": target,
                "transport": node["transport"],
                "argv": prefix + ["--"] + remote_argv,
                "remote_execution_authorized": False,
            }
        )
    if records[0]["label"] == records[1]["label"] or records[0]["target"] == records[1]["target"]:
        raise ValueError("launch-plan nodes must have distinct labels and targets")
    payload = {
        "format": LAUNCH_PLAN_FORMAT,
        "status": LAUNCH_PLAN_STATUS,
        "input_mode": manifest["mode"],
        "input_manifest_sha256": expected_manifest_sha256,
        "protocol_sha256": manifest["protocol_sha256"],
        "source_sha256": smoke_source_hashes(),
        "transport_policy": "Tailscale_only",
        "nodes": records,
        "remote_execution_authorized": False,
        "outer_prepare_authorized": False,
    }
    payload["plan_sha256"] = canonical_json_sha256(payload)
    strict_json(output_path, payload)
    return payload


def validate_dual_node_launch_plan(
    path: Path, *, expected_plan_artifact_sha256: str
) -> dict:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("launch plan must be a regular non-symlink file")
    path = unresolved.resolve(strict=True)
    if not path.is_file():
        raise ValueError("launch plan must be a regular non-symlink file")
    if sha256(path) != _validate_sha256(
        expected_plan_artifact_sha256, "launch-plan artifact SHA-256"
    ):
        raise ValueError("launch-plan artifact SHA-256 mismatch")
    payload = _read_json(path, "launch plan")
    expected_keys = {
        "format",
        "status",
        "input_mode",
        "input_manifest_sha256",
        "protocol_sha256",
        "source_sha256",
        "transport_policy",
        "nodes",
        "remote_execution_authorized",
        "outer_prepare_authorized",
        "plan_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("launch-plan fields changed")
    recorded = payload.pop("plan_sha256")
    if canonical_json_sha256(payload) != recorded:
        raise ValueError("launch-plan canonical SHA-256 mismatch")
    payload["plan_sha256"] = recorded
    if payload["format"] != LAUNCH_PLAN_FORMAT or payload["status"] != LAUNCH_PLAN_STATUS:
        raise ValueError("launch plan is not waiting for independent review")
    if payload["remote_execution_authorized"] is not False:
        raise ValueError("launch plan improperly authorizes remote execution")
    if payload["outer_prepare_authorized"] is not False:
        raise ValueError("launch plan improperly authorizes outer preparation")
    if payload["transport_policy"] != "Tailscale_only":
        raise ValueError("launch plan lost its Tailscale-only policy")
    if payload["source_sha256"] != smoke_source_hashes():
        raise ValueError("launch-plan source code changed")
    if len(payload["nodes"]) != 2:
        raise ValueError("launch plan does not contain exactly two nodes")
    for node in payload["nodes"]:
        if set(node) != {
            "label",
            "target",
            "transport",
            "argv",
            "remote_execution_authorized",
        }:
            raise ValueError("launch-plan node receipt fields changed")
        _tailscale_target(node["target"])
        if node["transport"] not in {
            "tailscale_ssh",
            "openssh_over_tailscale_tun",
        }:
            raise ValueError("launch-plan node transport changed")
        if node["remote_execution_authorized"] is not False:
            raise ValueError("launch-plan node improperly authorizes execution")
        if not isinstance(node["argv"], list) or not all(
            isinstance(value, str) and value for value in node["argv"]
        ):
            raise ValueError("launch-plan argv is not an inert string list")
    return payload


def remote_launch(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise PermissionError(
        "remote launch is intentionally disabled pending independent review and a real formal bundle decision"
    )
