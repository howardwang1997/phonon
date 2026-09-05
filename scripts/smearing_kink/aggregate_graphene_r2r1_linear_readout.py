#!/usr/bin/env python3
"""Atomic local materialization and recovery for the R2R-1 fit release."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import hashlib
import io
import json
import os
import secrets
import stat
import struct
import tempfile
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import graphene_r2r1_frozen_readout as frozen
import graphene_r2r1_linear_readout as linear


FORMAT = "graphene_r2r1_local_fit_release_v2"
FIT_RECEIPT_FORMAT = "graphene_r2r1_frozen_fit_receipt_v2"
COMPLETION_FORMAT = "graphene_r2r1_local_completion_v2"
RELEASE_MANIFEST_FORMAT = "graphene_r2r1_external_release_manifest_v2"
TERMINAL_LEDGER_FORMAT = "graphene_r2r1_terminal_identity_ledger_v2"
FIT_MANIFEST_BASENAME = "fit_manifest.json"
FIT_ARRAYS_BASENAME = "fit_arrays.npz"
FIT_RECEIPT_BASENAME = "fit_receipt.json"
COMPLETION_BASENAME = "completion.json"
CHECKPOINT_DIRNAME = "checkpoint"
TERMINAL_SUCCESS = {"DONE"}
SCIENTIFIC_STATUSES = {
    "R2R1_CONDITIONAL_OOF_FAILED",
    "R2R1_FINAL_READOUT_FAILED",
    "R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING",
}
CHECKPOINT_STATUS_SOURCE = (
    "R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING"
)
OOF_FAILED_STATUS = "R2R1_CONDITIONAL_OOF_FAILED"
FINAL_FAILED_STATUS = "R2R1_FINAL_READOUT_FAILED"
EXPECTED_ARRAY_SHAPES_BY_STATUS = {
    OOF_FAILED_STATUS: {
        "OOF_predicted_force_eV_A": (92, 72, 3),
    },
    FINAL_FAILED_STATUS: {
        "OOF_predicted_force_eV_A": (92, 72, 3),
        "final_predicted_force_eV_A": (92, 72, 3),
        "final_coefficients": (65,),
        "final_scale_eV_A": (65,),
    },
    CHECKPOINT_STATUS_SOURCE: {
        "OOF_predicted_force_eV_A": (92, 72, 3),
        "final_predicted_force_eV_A": (92, 72, 3),
        "final_coefficients": (65,),
        "final_scale_eV_A": (65,),
    },
}
PIPELINE_KEYS_BY_STATUS = {
    OOF_FAILED_STATUS: {
        "format",
        "status",
        "numerically_inconclusive",
        "conditional_OOF_pass",
        "final_fit_performed",
        "development_or_held_access",
        "split_design_audits",
        "nested_OOF",
        "execution_authorization",
        "attempt3_completed_recovery",
        "label_parser_receipt",
        "thermal_label_raw_sha256",
        "energy_labels_used",
    },
    FINAL_FAILED_STATUS: {
        "format",
        "status",
        "numerically_inconclusive",
        "conditional_OOF_pass",
        "final_fit_performed",
        "final_train_gate_pass",
        "mechanics_pending",
        "development_or_held_access",
        "energy_labels_used",
        "encoder_updated",
        "split_design_audits",
        "nested_OOF",
        "final_alpha_selection",
        "final_fit",
        "final_train_metrics",
        "final_coefficients",
        "final_coefficients_sha256",
        "final_scale_raw_sha256",
        "execution_authorization",
        "attempt3_completed_recovery",
        "label_parser_receipt",
        "thermal_label_raw_sha256",
    },
}
PIPELINE_KEYS_BY_STATUS[CHECKPOINT_STATUS_SOURCE] = set(
    PIPELINE_KEYS_BY_STATUS[FINAL_FAILED_STATUS]
)
FIT_RECEIPT_KEYS = {
    "format",
    "status",
    "fit_manifest",
    "fit_arrays",
    "pipeline_receipt",
    "pipeline_recursive_schema_sha256",
    "runtime_environment",
    "checkpoint_required",
    "checkpoint_pending_in_this_immutable_receipt",
    "mechanics_pending",
    "energy_labels_used",
    "development_or_held_access",
}
COMPLETION_KEYS = {
    "format",
    "status",
    "execution_exit_code",
    "scientific_gate_failure_is_terminal_success",
    "fit_manifest_sha256",
    "fit_arrays_sha256",
    "fit_receipt_sha256",
    "checkpoint",
    "checkpoint_published",
    "mechanics_completed",
    "mechanics_status",
    "development_or_held_access",
    "energy_labels_used",
    "release_root_binding_identity",
    "precompletion_artifact_snapshot",
    "precompletion_artifact_snapshot_sha256",
    "recursive_schema_sha256",
}
CHECKPOINT_SUMMARY_KEYS = {
    "receipt_sha256",
    "array_sha256",
    "physical_p_raw_sha256",
    "file_sha256",
}
RELEASE_MANIFEST_KEYS = {
    "format",
    "release_root",
    "status",
    "fit_manifest_sha256",
    "expected_completion_sha256",
    "expected_fit_receipt_sha256",
    "expected_checkpoint_receipt_sha256",
    "mechanics_completed",
    "development_or_held_access",
    "energy_labels_used",
    "expected_done_sha256",
    "expected_done_identity",
    "precompletion_artifact_snapshot_sha256",
}
TERMINAL_LEDGER_KEYS = {
    "format",
    "terminal",
    "status",
    "payload_basename",
    "payload_sha256",
    "payload_identity",
    "exit_sha256",
    "exit_identity",
    "root_binding_identity",
    "artifact_snapshot_sha256",
    "terminal_inode_stable_identity",
}
FULL_FILE_IDENTITY_KEYS = {
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
}
OWNED_REGULAR_IDENTITY_KEYS = FULL_FILE_IDENTITY_KEYS - {"st_ctime_ns"}
DIRECTORY_BINDING_IDENTITY_KEYS = {
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
}
FILE_STABLE_IDENTITY_KEYS = {
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
}
SNAPSHOT_FILE_KEYS = {"kind", "identity", "sha256", "size"}
SNAPSHOT_DIRECTORY_KEYS = {"kind", "identity"}
FAILURE_KEYS = {
    "format",
    "error_type",
    "error_message",
    "scientific_gate_result",
    "numerically_inconclusive",
    "fit_release_published",
    "checkpoint_published",
    "development_or_held_access",
    "energy_labels_used",
    "authorization_verified_before_transaction",
}
ROOT_FILE_SIZE_LIMITS = {
    FIT_MANIFEST_BASENAME: 4 * 1024 * 1024,
    FIT_ARRAYS_BASENAME: 64 * 1024 * 1024,
    FIT_RECEIPT_BASENAME: 32 * 1024 * 1024,
    COMPLETION_BASENAME: 2 * 1024 * 1024,
    "failure.json": 2 * 1024 * 1024,
    "EXIT_CODE": 64,
    "RUNNING": 4096,
    "DONE": 4096,
    "FAILED": 4096,
}
FREEZE_MANIFEST_BYTES_LIMIT = 8 * 1024 * 1024
NPY_HEADER_BYTES_LIMIT = 4096
NPZ_CENTRAL_DIRECTORY_BYTES_LIMIT = 64 * 1024


class _TerminalCommittedError(RuntimeError):
    """An error observed after RUNNING was atomically renamed to a terminal."""

    def __init__(self, terminal: str, cause: BaseException) -> None:
        super().__init__(f"R2R-1 {terminal} committed before post-commit error: {cause}")
        self.terminal = terminal
        self.cause = cause


class _AnchorCommittedError(RuntimeError):
    """The external anchor became visible before a later local fault."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"R2R-1 external anchor committed before error: {cause}")
        self.cause = cause


class MaterializationResult(dict[str, Any]):
    """JSON-compatible receipt with a non-serialized first-anchor witness."""

    def __init__(
        self, payload: Mapping[str, Any], witness: object
    ) -> None:
        super().__init__(payload)
        self.anchor_witness = witness


def _build_materialization_witness_boundary(implementation):
    """Seal first-anchor authority inside the materializer's lexical closure."""
    seal = object()
    issued: dict[object, bytes] = {}
    witness_keys = {
        "release_root",
        "status",
        "done_sha256",
        "done_identity",
        "expected_release_snapshot",
    }

    class MaterializationWitness:
        __slots__ = ()

        def __new__(cls, token: object):
            if token is not seal:
                raise PermissionError(
                    "R2R-1 materialization witness cannot be constructed"
                )
            return super().__new__(cls)

        def __init__(self, token: object) -> None:
            del token

        def __copy__(self):
            raise PermissionError("R2R-1 materialization witness is non-copyable")

        def __deepcopy__(self, memo):
            del memo
            raise PermissionError("R2R-1 materialization witness is non-copyable")

        def __reduce_ex__(self, protocol):
            del protocol
            raise TypeError("R2R-1 materialization witness is non-serializable")

    def validate_payload(value: Any) -> dict[str, Any]:
        payload = _exact_keys(value, witness_keys, "R2R-1 materialization witness")
        if type(payload.get("release_root")) is not str or not payload["release_root"]:
            raise PermissionError("R2R-1 materialization witness root changed")
        if payload.get("status") not in SCIENTIFIC_STATUSES:
            raise PermissionError("R2R-1 materialization witness status changed")
        _expected_sha256(
            payload.get("done_sha256"), "materialization witness DONE SHA-256"
        )
        done_identity = _validate_owned_regular_identity(
            payload.get("done_identity"),
            "materialization witness DONE identity",
        )
        raw_snapshot = payload.get("expected_release_snapshot")
        expected_names = _snapshot_expected_names(
            payload["status"], include_completion=True
        ) | {"EXIT_CODE", "DONE"}
        if not isinstance(raw_snapshot, Mapping) or set(raw_snapshot) != expected_names:
            raise PermissionError("R2R-1 materialization witness inventory changed")
        snapshot = _validate_persisted_artifact_snapshot(
            {
                name: item
                for name, item in raw_snapshot.items()
                if name not in {"EXIT_CODE", "DONE"}
            },
            status=payload["status"],
            include_completion=True,
            label="R2R-1 materialization witness science snapshot",
        )
        for name in ("EXIT_CODE", "DONE"):
            item = _exact_keys(
                raw_snapshot[name],
                SNAPSHOT_FILE_KEYS,
                f"R2R-1 materialization witness {name}",
            )
            if item.get("kind") != "file":
                raise PermissionError(
                    f"R2R-1 materialization witness {name} kind changed"
                )
            identity = _validate_owned_regular_identity(
                item.get("identity"),
                f"R2R-1 materialization witness {name} identity",
            )
            size = item.get("size")
            if type(size) is not int or size < 0 or size != identity["st_size"]:
                raise PermissionError(
                    f"R2R-1 materialization witness {name} size changed"
                )
            snapshot[name] = {
                "kind": "file",
                "identity": identity,
                "sha256": _expected_sha256(
                    item.get("sha256"),
                    f"R2R-1 materialization witness {name} SHA-256",
                ),
                "size": size,
            }
        done_item = snapshot["DONE"]
        if (
            done_item["sha256"] != payload["done_sha256"]
            or not linear._json_type_exact_equal(
                done_item["identity"], done_identity
            )
        ):
            raise PermissionError("R2R-1 materialization witness DONE binding changed")
        return {
            **dict(payload),
            "done_identity": dict(done_identity),
            "expected_release_snapshot": snapshot,
        }

    def materialize(
        *,
        output_root: Path,
        attempt3_root: Path,
        thermal92_path: Path,
        freeze_manifest: Path,
        authorization_marker: Path,
    ) -> MaterializationResult:
        result_payload, witness_payload = implementation(
            output_root=output_root,
            attempt3_root=attempt3_root,
            thermal92_path=thermal92_path,
            freeze_manifest=freeze_manifest,
            authorization_marker=authorization_marker,
        )
        validated = validate_payload(witness_payload)
        witness = MaterializationWitness(seal)
        issued[witness] = linear.canonical_json_bytes(validated)
        return MaterializationResult(result_payload, witness)

    def require(value: object) -> dict[str, Any]:
        if type(value) is not MaterializationWitness or value not in issued:
            raise PermissionError(
                "R2R-1 first anchor requires its sealed in-process "
                "materialization witness"
            )
        payload_raw = issued[value]
        payload = _strict_canonical_json(
            payload_raw, "R2R-1 sealed materialization witness"
        )
        validated = validate_payload(payload)
        if linear.canonical_json_bytes(validated) != payload_raw:
            raise PermissionError("R2R-1 materialization witness changed")
        return validated

    return materialize, require


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _authoritative_owned_xattrs(
    descriptor: int, *, label: str
) -> dict[str, dict[str, Any]]:
    return linear._authoritative_owned_xattrs(descriptor, label=label)


def _expected_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase 64-hex SHA-256")
    return value


def _recursive_type_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": {
                str(key): _recursive_type_schema(value[key]) for key in sorted(value)
            },
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "items": [_recursive_type_schema(item) for item in value],
        }
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "boolean"}
    if type(value) is int:
        return {"type": "integer"}
    if type(value) is float:
        return {"type": "number"}
    if type(value) is str:
        return {"type": "string"}
    raise ValueError(f"unsupported receipt value type: {type(value).__name__}")


def _recursive_schema_sha256(value: Any) -> str:
    return linear.semantic_sha256(_recursive_type_schema(value))


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{label} key set changed: extra={sorted(observed - expected)}, "
            f"missing={sorted(expected - observed)}"
        )
    return value


def _strict_canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    payload = linear._strict_json_from_raw(raw, label)
    return payload


def _read(
    path: Path,
    label: str,
    *,
    expected_parent_chain: tuple[dict[str, int], ...] | None = None,
    expected_file_identity: Mapping[str, int] | None = None,
    size_limit: int,
) -> tuple[bytes, dict[str, int]]:
    return linear._read_regular_file_once(
        path,
        label,
        expected_parent_chain=expected_parent_chain,
        expected_file_identity=expected_file_identity,
        size_limit=size_limit,
    )


def _read_owned(
    path: Path,
    label: str,
    *,
    expected_parent_chain: tuple[dict[str, int], ...],
    expected_file_identity: Mapping[str, int],
    size_limit: int,
) -> tuple[bytes, dict[str, int]]:
    """Read one R2R-1-owned pathname without upgrading it to full9."""
    source = linear._lexical_path(
        path,
        label,
        forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if not source.is_absolute() or len(source.parts) < 2:
        raise ValueError(f"{label} must be an absolute file path")
    _parent, parent_fd, parent_chain = linear._open_directory_chain(
        source.parent,
        f"{label} parent",
        expected_chain=expected_parent_chain,
    )
    try:
        raw, identity = linear._read_owned_regular_file_at_fd(
            parent_fd,
            source.name,
            label,
            size_limit=size_limit,
            expected_identity=expected_file_identity,
        )
        linear._verify_directory_chain(
            source.parent,
            f"{label} final parent",
            parent_chain,
        )
        rebound_raw, rebound_identity = linear._read_owned_regular_file_at_fd(
            parent_fd,
            source.name,
            f"{label} after final parent rebind",
            size_limit=size_limit,
            expected_identity=identity,
        )
        if (
            rebound_raw != raw
            or not linear._json_type_exact_equal(rebound_identity, identity)
        ):
            raise ValueError(f"{label} bytes changed after final parent rebind")
        return raw, identity
    finally:
        os.close(parent_fd)


def _exact_path(
    path: Path,
    expected: Path,
    label: str,
    *,
    must_exist: bool,
) -> Path:
    observed = (
        linear._reject_path(path, label, must_exist=True)
        if must_exist
        else linear._lexical_path(
            path, label, forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS
        )
    )
    frozen = Path(os.path.abspath(str(expected)))
    if observed != frozen:
        raise PermissionError(f"{label} differs from the frozen production path")
    return observed


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_production_path_contract(
    *,
    output_root: Path,
    attempt3_root: Path,
    thermal92_path: Path,
    freeze_manifest: Path,
    authorization_marker: Path,
    release_manifest: Path | None = None,
    input_existence_required: bool = True,
) -> dict[str, str]:
    output = _exact_path(
        output_root,
        linear.RECOMMENDED_FIT_OUTPUT_ROOT,
        "R2R-1 fit output root",
        must_exist=False,
    )
    attempt3 = _exact_path(
        attempt3_root,
        linear.ATTEMPT3_ROOT,
        "R2R-1 attempt3 input root",
        must_exist=input_existence_required,
    )
    thermal = _exact_path(
        thermal92_path,
        linear.RECOMMENDED_THERMAL92,
        "R2R-1 thermal92 input",
        must_exist=input_existence_required,
    )
    manifest = _exact_path(
        freeze_manifest,
        linear.RECOMMENDED_FREEZE_MANIFEST,
        "R2R-1 freeze manifest",
        must_exist=input_existence_required,
    )
    marker = _exact_path(
        authorization_marker,
        linear.RECOMMENDED_AUTHORIZATION_MARKER,
        "R2R-1 authorization marker",
        must_exist=False,
    )
    writes = [output]
    if release_manifest is not None:
        release = _exact_path(
            release_manifest,
            linear.RECOMMENDED_RELEASE_MANIFEST,
            "R2R-1 external release manifest",
            must_exist=False,
        )
        writes.append(release)
        if _paths_overlap(output, release):
            raise PermissionError("R2R-1 release manifest must be outside fit root")
    protected = [
        attempt3,
        thermal,
        manifest,
        marker,
        *(
            Path(os.path.abspath(str(path)))
            for path in (
                linear.ATTEMPT1_FIT_OUTPUT_ROOT,
                linear.ATTEMPT1_CONTROL_ROOT,
                linear.ATTEMPT1_FREEZE_MANIFEST,
                linear.ATTEMPT1_AUTHORIZATION_MARKER,
                linear.ATTEMPT1_CANDIDATE_ROOT,
                linear.ATTEMPT1_RELEASE_MANIFEST,
                linear.FAILED_ATTEMPT2_FIT_OUTPUT_ROOT,
                linear.FAILED_ATTEMPT2_CONTROL_ROOT,
                linear.FAILED_ATTEMPT2_FREEZE_MANIFEST,
                linear.FAILED_ATTEMPT2_AUTHORIZATION_MARKER,
                linear.FAILED_ATTEMPT2_CANDIDATE_ROOT,
                linear.FAILED_ATTEMPT2_RELEASE_MANIFEST,
            )
        ),
        *(
            Path(os.path.abspath(str(path)))
            for path in (
                *linear.R2R1_SOURCE_PATHS.values(),
                *linear.R2R1_DEPENDENCY_PATHS.values(),
            )
        ),
    ]
    for write in writes:
        for protected_path in protected:
            if _paths_overlap(write, protected_path):
                raise PermissionError(
                    "R2R-1 write target overlaps a frozen input/source tree"
                )
    return {
        "output_root": str(output),
        "attempt3_root": str(attempt3),
        "thermal92_path": str(thermal),
        "freeze_manifest": str(manifest),
        "authorization_marker": str(marker),
        **(
            {"release_manifest": str(writes[-1])}
            if release_manifest is not None
            else {}
        ),
    }


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype != np.dtype("<f8") or not np.all(np.isfinite(value)):
            raise ValueError(f"R2R-1 fit array {name} must be finite little-endian FP64")
        schema[name] = {
            "shape": list(value.shape),
            "dtype": "float64",
            "raw_sha256": linear.raw_array_sha256(value, "<f8"),
        }
    return schema


def _validate_fit_array_schema_for_status(
    status: str, schema: Mapping[str, Any]
) -> None:
    expected_shapes = EXPECTED_ARRAY_SHAPES_BY_STATUS.get(status)
    if expected_shapes is None or set(schema) != set(expected_shapes):
        raise ValueError("R2R-1 fit-array member set differs from frozen status schema")
    for name, expected_shape in expected_shapes.items():
        item = _exact_keys(
            schema.get(name), {"shape", "dtype", "raw_sha256"}, f"array {name}"
        )
        observed_shape = item.get("shape")
        if (
            not isinstance(observed_shape, list)
            or len(observed_shape) != len(expected_shape)
            or any(type(value) is not int for value in observed_shape)
            or tuple(observed_shape) != expected_shape
            or type(item.get("dtype")) is not str
            or item.get("dtype") != "float64"
        ):
            raise ValueError(f"R2R-1 fit-array shape/dtype changed: {name}")
        _expected_sha256(item.get("raw_sha256"), f"fit array {name} raw SHA-256")


def _validate_pipeline_receipt_contract(
    status: str, pipeline: Mapping[str, Any]
) -> None:
    expected_keys = PIPELINE_KEYS_BY_STATUS.get(status)
    _exact_keys(pipeline, expected_keys or set(), "R2R-1 pipeline receipt")
    if (
        pipeline.get("format") != linear.AGGREGATE_FORMAT
        or pipeline.get("status") != status
        or pipeline.get("numerically_inconclusive") is not False
        or pipeline.get("development_or_held_access") is not False
        or pipeline.get("energy_labels_used") is not False
    ):
        raise ValueError("R2R-1 pipeline status/safety contract changed")
    oof_failed = status == OOF_FAILED_STATUS
    expected_conditional_pass = not oof_failed
    expected_final_fit = not oof_failed
    if (
        pipeline.get("conditional_OOF_pass") is not expected_conditional_pass
        or pipeline.get("final_fit_performed") is not expected_final_fit
    ):
        raise ValueError("R2R-1 pipeline OOF/final-fit cross-field changed")
    if not oof_failed:
        expected_final_pass = status == CHECKPOINT_STATUS_SOURCE
        if (
            pipeline.get("final_train_gate_pass") is not expected_final_pass
            or pipeline.get("mechanics_pending") is not expected_final_pass
            or pipeline.get("encoder_updated") is not False
        ):
            raise ValueError("R2R-1 pipeline final-gate cross-field changed")


def _prevalidate_npz_central_directory(
    raw: bytes, expected_members: set[str], label: str
) -> None:
    """Bound ZIP metadata allocation before ``zipfile.ZipFile`` sees it."""
    # EOCD is 22 bytes plus a uint16-sized comment.  Search only that bounded
    # suffix, require the selected record to close the input exactly, and
    # reject every multi-disk/ZIP64 sentinel before zipfile allocates ZipInfo
    # objects from attacker-controlled central-directory bytes.
    eocd_signature = b"PK\x05\x06"
    suffix_start = max(0, len(raw) - (22 + 0xFFFF))
    eocd_offset = raw.rfind(eocd_signature, suffix_start)
    if eocd_offset < 0 or eocd_offset + 22 > len(raw):
        raise ValueError(f"{label} ZIP EOCD is missing or truncated")
    try:
        (
            signature,
            disk_number,
            central_disk,
            entries_on_disk,
            entries_total,
            central_size,
            central_offset,
            comment_size,
        ) = struct.unpack_from("<4s4H2LH", raw, eocd_offset)
    except struct.error as error:
        raise ValueError(f"{label} ZIP EOCD is malformed") from error
    expected_count = len(expected_members)
    if (
        signature != eocd_signature
        or disk_number != 0
        or central_disk != 0
        or entries_on_disk != expected_count
        or entries_total != expected_count
        or entries_on_disk == 0xFFFF
        or entries_total == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
        or central_size > NPZ_CENTRAL_DIRECTORY_BYTES_LIMIT
        or eocd_offset + 22 + comment_size != len(raw)
        or central_offset + central_size != eocd_offset
    ):
        raise ValueError(f"{label} ZIP EOCD/resource contract changed")
    # A ZIP64 locator/record is forbidden even when a crafted classic EOCD
    # carries non-sentinel values.  The central-directory parser below also
    # rejects per-entry ZIP64 size/offset sentinels.
    eocd_prefix = raw[max(0, eocd_offset - 64) : eocd_offset]
    if b"PK\x06\x06" in eocd_prefix or b"PK\x06\x07" in eocd_prefix:
        raise ValueError(f"{label} ZIP64 metadata is forbidden")

    position = central_offset
    central_end = eocd_offset
    observed_members: set[str] = set()
    observed_count = 0
    while position < central_end:
        if position + 46 > central_end or raw[position : position + 4] != b"PK\x01\x02":
            raise ValueError(f"{label} central directory is malformed")
        try:
            compressed_size = struct.unpack_from("<L", raw, position + 20)[0]
            uncompressed_size = struct.unpack_from("<L", raw, position + 24)[0]
            filename_size, extra_size, member_comment_size = struct.unpack_from(
                "<HHH", raw, position + 28
            )
            member_disk = struct.unpack_from("<H", raw, position + 34)[0]
            local_header_offset = struct.unpack_from("<L", raw, position + 42)[0]
        except struct.error as error:
            raise ValueError(f"{label} central directory is truncated") from error
        record_end = position + 46 + filename_size + extra_size + member_comment_size
        if (
            record_end > central_end
            or member_disk != 0
            or compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            raise ValueError(f"{label} central entry resource contract changed")
        try:
            filename = raw[position + 46 : position + 46 + filename_size].decode(
                "utf-8", "strict"
            )
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} central filename is not UTF-8") from error
        if filename in observed_members:
            raise ValueError(f"{label} central member is duplicated")
        observed_members.add(filename)
        observed_count += 1
        if observed_count > expected_count:
            raise ValueError(f"{label} central member count exceeds its schema")
        position = record_end
    if (
        position != central_end
        or observed_count != expected_count
        or observed_members != expected_members
    ):
        raise ValueError(f"{label} ZIP member set changed")


def _load_npz_bytes(
    raw: bytes, expected_schema: Mapping[str, Any], label: str
) -> dict[str, np.ndarray]:
    if len(raw) > ROOT_FILE_SIZE_LIMITS[FIT_ARRAYS_BASENAME]:
        raise ValueError(f"{label} exceeds the frozen archive-size limit")
    expected_members = {f"{name}.npy" for name in expected_schema}
    _prevalidate_npz_central_directory(raw, expected_members, label)
    try:
        with zipfile.ZipFile(io.BytesIO(raw), mode="r") as archive:
            infos = archive.infolist()
            if (
                {info.filename for info in infos} != expected_members
                or len(infos) != len(expected_members)
            ):
                raise ValueError(f"{label} ZIP member set changed")
            total_uncompressed = 0
            for info in infos:
                member = info.filename[:-4]
                schema = expected_schema[member]
                shape = schema.get("shape")
                if (
                    not isinstance(shape, list)
                    or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                    or schema.get("dtype") != "float64"
                ):
                    raise ValueError(f"{label} expected member schema is invalid")
                expected_count = 1
                for dimension in shape:
                    expected_count *= dimension
                    if expected_count > ROOT_FILE_SIZE_LIMITS[FIT_ARRAYS_BASENAME] // 8:
                        raise ValueError(f"{label} expected member shape exceeds its limit")
                expected_nbytes = expected_count * 8
                if (
                    info.file_size > expected_nbytes + 4096
                    or info.compress_size > ROOT_FILE_SIZE_LIMITS[FIT_ARRAYS_BASENAME]
                ):
                    raise ValueError(f"{label} ZIP member exceeds its shape-bound limit")
                with archive.open(info, mode="r") as member_stream:
                    version = np.lib.format.read_magic(member_stream)
                    if version == (1, 0):
                        observed_shape, fortran_order, dtype = (
                            np.lib.format.read_array_header_1_0(
                                member_stream,
                                max_header_size=NPY_HEADER_BYTES_LIMIT,
                            )
                        )
                    elif version == (2, 0):
                        observed_shape, fortran_order, dtype = (
                            np.lib.format.read_array_header_2_0(
                                member_stream,
                                max_header_size=NPY_HEADER_BYTES_LIMIT,
                            )
                        )
                    else:
                        raise ValueError(f"{label} NPY member version changed")
                    header_bytes = member_stream.tell()
                if (
                    type(observed_shape) is not tuple
                    or any(type(dimension) is not int for dimension in observed_shape)
                    or observed_shape != tuple(shape)
                    or type(fortran_order) is not bool
                    or fortran_order is not False
                    or not isinstance(dtype, np.dtype)
                    or dtype.str != "<f8"
                    or header_bytes > NPY_HEADER_BYTES_LIMIT
                    or header_bytes + expected_nbytes != info.file_size
                ):
                    raise ValueError(f"{label} NPY header differs from its frozen schema")
                total_uncompressed += info.file_size
            if total_uncompressed > ROOT_FILE_SIZE_LIMITS[FIT_ARRAYS_BASENAME]:
                raise ValueError(f"{label} ZIP payload exceeds its total limit")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if sorted(archive.files) != sorted(expected_schema):
                raise ValueError(f"{label} member set changed")
            arrays = {
                name: np.asarray(archive[name]) for name in sorted(archive.files)
            }
    except (EOFError, OSError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and str(error).startswith(label):
            raise
        raise ValueError(f"{label} is not a valid non-pickle NPZ") from error
    if _array_schema(arrays) != dict(expected_schema):
        raise ValueError(f"{label} array schema/raw hashes changed")
    return arrays


def _validate_fit_receipt_invariants(receipt: Mapping[str, Any]) -> None:
    _exact_keys(receipt, FIT_RECEIPT_KEYS, "R2R-1 fit receipt")
    status = receipt.get("status")
    if receipt.get("format") != FIT_RECEIPT_FORMAT or status not in SCIENTIFIC_STATUSES:
        raise ValueError("R2R-1 fit receipt format/status changed")
    checkpoint_required = status == CHECKPOINT_STATUS_SOURCE
    for key in (
        "checkpoint_required",
        "checkpoint_pending_in_this_immutable_receipt",
        "mechanics_pending",
    ):
        if receipt.get(key) is not checkpoint_required:
            raise ValueError(f"R2R-1 fit receipt cross-field changed: {key}")
    for key in ("energy_labels_used", "development_or_held_access"):
        if receipt.get(key) is not False:
            raise ValueError(f"R2R-1 fit receipt safety field changed: {key}")
    manifest = _exact_keys(
        receipt.get("fit_manifest"), {"basename", "sha256"}, "fit manifest"
    )
    if manifest.get("basename") != FIT_MANIFEST_BASENAME:
        raise ValueError("R2R-1 fit manifest basename changed")
    _expected_sha256(manifest.get("sha256"), "fit manifest SHA-256")
    artifact = _exact_keys(
        receipt.get("fit_arrays"),
        {"basename", "sha256", "schema"},
        "fit arrays artifact",
    )
    if artifact.get("basename") != FIT_ARRAYS_BASENAME:
        raise ValueError("R2R-1 fit arrays basename changed")
    _expected_sha256(artifact.get("sha256"), "fit arrays SHA-256")
    _validate_fit_array_schema_for_status(status, artifact.get("schema"))
    pipeline = receipt.get("pipeline_receipt")
    if not isinstance(pipeline, Mapping) or pipeline.get("status") != status:
        raise ValueError("R2R-1 pipeline receipt status changed")
    _validate_pipeline_receipt_contract(status, pipeline)
    if (
        pipeline.get("energy_labels_used") is not False
        or pipeline.get("development_or_held_access") is not False
    ):
        raise ValueError("R2R-1 pipeline safety fields changed")
    if receipt.get("pipeline_recursive_schema_sha256") != _recursive_schema_sha256(
        pipeline
    ):
        raise ValueError("R2R-1 pipeline recursive key/type schema changed")


def _validate_completion_invariants(completion: Mapping[str, Any]) -> None:
    _exact_keys(completion, COMPLETION_KEYS, "R2R-1 completion")
    payload_without_fingerprint = {
        key: completion[key] for key in completion if key != "recursive_schema_sha256"
    }
    if completion.get("recursive_schema_sha256") != _recursive_schema_sha256(
        payload_without_fingerprint
    ):
        raise ValueError("R2R-1 completion recursive key/type schema changed")
    status = completion.get("status")
    if completion.get("format") != COMPLETION_FORMAT or status not in SCIENTIFIC_STATUSES:
        raise ValueError("R2R-1 completion format/status changed")
    if type(completion.get("execution_exit_code")) is not int or completion[
        "execution_exit_code"
    ] != 0:
        raise ValueError("R2R-1 completion execution exit changed")
    checkpoint_required = status == CHECKPOINT_STATUS_SOURCE
    scientific_failure = status in {
        "R2R1_CONDITIONAL_OOF_FAILED",
        "R2R1_FINAL_READOUT_FAILED",
    }
    if completion.get("scientific_gate_failure_is_terminal_success") is not scientific_failure:
        raise ValueError("R2R-1 scientific-failure terminal semantics changed")
    if completion.get("checkpoint_published") is not checkpoint_required:
        raise ValueError("R2R-1 checkpoint publication state changed")
    if completion.get("mechanics_completed") is not False:
        raise ValueError("R2R-1 fit stage cannot claim completed mechanics")
    expected_mechanics = "PENDING" if checkpoint_required else "NOT_RUN_STOPPED"
    if completion.get("mechanics_status") != expected_mechanics:
        raise ValueError("R2R-1 fit-stage mechanics status changed")
    for key in ("development_or_held_access", "energy_labels_used"):
        if completion.get(key) is not False:
            raise ValueError(f"R2R-1 completion safety field changed: {key}")
    root_binding = _exact_keys(
        completion.get("release_root_binding_identity"),
        DIRECTORY_BINDING_IDENTITY_KEYS,
        "completion release-root binding",
    )
    if (
        any(type(root_binding[key]) is not int for key in root_binding)
        or not stat.S_ISDIR(root_binding["st_mode"])
    ):
        raise ValueError("R2R-1 completion release-root binding changed")
    for key in (
        "fit_manifest_sha256",
        "fit_arrays_sha256",
        "fit_receipt_sha256",
    ):
        _expected_sha256(completion.get(key), f"completion {key}")
    checkpoint = completion.get("checkpoint")
    if checkpoint_required:
        summary = _exact_keys(
            checkpoint, CHECKPOINT_SUMMARY_KEYS, "checkpoint summary"
        )
        for key in ("receipt_sha256", "array_sha256", "physical_p_raw_sha256"):
            _expected_sha256(summary.get(key), f"checkpoint {key}")
        files = _exact_keys(
            summary.get("file_sha256"),
            set(frozen.CHECKPOINT_FILE_SET),
            "checkpoint file hashes",
        )
        for name, digest in files.items():
            _expected_sha256(digest, f"checkpoint file {name}")
    elif checkpoint is not None:
        raise ValueError("R2R-1 stopped fit stage unexpectedly contains checkpoint")
    artifact_snapshot = _validate_persisted_artifact_snapshot(
        completion.get("precompletion_artifact_snapshot"),
        status=status,
        include_completion=False,
        label="R2R-1 completion precompletion artifact snapshot",
    )
    snapshot_digest = _expected_sha256(
        completion.get("precompletion_artifact_snapshot_sha256"),
        "completion precompletion artifact snapshot SHA-256",
    )
    if snapshot_digest != _artifact_snapshot_sha256(artifact_snapshot):
        raise ValueError("R2R-1 completion artifact snapshot digest changed")
    if (
        artifact_snapshot[FIT_MANIFEST_BASENAME]["sha256"]
        != completion["fit_manifest_sha256"]
        or artifact_snapshot[FIT_ARRAYS_BASENAME]["sha256"]
        != completion["fit_arrays_sha256"]
        or artifact_snapshot[FIT_RECEIPT_BASENAME]["sha256"]
        != completion["fit_receipt_sha256"]
    ):
        raise ValueError("R2R-1 completion artifact snapshot/hash binding changed")
    if checkpoint_required:
        checkpoint_hashes = {
            name.split("/", 1)[1]: item["sha256"]
            for name, item in artifact_snapshot.items()
            if name.startswith(CHECKPOINT_DIRNAME + "/")
        }
        if not linear._json_type_exact_equal(
            checkpoint_hashes, checkpoint["file_sha256"]
        ):
            raise ValueError("R2R-1 completion checkpoint snapshot/hash binding changed")


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    label: str,
    *,
    size_limit: int,
    expected_identity: Mapping[str, int] | None = None,
) -> bytes:
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > size_limit
        or (
            expected_identity is not None
            and not linear._json_type_exact_equal(
                linear._owned_regular_identity(before), dict(expected_identity)
            )
        )
    ):
        raise ValueError(f"{label} is not one bounded single-link regular file")
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd
    )
    try:
        identity = linear._owned_regular_identity(os.fstat(descriptor))
        if identity != linear._owned_regular_identity(before):
            raise ValueError(f"{label} changed across stat/open")
        with linear._HeldRegularWriteGuard(f"{label} read boundary") as write_guard:
            write_guard.watch_owned_descriptor(
                descriptor,
                expected_identity=identity,
                label=label,
            )
            _authoritative_owned_xattrs(descriptor, label=label)
            chunks: list[bytes] = []
            total = 0
            while True:
                block = os.read(
                    descriptor, min(1024 * 1024, size_limit + 1 - total)
                )
                if not block:
                    break
                total += len(block)
                if total > size_limit:
                    raise ValueError(f"{label} exceeds its frozen size limit")
                chunks.append(block)
            if linear._owned_regular_identity(os.fstat(descriptor)) != identity:
                raise ValueError(f"{label} changed while read")
            _authoritative_owned_xattrs(descriptor, label=label)
            rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if linear._owned_regular_identity(rebound) != identity:
                raise ValueError(f"{label} directory entry changed while read")
            raw = b"".join(chunks)
        return raw
    finally:
        os.close(descriptor)


def _capture_held_regular_descriptor(
    descriptor: int,
    expected_raw: bytes,
    *,
    label: str,
    expected_identity: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    before = linear._owned_regular_identity(os.fstat(descriptor))
    if expected_identity is not None and not linear._json_type_exact_equal(
        before, dict(expected_identity)
    ):
        raise ValueError(f"{label} changed from its write/fsync identity")
    if (
        not stat.S_ISREG(before["st_mode"])
        or before["st_nlink"] != 1
        or before["st_size"] != len(expected_raw)
    ):
        raise ValueError(f"{label} held inode is not the expected regular file")
    with linear._HeldRegularWriteGuard(f"{label} held boundary") as write_guard:
        write_guard.watch_owned_descriptor(
            descriptor,
            expected_identity=before,
            label=label,
        )
        _authoritative_owned_xattrs(descriptor, label=label)
        if os.lseek(descriptor, 0, os.SEEK_SET) != 0:
            raise OSError(f"{label} held inode could not rewind")
        blocks: list[bytes] = []
        remaining = len(expected_raw) + 1
        while remaining > 0:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
        raw = b"".join(blocks)
        if raw != expected_raw:
            raise ValueError(f"{label} held bytes changed before publication")
        xattrs = _authoritative_owned_xattrs(descriptor, label=label)
        after = linear._owned_regular_identity(os.fstat(descriptor))
        if not linear._json_type_exact_equal(after, before):
            raise ValueError(f"{label} identity changed during held capture")
    return {
        "kind": "file",
        "identity": before,
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "xattrs": xattrs,
    }


def _capture_empty_owned_descriptor(
    descriptor: int,
    *,
    label: str,
    expected_identity: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Capture an empty private inode without treating OS metadata as authority."""
    before = linear._owned_regular_identity(os.fstat(descriptor))
    if expected_identity is not None and not linear._json_type_exact_equal(
        before, dict(expected_identity)
    ):
        raise ValueError(f"{label} changed from its create identity")
    if (
        not stat.S_ISREG(before["st_mode"])
        or before["st_nlink"] != 1
        or before["st_size"] != 0
    ):
        raise ValueError(f"{label} is not one empty private regular inode")
    xattrs = _authoritative_owned_xattrs(descriptor, label=label)
    after = linear._owned_regular_identity(os.fstat(descriptor))
    _authoritative_owned_xattrs(descriptor, label=label)
    if not linear._json_type_exact_equal(after, before):
        raise ValueError(f"{label} changed during empty-inode capture")
    return {"identity": before, "xattrs": xattrs}


def _verify_empty_to_payload_transition(
    empty_record: Mapping[str, Any],
    payload_record: Mapping[str, Any],
    *,
    payload_size: int,
    label: str,
) -> None:
    """Allow only this process's payload write after the empty owned record."""
    stable_keys = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
    )
    empty_identity = empty_record["identity"]
    payload_identity = payload_record["identity"]
    if (
        any(empty_identity[key] != payload_identity[key] for key in stable_keys)
        or payload_identity["st_size"] != payload_size
        or not linear._json_type_exact_equal(
            empty_record["xattrs"], payload_record["xattrs"]
        )
    ):
        raise ValueError(f"{label} changed outside the owned payload write")


def _atomic_write_bytes_at(directory_fd: int, name: str, payload: bytes) -> dict[str, int]:
    """Atomically create one file relative to a directory fd, without replace."""
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("R2R-1 atomic basename is invalid")
    linear._reject_forbidden_components(
        (name,), "R2R-1 atomic basename", linear.FORBIDDEN_LABEL_PATH_TOKENS
    )
    data = bytes(payload)
    temporary = f".{name}.{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    published = False
    completed = False
    try:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"R2R-1 output already exists: {name}")
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        created_identity = linear._owned_regular_identity(os.fstat(descriptor))
        initial_empty = _capture_empty_owned_descriptor(
            descriptor,
            label=f"R2R-1 initial empty atomic inode for {name}",
            expected_identity=created_identity,
        )
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while materializing R2R-1 artifact")
            view = view[written:]
        os.fsync(descriptor)
        held = os.fstat(descriptor)
        if not stat.S_ISREG(held.st_mode):
            raise ValueError("R2R-1 atomic temporary is not regular")
        initial_private = _capture_held_regular_descriptor(
            descriptor,
            data,
            label=f"R2R-1 initial private atomic inode for {name}",
            expected_identity=linear._owned_regular_identity(held),
        )
        _verify_empty_to_payload_transition(
            initial_empty,
            initial_private,
            payload_size=len(data),
            label=f"R2R-1 private atomic inode for {name}",
        )
        temporary_observed = os.stat(
            temporary, dir_fd=directory_fd, follow_symlinks=False
        )
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(temporary_observed),
            initial_private["identity"],
        ):
            raise ValueError("R2R-1 atomic temporary changed before publication")
        os.link(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary, dir_fd=directory_fd)
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_dev != held.st_dev
            or observed.st_ino != held.st_ino
            or observed.st_size != len(data)
        ):
            raise ValueError("R2R-1 atomic file identity changed")
        published_record = _capture_held_regular_descriptor(
            descriptor,
            data,
            label=f"R2R-1 published atomic inode for {name}",
        )
        if (
            not linear._json_type_exact_equal(
                linear._owned_regular_identity(observed),
                published_record["identity"],
            )
            or not linear._json_type_exact_equal(
                published_record["xattrs"], initial_private["xattrs"]
            )
        ):
            raise ValueError("R2R-1 atomic file changed across publication")
        os.fsync(directory_fd)
        final_record = _capture_held_regular_descriptor(
            descriptor,
            data,
            label=f"R2R-1 final published atomic inode for {name}",
        )
        if not linear._json_type_exact_equal(final_record, published_record):
            raise ValueError("R2R-1 atomic file changed after directory fsync")
        completed = True
        return dict(final_record["identity"])
    finally:
        try:
            if descriptor is not None:
                held = os.fstat(descriptor)
                if published and not completed:
                    try:
                        observed = os.stat(
                            name, dir_fd=directory_fd, follow_symlinks=False
                        )
                        if (
                            observed.st_dev == held.st_dev
                            and observed.st_ino == held.st_ino
                        ):
                            os.unlink(name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
                try:
                    temporary_observed = os.stat(
                        temporary, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if (
                        temporary_observed.st_dev == held.st_dev
                        and temporary_observed.st_ino == held.st_ino
                    ):
                        os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _atomic_write_json_at(directory_fd: int, name: str, payload: object) -> dict[str, int]:
    return _atomic_write_bytes_at(directory_fd, name, linear.canonical_json_bytes(payload))


def _atomic_write_npz_at(
    directory_fd: int, name: str, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.savez(buffer, **{key: np.asarray(value) for key, value in arrays.items()})
    return _atomic_write_bytes_at(directory_fd, name, buffer.getvalue())


def _rewrite_held_file(descriptor: int, payload: bytes) -> dict[str, int]:
    os.ftruncate(descriptor, 0)
    if os.lseek(descriptor, 0, os.SEEK_SET) != 0:
        raise OSError("R2R-1 held terminal could not rewind")
    view = memoryview(bytes(payload))
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while preparing R2R-1 terminal")
        view = view[written:]
    os.fsync(descriptor)
    return linear._owned_regular_identity(os.fstat(descriptor))


def _rename_noreplace_at(
    directory_fd: int,
    source_name: str,
    destination_name: str,
    *,
    write_guard: linear._HeldRegularWriteGuard | None = None,
) -> None:
    """Atomically rename one basename without replacing an existing entry."""
    library_name = ctypes.util.find_library("c")
    if not library_name:
        raise RuntimeError("atomic no-replace rename is unavailable")
    libc = ctypes.CDLL(library_name, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        rename_flags = 0x00000004
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        rename_flags = 1
    else:
        raise RuntimeError("atomic no-replace rename is unsupported on this platform")
    if write_guard is not None:
        # This is intentionally the final userspace operation before the
        # kernel no-replace syscall.  A wrapper that mutates first and then
        # calls the trusted helper is therefore caught before publication.
        write_guard.assert_clean()
    result = rename(directory_fd, source, directory_fd, destination, rename_flags)
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                observed_errno,
                "R2R-1 terminal destination already exists",
                destination_name,
            )
        raise OSError(observed_errno, os.strerror(observed_errno), destination_name)


def _verify_root_entry(
    parent_fd: int, root_name: str, root_fd: int, *, label: str
) -> None:
    observed = os.stat(root_name, dir_fd=parent_fd, follow_symlinks=False)
    held = os.fstat(root_fd)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != held.st_dev
        or observed.st_ino != held.st_ino
    ):
        raise ValueError(f"{label} canonical basename no longer names the held root")


def _verify_active_fit_publication_pair(
    parent_fd: int,
    fit_name: str,
    fit_fd: int,
    candidate_name: str,
    *,
    label: str,
    expected_parent_identity: Mapping[str, int],
    expected_root_identity: Mapping[str, int],
    running_fd: int,
    expected_running_identity: Mapping[str, int],
    expected_marker_raw: bytes,
    expected_root_names: set[str],
    marker_name: str,
) -> None:
    _verify_root_entry(parent_fd, fit_name, fit_fd, label=label)
    parent_identity = linear._directory_binding_identity(os.fstat(parent_fd))
    held_root_identity = linear._directory_binding_identity(os.fstat(fit_fd))
    root_entry_identity = linear._directory_binding_identity(
        os.stat(fit_name, dir_fd=parent_fd, follow_symlinks=False)
    )
    if (
        not linear._json_type_exact_equal(
            parent_identity, dict(expected_parent_identity)
        )
        or not linear._json_type_exact_equal(
            held_root_identity, dict(expected_root_identity)
        )
        or not linear._json_type_exact_equal(
            root_entry_identity, dict(expected_root_identity)
        )
    ):
        raise ValueError(f"{label}: stable directory binding changed")
    try:
        os.stat(candidate_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"{label}: candidate root is no longer absent")
    if set(os.listdir(fit_fd)) != expected_root_names:
        raise ValueError(f"{label}: fit-root inventory changed")
    running_entry = os.stat(
        marker_name, dir_fd=fit_fd, follow_symlinks=False
    )
    if (
        not linear._json_type_exact_equal(
            linear._owned_regular_identity(os.fstat(running_fd)),
            dict(expected_running_identity),
        )
        or not linear._json_type_exact_equal(
            linear._owned_regular_identity(running_entry),
            dict(expected_running_identity),
        )
    ):
        raise ValueError(f"{label}: active marker identity changed")
    marker_raw = _read_regular_file_at(
        fit_fd,
        marker_name,
        f"{label}: active marker payload",
        size_limit=ROOT_FILE_SIZE_LIMITS[marker_name],
        expected_identity=expected_running_identity,
    )
    if marker_raw != expected_marker_raw:
        raise ValueError(f"{label}: active marker payload changed")
    if set(os.listdir(fit_fd)) != expected_root_names:
        raise ValueError(f"{label}: fit-root inventory changed after marker read")
    try:
        os.stat(candidate_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"{label}: candidate root appeared during marker read")
    if (
        not linear._json_type_exact_equal(
            linear._directory_binding_identity(os.fstat(parent_fd)),
            dict(expected_parent_identity),
        )
        or not linear._json_type_exact_equal(
            linear._directory_binding_identity(os.fstat(fit_fd)),
            dict(expected_root_identity),
        )
    ):
        raise ValueError(f"{label}: stable directory binding changed after marker read")


def _reserve_canonical_root(
    final_root: Path,
    *,
    parent_fd: int,
    parent_chain: tuple[dict[str, int], ...],
) -> tuple[Path, int, tuple[dict[str, int], ...]]:
    """Reserve the one canonical release root with mkdirat(O_EXCL semantics)."""
    linear._verify_directory_chain(
        final_root.parent,
        "R2R-1 pre-reservation held parent binding",
        parent_chain,
    )
    try:
        os.stat(final_root.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("R2R-1 output root must be fresh")
    os.mkdir(final_root.name, 0o700, dir_fd=parent_fd)
    before = os.stat(final_root.name, dir_fd=parent_fd, follow_symlinks=False)
    root_fd: int | None = None
    try:
        root_fd = os.open(
            final_root.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        root_identity = linear._directory_binding_identity(os.fstat(root_fd))
        if root_identity != linear._directory_binding_identity(before):
            raise ValueError("R2R-1 canonical root changed across mkdir/open")
        root_chain = (*parent_chain, root_identity)
        os.fsync(parent_fd)
        _verify_root_entry(
            parent_fd, final_root.name, root_fd, label="R2R-1 reserved root"
        )
        linear._verify_directory_chain(
            final_root.parent,
            "R2R-1 reserved-root parent binding",
            parent_chain,
        )
        return final_root, root_fd, root_chain
    except BaseException:
        if root_fd is not None:
            os.close(root_fd)
        raise


def _open_existing_bound_child_directory(
    *,
    parent_path: Path,
    parent_fd: int,
    parent_chain: tuple[dict[str, int], ...],
    child_name: str,
    label: str,
) -> tuple[Path, int, tuple[dict[str, int], ...]]:
    before = os.stat(child_name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} is not a directory")
    child_fd: int | None = None
    try:
        child_fd = os.open(
            child_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        identity = linear._directory_binding_identity(os.fstat(child_fd))
        if identity != linear._directory_binding_identity(before):
            raise ValueError(f"{label} changed across stat/open")
        return parent_path / child_name, child_fd, (*parent_chain, identity)
    except BaseException:
        if child_fd is not None:
            os.close(child_fd)
        raise


def _create_bound_candidate_temporary(
    *,
    parent_path: Path,
    parent_fd: int,
    parent_chain: tuple[dict[str, int], ...],
) -> tuple[str, Path, int, tuple[dict[str, int], ...]]:
    linear._verify_directory_chain(
        parent_path, "R2R-1 candidate parent before temporary create", parent_chain
    )
    name = f".r2r1-candidate-{secrets.token_hex(16)}.tmp"
    os.mkdir(name, 0o700, dir_fd=parent_fd)
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        identity = linear._directory_binding_identity(os.fstat(descriptor))
        if identity != linear._directory_binding_identity(before):
            raise ValueError("R2R-1 candidate temporary changed across mkdir/open")
        os.fsync(parent_fd)
        return name, parent_path / name, descriptor, (*parent_chain, identity)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _commit_candidate_directory_at_guarded(
    parent_fd: int,
    temporary_name: str,
    destination_name: str,
    candidate_fd: int,
    *,
    candidate_directory_binding_identity: Mapping[str, int],
    anchor_name: str,
    anchor_identity: Mapping[str, int],
    anchor_raw: bytes,
    release_root_fd: int,
    expected_release_snapshot: Mapping[str, Mapping[str, Any]],
    release_parent_fd: int,
    release_root_name: str,
    release_parent_path: Path,
    release_parent_chain: Sequence[Mapping[str, int]],
    precommit_validator: Any,
    _write_guard: linear._HeldRegularWriteGuard,
) -> None:
    """NOREPLACE-commit a complete candidate directory without rollback."""
    held = os.fstat(candidate_fd)
    before = os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(before.st_mode)
        or not linear._json_type_exact_equal(
            linear._directory_binding_identity(before),
            dict(candidate_directory_binding_identity),
        )
        or not linear._json_type_exact_equal(
            linear._directory_binding_identity(held),
            dict(candidate_directory_binding_identity),
        )
    ):
        raise ValueError("R2R-1 candidate temporary basename changed before commit")
    if set(os.listdir(candidate_fd)) != {anchor_name}:
        raise ValueError("R2R-1 candidate temporary inventory changed before commit")
    observed_anchor = _read_regular_file_at(
        candidate_fd,
        anchor_name,
        "R2R-1 candidate anchor immediately before commit",
        size_limit=2 * 1024 * 1024,
        expected_identity=anchor_identity,
    )
    if observed_anchor != anchor_raw or _sha256_bytes(observed_anchor) != _sha256_bytes(
        anchor_raw
    ):
        raise ValueError("R2R-1 candidate anchor changed before directory commit")
    os.fsync(candidate_fd)
    rebound = os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not linear._json_type_exact_equal(
            linear._directory_binding_identity(rebound),
            dict(candidate_directory_binding_identity),
        )
        or not linear._json_type_exact_equal(
            linear._directory_binding_identity(os.fstat(candidate_fd)),
            dict(candidate_directory_binding_identity),
        )
        or set(os.listdir(candidate_fd)) != {anchor_name}
    ):
        raise ValueError("R2R-1 candidate temporary changed at commit boundary")
    final_anchor = _read_regular_file_at(
        candidate_fd,
        anchor_name,
        "R2R-1 candidate anchor at no-replace boundary",
        size_limit=2 * 1024 * 1024,
        expected_identity=anchor_identity,
    )
    if final_anchor != anchor_raw or _sha256_bytes(final_anchor) != _sha256_bytes(
        anchor_raw
    ):
        raise ValueError("R2R-1 candidate anchor changed at no-replace boundary")
    precommit_validator()
    # The callback performs the long content readback.  The helper itself then
    # closes the same frozen release baseline by bytes and owned identity, so
    # a callback-phase byte change cannot publish an anchor.
    linear._verify_directory_chain(
        release_parent_path,
        "R2R-1 release parent after anchor callback",
        release_parent_chain,
    )
    adjacent_anchor_raw = _read_regular_file_at(
        candidate_fd,
        anchor_name,
        "R2R-1 candidate anchor after release callback",
        size_limit=2 * 1024 * 1024,
        expected_identity=anchor_identity,
    )
    if adjacent_anchor_raw != anchor_raw:
        raise ValueError("R2R-1 candidate anchor bytes changed after callback")
    _verify_snapshot_bytes_fd(release_root_fd, expected_release_snapshot)
    if set(os.listdir(release_root_fd)) != {
        name for name in expected_release_snapshot if "/" not in name
    }:
        raise ValueError("R2R-1 release inventory changed after anchor callback")
    _verify_root_entry(
        release_parent_fd,
        release_root_name,
        release_root_fd,
        label="R2R-1 release root after anchor callback",
    )
    final_anchor_raw = _read_regular_file_at(
        candidate_fd,
        anchor_name,
        "R2R-1 candidate anchor after final release byte closure",
        size_limit=2 * 1024 * 1024,
        expected_identity=anchor_identity,
    )
    if final_anchor_raw != anchor_raw or _sha256_bytes(
        final_anchor_raw
    ) != _sha256_bytes(anchor_raw):
        raise ValueError(
            "R2R-1 candidate anchor bytes changed during final release validation"
        )
    if set(os.listdir(candidate_fd)) != {anchor_name}:
        raise ValueError(
            "R2R-1 candidate inventory changed during final release validation"
        )
    adjacent_anchor = os.stat(
        anchor_name, dir_fd=candidate_fd, follow_symlinks=False
    )
    if not linear._json_type_exact_equal(
        linear._owned_regular_identity(adjacent_anchor), dict(anchor_identity)
    ):
        raise ValueError("R2R-1 candidate anchor identity changed after release validation")
    if set(os.listdir(candidate_fd)) != {anchor_name}:
        raise ValueError("R2R-1 candidate inventory changed at rename boundary")
    if not linear._json_type_exact_equal(
        linear._directory_binding_identity(os.fstat(candidate_fd)),
        dict(candidate_directory_binding_identity),
    ):
        raise ValueError("R2R-1 candidate temporary metadata changed at rename boundary")
    _verify_root_entry(
        parent_fd,
        temporary_name,
        candidate_fd,
        label="R2R-1 candidate temporary at no-replace boundary",
    )
    try:
        _rename_noreplace_at(
            parent_fd,
            temporary_name,
            destination_name,
            write_guard=_write_guard,
        )
    except BaseException as error:
        # A wrapper may commit the directory, move the held inode back to the
        # temporary basename, and only then raise.  No post-exception path
        # probe can distinguish that from a pre-syscall failure.  Once the
        # no-replace helper has been entered, every exception is therefore a
        # committed/uncertain boundary and the candidate must not be cleaned.
        raise _AnchorCommittedError(error) from error
    # The guard was drained inside the trusted helper immediately before the
    # kernel syscall.  Ignore the intentional rename event from this point on.
    _write_guard.close()
    try:
        destination = os.stat(
            destination_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(destination.st_mode)
            or destination.st_dev != held.st_dev
            or destination.st_ino != held.st_ino
        ):
            raise ValueError("R2R-1 committed candidate directory identity changed")
        os.fsync(parent_fd)
    except BaseException as error:
        raise _AnchorCommittedError(error) from error


def _commit_candidate_directory_at(
    parent_fd: int,
    temporary_name: str,
    destination_name: str,
    candidate_fd: int,
    *,
    candidate_directory_binding_identity: Mapping[str, int],
    anchor_name: str,
    anchor_identity: Mapping[str, int],
    anchor_raw: bytes,
    release_root_fd: int,
    expected_release_snapshot: Mapping[str, Mapping[str, Any]],
    release_parent_fd: int,
    release_root_name: str,
    release_parent_path: Path,
    release_parent_chain: Sequence[Mapping[str, int]],
    precommit_validator: Any,
) -> None:
    """Guard every payload vnode through the candidate no-replace boundary."""
    with linear._HeldRegularWriteGuard(
        "R2R-1 candidate precommit authority set"
    ) as write_guard:
        linear._watch_owned_snapshot_fd(
            write_guard,
            release_root_fd,
            expected_release_snapshot,
            label="R2R-1 candidate-bound release snapshot",
        )
        write_guard.watch_owned_at(
            candidate_fd,
            anchor_name,
            expected_identity=anchor_identity,
            label="R2R-1 candidate anchor",
        )
        _commit_candidate_directory_at_guarded(
            parent_fd,
            temporary_name,
            destination_name,
            candidate_fd,
            candidate_directory_binding_identity=candidate_directory_binding_identity,
            anchor_name=anchor_name,
            anchor_identity=anchor_identity,
            anchor_raw=anchor_raw,
            release_root_fd=release_root_fd,
            expected_release_snapshot=expected_release_snapshot,
            release_parent_fd=release_parent_fd,
            release_root_name=release_root_name,
            release_parent_path=release_parent_path,
            release_parent_chain=release_parent_chain,
            precommit_validator=precommit_validator,
            _write_guard=write_guard,
        )


def _cleanup_owned_candidate_temporary(
    *,
    parent_fd: int,
    temporary_name: str,
    candidate_fd: int,
    anchor_name: str,
    anchor_identity: Mapping[str, int] | None,
    candidate_directory_binding_identity: Mapping[str, int] | None,
) -> bool:
    """Remove only the exact private temporary owned by this invocation."""
    try:
        _verify_root_entry(
            parent_fd,
            temporary_name,
            candidate_fd,
            label="R2R-1 candidate temporary cleanup",
        )
    except (FileNotFoundError, ValueError):
        return False
    if candidate_directory_binding_identity is None or not linear._json_type_exact_equal(
        linear._directory_binding_identity(os.fstat(candidate_fd)),
        dict(candidate_directory_binding_identity),
    ):
        return False
    names = set(os.listdir(candidate_fd))
    if names == {anchor_name} and anchor_identity is not None:
        observed = os.stat(anchor_name, dir_fd=candidate_fd, follow_symlinks=False)
        if linear._owned_regular_identity(observed) != dict(anchor_identity):
            return False
        os.unlink(anchor_name, dir_fd=candidate_fd)
    elif names:
        return False
    os.fsync(candidate_fd)
    if os.listdir(candidate_fd):
        return False
    _verify_root_entry(
        parent_fd,
        temporary_name,
        candidate_fd,
        label="R2R-1 candidate temporary before rmdir",
    )
    os.rmdir(temporary_name, dir_fd=parent_fd)
    os.fsync(parent_fd)
    return True


def _open_marker_bound_release_root(
    root: Path, authorization: Mapping[str, Any], *, label: str
) -> tuple[
    Path,
    int,
    tuple[dict[str, int], ...],
    Path,
    int,
    tuple[dict[str, int], ...],
]:
    binding = authorization["result_parent_binding"]
    parent, parent_fd, parent_chain = linear._open_directory_chain(
        root.parent, f"{label} parent", expected_chain=binding["chain"]
    )
    root_fd: int | None = None
    try:
        if (
            not linear._json_type_exact_equal(
                linear._directory_binding_identity(os.fstat(parent_fd)),
                binding["directory_identity"],
            )
            or root.name != binding["fit_output_child_basename"]
        ):
            raise PermissionError(f"{label} parent differs from marker binding")
        before = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"{label} is not a directory")
        root_fd = os.open(
            root.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        identity = linear._directory_binding_identity(os.fstat(root_fd))
        if identity != linear._directory_binding_identity(before):
            raise ValueError(f"{label} changed across stat/open")
        return parent, parent_fd, parent_chain, root, root_fd, (*parent_chain, identity)
    except BaseException:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
        raise


def _create_running_marker(root_fd: int) -> tuple[int, dict[str, int]]:
    temporary = f".RUNNING.{secrets.token_hex(16)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=root_fd,
    )
    published = False
    try:
        raw = (FORMAT + "\n").encode("ascii")
        created_identity = linear._owned_regular_identity(os.fstat(descriptor))
        initial_empty = _capture_empty_owned_descriptor(
            descriptor,
            label="R2R-1 initial empty RUNNING inode",
            expected_identity=created_identity,
        )
        written_identity = _rewrite_held_file(descriptor, raw)
        initial_private = _capture_held_regular_descriptor(
            descriptor,
            raw,
            label="R2R-1 initial private RUNNING inode",
            expected_identity=written_identity,
        )
        _verify_empty_to_payload_transition(
            initial_empty,
            initial_private,
            payload_size=len(raw),
            label="R2R-1 private RUNNING inode",
        )
        temporary_observed = os.stat(
            temporary, dir_fd=root_fd, follow_symlinks=False
        )
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(temporary_observed),
            initial_private["identity"],
        ):
            raise ValueError("R2R-1 private RUNNING changed before publication")
        os.link(
            temporary,
            "RUNNING",
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary, dir_fd=root_fd)
        published_record = _capture_held_regular_descriptor(
            descriptor, raw, label="R2R-1 published RUNNING inode"
        )
        if not linear._json_type_exact_equal(
            published_record["xattrs"], initial_private["xattrs"]
        ):
            raise ValueError("R2R-1 RUNNING authoritative xattrs changed")
        identity = dict(published_record["identity"])
        observed = os.stat("RUNNING", dir_fd=root_fd, follow_symlinks=False)
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(observed), identity
        ):
            raise ValueError("R2R-1 RUNNING marker changed across create/open")
        os.fsync(root_fd)
        final_record = _capture_held_regular_descriptor(
            descriptor, raw, label="R2R-1 final published RUNNING inode"
        )
        if not linear._json_type_exact_equal(final_record, published_record):
            raise ValueError("R2R-1 RUNNING changed after root fsync")
        return descriptor, dict(final_record["identity"])
    except BaseException:
        try:
            held = os.fstat(descriptor)
            for name in ("RUNNING" if published else None, temporary):
                if name is None:
                    continue
                try:
                    observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                    if (
                        observed.st_dev == held.st_dev
                        and observed.st_ino == held.st_ino
                    ):
                        os.unlink(name, dir_fd=root_fd)
                except FileNotFoundError:
                    pass
        finally:
            os.close(descriptor)
        raise


def _bound_checkpoint_capture(
    root_fd: int,
    *,
    expected_directory_identity: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    before = os.stat(
        CHECKPOINT_DIRNAME, dir_fd=root_fd, follow_symlinks=False
    )
    before_identity = linear._directory_binding_identity(before)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or (
            expected_directory_identity is not None
            and not linear._json_type_exact_equal(
                before_identity, dict(expected_directory_identity)
            )
        )
    ):
        raise ValueError("R2R-1 checkpoint entry is not a directory")
    checkpoint_fd = os.open(
        CHECKPOINT_DIRNAME,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=root_fd,
    )
    try:
        if (
            linear._directory_binding_identity(os.fstat(checkpoint_fd))
            != before_identity
        ):
            raise ValueError("R2R-1 checkpoint changed across stat/open")
        names = set(os.listdir(checkpoint_fd))
        if names != set(frozen.CHECKPOINT_FILE_SET):
            raise ValueError("R2R-1 checkpoint release must contain exactly three files")
        files: dict[str, dict[str, Any]] = {}
        for name in sorted(names):
            child_identity = linear._owned_regular_identity(
                os.stat(name, dir_fd=checkpoint_fd, follow_symlinks=False)
            )
            raw = _read_regular_file_at(
                checkpoint_fd,
                name,
                f"R2R-1 checkpoint {name}",
                size_limit=(
                    1024 * 1024
                    if name != frozen.CHECKPOINT_MARKER_BASENAME
                    else 4096
                ),
                expected_identity=child_identity,
            )
            files[name] = {"raw": raw, "identity": child_identity}
        with linear._HeldRegularWriteGuard(
            "R2R-1 checkpoint capture closure"
        ) as write_guard:
            for name in sorted(names):
                write_guard.watch_owned_at(
                    checkpoint_fd,
                    name,
                    expected_identity=files[name]["identity"],
                    label=f"R2R-1 checkpoint {name}",
                )
            for name in sorted(names, reverse=True):
                expected = files[name]
                rebound_raw = _read_regular_file_at(
                    checkpoint_fd,
                    name,
                    f"R2R-1 checkpoint second byte pass {name}",
                    size_limit=(
                        1024 * 1024
                        if name != frozen.CHECKPOINT_MARKER_BASENAME
                        else 4096
                    ),
                    expected_identity=expected["identity"],
                )
                if rebound_raw != expected["raw"]:
                    raise ValueError(
                        f"R2R-1 checkpoint child bytes changed while captured: {name}"
                    )
            rebound = os.stat(
                CHECKPOINT_DIRNAME, dir_fd=root_fd, follow_symlinks=False
            )
            if (
                not linear._json_type_exact_equal(
                    linear._directory_binding_identity(rebound), before_identity
                )
                or not linear._json_type_exact_equal(
                    linear._directory_binding_identity(os.fstat(checkpoint_fd)),
                    before_identity,
                )
                or set(os.listdir(checkpoint_fd)) != names
            ):
                raise ValueError("R2R-1 checkpoint changed while captured")
        return files
    finally:
        os.close(checkpoint_fd)


def _bound_checkpoint_files(root_fd: int) -> dict[str, bytes]:
    return {
        name: item["raw"]
        for name, item in _bound_checkpoint_capture(root_fd).items()
    }


def _release_manifest_payload_for_bound_root(
    *, root: Path, materialization_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    status = materialization_receipt.get("status")
    if status not in SCIENTIFIC_STATUSES:
        raise ValueError("cannot bind an unknown R2R-1 release status")
    checkpoint = materialization_receipt.get("checkpoint")
    checkpoint_receipt = (
        checkpoint.get("receipt_sha256") if isinstance(checkpoint, Mapping) else None
    )
    return {
        "format": RELEASE_MANIFEST_FORMAT,
        "release_root": str(
            linear._lexical_path(root, "R2R-1 bound release-root payload")
        ),
        "status": status,
        "fit_manifest_sha256": materialization_receipt["fit_manifest_sha256"],
        "expected_completion_sha256": materialization_receipt["completion_sha256"],
        "expected_fit_receipt_sha256": materialization_receipt[
            "fit_receipt_sha256"
        ],
        "expected_checkpoint_receipt_sha256": checkpoint_receipt,
        "mechanics_completed": False,
        "development_or_held_access": False,
        "energy_labels_used": False,
        "expected_done_sha256": materialization_receipt.get("done_sha256"),
        "expected_done_identity": materialization_receipt.get("done_identity"),
        "precompletion_artifact_snapshot_sha256": materialization_receipt[
            "precompletion_artifact_snapshot_sha256"
        ],
    }


def _completed_materialization_receipt_bound(
    root_fd: int,
    *,
    expected_done_identity: Mapping[str, int] | None = None,
    expected_done_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a DONE release from its persistent ledger, never a live baseline."""
    names = set(os.listdir(root_fd))
    if names & {"RUNNING", "FAILED"} or "DONE" not in names:
        raise ValueError("R2R-1 completed release terminal XOR changed")
    done_identity = linear._owned_regular_identity(
        os.stat("DONE", dir_fd=root_fd, follow_symlinks=False)
    )
    if expected_done_identity is not None and not linear._json_type_exact_equal(
        done_identity, dict(expected_done_identity)
    ):
        raise ValueError("R2R-1 DONE identity differs from its external anchor")
    done_raw = _read_regular_file_at(
        root_fd,
        "DONE",
        "R2R-1 completed DONE ledger",
        size_limit=ROOT_FILE_SIZE_LIMITS["DONE"],
        expected_identity=done_identity,
    )
    done_sha = _sha256_bytes(done_raw)
    if expected_done_sha256 is not None and done_sha != _expected_sha256(
        expected_done_sha256, "externally pinned DONE SHA-256"
    ):
        raise ValueError("R2R-1 DONE bytes differ from their external anchor")
    ledger = _validate_terminal_ledger(
        _strict_canonical_json(done_raw, "R2R-1 completed DONE ledger"),
        expected_terminal="DONE",
    )
    if not linear._json_type_exact_equal(
        _file_stable_identity(done_identity),
        ledger["terminal_inode_stable_identity"],
    ):
        raise ValueError("R2R-1 DONE inode differs from its RUNNING provenance")
    root_binding = linear._directory_binding_identity(os.fstat(root_fd))
    if not linear._json_type_exact_equal(
        root_binding, ledger["root_binding_identity"]
    ):
        raise ValueError("R2R-1 completed root stable binding changed")

    completion_identity = _validate_owned_regular_identity(
        ledger["payload_identity"], "R2R-1 ledger completion identity"
    )
    completion_raw = _read_regular_file_at(
        root_fd,
        COMPLETION_BASENAME,
        "R2R-1 ledger-bound completion",
        size_limit=ROOT_FILE_SIZE_LIMITS[COMPLETION_BASENAME],
        expected_identity=completion_identity,
    )
    completion_sha = _sha256_bytes(completion_raw)
    if completion_sha != ledger["payload_sha256"]:
        raise ValueError("R2R-1 completion differs from the DONE ledger")
    completion = _strict_canonical_json(
        completion_raw, "R2R-1 completed release completion"
    )
    _validate_completion_invariants(completion)
    if (
        completion["status"] != ledger["status"]
        or not linear._json_type_exact_equal(
            completion["release_root_binding_identity"],
            ledger["root_binding_identity"],
        )
        or completion["precompletion_artifact_snapshot_sha256"]
        != ledger["artifact_snapshot_sha256"]
    ):
        raise ValueError("R2R-1 completion/DONE ledger binding changed")

    exit_identity = _validate_owned_regular_identity(
        ledger["exit_identity"], "R2R-1 ledger EXIT_CODE identity"
    )
    exit_raw = _read_regular_file_at(
        root_fd,
        "EXIT_CODE",
        "R2R-1 ledger-bound EXIT_CODE",
        size_limit=ROOT_FILE_SIZE_LIMITS["EXIT_CODE"],
        expected_identity=exit_identity,
    )
    if exit_raw != b"0\n" or _sha256_bytes(exit_raw) != ledger["exit_sha256"]:
        raise ValueError("R2R-1 EXIT_CODE differs from the DONE ledger")

    precompletion_snapshot = _validate_persisted_artifact_snapshot(
        completion["precompletion_artifact_snapshot"],
        status=completion["status"],
        include_completion=False,
        label="R2R-1 persisted precompletion snapshot",
    )
    expected_science_snapshot = {
        **precompletion_snapshot,
        COMPLETION_BASENAME: _owned_file_snapshot(
            completion_identity, completion_raw
        ),
    }
    expected_names = _science_names(
        completion["status"] == CHECKPOINT_STATUS_SOURCE
    ) | {"EXIT_CODE", "DONE"}
    if names != expected_names:
        raise ValueError("R2R-1 completed release inventory changed")
    _verify_snapshot_bytes_fd(root_fd, expected_science_snapshot)

    fit_receipt_item = expected_science_snapshot[FIT_RECEIPT_BASENAME]
    fit_receipt_raw = _read_regular_file_at(
        root_fd,
        FIT_RECEIPT_BASENAME,
        "R2R-1 ledger-bound fit receipt",
        size_limit=ROOT_FILE_SIZE_LIMITS[FIT_RECEIPT_BASENAME],
        expected_identity=fit_receipt_item["identity"],
    )
    fit_receipt = _strict_canonical_json(
        fit_receipt_raw, "R2R-1 completed release fit receipt"
    )
    _validate_fit_receipt_invariants(fit_receipt)
    arrays_item = expected_science_snapshot[FIT_ARRAYS_BASENAME]
    arrays_raw = _read_regular_file_at(
        root_fd,
        FIT_ARRAYS_BASENAME,
        "R2R-1 ledger-bound arrays",
        size_limit=ROOT_FILE_SIZE_LIMITS[FIT_ARRAYS_BASENAME],
        expected_identity=arrays_item["identity"],
    )
    _load_npz_bytes(
        arrays_raw,
        fit_receipt["fit_arrays"]["schema"],
        "R2R-1 completed release arrays",
    )
    _verify_snapshot_bytes_fd(root_fd, expected_science_snapshot)
    if not linear._json_type_exact_equal(
        linear._owned_regular_identity(
            os.stat("DONE", dir_fd=root_fd, follow_symlinks=False)
        ),
        done_identity,
    ):
        raise ValueError("R2R-1 DONE identity changed during ledger validation")
    return {
        **completion,
        "completion_sha256": completion_sha,
        "fit_receipt_sha256": fit_receipt_item["sha256"],
        "done_sha256": done_sha,
        "done_identity": done_identity,
        "exit_identity": exit_identity,
        "terminal_ledger": ledger,
        "expected_release_snapshot": {
            **expected_science_snapshot,
            "EXIT_CODE": _owned_file_snapshot(exit_identity, exit_raw),
            "DONE": _owned_file_snapshot(done_identity, done_raw),
        },
    }


def _load_external_release_manifest_raw(
    raw: bytes,
    *,
    expected_sha256: str,
    output_root: Path,
    authorization: Mapping[str, Any],
    require_terminal: bool,
) -> dict[str, Any]:
    if _sha256_bytes(raw) != _expected_sha256(
        expected_sha256, "expected external release manifest SHA-256"
    ):
        raise ValueError("R2R-1 external release manifest SHA changed")
    payload = _strict_canonical_json(raw, "R2R-1 external release manifest")
    _exact_keys(payload, RELEASE_MANIFEST_KEYS, "external release manifest")
    if (
        payload.get("format") != RELEASE_MANIFEST_FORMAT
        or payload.get("release_root")
        != str(linear._lexical_path(output_root, "R2R-1 release-root manifest check"))
        or payload.get("status") not in SCIENTIFIC_STATUSES
        or payload.get("fit_manifest_sha256")
        != authorization["freeze_manifest_sha256"]
        or payload.get("mechanics_completed") is not False
        or payload.get("development_or_held_access") is not False
        or payload.get("energy_labels_used") is not False
    ):
        raise ValueError("R2R-1 external release manifest invariants changed")
    _expected_sha256(
        payload.get("expected_completion_sha256"),
        "release manifest completion SHA-256",
    )
    _expected_sha256(
        payload.get("expected_fit_receipt_sha256"),
        "release manifest fit receipt SHA-256",
    )
    checkpoint = payload.get("expected_checkpoint_receipt_sha256")
    if payload["status"] == CHECKPOINT_STATUS_SOURCE:
        _expected_sha256(checkpoint, "release manifest checkpoint receipt SHA-256")
    elif checkpoint is not None:
        raise ValueError("failed-gate release manifest cannot bind a checkpoint")
    snapshot_digest = _expected_sha256(
        payload.get("precompletion_artifact_snapshot_sha256"),
        "release manifest precompletion snapshot SHA-256",
    )
    del snapshot_digest
    done_sha = payload.get("expected_done_sha256")
    done_identity = payload.get("expected_done_identity")
    if require_terminal:
        _expected_sha256(done_sha, "release manifest DONE SHA-256")
        _validate_owned_regular_identity(done_identity, "release manifest DONE identity")
    elif done_sha is not None or done_identity is not None:
        raise ValueError("prepublish release manifest cannot bind a DONE inode")
    return payload


def _load_settled_private_checkpoint(
    checkpoint_files: Mapping[str, bytes],
    *,
    expected_receipt_sha256: str,
    prefix: str,
) -> frozen.FrozenReadoutCheckpoint:
    """Load a provenance-settled, identity-frozen private checkpoint copy."""
    if set(checkpoint_files) != set(frozen.CHECKPOINT_FILE_SET):
        raise ValueError("R2R-1 private checkpoint file set changed")
    with tempfile.TemporaryDirectory(prefix=prefix, dir="/private/tmp") as temporary:
        checkpoint_copy = Path(temporary) / "checkpoint"
        checkpoint_copy.mkdir(mode=0o700)
        copy_fd = os.open(
            checkpoint_copy,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            directory_binding = linear._directory_binding_identity(
                os.fstat(copy_fd)
            )
            if not linear._json_type_exact_equal(
                linear._directory_binding_identity(
                    os.stat(checkpoint_copy, follow_symlinks=False)
                ),
                directory_binding,
            ):
                raise ValueError(
                    "R2R-1 private checkpoint directory changed across open"
                )
            records: dict[str, dict[str, Any]] = {}
            for name, raw_value in sorted(checkpoint_files.items()):
                raw = bytes(raw_value)
                identity = _atomic_write_bytes_at(copy_fd, name, raw)
                records[name] = {
                    "identity": identity,
                    "raw": raw,
                }
            os.fsync(copy_fd)
            if (
                set(os.listdir(copy_fd)) != set(frozen.CHECKPOINT_FILE_SET)
                or not linear._json_type_exact_equal(
                    linear._directory_binding_identity(os.fstat(copy_fd)),
                    directory_binding,
                )
            ):
                raise ValueError(
                    "R2R-1 private checkpoint changed before frozen load"
                )
            loaded = frozen.load_frozen_readout_checkpoint(
                checkpoint_copy,
                expected_receipt_sha256=expected_receipt_sha256,
            )
            # The frozen loader performs several lexical opens.  Close that
            # long phase by rebinding every owned regular inode/byte record and
            # the stable directory/inventory before returning its in-memory
            # state to the caller.
            if (
                set(os.listdir(copy_fd)) != set(frozen.CHECKPOINT_FILE_SET)
                or not linear._json_type_exact_equal(
                    linear._directory_binding_identity(os.fstat(copy_fd)),
                    directory_binding,
                )
                or not linear._json_type_exact_equal(
                    linear._directory_binding_identity(
                        os.stat(checkpoint_copy, follow_symlinks=False)
                    ),
                    directory_binding,
                )
            ):
                raise ValueError(
                    "R2R-1 private checkpoint directory changed during frozen load"
                )
            for name, record in sorted(records.items()):
                observed_raw = _read_regular_file_at(
                    copy_fd,
                    name,
                    f"R2R-1 private checkpoint post-load {name}",
                    size_limit=1024 * 1024,
                    expected_identity=record["identity"],
                )
                if observed_raw != record["raw"]:
                    raise ValueError(
                        "R2R-1 private checkpoint changed during frozen load"
                    )
            if set(os.listdir(copy_fd)) != set(frozen.CHECKPOINT_FILE_SET):
                raise ValueError(
                    "R2R-1 private checkpoint inventory changed after frozen load"
                )
            return loaded
        finally:
            os.close(copy_fd)


def _write_checkpoint(
    physical_p: np.ndarray,
    *,
    fit_manifest_sha256: str,
    fit_receipt_sha256: str,
    release_root_fd: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    os.mkdir(CHECKPOINT_DIRNAME, 0o700, dir_fd=release_root_fd)
    before = os.stat(
        CHECKPOINT_DIRNAME, dir_fd=release_root_fd, follow_symlinks=False
    )
    checkpoint_fd = os.open(
        CHECKPOINT_DIRNAME,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=release_root_fd,
    )
    try:
        checkpoint_binding = linear._directory_binding_identity(before)
        if not linear._json_type_exact_equal(
            linear._directory_binding_identity(os.fstat(checkpoint_fd)),
            checkpoint_binding,
        ):
            raise ValueError("R2R-1 checkpoint changed across mkdir/open")
        physical = np.asarray(physical_p, dtype="<f8")
        if physical.shape != (65,) or not np.all(np.isfinite(physical)):
            raise ValueError("R2R-1 final physical_p is not one finite 65-vector")
        array_identity = _atomic_write_npz_at(
            checkpoint_fd,
            frozen.CHECKPOINT_ARRAY_BASENAME,
            {"physical_p": physical},
        )
        array_raw = _read_regular_file_at(
            checkpoint_fd,
            frozen.CHECKPOINT_ARRAY_BASENAME,
            "R2R-1 checkpoint arrays",
            size_limit=1024 * 1024,
        )
        array_sha = _sha256_bytes(array_raw)
        receipt = frozen.checkpoint_receipt_payload(
            physical,
            array_sha256=array_sha,
            fit_manifest_sha256=fit_manifest_sha256,
            fit_receipt_sha256=fit_receipt_sha256,
        )
        receipt_identity = _atomic_write_json_at(
            checkpoint_fd, frozen.CHECKPOINT_RECEIPT_BASENAME, receipt
        )
        receipt_raw = _read_regular_file_at(
            checkpoint_fd,
            frozen.CHECKPOINT_RECEIPT_BASENAME,
            "R2R-1 checkpoint receipt",
            size_limit=1024 * 1024,
        )
        receipt_sha = _sha256_bytes(receipt_raw)
        marker = (
            frozen.CHECKPOINT_STATUS + "\n" + receipt_sha + "\n"
        ).encode("ascii")
        marker_identity = _atomic_write_bytes_at(
            checkpoint_fd, frozen.CHECKPOINT_MARKER_BASENAME, marker
        )
        os.fsync(checkpoint_fd)
    finally:
        os.close(checkpoint_fd)
    checkpoint_files = _bound_checkpoint_files(release_root_fd)
    loaded = _load_settled_private_checkpoint(
        checkpoint_files,
        expected_receipt_sha256=receipt_sha,
        prefix="graphene_r2r1_checkpoint_write_verify_",
    )
    if not np.array_equal(loaded.physical_p, physical):
        raise ValueError("R2R-1 frozen checkpoint round-trip changed physical_p")
    summary = {
        "receipt_sha256": receipt_sha,
        "array_sha256": array_sha,
        "physical_p_raw_sha256": linear.raw_array_sha256(physical, "<f8"),
        "file_sha256": {
            name: _sha256_bytes(raw) for name, raw in checkpoint_files.items()
        },
    }
    owned_snapshot: dict[str, dict[str, Any]] = {
        CHECKPOINT_DIRNAME: {
            "kind": "directory",
            "identity": checkpoint_binding,
        },
        f"{CHECKPOINT_DIRNAME}/{frozen.CHECKPOINT_ARRAY_BASENAME}": (
            _owned_file_snapshot(array_identity, array_raw)
        ),
        f"{CHECKPOINT_DIRNAME}/{frozen.CHECKPOINT_RECEIPT_BASENAME}": (
            _owned_file_snapshot(receipt_identity, receipt_raw)
        ),
        f"{CHECKPOINT_DIRNAME}/{frozen.CHECKPOINT_MARKER_BASENAME}": (
            _owned_file_snapshot(marker_identity, marker)
        ),
    }
    return summary, owned_snapshot


def _root_file(root_fd: int, name: str, label: str) -> bytes:
    return _read_regular_file_at(
        root_fd,
        name,
        label,
        size_limit=ROOT_FILE_SIZE_LIMITS[name],
    )


def _science_names(checkpoint_required: bool) -> set[str]:
    names = {
        FIT_MANIFEST_BASENAME,
        FIT_ARRAYS_BASENAME,
        FIT_RECEIPT_BASENAME,
        COMPLETION_BASENAME,
    }
    if checkpoint_required:
        names.add(CHECKPOINT_DIRNAME)
    return names


def _owned_file_snapshot(
    identity: Mapping[str, int], raw: bytes
) -> dict[str, Any]:
    """Build a release record from the identity returned by atomic creation."""
    payload = bytes(raw)
    return {
        "kind": "file",
        "identity": dict(identity),
        "sha256": _sha256_bytes(payload),
        "size": len(payload),
    }


def _validate_full_file_identity(value: Any, label: str) -> dict[str, int]:
    identity = _exact_keys(value, FULL_FILE_IDENTITY_KEYS, label)
    if (
        any(type(identity[key]) is not int for key in identity)
        or not stat.S_ISREG(identity["st_mode"])
        or identity["st_nlink"] != 1
        or identity["st_size"] < 0
    ):
        raise ValueError(f"{label} is not one frozen single-link regular identity")
    return dict(identity)


def _validate_owned_regular_identity(value: Any, label: str) -> dict[str, int]:
    identity = _exact_keys(value, OWNED_REGULAR_IDENTITY_KEYS, label)
    if (
        any(type(identity[key]) is not int for key in identity)
        or not stat.S_ISREG(identity["st_mode"])
        or identity["st_nlink"] != 1
        or identity["st_size"] < 0
    ):
        raise ValueError(f"{label} is not one owned single-link regular identity")
    return dict(identity)


def _validate_directory_binding_identity(
    value: Any, label: str
) -> dict[str, int]:
    identity = _exact_keys(value, DIRECTORY_BINDING_IDENTITY_KEYS, label)
    if any(type(identity[key]) is not int for key in identity) or not stat.S_ISDIR(
        identity["st_mode"]
    ):
        raise ValueError(f"{label} is not one frozen directory binding")
    return dict(identity)


def _snapshot_expected_names(
    status: str, *, include_completion: bool
) -> set[str]:
    names = {
        FIT_MANIFEST_BASENAME,
        FIT_ARRAYS_BASENAME,
        FIT_RECEIPT_BASENAME,
    }
    if include_completion:
        names.add(COMPLETION_BASENAME)
    if status == CHECKPOINT_STATUS_SOURCE:
        names.add(CHECKPOINT_DIRNAME)
        names.update(
            f"{CHECKPOINT_DIRNAME}/{name}"
            for name in frozen.CHECKPOINT_FILE_SET
        )
    return names


def _validate_persisted_artifact_snapshot(
    value: Any,
    *,
    status: str,
    include_completion: bool,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != _snapshot_expected_names(
        status, include_completion=include_completion
    ):
        raise ValueError(f"{label} inventory differs from the frozen status schema")
    snapshot: dict[str, dict[str, Any]] = {}
    for name in sorted(value):
        item = value[name]
        if name == CHECKPOINT_DIRNAME:
            directory = _exact_keys(item, SNAPSHOT_DIRECTORY_KEYS, f"{label} {name}")
            if directory.get("kind") != "directory":
                raise ValueError(f"{label} checkpoint kind changed")
            snapshot[name] = {
                "kind": "directory",
                "identity": _validate_directory_binding_identity(
                    directory.get("identity"), f"{label} {name} identity"
                ),
            }
            continue
        artifact = _exact_keys(item, SNAPSHOT_FILE_KEYS, f"{label} {name}")
        if artifact.get("kind") != "file":
            raise ValueError(f"{label} file kind changed: {name}")
        identity = _validate_owned_regular_identity(
            artifact.get("identity"), f"{label} {name} identity"
        )
        digest = _expected_sha256(
            artifact.get("sha256"), f"{label} {name} SHA-256"
        )
        size = artifact.get("size")
        if type(size) is not int or size < 0 or size != identity["st_size"]:
            raise ValueError(f"{label} file size/identity changed: {name}")
        snapshot[name] = {
            "kind": "file",
            "identity": identity,
            "sha256": digest,
            "size": size,
        }
    return snapshot


def _artifact_snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    return linear.semantic_sha256(snapshot)


def _file_stable_identity(identity: Mapping[str, int]) -> dict[str, int]:
    return {key: int(identity[key]) for key in FILE_STABLE_IDENTITY_KEYS}


def _terminal_ledger_payload(
    *,
    terminal: str,
    status: str,
    payload_basename: str,
    payload_raw: bytes,
    payload_identity: Mapping[str, int],
    exit_raw: bytes,
    exit_identity: Mapping[str, int],
    root_binding_identity: Mapping[str, int],
    artifact_snapshot_sha256: str,
    terminal_inode_identity: Mapping[str, int],
) -> dict[str, Any]:
    if terminal not in {"DONE", "FAILED"}:
        raise ValueError("R2R-1 terminal ledger destination changed")
    ledger = {
        "format": TERMINAL_LEDGER_FORMAT,
        "terminal": terminal,
        "status": status,
        "payload_basename": payload_basename,
        "payload_sha256": _sha256_bytes(payload_raw),
        "payload_identity": dict(payload_identity),
        "exit_sha256": _sha256_bytes(exit_raw),
        "exit_identity": dict(exit_identity),
        "root_binding_identity": dict(root_binding_identity),
        "artifact_snapshot_sha256": artifact_snapshot_sha256,
        "terminal_inode_stable_identity": {
            key: terminal_inode_identity[key]
            for key in FILE_STABLE_IDENTITY_KEYS
        },
    }
    _validate_terminal_ledger(ledger, expected_terminal=terminal)
    return ledger


def _validate_terminal_ledger(
    value: Any, *, expected_terminal: str
) -> dict[str, Any]:
    ledger = _exact_keys(value, TERMINAL_LEDGER_KEYS, "R2R-1 terminal ledger")
    expected_payload = (
        COMPLETION_BASENAME if expected_terminal == "DONE" else "failure.json"
    )
    expected_status = None if expected_terminal == "DONE" else "R2R1_NUMERICAL_INCONCLUSIVE"
    if (
        ledger.get("format") != TERMINAL_LEDGER_FORMAT
        or ledger.get("terminal") != expected_terminal
        or ledger.get("payload_basename") != expected_payload
        or type(ledger.get("status")) is not str
        or (
            expected_terminal == "DONE"
            and ledger.get("status") not in SCIENTIFIC_STATUSES
        )
        or (
            expected_status is not None and ledger.get("status") != expected_status
        )
    ):
        raise ValueError("R2R-1 terminal ledger format/status changed")
    _expected_sha256(ledger.get("payload_sha256"), "terminal payload SHA-256")
    _expected_sha256(ledger.get("exit_sha256"), "terminal EXIT_CODE SHA-256")
    _expected_sha256(
        ledger.get("artifact_snapshot_sha256"), "terminal artifact snapshot SHA-256"
    )
    _validate_owned_regular_identity(
        ledger.get("payload_identity"), "terminal payload identity"
    )
    _validate_owned_regular_identity(ledger.get("exit_identity"), "terminal EXIT identity")
    _validate_directory_binding_identity(
        ledger.get("root_binding_identity"), "terminal root binding"
    )
    stable_terminal = _exact_keys(
        ledger.get("terminal_inode_stable_identity"),
        FILE_STABLE_IDENTITY_KEYS,
        "terminal inode stable identity",
    )
    if (
        any(type(stable_terminal[key]) is not int for key in stable_terminal)
        or not stat.S_ISREG(stable_terminal["st_mode"])
        or stable_terminal["st_nlink"] != 1
    ):
        raise ValueError("R2R-1 terminal inode stable identity changed")
    return dict(ledger)


def _verify_snapshot_bytes_fd(
    root_fd: int,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    """Verify a frozen snapshot while every member vnode is write-watched."""
    with linear._HeldRegularWriteGuard(
        "R2R-1 ledger-bound snapshot closure"
    ) as write_guard:
        linear._watch_owned_snapshot_fd(
            write_guard,
            root_fd,
            snapshot,
            label="R2R-1 ledger-bound snapshot",
        )
        _verify_snapshot_bytes_fd_unwatched(root_fd, snapshot)


def _verify_snapshot_and_anchor_bytes_fd(
    root_fd: int,
    snapshot: Mapping[str, Mapping[str, Any]],
    candidate_fd: int,
    anchor_name: str,
    anchor_identity: Mapping[str, int],
    anchor_raw: bytes,
    *,
    label: str,
) -> None:
    """Close release and anchor bytes under one held-vnode event history."""
    with linear._HeldRegularWriteGuard(f"{label} authority set") as write_guard:
        linear._watch_owned_snapshot_fd(
            write_guard,
            root_fd,
            snapshot,
            label=f"{label} release snapshot",
        )
        write_guard.watch_owned_at(
            candidate_fd,
            anchor_name,
            expected_identity=anchor_identity,
            label=f"{label} candidate anchor",
        )
        _verify_snapshot_bytes_fd(root_fd, snapshot)
        observed_anchor = _read_regular_file_at(
            candidate_fd,
            anchor_name,
            f"{label} candidate anchor bytes",
            size_limit=2 * 1024 * 1024,
            expected_identity=anchor_identity,
        )
        if observed_anchor != anchor_raw or _sha256_bytes(
            observed_anchor
        ) != _sha256_bytes(anchor_raw):
            raise ValueError(f"{label} candidate anchor bytes changed")


def _verify_snapshot_bytes_fd_unwatched(
    root_fd: int,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    """Verify one frozen owned snapshot with two complete byte passes."""
    top_level = {name for name in snapshot if "/" not in name}
    expected_inventory = set(os.listdir(root_fd))
    if not top_level.issubset(expected_inventory):
        raise ValueError("R2R-1 ledger-bound snapshot is missing from the root")
    checkpoint_children = {
        name.split("/", 1)[1]: item
        for name, item in snapshot.items()
        if name.startswith(CHECKPOINT_DIRNAME + "/")
    }

    def verify_top_level(*, reverse: bool) -> None:
        for name in sorted(top_level, reverse=reverse):
            item = snapshot[name]
            if item.get("kind") == "directory":
                continue
            raw = _read_regular_file_at(
                root_fd,
                name,
                f"R2R-1 ledger-bound artifact {name}",
                size_limit=ROOT_FILE_SIZE_LIMITS[name],
                expected_identity=item["identity"],
            )
            if len(raw) != item["size"] or _sha256_bytes(raw) != item["sha256"]:
                raise ValueError(
                    f"R2R-1 ledger-bound artifact bytes changed: {name}"
                )

    def verify_checkpoint() -> None:
        if not checkpoint_children:
            return
        parent = snapshot[CHECKPOINT_DIRNAME]
        captured = _bound_checkpoint_capture(
            root_fd, expected_directory_identity=parent["identity"]
        )
        if set(captured) != set(checkpoint_children):
            raise ValueError("R2R-1 ledger-bound checkpoint inventory changed")
        for child, item in checkpoint_children.items():
            observed = captured[child]
            if (
                not linear._json_type_exact_equal(
                    observed["identity"], item["identity"]
                )
                or len(observed["raw"]) != item["size"]
                or _sha256_bytes(observed["raw"]) != item["sha256"]
            ):
                raise ValueError(
                    f"R2R-1 ledger-bound checkpoint child changed: {child}"
                )

    for reverse in (False, True):
        _verify_snapshot_identities_fd(root_fd, snapshot)
        if reverse:
            verify_checkpoint()
            verify_top_level(reverse=True)
        else:
            verify_top_level(reverse=False)
            verify_checkpoint()
        if set(os.listdir(root_fd)) != expected_inventory:
            raise ValueError("R2R-1 ledger-bound root inventory changed")
    _verify_snapshot_identities_fd(root_fd, snapshot)
    if set(os.listdir(root_fd)) != expected_inventory:
        raise ValueError("R2R-1 ledger-bound root inventory changed after closure")


def _verify_snapshot_identities_fd(
    root_fd: int, snapshot: Mapping[str, Mapping[str, Any]]
) -> None:
    """Second-pass rebind of every entry captured through the held root fd."""
    top_level = {name for name in snapshot if "/" not in name}
    for name in sorted(top_level):
        item = snapshot[name]
        observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        observed_identity = (
            linear._directory_binding_identity(observed)
            if item.get("kind") == "directory"
            else linear._owned_regular_identity(observed)
        )
        if not linear._json_type_exact_equal(
            observed_identity, item["identity"]
        ):
            raise ValueError(f"R2R-1 snapshot identity changed after capture: {name}")
    child_names = {
        name.split("/", 1)[1]
        for name in snapshot
        if name.startswith(CHECKPOINT_DIRNAME + "/")
    }
    if child_names:
        directory_item = snapshot.get(CHECKPOINT_DIRNAME)
        if directory_item is None:
            raise ValueError("R2R-1 checkpoint snapshot lost its parent directory")
        before = os.stat(
            CHECKPOINT_DIRNAME, dir_fd=root_fd, follow_symlinks=False
        )
        if not linear._json_type_exact_equal(
            linear._directory_binding_identity(before), directory_item["identity"]
        ):
            raise ValueError("R2R-1 checkpoint identity changed after capture")
        checkpoint_fd = os.open(
            CHECKPOINT_DIRNAME,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        try:
            if not linear._json_type_exact_equal(
                linear._directory_binding_identity(os.fstat(checkpoint_fd)),
                directory_item["identity"],
            ):
                raise ValueError("R2R-1 checkpoint changed across final rebind")
            if set(os.listdir(checkpoint_fd)) != child_names:
                raise ValueError("R2R-1 checkpoint inventory changed after capture")
            for child in sorted(child_names):
                observed = os.stat(
                    child, dir_fd=checkpoint_fd, follow_symlinks=False
                )
                if not linear._json_type_exact_equal(
                    linear._owned_regular_identity(observed),
                    snapshot[f"{CHECKPOINT_DIRNAME}/{child}"]["identity"],
                ):
                    raise ValueError(
                        f"R2R-1 checkpoint child identity changed: {child}"
                    )
        finally:
            os.close(checkpoint_fd)


def _science_snapshot_fd(
    root_fd: int, *, checkpoint_required: bool
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name in sorted(_science_names(checkpoint_required)):
        observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if name == CHECKPOINT_DIRNAME:
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ValueError("R2R-1 checkpoint is not a directory")
            snapshot[name] = {
                "kind": "directory",
                "identity": linear._directory_binding_identity(observed),
            }
            files = _bound_checkpoint_capture(
                root_fd,
                expected_directory_identity=linear._directory_binding_identity(
                    observed
                ),
            )
            for child, item in sorted(files.items()):
                raw = item["raw"]
                snapshot[f"{name}/{child}"] = {
                    "kind": "file",
                    "identity": item["identity"],
                    "sha256": _sha256_bytes(raw),
                    "size": len(raw),
                }
        else:
            raw = _root_file(root_fd, name, f"R2R-1 snapshot {name}")
            snapshot[name] = {
                "kind": "file",
                "identity": linear._owned_regular_identity(observed),
                "sha256": _sha256_bytes(raw),
                "size": len(raw),
            }
    _verify_snapshot_bytes_fd(root_fd, snapshot)
    return snapshot


def _release_snapshot_fd(
    root_fd: int,
    *,
    checkpoint_required: bool,
    terminal_name: str,
) -> dict[str, dict[str, Any]]:
    """Capture every trusted release byte through one held root descriptor."""
    expected_names = _science_names(checkpoint_required) | {terminal_name}
    if terminal_name == "DONE":
        expected_names.add("EXIT_CODE")
    if set(os.listdir(root_fd)) != expected_names:
        raise ValueError("R2R-1 release snapshot inventory changed")
    snapshot = _science_snapshot_fd(
        root_fd, checkpoint_required=checkpoint_required
    )
    for name in sorted(expected_names - _science_names(checkpoint_required)):
        observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        raw = _root_file(root_fd, name, f"R2R-1 release snapshot {name}")
        snapshot[name] = {
            "kind": "file",
            "identity": linear._owned_regular_identity(observed),
            "sha256": _sha256_bytes(raw),
            "size": len(raw),
        }
    _verify_snapshot_bytes_fd(root_fd, snapshot)
    if set(os.listdir(root_fd)) != expected_names:
        raise ValueError("R2R-1 release snapshot inventory changed after capture")
    return snapshot


def _validate_success_content_fd(
    root_fd: int,
    completion: Mapping[str, Any],
    expected_science_snapshot: Mapping[str, Any],
    *,
    terminal_name: str,
    expected_exit_identity: Mapping[str, int],
    expected_terminal_raw: bytes,
) -> None:
    checkpoint_required = completion["status"] == CHECKPOINT_STATUS_SOURCE
    expected_names = _science_names(checkpoint_required) | {"EXIT_CODE", terminal_name}
    if set(os.listdir(root_fd)) != expected_names:
        raise ValueError("R2R-1 success tree inventory changed before commit")
    def entry_identity(name: str) -> dict[str, int]:
        observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        return (
            linear._directory_binding_identity(observed)
            if name == CHECKPOINT_DIRNAME
            else linear._owned_regular_identity(observed)
        )

    initial_identities = {
        name: entry_identity(name) for name in sorted(expected_names)
    }
    if not linear._json_type_exact_equal(
        initial_identities["EXIT_CODE"], dict(expected_exit_identity)
    ):
        raise ValueError("R2R-1 success EXIT_CODE identity changed")
    if _root_file(root_fd, "EXIT_CODE", "R2R-1 success EXIT_CODE") != b"0\n":
        raise ValueError("R2R-1 success EXIT_CODE changed")
    completion_raw = _root_file(
        root_fd, COMPLETION_BASENAME, "R2R-1 success completion"
    )
    if completion_raw != linear.canonical_json_bytes(completion):
        raise ValueError("R2R-1 completion changed before terminal commit")
    terminal_raw = _root_file(
        root_fd, terminal_name, f"R2R-1 success {terminal_name} marker"
    )
    if terminal_raw != expected_terminal_raw:
        raise ValueError("R2R-1 success terminal/completion binding changed")
    ledger = _validate_terminal_ledger(
        _strict_canonical_json(terminal_raw, "R2R-1 success terminal ledger"),
        expected_terminal="DONE",
    )
    if (
        ledger["payload_sha256"] != _sha256_bytes(completion_raw)
        or not linear._json_type_exact_equal(
            ledger["payload_identity"],
            expected_science_snapshot[COMPLETION_BASENAME]["identity"],
        )
        or not linear._json_type_exact_equal(
            ledger["exit_identity"], dict(expected_exit_identity)
        )
        or ledger["exit_sha256"] != _sha256_bytes(b"0\n")
        or ledger["artifact_snapshot_sha256"]
        != completion["precompletion_artifact_snapshot_sha256"]
    ):
        raise ValueError("R2R-1 success terminal ledger binding changed")
    fit_manifest_raw = _root_file(
        root_fd, FIT_MANIFEST_BASENAME, "R2R-1 success fit manifest"
    )
    fit_receipt_raw = _root_file(
        root_fd, FIT_RECEIPT_BASENAME, "R2R-1 success fit receipt"
    )
    arrays_raw = _root_file(root_fd, FIT_ARRAYS_BASENAME, "R2R-1 success arrays")
    if (
        _sha256_bytes(fit_manifest_raw) != completion["fit_manifest_sha256"]
        or _sha256_bytes(fit_receipt_raw) != completion["fit_receipt_sha256"]
        or _sha256_bytes(arrays_raw) != completion["fit_arrays_sha256"]
    ):
        raise ValueError("R2R-1 success artifact hash binding changed")
    fit_receipt = _strict_canonical_json(fit_receipt_raw, "R2R-1 fit receipt")
    _validate_fit_receipt_invariants(fit_receipt)
    _load_npz_bytes(arrays_raw, fit_receipt["fit_arrays"]["schema"], "R2R-1 fit arrays")
    checkpoint = completion["checkpoint"]
    if checkpoint_required:
        files = _bound_checkpoint_files(root_fd)
        if {name: _sha256_bytes(raw) for name, raw in files.items()} != checkpoint[
            "file_sha256"
        ]:
            raise ValueError("R2R-1 success checkpoint content changed")
    if _science_snapshot_fd(
        root_fd, checkpoint_required=checkpoint_required
    ) != dict(expected_science_snapshot):
        raise ValueError("R2R-1 science artifacts changed around terminal commit")
    if _root_file(root_fd, "EXIT_CODE", "R2R-1 final success EXIT_CODE") != b"0\n":
        raise ValueError("R2R-1 success EXIT_CODE changed after science validation")
    if _root_file(
        root_fd,
        terminal_name,
        f"R2R-1 final success {terminal_name} marker",
    ) != expected_terminal_raw:
        raise ValueError("R2R-1 success terminal changed after science validation")
    if _root_file(root_fd, "EXIT_CODE", "R2R-1 closed success EXIT_CODE") != b"0\n":
        raise ValueError("R2R-1 success EXIT_CODE changed during terminal closure")
    final_identities = {
        name: entry_identity(name) for name in sorted(expected_names)
    }
    if not linear._json_type_exact_equal(final_identities, initial_identities):
        raise ValueError("R2R-1 success entry identity changed during validation")
    if set(os.listdir(root_fd)) != expected_names:
        raise ValueError("R2R-1 success inventory changed after validation")


def _validate_failure_content_fd(
    root_fd: int,
    *,
    terminal_name: str,
    expected_failure_identity: Mapping[str, int],
    expected_exit_identity: Mapping[str, int],
    expected_terminal_raw: bytes,
) -> None:
    expected_names = {"failure.json", "EXIT_CODE", terminal_name}
    if set(os.listdir(root_fd)) != expected_names:
        raise ValueError("R2R-1 failure tree inventory changed before commit")
    initial_identities = {
        name: linear._owned_regular_identity(
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        )
        for name in sorted(expected_names)
    }
    if (
        not linear._json_type_exact_equal(
            initial_identities["failure.json"], dict(expected_failure_identity)
        )
        or not linear._json_type_exact_equal(
            initial_identities["EXIT_CODE"], dict(expected_exit_identity)
        )
    ):
        raise ValueError("R2R-1 failure artifact identity changed")
    raw = _root_file(root_fd, "failure.json", "R2R-1 failure receipt")
    failure = _strict_canonical_json(raw, "R2R-1 failure receipt")
    _exact_keys(failure, FAILURE_KEYS, "R2R-1 failure receipt")
    if (
        failure.get("format") != "graphene_r2r1_local_failure_v1"
        or type(failure.get("error_type")) is not str
        or not failure["error_type"]
        or type(failure.get("error_message")) is not str
        or failure.get("scientific_gate_result") is not False
        or failure.get("numerically_inconclusive") is not True
        or failure.get("fit_release_published") is not False
        or failure.get("checkpoint_published") is not False
        or failure.get("development_or_held_access") is not False
        or failure.get("energy_labels_used") is not False
        or failure.get("authorization_verified_before_transaction") is not True
    ):
        raise ValueError("R2R-1 failure receipt invariants changed")
    if _root_file(root_fd, "EXIT_CODE", "R2R-1 failure EXIT_CODE") != b"1\n":
        raise ValueError("R2R-1 failure EXIT_CODE changed")
    terminal = _root_file(
        root_fd, terminal_name, f"R2R-1 failure {terminal_name} marker"
    )
    if terminal != expected_terminal_raw:
        raise ValueError("R2R-1 FAILED/failure receipt binding changed")
    ledger = _validate_terminal_ledger(
        _strict_canonical_json(terminal, "R2R-1 failure terminal ledger"),
        expected_terminal="FAILED",
    )
    if (
        ledger["payload_sha256"] != _sha256_bytes(raw)
        or not linear._json_type_exact_equal(
            ledger["payload_identity"], dict(expected_failure_identity)
        )
        or ledger["exit_sha256"] != _sha256_bytes(b"1\n")
        or not linear._json_type_exact_equal(
            ledger["exit_identity"], dict(expected_exit_identity)
        )
        or ledger["artifact_snapshot_sha256"]
        != _artifact_snapshot_sha256(
            {"failure.json": _owned_file_snapshot(expected_failure_identity, raw)}
        )
    ):
        raise ValueError("R2R-1 failure terminal ledger binding changed")
    final_identities = {
        name: linear._owned_regular_identity(
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        )
        for name in sorted(expected_names)
    }
    if not linear._json_type_exact_equal(final_identities, initial_identities):
        raise ValueError("R2R-1 failure entry identity changed during validation")
    if _root_file(
        root_fd, "failure.json", "R2R-1 final failure receipt"
    ) != raw:
        raise ValueError("R2R-1 failure receipt bytes changed during validation")
    if _root_file(root_fd, "EXIT_CODE", "R2R-1 final failure EXIT_CODE") != b"1\n":
        raise ValueError("R2R-1 failure EXIT_CODE bytes changed during validation")
    if _root_file(
        root_fd,
        terminal_name,
        f"R2R-1 final failure {terminal_name} marker",
    ) != expected_terminal_raw:
        raise ValueError("R2R-1 failure terminal bytes changed during validation")
    if _root_file(
        root_fd, "failure.json", "R2R-1 closed failure receipt"
    ) != raw or _root_file(
        root_fd, "EXIT_CODE", "R2R-1 closed failure EXIT_CODE"
    ) != b"1\n":
        raise ValueError("R2R-1 failure bytes changed during terminal closure")
    if set(os.listdir(root_fd)) != expected_names:
        raise ValueError("R2R-1 failure inventory changed after validation")


def _commit_held_terminal_guarded(
    root_fd: int,
    running_fd: int,
    *,
    root_parent_fd: int,
    root_name: str,
    root_parent_path: Path,
    root_parent_chain: Sequence[Mapping[str, int]],
    expected_root_identity: Mapping[str, int],
    absent_sibling_name: str,
    expected_running_identity: Mapping[str, int],
    destination: str,
    marker: bytes,
    precommit_validator: Any,
    adjacent_expected_names: set[str],
    adjacent_entry_identities: Mapping[str, Mapping[str, int]],
    adjacent_recursive_snapshot: Mapping[str, Mapping[str, Any]] | None,
    postcommit_validator: Any,
    _write_guard: linear._HeldRegularWriteGuard,
) -> dict[str, Any]:
    held = os.fstat(running_fd)
    observed = os.stat("RUNNING", dir_fd=root_fd, follow_symlinks=False)
    if (
        linear._owned_regular_identity(held) != dict(expected_running_identity)
        or linear._owned_regular_identity(observed) != dict(expected_running_identity)
    ):
        raise ValueError("R2R-1 RUNNING basename changed before terminal preparation")
    initial_running_record = _capture_held_regular_descriptor(
        running_fd,
        (FORMAT + "\n").encode("ascii"),
        label="R2R-1 RUNNING before terminal rewrite",
        expected_identity=expected_running_identity,
    )
    rewritten_identity = _rewrite_held_file(running_fd, marker)
    rebound = os.stat("RUNNING", dir_fd=root_fd, follow_symlinks=False)
    current = os.fstat(running_fd)
    if (
        not linear._json_type_exact_equal(
            linear._owned_regular_identity(current), rewritten_identity
        )
        or rebound.st_dev != current.st_dev
        or rebound.st_ino != current.st_ino
    ):
        raise ValueError("R2R-1 RUNNING basename changed before terminal commit")
    prepared_running_record = _capture_held_regular_descriptor(
        running_fd,
        marker,
        label="R2R-1 RUNNING after terminal rewrite",
        expected_identity=rewritten_identity,
    )
    _verify_empty_to_payload_transition(
        initial_running_record,
        prepared_running_record,
        payload_size=len(marker),
        label="R2R-1 terminal RUNNING inode",
    )
    prepared_identity = linear._owned_regular_identity(current)
    if adjacent_recursive_snapshot is not None:
        linear._watch_owned_snapshot_fd(
            _write_guard,
            root_fd,
            adjacent_recursive_snapshot,
            label=f"R2R-1 {destination} precommit snapshot",
        )
    else:
        for name, expected_identity in sorted(adjacent_entry_identities.items()):
            _write_guard.watch_owned_at(
                root_fd,
                name,
                expected_identity=expected_identity,
                label=f"R2R-1 {destination} adjacent {name}",
            )
    _write_guard.watch_owned_descriptor(
        running_fd,
        expected_identity=prepared_identity,
        label=f"R2R-1 prepared {destination} terminal inode",
    )
    precommit_validator()
    # The validator may be long.  Rebind both the lexical parent/root and the
    # exact prepared RUNNING inode immediately before the no-replace rename.
    # An exact-content replacement inode is not accepted as provenance.
    linear._verify_directory_chain(
        root_parent_path,
        "R2R-1 terminal immediate parent binding",
        root_parent_chain,
    )
    _verify_root_entry(
        root_parent_fd,
        root_name,
        root_fd,
        label="R2R-1 terminal immediate root binding",
    )
    try:
        os.stat(
            absent_sibling_name,
            dir_fd=root_parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            "R2R-1 candidate root appeared before terminal commit"
        )
    # This short closure is owned by the commit helper itself, rather than a
    # second replaceable callback.  It runs after all long lexical checks and
    # is the final artifact operation before direct inode checks and rename.
    if set(os.listdir(root_fd)) != adjacent_expected_names:
        raise ValueError("R2R-1 terminal inventory changed after precommit readback")
    if adjacent_recursive_snapshot is not None:
        _verify_snapshot_bytes_fd(root_fd, adjacent_recursive_snapshot)
    for name, expected_identity in adjacent_entry_identities.items():
        entry = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(entry), expected_identity
        ):
            raise ValueError(
                f"R2R-1 terminal artifact identity changed after precommit: {name}"
            )
    if set(os.listdir(root_fd)) != adjacent_expected_names:
        raise ValueError("R2R-1 terminal inventory changed at rename boundary")
    parent_identity = linear._directory_binding_identity(os.fstat(root_parent_fd))
    if not linear._json_type_exact_equal(parent_identity, root_parent_chain[-1]):
        raise ValueError("R2R-1 terminal held parent identity changed")
    final_root_entry = os.stat(
        root_name, dir_fd=root_parent_fd, follow_symlinks=False
    )
    held_root = os.fstat(root_fd)
    if (
        stat.S_ISLNK(final_root_entry.st_mode)
        or not stat.S_ISDIR(final_root_entry.st_mode)
        or final_root_entry.st_dev != held_root.st_dev
        or final_root_entry.st_ino != held_root.st_ino
        or not linear._json_type_exact_equal(
            linear._directory_binding_identity(final_root_entry),
            dict(expected_root_identity),
        )
        or not linear._json_type_exact_equal(
            linear._directory_binding_identity(held_root),
            dict(expected_root_identity),
        )
    ):
        raise ValueError("R2R-1 terminal root entry changed at rename boundary")
    try:
        os.stat(
            absent_sibling_name,
            dir_fd=root_parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            "R2R-1 candidate root appeared at terminal rename boundary"
        )
    final_running_entry = os.stat(
        "RUNNING", dir_fd=root_fd, follow_symlinks=False
    )
    _capture_held_regular_descriptor(
        running_fd,
        marker,
        label="R2R-1 terminal marker at rename boundary",
        expected_identity=prepared_identity,
    )
    if (
        not linear._json_type_exact_equal(
            linear._owned_regular_identity(os.fstat(running_fd)), prepared_identity
        )
        or not linear._json_type_exact_equal(
            linear._owned_regular_identity(final_running_entry), prepared_identity
        )
    ):
        raise ValueError("R2R-1 RUNNING inode changed during terminal validation")
    try:
        _rename_noreplace_at(
            root_fd,
            "RUNNING",
            destination,
            write_guard=_write_guard,
        )
    except BaseException as error:
        # A wrapper may commit, move the held inode back to RUNNING, and then
        # raise.  Probing basenames after the exception would misclassify that
        # history as precommit and allow EXIT_CODE/artifact rollback.  Treat
        # every exception after entering the no-replace helper as
        # committed/uncertain; the release tree is henceforth read-only.
        raise _TerminalCommittedError(destination, error) from error
    # Ignore the expected RUNNING->terminal vnode rename after the helper's
    # last pre-syscall drain.
    _write_guard.close()
    try:
        committed = os.stat(destination, dir_fd=root_fd, follow_symlinks=False)
        current = os.fstat(running_fd)
        if committed.st_dev != current.st_dev or committed.st_ino != current.st_ino:
            raise ValueError("R2R-1 terminal basename differs from held marker inode")
        committed_identity = linear._owned_regular_identity(current)
        _authoritative_owned_xattrs(
            running_fd, label="R2R-1 committed terminal inode"
        )
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(committed), committed_identity
        ):
            raise ValueError("R2R-1 terminal identity changed across rename")
        os.fsync(root_fd)
        settled_terminal = os.stat(
            destination, dir_fd=root_fd, follow_symlinks=False
        )
        if (
            not linear._json_type_exact_equal(
                linear._owned_regular_identity(os.fstat(running_fd)), committed_identity
            )
            or not linear._json_type_exact_equal(
                linear._owned_regular_identity(settled_terminal), committed_identity
            )
        ):
            raise ValueError("R2R-1 committed terminal metadata did not remain stable")
        if not linear._json_type_exact_equal(
            linear._directory_binding_identity(os.fstat(root_fd)),
            dict(expected_root_identity),
        ):
            raise ValueError("R2R-1 committed root stable identity changed")
        with linear._HeldRegularWriteGuard(
            f"R2R-1 committed {destination} authority set"
        ) as committed_guard:
            if adjacent_recursive_snapshot is not None:
                linear._watch_owned_snapshot_fd(
                    committed_guard,
                    root_fd,
                    adjacent_recursive_snapshot,
                    label=f"R2R-1 committed {destination} snapshot",
                )
            committed_guard.watch_owned_descriptor(
                running_fd,
                expected_identity=committed_identity,
                label=f"R2R-1 committed {destination} terminal inode",
            )
            postcommit_validator()
            _capture_held_regular_descriptor(
                running_fd,
                marker,
                label="R2R-1 committed terminal after postcommit validation",
                expected_identity=committed_identity,
            )
            final_terminal = os.stat(
                destination, dir_fd=root_fd, follow_symlinks=False
            )
            if (
                not linear._json_type_exact_equal(
                    linear._owned_regular_identity(os.fstat(running_fd)),
                    committed_identity,
                )
                or not linear._json_type_exact_equal(
                    linear._owned_regular_identity(final_terminal),
                    committed_identity,
                )
                or not linear._json_type_exact_equal(
                    linear._directory_binding_identity(os.fstat(root_fd)),
                    dict(expected_root_identity),
                )
            ):
                raise ValueError("R2R-1 committed terminal changed after validation")
            _authoritative_owned_xattrs(
                running_fd, label="R2R-1 final committed terminal inode"
            )
        return {"identity": committed_identity}
    except BaseException as error:
        raise _TerminalCommittedError(destination, error) from error


def _commit_held_terminal(
    root_fd: int,
    running_fd: int,
    *,
    root_parent_fd: int,
    root_name: str,
    root_parent_path: Path,
    root_parent_chain: Sequence[Mapping[str, int]],
    expected_root_identity: Mapping[str, int],
    absent_sibling_name: str,
    expected_running_identity: Mapping[str, int],
    destination: str,
    marker: bytes,
    precommit_validator: Any,
    adjacent_expected_names: set[str],
    adjacent_entry_identities: Mapping[str, Mapping[str, int]],
    adjacent_recursive_snapshot: Mapping[str, Mapping[str, Any]] | None,
    postcommit_validator: Any,
) -> dict[str, Any]:
    """Guard every payload vnode through the terminal no-replace boundary."""
    with linear._HeldRegularWriteGuard(
        f"R2R-1 {destination} precommit authority set"
    ) as write_guard:
        return _commit_held_terminal_guarded(
            root_fd,
            running_fd,
            root_parent_fd=root_parent_fd,
            root_name=root_name,
            root_parent_path=root_parent_path,
            root_parent_chain=root_parent_chain,
            expected_root_identity=expected_root_identity,
            absent_sibling_name=absent_sibling_name,
            expected_running_identity=expected_running_identity,
            destination=destination,
            marker=marker,
            precommit_validator=precommit_validator,
            adjacent_expected_names=adjacent_expected_names,
            adjacent_entry_identities=adjacent_entry_identities,
            adjacent_recursive_snapshot=adjacent_recursive_snapshot,
            postcommit_validator=postcommit_validator,
            _write_guard=write_guard,
        )


def _finish_success(
    root_fd: int,
    running_fd: int,
    running_identity: Mapping[str, int],
    completion: Mapping[str, Any],
    expected_science_snapshot: Mapping[str, Any],
    *,
    root_parent_fd: int,
    root_name: str,
    root_parent_path: Path,
    root_parent_chain: Sequence[Mapping[str, int]],
    expected_root_identity: Mapping[str, int],
    absent_sibling_name: str,
) -> dict[str, Any]:
    completion_raw = _root_file(root_fd, COMPLETION_BASENAME, "R2R-1 completion")
    if completion_raw != linear.canonical_json_bytes(completion):
        raise ValueError("R2R-1 prevalidated completion changed before terminal")
    completion_sha = _sha256_bytes(completion_raw)
    exit_identity = _atomic_write_bytes_at(root_fd, "EXIT_CODE", b"0\n")
    checkpoint_required = completion["status"] == CHECKPOINT_STATUS_SOURCE
    if not linear._json_type_exact_equal(
        _science_snapshot_fd(root_fd, checkpoint_required=checkpoint_required),
        dict(expected_science_snapshot),
    ):
        raise ValueError("R2R-1 science changed before terminal preparation")
    if not linear._json_type_exact_equal(
        linear._owned_regular_identity(os.fstat(running_fd)), dict(running_identity)
    ):
        raise ValueError("R2R-1 RUNNING changed before success terminal preparation")
    _capture_held_regular_descriptor(
        running_fd,
        (FORMAT + "\n").encode("ascii"),
        label="R2R-1 success RUNNING ledger provenance",
        expected_identity=running_identity,
    )
    completion_item = expected_science_snapshot[COMPLETION_BASENAME]
    marker_payload = _terminal_ledger_payload(
        terminal="DONE",
        status=str(completion["status"]),
        payload_basename=COMPLETION_BASENAME,
        payload_raw=completion_raw,
        payload_identity=completion_item["identity"],
        exit_raw=b"0\n",
        exit_identity=exit_identity,
        root_binding_identity=expected_root_identity,
        artifact_snapshot_sha256=completion[
            "precompletion_artifact_snapshot_sha256"
        ],
        terminal_inode_identity=running_identity,
    )
    marker = linear.canonical_json_bytes(marker_payload)
    if len(marker) > ROOT_FILE_SIZE_LIMITS["DONE"]:
        raise ValueError("R2R-1 DONE ledger exceeds its frozen size limit")

    try:
        committed_done_record = _commit_held_terminal(
            root_fd,
            running_fd,
            root_parent_fd=root_parent_fd,
            root_name=root_name,
            root_parent_path=root_parent_path,
            root_parent_chain=root_parent_chain,
            expected_root_identity=expected_root_identity,
            absent_sibling_name=absent_sibling_name,
            expected_running_identity=running_identity,
            destination="DONE",
            marker=marker,
            precommit_validator=lambda: _validate_success_content_fd(
                root_fd,
                completion,
                expected_science_snapshot,
                terminal_name="RUNNING",
                expected_exit_identity=exit_identity,
                expected_terminal_raw=marker,
            ),
            adjacent_expected_names=_science_names(
                completion["status"] == CHECKPOINT_STATUS_SOURCE
            )
            | {"EXIT_CODE", "RUNNING"},
            adjacent_entry_identities={"EXIT_CODE": exit_identity},
            adjacent_recursive_snapshot={
                **expected_science_snapshot,
                "EXIT_CODE": _owned_file_snapshot(exit_identity, b"0\n"),
            },
            postcommit_validator=lambda: _validate_success_content_fd(
                root_fd,
                completion,
                expected_science_snapshot,
                terminal_name="DONE",
                expected_exit_identity=exit_identity,
                expected_terminal_raw=marker,
            ),
        )
    except _TerminalCommittedError:
        raise
    except BaseException:
        try:
            observed = os.stat("EXIT_CODE", dir_fd=root_fd, follow_symlinks=False)
            if linear._owned_regular_identity(observed) == exit_identity:
                os.unlink("EXIT_CODE", dir_fd=root_fd)
        except FileNotFoundError:
            pass
        raise
    return {
        "exit_identity": exit_identity,
        "done_identity": committed_done_record["identity"],
        "done_sha256": _sha256_bytes(marker),
        "done_raw": marker,
    }


def _finish_failure(
    root_fd: int,
    running_fd: int,
    running_identity: Mapping[str, int],
    error: BaseException,
    *,
    root_parent_fd: int,
    root_name: str,
    root_parent_path: Path,
    root_parent_chain: Sequence[Mapping[str, int]],
    expected_root_identity: Mapping[str, int],
    absent_sibling_name: str,
) -> None:
    """Commit FAILED only while the transaction still contains RUNNING alone.

    Any error after science-artifact materialization is an auditable partial
    transaction, not something this function destructively cleans.  This rule
    makes an injected/unknown entry impossible to traverse, open, or delete.
    """
    names = set(os.listdir(root_fd))
    if names != {"RUNNING"}:
        raise ValueError(
            "R2R-1 post-artifact/foreign-entry failure remains RUNNING partial"
        )
    observed_running = os.stat("RUNNING", dir_fd=root_fd, follow_symlinks=False)
    held_running = os.fstat(running_fd)
    if (
        stat.S_ISLNK(observed_running.st_mode)
        or not stat.S_ISREG(observed_running.st_mode)
        or observed_running.st_nlink != 1
        or linear._owned_regular_identity(observed_running) != dict(running_identity)
        or linear._owned_regular_identity(held_running) != dict(running_identity)
    ):
        raise ValueError("R2R-1 held RUNNING identity changed before FAILED")
    failure = {
        "format": "graphene_r2r1_local_failure_v1",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "scientific_gate_result": False,
        "numerically_inconclusive": True,
        "fit_release_published": False,
        "checkpoint_published": False,
        "development_or_held_access": False,
        "energy_labels_used": False,
        "authorization_verified_before_transaction": True,
    }
    failure_raw = linear.canonical_json_bytes(failure)
    if len(failure_raw) > ROOT_FILE_SIZE_LIMITS["failure.json"]:
        raise ValueError("R2R-1 failure receipt exceeds its frozen size limit")
    failure_identity = _atomic_write_bytes_at(root_fd, "failure.json", failure_raw)
    raw = _root_file(root_fd, "failure.json", "R2R-1 failure receipt")
    exit_identity = _atomic_write_bytes_at(root_fd, "EXIT_CODE", b"1\n")
    if not linear._json_type_exact_equal(
        linear._owned_regular_identity(os.fstat(running_fd)), dict(running_identity)
    ):
        raise ValueError("R2R-1 RUNNING changed before failure terminal freeze")
    _capture_held_regular_descriptor(
        running_fd,
        (FORMAT + "\n").encode("ascii"),
        label="R2R-1 failure RUNNING ledger provenance",
        expected_identity=running_identity,
    )
    failure_snapshot = {
        "failure.json": _owned_file_snapshot(failure_identity, raw)
    }
    marker_payload = _terminal_ledger_payload(
        terminal="FAILED",
        status="R2R1_NUMERICAL_INCONCLUSIVE",
        payload_basename="failure.json",
        payload_raw=raw,
        payload_identity=failure_identity,
        exit_raw=b"1\n",
        exit_identity=exit_identity,
        root_binding_identity=expected_root_identity,
        artifact_snapshot_sha256=_artifact_snapshot_sha256(failure_snapshot),
        terminal_inode_identity=running_identity,
    )
    marker = linear.canonical_json_bytes(marker_payload)
    if len(marker) > ROOT_FILE_SIZE_LIMITS["FAILED"]:
        raise ValueError("R2R-1 FAILED ledger exceeds its frozen size limit")

    _commit_held_terminal(
        root_fd,
        running_fd,
        root_parent_fd=root_parent_fd,
        root_name=root_name,
        root_parent_path=root_parent_path,
        root_parent_chain=root_parent_chain,
        expected_root_identity=expected_root_identity,
        absent_sibling_name=absent_sibling_name,
        expected_running_identity=running_identity,
        destination="FAILED",
        marker=marker,
        precommit_validator=lambda: _validate_failure_content_fd(
            root_fd,
            terminal_name="RUNNING",
            expected_failure_identity=failure_identity,
            expected_exit_identity=exit_identity,
            expected_terminal_raw=marker,
        ),
        adjacent_expected_names={"failure.json", "EXIT_CODE", "RUNNING"},
        adjacent_entry_identities={
            "failure.json": failure_identity,
            "EXIT_CODE": exit_identity,
        },
        adjacent_recursive_snapshot={
            **failure_snapshot,
            "EXIT_CODE": _owned_file_snapshot(exit_identity, b"1\n"),
        },
        postcommit_validator=lambda: _validate_failure_content_fd(
            root_fd,
            terminal_name="FAILED",
            expected_failure_identity=failure_identity,
            expected_exit_identity=exit_identity,
            expected_terminal_raw=marker,
        ),
    )


def _materialize_authorized_fit_impl(
    *,
    output_root: Path,
    attempt3_root: Path,
    thermal92_path: Path,
    freeze_manifest: Path,
    authorization_marker: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the sole authorized local fit in one exclusively reserved root.

    Authorization is checked before the output root exists.  After that check,
    the canonical basename itself is created with mkdirat-style exclusive
    semantics and its parent/root descriptors remain held until the terminal
    transition.  The canonical root is the only transaction directory.
    """
    path_contract = _validate_production_path_contract(
        output_root=output_root,
        attempt3_root=attempt3_root,
        thermal92_path=thermal92_path,
        freeze_manifest=freeze_manifest,
        authorization_marker=authorization_marker,
    )
    final_root = Path(path_contract["output_root"])
    candidate_root = linear._lexical_path(
        linear.RECOMMENDED_CANDIDATE_ROOT,
        "R2R-1 fit-stage candidate root",
        forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if candidate_root.parent != final_root.parent:
        raise PermissionError("R2R-1 fit and candidate roots lost their shared parent")
    _parent, output_parent_fd, output_parent_chain = linear._open_directory_chain(
        final_root.parent, "R2R-1 held fit-output parent"
    )
    del _parent
    try:
        try:
            os.stat(final_root.name, dir_fd=output_parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("R2R-1 output root must be fresh")
        try:
            os.stat(
                candidate_root.name,
                dir_fd=output_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                "R2R-1 candidate root must be absent before authorization"
            )
        # Pre-authorization failures leave the canonical fit root absent.  The
        # output parent remains held across authorization, so a phase-boundary
        # path swap cannot redirect the subsequent mkdirat.
        initial_authorization = linear.validate_execution_authorization(
            freeze_manifest,
            authorization_marker,
            require_fresh_result_children=True,
        )
        fit_parent_binding = initial_authorization["result_parent_binding"]
        if (
            not linear._json_type_exact_equal(
                [dict(item) for item in output_parent_chain],
                fit_parent_binding["chain"],
            )
            or not linear._json_type_exact_equal(
                linear._directory_binding_identity(os.fstat(output_parent_fd)),
                fit_parent_binding["directory_identity"],
            )
            or final_root.name != fit_parent_binding["fit_output_child_basename"]
            or candidate_root.name
            != fit_parent_binding["candidate_child_basename"]
        ):
            raise PermissionError(
                "R2R-1 fit-output parent differs from marker-bound identity"
            )
        try:
            os.stat(
                candidate_root.name,
                dir_fd=output_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                "R2R-1 candidate root appeared before fit reservation"
            )
        root, root_fd, root_chain = _reserve_canonical_root(
            final_root,
            parent_fd=output_parent_fd,
            parent_chain=output_parent_chain,
        )
    except BaseException:
        os.close(output_parent_fd)
        raise
    running_fd: int | None = None
    running_identity: dict[str, int] | None = None
    terminal_committed = False
    try:
        running_fd, running_identity = _create_running_marker(root_fd)
        running_raw = (FORMAT + "\n").encode("ascii")
        if (
            not linear._json_type_exact_equal(
                linear._owned_regular_identity(os.fstat(running_fd)), running_identity
            )
            or not linear._json_type_exact_equal(
                linear._owned_regular_identity(
                    os.stat("RUNNING", dir_fd=root_fd, follow_symlinks=False)
                ),
                running_identity,
            )
        ):
            raise ValueError("R2R-1 RUNNING changed after prepublication settlement")
        if set(os.listdir(root_fd)) != {"RUNNING"}:
            raise ValueError("R2R-1 initial owned-root inventory changed")
        _verify_active_fit_publication_pair(
            output_parent_fd,
            final_root.name,
            root_fd,
            candidate_root.name,
            label="R2R-1 publication pair before fit pipeline",
            expected_parent_identity=output_parent_chain[-1],
            expected_root_identity=root_chain[-1],
            running_fd=running_fd,
            expected_running_identity=running_identity,
            expected_marker_raw=running_raw,
            expected_root_names={"RUNNING"},
            marker_name="RUNNING",
        )
        fit_publication_boundary = linear._issue_fit_publication_boundary(
            release_parent_fd=output_parent_fd,
            release_parent_path=final_root.parent,
            release_parent_chain=output_parent_chain,
            root_fd=root_fd,
            root_path=root,
            expected_root_identity=root_chain[-1],
            expected_root_snapshot={
                "RUNNING": _owned_file_snapshot(running_identity, running_raw)
            },
            candidate_parent_fd=output_parent_fd,
            candidate_parent_path=candidate_root.parent,
            candidate_parent_chain=output_parent_chain,
            candidate_name=candidate_root.name,
            candidate_state="absent",
        )
        pipeline_receipt, arrays = linear.authorized_fit_pipeline(
            attempt3_root=attempt3_root,
            thermal92_path=thermal92_path,
            freeze_manifest=freeze_manifest,
            authorization_marker=authorization_marker,
            publication_boundary=fit_publication_boundary,
        )
        _verify_active_fit_publication_pair(
            output_parent_fd,
            final_root.name,
            root_fd,
            candidate_root.name,
            label="R2R-1 publication pair after fit pipeline",
            expected_parent_identity=output_parent_chain[-1],
            expected_root_identity=root_chain[-1],
            running_fd=running_fd,
            expected_running_identity=running_identity,
            expected_marker_raw=running_raw,
            expected_root_names={"RUNNING"},
            marker_name="RUNNING",
        )
        final_authorization = linear.validate_execution_authorization(
            freeze_manifest, authorization_marker
        )
        if not linear._json_type_exact_equal(
            final_authorization, initial_authorization
        ):
            raise PermissionError("R2R-1 authorization changed around fit")
        status = pipeline_receipt.get("status")
        if status not in SCIENTIFIC_STATUSES:
            raise ValueError("R2R-1 fit returned an unknown scientific status")

        manifest_raw, source_manifest_identity = _read_owned(
            freeze_manifest,
            "R2R-1 fit manifest source",
            expected_parent_chain=tuple(
                initial_authorization["freeze_manifest_parent_chain"]
            ),
            expected_file_identity=initial_authorization[
                "freeze_manifest_file_identity"
            ],
            size_limit=FREEZE_MANIFEST_BYTES_LIMIT,
        )
        manifest_sha = _sha256_bytes(manifest_raw)
        if (
                manifest_sha != initial_authorization["freeze_manifest_sha256"]
                or not linear._json_type_exact_equal(
                source_manifest_identity,
                initial_authorization["freeze_manifest_file_identity"],
            )
        ):
            raise PermissionError("R2R-1 fit manifest changed before materialization")
        fit_manifest_identity = _atomic_write_bytes_at(
            root_fd, FIT_MANIFEST_BASENAME, manifest_raw
        )

        normalized_arrays = {
            name: np.asarray(value, dtype="<f8") for name, value in arrays.items()
        }
        array_schema = _array_schema(normalized_arrays)
        _validate_fit_array_schema_for_status(status, array_schema)
        fit_arrays_identity = _atomic_write_npz_at(
            root_fd, FIT_ARRAYS_BASENAME, normalized_arrays
        )
        arrays_raw = _root_file(root_fd, FIT_ARRAYS_BASENAME, "R2R-1 fit arrays")
        arrays_sha = _sha256_bytes(arrays_raw)
        _load_npz_bytes(arrays_raw, array_schema, "R2R-1 fit arrays")

        checkpoint_required = status == CHECKPOINT_STATUS_SOURCE
        fit_receipt = {
            "format": FIT_RECEIPT_FORMAT,
            "status": status,
            "fit_manifest": {
                "basename": FIT_MANIFEST_BASENAME,
                "sha256": manifest_sha,
            },
            "fit_arrays": {
                "basename": FIT_ARRAYS_BASENAME,
                "sha256": arrays_sha,
                "schema": array_schema,
            },
            "pipeline_receipt": pipeline_receipt,
            "pipeline_recursive_schema_sha256": _recursive_schema_sha256(
                pipeline_receipt
            ),
            "runtime_environment": linear.runtime_environment_receipt(),
            "checkpoint_required": checkpoint_required,
            "checkpoint_pending_in_this_immutable_receipt": checkpoint_required,
            "mechanics_pending": checkpoint_required,
            "energy_labels_used": False,
            "development_or_held_access": False,
        }
        fit_receipt_identity = _atomic_write_json_at(
            root_fd, FIT_RECEIPT_BASENAME, fit_receipt
        )
        fit_receipt_raw = _root_file(
            root_fd, FIT_RECEIPT_BASENAME, "R2R-1 fit receipt"
        )
        fit_receipt_sha = _sha256_bytes(fit_receipt_raw)

        checkpoint: dict[str, Any] | None = None
        checkpoint_snapshot: dict[str, dict[str, Any]] = {}
        if checkpoint_required:
            if set(normalized_arrays).isdisjoint({"final_coefficients"}):
                raise ValueError("R2R-1 passing fit omitted final coefficients")
            checkpoint, checkpoint_snapshot = _write_checkpoint(
                normalized_arrays["final_coefficients"],
                fit_manifest_sha256=manifest_sha,
                fit_receipt_sha256=fit_receipt_sha,
                release_root_fd=root_fd,
            )
        precompletion_artifact_snapshot: dict[str, dict[str, Any]] = {
            FIT_MANIFEST_BASENAME: _owned_file_snapshot(
                fit_manifest_identity, manifest_raw
            ),
            FIT_ARRAYS_BASENAME: _owned_file_snapshot(
                fit_arrays_identity, arrays_raw
            ),
            FIT_RECEIPT_BASENAME: _owned_file_snapshot(
                fit_receipt_identity, fit_receipt_raw
            ),
            **checkpoint_snapshot,
        }
        _validate_persisted_artifact_snapshot(
            precompletion_artifact_snapshot,
            status=status,
            include_completion=False,
            label="R2R-1 owned precompletion artifact snapshot",
        )
        precompletion_artifact_snapshot_sha256 = _artifact_snapshot_sha256(
            precompletion_artifact_snapshot
        )
        completion_without_fingerprint = {
            "format": COMPLETION_FORMAT,
            "status": status,
            "execution_exit_code": 0,
            "scientific_gate_failure_is_terminal_success": status
            in {"R2R1_CONDITIONAL_OOF_FAILED", "R2R1_FINAL_READOUT_FAILED"},
            "fit_manifest_sha256": manifest_sha,
            "fit_arrays_sha256": arrays_sha,
            "fit_receipt_sha256": fit_receipt_sha,
            "checkpoint": checkpoint,
            "checkpoint_published": checkpoint is not None,
            "mechanics_completed": False,
            "mechanics_status": "PENDING" if checkpoint is not None else "NOT_RUN_STOPPED",
            "development_or_held_access": False,
            "energy_labels_used": False,
            "release_root_binding_identity": dict(root_chain[-1]),
            "precompletion_artifact_snapshot": precompletion_artifact_snapshot,
            "precompletion_artifact_snapshot_sha256": (
                precompletion_artifact_snapshot_sha256
            ),
        }
        completion = {
            **completion_without_fingerprint,
            "recursive_schema_sha256": _recursive_schema_sha256(
                completion_without_fingerprint
            ),
        }
        _validate_fit_receipt_invariants(fit_receipt)
        _validate_completion_invariants(completion)
        completion_raw = linear.canonical_json_bytes(completion)
        completion_identity = _atomic_write_bytes_at(
            root_fd, COMPLETION_BASENAME, completion_raw
        )
        expected_science_snapshot: dict[str, dict[str, Any]] = {
            **precompletion_artifact_snapshot,
            COMPLETION_BASENAME: _owned_file_snapshot(
                completion_identity, completion_raw
            ),
        }
        observed_science_snapshot = _science_snapshot_fd(
            root_fd, checkpoint_required=checkpoint_required
        )
        if not linear._json_type_exact_equal(
            observed_science_snapshot, expected_science_snapshot
        ):
            raise ValueError(
                "R2R-1 science artifact identity differs from atomic creation"
            )
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(os.fstat(running_fd)), running_identity
        ):
            raise ValueError("R2R-1 RUNNING changed before scientific readback")
        completion_sha = _sha256_bytes(linear.canonical_json_bytes(completion))
        materialization = {
            **completion,
            "completion_sha256": completion_sha,
            "fit_receipt_sha256": fit_receipt_sha,
        }
        internal_payload = _release_manifest_payload_for_bound_root(
            root=root, materialization_receipt=materialization
        )
        internal_raw = linear.canonical_json_bytes(internal_payload)
        internal_anchor = {"sha256": _sha256_bytes(internal_raw)}
        expected_prepublish_snapshot = {
            **expected_science_snapshot,
            "RUNNING": _owned_file_snapshot(running_identity, running_raw),
        }
        internal_publication_boundary = _PreSolvePublicationBoundary(
            release_parent_fd=output_parent_fd,
            release_parent_path=final_root.parent,
            release_parent_chain=output_parent_chain,
            candidate_parent_fd=output_parent_fd,
            candidate_parent_path=final_root.parent,
            candidate_parent_chain=output_parent_chain,
            candidate_name=candidate_root.name,
            candidate_state="absent",
        )
        internal_recovery = _recover_fit_release_bound(
            root=root,
            root_fd=root_fd,
            root_chain=root_chain,
            output_root=root,
            attempt3_root=attempt3_root,
            thermal92_path=thermal92_path,
            freeze_manifest=freeze_manifest,
            authorization_marker=authorization_marker,
            release_manifest_raw=internal_raw,
            expected_release_manifest_sha256=internal_anchor["sha256"],
            require_terminal=False,
            publication_boundary=internal_publication_boundary,
            expected_prepublish_snapshot=expected_prepublish_snapshot,
        )
        if internal_recovery.get("scientific_recomputed") is not True:
            raise ValueError("R2R-1 reserved release did not pass full readback/re-solve")
        if _science_snapshot_fd(
            root_fd, checkpoint_required=checkpoint_required
        ) != expected_science_snapshot:
            raise ValueError("R2R-1 science baseline changed after first readback")
        os.fsync(root_fd)
        final_readback = _recover_fit_release_bound(
            root=root,
            root_fd=root_fd,
            root_chain=root_chain,
            output_root=root,
            attempt3_root=attempt3_root,
            thermal92_path=thermal92_path,
            freeze_manifest=freeze_manifest,
            authorization_marker=authorization_marker,
            release_manifest_raw=internal_raw,
            expected_release_manifest_sha256=internal_anchor["sha256"],
            require_terminal=False,
            publication_boundary=internal_publication_boundary,
            expected_prepublish_snapshot=expected_prepublish_snapshot,
        )
        if final_readback.get("scientific_recomputed") is not True:
            raise ValueError("R2R-1 final reserved-root readback changed")
        if _science_snapshot_fd(
            root_fd, checkpoint_required=checkpoint_required
        ) != expected_science_snapshot:
            raise ValueError("R2R-1 science baseline changed after final readback")
        _verify_root_entry(
            output_parent_fd,
            final_root.name,
            root_fd,
            label="R2R-1 root before terminal commit",
        )
        linear._verify_directory_chain(
            final_root.parent,
            "R2R-1 publication parent before terminal commit",
            output_parent_chain,
        )
        terminal_receipt = _finish_success(
            root_fd,
            running_fd,
            running_identity,
            completion,
            expected_science_snapshot,
            root_parent_fd=output_parent_fd,
            root_name=final_root.name,
            root_parent_path=final_root.parent,
            root_parent_chain=output_parent_chain,
            expected_root_identity=root_chain[-1],
            absent_sibling_name=candidate_root.name,
        )
        committed_exit_identity = terminal_receipt["exit_identity"]
        committed_done_identity = terminal_receipt["done_identity"]
        committed_done_raw = terminal_receipt["done_raw"]
        expected_release_snapshot = {
            **expected_science_snapshot,
            "EXIT_CODE": _owned_file_snapshot(committed_exit_identity, b"0\n"),
            "DONE": _owned_file_snapshot(
                committed_done_identity, committed_done_raw
            ),
        }
        terminal_committed = True
        _verify_root_entry(
            output_parent_fd,
            final_root.name,
            root_fd,
            label="R2R-1 committed root",
        )
        os.fsync(output_parent_fd)
        linear._verify_directory_chain(
            final_root.parent,
            "R2R-1 publication parent after terminal commit",
            output_parent_chain,
        )
        # DONE is already the irreversible terminal.  Revalidate every
        # artifact immediately before returning; a late fault is reported but
        # must never downgrade or rewrite the committed release.
        _validate_success_content_fd(
            root_fd,
            completion,
            expected_science_snapshot,
            terminal_name="DONE",
            expected_exit_identity=committed_exit_identity,
            expected_terminal_raw=committed_done_raw,
        )
        _verify_root_entry(
            output_parent_fd,
            final_root.name,
            root_fd,
            label="R2R-1 root at materializer return",
        )
        linear._verify_directory_chain(
            final_root.parent,
            "R2R-1 publication parent at materializer return",
            output_parent_chain,
        )
        expected_return_names = _science_names(checkpoint_required) | {
            "EXIT_CODE",
            "DONE",
        }
        _verify_active_fit_publication_pair(
            output_parent_fd,
            final_root.name,
            root_fd,
            candidate_root.name,
            label="R2R-1 publication pair at materializer return",
            expected_parent_identity=output_parent_chain[-1],
            expected_root_identity=root_chain[-1],
            running_fd=running_fd,
            expected_running_identity=linear._owned_regular_identity(os.fstat(running_fd)),
            expected_marker_raw=committed_done_raw,
            expected_root_names=expected_return_names,
            marker_name="DONE",
        )
        # This complete release pass follows every long validator/parent/pair
        # operation.  With owned8 identities, EXIT/DONE and science bytes must
        # all be closed by raw/SHA rather than by a final stat-only loop.
        _verify_snapshot_bytes_fd(root_fd, expected_release_snapshot)
        if set(os.listdir(root_fd)) != expected_return_names:
            raise ValueError("R2R-1 committed release inventory changed at return")
        for name, expected_identity in (
            ("EXIT_CODE", committed_exit_identity),
            ("DONE", committed_done_identity),
        ):
            observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if not linear._json_type_exact_equal(
                linear._owned_regular_identity(observed), expected_identity
            ):
                raise ValueError(f"R2R-1 committed {name} identity changed at return")
        result_payload = {
            **completion,
            "completion_sha256": completion_sha,
            "newly_materialized": True,
            "canonical_root_reserved_before_fit": True,
            "terminal_committed_by_RUNNING_inode_rename": True,
            "done_sha256": terminal_receipt["done_sha256"],
            "done_identity": committed_done_identity,
            "exit_identity": committed_exit_identity,
            "expected_release_snapshot": expected_release_snapshot,
            "reserved_root_full_readback_and_scientific_resolve_passed": True,
            "candidate_release_pins_not_independent_authorization": {
                "expected_completion_sha256": _sha256_bytes(
                    linear.canonical_json_bytes(completion)
                ),
                "expected_fit_receipt_sha256": fit_receipt_sha,
                "expected_checkpoint_receipt_sha256": (
                    checkpoint["receipt_sha256"] if checkpoint is not None else None
                ),
            },
        }
        witness_payload = {
            "release_root": str(root),
            "status": str(completion["status"]),
            "done_sha256": terminal_receipt["done_sha256"],
            "done_identity": dict(committed_done_identity),
            "expected_release_snapshot": expected_release_snapshot,
        }
        return result_payload, witness_payload
    except Exception as error:
        if isinstance(error, _TerminalCommittedError) or terminal_committed:
            # A terminal-name rename is the point of no return.  Any later
            # error is reported without changing DONE/FAILED or its artifacts.
            raise
        if running_fd is None or running_identity is None:
            # mkdirat succeeded but RUNNING creation did not.  Preserve the
            # canonical empty/partial directory for explicit incomplete-run
            # diagnosis; it is never treated as fresh on a retry.
            raise
        try:
            _finish_failure(
                root_fd,
                running_fd,
                running_identity,
                error,
                root_parent_fd=output_parent_fd,
                root_name=final_root.name,
                root_parent_path=final_root.parent,
                root_parent_chain=output_parent_chain,
                expected_root_identity=root_chain[-1],
                absent_sibling_name=candidate_root.name,
            )
        except _TerminalCommittedError:
            # FAILED is already committed; propagate the original execution
            # failure, never rewrite the canonical terminal tree.
            raise error
        except BaseException as failure_error:
            # Failure conversion itself was not safely committable.  RUNNING
            # remains the sole recognized state (or a tamper-visible partial
            # state); no destructive cleanup or automatic refit is attempted.
            raise error from failure_error
        raise
    finally:
        if running_fd is not None:
            os.close(running_fd)
        os.close(root_fd)
        os.close(output_parent_fd)


(
    materialize_authorized_fit,
    _require_materialization_witness,
) = _build_materialization_witness_boundary(_materialize_authorized_fit_impl)


def inspect_release_state(output_root: Path) -> dict[str, Any]:
    """Classify the canonical output without parsing labels or fitting."""
    root = linear._lexical_path(
        output_root,
        "R2R-1 release-state root",
        forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
    )
    expected = linear._lexical_path(
        linear.RECOMMENDED_FIT_OUTPUT_ROOT,
        "R2R-1 frozen release-state root",
        forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if root != expected:
        raise PermissionError("R2R-1 release-state root differs from frozen path")
    parent, parent_fd, parent_chain = linear._open_directory_chain(
        root.parent, "R2R-1 release-state parent"
    )
    del parent
    root_fd: int | None = None
    try:
        try:
            before = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return {"state": "ABSENT", "fit_may_start": True}
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise FileExistsError("R2R-1 output basename is a non-directory collision")
        root_fd = os.open(
            root.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        if (
            linear._directory_binding_identity(os.fstat(root_fd))
            != linear._directory_binding_identity(before)
        ):
            raise ValueError("R2R-1 release root changed across stat/open")
        names = set(os.listdir(root_fd))
        terminals = names & {"RUNNING", "DONE", "FAILED"}
        if terminals == {"DONE"}:
            state = "DONE"
        elif terminals == {"FAILED"}:
            state = "FAILED"
        elif terminals == {"RUNNING"}:
            state = "RUNNING_INCOMPLETE"
        elif not terminals:
            state = "EMPTY_OR_PARTIAL_INCOMPLETE"
        else:
            state = "CONFLICTING_TERMINALS_INCOMPLETE"
        _verify_root_entry(
            parent_fd, root.name, root_fd, label="R2R-1 classified release root"
        )
        linear._verify_directory_chain(
            root.parent, "R2R-1 release-state final parent", parent_chain
        )
        return {
            "state": state,
            "fit_may_start": False,
            "terminal_names": sorted(terminals),
            "root_entry_names": sorted(names),
        }
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _inspect_fresh_publication_pair(output_root: Path) -> dict[str, Any]:
    """Classify fit/candidate basenames from one label-blind held parent fd."""
    output = _exact_path(
        output_root,
        linear.RECOMMENDED_FIT_OUTPUT_ROOT,
        "R2R-1 fresh-pair fit output root",
        must_exist=False,
    )
    candidate = linear._lexical_path(
        linear.RECOMMENDED_CANDIDATE_ROOT,
        "R2R-1 fresh-pair candidate root",
        forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if output.parent != candidate.parent:
        raise PermissionError("R2R-1 publication pair has different parents")
    parent, parent_fd, parent_chain = linear._open_directory_chain(
        output.parent, "R2R-1 fresh publication pair parent"
    )
    try:
        states: dict[str, str] = {}
        for key, name in (
            ("fit_output", output.name),
            ("candidate", candidate.name),
        ):
            try:
                observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                states[key] = "ABSENT"
            else:
                if stat.S_ISLNK(observed.st_mode):
                    states[key] = "SYMLINK"
                elif stat.S_ISDIR(observed.st_mode):
                    states[key] = "DIRECTORY"
                elif stat.S_ISREG(observed.st_mode):
                    states[key] = "FILE"
                else:
                    states[key] = "SPECIAL"
        linear._verify_directory_chain(
            parent, "R2R-1 fresh publication pair final parent", parent_chain
        )
        return {
            **states,
            "both_absent": all(value == "ABSENT" for value in states.values()),
        }
    finally:
        os.close(parent_fd)


@dataclass(frozen=True)
class _PreSolvePublicationBoundary:
    release_parent_fd: int
    release_parent_path: Path
    release_parent_chain: Sequence[Mapping[str, int]]
    candidate_parent_fd: int
    candidate_parent_path: Path
    candidate_parent_chain: Sequence[Mapping[str, int]]
    candidate_name: str
    candidate_state: str
    candidate_fd: int | None = None
    candidate_path: Path | None = None
    candidate_chain: Sequence[Mapping[str, int]] | None = None
    anchor_name: str | None = None
    anchor_identity: Mapping[str, int] | None = None
    anchor_raw: bytes | None = None


def _verify_pre_solve_publication_boundary(
    *,
    root: Path,
    root_fd: int,
    expected_root_identity: Mapping[str, int],
    expected_release_snapshot: Mapping[str, Mapping[str, Any]],
    boundary: _PreSolvePublicationBoundary,
) -> None:
    """Helper-owned canonical closure immediately before label/SVD work."""
    if boundary.candidate_state not in {"absent", "existing"}:
        raise ValueError("R2R-1 pre-solve candidate state is invalid")
    if boundary.candidate_state == "existing":
        if (
            boundary.candidate_fd is None
            or boundary.candidate_path is None
            or boundary.candidate_chain is None
            or boundary.anchor_name is None
            or boundary.anchor_identity is None
            or boundary.anchor_raw is None
        ):
            raise ValueError("R2R-1 existing candidate boundary is incomplete")
        observed_anchor = _read_regular_file_at(
            boundary.candidate_fd,
            boundary.anchor_name,
            "R2R-1 candidate anchor immediately before scientific solve",
            size_limit=2 * 1024 * 1024,
            expected_identity=boundary.anchor_identity,
        )
        if observed_anchor != boundary.anchor_raw:
            raise ValueError("R2R-1 candidate anchor bytes changed before solve")
    elif any(
        value is not None
        for value in (
            boundary.candidate_fd,
            boundary.candidate_path,
            boundary.candidate_chain,
            boundary.anchor_name,
            boundary.anchor_identity,
            boundary.anchor_raw,
        )
    ):
        raise ValueError("R2R-1 absent candidate boundary carried existing state")

    # Complete all potentially longer lexical walks before the final held-fd
    # inventory/identity checks.  No payload operation follows these checks
    # before authorized_fit_pipeline is entered.
    linear._verify_directory_chain(
        boundary.release_parent_path,
        "R2R-1 pre-solve release-parent binding",
        boundary.release_parent_chain,
    )
    linear._verify_directory_chain(
        boundary.candidate_parent_path,
        "R2R-1 pre-solve candidate-parent binding",
        boundary.candidate_parent_chain,
    )
    if boundary.candidate_state == "existing":
        assert boundary.candidate_path is not None
        assert boundary.candidate_chain is not None
        linear._verify_directory_chain(
            boundary.candidate_path,
            "R2R-1 pre-solve existing candidate binding",
            boundary.candidate_chain,
        )

    _verify_snapshot_bytes_fd(root_fd, expected_release_snapshot)
    if set(os.listdir(root_fd)) != {
        name for name in expected_release_snapshot if "/" not in name
    }:
        raise ValueError("R2R-1 release inventory changed immediately before solve")
    _verify_root_entry(
        boundary.release_parent_fd,
        root.name,
        root_fd,
        label="R2R-1 release root immediately before solve",
    )
    release_parent_identity = linear._directory_binding_identity(
        os.fstat(boundary.release_parent_fd)
    )
    candidate_parent_identity = linear._directory_binding_identity(
        os.fstat(boundary.candidate_parent_fd)
    )
    held_root_identity = linear._directory_binding_identity(os.fstat(root_fd))
    root_entry_identity = linear._directory_binding_identity(
        os.stat(root.name, dir_fd=boundary.release_parent_fd, follow_symlinks=False)
    )
    if (
        not linear._json_type_exact_equal(
            release_parent_identity, boundary.release_parent_chain[-1]
        )
        or not linear._json_type_exact_equal(
            candidate_parent_identity, boundary.candidate_parent_chain[-1]
        )
        or not linear._json_type_exact_equal(
            held_root_identity, dict(expected_root_identity)
        )
        or not linear._json_type_exact_equal(
            root_entry_identity, dict(expected_root_identity)
        )
    ):
        raise ValueError("R2R-1 stable publication binding changed before solve")
    if boundary.candidate_state == "absent":
        try:
            os.stat(
                boundary.candidate_name,
                dir_fd=boundary.candidate_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        raise FileExistsError("R2R-1 candidate appeared immediately before solve")

    assert boundary.candidate_fd is not None
    assert boundary.anchor_name is not None
    assert boundary.anchor_identity is not None
    _verify_root_entry(
        boundary.candidate_parent_fd,
        boundary.candidate_name,
        boundary.candidate_fd,
        label="R2R-1 candidate root immediately before solve",
    )
    assert boundary.candidate_chain is not None
    candidate_binding = linear._directory_binding_identity(
        os.fstat(boundary.candidate_fd)
    )
    candidate_entry_binding = linear._directory_binding_identity(
        os.stat(
            boundary.candidate_name,
            dir_fd=boundary.candidate_parent_fd,
            follow_symlinks=False,
        )
    )
    if (
        not linear._json_type_exact_equal(
            candidate_binding, boundary.candidate_chain[-1]
        )
        or not linear._json_type_exact_equal(
            candidate_entry_binding, boundary.candidate_chain[-1]
        )
    ):
        raise ValueError("R2R-1 candidate stable binding changed before solve")
    if set(os.listdir(boundary.candidate_fd)) != {boundary.anchor_name}:
        raise ValueError("R2R-1 candidate inventory changed immediately before solve")
    anchor_entry = os.stat(
        boundary.anchor_name,
        dir_fd=boundary.candidate_fd,
        follow_symlinks=False,
    )
    if not linear._json_type_exact_equal(
        linear._owned_regular_identity(anchor_entry), boundary.anchor_identity
    ):
        raise ValueError("R2R-1 candidate anchor identity changed before solve")
    final_anchor_raw = _read_regular_file_at(
        boundary.candidate_fd,
        boundary.anchor_name,
        "R2R-1 candidate anchor at final pre-solve gate",
        size_limit=2 * 1024 * 1024,
        expected_identity=boundary.anchor_identity,
    )
    if final_anchor_raw != boundary.anchor_raw:
        raise ValueError("R2R-1 candidate anchor bytes changed at pre-solve gate")


def _recover_fit_release(
    *,
    output_root: Path,
    attempt3_root: Path,
    thermal92_path: Path,
    freeze_manifest: Path,
    authorization_marker: Path,
    release_manifest: Path,
    expected_release_manifest_sha256: str,
    require_terminal: bool,
) -> dict[str, Any]:
    """Hold the anchor and release roots for the entire scientific recovery."""
    paths = _validate_production_path_contract(
        output_root=output_root,
        attempt3_root=attempt3_root,
        thermal92_path=thermal92_path,
        freeze_manifest=freeze_manifest,
        authorization_marker=authorization_marker,
        release_manifest=release_manifest,
    )
    root = Path(paths["output_root"])
    release_path = Path(paths["release_manifest"])
    authorization = linear.validate_execution_authorization(
        freeze_manifest, authorization_marker
    )
    candidate_root = linear._lexical_path(
        linear.RECOMMENDED_CANDIDATE_ROOT,
        "R2R-1 recovery candidate root",
        forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if release_path.parent != candidate_root:
        raise PermissionError("R2R-1 recovery manifest escaped candidate root")
    binding = authorization["result_parent_binding"]
    candidate_parent_fd: int | None = None
    candidate_fd: int | None = None
    root_parent_fd: int | None = None
    root_fd: int | None = None
    try:
        candidate_parent, candidate_parent_fd, candidate_parent_chain = (
            linear._open_directory_chain(
                candidate_root.parent,
                "R2R-1 recovery candidate parent",
                expected_chain=binding["chain"],
            )
        )
        if not linear._json_type_exact_equal(
            linear._directory_binding_identity(os.fstat(candidate_parent_fd)),
            binding["directory_identity"],
        ) or candidate_root.name != binding["candidate_child_basename"]:
            raise PermissionError("R2R-1 recovery candidate parent changed")
        before = os.stat(
            candidate_root.name,
            dir_fd=candidate_parent_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ValueError("R2R-1 recovery candidate root is not a directory")
        candidate_fd = os.open(
            candidate_root.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=candidate_parent_fd,
        )
        candidate_identity = linear._directory_binding_identity(
            os.fstat(candidate_fd)
        )
        if candidate_identity != linear._directory_binding_identity(before):
            raise ValueError("R2R-1 recovery candidate root changed across open")
        candidate_chain = (
            *candidate_parent_chain,
            linear._directory_binding_identity(os.fstat(candidate_fd)),
        )
        if set(os.listdir(candidate_fd)) != {release_path.name}:
            raise ValueError("R2R-1 recovery candidate inventory changed")
        release_identity = linear._owned_regular_identity(
            os.stat(release_path.name, dir_fd=candidate_fd, follow_symlinks=False)
        )
        release_raw = _read_regular_file_at(
            candidate_fd,
            release_path.name,
            "R2R-1 external release manifest",
            size_limit=2 * 1024 * 1024,
            expected_identity=release_identity,
        )
        (
            root_parent,
            root_parent_fd,
            root_parent_chain,
            bound_root,
            root_fd,
            root_chain,
        ) = _open_marker_bound_release_root(
            root, authorization, label="R2R-1 recovery root"
        )
        external = _load_external_release_manifest_raw(
            release_raw,
            expected_sha256=expected_release_manifest_sha256,
            output_root=bound_root,
            authorization=authorization,
            require_terminal=require_terminal,
        )
        if not require_terminal:
            raise ValueError("public R2R-1 recovery requires a DONE release")
        baseline_materialization = _completed_materialization_receipt_bound(
            root_fd,
            expected_done_identity=external["expected_done_identity"],
            expected_done_sha256=external["expected_done_sha256"],
        )
        baseline_release_snapshot = baseline_materialization[
            "expected_release_snapshot"
        ]
        baseline_checkpoint_required = (
            baseline_materialization["status"] == CHECKPOINT_STATUS_SOURCE
        )
        if (
            baseline_materialization["status"] == CHECKPOINT_STATUS_SOURCE
        ) is not baseline_checkpoint_required:
            raise ValueError("R2R-1 recovery checkpoint/status baseline changed")
        # The baseline reads above can be long.  Close both canonical roots
        # again immediately before entering the scientific re-solve so a
        # phase-boundary replacement cannot be adopted through detached fds.
        _verify_snapshot_and_anchor_bytes_fd(
            root_fd,
            baseline_release_snapshot,
            candidate_fd,
            release_path.name,
            release_identity,
            release_raw,
            label="R2R-1 recovery pre-solve",
        )
        if set(os.listdir(root_fd)) != {
            name for name in baseline_release_snapshot if "/" not in name
        }:
            raise ValueError("R2R-1 recovery release inventory changed pre-solve")
        linear._verify_directory_chain(
            root_parent,
            "R2R-1 recovery shared parent immediately before solve",
            root_parent_chain,
        )
        _verify_root_entry(
            root_parent_fd,
            root.name,
            root_fd,
            label="R2R-1 recovery root immediately before solve",
        )
        linear._verify_directory_chain(
            candidate_root,
            "R2R-1 recovery candidate immediately before solve",
            candidate_chain,
        )
        _verify_root_entry(
            candidate_parent_fd,
            candidate_root.name,
            candidate_fd,
            label="R2R-1 recovery candidate root immediately before solve",
        )
        if set(os.listdir(candidate_fd)) != {release_path.name}:
            raise ValueError("R2R-1 recovery candidate inventory changed pre-solve")
        candidate_entry = os.stat(
            release_path.name, dir_fd=candidate_fd, follow_symlinks=False
        )
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(candidate_entry), release_identity
        ):
            raise ValueError("R2R-1 recovery anchor identity changed pre-solve")
        recovery_publication_boundary = _PreSolvePublicationBoundary(
            release_parent_fd=root_parent_fd,
            release_parent_path=root_parent,
            release_parent_chain=root_parent_chain,
            candidate_parent_fd=candidate_parent_fd,
            candidate_parent_path=candidate_parent,
            candidate_parent_chain=candidate_parent_chain,
            candidate_name=candidate_root.name,
            candidate_state="existing",
            candidate_fd=candidate_fd,
            candidate_path=candidate_root,
            candidate_chain=candidate_chain,
            anchor_name=release_path.name,
            anchor_identity=release_identity,
            anchor_raw=release_raw,
        )
        receipt = _recover_fit_release_bound(
            root=bound_root,
            root_fd=root_fd,
            root_chain=root_chain,
            output_root=output_root,
            attempt3_root=attempt3_root,
            thermal92_path=thermal92_path,
            freeze_manifest=freeze_manifest,
            authorization_marker=authorization_marker,
            release_manifest_raw=release_raw,
            expected_release_manifest_sha256=expected_release_manifest_sha256,
            require_terminal=require_terminal,
            publication_boundary=recovery_publication_boundary,
        )
        if receipt["status"] != baseline_materialization["status"]:
            raise ValueError("R2R-1 recovery status changed from frozen baseline")
        if set(os.listdir(candidate_fd)) != {release_path.name}:
            raise ValueError("R2R-1 recovery candidate inventory changed at return")
        final_release_entry = os.stat(
            release_path.name,
            dir_fd=candidate_fd,
            follow_symlinks=False,
        )
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(final_release_entry), release_identity
        ):
            raise ValueError("R2R-1 external release identity changed during recovery")
        linear._verify_directory_chain(
            candidate_root,
            "R2R-1 recovery final candidate-root binding",
            candidate_chain,
        )
        linear._verify_directory_chain(
            candidate_parent,
            "R2R-1 recovery final candidate-parent binding",
            candidate_parent_chain,
        )
        linear._verify_directory_chain(
            root_parent,
            "R2R-1 recovery final marker-bound parent",
            root_parent_chain,
        )
        if _release_snapshot_fd(
            root_fd,
            checkpoint_required=baseline_checkpoint_required,
            terminal_name="DONE" if require_terminal else "RUNNING",
        ) != baseline_release_snapshot:
            raise ValueError("R2R-1 recovery release changed from frozen baseline")
        # The full release recapture above is the last long payload operation.
        # Rebind the shared lexical parent afterwards so a phase-boundary
        # parent swap cannot leave the held descriptors detached from the
        # canonical release while recovery still reports success.
        linear._verify_directory_chain(
            root_parent,
            "R2R-1 recovery return shared-parent binding",
            root_parent_chain,
        )
        _verify_snapshot_and_anchor_bytes_fd(
            root_fd,
            baseline_release_snapshot,
            candidate_fd,
            release_path.name,
            release_identity,
            release_raw,
            label="R2R-1 recovery return",
        )
        if set(os.listdir(root_fd)) != {
            name for name in baseline_release_snapshot if "/" not in name
        }:
            raise ValueError("R2R-1 recovery release inventory changed at return")
        if set(os.listdir(candidate_fd)) != {release_path.name}:
            raise ValueError("R2R-1 recovery anchor inventory changed at return")
        return_release_entry = os.stat(
            release_path.name,
            dir_fd=candidate_fd,
            follow_symlinks=False,
        )
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(return_release_entry), release_identity
        ):
            raise ValueError("R2R-1 recovery anchor identity changed at return")
        _verify_root_entry(
            candidate_parent_fd,
            candidate_root.name,
            candidate_fd,
            label="R2R-1 recovery candidate root at return",
        )
        _verify_root_entry(
            root_parent_fd,
            root.name,
            root_fd,
            label="R2R-1 recovered root at return",
        )
        return receipt
    finally:
        if root_fd is not None:
            os.close(root_fd)
        if root_parent_fd is not None:
            os.close(root_parent_fd)
        if candidate_fd is not None:
            os.close(candidate_fd)
        if candidate_parent_fd is not None:
            os.close(candidate_parent_fd)


def _recover_fit_release_bound(
    *,
    root: Path,
    root_fd: int,
    root_chain: tuple[dict[str, int], ...],
    output_root: Path,
    attempt3_root: Path,
    thermal92_path: Path,
    freeze_manifest: Path,
    authorization_marker: Path,
    release_manifest_raw: bytes,
    expected_release_manifest_sha256: str,
    require_terminal: bool,
    publication_boundary: _PreSolvePublicationBoundary,
    expected_prepublish_snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bound-root implementation for scientific recovery."""
    # Directory mtime/ctime can change asynchronously on Darwin when
    # com.apple.provenance is attached.  Directory identity is therefore the
    # stable binding tuple; exact inventory and every child full identity/byte
    # remain protected by the release snapshots below.
    root_identity = linear._directory_binding_identity(os.fstat(root_fd))
    authorization = linear.validate_execution_authorization(
        freeze_manifest, authorization_marker
    )
    external = _load_external_release_manifest_raw(
        release_manifest_raw,
        expected_sha256=expected_release_manifest_sha256,
        output_root=root,
        authorization=authorization,
        require_terminal=require_terminal,
    )
    expected_completion = external["expected_completion_sha256"]
    expected_fit_receipt = external["expected_fit_receipt_sha256"]
    expected_checkpoint = external["expected_checkpoint_receipt_sha256"]
    root_names = set(os.listdir(root_fd))
    terminal = {name for name in ("RUNNING", "FAILED", "DONE") if name in root_names}
    expected_terminal = {"DONE"} if require_terminal else {"RUNNING"}
    if terminal != expected_terminal:
        raise ValueError("R2R-1 release terminal/prepublish state changed")
    if require_terminal:
        materialization = _completed_materialization_receipt_bound(
            root_fd,
            expected_done_identity=external["expected_done_identity"],
            expected_done_sha256=external["expected_done_sha256"],
        )
        initial_release_snapshot = materialization["expected_release_snapshot"]
        completion = {
            key: materialization[key]
            for key in COMPLETION_KEYS
        }
        completion_item = initial_release_snapshot[COMPLETION_BASENAME]
        completion_raw = _read_regular_file_at(
            root_fd,
            COMPLETION_BASENAME,
            "R2R-1 externally ledger-bound completion",
            size_limit=ROOT_FILE_SIZE_LIMITS[COMPLETION_BASENAME],
            expected_identity=completion_item["identity"],
        )
    else:
        if expected_prepublish_snapshot is None:
            raise ValueError("R2R-1 internal recovery lacks its in-memory baseline")
        initial_release_snapshot = {
            name: dict(item) for name, item in expected_prepublish_snapshot.items()
        }
        if "RUNNING" not in initial_release_snapshot or "DONE" in initial_release_snapshot:
            raise ValueError("R2R-1 internal recovery baseline terminal changed")
        if set(os.listdir(root_fd)) != {
            name for name in initial_release_snapshot if "/" not in name
        }:
            raise ValueError("R2R-1 internal recovery inventory changed")
        _verify_snapshot_bytes_fd(root_fd, initial_release_snapshot)
        completion_item = initial_release_snapshot.get(COMPLETION_BASENAME)
        if not isinstance(completion_item, Mapping):
            raise ValueError("R2R-1 internal recovery baseline lost completion")
        completion_raw = _read_regular_file_at(
            root_fd,
            COMPLETION_BASENAME,
            "R2R-1 in-memory ledger-bound completion",
            size_limit=ROOT_FILE_SIZE_LIMITS[COMPLETION_BASENAME],
            expected_identity=completion_item["identity"],
        )
        completion = _strict_canonical_json(completion_raw, "R2R-1 completion")
        _validate_completion_invariants(completion)
    expected_root_binding = completion["release_root_binding_identity"]
    if (
        not linear._json_type_exact_equal(root_identity, expected_root_binding)
        or not linear._json_type_exact_equal(root_chain[-1], expected_root_binding)
    ):
        raise ValueError("R2R-1 release root differs from its completion binding")
    completion_sha = _sha256_bytes(completion_raw)
    if completion_sha != expected_completion:
        raise ValueError("R2R-1 completion differs from external frozen SHA-256")
    if completion.get("status") not in SCIENTIFIC_STATUSES:
        raise ValueError("R2R-1 completion scientific status changed")
    if completion.get("status") != external.get("status"):
        raise ValueError("R2R-1 completion status differs from external manifest")
    if completion["precompletion_artifact_snapshot_sha256"] != external[
        "precompletion_artifact_snapshot_sha256"
    ]:
        raise ValueError("R2R-1 release snapshot digest differs from its manifest")

    manifest_item = initial_release_snapshot[FIT_MANIFEST_BASENAME]
    manifest_raw = _read_regular_file_at(
        root_fd,
        FIT_MANIFEST_BASENAME,
        "R2R-1 frozen fit manifest",
        size_limit=ROOT_FILE_SIZE_LIMITS[FIT_MANIFEST_BASENAME],
        expected_identity=manifest_item["identity"],
    )
    external_raw, external_identity = _read_owned(
        freeze_manifest,
        "R2R-1 external fit manifest",
        expected_parent_chain=tuple(authorization["freeze_manifest_parent_chain"]),
        expected_file_identity=authorization["freeze_manifest_file_identity"],
        size_limit=FREEZE_MANIFEST_BYTES_LIMIT,
    )
    if (
        manifest_raw != external_raw
        or _sha256_bytes(manifest_raw) != completion.get("fit_manifest_sha256")
        or not linear._json_type_exact_equal(
            external_identity, authorization["freeze_manifest_file_identity"]
        )
    ):
        raise ValueError("R2R-1 recovered fit manifest binding changed")
    fit_receipt_item = initial_release_snapshot[FIT_RECEIPT_BASENAME]
    fit_receipt_raw = _read_regular_file_at(
        root_fd,
        FIT_RECEIPT_BASENAME,
        "R2R-1 frozen fit receipt",
        size_limit=ROOT_FILE_SIZE_LIMITS[FIT_RECEIPT_BASENAME],
        expected_identity=fit_receipt_item["identity"],
    )
    if (
        _sha256_bytes(fit_receipt_raw) != completion.get("fit_receipt_sha256")
        or _sha256_bytes(fit_receipt_raw) != expected_fit_receipt
    ):
        raise ValueError("R2R-1 recovered fit receipt SHA changed")
    fit_receipt = _strict_canonical_json(fit_receipt_raw, "R2R-1 fit receipt")
    _validate_fit_receipt_invariants(fit_receipt)
    if (
        fit_receipt.get("format") != FIT_RECEIPT_FORMAT
        or fit_receipt.get("status") != completion.get("status")
        or fit_receipt.get("fit_manifest", {}).get("sha256")
        != completion.get("fit_manifest_sha256")
    ):
        raise ValueError("R2R-1 recovered fit receipt schema/status changed")
    if not linear._json_type_exact_equal(
        fit_receipt.get("runtime_environment"), linear.runtime_environment_receipt()
    ):
        raise ValueError("R2R-1 recovery runtime environment changed")
    arrays_item = initial_release_snapshot[FIT_ARRAYS_BASENAME]
    arrays_raw = _read_regular_file_at(
        root_fd,
        FIT_ARRAYS_BASENAME,
        "R2R-1 frozen fit arrays",
        size_limit=ROOT_FILE_SIZE_LIMITS[FIT_ARRAYS_BASENAME],
        expected_identity=arrays_item["identity"],
    )
    if _sha256_bytes(arrays_raw) != completion.get("fit_arrays_sha256"):
        raise ValueError("R2R-1 recovered fit arrays SHA changed")
    arrays = _load_npz_bytes(
        arrays_raw, fit_receipt["fit_arrays"]["schema"], "R2R-1 fit arrays"
    )
    checkpoint_required = completion["status"] == CHECKPOINT_STATUS_SOURCE
    _verify_snapshot_bytes_fd(root_fd, initial_release_snapshot)
    if set(os.listdir(root_fd)) != {
        name for name in initial_release_snapshot if "/" not in name
    }:
        raise ValueError("R2R-1 bound recovery inventory changed before solve")
    linear._verify_directory_chain(
        root, "R2R-1 bound recovery lexical root before solve", root_chain
    )
    _verify_pre_solve_publication_boundary(
        root=root,
        root_fd=root_fd,
        expected_root_identity=expected_root_binding,
        expected_release_snapshot=initial_release_snapshot,
        boundary=publication_boundary,
    )
    fit_publication_boundary = linear._issue_fit_publication_boundary(
        release_parent_fd=publication_boundary.release_parent_fd,
        release_parent_path=publication_boundary.release_parent_path,
        release_parent_chain=publication_boundary.release_parent_chain,
        root_fd=root_fd,
        root_path=root,
        expected_root_identity=expected_root_binding,
        expected_root_snapshot=initial_release_snapshot,
        candidate_parent_fd=publication_boundary.candidate_parent_fd,
        candidate_parent_path=publication_boundary.candidate_parent_path,
        candidate_parent_chain=publication_boundary.candidate_parent_chain,
        candidate_name=publication_boundary.candidate_name,
        candidate_state=publication_boundary.candidate_state,
        candidate_fd=publication_boundary.candidate_fd,
        candidate_path=publication_boundary.candidate_path,
        candidate_chain=publication_boundary.candidate_chain,
        anchor_name=publication_boundary.anchor_name,
        anchor_identity=publication_boundary.anchor_identity,
        anchor_raw=publication_boundary.anchor_raw,
    )
    recomputed_receipt, recomputed_arrays = linear.authorized_fit_pipeline(
        attempt3_root=attempt3_root,
        thermal92_path=thermal92_path,
        freeze_manifest=freeze_manifest,
        authorization_marker=authorization_marker,
        publication_boundary=fit_publication_boundary,
    )
    stored_pipeline = fit_receipt.get("pipeline_receipt")
    if linear.canonical_json_bytes(recomputed_receipt) != linear.canonical_json_bytes(
        stored_pipeline
    ):
        raise ValueError("R2R-1 scientific receipt changed on authorized re-solve")
    if fit_receipt.get("pipeline_recursive_schema_sha256") != _recursive_schema_sha256(
        recomputed_receipt
    ):
        raise ValueError("R2R-1 pipeline type schema changed on authorized re-solve")
    if set(recomputed_arrays) != set(arrays):
        raise ValueError("R2R-1 scientific array member set changed on re-solve")
    for name in arrays:
        expected_array = np.asarray(recomputed_arrays[name], dtype="<f8")
        if (
            arrays[name].shape != expected_array.shape
            or arrays[name].dtype != expected_array.dtype
            or linear.raw_array_sha256(arrays[name], "<f8")
            != linear.raw_array_sha256(expected_array, "<f8")
        ):
            raise ValueError(
                f"R2R-1 scientific array changed on authorized re-solve: {name}"
            )

    checkpoint_summary = completion.get("checkpoint")
    expected_root = {
        FIT_MANIFEST_BASENAME,
        FIT_ARRAYS_BASENAME,
        FIT_RECEIPT_BASENAME,
        COMPLETION_BASENAME,
    }
    expected_root.update({"EXIT_CODE", "DONE"} if require_terminal else {"RUNNING"})
    if checkpoint_required:
        expected_root.add(CHECKPOINT_DIRNAME)
        if not isinstance(checkpoint_summary, dict):
            raise ValueError("R2R-1 passing release lacks checkpoint summary")
        if (
            expected_checkpoint is None
            or checkpoint_summary.get("receipt_sha256") != expected_checkpoint
        ):
            raise ValueError("R2R-1 checkpoint differs from external frozen SHA-256")
        checkpoint_files = _bound_checkpoint_files(root_fd)
        observed_hashes = {
            name: _sha256_bytes(raw) for name, raw in checkpoint_files.items()
        }
        if observed_hashes != checkpoint_summary.get("file_sha256"):
            raise ValueError("R2R-1 checkpoint three-file hashes changed")
        checkpoint = _load_settled_private_checkpoint(
            checkpoint_files,
            expected_receipt_sha256=checkpoint_summary["receipt_sha256"],
            prefix="graphene_r2r1_checkpoint_recovery_",
        )
        if (
            checkpoint.receipt["fit_provenance"]["fit_manifest_sha256"]
            != completion["fit_manifest_sha256"]
            or checkpoint.receipt["fit_provenance"]["fit_receipt_sha256"]
            != completion["fit_receipt_sha256"]
            or linear.raw_array_sha256(checkpoint.physical_p, "<f8")
            != linear.raw_array_sha256(arrays["final_coefficients"], "<f8")
        ):
            raise ValueError("R2R-1 checkpoint fit provenance changed")
    elif (
        expected_checkpoint is not None
        or checkpoint_summary is not None
        or CHECKPOINT_DIRNAME in root_names
    ):
        raise ValueError("R2R-1 failed-gate release must not contain a checkpoint")
    if set(os.listdir(root_fd)) != expected_root:
        raise ValueError("R2R-1 completed release root inventory changed")
    final_authorization = linear.validate_execution_authorization(
        freeze_manifest, authorization_marker
    )
    if not linear._json_type_exact_equal(final_authorization, authorization):
        raise PermissionError("R2R-1 authorization changed during recovery")
    if not linear._json_type_exact_equal(
        linear._directory_binding_identity(os.fstat(root_fd)), root_identity
    ):
        raise ValueError("R2R-1 release root changed during recovery")
    final_release_snapshot = _release_snapshot_fd(
        root_fd,
        checkpoint_required=checkpoint_required,
        terminal_name="DONE" if require_terminal else "RUNNING",
    )
    if final_release_snapshot != initial_release_snapshot:
        raise ValueError("R2R-1 release content changed during scientific re-solve")
    linear._verify_directory_chain(
        root, "R2R-1 recovery final lexical root binding", root_chain
    )
    return {
        **completion,
        "completion_sha256": completion_sha,
        "scientific_recomputed": True,
        "artifacts_recovered_without_rewrite": True,
        "terminal_verified": require_terminal,
        "fit_arrays_member_names": sorted(arrays),
        "terminal_inventory_exact": True,
    }


def recover_completed_fit(
    *,
    output_root: Path,
    attempt3_root: Path,
    thermal92_path: Path,
    freeze_manifest: Path,
    authorization_marker: Path,
    release_manifest: Path,
    expected_release_manifest_sha256: str,
) -> dict[str, Any]:
    """Public recovery requires an externally SHA-pinned release manifest."""
    return _recover_fit_release(
        output_root=output_root,
        attempt3_root=attempt3_root,
        thermal92_path=thermal92_path,
        freeze_manifest=freeze_manifest,
        authorization_marker=authorization_marker,
        release_manifest=release_manifest,
        expected_release_manifest_sha256=expected_release_manifest_sha256,
        require_terminal=True,
    )


def anchor_completed_release(
    *,
    output_root: Path,
    attempt3_root: Path,
    thermal92_path: Path,
    freeze_manifest: Path,
    authorization_marker: Path,
    release_manifest: Path,
    materialization_witness: object | None = None,
) -> dict[str, Any]:
    """Validate an existing DONE release, then create/recover its external anchor.

    This operation never invokes ``materialize_authorized_fit``.  It is safe to
    rerun after an anchor-write failure without performing a second publication.
    """
    release_lexical = Path(os.path.abspath(str(release_manifest)))
    expected_release = Path(
        os.path.abspath(str(linear.RECOMMENDED_RELEASE_MANIFEST))
    )
    if release_lexical != expected_release:
        raise PermissionError(
            "external release manifest differs from frozen production path"
        )
    paths = _validate_production_path_contract(
        output_root=output_root,
        attempt3_root=attempt3_root,
        thermal92_path=thermal92_path,
        freeze_manifest=freeze_manifest,
        authorization_marker=authorization_marker,
        release_manifest=release_manifest,
    )
    target = Path(paths["release_manifest"])
    root = Path(paths["output_root"])
    authorization = linear.validate_execution_authorization(
        freeze_manifest, authorization_marker
    )
    candidate_root = linear._lexical_path(
        linear.RECOMMENDED_CANDIDATE_ROOT,
        "R2R-1 frozen external candidate root",
        forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if target.parent != candidate_root:
        raise PermissionError("R2R-1 release manifest escaped candidate root")
    parent_binding = authorization["result_parent_binding"]
    candidate_parent, candidate_parent_fd, candidate_parent_chain = (
        linear._open_directory_chain(
            candidate_root.parent,
            "R2R-1 marker-bound candidate parent",
            expected_chain=parent_binding["chain"],
        )
    )
    candidate_fd: int | None = None
    release_root_fd: int | None = None
    release_parent_fd: int | None = None
    candidate_temporary_name: str | None = None
    candidate_chain: tuple[dict[str, int], ...] | None = None
    anchor_identity: dict[str, int] | None = None
    candidate_directory_binding_identity: dict[str, int] | None = None
    candidate_committed = False
    try:
        if (
            not linear._json_type_exact_equal(
                linear._directory_binding_identity(os.fstat(candidate_parent_fd)),
                parent_binding["directory_identity"],
            )
            or candidate_root.name != parent_binding["candidate_child_basename"]
        ):
            raise PermissionError("R2R-1 candidate parent identity changed")
        try:
            os.stat(
                candidate_root.name,
                dir_fd=candidate_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            candidate_preexisting = False
            bound_candidate = candidate_root
        else:
            candidate_preexisting = True
            bound_candidate, candidate_fd, candidate_chain = (
                _open_existing_bound_child_directory(
                    parent_path=candidate_parent,
                    parent_fd=candidate_parent_fd,
                    parent_chain=candidate_parent_chain,
                    child_name=candidate_root.name,
                    label="R2R-1 existing external candidate root",
                )
            )
            if set(os.listdir(candidate_fd)) != {target.name}:
                raise ValueError(
                    "existing R2R-1 candidate root is incomplete; a fresh "
                    "candidate attempt and authorization are required"
                )
            anchor_identity = linear._owned_regular_identity(
                os.stat(target.name, dir_fd=candidate_fd, follow_symlinks=False)
            )
        (
            release_parent,
            release_parent_fd,
            release_parent_chain,
            bound_root,
            release_root_fd,
            release_root_chain,
        ) = _open_marker_bound_release_root(
            root, authorization, label="R2R-1 anchor release root"
        )
        preexisting_anchor_raw: bytes | None = None
        preexisting_external: dict[str, Any] | None = None
        if candidate_preexisting:
            assert candidate_fd is not None and anchor_identity is not None
            preexisting_anchor_raw = _read_regular_file_at(
                candidate_fd,
                target.name,
                "R2R-1 preexisting external release constraint",
                size_limit=2 * 1024 * 1024,
                expected_identity=anchor_identity,
            )
            try:
                preexisting_external = _load_external_release_manifest_raw(
                    preexisting_anchor_raw,
                    expected_sha256=_sha256_bytes(preexisting_anchor_raw),
                    output_root=bound_root,
                    authorization=authorization,
                    require_terminal=True,
                )
            except (TypeError, ValueError, PermissionError) as error:
                raise FileExistsError(
                    "existing external release manifest differs"
                ) from error
            materialization = _completed_materialization_receipt_bound(
                release_root_fd,
                expected_done_identity=preexisting_external[
                    "expected_done_identity"
                ],
                expected_done_sha256=preexisting_external["expected_done_sha256"],
            )
        else:
            if materialization_witness is None:
                raise PermissionError(
                    "an unanchored DONE release requires its in-process "
                    "materialization witness or a separately authorized pin"
                )
            witness = _require_materialization_witness(materialization_witness)
            if witness["release_root"] != str(bound_root):
                raise PermissionError(
                    "R2R-1 materialization witness release root changed"
                )
            materialization = _completed_materialization_receipt_bound(
                release_root_fd,
                expected_done_identity=witness["done_identity"],
                expected_done_sha256=witness["done_sha256"],
            )
            witness_snapshot = witness["expected_release_snapshot"]
            current_snapshot = materialization["expected_release_snapshot"]
            if (
                materialization["status"] != witness["status"]
                or set(current_snapshot) != set(witness_snapshot)
                or any(
                    not linear._json_type_exact_equal(
                        current_snapshot[name], witness_snapshot[name]
                    )
                    for name in current_snapshot
                )
                or materialization["done_sha256"]
                != witness["done_sha256"]
                or not linear._json_type_exact_equal(
                    materialization["done_identity"], witness["done_identity"]
                )
            ):
                raise ValueError(
                    "R2R-1 DONE release differs from its materialization witness"
                )
        expected_release_snapshot = materialization["expected_release_snapshot"]
        checkpoint_required = materialization["status"] == CHECKPOINT_STATUS_SOURCE
        payload = _release_manifest_payload_for_bound_root(
            root=bound_root, materialization_receipt=materialization
        )
        raw = linear.canonical_json_bytes(payload)
        digest = _sha256_bytes(raw)
        if candidate_preexisting:
            assert candidate_fd is not None and anchor_identity is not None
            assert preexisting_anchor_raw is not None
            if preexisting_anchor_raw != raw:
                raise FileExistsError("existing external release manifest differs")
            # The candidate was classified before the long release baseline
            # and materialization reads.  Re-read its expected bytes first,
            # then perform only short canonical identity/inventory checks
            # immediately before the scientific recovery.
            linear._verify_directory_chain(
                candidate_root,
                "R2R-1 existing candidate immediately before recovery",
                candidate_chain,
            )
            _verify_root_entry(
                candidate_parent_fd,
                candidate_root.name,
                candidate_fd,
                label="R2R-1 existing candidate root immediately before recovery",
            )
            if set(os.listdir(candidate_fd)) != {target.name}:
                raise ValueError("R2R-1 existing candidate inventory changed pre-recovery")
            adjacent_anchor = os.stat(
                target.name, dir_fd=candidate_fd, follow_symlinks=False
            )
            if not linear._json_type_exact_equal(
                linear._owned_regular_identity(adjacent_anchor), anchor_identity
            ):
                raise ValueError("R2R-1 existing anchor identity changed pre-recovery")
        else:
            linear._verify_directory_chain(
                candidate_parent,
                "R2R-1 absent candidate parent immediately before recovery",
                candidate_parent_chain,
            )
            try:
                os.stat(
                    candidate_root.name,
                    dir_fd=candidate_parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(
                    "R2R-1 candidate root appeared before scientific recovery"
                )
        anchor_publication_boundary = _PreSolvePublicationBoundary(
            release_parent_fd=release_parent_fd,
            release_parent_path=release_parent,
            release_parent_chain=release_parent_chain,
            candidate_parent_fd=candidate_parent_fd,
            candidate_parent_path=candidate_parent,
            candidate_parent_chain=candidate_parent_chain,
            candidate_name=candidate_root.name,
            candidate_state="existing" if candidate_preexisting else "absent",
            candidate_fd=candidate_fd if candidate_preexisting else None,
            candidate_path=candidate_root if candidate_preexisting else None,
            candidate_chain=candidate_chain if candidate_preexisting else None,
            anchor_name=target.name if candidate_preexisting else None,
            anchor_identity=anchor_identity if candidate_preexisting else None,
            anchor_raw=raw if candidate_preexisting else None,
        )
        recovered = _recover_fit_release_bound(
            root=bound_root,
            root_fd=release_root_fd,
            root_chain=release_root_chain,
            output_root=bound_root,
            attempt3_root=attempt3_root,
            thermal92_path=thermal92_path,
            freeze_manifest=freeze_manifest,
            authorization_marker=authorization_marker,
            release_manifest_raw=raw,
            expected_release_manifest_sha256=digest,
            require_terminal=True,
            publication_boundary=anchor_publication_boundary,
        )
        if recovered.get("scientific_recomputed") is not True:
            raise ValueError("R2R-1 completed release failed pre-anchor revalidation")
        if _release_snapshot_fd(
            release_root_fd,
            checkpoint_required=checkpoint_required,
            terminal_name="DONE",
        ) != expected_release_snapshot:
            raise ValueError("R2R-1 release changed from pre-recovery baseline")

        def revalidate_release_at_candidate_commit() -> None:
            if not linear._json_type_exact_equal(
                _completed_materialization_receipt_bound(release_root_fd),
                materialization,
            ):
                raise ValueError("R2R-1 release changed at candidate commit")
            if _release_snapshot_fd(
                release_root_fd,
                checkpoint_required=checkpoint_required,
                terminal_name="DONE",
            ) != expected_release_snapshot:
                raise ValueError(
                    "R2R-1 full release snapshot changed at candidate commit"
                )
            # The full payload checks above may be long.  Rebind the lexical
            # shared parent and held release inode only after those reads, so
            # the callback returns with the release immediately closed.
            linear._verify_directory_chain(
                release_parent,
                "R2R-1 release parent at candidate commit",
                release_parent_chain,
            )
            _verify_root_entry(
                release_parent_fd,
                root.name,
                release_root_fd,
                label="R2R-1 release root adjacent to candidate commit",
            )
        if candidate_preexisting:
            assert candidate_fd is not None and anchor_identity is not None
            existing = _read_regular_file_at(
                candidate_fd,
                target.name,
                "R2R-1 existing external release manifest",
                size_limit=2 * 1024 * 1024,
                expected_identity=anchor_identity,
            )
            if existing != raw:
                raise FileExistsError("existing external release manifest differs")
            newly_materialized = False
        else:
            # The canonical candidate basename stays absent throughout the
            # long scientific recovery.  Build the complete one-file release
            # in a private sibling, then publish the directory itself with one
            # no-replace rename.  Thus an interrupted precommit never strands
            # an ambiguous canonical empty/partial directory.
            try:
                os.stat(
                    candidate_root.name,
                    dir_fd=candidate_parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("R2R-1 candidate root appeared during recovery")
            (
                candidate_temporary_name,
                _candidate_temporary_path,
                candidate_fd,
                candidate_chain,
            ) = _create_bound_candidate_temporary(
                parent_path=candidate_parent,
                parent_fd=candidate_parent_fd,
                parent_chain=candidate_parent_chain,
            )
            candidate_directory_binding_identity = (
                linear._directory_binding_identity(os.fstat(candidate_fd))
            )
            anchor_identity = _atomic_write_bytes_at(candidate_fd, target.name, raw)
            existing = _read_regular_file_at(
                candidate_fd,
                target.name,
                "R2R-1 new external release manifest",
                size_limit=2 * 1024 * 1024,
                expected_identity=anchor_identity,
            )
            if existing != raw:
                raise ValueError("R2R-1 external release manifest changed after write")
            if set(os.listdir(candidate_fd)) != {target.name}:
                raise ValueError("R2R-1 candidate temporary inventory changed")
            os.fsync(candidate_fd)
            precommit_materialization = _completed_materialization_receipt_bound(
                release_root_fd
            )
            if not linear._json_type_exact_equal(
                precommit_materialization, materialization
            ):
                raise ValueError("R2R-1 release changed before candidate commit")
            _verify_root_entry(
                release_parent_fd,
                root.name,
                release_root_fd,
                label="R2R-1 release root before candidate commit",
            )
            linear._verify_directory_chain(
                release_parent,
                "R2R-1 release parent before candidate commit",
                release_parent_chain,
            )
            try:
                _commit_candidate_directory_at(
                    candidate_parent_fd,
                    candidate_temporary_name,
                    candidate_root.name,
                    candidate_fd,
                    candidate_directory_binding_identity=(
                        candidate_directory_binding_identity
                    ),
                    anchor_name=target.name,
                    anchor_identity=anchor_identity,
                    anchor_raw=raw,
                    release_root_fd=release_root_fd,
                    expected_release_snapshot=expected_release_snapshot,
                    release_parent_fd=release_parent_fd,
                    release_root_name=root.name,
                    release_parent_path=release_parent,
                    release_parent_chain=release_parent_chain,
                    precommit_validator=revalidate_release_at_candidate_commit,
                )
            except _AnchorCommittedError:
                candidate_committed = True
                candidate_temporary_name = None
                raise
            candidate_committed = True
            candidate_temporary_name = None
            bound_candidate = candidate_root
            newly_materialized = True
        assert candidate_fd is not None
        assert candidate_chain is not None
        assert anchor_identity is not None
        if candidate_preexisting:
            # A retry after a post-rename parent-fsync fault completes the
            # directory-entry durability step without changing anchor bytes.
            os.fsync(candidate_parent_fd)
        final_anchor = _read_regular_file_at(
            candidate_fd,
            target.name,
            "R2R-1 final external release manifest",
            size_limit=2 * 1024 * 1024,
            expected_identity=anchor_identity,
        )
        if final_anchor != raw or _sha256_bytes(final_anchor) != digest:
            raise ValueError("R2R-1 final external release manifest changed")
        # Candidate bytes/durability are complete before the final release
        # read.  This makes the last large payload operation the immutable
        # release snapshot, followed only by short identity/entry rebinds.
        final_materialization = _completed_materialization_receipt_bound(
            release_root_fd
        )
        if not linear._json_type_exact_equal(final_materialization, materialization):
            raise ValueError("R2R-1 release changed across anchor publication")
        final_release_snapshot = _release_snapshot_fd(
            release_root_fd,
            checkpoint_required=checkpoint_required,
            terminal_name="DONE",
        )
        if final_release_snapshot != expected_release_snapshot:
            raise ValueError("R2R-1 full release changed across anchor publication")
        _verify_snapshot_bytes_fd(release_root_fd, final_release_snapshot)
        if set(os.listdir(candidate_fd)) != {target.name}:
            raise ValueError("R2R-1 final candidate inventory changed")
        final_anchor_entry = os.stat(
            target.name, dir_fd=candidate_fd, follow_symlinks=False
        )
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(final_anchor_entry), anchor_identity
        ):
            raise ValueError("R2R-1 final anchor identity changed")
        linear._verify_directory_chain(
            bound_candidate,
            "R2R-1 final candidate-root binding",
            candidate_chain,
        )
        linear._verify_directory_chain(
            candidate_parent,
            "R2R-1 final candidate-parent binding",
            candidate_parent_chain,
        )
        linear._verify_directory_chain(
            release_parent,
            "R2R-1 final anchored release-parent binding",
            release_parent_chain,
        )
        _verify_snapshot_and_anchor_bytes_fd(
            release_root_fd,
            expected_release_snapshot,
            candidate_fd,
            target.name,
            anchor_identity,
            raw,
            label="R2R-1 anchor return",
        )
        expected_release_names = {
            name for name in expected_release_snapshot if "/" not in name
        }
        if set(os.listdir(release_root_fd)) != expected_release_names:
            raise ValueError("R2R-1 release inventory changed at anchor return")
        if set(os.listdir(candidate_fd)) != {target.name}:
            raise ValueError("R2R-1 candidate inventory changed at anchor return")
        return_anchor_entry = os.stat(
            target.name, dir_fd=candidate_fd, follow_symlinks=False
        )
        if not linear._json_type_exact_equal(
            linear._owned_regular_identity(return_anchor_entry), anchor_identity
        ):
            raise ValueError("R2R-1 anchor identity changed at return")
        _verify_root_entry(
            candidate_parent_fd,
            candidate_root.name,
            candidate_fd,
            label="R2R-1 candidate root at anchor return",
        )
        _verify_root_entry(
            release_parent_fd,
            root.name,
            release_root_fd,
            label="R2R-1 release root at anchor return",
        )
        return {
            "path": str(target),
            "sha256": digest,
            "newly_materialized": newly_materialized,
            "release_status": recovered["status"],
            "mechanics_status": recovered["mechanics_status"],
            "completed_release_scientifically_revalidated": True,
            "next_stage_manifest_binding_required": True,
        }
    finally:
        if (
            candidate_temporary_name is not None
            and candidate_fd is not None
            and not candidate_committed
        ):
            try:
                _cleanup_owned_candidate_temporary(
                    parent_fd=candidate_parent_fd,
                    temporary_name=candidate_temporary_name,
                    candidate_fd=candidate_fd,
                    anchor_name=target.name,
                    anchor_identity=anchor_identity,
                    candidate_directory_binding_identity=(
                        candidate_directory_binding_identity
                    ),
                )
            except BaseException:
                # An uncertain private temporary is left for inspection.  The
                # canonical candidate basename remains absent and consumers
                # cannot mistake it for a published anchor.
                pass
        if release_root_fd is not None:
            os.close(release_root_fd)
        if release_parent_fd is not None:
            os.close(release_parent_fd)
        if candidate_fd is not None:
            os.close(candidate_fd)
        os.close(candidate_parent_fd)


__all__ = [
    "CHECKPOINT_DIRNAME",
    "COMPLETION_BASENAME",
    "COMPLETION_FORMAT",
    "FIT_ARRAYS_BASENAME",
    "FIT_MANIFEST_BASENAME",
    "FIT_RECEIPT_BASENAME",
    "FORMAT",
    "anchor_completed_release",
    "inspect_release_state",
    "materialize_authorized_fit",
    "recover_completed_fit",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    recover = subparsers.add_parser("recover")
    recover.add_argument("--output-root", type=Path, required=True)
    recover.add_argument("--freeze-manifest", type=Path, required=True)
    recover.add_argument("--authorization-marker", type=Path, required=True)
    recover.add_argument("--attempt3-root", type=Path, default=linear.ATTEMPT3_ROOT)
    recover.add_argument(
        "--thermal92",
        type=Path,
        default=linear.ROOT
        / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
    )
    recover.add_argument("--release-manifest", type=Path, required=True)
    recover.add_argument("--expected-release-manifest-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "output_root": args.output_root,
        "attempt3_root": args.attempt3_root,
        "thermal92_path": args.thermal92,
        "freeze_manifest": args.freeze_manifest,
        "authorization_marker": args.authorization_marker,
    }
    receipt = recover_completed_fit(
        **common,
        release_manifest=args.release_manifest,
        expected_release_manifest_sha256=args.expected_release_manifest_sha256,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
