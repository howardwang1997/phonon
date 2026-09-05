#!/usr/bin/env python3
"""Hash-bound deployment API for a frozen 65-coefficient R2R-1 readout.

This module deliberately does not fit coefficients and does not load training
or evaluation data.  It accepts only a separately frozen three-file checkpoint
root, revalidates that root and the in-memory coefficient bytes on every public
physics call, and combines the already frozen R2R carrier and parameter basis
as one scalar before differentiating.

The R2R-0 primitive is kept byte-for-byte frozen.  The few private primitive
helpers used here are therefore guarded by the primitive source SHA-256 and by
the primitive's canonical semantic SHA-256 on every call.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Mapping

import numpy as np
import torch
from ase import Atoms

import graphene_r2r_multipolar_background as r2r


FORMAT = "graphene_r2r1_frozen_readout_runtime_v1"
CHECKPOINT_FORMAT = "graphene_r2r1_65_coefficient_checkpoint_v1"
CHECKPOINT_STATUS = "R2R1_65_COEFFICIENT_CHECKPOINT_FROZEN"
QUERY_FORMAT = "graphene_r2r1_frozen_readout_query_v1"
MECHANICS_FORMAT = "graphene_r2r1_frozen_readout_mechanics_v1"

CHECKPOINT_ARRAY_BASENAME = "checkpoint_arrays.npz"
CHECKPOINT_RECEIPT_BASENAME = "checkpoint_receipt.json"
CHECKPOINT_MARKER_BASENAME = "CHECKPOINT_FROZEN"
CHECKPOINT_FILE_SET = {
    CHECKPOINT_ARRAY_BASENAME,
    CHECKPOINT_RECEIPT_BASENAME,
    CHECKPOINT_MARKER_BASENAME,
}

FROZEN_R2R_PRIMITIVE_SHA256 = (
    "4db3665f945783086de5c7a2f6262843bbe7d03df2a32f9153e450200b25064c"
)
FROZEN_R2R_CANONICAL_SHA256 = (
    "e589497c7b9f6a9d4ff0cdcc434cf07804fa0fd732d791dba811d4c7e1c5aecc"
)
FROZEN_ENDPOINT_STATE_SHA256 = (
    "733e223805ed3cbc20b645615a8edbd8d60f3cd25898d7cf629848b0aee29d47"
)
FROZEN_AFFINE_COMPONENT_NAMES_SHA256 = r2r.AFFINE_COMPONENT_NAMES_SHA256

R2R0_FORMAL_STATUS = "R2R0_FORMAL_REPRESENTATION_PRECHECK_PASSED"
R2R0_AGGREGATE_ARRAYS_SHA256 = (
    "caa344eda65448cea0bc0c48c307b8777c0062243777927f3a34c22c3dd6af45"
)
R2R0_AGGREGATE_RECEIPT_SHA256 = (
    "535ab43e1771523dbee0c949caa747f8897d89ba38469d8ef1f9d4fef716a205"
)
R2R0_LAUNCH_RECEIPT_SHA256 = (
    "cadc0616a5426c84e27ae13b3fc1acf90aa11e76225a68b92ec58299a682d675"
)

FORBIDDEN_CHECKPOINT_PATH_TOKENS = (
    "seed1",
    "seed_1",
    "seed-1",
    "seed2",
    "seed_2",
    "seed-2",
    "small",
    "support",
    "held",
    "holdout",
    "reserved",
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_TOP_LEVEL_KEYS = {
    "format",
    "status",
    "array_artifact",
    "physical_p_semantic_sha256",
    "physical_p_raw_sha256",
    "affine_1_plus_p_semantic_sha256",
    "coefficient_layout",
    "fit_provenance",
    "primitive_provenance",
    "r2r0_provenance",
    "safety",
}
_ARRAY_ARTIFACT_KEYS = {"basename", "sha256", "schema"}
_ARRAY_SCHEMA_KEYS = {"physical_p"}
_PHYSICAL_P_SCHEMA_KEYS = {"shape", "dtype"}
_COEFFICIENT_LAYOUT_KEYS = {
    "fixed_carrier_coefficient",
    "parameter_columns",
    "physical_p_only",
    "global_intercept_present",
}
_FIT_PROVENANCE_KEYS = {"fit_manifest_sha256", "fit_receipt_sha256"}
_PRIMITIVE_PROVENANCE_KEYS = {
    "module_sha256",
    "canonical_contract_sha256",
    "endpoint_state_sha256",
    "affine_component_names_sha256",
}
_R2R0_PROVENANCE_KEYS = {
    "formal_status",
    "aggregate_arrays_sha256",
    "aggregate_receipt_sha256",
    "launch_receipt_sha256",
}
_SAFETY_KEYS = {
    "energy_labels_used",
    "encoder_updated",
    "old_R2Q_tail_added",
    "held_or_support_accessed",
    "seed1_or_seed2_accessed",
    "checkpoint_selected_after_held",
}
_CHECKPOINT_FILE_SIZE_LIMITS = {
    CHECKPOINT_ARRAY_BASENAME: 1024 * 1024,
    CHECKPOINT_RECEIPT_BASENAME: 1024 * 1024,
    CHECKPOINT_MARKER_BASENAME: 4096,
}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase 64-hex SHA-256")
    return value


def _strict_json_object(content: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in checkpoint receipt")

    def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate checkpoint receipt key: {key}")
            result[key] = value
        return result

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("checkpoint receipt is not valid UTF-8") from error
    payload = json.loads(
        text, parse_constant=reject_constant, object_pairs_hook=no_duplicate_pairs
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint receipt must be one JSON object")
    return payload


def _compact_checkpoint_path_component(value: str) -> str:
    compact = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    # Treat zero-padded numeric seed aliases as the same restricted seed.
    return re.sub(r"seed0+([12])", r"seed\1", compact)


def _reject_forbidden_checkpoint_components(parts: tuple[str, ...]) -> None:
    compact_parts = [_compact_checkpoint_path_component(part) for part in parts]
    for token in FORBIDDEN_CHECKPOINT_PATH_TOKENS:
        compact_token = _compact_checkpoint_path_component(token)
        if compact_token and any(
            compact_token in component for component in compact_parts
        ):
            raise ValueError(f"forbidden checkpoint path token: {token}")


def _safe_checkpoint_root(root: Path) -> _BoundCheckpointRoot:
    value = Path(root)
    if ".." in PurePath(value).parts:
        raise ValueError("checkpoint path traversal is forbidden")
    _reject_forbidden_checkpoint_components(PurePath(value).parts)
    expanded = value.expanduser()
    # This is deliberately lexical normalization only.  In particular, never
    # call Path.resolve() here: a parent directory can be replaced by a symlink
    # between a path-component check and resolve/open.  The reader below opens
    # every component relative to an already-open directory descriptor.
    absolute = Path(os.path.abspath(os.fspath(expanded)))
    if not absolute.is_absolute() or not absolute.anchor:
        raise ValueError("checkpoint root must have an absolute lexical anchor")
    if any(part in {"", ".", ".."} for part in absolute.parts[1:]):
        raise ValueError("checkpoint root has an unsafe lexical component")
    _reject_forbidden_checkpoint_components(absolute.parts)
    directory_fd, component_identities = _open_checkpoint_root_fd(absolute)
    return _BoundCheckpointRoot(
        path=absolute,
        directory_fd=directory_fd,
        component_identities=component_identities,
    )


def _snapshot_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Metadata identity used to reject mutation around one fd-bound read."""
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass(frozen=True)
class _BoundCheckpointRoot:
    path: Path
    directory_fd: int = field(repr=False)
    component_identities: tuple[tuple[int, ...], ...] = field(repr=False)


def _lexical_absolute_file_path(path: Path, label: str) -> Path:
    raw = Path(path)
    if ".." in PurePath(raw).parts:
        raise ValueError(f"path traversal is forbidden in {label}")
    absolute = Path(os.path.abspath(os.fspath(raw.expanduser())))
    if (
        not absolute.is_absolute()
        or not absolute.anchor
        or len(absolute.parts) < 2
        or any(part in {"", ".", ".."} for part in absolute.parts[1:])
    ):
        raise ValueError(f"{label} has an unsafe lexical path")
    return absolute


def _open_directory_path_fd(
    path: Path,
    label: str,
    *,
    expected_component_identities: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """Open a directory chain with lstat/open/fstat binding at every level."""
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    if expected_component_identities is not None and len(
        expected_component_identities
    ) != len(path.parts):
        raise ValueError(f"{label} component identity chain length changed")
    descriptor = -1
    identities: list[tuple[int, ...]] = []
    try:
        anchor_entry = os.stat(path.anchor, follow_symlinks=False)
        anchor_identity = _snapshot_stat_identity(anchor_entry)
        if expected_component_identities is not None and (
            anchor_identity != expected_component_identities[0]
        ):
            raise ValueError(f"{label} anchor identity changed")
        if not stat.S_ISDIR(anchor_entry.st_mode):
            raise ValueError(f"{label} anchor is not an ordinary directory")
        descriptor = os.open(path.anchor, directory_flags)
        if _snapshot_stat_identity(os.fstat(descriptor)) != anchor_identity:
            raise ValueError(f"{label} anchor changed before open")
        identities.append(anchor_identity)
        for index, component in enumerate(path.parts[1:], start=1):
            entry = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            entry_identity = _snapshot_stat_identity(entry)
            if expected_component_identities is not None and (
                entry_identity != expected_component_identities[index]
            ):
                raise ValueError(f"{label} path component identity changed")
            if not stat.S_ISDIR(entry.st_mode):
                raise ValueError(
                    f"{label} path component is a symlink/non-directory"
                )
            next_descriptor = os.open(
                component, directory_flags, dir_fd=descriptor
            )
            if _snapshot_stat_identity(os.fstat(next_descriptor)) != entry_identity:
                os.close(next_descriptor)
                raise ValueError(f"{label} path component changed before open")
            os.close(descriptor)
            descriptor = next_descriptor
            identities.append(entry_identity)
        return descriptor, tuple(identities)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise ValueError(
            f"{label} path component changed or is a symlink/non-directory"
        ) from error
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _open_lexical_regular_file_fd(
    path: Path,
    label: str,
    *,
    expected_component_identities: tuple[tuple[int, ...], ...] | None = None,
    expected_file_identity: tuple[int, ...] | None = None,
) -> tuple[int, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Open a regular file after binding every parent directory identity."""
    parent_fd, component_identities = _open_directory_path_fd(
        path.parent,
        label,
        expected_component_identities=expected_component_identities,
    )
    file_fd = -1
    try:
        entry = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        entry_identity = _snapshot_stat_identity(entry)
        if (
            expected_file_identity is not None
            and entry_identity != expected_file_identity
        ):
            raise ValueError(f"{label} lexical file identity changed")
        if not stat.S_ISREG(entry.st_mode):
            raise ValueError(f"{label} is not an ordinary file")
        if entry.st_nlink != 1:
            raise ValueError(f"{label} must have one hard link")
        file_flags = os.O_RDONLY
        file_flags |= getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(path.name, file_flags, dir_fd=parent_fd)
        if _snapshot_stat_identity(os.fstat(file_fd)) != entry_identity:
            raise ValueError(f"{label} changed before open")
        result_fd = file_fd
        file_fd = -1
        return result_fd, component_identities, entry_identity
    except OSError as error:
        raise ValueError(f"{label} changed or is not an ordinary file") from error
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def _read_single_link_regular_file_bytes(path: Path, label: str) -> bytes:
    """Read and re-bind one immutable, single-link file fd snapshot."""
    source = _lexical_absolute_file_path(path, label)
    descriptor, component_identities, identity = _open_lexical_regular_file_fd(
        source, label
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not an ordinary file")
        if before.st_nlink != 1:
            raise ValueError(f"{label} must have one hard link")
        if before.st_size > 16 * 1024 * 1024:
            raise ValueError(f"{label} is unexpectedly large")
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            observed_size += len(block)
            if observed_size > 16 * 1024 * 1024:
                raise ValueError(f"{label} grew beyond its size limit")
        if (
            _snapshot_stat_identity(os.fstat(descriptor)) != identity
            or observed_size != before.st_size
        ):
            raise ValueError(f"{label} changed while read")
        verification_fd, _, _ = _open_lexical_regular_file_fd(
            source,
            label,
            expected_component_identities=component_identities,
            expected_file_identity=identity,
        )
        try:
            if _snapshot_stat_identity(os.fstat(verification_fd)) != identity:
                raise ValueError(f"{label} lexical identity changed while read")
        finally:
            os.close(verification_fd)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_checkpoint_root_fd(
    root: Path,
    *,
    expected_component_identities: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """Open and identity-bind every directory in a checkpoint root path."""
    return _open_directory_path_fd(
        root,
        "checkpoint",
        expected_component_identities=expected_component_identities,
    )


def _read_checkpoint_release_bytes(root: _BoundCheckpointRoot) -> dict[str, bytes]:
    """Read one fd-bound snapshot of the exact checkpoint release members."""
    directory_fd = root.directory_fd
    try:
        root_before = os.fstat(directory_fd)
        if not stat.S_ISDIR(root_before.st_mode):
            raise ValueError("checkpoint release root is not an ordinary directory")
        root_identity = _snapshot_stat_identity(root_before)
        if root_identity != root.component_identities[-1]:
            raise ValueError(
                "checkpoint release root identity changed before snapshot"
            )
        observed_before = set(os.listdir(directory_fd))
        if observed_before != CHECKPOINT_FILE_SET:
            raise ValueError(
                "checkpoint release root file set changed before snapshot"
            )
        snapshot: dict[str, bytes] = {}
        for name in sorted(CHECKPOINT_FILE_SET):
            entry_before = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
            if not stat.S_ISREG(entry_before.st_mode):
                raise ValueError(
                    f"checkpoint release member is not an ordinary file: {name}"
                )
            if entry_before.st_nlink != 1:
                raise ValueError(
                    f"checkpoint release member must have one hard link: {name}"
                )
            file_flags = os.O_RDONLY
            file_flags |= getattr(os, "O_CLOEXEC", 0)
            file_flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(name, file_flags, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(
                    f"checkpoint release member changed before open: {name}"
                ) from error
            try:
                before = os.fstat(descriptor)
                if _snapshot_stat_identity(before) != _snapshot_stat_identity(
                    entry_before
                ):
                    raise ValueError(
                        f"checkpoint release member changed before open: {name}"
                    )
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(
                        f"checkpoint release member is not an ordinary file: {name}"
                    )
                if before.st_nlink != 1:
                    raise ValueError(
                        f"checkpoint release member must have one hard link: {name}"
                    )
                limit = _CHECKPOINT_FILE_SIZE_LIMITS[name]
                if before.st_size > limit:
                    raise ValueError(f"checkpoint release member is too large: {name}")
                chunks: list[bytes] = []
                observed_size = 0
                while True:
                    block = os.read(descriptor, min(1024 * 1024, limit + 1))
                    if not block:
                        break
                    chunks.append(block)
                    observed_size += len(block)
                    if observed_size > limit:
                        raise ValueError(
                            f"checkpoint release member grew beyond its limit: {name}"
                        )
                after = os.fstat(descriptor)
                identity_before = _snapshot_stat_identity(before)
                identity_after = _snapshot_stat_identity(after)
                if identity_after != identity_before or observed_size != before.st_size:
                    raise ValueError(
                        f"checkpoint release member changed while read: {name}"
                    )
                snapshot[name] = b"".join(chunks)
            finally:
                os.close(descriptor)
        if set(os.listdir(directory_fd)) != CHECKPOINT_FILE_SET:
            raise ValueError(
                "checkpoint release root file set changed during snapshot"
            )
        root_after = os.fstat(directory_fd)
        if _snapshot_stat_identity(root_after) != root_identity:
            raise ValueError("checkpoint release root identity changed during snapshot")
        # Re-open the same lexical path by the same no-follow walk while the
        # snapshot root descriptor is still held.  This catches an ancestor
        # rename/symlink replacement and binds the final lexical name back to
        # the exact directory from which all three members were read.
        verification_fd, _ = _open_checkpoint_root_fd(
            root.path,
            expected_component_identities=root.component_identities,
        )
        try:
            lexical_after = os.fstat(verification_fd)
            if _snapshot_stat_identity(lexical_after) != root_identity:
                raise ValueError(
                    "checkpoint lexical root identity changed during snapshot"
                )
        finally:
            os.close(verification_fd)
        return snapshot
    finally:
        os.close(directory_fd)


def _physical_p_array(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("physical_p must be an ndarray, not an implicitly cast value")
    if value.shape != (r2r.LINEAR_DESIGN_WIDTH,):
        raise ValueError("physical_p must contain exactly 65 coefficients")
    if value.dtype.str != "<f8":
        raise ValueError("physical_p must be contiguous little-endian FP64")
    if not value.flags.c_contiguous:
        raise ValueError("physical_p must be C-contiguous")
    if not np.all(np.isfinite(value)):
        raise ValueError("physical_p contains a non-finite coefficient")
    result = np.array(value, dtype=np.dtype("<f8"), order="C", copy=True)
    result.setflags(write=False)
    return result


def physical_p_hashes(physical_p: np.ndarray) -> dict[str, str]:
    """Return the three frozen coefficient identities used by deployment."""
    value = _physical_p_array(physical_p)
    values = value.tolist()
    return {
        "physical_p_semantic_sha256": r2r.semantic_sha256(values),
        "physical_p_raw_sha256": _sha256_bytes(value.tobytes(order="C")),
        "affine_1_plus_p_semantic_sha256": r2r.semantic_sha256([1.0, *values]),
    }


def _validate_live_primitive() -> dict[str, str]:
    primitive_bytes = _read_single_link_regular_file_bytes(
        Path(r2r.__file__), "frozen R2R primitive source"
    )
    observed_module = _sha256_bytes(primitive_bytes)
    if observed_module != FROZEN_R2R_PRIMITIVE_SHA256:
        raise ValueError("frozen R2R primitive source SHA-256 changed")
    if r2r.CANONICAL_CONTRACT_SHA256 != FROZEN_R2R_CANONICAL_SHA256:
        raise ValueError("frozen R2R canonical semantic SHA-256 changed")
    if r2r.R2Q_ENDPOINT_STATE_SHA256 != FROZEN_ENDPOINT_STATE_SHA256:
        raise ValueError("frozen endpoint state constant changed")
    if r2r.AFFINE_COMPONENT_NAMES_SHA256 != FROZEN_AFFINE_COMPONENT_NAMES_SHA256:
        raise ValueError("frozen affine-component ordering changed")
    if r2r.LINEAR_DESIGN_WIDTH != 65:
        raise ValueError("frozen R2R parameter width changed")
    for name in (
        "_verified_production_context",
        "_production_energy_components",
        "_source_order_force",
    ):
        if not callable(getattr(r2r, name, None)):
            raise ValueError(f"required hash-bound R2R primitive helper is missing: {name}")
    return {
        "module_sha256": observed_module,
        "canonical_contract_sha256": r2r.CANONICAL_CONTRACT_SHA256,
        "endpoint_state_sha256": r2r.R2Q_ENDPOINT_STATE_SHA256,
        "affine_component_names_sha256": r2r.AFFINE_COMPONENT_NAMES_SHA256,
    }


def checkpoint_receipt_payload(
    physical_p: np.ndarray,
    *,
    array_sha256: str,
    fit_manifest_sha256: str,
    fit_receipt_sha256: str,
) -> dict[str, Any]:
    """Build the exact receipt payload expected beside a frozen coefficient NPZ.

    The caller remains responsible for writing and externally authorizing the
    checkpoint.  This helper performs no fitting and creates no marker.
    """
    value = _physical_p_array(physical_p)
    hashes = physical_p_hashes(value)
    primitive = _validate_live_primitive()
    return {
        "format": CHECKPOINT_FORMAT,
        "status": CHECKPOINT_STATUS,
        "array_artifact": {
            "basename": CHECKPOINT_ARRAY_BASENAME,
            "sha256": _require_hex64(array_sha256, "checkpoint array SHA-256"),
            "schema": {
                "physical_p": {"shape": [65], "dtype": "float64"},
            },
        },
        **hashes,
        "coefficient_layout": {
            "fixed_carrier_coefficient": 1.0,
            "parameter_columns": 65,
            "physical_p_only": True,
            "global_intercept_present": False,
        },
        "fit_provenance": {
            "fit_manifest_sha256": _require_hex64(
                fit_manifest_sha256, "fit manifest SHA-256"
            ),
            "fit_receipt_sha256": _require_hex64(
                fit_receipt_sha256, "fit receipt SHA-256"
            ),
        },
        "primitive_provenance": primitive,
        "r2r0_provenance": {
            "formal_status": R2R0_FORMAL_STATUS,
            "aggregate_arrays_sha256": R2R0_AGGREGATE_ARRAYS_SHA256,
            "aggregate_receipt_sha256": R2R0_AGGREGATE_RECEIPT_SHA256,
            "launch_receipt_sha256": R2R0_LAUNCH_RECEIPT_SHA256,
        },
        "safety": {
            "energy_labels_used": False,
            "encoder_updated": False,
            "old_R2Q_tail_added": False,
            "held_or_support_accessed": False,
            "seed1_or_seed2_accessed": False,
            "checkpoint_selected_after_held": False,
        },
    }


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{label} key set changed: extra={sorted(observed - expected)}, "
            f"missing={sorted(expected - observed)}"
        )
    return value


def _validate_receipt(
    receipt: Mapping[str, Any], physical_p: np.ndarray, array_sha256: str
) -> None:
    _exact_keys(receipt, _CHECKPOINT_TOP_LEVEL_KEYS, "checkpoint receipt")
    if receipt["format"] != CHECKPOINT_FORMAT or receipt["status"] != CHECKPOINT_STATUS:
        raise ValueError("checkpoint receipt format/status changed")

    artifact = _exact_keys(
        receipt["array_artifact"], _ARRAY_ARTIFACT_KEYS, "checkpoint array artifact"
    )
    if artifact["basename"] != CHECKPOINT_ARRAY_BASENAME:
        raise ValueError("checkpoint array basename changed")
    if artifact["sha256"] != array_sha256:
        raise ValueError("checkpoint receipt does not bind the coefficient NPZ")
    schema = _exact_keys(
        artifact["schema"], _ARRAY_SCHEMA_KEYS, "checkpoint array schema"
    )
    item_schema = _exact_keys(
        schema["physical_p"], _PHYSICAL_P_SCHEMA_KEYS, "physical_p schema"
    )
    if item_schema != {"shape": [65], "dtype": "float64"}:
        raise ValueError("physical_p receipt schema changed")

    observed_hashes = physical_p_hashes(physical_p)
    for key, expected in observed_hashes.items():
        if receipt[key] != expected:
            raise ValueError(f"checkpoint coefficient identity changed: {key}")

    layout = _exact_keys(
        receipt["coefficient_layout"],
        _COEFFICIENT_LAYOUT_KEYS,
        "checkpoint coefficient layout",
    )
    if (
        type(layout["fixed_carrier_coefficient"]) is not float
        or layout["fixed_carrier_coefficient"] != 1.0
        or type(layout["parameter_columns"]) is not int
        or layout["parameter_columns"] != 65
        or layout["physical_p_only"] is not True
        or layout["global_intercept_present"] is not False
    ):
        raise ValueError("checkpoint coefficient layout changed")

    fit = _exact_keys(
        receipt["fit_provenance"], _FIT_PROVENANCE_KEYS, "fit provenance"
    )
    for key in _FIT_PROVENANCE_KEYS:
        _require_hex64(fit[key], f"fit provenance {key}")

    primitive = _exact_keys(
        receipt["primitive_provenance"],
        _PRIMITIVE_PROVENANCE_KEYS,
        "primitive provenance",
    )
    if dict(primitive) != _validate_live_primitive():
        raise ValueError("checkpoint primitive/endpoint provenance changed")

    r2r0 = _exact_keys(
        receipt["r2r0_provenance"], _R2R0_PROVENANCE_KEYS, "R2R-0 provenance"
    )
    expected_r2r0 = {
        "formal_status": R2R0_FORMAL_STATUS,
        "aggregate_arrays_sha256": R2R0_AGGREGATE_ARRAYS_SHA256,
        "aggregate_receipt_sha256": R2R0_AGGREGATE_RECEIPT_SHA256,
        "launch_receipt_sha256": R2R0_LAUNCH_RECEIPT_SHA256,
    }
    if dict(r2r0) != expected_r2r0:
        raise ValueError("checkpoint R2R-0 attempt3 provenance changed")

    safety = _exact_keys(receipt["safety"], _SAFETY_KEYS, "checkpoint safety")
    if any(safety[key] is not False for key in _SAFETY_KEYS):
        raise ValueError("checkpoint safety fields must all be exact JSON false")


@dataclass(frozen=True)
class _CheckpointState:
    root: Path
    physical_p: np.ndarray = field(repr=False)
    receipt: dict[str, Any] = field(repr=False)
    receipt_sha256: str
    array_sha256: str
    coefficient_hashes: dict[str, str]


def _read_checkpoint_state(
    root: Path, expected_receipt_sha256: str
) -> _CheckpointState:
    expected = _require_hex64(
        expected_receipt_sha256, "externally expected checkpoint receipt SHA-256"
    )
    bound_release_root = _safe_checkpoint_root(root)
    release_bytes = _read_checkpoint_release_bytes(bound_release_root)
    release_root = bound_release_root.path
    receipt_bytes = release_bytes[CHECKPOINT_RECEIPT_BASENAME]
    receipt_sha = _sha256_bytes(receipt_bytes)
    if receipt_sha != expected:
        raise ValueError("checkpoint receipt differs from the externally frozen SHA-256")
    marker_expected = (CHECKPOINT_STATUS + "\n" + receipt_sha + "\n").encode("ascii")
    if release_bytes[CHECKPOINT_MARKER_BASENAME] != marker_expected:
        raise ValueError("checkpoint marker does not bind the exact receipt")

    array_bytes = release_bytes[CHECKPOINT_ARRAY_BASENAME]
    array_sha = _sha256_bytes(array_bytes)
    try:
        with zipfile.ZipFile(io.BytesIO(array_bytes), mode="r") as archive:
            members = archive.infolist()
            if (
                len(members) != 1
                or members[0].filename != "physical_p.npy"
                or members[0].is_dir()
                or members[0].flag_bits & 0x1
                or members[0].file_size > 4096
            ):
                raise ValueError("checkpoint NPZ must contain only physical_p")
        with np.load(io.BytesIO(array_bytes), allow_pickle=False) as loaded:
            if loaded.files != ["physical_p"]:
                raise ValueError("checkpoint NPZ must contain only physical_p")
            raw_physical_p = loaded["physical_p"]
    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError) and str(error).startswith("checkpoint"):
            raise
        raise ValueError("checkpoint coefficient NPZ is invalid") from error
    # Keep schema validation outside the archive-error wrapper so callers get
    # a precise fail-closed reason for an otherwise readable coefficient array.
    physical_p = _physical_p_array(raw_physical_p)

    receipt = _strict_json_object(receipt_bytes)
    _validate_receipt(receipt, physical_p, array_sha)
    return _CheckpointState(
        root=release_root,
        physical_p=physical_p,
        receipt=receipt,
        receipt_sha256=receipt_sha,
        array_sha256=array_sha,
        coefficient_hashes=physical_p_hashes(physical_p),
    )


@dataclass(frozen=True)
class FrozenReadoutCheckpoint:
    """Immutable handle whose backing release root is revalidated per call."""

    root: Path
    receipt_sha256: str
    array_sha256: str
    physical_p: np.ndarray = field(repr=False)
    coefficient_hashes: Mapping[str, str]
    receipt: Mapping[str, Any] = field(repr=False)

    def validated_physical_p(self) -> np.ndarray:
        # Detect in-memory mutation even if a caller deliberately re-enabled the
        # ndarray write flag, then independently re-open the frozen release root.
        if self.physical_p.flags.writeable:
            raise ValueError("in-memory frozen physical_p write protection was removed")
        in_memory = _physical_p_array(self.physical_p)
        if physical_p_hashes(in_memory) != dict(self.coefficient_hashes):
            raise ValueError("in-memory frozen physical_p was modified")
        current = _read_checkpoint_state(self.root, self.receipt_sha256)
        if (
            current.array_sha256 != self.array_sha256
            or current.coefficient_hashes != dict(self.coefficient_hashes)
            or not np.array_equal(current.physical_p, in_memory)
            or current.receipt != dict(self.receipt)
        ):
            raise ValueError("frozen checkpoint changed after it was loaded")
        return current.physical_p


def load_frozen_readout_checkpoint(
    root: Path, *, expected_receipt_sha256: str
) -> FrozenReadoutCheckpoint:
    """Load a checkpoint only when an external manifest supplies its receipt SHA.

    The external SHA binds this runtime release, but is not a substitute for
    separately authorizing and verifying the fit manifest/receipt before the
    checkpoint is published.
    """
    state = _read_checkpoint_state(root, expected_receipt_sha256)
    return FrozenReadoutCheckpoint(
        root=state.root,
        receipt_sha256=state.receipt_sha256,
        array_sha256=state.array_sha256,
        physical_p=state.physical_p,
        coefficient_hashes=copy.deepcopy(state.coefficient_hashes),
        receipt=copy.deepcopy(state.receipt),
    )


@dataclass
class FrozenReadoutProbe:
    energy_eV: torch.Tensor
    force_source_order_eV_A: torch.Tensor
    coefficients_sha256: str
    checkpoint_receipt_sha256: str
    query_receipt: dict[str, Any]


@dataclass
class FrozenReadoutMechanics:
    energy_eV: torch.Tensor
    force_source_order_eV_A: torch.Tensor
    Hessian_source_order_eV_A2: torch.Tensor
    Hessian_antisymmetry_max_abs_eV_A2: float
    Hessian_translation_ASR_max_abs_eV_A2: float
    coefficients_sha256: str
    checkpoint_receipt_sha256: str
    query_receipt: dict[str, Any]


def _validate_context_receipt(context: Any) -> None:
    receipt = context.receipt
    required_hashes = (
        "endpoint_state_sha256",
        "reference_semantic_sha256",
        "formal_R2O_graph_semantic_sha256",
        "assignment_and_MIC_semantic_sha256",
        "affine_component_names_sha256",
    )
    if not isinstance(receipt, Mapping):
        raise ValueError("verified production context receipt is missing")
    for key in required_hashes:
        _require_hex64(receipt.get(key), f"production context {key}")
    if receipt["endpoint_state_sha256"] != FROZEN_ENDPOINT_STATE_SHA256:
        raise ValueError("production context endpoint state changed")
    if receipt["affine_component_names_sha256"] != FROZEN_AFFINE_COMPONENT_NAMES_SHA256:
        raise ValueError("production context affine column ordering changed")
    if receipt.get("parameter_columns") != 65:
        raise ValueError("production context parameter width changed")
    if receipt.get("fixed_offset_columns") != 1:
        raise ValueError("production context fixed carrier width changed")
    if receipt.get("source_order_force_rows") != 3 * int(context.atom_count):
        raise ValueError("production context source-order force shape changed")
    if receipt.get("production_integration_API_only") is not True:
        raise ValueError("production context no longer asserts the verified API boundary")


def _runtime_receipt(
    context: Any,
    checkpoint: FrozenReadoutCheckpoint,
    *,
    mechanics_path: str,
) -> dict[str, Any]:
    production = copy.deepcopy(dict(context.receipt))
    hashes = dict(checkpoint.coefficient_hashes)
    return {
        "format": QUERY_FORMAT,
        "mechanics_path": mechanics_path,
        "single_scalar_before_autograd": True,
        "fixed_carrier_coefficient": 1.0,
        "parameter_columns": 65,
        "global_intercept_present": False,
        **hashes,
        "checkpoint_receipt_sha256": checkpoint.receipt_sha256,
        "checkpoint_array_sha256": checkpoint.array_sha256,
        "checkpoint_fit_provenance": copy.deepcopy(
            dict(checkpoint.receipt["fit_provenance"])
        ),
        "checkpoint_R2R0_provenance": copy.deepcopy(
            dict(checkpoint.receipt["r2r0_provenance"])
        ),
        "production_context_receipt": production,
        "endpoint_state_sha256": production["endpoint_state_sha256"],
        "reference_semantic_sha256": production["reference_semantic_sha256"],
        "formal_R2O_graph_semantic_sha256": production[
            "formal_R2O_graph_semantic_sha256"
        ],
        "assignment_and_MIC_semantic_sha256": production[
            "assignment_and_MIC_semantic_sha256"
        ],
        "old_R2Q_tail_added": False,
        "canonical_mechanics_constant_modified": False,
        "legacy_66_column_create_graph_used": False,
    }


def _combined_scalar_context(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    checkpoint: FrozenReadoutCheckpoint,
    *,
    device: torch.device | str,
    formal_graph_data: Mapping[str, torch.Tensor] | None,
    graph_mode: str,
    baseline_reference_template: Atoms | None,
    baseline_structure_template: Atoms | None,
    rigid_transform: np.ndarray | None,
) -> tuple[Any, torch.Tensor]:
    if not isinstance(checkpoint, FrozenReadoutCheckpoint):
        raise TypeError("checkpoint must be a FrozenReadoutCheckpoint")
    _validate_live_primitive()
    physical_p = checkpoint.validated_physical_p()
    context = r2r._verified_production_context(
        model,
        structure,
        reference_template,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    _validate_context_receipt(context)
    if context.current_positions_reference_order.dtype != torch.float64:
        raise ValueError("R2R-1 frozen readout requires FP64 production mechanics")
    fixed, parameters = r2r._production_energy_components(context)
    if fixed.ndim != 0 or parameters.shape != (65,):
        raise ValueError("R2R fixed/parameter scalar layout changed")
    if fixed.dtype != torch.float64 or parameters.dtype != torch.float64:
        raise ValueError("R2R fixed/parameter scalars must be FP64")
    if not bool(torch.isfinite(fixed).all()) or not bool(torch.isfinite(parameters).all()):
        raise ValueError("R2R fixed/parameter scalar contains a non-finite value")
    coefficients = torch.as_tensor(
        np.array(physical_p, copy=True),
        dtype=torch.float64,
        device=context.current_positions_reference_order.device,
    )
    energy = fixed + torch.dot(parameters, coefficients)
    if energy.ndim != 0 or not bool(torch.isfinite(energy).all()):
        raise ValueError("R2R-1 combined scalar is invalid")
    return context, energy


def production_frozen_readout_energy_force(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    checkpoint: FrozenReadoutCheckpoint,
    *,
    device: torch.device | str,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
) -> FrozenReadoutProbe:
    """Evaluate arbitrary frozen physical coefficients through one scalar E/F path."""
    context, energy = _combined_scalar_context(
        model,
        structure,
        reference_template,
        checkpoint,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    force_reference = -torch.autograd.grad(
        energy, context.current_positions_reference_order, create_graph=False
    )[0]
    force_source = r2r._source_order_force(force_reference, context.assignment)
    if force_source.shape != (context.atom_count, 3) or not bool(
        torch.isfinite(force_source).all()
    ):
        raise ValueError("R2R-1 source-order force is invalid")
    receipt = _runtime_receipt(
        context,
        checkpoint,
        mechanics_path="frozen_fixed_carrier_plus_arbitrary_65_physical_p_EF",
    )
    return FrozenReadoutProbe(
        energy_eV=energy,
        force_source_order_eV_A=force_source,
        coefficients_sha256=checkpoint.coefficient_hashes[
            "physical_p_semantic_sha256"
        ],
        checkpoint_receipt_sha256=checkpoint.receipt_sha256,
        query_receipt=receipt,
    )


def production_frozen_readout_energy_force_hessian(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    checkpoint: FrozenReadoutCheckpoint,
    *,
    device: torch.device | str,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
) -> FrozenReadoutMechanics:
    """Evaluate E/F/full-H for arbitrary frozen p with one retained scalar graph."""
    context, energy = _combined_scalar_context(
        model,
        structure,
        reference_template,
        checkpoint,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    live = context.current_positions_reference_order
    force_reference = -torch.autograd.grad(
        energy, live, create_graph=True, retain_graph=True
    )[0]
    hessian_rows = []
    for component in force_reference.reshape(-1):
        derivative = torch.autograd.grad(component, live, retain_graph=True)[0]
        hessian_rows.append(-derivative.reshape(-1))
    reference_hessian = torch.stack(hessian_rows)

    reference_to_source = torch.as_tensor(
        context.assignment.reference_to_source,
        dtype=torch.long,
        device=reference_hessian.device,
    )
    source_components = (
        3 * reference_to_source[:, None]
        + torch.arange(3, device=reference_hessian.device)[None, :]
    ).reshape(-1)
    hessian = torch.empty_like(reference_hessian)
    hessian[source_components[:, None], source_components[None, :]] = (
        reference_hessian
    )
    force_source = r2r._source_order_force(force_reference, context.assignment)
    if not bool(torch.isfinite(force_source).all()) or not bool(
        torch.isfinite(hessian).all()
    ):
        raise ValueError("R2R-1 full mechanics contains a non-finite value")
    atom_count = context.atom_count
    if hessian.shape != (3 * atom_count, 3 * atom_count):
        raise ValueError("R2R-1 source-order Hessian shape changed")
    blocks = hessian.reshape(atom_count, 3, atom_count, 3)
    antisymmetry = float(torch.max(torch.abs(hessian - hessian.T)).detach())
    translation_asr = float(torch.max(torch.abs(blocks.sum(dim=2))).detach())
    if not math.isfinite(antisymmetry) or not math.isfinite(translation_asr):
        raise ValueError("R2R-1 Hessian diagnostics are non-finite")
    receipt = _runtime_receipt(
        context,
        checkpoint,
        mechanics_path=(
            "frozen_fixed_carrier_plus_arbitrary_65_physical_p_scalar_first_full_Hessian"
        ),
    )
    return FrozenReadoutMechanics(
        energy_eV=energy,
        force_source_order_eV_A=force_source,
        Hessian_source_order_eV_A2=hessian,
        Hessian_antisymmetry_max_abs_eV_A2=antisymmetry,
        Hessian_translation_ASR_max_abs_eV_A2=translation_asr,
        coefficients_sha256=checkpoint.coefficient_hashes[
            "physical_p_semantic_sha256"
        ],
        checkpoint_receipt_sha256=checkpoint.receipt_sha256,
        query_receipt=receipt,
    )


__all__ = [
    "CHECKPOINT_ARRAY_BASENAME",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_MARKER_BASENAME",
    "CHECKPOINT_RECEIPT_BASENAME",
    "CHECKPOINT_STATUS",
    "FORMAT",
    "FrozenReadoutCheckpoint",
    "FrozenReadoutMechanics",
    "FrozenReadoutProbe",
    "checkpoint_receipt_payload",
    "load_frozen_readout_checkpoint",
    "physical_p_hashes",
    "production_frozen_readout_energy_force",
    "production_frozen_readout_energy_force_hessian",
]
