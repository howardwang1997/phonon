#!/usr/bin/env python3
"""Frozen R2R-1 conditional linear-readout primitives.

This module is the only implementation of the 65-parameter R2R-1 ridge
problem.  It deliberately separates label-blind input/design validation from
the authorized label loader and fit.  The frozen R2Q encoder is never updated.
"""
from __future__ import annotations

import hashlib
import ctypes
import ctypes.util
import io
import json
import math
import os
import platform
import re
import secrets
import select
import shlex
import socket
import stat
import sys
import tempfile
import time
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import numpy as np
from ase import Atoms

import graphene_r2r0_formal as r2r0
import launch_graphene_r2r0_formal as r2r0_launcher
import graphene_r2r_multipolar_background as r2r


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "graphene_r2r1_conditional_linear_readout_v1"
SHARD_FORMAT = "graphene_r2r1_label_feature_shard_v1"
AGGREGATE_FORMAT = "graphene_r2r1_nested_oof_aggregate_v2"

ATTEMPT3_ROOT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/formal_r2r0_three_host_seed83_attempt3"
)
R2R1_RESULT_PARENT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
ATTEMPT1_FIT_OUTPUT_ROOT = (
    R2R1_RESULT_PARENT
    / "R2R1_conditional_linear_readout_fit_seed83_20260826"
)
ATTEMPT1_CONTROL_ROOT = (
    R2R1_RESULT_PARENT / "R2R1_conditional_linear_readout_control_20260826"
)
ATTEMPT1_FREEZE_MANIFEST = ATTEMPT1_CONTROL_ROOT / "freeze_manifest.json"
ATTEMPT1_AUTHORIZATION_MARKER = ATTEMPT1_CONTROL_ROOT / "R2R1_FORMAL_GO"
ATTEMPT1_CANDIDATE_ROOT = (
    R2R1_RESULT_PARENT / "R2R1_conditional_linear_readout_candidate_20260826"
)
ATTEMPT1_RELEASE_MANIFEST = ATTEMPT1_CANDIDATE_ROOT / "release_manifest.json"
FAILED_ATTEMPT2_FIT_OUTPUT_ROOT = (
    R2R1_RESULT_PARENT
    / "R2R1_conditional_linear_readout_fit_seed83_attempt2_20260826"
)
FAILED_ATTEMPT2_CONTROL_ROOT = (
    R2R1_RESULT_PARENT
    / "R2R1_conditional_linear_readout_control_attempt2_20260826"
)
FAILED_ATTEMPT2_FREEZE_MANIFEST = (
    FAILED_ATTEMPT2_CONTROL_ROOT / "freeze_manifest.json"
)
FAILED_ATTEMPT2_AUTHORIZATION_MARKER = (
    FAILED_ATTEMPT2_CONTROL_ROOT / "R2R1_FORMAL_GO"
)
FAILED_ATTEMPT2_CANDIDATE_ROOT = (
    R2R1_RESULT_PARENT
    / "R2R1_conditional_linear_readout_candidate_attempt2_20260826"
)
FAILED_ATTEMPT2_RELEASE_MANIFEST = (
    FAILED_ATTEMPT2_CANDIDATE_ROOT / "release_manifest.json"
)
RECOMMENDED_FIT_OUTPUT_ROOT = (
    R2R1_RESULT_PARENT
    / "R2R1_conditional_linear_readout_fit_seed83_attempt2_retry1_20260826"
)
RECOMMENDED_CONTROL_ROOT = (
    R2R1_RESULT_PARENT
    / "R2R1_conditional_linear_readout_control_attempt2_retry1_20260826"
)
RECOMMENDED_FREEZE_MANIFEST = RECOMMENDED_CONTROL_ROOT / "freeze_manifest.json"
RECOMMENDED_AUTHORIZATION_MARKER = RECOMMENDED_CONTROL_ROOT / "R2R1_FORMAL_GO"
RECOMMENDED_CANDIDATE_ROOT = (
    R2R1_RESULT_PARENT
    / "R2R1_conditional_linear_readout_candidate_attempt2_retry1_20260826"
)
RECOMMENDED_RELEASE_MANIFEST = RECOMMENDED_CANDIDATE_ROOT / "release_manifest.json"
RECOMMENDED_THERMAL92 = (
    ROOT / "data/graphene_r2o_taylor_null_core/train_thermal.xyz"
)
ATTEMPT3_AGGREGATE_RECEIPT_SHA256 = (
    "535ab43e1771523dbee0c949caa747f8897d89ba38469d8ef1f9d4fef716a205"
)
ATTEMPT3_AGGREGATE_ARRAYS_SHA256 = (
    "caa344eda65448cea0bc0c48c307b8777c0062243777927f3a34c22c3dd6af45"
)
ATTEMPT3_FREEZE_MANIFEST_SHA256 = (
    "a9a1732af38c4513242fbbe7d23c6e5fd6fdd091ee27cf62a8600544bb8cf73f"
)
ATTEMPT3_LAUNCH_RECEIPT_SHA256 = (
    "cadc0616a5426c84e27ae13b3fc1acf90aa11e76225a68b92ec58299a682d675"
)
ATTEMPT3_SHARD_RECEIPT_SHA256 = (
    "51b90efe6fc1b5b785f271b120fbc8a1caa12aefe61f7a9406998b88417b26fd",
    "7075bce680bc30d9096de1ed97b72799350730830e7cb2af221b992ee6431b7f",
    "0a7d881faa444d4f6fab51815b331589fdb2d9562da229cfafc2a9e4cd3f6cc0",
)
ATTEMPT3_SHARD_THERMAL_ARRAYS_SHA256 = (
    "f6f4f7e8c596c931a3bafdf056be42a4682642b465c511368350c6f2e31a8948",
    "219e35f3470f8496d68a5182aa4981f0415032148a88a04ddb0629e29817f586",
    "66b03b0e4d567c608aa669cea0ecb4f337da22afd75b3512b541393b4668deeb",
)
R2R_CANONICAL_SHA256 = (
    "e589497c7b9f6a9d4ff0cdcc434cf07804fa0fd732d791dba811d4c7e1c5aecc"
)
R2R0_FORMAL_CONTRACT_SHA256 = (
    "cc407b3e1b18ad00097fc433fbe9458100790b8150dac599908cfcf7d2f4ddea"
)
FROZEN_READOUT_SOURCE_SHA256 = (
    "356c8506582a201be11dfd58fe3acd1c079e843dba568b73132f405519ff77bb"
)
FROZEN_READOUT_TEST_SHA256 = (
    "2ff64cb565d5a6ce2550898f2b390573ede31cc6623b368765c84675af3056c0"
)
R2R0_FORMAL_SOURCE_SHA256 = (
    "e1013323a9a9664e885a806362d0e4312fb0947b96a6ab3a44385d8a9f907769"
)
R2R0_LAUNCHER_SOURCE_SHA256 = (
    "32c6af5475954880b4bc9192763f948609421fded9cd9fb17f387fac55ccb358"
)
R2R_PRIMITIVE_SOURCE_SHA256 = (
    "4db3665f945783086de5c7a2f6262843bbe7d03df2a32f9153e450200b25064c"
)
THERMAL92_FILE_SHA256 = (
    "cc2e9c68d418fba996e8ac89c8156c8adf00768406f6171aaa57f0a9bd2720c1"
)
THERMAL92_FILE_SIZE_BYTES = 2_634_387
THERMAL_STRUCTURE_IDENTITY_SHA256 = (
    "143397937ae328700f7d4cd873e33423030189e99ebee27144e423a9e1890511"
)

EXPECTED_AGGREGATE_ARRAY_RAW_SHA256 = {
    "thermal_global_index": (
        "d55134adfc6b7022c523e96db6e8645053495abd0d09500d45ae4657ac3b64f1"
    ),
    "thermal_fixed_energy_eV": (
        "68f12611817fd8b427b05d7e98d58c5ec3d6cc97845e5adb7bd29a7801eda2da"
    ),
    "thermal_fixed_force_eV_A": (
        "c6459749ba0eea8c6a1e5cae963fd3cea8c8023fee65a5c3a1fb3f9a79726d7a"
    ),
    "thermal_parameter_energy_design_eV": (
        "223894c3749a6ec15d000d0ca6253c58c541143b8f6af8cbd6d490a2c59b29f7"
    ),
    "thermal_parameter_force_design_eV_A": (
        "76278c36f35b269873140bf750177eb2e52513800ca9970817492dae4c6831e2"
    ),
}
EXPECTED_LABEL_RAW_SHA256 = {
    "REF_forces_all92": (
        "75da2423b0584112c2fc85c0788eefae4ba6263d0e712a719491cad68439c65b"
    ),
    "REF_forces_E50": (
        "6b418481503ae5457c7755b717283c923f2369a30e67c1982d9b917396d1d7b0"
    ),
    "REF_forces_T300": (
        "f5818695082b9af9d8b271953d1cacfd3d51c41db036de26b6deba5fed6e64b0"
    ),
    "REF_forces_T600": (
        "18e25a9b2499d86c780e1bfe5f46b38b5a43d9ee9e645ef180e9a396f72a19ef"
    ),
    "APRIME_mode_real": (
        "8a94f82d559da0e41dfcd8834140f9937e6076c3b7a024d9ae1305dccc632bbb"
    ),
    "APRIME_mode_imag": (
        "4514e35b7a98d710ffdb66766240c9e3d29a946c63cd6a62a09e0acee91fb312"
    ),
    "APRIME_coordinates": (
        "d02189db4494c4021a3b44c632655db4a096abcd640ad0b280f233d9f34d0807"
    ),
    "FOUNDATION_BASE_forces": (
        "525265d529ed19e7222ac692d924a34d7dd665f36f44c72c0585adee22332112"
    ),
    "FROZEN_Q6_forces": (
        "fc518fc5f187bea6b5cbf7149bcaa5eec5977e12eb74c296df1c97c72a0372ee"
    ),
}

THERMAL_GROUP_RANGES = {
    "E50_seed0": (0, 20),
    "T300": (20, 56),
    "T600": (56, 92),
}
THERMAL_CONFIG_TYPES = {
    "E50_seed0": "r2o_exact_e50_seed0_train",
    "T300": "r2o_auxiliary_T300_train",
    "T600": "r2o_auxiliary_T600_train",
}
GROUP_MASSES = {"E50_seed0": 0.50, "T300": 0.25, "T600": 0.25}
FORCE_SCALE_EV_A = 0.030
ALPHA_GRID = tuple(float(10.0**exponent) for exponent in range(-10, 3))
LINEAR_WIDTH = 65
ZERO_COLUMN_RELATIVE_RMS = 1.0e-12
SCALED_CONDITION_LIMIT = 1.0e8
SCORE_ROUND_DIGITS = 12
GATE_THRESHOLDS = {
    "force_RMSE_meV_A": 30.0,
    "force_max_abs_meV_A": 200.0,
    "E50_Aprime_RMS_meV_A": 15.0,
    "E50_slope_relative_abs_error": 0.05,
}

FOLD_GLOBAL_INDICES = tuple(
    tuple(
        list(range(5 * fold, 5 * (fold + 1)))
        + list(range(20 + 9 * fold, 20 + 9 * (fold + 1)))
        + list(range(56 + 9 * fold, 56 + 9 * (fold + 1)))
    )
    for fold in range(4)
)
FOLD_GLOBAL_INDEX_SHA256 = (
    "b24b8a36ca6b04698e81a5e4f110059ba7415a10138c54669557058bd838ffef",
    "351db85669de7303e25e38b51e4289032550a6bd3d4578aba57ad838a37196f1",
    "616490f0e7aea68f74c7a99674f084421d74015568c186e1a2b2c40ba840f14b",
    "14668a31b188a45c46f69e0818ebac08a96fff2c41bab5e8dcdb333bff00a920",
)
FOLD_CONTRACT_SHA256 = (
    "f2dd459f6d2ba9633843ba68ebddf09ee1dac0b74e78bf9756ac8183b5d52bc6"
)
EXPECTED_OUTER_TRAIN69_CONDITION = (
    17597.33862720773,
    17562.743672260356,
    17758.54555501026,
    18056.66252377068,
)
EXPECTED_INNER_TRAIN46_CONDITION_RANGE = (
    17308.237617401574,
    18328.356889983883,
)
EXPECTED_SPLIT_MIN_RELATIVE_COLUMN_RMS_RANGE = (
    0.0466546090510323,
    0.048107482626134304,
)
R2R0_SHARD_GLOBAL_INDICES = tuple(tuple(range(shard, 92, 3)) for shard in range(3))
ATTEMPT3_TREE_ITEM_LIMIT = 710
ATTEMPT3_TREE_TOTAL_BYTES_LIMIT = 64 * 1024 * 1024
ATTEMPT3_TREE_SINGLE_FILE_BYTES_LIMIT = 16 * 1024 * 1024
ATTEMPT3_JSON_FILE_BYTES_LIMIT = 2 * 1024 * 1024
ATTEMPT3_TERMINAL_MARKER_BYTES_LIMIT = 4096
ATTEMPT3_EXIT_CODE_BYTES_LIMIT = 64
SOURCE_CLOSURE_SINGLE_FILE_BYTES_LIMIT = 2 * 1024 * 1024

E50_PROPERTY_SCHEMA = (
    ("species", "S", 1),
    ("pos", "R", 3),
    ("REF_forces", "R", 3),
    ("CORE_TARGET_forces", "R", 3),
    ("DFT_TOTAL_forces", "R", 3),
    ("FOUNDATION_BASE_forces", "R", 3),
    ("FROZEN_Q6_forces", "R", 3),
    ("CURRENT_S0_forces_DIAGNOSTIC_NOT_DEPLOYED", "R", 3),
    ("APRIME_mode_real", "R", 3),
    ("APRIME_mode_imag", "R", 3),
)
AUXILIARY_PROPERTY_SCHEMA = (
    ("species", "S", 1),
    ("pos", "R", 3),
    ("REF_forces", "R", 3),
    ("TOTAL_forces", "R", 3),
    ("LONG_RANGE_forces", "R", 3),
    ("SHORT_RANGE_TARGET_forces", "R", 3),
    ("BASE_forces", "R", 3),
    ("CORE_TARGET_forces", "R", 3),
)
E50_NUMERIC_WHITELIST = {
    "pos",
    "REF_forces",
    "FOUNDATION_BASE_forces",
    "FROZEN_Q6_forces",
    "APRIME_mode_real",
    "APRIME_mode_imag",
}
AUXILIARY_NUMERIC_WHITELIST = {"pos", "REF_forces"}
HEADER_NUMERIC_WHITELIST = {
    "APRIME_coordinate_real_A",
    "APRIME_coordinate_imag_A",
}
FORBIDDEN_LABEL_PATH_TOKENS = (
    "seed1",
    "seed2",
    "small",
    "small12",
    "small_gate",
    "support",
    "reserved",
    "outer_fold",
    "held",
    "holdout",
    "valid_e50",
    "full_report",
)

R2R1_SOURCE_PATHS = {
    "linear_core": ROOT / "scripts/smearing_kink/graphene_r2r1_linear_readout.py",
    "frozen_readout": ROOT / "scripts/smearing_kink/graphene_r2r1_frozen_readout.py",
    "run_cli": ROOT / "scripts/smearing_kink/run_graphene_r2r1_linear_readout.py",
    "aggregate_cli": ROOT
    / "scripts/smearing_kink/aggregate_graphene_r2r1_linear_readout.py",
    "launcher": ROOT / "scripts/smearing_kink/launch_graphene_r2r1_linear_readout.py",
    "tests": ROOT / "tests/test_graphene_r2r1_linear_readout.py",
    "frozen_readout_tests": ROOT / "tests/test_graphene_r2r1_frozen_readout.py",
    "contract_doc": ROOT
    / "docs/GRAPHENE_R2R1_LINEAR_READOUT_CONTRACT_2026-08-26.md",
}
R2R1_DEPENDENCY_PATHS = {
    "r2r0_formal": ROOT / "scripts/smearing_kink/graphene_r2r0_formal.py",
    "r2r0_launcher_provenance_only": ROOT
    / "scripts/smearing_kink/launch_graphene_r2r0_formal.py",
    "r2r_primitive": ROOT
    / "scripts/smearing_kink/graphene_r2r_multipolar_background.py",
}
EXPECTED_R2R1_DEPENDENCY_SHA256 = {
    "r2r0_formal": R2R0_FORMAL_SOURCE_SHA256,
    "r2r0_launcher_provenance_only": R2R0_LAUNCHER_SOURCE_SHA256,
    "r2r_primitive": R2R_PRIMITIVE_SOURCE_SHA256,
}


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _json_type_exact_equal(left: object, right: object) -> bool:
    """Compare already-JSON-compatible values without bool/int coercion."""
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def semantic_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    """Return the immutable identity fields used for a same-file check."""
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode),
        "st_nlink": int(value.st_nlink),
        "st_uid": int(value.st_uid),
        "st_gid": int(value.st_gid),
        "st_size": int(value.st_size),
        "st_mtime_ns": int(value.st_mtime_ns),
        "st_ctime_ns": int(value.st_ctime_ns),
    }


def _owned_regular_identity(value: os.stat_result) -> dict[str, int]:
    """Return the portable identity fields for an R2R-1-owned regular file.

    Darwin may update ``st_ctime_ns`` asynchronously for filesystem-managed
    metadata without changing the owned bytes or publication binding.  Owned
    records therefore freeze every regular-file field except ctime; external
    inputs continue to use :func:`_stat_identity` and its full nine fields.
    """
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode),
        "st_nlink": int(value.st_nlink),
        "st_uid": int(value.st_uid),
        "st_gid": int(value.st_gid),
        "st_size": int(value.st_size),
        "st_mtime_ns": int(value.st_mtime_ns),
    }


DARWIN_MANAGED_XATTRS = frozenset(
    {"com.apple.provenance", "com.apple.decmpfs"}
)
OWNED_XATTR_NAME_BYTES_LIMIT = 64 * 1024


def _authoritative_owned_xattrs(
    descriptor: int, *, label: str
) -> dict[str, dict[str, Any]]:
    """Reject non-managed xattrs without reading Darwin-managed values."""
    if sys.platform != "darwin":
        return {}
    library_name = ctypes.util.find_library("c")
    if not library_name:
        raise RuntimeError("Darwin fd-bound xattr APIs are unavailable")
    libc = ctypes.CDLL(library_name, use_errno=True)
    flistxattr = getattr(libc, "flistxattr", None)
    if flistxattr is None:
        raise RuntimeError("Darwin fd-bound xattr APIs are unavailable")
    flistxattr.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    flistxattr.restype = ctypes.c_ssize_t
    name_buffer = ctypes.create_string_buffer(OWNED_XATTR_NAME_BYTES_LIMIT)
    observed_size = int(
        flistxattr(
            descriptor,
            name_buffer,
            OWNED_XATTR_NAME_BYTES_LIMIT,
            0,
        )
    )
    if observed_size < 0:
        observed_errno = ctypes.get_errno()
        raise OSError(observed_errno, os.strerror(observed_errno))
    if observed_size > OWNED_XATTR_NAME_BYTES_LIMIT:
        raise ValueError(f"{label} xattr name list exceeds its fixed limit")
    raw_names = bytes(name_buffer.raw[:observed_size])
    encoded_names = [item for item in raw_names.split(b"\0") if item]
    if len(set(encoded_names)) != len(encoded_names):
        raise ValueError(f"{label} xattr names are not unique")
    names: set[str] = set()
    for encoded_name in encoded_names:
        try:
            names.add(encoded_name.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} xattr name is not UTF-8") from error
    unknown = sorted(names - DARWIN_MANAGED_XATTRS)
    if unknown:
        raise ValueError(f"{label} has unsupported authoritative xattrs: {unknown}")
    return {}


class _HeldRegularWriteGuard:
    """Detect payload writes while a held authority set is being closed.

    Owned identities deliberately exclude Darwin ``ctime`` because APFS may
    update filesystem-managed metadata asynchronously.  That makes a final
    stat insufficient to detect a same-inode write whose size and mtime were
    restored.  On Darwin, a private kqueue watches each held vnode for data or
    binding events while the caller performs its bounded byte closure.  The
    filter intentionally excludes ``NOTE_ATTRIB`` so provenance/decmpfs ctime
    drift is not promoted back into authority.  Other platforms use the full
    held stat (including ctime) as a conservative fallback.

    Watch descriptors are private duplicates or no-follow opens owned by this
    object.  Callers must register every regular member before a multi-file
    closure and leave the context before an intentional rename/write.
    """

    def __init__(self, label: str):
        self._label = label
        self._records: list[dict[str, Any]] = []
        if sys.platform == "darwin":
            if not hasattr(select, "kqueue"):
                raise RuntimeError("Darwin held-vnode write guard is unavailable")
            self._queue = select.kqueue()
        else:
            self._queue = None
        self._closed = False

    def __enter__(self) -> "_HeldRegularWriteGuard":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            if exc_type is None and not self._closed:
                self.assert_clean()
        finally:
            self.close()
        return False

    def _register(
        self,
        descriptor: int,
        *,
        expected_identity: Mapping[str, int],
        owned: bool,
        label: str,
    ) -> None:
        if self._closed:
            raise RuntimeError("R2R-1 write guard is already closed")
        held = os.dup(descriptor)
        try:
            observed = os.fstat(held)
            identity_fn = _owned_regular_identity if owned else _stat_identity
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or not _json_type_exact_equal(
                    identity_fn(observed), dict(expected_identity)
                )
            ):
                raise ValueError(f"{label} changed before write-event guard")
            if owned:
                _authoritative_owned_xattrs(held, label=label)
            full_baseline = _stat_identity(observed)
            if self._queue is not None:
                fflags = (
                    select.KQ_NOTE_WRITE
                    | select.KQ_NOTE_EXTEND
                    | select.KQ_NOTE_DELETE
                    | select.KQ_NOTE_RENAME
                    | select.KQ_NOTE_REVOKE
                )
                event = select.kevent(
                    held,
                    filter=select.KQ_FILTER_VNODE,
                    flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                    fflags=fflags,
                )
                self._queue.control([event], 0, 0)
            self._records.append(
                {
                    "fd": held,
                    "expected_identity": dict(expected_identity),
                    "full_baseline": full_baseline,
                    "owned": owned,
                    "label": label,
                }
            )
        except BaseException:
            os.close(held)
            raise

    def _register_directory(
        self,
        descriptor: int,
        *,
        expected_identity: Mapping[str, int],
        label: str,
    ) -> None:
        if self._closed:
            raise RuntimeError("R2R-1 write guard is already closed")
        held = os.dup(descriptor)
        try:
            observed = os.fstat(held)
            if (
                not stat.S_ISDIR(observed.st_mode)
                or not _json_type_exact_equal(
                    _directory_binding_identity(observed), dict(expected_identity)
                )
            ):
                raise ValueError(f"{label} changed before directory event guard")
            full_baseline = _stat_identity(observed)
            if self._queue is not None:
                fflags = (
                    select.KQ_NOTE_WRITE
                    | select.KQ_NOTE_EXTEND
                    | select.KQ_NOTE_DELETE
                    | select.KQ_NOTE_RENAME
                    | select.KQ_NOTE_REVOKE
                )
                event = select.kevent(
                    held,
                    filter=select.KQ_FILTER_VNODE,
                    flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                    fflags=fflags,
                )
                self._queue.control([event], 0, 0)
            self._records.append(
                {
                    "fd": held,
                    "expected_identity": dict(expected_identity),
                    "full_baseline": full_baseline,
                    "owned": False,
                    "directory": True,
                    "label": label,
                }
            )
        except BaseException:
            os.close(held)
            raise

    def watch_owned_descriptor(
        self,
        descriptor: int,
        *,
        expected_identity: Mapping[str, int],
        label: str,
    ) -> None:
        self._register(
            descriptor,
            expected_identity=expected_identity,
            owned=True,
            label=label,
        )

    def watch_external_descriptor(
        self,
        descriptor: int,
        *,
        expected_identity: Mapping[str, int],
        label: str,
    ) -> None:
        self._register(
            descriptor,
            expected_identity=expected_identity,
            owned=False,
            label=label,
        )

    def watch_directory_descriptor(
        self,
        descriptor: int,
        *,
        expected_identity: Mapping[str, int],
        label: str,
    ) -> None:
        self._register_directory(
            descriptor,
            expected_identity=expected_identity,
            label=label,
        )

    def watch_owned_at(
        self,
        directory_fd: int,
        name: str,
        *,
        expected_identity: Mapping[str, int],
        label: str,
    ) -> None:
        descriptor = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
        try:
            self.watch_owned_descriptor(
                descriptor,
                expected_identity=expected_identity,
                label=label,
            )
        finally:
            os.close(descriptor)

    def _drain_events(self) -> None:
        if self._queue is None or not self._records:
            return
        events = self._queue.control(None, len(self._records), 0)
        if events:
            labels_by_fd = {record["fd"]: record["label"] for record in self._records}
            changed = sorted(
                {
                    labels_by_fd.get(int(event.ident), self._label)
                    for event in events
                }
            )
            raise ValueError(
                f"{self._label} observed a held payload write: {changed}"
            )

    def assert_clean(self) -> None:
        """Drain after validation, rebind each vnode, then drain once more."""
        self._drain_events()
        for record in self._records:
            observed = os.fstat(record["fd"])
            identity_fn = (
                _directory_binding_identity
                if record.get("directory")
                else (_owned_regular_identity if record["owned"] else _stat_identity)
            )
            if not _json_type_exact_equal(
                identity_fn(observed), record["expected_identity"]
            ):
                raise ValueError(
                    f"{record['label']} changed during held write-event closure"
                )
            if record["owned"] and not record.get("directory"):
                _authoritative_owned_xattrs(record["fd"], label=record["label"])
            if self._queue is None and not _json_type_exact_equal(
                _stat_identity(observed), record["full_baseline"]
            ):
                raise ValueError(
                    f"{record['label']} changed during portable write closure"
                )
        # A write triggered from the final identity/xattr phase is observable
        # here; only the following nonblocking drain syscall itself remains an
        # indivisible kernel boundary.
        self._drain_events()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue is not None:
            self._queue.close()
        for record in reversed(self._records):
            os.close(record["fd"])
        self._records.clear()


def _watch_owned_snapshot_fd(
    guard: _HeldRegularWriteGuard,
    root_fd: int,
    snapshot: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    """Register every regular member of one flattened held-dir snapshot."""
    guard.watch_directory_descriptor(
        root_fd,
        expected_identity=_directory_binding_identity(os.fstat(root_fd)),
        label=f"{label} root directory",
    )
    watched_directories: set[str] = set()
    for relative, record in sorted(snapshot.items()):
        if record.get("kind") != "file":
            continue
        parts = relative.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            raise ValueError(f"{label} contains an invalid relative path")
        directory_fd = os.dup(root_fd)
        try:
            prefix: list[str] = []
            for component in parts[:-1]:
                prefix.append(component)
                directory_record = snapshot.get("/".join(prefix))
                if directory_record is None or directory_record.get("kind") != "directory":
                    raise ValueError(f"{label} directory schema changed")
                before = os.stat(
                    component, dir_fd=directory_fd, follow_symlinks=False
                )
                expected_directory = directory_record.get("identity")
                if (
                    stat.S_ISLNK(before.st_mode)
                    or not stat.S_ISDIR(before.st_mode)
                    or not _json_type_exact_equal(
                        _directory_binding_identity(before), expected_directory
                    )
                ):
                    raise ValueError(f"{label} directory binding changed")
                child_fd = os.open(
                    component,
                    _DIRECTORY_FLAGS | _NOFOLLOW,
                    dir_fd=directory_fd,
                )
                if not _json_type_exact_equal(
                    _directory_binding_identity(os.fstat(child_fd)),
                    expected_directory,
                ):
                    os.close(child_fd)
                    raise ValueError(f"{label} directory changed across open")
                directory_relative = "/".join(prefix)
                if directory_relative not in watched_directories:
                    guard.watch_directory_descriptor(
                        child_fd,
                        expected_identity=expected_directory,
                        label=f"{label} {directory_relative}",
                    )
                    watched_directories.add(directory_relative)
                os.close(directory_fd)
                directory_fd = child_fd
            guard.watch_owned_at(
                directory_fd,
                parts[-1],
                expected_identity=record["identity"],
                label=f"{label} {relative}",
            )
        finally:
            os.close(directory_fd)


def _directory_binding_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode),
        "st_uid": int(value.st_uid),
        "st_gid": int(value.st_gid),
    }


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _open_directory_chain(
    path: Path,
    label: str,
    *,
    expected_chain: Sequence[Mapping[str, int]] | None = None,
) -> tuple[Path, int, tuple[dict[str, int], ...]]:
    """Open a lexical directory path one component at a time.

    An expected identity is checked immediately after each component is
    opened, before any child of a replaced component can be inspected.
    """
    source = (
        _lexical_path(path, label)
        if expected_chain is not None
        else _reject_path(path, label)
    )
    if not source.is_absolute():
        raise ValueError(f"{label} must be an absolute directory path")
    descriptor = os.open(source.anchor, _DIRECTORY_FLAGS | _NOFOLLOW)
    try:
        identities: list[dict[str, int]] = [
            _directory_binding_identity(os.fstat(descriptor))
        ]
        if expected_chain is not None:
            if not expected_chain or not _json_type_exact_equal(
                identities[0], dict(expected_chain[0])
            ):
                raise ValueError(f"{label} anchor identity changed")
        for index, component in enumerate(source.parts[1:], start=1):
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"{label} contains a non-directory component")
            if expected_chain is not None and (
                index >= len(expected_chain)
                or not _json_type_exact_equal(
                    _directory_binding_identity(before),
                    dict(expected_chain[index]),
                )
            ):
                raise ValueError(f"{label} directory identity changed")
            next_descriptor = os.open(
                component,
                _DIRECTORY_FLAGS | _NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                after = os.fstat(next_descriptor)
                identity = _directory_binding_identity(after)
                if identity != _directory_binding_identity(before):
                    raise ValueError(f"{label} directory changed across stat/open")
                if expected_chain is not None:
                    if index >= len(expected_chain) or not _json_type_exact_equal(
                        identity, dict(expected_chain[index])
                    ):
                        raise ValueError(f"{label} directory identity changed")
                os.close(descriptor)
            except BaseException:
                os.close(next_descriptor)
                raise
            descriptor = next_descriptor
            identities.append(identity)
        if expected_chain is not None and len(identities) != len(expected_chain):
            raise ValueError(f"{label} directory depth changed")
        return source, descriptor, tuple(identities)
    except BaseException:
        os.close(descriptor)
        raise


def _verify_directory_chain(
    path: Path,
    label: str,
    expected_chain: Sequence[Mapping[str, int]],
) -> None:
    _source, descriptor, _identities = _open_directory_chain(
        path, label, expected_chain=expected_chain
    )
    os.close(descriptor)


def ensure_bound_directory(
    path: Path,
    label: str,
    *,
    expected_parent_chain: Sequence[Mapping[str, int]] | None = None,
) -> Path:
    """Create one final directory component through a held parent fd."""
    target = _reject_path(path, label, must_exist=False)
    parent, parent_fd, parent_chain = _open_directory_chain(
        target.parent,
        f"{label} parent",
        expected_chain=expected_parent_chain,
    )
    del parent
    try:
        created = False
        try:
            before = os.stat(
                target.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            os.mkdir(target.name, 0o700, dir_fd=parent_fd)
            created = True
            before = os.stat(
                target.name, dir_fd=parent_fd, follow_symlinks=False
            )
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"{label} is not a directory")
        child_fd = os.open(
            target.name,
            _DIRECTORY_FLAGS | _NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            if _stat_identity(os.fstat(child_fd)) != _stat_identity(before):
                raise ValueError(f"{label} changed across stat/open")
            if created:
                os.fsync(parent_fd)
        finally:
            os.close(child_fd)
        _verify_directory_chain(target.parent, f"{label} final parent", parent_chain)
    finally:
        os.close(parent_fd)
    return target


def _open_bound_regular_file(
    path: Path,
    label: str,
    *,
    expected_parent_chain: Sequence[Mapping[str, int]] | None = None,
    expected_file_identity: Mapping[str, int] | None = None,
) -> tuple[
    Path,
    int,
    int,
    dict[str, int],
    tuple[dict[str, int], ...],
]:
    source = (
        _lexical_path(path, label)
        if expected_parent_chain is not None
        else _reject_path(path, label)
    )
    if not source.is_absolute() or len(source.parts) < 2:
        raise ValueError(f"{label} must be an absolute file path")
    _parent, parent_fd, parent_chain = _open_directory_chain(
        source.parent,
        f"{label} parent",
        expected_chain=expected_parent_chain,
    )
    file_fd: int | None = None
    try:
        before = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if before.st_nlink != 1:
            raise ValueError(f"{label} must have exactly one hard link")
        if expected_file_identity is not None and not _json_type_exact_equal(
            _stat_identity(before), dict(expected_file_identity)
        ):
            raise ValueError(f"{label} file identity changed before open")
        file_fd = os.open(
            source.name, os.O_RDONLY | _NOFOLLOW, dir_fd=parent_fd
        )
        identity = _stat_identity(os.fstat(file_fd))
        if identity != _stat_identity(before):
            raise ValueError(f"{label} changed across stat/open")
        if expected_file_identity is not None and not _json_type_exact_equal(
            identity, dict(expected_file_identity)
        ):
            raise ValueError(f"{label} file identity changed before read")
        return source, file_fd, parent_fd, identity, parent_chain
    except BaseException:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)
        raise


def _finish_bound_regular_file(
    source: Path,
    file_fd: int,
    parent_fd: int,
    expected_identity: Mapping[str, int],
    parent_chain: Sequence[Mapping[str, int]],
    label: str,
) -> None:
    """Validate through held fds and a no-follow lexical re-walk, then close."""
    error: BaseException | None = None
    try:
        if not _json_type_exact_equal(
            _stat_identity(os.fstat(file_fd)), dict(expected_identity)
        ):
            raise ValueError(f"{label} changed while it was read")
        parent_observed = os.stat(
            source.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if stat.S_ISLNK(parent_observed.st_mode) or not _json_type_exact_equal(
            _stat_identity(parent_observed), dict(expected_identity)
        ):
            raise ValueError(f"{label} entry changed in its bound parent")
        # Re-walk before releasing the original fds.  A replaced directory is
        # rejected before any of its descendants are accessed.
        _parent, verify_parent_fd, _chain = _open_directory_chain(
            source.parent,
            f"{label} final parent binding",
            expected_chain=parent_chain,
        )
        try:
            final_observed = os.stat(
                source.name, dir_fd=verify_parent_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(final_observed.st_mode) or not _json_type_exact_equal(
                _stat_identity(final_observed), dict(expected_identity)
            ):
                raise ValueError(f"{label} lexical identity changed")
        finally:
            os.close(verify_parent_fd)
    except BaseException as caught:
        error = caught
    finally:
        try:
            os.close(file_fd)
        finally:
            os.close(parent_fd)
    if error is not None:
        raise error


def _read_regular_file_once(
    path: Path,
    label: str,
    *,
    expected_parent_chain: Sequence[Mapping[str, int]] | None = None,
    expected_file_identity: Mapping[str, int] | None = None,
    size_limit: int,
) -> tuple[bytes, dict[str, int]]:
    """Read one regular file from one O_NOFOLLOW fd and bind its identity.

    The caller must already have applied any path-specific denylist.  Component
    traversal uses directory fds, so changing an intermediate component to a
    symlink cannot redirect this read after lexical validation.
    """
    if type(size_limit) is not int or size_limit <= 0:
        raise ValueError(f"{label} size limit must be a positive integer")
    source, file_fd, parent_fd, before_identity, parent_chain = (
        _open_bound_regular_file(
            path,
            label,
            expected_parent_chain=expected_parent_chain,
            expected_file_identity=expected_file_identity,
        )
    )
    chunks: list[bytes] = []
    total = 0
    try:
        if before_identity["st_size"] > size_limit:
            raise ValueError(f"{label} exceeds its size limit")
        while True:
            request = min(1024 * 1024, size_limit + 1 - total)
            chunk = os.read(file_fd, request)
            if not chunk:
                break
            total += len(chunk)
            if total > size_limit:
                raise ValueError(f"{label} exceeds its size limit while read")
            chunks.append(chunk)
    finally:
        _finish_bound_regular_file(
            source,
            file_fd,
            parent_fd,
            before_identity,
            parent_chain,
            label,
        )
    return b"".join(chunks), before_identity


def _sha256_fd_exact_size(descriptor: int, expected_size: int, label: str) -> str:
    """Hash exactly one frozen-size fd without ever reading past size + 1."""
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError(f"{label} expected size must be a nonnegative integer")
    digest = hashlib.sha256()
    total = 0
    while True:
        block = os.read(
            descriptor, min(1024 * 1024, expected_size + 1 - total)
        )
        if not block:
            break
        total += len(block)
        if total > expected_size:
            raise ValueError(f"{label} grew beyond its frozen size while read")
        digest.update(block)
    if total != expected_size:
        raise ValueError(f"{label} byte count differs from its frozen size")
    return digest.hexdigest()


class _ExpectedSizeRawReader(io.RawIOBase):
    """Non-owning fd reader enforcing one exact cumulative byte count."""

    def __init__(self, descriptor: int, expected_size: int, label: str) -> None:
        super().__init__()
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError(f"{label} expected size must be a nonnegative integer")
        self._descriptor = descriptor
        self._expected_size = expected_size
        self._label = label
        self._total = 0

    @property
    def total(self) -> int:
        return self._total

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        if not buffer:
            return 0
        block = os.read(
            self._descriptor,
            min(len(buffer), self._expected_size + 1 - self._total),
        )
        if not block:
            if self._total != self._expected_size:
                raise ValueError(
                    f"{self._label} byte count differs from its frozen size"
                )
            return 0
        self._total += len(block)
        if self._total > self._expected_size:
            raise ValueError(
                f"{self._label} grew beyond its frozen size while parsed"
            )
        buffer[: len(block)] = block
        return len(block)


def _read_regular_file_at_fd(
    directory_fd: int,
    name: str,
    label: str,
    *,
    size_limit: int,
    expected_identity: Mapping[str, int] | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Bounded same-dirfd read used for manifest/marker sibling transactions."""
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    identity = _stat_identity(before)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > size_limit
        or (
            expected_identity is not None
            and not _json_type_exact_equal(identity, dict(expected_identity))
        )
    ):
        raise ValueError(f"{label} is not the expected bounded regular file")
    descriptor = os.open(
        name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd
    )
    try:
        if _stat_identity(os.fstat(descriptor)) != identity:
            raise ValueError(f"{label} changed across stat/open")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, size_limit + 1 - total))
            if not block:
                break
            total += len(block)
            if total > size_limit:
                raise ValueError(f"{label} exceeds its size limit")
            chunks.append(block)
        if _stat_identity(os.fstat(descriptor)) != identity:
            raise ValueError(f"{label} changed while read")
        rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _stat_identity(rebound) != identity:
            raise ValueError(f"{label} entry changed while read")
        return b"".join(chunks), identity
    finally:
        os.close(descriptor)


def _read_owned_regular_file_at_fd(
    directory_fd: int,
    name: str,
    label: str,
    *,
    size_limit: int,
    expected_identity: Mapping[str, int] | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Bounded sibling read for an R2R-1-owned regular inode."""
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    identity = _owned_regular_identity(before)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > size_limit
        or (
            expected_identity is not None
            and not _json_type_exact_equal(identity, dict(expected_identity))
        )
    ):
        raise ValueError(f"{label} is not the expected bounded owned file")
    descriptor = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
    try:
        if _owned_regular_identity(os.fstat(descriptor)) != identity:
            raise ValueError(f"{label} changed across stat/open")
        with _HeldRegularWriteGuard(f"{label} read boundary") as write_guard:
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
                    raise ValueError(f"{label} exceeds its size limit")
                chunks.append(block)
            _authoritative_owned_xattrs(descriptor, label=label)
            if _owned_regular_identity(os.fstat(descriptor)) != identity:
                raise ValueError(f"{label} changed while read")
            rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _owned_regular_identity(rebound) != identity:
                raise ValueError(f"{label} entry changed while read")
            raw = b"".join(chunks)
        return raw, identity
    finally:
        os.close(descriptor)


def _open_regular_file_fd(
    path: Path,
    label: str,
    *,
    expected_parent_chain: Sequence[Mapping[str, int]] | None = None,
    expected_file_identity: Mapping[str, int] | None = None,
) -> tuple[Path, int, int, dict[str, int], tuple[dict[str, int], ...]]:
    """Open a single-link regular file without following any path symlink."""
    return _open_bound_regular_file(
        path,
        label,
        expected_parent_chain=expected_parent_chain,
        expected_file_identity=expected_file_identity,
    )


def _finish_regular_file_fd(
    source: Path,
    file_fd: int,
    parent_fd: int,
    expected_identity: Mapping[str, int],
    parent_chain: Sequence[Mapping[str, int]],
    label: str,
) -> None:
    """Close a streamed fd after validating its fd and path identity."""
    _finish_bound_regular_file(
        source,
        file_fd,
        parent_fd,
        expected_identity,
        parent_chain,
        label,
    )


def _require_bound_file_unchanged(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_identity: Mapping[str, int],
    expected_parent_chain: Sequence[Mapping[str, int]],
) -> None:
    source, file_fd, parent_fd, identity, parent_chain = _open_bound_regular_file(
        path,
        label,
        expected_parent_chain=expected_parent_chain,
        expected_file_identity=expected_identity,
    )
    expected_size = expected_identity.get("st_size")
    try:
        digest = _sha256_fd_exact_size(file_fd, expected_size, label)
    finally:
        _finish_bound_regular_file(
            source,
            file_fd,
            parent_fd,
            identity,
            parent_chain,
            label,
        )
    if digest != expected_sha256:
        raise PermissionError(f"{label} content changed after it was authorized")
    if not _json_type_exact_equal(identity, dict(expected_identity)):
        raise PermissionError(f"{label} identity changed after it was authorized")


def regular_file_binding(path: Path, label: str) -> dict[str, Any]:
    source, file_fd, parent_fd, identity, parent_chain = _open_bound_regular_file(
        path, label
    )
    _finish_bound_regular_file(
        source, file_fd, parent_fd, identity, parent_chain, label
    )
    return {
        "path": str(source),
        "file_identity": identity,
        "parent_chain": [dict(item) for item in parent_chain],
    }


def regular_file_metadata_binding(path: Path, label: str) -> dict[str, Any]:
    """Bind a file by parent/name metadata without opening or reading it."""
    source = _lexical_path(
        path, label, forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS
    )
    _parent, parent_fd, parent_chain = _open_directory_chain(
        source.parent, f"{label} parent"
    )
    try:
        before = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise ValueError(f"{label} is not one single-link regular file")
        identity = _stat_identity(before)
        after = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if _stat_identity(after) != identity:
            raise ValueError(f"{label} changed during metadata binding")
        _verify_directory_chain(
            source.parent, f"{label} final parent binding", parent_chain
        )
    finally:
        os.close(parent_fd)
    return {
        "path": str(source),
        "file_identity": identity,
        "parent_chain": [dict(item) for item in parent_chain],
        "final_file_opened": False,
        "payload_bytes_read": 0,
    }


def directory_metadata_binding(path: Path, label: str) -> dict[str, Any]:
    source, descriptor, chain = _open_directory_chain(path, label)
    try:
        identity = _directory_binding_identity(os.fstat(descriptor))
        _verify_directory_chain(source, f"{label} final binding", chain)
    finally:
        os.close(descriptor)
    return {
        "path": str(source),
        "directory_identity": identity,
        "chain": [dict(item) for item in chain],
        "payload_bytes_read": 0,
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_attempt2_path_partition() -> dict[str, str]:
    """Freeze attempt2-retry1 as fresh siblings disjoint from prior attempts."""
    current = {
        "fit": _lexical_path(
            RECOMMENDED_FIT_OUTPUT_ROOT,
            "R2R-1 attempt2 fit root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
        "control": _lexical_path(
            RECOMMENDED_CONTROL_ROOT,
            "R2R-1 attempt2 control root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
        "candidate": _lexical_path(
            RECOMMENDED_CANDIDATE_ROOT,
            "R2R-1 attempt2 candidate root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
    }
    superseded = {
        "attempt1_fit": _lexical_path(
            ATTEMPT1_FIT_OUTPUT_ROOT,
            "R2R-1 attempt1 forensic fit root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
        "attempt1_control": _lexical_path(
            ATTEMPT1_CONTROL_ROOT,
            "R2R-1 attempt1 forensic control root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
        "attempt1_candidate": _lexical_path(
            ATTEMPT1_CANDIDATE_ROOT,
            "R2R-1 attempt1 forensic candidate root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
        "failed_attempt2_fit": _lexical_path(
            FAILED_ATTEMPT2_FIT_OUTPUT_ROOT,
            "R2R-1 failed attempt2 forensic fit root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
        "failed_attempt2_control": _lexical_path(
            FAILED_ATTEMPT2_CONTROL_ROOT,
            "R2R-1 failed attempt2 forensic control root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
        "failed_attempt2_candidate": _lexical_path(
            FAILED_ATTEMPT2_CANDIDATE_ROOT,
            "R2R-1 failed attempt2 forensic candidate root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
    }
    if any(
        "attempt2_retry1" not in path.name.casefold() for path in current.values()
    ):
        raise PermissionError(
            "R2R-1 production roots must explicitly name attempt2_retry1"
        )
    if len({str(path) for path in current.values()}) != len(current):
        raise PermissionError("R2R-1 attempt2 publication roots must be distinct")
    for left_name, left in current.items():
        for right_name, right in current.items():
            if left_name < right_name and _paths_overlap(left, right):
                raise PermissionError(
                    "R2R-1 attempt2 publication roots overlap each other"
                )
    protected = {
        **superseded,
        "attempt3": _lexical_path(
            ATTEMPT3_ROOT,
            "R2R-1 frozen attempt3 root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
        "thermal92": _lexical_path(
            RECOMMENDED_THERMAL92,
            "R2R-1 frozen thermal92 path",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        ),
        **{
            f"source:{name}": _lexical_path(
                path,
                f"R2R-1 source path {name}",
                forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
            )
            for name, path in R2R1_SOURCE_PATHS.items()
        },
        **{
            f"dependency:{name}": _lexical_path(
                path,
                f"R2R-1 dependency path {name}",
                forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
            )
            for name, path in R2R1_DEPENDENCY_PATHS.items()
        },
    }
    for write in current.values():
        for protected_path in protected.values():
            if _paths_overlap(write, protected_path):
                raise PermissionError(
                    "R2R-1 attempt2 publication root overlaps attempt1/input/source"
                )
    return {name: str(path) for name, path in {**current, **superseded}.items()}


def _result_parent_binding_from_fd(
    *,
    parent: Path,
    descriptor: int,
    chain: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    """Capture both absent publication children from one held parent fd."""
    fit_target = _lexical_path(
        RECOMMENDED_FIT_OUTPUT_ROOT,
        "R2R-1 fit-output-root manifest absence binding",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    candidate_target = _lexical_path(
        RECOMMENDED_CANDIDATE_ROOT,
        "R2R-1 candidate-root manifest absence binding",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if fit_target.parent != candidate_target.parent:
        raise PermissionError(
            "R2R-1 fit and candidate roots must share one result parent"
        )
    if fit_target.name == candidate_target.name:
        raise PermissionError("R2R-1 publication basenames must be distinct")
    if parent != fit_target.parent:
        raise PermissionError("held result parent differs from publication parent")
    for target, label in (
        (fit_target, "fit output root"),
        (candidate_target, "candidate root"),
    ):
        try:
            os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"R2R-1 {label} must be absent when the manifest is frozen"
            )
    identity = _directory_binding_identity(os.fstat(descriptor))
    if not _json_type_exact_equal(identity, dict(chain[-1])):
        raise ValueError("R2R-1 held result-parent identity changed")
    _verify_directory_chain(parent, "R2R-1 final result-parent binding", chain)
    return {
        "path": str(parent),
        "directory_identity": identity,
        "chain": [dict(item) for item in chain],
        "fit_output_child_basename": fit_target.name,
        "candidate_child_basename": candidate_target.name,
        "fit_output_child_absent_at_freeze": True,
        "candidate_child_absent_at_freeze": True,
        "payload_bytes_read": 0,
    }


def result_parent_binding() -> dict[str, Any]:
    """Bind the one shared parent and both absent publication basenames."""
    target = _lexical_path(
        RECOMMENDED_FIT_OUTPUT_ROOT,
        "R2R-1 shared result-parent target",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    parent, descriptor, chain = _open_directory_chain(
        target.parent, "R2R-1 shared result parent"
    )
    try:
        return _result_parent_binding_from_fd(
            parent=parent, descriptor=descriptor, chain=chain
        )
    finally:
        os.close(descriptor)


def raw_array_sha256(array: np.ndarray, dtype: str) -> str:
    value = np.asarray(array, dtype=np.dtype(dtype), order="C")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def runtime_environment_receipt() -> dict[str, Any]:
    configuration = io.StringIO()
    with redirect_stdout(configuration):
        np.show_config()
    package_versions = {}
    for distribution in ("numpy", "scipy", "ase"):
        try:
            package_versions[distribution] = package_version(distribution)
        except PackageNotFoundError:
            package_versions[distribution] = None
    thread_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYTHONHASHSEED",
    )
    config_text = configuration.getvalue()
    return {
        "format": "graphene_r2r1_runtime_environment_v1",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "package_versions": package_versions,
        "numpy_show_config_sha256": hashlib.sha256(
            config_text.encode("utf-8")
        ).hexdigest(),
        "numpy_show_config": config_text,
        "thread_environment": {key: os.environ.get(key) for key in thread_keys},
    }


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    expected_parent_chain: Sequence[Mapping[str, int]] | None = None,
) -> None:
    """Create one new file atomically inside a held, no-follow parent fd."""
    target = (
        _lexical_path(path, "R2R-1 atomic output")
        if expected_parent_chain is not None
        else _reject_path(path, "R2R-1 atomic output", must_exist=False)
    )
    _parent, parent_fd, parent_chain = _open_directory_chain(
        target.parent,
        "R2R-1 atomic output parent",
        expected_chain=expected_parent_chain,
    )
    temporary = f".{target.name}.{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    published = False
    completed = False
    try:
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"atomic output already exists: {target}")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(bytes(payload))
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while materializing R2R-1 output")
            view = view[written:]
        os.fsync(descriptor)
        temporary_identity = _stat_identity(os.fstat(descriptor))
        if not stat.S_ISREG(temporary_identity["st_mode"]):
            raise ValueError("atomic output temporary is not regular")
        # link(2) is an atomic no-replace publication within the bound parent.
        os.link(
            temporary,
            target.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary, dir_fd=parent_fd)
        observed = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        current = os.fstat(descriptor)
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_dev != current.st_dev
            or observed.st_ino != current.st_ino
            or observed.st_size != len(payload)
        ):
            raise ValueError("atomic output identity changed during publication")
        os.fsync(parent_fd)
        _verify_directory_chain(
            target.parent, "R2R-1 atomic output final parent", parent_chain
        )
        completed = True
    finally:
        try:
            if descriptor is not None:
                current = os.fstat(descriptor)
                if published and not completed:
                    try:
                        observed = os.stat(
                            target.name, dir_fd=parent_fd, follow_symlinks=False
                        )
                        current = os.fstat(descriptor)
                        if (
                            observed.st_dev == current.st_dev
                            and observed.st_ino == current.st_ino
                        ):
                            os.unlink(target.name, dir_fd=parent_fd)
                    except FileNotFoundError:
                        pass
                try:
                    temporary_observed = os.stat(
                        temporary, dir_fd=parent_fd, follow_symlinks=False
                    )
                    if (
                        temporary_observed.st_dev == current.st_dev
                        and temporary_observed.st_ino == current.st_ino
                    ):
                        os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
        finally:
            try:
                if descriptor is not None:
                    os.close(descriptor)
            finally:
                os.close(parent_fd)


def atomic_write_json(
    path: Path,
    payload: object,
    *,
    expected_parent_chain: Sequence[Mapping[str, int]] | None = None,
) -> None:
    atomic_write_bytes(
        path,
        canonical_json_bytes(payload),
        expected_parent_chain=expected_parent_chain,
    )


def atomic_write_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    expected_parent_chain: Sequence[Mapping[str, int]] | None = None,
) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **{key: np.asarray(value) for key, value in arrays.items()})
    atomic_write_bytes(
        path,
        buffer.getvalue(),
        expected_parent_chain=expected_parent_chain,
    )


def _reject_forbidden_components(
    components: Sequence[str], label: str, forbidden_tokens: Sequence[str]
) -> None:
    normalized_components = []
    lowered_components = [component.casefold() for component in components]
    for component in components:
        compact = "".join(
            character for character in component.casefold() if character.isalnum()
        )
        normalized_components.append(re.sub(r"seed0+([12])", r"seed\1", compact))
    for token in forbidden_tokens:
        if any(token.casefold() in component for component in lowered_components):
            raise ValueError(f"forbidden path token {token!r} in {label}")
        normalized_token = "".join(
            character for character in token.casefold() if character.isalnum()
        )
        normalized_token = re.sub(r"seed0+([12])", r"seed\1", normalized_token)
        if normalized_token and any(
            normalized_token in component for component in normalized_components
        ):
            raise ValueError(f"normalized forbidden path token {token!r} in {label}")


def _lexical_path(
    path: Path,
    label: str,
    *,
    forbidden_tokens: Sequence[str] = (),
) -> Path:
    raw = Path(path)
    if ".." in PurePath(raw).parts:
        raise ValueError(f"path traversal is forbidden in {label}")
    _reject_forbidden_components(PurePath(raw).parts, label, forbidden_tokens)
    lexical = raw if raw.is_absolute() else Path.cwd() / raw
    return Path(os.path.abspath(str(lexical)))


def _reject_path(
    path: Path,
    label: str,
    *,
    forbidden_tokens: Sequence[str] = (),
    must_exist: bool = True,
) -> Path:
    """Apply lexical policy plus a no-follow component audit."""
    absolute = _lexical_path(
        path, label, forbidden_tokens=forbidden_tokens
    )
    # Perform a no-follow lexical audit for immediate policy feedback.  This
    # is not used as authorization: every reader repeats and binds the walk.
    descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS | _NOFOLLOW)
    try:
        for index, component in enumerate(absolute.parts[1:], start=1):
            final = index == len(absolute.parts) - 1
            try:
                observed = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                if not must_exist and final:
                    break
                raise
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"symlink path component is forbidden in {label}")
            if not final:
                if not stat.S_ISDIR(observed.st_mode):
                    raise ValueError(f"non-directory path component in {label}")
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_FLAGS | _NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    if _stat_identity(os.fstat(next_descriptor)) != _stat_identity(
                        observed
                    ):
                        raise ValueError(f"path component changed in {label}")
                    os.close(descriptor)
                except BaseException:
                    os.close(next_descriptor)
                    raise
                descriptor = next_descriptor
    finally:
        os.close(descriptor)
    return absolute


def group_indices(group: str) -> np.ndarray:
    start, stop = THERMAL_GROUP_RANGES[group]
    return np.arange(start, stop, dtype=np.int64)


def group_for_global_index(index: int) -> str:
    for group, (start, stop) in THERMAL_GROUP_RANGES.items():
        if start <= index < stop:
            return group
    raise ValueError(f"thermal global index out of range: {index}")


def validate_fold_contract() -> dict[str, Any]:
    flattened = [index for fold in FOLD_GLOBAL_INDICES for index in fold]
    if sorted(flattened) != list(range(92)) or len(set(flattened)) != 92:
        raise RuntimeError("R2R-1 outer folds do not partition thermal92")
    per_fold = []
    for fold in FOLD_GLOBAL_INDICES:
        counts = {
            group: sum(group_for_global_index(index) == group for index in fold)
            for group in THERMAL_GROUP_RANGES
        }
        if counts != {"E50_seed0": 5, "T300": 9, "T600": 9}:
            raise RuntimeError("R2R-1 fold strata changed")
        per_fold.append(counts)
    hashes = [semantic_sha256(list(fold)) for fold in FOLD_GLOBAL_INDICES]
    if tuple(hashes) != FOLD_GLOBAL_INDEX_SHA256:
        raise RuntimeError("R2R-1 fold index hashes changed")
    if semantic_sha256([list(fold) for fold in FOLD_GLOBAL_INDICES]) != FOLD_CONTRACT_SHA256:
        raise RuntimeError("R2R-1 full fold contract hash changed")
    return {
        "folds": [list(fold) for fold in FOLD_GLOBAL_INDICES],
        "fold_index_sha256": hashes,
        "fold_contract_sha256": FOLD_CONTRACT_SHA256,
        "per_fold_counts": per_fold,
        "partition_exact": True,
    }


def _json_object_from_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return payload


def validate_attempt3_aggregate(
    receipt_path: Path, arrays_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    receipt_path = _reject_path(receipt_path, "attempt3 aggregate receipt")
    arrays_path = _reject_path(arrays_path, "attempt3 aggregate arrays")
    receipt_raw, _receipt_identity = _read_regular_file_once(
        receipt_path,
        "attempt3 aggregate receipt",
        size_limit=ATTEMPT3_JSON_FILE_BYTES_LIMIT,
    )
    arrays_raw, _arrays_identity = _read_regular_file_once(
        arrays_path,
        "attempt3 aggregate arrays",
        size_limit=ATTEMPT3_TREE_SINGLE_FILE_BYTES_LIMIT,
    )
    if hashlib.sha256(receipt_raw).hexdigest() != ATTEMPT3_AGGREGATE_RECEIPT_SHA256:
        raise ValueError("attempt3 aggregate receipt SHA changed")
    if hashlib.sha256(arrays_raw).hexdigest() != ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("attempt3 aggregate arrays SHA changed")
    receipt = _json_object_from_bytes(receipt_raw, "attempt3 aggregate receipt")
    required = {
        "status": "R2R0_FORMAL_REPRESENTATION_PRECHECK_PASSED",
        "representation_precheck_pass": True,
        "numerically_inconclusive": False,
        "borderline": False,
        "fit_performed": False,
        "can_authorize_fit_or_training": False,
        "held_or_support_access": False,
        "seed1_or_seed2_access": False,
        "force_labels_used": False,
        "energy_labels_used": False,
        "formal_contract_sha256": R2R0_FORMAL_CONTRACT_SHA256,
        "frozen_R2R_canonical_sha256": R2R_CANONICAL_SHA256,
        "arrays_sha256": ATTEMPT3_AGGREGATE_ARRAYS_SHA256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"attempt3 aggregate safety binding changed: {key}")
    if receipt.get("configuration_identity", {}).get(
        "thermal_structure_identity_sha256"
    ) != THERMAL_STRUCTURE_IDENTITY_SHA256:
        raise ValueError("attempt3 thermal structure identity changed")
    with np.load(io.BytesIO(arrays_raw), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    expected_shapes = {
        "thermal_global_index": (92,),
        "thermal_fixed_energy_eV": (92, 1),
        "thermal_fixed_force_eV_A": (92, 72, 3),
        "thermal_parameter_energy_design_eV": (92, 65),
        "thermal_parameter_force_design_eV_A": (92, 72, 3, 65),
    }
    selected: dict[str, np.ndarray] = {}
    for key, shape in expected_shapes.items():
        value = arrays.get(key)
        if value is None or value.shape != shape:
            raise ValueError(f"attempt3 aggregate array shape changed: {key}")
        dtype = "<i8" if key == "thermal_global_index" else "<f8"
        value = np.asarray(value, dtype=np.dtype(dtype), order="C")
        if key != "thermal_global_index" and not np.all(np.isfinite(value)):
            raise ValueError(f"attempt3 aggregate contains non-finite values: {key}")
        if raw_array_sha256(value, dtype) != EXPECTED_AGGREGATE_ARRAY_RAW_SHA256[key]:
            raise ValueError(f"attempt3 aggregate raw-array SHA changed: {key}")
        selected[key] = value
    if not np.array_equal(selected["thermal_global_index"], np.arange(92)):
        raise ValueError("attempt3 thermal indices are not exact 0..91")
    return receipt, selected


def validate_attempt3_shard(
    shard_id: int, receipt_path: Path, arrays_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if shard_id not in (0, 1, 2):
        raise ValueError("R2R-1 requires attempt3 shard id 0, 1, or 2")
    receipt_path = _reject_path(receipt_path, f"attempt3 shard{shard_id} receipt")
    arrays_path = _reject_path(arrays_path, f"attempt3 shard{shard_id} thermal arrays")
    receipt_raw, _receipt_identity = _read_regular_file_once(
        receipt_path,
        f"attempt3 shard{shard_id} receipt",
        size_limit=ATTEMPT3_JSON_FILE_BYTES_LIMIT,
    )
    arrays_raw, _arrays_identity = _read_regular_file_once(
        arrays_path,
        f"attempt3 shard{shard_id} thermal arrays",
        size_limit=ATTEMPT3_TREE_SINGLE_FILE_BYTES_LIMIT,
    )
    if hashlib.sha256(receipt_raw).hexdigest() != ATTEMPT3_SHARD_RECEIPT_SHA256[shard_id]:
        raise ValueError("attempt3 shard receipt SHA changed")
    if hashlib.sha256(arrays_raw).hexdigest() != ATTEMPT3_SHARD_THERMAL_ARRAYS_SHA256[shard_id]:
        raise ValueError("attempt3 shard thermal-array SHA changed")
    receipt = _json_object_from_bytes(
        receipt_raw, f"attempt3 shard{shard_id} receipt"
    )
    if receipt.get("shard_id") != shard_id or receipt.get("status") != r2r0.STATUS_SHARD:
        raise ValueError("attempt3 shard identity/status changed")
    for key in (
        "force_labels_used",
        "energy_labels_used",
        "can_authorize_fit_or_training",
    ):
        if receipt.get(key) is not False:
            raise ValueError(f"attempt3 shard safety field changed: {key}")
    with np.load(io.BytesIO(arrays_raw), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    expected_indices = np.asarray(R2R0_SHARD_GLOBAL_INDICES[shard_id], dtype="<i8")
    if not np.array_equal(arrays.get("thermal_global_index"), expected_indices):
        raise ValueError("attempt3 shard modulo partition changed")
    nt = len(expected_indices)
    shapes = {
        "thermal_global_index": (nt,),
        "thermal_fixed_energy_eV": (nt, 1),
        "thermal_fixed_force_eV_A": (nt, 72, 3),
        "thermal_parameter_energy_design_eV": (nt, 65),
        "thermal_parameter_force_design_eV_A": (nt, 72, 3, 65),
    }
    for key, shape in shapes.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ValueError(f"attempt3 shard array schema changed: {key}")
        if key != "thermal_global_index" and not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"attempt3 shard array is non-finite: {key}")
    return receipt, {key: arrays[key] for key in shapes}


def _capture_tree_from_directory_fd(
    directory_fd: int,
    *,
    prefix: str,
    snapshot: dict[str, dict[str, Any]],
    file_bytes: dict[str, bytes],
    budget: dict[str, int],
    stable_directories: bool = False,
) -> None:
    before_directory = os.fstat(directory_fd)
    before_identity = (
        _directory_binding_identity(before_directory)
        if stable_directories
        else _stat_identity(before_directory)
    )
    collected_names: list[str] = []
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            budget["items"] += 1
            if budget["items"] > ATTEMPT3_TREE_ITEM_LIMIT:
                raise ValueError("R2R-0 completed tree exceeds frozen item limit")
            name = entry.name
            if name in {"", ".", ".."} or "/" in name:
                raise ValueError("R2R-0 completed tree contains an invalid entry name")
            relative = f"{prefix}/{name}" if prefix else name
            _reject_forbidden_components(
                tuple(relative.split("/")),
                "R2R-0 completed recovery tree entry",
                FORBIDDEN_LABEL_PATH_TOKENS,
            )
            collected_names.append(name)
    names = tuple(sorted(collected_names))
    for name in names:
        relative = f"{prefix}/{name}" if prefix else name
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise ValueError("R2R-0 completed recovery tree contains a symlink")
        if stat.S_ISDIR(before.st_mode):
            child_fd = os.open(
                name, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=directory_fd
            )
            try:
                child_identity = (
                    _directory_binding_identity(os.fstat(child_fd))
                    if stable_directories
                    else _stat_identity(os.fstat(child_fd))
                )
                before_child_identity = (
                    _directory_binding_identity(before)
                    if stable_directories
                    else _stat_identity(before)
                )
                if child_identity != before_child_identity:
                    raise ValueError("R2R-0 directory changed across stat/open")
                snapshot[relative] = {"kind": "directory", **child_identity}
                _capture_tree_from_directory_fd(
                    child_fd,
                    prefix=relative,
                    snapshot=snapshot,
                    file_bytes=file_bytes,
                    budget=budget,
                    stable_directories=stable_directories,
                )
                final_child_identity = (
                    _directory_binding_identity(os.fstat(child_fd))
                    if stable_directories
                    else _stat_identity(os.fstat(child_fd))
                )
                if final_child_identity != child_identity:
                    raise ValueError("R2R-0 directory changed while captured")
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(before.st_mode):
            if before.st_nlink != 1:
                raise ValueError("R2R-0 completed tree file has a hard-link alias")
            if before.st_size > ATTEMPT3_TREE_SINGLE_FILE_BYTES_LIMIT:
                raise ValueError("R2R-0 completed tree file exceeds frozen size limit")
            remaining_total = ATTEMPT3_TREE_TOTAL_BYTES_LIMIT - budget["bytes"]
            if before.st_size > remaining_total:
                raise ValueError("R2R-0 completed tree exceeds frozen byte limit")
            file_fd = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
            try:
                identity = (
                    _owned_regular_identity(os.fstat(file_fd))
                    if stable_directories
                    else _stat_identity(os.fstat(file_fd))
                )
                before_file_identity = (
                    _owned_regular_identity(before)
                    if stable_directories
                    else _stat_identity(before)
                )
                if identity != before_file_identity:
                    raise ValueError("R2R-0 file changed across stat/open")
                chunks: list[bytes] = []
                actual_size = 0
                actual_limit = min(
                    ATTEMPT3_TREE_SINGLE_FILE_BYTES_LIMIT, remaining_total
                )
                while True:
                    chunk = os.read(
                        file_fd,
                        min(1024 * 1024, actual_limit - actual_size + 1),
                    )
                    if not chunk:
                        break
                    actual_size += len(chunk)
                    if actual_size > actual_limit:
                        raise ValueError(
                            "R2R-0 completed tree file grew beyond frozen byte limit"
                        )
                    chunks.append(chunk)
                final_file_identity = (
                    _owned_regular_identity(os.fstat(file_fd))
                    if stable_directories
                    else _stat_identity(os.fstat(file_fd))
                )
                if final_file_identity != identity:
                    raise ValueError("R2R-0 file changed while captured")
                still_bound = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                still_bound_identity = (
                    _owned_regular_identity(still_bound)
                    if stable_directories
                    else _stat_identity(still_bound)
                )
                if still_bound_identity != identity:
                    raise ValueError("R2R-0 file entry changed while captured")
            finally:
                os.close(file_fd)
            raw = b"".join(chunks)
            budget["bytes"] += len(raw)
            file_bytes[relative] = raw
            snapshot[relative] = {
                "kind": "file",
                **identity,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        else:
            raise ValueError("R2R-0 completed recovery tree contains a special file")
    final_names: list[str] = []
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            final_names.append(entry.name)
            if len(final_names) > len(names):
                raise ValueError("R2R-0 directory inventory grew while captured")
    if tuple(sorted(final_names)) != names:
        raise ValueError("R2R-0 directory inventory changed while captured")
    for name in names:
        relative = f"{prefix}/{name}" if prefix else name
        record = snapshot[relative]
        rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        expected_kind = "directory" if stat.S_ISDIR(rebound.st_mode) else "file"
        observed_identity = (
            _directory_binding_identity(rebound)
            if expected_kind == "directory" and stable_directories
            else (
                _owned_regular_identity(rebound)
                if expected_kind == "file" and stable_directories
                else _stat_identity(rebound)
            )
        )
        expected_identity = {key: record[key] for key in observed_identity}
        if (
            stat.S_ISLNK(rebound.st_mode)
            or record.get("kind") != expected_kind
            or not _json_type_exact_equal(observed_identity, expected_identity)
        ):
            raise ValueError(
                "R2R-0 completed tree entry identity changed after capture"
            )
    final_directory_identity = (
        _directory_binding_identity(os.fstat(directory_fd))
        if stable_directories
        else _stat_identity(os.fstat(directory_fd))
    )
    if final_directory_identity != before_identity:
        raise ValueError("R2R-0 directory metadata changed while captured")


def _verify_captured_tree_identities_from_fd(
    directory_fd: int,
    *,
    prefix: str,
    snapshot: Mapping[str, Mapping[str, Any]],
    stable_directories: bool = False,
) -> None:
    """Rebind a completed capture through the still-held root descriptor.

    The first recursive walk closes each directory locally, but a later
    top-level sibling could otherwise modify a grandchild in an already
    closed subtree without changing that subtree directory's metadata.  This
    identity-only second walk starts only after the complete payload capture
    and verifies every descendant again.  Expected entry identity is checked
    before opening a directory, then checked against fstat after open, so a
    replacement subtree is rejected before any of its children are opened.
    """
    direct: dict[str, str] = {}
    prefix_with_slash = f"{prefix}/" if prefix else ""
    for relative in snapshot:
        if relative == "." or not relative.startswith(prefix_with_slash):
            continue
        suffix = relative[len(prefix_with_slash) :]
        if not suffix or "/" in suffix:
            continue
        direct[suffix] = relative
    expected_names = set(direct)
    observed_names: list[str] = []
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            if len(observed_names) >= len(expected_names):
                raise ValueError(
                    "R2R-0 completed tree inventory grew before global rebind"
                )
            observed_names.append(entry.name)
    if set(observed_names) != expected_names:
        raise ValueError(
            "R2R-0 completed tree inventory changed before global rebind"
        )
    directory_identity = (
        _directory_binding_identity(os.fstat(directory_fd))
        if stable_directories
        else _stat_identity(os.fstat(directory_fd))
    )
    expected_directory = snapshot["." if not prefix else prefix]
    expected_directory_identity = {
        key: expected_directory[key] for key in directory_identity
    }
    if not _json_type_exact_equal(
        directory_identity, expected_directory_identity
    ):
        raise ValueError(
            "R2R-0 completed tree directory changed before global rebind"
        )
    for name in sorted(expected_names):
        relative = direct[name]
        _reject_forbidden_components(
            tuple(relative.split("/")),
            "R2R-0 completed recovery global-rebind entry",
            FORBIDDEN_LABEL_PATH_TOKENS,
        )
        record = snapshot[relative]
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        before_identity = (
            _directory_binding_identity(before)
            if record.get("kind") == "directory" and stable_directories
            else (
                _owned_regular_identity(before)
                if record.get("kind") == "file" and stable_directories
                else _stat_identity(before)
            )
        )
        expected_identity = {key: record[key] for key in before_identity}
        if not _json_type_exact_equal(before_identity, expected_identity):
            raise ValueError(
                "R2R-0 completed tree entry changed before global rebind"
            )
        if record.get("kind") == "directory":
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    "R2R-0 completed tree directory kind changed before rebind"
                )
            child_fd = os.open(
                name, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=directory_fd
            )
            try:
                if not _json_type_exact_equal(
                    (
                        _directory_binding_identity(os.fstat(child_fd))
                        if stable_directories
                        else _stat_identity(os.fstat(child_fd))
                    ),
                    expected_identity,
                ):
                    raise ValueError(
                        "R2R-0 completed tree directory changed across global open"
                    )
                _verify_captured_tree_identities_from_fd(
                    child_fd,
                    prefix=relative,
                    snapshot=snapshot,
                    stable_directories=stable_directories,
                )
                if not _json_type_exact_equal(
                    (
                        _directory_binding_identity(os.fstat(child_fd))
                        if stable_directories
                        else _stat_identity(os.fstat(child_fd))
                    ),
                    expected_identity,
                ):
                    raise ValueError(
                        "R2R-0 completed tree directory changed during global rebind"
                    )
            finally:
                os.close(child_fd)
            rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not _json_type_exact_equal(
                (
                    _directory_binding_identity(rebound)
                    if stable_directories
                    else _stat_identity(rebound)
                ),
                expected_identity,
            ):
                raise ValueError(
                    "R2R-0 completed tree directory entry changed after global rebind"
                )
        elif record.get("kind") == "file":
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
            ):
                raise ValueError(
                    "R2R-0 completed tree file kind changed before global rebind"
                )
        else:
            raise ValueError("R2R-0 completed tree snapshot kind changed")
    if not _json_type_exact_equal(
        (
            _directory_binding_identity(os.fstat(directory_fd))
            if stable_directories
            else _stat_identity(os.fstat(directory_fd))
        ),
        expected_directory_identity,
    ):
        raise ValueError(
            "R2R-0 completed tree directory changed during global rebind"
        )


def _fd_bound_tree_capture(
    root: Path,
    *,
    expected_chain: Sequence[Mapping[str, int]] | None = None,
    stable_directories: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    """Capture a complete tree through held directory fds without following links."""
    base, root_fd, root_chain = _open_directory_chain(
        root, "R2R-0 completed recovery tree", expected_chain=expected_chain
    )
    root_identity = (
        _directory_binding_identity(os.fstat(root_fd))
        if stable_directories
        else _stat_identity(os.fstat(root_fd))
    )
    snapshot: dict[str, dict[str, Any]] = {
        ".": {"kind": "directory", **root_identity}
    }
    files: dict[str, bytes] = {}
    budget = {"items": 1, "bytes": 0}
    try:
        _capture_tree_from_directory_fd(
            root_fd,
            prefix="",
            snapshot=snapshot,
            file_bytes=files,
            budget=budget,
            stable_directories=stable_directories,
        )
        _verify_captured_tree_identities_from_fd(
            root_fd,
            prefix="",
            snapshot=snapshot,
            stable_directories=stable_directories,
        )
        final_root_identity = (
            _directory_binding_identity(os.fstat(root_fd))
            if stable_directories
            else _stat_identity(os.fstat(root_fd))
        )
        if final_root_identity != root_identity:
            raise ValueError("R2R-0 root changed while captured")
        _verify_directory_chain(
            base, "R2R-0 completed recovery tree final binding", root_chain
        )
    finally:
        os.close(root_fd)
    return snapshot, files


def _immutable_tree_snapshot(
    root: Path, *, stable_directories: bool = False
) -> dict[str, dict[str, Any]]:
    snapshot, _files = _fd_bound_tree_capture(
        root, stable_directories=stable_directories
    )
    return snapshot


def _directory_entry_names(path: Path, label: str) -> set[str]:
    source, descriptor, chain = _open_directory_chain(path, label)
    identity = _stat_identity(os.fstat(descriptor))
    try:
        first = set(os.listdir(descriptor))
        if set(os.listdir(descriptor)) != first:
            raise ValueError(f"{label} inventory changed while listed")
        if _stat_identity(os.fstat(descriptor)) != identity:
            raise ValueError(f"{label} changed while listed")
        _verify_directory_chain(source, f"{label} final binding", chain)
        return first
    finally:
        os.close(descriptor)


def _materialize_captured_tree(
    destination: Path,
    snapshot: Mapping[str, Mapping[str, Any]],
    file_bytes: Mapping[str, bytes],
) -> dict[str, dict[str, Any]]:
    """Materialize an owned private copy and freeze portable identities."""
    # Imported lazily to avoid a module-import cycle.  The aggregate module's
    # held-fd owned-record primitives are also the single implementation used
    # for publication artifacts.  The source attempt3 tree remains full-nine
    # externally bound; this opaque process-private copy uses owned-eight
    # regular identities and stable directory bindings.
    import aggregate_graphene_r2r1_linear_readout as owned

    destination.mkdir(mode=0o700)
    directories = sorted(
        (
            relative
            for relative, record in snapshot.items()
            if relative != "." and record.get("kind") == "directory"
        ),
        key=lambda value: (value.count("/"), value),
    )
    for relative in directories:
        (destination / relative).mkdir(mode=0o700)

    expected_snapshot: dict[str, dict[str, Any]] = {
        ".": {
            "kind": "directory",
            **_directory_binding_identity(destination.stat()),
        }
    }
    for relative in directories:
        expected_snapshot[relative] = {
            "kind": "directory",
            **_directory_binding_identity((destination / relative).stat()),
        }

    payload_records: dict[str, dict[str, Any]] = {}
    for relative, raw in sorted(file_bytes.items()):
        target = destination / relative
        parent_fd = os.open(
            target.parent,
            _DIRECTORY_FLAGS | _NOFOLLOW,
        )
        try:
            identity = owned._atomic_write_bytes_at(parent_fd, target.name, raw)
        finally:
            os.close(parent_fd)
        payload_records[relative] = {
            "identity": identity,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    for relative, record in payload_records.items():
        expected_snapshot[relative] = {
            "kind": "file",
            **dict(record["identity"]),
            "sha256": record["sha256"],
        }
    return expected_snapshot


def _tree_content_signature(
    snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, str | None]]:
    return {
        relative: (str(record["kind"]), record.get("sha256"))
        for relative, record in snapshot.items()
    }


def validate_attempt3_completed_recovery(
    attempt3_root: Path = ATTEMPT3_ROOT,
    *,
    expected_root_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate a captured attempt3 copy; no live launcher execute path is called."""
    observed_root = _lexical_path(
        attempt3_root,
        "attempt3 completed run root",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    frozen_root = _lexical_path(
        ATTEMPT3_ROOT,
        "frozen attempt3 completed run root",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if observed_root != frozen_root:
        raise PermissionError("attempt3 recovery root differs from frozen path")
    dependency_sha256 = current_dependency_sha256()
    root = (
        _lexical_path(
            observed_root,
            "attempt3 completed run root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        )
        if expected_root_binding is not None
        else _reject_path(
            observed_root,
            "attempt3 completed run root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        )
    )
    expected_chain: Sequence[Mapping[str, int]]
    if expected_root_binding is not None:
        if (
            not isinstance(expected_root_binding, Mapping)
            or set(expected_root_binding)
            != {"path", "directory_identity", "chain"}
            or expected_root_binding.get("path") != str(observed_root)
        ):
            raise PermissionError("attempt3 expected root binding schema changed")
        expected_chain = expected_root_binding["chain"]
        if (
            not isinstance(expected_chain, list)
            or not expected_chain
            or not _json_type_exact_equal(
                expected_root_binding.get("directory_identity"), expected_chain[-1]
            )
        ):
            raise PermissionError("attempt3 expected root identity type/schema changed")
    else:
        # Even the public label-blind/default validator freezes one observed
        # metadata-only chain before its first payload capture, then reuses it
        # for both full walks.  A phase-boundary replacement is therefore
        # rejected before any child of the replacement can be opened.
        observed_binding = directory_metadata_binding(
            root, "attempt3 default recovery metadata binding"
        )
        expected_chain = observed_binding["chain"]
    tree_before, captured_files = _fd_bound_tree_capture(
        root, expected_chain=expected_chain
    )
    expected_root_files = {
        "DONE",
        "EXIT_CODE",
        "launch_receipt.json",
        "partial_state.json",
    }
    if not expected_root_files.issubset(
        {name for name in captured_files if "/" not in name}
    ):
        raise ValueError("attempt3 launcher terminal inventory is incomplete")
    if len(tree_before) != ATTEMPT3_TREE_ITEM_LIMIT:
        raise ValueError("attempt3 completed tree item count changed")

    with tempfile.TemporaryDirectory(
        prefix="graphene_r2r1_attempt3_capture_", dir="/private/tmp"
    ) as temp:
        copy_root = Path(temp) / "attempt3"
        materialized_snapshot = _materialize_captured_tree(
            copy_root, tree_before, captured_files
        )
        copy_before = _immutable_tree_snapshot(
            copy_root, stable_directories=True
        )
        if copy_before != materialized_snapshot:
            raise ValueError(
                "R2R-1 private attempt3 copy changed after owned-inode settlement"
            )
        control = copy_root / "control"
        shard_roots = [
            copy_root / "artifacts" / f"shard{index}" for index in range(3)
        ]
        mechanics_root = copy_root / "artifacts" / "mechanics"
        aggregate_root = copy_root / "aggregate"

        exit_raw, _exit_identity = _read_regular_file_once(
            copy_root / "EXIT_CODE",
            "attempt3 captured launcher EXIT_CODE",
            size_limit=ATTEMPT3_EXIT_CODE_BYTES_LIMIT,
        )
        if exit_raw != b"0\n":
            raise ValueError("attempt3 launcher EXIT_CODE changed")
        launch_raw, _launch_identity = _read_regular_file_once(
            copy_root / "launch_receipt.json",
            "attempt3 captured launcher receipt",
            size_limit=ATTEMPT3_JSON_FILE_BYTES_LIMIT,
        )
        if hashlib.sha256(launch_raw).hexdigest() != ATTEMPT3_LAUNCH_RECEIPT_SHA256:
            raise ValueError("attempt3 launch receipt SHA changed")
        launch_receipt = _json_object_from_bytes(
            launch_raw, "attempt3 captured launcher receipt"
        )
        expected_done = (
            str(launch_receipt.get("status"))
            + "\n"
            + ATTEMPT3_LAUNCH_RECEIPT_SHA256
            + "\n"
        ).encode("ascii")
        done_raw, _done_identity = _read_regular_file_once(
            copy_root / "DONE",
            "attempt3 captured launcher DONE marker",
            size_limit=ATTEMPT3_TERMINAL_MARKER_BYTES_LIMIT,
        )
        if done_raw != expected_done:
            raise ValueError("attempt3 launcher DONE marker changed")
        control_manifest_raw, _control_manifest_identity = _read_regular_file_once(
            control / "freeze_manifest.json",
            "attempt3 captured freeze manifest",
            size_limit=ATTEMPT3_JSON_FILE_BYTES_LIMIT,
        )
        if (
            hashlib.sha256(control_manifest_raw).hexdigest()
            != ATTEMPT3_FREEZE_MANIFEST_SHA256
        ):
            raise ValueError("attempt3 freeze manifest SHA changed")

        # This private validator contains no execution branch.  It checks all
        # 220 launcher/control/log fields against the captured immutable copy.
        # Frozen R2R-0 still compares Darwin ctime on this process-private
        # copy.  Permit exactly one pure-read retry only when an owned8+raw
        # recapture proves that every inode, byte, and inventory item remains
        # equal to the pre-call baseline.  A real content/inode change never
        # reaches a retry, and a second legacy failure propagates unchanged.
        def recover_legacy_done_copy() -> tuple[dict[str, Any], dict[str, Any]]:
            observed_authorization = r2r0.validate_execution_authorization(
                control / "freeze_manifest.json", control / "R2R0_FORMAL_GO"
            )
            run_id = observed_authorization["freeze_manifest_sha256"][:12]
            r2r0_launcher._validate_launch_recovery(
                copy_root, launch_receipt, observed_authorization, run_id, 3
            )
            observed_aggregate = r2r0.aggregate_shards(
                shard_roots,
                mechanics_root,
                aggregate_root,
                freeze_manifest=control / "freeze_manifest.json",
                authorization_marker=control / "R2R0_FORMAL_GO",
            )
            return observed_authorization, observed_aggregate

        try:
            authorization, aggregate_recovered = recover_legacy_done_copy()
        except Exception:
            retry_snapshot = _immutable_tree_snapshot(
                copy_root, stable_directories=True
            )
            if retry_snapshot != copy_before:
                raise
            authorization, aggregate_recovered = recover_legacy_done_copy()
        if aggregate_recovered.get("completion_recovered_without_recompute") is not True:
            raise ValueError("attempt3 aggregate did not take its DONE recovery branch")
        aggregate_receipt, aggregate_arrays = validate_attempt3_aggregate(
            aggregate_root / "receipt.json", aggregate_root / "arrays.npz"
        )
        aggregate_exit, _aggregate_exit_identity = _read_regular_file_once(
            aggregate_root / "EXIT_CODE",
            "attempt3 aggregate EXIT_CODE",
            size_limit=ATTEMPT3_EXIT_CODE_BYTES_LIMIT,
        )
        aggregate_done, _aggregate_done_identity = _read_regular_file_once(
            aggregate_root / "DONE",
            "attempt3 aggregate DONE marker",
            size_limit=ATTEMPT3_TERMINAL_MARKER_BYTES_LIMIT,
        )
        if aggregate_exit != b"0\n" or aggregate_done != (
            aggregate_receipt["status"]
            + "\n"
            + ATTEMPT3_AGGREGATE_RECEIPT_SHA256
            + "\n"
        ).encode("ascii"):
            raise ValueError("attempt3 aggregate terminal binding changed")
        loaded = []
        for shard_id, shard_root in enumerate(shard_roots):
            receipt, arrays = validate_attempt3_shard(
                shard_id,
                shard_root / "receipt.json",
                shard_root / "thermal_arrays.npz",
            )
            loaded.append((receipt, arrays))
        concatenated_indices = np.concatenate(
            [arrays["thermal_global_index"] for _receipt, arrays in loaded]
        )
        order = np.argsort(concatenated_indices)
        if not np.array_equal(concatenated_indices[order], np.arange(92)):
            raise ValueError("attempt3 three-shard rebuild does not cover thermal92")
        rebuilt: dict[str, np.ndarray] = {}
        for key in EXPECTED_AGGREGATE_ARRAY_RAW_SHA256:
            rebuilt[key] = np.concatenate(
                [arrays[key] for _receipt, arrays in loaded], axis=0
            )[order]
            if rebuilt[key].dtype != aggregate_arrays[key].dtype or not np.array_equal(
                rebuilt[key], aggregate_arrays[key]
            ):
                raise ValueError(
                    f"attempt3 three-shard rebuild differs from aggregate: {key}"
                )
        copy_after = _immutable_tree_snapshot(
            copy_root, stable_directories=True
        )
        if copy_after != copy_before:
            raise ValueError("attempt3 captured recovery copy was modified")

    tree_after, _after_files = _fd_bound_tree_capture(
        root, expected_chain=expected_chain
    )
    if tree_after != tree_before:
        raise ValueError("attempt3 source tree changed during captured recovery")
    if current_dependency_sha256() != dependency_sha256:
        raise ValueError("R2R-0 live recovery dependency changed during validation")
    return (
        {
            "format": "graphene_r2r1_attempt3_completed_recovery_v1",
            "attempt3_status": aggregate_receipt["status"],
            "R2R0_launcher_terminal_pure_read_verified": True,
            "R2R0_aggregate_terminal_pure_read_verified": True,
            "R2R0_launcher_terminal_captured_copy_verified": True,
            "R2R0_aggregate_terminal_captured_copy_verified": True,
            "R2R0_launcher_completion_recovered_without_recompute": True,
            "R2R0_aggregate_completion_recovered_without_recompute": True,
            "live_launcher_execute_called": False,
            "remote_or_staging_code_triggered": False,
            "completed_tree_file_and_directory_count": len(tree_before),
            "completed_tree_hash_mtime_identity_unchanged": True,
            "launch_receipt_sha256": ATTEMPT3_LAUNCH_RECEIPT_SHA256,
            "dependency_sha256": dependency_sha256,
            "three_shard_modulo_partition_exact": True,
            "three_shard_arrays_bit_exact_to_aggregate": True,
            "shard_receipt_sha256": list(ATTEMPT3_SHARD_RECEIPT_SHA256),
            "shard_thermal_arrays_sha256": list(
                ATTEMPT3_SHARD_THERMAL_ARRAYS_SHA256
            ),
        },
        rebuilt,
    )


def design_subset_audit(force_design: np.ndarray, indices: Sequence[int]) -> dict[str, Any]:
    design = np.asarray(force_design, dtype=np.float64)[np.asarray(indices, dtype=int)]
    if design.ndim != 4 or design.shape[-1] != LINEAR_WIDTH:
        raise ValueError("R2R-1 force design must have shape [config,atom,3,65]")
    matrix = design.reshape(-1, LINEAR_WIDTH)
    scale = np.sqrt(np.mean(np.square(matrix), axis=0))
    maximum = float(np.max(scale))
    inactive = np.flatnonzero(scale <= maximum * ZERO_COLUMN_RELATIVE_RMS).tolist()
    if inactive:
        return {
            "pass": False,
            "zero_or_near_zero_columns": inactive,
            "rank": 0,
            "condition": None,
        }
    scaled = matrix / scale
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = float(singular[0] * max(scaled.shape) * np.finfo(np.float64).eps)
    rank = int(np.sum(singular > tolerance))
    condition = float(singular[0] / singular[-1])
    return {
        "pass": rank == LINEAR_WIDTH and condition <= SCALED_CONDITION_LIMIT,
        "configuration_count": int(design.shape[0]),
        "rows": int(matrix.shape[0]),
        "columns": LINEAR_WIDTH,
        "column_RMS": scale.tolist(),
        "min_relative_column_RMS": float(np.min(scale) / maximum),
        "zero_or_near_zero_columns": inactive,
        "rank": rank,
        "rank_tolerance": tolerance,
        "condition": condition,
        "condition_limit": SCALED_CONDITION_LIMIT,
        "scaling": "train_force_RMS_without_mean_centering",
    }


def all_split_design_audits(force_design: np.ndarray) -> dict[str, Any]:
    all_indices = np.arange(92)
    outer: dict[str, Any] = {}
    inner: dict[str, Any] = {}
    for outer_id, outer_hold in enumerate(FOLD_GLOBAL_INDICES):
        outer_train = np.setdiff1d(all_indices, np.asarray(outer_hold))
        outer[str(outer_id)] = design_subset_audit(force_design, outer_train)
        for inner_hold_id in range(4):
            if inner_hold_id == outer_id:
                continue
            inner_train = np.setdiff1d(
                outer_train, np.asarray(FOLD_GLOBAL_INDICES[inner_hold_id])
            )
            inner[f"outer{outer_id}_inner_hold{inner_hold_id}"] = design_subset_audit(
                force_design, inner_train
            )
    final = design_subset_audit(force_design, all_indices)
    observed_outer = [outer[str(index)]["condition"] for index in range(4)]
    observed_inner = [value["condition"] for value in inner.values()]
    observed_relative = [
        value["min_relative_column_RMS"]
        for value in [*outer.values(), *inner.values()]
    ]
    expected_numeric_match = bool(
        np.allclose(
            observed_outer,
            EXPECTED_OUTER_TRAIN69_CONDITION,
            rtol=2.0e-10,
            atol=0.0,
        )
        and math.isclose(
            min(observed_inner),
            EXPECTED_INNER_TRAIN46_CONDITION_RANGE[0],
            rel_tol=2.0e-10,
            abs_tol=0.0,
        )
        and math.isclose(
            max(observed_inner),
            EXPECTED_INNER_TRAIN46_CONDITION_RANGE[1],
            rel_tol=2.0e-10,
            abs_tol=0.0,
        )
        and math.isclose(
            min(observed_relative),
            EXPECTED_SPLIT_MIN_RELATIVE_COLUMN_RMS_RANGE[0],
            rel_tol=2.0e-10,
            abs_tol=0.0,
        )
        and math.isclose(
            max(observed_relative),
            EXPECTED_SPLIT_MIN_RELATIVE_COLUMN_RMS_RANGE[1],
            rel_tol=2.0e-10,
            abs_tol=0.0,
        )
    )
    return {
        "outer_train69": outer,
        "inner_train46": inner,
        "final_train92": final,
        "frozen_design_only_expectation": {
            "outer_train69_condition": list(EXPECTED_OUTER_TRAIN69_CONDITION),
            "inner_train46_condition_range": list(
                EXPECTED_INNER_TRAIN46_CONDITION_RANGE
            ),
            "split_min_relative_column_RMS_range": list(
                EXPECTED_SPLIT_MIN_RELATIVE_COLUMN_RMS_RANGE
            ),
            "comparison_relative_tolerance": 2.0e-10,
            "matches": expected_numeric_match,
        },
        "pass": final["pass"]
        and all(value["pass"] for value in outer.values())
        and all(value["pass"] for value in inner.values())
        and expected_numeric_match,
    }


@dataclass(frozen=True)
class AprimeData:
    mode_real: np.ndarray
    mode_imag: np.ndarray
    coordinates: np.ndarray
    foundation_base_force_eV_A: np.ndarray
    frozen_q6_force_eV_A: np.ndarray


@dataclass(frozen=True)
class ThermalLabels:
    reference_force_eV_A: np.ndarray
    aprime: AprimeData
    structure_identity_sha256: str
    raw_sha256: Mapping[str, str]
    parser_receipt: Mapping[str, Any]


def _parse_properties(value: str) -> tuple[tuple[str, str, int], ...]:
    pieces = value.split(":")
    if not pieces or len(pieces) % 3:
        raise ValueError("extxyz Properties schema changed")
    output = []
    for offset in range(0, len(pieces), 3):
        name, kind, width_text = pieces[offset : offset + 3]
        width = int(width_text)
        if width <= 0:
            raise ValueError("extxyz property width must be positive")
        output.append((name, kind, width))
    return tuple(output)


def _property_offsets(
    schema: Sequence[tuple[str, str, int]],
) -> tuple[dict[str, tuple[int, int]], int]:
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, _kind, width in schema:
        if name in offsets:
            raise ValueError("duplicate extxyz property")
        offsets[name] = (cursor, width)
        cursor += width
    return offsets, cursor


def _header_fields(header_line: str) -> dict[str, str]:
    allowed = {
        "Properties",
        "Lattice",
        "pbc",
        "config_type",
        "APRIME_coordinate_real_A",
        "APRIME_coordinate_imag_A",
    }
    fields: dict[str, str] = {}
    for token in shlex.split(header_line, posix=True):
        if "=" in token:
            key, value = token.split("=", 1)
            if key not in allowed:
                # Unlisted label/header values are discarded immediately and
                # never retained in a mapping or converted to a numeric value.
                continue
            if key in fields:
                raise ValueError(f"duplicate extxyz header key: {key}")
            fields[key] = value
    return fields


def current_source_sha256() -> dict[str, str]:
    observed = {}
    for name, path in R2R1_SOURCE_PATHS.items():
        # The bound reader is the sole existence/type check.  A preliminary
        # Path.is_file() would follow a symlink before the no-follow openat
        # chain has had a chance to reject it.
        raw, _identity = _read_regular_file_once(
            path,
            f"R2R-1 source {name}",
            size_limit=SOURCE_CLOSURE_SINGLE_FILE_BYTES_LIMIT,
        )
        observed[name] = hashlib.sha256(raw).hexdigest()
    expected_frozen = {
        "frozen_readout": FROZEN_READOUT_SOURCE_SHA256,
        "frozen_readout_tests": FROZEN_READOUT_TEST_SHA256,
    }
    for name, expected in expected_frozen.items():
        if observed[name] != expected:
            raise ValueError(f"independently frozen R2R-1 source changed: {name}")
    return observed


def current_dependency_sha256() -> dict[str, str]:
    observed = {}
    for name, path in R2R1_DEPENDENCY_PATHS.items():
        raw, _identity = _read_regular_file_once(
            path,
            f"R2R-1 dependency {name}",
            size_limit=SOURCE_CLOSURE_SINGLE_FILE_BYTES_LIMIT,
        )
        observed[name] = hashlib.sha256(raw).hexdigest()
    if observed != EXPECTED_R2R1_DEPENDENCY_SHA256:
        raise ValueError("R2R-1 imported/provenance dependency source changed")
    return observed


def _freeze_manifest_static_payload() -> dict[str, Any]:
    """Return manifest fields that do not inspect label/design input roots."""
    attempt_partition = _validate_attempt2_path_partition()
    return {
        "format": "graphene_r2r1_freeze_manifest_v2",
        "contract_semantic_sha256": CONTRACT_SEMANTIC_SHA256,
        "source_sha256": current_source_sha256(),
        "dependency_sha256": current_dependency_sha256(),
        "runtime_environment": runtime_environment_receipt(),
        "attempt3": {
            "aggregate_receipt_sha256": ATTEMPT3_AGGREGATE_RECEIPT_SHA256,
            "aggregate_arrays_sha256": ATTEMPT3_AGGREGATE_ARRAYS_SHA256,
            "freeze_manifest_sha256": ATTEMPT3_FREEZE_MANIFEST_SHA256,
            "launch_receipt_sha256": ATTEMPT3_LAUNCH_RECEIPT_SHA256,
            "shard_receipt_sha256": list(ATTEMPT3_SHARD_RECEIPT_SHA256),
            "shard_thermal_arrays_sha256": list(
                ATTEMPT3_SHARD_THERMAL_ARRAYS_SHA256
            ),
        },
        "thermal92_file_sha256": THERMAL92_FILE_SHA256,
        "thermal92_canonical_path": str(
            _lexical_path(
                RECOMMENDED_THERMAL92,
                "R2R-1 frozen thermal92 manifest path",
                forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
            )
        ),
        "thermal92_payload_open_deferred_until_external_GO": True,
        "control_root_canonical_path": str(
            _lexical_path(RECOMMENDED_CONTROL_ROOT, "R2R-1 control root path")
        ),
        "fit_output_parent_canonical_path": str(
            _lexical_path(
                RECOMMENDED_FIT_OUTPUT_ROOT.parent,
                "R2R-1 fit output parent path",
            )
        ),
        "candidate_parent_canonical_path": str(
            _lexical_path(
                RECOMMENDED_CANDIDATE_ROOT.parent,
                "R2R-1 candidate parent path",
            )
        ),
        "candidate_root_expected_absent_at_freeze": True,
        "attempt_id": "attempt2_retry1",
        "attempt2_path_partition": attempt_partition,
        "attempt1_forensic_partial_must_remain_read_only": True,
        "failed_attempt2_forensic_partial_must_remain_read_only": True,
        "thermal_structure_identity_sha256": THERMAL_STRUCTURE_IDENTITY_SHA256,
        "label_raw_sha256": EXPECTED_LABEL_RAW_SHA256,
        "fold_contract_sha256": FOLD_CONTRACT_SHA256,
        "alpha_grid": list(ALPHA_GRID),
        "group_masses": GROUP_MASSES,
        "gate_thresholds": GATE_THRESHOLDS,
        "authorization_marker_created": False,
        "fit_authorized_by_manifest_alone": False,
        "thermal92_force_labels_authorized_by_manifest_alone": False,
        "energy_labels_authorized": False,
        "development_or_held_access_authorized": False,
    }


def _validate_manifest_bound_prefix_consistency(payload: Mapping[str, Any]) -> None:
    """Require every repeated lexical directory prefix to bind one identity."""
    def fields(
        key: str, *, path_key: str, chain_key: str, parent_of_path: bool = False
    ) -> tuple[Any, Any, str]:
        binding = payload.get(key)
        if not isinstance(binding, Mapping):
            raise PermissionError(f"R2R-1 {key} schema changed")
        path_value = binding.get(path_key)
        if parent_of_path and type(path_value) is str:
            path_value = str(Path(path_value).parent)
        return path_value, binding.get(chain_key), key

    bindings = [
        fields("control_root_binding", path_key="path", chain_key="chain"),
        fields("result_parent_binding", path_key="path", chain_key="chain"),
        fields("attempt3_root_binding", path_key="path", chain_key="chain"),
        fields(
            "thermal92_metadata_binding",
            path_key="path",
            chain_key="parent_chain",
            parent_of_path=True,
        ),
    ]
    for collection_key in ("source_file_bindings", "dependency_file_bindings"):
        collection = payload.get(collection_key)
        if not isinstance(collection, Mapping):
            raise PermissionError(f"R2R-1 {collection_key} schema changed")
        for name, binding in sorted(collection.items()):
            if type(name) is not str or not isinstance(binding, Mapping):
                raise PermissionError(f"R2R-1 {collection_key} entry changed")
            path_value = binding.get("path")
            if type(path_value) is str:
                path_value = str(Path(path_value).parent)
            bindings.append(
                (path_value, binding.get("parent_chain"), f"{collection_key}.{name}")
            )
    observed: dict[str, Any] = {}
    expected_identity_keys = {
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
    }
    for path_value, chain, label in bindings:
        if type(path_value) is not str or not path_value.startswith(os.sep):
            raise PermissionError(f"R2R-1 {label} binding path schema changed")
        path = Path(path_value)
        if not isinstance(chain, list) or len(chain) != len(path.parts):
            raise PermissionError(f"R2R-1 {label} binding chain depth changed")
        prefix = Path(path.anchor)
        for index, identity in enumerate(chain):
            if (
                not isinstance(identity, Mapping)
                or set(identity) != expected_identity_keys
                or any(type(value) is not int for value in identity.values())
            ):
                raise PermissionError(
                    f"R2R-1 {label} binding identity schema changed"
                )
            if index > 0:
                prefix /= path.parts[index]
            key = str(prefix)
            previous = observed.get(key)
            if previous is not None and not _json_type_exact_equal(
                previous, identity
            ):
                raise PermissionError(
                    f"R2R-1 metadata bindings disagree at shared prefix {key}"
                )
            observed[key] = dict(identity)


def freeze_manifest_payload(
    *,
    control_root_binding: Mapping[str, Any] | None = None,
    held_result_parent_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a label-blind candidate manifest without reading thermal92.

    The thermal file is never opened or read here.  Its canonical lexical path,
    previously frozen whole-file digest, parent chain, and metadata-only final
    entry identity are bound before GO using no-follow ``stat`` operations.
    Only after the external GO marker has authenticated that binding may the
    exact same entry be opened, hashed, and parsed.
    """
    payload = {
        **_freeze_manifest_static_payload(),
        "control_root_binding": dict(control_root_binding)
        if control_root_binding is not None
        else directory_metadata_binding(
            RECOMMENDED_CONTROL_ROOT, "R2R-1 control-root manifest binding"
        ),
        "result_parent_binding": dict(held_result_parent_binding)
        if held_result_parent_binding is not None
        else result_parent_binding(),
        "thermal92_metadata_binding": regular_file_metadata_binding(
            RECOMMENDED_THERMAL92, "R2R-1 thermal92 metadata-only binding"
        ),
        "attempt3_root_binding": directory_metadata_binding(
            ATTEMPT3_ROOT, "R2R-1 attempt3 metadata-only binding"
        ),
        "source_file_bindings": {
            name: regular_file_metadata_binding(
                path, f"R2R-1 source metadata-only binding {name}"
            )
            for name, path in sorted(R2R1_SOURCE_PATHS.items())
        },
        "dependency_file_bindings": {
            name: regular_file_metadata_binding(
                path, f"R2R-1 dependency metadata-only binding {name}"
            )
            for name, path in sorted(R2R1_DEPENDENCY_PATHS.items())
        },
    }
    if (
        payload["thermal92_metadata_binding"]["file_identity"].get("st_size")
        != THERMAL92_FILE_SIZE_BYTES
    ):
        raise ValueError("R2R-1 thermal92 byte size differs from the frozen file")
    _validate_manifest_bound_prefix_consistency(payload)
    return payload


def _strict_json_from_raw(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("ascii")
        payload = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict ASCII JSON") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return payload


def _validate_bound_control_authorization(
    manifest_path: Path, marker_path: Path
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, int],
    bytes,
    dict[str, int],
    tuple[dict[str, int], ...],
]:
    control = _lexical_path(
        RECOMMENDED_CONTROL_ROOT,
        "R2R-1 authorization control root",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if manifest_path.parent != control or marker_path.parent != control:
        raise PermissionError("R2R-1 authorization files are not control siblings")
    bound_control, control_fd, control_chain = _open_directory_chain(
        control, "R2R-1 authorization control root"
    )
    try:
        if set(os.listdir(control_fd)) != {manifest_path.name, marker_path.name}:
            raise PermissionError("R2R-1 control root inventory changed")
        manifest_raw, manifest_identity = _read_owned_regular_file_at_fd(
            control_fd,
            manifest_path.name,
            "R2R-1 freeze manifest",
            size_limit=8 * 1024 * 1024,
        )
        payload = _strict_json_from_raw(manifest_raw, "R2R-1 freeze manifest")
        dynamic_keys = {
            "control_root_binding",
            "result_parent_binding",
            "thermal92_metadata_binding",
            "attempt3_root_binding",
            "source_file_bindings",
            "dependency_file_bindings",
        }
        static_expected = _freeze_manifest_static_payload()
        if set(payload) != set(static_expected) | dynamic_keys:
            raise PermissionError("R2R-1 freeze manifest key set changed")
        static_payload = {
            key: value for key, value in payload.items() if key not in dynamic_keys
        }
        if not _json_type_exact_equal(static_payload, static_expected):
            raise PermissionError(
                "R2R-1 freeze manifest differs from live source/input closure"
            )
        _validate_manifest_bound_prefix_consistency(payload)
        control_binding = payload["control_root_binding"]
        if (
            not isinstance(control_binding, Mapping)
            or set(control_binding)
            != {"path", "directory_identity", "chain", "payload_bytes_read"}
            or control_binding.get("path") != str(bound_control)
            or type(control_binding.get("payload_bytes_read")) is not int
            or control_binding.get("payload_bytes_read") != 0
            or not _json_type_exact_equal(
                control_binding.get("directory_identity"),
                _directory_binding_identity(os.fstat(control_fd)),
            )
            or not _json_type_exact_equal(
                control_binding.get("chain"),
                [dict(item) for item in control_chain],
            )
        ):
            raise PermissionError("R2R-1 marker-bound control identity changed")
        marker_raw, marker_identity = _read_regular_file_at_fd(
            control_fd,
            marker_path.name,
            "R2R-1 external GO marker",
            size_limit=4096,
        )
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        if marker_raw != (manifest_sha + "\n").encode("ascii"):
            raise PermissionError(
                "R2R-1 external GO marker bytes do not bind the manifest"
            )
        final_manifest, _ = _read_owned_regular_file_at_fd(
            control_fd,
            manifest_path.name,
            "R2R-1 freeze manifest final check",
            size_limit=8 * 1024 * 1024,
            expected_identity=manifest_identity,
        )
        final_marker, _ = _read_regular_file_at_fd(
            control_fd,
            marker_path.name,
            "R2R-1 GO marker final check",
            size_limit=4096,
            expected_identity=marker_identity,
        )
        closed_manifest, _ = _read_owned_regular_file_at_fd(
            control_fd,
            manifest_path.name,
            "R2R-1 freeze manifest post-marker closure",
            size_limit=8 * 1024 * 1024,
            expected_identity=manifest_identity,
        )
        if (
            final_manifest != manifest_raw
            or final_marker != marker_raw
            or closed_manifest != manifest_raw
        ):
            raise PermissionError("R2R-1 control files changed during authorization")
        for name, expected_identity, owned in (
            (manifest_path.name, manifest_identity, True),
            (marker_path.name, marker_identity, False),
        ):
            identity_fn = _owned_regular_identity if owned else _stat_identity
            if not _json_type_exact_equal(
                identity_fn(
                    os.stat(name, dir_fd=control_fd, follow_symlinks=False)
                ),
                expected_identity,
            ):
                raise PermissionError(
                    "R2R-1 control sibling changed after byte closure"
                )
        if set(os.listdir(control_fd)) != {manifest_path.name, marker_path.name}:
            raise PermissionError("R2R-1 control inventory changed during authorization")
        _verify_directory_chain(
            bound_control, "R2R-1 authorization final control binding", control_chain
        )
        return (
            payload,
            manifest_raw,
            manifest_identity,
            marker_raw,
            marker_identity,
            control_chain,
        )
    finally:
        os.close(control_fd)


def validate_execution_authorization(
    freeze_manifest: Path,
    authorization_marker: Path,
    *,
    require_fresh_result_children: bool = False,
    publication_boundary: object | None = None,
) -> dict[str, Any]:
    if type(require_fresh_result_children) is not bool:
        raise TypeError("R2R-1 fresh-result authorization flag must be bool")
    manifest_lexical = _lexical_path(
        freeze_manifest,
        "R2R-1 freeze manifest",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    marker_lexical = _lexical_path(
        authorization_marker,
        "R2R-1 external GO marker",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    expected_manifest = _lexical_path(
        RECOMMENDED_FREEZE_MANIFEST,
        "R2R-1 frozen freeze-manifest path",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    expected_marker = _lexical_path(
        RECOMMENDED_AUTHORIZATION_MARKER,
        "R2R-1 frozen authorization-marker path",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if manifest_lexical != expected_manifest or marker_lexical != expected_marker:
        raise PermissionError(
            "R2R-1 authorization files differ from frozen production paths"
        )
    (
        payload,
        manifest_raw,
        manifest_identity,
        marker_raw,
        marker_identity,
        control_chain,
    ) = _validate_bound_control_authorization(
        manifest_lexical, marker_lexical
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    marker_sha = hashlib.sha256(marker_raw).hexdigest()
    # Fit output and external candidate are two children of one marker-bound
    # result parent.  They are sampled through one held fd at manifest time and
    # validated through one held fd here, so authorization cannot combine two
    # different phase snapshots of the same lexical parent.
    fit_target = _lexical_path(RECOMMENDED_FIT_OUTPUT_ROOT, "R2R-1 fit output path")
    candidate_target = _lexical_path(
        RECOMMENDED_CANDIDATE_ROOT, "R2R-1 candidate root path"
    )
    binding = payload["result_parent_binding"]
    if (
        fit_target.parent != candidate_target.parent
        or not isinstance(binding, Mapping)
        or set(binding)
        != {
            "path",
            "directory_identity",
            "chain",
            "fit_output_child_basename",
            "candidate_child_basename",
            "fit_output_child_absent_at_freeze",
            "candidate_child_absent_at_freeze",
            "payload_bytes_read",
        }
        or type(binding.get("path")) is not str
        or binding.get("path") != str(fit_target.parent)
        or type(binding.get("fit_output_child_basename")) is not str
        or binding.get("fit_output_child_basename") != fit_target.name
        or type(binding.get("candidate_child_basename")) is not str
        or binding.get("candidate_child_basename") != candidate_target.name
        or binding.get("fit_output_child_absent_at_freeze") is not True
        or binding.get("candidate_child_absent_at_freeze") is not True
        or type(binding.get("payload_bytes_read")) is not int
        or binding.get("payload_bytes_read") != 0
        or not isinstance(binding.get("chain"), list)
        or not binding["chain"]
        or not _json_type_exact_equal(
            binding.get("directory_identity"), binding["chain"][-1]
        )
    ):
        raise PermissionError("R2R-1 marker-bound result-parent binding changed")
    attempt3_binding = payload["attempt3_root_binding"]
    if (
        not isinstance(attempt3_binding, Mapping)
        or set(attempt3_binding)
        != {"path", "directory_identity", "chain", "payload_bytes_read"}
        or attempt3_binding.get("path")
        != str(_lexical_path(ATTEMPT3_ROOT, "R2R-1 authorized attempt3 path"))
        or type(attempt3_binding.get("payload_bytes_read")) is not int
        or attempt3_binding.get("payload_bytes_read") != 0
        or not isinstance(attempt3_binding.get("chain"), list)
        or not attempt3_binding["chain"]
        or not _json_type_exact_equal(
            attempt3_binding.get("directory_identity"),
            attempt3_binding["chain"][-1],
        )
    ):
        raise PermissionError("R2R-1 attempt3 metadata binding schema changed")

    def validate_closure_binding_collection(
        collection_key: str, paths: Mapping[str, Path]
    ) -> Mapping[str, Any]:
        collection = payload.get(collection_key)
        if not isinstance(collection, Mapping) or set(collection) != set(paths):
            raise PermissionError(f"R2R-1 {collection_key} key set changed")
        file_identity_keys = {
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
        parent_identity_keys = {"st_dev", "st_ino", "st_mode", "st_uid", "st_gid"}
        for name, expected_path in paths.items():
            item = collection.get(name)
            expected_lexical = _lexical_path(
                expected_path,
                f"R2R-1 {collection_key} path {name}",
                forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
            )
            if (
                not isinstance(item, Mapping)
                or set(item)
                != {
                    "path",
                    "file_identity",
                    "parent_chain",
                    "final_file_opened",
                    "payload_bytes_read",
                }
                or item.get("path") != str(expected_lexical)
                or item.get("final_file_opened") is not False
                or type(item.get("payload_bytes_read")) is not int
                or item.get("payload_bytes_read") != 0
                or not isinstance(item.get("file_identity"), Mapping)
                or set(item["file_identity"]) != file_identity_keys
                or any(type(value) is not int for value in item["file_identity"].values())
                or not isinstance(item.get("parent_chain"), list)
                or not item["parent_chain"]
                or any(
                    not isinstance(identity, Mapping)
                    or set(identity) != parent_identity_keys
                    or any(type(value) is not int for value in identity.values())
                    for identity in item["parent_chain"]
                )
            ):
                raise PermissionError(f"R2R-1 {collection_key}.{name} schema changed")
        return collection

    source_bindings = validate_closure_binding_collection(
        "source_file_bindings", R2R1_SOURCE_PATHS
    )
    dependency_bindings = validate_closure_binding_collection(
        "dependency_file_bindings", R2R1_DEPENDENCY_PATHS
    )
    thermal_path = _lexical_path(
        RECOMMENDED_THERMAL92,
        "R2R-1 authorized thermal92 path",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if payload["thermal92_canonical_path"] != str(thermal_path):
        raise PermissionError("R2R-1 manifest thermal92 path changed")
    thermal_metadata = payload["thermal92_metadata_binding"]
    thermal_file_identity_keys = {
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
    thermal_parent_identity_keys = {
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
    }
    if (
        not isinstance(thermal_metadata, Mapping)
        or set(thermal_metadata)
        != {
            "path",
            "file_identity",
            "parent_chain",
            "final_file_opened",
            "payload_bytes_read",
        }
        or thermal_metadata.get("path") != str(thermal_path)
        or thermal_metadata.get("final_file_opened") is not False
        or type(thermal_metadata.get("payload_bytes_read")) is not int
        or thermal_metadata.get("payload_bytes_read") != 0
        or not isinstance(thermal_metadata.get("file_identity"), Mapping)
        or set(thermal_metadata["file_identity"]) != thermal_file_identity_keys
        or any(
            type(value) is not int
            for value in thermal_metadata["file_identity"].values()
        )
        or thermal_metadata["file_identity"].get("st_size")
        != THERMAL92_FILE_SIZE_BYTES
        or not isinstance(thermal_metadata.get("parent_chain"), list)
        or len(thermal_metadata["parent_chain"]) != len(thermal_path.parent.parts)
        or any(
            not isinstance(identity, Mapping)
            or set(identity) != thermal_parent_identity_keys
            or any(type(value) is not int for value in identity.values())
            for identity in thermal_metadata["parent_chain"]
        )
    ):
        raise PermissionError("R2R-1 thermal92 metadata binding schema changed")
    # Hold all non-label authorization roots across the first thermal open.
    # This prevents a valid control snapshot from being replaced after its
    # first check but before access to the label-bearing file.
    with ExitStack() as held:
        bound_control, control_fd, observed_control_chain = _open_directory_chain(
            RECOMMENDED_CONTROL_ROOT,
            "R2R-1 held authorization control root",
            expected_chain=control_chain,
        )
        held.callback(os.close, control_fd)

        def hold_control_child(
            name: str,
            expected_identity: Mapping[str, int],
            label: str,
            *,
            owned: bool,
        ) -> int:
            before = os.stat(name, dir_fd=control_fd, follow_symlinks=False)
            identity_fn = _owned_regular_identity if owned else _stat_identity
            if not _json_type_exact_equal(
                identity_fn(before), dict(expected_identity)
            ):
                raise PermissionError(f"{label} identity changed before held open")
            descriptor = os.open(
                name, os.O_RDONLY | _NOFOLLOW, dir_fd=control_fd
            )
            try:
                if not _json_type_exact_equal(
                    identity_fn(os.fstat(descriptor)), dict(expected_identity)
                ):
                    raise PermissionError(f"{label} changed across held open")
                if owned:
                    _authoritative_owned_xattrs(descriptor, label=label)
                held.callback(os.close, descriptor)
                return descriptor
            except BaseException:
                os.close(descriptor)
                raise

        manifest_fd = hold_control_child(
            manifest_lexical.name,
            manifest_identity,
            "R2R-1 held freeze manifest",
            owned=True,
        )
        marker_fd = hold_control_child(
            marker_lexical.name,
            marker_identity,
            "R2R-1 held GO marker",
            owned=False,
        )
        bound_result, result_fd, result_chain = _open_directory_chain(
            fit_target.parent,
            "R2R-1 held shared result parent",
            expected_chain=binding["chain"],
        )
        held.callback(os.close, result_fd)
        bound_attempt3, attempt3_fd, attempt3_chain = _open_directory_chain(
            ATTEMPT3_ROOT,
            "R2R-1 held attempt3 root",
            expected_chain=attempt3_binding["chain"],
        )
        held.callback(os.close, attempt3_fd)

        held_closure_files: list[dict[str, Any]] = []

        def read_held_closure_file(record: Mapping[str, Any]) -> None:
            descriptor = int(record["file_fd"])
            expected_identity = record["identity"]
            if not _json_type_exact_equal(
                _stat_identity(os.fstat(descriptor)), expected_identity
            ):
                raise PermissionError(
                    f"R2R-1 held closure identity changed: {record['label']}"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            total = 0
            while True:
                block = os.read(
                    descriptor,
                    min(1024 * 1024, 2 * 1024 * 1024 + 1 - total),
                )
                if not block:
                    break
                total += len(block)
                if total > 2 * 1024 * 1024:
                    raise PermissionError("R2R-1 source closure file exceeds 2 MiB")
                digest.update(block)
            entry = os.stat(
                record["source"].name,
                dir_fd=int(record["parent_fd"]),
                follow_symlinks=False,
            )
            if (
                digest.hexdigest() != record["sha256"]
                or not _json_type_exact_equal(
                    _stat_identity(os.fstat(descriptor)), expected_identity
                )
                or not _json_type_exact_equal(
                    _stat_identity(entry), expected_identity
                )
            ):
                raise PermissionError(
                    f"R2R-1 source closure changed: {record['label']}"
                )

        for collection_label, paths, bindings, hashes in (
            (
                "source",
                R2R1_SOURCE_PATHS,
                source_bindings,
                payload["source_sha256"],
            ),
            (
                "dependency",
                R2R1_DEPENDENCY_PATHS,
                dependency_bindings,
                payload["dependency_sha256"],
            ),
        ):
            for name, path in sorted(paths.items()):
                item = bindings[name]
                source, file_fd, parent_fd, identity, parent_chain = (
                    _open_bound_regular_file(
                        path,
                        f"R2R-1 held {collection_label} closure {name}",
                        expected_parent_chain=item["parent_chain"],
                        expected_file_identity=item["file_identity"],
                    )
                )
                held.callback(os.close, parent_fd)
                held.callback(os.close, file_fd)
                record = {
                    "label": f"{collection_label}.{name}",
                    "source": source,
                    "file_fd": file_fd,
                    "parent_fd": parent_fd,
                    "identity": identity,
                    "parent_chain": [dict(value) for value in parent_chain],
                    "sha256": hashes[name],
                }
                read_held_closure_file(record)
                held_closure_files.append(record)

        def reread_held_control_child(
            descriptor: int,
            name: str,
            expected_identity: Mapping[str, int],
            expected_raw: bytes,
            size_limit: int,
            label: str,
            *,
            owned: bool,
        ) -> None:
            identity_fn = _owned_regular_identity if owned else _stat_identity
            if not _json_type_exact_equal(
                identity_fn(os.fstat(descriptor)), dict(expected_identity)
            ):
                raise PermissionError(f"{label} held identity changed")
            if owned:
                _authoritative_owned_xattrs(descriptor, label=label)
            os.lseek(descriptor, 0, os.SEEK_SET)
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
                    raise PermissionError(f"{label} exceeded its frozen bound")
                chunks.append(block)
            observed_raw = b"".join(chunks)
            entry = os.stat(name, dir_fd=control_fd, follow_symlinks=False)
            if owned:
                _authoritative_owned_xattrs(descriptor, label=label)
            if (
                observed_raw != expected_raw
                or not _json_type_exact_equal(
                    identity_fn(os.fstat(descriptor)), dict(expected_identity)
                )
                or not _json_type_exact_equal(
                    identity_fn(entry), dict(expected_identity)
                )
            ):
                raise PermissionError(f"{label} bytes or identity changed")

        def revalidate_nonlabel_authorization() -> None:
            for record in held_closure_files:
                _verify_directory_chain(
                    record["source"].parent,
                    f"R2R-1 held closure parent rebind {record['label']}",
                    record["parent_chain"],
                )
                read_held_closure_file(record)
            if (
                not _json_type_exact_equal(
                    _directory_binding_identity(os.fstat(control_fd)),
                    payload["control_root_binding"]["directory_identity"],
                )
                or not _json_type_exact_equal(
                    [dict(item) for item in observed_control_chain], control_chain
                )
                or not _json_type_exact_equal(
                    _directory_binding_identity(os.fstat(result_fd)),
                    binding["directory_identity"],
                )
                or not _json_type_exact_equal(
                    [dict(item) for item in result_chain], binding["chain"]
                )
                or not _json_type_exact_equal(
                    _directory_binding_identity(os.fstat(attempt3_fd)),
                    attempt3_binding["directory_identity"],
                )
                or not _json_type_exact_equal(
                    [dict(item) for item in attempt3_chain],
                    attempt3_binding["chain"],
                )
            ):
                raise PermissionError("R2R-1 held authorization identity changed")
            _verify_directory_chain(
                bound_result,
                "R2R-1 held result-parent lexical rebind",
                result_chain,
            )
            _verify_directory_chain(
                bound_attempt3,
                "R2R-1 held attempt3 lexical rebind",
                attempt3_chain,
            )
            _verify_directory_chain(
                bound_control,
                "R2R-1 held control-root lexical rebind",
                observed_control_chain,
            )
            if require_fresh_result_children:
                for child_name in (
                    binding["fit_output_child_basename"],
                    binding["candidate_child_basename"],
                ):
                    try:
                        os.stat(
                            child_name,
                            dir_fd=result_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        continue
                    raise FileExistsError(
                        "R2R-1 marker-bound result child appeared before fresh fit"
                    )
            # Control bytes are intentionally the last substantive check
            # before label-file access.  Both child descriptors stay held
            # across all longer result/attempt3 rebinds above.
            if set(os.listdir(control_fd)) != {
                manifest_lexical.name,
                marker_lexical.name,
            }:
                raise PermissionError("R2R-1 control inventory changed")
            reread_held_control_child(
                manifest_fd,
                manifest_lexical.name,
                manifest_identity,
                manifest_raw,
                8 * 1024 * 1024,
                "R2R-1 held freeze manifest recheck",
                owned=True,
            )
            reread_held_control_child(
                marker_fd,
                marker_lexical.name,
                marker_identity,
                marker_raw,
                4096,
                "R2R-1 held GO marker recheck",
                owned=False,
            )
            # Close the whole source/dependency set after all long hash reads.
            # A later sibling cannot replace an already-read lexical basename
            # or parent while leaving only the held old descriptor valid.
            for record in held_closure_files:
                _verify_directory_chain(
                    record["source"].parent,
                    f"R2R-1 final closure parent rebind {record['label']}",
                    record["parent_chain"],
                )
            _verify_directory_chain(
                bound_result,
                "R2R-1 final held result-parent lexical rebind",
                result_chain,
            )
            _verify_directory_chain(
                bound_attempt3,
                "R2R-1 final held attempt3 lexical rebind",
                attempt3_chain,
            )
            _verify_directory_chain(
                bound_control,
                "R2R-1 final held control-root lexical rebind",
                observed_control_chain,
            )
            for record in held_closure_files:
                parent_identity = _directory_binding_identity(
                    os.fstat(int(record["parent_fd"]))
                )
                expected_parent_identity = record["parent_chain"][-1]
                entry = os.stat(
                    record["source"].name,
                    dir_fd=int(record["parent_fd"]),
                    follow_symlinks=False,
                )
                if (
                    not _json_type_exact_equal(
                        parent_identity, expected_parent_identity
                    )
                    or not _json_type_exact_equal(
                        _stat_identity(entry), record["identity"]
                    )
                    or not _json_type_exact_equal(
                        _stat_identity(os.fstat(int(record["file_fd"]))),
                        record["identity"],
                    )
                ):
                    raise PermissionError(
                        f"R2R-1 held closure identity changed: {record['label']}"
                    )
            # Close control bytes after every long source/result/attempt3 walk.
            # Manifest is intentionally read again after the external marker:
            # owned8 excludes ctime, so an identity-only sibling pass cannot
            # detect a same-size manifest byte change with restored mtime.
            reread_held_control_child(
                manifest_fd,
                manifest_lexical.name,
                manifest_identity,
                manifest_raw,
                8 * 1024 * 1024,
                "R2R-1 final held freeze manifest before marker",
                owned=True,
            )
            reread_held_control_child(
                marker_fd,
                marker_lexical.name,
                marker_identity,
                marker_raw,
                4096,
                "R2R-1 final held GO marker",
                owned=False,
            )
            reread_held_control_child(
                manifest_fd,
                manifest_lexical.name,
                manifest_identity,
                manifest_raw,
                8 * 1024 * 1024,
                "R2R-1 final held freeze manifest after marker",
                owned=True,
            )
            # Common second pass for the two control siblings.  In particular,
            # a marker-read hook cannot mutate an already-read manifest and
            # reach the thermal open with only a self-consistent inventory.
            for descriptor, name, expected_identity, owned in (
                (manifest_fd, manifest_lexical.name, manifest_identity, True),
                (marker_fd, marker_lexical.name, marker_identity, False),
            ):
                identity_fn = _owned_regular_identity if owned else _stat_identity
                entry = os.stat(name, dir_fd=control_fd, follow_symlinks=False)
                if (
                    not _json_type_exact_equal(
                        identity_fn(os.fstat(descriptor)), expected_identity
                    )
                    or not _json_type_exact_equal(
                        identity_fn(entry), expected_identity
                    )
                ):
                    raise PermissionError(
                        "R2R-1 control sibling changed after joint reread"
                    )
                if owned:
                    _authoritative_owned_xattrs(
                        descriptor, label="R2R-1 held freeze manifest"
                    )
            if set(os.listdir(control_fd)) != {
                manifest_lexical.name,
                marker_lexical.name,
            }:
                raise PermissionError("R2R-1 control inventory changed after reread")
            if require_fresh_result_children:
                for child_name in (
                    binding["fit_output_child_basename"],
                    binding["candidate_child_basename"],
                ):
                    try:
                        os.stat(
                            child_name,
                            dir_fd=result_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        continue
                    raise FileExistsError(
                        "R2R-1 marker-bound result child appeared before fresh fit"
                    )

        def close_control_after_publication() -> None:
            """Close held control bytes after the long publication-tree walk."""
            reread_held_control_child(
                manifest_fd,
                manifest_lexical.name,
                manifest_identity,
                manifest_raw,
                8 * 1024 * 1024,
                "R2R-1 post-publication freeze manifest before marker",
                owned=True,
            )
            reread_held_control_child(
                marker_fd,
                marker_lexical.name,
                marker_identity,
                marker_raw,
                4096,
                "R2R-1 post-publication GO marker",
                owned=False,
            )
            reread_held_control_child(
                manifest_fd,
                manifest_lexical.name,
                manifest_identity,
                manifest_raw,
                8 * 1024 * 1024,
                "R2R-1 post-publication freeze manifest after marker",
                owned=True,
            )
            if (
                not _json_type_exact_equal(
                    _directory_binding_identity(os.fstat(control_fd)),
                    payload["control_root_binding"]["directory_identity"],
                )
                or not _json_type_exact_equal(
                    _directory_binding_identity(os.fstat(result_fd)),
                    binding["directory_identity"],
                )
            ):
                raise PermissionError(
                    "R2R-1 post-publication held directory binding changed"
                )
            for descriptor, name, expected_identity, owned in (
                (manifest_fd, manifest_lexical.name, manifest_identity, True),
                (marker_fd, marker_lexical.name, marker_identity, False),
            ):
                identity_fn = _owned_regular_identity if owned else _stat_identity
                entry = os.stat(name, dir_fd=control_fd, follow_symlinks=False)
                if (
                    not _json_type_exact_equal(
                        identity_fn(os.fstat(descriptor)), expected_identity
                    )
                    or not _json_type_exact_equal(
                        identity_fn(entry), expected_identity
                    )
                ):
                    raise PermissionError(
                        "R2R-1 control sibling changed after publication closure"
                    )
                if owned:
                    _authoritative_owned_xattrs(
                        descriptor,
                        label="R2R-1 post-publication held freeze manifest",
                    )
            if set(os.listdir(control_fd)) != {
                manifest_lexical.name,
                marker_lexical.name,
            }:
                raise PermissionError(
                    "R2R-1 control inventory changed after publication closure"
                )
            if require_fresh_result_children:
                for child_name in (
                    binding["fit_output_child_basename"],
                    binding["candidate_child_basename"],
                ):
                    try:
                        os.stat(
                            child_name,
                            dir_fd=result_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        continue
                    raise FileExistsError(
                        "R2R-1 result child appeared at the thermal-open boundary"
                    )

        def close_publication_control_union() -> None:
            """Close the held authority set under one payload-write guard."""
            with _HeldRegularWriteGuard(
                "R2R-1 control/publication authority closure"
            ) as write_guard:
                write_guard.watch_owned_descriptor(
                    manifest_fd,
                    expected_identity=manifest_identity,
                    label="R2R-1 held freeze manifest",
                )
                write_guard.watch_external_descriptor(
                    marker_fd,
                    expected_identity=marker_identity,
                    label="R2R-1 held GO marker",
                )
                for record in held_closure_files:
                    write_guard.watch_external_descriptor(
                        int(record["file_fd"]),
                        expected_identity=record["identity"],
                        label=f"R2R-1 held closure {record['label']}",
                    )
                if publication_boundary is not None:
                    _watch_fit_publication_boundary(
                        publication_boundary, write_guard
                    )
                close_control_after_publication()
                if publication_boundary is not None:
                    _require_fit_publication_boundary(publication_boundary)
                    _require_fit_publication_boundary(publication_boundary)
                close_control_after_publication()
                if publication_boundary is not None:
                    _require_fit_publication_boundary(publication_boundary)
                    _close_fit_publication_boundary(publication_boundary)

        revalidate_nonlabel_authorization()
        close_publication_control_union()
        # This is the first authorized open of the label-bearing thermal file.
        # The entry is compared with marker-bound metadata before openat.
        (
            thermal_source,
            thermal_fd,
            thermal_parent_fd,
            thermal_identity,
            thermal_parent_chain,
        ) = _open_bound_regular_file(
            thermal_path,
            "R2R-1 post-GO thermal92 binding",
            expected_parent_chain=thermal_metadata["parent_chain"],
            expected_file_identity=thermal_metadata["file_identity"],
        )
        _finish_bound_regular_file(
            thermal_source,
            thermal_fd,
            thermal_parent_fd,
            thermal_identity,
            thermal_parent_chain,
            "R2R-1 post-GO thermal92 binding",
        )
        revalidate_nonlabel_authorization()
        close_publication_control_union()
        thermal_binding = {
            "path": str(thermal_source),
            "file_identity": thermal_identity,
            "parent_chain": [dict(item) for item in thermal_parent_chain],
        }
        return {
            "format": "graphene_r2r1_external_execution_authorization_v2",
            "freeze_manifest_sha256": manifest_sha,
            "authorization_marker_sha256": marker_sha,
            "freeze_manifest_file_identity": manifest_identity,
            "authorization_marker_file_identity": marker_identity,
            "freeze_manifest_parent_chain": [dict(item) for item in control_chain],
            "authorization_marker_parent_chain": [dict(item) for item in control_chain],
            "control_root_binding": payload["control_root_binding"],
            "result_parent_binding": payload["result_parent_binding"],
            "thermal92_file_binding": thermal_binding,
            "attempt3_root_binding": {
                "path": attempt3_binding["path"],
                "directory_identity": dict(attempt3_binding["directory_identity"]),
                "chain": [dict(item) for item in attempt3_chain],
            },
            "fit_authorized": True,
            "thermal92_force_labels_authorized": True,
            "energy_labels_authorized": False,
            "development_or_held_access_authorized": False,
            "source_sha256": payload["source_sha256"],
        }


def _build_fit_publication_boundary():
    """Issue opaque, in-process publication guards checked before labels/SVD."""
    issued: dict[object, dict[str, Any]] = {}
    seal = object()

    class FitPublicationBoundary:
        __slots__ = ()

        def __new__(cls, token: object | None = None):
            if token is not seal:
                raise PermissionError(
                    "R2R-1 fit publication boundary cannot be constructed"
                )
            return super().__new__(cls)

    def json_copy(value: Any, label: str) -> Any:
        raw = canonical_json_bytes(value)
        try:
            return json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} is not strict JSON data") from error

    def issue(
        *,
        release_parent_fd: int,
        release_parent_path: Path,
        release_parent_chain: Sequence[Mapping[str, int]],
        root_fd: int,
        root_path: Path,
        expected_root_identity: Mapping[str, int],
        expected_root_snapshot: Mapping[str, Mapping[str, Any]],
        candidate_parent_fd: int,
        candidate_parent_path: Path,
        candidate_parent_chain: Sequence[Mapping[str, int]],
        candidate_name: str,
        candidate_state: str,
        candidate_fd: int | None = None,
        candidate_path: Path | None = None,
        candidate_chain: Sequence[Mapping[str, int]] | None = None,
        anchor_name: str | None = None,
        anchor_identity: Mapping[str, int] | None = None,
        anchor_raw: bytes | None = None,
    ) -> object:
        frozen_root = _lexical_path(
            RECOMMENDED_FIT_OUTPUT_ROOT,
            "R2R-1 guarded fit output root",
        )
        frozen_candidate = _lexical_path(
            RECOMMENDED_CANDIDATE_ROOT,
            "R2R-1 guarded candidate root",
            forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
        )
        observed_root = _lexical_path(root_path, "R2R-1 publication root")
        observed_release_parent = _lexical_path(
            release_parent_path, "R2R-1 publication parent"
        )
        observed_candidate_parent = _lexical_path(
            candidate_parent_path, "R2R-1 candidate publication parent"
        )
        if (
            observed_root != frozen_root
            or observed_release_parent != frozen_root.parent
            or observed_candidate_parent != frozen_candidate.parent
            or candidate_name != frozen_candidate.name
            or candidate_state not in {"absent", "existing"}
        ):
            raise PermissionError("R2R-1 publication boundary path/state changed")
        if candidate_state == "existing":
            if (
                candidate_fd is None
                or candidate_path is None
                or candidate_chain is None
                or anchor_name != RECOMMENDED_RELEASE_MANIFEST.name
                or anchor_identity is None
                or anchor_raw is None
                or len(anchor_raw) > 2 * 1024 * 1024
                or _lexical_path(candidate_path, "R2R-1 existing candidate root")
                != frozen_candidate
            ):
                raise ValueError("R2R-1 existing publication boundary is incomplete")
        elif any(
            value is not None
            for value in (
                candidate_fd,
                candidate_path,
                candidate_chain,
                anchor_name,
                anchor_identity,
                anchor_raw,
            )
        ):
            raise ValueError("R2R-1 absent publication boundary carried state")
        token = FitPublicationBoundary(seal)
        issued[token] = {
            "release_parent_fd": release_parent_fd,
            "release_parent_path": observed_release_parent,
            "release_parent_chain": json_copy(
                list(release_parent_chain), "R2R-1 publication parent chain"
            ),
            "root_fd": root_fd,
            "root_path": observed_root,
            "expected_root_identity": json_copy(
                dict(expected_root_identity), "R2R-1 publication root identity"
            ),
            "expected_root_snapshot": json_copy(
                dict(expected_root_snapshot), "R2R-1 publication snapshot"
            ),
            "candidate_parent_fd": candidate_parent_fd,
            "candidate_parent_path": observed_candidate_parent,
            "candidate_parent_chain": json_copy(
                list(candidate_parent_chain), "R2R-1 candidate parent chain"
            ),
            "candidate_name": candidate_name,
            "candidate_state": candidate_state,
            "candidate_fd": candidate_fd,
            "candidate_path": Path(candidate_path) if candidate_path is not None else None,
            "candidate_chain": json_copy(
                list(candidate_chain), "R2R-1 candidate chain"
            )
            if candidate_chain is not None
            else None,
            "anchor_name": anchor_name,
            "anchor_identity": json_copy(
                dict(anchor_identity), "R2R-1 candidate anchor identity"
            )
            if anchor_identity is not None
            else None,
            "anchor_raw": bytes(anchor_raw) if anchor_raw is not None else None,
        }
        return token

    def verify_snapshot(
        directory_fd: int,
        snapshot: Mapping[str, Mapping[str, Any]],
        *,
        prefix: str = "",
        budget: dict[str, int] | None = None,
    ) -> None:
        if budget is None:
            budget = {"files": 0, "bytes": 0}
        prefix_slash = f"{prefix}/" if prefix else ""
        direct: dict[str, str] = {}
        for relative in snapshot:
            if not relative.startswith(prefix_slash):
                continue
            suffix = relative[len(prefix_slash) :]
            if suffix and "/" not in suffix:
                direct[suffix] = relative
        if set(os.listdir(directory_fd)) != set(direct):
            raise ValueError("R2R-1 guarded release inventory changed")
        for name, relative in sorted(direct.items()):
            record = snapshot[relative]
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if record.get("kind") == "directory":
                observed = _directory_binding_identity(before)
                expected = record.get("identity")
                if expected is None:
                    expected = {key: record[key] for key in observed}
                if (
                    stat.S_ISLNK(before.st_mode)
                    or not stat.S_ISDIR(before.st_mode)
                    or not _json_type_exact_equal(observed, expected)
                ):
                    raise ValueError("R2R-1 guarded release directory changed")
                child_fd = os.open(
                    name, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=directory_fd
                )
                try:
                    if not _json_type_exact_equal(
                        _directory_binding_identity(os.fstat(child_fd)), expected
                    ):
                        raise ValueError(
                            "R2R-1 guarded release directory changed across open"
                        )
                    verify_snapshot(
                        child_fd, snapshot, prefix=relative, budget=budget
                    )
                    rebound = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if not _json_type_exact_equal(
                        _directory_binding_identity(rebound), expected
                    ):
                        raise ValueError(
                            "R2R-1 guarded release directory changed after walk"
                        )
                finally:
                    os.close(child_fd)
            elif record.get("kind") == "file":
                expected = record.get("identity")
                if expected is None:
                    expected = {
                        key: record[key]
                        for key in _owned_regular_identity(before)
                    }
                if (
                    stat.S_ISLNK(before.st_mode)
                    or not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or not _json_type_exact_equal(
                        _owned_regular_identity(before), expected
                    )
                ):
                    raise ValueError("R2R-1 guarded release file changed")
                size = record.get("size")
                digest = record.get("sha256")
                if (
                    type(size) is not int
                    or size < 0
                    or size > 64 * 1024 * 1024
                    or size != expected.get("st_size")
                    or type(digest) is not str
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError("R2R-1 guarded release file schema changed")
                budget["files"] += 1
                budget["bytes"] += size
                if budget["files"] > 16 or budget["bytes"] > 128 * 1024 * 1024:
                    raise ValueError("R2R-1 guarded release snapshot exceeds limits")
                file_fd = os.open(
                    name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd
                )
                try:
                    if not _json_type_exact_equal(
                        _owned_regular_identity(os.fstat(file_fd)), expected
                    ):
                        raise ValueError(
                            "R2R-1 guarded release file changed across open"
                        )
                    _authoritative_owned_xattrs(
                        file_fd, label="R2R-1 guarded release file"
                    )
                    hasher = hashlib.sha256()
                    total = 0
                    while True:
                        block = os.read(
                            file_fd, min(1024 * 1024, size + 1 - total)
                        )
                        if not block:
                            break
                        total += len(block)
                        if total > size:
                            raise ValueError(
                                "R2R-1 guarded release file exceeds frozen size"
                            )
                        hasher.update(block)
                    if total != size or hasher.hexdigest() != digest:
                        raise ValueError("R2R-1 guarded release bytes changed")
                    _authoritative_owned_xattrs(
                        file_fd, label="R2R-1 guarded release file"
                    )
                    if not _json_type_exact_equal(
                        _owned_regular_identity(os.fstat(file_fd)), expected
                    ):
                        raise ValueError(
                            "R2R-1 guarded release file changed while read"
                        )
                finally:
                    os.close(file_fd)
                rebound = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if not _json_type_exact_equal(
                    _owned_regular_identity(rebound), expected
                ):
                    raise ValueError(
                        "R2R-1 guarded release file entry changed after read"
                    )
            else:
                raise ValueError("R2R-1 guarded release snapshot kind changed")

    def require(value: object) -> None:
        if type(value) is not FitPublicationBoundary or value not in issued:
            raise PermissionError(
                "R2R-1 production fit requires an issued publication boundary"
            )
        policy = issued[value]
        release_parent_chain = policy["release_parent_chain"]
        candidate_parent_chain = policy["candidate_parent_chain"]
        root_chain = [*release_parent_chain, policy["expected_root_identity"]]
        # Long lexical walks precede the final held-fd checks.
        _verify_directory_chain(
            policy["release_parent_path"],
            "R2R-1 guarded release-parent binding",
            release_parent_chain,
        )
        _verify_directory_chain(
            policy["root_path"], "R2R-1 guarded release-root binding", root_chain
        )
        _verify_directory_chain(
            policy["candidate_parent_path"],
            "R2R-1 guarded candidate-parent binding",
            candidate_parent_chain,
        )
        if policy["candidate_state"] == "existing":
            _verify_directory_chain(
                policy["candidate_path"],
                "R2R-1 guarded existing candidate binding",
                policy["candidate_chain"],
            )
        release_parent_identity = _directory_binding_identity(
            os.fstat(policy["release_parent_fd"])
        )
        candidate_parent_identity = _directory_binding_identity(
            os.fstat(policy["candidate_parent_fd"])
        )
        held_root_identity = _directory_binding_identity(os.fstat(policy["root_fd"]))
        root_entry_identity = _directory_binding_identity(
            os.stat(
                policy["root_path"].name,
                dir_fd=policy["release_parent_fd"],
                follow_symlinks=False,
            )
        )
        if (
            not _json_type_exact_equal(
                release_parent_identity, release_parent_chain[-1]
            )
            or not _json_type_exact_equal(
                candidate_parent_identity, candidate_parent_chain[-1]
            )
            or not _json_type_exact_equal(
                held_root_identity, policy["expected_root_identity"]
            )
            or not _json_type_exact_equal(
                root_entry_identity, policy["expected_root_identity"]
            )
        ):
            raise ValueError("R2R-1 guarded stable publication binding changed")
        verify_snapshot(policy["root_fd"], policy["expected_root_snapshot"])
        # The first recursive pass can spend time in a later checkpoint child
        # after it has already inspected an earlier science file.  Rewalk the
        # complete held tree against the same frozen snapshot, then perform a
        # short root/parent closure below.  No payload read or solve occurs
        # between this second identity-only pass and the candidate check.
        verify_snapshot(policy["root_fd"], policy["expected_root_snapshot"])
        if set(os.listdir(policy["root_fd"])) != {
            name
            for name in policy["expected_root_snapshot"]
            if "/" not in name
        }:
            raise ValueError("R2R-1 guarded release inventory changed at closure")
        release_parent_identity = _directory_binding_identity(
            os.fstat(policy["release_parent_fd"])
        )
        candidate_parent_identity = _directory_binding_identity(
            os.fstat(policy["candidate_parent_fd"])
        )
        held_root_identity = _directory_binding_identity(os.fstat(policy["root_fd"]))
        root_entry_identity = _directory_binding_identity(
            os.stat(
                policy["root_path"].name,
                dir_fd=policy["release_parent_fd"],
                follow_symlinks=False,
            )
        )
        if (
            not _json_type_exact_equal(
                release_parent_identity, release_parent_chain[-1]
            )
            or not _json_type_exact_equal(
                candidate_parent_identity, candidate_parent_chain[-1]
            )
            or not _json_type_exact_equal(
                held_root_identity, policy["expected_root_identity"]
            )
            or not _json_type_exact_equal(
                root_entry_identity, policy["expected_root_identity"]
            )
        ):
            raise ValueError("R2R-1 guarded publication changed after tree walk")
        if policy["candidate_state"] == "absent":
            try:
                os.stat(
                    policy["candidate_name"],
                    dir_fd=policy["candidate_parent_fd"],
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            raise FileExistsError("R2R-1 candidate appeared before label access")
        candidate_binding = _directory_binding_identity(
            os.fstat(policy["candidate_fd"])
        )
        candidate_entry = _directory_binding_identity(
            os.stat(
                policy["candidate_name"],
                dir_fd=policy["candidate_parent_fd"],
                follow_symlinks=False,
            )
        )
        if (
            not _json_type_exact_equal(candidate_binding, policy["candidate_chain"][-1])
            or not _json_type_exact_equal(
                candidate_entry, policy["candidate_chain"][-1]
            )
            or set(os.listdir(policy["candidate_fd"])) != {policy["anchor_name"]}
        ):
            raise ValueError("R2R-1 guarded candidate binding changed")
        anchor_entry = os.stat(
            policy["anchor_name"],
            dir_fd=policy["candidate_fd"],
            follow_symlinks=False,
        )
        if not _json_type_exact_equal(
            _owned_regular_identity(anchor_entry), policy["anchor_identity"]
        ):
            raise ValueError("R2R-1 guarded candidate anchor changed")
        anchor_fd = os.open(
            policy["anchor_name"],
            os.O_RDONLY | _NOFOLLOW,
            dir_fd=policy["candidate_fd"],
        )
        try:
            if not _json_type_exact_equal(
                _owned_regular_identity(os.fstat(anchor_fd)),
                policy["anchor_identity"],
            ):
                raise ValueError("R2R-1 guarded candidate anchor changed across open")
            _authoritative_owned_xattrs(
                anchor_fd, label="R2R-1 guarded candidate anchor"
            )
            expected_raw = policy["anchor_raw"]
            chunks: list[bytes] = []
            remaining = len(expected_raw) + 1
            while remaining > 0:
                block = os.read(anchor_fd, min(1024 * 1024, remaining))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            if b"".join(chunks) != expected_raw:
                raise ValueError("R2R-1 guarded candidate anchor bytes changed")
            _authoritative_owned_xattrs(
                anchor_fd, label="R2R-1 guarded candidate anchor"
            )
            if not _json_type_exact_equal(
                _owned_regular_identity(os.fstat(anchor_fd)),
                policy["anchor_identity"],
            ):
                raise ValueError("R2R-1 guarded candidate anchor changed while read")
        finally:
            os.close(anchor_fd)
        rebound_anchor = os.stat(
            policy["anchor_name"],
            dir_fd=policy["candidate_fd"],
            follow_symlinks=False,
        )
        if not _json_type_exact_equal(
            _owned_regular_identity(rebound_anchor), policy["anchor_identity"]
        ):
            raise ValueError("R2R-1 guarded candidate anchor entry changed")

    def close(value: object) -> None:
        """Perform the short held publication closure after another long tree."""
        if type(value) is not FitPublicationBoundary or value not in issued:
            raise PermissionError(
                "R2R-1 production fit requires an issued publication boundary"
            )
        policy = issued[value]
        release_parent_identity = _directory_binding_identity(
            os.fstat(policy["release_parent_fd"])
        )
        candidate_parent_identity = _directory_binding_identity(
            os.fstat(policy["candidate_parent_fd"])
        )
        held_root_identity = _directory_binding_identity(os.fstat(policy["root_fd"]))
        root_entry_identity = _directory_binding_identity(
            os.stat(
                policy["root_path"].name,
                dir_fd=policy["release_parent_fd"],
                follow_symlinks=False,
            )
        )
        expected_root_names = {
            name
            for name in policy["expected_root_snapshot"]
            if "/" not in name
        }
        if (
            not _json_type_exact_equal(
                release_parent_identity, policy["release_parent_chain"][-1]
            )
            or not _json_type_exact_equal(
                candidate_parent_identity, policy["candidate_parent_chain"][-1]
            )
            or not _json_type_exact_equal(
                held_root_identity, policy["expected_root_identity"]
            )
            or not _json_type_exact_equal(
                root_entry_identity, policy["expected_root_identity"]
            )
            or set(os.listdir(policy["root_fd"])) != expected_root_names
        ):
            raise ValueError("R2R-1 guarded publication changed at final closure")
        if policy["candidate_state"] == "absent":
            try:
                os.stat(
                    policy["candidate_name"],
                    dir_fd=policy["candidate_parent_fd"],
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            raise FileExistsError("R2R-1 candidate appeared before label access")
        candidate_binding = _directory_binding_identity(
            os.fstat(policy["candidate_fd"])
        )
        candidate_entry = _directory_binding_identity(
            os.stat(
                policy["candidate_name"],
                dir_fd=policy["candidate_parent_fd"],
                follow_symlinks=False,
            )
        )
        if (
            not _json_type_exact_equal(
                candidate_binding, policy["candidate_chain"][-1]
            )
            or not _json_type_exact_equal(
                candidate_entry, policy["candidate_chain"][-1]
            )
            or set(os.listdir(policy["candidate_fd"])) != {policy["anchor_name"]}
        ):
            raise ValueError("R2R-1 guarded candidate changed at final closure")
        anchor_entry = os.stat(
            policy["anchor_name"],
            dir_fd=policy["candidate_fd"],
            follow_symlinks=False,
        )
        if not _json_type_exact_equal(
            _owned_regular_identity(anchor_entry), policy["anchor_identity"]
        ):
            raise ValueError("R2R-1 guarded candidate anchor changed at closure")
        anchor_fd = os.open(
            policy["anchor_name"],
            os.O_RDONLY | _NOFOLLOW,
            dir_fd=policy["candidate_fd"],
        )
        try:
            if not _json_type_exact_equal(
                _owned_regular_identity(os.fstat(anchor_fd)),
                policy["anchor_identity"],
            ):
                raise ValueError(
                    "R2R-1 guarded candidate anchor changed across final open"
                )
            _authoritative_owned_xattrs(
                anchor_fd, label="R2R-1 guarded candidate anchor final closure"
            )
        finally:
            os.close(anchor_fd)

    def watch(value: object, guard: _HeldRegularWriteGuard) -> None:
        """Register every owned publication member in an outer union guard."""
        if type(value) is not FitPublicationBoundary or value not in issued:
            raise PermissionError(
                "R2R-1 production fit requires an issued publication boundary"
            )
        policy = issued[value]
        _watch_owned_snapshot_fd(
            guard,
            policy["root_fd"],
            policy["expected_root_snapshot"],
            label="R2R-1 guarded release snapshot",
        )
        if policy["candidate_state"] == "existing":
            guard.watch_owned_at(
                policy["candidate_fd"],
                policy["anchor_name"],
                expected_identity=policy["anchor_identity"],
                label="R2R-1 guarded candidate anchor",
            )

    return issue, require, close, watch


(
    _issue_fit_publication_boundary,
    _require_fit_publication_boundary,
    _close_fit_publication_boundary,
    _watch_fit_publication_boundary,
) = _build_fit_publication_boundary()


def _build_fit_authorization_boundary():
    """Keep the production token type and seal inside a lexical closure."""
    seal = object()

    class FitAuthorization:
        __slots__ = (
            "receipt",
            "freeze_manifest",
            "authorization_marker",
            "_receipt_bytes",
            "_seal",
            "_publication_boundary",
        )

        def __init__(
            self,
            receipt: Mapping[str, Any],
            freeze_manifest: Path,
            authorization_marker: Path,
            token: object,
            publication_boundary: object | None,
        ) -> None:
            if token is not seal:
                raise PermissionError("R2R-1 fit authorization cannot be constructed")
            self._receipt_bytes = canonical_json_bytes(receipt)
            self.receipt = _strict_json_from_raw(
                self._receipt_bytes, "R2R-1 sealed authorization receipt"
            )
            self.freeze_manifest = freeze_manifest
            self.authorization_marker = authorization_marker
            self._seal = token
            self._publication_boundary = publication_boundary

    def issue(
        freeze_manifest: Path,
        authorization_marker: Path,
        *,
        publication_boundary: object | None = None,
    ) -> object:
        receipt = validate_execution_authorization(
            freeze_manifest,
            authorization_marker,
            publication_boundary=publication_boundary,
        )
        return FitAuthorization(
            receipt,
            _lexical_path(
                freeze_manifest,
                "R2R-1 freeze manifest",
                forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
            ),
            _lexical_path(
                authorization_marker,
                "R2R-1 external GO marker",
                forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
            ),
            seal,
            publication_boundary,
        )

    def require(value: object) -> object:
        if type(value) is not FitAuthorization or value._seal is not seal:
            raise PermissionError(
                "R2R-1 fit requires a physically revalidated execution authorization"
            )
        # validate_execution_authorization owns the combined ordering:
        # publication bytes, held manifest/GO/manifest bytes, then the short
        # publication-state closure.  A separate publication walk here would
        # reopen the control-byte window immediately before thermal access.
        current = validate_execution_authorization(
            value.freeze_manifest,
            value.authorization_marker,
            publication_boundary=value._publication_boundary,
        )
        if (
            canonical_json_bytes(value.receipt) != value._receipt_bytes
            or canonical_json_bytes(current) != value._receipt_bytes
        ):
            raise PermissionError("R2R-1 fit authorization changed before SVD")
        return value

    return issue, require


(
    _issue_production_fit_authorization,
    _require_fit_authorization,
) = _build_fit_authorization_boundary()


def _parse_whitelisted_float(value: str) -> float:
    return float(value)


def _load_thermal92_force_labels_streaming_authorized(
    path: Path,
    *,
    authorization: object,
) -> ThermalLabels:
    """Read only the force-gap/A-prime whitelist from the frozen thermal file.

    Unlisted numeric properties and all energy-valued header fields remain
    opaque strings: they are never converted to floats or attached to Atoms.
    """
    # Reject an unissued capability before any path stat/open.  A second full
    # require below follows the lexical checks and remains adjacent to the
    # payload open, so those checks cannot reopen the authorization boundary.
    authorization = _require_fit_authorization(authorization)
    source = _reject_path(
        path,
        "R2R-1 thermal92 force labels",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if source != _lexical_path(
        RECOMMENDED_THERMAL92, "R2R-1 frozen thermal92 path"
    ):
        raise PermissionError("R2R-1 thermal parser path differs from frozen path")
    # Keep the combined publication/control closure adjacent to the first
    # payload open.  The lexical path checks above are non-authoritative and
    # intentionally precede this final authorization boundary.
    authorization = _require_fit_authorization(authorization)
    thermal_binding = authorization.receipt["thermal92_file_binding"]
    (
        source,
        source_fd,
        source_parent_fd,
        source_identity,
        source_parent_chain,
    ) = _open_regular_file_fd(
        source,
        "R2R-1 thermal92 force labels",
        expected_parent_chain=thermal_binding["parent_chain"],
        expected_file_identity=thermal_binding["file_identity"],
    )
    try:
        if source_identity.get("st_size") != THERMAL92_FILE_SIZE_BYTES:
            raise ValueError("R2R-1 thermal92 byte size changed")
        # Close the authorization-return -> payload-read phase boundary while
        # holding this exact thermal fd.  If GO/source authorization changes
        # after the first require, no byte from the label-bearing fd is read.
        _require_fit_authorization(authorization)
        first_sha256 = _sha256_fd_exact_size(
            source_fd,
            THERMAL92_FILE_SIZE_BYTES,
            "R2R-1 thermal92 first hash",
        )
        if first_sha256 != THERMAL92_FILE_SHA256:
            raise ValueError("R2R-1 thermal92 file SHA changed before label parsing")
        _require_fit_authorization(authorization)
        if os.lseek(source_fd, 0, os.SEEK_SET) != 0:
            raise ValueError("R2R-1 thermal92 fd could not rewind for parsing")
    except Exception:
        _finish_regular_file_fd(
            source,
            source_fd,
            source_parent_fd,
            source_identity,
            source_parent_chain,
            "R2R-1 thermal92 force labels",
        )
        raise
    source_digest = hashlib.sha256()
    reference_forces = []
    mode_real = []
    mode_imag = []
    coordinates = []
    foundation = []
    q6 = []
    identities: list[tuple[int, str]] = []
    try:
        bounded_source = _ExpectedSizeRawReader(
            source_fd,
            THERMAL92_FILE_SIZE_BYTES,
            "R2R-1 thermal92 parser",
        )
        handle = io.BufferedReader(bounded_source, buffer_size=64 * 1024)
    except Exception:
        _finish_regular_file_fd(
            source,
            source_fd,
            source_parent_fd,
            source_identity,
            source_parent_chain,
            "R2R-1 thermal92 force labels",
        )
        raise

    def read_stream_line() -> str:
        raw_line = handle.readline(4 * 1024 * 1024 + 1)
        if len(raw_line) > 4 * 1024 * 1024:
            raise ValueError("R2R-1 thermal92 contains an oversized line")
        source_digest.update(raw_line)
        try:
            return raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("R2R-1 thermal92 line is not valid UTF-8") from error

    try:
        global_index = 0
        while True:
            count_line = read_stream_line()
            if not count_line:
                break
            if not count_line.strip():
                continue
            try:
                count = int(count_line.strip())
            except ValueError as error:
                raise ValueError("extxyz atom count changed") from error
            if count != 72:
                raise ValueError("R2R-1 thermal atom count changed")
            header_line = read_stream_line()
            if not header_line:
                raise ValueError("truncated extxyz header")
            fields = _header_fields(header_line)
            schema = _parse_properties(fields.get("Properties", ""))
            group = group_for_global_index(global_index)
            expected_schema = E50_PROPERTY_SCHEMA if group == "E50_seed0" else AUXILIARY_PROPERTY_SCHEMA
            if schema != expected_schema:
                raise ValueError(f"R2R-1 {group} Properties schema changed")
            if fields.get("config_type") != THERMAL_CONFIG_TYPES[group]:
                raise ValueError(f"R2R-1 {group} config_type/order changed")
            offsets, row_width = _property_offsets(schema)
            allowed = E50_NUMERIC_WHITELIST if group == "E50_seed0" else AUXILIARY_NUMERIC_WHITELIST
            converted: dict[str, list[list[float]]] = {
                name: [] for name in allowed if name != "pos"
            }
            symbols: list[str] = []
            positions: list[list[float]] = []
            species_start, species_width = offsets["species"]
            position_start, position_width = offsets["pos"]
            if species_width != 1 or position_width != 3:
                raise ValueError("R2R-1 geometry column width changed")
            for _ in range(count):
                row = read_stream_line()
                if not row:
                    raise ValueError("truncated extxyz atom rows")
                tokens = row.split()
                if len(tokens) != row_width:
                    raise ValueError("extxyz atom row width changed")
                symbols.append(tokens[species_start])
                positions.append(
                    [_parse_whitelisted_float(tokens[position_start + axis]) for axis in range(3)]
                )
                for name in converted:
                    start, width = offsets[name]
                    converted[name].append(
                        [_parse_whitelisted_float(tokens[start + component]) for component in range(width)]
                    )
            lattice = fields.get("Lattice", "").split()
            pbc = fields.get("pbc", "").split()
            if len(lattice) != 9 or len(pbc) != 3 or any(value not in {"T", "F"} for value in pbc):
                raise ValueError("R2R-1 Lattice/pbc schema changed")
            atoms = Atoms(
                symbols=symbols,
                positions=np.asarray(positions, dtype=np.float64),
                cell=np.asarray([_parse_whitelisted_float(value) for value in lattice], dtype=np.float64).reshape(3, 3),
                pbc=tuple(value == "T" for value in pbc),
            )
            atoms.info["config_type"] = fields["config_type"]
            identities.append((global_index, r2r0.structure_semantic_sha256(atoms)))
            reference_forces.append(np.asarray(converted["REF_forces"], dtype=np.float64))
            if group == "E50_seed0":
                for key in HEADER_NUMERIC_WHITELIST:
                    if key not in fields:
                        raise ValueError(f"R2R-1 E50 header lacks {key}")
                mode_real.append(np.asarray(converted["APRIME_mode_real"], dtype=np.float64))
                mode_imag.append(np.asarray(converted["APRIME_mode_imag"], dtype=np.float64))
                foundation.append(np.asarray(converted["FOUNDATION_BASE_forces"], dtype=np.float64))
                q6.append(np.asarray(converted["FROZEN_Q6_forces"], dtype=np.float64))
                coordinates.append(
                    [
                        _parse_whitelisted_float(fields["APRIME_coordinate_real_A"]),
                        _parse_whitelisted_float(fields["APRIME_coordinate_imag_A"]),
                    ]
                )
            global_index += 1
        if bounded_source.total != THERMAL92_FILE_SIZE_BYTES:
            raise ValueError("R2R-1 thermal92 parser byte count changed")
    finally:
        try:
            handle.close()
        finally:
            _finish_regular_file_fd(
                source,
                source_fd,
                source_parent_fd,
                source_identity,
                source_parent_chain,
                "R2R-1 thermal92 force labels",
            )
    source_sha256 = source_digest.hexdigest()
    if source_sha256 != first_sha256 or source_sha256 != THERMAL92_FILE_SHA256:
        raise ValueError("R2R-1 thermal92 changed between hash and parse passes")
    if global_index != 92:
        raise ValueError("R2R-1 thermal92 configuration count changed")
    force_array = np.asarray(reference_forces, dtype="<f8", order="C")
    aprime = AprimeData(
        mode_real=np.asarray(mode_real, dtype="<f8", order="C"),
        mode_imag=np.asarray(mode_imag, dtype="<f8", order="C"),
        coordinates=np.asarray(coordinates, dtype="<f8", order="C"),
        foundation_base_force_eV_A=np.asarray(foundation, dtype="<f8", order="C"),
        frozen_q6_force_eV_A=np.asarray(q6, dtype="<f8", order="C"),
    )
    raw_hashes = {
        "REF_forces_all92": raw_array_sha256(force_array, "<f8"),
        "REF_forces_E50": raw_array_sha256(force_array[0:20], "<f8"),
        "REF_forces_T300": raw_array_sha256(force_array[20:56], "<f8"),
        "REF_forces_T600": raw_array_sha256(force_array[56:92], "<f8"),
        "APRIME_mode_real": raw_array_sha256(aprime.mode_real, "<f8"),
        "APRIME_mode_imag": raw_array_sha256(aprime.mode_imag, "<f8"),
        "APRIME_coordinates": raw_array_sha256(aprime.coordinates, "<f8"),
        "FOUNDATION_BASE_forces": raw_array_sha256(
            aprime.foundation_base_force_eV_A, "<f8"
        ),
        "FROZEN_Q6_forces": raw_array_sha256(aprime.frozen_q6_force_eV_A, "<f8"),
    }
    if raw_hashes != dict(EXPECTED_LABEL_RAW_SHA256):
        raise ValueError("R2R-1 whitelisted label/auxiliary raw SHA changed")
    identity = r2r.semantic_sha256(identities)
    if identity != THERMAL_STRUCTURE_IDENTITY_SHA256:
        raise ValueError("R2R-1 label geometry differs from attempt3 structures")
    if not all(
        np.all(np.isfinite(value))
        for value in (
            force_array,
            aprime.mode_real,
            aprime.mode_imag,
            aprime.coordinates,
            aprime.foundation_base_force_eV_A,
            aprime.frozen_q6_force_eV_A,
        )
    ):
        raise ValueError("R2R-1 whitelisted labels contain non-finite values")
    _require_fit_authorization(authorization)
    return ThermalLabels(
        reference_force_eV_A=force_array,
        aprime=aprime,
        structure_identity_sha256=identity,
        raw_sha256=raw_hashes,
        parser_receipt={
            "format": "graphene_r2r1_streaming_force_label_whitelist_v1",
            "file_sha256": source_sha256,
            "source_file_identity": source_identity,
            "source_parent_chain": [dict(item) for item in source_parent_chain],
            "same_fd_hash_and_parse": True,
            "hash_before_any_label_parse": True,
            "same_fd_two_pass_hash_then_parse": True,
            "streaming_binary_readline": True,
            "full_file_bytes_buffered": False,
            "configuration_count": 92,
            "atom_count": 72,
            "fit_target_columns": ["REF_forces"],
            "E50_metric_auxiliary_columns": sorted(
                E50_NUMERIC_WHITELIST - {"pos", "REF_forces"}
            ),
            "E50_metric_auxiliary_headers": sorted(HEADER_NUMERIC_WHITELIST),
            "header_label_values_retained": False,
            "energy_header_values_converted": False,
            "unlisted_numeric_columns_converted": False,
            "temperature_or_smearing_used_as_feature": False,
            "development_or_held_access": False,
        },
    )


def load_thermal92_force_labels_streaming(
    path: Path,
    *,
    freeze_manifest: Path,
    authorization_marker: Path,
    publication_boundary: object | None = None,
) -> ThermalLabels:
    """Production loader with non-overridable hashes and external authorization."""
    if publication_boundary is None:
        raise PermissionError(
            "standalone production label loading requires a held publication boundary"
        )
    fit_authorization = _issue_production_fit_authorization(
        freeze_manifest,
        authorization_marker,
        publication_boundary=publication_boundary,
    )
    authorization = _strict_json_from_raw(
        canonical_json_bytes(fit_authorization.receipt),
        "R2R-1 pipeline authorization snapshot",
    )
    labels = _load_thermal92_force_labels_streaming_authorized(
        path,
        authorization=fit_authorization,
    )
    _require_bound_file_unchanged(
        path,
        label="R2R-1 thermal92 force labels",
        expected_sha256=THERMAL92_FILE_SHA256,
        expected_identity=labels.parser_receipt["source_file_identity"],
        expected_parent_chain=labels.parser_receipt["source_parent_chain"],
    )
    final_authorization = validate_execution_authorization(
        freeze_manifest,
        authorization_marker,
        publication_boundary=publication_boundary,
    )
    if not _json_type_exact_equal(final_authorization, authorization):
        raise PermissionError("R2R-1 execution authorization changed during label load")
    return ThermalLabels(
        reference_force_eV_A=labels.reference_force_eV_A,
        aprime=labels.aprime,
        structure_identity_sha256=labels.structure_identity_sha256,
        raw_sha256=labels.raw_sha256,
        parser_receipt={
            **labels.parser_receipt,
            "execution_authorization": authorization,
            "production_hash_overrides_allowed": False,
        },
    )


def _force_metrics(error_eV_A: np.ndarray) -> dict[str, Any]:
    flattened = np.asarray(error_eV_A, dtype=np.float64).reshape(-1)
    if flattened.size == 0 or not np.all(np.isfinite(flattened)):
        raise ValueError("force metrics require finite nonempty errors")
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(np.square(flattened)))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(flattened))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(flattened))),
        "n_force_components": int(flattened.size),
    }


def _restoring_slope(coordinates: np.ndarray, projected_force: np.ndarray) -> float:
    denominator = float(np.vdot(coordinates, coordinates).real)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("A-prime coordinates are non-finite or all zero")
    value = -float(np.vdot(coordinates, projected_force).real / denominator)
    if not math.isfinite(value):
        raise ValueError("A-prime restoring slope is non-finite")
    return value


def gate_metrics(
    predicted_force_eV_A: np.ndarray,
    reference_force_eV_A: np.ndarray,
    aprime: AprimeData,
    indices: Sequence[int],
) -> dict[str, Any]:
    predicted = np.asarray(predicted_force_eV_A, dtype=np.float64)
    target = np.asarray(reference_force_eV_A, dtype=np.float64)
    selected = np.asarray(indices, dtype=int)
    if predicted.shape != target.shape or predicted.shape[0] != 92:
        raise ValueError("R2R-1 metric force arrays must cover thermal92")
    if aprime.mode_real.shape != (20, *predicted.shape[1:]) or aprime.mode_imag.shape != (
        20,
        *predicted.shape[1:],
    ):
        raise ValueError("R2R-1 A-prime mode shape must be [20,atom,3]")
    if aprime.coordinates.shape != (20, 2):
        raise ValueError("R2R-1 A-prime coordinates must have shape [20,2]")
    if aprime.foundation_base_force_eV_A.shape != (20, *predicted.shape[1:]) or aprime.frozen_q6_force_eV_A.shape != (
        20,
        *predicted.shape[1:],
    ):
        raise ValueError("R2R-1 E50 base/q6 force shape changed")
    metrics: dict[str, Any] = {"force_by_group": {}}
    for group in THERMAL_GROUP_RANGES:
        members = np.intersect1d(selected, group_indices(group), assume_unique=False)
        if not len(members):
            raise ValueError(f"R2R-1 metric selection lacks group {group}")
        metrics["force_by_group"][group] = _force_metrics(
            predicted[members] - target[members]
        )
    e50_global = np.intersect1d(selected, group_indices("E50_seed0"))
    local = e50_global
    complex_mode = aprime.mode_real[local] + 1.0j * aprime.mode_imag[local]
    base = (
        aprime.foundation_base_force_eV_A[local]
        + aprime.frozen_q6_force_eV_A[local]
    )
    predicted_modes = np.asarray(
        [
            np.vdot(mode.reshape(-1), value.reshape(-1))
            for mode, value in zip(
                complex_mode, base + predicted[e50_global], strict=True
            )
        ]
    )
    target_modes = np.asarray(
        [
            np.vdot(mode.reshape(-1), value.reshape(-1))
            for mode, value in zip(complex_mode, base + target[e50_global], strict=True)
        ]
    )
    error_modes = np.asarray(
        [
            np.vdot(mode.reshape(-1), value.reshape(-1))
            for mode, value in zip(
                complex_mode,
                predicted[e50_global] - target[e50_global],
                strict=True,
            )
        ]
    )
    coordinates = aprime.coordinates[local, 0] + 1.0j * aprime.coordinates[local, 1]
    predicted_slope = _restoring_slope(coordinates, predicted_modes)
    target_slope = _restoring_slope(coordinates, target_modes)
    if abs(target_slope) <= 1.0e-14:
        raise ValueError("A-prime target restoring slope magnitude is <=1e-14")
    metrics["E50_Aprime"] = {
        "RMS_meV_A": float(1000.0 * np.sqrt(np.mean(np.abs(error_modes) ** 2))),
        "predicted_restoring_slope_eV_A2": predicted_slope,
        "target_restoring_slope_eV_A2": target_slope,
        "slope_relative_error": float((predicted_slope - target_slope) / target_slope),
        "count": int(len(e50_global)),
    }
    ratios: dict[str, float] = {}
    for group, force in metrics["force_by_group"].items():
        ratios[f"{group}_force_RMSE"] = (
            force["RMSE_meV_A"] / GATE_THRESHOLDS["force_RMSE_meV_A"]
        )
        ratios[f"{group}_force_max_abs"] = (
            force["max_abs_meV_A"] / GATE_THRESHOLDS["force_max_abs_meV_A"]
        )
    ratios["E50_Aprime_RMS"] = (
        metrics["E50_Aprime"]["RMS_meV_A"]
        / GATE_THRESHOLDS["E50_Aprime_RMS_meV_A"]
    )
    ratios["E50_slope_relative_abs_error"] = (
        abs(metrics["E50_Aprime"]["slope_relative_error"])
        / GATE_THRESHOLDS["E50_slope_relative_abs_error"]
    )
    score = float(max(ratios.values()))
    metrics["normalized_gate_ratios"] = ratios
    metrics["raw_selection_score"] = score
    metrics["selection_score_rounded_12"] = round(score, SCORE_ROUND_DIGITS)
    metrics["passes_fixed_gate"] = all(value <= 1.0 for value in ratios.values())
    return metrics


@dataclass(frozen=True)
class RidgeFit:
    alpha: float
    scale_eV_A: np.ndarray
    normalized_coefficient: np.ndarray
    physical_coefficient: np.ndarray
    design_audit: Mapping[str, Any]
    receipt: Mapping[str, Any]


def _fit_weighted_ridge(
    force_design_eV_A: np.ndarray,
    fixed_force_eV_A: np.ndarray,
    reference_force_eV_A: np.ndarray,
    train_indices: Sequence[int],
    alpha: float,
    *,
    authorization: object,
) -> RidgeFit:
    _require_fit_authorization(authorization)
    if alpha not in ALPHA_GRID:
        raise ValueError("R2R-1 alpha is outside the frozen grid")
    indices = np.asarray(sorted(set(int(index) for index in train_indices)), dtype=int)
    audit = design_subset_audit(force_design_eV_A, indices)
    if not audit["pass"]:
        raise ValueError("R2R-1 train-subset design audit failed")
    design = np.asarray(force_design_eV_A, dtype=np.float64)[indices]
    fixed = np.asarray(fixed_force_eV_A, dtype=np.float64)[indices]
    target = np.asarray(reference_force_eV_A, dtype=np.float64)[indices]
    if design.shape[:3] != fixed.shape or fixed.shape != target.shape:
        raise ValueError("R2R-1 design/fixed/target shapes disagree")
    components = int(np.prod(design.shape[1:3]))
    scale = np.asarray(audit["column_RMS"], dtype=np.float64)
    matrix = (design / scale).reshape(-1, LINEAR_WIDTH)
    response = ((target - fixed) / FORCE_SCALE_EV_A).reshape(-1)
    weights_by_config = []
    counts = {
        group: int(sum(group_for_global_index(index) == group for index in indices))
        for group in THERMAL_GROUP_RANGES
    }
    if any(count <= 0 for count in counts.values()):
        raise ValueError("R2R-1 ridge train subset lacks a thermal group")
    for index in indices:
        group = group_for_global_index(int(index))
        weight = GROUP_MASSES[group] / (counts[group] * components)
        weights_by_config.extend([weight] * components)
    weights = np.asarray(weights_by_config, dtype=np.float64)
    if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=0.0, abs_tol=2.0e-15):
        raise RuntimeError("R2R-1 group/component weights do not sum to one")
    root_weight = np.sqrt(weights)
    weighted_matrix = matrix * root_weight[:, None]
    weighted_response = response * root_weight
    _require_fit_authorization(authorization)
    u, singular, vt = np.linalg.svd(weighted_matrix, full_matrices=False)
    _require_fit_authorization(authorization)
    normalized = vt.T @ (
        (singular / (np.square(singular) + float(alpha)))
        * (u.T @ weighted_response)
    )
    physical = FORCE_SCALE_EV_A * normalized / scale
    if not np.all(np.isfinite(physical)):
        raise ValueError("R2R-1 ridge produced non-finite coefficients")
    receipt = {
        "format": "graphene_r2r1_weighted_svd_ridge_v1",
        "alpha": float(alpha),
        "train_global_indices": indices.tolist(),
        "group_counts": counts,
        "group_masses": GROUP_MASSES,
        "component_count_per_configuration": components,
        "force_scale_eV_A": FORCE_SCALE_EV_A,
        "scaling": "train_force_RMS_without_mean_centering",
        "target_centering": False,
        "fit_intercept": False,
        "solver": "FP64_thin_SVD_filter_s/(s^2+alpha)",
        "weight_sum": float(np.sum(weights)),
        "weighted_singular_values": singular.tolist(),
        "physical_coefficients_sha256": semantic_sha256(physical.tolist()),
        "scale_raw_sha256": raw_array_sha256(scale, "<f8"),
    }
    return RidgeFit(
        alpha=float(alpha),
        scale_eV_A=scale,
        normalized_coefficient=normalized,
        physical_coefficient=physical,
        design_audit=audit,
        receipt=receipt,
    )


def predict_force(
    force_design_eV_A: np.ndarray,
    fixed_force_eV_A: np.ndarray,
    coefficient: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    selected = np.asarray(indices, dtype=int)
    design = np.asarray(force_design_eV_A, dtype=np.float64)[selected]
    fixed = np.asarray(fixed_force_eV_A, dtype=np.float64)[selected]
    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.shape != (LINEAR_WIDTH,):
        raise ValueError("R2R-1 coefficient vector must have width 65")
    return fixed + np.einsum("natk,k->nat", design, coefficient, optimize=False)


def _empty_prediction_like(reference_force: np.ndarray) -> np.ndarray:
    output = np.empty_like(np.asarray(reference_force, dtype=np.float64))
    output.fill(np.nan)
    return output


def _candidate_order_key(record: Mapping[str, Any]) -> tuple[float, float]:
    # Larger alpha wins an exact rounded-score tie.
    return (float(record["metrics"]["selection_score_rounded_12"]), -float(record["alpha"]))


def _nested_conditional_oof(
    force_design_eV_A: np.ndarray,
    fixed_force_eV_A: np.ndarray,
    labels: ThermalLabels,
    *,
    authorization: object,
) -> tuple[dict[str, Any], np.ndarray]:
    _require_fit_authorization(authorization)
    validate_fold_contract()
    all_indices = np.arange(92, dtype=int)
    oof = _empty_prediction_like(labels.reference_force_eV_A)
    outer_records = []
    for outer_id, outer_hold_tuple in enumerate(FOLD_GLOBAL_INDICES):
        outer_hold = np.asarray(outer_hold_tuple, dtype=int)
        outer_train = np.setdiff1d(all_indices, outer_hold)
        candidates = []
        for alpha in ALPHA_GRID:
            inner_prediction = _empty_prediction_like(labels.reference_force_eV_A)
            for inner_hold_id, inner_hold_tuple in enumerate(FOLD_GLOBAL_INDICES):
                if inner_hold_id == outer_id:
                    continue
                inner_hold = np.asarray(inner_hold_tuple, dtype=int)
                inner_train = np.setdiff1d(outer_train, inner_hold)
                fit = _fit_weighted_ridge(
                    force_design_eV_A,
                    fixed_force_eV_A,
                    labels.reference_force_eV_A,
                    inner_train,
                    alpha,
                    authorization=authorization,
                )
                inner_prediction[inner_hold] = predict_force(
                    force_design_eV_A,
                    fixed_force_eV_A,
                    fit.physical_coefficient,
                    inner_hold,
                )
            if not np.all(np.isfinite(inner_prediction[outer_train])):
                raise RuntimeError("nested inner predictions do not cover outer-train69")
            metrics = gate_metrics(
                inner_prediction,
                labels.reference_force_eV_A,
                labels.aprime,
                outer_train,
            )
            candidates.append({"alpha": alpha, "metrics": metrics})
        selected = min(candidates, key=_candidate_order_key)
        outer_fit = _fit_weighted_ridge(
            force_design_eV_A,
            fixed_force_eV_A,
            labels.reference_force_eV_A,
            outer_train,
            float(selected["alpha"]),
            authorization=authorization,
        )
        oof[outer_hold] = predict_force(
            force_design_eV_A,
            fixed_force_eV_A,
            outer_fit.physical_coefficient,
            outer_hold,
        )
        outer_records.append(
            {
                "outer_fold": outer_id,
                "outer_hold_global_indices": outer_hold.tolist(),
                "outer_train_global_indices": outer_train.tolist(),
                "inner_candidates": candidates,
                "selected_alpha": float(selected["alpha"]),
                "selected_inner_metrics": selected["metrics"],
                "outer_fit_receipt": outer_fit.receipt,
                "outer_coefficient_sha256": semantic_sha256(
                    outer_fit.physical_coefficient.tolist()
                ),
            }
        )
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("nested outer predictions do not cover thermal92")
    pooled = gate_metrics(
        oof, labels.reference_force_eV_A, labels.aprime, all_indices
    )
    receipt = {
        "format": "graphene_r2r1_nested_outer4_inner3_oof_v1",
        "interpretation": "conditional_linear_readout_OOF_not_encoder_independence",
        "fold_contract": validate_fold_contract(),
        "alpha_grid": list(ALPHA_GRID),
        "selection_score_round_digits": SCORE_ROUND_DIGITS,
        "selection_tie_break": "larger_alpha",
        "outer_records": outer_records,
        "pooled_OOF_metrics": pooled,
        "OOF_prediction_raw_sha256": raw_array_sha256(oof, "<f8"),
        "pass": bool(pooled["passes_fixed_gate"]),
    }
    return receipt, oof


def _select_final_alpha_cv(
    force_design_eV_A: np.ndarray,
    fixed_force_eV_A: np.ndarray,
    labels: ThermalLabels,
    *,
    authorization: object,
) -> tuple[dict[str, Any], float]:
    _require_fit_authorization(authorization)
    all_indices = np.arange(92, dtype=int)
    candidates = []
    for alpha in ALPHA_GRID:
        prediction = _empty_prediction_like(labels.reference_force_eV_A)
        fold_fit_receipts = []
        for hold_tuple in FOLD_GLOBAL_INDICES:
            hold = np.asarray(hold_tuple, dtype=int)
            train = np.setdiff1d(all_indices, hold)
            fit = _fit_weighted_ridge(
                force_design_eV_A,
                fixed_force_eV_A,
                labels.reference_force_eV_A,
                train,
                alpha,
                authorization=authorization,
            )
            prediction[hold] = predict_force(
                force_design_eV_A,
                fixed_force_eV_A,
                fit.physical_coefficient,
                hold,
            )
            fold_fit_receipts.append(fit.receipt)
        metrics = gate_metrics(
            prediction, labels.reference_force_eV_A, labels.aprime, all_indices
        )
        candidates.append(
            {
                "alpha": alpha,
                "metrics": metrics,
                "fold_fit_receipts": fold_fit_receipts,
                "prediction_raw_sha256": raw_array_sha256(prediction, "<f8"),
            }
        )
    selected = min(candidates, key=_candidate_order_key)
    receipt = {
        "format": "graphene_r2r1_final_alpha_selection_only_cv_v1",
        "is_OOF_gate_evidence": False,
        "candidates": candidates,
        "selected_alpha": float(selected["alpha"]),
        "selection_tie_break": "round_score_12_then_larger_alpha",
    }
    return receipt, float(selected["alpha"])


def _fit_final_readout(
    force_design_eV_A: np.ndarray,
    fixed_force_eV_A: np.ndarray,
    labels: ThermalLabels,
    alpha: float,
    *,
    authorization: object,
) -> tuple[RidgeFit, dict[str, Any], np.ndarray]:
    _require_fit_authorization(authorization)
    indices = np.arange(92, dtype=int)
    fit = _fit_weighted_ridge(
        force_design_eV_A,
        fixed_force_eV_A,
        labels.reference_force_eV_A,
        indices,
        alpha,
        authorization=authorization,
    )
    prediction = predict_force(
        force_design_eV_A,
        fixed_force_eV_A,
        fit.physical_coefficient,
        indices,
    )
    metrics = gate_metrics(
        prediction, labels.reference_force_eV_A, labels.aprime, indices
    )
    return fit, metrics, prediction


def _fit_pipeline_from_loaded_labels(
    force_design_eV_A: np.ndarray,
    fixed_force_eV_A: np.ndarray,
    labels: ThermalLabels,
    *,
    authorization: object,
    prevalidated_split_audits: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _require_fit_authorization(authorization)
    split_audits = all_split_design_audits(force_design_eV_A)
    if split_audits != dict(prevalidated_split_audits):
        raise ValueError("R2R-1 split design changed after label authorization")
    if not split_audits["pass"]:
        raise ValueError("R2R-1 split design preflight failed")
    oof_receipt, oof_prediction = _nested_conditional_oof(
        force_design_eV_A,
        fixed_force_eV_A,
        labels,
        authorization=authorization,
    )
    if not oof_receipt["pass"]:
        return (
            {
                "format": AGGREGATE_FORMAT,
                "status": "R2R1_CONDITIONAL_OOF_FAILED",
                "numerically_inconclusive": False,
                "conditional_OOF_pass": False,
                "final_fit_performed": False,
                "development_or_held_access": False,
                "split_design_audits": split_audits,
                "nested_OOF": oof_receipt,
            },
            {"OOF_predicted_force_eV_A": oof_prediction.astype("<f8")},
        )
    final_cv, alpha = _select_final_alpha_cv(
        force_design_eV_A,
        fixed_force_eV_A,
        labels,
        authorization=authorization,
    )
    final_fit, final_metrics, final_prediction = _fit_final_readout(
        force_design_eV_A,
        fixed_force_eV_A,
        labels,
        alpha,
        authorization=authorization,
    )
    final_pass = bool(final_metrics["passes_fixed_gate"])
    status = (
        "R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING"
        if final_pass
        else "R2R1_FINAL_READOUT_FAILED"
    )
    receipt = {
        "format": AGGREGATE_FORMAT,
        "status": status,
        "numerically_inconclusive": False,
        "conditional_OOF_pass": True,
        "final_fit_performed": True,
        "final_train_gate_pass": final_pass,
        "mechanics_pending": final_pass,
        "development_or_held_access": False,
        "energy_labels_used": False,
        "encoder_updated": False,
        "split_design_audits": split_audits,
        "nested_OOF": oof_receipt,
        "final_alpha_selection": final_cv,
        "final_fit": final_fit.receipt,
        "final_train_metrics": final_metrics,
        "final_coefficients": final_fit.physical_coefficient.tolist(),
        "final_coefficients_sha256": semantic_sha256(
            final_fit.physical_coefficient.tolist()
        ),
        "final_scale_raw_sha256": raw_array_sha256(final_fit.scale_eV_A, "<f8"),
    }
    arrays = {
        "OOF_predicted_force_eV_A": oof_prediction.astype("<f8"),
        "final_predicted_force_eV_A": final_prediction.astype("<f8"),
        "final_coefficients": final_fit.physical_coefficient.astype("<f8"),
        "final_scale_eV_A": final_fit.scale_eV_A.astype("<f8"),
    }
    return receipt, arrays


def authorized_fit_pipeline(
    *,
    attempt3_root: Path,
    thermal92_path: Path,
    freeze_manifest: Path,
    authorization_marker: Path,
    publication_boundary: object | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The unique production fit API: authorize, recover, parse, then solve."""
    # Authorization is deliberately the first operation.  A missing or
    # tampered marker therefore reaches neither the label parser nor an SVD.
    fit_authorization = _issue_production_fit_authorization(
        freeze_manifest,
        authorization_marker,
        publication_boundary=publication_boundary,
    )
    authorization = dict(fit_authorization.receipt)
    frozen_attempt3 = _reject_path(
        attempt3_root,
        "R2R-1 authorized attempt3 root",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if frozen_attempt3 != _lexical_path(
        ATTEMPT3_ROOT, "R2R-1 frozen attempt3 root"
    ):
        raise PermissionError("R2R-1 attempt3 root differs from frozen path")
    recovery, arrays = validate_attempt3_completed_recovery(
        frozen_attempt3,
        expected_root_binding=authorization["attempt3_root_binding"],
    )
    split_audits = all_split_design_audits(
        arrays["thermal_parameter_force_design_eV_A"]
    )
    if not split_audits["pass"]:
        raise ValueError("R2R-1 split design preflight failed before label access")
    if publication_boundary is None:
        raise PermissionError(
            "R2R-1 production fit lacks its held publication boundary"
        )
    # This require performs a fresh, long non-label authorization recheck and
    # then the helper-owned held publication check as its final operation.
    # The next operation is the label loader; a candidate appearing during
    # attempt3/design work therefore cannot reach thermal payload bytes.
    _require_fit_authorization(fit_authorization)
    labels = _load_thermal92_force_labels_streaming_authorized(
        thermal92_path,
        authorization=fit_authorization,
    )
    receipt, output_arrays = _fit_pipeline_from_loaded_labels(
        arrays["thermal_parameter_force_design_eV_A"],
        arrays["thermal_fixed_force_eV_A"],
        labels,
        authorization=fit_authorization,
        prevalidated_split_audits=split_audits,
    )
    _require_bound_file_unchanged(
        thermal92_path,
        label="R2R-1 thermal92 force labels",
        expected_sha256=THERMAL92_FILE_SHA256,
        expected_identity=labels.parser_receipt["source_file_identity"],
        expected_parent_chain=labels.parser_receipt["source_parent_chain"],
    )
    final_authorization = validate_execution_authorization(
        freeze_manifest,
        authorization_marker,
        publication_boundary=publication_boundary,
    )
    if not _json_type_exact_equal(final_authorization, authorization):
        raise PermissionError("R2R-1 execution authorization changed during fit")
    return (
        {
            **receipt,
            "execution_authorization": authorization,
            "attempt3_completed_recovery": recovery,
            "label_parser_receipt": labels.parser_receipt,
            "thermal_label_raw_sha256": dict(labels.raw_sha256),
            "energy_labels_used": False,
            "development_or_held_access": False,
        },
        output_arrays,
    )


def label_blind_preflight(
    attempt3_root: Path = ATTEMPT3_ROOT,
    thermal92_path: Path = ROOT / "data/graphene_r2o_taylor_null_core/train_thermal.xyz",
) -> dict[str, Any]:
    frozen_attempt3 = _lexical_path(
        ATTEMPT3_ROOT,
        "R2R-1 frozen preflight attempt3 root",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    observed_attempt3 = _lexical_path(
        attempt3_root,
        "R2R-1 preflight attempt3 root",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if observed_attempt3 != frozen_attempt3:
        raise PermissionError("R2R-1 preflight attempt3 root differs from frozen path")
    observed_thermal = _lexical_path(
        thermal92_path,
        "R2R-1 label-blind thermal92 path",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    frozen_thermal = _lexical_path(
        RECOMMENDED_THERMAL92,
        "R2R-1 frozen label-blind thermal92 path",
        forbidden_tokens=FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if observed_thermal != frozen_thermal:
        raise PermissionError("R2R-1 preflight thermal92 path differs from frozen path")
    preflight_binding = directory_metadata_binding(
        observed_attempt3, "R2R-1 label-blind attempt3 phase binding"
    )
    recovery, arrays = validate_attempt3_completed_recovery(
        observed_attempt3,
        expected_root_binding={
            "path": preflight_binding["path"],
            "directory_identity": preflight_binding["directory_identity"],
            "chain": preflight_binding["chain"],
        },
    )
    audits = all_split_design_audits(
        arrays["thermal_parameter_force_design_eV_A"]
    )
    return {
        "format": "graphene_r2r1_label_blind_preflight_v1",
        "status": "R2R1_LABEL_BLIND_PREFLIGHT_PASSED" if audits["pass"] else "R2R1_LABEL_BLIND_PREFLIGHT_FAILED",
        "attempt3_receipt_sha256": ATTEMPT3_AGGREGATE_RECEIPT_SHA256,
        "attempt3_arrays_sha256": ATTEMPT3_AGGREGATE_ARRAYS_SHA256,
        "attempt3_recovery": recovery,
        "thermal92_file_sha256": THERMAL92_FILE_SHA256,
        "thermal92_file_access": "not_opened_before_external_GO",
        "fold_contract": validate_fold_contract(),
        "split_design_audits": audits,
        "thermal_force_labels_opened": False,
        "fit_performed": False,
        "energy_labels_used": False,
        "development_or_held_access": False,
        "can_authorize_fit": False,
    }


CONTRACT_PAYLOAD = {
    "format": FORMAT,
    "attempt3": {
        "aggregate_receipt_sha256": ATTEMPT3_AGGREGATE_RECEIPT_SHA256,
        "aggregate_arrays_sha256": ATTEMPT3_AGGREGATE_ARRAYS_SHA256,
        "formal_contract_sha256": R2R0_FORMAL_CONTRACT_SHA256,
        "frozen_R2R_canonical_sha256": R2R_CANONICAL_SHA256,
    },
    "data": {
        "thermal92_file_sha256": THERMAL92_FILE_SHA256,
        "structure_identity_sha256": THERMAL_STRUCTURE_IDENTITY_SHA256,
        "label_raw_sha256": EXPECTED_LABEL_RAW_SHA256,
        "fit_target": "REF_forces_only",
        "energy_labels_used": False,
        "temperature_or_smearing_is_feature": False,
    },
    "readout": {
        "columns": LINEAR_WIDTH,
        "alpha_grid": ALPHA_GRID,
        "force_scale_eV_A": FORCE_SCALE_EV_A,
        "group_masses": GROUP_MASSES,
        "scaler": "train_force_RMS_without_mean_centering",
        "fit_intercept": False,
        "energy_gauge": "Taylor_null_reference_zero_no_fitted_offset",
    },
    "validation": {
        "folds": FOLD_GLOBAL_INDICES,
        "nested": "outer4_inner3",
        "final_alpha": "selection_only_standard4fold_after_OOF_pass",
        "gate_thresholds": GATE_THRESHOLDS,
        "score_round_digits": SCORE_ROUND_DIGITS,
        "tie_break": "larger_alpha",
        "fold_index_sha256": FOLD_GLOBAL_INDEX_SHA256,
        "fold_contract_sha256": FOLD_CONTRACT_SHA256,
    },
    "safety": {
        "encoder_update": False,
        "seed1_seed2_small_support_held_access": False,
        "actual_fit_before_external_authorization": False,
        "old_R2Q_tail_added": False,
    },
}
CONTRACT_SEMANTIC_SHA256 = semantic_sha256(CONTRACT_PAYLOAD)


__all__ = [
    "AGGREGATE_FORMAT",
    "ALPHA_GRID",
    "AprimeData",
    "CONTRACT_PAYLOAD",
    "CONTRACT_SEMANTIC_SHA256",
    "FOLD_GLOBAL_INDICES",
    "FORCE_SCALE_EV_A",
    "FORMAT",
    "GATE_THRESHOLDS",
    "GROUP_MASSES",
    "R2R0_SHARD_GLOBAL_INDICES",
    "RidgeFit",
    "ThermalLabels",
    "all_split_design_audits",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_npz",
    "authorized_fit_pipeline",
    "current_source_sha256",
    "design_subset_audit",
    "freeze_manifest_payload",
    "gate_metrics",
    "label_blind_preflight",
    "load_thermal92_force_labels_streaming",
    "predict_force",
    "raw_array_sha256",
    "semantic_sha256",
    "validate_attempt3_aggregate",
    "validate_attempt3_completed_recovery",
    "validate_attempt3_shard",
    "validate_execution_authorization",
    "validate_fold_contract",
]
