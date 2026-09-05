#!/usr/bin/env python3
"""Independent, non-adjudicating post-attempt2 R2R-0D diagnostics.

This module explains two numerical failures in the frozen R2R-0 attempt2
artifact.  It cannot change that artifact's status, fit a model, authorize a
formal run, inspect held/support data, or create a GO marker.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.io import read


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import graphene_r2r0_formal as formal  # noqa: E402
import graphene_r2r_multipolar_background as r2r  # noqa: E402
import graphene_r2o_taylor_null as r2o  # noqa: E402


FORMAT = "graphene_r2r0d_post_attempt2_numerical_diagnostic_v2"
PREFLIGHT_FORMAT = "graphene_r2r0d_cpu_preflight_v2"
FREEZE_MANIFEST_FORMAT = "graphene_r2r0d_external_freeze_manifest_v2"
ATTEMPT2_SCIENTIFIC_STATUS = "R2R0_FORMAL_REPRESENTATION_PRECHECK_FAILED"

DATA = ROOT / "data/graphene_r2o_taylor_null_core"
ENDPOINT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2Q_four_step_trust_region/formal_4step_seed83_rtx"
)
V5_MANIFEST = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R0_formal_train_only_candidate_attempt2_v5_20260825/freeze_manifest.json"
)
ATTEMPT2_ROOT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/formal_r2r0_three_host_seed83_attempt2"
)
RECOMMENDED_OUTPUT_ROOT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/R2R0D_numerical_diagnostic"
)

# These are roles, not caller-selectable paths.  The external manifest binds all
# four files without placing their hashes back into any of the files themselves.
DIAGNOSTIC_SOURCE_PATHS = {
    "diagnostic_core": Path(__file__).resolve(),
    "diagnostic_CLI": HERE / "run_graphene_r2r0_diagnostic.py",
    "diagnostic_tests": ROOT / "tests/test_graphene_r2r0_diagnostic.py",
    "diagnostic_contract_doc": (
        ROOT / "docs/GRAPHENE_R2R0D_NUMERICAL_DIAGNOSTIC_CONTRACT_2026-08-25.md"
    ),
}
DIAGNOSTIC_SOURCE_ROLES = frozenset(DIAGNOSTIC_SOURCE_PATHS)
ALLOWED_V100_DEVICE_NAMES = ("Tesla V100-SXM2-32GB",)
FORBIDDEN_PATH_TOKENS = ("seed1", "seed2", "support", "held", "holdout")

EXPECTED_V5_MANIFEST_SHA256 = (
    "f2abdb5e997db69e0f406458c2dd3b0331e22d0e5d8828b4986b6f964c8753a6"
)
EXPECTED_GO_MARKER_SHA256 = (
    "8f0fde431f7f180d0541556fbd724d3c2d938ebeda234a4a905365a4749c5f4e"
)
EXPECTED_LAUNCH_RECEIPT_SHA256 = (
    "eeff934de678904eb43ce9d24acf7c72594228f974a30e5a4abc1bd27a1f384a"
)
EXPECTED_ATTEMPT2_ROOT_SHA256 = {
    "DONE": "665cf8bbf78f9cb37587150c7358bb0d5e7fb8fac460ef5e1e4a4830140d1587",
    "EXIT_CODE": (
        "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
    ),
}
EXPECTED_ATTEMPT2_PARTIAL_STATE_SHA256 = (
    "8a71ac66706fc95adddf320fb6fb1122d88d0e5cbe711d46c640ae2e73a89c98"
)
EXPECTED_ATTEMPT2_LOG_MANIFEST = {
    "file_count": 526,
    "semantic_sha256": (
        "a17e49f5b1157b50d815a887fb36a614649c43a8f62fd84fcc25bd9663d5fcce"
    ),
}
EXPECTED_ATTEMPT2_ARTIFACT_SHA256 = {
    "aggregate_receipt": (
        "63c89ff68e4799e4d1ccc3e207835830df9144928e49db3b9b8a3b3fe1830d6e"
    ),
    "aggregate_arrays": (
        "e93646f5d18218e8a7283fa8835a23a37c1248d8a802e6f63245814dc53e0a77"
    ),
    "mechanics_receipt": (
        "6aa8e6cd059a8eb55aa1def117e73147d89d572bd65ac4a33782806a4daa9187"
    ),
    "mechanics_arrays": (
        "f84607c41d7879798701cb77979564779ae680a59909f98723c37337f0ae65c7"
    ),
    "launch_receipt": EXPECTED_LAUNCH_RECEIPT_SHA256,
}
EXPECTED_R2R_CANONICAL_SHA256 = (
    "e589497c7b9f6a9d4ff0cdcc434cf07804fa0fd732d791dba811d4c7e1c5aecc"
)

COORDINATE = (2, 1)
FLAT_COORDINATE = 3 * COORDINATE[0] + COORDINATE[1]
FD_STEPS_A = (8.0e-4, 4.0e-4, 2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5)
ROUND_DECIMALS = (12, 13, 14, 15, 16)
CPU_BUDGET_SECONDS = 120.0
GPU_BUDGET_SECONDS = 120.0
Q_MAX_ABS_TOLERANCE = 5.0e-13

DIAGNOSTIC_CONTRACT: dict[str, Any] = {
    "format": FORMAT,
    "scope": {
        "attempt2_status_is_preserved": ATTEMPT2_SCIENTIFIC_STATUS,
        "adjudicates_attempt2": False,
        "formal_authorization": False,
        "fit_or_training_performed": False,
        "deployment_performed": False,
        "labels_used": False,
        "held_or_support_access": False,
        "GO_marker_creation": False,
    },
    "probe": {
        "structure": "geometry-only thermal92 global index 0",
        "reference": "reference_6x6 index 0",
        "coordinate_zero_based": list(COORDINATE),
        "flat_coordinate_zero_based": FLAT_COORDINATE,
        "public_path": (
            "graphene_r2r0_formal._combined_ef -> "
            "production_combined_energy_force"
        ),
    },
    "finite_difference": {
        "steps_A_in_fixed_order": list(FD_STEPS_A),
        "evaluation_order_each_step": ["minus", "plus"],
        "energy_force": "D_E(h)=-(E(+h)-E(-h))/(2h)",
        "force_jacobian": "D_F(h)=(F_y(+h)-F_y(-h))/(2h)",
        "left_slope": "L(h)=(F0-F(-h))/h",
        "right_slope": "R(h)=(F(+h)-F0)/h",
        "slope_jump": "jump(h)=abs(R(h)-L(h))",
        "empirical_order": "p(h)=log2(err(2h)/err(h)); null if either error is zero",
        "Richardson": "R_F(h)=D_F(h)+(D_F(h)-D_F(2h))/3",
        "attribution_target": (
            "attempt2 force-to-Hessian subgate failure; its energy-to-force "
            "subgate already passed"
        ),
        "energy_multistep_error_and_order_role": "report_only_not_a_gate",
        "FD_TRUNCATION_CONFIRMED": {
            "all_six_force_errors_strictly_decrease": True,
            "all_five_force_orders_inclusive": [1.8, 2.2],
            "all_five_jump_orders_inclusive": [0.8, 1.2],
            "raw_force_error_at_5e-5_A_max_eV_A2": 1.0e-5,
            "last_three_Richardson_errors_max_eV_A2": 1.0e-7,
        },
        "nonsmooth_candidate": (
            "any graph/assignment topology change, or two adjacent p_jump values "
            "outside [0.5,1.5]"
        ),
        "float_floor_candidate": (
            "with unchanged graph/assignment, smallest-h force error is at least "
            "0.8 times the preceding error"
        ),
    },
    "q_comparison": {
        "references": ["reference_6x6", "reference_8x8"],
        "devices": ["cpu", "cuda"],
        "raw_arrays": [
            "receiver",
            "sender",
            "image_integer",
            "local_source_order_offsets",
            "local_source_order_sender",
            "reference_distance_A",
            "weight",
            "normalization",
            "center_q",
            "neighbor_q",
        ],
        "round_decimals": list(ROUND_DECIMALS),
        "round_payload": (
            "concat(center_q in node order, neighbor_q in production edge order); "
            "np.round then contiguous little-endian FP64 bytes"
        ),
        "Q_FLOAT_SERIALIZATION_ONLY": {
            "discrete_topology_and_source_order_exact": True,
            "normalized_q_CPU_CUDA_max_abs": Q_MAX_ABS_TOLERANCE,
            "round12_and_round13_elementwise_and_hash_equal": True,
            "physical_zero_jet_value_J_full_H_exact_zero_and_finite": True,
            "nonzero_local_to_production_value_gradient_parity_pass": True,
            "raw_q_orbit_hash_is_a_gate": False,
        },
    },
    "classification": {
        "precedence": [
            "ATTRIBUTED_BOTH",
            "DIAGNOSTIC_TRUNCATION",
            "NONSMOOTH_CANDIDATE",
            "FLOAT_FLOOR_CANDIDATE",
            "NUMERICAL_IDENTITY",
            "INCONCLUSIVE",
        ],
        "both": "ATTRIBUTED_BOTH",
        "finite_difference_only": "DIAGNOSTIC_TRUNCATION",
        "q_only": "NUMERICAL_IDENTITY",
        "nonsmooth": "NONSMOOTH_CANDIDATE",
        "float_floor": "FLOAT_FLOOR_CANDIDATE",
        "otherwise": "INCONCLUSIVE",
        "budget": "DIAGNOSTIC_BUDGET_EXCEEDED",
        "none_of_these_is_a_formal_PASS": True,
    },
    "output_artifact": {
        "strict_descendant_of_RECOMMENDED_OUTPUT_ROOT_only": True,
        "raw_and_resolved_path_checked": True,
        "symlink_and_dotdot_rejected": True,
        "NPZ_receipt_binding": (
            "every array has exactly one receipt reference with exact key, shape, "
            "dtype and raw SHA-256; no missing, extra or duplicate reference"
        ),
        "terminal_success": "DONE=status+receipt_sha256; EXIT_CODE=0; XOR",
        "terminal_early_stop": "FAILED=status+receipt_sha256; EXIT_CODE=2; XOR",
        "completed_recovery": (
            "full terminal, receipt, NPZ hash/schema/reference, authorization, "
            "source, v5, input, and attempt2 revalidation without recomputation"
        ),
        "partial_or_FAILED_recovery": False,
        "formal_marker_created": False,
    },
    "stop_rules": [
        "source, input, v5 manifest, GO snapshot, launch, or attempt2 artifact hash differs",
        "attempt2 scientific status is not the frozen FAIL status",
        "a force/energy label enters an Atoms object or any diagnostic calculation",
        "any evaluated scalar or array is nonfinite",
        "CPU/CUDA receiver, sender, image, or local source order differs",
        "CPU or GPU wall-time budget is exceeded",
        "no automatic retry, threshold relaxation, point change, or step change",
    ],
    "budget": {
        "CPU_wall_seconds_max": CPU_BUDGET_SECONDS,
        "GPU_wall_seconds_max": GPU_BUDGET_SECONDS,
        "CUDA_timing_and_runtime_identity": (
            "normalize the requested CUDA index once; synchronize and query that "
            "exact device, never an implicit current device"
        ),
    },
    "execution_authorization": {
        "external_freeze_manifest": FREEZE_MANIFEST_FORMAT,
        "manifest_binds_exact_source_roles": sorted(DIAGNOSTIC_SOURCE_ROLES),
        "manifest_binds": [
            "diagnostic canonical contract",
            "v5 manifest/formal/primitive/input snapshot",
            "attempt2 terminal/artifact/log/control snapshot",
        ],
        "external_marker_payload": "exact freeze-manifest SHA-256 plus newline",
        "prepare_function_creates_authorization_marker": False,
        "validated_before_output_creation_and_after_computation": True,
    },
    "runtime": {
        "cpu_preflight_device": "cpu",
        "full_diagnostic_allowed_exact_CUDA_device_names": list(
            ALLOWED_V100_DEVICE_NAMES
        ),
        "full_diagnostic_requires_float32_origin_default_dtype": True,
    },
    "frozen_provenance": {
        "v5_manifest_sha256": EXPECTED_V5_MANIFEST_SHA256,
        "attempt2_root_marker_sha256": EXPECTED_ATTEMPT2_ROOT_SHA256,
        "attempt2_partial_state_sha256": EXPECTED_ATTEMPT2_PARTIAL_STATE_SHA256,
        "attempt2_log_manifest": EXPECTED_ATTEMPT2_LOG_MANIFEST,
        "attempt2_artifact_sha256": EXPECTED_ATTEMPT2_ARTIFACT_SHA256,
        "R2R_canonical_sha256": EXPECTED_R2R_CANONICAL_SHA256,
    },
}

# Tests recompute this digest and fail closed on any semantic edit.
DIAGNOSTIC_CONTRACT_SHA256 = (
    "bfe73de26b346527b3c2ac7d17c50b8abf9535417fbcff48520ba03c4946204f"
)


class DiagnosticStop(RuntimeError):
    """Fail-closed stop with a diagnostic-only classification."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class Budget:
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0

    def add(self, kind: str, elapsed: float) -> None:
        if kind == "cpu":
            self.cpu_seconds += float(elapsed)
        elif kind == "gpu":
            self.gpu_seconds += float(elapsed)
        else:
            raise ValueError(f"unknown budget kind: {kind}")
        self.check()

    def check(self) -> None:
        if self.cpu_seconds > CPU_BUDGET_SECONDS:
            raise DiagnosticStop(
                "DIAGNOSTIC_BUDGET_EXCEEDED",
                f"CPU wall budget exceeded: {self.cpu_seconds:.6f} s",
            )
        if self.gpu_seconds > GPU_BUDGET_SECONDS:
            raise DiagnosticStop(
                "DIAGNOSTIC_BUDGET_EXCEEDED",
                f"GPU wall budget exceeded: {self.gpu_seconds:.6f} s",
            )

    def reconcile_total_wall(self, total_elapsed: float) -> None:
        """Account for all non-GPU wall time omitted by fine-grained timers."""
        inferred_cpu = max(0.0, float(total_elapsed) - self.gpu_seconds)
        self.cpu_seconds = max(self.cpu_seconds, inferred_cpu)
        self.check()

    def receipt(self) -> dict[str, Any]:
        return {
            "CPU_wall_seconds": self.cpu_seconds,
            "CPU_wall_seconds_max": CPU_BUDGET_SECONDS,
            "GPU_wall_seconds": self.gpu_seconds,
            "GPU_wall_seconds_max": GPU_BUDGET_SECONDS,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise DiagnosticStop("INCONCLUSIVE", "nonfinite value entered receipt")
        return number
    return value


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(
        _json_safe(dict(payload)), sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, content)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        np.savez(handle, **{key: np.asarray(value) for key, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def default_inputs() -> formal.FormalInputs:
    """Return the only allowed model/geometry inputs; no caller overrides exist."""
    return formal.FormalInputs(
        endpoint_checkpoint=ENDPOINT / "endpoint.pt",
        endpoint_receipt=ENDPOINT / "endpoint_receipt.json",
        endpoint_marker=ENDPOINT / "ENDPOINT_FROZEN",
        reference_6x6=DATA / "reference_6x6.xyz",
        reference_8x8=DATA / "reference_8x8.xyz",
        thermal92=DATA / "train_thermal.xyz",
        harmonic_zero32=DATA / "train_harmonic_lambda1_small_zero.xyz",
    )


def _path_has_symlink(path: Path) -> bool:
    absolute = path if path.is_absolute() else Path.cwd() / path
    probe = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        probe = probe / part
        if probe.is_symlink():
            return True
    return False


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_control_path(path: Path, label: str, *, must_exist: bool) -> Path:
    """Resolve a manifest/marker path without permitting ambiguous traversal."""
    raw = Path(path)
    if ".." in PurePath(raw).parts:
        raise ValueError(f"R2R-0D {label} path traversal is forbidden")
    if any(
        token in part.casefold()
        for part in PurePath(raw).parts
        for token in FORBIDDEN_PATH_TOKENS
    ):
        raise ValueError(f"R2R-0D {label} contains a forbidden data token")
    if _path_has_symlink(raw):
        raise ValueError(f"R2R-0D {label} path contains a symlink")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    resolved = absolute.resolve(strict=must_exist)
    if must_exist and (not resolved.is_file() or resolved.is_symlink()):
        raise ValueError(f"R2R-0D {label} must be a regular file")
    return resolved


def _validated_output_path(path: Path) -> Path:
    """Return the only permitted output location, without creating it."""
    raw = Path(path)
    if ".." in PurePath(raw).parts:
        raise ValueError("R2R-0D output path traversal is forbidden")
    if _path_has_symlink(raw):
        raise ValueError("R2R-0D output path contains a symlink")
    raw_absolute = raw if raw.is_absolute() else Path.cwd() / raw
    raw_absolute = Path(os.path.abspath(raw_absolute))
    recommended_raw = Path(os.path.abspath(RECOMMENDED_OUTPUT_ROOT))
    if raw_absolute == recommended_raw or not _is_relative_to(
        raw_absolute, recommended_raw
    ):
        raise ValueError(
            "R2R-0D output must be a strict raw-path descendant of "
            "RECOMMENDED_OUTPUT_ROOT"
        )
    output_relative_parts = raw_absolute.relative_to(recommended_raw).parts
    if any(
        token in part.casefold()
        for part in output_relative_parts
        for token in FORBIDDEN_PATH_TOKENS
    ):
        raise ValueError("R2R-0D output contains a forbidden data token")
    resolved = raw_absolute.resolve(strict=False)
    recommended_resolved = RECOMMENDED_OUTPUT_ROOT.resolve(strict=False)
    if resolved == recommended_resolved or not _is_relative_to(
        resolved, recommended_resolved
    ):
        raise ValueError(
            "R2R-0D output must be a strict resolved descendant of "
            "RECOMMENDED_OUTPUT_ROOT"
        )
    if _path_has_symlink(resolved):
        raise ValueError("R2R-0D resolved output path contains a symlink")
    return resolved


def _fresh_output(path: Path) -> Path:
    """Create a fresh output and its sole in-progress terminal marker."""
    root = _validated_output_path(path)
    root.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(root / "RUNNING", b"R2R-0D diagnostic in progress\n")
    return root


def _tree_manifest(root: Path, *, require_terminal: bool) -> dict[str, Any]:
    if not root.is_dir() or _path_has_symlink(root):
        raise DiagnosticStop("INCONCLUSIVE", f"artifact root invalid: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DiagnosticStop("INCONCLUSIVE", "attempt2 artifact contains symlink")
        if path.is_file():
            files[str(path.relative_to(root))] = sha256(path)
    if require_terminal and not {"DONE", "EXIT_CODE", "receipt.json"}.issubset(files):
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 artifact terminal files missing")
    payload: dict[str, Any] = {"files": files, "file_count": len(files)}
    payload["semantic_sha256"] = _semantic_sha256(payload)
    return payload


def _require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise DiagnosticStop(
            "INCONCLUSIVE", f"{label} SHA-256 changed: {observed} != {expected}"
        )
    return observed


def _validate_attempt2_terminal(root: Path) -> dict[str, str]:
    terminal = {
        name for name in ("RUNNING", "FAILED", "DONE") if (root / name).exists()
    }
    expected_done = (
        ATTEMPT2_SCIENTIFIC_STATUS
        + "\n"
        + EXPECTED_LAUNCH_RECEIPT_SHA256
        + "\n"
    ).encode("ascii")
    if terminal != {"DONE"}:
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 execution terminal XOR changed")
    if (root / "EXIT_CODE").read_bytes() != b"0\n":
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 execution exit code changed")
    if (root / "DONE").read_bytes() != expected_done:
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 DONE binding changed")
    return {
        "DONE": sha256(root / "DONE"),
        "EXIT_CODE": sha256(root / "EXIT_CODE"),
    }


def validate_frozen_provenance() -> dict[str, Any]:
    """Validate v5, attempt2, all collected artifact manifests, and inputs.

    Artifact arrays are byte-hashed only.  The only NPZ values later loaded are
    the two frozen model-output targets from mechanics_arrays.npz.
    """
    manifest_hash = _require_hash(
        V5_MANIFEST, EXPECTED_V5_MANIFEST_SHA256, "v5 freeze manifest"
    )
    manifest = json.loads(V5_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("format") != formal.FREEZE_MANIFEST_FORMAT:
        raise DiagnosticStop("INCONCLUSIVE", "v5 manifest format changed")
    if manifest.get("formal_contract_sha256") != formal.FORMAL_CONTRACT_SHA256:
        raise DiagnosticStop("INCONCLUSIVE", "v5 formal contract binding changed")
    if formal.FROZEN_R2R_CANONICAL_SHA256 != EXPECTED_R2R_CANONICAL_SHA256:
        raise DiagnosticStop("INCONCLUSIVE", "R2R canonical hash changed")

    current_formal = {
        name: sha256(path) for name, path in formal.FORMAL_SOURCE_PATHS.items()
    }
    current_primitive = {
        name: sha256(path) for name, path in formal.FROZEN_SOURCE_PATHS.items()
    }
    if current_formal != manifest.get("formal_source_sha256"):
        raise DiagnosticStop("INCONCLUSIVE", "v5 formal source hashes changed")
    if current_primitive != manifest.get("frozen_primitive_sha256"):
        raise DiagnosticStop("INCONCLUSIVE", "v5 primitive source hashes changed")

    inputs = default_inputs()
    input_hashes = {name: sha256(path) for name, path in inputs.mapping().items()}
    if input_hashes != manifest.get("input_sha256"):
        raise DiagnosticStop("INCONCLUSIVE", "v5 input hashes changed")

    control_manifest = ATTEMPT2_ROOT / "control/freeze_manifest.json"
    control_go = ATTEMPT2_ROOT / "control/R2R0_FORMAL_GO"
    _require_hash(control_manifest, EXPECTED_V5_MANIFEST_SHA256, "attempt2 manifest")
    _require_hash(control_go, EXPECTED_GO_MARKER_SHA256, "attempt2 GO snapshot")
    if control_manifest.read_bytes() != V5_MANIFEST.read_bytes():
        raise DiagnosticStop("INCONCLUSIVE", "attempt2/v5 manifest bytes differ")
    if control_go.read_bytes() != (manifest_hash + "\n").encode("ascii"):
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 GO snapshot binding changed")

    root_hashes = _validate_attempt2_terminal(ATTEMPT2_ROOT)
    if root_hashes != EXPECTED_ATTEMPT2_ROOT_SHA256:
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 root marker hashes changed")
    expected_root_entries = {
        "DONE",
        "EXIT_CODE",
        "launch_receipt.json",
        "partial_state.json",
        "aggregate",
        "artifacts",
        "control",
        "logs",
    }
    observed_root_entries = {path.name for path in ATTEMPT2_ROOT.iterdir()}
    if observed_root_entries != expected_root_entries or any(
        path.is_symlink() for path in ATTEMPT2_ROOT.iterdir()
    ):
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 root inventory changed")
    artifact_roles = {path.name for path in (ATTEMPT2_ROOT / "artifacts").iterdir()}
    if artifact_roles != {"shard0", "shard1", "shard2", "mechanics"}:
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 artifact role inventory changed")

    paths = {
        "aggregate_receipt": ATTEMPT2_ROOT / "aggregate/receipt.json",
        "aggregate_arrays": ATTEMPT2_ROOT / "aggregate/arrays.npz",
        "mechanics_receipt": ATTEMPT2_ROOT / "artifacts/mechanics/receipt.json",
        "mechanics_arrays": ATTEMPT2_ROOT / "artifacts/mechanics/mechanics_arrays.npz",
        "launch_receipt": ATTEMPT2_ROOT / "launch_receipt.json",
    }
    observed_artifact_hashes = {
        name: _require_hash(path, EXPECTED_ATTEMPT2_ARTIFACT_SHA256[name], name)
        for name, path in paths.items()
    }
    launch = json.loads(paths["launch_receipt"].read_text(encoding="utf-8"))
    aggregate = json.loads(paths["aggregate_receipt"].read_text(encoding="utf-8"))
    mechanics = json.loads(paths["mechanics_receipt"].read_text(encoding="utf-8"))
    if (
        launch.get("format") != "graphene_r2r0_formal_launch_receipt_v2"
        or launch.get("attempt") != 2
        or launch.get("run_id") != EXPECTED_V5_MANIFEST_SHA256[:12]
        or launch.get("status") != ATTEMPT2_SCIENTIFIC_STATUS
        or aggregate.get("status") != ATTEMPT2_SCIENTIFIC_STATUS
        or aggregate.get("representation_precheck_pass") is not False
        or aggregate.get("numerically_inconclusive") is not False
    ):
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 frozen scientific FAIL changed")
    if launch.get("fit_or_training") is not False or launch.get(
        "held_or_support_access"
    ) is not False:
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 scope receipt changed")
    if (
        mechanics.get("format") != formal.MECHANICS_FORMAT
        or mechanics.get("status") != formal.STATUS_MECHANICS
        or mechanics.get("mechanics_pass") is not False
        or mechanics.get("arrays_sha256")
        != EXPECTED_ATTEMPT2_ARTIFACT_SHA256["mechanics_arrays"]
    ):
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 mechanics receipt changed")
    if aggregate.get("arrays_sha256") != EXPECTED_ATTEMPT2_ARTIFACT_SHA256[
        "aggregate_arrays"
    ]:
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 aggregate arrays binding changed")

    expected_zerojet = r2r.CANONICAL_CONTRACT["linear_design"][
        "rank0_zero_jet_optimization"
    ]["local_direct_numeric_receipts"]
    observed_zerojet = mechanics["mechanics"]["zero_jet"]
    q_failure_identity = {}
    for name in ("reference_6x6", "reference_8x8"):
        expected_orbit = expected_zerojet[name]["q_signature_orbits_sha256"]
        attempt2_orbit = observed_zerojet[name]["q_signature_orbits_sha256"]
        expected_source = expected_zerojet[name][
            "production_local_source_order_sha256"
        ]
        attempt2_source = observed_zerojet[name][
            "production_local_source_order_sha256"
        ]
        if expected_orbit == attempt2_orbit or expected_source != attempt2_source:
            raise DiagnosticStop(
                "INCONCLUSIVE", "attempt2 q failure identity no longer matches"
            )
        q_failure_identity[name] = {
            "canonical_expected_raw_q_orbit_sha256": expected_orbit,
            "attempt2_mechanics_raw_q_orbit_sha256": attempt2_orbit,
            "raw_q_orbit_hash_equal": False,
            "canonical_expected_source_order_sha256": expected_source,
            "attempt2_mechanics_source_order_sha256": attempt2_source,
            "source_order_hash_equal": True,
            "role": "hash_bound_failure_identity_diagnostic_only_not_a_gate",
        }
    attempt2_fd = mechanics["mechanics"]["finite_difference"]
    fd_limits = r2r.CANONICAL_CONTRACT["fixed_gates"]["finite_difference"]
    energy_fd_passed = bool(
        attempt2_fd["force_abs_difference_eV_A"] <= fd_limits["force_abs_eV_A"]
    )
    force_hessian_failed = bool(
        attempt2_fd["force_Hessian_abs_difference_eV_A2"]
        > fd_limits["force_Hessian_abs_eV_A2"]
    )
    if (
        attempt2_fd["coordinate"] != list(COORDINATE)
        or attempt2_fd["step_A"] != 1.0e-4
        or not energy_fd_passed
        or not force_hessian_failed
    ):
        raise DiagnosticStop(
            "INCONCLUSIVE", "attempt2 finite-difference failure identity changed"
        )
    fd_failure_identity = {
        "coordinate": list(COORDINATE),
        "attempt2_step_A": attempt2_fd["step_A"],
        "energy_FD_force_abs_error_eV_A": attempt2_fd[
            "force_abs_difference_eV_A"
        ],
        "energy_FD_force_limit_eV_A": fd_limits["force_abs_eV_A"],
        "energy_FD_force_subgate_passed": energy_fd_passed,
        "force_FD_Hessian_abs_error_eV_A2": attempt2_fd[
            "force_Hessian_abs_difference_eV_A2"
        ],
        "force_FD_Hessian_limit_eV_A2": fd_limits[
            "force_Hessian_abs_eV_A2"
        ],
        "force_FD_Hessian_subgate_failed": force_hessian_failed,
        "R2R0D_attribution_scope": (
            "force-to-Hessian truncation failure only; energy-to-force was already "
            "inside the frozen attempt2 threshold and its multistep values are report-only"
        ),
    }

    observed_manifests = {
        role: _tree_manifest(ATTEMPT2_ROOT / "artifacts" / role, require_terminal=True)
        for role in ("shard0", "shard1", "shard2", "mechanics")
    }
    observed_manifests["aggregate"] = _tree_manifest(
        ATTEMPT2_ROOT / "aggregate", require_terminal=True
    )
    if observed_manifests != launch.get("artifact_manifests"):
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 artifact manifests changed")
    if _tree_manifest(
        ATTEMPT2_ROOT / "control", require_terminal=False
    ) != launch.get("control_manifest"):
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 control manifest changed")
    log_manifest = _tree_manifest(ATTEMPT2_ROOT / "logs", require_terminal=False)
    if (
        log_manifest != launch.get("log_manifest")
        or log_manifest["file_count"] != EXPECTED_ATTEMPT2_LOG_MANIFEST["file_count"]
        or log_manifest["semantic_sha256"]
        != EXPECTED_ATTEMPT2_LOG_MANIFEST["semantic_sha256"]
    ):
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 log manifest changed")
    partial_state_path = ATTEMPT2_ROOT / "partial_state.json"
    _require_hash(
        partial_state_path,
        EXPECTED_ATTEMPT2_PARTIAL_STATE_SHA256,
        "attempt2 partial state",
    )
    partial_state = json.loads(partial_state_path.read_text(encoding="utf-8"))
    completion_keys = (
        "staging_complete",
        "shards_complete",
        "mechanics_complete",
        "collection_complete",
    )
    if (
        set(partial_state)
        != {
            *completion_keys,
            "attempt",
            "run_id",
            "execution_authorization",
        }
        or {key: partial_state[key] for key in completion_keys}
        != launch.get("partial_state")
        or not all(partial_state[key] is True for key in completion_keys)
        or partial_state["attempt"] != launch["attempt"]
        or partial_state["run_id"] != launch["run_id"]
        or partial_state["execution_authorization"]
        != launch["execution_authorization"]
    ):
        raise DiagnosticStop("INCONCLUSIVE", "attempt2 partial state changed")
    if launch.get("aggregate_receipt") != {
        "sha256": EXPECTED_ATTEMPT2_ARTIFACT_SHA256["aggregate_receipt"],
        "status": ATTEMPT2_SCIENTIFIC_STATUS,
        "arrays_sha256": EXPECTED_ATTEMPT2_ARTIFACT_SHA256["aggregate_arrays"],
        "representation_precheck_pass": False,
        "numerically_inconclusive": False,
    }:
        raise DiagnosticStop("INCONCLUSIVE", "launch/aggregate binding changed")

    return {
        "v5_manifest_sha256": manifest_hash,
        "v5_formal_contract_sha256": formal.FORMAL_CONTRACT_SHA256,
        "v5_formal_source_sha256": current_formal,
        "v5_primitive_source_sha256": current_primitive,
        "v5_R2R_canonical_sha256": formal.FROZEN_R2R_CANONICAL_SHA256,
        "v5_input_sha256": input_hashes,
        "attempt2_artifact_sha256": observed_artifact_hashes,
        "attempt2_root_marker_sha256": root_hashes,
        "attempt2_q_failure_identity": q_failure_identity,
        "attempt2_finite_difference_failure_identity": fd_failure_identity,
        "attempt2_artifact_manifest_semantic_sha256": {
            name: value["semantic_sha256"]
            for name, value in observed_manifests.items()
        },
        "attempt2_control_manifest_semantic_sha256": launch["control_manifest"][
            "semantic_sha256"
        ],
        "attempt2_log_manifest": {
            "file_count": log_manifest["file_count"],
            "semantic_sha256": log_manifest["semantic_sha256"],
        },
        "attempt2_partial_state_sha256": EXPECTED_ATTEMPT2_PARTIAL_STATE_SHA256,
        "attempt2_scientific_status_observed": ATTEMPT2_SCIENTIFIC_STATUS,
        "attempt2_status_preserved": True,
        "artifact_arrays_content_interpreted_as_labels": False,
        "original_extxyz_force_or_energy_columns_converted": False,
    }


def _diagnostic_source_sha256() -> dict[str, str]:
    if set(DIAGNOSTIC_SOURCE_PATHS) != DIAGNOSTIC_SOURCE_ROLES:
        raise DiagnosticStop("INCONCLUSIVE", "diagnostic source role inventory changed")
    observed: dict[str, str] = {}
    for role, path in DIAGNOSTIC_SOURCE_PATHS.items():
        if not path.is_file() or path.is_symlink():
            raise DiagnosticStop(
                "INCONCLUSIVE", f"diagnostic source is missing or symlinked: {role}"
            )
        observed[role] = sha256(path)
    return observed


def _attempt2_manifest_binding(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scientific_status": provenance["attempt2_scientific_status_observed"],
        "status_preserved": provenance["attempt2_status_preserved"],
        "root_marker_sha256": provenance["attempt2_root_marker_sha256"],
        "partial_state_sha256": provenance["attempt2_partial_state_sha256"],
        "artifact_sha256": provenance["attempt2_artifact_sha256"],
        "artifact_manifest_semantic_sha256": provenance[
            "attempt2_artifact_manifest_semantic_sha256"
        ],
        "control_manifest_semantic_sha256": provenance[
            "attempt2_control_manifest_semantic_sha256"
        ],
        "log_manifest": provenance["attempt2_log_manifest"],
    }


def _freeze_manifest_payload(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if _semantic_sha256(DIAGNOSTIC_CONTRACT) != DIAGNOSTIC_CONTRACT_SHA256:
        raise DiagnosticStop("INCONCLUSIVE", "diagnostic canonical contract changed")
    return {
        "format": FREEZE_MANIFEST_FORMAT,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_source_sha256": _diagnostic_source_sha256(),
        "R2R_canonical_sha256": provenance["v5_R2R_canonical_sha256"],
        "v5_binding": {
            "manifest_sha256": provenance["v5_manifest_sha256"],
            "formal_contract_sha256": provenance["v5_formal_contract_sha256"],
            "formal_source_sha256": provenance["v5_formal_source_sha256"],
            "primitive_source_sha256": provenance[
                "v5_primitive_source_sha256"
            ],
            "input_sha256": provenance["v5_input_sha256"],
        },
        "attempt2_binding": _attempt2_manifest_binding(provenance),
        "authorization_marker_created": False,
        "can_authorize_fit_or_training": False,
        "can_change_attempt2_status": False,
    }


def prepare_freeze_manifest(output: Path) -> dict[str, Any]:
    """Write a local candidate manifest and deliberately create no marker."""
    destination = _reject_control_path(
        output, "freeze manifest output", must_exist=False
    )
    if destination.exists():
        raise FileExistsError("R2R-0D freeze manifest output must be fresh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = validate_frozen_provenance()
    payload = _freeze_manifest_payload(provenance)
    _atomic_json(destination, payload)
    return payload


def validate_execution_authorization(
    freeze_manifest: Path, authorization_marker: Path
) -> dict[str, Any]:
    """Validate the external, non-GO authorization against all live inputs."""
    manifest_path = _reject_control_path(
        freeze_manifest, "freeze manifest", must_exist=True
    )
    marker_path = _reject_control_path(
        authorization_marker, "authorization marker", must_exist=True
    )
    if "go" in marker_path.name.casefold():
        raise ValueError("R2R-0D authorization marker must not be named as GO")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise ValueError("R2R-0D freeze manifest is not canonical JSON") from exception
    if not isinstance(payload, dict):
        raise ValueError("R2R-0D freeze manifest must be an object")
    provenance = validate_frozen_provenance()
    expected = _freeze_manifest_payload(provenance)
    if set(payload) != set(expected):
        raise ValueError("R2R-0D freeze manifest field inventory changed")
    source_roles = payload.get("diagnostic_source_sha256")
    if not isinstance(source_roles, dict) or set(source_roles) != DIAGNOSTIC_SOURCE_ROLES:
        raise ValueError("R2R-0D freeze manifest source role inventory changed")
    if payload != expected:
        raise ValueError("R2R-0D freeze manifest binding changed")
    manifest_sha = sha256(manifest_path)
    expected_marker = (manifest_sha + "\n").encode("ascii")
    if marker_path.read_bytes() != expected_marker:
        raise ValueError(
            "R2R-0D authorization marker does not bind the exact freeze manifest"
        )
    return {
        "freeze_manifest_sha256": manifest_sha,
        "authorization_marker_sha256": sha256(marker_path),
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_source_sha256": expected["diagnostic_source_sha256"],
        "R2R_canonical_sha256": expected["R2R_canonical_sha256"],
        "v5_binding": expected["v5_binding"],
        "attempt2_binding": expected["attempt2_binding"],
    }


def _validate_label_free_atoms(structure: Atoms, label: str) -> dict[str, Any]:
    forbidden = {"energy", "energies", "force", "forces", "stress", "virial"}
    array_keys = set(structure.arrays)
    info_keys = set(structure.info)
    calculator_attached = structure.calc is not None
    label_like_keys = sorted(
        key
        for key in array_keys | info_keys
        if any(token in str(key).casefold() for token in forbidden)
    )
    contaminated = (
        bool(label_like_keys)
        or calculator_attached
    )
    if contaminated:
        raise DiagnosticStop("INCONCLUSIVE", f"label entered Atoms object: {label}")
    return {
        "label": label,
        "array_keys": sorted(array_keys),
        "info_keys": sorted(info_keys),
        "label_like_keys": label_like_keys,
        "calculator_attached": calculator_attached,
        "force_or_energy_labels_present": False,
    }


def load_label_free_geometry() -> tuple[list[Atoms], Atoms, Atoms, dict[str, Any]]:
    inputs = default_inputs()
    thermal = formal.read_geometry_only_extxyz(inputs.thermal92)
    if len(thermal) != 92:
        raise DiagnosticStop("INCONCLUSIVE", "thermal92 geometry count changed")
    probe = thermal[0]
    reference6 = read(inputs.reference_6x6, index=0)
    reference8 = read(inputs.reference_8x8, index=0)
    audit = {
        "thermal92_global_index0": _validate_label_free_atoms(probe, "thermal92[0]"),
        "reference_6x6": _validate_label_free_atoms(reference6, "reference_6x6"),
        "reference_8x8": _validate_label_free_atoms(reference8, "reference_8x8"),
        "thermal_count": len(thermal),
        "thermal_loader": "formal.read_geometry_only_extxyz",
        "header_label_values_retained": False,
        "numeric_label_columns_converted": False,
    }
    return thermal, reference6, reference8, audit


def _little_endian_array(value: torch.Tensor, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.dtype(dtype))


def _array_receipt(name: str, value: np.ndarray, npz_key: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if not np.all(np.isfinite(array)) and array.dtype.kind == "f":
        raise DiagnosticStop("INCONCLUSIVE", f"nonfinite q array: {name}")
    return {
        "name": name,
        "npz_key": npz_key,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "raw_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(np.asarray(value).shape),
            "dtype": np.asarray(value).dtype.str,
            "raw_sha256": hashlib.sha256(
                np.ascontiguousarray(value).tobytes(order="C")
            ).hexdigest(),
        }
        for key, value in sorted(arrays.items())
    }


def _validate_npz_references(
    payload: Any, schema: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Require every nested NPZ reference to resolve to its exact array bytes."""
    checked = []

    def visit(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            if "npz_key" in value:
                key = value["npz_key"]
                if key not in schema:
                    raise DiagnosticStop(
                        "INCONCLUSIVE", f"receipt NPZ key is absent: {location}:{key}"
                    )
                expected = schema[key]
                observed_hash = value.get("raw_sha256", value.get("sha256"))
                if (
                    value.get("shape") != expected["shape"]
                    or value.get("dtype") != expected["dtype"]
                    or observed_hash != expected["raw_sha256"]
                ):
                    raise DiagnosticStop(
                        "INCONCLUSIVE", f"receipt NPZ binding differs: {location}:{key}"
                    )
                checked.append({"location": location, "npz_key": key})
            for key, item in value.items():
                visit(item, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{location}[{index}]")

    visit(payload, "receipt")
    keys = [item["npz_key"] for item in checked]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    missing = sorted(set(schema) - set(keys))
    if duplicates or missing:
        raise DiagnosticStop(
            "INCONCLUSIVE",
            f"receipt NPZ coverage changed: missing={missing}, duplicates={duplicates}",
        )
    stable_checked = sorted(checked, key=lambda item: (item["location"], item["npz_key"]))
    return {
        "reference_count": len(checked),
        "referenced_key_count": len(set(keys)),
        "schema_key_count": len(schema),
        "missing_keys": [],
        "duplicate_keys": [],
        "referenced_key_set_equals_schema_key_set": True,
        "all_references_resolve_exactly": True,
        "reference_locations_semantic_sha256": r2r.semantic_sha256(stable_checked),
    }


def _zerojet_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "format",
        "multipolar_gate_value_max_abs",
        "global_all_node_Jacobian_max_abs_A-1",
        "global_all_node_value_and_Jacobian_exact_zero",
        "q_signature_orbit_count",
        "q_signature_orbits_sha256",
        "all_nodes_covered_exactly_once",
        "production_local_source_order_sha256",
        "all_local_senders_unique",
        "full_local_hessian_performed",
        "node_count",
        "local_sample_count",
        "local_Hessian_shape_per_node",
        "local_all_node_value_max_abs",
        "local_all_node_Jacobian_max_abs_A-1",
        "local_all_node_Hessian_max_abs_A-2",
        "local_all_node_finite",
        "nonzero_local_production_probe",
        "nonzero_probe_base_sha256",
        "nonzero_probe_nodes_sha256",
        "pass",
    )
    summary = {key: receipt.get(key) for key in keys}
    summary["raw_q_orbit_hash_role"] = "diagnostic_only_not_a_gate"
    return summary


def _zerojet_physical_exact(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("multipolar_gate_value_max_abs") == 0.0
        and summary.get("global_all_node_Jacobian_max_abs_A-1") == 0.0
        and summary.get("global_all_node_value_and_Jacobian_exact_zero") is True
        and summary.get("local_all_node_value_max_abs") == 0.0
        and summary.get("local_all_node_Jacobian_max_abs_A-1") == 0.0
        and summary.get("local_all_node_Hessian_max_abs_A-2") == 0.0
        and summary.get("local_all_node_finite") is True
        and summary.get("all_nodes_covered_exactly_once") is True
        and summary.get("all_local_senders_unique") is True
        and summary.get("full_local_hessian_performed") is True
        and summary.get("pass") is True
    )


def capture_q_snapshot(
    reference: Atoms,
    reference_name: str,
    device_name: str,
    *,
    full_local_hessian: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build and fully record one actual production reference neighborhood."""
    graph = r2r.fixed_reference_neighborhood(
        reference, device=device_name, dtype=torch.float64
    )
    # These divisions must execute on the graph's actual device.  Recomputing
    # q from host-copied weight/normalization arrays would erase the CUDA
    # index_add_/division rounding that this diagnostic is designed to record.
    center_q_t = 1.0 / graph.normalization
    neighbor_q_t = graph.weight / graph.normalization[graph.receiver]
    zerojet = r2r.background_rank0_zero_jet_audit(
        graph, full_local_hessian=full_local_hessian
    )
    receiver = _little_endian_array(graph.receiver, "<i8")
    sender = _little_endian_array(graph.sender, "<i8")
    image = _little_endian_array(graph.image_integer, "<i8")
    distance = _little_endian_array(graph.reference_distance_A, "<f8")
    weight = _little_endian_array(graph.weight, "<f8")
    normalization = _little_endian_array(graph.normalization, "<f8")
    center_q = _little_endian_array(center_q_t, "<f8")
    neighbor_q = _little_endian_array(neighbor_q_t, "<f8")
    edge_indices = []
    local_sender = []
    offsets = [0]
    source_payload = []
    for node in range(len(reference)):
        selected = np.flatnonzero(receiver == node).astype("<i8", copy=False)
        node_sender = sender[selected]
        edge_indices.extend(int(value) for value in selected)
        local_sender.extend(int(value) for value in node_sender)
        offsets.append(len(local_sender))
        source_payload.append(
            {"receiver": node, "sender_order": [int(value) for value in node_sender]}
        )
    source_offsets = np.asarray(offsets, dtype="<i8")
    source_sender = np.asarray(local_sender, dtype="<i8")
    source_edge_index = np.asarray(edge_indices, dtype="<i8")
    arrays = {
        "receiver": receiver,
        "sender": sender,
        "image_integer": image,
        "local_source_order_offsets": source_offsets,
        "local_source_order_sender": source_sender,
        "local_source_order_edge_index": source_edge_index,
        "reference_distance_A": distance,
        "weight": weight,
        "normalization": normalization,
        "center_q": center_q,
        "neighbor_q": neighbor_q,
    }
    device_tag = "CPU" if device_name == "cpu" else "CUDA"
    prefix = f"{reference_name}_{device_tag}"
    array_receipts = {
        name: _array_receipt(name, value, f"q_{prefix}_{name}")
        for name, value in arrays.items()
    }
    round_payload = np.ascontiguousarray(
        np.concatenate((center_q, neighbor_q)), dtype="<f8"
    )
    rounded = {}
    for decimals in ROUND_DECIMALS:
        value = np.ascontiguousarray(
            np.round(round_payload.astype(np.float64), decimals=decimals),
            dtype="<f8",
        )
        rounded[str(decimals)] = {
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            "npz_key": f"q_{prefix}_round{decimals}_payload",
            "shape": list(value.shape),
            "dtype": value.dtype.str,
        }
        arrays[f"round{decimals}_payload"] = value
    source_hash = r2r.semantic_sha256(source_payload)
    if source_hash != zerojet.get("production_local_source_order_sha256"):
        raise DiagnosticStop("INCONCLUSIVE", "q source-order reconstruction changed")
    public = {
        "reference": reference_name,
        "device": device_name,
        "atom_count": len(reference),
        "edge_count": int(receiver.size),
        "arrays": array_receipts,
        "local_source_order_semantic_sha256": source_hash,
        "round_payload": rounded,
        "zero_jet_actual": _zerojet_summary(zerojet),
        "zero_jet_physical_exact": _zerojet_physical_exact(
            _zerojet_summary(zerojet)
        ),
    }
    return public, arrays


def _float_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.ascontiguousarray(left, dtype="<f8")
    right = np.ascontiguousarray(right, dtype="<f8")
    if left.shape != right.shape:
        raise DiagnosticStop("INCONCLUSIVE", "q float array shape changed")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise DiagnosticStop("INCONCLUSIVE", "q float comparison is nonfinite")
    difference = np.abs(left - right)
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0e-300)
    nonnegative = bool(np.all(left >= 0.0) and np.all(right >= 0.0))
    ulp: dict[str, Any] | None = None
    if nonnegative:
        left_bits = left.view("<u8").reshape(-1)
        right_bits = right.view("<u8").reshape(-1)
        ulp_values = np.asarray(
            [abs(int(a) - int(b)) for a, b in zip(left_bits, right_bits)],
            dtype=np.float64,
        )
        nonzero = ulp_values[ulp_values > 0.0]
        ulp = {
            "max": float(np.max(ulp_values)) if ulp_values.size else 0.0,
            "p99": float(np.percentile(ulp_values, 99.0)) if ulp_values.size else 0.0,
            "median_nonzero": float(np.median(nonzero)) if nonzero.size else 0.0,
            "nonzero_count": int(nonzero.size),
        }
    return {
        "shape": list(left.shape),
        "exact_different_count": int(np.count_nonzero(left != right)),
        "max_abs": float(np.max(difference)) if difference.size else 0.0,
        "max_rel": float(np.max(difference / denominator)) if difference.size else 0.0,
        "nonnegative": nonnegative,
        "ULP": ulp,
    }


def compare_q_snapshots(
    cpu_public: Mapping[str, Any],
    cpu_arrays: Mapping[str, np.ndarray],
    gpu_public: Mapping[str, Any],
    gpu_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    discrete_names = (
        "receiver",
        "sender",
        "image_integer",
        "local_source_order_offsets",
        "local_source_order_sender",
        "local_source_order_edge_index",
    )
    float_names = (
        "reference_distance_A",
        "weight",
        "normalization",
        "center_q",
        "neighbor_q",
    )
    discrete = {
        name: bool(np.array_equal(cpu_arrays[name], gpu_arrays[name]))
        for name in discrete_names
    }
    source_hash_equal = bool(
        cpu_public["local_source_order_semantic_sha256"]
        == gpu_public["local_source_order_semantic_sha256"]
    )
    topology_exact = bool(all(discrete.values()) and source_hash_equal)
    if not topology_exact:
        raise DiagnosticStop(
            "INCONCLUSIVE", "CPU/CUDA q topology or production source order differs"
        )
    floats = {
        name: _float_comparison(cpu_arrays[name], gpu_arrays[name])
        for name in float_names
    }
    normalized_q_max_abs = max(
        floats["center_q"]["max_abs"], floats["neighbor_q"]["max_abs"]
    )

    receiver = cpu_arrays["receiver"]
    sender = cpu_arrays["sender"]
    per_node_sorted_q_max_abs = 0.0
    stable_q_sort_sender_different_nodes = []
    for node in range(int(cpu_public["atom_count"])):
        selected = np.flatnonzero(receiver == node)
        left = cpu_arrays["neighbor_q"][selected]
        right = gpu_arrays["neighbor_q"][selected]
        per_node_sorted_q_max_abs = max(
            per_node_sorted_q_max_abs,
            float(np.max(np.abs(np.sort(left) - np.sort(right)))),
        )
        left_order = np.argsort(left, kind="stable")
        right_order = np.argsort(right, kind="stable")
        if not np.array_equal(sender[selected][left_order], sender[selected][right_order]):
            stable_q_sort_sender_different_nodes.append(node)

    rounded = {}
    for decimals in ROUND_DECIMALS:
        name = f"round{decimals}_payload"
        comparison = _float_comparison(cpu_arrays[name], gpu_arrays[name])
        comparison["CPU_sha256"] = cpu_public["round_payload"][str(decimals)][
            "sha256"
        ]
        comparison["CUDA_sha256"] = gpu_public["round_payload"][str(decimals)][
            "sha256"
        ]
        comparison["hash_equal"] = bool(
            comparison["CPU_sha256"] == comparison["CUDA_sha256"]
        )
        rounded[str(decimals)] = comparison

    zerojet_exact = bool(
        cpu_public["zero_jet_physical_exact"]
        and gpu_public["zero_jet_physical_exact"]
    )
    round12_13_equal = all(
        rounded[str(decimals)]["exact_different_count"] == 0
        and rounded[str(decimals)]["hash_equal"]
        for decimals in (12, 13)
    )
    serialization_only = bool(
        topology_exact
        and normalized_q_max_abs <= Q_MAX_ABS_TOLERANCE
        and round12_13_equal
        and zerojet_exact
    )
    return {
        "discrete_arrays_exact": discrete,
        "local_source_order_hash_exact": source_hash_equal,
        "topology_and_source_order_exact": topology_exact,
        "float_arrays_CPU_vs_CUDA": floats,
        "normalized_q_max_abs": normalized_q_max_abs,
        "normalized_q_max_abs_tolerance": Q_MAX_ABS_TOLERANCE,
        "per_node_sorted_neighbor_q_max_abs": per_node_sorted_q_max_abs,
        "stable_q_sort_sender_different_node_count": len(
            stable_q_sort_sender_different_nodes
        ),
        "stable_q_sort_sender_different_nodes": stable_q_sort_sender_different_nodes,
        "round_payload_CPU_vs_CUDA": rounded,
        "round12_and_round13_elementwise_and_hash_equal": round12_13_equal,
        "CPU_and_CUDA_physical_zero_jet_exact": zerojet_exact,
        "raw_q_orbit_hash_equal": bool(
            cpu_public["zero_jet_actual"]["q_signature_orbits_sha256"]
            == gpu_public["zero_jet_actual"]["q_signature_orbits_sha256"]
        ),
        "raw_q_orbit_hash_role": "diagnostic_only_not_a_gate",
        "Q_FLOAT_SERIALIZATION_ONLY": serialization_only,
    }


def _point_provenance(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "formal_R2O_graph_semantic_sha256",
        "baseline_formal_R2O_graph_sha256",
        "baseline_formal_R2O_graph_hash_match",
        "formal_R2O_graph_mode",
        "assignment_and_MIC_semantic_sha256",
        "assignment_maximum_distance_A",
        "assignment_minimum_uniqueness_gap_A",
        "reference_semantic_sha256",
        "endpoint_state_sha256",
    )
    if any(key not in receipt for key in required):
        raise DiagnosticStop("INCONCLUSIVE", "combined E/F query receipt is incomplete")
    result = {key: receipt[key] for key in required}
    if (
        result["formal_R2O_graph_mode"] != "baseline"
        or result["baseline_formal_R2O_graph_hash_match"] is not True
        or float(result["assignment_minimum_uniqueness_gap_A"]) <= 0.0
    ):
        raise DiagnosticStop("NONSMOOTH_CANDIDATE", "point graph/assignment invalid")
    return result


def _empirical_order(previous_error: float, current_error: float) -> dict[str, Any]:
    if previous_error == 0.0 or current_error == 0.0:
        if previous_error == 0.0 and current_error == 0.0:
            case = "both_zero"
        elif previous_error == 0.0:
            case = "coarser_zero"
        else:
            case = "finer_zero"
        return {"value": None, "zero_case": case}
    return {
        "value": math.log2(previous_error / current_error),
        "zero_case": None,
    }


def analyze_fd_points(
    points: Sequence[Mapping[str, Any]],
    *,
    force_target: float,
    jacobian_target: float,
    base_point_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen six-step formulas and predeclared classifications."""
    if [float(item["h_A"]) for item in points] != list(FD_STEPS_A):
        raise ValueError("FD point order/steps differ from the canonical contract")
    rows = []
    for item in points:
        h = float(item["h_A"])
        energy_derivative = -(
            float(item["energy_plus_eV"]) - float(item["energy_minus_eV"])
        ) / (2.0 * h)
        force_derivative = (
            float(item["force_plus_eV_A"]) - float(item["force_minus_eV_A"])
        ) / (2.0 * h)
        left = (force_target - float(item["force_minus_eV_A"])) / h
        right = (float(item["force_plus_eV_A"]) - force_target) / h
        values = (
            energy_derivative,
            force_derivative,
            left,
            right,
            abs(energy_derivative - force_target),
            abs(force_derivative - jacobian_target),
            abs(right - left),
        )
        if not all(math.isfinite(value) for value in values):
            raise DiagnosticStop("INCONCLUSIVE", "FD analysis produced nonfinite value")
        rows.append(
            {
                **dict(item),
                "energy_FD_force_eV_A": energy_derivative,
                "energy_FD_force_abs_error_eV_A": abs(
                    energy_derivative - force_target
                ),
                "force_FD_Hessian_target_sign_eV_A2": force_derivative,
                "force_FD_Hessian_abs_error_eV_A2": abs(
                    force_derivative - jacobian_target
                ),
                "left_force_slope_eV_A2": left,
                "right_force_slope_eV_A2": right,
                "left_right_slope_jump_eV_A2": abs(right - left),
            }
        )
    for index, row in enumerate(rows):
        if index == 0:
            row.update(
                {
                    "empirical_order_energy_error": None,
                    "empirical_order_force_error": None,
                    "empirical_order_slope_jump": None,
                    "Richardson_force_FD_eV_A2": None,
                    "Richardson_force_abs_error_eV_A2": None,
                }
            )
            continue
        previous = rows[index - 1]
        energy_order = _empirical_order(
            previous["energy_FD_force_abs_error_eV_A"],
            row["energy_FD_force_abs_error_eV_A"],
        )
        force_order = _empirical_order(
            previous["force_FD_Hessian_abs_error_eV_A2"],
            row["force_FD_Hessian_abs_error_eV_A2"],
        )
        jump_order = _empirical_order(
            previous["left_right_slope_jump_eV_A2"],
            row["left_right_slope_jump_eV_A2"],
        )
        richardson = row["force_FD_Hessian_target_sign_eV_A2"] + (
            row["force_FD_Hessian_target_sign_eV_A2"]
            - previous["force_FD_Hessian_target_sign_eV_A2"]
        ) / 3.0
        row.update(
            {
                "empirical_order_energy_error": energy_order,
                "empirical_order_force_error": force_order,
                "empirical_order_slope_jump": jump_order,
                "Richardson_force_FD_eV_A2": richardson,
                "Richardson_force_abs_error_eV_A2": abs(
                    richardson - jacobian_target
                ),
            }
        )

    force_errors = [row["force_FD_Hessian_abs_error_eV_A2"] for row in rows]
    energy_errors = [row["energy_FD_force_abs_error_eV_A"] for row in rows]
    energy_orders = [row["empirical_order_energy_error"] for row in rows[1:]]
    force_orders = [row["empirical_order_force_error"] for row in rows[1:]]
    jump_orders = [row["empirical_order_slope_jump"] for row in rows[1:]]
    graph_keys = (
        "formal_R2O_graph_semantic_sha256",
        "assignment_and_MIC_semantic_sha256",
    )
    graph_stable = all(
        len(
            {
                base_point_provenance[key],
                *(
                    row[side + "_point_provenance"][key]
                    for row in rows
                    for side in ("minus", "plus")
                ),
            }
        )
        == 1
        for key in graph_keys
    )
    strictly_decreasing = all(
        finer < coarser for coarser, finer in zip(force_errors, force_errors[1:])
    )
    energy_strictly_decreasing = all(
        finer < coarser for coarser, finer in zip(energy_errors, energy_errors[1:])
    )
    energy_orders_ok = all(
        item["value"] is not None and 1.8 <= item["value"] <= 2.2
        for item in energy_orders
    )
    energy_path_consistent = bool(
        graph_stable and energy_strictly_decreasing and energy_orders_ok
    )
    force_orders_ok = all(
        item["value"] is not None and 1.8 <= item["value"] <= 2.2
        for item in force_orders
    )
    jump_orders_ok = all(
        item["value"] is not None and 0.8 <= item["value"] <= 1.2
        for item in jump_orders
    )
    h5_index = list(FD_STEPS_A).index(5.0e-5)
    richardson_last_three = [
        row["Richardson_force_abs_error_eV_A2"] for row in rows[-3:]
    ]
    fd_confirmed = bool(
        graph_stable
        and strictly_decreasing
        and force_orders_ok
        and jump_orders_ok
        and force_errors[h5_index] <= 1.0e-5
        and all(value <= 1.0e-7 for value in richardson_last_three)
    )
    jump_outside_broad = [
        item["value"] is None or not (0.5 <= item["value"] <= 1.5)
        for item in jump_orders
    ]
    two_adjacent_jump_failures = any(
        left and right
        for left, right in zip(jump_outside_broad, jump_outside_broad[1:])
    )
    nonsmooth_candidate = bool(not graph_stable or two_adjacent_jump_failures)
    float_floor_candidate = bool(
        graph_stable
        and not nonsmooth_candidate
        and force_errors[-1] >= 0.8 * force_errors[-2]
    )
    return {
        "points": rows,
        "checks": {
            "all_point_graph_and_assignment_topology_stable": graph_stable,
            "all_six_force_errors_strictly_decrease": strictly_decreasing,
            "all_six_energy_errors_strictly_decrease_report_only": (
                energy_strictly_decreasing
            ),
            "all_five_energy_orders_in_1p8_2p2_report_only": energy_orders_ok,
            "energy_path_consistent_report_only": energy_path_consistent,
            "energy_path_consistency_physics_gate_authorized": False,
            "all_five_force_orders_in_1p8_2p2": force_orders_ok,
            "all_five_jump_orders_in_0p8_1p2": jump_orders_ok,
            "force_error_at_5e-5_A_eV_A2": force_errors[h5_index],
            "force_error_at_5e-5_A_limit_eV_A2": 1.0e-5,
            "last_three_Richardson_errors_eV_A2": richardson_last_three,
            "last_three_Richardson_limit_eV_A2": 1.0e-7,
            "two_adjacent_jump_orders_outside_0p5_1p5": (
                two_adjacent_jump_failures
            ),
            "smallest_h_error_at_least_0p8_previous": bool(
                force_errors[-1] >= 0.8 * force_errors[-2]
            ),
        },
        "FD_TRUNCATION_CONFIRMED": fd_confirmed,
        "energy_path_consistent_report_only": energy_path_consistent,
        "NONSMOOTH_CANDIDATE": nonsmooth_candidate,
        "FLOAT_FLOOR_CANDIDATE": float_floor_candidate,
    }


def classify_diagnostic(fd: Mapping[str, Any], q: Mapping[str, Any]) -> str:
    fd_ok = bool(fd.get("FD_TRUNCATION_CONFIRMED"))
    q_ok = bool(q.get("Q_FLOAT_SERIALIZATION_ONLY"))
    if fd_ok and q_ok:
        return "ATTRIBUTED_BOTH"
    if fd_ok:
        return "DIAGNOSTIC_TRUNCATION"
    if fd.get("NONSMOOTH_CANDIDATE"):
        return "NONSMOOTH_CANDIDATE"
    if fd.get("FLOAT_FLOOR_CANDIDATE"):
        return "FLOAT_FLOOR_CANDIDATE"
    if q_ok:
        return "NUMERICAL_IDENTITY"
    return "INCONCLUSIVE"


def _load_mechanics_targets() -> dict[str, float]:
    path = ATTEMPT2_ROOT / "artifacts/mechanics/mechanics_arrays.npz"
    _require_hash(
        path,
        EXPECTED_ATTEMPT2_ARTIFACT_SHA256["mechanics_arrays"],
        "mechanics target arrays",
    )
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != {
            "reference_force_eV_A",
            "reference_Hessian_eV_A2",
            "thermal0_force_eV_A",
            "thermal0_Hessian_eV_A2",
        }:
            raise DiagnosticStop("INCONCLUSIVE", "mechanics target schema changed")
        force_target = float(loaded["thermal0_force_eV_A"][COORDINATE])
        jacobian_target = float(
            -loaded["thermal0_Hessian_eV_A2"][
                FLAT_COORDINATE, FLAT_COORDINATE
            ]
        )
    if not math.isfinite(force_target) or not math.isfinite(jacobian_target):
        raise DiagnosticStop("INCONCLUSIVE", "mechanics targets are nonfinite")
    return {
        "force_target_eV_A": force_target,
        "force_jacobian_target_eV_A2": jacobian_target,
        "target_definition": (
            "F0=attempt2 thermal0 model-output force[2,1]; "
            "J0=-attempt2 thermal0 model-output Hessian[7,7]"
        ),
        "force_or_energy_label": False,
    }


def _evaluate_public_combined(
    model: torch.nn.Module,
    structure: Atoms,
    reference: Atoms,
    device: str,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    energy, force, receipt = formal._combined_ef(model, structure, reference, device)
    force = np.ascontiguousarray(force, dtype="<f8")
    if not math.isfinite(energy) or not np.all(np.isfinite(force)):
        raise DiagnosticStop("INCONCLUSIVE", "public combined E/F is nonfinite")
    return energy, force, _point_provenance(receipt)


def _normalize_cuda_device(device: str) -> tuple[str, int]:
    requested = torch.device(device)
    if requested.type != "cuda":
        raise DiagnosticStop("INCONCLUSIVE", "full R2R-0D requires CUDA")
    if not torch.cuda.is_available():
        raise DiagnosticStop("INCONCLUSIVE", "CUDA is unavailable")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise DiagnosticStop("INCONCLUSIVE", "requested CUDA index is unavailable")
    return f"cuda:{index}", index


def _validate_v100_runtime(cuda_index: int) -> str:
    name = torch.cuda.get_device_name(cuda_index)
    if name not in ALLOWED_V100_DEVICE_NAMES:
        raise DiagnosticStop(
            "INCONCLUSIVE",
            "full R2R-0D requires an exact allowlisted NVIDIA V100 device; "
            f"observed {name!r}",
        )
    return name


def _validate_float32_origin() -> None:
    if torch.get_default_dtype() != torch.float32:
        raise DiagnosticStop(
            "INCONCLUSIVE", "default dtype must originate as float32"
        )


def _synchronize_cuda(index: int) -> None:
    torch.cuda.synchronize(index)


def _runtime_receipt(
    device_request: str,
    resolved_device: str,
    cuda_index: int,
    validated_device_name: str,
) -> dict[str, Any]:
    result = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device_request": device_request,
        "resolved_device": resolved_device,
        "resolved_cuda_index": cuda_index,
        "default_dtype": str(torch.get_default_dtype()),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": os.environ.get(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        ),
        "cuda_available": torch.cuda.is_available(),
    }
    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        live_name = torch.cuda.get_device_name(cuda_index)
        if live_name != validated_device_name or live_name not in ALLOWED_V100_DEVICE_NAMES:
            raise DiagnosticStop("INCONCLUSIVE", "validated V100 identity changed")
        result["cuda"] = {
            "name": live_name,
            "exact_name_allowlist": list(ALLOWED_V100_DEVICE_NAMES),
            "exact_name_allowlist_pass": True,
            "capability": list(torch.cuda.get_device_capability(cuda_index)),
            "runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    return result


def _base_receipt(
    *,
    status: str,
    provenance: Mapping[str, Any] | None,
    budget: Budget,
    execution_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "status": status,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "adjudicates_attempt2": False,
        "attempt2_scientific_status_preserved": ATTEMPT2_SCIENTIFIC_STATUS,
        "formal_authorization": False,
        "fit_or_training_performed": False,
        "deployment_performed": False,
        "labels_used": False,
        "held_or_support_access": False,
        "GO_marker_created": False,
        "execution_authorization": dict(execution_authorization),
        "provenance": provenance,
        "budget": budget.receipt(),
    }


DIAGNOSTIC_SUCCESS_STATUSES = frozenset(
    {
        "ATTRIBUTED_BOTH",
        "DIAGNOSTIC_TRUNCATION",
        "NONSMOOTH_CANDIDATE",
        "FLOAT_FLOOR_CANDIDATE",
        "NUMERICAL_IDENTITY",
        "INCONCLUSIVE",
    }
)


def _terminal_set(root: Path) -> set[str]:
    return {
        name for name in ("RUNNING", "FAILED", "DONE") if (root / name).exists()
    }


def _finalize_success(root: Path, receipt_path: Path, status: str) -> None:
    receipt_hash = sha256(receipt_path)
    _atomic_bytes(root / "EXIT_CODE", b"0\n")
    running = root / "RUNNING"
    if running.exists():
        running.unlink()
    _atomic_bytes(
        root / "DONE", (status + "\n" + receipt_hash + "\n").encode("ascii")
    )
    if _terminal_set(root) != {"DONE"}:
        raise RuntimeError("R2R-0D success terminal XOR failed")


def _finalize_failure(root: Path, receipt_path: Path, status: str) -> None:
    receipt_hash = sha256(receipt_path)
    _atomic_bytes(root / "EXIT_CODE", b"2\n")
    running = root / "RUNNING"
    if running.exists():
        running.unlink()
    _atomic_bytes(
        root / "FAILED", (status + "\n" + receipt_hash + "\n").encode("ascii")
    )
    if _terminal_set(root) != {"FAILED"}:
        raise RuntimeError("R2R-0D failure terminal XOR failed")


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if len(loaded.files) != len(set(loaded.files)):
                raise ValueError("duplicate NPZ keys")
            return {key: loaded[key].copy() for key in loaded.files}
    except Exception as exception:
        raise ValueError("R2R-0D completed NPZ is invalid") from exception


def _output_artifact_manifest(
    arrays_path: Path, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    schema = _array_schema(arrays)
    return {
        "array_file": arrays_path.name,
        "array_file_sha256": sha256(arrays_path),
        "array_schema": schema,
        "array_schema_semantic_sha256": _semantic_sha256(schema),
    }


def _validate_completed_output(
    root: Path,
    *,
    mode: str,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Fully validate a DONE directory and return without scientific recompute."""
    if not root.is_dir() or root.is_symlink() or _path_has_symlink(root):
        raise ValueError("R2R-0D completed output root is invalid")
    if _terminal_set(root) != {"DONE"}:
        raise ValueError("R2R-0D completed terminal markers violate XOR")
    if not (root / "EXIT_CODE").is_file() or (root / "EXIT_CODE").read_bytes() != b"0\n":
        raise ValueError("R2R-0D completed EXIT_CODE is not 0")
    arrays_name = (
        "preflight_arrays.npz" if mode == "cpu-preflight" else "diagnostic_arrays.npz"
    )
    expected_inventory = {"DONE", "EXIT_CODE", "diagnostic_receipt.json", arrays_name}
    observed_inventory = {path.name for path in root.iterdir()}
    if observed_inventory != expected_inventory or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise ValueError("R2R-0D completed output inventory changed")
    receipt_path = root / "diagnostic_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise ValueError("R2R-0D completed receipt is invalid") from exception
    expected_format = PREFLIGHT_FORMAT if mode == "cpu-preflight" else FORMAT
    allowed_status = (
        {"DIAGNOSTIC_PREFLIGHT_ONLY"}
        if mode == "cpu-preflight"
        else DIAGNOSTIC_SUCCESS_STATUSES
    )
    if (
        receipt.get("format") != expected_format
        or receipt.get("status") not in allowed_status
        or receipt.get("stopped_early") is True
    ):
        raise ValueError("R2R-0D completed receipt format/status changed")
    expected_done = (
        receipt["status"] + "\n" + sha256(receipt_path) + "\n"
    ).encode("ascii")
    if (root / "DONE").read_bytes() != expected_done:
        raise ValueError("R2R-0D DONE does not bind the exact receipt")
    if (
        receipt.get("diagnostic_contract_sha256") != DIAGNOSTIC_CONTRACT_SHA256
        or receipt.get("execution_authorization") != authorization
        or receipt.get("completion_authorization_revalidation") != authorization
        or receipt.get("attempt2_scientific_status_preserved")
        != ATTEMPT2_SCIENTIFIC_STATUS
        or receipt.get("formal_authorization") is not False
        or receipt.get("fit_or_training_performed") is not False
        or receipt.get("labels_used") is not False
        or receipt.get("held_or_support_access") is not False
        or receipt.get("GO_marker_created") is not False
    ):
        raise ValueError("R2R-0D completed receipt authorization/scope changed")
    if receipt.get("provenance") != validate_frozen_provenance():
        raise ValueError("R2R-0D completed frozen provenance receipt changed")
    budget = receipt.get("budget")
    if (
        not isinstance(budget, dict)
        or not 0.0 <= budget.get("CPU_wall_seconds", -1.0) <= CPU_BUDGET_SECONDS
        or not 0.0 <= budget.get("GPU_wall_seconds", -1.0) <= GPU_BUDGET_SECONDS
        or budget.get("CPU_wall_seconds_max") != CPU_BUDGET_SECONDS
        or budget.get("GPU_wall_seconds_max") != GPU_BUDGET_SECONDS
    ):
        raise ValueError("R2R-0D completed budget receipt changed")
    arrays_path = root / arrays_name
    arrays = _load_npz_arrays(arrays_path)
    manifest = _output_artifact_manifest(arrays_path, arrays)
    if receipt.get("output_artifact_manifest") != manifest:
        raise ValueError("R2R-0D completed artifact hash/schema manifest changed")
    schema = manifest["array_schema"]
    if mode == "cpu-preflight":
        references = {"q_CPU_snapshots": receipt.get("q_CPU_snapshots")}
        expected_audit = receipt.get("preflight_array_reference_audit")
        if receipt.get("CUDA_evaluated") is not False:
            raise ValueError("R2R-0D preflight recovery claims CUDA evaluation")
        if (
            receipt.get("preflight_arrays_sha256")
            != manifest["array_file_sha256"]
            or receipt.get("preflight_array_schema") != schema
        ):
            raise ValueError("R2R-0D preflight redundant array binding changed")
    else:
        runtime = receipt.get("runtime", {})
        cuda = runtime.get("cuda", {}) if isinstance(runtime, dict) else {}
        if (
            cuda.get("name") not in ALLOWED_V100_DEVICE_NAMES
            or cuda.get("exact_name_allowlist_pass") is not True
        ):
            raise ValueError("R2R-0D completed runtime is not an allowlisted V100")
        probe = receipt.get("probe", {})
        references = {
            "q_snapshots": receipt.get("q_snapshots"),
            "base_full_force_array": probe.get("base", {}).get("full_force_array"),
            "finite_difference_points": receipt.get("finite_difference", {}).get(
                "points"
            ),
        }
        expected_audit = receipt.get("diagnostic_array_reference_audit")
        if (
            receipt.get("diagnostic_arrays_sha256")
            != manifest["array_file_sha256"]
            or receipt.get("diagnostic_array_schema") != schema
        ):
            raise ValueError("R2R-0D diagnostic redundant array binding changed")
    observed_audit = _validate_npz_references(references, schema)
    if observed_audit != expected_audit:
        raise ValueError("R2R-0D completed NPZ reference audit changed")
    return {**receipt, "completion_recovered_without_recompute": True}


def _existing_completion_or_fresh(
    output: Path,
    *,
    mode: str,
    authorization: Mapping[str, Any],
) -> tuple[Path | None, dict[str, Any] | None]:
    root = _validated_output_path(output)
    if root.exists():
        if not root.is_dir():
            raise FileExistsError("R2R-0D output exists and is not a directory")
        if (root / "DONE").exists():
            return None, _validate_completed_output(
                root, mode=mode, authorization=authorization
            )
        # RUNNING, FAILED, an empty directory, or any partial artifact is never
        # resumable because the diagnostic has no retry semantics.
        raise FileExistsError("R2R-0D partial/FAILED output recovery is forbidden")
    return _fresh_output(root), None


def _reconcile_exception_budget(
    budget: Budget, run_wall_started: float, exception: BaseException
) -> BaseException:
    """Make a total-wall overrun take precedence in every terminal path."""
    try:
        budget.reconcile_total_wall(time.perf_counter() - run_wall_started)
    except DiagnosticStop as budget_exception:
        return budget_exception
    return exception


def run_diagnostic(
    output: Path,
    *,
    device: str,
    freeze_manifest: Path,
    authorization_marker: Path,
) -> dict[str, Any]:
    """Run the fixed CPU/CUDA R2R-0D package in a fresh independent directory."""
    # Authorization and all four live source files are checked before an output
    # directory can be created.  This also validates v5/input/attempt2 bytes.
    authorization = validate_execution_authorization(
        freeze_manifest, authorization_marker
    )
    root, recovered = _existing_completion_or_fresh(
        output, mode="diagnostic", authorization=authorization
    )
    if recovered is not None:
        return recovered
    assert root is not None
    run_wall_started = time.perf_counter()
    budget = Budget()
    provenance: dict[str, Any] | None = None
    arrays: dict[str, np.ndarray] = {}
    try:
        device_request = device
        device, cuda_index = _normalize_cuda_device(device)
        validated_device_name = _validate_v100_runtime(cuda_index)
        _validate_float32_origin()
        if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1":
            raise DiagnosticStop(
                "INCONCLUSIVE", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD must equal 1"
            )

        started = time.perf_counter()
        provenance = validate_frozen_provenance()
        thermal, reference6, reference8, label_audit = load_label_free_geometry()
        targets = _load_mechanics_targets()
        budget.add("cpu", time.perf_counter() - started)

        q_public: dict[str, Any] = {}
        q_internal: dict[str, dict[str, np.ndarray]] = {}
        for reference_name, reference in (
            ("reference_6x6", reference6),
            ("reference_8x8", reference8),
        ):
            started = time.perf_counter()
            public, values = capture_q_snapshot(
                reference,
                reference_name,
                "cpu",
                full_local_hessian=True,
            )
            budget.add("cpu", time.perf_counter() - started)
            q_public[f"{reference_name}_CPU"] = public
            q_internal[f"{reference_name}_CPU"] = values
            arrays.update(
                {
                    f"q_{reference_name}_CPU_{name}": value
                    for name, value in values.items()
                }
            )

            _synchronize_cuda(cuda_index)
            started = time.perf_counter()
            public, values = capture_q_snapshot(
                reference,
                reference_name,
                device,
                full_local_hessian=True,
            )
            _synchronize_cuda(cuda_index)
            budget.add("gpu", time.perf_counter() - started)
            q_public[f"{reference_name}_CUDA"] = public
            q_internal[f"{reference_name}_CUDA"] = values
            arrays.update(
                {
                    f"q_{reference_name}_CUDA_{name}": value
                    for name, value in values.items()
                }
            )

        started = time.perf_counter()
        q_comparisons = {
            reference_name: compare_q_snapshots(
                q_public[f"{reference_name}_CPU"],
                q_internal[f"{reference_name}_CPU"],
                q_public[f"{reference_name}_CUDA"],
                q_internal[f"{reference_name}_CUDA"],
            )
            for reference_name in ("reference_6x6", "reference_8x8")
        }
        for reference_name, comparison in q_comparisons.items():
            frozen_identity = provenance["attempt2_q_failure_identity"][
                reference_name
            ]
            current_cpu_orbit = q_public[f"{reference_name}_CPU"][
                "zero_jet_actual"
            ]["q_signature_orbits_sha256"]
            current_cuda_orbit = q_public[f"{reference_name}_CUDA"][
                "zero_jet_actual"
            ]["q_signature_orbits_sha256"]
            comparison["attempt2_failure_identity_diagnostic"] = {
                **frozen_identity,
                "current_CPU_raw_q_orbit_sha256": current_cpu_orbit,
                "current_CUDA_raw_q_orbit_sha256": current_cuda_orbit,
                "current_CPU_matches_canonical_expected": bool(
                    current_cpu_orbit
                    == frozen_identity["canonical_expected_raw_q_orbit_sha256"]
                ),
                "current_CUDA_matches_attempt2_mechanics": bool(
                    current_cuda_orbit
                    == frozen_identity["attempt2_mechanics_raw_q_orbit_sha256"]
                ),
                "physics_gate_authorized": False,
            }
        q_classification = {
            "per_reference": q_comparisons,
            "Q_FLOAT_SERIALIZATION_ONLY": all(
                item["Q_FLOAT_SERIALIZATION_ONLY"]
                for item in q_comparisons.values()
            ),
        }
        budget.add("cpu", time.perf_counter() - started)

        _synchronize_cuda(cuda_index)
        started = time.perf_counter()
        model = formal.load_endpoint(default_inputs(), device)
        state_before = r2o.state_dict_sha256(model)
        _synchronize_cuda(cuda_index)
        budget.add("gpu", time.perf_counter() - started)
        probe = thermal[0]
        _synchronize_cuda(cuda_index)
        started = time.perf_counter()
        base_energy, base_force, base_point = _evaluate_public_combined(
            model, probe, reference6, device
        )
        _synchronize_cuda(cuda_index)
        budget.add("gpu", time.perf_counter() - started)
        arrays["base_live_force_eV_A"] = base_force
        base_force_array_receipt = _array_receipt(
            "base_live_force_eV_A", base_force, "base_live_force_eV_A"
        )
        points = []
        for index, h in enumerate(FD_STEPS_A):
            point: dict[str, Any] = {"h_A": h, "evaluation_order": ["minus", "plus"]}
            for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
                displaced = probe.copy()
                displaced.positions[COORDINATE] += sign * h
                _synchronize_cuda(cuda_index)
                started = time.perf_counter()
                energy, force, point_provenance = _evaluate_public_combined(
                    model, displaced, reference6, device
                )
                _synchronize_cuda(cuda_index)
                budget.add("gpu", time.perf_counter() - started)
                point[f"energy_{sign_name}_eV"] = energy
                point[f"force_{sign_name}_eV_A"] = float(force[COORDINATE])
                point[f"{sign_name}_point_provenance"] = point_provenance
                npz_key = f"h{index}_{sign_name}_force_eV_A"
                arrays[npz_key] = force
                point[f"{sign_name}_full_force_array"] = _array_receipt(
                    f"h{index}_{sign_name}_full_force_eV_A", force, npz_key
                )
            points.append(point)
        _synchronize_cuda(cuda_index)
        started = time.perf_counter()
        state_after = r2o.state_dict_sha256(model)
        _synchronize_cuda(cuda_index)
        budget.add("gpu", time.perf_counter() - started)
        if state_before != r2r.R2Q_ENDPOINT_STATE_SHA256 or state_after != state_before:
            raise DiagnosticStop("INCONCLUSIVE", "endpoint state changed during R2R-0D")
        started = time.perf_counter()
        fd = analyze_fd_points(
            points,
            force_target=targets["force_target_eV_A"],
            jacobian_target=targets["force_jacobian_target_eV_A2"],
            base_point_provenance=base_point,
        )
        base_force_difference = float(
            abs(base_force[COORDINATE] - targets["force_target_eV_A"])
        )
        if not math.isfinite(base_force_difference):
            raise DiagnosticStop("INCONCLUSIVE", "base force comparison is nonfinite")
        status = classify_diagnostic(fd, q_classification)
        budget.add("cpu", time.perf_counter() - started)

        array_schema = _array_schema(arrays)
        receipt_array_references = {
            "q_snapshots": q_public,
            "base_full_force_array": base_force_array_receipt,
            "finite_difference_points": fd["points"],
        }
        array_reference_audit = _validate_npz_references(
            receipt_array_references, array_schema
        )
        arrays_path = root / "diagnostic_arrays.npz"
        _atomic_npz(arrays_path, arrays)
        runtime = _runtime_receipt(
            device_request, device, cuda_index, validated_device_name
        )
        artifact_manifest = _output_artifact_manifest(arrays_path, arrays)
        completion_authorization = validate_execution_authorization(
            freeze_manifest, authorization_marker
        )
        if completion_authorization != authorization:
            raise DiagnosticStop(
                "INCONCLUSIVE", "execution authorization changed during R2R-0D"
            )
        budget.reconcile_total_wall(time.perf_counter() - run_wall_started)
        receipt = {
            **_base_receipt(
                status=status,
                provenance=provenance,
                budget=budget,
                execution_authorization=authorization,
            ),
            "completion_authorization_revalidation": completion_authorization,
            "runtime": runtime,
            "probe": {
                "thermal_global_index": 0,
                "coordinate": list(COORDINATE),
                "flat_coordinate": FLAT_COORDINATE,
                "label_audit": label_audit,
                "targets": targets,
                "base": {
                    "energy_eV": base_energy,
                    "force_coordinate_eV_A": float(base_force[COORDINATE]),
                    "force_artifact_target_abs_difference_eV_A": (
                        base_force_difference
                    ),
                    "point_provenance": base_point,
                    "full_force_array": base_force_array_receipt,
                },
            },
            "finite_difference": fd,
            "q_snapshots": q_public,
            "q_comparison": q_classification,
            "endpoint_state_sha256_before": state_before,
            "endpoint_state_sha256_after": state_after,
            "diagnostic_arrays_sha256": sha256(arrays_path),
            "diagnostic_array_schema": array_schema,
            "diagnostic_array_reference_audit": array_reference_audit,
            "output_artifact_manifest": artifact_manifest,
            "raw_plus_minus_full_force_arrays_recorded": True,
            "classification_scope": {
                "DIAGNOSTIC_TRUNCATION": (
                    "attributes only the attempt2 force-to-Hessian subgate to "
                    "second-order finite-difference truncation"
                ),
                "energy_path_consistent_report_only": fd[
                    "energy_path_consistent_report_only"
                ],
                "energy_path_can_authorize_or_veto_Hessian_attribution": False,
            },
            "interpretation": (
                "diagnostic attribution only; attempt2 remains scientific FAIL and "
                "this receipt cannot authorize fitting, training, deployment, GO, "
                "or a phonon claim"
            ),
        }
        receipt_path = root / "diagnostic_receipt.json"
        _atomic_json(receipt_path, receipt)
        _finalize_success(root, receipt_path, status)
        return receipt
    except DiagnosticStop as exception:
        exception = _reconcile_exception_budget(
            budget, run_wall_started, exception
        )
        assert isinstance(exception, DiagnosticStop)
        receipt = {
            **_base_receipt(
                status=exception.status,
                provenance=provenance,
                budget=budget,
                execution_authorization=authorization,
            ),
            "stopped_early": True,
            "stop_reason": exception.reason,
            "partial_arrays_recorded": bool(arrays),
            "interpretation": (
                "fail-closed diagnostic stop; attempt2 remains scientific FAIL"
            ),
        }
        if arrays:
            arrays_path = root / "partial_diagnostic_arrays.npz"
            _atomic_npz(arrays_path, arrays)
            receipt["partial_diagnostic_arrays_sha256"] = sha256(arrays_path)
            receipt["partial_diagnostic_array_schema"] = _array_schema(arrays)
        receipt_path = root / "diagnostic_receipt.json"
        _atomic_json(receipt_path, receipt)
        _finalize_failure(root, receipt_path, exception.status)
        return receipt
    except Exception as exception:
        original_exception = exception
        exception = _reconcile_exception_budget(
            budget, run_wall_started, exception
        )
        budget_stop = exception if isinstance(exception, DiagnosticStop) else None
        receipt = {
            **_base_receipt(
                status=(
                    budget_stop.status if budget_stop is not None else "INCONCLUSIVE"
                ),
                provenance=provenance,
                budget=budget,
                execution_authorization=authorization,
            ),
            "stopped_early": True,
            "stop_reason": (
                budget_stop.reason
                if budget_stop is not None
                else "unexpected fail-closed diagnostic exception"
            ),
            "exception_type": type(original_exception).__name__,
            "exception_message": str(original_exception),
            "partial_arrays_recorded": bool(arrays),
            "interpretation": (
                "fail-closed diagnostic stop; attempt2 remains scientific FAIL"
            ),
        }
        if arrays:
            arrays_path = root / "partial_diagnostic_arrays.npz"
            _atomic_npz(arrays_path, arrays)
            receipt["partial_diagnostic_arrays_sha256"] = sha256(arrays_path)
            receipt["partial_diagnostic_array_schema"] = _array_schema(arrays)
        receipt_path = root / "diagnostic_receipt.json"
        _atomic_json(receipt_path, receipt)
        _finalize_failure(root, receipt_path, receipt["status"])
        return receipt


def run_cpu_preflight(
    output: Path,
    *,
    freeze_manifest: Path,
    authorization_marker: Path,
) -> dict[str, Any]:
    """Bounded label-free CPU preflight; never performs the attribution run."""
    authorization = validate_execution_authorization(
        freeze_manifest, authorization_marker
    )
    root, recovered = _existing_completion_or_fresh(
        output, mode="cpu-preflight", authorization=authorization
    )
    if recovered is not None:
        return recovered
    assert root is not None
    run_wall_started = time.perf_counter()
    budget = Budget()
    provenance: dict[str, Any] | None = None
    try:
        started = time.perf_counter()
        provenance = validate_frozen_provenance()
        _, reference6, reference8, label_audit = load_label_free_geometry()
        q_public = {}
        arrays = {}
        for reference_name, reference in (
            ("reference_6x6", reference6),
            ("reference_8x8", reference8),
        ):
            public, values = capture_q_snapshot(
                reference,
                reference_name,
                "cpu",
                full_local_hessian=False,
            )
            q_public[reference_name] = public
            arrays.update(
                {
                    f"q_{reference_name}_CPU_{name}": value
                    for name, value in values.items()
                }
            )
        budget.add("cpu", time.perf_counter() - started)
        array_schema = _array_schema(arrays)
        array_reference_audit = _validate_npz_references(
            {"q_CPU_snapshots": q_public}, array_schema
        )
        arrays_path = root / "preflight_arrays.npz"
        _atomic_npz(arrays_path, arrays)
        completion_authorization = validate_execution_authorization(
            freeze_manifest, authorization_marker
        )
        if completion_authorization != authorization:
            raise DiagnosticStop(
                "INCONCLUSIVE", "execution authorization changed during preflight"
            )
        budget.reconcile_total_wall(time.perf_counter() - run_wall_started)
        receipt = {
            **_base_receipt(
                status="DIAGNOSTIC_PREFLIGHT_ONLY",
                provenance=provenance,
                budget=budget,
                execution_authorization=authorization,
            ),
            "format": PREFLIGHT_FORMAT,
            "completion_authorization_revalidation": completion_authorization,
            "label_audit": label_audit,
            "q_CPU_snapshots": q_public,
            "preflight_arrays_sha256": sha256(arrays_path),
            "preflight_array_schema": array_schema,
            "preflight_array_reference_audit": array_reference_audit,
            "output_artifact_manifest": _output_artifact_manifest(
                arrays_path, arrays
            ),
            "CUDA_evaluated": False,
            "FD_evaluated": False,
            "attribution_made": False,
        }
        receipt_path = root / "diagnostic_receipt.json"
        _atomic_json(receipt_path, receipt)
        _finalize_success(root, receipt_path, receipt["status"])
        return receipt
    except DiagnosticStop as exception:
        exception = _reconcile_exception_budget(
            budget, run_wall_started, exception
        )
        assert isinstance(exception, DiagnosticStop)
        receipt = {
            **_base_receipt(
                status=exception.status,
                provenance=provenance,
                budget=budget,
                execution_authorization=authorization,
            ),
            "format": PREFLIGHT_FORMAT,
            "stopped_early": True,
            "stop_reason": exception.reason,
            "attribution_made": False,
        }
        receipt_path = root / "diagnostic_receipt.json"
        _atomic_json(receipt_path, receipt)
        _finalize_failure(root, receipt_path, exception.status)
        return receipt
    except Exception as exception:
        original_exception = exception
        exception = _reconcile_exception_budget(
            budget, run_wall_started, exception
        )
        budget_stop = exception if isinstance(exception, DiagnosticStop) else None
        receipt = {
            **_base_receipt(
                status=(
                    budget_stop.status if budget_stop is not None else "INCONCLUSIVE"
                ),
                provenance=provenance,
                budget=budget,
                execution_authorization=authorization,
            ),
            "format": PREFLIGHT_FORMAT,
            "stopped_early": True,
            "stop_reason": (
                budget_stop.reason
                if budget_stop is not None
                else "unexpected fail-closed preflight exception"
            ),
            "exception_type": type(original_exception).__name__,
            "exception_message": str(original_exception),
            "attribution_made": False,
        }
        receipt_path = root / "diagnostic_receipt.json"
        _atomic_json(receipt_path, receipt)
        _finalize_failure(root, receipt_path, receipt["status"])
        return receipt


__all__ = [
    "ATTEMPT2_ROOT",
    "ATTEMPT2_SCIENTIFIC_STATUS",
    "CPU_BUDGET_SECONDS",
    "COORDINATE",
    "DIAGNOSTIC_CONTRACT",
    "DIAGNOSTIC_CONTRACT_SHA256",
    "DIAGNOSTIC_SOURCE_PATHS",
    "FREEZE_MANIFEST_FORMAT",
    "DiagnosticStop",
    "FD_STEPS_A",
    "FLAT_COORDINATE",
    "FORMAT",
    "GPU_BUDGET_SECONDS",
    "PREFLIGHT_FORMAT",
    "RECOMMENDED_OUTPUT_ROOT",
    "analyze_fd_points",
    "capture_q_snapshot",
    "classify_diagnostic",
    "compare_q_snapshots",
    "default_inputs",
    "load_label_free_geometry",
    "prepare_freeze_manifest",
    "run_cpu_preflight",
    "run_diagnostic",
    "sha256",
    "validate_frozen_provenance",
    "validate_execution_authorization",
]
