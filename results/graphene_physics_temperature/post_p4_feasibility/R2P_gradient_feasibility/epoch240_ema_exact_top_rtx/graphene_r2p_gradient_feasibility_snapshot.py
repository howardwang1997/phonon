#!/usr/bin/env python3
"""R2P gate-aligned, zero-update gradient-feasibility audit.

This module is deliberately diagnostic.  It measures FP64 parameter gradients
of frozen, per-configuration R2O Taylor-remainder force objectives and solves
the max-min normalized common-descent cone.  It never constructs an optimizer,
never calls ``backward`` or ``optimizer.step``, and never changes model state.

The primary cone contains, in a fixed order,

* 20 E50-seed0 complex A-prime projection objectives;
* 92 thermal total-force objectives (20 E50 + 36 T300 + 36 T600);
* 32 exact-zero small-harmonic component-MSE objectives; and
* 32 exact top-component objectives using the strict 10-6.41582 meV/A
  correction margin.

Shifted log-mean-exp and actual combined-error cones are report-only because a
shifted smooth maximum can underestimate the exact maximum by tau*log(n).
E50 seed1 is opened only after the primary certificate is atomically frozen;
it is evaluated without parameter gradients and cannot affect any direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy
import torch
from ase.io import read
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from graphene_r2o_taylor_null import (  # noqa: E402
    evaluate_structure,
    model_dtype,
    sha256,
    source_order_forces,
    state_dict_sha256,
    torch_load,
    validate_mace_architecture,
)


FORMAT = "graphene_r2p_per_configuration_gradient_feasibility_v1"
CONTRACT_FORMAT = "graphene_r2p_gradient_feasibility_contract_v1"
PRIMARY_FORMAT = "graphene_r2p_primary_common_descent_certificate_v1"
REPORT_FORMAT = "graphene_r2p_gradient_feasibility_report_v1"

FORBIDDEN_PATH_TOKENS = ("seed2", "support", "reserved", "outer_fold")
EXPECTED_GATE_STATUS = "R2O_core_checkpoint_gate_failed"
EXPECTED_BUNDLE_KIND = "R2O_diagnostic_bundle_not_authorized_for_deployment"
EXPECTED_BUNDLE_FORMAT = "graphene_r2o_deployment_bundle_v1"
EXPECTED_MODEL_STATE_SHA256 = (
    "09c29479be6c8f207211614c2b3da20f09aba0702092a2b1bf8edab5a5e6d236"
)
EXPECTED_FORMAL_GATE_SHA256 = (
    "924b13de54d2a918c16845524289036e117006e48ee454f364f442fd88ee8898"
)
EXPECTED_FORMAL_DONE_BYTES = b"R2O_core_checkpoint_gate_failed\n"

APRIME_SCALE_EV_A = 0.015
THERMAL_FORCE_SCALE_EV_A = 0.030
SMALL_RMS_SCALE_EV_A = 0.0005
SMALL_GATE_MAX_EV_A = 0.010
SMALL_ZERO_BASELINE_MAX_EV_A = 0.00641582
SMALL_CORRECTION_MARGIN_EV_A = (
    SMALL_GATE_MAX_EV_A - SMALL_ZERO_BASELINE_MAX_EV_A
)
SMOOTH_TAU_EV_A = 0.001
TOP_EXACT_TIE_TOL_EV_A = 0.0
TOP_NEAR_TIE_TOL_EV_A = 1.0e-8
FORMAL_GATE_PARITY_TOL_MEV_A = 1.0e-7

FORMAL_GRAPH_SOURCE_SEMANTICS = {
    "builder": "graphene_r2o_taylor_null.fixed_reference_graph",
    "AtomicData_source_default_dtype": "torch.float32",
    "stored_continuous_and_model_dtype": "torch.float64",
    "meaning": (
        "reproduce the frozen R2O trainer/evaluator: AtomicData is constructed "
        "under the process float32 default and floating graph tensors are then "
        "cast to float64; native-FP64 graph construction is a different semantic"
    ),
}

CONE_GO_TOL = 1.0e-3
CONE_SOLVER_GAP_TOL = 1.0e-12
CONE_SOLVER_MAX_ITERATIONS = 200_000
GRADIENT_ZERO_NORM_TOL = 1.0e-14

THERMAL_ROLE_COUNTS = {"E50": 20, "T300": 36, "T600": 36}
THERMAL_GROUP_MASS = {"E50": 0.50, "T300": 0.20, "T600": 0.20}
PRIMARY_FAMILY_COUNTS = {
    "Aprime_seed0": 20,
    "thermal_total_force": 92,
    "small_zero_RMS": 32,
    "small_zero_exact_top": 32,
}

PROTOCOL = {
    "format": FORMAT,
    "formal_failed_gate_sha256": EXPECTED_FORMAL_GATE_SHA256,
    "model_parameter_target_and_autograd_dtype": "torch.float64",
    "formal_graph_source_semantics": FORMAL_GRAPH_SOURCE_SEMANTICS,
    "optimizer_instantiated": False,
    "optimizer_step_called": False,
    "primary_family_counts": PRIMARY_FAMILY_COUNTS,
    "primary_block_count": 176,
    "Aprime_scale_eV_A": APRIME_SCALE_EV_A,
    "thermal_force_scale_eV_A": THERMAL_FORCE_SCALE_EV_A,
    "small_RMS_scale_eV_A": SMALL_RMS_SCALE_EV_A,
    "small_exact_top_correction_margin_eV_A": SMALL_CORRECTION_MARGIN_EV_A,
    "small_exact_top_margin_derivation": "0.010 - 0.00641582 eV/A",
    "exact_top_tie_tolerance_eV_A": TOP_EXACT_TIE_TOL_EV_A,
    "exact_top_tie_policy": "fail_closed_unless_the_FP64_absolute_top_is_unique",
    "exact_top_near_tie_tolerance_eV_A": TOP_NEAR_TIE_TOL_EV_A,
    "smooth_Linf_tau_eV_A": SMOOTH_TAU_EV_A,
    "smooth_Linf_role": "report_only_not_a_sufficient_max_certificate",
    "smooth_Linf_ncomp_small": 384,
    "smooth_Linf_max_underestimate_bound_eV_A": SMOOTH_TAU_EV_A
    * math.log(384),
    "seed1_policy": "opened_after_primary_certificate_and_never_differentiated",
    "actual_combined_small_policy": "report_only_cone_not_used_for_primary_direction",
    "cone_problem": (
        "max_{||d||<=1} min_i <g_i/||g_i||,d>; dual is the minimum norm "
        "convex combination of normalized gradients"
    ),
    "direction_semantics": (
        "stored d has positive dot product with each loss gradient; any future "
        "descent update would use theta <- theta - eta*d, but R2P authorizes no update"
    ),
    "cone_solver": (
        "deterministic Frank-Wolfe on the simplex with lowest-index tie break "
        "and exact quadratic line search; deterministic SLSQP is an independent "
        "cross-check and the lower-objective feasible simplex point is certified"
    ),
    "cone_solver_gap_tolerance": CONE_SOLVER_GAP_TOL,
    "cone_solver_max_iterations": CONE_SOLVER_MAX_ITERATIONS,
    "common_descent_GO_min_cosine_strictly_greater_than": CONE_GO_TOL,
}


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


PROTOCOL_SHA256 = canonical_json_sha256(PROTOCOL)
FORMAL_GRAPH_SOURCE_SEMANTICS_SHA256 = canonical_json_sha256(
    FORMAL_GRAPH_SOURCE_SEMANTICS
)


def strict_json_load(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object at {path}")
    return payload


def strict_json(path: Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    content = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _atomic_bytes(path: Path, content: bytes) -> None:
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with Path(source).open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)


def reject_forbidden_path(path: Path, label: str, *, must_exist: bool) -> Path:
    candidate = Path(path)
    lowered = str(candidate).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"{label} contains forbidden unopened data token: {candidate}")
    if ".." in candidate.parts:
        raise ValueError(f"{label} contains traversal: {candidate}")
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    # Resolve would erase a symlink component, so inspect every lexical path
    # prefix before any resolve, existence validation, open, or hash operation.
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink: {cursor}")
    resolved = candidate.resolve(strict=must_exist)
    if any(token in str(resolved).lower() for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"{label} resolves to a forbidden unopened data token")
    if must_exist and not resolved.is_file() and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _require_file(root: Path, relative: str, label: str) -> Path:
    candidate = reject_forbidden_path(root / relative, label, must_exist=True)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _array_sha256(array: np.ndarray, dtype: str = "<f8") -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.dtype(dtype)))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _state_schema(named_parameters: Sequence[tuple[str, torch.nn.Parameter]]) -> dict:
    records = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in named_parameters
    ]
    return {
        "parameters": records,
        "parameter_count": int(sum(item["numel"] for item in records)),
        "schema_sha256": canonical_json_sha256(records),
    }


def runtime_fingerprint(device: torch.device) -> dict:
    cuda = None
    if device.type == "cuda":
        cuda = {
            "device_name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "device": str(device),
        "cuda": cuda,
    }


def assert_formal_graph_source_semantics() -> None:
    """Require, but never mutate, the frozen R2O graph construction default."""

    observed = torch.get_default_dtype()
    if observed != torch.float32:
        raise RuntimeError(
            "R2P must enter with torch default dtype float32 to reproduce the "
            f"frozen R2O AtomicData graph; observed {observed}"
        )


@dataclass(frozen=True)
class ConeCertificate:
    weights: np.ndarray
    direction: np.ndarray | None
    cosines: np.ndarray
    dual_norm: float
    primal_min_cosine: float
    selected_simplex_stationarity_gap: float
    iterations: int
    converged: bool
    common_descent: bool
    scientific_status: str
    solution_source: str
    crosscheck: dict[str, Any]
    cosine_duality_gap: float
    kkt: dict[str, Any]
    gram_audit: dict[str, Any]
    gram: np.ndarray


def solve_max_min_common_descent(
    gradients: np.ndarray,
    *,
    go_tolerance: float = CONE_GO_TOL,
    solver_gap_tolerance: float = CONE_SOLVER_GAP_TOL,
    max_iterations: int = CONE_SOLVER_MAX_ITERATIONS,
) -> ConeCertificate:
    """Solve the normalized max-min cone through its simplex dual.

    With unit gradient rows ``u_i``, minimax duality gives

      max_{||d||<=1} min_i u_i.d = min_{p in simplex} ||sum_i p_i u_i||.

    The simplex quadratic is solved by deterministic Frank-Wolfe.  The initial
    vertex and every linear-minimization tie use the lowest row index; the line
    search is the exact minimizer of the quadratic along the FW segment.
    """

    value = np.asarray(gradients, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("gradients must be a nonempty 2D matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError("gradient matrix contains non-finite values")
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms <= GRADIENT_ZERO_NORM_TOL):
        bad = np.flatnonzero(norms <= GRADIENT_ZERO_NORM_TOL).tolist()
        raise ValueError(f"zero-norm normalized-cone blocks: {bad}")
    units = value / norms[:, None]
    raw_gram = units @ units.T
    symmetry_error = float(np.max(np.abs(raw_gram - raw_gram.T)))
    diagonal_error = float(np.max(np.abs(np.diag(raw_gram) - 1.0)))
    symmetric_gram = 0.5 * (raw_gram + raw_gram.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric_gram)))
    gram_audit = {
        "raw_symmetry_max_abs": symmetry_error,
        "raw_diagonal_unit_max_abs_error": diagonal_error,
        "symmetric_minimum_eigenvalue": minimum_eigenvalue,
        "symmetry_tolerance": 1.0e-12,
        "diagonal_tolerance": 1.0e-12,
        "PSD_negative_tolerance": 1.0e-10,
    }
    if symmetry_error > 1.0e-12:
        raise ValueError("normalized gradient Gram is not symmetric")
    if diagonal_error > 1.0e-12:
        raise ValueError("normalized gradient Gram diagonal is not unit")
    if minimum_eigenvalue < -1.0e-10:
        raise ValueError("normalized gradient Gram is not positive semidefinite")
    gram = symmetric_gram
    count = value.shape[0]
    weights = np.zeros(count, dtype=np.float64)
    weights[0] = 1.0
    gap = math.inf
    iterations = 0
    converged = False
    for iteration in range(max_iterations):
        gradient = gram @ weights
        vertex = int(np.argmin(gradient))
        direction = -weights
        direction[vertex] += 1.0
        gap = float(weights @ gradient - gradient[vertex])
        iterations = iteration + 1
        if gap <= solver_gap_tolerance:
            converged = True
            break
        curvature = float(direction @ gram @ direction)
        if curvature <= 0.0:
            step = 1.0
        else:
            step = min(1.0, max(0.0, gap / curvature))
        weights += step * direction
        weights = np.maximum(weights, 0.0)
        weights /= float(np.sum(weights))
    fw_weights = weights.copy()
    fw_converged = converged
    fw_gradient = gram @ fw_weights
    fw_gap = float(fw_weights @ fw_gradient - np.min(fw_gradient))
    uniform = np.full(count, 1.0 / count, dtype=np.float64)
    slsqp = minimize(
        fun=lambda p: 0.5 * float(p @ gram @ p),
        x0=uniform,
        jac=lambda p: gram @ p,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={
            "type": "eq",
            "fun": lambda p: float(np.sum(p) - 1.0),
            "jac": lambda p: np.ones(count, dtype=np.float64),
        },
        options={"ftol": 1.0e-14, "maxiter": 20_000, "disp": False},
    )
    slsqp_x = np.asarray(getattr(slsqp, "x", np.asarray([])), dtype=np.float64)
    slsqp_feasible = bool(
        slsqp_x.shape == (count,)
        and np.all(np.isfinite(slsqp_x))
        and np.min(slsqp_x) >= -1.0e-10
        and abs(float(np.sum(slsqp_x)) - 1.0) <= 1.0e-8
    )
    slsqp_weights = (
        np.maximum(slsqp_x, 0.0)
        if slsqp_x.shape == (count,)
        else np.full(count, math.nan, dtype=np.float64)
    )
    if slsqp_feasible:
        slsqp_weights /= float(np.sum(slsqp_weights))
    fw_objective = 0.5 * float(fw_weights @ gram @ fw_weights)
    slsqp_objective = (
        0.5 * float(slsqp_weights @ gram @ slsqp_weights)
        if slsqp_feasible
        else math.inf
    )
    if slsqp_objective + 1.0e-15 < fw_objective:
        weights = slsqp_weights
        source = "SLSQP_crosscheck_lower_feasible_objective"
        selected_iterations = int(getattr(slsqp, "nit", 0))
        selected_solver_reported_success = bool(slsqp.success)
    else:
        weights = fw_weights
        source = "deterministic_Frank_Wolfe"
        selected_iterations = iterations
        selected_solver_reported_success = fw_converged
    selected_gradient = gram @ weights
    gap = float(weights @ selected_gradient - np.min(selected_gradient))
    selected_converged = bool(
        selected_solver_reported_success and gap <= solver_gap_tolerance
    )
    combination = units.T @ weights
    dual_norm = float(np.linalg.norm(combination))
    if dual_norm <= GRADIENT_ZERO_NORM_TOL:
        common_direction = None
        cosines = np.zeros(count, dtype=np.float64)
        primal = 0.0
    else:
        common_direction = combination / dual_norm
        cosines = units @ common_direction
        primal = float(np.min(cosines))
    norm_squared = dual_norm**2
    slack = selected_gradient - norm_squared
    active_mask = weights > 1.0e-10
    active_slack_max_abs = (
        float(np.max(np.abs(slack[active_mask]))) if np.any(active_mask) else math.inf
    )
    inactive_slack_min = (
        float(np.min(slack[~active_mask])) if np.any(~active_mask) else math.inf
    )
    simplex_sum_residual = abs(float(np.sum(weights)) - 1.0)
    minimum_weight = float(np.min(weights))
    kkt_max_violation = max(
        active_slack_max_abs,
        max(0.0, -inactive_slack_min) if math.isfinite(inactive_slack_min) else 0.0,
        simplex_sum_residual,
        max(0.0, -minimum_weight),
    )
    kkt = {
        "definition": "K*lambda - ||U^T lambda||^2 * 1",
        "active_weight_tolerance": 1.0e-10,
        "active_slack_max_abs": active_slack_max_abs,
        "inactive_slack_min": inactive_slack_min,
        "simplex_sum_residual": simplex_sum_residual,
        "minimum_weight": minimum_weight,
        "maximum_violation": kkt_max_violation,
        "certificate_tolerance": go_tolerance,
    }
    cosine_duality_gap = float(dual_norm - primal)
    if cosine_duality_gap < -1.0e-8:
        raise FloatingPointError("cone primal value exceeds its dual upper bound")
    cosine_duality_gap = max(0.0, cosine_duality_gap)
    certificate_numeric_pass = bool(
        cosine_duality_gap <= go_tolerance and kkt_max_violation <= go_tolerance
    )
    if certificate_numeric_pass and primal > go_tolerance:
        status = "GO"
    elif dual_norm <= go_tolerance:
        # Any feasible simplex point provides an upper bound on the optimum.
        # Thus dual_norm<=tol is a reliable NO_GO even if an iterative solver
        # has not completely minimized the dual objective.
        status = "NO_GO"
    else:
        status = "NUMERICAL_INCONCLUSIVE"
    common = status == "GO"
    return ConeCertificate(
        weights=weights,
        direction=common_direction,
        cosines=cosines,
        dual_norm=dual_norm,
        primal_min_cosine=primal,
        selected_simplex_stationarity_gap=float(gap),
        iterations=selected_iterations,
        converged=selected_converged,
        common_descent=common,
        scientific_status=status,
        solution_source=source,
        crosscheck={
            "Frank_Wolfe_objective": fw_objective,
            "Frank_Wolfe_gap": fw_gap,
            "Frank_Wolfe_iterations": iterations,
            "Frank_Wolfe_converged": fw_converged,
            "SLSQP_success": bool(slsqp.success),
            "SLSQP_status": int(slsqp.status),
            "SLSQP_message": str(slsqp.message),
            "SLSQP_feasible": slsqp_feasible,
            "SLSQP_objective": None
            if not math.isfinite(slsqp_objective)
            else slsqp_objective,
            "SLSQP_iterations": int(getattr(slsqp, "nit", 0)),
            "selected_solution": source,
            "selected_solver_reported_success": selected_solver_reported_success,
            "selected_simplex_stationarity_gap": gap,
            "KKT_certificate_tolerance": go_tolerance,
            "KKT_certificate_pass": certificate_numeric_pass,
        },
        cosine_duality_gap=cosine_duality_gap,
        kkt=kkt,
        gram_audit=gram_audit,
        gram=gram,
    )


def _certificate_payload(
    certificate: ConeCertificate,
    block_records: Sequence[dict],
    gradient_matrix: np.ndarray,
    *,
    authorization_role: str,
) -> dict:
    if len(block_records) != gradient_matrix.shape[0]:
        raise ValueError("block records and gradients differ")
    active = np.flatnonzero(certificate.weights > 1.0e-10)
    per_block = []
    norms = np.linalg.norm(gradient_matrix, axis=1)
    for index, record in enumerate(block_records):
        per_block.append(
            {
                **record,
                "gradient_norm": float(norms[index]),
                "common_direction_cosine": float(certificate.cosines[index]),
                "dual_weight": float(certificate.weights[index]),
            }
        )
    direction_sha = None
    direction_norm = None
    if certificate.direction is not None:
        direction_sha = _array_sha256(certificate.direction)
        direction_norm = float(np.linalg.norm(certificate.direction))
    return {
        "format": PRIMARY_FORMAT,
        "authorization_role": authorization_role,
        "can_authorize_parameter_update_or_training": False,
        "can_authorize_postcore_or_tail": False,
        "scientific_scope": "local_first_order_gradient_feasibility_only",
        "protocol_sha256": PROTOCOL_SHA256,
        "block_count": int(gradient_matrix.shape[0]),
        "parameter_count": int(gradient_matrix.shape[1]),
        "gradient_matrix_sha256": _array_sha256(gradient_matrix),
        "normalized_gram_sha256": _array_sha256(certificate.gram),
        "direction_sha256": direction_sha,
        "direction_norm": direction_norm,
        "direction_semantics": PROTOCOL["direction_semantics"],
        "solver": {
            "algorithm": PROTOCOL["cone_solver"],
            "gap_tolerance": CONE_SOLVER_GAP_TOL,
            "GO_tolerance": CONE_GO_TOL,
            "iterations": certificate.iterations,
            "converged": certificate.converged,
            "selected_simplex_stationarity_gap": (
                certificate.selected_simplex_stationarity_gap
            ),
            "dual_minimum_norm": certificate.dual_norm,
            "primal_minimum_cosine": certificate.primal_min_cosine,
            "duality_abs_difference": abs(
                certificate.dual_norm - certificate.primal_min_cosine
            ),
            "cosine_duality_gap": certificate.cosine_duality_gap,
            "KKT": certificate.kkt,
            "Gram_audit_before_symmetrization": certificate.gram_audit,
            "common_descent_pass": certificate.common_descent,
            "scientific_status": certificate.scientific_status,
            "solution_source": certificate.solution_source,
            "crosscheck": certificate.crosscheck,
            "active_dual_indices": active.tolist(),
        },
        "per_block": per_block,
    }


@dataclass
class LoadedTrainingInputs:
    data_root: Path
    formal_run_root: Path
    data_manifest: dict
    gate: dict
    bundle_path: Path
    model: torch.nn.Module
    reference6: Any
    reference8: Any
    thermal: list[Any]
    small_zero: list[Any]
    input_sha256: dict[str, str]
    wrapper_sha256: str
    formal_terminal_receipt: dict[str, Any]


def validate_formal_failed_run_terminal_receipt(run_root: Path) -> dict:
    """Bind the immutable failed formal gate before any training artifact opens."""

    run_root = reject_forbidden_path(
        run_root, "formal terminal receipt root", must_exist=True
    )
    if not run_root.is_dir():
        raise NotADirectoryError(run_root)
    forbidden = (
        "RUNNING",
        "FAILED",
        "PASSED",
        "CORE_GATE_PASSED",
        "SMOKE_PASS",
        "SMOKE_FAILED",
    )
    present_forbidden = [
        name
        for name in forbidden
        if (run_root / name).exists() or (run_root / name).is_symlink()
    ]
    if present_forbidden:
        raise ValueError(
            f"formal failed run has incompatible terminal markers: {present_forbidden}"
        )
    required = {
        name: _require_file(run_root, name, f"formal terminal marker {name}")
        for name in (
            "EXIT_CODE",
            "DONE",
            "PRETRAIN_DONE",
            "TRAINING_DONE",
            "CORE_GATE_FAILED",
            "COMPLETED_AT",
            "core_checkpoint_gate.json",
        )
    }
    if required["EXIT_CODE"].read_bytes() != b"0\n":
        raise ValueError("formal R2O terminal receipt requires exact EXIT_CODE=0")
    if required["DONE"].read_bytes() != EXPECTED_FORMAL_DONE_BYTES:
        raise ValueError("formal R2O DONE content differs from failed gate status")
    for name in ("PRETRAIN_DONE", "TRAINING_DONE", "CORE_GATE_FAILED"):
        if required[name].read_bytes() != b"":
            raise ValueError(f"formal R2O marker {name} must be empty")
    # Failed/passed must be an XOR, and this audit accepts only the failed arm.
    if not required["CORE_GATE_FAILED"].is_file() or (
        run_root / "CORE_GATE_PASSED"
    ).exists():
        raise ValueError("formal R2O gate marker XOR is not the failed arm")
    gate_hash = sha256(required["core_checkpoint_gate.json"])
    if gate_hash != EXPECTED_FORMAL_GATE_SHA256:
        raise ValueError(
            "formal gate SHA-256 differs from the frozen failed R2O receipt"
        )
    completed = required["COMPLETED_AT"].read_text(encoding="ascii")
    if not completed.endswith("\n") or len(completed.strip()) < 20:
        raise ValueError("formal COMPLETED_AT marker is malformed")
    return {
        "status": "immutable_R2O_formal_failed_terminal_receipt",
        "gate_sha256": gate_hash,
        "expected_gate_sha256": EXPECTED_FORMAL_GATE_SHA256,
        "EXIT_CODE_bytes_sha256": sha256(required["EXIT_CODE"]),
        "DONE_bytes_sha256": sha256(required["DONE"]),
        "COMPLETED_AT_bytes_sha256": sha256(required["COMPLETED_AT"]),
        "required_empty_markers": [
            "PRETRAIN_DONE",
            "TRAINING_DONE",
            "CORE_GATE_FAILED",
        ],
        "forbidden_markers_absent": list(forbidden),
        "gate_marker_arm": "CORE_GATE_FAILED",
    }


def _validate_direct_training_paths(data_root: Path, run_root: Path) -> dict[str, Path]:
    paths = {
        "manifest": _require_file(data_root, "manifest.json", "data manifest"),
        "thermal_train": _require_file(
            data_root, "train_thermal.xyz", "thermal training data"
        ),
        "harmonic_zero_train": _require_file(
            data_root,
            "train_harmonic_lambda1_small_zero.xyz",
            "small-zero training data",
        ),
        "reference_6x6": _require_file(
            data_root, "reference_6x6.xyz", "6x6 reference"
        ),
        "reference_8x8": _require_file(
            data_root, "reference_8x8.xyz", "8x8 reference"
        ),
        "gate": _require_file(run_root, "core_checkpoint_gate.json", "R2O gate"),
        "training_freeze": _require_file(
            run_root, "training_freeze.json", "R2O training freeze"
        ),
        "stage2_contract": _require_file(
            run_root, "stage2/stage2_contract.json", "R2O stage2 contract"
        ),
        "stage2_runtime": _require_file(
            run_root, "stage2/stage2_runtime.json", "R2O stage2 runtime"
        ),
        "bundle": _require_file(
            run_root, "selected_or_diagnostic_bundle.pt", "R2O diagnostic bundle"
        ),
        "wrapper_snapshot": _require_file(
            run_root,
            "code_snapshots/graphene_r2o_taylor_null.py",
            "R2O wrapper snapshot",
        ),
    }
    return paths


def _validate_training_structures(thermal: list[Any], small_zero: list[Any]) -> None:
    roles = Counter(str(item.info.get("config_type")) for item in thermal)
    expected = Counter(
        {
            "r2o_exact_e50_seed0_train": 20,
            "r2o_auxiliary_T300_train": 36,
            "r2o_auxiliary_T600_train": 36,
        }
    )
    if roles != expected:
        raise ValueError(f"thermal training roles changed: {roles}")
    if len(small_zero) != 32 or any(len(item) != 128 for item in small_zero):
        raise ValueError("R2P requires exactly 32 128-atom small-zero anchors")
    if any(
        str(item.info.get("config_type"))
        != "r2o_harmonic_lambda1_small_zero_train"
        for item in small_zero
    ):
        raise ValueError("small-zero role changed")
    if any(
        float(np.max(np.abs(np.asarray(item.arrays["REF_forces"], float))))
        > 1.0e-14
        for item in small_zero
    ):
        raise ValueError("small-zero training targets are no longer exact zero")
    for item in thermal:
        role = str(item.info["config_type"])
        if role == "r2o_exact_e50_seed0_train":
            if int(item.info.get("trajectory_seed", -1)) != 0:
                raise ValueError("E50 training block is not seed0")
            for key in (
                "APRIME_mode_real",
                "APRIME_mode_imag",
                "REF_forces",
            ):
                if key not in item.arrays:
                    raise ValueError(f"E50 seed0 missing {key}")
        if "seed2" in str(item.info).lower() or "support" in str(item.info).lower():
            raise ValueError("training structure metadata contains forbidden scope")


def load_training_inputs(
    data_root: Path,
    formal_run_root: Path,
    *,
    device: torch.device,
) -> LoadedTrainingInputs:
    """Load only train/reference artifacts; report-only files remain unopened."""

    assert_formal_graph_source_semantics()
    data_root = reject_forbidden_path(data_root, "data root", must_exist=True)
    formal_run_root = reject_forbidden_path(
        formal_run_root, "formal run root", must_exist=True
    )
    if not data_root.is_dir() or not formal_run_root.is_dir():
        raise NotADirectoryError("R2P roots must be directories")
    terminal_receipt = validate_formal_failed_run_terminal_receipt(formal_run_root)
    paths = _validate_direct_training_paths(data_root, formal_run_root)
    digests = {name: sha256(path) for name, path in paths.items()}
    if digests["gate"] != terminal_receipt["gate_sha256"]:
        raise RuntimeError("formal gate changed after terminal receipt validation")
    manifest = strict_json_load(paths["manifest"])
    gate = strict_json_load(paths["gate"])
    freeze = strict_json_load(paths["training_freeze"])
    contract = strict_json_load(paths["stage2_contract"])
    runtime = strict_json_load(paths["stage2_runtime"])
    for name in (
        "manifest",
        "gate",
        "training_freeze",
        "stage2_contract",
        "stage2_runtime",
    ):
        if sha256(paths[name]) != digests[name]:
            raise RuntimeError(f"{name} changed while R2P loaded its receipt")

    if gate.get("status") != EXPECTED_GATE_STATUS:
        raise ValueError("R2P audit requires the frozen failed R2O formal gate")
    if gate.get("postcore_or_deployment_authorized") is not False:
        raise ValueError("failed R2O diagnostic unexpectedly authorizes deployment")
    if gate.get("seed2_or_support_read") is not False:
        raise ValueError("R2O gate reports forbidden data read")
    if gate.get("bundle", {}).get("kind") != EXPECTED_BUNDLE_KIND:
        raise ValueError("R2P requires the non-deployable diagnostic bundle")
    if gate["bundle"].get("sha256") != digests["bundle"]:
        raise ValueError("diagnostic bundle hash differs from gate")
    if gate.get("inputs", {}).get("training_freeze_sha256") != digests[
        "training_freeze"
    ]:
        raise ValueError("training freeze hash differs from gate")
    if gate.get("inputs", {}).get("stage2_contract_sha256") != digests[
        "stage2_contract"
    ]:
        raise ValueError("stage2 contract hash differs from gate")
    if gate.get("inputs", {}).get("stage2_runtime_sha256") != digests[
        "stage2_runtime"
    ]:
        raise ValueError("stage2 runtime hash differs from gate")
    if gate.get("inputs", {}).get("data_manifest_sha256") != digests["manifest"]:
        raise ValueError("data manifest hash differs from gate")
    if freeze.get("forbidden") != {
        "seed1_in_gradients_or_scales": False,
        "seed2_opened": False,
        "support_opened": False,
    }:
        raise ValueError("training freeze forbidden-data statement changed")
    if contract.get("selection_data_read") is not False or contract.get(
        "seed1_seed2_support_read"
    ) is not False:
        raise ValueError("stage2 contract reports held data use")
    if runtime.get("seed1_seed2_support_read") is not False:
        raise ValueError("stage2 runtime reports held data use")
    if runtime.get("ema_state_sha256") != EXPECTED_MODEL_STATE_SHA256:
        raise ValueError("R2P is bound to the epoch240 EMA state")
    choice = gate.get("checkpoint_choice", {})
    if choice.get("label") != "epoch240" or choice.get(
        "model_state_sha256"
    ) != EXPECTED_MODEL_STATE_SHA256:
        raise ValueError("gate did not choose the fixed epoch240 EMA diagnostic")

    outputs = manifest.get("outputs", {})
    output_binding = {
        "train_thermal.xyz": "thermal_train",
        "train_harmonic_lambda1_small_zero.xyz": "harmonic_zero_train",
        "reference_6x6.xyz": "reference_6x6",
        "reference_8x8.xyz": "reference_8x8",
    }
    for key, local in output_binding.items():
        if outputs.get(key, {}).get("sha256") != digests[local]:
            raise ValueError(f"manifest hash differs for {key}")
    counts = manifest.get("counts", {})
    if counts.get("gradient_train_thermal_only") != 92 or counts.get(
        "stage2_gradient_harmonic_lambda1_small_zero"
    ) != 32:
        raise ValueError("R2P training counts differ from frozen manifest")
    if counts.get("seed2_opened") != 0 or counts.get("support_opened") != 0:
        raise ValueError("manifest reports forbidden data open")
    wrapper_hash = sha256(Path(__file__).with_name("graphene_r2o_taylor_null.py"))
    if wrapper_hash != digests["wrapper_snapshot"] or wrapper_hash != gate.get(
        "inputs", {}
    ).get("source_sha256", {}).get("wrapper"):
        raise ValueError("live and frozen R2O wrappers differ")

    # Only now open train/reference extxyz files.  seed1 and small validation
    # are intentionally absent from this function.
    thermal = read(paths["thermal_train"], index=":")
    small_zero = read(paths["harmonic_zero_train"], index=":")
    reference6 = read(paths["reference_6x6"], index=0)
    reference8 = read(paths["reference_8x8"], index=0)
    for name in (
        "thermal_train",
        "harmonic_zero_train",
        "reference_6x6",
        "reference_8x8",
    ):
        if sha256(paths[name]) != digests[name]:
            raise RuntimeError(f"{name} changed while R2P loaded its data")
    _validate_training_structures(thermal, small_zero)
    if len(reference6) != 72 or len(reference8) != 128:
        raise ValueError("R2P pristine reference atom counts changed")

    bundle = torch_load(paths["bundle"], map_location="cpu")
    if sha256(paths["bundle"]) != digests["bundle"]:
        raise RuntimeError("diagnostic bundle changed while R2P loaded it")
    if bundle.get("format") != EXPECTED_BUNDLE_FORMAT:
        raise ValueError("wrong R2O bundle format")
    if bundle.get("raw_MACE_must_not_be_deployed_without_Taylor_wrapper") is not True:
        raise ValueError("bundle lost raw-MACE deployment guard")
    metadata = bundle.get("metadata", {})
    if metadata.get("kind") != EXPECTED_BUNDLE_KIND:
        raise ValueError("bundle metadata is not diagnostic")
    if metadata.get("model_state_sha256") != EXPECTED_MODEL_STATE_SHA256:
        raise ValueError("bundle metadata state differs")
    if metadata.get("raw_MACE_direct_deployment_forbidden") is not True:
        raise ValueError("bundle metadata lost raw-MACE guard")
    if metadata.get("data_manifest_sha256") != digests["manifest"]:
        raise ValueError("bundle data manifest differs")
    if metadata.get("reference_6x6_sha256") != digests["reference_6x6"] or metadata.get(
        "reference_8x8_sha256"
    ) != digests["reference_8x8"]:
        raise ValueError("bundle reference hashes differ")
    model = bundle.get("model")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("diagnostic bundle contains no model")
    if state_dict_sha256(model) != EXPECTED_MODEL_STATE_SHA256:
        raise ValueError("loaded EMA model state differs")
    validate_mace_architecture(model)
    if model_dtype(model) != torch.float64:
        raise ValueError("R2P gradient audit is FP64 only")
    model = model.to(device=device, dtype=torch.float64).eval()
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("frozen R2O bundle must retain trainable parameter flags")
    for parameter in model.parameters():
        if parameter.grad is not None:
            raise ValueError("model entered R2P audit with populated .grad")
    return LoadedTrainingInputs(
        data_root=data_root,
        formal_run_root=formal_run_root,
        data_manifest=manifest,
        gate=gate,
        bundle_path=paths["bundle"],
        model=model,
        reference6=reference6,
        reference8=reference8,
        thermal=thermal,
        small_zero=small_zero,
        input_sha256=digests,
        wrapper_sha256=wrapper_hash,
        formal_terminal_receipt=terminal_receipt,
    )


@dataclass(frozen=True)
class GradientBlock:
    block_id: str
    family: str
    role: str
    objective: float
    gradient: np.ndarray
    metadata: dict[str, Any]


def _flatten_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> np.ndarray:
    pieces = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        value = (
            torch.zeros_like(parameter)
            if gradient is None
            else gradient.detach()
        )
        pieces.append(value.reshape(-1).to(dtype=torch.float64, device="cpu"))
    result = torch.cat(pieces).numpy()
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite parameter gradient")
    return np.asarray(result, dtype=np.float64)


def _objective_gradient(
    objective: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> np.ndarray:
    if objective.dtype != torch.float64 or objective.numel() != 1:
        raise ValueError("R2P objectives must be scalar FP64 tensors")
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError("non-finite R2P objective")
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return _flatten_gradients(gradients, parameters)


def _thermal_role(structure: Any) -> str:
    config_type = str(structure.info.get("config_type"))
    mapping = {
        "r2o_exact_e50_seed0_train": "E50",
        "r2o_auxiliary_T300_train": "T300",
        "r2o_auxiliary_T600_train": "T600",
    }
    try:
        return mapping[config_type]
    except KeyError as error:
        raise ValueError(f"unknown R2P thermal role {config_type}") from error


def _live_source_force(
    model: torch.nn.Module,
    structure: Any,
    reference: Any,
    device: torch.device,
) -> torch.Tensor:
    assert_formal_graph_source_semantics()
    result, assignment = evaluate_structure(
        model,
        structure,
        reference,
        device=str(device),
        create_graph=True,
    )
    value = source_order_forces(result.forces_reference_order, assignment)
    if value.dtype != torch.float64 or value.shape != (len(structure), 3):
        raise ValueError("R2P source-order force has wrong dtype or shape")
    return value


def _complex_aprime_objective(
    error: torch.Tensor,
    structure: Any,
) -> tuple[torch.Tensor, dict]:
    mode_real = torch.as_tensor(
        np.asarray(structure.arrays["APRIME_mode_real"], float),
        dtype=torch.float64,
        device=error.device,
    )
    mode_imag = torch.as_tensor(
        np.asarray(structure.arrays["APRIME_mode_imag"], float),
        dtype=torch.float64,
        device=error.device,
    )
    real = torch.sum(mode_real * error)
    # np.vdot(mode_real+i*mode_imag,error) has the negative imaginary sign;
    # the squared magnitude is independent of that sign.
    imag = -torch.sum(mode_imag * error)
    raw = (real.square() + imag.square()) / (APRIME_SCALE_EV_A**2)
    objective = raw / THERMAL_ROLE_COUNTS["E50"]
    return objective, {
        "projection_real_meV_A": float(1000.0 * real.detach().cpu()),
        "projection_imag_meV_A": float(1000.0 * imag.detach().cpu()),
        "projection_abs_meV_A": float(
            1000.0 * torch.sqrt(real.square() + imag.square()).detach().cpu()
        ),
        "mode_norm": float(
            torch.sqrt(torch.sum(mode_real.square() + mode_imag.square()))
            .detach()
            .cpu()
        ),
        "normalization_eV_A": APRIME_SCALE_EV_A,
        "aggregate_coefficient": 1.0 / THERMAL_ROLE_COUNTS["E50"],
    }


def _smooth_log_mean_exp(
    error: torch.Tensor,
    *,
    tau: float = SMOOTH_TAU_EV_A,
) -> torch.Tensor:
    flat = error.reshape(-1)
    smooth_abs = torch.sqrt(flat.square() + 1.0e-24)
    return tau * (
        torch.logsumexp(smooth_abs / tau, dim=0) - math.log(flat.numel())
    )


def _exact_top_objective(
    force: torch.Tensor,
    *,
    scale_eV_A: float,
    aggregate_coefficient: float,
) -> tuple[torch.Tensor, dict]:
    flat = force.reshape(-1)
    detached = flat.detach().abs().cpu().numpy()
    order = np.argsort(-detached, kind="stable")
    top_index = int(order[0])
    if order.size < 2:
        raise ValueError("exact-top objective requires at least two force components")
    second_index = int(order[1])
    top_abs = float(detached[top_index])
    second_abs = float(detached[second_index])
    exact_active = np.flatnonzero(detached == top_abs)
    if len(exact_active) != 1:
        raise ValueError(
            "exact-top objective has a nonsmooth exact tie and cannot provide "
            f"one sufficient descent constraint: indices={exact_active.tolist()}"
        )
    nonactive_gap = float(top_abs - second_abs)
    if nonactive_gap <= TOP_NEAR_TIE_TOL_EV_A:
        raise ValueError(
            "exact-top active branch has a numerically unstable near tie: "
            f"gap={nonactive_gap:.17g} eV/A"
        )
    # Select the unique index from detached FP64 values, then construct the
    # exact squared maximum from the corresponding live tensor branch.
    top_squared = flat[top_index].square()
    objective = aggregate_coefficient * top_squared / (scale_eV_A**2)
    atom, component = divmod(top_index, 3)
    second_atom, second_component = divmod(second_index, 3)
    signed_top = float(flat[top_index].detach().cpu())
    signed_second = float(flat[second_index].detach().cpu())
    relative_gap = nonactive_gap / top_abs if top_abs > 0.0 else math.inf
    return objective, {
        "top_flat_index": top_index,
        "top_atom_source_order_0based": int(atom),
        "top_cartesian_component": "xyz"[component],
        "top_signed_eV_A": signed_top,
        "top_sign": int(np.sign(signed_top)),
        "top_abs_eV_A": top_abs,
        "second_flat_index": second_index,
        "second_atom_source_order_0based": int(second_atom),
        "second_cartesian_component": "xyz"[second_component],
        "second_signed_eV_A": signed_second,
        "second_sign": int(np.sign(signed_second)),
        "second_abs_eV_A": second_abs,
        "exact_active_indices": exact_active.tolist(),
        "exact_active_count": int(len(exact_active)),
        "nonactive_absolute_gap_eV_A": nonactive_gap,
        "nonactive_relative_gap": relative_gap,
        "exact_tie_tolerance_eV_A": TOP_EXACT_TIE_TOL_EV_A,
        "near_tie_fail_tolerance_eV_A": TOP_NEAR_TIE_TOL_EV_A,
        "exact_top_gradient_policy": (
            "unique detached FP64 argmax selects one live squared branch; "
            "exact and near ties fail closed"
        ),
        "normalization_eV_A": scale_eV_A,
        "aggregate_coefficient": aggregate_coefficient,
    }


def compute_primary_gradient_blocks(
    loaded: LoadedTrainingInputs,
    *,
    device: torch.device,
) -> tuple[list[GradientBlock], dict]:
    """Compute the 176 fixed primary blocks without any parameter update."""

    model = loaded.model
    named_parameters = list(model.named_parameters())
    parameters = [parameter for _, parameter in named_parameters]
    before = state_dict_sha256(model)
    if before != EXPECTED_MODEL_STATE_SHA256:
        raise ValueError("primary audit did not start from epoch240 EMA")
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("primary audit requires all .grad fields to be None")
    aprime: list[GradientBlock] = []
    thermal: list[GradientBlock] = []
    small_rms: list[GradientBlock] = []
    small_top: list[GradientBlock] = []
    family_sum = {
        family: np.zeros(sum(p.numel() for p in parameters), dtype=np.float64)
        for family in PRIMARY_FAMILY_COUNTS
    }
    started = time.monotonic()

    role_seen = Counter()
    for index, structure in enumerate(loaded.thermal):
        role = _thermal_role(structure)
        role_index = role_seen[role]
        role_seen[role] += 1
        reference = loaded.reference6 if len(structure) == 72 else loaded.reference8
        predicted = _live_source_force(model, structure, reference, device)
        target = torch.as_tensor(
            np.asarray(structure.arrays["REF_forces"], float),
            dtype=torch.float64,
            device=device,
        )
        error = predicted - target
        total_raw = torch.mean(error.square()) / (THERMAL_FORCE_SCALE_EV_A**2)
        coefficient = THERMAL_GROUP_MASS[role] / THERMAL_ROLE_COUNTS[role]
        total_objective = coefficient * total_raw
        base_metadata = {
            "source_file_index": index,
            "role_index": role_index,
            "atom_count": len(structure),
            "config_type": str(structure.info["config_type"]),
            "snapshot_index": int(structure.info.get("snapshot_index", -1)),
            "normalization_eV_A": THERMAL_FORCE_SCALE_EV_A,
            "aggregate_coefficient": coefficient,
            "unweighted_normalized_MSE": float(total_raw.detach().cpu()),
        }
        if role == "E50":
            a_objective, a_metadata = _complex_aprime_objective(error, structure)
            a_gradient = _objective_gradient(
                a_objective, parameters, retain_graph=True
            )
            block = GradientBlock(
                block_id=f"Aprime_seed0:{role_index:02d}",
                family="Aprime_seed0",
                role=role,
                objective=float(a_objective.detach().cpu()),
                gradient=a_gradient,
                metadata={**base_metadata, **a_metadata},
            )
            aprime.append(block)
            family_sum["Aprime_seed0"] += a_gradient
        total_gradient = _objective_gradient(
            total_objective, parameters, retain_graph=False
        )
        block = GradientBlock(
            block_id=f"thermal_total_force:{role}:{role_index:02d}",
            family="thermal_total_force",
            role=role,
            objective=float(total_objective.detach().cpu()),
            gradient=total_gradient,
            metadata=base_metadata,
        )
        thermal.append(block)
        family_sum["thermal_total_force"] += total_gradient
        del predicted, target, error, total_raw, total_objective

    for index, structure in enumerate(loaded.small_zero):
        predicted = _live_source_force(
            model, structure, loaded.reference8, device
        )
        rms_raw = torch.mean(predicted.square()) / (SMALL_RMS_SCALE_EV_A**2)
        rms_objective = rms_raw / len(loaded.small_zero)
        rms_gradient = _objective_gradient(
            rms_objective, parameters, retain_graph=True
        )
        common_metadata = {
            "source_file_index": index,
            "harmonic_train_source_index": int(
                structure.info["r2o_harmonic_train_source_index"]
            ),
            "atom_count": len(structure),
            "force_component_count": int(predicted.numel()),
            "bond_RMS_A": float(structure.info["r2o_bond_length_RMS_A"]),
        }
        rms_block = GradientBlock(
            block_id=f"small_zero_RMS:{index:02d}",
            family="small_zero_RMS",
            role="small_zero",
            objective=float(rms_objective.detach().cpu()),
            gradient=rms_gradient,
            metadata={
                **common_metadata,
                "normalization_eV_A": SMALL_RMS_SCALE_EV_A,
                "aggregate_coefficient": 1.0 / len(loaded.small_zero),
                "unweighted_normalized_MSE": float(rms_raw.detach().cpu()),
                "force_RMS_meV_A": float(
                    1000.0 * torch.sqrt(torch.mean(predicted.square())).detach().cpu()
                ),
            },
        )
        small_rms.append(rms_block)
        family_sum["small_zero_RMS"] += rms_gradient
        top_objective, top_metadata = _exact_top_objective(
            predicted,
            scale_eV_A=SMALL_CORRECTION_MARGIN_EV_A,
            aggregate_coefficient=1.0 / len(loaded.small_zero),
        )
        top_gradient = _objective_gradient(
            top_objective, parameters, retain_graph=False
        )
        top_block = GradientBlock(
            block_id=f"small_zero_exact_top:{index:02d}",
            family="small_zero_exact_top",
            role="small_zero",
            objective=float(top_objective.detach().cpu()),
            gradient=top_gradient,
            metadata={**common_metadata, **top_metadata},
        )
        small_top.append(top_block)
        family_sum["small_zero_exact_top"] += top_gradient
        del predicted, rms_raw, rms_objective, top_objective

    blocks = aprime + thermal + small_rms + small_top
    observed = Counter(block.family for block in blocks)
    if observed != Counter(PRIMARY_FAMILY_COUNTS):
        raise ValueError(f"primary block family count changed: {observed}")
    if [block.block_id for block in blocks] != list(
        dict.fromkeys(block.block_id for block in blocks)
    ):
        raise ValueError("primary block ids are not unique")
    after = state_dict_sha256(model)
    if after != before or any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("autograd-only primary audit changed model state or .grad")
    aggregate = {
        family: {
            "gradient_norm": float(np.linalg.norm(value)),
            "gradient_sha256": _array_sha256(value),
            "objective_sum": float(
                sum(block.objective for block in blocks if block.family == family)
            ),
        }
        for family, value in family_sum.items()
    }
    return blocks, {
        "wall_seconds": time.monotonic() - started,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_unchanged": after == before,
        "all_parameter_grad_fields_none": all(
            parameter.grad is None for parameter in parameters
        ),
        "aggregate_from_per_configuration_blocks": aggregate,
    }


def blocks_to_matrix_and_records(
    blocks: Sequence[GradientBlock],
) -> tuple[np.ndarray, list[dict]]:
    matrix = np.stack([block.gradient for block in blocks]).astype(
        np.float64, copy=False
    )
    records = [
        {
            "block_index": index,
            "block_id": block.block_id,
            "family": block.family,
            "role": block.role,
            "objective": block.objective,
            "metadata": block.metadata,
        }
        for index, block in enumerate(blocks)
    ]
    return matrix, records


def _validate_report_only_structures(
    seed1: list[Any], actual_small: list[Any]
) -> None:
    if len(seed1) != 20 or any(len(item) != 72 for item in seed1):
        raise ValueError("seed1 report must contain exactly 20 72-atom frames")
    if any(
        str(item.info.get("config_type")) != "r2o_exact_e50_seed1_gate"
        or int(item.info.get("trajectory_seed", -1)) != 1
        for item in seed1
    ):
        raise ValueError("seed1 report scope changed")
    if len(actual_small) != 12 or any(len(item) != 128 for item in actual_small):
        raise ValueError("actual small report must contain 12 128-atom frames")
    if any(
        str(item.info.get("config_type")) != "r2o_harmonic_lambda1_small_gate"
        for item in actual_small
    ):
        raise ValueError("actual small report scope changed")


def load_report_only_structures(
    loaded: LoadedTrainingInputs,
    *,
    primary_certificate_path: Path,
    expected_primary_certificate_sha256: str,
) -> tuple[list[Any], list[Any], dict[str, str]]:
    """Open held/report files only after a hash-bound primary exists."""

    primary_certificate_path = reject_forbidden_path(
        primary_certificate_path, "primary certificate", must_exist=True
    )
    if sha256(primary_certificate_path) != expected_primary_certificate_sha256:
        raise ValueError("primary certificate changed before report-only open")
    primary = strict_json_load(primary_certificate_path)
    if primary.get("format") != PRIMARY_FORMAT or primary.get(
        "authorization_role"
    ) != "primary_hard_exact_top":
        raise ValueError("wrong primary certificate before report-only open")
    seed1_path = _require_file(
        loaded.data_root, "valid_e50_seed1.xyz", "seed1 fixed report"
    )
    actual_path = _require_file(
        loaded.data_root,
        "harmonic_lambda1_small_gate.xyz",
        "actual combined small report",
    )
    hashes = {"seed1_report": sha256(seed1_path), "actual_small": sha256(actual_path)}
    outputs = loaded.data_manifest.get("outputs", {})
    if outputs.get("valid_e50_seed1.xyz", {}).get("sha256") != hashes[
        "seed1_report"
    ]:
        raise ValueError("seed1 report hash differs from manifest")
    if outputs.get("harmonic_lambda1_small_gate.xyz", {}).get("sha256") != hashes[
        "actual_small"
    ]:
        raise ValueError("actual small report hash differs from manifest")
    seed1 = read(seed1_path, index=":")
    actual = read(actual_path, index=":")
    if sha256(seed1_path) != hashes["seed1_report"] or sha256(actual_path) != hashes[
        "actual_small"
    ]:
        raise RuntimeError("report-only input changed while R2P loaded it")
    _validate_report_only_structures(seed1, actual)
    return seed1, actual, hashes


def _force_metrics(errors: Sequence[np.ndarray]) -> dict:
    value = np.concatenate([np.asarray(error).reshape(-1) for error in errors])
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(value * value))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(value))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(value))),
        "n_force_components": int(value.size),
    }


def compute_seed1_fixed_report(
    loaded: LoadedTrainingInputs,
    seed1: Sequence[Any],
    *,
    device: torch.device,
) -> dict:
    assert_formal_graph_source_semantics()
    errors = []
    projections = []
    rows = []
    for index, structure in enumerate(seed1):
        result, assignment = evaluate_structure(
            loaded.model,
            structure,
            loaded.reference6,
            device=str(device),
            create_graph=False,
        )
        predicted = (
            source_order_forces(result.forces_reference_order, assignment)
            .detach()
            .cpu()
            .numpy()
        )
        error = predicted - np.asarray(structure.arrays["REF_forces"], float)
        mode = np.asarray(structure.arrays["APRIME_mode_real"], float) + 1j * np.asarray(
            structure.arrays["APRIME_mode_imag"], float
        )
        projection = np.vdot(mode.reshape(-1), error.reshape(-1))
        errors.append(error)
        projections.append(projection)
        rows.append(
            {
                "source_file_index": index,
                "snapshot_index": int(structure.info["snapshot_index"]),
                "Aprime_error_real_meV_A": float(1000.0 * projection.real),
                "Aprime_error_imag_meV_A": float(1000.0 * projection.imag),
                "Aprime_error_abs_meV_A": float(1000.0 * abs(projection)),
            }
        )
    values = np.asarray(projections)
    return {
        "role": "fixed_report_only_never_differentiated_or_selected",
        "force_error": _force_metrics(errors),
        "Aprime_RMS_meV_A": float(1000.0 * np.sqrt(np.mean(np.abs(values) ** 2))),
        "per_configuration": rows,
    }


def compute_report_only_gradient_cones(
    loaded: LoadedTrainingInputs,
    actual_small: Sequence[Any],
    primary_blocks: Sequence[GradientBlock],
    *,
    device: torch.device,
) -> tuple[dict, dict]:
    """Compute non-authorizing smooth and actual-combined comparisons."""

    parameters = list(loaded.model.parameters())
    smooth_blocks: list[GradientBlock] = []
    actual_rms: list[GradientBlock] = []
    actual_top: list[GradientBlock] = []
    actual_errors: list[np.ndarray] = []
    actual_rows: list[dict] = []
    for index, structure in enumerate(loaded.small_zero):
        predicted = _live_source_force(
            loaded.model, structure, loaded.reference8, device
        )
        value = _smooth_log_mean_exp(predicted)
        objective = (value / SMALL_CORRECTION_MARGIN_EV_A).square() / len(
            loaded.small_zero
        )
        gradient = _objective_gradient(objective, parameters, retain_graph=False)
        smooth_blocks.append(
            GradientBlock(
                block_id=f"small_zero_shifted_LME_report:{index:02d}",
                family="small_zero_shifted_LME_report",
                role="small_zero",
                objective=float(objective.detach().cpu()),
                gradient=gradient,
                metadata={
                    "source_file_index": index,
                    "atom_count": len(structure),
                    "force_component_count": int(predicted.numel()),
                    "tau_eV_A": SMOOTH_TAU_EV_A,
                    "shifted_LME_eV_A": float(value.detach().cpu()),
                    "normalization_eV_A": SMALL_CORRECTION_MARGIN_EV_A,
                    "maximum_underestimate_bound_eV_A": SMOOTH_TAU_EV_A
                    * math.log(predicted.numel()),
                    "sufficient_for_exact_max_gate": False,
                },
            )
        )

    for index, structure in enumerate(actual_small):
        predicted = _live_source_force(
            loaded.model, structure, loaded.reference8, device
        )
        target = torch.as_tensor(
            np.asarray(structure.arrays["REF_forces"], float),
            dtype=torch.float64,
            device=device,
        )
        error = predicted - target
        detached_error = error.detach().cpu().numpy()
        actual_errors.append(detached_error)
        actual_rows.append(
            {
                "source_file_index": index,
                "harmonic_index": int(structure.info["r2o_harmonic_index"]),
                **_force_metrics([detached_error]),
            }
        )
        rms_raw = torch.mean(error.square()) / (SMALL_RMS_SCALE_EV_A**2)
        rms_objective = rms_raw / len(actual_small)
        rms_gradient = _objective_gradient(
            rms_objective, parameters, retain_graph=True
        )
        actual_rms.append(
            GradientBlock(
                block_id=f"actual_small_RMS_report:{index:02d}",
                family="actual_small_RMS_report",
                role="actual_small_validation",
                objective=float(rms_objective.detach().cpu()),
                gradient=rms_gradient,
                metadata={
                    "source_file_index": index,
                    "harmonic_index": int(structure.info["r2o_harmonic_index"]),
                    "normalization_eV_A": SMALL_RMS_SCALE_EV_A,
                    "report_only": True,
                },
            )
        )
        top_objective, top_metadata = _exact_top_objective(
            error,
            scale_eV_A=SMALL_GATE_MAX_EV_A,
            aggregate_coefficient=1.0 / len(actual_small),
        )
        top_gradient = _objective_gradient(
            top_objective, parameters, retain_graph=False
        )
        actual_top.append(
            GradientBlock(
                block_id=f"actual_small_exact_top_report:{index:02d}",
                family="actual_small_exact_top_report",
                role="actual_small_validation",
                objective=float(top_objective.detach().cpu()),
                gradient=top_gradient,
                metadata={**top_metadata, "report_only": True},
            )
        )

    fixed_prefix = [
        block
        for block in primary_blocks
        if block.family in {"Aprime_seed0", "thermal_total_force"}
    ]
    comparisons = {
        "shifted_LME_instead_of_exact_top": [
            block
            for block in primary_blocks
            if block.family != "small_zero_exact_top"
        ]
        + smooth_blocks,
        "actual_combined_small_0p5_10": fixed_prefix + actual_rms + actual_top,
    }
    output = {}
    for name, blocks in comparisons.items():
        matrix, records = blocks_to_matrix_and_records(blocks)
        certificate = solve_max_min_common_descent(matrix)
        payload = _certificate_payload(
            certificate,
            records,
            matrix,
            authorization_role=f"report_only_{name}",
        )
        payload["can_authorize_primary_direction"] = False
        output[name] = payload
    actual_fixed_report = {
        "role": "fixed_report_only_never_used_for_primary_direction_or_selection",
        "force_error": _force_metrics(actual_errors),
        "per_configuration_force": actual_rows,
    }
    return output, actual_fixed_report


def validate_formal_gate_prediction_parity(
    seed1_report: dict,
    actual_small_report: dict,
    checkpoint_choice: dict,
) -> dict:
    """Require report predictions to reproduce the frozen formal gate metrics."""

    expected_seed1 = checkpoint_choice.get("E50_seed1", {})
    expected_small = checkpoint_choice.get("harmonic_lambda1_small_gate", {})
    checks: list[dict] = []

    def add_numeric(name: str, observed: float, expected: float) -> None:
        difference = abs(float(observed) - float(expected))
        checks.append(
            {
                "name": name,
                "observed": float(observed),
                "expected": float(expected),
                "absolute_difference": difference,
                "tolerance": FORMAL_GATE_PARITY_TOL_MEV_A,
                "pass": difference <= FORMAL_GATE_PARITY_TOL_MEV_A,
            }
        )

    def add_force(prefix: str, observed: dict, expected: dict) -> None:
        for key in ("RMSE_meV_A", "MAE_meV_A", "max_abs_meV_A"):
            add_numeric(f"{prefix}.{key}", observed[key], expected[key])
        observed_count = int(observed["n_force_components"])
        expected_count = int(expected["n_force_components"])
        checks.append(
            {
                "name": f"{prefix}.n_force_components",
                "observed": observed_count,
                "expected": expected_count,
                "absolute_difference": abs(observed_count - expected_count),
                "tolerance": 0,
                "pass": observed_count == expected_count,
            }
        )

    add_force(
        "E50_seed1.force_error",
        seed1_report["force_error"],
        expected_seed1["force_error"],
    )
    add_numeric(
        "E50_seed1.Aprime_RMS_meV_A",
        seed1_report["Aprime_RMS_meV_A"],
        expected_seed1["Aprime_force_error_RMS_meV_A"],
    )
    add_force(
        "harmonic_lambda1_small_gate.force_error",
        actual_small_report["force_error"],
        expected_small["force_error"],
    )
    observed_rows = actual_small_report["per_configuration_force"]
    expected_rows = expected_small["per_configuration_force"]
    if len(observed_rows) != len(expected_rows):
        raise ValueError("formal small-gate parity row count changed")
    for index, (observed, expected) in enumerate(
        zip(observed_rows, expected_rows, strict=True)
    ):
        add_force(f"harmonic_small[{index}]", observed, expected)
    passed = all(item["pass"] for item in checks)
    payload = {
        "meaning": (
            "formal float32-origin graph followed by FP64 model/autograd must "
            "reproduce the checkpoint-gate predictions"
        ),
        "tolerance_meV_A": FORMAL_GATE_PARITY_TOL_MEV_A,
        "check_count": len(checks),
        "maximum_absolute_numeric_difference": max(
            float(item["absolute_difference"]) for item in checks
        ),
        "pass": passed,
        "checks": checks,
    }
    if not passed:
        raise ValueError("R2P prediction does not reproduce the frozen formal gate")
    return payload


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("xb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError as error:
        raise ValueError(f"artifact is outside run root: {path}") from error


def _prepare_output_root(output_root: Path) -> Path:
    output_root = reject_forbidden_path(
        output_root, "R2P output root", must_exist=False
    )
    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(output_root)
        if any(output_root.iterdir()):
            raise FileExistsError(
                f"R2P output must be a fresh empty directory: {output_root}"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    running = output_root / "RUNNING"
    descriptor = os.open(running, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(b"R2P zero-update gradient audit running\n")
        handle.flush()
        os.fsync(handle.fileno())
    return output_root


def _contract_payload(
    loaded: LoadedTrainingInputs,
    *,
    device: torch.device,
    source_snapshot_relative: str,
    source_snapshot_sha256: str,
) -> dict:
    schema = _state_schema(list(loaded.model.named_parameters()))
    return {
        "format": CONTRACT_FORMAT,
        "status": "frozen_before_any_R2P_parameter_gradient",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "authorization": {
            "optimizer_instantiated": False,
            "optimizer_step_called": False,
            "model_parameter_update_authorized": False,
            "support_or_seed2_read_authorized": False,
            "postcore_or_tail_authorized": False,
        },
        "input_sha256": loaded.input_sha256,
        "formal_failed_terminal_receipt": loaded.formal_terminal_receipt,
        "model_state_sha256": state_dict_sha256(loaded.model),
        "expected_model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
        "R2O_wrapper_sha256": loaded.wrapper_sha256,
        "formal_graph_source_semantics": FORMAL_GRAPH_SOURCE_SEMANTICS,
        "formal_graph_source_semantics_sha256": (
            FORMAL_GRAPH_SOURCE_SEMANTICS_SHA256
        ),
        "R2P_source_snapshot": {
            "relative_path": source_snapshot_relative,
            "sha256": source_snapshot_sha256,
        },
        "parameter_schema": schema,
        "runtime": runtime_fingerprint(device),
        "report_only_expected_from_manifest_not_opened_yet": {
            "valid_e50_seed1.xyz": loaded.data_manifest["outputs"][
                "valid_e50_seed1.xyz"
            ]["sha256"],
            "harmonic_lambda1_small_gate.xyz": loaded.data_manifest["outputs"][
                "harmonic_lambda1_small_gate.xyz"
            ]["sha256"],
        },
    }


def run_formal_audit(
    *,
    data_root: Path,
    formal_run_root: Path,
    output_root: Path,
    device: str = "cuda",
) -> dict:
    # Formal R2O did not switch the process default: AtomicData was sourced in
    # float32 and its continuous tensors were subsequently cast to FP64.  An
    # R2P audit must reproduce that graph, not silently build a native-FP64 one.
    assert_formal_graph_source_semantics()
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("R2P CUDA audit requested but CUDA is unavailable")
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("R2P supports only cpu or cuda")
    output = _prepare_output_root(output_root)
    started = time.monotonic()
    try:
        loaded = load_training_inputs(
            data_root, formal_run_root, device=selected_device
        )
        source = Path(__file__).resolve()
        snapshot = output / "graphene_r2p_gradient_feasibility_snapshot.py"
        _atomic_copy(source, snapshot)
        source_hash = sha256(snapshot)
        if source_hash != sha256(source):
            raise RuntimeError("R2P source snapshot copy changed bytes")
        contract_path = output / "audit_contract.json"
        contract = _contract_payload(
            loaded,
            device=selected_device,
            source_snapshot_relative=_relative_to_root(snapshot, output),
            source_snapshot_sha256=source_hash,
        )
        contract["contract_sha256"] = canonical_json_sha256(contract)
        strict_json(contract_path, contract)
        contract_file_hash = sha256(contract_path)

        if selected_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected_device)
        blocks, mechanics = compute_primary_gradient_blocks(
            loaded, device=selected_device
        )
        matrix, records = blocks_to_matrix_and_records(blocks)
        certificate = solve_max_min_common_descent(matrix)
        certificate_payload = _certificate_payload(
            certificate,
            records,
            matrix,
            authorization_role="primary_hard_exact_top",
        )
        certificate_payload.update(
            {
                "status": "primary_frozen_before_any_seed1_or_actual_small_open",
                "contract_file_sha256": contract_file_hash,
                "model_state_sha256_before": mechanics["state_sha256_before"],
                "model_state_sha256_after_primary": mechanics["state_sha256_after"],
                "state_unchanged": mechanics["state_unchanged"],
                "primary_family_counts": dict(Counter(b.family for b in blocks)),
                "aggregate_from_per_configuration_blocks": mechanics[
                    "aggregate_from_per_configuration_blocks"
                ],
                "primary_wall_seconds": mechanics["wall_seconds"],
                "seed1_opened_before_primary_freeze": False,
                "actual_combined_small_opened_before_primary_freeze": False,
                "optimizer_instantiated": False,
                "optimizer_step_called": False,
            }
        )
        gradient_path = output / "primary_gradient_matrix.npy"
        gram_path = output / "primary_normalized_gram.npy"
        _atomic_numpy(gradient_path, matrix.astype("<f8", copy=False))
        _atomic_numpy(gram_path, certificate.gram.astype("<f8", copy=False))
        direction_path = None
        if certificate.direction is not None:
            direction_path = output / "primary_common_direction.npy"
            _atomic_numpy(
                direction_path,
                certificate.direction.astype("<f8", copy=False),
            )
        certificate_payload["binary_artifacts"] = {
            "primary_gradient_matrix": {
                "relative_path": _relative_to_root(gradient_path, output),
                "sha256": sha256(gradient_path),
                "shape": list(matrix.shape),
                "dtype": "float64",
            },
            "primary_normalized_gram": {
                "relative_path": _relative_to_root(gram_path, output),
                "sha256": sha256(gram_path),
                "shape": list(certificate.gram.shape),
                "dtype": "float64",
            },
            "primary_common_direction": None
            if direction_path is None
            else {
                "relative_path": _relative_to_root(direction_path, output),
                "sha256": sha256(direction_path),
                "shape": list(certificate.direction.shape),
                "dtype": "float64",
            },
        }
        primary_path = output / "primary_certificate.json"
        strict_json(primary_path, certificate_payload)
        primary_hash = sha256(primary_path)
        _atomic_bytes(
            output / "PRIMARY_CERTIFICATE_FROZEN",
            (primary_hash + "\n").encode("ascii"),
        )

        # Held/report-only files are first opened here, after the immutable
        # primary certificate and its marker are both on disk.
        seed1, actual_small, report_hashes = load_report_only_structures(
            loaded,
            primary_certificate_path=primary_path,
            expected_primary_certificate_sha256=primary_hash,
        )
        seed1_report = compute_seed1_fixed_report(
            loaded, seed1, device=selected_device
        )
        comparison_cones, actual_small_report = compute_report_only_gradient_cones(
            loaded, actual_small, blocks, device=selected_device
        )
        formal_gate_prediction_parity = validate_formal_gate_prediction_parity(
            seed1_report,
            actual_small_report,
            loaded.gate["checkpoint_choice"],
        )
        after_all = state_dict_sha256(loaded.model)
        if after_all != EXPECTED_MODEL_STATE_SHA256 or any(
            parameter.grad is not None for parameter in loaded.model.parameters()
        ):
            raise RuntimeError("report-only audit changed model state or .grad")
        report_only = {
            "format": REPORT_FORMAT,
            "status": "report_only_complete_cannot_change_primary_certificate",
            "primary_certificate_sha256": primary_hash,
            "primary_certificate_rechecked_sha256": sha256(primary_path),
            "report_input_sha256": report_hashes,
            "seed1_fixed_report": seed1_report,
            "actual_small_fixed_report": actual_small_report,
            "formal_gate_prediction_parity": formal_gate_prediction_parity,
            "comparison_cones": comparison_cones,
            "smooth_Linf_warning": {
                "shifted_log_mean_exp_can_underestimate_exact_max": True,
                "tau_eV_A": SMOOTH_TAU_EV_A,
                "n_components": 384,
                "underestimate_upper_bound_eV_A": SMOOTH_TAU_EV_A
                * math.log(384),
                "can_authorize_primary": False,
            },
            "model_state_sha256_after_all_report_only_work": after_all,
            "optimizer_instantiated": False,
            "optimizer_step_called": False,
        }
        report_path = output / "report_only_comparisons.json"
        strict_json(report_path, report_only)

        result = {
            "format": FORMAT,
            "status": "R2P_zero_update_gradient_feasibility_complete",
            "scientific_primary_status": certificate.scientific_status,
            "primary_common_descent_pass": certificate.common_descent,
            "postcore_or_training_authorized": False,
            "interpretation": (
                "GO/NO_GO/NUMERICAL_INCONCLUSIVE applies only to local first-order "
                "gate-aligned gradient feasibility; it never authorizes training"
            ),
            "protocol_sha256": PROTOCOL_SHA256,
            "contract": {
                "relative_path": _relative_to_root(contract_path, output),
                "sha256": contract_file_hash,
            },
            "source_snapshot": {
                "relative_path": _relative_to_root(snapshot, output),
                "sha256": source_hash,
            },
            "primary_certificate": {
                "relative_path": _relative_to_root(primary_path, output),
                "sha256": primary_hash,
            },
            "report_only_comparisons": {
                "relative_path": _relative_to_root(report_path, output),
                "sha256": sha256(report_path),
            },
            "model_state_sha256_before": mechanics["state_sha256_before"],
            "model_state_sha256_after": after_all,
            "state_unchanged": after_all == mechanics["state_sha256_before"],
            "optimizer_instantiated": False,
            "optimizer_step_called": False,
            "forbidden": {
                "seed2_opened": False,
                "support_opened": False,
                "seed1_used_in_primary_direction_or_selection": False,
                "actual_small_used_in_primary_direction_or_selection": False,
                "parameter_update_performed": False,
            },
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_memory_MiB": None
            if selected_device.type != "cuda"
            else float(torch.cuda.max_memory_allocated(selected_device) / 2**20),
        }
        result_path = output / "result.json"
        strict_json(result_path, result)
        _atomic_bytes(output / "EXIT_CODE", b"0\n")
        (output / "RUNNING").unlink()
        _atomic_bytes(output / "DONE", b"R2P gradient feasibility complete\n")
        return result
    except BaseException:
        try:
            _atomic_bytes(output / "EXIT_CODE", b"1\n")
            _atomic_bytes(output / "FAILED", b"R2P gradient feasibility failed\n")
            (output / "RUNNING").unlink(missing_ok=True)
        except BaseException:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the FP64 zero-update R2P gradient-feasibility audit"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--formal-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_formal_audit(
        data_root=args.data_root,
        formal_run_root=args.formal_run_root,
        output_root=args.output_root,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
