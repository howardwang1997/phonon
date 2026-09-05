#!/usr/bin/env python3
"""Four-step, train-only R2Q trust-region repair and one held endpoint gate.

The run is deliberately bounded: exactly four accepted normalized-cone steps,
six fixed Armijo candidates per step, and no optimizer.  Held structures are
opened only after the fourth-step checkpoint and its complete train/reference
receipt have been atomically frozen.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ase.io import read


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import graphene_r2p_gradient_feasibility as r2p  # noqa: E402
import evaluate_graphene_r2o_taylor_null as r2o_evaluator  # noqa: E402
import graphene_r2o_taylor_null as r2o_wrapper  # noqa: E402
from evaluate_graphene_r2o_taylor_null import (  # noqa: E402
    endpoint_metrics,
    implementation_checks,
    raw_energy_parity,
    reference_null,
)
from graphene_r2o_taylor_null import (  # noqa: E402
    cutoff_c2_metrics,
    evaluate_structure,
    fixed_reference_graph,
    model_dtype,
    sha256,
    source_order_forces,
    state_dict_sha256,
    taylor_remainder_energy_and_forces,
    torch_load,
    validate_mace_architecture,
)


FORMAT = "graphene_r2q_four_step_trust_region_v1"
STEP_FORMAT = "graphene_r2q_accepted_step_v1"
ENDPOINT_FORMAT = "graphene_r2q_frozen_endpoint_v1"
HELD_FORMAT = "graphene_r2q_single_held_endpoint_gate_v1"

EXPECTED_CONTRACT_DOC_SHA256 = (
    "8b287433b1046d4c229d1d3bb12aeba6bb9d5b0180af2ebf7605802935f614f0"
)
EXPECTED_R2P_SOURCE_SHA256 = (
    "402e44ab26b70cf8f4e265fd729ad6d96b73733b0a7aeb9baf58d7043ef43371"
)
EXPECTED_R2P_RESULT_SHA256 = (
    "333eb1b84209816f8b3044d6c30960fc568fa3714f1087f1b82ec56646ffbced"
)
EXPECTED_R2P_PRIMARY_SHA256 = (
    "54bf9056d8ac90641875f2dbe997336ffe909c3babf9a17440d6c2237791842c"
)
EXPECTED_R2P_REPORT_SHA256 = (
    "9da429464ce47edc743e17479fd55f5350702eae7d3b1f476cd757603a1d1ed1"
)
EXPECTED_R2P_GRADIENT_FILE_SHA256 = (
    "566844a4dd4003b52560e1d2cb8ae1eb1d4b2e1011843b3e7aeefd9703885a38"
)
EXPECTED_R2P_DIRECTION_FILE_SHA256 = (
    "a6df2bc9395915d05739516e5bb7aba68bfdbcf6ff4673f3cdc837283e81d246"
)
EXPECTED_BASE_STATE_SHA256 = r2p.EXPECTED_MODEL_STATE_SHA256
EXPECTED_R2O_WRAPPER_SHA256 = (
    "0eedea5a59b8f717559feda97b2c2956dd6f274fe4963de10c214db53b4610e2"
)
EXPECTED_R2O_EVALUATOR_SHA256 = (
    "14cf2a160a9b0281c0c2149caec6d62f9a31e40a9c7af39ea3ea2e14956c79ce"
)
EXPECTED_STEP1_ETA0 = 0.21654915896171457

STEP_COUNT = 4
HALVING_COUNT = 6
GLOBAL_RELATIVE_TRUST_CAP = 1.0e-3
PER_TENSOR_RELATIVE_TRUST_CAP = 2.0e-2
ARMIJO_C1 = 0.1
ARMIJO_NUMERIC_EPS = 1.0e-12
UPDATE_REPLAY_EPS_MULTIPLIER = 16.0
BASE_OBJECTIVE_PARITY_TOL = 1.0e-10
STEP1_GRADIENT_NORM_REL_TOL = 1.0e-10
STEP1_UNIT_COSINE_FLOOR = 1.0 - 1.0e-10
STEP1_MIN_COSINE_ABS_TOL = 1.0e-10

REFERENCE_LIMITS = {
    "energy_abs_eV": 1.0e-10,
    "force_max_abs_eV_A": 1.0e-9,
    "Hessian_max_abs_eV_A2": 1.0e-7,
    "Hessian_symmetry_max_abs_eV_A2": 1.0e-7,
    "Hessian_ASR_row_sum_max_abs_eV_A2": 1.0e-7,
    "Gamma_K_frequency_drift_upper_bound_cm-1": 2.0,
}

IMPLEMENTATION_LIMITS = {
    "actual_raw_energy_parity_eV": 1.0e-10,
    "actual_raw_position_gradient_parity_eV_A": 1.0e-9,
    "nontrivial_probe_raw_energy_parity_eV": 1.0e-10,
    "nontrivial_probe_raw_position_gradient_parity_eV_A": 1.0e-9,
    "O3_proper_energy_eV": 1.0e-6,
    "O3_proper_force_eV_A": 1.0e-5,
    "O3_improper_energy_eV": 1.0e-6,
    "O3_improper_force_eV_A": 1.0e-5,
    "translation_energy_eV": 1.0e-9,
    "translation_force_eV_A": 1.0e-8,
    "permutation_energy_eV": 1.0e-6,
    "permutation_force_eV_A": 1.0e-5,
    "finite_difference_force_eV_A": 1.0e-5,
    "force_Hessian_finite_difference_eV_A2": 1.0e-5,
    "nonreference_Hessian_antisymmetry_eV_A2": 1.0e-7,
    "nonreference_Hessian_translation_ASR_eV_A2": 1.0e-7,
    "size_energy_eV": 1.0e-7,
    "size_force_eV_A": 1.0e-5,
}

PROTOCOL = {
    "format": FORMAT,
    "steps_exact": STEP_COUNT,
    "gradient_blocks_each_step": dict(r2p.PRIMARY_FAMILY_COUNTS),
    "candidate_halving_indices": list(range(HALVING_COUNT)),
    "eta0_formula": "1e-3 * ||theta_j||_2 / ||d_j||_2",
    "global_relative_L2_cap": GLOBAL_RELATIVE_TRUST_CAP,
    "per_tensor_relative_L2_guard": PER_TENSOR_RELATIVE_TRUST_CAP,
    "Armijo_c1": ARMIJO_C1,
    "Armijo_epsilon": "1e-12 * max(1, abs(L_i_base))",
    "candidate_exact_top": "max(abs(force))^2 / (32 * 0.00358418^2)",
    "candidate_top_tie_policy": "record_only_because_candidate_has_no_gradient",
    "gradient_base_top_tie_policy": "R2P_exact_and_near_tie_fail_closed",
    "held_policy": "open_once_only_after_step4_endpoint_artifact_is_frozen",
    "optimizer_instantiated": False,
    "optimizer_step_called": False,
    "DFT_calculations": 0,
}
PROTOCOL_SHA256 = r2p.canonical_json_sha256(PROTOCOL)


class TrainNoGo(RuntimeError):
    """A reliable train-only cone or fixed candidate grid has no continuation."""

    def __init__(self, step: int, reason: str, payload: dict[str, Any]):
        super().__init__(reason)
        self.step = int(step)
        self.reason = reason
        self.payload = payload


class NumericalInconclusive(RuntimeError):
    """A receipt, invariant, or numerical failure has no scientific meaning."""

    def __init__(self, step: int | None, reason: str, payload: dict[str, Any]):
        super().__init__(reason)
        self.step = None if step is None else int(step)
        self.reason = reason
        self.payload = payload


@dataclass(frozen=True)
class R2PArtifacts:
    root: Path
    result: dict
    primary: dict
    gradient_matrix: np.ndarray
    direction: np.ndarray
    source_snapshot_sha256: str
    input_sha256: dict[str, str]


@dataclass(frozen=True)
class CurrentCone:
    blocks: list[r2p.GradientBlock]
    matrix: np.ndarray
    records: list[dict]
    certificate: r2p.ConeCertificate
    state_sha256: str
    wall_seconds: float


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


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("xb") as handle:
        np.save(handle, np.asarray(array, dtype="<f8"), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    """Make diagnostic error details finite without altering science artifacts."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else f"nonfinite:{number!r}"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _validated_source_hashes() -> dict[str, str]:
    """Bind every live helper whose semantics enter an R2Q decision."""

    paths = {
        "R2P_gradient_and_cone": Path(r2p.__file__).resolve(),
        "R2O_Taylor_wrapper": Path(r2o_wrapper.__file__).resolve(),
        "R2O_endpoint_evaluator": Path(r2o_evaluator.__file__).resolve(),
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    expected = {
        "R2P_gradient_and_cone": EXPECTED_R2P_SOURCE_SHA256,
        "R2O_Taylor_wrapper": EXPECTED_R2O_WRAPPER_SHA256,
        "R2O_endpoint_evaluator": EXPECTED_R2O_EVALUATOR_SHA256,
    }
    if observed != expected:
        raise ValueError(
            f"R2Q live helper source differs from the reviewed contract: {observed}"
        )
    return observed


def _prepare_output_root(path: Path) -> Path:
    output = r2p.reject_forbidden_path(path, "R2Q output", must_exist=False)
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(output)
        if any(output.iterdir()):
            raise FileExistsError(f"R2Q output must be fresh and empty: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(output / "RUNNING", b"R2Q four-step run in progress\n")
    return output


def _require_exact_file(root: Path, name: str, label: str) -> Path:
    return r2p._require_file(root, name, label)


def validate_r2p_artifacts(root: Path) -> R2PArtifacts:
    """Validate the frozen R2P GO receipt without interpreting held metrics."""

    run = r2p.reject_forbidden_path(root, "R2P run root", must_exist=True)
    if not run.is_dir():
        raise NotADirectoryError(run)
    forbidden = ("RUNNING", "FAILED", "TRAIN_NO_GO", "NUMERICAL_INCONCLUSIVE")
    present = [name for name in forbidden if (run / name).exists()]
    if present:
        raise ValueError(f"R2P run has incompatible markers: {present}")
    files = {
        name: _require_exact_file(run, name, f"R2P {name}")
        for name in (
            "EXIT_CODE",
            "DONE",
            "PRIMARY_CERTIFICATE_FROZEN",
            "result.json",
            "primary_certificate.json",
            "report_only_comparisons.json",
            "primary_gradient_matrix.npy",
            "primary_common_direction.npy",
            "graphene_r2p_gradient_feasibility_snapshot.py",
        )
    }
    if files["EXIT_CODE"].read_bytes() != b"0\n":
        raise ValueError("R2P requires exact EXIT_CODE=0")
    if files["DONE"].read_bytes() != b"R2P gradient feasibility complete\n":
        raise ValueError("R2P DONE bytes changed")
    expected_hashes = {
        "result.json": EXPECTED_R2P_RESULT_SHA256,
        "primary_certificate.json": EXPECTED_R2P_PRIMARY_SHA256,
        "report_only_comparisons.json": EXPECTED_R2P_REPORT_SHA256,
        "primary_gradient_matrix.npy": EXPECTED_R2P_GRADIENT_FILE_SHA256,
        "primary_common_direction.npy": EXPECTED_R2P_DIRECTION_FILE_SHA256,
        "graphene_r2p_gradient_feasibility_snapshot.py": EXPECTED_R2P_SOURCE_SHA256,
    }
    observed = {name: sha256(files[name]) for name in expected_hashes}
    if observed != expected_hashes:
        raise ValueError(f"R2P frozen file hash mismatch: {observed}")
    if files["PRIMARY_CERTIFICATE_FROZEN"].read_bytes() != (
        EXPECTED_R2P_PRIMARY_SHA256 + "\n"
    ).encode("ascii"):
        raise ValueError("R2P primary marker does not bind the primary file")
    result = r2p.strict_json_load(files["result.json"])
    primary = r2p.strict_json_load(files["primary_certificate.json"])
    if result.get("status") != "R2P_zero_update_gradient_feasibility_complete":
        raise ValueError("R2P result is not complete")
    if result.get("scientific_primary_status") != "GO" or result.get(
        "primary_common_descent_pass"
    ) is not True:
        raise ValueError("R2Q requires the frozen R2P GO")
    if result.get("postcore_or_training_authorized") is not False:
        raise ValueError("R2P authorization boundary changed")
    if result.get("model_state_sha256_before") != EXPECTED_BASE_STATE_SHA256 or result.get(
        "model_state_sha256_after"
    ) != EXPECTED_BASE_STATE_SHA256:
        raise ValueError("R2P model state receipt changed")
    if result.get("source_snapshot", {}).get("sha256") != EXPECTED_R2P_SOURCE_SHA256:
        raise ValueError("R2P result source binding changed")
    if result.get("primary_certificate", {}).get("sha256") != EXPECTED_R2P_PRIMARY_SHA256:
        raise ValueError("R2P result primary binding changed")
    if result.get("report_only_comparisons", {}).get("sha256") != EXPECTED_R2P_REPORT_SHA256:
        raise ValueError("R2P result report binding changed")
    if primary.get("block_count") != 176 or primary.get(
        "primary_family_counts"
    ) != r2p.PRIMARY_FAMILY_COUNTS:
        raise ValueError("R2P primary block contract changed")
    if primary.get("state_unchanged") is not True or primary.get(
        "optimizer_instantiated"
    ) is not False or primary.get("optimizer_step_called") is not False:
        raise ValueError("R2P primary zero-update receipt changed")
    if primary.get("seed1_opened_before_primary_freeze") is not False or primary.get(
        "actual_combined_small_opened_before_primary_freeze"
    ) is not False:
        raise ValueError("R2P primary-before-held isolation receipt changed")
    aggregate = primary.get("aggregate_from_per_configuration_blocks")
    if not isinstance(aggregate, dict) or set(aggregate) != set(
        r2p.PRIMARY_FAMILY_COUNTS
    ):
        raise ValueError("R2P primary no longer certifies family aggregates")
    if primary.get("model_state_sha256_before") != EXPECTED_BASE_STATE_SHA256 or primary.get(
        "model_state_sha256_after_primary"
    ) != EXPECTED_BASE_STATE_SHA256:
        raise ValueError("R2P primary state binding changed")
    if primary.get("solver", {}).get("scientific_status") != "GO":
        raise ValueError("R2P primary solver is not GO")
    primary_rows = primary.get("per_block", [])
    if len(primary_rows) != 176 or len(
        {item.get("block_id") for item in primary_rows}
    ) != 176:
        raise ValueError("R2P primary block ids changed")

    gradient = np.load(files["primary_gradient_matrix.npy"], allow_pickle=False)
    direction = np.load(files["primary_common_direction.npy"], allow_pickle=False)
    if gradient.dtype != np.float64 or gradient.shape != (176, 42096):
        raise ValueError("R2P gradient matrix shape/dtype changed")
    if direction.dtype != np.float64 or direction.shape != (42096,):
        raise ValueError("R2P direction shape/dtype changed")
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(direction)):
        raise FloatingPointError("R2P binary artifact is non-finite")
    if abs(float(np.linalg.norm(direction)) - 1.0) > 1.0e-12:
        raise ValueError("R2P common direction is not unit norm")
    if r2p._array_sha256(gradient) != primary.get("gradient_matrix_sha256"):
        raise ValueError("R2P gradient semantic hash changed")
    if r2p._array_sha256(direction) != primary.get("direction_sha256"):
        raise ValueError("R2P direction semantic hash changed")
    observed_after_load = {name: sha256(files[name]) for name in expected_hashes}
    if observed_after_load != expected_hashes:
        raise RuntimeError("R2P frozen artifacts changed while they were loaded")
    binary = primary.get("binary_artifacts", {})
    if binary.get("primary_gradient_matrix", {}).get("relative_path") != (
        "primary_gradient_matrix.npy"
    ) or binary.get("primary_gradient_matrix", {}).get("sha256") != (
        EXPECTED_R2P_GRADIENT_FILE_SHA256
    ):
        raise ValueError("R2P gradient file receipt changed")
    if binary.get("primary_common_direction", {}).get("relative_path") != (
        "primary_common_direction.npy"
    ) or binary.get("primary_common_direction", {}).get("sha256") != (
        EXPECTED_R2P_DIRECTION_FILE_SHA256
    ):
        raise ValueError("R2P direction file receipt changed")
    live_source = Path(r2p.__file__).resolve()
    if sha256(live_source) != EXPECTED_R2P_SOURCE_SHA256:
        raise ValueError("live R2P source differs from the reviewed snapshot")
    return R2PArtifacts(
        root=run,
        result=result,
        primary=primary,
        gradient_matrix=np.asarray(gradient, dtype=np.float64),
        direction=np.asarray(direction, dtype=np.float64),
        source_snapshot_sha256=observed[
            "graphene_r2p_gradient_feasibility_snapshot.py"
        ],
        input_sha256=observed,
    )


def _source_force(
    model: torch.nn.Module,
    structure: Any,
    reference: Any,
    device: torch.device,
    *,
    create_graph: bool,
) -> torch.Tensor:
    r2p.assert_formal_graph_source_semantics()
    result, assignment = evaluate_structure(
        model,
        structure,
        reference,
        device=str(device),
        create_graph=create_graph,
    )
    force = source_order_forces(result.forces_reference_order, assignment)
    if force.dtype != torch.float64 or force.shape != (len(structure), 3):
        raise ValueError("R2Q force shape/dtype changed")
    return force


def _current_gradient_blocks(
    loaded: r2p.LoadedTrainingInputs,
    model: torch.nn.Module,
    device: torch.device,
) -> CurrentCone:
    """Recompute the exact 176 R2P blocks at an arbitrary current state."""

    started = time.monotonic()
    validate_mace_architecture(model)
    if model_dtype(model) != torch.float64:
        raise ValueError("R2Q is FP64 only")
    named = list(model.named_parameters())
    parameters = [parameter for _, parameter in named]
    if sum(parameter.numel() for parameter in parameters) != 42096:
        raise ValueError("R2Q parameter count changed")
    before = state_dict_sha256(model)
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("R2Q cone requires empty .grad fields")
    aprime: list[r2p.GradientBlock] = []
    thermal: list[r2p.GradientBlock] = []
    small_rms: list[r2p.GradientBlock] = []
    small_top: list[r2p.GradientBlock] = []
    role_seen: Counter[str] = Counter()

    for source_index, structure in enumerate(loaded.thermal):
        role = r2p._thermal_role(structure)
        role_index = role_seen[role]
        role_seen[role] += 1
        reference = loaded.reference6 if len(structure) == 72 else loaded.reference8
        predicted = _source_force(
            model, structure, reference, device, create_graph=True
        )
        target = torch.as_tensor(
            np.asarray(structure.arrays["REF_forces"], float),
            dtype=torch.float64,
            device=device,
        )
        error = predicted - target
        total_raw = torch.mean(error.square()) / (r2p.THERMAL_FORCE_SCALE_EV_A**2)
        coefficient = r2p.THERMAL_GROUP_MASS[role] / r2p.THERMAL_ROLE_COUNTS[role]
        total_objective = coefficient * total_raw
        metadata = {
            "source_file_index": source_index,
            "role_index": role_index,
            "atom_count": len(structure),
            "config_type": str(structure.info["config_type"]),
            "snapshot_index": int(structure.info.get("snapshot_index", -1)),
            "normalization_eV_A": r2p.THERMAL_FORCE_SCALE_EV_A,
            "aggregate_coefficient": coefficient,
        }
        if role == "E50":
            objective, mode_metadata = r2p._complex_aprime_objective(
                error, structure
            )
            gradient = r2p._objective_gradient(
                objective, parameters, retain_graph=True
            )
            aprime.append(
                r2p.GradientBlock(
                    block_id=f"Aprime_seed0:{role_index:02d}",
                    family="Aprime_seed0",
                    role=role,
                    objective=float(objective.detach().cpu()),
                    gradient=gradient,
                    metadata={**metadata, **mode_metadata},
                )
            )
        gradient = r2p._objective_gradient(
            total_objective, parameters, retain_graph=False
        )
        thermal.append(
            r2p.GradientBlock(
                block_id=f"thermal_total_force:{role}:{role_index:02d}",
                family="thermal_total_force",
                role=role,
                objective=float(total_objective.detach().cpu()),
                gradient=gradient,
                metadata=metadata,
            )
        )

    for source_index, structure in enumerate(loaded.small_zero):
        predicted = _source_force(
            model, structure, loaded.reference8, device, create_graph=True
        )
        rms_raw = torch.mean(predicted.square()) / (r2p.SMALL_RMS_SCALE_EV_A**2)
        rms_objective = rms_raw / len(loaded.small_zero)
        rms_gradient = r2p._objective_gradient(
            rms_objective, parameters, retain_graph=True
        )
        common = {
            "source_file_index": source_index,
            "harmonic_train_source_index": int(
                structure.info["r2o_harmonic_train_source_index"]
            ),
            "atom_count": len(structure),
            "force_component_count": int(predicted.numel()),
            "bond_RMS_A": float(structure.info["r2o_bond_length_RMS_A"]),
        }
        small_rms.append(
            r2p.GradientBlock(
                block_id=f"small_zero_RMS:{source_index:02d}",
                family="small_zero_RMS",
                role="small_zero",
                objective=float(rms_objective.detach().cpu()),
                gradient=rms_gradient,
                metadata={
                    **common,
                    "normalization_eV_A": r2p.SMALL_RMS_SCALE_EV_A,
                    "aggregate_coefficient": 1.0 / len(loaded.small_zero),
                },
            )
        )
        top_objective, top_metadata = r2p._exact_top_objective(
            predicted,
            scale_eV_A=r2p.SMALL_CORRECTION_MARGIN_EV_A,
            aggregate_coefficient=1.0 / len(loaded.small_zero),
        )
        top_gradient = r2p._objective_gradient(
            top_objective, parameters, retain_graph=False
        )
        small_top.append(
            r2p.GradientBlock(
                block_id=f"small_zero_exact_top:{source_index:02d}",
                family="small_zero_exact_top",
                role="small_zero",
                objective=float(top_objective.detach().cpu()),
                gradient=top_gradient,
                metadata={**common, **top_metadata},
            )
        )

    blocks = aprime + thermal + small_rms + small_top
    observed = Counter(block.family for block in blocks)
    if observed != Counter(r2p.PRIMARY_FAMILY_COUNTS):
        raise ValueError(f"R2Q block count changed: {observed}")
    matrix, records = r2p.blocks_to_matrix_and_records(blocks)
    certificate = r2p.solve_max_min_common_descent(matrix)
    after = state_dict_sha256(model)
    if after != before or any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("R2Q cone autograd changed state or populated .grad")
    return CurrentCone(
        blocks=blocks,
        matrix=matrix,
        records=records,
        certificate=certificate,
        state_sha256=before,
        wall_seconds=time.monotonic() - started,
    )


def _top_value_metadata(force: np.ndarray) -> tuple[float, dict[str, Any]]:
    flat = np.asarray(force, dtype=np.float64).reshape(-1)
    if flat.size < 2 or not np.all(np.isfinite(flat)):
        raise FloatingPointError("candidate exact-top force is malformed or non-finite")
    absolute = np.abs(flat)
    order = np.argsort(-absolute, kind="stable")
    top = int(order[0])
    second = int(order[1])
    top_value = float(absolute[top])
    second_value = float(absolute[second])
    exact = np.flatnonzero(absolute == top_value)
    atom, component = divmod(top, 3)
    second_atom, second_component = divmod(second, 3)
    gap = top_value - second_value
    value = (
        top_value**2
        / r2p.PRIMARY_FAMILY_COUNTS["small_zero_exact_top"]
        / (r2p.SMALL_CORRECTION_MARGIN_EV_A**2)
    )
    return value, {
        "top_flat_index": top,
        "top_atom_source_order_0based": atom,
        "top_cartesian_component": "xyz"[component],
        "top_signed_eV_A": float(flat[top]),
        "top_abs_eV_A": top_value,
        "second_flat_index": second,
        "second_atom_source_order_0based": second_atom,
        "second_cartesian_component": "xyz"[second_component],
        "second_signed_eV_A": float(flat[second]),
        "second_abs_eV_A": second_value,
        "exact_tie_count": int(len(exact)),
        "exact_tie_indices": exact.tolist(),
        "absolute_gap_eV_A": float(gap),
        "relative_gap": 0.0 if top_value == 0.0 else float(gap / top_value),
        "candidate_tie_role": "record_only_no_candidate_gradient",
    }


def evaluate_exact_objectives(
    loaded: r2p.LoadedTrainingInputs,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Forward-only values in the exact R2P block order."""

    aprime: list[tuple[str, float, dict[str, Any]]] = []
    thermal: list[tuple[str, float, dict[str, Any]]] = []
    rms: list[tuple[str, float, dict[str, Any]]] = []
    top: list[tuple[str, float, dict[str, Any]]] = []
    role_seen: Counter[str] = Counter()
    for source_index, structure in enumerate(loaded.thermal):
        role = r2p._thermal_role(structure)
        role_index = role_seen[role]
        role_seen[role] += 1
        reference = loaded.reference6 if len(structure) == 72 else loaded.reference8
        prediction = (
            _source_force(model, structure, reference, device, create_graph=False)
            .detach()
            .cpu()
            .numpy()
        )
        target = np.asarray(structure.arrays["REF_forces"], dtype=np.float64)
        error = prediction - target
        coefficient = r2p.THERMAL_GROUP_MASS[role] / r2p.THERMAL_ROLE_COUNTS[role]
        total_value = coefficient * float(
            np.mean(np.square(error)) / (r2p.THERMAL_FORCE_SCALE_EV_A**2)
        )
        metadata = {
            "source_file_index": source_index,
            "role_index": role_index,
            "config_type": str(structure.info["config_type"]),
        }
        thermal.append(
            (f"thermal_total_force:{role}:{role_index:02d}", total_value, metadata)
        )
        if role == "E50":
            mode_real = np.asarray(structure.arrays["APRIME_mode_real"], float)
            mode_imag = np.asarray(structure.arrays["APRIME_mode_imag"], float)
            real = float(np.sum(mode_real * error))
            imag = -float(np.sum(mode_imag * error))
            value = (
                (real**2 + imag**2)
                / (r2p.APRIME_SCALE_EV_A**2)
                / r2p.THERMAL_ROLE_COUNTS["E50"]
            )
            aprime.append(
                (
                    f"Aprime_seed0:{role_index:02d}",
                    value,
                    {**metadata, "projection_real_eV_A": real, "projection_imag_eV_A": imag},
                )
            )
    for source_index, structure in enumerate(loaded.small_zero):
        prediction = (
            _source_force(
                model,
                structure,
                loaded.reference8,
                device,
                create_graph=False,
            )
            .detach()
            .cpu()
            .numpy()
        )
        rms_value = float(
            np.mean(np.square(prediction))
            / (r2p.SMALL_RMS_SCALE_EV_A**2)
            / len(loaded.small_zero)
        )
        common = {
            "source_file_index": source_index,
            "harmonic_train_source_index": int(
                structure.info["r2o_harmonic_train_source_index"]
            ),
        }
        rms.append((f"small_zero_RMS:{source_index:02d}", rms_value, common))
        top_value, top_metadata = _top_value_metadata(prediction)
        top.append(
            (
                f"small_zero_exact_top:{source_index:02d}",
                top_value,
                {**common, **top_metadata},
            )
        )
    ordered = aprime + thermal + rms + top
    if len(ordered) != 176 or len({item[0] for item in ordered}) != 176:
        raise ValueError("R2Q forward objective order/count changed")
    values = np.asarray([item[1] for item in ordered], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("R2Q objective is non-finite")
    records = [
        {"block_id": block_id, "objective": value, "metadata": metadata}
        for block_id, value, metadata in ordered
    ]
    return values, records


def _parameter_schema(model: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]


def parameter_l2_norm(model: torch.nn.Module) -> float:
    return math.sqrt(
        sum(float(torch.sum(parameter.detach().square()).cpu()) for parameter in model.parameters())
    )


def fixed_candidate_grid(theta_norm: float, direction_norm: float) -> list[float]:
    if not math.isfinite(theta_norm) or theta_norm <= 0.0:
        raise ValueError("theta norm must be finite and positive")
    if not math.isfinite(direction_norm) or direction_norm <= 0.0:
        raise ValueError("direction norm must be finite and positive")
    eta0 = GLOBAL_RELATIVE_TRUST_CAP * theta_norm / direction_norm
    return [eta0 / (2**index) for index in range(HALVING_COUNT)]


def trust_receipt_passes(receipt: dict[str, Any]) -> bool:
    global_relative = float(receipt["global_relative_update"])
    tensor_relative = float(receipt["per_tensor_max_relative_update"])
    actual_global_relative = float(receipt["global_actual_relative_update"])
    planned_global_allowance = float(
        receipt["planned_global_cap_roundoff_allowance"]
    )
    actual_global_allowance = float(
        receipt["actual_global_cap_roundoff_allowance"]
    )
    rows = receipt["per_tensor"]
    planned_pass = bool(
        math.isfinite(global_relative)
        and math.isfinite(tensor_relative)
        and math.isfinite(planned_global_allowance)
        and planned_global_allowance >= 0.0
        and global_relative
        <= GLOBAL_RELATIVE_TRUST_CAP + planned_global_allowance
        and tensor_relative <= PER_TENSOR_RELATIVE_TRUST_CAP
    )
    actual_tensor_pass = bool(
        isinstance(rows, list)
        and len(rows) > 0
        and all(
            math.isfinite(float(row["actual_relative_update"]))
            and math.isfinite(float(row["actual_cap_roundoff_allowance"]))
            and float(row["actual_cap_roundoff_allowance"]) >= 0.0
            and float(row["actual_relative_update"])
            <= PER_TENSOR_RELATIVE_TRUST_CAP
            + float(row["actual_cap_roundoff_allowance"])
            for row in rows
        )
    )
    actual_pass = bool(
        math.isfinite(actual_global_relative)
        and math.isfinite(actual_global_allowance)
        and actual_global_allowance >= 0.0
        and actual_global_relative
        <= GLOBAL_RELATIVE_TRUST_CAP + actual_global_allowance
        and actual_tensor_pass
    )
    replay_pass = bool(
        receipt.get("update_replay_pass") is True
        and all(row.get("update_replay_pass") is True for row in rows)
    )
    recorded_passes_match = bool(
        receipt.get("planned_trust_pass") is planned_pass
        and receipt.get("actual_trust_pass") is actual_pass
        and receipt.get("combined_trust_pass")
        is bool(planned_pass and actual_pass and replay_pass)
    )
    return bool(
        planned_pass
        and actual_pass
        and replay_pass
        and recorded_passes_match
    )


def apply_direction_to_clone(
    base: torch.nn.Module,
    direction: np.ndarray,
    eta: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Return theta-eta*d while proving the source model remains untouched."""

    source_hash = state_dict_sha256(base)
    candidate = copy.deepcopy(base)
    base_buffers = {
        name: value.detach().cpu().clone() for name, value in base.named_buffers()
    }
    base_parameters = dict(base.named_parameters())
    offset = 0
    tensor_rows = []
    global_delta_squared = 0.0
    global_actual_delta_squared = 0.0
    global_replay_l2_bound_squared = 0.0
    with torch.no_grad():
        for name, parameter in candidate.named_parameters():
            if name not in base_parameters:
                raise ValueError(f"candidate parameter {name} has no base tensor")
            base_parameter = base_parameters[name]
            count = parameter.numel()
            block = np.asarray(direction[offset : offset + count], dtype=np.float64)
            offset += count
            block_tensor = torch.as_tensor(
                block.reshape(tuple(parameter.shape)),
                dtype=parameter.dtype,
                device=parameter.device,
            )
            base_norm = float(torch.linalg.vector_norm(base_parameter.detach()).cpu())
            direction_tensor_norm = float(
                torch.linalg.vector_norm(block_tensor).cpu()
            )
            delta_norm = abs(float(eta)) * float(
                direction_tensor_norm
            )
            if base_norm == 0.0 and delta_norm > 0.0:
                raise FloatingPointError(
                    f"zero-norm parameter tensor {name} has a nonzero update"
                )
            else:
                relative = 0.0 if base_norm == 0.0 else delta_norm / base_norm
            parameter.add_(block_tensor, alpha=-float(eta))
            actual_delta = parameter.detach() - base_parameter.detach()
            actual_delta_norm = float(torch.linalg.vector_norm(actual_delta).cpu())
            replay_residual = actual_delta + float(eta) * block_tensor
            replay_max_abs = float(replay_residual.abs().max().cpu())
            replay_scale = max(
                1.0,
                float(base_parameter.detach().abs().max().cpu()),
                abs(float(eta)) * float(block_tensor.abs().max().cpu()),
            )
            replay_tolerance = (
                UPDATE_REPLAY_EPS_MULTIPLIER
                * torch.finfo(parameter.dtype).eps
                * replay_scale
            )
            actual_cap_roundoff_allowance = (
                0.0
                if base_norm == 0.0
                else math.sqrt(parameter.numel()) * replay_tolerance / base_norm
            )
            if not math.isfinite(actual_delta_norm) or not math.isfinite(
                replay_max_abs
            ):
                raise FloatingPointError(f"candidate update for {name} is non-finite")
            if replay_max_abs > replay_tolerance:
                raise FloatingPointError(
                    f"candidate update for {name} does not replay theta-eta*d"
                )
            global_delta_squared += delta_norm**2
            global_actual_delta_squared += actual_delta_norm**2
            global_replay_l2_bound_squared += (
                math.sqrt(parameter.numel()) * replay_tolerance
            ) ** 2
            tensor_rows.append(
                {
                    "name": name,
                    "parameter_norm": base_norm,
                    "direction_norm": direction_tensor_norm,
                    "delta_norm": delta_norm,
                    "relative_update": relative,
                    "actual_delta_norm": actual_delta_norm,
                    "actual_relative_update": (
                        0.0 if base_norm == 0.0 else actual_delta_norm / base_norm
                    ),
                    "actual_cap_roundoff_allowance": actual_cap_roundoff_allowance,
                    "update_replay_max_abs": replay_max_abs,
                    "update_replay_tolerance": replay_tolerance,
                    "update_replay_pass": True,
                }
            )
    if offset != direction.size:
        raise ValueError("direction does not match the complete parameter schema")
    if _parameter_schema(candidate) != _parameter_schema(base):
        raise RuntimeError("candidate parameter schema changed")
    if state_dict_sha256(base) != source_hash:
        raise RuntimeError("candidate construction changed its source model")
    if any(
        name not in base_buffers
        or not torch.equal(base_buffers[name], value.detach().cpu())
        for name, value in candidate.named_buffers()
    ) or len(base_buffers) != len(list(candidate.named_buffers())):
        raise RuntimeError("candidate construction changed a non-parameter buffer")
    if not all(torch.isfinite(parameter).all() for parameter in candidate.parameters()):
        raise FloatingPointError("candidate parameter is non-finite")
    theta_norm = parameter_l2_norm(base)
    global_delta = math.sqrt(global_delta_squared)
    global_actual_delta = math.sqrt(global_actual_delta_squared)
    planned_global_cap_roundoff_allowance = (
        UPDATE_REPLAY_EPS_MULTIPLIER
        * torch.finfo(torch.float64).eps
        * GLOBAL_RELATIVE_TRUST_CAP
    )
    actual_global_cap_roundoff_allowance = (
        math.sqrt(global_replay_l2_bound_squared) / theta_norm
    )
    planned_trust_pass = bool(
        global_delta / theta_norm
        <= GLOBAL_RELATIVE_TRUST_CAP + planned_global_cap_roundoff_allowance
        and max(float(item["relative_update"]) for item in tensor_rows)
        <= PER_TENSOR_RELATIVE_TRUST_CAP
    )
    actual_trust_pass = bool(
        global_actual_delta / theta_norm
        <= GLOBAL_RELATIVE_TRUST_CAP + actual_global_cap_roundoff_allowance
        and all(
            float(item["actual_relative_update"])
            <= PER_TENSOR_RELATIVE_TRUST_CAP
            + float(item["actual_cap_roundoff_allowance"])
            for item in tensor_rows
        )
    )
    receipt = {
        "eta": float(eta),
        "base_state_sha256": source_hash,
        "candidate_state_sha256": state_dict_sha256(candidate),
        "theta_norm": theta_norm,
        "global_delta_norm": global_delta,
        "global_relative_update": global_delta / theta_norm,
        "planned_global_cap_roundoff_allowance": planned_global_cap_roundoff_allowance,
        "planned_trust_pass": planned_trust_pass,
        "global_actual_delta_norm": global_actual_delta,
        "global_actual_relative_update": global_actual_delta / theta_norm,
        "actual_global_cap_roundoff_allowance": actual_global_cap_roundoff_allowance,
        "actual_trust_pass": actual_trust_pass,
        "global_actual_vs_planned_delta_norm_abs_difference": abs(
            global_actual_delta - global_delta
        ),
        "per_tensor_max_relative_update": max(
            float(item["relative_update"]) for item in tensor_rows
        ),
        "per_tensor_max_actual_relative_update": max(
            float(item["actual_relative_update"]) for item in tensor_rows
        ),
        "per_tensor": tensor_rows,
        "update_replay_epsilon_multiplier": UPDATE_REPLAY_EPS_MULTIPLIER,
        "update_replay_max_abs": max(
            float(item["update_replay_max_abs"]) for item in tensor_rows
        ),
        "update_replay_pass": all(
            item["update_replay_pass"] for item in tensor_rows
        ),
        "combined_trust_pass": bool(
            planned_trust_pass
            and actual_trust_pass
            and all(item["update_replay_pass"] for item in tensor_rows)
        ),
        "nonparameter_buffers_unchanged": True,
    }
    return candidate, receipt


def armijo_receipt(
    base_values: np.ndarray,
    slopes: np.ndarray,
    candidate_values: np.ndarray,
    eta: float,
    block_ids: Sequence[str],
) -> dict[str, Any]:
    base = np.asarray(base_values, dtype=np.float64)
    slope = np.asarray(slopes, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    if base.shape != (176,) or slope.shape != (176,) or candidate.shape != (176,):
        raise ValueError("Armijo requires exactly 176 aligned values")
    if len(block_ids) != 176 or len(set(block_ids)) != 176:
        raise ValueError("Armijo block ids changed")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(slope)) or not np.all(
        np.isfinite(candidate)
    ):
        raise FloatingPointError("Armijo input is non-finite")
    if np.any(slope <= 0.0):
        raise ValueError("common direction has a nonpositive raw directional slope")
    epsilon = ARMIJO_NUMERIC_EPS * np.maximum(1.0, np.abs(base))
    upper = base - ARMIJO_C1 * float(eta) * slope + epsilon
    margin = upper - candidate
    passed = margin >= 0.0
    rows = [
        {
            "block_index": index,
            "block_id": block_ids[index],
            "base_objective": float(base[index]),
            "candidate_objective": float(candidate[index]),
            "raw_directional_slope": float(slope[index]),
            "required_upper_bound": float(upper[index]),
            "Armijo_margin": float(margin[index]),
            "pass": bool(passed[index]),
        }
        for index in range(176)
    ]
    worst = int(np.argmin(margin))
    return {
        "c1": ARMIJO_C1,
        "eta": float(eta),
        "block_count": 176,
        "all_blocks_pass": bool(np.all(passed)),
        "failed_block_count": int(np.sum(~passed)),
        "minimum_margin": float(margin[worst]),
        "minimum_margin_block_id": block_ids[worst],
        "per_block": rows,
    }


def reference_energy_force(
    model: torch.nn.Module, reference: Any, device: torch.device
) -> dict[str, float]:
    dtype = model_dtype(model)
    graph = fixed_reference_graph(reference, device=str(device), dtype=dtype)
    position = torch.as_tensor(
        np.asarray(reference.positions, float), dtype=dtype, device=device
    ).clone().requires_grad_(True)
    image = torch.zeros_like(position, dtype=torch.int64)
    result = taylor_remainder_energy_and_forces(
        model, graph, position, position.detach(), image, create_graph=False
    )
    return {
        "energy_abs_eV": float(result.energy.detach().abs().max().cpu()),
        "force_max_abs_eV_A": float(
            result.forces_reference_order.detach().abs().max().cpu()
        ),
    }


def _reference_pass(payload: dict[str, float], *, complete: bool) -> bool:
    names = tuple(REFERENCE_LIMITS) if complete else (
        "energy_abs_eV",
        "force_max_abs_eV_A",
    )
    return all(
        math.isfinite(float(payload[name]))
        and float(payload[name]) <= REFERENCE_LIMITS[name]
        for name in names
    )


def implementation_gate(
    model: torch.nn.Module,
    probe: Any,
    references: dict[int, Any],
    reference6: Any,
    device: torch.device,
    *,
    probe_role: str,
) -> dict[str, Any]:
    checks = implementation_checks(model, probe, references, str(device))
    parity = raw_energy_parity(model, probe, reference6, str(device))
    cutoff = cutoff_c2_metrics(model)
    observed = {
        "actual_raw_energy_parity_eV": parity["actual_checkpoint"][
            "absolute_difference_eV"
        ],
        "actual_raw_position_gradient_parity_eV_A": parity["actual_checkpoint"][
            "position_gradient_max_abs_difference_eV_A"
        ],
        "nontrivial_probe_raw_energy_parity_eV": parity[
            "nontrivial_in_memory_semantic_probe"
        ]["absolute_difference_eV"],
        "nontrivial_probe_raw_position_gradient_parity_eV_A": parity[
            "nontrivial_in_memory_semantic_probe"
        ]["position_gradient_max_abs_difference_eV_A"],
        "O3_proper_energy_eV": checks["O3"]["proper"][
            "energy_abs_difference_eV"
        ],
        "O3_proper_force_eV_A": checks["O3"]["proper"][
            "force_max_abs_difference_eV_A"
        ],
        "O3_improper_energy_eV": checks["O3"]["improper"][
            "energy_abs_difference_eV"
        ],
        "O3_improper_force_eV_A": checks["O3"]["improper"][
            "force_max_abs_difference_eV_A"
        ],
        "translation_energy_eV": checks["global_translation"][
            "energy_abs_difference_eV"
        ],
        "translation_force_eV_A": checks["global_translation"][
            "force_max_abs_difference_eV_A"
        ],
        "permutation_energy_eV": checks[
            "permutation_image_energy_abs_difference_eV"
        ],
        "permutation_force_eV_A": checks[
            "permutation_image_force_max_abs_difference_eV_A"
        ],
        "finite_difference_force_eV_A": checks["finite_difference"][
            "absolute_difference_eV_A"
        ],
        "force_Hessian_finite_difference_eV_A2": checks[
            "force_Hessian_finite_difference"
        ]["absolute_difference_eV_A2"],
        "nonreference_Hessian_antisymmetry_eV_A2": checks[
            "nonreference_complete_Hessian"
        ]["antisymmetry_max_abs_eV_A2"],
        "nonreference_Hessian_translation_ASR_eV_A2": checks[
            "nonreference_complete_Hessian"
        ]["translation_ASR_max_abs_eV_A2"],
        "size_energy_eV": checks["size_6x6_8x8"]["energy_abs_difference_eV"],
        "size_force_eV_A": checks["size_6x6_8x8"][
            "central_force_max_abs_difference_eV_A"
        ],
    }
    finite = all(math.isfinite(float(value)) for value in observed.values())
    implementation_pass = finite and all(
        float(observed[name]) <= limit
        for name, limit in IMPLEMENTATION_LIMITS.items()
    )
    cutoff_pass = bool(
        abs(cutoff["value"]) <= 1.0e-12
        and abs(cutoff["first_derivative_A-1"]) <= 1.0e-11
        and abs(cutoff["second_derivative_A-2"]) <= 1.0e-10
        and abs(cutoff["third_derivative_A-3"]) > 1.0e-6
    )
    return {
        "probe_role": probe_role,
        "implementation_limits": IMPLEMENTATION_LIMITS,
        "implementation_observed": observed,
        "implementation_pass": implementation_pass,
        "cutoff_C2": cutoff,
        "cutoff_C2_pass": cutoff_pass,
        "raw_energy_parity": parity,
        "implementation_checks": checks,
        "pass": bool(implementation_pass and cutoff_pass),
    }


def step1_parity(cone: CurrentCone, artifacts: R2PArtifacts) -> dict[str, Any]:
    old = artifacts.gradient_matrix
    new = cone.matrix
    old_norm = np.linalg.norm(old, axis=1)
    new_norm = np.linalg.norm(new, axis=1)
    relative_norm = np.abs(new_norm - old_norm) / old_norm
    old_unit = old / old_norm[:, None]
    new_unit = new / new_norm[:, None]
    row_cosine = np.sum(old_unit * new_unit, axis=1)
    new_direction = cone.certificate.direction
    if new_direction is None:
        raise NumericalInconclusive(1, "step1 cone has no direction", {})
    direction_cosine = float(np.dot(new_direction, artifacts.direction))
    old_min = float(np.min(old_unit @ artifacts.direction))
    new_min = float(np.min(new_unit @ new_direction))
    old_ids = [item["block_id"] for item in artifacts.primary["per_block"]]
    new_ids = [item["block_id"] for item in cone.records]
    old_objectives = np.asarray(
        [item["objective"] for item in artifacts.primary["per_block"]],
        dtype=np.float64,
    )
    new_objectives = np.asarray(
        [item["objective"] for item in cone.records], dtype=np.float64
    )
    objective_error = np.abs(new_objectives - old_objectives)
    objective_tolerance = BASE_OBJECTIVE_PARITY_TOL * np.maximum(
        1.0, np.abs(old_objectives)
    )
    payload = {
        "state_match": cone.state_sha256 == EXPECTED_BASE_STATE_SHA256,
        "block_id_order_match": new_ids == old_ids,
        "gradient_norm_relative_max": float(np.max(relative_norm)),
        "normalized_row_cosine_min": float(np.min(row_cosine)),
        "direction_cosine": direction_cosine,
        "old_min_cosine": old_min,
        "new_min_cosine": new_min,
        "min_cosine_absolute_difference": abs(new_min - old_min),
        "objective_max_abs_difference": float(np.max(objective_error)),
        "objectives_within_tolerance": bool(
            np.all(objective_error <= objective_tolerance)
        ),
    }
    payload["pass"] = bool(
        payload["state_match"]
        and payload["block_id_order_match"]
        and payload["objectives_within_tolerance"]
        and payload["gradient_norm_relative_max"] <= STEP1_GRADIENT_NORM_REL_TOL
        and payload["normalized_row_cosine_min"] >= STEP1_UNIT_COSINE_FLOOR
        and payload["direction_cosine"] >= STEP1_UNIT_COSINE_FLOOR
        and payload["min_cosine_absolute_difference"]
        <= STEP1_MIN_COSINE_ABS_TOL
    )
    return payload


def _load_step_checkpoint(path: Path, expected_state: str, device: torch.device):
    payload = torch_load(path, map_location="cpu")
    if payload.get("format") != STEP_FORMAT:
        raise ValueError("wrong R2Q step checkpoint format")
    model = payload.get("model")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("R2Q step checkpoint contains no model")
    if state_dict_sha256(model) != expected_state:
        raise ValueError("R2Q step checkpoint semantic state changed")
    return model.to(device=device, dtype=torch.float64).eval(), payload


def _assert_model_state_clean(
    model: torch.nn.Module, expected_state: str, label: str
) -> None:
    observed = state_dict_sha256(model)
    if observed != expected_state:
        raise RuntimeError(
            f"{label} semantic state changed: expected {expected_state}, observed {observed}"
        )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError(f"{label} has populated .grad fields")


def _structure_fingerprint(structure: Any) -> str:
    digest = hashlib.sha256()
    for value in (
        np.asarray(structure.numbers, dtype="<i8"),
        np.asarray(structure.cell, dtype="<f8"),
        np.asarray(structure.positions, dtype="<f8"),
        np.asarray(structure.arrays["REF_forces"], dtype="<f8"),
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(
        json.dumps(structure.info, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()


def _write_step_artifacts(
    step_root: Path,
    *,
    step: int,
    parent_state_sha256: str,
    model: torch.nn.Module,
    cone: CurrentCone,
    cone_payload: dict[str, Any],
    direction: np.ndarray,
    armijo: dict[str, Any],
    reference: dict[str, Any],
    update: dict[str, Any],
) -> dict[str, Any]:
    _assert_model_state_clean(
        model, str(update["candidate_state_sha256"]), "accepted R2Q candidate"
    )
    if update.get("base_state_sha256") != parent_state_sha256:
        raise ValueError("accepted update does not bind the parent state")
    if cone.state_sha256 != parent_state_sha256:
        raise ValueError("accepted cone does not bind the parent state")
    step_root.mkdir(parents=True, exist_ok=False)
    gradient_path = step_root / "gradient_matrix.npy"
    gram_path = step_root / "normalized_gram.npy"
    direction_path = step_root / "direction.npy"
    _atomic_numpy(gradient_path, cone.matrix)
    _atomic_numpy(gram_path, cone.certificate.gram)
    _atomic_numpy(direction_path, direction)
    r2p.strict_json(step_root / "cone_certificate.json", cone_payload)
    r2p.strict_json(step_root / "armijo_receipt.json", armijo)
    r2p.strict_json(step_root / "reference_receipt.json", reference)
    state = state_dict_sha256(model)
    checkpoint_path = step_root / "accepted_step.pt"
    checkpoint_payload = {
        "format": STEP_FORMAT,
        "step": step,
        "parent_state_sha256": parent_state_sha256,
        "model_state_sha256": state,
        "raw_MACE_direct_deployment_forbidden": True,
        "model": copy.deepcopy(model).cpu(),
    }
    _atomic_torch(checkpoint_path, checkpoint_payload)
    receipt = {
        "format": STEP_FORMAT,
        "step": step,
        "parent_state_sha256": parent_state_sha256,
        "model_state_sha256": state,
        "checkpoint_sha256": sha256(checkpoint_path),
        "gradient_matrix_sha256": sha256(gradient_path),
        "gradient_matrix_semantic_sha256": r2p._array_sha256(cone.matrix),
        "normalized_gram_file_sha256": sha256(gram_path),
        "normalized_gram_semantic_sha256": r2p._array_sha256(
            cone.certificate.gram
        ),
        "direction_file_sha256": sha256(direction_path),
        "direction_semantic_sha256": r2p._array_sha256(direction),
        "cone_certificate_sha256": sha256(step_root / "cone_certificate.json"),
        "armijo_receipt_sha256": sha256(step_root / "armijo_receipt.json"),
        "reference_receipt_sha256": sha256(step_root / "reference_receipt.json"),
        "update": update,
    }
    r2p.strict_json(step_root / "step_receipt.json", receipt)
    receipt_hash = sha256(step_root / "step_receipt.json")
    _atomic_bytes(step_root / "STEP_ACCEPTED", (receipt_hash + "\n").encode("ascii"))
    return {**receipt, "step_receipt_sha256": receipt_hash}


def _write_train_no_go_evidence(
    output: Path,
    *,
    step: int,
    reason: str,
    cone: CurrentCone,
    cone_payload: dict[str, Any],
    direction: np.ndarray | None,
    candidate_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze enough evidence to audit a reliable scientific TRAIN_NO_GO."""

    root = output / "steps" / f"step_{step:02d}_train_no_go"
    root.mkdir(parents=True, exist_ok=False)
    gradient_path = root / "gradient_matrix.npy"
    gram_path = root / "normalized_gram.npy"
    weights_path = root / "dual_weights.npy"
    _atomic_numpy(gradient_path, cone.matrix)
    _atomic_numpy(gram_path, cone.certificate.gram)
    _atomic_numpy(weights_path, cone.certificate.weights)
    direction_receipt = None
    if direction is not None:
        direction_path = root / "direction.npy"
        _atomic_numpy(direction_path, direction)
        direction_receipt = {
            "file_sha256": sha256(direction_path),
            "semantic_sha256": r2p._array_sha256(direction),
        }
    r2p.strict_json(root / "cone_certificate.json", cone_payload)
    r2p.strict_json(
        root / "candidate_receipts.json",
        {
            "step": step,
            "count": len(candidate_receipts),
            "candidates": candidate_receipts,
        },
    )
    receipt = {
        "format": "graphene_r2q_train_no_go_evidence_v1",
        "step": step,
        "reason": reason,
        "state_sha256": cone.state_sha256,
        "gradient_matrix_file_sha256": sha256(gradient_path),
        "gradient_matrix_semantic_sha256": r2p._array_sha256(cone.matrix),
        "normalized_gram_file_sha256": sha256(gram_path),
        "normalized_gram_semantic_sha256": r2p._array_sha256(
            cone.certificate.gram
        ),
        "dual_weights_file_sha256": sha256(weights_path),
        "dual_weights_semantic_sha256": r2p._array_sha256(
            cone.certificate.weights
        ),
        "direction": direction_receipt,
        "cone_certificate_sha256": sha256(root / "cone_certificate.json"),
        "candidate_receipts_sha256": sha256(root / "candidate_receipts.json"),
        "candidate_count": len(candidate_receipts),
        "optimizer_instantiated": False,
        "optimizer_step_called": False,
    }
    r2p.strict_json(root / "failure_receipt.json", receipt)
    receipt_sha = sha256(root / "failure_receipt.json")
    _atomic_bytes(root / "TRAIN_NO_GO_EVIDENCE_FROZEN", (receipt_sha + "\n").encode("ascii"))
    return {
        "relative_path": str(root.relative_to(output)),
        "failure_receipt_sha256": receipt_sha,
    }


def _write_candidate_receipt(
    output: Path, step: int, halving_index: int, payload: dict[str, Any]
) -> dict[str, str]:
    """Persist every fully evaluated candidate, including rejected candidates."""

    root = output / "candidate_search" / f"step_{step:02d}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"candidate_{halving_index:02d}.json"
    marker = root / f"candidate_{halving_index:02d}.FROZEN"
    if path.exists() or marker.exists():
        raise FileExistsError(f"candidate receipt already exists: {path}")
    r2p.strict_json(path, payload)
    digest = sha256(path)
    _atomic_bytes(marker, (digest + "\n").encode("ascii"))
    return {
        "relative_path": str(path.relative_to(output)),
        "sha256": digest,
        "marker_relative_path": str(marker.relative_to(output)),
    }


def _load_frozen_endpoint(
    endpoint_path: Path,
    endpoint_receipt_path: Path,
    endpoint_marker: Path,
    *,
    expected_receipt_hash: str,
    expected_state: str,
    expected_run_contract_sha256: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Verify the complete train-only endpoint before authorizing held paths."""

    if endpoint_marker.read_bytes() != (expected_receipt_hash + "\n").encode(
        "ascii"
    ):
        raise ValueError("endpoint marker does not bind the endpoint receipt")
    if sha256(endpoint_receipt_path) != expected_receipt_hash:
        raise ValueError("endpoint receipt changed after freeze")
    receipt = r2p.strict_json_load(endpoint_receipt_path)
    if receipt.get("status") != "FOUR_STEP_ENDPOINT_FROZEN" or receipt.get(
        "run_contract_sha256"
    ) != expected_run_contract_sha256:
        raise ValueError("endpoint receipt closure changed")
    _verify_step_chain(
        endpoint_receipt_path.parent,
        receipt.get("step_chain"),
        expected_base_state=receipt.get("base_model_state_sha256"),
        expected_endpoint_state=expected_state,
    )
    expected_checkpoint_hash = receipt.get("endpoint_checkpoint_sha256")
    if sha256(endpoint_path) != expected_checkpoint_hash:
        raise ValueError("endpoint checkpoint file changed")
    payload = torch_load(endpoint_path, map_location="cpu")
    if payload.get("format") != ENDPOINT_FORMAT or payload.get(
        "model_state_sha256"
    ) != expected_state:
        raise ValueError("endpoint checkpoint metadata changed")
    model = payload.get("model")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("endpoint checkpoint contains no model")
    _assert_model_state_clean(model, expected_state, "frozen R2Q endpoint")
    if sha256(endpoint_path) != expected_checkpoint_hash or sha256(
        endpoint_receipt_path
    ) != expected_receipt_hash:
        raise RuntimeError("endpoint artifact changed while it was loaded")
    return model, receipt


def _verify_step_chain(
    output: Path,
    chain: Any,
    *,
    expected_base_state: Any,
    expected_endpoint_state: str,
) -> None:
    """Reload every accepted-step artifact and its parent/file/semantic closure."""

    if not isinstance(chain, list) or len(chain) != STEP_COUNT:
        raise ValueError("endpoint step chain must contain exactly four receipts")
    parent = str(expected_base_state)
    if parent != EXPECTED_BASE_STATE_SHA256:
        raise ValueError("endpoint step chain base state changed")
    for expected_step, frozen in enumerate(chain, start=1):
        if not isinstance(frozen, dict) or int(frozen.get("step", -1)) != expected_step:
            raise ValueError("endpoint step chain order changed")
        root = output / "steps" / f"step_{expected_step:02d}"
        receipt_path = root / "step_receipt.json"
        marker_path = root / "STEP_ACCEPTED"
        expected_receipt_sha = frozen.get("step_receipt_sha256")
        if sha256(receipt_path) != expected_receipt_sha or marker_path.read_bytes() != (
            str(expected_receipt_sha) + "\n"
        ).encode("ascii"):
            raise ValueError(f"step {expected_step} receipt marker/hash changed")
        receipt = r2p.strict_json_load(receipt_path)
        if {**receipt, "step_receipt_sha256": expected_receipt_sha} != frozen:
            raise ValueError(f"step {expected_step} endpoint receipt copy changed")
        if receipt.get("parent_state_sha256") != parent:
            raise ValueError(f"step {expected_step} parent chain changed")

        array_specs = {
            "gradient_matrix.npy": (
                receipt.get("gradient_matrix_sha256"),
                receipt.get("gradient_matrix_semantic_sha256"),
                (176, 42096),
            ),
            "normalized_gram.npy": (
                receipt.get("normalized_gram_file_sha256"),
                receipt.get("normalized_gram_semantic_sha256"),
                (176, 176),
            ),
            "direction.npy": (
                receipt.get("direction_file_sha256"),
                receipt.get("direction_semantic_sha256"),
                (42096,),
            ),
        }
        for name, (file_sha, semantic_sha, shape) in array_specs.items():
            path = root / name
            if sha256(path) != file_sha:
                raise ValueError(f"step {expected_step} {name} file hash changed")
            value = np.load(path, allow_pickle=False)
            if value.dtype != np.float64 or value.shape != shape or not np.all(
                np.isfinite(value)
            ):
                raise ValueError(f"step {expected_step} {name} array contract changed")
            if r2p._array_sha256(value) != semantic_sha or sha256(path) != file_sha:
                raise RuntimeError(f"step {expected_step} {name} changed while loaded")

        file_specs = {
            "cone_certificate.json": receipt.get("cone_certificate_sha256"),
            "armijo_receipt.json": receipt.get("armijo_receipt_sha256"),
            "reference_receipt.json": receipt.get("reference_receipt_sha256"),
            "accepted_step.pt": receipt.get("checkpoint_sha256"),
        }
        for name, expected_sha in file_specs.items():
            if sha256(root / name) != expected_sha:
                raise ValueError(f"step {expected_step} {name} hash changed")
        checkpoint = torch_load(root / "accepted_step.pt", map_location="cpu")
        model = checkpoint.get("model")
        state = receipt.get("model_state_sha256")
        if (
            checkpoint.get("format") != STEP_FORMAT
            or int(checkpoint.get("step", -1)) != expected_step
            or checkpoint.get("parent_state_sha256") != parent
            or checkpoint.get("model_state_sha256") != state
            or not isinstance(model, torch.nn.Module)
        ):
            raise ValueError(f"step {expected_step} checkpoint metadata changed")
        _assert_model_state_clean(model, str(state), f"step {expected_step} checkpoint")
        if sha256(root / "accepted_step.pt") != receipt.get("checkpoint_sha256"):
            raise RuntimeError(f"step {expected_step} checkpoint changed while loaded")
        if (
            sha256(receipt_path) != expected_receipt_sha
            or marker_path.read_bytes()
            != (str(expected_receipt_sha) + "\n").encode("ascii")
            or any(sha256(root / name) != expected_sha for name, expected_sha in file_specs.items())
        ):
            raise RuntimeError(f"step {expected_step} closure changed while verified")
        parent = str(state)
    if parent != expected_endpoint_state:
        raise ValueError("four-step chain does not terminate at the endpoint state")


def _held_structures(
    loaded: r2p.LoadedTrainingInputs,
    endpoint_marker: Path,
    held_started_marker: Path,
    expected_endpoint_receipt_hash: str,
) -> tuple[
    list[Any],
    list[Any],
    list[Any],
    dict[str, str],
    dict[str, Path],
]:
    if endpoint_marker.read_bytes() != (expected_endpoint_receipt_hash + "\n").encode(
        "ascii"
    ):
        raise ValueError("endpoint marker changed before held open")
    endpoint_receipt = endpoint_marker.with_name("endpoint_receipt.json")
    if sha256(endpoint_receipt) != expected_endpoint_receipt_hash:
        raise ValueError("endpoint receipt changed before held open")
    if held_started_marker.read_bytes() != (
        expected_endpoint_receipt_hash + "\n"
    ).encode("ascii"):
        raise ValueError("held-start marker does not bind the endpoint receipt")
    seed1_path = r2p._require_file(
        loaded.data_root, "valid_e50_seed1.xyz", "R2Q held seed1"
    )
    small_path = r2p._require_file(
        loaded.data_root,
        "harmonic_lambda1_small_gate.xyz",
        "R2Q held actual small",
    )
    full_path = r2p._require_file(
        loaded.data_root, "harmonic_full_report.xyz", "R2Q full25 report"
    )
    expected = loaded.data_manifest["outputs"]
    hashes = {
        "seed1": sha256(seed1_path),
        "actual_small": sha256(small_path),
        "full25_report": sha256(full_path),
    }
    if hashes["seed1"] != expected["valid_e50_seed1.xyz"]["sha256"]:
        raise ValueError("R2Q held seed1 hash changed")
    if hashes["actual_small"] != expected["harmonic_lambda1_small_gate.xyz"][
        "sha256"
    ]:
        raise ValueError("R2Q actual-small hash changed")
    if hashes["full25_report"] != expected["harmonic_full_report.xyz"]["sha256"]:
        raise ValueError("R2Q full25 report hash changed")
    seed1 = read(seed1_path, index=":")
    small = read(small_path, index=":")
    full = read(full_path, index=":")
    hashes_after_open = {
        "seed1": sha256(seed1_path),
        "actual_small": sha256(small_path),
        "full25_report": sha256(full_path),
    }
    if hashes_after_open != hashes:
        raise RuntimeError("held input changed while it was opened")
    r2p._validate_report_only_structures(seed1, small)
    if len(full) != 25:
        raise ValueError("R2Q full25 report count changed")
    return seed1, small, full, hashes, {
        "seed1": seed1_path,
        "actual_small": small_path,
        "full25_report": full_path,
    }


def _held_science_gate(endpoint: dict, thresholds: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "e50_seed1_force_RMSE_meV_A": endpoint["E50_seed1"]["force_error"][
            "RMSE_meV_A"
        ],
        "e50_seed1_force_max_abs_meV_A": endpoint["E50_seed1"]["force_error"][
            "max_abs_meV_A"
        ],
        "e50_seed1_Aprime_RMS_meV_A": endpoint["E50_seed1"][
            "Aprime_force_error_RMS_meV_A"
        ],
        "e50_seed1_Aprime_slope_relative_error_abs": abs(
            endpoint["E50_seed1"]["relative_slope_error"]
        ),
        "harmonic_small_force_RMSE_meV_A": endpoint[
            "harmonic_lambda1_small_gate"
        ]["force_error"]["RMSE_meV_A"],
        "harmonic_small_force_max_abs_meV_A": endpoint[
            "harmonic_lambda1_small_gate"
        ]["force_error"]["max_abs_meV_A"],
    }
    missing = [name for name in observed if name not in thresholds]
    if missing:
        raise ValueError(f"R2Q manifest lacks held thresholds: {missing}")
    if not all(math.isfinite(float(value)) for value in observed.values()):
        raise FloatingPointError("R2Q held scientific metric is non-finite")
    if not all(
        math.isfinite(float(thresholds[name])) and float(thresholds[name]) > 0.0
        for name in observed
    ):
        raise ValueError("R2Q held scientific threshold is invalid")
    ratios = {name: float(value) / float(thresholds[name]) for name, value in observed.items()}
    return {
        "observed": observed,
        "thresholds": {name: float(thresholds[name]) for name in observed},
        "normalized_ratios": ratios,
        "pass": all(math.isfinite(value) and value <= 1.0 for value in ratios.values()),
    }


def run_four_step(
    *,
    data_root: Path,
    formal_run_root: Path,
    r2p_run_root: Path,
    contract_doc: Path,
    output_root: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    r2p.assert_formal_graph_source_semantics()
    source_hashes = _validated_source_hashes()
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("R2Q CUDA requested but unavailable")
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("R2Q supports cpu or cuda only")
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    doc = r2p.reject_forbidden_path(
        contract_doc, "R2Q contract doc", must_exist=True
    )
    if not doc.is_file() or sha256(doc) != EXPECTED_CONTRACT_DOC_SHA256:
        raise ValueError("R2Q contract document hash changed")
    artifacts = validate_r2p_artifacts(r2p_run_root)
    output = _prepare_output_root(output_root)
    started = time.monotonic()
    try:
        try:
            loaded = r2p.load_training_inputs(
                data_root, formal_run_root, device=selected_device
            )
        except (FloatingPointError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise NumericalInconclusive(
                None,
                "frozen R2O train/reference inputs could not be loaded",
                {"exception_type": type(error).__name__, "message": str(error)},
            ) from error
        if _validated_source_hashes() != source_hashes:
            raise NumericalInconclusive(
                None, "reviewed helper source changed while R2Q loaded inputs", {}
            )
        if state_dict_sha256(loaded.model) != EXPECTED_BASE_STATE_SHA256:
            raise NumericalInconclusive(None, "R2Q base state changed", {})
        base_model_hash = state_dict_sha256(loaded.model)
        source = Path(__file__).resolve()
        source_snapshot = output / "graphene_r2q_four_step_snapshot.py"
        _atomic_copy(source, source_snapshot)
        if sha256(source_snapshot) != sha256(source):
            raise RuntimeError("R2Q source snapshot copy changed")
        doc_snapshot = output / "R2Q_CONTRACT.md"
        _atomic_copy(doc, doc_snapshot)
        contract = {
            "format": FORMAT,
            "status": "frozen_before_any_R2Q_parameter_gradient",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "authorization": {
                "exact_train_steps": STEP_COUNT,
                "optimizer_instantiated": False,
                "optimizer_step_called": False,
                "held_before_endpoint_authorized": False,
                "seed2_or_support_authorized": False,
                "tail_or_deployment_authorized": False,
            },
            "R2Q_source_snapshot_sha256": sha256(source_snapshot),
            "contract_doc_sha256": sha256(doc_snapshot),
            "reviewed_helper_source_sha256": source_hashes,
            "R2P_input_sha256": artifacts.input_sha256,
            "R2P_primary_sha256": EXPECTED_R2P_PRIMARY_SHA256,
            "R2O_input_sha256": loaded.input_sha256,
            "base_model_state_sha256": base_model_hash,
            "parameter_schema": _parameter_schema(loaded.model),
            "parameter_schema_sha256": r2p.canonical_json_sha256(
                _parameter_schema(loaded.model)
            ),
            "runtime": r2p.runtime_fingerprint(selected_device),
        }
        contract["contract_sha256"] = r2p.canonical_json_sha256(contract)
        r2p.strict_json(output / "run_contract.json", contract)
        run_contract_file_sha256 = sha256(output / "run_contract.json")

        current_model = copy.deepcopy(loaded.model).to(selected_device).eval()
        parent_state = base_model_hash
        step_chain: list[dict[str, Any]] = []
        for step in range(1, STEP_COUNT + 1):
            if state_dict_sha256(current_model) != parent_state:
                raise NumericalInconclusive(
                    step, "current state differs from parent receipt", {}
                )
            try:
                cone = _current_gradient_blocks(
                    loaded, current_model, selected_device
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                raise NumericalInconclusive(
                    step,
                    "current-state gradient cone could not be evaluated",
                    {"exception_type": type(error).__name__, "message": str(error)},
                ) from error
            cone_payload = r2p._certificate_payload(
                cone.certificate,
                cone.records,
                cone.matrix,
                authorization_role=f"R2Q_step_{step}_direction_only",
            )
            cone_payload.update(
                {
                    "R2Q_step": step,
                    "current_state_sha256": parent_state,
                    "step1_R2P_parity": None,
                    "direction_authorized_only_for_this_fixed_step": True,
                    "optimizer_instantiated": False,
                    "optimizer_step_called": False,
                }
            )
            if cone.certificate.scientific_status == "NO_GO":
                evidence = _write_train_no_go_evidence(
                    output,
                    step=step,
                    reason="normalized common-descent cone is reliably NO_GO",
                    cone=cone,
                    cone_payload=cone_payload,
                    direction=None,
                    candidate_receipts=[],
                )
                raise TrainNoGo(
                    step,
                    "normalized common-descent cone is reliably NO_GO",
                    {
                        "dual_norm": cone.certificate.dual_norm,
                        "primal_min_cosine": cone.certificate.primal_min_cosine,
                        "frozen_evidence": evidence,
                    },
                )
            if cone.certificate.scientific_status != "GO" or cone.certificate.direction is None:
                raise NumericalInconclusive(
                    step,
                    "normalized common-descent cone is numerically inconclusive",
                    {
                        "status": cone.certificate.scientific_status,
                        "dual_norm": cone.certificate.dual_norm,
                        "primal_min_cosine": cone.certificate.primal_min_cosine,
                    },
                )
            direction = np.asarray(cone.certificate.direction, dtype=np.float64)
            parity = step1_parity(cone, artifacts) if step == 1 else None
            if parity is not None and not parity["pass"]:
                raise NumericalInconclusive(1, "step1 R2P parity failed", parity)
            cone_payload["step1_R2P_parity"] = parity
            base_values = np.asarray(
                [block.objective for block in cone.blocks], dtype=np.float64
            )
            try:
                base_forward, base_forward_records = evaluate_exact_objectives(
                    loaded, current_model, selected_device
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                raise NumericalInconclusive(
                    step,
                    "current-state forward objectives could not be replayed",
                    {"exception_type": type(error).__name__, "message": str(error)},
                ) from error
            base_difference = np.abs(base_forward - base_values)
            base_tolerance = BASE_OBJECTIVE_PARITY_TOL * np.maximum(
                1.0, np.abs(base_values)
            )
            if np.any(base_difference > base_tolerance):
                index = int(np.argmax(base_difference - base_tolerance))
                raise NumericalInconclusive(
                    step,
                    "gradient and forward objective paths differ",
                    {
                        "block_id": cone.records[index]["block_id"],
                        "absolute_difference": float(base_difference[index]),
                        "tolerance": float(base_tolerance[index]),
                    },
                )
            if [item["block_id"] for item in base_forward_records] != [
                item["block_id"] for item in cone.records
            ]:
                raise NumericalInconclusive(step, "forward block order changed", {})
            try:
                _assert_model_state_clean(
                    current_model, parent_state, f"step {step} forward-replayed base"
                )
                _assert_model_state_clean(
                    loaded.model, base_model_hash, "immutable R2O base bundle"
                )
            except RuntimeError as error:
                raise NumericalInconclusive(
                    step,
                    "base forward replay changed model state or .grad fields",
                    {"exception_type": type(error).__name__, "message": str(error)},
                ) from error
            slopes = cone.matrix @ direction
            if np.any(slopes <= 0.0) or not np.all(np.isfinite(slopes)):
                raise NumericalInconclusive(
                    step, "cone direction has nonpositive raw slope", {}
                )
            theta_norm = parameter_l2_norm(current_model)
            direction_norm = float(np.linalg.norm(direction))
            candidate_grid = fixed_candidate_grid(theta_norm, direction_norm)
            eta0 = candidate_grid[0]
            if step == 1 and abs(eta0 - EXPECTED_STEP1_ETA0) > 1.0e-12:
                raise NumericalInconclusive(
                    1,
                    "step1 eta0 no longer reproduces the frozen R2P scale",
                    {
                        "observed_eta0": eta0,
                        "expected_eta0": EXPECTED_STEP1_ETA0,
                        "absolute_difference": abs(eta0 - EXPECTED_STEP1_ETA0),
                    },
                )
            candidate_receipts = []
            selected_model = None
            selected_armijo = None
            selected_update = None
            selected_objective_records = None
            for halving_index, eta in enumerate(candidate_grid):
                try:
                    candidate, update = apply_direction_to_clone(
                        current_model, direction, eta
                    )
                    candidate_values, candidate_records = evaluate_exact_objectives(
                        loaded, candidate, selected_device
                    )
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    raise NumericalInconclusive(
                        step,
                        "candidate exact objectives could not be evaluated",
                        {
                            "halving_index": halving_index,
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        },
                    ) from error
                try:
                    armijo = armijo_receipt(
                        base_values,
                        slopes,
                        candidate_values,
                        eta,
                        [item["block_id"] for item in cone.records],
                    )
                    ef = reference_energy_force(
                        candidate, loaded.reference6, selected_device
                    )
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    raise NumericalInconclusive(
                        step,
                        "candidate Armijo/reference evaluation was not reliable",
                        {
                            "halving_index": halving_index,
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        },
                    ) from error
                trust_pass = trust_receipt_passes(update)
                try:
                    _assert_model_state_clean(
                        candidate,
                        str(update["candidate_state_sha256"]),
                        f"step {step} candidate k={halving_index}",
                    )
                    _assert_model_state_clean(
                        current_model, parent_state, f"step {step} current parent"
                    )
                    _assert_model_state_clean(
                        loaded.model, base_model_hash, "immutable R2O base bundle"
                    )
                except RuntimeError as error:
                    raise NumericalInconclusive(
                        step,
                        "candidate evaluation changed model state or .grad fields",
                        {
                            "halving_index": halving_index,
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        },
                    ) from error
                candidate_receipt = {
                    "format": "graphene_r2q_candidate_evaluation_v1",
                    "step": step,
                    "parent_state_sha256": parent_state,
                    "gradient_matrix_semantic_sha256": r2p._array_sha256(cone.matrix),
                    "direction_semantic_sha256": r2p._array_sha256(direction),
                    "halving_index": halving_index,
                    "eta": eta,
                    "Armijo_pass": armijo["all_blocks_pass"],
                    "trust_pass": trust_pass,
                    "update": update,
                    "reference_EF": ef,
                    "Armijo": armijo,
                    "candidate_objectives": candidate_records,
                    "candidate_objective_count": len(candidate_records),
                    "candidate_state_verified_after_forward_and_reference": True,
                    "parent_state_verified_after_candidate": True,
                    "base_bundle_verified_after_candidate": True,
                }
                candidate_receipt = _json_safe(candidate_receipt)
                artifact = _write_candidate_receipt(
                    output, step, halving_index, candidate_receipt
                )
                candidate_receipt = {**candidate_receipt, "artifact": artifact}
                candidate_receipts.append(candidate_receipt)
                if not _reference_pass(ef, complete=False):
                    raise NumericalInconclusive(
                        step,
                        "candidate reference E/F invariant failed",
                        {
                            "halving_index": halving_index,
                            "reference": ef,
                            "candidate_receipt": artifact,
                        },
                    )
                if armijo["all_blocks_pass"] and trust_pass:
                    selected_model = candidate
                    selected_armijo = armijo
                    selected_update = update
                    selected_objective_records = candidate_records
                    break
                del candidate
                gc.collect()
                if selected_device.type == "cuda":
                    torch.cuda.empty_cache()
            if selected_model is None:
                evidence = _write_train_no_go_evidence(
                    output,
                    step=step,
                    reason="no useful finite step in the fixed six-candidate grid",
                    cone=cone,
                    cone_payload=cone_payload,
                    direction=direction,
                    candidate_receipts=candidate_receipts,
                )
                raise TrainNoGo(
                    step,
                    "no useful finite step in the fixed six-candidate grid",
                    {
                        "candidate_count": len(candidate_receipts),
                        "frozen_evidence": evidence,
                    },
                )
            try:
                full_reference = reference_null(
                    selected_model, loaded.reference6, str(selected_device)
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                raise NumericalInconclusive(
                    step,
                    "complete reference gate could not be evaluated",
                    {
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "selected_candidate_receipt": candidate_receipts[-1][
                            "artifact"
                        ],
                    },
                ) from error
            if not _reference_pass(full_reference, complete=True):
                raise NumericalInconclusive(
                    step,
                    "provisional selected state failed complete reference gate",
                    {
                        "reference": full_reference,
                        "selected_candidate_receipt": candidate_receipts[-1][
                            "artifact"
                        ],
                    },
                )
            try:
                _assert_model_state_clean(
                    selected_model,
                    str(selected_update["candidate_state_sha256"]),
                    f"step {step} provisional selected state",
                )
                _assert_model_state_clean(
                    current_model, parent_state, f"step {step} current parent"
                )
            except RuntimeError as error:
                raise NumericalInconclusive(
                    step,
                    "complete reference gate changed model state",
                    {"exception_type": type(error).__name__, "message": str(error)},
                ) from error
            step_root = output / "steps" / f"step_{step:02d}"
            armijo_payload = {
                "step": step,
                "eta0": eta0,
                "candidate_grid": candidate_grid,
                "candidate_receipts_until_first_pass": candidate_receipts,
                "selected": selected_armijo,
                "selected_candidate_objectives": selected_objective_records,
                "base_forward_objective_max_abs_difference": float(
                    np.max(base_difference)
                ),
                "first_largest_pass": True,
            }
            reference_payload = {
                "step": step,
                "complete_reference": full_reference,
                "limits": REFERENCE_LIMITS,
                "pass": True,
            }
            receipt = _write_step_artifacts(
                step_root,
                step=step,
                parent_state_sha256=parent_state,
                model=selected_model,
                cone=cone,
                cone_payload=cone_payload,
                direction=direction,
                armijo=armijo_payload,
                reference=reference_payload,
                update=selected_update,
            )
            step_chain.append(receipt)
            next_state = receipt["model_state_sha256"]
            del current_model, selected_model
            try:
                current_model, checkpoint_payload = _load_step_checkpoint(
                    step_root / "accepted_step.pt", next_state, selected_device
                )
            except (RuntimeError, TypeError, ValueError) as error:
                raise NumericalInconclusive(
                    step,
                    "accepted checkpoint could not be reloaded exactly",
                    {"exception_type": type(error).__name__, "message": str(error)},
                ) from error
            if (
                checkpoint_payload.get("step") != step
                or checkpoint_payload.get("parent_state_sha256") != parent_state
                or sha256(step_root / "accepted_step.pt")
                != receipt["checkpoint_sha256"]
                or (step_root / "STEP_ACCEPTED").read_bytes()
                != (receipt["step_receipt_sha256"] + "\n").encode("ascii")
                or sha256(step_root / "step_receipt.json")
                != receipt["step_receipt_sha256"]
            ):
                raise NumericalInconclusive(
                    step, "accepted checkpoint chain changed", {}
                )
            parent_state = next_state

        seed0_probe = loaded.thermal[0]
        if str(seed0_probe.info.get("config_type")) != "r2o_exact_e50_seed0_train" or int(
            seed0_probe.info.get("trajectory_seed", -1)
        ) != 0:
            raise NumericalInconclusive(
                4, "pre-held mechanics probe is not E50 seed0 train[0]", {}
            )
        references = {72: loaded.reference6, 128: loaded.reference8}
        try:
            preheld_mechanics = implementation_gate(
                current_model,
                seed0_probe,
                references,
                loaded.reference6,
                selected_device,
                probe_role="E50_seed0_train_source_index_0_preheld",
            )
        except (FloatingPointError, KeyError, RuntimeError, TypeError, ValueError) as error:
            raise NumericalInconclusive(
                4,
                "seed0-only pre-held mechanics could not be evaluated",
                {"exception_type": type(error).__name__, "message": str(error)},
            ) from error
        preheld_mechanics["probe_fingerprint_sha256"] = _structure_fingerprint(
            seed0_probe
        )
        if not preheld_mechanics["pass"]:
            raise NumericalInconclusive(
                4, "seed0-only pre-held implementation mechanics failed", preheld_mechanics
            )
        try:
            _assert_model_state_clean(
                current_model, parent_state, "step4 pre-held endpoint"
            )
        except RuntimeError as error:
            raise NumericalInconclusive(
                4,
                "pre-held mechanics changed endpoint state or .grad fields",
                {"exception_type": type(error).__name__, "message": str(error)},
            ) from error
        if (
            sha256(source) != sha256(source_snapshot)
            or _validated_source_hashes() != source_hashes
            or r2p.runtime_fingerprint(selected_device) != contract["runtime"]
        ):
            raise NumericalInconclusive(
                4, "R2Q source closure changed before endpoint freeze", {}
            )

        endpoint_path = output / "endpoint.pt"
        endpoint_payload = {
            "format": ENDPOINT_FORMAT,
            "status": "FOUR_STEP_ENDPOINT_FROZEN",
            "model_state_sha256": parent_state,
            "base_state_sha256": base_model_hash,
            "step_count": STEP_COUNT,
            "step_receipt_sha256": [item["step_receipt_sha256"] for item in step_chain],
            "raw_MACE_direct_deployment_forbidden": True,
            "postcore_or_tail_authorized": False,
            "model": copy.deepcopy(current_model).cpu(),
        }
        _atomic_torch(endpoint_path, endpoint_payload)
        endpoint_receipt = {
            "format": ENDPOINT_FORMAT,
            "status": "FOUR_STEP_ENDPOINT_FROZEN",
            "run_contract_sha256": run_contract_file_sha256,
            "run_contract_semantic_sha256": contract["contract_sha256"],
            "protocol_sha256": PROTOCOL_SHA256,
            "contract_doc_sha256": contract["contract_doc_sha256"],
            "R2Q_source_snapshot_sha256": contract[
                "R2Q_source_snapshot_sha256"
            ],
            "reviewed_helper_source_sha256": source_hashes,
            "parameter_schema_sha256": contract["parameter_schema_sha256"],
            "runtime_fingerprint": contract["runtime"],
            "R2P_input_sha256": artifacts.input_sha256,
            "R2O_input_sha256": loaded.input_sha256,
            "endpoint_checkpoint_sha256": sha256(endpoint_path),
            "endpoint_model_state_sha256": parent_state,
            "base_model_state_sha256": base_model_hash,
            "step_count": STEP_COUNT,
            "step_chain": step_chain,
            "preheld_mechanics": preheld_mechanics,
            "preheld_probe_role": "E50_seed0_train_source_index_0",
            "preheld_probe_fingerprint_sha256": _structure_fingerprint(seed0_probe),
            "held_opened_before_endpoint_freeze": False,
            "raw_MACE_direct_deployment_forbidden": True,
            "postcore_or_tail_authorized": False,
        }
        r2p.strict_json(output / "endpoint_receipt.json", endpoint_receipt)
        endpoint_receipt_hash = sha256(output / "endpoint_receipt.json")
        _atomic_bytes(
            output / "ENDPOINT_FROZEN",
            (endpoint_receipt_hash + "\n").encode("ascii"),
        )

        try:
            held_model, reloaded_endpoint_receipt = _load_frozen_endpoint(
                endpoint_path,
                output / "endpoint_receipt.json",
                output / "ENDPOINT_FROZEN",
                expected_receipt_hash=endpoint_receipt_hash,
                expected_state=parent_state,
                expected_run_contract_sha256=run_contract_file_sha256,
            )
        except (FloatingPointError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise NumericalInconclusive(
                None,
                "frozen endpoint could not be independently reloaded",
                {"exception_type": type(error).__name__, "message": str(error)},
            ) from error
        if reloaded_endpoint_receipt != endpoint_receipt:
            raise NumericalInconclusive(None, "endpoint receipt reload changed", {})
        _atomic_bytes(
            output / "HELD_EVALUATION_STARTED",
            (endpoint_receipt_hash + "\n").encode("ascii"),
        )
        try:
            seed1, actual_small, full25, held_hashes, held_paths = _held_structures(
                loaded,
                output / "ENDPOINT_FROZEN",
                output / "HELD_EVALUATION_STARTED",
                endpoint_receipt_hash,
            )
        except (FloatingPointError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise NumericalInconclusive(
                None,
                "held inputs could not be opened under the frozen endpoint",
                {"exception_type": type(error).__name__, "message": str(error)},
            ) from error
        try:
            held_model = held_model.to(selected_device).eval()
            held_endpoint = endpoint_metrics(
                held_model,
                seed1,
                actual_small,
                full25,
                references,
                str(selected_device),
            )
            held_reference = reference_null(
                held_model, loaded.reference6, str(selected_device)
            )
            held_mechanics = implementation_gate(
                held_model,
                seed1[0],
                references,
                loaded.reference6,
                selected_device,
                probe_role="E50_seed1_gate_source_index_0_held",
            )
        except (FloatingPointError, RuntimeError, TypeError, ValueError) as error:
            raise NumericalInconclusive(
                None,
                "held endpoint gate could not be evaluated",
                {"exception_type": type(error).__name__, "message": str(error)},
            ) from error
        if not _reference_pass(held_reference, complete=True) or not held_mechanics[
            "pass"
        ]:
            held_status = "NUMERICAL_INCONCLUSIVE_HELD"
            held_science = None
        else:
            try:
                held_science = _held_science_gate(
                    held_endpoint,
                    loaded.data_manifest["fixed_endpoint_thresholds"],
                )
            except (
                FloatingPointError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                raise NumericalInconclusive(
                    None,
                    "held scientific gate could not be interpreted",
                    {"exception_type": type(error).__name__, "message": str(error)},
                ) from error
            held_status = (
                "R2Q_FOUR_STEP_HELD_GATE_PASSED"
                if held_science["pass"]
                else "R2Q_FOUR_STEP_HELD_GATE_FAILED"
            )
        if state_dict_sha256(held_model) != parent_state or any(
            parameter.grad is not None for parameter in held_model.parameters()
        ):
            raise NumericalInconclusive(None, "held evaluation changed endpoint", {})
        if state_dict_sha256(loaded.model) != base_model_hash:
            raise NumericalInconclusive(None, "R2Q changed original base bundle", {})
        if {name: sha256(path) for name, path in held_paths.items()} != held_hashes:
            raise NumericalInconclusive(None, "held inputs changed during evaluation", {})
        if (
            sha256(endpoint_path) != endpoint_receipt["endpoint_checkpoint_sha256"]
            or sha256(output / "endpoint_receipt.json") != endpoint_receipt_hash
            or (output / "ENDPOINT_FROZEN").read_bytes()
            != (endpoint_receipt_hash + "\n").encode("ascii")
            or (output / "HELD_EVALUATION_STARTED").read_bytes()
            != (endpoint_receipt_hash + "\n").encode("ascii")
            or sha256(source) != sha256(source_snapshot)
            or _validated_source_hashes() != source_hashes
            or r2p.runtime_fingerprint(selected_device) != contract["runtime"]
        ):
            raise NumericalInconclusive(
                None, "endpoint/source closure changed during held evaluation", {}
            )
        try:
            _verify_step_chain(
                output,
                endpoint_receipt["step_chain"],
                expected_base_state=base_model_hash,
                expected_endpoint_state=parent_state,
            )
        except (
            FloatingPointError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise NumericalInconclusive(
                None,
                "accepted-step closure changed during held evaluation",
                {"exception_type": type(error).__name__, "message": str(error)},
            ) from error
        held_report = {
            "format": HELD_FORMAT,
            "status": held_status,
            "endpoint_receipt_sha256": endpoint_receipt_hash,
            "endpoint_checkpoint_sha256": sha256(endpoint_path),
            "endpoint_model_state_sha256": parent_state,
            "held_input_sha256": held_hashes,
            "held_endpoint_metrics": _json_safe(held_endpoint)
            if held_status == "NUMERICAL_INCONCLUSIVE_HELD"
            else held_endpoint,
            "held_science_gate": held_science,
            "held_reference": _json_safe(held_reference)
            if held_status == "NUMERICAL_INCONCLUSIVE_HELD"
            else held_reference,
            "held_mechanics": _json_safe(held_mechanics)
            if held_status == "NUMERICAL_INCONCLUSIVE_HELD"
            else held_mechanics,
            "held_opened_after_endpoint_freeze": True,
            "seed1_or_actual_used_for_step_direction_or_eta": False,
            "held_read_count_policy": "one endpoint evaluation only",
            "postcore_or_tail_authorized": False,
        }
        r2p.strict_json(output / "held_report.json", held_report)
        result = {
            "format": FORMAT,
            "status": held_status,
            "scientific_train_status": "FOUR_STEP_ENDPOINT_FROZEN",
            "held_scientific_pass": None
            if held_science is None
            else held_science["pass"],
            "held_opened": True,
            "endpoint_receipt_sha256": endpoint_receipt_hash,
            "endpoint_checkpoint_sha256": sha256(endpoint_path),
            "held_report_sha256": sha256(output / "held_report.json"),
            "base_state_sha256_before": base_model_hash,
            "base_state_sha256_after": state_dict_sha256(loaded.model),
            "endpoint_state_sha256": parent_state,
            "steps_accepted": STEP_COUNT,
            "optimizer_instantiated": False,
            "optimizer_step_called": False,
            "DFT_calculations": 0,
            "seed2_or_support_read": False,
            "postcore_or_tail_authorized": False,
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_memory_MiB": None
            if selected_device.type != "cuda"
            else float(torch.cuda.max_memory_allocated(selected_device) / 2**20),
        }
        r2p.strict_json(output / "result.json", result)
        terminal_exit = 2 if held_status == "NUMERICAL_INCONCLUSIVE_HELD" else 0
        _atomic_bytes(output / "EXIT_CODE", f"{terminal_exit}\n".encode("ascii"))
        (output / "RUNNING").unlink()
        marker = (
            "HELD_GATE_PASSED"
            if held_status == "R2Q_FOUR_STEP_HELD_GATE_PASSED"
            else "HELD_GATE_FAILED"
            if held_status == "R2Q_FOUR_STEP_HELD_GATE_FAILED"
            else "HELD_NUMERICAL_INCONCLUSIVE"
        )
        _atomic_bytes(output / marker, b"")
        _atomic_bytes(output / "DONE", (held_status + "\n").encode("ascii"))
        return result
    except TrainNoGo as error:
        payload = {
            "format": FORMAT,
            "status": f"TRAIN_NO_GO_AT_STEP_{error.step}",
            "step": error.step,
            "reason": error.reason,
            "details": error.payload,
            "held_opened": False,
            "steps_requested": STEP_COUNT,
            "optimizer_instantiated": False,
            "optimizer_step_called": False,
            "postcore_or_tail_authorized": False,
        }
        payload["details"] = _json_safe(payload["details"])
        r2p.strict_json(output / "result.json", payload)
        _atomic_bytes(output / "EXIT_CODE", b"0\n")
        (output / "RUNNING").unlink(missing_ok=True)
        _atomic_bytes(output / "TRAIN_NO_GO", b"")
        _atomic_bytes(output / "DONE", (payload["status"] + "\n").encode("ascii"))
        return payload
    except NumericalInconclusive as error:
        held_opened = (output / "HELD_EVALUATION_STARTED").exists()
        status = (
            "NUMERICAL_INCONCLUSIVE_HELD"
            if held_opened
            else "NUMERICAL_INCONCLUSIVE"
            if error.step is None
            else f"NUMERICAL_INCONCLUSIVE_AT_STEP_{error.step}"
        )
        payload = {
            "format": FORMAT,
            "status": status,
            "step": error.step,
            "reason": error.reason,
            "details": _json_safe(error.payload),
            "held_opened": held_opened,
            "postcore_or_tail_authorized": False,
        }
        r2p.strict_json(output / "result.json", payload)
        _atomic_bytes(output / "EXIT_CODE", b"2\n")
        (output / "RUNNING").unlink(missing_ok=True)
        marker = (
            "HELD_NUMERICAL_INCONCLUSIVE"
            if held_opened
            else "NUMERICAL_INCONCLUSIVE"
        )
        _atomic_bytes(output / marker, b"")
        _atomic_bytes(output / "DONE", (payload["status"] + "\n").encode("ascii"))
        return payload
    except BaseException:
        try:
            _atomic_bytes(output / "EXIT_CODE", b"1\n")
            _atomic_bytes(output / "FAILED", b"R2Q implementation failure\n")
            (output / "RUNNING").unlink(missing_ok=True)
        except BaseException:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded four-step R2Q train cone and one held gate"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--formal-run-root", type=Path, required=True)
    parser.add_argument("--r2p-run-root", type=Path, required=True)
    parser.add_argument("--contract-doc", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_four_step(
        data_root=args.data_root,
        formal_run_root=args.formal_run_root,
        r2p_run_root=args.r2p_run_root,
        contract_doc=args.contract_doc,
        output_root=args.output_root,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"].startswith("NUMERICAL_INCONCLUSIVE"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
