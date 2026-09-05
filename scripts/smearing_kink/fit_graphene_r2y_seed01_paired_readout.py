#!/usr/bin/env python3
"""Fit frozen R2Y coefficients after promoting E50 seed1 to training data."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase.io import read

import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import (
    DEFAULT_AGGREGATE,
    _parse_whitelist,
    canonical_json_bytes,
    file_sha256,
)
from evaluate_graphene_r2t_bilinear_nested_objective import bilinear_indices, design_audit
from graphene_r2x_paired_readout import SELECTED_BILINEAR_INDICES


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/graphene_r2o_taylor_null_core"
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_R2X_ROOT = BASE / "R2X_paired_readout_freeze_20260826"
DEFAULT_SEED1_DESIGN_ROOT = BASE / "R2Y_seed1_selected_design_20260826"
DEFAULT_OUTPUT = BASE / "R2Y_seed01_paired_readout_freeze_20260826"
SEED1_LABEL_PATH = DATA / "valid_e50_seed1.xyz"
PROJECTION_MASS = 0.45
RIDGE_ALPHA = 1.0e-3
FORCE_SCALE = 0.030
APRIME_SCALE = 0.015
COMPONENTS_PER_STRUCTURE = 72 * 3
GROUP_RANGES = {
    "E50_seed0": (0, 20),
    "E50_seed1": (20, 40),
    "T300": (40, 76),
    "T600": (76, 112),
}
GROUP_MASSES = {
    "E50_seed0": 0.25,
    "E50_seed1": 0.25,
    "T300": 0.25,
    "T600": 0.25,
}
FOLDS = tuple(
    tuple(
        list(range(5 * fold, 5 * (fold + 1)))
        + list(range(20 + 5 * fold, 20 + 5 * (fold + 1)))
        + list(range(40 + 9 * fold, 40 + 9 * (fold + 1)))
        + list(range(76 + 9 * fold, 76 + 9 * (fold + 1)))
    )
    for fold in range(4)
)


def group_indices(group: str) -> np.ndarray:
    start, stop = GROUP_RANGES[group]
    return np.arange(start, stop, dtype=int)


def fit_coefficient(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    modes: np.ndarray,
    train_indices: Sequence[int],
) -> tuple[np.ndarray, dict[str, float]]:
    selected = np.asarray(train_indices, dtype=int)
    width = design.shape[-1]
    response = (reference - fixed) / FORCE_SCALE
    scale = np.sqrt(
        np.sum(np.square(design[selected]), axis=(0, 1, 2))
        / (len(selected) * COMPONENTS_PER_STRUCTURE)
    )
    if np.any(scale <= np.max(scale) * 1.0e-12):
        raise ValueError("R2Y fit subset contains a zero design column")
    force_gram = np.zeros((width, width), dtype=np.float64)
    force_rhs = np.zeros(width, dtype=np.float64)
    for group in GROUP_RANGES:
        members = np.intersect1d(selected, group_indices(group))
        if not len(members):
            raise ValueError(f"R2Y fit subset lacks group {group}")
        matrix = design[members].reshape(-1, width)
        target = response[members].reshape(-1)
        weight = GROUP_MASSES[group] / (len(members) * COMPONENTS_PER_STRUCTURE)
        force_gram += weight * (matrix.T @ matrix)
        force_rhs += weight * (matrix.T @ target)

    e50 = selected[selected < 40]
    projected_design = np.einsum(
        "nat,natk->nk", np.conj(modes[e50]), design[e50], optimize=True
    )
    projected_target = np.einsum(
        "nat,nat->n", np.conj(modes[e50]), reference[e50] - fixed[e50], optimize=True
    )
    projected_design *= FORCE_SCALE / APRIME_SCALE
    projected_target /= APRIME_SCALE
    projection_gram = (
        projected_design.real.T @ projected_design.real
        + projected_design.imag.T @ projected_design.imag
    ) / (2 * len(e50))
    projection_rhs = (
        projected_design.real.T @ projected_target.real
        + projected_design.imag.T @ projected_target.imag
    ) / (2 * len(e50))
    gram = (1.0 - PROJECTION_MASS) * force_gram + PROJECTION_MASS * projection_gram
    rhs = (1.0 - PROJECTION_MASS) * force_rhs + PROJECTION_MASS * projection_rhs
    normalized_gram = gram / scale[:, None] / scale[None, :]
    normalized_rhs = rhs / scale
    regularized = 0.5 * (normalized_gram + normalized_gram.T)
    regularized += RIDGE_ALPHA * np.eye(width)
    normalized_coefficient = np.linalg.solve(regularized, normalized_rhs)
    coefficient = FORCE_SCALE * normalized_coefficient / scale
    condition = float(np.linalg.cond(regularized))
    return coefficient, {
        "regularized_condition": condition,
        "min_relative_column_RMS": float(np.min(scale) / np.max(scale)),
    }


def predict(design: np.ndarray, fixed: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return fixed + np.einsum("natk,k->nat", design, coefficient, optimize=False)


def force_metrics(error: np.ndarray) -> dict[str, float | int]:
    flattened = np.asarray(error).reshape(-1)
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(np.square(flattened)))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(flattened))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(flattened))),
        "n_force_components": int(flattened.size),
    }


def restoring_slope(coordinates: np.ndarray, values: np.ndarray) -> float:
    denominator = float(np.vdot(coordinates, coordinates).real)
    return -float(np.vdot(coordinates, values).real / denominator)


def metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    modes: np.ndarray,
    coordinates: np.ndarray,
    base: np.ndarray,
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    selected = np.asarray(selected_indices, dtype=int)
    output: dict[str, Any] = {"force_by_group": {}, "Aprime_by_seed": {}}
    ratios = []
    for group in GROUP_RANGES:
        members = np.intersect1d(selected, group_indices(group))
        current = force_metrics(prediction[members] - reference[members])
        output["force_by_group"][group] = current
        ratios.extend(
            (current["RMSE_meV_A"] / 30.0, current["max_abs_meV_A"] / 200.0)
        )
    all_error_modes = []
    for name, start, stop in (("seed0", 0, 20), ("seed1", 20, 40)):
        members = np.intersect1d(selected, np.arange(start, stop, dtype=int))
        error_modes = np.einsum(
            "nat,nat->n",
            np.conj(modes[members]),
            prediction[members] - reference[members],
        )
        predicted_modes = np.einsum(
            "nat,nat->n", np.conj(modes[members]), base[members] + prediction[members]
        )
        target_modes = np.einsum(
            "nat,nat->n", np.conj(modes[members]), base[members] + reference[members]
        )
        predicted_slope = restoring_slope(coordinates[members], predicted_modes)
        target_slope = restoring_slope(coordinates[members], target_modes)
        slope_error = (predicted_slope - target_slope) / target_slope
        rms = float(1000.0 * np.sqrt(np.mean(np.abs(error_modes) ** 2)))
        output["Aprime_by_seed"][name] = {
            "RMS_meV_A": rms,
            "predicted_restoring_slope_eV_A2": predicted_slope,
            "target_restoring_slope_eV_A2": target_slope,
            "slope_relative_error": float(slope_error),
            "count": int(len(members)),
        }
        ratios.extend((rms / 15.0, abs(slope_error) / 0.05))
        all_error_modes.append(error_modes)
    combined_rms = float(
        1000.0 * np.sqrt(np.mean(np.abs(np.concatenate(all_error_modes)) ** 2))
    )
    output["Aprime_combined_RMS_meV_A"] = combined_rms
    ratios.append(combined_rms / 15.0)
    output["raw_gate_score"] = float(max(ratios))
    output["passes_fixed_gate"] = bool(all(value <= 1.0 for value in ratios))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--r2x-root", type=Path, default=DEFAULT_R2X_ROOT)
    parser.add_argument(
        "--seed1-design-root", type=Path, default=DEFAULT_SEED1_DESIGN_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    reference92, aprime0, label_hashes0 = _parse_whitelist(
        r2r1.RECOMMENDED_THERMAL92
    )
    seed1_structures = read(SEED1_LABEL_PATH, index=":")
    if len(seed1_structures) != 20:
        raise ValueError("R2Y seed1 label count changed")
    seed1_reference = np.stack(
        [item.arrays["REF_forces"] for item in seed1_structures]
    )
    seed1_modes = np.stack(
        [
            item.arrays["APRIME_mode_real"]
            + 1.0j * item.arrays["APRIME_mode_imag"]
            for item in seed1_structures
        ]
    )
    seed1_coordinates = np.asarray(
        [
            complex(
                item.info["APRIME_coordinate_real_A"],
                item.info["APRIME_coordinate_imag_A"],
            )
            for item in seed1_structures
        ]
    )
    seed1_base = np.stack(
        [
            item.arrays["FOUNDATION_BASE_forces"]
            + item.arrays["FROZEN_Q6_forces"]
            for item in seed1_structures
        ]
    )
    seed0_modes = aprime0.mode_real + 1.0j * aprime0.mode_imag
    seed0_coordinates = aprime0.coordinates[:, 0] + 1.0j * aprime0.coordinates[:, 1]
    seed0_base = aprime0.foundation_base_force_eV_A + aprime0.frozen_q6_force_eV_A

    with np.load(args.r2x_root / "frozen_readout.npz", allow_pickle=False) as arrays:
        thermal_energy_design = np.asarray(
            arrays["thermal_energy_design_eV"], dtype=np.float64
        )
        thermal_force_design = np.asarray(
            arrays["thermal_force_design_eV_A"], dtype=np.float64
        )
        feature_mean = np.asarray(arrays["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(arrays["feature_scale"], dtype=np.float64)
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        thermal_fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], dtype=np.float64)
    seed1_arrays_path = args.seed1_design_root / "seed1_selected_design.npz"
    seed1_receipt_path = args.seed1_design_root / "receipt.json"
    seed1_receipt = json.loads(seed1_receipt_path.read_text())
    if seed1_receipt.get("status") != "R2Y_SEED1_SELECTED_DESIGN_COMPLETE" or (
        file_sha256(seed1_arrays_path) != seed1_receipt.get("arrays_sha256")
    ):
        raise ValueError("R2Y seed1 design receipt is not complete")
    with np.load(seed1_arrays_path, allow_pickle=False) as arrays:
        seed1_fixed = np.asarray(arrays["fixed_force_eV_A"], dtype=np.float64)
        seed1_energy_design = np.asarray(
            arrays["parameter_energy_design_eV"], dtype=np.float64
        )
        seed1_force_design = np.asarray(
            arrays["parameter_force_design_eV_A"], dtype=np.float64
        )
        if not np.array_equal(
            arrays["selected_bilinear_indices"], SELECTED_BILINEAR_INDICES
        ):
            raise ValueError("R2Y seed1 bilinear ordering changed")

    design = np.concatenate(
        (
            thermal_force_design[:20],
            seed1_force_design,
            thermal_force_design[20:],
        )
    )
    energy_design = np.concatenate(
        (
            thermal_energy_design[:20],
            seed1_energy_design,
            thermal_energy_design[20:],
        )
    )
    fixed = np.concatenate((thermal_fixed[:20], seed1_fixed, thermal_fixed[20:]))
    reference = np.concatenate(
        (reference92[:20], seed1_reference, reference92[20:])
    )
    modes = np.concatenate((seed0_modes, seed1_modes))
    coordinates = np.concatenate((seed0_coordinates, seed1_coordinates))
    base = np.concatenate((seed0_base, seed1_base))
    if design.shape != (112, 72, 3, 99):
        raise ValueError("R2Y combined design shape changed")
    audit = design_audit(design)
    if not audit["pass"]:
        raise ValueError(f"R2Y combined design audit failed: {audit}")

    oof = np.full_like(reference, np.nan)
    outer_records = []
    all_indices = np.arange(112, dtype=int)
    for hold_id, hold_tuple in enumerate(FOLDS):
        hold = np.asarray(hold_tuple, dtype=int)
        train = np.setdiff1d(all_indices, hold)
        coefficient, diagnostics = fit_coefficient(
            design, fixed, reference, modes, train
        )
        oof[hold] = predict(design[hold], fixed[hold], coefficient)
        outer_records.append(
            {
                "outer_fold": hold_id,
                "hold_indices": hold.tolist(),
                "coefficient_raw_sha256": r2r1.raw_array_sha256(
                    coefficient, "<f8"
                ),
                "solver": diagnostics,
            }
        )
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("R2Y OOF does not cover all112")
    oof_metrics = metrics(oof, reference, modes, coordinates, base, all_indices)
    if not oof_metrics["passes_fixed_gate"]:
        status = "R2Y_SEED01_FIXED_OOF_FAILED"
    else:
        status = "R2Y_SEED01_TRAIN_GATE_PASSED_MECHANICS_PENDING"
    coefficient, final_diagnostics = fit_coefficient(
        design, fixed, reference, modes, all_indices
    )
    train_prediction = predict(design, fixed, coefficient)
    train_metrics = metrics(
        train_prediction, reference, modes, coordinates, base, all_indices
    )
    if oof_metrics["passes_fixed_gate"] and not train_metrics["passes_fixed_gate"]:
        raise ValueError("R2Y final fit failed after passing OOF")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "frozen_readout.npz"
    np.savez(
        arrays_path,
        physical_coefficient=np.asarray(coefficient, dtype="<f8"),
        base65_physical_coefficient=np.asarray(coefficient[:65], dtype="<f8"),
        bilinear34_physical_coefficient=np.asarray(coefficient[65:], dtype="<f8"),
        selected_bilinear_indices=np.asarray(SELECTED_BILINEAR_INDICES, dtype="<i8"),
        feature_mean=np.asarray(feature_mean, dtype="<f8"),
        feature_scale=np.asarray(feature_scale, dtype="<f8"),
        OOF_predicted_force_eV_A=np.asarray(oof, dtype="<f8"),
        train_predicted_force_eV_A=np.asarray(train_prediction, dtype="<f8"),
        thermal_energy_design_eV=np.asarray(energy_design, dtype="<f8"),
        thermal_force_design_eV_A=np.asarray(design, dtype="<f8"),
    )
    summary = {
        "format": "graphene_r2y_seed01_paired_bilinear_readout_v1",
        "status": status,
        "deployable": False,
        "seed1_role": "training after terminal R2X development failure",
        "representation_or_hyperparameters_changed_after_seed1_failure": False,
        "projection_mass": PROJECTION_MASS,
        "ridge_alpha": RIDGE_ALPHA,
        "group_masses": GROUP_MASSES,
        "folds": [list(item) for item in FOLDS],
        "design_audit": audit,
        "fixed_hyperparameter_OOF": {
            "outer_records": outer_records,
            "metrics": oof_metrics,
            "prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
        },
        "final_all112_fit": {
            "metrics": train_metrics,
            "solver": final_diagnostics,
            "coefficient_raw_sha256": r2r1.raw_array_sha256(coefficient, "<f8"),
        },
        "input_sha256": {
            "thermal92": file_sha256(r2r1.RECOMMENDED_THERMAL92),
            "thermal_aggregate": file_sha256(args.aggregate),
            "R2X_checkpoint": file_sha256(args.r2x_root / "frozen_readout.npz"),
            "seed1_labels": file_sha256(SEED1_LABEL_PATH),
            "seed1_design": file_sha256(seed1_arrays_path),
            "seed1_design_receipt": file_sha256(seed1_receipt_path),
        },
        "thermal92_label_raw_sha256": label_hashes0,
        "output_arrays_sha256": file_sha256(arrays_path),
        "energy_labels_used": False,
        "unseen_525K_access": False,
        "mechanics_pending": bool(oof_metrics["passes_fixed_gate"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(status + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if not oof_metrics["passes_fixed_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
