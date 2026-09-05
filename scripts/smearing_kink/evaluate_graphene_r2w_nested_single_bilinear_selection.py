#!/usr/bin/env python3
"""Nested selection of one off-diagonal R2T bilinear feature.

The frozen 97-column R2S core is augmented by exactly one of the remaining
480 bilinear columns.  Feature index and ridge alpha are selected exclusively
inside each outer training split.  The resulting outer predictions therefore
measure the complete feature-selection algorithm, rather than the optimistic
minimum over 480 pooled OOF diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import (
    DEFAULT_AGGREGATE,
    _parse_whitelist,
    canonical_json_bytes,
    file_sha256,
)
from evaluate_graphene_r2t_bilinear_nested_objective import (
    bilinear_indices,
    design_audit,
    fold_statistics,
    objective_normal_equations,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_DESIGN_ROOT = BASE / "R2T_full_bilinear_materialization_20260826"
DEFAULT_OUTPUT = BASE / "R2W_nested_single_bilinear_selection_20260826"
CORE_WIDTH = 97
OFFDIAGONAL_WIDTH = 480
COMPONENTS_PER_STRUCTURE = 72 * 3


def solve_all_single_feature_coefficients(
    normalizer: np.ndarray,
    gram: np.ndarray,
    rhs: np.ndarray,
    alpha: float,
    core_width: int = CORE_WIDTH,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Solve every core-plus-one-feature ridge problem in one block solve."""
    normalizer = np.asarray(normalizer, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    width = len(normalizer)
    if gram.shape != (width, width) or rhs.shape != (width,):
        raise ValueError("single-feature normal equations have inconsistent shapes")
    if not 0 < core_width < width:
        raise ValueError("single-feature core width must be inside total width")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("single-feature ridge alpha must be finite and positive")

    core_gram = gram[:core_width, :core_width]
    cross = gram[:core_width, core_width:]
    core_rhs = rhs[:core_width]
    off_rhs = rhs[core_width:]
    regularized_core = core_gram + float(alpha) * np.eye(core_width)
    solved = np.linalg.solve(
        regularized_core, np.column_stack((core_rhs, cross))
    )
    inverse_rhs = solved[:, 0]
    inverse_cross = solved[:, 1:]
    schur = (
        np.diag(gram[core_width:, core_width:])
        + float(alpha)
        - np.sum(cross * inverse_cross, axis=0)
    )
    tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.diag(gram))) + float(alpha)
    )
    if np.any(~np.isfinite(schur)) or np.any(schur <= tolerance):
        raise ValueError("single-feature ridge Schur complement is not positive")
    off_normalized = (
        off_rhs - cross.T @ inverse_rhs
    ) / schur
    core_normalized = inverse_rhs[:, None] - inverse_cross * off_normalized
    core_physical = (
        r2r1.FORCE_SCALE_EV_A
        * core_normalized
        / normalizer[:core_width, None]
    )
    off_physical = (
        r2r1.FORCE_SCALE_EV_A
        * off_normalized
        / normalizer[core_width:]
    )
    diagnostics = {
        "minimum_Schur_complement": float(np.min(schur)),
        "maximum_Schur_complement": float(np.max(schur)),
    }
    return core_physical, off_physical, diagnostics


def predict_all_single_features(
    core_design: np.ndarray,
    off_design: np.ndarray,
    fixed: np.ndarray,
    core_coefficients: np.ndarray,
    off_coefficients: np.ndarray,
) -> np.ndarray:
    """Return [structure, flattened force component, candidate] predictions."""
    count = core_design.shape[0]
    core_matrix = core_design.reshape(count * COMPONENTS_PER_STRUCTURE, -1)
    off_matrix = off_design.reshape(count * COMPONENTS_PER_STRUCTURE, -1)
    fixed_vector = fixed.reshape(count * COMPONENTS_PER_STRUCTURE)
    if core_coefficients.shape != (core_matrix.shape[1], off_matrix.shape[1]):
        raise ValueError("single-feature core coefficient matrix shape changed")
    if off_coefficients.shape != (off_matrix.shape[1],):
        raise ValueError("single-feature off coefficient vector shape changed")
    prediction = (
        fixed_vector[:, None]
        + core_matrix @ core_coefficients
        + off_matrix * off_coefficients[None, :]
    )
    return prediction.reshape(count, COMPONENTS_PER_STRUCTURE, -1)


def vectorized_gate_scores(
    predicted_flat: np.ndarray,
    reference: np.ndarray,
    aprime: r2r1.AprimeData,
    indices: Sequence[int],
) -> np.ndarray:
    """Compute the fixed gate score for every prediction candidate."""
    predicted = np.asarray(predicted_flat, dtype=np.float64)
    candidate_count = predicted.shape[-1]
    if predicted.shape != (92, COMPONENTS_PER_STRUCTURE, candidate_count):
        raise ValueError("vectorized gate predictions must cover thermal92")
    selected = np.asarray(indices, dtype=int)
    reference_flat = np.asarray(reference, dtype=np.float64).reshape(
        92, COMPONENTS_PER_STRUCTURE
    )
    error = predicted - reference_flat[..., None]
    ratios: list[np.ndarray] = []
    for group in r2r1.THERMAL_GROUP_RANGES:
        members = np.intersect1d(
            selected, r2r1.group_indices(group), assume_unique=False
        )
        if not len(members):
            raise ValueError(f"vectorized gate selection lacks group {group}")
        current = error[members]
        rmse = 1000.0 * np.sqrt(np.mean(np.square(current), axis=(0, 1)))
        maximum = 1000.0 * np.max(np.abs(current), axis=(0, 1))
        ratios.append(rmse / r2r1.GATE_THRESHOLDS["force_RMSE_meV_A"])
        ratios.append(maximum / r2r1.GATE_THRESHOLDS["force_max_abs_meV_A"])

    e50_global = np.intersect1d(
        selected, r2r1.group_indices("E50_seed0"), assume_unique=False
    )
    local = e50_global
    mode = (
        aprime.mode_real[local] + 1.0j * aprime.mode_imag[local]
    ).reshape(len(local), COMPONENTS_PER_STRUCTURE)
    error_modes = np.einsum(
        "nc,ncq->nq", np.conj(mode), error[e50_global], optimize=True
    )
    aprime_rms = 1000.0 * np.sqrt(np.mean(np.abs(error_modes) ** 2, axis=0))
    ratios.append(
        aprime_rms / r2r1.GATE_THRESHOLDS["E50_Aprime_RMS_meV_A"]
    )

    base = (
        aprime.foundation_base_force_eV_A[local]
        + aprime.frozen_q6_force_eV_A[local]
    ).reshape(len(local), COMPONENTS_PER_STRUCTURE)
    target_total = base + reference_flat[e50_global]
    target_modes = np.einsum(
        "nc,nc->n", np.conj(mode), target_total, optimize=True
    )
    coordinates = (
        aprime.coordinates[local, 0] + 1.0j * aprime.coordinates[local, 1]
    )
    denominator = float(np.vdot(coordinates, coordinates).real)
    target_slope = -float(
        np.vdot(coordinates, target_modes).real / denominator
    )
    error_slope = -np.real(
        np.sum(np.conj(coordinates)[:, None] * error_modes, axis=0)
    ) / denominator
    ratios.append(
        np.abs(error_slope / target_slope)
        / r2r1.GATE_THRESHOLDS["E50_slope_relative_abs_error"]
    )
    return np.maximum.reduce(ratios)


def one_candidate_prediction(
    predicted_flat: np.ndarray, candidate: int
) -> np.ndarray:
    return predicted_flat[..., int(candidate)].reshape(92, 72, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--design-root", type=Path, default=DEFAULT_DESIGN_ROOT)
    parser.add_argument("--projection-mass", type=float, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    projection_masses = (
        tuple(args.projection_mass) if args.projection_mass else (0.50,)
    )
    if any(not 0.0 <= mass < 1.0 for mass in projection_masses):
        raise ValueError("projection masses must be in [0, 1)")
    if len(set(projection_masses)) != len(projection_masses):
        raise ValueError("projection masses must be unique")
    started = time.perf_counter()

    reference, aprime, label_hashes = _parse_whitelist(
        r2r1.RECOMMENDED_THERMAL92
    )
    receipt_path = args.design_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    bilinear_energy_path = args.design_root / "bilinear_energy_design_eV.npy"
    bilinear_force_path = args.design_root / "bilinear_force_design_eV_A.npy"
    bilinear_energy = np.load(bilinear_energy_path, allow_pickle=False)
    bilinear_force = np.load(bilinear_force_path, allow_pickle=False)
    if r2r1.raw_array_sha256(bilinear_energy, "<f8") != receipt[
        "array_raw_sha256"
    ]["energy"]:
        raise ValueError("R2T bilinear energy differs from its receipt")
    if r2r1.raw_array_sha256(bilinear_force, "<f8") != receipt[
        "array_raw_sha256"
    ]["force"]:
        raise ValueError("R2T bilinear force differs from its receipt")
    if file_sha256(args.aggregate.resolve()) != r2r1.ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("attempt3 aggregate SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        base_energy = np.asarray(
            arrays["thermal_parameter_energy_design_eV"], dtype=np.float64
        )
        base_force = np.asarray(
            arrays["thermal_parameter_force_design_eV_A"], dtype=np.float64
        )
        fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], dtype=np.float64)

    diagonal = bilinear_indices("cross32")
    diagonal_set = set(int(value) for value in diagonal)
    offdiagonal = np.asarray(
        [value for value in range(512) if value not in diagonal_set], dtype=int
    )
    if len(offdiagonal) != OFFDIAGONAL_WIDTH:
        raise ValueError("R2W off-diagonal candidate count changed")
    core_energy = np.concatenate(
        (base_energy, np.take(bilinear_energy, diagonal, axis=-1)), axis=1
    )
    core_force = np.concatenate(
        (base_force, np.take(bilinear_force, diagonal, axis=-1)), axis=3
    )
    off_energy = np.take(bilinear_energy, offdiagonal, axis=-1)
    off_force = np.take(bilinear_force, offdiagonal, axis=-1)
    design = np.concatenate((core_force, off_force), axis=3)
    audit = design_audit(design)
    if not audit["pass"]:
        raise ValueError(f"R2W reordered full design audit failed: {audit}")
    statistics = fold_statistics(design, fixed, reference, aprime)

    coefficient_cache: dict[
        tuple[tuple[int, ...], float, float], tuple[np.ndarray, np.ndarray]
    ] = {}

    def coefficients(
        train_folds: Sequence[int], mass: float, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (
            tuple(sorted(int(value) for value in train_folds)),
            float(mass),
            float(alpha),
        )
        if key not in coefficient_cache:
            normalizer, gram, rhs = objective_normal_equations(
                statistics, key[0], key[1]
            )
            core_value, off_value, _ = solve_all_single_feature_coefficients(
                normalizer, gram, rhs, key[2]
            )
            coefficient_cache[key] = (core_value, off_value)
        return coefficient_cache[key]

    all_folds = tuple(range(4))
    all_indices = np.arange(92, dtype=int)
    oof_flat = np.full((92, COMPONENTS_PER_STRUCTURE), np.nan, dtype=np.float64)
    outer_records = []
    for outer_id, outer_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
        outer_hold = np.asarray(outer_tuple, dtype=int)
        outer_train = np.setdiff1d(all_indices, outer_hold)
        inner_folds = tuple(fold for fold in all_folds if fold != outer_id)
        selected_key: tuple[float, float, int, float] | None = None
        selected_inner_flat: np.ndarray | None = None
        selected_position = -1
        selected_mass = math.nan
        selected_alpha = math.nan
        for mass in projection_masses:
            for alpha in r2r1.ALPHA_GRID:
                inner_flat = np.full(
                    (92, COMPONENTS_PER_STRUCTURE, OFFDIAGONAL_WIDTH),
                    np.nan,
                    dtype=np.float64,
                )
                for inner_id in inner_folds:
                    inner_hold = np.asarray(
                        r2r1.FOLD_GLOBAL_INDICES[inner_id], dtype=int
                    )
                    train_folds = tuple(
                        fold for fold in inner_folds if fold != inner_id
                    )
                    core_value, off_value = coefficients(
                        train_folds, mass, alpha
                    )
                    inner_flat[inner_hold] = predict_all_single_features(
                        core_force[inner_hold],
                        off_force[inner_hold],
                        fixed[inner_hold],
                        core_value,
                        off_value,
                    )
                scores = vectorized_gate_scores(
                    inner_flat, reference, aprime, outer_train
                )
                rounded = np.round(scores, r2r1.SCORE_ROUND_DIGITS)
                position = int(np.argmin(rounded))
                key = (
                    float(rounded[position]),
                    float(mass),
                    int(offdiagonal[position]),
                    -float(alpha),
                )
                if selected_key is None or key < selected_key:
                    selected_key = key
                    selected_inner_flat = inner_flat[..., position].copy()
                    selected_position = position
                    selected_mass = float(mass)
                    selected_alpha = float(alpha)
        if selected_inner_flat is None or selected_key is None:
            raise RuntimeError("R2W inner feature selection produced no candidate")
        selected_inner = selected_inner_flat.reshape(92, 72, 3)
        inner_metrics = r2r1.gate_metrics(
            selected_inner, reference, aprime, outer_train
        )
        train_folds = tuple(fold for fold in all_folds if fold != outer_id)
        core_value, off_value = coefficients(
            train_folds, selected_mass, selected_alpha
        )
        outer_all = predict_all_single_features(
            core_force[outer_hold],
            off_force[outer_hold],
            fixed[outer_hold],
            core_value,
            off_value,
        )
        oof_flat[outer_hold] = outer_all[..., selected_position]
        selected_coefficient = np.concatenate(
            (
                core_value[:, selected_position],
                np.asarray([off_value[selected_position]]),
            )
        )
        outer_records.append(
            {
                "outer_fold": outer_id,
                "outer_hold_global_indices": outer_hold.tolist(),
                "selected_projection_mass": selected_mass,
                "selected_alpha": selected_alpha,
                "selected_offdiagonal_position": selected_position,
                "selected_bilinear_index": int(offdiagonal[selected_position]),
                "selected_inner_metrics": inner_metrics,
                "candidate_count": int(
                    len(projection_masses)
                    * len(r2r1.ALPHA_GRID)
                    * OFFDIAGONAL_WIDTH
                ),
                "outer_coefficient_raw_sha256": r2r1.raw_array_sha256(
                    selected_coefficient, "<f8"
                ),
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold_complete": outer_id,
                    "selected_bilinear_index": int(
                        offdiagonal[selected_position]
                    ),
                    "selected_projection_mass": selected_mass,
                    "selected_alpha": selected_alpha,
                    "inner_score": selected_key[0],
                }
            ),
            flush=True,
        )

    if not np.all(np.isfinite(oof_flat)):
        raise RuntimeError("R2W nested OOF does not cover thermal92")
    oof = oof_flat.reshape(92, 72, 3)
    pooled_metrics = r2r1.gate_metrics(oof, reference, aprime, all_indices)

    final_key: tuple[float, float, int, float] | None = None
    final_cv_flat: np.ndarray | None = None
    final_position = -1
    final_mass = math.nan
    final_alpha = math.nan
    for mass in projection_masses:
        for alpha in r2r1.ALPHA_GRID:
            cv_flat = np.full(
                (92, COMPONENTS_PER_STRUCTURE, OFFDIAGONAL_WIDTH),
                np.nan,
                dtype=np.float64,
            )
            for hold_id, hold_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
                hold = np.asarray(hold_tuple, dtype=int)
                train_folds = tuple(
                    fold for fold in all_folds if fold != hold_id
                )
                core_value, off_value = coefficients(train_folds, mass, alpha)
                cv_flat[hold] = predict_all_single_features(
                    core_force[hold],
                    off_force[hold],
                    fixed[hold],
                    core_value,
                    off_value,
                )
            scores = vectorized_gate_scores(cv_flat, reference, aprime, all_indices)
            rounded = np.round(scores, r2r1.SCORE_ROUND_DIGITS)
            position = int(np.argmin(rounded))
            key = (
                float(rounded[position]),
                float(mass),
                int(offdiagonal[position]),
                -float(alpha),
            )
            if final_key is None or key < final_key:
                final_key = key
                final_cv_flat = cv_flat[..., position].copy()
                final_position = position
                final_mass = float(mass)
                final_alpha = float(alpha)
    if final_cv_flat is None or final_key is None:
        raise RuntimeError("R2W final feature selection produced no candidate")
    final_cv = final_cv_flat.reshape(92, 72, 3)
    final_cv_metrics = r2r1.gate_metrics(
        final_cv, reference, aprime, all_indices
    )
    normalizer, gram, rhs = objective_normal_equations(
        statistics, all_folds, final_mass
    )
    final_core_all, final_off_all, final_diagnostics = (
        solve_all_single_feature_coefficients(
            normalizer, gram, rhs, final_alpha
        )
    )
    final_coefficient = np.concatenate(
        (
            final_core_all[:, final_position],
            np.asarray([final_off_all[final_position]]),
        )
    )
    final_energy_design = np.column_stack(
        (core_energy, off_energy[:, final_position])
    )
    final_force_design = np.concatenate(
        (core_force, off_force[..., final_position, None]), axis=3
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "selected_basis_readout_and_OOF.npz"
    np.savez(
        arrays_path,
        nested_OOF_predicted_force_eV_A=oof,
        final_selection_CV_predicted_force_eV_A=final_cv,
        core_bilinear_indices=diagonal,
        offdiagonal_candidate_indices=offdiagonal,
        selected_offdiagonal_position=np.asarray(final_position),
        selected_bilinear_index=np.asarray(offdiagonal[final_position]),
        final_energy_design_eV=final_energy_design,
        final_force_design_eV_A=final_force_design,
        final_physical_coefficient=final_coefficient,
    )
    passed = bool(pooled_metrics["passes_fixed_gate"])
    summary: dict[str, Any] = {
        "format": "graphene_r2w_nested_single_bilinear_selection_v1",
        "status": (
            "R2W_NESTED_FEATURE_SELECTION_OOF_PASSED_DEVELOPMENT_ONLY"
            if passed
            else "R2W_NESTED_FEATURE_SELECTION_OOF_FAILED"
        ),
        "deployable": False,
        "algorithm": (
            "retain frozen 97-column R2S core; select exactly one of 480 "
            "off-diagonal bilinear columns and ridge alpha inside every outer split"
        ),
        "core_column_count": CORE_WIDTH,
        "candidate_count": OFFDIAGONAL_WIDTH,
        "final_column_count": CORE_WIDTH + 1,
        "projection_mass_grid": list(projection_masses),
        "alpha_grid": list(r2r1.ALPHA_GRID),
        "selection_tie_break": (
            "rounded gate score, lower projection mass, lower original bilinear "
            "index, then larger alpha"
        ),
        "design_audit_full_candidate_matrix": audit,
        "nested_feature_selection_OOF": {
            "outer_records": outer_records,
            "pooled_OOF_metrics": pooled_metrics,
        },
        "final_full92_selection": {
            "selected_projection_mass": final_mass,
            "selected_alpha": final_alpha,
            "selected_offdiagonal_position": final_position,
            "selected_bilinear_index": int(offdiagonal[final_position]),
            "selection_CV_metrics": final_cv_metrics,
            "coefficient_raw_sha256": r2r1.raw_array_sha256(
                final_coefficient, "<f8"
            ),
            "energy_design_raw_sha256": r2r1.raw_array_sha256(
                final_energy_design, "<f8"
            ),
            "force_design_raw_sha256": r2r1.raw_array_sha256(
                final_force_design, "<f8"
            ),
            "solver_diagnostics": final_diagnostics,
        },
        "coefficient_cache_entry_count": len(coefficient_cache),
        "input_sha256": {
            "thermal92": file_sha256(r2r1.RECOMMENDED_THERMAL92),
            "aggregate_arrays": file_sha256(args.aggregate),
            "bilinear_design_receipt": file_sha256(receipt_path),
            "bilinear_energy_design": file_sha256(bilinear_energy_path),
            "bilinear_force_design": file_sha256(bilinear_force_path),
        },
        "label_raw_sha256": label_hashes,
        "output_arrays_sha256": file_sha256(arrays_path),
        "energy_labels_used": False,
        "development_or_held_access": False,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "the nested OOF evaluates feature selection without outer-fold label "
            "leakage; the thermal92-developed algorithm still requires the frozen "
            "mechanics and independent development/unseen gates before deployment"
        ),
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
