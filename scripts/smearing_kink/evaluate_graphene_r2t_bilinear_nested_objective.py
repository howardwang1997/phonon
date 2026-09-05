#!/usr/bin/env python3
"""Strict nested mass+ridge OOF for frozen R2T bilinear representations.

The implementation accumulates exact per-fold normal-equation statistics.
This avoids repeating tall SVDs while retaining the same group-balanced force
and A-prime projection objective used by R2S.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
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


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_DESIGN_ROOT = BASE / "R2T_full_bilinear_materialization_20260826"
DEFAULT_OUTPUT = BASE / "R2T_full512_nested_mass_alpha_20260826"
PROJECTION_MASS_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
COMPONENTS_PER_STRUCTURE = 72 * 3


@dataclass(frozen=True)
class FoldStatistics:
    column_square_sum: np.ndarray
    group_count: dict[str, int]
    group_gram: dict[str, np.ndarray]
    group_rhs: dict[str, np.ndarray]
    projection_count: int
    projection_gram: np.ndarray
    projection_rhs: np.ndarray


@dataclass(frozen=True)
class EigenRidgeSystem:
    normalizer: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    rhs: np.ndarray

    def coefficient(self, alpha: float) -> np.ndarray:
        projected_rhs = self.eigenvectors.T @ self.rhs
        normalized = self.eigenvectors @ (
            projected_rhs / (self.eigenvalues + float(alpha))
        )
        return r2r1.FORCE_SCALE_EV_A * normalized / self.normalizer


def bilinear_indices(kind: str) -> np.ndarray:
    diagonal_plain = [channel * 16 + channel for channel in range(16)]
    diagonal_modulated = [256 + channel * 16 + channel for channel in range(16)]
    if kind == "cross32":
        values = diagonal_plain + diagonal_modulated
    elif kind == "plain256":
        values = list(range(256))
    elif kind == "plain256_plus_c_diagonal16":
        values = list(range(256)) + diagonal_modulated
    elif kind == "diagonal16_plus_c_full256":
        values = diagonal_plain + list(range(256, 512))
    elif kind == "full512":
        values = list(range(512))
    else:
        raise ValueError(f"unknown R2T bilinear basis: {kind}")
    return np.asarray(values, dtype=int)


def fold_statistics(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    aprime: r2r1.AprimeData,
) -> tuple[FoldStatistics, ...]:
    width = design.shape[-1]
    response = (reference - fixed) / r2r1.FORCE_SCALE_EV_A
    output = []
    for fold_id, fold_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
        fold = np.asarray(fold_tuple, dtype=int)
        selected = design[fold]
        square_sum = np.sum(np.square(selected), axis=(0, 1, 2))
        group_count: dict[str, int] = {}
        group_gram: dict[str, np.ndarray] = {}
        group_rhs: dict[str, np.ndarray] = {}
        for group in r2r1.THERMAL_GROUP_RANGES:
            indices = np.asarray(
                [index for index in fold if r2r1.group_for_global_index(index) == group],
                dtype=int,
            )
            group_count[group] = int(len(indices))
            matrix = design[indices].reshape(-1, width)
            target = response[indices].reshape(-1)
            group_gram[group] = matrix.T @ matrix
            group_rhs[group] = matrix.T @ target

        e50 = fold[fold < 20]
        mode = aprime.mode_real[e50] + 1.0j * aprime.mode_imag[e50]
        projected_design = np.einsum(
            "nat,natk->nk", np.conj(mode), design[e50], optimize=True
        )
        projected_target = np.einsum(
            "nat,nat->n", np.conj(mode), reference[e50] - fixed[e50], optimize=True
        )
        projected_design *= r2r1.FORCE_SCALE_EV_A / 0.015
        projected_target /= 0.015
        projection_gram = (
            projected_design.real.T @ projected_design.real
            + projected_design.imag.T @ projected_design.imag
        )
        projection_rhs = (
            projected_design.real.T @ projected_target.real
            + projected_design.imag.T @ projected_target.imag
        )
        output.append(
            FoldStatistics(
                column_square_sum=square_sum,
                group_count=group_count,
                group_gram=group_gram,
                group_rhs=group_rhs,
                projection_count=int(len(e50)),
                projection_gram=projection_gram,
                projection_rhs=projection_rhs,
            )
        )
        print(
            json.dumps(
                {
                    "fold_statistics_complete": fold_id,
                    "width": width,
                }
            ),
            flush=True,
        )
    return tuple(output)


def objective_normal_equations(
    statistics: Sequence[FoldStatistics],
    train_folds: Sequence[int],
    projection_mass: float,
    penalty: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = [statistics[int(fold)] for fold in train_folds]
    structure_count = sum(sum(item.group_count.values()) for item in chosen)
    square_sum = sum(
        (item.column_square_sum for item in chosen),
        start=np.zeros_like(chosen[0].column_square_sum),
    )
    scale = np.sqrt(square_sum / (structure_count * COMPONENTS_PER_STRUCTURE))
    relative = scale / np.max(scale)
    if np.any(relative <= 1.0e-12):
        raise ValueError("R2T readout contains a zero or near-zero force column")
    if penalty is None:
        penalty = np.ones_like(scale)
    penalty = np.asarray(penalty, dtype=np.float64)
    if penalty.shape != scale.shape or np.any(~np.isfinite(penalty)) or np.any(
        penalty < 1.0
    ):
        raise ValueError("R2T ridge penalty must be finite, >=1, and match width")
    normalizer = scale * np.sqrt(penalty)

    width = len(scale)
    force_gram = np.zeros((width, width), dtype=np.float64)
    force_rhs = np.zeros(width, dtype=np.float64)
    for group in r2r1.THERMAL_GROUP_RANGES:
        count = sum(item.group_count[group] for item in chosen)
        raw_gram = sum(
            (item.group_gram[group] for item in chosen),
            start=np.zeros((width, width), dtype=np.float64),
        )
        raw_rhs = sum(
            (item.group_rhs[group] for item in chosen),
            start=np.zeros(width, dtype=np.float64),
        )
        weight = r2r1.GROUP_MASSES[group] / (count * COMPONENTS_PER_STRUCTURE)
        force_gram += weight * raw_gram
        force_rhs += weight * raw_rhs

    projection_count = sum(item.projection_count for item in chosen)
    projection_gram = sum(
        (item.projection_gram for item in chosen),
        start=np.zeros((width, width), dtype=np.float64),
    )
    projection_rhs = sum(
        (item.projection_rhs for item in chosen),
        start=np.zeros(width, dtype=np.float64),
    )
    projection_gram /= 2 * projection_count
    projection_rhs /= 2 * projection_count
    gram = (1.0 - projection_mass) * force_gram + projection_mass * projection_gram
    rhs = (1.0 - projection_mass) * force_rhs + projection_mass * projection_rhs
    gram = gram / normalizer[:, None] / normalizer[None, :]
    rhs = rhs / normalizer
    gram = 0.5 * (gram + gram.T)
    return normalizer, gram, rhs


def ridge_system(
    statistics: Sequence[FoldStatistics],
    train_folds: Sequence[int],
    projection_mass: float,
    penalty: np.ndarray | None = None,
) -> EigenRidgeSystem:
    normalizer, gram, rhs = objective_normal_equations(
        statistics, train_folds, projection_mass, penalty
    )
    width = len(normalizer)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo(np.float64).eps * width * max(float(eigenvalues[-1]), 1.0)
    if float(eigenvalues[0]) < -10.0 * tolerance:
        raise ValueError("R2T normal-equation Gram matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return EigenRidgeSystem(normalizer, eigenvalues, eigenvectors, rhs)


def predict(
    design: np.ndarray, fixed: np.ndarray, coefficient: np.ndarray
) -> np.ndarray:
    return fixed + np.einsum("natk,k->nat", design, coefficient, optimize=False)


def design_audit(design: np.ndarray) -> dict[str, Any]:
    width = design.shape[-1]
    scale = np.sqrt(np.mean(np.square(design), axis=(0, 1, 2)))
    matrix = (design / scale).reshape(-1, width)
    gram = matrix.T @ matrix
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    singular_tolerance = (
        np.finfo(np.float64).eps
        * max(matrix.shape)
        * math.sqrt(float(eigenvalues[-1]))
    )
    rank = int(np.sum(eigenvalues > singular_tolerance**2))
    smallest = max(float(eigenvalues[0]), np.finfo(np.float64).tiny)
    return {
        "width": width,
        "rank": rank,
        "condition": float(math.sqrt(float(eigenvalues[-1]) / smallest)),
        "min_relative_column_RMS": float(np.min(scale) / np.max(scale)),
        "pass": bool(rank == width and np.all(scale > np.max(scale) * 1.0e-12)),
    }


def nested_mass_alpha(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    aprime: r2r1.AprimeData,
    statistics: Sequence[FoldStatistics],
    projection_masses: Sequence[float],
    penalties: Sequence[tuple[float, np.ndarray]],
    alpha_grid: Sequence[float] = r2r1.ALPHA_GRID,
) -> tuple[np.ndarray, dict[str, Any]]:
    all_indices = np.arange(92, dtype=int)
    all_folds = tuple(range(4))
    cache: dict[tuple[tuple[int, ...], float, float], EigenRidgeSystem] = {}

    def system(
        train_folds: Sequence[int], mass: float, penalty_value: float
    ) -> EigenRidgeSystem:
        key = (
            tuple(sorted(int(value) for value in train_folds)),
            float(mass),
            float(penalty_value),
        )
        if key not in cache:
            penalty = next(
                vector for value, vector in penalties if float(value) == key[2]
            )
            cache[key] = ridge_system(statistics, key[0], key[1], penalty)
        return cache[key]

    oof = np.full_like(reference, np.nan)
    outer_records = []
    for outer_id, outer_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
        outer_hold = np.asarray(outer_tuple, dtype=int)
        outer_train = np.setdiff1d(all_indices, outer_hold)
        inner_folds = tuple(fold for fold in all_folds if fold != outer_id)
        predictions = {
            (mass, penalty_value, alpha): np.full_like(reference, np.nan)
            for mass in projection_masses
            for penalty_value, _ in penalties
            for alpha in alpha_grid
        }
        for inner_id in inner_folds:
            inner_hold = np.asarray(r2r1.FOLD_GLOBAL_INDICES[inner_id], dtype=int)
            inner_train_folds = tuple(
                fold for fold in inner_folds if fold != inner_id
            )
            for mass in projection_masses:
                for penalty_value, _ in penalties:
                    current = system(inner_train_folds, mass, penalty_value)
                    for alpha in alpha_grid:
                        coefficient = current.coefficient(alpha)
                        predictions[(mass, penalty_value, alpha)][inner_hold] = predict(
                            design[inner_hold], fixed[inner_hold], coefficient
                        )
        candidates = []
        for mass in projection_masses:
            for penalty_value, _ in penalties:
                for alpha in alpha_grid:
                    metrics = r2r1.gate_metrics(
                        predictions[(mass, penalty_value, alpha)],
                        reference,
                        aprime,
                        outer_train,
                    )
                    candidates.append(
                        {
                            "projection_mass": mass,
                            "offdiagonal_penalty_multiplier": penalty_value,
                            "alpha": float(alpha),
                            "metrics": metrics,
                        }
                    )
        selected = min(
            candidates,
            key=lambda item: (
                float(item["metrics"]["selection_score_rounded_12"]),
                float(item["projection_mass"]),
                -float(item["offdiagonal_penalty_multiplier"]),
                -float(item["alpha"]),
            ),
        )
        outer_folds = tuple(fold for fold in all_folds if fold != outer_id)
        current = system(
            outer_folds,
            float(selected["projection_mass"]),
            float(selected["offdiagonal_penalty_multiplier"]),
        )
        coefficient = current.coefficient(float(selected["alpha"]))
        oof[outer_hold] = predict(design[outer_hold], fixed[outer_hold], coefficient)
        outer_records.append(
            {
                "outer_fold": outer_id,
                "outer_hold_global_indices": outer_hold.tolist(),
                "selected_projection_mass": selected["projection_mass"],
                "selected_offdiagonal_penalty_multiplier": selected[
                    "offdiagonal_penalty_multiplier"
                ],
                "selected_alpha": selected["alpha"],
                "selected_inner_metrics": selected["metrics"],
                "candidate_count": len(candidates),
                "outer_coefficient_raw_sha256": r2r1.raw_array_sha256(
                    coefficient, "<f8"
                ),
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold_complete": outer_id,
                    "selected_projection_mass": selected["projection_mass"],
                    "selected_offdiagonal_penalty_multiplier": selected[
                        "offdiagonal_penalty_multiplier"
                    ],
                    "selected_alpha": selected["alpha"],
                }
            ),
            flush=True,
        )
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("R2T nested OOF does not cover thermal92")
    pooled = r2r1.gate_metrics(oof, reference, aprime, all_indices)
    return oof, {
        "outer_records": outer_records,
        "pooled_OOF_metrics": pooled,
        "unique_ridge_system_count": len(cache),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="full512")
    parser.add_argument(
        "--bilinear-index",
        type=int,
        action="append",
        help="override --basis with an explicit unique 0..511 index list",
    )
    parser.add_argument("--projection-mass", type=float, action="append")
    parser.add_argument("--offdiagonal-penalty", type=float, action="append")
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--design-root", type=Path, default=DEFAULT_DESIGN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    projection_masses = (
        tuple(args.projection_mass)
        if args.projection_mass is not None
        else PROJECTION_MASS_GRID
    )
    if not projection_masses or any(
        not 0.0 <= value < 1.0 for value in projection_masses
    ):
        raise ValueError("projection masses must be in [0, 1)")
    if len(set(projection_masses)) != len(projection_masses):
        raise ValueError("projection masses must be unique")
    penalty_values = (
        tuple(args.offdiagonal_penalty)
        if args.offdiagonal_penalty is not None
        else (1.0,)
    )
    if not penalty_values or any(
        not math.isfinite(value) or value < 1.0 for value in penalty_values
    ):
        raise ValueError("offdiagonal penalties must be finite and >= 1")
    if len(set(penalty_values)) != len(penalty_values):
        raise ValueError("offdiagonal penalties must be unique")
    started = time.perf_counter()

    reference, aprime, label_hashes = _parse_whitelist(r2r1.RECOMMENDED_THERMAL92)
    design_receipt_path = args.design_root / "receipt.json"
    design_receipt = json.loads(design_receipt_path.read_text())
    bilinear_force_path = args.design_root / "bilinear_force_design_eV_A.npy"
    bilinear_force = np.load(bilinear_force_path, allow_pickle=False)
    if r2r1.raw_array_sha256(bilinear_force, "<f8") != design_receipt[
        "array_raw_sha256"
    ]["force"]:
        raise ValueError("R2T bilinear force design differs from its receipt")
    if file_sha256(args.aggregate.resolve()) != r2r1.ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("attempt3 aggregate SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        base_force = np.asarray(
            arrays["thermal_parameter_force_design_eV_A"], dtype=np.float64
        )
        fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], dtype=np.float64)
    if args.bilinear_index is None:
        selected = bilinear_indices(args.basis)
        basis_name = args.basis
    else:
        if (
            not args.bilinear_index
            or len(set(args.bilinear_index)) != len(args.bilinear_index)
            or any(index < 0 or index >= 512 for index in args.bilinear_index)
        ):
            raise ValueError("explicit bilinear indices must be unique values in 0..511")
        selected = np.asarray(args.bilinear_index, dtype=int)
        basis_name = "explicit_indices"
    design = np.concatenate((base_force, bilinear_force[..., selected]), axis=3)
    diagonal = set(int(index) for index in bilinear_indices("cross32"))
    penalties = []
    for penalty_value in penalty_values:
        vector = np.ones(design.shape[-1], dtype=np.float64)
        for position, original_index in enumerate(selected, start=65):
            if int(original_index) not in diagonal:
                vector[position] = penalty_value
        penalties.append((float(penalty_value), vector))
    audit = design_audit(design)
    if not audit["pass"]:
        raise ValueError(f"R2T {basis_name} design audit failed: {audit}")
    statistics = fold_statistics(design, fixed, reference, aprime)
    oof, nested = nested_mass_alpha(
        design,
        fixed,
        reference,
        aprime,
        statistics,
        projection_masses,
        penalties,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "OOF_predictions.npz", OOF_predicted_force_eV_A=oof)
    passed = bool(nested["pooled_OOF_metrics"]["passes_fixed_gate"])
    summary = {
        "format": "graphene_r2t_bilinear_nested_projection_mass_alpha_oof_v1",
        "status": (
            "R2T_BILINEAR_NESTED_OOF_PASSED_DEVELOPMENT_ONLY"
            if passed
            else "R2T_BILINEAR_NESTED_OOF_FAILED"
        ),
        "deployable": False,
        "basis": basis_name,
        "base_column_count": 65,
        "bilinear_column_count": int(len(selected)),
        "total_column_count": int(design.shape[-1]),
        "bilinear_indices": selected.tolist(),
        "projection_mass_grid": list(projection_masses),
        "offdiagonal_penalty_multiplier_grid": list(penalty_values),
        "alpha_grid": list(r2r1.ALPHA_GRID),
        "selection_tie_break": (
            "lower projection mass, then larger offdiagonal penalty, then larger alpha"
        ),
        "design_audit": audit,
        "nested_OOF": nested,
        "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
        "input_sha256": {
            "thermal92": file_sha256(r2r1.RECOMMENDED_THERMAL92),
            "aggregate_arrays": file_sha256(args.aggregate),
            "bilinear_design_receipt": file_sha256(design_receipt_path),
            "bilinear_force_design": file_sha256(bilinear_force_path),
        },
        "label_raw_sha256": label_hashes,
        "energy_labels_used": False,
        "development_or_held_access": False,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "conditional nested readout OOF; the representation was developed on "
            "thermal92 and still requires independent mechanics/development data"
        ),
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
