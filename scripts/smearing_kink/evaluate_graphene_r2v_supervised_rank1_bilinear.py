#!/usr/bin/env python3
"""Nested OOF for one supervised off-diagonal bilinear direction.

For every fit subset and ridge alpha, the 97-column R2S core is fit first.
The objective gradient of its residual with respect to all 480 standardized
off-diagonal bilinear columns defines one unit direction.  The core and that
single conservative direction are then refit jointly.  The direction is
relearned inside every inner/outer training split, so no held-fold force label
enters its prediction.
"""

from __future__ import annotations

import argparse
import json
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
    predict,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_DESIGN_ROOT = BASE / "R2T_full_bilinear_materialization_20260826"
DEFAULT_OUTPUT = BASE / "R2V_supervised_rank1_bilinear_nested_20260826"
CORE_WIDTH = 97
OFFDIAGONAL_WIDTH = 480
PROJECTION_MASS = 0.50


def supervised_rank1_coefficient(
    statistics: Sequence[Any],
    train_folds: Sequence[int],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    normalizer, gram, rhs = objective_normal_equations(
        statistics, train_folds, PROJECTION_MASS
    )
    if gram.shape != (CORE_WIDTH + OFFDIAGONAL_WIDTH,) * 2:
        raise ValueError("R2V reordered design width changed")
    core_gram = gram[:CORE_WIDTH, :CORE_WIDTH]
    core_rhs = rhs[:CORE_WIDTH]
    core_regularized = core_gram + float(alpha) * np.eye(CORE_WIDTH)
    core_initial = np.linalg.solve(core_regularized, core_rhs)
    gradient = rhs[CORE_WIDTH:] - gram[CORE_WIDTH:, :CORE_WIDTH] @ core_initial
    gradient_norm = float(np.linalg.norm(gradient))
    if not np.isfinite(gradient_norm) or gradient_norm <= 1.0e-14:
        raise ValueError("R2V supervised off-diagonal gradient vanished")
    direction = gradient / gradient_norm

    cross = gram[:CORE_WIDTH, CORE_WIDTH:] @ direction
    scalar_gram = float(direction @ gram[CORE_WIDTH:, CORE_WIDTH:] @ direction)
    scalar_rhs = float(direction @ rhs[CORE_WIDTH:])
    augmented = np.empty((CORE_WIDTH + 1, CORE_WIDTH + 1), dtype=np.float64)
    augmented[:CORE_WIDTH, :CORE_WIDTH] = core_gram
    augmented[:CORE_WIDTH, CORE_WIDTH] = cross
    augmented[CORE_WIDTH, :CORE_WIDTH] = cross
    augmented[CORE_WIDTH, CORE_WIDTH] = scalar_gram
    augmented += float(alpha) * np.eye(CORE_WIDTH + 1)
    augmented_rhs = np.concatenate((core_rhs, np.asarray([scalar_rhs])))
    fitted = np.linalg.solve(augmented, augmented_rhs)
    normalized_coefficient = np.concatenate(
        (fitted[:CORE_WIDTH], direction * fitted[CORE_WIDTH])
    )
    physical = r2r1.FORCE_SCALE_EV_A * normalized_coefficient / normalizer
    diagnostics = {
        "gradient_norm": gradient_norm,
        "direction_coefficient": float(fitted[CORE_WIDTH]),
        "offdiagonal_normalized_coefficient_norm": float(
            np.linalg.norm(direction * fitted[CORE_WIDTH])
        ),
    }
    return physical, direction, diagnostics


def nested_oof(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    aprime: r2r1.AprimeData,
    statistics: Sequence[Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    all_indices = np.arange(92, dtype=int)
    all_folds = tuple(range(4))
    oof = np.full_like(reference, np.nan)
    records = []
    for outer_id, outer_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
        outer_hold = np.asarray(outer_tuple, dtype=int)
        outer_train = np.setdiff1d(all_indices, outer_hold)
        inner_folds = tuple(fold for fold in all_folds if fold != outer_id)
        predictions = {
            alpha: np.full_like(reference, np.nan) for alpha in r2r1.ALPHA_GRID
        }
        for inner_id in inner_folds:
            inner_hold = np.asarray(r2r1.FOLD_GLOBAL_INDICES[inner_id], dtype=int)
            train_folds = tuple(fold for fold in inner_folds if fold != inner_id)
            for alpha in r2r1.ALPHA_GRID:
                coefficient, _, _ = supervised_rank1_coefficient(
                    statistics, train_folds, alpha
                )
                predictions[alpha][inner_hold] = predict(
                    design[inner_hold], fixed[inner_hold], coefficient
                )
        candidates = []
        for alpha in r2r1.ALPHA_GRID:
            metrics = r2r1.gate_metrics(
                predictions[alpha], reference, aprime, outer_train
            )
            candidates.append(
                {"alpha": float(alpha), "metrics": metrics}
            )
        selected = min(
            candidates,
            key=lambda item: (
                float(item["metrics"]["selection_score_rounded_12"]),
                -float(item["alpha"]),
            ),
        )
        train_folds = tuple(fold for fold in all_folds if fold != outer_id)
        coefficient, direction, diagnostics = supervised_rank1_coefficient(
            statistics, train_folds, float(selected["alpha"])
        )
        oof[outer_hold] = predict(
            design[outer_hold], fixed[outer_hold], coefficient
        )
        records.append(
            {
                "outer_fold": outer_id,
                "outer_hold_global_indices": outer_hold.tolist(),
                "selected_alpha": selected["alpha"],
                "selected_inner_metrics": selected["metrics"],
                "candidate_count": len(candidates),
                "outer_coefficient_raw_sha256": r2r1.raw_array_sha256(
                    coefficient, "<f8"
                ),
                "outer_supervised_direction_raw_sha256": r2r1.raw_array_sha256(
                    direction, "<f8"
                ),
                "outer_supervised_direction_diagnostics": diagnostics,
            }
        )
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("R2V nested OOF does not cover thermal92")
    return oof, {
        "outer_records": records,
        "pooled_OOF_metrics": r2r1.gate_metrics(
            oof, reference, aprime, all_indices
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--design-root", type=Path, default=DEFAULT_DESIGN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference, aprime, label_hashes = _parse_whitelist(r2r1.RECOMMENDED_THERMAL92)
    receipt_path = args.design_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    bilinear_force_path = args.design_root / "bilinear_force_design_eV_A.npy"
    bilinear_force = np.load(bilinear_force_path, allow_pickle=False)
    if r2r1.raw_array_sha256(bilinear_force, "<f8") != receipt[
        "array_raw_sha256"
    ]["force"]:
        raise ValueError("R2T bilinear force differs from its receipt")
    if file_sha256(args.aggregate.resolve()) != r2r1.ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("attempt3 aggregate SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        base_force = np.asarray(
            arrays["thermal_parameter_force_design_eV_A"], dtype=np.float64
        )
        fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], dtype=np.float64)

    diagonal = bilinear_indices("cross32")
    diagonal_set = set(int(value) for value in diagonal)
    offdiagonal = np.asarray(
        [value for value in range(512) if value not in diagonal_set], dtype=int
    )
    design = np.concatenate(
        (
            base_force,
            np.take(bilinear_force, diagonal, axis=-1),
            np.take(bilinear_force, offdiagonal, axis=-1),
        ),
        axis=3,
    )
    audit = design_audit(design)
    if not audit["pass"]:
        raise ValueError(f"R2V full reordered design audit failed: {audit}")
    statistics = fold_statistics(design, fixed, reference, aprime)
    oof, nested = nested_oof(design, fixed, reference, aprime, statistics)
    passed = bool(nested["pooled_OOF_metrics"]["passes_fixed_gate"])

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "OOF_predictions.npz"
    np.savez(arrays_path, OOF_predicted_force_eV_A=oof)
    summary = {
        "format": "graphene_r2v_supervised_rank1_offdiagonal_bilinear_nested_oof_v1",
        "status": (
            "R2V_SUPERVISED_RANK1_NESTED_OOF_PASSED_DEVELOPMENT_ONLY"
            if passed
            else "R2V_SUPERVISED_RANK1_NESTED_OOF_FAILED"
        ),
        "deployable": False,
        "algorithm": (
            "fit core; take unit offdiagonal objective-gradient direction; "
            "jointly refit core plus one direction inside every split"
        ),
        "core_column_count": CORE_WIDTH,
        "offdiagonal_candidate_count": OFFDIAGONAL_WIDTH,
        "effective_fitted_column_count": CORE_WIDTH + 1,
        "projection_mass": PROJECTION_MASS,
        "alpha_grid": list(r2r1.ALPHA_GRID),
        "design_audit": audit,
        "nested_OOF": nested,
        "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
        "label_raw_sha256": label_hashes,
        "energy_labels_used": False,
        "development_or_held_access": False,
        "input_sha256": {
            "thermal92": file_sha256(r2r1.RECOMMENDED_THERMAL92),
            "aggregate_arrays": file_sha256(args.aggregate),
            "bilinear_design_receipt": file_sha256(receipt_path),
            "bilinear_force_design": file_sha256(bilinear_force_path),
        },
        "interpretation": (
            "conditional nested algorithmic OOF; final direction must be refit on all92 "
            "and pass independent mechanics/development data"
        ),
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
