#!/usr/bin/env python3
"""Extend boundary hyperparameters for the fixed seed012 step-32 design."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_graphene_r2z_forward_bilinear as r2z
import fit_graphene_r2ad_seed012_step32 as r2ad
import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_OUTPUT = BASE / "R2AE_seed012_step32_boundary_20260826"
PROJECTION_MASSES = (0.55, 0.625, 0.70, 0.75, 0.80)
BILINEAR_PENALTIES = (0.40, 0.75, 1.25, 2.0)
T600_FORCE_FRACTIONS = (0.90, 0.93, 0.95, 0.97)
ALPHAS = tuple(float(value) for value in np.geomspace(1.5e-2, 1.5e-1, 25))


def seed_weight_grid() -> tuple[tuple[float, float, float], ...]:
    output = []
    for seed2_weight in (0.10, 0.15, 0.20, 0.25, 0.30):
        for seed1_fraction in (0.50, 0.575, 0.65):
            remainder = 1.0 - seed2_weight
            output.append(
                (
                    remainder * (1.0 - seed1_fraction),
                    remainder * seed1_fraction,
                    seed2_weight,
                )
            )
    return tuple(output)


SEED_PROJECTION_WEIGHTS = seed_weight_grid()


def coefficient_matrix(
    system: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    normalizer, gram, rhs = system
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo(float).eps * len(normalizer) * max(
        float(eigenvalues[-1]), 1.0
    )
    if eigenvalues[0] < -10.0 * tolerance:
        raise ValueError("R2AE Gram is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ rhs
    normalized = eigenvectors @ (
        projected_rhs[:, None]
        / (eigenvalues[:, None] + np.asarray(ALPHAS)[None, :])
    )
    return r2ad.FORCE_SCALE * normalized / normalizer[:, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=r2z.DEFAULT_AGGREGATE)
    parser.add_argument("--failed-root", type=Path, default=r2z.DEFAULT_R2Y_FAILED)
    parser.add_argument(
        "--seed1-selected-root", type=Path, default=r2z.DEFAULT_SEED1_SELECTED
    )
    parser.add_argument("--thermal-full-root", type=Path, default=r2z.DEFAULT_THERMAL_FULL)
    parser.add_argument("--seed1-full-root", type=Path, default=r2z.DEFAULT_SEED1_FULL)
    parser.add_argument("--seed2-labels", type=Path, default=r2ad.DEFAULT_SEED2_LABELS)
    parser.add_argument("--seed2-base-root", type=Path, default=r2ad.DEFAULT_SEED2_BASE)
    parser.add_argument("--seed2-full-root", type=Path, default=r2ad.DEFAULT_SEED2_FULL)
    parser.add_argument("--path-root", type=Path, default=r2ad.DEFAULT_PATH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    data, input_receipt = r2ad.load_training(args)
    statistics = r2ad.fold_raw_statistics(data)
    records = []
    best_score = np.inf
    best_prediction: np.ndarray | None = None
    best_record: dict[str, Any] | None = None
    for mass in PROJECTION_MASSES:
        for seed_weights in SEED_PROJECTION_WEIGHTS:
            for penalty in BILINEAR_PENALTIES:
                for t600_fraction in T600_FORCE_FRACTIONS:
                    oof = np.full((132, 72, 3, len(ALPHAS)), np.nan)
                    for hold_id, fold_tuple in enumerate(r2ad.FOLDS):
                        hold = np.asarray(fold_tuple, dtype=int)
                        coefficient = coefficient_matrix(
                            r2ad.normal_equations(
                                statistics,
                                tuple(value for value in range(4) if value != hold_id),
                                mass,
                                seed_weights,
                                penalty,
                                t600_fraction,
                            )
                        )
                        oof[hold] = data["fixed"][hold][..., None] + np.einsum(
                            "natk,kq->natq",
                            data["design"][hold],
                            coefficient,
                            optimize=True,
                        )
                    scores = r2ad.vectorized_scores(oof, data)
                    for alpha_id, score in enumerate(scores):
                        record = {
                            "raw_gate_score": float(score),
                            "projection_mass": mass,
                            "seed_projection_weights": {
                                name: float(weight)
                                for name, weight in zip(
                                    r2ad.SEED_RANGES, seed_weights, strict=True
                                )
                            },
                            "bilinear_penalty": penalty,
                            "T600_force_fraction": t600_fraction,
                            "group_masses": r2ad.group_masses(t600_fraction),
                            "alpha": ALPHAS[alpha_id],
                        }
                        records.append(record)
                        if score < best_score:
                            best_score = float(score)
                            best_prediction = oof[..., alpha_id].copy()
                            best_record = record
        print(json.dumps({"projection_mass_complete": mass}), flush=True)
    if best_prediction is None or best_record is None:
        raise RuntimeError("R2AE boundary scan produced no candidate")
    metrics = r2ad.scalar_metrics(best_prediction, data)
    if abs(metrics["raw_gate_score"] - best_score) > 1.0e-10:
        raise ValueError("R2AE vectorized/scalar score mismatch")
    records.sort(key=lambda item: (item["raw_gate_score"], item["alpha"]))
    status = (
        "R2AE_FIXED_STEP32_DEVELOPMENT_GATE_PASSED"
        if metrics["passes_fixed_gate"]
        else "R2AE_FIXED_STEP32_DEVELOPMENT_GATE_FAILED"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "best_OOF_predicted_force_eV_A.npy"
    np.save(prediction_path, np.asarray(best_prediction, dtype="<f8"), allow_pickle=False)
    summary = {
        "format": "graphene_r2ae_seed012_step32_boundary_v1",
        "status": status,
        "scope": "fixed step-32 representation boundary audit; seed012 development training; 525 K unopened",
        "candidate_count": int(
            len(PROJECTION_MASSES)
            * len(SEED_PROJECTION_WEIGHTS)
            * len(BILINEAR_PENALTIES)
            * len(T600_FORCE_FRACTIONS)
            * len(ALPHAS)
        ),
        "best": {**best_record, "full_gate": metrics},
        "top_candidates": records[:300],
        "input_receipt": input_receipt,
        "prediction_sha256": file_sha256(prediction_path),
        "prediction_raw_sha256": r2r1.raw_array_sha256(best_prediction, "<f8"),
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(status + "\n")
    print(json.dumps({"status": status, "best": summary["best"]}, indent=2))


if __name__ == "__main__":
    main()
