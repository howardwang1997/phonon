#!/usr/bin/env python3
"""Exhaustive full-gate scan for the fixed R2Z step-28 representation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_graphene_r2z_forward_bilinear as r2z
import fit_graphene_r2y_seed01_paired_readout as r2y
import graphene_r2r1_linear_readout as r2r1
import refine_graphene_r2z_step28 as refine
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_PATH_ROOT = BASE / "R2Z_forward_bilinear_development_32step_20260826"
DEFAULT_OUTPUT = BASE / "R2Z_step28_full_gate_scan_20260826"
MASSES = (0.81, 0.82, 0.83, 0.84, 0.85, 0.86)
SEED1_WEIGHTS = (0.35, 0.375, 0.40, 0.425, 0.45)
BILINEAR_PENALTIES = (0.25, 0.30, 0.40, 0.50, 0.75)
T600_FORCE_FRACTIONS = (0.78, 0.82, 0.85, 0.88, 0.92)
ALPHAS = tuple(float(value) for value in np.geomspace(6.0e-3, 1.5e-2, 31))
COMPONENTS = 72 * 3
FORCE_SCALE = 0.030


def coefficient_matrix(
    system: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    normalizer, gram, rhs = system
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo(float).eps * len(normalizer) * max(
        float(eigenvalues[-1]), 1.0
    )
    if eigenvalues[0] < -10.0 * tolerance:
        raise ValueError("R2Z full-gate Gram is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ rhs
    normalized = eigenvectors @ (
        projected_rhs[:, None] / (eigenvalues[:, None] + np.asarray(ALPHAS)[None, :])
    )
    return FORCE_SCALE * normalized / normalizer[:, None]


def vectorized_full_scores(
    predicted: np.ndarray, data: dict[str, np.ndarray]
) -> np.ndarray:
    candidate_count = predicted.shape[-1]
    if predicted.shape != (112, 72, 3, candidate_count):
        raise ValueError("R2Z full-gate prediction shape changed")
    error = predicted - data["reference"][..., None]
    ratios = []
    for group in r2y.GROUP_RANGES:
        members = r2y.group_indices(group)
        current = error[members]
        rmse = 1000.0 * np.sqrt(np.mean(np.square(current), axis=(0, 1, 2)))
        maximum = 1000.0 * np.max(np.abs(current), axis=(0, 1, 2))
        ratios.extend((rmse / 30.0, maximum / 200.0))
    all_mode_error = []
    for start, stop in ((0, 20), (20, 40)):
        mode_error = np.einsum(
            "nat,natq->nq",
            np.conj(data["modes"][start:stop]),
            error[start:stop],
            optimize=True,
        )
        rms = 1000.0 * np.sqrt(np.mean(np.abs(mode_error) ** 2, axis=0))
        ratios.append(rms / 15.0)
        coordinates = data["coordinates"][start:stop]
        denominator = float(np.vdot(coordinates, coordinates).real)
        target_modes = np.einsum(
            "nat,nat->n",
            np.conj(data["modes"][start:stop]),
            data["base"][start:stop] + data["reference"][start:stop],
        )
        target_slope = r2y.restoring_slope(coordinates, target_modes)
        error_slope = -np.real(
            np.sum(np.conj(coordinates)[:, None] * mode_error, axis=0)
        ) / denominator
        ratios.append(np.abs(error_slope / target_slope) / 0.05)
        all_mode_error.append(mode_error)
    combined = 1000.0 * np.sqrt(
        np.mean(np.abs(np.concatenate(all_mode_error)) ** 2, axis=0)
    )
    ratios.append(combined / 15.0)
    return np.maximum.reduce(ratios)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=r2z.DEFAULT_AGGREGATE)
    parser.add_argument("--failed-root", type=Path, default=r2z.DEFAULT_R2Y_FAILED)
    parser.add_argument(
        "--seed1-selected-root", type=Path, default=r2z.DEFAULT_SEED1_SELECTED
    )
    parser.add_argument("--thermal-full-root", type=Path, default=r2z.DEFAULT_THERMAL_FULL)
    parser.add_argument("--seed1-full-root", type=Path, default=r2z.DEFAULT_SEED1_FULL)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--path-step", type=int, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    data, input_receipt = r2z.load_full_training(args)
    statistics = r2z.fold_raw_statistics(
        data["design"], data["fixed"], data["reference"], data["modes"]
    )
    path_summary_path = args.path_root / "summary.json"
    path_summary = json.loads(path_summary_path.read_text())
    path_steps = tuple(args.path_step) if args.path_step else (28,)
    if len(set(path_steps)) != len(path_steps) or any(
        not 1 <= value <= len(path_summary["steps"]) for value in path_steps
    ):
        raise ValueError("R2Z requested path steps are invalid")
    records = []
    best_score = np.inf
    best_prediction: np.ndarray | None = None
    best_record: dict[str, Any] | None = None
    selected_by_step: dict[int, list[int]] = {}
    for path_step in path_steps:
        step = path_summary["steps"][path_step - 1]
        selected = list(step["starting_bilinear_indices"]) + [
            int(step["best"]["selected_bilinear_index"])
        ]
        if len(selected) != 32 + path_step or len(set(selected)) != len(selected):
            raise ValueError(f"R2Z step-{path_step} representation changed")
        selected_by_step[path_step] = selected
        columns = np.r_[np.arange(65), 65 + np.asarray(selected)]
        for mass in MASSES:
            for seed1_weight in SEED1_WEIGHTS:
                for penalty in BILINEAR_PENALTIES:
                    for t600_fraction in T600_FORCE_FRACTIONS:
                        oof = np.full((112, 72, 3, len(ALPHAS)), np.nan)
                        for hold_id, fold_tuple in enumerate(r2y.FOLDS):
                            hold = np.asarray(fold_tuple)
                            train_folds = tuple(
                                value for value in range(4) if value != hold_id
                            )
                            coefficient = coefficient_matrix(
                                refine.normal_equations(
                                    statistics,
                                    train_folds,
                                    columns,
                                    mass,
                                    seed1_weight,
                                    penalty,
                                    t600_fraction,
                                )
                            )
                            matrix = data["design"][hold][..., columns].reshape(
                                -1, len(columns)
                            )
                            prediction = (
                                data["fixed"][hold].reshape(-1, 1)
                                + matrix @ coefficient
                            )
                            oof[hold] = prediction.reshape(
                                len(hold), 72, 3, len(ALPHAS)
                            )
                        if not np.all(np.isfinite(oof)):
                            raise RuntimeError("R2Z full-gate OOF coverage failed")
                        scores = vectorized_full_scores(oof, data)
                        alpha_id = int(np.argmin(scores))
                        record = {
                            "path_step": path_step,
                            "raw_gate_score": float(scores[alpha_id]),
                            "projection_mass": mass,
                            "seed1_projection_weight": seed1_weight,
                            "bilinear_penalty": penalty,
                            "T600_force_fraction": t600_fraction,
                            "group_masses": refine.group_masses(t600_fraction),
                            "alpha": ALPHAS[alpha_id],
                        }
                        records.append(record)
                        if scores[alpha_id] < best_score:
                            best_score = float(scores[alpha_id])
                            best_prediction = oof[..., alpha_id].copy()
                            best_record = record
    if best_prediction is None or best_record is None:
        raise RuntimeError("R2Z full-gate scan produced no candidate")
    best_metrics = r2y.metrics(
        best_prediction,
        data["reference"],
        data["modes"],
        data["coordinates"],
        data["base"],
        np.arange(112),
    )
    if abs(best_metrics["raw_gate_score"] - best_score) > 1.0e-10:
        raise ValueError("vectorized/scalar R2Z gate score mismatch")
    records.sort(key=lambda item: (item["raw_gate_score"], item["alpha"]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.save(
        output / "best_OOF_predicted_force_eV_A.npy",
        np.asarray(best_prediction, dtype="<f8"),
        allow_pickle=False,
    )
    summary = {
        "format": "graphene_r2z_multistep_exhaustive_full_gate_scan_v1",
        "status": (
            "R2Z_STEP28_FULL_DEVELOPMENT_GATE_PASSED"
            if best_metrics["passes_fixed_gate"]
            else "R2Z_STEP28_FULL_DEVELOPMENT_GATE_FAILED"
        ),
        "scope": "fixed forward-path representations; every candidate receives full OOF gate; 525 K unopened",
        "path_steps": list(path_steps),
        "selected_bilinear_indices": selected_by_step[int(best_record["path_step"])],
        "candidate_count": int(
            len(path_steps)
            * len(MASSES)
            * len(SEED1_WEIGHTS)
            * len(BILINEAR_PENALTIES)
            * len(T600_FORCE_FRACTIONS)
            * len(ALPHAS)
        ),
        "best": {**best_record, "full_gate": best_metrics},
        "top_system_candidates": records[:200],
        "best_OOF_raw_sha256": r2r1.raw_array_sha256(best_prediction, "<f8"),
        "path_summary_sha256": file_sha256(path_summary_path),
        "input_receipt": input_receipt,
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps({"status": summary["status"], "best": summary["best"]}, indent=2))


if __name__ == "__main__":
    main()
