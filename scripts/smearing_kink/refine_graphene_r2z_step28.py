#!/usr/bin/env python3
"""Refine objective weights for the fixed R2Z step-28 representation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import evaluate_graphene_r2z_forward_bilinear as r2z
import fit_graphene_r2y_seed01_paired_readout as r2y
import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_PATH_ROOT = BASE / "R2Z_forward_bilinear_development_32step_20260826"
DEFAULT_OUTPUT = BASE / "R2Z_step28_objective_refinement_20260826"
MASSES = tuple(float(value) for value in np.linspace(0.80, 0.90, 11))
SEED1_WEIGHTS = tuple(float(value) for value in np.linspace(0.35, 0.55, 9))
BILINEAR_PENALTIES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.75)
T600_FORCE_FRACTIONS = (0.55, 0.60, 0.65, 0.70, 0.75)
ALPHAS = tuple(float(value) for value in np.geomspace(1.5e-3, 1.0e-2, 21))
COMPONENTS = 72 * 3
FORCE_SCALE = 0.030
APRIME_SCALE = 0.015


def group_masses(t600_fraction: float) -> dict[str, float]:
    other = (1.0 - t600_fraction) / 3.0
    return {
        "E50_seed0": other,
        "E50_seed1": other,
        "T300": other,
        "T600": t600_fraction,
    }


def normal_equations(
    statistics: Sequence[r2z.FoldRaw],
    train_folds: Sequence[int],
    columns: np.ndarray,
    mass: float,
    seed1_weight: float,
    bilinear_penalty: float,
    t600_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = [statistics[int(value)] for value in train_folds]
    structure_count = sum(item.structure_count for item in chosen)
    square_sum = sum(
        (item.square_sum for item in chosen),
        start=np.zeros_like(chosen[0].square_sum),
    )[columns]
    scale = np.sqrt(square_sum / (structure_count * COMPONENTS))
    penalty = np.ones(len(columns))
    penalty[columns >= 65] = bilinear_penalty
    normalizer = scale * np.sqrt(penalty)
    subset = np.ix_(columns, columns)
    force_gram = np.zeros((len(columns), len(columns)))
    force_rhs = np.zeros(len(columns))
    masses = group_masses(t600_fraction)
    for group in r2y.GROUP_RANGES:
        count = sum(
            len(
                np.intersect1d(
                    np.asarray(r2y.FOLDS[int(fold)]), r2y.group_indices(group)
                )
            )
            for fold in train_folds
        )
        raw_gram = sum(
            (item.group_gram[group] for item in chosen),
            start=np.zeros_like(chosen[0].group_gram[group]),
        )[subset]
        raw_rhs = sum(
            (item.group_rhs[group] for item in chosen),
            start=np.zeros_like(chosen[0].group_rhs[group]),
        )[columns]
        weight = masses[group] / (count * COMPONENTS)
        force_gram += weight * raw_gram
        force_rhs += weight * raw_rhs
    projection_gram = np.zeros_like(force_gram)
    projection_rhs = np.zeros_like(force_rhs)
    for name, weight in (("seed0", 1.0 - seed1_weight), ("seed1", seed1_weight)):
        count = 5 * len(chosen)
        raw_gram = sum(
            (item.projection_gram[name] for item in chosen),
            start=np.zeros_like(chosen[0].projection_gram[name]),
        )[subset]
        raw_rhs = sum(
            (item.projection_rhs[name] for item in chosen),
            start=np.zeros_like(chosen[0].projection_rhs[name]),
        )[columns]
        projection_gram += weight * raw_gram / (2 * count)
        projection_rhs += weight * raw_rhs / (2 * count)
    gram = (1.0 - mass) * force_gram + mass * projection_gram
    rhs = (1.0 - mass) * force_rhs + mass * projection_rhs
    gram = gram / normalizer[:, None] / normalizer[None, :]
    rhs = rhs / normalizer
    return normalizer, 0.5 * (gram + gram.T), rhs


def coefficient_matrix(
    system: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    normalizer, gram, rhs = system
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo(float).eps * len(normalizer) * max(
        float(eigenvalues[-1]), 1.0
    )
    if eigenvalues[0] < -10.0 * tolerance:
        raise ValueError("R2Z refinement Gram is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ rhs
    normalized = eigenvectors @ (
        projected_rhs[:, None] / (eigenvalues[:, None] + np.asarray(ALPHAS)[None, :])
    )
    return FORCE_SCALE * normalized / normalizer[:, None]


def full_gate(
    data: dict[str, np.ndarray],
    statistics: Sequence[r2z.FoldRaw],
    columns: np.ndarray,
    record: dict[str, Any],
) -> dict[str, Any]:
    oof = np.full_like(data["reference"], np.nan)
    alpha_id = ALPHAS.index(float(record["alpha"]))
    for hold_id, fold_tuple in enumerate(r2y.FOLDS):
        hold = np.asarray(fold_tuple)
        train_folds = tuple(value for value in range(4) if value != hold_id)
        coefficient = coefficient_matrix(
            normal_equations(
                statistics,
                train_folds,
                columns,
                float(record["projection_mass"]),
                float(record["seed1_projection_weight"]),
                float(record["bilinear_penalty"]),
                float(record["T600_force_fraction"]),
            )
        )[:, alpha_id]
        oof[hold] = data["fixed"][hold] + np.einsum(
            "natk,k->nat",
            data["design"][hold][..., columns],
            coefficient,
            optimize=False,
        )
    return {
        **record,
        "full_gate": r2y.metrics(
            oof,
            data["reference"],
            data["modes"],
            data["coordinates"],
            data["base"],
            np.arange(112),
        ),
        "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
    }


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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    data, input_receipt = r2z.load_full_training(args)
    statistics = r2z.fold_raw_statistics(
        data["design"], data["fixed"], data["reference"], data["modes"]
    )
    path_summary_path = args.path_root / "summary.json"
    path_summary = json.loads(path_summary_path.read_text())
    step = path_summary["steps"][27]
    selected = list(step["starting_bilinear_indices"]) + [
        int(step["best"]["selected_bilinear_index"])
    ]
    if len(selected) != 60 or len(set(selected)) != 60:
        raise ValueError("R2Z step-28 representation changed")
    columns = np.r_[np.arange(65), 65 + np.asarray(selected)]
    mode_projected = np.einsum(
        "nat,natk->nk",
        np.conj(data["modes"]),
        data["design"][:40][..., columns],
        optimize=True,
    )
    fixed_modes = np.einsum(
        "nat,nat->n",
        np.conj(data["modes"]),
        data["base"] + data["fixed"][:40],
    )
    target_modes = np.einsum(
        "nat,nat->n",
        np.conj(data["modes"]),
        data["base"] + data["reference"][:40],
    )
    provisional = []
    for mass in MASSES:
        for seed1_weight in SEED1_WEIGHTS:
            for penalty in BILINEAR_PENALTIES:
                for t600_fraction in T600_FORCE_FRACTIONS:
                    predicted = np.full((40, len(ALPHAS)), np.nan + 0.0j)
                    for hold_id, fold_tuple in enumerate(r2y.FOLDS):
                        hold = np.asarray(fold_tuple)
                        e50_hold = hold[hold < 40]
                        train_folds = tuple(
                            value for value in range(4) if value != hold_id
                        )
                        current = coefficient_matrix(
                            normal_equations(
                                statistics,
                                train_folds,
                                columns,
                                mass,
                                seed1_weight,
                                penalty,
                                t600_fraction,
                            )
                        )
                        predicted[e50_hold] = (
                            fixed_modes[e50_hold, None]
                            + mode_projected[e50_hold] @ current
                        )
                    scores = r2z.projected_scores(
                        predicted, target_modes, data["coordinates"]
                    )
                    for alpha_id in np.argpartition(scores, 2)[:3]:
                        provisional.append(
                            {
                                "projected_score": float(scores[alpha_id]),
                                "projection_mass": mass,
                                "seed1_projection_weight": seed1_weight,
                                "bilinear_penalty": penalty,
                                "T600_force_fraction": t600_fraction,
                                "group_masses": group_masses(t600_fraction),
                                "alpha": ALPHAS[int(alpha_id)],
                            }
                        )
    provisional.sort(key=lambda item: (item["projected_score"], item["alpha"]))
    full = [full_gate(data, statistics, columns, item) for item in provisional[:800]]
    full.sort(key=lambda item: (item["full_gate"]["raw_gate_score"], item["alpha"]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "format": "graphene_r2z_step28_objective_refinement_v1",
        "status": (
            "R2Z_STEP28_DEVELOPMENT_GATE_PASSED"
            if full[0]["full_gate"]["passes_fixed_gate"]
            else "R2Z_STEP28_DEVELOPMENT_GATE_FAILED"
        ),
        "scope": "fixed step-28 representation; training-domain refinement; 525 K unopened",
        "selected_bilinear_indices": selected,
        "candidate_count": int(
            len(MASSES)
            * len(SEED1_WEIGHTS)
            * len(BILINEAR_PENALTIES)
            * len(T600_FORCE_FRACTIONS)
            * len(ALPHAS)
        ),
        "masses": list(MASSES),
        "seed1_weights": list(SEED1_WEIGHTS),
        "bilinear_penalties": list(BILINEAR_PENALTIES),
        "T600_force_fractions": list(T600_FORCE_FRACTIONS),
        "alphas": list(ALPHAS),
        "best": full[0],
        "top_full_gate_candidates": full,
        "path_summary_sha256": file_sha256(path_summary_path),
        "input_receipt": input_receipt,
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps({"status": summary["status"], "best": full[0]}, indent=2))


if __name__ == "__main__":
    main()
