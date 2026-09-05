#!/usr/bin/env python3
"""Forward-select sparse conservative quadratic PCA energy columns."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_graphene_r2aa_global_quadratic as r2aa
import evaluate_graphene_r2z_forward_bilinear as r2z
import fit_graphene_r2y_seed01_paired_readout as r2y
import graphene_r2r1_linear_readout as r2r1
import refine_graphene_r2z_step28 as refine
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from evaluate_graphene_r2w_nested_single_bilinear_selection import (
    solve_all_single_feature_coefficients,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_PATH_ROOT = BASE / "R2Z_forward_bilinear_development_32step_20260826"
DEFAULT_OUTPUT = BASE / "R2AB_sparse_quadratic_development_20260826"
PCA_WIDTH = 32
MASSES = (0.80, 0.84, 0.88)
SEED1_WEIGHTS = (0.35, 0.425, 0.50)
TAIL_PENALTIES = (0.20, 0.50, 1.00)
T600_FORCE_FRACTIONS = (0.65, 0.80, 0.90)
ALPHAS = tuple(float(value) for value in np.geomspace(1.0e-3, 3.0e-2, 15))


def projected_scores(
    predicted: np.ndarray, target: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    return r2z.projected_scores(predicted, target, coordinates)


def full_gate_candidate(
    data: dict[str, np.ndarray],
    design: np.ndarray,
    statistics: tuple[r2z.FoldRaw, ...],
    core_columns: np.ndarray,
    candidate_columns: np.ndarray,
    record: dict[str, Any],
) -> dict[str, Any]:
    ordered = np.concatenate((core_columns, candidate_columns))
    position = int(record["candidate_position"])
    oof = np.full_like(data["reference"], np.nan)
    for hold_id, fold_tuple in enumerate(r2y.FOLDS):
        hold = np.asarray(fold_tuple)
        train_folds = tuple(value for value in range(4) if value != hold_id)
        normalizer, gram, rhs = refine.normal_equations(
            statistics,
            train_folds,
            ordered,
            float(record["projection_mass"]),
            float(record["seed1_projection_weight"]),
            float(record["tail_penalty"]),
            float(record["T600_force_fraction"]),
        )
        core, off, _ = solve_all_single_feature_coefficients(
            normalizer,
            gram,
            rhs,
            float(record["alpha"]),
            core_width=len(core_columns),
        )
        coefficient = np.concatenate((core[:, position], [off[position]]))
        chosen = np.concatenate((core_columns, [candidate_columns[position]]))
        oof[hold] = data["fixed"][hold] + np.einsum(
            "natk,k->nat", design[hold][..., chosen], coefficient, optimize=False
        )
    metrics = r2y.metrics(
        oof,
        data["reference"],
        data["modes"],
        data["coordinates"],
        data["base"],
        np.arange(112),
    )
    return {
        **record,
        "selected_quadratic_position": int(
            candidate_columns[position] - 129
        ),
        "full_gate": metrics,
        "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
    }


def scan_step(
    data: dict[str, np.ndarray],
    design: np.ndarray,
    statistics: tuple[r2z.FoldRaw, ...],
    selected_quadratic: list[int],
) -> dict[str, Any]:
    remaining = np.asarray(
        [value for value in range(528) if value not in set(selected_quadratic)],
        dtype=int,
    )
    core_columns = np.r_[
        np.arange(129, dtype=int),
        129 + np.asarray(selected_quadratic, dtype=int),
    ].astype(int, copy=False)
    candidate_columns = 129 + remaining
    ordered = np.concatenate((core_columns, candidate_columns))
    mode_projected = np.einsum(
        "nat,natk->nk",
        np.conj(data["modes"]),
        design[:40],
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
            for penalty in TAIL_PENALTIES:
                for t600_fraction in T600_FORCE_FRACTIONS:
                    systems = []
                    for hold_id in range(4):
                        train_folds = tuple(
                            value for value in range(4) if value != hold_id
                        )
                        systems.append(
                            refine.normal_equations(
                                statistics,
                                train_folds,
                                ordered,
                                mass,
                                seed1_weight,
                                penalty,
                                t600_fraction,
                            )
                        )
                    for alpha in ALPHAS:
                        predicted = np.full((40, len(remaining)), np.nan + 0.0j)
                        for hold_id, fold_tuple in enumerate(r2y.FOLDS):
                            hold = np.asarray(fold_tuple)
                            e50_hold = hold[hold < 40]
                            normalizer, gram, rhs = systems[hold_id]
                            core, off, _ = solve_all_single_feature_coefficients(
                                normalizer,
                                gram,
                                rhs,
                                alpha,
                                core_width=len(core_columns),
                            )
                            predicted[e50_hold] = (
                                fixed_modes[e50_hold, None]
                                + mode_projected[e50_hold][:, core_columns] @ core
                                + mode_projected[e50_hold][:, candidate_columns]
                                * off[None, :]
                            )
                        scores = projected_scores(
                            predicted, target_modes, data["coordinates"]
                        )
                        take = min(4, len(scores))
                        for position in np.argpartition(scores, take - 1)[:take]:
                            provisional.append(
                                {
                                    "projected_score": float(scores[position]),
                                    "candidate_position": int(position),
                                    "candidate_quadratic_position": int(
                                        remaining[position]
                                    ),
                                    "projection_mass": mass,
                                    "seed1_projection_weight": seed1_weight,
                                    "tail_penalty": penalty,
                                    "T600_force_fraction": t600_fraction,
                                    "group_masses": refine.group_masses(
                                        t600_fraction
                                    ),
                                    "alpha": alpha,
                                }
                            )
    provisional.sort(
        key=lambda item: (
            item["projected_score"],
            item["candidate_quadratic_position"],
            item["alpha"],
        )
    )
    unique = []
    seen = set()
    for item in provisional:
        key = (
            item["candidate_position"],
            item["projection_mass"],
            item["seed1_projection_weight"],
            item["tail_penalty"],
            item["T600_force_fraction"],
            item["alpha"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
        if len(unique) == 120:
            break
    full = [
        full_gate_candidate(
            data, design, statistics, core_columns, candidate_columns, item
        )
        for item in unique
    ]
    full.sort(
        key=lambda item: (
            item["full_gate"]["raw_gate_score"],
            item["selected_quadratic_position"],
            item["alpha"],
        )
    )
    return {
        "starting_quadratic_positions": list(selected_quadratic),
        "remaining_candidate_count": len(remaining),
        "best": full[0],
        "top_full_gate_candidates": full,
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
    parser.add_argument("--max-forward-steps", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.max_forward_steps <= 20:
        raise ValueError("R2AB max forward steps must be 1..20")
    started = time.perf_counter()
    data, input_receipt = r2z.load_full_training(args)
    path_summary_path = args.path_root / "summary.json"
    path_summary = json.loads(path_summary_path.read_text())
    step32 = path_summary["steps"][31]
    selected_linear = list(step32["starting_bilinear_indices"]) + [
        int(step32["best"]["selected_bilinear_index"])
    ]
    linear_columns = np.r_[np.arange(65), 65 + np.asarray(selected_linear)]
    linear_force = data["design"][..., linear_columns]
    linear_energy, energy_receipt = r2aa.load_energy_design(args, selected_linear)
    q_energy, q_force, pca_receipt = r2aa.quadratic_basis(
        linear_energy, linear_force, PCA_WIDTH
    )
    if q_force.shape[-1] != 528:
        raise ValueError("R2AB quadratic candidate count changed")
    design = np.concatenate((linear_force, q_force), axis=-1)
    statistics = r2z.fold_raw_statistics(
        design, data["fixed"], data["reference"], data["modes"]
    )
    selected_quadratic: list[int] = []
    steps = []
    for step_id in range(args.max_forward_steps):
        record = scan_step(data, design, statistics, selected_quadratic)
        record["forward_step"] = step_id + 1
        steps.append(record)
        selected_quadratic.append(
            int(record["best"]["selected_quadratic_position"])
        )
        print(
            json.dumps(
                {
                    "forward_step_complete": step_id + 1,
                    "selected_quadratic_position": selected_quadratic[-1],
                    "raw_gate_score": record["best"]["full_gate"][
                        "raw_gate_score"
                    ],
                    "passes_fixed_gate": record["best"]["full_gate"][
                        "passes_fixed_gate"
                    ],
                }
            ),
            flush=True,
        )
        if record["best"]["full_gate"]["passes_fixed_gate"]:
            break
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "format": "graphene_r2ab_sparse_quadratic_development_v1",
        "status": (
            "R2AB_SPARSE_QUADRATIC_DEVELOPMENT_GATE_PASSED"
            if steps[-1]["best"]["full_gate"]["passes_fixed_gate"]
            else "R2AB_SPARSE_QUADRATIC_DEVELOPMENT_GATE_FAILED"
        ),
        "scope": "seed0 + seed1 + T300 + T600 development; 525 K unopened",
        "selected_linear_bilinear_indices": selected_linear,
        "PCA_receipt": pca_receipt,
        "selected_quadratic_positions": selected_quadratic,
        "steps": steps,
        "energy_design_input_sha256": energy_receipt,
        "path_summary_sha256": file_sha256(path_summary_path),
        "input_receipt": input_receipt,
        "energy_labels_used": False,
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(
        json.dumps(
            {"status": summary["status"], "selected": selected_quadratic},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
