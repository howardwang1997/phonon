#!/usr/bin/env python3
"""Forward-select full bilinear columns on the R2Y development training set.

The complete feature-selection path is development-only.  E50 seed0/seed1
and the 300/600 K auxiliary structures are training data; the planned 525 K
off-policy validation set remains unopened.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import evaluate_graphene_r2y_seed01_grid as r2y_grid
import fit_graphene_r2y_seed01_paired_readout as r2y
import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import (
    DEFAULT_AGGREGATE,
    canonical_json_bytes,
    file_sha256,
)
from evaluate_graphene_r2w_nested_single_bilinear_selection import (
    solve_all_single_feature_coefficients,
)
from graphene_r2x_paired_readout import SELECTED_BILINEAR_INDICES


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_R2Y_FAILED = BASE / "R2Y_seed01_paired_readout_freeze_20260826"
DEFAULT_SEED1_SELECTED = BASE / "R2Y_seed1_selected_design_20260826"
DEFAULT_THERMAL_FULL = BASE / "R2T_full_bilinear_materialization_20260826"
DEFAULT_SEED1_FULL = BASE / "R2Y_seed1_full_bilinear_20260826"
DEFAULT_OUTPUT = BASE / "R2Z_forward_bilinear_development_20260826"
COMPONENTS = 72 * 3
FORCE_SCALE = 0.030
APRIME_SCALE = 0.015
MASSES = (0.30, 0.50, 0.70, 0.85, 0.95)
SEED1_WEIGHTS = (0.25, 0.375, 0.50, 0.625, 0.75)
PENALTIES = (1.0, 10.0, 100.0)
ALPHAS = tuple(float(value) for value in np.logspace(-6.0, 1.0, 15))
CROSS32 = tuple(int(value) for value in SELECTED_BILINEAR_INDICES[:32])


@dataclass(frozen=True)
class FoldRaw:
    square_sum: np.ndarray
    structure_count: int
    group_gram: dict[str, np.ndarray]
    group_rhs: dict[str, np.ndarray]
    projection_gram: dict[str, np.ndarray]
    projection_rhs: dict[str, np.ndarray]


def verified_array(root: Path, filename: str, receipt: dict[str, Any], key: str) -> np.ndarray:
    path = root / filename
    if file_sha256(path) != receipt["array_file_sha256"][key]:
        raise ValueError(f"full bilinear {key} file differs from receipt")
    value = np.load(path, allow_pickle=False)
    if r2r1.raw_array_sha256(value, "<f8") != receipt["array_raw_sha256"][key]:
        raise ValueError(f"full bilinear {key} raw array differs from receipt")
    return np.asarray(value, np.float64)


def load_full_training(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = r2y_grid.load_training(
        args.aggregate, args.failed_root, args.seed1_selected_root
    )
    thermal_receipt_path = args.thermal_full_root / "receipt.json"
    seed1_receipt_path = args.seed1_full_root / "receipt.json"
    thermal_receipt = json.loads(thermal_receipt_path.read_text())
    seed1_receipt = json.loads(seed1_receipt_path.read_text())
    if thermal_receipt.get("column_count") != 512 or thermal_receipt.get(
        "structure_count"
    ) != 92:
        raise ValueError("thermal full bilinear receipt schema changed")
    if seed1_receipt.get("status") != "R2Y_SEED1_FULL_BILINEAR_COMPLETE":
        raise ValueError("seed1 full bilinear receipt is incomplete")
    thermal_force = verified_array(
        args.thermal_full_root,
        "bilinear_force_design_eV_A.npy",
        thermal_receipt,
        "force",
    )
    seed1_force = verified_array(
        args.seed1_full_root,
        "bilinear_force_design_eV_A.npy",
        seed1_receipt,
        "force",
    )
    full_bilinear = np.concatenate(
        (thermal_force[:20], seed1_force, thermal_force[20:])
    )
    selected = np.take(full_bilinear, SELECTED_BILINEAR_INDICES, axis=-1)
    replay = float(np.max(np.abs(selected - data["design"][..., 65:])))
    if replay > 2.0e-10:
        raise ValueError(f"full/selected bilinear replay failed: {replay:.3e}")
    data["design"] = np.concatenate((data["design"][..., :65], full_bilinear), axis=-1)
    receipt = {
        "thermal_receipt_sha256": file_sha256(thermal_receipt_path),
        "seed1_receipt_sha256": file_sha256(seed1_receipt_path),
        "thermal_force_sha256": file_sha256(
            args.thermal_full_root / "bilinear_force_design_eV_A.npy"
        ),
        "seed1_force_sha256": file_sha256(
            args.seed1_full_root / "bilinear_force_design_eV_A.npy"
        ),
        "selected99_max_abs_replay_eV_A": replay,
    }
    return data, receipt


def fold_raw_statistics(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    modes: np.ndarray,
) -> tuple[FoldRaw, ...]:
    response = (reference - fixed) / FORCE_SCALE
    output = []
    for fold_tuple in r2y.FOLDS:
        fold = np.asarray(fold_tuple, dtype=int)
        square_sum = np.sum(np.square(design[fold]), axis=(0, 1, 2))
        group_gram = {}
        group_rhs = {}
        for group in r2y.GROUP_RANGES:
            members = np.intersect1d(fold, r2y.group_indices(group))
            matrix = design[members].reshape(-1, design.shape[-1])
            target = response[members].reshape(-1)
            group_gram[group] = matrix.T @ matrix
            group_rhs[group] = matrix.T @ target
        projection_gram = {}
        projection_rhs = {}
        for name, start, stop in (("seed0", 0, 20), ("seed1", 20, 40)):
            members = np.intersect1d(fold, np.arange(start, stop))
            projected_design = np.einsum(
                "nat,natk->nk", np.conj(modes[members]), design[members], optimize=True
            )
            projected_target = np.einsum(
                "nat,nat->n",
                np.conj(modes[members]),
                reference[members] - fixed[members],
            )
            projected_design *= FORCE_SCALE / APRIME_SCALE
            projected_target /= APRIME_SCALE
            projection_gram[name] = (
                projected_design.real.T @ projected_design.real
                + projected_design.imag.T @ projected_design.imag
            )
            projection_rhs[name] = (
                projected_design.real.T @ projected_target.real
                + projected_design.imag.T @ projected_target.imag
            )
        output.append(
            FoldRaw(
                square_sum=square_sum,
                structure_count=len(fold),
                group_gram=group_gram,
                group_rhs=group_rhs,
                projection_gram=projection_gram,
                projection_rhs=projection_rhs,
            )
        )
    return tuple(output)


def normal_equations(
    statistics: Sequence[FoldRaw],
    train_folds: Sequence[int],
    columns: np.ndarray,
    mass: float,
    seed1_weight: float,
    bilinear_penalty: float,
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
    if np.any(normalizer <= np.max(normalizer) * 1.0e-12):
        raise ValueError("R2Z forward scan contains a zero column")
    force_gram = np.zeros((len(columns), len(columns)))
    force_rhs = np.zeros(len(columns))
    subset = np.ix_(columns, columns)
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
        weight = r2y.GROUP_MASSES[group] / (count * COMPONENTS)
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


def projected_scores(
    predicted: np.ndarray, target: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    ratios = []
    all_error = []
    for start, stop in ((0, 20), (20, 40)):
        error = predicted[start:stop] - target[start:stop, None]
        rms = 1000.0 * np.sqrt(np.mean(np.abs(error) ** 2, axis=0))
        ratios.append(rms / 15.0)
        denominator = float(
            np.vdot(coordinates[start:stop], coordinates[start:stop]).real
        )
        target_slope = r2y.restoring_slope(
            coordinates[start:stop], target[start:stop]
        )
        error_slope = -np.real(
            np.sum(
                np.conj(coordinates[start:stop])[:, None] * error,
                axis=0,
            )
        ) / denominator
        ratios.append(np.abs(error_slope / target_slope) / 0.05)
        all_error.append(error)
    combined = 1000.0 * np.sqrt(
        np.mean(np.abs(np.concatenate(all_error)) ** 2, axis=0)
    )
    ratios.append(combined / 15.0)
    return np.maximum.reduce(ratios)


def solve_combo(
    statistics: Sequence[FoldRaw],
    train_folds: Sequence[int],
    columns: np.ndarray,
    core_width: int,
    mass: float,
    seed1_weight: float,
    penalty: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    normalizer, gram, rhs = normal_equations(
        statistics, train_folds, columns, mass, seed1_weight, penalty
    )
    core, off, _ = solve_all_single_feature_coefficients(
        normalizer, gram, rhs, alpha, core_width=core_width
    )
    return core, off


def full_gate_for_candidate(
    data: dict[str, np.ndarray],
    statistics: Sequence[FoldRaw],
    core_global: np.ndarray,
    candidates_global: np.ndarray,
    record: dict[str, Any],
) -> dict[str, Any]:
    ordered = np.concatenate((core_global, candidates_global))
    core_width = len(core_global)
    candidate_position = int(record["candidate_position"])
    oof = np.full_like(data["reference"], np.nan)
    for hold_id, fold_tuple in enumerate(r2y.FOLDS):
        hold = np.asarray(fold_tuple)
        train_folds = tuple(value for value in range(4) if value != hold_id)
        core, off = solve_combo(
            statistics,
            train_folds,
            ordered,
            core_width,
            float(record["projection_mass"]),
            float(record["seed1_projection_weight"]),
            float(record["bilinear_penalty"]),
            float(record["alpha"]),
        )
        coefficient = np.concatenate(
            (core[:, candidate_position], [off[candidate_position]])
        )
        chosen_design = data["design"][hold][
            ..., np.concatenate((core_global, [candidates_global[candidate_position]]))
        ]
        oof[hold] = data["fixed"][hold] + np.einsum(
            "natk,k->nat", chosen_design, coefficient, optimize=False
        )
    metric = r2y.metrics(
        oof,
        data["reference"],
        data["modes"],
        data["coordinates"],
        data["base"],
        np.arange(112),
    )
    return {
        **record,
        "selected_bilinear_index": int(
            candidates_global[candidate_position] - 65
        ),
        "full_gate": metric,
        "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
    }


def scan_step(
    data: dict[str, np.ndarray],
    statistics: Sequence[FoldRaw],
    selected_bilinear: list[int],
) -> dict[str, Any]:
    remaining = np.asarray(
        [value for value in range(512) if value not in set(selected_bilinear)],
        dtype=int,
    )
    core_global = np.r_[np.arange(65), 65 + np.asarray(selected_bilinear, dtype=int)]
    candidates_global = 65 + remaining
    ordered = np.concatenate((core_global, candidates_global))
    core_width = len(core_global)
    mode_projected = np.einsum(
        "nat,natk->nk",
        np.conj(data["modes"]),
        data["design"][:40],
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
            for penalty in PENALTIES:
                systems = []
                for hold_id in range(4):
                    train_folds = tuple(
                        value for value in range(4) if value != hold_id
                    )
                    systems.append(
                        normal_equations(
                            statistics,
                            train_folds,
                            ordered,
                            mass,
                            seed1_weight,
                            penalty,
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
                            core_width=core_width,
                        )
                        predicted[e50_hold] = (
                            fixed_modes[e50_hold, None]
                            + mode_projected[e50_hold][:, core_global] @ core
                            + mode_projected[e50_hold][:, candidates_global]
                            * off[None, :]
                        )
                    scores = projected_scores(
                        predicted, target_modes, data["coordinates"]
                    )
                    take = min(3, len(scores))
                    positions = np.argpartition(scores, take - 1)[:take]
                    for position in positions:
                        provisional.append(
                            {
                                "projected_score": float(scores[position]),
                                "candidate_position": int(position),
                                "candidate_bilinear_index": int(remaining[position]),
                                "projection_mass": mass,
                                "seed1_projection_weight": seed1_weight,
                                "bilinear_penalty": penalty,
                                "alpha": alpha,
                            }
                        )
    provisional.sort(
        key=lambda item: (
            item["projected_score"],
            item["candidate_bilinear_index"],
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
            item["bilinear_penalty"],
            item["alpha"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
        if len(unique) == 60:
            break
    full = [
        full_gate_for_candidate(
            data, statistics, core_global, candidates_global, item
        )
        for item in unique
    ]
    full.sort(
        key=lambda item: (
            item["full_gate"]["raw_gate_score"],
            item["selected_bilinear_index"],
            item["alpha"],
        )
    )
    return {
        "starting_bilinear_indices": list(selected_bilinear),
        "candidate_count": len(remaining),
        "grid_combination_count": int(
            len(MASSES)
            * len(SEED1_WEIGHTS)
            * len(PENALTIES)
            * len(ALPHAS)
            * len(remaining)
        ),
        "top_full_gate_candidates": full,
        "best": full[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--failed-root", type=Path, default=DEFAULT_R2Y_FAILED)
    parser.add_argument(
        "--seed1-selected-root", type=Path, default=DEFAULT_SEED1_SELECTED
    )
    parser.add_argument("--thermal-full-root", type=Path, default=DEFAULT_THERMAL_FULL)
    parser.add_argument("--seed1-full-root", type=Path, default=DEFAULT_SEED1_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-forward-steps", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_forward_steps <= 32:
        raise ValueError("max forward steps must be between 1 and 32")
    started = time.perf_counter()
    data, input_receipt = load_full_training(args)
    statistics = fold_raw_statistics(
        data["design"], data["fixed"], data["reference"], data["modes"]
    )
    selected = list(CROSS32)
    steps = []
    for step in range(args.max_forward_steps):
        record = scan_step(data, statistics, selected)
        record["forward_step"] = step + 1
        steps.append(record)
        selected.append(int(record["best"]["selected_bilinear_index"]))
        print(
            json.dumps(
                {
                    "forward_step_complete": step + 1,
                    "selected_bilinear_index": selected[-1],
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
        "format": "graphene_r2z_forward_bilinear_development_v1",
        "status": (
            "R2Z_FORWARD_DEVELOPMENT_GATE_PASSED"
            if steps[-1]["best"]["full_gate"]["passes_fixed_gate"]
            else "R2Z_FORWARD_DEVELOPMENT_GATE_FAILED"
        ),
        "scope": "seed0 + seed1 + T300 + T600 development; 525 K unopened",
        "initial_bilinear_indices": list(CROSS32),
        "final_bilinear_indices": selected,
        "masses": list(MASSES),
        "seed1_projection_weights": list(SEED1_WEIGHTS),
        "bilinear_penalties": list(PENALTIES),
        "alphas": list(ALPHAS),
        "steps": steps,
        "input_receipt": input_receipt,
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps({"status": summary["status"], "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
