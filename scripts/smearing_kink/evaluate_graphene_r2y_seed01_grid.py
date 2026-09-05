#!/usr/bin/env python3
"""Development-only hyperparameter and subset scan for the R2Y training set.

E50 seed0 and seed1 are both training data here.  The script keeps the 525 K
off-policy set unopened and uses fast projected-mode screening before computing
the full force gate for the best candidates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

import fit_graphene_r2y_seed01_paired_readout as r2y
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
DEFAULT_R2Y_FAILED = BASE / "R2Y_seed01_paired_readout_freeze_20260826"
DEFAULT_SEED1_DESIGN = BASE / "R2Y_seed1_selected_design_20260826"
DEFAULT_OUTPUT = BASE / "R2Y_seed01_grid_20260826"
FORCE_SCALE = 0.030
APRIME_SCALE = 0.015
COMPONENTS = 72 * 3

REPRESENTATIONS: dict[str, np.ndarray] = {
    "base65": np.arange(65),
    "base_plus_plain_diagonal16": np.r_[np.arange(65), np.arange(65, 81)],
    "base_plus_modulated_diagonal16": np.r_[np.arange(65), np.arange(81, 97)],
    "cross32": np.arange(97),
    "cross32_plus_plain_pair": np.arange(98),
    "cross32_plus_modulated_pair": np.r_[np.arange(97), 98],
    "paired99": np.arange(99),
}
PROJECTION_MASSES = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
)
ALPHAS = tuple(float(value) for value in np.logspace(-6.0, 0.0, 25))
BILINEAR_PENALTIES = (1.0, 10.0, 100.0)
SEED1_PROJECTION_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def load_training(
    aggregate: Path, failed_root: Path, seed1_design_root: Path
) -> dict[str, np.ndarray]:
    reference92, aprime0, _ = _parse_whitelist(r2r1.RECOMMENDED_THERMAL92)
    seed1 = read(r2y.SEED1_LABEL_PATH, index=":")
    if len(seed1) != 20:
        raise ValueError("seed1 label count changed")
    reference1 = np.stack([item.arrays["REF_forces"] for item in seed1])
    mode1 = np.stack(
        [
            item.arrays["APRIME_mode_real"]
            + 1.0j * item.arrays["APRIME_mode_imag"]
            for item in seed1
        ]
    )
    coordinate1 = np.asarray(
        [
            complex(
                item.info["APRIME_coordinate_real_A"],
                item.info["APRIME_coordinate_imag_A"],
            )
            for item in seed1
        ]
    )
    base1 = np.stack(
        [
            item.arrays["FOUNDATION_BASE_forces"]
            + item.arrays["FROZEN_Q6_forces"]
            for item in seed1
        ]
    )
    with np.load(failed_root / "frozen_readout.npz", allow_pickle=False) as arrays:
        design = np.asarray(arrays["thermal_force_design_eV_A"], np.float64)
    with np.load(aggregate, allow_pickle=False) as arrays:
        fixed92 = np.asarray(arrays["thermal_fixed_force_eV_A"], np.float64)
    with np.load(
        seed1_design_root / "seed1_selected_design.npz", allow_pickle=False
    ) as arrays:
        fixed1 = np.asarray(arrays["fixed_force_eV_A"], np.float64)
    fixed = np.concatenate((fixed92[:20], fixed1, fixed92[20:]))
    reference = np.concatenate((reference92[:20], reference1, reference92[20:]))
    modes = np.concatenate(
        (aprime0.mode_real + 1.0j * aprime0.mode_imag, mode1)
    )
    coordinates = np.concatenate(
        (aprime0.coordinates[:, 0] + 1.0j * aprime0.coordinates[:, 1], coordinate1)
    )
    base = np.concatenate(
        (
            aprime0.foundation_base_force_eV_A + aprime0.frozen_q6_force_eV_A,
            base1,
        )
    )
    if design.shape != (112, 72, 3, 99):
        raise ValueError("R2Y failed-design shape changed")
    return {
        "design": design,
        "fixed": fixed,
        "reference": reference,
        "modes": modes,
        "coordinates": coordinates,
        "base": base,
    }


def raw_system(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    modes: np.ndarray,
    train: np.ndarray,
    projection_mass: float,
    bilinear_penalty: float,
    seed1_projection_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = design.shape[-1]
    response = (reference - fixed) / FORCE_SCALE
    scale = np.sqrt(
        np.sum(np.square(design[train]), axis=(0, 1, 2))
        / (len(train) * COMPONENTS)
    )
    penalty = np.ones(width)
    if width > 65:
        penalty[65:] = bilinear_penalty
    normalizer = scale * np.sqrt(penalty)
    if np.any(normalizer <= np.max(normalizer) * 1.0e-12):
        raise ValueError("zero design column in R2Y grid")

    force_gram = np.zeros((width, width))
    force_rhs = np.zeros(width)
    for group in r2y.GROUP_RANGES:
        members = np.intersect1d(train, r2y.group_indices(group))
        matrix = design[members].reshape(-1, width)
        target = response[members].reshape(-1)
        weight = r2y.GROUP_MASSES[group] / (len(members) * COMPONENTS)
        force_gram += weight * (matrix.T @ matrix)
        force_rhs += weight * (matrix.T @ target)

    projection_gram = np.zeros((width, width))
    projection_rhs = np.zeros(width)
    for seed_start, seed_stop, seed_weight in (
        (0, 20, 1.0 - seed1_projection_weight),
        (20, 40, seed1_projection_weight),
    ):
        e50 = np.intersect1d(train, np.arange(seed_start, seed_stop))
        projected_design = np.einsum(
            "nat,natk->nk", np.conj(modes[e50]), design[e50], optimize=True
        )
        projected_target = np.einsum(
            "nat,nat->n", np.conj(modes[e50]), reference[e50] - fixed[e50]
        )
        projected_design *= FORCE_SCALE / APRIME_SCALE
        projected_target /= APRIME_SCALE
        projection_gram += seed_weight * (
            projected_design.real.T @ projected_design.real
            + projected_design.imag.T @ projected_design.imag
        ) / (2 * len(e50))
        projection_rhs += seed_weight * (
            projected_design.real.T @ projected_target.real
            + projected_design.imag.T @ projected_target.imag
        ) / (2 * len(e50))
    gram = (1.0 - projection_mass) * force_gram + projection_mass * projection_gram
    rhs = (1.0 - projection_mass) * force_rhs + projection_mass * projection_rhs
    gram = gram / normalizer[:, None] / normalizer[None, :]
    rhs = rhs / normalizer
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    tolerance = np.finfo(float).eps * width * max(float(eigenvalues[-1]), 1.0)
    if eigenvalues[0] < -10.0 * tolerance:
        raise ValueError("R2Y grid Gram is not positive semidefinite")
    return normalizer, np.maximum(eigenvalues, 0.0), eigenvectors, rhs


def coefficients(
    system: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    normalizer, eigenvalues, eigenvectors, rhs = system
    projected_rhs = eigenvectors.T @ rhs
    normalized = eigenvectors @ (
        projected_rhs[:, None] / (eigenvalues[:, None] + np.asarray(ALPHAS)[None, :])
    )
    return FORCE_SCALE * normalized / normalizer[:, None]


def aprime_metrics(
    predicted_modes: np.ndarray,
    target_modes: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    ratios = []
    all_error = []
    for name, start, stop in (("seed0", 0, 20), ("seed1", 20, 40)):
        error = predicted_modes[start:stop] - target_modes[start:stop]
        predicted_slope = r2y.restoring_slope(
            coordinates[start:stop], predicted_modes[start:stop]
        )
        target_slope = r2y.restoring_slope(
            coordinates[start:stop], target_modes[start:stop]
        )
        slope_error = (predicted_slope - target_slope) / target_slope
        rms = float(1000.0 * np.sqrt(np.mean(np.abs(error) ** 2)))
        records[name] = {
            "RMS_meV_A": rms,
            "slope_relative_error": float(slope_error),
        }
        ratios.extend((rms / 15.0, abs(slope_error) / 0.05))
        all_error.append(error)
    combined = float(
        1000.0 * np.sqrt(np.mean(np.abs(np.concatenate(all_error)) ** 2))
    )
    ratios.append(combined / 15.0)
    return {
        "by_seed": records,
        "combined_RMS_meV_A": combined,
        "score": float(max(ratios)),
        "pass": bool(all(value <= 1.0 for value in ratios)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--failed-root", type=Path, default=DEFAULT_R2Y_FAILED)
    parser.add_argument("--seed1-design-root", type=Path, default=DEFAULT_SEED1_DESIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    data = load_training(args.aggregate, args.failed_root, args.seed1_design_root)
    design99 = data["design"]
    fixed = data["fixed"]
    reference = data["reference"]
    modes = data["modes"]
    coordinates = data["coordinates"]
    base = data["base"]
    target_modes = np.einsum(
        "nat,nat->n", np.conj(modes), base + reference[:40]
    )
    all_indices = np.arange(112)
    candidates = []
    coefficient_cache: dict[tuple[str, float, float, float], list[np.ndarray]] = {}
    for representation, columns in REPRESENTATIONS.items():
        design = design99[..., columns]
        penalties = (1.0,) if design.shape[-1] == 65 else BILINEAR_PENALTIES
        projected_design = np.einsum(
            "nat,natk->nk", np.conj(modes), design[:40], optimize=True
        )
        fixed_modes = np.einsum(
            "nat,nat->n", np.conj(modes), base + fixed[:40]
        )
        for penalty in penalties:
            for mass in PROJECTION_MASSES:
                for seed1_weight in SEED1_PROJECTION_WEIGHTS:
                    predicted_by_alpha = np.full(
                        (40, len(ALPHAS)), np.nan + 0.0j
                    )
                    fold_coefficients = []
                    for fold_tuple in r2y.FOLDS:
                        hold = np.asarray(fold_tuple)
                        train = np.setdiff1d(all_indices, hold)
                        current = coefficients(
                            raw_system(
                                design,
                                fixed,
                                reference,
                                modes,
                                train,
                                mass,
                                penalty,
                                seed1_weight,
                            )
                        )
                        fold_coefficients.append(current)
                        e50_hold = hold[hold < 40]
                        predicted_by_alpha[e50_hold] = (
                            fixed_modes[e50_hold, None]
                            + projected_design[e50_hold] @ current
                        )
                    coefficient_cache[
                        (representation, penalty, mass, seed1_weight)
                    ] = fold_coefficients
                    if not np.all(np.isfinite(predicted_by_alpha)):
                        raise RuntimeError("R2Y projected OOF coverage failed")
                    for alpha_id, alpha in enumerate(ALPHAS):
                        metric = aprime_metrics(
                            predicted_by_alpha[:, alpha_id], target_modes, coordinates
                        )
                        candidates.append(
                            {
                                "representation": representation,
                                "bilinear_penalty": penalty,
                                "projection_mass": mass,
                                "seed1_projection_weight": seed1_weight,
                                "alpha": alpha,
                                "Aprime": metric,
                            }
                        )
        print(json.dumps({"representation_complete": representation}), flush=True)

    candidates.sort(
        key=lambda item: (
            item["Aprime"]["score"],
            len(REPRESENTATIONS[item["representation"]]),
            item["alpha"],
        )
    )
    full_records = []
    for candidate in candidates[:100]:
        columns = REPRESENTATIONS[candidate["representation"]]
        design = design99[..., columns]
        cache_key = (
            candidate["representation"],
            candidate["bilinear_penalty"],
            candidate["projection_mass"],
            candidate["seed1_projection_weight"],
        )
        alpha_id = ALPHAS.index(candidate["alpha"])
        oof = np.full_like(reference, np.nan)
        for fold_tuple, coefficient_matrix in zip(
            r2y.FOLDS, coefficient_cache[cache_key], strict=True
        ):
            hold = np.asarray(fold_tuple)
            coefficient = coefficient_matrix[:, alpha_id]
            oof[hold] = fixed[hold] + np.einsum(
                "natk,k->nat", design[hold], coefficient, optimize=False
            )
        metric = r2y.metrics(
            oof, reference, modes, coordinates, base, all_indices
        )
        record = dict(candidate)
        record["full_gate"] = metric
        full_records.append(record)
    full_records.sort(
        key=lambda item: (
            item["full_gate"]["raw_gate_score"],
            len(REPRESENTATIONS[item["representation"]]),
            item["alpha"],
        )
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "format": "graphene_r2y_seed01_development_grid_v1",
        "status": "R2Y_TRAINING_DOMAIN_GRID_COMPLETE",
        "scope": "seed0 + seed1 + T300 + T600 training only; 525 K unopened",
        "candidate_count": len(candidates),
        "projection_masses": list(PROJECTION_MASSES),
        "alphas": list(ALPHAS),
        "bilinear_penalties": list(BILINEAR_PENALTIES),
        "seed1_projection_weights": list(SEED1_PROJECTION_WEIGHTS),
        "representations": {
            key: value.tolist() for key, value in REPRESENTATIONS.items()
        },
        "Aprime_pass_count": sum(item["Aprime"]["pass"] for item in candidates),
        "top_Aprime_candidates": candidates[:100],
        "top_full_gate_candidates": full_records[:100],
        "best_full_gate_pass": bool(
            full_records and full_records[0]["full_gate"]["passes_fixed_gate"]
        ),
        "input_sha256": {
            "failed_R2Y_checkpoint": file_sha256(
                args.failed_root / "frozen_readout.npz"
            ),
            "aggregate": file_sha256(args.aggregate),
            "seed1_design": file_sha256(
                args.seed1_design_root / "seed1_selected_design.npz"
            ),
            "seed1_labels": file_sha256(r2y.SEED1_LABEL_PATH),
        },
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary["top_full_gate_candidates"][:5], indent=2))


if __name__ == "__main__":
    main()
