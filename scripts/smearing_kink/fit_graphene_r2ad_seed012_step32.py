#!/usr/bin/env python3
"""Fit the fixed R2Z step-32 representation on promoted seed0/1/2 data.

All three E50 trajectories are development training data.  Four-fold OOF
holds five structures from each trajectory and nine structures from each of
the 300 and 600 K groups.  The planned 525 K off-policy set remains unopened.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase.io import read

import evaluate_graphene_r2z_forward_bilinear as r2z
import fit_graphene_r2y_seed01_paired_readout as r2y
import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from evaluate_graphene_r2t_bilinear_nested_objective import design_audit


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_SEED2_LABELS = (
    ROOT / "data/graphene_r2m_support_free_core/reserved_e50_seed2.xyz"
)
DEFAULT_SEED2_BASE = BASE / "R2AC_seed2_base_design_20260826"
DEFAULT_SEED2_FULL = BASE / "R2AC_seed2_full_bilinear_20260826"
DEFAULT_PATH_ROOT = BASE / "R2Z_forward_bilinear_development_32step_20260826"
DEFAULT_OUTPUT = BASE / "R2AD_seed012_step32_development_20260826"

FORCE_SCALE = 0.030
APRIME_SCALE = 0.015
COMPONENTS = 72 * 3
GROUP_RANGES = {
    "E50_seed0": (0, 20),
    "E50_seed1": (20, 40),
    "E50_seed2": (40, 60),
    "T300": (60, 96),
    "T600": (96, 132),
}
SEED_RANGES = {
    "seed0": (0, 20),
    "seed1": (20, 40),
    "seed2": (40, 60),
}
FOLDS = tuple(
    tuple(
        list(range(5 * fold, 5 * (fold + 1)))
        + list(range(20 + 5 * fold, 20 + 5 * (fold + 1)))
        + list(range(40 + 5 * fold, 40 + 5 * (fold + 1)))
        + list(range(60 + 9 * fold, 60 + 9 * (fold + 1)))
        + list(range(96 + 9 * fold, 96 + 9 * (fold + 1)))
    )
    for fold in range(4)
)
PROJECTION_MASSES = (0.80, 0.83, 0.86, 0.89)
BILINEAR_PENALTIES = (0.15, 0.25, 0.40)
T600_FORCE_FRACTIONS = (0.84, 0.88, 0.90)
ALPHAS = tuple(float(value) for value in np.geomspace(5.0e-3, 1.8e-2, 21))


def seed_weight_grid() -> tuple[tuple[float, float, float], ...]:
    output = []
    for seed2_weight in (0.25, 1.0 / 3.0, 0.40):
        for seed1_fraction in (0.425, 0.50, 0.575):
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


@dataclass(frozen=True)
class FoldRaw:
    square_sum: np.ndarray
    structure_count: int
    group_gram: dict[str, np.ndarray]
    group_rhs: dict[str, np.ndarray]
    group_count: dict[str, int]
    projection_gram: dict[str, np.ndarray]
    projection_rhs: dict[str, np.ndarray]
    projection_count: dict[str, int]


def indices_for(bounds: tuple[int, int]) -> np.ndarray:
    return np.arange(bounds[0], bounds[1], dtype=int)


def _verified_seed2_base(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    receipt_path = root / "receipt.json"
    arrays_path = root / "seed2_base_design.npz"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "R2AC_SEED2_BASE_DESIGN_ASSEMBLED_AND_VERIFIED":
        raise ValueError("R2AD seed2 base design is incomplete")
    if file_sha256(arrays_path) != receipt.get("arrays_file_sha256"):
        raise ValueError("R2AD seed2 base design differs from receipt")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        indices = np.asarray(arrays["structure_indices"], dtype=int)
        fixed = np.asarray(arrays["fixed_force_eV_A"], dtype=np.float64)
        design = np.asarray(
            arrays["parameter_force_design_eV_A"], dtype=np.float64
        )
    if not np.array_equal(indices, np.arange(20)):
        raise ValueError("R2AD seed2 base structure order changed")
    if fixed.shape != (20, 72, 3) or design.shape != (20, 72, 3, 65):
        raise ValueError("R2AD seed2 base array shape changed")
    for key, value in (
        ("fixed_force_eV_A", fixed),
        ("parameter_force_design_eV_A", design),
    ):
        if r2r1.raw_array_sha256(value, "<f8") != receipt["array_raw_sha256"][key]:
            raise ValueError(f"R2AD seed2 base raw hash changed: {key}")
    return fixed, design, {
        "receipt_sha256": file_sha256(receipt_path),
        "arrays_sha256": file_sha256(arrays_path),
    }


def _verified_seed2_full(root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "R2AC_SEED2_FULL_BILINEAR_ASSEMBLED_AND_VERIFIED":
        raise ValueError("R2AD seed2 full bilinear design is incomplete")
    force = r2z.verified_array(
        root, "bilinear_force_design_eV_A.npy", receipt, "force"
    )
    if force.shape != (20, 72, 3, 512):
        raise ValueError("R2AD seed2 full bilinear shape changed")
    return force, {
        "receipt_sha256": file_sha256(receipt_path),
        "force_sha256": file_sha256(root / "bilinear_force_design_eV_A.npy"),
    }


def _seed2_labels(path: Path) -> tuple[np.ndarray, ...]:
    structures = read(path, index=":")
    if len(structures) != 20 or any(len(item) != 72 for item in structures):
        raise ValueError("R2AD seed2 label schema changed")
    for item in structures:
        if (
            item.info.get("r2m_split") != "opened_development"
            or item.info.get("r2m_target_role") != "exact_e50_seed2"
            or float(item.info.get("degauss_Ry")) != 0.0019000869
            or float(item.info.get("lattice_temperature_K")) != 450.0
        ):
            raise ValueError("R2AD seed2 opened-development scope changed")
    reference = np.stack([item.arrays["REF_forces"] for item in structures])
    modes = np.stack(
        [
            item.arrays["APRIME_mode_real"]
            + 1.0j * item.arrays["APRIME_mode_imag"]
            for item in structures
        ]
    )
    coordinates = np.asarray(
        [
            complex(
                item.info["APRIME_coordinate_real_A"],
                item.info["APRIME_coordinate_imag_A"],
            )
            for item in structures
        ]
    )
    base = np.stack(
        [
            item.arrays["FOUNDATION_BASE_forces"]
            + item.arrays["FROZEN_Q6_forces"]
            for item in structures
        ]
    )
    return reference, modes, coordinates, base


def selected_step32(path_root: Path) -> tuple[np.ndarray, str]:
    path = path_root / "summary.json"
    summary = json.loads(path.read_text())
    if len(summary.get("steps", [])) != 32:
        raise ValueError("R2AD fixed forward path no longer has 32 steps")
    step = summary["steps"][31]
    selected = np.asarray(
        step["starting_bilinear_indices"]
        + [int(step["best"]["selected_bilinear_index"])],
        dtype=int,
    )
    if selected.shape != (64,) or len(np.unique(selected)) != 64:
        raise ValueError("R2AD step-32 representation changed")
    return selected, file_sha256(path)


def load_training(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data112, receipt112 = r2z.load_full_training(args)
    fixed2, base_design2, receipt_base2 = _verified_seed2_base(
        args.seed2_base_root
    )
    full2, receipt_full2 = _verified_seed2_full(args.seed2_full_root)
    reference2, modes2, coordinates2, base2 = _seed2_labels(args.seed2_labels)
    selected, path_sha = selected_step32(args.path_root)
    design2 = np.concatenate((base_design2, full2), axis=-1)
    design577 = np.concatenate(
        (data112["design"][:40], design2, data112["design"][40:]), axis=0
    )
    columns = np.r_[np.arange(65), 65 + selected]
    design = np.take(design577, columns, axis=-1)
    fixed = np.concatenate(
        (data112["fixed"][:40], fixed2, data112["fixed"][40:]), axis=0
    )
    reference = np.concatenate(
        (data112["reference"][:40], reference2, data112["reference"][40:]),
        axis=0,
    )
    modes = np.concatenate((data112["modes"], modes2), axis=0)
    coordinates = np.concatenate((data112["coordinates"], coordinates2), axis=0)
    base = np.concatenate((data112["base"], base2), axis=0)
    if (
        design.shape != (132, 72, 3, 129)
        or fixed.shape != (132, 72, 3)
        or reference.shape != (132, 72, 3)
        or modes.shape != (60, 72, 3)
        or coordinates.shape != (60,)
        or base.shape != (60, 72, 3)
    ):
        raise ValueError("R2AD combined training schema changed")
    return {
        "design": design,
        "fixed": fixed,
        "reference": reference,
        "modes": modes,
        "coordinates": coordinates,
        "base": base,
        "selected_bilinear_indices": selected,
    }, {
        "seed01_thermal": receipt112,
        "seed2_base": receipt_base2,
        "seed2_full": receipt_full2,
        "seed2_labels_sha256": file_sha256(args.seed2_labels),
        "step32_path_summary_sha256": path_sha,
    }


def fold_raw_statistics(data: dict[str, np.ndarray]) -> tuple[FoldRaw, ...]:
    design = data["design"]
    response = (data["reference"] - data["fixed"]) / FORCE_SCALE
    output = []
    for fold_tuple in FOLDS:
        fold = np.asarray(fold_tuple, dtype=int)
        group_gram = {}
        group_rhs = {}
        group_count = {}
        for name, bounds in GROUP_RANGES.items():
            members = np.intersect1d(fold, indices_for(bounds))
            matrix = design[members].reshape(-1, design.shape[-1])
            target = response[members].reshape(-1)
            group_gram[name] = matrix.T @ matrix
            group_rhs[name] = matrix.T @ target
            group_count[name] = len(members)
        projection_gram = {}
        projection_rhs = {}
        projection_count = {}
        for name, bounds in SEED_RANGES.items():
            members = np.intersect1d(fold, indices_for(bounds))
            matrix = np.einsum(
                "nat,natk->nk",
                np.conj(data["modes"][members]),
                design[members],
                optimize=True,
            )
            target = np.einsum(
                "nat,nat->n",
                np.conj(data["modes"][members]),
                data["reference"][members] - data["fixed"][members],
            )
            matrix *= FORCE_SCALE / APRIME_SCALE
            target /= APRIME_SCALE
            projection_gram[name] = (
                matrix.real.T @ matrix.real + matrix.imag.T @ matrix.imag
            )
            projection_rhs[name] = (
                matrix.real.T @ target.real + matrix.imag.T @ target.imag
            )
            projection_count[name] = len(members)
        output.append(
            FoldRaw(
                square_sum=np.sum(np.square(design[fold]), axis=(0, 1, 2)),
                structure_count=len(fold),
                group_gram=group_gram,
                group_rhs=group_rhs,
                group_count=group_count,
                projection_gram=projection_gram,
                projection_rhs=projection_rhs,
                projection_count=projection_count,
            )
        )
    return tuple(output)


def group_masses(t600_fraction: float) -> dict[str, float]:
    other = (1.0 - t600_fraction) / 4.0
    return {
        "E50_seed0": other,
        "E50_seed1": other,
        "E50_seed2": other,
        "T300": other,
        "T600": t600_fraction,
    }


def normal_equations(
    statistics: Sequence[FoldRaw],
    chosen_folds: Sequence[int],
    projection_mass: float,
    seed_weights: Sequence[float],
    bilinear_penalty: float,
    t600_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = [statistics[int(value)] for value in chosen_folds]
    width = len(chosen[0].square_sum)
    structure_count = sum(item.structure_count for item in chosen)
    square_sum = sum(
        (item.square_sum for item in chosen), start=np.zeros(width)
    )
    scale = np.sqrt(square_sum / (structure_count * COMPONENTS))
    penalty = np.ones(width)
    penalty[65:] = bilinear_penalty
    normalizer = scale * np.sqrt(penalty)
    if np.any(normalizer <= np.max(normalizer) * 1.0e-12):
        raise ValueError("R2AD selected design contains a zero column")

    force_gram = np.zeros((width, width))
    force_rhs = np.zeros(width)
    masses = group_masses(t600_fraction)
    for name in GROUP_RANGES:
        count = sum(item.group_count[name] for item in chosen)
        raw_gram = sum(
            (item.group_gram[name] for item in chosen),
            start=np.zeros((width, width)),
        )
        raw_rhs = sum(
            (item.group_rhs[name] for item in chosen), start=np.zeros(width)
        )
        force_gram += masses[name] * raw_gram / (count * COMPONENTS)
        force_rhs += masses[name] * raw_rhs / (count * COMPONENTS)

    projection_gram = np.zeros((width, width))
    projection_rhs = np.zeros(width)
    for name, weight in zip(SEED_RANGES, seed_weights, strict=True):
        count = sum(item.projection_count[name] for item in chosen)
        raw_gram = sum(
            (item.projection_gram[name] for item in chosen),
            start=np.zeros((width, width)),
        )
        raw_rhs = sum(
            (item.projection_rhs[name] for item in chosen), start=np.zeros(width)
        )
        projection_gram += weight * raw_gram / (2 * count)
        projection_rhs += weight * raw_rhs / (2 * count)

    gram = (1.0 - projection_mass) * force_gram + projection_mass * projection_gram
    rhs = (1.0 - projection_mass) * force_rhs + projection_mass * projection_rhs
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
        raise ValueError("R2AD Gram is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ rhs
    normalized = eigenvectors @ (
        projected_rhs[:, None]
        / (eigenvalues[:, None] + np.asarray(ALPHAS)[None, :])
    )
    return FORCE_SCALE * normalized / normalizer[:, None]


def vectorized_scores(predicted: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    candidate_count = predicted.shape[-1]
    if predicted.shape != (132, 72, 3, candidate_count):
        raise ValueError("R2AD vectorized prediction shape changed")
    error = predicted - data["reference"][..., None]
    ratios = []
    for bounds in GROUP_RANGES.values():
        current = error[indices_for(bounds)]
        ratios.append(
            1000.0 * np.sqrt(np.mean(np.square(current), axis=(0, 1, 2))) / 30.0
        )
        ratios.append(1000.0 * np.max(np.abs(current), axis=(0, 1, 2)) / 200.0)
    all_mode_error = []
    for bounds in SEED_RANGES.values():
        members = indices_for(bounds)
        mode_error = np.einsum(
            "nat,natq->nq",
            np.conj(data["modes"][members]),
            error[members],
            optimize=True,
        )
        ratios.append(
            1000.0 * np.sqrt(np.mean(np.abs(mode_error) ** 2, axis=0)) / 15.0
        )
        coordinates = data["coordinates"][members]
        denominator = float(np.vdot(coordinates, coordinates).real)
        target_modes = np.einsum(
            "nat,nat->n",
            np.conj(data["modes"][members]),
            data["base"][members] + data["reference"][members],
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


def scalar_metrics(
    prediction: np.ndarray, data: dict[str, np.ndarray]
) -> dict[str, Any]:
    output: dict[str, Any] = {"force_by_group": {}, "Aprime_by_seed": {}}
    ratios = []
    for name, bounds in GROUP_RANGES.items():
        members = indices_for(bounds)
        current = r2y.force_metrics(
            prediction[members] - data["reference"][members]
        )
        output["force_by_group"][name] = current
        ratios.extend(
            (current["RMSE_meV_A"] / 30.0, current["max_abs_meV_A"] / 200.0)
        )
    all_mode_error = []
    for name, bounds in SEED_RANGES.items():
        members = indices_for(bounds)
        mode_error = np.einsum(
            "nat,nat->n",
            np.conj(data["modes"][members]),
            prediction[members] - data["reference"][members],
        )
        predicted_modes = np.einsum(
            "nat,nat->n",
            np.conj(data["modes"][members]),
            data["base"][members] + prediction[members],
        )
        target_modes = np.einsum(
            "nat,nat->n",
            np.conj(data["modes"][members]),
            data["base"][members] + data["reference"][members],
        )
        predicted_slope = r2y.restoring_slope(
            data["coordinates"][members], predicted_modes
        )
        target_slope = r2y.restoring_slope(
            data["coordinates"][members], target_modes
        )
        slope_error = (predicted_slope - target_slope) / target_slope
        rms = float(1000.0 * np.sqrt(np.mean(np.abs(mode_error) ** 2)))
        output["Aprime_by_seed"][name] = {
            "RMS_meV_A": rms,
            "predicted_restoring_slope_eV_A2": predicted_slope,
            "target_restoring_slope_eV_A2": target_slope,
            "slope_relative_error": float(slope_error),
            "count": len(members),
        }
        ratios.extend((rms / 15.0, abs(slope_error) / 0.05))
        all_mode_error.append(mode_error)
    combined = float(
        1000.0 * np.sqrt(np.mean(np.abs(np.concatenate(all_mode_error)) ** 2))
    )
    output["Aprime_combined_RMS_meV_A"] = combined
    ratios.append(combined / 15.0)
    output["raw_gate_score"] = float(max(ratios))
    output["passes_fixed_gate"] = bool(all(value <= 1.0 for value in ratios))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=r2z.DEFAULT_AGGREGATE)
    parser.add_argument("--failed-root", type=Path, default=r2z.DEFAULT_R2Y_FAILED)
    parser.add_argument(
        "--seed1-selected-root", type=Path, default=r2z.DEFAULT_SEED1_SELECTED
    )
    parser.add_argument("--thermal-full-root", type=Path, default=r2z.DEFAULT_THERMAL_FULL)
    parser.add_argument("--seed1-full-root", type=Path, default=r2z.DEFAULT_SEED1_FULL)
    parser.add_argument("--seed2-labels", type=Path, default=DEFAULT_SEED2_LABELS)
    parser.add_argument("--seed2-base-root", type=Path, default=DEFAULT_SEED2_BASE)
    parser.add_argument("--seed2-full-root", type=Path, default=DEFAULT_SEED2_FULL)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    data, input_receipt = load_training(args)
    audit = design_audit(data["design"])
    if not audit["pass"]:
        raise ValueError(f"R2AD design audit failed: {audit}")
    statistics = fold_raw_statistics(data)

    records = []
    best_score = np.inf
    best_prediction: np.ndarray | None = None
    best_record: dict[str, Any] | None = None
    all_indices = np.arange(132, dtype=int)
    for mass in PROJECTION_MASSES:
        for seed_weights in SEED_PROJECTION_WEIGHTS:
            for penalty in BILINEAR_PENALTIES:
                for t600_fraction in T600_FORCE_FRACTIONS:
                    oof = np.full((132, 72, 3, len(ALPHAS)), np.nan)
                    for hold_id, fold_tuple in enumerate(FOLDS):
                        hold = np.asarray(fold_tuple, dtype=int)
                        train_folds = tuple(
                            value for value in range(4) if value != hold_id
                        )
                        coefficient = coefficient_matrix(
                            normal_equations(
                                statistics,
                                train_folds,
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
                    if not np.all(np.isfinite(oof)):
                        raise RuntimeError("R2AD OOF coverage failed")
                    scores = vectorized_scores(oof, data)
                    for alpha_id, score in enumerate(scores):
                        record = {
                            "raw_gate_score": float(score),
                            "projection_mass": mass,
                            "seed_projection_weights": {
                                name: float(weight)
                                for name, weight in zip(
                                    SEED_RANGES, seed_weights, strict=True
                                )
                            },
                            "bilinear_penalty": penalty,
                            "T600_force_fraction": t600_fraction,
                            "group_masses": group_masses(t600_fraction),
                            "alpha": ALPHAS[alpha_id],
                        }
                        records.append(record)
                        if score < best_score:
                            best_score = float(score)
                            best_prediction = oof[..., alpha_id].copy()
                            best_record = record
        print(json.dumps({"projection_mass_complete": mass}), flush=True)
    if best_prediction is None or best_record is None:
        raise RuntimeError("R2AD grid produced no candidate")
    best_metrics = scalar_metrics(best_prediction, data)
    if abs(best_metrics["raw_gate_score"] - best_score) > 1.0e-10:
        raise ValueError("R2AD vectorized/scalar gate score mismatch")

    best_seed_weights = tuple(
        best_record["seed_projection_weights"][name] for name in SEED_RANGES
    )
    final_matrix = coefficient_matrix(
        normal_equations(
            statistics,
            tuple(range(4)),
            float(best_record["projection_mass"]),
            best_seed_weights,
            float(best_record["bilinear_penalty"]),
            float(best_record["T600_force_fraction"]),
        )
    )
    alpha_id = ALPHAS.index(float(best_record["alpha"]))
    coefficient = final_matrix[:, alpha_id]
    train_prediction = data["fixed"] + np.einsum(
        "natk,k->nat", data["design"], coefficient, optimize=False
    )
    train_metrics = scalar_metrics(train_prediction, data)
    if best_metrics["passes_fixed_gate"] and not train_metrics["passes_fixed_gate"]:
        raise ValueError("R2AD all132 fit failed after OOF passed")

    with np.load(args.failed_root / "frozen_readout.npz", allow_pickle=False) as arrays:
        feature_mean = np.asarray(arrays["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(arrays["feature_scale"], dtype=np.float64)
    status = (
        "R2AD_SEED012_DEVELOPMENT_GATE_PASSED_MECHANICS_PENDING"
        if best_metrics["passes_fixed_gate"]
        else "R2AD_SEED012_DEVELOPMENT_GATE_FAILED"
    )
    records.sort(key=lambda item: (item["raw_gate_score"], item["alpha"]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "frozen_readout.npz"
    np.savez(
        arrays_path,
        physical_coefficient=np.asarray(coefficient, dtype="<f8"),
        base65_physical_coefficient=np.asarray(coefficient[:65], dtype="<f8"),
        bilinear64_physical_coefficient=np.asarray(coefficient[65:], dtype="<f8"),
        selected_bilinear_indices=np.asarray(
            data["selected_bilinear_indices"], dtype="<i8"
        ),
        feature_mean=np.asarray(feature_mean, dtype="<f8"),
        feature_scale=np.asarray(feature_scale, dtype="<f8"),
        OOF_predicted_force_eV_A=np.asarray(best_prediction, dtype="<f8"),
        train_predicted_force_eV_A=np.asarray(train_prediction, dtype="<f8"),
        combined_force_design_eV_A=np.asarray(data["design"], dtype="<f8"),
    )
    summary = {
        "format": "graphene_r2ad_seed012_step32_development_v1",
        "status": status,
        "deployable": False,
        "scope": "seed0 + seed1 + promoted opened-development seed2 + T300 + T600 training; 525 K unopened",
        "data_roles": {
            "seed0": "training",
            "seed1": "promoted_training",
            "seed2": "promoted_training_from_opened_development",
            "new_525K_off_policy": "unopened_true_unseen",
        },
        "folds": [list(item) for item in FOLDS],
        "fixed_gate": {
            "force_RMSE_meV_A": 30.0,
            "force_max_abs_meV_A": 200.0,
            "Aprime_RMS_meV_A_per_seed_and_combined": 15.0,
            "Aprime_restoring_slope_relative_error": 0.05,
        },
        "selected_bilinear_indices": data[
            "selected_bilinear_indices"
        ].tolist(),
        "candidate_count": int(
            len(PROJECTION_MASSES)
            * len(SEED_PROJECTION_WEIGHTS)
            * len(BILINEAR_PENALTIES)
            * len(T600_FORCE_FRACTIONS)
            * len(ALPHAS)
        ),
        "best_OOF": {**best_record, "full_gate": best_metrics},
        "final_all132_fit": {
            "full_gate": train_metrics,
            "coefficient_raw_sha256": r2r1.raw_array_sha256(coefficient, "<f8"),
        },
        "top_candidates": records[:300],
        "design_audit": audit,
        "input_receipt": input_receipt,
        "output_arrays_sha256": file_sha256(arrays_path),
        "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(
            best_prediction, "<f8"
        ),
        "energy_labels_used": False,
        "unseen_525K_access": False,
        "mechanics_pending": bool(best_metrics["passes_fixed_gate"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(status + "\n")
    print(json.dumps({"status": status, "best": summary["best_OOF"]}, indent=2))


if __name__ == "__main__":
    main()
