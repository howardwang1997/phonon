#!/usr/bin/env python3
"""Development ablation for a transferable graphene K long-range kernel.

This script intentionally consumes only the already-opened 300/450/600 K
line-cut table.  It does not launch or request new DFPT calculations.  The
main comparison asks whether separating the K-point anchor from a
centre-normalised Dirac rounding kernel removes the kink-shape error of the
previous shared model.

All fits act on the squared-frequency residual relative to the archived
broad-smearing background.  Model selection uses leave-one-smearing-out
(LOSO) predictions.  These are development diagnostics, not an independent
holdout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from graphene_fd_p4_common import line_metrics  # noqa: E402


RY_TO_EV = 13.605693122994
A_GRAPHENE = 2.4600000087
K_MAGNITUDE_A_INV = 4.0 * math.pi / (3.0 * A_GRAPHENE)
S_REF_RY = 0.04
TEMPERATURES = (300, 450, 600)

CANDIDATES = (
    "legacy_uncentred",
    "centred_anchor",
    "centred_anchor_amplitude",
    "centred_anchor_amplitude_curvature",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for row in raw:
        rows.append(
            {
                "temperature_K": int(row["temperature_K"]),
                "degauss_Ry": float(row["degauss_Ry"]),
                "region": row["region"],
                "t_GK": float(row["t_GK"]),
                "target_cm-1": float(row["target_cm-1"]),
                "background_cm-1": float(row["background_cm-1"]),
            }
        )
    if {row["temperature_K"] for row in rows} != set(TEMPERATURES):
        raise ValueError("input does not contain exactly the 300/450/600 K blocks")
    if len(rows) != 57:
        raise ValueError(f"expected 57 rows, found {len(rows)}")
    return rows


def distance(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            row["t_GK"] if row["region"] == "G" else abs(row["t_GK"] - 1.0)
            for row in rows
        ],
        float,
    )


def thermal_kernel(
    x: np.ndarray, degauss: np.ndarray, v_f_eV_A: float, width_scale: float
) -> np.ndarray:
    """Return 2*s*log(2*cosh(v*q/(2*s))) in Ry-like energy units."""
    x = np.asarray(x, float)
    effective_smearing = float(width_scale) * np.asarray(degauss, float)
    q_A_inv = x * K_MAGNITUDE_A_INV
    energy_Ry = float(v_f_eV_A) * q_A_inv / RY_TO_EV
    z = energy_Ry / (2.0 * effective_smearing)
    return 2.0 * effective_smearing * np.logaddexp(z, -z)


def feature_names(candidate: str) -> list[str]:
    if candidate == "legacy_uncentred":
        return ["constant", "distance_squared", "uncentred_kernel_difference"]
    names = [
        "constant",
        "anchor_linear_in_smearing",
        "distance_squared",
        "centred_kernel_difference",
    ]
    if candidate in (
        "centred_anchor_amplitude",
        "centred_anchor_amplitude_curvature",
    ):
        names.append("kernel_amplitude_linear_in_smearing")
    if candidate == "centred_anchor_amplitude_curvature":
        names.append("curvature_linear_in_smearing")
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    return names


def features(
    rows: list[dict], candidate: str, v_f_eV_A: float, width_scale: float
) -> np.ndarray:
    x = distance(rows)
    smear = np.asarray([row["degauss_Ry"] for row in rows], float)
    reduced_smearing = smear / S_REF_RY - 1.0
    reference = np.full_like(smear, S_REF_RY)
    phi_s = thermal_kernel(x, smear, v_f_eV_A, width_scale)
    phi_ref = thermal_kernel(x, reference, v_f_eV_A, width_scale)
    if candidate == "legacy_uncentred":
        return np.column_stack((np.ones_like(x), x * x, phi_s - phi_ref))

    zero = np.zeros_like(x)
    phi_s_zero = thermal_kernel(zero, smear, v_f_eV_A, width_scale)
    phi_ref_zero = thermal_kernel(zero, reference, v_f_eV_A, width_scale)
    centred_s = phi_s - phi_s_zero
    centred_ref = phi_ref - phi_ref_zero
    columns = [
        np.ones_like(x),
        reduced_smearing,
        x * x,
        centred_s - centred_ref,
    ]
    if candidate in (
        "centred_anchor_amplitude",
        "centred_anchor_amplitude_curvature",
    ):
        # If A(s)=A0+A1*u, then A0 multiplies the centred difference while
        # A1*u*Phi_bar(s) changes the anomaly amplitude without changing the
        # exact K anchor.  This is the minimum extra degree of freedom suggested
        # by the previous 450/600 K kink failures.
        columns.append(reduced_smearing * centred_s)
    if candidate == "centred_anchor_amplitude_curvature":
        columns.append(reduced_smearing * x * x)
    return np.column_stack(columns)


def fit_at_parameters(
    rows: list[dict],
    train_temperatures: set[int],
    candidate: str,
    v_f_eV_A: float,
    width_scale: float,
) -> tuple[np.ndarray, dict[str, list[float]], float]:
    prediction = np.zeros(len(rows), float)
    coefficients: dict[str, list[float]] = {}
    for region in ("G", "K"):
        region_indices = [index for index, row in enumerate(rows) if row["region"] == region]
        train_indices = [
            index
            for index in region_indices
            if rows[index]["temperature_K"] in train_temperatures
        ]
        train_rows = [rows[index] for index in train_indices]
        design = features(train_rows, candidate, v_f_eV_A, width_scale)
        target_delta = np.asarray(
            [
                rows[index]["target_cm-1"] ** 2
                - rows[index]["background_cm-1"] ** 2
                for index in train_indices
            ],
            float,
        )
        solved, *_ = np.linalg.lstsq(design, target_delta, rcond=None)
        coefficients[region] = solved.tolist()
        target_rows = [rows[index] for index in region_indices]
        correction = features(
            target_rows, candidate, v_f_eV_A, width_scale
        ) @ solved
        baseline_squared = np.asarray(
            [rows[index]["background_cm-1"] ** 2 for index in region_indices],
            float,
        )
        prediction[region_indices] = np.sqrt(
            np.maximum(baseline_squared + correction, 0.0)
        )
    train_mask = np.asarray(
        [row["temperature_K"] in train_temperatures for row in rows], bool
    )
    target = np.asarray([row["target_cm-1"] for row in rows], float)
    train_rmse = float(
        np.sqrt(np.mean((prediction[train_mask] - target[train_mask]) ** 2))
    )
    return prediction, coefficients, train_rmse


def fit_model(
    rows: list[dict],
    train_temperatures: set[int],
    candidate: str,
    velocity_grid: np.ndarray,
    width_grid: np.ndarray,
) -> dict:
    best = None
    for velocity in velocity_grid:
        for width in width_grid:
            prediction, coefficients, train_rmse = fit_at_parameters(
                rows,
                train_temperatures,
                candidate,
                float(velocity),
                float(width),
            )
            item = (
                train_rmse,
                float(velocity),
                float(width),
                prediction,
                coefficients,
            )
            if best is None or item[0] < best[0]:
                best = item
    assert best is not None
    return {
        "train_RMSE_cm-1": best[0],
        "v_F_eV_A": best[1],
        "width_scale": best[2],
        "prediction": best[3],
        "coefficients": best[4],
    }


def metrics_for(
    rows: list[dict], prediction: np.ndarray, temperatures: set[int]
) -> dict:
    result = {}
    for temperature in sorted(temperatures):
        result[str(temperature)] = {}
        for region in ("G", "K"):
            indices = [
                index
                for index, row in enumerate(rows)
                if row["temperature_K"] == temperature and row["region"] == region
            ]
            t = np.asarray([rows[index]["t_GK"] for index in indices], float)
            target = np.asarray([rows[index]["target_cm-1"] for index in indices], float)
            result[str(temperature)][region] = line_metrics(
                region, t, prediction[indices], target
            )
    return result


def candidate_evaluation(
    rows: list[dict],
    candidate: str,
    velocity_grid: np.ndarray,
    width_grid: np.ndarray,
) -> tuple[dict, np.ndarray]:
    all_temperatures = set(TEMPERATURES)
    loso_prediction = np.zeros(len(rows), float)
    folds = {}
    for held in TEMPERATURES:
        fitted = fit_model(
            rows,
            all_temperatures - {held},
            candidate,
            velocity_grid,
            width_grid,
        )
        mask = np.asarray([row["temperature_K"] == held for row in rows], bool)
        loso_prediction[mask] = fitted["prediction"][mask]
        folds[str(held)] = {
            "heldout_temperature_K": held,
            "train_RMSE_cm-1": fitted["train_RMSE_cm-1"],
            "v_F_eV_A": fitted["v_F_eV_A"],
            "width_scale": fitted["width_scale"],
            "coefficients": fitted["coefficients"],
            "metrics": metrics_for(rows, fitted["prediction"], {held})[str(held)],
        }
    target = np.asarray([row["target_cm-1"] for row in rows], float)
    loso_rmse = float(np.sqrt(np.mean((loso_prediction - target) ** 2)))
    loso_mae = float(np.mean(np.abs(loso_prediction - target)))
    kink_errors = {
        str(temperature): float(folds[str(temperature)]["metrics"]["K"]["kink_relative_error"])
        for temperature in TEMPERATURES
    }
    max_line_mae = max(
        float(folds[str(temperature)]["metrics"][region]["line_MAE_cm-1"])
        for temperature in TEMPERATURES
        for region in ("G", "K")
    )
    full = fit_model(
        rows, all_temperatures, candidate, velocity_grid, width_grid
    )
    n_coefficients_per_region = len(feature_names(candidate))
    complexity_penalty = 0.03 * max(0, n_coefficients_per_region - 3)
    passes_shape_gate = max(kink_errors.values()) < 0.20
    summary = {
        "candidate": candidate,
        "feature_names": feature_names(candidate),
        "n_coefficients_per_region": n_coefficients_per_region,
        "LOSO_RMSE_cm-1": loso_rmse,
        "LOSO_MAE_cm-1": loso_mae,
        "LOSO_max_line_MAE_cm-1": max_line_mae,
        "LOSO_K_kink_relative_error": kink_errors,
        "LOSO_max_K_kink_relative_error": max(kink_errors.values()),
        "passes_20pct_K_kink_gate": passes_shape_gate,
        "complexity_penalty": complexity_penalty,
        "selection_score": loso_rmse + complexity_penalty,
        "folds": folds,
        "full_fit": {
            "train_RMSE_cm-1": full["train_RMSE_cm-1"],
            "v_F_eV_A": full["v_F_eV_A"],
            "width_scale": full["width_scale"],
            "coefficients": full["coefficients"],
            "metrics": metrics_for(rows, full["prediction"], all_temperatures),
        },
    }
    return summary, loso_prediction


def write_predictions(
    path: Path,
    rows: list[dict],
    predictions: dict[str, np.ndarray],
) -> None:
    fields = list(rows[0]) + [f"{name}_LOSO_cm-1" for name in predictions]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = dict(row)
            for name, values in predictions.items():
                output[f"{name}_LOSO_cm-1"] = float(values[index])
            writer.writerow(output)


def make_figure(
    path: Path,
    rows: list[dict],
    selected: str,
    evaluations: dict[str, dict],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharey=True)
    for axis, temperature in zip(axes, TEMPERATURES, strict=True):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["temperature_K"] == temperature and row["region"] == "K"
        ]
        t = np.asarray([rows[index]["t_GK"] for index in indices], float)
        target = np.asarray([rows[index]["target_cm-1"] for index in indices], float)
        background = np.asarray(
            [rows[index]["background_cm-1"] for index in indices], float
        )
        dense_t = np.linspace(float(np.min(t)), float(np.max(t)), 801)
        dense_background = PchipInterpolator(t, background)(dense_t)
        dense_rows = [
            {
                "temperature_K": temperature,
                "degauss_Ry": rows[indices[0]]["degauss_Ry"],
                "region": "K",
                "t_GK": float(value),
                "target_cm-1": 0.0,
                "background_cm-1": float(base),
            }
            for value, base in zip(dense_t, dense_background, strict=True)
        ]
        curves = {}
        for name in ("legacy_uncentred", selected):
            fold = next(
                item
                for item in evaluations[name]["folds"].values()
                if int(item["heldout_temperature_K"]) == temperature
            )
            correction = (
                features(
                    dense_rows,
                    name,
                    float(fold["v_F_eV_A"]),
                    float(fold["width_scale"]),
                )
                @ np.asarray(fold["coefficients"]["K"], float)
            )
            curves[name] = np.sqrt(
                np.maximum(dense_background**2 + correction, 0.0)
            )
        axis.plot(
            t,
            target,
            "o",
            color="#202020",
            ms=3.4,
            label="direct DFPT points",
            zorder=4,
        )
        axis.plot(
            dense_t,
            curves["legacy_uncentred"],
            "--",
            color="#D55E00",
            lw=1.3,
            label="legacy shared kernel",
        )
        axis.plot(
            dense_t,
            curves[selected],
            "-",
            color="#0072B2",
            lw=1.6,
            label="centred anchor + A(smearing)",
        )
        axis.set_title(f"degauss = {rows[indices[0]]['degauss_Ry']:.7f} Ry")
        axis.set_xlabel("t along q=tK")
        axis.grid(axis="y", color="#E6E6E6", lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$A_1'$ frequency (cm$^{-1}$)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Graphene K long-range kernel: leave-one-smearing-out comparison")
    fig.subplots_adjust(left=0.07, right=0.79, top=0.82, bottom=0.18, wspace=0.08)
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E0_shared_dirac/shared_dirac_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E2_shared_kernel_v2"
        ),
    )
    parser.add_argument("--velocity-points", type=int, default=81)
    parser.add_argument("--width-points", type=int, default=31)
    args = parser.parse_args()

    rows = load_rows(args.input)
    velocity_grid = np.linspace(4.0, 8.0, args.velocity_points)
    predictions = {}
    evaluations = []
    for candidate in CANDIDATES:
        width_grid = (
            np.asarray([1.0])
            if candidate == "legacy_uncentred"
            else np.linspace(0.50, 2.00, args.width_points)
        )
        evaluation, prediction = candidate_evaluation(
            rows, candidate, velocity_grid, width_grid
        )
        evaluations.append(evaluation)
        predictions[candidate] = prediction

    eligible = [item for item in evaluations if item["passes_20pct_K_kink_gate"]]
    ranked = sorted(
        eligible if eligible else evaluations,
        key=lambda item: (item["selection_score"], item["LOSO_max_K_kink_relative_error"]),
    )
    selected = ranked[0]["candidate"]
    status = "development_shape_gate_passed" if eligible else "development_shape_gate_failed"
    summary = {
        "status": status,
        "scope": (
            "development-only finite-smearing spectral ablation; no new DFPT and "
            "no independent holdout"
        ),
        "selected_candidate": selected,
        "selection_rule": (
            "first require every leave-one-smearing-out K kink error <20%; "
            "then minimize LOSO RMSE plus 0.03 per coefficient beyond the legacy three"
        ),
        "physical_constraints": {
            "finite_smearing": "quadratic at K and asymptotically linear outside the rounding width",
            "zero_smearing_limit": "centre-normalised kernel tends to v_F*|q-K|",
            "anchor_decoupling": "K-point intercept is separate from cusp amplitude",
        },
        "candidates": evaluations,
        "inputs": {"path": str(args.input), "sha256": sha256(args.input)},
        "limitations": [
            "The zero-smearing rows are not fitted in this first ablation.",
            "The selected form must next be checked jointly against the opened zero-smearing data.",
            "LOSO uses opened development smearings and is not a prospective validation.",
            "This is a spectral kernel test; dynamical-matrix replay remains a separate gate.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "shared_kernel_v2_ablation.json", summary)
    write_predictions(args.output_dir / "shared_kernel_v2_predictions.csv", rows, predictions)
    make_figure(
        args.output_dir / "shared_kernel_v2_comparison.png",
        rows,
        selected,
        {item["candidate"]: item for item in evaluations},
    )
    print(json.dumps(summary, indent=2))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
