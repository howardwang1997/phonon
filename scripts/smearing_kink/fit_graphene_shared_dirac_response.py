#!/usr/bin/env python3
"""Fit a shared finite-smearing Dirac/Mermin response to graphene Γ/K lines."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import friedel_module as fm  # noqa: E402
from graphene_fd_p4_common import (  # noqa: E402
    atomic_json,
    line_metrics,
    qpoint_fractional,
    sha256,
)


RY_TO_EV = 13.605693122994
A_GRAPHENE = 2.4600000087
K_MAGNITUDE_A_INV = 4.0 * math.pi / (3.0 * A_GRAPHENE)
S_REF_RY = 0.04
CONDITIONS = (
    (300, 0.0019000869),
    (450, 0.00285013035),
    (600, 0.0038001738),
)


def load_dfpt(path: Path, temperature: int, degauss: float) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 19:
        raise ValueError(f"expected 19 rows in {path}, found {len(rows)}")
    result = []
    for row in rows:
        observed = float(row["degauss_Ry"])
        if not np.isclose(observed, degauss, rtol=0.0, atol=5.0e-12):
            raise ValueError(f"unexpected degauss in {path}: {observed}")
        result.append(
            {
                "temperature_K": temperature,
                "degauss_Ry": degauss,
                "region": row["region"],
                "t_GK": float(row["t_GK"]),
                "target_cm-1": float(row["f6_cm-1"]),
            }
        )
    expected = [(row["region"], round(row["t_GK"], 3)) for row in result]
    reference = [
        ("G", 0.000),
        ("G", 0.005),
        ("G", 0.010),
        ("G", 0.015),
        ("G", 0.025),
        ("G", 0.040),
        ("G", 0.060),
        ("G", 0.080),
        ("K", 0.940),
        ("K", 0.960),
        ("K", 0.975),
        ("K", 0.985),
        ("K", 0.992),
        ("K", 1.000),
        ("K", 1.008),
        ("K", 1.015),
        ("K", 1.025),
        ("K", 1.040),
        ("K", 1.060),
    ]
    if expected != reference:
        raise ValueError(f"unexpected Γ/K line ordering in {path}")
    return result


def load_background(path: Path, rows: list[dict]) -> np.ndarray:
    phonon = fm.load_ph(path)
    t = np.asarray([row["t_GK"] for row in rows], float)
    phonon.run_qpoints(qpoint_fractional(t))
    frequencies = np.sort(
        np.asarray(phonon.get_qpoints_dict()["frequencies"], float), axis=1
    )
    return frequencies[:, -1] * 33.35641


def thermal_kernel(distance: np.ndarray, degauss: np.ndarray, v_f_eV_A: float):
    q_A_inv = np.asarray(distance, float) * K_MAGNITUDE_A_INV
    energy_Ry = float(v_f_eV_A) * q_A_inv / RY_TO_EV
    smearing = np.asarray(degauss, float)
    z = energy_Ry / (2.0 * smearing)
    return 2.0 * smearing * np.logaddexp(z, -z)


def kernel_difference(distance, degauss, v_f_eV_A):
    distance = np.asarray(distance, float)
    degauss = np.asarray(degauss, float)
    reference = np.full_like(degauss, S_REF_RY)
    return thermal_kernel(distance, degauss, v_f_eV_A) - thermal_kernel(
        distance, reference, v_f_eV_A
    )


def design_matrix(rows: list[dict], v_f_eV_A: float, include_kernel: bool):
    distance = np.asarray(
        [row["t_GK"] if row["region"] == "G" else abs(row["t_GK"] - 1.0) for row in rows],
        float,
    )
    columns = [np.ones_like(distance), distance * distance]
    if include_kernel:
        columns.append(
            kernel_difference(
                distance,
                np.asarray([row["degauss_Ry"] for row in rows], float),
                v_f_eV_A,
            )
        )
    return np.column_stack(columns)


def fit_at_velocity(
    rows: list[dict],
    train_temperatures: set[int],
    v_f_eV_A: float,
    *,
    include_kernel: bool,
):
    coefficients = {}
    predictions = np.zeros(len(rows), float)
    for region in ("G", "K"):
        region_indices = [index for index, row in enumerate(rows) if row["region"] == region]
        train_indices = [
            index
            for index in region_indices
            if int(rows[index]["temperature_K"]) in train_temperatures
        ]
        train_rows = [rows[index] for index in train_indices]
        y = np.asarray(
            [
                rows[index]["target_cm-1"] ** 2
                - rows[index]["background_cm-1"] ** 2
                for index in train_indices
            ],
            float,
        )
        matrix = design_matrix(train_rows, v_f_eV_A, include_kernel)
        fitted, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        coefficients[region] = fitted
        target_rows = [rows[index] for index in region_indices]
        correction = design_matrix(target_rows, v_f_eV_A, include_kernel) @ fitted
        squared = np.asarray(
            [rows[index]["background_cm-1"] ** 2 for index in region_indices], float
        ) + correction
        predictions[region_indices] = np.sqrt(np.maximum(squared, 0.0))
    return predictions, coefficients


def fit_model(
    rows: list[dict],
    train_temperatures: set[int],
    *,
    include_kernel: bool,
    velocity_min: float,
    velocity_max: float,
    velocity_points: int,
):
    velocities = (
        np.linspace(velocity_min, velocity_max, velocity_points)
        if include_kernel
        else np.asarray([(velocity_min + velocity_max) / 2.0])
    )
    best = None
    train_mask = np.asarray(
        [int(row["temperature_K"]) in train_temperatures for row in rows], bool
    )
    target = np.asarray([row["target_cm-1"] for row in rows], float)
    for velocity in velocities:
        prediction, coefficients = fit_at_velocity(
            rows,
            train_temperatures,
            float(velocity),
            include_kernel=include_kernel,
        )
        rmse = float(np.sqrt(np.mean((prediction[train_mask] - target[train_mask]) ** 2)))
        candidate = (rmse, float(velocity), prediction, coefficients)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return {
        "train_RMSE_cm-1": best[0],
        "v_F_eV_A": best[1] if include_kernel else None,
        "prediction": best[2],
        "coefficients": {
            region: {
                "B0_cm-2": float(values[0]),
                "B2_cm-2": float(values[1]),
                **(
                    {"A_cm-2_per_Ry": float(values[2])}
                    if include_kernel
                    else {}
                ),
            }
            for region, values in best[3].items()
        },
    }


def evaluate_prediction(rows, prediction, temperatures: set[int]):
    result = {}
    passes = True
    for temperature in sorted(temperatures):
        result[str(temperature)] = {}
        for region in ("G", "K"):
            indices = [
                index
                for index, row in enumerate(rows)
                if int(row["temperature_K"]) == temperature and row["region"] == region
            ]
            t = np.asarray([rows[index]["t_GK"] for index in indices], float)
            target = np.asarray([rows[index]["target_cm-1"] for index in indices], float)
            metrics = line_metrics(region, t, np.asarray(prediction)[indices], target)
            result[str(temperature)][region] = metrics
            passes = passes and metrics["line_MAE_cm-1"] < 10.0
            passes = passes and metrics["high_symmetry_abs_error_cm-1"] < 15.0
            if region == "K":
                passes = passes and metrics["kink_relative_error"] < 0.20
    return result, bool(passes)


def old_p4_lookup(path: Path | None):
    if path is None:
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["region"], round(float(row["t_GK"]), 3)): float(
            row["static_frozen_prediction_cm-1"]
        )
        for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfpt-300", type=Path, required=True)
    parser.add_argument("--dfpt-450", type=Path, required=True)
    parser.add_argument("--dfpt-600", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--old-p4-qspace", type=Path)
    parser.add_argument("--velocity-min", type=float, default=4.0)
    parser.add_argument("--velocity-max", type=float, default=8.0)
    parser.add_argument("--velocity-points", type=int, default=401)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {300: args.dfpt_300, 450: args.dfpt_450, 600: args.dfpt_600}
    rows = []
    templates = {}
    for temperature, degauss in CONDITIONS:
        block = load_dfpt(paths[temperature], temperature, degauss)
        templates[temperature] = block
        rows.extend(block)
    background = load_background(args.background, templates[300])
    for row_index, row in enumerate(rows):
        row["background_cm-1"] = float(background[row_index % len(background)])

    all_temperatures = {300, 450, 600}
    full = fit_model(
        rows,
        all_temperatures,
        include_kernel=True,
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
        velocity_points=args.velocity_points,
    )
    full_metrics, full_pass = evaluate_prediction(rows, full["prediction"], all_temperatures)
    no_kernel = fit_model(
        rows,
        all_temperatures,
        include_kernel=False,
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
        velocity_points=1,
    )
    no_kernel_metrics, _ = evaluate_prediction(
        rows, no_kernel["prediction"], all_temperatures
    )

    folds = {}
    all_fold_pass = True
    loso_prediction = np.zeros(len(rows), float)
    control_prediction = np.zeros(len(rows), float)
    for heldout in sorted(all_temperatures):
        training = all_temperatures - {heldout}
        model = fit_model(
            rows,
            training,
            include_kernel=True,
            velocity_min=args.velocity_min,
            velocity_max=args.velocity_max,
            velocity_points=args.velocity_points,
        )
        control = fit_model(
            rows,
            training,
            include_kernel=False,
            velocity_min=args.velocity_min,
            velocity_max=args.velocity_max,
            velocity_points=1,
        )
        mask = np.asarray(
            [int(row["temperature_K"]) == heldout for row in rows], bool
        )
        loso_prediction[mask] = model["prediction"][mask]
        control_prediction[mask] = control["prediction"][mask]
        held_metrics, held_pass = evaluate_prediction(rows, model["prediction"], {heldout})
        control_metrics, control_pass = evaluate_prediction(
            rows, control["prediction"], {heldout}
        )
        folds[str(heldout)] = {
            "heldout_temperature_K": heldout,
            "heldout_degauss_Ry": next(
                degauss for temperature, degauss in CONDITIONS if temperature == heldout
            ),
            "training_temperatures_K": sorted(training),
            "shared_dirac": {
                "train_RMSE_cm-1": model["train_RMSE_cm-1"],
                "v_F_eV_A": model["v_F_eV_A"],
                "coefficients": model["coefficients"],
                "metrics": held_metrics[str(heldout)],
                "passes_gate": held_pass,
            },
            "no_electronic_kernel_control": {
                "train_RMSE_cm-1": control["train_RMSE_cm-1"],
                "coefficients": control["coefficients"],
                "metrics": control_metrics[str(heldout)],
                "passes_same_numeric_gate": control_pass,
            },
        }
        all_fold_pass = all_fold_pass and held_pass

    target = np.asarray([row["target_cm-1"] for row in rows], float)
    loso_rmse = float(np.sqrt(np.mean((loso_prediction - target) ** 2)))
    control_loso_rmse = float(np.sqrt(np.mean((control_prediction - target) ** 2)))
    improves_control = loso_rmse < control_loso_rmse
    velocity_not_boundary = bool(
        full["v_F_eV_A"] > args.velocity_min + 1.0e-12
        and full["v_F_eV_A"] < args.velocity_max - 1.0e-12
    )
    overall_pass = bool(full_pass and all_fold_pass and improves_control)

    old_lookup = old_p4_lookup(args.old_p4_qspace)
    output_rows = []
    for index, row in enumerate(rows):
        key = (row["region"], round(float(row["t_GK"]), 3))
        output_rows.append(
            {
                **row,
                "shared_dirac_full_fit_cm-1": float(full["prediction"][index]),
                "shared_dirac_LOSO_cm-1": float(loso_prediction[index]),
                "no_kernel_full_fit_cm-1": float(no_kernel["prediction"][index]),
                "no_kernel_LOSO_cm-1": float(control_prediction[index]),
                "old_P4_static_450_cm-1": (
                    old_lookup.get(key, "") if int(row["temperature_K"]) == 450 else ""
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "shared_dirac_predictions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "status": "passed" if overall_pass else "failed",
        "scope": (
            "development-only shared finite-smearing Dirac/Mermin spectral feasibility; "
            "450 K remains an opened failed holdout"
        ),
        "formula": {
            "kernel": "Phi=2*s*log(2*cosh(hbar*v_F*|Delta q|/(2*s)))",
            "squared_frequency": (
                "lambda=lambda_bg+B0_r+B2_r*x^2+A_r*(Phi(s)-Phi(s_ref))"
            ),
            "s_ref_Ry": S_REF_RY,
            "chemical_potential": "mu=0 fixed",
            "shared_parameters": "one v_F across Gamma/K and all smearings",
            "region_parameters": "B0, B2, and coupling A shared across smearings",
        },
        "acceptance": {
            "line_MAE_cm-1": 10.0,
            "high_symmetry_abs_error_cm-1": 15.0,
            "K_kink_relative_error": 0.20,
            "v_F_range_eV_A": [args.velocity_min, args.velocity_max],
            "must_improve_no_electronic_kernel_LOSO": True,
        },
        "full_development_fit": {
            "train_RMSE_cm-1": full["train_RMSE_cm-1"],
            "v_F_eV_A": full["v_F_eV_A"],
            "v_F_at_search_boundary": not velocity_not_boundary,
            "coefficients": full["coefficients"],
            "metrics": full_metrics,
            "passes_numeric_gate": full_pass,
        },
        "leave_one_smearing_out": {
            "shared_dirac_RMSE_cm-1": loso_rmse,
            "no_electronic_kernel_RMSE_cm-1": control_loso_rmse,
            "improves_no_electronic_kernel": improves_control,
            "all_folds_pass": all_fold_pass,
            "folds": folds,
        },
        "passes_E0_S_gate": overall_pass,
        "limitations": [
            "This is a q-space spectral test, not yet a real-space force operator.",
            "The 450 K data are development data after the failed P4 evaluation.",
            *(
                ["The fitted v_F is at the fixed search boundary; parameter identification is weak."]
                if not velocity_not_boundary
                else []
            ),
        ],
        "inputs": {
            "dfpt": {
                str(temperature): {"path": str(path), "sha256": sha256(path)}
                for temperature, path in paths.items()
            },
            "background": {
                "path": str(args.background),
                "sha256": sha256(args.background),
            },
            "old_p4_qspace": (
                {"path": str(args.old_p4_qspace), "sha256": sha256(args.old_p4_qspace)}
                if args.old_p4_qspace
                else None
            ),
        },
        "outputs": {"predictions_csv": str(csv_path)},
    }
    atomic_json(args.output_dir / "shared_dirac_fit.json", summary)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.3), sharey="row")
    for column, (temperature, _) in enumerate(CONDITIONS):
        for row_index, region in enumerate(("G", "K")):
            axis = axes[row_index, column]
            selected = [
                (index, row)
                for index, row in enumerate(output_rows)
                if int(row["temperature_K"]) == temperature and row["region"] == region
            ]
            t = [row["t_GK"] for _, row in selected]
            axis.plot(t, [row["target_cm-1"] for _, row in selected], "o-", color="#222222", label="DFPT")
            axis.plot(t, [row["background_cm-1"] for _, row in selected], ":", color="#999999", label="broad-smearing background")
            axis.plot(t, [row["shared_dirac_LOSO_cm-1"] for _, row in selected], "s--", color="#0072B2", label="shared Dirac LOSO")
            axis.plot(t, [row["no_kernel_LOSO_cm-1"] for _, row in selected], "^--", color="#D55E00", label="no electronic kernel")
            axis.set_title(f"{temperature} K · {region}")
            axis.set_xlabel("t along q=tK")
            axis.set_ylabel("top optical frequency (cm$^{-1}$)")
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "shared_dirac_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
