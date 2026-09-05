#!/usr/bin/env python3
"""Explore a finite-slope no-degauss graphene K-cusp law after v1 holdout.

This is a development-only analysis.  The former blind points have now been
revealed and are explicitly treated as v2 development data.  No model emitted
by this script is a frozen or independently validated result.  Candidate laws
are required to be continuous at K, to have finite one-sided slopes, and not to
vary on a fitted scale shorter than the smallest sampled distance from K.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_k_cusp_nosmear_method as method  # noqa: E402


MODEL_NAMES = (
    "linear_directional",
    "finite_slope_exponential",
    "finite_slope_rational",
)
EXPECTED_LABELS = {
    "K",
    "KG_d003", "KG_d007", "KG_d011", "KG_d015", "KG_d019", "KG_d023",
    "KM_d003", "KM_d007", "KM_d011", "KM_d015", "KM_d019", "KM_d023",
}
# There are no direct calculations between K and d=0.003.  Allowing a shorter
# scale made the previous exploratory fit collapse to an almost discontinuous
# step.  This resolution floor is fixed from the q-point design, not fitted.
RESOLUTION_FLOOR = 3.0e-3
SCALE_GRID = np.geomspace(RESOLUTION_FLOOR, 2.0e-2, 241)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def q_reduced(direction: str, distance: float) -> np.ndarray:
    if direction == "K":
        h = k = 1.0 / 3.0
    elif direction == "KG":
        h = k = (1.0 - distance) / 3.0
    elif direction == "KM":
        h, k = (1.0 + distance) / 3.0, (1.0 - 2.0 * distance) / 3.0
    else:
        raise ValueError(direction)
    return np.asarray([h, k, 0.0])


def load_points(development: Path, revealed: Path) -> list[dict]:
    points = []
    for row in read_csv(development):
        points.append(
            {
                "label": row["label"],
                "direction": row["direction"],
                "delta": float(row["delta"]),
                "frequency_cm-1": float(row["frequency_cm-1"]),
                "v2_role": "original_development",
            }
        )
    for row in read_csv(revealed):
        points.append(
            {
                "label": row["label"],
                "direction": row["direction"],
                "delta": float(row["delta"]),
                "frequency_cm-1": float(row["frequency_cm-1"]),
                "v2_role": "revealed_v1_holdout",
            }
        )
    labels = [row["label"] for row in points]
    if set(labels) != EXPECTED_LABELS or len(labels) != len(EXPECTED_LABELS):
        raise ValueError(
            f"v2 development points differ from plan: missing={EXPECTED_LABELS-set(labels)}, "
            f"extra={set(labels)-EXPECTED_LABELS}, duplicates={len(labels)-len(set(labels))}"
        )
    points.sort(key=lambda row: (0 if row["direction"] == "K" else 1, row["direction"], row["delta"]))
    return points


def scale_feature(model: str, distance: np.ndarray, scale: float | None) -> np.ndarray:
    distance = np.asarray(distance, float)
    if model == "linear_directional":
        return np.zeros_like(distance)
    if scale is None or scale <= 0.0:
        raise ValueError(f"{model} requires a positive scale")
    # Multiplication by scale makes the fitted coefficient the excess inner
    # slope of the dynamical-matrix eigenvalue.  Both features equal distance
    # + O(distance**2) near K, so their one-sided derivatives are finite.
    if model == "finite_slope_exponential":
        return scale * (1.0 - np.exp(-distance / scale))
    if model == "finite_slope_rational":
        return scale * distance / (distance + scale)
    raise ValueError(model)


def design(rows: list[dict], model: str, scale: float | None) -> np.ndarray:
    distance = np.asarray([float(row["delta"]) for row in rows])
    kg = np.asarray([row["direction"] == "KG" for row in rows], float)
    km = np.asarray([row["direction"] == "KM" for row in rows], float)
    columns = [kg * distance, km * distance]
    if model != "linear_directional":
        columns.append(scale_feature(model, distance, scale))
    return np.column_stack(columns)


def prediction(rows: list[dict], fit: dict) -> np.ndarray:
    matrix = design(rows, fit["model"], fit.get("scale"))
    squared = float(fit["lambda_K_cm-2"]) + matrix @ np.asarray(fit["linear_coefficients_cm-2"], float)
    if np.any(squared <= 0.0):
        raise ValueError("candidate produced a non-positive squared frequency")
    return np.sqrt(squared)


def fit_model(rows: list[dict], model: str) -> dict:
    k_rows = [row for row in rows if row["direction"] == "K"]
    if len(k_rows) != 1:
        raise ValueError("each v2 fit must contain exactly one K anchor")
    lambda_k = float(k_rows[0]["frequency_cm-1"]) ** 2
    target = np.asarray([float(row["frequency_cm-1"]) ** 2 - lambda_k for row in rows])
    scales: list[float | None] = [None] if model == "linear_directional" else SCALE_GRID.tolist()
    best = None
    for scale in scales:
        matrix = design(rows, model, scale)
        solved = lsq_linear(matrix, target, bounds=(0.0, np.inf), lsmr_tol="auto")
        candidate = {
            "model": model,
            "lambda_K_cm-2": lambda_k,
            "linear_coefficients_cm-2": solved.x.tolist(),
            "scale": scale,
        }
        predicted = prediction(rows, candidate)
        observed = np.asarray([float(row["frequency_cm-1"]) for row in rows])
        loss = float(np.mean((predicted - observed) ** 2))
        if best is None or loss < best[0]:
            best = (loss, candidate)
    if best is None:
        raise RuntimeError(f"no fit for {model}")
    result = best[1]
    distance_min = min(float(row["delta"]) for row in rows if row["direction"] != "K")
    result["scale_at_resolution_floor"] = bool(
        result["scale"] is not None
        and np.isclose(float(result["scale"]), RESOLUTION_FLOOR, rtol=1.0e-8, atol=1.0e-12)
    )
    result["scale_resolved_by_development"] = bool(
        result["scale"] is None or not result["scale_at_resolution_floor"]
    )
    result["minimum_nonzero_development_distance"] = distance_min
    result["fixed_resolution_floor"] = RESOLUTION_FLOOR
    return result


def cross_validation(rows: list[dict], model: str) -> dict:
    distances = sorted({float(row["delta"]) for row in rows if row["direction"] != "K"})
    errors = []
    by_distance = {}
    fitted_scales = {}
    for held in distances:
        training = [
            row for row in rows
            if row["direction"] == "K" or not np.isclose(float(row["delta"]), held)
        ]
        validation = [row for row in rows if np.isclose(float(row["delta"]), held)]
        fitted = fit_model(training, model)
        predicted = prediction(validation, fitted)
        observed = np.asarray([float(row["frequency_cm-1"]) for row in validation])
        fold_errors = np.abs(predicted - observed)
        errors.extend(fold_errors.tolist())
        by_distance[f"{held:.3f}"] = float(np.mean(fold_errors))
        fitted_scales[f"{held:.3f}"] = fitted.get("scale")
    parameter_count = 3 if model == "linear_directional" else 5
    mae = float(np.mean(errors))
    penalty = 0.10 * max(0, parameter_count - 3)
    return {
        "model": model,
        "n_parameters_including_K_anchor": parameter_count,
        "leave_one_distance_out_MAE_cm-1": mae,
        "MAE_by_held_distance_cm-1": by_distance,
        "fitted_scale_by_held_distance": fitted_scales,
        "complexity_penalty_cm-1": penalty,
        "selection_score": mae + penalty,
    }


def sharpness_metrics(rows: list[dict], values: np.ndarray) -> dict:
    lookup = {row["label"]: float(value) for row, value in zip(rows, values, strict=True)}
    k_value = lookup["K"]
    result = {}
    for direction in ("KG", "KM"):
        inner = (lookup[f"{direction}_d003"] - k_value) / 0.003
        outer = (lookup[f"{direction}_d019"] - lookup[f"{direction}_d011"]) / 0.008
        result[direction] = {
            "inner_secant_slope_cm-1_per_d": inner,
            "outer_secant_slope_cm-1_per_d": outer,
            "inner_to_outer_slope_ratio": inner / outer,
        }
    return result


def one_sided_slopes_at_k(fit: dict) -> dict[str, float]:
    """Return finite dω/dd slopes on the two sampled rays at K."""
    coefficients = np.asarray(fit["linear_coefficients_cm-2"], float)
    omega_k = math.sqrt(float(fit["lambda_K_cm-2"]))
    excess_inner = 0.0 if fit["model"] == "linear_directional" else float(coefficients[2])
    return {
        "KG_cm-1_per_d": float((coefficients[0] + excess_inner) / (2.0 * omega_k)),
        "KM_cm-1_per_d": float((coefficients[1] + excess_inner) / (2.0 * omega_k)),
    }


def model_equation(model: str) -> str:
    if model == "finite_slope_exponential":
        feature = "delta_a*ell*(1-exp(-d/ell))"
    elif model == "finite_slope_rational":
        feature = "delta_a*ell*d/(d+ell)"
    elif model == "linear_directional":
        feature = "0"
    else:
        raise ValueError(model)
    return f"lambda_dir(d) = lambda_K + a_dir*d + {feature}; ell >= 0.003"


def write_points(path: Path, rows: list[dict], predictions: dict[str, np.ndarray]) -> None:
    fields = ["label", "direction", "delta", "frequency_cm-1", "v2_role"] + [
        f"{name}_cm-1" for name in predictions
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = dict(row)
            output.update({f"{name}_cm-1": float(values[index]) for name, values in predictions.items()})
            writer.writerow(output)


def make_figure(
    rows: list[dict], predictions: dict[str, np.ndarray], selected: str, path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.7))
    colors = {"KG": "#1F77B4", "KM": "#D55E00"}
    dense_distance = np.linspace(0.0, 0.023, 1001)
    k_row = next(row for row in rows if row["direction"] == "K")
    for direction, direction_label in (("KM", "K→M"), ("KG", "K→Γ")):
        sign = -1.0 if direction == "KM" else 1.0
        direct_rows = [k_row] + sorted(
            [row for row in rows if row["direction"] == direction],
            key=lambda row: float(row["delta"]),
        )
        direct_distance = np.asarray([float(row["delta"]) for row in direct_rows])
        direct_relative = np.asarray(
            [
                float(row["frequency_cm-1"]) - float(k_row["frequency_cm-1"])
                for row in direct_rows
            ]
        )
        direct_guide = PchipInterpolator(direct_distance, direct_relative)(dense_distance)
        axis.plot(
            sign * dense_distance,
            direct_guide,
            color="#303030",
            lw=2.0,
            alpha=0.78,
            label="direct DFPT guide to eye" if direction == "KG" else None,
            zorder=1,
        )
        dense_rows = [k_row] + [
            {"direction": direction, "delta": float(value)} for value in dense_distance[1:]
        ]
        for model_name, linestyle, alpha in (
            ("linear_directional", "--", 0.55),
            (selected, "-", 1.0),
        ):
            fit = predictions[f"{model_name}__fit"]
            y = prediction(dense_rows, fit) - float(k_row["frequency_cm-1"])
            axis.plot(
                sign * dense_distance,
                y,
                linestyle,
                color=colors[direction],
                lw=1.25 if linestyle == "--" else 1.8,
                alpha=alpha,
                label=(
                    f"{direction_label} linear v1"
                    if model_name == "linear_directional"
                    else f"{direction_label} finite-slope v2"
                ),
                zorder=2,
            )
        for role, marker, face, label_suffix in (
            ("original_development", "o", colors[direction], "original development"),
            ("revealed_v1_holdout", "D", "white", "revealed v1 holdout"),
        ):
            selected_rows = [row for row in rows if row["direction"] == direction and row["v2_role"] == role]
            axis.scatter(
                [sign * float(row["delta"]) for row in selected_rows],
                [float(row["frequency_cm-1"]) - float(k_row["frequency_cm-1"]) for row in selected_rows],
                marker=marker,
                s=31,
                facecolors=face,
                edgecolors=colors[direction],
                linewidths=0.9,
                zorder=4,
                label=label_suffix if direction == "KG" else None,
            )
    axis.axvline(0.0, color="#C8C8C8", lw=0.8)
    axis.set_xticks([-0.023, 0.0, 0.023], ["K→M", "K", "K→Γ"])
    axis.set_xlabel("equal-distance coordinate around K")
    axis.set_ylabel(r"frequency relative to K (cm$^{-1}$)")
    axis.set_title(r"Graphene no-degauss $A_1'$ near K — finite-slope cusp")
    axis.grid(axis="y", color="#E6E6E6", lw=0.55)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.text(
        0.5,
        0.035,
        "DFPT connecting curves are shape-preserving guides; no sampled point lies inside |d| < 0.003",
        ha="center",
        fontsize=7.7,
    )
    fig.text(
        0.5,
        0.012,
        "revealed S4 points are v2 development data; no independent v2 holdout has been evaluated",
        ha="center",
        fontsize=7.7,
    )
    fig.subplots_adjust(left=0.12, right=0.68, top=0.89, bottom=0.20)
    fig.savefig(path, dpi=230, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-points", type=Path, required=True)
    parser.add_argument("--revealed-holdout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_points(args.development_points, args.revealed_holdout)
    observed = np.asarray([float(row["frequency_cm-1"]) for row in rows])

    scores = [cross_validation(rows, name) for name in MODEL_NAMES]
    scores.sort(key=lambda item: item["selection_score"])
    fits = {name: fit_model(rows, name) for name in MODEL_NAMES}
    fitted_predictions = {name: prediction(rows, fits[name]) for name in MODEL_NAMES}
    selected = scores[0]["model"]
    selected_fit = fits[selected]
    errors = fitted_predictions[selected] - observed
    observed_sharpness = sharpness_metrics(rows, observed)
    predicted_sharpness = sharpness_metrics(rows, fitted_predictions[selected])
    sharpness_relative_errors = {
        direction: abs(
            predicted_sharpness[direction]["inner_to_outer_slope_ratio"]
            - observed_sharpness[direction]["inner_to_outer_slope_ratio"]
        ) / abs(observed_sharpness[direction]["inner_to_outer_slope_ratio"])
        for direction in ("KG", "KM")
    }

    projector_rows = [
        {
            "direction": row["direction"],
            "delta": float(row["delta"]),
            "q_reduced": q_reduced(row["direction"], float(row["delta"])),
        }
        for row in rows
    ]
    projector = method.static_projector_replay(
        projector_rows, fitted_predictions[selected]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_points(args.output_dir / "v2_development_predictions.csv", rows, fitted_predictions)
    plot_payload = dict(fitted_predictions)
    plot_payload.update({f"{name}__fit": fit for name, fit in fits.items()})
    make_figure(
        rows,
        plot_payload,
        selected,
        args.output_dir / "v2_development_exploratory.png",
    )

    scale_resolved = bool(selected_fit["scale_resolved_by_development"])
    if selected == "linear_directional":
        status = "no_finite_slope_improvement"
    elif scale_resolved:
        status = "finite_slope_selected_wait_for_kgrid"
    else:
        status = "finite_slope_selected_at_resolution_floor_wait_for_kgrid"
    summary = {
        "status": status,
        "scope": "v2 development only after the v1 blind holdout was revealed",
        "independent_v2_validation": False,
        "electronic_integration": "tetrahedra_opt (no degauss)",
        "selected_candidate": selected,
        "selected_fit": selected_fit,
        "selected_one_sided_slopes_at_K": one_sided_slopes_at_k(selected_fit),
        "candidate_scores": scores,
        "revealed_data_fit_metrics": {
            "MAE_cm-1": float(np.mean(np.abs(errors))),
            "max_abs_error_cm-1": float(np.max(np.abs(errors))),
        },
        "sharpness": {
            "definition": "inner=[omega(0.003)-omega(K)]/0.003; outer=[omega(0.019)-omega(0.011)]/0.008",
            "direct": observed_sharpness,
            "selected_candidate": predicted_sharpness,
            "ratio_relative_errors": sharpness_relative_errors,
        },
        "matrix_checks": {
            "Hermiticity_max_abs": projector["Hermiticity_max_abs"],
            "rank_one_replay_max_abs_cm-1": projector["rank_one_replay_max_abs_cm-1"],
        },
        "sources": {
            "original_development": {
                "path": str(args.development_points),
                "sha256": method.digest(args.development_points),
            },
            "revealed_v1_holdout": {
                "path": str(args.revealed_holdout),
                "sha256": method.digest(args.revealed_holdout),
            },
        },
        "literature_constraint": {
            "form": "omega(K+q') = omega_K + alpha_K |q'| + O(|q'|^2)",
            "implementation": model_equation(selected),
            "reference": "Piscanec et al., Phys. Rev. Lett. 93, 185503 (2004)",
            "doi": "10.1103/PhysRevLett.93.185503",
        },
        "rejected_previous_exploration": {
            "model": "unconstrained two_scale_exponential with scale >= 1e-5",
            "reason": (
                "the fitted scale hit 1e-5, below the d=0.003 sampling resolution, "
                "and produced an almost discontinuous step at K"
            ),
        },
        "limitations": [
            "the former v1 holdout is now development data and cannot validate v2",
            "the selected crossover scale must not be frozen until d=0.003 k-grid convergence is resolved",
            "the crossover scale is not allowed below the smallest sampled distance d=0.003",
            "a selected scale at the fixed resolution floor is reported as boundary-limited, not measured",
            "PCHIP curves through direct DFPT points are guides to the eye and are not model predictions",
        ],
    }
    atomic_json(args.output_dir / "v2_development_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
