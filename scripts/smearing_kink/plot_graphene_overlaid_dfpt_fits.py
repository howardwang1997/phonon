#!/usr/bin/env python3
"""Overlay fixed-background MLIP+EPC curves and smooth DFPT fits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
SMEARINGS = (0.0, 0.0019000869, 0.00285013035, 0.0038001738)
COLORS = ("#111111", "#0072B2", "#009E73", "#D55E00")
FINITE_DFPT = {
    0.0019000869: ROOT
    / "results/graphene_physical_fd_dfpt/campaigns/FD300_LINE/graphene_FD300_LINE_dfpt.csv",
    0.00285013035: BASE
    / "source_p4_450/dfpt/FD450_LINE/graphene_FD450_LINE_dfpt.csv",
    0.0038001738: ROOT
    / "results/graphene_physical_fd_dfpt/campaigns/FD600_LINE/graphene_FD600_LINE_dfpt.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-sweep",
        type=Path,
        default=BASE
        / "E31_fixed_background_smearing_sweep/fixed_background_smearing_sweep_Aprime.csv",
    )
    parser.add_argument(
        "--zero-summary",
        type=Path,
        default=BASE
        / "E14_zero_quadrature_convergence/zero_quadrature_convergence_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E33_overlaid_DFPT_fits",
    )
    return parser.parse_args()


def load_model(path: Path) -> dict[float, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for smearing in SMEARINGS:
        block = [
            row
            for row in rows
            if np.isclose(
                float(row["smearing_degauss_Ry"]), smearing, atol=1.0e-14
            )
        ]
        if len(block) != 121:
            raise RuntimeError(f"expected 121 model points at smearing {smearing}")
        result[smearing] = block
    return result


def load_dfpt_KG(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["region"] == "K" and float(row["t_GK"]) <= 1.0 + 1.0e-12
        ]
    rows.sort(key=lambda row: 1.0 - float(row["t_GK"]))
    return {
        "distance": np.asarray([1.0 - float(row["t_GK"]) for row in rows], float),
        "frequency_cm1": np.asarray([float(row["f6_cm-1"]) for row in rows], float),
        "kgrid": int(rows[0]["kgrid"]),
    }


def rounded_cusp(
    distance: np.ndarray,
    omega_K: float,
    slope: float,
    width: float,
    curvature: float,
) -> np.ndarray:
    distance = np.asarray(distance, float)
    rounded = np.sqrt(distance**2 + width**2) - width
    return omega_K + slope * rounded + curvature * distance**2


def fit_finite_dfpt(data: dict) -> dict:
    distance = data["distance"]
    target = data["frequency_cm1"]
    omega_K = float(target[0])

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (
            rounded_cusp(
                distance,
                omega_K,
                float(parameters[0]),
                float(parameters[1]),
                float(parameters[2]),
            )
            - target
        )

    result = least_squares(
        residual,
        x0=np.asarray([750.0, 0.01, -2500.0]),
        bounds=(
            np.asarray([0.0, 1.0e-6, -1.0e5]),
            np.asarray([5000.0, 0.2, 1.0e5]),
        ),
    )
    prediction = target + result.fun
    return {
        "omega_K_cm-1": omega_K,
        "slope_cm-1_per_d": float(result.x[0]),
        "rounding_width_d": float(result.x[1]),
        "curvature_cm-1_per_d2": float(result.x[2]),
        "fit_RMSE_cm-1": float(np.sqrt(np.mean(result.fun**2))),
        "fit_max_abs_cm-1": float(np.max(np.abs(result.fun))),
        "success": bool(result.success),
        "prediction_at_samples_cm-1": prediction,
    }


def model_arrays(block: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([float(row["signed_distance"]) for row in block], float)
    y = np.asarray(
        [float(row["controlled_MLIP_plus_full_EPC_cm-1"]) for row in block],
        float,
    )
    return x, y


def main() -> int:
    args = parse_args()
    model = load_model(args.controlled_sweep)
    zero_summary = json.loads(args.zero_summary.read_text(encoding="utf-8"))
    finite_data = {
        smearing: load_dfpt_KG(FINITE_DFPT[smearing])
        for smearing in SMEARINGS[1:]
    }
    finite_fits = {
        smearing: fit_finite_dfpt(finite_data[smearing])
        for smearing in SMEARINGS[1:]
    }
    if not all(row["success"] for row in finite_fits.values()):
        raise RuntimeError("at least one DFPT rounded-cusp fit failed")

    figure, axis = plt.subplots(figsize=(9.6, 5.5))
    for smearing, color in zip(SMEARINGS, COLORS):
        x, y = model_arrays(model[smearing])
        order = np.argsort(x)
        axis.plot(
            x[order],
            y[order],
            color=color,
            lw=2.7 if smearing == 0.0 else 2.1,
            zorder=2,
        )

    fit_curve_rows = []
    dense_distance = np.linspace(0.0, 0.03, 401)
    for smearing, color in zip(SMEARINGS[1:], COLORS[1:]):
        fit = finite_fits[smearing]
        fit_frequency = rounded_cusp(
            dense_distance,
            fit["omega_K_cm-1"],
            fit["slope_cm-1_per_d"],
            fit["rounding_width_d"],
            fit["curvature_cm-1_per_d2"],
        )
        axis.plot(
            -dense_distance,
            fit_frequency,
            color=color,
            lw=1.9,
            ls="--",
            zorder=3,
        )
        data = finite_data[smearing]
        display_mask = data["distance"] <= 0.03 + 1.0e-12
        axis.plot(
            -data["distance"][display_mask],
            data["frequency_cm1"][display_mask],
            "o",
            ms=4.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            zorder=4,
        )
        for distance, value in zip(dense_distance, fit_frequency):
            fit_curve_rows.append(
                {
                    "smearing_degauss_Ry": smearing,
                    "fit_scope": "K_to_Gamma",
                    "signed_distance": -float(distance),
                    "DFPT_fit_frequency_cm-1": float(value),
                }
            )

    zero_x, zero_y = model_arrays(model[0.0])
    zero_K = float(zero_y[np.argmin(np.abs(zero_x))])
    zero_dense_x = np.linspace(-0.003, 0.003, 241)
    zero_fit_y = np.empty_like(zero_dense_x)
    zero_KG_depth = float(
        zero_summary["directions"]["KG"][
            "DFPT_three_point_1_over_k_limit_cm-1"
        ]
    )
    zero_KM_depth = float(
        zero_summary["directions"]["KM"][
            "DFPT_three_point_1_over_k_limit_cm-1"
        ]
    )
    negative = zero_dense_x < 0.0
    zero_fit_y[negative] = zero_K + zero_KG_depth * (
        -zero_dense_x[negative] / 0.003
    )
    zero_fit_y[~negative] = zero_K + zero_KM_depth * (
        zero_dense_x[~negative] / 0.003
    )
    axis.plot(
        zero_dense_x,
        zero_fit_y,
        color=COLORS[0],
        lw=1.9,
        ls="--",
        zorder=3,
    )
    zero_anchor_x = np.asarray([-0.003, 0.0, 0.003])
    zero_anchor_y = np.asarray(
        [zero_K + zero_KG_depth, zero_K, zero_K + zero_KM_depth]
    )
    axis.plot(
        zero_anchor_x,
        zero_anchor_y,
        "o",
        ms=4.2,
        markerfacecolor="white",
        markeredgecolor=COLORS[0],
        markeredgewidth=1.1,
        zorder=4,
    )
    for x_value, value in zip(zero_dense_x, zero_fit_y):
        fit_curve_rows.append(
            {
                "smearing_degauss_Ry": 0.0,
                "fit_scope": "local_KG_K_KM_abs_d_le_0.003",
                "signed_distance": float(x_value),
                "DFPT_fit_frequency_cm-1": float(value),
            }
        )

    axis.axvline(0.0, color="#AAAAAA", lw=0.9, zorder=0)
    axis.set_xlim(-0.03, 0.03)
    axis.set_xlabel("signed distance from K (K→Γ < 0; K→M > 0)")
    axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    axis.set_title(
        "Graphene K-neighbourhood A′: MLIP + full EPC LR and fitted DFPT"
    )
    axis.grid(alpha=0.18)

    color_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2.3,
            label="0 Ry" if smearing == 0.0 else f"{smearing:.7f} Ry",
        )
        for smearing, color in zip(SMEARINGS, COLORS)
    ]
    style_handles = [
        Line2D([0], [0], color="#555555", lw=2.3, label="MLIP + full EPC LR"),
        Line2D(
            [0], [0], color="#555555", lw=1.9, ls="--", label="DFPT fit"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            label="DFPT samples / anchors",
        ),
    ]
    first_legend = axis.legend(
        handles=color_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),
        frameon=False,
        title="smearing/degauss (Ry)",
        borderaxespad=0.0,
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=style_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=False,
        title="method",
        borderaxespad=0.0,
    )
    figure.text(
        0.77,
        0.25,
        "Finite DFPT dashed fits: K→Γ only.\n"
        "Zero-smearing DFPT dashed fit:\n"
        "local |d|≤0.003 only.\n"
        "No unsupported finite K→M fit is drawn.",
        fontsize=9.0,
        color="#555555",
        ha="left",
        va="top",
    )
    figure.subplots_adjust(left=0.10, right=0.74, bottom=0.14, top=0.89)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    png_path = output / "absolute_MLIP_full_EPC_LR_with_DFPT_fits.png"
    pdf_path = output / "absolute_MLIP_full_EPC_LR_with_DFPT_fits.pdf"
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    fit_csv = output / "DFPT_fitted_curves.csv"
    with fit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_curve_rows[0]))
        writer.writeheader()
        writer.writerows(fit_curve_rows)

    summary = {
        "status": "absolute_overlay_with_smooth_DFPT_fits_generated",
        "plot_semantics": {
            "solid": "fixed-background MLIP + full EPC long-range response",
            "dashed": "smooth DFPT fit",
            "open_markers": "DFPT samples or zero-smearing convergence anchors",
        },
        "finite_DFPT_fit": {
            "formula": (
                "omega(d)=omega_K+a*(sqrt(d^2+w^2)-w)+b*d^2"
            ),
            "scope": "K-to-Gamma, fit using all available d<=0.06 points; displayed to d=0.03",
            "by_smearing": {
                f"{smearing:.10f}": {
                    **{
                        key: value
                        for key, value in finite_fits[smearing].items()
                        if key != "prediction_at_samples_cm-1"
                    },
                    "DFPT_kgrid": finite_data[smearing]["kgrid"],
                }
                for smearing in SMEARINGS[1:]
            },
        },
        "zero_DFPT_fit": {
            "formula": "piecewise linear local cusp through K and d=0.003 1/Nk limits",
            "scope": "only -0.003<=signed_distance<=0.003",
            "KG_depth_cm-1": zero_KG_depth,
            "KM_depth_cm-1": zero_KM_depth,
        },
        "checks": {
            "all_finite_fit_RMSE_below_0p1_cm-1": bool(
                all(row["fit_RMSE_cm-1"] < 0.1 for row in finite_fits.values())
            ),
            "no_finite_KM_DFPT_fit_drawn": True,
            "zero_fit_not_extrapolated_beyond_d003": True,
        },
        "limitations": [
            "finite-smearing DFPT is available only on K-to-Gamma, so dashed finite fits stop at K",
            "the zero-smearing converged reference supports only K and d=0.003; no full-window zero DFPT fit is claimed",
            "finite direct DFPT uses the existing k144, k144, and k120 calculations rather than new k-grid extrapolations",
        ],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "fit_csv": str(fit_csv.resolve()),
        },
    }
    summary_path = output / "absolute_overlay_DFPT_fits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
