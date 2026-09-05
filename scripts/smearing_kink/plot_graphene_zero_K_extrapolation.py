#!/usr/bin/env python3
"""Plot the zero-smearing q=K frequency extrapolation versus electronic k mesh."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
KGRIDS = np.asarray([192, 240, 288], int)
FIXED_Q = np.asarray([-0.003, 0.0, 0.003], float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-csv",
        type=Path,
        default=BASE
        / "E34_zero_raw_DFPT_kgrid_diagnostic"
        / "zero_smearing_raw_DFPT_kgrid_diagnostic.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E35_zero_K_frequency_extrapolation",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def linear_extrapolation(N: np.ndarray, frequency: np.ndarray) -> dict:
    inverse_N = 1.0 / np.asarray(N, float)
    frequency = np.asarray(frequency, float)
    slope, intercept = np.polyfit(inverse_N, frequency, 1)
    fitted = intercept + slope * inverse_N
    pairwise_intercepts = []
    for first, second in itertools.combinations(range(len(N)), 2):
        _, pair_intercept = np.polyfit(
            inverse_N[[first, second]], frequency[[first, second]], 1
        )
        pairwise_intercepts.append(float(pair_intercept))
    quadratic = np.polyfit(inverse_N, frequency, 2)
    return {
        "slope_cm-1_times_N": float(slope),
        "intercept_N_infinity_cm-1": float(intercept),
        "fitted_frequency_cm-1": fitted,
        "residual_cm-1": frequency - fitted,
        "max_abs_residual_cm-1": float(np.max(np.abs(frequency - fitted))),
        "pairwise_intercepts_cm-1": pairwise_intercepts,
        "pairwise_intercept_min_cm-1": float(min(pairwise_intercepts)),
        "pairwise_intercept_max_cm-1": float(max(pairwise_intercepts)),
        "quadratic_in_1_over_N_intercept_cm-1": float(quadratic[2]),
    }


def select_raw_frequency(
    rows: list[dict[str, str]], kgrid: int, signed_distance: float
) -> float:
    matches = [
        float(row["frequency_cm-1"])
        for row in rows
        if row["series"] == "raw_DFPT"
        and int(row["kgrid"]) == kgrid
        and np.isclose(
            float(row["signed_distance"]), signed_distance, atol=1.0e-12
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one raw value at k{kgrid}, d={signed_distance}"
        )
    return matches[0]


def load_model(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    model_rows = [row for row in rows if row["series"] == "MLIP_plus_full_EPC_LR"]
    x = np.asarray([float(row["signed_distance"]) for row in model_rows], float)
    frequency = np.asarray([float(row["frequency_cm-1"]) for row in model_rows], float)
    order = np.argsort(x)
    return {"x": x[order], "frequency_cm-1": frequency[order]}


def main() -> int:
    args = parse_args()
    rows = read_csv(args.diagnostic_csv)
    model = load_model(rows)

    raw_by_q = {}
    fits = {}
    for q_value in FIXED_Q:
        raw_frequency = np.asarray(
            [
                select_raw_frequency(rows, int(kgrid), float(q_value))
                for kgrid in KGRIDS
            ],
            float,
        )
        raw_by_q[float(q_value)] = raw_frequency
        fits[float(q_value)] = linear_extrapolation(KGRIDS, raw_frequency)

    K_fit = fits[0.0]
    K_infinity = float(K_fit["intercept_N_infinity_cm-1"])
    fixed_q_infinity = np.asarray(
        [fits[float(q)]["intercept_N_infinity_cm-1"] for q in FIXED_Q],
        float,
    )
    extrapolated_depth = fixed_q_infinity - K_infinity
    model_K_index = int(np.argmin(np.abs(model["x"])))
    model_K = float(model["frequency_cm-1"][model_K_index])
    reanchor_shift = K_infinity - model_K
    reanchored_model = model["frequency_cm-1"] + reanchor_shift

    colors = {192: "#0072B2", 240: "#D55E00", 288: "#CC79A7"}
    markers = {192: "o", 240: "s", 288: "D"}
    green = "#009E73"

    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
    convergence_axis, spectrum_axis = axes

    inverse_N_scaled = 1000.0 / KGRIDS.astype(float)
    fit_x_scaled = np.linspace(0.0, 1.04 * np.max(inverse_N_scaled), 300)
    fit_y = K_infinity + K_fit["slope_cm-1_times_N"] * fit_x_scaled / 1000.0
    convergence_axis.plot(
        fit_x_scaled,
        fit_y,
        color=green,
        lw=2.0,
        ls="--",
        zorder=2,
    )
    for kgrid, x_value, frequency in zip(
        KGRIDS, inverse_N_scaled, raw_by_q[0.0]
    ):
        convergence_axis.plot(
            x_value,
            frequency,
            ls="none",
            marker=markers[int(kgrid)],
            ms=7.0,
            markerfacecolor="white",
            markeredgecolor=colors[int(kgrid)],
            markeredgewidth=1.5,
            zorder=4,
        )
        convergence_axis.annotate(
            f"k{int(kgrid)}",
            (x_value, frequency),
            xytext=(5, -14 if int(kgrid) == 240 else 7),
            textcoords="offset points",
            fontsize=9.0,
            color=colors[int(kgrid)],
        )

    pair_min = K_fit["pairwise_intercept_min_cm-1"]
    pair_max = K_fit["pairwise_intercept_max_cm-1"]
    convergence_axis.errorbar(
        [0.0],
        [K_infinity],
        yerr=np.asarray([[K_infinity - pair_min], [pair_max - K_infinity]]),
        fmt="*",
        ms=13,
        color=green,
        markeredgecolor="#005E46",
        markeredgewidth=0.8,
        elinewidth=1.4,
        capsize=4,
        zorder=5,
    )
    convergence_axis.annotate(
        rf"$N\to\infty$: {K_infinity:.3f} cm$^{{-1}}$",
        (0.0, K_infinity),
        xytext=(16, -4),
        textcoords="offset points",
        fontsize=9.5,
        color="#006B50",
        va="center",
    )
    convergence_axis.axvline(0.0, color="#AAAAAA", lw=0.9, zorder=0)
    convergence_axis.set_xlim(-0.25, 5.65)
    convergence_axis.set_xlabel(r"electronic mesh spacing, $10^3/N$  ($N\times N\times1$)")
    convergence_axis.set_ylabel(r"DFPT A$'$ frequency at phonon $q=K$ (cm$^{-1}$)")
    convergence_axis.set_title(r"(a) Extrapolate the phonon $q=K$ frequency to $1/N=0$")
    convergence_axis.grid(alpha=0.18)

    k288_frequency = np.asarray(
        [raw_by_q[float(q)][2] for q in FIXED_Q], float
    )
    spectrum_axis.plot(
        model["x"],
        model["frequency_cm-1"],
        color="#888888",
        lw=2.0,
        zorder=1,
    )
    spectrum_axis.plot(
        model["x"],
        reanchored_model,
        color="#111111",
        lw=2.6,
        zorder=2,
    )
    spectrum_axis.plot(
        FIXED_Q,
        k288_frequency,
        ls="none",
        marker="D",
        ms=6.0,
        markerfacecolor="white",
        markeredgecolor=colors[288],
        markeredgewidth=1.4,
        zorder=4,
    )
    spectrum_axis.plot(
        FIXED_Q,
        fixed_q_infinity,
        ls="none",
        marker="*",
        ms=12.0,
        markerfacecolor=green,
        markeredgecolor="#005E46",
        markeredgewidth=0.8,
        zorder=5,
    )
    spectrum_axis.annotate(
        rf"extrapolated K = {K_infinity:.3f} cm$^{{-1}}$",
        (0.0, K_infinity),
        xytext=(9, -21),
        textcoords="offset points",
        fontsize=9.2,
        color="#006B50",
        ha="left",
    )
    spectrum_axis.axvline(0.0, color="#AAAAAA", lw=0.9, zorder=0)
    spectrum_axis.set_xlim(-0.006, 0.006)
    spectrum_axis.set_ylim(1281.0, 1289.0)
    spectrum_axis.set_xlabel("signed distance from K (K→Γ < 0; K→M > 0)")
    spectrum_axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    spectrum_axis.set_title("(b) Insert the fixed-q extrapolations into the spectrum")
    spectrum_axis.grid(alpha=0.18)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=green,
            lw=2.0,
            ls="--",
            label=r"linear fit: $\omega_K=\omega_K(\infty)+A/N$",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markersize=12,
            markerfacecolor=green,
            markeredgecolor="#005E46",
            label=r"fixed-q $N\to\infty$ extrapolation",
        ),
        Line2D(
            [0],
            [0],
            color="#888888",
            lw=2.0,
            label="MLIP + full EPC LR (k288 K anchor)",
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            lw=2.6,
            label=r"same model reanchored to DFPT $K_\infty$",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="D",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=colors[288],
            label="raw DFPT k288",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.815, 0.84),
        frameon=False,
        title="smearing/degauss = 0 Ry",
        borderaxespad=0.0,
    )
    figure.text(
        0.815,
        0.44,
        f"Linear 3-grid K intercept: {K_infinity:.3f} cm⁻¹\n"
        f"Pairwise intercept range: {pair_min:.3f}–{pair_max:.3f} cm⁻¹\n"
        f"Maximum fit residual: {K_fit['max_abs_residual_cm-1']:.3f} cm⁻¹\n"
        f"Quadratic-in-1/N sensitivity: "
        f"{K_fit['quadratic_in_1_over_N_intercept_cm-1']:.3f} cm⁻¹\n\n"
        f"Reanchoring shift applied in panel (b):\n"
        f"+{reanchor_shift:.3f} cm⁻¹; curve shape unchanged.\n\n"
        "The pairwise range is a sensitivity diagnostic,\n"
        "not a statistical confidence interval.",
        fontsize=9.1,
        color="#555555",
        ha="left",
        va="top",
        linespacing=1.35,
    )
    figure.suptitle(
        "Graphene zero-smearing K-point frequency: electronic k-grid extrapolation",
        x=0.405,
        y=0.98,
        fontsize=13.0,
    )
    figure.subplots_adjust(
        left=0.075, right=0.79, bottom=0.15, top=0.88, wspace=0.28
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "zero_K_frequency_1_over_N_extrapolation.png"
    pdf_path = args.output_dir / "zero_K_frequency_1_over_N_extrapolation.pdf"
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    output_rows = []
    for q_value in FIXED_Q:
        fit = fits[float(q_value)]
        for kgrid, frequency, fitted, residual in zip(
            KGRIDS,
            raw_by_q[float(q_value)],
            fit["fitted_frequency_cm-1"],
            fit["residual_cm-1"],
        ):
            output_rows.append(
                {
                    "signed_distance": float(q_value),
                    "kgrid_N": int(kgrid),
                    "inverse_N": 1.0 / float(kgrid),
                    "raw_DFPT_frequency_cm-1": float(frequency),
                    "linear_fit_frequency_cm-1": float(fitted),
                    "fit_residual_cm-1": float(residual),
                    "N_infinity_frequency_cm-1": fit[
                        "intercept_N_infinity_cm-1"
                    ],
                }
            )
    csv_path = args.output_dir / "zero_fixed_q_1_over_N_extrapolation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    fit_summary = {}
    for q_value in FIXED_Q:
        fit = fits[float(q_value)]
        fit_summary[f"{q_value:+.3f}"] = {
            "raw_frequency_cm-1_by_N": {
                str(int(kgrid)): float(frequency)
                for kgrid, frequency in zip(KGRIDS, raw_by_q[float(q_value)])
            },
            "linear_1_over_N_intercept_cm-1": fit[
                "intercept_N_infinity_cm-1"
            ],
            "linear_1_over_N_slope_cm-1_times_N": fit["slope_cm-1_times_N"],
            "linear_fit_max_abs_residual_cm-1": fit[
                "max_abs_residual_cm-1"
            ],
            "pairwise_intercepts_cm-1": fit["pairwise_intercepts_cm-1"],
            "quadratic_1_over_N_intercept_cm-1": fit[
                "quadratic_in_1_over_N_intercept_cm-1"
            ],
        }
    summary = {
        "status": "zero_K_frequency_1_over_N_extrapolation_generated",
        "electronic_integration": "tetrahedra_opt (no degauss)",
        "fit_model": "omega(q,N)=omega(q,infinity)+A(q)/N, N is the linear size of the N x N x 1 electronic k mesh",
        "fixed_q_fits": fit_summary,
        "K_frequency": {
            "linear_1_over_N_intercept_cm-1": K_infinity,
            "pairwise_intercept_range_cm-1": [pair_min, pair_max],
            "quadratic_1_over_N_sensitivity_cm-1": K_fit[
                "quadratic_in_1_over_N_intercept_cm-1"
            ],
            "accepted_as_frozen_reference": False,
        },
        "extrapolated_d003_depth_cm-1": {
            "KG": float(extrapolated_depth[0]),
            "KM": float(extrapolated_depth[2]),
        },
        "model_reanchoring_diagnostic": {
            "original_model_K_cm-1": model_K,
            "DFPT_extrapolated_K_cm-1": K_infinity,
            "common_shift_cm-1": reanchor_shift,
            "shape_changed": False,
        },
        "limitations": [
            "only three electronic k grids are available",
            "linear 1/N and quadratic 1/N sensitivity models cannot be distinguished with these data",
            "the reanchored model in panel (b) is a diagnostic common-frequency shift, not the original absolute calibration",
        ],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    summary_path = args.output_dir / "zero_K_frequency_extrapolation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
