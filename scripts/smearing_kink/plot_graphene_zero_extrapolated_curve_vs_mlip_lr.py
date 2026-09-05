#!/usr/bin/env python3
"""Compare the local zero-smearing DFPT N→∞ cusp with MLIP+EPC LR."""

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
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-csv",
        type=Path,
        default=BASE
        / "E31_fixed_background_smearing_sweep"
        / "fixed_background_smearing_sweep_Aprime.csv",
    )
    parser.add_argument(
        "--extrapolation-csv",
        type=Path,
        default=BASE
        / "E35_zero_K_frequency_extrapolation"
        / "zero_fixed_q_1_over_N_extrapolation.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E36_zero_DFPT_extrapolated_curve_vs_MLIP_LR",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_zero_model(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        row
        for row in read_csv(path)
        if np.isclose(float(row["smearing_degauss_Ry"]), 0.0, atol=1.0e-14)
        and abs(float(row["signed_distance"])) <= 0.003 + 1.0e-12
    ]
    x = np.asarray([float(row["signed_distance"]) for row in rows], float)
    frequency = np.asarray(
        [float(row["controlled_MLIP_plus_full_EPC_cm-1"]) for row in rows],
        float,
    )
    order = np.argsort(x)
    x = x[order]
    frequency = frequency[order]
    if len(x) != 13 or not np.isclose(x[0], -0.003) or not np.isclose(
        x[-1], 0.003
    ):
        raise RuntimeError("expected 13 zero-smearing model points on |d|<=0.003")
    return x, frequency


def load_fixed_q_limits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    values: dict[float, list[float]] = {}
    for row in rows:
        q_value = round(float(row["signed_distance"]), 12)
        values.setdefault(q_value, []).append(
            float(row["N_infinity_frequency_cm-1"])
        )
    points = []
    for q_value, frequency_values in values.items():
        if np.ptp(frequency_values) > 1.0e-9:
            raise RuntimeError(f"inconsistent fixed-q intercept at d={q_value}")
        points.append((q_value, float(np.mean(frequency_values))))
    points.sort()
    x = np.asarray([point[0] for point in points], float)
    frequency = np.asarray([point[1] for point in points], float)
    expected = np.asarray([-0.003, 0.0, 0.003])
    if len(x) != 3 or not np.allclose(x, expected, atol=1.0e-12):
        raise RuntimeError(f"unexpected fixed-q extrapolation anchors: {x}")
    return x, frequency


def interpolate_model_with_cusp(
    x: np.ndarray, frequency: np.ndarray, dense_x: np.ndarray
) -> np.ndarray:
    result = np.empty_like(dense_x)
    for select_input, select_output in (
        (x <= 0.0, dense_x <= 0.0),
        (x >= 0.0, dense_x >= 0.0),
    ):
        interpolator = PchipInterpolator(x[select_input], frequency[select_input])
        result[select_output] = interpolator(dense_x[select_output])
    return result


def piecewise_linear_cusp(
    dense_x: np.ndarray, anchor_x: np.ndarray, anchor_y: np.ndarray
) -> np.ndarray:
    result = np.empty_like(dense_x)
    negative = dense_x < 0.0
    result[negative] = anchor_y[1] + (anchor_y[0] - anchor_y[1]) * (
        -dense_x[negative] / abs(anchor_x[0])
    )
    result[~negative] = anchor_y[1] + (anchor_y[2] - anchor_y[1]) * (
        dense_x[~negative] / anchor_x[2]
    )
    return result


def value_at(x: np.ndarray, y: np.ndarray, target: float) -> float:
    indices = np.flatnonzero(np.isclose(x, target, atol=1.0e-12))
    if len(indices) != 1:
        raise RuntimeError(f"expected exactly one value at d={target}")
    return float(y[indices[0]])


def main() -> int:
    args = parse_args()
    model_x, model_frequency = load_zero_model(args.model_csv)
    anchor_x, anchor_frequency = load_fixed_q_limits(args.extrapolation_csv)

    dense_x = np.linspace(-0.003, 0.003, 1201)
    model_dense = interpolate_model_with_cusp(model_x, model_frequency, dense_x)
    DFPT_dense = piecewise_linear_cusp(dense_x, anchor_x, anchor_frequency)

    model_K = value_at(model_x, model_frequency, 0.0)
    DFPT_K = value_at(anchor_x, anchor_frequency, 0.0)
    reanchor_shift = DFPT_K - model_K
    reanchored_model_dense = model_dense + reanchor_shift
    aligned_difference = reanchored_model_dense - DFPT_dense

    model_depth = {
        "KG": value_at(model_x, model_frequency, -0.003) - model_K,
        "KM": value_at(model_x, model_frequency, 0.003) - model_K,
    }
    DFPT_depth = {
        "KG": value_at(anchor_x, anchor_frequency, -0.003) - DFPT_K,
        "KM": value_at(anchor_x, anchor_frequency, 0.003) - DFPT_K,
    }
    depth_error = {
        direction: model_depth[direction] - DFPT_depth[direction]
        for direction in ("KG", "KM")
    }
    depth_relative_error = {
        direction: depth_error[direction] / DFPT_depth[direction]
        for direction in ("KG", "KM")
    }

    gray = "#888888"
    black = "#111111"
    green = "#009E73"
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.45))
    absolute_axis, aligned_axis = axes
    plot_x = 1000.0 * dense_x
    anchor_plot_x = 1000.0 * anchor_x

    absolute_axis.plot(
        plot_x,
        model_dense,
        color=gray,
        lw=2.5,
        zorder=2,
    )
    absolute_axis.plot(
        plot_x,
        DFPT_dense,
        color=green,
        lw=2.3,
        ls="--",
        zorder=3,
    )
    absolute_axis.plot(
        anchor_plot_x,
        anchor_frequency,
        ls="none",
        marker="*",
        ms=11.5,
        markerfacecolor=green,
        markeredgecolor="#005E46",
        markeredgewidth=0.8,
        zorder=5,
    )
    absolute_axis.annotate(
        rf"K offset = {model_K - DFPT_K:+.3f} cm$^{{-1}}$",
        (0.0, model_K),
        xytext=(10, 9),
        textcoords="offset points",
        fontsize=9.4,
        color="#555555",
    )

    aligned_axis.plot(
        plot_x,
        reanchored_model_dense,
        color=black,
        lw=2.7,
        zorder=2,
    )
    aligned_axis.plot(
        plot_x,
        DFPT_dense,
        color=green,
        lw=2.3,
        ls="--",
        zorder=3,
    )
    aligned_axis.plot(
        anchor_plot_x,
        anchor_frequency,
        ls="none",
        marker="*",
        ms=11.5,
        markerfacecolor=green,
        markeredgecolor="#005E46",
        markeredgewidth=0.8,
        zorder=5,
    )
    aligned_axis.fill_between(
        plot_x,
        reanchored_model_dense,
        DFPT_dense,
        color=green,
        alpha=0.10,
        linewidth=0.0,
        zorder=1,
    )

    for axis in axes:
        axis.axvline(0.0, color="#AAAAAA", lw=0.9, zorder=0)
        axis.set_xlim(-3.15, 3.15)
        axis.set_xlabel(
            r"signed distance from K, $10^3d$ (K→Γ < 0; K→M > 0)"
        )
        axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
        axis.grid(alpha=0.18)

    absolute_axis.set_ylim(1281.8, 1286.55)
    absolute_axis.set_title("(a) Absolute frequencies with current calibration")
    aligned_axis.set_ylim(1283.95, 1286.55)
    aligned_axis.set_title("(b) Cusp shape after using the extrapolated K anchor")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=gray,
            lw=2.5,
            label="MLIP + full EPC LR (original k288 K anchor)",
        ),
        Line2D(
            [0],
            [0],
            color=black,
            lw=2.7,
            label=r"same MLIP + LR curve reanchored to DFPT $K_\infty$",
        ),
        Line2D(
            [0],
            [0],
            color=green,
            lw=2.3,
            ls="--",
            label=r"local DFPT $N\to\infty$ cusp curve",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markersize=11.5,
            markerfacecolor=green,
            markeredgecolor="#005E46",
            label=r"fixed-q DFPT $N\to\infty$ anchors",
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
        0.47,
        f"DFPT K∞: {DFPT_K:.3f} cm⁻¹\n"
        f"Original model K: {model_K:.3f} cm⁻¹\n"
        f"Common reanchoring shift: +{reanchor_shift:.3f} cm⁻¹\n\n"
        "d=0.003 cusp depth (KG / KM):\n"
        f"DFPT N→∞: {DFPT_depth['KG']:.3f} / "
        f"{DFPT_depth['KM']:.3f} cm⁻¹\n"
        f"MLIP + LR: {model_depth['KG']:.3f} / "
        f"{model_depth['KM']:.3f} cm⁻¹\n"
        f"Signed error: {depth_relative_error['KG'] * 100:+.2f}% / "
        f"{depth_relative_error['KM'] * 100:+.2f}%\n\n"
        "DFPT dashed curve is supported only on |d|≤0.003.\n"
        "It is piecewise linear through three fixed-q intercepts.",
        fontsize=9.1,
        color="#555555",
        ha="left",
        va="top",
        linespacing=1.35,
    )
    figure.suptitle(
        "Graphene zero-smearing K cusp: extrapolated DFPT curve vs MLIP + long-range correction",
        x=0.405,
        y=0.98,
        fontsize=13.0,
    )
    figure.subplots_adjust(
        left=0.075, right=0.79, bottom=0.16, top=0.88, wspace=0.28
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "zero_DFPT_extrapolated_curve_vs_MLIP_LR.png"
    pdf_path = args.output_dir / "zero_DFPT_extrapolated_curve_vs_MLIP_LR.pdf"
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    output_rows = []
    for x_value, original, reanchored, DFPT_value, difference in zip(
        dense_x,
        model_dense,
        reanchored_model_dense,
        DFPT_dense,
        aligned_difference,
    ):
        output_rows.append(
            {
                "signed_distance": float(x_value),
                "MLIP_full_EPC_LR_original_cm-1": float(original),
                "MLIP_full_EPC_LR_reanchored_cm-1": float(reanchored),
                "DFPT_N_infinity_local_cusp_cm-1": float(DFPT_value),
                "K_aligned_model_minus_DFPT_cm-1": float(difference),
            }
        )
    csv_path = args.output_dir / "zero_DFPT_extrapolated_curve_vs_MLIP_LR.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "status": "zero_DFPT_extrapolated_curve_vs_MLIP_LR_generated",
        "smearing_degauss_Ry": 0.0,
        "scope": "local |signed distance from K| <= 0.003",
        "DFPT_curve": {
            "construction": "piecewise-linear zero-smearing cusp through independently fitted fixed-q N-infinity intercepts",
            "anchors": [
                {
                    "signed_distance": float(x_value),
                    "frequency_cm-1": float(frequency),
                }
                for x_value, frequency in zip(anchor_x, anchor_frequency)
            ],
            "K_frequency_cm-1": DFPT_K,
            "d003_depth_cm-1": DFPT_depth,
        },
        "MLIP_plus_full_EPC_LR": {
            "original_K_frequency_cm-1": model_K,
            "DFPT_K_reanchoring_shift_cm-1": reanchor_shift,
            "shape_changed_by_reanchoring": False,
            "d003_depth_cm-1": model_depth,
        },
        "K_aligned_shape_comparison": {
            "d003_signed_depth_error_cm-1": depth_error,
            "d003_signed_relative_depth_error": depth_relative_error,
            "curve_RMSE_cm-1": float(np.sqrt(np.mean(aligned_difference**2))),
            "curve_max_abs_error_cm-1": float(
                np.max(np.abs(aligned_difference))
            ),
        },
        "limitations": [
            "the DFPT curve is local and is not extended beyond |d|=0.003",
            "only three fixed q anchors support the piecewise-linear DFPT cusp",
            "the absolute K intercept uses a three-grid linear 1/N extrapolation and is not a frozen converged reference",
            "the reanchored MLIP curve is a diagnostic common shift of the original curve",
        ],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    summary_path = args.output_dir / "zero_DFPT_extrapolated_curve_vs_MLIP_LR_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
