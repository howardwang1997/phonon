#!/usr/bin/env python3
"""Plot an extended zero-smearing K region with explicit DFPT support limits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plot_graphene_zero_wide_extrapolation_sketch as e37  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"


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
        "--extended-model-csv",
        type=Path,
        default=BASE / "E38_epc_zero_extended_curve/zero_dense_curve.csv",
    )
    parser.add_argument(
        "--extended-model-summary",
        type=Path,
        default=BASE / "E38_epc_zero_extended_curve/zero_dense_curve_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E39_zero_extended_region_sketch",
    )
    parser.add_argument(
        "--publication-pdf",
        type=Path,
        default=ROOT / "output/pdf/graphene_zero_K_extended_region_sketch.pdf",
    )
    return parser.parse_args()


def load_extended_model(path: Path) -> dict[str, np.ndarray]:
    rows = e37.read_csv(path)
    x = np.asarray([float(row["signed_distance"]) for row in rows], float)
    frequency = np.asarray(
        [float(row["MLIP_plus_full_EPC_cm-1"]) for row in rows], float
    )
    order = np.argsort(x)
    x = x[order]
    frequency = frequency[order]
    if len(x) != 241 or not np.isclose(x[0], -0.06) or not np.isclose(
        x[-1], 0.06
    ):
        raise RuntimeError("expected a 241-point model curve spanning d=+/-0.06")
    return {"x": x, "frequency_cm-1": frequency}


def plot_branch(axis, branch: dict, sign: int, *, relative: bool) -> None:
    distance = branch["distance"]
    x_value = sign * e37.Q_DISTANCE_PER_D * distance
    y_key = "central_relative_cm-1" if relative else "central_frequency_cm-1"
    y_value = branch[y_key]
    for lower, upper, color, linestyle, linewidth in (
        (0.0, 0.003, "#009E73", "--", 2.5),
        (0.003, 0.007, "#3FA985", "--", 2.3),
        (0.007, 0.023, "#83C5AE", ":", 2.4),
    ):
        select = (distance >= lower - 1.0e-14) & (
            distance <= upper + 1.0e-14
        )
        order = np.argsort(x_value[select])
        axis.plot(
            x_value[select][order],
            y_value[select][order],
            color=color,
            ls=linestyle,
            lw=linewidth,
            zorder=5,
        )


def add_support_shading(axis, full_distance: float) -> None:
    outer = e37.Q_DISTANCE_PER_D * 0.023
    limit = e37.Q_DISTANCE_PER_D * full_distance
    axis.axvspan(-limit, -outer, color="#777777", alpha=0.07, lw=0, zorder=0)
    axis.axvspan(outer, limit, color="#777777", alpha=0.07, lw=0, zorder=0)


def add_raw_points(
    axis,
    raw: dict[int, dict[float, float]],
    styles: dict,
    *,
    relative: bool,
) -> None:
    for kgrid, points in raw.items():
        style = styles[kgrid]
        x = np.asarray(sorted(points), float)
        y = np.asarray([points[float(value)] for value in x], float)
        if relative:
            y = y - points[0.0]
        axis.plot(
            e37.Q_DISTANCE_PER_D * x,
            y,
            ls="none",
            marker=style["marker"],
            ms=style["size"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.25,
            zorder=7,
        )


def main() -> int:
    args = parse_args()
    _, raw = e37.load_data(args.diagnostic_csv)
    model = load_extended_model(args.extended_model_csv)
    model_summary = json.loads(
        args.extended_model_summary.read_text(encoding="utf-8")
    )
    variants = e37.three_grid_variants(raw)
    KG = e37.build_branch(raw, variants, sign=-1)
    KM = e37.build_branch(raw, variants, sign=1)
    branches = {-1: KG, 1: KM}

    black = "#111111"
    green = "#009E73"
    raw_styles = {
        192: {"color": "#0072B2", "marker": "o", "size": 5.2},
        240: {"color": "#D55E00", "marker": "s", "size": 5.6},
        288: {"color": "#CC79A7", "marker": "D", "size": 5.5},
    }
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 5.8))
    extended_axis, supported_axis = axes

    model_K = e37.model_K(model)
    extended_axis.plot(
        e37.Q_DISTANCE_PER_D * model["x"],
        model["frequency_cm-1"],
        color=black,
        lw=2.7,
        zorder=3,
    )
    supported_select = np.abs(model["x"]) <= 0.025 + 1.0e-12
    supported_axis.plot(
        e37.Q_DISTANCE_PER_D * model["x"][supported_select],
        model["frequency_cm-1"][supported_select] - model_K,
        color=black,
        lw=2.7,
        zorder=3,
    )
    add_support_shading(extended_axis, 0.06)
    add_support_shading(supported_axis, 0.025)

    for sign, branch in branches.items():
        x_value = sign * e37.Q_DISTANCE_PER_D * branch["distance"]
        order = np.argsort(x_value)
        extended_axis.fill_between(
            x_value[order],
            branch["lower_cm-1"][order],
            branch["upper_cm-1"][order],
            color=green,
            alpha=0.13,
            lw=0.0,
            zorder=1,
        )
        supported_axis.fill_between(
            x_value[order],
            branch["relative_lower_cm-1"][order],
            branch["relative_upper_cm-1"][order],
            color=green,
            alpha=0.13,
            lw=0.0,
            zorder=1,
        )
        plot_branch(extended_axis, branch, sign, relative=False)
        plot_branch(supported_axis, branch, sign, relative=True)

    add_raw_points(extended_axis, raw, raw_styles, relative=False)
    add_raw_points(supported_axis, raw, raw_styles, relative=True)

    central_K = variants[0.0]["linear_all"]
    fixed_x = np.asarray([-0.003, 0.0, 0.003])
    fixed_y = np.asarray(
        [
            variants[-0.003]["linear_all"],
            central_K,
            variants[0.003]["linear_all"],
        ]
    )
    star_style = {
        "ls": "none",
        "marker": "*",
        "ms": 11.5,
        "markerfacecolor": green,
        "markeredgecolor": "#005E46",
        "markeredgewidth": 0.8,
        "zorder": 9,
    }
    extended_axis.plot(e37.Q_DISTANCE_PER_D * fixed_x, fixed_y, **star_style)
    supported_axis.plot(
        e37.Q_DISTANCE_PER_D * fixed_x, fixed_y - central_K, **star_style
    )

    d007_x = np.asarray([-0.007, 0.007])
    d007_y = np.asarray(
        [KG["d007"]["intercept_cm-1"], KM["d007"]["intercept_cm-1"]]
    )
    triangle_style = {
        "ls": "none",
        "marker": "^",
        "ms": 6.2,
        "markerfacecolor": "white",
        "markeredgecolor": "#3FA985",
        "markeredgewidth": 1.3,
        "zorder": 8,
    }
    extended_axis.plot(
        e37.Q_DISTANCE_PER_D * d007_x, d007_y, **triangle_style
    )
    supported_axis.plot(
        e37.Q_DISTANCE_PER_D * d007_x,
        d007_y - central_K,
        **triangle_style,
    )

    for axis in axes:
        axis.axvline(0.0, color="#AAAAAA", lw=0.9, zorder=0)
        axis.grid(alpha=0.18)
        axis.set_xlabel(r"signed $|q-K|/(2\pi/a)$  (K→Γ < 0; K→M > 0)")
        axis.text(
            0.5,
            0.965,
            r"$\Gamma\ \leftarrow\ K\ \rightarrow\ M$",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10.2,
            color="#444444",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
                "pad": 1.5,
            },
            zorder=10,
        )

    extended_axis.set_xlim(-0.04, 0.04)
    extended_axis.set_ylim(1280.0, 1311.5)
    extended_axis.set_ylabel(r"A$_1'$ frequency (cm$^{-1}$)")
    extended_axis.set_title("(a) Extended view: model to |d|=0.06")
    extended_axis.text(
        -0.028,
        1309.8,
        "no zero-smearing\nDFPT anchors",
        ha="center",
        va="top",
        fontsize=8.4,
        color="#777777",
    )
    extended_axis.text(
        0.028,
        1309.8,
        "no zero-smearing\nDFPT anchors",
        ha="center",
        va="top",
        fontsize=8.4,
        color="#777777",
    )

    supported_axis.set_xlim(
        -e37.Q_DISTANCE_PER_D * 0.025,
        e37.Q_DISTANCE_PER_D * 0.025,
    )
    supported_axis.set_ylim(-0.5, 19.3)
    supported_axis.set_ylabel(
        r"K-referenced rise, $\omega(q)-\omega(K)$ (cm$^{-1}$)"
    )
    supported_axis.set_title("(b) DFPT-supported window: each series referenced to its own K")

    legend_handles = [
        Line2D(
            [0], [0], color=black, lw=2.7, label="MLIP + full EPC LR"
        ),
        Line2D(
            [0],
            [0],
            color=green,
            lw=2.4,
            ls="--",
            label=r"central DFPT $N\to\infty$ guide",
        ),
        Patch(
            facecolor=green,
            alpha=0.13,
            edgecolor="none",
            label="fit/support envelope",
        ),
        Patch(
            facecolor="#777777",
            alpha=0.07,
            edgecolor="none",
            label="outside current zero-smearing DFPT support",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markersize=11.5,
            markerfacecolor=green,
            markeredgecolor="#005E46",
            label="three-grid fixed-q intercept",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="^",
            markersize=6.2,
            markerfacecolor="white",
            markeredgecolor="#3FA985",
            label="two-grid d=0.007 intercept",
        ),
    ]
    for kgrid in (192, 240, 288):
        style = raw_styles[kgrid]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="none",
                marker=style["marker"],
                markersize=style["size"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                label=f"raw DFPT k{kgrid}",
            )
        )
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
        0.31,
        "Black curve outside |d|=0.023:\n"
        "stored Wannier/full-EPC replay only;\n"
        "no new DFT and no q-shape fit.\n\n"
        "The green guide stops at the last raw\n"
        "zero-smearing DFPT location. It is dotted\n"
        "for 0.007 < |d| <= 0.023 because this\n"
        "region has k192 support only.\n\n"
        "Layout reference: Piscanec et al.,\n"
        "Phys. Rev. Lett. 93, 185503 (2004), Fig. 2.",
        fontsize=8.9,
        color="#555555",
        ha="left",
        va="top",
        linespacing=1.32,
    )
    figure.suptitle(
        r"Graphene K-point $A_1'$ branch: extended zero-smearing view",
        x=0.405,
        y=0.985,
        fontsize=13.2,
    )
    figure.subplots_adjust(
        left=0.075, right=0.79, bottom=0.16, top=0.86, wspace=0.28
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.publication_pdf.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "zero_extended_region_sketch.png"
    pdf_path = args.output_dir / "zero_extended_region_sketch.pdf"
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    figure.savefig(args.publication_pdf, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    summary = {
        "status": "extended_zero_smearing_region_sketch_generated",
        "extended_model": {
            "signed_distance_d": [-0.06, 0.06],
            "signed_q_distance_2pi_over_a": [-0.04, 0.04],
            "number_of_qpoints": len(model["x"]),
            "source": str(args.extended_model_csv.resolve()),
            "new_DFT_used": False,
            "monotone_KG": model_summary["checks"][
                "frequency_increases_monotonically_away_from_K"
            ]["KG"],
            "monotone_KM": model_summary["checks"][
                "frequency_increases_monotonically_away_from_K"
            ]["KM"],
        },
        "DFPT_support": {
            "guide_signed_distance_d": [-0.023, 0.023],
            "three_grid_region_abs_d_le": 0.003,
            "two_grid_region_abs_d_le": 0.007,
            "k192_only_region_abs_d_le": 0.023,
            "no_DFPT_extrapolation_drawn_beyond_abs_d": 0.023,
        },
        "interpretation": [
            "the extended black curve is a model prediction from the stored full-EPC response",
            "the gray outer regions have no zero-smearing direct DFPT anchors",
            "the green guide must not be extended into the gray regions without additional reference data",
        ],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "publication_pdf": str(args.publication_pdf.resolve()),
        },
    }
    summary_path = args.output_dir / "zero_extended_region_sketch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
