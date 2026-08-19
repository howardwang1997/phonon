#!/usr/bin/env python3
"""Draw a literature-style wide zero-smearing K-cusp extrapolation sketch."""

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
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
KGRIDS = np.asarray([192, 240, 288], int)
THREE_GRID_Q = (-0.003, 0.0, 0.003)
OUTER_D = np.asarray([0.011, 0.015, 0.019, 0.023], float)
Q_DISTANCE_PER_D = 2.0 / 3.0


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
        default=BASE / "E37_zero_wide_extrapolation_sketch",
    )
    parser.add_argument(
        "--publication-pdf",
        type=Path,
        default=ROOT
        / "output/pdf/graphene_zero_K_wide_extrapolation_sketch.pdf",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_data(path: Path) -> tuple[dict, dict[int, dict[float, float]]]:
    rows = read_csv(path)
    model_rows = [row for row in rows if row["series"] == "MLIP_plus_full_EPC_LR"]
    model_x = np.asarray([float(row["signed_distance"]) for row in model_rows])
    model_y = np.asarray([float(row["frequency_cm-1"]) for row in model_rows])
    order = np.argsort(model_x)
    model = {"x": model_x[order], "frequency_cm-1": model_y[order]}

    raw: dict[int, dict[float, float]] = {192: {}, 240: {}, 288: {}}
    for row in rows:
        if row["series"] != "raw_DFPT":
            continue
        kgrid = int(row["kgrid"])
        x_value = round(float(row["signed_distance"]), 12)
        raw[kgrid][x_value] = float(row["frequency_cm-1"])
    expected = {192: 13, 240: 5, 288: 3}
    actual = {kgrid: len(values) for kgrid, values in raw.items()}
    if actual != expected:
        raise RuntimeError(f"unexpected raw DFPT point inventory: {actual}")
    return model, raw


def intercept_and_slope(N: np.ndarray, frequency: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(1.0 / np.asarray(N, float), frequency, 1)
    return float(intercept), float(slope)


def three_grid_variants(raw: dict[int, dict[float, float]]) -> dict:
    result = {q_value: {} for q_value in THREE_GRID_Q}
    pairs = list(itertools.combinations(range(len(KGRIDS)), 2))
    for q_value in THREE_GRID_Q:
        frequency = np.asarray(
            [raw[int(kgrid)][q_value] for kgrid in KGRIDS], float
        )
        result[q_value]["linear_all"] = intercept_and_slope(
            KGRIDS.astype(float), frequency
        )[0]
        quadratic = np.polyfit(1.0 / KGRIDS.astype(float), frequency, 2)
        result[q_value]["quadratic_all"] = float(quadratic[2])
        for first, second in pairs:
            key = f"pair_{KGRIDS[first]}_{KGRIDS[second]}"
            result[q_value][key] = intercept_and_slope(
                KGRIDS[[first, second]].astype(float),
                frequency[[first, second]],
            )[0]
    return result


def two_grid_d007(raw: dict[int, dict[float, float]], sign: int) -> dict:
    q_value = round(sign * 0.007, 12)
    N = np.asarray([192.0, 240.0])
    frequency = np.asarray([raw[192][q_value], raw[240][q_value]])
    intercept, slope = intercept_and_slope(N, frequency)
    return {"intercept_cm-1": intercept, "A_cm-1_times_N": slope}


def tail_correction(A_d007: float, distance: float, tail: str) -> float:
    if tail == "zero":
        return 0.0
    if tail == "constant":
        return A_d007
    if not tail.startswith("exp_"):
        raise ValueError(f"unknown finite-grid tail model: {tail}")
    scale = float(tail.removeprefix("exp_"))
    return A_d007 * np.exp(-(distance - 0.007) / scale)


def branch_anchors(
    raw: dict[int, dict[float, float]],
    variants: dict,
    *,
    sign: int,
    variant_name: str,
    tail: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    d007 = two_grid_d007(raw, sign)
    distance = np.asarray([0.0, 0.003, 0.007, *OUTER_D], float)
    frequency = [
        variants[0.0][variant_name],
        variants[round(sign * 0.003, 12)][variant_name],
        d007["intercept_cm-1"],
    ]
    for d_value in OUTER_D:
        A_value = tail_correction(d007["A_cm-1_times_N"], d_value, tail)
        raw_frequency = raw[192][round(sign * float(d_value), 12)]
        frequency.append(raw_frequency - A_value / 192.0)
    frequency_array = np.asarray(frequency, float)
    if np.any(np.diff(frequency_array) <= 0.0):
        raise RuntimeError(
            f"non-monotonic extrapolation anchors for sign={sign}, "
            f"variant={variant_name}, tail={tail}: {frequency_array}"
        )
    return distance, frequency_array, d007


def support_padding(distance: np.ndarray) -> np.ndarray:
    """Heuristic support penalty, explicitly not a statistical interval."""
    distance = np.asarray(distance, float)
    return np.interp(
        distance,
        [0.0, 0.003, 0.007, 0.023],
        [0.0, 0.0, 0.12, 0.65],
    )


def build_branch(
    raw: dict[int, dict[float, float]],
    variants: dict,
    *,
    sign: int,
) -> dict:
    dense_d = np.linspace(0.0, 0.023, 921)
    variant_names = list(variants[0.0])
    tails = ("zero", "exp_0.004", "exp_0.006", "exp_0.008", "constant")
    ensemble = []
    for variant_name in variant_names:
        for tail in tails:
            anchor_d, anchor_y, _ = branch_anchors(
                raw,
                variants,
                sign=sign,
                variant_name=variant_name,
                tail=tail,
            )
            ensemble.append(PchipInterpolator(anchor_d, anchor_y)(dense_d))
    ensemble_array = np.asarray(ensemble)

    central_d, central_y, d007 = branch_anchors(
        raw,
        variants,
        sign=sign,
        variant_name="linear_all",
        tail="exp_0.006",
    )
    central = PchipInterpolator(central_d, central_y)(dense_d)
    padding = support_padding(dense_d)
    lower = np.min(ensemble_array, axis=0) - padding
    upper = np.max(ensemble_array, axis=0) + padding
    relative_ensemble = ensemble_array - ensemble_array[:, [0]]
    relative_lower = np.min(relative_ensemble, axis=0) - padding
    relative_upper = np.max(relative_ensemble, axis=0) + padding
    return {
        "distance": dense_d,
        "central_frequency_cm-1": central,
        "lower_cm-1": lower,
        "upper_cm-1": upper,
        "central_relative_cm-1": central - central[0],
        "relative_lower_cm-1": relative_lower,
        "relative_upper_cm-1": relative_upper,
        "anchor_distance": central_d,
        "anchor_frequency_cm-1": central_y,
        "d007": d007,
        "ensemble_count": len(ensemble),
    }


def model_K(model: dict) -> float:
    indices = np.flatnonzero(np.isclose(model["x"], 0.0, atol=1.0e-14))
    if len(indices) != 1:
        raise RuntimeError("expected one model K point")
    return float(model["frequency_cm-1"][indices[0]])


def plot_branch_segments(axis, branch: dict, sign: int, *, relative: bool) -> None:
    distance = branch["distance"]
    x_value = sign * Q_DISTANCE_PER_D * distance
    y_key = "central_relative_cm-1" if relative else "central_frequency_cm-1"
    y_value = branch[y_key]
    segments = (
        (0.0, 0.003, "#009E73", "--", 2.5),
        (0.003, 0.007, "#3FA985", "--", 2.3),
        (0.007, 0.023, "#83C5AE", ":", 2.4),
    )
    for lower, upper, color, linestyle, linewidth in segments:
        select = (distance >= lower - 1.0e-14) & (distance <= upper + 1.0e-14)
        order = np.argsort(x_value[select])
        axis.plot(
            x_value[select][order],
            y_value[select][order],
            color=color,
            ls=linestyle,
            lw=linewidth,
            zorder=4,
        )


def main() -> int:
    args = parse_args()
    model, raw = load_data(args.diagnostic_csv)
    variants = three_grid_variants(raw)
    KG = build_branch(raw, variants, sign=-1)
    KM = build_branch(raw, variants, sign=1)
    branches = {-1: KG, 1: KM}

    black = "#111111"
    green = "#009E73"
    raw_styles = {
        192: {"color": "#0072B2", "marker": "o", "size": 5.2},
        240: {"color": "#D55E00", "marker": "s", "size": 5.6},
        288: {"color": "#CC79A7", "marker": "D", "size": 5.5},
    }
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 5.8))
    wide_axis, zoom_axis = axes

    model_select = np.abs(model["x"]) <= 0.025 + 1.0e-12
    model_plot_x = Q_DISTANCE_PER_D * model["x"][model_select]
    model_plot_y = model["frequency_cm-1"][model_select]
    K_model = model_K(model)
    wide_axis.plot(
        model_plot_x,
        model_plot_y,
        color=black,
        lw=2.7,
        zorder=3,
    )
    zoom_select = np.abs(model["x"]) <= 0.008 + 1.0e-12
    zoom_axis.plot(
        Q_DISTANCE_PER_D * model["x"][zoom_select],
        model["frequency_cm-1"][zoom_select] - K_model,
        color=black,
        lw=2.7,
        zorder=3,
    )

    for sign, branch in branches.items():
        x_value = sign * Q_DISTANCE_PER_D * branch["distance"]
        order = np.argsort(x_value)
        wide_axis.fill_between(
            x_value[order],
            branch["lower_cm-1"][order],
            branch["upper_cm-1"][order],
            color=green,
            alpha=0.13,
            linewidth=0.0,
            zorder=1,
        )
        near = branch["distance"] <= 0.008 + 1.0e-12
        near_order = np.argsort(x_value[near])
        zoom_axis.fill_between(
            x_value[near][near_order],
            branch["relative_lower_cm-1"][near][near_order],
            branch["relative_upper_cm-1"][near][near_order],
            color=green,
            alpha=0.13,
            linewidth=0.0,
            zorder=1,
        )
        plot_branch_segments(wide_axis, branch, sign, relative=False)
        plot_branch_segments(zoom_axis, branch, sign, relative=True)

    for kgrid, points in raw.items():
        style = raw_styles[kgrid]
        x = np.asarray(sorted(points), float)
        y = np.asarray([points[float(value)] for value in x], float)
        common = {
            "ls": "none",
            "marker": style["marker"],
            "ms": style["size"],
            "markerfacecolor": "white",
            "markeredgecolor": style["color"],
            "markeredgewidth": 1.25,
            "zorder": 6,
        }
        wide_axis.plot(Q_DISTANCE_PER_D * x, y, **common)
        K_raw = points[0.0]
        near = np.abs(x) <= 0.007 + 1.0e-12
        zoom_axis.plot(
            Q_DISTANCE_PER_D * x[near], y[near] - K_raw, **common
        )

    central_K = variants[0.0]["linear_all"]
    three_anchor_x = np.asarray([-0.003, 0.0, 0.003])
    three_anchor_y = np.asarray(
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
        "zorder": 8,
    }
    wide_axis.plot(
        Q_DISTANCE_PER_D * three_anchor_x, three_anchor_y, **star_style
    )
    zoom_axis.plot(
        Q_DISTANCE_PER_D * three_anchor_x,
        three_anchor_y - central_K,
        **star_style,
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
        "zorder": 7,
    }
    wide_axis.plot(Q_DISTANCE_PER_D * d007_x, d007_y, **triangle_style)
    zoom_axis.plot(
        Q_DISTANCE_PER_D * d007_x, d007_y - central_K, **triangle_style
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

    wide_axis.set_xlim(-Q_DISTANCE_PER_D * 0.025, Q_DISTANCE_PER_D * 0.025)
    wide_axis.set_ylim(1280.4, 1300.7)
    wide_axis.set_ylabel(r"A$_1'$ frequency (cm$^{-1}$)")
    wide_axis.set_title("(a) Wide K neighbourhood: absolute frequency")

    zoom_axis.set_xlim(-Q_DISTANCE_PER_D * 0.008, Q_DISTANCE_PER_D * 0.008)
    zoom_axis.set_ylim(-0.4, 9.1)
    zoom_axis.set_ylabel(r"K-referenced rise, $\omega(q)-\omega(K)$ (cm$^{-1}$)")
    zoom_axis.set_title("(b) Near-K zoom: each series referenced to its own K")

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
            label="fit/support envelope (not a confidence interval)",
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
        0.36,
        "Extrapolation support:\n"
        "|d| <= 0.003: three electronic k grids\n"
        "0.003 < |d| <= 0.007: two grids\n"
        "0.007 < |d| <= 0.023: k192 plus a\n"
        "decaying finite-grid correction (dotted).\n\n"
        "The outer green curve is a provisional guide,\n"
        "not a converged full DFPT dispersion.\n\n"
        "Layout reference: Piscanec et al.,\n"
        "Phys. Rev. Lett. 93, 185503 (2004), Fig. 2.",
        fontsize=8.9,
        color="#555555",
        ha="left",
        va="top",
        linespacing=1.32,
    )
    figure.suptitle(
        r"Graphene K-point $A_1'$ branch: wide zero-smearing extrapolation sketch",
        x=0.405,
        y=0.985,
        fontsize=13.2,
    )
    figure.subplots_adjust(
        left=0.075, right=0.79, bottom=0.16, top=0.86, wspace=0.28
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.publication_pdf.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "zero_wide_extrapolation_sketch.png"
    pdf_path = args.output_dir / "zero_wide_extrapolation_sketch.pdf"
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    figure.savefig(args.publication_pdf, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    curve_rows = []
    for sign, branch in branches.items():
        signed_d = sign * branch["distance"]
        model_interpolated = np.interp(signed_d, model["x"], model["frequency_cm-1"])
        for d_value, central, lower, upper, relative, rel_lower, rel_upper, model_value in zip(
            signed_d,
            branch["central_frequency_cm-1"],
            branch["lower_cm-1"],
            branch["upper_cm-1"],
            branch["central_relative_cm-1"],
            branch["relative_lower_cm-1"],
            branch["relative_upper_cm-1"],
            model_interpolated,
        ):
            abs_d = abs(float(d_value))
            support = (
                "three_grid"
                if abs_d <= 0.003 + 1.0e-12
                else "two_grid"
                if abs_d <= 0.007 + 1.0e-12
                else "k192_model_assisted"
            )
            curve_rows.append(
                {
                    "signed_distance_d": float(d_value),
                    "signed_q_distance_2pi_over_a": float(
                        Q_DISTANCE_PER_D * d_value
                    ),
                    "support_tier": support,
                    "DFPT_N_infinity_guide_cm-1": float(central),
                    "DFPT_guide_lower_cm-1": float(lower),
                    "DFPT_guide_upper_cm-1": float(upper),
                    "DFPT_K_referenced_guide_cm-1": float(relative),
                    "DFPT_K_referenced_lower_cm-1": float(rel_lower),
                    "DFPT_K_referenced_upper_cm-1": float(rel_upper),
                    "MLIP_full_EPC_LR_cm-1": float(model_value),
                    "MLIP_K_referenced_cm-1": float(model_value - K_model),
                }
            )
    curve_rows.sort(key=lambda row: row["signed_distance_d"])
    csv_path = args.output_dir / "zero_wide_extrapolation_sketch.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    summary = {
        "status": "wide_zero_smearing_extrapolation_sketch_generated",
        "scope": {
            "smearing_degauss_Ry": 0.0,
            "wide_signed_distance_d": [-0.025, 0.025],
            "extrapolated_guide_signed_distance_d": [-0.023, 0.023],
            "near_K_zoom_signed_distance_d": [-0.008, 0.008],
        },
        "data_inventory": {
            "raw_DFPT_point_count_by_kgrid": {
                str(kgrid): len(points) for kgrid, points in raw.items()
            },
            "three_grid_fixed_q": [-0.003, 0.0, 0.003],
            "two_grid_fixed_q": [-0.007, 0.007],
            "k192_only_outer_q": [
                -0.023,
                -0.019,
                -0.015,
                -0.011,
                0.011,
                0.015,
                0.019,
                0.023,
            ],
        },
        "central_extrapolation": {
            "local_fixed_q_model": "linear in 1/N using all available grids",
            "outer_finite_grid_tail": "A(d)=A(d007)*exp(-(d-0.007)/0.006)",
            "q_interpolation": "separate shape-preserving PCHIP branches; cusp retained at K",
            "K_frequency_cm-1": central_K,
            "KG_d003_depth_cm-1": variants[-0.003]["linear_all"]
            - central_K,
            "KM_d003_depth_cm-1": variants[0.003]["linear_all"]
            - central_K,
            "KG_d007_frequency_cm-1": KG["d007"]["intercept_cm-1"],
            "KM_d007_frequency_cm-1": KM["d007"]["intercept_cm-1"],
        },
        "envelope": {
            "fit_variants": list(variants[0.0]),
            "finite_grid_tail_variants": [
                "zero",
                "exp_0.004",
                "exp_0.006",
                "exp_0.008",
                "constant",
            ],
            "curves_per_direction": KG["ensemble_count"],
            "support_padding_cm-1_at_d": {
                "0.000": 0.0,
                "0.003": 0.0,
                "0.007": 0.12,
                "0.023": 0.65,
            },
            "interpretation": "model/support envelope, not a statistical confidence interval",
        },
        "limitations": [
            "the curve outside |d|=0.007 is a model-assisted sketch because only k192 direct DFPT is available",
            "the absolute K intercept remains sensitive to the chosen 1/N extrapolation form",
            "the extrapolated guide is not a converged full DFPT dispersion and must not be used as a training target without additional validation",
        ],
        "literature_layout_reference": {
            "citation": "S. Piscanec et al., Phys. Rev. Lett. 93, 185503 (2004), Fig. 2",
            "doi": "10.1103/PhysRevLett.93.185503",
        },
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "publication_pdf": str(args.publication_pdf.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    summary_path = args.output_dir / "zero_wide_extrapolation_sketch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
