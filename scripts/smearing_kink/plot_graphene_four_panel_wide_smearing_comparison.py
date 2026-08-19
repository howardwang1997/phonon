#!/usr/bin/env python3
"""Compose wide and near-K fixed-lattice smearing comparisons as four panels."""

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

import plot_graphene_extended_fixed_lattice_smearing_overlay as overlay  # noqa: E402
import plot_graphene_zero_wide_extrapolation_sketch as e37  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wide-sweep",
        type=Path,
        default=BASE
        / "E45_wider_fixed_lattice_smearing_sweep"
        / "extended_fixed_lattice_smearing_sweep.csv",
    )
    parser.add_argument(
        "--near-sweep",
        type=Path,
        default=BASE
        / "E42_extended_fixed_lattice_smearing_sweep"
        / "extended_fixed_lattice_smearing_sweep.csv",
    )
    parser.add_argument(
        "--zero-diagnostic",
        type=Path,
        default=BASE
        / "E34_zero_raw_DFPT_kgrid_diagnostic"
        / "zero_smearing_raw_DFPT_kgrid_diagnostic.csv",
    )
    parser.add_argument(
        "--finite-dfpt",
        type=Path,
        default=BASE
        / "E32_absolute_and_DFPT_comparison"
        / "MLIP_full_EPC_LR_vs_DFPT_points.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E46_four_panel_wider_smearing_comparison",
    )
    parser.add_argument(
        "--publication-pdf",
        type=Path,
        default=ROOT
        / "output/pdf/graphene_K_four_panel_wider_smearing_comparison.pdf",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_sweep(path: Path, expected_maximum_d: float) -> dict:
    rows = read_csv(path)
    smearings = np.asarray(
        sorted({float(row["smearing_degauss_Ry"]) for row in rows}), float
    )
    blocks = [
        [
            row
            for row in rows
            if np.isclose(
                float(row["smearing_degauss_Ry"]), smearing, atol=1.0e-14
            )
        ]
        for smearing in smearings
    ]
    if len(smearings) != 7 or not np.isclose(smearings[0], 0.0):
        raise RuntimeError("expected zero plus six finite-smearing series")
    lengths = {len(block) for block in blocks}
    if len(lengths) != 1:
        raise RuntimeError(f"inconsistent response block lengths: {lengths}")
    signature = [
        (
            row["direction"],
            round(float(row["distance"]), 12),
            round(float(row["signed_distance"]), 12),
            round(float(row["q1"]), 12),
            round(float(row["q2"]), 12),
        )
        for row in blocks[0]
    ]
    for block in blocks[1:]:
        current = [
            (
                row["direction"],
                round(float(row["distance"]), 12),
                round(float(row["signed_distance"]), 12),
                round(float(row["q1"]), 12),
                round(float(row["q2"]), 12),
            )
            for row in block
        ]
        if current != signature:
            raise RuntimeError("smearing series do not share the same q path")

    signed_d = np.asarray(
        [float(row["signed_distance"]) for row in blocks[0]], float
    )
    if not np.isclose(np.min(signed_d), -expected_maximum_d) or not np.isclose(
        np.max(signed_d), expected_maximum_d
    ):
        raise RuntimeError(
            f"expected d=+/-{expected_maximum_d}, got "
            f"[{np.min(signed_d)}, {np.max(signed_d)}]"
        )
    frequencies = np.asarray(
        [
            [float(row["MLIP_plus_full_EPC_cm-1"]) for row in block]
            for block in blocks
        ],
        float,
    )
    K_indices = np.flatnonzero(np.isclose(signed_d, 0.0, atol=1.0e-14))
    if len(K_indices) != 1:
        raise RuntimeError("expected one K point per smearing series")
    return {
        "path": path,
        "smearings": smearings,
        "blocks": blocks,
        "signed_d": signed_d,
        "x": e37.Q_DISTANCE_PER_D * signed_d,
        "frequencies": frequencies,
        "K_index": int(K_indices[0]),
        "maximum_d": expected_maximum_d,
    }


def overlap_audit(wide: dict, near: dict) -> dict:
    reference = {}
    for series_index, smearing in enumerate(near["smearings"]):
        for q_index, row in enumerate(near["blocks"][series_index]):
            key = (
                round(float(smearing), 12),
                row["direction"],
                round(float(row["distance"]), 12),
            )
            reference[key] = near["frequencies"][series_index, q_index]
    errors = []
    for series_index, smearing in enumerate(wide["smearings"]):
        for q_index, row in enumerate(wide["blocks"][series_index]):
            key = (
                round(float(smearing), 12),
                row["direction"],
                round(float(row["distance"]), 12),
            )
            if key in reference:
                errors.append(
                    wide["frequencies"][series_index, q_index] - reference[key]
                )
    error = np.asarray(errors, float)
    return {
        "number_of_shared_values": len(error),
        "maximum_absolute_error_cm-1": float(np.max(np.abs(error))),
        "RMSE_cm-1": float(np.sqrt(np.mean(error**2))),
    }


def monotonicity_audit(wide: dict, maximum_d: float) -> list[dict]:
    output = []
    for series_index, smearing in enumerate(wide["smearings"]):
        block = wide["blocks"][series_index]
        frequency = wide["frequencies"][series_index]
        for direction in ("KG", "KM"):
            pairs = sorted(
                (
                    float(row["distance"]),
                    float(frequency[index]),
                )
                for index, row in enumerate(block)
                if row["direction"] == direction
                and float(row["distance"]) <= maximum_d + 1.0e-12
            )
            distance = np.asarray([pair[0] for pair in pairs], float)
            values = np.asarray([pair[1] for pair in pairs], float)
            difference = np.diff(values)
            negative = difference < -1.0e-8
            output.append(
                {
                    "smearing_degauss_Ry": float(smearing),
                    "direction": direction,
                    "strictly_monotone_away_from_K": bool(not np.any(negative)),
                    "number_of_negative_steps": int(np.sum(negative)),
                    "largest_downward_step_cm-1": (
                        0.0 if not np.any(negative) else float(-np.min(difference))
                    ),
                    "first_negative_step_ends_at_d": (
                        None
                        if not np.any(negative)
                        else float(distance[np.flatnonzero(negative)[0] + 1])
                    ),
                }
            )
    return output


def fit_finite_dfpt_curves(
    finite_dfpt: dict[float, list[dict]],
) -> dict[float, dict]:
    """Fit the available finite-smearing K->Gamma DFPT points.

    Finite electronic smearing regularizes the K-point cusp.  We therefore fit
    the K-referenced rise with a zero-slope-at-K quartic,

        delta_omega(d) = a*d**2 + b*d**3 + c*d**4,

    over the measured interval only.  The cubic term allows the one-sided KG
    branch to be asymmetric without inventing a mirrored KM branch.
    """
    fits: dict[float, dict] = {}
    expected_distances = np.asarray([0.0, 0.008, 0.015, 0.025], float)
    for smearing, rows in finite_dfpt.items():
        points = sorted(rows, key=lambda row: row["distance"])
        distance = np.asarray([row["distance"] for row in points], float)
        frequency = np.asarray([row["frequency_cm-1"] for row in points], float)
        if not np.allclose(distance, expected_distances, atol=1.0e-12, rtol=0.0):
            raise RuntimeError(
                f"unexpected finite-DFPT distances for smearing {smearing}: "
                f"{distance.tolist()}"
            )
        if any(
            row["direction"] not in {"K", "KG"}
            or (row["distance"] > 0.0 and row["direction"] != "KG")
            for row in points
        ):
            raise RuntimeError("finite-smearing DFPT fit expects K and KG points only")

        rise = frequency - frequency[0]
        design = np.column_stack(
            (distance[1:] ** 2, distance[1:] ** 3, distance[1:] ** 4)
        )
        coefficients = np.linalg.solve(design, rise[1:])
        dense_distance = np.linspace(distance[0], distance[-1], 301)
        a, b, c = coefficients
        dense_rise = (
            a * dense_distance**2
            + b * dense_distance**3
            + c * dense_distance**4
        )
        derivative = (
            2.0 * a * dense_distance
            + 3.0 * b * dense_distance**2
            + 4.0 * c * dense_distance**3
        )
        fitted_at_data = (
            a * distance**2 + b * distance**3 + c * distance**4
        )
        maximum_residual = float(np.max(np.abs(fitted_at_data - rise)))
        minimum_derivative = float(np.min(derivative))
        if maximum_residual > 1.0e-8:
            raise RuntimeError("finite-smearing DFPT constrained fit missed input data")
        if minimum_derivative < -1.0e-8:
            raise RuntimeError("finite-smearing DFPT constrained fit is nonmonotone")
        fits[float(smearing)] = {
            "distance": dense_distance,
            "absolute_cm-1": frequency[0] + dense_rise,
            "relative_cm-1": dense_rise,
            "K_frequency_cm-1": float(frequency[0]),
            "coefficients": {
                "a_cm-1_per_d2": float(a),
                "b_cm-1_per_d3": float(b),
                "c_cm-1_per_d4": float(c),
            },
            "maximum_residual_cm-1": maximum_residual,
            "minimum_derivative_cm-1_per_d": minimum_derivative,
        }
    return fits


def add_finite_dfpt_fit_curves(
    axis: plt.Axes,
    smearings: np.ndarray,
    colors: list,
    fits: dict[float, dict],
    *,
    relative: bool,
) -> None:
    for color, smearing in zip(colors[1:], smearings[1:]):
        key = overlay.optional_smearing_key(fits, float(smearing))
        if key is None:
            continue
        fit = fits[key]
        axis.plot(
            -e37.Q_DISTANCE_PER_D * fit["distance"],
            fit["relative_cm-1"] if relative else fit["absolute_cm-1"],
            color=color,
            lw=1.75,
            ls=(0, (4.0, 2.0)),
            zorder=9,
        )


def plot_model_family(
    axis: plt.Axes,
    sweep: dict,
    colors: list,
    *,
    maximum_d: float,
    relative: bool,
) -> None:
    select = np.abs(sweep["signed_d"]) <= maximum_d + 1.0e-12
    order = np.argsort(sweep["x"][select])
    for series_index in reversed(range(len(sweep["smearings"]))):
        frequency = sweep["frequencies"][series_index]
        if relative:
            frequency = frequency - frequency[sweep["K_index"]]
        axis.plot(
            sweep["x"][select][order],
            frequency[select][order],
            color=colors[series_index],
            lw=2.65 if series_index == 0 else 1.85,
            zorder=4 if series_index == 0 else 3,
        )


def add_axis_common(axis: plt.Axes) -> None:
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
        fontsize=9.8,
        color="#444444",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 1.4,
        },
        zorder=12,
    )


def make_figure(
    png_path: Path,
    pdf_path: Path,
    publication_pdf: Path,
    wide: dict,
    near: dict,
    branches: dict[int, dict],
    raw: dict[int, dict[float, float]],
    finite_dfpt: dict[float, list[dict]],
    finite_dfpt_fits: dict[float, dict],
) -> None:
    colors = overlay.smearing_colors(len(wide["smearings"]))
    figure, axes = plt.subplots(2, 2, figsize=(16.8, 10.9))
    wide_absolute, wide_shape, near_absolute, near_shape = axes.flat
    plot_model_family(
        wide_absolute, wide, colors, maximum_d=0.10, relative=False
    )
    plot_model_family(wide_shape, wide, colors, maximum_d=0.10, relative=True)
    plot_model_family(
        near_absolute, near, colors, maximum_d=0.06, relative=False
    )
    plot_model_family(near_shape, near, colors, maximum_d=0.025, relative=True)

    overlay.add_support_shading(wide_absolute, 0.10)
    overlay.add_support_shading(wide_shape, 0.10)
    overlay.add_support_shading(near_absolute, 0.06)
    overlay.add_support_shading(near_shape, 0.025)

    previous_limit = e37.Q_DISTANCE_PER_D * 0.06
    for axis in (wide_absolute, wide_shape):
        axis.axvline(
            -previous_limit,
            color="#777777",
            lw=1.0,
            ls=":",
            alpha=0.75,
            zorder=2,
        )
        axis.axvline(
            previous_limit,
            color="#777777",
            lw=1.0,
            ls=":",
            alpha=0.75,
            zorder=2,
        )
        axis.text(
            0.5,
            0.88,
            "outer region: model diagnostic only",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.3,
            color="#666666",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=11,
        )

    overlay.add_zero_dfpt_guide(near_absolute, branches, relative=False)
    overlay.add_zero_dfpt_guide(near_shape, branches, relative=True)
    add_finite_dfpt_fit_curves(
        near_absolute,
        near["smearings"],
        colors,
        finite_dfpt_fits,
        relative=False,
    )
    add_finite_dfpt_fit_curves(
        near_shape,
        near["smearings"],
        colors,
        finite_dfpt_fits,
        relative=True,
    )
    raw_styles = {
        192: {"color": "#666666", "marker": "o", "size": 4.5},
        240: {"color": "#888888", "marker": "s", "size": 4.8},
        288: {"color": "#AAAAAA", "marker": "D", "size": 4.7},
    }
    overlay.add_raw_zero_dfpt(
        near_absolute, raw, raw_styles, relative=False
    )
    overlay.add_raw_zero_dfpt(near_shape, raw, raw_styles, relative=True)
    overlay.add_finite_dfpt_points(
        near_absolute,
        near["smearings"],
        colors,
        finite_dfpt,
        relative=False,
    )
    overlay.add_finite_dfpt_points(
        near_shape,
        near["smearings"],
        colors,
        finite_dfpt,
        relative=True,
    )

    for axis in axes.flat:
        add_axis_common(axis)

    displayed_wide_d = 0.10
    wide_q_limit = e37.Q_DISTANCE_PER_D * displayed_wide_d
    wide_absolute.set_xlim(-wide_q_limit, wide_q_limit)
    wide_shape.set_xlim(-wide_q_limit, wide_q_limit)
    displayed_wide = np.abs(wide["signed_d"]) <= displayed_wide_d + 1.0e-12
    wide_min = float(np.min(wide["frequencies"][:, displayed_wide]))
    wide_max = float(np.max(wide["frequencies"][:, displayed_wide]))
    wide_absolute.set_ylim(wide_min - 1.0, wide_max + 1.0)
    wide_centered = (
        wide["frequencies"]
        - wide["frequencies"][:, [wide["K_index"]]]
    )
    wide_shape_max = float(np.max(wide_centered[:, displayed_wide]))
    wide_shape_min = float(np.min(wide_centered[:, displayed_wide]))
    wide_shape.set_ylim(min(-0.5, wide_shape_min - 0.5), 1.055 * wide_shape_max)

    near_absolute.set_xlim(-0.04, 0.04)
    near_absolute_values = [near["frequencies"].ravel()]
    near_absolute_values.extend(
        np.asarray(list(points.values()), float) for points in raw.values()
    )
    near_absolute_values.extend(
        np.asarray(branch["lower_cm-1"], float) for branch in branches.values()
    )
    near_absolute_values.extend(
        np.asarray(branch["upper_cm-1"], float) for branch in branches.values()
    )
    near_absolute_values.extend(
        np.asarray(fit["absolute_cm-1"], float)
        for fit in finite_dfpt_fits.values()
    )
    near_absolute.set_ylim(
        min(float(np.min(values)) for values in near_absolute_values) - 0.8,
        max(float(np.max(values)) for values in near_absolute_values) + 0.8,
    )
    near_shape_limit = e37.Q_DISTANCE_PER_D * 0.025
    near_shape.set_xlim(-near_shape_limit, near_shape_limit)
    near_select = np.abs(near["signed_d"]) <= 0.025 + 1.0e-12
    near_centered = (
        near["frequencies"]
        - near["frequencies"][:, [near["K_index"]]]
    )
    model_shape_max = float(np.max(near_centered[:, near_select]))
    raw_shape_max = max(
        max(frequency - points[0.0] for frequency in points.values())
        for points in raw.values()
    )
    guide_shape_max = max(
        float(np.max(branch["relative_upper_cm-1"]))
        for branch in branches.values()
    )
    finite_fit_shape_max = max(
        float(np.max(fit["relative_cm-1"]))
        for fit in finite_dfpt_fits.values()
    )
    near_shape.set_ylim(
        -0.5,
        1.055
        * max(
            model_shape_max,
            raw_shape_max,
            guide_shape_max,
            finite_fit_shape_max,
        ),
    )

    wide_absolute.set_ylabel(r"A$_1'$ frequency (cm$^{-1}$)")
    wide_shape.set_ylabel(
        r"K-referenced rise, $\omega(q)-\omega(K)$ (cm$^{-1}$)"
    )
    near_absolute.set_ylabel(r"A$_1'$ frequency (cm$^{-1}$)")
    near_shape.set_ylabel(
        r"K-referenced rise, $\omega(q)-\omega(K)$ (cm$^{-1}$)"
    )
    wide_absolute.set_title("(a) Wider absolute view: |d| ≤ 0.10")
    wide_shape.set_title("(b) Wider K-referenced view: |d| ≤ 0.10")
    near_absolute.set_title("(c) Near-K absolute view: |d| ≤ 0.06")
    near_shape.set_title("(d) Near-K cusp shape: |d| ≤ 0.025")

    labels = ["0"] + [f"{value:.7f}" for value in wide["smearings"][1:]]
    line_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2.65 if index == 0 else 1.85,
            label=label,
        )
        for index, (color, label) in enumerate(zip(colors, labels))
    ]
    evidence_handles = [
        Line2D(
            [0],
            [0],
            color=overlay.GUIDE_COLOR,
            lw=2.3,
            ls="--",
            label=r"zero-smearing DFPT fitted $N\to\infty$ guide",
        ),
        Patch(
            facecolor=overlay.GUIDE_COLOR,
            alpha=0.11,
            edgecolor="none",
            label="zero-smearing guide envelope",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor="#777777",
            label="raw zero-smearing DFPT (k192/240/288)",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor="#222222",
            label="direct finite-smearing DFPT (K→Γ only)",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=1.75,
            ls=(0, (4.0, 2.0)),
            label="finite-smearing DFPT constrained fits (K→Γ only)",
        ),
        Patch(
            facecolor="#777777",
            alpha=0.07,
            edgecolor="none",
            label="outside direct zero-smearing DFPT support",
        ),
        Line2D(
            [0],
            [0],
            color="#777777",
            lw=1.0,
            ls=":",
            label="previous plotted limit: |d| = 0.06",
        ),
    ]
    figure.legend(
        handles=line_handles,
        loc="upper left",
        bbox_to_anchor=(0.805, 0.91),
        frameon=False,
        title="MLIP + full EPC LR\nsmearing/degauss (Ry)",
        borderaxespad=0.0,
    )
    figure.legend(
        handles=evidence_handles,
        loc="upper left",
        bbox_to_anchor=(0.805, 0.56),
        frameon=False,
        fontsize=8.4,
        borderaxespad=0.0,
    )
    figure.text(
        0.805,
        0.30,
        "Fixed lattice condition:\n"
        "same static MLIP D_SR(q) for all 7 curves;\n"
        "no lattice/TDEP-temperature mixing.\n\n"
        "Top row: direct full-EPC q evaluations to |d|=0.10.\n"
        "Beyond |d|=0.023 there are no direct zero-smearing\n"
        "DFPT anchors; the upper panels are model diagnostics.\n\n"
        "A full run to |d|=0.15 found outer numerical wiggles;\n"
        "that unstable segment is retained in data but not plotted.\n\n"
        "Bottom row: near-K view with available DFPT evidence.\n"
        "Finite-DFPT fits: Δω = ad² + bd³ + cd⁴, zero slope at K;\n"
        "drawn only over measured 0 ≤ d ≤ 0.025 (K→Γ).\n"
        "0.020 Ry is the frozen high-smearing reference.",
        fontsize=8.25,
        color="#555555",
        ha="left",
        va="top",
        linespacing=1.28,
    )
    figure.suptitle(
        r"Graphene K-point A$_1'$: wider and near-K electronic-smearing views",
        x=0.405,
        y=0.985,
        fontsize=14.0,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.785,
        bottom=0.08,
        top=0.92,
        wspace=0.27,
        hspace=0.34,
    )
    figure.savefig(png_path, dpi=230, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    figure.savefig(publication_pdf, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    wide = load_sweep(args.wide_sweep, expected_maximum_d=0.15)
    near = load_sweep(args.near_sweep, expected_maximum_d=0.06)
    if not np.allclose(wide["smearings"], near["smearings"], atol=0.0, rtol=0.0):
        raise RuntimeError("wide and near sweeps use different smearing ladders")
    _, raw = e37.load_data(args.zero_diagnostic)
    variants = e37.three_grid_variants(raw)
    branches = {
        -1: e37.build_branch(raw, variants, sign=-1),
        1: e37.build_branch(raw, variants, sign=1),
    }
    finite_dfpt = overlay.load_finite_dfpt(args.finite_dfpt)
    finite_dfpt_fits = fit_finite_dfpt_curves(finite_dfpt)
    overlap = overlap_audit(wide, near)
    displayed_monotonicity = monotonicity_audit(wide, maximum_d=0.10)
    full_monotonicity = monotonicity_audit(wide, maximum_d=0.15)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    args.publication_pdf.parent.mkdir(parents=True, exist_ok=True)
    png_path = output / "four_panel_wider_smearing_comparison.png"
    pdf_path = output / "four_panel_wider_smearing_comparison.pdf"
    make_figure(
        png_path,
        pdf_path,
        args.publication_pdf,
        wide,
        near,
        branches,
        raw,
        finite_dfpt,
        finite_dfpt_fits,
    )

    displayed_irregular = [
        row
        for row in displayed_monotonicity
        if not row["strictly_monotone_away_from_K"]
    ]
    full_irregular = [
        row
        for row in full_monotonicity
        if not row["strictly_monotone_away_from_K"]
    ]
    summary = {
        "status": "four_panel_wider_smearing_comparison_generated",
        "layout": {
            "a": "wide absolute, displayed d=+/-0.10",
            "b": "wide K-referenced, displayed d=+/-0.10",
            "c": "near-K absolute, d=+/-0.06",
            "d": "near-K K-referenced, d=+/-0.025",
        },
        "ranges": {
            "displayed_wide_signed_distance_d": [-0.10, 0.10],
            "displayed_wide_signed_q_distance_2pi_over_a": [
                -0.06666666666666667,
                0.06666666666666667,
            ],
            "computed_diagnostic_signed_distance_d": [-0.15, 0.15],
            "computed_diagnostic_signed_q_distance_2pi_over_a": [-0.10, 0.10],
            "previous_signed_distance_d": [-0.06, 0.06],
            "previous_signed_q_distance_2pi_over_a": [-0.04, 0.04],
        },
        "lattice_control": {
            "same_static_short_range_MLIP_Dq_for_all_smearings": True,
            "thermal_lattice_or_TDEP_background_used": False,
            "per_smearing_K_reanchoring": False,
        },
        "finite_smearing_DFPT_fits": {
            "form": "delta_omega(d) = a*d^2 + b*d^3 + c*d^4",
            "direction": "K to Gamma only",
            "fit_domain_d": [0.0, 0.025],
            "zero_slope_at_K": True,
            "mirrored_to_KM": False,
            "extrapolated_beyond_measured_domain": False,
            "series": {
                f"{smearing:.12g}": {
                    "K_frequency_cm-1": fit["K_frequency_cm-1"],
                    "coefficients": fit["coefficients"],
                    "maximum_residual_cm-1": fit[
                        "maximum_residual_cm-1"
                    ],
                    "minimum_derivative_cm-1_per_d": fit[
                        "minimum_derivative_cm-1_per_d"
                    ],
                }
                for smearing, fit in finite_dfpt_fits.items()
            },
        },
        "overlap_with_previous_range": overlap,
        "displayed_range_monotonicity_audit": displayed_monotonicity,
        "computed_diagnostic_range_monotonicity_audit": full_monotonicity,
        "displayed_number_of_nonmonotone_series_directions": len(
            displayed_irregular
        ),
        "computed_number_of_nonmonotone_series_directions": len(full_irregular),
        "largest_computed_range_downward_step_cm-1": (
            0.0
            if not full_irregular
            else max(
                row["largest_downward_step_cm-1"] for row in full_irregular
            )
        ),
        "interpretation": [
            (
                "the displayed upper panels are model diagnostics beyond the direct "
                "zero-smearing DFPT support at |d|=0.023"
            ),
            (
                "the computed 0.10<|d|<=0.15 segment contains small nonmonotonic "
                "numerical structures and is deliberately excluded from the figure"
            ),
            (
                "the bottom panels retain the existing near-K comparison and "
                "available static-lattice DFPT evidence"
            ),
        ],
        "sources": {
            "wide_sweep": str(args.wide_sweep.resolve()),
            "near_sweep": str(args.near_sweep.resolve()),
            "zero_smearing_DFPT": str(args.zero_diagnostic.resolve()),
            "finite_smearing_DFPT": str(args.finite_dfpt.resolve()),
        },
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "publication_pdf": str(args.publication_pdf.resolve()),
        },
    }
    summary_path = output / "four_panel_wider_smearing_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
