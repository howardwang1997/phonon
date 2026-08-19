#!/usr/bin/env python3
"""Overlay zero and finite electronic smearings on one fixed lattice background.

The four model curves share the zero-smearing curve's static short-range MLIP
dynamical matrix and one common long-range reference constant.  The finite
300/450/600 K lattice/TDEP backgrounds stored in the source calculation are
deliberately discarded.  Only the electronic full-EPC response changes with
smearing/degauss.
"""

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

from extract_graphene_epw_dynamical_matrices import RYDBERG_CM1  # noqa: E402
import plot_graphene_zero_wide_extrapolation_sketch as e37  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
GUIDE_COLOR = "#009E73"
LATTICE_CONDITION = "fixed_static_lattice_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-response",
        type=Path,
        default=BASE
        / "E42_extended_fixed_lattice_smearing_sweep"
        / "extended_fixed_lattice_smearing_sweep.csv",
    )
    parser.add_argument(
        "--full-response-summary",
        type=Path,
        default=BASE
        / "E42_extended_fixed_lattice_smearing_sweep"
        / "extended_fixed_lattice_smearing_sweep_summary.json",
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
        default=BASE / "E43_fixed_lattice_higher_smearing_overlay",
    )
    parser.add_argument(
        "--publication-pdf",
        type=Path,
        default=ROOT
        / "output/pdf/graphene_K_fixed_lattice_higher_smearing_overlay.pdf",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_response_series(
    path: Path,
) -> tuple[np.ndarray, list[list[dict[str, str]]]]:
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
    if len(smearings) < 4 or not np.isclose(smearings[0], 0.0):
        raise RuntimeError("expected zero plus at least three finite smearings")
    if any(len(block) != 241 for block in blocks):
        raise RuntimeError(
            f"expected 241 points in every response block, got "
            f"{[len(b) for b in blocks]}"
        )
    reference = [
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
        if current != reference:
            raise RuntimeError("smearing response blocks do not share identical q points")
    distance = np.asarray([float(row["signed_distance"]) for row in blocks[0]])
    if not np.isclose(distance.min(), -0.06) or not np.isclose(
        distance.max(), 0.06
    ):
        raise RuntimeError("expected extended response range d=+/-0.06")
    return smearings, blocks


def signed_frequency(squared_frequency: np.ndarray) -> np.ndarray:
    values = np.asarray(squared_frequency, float)
    return np.sign(values) * np.sqrt(np.abs(values))


def controlled_fixed_lattice_frequencies(
    blocks: list[list[dict[str, str]]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    np.ndarray,
]:
    zero_block = blocks[0]
    common_background = np.asarray(
        [float(row["short_range_background_cm-1"]) for row in zero_block],
        float,
    )
    responses_Ry2 = np.asarray(
        [
            [float(row["full_EPC_response_Ry2"]) for row in block]
            for block in blocks
        ],
        float,
    )
    responses_cm2 = responses_Ry2 * RYDBERG_CM1**2
    zero_stored_correction = np.asarray(
        [float(row["long_range_correction_cm-2"]) for row in zero_block],
        float,
    )
    constant_samples = zero_stored_correction - responses_cm2[0]
    common_constant = float(np.mean(constant_samples))
    constant_spread = float(np.ptp(constant_samples))
    if constant_spread > 1.0e-5:
        raise RuntimeError("zero-smearing curve has no unique EPC reference constant")
    correction_cm2 = common_constant + responses_cm2
    frequencies = signed_frequency(
        common_background[None, :] ** 2 + correction_cm2
    )
    stored_zero = np.asarray(
        [float(row["MLIP_plus_full_EPC_cm-1"]) for row in zero_block], float
    )
    zero_replay_error = float(np.max(np.abs(frequencies[0] - stored_zero)))
    return (
        common_background,
        responses_Ry2,
        correction_cm2,
        common_constant,
        constant_spread,
        zero_replay_error,
        frequencies,
    )


def point_index(block: list[dict[str, str]], direction: str, distance: float) -> int:
    if np.isclose(distance, 0.0, atol=1.0e-14):
        direction = "K"
    matches = [
        index
        for index, row in enumerate(block)
        if row["direction"] == direction
        and np.isclose(float(row["distance"]), distance, atol=1.0e-14)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {direction} d={distance} point, found {len(matches)}"
        )
    return matches[0]


def series_metrics(
    smearings: np.ndarray,
    zero_block: list[dict[str, str]],
    frequencies: np.ndarray,
) -> list[dict]:
    K_index = point_index(zero_block, "K", 0.0)
    output = []
    for series_index, smearing in enumerate(smearings):
        row = {
            "smearing_degauss_Ry": float(smearing),
            "K_frequency_cm-1": float(frequencies[series_index, K_index]),
        }
        for direction in ("KG", "KM"):
            for distance in (0.003, 0.023, 0.06):
                index = point_index(zero_block, direction, distance)
                row[f"{direction}_d{distance:.3f}_rise_cm-1"] = float(
                    frequencies[series_index, index]
                    - frequencies[series_index, K_index]
                )
        output.append(row)
    return output


def load_finite_dfpt(path: Path) -> dict[float, list[dict]]:
    output: dict[float, list[dict]] = {}
    for row in read_csv(path):
        smearing = float(row["smearing_degauss_Ry"])
        if smearing <= 0.0:
            continue
        output.setdefault(smearing, []).append(
            {
                "direction": row["direction"],
                "distance": float(row["distance"]),
                "frequency_cm-1": float(row["DFPT_frequency_cm-1"]),
                "kgrid": row["DFPT_kgrid"],
            }
        )
    if sorted(len(rows) for rows in output.values()) != [4, 4, 4]:
        raise RuntimeError("expected four static-lattice DFPT points per smearing")
    return output


def smearing_colors(number_of_series: int) -> list:
    if number_of_series < 2:
        raise ValueError("expected at least two smearing series")
    finite = plt.get_cmap("plasma")(
        np.linspace(0.10, 0.88, number_of_series - 1)
    )
    return ["#111111", *finite]


def optional_smearing_key(values: dict[float, object], target: float) -> float | None:
    matches = [value for value in values if np.isclose(value, target, atol=1e-12)]
    if len(matches) > 1:
        raise RuntimeError(f"multiple finite-DFPT blocks for smearing {target}")
    return None if not matches else matches[0]


def matched_smearing_key(values: dict[float, object], target: float) -> float:
    match = optional_smearing_key(values, target)
    if match is None:
        raise RuntimeError(f"no unique finite-DFPT block for smearing {target}")
    return match


def finite_dfpt_metrics(
    smearings: np.ndarray,
    zero_block: list[dict[str, str]],
    frequencies: np.ndarray,
    finite_dfpt: dict[float, list[dict]],
) -> dict:
    metrics = {}
    for series_index, smearing in enumerate(smearings[1:], start=1):
        key = optional_smearing_key(finite_dfpt, float(smearing))
        if key is None:
            continue
        points = sorted(finite_dfpt[key], key=lambda row: row["distance"])
        dfpt = np.asarray([row["frequency_cm-1"] for row in points], float)
        model = np.asarray(
            [
                frequencies[
                    series_index,
                    point_index(zero_block, row["direction"], row["distance"]),
                ]
                for row in points
            ],
            float,
        )
        error = model - dfpt
        centered_error = (model - model[0]) - (dfpt - dfpt[0])
        metrics[f"{smearing:.10f}"] = {
            "smearing_degauss_Ry": float(smearing),
            "DFPT_kgrid": points[0]["kgrid"],
            "number_of_KG_points": len(points),
            "absolute_RMSE_cm-1": float(np.sqrt(np.mean(error**2))),
            "maximum_absolute_error_cm-1": float(np.max(np.abs(error))),
            "K_error_cm-1": float(error[0]),
            "K_referenced_shape_RMSE_cm-1": float(
                np.sqrt(np.mean(centered_error**2))
            ),
        }
    return metrics


def add_support_shading(axis: plt.Axes, maximum_d: float) -> None:
    outer = e37.Q_DISTANCE_PER_D * 0.023
    limit = e37.Q_DISTANCE_PER_D * maximum_d
    axis.axvspan(-limit, -outer, color="#777777", alpha=0.07, lw=0, zorder=0)
    axis.axvspan(outer, limit, color="#777777", alpha=0.07, lw=0, zorder=0)


def add_zero_dfpt_guide(
    axis: plt.Axes,
    branches: dict[int, dict],
    *,
    relative: bool,
) -> None:
    for sign, branch in branches.items():
        x = sign * e37.Q_DISTANCE_PER_D * branch["distance"]
        order = np.argsort(x)
        if relative:
            lower = branch["relative_lower_cm-1"]
            upper = branch["relative_upper_cm-1"]
        else:
            lower = branch["lower_cm-1"]
            upper = branch["upper_cm-1"]
        axis.fill_between(
            x[order],
            lower[order],
            upper[order],
            color=GUIDE_COLOR,
            alpha=0.11,
            lw=0.0,
            zorder=1,
        )
        e37.plot_branch_segments(axis, branch, sign, relative=relative)


def add_raw_zero_dfpt(
    axis: plt.Axes,
    raw: dict[int, dict[float, float]],
    styles: dict[int, dict],
    *,
    relative: bool,
) -> None:
    for kgrid, points in raw.items():
        signed_d = np.asarray(sorted(points), float)
        frequency = np.asarray([points[float(value)] for value in signed_d])
        if relative:
            frequency = frequency - points[0.0]
        style = styles[kgrid]
        axis.plot(
            e37.Q_DISTANCE_PER_D * signed_d,
            frequency,
            ls="none",
            marker=style["marker"],
            ms=style["size"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.1,
            zorder=8,
        )


def add_finite_dfpt_points(
    axis: plt.Axes,
    smearings: np.ndarray,
    colors: list,
    finite_dfpt: dict[float, list[dict]],
    *,
    relative: bool,
    inner_only: bool = False,
) -> None:
    for color, smearing in zip(colors[1:], smearings[1:]):
        key = optional_smearing_key(finite_dfpt, float(smearing))
        if key is None:
            continue
        points = sorted(finite_dfpt[key], key=lambda row: row["distance"])
        if inner_only:
            points = [row for row in points if row["distance"] <= 0.008 + 1e-12]
        K_frequency = next(
            row["frequency_cm-1"]
            for row in points
            if np.isclose(row["distance"], 0.0)
        )
        x = np.asarray(
            [
                (-1.0 if row["direction"] == "KG" else 1.0)
                * e37.Q_DISTANCE_PER_D
                * row["distance"]
                for row in points
            ],
            float,
        )
        y = np.asarray([row["frequency_cm-1"] for row in points], float)
        if relative:
            y = y - K_frequency
        axis.plot(
            x,
            y,
            ls="none",
            marker="o",
            ms=6.0 if not inner_only else 4.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.35,
            zorder=10,
        )


def make_figure(
    png_path: Path,
    pdf_path: Path,
    publication_pdf: Path,
    smearings: np.ndarray,
    zero_block: list[dict[str, str]],
    frequencies: np.ndarray,
    branches: dict[int, dict],
    raw: dict[int, dict[float, float]],
    finite_dfpt: dict[float, list[dict]],
) -> None:
    signed_d = np.asarray(
        [float(row["signed_distance"]) for row in zero_block], float
    )
    x = e37.Q_DISTANCE_PER_D * signed_d
    order = np.argsort(x)
    K_index = point_index(zero_block, "K", 0.0)
    raw_styles = {
        192: {"color": "#666666", "marker": "o", "size": 4.8},
        240: {"color": "#888888", "marker": "s", "size": 5.0},
        288: {"color": "#AAAAAA", "marker": "D", "size": 4.9},
    }

    colors = smearing_colors(len(smearings))
    figure, (absolute_axis, shape_axis) = plt.subplots(
        1, 2, figsize=(16.6, 6.75)
    )
    labels = ["0"] + [f"{value:.7f}" for value in smearings[1:]]
    line_handles = []
    for series_index, (smearing, color, label) in enumerate(
        zip(smearings, colors, labels)
    ):
        linewidth = 2.8 if series_index == 0 else 2.15
        line_handles.append(
            Line2D([0], [0], color=color, lw=linewidth, label=label)
        )
        absolute_axis.plot(
            x[order],
            frequencies[series_index, order],
            color=color,
            lw=linewidth,
            zorder=4 if series_index == 0 else 3,
        )
        supported = np.abs(signed_d) <= 0.025 + 1.0e-12
        supported_order = np.argsort(x[supported])
        centered = frequencies[series_index] - frequencies[series_index, K_index]
        shape_axis.plot(
            x[supported][supported_order],
            centered[supported][supported_order],
            color=color,
            lw=linewidth,
            zorder=4 if series_index == 0 else 3,
        )

    add_support_shading(absolute_axis, 0.06)
    add_support_shading(shape_axis, 0.025)
    add_zero_dfpt_guide(absolute_axis, branches, relative=False)
    add_zero_dfpt_guide(shape_axis, branches, relative=True)
    add_raw_zero_dfpt(absolute_axis, raw, raw_styles, relative=False)
    add_raw_zero_dfpt(shape_axis, raw, raw_styles, relative=True)
    add_finite_dfpt_points(
        absolute_axis, smearings, colors, finite_dfpt, relative=False
    )
    add_finite_dfpt_points(
        shape_axis, smearings, colors, finite_dfpt, relative=True
    )

    inset = shape_axis.inset_axes([0.27, 0.49, 0.46, 0.40])
    inner = np.abs(signed_d) <= 0.008 + 1.0e-12
    inner_order = np.argsort(x[inner])
    for series_index, color in enumerate(colors):
        centered = frequencies[series_index] - frequencies[series_index, K_index]
        inset.plot(
            x[inner][inner_order],
            centered[inner][inner_order],
            color=color,
            lw=2.0 if series_index == 0 else 1.55,
        )
    add_finite_dfpt_points(
        inset,
        smearings,
        colors,
        finite_dfpt,
        relative=True,
        inner_only=True,
    )
    inset.set_xlim(
        -e37.Q_DISTANCE_PER_D * 0.008,
        e37.Q_DISTANCE_PER_D * 0.008,
    )
    inner_max = float(
        np.max(frequencies[:, inner] - frequencies[:, [K_index]])
    )
    inset.set_ylim(-0.08, 1.08 * inner_max)
    inset.set_title("inner K window", fontsize=8.3, pad=2.0)
    inset.grid(alpha=0.17)
    inset.tick_params(labelsize=7.0)
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    inset.set_zorder(20)

    for axis in (absolute_axis, shape_axis):
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
            fontsize=10.0,
            color="#444444",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
                "pad": 1.5,
            },
            zorder=12,
        )

    absolute_axis.set_xlim(-0.04, 0.04)
    all_absolute = [frequencies.ravel()]
    all_absolute.extend(
        np.asarray([row["frequency_cm-1"] for row in rows], float)
        for rows in finite_dfpt.values()
    )
    all_absolute.extend(
        np.asarray(list(points.values()), float) for points in raw.values()
    )
    all_absolute.extend(
        np.asarray(branch["lower_cm-1"], float) for branch in branches.values()
    )
    all_absolute.extend(
        np.asarray(branch["upper_cm-1"], float) for branch in branches.values()
    )
    absolute_min = float(min(np.min(values) for values in all_absolute))
    absolute_max = float(max(np.max(values) for values in all_absolute))
    absolute_axis.set_ylim(absolute_min - 0.8, absolute_max + 0.8)
    absolute_axis.set_ylabel(r"A$_1'$ frequency (cm$^{-1}$)")
    absolute_axis.set_title("(a) Absolute frequency: extended K region")
    absolute_axis.text(
        -0.028,
        absolute_max + 0.25,
        "no direct zero-smearing\nDFPT anchors",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#777777",
    )
    absolute_axis.text(
        0.028,
        absolute_max + 0.25,
        "no direct zero-smearing\nDFPT anchors",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#777777",
    )

    shape_axis.set_xlim(
        -e37.Q_DISTANCE_PER_D * 0.025,
        e37.Q_DISTANCE_PER_D * 0.025,
    )
    supported = np.abs(signed_d) <= 0.025 + 1.0e-12
    model_shape_max = float(
        np.max(frequencies[:, supported] - frequencies[:, [K_index]])
    )
    raw_shape_max = max(
        max(frequency - points[0.0] for frequency in points.values())
        for points in raw.values()
    )
    guide_shape_max = max(
        float(np.max(branch["relative_upper_cm-1"]))
        for branch in branches.values()
    )
    shape_max = max(model_shape_max, raw_shape_max, guide_shape_max)
    shape_axis.set_ylim(-0.5, 1.055 * shape_max)
    shape_axis.set_ylabel(
        r"K-referenced rise, $\omega(q)-\omega(K)$ (cm$^{-1}$)"
    )
    shape_axis.set_title("(b) Cusp shape: each series referenced to its own K")

    support_handles = [
        Line2D(
            [0],
            [0],
            color=GUIDE_COLOR,
            lw=2.3,
            ls="--",
            label=r"zero-smearing DFPT $N\to\infty$ guide",
        ),
        Patch(
            facecolor=GUIDE_COLOR,
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
        Patch(
            facecolor="#777777",
            alpha=0.07,
            edgecolor="none",
            label="outside direct zero-smearing DFPT support",
        ),
    ]
    figure.legend(
        handles=line_handles,
        loc="upper left",
        bbox_to_anchor=(0.805, 0.90),
        frameon=False,
        title="MLIP + full EPC LR\nsmearing/degauss (Ry)",
        borderaxespad=0.0,
    )
    figure.legend(
        handles=support_handles,
        loc="upper left",
        bbox_to_anchor=(0.805, 0.47),
        frameon=False,
        borderaxespad=0.0,
        fontsize=8.5,
    )
    figure.text(
        0.805,
        0.225,
        "Lattice condition held fixed:\n"
        f"same static MLIP D_SR(q) for all {len(smearings)} curves;\n"
        "no 300/450/600 K lattice/TDEP backgrounds.\n\n"
        "One common absolute EPC reference constant;\n"
        "no per-smearing K re-anchoring.\n"
        "0.020 Ry is the frozen response reference.\n"
        "Finite-smearing markers: direct static-lattice DFPT.",
        fontsize=8.55,
        color="#555555",
        ha="left",
        va="top",
        linespacing=1.28,
    )
    figure.suptitle(
        r"Graphene K-point A$_1'$: electronic smearing at one fixed lattice condition",
        x=0.405,
        y=0.985,
        fontsize=13.2,
    )
    figure.subplots_adjust(
        left=0.073, right=0.785, bottom=0.16, top=0.86, wspace=0.28
    )
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    figure.savefig(publication_pdf, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def write_controlled_csv(
    path: Path,
    smearings: np.ndarray,
    blocks: list[list[dict[str, str]]],
    background: np.ndarray,
    responses_Ry2: np.ndarray,
    correction_cm2: np.ndarray,
    common_constant: float,
    frequencies: np.ndarray,
) -> None:
    fields = [
        "smearing_degauss_Ry",
        "lattice_condition",
        "direction",
        "distance",
        "signed_distance",
        "q1",
        "q2",
        "common_static_MLIP_background_cm-1",
        "full_EPC_response_Ry2",
        "common_reference_constant_cm-2",
        "controlled_long_range_correction_cm-2",
        "controlled_MLIP_plus_full_EPC_cm-1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for series_index, smearing in enumerate(smearings):
            for q_index, row in enumerate(blocks[series_index]):
                writer.writerow(
                    {
                        "smearing_degauss_Ry": float(smearing),
                        "lattice_condition": LATTICE_CONDITION,
                        "direction": row["direction"],
                        "distance": row["distance"],
                        "signed_distance": row["signed_distance"],
                        "q1": row["q1"],
                        "q2": row["q2"],
                        "common_static_MLIP_background_cm-1": background[q_index],
                        "full_EPC_response_Ry2": responses_Ry2[
                            series_index, q_index
                        ],
                        "common_reference_constant_cm-2": common_constant,
                        "controlled_long_range_correction_cm-2": correction_cm2[
                            series_index, q_index
                        ],
                        "controlled_MLIP_plus_full_EPC_cm-1": frequencies[
                            series_index, q_index
                        ],
                    }
                )


def main() -> int:
    args = parse_args()
    smearings, blocks = load_response_series(args.full_response)
    (
        common_background,
        responses_Ry2,
        correction_cm2,
        common_constant,
        constant_spread,
        zero_replay_error,
        frequencies,
    ) = controlled_fixed_lattice_frequencies(blocks)
    _, raw = e37.load_data(args.zero_diagnostic)
    variants = e37.three_grid_variants(raw)
    branches = {
        -1: e37.build_branch(raw, variants, sign=-1),
        1: e37.build_branch(raw, variants, sign=1),
    }
    finite_dfpt = load_finite_dfpt(args.finite_dfpt)
    metrics = series_metrics(smearings, blocks[0], frequencies)
    dfpt_metrics = finite_dfpt_metrics(
        smearings, blocks[0], frequencies, finite_dfpt
    )
    source_summary = json.loads(
        args.full_response_summary.read_text(encoding="utf-8")
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    args.publication_pdf.parent.mkdir(parents=True, exist_ok=True)
    png_path = output / "fixed_lattice_higher_smearing_overlay.png"
    pdf_path = output / "fixed_lattice_higher_smearing_overlay.pdf"
    csv_path = output / "fixed_lattice_higher_smearing_overlay.csv"
    write_controlled_csv(
        csv_path,
        smearings,
        blocks,
        common_background,
        responses_Ry2,
        correction_cm2,
        common_constant,
        frequencies,
    )
    make_figure(
        png_path,
        pdf_path,
        args.publication_pdf,
        smearings,
        blocks[0],
        frequencies,
        branches,
        raw,
        finite_dfpt,
    )

    K_frequencies = np.asarray(
        [row["K_frequency_cm-1"] for row in metrics], float
    )
    mean_d003 = np.asarray(
        [
            0.5
            * (
                row["KG_d0.003_rise_cm-1"]
                + row["KM_d0.003_rise_cm-1"]
            )
            for row in metrics
        ],
        float,
    )
    summary = {
        "status": "fixed_lattice_higher_smearing_overlay_generated",
        "scope": (
            f"graphene K-neighbourhood A1-prime; zero plus {len(smearings) - 1} "
            "electronic smearings on one common static-lattice MLIP background"
        ),
        "lattice_control": {
            "condition_id": LATTICE_CONDITION,
            "same_short_range_MLIP_Dq_for_every_smearing": True,
            "thermal_lattice_or_TDEP_background_used": False,
            "source_paired_300_450_600_K_backgrounds_used": False,
            "note": (
                "This isolates electronic smearing on a static lattice "
                "reference; it is not a finite-lattice-temperature sweep."
            ),
        },
        "construction": {
            "formula": (
                "omega(q,s)^2 = omega_SR_common(q)^2 + C_common "
                "+ Pi_full_EPC(q,s)"
            ),
            "common_reference_constant_cm-2": common_constant,
            "common_constant_spread_cm-2": constant_spread,
            "per_smearing_K_reanchoring": False,
            "number_of_qpoints": len(blocks[0]),
            "signed_distance_d": [-0.06, 0.06],
            "signed_q_distance_2pi_over_a": [-0.04, 0.04],
            "new_DFT_used": False,
            "finite_response_wall_time_seconds": source_summary["integration"][
                "wall_time_seconds"
            ],
        },
        "series_metrics": metrics,
        "finite_static_lattice_DFPT_comparison": dfpt_metrics,
        "checks": {
            "zero_curve_replay_max_abs_cm-1": zero_replay_error,
            "all_frequencies_finite": bool(np.isfinite(frequencies).all()),
            "K_frequency_increases_monotonically_with_smearing": bool(
                np.all(np.diff(K_frequencies) > 0.0)
            ),
            "mean_d003_cusp_depth_decreases_monotonically_with_smearing": bool(
                np.all(np.diff(mean_d003) < 0.0)
            ),
        },
        "DFPT_support": {
            "zero_smearing_guide_abs_d_le": 0.023,
            "finite_smearing_direct_DFPT": "K-to-Gamma only at d=0, 0.008, 0.015, 0.025",
            "finite_smearing_K_to_M_DFPT_available": False,
        },
        "limitations": [
            (
                "the gray outer region is a full-EPC model prediction without "
                "direct zero-smearing DFPT anchors"
            ),
            (
                "finite-smearing K-to-M curves have no independent direct "
                "DFPT validation"
            ),
            (
                "the finite-smearing direct static-lattice DFPT comparison "
                "shows remaining absolute and K-referenced model errors"
            ),
            (
                "0.005, 0.010, and 0.020 Ry have no new direct DFPT points; "
                "their curves are full-EPC model predictions"
            ),
        ],
        "sources": {
            "extended_full_EPC_response": str(args.full_response.resolve()),
            "zero_smearing_DFPT_diagnostic": str(
                args.zero_diagnostic.resolve()
            ),
            "finite_static_lattice_DFPT": str(args.finite_dfpt.resolve()),
        },
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "publication_pdf": str(args.publication_pdf.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    summary_path = output / "fixed_lattice_higher_smearing_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
