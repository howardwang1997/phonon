#!/usr/bin/env python3
"""Plot zero-smearing MLIP+EPC against every available raw DFPT q point."""

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


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
NOSMEAR = ROOT / "results/graphene_k_cusp_nosmear"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-sweep",
        type=Path,
        default=BASE
        / "E31_fixed_background_smearing_sweep"
        / "fixed_background_smearing_sweep_Aprime.csv",
    )
    parser.add_argument(
        "--zero-summary",
        type=Path,
        default=BASE
        / "E14_zero_quadrature_convergence"
        / "zero_quadrature_convergence_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E34_zero_raw_DFPT_kgrid_diagnostic",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def signed_distance(direction: str, distance: float) -> float:
    if direction == "KG":
        return -distance
    if direction == "KM":
        return distance
    if direction == "K" and abs(distance) < 1.0e-14:
        return 0.0
    raise ValueError(f"unrecognised direction/distance: {direction}, {distance}")


def consolidate_points(
    rows: list[dict[str, str]],
    *,
    distance_key: str,
    frequency_key: str,
) -> dict[str, np.ndarray]:
    values: dict[float, list[float]] = {}
    for row in rows:
        if "integration" in row and row["integration"]:
            if row["integration"] != "tetrahedra_opt_no_degauss":
                raise RuntimeError(
                    f"unexpected electronic integration: {row['integration']}"
                )
        distance = float(row[distance_key])
        x_value = signed_distance(row["direction"], distance)
        key = round(x_value, 12)
        values.setdefault(key, []).append(float(row[frequency_key]))

    points = []
    for x_value, frequencies in values.items():
        if np.ptp(frequencies) > 1.0e-5:
            raise RuntimeError(
                f"duplicate DFPT values disagree at d={x_value}: {frequencies}"
            )
        points.append((x_value, float(np.mean(frequencies))))
    points.sort()

    x = np.asarray([point[0] for point in points], float)
    frequency = np.asarray([point[1] for point in points], float)
    zero_indices = np.flatnonzero(np.isclose(x, 0.0, atol=1.0e-14))
    if len(zero_indices) != 1:
        raise RuntimeError("each k grid must contain exactly one K-point anchor")
    K_frequency = float(frequency[zero_indices[0]])
    return {
        "x": x,
        "frequency_cm-1": frequency,
        "K_frequency_cm-1": np.asarray(K_frequency),
        "rise_from_K_cm-1": frequency - K_frequency,
    }


def load_raw_dfpt() -> dict[int, dict[str, np.ndarray]]:
    k192_rows = read_csv(
        NOSMEAR / "S3_development_freeze/development_points.csv"
    ) + read_csv(NOSMEAR / "S4_holdout/holdout_predictions.csv")

    k240_paths = [
        NOSMEAR
        / "raw/convergence/k240_qe75_npk120k"
        / "convergence_A_k240_qe75_npk120k/dfpt_points.csv",
        NOSMEAR
        / "raw/convergence/k240_qe75_npk120k"
        / "convergence_B_k240_qe75_npk120k/dfpt_points.csv",
        NOSMEAR
        / "raw/diagnostic_d003/k240_qe75_npk120k"
        / "diagnostic_A_k240_qe75_npk120k/dfpt_points.csv",
        NOSMEAR
        / "raw/diagnostic_d003/k240_qe75_npk120k"
        / "diagnostic_B_k240_qe75_npk120k/dfpt_points.csv",
    ]
    k288_paths = [
        NOSMEAR
        / "raw/diagnostic_d003/k288_qe75_npk120k"
        / "diagnostic_A_k288_qe75_npk120k/dfpt_points.csv",
        NOSMEAR
        / "raw/diagnostic_d003/k288_qe75_npk120k"
        / "diagnostic_B_k288_qe75_npk120k/dfpt_points.csv",
    ]
    k240_rows = [row for path in k240_paths for row in read_csv(path)]
    k288_rows = [row for path in k288_paths for row in read_csv(path)]

    result = {
        192: consolidate_points(
            k192_rows,
            distance_key="delta",
            frequency_key="frequency_cm-1",
        ),
        240: consolidate_points(
            k240_rows,
            distance_key="delta_equal_distance",
            frequency_key="f6_cm-1",
        ),
        288: consolidate_points(
            k288_rows,
            distance_key="delta_equal_distance",
            frequency_key="f6_cm-1",
        ),
    }
    expected_counts = {192: 13, 240: 5, 288: 3}
    actual_counts = {kgrid: len(data["x"]) for kgrid, data in result.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"unexpected raw DFPT point counts: {actual_counts}"
        )
    return result


def load_zero_model(path: Path) -> dict[str, np.ndarray]:
    rows = [
        row
        for row in read_csv(path)
        if np.isclose(float(row["smearing_degauss_Ry"]), 0.0, atol=1.0e-14)
    ]
    if len(rows) != 121:
        raise RuntimeError(f"expected 121 zero-smearing model points, got {len(rows)}")
    x = np.asarray([float(row["signed_distance"]) for row in rows], float)
    frequency = np.asarray(
        [float(row["controlled_MLIP_plus_full_EPC_cm-1"]) for row in rows],
        float,
    )
    order = np.argsort(x)
    x = x[order]
    frequency = frequency[order]
    K_frequency = float(frequency[np.argmin(np.abs(x))])
    return {
        "x": x,
        "frequency_cm-1": frequency,
        "K_frequency_cm-1": np.asarray(K_frequency),
        "rise_from_K_cm-1": frequency - K_frequency,
    }


def value_at(data: dict[str, np.ndarray], x_value: float, key: str) -> float:
    indices = np.flatnonzero(np.isclose(data["x"], x_value, atol=1.0e-12))
    if len(indices) != 1:
        raise RuntimeError(f"expected one value at signed distance {x_value}")
    return float(data[key][indices[0]])


def main() -> int:
    args = parse_args()
    model = load_zero_model(args.controlled_sweep)
    raw = load_raw_dfpt()
    zero_summary = json.loads(args.zero_summary.read_text(encoding="utf-8"))

    extrapolated_depth = {
        "KG": float(
            zero_summary["directions"]["KG"]
            ["DFPT_three_point_1_over_k_limit_cm-1"]
        ),
        "KM": float(
            zero_summary["directions"]["KM"]
            ["DFPT_three_point_1_over_k_limit_cm-1"]
        ),
    }
    extrapolated_x = np.asarray([-0.003, 0.003])
    extrapolated_rise = np.asarray(
        [extrapolated_depth["KG"], extrapolated_depth["KM"]]
    )
    k288_anchor = float(raw[288]["K_frequency_cm-1"])
    if not np.isclose(
        k288_anchor, float(model["K_frequency_cm-1"]), atol=1.0e-6
    ):
        raise RuntimeError("the model and k288 K anchors unexpectedly differ")
    extrapolated_absolute = k288_anchor + extrapolated_rise

    styles = {
        192: {"color": "#0072B2", "marker": "o", "size": 5.5},
        240: {"color": "#D55E00", "marker": "s", "size": 5.8},
        288: {"color": "#CC79A7", "marker": "D", "size": 5.8},
    }
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
    absolute_axis, depth_axis = axes

    absolute_axis.plot(
        model["x"],
        model["frequency_cm-1"],
        color="#111111",
        lw=2.6,
        zorder=2,
    )
    depth_axis.plot(
        model["x"],
        model["rise_from_K_cm-1"],
        color="#111111",
        lw=2.6,
        zorder=2,
    )
    for kgrid, data in raw.items():
        style = styles[kgrid]
        common = {
            "ls": "none",
            "marker": style["marker"],
            "ms": style["size"],
            "markerfacecolor": "white",
            "markeredgecolor": style["color"],
            "markeredgewidth": 1.35,
            "zorder": 4,
        }
        absolute_axis.plot(data["x"], data["frequency_cm-1"], **common)
        depth_axis.plot(data["x"], data["rise_from_K_cm-1"], **common)

    star_style = {
        "ls": "none",
        "marker": "*",
        "ms": 11.5,
        "markerfacecolor": "#009E73",
        "markeredgecolor": "#005E46",
        "markeredgewidth": 0.8,
        "zorder": 5,
    }
    absolute_axis.plot(extrapolated_x, extrapolated_absolute, **star_style)
    depth_axis.plot(extrapolated_x, extrapolated_rise, **star_style)

    for axis in axes:
        axis.axvline(0.0, color="#AAAAAA", lw=0.9, zorder=0)
        axis.grid(alpha=0.18)
        axis.set_xlabel("signed distance from K (K→Γ < 0; K→M > 0)")

    absolute_axis.set_xlim(-0.03, 0.03)
    absolute_axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    absolute_axis.set_title("(a) Absolute frequency and all raw DFPT points")
    absolute_axis.margins(y=0.08)

    depth_axis.set_xlim(-0.008, 0.008)
    depth_axis.set_ylim(-0.45, 9.2)
    depth_axis.set_ylabel(r"K-referenced rise, $\omega(q)-\omega(K)$ (cm$^{-1}$)")
    depth_axis.set_title("(b) Near-K cusp depth and k-grid convergence")

    legend_handles = [
        Line2D(
            [0], [0], color="#111111", lw=2.6, label="MLIP + full EPC LR"
        )
    ]
    for kgrid in (192, 240, 288):
        style = styles[kgrid]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="none",
                marker=style["marker"],
                markersize=style["size"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.35,
                label=f"raw DFPT, k{kgrid} ({len(raw[kgrid]['x'])} points)",
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markersize=11.5,
            markerfacecolor="#009E73",
            markeredgecolor="#005E46",
            label=r"DFPT $1/N_k$ depth limit at $|d|=0.003$",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.815, 0.84),
        frameon=False,
        title="zero smearing/degauss (Ry)",
        borderaxespad=0.0,
    )
    figure.text(
        0.815,
        0.45,
        "DFPT integration: tetrahedra_opt; no degauss.\n"
        "Raw markers are not joined or fitted.\n"
        "The green stars extrapolate cusp depth only;\n"
        "panel (a) anchors them to the k288 K frequency.\n\n"
        "k240→k288 change in d=0.003 depth:\n"
        "K→Γ 12.00%; K→M 11.97%.",
        fontsize=9.2,
        color="#555555",
        ha="left",
        va="top",
        linespacing=1.35,
    )
    figure.suptitle(
        "Graphene K-point A′ at zero smearing: raw DFPT convergence and MLIP + long-range correction",
        x=0.405,
        y=0.98,
        fontsize=13.0,
    )
    figure.subplots_adjust(
        left=0.075, right=0.79, bottom=0.15, top=0.88, wspace=0.28
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "zero_smearing_raw_DFPT_kgrid_diagnostic.png"
    pdf_path = args.output_dir / "zero_smearing_raw_DFPT_kgrid_diagnostic.pdf"
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    rows = []
    for x_value, frequency, rise in zip(
        model["x"], model["frequency_cm-1"], model["rise_from_K_cm-1"]
    ):
        rows.append(
            {
                "series": "MLIP_plus_full_EPC_LR",
                "kgrid": "",
                "signed_distance": float(x_value),
                "frequency_cm-1": float(frequency),
                "K_reference_cm-1": float(model["K_frequency_cm-1"]),
                "rise_from_K_cm-1": float(rise),
                "status": "model_curve",
            }
        )
    for kgrid, data in raw.items():
        for x_value, frequency, rise in zip(
            data["x"], data["frequency_cm-1"], data["rise_from_K_cm-1"]
        ):
            rows.append(
                {
                    "series": "raw_DFPT",
                    "kgrid": kgrid,
                    "signed_distance": float(x_value),
                    "frequency_cm-1": float(frequency),
                    "K_reference_cm-1": float(data["K_frequency_cm-1"]),
                    "rise_from_K_cm-1": float(rise),
                    "status": "raw_not_kgrid_converged",
                }
            )
    for x_value, frequency, rise in zip(
        extrapolated_x, extrapolated_absolute, extrapolated_rise
    ):
        rows.append(
            {
                "series": "DFPT_1_over_Nk_depth_limit",
                "kgrid": "infinity_extrapolation",
                "signed_distance": float(x_value),
                "frequency_cm-1": float(frequency),
                "K_reference_cm-1": k288_anchor,
                "rise_from_K_cm-1": float(rise),
                "status": "depth_limit_only_absolute_value_anchored_at_k288_K",
            }
        )
    csv_path = args.output_dir / "zero_smearing_raw_DFPT_kgrid_diagnostic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    d003_depths = {
        str(kgrid): {
            "KG_cm-1": value_at(data, -0.003, "rise_from_K_cm-1"),
            "KM_cm-1": value_at(data, 0.003, "rise_from_K_cm-1"),
        }
        for kgrid, data in raw.items()
    }
    k240_to_k288 = {
        direction: abs(
            d003_depths["288"][direction]
            - d003_depths["240"][direction]
        )
        / abs(d003_depths["288"][direction])
        for direction in ("KG_cm-1", "KM_cm-1")
    }
    summary = {
        "status": "zero_smearing_raw_DFPT_kgrid_diagnostic_generated",
        "electronic_integration": "tetrahedra_opt (no degauss)",
        "raw_DFPT": {
            "point_count_by_kgrid": {
                str(kgrid): len(data["x"]) for kgrid, data in raw.items()
            },
            "K_frequency_cm-1_by_kgrid": {
                str(kgrid): float(data["K_frequency_cm-1"])
                for kgrid, data in raw.items()
            },
            "d003_depth_cm-1_by_kgrid": d003_depths,
            "k240_to_k288_d003_relative_change": k240_to_k288,
            "points_joined_or_fitted": False,
        },
        "DFPT_1_over_Nk_depth_limit_cm-1": extrapolated_depth,
        "model": {
            "name": "fixed-background MLIP + full EPC long-range correction",
            "K_frequency_cm-1": float(model["K_frequency_cm-1"]),
            "d003_depth_cm-1": {
                "KG": value_at(model, -0.003, "rise_from_K_cm-1"),
                "KM": value_at(model, 0.003, "rise_from_K_cm-1"),
            },
        },
        "plot_semantics": {
            "solid_black": "MLIP + full EPC LR zero-smearing curve",
            "open_markers": "raw zero-smearing DFPT at each k grid",
            "green_stars": "d=0.003 DFPT 1/Nk depth limits; absolute values anchored at k288 K",
        },
        "limitations": [
            "outer zero-smearing DFPT q points are available mainly at k192 and are not k-grid-converged",
            "the 1/Nk result extrapolates the d=0.003 cusp depth, not the absolute K frequency",
            "only three k grids support the current 1/Nk depth extrapolation",
        ],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    summary_path = args.output_dir / "zero_smearing_raw_DFPT_kgrid_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
