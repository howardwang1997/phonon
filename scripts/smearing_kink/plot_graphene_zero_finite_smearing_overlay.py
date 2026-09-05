#!/usr/bin/env python3
"""Plot graphene K->Gamma A' curves at zero and finite electronic smearing.

The zero-smearing dense curve and the finite-smearing replay only share the
K->Gamma direction.  This script deliberately compares that common path and
adds a K-referenced panel so that the cusp shape can be compared separately
from the lattice-temperature-dependent absolute frequency offset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


REPOSITORY = Path(__file__).resolve().parents[2]
BASE = REPOSITORY / "results/graphene_physics_temperature/post_p4_feasibility"
TEMPERATURES_K = (300, 450, 600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zero-curve",
        type=Path,
        default=BASE / "E15_epc_zero_dense_curve/zero_dense_curve.csv",
    )
    parser.add_argument(
        "--finite-replay",
        type=Path,
        default=(
            BASE
            / "E9_epc_bandsum_replay_nk720/epc_bandsum_replay_arrays.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E29_zero_finite_smearing_overlay",
    )
    parser.add_argument("--maximum-distance", type=float, default=0.03)
    return parser.parse_args()


def load_zero_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    records: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["direction"] not in {"K", "KG"}:
                continue
            records.append(
                (
                    float(row["distance"]),
                    float(row["MLIP_plus_full_EPC_cm-1"]),
                )
            )
    records.sort(key=lambda item: item[0])
    distance = np.asarray([record[0] for record in records], float)
    frequency = np.asarray([record[1] for record in records], float)
    if len(distance) == 0 or not np.isclose(distance[0], 0.0):
        raise RuntimeError("zero-smearing K->Gamma curve does not contain K")
    if np.any(np.diff(distance) <= 0.0):
        raise RuntimeError("zero-smearing K->Gamma distances are not unique")
    return distance, frequency


def signed_frequency(background_cm1: np.ndarray, correction_cm2: np.ndarray) -> np.ndarray:
    squared = background_cm1**2 + correction_cm2
    return np.sign(squared) * np.sqrt(np.abs(squared))


def load_finite_curves(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    with np.load(path) as arrays:
        t_gk = np.asarray(arrays["t_GK"], float)
        degauss_ry = np.asarray(arrays["degauss_Ry"], float)
        target_correction = np.asarray(arrays["target_correction_cm2"], float)
        replay_correction = np.asarray(arrays["replay_correction_cm2"], float)
        background = np.asarray(arrays["background_cm1"], float)

    center = int(np.argmin(np.abs(t_gk - 1.0)))
    anchored_correction = (
        target_correction[:, [center]]
        + replay_correction
        - replay_correction[:, [center]]
    )
    frequency = signed_frequency(background, anchored_correction)

    kg_mask = t_gk <= 1.0 + 1.0e-10
    distance = 1.0 - t_gk[kg_mask]
    order = np.argsort(distance)
    distance = distance[order]
    curves = [(distance, row[kg_mask][order]) for row in frequency]
    if not np.isclose(distance[0], 0.0):
        raise RuntimeError("finite-smearing K->Gamma curves do not contain K")
    return degauss_ry, t_gk, curves


def pchip_values(
    distance: np.ndarray,
    frequency: np.ndarray,
    evaluation_distance: np.ndarray,
) -> np.ndarray:
    return np.asarray(PchipInterpolator(distance, frequency)(evaluation_distance), float)


def main() -> int:
    args = parse_args()
    zero_distance, zero_frequency = load_zero_curve(args.zero_curve)
    degauss_ry, _, finite_curves = load_finite_curves(args.finite_replay)
    if len(degauss_ry) != len(TEMPERATURES_K):
        raise RuntimeError("unexpected number of finite-smearing curves")

    series = [
        {
            "smearing_degauss_Ry": 0.0,
            "lattice_temperature_K": None,
            "distance": zero_distance,
            "frequency": zero_frequency,
            "color": "#111111",
            "label": "0 Ry",
            "linewidth": 2.7,
        }
    ]
    finite_colors = ("#0072B2", "#009E73", "#D55E00")
    for temperature, smearing, color, (distance, frequency) in zip(
        TEMPERATURES_K, degauss_ry, finite_colors, finite_curves
    ):
        series.append(
            {
                "smearing_degauss_Ry": float(smearing),
                "lattice_temperature_K": temperature,
                "distance": distance,
                "frequency": frequency,
                "color": color,
                "label": f"{smearing:.7f} Ry; lattice {temperature} K",
                "linewidth": 2.1,
            }
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for index, item in enumerate(series):
        mask = item["distance"] <= args.maximum_distance + 1.0e-12
        raw_x = item["distance"][mask]
        raw_y = item["frequency"][mask]
        smooth_x = np.linspace(0.0, args.maximum_distance, 401)
        smooth_y = pchip_values(item["distance"], item["frequency"], smooth_x)
        for axis_index, axis in enumerate(axes):
            y = smooth_y if axis_index == 0 else smooth_y - smooth_y[0]
            axis.plot(
                smooth_x,
                y,
                color=item["color"],
                lw=item["linewidth"],
                label=item["label"],
            )
            if index > 0:
                raw_plot_y = raw_y if axis_index == 0 else raw_y - raw_y[0]
                axis.plot(
                    raw_x,
                    raw_plot_y,
                    "o",
                    color=item["color"],
                    ms=3.0,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    zorder=3,
                )

    comparison_distance = 1.0 / 120.0
    axes[1].axvline(
        comparison_distance,
        color="#999999",
        lw=1.0,
        ls="--",
        zorder=0,
    )
    axes[1].text(
        comparison_distance + 0.00045,
        0.97,
        r"$|q-K|=1/120$",
        color="#666666",
        fontsize=9,
        ha="left",
        va="top",
        transform=axes[1].get_xaxis_transform(),
    )
    axes[0].set_title("Absolute A′ frequency")
    axes[1].set_title("K-referenced cusp shape")
    axes[0].set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    axes[1].set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    for axis in axes:
        axis.set_xlabel(r"distance from K toward $\Gamma$")
        axis.set_xlim(0.0, args.maximum_distance)
        axis.grid(alpha=0.18)
    figure.suptitle(
        "Graphene K→Γ A′: zero and finite smearing/degauss",
        fontsize=16,
        y=0.98,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.81, 0.56),
        frameon=False,
        title="smearing/degauss (Ry)",
    )
    figure.text(
        0.81,
        0.28,
        "Finite-smearing curves use their\n"
        "paired lattice-temperature backgrounds.\n"
        "Dots: computed q points; lines: PCHIP.",
        fontsize=9,
        color="#555555",
        ha="left",
        va="top",
    )
    figure.subplots_adjust(left=0.08, right=0.79, bottom=0.14, top=0.85, wspace=0.27)
    png_path = output / "zero_finite_smearing_KG_Aprime_overlay.png"
    pdf_path = output / "zero_finite_smearing_KG_Aprime_overlay.pdf"
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    csv_path = output / "zero_finite_smearing_KG_Aprime_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "smearing_degauss_Ry",
            "lattice_temperature_K",
            "distance_from_K_toward_Gamma",
            "frequency_cm-1",
            "frequency_minus_K_cm-1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in series:
            k_frequency = float(item["frequency"][0])
            for distance, frequency in zip(item["distance"], item["frequency"]):
                if distance > args.maximum_distance + 1.0e-12:
                    continue
                writer.writerow(
                    {
                        "smearing_degauss_Ry": item["smearing_degauss_Ry"],
                        "lattice_temperature_K": (
                            ""
                            if item["lattice_temperature_K"] is None
                            else item["lattice_temperature_K"]
                        ),
                        "distance_from_K_toward_Gamma": float(distance),
                        "frequency_cm-1": float(frequency),
                        "frequency_minus_K_cm-1": float(frequency - k_frequency),
                    }
                )

    comparison = []
    for item in series:
        value = float(
            pchip_values(
                item["distance"],
                item["frequency"],
                np.asarray([comparison_distance]),
            )[0]
        )
        comparison.append(
            {
                "smearing_degauss_Ry": item["smearing_degauss_Ry"],
                "lattice_temperature_K": item["lattice_temperature_K"],
                "K_frequency_cm-1": float(item["frequency"][0]),
                "rise_at_distance_1_over_120_cm-1": value
                - float(item["frequency"][0]),
            }
        )
    summary = {
        "status": "zero_and_finite_smearing_common_path_overlay_generated",
        "path": "K toward Gamma",
        "maximum_distance": args.maximum_distance,
        "curves": {
            "zero": "short-range MLIP + full EPC long-range correction",
            "finite": (
                "finite-lattice short-range background + nk720 full-EPC "
                "band-sum replay, anchored to each finite-smearing K point"
            ),
        },
        "comparison_at_distance_1_over_120": comparison,
        "sources": {
            "zero_curve": str(args.zero_curve.resolve()),
            "finite_replay": str(args.finite_replay.resolve()),
        },
        "display": {
            "finite_computed_q_points_are_marked": True,
            "line_interpolation": "PCHIP shape-preserving interpolation",
        },
        "limitations": [
            (
                "finite-smearing curves use lattice-temperature-dependent "
                "short-range backgrounds at 300, 450, and 600 K, so absolute "
                "vertical shifts are not a fixed-lattice-temperature smearing sweep"
            ),
            (
                "the K-referenced panel removes each curve's K frequency and is "
                "the appropriate panel for comparing cusp rounding"
            ),
            (
                "only the common K-to-Gamma path is overlaid; the finite data do "
                "not contain the K-to-M path used on the positive side of E15"
            ),
        ],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "samples_csv": str(csv_path.resolve()),
        },
    }
    summary_path = output / "zero_finite_smearing_KG_Aprime_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
