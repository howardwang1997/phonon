#!/usr/bin/env python3
"""Plot a controlled graphene smearing sweep on one fixed MLIP background.

All curves use the same static short-range MLIP dynamical matrix and one
common long-range reference constant.  No lattice-temperature-dependent L0
background and no per-smearing K anchor enters this comparison.
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


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_graphene_epw_dynamical_matrices import RYDBERG_CM1  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
COLORS = ("#111111", "#0072B2", "#009E73", "#D55E00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-response",
        type=Path,
        default=BASE
        / "E30_zero_finite_full_kpath/zero_finite_full_kpath_Aprime.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E31_fixed_background_smearing_sweep",
    )
    return parser.parse_args()


def signed_frequency(background_cm1: np.ndarray, correction_cm2: np.ndarray) -> np.ndarray:
    squared = np.asarray(background_cm1, float) ** 2 + np.asarray(
        correction_cm2, float
    )
    return np.sign(squared) * np.sqrt(np.abs(squared))


def load_series(path: Path) -> tuple[np.ndarray, list[list[dict[str, str]]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    smearings = np.asarray(
        sorted({float(row["smearing_degauss_Ry"]) for row in rows}), float
    )
    blocks = [
        [row for row in rows if float(row["smearing_degauss_Ry"]) == smearing]
        for smearing in smearings
    ]
    if len(smearings) != 4 or any(len(block) != 121 for block in blocks):
        raise RuntimeError("expected four 121-point smearing curves")
    reference_q = np.asarray(
        [[float(row["q1"]), float(row["q2"])] for row in blocks[0]], float
    )
    for block in blocks[1:]:
        qpoints = np.asarray(
            [[float(row["q1"]), float(row["q2"])] for row in block], float
        )
        if not np.allclose(qpoints, reference_q, atol=1.0e-12, rtol=0.0):
            raise RuntimeError("smearing curves do not share identical q points")
    return smearings, blocks


def distance_index(block: list[dict[str, str]], direction: str, distance: float) -> int:
    return next(
        index
        for index, row in enumerate(block)
        if row["direction"] == direction
        and np.isclose(float(row["distance"]), distance, atol=1.0e-14)
    )


def make_figure(
    png_path: Path,
    pdf_path: Path,
    smearings: np.ndarray,
    block: list[dict[str, str]],
    frequencies: np.ndarray,
) -> None:
    signed_distance = np.asarray(
        [float(row["signed_distance"]) for row in block], float
    )
    order = np.argsort(signed_distance)
    figure, axes = plt.subplots(2, 2, figsize=(14.4, 8.7))
    absolute_axis, centered_axis, zoom_axis, K_axis = axes.flat
    for index, (smearing, color) in enumerate(zip(smearings, COLORS)):
        label = f"{smearing:.7f} Ry" if smearing > 0.0 else "0 Ry"
        linewidth = 2.7 if index == 0 else 2.1
        centered = frequencies[index] - frequencies[index, 0]
        absolute_axis.plot(
            signed_distance[order],
            frequencies[index, order],
            color=color,
            lw=linewidth,
            label=label,
        )
        centered_axis.plot(
            signed_distance[order], centered[order], color=color, lw=linewidth
        )
        zoom_axis.plot(
            signed_distance[order], centered[order], color=color, lw=linewidth
        )

    K_axis.plot(
        smearings,
        frequencies[:, 0],
        "o-",
        color="#3A3A3A",
        lw=2.0,
        ms=5.5,
    )
    for smearing, frequency in zip(smearings, frequencies[:, 0]):
        K_axis.annotate(
            f"{frequency:.2f}",
            (smearing, frequency),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.7,
            color="#333333",
        )

    absolute_axis.set_title("Absolute frequency on one fixed background")
    centered_axis.set_title("K-referenced cusp: full window")
    zoom_axis.set_title("K-referenced cusp: inner window")
    K_axis.set_title("K-point frequency: controlled smearing sweep")
    absolute_axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    centered_axis.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    zoom_axis.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    K_axis.set_ylabel(r"$\omega(K)$ (cm$^{-1}$)")
    for axis in (absolute_axis, centered_axis, zoom_axis):
        axis.axvline(0.0, color="#AAAAAA", lw=0.9, zorder=0)
        axis.set_xlabel("signed distance from K (K→Γ < 0; K→M > 0)")
        axis.grid(alpha=0.18)
    absolute_axis.set_xlim(-0.03, 0.03)
    centered_axis.set_xlim(-0.03, 0.03)
    zoom_axis.set_xlim(-0.008, 0.008)
    inner_mask = np.abs(signed_distance) <= 0.008 + 1.0e-12
    inner_maximum = float(
        np.max(frequencies[:, inner_mask] - frequencies[:, [0]])
    )
    zoom_axis.set_ylim(-0.08, 1.08 * inner_maximum)
    K_axis.set_xlabel("smearing/degauss (Ry)")
    K_axis.grid(alpha=0.18)
    K_axis.margins(x=0.08, y=0.16)

    figure.suptitle(
        "Graphene K-neighbourhood A′: electronic-smearing-only comparison",
        fontsize=16,
        y=0.985,
    )
    handles, labels = absolute_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.80, 0.69),
        frameon=False,
        title="smearing/degauss (Ry)",
    )
    figure.text(
        0.80,
        0.47,
        "Same static MLIP short-range D(q)\n"
        "for every curve.\n"
        "One common EPC reference constant.\n"
        "No lattice-temperature background.\n"
        "No per-smearing K re-anchoring.",
        fontsize=9.4,
        color="#555555",
        ha="left",
        va="top",
    )
    figure.subplots_adjust(
        left=0.07, right=0.78, bottom=0.09, top=0.90, wspace=0.25, hspace=0.33
    )
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    smearings, blocks = load_series(args.full_response)
    zero_block = blocks[0]
    common_background = np.asarray(
        [float(row["short_range_background_cm-1"]) for row in zero_block], float
    )
    response_Ry2 = np.asarray(
        [
            [float(row["full_EPC_response_Ry2"]) for row in block]
            for block in blocks
        ],
        float,
    )
    response_cm2 = response_Ry2 * RYDBERG_CM1**2
    zero_stored_correction = np.asarray(
        [float(row["long_range_correction_cm-2"]) for row in zero_block], float
    )
    common_constant_samples = zero_stored_correction - response_cm2[0]
    common_constant = float(np.mean(common_constant_samples))
    common_constant_spread = float(np.ptp(common_constant_samples))
    if common_constant_spread > 1.0e-5:
        raise RuntimeError("zero curve does not contain one common EPC reference constant")

    controlled_correction = common_constant + response_cm2
    controlled_frequency = signed_frequency(
        common_background[None, :], controlled_correction
    )
    zero_replay_error = float(
        np.max(
            np.abs(
                controlled_frequency[0]
                - np.asarray(
                    [
                        float(row["MLIP_plus_full_EPC_cm-1"])
                        for row in zero_block
                    ],
                    float,
                )
            )
        )
    )

    metrics = []
    for series_index, smearing in enumerate(smearings):
        row = {
            "smearing_degauss_Ry": float(smearing),
            "K_frequency_cm-1": float(controlled_frequency[series_index, 0]),
        }
        for direction in ("KG", "KM"):
            index = distance_index(zero_block, direction, 0.003)
            row[f"{direction}_d003_rise_cm-1"] = float(
                controlled_frequency[series_index, index]
                - controlled_frequency[series_index, 0]
            )
        metrics.append(row)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "fixed_background_smearing_sweep_Aprime.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "smearing_degauss_Ry",
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for series_index, smearing in enumerate(smearings):
            for q_index, source_row in enumerate(zero_block):
                writer.writerow(
                    {
                        "smearing_degauss_Ry": float(smearing),
                        "direction": source_row["direction"],
                        "distance": source_row["distance"],
                        "signed_distance": source_row["signed_distance"],
                        "q1": source_row["q1"],
                        "q2": source_row["q2"],
                        "common_static_MLIP_background_cm-1": common_background[
                            q_index
                        ],
                        "full_EPC_response_Ry2": response_Ry2[
                            series_index, q_index
                        ],
                        "common_reference_constant_cm-2": common_constant,
                        "controlled_long_range_correction_cm-2": controlled_correction[
                            series_index, q_index
                        ],
                        "controlled_MLIP_plus_full_EPC_cm-1": controlled_frequency[
                            series_index, q_index
                        ],
                    }
                )

    png_path = output / "fixed_background_smearing_sweep_Aprime.png"
    pdf_path = output / "fixed_background_smearing_sweep_Aprime.pdf"
    make_figure(
        png_path,
        pdf_path,
        smearings,
        zero_block,
        controlled_frequency,
    )
    K_frequencies = controlled_frequency[:, 0]
    mean_depths = np.asarray(
        [
            0.5
            * (row["KG_d003_rise_cm-1"] + row["KM_d003_rise_cm-1"])
            for row in metrics
        ]
    )
    summary = {
        "status": "fixed_background_electronic_smearing_sweep_generated",
        "scope": (
            "same static MLIP short-range background and one common EPC "
            "reference constant for all electronic smearings"
        ),
        "construction": {
            "formula": (
                "omega(q,s)^2 = omega_SR_static(q)^2 + C_common "
                "+ Pi_full_EPC(q,s)-Pi_reference(q)"
            ),
            "common_reference_constant_cm-2": common_constant,
            "common_constant_spread_cm-2": common_constant_spread,
            "per_smearing_K_reanchoring": False,
            "lattice_temperature_background_used": False,
        },
        "series_metrics": metrics,
        "checks": {
            "zero_curve_replay_max_abs_cm-1": zero_replay_error,
            "K_frequency_increases_monotonically_with_smearing": bool(
                np.all(np.diff(K_frequencies) > 0.0)
            ),
            "mean_d003_cusp_depth_decreases_monotonically_with_smearing": bool(
                np.all(np.diff(mean_depths) < 0.0)
            ),
            "all_frequencies_finite": bool(np.isfinite(controlled_frequency).all()),
        },
        "source": str(args.full_response.resolve()),
        "limitations": [
            (
                "the one common absolute EPC reference constant is calibrated "
                "once from the accepted zero-smearing K anchor"
            ),
            (
                "the stored zero and finite electronic response integrations "
                "retain their previously validated numerical-regulator settings"
            ),
            (
                "this is a controlled electronic-smearing comparison, not a "
                "finite-lattice-temperature phonon spectrum"
            ),
        ],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    summary_path = output / "fixed_background_smearing_sweep_Aprime_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
