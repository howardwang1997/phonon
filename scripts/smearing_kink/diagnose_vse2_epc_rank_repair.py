#!/usr/bin/env python3
"""Post-failure rank sweep for the fixed VSe2 five-q EPC holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "E18_VSe2_rank3_five_q_blind"
)


def interpolate(
    coordinate: np.ndarray,
    training_indices: np.ndarray,
    latent: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(coordinate), latent.shape[1]), complex)
    for component in range(latent.shape[1]):
        output[:, component] = CubicSpline(
            coordinate[training_indices],
            latent[:, component].real,
            bc_type="natural",
        )(coordinate) + 1j * CubicSpline(
            coordinate[training_indices],
            latent[:, component].imag,
            bc_type="natural",
        )(coordinate)
    output[training_indices] = latent
    return output


def interpolate_chebyshev(
    coordinate: np.ndarray,
    training_indices: np.ndarray,
    latent: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(coordinate), latent.shape[1]), complex)
    domain = [float(coordinate[0]), float(coordinate[-1])]
    for component in range(latent.shape[1]):
        real = Chebyshev.fit(
            coordinate[training_indices],
            latent[:, component].real,
            deg=len(training_indices) - 1,
            domain=domain,
        )(coordinate)
        imaginary = Chebyshev.fit(
            coordinate[training_indices],
            latent[:, component].imag,
            deg=len(training_indices) - 1,
            domain=domain,
        )(coordinate)
        output[:, component] = real + 1j * imaginary
    output[training_indices] = latent
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE / "rank3_five_q_blind_arrays.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE.parent / "E19_VSe2_rank_repair",
    )
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as payload:
        coordinate = np.asarray(payload["coordinate"])
        vertex = np.asarray(payload["full_vertex"])
        training_indices = np.asarray(payload["training_indices"], int)
        adjacent_overlaps = np.asarray(payload["adjacent_mode_overlaps"])
    heldout = np.ones(len(coordinate), bool)
    heldout[training_indices] = False
    row_norm = np.linalg.norm(vertex, axis=1)

    full_left, full_singular, full_right = np.linalg.svd(
        vertex, full_matrices=False
    )
    cumulative = np.cumsum(full_singular**2) / np.sum(full_singular**2)
    train_left, train_singular, train_right = np.linalg.svd(
        vertex[training_indices], full_matrices=False
    )

    results = {}
    for rank in range(1, 6):
        full_approximation = (
            full_left[:, :rank] * full_singular[None, :rank]
        ) @ full_right[:rank]
        full_relative = np.linalg.norm(full_approximation - vertex, axis=1) / row_norm
        training_latent = (
            train_left[:, :rank] * train_singular[None, :rank]
        )
        predicted_latent = interpolate(
            coordinate, training_indices, training_latent
        )
        blind = predicted_latent @ train_right[:rank]
        blind_relative = np.linalg.norm(blind - vertex, axis=1) / row_norm
        results[str(rank)] = {
            "full_data_capture": float(cumulative[rank - 1]),
            "full_data_maximum_relative_error": float(np.max(full_relative)),
            "blind_heldout_RMSE_relative_error": float(
                np.sqrt(np.mean(blind_relative[heldout] ** 2))
            ),
            "blind_heldout_median_relative_error": float(
                np.median(blind_relative[heldout])
            ),
            "blind_heldout_maximum_relative_error": float(
                np.max(blind_relative[heldout])
            ),
        }

    selected_rank = min(
        (
            rank
            for rank in range(1, 6)
            if results[str(rank)]["blind_heldout_RMSE_relative_error"] < 0.02
            and results[str(rank)]["blind_heldout_maximum_relative_error"] < 0.05
        ),
        default=None,
    )
    status = (
        "post_failure_rank_repair_identified"
        if selected_rank is not None
        else "rank_only_repair_insufficient"
    )

    interpolation_models = {}
    rank = 4
    training_latent = train_left[:, :rank] * train_singular[None, :rank]
    for name, predicted_latent in {
        "natural_cubic": interpolate(
            coordinate, training_indices, training_latent
        ),
        "degree4_Chebyshev": interpolate_chebyshev(
            coordinate, training_indices, training_latent
        ),
    }.items():
        predicted = predicted_latent @ train_right[:rank]
        relative = np.linalg.norm(predicted - vertex, axis=1) / row_norm
        interpolation_models[name] = {
            "heldout_RMSE_relative_vertex_error": float(
                np.sqrt(np.mean(relative[heldout] ** 2))
            ),
            "heldout_median_relative_vertex_error": float(
                np.median(relative[heldout])
            ),
            "heldout_maximum_relative_vertex_error": float(
                np.max(relative[heldout])
            ),
        }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    ranks = np.arange(1, 6)
    axis.semilogy(
        ranks,
        [results[str(rank)]["blind_heldout_RMSE_relative_error"] for rank in ranks],
        "o-",
        color="#2676b8",
        label="held-out RMSE",
    )
    axis.semilogy(
        ranks,
        [results[str(rank)]["blind_heldout_maximum_relative_error"] for rank in ranks],
        "s-",
        color="#d95f02",
        label="held-out maximum",
    )
    axis.axhline(0.02, color="#2676b8", ls="--", lw=0.9, alpha=0.7)
    axis.axhline(0.05, color="#d95f02", ls="--", lw=0.9, alpha=0.7)
    axis.set_xticks(ranks)
    axis.set_xlabel("material-specific complex EPC rank")
    axis.set_ylabel("relative vertex error")
    axis.set_title("1T-VSe₂ five-q holdout: post-failure rank sweep")
    axis.grid(alpha=0.18, which="both")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.78, 0.5),
        frameon=False,
    )
    figure.subplots_adjust(left=0.13, right=0.76, bottom=0.15, top=0.88)
    figure.savefig(
        output / "VSe2_rank_repair.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "status": status,
        "scope": (
            "post-failure diagnostic on the same fixed five-q VSe2 split; "
            "not counted as a blind validation"
        ),
        "rank_results": results,
        "smallest_rank_meeting_original_vertex_error_thresholds": selected_rank,
        "minimum_adjacent_mode_overlap": float(np.min(adjacent_overlaps)),
        "rank4_five_q_interpolation_models": interpolation_models,
        "interpretation": (
            "A high adjacent-mode overlap indicates smooth mode rotation.  If rank 4 "
            "passes while rank 3 narrowly fails, the reusable model should predict an "
            "adaptive rank capped at 4 rather than add q labels or dense DFT."
        ),
    }
    (output / "VSe2_rank_repair_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
