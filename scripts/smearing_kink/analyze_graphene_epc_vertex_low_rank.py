#!/usr/bin/env python3
"""Measure the low-rank structure of the graphene K-A' EPC vertex.

The stored EPW real-space vertex is projected onto the tracked A' phonon mode
on both sides of K.  A consistent phonon-mode phase is imposed before an SVD.
The script reports representation rank and a coarse-q interpolation ablation;
it launches no DFT calculation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_graphene_epc_zero_adaptive import (  # noqa: E402
    mode_projected_coefficients,
)
from evaluate_graphene_epc_zero_transfer import qpoint  # noqa: E402
from extract_graphene_epw_dynamical_matrices import (  # noqa: E402
    load_crystal,
    load_rdw,
    zone_centered_wigner,
)
from replay_graphene_epc_bandsum import (  # noqa: E402
    dynamical_matrix_and_mode,
    load_epmatwp,
)


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"


def rank_for_capture(cumulative: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumulative, threshold) + 1)


def interpolate_latent(
    distance: np.ndarray,
    latent: np.ndarray,
    training_indices: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(latent)
    for component in range(latent.shape[1]):
        real = CubicSpline(
            distance[training_indices],
            latent[training_indices, component].real,
            bc_type="natural",
        )(distance)
        imaginary = CubicSpline(
            distance[training_indices],
            latent[training_indices, component].imag,
            bc_type="natural",
        )(distance)
        output[:, component] = real + 1j * imaginary
    output[training_indices] = latent[training_indices]
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-distance", type=float, default=0.03)
    parser.add_argument("--points-per-direction", type=int, default=61)
    parser.add_argument(
        "--epwdata",
        type=Path,
        default=BASE / "E0_epw_matched/k18_q9_ex1_pifroz/restart_small/epwdata.fmt",
    )
    parser.add_argument(
        "--crystal",
        type=Path,
        default=BASE / "E0_epw_matched/k18_q9_ex1_pifroz/restart_small/crystal.fmt",
    )
    parser.add_argument(
        "--epmatwp",
        type=Path,
        default=BASE / "E9_epc_bandsum_assets/graphene.epmatwp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E16_epc_vertex_low_rank",
    )
    args = parser.parse_args()
    if args.points_per_direction < 17:
        raise ValueError("points_per_direction must be at least 17")

    crystal = load_crystal(args.crystal, natoms=2)
    rdw, dimensions = load_rdw(args.epwdata)
    vectors_k, degeneracies_k, _ = zone_centered_wigner(
        crystal["at"], (18, 18, 1)
    )
    vectors_q, degeneracies_q, _ = zone_centered_wigner(
        crystal["at"], (9, 9, 1)
    )
    epmatwp = load_epmatwp(
        args.epmatwp,
        dimensions["nbndsub"],
        dimensions["nrr_k"],
        dimensions["nmodes"],
        dimensions["nrr_g"],
    )

    distances = np.linspace(
        0.0, args.maximum_distance, args.points_per_direction
    )
    records = [("K", 0.0)]
    records.extend(("KG", float(value)) for value in distances[1:])
    records.extend(("KM", float(value)) for value in distances[1:])
    qpoints = np.asarray([qpoint(direction, distance) for direction, distance in records])

    _, _, K_mode = dynamical_matrix_and_mode(
        rdw,
        qpoint("K", 0.0),
        vectors_q,
        degeneracies_q,
        crystal["masses_au"],
        crystal["atom_types"],
    )
    flattened = []
    mode_overlaps = []
    for q in qpoints:
        _, _, physical_mode = dynamical_matrix_and_mode(
            rdw,
            q,
            vectors_q,
            degeneracies_q,
            crystal["masses_au"],
            crystal["atom_types"],
        )
        overlap = np.vdot(K_mode, physical_mode)
        if abs(overlap) < 1e-10:
            raise RuntimeError("A' phonon-mode phase cannot be aligned")
        physical_mode = physical_mode * np.exp(-1j * np.angle(overlap))
        aligned_overlap = np.vdot(K_mode, physical_mode)
        mode_overlaps.append(abs(aligned_overlap) / (np.linalg.norm(K_mode) * np.linalg.norm(physical_mode)))
        coefficients = mode_projected_coefficients(
            epmatwp,
            q,
            vectors_q,
            degeneracies_q,
            physical_mode,
            degeneracies_k,
        )
        flattened.append(coefficients.reshape(-1))
    vertex = np.asarray(flattened)
    left, singular, right = np.linalg.svd(vertex, full_matrices=False)
    cumulative = np.cumsum(singular**2) / np.sum(singular**2)
    latent = left * singular[None, :]

    ranks = sorted(
        set(
            [1, 2, 4, 8, 12, 16, 24, 32]
            + [
                rank_for_capture(cumulative, threshold)
                for threshold in (0.99, 0.999, 0.9999, 0.99999)
            ]
        )
    )
    ranks = [rank for rank in ranks if rank <= len(singular)]
    reconstruction = {}
    row_norm = np.linalg.norm(vertex, axis=1)
    for rank in ranks:
        approximate = latent[:, :rank] @ right[:rank]
        relative = np.linalg.norm(approximate - vertex, axis=1) / row_norm
        reconstruction[str(rank)] = {
            "cumulative_squared_singular_value": float(cumulative[rank - 1]),
            "median_q_relative_vertex_error": float(np.median(relative)),
            "maximum_q_relative_vertex_error": float(np.max(relative)),
        }

    direction_indices = {
        direction: np.asarray(
            [0]
            + [
                index
                for index, (label, _) in enumerate(records)
                if label == direction
            ],
            int,
        )
        for direction in ("KG", "KM")
    }
    interpolation = {}
    interpolation_ranks = [
        rank for rank in (2, 3, 4, 8, 12, 16) if rank <= len(singular)
    ]
    for number_of_labels in (5, 7, 9, 13, 17):
        if number_of_labels > args.points_per_direction:
            continue
        label_result = {}
        local_training = np.unique(
            np.rint(
                np.linspace(0, args.points_per_direction - 1, number_of_labels)
            ).astype(int)
        )
        for rank in interpolation_ranks:
            by_direction = {}
            all_errors = []
            for direction, indices in direction_indices.items():
                direction_latent = latent[indices, :rank]
                predicted_latent = interpolate_latent(
                    distances, direction_latent, local_training
                )
                predicted = predicted_latent @ right[:rank]
                relative = (
                    np.linalg.norm(predicted - vertex[indices], axis=1)
                    / np.linalg.norm(vertex[indices], axis=1)
                )
                held_out = np.ones(args.points_per_direction, bool)
                held_out[local_training] = False
                held_errors = relative[held_out]
                by_direction[direction] = {
                    "heldout_median_relative_vertex_error": float(
                        np.median(held_errors)
                    ),
                    "heldout_maximum_relative_vertex_error": float(
                        np.max(held_errors)
                    ),
                }
                all_errors.extend(held_errors.tolist())
            label_result[str(rank)] = {
                "directions": by_direction,
                "pooled_heldout_RMSE_relative_vertex_error": float(
                    np.sqrt(np.mean(np.square(all_errors)))
                ),
                "pooled_heldout_maximum_relative_vertex_error": float(
                    np.max(all_errors)
                ),
            }
        interpolation[str(number_of_labels)] = {
            "training_indices_per_direction": local_training.tolist(),
            "rank_results": label_result,
        }

    threshold_ranks = {
        str(threshold): rank_for_capture(cumulative, threshold)
        for threshold in (0.99, 0.999, 0.9999, 0.99999)
    }
    selected_rank = threshold_ranks["0.9999"]
    original_complex_values = int(np.prod(epmatwp.shape))
    projected_dense_complex_values = int(vertex.size)
    low_rank_complex_values = int(
        vertex.shape[0] * selected_rank + selected_rank * vertex.shape[1]
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "Aprime_vertex_low_rank_basis.npz",
        signed_distance=np.asarray(
            [
                0.0
                if direction == "K"
                else (-distance if direction == "KG" else distance)
                for direction, distance in records
            ]
        ),
        qpoints=qpoints,
        singular_values=singular,
        right_basis=right[:selected_rank],
        latent_coefficients=latent[:, :selected_rank],
        selected_rank=np.asarray(selected_rank),
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    axis = axes[0]
    axis.semilogx(
        np.arange(1, len(cumulative) + 1),
        cumulative,
        "o-",
        ms=3,
        lw=1.4,
        color="#2676b8",
    )
    for threshold in (0.99, 0.999, 0.9999):
        axis.axhline(threshold, color="#BBBBBB", lw=0.7)
    axis.set_xlabel("complex SVD rank")
    axis.set_ylabel("cumulative squared singular value")
    axis.set_title("A′ mode-projected vertex")
    axis.grid(alpha=0.18)

    axis = axes[1]
    plot_ranks = np.asarray(ranks, int)
    axis.loglog(
        plot_ranks,
        [reconstruction[str(rank)]["median_q_relative_vertex_error"] for rank in plot_ranks],
        "o-",
        color="#2676b8",
        label="median q",
    )
    axis.loglog(
        plot_ranks,
        [reconstruction[str(rank)]["maximum_q_relative_vertex_error"] for rank in plot_ranks],
        "s-",
        color="#d95f02",
        label="maximum q",
    )
    axis.set_xlabel("complex SVD rank")
    axis.set_ylabel("relative vertex error")
    axis.set_title("Direct low-rank reconstruction")
    axis.grid(alpha=0.18, which="both")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=False,
    )
    figure.suptitle("Graphene K-A′ effective EPC compression")
    figure.subplots_adjust(left=0.09, right=0.80, bottom=0.15, top=0.86, wspace=0.30)
    figure.savefig(
        output / "Aprime_vertex_low_rank.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "status": "Aprime_effective_vertex_low_rank_measured",
        "scope": (
            "stored full EPW vertex projected onto a phase-aligned A' mode in "
            "the K neighborhood; representation study only; no new DFT"
        ),
        "q_window": {
            "maximum_distance": args.maximum_distance,
            "points_per_direction_including_K": args.points_per_direction,
            "total_unique_qpoints": len(qpoints),
            "minimum_phonon_mode_overlap_with_K": float(min(mode_overlaps)),
        },
        "matrix_shape_q_by_complex_vertex": list(vertex.shape),
        "rank_for_cumulative_squared_singular_value": threshold_ranks,
        "reconstruction_by_rank": reconstruction,
        "coarse_q_latent_interpolation": interpolation,
        "storage_complex_values": {
            "original_full_epmatwp": original_complex_values,
            "dense_Aprime_q_window": projected_dense_complex_values,
            f"rank_{selected_rank}_Aprime_factorization": low_rank_complex_values,
            "rank_factorization_fraction_of_full_epmatwp": (
                low_rank_complex_values / original_complex_values
            ),
        },
        "selected_export_rank": selected_rank,
        "limitations": [
            "the SVD basis is learned from this graphene q window and is not yet a cross-material generative prior",
            "coarse-q interpolation uses a basis obtained from all q points, so it measures latent smoothness rather than a blind label-count result",
            "vertex-norm error must next be propagated through the band-sum response before choosing the deployable rank",
        ],
    }
    (output / "Aprime_vertex_low_rank_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
