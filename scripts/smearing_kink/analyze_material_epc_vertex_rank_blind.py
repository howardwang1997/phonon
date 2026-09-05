#!/usr/bin/env python3
"""Blind five-q low-rank EPC-vertex test for a material-specific mode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_graphene_epc_zero_adaptive import (  # noqa: E402
    mode_projected_coefficients,
)
from extract_graphene_epw_dynamical_matrices import (  # noqa: E402
    RYDBERG_CM1,
    load_crystal,
    load_rdw,
    zone_centered_wigner,
)
from replay_graphene_epc_bandsum import load_epmatwp  # noqa: E402


BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "E18_cross_system_assets/1T-VSe2"
)


def dynamical_modes(
    rdw: np.ndarray,
    qpoint: np.ndarray,
    vectors_q: np.ndarray,
    degeneracies_q: np.ndarray,
    masses: np.ndarray,
    atom_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phases = np.exp(2j * np.pi * (vectors_q @ qpoint)) / degeneracies_q
    matrix = np.einsum("abr,r->ab", rdw, phases, optimize=True)
    for atom_a, type_a in enumerate(atom_types):
        for atom_b, type_b in enumerate(atom_types):
            factor = np.sqrt(masses[type_a - 1] * masses[type_b - 1])
            matrix[
                3 * atom_a : 3 * atom_a + 3,
                3 * atom_b : 3 * atom_b + 3,
            ] /= factor
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def interpolate_from_five_labels(
    coordinate: np.ndarray,
    training_indices: np.ndarray,
    training_latent: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(coordinate), training_latent.shape[1]), complex)
    for component in range(training_latent.shape[1]):
        output[:, component] = CubicSpline(
            coordinate[training_indices],
            training_latent[:, component].real,
            bc_type="natural",
        )(coordinate) + 1j * CubicSpline(
            coordinate[training_indices],
            training_latent[:, component].imag,
            bc_type="natural",
        )(coordinate)
    output[training_indices] = training_latent
    return output


def interpolate_chebyshev_from_five_labels(
    coordinate: np.ndarray,
    training_indices: np.ndarray,
    training_latent: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(coordinate), training_latent.shape[1]), complex)
    domain = [float(coordinate[0]), float(coordinate[-1])]
    for component in range(training_latent.shape[1]):
        output[:, component] = Chebyshev.fit(
            coordinate[training_indices],
            training_latent[:, component].real,
            deg=len(training_indices) - 1,
            domain=domain,
        )(coordinate) + 1j * Chebyshev.fit(
            coordinate[training_indices],
            training_latent[:, component].imag,
            deg=len(training_indices) - 1,
            domain=domain,
        )(coordinate)
    output[training_indices] = training_latent
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="1T-VSe2")
    parser.add_argument("--natoms", type=int, default=3)
    parser.add_argument("--coarse-k", type=int, nargs=3, default=(12, 12, 1))
    parser.add_argument("--coarse-q", type=int, nargs=3, default=(3, 3, 1))
    parser.add_argument("--q-center", type=float, nargs=3, default=(0.25, 0.25, 0.0))
    parser.add_argument("--q-direction", type=float, nargs=3, default=(1.0, 1.0, 0.0))
    parser.add_argument("--halfwidth", type=float, default=0.05)
    parser.add_argument("--number-of-qpoints", type=int, default=41)
    parser.add_argument("--tracked-mode-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument(
        "--interpolation",
        choices=("natural_cubic", "degree4_Chebyshev"),
        default="natural_cubic",
    )
    parser.add_argument("--hr", type=Path, default=BASE / "1TVSe2_hr.dat")
    parser.add_argument("--epwdata", type=Path, default=BASE / "epwdata.fmt")
    parser.add_argument("--crystal", type=Path, default=BASE / "crystal.fmt")
    parser.add_argument("--epmatwp", type=Path, default=BASE / "1TVSe2.epmatwp")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE.parent.parent / "E18_VSe2_rank3_five_q_blind",
    )
    args = parser.parse_args()
    if args.number_of_qpoints != 41:
        raise ValueError("the fixed blind test requires exactly 41 q points")
    if not 1 <= args.rank <= 5:
        raise ValueError("rank must be between one and five for the five-q test")

    crystal = load_crystal(args.crystal, natoms=args.natoms)
    rdw, dimensions = load_rdw(args.epwdata)
    vectors_k, degeneracies_k, _ = zone_centered_wigner(
        crystal["at"], tuple(args.coarse_k)
    )
    vectors_q, degeneracies_q, _ = zone_centered_wigner(
        crystal["at"], tuple(args.coarse_q)
    )
    if len(vectors_k) != dimensions["nrr_k"]:
        raise ValueError("electronic WS count does not match epwdata")
    if len(vectors_q) != dimensions["nrr_g"]:
        raise ValueError("phonon WS count does not match epwdata")
    epmatwp = load_epmatwp(
        args.epmatwp,
        dimensions["nbndsub"],
        dimensions["nrr_k"],
        dimensions["nmodes"],
        dimensions["nrr_g"],
    )

    coordinate = np.linspace(-args.halfwidth, args.halfwidth, args.number_of_qpoints)
    q_center = np.asarray(args.q_center, float)
    q_direction = np.asarray(args.q_direction, float)
    qpoints = q_center[None, :] + coordinate[:, None] * q_direction[None, :]
    center_index = args.number_of_qpoints // 2
    reference_eigenvalues, reference_eigenvectors = dynamical_modes(
        rdw,
        q_center,
        vectors_q,
        degeneracies_q,
        crystal["masses_au"],
        crystal["atom_types"],
    )
    if not 0 <= args.tracked_mode_index < dimensions["nmodes"]:
        raise ValueError("tracked mode index is invalid")
    reference_vector = reference_eigenvectors[:, args.tracked_mode_index]
    mass_by_component = np.repeat(
        crystal["masses_au"][crystal["atom_types"] - 1], 3
    )

    vertex_rows = []
    tracked_frequencies = []
    tracked_indices = []
    overlaps = []
    aligned_vectors = []
    for qpoint in qpoints:
        eigenvalues, eigenvectors = dynamical_modes(
            rdw,
            qpoint,
            vectors_q,
            degeneracies_q,
            crystal["masses_au"],
            crystal["atom_types"],
        )
        candidate_overlaps = np.abs(eigenvectors.conj().T @ reference_vector)
        mode_index = int(np.argmax(candidate_overlaps))
        vector = eigenvectors[:, mode_index]
        phase_overlap = np.vdot(reference_vector, vector)
        vector *= np.exp(-1j * np.angle(phase_overlap))
        normalized_overlap = abs(np.vdot(reference_vector, vector)) / (
            np.linalg.norm(reference_vector) * np.linalg.norm(vector)
        )
        physical_mode = vector / np.sqrt(mass_by_component)
        coefficients = mode_projected_coefficients(
            epmatwp,
            qpoint,
            vectors_q,
            degeneracies_q,
            physical_mode,
            degeneracies_k,
        )
        vertex_rows.append(coefficients.reshape(-1))
        eigenvalue = float(eigenvalues[mode_index])
        tracked_frequencies.append(
            np.sign(eigenvalue) * np.sqrt(abs(eigenvalue)) * RYDBERG_CM1
        )
        tracked_indices.append(mode_index)
        overlaps.append(normalized_overlap)
        aligned_vectors.append(vector)
    vertex = np.asarray(vertex_rows)
    row_norm = np.linalg.norm(vertex, axis=1)
    adjacent_overlaps = [
        float(
            abs(np.vdot(first, second))
            / (np.linalg.norm(first) * np.linalg.norm(second))
        )
        for first, second in zip(aligned_vectors[:-1], aligned_vectors[1:])
    ]

    # These five locations are fixed before inspecting the test-material vertex.
    training_indices = np.asarray([0, 10, 20, 30, 40], int)
    heldout_mask = np.ones(args.number_of_qpoints, bool)
    heldout_mask[training_indices] = False
    train_left, train_singular, train_right = np.linalg.svd(
        vertex[training_indices], full_matrices=False
    )
    training_latent = (
        train_left[:, : args.rank] * train_singular[None, : args.rank]
    )
    if args.interpolation == "natural_cubic":
        predicted_latent = interpolate_from_five_labels(
            coordinate, training_indices, training_latent
        )
    else:
        predicted_latent = interpolate_chebyshev_from_five_labels(
            coordinate, training_indices, training_latent
        )
    blind_prediction = predicted_latent @ train_right[: args.rank]
    blind_relative_error = np.linalg.norm(blind_prediction - vertex, axis=1) / row_norm

    full_left, full_singular, full_right = np.linalg.svd(
        vertex, full_matrices=False
    )
    full_cumulative = np.cumsum(full_singular**2) / np.sum(full_singular**2)
    full_selected_rank = (
        full_left[:, : args.rank] * full_singular[None, : args.rank]
    ) @ full_right[: args.rank]
    direct_relative_error = (
        np.linalg.norm(full_selected_rank - vertex, axis=1) / row_norm
    )

    rank_thresholds = {
        str(threshold): int(np.searchsorted(full_cumulative, threshold) + 1)
        for threshold in (0.99, 0.999, 0.9999, 0.99999)
    }
    checks = {
        "tracked_adjacent_branch_overlap_gt_0p99": bool(
            min(adjacent_overlaps) > 0.99
        ),
        "selected_rank_direct_capture_gt_0p9999": bool(
            float(full_cumulative[args.rank - 1]) > 0.9999
        ),
        "blind_five_q_heldout_RMSE_lt_0p02": bool(
            float(np.sqrt(np.mean(blind_relative_error[heldout_mask] ** 2)))
            < 0.02
        ),
        "blind_five_q_heldout_max_lt_0p05": bool(
            float(np.max(blind_relative_error[heldout_mask])) < 0.05
        ),
    }
    status = (
        "cross_material_low_rank_five_q_vertex_gate_passed"
        if all(checks.values())
        else "cross_material_low_rank_five_q_vertex_gate_failed"
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "rank3_five_q_blind_arrays.npz",
        coordinate=coordinate,
        qpoints=qpoints,
        tracked_mode_index=np.asarray(tracked_indices),
        tracked_frequency_cm1=np.asarray(tracked_frequencies),
        adjacent_mode_overlaps=np.asarray(adjacent_overlaps),
        training_indices=training_indices,
        full_vertex=vertex,
        blind_rank3_vertex=blind_prediction,
        blind_relative_vertex_error=blind_relative_error,
        full_singular_values=full_singular,
        full_cumulative_squared_singular_value=full_cumulative,
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    axes[0].plot(
        coordinate,
        tracked_frequencies,
        "-",
        color="#2676b8",
        lw=1.7,
    )
    axes[0].plot(
        coordinate[training_indices],
        np.asarray(tracked_frequencies)[training_indices],
        "o",
        color="black",
        ms=5,
        label="five training q points",
    )
    axes[0].set_xlabel("displacement along (1,1,0) from q₀")
    axes[0].set_ylabel(r"tracked mode frequency (cm$^{-1}$)")
    axes[0].set_title(f"{args.material} branch tracking")
    axes[0].grid(alpha=0.18)

    axes[1].semilogy(
        coordinate,
        blind_relative_error,
        "-",
        color="#d95f02",
        lw=1.7,
        label="held-out vertex error",
    )
    axes[1].plot(
        coordinate[training_indices],
        blind_relative_error[training_indices],
        "o",
        color="black",
        ms=5,
        label="training q points",
    )
    axes[1].set_xlabel("displacement along (1,1,0) from q₀")
    axes[1].set_ylabel("relative EPC vertex error")
    axes[1].set_title(f"Blind rank-{args.rank} reconstruction")
    axes[1].grid(alpha=0.18, which="both")
    handles, labels = axes[1].get_legend_handles_labels()
    handles0, labels0 = axes[0].get_legend_handles_labels()
    figure.legend(
        handles0 + handles,
        labels0 + labels,
        loc="center left",
        bbox_to_anchor=(0.81, 0.5),
        frameon=False,
    )
    figure.suptitle("Cross-material EPC compression: fixed five-q test")
    figure.subplots_adjust(left=0.09, right=0.79, bottom=0.15, top=0.86, wspace=0.30)
    figure.savefig(
        output / "rank3_five_q_blind.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "status": status,
        "material": args.material,
        "scope": (
            f"five fixed coarse-q labels train a material-specific complex rank-{args.rank} "
            f"mode-projected EPC basis and {args.interpolation} latent curve; remaining 36 q points "
            "are held out; existing EPW asset only and no new DFT"
        ),
        "asset_dimensions": dimensions,
        "q_center": list(args.q_center),
        "q_direction": list(args.q_direction),
        "q_halfwidth": args.halfwidth,
        "number_of_qpoints": args.number_of_qpoints,
        "training_indices": training_indices.tolist(),
        "selected_rank": args.rank,
        "latent_interpolation": args.interpolation,
        "tracked_reference_mode_index": args.tracked_mode_index,
        "tracked_mode_index_range": [min(tracked_indices), max(tracked_indices)],
        "tracked_frequency_range_cm-1": [
            float(min(tracked_frequencies)),
            float(max(tracked_frequencies)),
        ],
        "minimum_branch_overlap_with_center": float(min(overlaps)),
        "minimum_adjacent_branch_overlap": float(min(adjacent_overlaps)),
        "full_data_rank_for_capture": rank_thresholds,
        "full_data_selected_rank_capture": float(full_cumulative[args.rank - 1]),
        "full_data_selected_rank_maximum_relative_vertex_error": float(
            np.max(direct_relative_error)
        ),
        "blind_five_q_selected_rank": {
            "heldout_RMSE_relative_vertex_error": float(
                np.sqrt(np.mean(blind_relative_error[heldout_mask] ** 2))
            ),
            "heldout_median_relative_vertex_error": float(
                np.median(blind_relative_error[heldout_mask])
            ),
            "heldout_maximum_relative_vertex_error": float(
                np.max(blind_relative_error[heldout_mask])
            ),
        },
        "checks": checks,
        "limitations": [
            "this first cross-material gate evaluates vertex reconstruction, not the propagated phonon self-energy",
            f"the material-specific rank-{args.rank} basis is learned from five {args.material} EPC labels rather than generated from a pretrained cross-material model",
            f"only the branch selected as mode index {args.tracked_mode_index} at q0={list(args.q_center)} is tested",
        ],
    }
    (output / "rank3_five_q_blind_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
