#!/usr/bin/env python3
"""Generate a smooth zero-smearing graphene K-A' curve with full EPC.

The short-range background is the frozen MLIP force-constant model.  The
non-local correction is the stored Wannier Hamiltonian plus full A' EPC
band-sum, integrated with a coarse full BZ and a triangle-centroid K-valley
patch.  No new DFT data or q-shape fit is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402
import fit_graphene_joint_zero_finite_rank1 as joint  # noqa: E402
from ablate_graphene_constant_vertex_bandsum import (  # noqa: E402
    hamiltonian_grid,
)
from diagnose_graphene_epw_wannier import read_hr  # noqa: E402
from evaluate_graphene_epc_zero_adaptive import (  # noqa: E402
    dense_realspace_array,
    fft_from_realspace,
    interpolate_rectangle,
    mode_projected_coefficients,
    response_integrand,
)
from evaluate_graphene_epc_zero_transfer import (  # noqa: E402
    qpoint,
    read_zero_rows,
)
from extract_graphene_epw_dynamical_matrices import (  # noqa: E402
    RYDBERG_CM1,
    load_crystal,
    load_rdw,
    zone_centered_wigner,
)
from replay_graphene_epc_bandsum import (  # noqa: E402
    dynamical_matrix_and_mode,
    load_epmatwp,
    signed_frequency,
)


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-nk", type=int, default=144)
    parser.add_argument("--patch-n", type=int, default=360)
    parser.add_argument("--patch-halfwidth", type=float, default=0.04)
    parser.add_argument("--eta-eV", type=float, default=0.00005)
    parser.add_argument("--maximum-distance", type=float, default=0.03)
    parser.add_argument("--points-per-direction", type=int, default=61)
    parser.add_argument("--target-fermi-eV", type=float, default=-1.7005)
    parser.add_argument("--reference-fermi-eV", type=float, default=-1.7187)
    parser.add_argument("--reference-degauss-Ry", type=float, default=0.020000)
    parser.add_argument(
        "--hr",
        type=Path,
        default=BASE
        / "E0_epw_matched/wannier_diagnostics/matched_k18q9_ex1_pifroz_hr.dat",
    )
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
        "--zero-data",
        type=Path,
        default=BASE / "E4_joint_zero_finite_rank1/joint_zero_finite_rank1_predictions.csv",
    )
    parser.add_argument(
        "--quadrature-summary",
        type=Path,
        default=BASE
        / "E14_zero_quadrature_convergence/zero_quadrature_convergence_summary.json",
    )
    parser.add_argument(
        "--static-short",
        type=Path,
        default=BASE
        / "source_p4_450/on_policy/frozen_prediction/frozen_static_fc2.npz",
    )
    parser.add_argument(
        "--operator-geometry",
        type=Path,
        default=a0.operator_path(450),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E15_epc_zero_dense_curve",
    )
    args = parser.parse_args()
    if args.coarse_nk % 3 != 0:
        raise ValueError("coarse_nk must be divisible by 3")
    if args.points_per_direction < 7:
        raise ValueError("points_per_direction is too small for a smooth curve")

    started = time.perf_counter()
    zero_rows = read_zero_rows(args.zero_data)
    K_row = next(row for row in zero_rows if row["direction"] == "K")
    K_frequency = float(K_row["target_total_cm-1"])
    quadrature_summary = json.loads(
        args.quadrature_summary.read_text(encoding="utf-8")
    )

    distances = np.linspace(
        0.0, args.maximum_distance, args.points_per_direction
    )
    records = [("K", 0.0)]
    records.extend(("KG", float(value)) for value in distances[1:])
    records.extend(("KM", float(value)) for value in distances[1:])
    qpoints = np.asarray([qpoint(direction, distance) for direction, distance in records])

    crystal = load_crystal(args.crystal, natoms=2)
    rdw, dimensions = load_rdw(args.epwdata)
    vectors_k, degeneracies_k, _ = zone_centered_wigner(
        crystal["at"], (18, 18, 1)
    )
    vectors_q, degeneracies_q, _ = zone_centered_wigner(
        crystal["at"], (9, 9, 1)
    )
    translations, hamiltonian_matrices = read_hr(args.hr)
    if not np.array_equal(translations, vectors_k):
        raise ValueError("Hamiltonian and EPC WS vector orders differ")
    epmatwp = load_epmatwp(
        args.epmatwp,
        dimensions["nbndsub"],
        dimensions["nrr_k"],
        dimensions["nmodes"],
        dimensions["nrr_g"],
    )

    coarse_initial_h = hamiltonian_grid(
        translations, hamiltonian_matrices, args.coarse_nk, np.zeros(3)
    )
    coarse_initial_energy, coarse_initial_vectors = np.linalg.eigh(
        coarse_initial_h
    )
    coarse_coordinates = np.arange(args.coarse_nk, dtype=float) / args.coarse_nk
    coarse_k1, coarse_k2 = np.meshgrid(
        coarse_coordinates, coarse_coordinates, indexing="ij"
    )
    K_value = 1.0 / 3.0
    coarse_patch_mask = (
        (np.abs(coarse_k1 - K_value) < args.patch_halfwidth)
        & (np.abs(coarse_k2 - K_value) < args.patch_halfwidth)
    )

    patch_step = 2.0 * args.patch_halfwidth / args.patch_n
    patch_lower = (
        K_value
        - args.patch_halfwidth
        + np.arange(args.patch_n, dtype=float) * patch_step
    )
    quadrature_grids = [
        (
            patch_lower + (2.0 / 3.0) * patch_step,
            patch_lower + (1.0 / 3.0) * patch_step,
            0.5,
        ),
        (
            patch_lower + (1.0 / 3.0) * patch_step,
            patch_lower + (2.0 / 3.0) * patch_step,
            0.5,
        ),
    ]
    r1_h, r2_h, dense_h = dense_realspace_array(
        translations, np.moveaxis(hamiltonian_matrices, 0, -1)
    )
    patch_initial_states = []
    for coordinates1, coordinates2, weight in quadrature_grids:
        initial_h = interpolate_rectangle(
            r1_h, r2_h, dense_h, coordinates1, coordinates2
        )
        initial_h = 0.5 * (
            initial_h + np.swapaxes(initial_h.conj(), -1, -2)
        )
        initial_energy, initial_vectors = np.linalg.eigh(initial_h)
        patch_initial_states.append((initial_energy, initial_vectors, weight))

    corrected_Ry2 = np.empty(len(qpoints), float)
    patch_area = (2.0 * args.patch_halfwidth) ** 2
    for index, q in enumerate(qpoints):
        _, _, physical_mode = dynamical_matrix_and_mode(
            rdw,
            q,
            vectors_q,
            degeneracies_q,
            crystal["masses_au"],
            crystal["atom_types"],
        )
        g_coefficients = mode_projected_coefficients(
            epmatwp,
            q,
            vectors_q,
            degeneracies_q,
            physical_mode,
            degeneracies_k,
        )
        coarse_vertex = fft_from_realspace(
            vectors_k, g_coefficients, args.coarse_nk
        )
        coarse_final_h = hamiltonian_grid(
            translations, hamiltonian_matrices, args.coarse_nk, q
        )
        coarse_final_energy, coarse_final_vectors = np.linalg.eigh(coarse_final_h)
        coarse_integrand = response_integrand(
            coarse_initial_energy,
            coarse_initial_vectors,
            coarse_final_energy,
            coarse_final_vectors,
            coarse_vertex,
            args.target_fermi_eV,
            args.reference_fermi_eV,
            args.reference_degauss_Ry,
            args.eta_eV,
        )
        coarse_response = 2.0 * float(np.mean(coarse_integrand))
        coarse_patch_response = (
            2.0
            * float(np.sum(coarse_integrand[coarse_patch_mask]))
            / args.coarse_nk**2
        )

        h_shift_phase = np.exp(2j * np.pi * (translations @ q))
        shifted_h_coefficients = np.moveaxis(
            hamiltonian_matrices * h_shift_phase[:, None, None], 0, -1
        )
        r1_g, r2_g, dense_g = dense_realspace_array(vectors_k, g_coefficients)
        r1_q, r2_q, dense_hq = dense_realspace_array(
            translations, shifted_h_coefficients
        )
        if not np.array_equal(r1_g, r1_q) or not np.array_equal(r2_g, r2_q):
            raise RuntimeError("local Hamiltonian and EPC grids differ")

        patch_integrand_mean = 0.0
        for grid_index, (coordinates1, coordinates2, weight) in enumerate(
            quadrature_grids
        ):
            final_h = interpolate_rectangle(
                r1_q, r2_q, dense_hq, coordinates1, coordinates2
            )
            final_h = 0.5 * (final_h + np.swapaxes(final_h.conj(), -1, -2))
            final_energy, final_vectors = np.linalg.eigh(final_h)
            vertex = interpolate_rectangle(
                r1_g, r2_g, dense_g, coordinates1, coordinates2
            )
            initial_energy, initial_vectors, initial_weight = (
                patch_initial_states[grid_index]
            )
            if initial_weight != weight:
                raise RuntimeError("quadrature weights are inconsistent")
            integrand = response_integrand(
                initial_energy,
                initial_vectors,
                final_energy,
                final_vectors,
                vertex,
                args.target_fermi_eV,
                args.reference_fermi_eV,
                args.reference_degauss_Ry,
                args.eta_eV,
            )
            patch_integrand_mean += weight * float(np.mean(integrand))
        fine_patch_response = 2.0 * patch_area * patch_integrand_mean
        corrected_Ry2[index] = (
            coarse_response - coarse_patch_response + fine_patch_response
        )

    phonon, force_constants = joint.make_static_short_phonon(
        args.operator_geometry, args.static_short
    )
    background = joint.tracked_static_background(
        phonon, force_constants, qpoints
    )
    response_cm2 = corrected_Ry2 * RYDBERG_CM1**2
    K_correction_cm2 = K_frequency**2 - background[0] ** 2
    correction_cm2 = K_correction_cm2 + response_cm2 - response_cm2[0]
    frequency = signed_frequency(background, correction_cm2)

    index_by_key = {
        (direction, round(distance, 12)): index
        for index, (direction, distance) in enumerate(records)
    }
    depth_results = {}
    for direction in ("KG", "KM"):
        d003_index = index_by_key[(direction, round(0.003, 12))]
        predicted_depth = float(frequency[d003_index] - frequency[0])
        frozen = quadrature_summary["directions"][direction]
        depth_results[direction] = {
            "dense_n360_depth_cm-1": predicted_depth,
            "frozen_resolution_extrapolated_depth_cm-1": float(
                frozen["recommended_depth_cm-1"]
            ),
            "frozen_numerical_uncertainty_cm-1": float(
                frozen["conservative_numerical_uncertainty_cm-1"]
            ),
            "DFPT_1_over_k_limit_depth_cm-1": float(
                frozen["DFPT_three_point_1_over_k_limit_cm-1"]
            ),
        }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "direction",
        "distance",
        "signed_distance",
        "q1",
        "q2",
        "short_range_MLIP_cm-1",
        "full_EPC_response_Ry2",
        "long_range_correction_cm-2",
        "MLIP_plus_full_EPC_cm-1",
    ]
    with (output / "zero_dense_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, ((direction, distance), q) in enumerate(zip(records, qpoints)):
            signed_distance = (
                0.0
                if direction == "K"
                else (-distance if direction == "KG" else distance)
            )
            writer.writerow(
                {
                    "direction": direction,
                    "distance": distance,
                    "signed_distance": signed_distance,
                    "q1": q[0],
                    "q2": q[1],
                    "short_range_MLIP_cm-1": background[index],
                    "full_EPC_response_Ry2": corrected_Ry2[index],
                    "long_range_correction_cm-2": correction_cm2[index],
                    "MLIP_plus_full_EPC_cm-1": frequency[index],
                }
            )

    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    colors = {"KG": "#2676b8", "KM": "#7a3db8"}
    for direction in ("KG", "KM"):
        indices = [
            0,
            *[
                index
                for index, (label, _) in enumerate(records)
                if label == direction
            ],
        ]
        x = np.asarray(
            [
                0.0
                if index == 0
                else (
                    -records[index][1]
                    if direction == "KG"
                    else records[index][1]
                )
                for index in indices
            ]
        )
        axis.plot(
            x,
            frequency[indices],
            "-",
            color=colors[direction],
            lw=2.0,
            label=f"MLIP + full EPC LR ({direction})",
        )

    raw_label_used = False
    for row in zero_rows:
        direction = row["direction"]
        distance = float(row["distance"])
        signed = 0.0 if direction == "K" else (-distance if direction == "KG" else distance)
        axis.plot(
            [signed],
            [float(row["target_total_cm-1"])],
            marker="o",
            ls="none",
            ms=4.0,
            markerfacecolor="none",
            markeredgecolor="#777777",
            alpha=0.75,
            label="available DFPT (not k-converged)" if not raw_label_used else None,
        )
        raw_label_used = True
    for direction, signed in (("KG", -0.003), ("KM", 0.003)):
        axis.errorbar(
            [signed],
            [
                K_frequency
                + depth_results[direction]["DFPT_1_over_k_limit_depth_cm-1"]
            ],
            fmt="s",
            ms=5,
            color="black",
            label="DFPT $1/N_k$ extrapolation" if direction == "KG" else None,
        )
    axis.plot([0.0], [K_frequency], "o", color="black", ms=5, label="K anchor")
    axis.axvline(0.0, color="#BBBBBB", lw=0.8, zorder=0)
    axis.set_xlabel("signed distance from K (KG < 0, KM > 0)")
    axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    axis.set_title("Graphene K-A′, smearing/degauss = 0 Ry")
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.78, 0.5),
        frameon=False,
    )
    figure.subplots_adjust(left=0.12, right=0.76, bottom=0.15, top=0.89)
    figure.savefig(
        output / "zero_dense_curve_comparison.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    elapsed = time.perf_counter() - started
    monotone_by_direction = {}
    maximum_second_difference_by_direction = {}
    for direction in ("KG", "KM"):
        direction_frequency = np.asarray(
            [frequency[0]]
            + [
                frequency[index]
                for index, (label, _) in enumerate(records)
                if label == direction
            ]
        )
        monotone_by_direction[direction] = bool(
            np.all(np.diff(direction_frequency) > 0.0)
        )
        maximum_second_difference_by_direction[direction] = float(
            np.max(np.abs(np.diff(direction_frequency, n=2)))
        )
    summary = {
        "status": "smooth_zero_full_EPC_curve_generated",
        "scope": (
            "frozen short-range MLIP plus explicit full-EPC long-range response; "
            "zero smearing/degauss; no empirical q-shape fit and no new DFT"
        ),
        "integration": {
            "coarse_nk": args.coarse_nk,
            "patch_quadrature": "triangle_centroid",
            "patch_n": args.patch_n,
            "patch_halfwidth_fractional": args.patch_halfwidth,
            "eta_eV": args.eta_eV,
            "number_of_qpoints": len(qpoints),
            "wall_time_seconds": elapsed,
        },
        "K_anchor_frequency_cm-1": K_frequency,
        "d003_depths": depth_results,
        "checks": {
            "dense_curve_contains_d003_exactly": any(
                np.isclose(distances, 0.003, atol=1e-14)
            ),
            "dense_n360_depth_inside_frozen_uncertainty": all(
                abs(
                    row["dense_n360_depth_cm-1"]
                    - row["frozen_resolution_extrapolated_depth_cm-1"]
                )
                <= row["frozen_numerical_uncertainty_cm-1"]
                for row in depth_results.values()
            ),
            "rank_one_scalar_update_is_Hermitian_by_construction": True,
            "frequency_increases_monotonically_away_from_K": monotone_by_direction,
            "maximum_discrete_second_difference_cm-1": (
                maximum_second_difference_by_direction
            ),
        },
        "limitations": [
            "the smooth outer curve has not yet been compared with k-converged zero-smearing DFPT away from d=0.003",
            "the K anchor is calibrated separately, as specified by the long-range model",
            "the displayed dense curve uses n=360; the d=0.003 reported reference uses the n=180/360/720 resolution extrapolation",
        ],
    }
    (output / "zero_dense_curve_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
