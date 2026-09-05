#!/usr/bin/env python3
"""Zero-smearing full-EPC response with a refined K-valley patch.

The Brillouin-zone integral is evaluated on a coarse periodic mesh and the
square patch surrounding the singular initial-state K valley is replaced by a
much finer midpoint or triangle-centroid quadrature.  This resolves the small
d=0.003 cusp without increasing the DFT/Wannier data volume.
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


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ablate_graphene_constant_vertex_bandsum import (  # noqa: E402
    RYDBERG_EV,
    fermi,
    hamiltonian_grid,
)
from diagnose_graphene_epw_wannier import read_hr  # noqa: E402
from evaluate_graphene_epc_zero_transfer import (  # noqa: E402
    qpoint,
    read_zero_rows,
    shape_metrics,
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


def response_integrand(
    initial_energy_eV: np.ndarray,
    initial_vectors: np.ndarray,
    final_energy_eV: np.ndarray,
    final_vectors: np.ndarray,
    vertex_wannier: np.ndarray,
    target_fermi_eV: float,
    reference_fermi_eV: float,
    reference_degauss_Ry: float,
    eta_eV: float,
) -> np.ndarray:
    vertex_band = np.einsum(
        "...am,...ab,...bn->...mn",
        final_vectors.conj(),
        vertex_wannier,
        initial_vectors,
        optimize=True,
    )
    initial = initial_energy_eV[..., None, :]
    final = final_energy_eV[..., :, None]
    target_fact = fermi(final, target_fermi_eV, 0.0) - fermi(
        initial, target_fermi_eV, 0.0
    )
    reference_width_eV = reference_degauss_Ry * RYDBERG_EV
    reference_fact = fermi(
        final, reference_fermi_eV, reference_width_eV
    ) - fermi(initial, reference_fermi_eV, reference_width_eV)
    delta_Ry = (final - initial) / RYDBERG_EV
    denominator = np.real(1.0 / (delta_Ry + 1j * eta_eV / RYDBERG_EV))
    return np.sum(
        np.abs(vertex_band) ** 2
        * (target_fact - reference_fact)
        * denominator,
        axis=(-2, -1),
    )


def mode_projected_coefficients(
    epmatwp: np.ndarray,
    q: np.ndarray,
    vectors_q: np.ndarray,
    degeneracies_q: np.ndarray,
    physical_mode: np.ndarray,
    degeneracies_k: np.ndarray,
) -> np.ndarray:
    q_phase = np.exp(2j * np.pi * (vectors_q @ q)) / degeneracies_q
    cartesian = np.einsum("abrcp,p->abrc", epmatwp, q_phase, optimize=True)
    mode_projected = np.einsum(
        "abrc,c->abr", cartesian, physical_mode, optimize=True
    )
    return mode_projected / degeneracies_k[None, None, :]


def fft_from_realspace(
    vectors: np.ndarray, coefficients_by_R: np.ndarray, nk: int
) -> np.ndarray:
    coefficients = np.zeros((nk, nk, 2, 2), complex)
    for ir, translation in enumerate(vectors):
        coefficients[translation[0] % nk, translation[1] % nk] += coefficients_by_R[
            :, :, ir
        ]
    return np.fft.ifft2(coefficients, axes=(0, 1)) * nk * nk


def dense_realspace_array(
    vectors: np.ndarray, coefficients_by_R: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1 = np.arange(int(np.min(vectors[:, 0])), int(np.max(vectors[:, 0])) + 1)
    r2 = np.arange(int(np.min(vectors[:, 1])), int(np.max(vectors[:, 1])) + 1)
    dense = np.zeros((len(r1), len(r2), 2, 2), complex)
    for ir, translation in enumerate(vectors):
        dense[translation[0] - r1[0], translation[1] - r2[0]] += coefficients_by_R[
            :, :, ir
        ]
    return r1, r2, dense


def interpolate_square(
    r1: np.ndarray,
    r2: np.ndarray,
    dense_coefficients: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    phase1 = np.exp(2j * np.pi * np.outer(coordinates, r1))
    phase2 = np.exp(2j * np.pi * np.outer(coordinates, r2))
    first = np.einsum("ir,rsab->isab", phase1, dense_coefficients, optimize=True)
    return np.einsum("isab,js->ijab", first, phase2, optimize=True)


def interpolate_rectangle(
    r1: np.ndarray,
    r2: np.ndarray,
    dense_coefficients: np.ndarray,
    coordinates1: np.ndarray,
    coordinates2: np.ndarray,
) -> np.ndarray:
    phase1 = np.exp(2j * np.pi * np.outer(coordinates1, r1))
    phase2 = np.exp(2j * np.pi * np.outer(coordinates2, r2))
    first = np.einsum("ir,rsab->isab", phase1, dense_coefficients, optimize=True)
    return np.einsum("isab,js->ijab", first, phase2, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-nk", type=int, default=288)
    parser.add_argument("--patch-n", type=int, default=360)
    parser.add_argument("--patch-halfwidth", type=float, default=0.04)
    parser.add_argument(
        "--quadrature",
        choices=("midpoint", "triangle_centroid"),
        default="midpoint",
    )
    parser.add_argument("--target-fermi-eV", type=float, default=-1.7005)
    parser.add_argument("--reference-fermi-eV", type=float, default=-1.7187)
    parser.add_argument("--reference-degauss-Ry", type=float, default=0.020000)
    parser.add_argument("--eta-eV", type=float, default=0.005)
    parser.add_argument(
        "--hr",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility/E0_epw_matched"
            / "wannier_diagnostics/matched_k18q9_ex1_pifroz_hr.dat"
        ),
    )
    parser.add_argument(
        "--epwdata",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility/E0_epw_matched"
            / "k18_q9_ex1_pifroz/restart_small/epwdata.fmt"
        ),
    )
    parser.add_argument(
        "--crystal",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility/E0_epw_matched"
            / "k18_q9_ex1_pifroz/restart_small/crystal.fmt"
        ),
    )
    parser.add_argument(
        "--epmatwp",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E9_epc_bandsum_assets/graphene.epmatwp"
        ),
    )
    parser.add_argument(
        "--zero-data",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E4_joint_zero_finite_rank1/joint_zero_finite_rank1_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E11_epc_zero_adaptive"
        ),
    )
    args = parser.parse_args()
    if args.coarse_nk % 3 != 0:
        raise ValueError("coarse_nk must be divisible by 3")
    if not (0.0 < args.patch_halfwidth < 0.15):
        raise ValueError("patch_halfwidth is outside the supported range")

    rows = read_zero_rows(args.zero_data)
    qpoints = np.asarray(
        [qpoint(row["direction"], float(row["distance"])) for row in rows]
    )
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
    if args.quadrature == "midpoint":
        patch_coordinates = patch_lower + 0.5 * patch_step
        quadrature_grids = [(patch_coordinates, patch_coordinates, 1.0)]
    else:
        # Split every square cell along its lower-left to upper-right diagonal.
        # Each triangle is represented by its centroid and has half-cell area.
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
        patch_initial_h = interpolate_rectangle(
            r1_h, r2_h, dense_h, coordinates1, coordinates2
        )
        patch_initial_h = 0.5 * (
            patch_initial_h + np.swapaxes(patch_initial_h.conj(), -1, -2)
        )
        patch_initial_energy, patch_initial_vectors = np.linalg.eigh(
            patch_initial_h
        )
        patch_initial_states.append(
            (patch_initial_energy, patch_initial_vectors, weight)
        )

    corrected_Ry2 = np.empty(len(rows), float)
    coarse_Ry2 = np.empty(len(rows), float)
    coarse_patch_Ry2 = np.empty(len(rows), float)
    fine_patch_Ry2 = np.empty(len(rows), float)
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
        coarse_Ry2[index] = 2.0 * float(np.mean(coarse_integrand))
        coarse_patch_Ry2[index] = (
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
            raise RuntimeError("local Hamiltonian and EPC real-space grids differ")
        patch_integrand_mean = 0.0
        for grid_index, (coordinates1, coordinates2, weight) in enumerate(
            quadrature_grids
        ):
            patch_final_h = interpolate_rectangle(
                r1_q, r2_q, dense_hq, coordinates1, coordinates2
            )
            patch_final_h = 0.5 * (
                patch_final_h + np.swapaxes(patch_final_h.conj(), -1, -2)
            )
            patch_final_energy, patch_final_vectors = np.linalg.eigh(
                patch_final_h
            )
            patch_vertex = interpolate_rectangle(
                r1_g, r2_g, dense_g, coordinates1, coordinates2
            )
            patch_initial_energy, patch_initial_vectors, initial_weight = (
                patch_initial_states[grid_index]
            )
            if initial_weight != weight:
                raise RuntimeError("quadrature weights are inconsistent")
            patch_integrand = response_integrand(
                patch_initial_energy,
                patch_initial_vectors,
                patch_final_energy,
                patch_final_vectors,
                patch_vertex,
                args.target_fermi_eV,
                args.reference_fermi_eV,
                args.reference_degauss_Ry,
                args.eta_eV,
            )
            patch_integrand_mean += weight * float(np.mean(patch_integrand))
        patch_area = (2.0 * args.patch_halfwidth) ** 2
        fine_patch_Ry2[index] = 2.0 * patch_area * patch_integrand_mean
        corrected_Ry2[index] = (
            coarse_Ry2[index]
            - coarse_patch_Ry2[index]
            + fine_patch_Ry2[index]
        )

    corrected_cm2 = corrected_Ry2 * RYDBERG_CM1**2
    coarse_cm2 = coarse_Ry2 * RYDBERG_CM1**2
    target_correction = np.asarray(
        [float(row["target_delta_lambda_cm-2"]) for row in rows]
    )
    background = np.asarray([float(row["background_cm-1"]) for row in rows])
    K_index = next(index for index, row in enumerate(rows) if row["direction"] == "K")
    anchored_correction = (
        target_correction[K_index]
        + corrected_cm2
        - corrected_cm2[K_index]
    )
    coarse_anchored_correction = (
        target_correction[K_index] + coarse_cm2 - coarse_cm2[K_index]
    )
    prediction = signed_frequency(background, anchored_correction)
    coarse_prediction = signed_frequency(background, coarse_anchored_correction)
    metrics = shape_metrics(rows, prediction)
    coarse_metrics = shape_metrics(rows, coarse_prediction)
    passed = (
        metrics["maximum_d003_cusp_depth_relative_error"] < 0.20
        and metrics["maximum_slope_ratio_relative_error"] < 0.20
    )
    status = (
        "adaptive_zero_shape_gate_passed_requires_patch_convergence"
        if passed
        else "adaptive_zero_shape_gate_not_yet_passed"
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "epc_zero_adaptive_arrays.npz",
        qpoints_crystal=qpoints,
        coarse_response_Ry2=coarse_Ry2,
        coarse_patch_response_Ry2=coarse_patch_Ry2,
        fine_patch_response_Ry2=fine_patch_Ry2,
        corrected_response_Ry2=corrected_Ry2,
        target_correction_cm2=target_correction,
        prediction_cm1=prediction,
        coarse_prediction_cm1=coarse_prediction,
    )

    figure, axis = plt.subplots(figsize=(7.1, 4.3))
    axis.scatter(
        [0.0],
        [float(rows[K_index]["target_total_cm-1"])],
        color="black",
        s=28,
        zorder=5,
    )
    colors = {"KG": "#2676b8", "KM": "#7a3db8"}
    for direction in ("KG", "KM"):
        indices = [i for i, row in enumerate(rows) if row["direction"] == direction]
        distance = np.asarray([float(rows[i]["distance"]) for i in indices])
        target_frequency = np.asarray(
            [float(rows[i]["target_total_cm-1"]) for i in indices]
        )
        axis.plot(
            distance,
            target_frequency,
            "o",
            color=colors[direction],
            ms=4,
            label=f"DFPT {direction}",
        )
        axis.plot(
            distance,
            prediction[indices],
            "-",
            color=colors[direction],
            lw=1.8,
            label=f"adaptive EPC {direction}",
        )
    axis.set_xlabel("distance from K")
    axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    axis.set_title(
        "Zero-smearing EPC: refined K-valley "
        + args.quadrature.replace("_", " ")
    )
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.79, 0.5),
        frameon=False,
    )
    figure.subplots_adjust(left=0.12, right=0.77, bottom=0.14, top=0.89)
    figure.savefig(
        output / "epc_zero_adaptive_comparison.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "status": status,
        "scope": (
            "full stored EPC response with coarse full-BZ integral and a refined "
            "K-valley replacement; separate K anchor; no new DFT or zero-shape fit"
        ),
        "coarse_nk": args.coarse_nk,
        "patch_n": args.patch_n,
        "patch_halfwidth_fractional": args.patch_halfwidth,
        "patch_step_fractional": patch_step,
        "patch_quadrature": args.quadrature,
        "d003_shift_in_patch_steps": 0.003 / 3.0 / patch_step,
        "coarse_uniform_metrics": coarse_metrics,
        "adaptive_metrics": metrics,
        "checks": {
            "adaptive_max_d003_cusp_depth_relative_error_lt_0p20": metrics[
                "maximum_d003_cusp_depth_relative_error"
            ]
            < 0.20,
            "adaptive_max_slope_ratio_relative_error_lt_0p20": metrics[
                "maximum_slope_ratio_relative_error"
            ]
            < 0.20,
        },
        "limitations": [
            (
                "the refined patch uses a triangle-centroid convergence check, "
                "not an analytic linear-tetrahedron treatment of the Fermi boundary"
                if args.quadrature == "triangle_centroid"
                else "the refined patch uses midpoint quadrature rather than an analytic 2D triangle rule"
            ),
            "patch size and patch resolution must both be varied before freezing the zero-smearing result",
            "the zero reference outer shape is k192 while K and d=0.003 use k288",
        ],
    }
    (output / "epc_zero_adaptive_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
