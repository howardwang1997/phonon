#!/usr/bin/env python3
"""Propagate a five-q low-rank EPC model through the phonon self-energy."""
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

from diagnose_graphene_epw_wannier import read_hr  # noqa: E402
from extract_graphene_epw_dynamical_matrices import RYDBERG_CM1  # noqa: E402
from replay_graphene_epc_bandsum import (  # noqa: E402
    response_difference,
    signed_frequency,
)


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
ASSETS = BASE / "E18_cross_system_assets/1T-VSe2"


def hamiltonian_grid_generic(
    translations: np.ndarray,
    matrices: np.ndarray,
    nk: int,
    shift: np.ndarray,
) -> np.ndarray:
    nbands = matrices.shape[1]
    coefficients = np.zeros((nk, nk, nbands, nbands), complex)
    phase = np.exp(2j * np.pi * (translations @ shift))
    for ir, translation in enumerate(translations):
        coefficients[translation[0] % nk, translation[1] % nk] += (
            matrices[ir] * phase[ir]
        )
    hamiltonian = np.fft.ifft2(coefficients, axes=(0, 1)) * nk * nk
    return 0.5 * (hamiltonian + np.swapaxes(hamiltonian.conj(), -1, -2))


def vertex_grid_generic(
    vectors: np.ndarray,
    coefficients_by_R: np.ndarray,
    nk: int,
) -> np.ndarray:
    nbands = coefficients_by_R.shape[0]
    coefficients = np.zeros((nk, nk, nbands, nbands), complex)
    for ir, translation in enumerate(vectors):
        coefficients[translation[0] % nk, translation[1] % nk] += (
            coefficients_by_R[:, :, ir]
        )
    return np.fft.ifft2(coefficients, axes=(0, 1)) * nk * nk


def blind_reconstruction(
    coordinate: np.ndarray,
    vertex: np.ndarray,
    training_indices: np.ndarray,
    rank: int,
) -> np.ndarray:
    left, singular, right = np.linalg.svd(
        vertex[training_indices], full_matrices=False
    )
    training_latent = left[:, :rank] * singular[None, :rank]
    predicted = np.empty((len(coordinate), rank), complex)
    for component in range(rank):
        predicted[:, component] = CubicSpline(
            coordinate[training_indices],
            training_latent[:, component].real,
            bc_type="natural",
        )(coordinate) + 1j * CubicSpline(
            coordinate[training_indices],
            training_latent[:, component].imag,
            bc_type="natural",
        )(coordinate)
    predicted[training_indices] = training_latent
    return predicted @ right[:rank]


def blind_reconstruction_chebyshev(
    coordinate: np.ndarray,
    vertex: np.ndarray,
    training_indices: np.ndarray,
    rank: int,
) -> np.ndarray:
    left, singular, right = np.linalg.svd(
        vertex[training_indices], full_matrices=False
    )
    training_latent = left[:, :rank] * singular[None, :rank]
    predicted = np.empty((len(coordinate), rank), complex)
    domain = [float(coordinate[0]), float(coordinate[-1])]
    for component in range(rank):
        predicted[:, component] = Chebyshev.fit(
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
    predicted[training_indices] = training_latent
    return predicted @ right[:rank]


def candidate_metrics(
    full_frequency: np.ndarray,
    candidate_frequency: np.ndarray,
    center: int,
) -> dict:
    by_smearing = []
    for index in range(full_frequency.shape[0]):
        full_depth = float(
            0.5 * (full_frequency[index, 0] + full_frequency[index, -1])
            - full_frequency[index, center]
        )
        candidate_depth = float(
            0.5
            * (candidate_frequency[index, 0] + candidate_frequency[index, -1])
            - candidate_frequency[index, center]
        )
        by_smearing.append(
            {
                "full_window_depth_cm-1": full_depth,
                "candidate_window_depth_cm-1": candidate_depth,
                "window_depth_relative_error": abs(candidate_depth - full_depth)
                / max(abs(full_depth), 1e-12),
                "frequency_RMSE_cm-1": float(
                    np.sqrt(
                        np.mean(
                            (candidate_frequency[index] - full_frequency[index])
                            ** 2
                        )
                    )
                ),
                "frequency_max_abs_cm-1": float(
                    np.max(
                        np.abs(
                            candidate_frequency[index] - full_frequency[index]
                        )
                    )
                ),
            }
        )
    return {
        "frequency_RMSE_cm-1": float(
            np.sqrt(np.mean((candidate_frequency - full_frequency) ** 2))
        ),
        "frequency_max_abs_cm-1": float(
            np.max(np.abs(candidate_frequency - full_frequency))
        ),
        "maximum_window_depth_relative_error": max(
            row["window_depth_relative_error"] for row in by_smearing
        ),
        "by_smearing": by_smearing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="1T-VSe2")
    parser.add_argument("--nk", type=int, default=24)
    parser.add_argument("--fermi-eV", type=float, default=-2.4296)
    parser.add_argument("--eta-eV", type=float, default=0.005)
    parser.add_argument("--reference-degauss-Ry", type=float, default=0.020000)
    parser.add_argument(
        "--evaluation-kind",
        choices=("post_failure_repair", "independent_blind_followup"),
        default="post_failure_repair",
    )
    parser.add_argument(
        "--smearing-degauss-Ry",
        type=float,
        nargs="+",
        default=(0.005, 0.010, 0.015),
    )
    parser.add_argument("--hr", type=Path, default=ASSETS / "1TVSe2_hr.dat")
    parser.add_argument(
        "--blind-arrays",
        type=Path,
        default=BASE
        / "E18_VSe2_rank3_five_q_blind/rank3_five_q_blind_arrays.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E20_VSe2_rank_response",
    )
    args = parser.parse_args()

    with np.load(args.blind_arrays, allow_pickle=False) as payload:
        coordinate = np.asarray(payload["coordinate"])
        qpoints = np.asarray(payload["qpoints"])
        bare_frequency = np.asarray(payload["tracked_frequency_cm1"])
        training_indices = np.asarray(payload["training_indices"], int)
        full_vertex_rows = np.asarray(payload["full_vertex"])
    center = len(coordinate) // 2
    rank3_rows = blind_reconstruction(
        coordinate, full_vertex_rows, training_indices, 3
    )
    rank4_rows = blind_reconstruction(
        coordinate, full_vertex_rows, training_indices, 4
    )
    rank4_chebyshev_rows = blind_reconstruction_chebyshev(
        coordinate, full_vertex_rows, training_indices, 4
    )

    translations, hamiltonian_matrices = read_hr(args.hr)
    nbands = hamiltonian_matrices.shape[1]
    nrr_k = len(translations)
    if full_vertex_rows.shape[1] != nbands * nbands * nrr_k:
        raise ValueError("VSe2 vertex and Hamiltonian dimensions differ")
    initial_h = hamiltonian_grid_generic(
        translations, hamiltonian_matrices, args.nk, np.zeros(3)
    )
    initial_energy, initial_vectors = np.linalg.eigh(initial_h)
    smearings = np.asarray(args.smearing_degauss_Ry, float)
    responses_Ry2 = {
        name: np.empty((len(smearings), len(qpoints)), float)
        for name in (
            "full",
            "blind_rank3",
            "blind_rank4",
            "blind_rank4_Chebyshev",
        )
    }
    rows_by_name = {
        "full": full_vertex_rows,
        "blind_rank3": rank3_rows,
        "blind_rank4": rank4_rows,
        "blind_rank4_Chebyshev": rank4_chebyshev_rows,
    }
    for iq, qpoint in enumerate(qpoints):
        final_h = hamiltonian_grid_generic(
            translations, hamiltonian_matrices, args.nk, qpoint
        )
        final_energy, final_vectors = np.linalg.eigh(final_h)
        for name, rows in rows_by_name.items():
            coefficients = rows[iq].reshape(nbands, nbands, nrr_k)
            vertex = vertex_grid_generic(translations, coefficients, args.nk)
            responses_Ry2[name][:, iq] = response_difference(
                initial_energy,
                initial_vectors,
                final_energy,
                final_vectors,
                vertex,
                smearings,
                args.reference_degauss_Ry,
                args.fermi_eV,
                args.eta_eV,
            )

    response_cm2 = {
        name: values * RYDBERG_CM1**2 for name, values in responses_Ry2.items()
    }
    frequency = {
        name: signed_frequency(bare_frequency[None, :], correction)
        for name, correction in response_cm2.items()
    }
    metrics = {
        name: candidate_metrics(frequency["full"], frequency[name], center)
        for name in (
            "blind_rank3",
            "blind_rank4",
            "blind_rank4_Chebyshev",
        )
    }
    for name in metrics:
        for index, smearing in enumerate(smearings):
            metrics[name]["by_smearing"][index]["smearing_degauss_Ry"] = float(
                smearing
            )
    original_checks = {
        "rank3_frequency_RMSE_lt_0p10_cm-1": bool(
            metrics["blind_rank3"]["frequency_RMSE_cm-1"] < 0.10
        ),
        "rank3_max_window_depth_error_lt_0p03": bool(
            metrics["blind_rank3"]["maximum_window_depth_relative_error"] < 0.03
        ),
        "rank4_frequency_RMSE_lt_0p05_cm-1": bool(
            metrics["blind_rank4"]["frequency_RMSE_cm-1"] < 0.05
        ),
        "rank4_max_window_depth_error_lt_0p02": bool(
            metrics["blind_rank4"]["maximum_window_depth_relative_error"] < 0.02
        ),
    }
    repair_checks = {
        "rank4_Chebyshev_frequency_RMSE_lt_0p05_cm-1": bool(
            metrics["blind_rank4_Chebyshev"]["frequency_RMSE_cm-1"] < 0.05
        ),
        "rank4_Chebyshev_frequency_max_lt_0p15_cm-1": bool(
            metrics["blind_rank4_Chebyshev"]["frequency_max_abs_cm-1"] < 0.15
        ),
        "rank4_Chebyshev_max_window_depth_error_lt_0p02": bool(
            metrics["blind_rank4_Chebyshev"][
                "maximum_window_depth_relative_error"
            ]
            < 0.02
        ),
    }
    status = (
        "low_rank_response_comparison_passed"
        if all(repair_checks.values())
        else "low_rank_response_comparison_failed"
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "rank_response_arrays.npz",
        coordinate=coordinate,
        qpoints=qpoints,
        bare_frequency_cm1=bare_frequency,
        smearing_degauss_Ry=smearings,
        **{f"{name}_response_cm2": value for name, value in response_cm2.items()},
        **{f"{name}_frequency_cm1": value for name, value in frequency.items()},
    )

    figure, axes = plt.subplots(1, len(smearings), figsize=(12.0, 3.8), sharey=True)
    if len(smearings) == 1:
        axes = [axes]
    for index, (axis, smearing) in enumerate(zip(axes, smearings)):
        axis.plot(
            coordinate,
            frequency["full"][index],
            color="black",
            lw=2.0,
            label="full EPC",
        )
        axis.plot(
            coordinate,
            frequency["blind_rank4"][index],
            "--",
            color="#d95f02",
            lw=1.7,
            label="rank 4 + cubic",
        )
        axis.plot(
            coordinate,
            frequency["blind_rank4_Chebyshev"][index],
            ":",
            color="#2676b8",
            lw=1.8,
            label="rank 4 + Chebyshev",
        )
        axis.set_title(f"smearing/degauss = {smearing:.3f} Ry")
        axis.set_xlabel("displacement from q₀")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(r"tracked-mode frequency (cm$^{-1}$)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        frameon=False,
    )
    figure.suptitle(f"{args.material}: full and five-q low-rank EPC responses")
    figure.subplots_adjust(left=0.07, right=0.82, bottom=0.15, top=0.84, wspace=0.10)
    figure.savefig(
        output / "rank_response.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    independent = args.evaluation_kind == "independent_blind_followup"
    summary = {
        "status": status,
        "material": args.material,
        "scope": (
            f"independent downstream response gate following the fixed five-q "
            f"{args.material} blind vertex test; full stored EPC is the reference; no new DFT"
            if independent
            else f"post-failure downstream response diagnostic using the same five "
            f"fixed {args.material} q labels; full stored EPC is the reference; no new DFT"
        ),
        "evaluation_kind": args.evaluation_kind,
        "nk": args.nk,
        "fermi_eV": args.fermi_eV,
        "reference_smearing_degauss_Ry": args.reference_degauss_Ry,
        "target_smearing_degauss_Ry": smearings.tolist(),
        "metrics_vs_full_EPC": metrics,
        "original_cubic_checks": original_checks,
        "deployment_checks": repair_checks,
        "interpretation": (
            "The deployable representation should use an adaptive rank capped at "
            "four and an analytic latent q basis; adding EPC labels is unnecessary "
            "if the five-point Chebyshev repair passes."
        ),
        "limitations": [
            "the production EPW fine mesh is only 24x24, so this compares compressed and full vertices at identical integration resolution rather than a converged material spectrum",
            f"the calculation validates the stored electronic self-energy response, not the separate short-range MLIP background for {args.material}",
            (
                "the response propagation uses the model and five labels already fixed for the independent vertex holdout; it does not add labels or alter the representation"
                if independent
                else "this is a failure-analysis run on the same material and is not a new blind test"
            ),
        ],
    }
    (output / "rank_response_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if status == "low_rank_response_comparison_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
