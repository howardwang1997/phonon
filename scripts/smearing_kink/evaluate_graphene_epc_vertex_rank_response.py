#!/usr/bin/env python3
"""Propagate low-rank A' EPC vertices through the finite-smearing band sum."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ablate_graphene_constant_vertex_bandsum import (  # noqa: E402
    hamiltonian_grid,
)
from diagnose_graphene_epw_wannier import read_hr  # noqa: E402
from evaluate_graphene_epc_zero_adaptive import (  # noqa: E402
    fft_from_realspace,
    mode_projected_coefficients,
)
from extract_graphene_epw_dynamical_matrices import (  # noqa: E402
    RYDBERG_CM1,
    load_crystal,
    load_rdw,
    zone_centered_wigner,
)
from replay_graphene_epc_bandsum import (  # noqa: E402
    TEMPERATURES,
    dense_targets,
    dynamical_matrix_and_mode,
    load_epmatwp,
    response_difference,
    signed_frequency,
)


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"


def latent_cubic_interpolation(
    coordinate: np.ndarray,
    latent: np.ndarray,
    training_indices: np.ndarray,
) -> np.ndarray:
    predicted = np.empty_like(latent)
    for component in range(latent.shape[1]):
        predicted[:, component] = CubicSpline(
            coordinate[training_indices],
            latent[training_indices, component].real,
            bc_type="natural",
        )(coordinate) + 1j * CubicSpline(
            coordinate[training_indices],
            latent[training_indices, component].imag,
            bc_type="natural",
        )(coordinate)
    predicted[training_indices] = latent[training_indices]
    return predicted


def interpolate_training_latent(
    coordinate: np.ndarray,
    training_indices: np.ndarray,
    training_latent: np.ndarray,
) -> np.ndarray:
    predicted = np.empty((len(coordinate), training_latent.shape[1]), complex)
    for component in range(training_latent.shape[1]):
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
    return predicted


def response_metrics(
    background: np.ndarray,
    target_anchor: np.ndarray,
    t: np.ndarray,
    reference_cm2: np.ndarray,
    candidate_cm2: np.ndarray,
) -> dict:
    center = int(np.argmin(np.abs(t - 1.0)))
    reference_anchored = (
        target_anchor[:, None]
        + reference_cm2
        - reference_cm2[:, [center]]
    )
    candidate_anchored = (
        target_anchor[:, None]
        + candidate_cm2
        - candidate_cm2[:, [center]]
    )
    reference_frequency = signed_frequency(background, reference_anchored)
    candidate_frequency = signed_frequency(background, candidate_anchored)
    by_smearing = {}
    for index, temperature in enumerate(TEMPERATURES):
        reference_depth = float(
            0.5
            * (
                reference_frequency[index, center - 2]
                + reference_frequency[index, center + 2]
            )
            - reference_frequency[index, center]
        )
        candidate_depth = float(
            0.5
            * (
                candidate_frequency[index, center - 2]
                + candidate_frequency[index, center + 2]
            )
            - candidate_frequency[index, center]
        )
        by_smearing[str(temperature)] = {
            "smearing_degauss_Ry": None,
            "reference_full_EPC_depth_cm-1": reference_depth,
            "candidate_depth_cm-1": candidate_depth,
            "cusp_depth_relative_error": abs(candidate_depth - reference_depth)
            / abs(reference_depth),
            "frequency_RMSE_cm-1": float(
                np.sqrt(
                    np.mean(
                        (
                            candidate_frequency[index]
                            - reference_frequency[index]
                        )
                        ** 2
                    )
                )
            ),
            "frequency_max_abs_cm-1": float(
                np.max(
                    np.abs(
                        candidate_frequency[index]
                        - reference_frequency[index]
                    )
                )
            ),
        }
    return {
        "frequency_RMSE_cm-1": float(
            np.sqrt(np.mean((candidate_frequency - reference_frequency) ** 2))
        ),
        "frequency_max_abs_cm-1": float(
            np.max(np.abs(candidate_frequency - reference_frequency))
        ),
        "maximum_cusp_depth_relative_error": max(
            row["cusp_depth_relative_error"] for row in by_smearing.values()
        ),
        "by_temperature_K": by_smearing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=720)
    parser.add_argument("--fermi-eV", type=float, default=-1.7187)
    parser.add_argument("--reference-degauss-Ry", type=float, default=0.020000)
    parser.add_argument("--eta-eV", type=float, default=0.005)
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
        "--dense-data",
        type=Path,
        default=ROOT
        / "results/graphene_kohn_cusp_two_methods/B0_dense_k29/b0_dense_predictions.csv",
    )
    parser.add_argument(
        "--full-replay",
        type=Path,
        default=BASE
        / "E9_epc_bandsum_replay_nk720/epc_bandsum_replay_arrays.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E17_epc_vertex_rank_response_nk720",
    )
    args = parser.parse_args()
    if args.nk % 3 != 0:
        raise ValueError("nk must be divisible by 3")

    started = time.perf_counter()
    target = dense_targets(args.dense_data)
    with np.load(args.full_replay, allow_pickle=False) as payload:
        if int(payload["nk"]) != args.nk:
            raise ValueError("stored full replay and requested nk differ")
        if not np.allclose(payload["qpoints_crystal"], target["qpoints"]):
            raise ValueError("stored full replay q grid differs")
        full_correction_cm2 = np.asarray(payload["replay_correction_cm2"])

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

    center = int(np.argmin(np.abs(target["t_GK"] - 1.0)))
    _, _, reference_mode = dynamical_matrix_and_mode(
        rdw,
        target["qpoints"][center],
        vectors_q,
        degeneracies_q,
        crystal["masses_au"],
        crystal["atom_types"],
    )
    rows = []
    for q in target["qpoints"]:
        _, _, mode = dynamical_matrix_and_mode(
            rdw,
            q,
            vectors_q,
            degeneracies_q,
            crystal["masses_au"],
            crystal["atom_types"],
        )
        overlap = np.vdot(reference_mode, mode)
        mode *= np.exp(-1j * np.angle(overlap))
        rows.append(
            mode_projected_coefficients(
                epmatwp,
                q,
                vectors_q,
                degeneracies_q,
                mode,
                degeneracies_k,
            ).reshape(-1)
        )
    full_vertex_rows = np.asarray(rows)
    left, singular, right = np.linalg.svd(full_vertex_rows, full_matrices=False)
    latent = left * singular[None, :]
    cumulative = np.cumsum(singular**2) / np.sum(singular**2)

    variants = {
        f"rank_{rank}": latent[:, :rank] @ right[:rank]
        for rank in (1, 2, 3, 4)
    }
    training_indices = np.asarray([0, 7, center, 21, 28], int)
    interpolated_rank3 = latent_cubic_interpolation(
        target["t_GK"], latent[:, :3], training_indices
    )
    variants["rank_3_five_q_labels"] = interpolated_rank3 @ right[:3]
    training_left, training_singular, training_right = np.linalg.svd(
        full_vertex_rows[training_indices], full_matrices=False
    )
    training_latent = training_left[:, :3] * training_singular[None, :3]
    blind_latent = interpolate_training_latent(
        target["t_GK"], training_indices, training_latent
    )
    variants["rank_3_five_q_blind_basis"] = blind_latent @ training_right[:3]

    initial_h = hamiltonian_grid(
        translations, hamiltonian_matrices, args.nk, np.zeros(3)
    )
    initial_energy, initial_vectors = np.linalg.eigh(initial_h)
    correction_by_variant = {
        name: np.empty((len(TEMPERATURES), len(target["qpoints"])), float)
        for name in variants
    }
    for iq, q in enumerate(target["qpoints"]):
        final_h = hamiltonian_grid(
            translations, hamiltonian_matrices, args.nk, q
        )
        final_energy, final_vectors = np.linalg.eigh(final_h)
        for name, approximated_rows in variants.items():
            coefficients = approximated_rows[iq].reshape(
                dimensions["nbndsub"],
                dimensions["nbndsub"],
                dimensions["nrr_k"],
            )
            vertex = fft_from_realspace(vectors_k, coefficients, args.nk)
            correction_by_variant[name][:, iq] = response_difference(
                initial_energy,
                initial_vectors,
                final_energy,
                final_vectors,
                vertex,
                target["degauss_Ry"],
                args.reference_degauss_Ry,
                args.fermi_eV,
                args.eta_eV,
            )

    metrics = {}
    for name, correction_Ry2 in correction_by_variant.items():
        candidate_cm2 = correction_Ry2 * RYDBERG_CM1**2
        candidate_metrics = response_metrics(
            target["background_cm1"],
            target["correction_cm2"][:, center],
            target["t_GK"],
            full_correction_cm2,
            candidate_cm2,
        )
        for index, temperature in enumerate(TEMPERATURES):
            candidate_metrics["by_temperature_K"][str(temperature)][
                "smearing_degauss_Ry"
            ] = float(target["degauss_Ry"][index])
        vertex_relative = (
            np.linalg.norm(variants[name] - full_vertex_rows, axis=1)
            / np.linalg.norm(full_vertex_rows, axis=1)
        )
        candidate_metrics["maximum_q_relative_vertex_error"] = float(
            np.max(vertex_relative)
        )
        candidate_metrics["vertex_RMSE_relative_error"] = float(
            np.sqrt(np.mean(vertex_relative**2))
        )
        metrics[name] = candidate_metrics

    selected = metrics["rank_3"]
    five_labels = metrics["rank_3_five_q_labels"]
    blind_five_labels = metrics["rank_3_five_q_blind_basis"]
    checks = {
        "rank3_frequency_RMSE_vs_full_lt_0p01_cm-1": selected[
            "frequency_RMSE_cm-1"
        ]
        < 0.01,
        "rank3_max_cusp_depth_error_vs_full_lt_0p005": selected[
            "maximum_cusp_depth_relative_error"
        ]
        < 0.005,
        "rank3_five_q_labels_frequency_RMSE_vs_full_lt_0p05_cm-1": five_labels[
            "frequency_RMSE_cm-1"
        ]
        < 0.05,
        "rank3_five_q_labels_max_cusp_error_vs_full_lt_0p02": five_labels[
            "maximum_cusp_depth_relative_error"
        ]
        < 0.02,
        "blind_rank3_five_q_frequency_RMSE_vs_full_lt_0p10_cm-1": blind_five_labels[
            "frequency_RMSE_cm-1"
        ]
        < 0.10,
        "blind_rank3_five_q_max_cusp_error_vs_full_lt_0p03": blind_five_labels[
            "maximum_cusp_depth_relative_error"
        ]
        < 0.03,
    }
    status = (
        "rank3_Aprime_vertex_preserves_finite_smearing_response"
        if all(checks.values())
        else "deployable_Aprime_vertex_rank_not_yet_frozen"
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "finite_vertex_rank_response_arrays.npz",
        t_GK=target["t_GK"],
        qpoints=target["qpoints"],
        degauss_Ry=target["degauss_Ry"],
        singular_values=singular,
        cumulative_squared_singular_value=cumulative,
        full_vertex_rows=full_vertex_rows,
        rank3_vertex_rows=variants["rank_3"],
        rank3_five_q_labels_vertex_rows=variants["rank_3_five_q_labels"],
        rank3_five_q_blind_basis_vertex_rows=variants[
            "rank_3_five_q_blind_basis"
        ],
        training_indices=training_indices,
        full_correction_cm2=full_correction_cm2,
        **{
            f"{name}_correction_cm2": value * RYDBERG_CM1**2
            for name, value in correction_by_variant.items()
        },
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    full_anchored = (
        target["correction_cm2"][:, [center]]
        + full_correction_cm2
        - full_correction_cm2[:, [center]]
    )
    rank3_cm2 = correction_by_variant["rank_3"] * RYDBERG_CM1**2
    rank3_anchored = (
        target["correction_cm2"][:, [center]]
        + rank3_cm2
        - rank3_cm2[:, [center]]
    )
    full_frequency = signed_frequency(target["background_cm1"], full_anchored)
    rank3_frequency = signed_frequency(target["background_cm1"], rank3_anchored)
    for index, temperature in enumerate(TEMPERATURES):
        axes[index].plot(
            target["t_GK"] - 1.0,
            full_frequency[index],
            color="black",
            lw=2.0,
            label="full EPC",
        )
        axes[index].plot(
            target["t_GK"] - 1.0,
            rank3_frequency[index],
            "--",
            color="#d95f02",
            lw=1.8,
            label="rank-3 EPC",
        )
        axes[index].set_title(
            f"smearing/degauss = {target['degauss_Ry'][index]:.7f} Ry"
        )
        axes[index].set_xlabel("signed distance from K")
        axes[index].grid(alpha=0.18)
    axes[0].set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        frameon=False,
    )
    figure.suptitle("Full versus rank-3 A′ EPC band-sum response")
    figure.subplots_adjust(left=0.07, right=0.82, bottom=0.15, top=0.84, wspace=0.10)
    figure.savefig(
        output / "finite_vertex_rank3_response.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "status": status,
        "scope": (
            "finite-smearing nk720 band-sum propagation of SVD-compressed A' "
            "vertices; stored Wannier/EPC only; no new DFT"
        ),
        "nk": args.nk,
        "number_of_qpoints": len(target["qpoints"]),
        "rank_capture": {
            str(rank): float(cumulative[rank - 1]) for rank in (1, 2, 3, 4)
        },
        "five_q_training_indices": training_indices.tolist(),
        "metrics_vs_full_EPC": metrics,
        "checks": checks,
        "wall_time_seconds": time.perf_counter() - started,
        "limitations": [
            "the rank basis is obtained from all 29 graphene q points",
            "the five-q blind-basis variant is a genuine q holdout within graphene, but it is not yet a cross-system blind test",
            "a transferable generator must predict the rank-3 basis and latent coefficients from coarse electronic/EPC descriptors",
        ],
    }
    (output / "finite_vertex_rank_response_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
