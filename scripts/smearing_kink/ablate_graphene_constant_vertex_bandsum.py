#!/usr/bin/env python3
"""Test whether Wannier bands plus a constant A' vertex explain the K shape.

This is a deliberately restricted ablation, not a production EPC model.  It
evaluates a static intervalley band sum for four constant Hermitian vertices in
the two-Wannier-orbital basis, then compares the centered q-shape with existing
finite-smearing EPW curves.  No DFT or EPW calculation is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_graphene_epw_scalar_low_rank import (  # noqa: E402
    TEMPERATURES,
    signed_frequency,
    symmetrized_curves,
)
from diagnose_graphene_epw_wannier import read_hr  # noqa: E402


RYDBERG_EV = 13.605693122994


VERTICES = {
    "identity": np.asarray([[1.0, 0.0], [0.0, 1.0]], complex),
    "sigma_x": np.asarray([[0.0, 1.0], [1.0, 0.0]], complex),
    "sigma_y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], complex),
    "sigma_z": np.asarray([[1.0, 0.0], [0.0, -1.0]], complex),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def hamiltonian_grid(
    translations: np.ndarray,
    matrices: np.ndarray,
    nk: int,
    shift: np.ndarray,
) -> np.ndarray:
    coefficients = np.zeros((nk, nk, 2, 2), complex)
    phase = np.exp(2j * np.pi * (translations @ shift))
    for ir, translation in enumerate(translations):
        coefficients[translation[0] % nk, translation[1] % nk] += (
            matrices[ir] * phase[ir]
        )
    hamiltonian = np.fft.ifft2(coefficients, axes=(0, 1)) * nk * nk
    return 0.5 * (hamiltonian + np.swapaxes(hamiltonian.conj(), -1, -2))


def fermi(energy: np.ndarray, mu: float, sigma_eV: float) -> np.ndarray:
    if sigma_eV == 0.0:
        result = np.zeros_like(energy, float)
        result[energy < mu] = 1.0
        result[np.isclose(energy, mu, atol=1.0e-12, rtol=0.0)] = 0.5
        return result
    argument = np.clip((energy - mu) / sigma_eV, -60.0, 60.0)
    return 1.0 / (np.exp(argument) + 1.0)


def response_for_q(
    initial_energy: np.ndarray,
    initial_vectors: np.ndarray,
    final_energy: np.ndarray,
    final_vectors: np.ndarray,
    vertices: dict[str, np.ndarray],
    mu_eV: float,
    sigma_eV: np.ndarray,
) -> dict[str, np.ndarray]:
    initial = initial_energy[..., None, :]
    final = final_energy[..., :, None]
    denominator = initial - final
    mean_energy = 0.5 * (initial + final)
    results = {}
    for name, vertex in vertices.items():
        matrix_element = np.einsum(
            "...am,ab,...bn->...mn",
            final_vectors.conj(),
            vertex,
            initial_vectors,
            optimize=True,
        )
        weight = np.abs(matrix_element) ** 2
        by_smearing = []
        for width in sigma_eV:
            f_initial = fermi(initial, mu_eV, float(width))
            f_final = fermi(final, mu_eV, float(width))
            numerator = f_initial - f_final
            quotient = np.empty_like(denominator, float)
            regular = np.abs(denominator) > 1.0e-10
            quotient[regular] = numerator[regular] / denominator[regular]
            if width == 0.0:
                # The zero-temperature derivative is a delta function.  Its
                # contribution on a uniform mesh is handled by convergence in
                # nk; exact degeneracy points have zero measure.
                quotient[~regular] = 0.0
            else:
                f_mean = fermi(mean_energy, mu_eV, float(width))
                quotient[~regular] = (
                    -f_mean[~regular] * (1.0 - f_mean[~regular]) / float(width)
                )
            by_smearing.append(float(2.0 * np.mean(np.sum(weight * quotient, axis=(-2, -1)))))
        results[name] = np.asarray(by_smearing, float)
    return results


def centered_band_sums(
    translations: np.ndarray,
    matrices: np.ndarray,
    nk: int,
    distances: np.ndarray,
    smearings_Ry: np.ndarray,
    mu_eV: float,
) -> dict[str, np.ndarray]:
    initial_h = hamiltonian_grid(
        translations, matrices, nk, np.zeros(3, float)
    )
    initial_energy, initial_vectors = np.linalg.eigh(initial_h)
    sigma_eV = smearings_Ry * RYDBERG_EV
    responses = {
        name: np.empty((len(smearings_Ry), len(distances)), float)
        for name in VERTICES
    }
    for iq, distance in enumerate(distances):
        signs = (0.0,) if np.isclose(distance, 0.0) else (-1.0, 1.0)
        signed_results = {name: [] for name in VERTICES}
        for sign in signs:
            q = np.asarray(
                [
                    (1.0 + sign * distance) / 3.0,
                    (1.0 + sign * distance) / 3.0,
                    0.0,
                ]
            )
            final_h = hamiltonian_grid(translations, matrices, nk, q)
            final_energy, final_vectors = np.linalg.eigh(final_h)
            current = response_for_q(
                initial_energy,
                initial_vectors,
                final_energy,
                final_vectors,
                VERTICES,
                mu_eV,
                sigma_eV,
            )
            for name in VERTICES:
                signed_results[name].append(current[name])
        for name in VERTICES:
            responses[name][:, iq] = np.mean(signed_results[name], axis=0)
    return {
        name: response - response[:, [0]] for name, response in responses.items()
    }


def fit_and_score(
    response: np.ndarray,
    target_correction: np.ndarray,
    background: np.ndarray,
) -> dict:
    target_centered = target_correction - target_correction[:, [0]]
    global_scale = float(
        np.sum(response * target_centered) / np.sum(response * response)
    )
    separate_scale = np.sum(response * target_centered, axis=1) / np.sum(
        response * response, axis=1
    )

    def metrics(predicted_centered: np.ndarray) -> dict:
        predicted_correction = target_correction[:, [0]] + predicted_centered
        target_frequency = signed_frequency(background, target_correction)
        predicted_frequency = signed_frequency(background, predicted_correction)
        by_temperature = {}
        for index, temperature in enumerate(TEMPERATURES):
            target_depth = float(target_frequency[index, 2] - target_frequency[index, 0])
            predicted_depth = float(
                predicted_frequency[index, 2] - predicted_frequency[index, 0]
            )
            by_temperature[str(temperature)] = {
                "frequency_RMSE_cm-1": float(
                    np.sqrt(
                        np.mean(
                            (predicted_frequency[index] - target_frequency[index]) ** 2
                        )
                    )
                ),
                "frequency_max_abs_cm-1": float(
                    np.max(
                        np.abs(predicted_frequency[index] - target_frequency[index])
                    )
                ),
                "cusp_depth_relative_error": float(
                    abs(predicted_depth - target_depth) / abs(target_depth)
                ),
            }
        return {
            "frequency_RMSE_cm-1": float(
                np.sqrt(np.mean((predicted_frequency - target_frequency) ** 2))
            ),
            "frequency_max_abs_cm-1": float(
                np.max(np.abs(predicted_frequency - target_frequency))
            ),
            "maximum_cusp_depth_relative_error": max(
                row["cusp_depth_relative_error"] for row in by_temperature.values()
            ),
            "by_temperature": by_temperature,
        }

    global_prediction = global_scale * response
    separate_prediction = separate_scale[:, None] * response
    return {
        "global_scale_cm-2_per_response_unit": global_scale,
        "separate_scales_cm-2_per_response_unit": separate_scale.tolist(),
        "global_amplitude_metrics": metrics(global_prediction),
        "per_smearing_amplitude_shape_metrics": metrics(separate_prediction),
        "global_prediction_centered_cm-2": global_prediction,
        "separate_prediction_centered_cm-2": separate_prediction,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=192)
    parser.add_argument("--fermi-eV", type=float, default=-1.7187)
    parser.add_argument(
        "--hr",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E0_epw_matched/wannier_diagnostics/matched_k18q9_ex1_pifroz_hr.dat"
        ),
    )
    parser.add_argument(
        "--dense-data",
        type=Path,
        default=ROOT / "results/graphene_kohn_cusp_two_methods/B0_dense_k29/b0_dense_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E7_constant_vertex_bandsum"
        ),
    )
    args = parser.parse_args()
    if args.nk % 3 != 0:
        raise ValueError("nk must be divisible by 3 so K and K' lie on the mesh")

    data = symmetrized_curves(args.dense_data)
    translations, matrices = read_hr(args.hr)
    responses = centered_band_sums(
        translations,
        matrices,
        args.nk,
        data["distance"],
        data["degauss_Ry"],
        args.fermi_eV,
    )
    scores = {
        name: fit_and_score(
            response,
            data["correction"],
            data["background"],
        )
        for name, response in responses.items()
    }
    best_global = min(
        scores,
        key=lambda name: scores[name]["global_amplitude_metrics"][
            "frequency_RMSE_cm-1"
        ],
    )
    best_shape = min(
        scores,
        key=lambda name: scores[name]["per_smearing_amplitude_shape_metrics"][
            "frequency_RMSE_cm-1"
        ],
    )
    status = (
        "constant_vertex_rejected"
        if scores[best_shape]["per_smearing_amplitude_shape_metrics"][
            "maximum_cusp_depth_relative_error"
        ]
        > 0.15
        else "constant_vertex_not_rejected_on_opened_curves"
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    target_frequency = signed_frequency(data["background"], data["correction"])
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharey=True)
    colors = {"identity": "#777777", "sigma_x": "#2676b8", "sigma_y": "#d55e00", "sigma_z": "#7a3db8"}
    for index, temperature in enumerate(TEMPERATURES):
        axis = axes[index]
        axis.plot(
            data["distance"],
            target_frequency[index],
            color="black",
            lw=2.0,
            label="EPW target",
        )
        for name in VERTICES:
            prediction_centered = scores[name][
                "per_smearing_amplitude_shape_metrics"
            ]
            del prediction_centered
            predicted_correction = (
                data["correction"][index, 0]
                + scores[name]["separate_prediction_centered_cm-2"][index]
            )
            predicted_frequency = signed_frequency(
                data["background"][index], predicted_correction
            )
            axis.plot(
                data["distance"],
                predicted_frequency,
                color=colors[name],
                lw=1.15,
                label=name,
            )
        axis.set_title(f"degauss = {data['degauss_Ry'][index]:.7f} Ry")
        axis.set_xlabel("distance from K")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    figure.savefig(
        output / "constant_vertex_bandsum_comparison.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    serializable_scores = {}
    for name, score in scores.items():
        serializable_scores[name] = {
            key: value
            for key, value in score.items()
            if not isinstance(value, np.ndarray)
        }
    np.savez(
        output / "constant_vertex_bandsum_arrays.npz",
        distances_from_K=data["distance"],
        degauss_Ry=data["degauss_Ry"],
        target_correction_cm2=data["correction"],
        **{f"response_{name}": value for name, value in responses.items()},
        **{
            f"prediction_centered_{name}_cm2": score[
                "separate_prediction_centered_cm-2"
            ]
            for name, score in scores.items()
        },
    )
    summary = {
        "status": status,
        "scope": (
            "static intervalley band sum with production Wannier bands and four "
            "constant two-orbital vertices; opened finite-smearing curves; no new DFT"
        ),
        "nk": args.nk,
        "fermi_eV": args.fermi_eV,
        "degauss_Ry": data["degauss_Ry"].tolist(),
        "best_global_vertex": best_global,
        "best_shape_vertex": best_shape,
        "vertices": serializable_scores,
        "decision_rule": (
            "reject the constant-vertex family when even the best vertex with a "
            "separate fitted amplitude at each smearing has >15% maximum cusp-depth error"
        ),
        "limitations": [
            "constant Pauli vertices are gauge-dependent ablations, not reconstructed EPW A-prime matrix elements",
            "the per-smearing amplitude fit gives this family more freedom than the intended transferable model",
            "opened EPW curves are used for model diagnosis and are not an independent holdout",
        ],
        "inputs": {
            "hr": {"path": str(args.hr), "sha256": sha256(args.hr)},
            "dense_data": {
                "path": str(args.dense_data),
                "sha256": sha256(args.dense_data),
            },
        },
    }
    atomic_json(output / "constant_vertex_bandsum_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
