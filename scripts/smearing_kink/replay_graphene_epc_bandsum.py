#!/usr/bin/env python3
"""Replay graphene finite-smearing K response from the stored EPW EPC matrix.

The implementation follows the EPW 7.3.1 Wannier interpolation layout used to
write ``graphene.epmatwp``.  It reconstructs the A' phonon eigenvector, the
electron-phonon vertex on a uniform fine-k mesh, and the static response
difference Pi(T) - Pi(T_high).  No electronic-structure calculation is run.
"""
from __future__ import annotations

import argparse
import csv
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

from ablate_graphene_constant_vertex_bandsum import (  # noqa: E402
    RYDBERG_EV,
    fermi,
    hamiltonian_grid,
)
from diagnose_graphene_epw_wannier import read_hr  # noqa: E402
from extract_graphene_epw_dynamical_matrices import (  # noqa: E402
    BOHR_ANGSTROM,
    RYDBERG_CM1,
    load_crystal,
    load_rdw,
    zone_centered_wigner,
)


TEMPERATURES = (300, 450, 600)


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


def load_decay(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            rows.append([float(value) for value in stripped.split()])
    values = np.asarray(rows, float)
    return values[:, 0], values[:, 1]


def load_epmatwp(
    path: Path, nbnd: int, nrr_k: int, nmodes: int, nrr_g: int
) -> np.ndarray:
    expected = nbnd * nbnd * nrr_k * nmodes * nrr_g
    raw = np.fromfile(path, dtype="<c16")
    if raw.size != expected:
        raise ValueError(
            f"{path}: expected {expected} complex128 values, found {raw.size}"
        )
    return raw.reshape((nbnd, nbnd, nrr_k, nmodes, nrr_g), order="F")


def dense_targets(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["channel"] == "L0"]
    blocks = []
    for temperature in TEMPERATURES:
        block = sorted(
            [row for row in rows if int(row["temperature_K"]) == temperature],
            key=lambda row: float(row["t_GK"]),
        )
        if len(block) != 29:
            raise ValueError(f"expected 29 rows for {temperature} K")
        blocks.append(block)
    t = np.asarray([float(row["t_GK"]) for row in blocks[0]], float)
    if any(
        not np.allclose(t, [float(row["t_GK"]) for row in block])
        for block in blocks[1:]
    ):
        raise ValueError("temperature blocks use different q grids")
    return {
        "t_GK": t,
        "qpoints": np.column_stack((t / 3.0, t / 3.0, np.zeros_like(t))),
        "degauss_Ry": np.asarray(
            [float(block[0]["degauss_Ry"]) for block in blocks]
        ),
        "correction_cm2": np.asarray(
            [
                [float(row["dense_delta_lambda_top_cm-2"]) for row in block]
                for block in blocks
            ]
        ),
        "background_cm1": np.asarray(
            [
                [float(row["finite_lattice_short_background_cm-1"]) for row in block]
                for block in blocks
            ]
        ),
        "bare_top_cm1": np.asarray(
            [[float(row["EPW_bare_top_cm-1"]) for row in block] for block in blocks]
        ),
    }


def dynamical_matrix_and_mode(
    rdw: np.ndarray,
    qpoint: np.ndarray,
    vectors_q: np.ndarray,
    degeneracies_q: np.ndarray,
    masses: np.ndarray,
    atom_types: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
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
    mass_by_component = np.repeat(masses[atom_types - 1], 3)
    physical_mode = eigenvectors[:, -1] / np.sqrt(mass_by_component)
    frequency_cm1 = float(np.sqrt(max(eigenvalues[-1], 0.0)) * RYDBERG_CM1)
    return matrix, frequency_cm1, physical_mode


def epc_wannier_grid(
    epmatwp: np.ndarray,
    qpoint: np.ndarray,
    vectors_q: np.ndarray,
    degeneracies_q: np.ndarray,
    physical_mode: np.ndarray,
    vectors_k: np.ndarray,
    degeneracies_k: np.ndarray,
    nk: int,
) -> np.ndarray:
    q_phase = np.exp(2j * np.pi * (vectors_q @ qpoint)) / degeneracies_q
    cartesian = np.einsum("abrcp,p->abrc", epmatwp, q_phase, optimize=True)
    mode_projected = np.einsum(
        "abrc,c->abr", cartesian, physical_mode, optimize=True
    )
    coefficients = np.zeros((nk, nk, 2, 2), complex)
    for ir, translation in enumerate(vectors_k):
        coefficients[translation[0] % nk, translation[1] % nk] += (
            mode_projected[:, :, ir] / degeneracies_k[ir]
        )
    return np.fft.ifft2(coefficients, axes=(0, 1)) * nk * nk


def response_difference(
    initial_energy_eV: np.ndarray,
    initial_vectors: np.ndarray,
    final_energy_eV: np.ndarray,
    final_vectors: np.ndarray,
    vertex_wannier: np.ndarray,
    temperatures_Ry: np.ndarray,
    reference_Ry: float,
    fermi_eV: float,
    eta_eV: float,
    reference_fermi_eV: float | None = None,
) -> np.ndarray:
    vertex_band = np.einsum(
        "...am,...ab,...bn->...mn",
        final_vectors.conj(),
        vertex_wannier,
        initial_vectors,
        optimize=True,
    )
    g_squared = np.abs(vertex_band) ** 2
    initial = initial_energy_eV[..., None, :]
    final = final_energy_eV[..., :, None]
    delta_Ry = (final - initial) / RYDBERG_EV
    denominator = np.real(1.0 / (delta_Ry + 1j * eta_eV / RYDBERG_EV))

    reference_mu = fermi_eV if reference_fermi_eV is None else reference_fermi_eV
    initial_reference = fermi(initial, reference_mu, reference_Ry * RYDBERG_EV)
    final_reference = fermi(final, reference_mu, reference_Ry * RYDBERG_EV)
    fact_reference = final_reference - initial_reference
    result = []
    for temperature_Ry in temperatures_Ry:
        initial_target = fermi(initial, fermi_eV, temperature_Ry * RYDBERG_EV)
        final_target = fermi(final, fermi_eV, temperature_Ry * RYDBERG_EV)
        fact_target = final_target - initial_target
        integrand = g_squared * (fact_target - fact_reference) * denominator
        # wkf contains the spin factor in EPW; on a uniform mesh this is 2/Nk.
        result.append(float(2.0 * np.mean(np.sum(integrand, axis=(-2, -1)))))
    return np.asarray(result, float)


def signed_frequency(background: np.ndarray, correction: np.ndarray) -> np.ndarray:
    squared = np.asarray(background, float) ** 2 + np.asarray(correction, float)
    return np.sign(squared) * np.sqrt(np.abs(squared))


def comparison_metrics(
    target: dict, replay_cm2: np.ndarray, bare_replay_cm1: np.ndarray
) -> dict:
    observed = target["correction_cm2"]
    global_scale = float(np.sum(replay_cm2 * observed) / np.sum(replay_cm2**2))
    scaled = global_scale * replay_cm2
    center = int(np.argmin(np.abs(target["t_GK"] - 1.0)))
    anchored = observed[:, [center]] + replay_cm2 - replay_cm2[:, [center]]

    def frequency_metrics(correction: np.ndarray) -> dict:
        target_frequency = signed_frequency(target["background_cm1"], observed)
        replay_frequency = signed_frequency(target["background_cm1"], correction)
        by_temperature = {}
        for index, temperature in enumerate(TEMPERATURES):
            target_depth = float(
                0.5
                * (
                    target_frequency[index, center - 2]
                    + target_frequency[index, center + 2]
                )
                - target_frequency[index, center]
            )
            replay_depth = float(
                0.5
                * (
                    replay_frequency[index, center - 2]
                    + replay_frequency[index, center + 2]
                )
                - replay_frequency[index, center]
            )
            by_temperature[str(temperature)] = {
                "frequency_RMSE_cm-1": float(
                    np.sqrt(
                        np.mean(
                            (replay_frequency[index] - target_frequency[index]) ** 2
                        )
                    )
                ),
                "frequency_max_abs_cm-1": float(
                    np.max(
                        np.abs(replay_frequency[index] - target_frequency[index])
                    )
                ),
                "target_d1_over_120_cusp_depth_cm-1": target_depth,
                "replay_d1_over_120_cusp_depth_cm-1": replay_depth,
                "cusp_depth_relative_error": float(
                    abs(replay_depth - target_depth) / abs(target_depth)
                ),
            }
        return {
            "frequency_RMSE_cm-1": float(
                np.sqrt(np.mean((replay_frequency - target_frequency) ** 2))
            ),
            "frequency_max_abs_cm-1": float(
                np.max(np.abs(replay_frequency - target_frequency))
            ),
            "maximum_cusp_depth_relative_error": max(
                row["cusp_depth_relative_error"] for row in by_temperature.values()
            ),
            "by_temperature": by_temperature,
        }

    return {
        "bare_top_frequency_max_abs_cm-1": float(
            np.max(np.abs(bare_replay_cm1 - target["bare_top_cm1"][0]))
        ),
        "raw_correction_RMSE_cm-2": float(
            np.sqrt(np.mean((replay_cm2 - observed) ** 2))
        ),
        "raw_correction_max_abs_cm-2": float(np.max(np.abs(replay_cm2 - observed))),
        "best_global_multiplicative_scale": global_scale,
        "raw_absolute": frequency_metrics(replay_cm2),
        "single_scaled": frequency_metrics(scaled),
        "separate_exact_K_anchor": frequency_metrics(anchored),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=192)
    parser.add_argument("--fermi-eV", type=float, default=-1.7187)
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
        "--decay-epmate",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E9_epc_bandsum_assets/decay.epmate"
        ),
    )
    parser.add_argument(
        "--decay-epmatp",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E9_epc_bandsum_assets/decay.epmatp"
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
            / "E9_epc_bandsum_replay"
        ),
    )
    args = parser.parse_args()
    if args.nk % 3 != 0:
        raise ValueError("nk must be divisible by 3")

    target = dense_targets(args.dense_data)
    crystal = load_crystal(args.crystal, natoms=2)
    rdw, dimensions = load_rdw(args.epwdata)
    vectors_k, degeneracies_k, lengths_k = zone_centered_wigner(
        crystal["at"], (18, 18, 1)
    )
    vectors_q, degeneracies_q, lengths_q = zone_centered_wigner(
        crystal["at"], (9, 9, 1)
    )
    if len(vectors_k) != dimensions["nrr_k"] or len(vectors_q) != dimensions["nrr_g"]:
        raise ValueError("reconstructed Wigner-Seitz counts do not match epwdata")
    translations, hamiltonian_matrices = read_hr(args.hr)
    if not np.array_equal(translations, vectors_k):
        raise ValueError("Hamiltonian and reconstructed electronic WS order differ")
    epmatwp = load_epmatwp(
        args.epmatwp,
        dimensions["nbndsub"],
        dimensions["nrr_k"],
        dimensions["nmodes"],
        dimensions["nrr_g"],
    )

    decay_e_length, decay_e_max = load_decay(args.decay_epmate)
    decay_p_length, decay_p_max = load_decay(args.decay_epmatp)
    reconstructed_e_length = lengths_k * crystal["alat_bohr"] * BOHR_ANGSTROM
    reconstructed_p_length = lengths_q * crystal["alat_bohr"] * BOHR_ANGSTROM
    reconstructed_e_max = np.max(np.abs(epmatwp), axis=(0, 1, 3, 4))
    reconstructed_p_max = np.max(np.abs(epmatwp), axis=(0, 1, 2, 3))
    decay_metrics = {
        "electronic_length_max_abs_A": float(
            np.max(np.abs(reconstructed_e_length - decay_e_length))
        ),
        "phonon_length_max_abs_A": float(
            np.max(np.abs(reconstructed_p_length - decay_p_length))
        ),
        "electronic_max_g_max_abs_Ry": float(
            np.max(np.abs(reconstructed_e_max - decay_e_max))
        ),
        "phonon_max_g_max_abs_Ry": float(
            np.max(np.abs(reconstructed_p_max - decay_p_max))
        ),
    }

    initial_h = hamiltonian_grid(
        translations, hamiltonian_matrices, args.nk, np.zeros(3)
    )
    initial_energy, initial_vectors = np.linalg.eigh(initial_h)
    replay_Ry2 = np.empty((len(TEMPERATURES), len(target["qpoints"])), float)
    bare_top_cm1 = np.empty(len(target["qpoints"]), float)
    for iq, qpoint in enumerate(target["qpoints"]):
        _, bare_top_cm1[iq], physical_mode = dynamical_matrix_and_mode(
            rdw,
            qpoint,
            vectors_q,
            degeneracies_q,
            crystal["masses_au"],
            crystal["atom_types"],
        )
        vertex = epc_wannier_grid(
            epmatwp,
            qpoint,
            vectors_q,
            degeneracies_q,
            physical_mode,
            vectors_k,
            degeneracies_k,
            args.nk,
        )
        final_h = hamiltonian_grid(
            translations, hamiltonian_matrices, args.nk, qpoint
        )
        final_energy, final_vectors = np.linalg.eigh(final_h)
        replay_Ry2[:, iq] = response_difference(
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
    replay_cm2 = replay_Ry2 * RYDBERG_CM1**2
    metrics = comparison_metrics(target, replay_cm2, bare_top_cm1)
    checks = {
        "decay_lengths_max_abs_lt_1e-7_A": max(
            decay_metrics["electronic_length_max_abs_A"],
            decay_metrics["phonon_length_max_abs_A"],
        )
        < 1.0e-7,
        "decay_g_max_abs_lt_1e-9_Ry": max(
            decay_metrics["electronic_max_g_max_abs_Ry"],
            decay_metrics["phonon_max_g_max_abs_Ry"],
        )
        < 1.0e-9,
        "bare_top_frequency_max_abs_lt_0p1_cm-1": metrics[
            "bare_top_frequency_max_abs_cm-1"
        ]
        < 0.1,
        "raw_frequency_RMSE_lt_0p3_cm-1": metrics["raw_absolute"][
            "frequency_RMSE_cm-1"
        ]
        < 0.3,
        "raw_max_cusp_depth_relative_error_lt_0p10": metrics["raw_absolute"][
            "maximum_cusp_depth_relative_error"
        ]
        < 0.10,
        "absolute_amplitude_scale_within_0p02": abs(
            metrics["best_global_multiplicative_scale"] - 1.0
        )
        < 0.02,
    }
    status = (
        "finite_epc_bandsum_replay_passed"
        if all(checks.values())
        else "finite_epc_bandsum_replay_not_converged_or_failed"
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "epc_bandsum_replay_arrays.npz",
        nk=np.asarray(args.nk),
        t_GK=target["t_GK"],
        qpoints_crystal=target["qpoints"],
        degauss_Ry=target["degauss_Ry"],
        reference_degauss_Ry=np.asarray(args.reference_degauss_Ry),
        target_correction_cm2=target["correction_cm2"],
        replay_correction_cm2=replay_cm2,
        background_cm1=target["background_cm1"],
        replay_bare_top_cm1=bare_top_cm1,
    )

    center = int(np.argmin(np.abs(target["t_GK"] - 1.0)))
    observed_frequency = signed_frequency(
        target["background_cm1"], target["correction_cm2"]
    )
    anchored_correction = (
        target["correction_cm2"][:, [center]]
        + replay_cm2
        - replay_cm2[:, [center]]
    )
    replay_frequency = signed_frequency(target["background_cm1"], anchored_correction)
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), sharey=True)
    for index, temperature in enumerate(TEMPERATURES):
        axes[index].plot(
            target["t_GK"] - 1.0,
            observed_frequency[index],
            color="black",
            lw=2.0,
            label="EPW target",
        )
        axes[index].plot(
            target["t_GK"] - 1.0,
            replay_frequency[index],
            color="#d55e00",
            lw=1.7,
            label="EPC band-sum replay",
        )
        axes[index].set_title(f"degauss = {target['degauss_Ry'][index]:.7f} Ry")
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
    figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    figure.savefig(
        output / "epc_bandsum_replay_comparison.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "status": status,
        "scope": (
            "direct EPW 7.3.1 epmatwp replay of the A-prime Pi(T)-Pi(T_high) "
            "response on opened finite-smearing curves; no new DFT"
        ),
        "nk": args.nk,
        "fermi_eV": args.fermi_eV,
        "degauss_Ry": target["degauss_Ry"].tolist(),
        "reference_degauss_Ry": args.reference_degauss_Ry,
        "numerical_broadening_eV": args.eta_eV,
        "epmatwp_shape": list(epmatwp.shape),
        "decay_replay": decay_metrics,
        "metrics": metrics,
        "checks": checks,
        "limitations": [
            "opened EPW curves are used to verify the parser and response implementation, not as an independent holdout",
            "production EPW used nk=720; lower nk runs are convergence diagnostics",
            "zero-smearing tetrahedron transfer is a separate integration problem and is not claimed here",
        ],
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in {
                "hr": args.hr,
                "epwdata": args.epwdata,
                "crystal": args.crystal,
                "epmatwp": args.epmatwp,
                "decay_epmate": args.decay_epmate,
                "decay_epmatp": args.decay_epmatp,
                "dense_data": args.dense_data,
            }.items()
        },
    }
    atomic_json(output / "epc_bandsum_replay_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
