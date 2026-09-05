#!/usr/bin/env python3
"""Diagnose graphene Wannier-band geometry around K without new DFT.

This separates effects that can already be fixed by the electronic Hamiltonian
(Dirac velocity, curvature, and trigonal warping) from the missing A' EPC
vertex.  It also checks how much of the two empirical finite-smearing q-basis
subspace can be represented by simple band-geometry features.
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

from diagnose_graphene_epw_wannier import eigenvalues, read_hr  # noqa: E402


BOHR_ANGSTROM = 0.529177210903


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


def read_lattice(crystal_path: Path) -> np.ndarray:
    lines = [line.strip() for line in crystal_path.read_text().splitlines() if line.strip()]
    values = np.asarray([float(value) for value in lines[3].split()], float)
    at = values.reshape(3, 3, order="F")
    alat_angstrom = float(lines[6]) * BOHR_ANGSTROM
    return at * alat_angstrom


def reciprocal_lattice(real_lattice: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(real_lattice).T


def qpoint(direction: str, distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, float)
    if direction == "KG":
        h = (1.0 - distance) / 3.0
        k = h
    elif direction == "KM":
        h = (1.0 + distance) / 3.0
        k = (1.0 - 2.0 * distance) / 3.0
    else:
        raise ValueError(f"unsupported direction: {direction}")
    return np.column_stack((h, k, np.zeros_like(h)))


def radial_wavevector(
    direction: str, distance: np.ndarray, reciprocal: np.ndarray
) -> np.ndarray:
    kpoint = np.asarray([1.0 / 3.0, 1.0 / 3.0, 0.0])
    delta = qpoint(direction, distance) - kpoint
    cartesian = delta @ reciprocal.T
    return np.linalg.norm(cartesian, axis=1)


def fit_branch(q_invA: np.ndarray, energy_eV: np.ndarray) -> dict:
    origin = float(energy_eV[0])
    fit_mask = (q_invA >= 0.003) & (q_invA <= 0.105)
    design = np.column_stack(
        (q_invA[fit_mask], q_invA[fit_mask] ** 2, q_invA[fit_mask] ** 3)
    )
    coefficients, *_ = np.linalg.lstsq(
        design, energy_eV[fit_mask] - origin, rcond=None
    )
    prediction = origin + np.column_stack(
        (q_invA, q_invA**2, q_invA**3)
    ) @ coefficients
    linear_mask = (q_invA >= 0.003) & (q_invA <= 0.035)
    linear_slope = float(
        np.dot(q_invA[linear_mask], energy_eV[linear_mask] - origin)
        / np.dot(q_invA[linear_mask], q_invA[linear_mask])
    )
    return {
        "K_offset_eV": origin,
        "small_q_velocity_eV_A": linear_slope,
        "cubic_fit": {
            "linear_eV_A": float(coefficients[0]),
            "quadratic_eV_A2": float(coefficients[1]),
            "cubic_eV_A3": float(coefficients[2]),
            "RMSE_meV": float(
                1000.0
                * np.sqrt(np.mean((prediction[fit_mask] - energy_eV[fit_mask]) ** 2))
            ),
        },
    }


def orthonormal_columns(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, float)
    norms = np.linalg.norm(features, axis=0)
    keep = norms > 1.0e-14
    q, _ = np.linalg.qr(features[:, keep])
    return q


def subspace_capture(target_rows: np.ndarray, features: np.ndarray) -> dict:
    target = np.asarray(target_rows, float)
    feature_basis = orthonormal_columns(features)
    projected = target @ feature_basis @ feature_basis.T
    capture = float(np.sum(projected**2) / np.sum(target**2))
    return {
        "frobenius_capture": capture,
        "relative_residual": float(np.sqrt(max(0.0, 1.0 - capture))),
        "feature_rank": int(feature_basis.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
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
        "--crystal",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E0_epw_matched/k18_q9_ex1_pifroz/restart_small/crystal.fmt"
        ),
    )
    parser.add_argument(
        "--scalar-bases",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E5_epw_scalar_low_rank/epw_scalar_two_basis_development.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E6_wannier_k_geometry"
        ),
    )
    args = parser.parse_args()

    translations, matrices = read_hr(args.hr)
    real_lattice = read_lattice(args.crystal)
    reciprocal = reciprocal_lattice(real_lattice)
    distances = np.linspace(0.0, 0.06, 121)
    K = np.asarray([[1.0 / 3.0, 1.0 / 3.0, 0.0]])
    K_bands = eigenvalues(translations, matrices, K)[0]
    dirac_mid = float(np.mean(K_bands))

    rows: list[dict] = []
    branch_fits: dict[str, dict] = {}
    direction_arrays: dict[str, dict[str, np.ndarray]] = {}
    for direction in ("KG", "KM"):
        q_invA = radial_wavevector(direction, distances, reciprocal)
        bands = eigenvalues(translations, matrices, qpoint(direction, distances))
        electron = bands[:, 1] - dirac_mid
        hole = dirac_mid - bands[:, 0]
        mean_excitation = 0.5 * (electron + hole)
        direction_arrays[direction] = {
            "q_invA": q_invA,
            "electron": electron,
            "hole": hole,
            "mean": mean_excitation,
        }
        branch_fits[direction] = {
            "electron": fit_branch(q_invA, electron),
            "hole": fit_branch(q_invA, hole),
            "electron_hole_mean": fit_branch(q_invA, mean_excitation),
        }
        for index, distance in enumerate(distances):
            rows.append(
                {
                    "direction": direction,
                    "distance_from_K": float(distance),
                    "q_invA": float(q_invA[index]),
                    "electron_eV": float(electron[index]),
                    "hole_eV": float(hole[index]),
                    "electron_hole_mean_eV": float(mean_excitation[index]),
                }
            )

    mean_KG = direction_arrays["KG"]["mean"]
    mean_KM = direction_arrays["KM"]["mean"]
    directional_mean = 0.5 * (mean_KG + mean_KM)
    warping = 0.5 * (mean_KM - mean_KG)
    nonzero = distances > 0.0
    fractional_warping = np.abs(warping[nonzero]) / directional_mean[nonzero]

    with np.load(args.scalar_bases, allow_pickle=False) as payload:
        basis_distances = np.asarray(payload["distances_from_K"], float)
        empirical_bases = np.asarray(payload["scalar_q_bases"], float)
    basis_nonzero = basis_distances[1:]
    if empirical_bases.shape[1] != len(basis_nonzero):
        raise ValueError("scalar-basis distance grid mismatch")

    sample_KG = eigenvalues(
        translations, matrices, qpoint("KG", basis_nonzero)
    )
    sample_KM = eigenvalues(
        translations, matrices, qpoint("KM", basis_nonzero)
    )
    sample_mean_KG = 0.5 * (sample_KG[:, 1] - sample_KG[:, 0])
    sample_mean_KM = 0.5 * (sample_KM[:, 1] - sample_KM[:, 0])
    sample_isotropic = 0.5 * (sample_mean_KG + sample_mean_KM)
    sample_warping = 0.5 * (sample_mean_KM - sample_mean_KG)
    q_sample = radial_wavevector("KG", basis_nonzero, reciprocal)
    velocity = 0.5 * (
        branch_fits["KG"]["electron_hole_mean"]["small_q_velocity_eV_A"]
        + branch_fits["KM"]["electron_hole_mean"]["small_q_velocity_eV_A"]
    )
    curvature_residual = sample_isotropic - velocity * q_sample
    captures = {
        "analytic_q_q2": subspace_capture(
            empirical_bases, np.column_stack((q_sample, q_sample**2))
        ),
        "band_mean_and_curvature_residual": subspace_capture(
            empirical_bases,
            np.column_stack((sample_isotropic, curvature_residual)),
        ),
        "band_mean_and_trigonal_warping": subspace_capture(
            empirical_bases, np.column_stack((sample_isotropic, sample_warping))
        ),
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "wannier_k_geometry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    colors = {"KG": "#2266aa", "KM": "#cc5522"}
    for direction in ("KG", "KM"):
        arrays = direction_arrays[direction]
        axes[0].plot(
            arrays["q_invA"],
            arrays["mean"] * 1000.0,
            color=colors[direction],
            label=direction,
        )
    axes[0].set_xlabel(r"$|k-K|$ ($\mathrm{\AA}^{-1}$)")
    axes[0].set_ylabel("electron-hole mean energy (meV)")
    axes[0].set_title("Production Wannier cone")
    axes[0].legend(frameon=False)

    q = direction_arrays["KG"]["q_invA"]
    axes[1].plot(q, warping * 1000.0, color="#7a3db8")
    axes[1].axhline(0.0, color="0.6", lw=0.8)
    axes[1].set_xlabel(r"$|k-K|$ ($\mathrm{\AA}^{-1}$)")
    axes[1].set_ylabel(r"$(E_{KM}-E_{KG})/2$ (meV)")
    axes[1].set_title("Trigonal-warping component")
    fig.tight_layout()
    fig.savefig(
        output / "wannier_k_geometry.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)

    summary = {
        "status": "band_geometry_quantified_EPC_vertex_still_required",
        "scope": "production k18/q9 ex1_pifroz Wannier Hamiltonian; no new DFT and no EPC approximation",
        "K_bands_eV": K_bands.tolist(),
        "K_gap_eV": float(K_bands[1] - K_bands[0]),
        "dirac_mid_eV": dirac_mid,
        "lattice_A": real_lattice.tolist(),
        "distance_conversion_invA_per_unit_d": float(
            radial_wavevector("KG", np.asarray([1.0]), reciprocal)[0]
        ),
        "branch_fits": branch_fits,
        "trigonal_warping": {
            "maximum_abs_meV_to_d_0p06": float(1000.0 * np.max(np.abs(warping))),
            "maximum_fraction_of_mean_excitation_to_d_0p06": float(
                np.max(fractional_warping)
            ),
            "at_d_0p003_interpolated_meV": float(
                1000.0 * np.interp(0.003, distances, warping)
            ),
            "at_d_0p06_meV": float(1000.0 * warping[-1]),
        },
        "empirical_two_q_basis_subspace_capture": captures,
        "interpretation": (
            "The Hamiltonian fixes v_F, analytic curvature, and KG/KM warping. "
            "Any remaining finite-smearing two-basis structure cannot be assigned "
            "to the band geometry alone; an A-prime EPC vertex is required for the "
            "intervalley band-sum response."
        ),
        "inputs": {
            "hr": {"path": str(args.hr), "sha256": sha256(args.hr)},
            "crystal": {
                "path": str(args.crystal),
                "sha256": sha256(args.crystal),
            },
            "scalar_bases": {
                "path": str(args.scalar_bases),
                "sha256": sha256(args.scalar_bases),
            },
        },
    }
    atomic_json(output / "wannier_k_geometry_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
