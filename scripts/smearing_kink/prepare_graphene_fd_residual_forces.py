#!/usr/bin/env python3
"""Build short-range graphene targets by subtracting a harmonic Kohn force.

The physical-FD thermal labels contain both the local lattice force and the
non-local electronic (Kohn/Friedel) response.  A local MACE should be trained
on the residual

    F_short = F_DFT - F_long,

while the same ``F_long`` is added back during MD and phonon evaluation.  This
script applies a force-capable real-space operator derived from Fermi-Dirac
finite-displacement FC2 data.  FC2 at an intermediate degauss is linearly
interpolated before the long-range tail is extracted.  The interpolation is a
provisional pilot; dense direct-DFPT line cuts later calibrate the final
operator.

Harmonic replay structures are already short-range targets and are copied
without subtraction.  The original total labels and the subtracted component
are retained as ``TOTAL_*`` and ``LONG_RANGE_*`` fields for reconstruction
checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
from friedel_calc import FriedelCorrection  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def minimum_image_vectors(left: np.ndarray, right: np.ndarray, cell: np.ndarray):
    delta = left[:, None, :] - right[None, :, :]
    inverse = np.linalg.inv(cell)
    fractional = delta @ inverse
    fractional -= np.round(fractional)
    return fractional @ cell


def atom_mapping(structure, ideal) -> tuple[np.ndarray, float]:
    if len(structure) != len(ideal):
        raise ValueError("thermal structure and FC2 supercell have different sizes")
    cell = np.asarray(structure.cell, float)
    if not np.allclose(cell, np.asarray(ideal.cell, float), atol=2e-5, rtol=0.0):
        raise ValueError("thermal structure cell does not match FC2 supercell")
    vectors = minimum_image_vectors(
        np.asarray(structure.positions, float), np.asarray(ideal.positions, float), cell
    )
    costs = np.linalg.norm(vectors, axis=2)
    row, column = linear_sum_assignment(costs)
    if not np.array_equal(row, np.arange(len(structure))):
        order = np.argsort(row)
        column = column[order]
    if not np.array_equal(
        np.asarray(structure.numbers), np.asarray(ideal.numbers)[column]
    ):
        raise ValueError("atom assignment changes chemical species")
    return column.astype(int), float(costs[np.arange(len(structure)), column].max())


def interpolate_force_constants(low, high, target, low_degauss, high_degauss):
    weight = (target - low_degauss) / (high_degauss - low_degauss)
    clipped = float(np.clip(weight, 0.0, 1.0))
    return (1.0 - clipped) * low + clipped * high, clipped


def harmonic_energy_forces(fc_full, reference_positions, cell, positions):
    displacement = np.asarray(positions, float) - reference_positions
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    displacement = fractional @ cell
    phi_u = np.einsum("ijab,jb->ia", fc_full, displacement)
    energy = 0.5 * float(np.einsum("ia,ia->", displacement, phi_u))
    return energy, -phi_u, displacement


def thermal_structure(atom) -> bool:
    return str(atom.info.get("config_type", "")).startswith("physical_fd_thermal")


def build_lane(
    temperature: int,
    degauss: float,
    input_lane: Path,
    output_lane: Path,
    ideal,
    delta_fc_phonopy_order: np.ndarray,
) -> dict:
    output_lane.mkdir(parents=True, exist_ok=True)
    mapping = None
    reference_positions = None
    delta_fc = None
    metrics: dict[str, dict] = {}

    for split in ("train", "val", "test"):
        source = input_lane / f"{split}.xyz"
        if not source.is_file():
            continue
        structures = read(source, index=":")
        output = []
        long_range_values = []
        residual_values = []
        mapping_displacements = []
        for atom in structures:
            result = atom.copy()
            total_forces = np.asarray(atom.arrays["REF_forces"], float).copy()
            total_energy = float(atom.info["REF_energy"])
            result.arrays["TOTAL_forces"] = total_forces.copy()
            result.info["TOTAL_energy"] = total_energy

            if thermal_structure(atom):
                if mapping is None:
                    mapping, assignment_max = atom_mapping(atom, ideal)
                    reference_positions = np.asarray(ideal.positions, float)[mapping]
                    delta_fc = delta_fc_phonopy_order[mapping][:, mapping]
                else:
                    _, assignment_max = atom_mapping(atom, ideal)
                energy_long, force_long, displacement = harmonic_energy_forces(
                    delta_fc,
                    reference_positions,
                    np.asarray(atom.cell, float),
                    np.asarray(atom.positions, float),
                )
                net_force = np.linalg.norm(force_long.sum(axis=0))
                if net_force > 2e-5:
                    raise ValueError(
                        f"long-range operator violates translational invariance: {net_force}"
                    )
                if np.max(np.linalg.norm(displacement, axis=1)) > 1.5:
                    raise ValueError("thermal displacement exceeds the residual-pilot limit")
                result.arrays["REF_forces"] = total_forces - force_long
                result.info["REF_energy"] = total_energy - energy_long
                result.arrays["LONG_RANGE_forces"] = force_long
                result.info["LONG_RANGE_energy"] = energy_long
                result.info["force_target"] = "DFT_total_minus_FD_long_range"
                long_range_values.append(force_long.reshape(-1))
                residual_values.append(result.arrays["REF_forces"].reshape(-1))
                mapping_displacements.append(assignment_max)
            else:
                result.arrays["LONG_RANGE_forces"] = np.zeros_like(total_forces)
                result.info["LONG_RANGE_energy"] = 0.0
                result.info["force_target"] = "short_range_harmonic_replay"
            result.info["long_range_degauss_Ry"] = degauss
            result.info["long_range_temperature_K"] = temperature
            output.append(result)

        write(output_lane / f"{split}.xyz", output, format="extxyz")
        joined_long = np.concatenate(long_range_values) if long_range_values else np.zeros(1)
        joined_residual = (
            np.concatenate(residual_values) if residual_values else np.zeros(1)
        )
        metrics[split] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "n_structures": len(structures),
            "n_thermal_structures": len(long_range_values),
            "long_range_force_RMSE_meV_A": float(
                np.sqrt(np.mean(joined_long**2)) * 1000.0
            ),
            "long_range_force_max_abs_meV_A": float(
                np.max(np.abs(joined_long)) * 1000.0
            ),
            "residual_force_RMS_meV_A": float(
                np.sqrt(np.mean(joined_residual**2)) * 1000.0
            ),
            "assignment_max_distance_A": float(max(mapping_displacements, default=0.0)),
            "output_sha256": sha256(output_lane / f"{split}.xyz"),
        }

    if mapping is None or delta_fc is None or reference_positions is None:
        raise ValueError(f"no thermal structures found in {input_lane}")
    operator_path = output_lane / "long_range_operator.npz"
    np.savez(
        operator_path,
        temperature_K=np.array(temperature),
        degauss_Ry=np.array(degauss),
        delta_fc_full=np.asarray(delta_fc_phonopy_order, float),
        reference_positions=np.asarray(ideal.positions, float),
        cell=np.asarray(ideal.cell, float),
        atom_mapping=np.asarray(mapping, int),
    )
    return {
        "temperature_K": temperature,
        "degauss_Ry": degauss,
        "atom_mapping_is_identity": bool(np.array_equal(mapping, np.arange(len(mapping)))),
        "operator_sha256": sha256(operator_path),
        "splits": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--reference-low", type=Path, required=True)
    parser.add_argument("--reference-high", type=Path, required=True)
    parser.add_argument("--reference-low-degauss", type=float, default=0.002)
    parser.add_argument("--reference-high-degauss", type=float, default=0.005)
    parser.add_argument("--degauss-300", type=float, default=0.0019000869)
    parser.add_argument("--degauss-600", type=float, default=0.0038001738)
    parser.add_argument("--rmin", type=float, default=1.0)
    parser.add_argument("--rmax", type=float, default=12.0)
    args = parser.parse_args()

    phonon = fm.load_ph(args.background)
    low = fm.load_ph(args.reference_low)
    high = fm.load_ph(args.reference_high)
    shapes = {phonon.force_constants.shape, low.force_constants.shape, high.force_constants.shape}
    if len(shapes) != 1:
        raise ValueError("background and reference FC2 arrays have different shapes")
    ideal = phonopy_to_ase(phonon.supercell)
    ideal.wrap()

    lanes = {}
    interpolation = {}
    for temperature, degauss in ((300, args.degauss_300), (600, args.degauss_600)):
        target_fc, weight = interpolate_force_constants(
            low.force_constants,
            high.force_constants,
            degauss,
            args.reference_low_degauss,
            args.reference_high_degauss,
        )
        correction = FriedelCorrection(
            phonon,
            phonon.force_constants,
            target_fc,
            rmin=args.rmin,
            rmax=args.rmax,
            B_law=lambda _: 1.0,
            kappa_law=lambda _: 0.0,
        )
        delta_fc = correction.delta_fc_full(0.0)
        transpose_error = float(
            np.max(np.abs(delta_fc - delta_fc.transpose(1, 0, 3, 2)))
        )
        if transpose_error > 2e-5:
            raise ValueError(f"long-range FC2 pair symmetry error: {transpose_error}")
        interpolation[str(temperature)] = {
            "target_degauss_Ry": degauss,
            "clipped_linear_weight_high": weight,
            "pair_symmetry_max_abs_eV_A2": transpose_error,
        }
        lanes[str(temperature)] = build_lane(
            temperature,
            degauss,
            args.input / f"T{temperature}",
            args.output / f"T{temperature}",
            ideal,
            delta_fc,
        )

    manifest = {
        "method": "provisional Fermi-Dirac real-space long-range force subtraction",
        "status": "pilot_pending_dense_DFPT_calibration",
        "input": str(args.input),
        "background": str(args.background),
        "background_sha256": sha256(args.background),
        "reference_low": str(args.reference_low),
        "reference_low_sha256": sha256(args.reference_low),
        "reference_high": str(args.reference_high),
        "reference_high_sha256": sha256(args.reference_high),
        "rmin_A": args.rmin,
        "rmax_A": args.rmax,
        "interpolation": interpolation,
        "lanes": lanes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
