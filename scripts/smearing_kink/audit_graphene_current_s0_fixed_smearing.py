#!/usr/bin/env python3
"""Audit current S0 + one conservative q6 operator against DFT labels.

The script evaluates the frozen base and delta MACE models on an extxyz
dataset, reconstructs the total force and energy with a harmonic q6 operator,
and reports Cartesian and folded-K A' projections.  Atom order is resolved for
every structure by a periodic Hungarian assignment to the operator reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def force_metrics(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, float).reshape(-1)
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(values**2)) * 1000.0),
        "MAE_meV_A": float(np.mean(np.abs(values)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(values)) * 1000.0),
    }


def complex_metrics(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, complex).reshape(-1)
    magnitudes = np.abs(values)
    return {
        "RMS_meV_A": float(np.sqrt(np.mean(magnitudes**2)) * 1000.0),
        "MAE_meV_A": float(np.mean(magnitudes) * 1000.0),
        "max_abs_meV_A": float(np.max(magnitudes) * 1000.0),
    }


def minimum_image_vectors(left: np.ndarray, right: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def structure_mapping(structure, reference_positions: np.ndarray, cell: np.ndarray):
    if len(structure) != len(reference_positions):
        raise ValueError("structure and operator reference have different atom counts")
    if not np.allclose(np.asarray(structure.cell), cell, atol=2.0e-5, rtol=0.0):
        raise ValueError("structure and operator cells differ")
    vectors = minimum_image_vectors(
        np.asarray(structure.positions, float), reference_positions, cell
    )
    row, column = linear_sum_assignment(np.linalg.norm(vectors, axis=2))
    if not np.array_equal(row, np.arange(len(structure))):
        column = column[np.argsort(row)]
    displacement = vectors[np.arange(len(structure)), column]
    return column.astype(int), displacement


def load_operator(path: Path):
    with np.load(path, allow_pickle=False) as data:
        raw_fc = np.asarray(data["delta_fc_full"], float)
        raw_reference = np.asarray(data["reference_positions"], float)
        mapping = np.asarray(data["atom_mapping"], int)
        cell = np.asarray(data["cell"], float)
        degauss = float(np.asarray(data["degauss_Ry"]).reshape(()))
    if sorted(mapping.tolist()) != list(range(len(mapping))):
        raise ValueError("operator atom_mapping is not a permutation")
    return raw_fc[mapping][:, mapping], raw_reference[mapping], cell, degauss


def periodic_assignment(left: np.ndarray, right: np.ndarray, cell: np.ndarray) -> np.ndarray:
    vectors = minimum_image_vectors(left, right, cell)
    row, column = linear_sum_assignment(np.linalg.norm(vectors, axis=2))
    if not np.array_equal(row, np.arange(len(left))):
        column = column[np.argsort(row)]
    matched = vectors[np.arange(len(left)), column]
    if float(np.max(np.linalg.norm(matched, axis=1))) > 2.0e-5:
        raise ValueError("reference structures cannot be matched within tolerance")
    return column.astype(int)


def folded_k_aprime_mode(
    background: Path,
    thermal_result: Path,
    operator_reference: np.ndarray,
    operator_cell: np.ndarray,
) -> tuple[np.ndarray, dict]:
    phonon = fm.load_ph(background)
    with np.load(thermal_result, allow_pickle=False) as data:
        force_constants = np.asarray(data["free_energy_fc2_eV_A2"], float)
        lattice_temperature = int(np.asarray(data["lattice_temperature_K"]).reshape(()))
        operator_temperature = int(np.asarray(data["operator_temperature_K"]).reshape(()))
    phonon.force_constants = force_constants
    kpoint = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(kpoint)
    mode_index = int(np.argmax(frequencies))
    primitive_mode = np.asarray(eigenvectors[:, mode_index], complex).reshape(-1, 3)

    supercell = phonopy_to_ase(phonon.supercell)
    super_positions = np.asarray(supercell.positions, float)
    if not np.allclose(np.asarray(supercell.cell), operator_cell, atol=2.0e-5, rtol=0.0):
        raise ValueError("phonopy and operator supercells differ")
    operator_to_phonopy = periodic_assignment(
        operator_reference, super_positions, operator_cell
    )

    unitcell = phonon.unitcell
    unit_cell = np.asarray(unitcell.cell, float)
    unit_scaled = np.asarray(unitcell.scaled_positions, float)
    representatives = np.asarray(phonon.supercell.u2s_map, int)
    representative_to_basis = {
        int(representative): index for index, representative in enumerate(representatives)
    }
    s2u = np.asarray(phonon.supercell.s2u_map, int)
    expanded = np.zeros((len(supercell), 3), complex)
    translations = []
    basis_indices = []
    for index, position in enumerate(super_positions):
        basis = representative_to_basis[int(s2u[index])]
        fractional_unit = position @ np.linalg.inv(unit_cell)
        translation = np.rint(fractional_unit - unit_scaled[basis]).astype(int)
        mismatch = fractional_unit - unit_scaled[basis] - translation
        if float(np.max(np.abs(mismatch))) > 2.0e-5:
            raise ValueError("could not recover an integer supercell translation")
        phase = np.exp(2.0j * np.pi * float(kpoint @ translation))
        expanded[index] = primitive_mode[basis] * phase
        translations.append(translation.tolist())
        basis_indices.append(int(basis))
    expanded /= np.linalg.norm(expanded)
    mode_operator = expanded[operator_to_phonopy]
    mode_operator /= np.linalg.norm(mode_operator)
    provenance = {
        "qpoint_crystal": kpoint.tolist(),
        "primitive_mode_index": mode_index,
        "primitive_frequency_THz": float(frequencies[mode_index]),
        "primitive_frequency_cm_1": float(frequencies[mode_index] * 33.35641),
        "lattice_temperature_K": lattice_temperature,
        "operator_temperature_K": operator_temperature,
        "operator_to_phonopy_mapping": operator_to_phonopy.tolist(),
        "phonopy_basis_indices": basis_indices,
        "phonopy_cell_translations": translations,
        "normalization": float(np.vdot(mode_operator.reshape(-1), mode_operator.reshape(-1)).real),
    }
    return mode_operator, provenance


def scalar_key(structure) -> tuple[int, int]:
    return int(structure.info.get("trajectory_seed", -1)), int(
        structure.info.get("snapshot_index", -1)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--old-dataset", type=Path)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--delta-model", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--thermal-result", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    structures = read(args.dataset, index=":")
    if not structures:
        raise ValueError("empty audit dataset")
    force_constants, reference_positions, cell, operator_degauss = load_operator(
        args.operator
    )
    for structure in structures:
        if abs(float(structure.info["degauss_Ry"]) - operator_degauss) > 5.0e-10:
            raise ValueError("dataset and operator degauss differ")
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.thermal_result, reference_positions, cell
    )

    base = MACECalculator(
        model_paths=str(args.base_model), device=args.device, default_dtype="float32"
    )
    delta = MACECalculator(
        model_paths=str(args.delta_model), device=args.device, default_dtype="float32"
    )
    base_forces = []
    delta_forces = []
    long_forces = []
    total_forces = []
    targets = []
    errors = []
    displacements = []
    mappings = []
    mode_coordinates = []
    mode_predicted_forces = []
    mode_target_forces = []
    predicted_energies = []
    target_energies = []
    records = []
    for index, structure in enumerate(structures):
        mapping, displacement = structure_mapping(structure, reference_positions, cell)
        reordered_fc = force_constants[mapping][:, mapping]
        long_force = -np.einsum("ijab,jb->ia", reordered_fc, displacement)
        long_energy = 0.5 * float(
            np.einsum("ia,ijab,jb->", displacement, reordered_fc, displacement)
        )
        atoms = structure.copy()
        atoms.calc = base
        base_energy = float(atoms.get_potential_energy())
        base_force = np.asarray(atoms.get_forces(), float)
        atoms.calc = delta
        delta_energy = float(atoms.get_potential_energy())
        delta_force = np.asarray(atoms.get_forces(), float)
        predicted_force = base_force + delta_force + long_force
        target_force = np.asarray(structure.arrays["REF_forces"], float)
        error = predicted_force - target_force
        mode_structure = mode[mapping]
        mode_flat = mode_structure.reshape(-1)
        coordinate = np.vdot(mode_flat, displacement.reshape(-1))
        predicted_mode_force = np.vdot(mode_flat, predicted_force.reshape(-1))
        target_mode_force = np.vdot(mode_flat, target_force.reshape(-1))
        predicted_energy = base_energy + delta_energy + long_energy
        target_energy = float(structure.info["REF_energy"])

        base_forces.append(base_force)
        delta_forces.append(delta_force)
        long_forces.append(long_force)
        total_forces.append(predicted_force)
        targets.append(target_force)
        errors.append(error)
        displacements.append(displacement)
        mappings.append(mapping)
        mode_coordinates.append(coordinate)
        mode_predicted_forces.append(predicted_mode_force)
        mode_target_forces.append(target_mode_force)
        predicted_energies.append(predicted_energy)
        target_energies.append(target_energy)
        maximum_flat = int(np.argmax(np.abs(error)))
        maximum_atom, maximum_component = np.unravel_index(maximum_flat, error.shape)
        records.append(
            {
                "structure_index": index,
                "trajectory_seed": scalar_key(structure)[0],
                "snapshot_index": scalar_key(structure)[1],
                "force_RMSE_meV_A": float(np.sqrt(np.mean(error**2)) * 1000.0),
                "force_MAE_meV_A": float(np.mean(np.abs(error)) * 1000.0),
                "force_max_abs_meV_A": float(np.max(np.abs(error)) * 1000.0),
                "max_error_atom_index": int(maximum_atom),
                "max_error_component": "xyz"[int(maximum_component)],
                "max_displacement_vector_A": float(
                    np.max(np.linalg.norm(displacement, axis=1))
                ),
                "RMS_displacement_A": float(np.sqrt(np.mean(displacement**2))),
                "Aprime_coordinate_abs_A": float(abs(coordinate)),
                "Aprime_force_error_abs_meV_A": float(
                    abs(predicted_mode_force - target_mode_force) * 1000.0
                ),
                "predicted_energy_eV": predicted_energy,
                "target_energy_eV": target_energy,
            }
        )

    base_forces = np.asarray(base_forces)
    delta_forces = np.asarray(delta_forces)
    long_forces = np.asarray(long_forces)
    total_forces = np.asarray(total_forces)
    targets = np.asarray(targets)
    errors = np.asarray(errors)
    displacements = np.asarray(displacements)
    mappings = np.asarray(mappings)
    mode_coordinates = np.asarray(mode_coordinates)
    mode_predicted_forces = np.asarray(mode_predicted_forces)
    mode_target_forces = np.asarray(mode_target_forces)
    predicted_energies = np.asarray(predicted_energies)
    target_energies = np.asarray(target_energies)
    energy_error = predicted_energies - target_energies
    centered_energy_error = energy_error - np.mean(energy_error)
    for record, value in zip(records, centered_energy_error):
        record["centered_energy_error_meV_config"] = float(value * 1000.0)

    q = mode_coordinates
    denominator = float(np.vdot(q, q).real)
    if denominator <= 0.0:
        raise ValueError("all A-prime coordinates are zero")
    predicted_slope = -float(
        np.vdot(q, mode_predicted_forces).real / denominator
    )
    target_slope = -float(np.vdot(q, mode_target_forces).real / denominator)
    relative_slope_error = (
        (predicted_slope - target_slope) / target_slope
        if abs(target_slope) > 1.0e-14
        else float("nan")
    )

    seeds = np.asarray([scalar_key(structure)[0] for structure in structures], int)
    summary = {
        "status": "current_S0_fixed_smearing_audit_complete",
        "scope": "current base + S0 delta + fixed-smearing q6 operator versus E50 DFT labels",
        "n_structures": len(structures),
        "condition": {
            "lattice_temperature_K": float(structures[0].info["lattice_temperature_K"]),
            "smearing": "fermi-dirac",
            "degauss_Ry": operator_degauss,
        },
        "force_error": force_metrics(errors),
        "force_error_by_seed": {
            str(seed): force_metrics(errors[seeds == seed]) for seed in sorted(set(seeds))
        },
        "force_magnitudes": {
            "base": force_metrics(base_forces),
            "delta": force_metrics(delta_forces),
            "long_range": force_metrics(long_forces),
            "predicted_total": force_metrics(total_forces),
            "DFT_target": force_metrics(targets),
        },
        "Aprime_projection": {
            "force_error": complex_metrics(
                mode_predicted_forces - mode_target_forces
            ),
            "predicted_restoring_slope_eV_A2": predicted_slope,
            "DFT_restoring_slope_eV_A2": target_slope,
            "relative_slope_error": relative_slope_error,
            "mode": mode_provenance,
        },
        "energy_error_after_global_offset": {
            "RMSE_meV_config": float(
                np.sqrt(np.mean(centered_energy_error**2)) * 1000.0
            ),
            "MAE_meV_config": float(np.mean(np.abs(centered_energy_error)) * 1000.0),
            "max_abs_meV_config": float(np.max(np.abs(centered_energy_error)) * 1000.0),
            "RMSE_meV_atom": float(
                np.sqrt(np.mean(centered_energy_error**2)) * 1000.0 / len(structures[0])
            ),
        },
        "fixed_R1_thresholds_not_applied_to_opened_E50": {
            "force_RMSE_meV_A": 30.0,
            "force_max_abs_meV_A": 200.0,
            "Aprime_projected_force_RMS_meV_A": 15.0,
            "Aprime_restoring_slope_relative_error": 0.05,
            "centered_energy_RMSE_meV_config": 19.4,
            "importance_weight_ESS_fraction": 0.3,
        },
        "inputs": {
            "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
            "base_model": {
                "path": str(args.base_model),
                "sha256": sha256(args.base_model),
            },
            "delta_model": {
                "path": str(args.delta_model),
                "sha256": sha256(args.delta_model),
            },
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {
                "path": str(args.background),
                "sha256": sha256(args.background),
            },
            "thermal_result": {
                "path": str(args.thermal_result),
                "sha256": sha256(args.thermal_result),
            },
        },
        "records": records,
    }

    if args.old_dataset is not None:
        old_structures = read(args.old_dataset, index=":")
        old_by_key = {scalar_key(structure): structure for structure in old_structures}
        differences = []
        energy_differences = []
        for structure in structures:
            old = old_by_key[scalar_key(structure)]
            differences.append(
                np.asarray(structure.arrays["REF_forces"], float)
                - np.asarray(old.arrays["REF_forces"], float)
            )
            energy_differences.append(
                float(structure.info["REF_energy"]) - float(old.info["REF_energy"])
            )
        energy_differences = np.asarray(energy_differences)
        centered = energy_differences - np.mean(energy_differences)
        summary["new_minus_old_DFT_same_geometry"] = {
            "old_dataset": {
                "path": str(args.old_dataset),
                "sha256": sha256(args.old_dataset),
            },
            "force": force_metrics(np.asarray(differences)),
            "centered_energy_RMSE_meV_config": float(
                np.sqrt(np.mean(centered**2)) * 1000.0
            ),
        }

    flattened_error = errors.reshape(-1)
    flattened_long = long_forces.reshape(-1)
    extra_lr_scalar = -float(
        flattened_error @ flattened_long / (flattened_long @ flattened_long)
    )
    rescaled_error = errors + extra_lr_scalar * long_forces
    summary["long_range_global_rescale_diagnostic"] = {
        "optimal_extra_scalar": extra_lr_scalar,
        "force_RMSE_after_rescale_meV_A": force_metrics(rescaled_error)["RMSE_meV_A"],
        "squared_error_fraction_removed": float(
            1.0 - np.sum(rescaled_error**2) / np.sum(errors**2)
        ),
        "interpretation": "development-only diagnostic; does not authorize LR refitting",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        args.output_dir / "current_s0_fixed_smearing_predictions.npz",
        base_forces_eV_A=base_forces,
        delta_forces_eV_A=delta_forces,
        long_range_forces_eV_A=long_forces,
        predicted_total_forces_eV_A=total_forces,
        DFT_target_forces_eV_A=targets,
        force_errors_eV_A=errors,
        displacement_A=displacements,
        structure_to_operator_mapping=mappings,
        Aprime_mode_operator_order=mode,
        Aprime_coordinate_A=mode_coordinates,
        Aprime_predicted_force_eV_A=mode_predicted_forces,
        Aprime_DFT_force_eV_A=mode_target_forces,
        predicted_energy_eV=predicted_energies,
        DFT_target_energy_eV=target_energies,
        centered_energy_error_eV=centered_energy_error,
        trajectory_seed=seeds,
        snapshot_index=np.asarray([scalar_key(s)[1] for s in structures], int),
    )
    csv_path = args.output_dir / "current_s0_fixed_smearing_records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    atomic_json(args.output_dir / "current_s0_fixed_smearing_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
