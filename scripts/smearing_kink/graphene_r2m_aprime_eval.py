"""Self-contained, frozen A-prime projection helpers for R2M evaluation.

This module intentionally has no repository-relative imports.  A launcher can
copy it beside the evaluator, lock its SHA-256, and execute the snapshot
without silently falling back to mutable source-tree helpers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import phonopy
from ase import Atoms
from scipy.optimize import linear_sum_assignment


APRIME_MODE_DEFINITION = {
    "qpoint_crystal": [1.0 / 3.0, 1.0 / 3.0, 0.0],
    "branch_rule": "highest-frequency primitive eigenmode at K",
    "phase_rule": "exp(+2*pi*i*q dot integer_supercell_translation)",
    "normalization": "unit Euclidean norm after expansion and operator ordering",
    "assignment_tolerance_A": 2.0e-5,
}


def phonopy_to_ase(ph_atoms) -> Atoms:
    return Atoms(
        symbols=ph_atoms.symbols,
        scaled_positions=ph_atoms.scaled_positions,
        cell=ph_atoms.cell,
        pbc=True,
    )


def minimum_image_vectors(
    left: np.ndarray, right: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def structure_mapping(
    structure: Atoms, reference_positions: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def load_operator(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "delta_fc_full",
            "reference_positions",
            "atom_mapping",
            "cell",
            "degauss_Ry",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"operator is missing arrays: {sorted(missing)}")
        raw_fc = np.asarray(data["delta_fc_full"], float)
        raw_reference = np.asarray(data["reference_positions"], float)
        mapping = np.asarray(data["atom_mapping"], int)
        cell = np.asarray(data["cell"], float)
        degauss = float(np.asarray(data["degauss_Ry"]).reshape(()))
    if mapping.ndim != 1 or sorted(mapping.tolist()) != list(range(len(mapping))):
        raise ValueError("operator atom_mapping is not a permutation")
    if raw_reference.shape != (len(mapping), 3) or cell.shape != (3, 3):
        raise ValueError("operator reference geometry has the wrong shape")
    if raw_fc.shape[:2] != (len(mapping), len(mapping)) or raw_fc.shape[-2:] != (3, 3):
        raise ValueError("operator force constants have the wrong shape")
    return raw_fc[mapping][:, mapping], raw_reference[mapping], cell, degauss


def periodic_assignment(
    left: np.ndarray, right: np.ndarray, cell: np.ndarray
) -> np.ndarray:
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
    phonon = phonopy.load(str(background), produce_fc=False)
    with np.load(thermal_result, allow_pickle=False) as data:
        required = {
            "free_energy_fc2_eV_A2",
            "lattice_temperature_K",
            "operator_temperature_K",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"thermal result is missing arrays: {sorted(missing)}")
        force_constants = np.asarray(data["free_energy_fc2_eV_A2"], float)
        lattice_temperature = int(
            np.asarray(data["lattice_temperature_K"]).reshape(())
        )
        operator_temperature = int(
            np.asarray(data["operator_temperature_K"]).reshape(())
        )
    phonon.force_constants = force_constants
    kpoint = np.asarray(APRIME_MODE_DEFINITION["qpoint_crystal"], float)
    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(kpoint)
    mode_index = int(np.argmax(frequencies))
    primitive_mode = np.asarray(eigenvectors[:, mode_index], complex).reshape(-1, 3)

    supercell = phonopy_to_ase(phonon.supercell)
    super_positions = np.asarray(supercell.positions, float)
    if not np.allclose(
        np.asarray(supercell.cell), operator_cell, atol=2.0e-5, rtol=0.0
    ):
        raise ValueError("phonopy and operator supercells differ")
    operator_to_phonopy = periodic_assignment(
        operator_reference, super_positions, operator_cell
    )

    unit_cell = np.asarray(phonon.unitcell.cell, float)
    unit_scaled = np.asarray(phonon.unitcell.scaled_positions, float)
    representatives = np.asarray(phonon.supercell.u2s_map, int)
    representative_to_basis = {
        int(representative): index
        for index, representative in enumerate(representatives)
    }
    s2u = np.asarray(phonon.supercell.s2u_map, int)
    expanded = np.zeros((len(supercell), 3), complex)
    translations: list[list[int]] = []
    basis_indices: list[int] = []
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
        **APRIME_MODE_DEFINITION,
        "primitive_mode_index": mode_index,
        "primitive_frequency_THz": float(frequencies[mode_index]),
        "primitive_frequency_cm_1": float(frequencies[mode_index] * 33.35641),
        "lattice_temperature_K": lattice_temperature,
        "operator_temperature_K": operator_temperature,
        "operator_to_phonopy_mapping": operator_to_phonopy.tolist(),
        "phonopy_basis_indices": basis_indices,
        "phonopy_cell_translations": translations,
        "observed_normalization": float(
            np.vdot(mode_operator.reshape(-1), mode_operator.reshape(-1)).real
        ),
    }
    return mode_operator, provenance
