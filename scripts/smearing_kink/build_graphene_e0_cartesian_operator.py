#!/usr/bin/env python3
"""Build and gate a 6x6 force-capable Cartesian electronic correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann, electron_volt, physical_constants
from scipy.optimize import linear_sum_assignment

from audit_graphene_q6_fourier_gauge import (
    atom_mapping,
    expected_q_cart,
    load_matdyn_matrices,
    point_group_error,
    primitive_symmetries,
    qgrid,
    realspace_blocks,
    supercell_hessian,
)


MEV_TO_CM1 = 8.06554393734921
CARBON_MASS_AMU = 12.011
AMU_KG = physical_constants["atomic mass constant"][0]
RYDBERG_EV = physical_constants["Rydberg constant times hc in eV"][0]
LIGHT_CM_S = physical_constants["speed of light in vacuum"][0] * 100.0
EV_A2_TO_N_M = electron_volt / 1.0e-20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_qpoints(path: Path, ngrid: int) -> np.ndarray:
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if lines[0] != [str(ngrid * ngrid), "crystal"]:
        raise ValueError(f"unexpected q-point header in {path}: {lines[0]}")
    qpoints = np.asarray([[float(value) for value in row[:3]] for row in lines[1:]])
    weights = np.asarray([float(row[3]) for row in lines[1:]])
    if qpoints.shape != (ngrid * ngrid, 3):
        raise ValueError(f"unexpected q-point shape {qpoints.shape}")
    if not np.allclose(qpoints, qgrid(ngrid), atol=1.0e-11, rtol=0.0):
        raise ValueError("q-point order is not the frozen row-major q grid")
    if not np.isclose(weights.sum(), 1.0, atol=1.0e-9):
        raise ValueError("q-point weights do not sum to one")
    return qpoints


def load_matdyn_frequencies(path: Path, n_qpoints: int) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines or "nbnd=" not in lines[0]:
        raise ValueError(f"unexpected matdyn frequency file {path}")
    rows = []
    cursor = 1
    while cursor < len(lines):
        q_values = lines[cursor].split()
        cursor += 1
        if len(q_values) != 4:
            raise ValueError(f"unexpected q row in {path}: {q_values}")
        frequencies = []
        while cursor < len(lines) and len(frequencies) < 6:
            frequencies.extend(float(value) for value in lines[cursor].split())
            cursor += 1
        if len(frequencies) != 6:
            raise ValueError(f"unexpected frequency row in {path}")
        rows.append(frequencies)
    result = np.asarray(rows)
    if result.shape != (n_qpoints, 6):
        raise ValueError(f"expected {(n_qpoints, 6)} frequencies, found {result.shape}")
    return result


def load_bare_matrix_bundle(
    path: Path, frequency_path: Path | None, ngrid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            qpoints = np.asarray(data["qpoints_crystal"], float)
            matrices = np.asarray(data["dynamical_matrices"], complex)
            frequencies = np.asarray(data["frequencies_cm1"], float)
        expected_shape = (ngrid * ngrid, 6, 6)
        if matrices.shape != expected_shape or frequencies.shape != (ngrid * ngrid, 6):
            raise ValueError("unexpected EPW dynamical-matrix bundle shape")
        if not np.allclose(qpoints, qgrid(ngrid), atol=1.0e-11, rtol=0.0):
            raise ValueError("EPW dynamical-matrix q grid does not match the frozen grid")
        return expected_q_cart(qpoints), matrices, frequencies, "EPW_dynwan2bloch"
    if frequency_path is None:
        raise ValueError("--matdyn-frequencies is required for a QE text matrix file")
    q_cart, matrices = load_matdyn_matrices(path)
    frequencies = load_matdyn_frequencies(frequency_path, len(matrices))
    return q_cart, matrices, frequencies, "QE_matdyn"


def load_self_energy(path: Path, n_qpoints: int, temperature: float) -> dict:
    rows = {}
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 9:
            raise ValueError(f"unexpected self-energy row: {line}")
        iq, mode = int(fields[0]), int(fields[1])
        values = [float(value) for value in fields[2:]]
        if abs(values[3]) > 1.0e-12:
            continue
        key = (iq, mode)
        if key in rows:
            raise ValueError(f"duplicate zero-frequency row {key}")
        rows[key] = {
            "temperature_K": values[0],
            "broadening_eV": values[1],
            "bare_eV": values[2],
            "delta_pi_meV": values[4] - values[5],
        }
    if len(rows) != n_qpoints * 6:
        raise ValueError(f"expected {n_qpoints * 6} static rows, found {len(rows)}")
    temperatures = {round(row["temperature_K"], 8) for row in rows.values()}
    broadenings = {round(row["broadening_eV"], 8) for row in rows.values()}
    if temperatures != {round(float(temperature), 8)}:
        raise ValueError(f"temperature mismatch: {sorted(temperatures)}")
    if broadenings != {0.005}:
        raise ValueError(f"numerical broadening mismatch: {sorted(broadenings)}")
    return rows


def signed_frequency(eigenvalues: np.ndarray, scale: float) -> np.ndarray:
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues) * scale)


def matrix_frequency_scale(matrices: np.ndarray, frequencies: np.ndarray) -> float:
    ratios = []
    for matrix, row in zip(matrices, frequencies, strict=True):
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
        mask = (eigenvalues > 1.0e-9) & (row > 1.0)
        ratios.extend((row[mask] ** 2 / eigenvalues[mask]).tolist())
    if not ratios:
        raise ValueError("cannot determine the dynamical-matrix frequency scale")
    return float(np.median(ratios))


def average_degenerate_shifts(
    frequencies: np.ndarray, shifts: np.ndarray, tolerance_cm1: float = 0.1
) -> tuple[np.ndarray, float]:
    averaged = np.asarray(shifts, float).copy()
    start = 0
    max_change = 0.0
    while start < len(frequencies):
        stop = start + 1
        while stop < len(frequencies) and abs(frequencies[stop] - frequencies[stop - 1]) <= tolerance_cm1:
            stop += 1
        if stop - start > 1:
            value = float(np.mean(averaged[start:stop]))
            max_change = max(max_change, float(np.max(np.abs(averaged[start:stop] - value))))
            averaged[start:stop] = value
        start = stop
    return averaged, max_change


def time_reversal_symmetrize(matrices: np.ndarray, ngrid: int) -> np.ndarray:
    result = np.asarray(matrices, complex).copy().reshape(ngrid, ngrid, 6, 6)
    visited = set()
    for i in range(ngrid):
        for j in range(ngrid):
            pair = ((-i) % ngrid, (-j) % ngrid)
            key = tuple(sorted(((i, j), pair)))
            if key in visited:
                continue
            visited.add(key)
            a = (result[i, j] + result[i, j].conj().T) / 2.0
            b = (result[pair] + result[pair].conj().T) / 2.0
            mean = (a + b.conj()) / 2.0
            result[i, j] = mean
            result[pair] = mean.conj()
    return result.reshape(-1, 6, 6)


def project_gamma_translations(matrices: np.ndarray) -> np.ndarray:
    result = np.asarray(matrices, complex).copy()
    translations = np.zeros((6, 3))
    for atom in range(2):
        translations[3 * atom : 3 * atom + 3, :] = np.eye(3) / np.sqrt(2.0)
    projector = np.eye(6) - translations @ translations.T
    result[0] = projector @ result[0] @ projector
    result[0] = (result[0] + result[0].conj().T) / 2.0
    return result


def symmetrize_point_group(hessian: np.ndarray, ngrid: int) -> tuple[np.ndarray, int]:
    lattice, rotations, translations = primitive_symmetries()
    inverse_lattice_t = np.linalg.inv(lattice.T)
    accumulated = np.zeros_like(hessian, dtype=complex)
    for rotation, translation in zip(rotations, translations, strict=True):
        mapping = atom_mapping(rotation, translation, ngrid)
        cart_rotation = lattice.T @ rotation @ inverse_lattice_t
        for atom_a, mapped_a in enumerate(mapping):
            a = slice(3 * atom_a, 3 * atom_a + 3)
            pa = slice(3 * mapped_a, 3 * mapped_a + 3)
            for atom_b, mapped_b in enumerate(mapping):
                b = slice(3 * atom_b, 3 * atom_b + 3)
                pb = slice(3 * mapped_b, 3 * mapped_b + 3)
                accumulated[pa, pb] += cart_rotation @ hessian[a, b] @ cart_rotation.T
    accumulated /= len(rotations)
    return (accumulated + accumulated.conj().T) / 2.0, len(rotations)


def blocks_from_hessian(hessian: np.ndarray, ngrid: int) -> np.ndarray:
    blocks = np.zeros((ngrid, ngrid, 6, 6), complex)
    counts = np.zeros((ngrid, ngrid), int)
    for i in range(ngrid):
        for j in range(ngrid):
            cell_a = i * ngrid + j
            for k in range(ngrid):
                for l in range(ngrid):
                    cell_b = k * ngrid + l
                    delta = ((i - k) % ngrid, (j - l) % ngrid)
                    blocks[delta] += hessian[
                        6 * cell_a : 6 * cell_a + 6,
                        6 * cell_b : 6 * cell_b + 6,
                    ]
                    counts[delta] += 1
    blocks /= counts[:, :, None, None]
    return blocks


def qe_to_e1_orientation(hessian: np.ndarray, ngrid: int) -> np.ndarray:
    tau_qe = np.asarray([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]])
    tau_e1 = np.asarray([[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.0]])
    mapping = np.full(2 * ngrid * ngrid, -1, int)
    for i in range(ngrid):
        for j in range(ngrid):
            for atom, tau in enumerate(tau_qe):
                source = 2 * (i * ngrid + j) + atom
                transformed = -(np.asarray([i, j, 0.0]) + tau)
                for target_atom, target_tau in enumerate(tau_e1):
                    cell = transformed - target_tau
                    rounded = np.rint(cell)
                    if np.max(np.abs(cell - rounded)) < 1.0e-10:
                        ii, jj = int(rounded[0]) % ngrid, int(rounded[1]) % ngrid
                        mapping[source] = 2 * (ii * ngrid + jj) + target_atom
                        break
    if np.any(mapping < 0) or len(set(mapping.tolist())) != len(mapping):
        raise RuntimeError("failed to map QE orientation to E1 orientation")
    output = np.zeros_like(hessian)
    inversion = -np.eye(3)
    for source_a, target_a in enumerate(mapping):
        a = slice(3 * source_a, 3 * source_a + 3)
        ta = slice(3 * target_a, 3 * target_a + 3)
        for source_b, target_b in enumerate(mapping):
            b = slice(3 * source_b, 3 * source_b + 3)
            tb = slice(3 * target_b, 3 * target_b + 3)
            output[ta, tb] = inversion @ hessian[a, b] @ inversion.T
    return output


def reorder_to_reference(
    hessian: np.ndarray, ngrid: int, reference_operator: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(reference_operator, allow_pickle=False) as data:
        raw_reference = np.asarray(data["reference_positions"], float)
        cell = np.asarray(data["cell"], float)
        old_mapping = np.asarray(data["atom_mapping"], int)
    reference = raw_reference[old_mapping]
    if len(reference) != 2 * ngrid * ngrid:
        raise ValueError("reference operator does not contain 72 atoms")
    tau = np.asarray([[0.0, 0.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    target_fractional = []
    for i in range(ngrid):
        for j in range(ngrid):
            for basis in tau:
                target_fractional.append(
                    [(i + basis[0]) / ngrid, (j + basis[1]) / ngrid, basis[2]]
                )
    target = np.asarray(target_fractional) @ cell
    ref_fractional = reference @ np.linalg.inv(cell)
    target_fractional = np.asarray(target_fractional)
    cost = np.empty((len(target), len(reference)))
    for index, fractional in enumerate(target_fractional):
        difference = ref_fractional - fractional
        difference -= np.round(difference)
        cost[index] = np.linalg.norm(difference @ cell, axis=1)
    target_indices, reference_indices = linear_sum_assignment(cost)
    if not np.array_equal(target_indices, np.arange(len(target))):
        raise RuntimeError("unexpected assignment ordering")
    max_distance = float(cost[target_indices, reference_indices].max())
    if max_distance > 1.0e-5:
        raise ValueError(f"reference geometry mismatch: {max_distance} A")
    dof_reference = np.concatenate(
        [np.arange(3 * index, 3 * index + 3) for index in reference_indices]
    )
    reordered = np.zeros_like(hessian)
    reordered[np.ix_(dof_reference, dof_reference)] = hessian
    return reordered, reference, cell, max_distance


def force_constants_eV_A2(hessian: np.ndarray, scale_cm2: float) -> np.ndarray:
    angular_scale = (2.0 * np.pi * LIGHT_CM_S) ** 2 * scale_cm2
    mass_kg = CARBON_MASS_AMU * AMU_KG
    conversion = mass_kg * angular_scale / EV_A2_TO_N_M
    return np.asarray(hessian.real * conversion, float)


def energy_force_gradient_error(force_constants: np.ndarray) -> float:
    rng = np.random.default_rng(20260807)
    displacement = rng.normal(scale=0.01, size=force_constants.shape[0])
    gradient = force_constants @ displacement

    def energy(vector: np.ndarray) -> float:
        return 0.5 * float(vector @ force_constants @ vector)

    epsilon = 1.0e-6
    errors = []
    for index in rng.choice(len(displacement), size=12, replace=False):
        plus, minus = displacement.copy(), displacement.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (energy(plus) - energy(minus)) / (2.0 * epsilon)
        errors.append(abs(numerical - gradient[index]))
    return float(max(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpoints", type=Path, required=True)
    parser.add_argument("--static-self-energy", type=Path, required=True)
    parser.add_argument("--dynamical-matrices", type=Path, required=True)
    parser.add_argument("--matdyn-frequencies", type=Path)
    parser.add_argument("--dynamical-matrix-audit", type=Path)
    parser.add_argument("--full-q-audit", type=Path, required=True)
    parser.add_argument("--reference-operator", type=Path, required=True)
    parser.add_argument("--temperature-K", type=float, required=True)
    parser.add_argument("--ngrid", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.full_q_audit.read_text())
    if audit.get("status") != "ready" or audit.get("releases_E1") is not False:
        raise ValueError("full-q data audit is not a non-releasing ready record")
    if int(audit["ngrid"]) != args.ngrid:
        raise ValueError("full-q audit grid mismatch")
    if abs(float(audit["target_temperature_K"]) - args.temperature_K) > 1.0e-8:
        raise ValueError("full-q audit temperature mismatch")

    qpoints = load_qpoints(args.qpoints, args.ngrid)
    q_cart, bare_matrices, matrix_frequency_reference, matrix_source = (
        load_bare_matrix_bundle(
            args.dynamical_matrices, args.matdyn_frequencies, args.ngrid
        )
    )
    if args.dynamical_matrices.suffix == ".npz":
        if args.dynamical_matrix_audit is None:
            raise ValueError("EPW matrix bundles require --dynamical-matrix-audit")
        matrix_audit = json.loads(args.dynamical_matrix_audit.read_text())
        if matrix_audit.get("status") != "passed":
            raise ValueError("EPW dynamical-matrix reconstruction audit did not pass")
        if matrix_audit.get("output", {}).get("sha256") != sha256(
            args.dynamical_matrices
        ):
            raise ValueError("EPW dynamical-matrix reconstruction hash mismatch")
    if len(bare_matrices) != len(qpoints):
        raise ValueError("dynamical-matrix count mismatch")
    q_coordinate_error = float(np.max(np.abs(q_cart - expected_q_cart(qpoints))))
    bare_matrices = (bare_matrices + bare_matrices.conj().transpose(0, 2, 1)) / 2.0
    scale = matrix_frequency_scale(bare_matrices, matrix_frequency_reference)
    matrix_frequency_rows = []
    for matrix in bare_matrices:
        matrix_frequency_rows.append(signed_frequency(np.linalg.eigvalsh(matrix), scale))
    matrix_frequencies = np.asarray(matrix_frequency_rows)
    matrix_frequency_error = float(
        np.max(np.abs(matrix_frequencies - matrix_frequency_reference))
    )

    self_energy = load_self_energy(
        args.static_self_energy, len(qpoints), args.temperature_K
    )
    epw_bare = np.empty_like(matrix_frequency_reference)
    raw_shifts = np.empty_like(matrix_frequency_reference)
    for iq in range(1, len(qpoints) + 1):
        for mode in range(1, 7):
            row = self_energy[(iq, mode)]
            bare_meV = row["bare_eV"] * 1000.0
            epw_bare[iq - 1, mode - 1] = bare_meV * MEV_TO_CM1
            raw_shifts[iq - 1, mode - 1] = (
                2.0 * bare_meV * row["delta_pi_meV"] * MEV_TO_CM1**2
            )
    bare_frequency_match = float(
        np.max(np.abs(epw_bare - matrix_frequency_reference))
    )

    preliminary = []
    averaged_shifts = []
    max_degenerate_shift_change = 0.0
    for matrix, frequencies, shifts in zip(
        bare_matrices, matrix_frequency_reference, raw_shifts, strict=True
    ):
        shifts, change = average_degenerate_shifts(frequencies, shifts)
        max_degenerate_shift_change = max(max_degenerate_shift_change, change)
        _, eigenvectors = np.linalg.eigh(matrix)
        correction = eigenvectors @ np.diag(shifts / scale) @ eigenvectors.conj().T
        preliminary.append((correction + correction.conj().T) / 2.0)
        averaged_shifts.append(shifts)
    preliminary = project_gamma_translations(
        time_reversal_symmetrize(np.asarray(preliminary), args.ngrid)
    )
    averaged_shifts = np.asarray(averaged_shifts)

    pre_blocks = realspace_blocks(preliminary, args.ngrid)
    pre_hessian = supercell_hessian(pre_blocks, args.ngrid)
    pre_imaginary = float(np.max(np.abs(pre_hessian.imag)))
    sym_hessian, n_operations = symmetrize_point_group(pre_hessian.real, args.ngrid)
    final_blocks = blocks_from_hessian(sym_hessian, args.ngrid)
    final_q = np.fft.fft2(final_blocks, axes=(0, 1)).reshape(-1, 6, 6)
    final_q = project_gamma_translations(
        time_reversal_symmetrize(final_q, args.ngrid)
    )
    final_blocks = realspace_blocks(final_q, args.ngrid)
    final_hessian = supercell_hessian(final_blocks, args.ngrid)
    final_hessian = (final_hessian + final_hessian.conj().T) / 2.0

    corrected_formula = np.sign(matrix_frequency_reference**2 + averaged_shifts) * np.sqrt(
        np.abs(matrix_frequency_reference**2 + averaged_shifts)
    )
    corrected_matrix = []
    for bare, correction in zip(bare_matrices, final_q, strict=True):
        corrected_matrix.append(
            signed_frequency(np.linalg.eigvalsh(bare + correction), scale)
        )
    corrected_matrix = np.asarray(corrected_matrix)
    symmetry_projection_frequency_change = float(
        np.max(np.abs(corrected_matrix - corrected_formula))
    )

    replay_q = np.fft.fft2(final_blocks, axes=(0, 1)).reshape(final_q.shape)
    replay_matrix_error = float(np.max(np.abs(replay_q - final_q)))
    replay_frequency_error = 0.0
    replay_nonzero_frequency_error = 0.0
    replay_top_frequency_error = 0.0
    for bare, left, right in zip(bare_matrices, final_q, replay_q, strict=True):
        left_frequency = signed_frequency(np.linalg.eigvalsh(bare + left), scale)
        right_frequency = signed_frequency(np.linalg.eigvalsh(bare + right), scale)
        difference = np.abs(left_frequency - right_frequency)
        replay_frequency_error = max(
            replay_frequency_error,
            float(np.max(difference)),
        )
        nonzero = np.maximum(np.abs(left_frequency), np.abs(right_frequency)) > 1.0
        if np.any(nonzero):
            replay_nonzero_frequency_error = max(
                replay_nonzero_frequency_error, float(np.max(difference[nonzero]))
            )
        replay_top_frequency_error = max(
            replay_top_frequency_error, float(difference[-1])
        )

    time_reversal_error = 0.0
    shaped = final_q.reshape(args.ngrid, args.ngrid, 6, 6)
    for i in range(args.ngrid):
        for j in range(args.ngrid):
            time_reversal_error = max(
                time_reversal_error,
                float(
                    np.max(
                        np.abs(
                            shaped[(-i) % args.ngrid, (-j) % args.ngrid]
                            - shaped[i, j].conj()
                        )
                    )
                ),
            )
    point_symmetry_error, _ = point_group_error(final_hessian.real, args.ngrid)
    hessian_imaginary = float(np.max(np.abs(final_hessian.imag)))
    hessian_pair_error = float(np.max(np.abs(final_hessian - final_hessian.conj().T)))
    hessian_row_sum = float(np.max(np.abs(final_hessian.sum(axis=1))))

    oriented = qe_to_e1_orientation(final_hessian.real, args.ngrid)
    reordered, reference, cell, reference_match = reorder_to_reference(
        oriented, args.ngrid, args.reference_operator
    )
    force_hessian = force_constants_eV_A2(reordered, scale)
    force_pair_error = float(np.max(np.abs(force_hessian - force_hessian.T)))
    force_row_sum = float(np.max(np.abs(force_hessian.sum(axis=1))))
    gradient_error = energy_force_gradient_error(force_hessian)
    force_constants = force_hessian.reshape(72, 3, 72, 3).transpose(0, 2, 1, 3)

    degauss_ry = Boltzmann * args.temperature_K / electron_volt / RYDBERG_EV
    checks = {
        "q_coordinate_max_le_1e-8": q_coordinate_error <= 1.0e-8,
        "matrix_frequency_replay_max_le_0p2_cm-1": matrix_frequency_error <= 0.2,
        "EPW_bare_matches_matrix_basis_max_le_0p5_cm-1": bare_frequency_match <= 0.5,
        "pre_symmetry_realspace_imaginary_max_le_1e-8": pre_imaginary <= 1.0e-8,
        "symmetry_projection_frequency_change_max_le_0p5_cm-1": (
            symmetry_projection_frequency_change <= 0.5
        ),
        "time_reversal_max_le_1e-10": time_reversal_error <= 1.0e-10,
        "point_group_max_le_1e-8": point_symmetry_error <= 1.0e-8,
        "hessian_imaginary_max_le_1e-10": hessian_imaginary <= 1.0e-10,
        "hessian_pair_max_le_1e-10": hessian_pair_error <= 1.0e-10,
        "hessian_ASR_max_le_1e-10": hessian_row_sum <= 1.0e-10,
        "q_real_q_matrix_replay_max_le_1e-10": replay_matrix_error <= 1.0e-10,
        "q_real_q_nonzero_frequency_replay_max_lt_1e-5_cm-1": (
            replay_nonzero_frequency_error < 1.0e-5
        ),
        "q_real_q_top_branch_replay_max_lt_1e-5_cm-1": (
            replay_top_frequency_error < 1.0e-5
        ),
        "force_pair_max_le_1e-8_eV_A2": force_pair_error <= 1.0e-8,
        "force_ASR_max_le_1e-8_eV_A2": force_row_sum <= 1.0e-8,
        "analytic_gradient_max_le_1e-7_eV_A": gradient_error <= 1.0e-7,
        "reference_geometry_match_max_le_1e-5_A": reference_match <= 1.0e-5,
    }
    passed = all(checks.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    operator_path = args.output_dir / f"T{int(round(args.temperature_K))}_operator.npz"
    atomic_npz(
        operator_path,
        temperature_K=float(args.temperature_K),
        lattice_temperature_K=float(args.temperature_K),
        degauss_Ry=degauss_ry,
        degauss_formula="k_B*T/Ry",
        delta_fc_full=force_constants,
        reference_positions=reference,
        cell=cell,
        atom_mapping=np.arange(72, dtype=int),
        qpoints_crystal=qpoints,
        delta_dynamical_q=final_q,
        delta_dynamical_realspace=final_blocks,
        dynamical_matrix_scale_cm2=scale,
    )
    summary = {
        "status": "passed" if passed else "failed",
        "scope": (
            f"E0 Cartesian Hermitian q6 electronic operator at "
            f"{args.temperature_K:g} K"
        ),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "temperature_K": float(args.temperature_K),
        "degauss_Ry_from_kBT": degauss_ry,
        "bare_dynamical_matrix_source": matrix_source,
        "matrix_scale_cm-2_per_QE_unit": scale,
        "n_qpoints": len(qpoints),
        "n_space_group_operations": n_operations,
        "metrics": {
            "q_coordinate_max_abs": q_coordinate_error,
            "matrix_frequency_replay_max_abs_cm-1": matrix_frequency_error,
            "EPW_bare_matrix_basis_max_abs_cm-1": bare_frequency_match,
            "degenerate_shift_averaging_max_abs_cm-2": max_degenerate_shift_change,
            "pre_symmetry_realspace_imaginary_max_abs": pre_imaginary,
            "symmetry_projection_frequency_change_max_abs_cm-1": (
                symmetry_projection_frequency_change
            ),
            "time_reversal_max_abs": time_reversal_error,
            "point_group_max_abs": point_symmetry_error,
            "hessian_imaginary_max_abs": hessian_imaginary,
            "hessian_pair_max_abs": hessian_pair_error,
            "hessian_ASR_row_sum_max_abs": hessian_row_sum,
            "q_real_q_matrix_replay_max_abs": replay_matrix_error,
            "q_real_q_all_mode_frequency_replay_max_abs_cm-1": (
                replay_frequency_error
            ),
            "q_real_q_nonzero_frequency_replay_max_abs_cm-1": (
                replay_nonzero_frequency_error
            ),
            "q_real_q_top_branch_replay_max_abs_cm-1": replay_top_frequency_error,
            "force_pair_max_abs_eV_A2": force_pair_error,
            "force_ASR_row_sum_max_abs_eV_A2": force_row_sum,
            "analytic_gradient_max_abs_eV_A": gradient_error,
            "reference_geometry_match_max_abs_A": reference_match,
        },
        "checks": checks,
        "operator": {
            "path": str(operator_path),
            "sha256": sha256(operator_path),
        },
        "sources": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (
                args.qpoints,
                args.static_self_energy,
                args.dynamical_matrices,
                args.matdyn_frequencies,
                args.dynamical_matrix_audit,
                args.full_q_audit,
                args.reference_operator,
            )
            if path is not None
        ],
        "releases_E1": False,
        "next_stage": (
            "aggregate_300_450_600_operator_gates"
            if passed
            else "stop_and_diagnose_cartesian_operator"
        ),
    }
    atomic_json(args.output_dir / f"T{int(round(args.temperature_K))}_operator_gate.json", summary)
    marker = args.output_dir / (
        f"T{int(round(args.temperature_K))}_OPERATOR_PASS"
        if passed
        else f"T{int(round(args.temperature_K))}_OPERATOR_FAIL"
    )
    opposite_marker = args.output_dir / (
        f"T{int(round(args.temperature_K))}_OPERATOR_FAIL"
        if passed
        else f"T{int(round(args.temperature_K))}_OPERATOR_PASS"
    )
    opposite_marker.unlink(missing_ok=True)
    marker_temporary = marker.with_name(marker.name + ".tmp")
    marker_temporary.write_text(summary["built_at_utc"] + "\n")
    os.replace(marker_temporary, marker)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
