#!/usr/bin/env python3
"""Reconstruct EPW fine-q dynamical matrices from its formatted restart."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.constants import physical_constants


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
COMPLEX_RE = re.compile(rf"\(\s*({FLOAT})\s*,\s*({FLOAT})\s*\)")
RYDBERG_CM1 = physical_constants["Rydberg constant"][0] / 100.0
BOHR_ANGSTROM = physical_constants["Bohr radius"][0] * 1.0e10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def floats(line: str) -> list[float]:
    return [
        float(value.replace("D", "E").replace("d", "e"))
        for value in re.findall(FLOAT, line)
    ]


def complex_value(line: str) -> complex:
    match = COMPLEX_RE.fullmatch(line.strip())
    if match is None:
        raise ValueError(f"unexpected Fortran complex value: {line}")
    return complex(
        float(match.group(1).replace("D", "E").replace("d", "e")),
        float(match.group(2).replace("D", "E").replace("d", "e")),
    )


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_crystal(path: Path, natoms: int) -> dict:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if int(lines[0]) != natoms or int(lines[1]) != 3 * natoms:
        raise ValueError("crystal.fmt atom or mode count mismatch")
    at = np.asarray(floats(lines[3]), float).reshape(3, 3, order="F")
    alat_bohr = float(lines[6])
    masses = np.asarray(floats(lines[8]), float)
    atom_types = np.asarray([int(value) for value in lines[9].split()], int)
    if atom_types.shape != (natoms,):
        raise ValueError("crystal.fmt atom-type count mismatch")
    if np.any(atom_types < 1) or np.any(atom_types > len(masses)):
        raise ValueError("crystal.fmt contains invalid atom types")
    return {
        "at": at,
        "alat_bohr": alat_bohr,
        "masses_au": masses,
        "atom_types": atom_types,
    }


def load_rdw(path: Path) -> tuple[np.ndarray, dict]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    dimensions = [int(value) for value in lines[1].split()]
    if len(dimensions) != 5:
        raise ValueError("unexpected epwdata.fmt dimension header")
    nbndsub, nrr_k, nmodes, nrr_q, nrr_g = dimensions
    cursor = 3 + nbndsub * nbndsub * nrr_k
    count = nmodes * nmodes * nrr_q
    if cursor + count > len(lines):
        raise ValueError("epwdata.fmt is truncated before rdw")
    values = [complex_value(line) for line in lines[cursor : cursor + count]]
    rdw = np.asarray(values, complex).reshape(nmodes, nmodes, nrr_q)
    return rdw, {
        "nbndsub": nbndsub,
        "nrr_k": nrr_k,
        "nmodes": nmodes,
        "nrr_q": nrr_q,
        "nrr_g": nrr_g,
    }


def zone_centered_wigner(
    at: np.ndarray, mesh: tuple[int, int, int], tolerance: float = 1.0e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric = at.T @ at
    translations = np.asarray(
        [
            [i * mesh[0], j * mesh[1], k * mesh[2]]
            for i in range(-2, 3)
            for j in range(-2, 3)
            for k in range(-2, 3)
        ],
        float,
    )
    zero_translation = int(
        np.flatnonzero(np.all(translations == 0.0, axis=1))[0]
    )
    vectors, degeneracies, lengths = [], [], []
    for n1 in range(-2 * mesh[0], 2 * mesh[0] + 1):
        for n2 in range(-2 * mesh[1], 2 * mesh[1] + 1):
            for n3 in range(-2 * mesh[2], 2 * mesh[2] + 1):
                vector = np.asarray([n1, n2, n3], float)
                differences = vector - translations
                distances = np.einsum(
                    "ni,ij,nj->n", differences, metric, differences, optimize=True
                )
                minimum = float(np.min(distances))
                tied = np.abs(distances - minimum) < tolerance
                if tied[zero_translation]:
                    vectors.append([n1, n2, n3])
                    degeneracies.append(int(np.count_nonzero(tied)))
                    lengths.append(float(np.sqrt(vector @ metric @ vector)))
    result_vectors = np.asarray(vectors, int)
    result_degeneracies = np.asarray(degeneracies, int)
    result_lengths = np.asarray(lengths, float)
    expected_weight = int(np.prod(mesh))
    if abs(np.sum(1.0 / result_degeneracies) - expected_weight) > tolerance:
        raise ValueError("Wigner-Seitz weights do not sum to the coarse mesh size")
    return result_vectors, result_degeneracies, result_lengths


def load_decay(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            values = floats(stripped)
            if len(values) != 2:
                raise ValueError(f"unexpected decay.dynmat row: {line}")
            rows.append(values)
    data = np.asarray(rows, float)
    return data[:, 0], data[:, 1]


def load_qpoints(path: Path) -> np.ndarray:
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    count = int(lines[0][0])
    if lines[0][1] != "crystal" or len(lines) != count + 1:
        raise ValueError("unexpected q-point file")
    return np.asarray([[float(value) for value in row[:3]] for row in lines[1:]])


def load_epw_bare_frequencies(path: Path, n_qpoints: int) -> np.ndarray:
    rows = {}
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 9:
            raise ValueError(f"unexpected specfun_sup.phon row: {line}")
        iq, mode = int(fields[0]), int(fields[1])
        values = [float(value) for value in fields[2:]]
        if abs(values[3]) <= 1.0e-12:
            rows[(iq, mode)] = values[2] * 1000.0 * 8.06554393734921
    if len(rows) != n_qpoints * 6:
        raise ValueError("static self-energy file is incomplete")
    return np.asarray(
        [[rows[(iq, mode)] for mode in range(1, 7)] for iq in range(1, n_qpoints + 1)]
    )


def reconstruct_matrices(
    rdw: np.ndarray,
    vectors: np.ndarray,
    degeneracies: np.ndarray,
    qpoints: np.ndarray,
    masses: np.ndarray,
    atom_types: np.ndarray,
) -> np.ndarray:
    nmodes = rdw.shape[0]
    matrices = []
    for qpoint in qpoints:
        phases = np.exp(2j * np.pi * (vectors @ qpoint)) / degeneracies
        matrix = np.einsum("abr,r->ab", rdw, phases, optimize=True)
        for atom_a, type_a in enumerate(atom_types):
            for atom_b, type_b in enumerate(atom_types):
                factor = np.sqrt(masses[type_a - 1] * masses[type_b - 1])
                matrix[
                    3 * atom_a : 3 * atom_a + 3,
                    3 * atom_b : 3 * atom_b + 3,
                ] /= factor
        matrix = (matrix + matrix.conj().T) / 2.0
        if matrix.shape != (nmodes, nmodes):
            raise RuntimeError("internal dynamical-matrix shape error")
        matrices.append(matrix)
    return np.asarray(matrices)


def signed_frequencies(matrices: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(matrices)
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * RYDBERG_CM1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epwdata", type=Path, required=True)
    parser.add_argument("--crystal", type=Path, required=True)
    parser.add_argument("--decay-dynmat", type=Path, required=True)
    parser.add_argument("--qpoints", type=Path, required=True)
    parser.add_argument("--static-self-energy", type=Path, required=True)
    parser.add_argument("--nqc1", type=int, required=True)
    parser.add_argument("--nqc2", type=int, required=True)
    parser.add_argument("--nqc3", type=int, default=1)
    parser.add_argument("--ngrid", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    crystal = load_crystal(args.crystal, natoms=2)
    rdw, dimensions = load_rdw(args.epwdata)
    if dimensions["nmodes"] != 6:
        raise ValueError("this extractor expects the two-atom graphene primitive cell")
    mesh = (args.nqc1, args.nqc2, args.nqc3)
    vectors, degeneracies, lengths = zone_centered_wigner(crystal["at"], mesh)
    if len(vectors) != dimensions["nrr_q"]:
        raise ValueError(
            f"reconstructed {len(vectors)} WS vectors; restart contains "
            f"{dimensions['nrr_q']}"
        )
    decay_lengths, decay_maxima = load_decay(args.decay_dynmat)
    reconstructed_lengths = (
        lengths * crystal["alat_bohr"] * BOHR_ANGSTROM
    )
    reconstructed_maxima = np.max(np.abs(rdw), axis=(0, 1))
    if decay_lengths.shape != reconstructed_lengths.shape:
        raise ValueError("decay.dynmat row count mismatch")
    length_error = float(np.max(np.abs(decay_lengths - reconstructed_lengths)))
    maximum_error = float(np.max(np.abs(decay_maxima - reconstructed_maxima)))

    qpoints = load_qpoints(args.qpoints)
    matrices = reconstruct_matrices(
        rdw,
        vectors,
        degeneracies,
        qpoints,
        crystal["masses_au"],
        crystal["atom_types"],
    )
    frequencies = signed_frequencies(matrices)
    epw_bare = load_epw_bare_frequencies(args.static_self_energy, len(qpoints))
    frequency_error = float(np.max(np.abs(frequencies - epw_bare)))
    q_minus_q_error = 0.0
    output_mesh = np.asarray([args.ngrid, args.ngrid, 1], int)
    q_lookup = {
        tuple(np.mod(np.rint(point * output_mesh), output_mesh).astype(int)): index
        for index, point in enumerate(qpoints)
    }
    for index, point in enumerate(qpoints):
        key = tuple(np.mod(-np.rint(point * output_mesh), output_mesh).astype(int))
        paired = q_lookup[key]
        q_minus_q_error = max(
            q_minus_q_error,
            float(np.max(np.abs(matrices[paired] - matrices[index].conj()))),
        )
    hermitian_error = float(
        np.max(np.abs(matrices - matrices.conj().transpose(0, 2, 1)))
    )
    checks = {
        "decay_length_max_le_1e-7_A": length_error <= 1.0e-7,
        "decay_matrix_max_le_1e-12_Ry": maximum_error <= 1.0e-12,
        # specfun_sup.phon prints the bare energy to 1e-5 eV, or 0.0807 cm^-1.
        "EPW_bare_frequency_max_le_0p1_cm-1": frequency_error <= 0.1,
        "Hermitian_max_le_1e-12": hermitian_error <= 1.0e-12,
        "q_minus_q_max_le_1e-10": q_minus_q_error <= 1.0e-10,
    }
    passed = all(checks.values())
    atomic_npz(
        args.output,
        qpoints_crystal=qpoints,
        dynamical_matrices=matrices,
        frequencies_cm1=frequencies,
        epw_bare_frequencies_cm1=epw_bare,
        wigner_vectors=vectors,
        wigner_degeneracies=degeneracies,
        dynamical_matrix_scale_cm2=RYDBERG_CM1**2,
    )
    payload = {
        "status": "passed" if passed else "failed",
        "scope": "exact reconstruction of EPW dynwan2bloch matrices on the q6 grid",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "coarse_q_mesh": list(mesh),
        "n_qpoints": len(qpoints),
        "n_wigner_vectors": len(vectors),
        "wigner_weight_sum": float(np.sum(1.0 / degeneracies)),
        "metrics": {
            "decay_length_max_abs_A": length_error,
            "decay_matrix_max_abs_Ry": maximum_error,
            "EPW_bare_frequency_max_abs_cm-1": frequency_error,
            "Hermitian_max_abs": hermitian_error,
            "q_minus_q_max_abs": q_minus_q_error,
        },
        "checks": checks,
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    atomic_json(args.audit, payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
