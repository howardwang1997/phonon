#!/usr/bin/env python3
"""Identify and audit the QE dynamical-matrix Fourier gauge on a q6 mesh."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import spglib


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
Q_RE = re.compile(
    rf"q\s*=\s*\(\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*\)"
)


def numbers(line: str) -> list[float]:
    return [float(item.replace("D", "E").replace("d", "e")) for item in re.findall(FLOAT, line)]


def load_matdyn_matrices(path: Path, natoms: int = 2) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    qpoints, matrices = [], []
    cursor = 0
    while cursor < len(lines):
        if "Dynamical" not in lines[cursor] or "Matrix in cartesian axes" not in lines[cursor]:
            cursor += 1
            continue
        cursor += 1
        while cursor < len(lines) and Q_RE.search(lines[cursor]) is None:
            cursor += 1
        if cursor == len(lines):
            raise ValueError(f"truncated q block in {path}")
        match = Q_RE.search(lines[cursor])
        assert match is not None
        qpoints.append([float(match.group(i).replace("D", "E")) for i in range(1, 4)])
        cursor += 1
        matrix = np.zeros((3 * natoms, 3 * natoms), complex)
        blocks = 0
        while cursor < len(lines) and blocks < natoms * natoms:
            block = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", lines[cursor])
            cursor += 1
            if block is None:
                continue
            atom_i, atom_j = int(block.group(1)) - 1, int(block.group(2)) - 1
            rows = []
            for _ in range(3):
                values = numbers(lines[cursor])
                cursor += 1
                if len(values) != 6:
                    raise ValueError(f"bad matrix row in {path}: {lines[cursor - 1]}")
                rows.append([complex(values[k], values[k + 1]) for k in range(0, 6, 2)])
            matrix[3 * atom_i : 3 * atom_i + 3, 3 * atom_j : 3 * atom_j + 3] = rows
            blocks += 1
        if blocks != natoms * natoms:
            raise ValueError(f"incomplete matrix block in {path}")
        matrices.append(matrix)
    if not matrices:
        raise ValueError(f"no matrices in {path}")
    return np.asarray(qpoints), np.asarray(matrices)


def qgrid(ngrid: int) -> np.ndarray:
    return np.asarray(
        [[i / ngrid, j / ngrid, 0.0] for i in range(ngrid) for j in range(ngrid)]
    )


def expected_q_cart(q_crystal: np.ndarray) -> np.ndarray:
    result = np.zeros_like(q_crystal)
    result[:, 0] = q_crystal[:, 0]
    result[:, 1] = (q_crystal[:, 0] + 2.0 * q_crystal[:, 1]) / np.sqrt(3.0)
    return result


def gauge_matrices(
    matrices: np.ndarray, q_crystal: np.ndarray, convention: str
) -> np.ndarray:
    tau = np.asarray([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]])
    output = []
    for matrix, qpoint in zip(matrices, q_crystal, strict=True):
        phases = np.repeat(np.exp(2j * np.pi * (tau @ qpoint)), 3)
        gauge = np.diag(phases)
        if convention == "cell":
            transformed = matrix
        elif convention == "left_basis_phase":
            transformed = gauge @ matrix @ gauge.conj().T
        elif convention == "right_basis_phase":
            transformed = gauge.conj().T @ matrix @ gauge
        else:
            raise ValueError(convention)
        output.append((transformed + transformed.conj().T) / 2.0)
    return np.asarray(output)


def realspace_blocks(matrices: np.ndarray, ngrid: int) -> np.ndarray:
    shaped = matrices.reshape(ngrid, ngrid, 6, 6)
    return np.fft.ifft2(shaped, axes=(0, 1))


def supercell_hessian(blocks: np.ndarray, ngrid: int) -> np.ndarray:
    ndof = ngrid * ngrid * 6
    hessian = np.zeros((ndof, ndof), complex)
    for i in range(ngrid):
        for j in range(ngrid):
            cell_a = i * ngrid + j
            for k in range(ngrid):
                for l in range(ngrid):
                    cell_b = k * ngrid + l
                    delta = ((i - k) % ngrid, (j - l) % ngrid)
                    hessian[
                        6 * cell_a : 6 * cell_a + 6,
                        6 * cell_b : 6 * cell_b + 6,
                    ] = blocks[delta]
    return hessian


def primitive_symmetries() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lattice = np.asarray(
        [[2.46, 0.0, 0.0], [-1.23, 2.46 * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 15.0]]
    )
    tau = np.asarray([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]])
    symmetry = spglib.get_symmetry((lattice, tau, [6, 6]), symprec=1.0e-7)
    if symmetry is None:
        raise RuntimeError("spglib did not find graphene symmetry")
    return lattice, symmetry["rotations"], symmetry["translations"]


def atom_mapping(rotation: np.ndarray, translation: np.ndarray, ngrid: int) -> np.ndarray:
    tau = np.asarray([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]])
    mapping = np.full(2 * ngrid * ngrid, -1, int)
    for i in range(ngrid):
        for j in range(ngrid):
            for atom, position in enumerate(tau):
                source = 2 * (i * ngrid + j) + atom
                fractional = np.asarray([i, j, 0.0]) + position
                transformed = rotation @ fractional + translation
                for target_atom, target_position in enumerate(tau):
                    cell = transformed - target_position
                    rounded = np.rint(cell)
                    if np.max(np.abs(cell - rounded)) < 1.0e-7:
                        ii, jj = int(rounded[0]) % ngrid, int(rounded[1]) % ngrid
                        mapping[source] = 2 * (ii * ngrid + jj) + target_atom
                        break
    if np.any(mapping < 0) or len(set(mapping.tolist())) != len(mapping):
        raise RuntimeError("symmetry operation did not produce an atom permutation")
    return mapping


def point_group_error(hessian: np.ndarray, ngrid: int) -> tuple[float, int]:
    lattice, rotations, translations = primitive_symmetries()
    inverse_lattice_t = np.linalg.inv(lattice.T)
    max_error = 0.0
    for rotation, translation in zip(rotations, translations, strict=True):
        mapping = atom_mapping(rotation, translation, ngrid)
        cart_rotation = lattice.T @ rotation @ inverse_lattice_t
        for atom_a, mapped_a in enumerate(mapping):
            a = slice(3 * atom_a, 3 * atom_a + 3)
            pa = slice(3 * mapped_a, 3 * mapped_a + 3)
            for atom_b, mapped_b in enumerate(mapping):
                b = slice(3 * atom_b, 3 * atom_b + 3)
                pb = slice(3 * mapped_b, 3 * mapped_b + 3)
                expected = cart_rotation @ hessian[a, b] @ cart_rotation.T
                max_error = max(max_error, float(np.max(np.abs(hessian[pa, pb] - expected))))
    return max_error, len(rotations)


def metrics(matrices: np.ndarray, ngrid: int) -> dict:
    blocks = realspace_blocks(matrices, ngrid)
    hessian = supercell_hessian(blocks, ngrid)
    hermitian = float(np.max(np.abs(hessian - hessian.conj().T)))
    imaginary = float(np.max(np.abs(hessian.imag)))
    row_sum = float(np.max(np.abs(hessian.sum(axis=1))))
    replay = np.fft.fft2(blocks, axes=(0, 1)).reshape(matrices.shape)
    replay_error = float(np.max(np.abs(replay - matrices)))
    symmetry_error, n_symmetry = point_group_error(hessian.real, ngrid)
    return {
        "supercell_Hermitian_max_abs": hermitian,
        "supercell_imaginary_max_abs": imaginary,
        "supercell_row_sum_max_abs": row_sum,
        "q_real_q_replay_max_abs": replay_error,
        "point_group_max_abs": symmetry_error,
        "n_space_group_operations": n_symmetry,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamical-matrices", type=Path, required=True)
    parser.add_argument("--ngrid", type=int, default=6)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    q_cart, matrices = load_matdyn_matrices(args.dynamical_matrices)
    q_crystal = qgrid(args.ngrid)
    if len(matrices) != args.ngrid * args.ngrid:
        raise ValueError(f"expected {args.ngrid ** 2} matrices, found {len(matrices)}")
    q_error = float(np.max(np.abs(q_cart - expected_q_cart(q_crystal))))
    result = {
        "n_qpoints": len(matrices),
        "q_coordinate_max_abs": q_error,
        "input_Hermitian_max_abs": float(
            np.max(np.abs(matrices - matrices.conj().transpose(0, 2, 1)))
        ),
        "gauges": {},
    }
    for convention in ("cell", "left_basis_phase", "right_basis_phase"):
        result["gauges"][convention] = metrics(
            gauge_matrices(matrices, q_crystal, convention), args.ngrid
        )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_name(args.output_json.name + ".tmp")
        temporary.write_text(rendered)
        os.replace(temporary, args.output_json)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
