"""Small, strict parser for Quantum ESPRESSO ``ph.x`` dynamical-matrix files.

The P0 graphene line-cut campaign writes one ``gr.dyn`` per q point.  QE places
all symmetry-equivalent matrices in the file and then diagonalises the first q
point.  This module returns that first complex Cartesian dynamical matrix plus
the printed frequencies and eigenvectors.

It intentionally supports the text layout used by the repository's QE 7.x
build rather than trying to be a general parser for every historical QE format.
Unexpected or incomplete files fail loudly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
Q_RE = re.compile(
    rf"q\s*=\s*\(\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*\)"
)
FREQ_RE = re.compile(
    rf"freq\s*\(\s*(\d+)\s*\)\s*=\s*({FLOAT})\s*\[THz\]"
    rf"\s*=\s*({FLOAT})\s*\[cm-1\]"
)


def _floats(line: str) -> list[float]:
    return [float(value.replace("D", "E").replace("d", "e"))
            for value in re.findall(FLOAT, line)]


@dataclass(frozen=True)
class QEDyn:
    path: Path
    natoms: int
    q_cart_2pi_over_a: np.ndarray
    matrix: np.ndarray
    frequencies_thz: np.ndarray
    frequencies_cm: np.ndarray
    eigenvectors: np.ndarray

    @property
    def hermitian_error(self) -> float:
        return float(np.max(np.abs(self.matrix - self.matrix.conj().T)))

    @property
    def eigenvector_orthogonality_error(self) -> float:
        vectors = self.eigenvectors.reshape(len(self.frequencies_cm), -1)
        gram = vectors.conj() @ vectors.T
        return float(np.max(np.abs(gram - np.eye(len(vectors)))))

    def matrix_frequency_scale_cm2(self) -> float:
        """Return the scalar mapping matrix eigenvalues to squared cm^-1.

        The QE text matrix is already mass weighted, but its printed numerical
        units are not cm^-2.  A single positive conversion factor must map all
        six eigenvalues to the squared printed frequencies.
        """
        eig = np.linalg.eigvalsh((self.matrix + self.matrix.conj().T) / 2)
        mask = (eig > 1e-10) & (self.frequencies_cm > 1e-8)
        if not np.any(mask):
            raise ValueError(f"no positive modes in {self.path}")
        ratios = self.frequencies_cm[mask] ** 2 / eig[mask]
        return float(np.median(ratios))

    def frequencies_from_matrix_cm(self) -> np.ndarray:
        eig = np.linalg.eigvalsh((self.matrix + self.matrix.conj().T) / 2)
        scale = self.matrix_frequency_scale_cm2()
        return np.sign(eig) * np.sqrt(np.abs(eig) * scale)


def _natoms(lines: list[str], path: Path) -> int:
    # After the title and user comment QE writes: ntyp, nat, ibrav, ...
    for line in lines[2:10]:
        values = line.split()
        if len(values) >= 3 and all(re.fullmatch(r"[+-]?\d+", v) for v in values[:3]):
            nat = int(values[1])
            if nat <= 0:
                break
            return nat
    raise ValueError(f"could not read natoms from {path}")


def _first_matrix(lines: list[str], nat: int, path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        marker = next(
            i for i, line in enumerate(lines)
            if "Dynamical" in line and "Matrix in cartesian axes" in line
        )
    except StopIteration as exc:
        raise ValueError(f"no dynamical matrix marker in {path}") from exc

    q_index = next((i for i in range(marker + 1, len(lines)) if Q_RE.search(lines[i])), None)
    if q_index is None:
        raise ValueError(f"no q vector after matrix marker in {path}")
    q_match = Q_RE.search(lines[q_index])
    assert q_match is not None
    q = np.array([float(q_match.group(i).replace("D", "E"))
                  for i in range(1, 4)])

    matrix = np.zeros((3 * nat, 3 * nat), dtype=np.complex128)
    cursor = q_index + 1
    seen: set[tuple[int, int]] = set()
    while cursor < len(lines) and len(seen) < nat * nat:
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", lines[cursor])
        cursor += 1
        if not match:
            continue
        ia, ja = int(match.group(1)) - 1, int(match.group(2)) - 1
        if not (0 <= ia < nat and 0 <= ja < nat):
            continue
        rows = []
        for _ in range(3):
            if cursor >= len(lines):
                raise ValueError(f"truncated matrix block in {path}")
            values = _floats(lines[cursor])
            cursor += 1
            if len(values) != 6:
                raise ValueError(f"bad matrix row in {path}: {lines[cursor - 1]!r}")
            rows.append([complex(values[k], values[k + 1]) for k in range(0, 6, 2)])
        matrix[3 * ia:3 * ia + 3, 3 * ja:3 * ja + 3] = np.asarray(rows)
        seen.add((ia, ja))
    if len(seen) != nat * nat:
        raise ValueError(f"expected {nat * nat} atom blocks, found {len(seen)} in {path}")
    return q, matrix


def _modes(lines: list[str], nat: int, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        marker = max(i for i, line in enumerate(lines) if "Diagonalizing" in line)
    except ValueError as exc:
        raise ValueError(f"no diagonalisation section in {path}") from exc

    thz, cm, vectors = [], [], []
    cursor = marker + 1
    while cursor < len(lines) and len(thz) < 3 * nat:
        match = FREQ_RE.search(lines[cursor])
        cursor += 1
        if not match:
            continue
        expected_mode = len(thz) + 1
        if int(match.group(1)) != expected_mode:
            raise ValueError(f"unexpected mode order in {path}")
        thz.append(float(match.group(2).replace("D", "E")))
        cm.append(float(match.group(3).replace("D", "E")))
        atom_vectors = []
        for _ in range(nat):
            if cursor >= len(lines):
                raise ValueError(f"truncated eigenvector in {path}")
            values = _floats(lines[cursor])
            cursor += 1
            if len(values) != 6:
                raise ValueError(f"bad eigenvector row in {path}: {lines[cursor - 1]!r}")
            atom_vectors.append(
                [complex(values[k], values[k + 1]) for k in range(0, 6, 2)]
            )
        vectors.append(atom_vectors)
    if len(thz) != 3 * nat:
        raise ValueError(f"expected {3 * nat} modes, found {len(thz)} in {path}")
    return np.asarray(thz), np.asarray(cm), np.asarray(vectors)


def load_qe_dyn(path: str | Path) -> QEDyn:
    path = Path(path)
    lines = path.read_text(errors="strict").splitlines()
    nat = _natoms(lines, path)
    q, matrix = _first_matrix(lines, nat, path)
    thz, cm, vectors = _modes(lines, nat, path)
    return QEDyn(
        path=path,
        natoms=nat,
        q_cart_2pi_over_a=q,
        matrix=matrix,
        frequencies_thz=thz,
        frequencies_cm=cm,
        eigenvectors=vectors,
    )

