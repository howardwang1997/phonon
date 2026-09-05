#!/usr/bin/env python3
"""Small symmetry adapter for a rank-one long-range term at a K/K' pair.

The adapter works in reduced reciprocal coordinates.  Graphene's six
geometrical Brillouin-zone corners reduce, modulo reciprocal lattice vectors,
to the two valleys K and K'.  A real-space force-constant model obeys

    D(-q) = D(q).conj().

Consequently only the K-side projector and a real, time-reversal-even scalar
amplitude have to be generated.  The K' matrix is obtained by conjugation.
Using the projector, instead of a phase-bearing eigenvector, makes the mapping
gauge invariant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


K_REDUCED = np.asarray([1.0 / 3.0, 1.0 / 3.0, 0.0])
K_PRIME_REDUCED = np.asarray([2.0 / 3.0, 2.0 / 3.0, 0.0])


def fold_reduced(qpoint: np.ndarray) -> np.ndarray:
    """Fold a reduced-coordinate q point to [0, 1)."""

    return np.mod(np.asarray(qpoint, float), 1.0)


def periodic_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the shortest component-wise reduced-coordinate displacement."""

    delta = np.asarray(left, float) - np.asarray(right, float)
    return delta - np.rint(delta)


def time_reversal_partner(qpoint: np.ndarray) -> np.ndarray:
    """Return -q modulo reciprocal lattice vectors."""

    return fold_reduced(-np.asarray(qpoint, float))


def valley_index(qpoint: np.ndarray) -> int:
    """Return 0 for the K patch and 1 for the K' patch."""

    qpoint = fold_reduced(qpoint)
    distances = (
        np.linalg.norm(periodic_delta(qpoint, K_REDUCED)),
        np.linalg.norm(periodic_delta(qpoint, K_PRIME_REDUCED)),
    )
    return int(distances[1] < distances[0])


def canonical_k_patch(qpoint: np.ndarray) -> tuple[np.ndarray, bool]:
    """Map either valley patch to K.

    Returns the canonical K-side q point and whether complex conjugation is
    needed to lift a matrix back to the requested point.
    """

    qpoint = fold_reduced(qpoint)
    conjugate = valley_index(qpoint) == 1
    return (time_reversal_partner(qpoint) if conjugate else qpoint), conjugate


def hermitian(matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix, complex) + np.asarray(matrix, complex).conj().T) / 2.0


def symmetrize_time_reversal_pair(
    matrix_k: np.ndarray, matrix_k_prime: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project a K/K' pair onto Hermiticity and time reversal."""

    representative = hermitian(
        (hermitian(matrix_k) + hermitian(matrix_k_prime).conj()) / 2.0
    )
    return representative, representative.conj()


def common_rank_one_projector(matrices_at_k: list[np.ndarray]) -> np.ndarray:
    """Find one phase-free channel shared by several Hermitian corrections.

    Each matrix contributes its normalized squared operator.  This prevents a
    single smearing amplitude from dominating the shared direction.
    """

    if not matrices_at_k:
        raise ValueError("at least one matrix is required")
    dimension = np.asarray(matrices_at_k[0]).shape[0]
    channel_gram = np.zeros((dimension, dimension), complex)
    for raw in matrices_at_k:
        matrix = hermitian(raw)
        norm = float(np.linalg.norm(matrix))
        if norm == 0.0:
            raise ValueError("zero-norm matrix cannot define a projector")
        normalized = matrix / norm
        channel_gram += normalized @ normalized
    _, vectors = np.linalg.eigh(hermitian(channel_gram))
    vector = vectors[:, -1]
    projector = np.outer(vector, vector.conj())
    return hermitian(projector)


def projected_amplitude(matrix: np.ndarray, projector: np.ndarray) -> float:
    """Least-squares real coefficient of ``projector`` in ``matrix``."""

    numerator = np.vdot(projector, hermitian(matrix)).real
    denominator = np.vdot(projector, projector).real
    return float(numerator / denominator)


@dataclass(frozen=True)
class KStarRankOneAdapter:
    """Lift one K-side projector to the two-valley K star."""

    projector_k: np.ndarray

    def __post_init__(self) -> None:
        projector = hermitian(self.projector_k)
        trace = float(np.trace(projector).real)
        if trace <= 0.0:
            raise ValueError("projector must have positive trace")
        projector /= trace
        object.__setattr__(self, "projector_k", projector)

    def correction(self, qpoint: np.ndarray, amplitude: float) -> np.ndarray:
        """Generate the long-range matrix at q from a real even amplitude."""

        _, conjugate = canonical_k_patch(qpoint)
        matrix = float(amplitude) * self.projector_k
        return matrix.conj() if conjugate else matrix.copy()

    def pair(self, qpoint_at_k: np.ndarray, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate a matrix and its exact time-reversal partner."""

        qpoint_at_k, conjugate = canonical_k_patch(qpoint_at_k)
        if conjugate:
            raise ValueError("pair() expects a point in the canonical K patch")
        first = self.correction(qpoint_at_k, amplitude)
        second = self.correction(time_reversal_partner(qpoint_at_k), amplitude)
        return first, second


@dataclass(frozen=True)
class PointGroupOrbitAdapter:
    """Lift a dynamical matrix over a reciprocal-space point-group orbit.

    ``lattice`` uses row lattice vectors and ``positions`` are fractional
    atomic coordinates in the same Fourier gauge as the dynamical matrix.
    Rotations/translations follow the spglib convention.  The Bloch phase is
    essential when an operation moves an atom into a neighbouring cell.
    """

    lattice: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray

    def __post_init__(self) -> None:
        lattice = np.asarray(self.lattice, float)
        positions = np.asarray(self.positions, float)
        rotations = np.asarray(self.rotations, int)
        translations = np.asarray(self.translations, float)
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape (3, 3)")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (natoms, 3)")
        if rotations.shape[1:] != (3, 3) or translations.shape != (
            len(rotations),
            3,
        ):
            raise ValueError("invalid space-group operation arrays")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "translations", translations)

    @property
    def dimension(self) -> int:
        return 3 * len(self.positions)

    def q_image(self, qpoint: np.ndarray, operation_index: int) -> np.ndarray:
        """Return Rq in reduced reciprocal coordinates, folded to [0, 1)."""

        rotation = self.rotations[operation_index]
        unfolded = np.asarray(qpoint, float) @ np.linalg.inv(rotation)
        return fold_reduced(unfolded)

    def representation(self, qpoint: np.ndarray, operation_index: int) -> np.ndarray:
        """Return the Cartesian/Bloch representation U_g(q)."""

        rotation = self.rotations[operation_index]
        translation = self.translations[operation_index]
        q_image_unfolded = np.asarray(qpoint, float) @ np.linalg.inv(rotation)
        cartesian_rotation = (
            self.lattice.T @ rotation @ np.linalg.inv(self.lattice.T)
        )
        representation = np.zeros((self.dimension, self.dimension), complex)
        for source_atom, position in enumerate(self.positions):
            transformed = rotation @ position + translation
            differences = transformed[None, :] - self.positions
            cell_shifts = np.rint(differences)
            errors = np.linalg.norm(differences - cell_shifts, axis=1)
            target_atom = int(np.argmin(errors))
            if errors[target_atom] > 1.0e-7:
                raise ValueError("space-group operation does not map the atomic basis")
            cell_shift = cell_shifts[target_atom]
            phase = np.exp(
                -2j * np.pi * np.dot(q_image_unfolded, cell_shift)
            )
            target = slice(3 * target_atom, 3 * target_atom + 3)
            source = slice(3 * source_atom, 3 * source_atom + 3)
            representation[target, source] = phase * cartesian_rotation
        unitary_error = np.max(
            np.abs(representation.conj().T @ representation - np.eye(self.dimension))
        )
        if unitary_error > 1.0e-7:
            raise ValueError(f"non-unitary point-group representation: {unitary_error}")
        return representation

    def transform(
        self, matrix: np.ndarray, qpoint: np.ndarray, operation_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Rq, U_g D(q) U_g†)``."""

        representation = self.representation(qpoint, operation_index)
        transformed = representation @ hermitian(matrix) @ representation.conj().T
        return self.q_image(qpoint, operation_index), hermitian(transformed)

    def orbit(
        self, matrix: np.ndarray, qpoint: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate all operation images, retaining duplicates for audits."""

        q_images = []
        matrices = []
        for operation_index in range(len(self.rotations)):
            q_image, transformed = self.transform(
                matrix, qpoint, operation_index
            )
            q_images.append(q_image)
            matrices.append(transformed)
        return np.asarray(q_images), np.asarray(matrices)
