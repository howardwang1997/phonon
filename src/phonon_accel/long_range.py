"""Composable dynamical-matrix interface for MLIP plus long-range physics.

The short-range provider may be a finite-displacement MLIP, TDEP fit, or any
other source of mass-weighted dynamical matrices.  Long-range models supply a
Hermitian correction in frequency-squared units.  Keeping the interface at
the matrix level preserves mode polarization, Hermiticity, and symmetry; it
also avoids fitting a visually convenient frequency curve directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def hermitian(matrix: np.ndarray) -> np.ndarray:
    """Return the Hermitian part over the final two axes."""

    matrix = np.asarray(matrix, complex)
    return (matrix + np.swapaxes(matrix.conj(), -1, -2)) / 2.0


def signed_sqrt(values: np.ndarray) -> np.ndarray:
    """Map signed squared frequencies to signed frequencies."""

    values = np.asarray(values, float)
    return np.sign(values) * np.sqrt(np.abs(values))


def mode_projectors(eigenvectors: np.ndarray, mode_indices: np.ndarray) -> np.ndarray:
    """Build phase-invariant projectors from column eigenvectors."""

    eigenvectors = np.asarray(eigenvectors, complex)
    mode_indices = np.asarray(mode_indices, int)
    if eigenvectors.ndim != 3 or eigenvectors.shape[1] != eigenvectors.shape[2]:
        raise ValueError("eigenvectors must have shape (nq, nmodes, nmodes)")
    if mode_indices.shape != (len(eigenvectors),):
        raise ValueError("one mode index is required per q point")
    if np.any(mode_indices < 0) or np.any(mode_indices >= eigenvectors.shape[-1]):
        raise ValueError("mode index is out of bounds")
    vectors = np.asarray(
        [eigenvectors[index, :, mode] for index, mode in enumerate(mode_indices)]
    )
    return np.einsum("qi,qj->qij", vectors, vectors.conj(), optimize=True)


@dataclass(frozen=True)
class ProjectedLongRangeResult:
    """Result of adding a mode-projected long-range correction."""

    correction_matrices: np.ndarray
    total_matrices: np.ndarray
    frequencies_cm1: np.ndarray
    tracked_frequency_cm1: np.ndarray
    projectors: np.ndarray
    diagnostics: dict[str, float]


def apply_mode_projected_correction(
    short_matrices: np.ndarray,
    scale_cm2: np.ndarray,
    eigenvectors: np.ndarray,
    mode_indices: np.ndarray,
    delta_lambda_cm2: np.ndarray,
) -> ProjectedLongRangeResult:
    """Add a scalar long-range self-energy along one tracked phonon mode.

    Parameters
    ----------
    short_matrices
        Hermitian mass-weighted dynamical matrices from the short-range MLIP.
    scale_cm2
        Per-q conversion from raw dynamical-matrix eigenvalues to cm^-2.
    eigenvectors
        Column eigenvectors of ``short_matrices``.
    mode_indices
        Tracked anomalous mode at each q point.
    delta_lambda_cm2
        Long-range correction to the selected squared frequency in cm^-2.
    """

    short_matrices = hermitian(short_matrices)
    scale_cm2 = np.asarray(scale_cm2, float)
    delta_lambda_cm2 = np.asarray(delta_lambda_cm2, float)
    mode_indices = np.asarray(mode_indices, int)
    nq = len(short_matrices)
    if short_matrices.ndim != 3 or short_matrices.shape[1] != short_matrices.shape[2]:
        raise ValueError("short_matrices must have shape (nq, nmodes, nmodes)")
    for name, values in (
        ("scale_cm2", scale_cm2),
        ("mode_indices", mode_indices),
        ("delta_lambda_cm2", delta_lambda_cm2),
    ):
        if values.shape != (nq,):
            raise ValueError(f"{name} must have shape (nq,)")
    if np.any(scale_cm2 <= 0.0) or not np.all(np.isfinite(scale_cm2)):
        raise ValueError("scale_cm2 must be finite and positive")
    if not np.all(np.isfinite(delta_lambda_cm2)):
        raise ValueError("delta_lambda_cm2 must be finite")

    projectors = mode_projectors(eigenvectors, mode_indices)
    correction = hermitian(
        projectors * (delta_lambda_cm2 / scale_cm2)[:, None, None]
    )
    total = hermitian(short_matrices + correction)
    eigenvalues = np.linalg.eigvalsh(total)
    frequencies = signed_sqrt(eigenvalues * scale_cm2[:, None])

    tracked_vectors = np.asarray(
        [eigenvectors[index, :, mode] for index, mode in enumerate(mode_indices)]
    )
    tracked_eigenvalues = np.einsum(
        "qi,qij,qj->q", tracked_vectors.conj(), total, tracked_vectors, optimize=True
    ).real
    tracked_frequency = signed_sqrt(tracked_eigenvalues * scale_cm2)

    identity = np.eye(short_matrices.shape[-1], dtype=complex)[None, :, :]
    orthogonal = identity - projectors
    leakage = orthogonal @ correction @ orthogonal
    diagnostics = {
        "total_hermitian_max_abs": float(
            np.max(np.abs(total - np.swapaxes(total.conj(), -1, -2)))
        ),
        "correction_hermitian_max_abs": float(
            np.max(
                np.abs(correction - np.swapaxes(correction.conj(), -1, -2))
            )
        ),
        "projector_idempotency_max_abs": float(
            np.max(np.abs(projectors @ projectors - projectors))
        ),
        "orthogonal_correction_leakage_max_abs": float(np.max(np.abs(leakage))),
    }
    return ProjectedLongRangeResult(
        correction_matrices=correction,
        total_matrices=total,
        frequencies_cm1=frequencies,
        tracked_frequency_cm1=tracked_frequency,
        projectors=projectors,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class FewQChebyshevVertexAdapter:
    """Parameter-efficient material adapter fitted to a few complex EPC labels.

    This is the deployable *fine-tuning layer*, not a pretrained cross-material
    generator.  A future generator may predict its right basis and latent
    coefficients from gauge-invariant electronic descriptors while retaining
    this interface and its fixed rank cap.
    """

    domain: tuple[float, float]
    coefficients: np.ndarray
    right_basis: np.ndarray

    @classmethod
    def fit(
        cls,
        coordinate: np.ndarray,
        vertex_labels: np.ndarray,
        rank: int = 4,
    ) -> "FewQChebyshevVertexAdapter":
        coordinate = np.asarray(coordinate, float)
        vertex_labels = np.asarray(vertex_labels, complex)
        if coordinate.ndim != 1 or vertex_labels.ndim != 2:
            raise ValueError("coordinates and flattened vertex labels must be 1D/2D")
        if len(coordinate) != len(vertex_labels):
            raise ValueError("one vertex label is required per coordinate")
        if not 1 <= rank <= len(coordinate):
            raise ValueError("rank must be between one and the number of labels")
        left, singular, right = np.linalg.svd(vertex_labels, full_matrices=False)
        latent = left[:, :rank] * singular[None, :rank]
        degree = len(coordinate) - 1
        domain = (float(np.min(coordinate)), float(np.max(coordinate)))
        mapped = 2.0 * (coordinate - domain[0]) / (domain[1] - domain[0]) - 1.0
        coefficients = np.empty((rank, degree + 1), complex)
        for component in range(rank):
            coefficients[component] = np.polynomial.chebyshev.chebfit(
                mapped, latent[:, component], degree
            )
        return cls(domain=domain, coefficients=coefficients, right_basis=right[:rank])

    @property
    def rank(self) -> int:
        return int(self.right_basis.shape[0])

    def predict(self, coordinate: np.ndarray) -> np.ndarray:
        coordinate = np.asarray(coordinate, float)
        mapped = 2.0 * (coordinate - self.domain[0]) / (
            self.domain[1] - self.domain[0]
        ) - 1.0
        latent = np.column_stack(
            [
                np.polynomial.chebyshev.chebval(mapped, coefficients)
                for coefficients in self.coefficients
            ]
        )
        return latent @ self.right_basis
