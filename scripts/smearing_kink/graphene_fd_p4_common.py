"""Shared, target-independent utilities for the frozen graphene P4 evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


CM_PER_THZ = 33.35641
P4_LINE_QPOINTS = (
    ("G", 0.000),
    ("G", 0.005),
    ("G", 0.010),
    ("G", 0.015),
    ("G", 0.025),
    ("G", 0.040),
    ("G", 0.060),
    ("G", 0.080),
    ("K", 0.940),
    ("K", 0.960),
    ("K", 0.975),
    ("K", 0.985),
    ("K", 0.992),
    ("K", 1.000),
    ("K", 1.008),
    ("K", 1.015),
    ("K", 1.025),
    ("K", 1.040),
    ("K", 1.060),
)


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


def p4_key(seed: int, snapshot_index: int) -> str:
    return f"seed{int(seed)}:snapshot{int(snapshot_index):03d}"


def geometry_sha256(numbers, cell, positions) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(numbers, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(cell, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(positions, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def harmonic_energy_forces(fc, reference_positions, cell, positions):
    displacement = np.asarray(positions, float) - np.asarray(reference_positions, float)
    fractional = displacement @ np.linalg.inv(np.asarray(cell, float))
    fractional -= np.round(fractional)
    displacement = fractional @ np.asarray(cell, float)
    phi_u = np.einsum("ijab,jb->ia", np.asarray(fc, float), displacement)
    energy = 0.5 * float(np.einsum("ia,ia->", displacement, phi_u))
    return energy, -phi_u, displacement


def qpoint_fractional(t_values) -> np.ndarray:
    values = np.asarray(t_values, float)
    return np.column_stack((values / 3.0, values / 3.0, np.zeros_like(values)))


def rounded_quadratic_features(distance, degauss: float) -> np.ndarray:
    distance = np.asarray(distance, float)
    epsilon = 4.0 * float(degauss)
    rounded = np.sqrt(distance * distance + epsilon * epsilon) - epsilon
    return np.column_stack((np.ones_like(distance), rounded, distance * distance))


def apply_top_projector(matrix, delta_lambda_cm2, frequencies_cm, *, gamma: bool):
    hermitian = (np.asarray(matrix, complex) + np.asarray(matrix, complex).conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    frequencies = np.asarray(frequencies_cm, float)
    mask = (eigenvalues > 1.0e-10) & (frequencies > 1.0e-6)
    if not np.any(mask):
        raise ValueError("cannot determine dynamical-matrix frequency scale")
    scale = float(np.median(frequencies[mask] ** 2 / eigenvalues[mask]))
    correction = np.zeros_like(hermitian)
    for index in ((-2, -1) if gamma else (-1,)):
        vector = eigenvectors[:, index]
        correction += (float(delta_lambda_cm2) / scale) * np.outer(
            vector, vector.conj()
        )
    corrected_eigenvalues = np.linalg.eigvalsh(hermitian + correction)
    corrected = np.sign(corrected_eigenvalues) * np.sqrt(
        np.abs(corrected_eigenvalues) * scale
    )
    scalar_top = math.sqrt(max(frequencies[-1] ** 2 + float(delta_lambda_cm2), 0.0))
    return corrected, scalar_top


def kink_strength(t_values, frequencies) -> float:
    lookup = {
        round(float(t), 3): float(frequency)
        for t, frequency in zip(t_values, frequencies, strict=True)
    }
    required = (0.992, 1.000, 1.008)
    if not all(value in lookup for value in required):
        raise ValueError("K kink requires t=0.992, 1.000, and 1.008")
    dx = 0.008
    return float(
        abs(
            (lookup[1.008] - lookup[1.000]) / dx
            - (lookup[1.000] - lookup[0.992]) / dx
        )
    )


def line_metrics(region: str, t_values, prediction, target) -> dict:
    t = np.asarray(t_values, float)
    predicted = np.asarray(prediction, float)
    reference = np.asarray(target, float)
    if t.shape != predicted.shape or t.shape != reference.shape:
        raise ValueError("line metric arrays have different shapes")
    anchor = 0.0 if region == "G" else 1.0
    distance = t if region == "G" else np.abs(t - 1.0)
    central_limit = 0.010 if region == "G" else 0.008
    central = distance <= central_limit + 1.0e-12
    anchor_index = int(np.argmin(np.abs(t - anchor)))
    error = np.abs(predicted - reference)
    result = {
        "n_points": int(len(t)),
        "line_MAE_cm-1": float(np.mean(error)),
        "line_max_abs_cm-1": float(np.max(error)),
        "high_symmetry_prediction_cm-1": float(predicted[anchor_index]),
        "high_symmetry_target_cm-1": float(reference[anchor_index]),
        "high_symmetry_abs_error_cm-1": float(error[anchor_index]),
        "central_block_max_distance": central_limit,
        "central_block_n_points": int(np.count_nonzero(central)),
        "central_block_MAE_cm-1": float(np.mean(error[central])),
        "central_block_max_abs_cm-1": float(np.max(error[central])),
        "central_block_high_symmetry_abs_error_cm-1": float(error[anchor_index]),
    }
    if region == "K":
        predicted_kink = kink_strength(t, predicted)
        target_kink = kink_strength(t, reference)
        result.update(
            {
                "prediction_kink_cm-1_per_t": predicted_kink,
                "target_kink_cm-1_per_t": target_kink,
                "kink_relative_error": float(
                    abs(predicted_kink - target_kink) / max(abs(target_kink), 1.0e-12)
                ),
            }
        )
    return result


def percentile_higher(values, quantile: float = 0.95) -> float:
    return float(np.quantile(np.asarray(values, float), quantile, method="higher"))
