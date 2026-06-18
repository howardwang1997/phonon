"""Phonon accuracy metrics: predicted (MLIP) vs reference (DFT/DFPT).

All frequencies are in THz. Two ``PhononResult`` objects compared here must
come from the *same* primitive cell and band settings so the auto (seekpath)
q-path matches point-for-point.

Headline metrics (consistent with the foundation-MLIP phonon literature):
    freq_mae / freq_rmse  -- over all band branches and q-points
    omega_max_error       -- error of the maximum phonon frequency
    n_imaginary           -- count of imaginary (softened) modes on the mesh
    asr_residual          -- acoustic frequency at Gamma (ASR violation)
    cv/s/f errors         -- thermal-property errors at a reference T
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


def _stack_band(freqs_list) -> np.ndarray:
    """Concatenate per-segment band arrays into a single (N, n_band) array."""
    return np.concatenate([np.asarray(f) for f in freqs_list], axis=0)


@dataclass
class PhononMetrics:
    formula: str
    freq_mae: float          # THz
    freq_rmse: float         # THz
    omega_max_pred: float    # THz
    omega_max_ref: float     # THz
    omega_max_error: float   # THz
    n_imaginary_pred: int
    n_imaginary_ref: int
    asr_residual_pred: float
    cv_error_300k: float = np.nan   # J/K/mol
    s_error_300k: float = np.nan    # J/K/mol
    f_error_300k: float = np.nan    # kJ/mol

    def as_row(self) -> dict:
        return asdict(self)


def _thermal_at(result, T: float, key: str) -> float:
    temps = np.asarray(result.temperatures)
    vals = np.asarray(getattr(result, key))
    if temps is None or vals is None or temps.size == 0:
        return np.nan
    return float(np.interp(T, temps, vals))


def compare(pred, ref, T_ref: float = 300.0) -> PhononMetrics:
    """Compute metrics of ``pred`` (MLIP) against ``ref`` (DFT)."""
    fp = _stack_band(pred.band_frequencies)
    fr = _stack_band(ref.band_frequencies)
    n = min(fp.shape[0], fr.shape[0])
    m = min(fp.shape[1], fr.shape[1])
    fp, fr = fp[:n, :m], fr[:n, :m]
    diff = fp - fr

    out = PhononMetrics(
        formula=pred.formula,
        freq_mae=float(np.mean(np.abs(diff))),
        freq_rmse=float(np.sqrt(np.mean(diff**2))),
        omega_max_pred=float(np.max(fp)),
        omega_max_ref=float(np.max(fr)),
        omega_max_error=float(np.max(fp) - np.max(fr)),
        n_imaginary_pred=int(pred.n_imaginary_mesh),
        n_imaginary_ref=int(ref.n_imaginary_mesh),
        asr_residual_pred=float(pred.asr_residual),
    )
    out.cv_error_300k = _thermal_at(pred, T_ref, "heat_capacity") - _thermal_at(
        ref, T_ref, "heat_capacity"
    )
    out.s_error_300k = _thermal_at(pred, T_ref, "entropy") - _thermal_at(
        ref, T_ref, "entropy"
    )
    out.f_error_300k = _thermal_at(pred, T_ref, "free_energy") - _thermal_at(
        ref, T_ref, "free_energy"
    )
    return out


def stability_flags(result) -> dict:
    """Quick dynamical-stability summary of a single result."""
    return {
        "formula": result.formula,
        "dynamically_stable": bool(result.min_frequency > -0.1),  # THz tol
        "min_frequency_thz": float(result.min_frequency),
        "n_imaginary_mesh": int(result.n_imaginary_mesh),
        "asr_residual_thz": float(result.asr_residual),
    }
