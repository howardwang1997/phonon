#!/usr/bin/env python3
"""Runtime loader for the 129-column R2AO spectral candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from graphene_r2x_paired_readout import (
    PairedReadoutCheckpoint,
    R2XCorrectionCalculator,
    file_sha256,
    production_paired_readout_energy_force,
    production_paired_readout_energy_force_hessian,
    raw_array_sha256,
)


CHECKPOINT_MEMBERS = {
    "physical_coefficient",
    "base65_physical_coefficient",
    "bilinear64_physical_coefficient",
    "selected_bilinear_indices",
    "feature_mean",
    "feature_scale",
    "train_predicted_force_eV_A",
}


def _frozen_float64(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.str != "<f8" or not array.flags.c_contiguous:
        raise ValueError(f"R2AO {label} schema changed")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"R2AO {label} contains a non-finite value")
    output = np.array(array, dtype="<f8", order="C", copy=True)
    output.setflags(write=False)
    return output


def load_step32_spectral_checkpoint(
    path: Path, *, expected_sha256: str
) -> PairedReadoutCheckpoint:
    source = Path(path).resolve(strict=True)
    observed = file_sha256(source)
    if observed != expected_sha256:
        raise ValueError("R2AO checkpoint differs from the supplied SHA256")
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays.files) != CHECKPOINT_MEMBERS:
            raise ValueError("R2AO checkpoint member set changed")
        coefficient = _frozen_float64(
            arrays["physical_coefficient"], (129,), "physical coefficient"
        )
        base = _frozen_float64(
            arrays["base65_physical_coefficient"], (65,), "base coefficient"
        )
        bilinear = _frozen_float64(
            arrays["bilinear64_physical_coefficient"], (64,), "bilinear coefficient"
        )
        mean = _frozen_float64(arrays["feature_mean"], (34,), "feature mean")
        scale = _frozen_float64(arrays["feature_scale"], (34,), "feature scale")
        indices = np.asarray(arrays["selected_bilinear_indices"])
    if not np.array_equal(coefficient[:65], base) or not np.array_equal(
        coefficient[65:], bilinear
    ):
        raise ValueError("R2AO split coefficients do not replay the full vector")
    if (
        indices.shape != (64,)
        or indices.dtype.str != "<i8"
        or len(np.unique(indices)) != 64
        or np.any(indices < 0)
        or np.any(indices >= 512)
    ):
        raise ValueError("R2AO selected bilinear indices changed")
    if np.any(scale <= 0.0):
        raise ValueError("R2AO feature scale must be positive")
    frozen_indices = np.array(indices, dtype="<i8", order="C", copy=True)
    frozen_indices.setflags(write=False)
    return PairedReadoutCheckpoint(
        path=source,
        file_sha256=observed,
        physical_coefficient=coefficient,
        feature_mean=mean,
        feature_scale=scale,
        selected_bilinear_indices=frozen_indices,
        coefficient_raw_sha256=raw_array_sha256(coefficient),
    )


__all__ = [
    "R2XCorrectionCalculator",
    "load_step32_spectral_checkpoint",
    "production_paired_readout_energy_force",
    "production_paired_readout_energy_force_hessian",
]
