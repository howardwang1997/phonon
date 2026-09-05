from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))

import graphene_r2x_paired_readout as paired  # noqa: E402
from materialize_graphene_r2t_full_bilinear_basis import (  # noqa: E402
    full_bilinear_energy_columns,
)


class Fields:
    def __init__(self, gate: torch.Tensor, amplitude: torch.Tensor) -> None:
        self.multipolar_gate = gate
        self.amplitude_gate = amplitude


def test_selected_bilinear_columns_match_full_materializer() -> None:
    rng = np.random.default_rng(301)
    normalized = torch.as_tensor(rng.normal(size=(13, 34)), dtype=torch.float64)
    gate = torch.as_tensor(rng.uniform(size=13), dtype=torch.float64)
    amplitude = torch.as_tensor(rng.uniform(size=13), dtype=torch.float64)
    fields = Fields(gate, amplitude)
    full, _ = full_bilinear_energy_columns(normalized, fields)
    indices = torch.as_tensor(paired.SELECTED_BILINEAR_INDICES, dtype=torch.long)
    selected = paired.selected_bilinear_energy_columns(
        normalized, gate, amplitude, indices
    )
    torch.testing.assert_close(selected, full[indices], rtol=0.0, atol=0.0)


def test_selected_layout_has_cross32_then_frozen_pair() -> None:
    expected_diagonal = np.asarray(
        [channel * 16 + channel for channel in range(16)]
        + [256 + channel * 16 + channel for channel in range(16)]
    )
    np.testing.assert_array_equal(
        paired.SELECTED_BILINEAR_INDICES[:32], expected_diagonal
    )
    np.testing.assert_array_equal(
        paired.SELECTED_BILINEAR_INDICES[32:], np.asarray([127, 383])
    )
