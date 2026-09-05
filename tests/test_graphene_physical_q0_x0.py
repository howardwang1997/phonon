from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))

from analyze_graphene_physical_x0 import cross_metrics
from run_graphene_physical_q0_sscha import (
    full_fc_to_matrix,
    hexagonal_qpath,
    matrix_to_full_fc,
)


def test_hexagonal_qpath_contains_each_join_once():
    qpoints, distance, label_positions, labels = hexagonal_qpath(10)
    assert qpoints.shape == (31, 3)
    assert distance.shape == (31,)
    assert np.all(np.diff(distance) > 0.0)
    np.testing.assert_allclose(qpoints[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(qpoints[10], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(qpoints[20], [1.0 / 3.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(qpoints[-1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(label_positions, distance[[0, 10, 20, 30]])
    assert labels.tolist() == [r"$\Gamma$", "M", "K", r"$\Gamma$"]


def test_full_fc_matrix_round_trip():
    rng = np.random.default_rng(7)
    force_constants = rng.normal(size=(4, 4, 3, 3))
    recovered = matrix_to_full_fc(full_fc_to_matrix(force_constants))
    np.testing.assert_allclose(recovered, force_constants)


def test_cross_metrics_reports_top_branch_shift_in_wavenumbers():
    _, distance, label_positions, labels = hexagonal_qpath(10)
    formula = np.column_stack(
        [
            1.0 + 0.1 * np.sin(distance + phase)
            for phase in np.linspace(0.0, 1.0, 6)
        ]
    )
    coupled = formula.copy()
    coupled[:, -1] += 1.0 / 33.35641
    metrics = cross_metrics(distance, formula, coupled, label_positions, labels)
    assert abs(metrics["top_branch_max_abs_difference_cm-1"] - 1.0) < 1.0e-10
    assert abs(metrics["top_branch_RMSE_cm-1"] - 1.0) < 1.0e-10
    assert metrics["K_kink_relative_change"] < 1.0e-10
