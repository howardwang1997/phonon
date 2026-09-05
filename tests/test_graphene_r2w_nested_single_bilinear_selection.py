from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))

import evaluate_graphene_r2w_nested_single_bilinear_selection as selection  # noqa: E402
import graphene_r2r1_linear_readout as linear  # noqa: E402


def test_block_single_feature_solver_matches_individual_direct_solves() -> None:
    rng = np.random.default_rng(221)
    core_width = 5
    candidate_count = 7
    width = core_width + candidate_count
    matrix = rng.normal(size=(80, width))
    gram = matrix.T @ matrix / 80.0
    rhs = rng.normal(size=width)
    normalizer = np.exp(rng.normal(scale=0.2, size=width))
    alpha = 0.017
    core, off, _ = selection.solve_all_single_feature_coefficients(
        normalizer, gram, rhs, alpha, core_width
    )
    for candidate in range(candidate_count):
        indices = np.asarray([*range(core_width), core_width + candidate])
        direct = np.linalg.solve(
            gram[np.ix_(indices, indices)] + alpha * np.eye(core_width + 1),
            rhs[indices],
        )
        physical = linear.FORCE_SCALE_EV_A * direct / normalizer[indices]
        np.testing.assert_allclose(core[:, candidate], physical[:-1], rtol=1e-12)
        np.testing.assert_allclose(off[candidate], physical[-1], rtol=1e-12)


def test_vectorized_gate_score_matches_scalar_gate_metrics() -> None:
    rng = np.random.default_rng(222)
    reference = rng.normal(scale=0.03, size=(92, 72, 3))
    predicted = np.stack(
        (
            reference + rng.normal(scale=0.012, size=reference.shape),
            reference + rng.normal(scale=0.018, size=reference.shape),
            reference + rng.normal(scale=0.024, size=reference.shape),
        ),
        axis=-1,
    )
    mode_real = rng.normal(size=(20, 72, 3))
    mode_imag = rng.normal(size=(20, 72, 3))
    norm = np.sqrt(
        np.sum(np.square(mode_real) + np.square(mode_imag), axis=(1, 2))
    )
    mode_real /= norm[:, None, None]
    mode_imag /= norm[:, None, None]
    aprime = linear.AprimeData(
        mode_real=mode_real,
        mode_imag=mode_imag,
        coordinates=np.column_stack(
            (np.linspace(0.01, 0.03, 20), np.linspace(-0.006, 0.006, 20))
        ),
        foundation_base_force_eV_A=rng.normal(scale=0.01, size=(20, 72, 3)),
        frozen_q6_force_eV_A=rng.normal(scale=0.01, size=(20, 72, 3)),
    )
    flat = predicted.reshape(92, 72 * 3, 3)
    for indices in (
        np.arange(92, dtype=int),
        np.sort(
            np.concatenate(
                (
                    np.arange(5, 20),
                    np.arange(29, 56),
                    np.arange(65, 92),
                )
            )
        ),
    ):
        actual = selection.vectorized_gate_scores(flat, reference, aprime, indices)
        expected = np.asarray(
            [
                linear.gate_metrics(predicted[..., item], reference, aprime, indices)[
                    "raw_selection_score"
                ]
                for item in range(3)
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
