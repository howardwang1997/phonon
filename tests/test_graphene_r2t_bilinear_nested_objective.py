from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))

import diagnose_graphene_r2s_nonlinear_readout_basis as reference_solver  # noqa: E402
import evaluate_graphene_r2t_bilinear_nested_objective as bilinear  # noqa: E402
import graphene_r2r1_linear_readout as linear  # noqa: E402


def synthetic_problem(width: int = 8):
    rng = np.random.default_rng(83)
    design = rng.normal(scale=0.2, size=(92, 72, 3, width))
    fixed = rng.normal(scale=0.01, size=(92, 72, 3))
    coefficient = rng.normal(scale=0.03, size=width)
    reference = fixed + np.einsum(
        "natk,k->nat", design, coefficient, optimize=False
    )
    reference += rng.normal(scale=0.002, size=reference.shape)
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
            (np.linspace(0.01, 0.03, 20), np.linspace(-0.005, 0.005, 20))
        ),
        foundation_base_force_eV_A=np.zeros((20, 72, 3)),
        frozen_q6_force_eV_A=np.zeros((20, 72, 3)),
    )
    return design, fixed, reference, aprime


def test_bilinear_basis_indices_are_unique_and_expected_width() -> None:
    expected = {
        "cross32": 32,
        "plain256": 256,
        "plain256_plus_c_diagonal16": 272,
        "diagonal16_plus_c_full256": 272,
        "full512": 512,
    }
    for kind, width in expected.items():
        indices = bilinear.bilinear_indices(kind)
        assert indices.shape == (width,)
        assert len(np.unique(indices)) == width
        assert np.min(indices) >= 0
        assert np.max(indices) < 512


def test_fold_statistics_eigen_solver_matches_tall_svd_solver() -> None:
    design, fixed, reference, aprime = synthetic_problem()
    statistics = bilinear.fold_statistics(design, fixed, reference, aprime)
    train_folds = (0, 2, 3)
    train_indices = np.sort(
        np.concatenate(
            [np.asarray(linear.FOLD_GLOBAL_INDICES[fold], dtype=int) for fold in train_folds]
        )
    )
    for mass in (0.0, 0.3, 0.9):
        expected = reference_solver._ridge_system(
            design, fixed, reference, aprime, train_indices, mass
        )
        actual = bilinear.ridge_system(statistics, train_folds, mass)
        for alpha in (1.0e-8, 1.0e-3, 1.0, 100.0):
            np.testing.assert_allclose(
                actual.coefficient(alpha),
                expected.coefficient(alpha),
                rtol=2.0e-10,
                atol=2.0e-12,
            )
