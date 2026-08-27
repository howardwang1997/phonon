import numpy as np

from phonon_accel.long_range import (
    FewQChebyshevVertexAdapter,
    apply_mode_projected_correction,
    signed_sqrt,
)


def test_projected_correction_shifts_only_selected_mode():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    vectors, _ = np.linalg.qr(raw)
    eigenvalues = np.asarray([1.0, 2.0, 4.0, 7.0, 11.0])
    short = vectors @ np.diag(eigenvalues) @ vectors.conj().T
    delta = -3.25
    result = apply_mode_projected_correction(
        short[None, :, :],
        np.asarray([100.0]),
        vectors[None, :, :],
        np.asarray([4]),
        np.asarray([delta * 100.0]),
    )
    expected = signed_sqrt(np.asarray([1.0, 2.0, 4.0, 7.0, 11.0 + delta]) * 100.0)
    assert np.allclose(result.frequencies_cm1[0], expected, atol=1e-12)
    assert np.isclose(result.tracked_frequency_cm1[0], expected[-1], atol=1e-12)
    assert result.diagnostics["total_hermitian_max_abs"] == 0.0
    assert result.diagnostics["orthogonal_correction_leakage_max_abs"] < 1e-15


def test_five_q_chebyshev_adapter_replays_training_labels():
    coordinate = np.linspace(-0.05, 0.05, 5)
    features = np.column_stack(
        [
            1.0 + 2.0j * coordinate,
            coordinate**2 - 0.5j * coordinate**3,
            np.ones_like(coordinate) * (0.3 - 0.2j),
        ]
    )
    adapter = FewQChebyshevVertexAdapter.fit(coordinate, features, rank=3)
    assert np.allclose(adapter.predict(coordinate), features, atol=1e-12)
