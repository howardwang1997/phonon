from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))

from analyze_graphene_r2k_descriptor_separability import (  # noqa: E402
    ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC,
    ROUTER_FEATURE_MAP_LINEAR,
    atomic_json,
    canonical_json_sha256,
    feature_schema_with_pruning,
    fit_relative_variance_feature_mask,
    fit_ridge,
    fit_router,
    invariant_feature_schema,
    replay_router_score_numpy,
    replay_router_score_torch,
)


def test_atomic_json_maps_nonfinite_diagnostics_to_null():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "strict.json"
        atomic_json(
            path,
            {
                "finite": np.float64(1.25),
                "nan": float("nan"),
                "positive_infinity": float("inf"),
                "nested": [np.float32("-inf")],
            },
        )
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text
        assert json.loads(text) == {
            "finite": 1.25,
            "nan": None,
            "positive_infinity": None,
            "nested": [None],
        }


def router_record(values: np.ndarray, utility: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        router_features={"mace_invariants": np.asarray(values, float)},
        atom_utility_meV2_A2=np.asarray(utility, float),
    )


def test_diagonal_quadratic_numpy_replay_matches_manual_two_scaler_pipeline():
    values = np.array(
        [
            [-2.0, 0.5, 1.2],
            [-1.0, -0.3, 0.7],
            [-0.2, 1.1, -0.8],
            [0.4, -1.4, 0.1],
            [0.9, 0.2, 1.7],
            [1.5, -0.8, -1.1],
            [2.1, 1.6, 0.4],
            [2.8, -1.9, -1.6],
        ]
    )
    utility = np.array([-81.0, 64.0, -49.0, 36.0, 25.0, -16.0, 9.0, -4.0])
    probe = fit_router(
        [router_record(values, utility)],
        "mace_invariants",
        alpha=10.0,
        utility_deadband=1.0,
        feature_map=ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC,
    )
    state = probe.state()

    input_mean = np.asarray(state["input_scaler_mean"])
    input_scale = np.asarray(state["input_scaler_scale"])
    z = (values - input_mean) / input_scale
    phi = np.concatenate([z, z**2], axis=1)
    mapped_mean = np.asarray(state["scaler_mean"])
    mapped_scale = np.asarray(state["scaler_scale"])
    coefficient = np.asarray(state["ridge_coefficient"])
    manual = np.sum(
        (phi - mapped_mean) / mapped_scale * coefficient, axis=1
    ) + float(state["ridge_intercept"])

    np.testing.assert_allclose(probe.predict(values), manual, atol=2.0e-15)
    np.testing.assert_allclose(
        replay_router_score_numpy(values, state), manual, atol=2.0e-15
    )
    assert state["router_input_feature_dimension"] == values.shape[1]
    assert state["router_mapped_feature_dimension"] == 2 * values.shape[1]
    schema = copy.deepcopy(state["router_feature_map_schema"])
    recorded_hash = schema.pop("schema_sha256")
    assert canonical_json_sha256(schema) == recorded_hash
    assert state["router_feature_map_schema_sha256"] == recorded_hash


def test_diagonal_quadratic_torch_replay_matches_numpy_and_keeps_gradient():
    layout = [
        {
            "interaction": 0,
            "start": 0,
            "stop": 3,
            "multiplicity": 3,
            "dimension": 1,
            "l": 0,
            "parity": 1,
            "irrep": "0e",
        }
    ]
    base_schema = invariant_feature_schema(layout, 3)
    generator = torch.Generator().manual_seed(71)
    raw = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    pruning = fit_relative_variance_feature_mask(
        raw.numpy(), base_schema["feature_names"]
    )
    schema = feature_schema_with_pruning(base_schema, pruning)
    values = raw.numpy()[:, np.asarray(pruning["feature_mask"], bool)]
    utility = np.array(
        [-144.0, 121.0, -100.0, 81.0, -64.0, 49.0,
         -36.0, 25.0, -16.0, 9.0, -4.0, 1.0]
    )
    probe = fit_router(
        [router_record(values, utility)],
        "mace_invariants",
        alpha=10.0,
        utility_deadband=0.5,
        feature_map=ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC,
    )
    state = probe.state()

    differentiable_raw = raw.clone().requires_grad_(True)
    torch_score = replay_router_score_torch(
        differentiable_raw, schema, state
    )
    numpy_score = replay_router_score_numpy(values, state)
    np.testing.assert_allclose(
        torch_score.detach().numpy(), numpy_score, atol=2.0e-12, rtol=2.0e-12
    )
    torch_score.sum().backward()
    assert differentiable_raw.grad is not None
    assert torch.all(torch.isfinite(differentiable_raw.grad))
    assert torch.count_nonzero(differentiable_raw.grad) > 0


def test_quadratic_fold_input_scaler_does_not_see_held_configuration():
    training_values = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [1.0, 2.0],
            [2.0, -2.0],
            [3.0, 0.5],
        ]
    )
    training_utility = np.array([-36.0, 25.0, -16.0, 9.0, -4.0, 1.0])
    held_values = np.full((4, 2), 1.0e6)
    held_utility = np.array([-9.0, 9.0, -4.0, 4.0])
    training = [router_record(training_values, training_utility)]
    held = router_record(held_values, held_utility)

    fold_probe = fit_router(
        training,
        "mace_invariants",
        alpha=10.0,
        utility_deadband=0.5,
        feature_map=ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC,
    )
    np.testing.assert_allclose(
        fold_probe.input_scaler.mean_, np.mean(training_values, axis=0), atol=0.0
    )
    leaked_mean = np.mean(
        np.concatenate(
            [training_values, held.router_features["mace_invariants"]], axis=0
        ),
        axis=0,
    )
    assert not np.allclose(fold_probe.input_scaler.mean_, leaked_mean)


def test_linear_fit_and_legacy_replay_preserve_historical_path():
    values = np.array(
        [
            [-1.5, 0.1],
            [-0.5, 1.2],
            [0.5, -0.7],
            [1.5, 0.4],
        ]
    )
    utility = np.array([4.0, -9.0, 16.0, -25.0])
    record = router_record(values, utility)
    actual = fit_router(
        [record],
        "mace_invariants",
        alpha=10.0,
        utility_deadband=1.0,
        feature_map=ROUTER_FEATURE_MAP_LINEAR,
    )
    default = fit_router(
        [record], "mace_invariants", alpha=10.0, utility_deadband=1.0
    )

    target = (utility > 0.0).astype(float)
    weights = np.abs(utility) / np.sum(np.abs(utility))
    positive = target > 0.5
    weights[positive] *= 0.5 / np.sum(weights[positive])
    weights[~positive] *= 0.5 / np.sum(weights[~positive])
    weights *= len(weights) / np.sum(weights)
    historical = fit_ridge(values, target, weights, alpha=10.0)

    np.testing.assert_array_equal(actual.scaler.mean_, historical.scaler.mean_)
    np.testing.assert_array_equal(actual.scaler.scale_, historical.scaler.scale_)
    np.testing.assert_array_equal(actual.model.coef_, historical.model.coef_)
    assert actual.model.intercept_ == historical.model.intercept_
    np.testing.assert_array_equal(actual.predict(values), historical.predict(values))
    np.testing.assert_array_equal(default.predict(values), actual.predict(values))

    state = actual.state()
    legacy_state = {
        key: state[key]
        for key in (
            "scaler_mean",
            "scaler_scale",
            "ridge_coefficient",
            "ridge_intercept",
        )
    }
    np.testing.assert_array_equal(
        replay_router_score_numpy(values, legacy_state),
        replay_router_score_numpy(values, state),
    )
    manual_numpy = (
        (values - np.asarray(legacy_state["scaler_mean"]))
        / np.asarray(legacy_state["scaler_scale"])
        @ np.asarray(legacy_state["ridge_coefficient"])
        + float(legacy_state["ridge_intercept"])
    )
    np.testing.assert_array_equal(
        replay_router_score_numpy(values, legacy_state), manual_numpy
    )

    layout = [
        {
            "interaction": 0,
            "start": 0,
            "stop": 2,
            "multiplicity": 2,
            "dimension": 1,
            "l": 0,
            "parity": 1,
            "irrep": "0e",
        }
    ]
    base_schema = invariant_feature_schema(layout, 2)
    pruning = fit_relative_variance_feature_mask(
        values, base_schema["feature_names"]
    )
    schema = feature_schema_with_pruning(base_schema, pruning)
    raw = torch.as_tensor(values, dtype=torch.float64).requires_grad_(True)
    replayed_torch = replay_router_score_torch(raw, schema, legacy_state)
    mean = raw.new_tensor(legacy_state["scaler_mean"])
    scale = raw.new_tensor(legacy_state["scaler_scale"])
    coefficient = raw.new_tensor(legacy_state["ridge_coefficient"])
    manual_torch = torch.sum(
        (raw - mean) / scale * coefficient, dim=1
    ) + float(legacy_state["ridge_intercept"])
    torch.testing.assert_close(replayed_torch, manual_torch, atol=0.0, rtol=0.0)
