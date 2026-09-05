from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))

from analyze_graphene_r2k_descriptor_separability import (  # noqa: E402
    FEATURE_MODE_SIGNED_L0,
    FEATURE_MODE_TENSOR_POWER,
    canonical_json_sha256,
    feature_mask_from_schema,
    feature_schema_with_pruning,
    fit_relative_variance_feature_mask,
    invariant_feature_schema,
    numpy_invariant_features,
    replay_router_score_torch,
    torch_invariant_features,
)


def synthetic_layout() -> tuple[list[dict], int]:
    """Two scalar, two l=1, and one l=2 multiplicity channels."""
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
        },
        {
            "interaction": 0,
            "start": 2,
            "stop": 8,
            "multiplicity": 2,
            "dimension": 3,
            "l": 1,
            "parity": -1,
            "irrep": "1o",
        },
        {
            "interaction": 1,
            "start": 8,
            "stop": 9,
            "multiplicity": 1,
            "dimension": 1,
            "l": 0,
            "parity": 1,
            "irrep": "0e",
        },
        {
            "interaction": 1,
            "start": 9,
            "stop": 14,
            "multiplicity": 1,
            "dimension": 5,
            "l": 2,
            "parity": 1,
            "irrep": "2e",
        },
    ]
    return layout, 14


def test_signed_l0_default_is_the_historical_slice_concatenation():
    layout, width = synthetic_layout()
    raw = np.arange(4 * width, dtype=float).reshape(4, width) / 7.0
    schema = invariant_feature_schema(layout, width, FEATURE_MODE_SIGNED_L0)
    actual = numpy_invariant_features(raw, schema)
    expected = np.concatenate([raw[:, 0:2], raw[:, 8:9]], axis=1)
    np.testing.assert_array_equal(actual, expected)
    assert schema["output_dimension"] == 3
    assert [block["operation"] for block in schema["blocks"]] == [
        "signed_l0_value",
        "signed_l0_value",
    ]


def test_tensor_power_schema_dimensions_hash_and_torch_replay():
    layout, width = synthetic_layout()
    schema = invariant_feature_schema(layout, width, FEATURE_MODE_TENSOR_POWER)
    assert schema["output_dimension"] == 6
    assert [block["output_slice"] for block in schema["blocks"]] == [
        [0, 2],
        [2, 3],
        [3, 5],
        [5, 6],
    ]
    unhashed = dict(schema)
    recorded_hash = unhashed.pop("schema_sha256")
    assert canonical_json_sha256(unhashed) == recorded_hash

    generator = torch.Generator().manual_seed(19)
    raw = torch.randn(5, width, generator=generator, dtype=torch.float64)
    actual_torch = torch_invariant_features(raw, schema)
    actual_numpy = numpy_invariant_features(raw.numpy(), schema)
    np.testing.assert_allclose(actual_torch.numpy(), actual_numpy, atol=1.0e-13)

    expected = torch.cat(
        [
            raw[:, 0:2],
            raw[:, 8:9],
            torch.mean(raw[:, 2:8].reshape(5, 2, 3) ** 2, dim=-1),
            torch.mean(raw[:, 9:14].reshape(5, 1, 5) ** 2, dim=-1),
        ],
        dim=1,
    )
    torch.testing.assert_close(actual_torch, expected, atol=0.0, rtol=0.0)


def test_tensor_power_is_rotation_invariant_and_router_replay_is_differentiable():
    # Import after the R2K/MACE module so the repository's Torch 2.6/e3nn
    # compatibility setup is already active.
    from e3nn import o3

    layout, width = synthetic_layout()
    schema = invariant_feature_schema(layout, width, FEATURE_MODE_TENSOR_POWER)
    generator = torch.Generator().manual_seed(23)
    raw = torch.randn(7, width, generator=generator, dtype=torch.float64)
    rotated = raw.clone()
    rotation = o3.rand_matrix().to(dtype=raw.dtype)
    for item in layout:
        if item["l"] == 0:
            continue
        start, stop = item["start"], item["stop"]
        block = raw[:, start:stop].reshape(
            len(raw), item["multiplicity"], item["dimension"]
        )
        matrix = o3.Irrep(item["irrep"]).D_from_matrix(rotation).to(raw.dtype)
        rotated[:, start:stop] = torch.einsum(
            "ab,ncb->nca", matrix, block
        ).reshape(len(raw), -1)
    torch.testing.assert_close(
        torch_invariant_features(rotated, schema),
        torch_invariant_features(raw, schema),
        atol=2.0e-11,
        rtol=2.0e-11,
    )

    pruning = fit_relative_variance_feature_mask(
        torch_invariant_features(raw, schema).numpy(), schema["feature_names"]
    )
    schema = feature_schema_with_pruning(schema, pruning)
    differentiable_raw = raw.clone().requires_grad_(True)
    dimension = schema["output_dimension"]
    state = {
        "scaler_mean": np.linspace(-0.2, 0.3, dimension).tolist(),
        "scaler_scale": np.linspace(0.8, 1.3, dimension).tolist(),
        "ridge_coefficient": np.linspace(-0.7, 0.9, dimension).tolist(),
        "ridge_intercept": 0.125,
    }
    score = replay_router_score_torch(differentiable_raw, schema, state)
    score.sum().backward()
    assert score.shape == (len(raw),)
    assert differentiable_raw.grad is not None
    assert torch.all(torch.isfinite(differentiable_raw.grad))
    assert torch.count_nonzero(differentiable_raw.grad) > 0

    invalid_state = dict(state)
    invalid_scale = list(state["scaler_scale"])
    invalid_scale[0] = 0.0
    invalid_state["scaler_scale"] = invalid_scale
    try:
        replay_router_score_torch(differentiable_raw, schema, invalid_state)
    except ValueError as error:
        assert "negative or zero" in str(error)
    else:
        raise AssertionError("zero router scale was accepted")


def test_training_only_relative_variance_pruning_preserves_small_valid_channels():
    coordinate = np.linspace(-1.0, 1.0, 17)
    training = np.column_stack(
        [
            coordinate,
            1.0 + 1.0e-8 * coordinate,
            1.0e-12 * coordinate,
            2.0 * coordinate,
            np.full_like(coordinate, 5.0),
            coordinate**2,
        ]
    )
    names = [f"feature_{index}" for index in range(training.shape[1])]
    pruning = fit_relative_variance_feature_mask(training, names)
    assert pruning["feature_mask"] == [True, False, True, True, False, True]
    assert pruning["input_dimension"] == 6
    assert pruning["retained_dimension"] == 4
    assert pruning["training_population_std"][2] < 1.0e-12
    assert pruning["training_relative_std"][2] > 0.9
    unhashed_pruning = dict(pruning)
    pruning_hash = unhashed_pruning.pop("pruning_state_sha256")
    assert canonical_json_sha256(unhashed_pruning) == pruning_hash

    layout, width = synthetic_layout()
    base_schema = invariant_feature_schema(
        layout, width, FEATURE_MODE_TENSOR_POWER
    )
    pruned_schema = feature_schema_with_pruning(base_schema, pruning)
    assert pruned_schema["unpruned_output_dimension"] == 6
    assert pruned_schema["output_dimension"] == 4
    np.testing.assert_array_equal(
        feature_mask_from_schema(pruned_schema),
        [True, False, True, True, False, True],
    )
    unhashed_schema = dict(pruned_schema)
    schema_hash = unhashed_schema.pop("schema_sha256")
    assert canonical_json_sha256(unhashed_schema) == schema_hash

    generator = torch.Generator().manual_seed(31)
    raw = torch.randn(9, width, generator=generator, dtype=torch.float64)
    expected = torch_invariant_features(raw, base_schema)[
        :, torch.tensor(pruning["feature_mask"])
    ]
    actual = torch_invariant_features(raw, pruned_schema)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        numpy_invariant_features(raw.numpy(), pruned_schema), actual.numpy()
    )

    # The third unpruned feature has an absolute training RMS around 1e-12 but
    # order-one relative variation.  Its equally small scaler is valid because
    # the saved scale/RMS ratio is well above the fixed relative threshold.
    replay_state = {
        "scaler_mean": [0.0] * pruned_schema["output_dimension"],
        "scaler_scale": pruned_schema["retained_training_population_std"],
        "ridge_coefficient": [0.2] * pruned_schema["output_dimension"],
        "ridge_intercept": -0.1,
    }
    assert min(replay_state["scaler_scale"]) < 1.0e-12
    replay_score = replay_router_score_torch(raw, pruned_schema, replay_state)
    assert torch.all(torch.isfinite(replay_score))

    damaged_schema = dict(pruned_schema)
    damaged_schema["feature_mask"] = list(pruned_schema["feature_mask"])
    damaged_schema["feature_mask"][0] = False
    try:
        feature_mask_from_schema(damaged_schema)
    except ValueError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("damaged replay mask was accepted")


def test_schema_rejects_noncontiguous_or_noninvariant_raw_layouts():
    layout, width = synthetic_layout()
    noncontiguous = [dict(item) for item in layout]
    noncontiguous[1]["start"] += 1
    try:
        invariant_feature_schema(
            noncontiguous, width, FEATURE_MODE_TENSOR_POWER
        )
    except ValueError as error:
        assert "not contiguous" in str(error)
    else:
        raise AssertionError("noncontiguous raw layout was accepted")

    wrong_dimension = [dict(item) for item in layout]
    wrong_dimension[1]["dimension"] = 5
    try:
        invariant_feature_schema(
            wrong_dimension, width, FEATURE_MODE_TENSOR_POWER
        )
    except ValueError as error:
        assert "2*l+1" in str(error)
    else:
        raise AssertionError("wrong irrep dimension was accepted")

    pseudoscalar = [dict(item) for item in layout]
    pseudoscalar[0]["parity"] = -1
    pseudoscalar[0]["irrep"] = "0o"
    try:
        invariant_feature_schema(pseudoscalar, width, FEATURE_MODE_SIGNED_L0)
    except ValueError as error:
        assert "0o" in str(error)
    else:
        raise AssertionError("O(3)-odd pseudoscalar was accepted")
