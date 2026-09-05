from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))

from analyze_graphene_r2k_descriptor_separability import (  # noqa: E402
    FEATURE_MODE_GRAM_SKETCH16,
    FEATURE_MODE_SIGNED_L0,
    FEATURE_MODE_TENSOR_POWER,
    GRAM_SKETCH16_PHI_SHA256,
    GRAM_SKETCH16_R_SHA256,
    canonical_json_sha256,
    feature_schema_with_pruning,
    fit_relative_variance_feature_mask,
    invariant_feature_schema,
    mace_invariant_layout,
    numpy_invariant_features,
    replay_router_score_torch,
    torch_invariant_features,
)


def gram_layout() -> tuple[list[dict], int]:
    """Exact concatenated node_feats layout of the r3-h16-l1 descriptor."""
    return [
        {
            "interaction": 0,
            "start": 0,
            "stop": 16,
            "multiplicity": 16,
            "dimension": 1,
            "l": 0,
            "parity": 1,
            "irrep": "0e",
        },
        {
            "interaction": 0,
            "start": 16,
            "stop": 64,
            "multiplicity": 16,
            "dimension": 3,
            "l": 1,
            "parity": -1,
            "irrep": "1o",
        },
        {
            "interaction": 1,
            "start": 64,
            "stop": 80,
            "multiplicity": 16,
            "dimension": 1,
            "l": 0,
            "parity": 1,
            "irrep": "0e",
        },
    ], 80


def historical_test_layout() -> tuple[list[dict], int]:
    return [
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
    ], 14


def test_legacy_schema_hashes_and_values_are_unchanged():
    layout, width = historical_test_layout()
    signed = invariant_feature_schema(layout, width, FEATURE_MODE_SIGNED_L0)
    tensor = invariant_feature_schema(layout, width, FEATURE_MODE_TENSOR_POWER)
    assert signed["schema_sha256"] == (
        "c87dba932bcbbbcc82fbd1c246c42872f927e6fc442573d4b38133a961c316a0"
    )
    assert tensor["schema_sha256"] == (
        "db655fa962003c38ddd5bb1e0e0093fd6dfecf2c1d8b2727e997056ba80716ea"
    )
    raw = np.arange(42, dtype=float).reshape(3, width) / 11.0
    np.testing.assert_array_equal(
        numpy_invariant_features(raw, signed),
        np.concatenate([raw[:, 0:2], raw[:, 8:9]], axis=1),
    )


def test_fixed_R_Phi_hashes_pair_order_and_manual_formula():
    layout, width = gram_layout()
    schema = invariant_feature_schema(layout, width, FEATURE_MODE_GRAM_SKETCH16)
    sketch = schema["gram_sketch"]
    assert schema["unpruned_output_dimension"] == 64
    assert sketch["R_sha256"] == GRAM_SKETCH16_R_SHA256
    assert sketch["Phi_sha256"] == GRAM_SKETCH16_PHI_SHA256
    assert sketch["R_shape"] == [16, 16]
    assert sketch["Phi_shape"] == [16, 120]
    assert sketch["offdiagonal_pair_order"] == [
        [a, b] for a in range(16) for b in range(a + 1, 16)
    ]
    assert sketch["legacy_tensor_power_retained"] is True
    assert sketch["diagonal_self_power_in_gram_sketch"] is False
    unhashed = copy.deepcopy(schema)
    recorded_hash = unhashed.pop("schema_sha256")
    assert canonical_json_sha256(unhashed) == recorded_hash

    generator = np.random.default_rng(101)
    raw = generator.normal(size=(7, width))
    actual = numpy_invariant_features(raw, schema)
    tensor_schema = invariant_feature_schema(layout, width, FEATURE_MODE_TENSOR_POWER)
    np.testing.assert_array_equal(
        actual[:, :48], numpy_invariant_features(raw, tensor_schema)
    )
    h = raw[:, 16:64].reshape(len(raw), 16, 3)
    gram = np.einsum("nam,nbm->nab", h, h) / 3.0
    pairs = np.asarray(sketch["offdiagonal_pair_order"], dtype=int)
    phi = np.asarray(sketch["Phi"], dtype=float)
    manual = gram[:, pairs[:, 0], pairs[:, 1]] @ phi.T
    np.testing.assert_allclose(actual[:, 48:], manual, atol=2.0e-15, rtol=2.0e-15)

    single_channel = np.zeros((2, width), float)
    single_channel[:, 16:19] = [[1.0, -2.0, 3.0], [-4.0, 0.5, 2.0]]
    isolated = numpy_invariant_features(single_channel, schema)
    np.testing.assert_array_equal(isolated[:, 48:], np.zeros((2, 16)))


def test_gram_sketch_is_rotation_and_reflection_invariant():
    layout, width = gram_layout()
    schema = invariant_feature_schema(layout, width, FEATURE_MODE_GRAM_SKETCH16)
    generator = np.random.default_rng(131)
    raw = generator.normal(size=(9, width))
    expected = numpy_invariant_features(raw, schema)
    h = raw[:, 16:64].reshape(len(raw), 16, 3)

    for determinant in (1, -1):
        matrix, _ = np.linalg.qr(generator.normal(size=(3, 3)))
        if round(np.linalg.det(matrix)) != determinant:
            matrix[:, 0] *= -1.0
        transformed = raw.copy()
        transformed[:, 16:64] = np.einsum(
            "ab,ncb->nca", matrix, h
        ).reshape(len(raw), -1)
        np.testing.assert_allclose(
            numpy_invariant_features(transformed, schema),
            expected,
            atol=3.0e-14,
            rtol=3.0e-14,
        )


def test_numpy_torch_replay_and_router_gradient_match():
    layout, width = gram_layout()
    base_schema = invariant_feature_schema(layout, width, FEATURE_MODE_GRAM_SKETCH16)
    generator = torch.Generator().manual_seed(149)
    raw = torch.randn(13, width, generator=generator, dtype=torch.float64)
    numpy_features = numpy_invariant_features(raw.numpy(), base_schema)
    torch_features = torch_invariant_features(raw, base_schema)
    np.testing.assert_allclose(
        torch_features.numpy(), numpy_features, atol=3.0e-14, rtol=3.0e-14
    )

    pruning = fit_relative_variance_feature_mask(
        numpy_features, base_schema["feature_names"]
    )
    schema = feature_schema_with_pruning(base_schema, pruning)
    dimension = schema["output_dimension"]
    state = {
        "scaler_mean": np.linspace(-0.2, 0.3, dimension).tolist(),
        "scaler_scale": np.linspace(0.8, 1.3, dimension).tolist(),
        "ridge_coefficient": np.linspace(-0.7, 0.9, dimension).tolist(),
        "ridge_intercept": 0.125,
    }
    differentiable = raw.clone().requires_grad_(True)
    score = replay_router_score_torch(differentiable, schema, state)
    score.sum().backward()
    assert differentiable.grad is not None
    assert torch.all(torch.isfinite(differentiable.grad))
    assert torch.count_nonzero(differentiable.grad[:, 16:64]) > 0


def test_gram_mode_rejects_wrong_ambiguous_or_l2_layout_and_wrong_model_radius():
    layout, width = gram_layout()

    wrong_channels = copy.deepcopy(layout)
    wrong_channels[1]["multiplicity"] = 15
    wrong_channels[1]["stop"] = 61
    wrong_channels[2]["start"] = 61
    wrong_channels[2]["stop"] = 77
    try:
        invariant_feature_schema(
            wrong_channels, 77, FEATURE_MODE_GRAM_SKETCH16
        )
    except ValueError as error:
        assert "16x1o" in str(error)
    else:
        raise AssertionError("a 15x1o Gram-sketch source was accepted")

    duplicate = copy.deepcopy(layout)
    duplicate.insert(
        2,
        {
            "interaction": 0,
            "start": 64,
            "stop": 67,
            "multiplicity": 1,
            "dimension": 3,
            "l": 1,
            "parity": -1,
            "irrep": "1o",
        },
    )
    duplicate[3]["start"] = 67
    duplicate[3]["stop"] = 83
    try:
        invariant_feature_schema(duplicate, 83, FEATURE_MODE_GRAM_SKETCH16)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("an ambiguous interaction0 l=1 layout was accepted")

    with_l2 = copy.deepcopy(layout)
    with_l2.append(
        {
            "interaction": 1,
            "start": 80,
            "stop": 85,
            "multiplicity": 1,
            "dimension": 5,
            "l": 2,
            "parity": 1,
            "irrep": "2e",
        }
    )
    try:
        invariant_feature_schema(with_l2, 85, FEATURE_MODE_GRAM_SKETCH16)
    except ValueError as error:
        assert "l_max=1" in str(error)
    else:
        raise AssertionError("an l_max=2 descriptor layout was accepted")

    wrong_radius_model = SimpleNamespace(
        r_max=4.0, num_interactions=2, products=[]
    )
    try:
        mace_invariant_layout(wrong_radius_model, FEATURE_MODE_GRAM_SKETCH16)
    except ValueError as error:
        assert "r_max=3" in str(error)
    else:
        raise AssertionError("a non-r3 descriptor model was accepted")

    wrong_depth_model = SimpleNamespace(
        r_max=3.0, num_interactions=3, products=[]
    )
    try:
        mace_invariant_layout(wrong_depth_model, FEATURE_MODE_GRAM_SKETCH16)
    except ValueError as error:
        assert "two-interaction" in str(error)
    else:
        raise AssertionError("a non-depth2 descriptor model was accepted")


def test_serialized_projection_tampering_is_rejected_even_if_schema_is_rehashed():
    layout, width = gram_layout()
    schema = invariant_feature_schema(layout, width, FEATURE_MODE_GRAM_SKETCH16)
    damaged = copy.deepcopy(schema)
    damaged["gram_sketch"]["Phi"][0][0] *= -1.0
    damaged.pop("schema_sha256")
    damaged["schema_sha256"] = canonical_json_sha256(damaged)
    raw = np.zeros((1, width), float)
    try:
        numpy_invariant_features(raw, damaged)
    except ValueError as error:
        assert "Phi" in str(error)
    else:
        raise AssertionError("a rehashed but noncanonical Phi was accepted")

    wrong_source = copy.deepcopy(schema)
    gram_block = next(
        block
        for block in wrong_source["blocks"]
        if block["operation"] == "offdiagonal_gram_sketch16"
    )
    gram_block["raw_slice"] = [0, 48]
    wrong_source.pop("schema_sha256")
    wrong_source["schema_sha256"] = canonical_json_sha256(wrong_source)
    try:
        numpy_invariant_features(raw, wrong_source)
    except ValueError as error:
        assert "fixed interaction0 16x1o" in str(error)
    else:
        raise AssertionError("a rehashed Gram block selecting the wrong slice was accepted")
