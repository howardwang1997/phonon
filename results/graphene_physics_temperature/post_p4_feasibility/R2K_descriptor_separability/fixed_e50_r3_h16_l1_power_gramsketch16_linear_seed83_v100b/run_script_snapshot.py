#!/usr/bin/env python3
"""R2K: audit whether frozen MACE environments can route a local residual.

This is a development-data separability audit, not a new potential fit.  The
depth-3 delta MACE is frozen and is used to obtain the utility of switching the
whole delta expert on, its remaining force residual on the nine fixed-smearing
support configurations, and per-atom ``node_feats``.  Every fixed-map Ridge
prediction for a support structure is made by a probe that excluded that
entire structure; atoms and bonds are never randomly split.  The frozen
descriptor and depth-3 expert did see all nine support structures during their
earlier training, so this LOCO test is a development separability diagnostic,
not an independent model-generalization result.

The router label is not a dataset role.  At atom ``i`` it is the signed utility
``u_i = ||y_i||^2 - ||y_i - p_i||^2``, where ``y`` is the frozen delta target
and ``p`` is the depth-3 delta expert.  Positive utility means that enabling the
expert improves on the zero-delta baseline.  First-neighbour bond utilities are
exchange-invariant endpoint averages; their absolute magnitude supplies the
probe weight.  E50 fixed-smearing trajectory seeds 0 and 1 participate in the
screening-probe fit, while seed 2 is held out from fitting and used only as a
descriptor-selection screen.  It is not a final external validation after the
three descriptor families are compared.  This target-dependent utility
is explicitly a non-causal screening proxy, not a deployable router label.

Separately, the remaining correction ``target - depth3`` is projected onto
central forces on first-neighbour C--C bonds.  ``numpy.linalg.lstsq`` supplies
minimum-norm bond tensions for the repair-predictability LOCO test.  Three fixed
representation families are compared with fixed-alpha Ridge probes:

* exchange-invariant pairs of frozen MACE scalar invariants (the primary test),
* exchange-invariant local distance/angle geometry (a geometry control), and
* the first-neighbour bond length alone (a stricter control).

The router is evaluated at an operating point fixed by at most 5% activation
on harmonic validation bonds.  Support and fixed-smearing seed-2 retention are reported both
as positive-utility retention and target-delta radial-tension coverage.  A
separate ridge probe tests whether the signed remaining bond tension is
predictable.  No new DFT labels are made or read beyond the already frozen
development files.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.build import graphene
from ase.io import read
from mace.data import AtomicData, config_from_atoms
from mace.data.utils import KeySpecification
from mace.tools import AtomicNumberTable, torch_geometric
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


FIXED_DEGAUSS_RY = 0.0019000869380739254
HARMONIC_ACTIVATION_MAX = 0.05
SUPPORT_COVERAGE_MIN = 0.80
SUPPORT_PER_CONFIG_COVERAGE_MIN = 0.50
TENSION_SIGN_THRESHOLD_MEV_A = 30.0
TENSION_SIGN_ACCURACY_MIN = 0.80
TENSION_WEIGHTED_R2_MIN = 0.50
TENSION_PER_CONFIG_SIGN_ACCURACY_MIN = 0.65
TENSION_PER_CONFIG_WEIGHTED_R2_MIN = 0.0
THERMAL_UTILITY_RETENTION_MIN = 0.80
HARMFUL_UTILITY_EXPOSURE_MAX = 0.20
THERMAL_TARGET_TENSION_COVERAGE_MIN = 0.80
HARMONIC_FORCE_RMSE_MAX_MEV_A = 9.044673057081592
THERMAL_FORCE_RMSE_MAX_MEV_A = 30.0
THERMAL_FORCE_MAX_ABS_MAX_MEV_A = 200.0
APRIME_PROJECTED_FORCE_RMS_MAX_MEV_A = 15.0
NET_FORCE_NORM_MAX_MEV_A = 1.0
RADIAL_PROJECTION_SQUARED_FRACTION_MIN = 0.50
FEATURE_FAMILIES = ("mace_invariants", "geometry_control", "bond_length_control")
FEATURE_MODE_SIGNED_L0 = "signed_l0"
FEATURE_MODE_TENSOR_POWER = "signed_l0_plus_tensor_power"
FEATURE_MODE_GRAM_SKETCH16 = (
    "signed_l0_plus_tensor_power_plus_gram_sketch16"
)
FEATURE_MODES = (
    FEATURE_MODE_SIGNED_L0,
    FEATURE_MODE_TENSOR_POWER,
    FEATURE_MODE_GRAM_SKETCH16,
)
FEATURE_SCHEMA_VERSION = "mace_node_invariant_features_v1"
GRAM_SKETCH16_TAG = "R2K_GRAM_SKETCH_V1"
GRAM_SKETCH16_CHANNELS = 16
GRAM_SKETCH16_OUTPUTS = 16
GRAM_SKETCH16_R_SHA256 = (
    "c2a4ac2867047609ca539a384556bd5ec1b204a9c65c736758b28b6deef7a044"
)
GRAM_SKETCH16_PHI_SHA256 = (
    "929a48555356ff84cf5246adc1facd5f55e75f81e4c1f5bbefc0572ab210717b"
)
ROUTER_FEATURE_MAP_LINEAR = "linear"
ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC = "diagonal_quadratic"
ROUTER_FEATURE_MAPS = (
    ROUTER_FEATURE_MAP_LINEAR,
    ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC,
)
ROUTER_FEATURE_MAP_SCHEMA_VERSION = "router_feature_map_v1"
FEATURE_RELATIVE_STD_EPS_MULTIPLIER = 64.0
FEATURE_RELATIVE_STD_THRESHOLD = float(
    FEATURE_RELATIVE_STD_EPS_MULTIPLIER * np.finfo(np.float32).eps
)


@dataclass
class Record:
    """Geometry, frozen representation, and optional radial target for one config."""

    split: str
    source_index: int
    label: str
    temperature_K: float | None
    atom_invariants: np.ndarray
    geometry_invariants: np.ndarray
    atom_i: np.ndarray
    atom_j: np.ndarray
    distance_A: np.ndarray
    features: dict[str, np.ndarray]
    router_features: dict[str, np.ndarray]
    target_force_meV_A: np.ndarray
    remaining_force_meV_A: np.ndarray
    atom_utility_meV2_A2: np.ndarray
    bond_utility_meV2_A2: np.ndarray
    aprime_mode: np.ndarray | None = None
    target_tension_meV_A: np.ndarray | None = None
    remaining_tension_meV_A: np.ndarray | None = None
    target_radial: dict | None = None
    remaining_radial: dict | None = None
    force_utility: dict | None = None


@dataclass
class Probe:
    scaler: StandardScaler
    model: Ridge
    router_feature_map: str | None = None
    input_scaler: StandardScaler | None = None
    input_dimension: int | None = None

    def predict(self, values: np.ndarray) -> np.ndarray:
        mapped = numpy_router_feature_map(
            values,
            self.router_feature_map or ROUTER_FEATURE_MAP_LINEAR,
            self.input_scaler,
        )
        return np.asarray(self.model.predict(self.scaler.transform(mapped)), float)

    def state(self) -> dict:
        result = {
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "ridge_coefficient": np.asarray(self.model.coef_, float).tolist(),
            "ridge_intercept": float(self.model.intercept_),
        }
        # Tension probes retain their historical state.  Router states include
        # an explicit map schema, while replaying an older state with no map
        # field continues to mean the historical linear path.
        if self.router_feature_map is None:
            return result
        if self.input_dimension is None:
            raise RuntimeError("router probe has no recorded input dimension")
        schema = router_feature_map_schema(
            self.router_feature_map, self.input_dimension
        )
        result.update(
            {
                "router_feature_map": self.router_feature_map,
                "router_input_feature_dimension": int(self.input_dimension),
                "router_mapped_feature_dimension": int(
                    schema["output_dimension"]
                ),
                "router_feature_map_schema": schema,
                "router_feature_map_schema_sha256": schema["schema_sha256"],
            }
        )
        if self.router_feature_map == ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC:
            if self.input_scaler is None:
                raise RuntimeError(
                    "diagonal-quadratic router has no input StandardScaler"
                )
            result.update(
                {
                    "input_scaler_mean": self.input_scaler.mean_.tolist(),
                    "input_scaler_scale": self.input_scaler.scale_.tolist(),
                }
            )
        return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_value(value):
    """Return RFC-compliant JSON data, mapping non-finite diagnostics to null."""
    if isinstance(value, dict):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json_value(value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(strict_json_value(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def canonical_json_sha256(payload: dict) -> str:
    """Hash a JSON-compatible replay schema independently of file formatting."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def float64_little_endian_c_sha256(values: np.ndarray) -> str:
    """Hash the canonical little-endian float64 C-order byte representation."""
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def fixed_gram_sketch16_specification() -> dict:
    """Return the predeclared cross-channel Gram sketch and replay metadata.

    The projection is deliberately independent of all labels and descriptor
    values.  Its signs are fixed one at a time by a SHA-256 bit, and the
    off-diagonal channel-pair order is lexicographic.  Storing both ``R`` and
    the induced ``Phi`` makes it possible to audit that no diagonal Gram term
    entered the added sketch columns.
    """
    projection = np.asarray(
        [
            [
                (
                    0.25
                    if hashlib.sha256(
                        f"{GRAM_SKETCH16_TAG}|{sketch}|{channel}".encode("ascii")
                    ).digest()[0]
                    & 1
                    else -0.25
                )
                for channel in range(GRAM_SKETCH16_CHANNELS)
            ]
            for sketch in range(GRAM_SKETCH16_OUTPUTS)
        ],
        dtype=np.dtype("<f8"),
    )
    pair_order = [
        [left, right]
        for left in range(GRAM_SKETCH16_CHANNELS)
        for right in range(left + 1, GRAM_SKETCH16_CHANNELS)
    ]
    phi = np.asarray(
        [
            [
                2.0 * projection[sketch, left] * projection[sketch, right]
                for left, right in pair_order
            ]
            for sketch in range(GRAM_SKETCH16_OUTPUTS)
        ],
        dtype=np.dtype("<f8"),
    )
    projection_hash = float64_little_endian_c_sha256(projection)
    phi_hash = float64_little_endian_c_sha256(phi)
    if projection_hash != GRAM_SKETCH16_R_SHA256:
        raise RuntimeError(
            "fixed Gram-sketch R generation disagrees with the declared SHA-256"
        )
    if phi_hash != GRAM_SKETCH16_PHI_SHA256:
        raise RuntimeError(
            "fixed Gram-sketch Phi generation disagrees with the declared SHA-256"
        )
    return {
        "tag": GRAM_SKETCH16_TAG,
        "selection_status": "fixed_before_descriptor_screening",
        "selection_rationale": (
            "the unique interaction0 16x1o block of the r_max=3, h16, l_max=1 "
            "descriptor is the predeclared cross-channel source; l_max=2 and "
            "ambiguous layouts are rejected rather than selected after seeing data"
        ),
        "source_interaction": 0,
        "source_irrep": "1o",
        "source_multiplicity": GRAM_SKETCH16_CHANNELS,
        "source_irrep_dimension": 3,
        "K": GRAM_SKETCH16_OUTPUTS,
        "R": projection.tolist(),
        "R_shape": list(projection.shape),
        "R_dtype": "float64_little_endian",
        "R_storage_order": "C",
        "R_sha256": projection_hash,
        "R_element_definition": (
            "+1/4 if SHA256('R2K_GRAM_SKETCH_V1|k|a').digest()[0] "
            "has least-significant bit 1, else -1/4"
        ),
        "offdiagonal_pair_order": pair_order,
        "offdiagonal_pair_order_definition": (
            "lexicographic (a,b), a=0..14 and b=a+1..15"
        ),
        "Phi": phi.tolist(),
        "Phi_shape": list(phi.shape),
        "Phi_dtype": "float64_little_endian",
        "Phi_storage_order": "C",
        "Phi_sha256": phi_hash,
        "Phi_definition": "Phi[k,pair(a,b)] = 2*R[k,a]*R[k,b]",
        "gram_definition": "G[a,b] = mean_m H[a,m]*H[b,m]",
        "output_definition": (
            "c[k] = sum_{a<b} Phi[k,pair(a,b)]*G[a,b]"
        ),
        "legacy_tensor_power_retained": True,
        "diagonal_self_power_in_gram_sketch": False,
        "gram_normalization": "mean over the three l=1 components",
        "paired_router_feature_map": ROUTER_FEATURE_MAP_LINEAR,
    }


def gram_sketch16_source_block(layout: list) -> tuple[int, dict]:
    """Validate the one supported r3-h16-l1 irrep layout and return 16x1o."""
    candidates = [
        (index, item)
        for index, item in enumerate(layout)
        if int(item["interaction"]) == 0 and int(item["l"]) == 1
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Gram-sketch mode requires one unique interaction0 l=1 block; "
            f"observed {len(candidates)}"
        )
    block_index, source = candidates[0]
    source_signature = (
        int(source["multiplicity"]),
        int(source["dimension"]),
        int(source["parity"]),
        str(source["irrep"]),
    )
    if source_signature != (GRAM_SKETCH16_CHANNELS, 3, -1, "1o"):
        raise ValueError(
            "Gram-sketch mode requires interaction0 16x1o exactly; observed "
            f"multiplicity={source_signature[0]}, dimension={source_signature[1]}, "
            f"parity={source_signature[2]}, irrep={source_signature[3]!r}"
        )
    nonscalar = [item for item in layout if int(item["l"]) > 0]
    if len(nonscalar) != 1 or nonscalar[0] is not source:
        raise ValueError(
            "Gram-sketch mode supports the fixed l_max=1 descriptor layout only; "
            "additional non-scalar irrep blocks are not allowed"
        )
    scalar_signatures = sorted(
        (
            int(item["interaction"]),
            int(item["multiplicity"]),
            int(item["dimension"]),
            int(item["parity"]),
            str(item["irrep"]),
        )
        for item in layout
        if int(item["l"]) == 0
    )
    if scalar_signatures != [
        (0, GRAM_SKETCH16_CHANNELS, 1, 1, "0e"),
        (1, GRAM_SKETCH16_CHANNELS, 1, 1, "0e"),
    ]:
        raise ValueError(
            "Gram-sketch mode supports exactly the r3-h16-l1 two-interaction "
            "layout with interaction0 and interaction1 16x0e scalar blocks"
        )
    return block_index, source


def validated_gram_sketch16_arrays(
    schema: dict, block: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and load R, pair order, and Phi from a serialized schema."""
    specification = schema.get("gram_sketch")
    if not isinstance(specification, dict):
        raise ValueError("Gram-sketch feature schema has no gram_sketch specification")
    expected = fixed_gram_sketch16_specification()
    fixed_metadata = (
        "tag",
        "source_interaction",
        "source_irrep",
        "source_multiplicity",
        "source_irrep_dimension",
        "K",
        "R_shape",
        "R_dtype",
        "R_storage_order",
        "R_sha256",
        "offdiagonal_pair_order",
        "Phi_shape",
        "Phi_dtype",
        "Phi_storage_order",
        "Phi_sha256",
        "legacy_tensor_power_retained",
        "diagonal_self_power_in_gram_sketch",
        "paired_router_feature_map",
    )
    for key in fixed_metadata:
        if specification.get(key) != expected[key]:
            raise ValueError(f"Gram-sketch schema field {key!r} is not canonical")
    projection = np.ascontiguousarray(
        specification.get("R"), dtype=np.dtype("<f8")
    )
    phi = np.ascontiguousarray(
        specification.get("Phi"), dtype=np.dtype("<f8")
    )
    pairs = np.asarray(specification["offdiagonal_pair_order"], dtype=np.int64)
    if projection.shape != (GRAM_SKETCH16_OUTPUTS, GRAM_SKETCH16_CHANNELS):
        raise ValueError("Gram-sketch R shape disagrees with its specification")
    if phi.shape != (GRAM_SKETCH16_OUTPUTS, 120):
        raise ValueError("Gram-sketch Phi shape disagrees with its specification")
    if pairs.shape != (120, 2):
        raise ValueError("Gram-sketch off-diagonal pair order has the wrong shape")
    if not np.all(np.isfinite(projection)) or not np.all(np.isfinite(phi)):
        raise ValueError("Gram-sketch projection contains a non-finite value")
    if float64_little_endian_c_sha256(projection) != GRAM_SKETCH16_R_SHA256:
        raise ValueError("Gram-sketch R raw-byte SHA-256 mismatch")
    if float64_little_endian_c_sha256(phi) != GRAM_SKETCH16_PHI_SHA256:
        raise ValueError("Gram-sketch Phi raw-byte SHA-256 mismatch")
    reconstructed_phi = np.asarray(
        [
            [2.0 * projection[k, left] * projection[k, right] for left, right in pairs]
            for k in range(GRAM_SKETCH16_OUTPUTS)
        ],
        dtype=np.dtype("<f8"),
    )
    if not np.array_equal(phi, reconstructed_phi):
        raise ValueError("Gram-sketch Phi is not the off-diagonal map induced by R")
    if (
        int(block.get("raw_layout_block_index", -1)) != 1
        or int(block.get("interaction", -1)) != 0
        or int(block.get("l", -1)) != 1
        or int(block.get("parity", 0)) != -1
        or int(block.get("multiplicity", -1)) != GRAM_SKETCH16_CHANNELS
        or int(block.get("irrep_dimension", -1)) != 3
        or str(block.get("irrep")) != "1o"
        or block.get("raw_slice") != [16, 64]
        or block.get("raw_reshape") != [GRAM_SKETCH16_CHANNELS, 3]
        or block.get("output_slice") != [48, 64]
        or block.get("operation") != "offdiagonal_gram_sketch16"
    ):
        raise ValueError("Gram-sketch replay block is not the fixed interaction0 16x1o block")
    return projection, pairs, phi


def router_feature_map_schema(feature_map: str, input_dimension: int) -> dict:
    """Describe and hash the fixed atom-router feature map."""
    if feature_map not in ROUTER_FEATURE_MAPS:
        raise ValueError(
            f"unknown router feature map {feature_map!r}; "
            f"expected one of {ROUTER_FEATURE_MAPS}"
        )
    if input_dimension <= 0:
        raise ValueError("router feature-map input dimension must be positive")
    if feature_map == ROUTER_FEATURE_MAP_LINEAR:
        output_dimension = input_dimension
        operations = ["x"]
        preprocessing = "none"
    else:
        output_dimension = 2 * input_dimension
        operations = ["z", "z_squared"]
        preprocessing = "z = (x - input_scaler_mean) / input_scaler_scale"
    result = {
        "version": ROUTER_FEATURE_MAP_SCHEMA_VERSION,
        "name": feature_map,
        "input_dimension": int(input_dimension),
        "output_dimension": int(output_dimension),
        "output_block_order": operations,
        "input_preprocessing": preprocessing,
        "mapped_feature_standardization": (
            "StandardScaler fitted after the fixed feature map"
        ),
    }
    result["schema_sha256"] = canonical_json_sha256(result)
    return result


def numpy_router_feature_map(
    values: np.ndarray,
    feature_map: str,
    input_scaler: StandardScaler | None = None,
) -> np.ndarray:
    """Apply the fixed router map before its second StandardScaler."""
    # Preserve the input dtype on the linear path so the default audit remains
    # numerically identical to the historical scaler.transform(values) call.
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("router features must be a nonempty 2D matrix")
    if feature_map == ROUTER_FEATURE_MAP_LINEAR:
        if input_scaler is not None:
            raise ValueError("linear router must not have an input scaler")
        return values
    if feature_map != ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC:
        raise ValueError(
            f"unknown router feature map {feature_map!r}; "
            f"expected one of {ROUTER_FEATURE_MAPS}"
        )
    if input_scaler is None:
        raise ValueError("diagonal-quadratic router requires an input scaler")
    standardized = np.asarray(input_scaler.transform(values), float)
    return np.concatenate([standardized, standardized**2], axis=1)


def _router_map_from_state_numpy(
    values: np.ndarray, router_state: dict
) -> np.ndarray:
    """Replay the mapped features from a serialized router state."""
    feature_map = str(
        router_state.get("router_feature_map", ROUTER_FEATURE_MAP_LINEAR)
    )
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("router features must be a nonempty 2D matrix")
    recorded_input_dimension = int(
        router_state.get("router_input_feature_dimension", values.shape[1])
    )
    if recorded_input_dimension != values.shape[1]:
        raise ValueError("router input feature dimension disagrees with state")
    if "router_feature_map_schema" in router_state:
        map_schema = copy.deepcopy(router_state["router_feature_map_schema"])
        recorded_hash = map_schema.pop("schema_sha256", None)
        if recorded_hash is None or canonical_json_sha256(map_schema) != recorded_hash:
            raise ValueError("router feature-map schema SHA-256 mismatch")
        if recorded_hash != router_state.get("router_feature_map_schema_sha256"):
            raise ValueError("router feature-map state/schema SHA-256 mismatch")
        if map_schema["name"] != feature_map:
            raise ValueError("router feature-map state/schema name mismatch")
        if int(map_schema["input_dimension"]) != values.shape[1]:
            raise ValueError("router feature-map input dimension disagrees with state")
    if feature_map == ROUTER_FEATURE_MAP_LINEAR:
        mapped = values
    elif feature_map == ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC:
        mean = np.asarray(router_state.get("input_scaler_mean"), float)
        scale = np.asarray(router_state.get("input_scaler_scale"), float)
        if mean.ndim != 1 or scale.ndim != 1 or mean.size != values.shape[1]:
            raise ValueError("router input scaler dimension disagrees with input features")
        if scale.size != values.shape[1]:
            raise ValueError("router input scaler dimension disagrees with input features")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("router input scaler contains a non-finite value")
        if np.any(scale <= 0.0):
            raise ValueError("router input scaler contains a negative or zero scale")
        standardized = (values - mean) / scale
        mapped = np.concatenate([standardized, standardized**2], axis=1)
    else:
        raise ValueError(f"unsupported serialized router feature map {feature_map!r}")
    recorded_mapped_dimension = int(
        router_state.get("router_mapped_feature_dimension", mapped.shape[1])
    )
    if recorded_mapped_dimension != mapped.shape[1]:
        raise ValueError("router mapped feature dimension disagrees with state")
    if "router_feature_map_schema" in router_state:
        if (
            int(router_state["router_feature_map_schema"]["output_dimension"])
            != mapped.shape[1]
        ):
            raise ValueError("router feature-map output dimension disagrees with state")
    return mapped


def replay_router_score_numpy(values: np.ndarray, router_state: dict) -> np.ndarray:
    """Replay either router map plus its mapped-space scaler and Ridge."""
    mapped = _router_map_from_state_numpy(values, router_state)
    mean = np.asarray(router_state["scaler_mean"], float)
    scale = np.asarray(router_state["scaler_scale"], float)
    coefficient = np.asarray(router_state["ridge_coefficient"], float)
    if any(array.ndim != 1 for array in (mean, scale, coefficient)):
        raise ValueError("router state vectors must be one-dimensional")
    if not (mean.size == scale.size == coefficient.size == mapped.shape[1]):
        raise ValueError("router mapped scaler/Ridge dimension disagrees with feature map")
    if not all(np.all(np.isfinite(array)) for array in (mean, scale, coefficient)):
        raise ValueError("router mapped scaler/Ridge contains a non-finite value")
    if np.any(scale <= 0.0):
        raise ValueError("router mapped scaler contains a negative or zero scale")
    intercept = float(router_state["ridge_intercept"])
    if not np.isfinite(intercept):
        raise ValueError("router Ridge intercept is non-finite")
    return np.asarray((mapped - mean) / scale @ coefficient, float) + intercept


def boolean_mask_sha256(mask: list[bool]) -> str:
    if not mask or any(type(value) is not bool for value in mask):
        raise ValueError("feature mask must be a nonempty list of booleans")
    return hashlib.sha256(
        json.dumps(mask, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def role(structure: Atoms) -> str:
    return str(structure.info.get("delta_target_role", "unknown"))


def geometry_fingerprint(structure: Atoms) -> str:
    """Order-sensitive fingerprint used only to disclose exact geometry overlap."""
    digest = hashlib.sha256()
    digest.update(np.asarray(structure.numbers, np.int64).tobytes())
    digest.update(np.round(np.asarray(structure.cell, float), 10).tobytes())
    digest.update(np.round(np.asarray(structure.positions, float), 10).tobytes())
    return digest.hexdigest()


def minimum_image_vectors(structure: Atoms) -> np.ndarray:
    positions = np.asarray(structure.positions, float)
    cell = np.asarray(structure.cell, float)
    if abs(np.linalg.det(cell)) < 1.0e-12:
        raise ValueError("the separability audit requires a nonsingular cell")
    vectors = positions[None, :, :] - positions[:, None, :]
    fractional = vectors @ np.linalg.inv(cell)
    pbc = np.asarray(structure.pbc, bool)
    fractional[:, :, pbc] -= np.round(fractional[:, :, pbc])
    return fractional @ cell


def first_neighbour_bonds(
    structure: Atoms, cutoff_A: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unique first-neighbour bonds and all pair distances/vectors."""
    vectors = minimum_image_vectors(structure)
    distances = np.linalg.norm(vectors, axis=2)
    np.fill_diagonal(distances, np.inf)
    atom_i, atom_j = np.where(np.triu(distances < cutoff_A, k=1))
    if len(atom_i) == 0:
        raise ValueError("no first-neighbour bonds found")
    degree = np.bincount(
        np.concatenate([atom_i, atom_j]), minlength=len(structure)
    )
    if np.any(degree != 3):
        detail = Counter(int(value) for value in degree)
        raise ValueError(
            f"first-neighbour cutoff {cutoff_A:g} A does not give degree 3: {detail}"
        )
    bond_vectors = vectors[atom_i, atom_j]
    bond_distances = distances[atom_i, atom_j]
    units = bond_vectors / bond_distances[:, None]
    return atom_i, atom_j, bond_distances, units, distances


def atom_geometry_invariants(
    structure: Atoms, distances: np.ndarray, n_distances: int = 6
) -> np.ndarray:
    """Sorted local distances and first-shell angles; all are invariant scalars."""
    if len(structure) <= n_distances:
        raise ValueError("not enough atoms for the requested geometry descriptor")
    order = np.argsort(distances, axis=1)
    nearest = np.take_along_axis(distances, order[:, :n_distances], axis=1)
    vectors = minimum_image_vectors(structure)
    first = np.take_along_axis(vectors, order[:, :3, None], axis=1)
    lengths = np.linalg.norm(first, axis=2)
    unit = first / lengths[:, :, None]
    cosines = np.stack(
        [
            np.sum(unit[:, 0] * unit[:, 1], axis=1),
            np.sum(unit[:, 0] * unit[:, 2], axis=1),
            np.sum(unit[:, 1] * unit[:, 2], axis=1),
        ],
        axis=1,
    )
    cosines.sort(axis=1)
    return np.concatenate([nearest, cosines], axis=1)


def mace_invariant_layout(
    model: torch.nn.Module,
    feature_mode: str = FEATURE_MODE_SIGNED_L0,
) -> tuple[list, int, list[str]]:
    """Describe scalar/squared-norm invariants in concatenated MACE node_feats."""
    if feature_mode == FEATURE_MODE_GRAM_SKETCH16:
        if not np.isclose(float(model.r_max), 3.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Gram-sketch mode is fixed to the r_max=3 descriptor; observed "
                f"r_max={float(model.r_max):g}"
            )
        if int(model.num_interactions) != 2:
            raise ValueError(
                "Gram-sketch mode is fixed to the two-interaction r3-h16-l1 "
                f"descriptor; observed {int(model.num_interactions)} interactions"
            )
    layout = []
    raw_offset = 0
    invariant_names: list[str] = []
    for interaction, product in enumerate(model.products):
        irreps = product.linear.irreps_out
        for multiplicity, irrep in irreps:
            width = int(multiplicity) * int(irrep.dim)
            layout.append(
                {
                    "interaction": interaction,
                    "start": raw_offset,
                    "stop": raw_offset + width,
                    "multiplicity": int(multiplicity),
                    "dimension": int(irrep.dim),
                    "l": int(irrep.l),
                    "parity": int(irrep.p),
                    "irrep": str(irrep),
                }
            )
            if int(irrep.l) == 0:
                invariant_names.extend(
                    f"interaction{interaction}_{irrep}_{channel:03d}_value"
                    for channel in range(int(multiplicity))
                )
            raw_offset += width
    schema = invariant_feature_schema(layout, raw_offset, feature_mode)
    return layout, raw_offset, list(schema["feature_names"])


def invariant_feature_schema(
    layout: list,
    expected_width: int,
    feature_mode: str = FEATURE_MODE_SIGNED_L0,
) -> dict:
    """Build the exact raw-slice to invariant-column replay specification.

    ``signed_l0`` is the historical R2K representation.  The tensor-power
    modes preserve those signed scalar columns and append one
    ``mean_m h_m**2`` invariant for every multiplicity channel of every
    non-scalar irrep block.  Dividing the squared norm by ``2*l+1`` avoids an
    artificial angular-dimension scale factor.  The fixed Gram-sketch mode
    then appends 16 predeclared off-diagonal cross-channel contractions of the
    unique interaction-0 16x1o block.  No geometry or pooling enters any mode.
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(
            f"unknown MACE feature mode {feature_mode!r}; expected one of {FEATURE_MODES}"
        )
    raw_cursor = 0
    for block_index, item in enumerate(layout):
        required_keys = {
            "interaction",
            "start",
            "stop",
            "multiplicity",
            "dimension",
            "l",
            "parity",
            "irrep",
        }
        missing = required_keys.difference(item)
        if missing:
            raise ValueError(
                f"raw irrep block {block_index} is missing keys {sorted(missing)}"
            )
        start = int(item["start"])
        stop = int(item["stop"])
        multiplicity = int(item["multiplicity"])
        dimension = int(item["dimension"])
        angular_momentum = int(item["l"])
        parity = int(item["parity"])
        if start != raw_cursor:
            raise ValueError(
                f"raw irrep blocks are not contiguous at block {block_index}: "
                f"expected start {raw_cursor}, observed {start}"
            )
        if multiplicity <= 0 or angular_momentum < 0 or parity not in {-1, 1}:
            raise ValueError(f"invalid irrep metadata in raw block {block_index}")
        if dimension != 2 * angular_momentum + 1:
            raise ValueError(
                f"raw block {block_index} has dimension {dimension}, expected "
                f"2*l+1={2 * angular_momentum + 1}"
            )
        if stop - start != multiplicity * dimension:
            raise ValueError(
                f"raw block {block_index} width disagrees with multiplicity*dimension"
            )
        if angular_momentum == 0 and parity != 1:
            raise ValueError(
                "signed l=0 extraction encountered 0o; a pseudoscalar is not an "
                "O(3)-invariant scalar and cannot enter this router"
            )
        raw_cursor = stop
    if raw_cursor != int(expected_width):
        raise ValueError(
            f"raw irrep layout ends at {raw_cursor}, expected node width {expected_width}"
        )

    gram_source = None
    if feature_mode == FEATURE_MODE_GRAM_SKETCH16:
        gram_source = gram_sketch16_source_block(layout)

    blocks = []
    feature_names: list[str] = []
    output_offset = 0
    selected_block_indices = [
        index for index, item in enumerate(layout) if item["l"] == 0
    ]
    if feature_mode in (FEATURE_MODE_TENSOR_POWER, FEATURE_MODE_GRAM_SKETCH16):
        selected_block_indices.extend(
            index for index, item in enumerate(layout) if item["l"] > 0
        )
    for block_index in selected_block_indices:
        item = layout[block_index]
        if item["l"] == 0:
            operation = "signed_l0_value"
            suffix = "value"
        else:
            operation = "mean_m_squared"
            suffix = "mean_m_h2"
        multiplicity = int(item["multiplicity"])
        names = [
            (
                f"interaction{item['interaction']}_{item['irrep']}_"
                f"{channel:03d}_{suffix}"
            )
            for channel in range(multiplicity)
        ]
        blocks.append(
            {
                "raw_layout_block_index": block_index,
                "interaction": int(item["interaction"]),
                "irrep": item["irrep"],
                "l": int(item["l"]),
                "parity": int(item["parity"]),
                "multiplicity": multiplicity,
                "irrep_dimension": int(item["dimension"]),
                "raw_slice": [int(item["start"]), int(item["stop"])],
                "raw_reshape": [multiplicity, int(item["dimension"])],
                "output_slice": [output_offset, output_offset + multiplicity],
                "operation": operation,
                "power_normalization": (
                    None
                    if int(item["l"]) == 0
                    else f"1/(2*l+1)=1/{int(item['dimension'])}"
                ),
                "feature_names": names,
            }
        )
        feature_names.extend(names)
        output_offset += multiplicity
    gram_sketch = None
    if feature_mode == FEATURE_MODE_GRAM_SKETCH16:
        if output_offset != 48:
            raise ValueError(
                "Gram-sketch mode requires the exact 48-column signed-l0 plus "
                f"tensor-power prefix; observed {output_offset} columns"
            )
        if gram_source is None:
            raise RuntimeError("Gram-sketch source validation was not run")
        block_index, item = gram_source
        gram_sketch = fixed_gram_sketch16_specification()
        gram_names = [
            f"interaction0_16x1o_offdiag_gram_sketch_{index:03d}"
            for index in range(GRAM_SKETCH16_OUTPUTS)
        ]
        blocks.append(
            {
                "raw_layout_block_index": int(block_index),
                "interaction": 0,
                "irrep": "1o",
                "l": 1,
                "parity": -1,
                "multiplicity": GRAM_SKETCH16_CHANNELS,
                "irrep_dimension": 3,
                "raw_slice": [int(item["start"]), int(item["stop"])],
                "raw_reshape": [GRAM_SKETCH16_CHANNELS, 3],
                "output_slice": [
                    output_offset,
                    output_offset + GRAM_SKETCH16_OUTPUTS,
                ],
                "operation": "offdiagonal_gram_sketch16",
                "power_normalization": "mean_m=1/3 inside G[a,b]",
                "feature_names": gram_names,
            }
        )
        feature_names.extend(gram_names)
        output_offset += GRAM_SKETCH16_OUTPUTS
    identity_mask = [True] * output_offset
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_mode": feature_mode,
        "raw_node_feats_width": int(expected_width),
        "unpruned_output_dimension": output_offset,
        "output_dimension": output_offset,
        "blocks": blocks,
        "unpruned_feature_names": feature_names,
        "feature_names": feature_names,
        "feature_mask": identity_mask,
        "feature_mask_sha256": boolean_mask_sha256(identity_mask),
        "retained_feature_indices": list(range(output_offset)),
        "feature_pruning": {
            "status": "identity_before_training_only_mask_fit",
            "method": "relative_population_std_over_RMS",
            "relative_std_threshold": FEATURE_RELATIVE_STD_THRESHOLD,
            "float32_epsilon_multiplier": FEATURE_RELATIVE_STD_EPS_MULTIPLIER,
            "input_dimension": output_offset,
            "retained_dimension": output_offset,
        },
        "torch_replay_function": "torch_invariant_features",
    }
    if gram_sketch is not None:
        payload["gram_sketch"] = gram_sketch
    payload["schema_sha256"] = canonical_json_sha256(payload)
    return payload


def fit_relative_variance_feature_mask(
    values: np.ndarray,
    feature_names: list[str],
    relative_std_threshold: float = FEATURE_RELATIVE_STD_THRESHOLD,
) -> dict:
    """Fit a numerical-rank mask using training features only.

    An absolute standard-deviation cutoff would incorrectly remove a channel
    whose values and variations are both small.  The population standard
    deviation is therefore compared with that channel's RMS amplitude.  The
    fixed threshold is 64 float32 eps: variations below it do not retain
    enough relative precision for a float32 conservative-autograd replay.
    """
    matrix = np.asarray(values, float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError("feature-mask fit requires a nonempty 2D training matrix")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature-mask matrix width disagrees with feature names")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature-mask training matrix contains non-finite values")
    if not (0.0 < relative_std_threshold < 1.0):
        raise ValueError("relative feature-variation threshold must lie in (0, 1)")
    mean = np.mean(matrix, axis=0)
    population_std = np.std(matrix, axis=0, ddof=0)
    rms = np.sqrt(np.mean(np.square(matrix), axis=0))
    relative_std = np.divide(
        population_std,
        rms,
        out=np.zeros_like(population_std),
        where=rms > 0.0,
    )
    retained = relative_std > relative_std_threshold
    if not np.any(retained):
        raise RuntimeError("relative-variance pruning would remove every feature")
    mask = [bool(value) for value in retained]
    retained_indices = np.flatnonzero(retained).astype(int).tolist()
    removed_indices = np.flatnonzero(~retained).astype(int).tolist()
    payload = {
        "status": "fitted_on_probe_training_records_only",
        "method": "relative_population_std_over_RMS",
        "relative_std_definition": "population_std / sqrt(mean(feature**2))",
        "relative_std_threshold": float(relative_std_threshold),
        "float32_epsilon": float(np.finfo(np.float32).eps),
        "float32_epsilon_multiplier": FEATURE_RELATIVE_STD_EPS_MULTIPLIER,
        "fit_sample_count": int(matrix.shape[0]),
        "input_dimension": int(matrix.shape[1]),
        "retained_dimension": int(np.sum(retained)),
        "removed_dimension": int(np.sum(~retained)),
        "feature_mask": mask,
        "feature_mask_sha256": boolean_mask_sha256(mask),
        "retained_feature_indices": retained_indices,
        "removed_feature_indices": removed_indices,
        "retained_feature_names": [feature_names[index] for index in retained_indices],
        "removed_feature_names": [feature_names[index] for index in removed_indices],
        "training_mean": mean.tolist(),
        "training_population_std": population_std.tolist(),
        "training_RMS": rms.tolist(),
        "training_relative_std": relative_std.tolist(),
        "minimum_retained_relative_std": float(np.min(relative_std[retained])),
        "maximum_removed_relative_std": (
            float(np.max(relative_std[~retained])) if np.any(~retained) else None
        ),
    }
    payload["pruning_state_sha256"] = canonical_json_sha256(payload)
    return payload


def feature_schema_with_pruning(schema: dict, pruning: dict) -> dict:
    """Freeze a training-only mask into the NumPy/Torch replay schema."""
    if "pruning_state_sha256" not in pruning:
        raise ValueError("feature pruning state has no SHA-256")
    unhashed_pruning = copy.deepcopy(pruning)
    observed_pruning_hash = unhashed_pruning.pop("pruning_state_sha256")
    if canonical_json_sha256(unhashed_pruning) != observed_pruning_hash:
        raise ValueError("feature pruning state SHA-256 mismatch")
    result = copy.deepcopy(schema)
    result.pop("schema_sha256", None)
    unpruned_names = list(
        result.get("unpruned_feature_names", result["feature_names"])
    )
    input_dimension = int(result["unpruned_output_dimension"])
    mask = list(pruning["feature_mask"])
    if len(mask) != input_dimension or len(unpruned_names) != input_dimension:
        raise ValueError("pruning mask dimension disagrees with invariant schema")
    if boolean_mask_sha256(mask) != pruning["feature_mask_sha256"]:
        raise ValueError("pruning feature-mask hash mismatch")
    if int(pruning["input_dimension"]) != input_dimension:
        raise ValueError("pruning input dimension disagrees with invariant schema")
    retained_indices = [index for index, keep in enumerate(mask) if keep]
    if retained_indices != list(pruning["retained_feature_indices"]):
        raise ValueError("pruning retained indices disagree with feature mask")
    if len(retained_indices) != int(pruning["retained_dimension"]):
        raise ValueError("pruning retained dimension disagrees with feature mask")
    result.update(
        {
            "output_dimension": len(retained_indices),
            "feature_names": [unpruned_names[index] for index in retained_indices],
            "feature_mask": mask,
            "feature_mask_sha256": pruning["feature_mask_sha256"],
            "retained_feature_indices": retained_indices,
            "retained_training_population_std": [
                pruning["training_population_std"][index]
                for index in retained_indices
            ],
            "retained_training_RMS": [
                pruning["training_RMS"][index] for index in retained_indices
            ],
            "retained_training_relative_std": [
                pruning["training_relative_std"][index]
                for index in retained_indices
            ],
            "feature_pruning": copy.deepcopy(pruning),
        }
    )
    result["schema_sha256"] = canonical_json_sha256(result)
    return result


def feature_mask_from_schema(schema: dict) -> np.ndarray:
    """Validate and return the exact unpruned-to-retained replay mask."""
    if "schema_sha256" not in schema:
        raise ValueError("feature schema has no SHA-256")
    unhashed_schema = copy.deepcopy(schema)
    observed_schema_hash = unhashed_schema.pop("schema_sha256")
    if canonical_json_sha256(unhashed_schema) != observed_schema_hash:
        raise ValueError("feature schema SHA-256 mismatch")
    mask_values = schema.get(
        "feature_mask", [True] * int(schema["unpruned_output_dimension"])
    )
    if not isinstance(mask_values, list):
        raise ValueError("feature schema mask is not a list")
    expected_dimension = int(schema["unpruned_output_dimension"])
    if len(mask_values) != expected_dimension:
        raise ValueError("feature schema mask length disagrees with unpruned dimension")
    observed_hash = boolean_mask_sha256(mask_values)
    if observed_hash != schema["feature_mask_sha256"]:
        raise ValueError("feature schema mask hash mismatch")
    mask = np.asarray(mask_values, dtype=bool)
    if int(np.sum(mask)) != int(schema["output_dimension"]):
        raise ValueError("feature schema retained dimension disagrees with mask")
    expected_indices = np.flatnonzero(mask).astype(int).tolist()
    if expected_indices != list(schema["retained_feature_indices"]):
        raise ValueError("feature schema retained indices disagree with mask")
    return mask


def retained_training_rms_from_schema(schema: dict) -> np.ndarray:
    """Load the fitted training RMS needed to audit scaler relative precision."""
    if "retained_training_RMS" not in schema:
        raise ValueError(
            "feature schema has no fitted retained_training_RMS; fit the "
            "training-only pruning state before router replay"
        )
    rms = np.asarray(schema["retained_training_RMS"], float)
    if rms.ndim != 1 or len(rms) != int(schema["output_dimension"]):
        raise ValueError("retained training RMS dimension disagrees with schema")
    if not np.all(np.isfinite(rms)):
        raise ValueError("retained training RMS contains a non-finite value")
    return rms


def numpy_invariant_features(node_feats: np.ndarray, schema: dict) -> np.ndarray:
    """Apply a recorded invariant schema to a NumPy ``node_feats`` matrix."""
    values = np.asarray(node_feats, float)
    expected_width = int(schema["raw_node_feats_width"])
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(
            f"unexpected node_feats shape {values.shape}; expected width {expected_width}"
        )
    invariants = []
    for item in schema["blocks"]:
        start, stop = item["raw_slice"]
        block = values[:, start:stop].reshape(
            len(values), item["multiplicity"], item["irrep_dimension"]
        )
        if item["operation"] == "signed_l0_value":
            invariants.append(block[:, :, 0])
        elif item["operation"] == "mean_m_squared":
            invariants.append(np.mean(np.square(block), axis=-1))
        elif item["operation"] == "offdiagonal_gram_sketch16":
            _, pairs, phi = validated_gram_sketch16_arrays(schema, item)
            gram = np.einsum("nam,nbm->nab", block, block) / float(
                item["irrep_dimension"]
            )
            offdiagonal = gram[:, pairs[:, 0], pairs[:, 1]]
            invariants.append(offdiagonal @ phi.T)
        else:
            raise ValueError(f"unsupported invariant operation {item['operation']!r}")
    if not invariants:
        raise ValueError("invariant schema selects no node feature columns")
    unpruned = np.concatenate(invariants, axis=1)
    if unpruned.shape[1] != int(schema["unpruned_output_dimension"]):
        raise ValueError("unpruned invariant dimension disagrees with its schema")
    result = unpruned[:, feature_mask_from_schema(schema)]
    if result.shape[1] != int(schema["output_dimension"]):
        raise ValueError("retained invariant dimension disagrees with its schema")
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite frozen MACE invariants")
    return result


def torch_invariant_features(node_feats: torch.Tensor, schema: dict) -> torch.Tensor:
    """Replay the recorded feature map without leaving the Torch/autograd graph."""
    expected_width = int(schema["raw_node_feats_width"])
    if node_feats.ndim != 2 or node_feats.shape[1] != expected_width:
        raise ValueError(
            f"unexpected node_feats shape {tuple(node_feats.shape)}; "
            f"expected width {expected_width}"
        )
    invariants = []
    for item in schema["blocks"]:
        start, stop = item["raw_slice"]
        block = node_feats[:, start:stop].reshape(
            node_feats.shape[0], item["multiplicity"], item["irrep_dimension"]
        )
        if item["operation"] == "signed_l0_value":
            invariants.append(block[:, :, 0])
        elif item["operation"] == "mean_m_squared":
            invariants.append(torch.mean(torch.square(block), dim=-1))
        elif item["operation"] == "offdiagonal_gram_sketch16":
            _, pairs_numpy, phi_numpy = validated_gram_sketch16_arrays(
                schema, item
            )
            gram = torch.einsum("nam,nbm->nab", block, block) / float(
                item["irrep_dimension"]
            )
            left = torch.as_tensor(
                pairs_numpy[:, 0], dtype=torch.long, device=node_feats.device
            )
            right = torch.as_tensor(
                pairs_numpy[:, 1], dtype=torch.long, device=node_feats.device
            )
            phi = node_feats.new_tensor(phi_numpy)
            invariants.append(gram[:, left, right] @ torch.transpose(phi, 0, 1))
        else:
            raise ValueError(f"unsupported invariant operation {item['operation']!r}")
    if not invariants:
        raise ValueError("invariant schema selects no node feature columns")
    unpruned = torch.cat(invariants, dim=1)
    if unpruned.shape[1] != int(schema["unpruned_output_dimension"]):
        raise ValueError("unpruned invariant dimension disagrees with its schema")
    retained_indices = np.flatnonzero(feature_mask_from_schema(schema)).tolist()
    index = torch.as_tensor(retained_indices, dtype=torch.long, device=node_feats.device)
    result = torch.index_select(unpruned, dim=1, index=index)
    if result.shape[1] != int(schema["output_dimension"]):
        raise ValueError("retained invariant dimension disagrees with its schema")
    return result


def replay_router_score_torch(
    node_feats: torch.Tensor, schema: dict, router_state: dict
) -> torch.Tensor:
    """Replay the fixed router map and Ridge while retaining position gradients.

    Serialized states created before R2K-Q contain no ``router_feature_map``;
    those states continue to replay through the historical linear path.
    """
    features = torch_invariant_features(node_feats, schema)
    expected = int(schema["output_dimension"])
    if int(router_state.get("router_input_feature_dimension", expected)) != expected:
        raise ValueError("router input feature dimension disagrees with state")
    feature_map = str(
        router_state.get("router_feature_map", ROUTER_FEATURE_MAP_LINEAR)
    )
    if feature_map == ROUTER_FEATURE_MAP_LINEAR:
        mapped = features
    elif feature_map == ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC:
        input_mean = features.new_tensor(router_state.get("input_scaler_mean"))
        input_scale = features.new_tensor(router_state.get("input_scaler_scale"))
        if input_mean.ndim != 1 or input_scale.ndim != 1:
            raise ValueError("router input scaler vectors must be one-dimensional")
        if input_mean.numel() != expected or input_scale.numel() != expected:
            raise ValueError(
                "router input scaler dimension disagrees with feature schema"
            )
        if not torch.all(torch.isfinite(input_mean)) or not torch.all(
            torch.isfinite(input_scale)
        ):
            raise ValueError("router input scaler contains a non-finite value")
        if torch.any(input_scale <= 0.0):
            raise ValueError("router input scaler contains a negative or zero scale")
        standardized = (features - input_mean) / input_scale
        mapped = torch.cat([standardized, torch.square(standardized)], dim=1)
    else:
        raise ValueError(f"unsupported serialized router feature map {feature_map!r}")
    if "router_feature_map_schema" in router_state:
        map_schema = copy.deepcopy(router_state["router_feature_map_schema"])
        recorded_hash = map_schema.pop("schema_sha256", None)
        if recorded_hash is None or canonical_json_sha256(map_schema) != recorded_hash:
            raise ValueError("router feature-map schema SHA-256 mismatch")
        if recorded_hash != router_state.get("router_feature_map_schema_sha256"):
            raise ValueError("router feature-map state/schema SHA-256 mismatch")
        if map_schema["name"] != feature_map:
            raise ValueError("router feature-map state/schema name mismatch")
        if int(map_schema["input_dimension"]) != expected:
            raise ValueError("router feature-map input dimension disagrees with schema")
        if int(map_schema["output_dimension"]) != mapped.shape[1]:
            raise ValueError("router feature-map output dimension disagrees with schema")
    mean = mapped.new_tensor(router_state["scaler_mean"])
    scale = mapped.new_tensor(router_state["scaler_scale"])
    coefficient = mapped.new_tensor(router_state["ridge_coefficient"])
    mapped_expected = int(mapped.shape[1])
    if (
        int(router_state.get("router_mapped_feature_dimension", mapped_expected))
        != mapped_expected
    ):
        raise ValueError("router mapped feature dimension disagrees with state")
    if mean.numel() != mapped_expected or scale.numel() != mapped_expected:
        raise ValueError("router scaler dimension disagrees with mapped features")
    if coefficient.ndim != 1 or coefficient.numel() != mapped_expected:
        raise ValueError("router Ridge dimension disagrees with mapped features")
    if not torch.all(torch.isfinite(mean)) or not torch.all(torch.isfinite(scale)):
        raise ValueError("router scaler contains a non-finite value")
    if not torch.all(torch.isfinite(coefficient)):
        raise ValueError("router Ridge coefficient contains a non-finite value")
    if torch.any(scale <= 0.0):
        raise ValueError("router scaler contains a negative or zero scale")
    retained_rms = features.new_tensor(retained_training_rms_from_schema(schema))
    if torch.any(retained_rms <= torch.finfo(features.dtype).tiny):
        raise ValueError("retained training RMS is zero or subnormal for replay dtype")
    numerical_input_scale = (
        input_scale
        if feature_map == ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC
        else scale
    )
    scaler_relative_to_rms = numerical_input_scale / retained_rms
    if torch.any(scaler_relative_to_rms <= FEATURE_RELATIVE_STD_THRESHOLD):
        raise ValueError(
            "router scaler varies too little relative to the saved training RMS; "
            "the persisted training-only pruning mask is inconsistent with this state"
        )
    intercept = features.new_tensor(float(router_state["ridge_intercept"]))
    if not torch.isfinite(intercept):
        raise ValueError("router Ridge intercept is non-finite")
    return torch.sum((mapped - mean) / scale * coefficient, dim=1) + intercept


def scalar_invariants(
    node_feats: np.ndarray,
    layout: list,
    expected_width: int,
    feature_mode: str = FEATURE_MODE_SIGNED_L0,
) -> np.ndarray:
    """Extract the selected invariant representation (historical name retained)."""
    schema = invariant_feature_schema(layout, expected_width, feature_mode)
    return numpy_invariant_features(node_feats, schema)


def symmetric_bond_features(
    atom_features: np.ndarray,
    atom_i: np.ndarray,
    atom_j: np.ndarray,
    distance_A: np.ndarray,
) -> np.ndarray:
    """Smooth endpoint-exchange-invariant features for an unordered bond."""
    left = atom_features[atom_i]
    right = atom_features[atom_j]
    mean = 0.5 * (left + right)
    squared_difference = (left - right) ** 2
    return np.concatenate(
        [mean, squared_difference, distance_A[:, None], distance_A[:, None] ** 2],
        axis=1,
    )


def feature_families(
    atom_invariants: np.ndarray,
    geometry_invariants: np.ndarray,
    atom_i: np.ndarray,
    atom_j: np.ndarray,
    distance_A: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "mace_invariants": symmetric_bond_features(
            atom_invariants, atom_i, atom_j, distance_A
        ),
        "geometry_control": symmetric_bond_features(
            geometry_invariants, atom_i, atom_j, distance_A
        ),
        "bond_length_control": np.stack(
            [distance_A, distance_A**2, distance_A**3, distance_A**4], axis=1
        ),
    }


def aggregate_bond_features_to_atoms(
    bond_features: dict[str, np.ndarray],
    atom_i: np.ndarray,
    atom_j: np.ndarray,
    n_atoms: int,
) -> dict[str, np.ndarray]:
    """Mean incident exchange-invariant bond features for an atom-local router."""
    degree = np.bincount(np.concatenate([atom_i, atom_j]), minlength=n_atoms)
    result = {}
    for family, values in bond_features.items():
        accumulated = np.zeros((n_atoms, values.shape[1]), float)
        np.add.at(accumulated, atom_i, values)
        np.add.at(accumulated, atom_j, values)
        result[family] = accumulated / degree[:, None]
    return result


def apply_mace_invariant_mask_to_records(
    records: list[Record], feature_schema: dict
) -> None:
    """Apply one training-only mask consistently to router and bond features."""
    mask = feature_mask_from_schema(feature_schema)
    input_dimension = len(mask)
    retained_dimension = int(np.sum(mask))
    for record in records:
        if record.atom_invariants.shape[1] != input_dimension:
            raise ValueError(
                f"record {record.label} invariant width disagrees with pruning mask"
            )
        if retained_dimension != input_dimension:
            record.atom_invariants = record.atom_invariants[:, mask]
            record.router_features["mace_invariants"] = record.atom_invariants
            record.features["mace_invariants"] = symmetric_bond_features(
                record.atom_invariants,
                record.atom_i,
                record.atom_j,
                record.distance_A,
            )


def atomic_record(structure: Atoms, cutoff: float) -> AtomicData:
    specification = KeySpecification(
        info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
    )
    configuration = config_from_atoms(
        structure,
        key_specification=specification,
        config_type_weights=None,
        head_name="Default",
    )
    return AtomicData.from_config(
        configuration,
        z_table=AtomicNumberTable([6]),
        cutoff=cutoff,
        heads=["Default"],
    )


def frozen_forward(
    model: torch.nn.Module,
    structure: Atoms,
    device: torch.device,
    compute_force: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    record = atomic_record(structure, float(model.r_max))
    loader = torch_geometric.dataloader.DataLoader(
        [record], batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to(device)
    if compute_force:
        with torch.enable_grad():
            output = model(batch.to_dict(), training=False, compute_force=True)
    else:
        with torch.no_grad():
            output = model(batch.to_dict(), training=False, compute_force=False)
    node_feats = output["node_feats"].detach().cpu().numpy()
    forces = (
        output["forces"].detach().cpu().numpy() if compute_force else None
    )
    if len(node_feats) != len(structure):
        raise ValueError("MACE changed the atom count/order in a single-graph forward")
    return node_feats, forces


def minimum_norm_bond_tension(
    force_correction_eV_A: np.ndarray,
    atom_i: np.ndarray,
    atom_j: np.ndarray,
    units: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, dict]:
    """Least-squares central-bond representation with the minimum 2-norm tension."""
    n_atoms = len(force_correction_eV_A)
    matrix = np.zeros((3 * n_atoms, len(atom_i)), float)
    for column, (left, right) in enumerate(zip(atom_i, atom_j, strict=True)):
        matrix[3 * left : 3 * left + 3, column] = units[column]
        matrix[3 * right : 3 * right + 3, column] = -units[column]
    flat = np.asarray(force_correction_eV_A, float).reshape(-1)
    tension, _, rank, singular_values = np.linalg.lstsq(matrix, flat, rcond=rcond)
    projected = matrix @ tension
    remainder = flat - projected
    total2 = float(np.dot(flat, flat))
    radial2 = float(np.dot(projected, projected))
    return tension * 1000.0, {
        "n_bonds": int(len(atom_i)),
        "matrix_rank": int(rank),
        "minimum_singular_value": float(np.min(singular_values)),
        "maximum_singular_value": float(np.max(singular_values)),
        "condition_number": float(np.max(singular_values) / np.min(singular_values)),
        "force_correction_squared_norm_eV2_A2": total2,
        "radial_projection_squared_norm_eV2_A2": radial2,
        "nonradial_remainder_squared_norm_eV2_A2": float(np.dot(remainder, remainder)),
        "radial_force_squared_fraction": radial2 / total2 if total2 > 0.0 else 0.0,
        "net_force_correction_norm_meV_A": float(
            np.linalg.norm(np.sum(force_correction_eV_A, axis=0)) * 1000.0
        ),
        "minimum_norm_tension_squared_meV2_A2": float(np.dot(tension, tension) * 1.0e6),
    }


def build_record(
    descriptor_model: torch.nn.Module,
    expert_model: torch.nn.Module,
    layout: list,
    node_width: int,
    structure: Atoms,
    split: str,
    source_index: int,
    label: str,
    cutoff_A: float,
    rcond: float,
    device: torch.device,
    decompose_target: bool,
    feature_mode: str = FEATURE_MODE_SIGNED_L0,
) -> Record:
    atom_i, atom_j, distances, units, all_distances = first_neighbour_bonds(
        structure, cutoff_A
    )
    same_model = descriptor_model is expert_model
    node_feats, predicted_forces = frozen_forward(
        descriptor_model, structure, device, compute_force=same_model
    )
    if not same_model:
        _, predicted_forces = frozen_forward(
            expert_model, structure, device, compute_force=True
        )
    if predicted_forces is None:
        raise RuntimeError("delta-expert force forward returned no forces")
    atom_invariants = scalar_invariants(
        node_feats, layout, node_width, feature_mode=feature_mode
    )
    geometry = atom_geometry_invariants(structure, all_distances)
    features = feature_families(
        atom_invariants, geometry, atom_i, atom_j, distances
    )
    # The production router must act on one atom at a time inside the energy
    # graph.  Keep the primary probe identical to that deployable interface:
    # it reads only invariant node features, without a global pooling or a
    # supercell-size feature.  Geometry and bond-length-only probes are fixed
    # controls, not production inputs.
    nearest_three = geometry[:, :3]
    router_features = {
        "mace_invariants": atom_invariants,
        "geometry_control": geometry,
        "bond_length_control": np.concatenate(
            [
                nearest_three,
                nearest_three**2,
                np.mean(nearest_three, axis=1, keepdims=True),
                np.min(nearest_three, axis=1, keepdims=True),
                np.max(nearest_three, axis=1, keepdims=True),
            ],
            axis=1,
        ),
    }
    target = np.asarray(structure.arrays["REF_forces"], float)
    remaining = target - predicted_forces
    target_meV = target * 1000.0
    remaining_meV = remaining * 1000.0
    atom_utility = np.sum(target_meV**2, axis=1) - np.sum(remaining_meV**2, axis=1)
    bond_utility = 0.5 * (atom_utility[atom_i] + atom_utility[atom_j])
    target_tension = None
    remaining_tension = None
    target_radial = None
    remaining_radial = None
    if decompose_target:
        target_tension, target_radial = minimum_norm_bond_tension(
            target, atom_i, atom_j, units, rcond
        )
        remaining_tension, remaining_radial = minimum_norm_bond_tension(
            remaining, atom_i, atom_j, units, rcond
        )
    force_utility = {
        "n_atoms": len(structure),
        "n_force_components": int(target_meV.size),
        "zero_delta_squared_error_sum_meV2_A2": float(np.sum(target_meV**2)),
        "active_expert_squared_error_sum_meV2_A2": float(np.sum(remaining_meV**2)),
        "positive_utility_atom_count": int(np.sum(atom_utility > 0.0)),
        "zero_delta_component_RMSE_meV_A": float(np.sqrt(np.mean(target_meV**2))),
        "active_expert_component_RMSE_meV_A": float(
            np.sqrt(np.mean(remaining_meV**2))
        ),
        "positive_utility_atom_fraction": float(np.mean(atom_utility > 0.0)),
        "positive_utility_bond_fraction": float(np.mean(bond_utility > 0.0)),
        "positive_atom_utility_sum_meV2_A2": float(
            np.sum(np.maximum(atom_utility, 0.0))
        ),
        "negative_atom_utility_magnitude_sum_meV2_A2": float(
            np.sum(np.maximum(-atom_utility, 0.0))
        ),
    }
    temperature = structure.info.get("lattice_temperature_K")
    aprime_mode = None
    if "R2K_Aprime_mode_real" in structure.arrays:
        aprime_mode = np.asarray(
            structure.arrays["R2K_Aprime_mode_real"], float
        ) + 1.0j * np.asarray(structure.arrays["R2K_Aprime_mode_imag"], float)
    return Record(
        split=split,
        source_index=source_index,
        label=label,
        temperature_K=float(temperature) if temperature is not None else None,
        atom_invariants=atom_invariants,
        geometry_invariants=geometry,
        atom_i=atom_i,
        atom_j=atom_j,
        distance_A=distances,
        features=features,
        router_features=router_features,
        target_force_meV_A=target_meV,
        remaining_force_meV_A=remaining_meV,
        atom_utility_meV2_A2=atom_utility,
        bond_utility_meV2_A2=bond_utility,
        aprime_mode=aprime_mode,
        target_tension_meV_A=target_tension,
        remaining_tension_meV_A=remaining_tension,
        target_radial=target_radial,
        remaining_radial=remaining_radial,
        force_utility=force_utility,
    )


def concatenate(records: list[Record], family: str) -> np.ndarray:
    return np.concatenate([record.features[family] for record in records], axis=0)


def concatenate_router(records: list[Record], family: str) -> np.ndarray:
    return np.concatenate([record.router_features[family] for record in records], axis=0)


def configuration_balanced_weights(
    records: list[Record], kind: str, sign_threshold: float
) -> np.ndarray:
    """Give each complete configuration equal total weight."""
    pieces = []
    for record in records:
        if kind == "uniform":
            raw = np.ones(len(record.distance_A), float)
        elif kind == "target_tension_squared":
            if record.target_tension_meV_A is None:
                raise ValueError("target tension requested for an undecomposed record")
            raw = record.target_tension_meV_A**2
            raw = np.maximum(raw, np.finfo(float).eps)
        elif kind == "tension_regression":
            if record.remaining_tension_meV_A is None:
                raise ValueError("tension weights requested for an untargeted record")
            raw = np.maximum(
                (np.abs(record.remaining_tension_meV_A) / sign_threshold) ** 2, 1.0e-2
            )
        else:
            raise ValueError(f"unknown weighting scheme {kind}")
        pieces.append(raw / np.sum(raw) / len(records))
    result = np.concatenate(pieces)
    return result * len(result) / np.sum(result)


def fit_ridge(
    values: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> Probe:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1.0e-9)
    model.fit(scaled, target, sample_weight=weights)
    return Probe(scaler=scaler, model=model)


def fit_router(
    records: list[Record],
    family: str,
    alpha: float,
    utility_deadband: float,
    feature_map: str = ROUTER_FEATURE_MAP_LINEAR,
) -> Probe:
    """Fit a signed-utility probe; roles never enter the target definition."""
    if feature_map not in ROUTER_FEATURE_MAPS:
        raise ValueError(
            f"unknown router feature map {feature_map!r}; "
            f"expected one of {ROUTER_FEATURE_MAPS}"
        )
    values = []
    targets = []
    weights = []
    for record in records:
        utility = record.atom_utility_meV2_A2
        keep = np.abs(utility) > utility_deadband
        if not np.any(keep):
            continue
        raw = np.abs(utility[keep])
        # Equal configuration mass prevents a large supercell from dominating.
        raw /= np.sum(raw)
        values.append(record.router_features[family][keep])
        targets.append((utility[keep] > 0.0).astype(float))
        weights.append(raw)
    if not values:
        raise RuntimeError("no nonzero utility samples for router fit")
    values_array = np.concatenate(values)
    target_array = np.concatenate(targets)
    weight_array = np.concatenate(weights)
    # Balance utility signs without referring to harmonic/support/thermal roles.
    positive = target_array > 0.5
    negative = ~positive
    if not np.any(positive) or not np.any(negative):
        raise RuntimeError("router training needs both utility signs")
    weight_array[positive] *= 0.5 / np.sum(weight_array[positive])
    weight_array[negative] *= 0.5 / np.sum(weight_array[negative])
    weight_array *= len(weight_array) / np.sum(weight_array)
    input_dimension = int(values_array.shape[1])
    if feature_map == ROUTER_FEATURE_MAP_LINEAR:
        # Keep the historical StandardScaler -> Ridge fit path byte-for-byte in
        # numerical operation order.  Only replay metadata is added afterward.
        probe = fit_ridge(values_array, target_array, weight_array, alpha)
        probe.router_feature_map = ROUTER_FEATURE_MAP_LINEAR
        probe.input_dimension = input_dimension
        return probe

    # R2K-Q is intentionally a single fixed-capacity test: standardize the
    # original descriptor, append only its diagonal squares, standardize that
    # mapped matrix, and reuse the fixed Ridge alpha.  There is no alpha or map
    # scan here.
    input_scaler = StandardScaler()
    input_scaler.fit(values_array)
    mapped_values = numpy_router_feature_map(
        values_array, feature_map, input_scaler
    )
    probe = fit_ridge(mapped_values, target_array, weight_array, alpha)
    probe.router_feature_map = feature_map
    probe.input_scaler = input_scaler
    probe.input_dimension = input_dimension
    return probe


def fit_tension(
    support: list[Record],
    family: str,
    alpha: float,
    sign_threshold: float,
    bond_gates: list[np.ndarray] | None = None,
) -> Probe:
    values = concatenate(support, family)
    target = np.concatenate([record.remaining_tension_meV_A for record in support])
    if bond_gates is None:
        weights = configuration_balanced_weights(
            support, "tension_regression", sign_threshold
        )
    else:
        pieces = []
        for record, gate in zip(support, bond_gates, strict=True):
            raw = np.maximum(
                (np.abs(record.remaining_tension_meV_A) / sign_threshold) ** 2,
                1.0e-2,
            ) * np.asarray(gate, float) ** 2
            if float(np.sum(raw)) <= 0.0:
                raise RuntimeError("router leaves no active repair tension in a fold")
            pieces.append(raw / np.sum(raw) / len(support))
        weights = np.concatenate(pieces)
        weights *= len(weights) / np.sum(weights)
    return fit_ridge(values, target, weights, alpha)


def operating_threshold(scores: np.ndarray, maximum_activation: float) -> float:
    """Threshold with ``score > threshold`` activating no more than the limit."""
    scores = np.asarray(scores, float)
    if scores.size == 0:
        raise ValueError("cannot calibrate a router on no scores")
    return float(np.quantile(scores, 1.0 - maximum_activation, method="higher"))


def activation_fraction(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(scores) > threshold))


def c2_smootherstep(
    scores: np.ndarray, threshold: float, transition_end: float
) -> np.ndarray:
    """C2 gate proposed for R2L; R2K only evaluates it as a proxy."""
    if transition_end <= threshold:
        raise ValueError("C2 transition end must exceed its threshold")
    values = np.clip(
        (np.asarray(scores, float) - threshold) / (transition_end - threshold),
        0.0,
        1.0,
    )
    return values**3 * (values * (values * 6.0 - 15.0) + 10.0)


def tension_squared_coverage(
    tension: np.ndarray, active: np.ndarray
) -> float:
    squared = np.asarray(tension, float) ** 2
    denominator = float(np.sum(squared))
    return float(np.sum(squared[np.asarray(active, bool)]) / denominator) if denominator else 0.0


def tension_squared_soft_coverage(
    tension: np.ndarray, gate: np.ndarray
) -> float:
    """Squared-amplitude coverage of a tension field under a soft gate."""
    squared = np.asarray(tension, float) ** 2
    weights = np.asarray(gate, float) ** 2
    denominator = float(np.sum(squared))
    return float(np.sum(squared * weights) / denominator) if denominator else 0.0


def weighted_r2(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    target = np.asarray(target, float)
    prediction = np.asarray(prediction, float)
    weights = np.asarray(weights, float)
    mean = float(np.average(target, weights=weights))
    denominator = float(np.sum(weights * (target - mean) ** 2))
    numerator = float(np.sum(weights * (target - prediction) ** 2))
    return 1.0 - numerator / denominator if denominator > 0.0 else float("nan")


def sign_accuracy(
    target: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
    active: np.ndarray | None = None,
) -> tuple[float, int]:
    selected = np.abs(target) >= threshold
    if active is not None:
        selected &= np.asarray(active, bool)
    count = int(np.sum(selected))
    if count == 0:
        return float("nan"), 0
    correct = np.sign(target[selected]) == np.sign(prediction[selected])
    return float(np.mean(correct)), count


def thermal_groups(records: list[Record]) -> dict[str, list[Record]]:
    result: dict[str, list[Record]] = {"all": records}
    temperatures = sorted(
        {record.temperature_K for record in records if record.temperature_K is not None}
    )
    for temperature in temperatures:
        result[f"T{temperature:g}K"] = [
            record for record in records if record.temperature_K == temperature
        ]
    return result


def router_metrics(
    records: list[Record],
    score_arrays: list[np.ndarray],
    threshold: float,
    transition_end: float,
) -> dict:
    atom_utility = np.concatenate([record.atom_utility_meV2_A2 for record in records])
    atom_scores = np.concatenate(score_arrays)
    atom_active = atom_scores > threshold
    atom_gate = c2_smootherstep(atom_scores, threshold, transition_end)
    positive = np.maximum(atom_utility, 0.0)
    harmful = np.maximum(-atom_utility, 0.0)
    positive_denominator = float(np.sum(positive))
    harmful_denominator = float(np.sum(harmful))
    target_tensions = []
    remaining_tensions = []
    bond_active = []
    bond_gates = []
    proxy_errors = []
    soft_proxy_errors = []
    aprime_hard_errors = []
    aprime_soft_errors = []
    per_configuration = {}
    for record, scores in zip(records, score_arrays, strict=True):
        active_atoms = scores > threshold
        gate_atoms = c2_smootherstep(scores, threshold, transition_end)
        proxy_error = np.where(
            active_atoms[:, None],
            record.remaining_force_meV_A,
            record.target_force_meV_A,
        )
        proxy_errors.append(proxy_error)
        soft_proxy_error = (
            (1.0 - gate_atoms[:, None]) * record.target_force_meV_A
            + gate_atoms[:, None] * record.remaining_force_meV_A
        )
        soft_proxy_errors.append(soft_proxy_error)
        bond_scores = 0.5 * (scores[record.atom_i] + scores[record.atom_j])
        active_bonds = bond_scores > threshold
        gate_bonds = 0.5 * (gate_atoms[record.atom_i] + gate_atoms[record.atom_j])
        local_positive = np.maximum(record.atom_utility_meV2_A2, 0.0)
        local_harmful = np.maximum(-record.atom_utility_meV2_A2, 0.0)
        item = {
            "n_atoms": len(scores),
            "activation_fraction": float(np.mean(active_atoms)),
            "C2_gate_mean": float(np.mean(gate_atoms)),
            "C2_gate_zero_fraction": float(np.mean(gate_atoms == 0.0)),
            "C2_gate_transition_fraction": float(
                np.mean((gate_atoms > 0.0) & (gate_atoms < 1.0))
            ),
            "C2_gate_one_fraction": float(np.mean(gate_atoms == 1.0)),
            "positive_utility_retention": (
                float(np.sum(local_positive[active_atoms]) / np.sum(local_positive))
                if np.sum(local_positive) > 0.0
                else None
            ),
            "C2_positive_utility_retention": (
                float(np.sum(local_positive * gate_atoms) / np.sum(local_positive))
                if np.sum(local_positive) > 0.0
                else None
            ),
            "harmful_utility_exposure": (
                float(np.sum(local_harmful[active_atoms]) / np.sum(local_harmful))
                if np.sum(local_harmful) > 0.0
                else None
            ),
            "C2_harmful_utility_exposure": (
                float(np.sum(local_harmful * gate_atoms) / np.sum(local_harmful))
                if np.sum(local_harmful) > 0.0
                else None
            ),
            "hard_route_force_proxy_RMSE_meV_A": float(
                np.sqrt(np.mean(proxy_error**2))
            ),
            "hard_route_force_proxy_max_abs_meV_A": float(
                np.max(np.abs(proxy_error))
            ),
            "C2_route_force_proxy_RMSE_meV_A": float(
                np.sqrt(np.mean(soft_proxy_error**2))
            ),
            "C2_route_force_proxy_max_abs_meV_A": float(
                np.max(np.abs(soft_proxy_error))
            ),
        }
        if record.aprime_mode is not None:
            mode_flat = record.aprime_mode.reshape(-1)
            hard_projection = np.vdot(mode_flat, proxy_error.reshape(-1))
            soft_projection = np.vdot(mode_flat, soft_proxy_error.reshape(-1))
            aprime_hard_errors.append(hard_projection)
            aprime_soft_errors.append(soft_projection)
            item["Aprime_hard_route_force_error_abs_meV_A"] = float(
                abs(hard_projection)
            )
            item["Aprime_C2_route_force_error_abs_meV_A"] = float(
                abs(soft_projection)
            )
        if record.target_tension_meV_A is not None:
            item["target_delta_tension_squared_coverage"] = tension_squared_coverage(
                record.target_tension_meV_A, active_bonds
            )
            item["C2_target_delta_tension_squared_coverage"] = (
                tension_squared_soft_coverage(record.target_tension_meV_A, gate_bonds)
            )
            target_tensions.append(record.target_tension_meV_A)
            bond_active.append(active_bonds)
            bond_gates.append(gate_bonds)
        if record.remaining_tension_meV_A is not None:
            item["remaining_repair_tension_squared_coverage"] = tension_squared_coverage(
                record.remaining_tension_meV_A, active_bonds
            )
            item["C2_remaining_repair_tension_squared_coverage"] = (
                tension_squared_soft_coverage(
                    record.remaining_tension_meV_A, gate_bonds
                )
            )
            remaining_tensions.append(record.remaining_tension_meV_A)
        per_configuration[record.label] = item
    result = {
        "n_configurations": len(records),
        "n_atoms": len(atom_scores),
        "activation_fraction": float(np.mean(atom_active)),
        "C2_gate_mean": float(np.mean(atom_gate)),
        "C2_gate_zero_fraction": float(np.mean(atom_gate == 0.0)),
        "C2_gate_transition_fraction": float(
            np.mean((atom_gate > 0.0) & (atom_gate < 1.0))
        ),
        "C2_gate_one_fraction": float(np.mean(atom_gate == 1.0)),
        "positive_utility_retention": (
            float(np.sum(positive[atom_active]) / positive_denominator)
            if positive_denominator > 0.0
            else None
        ),
        "C2_positive_utility_retention": (
            float(np.sum(positive * atom_gate) / positive_denominator)
            if positive_denominator > 0.0
            else None
        ),
        "harmful_utility_exposure": (
            float(np.sum(harmful[atom_active]) / harmful_denominator)
            if harmful_denominator > 0.0
            else None
        ),
        "C2_harmful_utility_exposure": (
            float(np.sum(harmful * atom_gate) / harmful_denominator)
            if harmful_denominator > 0.0
            else None
        ),
        "hard_route_force_proxy": {
            "definition": (
                "per-atom hard selection between zero delta and frozen expert; "
                "diagnostic only, omits the conservative energy-gate gradient"
            ),
            "RMSE_meV_A": float(
                np.sqrt(np.mean(np.concatenate(proxy_errors, axis=0) ** 2))
            ),
            "max_abs_meV_A": float(
                np.max(np.abs(np.concatenate(proxy_errors, axis=0)))
            ),
        },
        "C2_route_force_proxy": {
            "definition": (
                "force-space interpolation over the proposed C2 score interval; "
                "diagnostic only and omits the conservative energy-gate gradient"
            ),
            "RMSE_meV_A": float(
                np.sqrt(np.mean(np.concatenate(soft_proxy_errors, axis=0) ** 2))
            ),
            "max_abs_meV_A": float(
                np.max(np.abs(np.concatenate(soft_proxy_errors, axis=0)))
            ),
        },
        "per_configuration": per_configuration,
    }
    if aprime_hard_errors:
        hard_values = np.asarray(aprime_hard_errors, complex)
        soft_values = np.asarray(aprime_soft_errors, complex)
        result["Aprime_route_force_screen"] = {
            "hard_RMS_meV_A": float(np.sqrt(np.mean(np.abs(hard_values) ** 2))),
            "hard_max_abs_meV_A": float(np.max(np.abs(hard_values))),
            "C2_RMS_meV_A": float(np.sqrt(np.mean(np.abs(soft_values) ** 2))),
            "C2_max_abs_meV_A": float(np.max(np.abs(soft_values))),
            "definition": (
                "K-point A-prime projection of non-causal force-space routing proxy; "
                "the conservative autograd projection is deferred to R2L"
            ),
        }
    if target_tensions:
        result["target_delta_tension_squared_coverage"] = tension_squared_coverage(
            np.concatenate(target_tensions), np.concatenate(bond_active)
        )
        result["C2_target_delta_tension_squared_coverage"] = (
            tension_squared_soft_coverage(
                np.concatenate(target_tensions), np.concatenate(bond_gates)
            )
        )
    if remaining_tensions:
        result["remaining_repair_tension_squared_coverage"] = tension_squared_coverage(
            np.concatenate(remaining_tensions), np.concatenate(bond_active)
        )
        result["C2_remaining_repair_tension_squared_coverage"] = (
            tension_squared_soft_coverage(
                np.concatenate(remaining_tensions), np.concatenate(bond_gates)
            )
        )
    return result


def audit_family(
    family: str,
    support: list[Record],
    router_train: list[Record],
    harmonic_validation: list[Record],
    fixed_smearing_external: list[Record],
    legacy_mixed_smearing: list[Record],
    router_alpha: float,
    tension_alpha: float,
    sign_threshold: float,
    utility_deadband: float,
    transition_width: float,
    router_feature_map: str = ROUTER_FEATURE_MAP_LINEAR,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Run nine complete-configuration holdouts for one feature family."""
    fold_rows = []
    bond_rows = []
    held_atom_scores = []
    held_bond_active = []
    held_bond_gates = []
    held_router_thresholds = []
    held_tension_predictions = []
    held_targets = []
    per_config_coverage = {}
    per_config_utility_retention = {}
    per_config_remaining_coverage = {}
    per_config_C2_utility_retention = {}
    per_config_C2_remaining_coverage = {}
    per_config_r2 = {}
    per_config_sign = {}

    harmonic_calibration = harmonic_validation[::2]
    harmonic_external = harmonic_validation[1::2]
    if not harmonic_calibration or not harmonic_external:
        raise ValueError("harmonic validation needs disjoint calibration/audit halves")
    validation_x = concatenate_router(harmonic_calibration, family)
    fold_router_states = {}
    for held_index, held in enumerate(support):
        training_support = [
            record for index, record in enumerate(support) if index != held_index
        ]
        router = fit_router(
            router_train + training_support,
            family,
            router_alpha,
            utility_deadband,
            router_feature_map,
        )
        validation_scores = router.predict(validation_x)
        threshold = operating_threshold(validation_scores, HARMONIC_ACTIVATION_MAX)
        held_scores = router.predict(held.router_features[family])
        held_metrics = router_metrics(
            [held], [held_scores], threshold, threshold + transition_width
        )
        bond_scores = 0.5 * (held_scores[held.atom_i] + held_scores[held.atom_j])
        active = bond_scores > threshold
        held_gate_atoms = c2_smootherstep(
            held_scores, threshold, threshold + transition_width
        )
        held_gate_bonds = 0.5 * (
            held_gate_atoms[held.atom_i] + held_gate_atoms[held.atom_j]
        )
        coverage = held_metrics["target_delta_tension_squared_coverage"]

        training_bond_gates = []
        for training_record in training_support:
            training_scores = router.predict(
                training_record.router_features[family]
            )
            training_atom_gate = c2_smootherstep(
                training_scores, threshold, threshold + transition_width
            )
            training_bond_gates.append(
                0.5
                * (
                    training_atom_gate[training_record.atom_i]
                    + training_atom_gate[training_record.atom_j]
                )
            )
        regressor = fit_tension(
            training_support,
            family,
            tension_alpha,
            sign_threshold,
            training_bond_gates,
        )
        prediction = regressor.predict(held.features[family])
        regression_weights = configuration_balanced_weights(
            [held], "tension_regression", sign_threshold
        )
        regression_weights *= held_gate_bonds**2
        if float(np.sum(regression_weights)) <= 0.0:
            r2 = float("nan")
        else:
            regression_weights *= len(regression_weights) / np.sum(regression_weights)
            r2 = weighted_r2(
                held.remaining_tension_meV_A, prediction, regression_weights
            )
        accuracy, sign_count = sign_accuracy(
            held.remaining_tension_meV_A,
            prediction,
            sign_threshold,
            held_gate_bonds >= 0.5,
        )
        per_config_coverage[held.label] = coverage
        per_config_utility_retention[held.label] = held_metrics[
            "positive_utility_retention"
        ]
        per_config_remaining_coverage[held.label] = held_metrics[
            "remaining_repair_tension_squared_coverage"
        ]
        per_config_C2_utility_retention[held.label] = held_metrics[
            "C2_positive_utility_retention"
        ]
        per_config_C2_remaining_coverage[held.label] = held_metrics[
            "C2_remaining_repair_tension_squared_coverage"
        ]
        per_config_r2[held.label] = r2
        per_config_sign[held.label] = {
            "accuracy": accuracy,
            "n_bonds": sign_count,
        }

        fold_rows.append(
            {
                "feature_family": family,
                "held_support_configuration": held.label,
                "n_training_support_configurations": len(training_support),
                "n_held_bonds": len(held.distance_A),
                "router_threshold": threshold,
                "router_transition_end": threshold + transition_width,
                "harmonic_calibration_activation_fraction": activation_fraction(
                    validation_scores, threshold
                ),
                "held_support_tension_squared_coverage": coverage,
                "held_support_positive_utility_retention": held_metrics[
                    "positive_utility_retention"
                ],
                "held_support_remaining_repair_tension_squared_coverage": held_metrics[
                    "remaining_repair_tension_squared_coverage"
                ],
                "held_support_active_bond_fraction": activation_fraction(
                    bond_scores, threshold
                ),
                "tension_weighted_R2": r2,
                "tension_sign_accuracy_abs_ge_threshold": accuracy,
                "n_tension_sign_bonds": sign_count,
            }
        )
        for bond in range(len(held.distance_A)):
            bond_rows.append(
                {
                    "feature_family": family,
                    "held_support_configuration": held.label,
                    "bond_index": bond,
                    "atom_i": int(held.atom_i[bond]),
                    "atom_j": int(held.atom_j[bond]),
                    "distance_A": float(held.distance_A[bond]),
                    "target_delta_minimum_norm_tension_meV_A": float(
                        held.target_tension_meV_A[bond]
                    ),
                    "remaining_repair_minimum_norm_tension_meV_A": float(
                        held.remaining_tension_meV_A[bond]
                    ),
                    "LOCO_router_score": float(bond_scores[bond]),
                    "LOCO_router_threshold": threshold,
                    "LOCO_router_active": int(active[bond]),
                    "LOCO_tension_prediction_meV_A": float(prediction[bond]),
                }
            )

        held_atom_scores.append(held_scores)
        held_bond_active.append(active)
        held_bond_gates.append(held_gate_bonds)
        held_router_thresholds.append(np.full(len(held_scores), threshold))
        held_tension_predictions.append(prediction)
        held_targets.append(held.remaining_tension_meV_A)
        fold_router_states[held.label] = {
            "t0": threshold,
            "t1": threshold + transition_width,
            "state": router.state(),
        }

    all_target = np.concatenate(held_targets)
    all_prediction = np.concatenate(held_tension_predictions)
    evaluation_weights = configuration_balanced_weights(
        support, "tension_regression", sign_threshold
    ) * np.concatenate(held_bond_gates) ** 2
    evaluation_weights *= len(evaluation_weights) / np.sum(evaluation_weights)
    overall_r2 = weighted_r2(all_target, all_prediction, evaluation_weights)
    overall_sign, overall_sign_count = sign_accuracy(
        all_target,
        all_prediction,
        sign_threshold,
        np.concatenate(held_bond_gates) >= 0.5,
    )
    # Per-fold thresholds differ, so recompute aggregate retained quantities.
    target_all = np.concatenate([record.target_tension_meV_A for record in support])
    remaining_all = np.concatenate(
        [record.remaining_tension_meV_A for record in support]
    )
    active_all = np.concatenate(held_bond_active)
    overall_coverage = tension_squared_coverage(target_all, active_all)
    overall_remaining_coverage = tension_squared_coverage(remaining_all, active_all)
    overall_C2_remaining_coverage = tension_squared_soft_coverage(
        remaining_all, np.concatenate(held_bond_gates)
    )
    utility_all = np.concatenate([record.atom_utility_meV2_A2 for record in support])
    utility_active = np.concatenate(
        [
            scores > thresholds
            for scores, thresholds in zip(
                held_atom_scores, held_router_thresholds, strict=True
            )
        ]
    )
    positive_utility = np.maximum(utility_all, 0.0)
    overall_utility_retention = float(
        np.sum(positive_utility[utility_active]) / np.sum(positive_utility)
    )
    held_gate_all = np.concatenate(
        [
            c2_smootherstep(scores, thresholds[0], thresholds[0] + transition_width)
            for scores, thresholds in zip(
                held_atom_scores, held_router_thresholds, strict=True
            )
        ]
    )
    overall_C2_utility_retention = float(
        np.sum(positive_utility * held_gate_all) / np.sum(positive_utility)
    )

    full_router = fit_router(
        router_train + support,
        family,
        router_alpha,
        utility_deadband,
        router_feature_map,
    )
    full_validation_scores = full_router.predict(validation_x)
    full_threshold = operating_threshold(
        full_validation_scores, HARMONIC_ACTIVATION_MAX
    )
    activation_rows = []
    evaluation_sets = {
        "harmonic_calibration": harmonic_calibration,
        "harmonic_external": harmonic_external,
    }
    for prefix, records in (
        ("fixed_smearing_external", fixed_smearing_external),
        ("legacy_mixed_smearing", legacy_mixed_smearing),
    ):
        for group, selected in thermal_groups(records).items():
            evaluation_sets[f"{prefix}_{group}"] = selected
    full_metrics = {}
    for name, records in evaluation_sets.items():
        scores = full_router.predict(concatenate_router(records, family))
        score_arrays = np.split(
            scores,
            np.cumsum([len(record.atom_invariants) for record in records])[:-1],
        )
        metrics = router_metrics(
            records,
            score_arrays,
            full_threshold,
            full_threshold + transition_width,
        )
        full_metrics[name] = metrics
        activation_rows.append(
            {
                "feature_family": family,
                "router_fit": (
                    "utility labels from E50 fixed-smearing seeds0+1, "
                    "harmonic train, and support9"
                ),
                "evaluation_set": name,
                "n_configurations": len(records),
                "n_atoms": metrics["n_atoms"],
                "router_threshold": full_threshold,
                "activation_fraction": metrics["activation_fraction"],
                "positive_utility_retention": metrics["positive_utility_retention"],
                "target_delta_tension_squared_coverage": metrics.get(
                    "target_delta_tension_squared_coverage"
                ),
            }
        )

    map_schema = router_feature_map_schema(
        router_feature_map, support[0].router_features[family].shape[1]
    )
    result = {
        "router_feature_dimension": int(support[0].router_features[family].shape[1]),
        "router_mapped_feature_dimension": int(map_schema["output_dimension"]),
        "router_feature_map": router_feature_map,
        "router_feature_map_schema": map_schema,
        "router_feature_map_schema_sha256": map_schema["schema_sha256"],
        "tension_feature_dimension": int(support[0].features[family].shape[1]),
        "LOCO": {
            "split_unit": "complete support configuration",
            "n_folds": len(support),
            "support_minimum_norm_bond_tension_squared_coverage": overall_coverage,
            "support_positive_utility_retention": overall_utility_retention,
            "support_remaining_repair_tension_squared_coverage": overall_remaining_coverage,
            "support_C2_positive_utility_retention": overall_C2_utility_retention,
            "support_C2_remaining_repair_tension_squared_coverage": (
                overall_C2_remaining_coverage
            ),
            "support_per_configuration_coverage": per_config_coverage,
            "support_per_configuration_positive_utility_retention": (
                per_config_utility_retention
            ),
            "support_per_configuration_remaining_repair_coverage": (
                per_config_remaining_coverage
            ),
            "support_per_configuration_C2_positive_utility_retention": (
                per_config_C2_utility_retention
            ),
            "support_per_configuration_C2_remaining_repair_coverage": (
                per_config_C2_remaining_coverage
            ),
            "support_minimum_per_configuration_coverage": float(
                min(per_config_coverage.values())
            ),
            "support_minimum_per_configuration_positive_utility_retention": float(
                min(per_config_utility_retention.values())
            ),
            "support_minimum_per_configuration_remaining_repair_coverage": float(
                min(per_config_remaining_coverage.values())
            ),
            "support_minimum_per_configuration_C2_positive_utility_retention": float(
                min(per_config_C2_utility_retention.values())
            ),
            "support_minimum_per_configuration_C2_remaining_repair_coverage": float(
                min(per_config_C2_remaining_coverage.values())
            ),
            "harmonic_calibration_activation_fraction_max_across_folds": float(
                max(row["harmonic_calibration_activation_fraction"] for row in fold_rows)
            ),
            "tension_weighted_R2": overall_r2,
            "tension_sign_accuracy_abs_ge_threshold": overall_sign,
            "n_tension_sign_bonds": overall_sign_count,
            "tension_per_configuration_weighted_R2": per_config_r2,
            "tension_per_configuration_sign_accuracy": per_config_sign,
            "tension_minimum_per_configuration_weighted_R2": float(
                min(per_config_r2.values())
            ),
            "tension_minimum_per_configuration_sign_accuracy": float(
                min(item["accuracy"] for item in per_config_sign.values())
            ),
            "router_states_by_held_support_configuration": fold_router_states,
        },
        "full_fit_router": {
            "threshold_calibration": (
                "score > empirical harmonic-calibration 95th percentile (higher method); "
                "the disjoint harmonic-external half is used for screening"
            ),
            "t0": full_threshold,
            "t1": full_threshold + transition_width,
            "state": full_router.state(),
            "external_evaluation": full_metrics,
        },
    }
    return result, fold_rows, bond_rows, activation_rows


def plot_results(
    output_dir: Path,
    support: list[Record],
    probes: dict,
    bond_rows: list[dict],
    activation_rows: list[dict],
) -> None:
    colors = {
        "mace_invariants": "#1f77b4",
        "geometry_control": "#ff7f0e",
        "bond_length_control": "#7f7f7f",
    }
    display = {
        "mace_invariants": "frozen MACE invariants",
        "geometry_control": "geometry control",
        "bond_length_control": "bond-length control",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)

    labels = [record.label for record in support]
    x = np.arange(len(labels))
    width = 0.25
    for offset, family in enumerate(FEATURE_FAMILIES):
        coverage = probes[family]["LOCO"]["support_per_configuration_coverage"]
        axes[0, 0].bar(
            x + (offset - 1) * width,
            [coverage[label] for label in labels],
            width,
            color=colors[family],
            label=display[family],
        )
    axes[0, 0].axhline(
        SUPPORT_PER_CONFIG_COVERAGE_MIN,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0.0, 1.04)
    axes[0, 0].set_xlabel("held support configuration (sscha index)")
    axes[0, 0].set_ylabel("LOCO tension-squared router coverage")

    for family in FEATURE_FAMILIES:
        rows = [row for row in bond_rows if row["feature_family"] == family]
        target = np.array(
            [row["remaining_repair_minimum_norm_tension_meV_A"] for row in rows]
        )
        prediction = np.array([row["LOCO_tension_prediction_meV_A"] for row in rows])
        axes[0, 1].scatter(
            target,
            prediction,
            s=10,
            alpha=0.24,
            color=colors[family],
            rasterized=True,
            label=display[family],
        )
    bound = max(
        abs(axes[0, 1].get_xlim()[0]),
        abs(axes[0, 1].get_xlim()[1]),
        abs(axes[0, 1].get_ylim()[0]),
        abs(axes[0, 1].get_ylim()[1]),
    )
    axes[0, 1].plot([-bound, bound], [-bound, bound], color="black", linewidth=1.0)
    axes[0, 1].axvline(sign_threshold := TENSION_SIGN_THRESHOLD_MEV_A, color="black", linestyle=":", linewidth=0.8)
    axes[0, 1].axvline(-sign_threshold, color="black", linestyle=":", linewidth=0.8)
    axes[0, 1].set_xlim(-bound, bound)
    axes[0, 1].set_ylim(-bound, bound)
    axes[0, 1].set_xlabel("minimum-norm tension target (meV/Å)")
    axes[0, 1].set_ylabel("LOCO ridge prediction (meV/Å)")

    metric_x = np.arange(len(FEATURE_FAMILIES))
    r2 = [probes[name]["LOCO"]["tension_weighted_R2"] for name in FEATURE_FAMILIES]
    sign = [
        probes[name]["LOCO"]["tension_sign_accuracy_abs_ge_threshold"]
        for name in FEATURE_FAMILIES
    ]
    axes[1, 0].bar(metric_x - 0.18, r2, 0.36, label="weighted $R^2$", color="#4c78a8")
    axes[1, 0].bar(metric_x + 0.18, sign, 0.36, label="sign accuracy", color="#59a14f")
    axes[1, 0].axhline(TENSION_WEIGHTED_R2_MIN, color="#4c78a8", linestyle="--", linewidth=1.0)
    axes[1, 0].axhline(TENSION_SIGN_ACCURACY_MIN, color="#59a14f", linestyle=":", linewidth=1.0)
    axes[1, 0].set_xticks(metric_x, [display[name] for name in FEATURE_FAMILIES], rotation=12)
    axes[1, 0].set_ylim(min(-0.2, min(r2) - 0.05), 1.03)
    axes[1, 0].set_ylabel("LOCO tension metric")

    sets = [
        "harmonic_external",
        "fixed_smearing_external_all",
        "legacy_mixed_smearing_T300K",
        "legacy_mixed_smearing_T450K",
        "legacy_mixed_smearing_T600K",
    ]
    set_labels = [
        "harmonic external",
        "fixed smear. 450 K",
        "legacy 300 K",
        "legacy 450 K",
        "legacy 600 K",
    ]
    activation_x = np.arange(len(sets))
    for offset, family in enumerate(FEATURE_FAMILIES):
        by_set = {
            row["evaluation_set"]: row
            for row in activation_rows
            if row["feature_family"] == family
        }
        axes[1, 1].bar(
            activation_x + (offset - 1) * width,
            [by_set[name]["activation_fraction"] for name in sets],
            width,
            color=colors[family],
            label=display[family],
        )
    axes[1, 1].axhline(HARMONIC_ACTIVATION_MAX, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xticks(activation_x, set_labels, rotation=12)
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("full-fit atom-router activation fraction")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
    )
    figure.suptitle("Graphene R2K frozen-descriptor separability audit", y=1.0)
    figure.savefig(output_dir / "descriptor_separability.png", dpi=210, bbox_inches="tight")
    figure.savefig(output_dir / "descriptor_separability.pdf", bbox_inches="tight")
    plt.close(figure)


def force_utility_group(records: list[Record]) -> dict:
    components = sum(record.force_utility["n_force_components"] for record in records)
    atoms = sum(record.force_utility["n_atoms"] for record in records)
    zero2 = sum(
        record.force_utility["zero_delta_squared_error_sum_meV2_A2"]
        for record in records
    )
    active2 = sum(
        record.force_utility["active_expert_squared_error_sum_meV2_A2"]
        for record in records
    )
    positive = sum(
        record.force_utility["positive_utility_atom_count"] for record in records
    )
    return {
        "n_configurations": len(records),
        "n_atoms": atoms,
        "zero_delta_component_RMSE_meV_A": float(np.sqrt(zero2 / components)),
        "active_expert_component_RMSE_meV_A": float(np.sqrt(active2 / components)),
        "positive_utility_atom_fraction": positive / atoms,
        "net_utility_squared_error_reduction_fraction": (
            (zero2 - active2) / zero2 if zero2 > 0.0 else None
        ),
    }


def load_fixed_smearing_delta_targets(
    trajectory_path: Path, prediction_path: Path
) -> tuple[list[Atoms], dict]:
    """Join E50 geometries to the audited R0 force components by trajectory key."""
    structures = read(trajectory_path, index=":")
    arrays = np.load(prediction_path, allow_pickle=False)
    required = {
        "base_forces_eV_A",
        "long_range_forces_eV_A",
        "DFT_target_forces_eV_A",
        "Aprime_mode_operator_order",
        "structure_to_operator_mapping",
        "trajectory_seed",
        "snapshot_index",
    }
    missing = required - set(arrays.files)
    if missing:
        raise ValueError(f"fixed-smearing prediction NPZ misses keys: {sorted(missing)}")
    count = len(arrays["trajectory_seed"])
    if len(structures) != count:
        raise ValueError(
            f"fixed-smearing XYZ/NPZ counts differ: {len(structures)} versus {count}"
        )
    rows = {}
    for row, (seed, snapshot) in enumerate(
        zip(arrays["trajectory_seed"], arrays["snapshot_index"], strict=True)
    ):
        key = (int(seed), int(snapshot))
        if key in rows:
            raise ValueError(f"duplicate fixed-smearing NPZ key {key}")
        rows[key] = row
    joined = []
    maximum_DFT_mismatch = 0.0
    for source in structures:
        key = (
            int(source.info["trajectory_seed"]),
            int(source.info["snapshot_index"]),
        )
        if key not in rows:
            raise ValueError(f"fixed-smearing XYZ key absent from NPZ: {key}")
        row = rows.pop(key)
        dft = np.asarray(arrays["DFT_target_forces_eV_A"][row], float)
        xyz_dft = np.asarray(source.arrays["REF_forces"], float)
        maximum_DFT_mismatch = max(
            maximum_DFT_mismatch, float(np.max(np.abs(dft - xyz_dft)))
        )
        if not np.allclose(dft, xyz_dft, atol=2.0e-8, rtol=0.0):
            raise ValueError(f"DFT force mismatch at fixed-smearing key {key}")
        target = (
            dft
            - np.asarray(arrays["base_forces_eV_A"][row], float)
            - np.asarray(arrays["long_range_forces_eV_A"][row], float)
        )
        atoms = source.copy()
        atoms.arrays["REF_forces"] = target
        mapping = np.asarray(arrays["structure_to_operator_mapping"][row], int)
        if sorted(mapping.tolist()) != list(range(len(atoms))):
            raise ValueError(f"invalid operator mapping at fixed-smearing key {key}")
        mode = np.asarray(arrays["Aprime_mode_operator_order"], complex)[mapping]
        atoms.arrays["R2K_Aprime_mode_real"] = mode.real
        atoms.arrays["R2K_Aprime_mode_imag"] = mode.imag
        atoms.info.update(
            {
                "delta_target_role": "fixed_smearing_thermal",
                "fixed_smearing_source": "E50_DFT_minus_R0_base_minus_R0_long",
                "fixed_smearing_npz_row": row,
            }
        )
        joined.append(atoms)
    if rows:
        raise ValueError(f"unmatched fixed-smearing NPZ keys remain: {sorted(rows)[:3]}")
    seeds = Counter(int(item.info["trajectory_seed"]) for item in joined)
    degauss = sorted({float(item.info["degauss_Ry"]) for item in joined})
    temperatures = sorted(
        {float(item.info["lattice_temperature_K"]) for item in joined}
    )
    provenance = {
        "n_configurations": len(joined),
        "trajectory_seed_counts": {str(key): value for key, value in sorted(seeds.items())},
        "degauss_Ry_values": degauss,
        "lattice_temperature_K_values": temperatures,
        "maximum_XYZ_vs_NPZ_DFT_force_mismatch_eV_A": maximum_DFT_mismatch,
        "target_definition": "DFT_target_forces - base_forces - long_range_forces",
    }
    return joined, provenance


def supercell_score_invariance_audit(
    descriptor_model: torch.nn.Module,
    layout: list,
    node_width: int,
    router_state: dict,
    t0: float,
    t1: float,
    lattice_constant_A: float,
    device: torch.device,
    feature_mode: str = FEATURE_MODE_SIGNED_L0,
    feature_schema: dict | None = None,
) -> dict:
    """Compare one identical local perturbation embedded in 6x6 and 8x8 cells."""

    def perturbed(size: int) -> tuple[Atoms, int]:
        atoms = graphene(
            a=lattice_constant_A,
            size=(size, size, 1),
            vacuum=7.5,
        )
        atoms.pbc = (True, True, True)
        scaled = atoms.get_scaled_positions(wrap=True)
        center = int(
            np.argmin(
                np.sum((scaled[:, :2] - np.array([0.5, 0.5])) ** 2, axis=1)
            )
        )
        vectors = minimum_image_vectors(atoms)[center]
        distances = np.linalg.norm(vectors, axis=1)
        neighbours = np.where((distances > 1.0e-8) & (distances < 1.75))[0]
        if len(neighbours) != 3:
            raise ValueError("ideal graphene perturbation did not have three neighbours")
        angles = np.arctan2(vectors[neighbours, 1], vectors[neighbours, 0])
        neighbours = neighbours[np.argsort(angles)]
        atoms.positions[center] += np.array([0.070, -0.040, 0.030])
        neighbour_offsets = np.array(
            [
                [-0.020, 0.010, -0.010],
                [0.015, -0.012, 0.006],
                [0.005, 0.002, 0.004],
            ]
        )
        atoms.positions[neighbours] += neighbour_offsets
        atoms.info["REF_energy"] = 0.0
        atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3), float)
        return atoms, center

    invariant_vectors = []
    scores = []
    for size in (6, 8):
        atoms, center = perturbed(size)
        node_feats, _ = frozen_forward(
            descriptor_model, atoms, device, compute_force=False
        )
        invariant = (
            numpy_invariant_features(node_feats, feature_schema)[center]
            if feature_schema is not None
            else scalar_invariants(
                node_feats, layout, node_width, feature_mode=feature_mode
            )[center]
        )
        score = float(
            replay_router_score_numpy(invariant[None, :], router_state)[0]
        )
        invariant_vectors.append(invariant)
        scores.append(score)
    gates = c2_smootherstep(np.asarray(scores), t0, t1)
    return {
        "construction": (
            "same central atom and three-neighbour perturbation in ideal 6x6/8x8 "
            "graphene; compare the central invariant and fitted screening score"
        ),
        "lattice_constant_A": lattice_constant_A,
        "score_6x6": scores[0],
        "score_8x8": scores[1],
        "absolute_score_difference": abs(scores[0] - scores[1]),
        "C2_gate_6x6": float(gates[0]),
        "C2_gate_8x8": float(gates[1]),
        "absolute_C2_gate_difference": float(abs(gates[0] - gates[1])),
        "maximum_absolute_invariant_difference": float(
            np.max(np.abs(invariant_vectors[0] - invariant_vectors[1]))
        ),
    }


def _legacy_incomplete_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "--depth3-model", dest="model", type=Path, required=True
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument(
        "--valid-file", "--validation-file", dest="valid_file", type=Path, required=True
    )
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--first-neighbour-cutoff-A", type=float, default=1.75)
    parser.add_argument("--lstsq-rcond", type=float, default=1.0e-10)
    parser.add_argument("--router-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--tension-ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--tension-sign-threshold-meV-A",
        type=float,
        default=TENSION_SIGN_THRESHOLD_MEV_A,
    )
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()

    if (
        args.first_neighbour_cutoff_A <= 0.0
        or args.lstsq_rcond <= 0.0
        or args.router_ridge_alpha <= 0.0
        or args.tension_ridge_alpha <= 0.0
        or args.tension_sign_threshold_meV_A <= 0.0
    ):
        raise ValueError("cutoff, rcond, ridge alpha, and sign threshold must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_structures = read(args.train_file, index=":")
    valid_structures = read(args.valid_file, index=":")
    test_structures = read(args.test_file, index=":")
    support_structures = [
        structure
        for structure in train_structures
        if role(structure) == "fixed_smearing_support_repair"
    ]
    harmonic_train_structures = [
        structure for structure in train_structures if role(structure) == "harmonic_replay"
    ]
    harmonic_validation_structures = [
        structure for structure in valid_structures if role(structure) == "harmonic_replay"
    ]
    valid_support = [
        structure
        for structure in valid_structures
        if role(structure) == "fixed_smearing_support_repair"
    ]
    if (
        len(support_structures),
        len(harmonic_train_structures),
        len(harmonic_validation_structures),
        len(valid_support),
        len(test_structures),
    ) != (9, 72, 25, 9, 26):
        raise ValueError("expected support/harmonic/validation/thermal counts 9/72/25/9/26")
    if any(role(structure) != "thermal" for structure in test_structures):
        raise ValueError("test file must contain only frozen thermal gate structures")
    support_ids = [int(structure.info["sscha_index"]) for structure in support_structures]
    if len(set(support_ids)) != 9:
        raise ValueError("support sscha indices must be unique")
    valid_by_id = {int(structure.info["sscha_index"]): structure for structure in valid_support}
    if set(valid_by_id) != set(support_ids):
        raise ValueError("validation support copy does not match the nine training IDs")
    for support_id, structure in zip(support_ids, support_structures, strict=True):
        if not np.allclose(
            structure.positions, valid_by_id[support_id].positions, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError(f"support geometry mismatch for sscha_index={support_id}")
    degauss_values = {
        float(structure.info["degauss_Ry"])
        for structure in test_structures
        if "degauss_Ry" in structure.info
    }
    if degauss_values != {FIXED_DEGAUSS_RY}:
        raise ValueError(f"thermal test has unexpected degauss values: {degauss_values}")

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    if int(model.num_interactions) != 3:
        raise ValueError("R2K requires the frozen depth-3 MACE")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(dtype=torch.float32, device=device)
    model.eval()
    layout, node_width, invariant_names = mace_invariant_layout(model)

    print("extracting nine support residuals and frozen node_feats", flush=True)
    support = [
        build_record(
            model,
            layout,
            node_width,
            structure,
            "support",
            index,
            str(support_ids[index]),
            args.first_neighbour_cutoff_A,
            args.lstsq_rcond,
            device,
            support_target=True,
        )
        for index, structure in enumerate(support_structures)
    ]

    def representation_records(
        structures: list[Atoms], split: str
    ) -> list[Record]:
        records = []
        for index, structure in enumerate(structures):
            if index % 12 == 0:
                print(f"extracting {split} node_feats {index}/{len(structures)}", flush=True)
            records.append(
                build_record(
                    model,
                    layout,
                    node_width,
                    structure,
                    split,
                    index,
                    f"{split}_{index}",
                    args.first_neighbour_cutoff_A,
                    args.lstsq_rcond,
                    device,
                    support_target=False,
                )
            )
        return records

    harmonic_train = representation_records(harmonic_train_structures, "harmonic_train")
    harmonic_validation = representation_records(
        harmonic_validation_structures, "harmonic_validation"
    )
    thermal = representation_records(test_structures, "thermal")

    probes = {}
    all_fold_rows: list[dict] = []
    all_bond_rows: list[dict] = []
    all_activation_rows: list[dict] = []
    for family in FEATURE_FAMILIES:
        print(f"running complete-configuration LOCO probes: {family}", flush=True)
        result, fold_rows, bond_rows, activation_rows = audit_family(
            family,
            support,
            harmonic_train,
            harmonic_validation,
            thermal,
            args.router_ridge_alpha,
            args.tension_ridge_alpha,
            args.tension_sign_threshold_meV_A,
        )
        probes[family] = result
        all_fold_rows.extend(fold_rows)
        all_bond_rows.extend(bond_rows)
        all_activation_rows.extend(activation_rows)

    primary = probes["mace_invariants"]
    checks = {
        "harmonic_validation_activation_le_5pct": (
            primary["full_fit_router_external_activation"]["harmonic_validation"][
                "activation_fraction"
            ]
            <= HARMONIC_ACTIVATION_MAX + 1.0e-12
        ),
        "support_LOCO_radial_squared_coverage_ge_80pct": (
            primary["LOCO"][
                "support_minimum_norm_bond_tension_squared_coverage"
            ]
            >= SUPPORT_COVERAGE_MIN
        ),
        "each_support_configuration_coverage_ge_50pct": (
            primary["LOCO"]["support_minimum_per_configuration_coverage"]
            >= SUPPORT_PER_CONFIG_COVERAGE_MIN
        ),
        "LOCO_tension_sign_accuracy_ge_80pct": (
            primary["LOCO"]["tension_sign_accuracy_abs_ge_threshold"]
            >= TENSION_SIGN_ACCURACY_MIN
        ),
        "LOCO_tension_weighted_R2_ge_0p5": (
            primary["LOCO"]["tension_weighted_R2"] >= TENSION_WEIGHTED_R2_MIN
        ),
        "thermal_full_fit_activation_le_10pct": (
            primary["full_fit_router_external_activation"]["thermal_all"][
                "activation_fraction"
            ]
            <= THERMAL_ACTIVATION_MAX
        ),
    }
    passed = all(checks.values())
    radial_by_config = {
        record.label: record.radial for record in support
    }
    all_tension = np.concatenate([record.tension_meV_A for record in support])
    summary = {
        "status": (
            "R2K_descriptor_separability_gate_passed"
            if passed
            else "R2K_descriptor_separability_gate_failed"
        ),
        "scope": (
            "zero-update audit of a frozen development representation; probes use "
            "complete-support-configuration LOCO and are not independent DFT holdouts"
        ),
        "decision": (
            "proceed_to_independent_conservative_smooth_routed_residual_branch"
            if passed
            else "increase_local_representation_or_receptive_field_before_residual_training"
        ),
        "new_DFT_labels": 0,
        "long_range_model_modified": False,
        "frozen_MACE_parameter_updates": 0,
        "leakage_control": {
            "support_split_unit": "complete sscha configuration",
            "random_atom_split": False,
            "random_bond_split": False,
            "harmonic_validation_used_for_fit": False,
            "thermal_structures_used_for_fit": False,
            "note": (
                "the frozen depth-3 representation is development-trained; only the "
                "linear probes and their scalers are refit in each LOCO fold"
            ),
        },
        "fixed_condition": {
            "lattice_temperature_K_for_support": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": FIXED_DEGAUSS_RY,
        },
        "gate_thresholds": {
            "harmonic_validation_activation_max": HARMONIC_ACTIVATION_MAX,
            "support_LOCO_minimum_norm_bond_tension_squared_coverage_min": SUPPORT_COVERAGE_MIN,
            "support_each_configuration_coverage_min": SUPPORT_PER_CONFIG_COVERAGE_MIN,
            "tension_sign_target_abs_min_meV_A": args.tension_sign_threshold_meV_A,
            "tension_sign_accuracy_min": TENSION_SIGN_ACCURACY_MIN,
            "tension_weighted_R2_min": TENSION_WEIGHTED_R2_MIN,
            "tension_per_configuration_sign_accuracy_min": (
                TENSION_PER_CONFIG_SIGN_ACCURACY_MIN
            ),
            "tension_per_configuration_weighted_R2_min": (
                TENSION_PER_CONFIG_WEIGHTED_R2_MIN
            ),
            "Aprime_projected_force_RMS_max_meV_A": (
                APRIME_PROJECTED_FORCE_RMS_MAX_MEV_A
            ),
            "support_target_net_force_norm_max_meV_A": NET_FORCE_NORM_MAX_MEV_A,
            "radial_projection_squared_fraction_min": (
                RADIAL_PROJECTION_SQUARED_FRACTION_MIN
            ),
            "thermal_activation_max": THERMAL_ACTIVATION_MAX,
        },
        "primary_gate_checks": checks,
        "primary_gate_passed": passed,
        "force_to_tension_definition": {
            "force_correction": "REF_forces - frozen_depth3_forces",
            "bonds": (
                f"unique C-C pairs below {args.first_neighbour_cutoff_A:g} A; "
                "every atom must have degree three"
            ),
            "orientation": (
                "positive tension pulls each endpoint toward the other; the scalar "
                "sign is unchanged by endpoint exchange"
            ),
            "solver": "numpy.linalg.lstsq minimum-norm least-squares solution",
            "lstsq_rcond": args.lstsq_rcond,
            "router_coverage": (
                "sum(tension^2 for active held-out bonds) / sum(tension^2), "
                "using the minimum-norm radial coordinate"
            ),
            "support_tension_quantiles_meV_A": {
                f"q{int(q * 100):02d}": float(value)
                for q, value in zip(
                    [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0],
                    np.quantile(all_tension, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]),
                    strict=True,
                )
            },
            "per_support_configuration": radial_by_config,
        },
        "feature_definitions": {
            "frozen_atom_invariants": {
                "source": "concatenated node_feats after each of three MACE products",
                "scalar_channels": "signed 0e value",
                "equivariant_channels": "sum of squared components per irrep copy",
                "dimension": len(invariant_names),
                "raw_node_feats_width": node_width,
                "irreps_layout": layout,
            },
            "bond_exchange_invariance": (
                "for each endpoint scalar vector h: [mean(h_i,h_j), "
                "(h_i-h_j)^2, r_ij, r_ij^2]"
            ),
            "geometry_control": (
                "the same bond symmetrization applied to six sorted endpoint-neighbour "
                "distances and three sorted first-shell angle cosines"
            ),
            "bond_length_control": "[r, r^2, r^3, r^4]",
            "probe": {
                "model": "fixed-alpha standardized linear ridge",
                "router_alpha": args.router_ridge_alpha,
                "tension_alpha": args.tension_ridge_alpha,
                "router_positive_weight": (
                    "minimum-norm tension squared, equal total weight per support config"
                ),
                "router_negative_weight": (
                    "uniform bonds, equal total weight per harmonic-train config"
                ),
                "tension_weight": (
                    "max((abs(tension)/30)^2, 0.01), equal total weight per config"
                ),
            },
        },
        "counts": {
            "support_configurations": len(support),
            "support_bonds": int(sum(len(record.distance_A) for record in support)),
            "harmonic_train_configurations": len(harmonic_train),
            "harmonic_validation_configurations": len(harmonic_validation),
            "thermal_configurations": len(thermal),
            "thermal_temperature_counts": dict(
                sorted(
                    Counter(
                        f"{record.temperature_K:g}K" for record in thermal
                    ).items()
                )
            ),
        },
        "probes": probes,
        "inputs": {
            "model": {"path": str(args.model), "sha256": sha256(args.model)},
            "train_file": {
                "path": str(args.train_file),
                "sha256": sha256(args.train_file),
            },
            "valid_file": {
                "path": str(args.valid_file),
                "sha256": sha256(args.valid_file),
            },
            "test_file": {
                "path": str(args.test_file),
                "sha256": sha256(args.test_file),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_source = Path(__file__).resolve()
    script_snapshot = args.output_dir / "run_script_snapshot.py"
    shutil.copy2(script_source, script_snapshot)
    script_snapshot_hash = sha256(script_snapshot)
    if script_snapshot_hash != summary["inputs"]["script"]["sha256"]:
        raise RuntimeError("run-script snapshot hash disagrees with the executed script")
    summary["inputs"]["script_snapshot"] = {
        "path": str(script_snapshot),
        "sha256": script_snapshot_hash,
    }
    atomic_json(args.output_dir / "descriptor_separability.json", summary)
    atomic_csv(
        args.output_dir / "loco_fold_metrics.csv",
        list(all_fold_rows[0]),
        all_fold_rows,
    )
    atomic_csv(
        args.output_dir / "support_bond_loco_predictions.csv",
        list(all_bond_rows[0]),
        all_bond_rows,
    )
    atomic_csv(
        args.output_dir / "router_activation_metrics.csv",
        list(all_activation_rows[0]),
        all_activation_rows,
    )
    plot_results(
        args.output_dir,
        support,
        probes,
        all_bond_rows,
        all_activation_rows,
    )
    concise = {
        "status": summary["status"],
        "decision": summary["decision"],
        "primary_gate_checks": checks,
        "primary": primary,
        "controls": {
            family: probes[family] for family in FEATURE_FAMILIES[1:]
        },
    }
    print(json.dumps(concise, indent=2))
    return 0


def main() -> int:
    """Run the corrected R2K audit on the full frozen R2C data split.

    The older entry point above was retained only to make the development
    history readable.  This production entry point uses existing thermal
    training data for utility-probe fitting, keeps thermal validation/test
    external, and permits a short-cutoff descriptor model distinct from the
    frozen depth-3 delta expert whose utility is being audited.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert-model", "--baseline-model", "--model", "--depth3-model",
        dest="expert_model", type=Path, required=True,
    )
    parser.add_argument(
        "--descriptor-model", type=Path,
        help="Frozen feature model; defaults to the expert model.",
    )
    parser.add_argument(
        "--feature-mode",
        "--mace-feature-mode",
        dest="feature_mode",
        choices=FEATURE_MODES,
        default=FEATURE_MODE_SIGNED_L0,
        help=(
            "Frozen node feature map. signed_l0 preserves the historical R2K "
            "behavior; signed_l0_plus_tensor_power also includes mean_m h_m^2 "
            "for each l>0 multiplicity channel; "
            "signed_l0_plus_tensor_power_plus_gram_sketch16 adds the fixed "
            "16-column off-diagonal Gram sketch for the r3-h16-l1 descriptor."
        ),
    )
    parser.add_argument("--train-file", "--train", dest="train_file", type=Path, required=True)
    parser.add_argument(
        "--valid-file", "--validation-file", "--valid", dest="valid_file",
        type=Path, required=True,
    )
    parser.add_argument("--test-file", "--test", dest="test_file", type=Path, required=True)
    parser.add_argument(
        "--fixed-thermal-file",
        type=Path,
        required=True,
        help=(
            "E50 fixed-smearing all60.xyz; seeds 0+1 fit the present probe and "
            "seed 2 is an opened descriptor-selection screen, not final external validation."
        ),
    )
    parser.add_argument(
        "--fixed-thermal-target-npz",
        type=Path,
        required=True,
        help="R0 audited base/long/DFT components aligned by seed and snapshot.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--first-neighbour-cutoff-A", type=float, default=1.75)
    parser.add_argument("--lstsq-rcond", type=float, default=1.0e-10)
    parser.add_argument("--router-ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--router-feature-map",
        choices=ROUTER_FEATURE_MAPS,
        default=ROUTER_FEATURE_MAP_LINEAR,
        help=(
            "Fixed utility-router map. linear preserves historical R2K; "
            "diagonal_quadratic uses train-fold StandardScaler x->z, [z,z^2], "
            "then the existing StandardScaler and fixed-alpha Ridge."
        ),
    )
    parser.add_argument("--tension-ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--utility-deadband-meV2-A2", type=float, default=25.0,
        help="Ignore near-zero atom utilities when fitting the router probe.",
    )
    parser.add_argument(
        "--router-transition-width", type=float, default=0.10,
        help="Frozen score width reserved for the later C2 smootherstep.",
    )
    parser.add_argument(
        "--tension-sign-threshold-meV-A",
        type=float,
        default=TENSION_SIGN_THRESHOLD_MEV_A,
    )
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()

    if (
        args.feature_mode == FEATURE_MODE_GRAM_SKETCH16
        and args.router_feature_map != ROUTER_FEATURE_MAP_LINEAR
    ):
        raise ValueError(
            "the fixed Gram-sketch16 lane must use --router-feature-map linear"
        )

    positive_arguments = (
        args.first_neighbour_cutoff_A,
        args.lstsq_rcond,
        args.router_ridge_alpha,
        args.tension_ridge_alpha,
        args.utility_deadband_meV2_A2,
        args.router_transition_width,
        args.tension_sign_threshold_meV_A,
    )
    if any(value <= 0.0 for value in positive_arguments):
        raise ValueError("cutoff, rcond, alphas, deadband, and transition width must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_structures = read(args.train_file, index=":")
    valid_structures = read(args.valid_file, index=":")
    test_structures = read(args.test_file, index=":")

    def select(structures: list[Atoms], target_role: str) -> list[Atoms]:
        return [structure for structure in structures if role(structure) == target_role]

    support_structures = select(train_structures, "fixed_smearing_support_repair")
    harmonic_train_structures = select(train_structures, "harmonic_replay")
    thermal_train_structures = select(train_structures, "thermal")
    harmonic_validation_structures = select(valid_structures, "harmonic_replay")
    thermal_validation_structures = select(valid_structures, "thermal")
    validation_support = select(valid_structures, "fixed_smearing_support_repair")
    thermal_test_structures = select(test_structures, "thermal")
    observed_counts = {
        "train_support": len(support_structures),
        "train_harmonic": len(harmonic_train_structures),
        "train_thermal": len(thermal_train_structures),
        "validation_support": len(validation_support),
        "validation_harmonic": len(harmonic_validation_structures),
        "validation_thermal": len(thermal_validation_structures),
        "test_thermal": len(thermal_test_structures),
    }
    expected_counts = {
        "train_support": 9,
        "train_harmonic": 72,
        "train_thermal": 224,
        "validation_support": 0,
        "validation_harmonic": 25,
        "validation_thermal": 68,
        "test_thermal": 26,
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"R2K requires the frozen R2C split; observed {observed_counts}, "
            f"expected {expected_counts}"
        )
    if len(test_structures) != len(thermal_test_structures):
        raise ValueError("test file contains a non-thermal configuration")
    support_ids = [int(structure.info["sscha_index"]) for structure in support_structures]
    if len(set(support_ids)) != 9:
        raise ValueError("support sscha indices must be unique")
    fixed_thermal_structures, fixed_thermal_provenance = (
        load_fixed_smearing_delta_targets(
            args.fixed_thermal_file, args.fixed_thermal_target_npz
        )
    )
    fixed_seed_counts = Counter(
        int(structure.info["trajectory_seed"])
        for structure in fixed_thermal_structures
    )
    if fixed_seed_counts != Counter({0: 20, 1: 20, 2: 20}):
        raise ValueError(f"unexpected E50 seed counts: {fixed_seed_counts}")
    fixed_degauss = fixed_thermal_provenance["degauss_Ry_values"]
    fixed_temperatures = fixed_thermal_provenance["lattice_temperature_K_values"]
    if (
        len(fixed_degauss) != 1
        or abs(fixed_degauss[0] - FIXED_DEGAUSS_RY) > 5.0e-10
        or fixed_temperatures != [450.0]
    ):
        raise ValueError(
            "E50 must be T_lat=450 K at the frozen degauss; observed "
            f"T={fixed_temperatures}, degauss={fixed_degauss}"
        )
    fixed_development_structures = [
        structure
        for structure in fixed_thermal_structures
        if int(structure.info["trajectory_seed"]) in {0, 1}
    ]
    fixed_external_structures = [
        structure
        for structure in fixed_thermal_structures
        if int(structure.info["trajectory_seed"]) == 2
    ]
    legacy_geometry_rows: dict[str, list[tuple[str, Atoms]]] = {}
    for split_name, structures in (
        ("train", thermal_train_structures),
        ("validation", thermal_validation_structures),
        ("test", thermal_test_structures),
    ):
        for structure in structures:
            legacy_geometry_rows.setdefault(geometry_fingerprint(structure), []).append(
                (split_name, structure)
            )
    overlap_records = []
    target_differences_by_seed: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    for structure in fixed_thermal_structures:
        seed = int(structure.info["trajectory_seed"])
        matches = legacy_geometry_rows.get(geometry_fingerprint(structure), [])
        if not matches:
            raise ValueError("an E50 geometry has no matching historical R2C geometry")
        raw_targets = [np.asarray(item.arrays["REF_forces"], float) for _, item in matches]
        if any(
            not np.allclose(raw_targets[0], value, atol=2.0e-8, rtol=0.0)
            for value in raw_targets[1:]
        ):
            raise ValueError("duplicate historical R2C geometry has inconsistent targets")
        difference = np.asarray(structure.arrays["REF_forces"], float) - raw_targets[0]
        target_differences_by_seed[seed].append(difference)
        overlap_records.append(
            {
                "trajectory_seed": seed,
                "snapshot_index": int(structure.info["snapshot_index"]),
                "historical_R2C_splits": sorted({name for name, _ in matches}),
                "historical_R2C_record_count": len(matches),
            }
        )

    def difference_metrics(values: list[np.ndarray]) -> dict:
        joined = np.concatenate([np.asarray(value, float).reshape(-1) for value in values])
        return {
            "component_RMSE_meV_A": float(np.sqrt(np.mean(joined**2)) * 1000.0),
            "component_max_abs_meV_A": float(np.max(np.abs(joined)) * 1000.0),
        }

    fixed_geometry_overlap = {
        "all_60_geometries_match_historical_R2C": len(overlap_records) == 60,
        "records": overlap_records,
        "fixed_minus_historical_R2C_delta_target": {
            f"seed{seed}": difference_metrics(values)
            for seed, values in target_differences_by_seed.items()
        },
        "interpretation": (
            "the fixed-smearing labels are exact E50 targets, but all geometries and "
            "nearby historical targets were already exposed during depth3/descriptor "
            "development; seed2 is only held out from the present fixed-map "
            "Ridge-probe fit"
        ),
    }

    descriptor_path = args.descriptor_model or args.expert_model
    expert_hash = sha256(args.expert_model)
    descriptor_hash = sha256(descriptor_path)
    expert_model = torch.load(args.expert_model, map_location="cpu", weights_only=False)
    if int(expert_model.num_interactions) != 3:
        raise ValueError("the audited utility expert must be the frozen depth-3 delta MACE")
    if descriptor_hash == expert_hash:
        descriptor_model = expert_model
    else:
        descriptor_model = torch.load(
            descriptor_path, map_location="cpu", weights_only=False
        )
    for model in {id(expert_model): expert_model, id(descriptor_model): descriptor_model}.values():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(dtype=torch.float32, device=device)
        model.eval()
    layout, node_width, invariant_names = mace_invariant_layout(
        descriptor_model, feature_mode=args.feature_mode
    )
    feature_schema = invariant_feature_schema(
        layout, node_width, feature_mode=args.feature_mode
    )
    if invariant_names != feature_schema["feature_names"]:
        raise RuntimeError("feature names disagree with the invariant replay schema")

    def make_records(
        structures: list[Atoms], split: str, decompose_target: bool,
        explicit_labels: list[str] | None = None,
    ) -> list[Record]:
        records = []
        for index, structure in enumerate(structures):
            if index % 25 == 0:
                print(f"extracting {split}: {index}/{len(structures)}", flush=True)
            temperature = structure.info.get("lattice_temperature_K")
            suffix = (
                f"_T{float(temperature):g}K" if temperature is not None else ""
            )
            label = (
                explicit_labels[index]
                if explicit_labels is not None
                else f"{split}_{index}{suffix}"
            )
            records.append(
                build_record(
                    descriptor_model,
                    expert_model,
                    layout,
                    node_width,
                    structure,
                    split,
                    index,
                    label,
                    args.first_neighbour_cutoff_A,
                    args.lstsq_rcond,
                    device,
                    decompose_target,
                    feature_mode=args.feature_mode,
                )
            )
        return records

    support = make_records(
        support_structures, "support", True, [str(value) for value in support_ids]
    )
    harmonic_train = make_records(
        harmonic_train_structures, "harmonic_train", False
    )
    thermal_train = make_records(thermal_train_structures, "thermal_train", False)
    harmonic_validation = make_records(
        harmonic_validation_structures, "harmonic_validation", False
    )
    thermal_validation = make_records(
        thermal_validation_structures, "thermal_validation", False
    )
    thermal_test = make_records(thermal_test_structures, "thermal_test", False)
    fixed_development = make_records(
        fixed_development_structures,
        "fixed_smearing_development",
        False,
        [
            f"fixed_seed{int(item.info['trajectory_seed'])}_snapshot{int(item.info['snapshot_index'])}"
            for item in fixed_development_structures
        ],
    )
    fixed_external = make_records(
        fixed_external_structures,
        "fixed_smearing_external",
        True,
        [
            f"fixed_seed2_snapshot{int(item.info['snapshot_index'])}"
            for item in fixed_external_structures
        ],
    )
    router_train = harmonic_train + fixed_development
    pruning_fit_values = concatenate_router(router_train, "mace_invariants")
    feature_pruning = fit_relative_variance_feature_mask(
        pruning_fit_values,
        invariant_names,
        relative_std_threshold=FEATURE_RELATIVE_STD_THRESHOLD,
    )
    feature_pruning.pop("pruning_state_sha256")
    feature_pruning.update(
        {
            "fit_record_sets": [
                "harmonic_train",
                "fixed_smearing_development_seed0_seed1",
            ],
            "fit_configuration_count": len(router_train),
            "support_configurations_used": False,
            "harmonic_validation_used": False,
            "fixed_smearing_seed2_used": False,
            "legacy_mixed_smearing_used": False,
            "target_or_utility_used": False,
        }
    )
    feature_pruning["pruning_state_sha256"] = canonical_json_sha256(
        feature_pruning
    )
    feature_schema = feature_schema_with_pruning(feature_schema, feature_pruning)
    invariant_names = list(feature_schema["feature_names"])
    all_records = (
        support
        + harmonic_train
        + thermal_train
        + harmonic_validation
        + thermal_validation
        + thermal_test
        + fixed_development
        + fixed_external
    )
    apply_mace_invariant_mask_to_records(all_records, feature_schema)
    legacy_mixed_smearing = thermal_train + thermal_validation + thermal_test

    probes = {}
    all_fold_rows: list[dict] = []
    all_bond_rows: list[dict] = []
    all_activation_rows: list[dict] = []
    for family in FEATURE_FAMILIES:
        print(f"running complete-configuration probes: {family}", flush=True)
        result, fold_rows, bond_rows, activation_rows = audit_family(
            family,
            support,
            router_train,
            harmonic_validation,
            fixed_external,
            legacy_mixed_smearing,
            args.router_ridge_alpha,
            args.tension_ridge_alpha,
            args.tension_sign_threshold_meV_A,
            args.utility_deadband_meV2_A2,
            args.router_transition_width,
            args.router_feature_map,
        )
        probes[family] = result
        all_fold_rows.extend(fold_rows)
        all_bond_rows.extend(bond_rows)
        all_activation_rows.extend(activation_rows)

    primary = probes["mace_invariants"]
    primary_router = primary["full_fit_router"]
    primary_router_state = primary_router["state"]
    router_feature_map = str(
        primary_router_state.get(
            "router_feature_map", ROUTER_FEATURE_MAP_LINEAR
        )
    )
    router_input_dimension = int(feature_schema["output_dimension"])
    router_map_schema = router_feature_map_schema(
        router_feature_map, router_input_dimension
    )
    if (
        primary_router_state.get("router_feature_map_schema_sha256")
        != router_map_schema["schema_sha256"]
        or primary_router_state.get("router_feature_map_schema")
        != router_map_schema
    ):
        raise RuntimeError("serialized router feature-map schema/hash mismatch")
    router_mapped_dimension = int(router_map_schema["output_dimension"])
    router_state_dimensions = {
        "unpruned_feature_schema": int(
            feature_schema["unpruned_output_dimension"]
        ),
        "retained_feature_schema": int(feature_schema["output_dimension"]),
        "retained_feature_names": len(invariant_names),
        "router_input_feature_dimension": int(
            primary_router_state.get(
                "router_input_feature_dimension", router_input_dimension
            )
        ),
        "router_mapped_feature_dimension": int(
            primary_router_state.get(
                "router_mapped_feature_dimension", router_mapped_dimension
            )
        ),
        "input_scaler_mean": len(
            primary_router_state.get("input_scaler_mean", [])
        ),
        "input_scaler_scale": len(
            primary_router_state.get("input_scaler_scale", [])
        ),
        "scaler_mean": len(primary_router_state["scaler_mean"]),
        "scaler_scale": len(primary_router_state["scaler_scale"]),
        "ridge_coefficient": int(
            np.asarray(primary_router_state["ridge_coefficient"]).size
        ),
    }
    input_dimension_fields = (
        "retained_feature_schema",
        "retained_feature_names",
        "router_input_feature_dimension",
    )
    if any(
        router_state_dimensions[name] != router_input_dimension
        for name in input_dimension_fields
    ):
        raise RuntimeError(
            "router input feature dimensions disagree: "
            f"{router_state_dimensions}"
        )
    if router_feature_map == ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC:
        input_scaler_fields = ("input_scaler_mean", "input_scaler_scale")
        if any(
            router_state_dimensions[name] != router_input_dimension
            for name in input_scaler_fields
        ):
            raise RuntimeError(
                "router input-scaler dimensions disagree: "
                f"{router_state_dimensions}"
            )
    elif any(
        router_state_dimensions[name] != 0
        for name in ("input_scaler_mean", "input_scaler_scale")
    ):
        raise RuntimeError("linear router unexpectedly contains an input scaler")
    mapped_dimension_fields = (
        "router_mapped_feature_dimension",
        "scaler_mean",
        "scaler_scale",
        "ridge_coefficient",
    )
    if any(
        router_state_dimensions[name] != router_mapped_dimension
        for name in mapped_dimension_fields
    ):
        raise RuntimeError(
            "router mapped feature/scaler dimensions disagree: "
            f"{router_state_dimensions}"
        )
    router_scale = np.asarray(primary_router_state["scaler_scale"], float)
    router_mean = np.asarray(primary_router_state["scaler_mean"], float)
    router_coefficient = np.asarray(
        primary_router_state["ridge_coefficient"], float
    )
    router_input_scale = np.asarray(
        primary_router_state.get("input_scaler_scale", router_scale), float
    )
    router_input_mean = np.asarray(
        primary_router_state.get("input_scaler_mean", router_mean), float
    )
    if not all(
        np.all(np.isfinite(values))
        for values in (
            router_scale,
            router_mean,
            router_coefficient,
            router_input_scale,
            router_input_mean,
        )
    ) or not np.isfinite(float(primary_router_state["ridge_intercept"])):
        raise RuntimeError("router/scaler state contains a non-finite value")
    if np.any(router_scale <= 0.0) or np.any(router_input_scale <= 0.0):
        raise RuntimeError("router scaler has a negative or zero scale")
    retained_training_rms = retained_training_rms_from_schema(feature_schema)
    if np.any(retained_training_rms <= np.finfo(np.float32).tiny):
        raise RuntimeError(
            "retained training RMS is zero or float32-subnormal after pruning"
        )
    router_scaler_relative_to_training_rms = (
        router_input_scale / retained_training_rms
    )
    if np.any(
        router_scaler_relative_to_training_rms
        <= FEATURE_RELATIVE_STD_THRESHOLD
    ):
        raise RuntimeError(
            "router scaler variation relative to saved training RMS is below the "
            "fixed training-only pruning threshold"
        )
    router_replay_numerics = {
        "minimum_scaler_over_training_RMS": float(
            np.min(router_scaler_relative_to_training_rms)
        ),
        "numerical_guard_scaler": (
            "input_scaler"
            if router_feature_map == ROUTER_FEATURE_MAP_DIAGONAL_QUADRATIC
            else "mapped_scaler"
        ),
        "mapped_scaler_minimum": float(np.min(router_scale)),
        "relative_std_threshold": FEATURE_RELATIVE_STD_THRESHOLD,
        "all_retained_scaler_over_RMS_gt_threshold": True,
    }
    primary_router_state_sha256 = canonical_json_sha256(primary_router_state)
    external = primary["full_fit_router"]["external_evaluation"]
    harmonic_metrics = external["harmonic_external"]
    fixed_external_metrics = external["fixed_smearing_external_all"]
    legacy_mixed_metrics = external["legacy_mixed_smearing_all"]
    harmonic_per_config = list(harmonic_metrics["per_configuration"].values())
    support_cell_lengths = [
        float(np.linalg.norm(vector))
        for vector, periodic in zip(
            np.asarray(support_structures[0].cell, float),
            np.asarray(support_structures[0].pbc, bool),
            strict=True,
        )
        if periodic
    ]
    minimum_support_periodic_cell_length_A = min(support_cell_lengths)
    descriptor_interaction_diameter_A = (
        2.0 * float(descriptor_model.r_max) * int(descriptor_model.num_interactions)
    )
    representation_has_no_periodic_wrap_by_bound = (
        descriptor_interaction_diameter_A
        < minimum_support_periodic_cell_length_A
    )
    supercell_invariance = supercell_score_invariance_audit(
        descriptor_model,
        layout,
        node_width,
        primary_router["state"],
        primary_router["t0"],
        primary_router["t1"],
        float(np.linalg.norm(np.asarray(support_structures[0].cell, float)[0]))
        / 6.0,
        device,
        feature_mode=args.feature_mode,
        feature_schema=feature_schema,
    )
    maximum_support_target_net_force_norm_meV_A = max(
        float(record.target_radial["net_force_correction_norm_meV_A"])
        for record in support
    )
    minimum_support_target_radial_fraction = min(
        float(record.target_radial["radial_force_squared_fraction"])
        for record in support
    )
    minimum_support_remaining_radial_fraction = min(
        float(record.remaining_radial["radial_force_squared_fraction"])
        for record in support
    )
    checks = {
        "representation_no_periodic_wrap_by_diameter_bound": (
            representation_has_no_periodic_wrap_by_bound
        ),
        "same_local_perturbation_6x6_8x8_score_invariant": (
            supercell_invariance["absolute_score_difference"] <= 1.0e-3
            and supercell_invariance["absolute_C2_gate_difference"] <= 1.0e-3
        ),
        "support_target_net_force_norm_le_1meV_A": (
            maximum_support_target_net_force_norm_meV_A
            <= NET_FORCE_NORM_MAX_MEV_A
        ),
        "each_support_target_radial_projection_fraction_ge_0p5": (
            minimum_support_target_radial_fraction
            >= RADIAL_PROJECTION_SQUARED_FRACTION_MIN
        ),
        "each_support_remaining_radial_projection_fraction_ge_0p5": (
            minimum_support_remaining_radial_fraction
            >= RADIAL_PROJECTION_SQUARED_FRACTION_MIN
        ),
        "harmonic_external_activation_le_5pct": (
            harmonic_metrics["activation_fraction"]
            <= HARMONIC_ACTIVATION_MAX + 1.0e-12
        ),
        "each_harmonic_external_activation_le_5pct": all(
            item["activation_fraction"] <= HARMONIC_ACTIVATION_MAX + 1.0e-12
            for item in harmonic_per_config
        ),
        "harmonic_external_hard_route_proxy_RMSE_le_gate": (
            harmonic_metrics["hard_route_force_proxy"]["RMSE_meV_A"]
            <= HARMONIC_FORCE_RMSE_MAX_MEV_A
        ),
        "each_harmonic_external_hard_route_proxy_RMSE_le_gate": all(
            item["hard_route_force_proxy_RMSE_meV_A"]
            <= HARMONIC_FORCE_RMSE_MAX_MEV_A
            for item in harmonic_per_config
        ),
        "harmonic_external_C2_gate_mean_le_5pct": (
            harmonic_metrics["C2_gate_mean"]
            <= HARMONIC_ACTIVATION_MAX + 1.0e-12
        ),
        "harmonic_external_C2_route_proxy_RMSE_le_gate": (
            harmonic_metrics["C2_route_force_proxy"]["RMSE_meV_A"]
            <= HARMONIC_FORCE_RMSE_MAX_MEV_A
        ),
        "support_LOCO_target_tension_coverage_ge_80pct": (
            primary["LOCO"]["support_minimum_norm_bond_tension_squared_coverage"]
            >= SUPPORT_COVERAGE_MIN
        ),
        "support_LOCO_positive_utility_retention_ge_80pct": (
            primary["LOCO"]["support_positive_utility_retention"]
            >= SUPPORT_COVERAGE_MIN
        ),
        "support_LOCO_remaining_repair_coverage_ge_80pct": (
            primary["LOCO"][
                "support_remaining_repair_tension_squared_coverage"
            ]
            >= SUPPORT_COVERAGE_MIN
        ),
        "support_LOCO_C2_positive_utility_retention_ge_80pct": (
            primary["LOCO"]["support_C2_positive_utility_retention"]
            >= SUPPORT_COVERAGE_MIN
        ),
        "support_LOCO_C2_remaining_repair_coverage_ge_80pct": (
            primary["LOCO"][
                "support_C2_remaining_repair_tension_squared_coverage"
            ]
            >= SUPPORT_COVERAGE_MIN
        ),
        "each_support_configuration_coverage_ge_50pct": (
            primary["LOCO"]["support_minimum_per_configuration_coverage"]
            >= SUPPORT_PER_CONFIG_COVERAGE_MIN
        ),
        "each_support_positive_utility_retention_ge_50pct": (
            primary["LOCO"][
                "support_minimum_per_configuration_positive_utility_retention"
            ]
            >= SUPPORT_PER_CONFIG_COVERAGE_MIN
        ),
        "each_support_remaining_repair_coverage_ge_50pct": (
            primary["LOCO"][
                "support_minimum_per_configuration_remaining_repair_coverage"
            ]
            >= SUPPORT_PER_CONFIG_COVERAGE_MIN
        ),
        "each_support_C2_positive_utility_retention_ge_50pct": (
            primary["LOCO"][
                "support_minimum_per_configuration_C2_positive_utility_retention"
            ]
            >= SUPPORT_PER_CONFIG_COVERAGE_MIN
        ),
        "each_support_C2_remaining_repair_coverage_ge_50pct": (
            primary["LOCO"][
                "support_minimum_per_configuration_C2_remaining_repair_coverage"
            ]
            >= SUPPORT_PER_CONFIG_COVERAGE_MIN
        ),
        "LOCO_tension_sign_accuracy_ge_80pct": (
            primary["LOCO"]["tension_sign_accuracy_abs_ge_threshold"]
            >= TENSION_SIGN_ACCURACY_MIN
        ),
        "LOCO_tension_weighted_R2_ge_0p5": (
            primary["LOCO"]["tension_weighted_R2"] >= TENSION_WEIGHTED_R2_MIN
        ),
        "each_LOCO_tension_sign_accuracy_ge_65pct": (
            primary["LOCO"][
                "tension_minimum_per_configuration_sign_accuracy"
            ]
            >= TENSION_PER_CONFIG_SIGN_ACCURACY_MIN
        ),
        "each_LOCO_tension_weighted_R2_ge_0": (
            primary["LOCO"][
                "tension_minimum_per_configuration_weighted_R2"
            ]
            >= TENSION_PER_CONFIG_WEIGHTED_R2_MIN
        ),
        "fixed_smearing_seed2_positive_utility_retention_ge_80pct": (
            fixed_external_metrics["positive_utility_retention"]
            >= THERMAL_UTILITY_RETENTION_MIN
        ),
        "fixed_smearing_seed2_harmful_utility_exposure_le_20pct": (
            fixed_external_metrics["harmful_utility_exposure"] is None
            or fixed_external_metrics["harmful_utility_exposure"]
            <= HARMFUL_UTILITY_EXPOSURE_MAX
        ),
        "fixed_smearing_seed2_target_tension_coverage_ge_80pct": (
            fixed_external_metrics["target_delta_tension_squared_coverage"]
            >= THERMAL_TARGET_TENSION_COVERAGE_MIN
        ),
        "fixed_smearing_seed2_remaining_repair_coverage_ge_80pct": (
            fixed_external_metrics[
                "remaining_repair_tension_squared_coverage"
            ]
            >= THERMAL_TARGET_TENSION_COVERAGE_MIN
        ),
        "fixed_smearing_seed2_C2_positive_utility_retention_ge_80pct": (
            fixed_external_metrics["C2_positive_utility_retention"]
            >= THERMAL_UTILITY_RETENTION_MIN
        ),
        "fixed_smearing_seed2_C2_harmful_utility_exposure_le_20pct": (
            fixed_external_metrics["C2_harmful_utility_exposure"] is None
            or fixed_external_metrics["C2_harmful_utility_exposure"]
            <= HARMFUL_UTILITY_EXPOSURE_MAX
        ),
        "fixed_smearing_seed2_C2_remaining_repair_coverage_ge_80pct": (
            fixed_external_metrics[
                "C2_remaining_repair_tension_squared_coverage"
            ]
            >= THERMAL_TARGET_TENSION_COVERAGE_MIN
        ),
        "fixed_smearing_seed2_hard_route_force_screen": (
            fixed_external_metrics["hard_route_force_proxy"]["RMSE_meV_A"]
            <= THERMAL_FORCE_RMSE_MAX_MEV_A
            and fixed_external_metrics["hard_route_force_proxy"]["max_abs_meV_A"]
            <= THERMAL_FORCE_MAX_ABS_MAX_MEV_A
        ),
        "fixed_smearing_seed2_C2_route_force_screen": (
            fixed_external_metrics["C2_route_force_proxy"]["RMSE_meV_A"]
            <= THERMAL_FORCE_RMSE_MAX_MEV_A
            and fixed_external_metrics["C2_route_force_proxy"]["max_abs_meV_A"]
            <= THERMAL_FORCE_MAX_ABS_MAX_MEV_A
        ),
        "fixed_smearing_seed2_Aprime_hard_proxy_le_15meV_A": (
            fixed_external_metrics["Aprime_route_force_screen"]["hard_RMS_meV_A"]
            <= APRIME_PROJECTED_FORCE_RMS_MAX_MEV_A
        ),
        "fixed_smearing_seed2_Aprime_C2_proxy_le_15meV_A": (
            fixed_external_metrics["Aprime_route_force_screen"]["C2_RMS_meV_A"]
            <= APRIME_PROJECTED_FORCE_RMS_MAX_MEV_A
        ),
    }
    repair_check_names = [
        name
        for name in checks
        if (
            "LOCO_tension_" in name
            or "remaining_repair_coverage" in name
            or "radial_projection" in name
            or "target_net_force" in name
        )
    ]
    router_check_names = [name for name in checks if name not in repair_check_names]
    router_passed = all(checks[name] for name in router_check_names)
    repair_passed = all(checks[name] for name in repair_check_names)
    passed = router_passed and repair_passed
    if passed:
        decision = "screening_proxy_passed_proceed_to_R2L_conservative_autograd_gate"
    elif router_passed:
        decision = "screening_router_separable_but_repair_representation_must_be_improved"
    else:
        decision = "increase_router_local_representation_without_new_DFT"

    all_target_tension = np.concatenate(
        [record.target_tension_meV_A for record in support]
    )
    all_remaining_tension = np.concatenate(
        [record.remaining_tension_meV_A for record in support]
    )
    quantile_levels = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]

    def quantiles(values: np.ndarray) -> dict[str, float]:
        return {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(
                quantile_levels, np.quantile(values, quantile_levels), strict=True
            )
        }

    approximate_receptive_field = float(descriptor_model.r_max) * int(
        descriptor_model.num_interactions
    )
    legacy_mixed_conditions = [
        {
            "lattice_temperature_K": temperature,
            "degauss_Ry": degauss,
        }
        for temperature, degauss in sorted(
            {
                (
                    float(structure.info["lattice_temperature_K"]),
                    float(structure.info["degauss_Ry"]),
                )
                for structure in (
                    thermal_train_structures
                    + thermal_validation_structures
                    + thermal_test_structures
                )
            }
        )
    ]
    summary = {
        "status": (
            "R2K_noncausal_screening_proxy_checks_satisfied"
            if passed
            else "R2K_noncausal_screening_proxy_checks_not_satisfied"
        ),
        "finite_temperature_validation_closed": False,
        "decision": decision,
        "scope": (
            "zero-optimizer-step, non-causal screening audit; fixed-smearing E50 "
            "seeds 0+1 fit the utility probe and seed 2 is a held discovery trajectory. "
            "Because multiple descriptor families are compared on it, seed 2 is not a "
            "final external/blind validation. The conservative autograd energy-gate "
            "test is intentionally deferred to R2L"
        ),
        "new_DFT_labels": 0,
        "optimizer_steps": 0,
        "long_range_model_modified": False,
        "fixed_condition": {
            "source": "E50 fixed-smearing T450 DFT reference",
            "development_trajectory_seeds": [0, 1],
            "held_screening_trajectory_seed": 2,
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": FIXED_DEGAUSS_RY,
        },
        "screening_proxy_checks": checks,
        "router_screening_checks_satisfied": router_passed,
        "repair_predictability_checks_satisfied": repair_passed,
        "screening_proxy_checks_satisfied": passed,
        "deployment_authorized": False,
        "conservative_autograd_gate": "pending_R2L",
        "gate_thresholds": {
            "harmonic_validation_activation_max": HARMONIC_ACTIVATION_MAX,
            "harmonic_force_RMSE_max_meV_A": HARMONIC_FORCE_RMSE_MAX_MEV_A,
            "support_LOCO_target_tension_coverage_min": SUPPORT_COVERAGE_MIN,
            "support_each_configuration_coverage_min": SUPPORT_PER_CONFIG_COVERAGE_MIN,
            "thermal_utility_retention_min": THERMAL_UTILITY_RETENTION_MIN,
            "harmful_utility_exposure_max": HARMFUL_UTILITY_EXPOSURE_MAX,
            "thermal_target_tension_coverage_min": THERMAL_TARGET_TENSION_COVERAGE_MIN,
            "thermal_force_RMSE_max_meV_A": THERMAL_FORCE_RMSE_MAX_MEV_A,
            "thermal_force_max_abs_max_meV_A": THERMAL_FORCE_MAX_ABS_MAX_MEV_A,
            "tension_sign_abs_min_meV_A": args.tension_sign_threshold_meV_A,
            "tension_sign_accuracy_min": TENSION_SIGN_ACCURACY_MIN,
            "tension_weighted_R2_min": TENSION_WEIGHTED_R2_MIN,
        },
        "residual_conventions": {
            "delta_target_y": "REF_forces, the correction missing from foundation base plus q6",
            "expert_prediction_p": "frozen depth-3 delta MACE force",
            "router_atom_utility": (
                "non-causal screening proxy ||y||^2 - ||y-p||^2 in (meV/A)^2; "
                "it uses the reference target and cannot itself be a deployed input"
            ),
            "repair_force": "y-p",
            "router_label_uses_dataset_role": False,
            "temperature_smearing_or_supercell_size_is_model_input": False,
        },
        "force_utility": {
            "support": force_utility_group(support),
            "harmonic_train": force_utility_group(harmonic_train),
            "harmonic_validation": force_utility_group(harmonic_validation),
            "fixed_smearing_development_seed0_seed1": force_utility_group(
                fixed_development
            ),
            "fixed_smearing_external_seed2": force_utility_group(fixed_external),
            "legacy_mixed_smearing_R2C": {
                "train": force_utility_group(thermal_train),
                "validation": force_utility_group(thermal_validation),
                "test": force_utility_group(thermal_test),
                "all": force_utility_group(legacy_mixed_smearing),
                "primary_gate_role": "report_only",
                "conditions": legacy_mixed_conditions,
            },
        },
        "leakage_control": {
            "support_split_unit": "complete sscha_index configuration",
            "random_atom_or_bond_split": False,
            "support_validation_duplicate_used": False,
            "fixed_smearing_seed0_seed1_used_for_probe_fit": True,
            "fixed_smearing_seed2_used_for_probe_fit": False,
            "fixed_smearing_seed2_used_for_descriptor_screening": True,
            "fixed_smearing_seed2_is_final_external_validation": False,
            "fixed_E50_vs_historical_R2C_geometry_overlap": fixed_geometry_overlap,
            "legacy_R2C_mixed_smearing_thermal_used_for_fit": False,
            "harmonic_train_used_for_probe_fit": True,
            "harmonic_validation_used_for_fit": False,
            "descriptor_r_max_A": float(descriptor_model.r_max),
            "descriptor_num_interactions": int(descriptor_model.num_interactions),
            "approximate_message_passing_receptive_field_upper_bound_A": approximate_receptive_field,
            "descriptor_interaction_diameter_upper_bound_A": (
                descriptor_interaction_diameter_A
            ),
            "minimum_support_periodic_cell_length_A": (
                minimum_support_periodic_cell_length_A
            ),
            "no_periodic_wrap_by_diameter_bound": (
                representation_has_no_periodic_wrap_by_bound
            ),
            "same_local_perturbation_supercell_invariance": supercell_invariance,
            "score_and_gate_difference_tolerance": 1.0e-3,
            "representation_and_expert_saw_all_support_during_prior_training": True,
            "LOCO_level": f"fixed {router_feature_map} map plus Ridge probe only",
            "warning": (
                "the depth-3 5 A descriptor can span about 15 A and may sense periodic "
                "wrap in the 6x6 support cell; short-cutoff descriptors must be audited in parallel"
            ),
        },
        "force_to_tension_definition": {
            "bonds": f"unique C-C pairs below {args.first_neighbour_cutoff_A:g} A",
            "solver": "numpy.linalg.lstsq minimum-norm central-force projection",
            "lstsq_rcond": args.lstsq_rcond,
            "maximum_support_target_net_force_norm_meV_A": (
                maximum_support_target_net_force_norm_meV_A
            ),
            "minimum_support_target_radial_force_squared_fraction": (
                minimum_support_target_radial_fraction
            ),
            "minimum_support_remaining_radial_force_squared_fraction": (
                minimum_support_remaining_radial_fraction
            ),
            "target_delta_tension_quantiles_meV_A": quantiles(all_target_tension),
            "remaining_repair_tension_quantiles_meV_A": quantiles(all_remaining_tension),
            "per_support_configuration": {
                record.label: {
                    "target_delta": record.target_radial,
                    "remaining_repair": record.remaining_radial,
                }
                for record in support
            },
        },
        "feature_definitions": {
            "feature_mode": args.feature_mode,
            "router_feature_map": router_feature_map,
            "router_feature_map_schema": router_map_schema,
            "router_feature_map_schema_sha256": router_map_schema[
                "schema_sha256"
            ],
            "router_probe_state_sha256": primary_router_state_sha256,
            "screening_router_probe": (
                "per-atom invariants selected by the recorded frozen descriptor-model "
                "node_feats schema; diagnostic coefficients are not a deployable "
                "conservative router"
            ),
            "screening_router_dimension": len(invariant_names),
            "screening_router_unpruned_dimension": int(
                feature_schema["unpruned_output_dimension"]
            ),
            "raw_node_feats_width": node_width,
            "irreps_layout": layout,
            "node_feature_extraction_schema": feature_schema,
            "node_feature_extraction_schema_sha256": feature_schema[
                "schema_sha256"
            ],
            "router_state_dimensions": router_state_dimensions,
            "router_replay_numerics": router_replay_numerics,
            "training_only_feature_pruning": feature_pruning,
            "repair_bond_features": (
                "endpoint mean and squared difference of invariant features plus r and r^2"
            ),
            "controls": {
                "geometry": "six sorted neighbour distances and three first-shell angle cosines",
                "bond_length": "three first-neighbour distances, squares, mean, min, and max",
            },
            "probe": {
                "model": "standardized fixed-alpha ridge",
                "router_feature_map": router_feature_map,
                "router_feature_map_hyperparameter_scanned": False,
                "router_alpha": args.router_ridge_alpha,
                "tension_alpha": args.tension_ridge_alpha,
                "utility_deadband_meV2_A2": args.utility_deadband_meV2_A2,
                "configuration_balanced_weights": True,
            },
        },
        "counts": {
            **observed_counts,
            "fixed_smearing_development": len(fixed_development),
            "fixed_smearing_external": len(fixed_external),
            "support_ids": support_ids,
            "atoms_by_split": {
                "support": sorted({len(item) for item in support_structures}),
                "harmonic_train": sorted({len(item) for item in harmonic_train_structures}),
                "thermal_train": sorted({len(item) for item in thermal_train_structures}),
            },
        },
        "probes": probes,
        "inputs": {
            "expert_model": {"path": str(args.expert_model), "sha256": expert_hash},
            "descriptor_model": {"path": str(descriptor_path), "sha256": descriptor_hash},
            "train_file": {"path": str(args.train_file), "sha256": sha256(args.train_file)},
            "valid_file": {"path": str(args.valid_file), "sha256": sha256(args.valid_file)},
            "test_file": {"path": str(args.test_file), "sha256": sha256(args.test_file)},
            "fixed_thermal_file": {
                "path": str(args.fixed_thermal_file),
                "sha256": sha256(args.fixed_thermal_file),
            },
            "fixed_thermal_target_npz": {
                "path": str(args.fixed_thermal_target_npz),
                "sha256": sha256(args.fixed_thermal_target_npz),
            },
            "fixed_thermal_join": fixed_thermal_provenance,
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "software": {"torch": torch.__version__},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_source = Path(__file__).resolve()
    script_snapshot = args.output_dir / "run_script_snapshot.py"
    shutil.copy2(script_source, script_snapshot)
    script_snapshot_hash = sha256(script_snapshot)
    if script_snapshot_hash != summary["inputs"]["script"]["sha256"]:
        raise RuntimeError("run-script snapshot hash disagrees with the executed script")
    summary["inputs"]["script_snapshot"] = {
        "path": str(script_snapshot),
        "sha256": script_snapshot_hash,
    }
    atomic_json(args.output_dir / "descriptor_separability.json", summary)
    atomic_json(
        args.output_dir / "router_state.json",
        {
            "status": "R2K_noncausal_screening_proxy_not_deployable",
            "deployment_authorized": False,
            "conservative_autograd_validation": "pending_R2L",
            "feature_family": "mace_invariants",
            "run_script_snapshot": {
                "path": str(script_snapshot),
                "sha256": script_snapshot_hash,
            },
            "descriptor_model_sha256": descriptor_hash,
            "expert_model_sha256": expert_hash,
            "train_file_sha256": sha256(args.train_file),
            "valid_file_sha256": sha256(args.valid_file),
            "test_file_sha256": sha256(args.test_file),
            "fixed_thermal_file_sha256": sha256(args.fixed_thermal_file),
            "fixed_thermal_target_npz_sha256": sha256(
                args.fixed_thermal_target_npz
            ),
            "fixed_smearing_development_seeds": [0, 1],
            "fixed_smearing_held_screening_seed": 2,
            "raw_node_feats_width": node_width,
            "feature_mode": args.feature_mode,
            "router_feature_map": router_feature_map,
            "router_feature_map_schema": router_map_schema,
            "router_feature_map_schema_sha256": router_map_schema[
                "schema_sha256"
            ],
            "router_probe_state_sha256": primary_router_state_sha256,
            "router_unpruned_feature_dimension": int(
                feature_schema["unpruned_output_dimension"]
            ),
            "router_feature_dimension": int(feature_schema["output_dimension"]),
            "router_mapped_feature_dimension": router_mapped_dimension,
            "router_state_dimensions": router_state_dimensions,
            "router_replay_numerics": router_replay_numerics,
            "feature_pruning": feature_pruning,
            "feature_pruning_state_sha256": feature_pruning[
                "pruning_state_sha256"
            ],
            "feature_mask": feature_schema["feature_mask"],
            "feature_mask_sha256": feature_schema["feature_mask_sha256"],
            "retained_feature_indices": feature_schema[
                "retained_feature_indices"
            ],
            "node_feature_extraction_schema": feature_schema,
            "node_feature_extraction_schema_sha256": feature_schema[
                "schema_sha256"
            ],
            "scalar_node_feat_slices": [
                [item["start"], item["stop"]]
                for item in layout
                if item["l"] == 0
            ],
            "selected_node_feat_slices": [
                item["raw_slice"] for item in feature_schema["blocks"]
            ],
            "router_feature_definition": (
                "per-atom concatenation of the invariant columns in "
                "node_feature_extraction_schema output-slice order"
            ),
            "torch_graph_replay": {
                "feature_function": "torch_invariant_features",
                "router_feature_map": router_feature_map,
                "router_score_function": "replay_router_score_torch",
                "router_parameters_object": "state",
                "autograd_preserved": True,
            },
            "screening_t0": primary_router["t0"],
            "candidate_t1_not_frozen": primary_router["t1"],
            "transition": (
                "candidate C2 interval only; R2K metrics use score > t0 and do not "
                "differentiate a gated energy; R2L must validate or recalibrate t1"
            ),
            "state": primary_router["state"],
            "utility_deadband_meV2_A2": args.utility_deadband_meV2_A2,
            "screening_checks_satisfied": passed,
            "repair_predictability_checks_satisfied": repair_passed,
        },
    )
    atomic_csv(
        args.output_dir / "loco_fold_metrics.csv",
        list(all_fold_rows[0]),
        all_fold_rows,
    )
    atomic_csv(
        args.output_dir / "support_bond_loco_predictions.csv",
        list(all_bond_rows[0]),
        all_bond_rows,
    )
    atomic_csv(
        args.output_dir / "router_activation_metrics.csv",
        list(all_activation_rows[0]),
        all_activation_rows,
    )
    plot_results(
        args.output_dir, support, probes, all_bond_rows, all_activation_rows
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": decision,
                "screening_proxy_checks": checks,
                "force_utility": summary["force_utility"],
                "primary_LOCO": primary["LOCO"],
                "external_evaluation": external,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
