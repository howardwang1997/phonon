#!/usr/bin/env python3
"""Train-only formal R2R-0 shard, mechanics and aggregation protocol.

This layer is deliberately label-blind and fit-free.  It consumes the frozen
R2R representation through its production APIs, materializes FP64 design
artifacts, and can only issue a representation precheck receipt.  It never
authorizes a readout fit, training, held-data access, or deployment.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import graphene_r2o_taylor_null as r2o  # noqa: E402
import graphene_r2r_multipolar_background as r2r  # noqa: E402
import evaluate_graphene_r2o_taylor_null as r2o_evaluator  # noqa: E402


FORMAT = "graphene_r2r0_formal_train_only_v6_attempt3_recursive_schema"
SHARD_FORMAT = "graphene_r2r0_formal_shard_v3_attempt3_recursive_schema"
MECHANICS_FORMAT = "graphene_r2r0_formal_mechanics_v5_attempt3_recursive_schema"
AGGREGATE_FORMAT = "graphene_r2r0_formal_aggregate_v4_attempt3_recursive_schema"
PREFLIGHT_FORMAT = "graphene_r2r0_bounded_preflight_v3_recursive_schema"
STATUS_SHARD = "R2R0_FORMAL_SHARD_COMPLETE"
STATUS_MECHANICS = "R2R0_FORMAL_MECHANICS_COMPLETE"
STATUS_AGGREGATE_PASS = "R2R0_FORMAL_REPRESENTATION_PRECHECK_PASSED"
STATUS_AGGREGATE_FAIL = "R2R0_FORMAL_REPRESENTATION_PRECHECK_FAILED"
STATUS_AGGREGATE_INCONCLUSIVE = "R2R0_FORMAL_NUMERICALLY_INCONCLUSIVE"
STATUS_PREFLIGHT = "R2R0_BOUNDED_PREFLIGHT_COMPLETE"
BORDERLINE_RELATIVE_BAND = 0.01
FREEZE_MANIFEST_FORMAT = "graphene_r2r0_formal_freeze_manifest_v1"
FD_STEPS_A = (8.0e-4, 4.0e-4, 2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5)
FD_ADJUDICATING_STEP_A = 5.0e-5
Q_MAX_ABS_TOLERANCE = 5.0e-13
Q_ROUND_DECIMALS = 12
Q_DISCRETE_ARRAY_NAMES = (
    "receiver",
    "sender",
    "image_integer",
    "local_source_order_offsets",
    "local_source_order_sender",
    "local_source_order_edge_index",
)
Q_FLOAT_ARRAY_NAMES = (
    "reference_distance_A",
    "weight",
    "normalization",
    "center_q",
    "neighbor_q",
)

FROZEN_R2R_MODULE_SHA256 = (
    "4db3665f945783086de5c7a2f6262843bbe7d03df2a32f9153e450200b25064c"
)
FROZEN_R2R_TEST_SHA256 = (
    "1255c5720cb0fe575fe7edd9d211035799feccebd922488bc33ebe20f7c36c08"
)
FROZEN_R2R_CANONICAL_SHA256 = (
    "e589497c7b9f6a9d4ff0cdcc434cf07804fa0fd732d791dba811d4c7e1c5aecc"
)
FROZEN_R2R_DOC_SHA256 = (
    "003f4c3e949e6432b56a350541ca15e7bde48749ad2150129e8ea227cf5c56c9"
)
FROZEN_R2O_EVALUATOR_SHA256 = (
    "14cf2a160a9b0281c0c2149caec6d62f9a31e40a9c7af39ea3ea2e14956c79ce"
)
FROZEN_R2O_WRAPPER_SHA256 = (
    "0eedea5a59b8f717559feda97b2c2956dd6f274fe4963de10c214db53b4610e2"
)

FROZEN_SOURCE_PATHS = {
    "R2R_module": ROOT / "scripts/smearing_kink/graphene_r2r_multipolar_background.py",
    "R2R_tests": ROOT / "tests/test_graphene_r2r_multipolar_background.py",
    "R2R_contract_doc": ROOT / "docs/GRAPHENE_R2R_MULTIPOLAR_BACKGROUND_CONTRACT_2026-08-25.md",
    "R2O_evaluator": ROOT / "scripts/smearing_kink/evaluate_graphene_r2o_taylor_null.py",
    "R2O_wrapper": ROOT / "scripts/smearing_kink/graphene_r2o_taylor_null.py",
}
FROZEN_SOURCE_SHA256 = {
    "R2R_module": FROZEN_R2R_MODULE_SHA256,
    "R2R_tests": FROZEN_R2R_TEST_SHA256,
    "R2R_contract_doc": FROZEN_R2R_DOC_SHA256,
    "R2O_evaluator": FROZEN_R2O_EVALUATOR_SHA256,
    "R2O_wrapper": FROZEN_R2O_WRAPPER_SHA256,
}

FORMAL_SOURCE_PATHS = {
    "formal_core": ROOT / "scripts/smearing_kink/graphene_r2r0_formal.py",
    "shard_cli": ROOT / "scripts/smearing_kink/run_graphene_r2r0_formal.py",
    "aggregate_cli": ROOT / "scripts/smearing_kink/aggregate_graphene_r2r0_formal.py",
    "launcher": ROOT / "scripts/smearing_kink/launch_graphene_r2r0_formal.py",
    "tests": ROOT / "tests/test_graphene_r2r0_formal.py",
    "contract_doc": ROOT / "docs/GRAPHENE_R2R0_FORMAL_CONTRACT_2026-08-25.md",
}

SHARD_INDICES = {
    shard: {
        "thermal": tuple(range(shard, 92, 3)),
        "harmonic": tuple(range(shard, 32, 3)),
    }
    for shard in range(3)
}
SENTINEL_GLOBAL_INDICES = (0, 20, 56, 91)
THERMAL_CONFIG_SEGMENTS = (
    (0, 20, "r2o_exact_e50_seed0_train"),
    (20, 56, "r2o_auxiliary_T300_train"),
    (56, 92, "r2o_auxiliary_T600_train"),
)
HARMONIC_CONFIG_TYPE = "r2o_harmonic_lambda1_small_zero_train"

FORBIDDEN_PATH_TOKENS = tuple(
    sorted(set(r2r.FORBIDDEN_PATH_TOKENS) | {"seed1", "seed2", "support"})
)
GEOMETRY_LOADER_ISOLATION = {
    "header_label_values_retained": False,
    "numeric_label_columns_converted": False,
    "force_labels_parsed": False,
    "energy_labels_parsed": False,
    "force_labels_loaded": False,
    "energy_labels_loaded": False,
    "force_labels_used": False,
    "energy_labels_used": False,
}
AGGREGATE_SAFETY_FIELDS = {
    "force_labels_used": False,
    "energy_labels_used": False,
    "fit_performed": False,
    "can_authorize_fit_or_training": False,
    "geometry_loader_isolation": GEOMETRY_LOADER_ISOLATION,
    "held_or_support_access": False,
    "seed1_or_seed2_access": False,
}
OPTIONAL_SAFETY_FALSE_FIELDS = (
    "fit_performed",
    "held_or_support_access",
    "held_data_access",
    "support_data_access",
    "seed1_access",
    "seed2_access",
    "seed1_or_seed2_access",
)
RECEIPT_TOP_LEVEL_KEYS = {
    "preflight": frozenset(
        {
            "format",
            "status",
            "kind",
            "formal_contract_sha256",
            "frozen_source_sha256",
            "input_sha256",
            "runtime_fingerprint",
            "payload",
            "pass",
            "elapsed_seconds",
            "force_labels_used",
            "energy_labels_used",
            "geometry_loader_isolation",
            "can_authorize_fit_or_training",
            "can_authorize_remote_launch",
        }
    ),
    "shard": frozenset(
        {
            "format",
            "status",
            "formal_contract_sha256",
            "execution_authorization",
            "frozen_source_sha256",
            "frozen_R2R_canonical_sha256",
            "input_sha256",
            "runtime_fingerprint",
            "shard_id",
            "partition",
            "thermal_count",
            "harmonic_count",
            "thermal_receipts",
            "harmonic_receipts",
            "sentinel_receipts",
            "sentinel_included_in_matrix",
            "sentinel_contract_sha256",
            "array_artifacts",
            "elapsed_seconds",
            "force_labels_used",
            "energy_labels_used",
            "geometry_loader_isolation",
            "can_authorize_fit_or_training",
        }
    ),
    "mechanics": frozenset(
        {
            "format",
            "status",
            "formal_contract_sha256",
            "execution_authorization",
            "frozen_source_sha256",
            "input_sha256",
            "runtime_fingerprint",
            "full_H_stage_receipts",
            "cuda_peak_allocated_bytes",
            "dtype",
            "mechanics",
            "mechanics_pass",
            "arrays_sha256",
            "array_schema",
            "elapsed_seconds",
            "force_labels_used",
            "energy_labels_used",
            "geometry_loader_isolation",
            "can_authorize_fit_or_training",
        }
    ),
    "aggregate": frozenset(
        {
            "format",
            "status",
            "formal_contract_sha256",
            "execution_authorization",
            "frozen_source_sha256",
            "frozen_R2R_canonical_sha256",
            "shard_receipt_sha256",
            "shard_root_manifest_sha256",
            "input_sha256",
            "mechanics_receipt_sha256",
            "mechanics_root_manifest_sha256",
            "array_schema",
            "arrays_sha256",
            *AGGREGATE_SAFETY_FIELDS,
            "sentinel",
            "shard_environment_semantic_sha256",
            "environment_mismatch",
            "environment_mismatch_merge_allowed",
            "mechanics_recomputed_gate",
            "train_geometry_and_rank1_gate",
            "rank_condition_precheck",
            "thermal_partition_exact",
            "harmonic_partition_exact",
            "configuration_identity",
            "representation_precheck_pass",
            "numerically_inconclusive",
            "borderline",
            "formal_total_GPU_hours",
            "aggregate_recomputation_format",
        }
    ),
}
RECEIPT_SAFETY_EXPECTED = {
    "preflight": {
        "force_labels_used": False,
        "energy_labels_used": False,
        "geometry_loader_isolation": GEOMETRY_LOADER_ISOLATION,
        "can_authorize_fit_or_training": False,
        "can_authorize_remote_launch": False,
    },
    "shard": {
        "force_labels_used": False,
        "energy_labels_used": False,
        "geometry_loader_isolation": GEOMETRY_LOADER_ISOLATION,
        "can_authorize_fit_or_training": False,
    },
    "mechanics": {
        "force_labels_used": False,
        "energy_labels_used": False,
        "geometry_loader_isolation": GEOMETRY_LOADER_ISOLATION,
        "can_authorize_fit_or_training": False,
    },
    "aggregate": AGGREGATE_SAFETY_FIELDS,
}
RECEIPT_FORMAT_AND_STATUS = {
    "preflight": (PREFLIGHT_FORMAT, frozenset({STATUS_PREFLIGHT})),
    "shard": (SHARD_FORMAT, frozenset({STATUS_SHARD})),
    "mechanics": (MECHANICS_FORMAT, frozenset({STATUS_MECHANICS})),
    "aggregate": (
        AGGREGATE_FORMAT,
        frozenset(
            {
                STATUS_AGGREGATE_PASS,
                STATUS_AGGREGATE_FAIL,
                STATUS_AGGREGATE_INCONCLUSIVE,
            }
        ),
    ),
}
SUSPICIOUS_SAFETY_COMPACT_FRAGMENTS = (
    "fit",
    "train",
    "label",
    "held",
    "holdout",
    "support",
    "seed1",
    "seed2",
    "targetforce",
    "targetenergy",
)
CORE_NESTED_SAFETY_FALSE_PATHS = {
    "preflight": frozenset(
        {
            ("payload", "thermal_receipt", "force_labels_used"),
            ("payload", "thermal_receipt", "energy_labels_used"),
            ("payload", "harmonic_receipt", "force_labels_used"),
            ("payload", "harmonic_receipt", "energy_labels_used"),
        }
    ),
    "shard": frozenset(
        {
            ("thermal_receipts", "*", "force_labels_used"),
            ("thermal_receipts", "*", "energy_labels_used"),
            ("harmonic_receipts", "*", "force_labels_used"),
            ("harmonic_receipts", "*", "energy_labels_used"),
            ("sentinel_receipts", "*", "force_labels_used"),
            ("sentinel_receipts", "*", "energy_labels_used"),
        }
    ),
    "mechanics": frozenset(
        {
            ("mechanics", "node_parity", "*", "force_labels_used"),
        }
    ),
    "aggregate": frozenset(
        {
            ("rank_condition_precheck", "can_authorize_fit_or_training"),
        }
    ),
}
CORE_BENIGN_SCIENTIFIC_PATHS = {
    "preflight": frozenset(
        {
            ("payload", "train_gate"),
            ("payload", "training_count"),
            ("payload", "force_design_rank"),
        }
    ),
    "shard": frozenset(
        {
            ("thermal_receipts", "*", "train_gate"),
            ("thermal_receipts", "*", "training_count"),
            ("thermal_receipts", "*", "force_design_rank"),
        }
    ),
    "mechanics": frozenset(
        {
            ("mechanics", "train_gate"),
            ("mechanics", "training_count"),
            ("mechanics", "force_design_rank"),
        }
    ),
    "aggregate": frozenset(
        {
            ("train_geometry_and_rank1_gate",),
            ("rank_condition_precheck", "train_gate"),
            ("rank_condition_precheck", "training_count"),
            ("rank_condition_precheck", "force_design_rank"),
        }
    ),
}


def _key_path_matches(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    return len(path) == len(pattern) and all(
        expected == "*" or observed == expected
        for observed, expected in zip(path, pattern)
    )


def _safety_key_forms(key: str) -> tuple[tuple[str, ...], str]:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", camel_split)
    tokens = tuple(
        token
        for token in re.split(r"[^a-z0-9]+", camel_split.lower())
        if token
    )
    return tokens, "".join(tokens)


def _is_suspicious_safety_key(key: str) -> bool:
    tokens, compact = _safety_key_forms(key)
    token_set = set(tokens)
    force_or_energy_action = bool(
        token_set.intersection({"force", "forces", "energy", "energies"})
        and token_set.intersection(
            {
                "access",
                "accessed",
                "consumed",
                "loaded",
                "opened",
                "parsed",
                "present",
                "read",
                "used",
            }
        )
    )
    return bool(
        any(fragment in compact for fragment in SUSPICIOUS_SAFETY_COMPACT_FRAGMENTS)
        or force_or_energy_action
    )


RECURSIVE_RECEIPT_SCHEMA_SHA256 = {
    "preflight:synthetic": (
        "ea0aa220598dc4c6517c70fcf9d75fd53aa3fe31a172719db1bc9cad8375e394"
    ),
    "preflight:real": (
        "6f19a3efdb9793a54cc56ef42b0f0a9db1281261d0850b8c1d7a27d07c69cc07"
    ),
    "shard": (
        "673acd629b9d7f6df15ab4a26a290674b3eebfcb0c03b3fbc900677a6db2f23e"
    ),
    "mechanics": (
        "a87b3d99c9c83c5436b425431efceba51bdf021a3e7e0daababd596e806f39fa"
    ),
    "aggregate": (
        "c64f92d287f98564865512943ff2341e703e0388a5ddf65f7a909ac4f8b75aaa"
    ),
}
RECURSIVE_RECEIPT_SCHEMA_PATH_COUNT = {
    "preflight:synthetic": 77,
    "preflight:real": 266,
    "shard": 654,
    "mechanics": 1815,
    "aggregate": 455,
}


def recursive_key_path_schema(
    value: Any,
    *,
    wildcard_mapping_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> tuple[tuple[str, ...], ...]:
    """Return a complete normalized dict/list key-path schema.

    Every dictionary key is represented explicitly. List elements share one
    ``[]`` path and must have an identical container/key schema. Selected
    externally controlled mappings may normalize their dynamic keys to ``{}``;
    their exact keysets are validated by the corresponding artifact inventory.
    Scalar values intentionally do not enter this structural schema.
    """

    def visit(item: Any, path: tuple[str, ...]) -> set[tuple[str, ...]]:
        paths: set[tuple[str, ...]] = set()
        if isinstance(item, Mapping):
            paths.add(("D", *path))
            if path in wildcard_mapping_paths:
                child_schemas = []
                for child in item.values():
                    child_schemas.append(visit(child, (*path, "{}")))
                if child_schemas and any(
                    schema != child_schemas[0] for schema in child_schemas[1:]
                ):
                    raise ValueError(
                        "controlled mapping values have inconsistent schema at "
                        + ".".join(path)
                    )
                paths.add(("K", *path, "{}"))
                if child_schemas:
                    paths.update(child_schemas[0])
                return paths
            for raw_key, child in item.items():
                key = str(raw_key)
                child_path = (*path, key)
                paths.add(("K", *child_path))
                paths.update(visit(child, child_path))
            return paths
        if isinstance(item, (list, tuple)):
            paths.add(("L", *path))
            child_schemas = [visit(child, (*path, "[]")) for child in item]
            if child_schemas and any(
                schema != child_schemas[0] for schema in child_schemas[1:]
            ):
                raise ValueError(
                    "list elements have inconsistent recursive schema at "
                    + ".".join(path)
                )
            if child_schemas:
                paths.update(child_schemas[0])
        return paths

    return tuple(sorted(visit(value, ())))


def recursive_key_path_schema_sha256(
    value: Any,
    *,
    wildcard_mapping_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> str:
    return r2r.semantic_sha256(
        recursive_key_path_schema(
            value, wildcard_mapping_paths=wildcard_mapping_paths
        )
    )


def _recursive_schema_variant(receipt: Mapping[str, Any], receipt_kind: str) -> str:
    if receipt_kind == "preflight":
        kind = receipt.get("kind")
        if kind not in {"synthetic", "real"}:
            raise ValueError("bounded preflight recursive schema kind changed")
        return f"preflight:{kind}"
    return receipt_kind


def _validate_recursive_receipt_schema(
    receipt: Mapping[str, Any], label: str, *, receipt_kind: str
) -> None:
    variant = _recursive_schema_variant(receipt, receipt_kind)
    observed_schema = recursive_key_path_schema(receipt)
    observed_count = len(observed_schema)
    expected_count = RECURSIVE_RECEIPT_SCHEMA_PATH_COUNT[variant]
    if observed_count != expected_count:
        raise ValueError(
            f"{label} recursive key-path schema count changed: "
            f"observed={observed_count}, expected={expected_count}"
        )
    observed = r2r.semantic_sha256(observed_schema)
    expected = RECURSIVE_RECEIPT_SCHEMA_SHA256[variant]
    if observed != expected:
        raise ValueError(
            f"{label} recursive key-path schema changed: "
            f"observed={observed}, expected={expected}"
        )

SENTINEL_CONTRACT = {
    "format": "graphene_r2r0_cross_hardware_sentinel_v1",
    "probe": "thermal92 global indices 0,20,56,91, each evaluated independently by every shard",
    "comparison": "symmetric abs(diff) <= atol + 5e-7*max(abs(left),abs(right)) elementwise",
    "rtol": 5.0e-7,
    "arrays": {
        "fixed_energy_eV": {"atol": 1.0e-9},
        "parameter_energy_design_eV": {"atol": 1.0e-9},
        "fixed_force_eV_A": {"atol": 1.0e-8},
        "parameter_force_design_eV_A": {"atol": 1.0e-8},
        "b_A4": {"atol": 1.0e-18},
        "a": {"atol": 1.0e-12},
        "c": {"atol": 1.0e-12},
    },
    "combined_probe": {
        "energy_abs_eV": 1.0e-6,
        "force_max_abs_eV_A": 1.0e-5,
    },
    "exact_metadata": (
        "shape,dtype,column/input/reference/endpoint-state/formal-graph/"
        "coefficient hashes and structure identity"
    ),
    "role": "hardware reproducibility only; never replaces canonical mechanics gates",
}
SENTINEL_CONTRACT_SHA256 = r2r.semantic_sha256(SENTINEL_CONTRACT)

FORMAL_CONTRACT = {
    "format": FORMAT,
    "frozen_R2R_module_sha256": FROZEN_R2R_MODULE_SHA256,
    "frozen_R2R_tests_sha256": FROZEN_R2R_TEST_SHA256,
    "frozen_R2R_canonical_sha256": FROZEN_R2R_CANONICAL_SHA256,
    "frozen_R2R_doc_sha256": FROZEN_R2R_DOC_SHA256,
    "frozen_R2O_wrapper_sha256": FROZEN_R2O_WRAPPER_SHA256,
    "endpoint_state_sha256": r2r.R2Q_ENDPOINT_STATE_SHA256,
    "input_sha256": r2r.EXPECTED_INPUTS,
    "shard_indices_modulo_3": SHARD_INDICES,
    "sentinel_global_indices": SENTINEL_GLOBAL_INDICES,
    "localized_reference_site_mapping": {
        "definition": "(atomic_number, round(reference MIC(r_i-r_0),7 decimals))",
        "common_6x6_to_8x8": 72,
        "only_6x6": 0,
        "only_8x8": 56,
        "semantic_sha256": (
            "5f12c337580f80428235644b196d703f7aa8affb426beec3f962d7e6861230c1"
        ),
    },
    "thermal_config_segments": THERMAL_CONFIG_SEGMENTS,
    "harmonic_config_type": HARMONIC_CONFIG_TYPE,
    "sentinel_contract": SENTINEL_CONTRACT,
    "force_labels_used": False,
    "energy_labels_used": False,
    "fit_or_training": False,
    "held_or_support_access": False,
    "remote_launch_authorized": False,
    "geometry_loader_isolation": GEOMETRY_LOADER_ISOLATION,
    "combined_EF_adapter_v5": (
        "production_combined_energy_force is adapted exactly once to a float energy, "
        "FP64 source-order force ndarray, and its unmodified query receipt; direct mock "
        "and real thermal0/O3 regression tests bind the tuple-return control flow"
    ),
    "O3_rigid_transform_probe_v4": {
        "baseline_structure_template": "thermal92 global index 0",
        "baseline_structure_full_semantic_sha256": (
            "d3f6eca52753a6407c58d107374e06e80cec470c3668c1c6f1f034ea59488766"
        ),
        "baseline_reference_full_semantic_sha256": (
            "39cf74c68a5ce629ab708ce9158f49d72a7e955895d5b01fa10c2883caa861e7"
        ),
        "formal_graph": (
            "verify frozen baseline graph, rotate positions/shifts/cell by Q.T, "
            "keep other formal graph tensors byte exact"
        ),
        "background": (
            "reuse exact baseline ReferenceNeighborhood topology/distances/weights/"
            "normalization; live centered q vectors rotate with structure/reference"
        ),
        "public_input_covariance": (
            "reference/live structure inputs must be array-exact to baseline @ Q.T; "
            "internal ordered/adapted arrays use the canonical named 1e-12 A bound"
        ),
        "native_multiset_veto": (
            "exact physical edge identity-key sets only; all native numeric differences "
            "are diagnostic and never gate physics"
        ),
        "native_rebuild": "diagnostic only, never the O3 physics gate",
    },
    "attempt3_R2R0D_prerequisite": {
        "status": "ATTRIBUTED_BOTH",
        "diagnostic_contract_sha256": (
            "bfe73de26b346527b3c2ac7d17c50b8abf9535417fbcff48520ba03c4946204f"
        ),
        "diagnostic_receipt_sha256": (
            "428c999351ef9698de53283754c11fbfafb7d998bbb0ab1f6f06714e21ac62b9"
        ),
        "diagnostic_arrays_sha256": (
            "db7e93ec00bac4401f447d560683890fd46240c06057883aa7db4c62265b7f75"
        ),
        "FD_TRUNCATION_CONFIRMED": True,
        "h_5e-5_energy_force_error_eV_A": 4.6717795265660556e-7,
        "h_5e-5_force_Hessian_error_eV_A2": 3.4242256781169544e-6,
        "Q_FLOAT_SERIALIZATION_ONLY": True,
        "role": "completed diagnostic attribution; never a runtime input or formal PASS",
    },
    "attempt3_finite_difference_v1": {
        "coordinate_zero_based": [2, 1],
        "steps_A_in_fixed_order": list(FD_STEPS_A),
        "evaluation_order_each_step": ["minus", "plus"],
        "adjudicating_step_A": FD_ADJUDICATING_STEP_A,
        "fixed_gate_source": "frozen R2R CANONICAL_CONTRACT fixed_gates finite_difference",
        "force_abs_eV_A_unchanged": 1.0e-5,
        "force_Hessian_abs_eV_A2_unchanged": 1.0e-5,
        "multistep_orders_jump_Richardson_and_energy_path_role": (
            "recorded and recomputed report-only diagnostics; no post-result step selection"
        ),
        "topology_provenance": "all base/minus/plus formal graph and assignment hashes exact",
    },
    "attempt3_q_portability_v1": {
        "references": {
            "reference_6x6": {
                "atom_count": 72,
                "edge_count": 2808,
                "local_source_order_semantic_sha256": (
                    "4f0fcf5acf9669588bf7438fdb9eaf0b00dcff5c62c528b8d7637ffac3b744db"
                ),
                "round12_payload_sha256": (
                    "480b1ae1aa52b9b9610840b664775428de2c78dd5e9121116a19da02c29fe4d3"
                ),
                "canonical_CPU_discrete_raw_sha256": {
                    "receiver": "6b96d7df45faf8252a3a28306af27e286a3e5fefea61297ab86c1b0b6bbeb933",
                    "sender": "6be269e57dbcd07e3c89f0260f6e7d945b78f87631172621b8e2935f166ad89f",
                    "image_integer": "3274f6dde6f2c3dcbc7f1d38f53beeb81f6efff63b2d34c1013b312a67348477",
                    "local_source_order_offsets": "5361f6ed771343530466ca1de5aafe925f17f4211c54e6936c47aacc848e14de",
                    "local_source_order_sender": "6be269e57dbcd07e3c89f0260f6e7d945b78f87631172621b8e2935f166ad89f",
                    "local_source_order_edge_index": "087b9da5d1ac093b1616db39f41bcf70120bbb2bc3a9139a89b9159626f4dcef",
                },
            },
            "reference_8x8": {
                "atom_count": 128,
                "edge_count": 4992,
                "local_source_order_semantic_sha256": (
                    "6e44497aa2b55fc433c8155ce08f9f9b2695d9adcf9b298046dfd291958ecc56"
                ),
                "round12_payload_sha256": (
                    "65a015a8d60edde0cf6eaeb12df358e9f31cf7ce72a5e4803a0a67ea35c733cc"
                ),
                "canonical_CPU_discrete_raw_sha256": {
                    "receiver": "46c3e7e4abcb87372b8da5b719be0fe887a898f1f33b38eb127c5177b4af6f99",
                    "sender": "7f5884aec419cb54c16ca081750c169be392547888be1e38c5a80dec32cf1056",
                    "image_integer": "d68c76bfa40942ccd6f0839a0a6a994f71f8bc5d60cb2078529a09827118920a",
                    "local_source_order_offsets": "bcc14236fcc4c55d87ef33beaf510dcd8dc00f8588f75c0855b16d0566b46069",
                    "local_source_order_sender": "7f5884aec419cb54c16ca081750c169be392547888be1e38c5a80dec32cf1056",
                    "local_source_order_edge_index": "1a4a72ba31c4c822097535fe4843c146faf5660d1a5c94688157d7fd3aa28b59",
                },
            },
        },
        "devices": ["CPU", "CUDA"],
        "discrete_topology_and_source_order": "array exact and canonical CPU hash bound",
        "normalized_q_max_abs": Q_MAX_ABS_TOLERANCE,
        "round_decimals": Q_ROUND_DECIMALS,
        "round_payload": (
            "concat(center_q node order, neighbor_q production edge order); "
            "np.round then contiguous little-endian FP64 bytes"
        ),
        "physical_zero_jet": (
            "CPU and CUDA value/J/full-H exact zero and finite on both references; "
            "nonzero local-to-production value and gradient parity recomputed from "
            "raw actual-device local/global values and gradients; stored differences "
            "and pass flags are diagnostic only"
        ),
        "raw_q_orbit_hash_role": "diagnostic_only_not_a_gate",
    },
    "attempt3_fresh_execution": {
        "attempt": 3,
        "fresh_roles": ["shard0", "shard1", "shard2", "mechanics"],
        "attempt2_artifact_reuse": False,
        "successful_command_count": 220,
        "successful_log_file_count": 660,
        "stdout_and_stderr_required_for_every_command": True,
    },
    "execution_authorization": (
        "external freeze_manifest.json plus separate marker containing its exact "
        "SHA-256; implementation stage creates no GO marker"
    ),
    "aggregate_policy": {
        "fresh_and_DONE_recovery": (
            "one shared function rebuilds canonical aggregate arrays, sentinel, train "
            "gate, SVD rank, borderline classification, mechanics gate and final status"
        ),
        "recovery_array_binding": (
            "every completed aggregate array is dtype/shape/value exact to the current "
            "three shard arrays before recovery"
        ),
        "stored_decision_fields": "must equal shared recomputation; never authoritative",
        "recovery_safety_fields": (
            "preflight, shard, mechanics and aggregate fresh/completed receipts first "
            "require their complete frozen recursive key-path schema/count/hash; exact "
            "top-level and geometry schemas plus exact Boolean false no-label/no-fit/"
            "no-held fields remain independent semantic checks; the alias scanner is "
            "defense in depth only"
        ),
    },
    "receipt_recursive_schema_v3": {
        "formats": {
            "preflight": PREFLIGHT_FORMAT,
            "shard": SHARD_FORMAT,
            "mechanics": MECHANICS_FORMAT,
            "aggregate": AGGREGATE_FORMAT,
        },
        "top_level_keys": {
            kind: sorted(keys) for kind, keys in RECEIPT_TOP_LEVEL_KEYS.items()
        },
        "path_encoding": {
            "dictionary_container": "D + full parent path",
            "dictionary_key": "K + full key path",
            "list_container": "L + full parent path",
            "list_element": "[] normalized path component",
            "scalar_values": "not part of structural schema",
        },
        "list_element_schema": "every element must have an identical recursive schema",
        "core_schema_path_count": RECURSIVE_RECEIPT_SCHEMA_PATH_COUNT,
        "core_schema_semantic_sha256": RECURSIVE_RECEIPT_SCHEMA_SHA256,
        "core_wildcard_mapping_paths": [],
        "validation_order": (
            "complete recursive schema count/hash first; exact top-level/format/status/"
            "safety semantics second"
        ),
        "geometry_loader_isolation": GEOMETRY_LOADER_ISOLATION,
        "nested_exact_false_paths": {
            kind: sorted(".".join(path) for path in paths)
            for kind, paths in CORE_NESTED_SAFETY_FALSE_PATHS.items()
        },
        "benign_scientific_paths": {
            kind: sorted(".".join(path) for path in paths)
            for kind, paths in CORE_BENIGN_SCIENTIFIC_PATHS.items()
        },
        "suspicious_compact_fragments": SUSPICIOUS_SAFETY_COMPACT_FRAGMENTS,
        "unexpected_safety_alias_policy": (
            "split camelCase, lowercase and compact keys; fail closed on every "
            "undeclared fit/train/label/held/holdout/support/seed substring or "
            "force/energy target-use alias; defense in depth after structural validation"
        ),
    },
    "cross_environment_merge_policy": (
        "environment package snapshots may differ only when every one of the four "
        "cross-node sentinels passes exact metadata plus all elementwise tolerances; "
        "otherwise aggregate is numerically inconclusive"
    ),
    "failure_policy": "diagnostic only; failure never authorizes reduced design or fit",
    "formal_workload": {
        "unique_thermal": 92,
        "harmonic_zero": 32,
        "cross_hardware_sentinel_evaluations": 12,
        "mechanics_runs": 1,
        "expected_total_GPU_hours_max": 0.75,
        "over_expected_policy": "record elapsed/peak and continue for manual review; never reduce precision, workload, or gates",
    },
    "borderline_policy": {
        "relative_band_around_fixed_threshold": BORDERLINE_RELATIVE_BAND,
        "scaled_smallest_singular_over_rank_tolerance_min": 10.0,
        "resolution": (
            "rerun the complete 92-configuration design on V100-A; any mismatch "
            "or unresolved threshold proximity remains INCONCLUSIVE"
        ),
        "status": STATUS_AGGREGATE_INCONCLUSIVE,
    },
}
FORMAL_CONTRACT_SHA256 = r2r.semantic_sha256(FORMAL_CONTRACT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_path(path: Path, label: str, *, must_exist: bool = True) -> Path:
    value = Path(path)
    if ".." in PurePath(value).parts:
        raise ValueError(f"path traversal is forbidden in {label}")
    lowered = [part.casefold() for part in PurePath(value).parts]
    for token in FORBIDDEN_PATH_TOKENS:
        folded = token.casefold()
        if any(folded in part for part in lowered):
            raise ValueError(f"forbidden R2R-0 path token {token!r} in {label}")
    expanded = value.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    # Check every component before resolve: a symlinked parent is as capable of
    # redirecting provenance as a symlinked leaf.
    probe = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        probe = probe / part
        if probe.is_symlink():
            raise ValueError(f"symlink path component is forbidden in {label}")
    resolved = expanded.resolve(strict=must_exist)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def validate_frozen_sources() -> dict[str, str]:
    observed = {name: sha256(path) for name, path in FROZEN_SOURCE_PATHS.items()}
    if observed != FROZEN_SOURCE_SHA256:
        raise ValueError(f"frozen R2R source/doc/test hashes changed: {observed}")
    if r2r.CANONICAL_CONTRACT_SHA256 != FROZEN_R2R_CANONICAL_SHA256:
        raise ValueError("live R2R canonical semantic SHA changed")
    return observed


def expected_input_sha256() -> dict[str, str]:
    return {role: record["sha256"] for role, record in r2r.EXPECTED_INPUTS.items()}


def prepare_freeze_manifest(output: Path) -> dict[str, Any]:
    """Create the external candidate manifest; never create an authorization marker."""
    destination = _reject_path(output, "freeze manifest output", must_exist=False)
    if destination.exists():
        raise FileExistsError("freeze manifest output must be fresh")
    payload = {
        "format": FREEZE_MANIFEST_FORMAT,
        "formal_contract_sha256": FORMAL_CONTRACT_SHA256,
        "formal_source_sha256": {
            name: sha256(path) for name, path in FORMAL_SOURCE_PATHS.items()
        },
        "frozen_primitive_sha256": validate_frozen_sources(),
        "input_sha256": expected_input_sha256(),
        "authorization_marker_created": False,
        "can_authorize_fit_or_training": False,
    }
    _atomic_json(destination, payload)
    return payload


def validate_execution_authorization(
    freeze_manifest: Path, authorization_marker: Path
) -> dict[str, Any]:
    manifest_path = _reject_path(freeze_manifest, "freeze manifest")
    marker_path = _reject_path(authorization_marker, "authorization marker")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != FREEZE_MANIFEST_FORMAT:
        raise ValueError("freeze manifest format changed")
    if payload.get("formal_contract_sha256") != FORMAL_CONTRACT_SHA256:
        raise ValueError("freeze manifest formal contract changed")
    observed_sources = {name: sha256(path) for name, path in FORMAL_SOURCE_PATHS.items()}
    if payload.get("formal_source_sha256") != observed_sources:
        raise ValueError("freeze manifest does not bind current formal sources")
    if payload.get("frozen_primitive_sha256") != validate_frozen_sources():
        raise ValueError("freeze manifest primitive sources changed")
    if payload.get("input_sha256") != expected_input_sha256():
        raise ValueError("freeze manifest input hashes changed")
    manifest_sha = sha256(manifest_path)
    if marker_path.read_bytes() != (manifest_sha + "\n").encode("ascii"):
        raise ValueError("authorization marker does not bind exact freeze manifest")
    return {
        "freeze_manifest_sha256": manifest_sha,
        "authorization_marker_sha256": sha256(marker_path),
        "formal_source_sha256": observed_sources,
    }


def runtime_fingerprint(device: str) -> dict[str, Any]:
    packages = {}
    for name in ("ase", "mace-torch", "numpy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    conda_records = []
    conda_meta = Path(sys.prefix) / "conda-meta"
    if conda_meta.is_dir():
        for record_path in sorted(conda_meta.glob("*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            conda_records.append(
                {
                    key: record.get(key)
                    for key in ("name", "version", "build", "channel")
                }
            )
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": packages,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "default_dtype": str(torch.get_default_dtype()),
        "device_request": str(device),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": os.environ.get(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        ),
        "cuda_available": torch.cuda.is_available(),
        "conda_prefix_basename": Path(sys.prefix).name,
        "conda_package_records": conda_records,
        "conda_package_records_sha256": r2r.semantic_sha256(conda_records),
    }
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA formal shard requested but CUDA is unavailable")
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            timeout=20,
        )
        driver_lines = [line.strip() for line in driver.stdout.decode("ascii", errors="strict").splitlines() if line.strip()]
        if driver.returncode != 0 or len(driver_lines) != 1:
            raise ValueError("formal CUDA runtime requires one exact nvidia-smi driver value")
        index = torch.cuda.current_device()
        payload["cuda"] = {
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "driver_version": driver_lines[0],
            "driver_query_stdout_sha256": hashlib.sha256(driver.stdout).hexdigest(),
        }
    payload["semantic_sha256"] = r2r.semantic_sha256(payload)
    return payload


def _atomic_bytes(path: Path, content: bytes) -> None:
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else f"nonfinite:{number!r}"
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(
        _json_safe(dict(payload)), sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, content)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("xb") as handle:
        np.savez(handle, **{key: np.asarray(value) for key, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fresh_run_root(path: Path) -> Path:
    root = _reject_path(path, "formal output", must_exist=False)
    root.mkdir(parents=True, exist_ok=True)
    occupied = [name for name in ("RUNNING", "FAILED", "DONE", "EXIT_CODE") if (root/name).exists()]
    if occupied:
        raise FileExistsError(f"formal output has terminal/in-progress markers: {occupied}")
    if any(root.iterdir()):
        raise FileExistsError(f"formal output must be fresh and empty: {root}")
    _atomic_bytes(root / "RUNNING", b"R2R0 formal run in progress\n")
    return root


def _failure(root: Path, exception: BaseException) -> None:
    payload = {
        "format": FORMAT,
        "status": "R2R0_FORMAL_DIAGNOSTIC_FAILURE",
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "can_authorize_fit_or_training": False,
    }
    running = root / "RUNNING"
    if running.exists():
        running.unlink()
    _atomic_json(root / "FAILED", payload)
    _atomic_bytes(root / "EXIT_CODE", b"1\n")


def _success(root: Path, receipt_path: Path, status: str) -> None:
    receipt_hash = sha256(receipt_path)
    _atomic_bytes(root / "EXIT_CODE", b"0\n")
    running = root / "RUNNING"
    if running.exists():
        running.unlink()
    _atomic_bytes(root / "DONE", (status + "\n" + receipt_hash + "\n").encode("ascii"))


def _validate_success_terminal_state(root: Path, label: str) -> None:
    present = {name for name in ("RUNNING", "FAILED", "DONE") if (root / name).exists()}
    if present != {"DONE"}:
        raise ValueError(f"{label} terminal markers must be exactly DONE, observed {sorted(present)}")
    if not (root / "EXIT_CODE").is_file() or (root / "EXIT_CODE").read_bytes() != b"0\n":
        raise ValueError(f"{label} exit marker changed")


def _validate_live_receipt_provenance(receipt: Mapping[str, Any], label: str) -> None:
    if receipt.get("formal_contract_sha256") != FORMAL_CONTRACT_SHA256:
        raise ValueError(f"{label} formal contract is stale")
    if receipt.get("frozen_source_sha256") != validate_frozen_sources():
        raise ValueError(f"{label} frozen primitive source receipt is stale")
    if receipt.get("input_sha256") != expected_input_sha256():
        raise ValueError(f"{label} input receipt is stale")
    if "frozen_R2R_canonical_sha256" in receipt and receipt.get(
        "frozen_R2R_canonical_sha256"
    ) != FROZEN_R2R_CANONICAL_SHA256:
        raise ValueError(f"{label} frozen R2R canonical receipt is stale")
    authorization = receipt.get("execution_authorization")
    current_sources = {name: sha256(path) for name, path in FORMAL_SOURCE_PATHS.items()}
    if not isinstance(authorization, dict) or authorization.get(
        "formal_source_sha256"
    ) != current_sources:
        raise ValueError(f"{label} formal source snapshot is stale")


def _validate_receipt_safety_fields(
    receipt: Mapping[str, Any],
    label: str,
    *,
    receipt_kind: str,
) -> None:
    def exact(observed: Any, required: Any) -> bool:
        if required is False:
            return observed is False
        if isinstance(required, Mapping):
            return bool(
                isinstance(observed, Mapping)
                and set(observed) == set(required)
                and all(exact(observed[key], value) for key, value in required.items())
            )
        return observed == required

    if receipt_kind not in RECEIPT_TOP_LEVEL_KEYS:
        raise ValueError(f"{label} unknown receipt safety schema: {receipt_kind}")
    expected_keys = RECEIPT_TOP_LEVEL_KEYS[receipt_kind]
    if set(receipt) != expected_keys:
        extra = sorted(set(receipt) - expected_keys)
        missing = sorted(expected_keys - set(receipt))
        raise ValueError(
            f"{label} top-level receipt schema changed; extra={extra}, missing={missing}"
        )
    expected_format, allowed_statuses = RECEIPT_FORMAT_AND_STATUS[receipt_kind]
    if receipt.get("format") != expected_format:
        raise ValueError(f"{label} receipt format changed")
    if receipt.get("status") not in allowed_statuses:
        raise ValueError(f"{label} receipt status changed")
    expected = RECEIPT_SAFETY_EXPECTED[receipt_kind]
    for key, value in expected.items():
        if not exact(receipt.get(key), value):
            raise ValueError(f"{label} safety field changed: {key}")
    for key in OPTIONAL_SAFETY_FALSE_FIELDS:
        if key in receipt and receipt[key] is not False:
            raise ValueError(f"{label} optional safety field changed: {key}")

    safety_false_patterns = set(CORE_NESTED_SAFETY_FALSE_PATHS[receipt_kind])
    safety_false_patterns.update(
        (key,) for key, value in expected.items() if value is False
    )
    safety_false_patterns.update(
        ("geometry_loader_isolation", key)
        for key in GEOMETRY_LOADER_ISOLATION
    )
    benign_patterns = CORE_BENIGN_SCIENTIFIC_PATHS[receipt_kind]

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = (*path, key)
                if _is_suspicious_safety_key(key):
                    benign = any(
                        _key_path_matches(child_path, pattern)
                        for pattern in benign_patterns
                    )
                    safety_false = any(
                        _key_path_matches(child_path, pattern)
                        for pattern in safety_false_patterns
                    )
                    if not benign and not safety_false:
                        raise ValueError(
                            f"{label} unexpected safety-like key: {'.'.join(child_path)}"
                        )
                    if safety_false and child is not False:
                        raise ValueError(
                            f"{label} safety-like field is not exact false: "
                            f"{'.'.join(child_path)}"
                        )
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(receipt, ())


def _validate_receipt_contract(
    receipt: Mapping[str, Any],
    label: str,
    *,
    receipt_kind: str,
) -> None:
    """Validate structural completeness before the secondary safety semantics."""
    _validate_recursive_receipt_schema(
        receipt, label, receipt_kind=receipt_kind
    )
    _validate_receipt_safety_fields(
        receipt, label, receipt_kind=receipt_kind
    )


def _array_hash(items: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def structure_semantic_sha256(structure: Atoms) -> str:
    return _array_hash(
        (
            ("numbers", np.asarray(structure.numbers, dtype=np.dtype("<i8"))),
            ("pbc", np.asarray(structure.pbc, dtype=np.bool_)),
            ("cell_A", np.asarray(structure.cell, dtype=np.dtype("<f8"))),
            ("positions_A", np.asarray(structure.positions, dtype=np.dtype("<f8"))),
        )
    )


def directory_manifest_sha256(root: Path) -> str:
    base = _reject_path(root, "formal artifact root")
    records = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError("formal artifact root contains a symlink")
        if path.is_file():
            records.append((str(path.relative_to(base)), sha256(path)))
    return r2r.semantic_sha256(records)


@dataclass(frozen=True)
class FormalInputs:
    endpoint_checkpoint: Path
    endpoint_receipt: Path
    endpoint_marker: Path
    reference_6x6: Path
    reference_8x8: Path
    thermal92: Path
    harmonic_zero32: Path

    def mapping(self) -> dict[str, Path]:
        return {
            "endpoint_checkpoint": self.endpoint_checkpoint,
            "endpoint_receipt": self.endpoint_receipt,
            "endpoint_marker": self.endpoint_marker,
            "reference_6x6": self.reference_6x6,
            "reference_8x8": self.reference_8x8,
            "thermal92": self.thermal92,
            "harmonic_zero32": self.harmonic_zero32,
        }


def validate_inputs(inputs: FormalInputs) -> dict[str, str]:
    mapping = {
        key: _reject_path(path, key) for key, path in inputs.mapping().items()
    }
    observed = r2r.verify_r2r0_input_hashes(mapping)
    receipt = json.loads(mapping["endpoint_receipt"].read_text(encoding="utf-8"))
    required = {
        "format": "graphene_r2q_frozen_endpoint_v1",
        "status": "FOUR_STEP_ENDPOINT_FROZEN",
        "endpoint_checkpoint_sha256": r2r.EXPECTED_INPUTS["endpoint_checkpoint"]["sha256"],
        "endpoint_model_state_sha256": r2r.R2Q_ENDPOINT_STATE_SHA256,
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            raise ValueError(f"R2Q endpoint receipt field changed: {key}")
    receipt_hash = observed["endpoint_receipt"]
    if mapping["endpoint_marker"].read_bytes() != (receipt_hash + "\n").encode("ascii"):
        raise ValueError("R2Q endpoint marker does not bind the exact receipt")
    return observed


def load_endpoint(inputs: FormalInputs, device: str) -> torch.nn.Module:
    validate_inputs(inputs)
    checkpoint = r2o.torch_load(inputs.endpoint_checkpoint, map_location="cpu")
    model = checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("R2Q endpoint checkpoint lacks complete model object")
    model = model.to(device=device, dtype=torch.float64).eval()
    if r2o.state_dict_sha256(model) != r2r.R2Q_ENDPOINT_STATE_SHA256:
        raise ValueError("loaded endpoint state differs from frozen R2Q state")
    r2o.validate_mace_architecture(model)
    return model


def _validate_structure_order(thermal: Sequence[Atoms], harmonic: Sequence[Atoms]) -> None:
    if len(thermal) != 92 or len(harmonic) != 32:
        raise ValueError("formal train-only structure counts changed")
    for start, stop, expected in THERMAL_CONFIG_SEGMENTS:
        observed = [item.info.get("config_type") for item in thermal[start:stop]]
        if observed != [expected] * (stop - start):
            raise ValueError(f"thermal config_type segment {start}:{stop} changed")
    if [item.info.get("config_type") for item in harmonic] != [HARMONIC_CONFIG_TYPE] * 32:
        raise ValueError("harmonic-zero config_type/order changed")


def read_geometry_only_extxyz(path: Path) -> list[Atoms]:
    """Read only species/positions/cell/PBC/config_type from an extxyz file.

    Numeric label columns are skipped as opaque tokens and are never converted,
    indexed, copied, or attached to an ``Atoms`` object.
    """
    source = _reject_path(path, "geometry-only extxyz")
    lines = source.read_text(encoding="utf-8").splitlines()
    structures: list[Atoms] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        count = int(lines[cursor].strip())
        cursor += 1
        header = shlex.split(lines[cursor], posix=True)
        cursor += 1
        fields: dict[str, str] = {}
        for token in header:
            if "=" in token:
                key, value = token.split("=", 1)
                if key in {"Properties", "Lattice", "pbc", "config_type"}:
                    fields[key] = value
        properties = fields.get("Properties", "").split(":")
        if len(properties) % 3:
            raise ValueError("extxyz Properties schema changed")
        offset = 0
        species_offset = None
        position_offset = None
        for index in range(0, len(properties), 3):
            name, _kind, width_text = properties[index : index + 3]
            width = int(width_text)
            if name == "species":
                if width != 1:
                    raise ValueError("geometry-only species width changed")
                species_offset = offset
            if name == "pos":
                if width != 3:
                    raise ValueError("geometry-only pos width changed")
                position_offset = offset
            offset += width
        if species_offset is None or position_offset is None:
            raise ValueError("geometry-only extxyz lacks species/pos")
        symbols = []
        positions = []
        for _ in range(count):
            tokens = lines[cursor].split()
            cursor += 1
            if len(tokens) != offset:
                raise ValueError("extxyz atom row width changed")
            symbols.append(tokens[species_offset])
            positions.append(
                [float(tokens[position_offset + component]) for component in range(3)]
            )
        lattice_tokens = fields["Lattice"].split()
        pbc_tokens = fields.get("pbc", "F F F").split()
        if len(lattice_tokens) != 9 or len(pbc_tokens) != 3 or any(value not in {"T", "F"} for value in pbc_tokens):
            raise ValueError("geometry-only Lattice/pbc width changed")
        lattice = np.asarray([float(value) for value in lattice_tokens], dtype=np.float64).reshape(3, 3)
        pbc = tuple(value == "T" for value in pbc_tokens)
        atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=pbc)
        atoms.info["config_type"] = fields.get("config_type")
        structures.append(atoms)
    return structures


def _design_record(
    model: torch.nn.Module, structure: Atoms, reference: Atoms, device: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    query = r2r.production_linear_design_query(
        model, structure, reference, device=device
    )
    background = r2r.background_from_structure(
        structure, reference, device=device, dtype=torch.float64, requires_grad=False
    )
    combined = query.canonical_combined_probe()
    arrays = {
        "fixed_energy_eV": np.asarray(query.fixed_offset_energy_eV.detach().cpu(), dtype="<f8"),
        "fixed_force_eV_A": np.asarray(
            query.fixed_offset_force_source_order_eV_A.detach().cpu(), dtype="<f8"
        ).reshape(len(structure), 3),
        "parameter_energy_design_eV": np.asarray(
            query.parameter_energy_design_eV.detach().cpu(), dtype="<f8"
        ),
        "parameter_force_design_eV_A": np.asarray(
            query.parameter_force_design_source_order_eV_A.detach().cpu(), dtype="<f8"
        ).reshape(len(structure), 3, 65),
        "b_A4": np.asarray(
            background.fields.cross_product_square_A4.detach().cpu(), dtype="<f8"
        ),
        "a": np.asarray(background.fields.multipolar_gate.detach().cpu(), dtype="<f8"),
        "c": np.asarray(background.fields.amplitude_gate.detach().cpu(), dtype="<f8"),
        "combined_energy_eV": np.asarray([float(combined.energy_eV.detach().cpu())], dtype="<f8"),
        "combined_force_eV_A": np.asarray(
            combined.force_source_order_eV_A.detach().cpu(), dtype="<f8"
        ),
    }
    receipt = {
        "structure_semantic_sha256": structure_semantic_sha256(structure),
        "query_receipt": query.receipt,
        "assignment_and_MIC_semantic_sha256": query.receipt[
            "assignment_and_MIC_semantic_sha256"
        ],
        "reference_semantic_sha256": query.receipt["reference_semantic_sha256"],
        "formal_graph_semantic_sha256": query.receipt[
            "formal_R2O_graph_semantic_sha256"
        ],
        "endpoint_state_sha256": query.receipt["endpoint_state_sha256"],
        "coefficients_sha256": combined.coefficients_sha256,
        "affine_component_names_sha256": query.receipt[
            "affine_component_names_sha256"
        ],
        "parameter_columns": query.receipt["parameter_columns"],
        "force_labels_used": False,
        "energy_labels_used": False,
    }
    return arrays, receipt


def _harmonic_record(
    model: torch.nn.Module, structure: Atoms, reference: Atoms, device: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    background = r2r.background_from_structure(
        structure, reference, device=device, dtype=torch.float64, requires_grad=False
    )
    combined = r2r.production_combined_energy_force(
        model, structure, reference, device=device
    )
    fixed = r2r.production_fixed_carrier_energy_force(
        model, structure, reference, device=device
    )
    arrays = {
        "b_A4": np.asarray(
            background.fields.cross_product_square_A4.detach().cpu(), dtype="<f8"
        ),
        "a": np.asarray(background.fields.multipolar_gate.detach().cpu(), dtype="<f8"),
        "c": np.asarray(background.fields.amplitude_gate.detach().cpu(), dtype="<f8"),
        "fixed_carrier_energy_eV": np.asarray(
            [float(fixed.energy_eV.detach().cpu())], dtype="<f8"
        ),
        "fixed_carrier_force_eV_A": np.asarray(
            fixed.force_source_order_eV_A.detach().cpu(), dtype="<f8"
        ),
        "combined_energy_eV": np.asarray([float(combined.energy_eV.detach().cpu())], dtype="<f8"),
        "combined_force_eV_A": np.asarray(
            combined.force_source_order_eV_A.detach().cpu(), dtype="<f8"
        ),
    }
    receipt = {
        "structure_semantic_sha256": structure_semantic_sha256(structure),
        "query_receipt": combined.query_receipt,
        "fixed_carrier_query_receipt": fixed.query_receipt,
        "fixed_carrier_coefficients_sha256": fixed.coefficients_sha256,
        "coefficients_sha256": combined.coefficients_sha256,
        "affine_component_names_sha256": r2r.AFFINE_COMPONENT_NAMES_SHA256,
        "parameter_columns": r2r.LINEAR_DESIGN_WIDTH,
        "force_labels_used": False,
        "energy_labels_used": False,
    }
    return arrays, receipt


def _stack_records(records: Sequence[dict[str, np.ndarray]], prefix: str) -> dict[str, np.ndarray]:
    keys = tuple(records[0])
    if any(tuple(item) != keys for item in records):
        raise ValueError("per-configuration array schema changed")
    return {f"{prefix}_{key}": np.stack([item[key] for item in records]) for key in keys}


def run_bounded_preflight(
    inputs: FormalInputs, output: Path, *, device: str, kind: str
) -> dict[str, Any]:
    if kind not in ("synthetic", "real"):
        raise ValueError("bounded preflight kind must be synthetic or real")
    root = _fresh_run_root(output)
    try:
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("bounded preflight requires float32-origin default dtype")
        source_hashes = validate_frozen_sources()
        input_hashes = validate_inputs(inputs)
        started = time.perf_counter()
        if kind == "synthetic":
            left = np.linspace(-1.0, 1.0, 65, dtype=np.float64)
            right = left + 1.0e-10 * np.sign(left)
            sentinel = _symmetric_sentinel_compare(left, right, 1.0e-9)
            payload = {
                "synthetic_symmetric_sentinel": sentinel,
                "geometry_only_thermal_count": len(
                    read_geometry_only_extxyz(inputs.thermal92)
                ),
                "geometry_only_harmonic_count": len(
                    read_geometry_only_extxyz(inputs.harmonic_zero32)
                ),
            }
            passed = bool(
                sentinel["pass"]
                and payload["geometry_only_thermal_count"] == 92
                and payload["geometry_only_harmonic_count"] == 32
            )
        else:
            if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1":
                raise ValueError(
                    "real bounded preflight requires TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
                )
            model = load_endpoint(inputs, device)
            state_before = r2o.state_dict_sha256(model)
            thermal = read_geometry_only_extxyz(inputs.thermal92)
            harmonic = read_geometry_only_extxyz(inputs.harmonic_zero32)
            _validate_structure_order(thermal, harmonic)
            reference6 = read(inputs.reference_6x6, index=0)
            reference8 = read(inputs.reference_8x8, index=0)
            design, design_receipt = _design_record(
                model, thermal[0], reference6, device
            )
            rank1, rank1_receipt = _harmonic_record(
                model, harmonic[0], reference8, device
            )
            state_after = r2o.state_dict_sha256(model)
            payload = {
                "thermal_global_index": 0,
                "thermal_structure_semantic_sha256": structure_semantic_sha256(
                    thermal[0]
                ),
                "thermal_array_schema": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in design.items()
                },
                "thermal_all_finite": all(
                    np.all(np.isfinite(value)) for value in design.values()
                ),
                "thermal_receipt": design_receipt,
                "harmonic_global_index": 0,
                "harmonic_structure_semantic_sha256": structure_semantic_sha256(
                    harmonic[0]
                ),
                "harmonic_array_schema": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in rank1.items()
                },
                "harmonic_all_finite": all(
                    np.all(np.isfinite(value)) for value in rank1.values()
                ),
                "harmonic_receipt": rank1_receipt,
                "endpoint_state_before": state_before,
                "endpoint_state_after": state_after,
            }
            passed = bool(
                payload["thermal_all_finite"]
                and payload["harmonic_all_finite"]
                and state_before == state_after == r2r.R2Q_ENDPOINT_STATE_SHA256
                and design["parameter_force_design_eV_A"].shape == (72, 3, 65)
                and rank1["combined_force_eV_A"].shape == (128, 3)
            )
        receipt = {
            "format": PREFLIGHT_FORMAT,
            "status": STATUS_PREFLIGHT,
            "kind": kind,
            "formal_contract_sha256": FORMAL_CONTRACT_SHA256,
            "frozen_source_sha256": source_hashes,
            "input_sha256": input_hashes,
            "runtime_fingerprint": runtime_fingerprint(device),
            "payload": payload,
            "pass": passed,
            "elapsed_seconds": time.perf_counter() - started,
            "force_labels_used": False,
            "energy_labels_used": False,
            "geometry_loader_isolation": dict(GEOMETRY_LOADER_ISOLATION),
            "can_authorize_fit_or_training": False,
            "can_authorize_remote_launch": False,
        }
        _validate_receipt_contract(
            receipt, "bounded formal preflight", receipt_kind="preflight"
        )
        receipt_path = root / "receipt.json"
        _atomic_json(receipt_path, receipt)
        _success(root, receipt_path, STATUS_PREFLIGHT)
        return receipt
    except BaseException as exception:
        _failure(root, exception)
        raise


def run_shard(
    inputs: FormalInputs,
    output: Path,
    shard_id: int,
    *,
    device: str,
    freeze_manifest: Path,
    authorization_marker: Path,
) -> dict[str, Any]:
    if shard_id not in SHARD_INDICES:
        raise ValueError("formal shard id must be 0, 1, or 2")
    authorization = validate_execution_authorization(
        freeze_manifest, authorization_marker
    )
    input_hashes = validate_inputs(inputs)
    existing = _reject_path(output, "formal shard output", must_exist=False)
    if existing.is_dir() and (existing / "DONE").exists():
        recovered, _ = _load_completed_shard(existing)
        if recovered["shard_id"] != shard_id:
            raise ValueError("completed shard belongs to a different shard id")
        if recovered.get("input_sha256") != input_hashes:
            raise ValueError("completed shard input hashes differ from live inputs")
        if recovered.get("execution_authorization") != authorization:
            raise ValueError("completed shard authorization binding changed")
        return {**recovered, "completion_recovered_without_recompute": True}
    root = _fresh_run_root(output)
    try:
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("formal production requires float32-origin default dtype")
        if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1":
            raise ValueError("formal run requires TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1")
        source_hashes = validate_frozen_sources()
        environment = runtime_fingerprint(device)
        model = load_endpoint(inputs, device)
        thermal = read_geometry_only_extxyz(inputs.thermal92)
        harmonic = read_geometry_only_extxyz(inputs.harmonic_zero32)
        _validate_structure_order(thermal, harmonic)
        reference6 = read(inputs.reference_6x6, index=0)
        reference8 = read(inputs.reference_8x8, index=0)
        partition = SHARD_INDICES[shard_id]
        started = time.perf_counter()
        thermal_arrays = []
        thermal_receipts = []
        for index in partition["thermal"]:
            arrays, receipt = _design_record(model, thermal[index], reference6, device)
            thermal_arrays.append(arrays)
            thermal_receipts.append({"global_index": index, **receipt})
        harmonic_arrays = []
        harmonic_receipts = []
        for index in partition["harmonic"]:
            arrays, receipt = _harmonic_record(model, harmonic[index], reference8, device)
            harmonic_arrays.append(arrays)
            harmonic_receipts.append({"global_index": index, **receipt})
        sentinel_records = {
            index: _design_record(model, thermal[index], reference6, device)
            for index in SENTINEL_GLOBAL_INDICES
        }
        thermal_artifact = {
            "thermal_global_index": np.asarray(partition["thermal"], dtype=np.dtype("<i8")),
            **_stack_records(thermal_arrays, "thermal"),
        }
        harmonic_artifact = {
            "harmonic_global_index": np.asarray(partition["harmonic"], dtype=np.dtype("<i8")),
            **_stack_records(harmonic_arrays, "harmonic"),
        }
        sentinel_artifact = {
            **{
                f"sentinel_{index}_{key}": value
                for index, (sentinel_arrays, _) in sentinel_records.items()
                for key, value in sentinel_arrays.items()
            },
        }
        artifact_arrays = {
            "thermal": thermal_artifact,
            "harmonic": harmonic_artifact,
            "sentinel": sentinel_artifact,
        }
        artifact_receipts = {}
        for role, role_arrays in artifact_arrays.items():
            artifact_path = root / f"{role}_arrays.npz"
            _atomic_npz(artifact_path, role_arrays)
            artifact_receipts[role] = {
                "basename": artifact_path.name,
                "sha256": sha256(artifact_path),
                "schema": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in role_arrays.items()
                },
            }
            _atomic_bytes(
                root / f"{role.upper()}_FROZEN",
                (artifact_receipts[role]["sha256"] + "\n").encode("ascii"),
            )
        receipt = {
            "format": SHARD_FORMAT,
            "status": STATUS_SHARD,
            "formal_contract_sha256": FORMAL_CONTRACT_SHA256,
            "execution_authorization": authorization,
            "frozen_source_sha256": source_hashes,
            "frozen_R2R_canonical_sha256": r2r.CANONICAL_CONTRACT_SHA256,
            "input_sha256": input_hashes,
            "runtime_fingerprint": environment,
            "shard_id": shard_id,
            "partition": partition,
            "thermal_count": len(partition["thermal"]),
            "harmonic_count": len(partition["harmonic"]),
            "thermal_receipts": thermal_receipts,
            "harmonic_receipts": harmonic_receipts,
            "sentinel_receipts": {
                str(index): {"global_index": index, **sentinel_receipt}
                for index, (_, sentinel_receipt) in sentinel_records.items()
            },
            "sentinel_included_in_matrix": False,
            "sentinel_contract_sha256": SENTINEL_CONTRACT_SHA256,
            "array_artifacts": artifact_receipts,
            "elapsed_seconds": time.perf_counter() - started,
            "force_labels_used": False,
            "energy_labels_used": False,
            "geometry_loader_isolation": dict(GEOMETRY_LOADER_ISOLATION),
            "can_authorize_fit_or_training": False,
        }
        _validate_receipt_contract(
            receipt, "fresh formal shard", receipt_kind="shard"
        )
        receipt_path = root / "receipt.json"
        _atomic_json(receipt_path, receipt)
        _success(root, receipt_path, STATUS_SHARD)
        return receipt
    except BaseException as exception:
        _failure(root, exception)
        raise


def _load_completed_shard(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    path = _reject_path(root, "formal completed run")
    _validate_success_terminal_state(path, "formal shard")
    if (path / "EXIT_CODE").read_bytes() != b"0\n":
        raise ValueError("formal run exit marker changed")
    receipt_path = path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format") != SHARD_FORMAT or receipt.get("status") != STATUS_SHARD:
        raise ValueError("formal run receipt format/status changed")
    _validate_live_receipt_provenance(receipt, "formal shard")
    _validate_receipt_contract(receipt, "formal shard", receipt_kind="shard")
    expected_done = (STATUS_SHARD + "\n" + sha256(receipt_path) + "\n").encode("ascii")
    if (path / "DONE").read_bytes() != expected_done:
        raise ValueError("formal DONE marker does not bind receipt")
    arrays = {}
    artifacts = receipt.get("array_artifacts")
    if set(artifacts or {}) != {"thermal", "harmonic", "sentinel"}:
        raise ValueError("formal shard artifact roles changed")
    for role, record in artifacts.items():
        if record.get("basename") != f"{role}_arrays.npz":
            raise ValueError(f"formal {role} artifact basename changed")
        arrays_path = _reject_path(path / record["basename"], f"formal {role} artifact")
        if sha256(arrays_path) != record["sha256"]:
            raise ValueError(f"formal {role} arrays hash changed")
        marker = path / f"{role.upper()}_FROZEN"
        if marker.read_bytes() != (record["sha256"] + "\n").encode("ascii"):
            raise ValueError(f"formal {role} marker does not bind artifact")
        with np.load(arrays_path, allow_pickle=False) as loaded:
            role_arrays = {key: loaded[key].copy() for key in loaded.files}
        observed_schema = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in role_arrays.items()
        }
        if observed_schema != record["schema"]:
            raise ValueError(f"formal {role} array shape/dtype schema changed")
        overlap = set(arrays) & set(role_arrays)
        if overlap:
            raise ValueError(f"formal artifact key overlap: {sorted(overlap)}")
        arrays.update(role_arrays)
    _validate_shard_array_contract(receipt, arrays)
    return receipt, arrays


def _validate_shard_array_contract(
    receipt: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    shard_id = int(receipt["shard_id"])
    nt = len(SHARD_INDICES[shard_id]["thermal"])
    nh = len(SHARD_INDICES[shard_id]["harmonic"])
    thermal_shapes = {
        "global_index": (nt,), "fixed_energy_eV": (nt, 1),
        "fixed_force_eV_A": (nt, 72, 3),
        "parameter_energy_design_eV": (nt, 65),
        "parameter_force_design_eV_A": (nt, 72, 3, 65),
        "b_A4": (nt, 72), "a": (nt, 72), "c": (nt, 72),
        "combined_energy_eV": (nt, 1), "combined_force_eV_A": (nt, 72, 3),
    }
    harmonic_shapes = {
        "global_index": (nh,), "b_A4": (nh, 128), "a": (nh, 128),
        "c": (nh, 128), "fixed_carrier_energy_eV": (nh, 1),
        "fixed_carrier_force_eV_A": (nh, 128, 3), "combined_energy_eV": (nh, 1),
        "combined_force_eV_A": (nh, 128, 3),
    }
    sentinel_shapes = {
        "fixed_energy_eV": (1,), "fixed_force_eV_A": (72, 3),
        "parameter_energy_design_eV": (65,),
        "parameter_force_design_eV_A": (72, 3, 65),
        "b_A4": (72,), "a": (72,), "c": (72,),
        "combined_energy_eV": (1,), "combined_force_eV_A": (72, 3),
    }
    expected = {
        **{f"thermal_{key}": shape for key, shape in thermal_shapes.items()},
        **{f"harmonic_{key}": shape for key, shape in harmonic_shapes.items()},
        **{
            f"sentinel_{index}_{key}": shape
            for index in SENTINEL_GLOBAL_INDICES
            for key, shape in sentinel_shapes.items()
        },
    }
    if set(arrays) != set(expected):
        raise ValueError("formal shard canonical array key set changed")
    for key, shape in expected.items():
        value = np.asarray(arrays[key])
        dtype = np.dtype("<i8") if key.endswith("global_index") else np.dtype("<f8")
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"formal shard canonical shape/dtype changed: {key}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"formal shard contains nonfinite values: {key}")


def _validate_aggregate_array_contract(arrays: Mapping[str, np.ndarray]) -> None:
    shapes = {
        "thermal_global_index": (92,),
        "thermal_fixed_energy_eV": (92, 1),
        "thermal_fixed_force_eV_A": (92, 72, 3),
        "thermal_parameter_energy_design_eV": (92, 65),
        "thermal_parameter_force_design_eV_A": (92, 72, 3, 65),
        "thermal_b_A4": (92, 72), "thermal_a": (92, 72), "thermal_c": (92, 72),
        "thermal_combined_energy_eV": (92, 1),
        "thermal_combined_force_eV_A": (92, 72, 3),
        "harmonic_global_index": (32,),
        "harmonic_b_A4": (32, 128), "harmonic_a": (32, 128), "harmonic_c": (32, 128),
        "harmonic_fixed_carrier_energy_eV": (32, 1),
        "harmonic_fixed_carrier_force_eV_A": (32, 128, 3),
        "harmonic_combined_energy_eV": (32, 1),
        "harmonic_combined_force_eV_A": (32, 128, 3),
    }
    if set(arrays) != set(shapes):
        raise ValueError("formal aggregate canonical array key set changed")
    for key, shape in shapes.items():
        value = np.asarray(arrays[key])
        dtype = np.dtype("<i8") if key.endswith("global_index") else np.dtype("<f8")
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"formal aggregate canonical shape/dtype changed: {key}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"formal aggregate contains nonfinite values: {key}")


def _symmetric_sentinel_compare(left: np.ndarray, right: np.ndarray, atol: float) -> dict:
    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape or first.dtype != second.dtype:
        return {"pass": False, "reason": "shape_or_dtype", "left_shape": list(first.shape), "right_shape": list(second.shape)}
    difference = np.abs(first - second)
    scale = np.maximum(np.abs(first), np.abs(second))
    bound = float(atol) + SENTINEL_CONTRACT["rtol"] * scale
    ratio = np.divide(difference, bound, out=np.zeros_like(difference), where=bound > 0)
    return {
        "pass": bool(np.all(np.isfinite(first)) and np.all(np.isfinite(second)) and np.all(difference <= bound)),
        "max_abs_difference": float(np.max(difference)),
        "max_bound_ratio": float(np.max(ratio)),
        "shape": list(first.shape),
        "dtype": str(first.dtype),
    }


def compare_sentinels(shards: Sequence[tuple[dict, dict[str, np.ndarray]]]) -> dict:
    exact_keys = (
        "structure_semantic_sha256",
        "reference_semantic_sha256",
        "formal_graph_semantic_sha256",
        "endpoint_state_sha256",
        "coefficients_sha256",
        "affine_component_names_sha256",
        "parameter_columns",
    )
    comparisons = []
    array_map = {
        "fixed_energy_eV": "sentinel_fixed_energy_eV",
        "parameter_energy_design_eV": "sentinel_parameter_energy_design_eV",
        "fixed_force_eV_A": "sentinel_fixed_force_eV_A",
        "parameter_force_design_eV_A": "sentinel_parameter_force_design_eV_A",
        "b_A4": "sentinel_b_A4",
        "a": "sentinel_a",
        "c": "sentinel_c",
    }
    for sentinel_index in SENTINEL_GLOBAL_INDICES:
        base_receipt = shards[0][0]["sentinel_receipts"][str(sentinel_index)]
        for receipt, _ in shards[1:]:
            candidate = receipt["sentinel_receipts"][str(sentinel_index)]
            if any(candidate.get(key) != base_receipt.get(key) for key in exact_keys):
                raise ValueError("sentinel exact provenance metadata differs across shards")
        for left_index in range(len(shards)):
            for right_index in range(left_index + 1, len(shards)):
                pair = {
                    "global_index": sentinel_index,
                    "shards": [left_index, right_index],
                    "arrays": {},
                }
                for contract_name, suffix in array_map.items():
                    array_name = f"sentinel_{sentinel_index}_{suffix.removeprefix('sentinel_')}"
                    tolerance = SENTINEL_CONTRACT["arrays"][contract_name]["atol"]
                    pair["arrays"][contract_name] = _symmetric_sentinel_compare(
                        shards[left_index][1][array_name],
                        shards[right_index][1][array_name],
                        tolerance,
                    )
                energy_name = f"sentinel_{sentinel_index}_combined_energy_eV"
                force_name = f"sentinel_{sentinel_index}_combined_force_eV_A"
                energy_difference = abs(float(
                    shards[left_index][1][energy_name].reshape(-1)[0]
                    - shards[right_index][1][energy_name].reshape(-1)[0]
                ))
                force_difference = float(np.max(np.abs(
                    shards[left_index][1][force_name]
                    - shards[right_index][1][force_name]
                )))
                pair["combined_probe"] = {
                    "energy_abs_difference_eV": energy_difference,
                    "force_max_abs_difference_eV_A": force_difference,
                    "pass": bool(
                        energy_difference <= SENTINEL_CONTRACT["combined_probe"]["energy_abs_eV"]
                        and force_difference <= SENTINEL_CONTRACT["combined_probe"]["force_max_abs_eV_A"]
                    ),
                }
                pair["pass"] = bool(
                    all(item["pass"] for item in pair["arrays"].values())
                    and pair["combined_probe"]["pass"]
                )
                comparisons.append(pair)
    return {
        "format": SENTINEL_CONTRACT["format"],
        "contract_sha256": SENTINEL_CONTRACT_SHA256,
        "comparisons": comparisons,
        "pass": all(item["pass"] for item in comparisons),
    }


def _validate_partition_receipts(
    loaded: Sequence[tuple[dict, dict[str, np.ndarray]]]
) -> dict[str, Any]:
    identities: dict[str, list[tuple[int, str]]] = {"thermal": [], "harmonic": []}
    for receipt, arrays in loaded:
        shard_id = int(receipt["shard_id"])
        expected = SHARD_INDICES[shard_id]
        for role in ("thermal", "harmonic"):
            records = receipt[f"{role}_receipts"]
            observed_indices = [int(item["global_index"]) for item in records]
            array_indices = arrays[f"{role}_global_index"].astype(int).tolist()
            if observed_indices != list(expected[role]) or array_indices != list(expected[role]):
                raise ValueError(f"{role} receipt/array partition differs from frozen modulo shard")
            if receipt[f"{role}_count"] != len(expected[role]):
                raise ValueError(f"{role} count differs from frozen modulo shard")
            for item in records:
                if item.get("force_labels_used") is not False or item.get("energy_labels_used") is not False:
                    raise ValueError(f"{role} receipt claims label use")
                if item.get("coefficients_sha256") != r2r.MECHANICS_PROBE_COEFFICIENTS_SHA256:
                    raise ValueError(f"{role} coefficient hash changed")
                if item.get("affine_component_names_sha256") != r2r.AFFINE_COMPONENT_NAMES_SHA256:
                    raise ValueError(f"{role} component column hash changed")
                if item.get("parameter_columns") != 65:
                    raise ValueError(f"{role} parameter column count changed")
                query = item.get("query_receipt", {})
                state_hash = item.get("endpoint_state_sha256", query.get("endpoint_state_sha256"))
                if state_hash != r2r.R2Q_ENDPOINT_STATE_SHA256:
                    raise ValueError(f"{role} endpoint state hash changed")
                if role == "harmonic":
                    fixed_query = item.get("fixed_carrier_query_receipt", {})
                    if item.get("fixed_carrier_coefficients_sha256") != r2r.FIXED_CARRIER_COEFFICIENT_SHA256:
                        raise ValueError("harmonic fixed-carrier coefficient hash changed")
                    shared_keys = (
                        "endpoint_state_sha256",
                        "reference_semantic_sha256",
                        "formal_R2O_graph_semantic_sha256",
                        "assignment_and_MIC_semantic_sha256",
                    )
                    if any(fixed_query.get(key) != query.get(key) for key in shared_keys):
                        raise ValueError("harmonic fixed/combined production provenance differs")
                for key in (
                    "structure_semantic_sha256",
                    "reference_semantic_sha256",
                    "formal_graph_semantic_sha256",
                ):
                    value = item.get(key)
                    if value is None and key == "formal_graph_semantic_sha256":
                        value = query.get("formal_R2O_graph_semantic_sha256")
                    if value is None and key == "reference_semantic_sha256":
                        value = query.get("reference_semantic_sha256")
                    if not isinstance(value, str) or len(value) != 64:
                        raise ValueError(f"{role} missing content provenance {key}")
                identities[role].append(
                    (int(item["global_index"]), item["structure_semantic_sha256"])
                )
        if set(receipt.get("sentinel_receipts", {})) != {
            str(index) for index in SENTINEL_GLOBAL_INDICES
        } or receipt.get("sentinel_included_in_matrix") is not False:
            raise ValueError("four sentinel receipts or matrix exclusion changed")
        if any(
            item.get("force_labels_used") is not False
            or item.get("energy_labels_used") is not False
            for item in receipt["sentinel_receipts"].values()
        ):
            raise ValueError("sentinel receipt claims label use")
    output = {}
    for role, pairs in identities.items():
        ordered = sorted(pairs)
        expected_count = 92 if role == "thermal" else 32
        if [index for index, _ in ordered] != list(range(expected_count)):
            raise ValueError(f"{role} identities overlap or omit a global index")
        hashes = [digest for _, digest in ordered]
        if len(set(hashes)) != expected_count:
            raise ValueError(f"{role} structure content identities are not unique")
        output[f"{role}_structure_identity_sha256"] = r2r.semantic_sha256(ordered)
    return output


def _node_parity(
    model: torch.nn.Module, structure: Atoms, reference_template: Atoms, device: str
) -> dict[str, Any]:
    reference = r2o.adapt_reference_cell(reference_template, structure)
    assignment = r2o.solve_assignment(structure, reference)
    ordered = r2o.reordered_structure(structure, assignment)
    aligned = (
        np.asarray(ordered.positions, dtype=np.float64)
        - assignment.image_integer_reference_order
        @ np.asarray(ordered.cell, dtype=np.float64)
    )
    graph = r2o.fixed_reference_graph(reference, device=device, dtype=torch.float64)
    positions = torch.as_tensor(aligned, dtype=torch.float64, device=device).requires_grad_(True)
    observables = r2r.r2q_node_observables(model, graph, positions)
    node_sum = observables.scale_shift_node_energy_eV.sum()
    formal = r2o.mace_interaction_energy(model, graph, positions).sum()
    node_gradient = torch.autograd.grad(node_sum, positions, retain_graph=True)[0]
    formal_gradient = torch.autograd.grad(formal, positions)[0]
    return {
        "structure_semantic_sha256": structure_semantic_sha256(structure),
        "energy_sum_abs_difference_eV": abs(float((node_sum - formal).detach().cpu())),
        "position_gradient_max_abs_difference_eV_A": float(
            torch.max(torch.abs(node_gradient - formal_gradient)).detach().cpu()
        ),
        "signed_l0_shape": list(observables.signed_l0.shape),
        "force_labels_used": False,
    }


def _combined_ef(
    model: torch.nn.Module,
    structure: Atoms,
    reference: Atoms,
    device: str,
    **kwargs: Any,
) -> tuple[float, np.ndarray, dict]:
    value = r2r.production_combined_energy_force(
        model, structure, reference, device=device, **kwargs
    )
    return (
        float(value.energy_eV.detach().cpu()),
        np.asarray(value.force_source_order_eV_A.detach().cpu(), dtype=np.float64),
        value.query_receipt,
    )


def _o3_covariant_provenance_ok(item: Mapping[str, Any]) -> bool:
    try:
        query = item["query_receipt"]
        rigid = query["rigid_transform_receipt"]
        graph_diagnostic = rigid["native_rebuild_diagnostic"]
        combined_diagnostic = item["native_rebuild_combined_EF_diagnostic"]
        frozen = FORMAL_CONTRACT["O3_rigid_transform_probe_v4"]
        return bool(
            query["formal_R2O_graph_mode"] == "rigid_transform_probe"
            and query["diagnostic_native_rebuild_selected"] is False
            and rigid["selected_graph_sha256"] == rigid["derived_graph_sha256"]
            and rigid["baseline_graph_frozen_hash_match"] is True
            and all(rigid["assignment_arrays_byte_equal"].values())
            and rigid["baseline_structure_full_semantic_sha256"]
            == frozen["baseline_structure_full_semantic_sha256"]
            and rigid["baseline_reference_full_semantic_sha256"]
            == frozen["baseline_reference_full_semantic_sha256"]
            and rigid["native_rebuild_used_for_physics"] is False
            and rigid["public_reference_input_covariance_array_exact"] is True
            and rigid["public_structure_input_covariance_array_exact"] is True
            and rigid["internal_covariance_atol_A"]
            == r2r.RIGID_INTERNAL_COVARIANCE_ATOL_A
            and all(
                rigid[
                    "reordered_and_reference_covariance_within_named_tolerance"
                ].values()
            )
            and rigid["native_background_physical_edge_multiset_equivalent"] is True
            and rigid["native_background_multiset_diagnostic"][
                "numeric_differences_are_diagnostic_only"
            ] is True
            and graph_diagnostic["native_physical_edge_multiset_equivalent"] is True
            and graph_diagnostic["numeric_differences_are_diagnostic_only"] is True
            and graph_diagnostic["physics_gate_authorized"] is False
            and combined_diagnostic["physics_gate_authorized"] is False
            and combined_diagnostic["authorization_role"]
            == "diagnostic_only_not_physics_gate"
            and math.isfinite(combined_diagnostic["energy_abs_difference_eV"])
            and math.isfinite(combined_diagnostic["force_max_abs_difference_eV_A"])
        )
    except (KeyError, TypeError, AttributeError):
        return False


def _frozen_r2q_full_mechanics(
    model: torch.nn.Module, structure: Atoms, reference: Atoms, device: str
) -> tuple[float, np.ndarray, np.ndarray]:
    result, assignment = r2o.evaluate_structure(
        model, structure, reference, device=device, create_graph=True
    )
    rows = []
    for component in result.forces_reference_order.reshape(-1):
        rows.append(
            -torch.autograd.grad(
                component,
                result.current_positions_reference_order,
                retain_graph=True,
            )[0].reshape(-1)
        )
    reference_hessian = torch.stack(rows)
    reference_to_source = torch.as_tensor(
        assignment.reference_to_source,
        dtype=torch.long,
        device=reference_hessian.device,
    )
    source_components = (
        3 * reference_to_source[:, None]
        + torch.arange(3, device=reference_hessian.device)[None, :]
    ).reshape(-1)
    source_hessian = torch.empty_like(reference_hessian)
    source_hessian[source_components[:, None], source_components[None, :]] = (
        reference_hessian
    )
    force = r2o.source_order_forces(result.forces_reference_order, assignment)
    energy_value = float(result.energy.detach().cpu())
    force_value = np.asarray(force.detach().cpu(), dtype=np.float64).copy()
    hessian_value = np.asarray(source_hessian.detach().cpu(), dtype=np.float64).copy()
    del result, rows, reference_hessian, source_hessian, force
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return energy_value, force_value, hessian_value


def _corrected_carrier_parity(
    model: torch.nn.Module, structure: Atoms, reference: Atoms, device: str
) -> dict[str, Any]:
    carrier = r2r.production_fixed_carrier_energy_force_hessian(
        model, structure, reference, device=device
    )
    carrier_energy = float(carrier.energy_eV.detach().cpu())
    carrier_force = np.asarray(
        carrier.force_source_order_eV_A.detach().cpu(), dtype=np.float64
    ).copy()
    carrier_hessian = np.asarray(
        carrier.Hessian_source_order_eV_A2.detach().cpu(), dtype=np.float64
    ).copy()
    carrier_receipt = carrier.query_receipt
    del carrier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    frozen_energy, frozen_force, frozen_hessian = _frozen_r2q_full_mechanics(
        model, structure, reference, device
    )
    return {
        "structure_semantic_sha256": structure_semantic_sha256(structure),
        "energy_abs_difference_eV": abs(
            carrier_energy - frozen_energy
        ),
        "force_max_abs_difference_eV_A": float(
            np.max(np.abs(carrier_force - frozen_force))
        ),
        "Hessian_max_abs_difference_eV_A2": float(
            np.max(np.abs(carrier_hessian - frozen_hessian))
        ),
        "carrier_query_receipt": carrier_receipt,
    }


def localized_reference_site_mapping(
    reference6: Atoms, reference8: Atoms
) -> dict[str, Any]:
    """Map 6x6 sites into 8x8 by species and rounded reference MIC vector."""
    def keyed(reference: Atoms) -> dict[tuple[int, float, float, float], int]:
        delta = np.asarray(reference.positions, dtype=np.float64) - np.asarray(
            reference.positions[0], dtype=np.float64
        )
        mic, _ = find_mic(delta, reference.cell, pbc=reference.pbc)
        rounded = np.round(np.asarray(mic, dtype=np.float64), decimals=7)
        result = {}
        for index, (number, vector) in enumerate(zip(reference.numbers, rounded)):
            key = (int(number), *(float(value) for value in vector))
            if key in result:
                raise ValueError("localized reference MIC key is not unique")
            result[key] = index
        return result

    keys6 = keyed(reference6)
    keys8 = keyed(reference8)
    if not set(keys6).issubset(keys8):
        raise ValueError("6x6 localized reference sites are not a subset of 8x8")
    common_keys = sorted(keys6)
    extra_keys = sorted(set(keys8) - set(keys6))
    mapping = [[keys6[key], keys8[key]] for key in common_keys]
    payload = {
        "definition": "(atomic_number, round(reference MIC(r_i-r_0),7 decimals))",
        "common_6x6_to_8x8": mapping,
        "extra_8x8_indices": [keys8[key] for key in extra_keys],
    }
    return {
        **payload,
        "common_count": len(mapping),
        "only_6x6_count": 0,
        "only_8x8_count": len(extra_keys),
        "semantic_sha256": r2r.semantic_sha256(payload),
    }


def _little_endian_array(value: Any, dtype: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.dtype(dtype)), dtype=dtype)


def _array_raw_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _round_q_payload(center_q: np.ndarray, neighbor_q: np.ndarray) -> np.ndarray:
    payload = np.ascontiguousarray(
        np.concatenate((center_q, neighbor_q)), dtype="<f8"
    )
    return np.ascontiguousarray(
        np.round(payload.astype(np.float64), decimals=Q_ROUND_DECIMALS),
        dtype="<f8",
    )


def _q_array_key(reference_name: str, device_tag: str, name: str) -> str:
    return f"q_{reference_name}_{device_tag}_{name}"


def _source_order_semantic_sha256(
    arrays: Mapping[str, np.ndarray], atom_count: int
) -> str:
    offsets = np.asarray(arrays["local_source_order_offsets"], dtype=np.int64)
    sender = np.asarray(arrays["local_source_order_sender"], dtype=np.int64)
    if (
        offsets.shape != (atom_count + 1,)
        or offsets[0] != 0
        or offsets[-1] != sender.size
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("q local source-order offsets changed")
    payload = [
        {
            "receiver": node,
            "sender_order": sender[offsets[node] : offsets[node + 1]].tolist(),
        }
        for node in range(atom_count)
    ]
    return r2r.semantic_sha256(payload)


def _attach_formal_zero_jet_probe_raw(
    graph: Any, zero_jet: dict[str, Any]
) -> None:
    """Attach actual-device primitive values needed for independent replay."""
    probes = zero_jet.get("nonzero_local_production_probe", [])
    expected_nodes = sorted(
        set(np.linspace(0, int(graph.atom_count) - 1, 4, dtype=int).tolist())
    )
    if [int(item.get("node", -1)) for item in probes] != expected_nodes:
        raise ValueError("zero-jet nonzero probe node inventory changed")
    neighbor_q = graph.weight / graph.normalization[graph.receiver]
    center_q = 1.0 / graph.normalization
    for item, node in zip(probes, expected_nodes):
        selected = torch.nonzero(graph.receiver == node, as_tuple=False).flatten()
        sender = graph.sender[selected]
        local_q = torch.cat((center_q[node : node + 1], neighbor_q[selected]))
        base = torch.arange(
            local_q.numel() * 3,
            dtype=graph.weight.dtype,
            device=graph.weight.device,
        )
        base = (
            torch.sin(0.37 * base + 0.19) + torch.cos(0.23 * base - 0.11)
        ).reshape(local_q.numel(), 3)
        base = 1.0e-3 * (base - base.mean(0))
        base_b = r2r._local_direct_multipolar_b(base, local_q)
        scale = (r2r.MULTIPOLAR_BETA_A4 / base_b).pow(0.25).detach()
        local_z = (scale * base).detach().requires_grad_(True)
        local_value = r2r._local_direct_multipolar_gate(local_z, local_q)
        local_gradient = torch.autograd.grad(local_value, local_z)[0]
        global_z = torch.zeros(
            (graph.atom_count, 3),
            dtype=graph.weight.dtype,
            device=graph.weight.device,
        )
        global_z[node] = local_z.detach()[0]
        global_z.index_copy_(0, sender, local_z.detach()[1:])
        global_z.requires_grad_(True)
        production_value = r2r.multipolar_background(
            global_z, graph
        ).multipolar_gate[node]
        production_gradient = torch.autograd.grad(production_value, global_z)[0]
        item.update(
            {
                "formal_raw_probe_format": (
                    "graphene_r2r0_zero_jet_probe_raw_v1_actual_device"
                ),
                "raw_local_gradient_A-1": _little_endian_array(
                    local_gradient, "<f8"
                ).tolist(),
                "raw_production_global_gradient_A-1": _little_endian_array(
                    production_gradient, "<f8"
                ).tolist(),
            }
        )


def _capture_formal_q_snapshot(
    reference: Atoms,
    reference_name: str,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Capture actual-device normalized q and the full physical zero jet."""
    graph = r2r.fixed_reference_neighborhood(
        reference, device=device_name, dtype=torch.float64
    )
    # These divisions must remain on the graph device.  Host recomputation would
    # erase the CUDA index_add_/division rounding being bounded by this gate.
    center_q_tensor = 1.0 / graph.normalization
    neighbor_q_tensor = graph.weight / graph.normalization[graph.receiver]
    receiver = _little_endian_array(graph.receiver, "<i8")
    sender = _little_endian_array(graph.sender, "<i8")
    image = _little_endian_array(graph.image_integer, "<i8")
    source_offsets = [0]
    source_sender: list[int] = []
    source_edge_index: list[int] = []
    for node in range(len(reference)):
        selected = np.flatnonzero(receiver == node).astype("<i8", copy=False)
        source_edge_index.extend(int(value) for value in selected)
        source_sender.extend(int(value) for value in sender[selected])
        source_offsets.append(len(source_sender))
    arrays = {
        "receiver": receiver,
        "sender": sender,
        "image_integer": image,
        "local_source_order_offsets": np.asarray(source_offsets, dtype="<i8"),
        "local_source_order_sender": np.asarray(source_sender, dtype="<i8"),
        "local_source_order_edge_index": np.asarray(source_edge_index, dtype="<i8"),
        "reference_distance_A": _little_endian_array(
            graph.reference_distance_A, "<f8"
        ),
        "weight": _little_endian_array(graph.weight, "<f8"),
        "normalization": _little_endian_array(graph.normalization, "<f8"),
        "center_q": _little_endian_array(center_q_tensor, "<f8"),
        "neighbor_q": _little_endian_array(neighbor_q_tensor, "<f8"),
    }
    source_hash = _source_order_semantic_sha256(arrays, len(reference))
    zero_jet = r2r.background_rank0_zero_jet_audit(
        graph, full_local_hessian=True
    )
    _attach_formal_zero_jet_probe_raw(graph, zero_jet)
    if source_hash != zero_jet.get("production_local_source_order_sha256"):
        raise ValueError("q source-order reconstruction differs from zero-jet audit")
    rounded = _round_q_payload(arrays["center_q"], arrays["neighbor_q"])
    public = {
        "reference": reference_name,
        "device": device_name,
        "atom_count": len(reference),
        "edge_count": int(receiver.size),
        "local_source_order_semantic_sha256": source_hash,
        "round12_payload_sha256": _array_raw_sha256(rounded),
        "array_raw_sha256": {
            name: _array_raw_sha256(value) for name, value in arrays.items()
        },
        "zero_jet": zero_jet,
        "raw_q_orbit_hash_role": "diagnostic_only_not_a_gate",
    }
    return public, arrays


def _zero_jet_physical_and_parity_exact(
    item: Mapping[str, Any],
    *,
    expected_node_count: int,
    expected_source_order_sha256: str,
    q_arrays: Mapping[str, np.ndarray],
) -> bool:
    try:
        probe = item["nonzero_local_production_probe"]
        expected_nodes = sorted(
            set(np.linspace(0, int(item["node_count"]) - 1, 4, dtype=int).tolist())
        )
        offsets = np.asarray(q_arrays["local_source_order_offsets"], dtype=np.int64)
        sender = np.asarray(q_arrays["local_source_order_sender"], dtype=np.int64)
        parity_items = []
        if not isinstance(probe, list) or [int(value["node"]) for value in probe] != expected_nodes:
            return False
        for value in probe:
            node = int(value["node"])
            local_gradient = np.asarray(
                value["raw_local_gradient_A-1"], dtype=np.float64
            )
            production_gradient = np.asarray(
                value["raw_production_global_gradient_A-1"], dtype=np.float64
            )
            local_sender = sender[offsets[node] : offsets[node + 1]]
            if (
                value.get("formal_raw_probe_format")
                != "graphene_r2r0_zero_jet_probe_raw_v1_actual_device"
                or local_gradient.shape != (int(item["local_sample_count"]), 3)
                or production_gradient.shape != (expected_node_count, 3)
                or local_sender.shape != (int(item["local_sample_count"]) - 1,)
                or not np.all(np.isfinite(local_gradient))
                or not np.all(np.isfinite(production_gradient))
            ):
                return False
            mapped_gradient = np.zeros_like(production_gradient)
            mapped_gradient[node] = local_gradient[0]
            np.add.at(mapped_gradient, local_sender, local_gradient[1:])
            touched = np.zeros(expected_node_count, dtype=bool)
            touched[node] = True
            touched[local_sender] = True
            outside = production_gradient[~touched]
            recomputed_value_difference = abs(
                float(value["production_a"]) - float(value["local_a"])
            )
            recomputed_gradient_difference = float(
                np.max(np.abs(production_gradient - mapped_gradient))
            )
            recomputed_outside_max = (
                float(np.max(np.abs(outside))) if outside.size else 0.0
            )
            recomputed_b_over_beta = (
                float(value["base_b_A4"])
                * float(value["scale"]) ** 4
                / r2r.MULTIPOLAR_BETA_A4
            )
            parity_items.append(
                all(
                    math.isfinite(float(raw))
                    for raw in (
                        value["base_b_A4"],
                        value["scale"],
                        value["target_a"],
                        value["local_a"],
                        value["production_a"],
                        recomputed_b_over_beta,
                        recomputed_value_difference,
                        recomputed_gradient_difference,
                        recomputed_outside_max,
                    )
                )
                and float(value["base_b_A4"]) > 0.0
                and float(value["scale"]) > 0.0
                and abs(recomputed_b_over_beta - 1.0) <= 1.0e-12
                and float(value["target_a"]) == -math.expm1(-1.0)
                and abs(float(value["local_a"]) - float(value["target_a"]))
                <= 1.0e-14
                and recomputed_value_difference <= 1.0e-14
                and recomputed_gradient_difference <= 1.0e-12
                and recomputed_outside_max == 0.0
            )
        parity = all(parity_items)
        return bool(
            item["format"]
            == "graphene_r2r_background_rank0_zero_jet_audit_v3_local_direct"
            and item["node_count"] == expected_node_count
            and item["local_sample_count"] == 40
            and item["local_Hessian_shape_per_node"] == [40, 3, 40, 3]
            and item["production_local_source_order_sha256"]
            == expected_source_order_sha256
            and item["multipolar_gate_value_max_abs"] == 0.0
            and item["global_all_node_Jacobian_max_abs_A-1"] == 0.0
            and item["global_all_node_value_and_Jacobian_exact_zero"] is True
            and item["local_all_node_value_max_abs"] == 0.0
            and item["local_all_node_Jacobian_max_abs_A-1"] == 0.0
            and item["local_all_node_Hessian_max_abs_A-2"] == 0.0
            and item["local_all_node_finite"] is True
            and item["all_nodes_covered_exactly_once"] is True
            and item["all_local_senders_unique"] is True
            and item["full_local_hessian_performed"] is True
            and parity
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _recompute_q_portability_gate(
    mechanics: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    frozen = FORMAL_CONTRACT["attempt3_q_portability_v1"]
    expected_snapshot_names = {
        f"{reference_name}_{device_tag}"
        for reference_name in frozen["references"]
        for device_tag in ("CPU", "CUDA")
    }
    snapshots = mechanics.get("q_snapshots", {})
    zero_jets = mechanics.get("zero_jet", {})
    inventory_exact = bool(
        set(snapshots) == expected_snapshot_names
        and set(zero_jets) == expected_snapshot_names
    )
    per_reference: dict[str, Any] = {}
    for reference_name, expected in frozen["references"].items():
        device_arrays = {}
        for device_tag in ("CPU", "CUDA"):
            device_arrays[device_tag] = {
                name: arrays[_q_array_key(reference_name, device_tag, name)]
                for name in (*Q_DISCRETE_ARRAY_NAMES, *Q_FLOAT_ARRAY_NAMES)
            }
        discrete_exact = {
            name: bool(
                np.array_equal(
                    device_arrays["CPU"][name], device_arrays["CUDA"][name]
                )
            )
            for name in Q_DISCRETE_ARRAY_NAMES
        }
        canonical_discrete = {
            name: _array_raw_sha256(device_arrays["CPU"][name])
            == expected["canonical_CPU_discrete_raw_sha256"][name]
            for name in Q_DISCRETE_ARRAY_NAMES
        }
        source_hashes = {
            device_tag: _source_order_semantic_sha256(
                device_arrays[device_tag], int(expected["atom_count"])
            )
            for device_tag in ("CPU", "CUDA")
        }
        source_order_exact = bool(
            len(set(source_hashes.values())) == 1
            and source_hashes["CPU"]
            == expected["local_source_order_semantic_sha256"]
        )
        float_comparison = {}
        for name in Q_FLOAT_ARRAY_NAMES:
            difference = np.abs(
                device_arrays["CPU"][name] - device_arrays["CUDA"][name]
            )
            float_comparison[name] = {
                "max_abs": float(np.max(difference)) if difference.size else 0.0,
                "exact_different_count": int(
                    np.count_nonzero(
                        device_arrays["CPU"][name]
                        != device_arrays["CUDA"][name]
                    )
                ),
            }
        normalized_q_max_abs = max(
            float_comparison["center_q"]["max_abs"],
            float_comparison["neighbor_q"]["max_abs"],
        )
        rounded = {
            device_tag: _round_q_payload(
                device_arrays[device_tag]["center_q"],
                device_arrays[device_tag]["neighbor_q"],
            )
            for device_tag in ("CPU", "CUDA")
        }
        rounded_hashes = {
            device_tag: _array_raw_sha256(value)
            for device_tag, value in rounded.items()
        }
        round12_exact = bool(
            np.array_equal(rounded["CPU"], rounded["CUDA"])
            and rounded_hashes["CPU"] == rounded_hashes["CUDA"]
            and rounded_hashes["CPU"] == expected["round12_payload_sha256"]
        )
        physical_zero_jet_exact = {
            device_tag: (
                inventory_exact
                and _zero_jet_physical_and_parity_exact(
                    zero_jets[f"{reference_name}_{device_tag}"],
                    expected_node_count=int(expected["atom_count"]),
                    expected_source_order_sha256=source_hashes[device_tag],
                    q_arrays=device_arrays[device_tag],
                )
            )
            for device_tag in ("CPU", "CUDA")
        }
        snapshot_array_binding = {}
        for device_tag in ("CPU", "CUDA"):
            snapshot = snapshots.get(f"{reference_name}_{device_tag}", {})
            snapshot_array_binding[device_tag] = bool(
                snapshot.get("reference") == reference_name
                and (
                    snapshot.get("device") == "cpu"
                    if device_tag == "CPU"
                    else str(snapshot.get("device", "")).startswith("cuda")
                )
                and snapshot.get("atom_count") == expected["atom_count"]
                and snapshot.get("edge_count") == expected["edge_count"]
                and snapshot.get("local_source_order_semantic_sha256")
                == source_hashes[device_tag]
                and snapshot.get("round12_payload_sha256")
                == rounded_hashes[device_tag]
                and snapshot.get("array_raw_sha256")
                == {
                    name: _array_raw_sha256(device_arrays[device_tag][name])
                    for name in (*Q_DISCRETE_ARRAY_NAMES, *Q_FLOAT_ARRAY_NAMES)
                }
                and snapshot.get("raw_q_orbit_hash_role")
                == "diagnostic_only_not_a_gate"
            )
        topology_and_source_order_exact = bool(
            all(discrete_exact.values())
            and all(canonical_discrete.values())
            and source_order_exact
        )
        passed = bool(
            inventory_exact
            and all(snapshot_array_binding.values())
            and topology_and_source_order_exact
            and normalized_q_max_abs <= Q_MAX_ABS_TOLERANCE
            and round12_exact
            and all(physical_zero_jet_exact.values())
        )
        per_reference[reference_name] = {
            "discrete_CPU_CUDA_exact": discrete_exact,
            "canonical_CPU_discrete_hashes_match": canonical_discrete,
            "source_order_semantic_sha256": source_hashes,
            "topology_and_source_order_exact": topology_and_source_order_exact,
            "float_arrays_CPU_vs_CUDA": float_comparison,
            "normalized_q_max_abs": normalized_q_max_abs,
            "normalized_q_max_abs_tolerance": Q_MAX_ABS_TOLERANCE,
            "round12_payload_sha256": rounded_hashes,
            "round12_elementwise_and_canonical_hash_exact": round12_exact,
            "CPU_and_CUDA_physical_zero_jet_exact": all(
                physical_zero_jet_exact.values()
            ),
            "physical_zero_jet_exact": physical_zero_jet_exact,
            "q_snapshot_arrays_exactly_bound": snapshot_array_binding,
            "raw_q_orbit_hash_equal": bool(
                inventory_exact
                and zero_jets[f"{reference_name}_CPU"].get(
                    "q_signature_orbits_sha256"
                )
                == zero_jets[f"{reference_name}_CUDA"].get(
                    "q_signature_orbits_sha256"
                )
            ),
            "raw_q_orbit_hash_role": "diagnostic_only_not_a_gate",
            "pass": passed,
        }
    return {
        "snapshot_and_zero_jet_inventory_exact": inventory_exact,
        "per_reference": per_reference,
        "pass": bool(inventory_exact and all(item["pass"] for item in per_reference.values())),
    }


def _empirical_order(coarser: float, finer: float) -> float | None:
    if coarser == 0.0 or finer == 0.0:
        return None
    return math.log2(coarser / finer)


def _analyze_formal_finite_difference(
    steps_A: np.ndarray,
    energy_eV: np.ndarray,
    force_coordinate_eV_A: np.ndarray,
    *,
    force_target_eV_A: float,
    force_jacobian_target_eV_A2: float,
) -> dict[str, Any]:
    rows = []
    for index, h in enumerate(steps_A):
        energy_derivative = -(energy_eV[index, 1] - energy_eV[index, 0]) / (2 * h)
        force_derivative = (
            force_coordinate_eV_A[index, 1]
            - force_coordinate_eV_A[index, 0]
        ) / (2 * h)
        left = (force_target_eV_A - force_coordinate_eV_A[index, 0]) / h
        right = (force_coordinate_eV_A[index, 1] - force_target_eV_A) / h
        rows.append(
            {
                "h_A": float(h),
                "energy_FD_force_eV_A": float(energy_derivative),
                "energy_FD_force_abs_error_eV_A": abs(
                    float(energy_derivative) - force_target_eV_A
                ),
                "force_FD_Hessian_target_sign_eV_A2": float(force_derivative),
                "force_FD_Hessian_abs_error_eV_A2": abs(
                    float(force_derivative) - force_jacobian_target_eV_A2
                ),
                "left_force_slope_eV_A2": float(left),
                "right_force_slope_eV_A2": float(right),
                "left_right_slope_jump_eV_A2": abs(float(right - left)),
            }
        )
    for index, row in enumerate(rows):
        if index == 0:
            row.update(
                {
                    "empirical_order_energy_error": None,
                    "empirical_order_force_error": None,
                    "empirical_order_slope_jump": None,
                    "Richardson_force_FD_eV_A2": None,
                    "Richardson_force_abs_error_eV_A2": None,
                }
            )
            continue
        previous = rows[index - 1]
        richardson = row["force_FD_Hessian_target_sign_eV_A2"] + (
            row["force_FD_Hessian_target_sign_eV_A2"]
            - previous["force_FD_Hessian_target_sign_eV_A2"]
        ) / 3.0
        row.update(
            {
                "empirical_order_energy_error": _empirical_order(
                    previous["energy_FD_force_abs_error_eV_A"],
                    row["energy_FD_force_abs_error_eV_A"],
                ),
                "empirical_order_force_error": _empirical_order(
                    previous["force_FD_Hessian_abs_error_eV_A2"],
                    row["force_FD_Hessian_abs_error_eV_A2"],
                ),
                "empirical_order_slope_jump": _empirical_order(
                    previous["left_right_slope_jump_eV_A2"],
                    row["left_right_slope_jump_eV_A2"],
                ),
                "Richardson_force_FD_eV_A2": richardson,
                "Richardson_force_abs_error_eV_A2": abs(
                    richardson - force_jacobian_target_eV_A2
                ),
            }
        )
    selected_index = list(FD_STEPS_A).index(FD_ADJUDICATING_STEP_A)
    selected = rows[selected_index]
    return {
        "steps_A_in_fixed_order": [float(value) for value in steps_A],
        "evaluation_order_each_step": ["minus", "plus"],
        "adjudicating_step_A": FD_ADJUDICATING_STEP_A,
        "force_target_eV_A": force_target_eV_A,
        "force_jacobian_target_eV_A2": force_jacobian_target_eV_A2,
        "points": rows,
        "selected_force_abs_difference_eV_A": selected[
            "energy_FD_force_abs_error_eV_A"
        ],
        "selected_force_Hessian_abs_difference_eV_A2": selected[
            "force_FD_Hessian_abs_error_eV_A2"
        ],
        "multistep_diagnostic_role": "report_only_not_a_formal_numeric_gate",
    }


def _finite_difference_topology_exact(item: Mapping[str, Any]) -> bool:
    keys = (
        "formal_R2O_graph_semantic_sha256",
        "assignment_and_MIC_semantic_sha256",
    )
    try:
        baseline = item["base_point_provenance"]
        points = item["point_provenance"]
        return bool(
            len(points) == len(FD_STEPS_A)
            and all(float(point["h_A"]) == step for point, step in zip(points, FD_STEPS_A))
            and all(
                point[side][key] == baseline[key]
                for point in points
                for side in ("minus", "plus")
                for key in keys
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def run_mechanics(
    inputs: FormalInputs,
    output: Path,
    *,
    device: str,
    freeze_manifest: Path,
    authorization_marker: Path,
) -> dict[str, Any]:
    authorization = validate_execution_authorization(
        freeze_manifest, authorization_marker
    )
    input_hashes = validate_inputs(inputs)
    existing = _reject_path(output, "formal mechanics output", must_exist=False)
    if existing.is_dir() and (existing / "DONE").exists():
        recovered, recovered_arrays = _load_completed_mechanics(existing)
        recovered_gate = recompute_mechanics_gate(recovered, recovered_arrays)
        if recovered_gate["pass"] is not bool(recovered.get("mechanics_pass")):
            raise ValueError("completed mechanics recomputed gate changed")
        if recovered.get("input_sha256") != input_hashes:
            raise ValueError("completed mechanics input hashes differ from live inputs")
        if recovered.get("execution_authorization") != authorization:
            raise ValueError("completed mechanics authorization binding changed")
        return {**recovered, "completion_recovered_without_recompute": True}
    root = _fresh_run_root(output)
    try:
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("formal production requires float32-origin default dtype")
        if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1":
            raise ValueError("formal run requires TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1")
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("formal mechanics is restricted to a CUDA V100 node")
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        if "V100" not in device_name.upper():
            raise ValueError("formal mechanics must run on an NVIDIA V100")
        source_hashes = validate_frozen_sources()
        environment = runtime_fingerprint(device)
        model = load_endpoint(inputs, device)
        thermal = read_geometry_only_extxyz(inputs.thermal92)
        harmonic = read_geometry_only_extxyz(inputs.harmonic_zero32)
        _validate_structure_order(thermal, harmonic)
        reference6 = read(inputs.reference_6x6, index=0)
        reference8 = read(inputs.reference_8x8, index=0)
        probe = thermal[0]
        thresholds = r2r.CANONICAL_CONTRACT["fixed_gates"]
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        stage_receipts = []
        stage_started = time.perf_counter()

        reference_mechanics = r2r.production_combined_energy_force_hessian(
            model, reference6, reference6, device=device
        )
        reference_energy = float(reference_mechanics.energy_eV.detach().cpu())
        reference_hessian = np.asarray(
            reference_mechanics.Hessian_source_order_eV_A2.detach().cpu(),
            dtype=np.float64,
        ).copy()
        reference_force = np.asarray(
            reference_mechanics.force_source_order_eV_A.detach().cpu(), dtype=np.float64
        ).copy()
        reference_query_receipt = reference_mechanics.query_receipt
        del reference_mechanics
        gc.collect()
        torch.cuda.empty_cache()
        stage_receipts.append(
            {
                "stage": "canonical_combined_reference_full_H",
                "elapsed_seconds": time.perf_counter() - stage_started,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        stage_started = time.perf_counter()
        probe_mechanics = r2r.production_combined_energy_force_hessian(
            model, probe, reference6, device=device
        )
        base_energy = float(probe_mechanics.energy_eV.detach().cpu())
        probe_hessian = np.asarray(
            probe_mechanics.Hessian_source_order_eV_A2.detach().cpu(), dtype=np.float64
        ).copy()
        probe_force = np.asarray(
            probe_mechanics.force_source_order_eV_A.detach().cpu(), dtype=np.float64
        ).copy()
        probe_query_receipt = probe_mechanics.query_receipt
        del probe_mechanics
        gc.collect()
        torch.cuda.empty_cache()
        stage_receipts.append(
            {
                "stage": "canonical_combined_thermal0_full_H",
                "elapsed_seconds": time.perf_counter() - stage_started,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        stage_started = time.perf_counter()
        mass_weighted = 0.5 * (reference_hessian + reference_hessian.T) / 12.011
        maximum_absolute_eigenvalue = float(
            np.max(np.abs(np.linalg.eigvalsh(mass_weighted)))
        )
        frequency_bound = float(
            r2o_evaluator.CM1_PER_SQRT_EV_A2_AMU
            * math.sqrt(maximum_absolute_eigenvalue)
        )
        reference_gate = {
            "energy_abs_eV": abs(reference_energy),
            "force_max_abs_eV_A": float(np.max(np.abs(reference_force))),
            "Hessian_max_abs_eV_A2": float(np.max(np.abs(reference_hessian))),
            "Hessian_antisymmetry_max_abs_eV_A2": float(
                np.max(np.abs(reference_hessian - reference_hessian.T))
            ),
            "Hessian_translation_ASR_max_abs_eV_A2": float(
                np.max(np.abs(reference_hessian.reshape(72, 3, 72, 3).sum(axis=2)))
            ),
            "Weyl_Gamma_K_drift_upper_bound_cm-1": frequency_bound,
            "frequency_interpretation": (
                "Weyl upper bound from full mass-weighted residual Hessian; "
                "not an actual phonon frequency calculation"
            ),
            "frequency_helper_source_sha256": source_hashes["R2O_evaluator"],
            "query_receipt": reference_query_receipt,
        }
        carrier_parity = {
            "reference_6x6": _corrected_carrier_parity(
                model, reference6, reference6, device
            ),
            "thermal92_global_index0": _corrected_carrier_parity(
                model, probe, reference6, device
            ),
        }
        stage_receipts.append(
            {
                "stage": "corrected_carrier_vs_frozen_R2Q_reference_and_thermal0_full_H",
                "elapsed_seconds": time.perf_counter() - stage_started,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )

        generator = np.random.default_rng(83)
        proper, _ = np.linalg.qr(generator.normal(size=(3, 3)))
        if np.linalg.det(proper) < 0:
            proper[:, 0] *= -1
        improper = proper.copy()
        improper[:, 0] *= -1
        o3 = {}
        for name, transformation in (("proper", proper), ("improper", improper)):
            transformed_reference = reference6.copy()
            transformed_reference.positions = np.asarray(reference6.positions) @ transformation.T
            transformed_reference.set_cell(
                np.asarray(reference6.cell) @ transformation.T, scale_atoms=False
            )
            transformed_probe = probe.copy()
            transformed_probe.positions = np.asarray(probe.positions) @ transformation.T
            transformed_probe.set_cell(
                np.asarray(probe.cell) @ transformation.T, scale_atoms=False
            )
            energy, force, query_receipt = _combined_ef(
                model,
                transformed_probe,
                transformed_reference,
                device,
                graph_mode="rigid_transform_probe",
                baseline_reference_template=reference6,
                baseline_structure_template=probe,
                rigid_transform=transformation,
            )
            rigid_receipt = query_receipt.get("rigid_transform_receipt", {})
            frozen_o3 = FORMAL_CONTRACT["O3_rigid_transform_probe_v4"]
            if rigid_receipt.get(
                "baseline_structure_full_semantic_sha256"
            ) != frozen_o3["baseline_structure_full_semantic_sha256"] or (
                structure_semantic_sha256(probe)
                != frozen_o3["baseline_structure_full_semantic_sha256"]
            ):
                raise ValueError("O3 receipt baseline structure identity changed")
            if rigid_receipt.get(
                "baseline_reference_full_semantic_sha256"
            ) != frozen_o3["baseline_reference_full_semantic_sha256"] or (
                structure_semantic_sha256(reference6)
                != frozen_o3["baseline_reference_full_semantic_sha256"]
            ):
                raise ValueError("O3 receipt baseline reference identity changed")
            native_rebuild_diagnostic = (
                r2r.production_rigid_transform_native_rebuild_diagnostic(
                    model,
                    transformed_probe,
                    transformed_reference,
                    baseline_reference_template=reference6,
                    baseline_structure_template=probe,
                    rigid_transform=transformation,
                    device=device,
                )
            )
            o3[name] = {
                "matrix": transformation.tolist(),
                "determinant": float(np.linalg.det(transformation)),
                "energy_abs_difference_eV": abs(energy - base_energy),
                "force_covariance_max_abs_difference_eV_A": float(
                    np.max(np.abs(force - probe_force @ transformation.T))
                ),
                "query_receipt": query_receipt,
                "native_rebuild_combined_EF_diagnostic": (
                    native_rebuild_diagnostic
                ),
            }

        translation = np.asarray([0.031, -0.027, 0.019], dtype=np.float64)
        translated = probe.copy()
        translated.positions = np.asarray(probe.positions) + translation
        translated_energy, translated_force, _ = _combined_ef(
            model, translated, reference6, device
        )
        permutation = generator.permutation(72)
        permuted = probe[permutation]
        permuted.set_cell(probe.cell, scale_atoms=False)
        permuted.pbc = probe.pbc
        permuted_energy, permuted_force, _ = _combined_ef(
            model, permuted, reference6, device
        )
        native_wrapped = probe.copy()
        native_wrapped.positions[5] += np.asarray(native_wrapped.cell[0])
        native_wrapped_energy, native_wrapped_force, _ = _combined_ef(
            model, native_wrapped, reference6, device
        )
        base_design = r2r.production_linear_design_query(
            model, probe, reference6, device=device
        )
        wrapped_design = r2r.production_linear_design_query(
            model, native_wrapped, reference6, device=device
        )
        native_wrap_diagnostic = r2r.native_cell_wrap_order_mic_receipt(
            base_design, wrapped_design
        )

        coordinate = (2, 1)
        flat_coordinate = 3 * coordinate[0] + coordinate[1]
        finite_difference_steps = np.asarray(FD_STEPS_A, dtype="<f8")
        finite_difference_energy = np.empty((len(FD_STEPS_A), 2), dtype="<f8")
        finite_difference_coordinate_force = np.empty(
            (len(FD_STEPS_A), 2), dtype="<f8"
        )
        finite_difference_point_provenance = []
        provenance_keys = (
            "formal_R2O_graph_semantic_sha256",
            "assignment_and_MIC_semantic_sha256",
        )
        base_point_provenance = {
            key: probe_query_receipt[key] for key in provenance_keys
        }
        for step_index, step in enumerate(FD_STEPS_A):
            point = {"h_A": step}
            for sign_index, (side, sign) in enumerate(
                (("minus", -1.0), ("plus", 1.0))
            ):
                displaced = probe.copy()
                displaced.positions[coordinate] += sign * step
                energy, force, query_receipt = _combined_ef(
                    model, displaced, reference6, device
                )
                finite_difference_energy[step_index, sign_index] = energy
                finite_difference_coordinate_force[step_index, sign_index] = float(
                    force[coordinate]
                )
                point[side] = {key: query_receipt[key] for key in provenance_keys}
            finite_difference_point_provenance.append(point)
        finite_difference = _analyze_formal_finite_difference(
            finite_difference_steps,
            finite_difference_energy,
            finite_difference_coordinate_force,
            force_target_eV_A=float(probe_force[coordinate]),
            force_jacobian_target_eV_A2=float(
                -probe_hessian[flat_coordinate, flat_coordinate]
            ),
        )
        finite_difference.update(
            {
                "coordinate": list(coordinate),
                "base_point_provenance": base_point_provenance,
                "point_provenance": finite_difference_point_provenance,
                "all_point_graph_and_assignment_topology_exact": False,
            }
        )
        finite_difference["all_point_graph_and_assignment_topology_exact"] = (
            _finite_difference_topology_exact(finite_difference)
        )

        localized = {}
        for count, reference in ((72, reference6), (128, reference8)):
            item = reference.copy()
            item.positions[0, 2] += 0.03
            energy, force, _ = _combined_ef(model, item, reference, device)
            localized[count] = {"energy_eV": energy, "central_force_eV_A": force[0]}
            localized[count]["force_eV_A"] = force
        site_mapping = localized_reference_site_mapping(reference6, reference8)
        index6 = np.asarray(
            [pair[0] for pair in site_mapping["common_6x6_to_8x8"]], dtype=int
        )
        index8 = np.asarray(
            [pair[1] for pair in site_mapping["common_6x6_to_8x8"]], dtype=int
        )
        extra8 = np.asarray(site_mapping["extra_8x8_indices"], dtype=int)
        mapped_force_difference = float(
            np.max(
                np.abs(
                    localized[72]["force_eV_A"][index6]
                    - localized[128]["force_eV_A"][index8]
                )
            )
        )
        extra_force_max = float(
            np.max(np.abs(localized[128]["force_eV_A"][extra8]))
        )

        node_parity = {
            "reference_6x6": _node_parity(model, reference6, reference6, device),
            "thermal92_global_index0": _node_parity(model, probe, reference6, device),
        }
        q_snapshots = {}
        q_arrays = {}
        zero_jet = {}
        for reference_name, reference in (
            ("reference_6x6", reference6),
            ("reference_8x8", reference8),
        ):
            for device_tag, q_device in (("CPU", "cpu"), ("CUDA", device)):
                snapshot, snapshot_arrays = _capture_formal_q_snapshot(
                    reference, reference_name, q_device
                )
                snapshot_name = f"{reference_name}_{device_tag}"
                zero_jet[snapshot_name] = snapshot.pop("zero_jet")
                q_snapshots[snapshot_name] = snapshot
                for name, value in snapshot_arrays.items():
                    q_arrays[_q_array_key(reference_name, device_tag, name)] = value
        background_cutoff = r2r.quintic_inside_cutoff_metrics()
        mace_cutoff = r2o.cutoff_c2_metrics(model)
        mechanics = {
            "reference": reference_gate,
            "corrected_carrier_vs_frozen_R2Q": carrier_parity,
            "nonreference_complete_Hessian": {
                "antisymmetry_max_abs_eV_A2": float(
                    np.max(np.abs(probe_hessian - probe_hessian.T))
                ),
                "translation_ASR_max_abs_eV_A2": float(
                    np.max(np.abs(probe_hessian.reshape(72, 3, 72, 3).sum(axis=2)))
                ),
                "query_receipt": probe_query_receipt,
            },
            "O3": o3,
            "translation": {
                "vector_A": translation.tolist(),
                "energy_abs_difference_eV": abs(translated_energy - base_energy),
                "force_max_abs_difference_eV_A": float(
                    np.max(np.abs(translated_force - probe_force))
                ),
            },
            "pure_permutation": {
                "permutation_sha256": r2r.semantic_sha256(permutation.tolist()),
                "energy_abs_difference_eV": abs(permuted_energy - base_energy),
                "force_max_abs_difference_eV_A": float(
                    np.max(np.abs(permuted_force - probe_force[permutation]))
                ),
            },
            "native_wrap_order_MIC": {
                "energy_abs_difference_eV": abs(
                    native_wrapped_energy - base_energy
                ),
                "force_max_abs_difference_eV_A": float(
                    np.max(np.abs(native_wrapped_force - probe_force))
                ),
                "raw_65_design_diagnostic": native_wrap_diagnostic,
            },
            "finite_difference": finite_difference,
            "localized_6x6_to_8x8": {
                "displacement_A": 0.03,
                "total_energy_abs_difference_eV": abs(
                    localized[72]["energy_eV"] - localized[128]["energy_eV"]
                ),
                "reference_site_mapping": site_mapping,
                "mapped_common_force_max_abs_difference_eV_A": mapped_force_difference,
                "extra_8x8_force_vs_zero_max_abs_eV_A": extra_force_max,
                "full_system_max_force_abs_difference_eV_A": float(
                    max(mapped_force_difference, extra_force_max)
                ),
            },
            "locality_no_wrap": {
                "reference_6x6_shortest_translation_A": float(
                    r2r.fixed_reference_neighborhood(reference6).shortest_in_plane_translation_A
                ),
                "reference_8x8_shortest_translation_A": float(
                    r2r.fixed_reference_neighborhood(reference8).shortest_in_plane_translation_A
                ),
                "interaction_diameter_A": r2r.BACKGROUND_INTERACTION_DIAMETER_A,
                "all_reference_neighborhoods_validated_unique_sender": True,
            },
            "node_parity": node_parity,
            "q_snapshots": q_snapshots,
            "zero_jet": zero_jet,
            "background_6A_quintic_C2": background_cutoff,
            "MACE_r3p2_PolynomialCutoff_C2": mace_cutoff,
        }
        q_portability = _recompute_q_portability_gate(mechanics, q_arrays)
        mechanics["q_portability"] = q_portability
        ref_limit = thresholds["reference_Taylor_remainder"]
        node_limit = thresholds["R2Q_node_parity"]
        mechanics["gate_checks"] = {
            "reference": (
                all(
                    reference_gate[key] <= ref_limit[key]
                    for key in (
                        "energy_abs_eV",
                        "force_max_abs_eV_A",
                        "Hessian_max_abs_eV_A2",
                        "Hessian_antisymmetry_max_abs_eV_A2",
                        "Hessian_translation_ASR_max_abs_eV_A2",
                    )
                )
                and reference_gate["Weyl_Gamma_K_drift_upper_bound_cm-1"]
                <= ref_limit["Gamma_K_frequency_drift_cm-1"]
            ),
            "corrected_carrier_parity": all(
                item["energy_abs_difference_eV"]
                <= thresholds["corrected_carrier_vs_frozen_R2Q_parity"]["energy_abs_eV"]
                and item["force_max_abs_difference_eV_A"]
                <= thresholds["corrected_carrier_vs_frozen_R2Q_parity"]["force_max_abs_eV_A"]
                and item["Hessian_max_abs_difference_eV_A2"]
                <= thresholds["corrected_carrier_vs_frozen_R2Q_parity"]["Hessian_max_abs_eV_A2"]
                for item in carrier_parity.values()
            ),
            "node_parity": all(
                item["energy_sum_abs_difference_eV"] <= node_limit["energy_sum_abs_eV"]
                and item["position_gradient_max_abs_difference_eV_A"]
                <= node_limit["position_gradient_max_abs_eV_A"]
                for item in node_parity.values()
            ),
            "O3": all(
                item["energy_abs_difference_eV"]
                <= thresholds["O3_proper_and_improper"]["energy_abs_eV"]
                and item["force_covariance_max_abs_difference_eV_A"]
                <= thresholds["O3_proper_and_improper"]["force_covariance_max_abs_eV_A"]
                and _o3_covariant_provenance_ok(item)
                for item in o3.values()
            ),
            "translation": (
                mechanics["translation"]["energy_abs_difference_eV"]
                <= thresholds["translation"]["energy_abs_eV"]
                and mechanics["translation"]["force_max_abs_difference_eV_A"]
                <= thresholds["translation"]["force_max_abs_eV_A"]
            ),
            "permutation_MIC": (
                mechanics["pure_permutation"]["energy_abs_difference_eV"]
                <= thresholds["permutation"]["energy_abs_eV"]
                and mechanics["pure_permutation"]["force_max_abs_difference_eV_A"]
                <= thresholds["permutation"]["force_max_abs_eV_A"]
            ),
            "native_wrap_order_MIC": (
                mechanics["native_wrap_order_MIC"]["energy_abs_difference_eV"]
                <= thresholds["native_cell_wrap_order_MIC_combined_probe"]["energy_abs_eV"]
                and mechanics["native_wrap_order_MIC"]["force_max_abs_difference_eV_A"]
                <= thresholds["native_cell_wrap_order_MIC_combined_probe"]["force_max_abs_eV_A"]
                and mechanics["native_wrap_order_MIC"]["raw_65_design_diagnostic"]["pass"]
            ),
            "finite_difference": (
                mechanics["finite_difference"]["selected_force_abs_difference_eV_A"]
                <= thresholds["finite_difference"]["force_abs_eV_A"]
                and mechanics["finite_difference"]["selected_force_Hessian_abs_difference_eV_A2"]
                <= thresholds["finite_difference"]["force_Hessian_abs_eV_A2"]
                and mechanics["finite_difference"][
                    "all_point_graph_and_assignment_topology_exact"
                ]
            ),
            "nonreference_Hessian": (
                mechanics["nonreference_complete_Hessian"]["antisymmetry_max_abs_eV_A2"]
                <= thresholds["nonreference_complete_Hessian"]["antisymmetry_max_abs_eV_A2"]
                and mechanics["nonreference_complete_Hessian"]["translation_ASR_max_abs_eV_A2"]
                <= thresholds["nonreference_complete_Hessian"]["translation_ASR_max_abs_eV_A2"]
            ),
            "localized_size": (
                mechanics["localized_6x6_to_8x8"]["total_energy_abs_difference_eV"]
                <= thresholds["localized_6x6_to_8x8_full_remainder"]["total_energy_abs_eV"]
                and mechanics["localized_6x6_to_8x8"]["full_system_max_force_abs_difference_eV_A"]
                <= thresholds["localized_6x6_to_8x8_full_remainder"]["max_force_abs_eV_A"]
            ),
            "locality_no_wrap": (
                mechanics["locality_no_wrap"]["reference_6x6_shortest_translation_A"]
                > mechanics["locality_no_wrap"]["interaction_diameter_A"]
                and mechanics["locality_no_wrap"]["reference_8x8_shortest_translation_A"]
                > mechanics["locality_no_wrap"]["interaction_diameter_A"]
                and mechanics["locality_no_wrap"]["all_reference_neighborhoods_validated_unique_sender"]
            ),
            "zero_jet": q_portability["pass"],
            "background_6A_quintic_C2": (
                abs(background_cutoff["value"])
                <= thresholds["quintic_cutoff_inside_limit"]["value_abs"]
                and abs(background_cutoff["first_derivative_A-1"])
                <= thresholds["quintic_cutoff_inside_limit"]["first_derivative_abs_A-1"]
                and abs(background_cutoff["second_derivative_A-2"])
                <= thresholds["quintic_cutoff_inside_limit"]["second_derivative_abs_A-2"]
                and abs(background_cutoff["third_derivative_A-3"])
                >= thresholds["quintic_cutoff_inside_limit"]["third_derivative_abs_A-3_min"]
            ),
            "MACE_r3p2_PolynomialCutoff_C2": (
                abs(mace_cutoff["value"])
                <= thresholds["quintic_cutoff_inside_limit"]["value_abs"]
                and abs(mace_cutoff["first_derivative_A-1"])
                <= thresholds["quintic_cutoff_inside_limit"]["first_derivative_abs_A-1"]
                and abs(mace_cutoff["second_derivative_A-2"])
                <= thresholds["quintic_cutoff_inside_limit"]["second_derivative_abs_A-2"]
                and abs(mace_cutoff["third_derivative_A-3"])
                >= thresholds["quintic_cutoff_inside_limit"]["third_derivative_abs_A-3_min"]
            ),
        }
        mechanics_pass = all(mechanics["gate_checks"].values())
        arrays = {
            "reference_force_eV_A": reference_force.astype("<f8"),
            "reference_Hessian_eV_A2": reference_hessian.astype("<f8"),
            "thermal0_force_eV_A": probe_force.astype("<f8"),
            "thermal0_Hessian_eV_A2": probe_hessian.astype("<f8"),
            "finite_difference_steps_A": finite_difference_steps,
            "finite_difference_energy_eV": finite_difference_energy,
            "finite_difference_force_coordinate_eV_A": (
                finite_difference_coordinate_force
            ),
            **q_arrays,
        }
        arrays_path = root / "mechanics_arrays.npz"
        _atomic_npz(arrays_path, arrays)
        receipt = {
            "format": MECHANICS_FORMAT,
            "status": STATUS_MECHANICS,
            "formal_contract_sha256": FORMAL_CONTRACT_SHA256,
            "execution_authorization": authorization,
            "frozen_source_sha256": source_hashes,
            "input_sha256": input_hashes,
            "runtime_fingerprint": environment,
            "full_H_stage_receipts": stage_receipts,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "dtype": "torch.float64",
            "mechanics": mechanics,
            "mechanics_pass": mechanics_pass,
            "arrays_sha256": sha256(arrays_path),
            "array_schema": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
            "elapsed_seconds": time.perf_counter() - started,
            "force_labels_used": False,
            "energy_labels_used": False,
            "geometry_loader_isolation": dict(GEOMETRY_LOADER_ISOLATION),
            "can_authorize_fit_or_training": False,
        }
        _validate_receipt_contract(
            receipt, "fresh formal mechanics", receipt_kind="mechanics"
        )
        receipt_path = root / "receipt.json"
        _atomic_json(receipt_path, receipt)
        _success(root, receipt_path, STATUS_MECHANICS)
        return receipt
    except BaseException as exception:
        _failure(root, exception)
        raise


def _mechanics_expected_array_schema() -> dict[str, tuple[tuple[int, ...], str]]:
    expected: dict[str, tuple[tuple[int, ...], str]] = {
        "reference_force_eV_A": ((72, 3), "<f8"),
        "reference_Hessian_eV_A2": ((216, 216), "<f8"),
        "thermal0_force_eV_A": ((72, 3), "<f8"),
        "thermal0_Hessian_eV_A2": ((216, 216), "<f8"),
        "finite_difference_steps_A": ((len(FD_STEPS_A),), "<f8"),
        "finite_difference_energy_eV": ((len(FD_STEPS_A), 2), "<f8"),
        "finite_difference_force_coordinate_eV_A": (
            (len(FD_STEPS_A), 2),
            "<f8",
        ),
    }
    for reference_name, atom_count, edge_count in (
        ("reference_6x6", 72, 2808),
        ("reference_8x8", 128, 4992),
    ):
        shapes = {
            "receiver": (edge_count,),
            "sender": (edge_count,),
            "image_integer": (edge_count, 3),
            "local_source_order_offsets": (atom_count + 1,),
            "local_source_order_sender": (edge_count,),
            "local_source_order_edge_index": (edge_count,),
            "reference_distance_A": (edge_count,),
            "weight": (edge_count,),
            "normalization": (atom_count,),
            "center_q": (atom_count,),
            "neighbor_q": (edge_count,),
        }
        for device_tag in ("CPU", "CUDA"):
            for name, shape in shapes.items():
                dtype = "<i8" if name in Q_DISCRETE_ARRAY_NAMES else "<f8"
                expected[_q_array_key(reference_name, device_tag, name)] = (
                    shape,
                    dtype,
                )
    return expected


def _load_completed_mechanics(
    root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = _reject_path(root, "formal mechanics run")
    _validate_success_terminal_state(path, "formal mechanics")
    receipt_path = path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format") != MECHANICS_FORMAT or receipt.get("status") != STATUS_MECHANICS:
        raise ValueError("formal mechanics receipt format/status changed")
    _validate_live_receipt_provenance(receipt, "formal mechanics")
    _validate_receipt_contract(
        receipt, "formal mechanics", receipt_kind="mechanics"
    )
    if (path / "DONE").read_bytes() != (
        STATUS_MECHANICS + "\n" + sha256(receipt_path) + "\n"
    ).encode("ascii"):
        raise ValueError("formal mechanics DONE marker changed")
    arrays_path = _reject_path(path / "mechanics_arrays.npz", "formal mechanics arrays")
    if sha256(arrays_path) != receipt.get("arrays_sha256"):
        raise ValueError("formal mechanics arrays hash changed")
    with np.load(arrays_path, allow_pickle=False) as loaded:
        expected_schema = _mechanics_expected_array_schema()
        if set(loaded.files) != set(expected_schema):
            raise ValueError("formal mechanics canonical array key set changed")
        arrays = {key: loaded[key].copy() for key in loaded.files}
        schema = {
            key: {"shape": list(loaded[key].shape), "dtype": str(loaded[key].dtype)}
            for key in loaded.files
        }
        for key, (shape, dtype) in expected_schema.items():
            if loaded[key].shape != shape or loaded[key].dtype != np.dtype(dtype):
                raise ValueError(f"formal mechanics canonical shape/dtype changed: {key}")
            if np.issubdtype(loaded[key].dtype, np.floating) and not np.all(
                np.isfinite(loaded[key])
            ):
                raise ValueError(f"formal mechanics nonfinite array: {key}")
    if schema != receipt.get("array_schema"):
        raise ValueError("formal mechanics array schema changed")
    return receipt, arrays


def recompute_mechanics_gate(
    receipt: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Recompute every mechanics decision from numeric receipt fields."""
    mechanics = receipt["mechanics"]
    thresholds = r2r.CANONICAL_CONTRACT["fixed_gates"]
    reference = mechanics["reference"]
    reference_limit = thresholds["reference_Taylor_remainder"]
    node_limit = thresholds["R2Q_node_parity"]
    carrier_limit = thresholds["corrected_carrier_vs_frozen_R2Q_parity"]
    fd_steps = arrays["finite_difference_steps_A"]
    fd_grid_exact = bool(
        np.array_equal(fd_steps, np.asarray(FD_STEPS_A, dtype="<f8"))
    )
    fd_recomputed = _analyze_formal_finite_difference(
        fd_steps,
        arrays["finite_difference_energy_eV"],
        arrays["finite_difference_force_coordinate_eV_A"],
        force_target_eV_A=float(arrays["thermal0_force_eV_A"][2, 1]),
        force_jacobian_target_eV_A2=float(
            -arrays["thermal0_Hessian_eV_A2"][7, 7]
        ),
    )
    q_recomputed = _recompute_q_portability_gate(mechanics, arrays)
    checks = {
        "reference": (
            reference["energy_abs_eV"] <= reference_limit["energy_abs_eV"]
            and reference["force_max_abs_eV_A"] <= reference_limit["force_max_abs_eV_A"]
            and reference["Hessian_max_abs_eV_A2"] <= reference_limit["Hessian_max_abs_eV_A2"]
            and reference["Hessian_antisymmetry_max_abs_eV_A2"] <= reference_limit["Hessian_antisymmetry_max_abs_eV_A2"]
            and reference["Hessian_translation_ASR_max_abs_eV_A2"] <= reference_limit["Hessian_translation_ASR_max_abs_eV_A2"]
            and reference["Weyl_Gamma_K_drift_upper_bound_cm-1"] <= reference_limit["Gamma_K_frequency_drift_cm-1"]
        ),
        "carrier_parity": all(
            item["energy_abs_difference_eV"] <= carrier_limit["energy_abs_eV"]
            and item["force_max_abs_difference_eV_A"] <= carrier_limit["force_max_abs_eV_A"]
            and item["Hessian_max_abs_difference_eV_A2"] <= carrier_limit["Hessian_max_abs_eV_A2"]
            for item in mechanics["corrected_carrier_vs_frozen_R2Q"].values()
        ),
        "node_parity": all(
            item["energy_sum_abs_difference_eV"] <= node_limit["energy_sum_abs_eV"]
            and item["position_gradient_max_abs_difference_eV_A"] <= node_limit["position_gradient_max_abs_eV_A"]
            and item["signed_l0_shape"][1] == 32
            for item in mechanics["node_parity"].values()
        ),
        "O3": all(
            item["energy_abs_difference_eV"] <= thresholds["O3_proper_and_improper"]["energy_abs_eV"]
            and item["force_covariance_max_abs_difference_eV_A"] <= thresholds["O3_proper_and_improper"]["force_covariance_max_abs_eV_A"]
            and _o3_covariant_provenance_ok(item)
            for item in mechanics["O3"].values()
        ),
        "translation": (
            mechanics["translation"]["energy_abs_difference_eV"] <= thresholds["translation"]["energy_abs_eV"]
            and mechanics["translation"]["force_max_abs_difference_eV_A"] <= thresholds["translation"]["force_max_abs_eV_A"]
        ),
        "pure_permutation": (
            mechanics["pure_permutation"]["energy_abs_difference_eV"] <= thresholds["permutation"]["energy_abs_eV"]
            and mechanics["pure_permutation"]["force_max_abs_difference_eV_A"] <= thresholds["permutation"]["force_max_abs_eV_A"]
        ),
        "native_wrap_order_MIC": (
            mechanics["native_wrap_order_MIC"]["energy_abs_difference_eV"] <= thresholds["native_cell_wrap_order_MIC_combined_probe"]["energy_abs_eV"]
            and mechanics["native_wrap_order_MIC"]["force_max_abs_difference_eV_A"] <= thresholds["native_cell_wrap_order_MIC_combined_probe"]["force_max_abs_eV_A"]
        ),
        "finite_difference": (
            fd_grid_exact
            and mechanics["finite_difference"].get("coordinate") == [2, 1]
            and mechanics["finite_difference"].get("adjudicating_step_A")
            == FD_ADJUDICATING_STEP_A
            and _finite_difference_topology_exact(mechanics["finite_difference"])
            and fd_recomputed["selected_force_abs_difference_eV_A"]
            <= thresholds["finite_difference"]["force_abs_eV_A"]
            and fd_recomputed["selected_force_Hessian_abs_difference_eV_A2"]
            <= thresholds["finite_difference"]["force_Hessian_abs_eV_A2"]
        ),
        "nonreference_Hessian": (
            mechanics["nonreference_complete_Hessian"]["antisymmetry_max_abs_eV_A2"] <= thresholds["nonreference_complete_Hessian"]["antisymmetry_max_abs_eV_A2"]
            and mechanics["nonreference_complete_Hessian"]["translation_ASR_max_abs_eV_A2"] <= thresholds["nonreference_complete_Hessian"]["translation_ASR_max_abs_eV_A2"]
        ),
        "localized_size": (
            mechanics["localized_6x6_to_8x8"]["reference_site_mapping"]["common_count"] == 72
            and mechanics["localized_6x6_to_8x8"]["reference_site_mapping"]["only_6x6_count"] == 0
            and mechanics["localized_6x6_to_8x8"]["reference_site_mapping"]["only_8x8_count"] == 56
            and mechanics["localized_6x6_to_8x8"]["reference_site_mapping"]["semantic_sha256"]
            == FORMAL_CONTRACT["localized_reference_site_mapping"]["semantic_sha256"]
            and mechanics["localized_6x6_to_8x8"]["total_energy_abs_difference_eV"] <= thresholds["localized_6x6_to_8x8_full_remainder"]["total_energy_abs_eV"]
            and mechanics["localized_6x6_to_8x8"]["full_system_max_force_abs_difference_eV_A"] <= thresholds["localized_6x6_to_8x8_full_remainder"]["max_force_abs_eV_A"]
        ),
        "locality_no_wrap": (
            mechanics["locality_no_wrap"]["reference_6x6_shortest_translation_A"] > mechanics["locality_no_wrap"]["interaction_diameter_A"]
            and mechanics["locality_no_wrap"]["reference_8x8_shortest_translation_A"] > mechanics["locality_no_wrap"]["interaction_diameter_A"]
            and mechanics["locality_no_wrap"]["all_reference_neighborhoods_validated_unique_sender"] is True
        ),
        "zero_jet": q_recomputed["pass"],
        **{
            name: (
                abs(mechanics[name]["value"]) <= thresholds["quintic_cutoff_inside_limit"]["value_abs"]
                and abs(mechanics[name]["first_derivative_A-1"]) <= thresholds["quintic_cutoff_inside_limit"]["first_derivative_abs_A-1"]
                and abs(mechanics[name]["second_derivative_A-2"]) <= thresholds["quintic_cutoff_inside_limit"]["second_derivative_abs_A-2"]
                and abs(mechanics[name]["third_derivative_A-3"]) >= thresholds["quintic_cutoff_inside_limit"]["third_derivative_abs_A-3_min"]
            )
            for name in ("background_6A_quintic_C2", "MACE_r3p2_PolynomialCutoff_C2")
        },
    }
    return {
        "checks": checks,
        "finite_difference_recomputed": fd_recomputed,
        "finite_difference_grid_exact": fd_grid_exact,
        "q_portability_recomputed": q_recomputed,
        "pass": all(checks.values()),
    }


def _validate_aggregate_input_bindings(
    loaded: Sequence[tuple[dict, dict[str, np.ndarray]]],
    mechanics_receipt: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if len(loaded) != 3:
        raise ValueError("formal aggregate requires exactly three loaded shards")
    for shard_receipt, shard_arrays in loaded:
        _validate_shard_array_contract(shard_receipt, shard_arrays)
        _validate_receipt_contract(
            shard_receipt, "formal aggregate shard input", receipt_kind="shard"
        )
    receipts = [item[0] for item in loaded]
    if [item["shard_id"] for item in receipts] != [0, 1, 2]:
        raise ValueError("formal shard ids/order changed")
    invariant_keys = (
        "formal_contract_sha256",
        "frozen_source_sha256",
        "frozen_R2R_canonical_sha256",
        "input_sha256",
        "sentinel_contract_sha256",
    )
    for key in invariant_keys:
        if any(item[key] != receipts[0][key] for item in receipts[1:]):
            raise ValueError(f"formal shard provenance differs: {key}")
    if any(item.get("execution_authorization") != authorization for item in receipts):
        raise ValueError("formal shard authorization differs from current GO binding")
    if mechanics_receipt.get("formal_contract_sha256") != FORMAL_CONTRACT_SHA256:
        raise ValueError("formal mechanics contract binding changed")
    if mechanics_receipt.get("input_sha256") != receipts[0]["input_sha256"]:
        raise ValueError("formal mechanics and shards use different inputs")
    if mechanics_receipt.get("frozen_source_sha256") != receipts[0]["frozen_source_sha256"]:
        raise ValueError("formal mechanics and shards use different frozen sources")
    if mechanics_receipt.get("execution_authorization") != authorization:
        raise ValueError("formal mechanics authorization differs from current GO binding")
    _validate_receipt_contract(
        mechanics_receipt,
        "formal aggregate mechanics input",
        receipt_kind="mechanics",
    )
    return receipts


def _recompute_aggregate_from_loaded(
    loaded: Sequence[tuple[dict, dict[str, np.ndarray]]],
    mechanics_receipt: Mapping[str, Any],
    mechanics_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Rebuild every aggregate scientific field from shard/mechanics artifacts."""
    receipts = [item[0] for item in loaded]
    configuration_identity = _validate_partition_receipts(loaded)
    sentinel = compare_sentinels(loaded)
    environment_hashes = [
        item["runtime_fingerprint"]["semantic_sha256"] for item in receipts
    ]
    environment_mismatch = len(set(environment_hashes)) != 1
    mechanics_recomputed = recompute_mechanics_gate(
        mechanics_receipt, mechanics_arrays
    )
    thermal_indices = np.concatenate(
        [item[1]["thermal_global_index"] for item in loaded]
    )
    harmonic_indices = np.concatenate(
        [item[1]["harmonic_global_index"] for item in loaded]
    )
    thermal_order = np.argsort(thermal_indices)
    harmonic_order = np.argsort(harmonic_indices)
    if not np.array_equal(thermal_indices[thermal_order], np.arange(92)):
        raise ValueError("thermal shards overlap, omit, duplicate, or reorder configs")
    if not np.array_equal(harmonic_indices[harmonic_order], np.arange(32)):
        raise ValueError("harmonic shards overlap, omit, duplicate, or reorder configs")

    def ordered(role: str, name: str, order: np.ndarray) -> np.ndarray:
        return np.concatenate([item[1][f"{role}_{name}"] for item in loaded])[order]

    parameter_force = ordered(
        "thermal", "parameter_force_design_eV_A", thermal_order
    )
    rank = r2r.rank_condition_precheck(parameter_force.reshape(-1, 65))
    thermal_b = ordered("thermal", "b_A4", thermal_order)
    thermal_a = ordered("thermal", "a", thermal_order)
    harmonic_b = ordered("harmonic", "b_A4", harmonic_order)
    harmonic_a = ordered("harmonic", "a", harmonic_order)
    harmonic_energy = ordered("harmonic", "combined_energy_eV", harmonic_order)
    harmonic_force = ordered("harmonic", "combined_force_eV_A", harmonic_order)
    harmonic_fixed_energy = ordered(
        "harmonic", "fixed_carrier_energy_eV", harmonic_order
    )
    harmonic_fixed_force = ordered(
        "harmonic", "fixed_carrier_force_eV_A", harmonic_order
    )
    gates = r2r.CANONICAL_CONTRACT["fixed_gates"]
    train_gate = {
        "thermal_min_b_A4": float(np.min(thermal_b)),
        "thermal_min_a": float(np.min(thermal_a)),
        "harmonic_max_b_A4": float(np.max(harmonic_b)),
        "harmonic_max_a": float(np.max(harmonic_a)),
        "harmonic_max_abs_energy_eV": float(np.max(np.abs(harmonic_energy))),
        "harmonic_max_abs_force_eV_A": float(np.max(np.abs(harmonic_force))),
        "harmonic_fixed_carrier_max_abs_energy_eV": float(
            np.max(np.abs(harmonic_fixed_energy))
        ),
        "harmonic_fixed_carrier_max_abs_force_eV_A": float(
            np.max(np.abs(harmonic_fixed_force))
        ),
    }
    train_gate["pass"] = bool(
        train_gate["thermal_min_b_A4"] >= gates["thermal92"]["min_b_A4"]
        and train_gate["thermal_min_a"] >= gates["thermal92"]["min_a"]
        and train_gate["harmonic_max_b_A4"]
        <= gates["harmonic_zero32"]["max_b_A4"]
        and train_gate["harmonic_max_a"] <= gates["harmonic_zero32"]["max_a"]
        and train_gate["harmonic_max_abs_energy_eV"]
        <= gates["harmonic_zero32"]["max_Taylor_remainder_energy_eV"]
        and train_gate["harmonic_max_abs_force_eV_A"]
        <= gates["harmonic_zero32"]["max_Taylor_remainder_force_eV_A"]
        and train_gate["harmonic_fixed_carrier_max_abs_energy_eV"]
        <= gates["harmonic_zero32"]["max_Taylor_remainder_energy_eV"]
        and train_gate["harmonic_fixed_carrier_max_abs_force_eV_A"]
        <= gates["harmonic_zero32"]["max_Taylor_remainder_force_eV_A"]
    )
    finite_values = np.asarray(
        [
            train_gate["thermal_min_b_A4"],
            train_gate["thermal_min_a"],
            train_gate["harmonic_max_b_A4"],
            train_gate["harmonic_max_a"],
            train_gate["harmonic_max_abs_energy_eV"],
            train_gate["harmonic_max_abs_force_eV_A"],
            train_gate["harmonic_fixed_carrier_max_abs_energy_eV"],
            train_gate["harmonic_fixed_carrier_max_abs_force_eV_A"],
            rank["scaled_condition_number"]
            if rank["scaled_condition_number"] is not None
            else np.nan,
        ],
        dtype=np.float64,
    )
    threshold_pairs = (
        (train_gate["thermal_min_b_A4"], gates["thermal92"]["min_b_A4"]),
        (train_gate["thermal_min_a"], gates["thermal92"]["min_a"]),
        (train_gate["harmonic_max_b_A4"], gates["harmonic_zero32"]["max_b_A4"]),
        (train_gate["harmonic_max_a"], gates["harmonic_zero32"]["max_a"]),
        (
            train_gate["harmonic_max_abs_energy_eV"],
            gates["harmonic_zero32"]["max_Taylor_remainder_energy_eV"],
        ),
        (
            train_gate["harmonic_max_abs_force_eV_A"],
            gates["harmonic_zero32"]["max_Taylor_remainder_force_eV_A"],
        ),
        (
            train_gate["harmonic_fixed_carrier_max_abs_energy_eV"],
            gates["harmonic_zero32"]["max_Taylor_remainder_energy_eV"],
        ),
        (
            train_gate["harmonic_fixed_carrier_max_abs_force_eV_A"],
            gates["harmonic_zero32"]["max_Taylor_remainder_force_eV_A"],
        ),
        (
            rank["scaled_condition_number"] or math.inf,
            gates["thermal_force_design"]["scaled_condition_number_max"],
        ),
    )
    borderline = any(
        math.isfinite(value)
        and abs(value - threshold)
        <= BORDERLINE_RELATIVE_BAND
        * max(abs(threshold), np.finfo(float).tiny)
        for value, threshold in threshold_pairs
    )
    singular = rank["scaled_singular_values"]
    if singular and singular[-1] <= 10.0 * rank["rank_tolerance"]:
        borderline = True
    numerically_inconclusive = bool(
        not np.all(np.isfinite(finite_values)) or not sentinel["pass"] or borderline
    )
    passed = bool(
        not numerically_inconclusive
        and train_gate["pass"]
        and rank["R2R0_precheck_pass"]
        and mechanics_recomputed["pass"]
    )
    if numerically_inconclusive:
        status = STATUS_AGGREGATE_INCONCLUSIVE
    elif passed:
        status = STATUS_AGGREGATE_PASS
    else:
        status = STATUS_AGGREGATE_FAIL
    arrays = {
        "thermal_global_index": thermal_indices[thermal_order].astype("<i8"),
        "thermal_fixed_energy_eV": ordered("thermal", "fixed_energy_eV", thermal_order),
        "thermal_fixed_force_eV_A": ordered("thermal", "fixed_force_eV_A", thermal_order),
        "thermal_parameter_energy_design_eV": ordered(
            "thermal", "parameter_energy_design_eV", thermal_order
        ),
        "thermal_parameter_force_design_eV_A": parameter_force,
        "thermal_b_A4": thermal_b,
        "thermal_a": thermal_a,
        "thermal_c": ordered("thermal", "c", thermal_order),
        "thermal_combined_energy_eV": ordered(
            "thermal", "combined_energy_eV", thermal_order
        ),
        "thermal_combined_force_eV_A": ordered(
            "thermal", "combined_force_eV_A", thermal_order
        ),
        "harmonic_global_index": harmonic_indices[harmonic_order].astype("<i8"),
        "harmonic_b_A4": harmonic_b,
        "harmonic_a": harmonic_a,
        "harmonic_c": ordered("harmonic", "c", harmonic_order),
        "harmonic_fixed_carrier_energy_eV": harmonic_fixed_energy,
        "harmonic_fixed_carrier_force_eV_A": harmonic_fixed_force,
        "harmonic_combined_energy_eV": harmonic_energy,
        "harmonic_combined_force_eV_A": harmonic_force,
    }
    _validate_aggregate_array_contract(arrays)
    total_gpu_hours = float(
        (
            sum(float(item.get("elapsed_seconds", 0.0)) for item in receipts)
            + float(mechanics_receipt.get("elapsed_seconds", 0.0))
        )
        / 3600.0
    )
    return {
        "format": "graphene_r2r0_aggregate_recomputation_v1_from_role_arrays",
        "status": status,
        "arrays": arrays,
        "receipt_fields": {
            "sentinel": sentinel,
            "shard_environment_semantic_sha256": environment_hashes,
            "environment_mismatch": environment_mismatch,
            "environment_mismatch_merge_allowed": bool(
                not environment_mismatch or sentinel["pass"]
            ),
            "mechanics_recomputed_gate": mechanics_recomputed,
            "train_geometry_and_rank1_gate": train_gate,
            "rank_condition_precheck": rank,
            "thermal_partition_exact": True,
            "harmonic_partition_exact": True,
            "configuration_identity": configuration_identity,
            "representation_precheck_pass": passed,
            "numerically_inconclusive": numerically_inconclusive,
            "borderline": borderline,
            "formal_total_GPU_hours": total_gpu_hours,
            "aggregate_recomputation_format": (
                "graphene_r2r0_aggregate_recomputation_v1_from_role_arrays"
            ),
        },
    }


def _aggregate_arrays_exact(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> bool:
    return bool(
        set(left) == set(right)
        and all(
            np.asarray(left[key]).dtype == np.asarray(right[key]).dtype
            and np.asarray(left[key]).shape == np.asarray(right[key]).shape
            and np.array_equal(left[key], right[key])
            for key in left
        )
    )


def aggregate_shards(
    shard_roots: Sequence[Path],
    mechanics_root: Path,
    output: Path,
    *,
    freeze_manifest: Path,
    authorization_marker: Path,
) -> dict[str, Any]:
    if len(shard_roots) != 3:
        raise ValueError("aggregate requires exactly three formal shards")
    authorization = validate_execution_authorization(
        freeze_manifest, authorization_marker
    )
    existing = _reject_path(output, "formal aggregate output", must_exist=False)
    if existing.is_dir() and (existing / "DONE").exists():
        _validate_success_terminal_state(existing, "formal aggregate")
        receipt_path = existing / "receipt.json"
        recovered = json.loads(receipt_path.read_text(encoding="utf-8"))
        if recovered.get("format") != AGGREGATE_FORMAT or recovered.get("status") not in {
            STATUS_AGGREGATE_PASS,
            STATUS_AGGREGATE_FAIL,
            STATUS_AGGREGATE_INCONCLUSIVE,
        }:
            raise ValueError("completed aggregate format changed")
        _validate_live_receipt_provenance(recovered, "formal aggregate")
        _validate_receipt_contract(
            recovered, "completed formal aggregate", receipt_kind="aggregate"
        )
        if recovered.get("execution_authorization") != authorization:
            raise ValueError("completed aggregate authorization binding changed")
        expected_done = (
            recovered["status"] + "\n" + sha256(receipt_path) + "\n"
        ).encode("ascii")
        if (existing / "DONE").read_bytes() != expected_done:
            raise ValueError("completed aggregate DONE marker changed")
        arrays_path = existing / "arrays.npz"
        if sha256(arrays_path) != recovered.get("arrays_sha256"):
            raise ValueError("completed aggregate arrays changed")
        with np.load(arrays_path, allow_pickle=False) as loaded_arrays:
            recovered_arrays = {key: loaded_arrays[key].copy() for key in loaded_arrays.files}
            observed_schema = {
                key: {"shape": list(loaded_arrays[key].shape), "dtype": str(loaded_arrays[key].dtype)}
                for key in loaded_arrays.files
            }
        if observed_schema != recovered.get("array_schema"):
            raise ValueError("completed aggregate array schema changed")
        _validate_aggregate_array_contract(recovered_arrays)
        current_shards = [_load_completed_shard(path) for path in shard_roots]
        if [item[0]["shard_id"] for item in current_shards] != [0, 1, 2]:
            raise ValueError("aggregate recovery shard order changed")
        current_mechanics, current_mechanics_arrays = _load_completed_mechanics(
            mechanics_root
        )
        _validate_aggregate_input_bindings(
            current_shards, current_mechanics, authorization
        )
        current_recomputed = _recompute_aggregate_from_loaded(
            current_shards, current_mechanics, current_mechanics_arrays
        )
        if [sha256(Path(path) / "receipt.json") for path in shard_roots] != recovered.get("shard_receipt_sha256"):
            raise ValueError("aggregate recovery shard receipts changed")
        if [directory_manifest_sha256(path) for path in shard_roots] != recovered.get("shard_root_manifest_sha256"):
            raise ValueError("aggregate recovery shard artifact roots changed")
        if sha256(Path(mechanics_root) / "receipt.json") != recovered.get("mechanics_receipt_sha256"):
            raise ValueError("aggregate recovery mechanics receipt changed")
        if directory_manifest_sha256(mechanics_root) != recovered.get("mechanics_root_manifest_sha256"):
            raise ValueError("aggregate recovery mechanics artifact root changed")
        if not _aggregate_arrays_exact(
            recovered_arrays, current_recomputed["arrays"]
        ):
            raise ValueError(
                "completed aggregate arrays differ from current shard arrays"
            )
        if recovered.get("status") != current_recomputed["status"]:
            raise ValueError("completed aggregate recomputed status changed")
        if any(
            recovered.get(key) != value
            for key, value in current_recomputed["receipt_fields"].items()
        ):
            raise ValueError("completed aggregate recomputed decision fields changed")
        return {**recovered, "completion_recovered_without_recompute": True}
    root = _fresh_run_root(output)
    try:
        source_hashes = validate_frozen_sources()
        loaded = [
            _load_completed_shard(path) for path in shard_roots
        ]
        mechanics_receipt, mechanics_arrays = _load_completed_mechanics(mechanics_root)
        receipts = _validate_aggregate_input_bindings(
            loaded, mechanics_receipt, authorization
        )
        aggregate_recomputed = _recompute_aggregate_from_loaded(
            loaded, mechanics_receipt, mechanics_arrays
        )
        status = aggregate_recomputed["status"]
        arrays = aggregate_recomputed["arrays"]
        arrays_path = root / "arrays.npz"
        _atomic_npz(arrays_path, arrays)
        receipt = {
            "format": AGGREGATE_FORMAT,
            "status": status,
            "formal_contract_sha256": FORMAL_CONTRACT_SHA256,
            "execution_authorization": authorization,
            "frozen_source_sha256": source_hashes,
            "frozen_R2R_canonical_sha256": FROZEN_R2R_CANONICAL_SHA256,
            "shard_receipt_sha256": [sha256(Path(path) / "receipt.json") for path in shard_roots],
            "shard_root_manifest_sha256": [directory_manifest_sha256(path) for path in shard_roots],
            "input_sha256": receipts[0]["input_sha256"],
            "mechanics_receipt_sha256": sha256(Path(mechanics_root) / "receipt.json"),
            "mechanics_root_manifest_sha256": directory_manifest_sha256(mechanics_root),
            "array_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
            "arrays_sha256": sha256(arrays_path),
            **AGGREGATE_SAFETY_FIELDS,
            **aggregate_recomputed["receipt_fields"],
        }
        _validate_receipt_contract(
            receipt, "fresh formal aggregate", receipt_kind="aggregate"
        )
        receipt_path = root / "receipt.json"
        _atomic_json(receipt_path, receipt)
        _success(root, receipt_path, status)
        return receipt
    except BaseException as exception:
        _failure(root, exception)
        raise


__all__ = [
    "AGGREGATE_FORMAT",
    "FORMAL_CONTRACT",
    "FORMAL_CONTRACT_SHA256",
    "FormalInputs",
    "SENTINEL_CONTRACT",
    "SENTINEL_CONTRACT_SHA256",
    "SHARD_FORMAT",
    "SHARD_INDICES",
    "aggregate_shards",
    "compare_sentinels",
    "localized_reference_site_mapping",
    "load_endpoint",
    "run_shard",
    "run_mechanics",
    "run_bounded_preflight",
    "recompute_mechanics_gate",
    "read_geometry_only_extxyz",
    "runtime_fingerprint",
    "structure_semantic_sha256",
    "validate_frozen_sources",
    "validate_inputs",
]
