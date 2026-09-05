#!/usr/bin/env python3
"""Fail-closed R2O Taylor-null routed-tail primitives.

This module is deliberately separate from the historical R2M routed-tail
implementation.  It contains only the pieces that can be exercised before
``support9`` is opened:

* validation and loading of one *formal passing* portable R2O bundle;
* fixed-reference, order/MIC-safe raw MACE features in FP64;
* a conservative raw routed scalar followed by a whole-energy Taylor-2 null;
* the train8 prediction/target energy gauge used by outer LOCO; and
* a fixed-epoch, relative-path/hash outer-LOCO contract skeleton.

The MACE object inside an R2O deployment bundle is never returned as a direct
energy/force calculator.  It is frozen and may only be used through the fixed
reference graph and the live Taylor wrapper.  Discrete assignment and MIC
integers are detached; query positions, features, router, gate, node energy,
displacement and HVP ``grad_outputs`` remain live.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from mace.modules.utils import get_edge_vectors_and_lengths

from graphene_r2o_taylor_null import (
    EXPECTED_HIDDEN_LAYOUT,
    EXPECTED_NUM_INTERACTIONS,
    EXPECTED_R_MAX_A,
    EXPECTED_RAW_NODE_WIDTH,
    Assignment,
    TaylorResult,
    adapt_reference_cell,
    fixed_reference_graph,
    mace_interaction_energy,
    reordered_structure,
    sha256,
    solve_assignment,
    source_order_forces,
    state_dict_sha256,
    torch_load,
    validate_current_edge_set,
    validate_mace_architecture,
    whole_energy_taylor2_remainder,
)


FORMAT = "graphene_r2o_taylor_null_routed_tail_v1"
FEATURE_SCHEMA_VERSION = "graphene_r2o_raw_mace_invariants_v1"
PORTABLE_RECEIPT_FORMAT = "graphene_r2o_formal_portable_bundle_receipt_v1"
PORTABLE_RECEIPT_STATUS = "frozen_R2O_formal_bundle_for_postcore"
OUTER_LOCO_CONTRACT_FORMAT = "graphene_r2o_routed_tail_outer_loco_contract_v1"
OUTER_LOCO_CONTRACT_STATUS = "frozen_before_support9_open"

FORMAL_GATE_FORMAT = "graphene_r2o_fixed_checkpoint_gate_v1"
FORMAL_GATE_STATUS = "R2O_core_checkpoint_gate_passed"
FORMAL_BUNDLE_FORMAT = "graphene_r2o_deployment_bundle_v1"
FORMAL_BUNDLE_KIND = "selected_R2O_Taylor_bundle_for_postcore_validation"

EXPECTED_INVARIANT_WIDTH = 64
OUTER_LOCO_FOLDS = 9
OUTER_LOCO_TRAIN_CONFIGURATIONS = 8
OUTER_LOCO_EPOCHS = 240
OUTER_LOCO_SELECTION_POLICY = "fixed_epoch_240_EMA_no_support_checkpoint_selection"
MAX_FROZEN_GRAPH_SOURCE_QUANTIZATION_A = 1.0e-6
FROZEN_GRAPH_SOURCE_DEFAULT_DTYPE = torch.float32
NATIVE_GRAPH_SOURCE_DEFAULT_DTYPE = torch.float64
GRAPH_SEMANTICS_VERSION = "graphene_r2o_fixed_graph_semantics_v2"
NATIVE_FP64_SENSITIVITY_LIMITS = {
    "raw_node_feature_max_abs": 1.0e-5,
    "raw_scalar_energy_max_abs_eV_per_atom": 1.0e-6 / 72.0,
    "raw_scalar_force_max_abs_eV_A": 1.0e-5,
}
OUTER_LOCO_SOURCE_ARTIFACT_WHITELIST = {
    "r2o_routed_tail_module",
    "r2o_taylor_wrapper",
    "r2o_routed_tail_tests",
    "outer_loco_trainer",
    "outer_loco_evaluator",
    "outer_loco_aggregator",
    "aprime_operator",
    "provenance_helper",
}
OUTER_LOCO_REQUIRED_SOURCE_ARTIFACTS = {
    "r2o_routed_tail_module",
    "r2o_taylor_wrapper",
    "r2o_routed_tail_tests",
}
FORBIDDEN_UNOPENED_PATH_TOKENS = ("support", "seed2", "reserved")

RECEIPT_ARTIFACT_KEYS = (
    "gate",
    "bundle",
    "wrapper",
    "data_manifest",
    "reference_6x6",
    "reference_8x8",
    "DONE",
    "TRAINING_DONE",
    "CORE_GATE_PASSED",
    "EXIT_CODE",
)
RECEIPT_TOP_LEVEL_KEYS = {
    "format",
    "status",
    "formal_core",
    "paths",
    "sha256",
    "portability",
    "forbidden",
}
FORMAL_CORE_KEYS = {
    "gate_status",
    "bundle_kind",
    "postcore_or_deployment_authorized",
    "model_state_sha256",
}

SCHEMA_TOP_LEVEL_KEYS = {
    "version",
    "core_r_max_A",
    "core_num_interactions",
    "dtype",
    "raw_node_feats_width",
    "output_dimension",
    "blocks",
    "feature_names",
    "power_normalization",
    "accepted_parity",
    "reference_graph_policy",
    "assignment_policy",
    "MIC_policy",
    "autograd_policy",
    "raw_MACE_energy_policy",
    "graph_semantics",
    "formal_binding",
}
SCHEMA_BLOCK_KEYS = {
    "interaction",
    "multiplicity",
    "l",
    "parity",
    "dimension",
    "raw_slice",
    "operation",
    "output_slice",
    "feature_names",
}
SCHEMA_BINDING_KEYS = {
    "model_state_sha256",
    "wrapper_sha256",
    "data_manifest_sha256",
    "reference_6x6_sha256",
    "reference_8x8_sha256",
}


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_semantic_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def numpy_semantic_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atoms_semantic_sha256(value: Atoms) -> str:
    digest = hashlib.sha256()
    for array in (
        np.asarray(value.numbers, dtype=np.dtype("<i8")),
        np.asarray(value.positions, dtype=np.dtype("<f8")),
        np.asarray(value.cell, dtype=np.dtype("<f8")),
        np.asarray(value.pbc, dtype=np.uint8),
    ):
        digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fixed_graph_semantic_sha256(data: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in (
        "edge_index",
        "unit_shifts",
        "shifts",
        "cell",
        "node_attrs",
        "batch",
        "head",
        "ptr",
    ):
        digest.update(key.encode("ascii") + b"\0")
        digest.update(tensor_semantic_sha256(data[key]).encode("ascii"))
    return digest.hexdigest()


def frozen_graph_semantics() -> dict:
    """Hash the legacy source-dtype semantics without changing the wrapper."""
    wrapper = Path(__file__).with_name("graphene_r2o_taylor_null.py")
    payload = {
        "version": GRAPH_SEMANTICS_VERSION,
        "builder": "graphene_r2o_taylor_null.fixed_reference_graph",
        "wrapper_sha256": sha256(wrapper),
        "formal_AtomicData_source_default_dtype": str(
            FROZEN_GRAPH_SOURCE_DEFAULT_DTYPE
        ),
        "formal_stored_continuous_dtype": "torch.float64",
        "formal_semantics": "AtomicData source values are created under the frozen source default and then cast to FP64",
        "native_FP64_comparator_source_default_dtype": str(
            NATIVE_GRAPH_SOURCE_DEFAULT_DTYPE
        ),
        "native_FP64_comparator_is_deployment": False,
        "formal_graph_source_geometry_bound_A": (
            MAX_FROZEN_GRAPH_SOURCE_QUANTIZATION_A
        ),
        "formal_graph_source_geometry_metric": (
            "max_abs(cell-exact_cell, shifts-unit_shifts@exact_cell)"
        ),
    }
    payload["graph_semantics_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    if value != value.lower():
        raise ValueError(f"{label} must use lowercase hexadecimal")
    return value


def _strict_relative_path(value: Any, label: str) -> PurePosixPath:
    """Accept one canonical, portable POSIX path and reject traversal."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise ValueError(f"{label} must be a canonical relative path")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} contains traversal or an empty component")
    return path


def resolve_relative_artifact(root: Path, value: Any, label: str) -> Path:
    """Resolve a receipt path below ``root`` without following symlinks."""
    relative = _strict_relative_path(value, label)
    root = Path(root).resolve(strict=True)
    candidate = root
    for component in relative.parts:
        candidate = candidate / component
        if candidate.is_symlink():
            raise ValueError(f"{label} may not contain symlinks")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError(f"{label} escapes the portable root")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return resolved


def _floating_tensors(module: torch.nn.Module):
    for name, parameter in module.named_parameters():
        if parameter.is_floating_point():
            yield f"parameter:{name}", parameter
    for name, buffer in module.named_buffers():
        if buffer.is_floating_point():
            yield f"buffer:{name}", buffer


def ensure_fp64_module(module: torch.nn.Module, label: str) -> None:
    observed = [(name, str(value.dtype)) for name, value in _floating_tensors(module)]
    if not observed:
        raise ValueError(f"{label} has no floating tensors")
    wrong = [(name, dtype) for name, dtype in observed if dtype != "torch.float64"]
    if wrong:
        raise ValueError(f"{label} must be entirely FP64; observed {wrong[:5]}")


def ensure_frozen_core(core: torch.nn.Module) -> None:
    live = [name for name, value in core.named_parameters() if value.requires_grad]
    if live:
        raise ValueError(
            "R2O raw MACE must be parameter-frozen for routed-tail autograd; "
            f"trainable parameters include {live[:5]}"
        )


def frozen_model_state_sha256(core: torch.nn.Module) -> str:
    """Recompute semantic state after proving that parameters are frozen.

    No cached digest is trusted: ``copy_`` and ``load_state_dict`` can mutate a
    parameter even when ``requires_grad=False``.
    """
    ensure_frozen_core(core)
    return _validate_sha256(
        state_dict_sha256(core), "recomputed R2O model-state SHA-256"
    )


@dataclass(frozen=True)
class FormalR2OBinding:
    model_state_sha256: str
    wrapper_sha256: str
    data_manifest_sha256: str
    reference_6x6_sha256: str
    reference_8x8_sha256: str

    def validate(self) -> None:
        for key, value in asdict(self).items():
            _validate_sha256(value, key)


@dataclass
class LoadedFormalR2O:
    """A validated descriptor/Taylor source, never a direct calculator."""

    portable_root: Path
    receipt_path: Path
    receipt_sha256: str
    gate_sha256: str
    bundle_sha256: str
    receipt: dict
    gate: dict
    model: torch.nn.Module
    reference_paths: dict[int, Path]
    binding: FormalR2OBinding

    def reference_for_count(self, count: int) -> Atoms:
        if count not in self.reference_paths:
            raise ValueError("formal R2O supports only the frozen 6x6/8x8 references")
        expected = {
            72: self.binding.reference_6x6_sha256,
            128: self.binding.reference_8x8_sha256,
        }[count]
        path = self.reference_paths[count]
        if sha256(path) != expected:
            raise ValueError("formal R2O reference artifact changed after bundle load")
        reference = read(path, index=0)
        if len(reference) != count:
            raise ValueError("formal R2O reference atom count changed")
        return reference


def _read_json_object(path: Path, label: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def validate_formal_portable_receipt(
    portable_root: Path,
    receipt_relative_path: str,
    *,
    expected_receipt_sha256: str,
    expected_gate_sha256: str,
    expected_bundle_sha256: str,
) -> tuple[dict, dict[str, Path]]:
    """Validate all cheap/hash checks before any pickle-bearing bundle load."""
    expected_receipt_sha256 = _validate_sha256(
        expected_receipt_sha256, "expected receipt SHA-256"
    )
    expected_gate_sha256 = _validate_sha256(
        expected_gate_sha256, "expected gate SHA-256"
    )
    expected_bundle_sha256 = _validate_sha256(
        expected_bundle_sha256, "expected bundle SHA-256"
    )
    root = Path(portable_root).resolve(strict=True)
    receipt_path = resolve_relative_artifact(
        root, receipt_relative_path, "receipt_relative_path"
    )
    if sha256(receipt_path) != expected_receipt_sha256:
        raise ValueError("portable R2O receipt SHA-256 mismatch")
    receipt = _read_json_object(receipt_path, "portable receipt")
    if set(receipt) != RECEIPT_TOP_LEVEL_KEYS:
        raise ValueError("portable R2O receipt top-level fields changed")
    if receipt["format"] != PORTABLE_RECEIPT_FORMAT:
        raise ValueError("unsupported portable R2O receipt format")
    if receipt["status"] != PORTABLE_RECEIPT_STATUS:
        raise ValueError("portable R2O receipt is not frozen for postcore")
    formal = receipt["formal_core"]
    if not isinstance(formal, dict) or set(formal) != FORMAL_CORE_KEYS:
        raise ValueError("portable R2O formal-core fields changed")
    if formal["gate_status"] != FORMAL_GATE_STATUS:
        raise ValueError("portable receipt is not bound to a passing formal gate")
    if formal["bundle_kind"] != FORMAL_BUNDLE_KIND:
        raise ValueError("portable receipt is not bound to a deployable Taylor bundle")
    if formal["postcore_or_deployment_authorized"] is not True:
        raise ValueError("portable receipt does not authorize postcore work")
    _validate_sha256(formal["model_state_sha256"], "formal model-state SHA-256")
    if receipt["portability"] != {
        "path_mode": "relative_to_portable_root",
        "symlinks_allowed": False,
        "content_hash_required_for_every_path": True,
    }:
        raise ValueError("portable R2O path policy changed")
    if receipt["forbidden"] != {
        "support9_opened": False,
        "seed2_opened": False,
        "raw_MACE_direct_deployment": False,
    }:
        raise ValueError("portable R2O leakage/deployment declaration changed")
    paths = receipt["paths"]
    hashes = receipt["sha256"]
    if not isinstance(paths, dict) or set(paths) != set(RECEIPT_ARTIFACT_KEYS):
        raise ValueError("portable R2O artifact path keys changed")
    if not isinstance(hashes, dict) or set(hashes) != set(RECEIPT_ARTIFACT_KEYS):
        raise ValueError("portable R2O artifact hash keys changed")
    resolved: dict[str, Path] = {}
    for key in RECEIPT_ARTIFACT_KEYS:
        expected = _validate_sha256(hashes[key], f"receipt {key} SHA-256")
        path = resolve_relative_artifact(root, paths[key], f"receipt paths.{key}")
        if sha256(path) != expected:
            raise ValueError(f"portable R2O artifact changed: {key}")
        resolved[key] = path
    if hashes["gate"] != expected_gate_sha256:
        raise ValueError("portable receipt gate differs from the frozen selection")
    if hashes["bundle"] != expected_bundle_sha256:
        raise ValueError("portable receipt bundle differs from the frozen selection")
    return receipt, resolved


def _validate_formal_gate_and_markers(
    receipt: dict, resolved: Mapping[str, Path]
) -> tuple[dict, FormalR2OBinding]:
    marker_parent = resolved["DONE"].parent
    if any(resolved[key].parent != marker_parent for key in (
        "TRAINING_DONE", "CORE_GATE_PASSED", "EXIT_CODE"
    )):
        raise ValueError("formal R2O completion markers do not share one run root")
    if resolved["gate"].parent != marker_parent or resolved["bundle"].parent != marker_parent:
        raise ValueError("formal R2O gate, bundle and completion markers must share one run root")
    for name in ("RUNNING", "FAILED", "CORE_GATE_FAILED"):
        if (marker_parent / name).exists():
            raise ValueError(f"formal R2O run still has forbidden marker {name}")
    if resolved["gate"].name != "core_checkpoint_gate.json":
        raise ValueError("formal R2O gate filename changed")
    expected_marker_names = {
        "DONE": "DONE",
        "TRAINING_DONE": "TRAINING_DONE",
        "CORE_GATE_PASSED": "CORE_GATE_PASSED",
        "EXIT_CODE": "EXIT_CODE",
    }
    for key, expected_name in expected_marker_names.items():
        if resolved[key].name != expected_name:
            raise ValueError(f"formal R2O {key} marker filename changed")
    if resolved["DONE"].read_text(encoding="utf-8").strip() != FORMAL_GATE_STATUS:
        raise ValueError("formal R2O DONE marker changed")
    if resolved["EXIT_CODE"].read_text(encoding="utf-8").strip() != "0":
        raise ValueError("formal R2O EXIT_CODE is not zero")
    for key in ("TRAINING_DONE", "CORE_GATE_PASSED"):
        if resolved[key].stat().st_size != 0:
            raise ValueError(f"formal R2O {key} marker must be empty")
    gate = _read_json_object(resolved["gate"], "formal R2O gate")
    required = {
        "format": FORMAL_GATE_FORMAT,
        "status": FORMAL_GATE_STATUS,
        "smoke": False,
        "postcore_or_deployment_authorized": True,
        "seed2_or_support_read": False,
        "implementation_pass": True,
        "cutoff_C2_pass": True,
        "reference_null_pass": True,
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise ValueError(f"formal R2O gate field {key!r} is not passing/frozen")
    if not isinstance(gate.get("bundle"), dict):
        raise ValueError("formal R2O gate lacks a bundle record")
    bundle = gate["bundle"]
    if bundle.get("kind") != FORMAL_BUNDLE_KIND:
        raise ValueError("formal R2O gate selected a non-deployable bundle")
    if bundle.get("sha256") != receipt["sha256"]["bundle"]:
        raise ValueError("formal R2O gate/bundle content hash mismatch")
    if bundle.get("model_state_sha256") != receipt["formal_core"][
        "model_state_sha256"
    ]:
        raise ValueError("formal R2O gate/receipt model-state hash mismatch")
    if Path(str(bundle.get("path", ""))).name != resolved["bundle"].name:
        raise ValueError("formal R2O gate refers to another bundle filename")
    matching = {
        "wrapper_sha256": "wrapper",
        "data_manifest_sha256": "data_manifest",
        "reference_6x6_sha256": "reference_6x6",
        "reference_8x8_sha256": "reference_8x8",
    }
    for gate_key, artifact_key in matching.items():
        if bundle.get(gate_key) != receipt["sha256"][artifact_key]:
            raise ValueError(f"formal R2O gate binding changed: {gate_key}")
    if gate.get("inputs", {}).get("data_manifest_sha256") != receipt["sha256"][
        "data_manifest"
    ]:
        raise ValueError("formal R2O gate input manifest hash mismatch")
    wrapper_source = Path(__file__).with_name("graphene_r2o_taylor_null.py")
    if sha256(wrapper_source) != receipt["sha256"]["wrapper"]:
        raise ValueError("runtime Taylor wrapper differs from the formal snapshot")
    manifest = _read_json_object(resolved["data_manifest"], "R2O data manifest")
    if manifest.get("status") != "frozen_before_R2O_training":
        raise ValueError("formal R2O data manifest status changed")
    counts = manifest.get("counts", {})
    if counts.get("seed2_opened") != 0 or counts.get("support_opened") != 0:
        raise ValueError("formal R2O data manifest opened a forbidden split")
    binding = FormalR2OBinding(
        model_state_sha256=receipt["formal_core"]["model_state_sha256"],
        wrapper_sha256=receipt["sha256"]["wrapper"],
        data_manifest_sha256=receipt["sha256"]["data_manifest"],
        reference_6x6_sha256=receipt["sha256"]["reference_6x6"],
        reference_8x8_sha256=receipt["sha256"]["reference_8x8"],
    )
    binding.validate()
    return gate, binding


def load_formal_r2o_bundle(
    portable_root: Path,
    receipt_relative_path: str,
    *,
    expected_receipt_sha256: str,
    expected_gate_sha256: str,
    expected_bundle_sha256: str,
    device: torch.device | str,
) -> LoadedFormalR2O:
    """Load only a formally passing, hash-bound R2O bundle in FP64.

    Hash/status/marker checks intentionally precede ``torch.load``.  The loaded
    raw MACE is frozen and returned only inside :class:`LoadedFormalR2O`.
    """
    receipt, resolved = validate_formal_portable_receipt(
        portable_root,
        receipt_relative_path,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_gate_sha256=expected_gate_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
    )
    gate, binding = _validate_formal_gate_and_markers(receipt, resolved)
    payload = torch_load(resolved["bundle"], map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != FORMAL_BUNDLE_FORMAT:
        raise ValueError("unsupported R2O deployment bundle")
    if payload.get("raw_MACE_must_not_be_deployed_without_Taylor_wrapper") is not True:
        raise ValueError("R2O bundle lost its raw-MACE deployment prohibition")
    model = payload.get("model")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("R2O deployment bundle has no model module")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("R2O deployment bundle has no metadata")
    expected_metadata = {
        "kind": FORMAL_BUNDLE_KIND,
        "model_state_sha256": binding.model_state_sha256,
        "wrapper_sha256": binding.wrapper_sha256,
        "data_manifest_sha256": binding.data_manifest_sha256,
        "reference_6x6_sha256": binding.reference_6x6_sha256,
        "reference_8x8_sha256": binding.reference_8x8_sha256,
        "raw_MACE_direct_deployment_forbidden": True,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"R2O bundle metadata changed: {key}")
    if state_dict_sha256(model) != binding.model_state_sha256:
        raise ValueError("R2O bundle semantic model-state SHA-256 mismatch")
    ensure_fp64_module(model, "formal R2O raw MACE")
    validate_mace_architecture(model)
    model.requires_grad_(False)
    model.eval()
    ensure_frozen_core(model)
    model.to(device=device, dtype=torch.float64)
    reference_paths = {
        72: resolved["reference_6x6"],
        128: resolved["reference_8x8"],
    }
    if len(read(reference_paths[72], index=0)) != 72 or len(
        read(reference_paths[128], index=0)
    ) != 128:
        raise ValueError("formal R2O reference atom counts changed")
    return LoadedFormalR2O(
        portable_root=Path(portable_root).resolve(strict=True),
        receipt_path=resolve_relative_artifact(
            portable_root, receipt_relative_path, "receipt_relative_path"
        ),
        receipt_sha256=expected_receipt_sha256,
        gate_sha256=expected_gate_sha256,
        bundle_sha256=expected_bundle_sha256,
        receipt=receipt,
        gate=gate,
        model=model,
        reference_paths=reference_paths,
        binding=binding,
    )


def validate_loaded_formal_r2o(formal: LoadedFormalR2O) -> str:
    """Revalidate the complete portable receipt chain and semantic model state."""
    try:
        receipt_relative = formal.receipt_path.relative_to(
            formal.portable_root
        ).as_posix()
    except ValueError as error:
        raise ValueError("formal R2O receipt escaped its portable root") from error
    receipt, resolved = validate_formal_portable_receipt(
        formal.portable_root,
        receipt_relative,
        expected_receipt_sha256=formal.receipt_sha256,
        expected_gate_sha256=formal.gate_sha256,
        expected_bundle_sha256=formal.bundle_sha256,
    )
    gate, binding = _validate_formal_gate_and_markers(receipt, resolved)
    if receipt != formal.receipt or gate != formal.gate or binding != formal.binding:
        raise ValueError("loaded formal R2O provenance differs from its portable chain")
    if resolved["reference_6x6"] != formal.reference_paths.get(72) or resolved[
        "reference_8x8"
    ] != formal.reference_paths.get(128):
        raise ValueError("loaded formal R2O reference paths changed")
    observed = frozen_model_state_sha256(formal.model)
    if observed != formal.binding.model_state_sha256:
        raise ValueError("formal R2O semantic model state changed after load")
    return observed


def _layout_blocks(core: torch.nn.Module) -> tuple[list[dict], int]:
    validate_mace_architecture(core)
    blocks: list[dict] = []
    observed = []
    offset = 0
    for interaction, product in enumerate(core.products):
        for multiplicity, irrep in product.linear.irreps_out:
            multiplicity = int(multiplicity)
            angular_momentum = int(irrep.l)
            parity = int(irrep.p)
            dimension = int(irrep.dim)
            width = multiplicity * dimension
            if angular_momentum == 0 and parity != 1:
                raise ValueError("0o pseudoscalars cannot enter the O(3) router")
            blocks.append(
                {
                    "interaction": int(interaction),
                    "multiplicity": multiplicity,
                    "l": angular_momentum,
                    "parity": parity,
                    "dimension": dimension,
                    "raw_slice": [offset, offset + width],
                    "operation": (
                        "signed_l0_value"
                        if angular_momentum == 0
                        else "mean_m_squared"
                    ),
                }
            )
            observed.append((interaction, multiplicity, angular_momentum, parity))
            offset += width
    if tuple(observed) != EXPECTED_HIDDEN_LAYOUT or offset != EXPECTED_RAW_NODE_WIDTH:
        raise ValueError("formal R2O raw feature layout changed")
    return blocks, offset


def r2o_invariant_schema(formal: LoadedFormalR2O) -> dict:
    """Publish the R2O-specific 64-D invariant map and provenance binding."""
    core = formal.model
    binding = formal.binding
    observed_state = validate_loaded_formal_r2o(formal)
    ensure_fp64_module(core, "formal R2O raw MACE")
    binding.validate()
    if observed_state != binding.model_state_sha256:
        raise ValueError("R2O invariant schema binding belongs to another core")
    blocks, raw_width = _layout_blocks(core)
    names: list[str] = []
    output_offset = 0
    for block in blocks:
        block_names = [
            (
                f"interaction{block['interaction']}_l{block['l']}_"
                f"p{block['parity']:+d}_channel{channel:02d}_"
                + (
                    "value"
                    if block["operation"] == "signed_l0_value"
                    else "mean_m_h2"
                )
            )
            for channel in range(block["multiplicity"])
        ]
        block["output_slice"] = [
            output_offset,
            output_offset + len(block_names),
        ]
        block["feature_names"] = block_names
        names.extend(block_names)
        output_offset += len(block_names)
    if output_offset != EXPECTED_INVARIANT_WIDTH:
        raise ValueError("formal R2O invariant width changed")
    payload = {
        "version": FEATURE_SCHEMA_VERSION,
        "core_r_max_A": EXPECTED_R_MAX_A,
        "core_num_interactions": EXPECTED_NUM_INTERACTIONS,
        "dtype": "torch.float64",
        "raw_node_feats_width": raw_width,
        "output_dimension": output_offset,
        "blocks": blocks,
        "feature_names": names,
        "power_normalization": "mean over m = squared norm/(2*l+1)",
        "accepted_parity": "signed 0e plus O(3)-even squared norms; reject 0o",
        "reference_graph_policy": "formal_bundle_fixed_6x6_or_8x8_graph",
        "assignment_policy": "species_aware_periodic_Hungarian_unique_nearest",
        "MIC_policy": "integer_image_detached_continuous_displacement_live",
        "autograd_policy": "query_features_router_gate_node_energy_u_and_HVP_direction_live",
        "raw_MACE_energy_policy": "descriptor_only_direct_prediction_forbidden",
        "graph_semantics": frozen_graph_semantics(),
        "formal_binding": asdict(binding),
    }
    payload["schema_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_r2o_invariant_schema(schema: dict) -> None:
    if not isinstance(schema, dict):
        raise ValueError("R2O invariant schema must be a mapping")
    payload = dict(schema)
    recorded_hash = payload.pop("schema_sha256", None)
    if recorded_hash is None or canonical_json_sha256(payload) != recorded_hash:
        raise ValueError("R2O invariant schema SHA-256 mismatch")
    if set(payload) != SCHEMA_TOP_LEVEL_KEYS:
        raise ValueError("R2O invariant schema top-level fields changed")
    required = {
        "version": FEATURE_SCHEMA_VERSION,
        "core_r_max_A": EXPECTED_R_MAX_A,
        "core_num_interactions": EXPECTED_NUM_INTERACTIONS,
        "dtype": "torch.float64",
        "raw_node_feats_width": EXPECTED_RAW_NODE_WIDTH,
        "output_dimension": EXPECTED_INVARIANT_WIDTH,
        "power_normalization": "mean over m = squared norm/(2*l+1)",
        "accepted_parity": "signed 0e plus O(3)-even squared norms; reject 0o",
        "reference_graph_policy": "formal_bundle_fixed_6x6_or_8x8_graph",
        "assignment_policy": "species_aware_periodic_Hungarian_unique_nearest",
        "MIC_policy": "integer_image_detached_continuous_displacement_live",
        "autograd_policy": "query_features_router_gate_node_energy_u_and_HVP_direction_live",
        "raw_MACE_energy_policy": "descriptor_only_direct_prediction_forbidden",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"R2O invariant schema field changed: {key}")
    if payload["graph_semantics"] != frozen_graph_semantics():
        raise ValueError("R2O invariant schema graph semantics/hash changed")
    binding = payload["formal_binding"]
    if not isinstance(binding, dict) or set(binding) != SCHEMA_BINDING_KEYS:
        raise ValueError("R2O invariant schema formal binding changed")
    FormalR2OBinding(**binding).validate()
    blocks = payload["blocks"]
    if not isinstance(blocks, list) or len(blocks) != len(EXPECTED_HIDDEN_LAYOUT):
        raise ValueError("R2O invariant schema block count changed")
    raw_offset = 0
    output_offset = 0
    all_names: list[str] = []
    for block, layout in zip(blocks, EXPECTED_HIDDEN_LAYOUT, strict=True):
        if not isinstance(block, dict) or set(block) != SCHEMA_BLOCK_KEYS:
            raise ValueError("R2O invariant schema block fields changed")
        interaction, multiplicity, angular_momentum, parity = layout
        dimension = 2 * angular_momentum + 1
        raw_stop = raw_offset + multiplicity * dimension
        output_stop = output_offset + multiplicity
        operation = (
            "signed_l0_value" if angular_momentum == 0 else "mean_m_squared"
        )
        names = [
            (
                f"interaction{interaction}_l{angular_momentum}_"
                f"p{parity:+d}_channel{channel:02d}_"
                + ("value" if operation == "signed_l0_value" else "mean_m_h2")
            )
            for channel in range(multiplicity)
        ]
        expected = {
            "interaction": interaction,
            "multiplicity": multiplicity,
            "l": angular_momentum,
            "parity": parity,
            "dimension": dimension,
            "raw_slice": [raw_offset, raw_stop],
            "operation": operation,
            "output_slice": [output_offset, output_stop],
            "feature_names": names,
        }
        if block != expected:
            raise ValueError("R2O invariant schema block mapping changed")
        raw_offset = raw_stop
        output_offset = output_stop
        all_names.extend(names)
    if payload["feature_names"] != all_names:
        raise ValueError("R2O invariant schema feature names/order changed")


def r2o_node_invariants(node_features: torch.Tensor, schema: dict) -> torch.Tensor:
    validate_r2o_invariant_schema(schema)
    if node_features.dtype != torch.float64:
        raise ValueError("R2O raw node features must remain FP64")
    if node_features.ndim != 2 or node_features.shape[1] != EXPECTED_RAW_NODE_WIDTH:
        raise ValueError(
            f"R2O node features require shape (n,{EXPECTED_RAW_NODE_WIDTH})"
        )
    values = []
    for block_record in schema["blocks"]:
        start, stop = block_record["raw_slice"]
        block = node_features[:, start:stop].reshape(
            node_features.shape[0],
            block_record["multiplicity"],
            block_record["dimension"],
        )
        if block_record["operation"] == "signed_l0_value":
            values.append(block[:, :, 0])
        else:
            values.append(torch.mean(torch.square(block), dim=-1))
    result = torch.cat(values, dim=1)
    if result.shape[1] != EXPECTED_INVARIANT_WIDTH:
        raise RuntimeError("R2O invariant replay produced the wrong width")
    return result


def _r2o_raw_node_features_prevalidated(
    core: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
) -> torch.Tensor:
    """Evaluate frozen R2O encoder features on its fixed reference graph."""
    ensure_frozen_core(core)
    ensure_fp64_module(core, "formal R2O raw MACE")
    validate_mace_architecture(core)
    if positions.dtype != torch.float64:
        raise ValueError("R2O query positions must be FP64")
    if not positions.requires_grad:
        raise ValueError("R2O query positions must remain live")
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=data["edge_index"],
        shifts=data["shifts"],
    )
    if hasattr(core, "joint_embedding"):
        raise ValueError("formal R2O does not allow auxiliary embeddings")
    features = core.node_embedding(data["node_attrs"])
    edge_attributes = core.spherical_harmonics(vectors)
    edge_features, cutoff = core.radial_embedding(
        lengths, data["node_attrs"], data["edge_index"], core.atomic_numbers
    )
    outputs = []
    for index, (interaction, product) in enumerate(
        zip(core.interactions, core.products, strict=True)
    ):
        features, skip = interaction(
            node_attrs=data["node_attrs"],
            node_feats=features,
            edge_attrs=edge_attributes,
            edge_feats=edge_features,
            edge_index=data["edge_index"],
            cutoff=cutoff,
            first_layer=index == 0,
        )
        features = product(
            node_feats=features,
            sc=skip,
            node_attrs=data["node_attrs"],
        )
        outputs.append(features)
    result = torch.cat(outputs, dim=-1)
    if result.shape != (positions.shape[0], EXPECTED_RAW_NODE_WIDTH):
        raise ValueError("formal R2O raw node feature layout changed at runtime")
    return result


def _fixed_reference_graph_with_source_default(
    reference: Atoms,
    *,
    device: torch.device | str,
    source_default_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Build one graph under an explicit source default, restoring global state."""
    if source_default_dtype not in (torch.float32, torch.float64):
        raise ValueError("unsupported R2O graph source default dtype")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(source_default_dtype)
        return fixed_reference_graph(reference, device=device, dtype=torch.float64)
    finally:
        torch.set_default_dtype(previous)


@dataclass
class R2OOrderMICQuery:
    data: dict[str, torch.Tensor]
    native_fp64_data: dict[str, torch.Tensor]
    assignment: Assignment
    current_positions_reference_order: torch.Tensor
    reference_positions: torch.Tensor
    image_shift: torch.Tensor
    native_fp64_image_shift: torch.Tensor
    reference: Atoms
    frozen_graph_source_quantization_max_A: float
    graph_semantics_sha256: str
    formal_model_state_sha256: str
    reference_atom_count: int
    reference_artifact_sha256: str
    reference_content_sha256: str
    reference_positions_sha256: str
    fixed_graph_content_sha256: str
    native_fp64_graph_content_sha256: str
    assignment_reference_to_source_sha256: str
    assignment_source_to_reference_sha256: str
    assignment_image_integer_reference_order_sha256: str
    image_shift_sha256: str
    native_fp64_image_shift_sha256: str
    formal_receipt_sha256: str

    @property
    def aligned_positions(self) -> torch.Tensor:
        return self.current_positions_reference_order - self.image_shift.detach()


def build_order_mic_query(
    formal: LoadedFormalR2O,
    structure: Atoms,
    *,
    check_edges: bool = True,
) -> R2OOrderMICQuery:
    """Build a query using only the hash-bound formal 6x6/8x8 reference."""
    core = formal.model
    ensure_frozen_core(core)
    ensure_fp64_module(core, "formal R2O raw MACE")
    validate_mace_architecture(core)
    observed_model_sha256 = validate_loaded_formal_r2o(formal)
    reference_template = formal.reference_for_count(len(structure))
    reference_artifact_sha256 = {
        72: formal.binding.reference_6x6_sha256,
        128: formal.binding.reference_8x8_sha256,
    }[len(structure)]
    try:
        device = next(core.parameters()).device
    except StopIteration as error:
        raise ValueError("formal R2O model has no parameters") from error
    reference = adapt_reference_cell(reference_template, structure)
    assignment = solve_assignment(structure, reference)
    ordered = reordered_structure(structure, assignment)
    cell = np.asarray(ordered.cell, float)
    aligned = ordered.copy()
    aligned.positions = (
        np.asarray(ordered.positions, float)
        - assignment.image_integer_reference_order @ cell
    )
    if check_edges:
        validate_current_edge_set(aligned, reference)
    data = _fixed_reference_graph_with_source_default(
        reference,
        device=device,
        source_default_dtype=FROZEN_GRAPH_SOURCE_DEFAULT_DTYPE,
    )
    native_data = _fixed_reference_graph_with_source_default(
        reference,
        device=device,
        source_default_dtype=NATIVE_GRAPH_SOURCE_DEFAULT_DTYPE,
    )
    exact_cell = torch.as_tensor(
        np.asarray(reference.cell, float), dtype=torch.float64, device=device
    )
    stored_cell = data["cell"].reshape(-1, 3, 3)[0]
    exact_shifts = data["unit_shifts"].to(dtype=torch.float64) @ exact_cell
    source_quantization = max(
        float(torch.max(torch.abs(stored_cell - exact_cell))),
        float(torch.max(torch.abs(data["shifts"] - exact_shifts))),
    )
    if source_quantization > MAX_FROZEN_GRAPH_SOURCE_QUANTIZATION_A:
        raise ValueError(
            "frozen R2O graph source quantization exceeds its audited bound"
        )
    current = torch.as_tensor(
        np.asarray(ordered.positions, float), dtype=torch.float64, device=device
    ).clone().requires_grad_(True)
    reference_positions = torch.as_tensor(
        np.asarray(reference.positions, float), dtype=torch.float64, device=device
    )
    image_integer = torch.as_tensor(
        assignment.image_integer_reference_order,
        dtype=torch.int64,
        device=device,
    )
    image_shift = image_integer.to(dtype=torch.float64) @ data["cell"].reshape(-1, 3, 3)[0]
    native_image_shift = (
        image_integer.to(dtype=torch.float64)
        @ native_data["cell"].reshape(-1, 3, 3)[0]
    )
    if image_shift.requires_grad:
        raise RuntimeError("discrete R2O MIC image unexpectedly entered autograd")
    if native_image_shift.requires_grad:
        raise RuntimeError("native FP64 comparator MIC unexpectedly entered autograd")
    graph_semantics = frozen_graph_semantics()
    return R2OOrderMICQuery(
        data=data,
        native_fp64_data=native_data,
        assignment=assignment,
        current_positions_reference_order=current,
        reference_positions=reference_positions,
        image_shift=image_shift,
        native_fp64_image_shift=native_image_shift,
        reference=reference,
        frozen_graph_source_quantization_max_A=source_quantization,
        graph_semantics_sha256=graph_semantics["graph_semantics_sha256"],
        formal_model_state_sha256=observed_model_sha256,
        reference_atom_count=len(reference),
        reference_artifact_sha256=reference_artifact_sha256,
        reference_content_sha256=atoms_semantic_sha256(reference),
        reference_positions_sha256=tensor_semantic_sha256(reference_positions),
        fixed_graph_content_sha256=fixed_graph_semantic_sha256(data),
        native_fp64_graph_content_sha256=fixed_graph_semantic_sha256(native_data),
        assignment_reference_to_source_sha256=numpy_semantic_sha256(
            assignment.reference_to_source
        ),
        assignment_source_to_reference_sha256=numpy_semantic_sha256(
            assignment.source_to_reference
        ),
        assignment_image_integer_reference_order_sha256=numpy_semantic_sha256(
            assignment.image_integer_reference_order
        ),
        image_shift_sha256=tensor_semantic_sha256(image_shift),
        native_fp64_image_shift_sha256=tensor_semantic_sha256(native_image_shift),
        formal_receipt_sha256=formal.receipt_sha256,
    )


def query_raw_invariants(
    formal: LoadedFormalR2O, query: R2OOrderMICQuery, schema: dict
) -> torch.Tensor:
    core = formal.model
    _validate_formal_query_binding(formal, query, schema)
    raw = _r2o_raw_node_features_prevalidated(
        core, query.data, query.aligned_positions
    )
    return r2o_node_invariants(raw, schema)


def smootherstep(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, 0.0, 1.0)
    return clipped**3 * (10.0 - 15.0 * clipped + 6.0 * clipped**2)


@dataclass(frozen=True)
class R2ORoutedTailSpecification:
    input_dimension: int = EXPECTED_INVARIANT_WIDTH
    tail_hidden_1: int = 64
    tail_hidden_2: int = 32
    router_hidden_1: int = 32
    router_hidden_2: int = 16
    gate_score_off: float = 0.0
    gate_score_on: float = 1.0
    dtype: str = "torch.float64"

    def validate(self) -> None:
        if self != R2ORoutedTailSpecification():
            raise ValueError("R2O routed-tail architecture is frozen for outer LOCO")


def _require_fp64_tensor(value: np.ndarray | torch.Tensor, label: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if result.dtype != torch.float64:
        raise ValueError(f"{label} must be supplied in FP64")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} contains non-finite values")
    return result


class R2OTaylorNullRoutedTail(torch.nn.Module):
    """Raw local routed scalar; Taylor null is applied by the public API."""

    def __init__(
        self,
        specification: R2ORoutedTailSpecification,
        feature_mean: np.ndarray | torch.Tensor,
        feature_scale: np.ndarray | torch.Tensor,
        pristine_invariants: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        specification.validate()
        mean = _require_fp64_tensor(feature_mean, "feature_mean")
        scale = _require_fp64_tensor(feature_scale, "feature_scale")
        pristine = _require_fp64_tensor(pristine_invariants, "pristine_invariants")
        if mean.shape != (EXPECTED_INVARIANT_WIDTH,):
            raise ValueError("R2O feature mean has the wrong shape")
        if scale.shape != mean.shape or bool(torch.any(scale <= 0.0)):
            raise ValueError("R2O feature scale must be positive and match the mean")
        if pristine.ndim != 2 or pristine.shape[1] != EXPECTED_INVARIANT_WIDTH:
            raise ValueError("R2O pristine invariants have the wrong shape")
        self.specification = specification
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_scale", scale.clone())
        self.register_buffer("pristine_invariants", pristine.clone())
        self.tail_encoder = torch.nn.Sequential(
            torch.nn.Linear(EXPECTED_INVARIANT_WIDTH, specification.tail_hidden_1),
            torch.nn.SiLU(),
            torch.nn.Linear(specification.tail_hidden_1, specification.tail_hidden_2),
            torch.nn.SiLU(),
            torch.nn.Linear(specification.tail_hidden_2, 1),
        ).double()
        self.router_encoder = torch.nn.Sequential(
            torch.nn.Linear(EXPECTED_INVARIANT_WIDTH, specification.router_hidden_1),
            torch.nn.SiLU(),
            torch.nn.Linear(specification.router_hidden_1, specification.router_hidden_2),
            torch.nn.SiLU(),
            torch.nn.Linear(specification.router_hidden_2, 1),
        ).double()
        torch.nn.init.zeros_(self.tail_encoder[-1].weight)
        torch.nn.init.zeros_(self.tail_encoder[-1].bias)
        torch.nn.init.zeros_(self.router_encoder[-1].weight)
        torch.nn.init.constant_(self.router_encoder[-1].bias, 0.5)
        ensure_fp64_module(self, "R2O routed tail")

    def standardized(self, invariants: torch.Tensor) -> torch.Tensor:
        if invariants.dtype != torch.float64:
            raise ValueError("R2O invariants must remain FP64")
        if invariants.ndim != 2 or invariants.shape[1] != EXPECTED_INVARIANT_WIDTH:
            raise ValueError("R2O routed-tail invariant shape changed")
        return (invariants - self.feature_mean) / self.feature_scale

    def pristine_carbon_gauge(self) -> torch.Tensor:
        return torch.mean(
            self.tail_encoder(self.standardized(self.pristine_invariants)).squeeze(-1)
        )

    def node_terms(
        self, invariants: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        standardized = self.standardized(invariants)
        epsilon = self.tail_encoder(standardized).squeeze(-1)
        score = self.router_encoder(standardized).squeeze(-1)
        interval = self.specification.gate_score_on - self.specification.gate_score_off
        gate = smootherstep((score - self.specification.gate_score_off) / interval)
        node_energy = gate * (epsilon - self.pristine_carbon_gauge())
        return node_energy, gate, score, epsilon

    def raw_graph_energies(
        self, invariants: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if batch.ndim != 1 or batch.shape[0] != invariants.shape[0]:
            raise ValueError("R2O routed-tail batch vector does not match invariants")
        if batch.dtype != torch.long:
            raise ValueError("R2O routed-tail batch vector must be torch.long")
        node_energy, gate, score, epsilon = self.node_terms(invariants)
        graph_count = int(batch.max().detach().cpu()) + 1 if batch.numel() else 0
        energies = node_energy.new_zeros(graph_count)
        energies.index_add_(0, batch, node_energy)
        return energies, {
            "node_energy": node_energy,
            "gate": gate,
            "score": score,
            "epsilon": epsilon,
            "c_C_eV": self.pristine_carbon_gauge(),
        }


def _validate_formal_query_binding(
    formal: LoadedFormalR2O,
    query: R2OOrderMICQuery,
    schema: dict | None = None,
) -> str:
    observed = validate_loaded_formal_r2o(formal)
    if query.formal_model_state_sha256 != observed:
        raise ValueError("R2O query belongs to another formal core state")
    if sha256(formal.receipt_path) != formal.receipt_sha256:
        raise ValueError("formal R2O portable receipt changed after load")
    if query.formal_receipt_sha256 != formal.receipt_sha256:
        raise ValueError("R2O query belongs to another portable receipt")
    if query.reference_atom_count != len(query.reference):
        raise ValueError("R2O query reference atom count record changed")
    expected_reference = {
        72: formal.binding.reference_6x6_sha256,
        128: formal.binding.reference_8x8_sha256,
    }.get(query.reference_atom_count)
    if expected_reference is None or query.reference_artifact_sha256 != expected_reference:
        raise ValueError("R2O query is not bound to a formal 6x6/8x8 reference")
    if sha256(formal.reference_paths[query.reference_atom_count]) != expected_reference:
        raise ValueError("formal R2O reference artifact changed after query creation")
    if atoms_semantic_sha256(query.reference) != query.reference_content_sha256:
        raise ValueError("R2O query reference content changed after query creation")
    if tensor_semantic_sha256(query.reference_positions) != (
        query.reference_positions_sha256
    ):
        raise ValueError("R2O query reference-position tensor changed")
    reference_from_artifact = adapt_reference_cell(
        formal.reference_for_count(query.reference_atom_count), query.reference
    )
    if atoms_semantic_sha256(reference_from_artifact) != query.reference_content_sha256:
        raise ValueError("R2O query Taylor reference differs from the formal artifact")
    if fixed_graph_semantic_sha256(query.data) != query.fixed_graph_content_sha256:
        raise ValueError("R2O query frozen-semantics graph content changed")
    if fixed_graph_semantic_sha256(query.native_fp64_data) != (
        query.native_fp64_graph_content_sha256
    ):
        raise ValueError("R2O query native-FP64 graph content changed")
    try:
        query.assignment.validate(query.reference_atom_count)
    except (IndexError, ValueError) as error:
        raise ValueError("R2O query assignment is no longer valid") from error
    if numpy_semantic_sha256(query.assignment.reference_to_source) != (
        query.assignment_reference_to_source_sha256
    ):
        raise ValueError("R2O query assignment reference_to_source changed")
    if numpy_semantic_sha256(query.assignment.source_to_reference) != (
        query.assignment_source_to_reference_sha256
    ):
        raise ValueError("R2O query assignment source_to_reference changed")
    if numpy_semantic_sha256(
        query.assignment.image_integer_reference_order
    ) != query.assignment_image_integer_reference_order_sha256:
        raise ValueError("R2O query assignment MIC integers changed")
    if tensor_semantic_sha256(query.image_shift) != query.image_shift_sha256:
        raise ValueError("R2O query frozen-semantics image shift changed")
    if tensor_semantic_sha256(query.native_fp64_image_shift) != (
        query.native_fp64_image_shift_sha256
    ):
        raise ValueError("R2O query native-FP64 image shift changed")
    if query.image_shift.requires_grad or query.native_fp64_image_shift.requires_grad:
        raise ValueError("R2O query discrete image shift entered autograd")
    image_integer = torch.as_tensor(
        query.assignment.image_integer_reference_order,
        dtype=torch.int64,
        device=query.image_shift.device,
    )
    expected_image_shift = image_integer.to(dtype=torch.float64) @ query.data[
        "cell"
    ].reshape(-1, 3, 3)[0]
    expected_native_image_shift = image_integer.to(dtype=torch.float64) @ (
        query.native_fp64_data["cell"].reshape(-1, 3, 3)[0]
    )
    if not torch.equal(query.image_shift, expected_image_shift):
        raise ValueError("R2O query image shift is inconsistent with MIC and graph cell")
    if not torch.equal(query.native_fp64_image_shift, expected_native_image_shift):
        raise ValueError(
            "R2O query native-FP64 image shift is inconsistent with MIC and graph cell"
        )
    if schema is not None:
        validate_r2o_invariant_schema(schema)
        if schema["formal_binding"] != asdict(formal.binding):
            raise ValueError("R2O routed-tail schema belongs to another formal bundle")
    return observed


def r2o_raw_node_features(
    formal: LoadedFormalR2O,
    query: R2OOrderMICQuery,
    *,
    positions: torch.Tensor | None = None,
    native_fp64_comparator: bool = False,
) -> torch.Tensor:
    """Public provenance-bound raw descriptor API."""
    _validate_formal_query_binding(formal, query)
    data = query.native_fp64_data if native_fp64_comparator else query.data
    if positions is None:
        positions = (
            query.current_positions_reference_order
            - (
                query.native_fp64_image_shift
                if native_fp64_comparator
                else query.image_shift
            ).detach()
        )
    return _r2o_raw_node_features_prevalidated(formal.model, data, positions)


def _raw_tail_energy_prevalidated(
    core: torch.nn.Module,
    tail: R2OTaylorNullRoutedTail,
    data: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
    schema: dict,
) -> torch.Tensor:
    raw_features = _r2o_raw_node_features_prevalidated(core, data, positions)
    invariants = r2o_node_invariants(raw_features, schema)
    energies, _ = tail.raw_graph_energies(invariants, data["batch"].to(torch.long))
    return energies


def raw_tail_energy(
    formal: LoadedFormalR2O,
    tail: R2OTaylorNullRoutedTail,
    query: R2OOrderMICQuery,
    positions: torch.Tensor,
    schema: dict,
    *,
    native_fp64_comparator: bool = False,
) -> torch.Tensor:
    """Public, provenance-bound raw tail scalar; never deploy it directly."""
    _validate_formal_query_binding(formal, query, schema)
    data = query.native_fp64_data if native_fp64_comparator else query.data
    return _raw_tail_energy_prevalidated(
        formal.model, tail, data, positions, schema
    )


def audit_paired_native_fp64_sensitivity(
    formal: LoadedFormalR2O,
    query: R2OOrderMICQuery,
    *,
    tail: R2OTaylorNullRoutedTail | None = None,
    schema: dict | None = None,
) -> dict:
    """Gate frozen graph semantics against a paired native-FP64 graph.

    The frozen-semantic result remains the deployable result.  The native-FP64
    branch is a sensitivity comparator only.  If ``tail`` is supplied, both
    branches evaluate the same complete raw ``core + tail`` scalar before
    taking its force; otherwise the raw core scalar is compared.  Taylor-null
    reference/mechanics gates remain separate mandatory tests.
    """
    if (tail is None) != (schema is None):
        raise ValueError("paired sensitivity requires both tail and schema, or neither")
    _validate_formal_query_binding(formal, query, schema)
    core = formal.model
    expected_semantics = frozen_graph_semantics()["graph_semantics_sha256"]
    if query.graph_semantics_sha256 != expected_semantics:
        raise ValueError("R2O query graph-semantics hash changed")
    formal_current = (
        query.current_positions_reference_order.detach().clone().requires_grad_(True)
    )
    native_current = (
        query.current_positions_reference_order.detach().clone().requires_grad_(True)
    )
    formal_positions = formal_current - query.image_shift.detach()
    native_positions = native_current - query.native_fp64_image_shift.detach()
    formal_features = _r2o_raw_node_features_prevalidated(
        core, query.data, formal_positions
    )
    native_features = _r2o_raw_node_features_prevalidated(
        core, query.native_fp64_data, native_positions
    )
    formal_energy = mace_interaction_energy(core, query.data, formal_positions)
    native_energy = mace_interaction_energy(
        core, query.native_fp64_data, native_positions
    )
    mode = "raw_core"
    if tail is not None and schema is not None:
        formal_energy = formal_energy + _raw_tail_energy_prevalidated(
            core, tail, query.data, formal_positions, schema
        )
        native_energy = native_energy + _raw_tail_energy_prevalidated(
            core, tail, query.native_fp64_data, native_positions, schema
        )
        mode = "raw_core_plus_tail"
    formal_force = -torch.autograd.grad(formal_energy.sum(), formal_current)[0]
    native_force = -torch.autograd.grad(native_energy.sum(), native_current)[0]
    energy_total_difference = float(
        torch.max(torch.abs(formal_energy - native_energy)).detach()
    )
    atom_count = int(query.reference_atom_count)
    energy_per_atom_difference = energy_total_difference / atom_count
    observed = {
        "raw_node_feature_max_abs": float(
            torch.max(torch.abs(formal_features - native_features)).detach()
        ),
        "raw_scalar_energy_max_abs_eV_per_atom": energy_per_atom_difference,
        "raw_scalar_force_max_abs_eV_A": float(
            torch.max(torch.abs(formal_force - native_force)).detach()
        ),
    }
    gates = {
        key: {
            "observed": observed[key],
            "threshold": threshold,
            "pass": bool(observed[key] <= threshold),
        }
        for key, threshold in NATIVE_FP64_SENSITIVITY_LIMITS.items()
    }
    return {
        "format": "graphene_r2o_paired_native_fp64_sensitivity_v1",
        "mode": mode,
        "deployment_branch": "frozen_graph_semantics",
        "comparator_branch": "native_FP64_not_for_deployment",
        "graph_semantics_sha256": expected_semantics,
        "source_default_dtype_pair": [
            str(FROZEN_GRAPH_SOURCE_DEFAULT_DTYPE),
            str(NATIVE_GRAPH_SOURCE_DEFAULT_DTYPE),
        ],
        "fixed_limits": dict(NATIVE_FP64_SENSITIVITY_LIMITS),
        "energy_size_normalization": {
            "atom_count": atom_count,
            "raw_scalar_energy_max_abs_total_eV": energy_total_difference,
            "raw_scalar_energy_max_abs_eV_per_atom": (
                energy_per_atom_difference
            ),
            "fixed_per_atom_limit_eV": NATIVE_FP64_SENSITIVITY_LIMITS[
                "raw_scalar_energy_max_abs_eV_per_atom"
            ],
            "derived_total_energy_cap_eV": atom_count
            * NATIVE_FP64_SENSITIVITY_LIMITS[
                "raw_scalar_energy_max_abs_eV_per_atom"
            ],
            "derivation": "original_6x6_total_limit_1e-6_eV_divided_by_72_atoms",
        },
        "observed": observed,
        "gates": gates,
        "passes_paired_native_FP64_sensitivity": all(
            record["pass"] for record in gates.values()
        ),
    }


def whole_energy_taylor2_null(
    raw_energy_fn: Callable[[torch.Tensor], torch.Tensor],
    current_positions: torch.Tensor,
    reference_positions: torch.Tensor,
    image_shift: torch.Tensor,
    *,
    create_graph: bool,
) -> TaylorResult:
    """FP64 public alias that makes the tail's whole-energy null explicit."""
    for name, tensor in (
        ("current_positions", current_positions),
        ("reference_positions", reference_positions),
        ("image_shift", image_shift),
    ):
        if tensor.dtype != torch.float64:
            raise ValueError(f"{name} must be FP64 for Taylor cancellation")
    return whole_energy_taylor2_remainder(
        raw_energy_fn,
        current_positions,
        reference_positions,
        image_shift,
        create_graph=create_graph,
    )


def tail_taylor_remainder(
    formal: LoadedFormalR2O,
    tail: R2OTaylorNullRoutedTail,
    schema: dict,
    query: R2OOrderMICQuery,
    *,
    create_graph: bool,
) -> TaylorResult:
    """Evaluate ``T_{>=3}[sum g*(epsilon-c_C)]`` conservatively."""
    _validate_formal_query_binding(formal, query, schema)
    core = formal.model
    ensure_fp64_module(tail, "R2O routed tail")
    return whole_energy_taylor2_null(
        lambda positions: _raw_tail_energy_prevalidated(
            core, tail, query.data, positions, schema
        ),
        query.current_positions_reference_order,
        query.reference_positions,
        query.image_shift,
        create_graph=create_graph,
    )


@dataclass
class CoreTailLinearityResult:
    core: TaylorResult
    tail: TaylorResult
    combined_once: TaylorResult
    energy_max_abs_difference_eV: float
    force_max_abs_difference_eV_A: float


def audit_core_tail_taylor_linearity(
    formal: LoadedFormalR2O,
    tail: R2OTaylorNullRoutedTail,
    schema: dict,
    query: R2OOrderMICQuery,
    *,
    energy_tolerance_eV: float = 1.0e-10,
    force_tolerance_eV_A: float = 1.0e-9,
) -> CoreTailLinearityResult:
    """Prove separate nulls equal one null of the combined raw scalar."""
    _validate_formal_query_binding(formal, query, schema)
    core = formal.model
    core_fn = lambda positions: mace_interaction_energy(core, query.data, positions)
    tail_fn = lambda positions: _raw_tail_energy_prevalidated(
        core, tail, query.data, positions, schema
    )
    core_result = whole_energy_taylor2_null(
        core_fn,
        query.current_positions_reference_order,
        query.reference_positions,
        query.image_shift,
        create_graph=False,
    )
    tail_result = whole_energy_taylor2_null(
        tail_fn,
        query.current_positions_reference_order,
        query.reference_positions,
        query.image_shift,
        create_graph=False,
    )
    combined = whole_energy_taylor2_null(
        lambda positions: core_fn(positions) + tail_fn(positions),
        query.current_positions_reference_order,
        query.reference_positions,
        query.image_shift,
        create_graph=False,
    )
    energy_difference = float(
        torch.max(torch.abs(combined.energy - core_result.energy - tail_result.energy))
    )
    force_difference = float(
        torch.max(
            torch.abs(
                combined.forces_reference_order
                - core_result.forces_reference_order
                - tail_result.forces_reference_order
            )
        )
    )
    if energy_difference > energy_tolerance_eV:
        raise ValueError("combined/separate Taylor-null energy linearity failed")
    if force_difference > force_tolerance_eV_A:
        raise ValueError("combined/separate Taylor-null force linearity failed")
    return CoreTailLinearityResult(
        core=core_result,
        tail=tail_result,
        combined_once=combined,
        energy_max_abs_difference_eV=energy_difference,
        force_max_abs_difference_eV_A=force_difference,
    )


def source_order_tail_forces(
    formal: LoadedFormalR2O,
    result: TaylorResult,
    query: R2OOrderMICQuery,
) -> torch.Tensor:
    """Map tail forces only after revalidating the query's formal MIC binding."""
    _validate_formal_query_binding(formal, query)
    expected_shape = (query.reference_atom_count, 3)
    if result.forces_reference_order.shape != expected_shape:
        raise ValueError("R2O tail force shape does not match its bound query")
    if result.forces_reference_order.dtype != torch.float64:
        raise ValueError("R2O tail source-order force mapping requires FP64")
    return source_order_forces(result.forces_reference_order, query.assignment)


def _validate_train8_pair(
    prediction_eV: torch.Tensor, target_eV: torch.Tensor
) -> None:
    if prediction_eV.dtype != torch.float64 or target_eV.dtype != torch.float64:
        raise ValueError("outer-LOCO energy gauge requires FP64 prediction and target")
    expected = (OUTER_LOCO_TRAIN_CONFIGURATIONS,)
    if prediction_eV.shape != expected or target_eV.shape != expected:
        raise ValueError("outer-LOCO energy gauge requires exactly train8 energies")
    if not bool(torch.isfinite(prediction_eV).all()) or not bool(
        torch.isfinite(target_eV).all()
    ):
        raise ValueError("outer-LOCO energy gauge received non-finite values")


def center_train8_prediction_and_target(
    prediction_eV: torch.Tensor, target_eV: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center prediction and target independently with their train8 means."""
    _validate_train8_pair(prediction_eV, target_eV)
    return (
        prediction_eV - torch.mean(prediction_eV),
        target_eV - torch.mean(target_eV),
    )


def train8_relative_energy_mse(
    prediction_eV: torch.Tensor, target_eV: torch.Tensor
) -> torch.Tensor:
    prediction_centered, target_centered = center_train8_prediction_and_target(
        prediction_eV, target_eV
    )
    return torch.mean(torch.square(prediction_centered - target_centered))


def held_energy_error_same_train8_gauge(
    held_prediction_eV: torch.Tensor,
    train8_prediction_eV: torch.Tensor,
    held_target_eV: torch.Tensor,
    train8_target_eV: torch.Tensor,
) -> torch.Tensor:
    """Held error after epoch/hash freeze, using the train8 means only."""
    _validate_train8_pair(train8_prediction_eV, train8_target_eV)
    for name, value in (
        ("held_prediction_eV", held_prediction_eV),
        ("held_target_eV", held_target_eV),
    ):
        if value.dtype != torch.float64 or value.numel() != 1 or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be one finite FP64 scalar")
    return (
        held_prediction_eV.reshape(()) - torch.mean(train8_prediction_eV)
    ) - (held_target_eV.reshape(()) - torch.mean(train8_target_eV))


def portable_artifact_record(path: str, digest: str) -> dict:
    _strict_relative_path(path, "portable artifact path")
    _validate_sha256(digest, "portable artifact SHA-256")
    return {"path": path, "sha256": digest}


def frozen_outer_loco_training_recipe() -> dict:
    """Canonical no-support training recipe; no hyperparameter is inferred."""
    return {
        "feature_scaler": {
            "source_rows": "all atom invariants from support train8 plus R2O replay-train only",
            "excluded": [
                "outer held",
                "E50 seed1",
                "harmonic full25 report",
                "E50 seed2",
                "reserved data",
            ],
            "mean": "population mean per each of 64 columns",
            "std": "population std per each of 64 columns (ddof=0)",
            "RMS": "sqrt(population mean(x^2)) per column",
            "scale": "max(std, 1e-6*RMS, 1e-8) per column",
            "feature_pruning": False,
            "fit_separately_per_fold": True,
        },
        "pristine_carbon_gauge": {
            "source_artifact": "pristine_reference_6x6",
            "reference_atom_count": 72,
            "definition": "mean raw tail epsilon over formal hash-bound pristine 6x6 carbon invariants",
            "maximum_relative_invariant_spread": 5.0e-5,
        },
        "optimizer": {
            "name": "AdamW",
            "amsgrad": True,
            "learning_rate": 1.0e-3,
            "constant_schedule": True,
            "weight_decay": 1.0e-6,
            "gradient_clip_global_norm": 20.0,
            "seed": 83,
            "EMA_decay": 0.99,
        },
        "per_epoch_exposure": {
            "optimizer_steps": 8,
            "support_train8": "each exactly once in seeded permutation",
            "small_harmonic32": "8 balanced cyclic selections",
            "E50_seed0_20": "8 balanced cyclic selections",
            "T300_36": "4 balanced cyclic selections",
            "T600_36": "4 balanced cyclic selections",
            "relative_energy": "full centered train8 MSE recomputed with current parameters at every optimizer step",
        },
        "loss": {
            "force_definition": "mean squared Cartesian error divided by the squared group force scale",
            "force_group_mass": {
                "support": 0.50,
                "small_harmonic": 0.25,
                "E50_seed0": 0.15,
                "auxiliary_T300_T600": 0.10,
            },
            "force_scale_eV_A": {
                "support": 0.030,
                "small_harmonic": 0.0005,
                "E50_seed0": 0.030,
                "auxiliary_T300_T600": 0.030,
            },
            "relative_energy_definition": "MSE of prediction and target after separately subtracting their train8 means",
            "relative_energy_weight": 0.25,
            "relative_energy_scale_eV": 0.0194,
            "harmonic_gate_off_weight": 0.05,
            "E50_gate_off_weight": 0.02,
            "gate_off_scale": 0.05,
        },
        "replay_schedule": {
            "support_permutation_formula": "default_rng(seed*1000003 + epoch*101 + salt).permutation(8)",
            "balanced_replay_formula": "one default_rng(seed*1000003 + salt) permutation; start=((epoch-1)*n_steps)%length; take cyclic n_steps",
            "auxiliary_interleave": "even optimizer steps T300 and odd optimizer steps T600 after their independent four-item cyclic selections",
            "support_permutation_salt": 11,
            "small_harmonic_salt": 23,
            "E50_seed0_salt": 37,
            "T300_salt": 53,
            "T600_salt": 71,
            "checkpoint_or_schedule_adaptation_from_held": False,
        },
    }


def _validate_outer_loco_source_records(
    source_artifacts: Mapping[str, Mapping[str, str]],
) -> None:
    if not isinstance(source_artifacts, Mapping):
        raise ValueError("R2O outer-LOCO source artifacts must be a mapping")
    names = set(source_artifacts)
    if not OUTER_LOCO_REQUIRED_SOURCE_ARTIFACTS.issubset(names):
        raise ValueError("R2O outer-LOCO required source artifacts are missing")
    if not names.issubset(OUTER_LOCO_SOURCE_ARTIFACT_WHITELIST):
        raise ValueError("R2O outer-LOCO source artifact name is not whitelisted")
    for name, record in source_artifacts.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"R2O outer-LOCO source record changed: {name}")
        path = str(record["path"])
        searchable = f"{name}/{path}".lower()
        if any(token in searchable for token in FORBIDDEN_UNOPENED_PATH_TOKENS):
            raise ValueError(
                f"R2O outer-LOCO source artifact contains a forbidden unopened token: {name}"
            )
        portable_artifact_record(path, str(record["sha256"]))


def make_outer_loco_contract(
    *,
    formal: LoadedFormalR2O,
    source_artifacts: Mapping[str, Mapping[str, str]],
) -> dict:
    """Create the no-support fixed policy that a later prepare must bind."""
    validate_loaded_formal_r2o(formal)
    _validate_outer_loco_source_records(source_artifacts)
    receipt_paths = formal.receipt["paths"]
    receipt_hashes = formal.receipt["sha256"]
    receipt_relative = formal.receipt_path.relative_to(formal.portable_root).as_posix()
    records = {
        "formal_receipt": portable_artifact_record(
            receipt_relative, formal.receipt_sha256
        ),
        "formal_gate": portable_artifact_record(
            receipt_paths["gate"], formal.gate_sha256
        ),
        "formal_bundle": portable_artifact_record(
            receipt_paths["bundle"], formal.bundle_sha256
        ),
        "pristine_reference_6x6": portable_artifact_record(
            receipt_paths["reference_6x6"], receipt_hashes["reference_6x6"]
        ),
        "source_artifacts": {
            name: portable_artifact_record(record["path"], record["sha256"])
            for name, record in sorted(source_artifacts.items())
        },
    }
    payload = {
        "format": OUTER_LOCO_CONTRACT_FORMAT,
        "status": OUTER_LOCO_CONTRACT_STATUS,
        "formal_R2O_condition": {
            "gate_status": FORMAL_GATE_STATUS,
            "bundle_kind": FORMAL_BUNDLE_KIND,
            "postcore_or_deployment_authorized": True,
            "raw_MACE_direct_deployment": False,
            "graph_semantics_sha256": frozen_graph_semantics()[
                "graph_semantics_sha256"
            ],
            "paired_native_FP64_sensitivity_gate_required": True,
        },
        "artifacts": records,
        "fold_policy": {
            "folds": OUTER_LOCO_FOLDS,
            "train_configurations_per_fold": OUTER_LOCO_TRAIN_CONFIGURATIONS,
            "held_configurations_per_fold": 1,
            "whole_configuration_LOCO": True,
        },
        "training_policy": {
            "epochs": OUTER_LOCO_EPOCHS,
            "EMA_endpoint_epoch": OUTER_LOCO_EPOCHS,
            "selection_policy": OUTER_LOCO_SELECTION_POLICY,
            "held_read_after_epoch240_EMA_and_model_hash_freeze": True,
        },
        "training_recipe": frozen_outer_loco_training_recipe(),
        "energy_gauge": {
            "prediction": "R_tail_i_minus_train8_mean_R_tail",
            "target": "d_i_minus_train8_mean_d",
            "loss": "mean_squared_difference_of_centered_prediction_and_target",
            "held_error": "same_train8_prediction_and_target_means_only",
        },
        "path_policy": {
            "all_artifact_paths_relative": True,
            "content_hash_required": True,
            "absolute_paths_allowed": False,
            "symlinks_allowed": False,
        },
        "leakage": {
            "support9_opened": False,
            "seed2_opened": False,
            "held_in_scaler_gradient_gauge_selection": False,
        },
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    validate_outer_loco_contract(payload, root=formal.portable_root)
    return payload


def validate_outer_loco_contract(payload: dict, root: Path | None = None) -> None:
    if not isinstance(payload, dict):
        raise ValueError("R2O outer-LOCO contract must be a mapping")
    body = dict(payload)
    recorded = body.pop("contract_sha256", None)
    if recorded is None or canonical_json_sha256(body) != recorded:
        raise ValueError("R2O outer-LOCO contract SHA-256 mismatch")
    if set(body) != {
        "format",
        "status",
        "formal_R2O_condition",
        "artifacts",
        "fold_policy",
        "training_policy",
        "training_recipe",
        "energy_gauge",
        "path_policy",
        "leakage",
    }:
        raise ValueError("R2O outer-LOCO contract fields changed")
    expected_sections = {
        "format": OUTER_LOCO_CONTRACT_FORMAT,
        "status": OUTER_LOCO_CONTRACT_STATUS,
        "formal_R2O_condition": {
            "gate_status": FORMAL_GATE_STATUS,
            "bundle_kind": FORMAL_BUNDLE_KIND,
            "postcore_or_deployment_authorized": True,
            "raw_MACE_direct_deployment": False,
            "graph_semantics_sha256": frozen_graph_semantics()[
                "graph_semantics_sha256"
            ],
            "paired_native_FP64_sensitivity_gate_required": True,
        },
        "fold_policy": {
            "folds": 9,
            "train_configurations_per_fold": 8,
            "held_configurations_per_fold": 1,
            "whole_configuration_LOCO": True,
        },
        "training_policy": {
            "epochs": 240,
            "EMA_endpoint_epoch": 240,
            "selection_policy": OUTER_LOCO_SELECTION_POLICY,
            "held_read_after_epoch240_EMA_and_model_hash_freeze": True,
        },
        "training_recipe": frozen_outer_loco_training_recipe(),
        "energy_gauge": {
            "prediction": "R_tail_i_minus_train8_mean_R_tail",
            "target": "d_i_minus_train8_mean_d",
            "loss": "mean_squared_difference_of_centered_prediction_and_target",
            "held_error": "same_train8_prediction_and_target_means_only",
        },
        "path_policy": {
            "all_artifact_paths_relative": True,
            "content_hash_required": True,
            "absolute_paths_allowed": False,
            "symlinks_allowed": False,
        },
        "leakage": {
            "support9_opened": False,
            "seed2_opened": False,
            "held_in_scaler_gradient_gauge_selection": False,
        },
    }
    for key, expected in expected_sections.items():
        if body[key] != expected:
            raise ValueError(f"R2O outer-LOCO fixed section changed: {key}")
    artifacts = body["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "formal_receipt",
        "formal_gate",
        "formal_bundle",
        "pristine_reference_6x6",
        "source_artifacts",
    }:
        raise ValueError("R2O outer-LOCO artifact records changed")
    records = {
        key: artifacts[key]
        for key in (
            "formal_receipt",
            "formal_gate",
            "formal_bundle",
            "pristine_reference_6x6",
        )
    }
    sources = artifacts["source_artifacts"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("R2O outer-LOCO source artifact records are empty")
    _validate_outer_loco_source_records(sources)
    records.update({f"source:{key}": value for key, value in sources.items()})
    for label, record in records.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"R2O outer-LOCO artifact record changed: {label}")
        _strict_relative_path(record["path"], f"outer-LOCO {label} path")
        _validate_sha256(record["sha256"], f"outer-LOCO {label} SHA-256")
        if root is not None:
            path = resolve_relative_artifact(root, record["path"], label)
            if sha256(path) != record["sha256"]:
                raise ValueError(f"R2O outer-LOCO artifact hash changed: {label}")


def strict_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
