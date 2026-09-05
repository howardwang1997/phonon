#!/usr/bin/env python3
"""Independently recalculate a completed R2P primary certificate from arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "graphene_r2p_independent_artifact_recalculation_v1"
FORBIDDEN_PATH_TOKENS = ("seed2", "support", "reserved", "outer_fold")
ACTIVE_WEIGHT_TOL = 1.0e-10
COMPARE_ABS_TOL = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.dtype("<f8")))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def canonical_sha256(payload: Any) -> str:
    content = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def strict_json_load(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value} in {path}")

    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object in {path}")
    return result


def strict_json_write(path: Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def checked_root(path: Path) -> Path:
    candidate = Path(path)
    if any(token in str(candidate).lower() for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError("R2P independent audit refuses forbidden path token")
    if ".." in candidate.parts:
        raise ValueError("R2P independent audit refuses path traversal")
    candidate = candidate.resolve(strict=True)
    if not candidate.is_dir():
        raise NotADirectoryError(candidate)
    return candidate


def require_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.parent != root or candidate.is_symlink():
        raise ValueError(f"invalid R2P artifact path {relative}")
    path = candidate.resolve(strict=True)
    if path.parent != root or not path.is_file():
        raise ValueError(f"invalid R2P artifact path {relative}")
    return path


def recalculate(run_root: Path) -> dict:
    root = checked_root(run_root)
    paths = {
        name: require_file(root, name)
        for name in (
            "DONE",
            "EXIT_CODE",
            "PRIMARY_CERTIFICATE_FROZEN",
            "audit_contract.json",
            "graphene_r2p_gradient_feasibility_snapshot.py",
            "primary_certificate.json",
            "primary_common_direction.npy",
            "primary_gradient_matrix.npy",
            "primary_normalized_gram.npy",
            "report_only_comparisons.json",
            "result.json",
        )
    }
    file_hashes = {name: sha256(path) for name, path in paths.items()}
    if paths["EXIT_CODE"].read_bytes() != b"0\n":
        raise ValueError("R2P EXIT_CODE is not exact zero")
    if paths["DONE"].read_bytes() != b"R2P gradient feasibility complete\n":
        raise ValueError("R2P DONE marker changed")
    marker_hash = paths["PRIMARY_CERTIFICATE_FROZEN"].read_text(
        encoding="ascii"
    ).strip()
    if marker_hash != file_hashes["primary_certificate.json"]:
        raise ValueError("primary freeze marker does not match primary certificate")

    contract = strict_json_load(paths["audit_contract.json"])
    primary = strict_json_load(paths["primary_certificate.json"])
    report = strict_json_load(paths["report_only_comparisons.json"])
    result = strict_json_load(paths["result.json"])
    protocol = contract["protocol"]
    contract_without_self = dict(contract)
    expected_contract_canonical_hash = contract_without_self.pop("contract_sha256")
    binary_mapping = {
        "primary_gradient_matrix": "primary_gradient_matrix.npy",
        "primary_normalized_gram": "primary_normalized_gram.npy",
        "primary_common_direction": "primary_common_direction.npy",
    }
    linkage = {
        "protocol_canonical_hash_matches": (
            canonical_sha256(protocol) == contract["protocol_sha256"]
        ),
        "contract_canonical_hash_matches": (
            canonical_sha256(contract_without_self) == expected_contract_canonical_hash
        ),
        "result_contract_file_hash_matches": (
            result["contract"]["sha256"] == file_hashes["audit_contract.json"]
        ),
        "result_source_snapshot_hash_matches": (
            result["source_snapshot"]["sha256"]
            == file_hashes["graphene_r2p_gradient_feasibility_snapshot.py"]
        ),
        "result_primary_hash_matches": (
            result["primary_certificate"]["sha256"]
            == file_hashes["primary_certificate.json"]
        ),
        "result_report_hash_matches": (
            result["report_only_comparisons"]["sha256"]
            == file_hashes["report_only_comparisons.json"]
        ),
        "report_primary_hash_matches": (
            report["primary_certificate_sha256"]
            == file_hashes["primary_certificate.json"]
            == report["primary_certificate_rechecked_sha256"]
        ),
        "primary_contract_hash_matches": (
            primary["contract_file_sha256"] == file_hashes["audit_contract.json"]
        ),
        "result_relative_paths_match": (
            result["contract"]["relative_path"] == "audit_contract.json"
            and result["source_snapshot"]["relative_path"]
            == "graphene_r2p_gradient_feasibility_snapshot.py"
            and result["primary_certificate"]["relative_path"]
            == "primary_certificate.json"
            and result["report_only_comparisons"]["relative_path"]
            == "report_only_comparisons.json"
        ),
        "contract_source_snapshot_binding_matches": (
            contract["R2P_source_snapshot"]["relative_path"]
            == "graphene_r2p_gradient_feasibility_snapshot.py"
            and contract["R2P_source_snapshot"]["sha256"]
            == file_hashes["graphene_r2p_gradient_feasibility_snapshot.py"]
        ),
        "binary_relative_paths_and_file_hashes_match": all(
            primary["binary_artifacts"][logical]["relative_path"] == filename
            and primary["binary_artifacts"][logical]["sha256"]
            == file_hashes[filename]
            for logical, filename in binary_mapping.items()
        ),
    }
    if not all(linkage.values()):
        raise ValueError(f"R2P receipt linkage failed: {linkage}")

    matrix = np.load(paths["primary_gradient_matrix.npy"], allow_pickle=False)
    gram_file = np.load(paths["primary_normalized_gram.npy"], allow_pickle=False)
    direction_file = np.load(
        paths["primary_common_direction.npy"], allow_pickle=False
    )
    if matrix.dtype != np.float64 or matrix.shape != (176, 42096):
        raise ValueError("R2P gradient matrix shape/dtype changed")
    if gram_file.dtype != np.float64 or gram_file.shape != (176, 176):
        raise ValueError("R2P Gram shape/dtype changed")
    if direction_file.dtype != np.float64 or direction_file.shape != (42096,):
        raise ValueError("R2P direction shape/dtype changed")
    semantic_hashes = {
        "gradient_matrix": array_sha256(matrix),
        "normalized_gram": array_sha256(gram_file),
        "common_direction": array_sha256(direction_file),
    }
    expected_semantic_hashes = {
        "gradient_matrix": primary["gradient_matrix_sha256"],
        "normalized_gram": primary["normalized_gram_sha256"],
        "common_direction": primary["direction_sha256"],
    }
    if semantic_hashes != expected_semantic_hashes:
        raise ValueError("R2P array semantic hash mismatch")

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("R2P matrix has invalid gradient norms")
    units = matrix / norms[:, None]
    direct_gram = units @ units.T
    direction_norm = float(np.linalg.norm(direction_file))
    direct_cosines = units @ direction_file
    records = primary["per_block"]
    if len(records) != 176:
        raise ValueError("R2P per-block record count changed")
    weights = np.asarray([row["dual_weight"] for row in records], dtype=np.float64)
    stored_cosines = np.asarray(
        [row["common_direction_cosine"] for row in records], dtype=np.float64
    )
    stored_norms = np.asarray([row["gradient_norm"] for row in records])
    combination = units.T @ weights
    dual_norm = float(np.linalg.norm(combination))
    dual_direction = combination / dual_norm
    selected_gradient = direct_gram @ weights
    norm_squared = dual_norm**2
    slack = selected_gradient - norm_squared
    active = weights > ACTIVE_WEIGHT_TOL
    direct = {
        "direction_norm": direction_norm,
        "direction_vs_dual_max_abs": float(np.max(np.abs(direction_file - dual_direction))),
        "gram_file_vs_direct_max_abs": float(np.max(np.abs(gram_file - direct_gram))),
        "stored_vs_direct_gradient_norm_max_abs": float(
            np.max(np.abs(stored_norms - norms))
        ),
        "stored_vs_direct_cosine_max_abs": float(
            np.max(np.abs(stored_cosines - direct_cosines))
        ),
        "primal_minimum_cosine": float(np.min(direct_cosines)),
        "primal_maximum_cosine": float(np.max(direct_cosines)),
        "dual_norm": dual_norm,
        "cosine_duality_gap": float(dual_norm - np.min(direct_cosines)),
        "simplex_sum": float(np.sum(weights)),
        "simplex_sum_residual": abs(float(np.sum(weights)) - 1.0),
        "minimum_weight": float(np.min(weights)),
        "active_weight_tolerance": ACTIVE_WEIGHT_TOL,
        "active_count": int(np.sum(active)),
        "active_slack_max_abs": float(np.max(np.abs(slack[active]))),
        "inactive_slack_min": float(np.min(slack[~active])),
        "minimum_cosine_strict_GO_margin_above_1e-3": float(
            np.min(direct_cosines) - 1.0e-3
        ),
    }
    comparisons = {
        "direction": direct["direction_vs_dual_max_abs"] <= COMPARE_ABS_TOL,
        "gram": direct["gram_file_vs_direct_max_abs"] <= COMPARE_ABS_TOL,
        "gradient_norms": (
            direct["stored_vs_direct_gradient_norm_max_abs"] <= COMPARE_ABS_TOL
        ),
        "cosines": direct["stored_vs_direct_cosine_max_abs"] <= COMPARE_ABS_TOL,
        "primal": abs(
            direct["primal_minimum_cosine"]
            - primary["solver"]["primal_minimum_cosine"]
        )
        <= COMPARE_ABS_TOL,
        "dual": abs(direct["dual_norm"] - primary["solver"]["dual_minimum_norm"])
        <= COMPARE_ABS_TOL,
        "simplex": direct["simplex_sum_residual"] <= COMPARE_ABS_TOL,
    }
    if not all(comparisons.values()):
        raise ValueError(f"R2P direct numerical comparison failed: {comparisons}")

    family_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        family_indices[row["family"]].append(index)
    family_summary = {}
    for family, indices_list in family_indices.items():
        indices = np.asarray(indices_list, dtype=int)
        family_cosines = direct_cosines[indices]
        family_weights = weights[indices]
        worst_local = int(np.argmin(family_cosines))
        worst_index = int(indices[worst_local])
        family_summary[family] = {
            "count": int(indices.size),
            "minimum_cosine": float(np.min(family_cosines)),
            "median_cosine": float(np.median(family_cosines)),
            "maximum_cosine": float(np.max(family_cosines)),
            "worst_block_index": worst_index,
            "worst_block_id": records[worst_index]["block_id"],
            "dual_weight_sum": float(np.sum(family_weights)),
            "active_count": int(np.sum(family_weights > ACTIVE_WEIGHT_TOL)),
        }
    per_block = []
    for index, row in enumerate(records):
        per_block.append(
            {
                "block_index": index,
                "block_id": row["block_id"],
                "family": row["family"],
                "gradient_norm": float(norms[index]),
                "direct_cosine": float(direct_cosines[index]),
                "dual_weight": float(weights[index]),
                "KKT_slack": float(slack[index]),
                "active": bool(active[index]),
            }
        )
    active_constraints = [row for row in per_block if row["active"]]
    highest_dual_weights = sorted(
        per_block, key=lambda row: (-row["dual_weight"], row["block_index"])
    )[:20]

    report_only_audit = {
        "report_status": report["status"],
        "primary_hash_before_report_matches_after_report": linkage[
            "report_primary_hash_matches"
        ],
        "formal_gate_prediction_parity_pass": report[
            "formal_gate_prediction_parity"
        ]["pass"],
        "formal_gate_prediction_parity_max_abs_meV_A": report[
            "formal_gate_prediction_parity"
        ]["maximum_absolute_numeric_difference"],
        "state_after_report_matches_primary_start": (
            report["model_state_sha256_after_all_report_only_work"]
            == primary["model_state_sha256_before"]
        ),
        "comparison_cones": {},
    }
    for name, cone in report["comparison_cones"].items():
        report_only_audit["comparison_cones"][name] = {
            "authorization_role": cone["authorization_role"],
            "can_authorize_parameter_update_or_training": cone[
                "can_authorize_parameter_update_or_training"
            ],
            "can_authorize_primary_direction": cone[
                "can_authorize_primary_direction"
            ],
            "scientific_status_report_only": cone["solver"]["scientific_status"],
            "block_count": cone["block_count"],
        }
    if any(
        item["can_authorize_parameter_update_or_training"]
        or item["can_authorize_primary_direction"]
        or not item["authorization_role"].startswith("report_only_")
        for item in report_only_audit["comparison_cones"].values()
    ):
        raise ValueError("a report-only cone acquired authorization")
    if not report_only_audit["state_after_report_matches_primary_start"]:
        raise ValueError("report-only work changed the model state")

    family_counts = Counter(row["family"] for row in records)
    result_pass = bool(
        result["scientific_primary_status"] == "GO"
        and result["primary_common_descent_pass"] is True
        and result["postcore_or_training_authorized"] is False
        and result["state_unchanged"] is True
        and all(linkage.values())
        and all(comparisons.values())
        and direct["primal_minimum_cosine"] > 1.0e-3
        and direct["cosine_duality_gap"] <= 1.0e-3
        and direct["active_slack_max_abs"] <= 1.0e-3
        and report_only_audit["formal_gate_prediction_parity_pass"]
    )
    return {
        "format": FORMAT,
        "status": "PASS" if result_pass else "FAIL",
        "scientific_interpretation": (
            "the frozen EMA240 point has a certified local first-order direction "
            "that decreases all 176 normalized training hard blocks under a "
            "negative step; this does not authorize training or postcore"
        ),
        "input_root": str(root),
        "input_file_sha256": file_hashes,
        "array_semantic_sha256": semantic_hashes,
        "receipt_linkage": linkage,
        "direct_recalculation": direct,
        "direct_comparison_pass": comparisons,
        "family_counts": dict(family_counts),
        "family_summary": family_summary,
        "active_constraints": active_constraints,
        "highest_20_dual_weights": highest_dual_weights,
        "per_block_recalculated": per_block,
        "report_only_noninterference": report_only_audit,
        "optimizer_or_training_authorized": False,
        "postcore_or_tail_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = recalculate(args.run_root)
    if args.output is None:
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
    else:
        strict_json_write(args.output, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "output": str(args.output),
                    "sha256": sha256(args.output),
                },
                sort_keys=True,
            )
        )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
