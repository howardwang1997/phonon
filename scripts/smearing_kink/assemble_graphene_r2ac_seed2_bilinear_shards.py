#!/usr/bin/env python3
"""Assemble and verify the two promoted-seed2 geometry-design shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_SHARDS = tuple(
    BASE / f"R2AC_seed2_full_bilinear_shard_{start}_20260826"
    for start in (0, 10)
)
DEFAULT_OUTPUT = BASE / "R2AC_seed2_full_bilinear_20260826"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    shard_paths = tuple(args.shard) if args.shard else DEFAULT_SHARDS
    if len(shard_paths) != 2:
        raise ValueError("R2AC assembly requires exactly two shards")

    records = []
    energy_parts = []
    force_parts = []
    indices_parts = []
    scaler_hashes = set()
    input_hashes = set()
    column_labels = None
    for shard in shard_paths:
        root = shard.resolve()
        receipt_path = root / "receipt.json"
        energy_path = root / "bilinear_energy_design_eV.npy"
        force_path = root / "bilinear_force_design_eV_A.npy"
        receipt = json.loads(receipt_path.read_text())
        energy = np.load(energy_path, allow_pickle=False)
        force = np.load(force_path, allow_pickle=False)
        indices = np.asarray(receipt["structure_indices"], dtype=int)
        if receipt["data_role"] != "promoted_training_from_opened_development":
            raise ValueError(f"R2AC shard data role changed: {root}")
        if not receipt["geometry_only_materialization"] or receipt[
            "force_or_energy_labels_accessed"
        ]:
            raise ValueError(f"R2AC shard label-access declaration changed: {root}")
        if energy.shape != (len(indices), 512):
            raise ValueError(f"R2AC shard energy shape changed: {root}")
        if force.shape != (len(indices), 72, 3, 512):
            raise ValueError(f"R2AC shard force shape changed: {root}")
        if r2r1.raw_array_sha256(energy, "<f8") != receipt["array_raw_sha256"][
            "energy"
        ]:
            raise ValueError(f"R2AC shard energy differs from receipt: {root}")
        if r2r1.raw_array_sha256(force, "<f8") != receipt["array_raw_sha256"][
            "force"
        ]:
            raise ValueError(f"R2AC shard force differs from receipt: {root}")
        current_labels = receipt["column_labels"]
        if column_labels is None:
            column_labels = current_labels
        elif current_labels != column_labels:
            raise ValueError("R2AC shard column labels disagree")
        scaler_hashes.add(receipt["input_sha256"]["feature_scaler"])
        input_hashes.add(
            (
                receipt["input_sha256"]["endpoint"],
                receipt["input_sha256"]["reference6"],
                receipt["input_sha256"]["seed2_geometry_container"],
            )
        )
        energy_parts.append(energy)
        force_parts.append(force)
        indices_parts.append(indices)
        records.append(
            {
                "root": str(root),
                "receipt_sha256": file_sha256(receipt_path),
                "energy_file_sha256": file_sha256(energy_path),
                "force_file_sha256": file_sha256(force_path),
                "structure_indices": indices.tolist(),
            }
        )
    if len(scaler_hashes) != 1 or len(input_hashes) != 1:
        raise ValueError("R2AC shards did not use identical frozen inputs")

    joined_indices = np.concatenate(indices_parts)
    order = np.argsort(joined_indices)
    sorted_indices = joined_indices[order]
    if not np.array_equal(sorted_indices, np.arange(20, dtype=int)):
        raise ValueError("R2AC shard indices do not cover 0..19 exactly once")
    energy = np.concatenate(energy_parts, axis=0)[order]
    force = np.concatenate(force_parts, axis=0)[order]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    energy_path = output / "bilinear_energy_design_eV.npy"
    force_path = output / "bilinear_force_design_eV_A.npy"
    np.save(energy_path, np.asarray(energy, dtype="<f8"), allow_pickle=False)
    np.save(force_path, np.asarray(force, dtype="<f8"), allow_pickle=False)
    summary = {
        "format": "graphene_r2ac_seed2_full_bilinear_geometry_assembled_v1",
        "status": "R2AC_SEED2_FULL_BILINEAR_ASSEMBLED_AND_VERIFIED",
        "data_role": "promoted_training_from_opened_development",
        "structure_count": 20,
        "structure_indices": list(range(20)),
        "column_count": 512,
        "column_labels": column_labels,
        "geometry_only_materialization": True,
        "force_or_energy_labels_accessed": False,
        "frozen_feature_scaler_file_sha256": next(iter(scaler_hashes)),
        "frozen_inputs_sha256": {
            "endpoint": next(iter(input_hashes))[0],
            "reference6": next(iter(input_hashes))[1],
            "seed2_geometry_container": next(iter(input_hashes))[2],
        },
        "shards": records,
        "array_raw_sha256": {
            "energy": r2r1.raw_array_sha256(energy, "<f8"),
            "force": r2r1.raw_array_sha256(force, "<f8"),
        },
        "array_file_sha256": {
            "energy": file_sha256(energy_path),
            "force": file_sha256(force_path),
        },
    }
    (output / "receipt.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
