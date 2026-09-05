#!/usr/bin/env python3
"""Assemble and verify the two promoted-seed2 base-design shards."""

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
    BASE / f"R2AC_seed2_base_design_shard_{start}_20260826"
    for start in (0, 10)
)
DEFAULT_OUTPUT = BASE / "R2AC_seed2_base_design_20260826"
ARRAY_KEYS = (
    "fixed_energy_eV",
    "fixed_force_eV_A",
    "parameter_energy_design_eV",
    "parameter_force_design_eV_A",
)
EXPECTED_SHAPES = {
    "fixed_energy_eV": (),
    "fixed_force_eV_A": (72, 3),
    "parameter_energy_design_eV": (65,),
    "parameter_force_design_eV_A": (72, 3, 65),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    shard_paths = tuple(args.shard) if args.shard else DEFAULT_SHARDS
    if len(shard_paths) != 2:
        raise ValueError("R2AC base assembly requires exactly two shards")

    records = []
    parts = {key: [] for key in ARRAY_KEYS}
    indices_parts = []
    input_hashes = set()
    for shard in shard_paths:
        root = shard.resolve()
        receipt_path = root / "receipt.json"
        arrays_path = root / "seed2_base_design.npz"
        receipt = json.loads(receipt_path.read_text())
        if file_sha256(arrays_path) != receipt["arrays_file_sha256"]:
            raise ValueError(f"R2AC base shard differs from receipt: {root}")
        if receipt["data_role"] != "promoted_training_from_opened_development":
            raise ValueError(f"R2AC base shard data role changed: {root}")
        if not receipt["geometry_only_materialization"] or receipt[
            "force_or_energy_labels_accessed"
        ]:
            raise ValueError(f"R2AC base shard label declaration changed: {root}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            indices = np.asarray(arrays["structure_indices"], dtype=int)
            for key in ARRAY_KEYS:
                value = np.asarray(arrays[key], dtype=np.float64)
                if value.shape != (len(indices),) + EXPECTED_SHAPES[key]:
                    raise ValueError(f"R2AC base {key} shape changed: {root}")
                raw_key = {
                    "fixed_energy_eV": "fixed_energy",
                    "fixed_force_eV_A": "fixed_force",
                    "parameter_energy_design_eV": "parameter_energy",
                    "parameter_force_design_eV_A": "parameter_force",
                }[key]
                if r2r1.raw_array_sha256(value, "<f8") != receipt[
                    "array_raw_sha256"
                ][raw_key]:
                    raise ValueError(f"R2AC base {key} raw hash changed: {root}")
                parts[key].append(value)
        indices_parts.append(indices)
        input_hashes.add(tuple(receipt["input_sha256"].values()))
        records.append(
            {
                "root": str(root),
                "receipt_sha256": file_sha256(receipt_path),
                "arrays_sha256": file_sha256(arrays_path),
                "structure_indices": indices.tolist(),
            }
        )
    if len(input_hashes) != 1:
        raise ValueError("R2AC base shards did not use identical frozen inputs")
    joined_indices = np.concatenate(indices_parts)
    order = np.argsort(joined_indices)
    if not np.array_equal(joined_indices[order], np.arange(20, dtype=int)):
        raise ValueError("R2AC base shards do not cover indices 0..19 exactly once")
    combined = {
        key: np.concatenate(value, axis=0)[order] for key, value in parts.items()
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "seed2_base_design.npz"
    np.savez(
        arrays_path,
        structure_indices=np.arange(20, dtype="<i8"),
        **{key: np.asarray(value, dtype="<f8") for key, value in combined.items()},
    )
    summary = {
        "format": "graphene_r2ac_seed2_base_geometry_design_assembled_v1",
        "status": "R2AC_SEED2_BASE_DESIGN_ASSEMBLED_AND_VERIFIED",
        "data_role": "promoted_training_from_opened_development",
        "structure_count": 20,
        "structure_indices": list(range(20)),
        "column_count": 65,
        "geometry_only_materialization": True,
        "force_or_energy_labels_accessed": False,
        "shards": records,
        "array_raw_sha256": {
            key: r2r1.raw_array_sha256(value, "<f8")
            for key, value in combined.items()
        },
        "arrays_file_sha256": file_sha256(arrays_path),
    }
    (output / "receipt.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
