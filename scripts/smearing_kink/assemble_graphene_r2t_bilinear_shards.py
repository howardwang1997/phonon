#!/usr/bin/env python3
"""Assemble and verify the four geometry-only R2T bilinear shards."""

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
    BASE / f"R2T_full_bilinear_shard_{start}_20260826"
    for start in (0, 23, 46, 69)
)
DEFAULT_CROSS32 = BASE / "R2S_node_nonlinear_basis_diagnostic_20260826"
DEFAULT_OUTPUT = BASE / "R2T_full_bilinear_materialization_20260826"
DIAGONAL_INDICES = np.asarray(
    [left * 16 + left for left in range(16)]
    + [256 + left * 16 + left for left in range(16)],
    dtype=int,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append")
    parser.add_argument("--cross32-root", type=Path, default=DEFAULT_CROSS32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    shard_paths = tuple(args.shard) if args.shard else DEFAULT_SHARDS
    if len(shard_paths) != 4:
        raise ValueError("R2T assembly requires exactly four shards")

    records = []
    energy_parts = []
    force_parts = []
    indices_parts = []
    scaler_hashes = set()
    for shard in shard_paths:
        root = shard.resolve()
        receipt_path = root / "receipt.json"
        energy_path = root / "bilinear_energy_design_eV.npy"
        force_path = root / "bilinear_force_design_eV_A.npy"
        receipt = json.loads(receipt_path.read_text())
        energy = np.load(energy_path, allow_pickle=False)
        force = np.load(force_path, allow_pickle=False)
        indices = np.asarray(receipt["structure_indices"], dtype=int)
        if energy.shape != (len(indices), 512):
            raise ValueError(f"R2T shard energy shape changed: {root}")
        if force.shape != (len(indices), 72, 3, 512):
            raise ValueError(f"R2T shard force shape changed: {root}")
        if r2r1.raw_array_sha256(energy, "<f8") != receipt["array_raw_sha256"][
            "energy"
        ]:
            raise ValueError(f"R2T shard energy differs from receipt: {root}")
        if r2r1.raw_array_sha256(force, "<f8") != receipt["array_raw_sha256"][
            "force"
        ]:
            raise ValueError(f"R2T shard force differs from receipt: {root}")
        scaler_hashes.add(
            (
                receipt["feature_scaler"]["mean_raw_sha256"],
                receipt["feature_scaler"]["scale_raw_sha256"],
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
    if len(scaler_hashes) != 1:
        raise ValueError("R2T shards did not use one frozen feature scaler")

    joined_indices = np.concatenate(indices_parts)
    order = np.argsort(joined_indices)
    sorted_indices = joined_indices[order]
    if not np.array_equal(sorted_indices, np.arange(92, dtype=int)):
        raise ValueError("R2T shard indices do not cover global indices 0..91 exactly once")
    energy = np.concatenate(energy_parts, axis=0)[order]
    force = np.concatenate(force_parts, axis=0)[order]

    with np.load(
        args.cross32_root / "node_nonlinear_design_and_oof.npz",
        allow_pickle=False,
    ) as arrays:
        old_energy = np.asarray(arrays["nonlinear_energy_design"], dtype=np.float64)
        old_force = np.asarray(arrays["nonlinear_force_design"], dtype=np.float64)
    diagonal_energy_difference = np.take(energy, DIAGONAL_INDICES, axis=-1) - old_energy[
        ..., 64:96
    ]
    diagonal_force_difference = np.take(force, DIAGONAL_INDICES, axis=-1) - old_force[
        ..., 64:96
    ]
    agreement = {
        "energy_max_abs_eV": float(np.max(np.abs(diagonal_energy_difference))),
        "energy_RMS_eV": float(np.sqrt(np.mean(np.square(diagonal_energy_difference)))),
        "force_max_abs_eV_A": float(np.max(np.abs(diagonal_force_difference))),
        "force_RMS_eV_A": float(np.sqrt(np.mean(np.square(diagonal_force_difference)))),
        "tolerance": 1.0e-10,
    }
    agreement["pass"] = bool(
        agreement["energy_max_abs_eV"] <= agreement["tolerance"]
        and agreement["force_max_abs_eV_A"] <= agreement["tolerance"]
    )
    if not agreement["pass"]:
        raise ValueError("R2T diagonal channels disagree with the frozen cross32 design")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    energy_path = output / "bilinear_energy_design_eV.npy"
    force_path = output / "bilinear_force_design_eV_A.npy"
    np.save(energy_path, energy, allow_pickle=False)
    np.save(force_path, force, allow_pickle=False)
    summary = {
        "format": "graphene_r2t_full_bilinear_geometry_design_assembled_v1",
        "status": "FULL_GEOMETRY_DESIGN_COMPLETE_AND_CROSS32_VERIFIED",
        "structure_count": 92,
        "column_count": 512,
        "label_access": False,
        "frozen_feature_scaler_raw_sha256": {
            "mean": next(iter(scaler_hashes))[0],
            "scale": next(iter(scaler_hashes))[1],
        },
        "shards": records,
        "cross32_diagonal_agreement": agreement,
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
