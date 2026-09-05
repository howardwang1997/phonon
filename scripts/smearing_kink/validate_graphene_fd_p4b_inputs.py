#!/usr/bin/env python3
"""Validate the frozen 450 K trajectories before releasing DFT force labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_path(value: str) -> tuple[int, Path]:
    try:
        seed, raw_path = value.split("=", 1)
        return int(seed), Path(raw_path)
    except ValueError as error:
        raise argparse.ArgumentTypeError("snapshots must use SEED=/path/to/snapshots.npz") from error


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--sampling-acceptance", type=Path, required=True)
    parser.add_argument("--sampling-bootstrap", type=Path, required=True)
    parser.add_argument("--snapshot", type=seed_path, action="append", required=True)
    parser.add_argument("--shard", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text())
    acceptance = json.loads(args.sampling_acceptance.read_text())
    bootstrap = json.loads(args.sampling_bootstrap.read_text())
    freeze_hash = sha256(args.freeze_manifest)
    if freeze["status"] != "frozen_before_450_holdout" or freeze["direct_450_targets_read"]:
        raise ValueError("the 450 K predictor was not frozen before direct targets")
    if acceptance["status"] != "passed" or not acceptance["passes_sampling_gate"]:
        raise ValueError("the 450 K sampling gate did not pass")
    if bootstrap["status"] != "passed" or not bootstrap["passes_sampling_bootstrap_gate"]:
        raise ValueError("the 450 K sampling bootstrap gate did not pass")
    if acceptance["freeze_manifest"]["sha256"] != freeze_hash:
        raise ValueError("sampling acceptance refers to a different freeze manifest")
    if bootstrap["freeze_manifest_sha256"] != freeze_hash:
        raise ValueError("sampling bootstrap refers to a different freeze manifest")
    if bootstrap["sampling_acceptance_sha256"] != sha256(args.sampling_acceptance):
        raise ValueError("sampling bootstrap refers to a different point-estimate acceptance")

    protocol = freeze["T450_on_policy"]
    if float(acceptance["temperature_K"]) != float(protocol["temperature_K"]):
        raise ValueError("sampling temperature differs from the frozen protocol")
    if not np.isclose(
        float(acceptance["degauss_Ry"]), float(protocol["degauss_Ry"]), rtol=0.0, atol=1e-14
    ):
        raise ValueError("sampling degauss differs from the frozen protocol")

    all_indices = [int(value) for value in protocol["dft_label_indices_per_seed"]]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("expected 20 distinct frozen label indices per seed")
    midpoint = len(all_indices) // 2
    assignments = {
        "A": {0: all_indices, 2: all_indices[:midpoint]},
        "B": {1: all_indices, 2: all_indices[midpoint:]},
    }
    expected_descriptions = {
        "A": "seed0 all 20 plus seed2 first 10",
        "B": "seed1 all 20 plus seed2 last 10",
    }
    machine_key = f"V100-{args.shard}"
    if protocol["dft_shards"].get(machine_key) != expected_descriptions[args.shard]:
        raise ValueError(f"frozen {machine_key} assignment differs from the implemented split")

    provided = dict(args.snapshot)
    if set(provided) != set(assignments[args.shard]):
        raise ValueError(
            f"shard {args.shard} requires seeds {sorted(assignments[args.shard])}, "
            f"received {sorted(provided)}"
        )
    checkpoint_records = {int(item["seed"]): item for item in acceptance["checkpoints"]}
    snapshots = []
    for seed, path in sorted(provided.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256(path)
        expected = checkpoint_records[seed]
        if observed_hash != expected["snapshots_sha256"]:
            raise ValueError(f"seed {seed} snapshot SHA-256 differs from sampling acceptance")
        with np.load(path, allow_pickle=False) as data:
            positions = np.asarray(data["positions"])
            if "cells" in data.files:
                cells = np.asarray(data["cells"])
            elif "cell" in data.files:
                cells = np.repeat(np.asarray(data["cell"])[None, :, :], len(positions), axis=0)
            else:
                raise KeyError(f"{path} contains neither cells nor cell")
        expected_count = int(checkpoint_records[seed]["n_snapshots"])
        if expected_count < int(protocol["snapshots_per_seed"]):
            raise ValueError(f"seed {seed} has fewer than the frozen minimum snapshots")
        if positions.shape != (expected_count, 72, 3):
            raise ValueError(f"unexpected seed {seed} positions shape {positions.shape}")
        if cells.shape != (len(positions), 3, 3):
            raise ValueError(f"unexpected seed {seed} cells shape {cells.shape}")
        snapshots.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": observed_hash,
                "n_snapshots": len(positions),
                "selected_indices": assignments[args.shard][seed],
            }
        )

    payload = {
        "status": "validated_for_dft_force_labels",
        "shard": args.shard,
        "logical_executor": machine_key,
        "temperature_K": protocol["temperature_K"],
        "degauss_Ry": protocol["degauss_Ry"],
        "dft_force_settings": protocol["dft_force_settings"],
        "freeze_manifest_sha256": freeze_hash,
        "sampling_acceptance_sha256": sha256(args.sampling_acceptance),
        "sampling_bootstrap_sha256": sha256(args.sampling_bootstrap),
        "snapshots": snapshots,
        "n_selected_structures": sum(len(item["selected_indices"]) for item in snapshots),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
