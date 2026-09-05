#!/usr/bin/env python3
"""Append development thermal structures after hyperparameters have been frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ase.io import read, write


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development-validation", type=Path, required=True)
    parser.add_argument("--development-thermal", type=Path, required=True)
    parser.add_argument("--thermal-repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.thermal_repeat < 1:
        parser.error("--thermal-repeat must be positive")

    train = read(args.train, index=":")
    development_validation = read(args.development_validation, index=":")
    development_thermal = read(args.development_thermal, index=":")
    appended = []
    sources = []

    validation_by_index = {}
    for structure in development_validation:
        if str(structure.info.get("delta_target_role")) != "thermal":
            continue
        index = int(structure.info["snapshot_index"])
        validation_by_index.setdefault(index, structure)
    thermal_by_index = {}
    for structure in development_thermal:
        if str(structure.info.get("delta_target_role")) != "thermal":
            raise ValueError("development thermal file contains a non-thermal structure")
        index = int(structure.info["snapshot_index"])
        if index in thermal_by_index:
            raise ValueError(f"duplicate development thermal snapshot index {index}")
        thermal_by_index[index] = structure

    overlap = set(validation_by_index) & set(thermal_by_index)
    if overlap:
        raise ValueError(f"validation/test thermal snapshot overlap: {sorted(overlap)}")
    for source_name, structures in (
        ("validation", validation_by_index),
        ("checkpoint_gate", thermal_by_index),
    ):
        sources.append(
            {
                "role": source_name,
                "snapshot_indices": sorted(structures),
                "n_unique_thermal": len(structures),
            }
        )
        for index in sorted(structures):
            structure = structures[index]
            for repeat_index in range(args.thermal_repeat):
                result = structure.copy()
                result.info["config_type"] = "physical_fd_thermal_600K"
                result.info["split"] = "final_refit_train_after_hyperparameter_freeze"
                result.info["delta_repeat_index"] = repeat_index
                result.info["final_refit_source"] = source_name
                appended.append(result)
    output = [structure.copy() for structure in train] + appended
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, output, format="extxyz")
    payload = {
        "status": "complete",
        "scope": (
            "unique development thermal structures appended only after replay weight and "
            "epoch were frozen; independent 15-structure holdout remains unread"
        ),
        "source_train": str(args.train),
        "source_train_sha256": sha256(args.train),
        "development_validation": str(args.development_validation),
        "development_validation_sha256": sha256(args.development_validation),
        "development_thermal": str(args.development_thermal),
        "development_thermal_sha256": sha256(args.development_thermal),
        "development_sources": sources,
        "thermal_repeat": args.thermal_repeat,
        "n_source_train": len(train),
        "n_appended": len(appended),
        "n_output": len(output),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_name(args.manifest.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.manifest)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
