#!/usr/bin/env python3
"""Build leakage-free finite-temperature graphene fine-tuning datasets.

Each physical-FD label file contains 12 training and three validation
structures.  A small deterministic replay subset from the v11 FC-distillation
data is mixed into each temperature-specific training set to reduce loss of the
already learned short-range harmonic curvature.  The optional final-wave
holdout keeps the last label file's three validation structures out of both
training and early stopping so model selection can use an independent test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from ase.io import read, write


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def require_reference_labels(atoms, source: Path) -> None:
    for index, structure in enumerate(atoms):
        if "REF_energy" not in structure.info or "REF_forces" not in structure.arrays:
            raise ValueError(f"missing REF labels in {source}, structure {index}")


def snapshot_index(structure) -> int:
    if "snapshot_index" in structure.info:
        return int(structure.info["snapshot_index"])
    match = re.search(r"snapshot_(\d+)", str(structure.info.get("config_type", "")))
    if not match:
        raise ValueError("cannot recover snapshot index from extxyz metadata")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-a", action="append", type=Path, required=True)
    parser.add_argument("--labels-b", action="append", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--replay-count", type=int, default=12)
    parser.add_argument(
        "--allow-unbalanced-replay",
        action="store_true",
        help="allow a replay count different from the number of thermal structures",
    )
    parser.add_argument(
        "--holdout-last-wave",
        action="store_true",
        help="write the last label wave's validation structures to test.xyz",
    )
    parser.add_argument(
        "--replay-validation",
        type=Path,
        help="append a labelled harmonic replay set to early-stopping validation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replay = read(args.replay, index=":")
    if args.replay_count < 0 or args.replay_count > len(replay):
        raise ValueError("--replay-count must be between zero and the replay-set size")
    require_reference_labels(replay, args.replay)
    replay_validation = (
        read(args.replay_validation, index=":")
        if args.replay_validation is not None
        else []
    )
    if args.replay_validation is not None:
        require_reference_labels(replay_validation, args.replay_validation)
    replay_indices = (
        np.linspace(0, len(replay) - 1, args.replay_count, dtype=int).tolist()
        if args.replay_count
        else []
    )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "temperature-specific v11 fine-tune with FC-distillation replay",
        "replay_source": str(args.replay),
        "replay_sha256": sha256(args.replay),
        "replay_indices": replay_indices,
        "replay_validation_source": (
            str(args.replay_validation) if args.replay_validation is not None else None
        ),
        "replay_validation_sha256": (
            sha256(args.replay_validation)
            if args.replay_validation is not None
            else None
        ),
        "temperatures": {},
    }
    for lane, temperature, sources in (
        ("A", 300, args.labels_a),
        ("B", 600, args.labels_b),
    ):
        training, validation, test, source_records = [], [], [], []
        for source_index, source in enumerate(sources):
            thermal = read(source, index=":")
            require_reference_labels(thermal, source)
            source_training = [
                atom.copy() for atom in thermal if atom.info.get("split") == "train"
            ]
            source_validation = [
                atom.copy()
                for atom in thermal
                if atom.info.get("split") == "validation"
            ]
            if len(source_training) != 12 or len(source_validation) != 3:
                raise ValueError(
                    f"{source} must contain 12 train and 3 validation structures; "
                    f"found {len(source_training)} and {len(source_validation)}"
                )
            training.extend(source_training)
            is_test_wave = args.holdout_last_wave and source_index == len(sources) - 1
            (test if is_test_wave else validation).extend(source_validation)
            source_records.append(
                {
                    "path": str(source),
                    "sha256": sha256(source),
                    "validation_role": "test" if is_test_wave else "early_stopping",
                }
            )
        if not validation:
            raise ValueError(f"lane {lane} has no early-stopping validation structures")
        all_indices = [snapshot_index(atom) for atom in training + validation + test]
        if len(all_indices) != len(set(all_indices)):
            raise ValueError(f"lane {lane} label waves contain duplicate snapshot indices")
        if not args.allow_unbalanced_replay and args.replay_count != len(training):
            raise ValueError(
                f"balanced replay requires {len(training)} replay structures; "
                f"received --replay-count={args.replay_count}"
            )
        for atom in training:
            atom.info["snapshot_index"] = snapshot_index(atom)
            atom.info["config_type"] = f"physical_fd_thermal_{temperature}K"
        for atom in validation:
            atom.info["snapshot_index"] = snapshot_index(atom)
            atom.info["config_type"] = f"physical_fd_thermal_{temperature}K_validation"
        for atom in test:
            atom.info["snapshot_index"] = snapshot_index(atom)
            atom.info["config_type"] = f"physical_fd_thermal_{temperature}K_test"
        replay_atoms = []
        for index in replay_indices:
            atom = replay[index].copy()
            atom.info["config_type"] = "v11_fc_distillation_replay"
            atom.info["split"] = "replay"
            replay_atoms.append(atom)

        # Interleave rather than concatenate so partial epochs see both data
        # sources.  Any longer source is retained in full.
        mixed = []
        for index in range(max(len(training), len(replay_atoms))):
            if index < len(training):
                mixed.append(training[index])
            if index < len(replay_atoms):
                mixed.append(replay_atoms[index])
        lane_dir = args.output / f"T{temperature}"
        lane_dir.mkdir(parents=True, exist_ok=True)
        write(lane_dir / "train.xyz", mixed, format="extxyz")
        joint_validation = validation + [atom.copy() for atom in replay_validation]
        write(lane_dir / "val.xyz", joint_validation, format="extxyz")
        if test:
            write(lane_dir / "test.xyz", test, format="extxyz")
        manifest["temperatures"][str(temperature)] = {
            "lane": lane,
            "thermal_sources": source_records,
            "n_thermal_train": len(training),
            "n_replay_train": len(replay_atoms),
            "n_validation": len(validation),
            "n_replay_validation": len(replay_validation),
            "n_joint_validation": len(joint_validation),
            "n_test": len(test),
            "validation_snapshot_indices": [
                int(atom.info["snapshot_index"]) for atom in validation
            ],
            "test_snapshot_indices": [
                int(atom.info["snapshot_index"]) for atom in test
            ],
        }

    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
