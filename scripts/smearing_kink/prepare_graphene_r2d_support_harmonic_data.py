#!/usr/bin/env python3
"""Filter R2D residual data to supported DFT plus harmonic replay.

This development-only objective tests whether thermal residual targets were
suppressing the compressed-environment correction.  Thermal configurations
remain untouched test/gate data and never enter gradient updates.
"""
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


def atomic_extxyz(path: Path, structures: list) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def role(structure) -> str:
    return str(structure.info.get("delta_target_role", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-weight", type=float, default=8.0)
    parser.add_argument("--harmonic-weight", type=float, default=8.0)
    args = parser.parse_args()

    source_manifest_path = args.input_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "frozen_before_R2D_local_adapter_training":
        raise ValueError("input is not the frozen full R2D residual dataset")
    if source_manifest.get("new_DFT_labels") != 0:
        raise ValueError("unexpected new DFT labels")
    if source_manifest.get("long_range_model_modified") is not False:
        raise ValueError("long-range model must remain frozen")

    train_all = read(args.input_root / "train.xyz", index=":")
    val_all = read(args.input_root / "val.xyz", index=":")
    support = read(args.input_root / "supported9_train.xyz", index=":")
    test = read(args.input_root / "test.xyz", index=":")
    train_harmonic = [structure for structure in train_all if role(structure) == "harmonic_replay"]
    val_harmonic = [structure for structure in val_all if role(structure) == "harmonic_replay"]
    if (len(support), len(train_harmonic), len(val_harmonic)) != (9, 72, 25):
        raise ValueError("unexpected support/harmonic counts")

    outputs = {
        "train": train_harmonic + support,
        "val": val_harmonic + support,
        "test": test,
        "supported9_train": support,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, structures in outputs.items():
        atomic_extxyz(args.output_dir / f"{name}.xyz", structures)

    weights = dict(source_manifest["config_type_weights"])
    weights["Default"] = float(args.harmonic_weight)
    weights["physical_s0_harmonic_replay"] = float(args.harmonic_weight)
    weights["physical_s0_harmonic_replay_validation"] = float(args.harmonic_weight)
    weights["r2c_fixed_smearing_supported"] = float(args.support_weight)
    manifest = {
        "status": "frozen_before_R2D_local_adapter_training",
        "scope": (
            "supported-DFT plus harmonic-residual objective; thermal structures "
            "are test/gate only"
        ),
        "objective_hypothesis": (
            "low-correlation thermal residual targets suppress the supported "
            "compressed-environment correction"
        ),
        "new_DFT_labels": 0,
        "long_range_model_modified": False,
        "temperature_or_degauss_is_model_input": False,
        "thermal_structures_used_for_gradient_updates": 0,
        "config_type_weights": weights,
        "counts": {name: len(structures) for name, structures in outputs.items()},
        "inputs": {
            "source_manifest": {
                "path": str(source_manifest_path),
                "sha256": sha256(source_manifest_path),
            }
        },
        "outputs": {},
    }
    for name in outputs:
        path = args.output_dir / f"{name}.xyz"
        manifest["outputs"][path.name] = {
            "path": str(path),
            "sha256": sha256(path),
        }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
