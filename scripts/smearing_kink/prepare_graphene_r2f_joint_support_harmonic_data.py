#!/usr/bin/env python3
"""Freeze support+harmonic full short-range targets for depth-3 last-layer tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
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


def assert_same_support(left: list, right: list) -> None:
    if len(left) != len(right):
        raise ValueError("support copies have different lengths")
    left_by_index = {int(item.info["sscha_index"]): item for item in left}
    right_by_index = {int(item.info["sscha_index"]): item for item in right}
    if set(left_by_index) != set(right_by_index):
        raise ValueError("support copies have different sscha indices")
    for index in sorted(left_by_index):
        a, b = left_by_index[index], right_by_index[index]
        if not np.allclose(a.positions, b.positions, atol=1.0e-12, rtol=0.0):
            raise ValueError(f"support geometry mismatch for sscha_index={index}")
        if not np.allclose(
            a.arrays["REF_forces"], b.arrays["REF_forces"], atol=1.0e-12, rtol=0.0
        ):
            raise ValueError(f"support force mismatch for sscha_index={index}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-weight", type=float, default=8.0)
    parser.add_argument("--harmonic-weight", type=float, default=8.0)
    args = parser.parse_args()

    source_manifest_path = args.input_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "frozen_before_R2C_training":
        raise ValueError("input is not the frozen R2C full-target dataset")
    if source_manifest.get("long_range_model_modified") is not False:
        raise ValueError("long-range model must remain frozen")
    if source_manifest.get("new_DFT", {}).get("n_supported") != 9:
        raise ValueError("expected nine historical support configurations")

    train_all = read(args.input_root / "train.xyz", index=":")
    val_all = read(args.input_root / "val.xyz", index=":")
    support = read(args.input_root / "supported9_train.xyz", index=":")
    test = read(args.input_root / "test.xyz", index=":")
    train_harmonic = [item for item in train_all if role(item) == "harmonic_replay"]
    val_harmonic = [item for item in val_all if role(item) == "harmonic_replay"]
    train_support = [
        item for item in train_all if role(item) == "fixed_smearing_support_repair"
    ]
    if (len(support), len(train_support), len(train_harmonic), len(val_harmonic)) != (
        9,
        9,
        72,
        25,
    ):
        raise ValueError("unexpected support/harmonic counts")
    assert_same_support(train_support, support)
    if any(role(item) != "thermal" for item in test):
        raise ValueError("test set must contain only frozen thermal configurations")

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
    weights["physical_s0_harmonic_replay_validation"] = float(
        args.harmonic_weight
    )
    weights["r2c_fixed_smearing_supported"] = float(args.support_weight)
    manifest = {
        "status": "frozen_before_R2F_depth3_last_layer_training",
        "scope": (
            "full short-range targets for depth-3 last-layer/readout tuning; "
            "thermal structures are test/gate only"
        ),
        "objective_hypothesis": (
            "the independently frozen additive adapter is too constrained; a "
            "small update of the depth-3 terminal representation may couple "
            "compressed radial and angular environments"
        ),
        "new_DFT_labels_for_this_stage": 0,
        "long_range_model_modified": False,
        "temperature_or_degauss_is_model_input": False,
        "thermal_structures_used_for_gradient_updates": 0,
        "config_type_weights": weights,
        "counts": {name: len(structures) for name, structures in outputs.items()},
        "inputs": {
            "source_manifest": {
                "path": str(source_manifest_path),
                "sha256": sha256(source_manifest_path),
            },
            "generator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
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
