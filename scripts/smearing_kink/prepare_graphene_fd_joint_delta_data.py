#!/usr/bin/env python3
"""Combine the 300/600 K residual lanes for one joint delta-MACE.

Thermal configurations from both temperatures are retained with equal counts.
The identical harmonic replay set is included once, rather than duplicated by
temperature.  The output manifest records a 300 K configuration weight that
equalizes the two endpoint thermal force-MSE scales relative to 600 K.
"""
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


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def role(structure) -> str:
    value = str(structure.info.get("delta_target_role", ""))
    if value not in {"thermal", "harmonic_replay"}:
        raise ValueError(f"unexpected delta_target_role={value!r}")
    return value


def replay_equal(left, right) -> bool:
    return bool(
        np.array_equal(left.numbers, right.numbers)
        and np.allclose(left.cell, right.cell, rtol=0.0, atol=1e-12)
        and np.allclose(left.positions, right.positions, rtol=0.0, atol=1e-12)
        and np.allclose(
            left.arrays["REF_forces"], right.arrays["REF_forces"], rtol=0.0, atol=1e-12
        )
        and abs(float(left.info["REF_energy"]) - float(right.info["REF_energy"]))
        <= 1e-12
    )


def force_metrics(structures) -> dict[str, float | int]:
    values = np.concatenate(
        [np.asarray(item.arrays["REF_forces"], float).reshape(-1) for item in structures]
    )
    return {
        "n_structures": len(structures),
        "RMSE_meV_A": float(np.sqrt(np.mean(values**2)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(values)) * 1000.0),
    }


def interleave(*groups):
    output = []
    for index in range(max(len(group) for group in groups)):
        for group in groups:
            if index < len(group):
                output.append(group[index].copy())
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads(args.source_manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "joint 300/600 K frozen-v11 additive delta dataset",
        "target": source_manifest["target"],
        "base_model": source_manifest["base_model"],
        "base_model_sha256": source_manifest["base_model_sha256"],
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "splits": {},
    }
    train_thermal_metrics = {}
    for split in ("train", "val", "test"):
        by_temperature = {}
        replay_by_temperature = {}
        source_records = {}
        for temperature in (300, 600):
            path = args.input_root / f"T{temperature}" / f"{split}.xyz"
            structures = read(path, index=":")
            by_temperature[temperature] = [item for item in structures if role(item) == "thermal"]
            replay_by_temperature[temperature] = [
                item for item in structures if role(item) == "harmonic_replay"
            ]
            source_records[str(temperature)] = {
                "path": str(path),
                "sha256": sha256(path),
                "n_thermal": len(by_temperature[temperature]),
                "n_harmonic_replay": len(replay_by_temperature[temperature]),
            }
        if len(by_temperature[300]) != len(by_temperature[600]):
            raise ValueError(f"{split} endpoint thermal counts are not balanced")

        replay300 = replay_by_temperature[300]
        replay600 = replay_by_temperature[600]
        if len(replay300) != len(replay600):
            raise ValueError(f"{split} endpoint replay counts differ")
        if any(not replay_equal(left, right) for left, right in zip(replay300, replay600)):
            raise ValueError(f"{split} endpoint harmonic replay sets are not identical")
        for item in replay300:
            item.info["config_type"] = (
                "v11_fc_distillation_replay_validation"
                if split == "val"
                else "v11_fc_distillation_replay"
            )

        output = interleave(by_temperature[300], by_temperature[600], replay300)
        destination = args.output / f"{split}.xyz"
        write(destination, output, format="extxyz")
        split_record = {
            "sources": source_records,
            "n_output_structures": len(output),
            "n_thermal_300": len(by_temperature[300]),
            "n_thermal_600": len(by_temperature[600]),
            "n_harmonic_replay": len(replay300),
            "output": str(destination),
            "output_sha256": sha256(destination),
        }
        for temperature in (300, 600):
            metrics = force_metrics(by_temperature[temperature])
            split_record[f"thermal_{temperature}_delta_target"] = metrics
            if split == "train":
                train_thermal_metrics[temperature] = metrics
        if replay300:
            split_record["harmonic_replay_delta_target"] = force_metrics(replay300)
        result["splits"][split] = split_record

    rms300 = float(train_thermal_metrics[300]["RMSE_meV_A"])
    rms600 = float(train_thermal_metrics[600]["RMSE_meV_A"])
    result["recommended_config_weights"] = {
        "thermal_300_relative_to_600": (rms600 / rms300) ** 2,
        "thermal_600": 1.0,
        "method": "inverse endpoint target-MSE normalization with 600 K as unit weight",
    }
    atomic_json(args.output / "manifest.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
