#!/usr/bin/env python3
"""Turn residual-force labels into targets for a frozen-backbone delta model.

The v11 short-range backbone is never updated.  A second, small MACE learns

    F_delta = (F_DFT - F_long) - F_v11.

At inference the conservative energies/forces of v11 and the delta network are
added, followed by the same harmonic long-range operator.  Harmonic replay
structures naturally have near-zero delta targets, which explicitly anchors
the correction instead of relying on a fine-tune to remember the backbone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read, write
from mace.calculators import MACECalculator


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


def is_thermal(atom) -> bool:
    return str(atom.info.get("force_target", "")).startswith("DFT_total_minus")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--thermal-repeat-train", type=int, default=2)
    parser.add_argument("--thermal-repeat-val", type=int, default=4)
    args = parser.parse_args()

    calculator = MACECalculator(
        model_paths=str(args.base_model),
        device=args.device,
        default_dtype="float32",
    )
    manifest = {
        "method": "frozen v11 plus additive delta MACE",
        "target": "(DFT total - harmonic long range) - v11",
        "input": str(args.input),
        "base_model": str(args.base_model),
        "base_model_sha256": sha256(args.base_model),
        "device_used_for_dataset_generation": args.device,
        "lanes": {},
    }
    for temperature in (300, 600):
        lane_metrics = {}
        output_lane = args.output / f"T{temperature}"
        output_lane.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            source = args.input / f"T{temperature}" / f"{split}.xyz"
            if not source.is_file():
                continue
            structures = read(source, index=":")
            delta_thermal = []
            delta_replay = []
            reconstruction_errors = []
            output = []
            for atom in structures:
                short_target = np.asarray(atom.arrays["REF_forces"], float).copy()
                short_energy = float(atom.info["REF_energy"])
                evaluated = atom.copy()
                evaluated.calc = calculator
                base_forces = np.asarray(evaluated.get_forces(), float)
                base_energy = float(evaluated.get_potential_energy())
                delta_forces = short_target - base_forces

                result = atom.copy()
                result.arrays["SHORT_RANGE_TARGET_forces"] = short_target
                result.info["SHORT_RANGE_TARGET_energy"] = short_energy
                result.arrays["BASE_forces"] = base_forces
                result.info["BASE_energy"] = base_energy
                result.arrays["REF_forces"] = delta_forces
                result.info["REF_energy"] = short_energy - base_energy
                result.info["force_target"] = "delta_on_frozen_v11"
                result.info["delta_target_role"] = (
                    "thermal" if is_thermal(atom) else "harmonic_replay"
                )

                long_range = np.asarray(result.arrays["LONG_RANGE_forces"], float)
                total = np.asarray(result.arrays["TOTAL_forces"], float)
                reconstructed = base_forces + delta_forces + long_range
                reconstruction_errors.append((reconstructed - total).reshape(-1))
                (delta_thermal if is_thermal(atom) else delta_replay).append(
                    delta_forces.reshape(-1)
                )
                output.append(result)
            repeat = (
                args.thermal_repeat_train
                if split == "train"
                else args.thermal_repeat_val if split == "val" else 1
            )
            expanded = []
            for result in output:
                count = repeat if result.info["delta_target_role"] == "thermal" else 1
                for repeat_index in range(count):
                    copy = result.copy()
                    copy.info["delta_repeat_index"] = repeat_index
                    expanded.append(copy)
            destination = output_lane / f"{split}.xyz"
            write(destination, expanded, format="extxyz")

            def force_metrics(values):
                joined = np.concatenate(values) if values else np.zeros(1)
                return {
                    "n_structures": len(values),
                    "RMSE_meV_A": float(np.sqrt(np.mean(joined**2)) * 1000.0),
                    "max_abs_meV_A": float(np.max(np.abs(joined)) * 1000.0),
                }

            reconstruction = np.concatenate(reconstruction_errors)
            lane_metrics[split] = {
                "source": str(source),
                "source_sha256": sha256(source),
                "output_sha256": sha256(destination),
                "n_unique_structures": len(structures),
                "n_output_structures": len(expanded),
                "thermal_repeat": repeat,
                "thermal_delta_target": force_metrics(delta_thermal),
                "harmonic_replay_delta_target": force_metrics(delta_replay),
                "reconstruction_max_abs_meV_A": float(
                    np.max(np.abs(reconstruction)) * 1000.0
                ),
            }
        operator_source = args.input / f"T{temperature}" / "long_range_operator.npz"
        operator_destination = output_lane / "long_range_operator.npz"
        operator_destination.write_bytes(operator_source.read_bytes())
        lane_metrics["operator_sha256"] = sha256(operator_destination)
        manifest["lanes"][str(temperature)] = lane_metrics

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
