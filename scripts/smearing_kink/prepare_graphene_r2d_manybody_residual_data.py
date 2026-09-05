#!/usr/bin/env python3
"""Build additive residual targets on top of the frozen R2C depth-3 model.

The adapter is trained as a separate conservative energy model.  For every
configuration its labels are

    E_adapter = E_short,target - E_depth3
    F_adapter = F_short,target - F_depth3.

Adding the adapter and depth-3 predictions therefore recovers the original
short-range target.  No DFT value, q6 operator, or full-EPC parameter is
changed by this transformation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read, write
from mace.calculators import MACECalculator


SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_extxyz(path: Path, structures: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, path)


def force_metrics(values: list[np.ndarray]) -> dict[str, float]:
    joined = np.concatenate([np.asarray(value, float).reshape(-1) for value in values])
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(joined**2)) * 1000.0),
        "MAE_meV_A": float(np.mean(np.abs(joined)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(joined)) * 1000.0),
    }


def centered_energy_metrics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, float)
    centered = array - np.mean(array)
    return {
        "mean_offset_eV": float(np.mean(array)),
        "centered_RMSE_meV_config": float(np.sqrt(np.mean(centered**2)) * 1000.0),
        "centered_max_abs_meV_config": float(np.max(np.abs(centered)) * 1000.0),
    }


def transform(
    sources: list,
    calculator: MACECalculator,
    parent_model_hash: str,
) -> tuple[list, dict]:
    output = []
    force_by_type: dict[str, list[np.ndarray]] = defaultdict(list)
    energy_by_type: dict[str, list[float]] = defaultdict(list)
    for source in sources:
        atoms = source.copy()
        parent_target_energy = float(source.info["REF_energy"])
        parent_target_force = np.asarray(source.arrays["REF_forces"], float)

        atoms.calc = calculator
        depth_energy = float(atoms.get_potential_energy())
        depth_force = np.asarray(atoms.get_forces(), float)
        atoms.calc = None

        residual_energy = parent_target_energy - depth_energy
        residual_force = parent_target_force - depth_force
        config_type = str(source.info.get("config_type", "Default"))

        atoms.info.update(
            {
                "adapter_target_role": "target_minus_frozen_depth3",
                "adapter_parent_model_sha256": parent_model_hash,
                "PARENT_SHORT_RANGE_TARGET_energy": parent_target_energy,
                "DEPTH3_energy": depth_energy,
                "RESIDUAL_TARGET_energy": residual_energy,
                "REF_energy": residual_energy,
            }
        )
        atoms.arrays["PARENT_SHORT_RANGE_TARGET_forces"] = parent_target_force
        atoms.arrays["DEPTH3_forces"] = depth_force
        atoms.arrays["RESIDUAL_TARGET_forces"] = residual_force
        atoms.arrays["REF_forces"] = residual_force
        output.append(atoms)
        force_by_type[config_type].append(residual_force)
        energy_by_type[config_type].append(residual_energy)

    summary = {
        "n_structures": len(output),
        "all_residual_force": force_metrics(list(force_by_type_value for values in force_by_type.values() for force_by_type_value in values)),
        "by_config_type": {
            config_type: {
                "n_structures": len(force_by_type[config_type]),
                "residual_force": force_metrics(force_by_type[config_type]),
                "residual_energy_after_type_offset": centered_energy_metrics(
                    energy_by_type[config_type]
                ),
            }
            for config_type in sorted(force_by_type)
        },
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-data-root", type=Path, required=True)
    parser.add_argument("--depth3-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--support-weight", type=float, default=8.0)
    parser.add_argument("--harmonic-weight", type=float, default=16.0)
    args = parser.parse_args()

    parent_manifest_path = args.parent_data_root / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("status") != "frozen_before_R2C_training":
        raise ValueError("parent R2C data manifest is not frozen")
    if parent_manifest.get("long_range_model_modified") is not False:
        raise ValueError("parent data modified the long-range model")
    if parent_manifest.get("temperature_or_degauss_is_model_input") is not False:
        raise ValueError("temperature or degauss must not be a local-model input")

    parent_hash = sha256(args.depth3_model)
    calculator = MACECalculator(
        model_paths=str(args.depth3_model),
        device=args.device,
        default_dtype="float32",
    )
    inputs = {
        split: read(args.parent_data_root / f"{split}.xyz", index=":")
        for split in SPLITS
    }
    inputs["supported9_train"] = read(
        args.parent_data_root / "supported9_train.xyz", index=":"
    )

    outputs = {}
    diagnostics = {}
    for name, structures in inputs.items():
        outputs[name], diagnostics[name] = transform(
            structures, calculator, parent_hash
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, structures in outputs.items():
        atomic_extxyz(args.output_dir / f"{name}.xyz", structures)

    config_weights = dict(parent_manifest["config_type_weights"])
    config_weights["r2c_fixed_smearing_supported"] = float(args.support_weight)
    config_weights["Default"] = float(args.harmonic_weight)
    config_weights["physical_s0_harmonic_replay"] = float(args.harmonic_weight)
    config_weights["physical_s0_harmonic_replay_validation"] = float(
        args.harmonic_weight
    )
    manifest = {
        "status": "frozen_before_R2D_local_adapter_training",
        "scope": (
            "small-cutoff conservative many-body adapter on frozen depth-3; "
            "no new DFT and no long-range modification"
        ),
        "target_definition": {
            "energy": "R2C short-range target minus frozen depth-3 energy",
            "forces": "R2C short-range target minus frozen depth-3 forces",
            "combination": "frozen depth-3 plus independently trained adapter",
        },
        "long_range_model_modified": False,
        "new_DFT_labels": 0,
        "temperature_or_degauss_is_model_input": False,
        "config_type_weights": config_weights,
        "counts": {name: len(structures) for name, structures in outputs.items()},
        "residual_diagnostics": diagnostics,
        "inputs": {
            "parent_manifest": {
                "path": str(parent_manifest_path),
                "sha256": sha256(parent_manifest_path),
            },
            "depth3_model": {
                "path": str(args.depth3_model),
                "sha256": parent_hash,
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
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "train_residual": diagnostics["train"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
