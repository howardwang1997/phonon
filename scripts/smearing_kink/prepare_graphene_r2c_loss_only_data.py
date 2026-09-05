#!/usr/bin/env python3
"""Prepare the first order-safe short-range repair dataset for graphene.

The existing S0 target is a local delta potential added to a frozen base MACE
and a conservative q6 electronic operator.  This script augments that target
with only the historical DFT geometries that were classified, before reading
their DFT values, as lying inside the corrected X0 geometry support.

For every added configuration,

    F_delta = F_DFT - F_base - F_q6(T300),
    E_delta = E_DFT - E_base - E_q6(T300).

One global energy constant is removed from every compatible thermal target.
Harmonic replay remains a force/Hessian anchor and receives zero energy
weight because its source uses a different absolute energy convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from mace.calculators import MACECalculator


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    load_operator,
    structure_mapping,
)


SPLITS = ("train", "val", "test")
NEW_CONFIG_TYPE = "r2c_fixed_smearing_supported"


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def energy_range(structures: list, key: str) -> dict[str, float | int | None]:
    values = [float(atoms.info[key]) for atoms in structures if key in atoms.info]
    if not values:
        return {"n": 0, "min_eV": None, "max_eV": None, "mean_eV": None}
    array = np.asarray(values, float)
    return {
        "n": len(values),
        "min_eV": float(np.min(array)),
        "max_eV": float(np.max(array)),
        "mean_eV": float(np.mean(array)),
    }


def force_metrics(values: list[np.ndarray]) -> dict[str, float]:
    joined = np.concatenate([np.asarray(value, float).reshape(-1) for value in values])
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(joined**2)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(joined)) * 1000.0),
    }


def prepare_new_structures(
    dft_structures: list,
    support: dict,
    base_model: Path,
    operator: Path,
    device: str,
) -> tuple[list, list, dict]:
    records = {
        int(record["sscha_index"]): record for record in support["records"]
    }
    if set(records) != {int(atoms.info["sscha_index"]) for atoms in dft_structures}:
        raise ValueError("geometry-support record and DFT structures differ")

    force_constants, reference, cell, degauss = load_operator(operator)
    calculator = MACECalculator(
        model_paths=str(base_model), device=device, default_dtype="float32"
    )
    included = []
    excluded = []
    base_force_values = []
    long_force_values = []
    mapping_maximum_displacements = []
    new_records = []
    for source in dft_structures:
        index = int(source.info["sscha_index"])
        level = str(records[index]["corrected_X0_support_level"])
        if level == "outside_corrected_X0_support":
            excluded.append(source.copy())
            continue

        atoms = source.copy()
        atoms.calc = calculator
        base_energy = float(atoms.get_potential_energy())
        base_force = np.asarray(atoms.get_forces(), float)
        atoms.calc = None

        mapping, displacement = structure_mapping(atoms, reference, cell)
        reordered_fc = force_constants[mapping][:, mapping]
        phi_u = np.einsum("ijab,jb->ia", reordered_fc, displacement)
        long_force = -phi_u
        long_energy = 0.5 * float(np.einsum("ia,ia->", displacement, phi_u))
        dft_force = np.asarray(source.arrays["REF_forces"], float)
        dft_energy = float(source.info["REF_energy"])
        delta_force = dft_force - base_force - long_force
        delta_energy = dft_energy - base_energy - long_energy

        atoms.info.update(
            {
                "config_type": NEW_CONFIG_TYPE,
                "config_energy_weight": 1.0,
                "config_forces_weight": 1.0,
                "source_family": "historical_X0_geometry_supported_DFT",
                "split": "train",
                "delta_target_role": "fixed_smearing_support_repair",
                "corrected_X0_support_level": level,
                "lattice_temperature_K": 450.0,
                "degauss_Ry": degauss,
                "long_range_degauss_Ry": degauss,
                "temperature_is_model_input": False,
                "TOTAL_energy": dft_energy,
                "BASE_energy": base_energy,
                "LONG_RANGE_energy": long_energy,
                "RAW_SHORT_RANGE_TARGET_energy": delta_energy,
                "SHORT_RANGE_TARGET_energy": delta_energy,
                "REF_energy": delta_energy,
            }
        )
        atoms.arrays["TOTAL_forces"] = dft_force
        atoms.arrays["BASE_forces"] = base_force
        atoms.arrays["LONG_RANGE_forces"] = long_force
        atoms.arrays["SHORT_RANGE_TARGET_forces"] = delta_force
        atoms.arrays["REF_forces"] = delta_force
        atoms.arrays["operator_mapping"] = mapping
        included.append(atoms)
        base_force_values.append(base_force)
        long_force_values.append(long_force)
        mapping_maximum_displacements.append(
            float(np.max(np.linalg.norm(displacement, axis=1)))
        )
        new_records.append(
            {
                "sscha_index": index,
                "support_level": level,
                "DFT_energy_eV": dft_energy,
                "base_energy_eV": base_energy,
                "q6_energy_eV": long_energy,
                "raw_delta_target_energy_eV": delta_energy,
                "delta_target_force": force_metrics([delta_force]),
            }
        )

    if len(included) != 9 or len(excluded) != 3:
        raise ValueError(
            f"expected 9 supported and 3 excluded structures, got "
            f"{len(included)} and {len(excluded)}"
        )
    provenance = {
        "n_supported": len(included),
        "n_excluded": len(excluded),
        "supported_sscha_indices": [int(a.info["sscha_index"]) for a in included],
        "excluded_sscha_indices": [int(a.info["sscha_index"]) for a in excluded],
        "base_force": force_metrics(base_force_values),
        "q6_force": force_metrics(long_force_values),
        "maximum_atom_displacement_A": float(max(mapping_maximum_displacements)),
        "records": new_records,
    }
    return included, excluded, provenance


def is_harmonic_replay(atoms) -> bool:
    return str(atoms.info.get("delta_target_role", "")) == "harmonic_replay"


def unique_thermal_energy_values(structures: list) -> list[float]:
    values = []
    for atoms in structures:
        if is_harmonic_replay(atoms):
            continue
        if int(atoms.info.get("delta_repeat_index", 0)) != 0:
            continue
        values.append(float(atoms.info["REF_energy"]))
    return values


def center_targets(structures: list, energy_zero: float) -> list:
    centered = []
    for source in structures:
        atoms = source.copy()
        atoms.calc = None
        raw_energy = float(atoms.info["REF_energy"])
        atoms.info["RAW_SHORT_RANGE_TARGET_energy"] = raw_energy
        atoms.info["config_forces_weight"] = 1.0
        if is_harmonic_replay(atoms):
            atoms.info["config_energy_weight"] = 0.0
            centered_energy = 0.0
        else:
            atoms.info["config_energy_weight"] = 1.0
            centered_energy = raw_energy - energy_zero
        atoms.info["REF_energy"] = centered_energy
        atoms.info["SHORT_RANGE_TARGET_energy"] = centered_energy
        centered.append(atoms)
    return centered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data-root", type=Path, required=True)
    parser.add_argument("--dft-dataset", type=Path, required=True)
    parser.add_argument("--geometry-support", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--new-config-weight", type=float, default=4.0)
    args = parser.parse_args()

    support = load_json(args.geometry_support)
    if support.get("status") != "geometry_only_support_frozen":
        raise ValueError("geometry support was not frozen")
    if support.get("DFT_energy_or_force_read") is not False:
        raise ValueError("geometry support classification read DFT values")
    dft_structures = read(args.dft_dataset, index=":")
    if len(dft_structures) != 12:
        raise ValueError("expected exactly 12 completed DFT structures")

    original = {
        split: read(args.base_data_root / f"{split}.xyz", index=":")
        for split in SPLITS
    }
    supported, excluded, new_provenance = prepare_new_structures(
        dft_structures,
        support,
        args.base_model,
        args.operator,
        args.device,
    )
    zero_candidates = unique_thermal_energy_values(original["train"])
    zero_candidates.extend(float(atoms.info["REF_energy"]) for atoms in supported)
    energy_zero = float(np.median(np.asarray(zero_candidates, float)))

    output = {
        "train": center_targets(original["train"] + supported, energy_zero),
        "val": center_targets(original["val"], energy_zero),
        "test": center_targets(original["test"], energy_zero),
    }
    supported_centered = center_targets(supported, energy_zero)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        atomic_extxyz(args.output_dir / f"{split}.xyz", output[split])
    atomic_extxyz(args.output_dir / "supported9_train.xyz", supported_centered)
    atomic_extxyz(args.output_dir / "excluded_outside3.extxyz", excluded)

    base_manifest_path = args.base_data_root / "manifest.json"
    base_manifest = load_json(base_manifest_path)
    config_weights = dict(
        base_manifest["recommended_config_weights"]["config_type_weights"]
    )
    config_weights["Default"] = 16.0
    config_weights["physical_s0_harmonic_replay"] = 16.0
    config_weights["physical_s0_harmonic_replay_validation"] = 16.0
    config_weights[NEW_CONFIG_TYPE] = float(args.new_config_weight)

    manifest = {
        "status": "frozen_before_R2C_training",
        "scope": (
            "first order-safe short-range loss/data ablation; nine historical "
            "DFT geometries inside corrected-X0 geometry support"
        ),
        "long_range_model_modified": False,
        "temperature_or_degauss_is_model_input": False,
        "energy_policy": {
            "global_short_range_energy_zero_eV": energy_zero,
            "thermal_energy_weight": 1.0,
            "harmonic_replay_energy_weight": 0.0,
            "reason": (
                "thermal labels share the same DFT/base convention; harmonic "
                "replay uses another absolute zero and remains a force anchor"
            ),
        },
        "config_type_weights": config_weights,
        "new_DFT": new_provenance,
        "counts": {split: len(output[split]) for split in SPLITS},
        "energy_ranges_after_centering": {
            split: energy_range(output[split], "REF_energy") for split in SPLITS
        },
        "inputs": {
            "base_data_manifest": {
                "path": str(base_manifest_path),
                "sha256": sha256(base_manifest_path),
            },
            "dft_dataset": {
                "path": str(args.dft_dataset),
                "sha256": sha256(args.dft_dataset),
            },
            "geometry_support": {
                "path": str(args.geometry_support),
                "sha256": sha256(args.geometry_support),
            },
            "base_model": {
                "path": str(args.base_model),
                "sha256": sha256(args.base_model),
            },
            "operator": {
                "path": str(args.operator),
                "sha256": sha256(args.operator),
            },
        },
        "outputs": {},
    }
    for name in [*SPLITS, "supported9_train", "excluded_outside3"]:
        suffix = ".xyz" if name != "excluded_outside3" else ".extxyz"
        path = args.output_dir / f"{name}{suffix}"
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
                "energy_zero_eV": energy_zero,
                "new_DFT": new_provenance,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
