#!/usr/bin/env python3
"""Freeze the support-free R2M replacement-core training data.

The new core *replaces* every support-exposed short-delta model.  The frozen
v11 foundation and q6 operator remain outside it.  Consequently the exact
fixed-smearing target is ``F_DFT - F_v11 - F_q6``; the harmonic and legacy
thermal files already store the corresponding short-delta target in
``REF_forces``.  Energies are deliberately disabled in this first force-only
representation lane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


FIXED_DEGAUSS_RY = 0.0019000869380739254
EXPECTED_HASHES = {
    "e50": "b5170472fc637e336f3856981f9db0d3e81eac047be36e6cafb76088e0dedc8e",
    "r0_npz": "08c92de89bc330c2c64c81fd0387b995a4614ddb4a56aac4da3fa3a16f08691e",
    "r0_summary": "9c9d7ec16302c5e86c021a793c8a269ab88f294999336a789774061fcf5edb28",
    "r2c_train": "db5bae0b8643fce8bcd1946f2619169efa6f82a7267292cec86fc2db0271937d",
    "r2c_valid": "cda0c678db7c0ea326bbda1dfe5de56d84fd2e5df275cbdd3186043b082c93e8",
    "r2c_manifest": "3d582d69ba5c5144a0a6029c0c1097e5e150256ffbe430dd3057556b15b627b6",
}
EXPECTED_RMS_MEV_A = {
    "exact_e50": 204.2305,
    "harmonic": 7.36940,
    "auxiliary": 197.1261,
}
GROUP_MASS = {"exact_e50": 0.4, "harmonic": 0.4, "auxiliary": 0.2}
EXPECTED_WEIGHT = {
    "exact_e50": 0.00468733,
    "harmonic": 1.0,
    "auxiliary": 0.000698789,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_extxyz(path: Path, structures: list[Atoms]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, path)


def fingerprint(structure: Atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(structure.numbers, np.dtype("<i8")).tobytes())
    digest.update(np.round(np.asarray(structure.cell, float), 10).astype("<f8").tobytes())
    digest.update(
        np.round(np.asarray(structure.positions, float), 10).astype("<f8").tobytes()
    )
    return digest.hexdigest()


def force_metrics(values: list[np.ndarray]) -> dict:
    joined = np.concatenate([np.asarray(value, float).reshape(-1) for value in values])
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(joined**2))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(joined))),
        "n_force_components": int(joined.size),
    }


def role(structure: Atoms) -> str:
    return str(structure.info.get("delta_target_role", ""))


def clone_target(
    source: Atoms,
    target: np.ndarray,
    config_type: str,
    target_role: str,
    source_scope: str,
) -> Atoms:
    atoms = source.copy()
    atoms.calc = None
    original_force = np.asarray(target, float).copy()
    atoms.arrays["CORE_TARGET_forces"] = original_force
    atoms.arrays["REF_forces"] = original_force
    atoms.info.update(
        {
            "REF_energy": 0.0,
            "config_energy_weight": 0.0,
            "config_forces_weight": 1.0,
            "config_type": config_type,
            "r2m_target_role": target_role,
            "r2m_source_scope": source_scope,
            "r2m_core_replaces_depth3_and_current_s0": True,
            "temperature_is_model_input": False,
            "smearing_is_model_input": False,
        }
    )
    return atoms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e50", type=Path, required=True)
    parser.add_argument("--r0-npz", type=Path, required=True)
    parser.add_argument("--r0-summary", type=Path, required=True)
    parser.add_argument("--r2c-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "e50": args.e50,
        "r0_npz": args.r0_npz,
        "r0_summary": args.r0_summary,
        "r2c_train": args.r2c_root / "train.xyz",
        "r2c_valid": args.r2c_root / "val.xyz",
        "r2c_manifest": args.r2c_root / "manifest.json",
    }
    observed_hashes = {name: sha256(path) for name, path in paths.items()}
    if observed_hashes != EXPECTED_HASHES:
        raise ValueError(
            f"frozen input hash mismatch: observed={observed_hashes}, "
            f"expected={EXPECTED_HASHES}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output {args.output_dir}")

    r0_summary = json.loads(args.r0_summary.read_text(encoding="utf-8"))
    if r0_summary.get("status") != "current_S0_fixed_smearing_audit_complete":
        raise ValueError("R0 summary status is not frozen complete")
    if r0_summary["condition"] != {
        "lattice_temperature_K": 450.0,
        "smearing": "fermi-dirac",
        "degauss_Ry": FIXED_DEGAUSS_RY,
    }:
        raise ValueError("R0 condition is not the fixed E50 condition")
    r2c_manifest = json.loads(paths["r2c_manifest"].read_text(encoding="utf-8"))
    if (
        r2c_manifest.get("status") != "frozen_before_R2C_training"
        or r2c_manifest.get("long_range_model_modified") is not False
        or r2c_manifest.get("temperature_or_degauss_is_model_input") is not False
    ):
        raise ValueError("R2C source manifest violates the frozen short-target contract")

    e50 = read(args.e50, index=":")
    r2c_train = read(paths["r2c_train"], index=":")
    r2c_valid = read(paths["r2c_valid"], index=":")
    if len(e50) != 60:
        raise ValueError("E50 must contain 60 configurations")
    seed_count = Counter(int(item.info["trajectory_seed"]) for item in e50)
    if seed_count != Counter({0: 20, 1: 20, 2: 20}):
        raise ValueError(f"unexpected E50 seed counts: {seed_count}")
    if any(
        float(item.info["lattice_temperature_K"]) != 450.0
        or abs(float(item.info["degauss_Ry"]) - FIXED_DEGAUSS_RY) > 5.0e-10
        for item in e50
    ):
        raise ValueError("E50 structures do not share the fixed condition")

    with np.load(args.r0_npz, allow_pickle=False) as data:
        required = {
            "base_forces_eV_A",
            "delta_forces_eV_A",
            "long_range_forces_eV_A",
            "DFT_target_forces_eV_A",
            "trajectory_seed",
            "snapshot_index",
            "Aprime_mode_operator_order",
            "Aprime_coordinate_A",
            "structure_to_operator_mapping",
        }
        if not required.issubset(data.files):
            raise ValueError(f"R0 NPZ lacks {sorted(required - set(data.files))}")
        base_force = np.asarray(data["base_forces_eV_A"], float)
        current_s0_force = np.asarray(data["delta_forces_eV_A"], float)
        long_force = np.asarray(data["long_range_forces_eV_A"], float)
        dft_force = np.asarray(data["DFT_target_forces_eV_A"], float)
        npz_seed = np.asarray(data["trajectory_seed"], int)
        npz_snapshot = np.asarray(data["snapshot_index"], int)
        mode = np.asarray(data["Aprime_mode_operator_order"], complex)
        coordinate = np.asarray(data["Aprime_coordinate_A"], complex)
        mapping = np.asarray(data["structure_to_operator_mapping"], int)
    if base_force.shape != (60, 72, 3) or mode.shape != (72, 3):
        raise ValueError("unexpected R0 array shapes")

    e50_by_seed: dict[int, list[Atoms]] = {0: [], 1: [], 2: []}
    e50_targets: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    for index, source in enumerate(e50):
        seed = int(source.info["trajectory_seed"])
        snapshot = int(source.info["snapshot_index"])
        if (seed, snapshot) != (int(npz_seed[index]), int(npz_snapshot[index])):
            raise ValueError("E50 and R0 NPZ ordering/provenance disagree")
        if not np.allclose(
            np.asarray(source.arrays["REF_forces"], float),
            dft_force[index],
            atol=2.0e-10,
            rtol=0.0,
        ):
            raise ValueError("E50 DFT force and R0 NPZ disagree")
        target = dft_force[index] - base_force[index] - long_force[index]
        reconstructed = base_force[index] + long_force[index] + target
        if not np.allclose(reconstructed, dft_force[index], atol=2.0e-12, rtol=0.0):
            raise RuntimeError("E50 replacement-core target does not reconstruct DFT")
        split = {0: "train", 1: "validation", 2: "opened_development"}[seed]
        config_type = {
            0: "r2m_exact_e50_train",
            1: "r2m_exact_e50_validation",
            2: "r2m_exact_e50_opened_seed2",
        }[seed]
        atoms = clone_target(
            source, target, config_type, f"exact_e50_seed{seed}", "E50_fixed_smearing"
        )
        atoms.info["r2m_split"] = split
        atoms.arrays["DFT_TOTAL_forces"] = dft_force[index]
        atoms.arrays["FOUNDATION_BASE_forces"] = base_force[index]
        atoms.arrays["FROZEN_Q6_forces"] = long_force[index]
        atoms.arrays["CURRENT_S0_forces_DIAGNOSTIC_NOT_DEPLOYED"] = current_s0_force[index]
        local_mode = mode[mapping[index]]
        atoms.arrays["APRIME_mode_real"] = local_mode.real
        atoms.arrays["APRIME_mode_imag"] = local_mode.imag
        atoms.info["APRIME_coordinate_real_A"] = float(coordinate[index].real)
        atoms.info["APRIME_coordinate_imag_A"] = float(coordinate[index].imag)
        e50_by_seed[seed].append(atoms)
        e50_targets[seed].append(target)

    harmonic_train_source = [item for item in r2c_train if role(item) == "harmonic_replay"]
    harmonic_valid_source = [item for item in r2c_valid if role(item) == "harmonic_replay"]
    support_source = [
        item for item in r2c_train if role(item) == "fixed_smearing_support_repair"
    ]
    thermal_unique = [
        item
        for item in r2c_train
        if role(item) == "thermal" and int(item.info.get("delta_repeat_index", -1)) == 0
    ]
    if (len(harmonic_train_source), len(harmonic_valid_source), len(support_source)) != (
        72,
        25,
        9,
    ):
        raise ValueError("unexpected harmonic/support counts")
    temperatures = Counter(float(item.info["lattice_temperature_K"]) for item in thermal_unique)
    if temperatures != Counter({300.0: 36, 450.0: 20, 600.0: 36}):
        raise ValueError(f"unexpected unique legacy thermal counts: {temperatures}")

    old_t450 = [item for item in thermal_unique if float(item.info["lattice_temperature_K"]) == 450.0]
    old_by_fp = {fingerprint(item): item for item in old_t450}
    if len(old_by_fp) != 20 or {fingerprint(item) for item in e50_by_seed[0]} != set(old_by_fp):
        raise ValueError("exact E50 seed0 does not replace the 20 legacy T450 geometries")
    replacement_differences = [
        np.asarray(exact.arrays["REF_forces"], float)
        - np.asarray(old_by_fp[fingerprint(exact)].arrays["REF_forces"], float)
        for exact in e50_by_seed[0]
    ]
    replacement_metrics = force_metrics(replacement_differences)
    if abs(replacement_metrics["RMSE_meV_A"] - 0.5991) > 0.002:
        raise ValueError(f"unexpected old-to-exact T450 label difference: {replacement_metrics}")

    auxiliary_source = [
        item for item in thermal_unique if float(item.info["lattice_temperature_K"]) in {300.0, 600.0}
    ]
    harmonic_train = [
        clone_target(
            item,
            np.asarray(item.arrays["REF_forces"], float),
            "r2m_harmonic_train",
            "harmonic_train",
            "R2C_original_short_delta_REF",
        )
        for item in harmonic_train_source
    ]
    harmonic_valid = [
        clone_target(
            item,
            np.asarray(item.arrays["REF_forces"], float),
            "r2m_harmonic_validation",
            "harmonic_validation",
            "R2C_original_short_delta_REF",
        )
        for item in harmonic_valid_source
    ]
    auxiliary = [
        clone_target(
            item,
            np.asarray(item.arrays["REF_forces"], float),
            f"r2m_auxiliary_T{int(float(item.info['lattice_temperature_K']))}",
            "auxiliary_legacy_thermal",
            "R2C_unique_repeat0_T300_T600",
        )
        for item in auxiliary_source
    ]

    groups = {
        "exact_e50": e50_targets[0],
        "harmonic": [np.asarray(item.arrays["REF_forces"], float) for item in harmonic_train],
        "auxiliary": [np.asarray(item.arrays["REF_forces"], float) for item in auxiliary],
    }
    scales = {name: force_metrics(values)["RMSE_meV_A"] / 1000.0 for name, values in groups.items()}
    for name, expected in EXPECTED_RMS_MEV_A.items():
        if abs(1000.0 * scales[name] - expected) > 0.002:
            raise ValueError(f"train-only {name} RMS changed: {1000.0 * scales[name]}")
    counts = {name: len(values) for name, values in groups.items()}
    raw_weight = {
        name: GROUP_MASS[name] / (counts[name] * scales[name] ** 2) for name in groups
    }
    maximum = max(raw_weight.values())
    normalized_weight = {name: raw_weight[name] / maximum for name in groups}
    for name, expected in EXPECTED_WEIGHT.items():
        if abs(normalized_weight[name] - expected) > 2.0e-6:
            raise ValueError(f"normalized {name} weight changed: {normalized_weight[name]}")
    config_weights = {
        "r2m_exact_e50_train": normalized_weight["exact_e50"],
        "r2m_exact_e50_validation": normalized_weight["exact_e50"],
        "r2m_harmonic_train": normalized_weight["harmonic"],
        "r2m_harmonic_validation": normalized_weight["harmonic"],
        "r2m_auxiliary_T300": normalized_weight["auxiliary"],
        "r2m_auxiliary_T600": normalized_weight["auxiliary"],
    }

    train = e50_by_seed[0] + harmonic_train + auxiliary
    valid = e50_by_seed[1] + harmonic_valid
    reserved = e50_by_seed[2]
    fingerprints = {
        "train": {fingerprint(item) for item in train},
        "valid": {fingerprint(item) for item in valid},
        "reserved_seed2": {fingerprint(item) for item in reserved},
        "support9": {fingerprint(item) for item in support_source},
    }
    if len(fingerprints["train"]) != len(train) or len(fingerprints["valid"]) != len(valid):
        raise ValueError("duplicate geometry survived the R2M collapse")
    for left, right in (
        ("train", "valid"),
        ("train", "reserved_seed2"),
        ("valid", "reserved_seed2"),
        ("train", "support9"),
        ("valid", "support9"),
    ):
        overlap = fingerprints[left] & fingerprints[right]
        if overlap:
            raise ValueError(f"forbidden geometry overlap {left}/{right}: {len(overlap)}")

    minimum_cell_length = min(
        float(np.linalg.norm(vector))
        for vector, periodic in zip(np.asarray(e50[0].cell), e50[0].pbc, strict=True)
        if periodic
    )
    diameter = 2.0 * 2.0 * 2
    if not diameter < minimum_cell_length:
        raise ValueError("fixed R2M core architecture can wrap the E50 supercell")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "train.xyz": train,
        "valid.xyz": valid,
        "reserved_e50_seed2.xyz": reserved,
    }
    for name, structures in outputs.items():
        atomic_extxyz(args.output_dir / name, structures)
    snapshot = args.output_dir / "prepare_script_snapshot.py"
    shutil.copy2(Path(__file__).resolve(), snapshot)
    output_hashes = {name: sha256(args.output_dir / name) for name in outputs}
    output_hashes[snapshot.name] = sha256(snapshot)

    manifest = {
        "status": "frozen_before_R2M_support_free_core_training",
        "model_role": "replacement_short_delta_core",
        "combination": "frozen_v11_foundation + new_R2M_core + frozen_q6",
        "depth3_or_current_S0_in_deployed_model": False,
        "new_DFT_labels": 0,
        "long_range_model_modified": False,
        "energy_training_enabled": False,
        "temperature_or_smearing_is_model_input": False,
        "target_definition": {
            "exact_E50_force": "DFT_target_forces - foundation_base_forces - frozen_q6_forces",
            "harmonic_force": "original R2C REF_forces (harmonic total minus frozen foundation; LR=0)",
            "auxiliary_force": "original R2C REF_forces for unique repeat0 T300/T600",
            "energy": "REF_energy=0 and config_energy_weight=0 for every configuration",
            "current_S0_force": "preserved on E50 as diagnostic only; excluded from target and deployment",
        },
        "fixed_smearing_scope": {
            "exact_anchor": {"lattice_temperature_K": 450.0, "degauss_Ry": FIXED_DEGAUSS_RY},
            "train_trajectory_seed": 0,
            "validation_trajectory_seed": 1,
            "opened_development_seed_not_for_training_or_selection": 2,
            "legacy_auxiliary": "unique T300 and T600 repeat0 targets; group mass 0.2",
            "legacy_T450_policy": "20 old labels removed and replaced by exact E50 seed0 labels",
            "old_T450_minus_exact_E50_seed0": replacement_metrics,
        },
        "counts": {
            "train": len(train),
            "train_exact_e50": 20,
            "train_harmonic": 72,
            "train_auxiliary_T300": 36,
            "train_auxiliary_T600": 36,
            "valid": len(valid),
            "valid_exact_e50_seed1": 20,
            "valid_harmonic": 25,
            "reserved_e50_seed2": 20,
            "support_used": 0,
        },
        "loss_weighting": {
            "formula": "raw_w_g = lambda_g / (N_g * train_only_force_RMS_g_eV_A^2); divide all raw_w by max(raw_w)",
            "batch_size": 1,
            "group_mass": GROUP_MASS,
            "train_only_force_RMS_eV_A": scales,
            "raw_config_weight": raw_weight,
            "normalized_config_type_weights": config_weights,
            "scaler_inputs": "train exact seed0, harmonic-train, and auxiliary T300/T600 only",
            "seed1_seed2_support_used_for_scales": False,
        },
        "architecture_freeze": {
            "model": "MACE",
            "random_initialization": True,
            "r_max_A": 2.0,
            "num_interactions": 2,
            "hidden_irreps": "16x0e+16x1o+16x2e",
            "max_ell": 2,
            "num_radial_basis": 24,
            "num_cutoff_basis": 5,
            "correlation": 3,
            "interaction_diameter_bound_A": diameter,
            "minimum_E50_periodic_cell_length_A": minimum_cell_length,
            "new_core_no_periodic_wrap_by_bound": True,
            "whole_composite_no_periodic_wrap_claimed": False,
            "foundation_note": "frozen v11 has a larger theoretical diameter and needs a separate 6x6/8x8 composite size-consistency gate",
        },
        "leakage_control": {
            "split_unit": "whole configuration; E50 split solely by trajectory_seed",
            "atom_or_bond_random_split": False,
            "E50_extxyz_split_metadata_used": False,
            "support_labels_or_geometries_in_train_or_valid": False,
            "seed1_in_gradients_or_scales": False,
            "seed2_in_gradients_scales_or_checkpoint_selection": False,
            "legacy_repeat_duplicates_collapsed": True,
            "exact_geometry_overlap_counts": {
                "old_T450_vs_E50_seed0": 20,
                "legacy_unique92_vs_E50_seed1": 0,
                "legacy_unique92_vs_E50_seed2": 0,
            },
            "opened_development_limitation": "seed2 was used in prior descriptor development and is not blind/external",
        },
        "inputs": {
            name: {"path": str(paths[name]), "sha256": observed_hashes[name]}
            for name in paths
        },
        "outputs": {
            name: {"path": str(args.output_dir / name), "sha256": output_hashes[name]}
            for name in output_hashes
        },
    }
    strict_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "weights": config_weights}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
