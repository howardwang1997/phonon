#!/usr/bin/env python3
"""Freeze nine leakage-safe outer-LOCO folds for the R2M routed tail.

The input core must already have passed the support-free R2M checkpoint gate.
This program is deliberately unable to consume E50 seed2.  It materializes
the residual short-range target for each support configuration, then creates
nine directories containing eight training configurations and one held
configuration.  No model is trained here.
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
import torch
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

from graphene_r2m_aprime_eval import APRIME_MODE_DEFINITION
from graphene_r2m_routed_tail import (
    core_invariant_schema,
    sha256,
    strict_json,
    torch_load,
)


EXPECTED_SUPPORT9_SHA256 = (
    "505cf1a22ec05f83b1cdc610809e5930652536a9b295bf45c4c1221fa4183a88"
)
EXPECTED_PRISTINE_SHA256 = (
    "4e845685b56561e7204cabeb32e9e5103089ee52e253a383e08778efdbf46e91"
)
EXPECTED_R2M_MANIFEST_SHA256 = (
    "bb20a86af073e344c38e39ab14d3a80af9e39fa051591bed2b039ae643ffe31e"
)
EXPECTED_R2M_TRAIN_SHA256 = (
    "789b65e1a2c3b260e24d80f008407c7ee55e02676ff3bcf9a875fe9de294519c"
)
EXPECTED_R2M_VALID_SHA256 = (
    "92ad1508e1067265445b79679cfe5142d23b2507dab2aa44397578cee8c7a6da"
)


def geometry_fingerprint(structure: Atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(structure.numbers, np.dtype("<i8")).tobytes())
    digest.update(np.round(np.asarray(structure.cell, float), 10).astype("<f8").tobytes())
    digest.update(
        np.round(np.asarray(structure.positions, float), 10).astype("<f8").tobytes()
    )
    return digest.hexdigest()


def atomic_extxyz(path: Path, structures: list[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, path)


def assert_no_seed2_path(path: Path) -> None:
    lowered = str(path).lower()
    forbidden = ("seed2", "reserved_e50", "reserved-e50")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"R2M routed-tail preparation rejects seed2 input: {path}")


def load_passing_core(
    core_path: Path, expected_hash: str, gate_path: Path
) -> tuple[torch.nn.Module, dict]:
    if len(expected_hash) != 64:
        raise ValueError("--core-sha256 must be a full SHA-256")
    int(expected_hash, 16)
    observed_hash = sha256(core_path)
    if observed_hash != expected_hash:
        raise ValueError("selected core SHA-256 mismatch")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "R2M_core_checkpoint_gate_passed"
        or gate.get("force_gate_allows_routed_tail_stage") is not True
        or gate.get("full_composite_deployment_authorized") is not False
    ):
        raise ValueError("core gate does not authorize routed-tail development")
    artifact = gate.get("checkpoint_artifact", {})
    if artifact.get("kind") != "selected_core_for_routed_tail_force_gate":
        raise ValueError("core gate did not publish the selected-core artifact")
    if artifact.get("sha256") != expected_hash:
        raise ValueError("core gate and --core-sha256 disagree")
    if gate.get("seed2_or_support_read_for_selection") is not False:
        raise ValueError("core gate does not prove support/seed2-free selection")
    published_path = Path(str(artifact.get("path", "")))
    if not published_path.is_file() or published_path.resolve() != core_path.resolve():
        raise ValueError("--core-model is not the exact artifact published by core gate")
    core = torch_load(core_path, map_location="cpu")
    core_invariant_schema(core)
    return core, gate


def validate_sources(
    support_path: Path, r2m_data: Path, pristine_path: Path
) -> tuple[list[Atoms], list[Atoms], list[Atoms], dict]:
    for path in (support_path, r2m_data / "train.xyz", r2m_data / "valid.xyz", pristine_path):
        assert_no_seed2_path(path)
    if sha256(support_path) != EXPECTED_SUPPORT9_SHA256:
        raise ValueError("support9 source changed")
    if sha256(pristine_path) != EXPECTED_PRISTINE_SHA256:
        raise ValueError("pristine graphene source changed")
    manifest_path = r2m_data / "manifest.json"
    train_path = r2m_data / "train.xyz"
    valid_path = r2m_data / "valid.xyz"
    if sha256(manifest_path) != EXPECTED_R2M_MANIFEST_SHA256:
        raise ValueError("support-free R2M source manifest changed")
    if sha256(train_path) != EXPECTED_R2M_TRAIN_SHA256:
        raise ValueError("support-free R2M train.xyz changed")
    if sha256(valid_path) != EXPECTED_R2M_VALID_SHA256:
        raise ValueError("support-free R2M valid.xyz changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_R2M_support_free_core_training":
        raise ValueError("wrong support-free R2M data manifest")
    if manifest.get("counts", {}).get("support_used") != 0:
        raise ValueError("R2M core data unexpectedly contains support")
    if manifest.get("energy_training_enabled") is not False:
        raise ValueError("support-free core manifest changed its energy policy")
    if (
        manifest.get("leakage_control", {}).get(
            "seed2_in_gradients_scales_or_checkpoint_selection"
        )
        is not False
    ):
        raise ValueError("support-free core manifest has an unsafe seed2 policy")
    for name in ("train.xyz", "valid.xyz"):
        expected = manifest["outputs"][name]["sha256"]
        if sha256(r2m_data / name) != expected:
            raise ValueError(f"support-free R2M {name} changed")
    support = read(support_path, index=":")
    train = read(train_path, index=":")
    valid = read(valid_path, index=":")
    if len(support) != 9 or len(train) != 164 or len(valid) != 45:
        raise ValueError("unexpected support/replay counts")
    if len({int(item.info["sscha_index"]) for item in support}) != 9:
        raise ValueError("support sscha_index values are not unique")
    if any(
        str(item.info.get("delta_target_role")) != "fixed_smearing_support_repair"
        for item in support
    ):
        raise ValueError("support9 contains a non-support role")
    train_roles = Counter(str(item.info.get("r2m_target_role", "")) for item in train)
    if train_roles != Counter(
        {
            "harmonic_train": 72,
            "exact_e50_seed0": 20,
            "auxiliary_legacy_thermal": 72,
        }
    ):
        raise ValueError("support-free R2M train roles/counts changed")
    valid_roles = Counter(str(item.info.get("r2m_target_role", "")) for item in valid)
    if valid_roles != Counter({"exact_e50_seed1": 20, "harmonic_validation": 25}):
        raise ValueError("support-free R2M valid roles/counts changed")
    if any(
        int(item.info.get("trajectory_seed", -1)) != 0
        for item in train
        if item.info.get("r2m_target_role") == "exact_e50_seed0"
    ):
        raise ValueError("non-seed0 E50 record entered support-free training data")
    if any(
        int(item.info.get("trajectory_seed", -1)) != 1
        for item in valid
        if item.info.get("r2m_target_role") == "exact_e50_seed1"
    ):
        raise ValueError("non-seed1 E50 record entered support-free validation data")
    if any("seed2" in str(item.info).lower() for item in train + valid):
        raise ValueError("seed2 metadata entered support-free train/validation data")
    support_fp = {geometry_fingerprint(item) for item in support}
    replay_fp = {geometry_fingerprint(item) for item in train + valid}
    if support_fp & replay_fp:
        raise ValueError("support geometry appears in support-free replay/validation")
    return support, train, valid, manifest


def core_predictions(
    structures: list[Atoms], core_path: Path, device: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    calculator = MACECalculator(
        model_paths=str(core_path), device=device, default_dtype="float32"
    )
    energies = []
    forces = []
    for source in structures:
        atoms = source.copy()
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), float))
    del calculator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(energies), forces


def residual_support_records(
    support: list[Atoms], core_energy: np.ndarray, core_force: list[np.ndarray]
) -> list[Atoms]:
    result = []
    energy_gauge_offsets = []
    for source, energy, force in zip(support, core_energy, core_force, strict=True):
        required_arrays = (
            "PARENT_SHORT_RANGE_TARGET_forces",
            "TOTAL_forces",
            "BASE_forces",
            "LONG_RANGE_forces",
        )
        if any(name not in source.arrays for name in required_arrays):
            raise ValueError("support record lacks the short-range force decomposition")
        short_target = np.asarray(source.arrays["PARENT_SHORT_RANGE_TARGET_forces"], float)
        raw_short = (
            np.asarray(source.arrays["TOTAL_forces"], float)
            - np.asarray(source.arrays["BASE_forces"], float)
            - np.asarray(source.arrays["LONG_RANGE_forces"], float)
        )
        if float(np.max(np.abs(short_target - raw_short))) > 2.0e-8:
            raise ValueError("support short-range force decomposition changed")
        for key in (
            "PARENT_SHORT_RANGE_TARGET_energy",
            "TOTAL_energy",
            "BASE_energy",
            "LONG_RANGE_energy",
        ):
            if key not in source.info:
                raise ValueError(f"support record lacks {key}")
        parent_energy = float(source.info["PARENT_SHORT_RANGE_TARGET_energy"])
        raw_energy = (
            float(source.info["TOTAL_energy"])
            - float(source.info["BASE_energy"])
            - float(source.info["LONG_RANGE_energy"])
        )
        energy_gauge_offsets.append(parent_energy - raw_energy)
        atoms = source.copy()
        atoms.calc = None
        tail_force = short_target - np.asarray(force, float)
        tail_energy = parent_energy - float(energy)
        atoms.arrays["R2M_CORE_forces"] = np.asarray(force, float)
        atoms.arrays["R2M_SHORT_TARGET_forces"] = short_target
        atoms.arrays["R2M_TAIL_TARGET_forces"] = tail_force
        atoms.arrays["REF_forces"] = tail_force
        atoms.info.update(
            {
                "R2M_CORE_energy": float(energy),
                "R2M_SHORT_TARGET_energy": parent_energy,
                "R2M_TAIL_RAW_TARGET_energy": tail_energy,
                "REF_energy": tail_energy,
                "r2m_routed_tail_role": "support_outer_loco",
                "r2m_core_and_tail_only_short_range": True,
                "foundation_and_q6_outside_tail": True,
            }
        )
        result.append(atoms)
    if float(np.ptp(np.asarray(energy_gauge_offsets))) > 1.0e-10:
        raise ValueError("support energy gauge is not a configuration-independent offset")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-model", type=Path, required=True)
    parser.add_argument("--core-sha256", required=True)
    parser.add_argument("--core-gate-summary", type=Path, required=True)
    parser.add_argument("--support9", type=Path, required=True)
    parser.add_argument("--r2m-data", type=Path, required=True)
    parser.add_argument("--pristine", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--thermal-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for path in (
        args.core_model,
        args.core_gate_summary,
        args.support9,
        args.r2m_data,
        args.pristine,
        args.operator,
        args.background,
        args.thermal_result,
        args.output_dir,
    ):
        assert_no_seed2_path(path)
    for path in (args.operator, args.background, args.thermal_result):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing frozen evaluation asset: {path}")
    with np.load(args.operator, allow_pickle=False) as data:
        operator_degauss = float(np.asarray(data["degauss_Ry"]).reshape(()))
    with np.load(args.thermal_result, allow_pickle=False) as data:
        lattice_temperature = int(
            np.asarray(data["lattice_temperature_K"]).reshape(())
        )
        operator_temperature = int(
            np.asarray(data["operator_temperature_K"]).reshape(())
        )
    if abs(operator_degauss - 0.0019000869380739254) > 5.0e-10:
        raise ValueError("routed-tail evaluation operator has the wrong degauss")
    if (lattice_temperature, operator_temperature) != (450, 300):
        raise ValueError("routed-tail A-prime mode assets have the wrong temperatures")
    core, core_gate = load_passing_core(
        args.core_model, args.core_sha256, args.core_gate_summary
    )
    feature_schema = core_invariant_schema(core)
    del core
    support, replay_train, replay_valid, source_manifest = validate_sources(
        args.support9, args.r2m_data, args.pristine
    )
    support = sorted(support, key=lambda item: int(item.info["sscha_index"]))
    print("evaluating the frozen passing core on support9", flush=True)
    energies, forces = core_predictions(support, args.core_model, args.device)
    residual = residual_support_records(support, energies, forces)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = args.output_dir / "code_snapshots"
    snapshots.mkdir()
    script_path = Path(__file__).resolve()
    module_path = script_path.with_name("graphene_r2m_routed_tail.py")
    aprime_helper_path = script_path.with_name("graphene_r2m_aprime_eval.py")
    shutil.copy2(script_path, snapshots / script_path.name)
    shutil.copy2(module_path, snapshots / module_path.name)
    shutil.copy2(aprime_helper_path, snapshots / aprime_helper_path.name)

    all_support_fingerprints = [geometry_fingerprint(item) for item in residual]
    folds = []
    for outer_index, held in enumerate(residual):
        training = [item.copy() for index, item in enumerate(residual) if index != outer_index]
        held_copy = held.copy()
        energy_gauge = float(
            np.mean([float(item.info["R2M_TAIL_RAW_TARGET_energy"]) for item in training])
        )
        for item in training + [held_copy]:
            item.info["R2M_TAIL_ENERGY_GAUGE_eV"] = energy_gauge
            item.info["R2M_TAIL_TARGET_energy_centered"] = (
                float(item.info["R2M_TAIL_RAW_TARGET_energy"]) - energy_gauge
            )
            item.info["REF_energy"] = float(
                item.info["R2M_TAIL_TARGET_energy_centered"]
            )
        sscha_index = int(held.info["sscha_index"])
        fold_dir = args.output_dir / f"outer{outer_index:02d}_sscha{sscha_index}"
        fold_dir.mkdir()
        train_path = fold_dir / "support_train8.xyz"
        held_path = fold_dir / "support_held1.xyz"
        atomic_extxyz(train_path, training)
        atomic_extxyz(held_path, [held_copy])
        training_fingerprints = [geometry_fingerprint(item) for item in training]
        held_fingerprint = geometry_fingerprint(held_copy)
        if held_fingerprint in training_fingerprints:
            raise RuntimeError("outer-held support geometry leaked into train8")
        fold_manifest = {
            "status": "R2M_routed_tail_outer_fold_frozen_before_training",
            "outer_fold": outer_index,
            "held_sscha_index": sscha_index,
            "selection_policy": "fixed_epoch_240_no_support_checkpoint_selection",
            "strict_nested_inner_CV_run": False,
            "support9_status": (
                "opened development data from prior R2C-R2K work; this outer fold "
                "tests within-stage encoder transfer and is not blind/external"
            ),
            "energy_gauge": {
                "definition": "mean train8 (parent short target energy - frozen core energy)",
                "value_eV": energy_gauge,
                "held_energy_used": False,
            },
            "leakage": {
                "split_unit": "whole support configuration",
                "held_geometry_or_label_in_scaler_gradient_or_selection": False,
                "tail_and_router_initialized_from_support_exposed_model": False,
                "core_is_support_free": True,
                "E50_seed2_read": False,
                "train_support_fingerprints": training_fingerprints,
                "held_support_fingerprint": held_fingerprint,
            },
            "inputs": {
                "core": {"path": str(args.core_model), "sha256": args.core_sha256},
                "core_gate": {
                    "path": str(args.core_gate_summary),
                    "sha256": sha256(args.core_gate_summary),
                },
                "support9": {"path": str(args.support9), "sha256": sha256(args.support9)},
                "replay_train": {
                    "path": str(args.r2m_data / "train.xyz"),
                    "sha256": sha256(args.r2m_data / "train.xyz"),
                },
                "replay_valid_post_freeze_only": {
                    "path": str(args.r2m_data / "valid.xyz"),
                    "sha256": sha256(args.r2m_data / "valid.xyz"),
                },
                "pristine": {"path": str(args.pristine), "sha256": sha256(args.pristine)},
                "operator_post_freeze_only": {
                    "path": str(args.operator),
                    "sha256": sha256(args.operator),
                },
                "background_post_freeze_only": {
                    "path": str(args.background),
                    "sha256": sha256(args.background),
                },
                "thermal_result_post_freeze_only": {
                    "path": str(args.thermal_result),
                    "sha256": sha256(args.thermal_result),
                },
                "aprime_evaluation_helper": {
                    "path": str(aprime_helper_path),
                    "sha256": sha256(aprime_helper_path),
                },
            },
            "outputs": {
                "support_train8.xyz": sha256(train_path),
                "support_held1.xyz": sha256(held_path),
            },
            "feature_schema": feature_schema,
        }
        strict_json(fold_dir / "fold_manifest.json", fold_manifest)
        folds.append(
            {
                "outer_fold": outer_index,
                "held_sscha_index": sscha_index,
                "directory": str(fold_dir),
                "manifest_sha256": sha256(fold_dir / "fold_manifest.json"),
            }
        )

    top_manifest = {
        "status": "R2M_routed_tail_nine_outer_folds_frozen_before_training",
        "model_formula": "foundation(frozen) + core(frozen) + sum_i g_i*(epsilon_i-c_C) + q6(frozen)",
        "foundation_and_q6_in_tail_training": False,
        "new_DFT_labels": 0,
        "n_outer_folds": 9,
        "selection_policy": "fixed_epoch_240_no_support_checkpoint_selection",
        "support9_status": (
            "opened development data; outer LOCO is not an external or blind test"
        ),
        "E50_seed2_read_or_accepted_as_input": False,
        "support_order_sscha_index": [int(item.info["sscha_index"]) for item in residual],
        "support_geometry_fingerprints": all_support_fingerprints,
        "feature_schema": feature_schema,
        "core_gate_status": core_gate["status"],
        "source_r2m_manifest_sha256": sha256(args.r2m_data / "manifest.json"),
        "source_r2m_counts": source_manifest["counts"],
        "replay_counts": {"train": len(replay_train), "valid_post_freeze": len(replay_valid)},
        "frozen_post_training_evaluation_assets": {
            "operator": {
                "path": str(args.operator),
                "sha256": sha256(args.operator),
                "degauss_Ry": operator_degauss,
            },
            "background": {
                "path": str(args.background),
                "sha256": sha256(args.background),
            },
            "thermal_result": {
                "path": str(args.thermal_result),
                "sha256": sha256(args.thermal_result),
                "lattice_temperature_K": lattice_temperature,
                "operator_temperature_K": operator_temperature,
            },
            "aprime_mode_definition": APRIME_MODE_DEFINITION,
            "aprime_evaluation_helper": {
                "path": str(aprime_helper_path),
                "sha256": sha256(aprime_helper_path),
            },
        },
        "folds": folds,
        "snapshots": {
            script_path.name: sha256(snapshots / script_path.name),
            module_path.name: sha256(snapshots / module_path.name),
            aprime_helper_path.name: sha256(snapshots / aprime_helper_path.name),
        },
    }
    strict_json(args.output_dir / "manifest.json", top_manifest)
    print(json.dumps({"status": top_manifest["status"], "folds": folds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
