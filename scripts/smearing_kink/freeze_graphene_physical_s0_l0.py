#!/usr/bin/env python3
"""Freeze the passing S0 predictor and the complete L0 sampling protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


TEMPERATURES = (300, 450, 600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def verify_selected_model(model_path: Path, checkpoint_path: Path) -> None:
    model = torch_load(model_path)
    checkpoint = torch_load(checkpoint_path)
    if "model" not in checkpoint:
        raise ValueError("selected checkpoint has no model state")
    standalone = model.state_dict()
    stored = checkpoint["model"]
    if standalone.keys() != stored.keys():
        raise ValueError("standalone model and selected checkpoint have different state keys")
    for key in standalone:
        left = standalone[key].detach().cpu()
        right = stored[key].detach().cpu()
        if left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"standalone model differs from checkpoint at {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-sweep", type=Path, required=True)
    parser.add_argument("--selected-model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--td-script", type=Path, required=True)
    parser.add_argument("--recompute-script", type=Path, required=True)
    parser.add_argument("--summary-script", type=Path, required=True)
    parser.add_argument("--bootstrap-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.data_manifest.read_text())
    sweep = json.loads(args.checkpoint_sweep.read_text())
    if data.get("status") != "passed" or not all(
        data.get("aggregate_gates", {}).get("checks", {}).values()
    ):
        raise ValueError("S0 data gate is not fully passed")
    if data.get("development_temperatures_K") != list(TEMPERATURES):
        raise ValueError("unexpected S0 development temperatures")
    if data.get("locked_validation_temperatures_K_not_accessed") != [375, 525]:
        raise ValueError("locked validation boundary is missing")
    if data.get("temperature_or_degauss_is_model_input") is not False:
        raise ValueError("S0 local model unexpectedly consumes temperature/degauss")
    if sweep.get("status") != "checkpoint_passed":
        raise ValueError("S0 checkpoint force gate did not pass")
    selected = sweep.get("selected", {})
    if selected.get("passes_fixed_gate") is not True:
        raise ValueError("selected S0 checkpoint does not pass every fixed force gate")
    thresholds = sweep.get("fixed_thresholds", {})
    if thresholds.get("thermal_force_RMSE_meV_A") != 50.0:
        raise ValueError("unexpected thermal RMSE threshold")
    if thresholds.get("thermal_force_max_abs_meV_A") != 250.0:
        raise ValueError("unexpected thermal maximum-force threshold")
    if thresholds.get("harmonic_RMSE_relative_to_frozen_v11") != 2.0:
        raise ValueError("unexpected harmonic replay threshold")
    checkpoint_path = Path(selected["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256(checkpoint_path) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash changed after evaluation")
    if sha256(args.base_model) != data["base_model_sha256"]:
        raise ValueError("frozen base-model hash differs from S0 data generation")
    verify_selected_model(args.selected_model, checkpoint_path)

    operators = {}
    for temperature in TEMPERATURES:
        record = data["operators"][str(temperature)]
        path = Path(record["output"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"T{temperature} operator hash changed")
        if record.get("degauss_formula") != "k_B*T/Ry":
            raise ValueError(f"T{temperature} operator does not use k_B*T/Ry")
        with np.load(path, allow_pickle=False) as payload:
            mapping = np.asarray(payload["atom_mapping"], int)
            force_constants = np.asarray(payload["delta_fc_full"], float)
        if not np.array_equal(mapping, np.arange(72)):
            raise ValueError(f"T{temperature} L0 currently requires identity atom mapping")
        if force_constants.shape != (72, 72, 3, 3):
            raise ValueError(f"T{temperature} operator shape mismatch")
        operators[str(temperature)] = {
            "path": str(path),
            "sha256": record["sha256"],
            "degauss_Ry": record["degauss_Ry"],
            "degauss_formula": "k_B*T/Ry",
        }

    scripts = {}
    for label, path in (
        ("sampling", args.td_script),
        ("short_tdep_recompute", args.recompute_script),
        ("point_gate", args.summary_script),
        ("bootstrap_gate", args.bootstrap_script),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        scripts[label] = {"path": str(path), "sha256": sha256(path)}

    payload = {
        "status": "frozen_for_L0_development",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "S0-passing predictor and fixed 300/450/600 K L0 classical-TDEP protocol",
        "development_temperatures_K": list(TEMPERATURES),
        "execution_order_K": [450, 300, 600],
        "locked_validation_temperatures_K_not_accessed": [375, 525],
        "force_formula": "F_total = F_v11 + F_delta + F_electronic(T)",
        "electronic_smearing_formula": "smearing/degauss (Ry) = k_B*T/Ry",
        "local_Mermin_term": "zero_by_E1_gate",
        "temperature_or_degauss_is_local_model_input": False,
        "base_model": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
        "delta_model": {
            "path": str(args.selected_model),
            "sha256": sha256(args.selected_model),
            "selected_checkpoint": str(checkpoint_path),
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "selected_epoch": int(selected["epoch"]),
        },
        "background": {"path": str(args.background), "sha256": sha256(args.background)},
        "operators": operators,
        "source_gates": {
            "data_manifest": {
                "path": str(args.data_manifest),
                "sha256": sha256(args.data_manifest),
            },
            "checkpoint_sweep": {
                "path": str(args.checkpoint_sweep),
                "sha256": sha256(args.checkpoint_sweep),
            },
            "selected_force_metrics": selected,
        },
        "trajectory_protocol": {
            "seeds": [0, 1, 2],
            "pilot_snapshots_per_seed": 120,
            "final_snapshots_per_seed": 3000,
            "saved_frame_spacing_fs": 40.0,
            "seed_settings": {
                "0": {"dt_fs": 0.5, "equilibration_steps": 3000, "stride": 80},
                "1": {"dt_fs": 0.25, "equilibration_steps": 6000, "stride": 160},
                "2": {"dt_fs": 0.5, "equilibration_steps": 3000, "stride": 80},
            },
            "checkpoint_every_saved_snapshots": 25,
            "pilot_stability_mean_temperature_relative_error_max": 0.10,
            "minimum_pair_distance_A": 0.8,
            "maximum_force_eV_A": 100.0,
            "maximum_temperature_factor": 5.0,
        },
        "fixed_acceptance_thresholds": {
            "seed_pair_full_band_MAE_cm-1": 5.0,
            "MD_mean_temperature_relative_error": 0.05,
            "bootstrap_replicates": 1000,
            "bootstrap_block_rule": "max(5, ceil(2*tau_int))",
            "bootstrap_confidence_rule": "one-sided empirical 95th percentile, method=higher",
        },
        "failure_policy": {
            "pilot_instability": "stop before 3000-snapshot extension",
            "point_or_bootstrap_gate_failure_at_3000": "stop; do not append more snapshots automatically",
            "validation_release": "forbidden; 375/525 K remain locked",
        },
        "scripts": scripts,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
