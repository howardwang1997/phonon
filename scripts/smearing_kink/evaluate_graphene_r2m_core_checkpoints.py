#!/usr/bin/env python3
"""Select the R2M support-free core using only seed1 and harmonic validation.

The candidate list and thresholds are fixed.  E50 seed2 and the nine support
configurations are intentionally not accepted as inputs to this program.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


FIXED_EPOCHS = (40, 80, 120, 160, 200, 235)
LIMITS = {
    "e50_force_RMSE_meV_A": 30.0,
    "e50_force_max_abs_meV_A": 200.0,
    "e50_Aprime_RMS_meV_A": 15.0,
    "e50_Aprime_slope_relative_error_abs": 0.05,
    "harmonic_force_RMSE_meV_A": 9.044673057081592,
    "harmonic_force_max_abs_meV_A": 200.0,
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


def torch_load(path: Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def exact_model_state_equal(left, right) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(
        left_state[name].shape == right_state[name].shape
        and left_state[name].dtype == right_state[name].dtype
        and torch.equal(left_state[name].cpu(), right_state[name].cpu())
        for name in left_state
    )


def force_metrics(values: list[np.ndarray]) -> dict:
    joined = np.concatenate([np.asarray(value, float).reshape(-1) for value in values])
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(joined**2))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(joined))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(joined))),
        "n_force_components": int(joined.size),
    }


def checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"_epoch-(\d+)(?:_swa)?\.pt$", path.name)
    return int(match.group(1)) if match else None


def fixed_checkpoint_paths(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    all_paths = list(directory.glob("*_epoch-*.pt"))
    for epoch in FIXED_EPOCHS:
        matches = [
            path
            for path in all_paths
            if checkpoint_epoch(path) == epoch and "_swa.pt" not in path.name
        ]
        if len(matches) != 1:
            raise ValueError(f"epoch {epoch} has {len(matches)} non-SWA checkpoints")
        result[f"epoch{epoch}"] = matches[0]
    return result


def predictions(structures: list, model, device: str) -> list[np.ndarray]:
    calculator = MACECalculator(models=model, device=device, default_dtype="float32")
    forces = []
    for source in structures:
        atoms = source.copy()
        atoms.calc = calculator
        forces.append(np.asarray(atoms.get_forces(), float))
    del calculator
    return forces


def restoring_slope(coordinates: np.ndarray, projected_forces: np.ndarray) -> float:
    denominator = float(np.vdot(coordinates, coordinates).real)
    if denominator <= 0.0:
        raise ValueError("all A-prime coordinates are zero")
    return -float(np.vdot(coordinates, projected_forces).real / denominator)


def evaluate_model(model, e50: list, harmonic: list, device: str) -> dict:
    structures = e50 + harmonic
    predicted = predictions(structures, model, device)
    e50_predicted = predicted[: len(e50)]
    harmonic_predicted = predicted[len(e50) :]
    e50_errors = [
        value - np.asarray(source.arrays["REF_forces"], float)
        for source, value in zip(e50, e50_predicted, strict=True)
    ]
    harmonic_errors = [
        value - np.asarray(source.arrays["REF_forces"], float)
        for source, value in zip(harmonic, harmonic_predicted, strict=True)
    ]

    coordinates = np.asarray(
        [
            complex(
                float(item.info["APRIME_coordinate_real_A"]),
                float(item.info["APRIME_coordinate_imag_A"]),
            )
            for item in e50
        ]
    )
    predicted_modes = []
    target_modes = []
    error_modes = []
    for source, core_force, error in zip(e50, e50_predicted, e50_errors, strict=True):
        mode = np.asarray(source.arrays["APRIME_mode_real"], float) + 1.0j * np.asarray(
            source.arrays["APRIME_mode_imag"], float
        )
        baseline = np.asarray(source.arrays["FOUNDATION_BASE_forces"], float) + np.asarray(
            source.arrays["FROZEN_Q6_forces"], float
        )
        target = np.asarray(source.arrays["DFT_TOTAL_forces"], float)
        predicted_modes.append(np.vdot(mode.reshape(-1), (baseline + core_force).reshape(-1)))
        target_modes.append(np.vdot(mode.reshape(-1), target.reshape(-1)))
        error_modes.append(np.vdot(mode.reshape(-1), error.reshape(-1)))
    predicted_modes = np.asarray(predicted_modes)
    target_modes = np.asarray(target_modes)
    error_modes = np.asarray(error_modes)
    predicted_slope = restoring_slope(coordinates, predicted_modes)
    target_slope = restoring_slope(coordinates, target_modes)
    if abs(target_slope) <= 1.0e-14:
        raise ValueError("DFT A-prime slope is zero")
    relative_slope_error = (predicted_slope - target_slope) / target_slope

    e50_force = force_metrics(e50_errors)
    harmonic_force = force_metrics(harmonic_errors)
    observed = {
        "e50_force_RMSE_meV_A": e50_force["RMSE_meV_A"],
        "e50_force_max_abs_meV_A": e50_force["max_abs_meV_A"],
        "e50_Aprime_RMS_meV_A": float(1000.0 * np.sqrt(np.mean(np.abs(error_modes) ** 2))),
        "e50_Aprime_slope_relative_error_abs": float(abs(relative_slope_error)),
        "harmonic_force_RMSE_meV_A": harmonic_force["RMSE_meV_A"],
        "harmonic_force_max_abs_meV_A": harmonic_force["max_abs_meV_A"],
    }
    ratios = {name: observed[name] / limit for name, limit in LIMITS.items()}
    return {
        "passes_fixed_gate": bool(all(value <= 1.0 for value in ratios.values())),
        "normalized_gate_ratios": ratios,
        "normalized_worst_gate_score": float(max(ratios.values())),
        "E50_seed1": {
            "force_error": e50_force,
            "Aprime_force_error_RMS_meV_A": observed["e50_Aprime_RMS_meV_A"],
            "predicted_restoring_slope_eV_A2": predicted_slope,
            "DFT_restoring_slope_eV_A2": target_slope,
            "relative_slope_error": float(relative_slope_error),
        },
        "harmonic_validation": {
            "force_error": harmonic_force,
            "per_configuration_force": [force_metrics([value]) for value in harmonic_errors],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--template-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-model", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_R2M_support_free_core_training":
        raise ValueError("wrong R2M data status")
    if manifest.get("energy_training_enabled") is not False:
        raise ValueError("the first R2M core gate must remain forces-only")
    if manifest["leakage_control"]["seed2_in_gradients_scales_or_checkpoint_selection"] is not False:
        raise ValueError("seed2 leakage flag is not false")
    valid_path = args.data / "valid.xyz"
    if sha256(valid_path) != manifest["outputs"]["valid.xyz"]["sha256"]:
        raise ValueError("validation data hash changed")
    freeze_path = args.training_dir / "training_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("status") != "frozen_before_R2M_core_training"
        or freeze.get("random_initialization") is not True
        or freeze.get("depth3_or_current_S0_initialized_or_deployed") is not False
        or freeze["training"]["fixed_evaluation_epochs"] != list(FIXED_EPOCHS)
        or freeze["training"]["validation_loss_cannot_stop_training"] is not True
        or freeze["optimizer"].get("scheduler") != "ExponentialLR"
        or float(freeze["optimizer"].get("lr_scheduler_gamma", -1.0)) != 1.0
        or freeze["optimizer"].get("validation_data_controls_lr") is not False
    ):
        raise ValueError("training freeze does not match the R2M core contract")
    if freeze["data_manifest"]["sha256"] != sha256(manifest_path):
        raise ValueError("training used another data manifest")
    if freeze["evaluator"]["sha256"] != sha256(Path(__file__).resolve()):
        raise ValueError("executed evaluator differs from the pre-training snapshot")
    runtime_path = args.training_dir / "training_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("status") != "training_complete_pending_fixed_checkpoint_gate":
        raise ValueError("training runtime is not ready for the checkpoint gate")
    recorded_template = Path(runtime["final_model"])
    if recorded_template.resolve() != args.template_model.resolve():
        raise ValueError("template model path differs from the completed training model")
    if runtime["final_model_sha256"] != sha256(args.template_model):
        raise ValueError("template model hash differs from training_runtime.json")

    valid = read(valid_path, index=":")
    e50 = [item for item in valid if str(item.info["r2m_target_role"]) == "exact_e50_seed1"]
    harmonic = [
        item for item in valid if str(item.info["r2m_target_role"]) == "harmonic_validation"
    ]
    if len(valid) != 45 or len(e50) != 20 or len(harmonic) != 25:
        raise ValueError("validation must be exactly seed1=20 plus harmonic=25")
    if any(int(item.info["trajectory_seed"]) != 1 for item in e50):
        raise ValueError("non-seed1 E50 record entered checkpoint selection")
    if any("support" in str(item.info.get("r2m_target_role", "")) for item in valid):
        raise ValueError("support record entered checkpoint selection")

    template = torch_load(args.template_model)
    if abs(float(template.r_max) - 2.0) > 1.0e-12 or int(template.num_interactions) != 2:
        raise ValueError("trained core violates r_max/interactions freeze")
    spherical_lmax = int(template.spherical_harmonics.irreps_out.lmax)
    if spherical_lmax != 2:
        raise ValueError(f"trained core max_ell={spherical_lmax}, expected 2")
    candidate_paths = fixed_checkpoint_paths(args.training_dir / "checkpoints")
    candidate_paths["final"] = args.template_model

    records = []
    models: dict[str, object] = {}
    for label, path in candidate_paths.items():
        if label == "final":
            model = copy.deepcopy(template)
        else:
            checkpoint = torch_load(path)
            model = copy.deepcopy(template)
            model.load_state_dict(checkpoint["model"], strict=True)
            del checkpoint
        result = evaluate_model(model, e50, harmonic, args.device)
        result.update({"label": label, "path": str(path), "sha256": sha256(path)})
        records.append(result)
        models[label] = model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    passing = [item for item in records if item["passes_fixed_gate"]]
    pool = passing if passing else records
    selected = min(
        pool,
        key=lambda item: (
            item["normalized_worst_gate_score"],
            240 if item["label"] == "final" else int(item["label"].replace("epoch", "")),
        ),
    )
    args.selected_model.parent.mkdir(parents=True, exist_ok=True)
    if passing:
        artifact_path = args.selected_model
        artifact_kind = "selected_core_for_routed_tail_force_gate"
    else:
        artifact_path = args.selected_model.with_name("best_diagnostic_core.model")
        artifact_kind = "best_diagnostic_core_not_authorized_for_routed_tail"
    opposite_artifact = (
        args.selected_model.with_name("best_diagnostic_core.model")
        if passing
        else args.selected_model
    )
    if opposite_artifact.exists():
        raise FileExistsError(
            f"opposite gate artifact exists; refusing an ambiguous recovery: {opposite_artifact}"
        )
    artifact_reused = False
    if artifact_path.exists():
        existing_model = torch_load(artifact_path)
        if not exact_model_state_equal(existing_model, models[selected["label"]]):
            raise ValueError(
                f"existing checkpoint artifact is not the exact selected state: {artifact_path}"
            )
        artifact_reused = True
        del existing_model
    else:
        temporary_model = artifact_path.with_name(artifact_path.name + ".tmp")
        torch.save(models[selected["label"]], temporary_model)
        os.replace(temporary_model, artifact_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.parent / "evaluate_script_snapshot.py"
    shutil.copy2(Path(__file__).resolve(), snapshot)
    summary = {
        "status": "R2M_core_checkpoint_gate_passed" if passing else "R2M_core_checkpoint_gate_failed",
        "force_gate_allows_routed_tail_stage": bool(passing),
        "full_composite_deployment_authorized": False,
        "finite_temperature_validation_closed": False,
        "selection_scope": "E50 exact fixed-smearing seed1 plus harmonic validation only",
        "fixed_candidate_labels": [f"epoch{value}" for value in FIXED_EPOCHS] + ["final"],
        "fixed_thresholds": LIMITS,
        "energy_gate": "not_evaluated_forces_only_first_R2M_lane",
        "seed2_or_support_read_for_selection": False,
        "seed2_limitation": "opened development from prior R2K; reserved for post-selection report, not blind/external",
        "model_semantics": {
            "role": "replacement short delta core",
            "combination": "frozen v11 foundation + this core + frozen q6",
            "depth3_or_current_S0_deployed": False,
            "new_core_no_wrap_diameter_bound_A": 8.0,
            "whole_composite_strict_no_wrap_claimed": False,
        },
        "checkpoint_choice": selected,
        "checkpoint_artifact": {
            "kind": artifact_kind,
            "path": str(artifact_path),
            "sha256": sha256(artifact_path),
            "reused_exact_state_after_interrupted_evaluation": artifact_reused,
        },
        "records": records,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "training_freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
            "training_runtime": {"path": str(runtime_path), "sha256": sha256(runtime_path)},
            "template_model": {"path": str(args.template_model), "sha256": sha256(args.template_model)},
            "script_snapshot": {"path": str(snapshot), "sha256": sha256(snapshot)},
        },
    }
    strict_json(args.output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
