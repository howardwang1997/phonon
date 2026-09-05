#!/usr/bin/env python3
"""Post-selection report for a passed R2M replacement core.

This program is deliberately separated from checkpoint selection.  It first
validates the completed core-gate artifact and its full hash chain.  Only
after that authorization boundary does it read the opened E50 seed2 data or
the nine support configurations.  Nothing reported here can change the
selected checkpoint.

The force-only R2M model has an arbitrary additive energy constant.  Energy
errors are therefore reported only after removing one global offset within a
fixed-condition ensemble.  E50 core target energies are reconstructed as
``E_DFT - E_v11 - E_q6``; support9 already carries the centered short-target
energy convention used by R2C.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase.build import graphene
from ase.io import read
from mace.calculators import MACECalculator


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    load_operator,
    structure_mapping,
)


FIXED_DEGAUSS_RY = 0.0019000869380739254
KB_EV_K = 8.617333262145e-5
EXPECTED_HASHES = {
    "prepared_manifest": "bb20a86af073e344c38e39ab14d3a80af9e39fa051591bed2b039ae643ffe31e",
    "prepared_train": "789b65e1a2c3b260e24d80f008407c7ee55e02676ff3bcf9a875fe9de294519c",
    "prepared_valid": "92ad1508e1067265445b79679cfe5142d23b2507dab2aa44397578cee8c7a6da",
    "reserved_seed2": "ab8fb629501134ba9e94d11fcc67a6363971c2ccbf2bbae0e2d236a8c23988b1",
    "support_source": "db5bae0b8643fce8bcd1946f2619169efa6f82a7267292cec86fc2db0271937d",
    "e50_all": "b5170472fc637e336f3856981f9db0d3e81eac047be36e6cafb76088e0dedc8e",
    "r0_predictions": "08c92de89bc330c2c64c81fd0387b995a4614ddb4a56aac4da3fa3a16f08691e",
    "r0_summary": "9c9d7ec16302c5e86c021a793c8a269ab88f294999336a789774061fcf5edb28",
    "base_model": "4d32426672d53e3adc86499d14e4c4164c05e879a56811ff54add03cd1bb6ce7",
    "q6_operator": "114fb9a369ed106e5a89a9c05223aed7667fd737714f3d42516ebdb8e2e475c3",
}
LIMITS = {
    "force_RMSE_meV_A": 30.0,
    "force_max_abs_meV_A": 200.0,
    "Aprime_RMS_meV_A": 15.0,
    "Aprime_slope_relative_error_abs": 0.05,
    "centered_energy_RMSE_meV_config": 19.4,
    "importance_weight_ESS_fraction": 0.3,
    "harmonic_force_RMSE_meV_A": 9.044673057081592,
    "harmonic_force_max_abs_meV_A": 200.0,
    "auxiliary_force_RMSE_meV_A": 30.0,
    "auxiliary_force_max_abs_meV_A": 200.0,
    "size_local_force_max_abs_difference_meV_A": 1.0,
    "size_defect_energy_abs_difference_meV": 1.0,
}
R2M_RMS_NORMALIZED_WEIGHTS = {
    "r2m_exact_e50_train": 0.004687324749326601,
    "r2m_exact_e50_validation": 0.004687324749326601,
    "r2m_harmonic_train": 1.0,
    "r2m_harmonic_validation": 1.0,
    "r2m_auxiliary_T300": 0.0006987885239726618,
    "r2m_auxiliary_T600": 0.0006987885239726618,
}
R2N_GATE_NORMALIZED_WEIGHTS = {
    "r2m_exact_e50_train": 0.3272244428,
    "r2m_exact_e50_validation": 0.3272244428,
    "r2m_harmonic_train": 1.0,
    "r2m_harmonic_validation": 1.0,
    "r2m_auxiliary_T300": 0.04544783928,
    "r2m_auxiliary_T600": 0.04544783928,
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


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} hash mismatch: observed={observed}, expected={expected}"
        )
    return observed


def force_metrics(errors: np.ndarray | list[np.ndarray]) -> dict:
    values = np.asarray(errors, float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("force metrics require nonempty finite values")
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(values**2))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(values))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(values))),
        "n_force_components": int(values.size),
    }


def complex_metrics(errors: np.ndarray | list[complex]) -> dict:
    values = np.asarray(errors, complex).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("complex metrics require nonempty finite values")
    magnitudes = np.abs(values)
    return {
        "RMS_meV_A": float(1000.0 * np.sqrt(np.mean(magnitudes**2))),
        "MAE_meV_A": float(1000.0 * np.mean(magnitudes)),
        "max_abs_meV_A": float(1000.0 * np.max(magnitudes)),
        "n_values": int(values.size),
    }


def centered_energy_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    predicted = np.asarray(predicted, float)
    target = np.asarray(target, float)
    if predicted.shape != target.shape or predicted.ndim != 1 or predicted.size < 2:
        raise ValueError("centered energy metrics require equal one-dimensional ensembles")
    error = predicted - target
    if not np.all(np.isfinite(error)):
        raise ValueError("energy errors are non-finite")
    offset = float(np.mean(error))
    centered = error - offset
    return centered, {
        "availability": "available",
        "global_offset_eV": offset,
        "RMSE_meV_config": float(1000.0 * np.sqrt(np.mean(centered**2))),
        "MAE_meV_config": float(1000.0 * np.mean(np.abs(centered))),
        "max_abs_meV_config": float(1000.0 * np.max(np.abs(centered))),
        "n_configurations": int(centered.size),
    }


def importance_ess_fraction(centered_energy_error_eV: np.ndarray, temperature_K: float) -> float:
    values = np.asarray(centered_energy_error_eV, float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("ESS requires a finite multi-configuration energy ensemble")
    if temperature_K <= 0.0:
        raise ValueError("ESS temperature must be positive")
    log_weights = -values / (KB_EV_K * temperature_K)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return float((np.sum(weights) ** 2 / np.sum(weights**2)) / len(weights))


def restoring_slope(coordinates: np.ndarray, projected_forces: np.ndarray) -> float:
    coordinates = np.asarray(coordinates, complex)
    projected_forces = np.asarray(projected_forces, complex)
    denominator = float(np.vdot(coordinates, coordinates).real)
    if denominator <= 1.0e-20:
        raise ValueError("all A-prime coordinates are zero")
    return -float(np.vdot(coordinates, projected_forces).real / denominator)


def geometry_fingerprint(structure) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(structure.numbers, dtype="<i8").tobytes())
    digest.update(np.round(np.asarray(structure.cell, float), 10).astype("<f8").tobytes())
    digest.update(
        np.round(np.asarray(structure.positions, float), 10).astype("<f8").tobytes()
    )
    return digest.hexdigest()


def gate_contract_errors(gate: dict, manifest: dict, selected_hash: str) -> list[str]:
    """Pure contract check; it intentionally does not inspect reserved data."""
    errors = []
    if gate.get("status") != "R2M_core_checkpoint_gate_passed":
        errors.append("core gate status is not passed")
    if gate.get("force_gate_allows_routed_tail_stage") is not True:
        errors.append("force gate does not allow the next stage")
    if gate.get("seed2_or_support_read_for_selection") is not False:
        errors.append("gate does not attest that seed2/support were excluded")
    if gate.get("full_composite_deployment_authorized") is not False:
        errors.append("unexpected deployment authorization in the core-only gate")
    if gate.get("finite_temperature_validation_closed") is not False:
        errors.append("unexpected finite-temperature closure in the core-only gate")
    choice = gate.get("checkpoint_choice", {})
    if choice.get("passes_fixed_gate") is not True:
        errors.append("selected checkpoint itself did not pass its fixed gate")
    artifact = gate.get("checkpoint_artifact", {})
    if artifact.get("kind") != "selected_core_for_routed_tail_force_gate":
        errors.append("gate artifact is not an authorized selected core")
    if artifact.get("sha256") != selected_hash:
        errors.append("selected core hash differs from the gate artifact")
    passing_labels = {
        str(record.get("label"))
        for record in gate.get("records", [])
        if record.get("passes_fixed_gate") is True
    }
    if str(choice.get("label")) not in passing_labels:
        errors.append("chosen label is absent from the passing record set")
    inputs = gate.get("inputs", {})
    if inputs.get("manifest", {}).get("sha256") != EXPECTED_HASHES["prepared_manifest"]:
        errors.append("gate was not evaluated against the frozen prepared manifest")
    if manifest.get("status") != "frozen_before_R2M_support_free_core_training":
        errors.append("prepared data status is not frozen")
    if manifest.get("model_role") != "replacement_short_delta_core":
        errors.append("prepared model role is not replacement core")
    if manifest.get("energy_training_enabled") is not False:
        errors.append("R2M core is not the frozen forces-only lane")
    leakage = manifest.get("leakage_control", {})
    if leakage.get("seed2_in_gradients_scales_or_checkpoint_selection") is not False:
        errors.append("manifest does not exclude seed2 from training/selection")
    if leakage.get("support_labels_or_geometries_in_train_or_valid") is not False:
        errors.append("manifest does not exclude support from training/validation")
    return errors


def authorize_before_reserved_open(
    data_dir: Path,
    training_dir: Path,
    gate_path: Path,
    selected_model_path: Path,
) -> tuple[dict, dict, dict]:
    """Validate the gate and training chain before any reserved ASE read."""
    if gate_path.resolve() != (training_dir / "core_checkpoint_gate.json").resolve():
        raise ValueError("gate JSON must be the training directory core gate")
    if selected_model_path.resolve() != (training_dir / "selected_core.model").resolve():
        raise ValueError("selected model must be the training directory selected core")
    if not (training_dir / "CORE_GATE_PASSED").exists():
        raise ValueError("CORE_GATE_PASSED marker is absent")
    if (training_dir / "CORE_GATE_FAILED").exists():
        raise ValueError("ambiguous CORE_GATE_FAILED marker is present")
    if (training_dir / "best_diagnostic_core.model").exists():
        raise ValueError("diagnostic-only core artifact is present beside selected core")

    manifest_path = data_dir / "manifest.json"
    require_hash(
        manifest_path, EXPECTED_HASHES["prepared_manifest"], "prepared manifest"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    selected_hash = require_hash(
        selected_model_path,
        str(gate.get("checkpoint_artifact", {}).get("sha256", "")),
        "selected core",
    )
    errors = gate_contract_errors(gate, manifest, selected_hash)
    if errors:
        raise ValueError("post-selection authorization failed: " + "; ".join(errors))

    expected_gate_inputs = {
        "manifest": manifest_path,
        "training_freeze": training_dir / "training_freeze.json",
        "training_runtime": training_dir / "training_runtime.json",
        "script_snapshot": training_dir
        / "code_snapshots"
        / "evaluate_graphene_r2m_core_checkpoints.py",
    }
    for label, path in expected_gate_inputs.items():
        recorded = gate.get("inputs", {}).get(label, {})
        require_hash(path, str(recorded.get("sha256", "")), f"gate input {label}")
    runtime_path = expected_gate_inputs["training_runtime"]
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("status") != "training_complete_pending_fixed_checkpoint_gate":
        raise ValueError("training runtime status is not complete pending gate")
    final_path = Path(str(runtime.get("final_model", "")))
    if not final_path.is_absolute():
        final_path = training_dir / final_path
    require_hash(
        final_path,
        str(runtime.get("final_model_sha256", "")),
        "training final model",
    )
    recorded_template = gate.get("inputs", {}).get("template_model", {})
    require_hash(
        final_path,
        str(recorded_template.get("sha256", "")),
        "gate template model",
    )
    freeze = json.loads(
        expected_gate_inputs["training_freeze"].read_text(encoding="utf-8")
    )
    if freeze.get("status") != "frozen_before_R2M_core_training":
        raise ValueError("training freeze status changed")
    if freeze.get("evaluator", {}).get("sha256") != gate["inputs"][
        "script_snapshot"
    ]["sha256"]:
        raise ValueError("training freeze/evaluator snapshot hash chain is broken")
    if freeze.get("r2n_status") is None:
        expected_weights = R2M_RMS_NORMALIZED_WEIGHTS
        loss_contract = "R2M_RMS_NORMALIZED_V1"
    else:
        if (
            freeze.get("r2n_status")
            != "frozen_before_R2N_gate_normalized_training"
            or freeze.get("experiment_stage")
            != "R2N_loss_only_correction_after_failed_R2M_core_gate"
            or freeze.get("loss", {}).get("contract_version")
            != "R2N_GATE_NORMALIZED_V1"
            or freeze.get("loss", {}).get("only_change_from_R2M")
            != "config_type_weights"
            or freeze.get("loss", {}).get("dynamic_gradient_balancing") is not False
        ):
            raise ValueError("R2N loss-only freeze contract changed")
        expected_weights = R2N_GATE_NORMALIZED_WEIGHTS
        loss_contract = "R2N_GATE_NORMALIZED_V1"
    if freeze.get("loss", {}).get("config_type_weights") != expected_weights:
        raise ValueError(f"{loss_contract} configuration weights changed")

    model = torch_load(selected_model_path)
    if abs(float(model.r_max) - 2.0) > 1.0e-12:
        raise ValueError("selected core r_max differs from the frozen architecture")
    if int(model.num_interactions) != 2:
        raise ValueError("selected core interaction count differs from the freeze")
    if int(model.spherical_harmonics.irreps_out.lmax) != 2:
        raise ValueError("selected core max_ell differs from the freeze")
    return gate, manifest, {
        "model": model,
        "sha256": selected_hash,
        "loss_contract": loss_contract,
        "config_type_weights": expected_weights,
    }


def make_calculator(model, device: str) -> MACECalculator:
    return MACECalculator(models=model, device=device, default_dtype="float32")


def predict_one(structure, calculator: MACECalculator) -> tuple[float, np.ndarray]:
    atoms = structure.copy()
    atoms.calc = calculator
    return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces(), float)


def predict_many(
    structures: list, calculator: MACECalculator
) -> tuple[np.ndarray, np.ndarray]:
    energies = []
    forces = []
    for structure in structures:
        energy, force = predict_one(structure, calculator)
        energies.append(energy)
        forces.append(force)
    return np.asarray(energies, float), np.asarray(forces, float)


def q6_response(
    structure, force_constants: np.ndarray, reference: np.ndarray, cell: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    mapping, displacement = structure_mapping(structure, reference, cell)
    local_fc = force_constants[mapping][:, mapping]
    force = -np.einsum("ijab,jb->ia", local_fc, displacement)
    energy = 0.5 * float(
        np.einsum("ia,ijab,jb->", displacement, local_fc, displacement)
    )
    return energy, force, mapping, displacement


def aprime_result(
    coordinates: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict:
    predicted_slope = restoring_slope(coordinates, predicted)
    target_slope = restoring_slope(coordinates, target)
    if abs(target_slope) <= 1.0e-14:
        raise ValueError("DFT A-prime restoring slope is zero")
    relative = (predicted_slope - target_slope) / target_slope
    return {
        "force_error": complex_metrics(predicted - target),
        "predicted_restoring_slope_eV_A2": predicted_slope,
        "DFT_restoring_slope_eV_A2": target_slope,
        "relative_slope_error": float(relative),
    }


def fixed_condition_gate(force: dict, aprime: dict, energy: dict, ess: float) -> dict:
    observed = {
        "force_RMSE_meV_A": force["RMSE_meV_A"],
        "force_max_abs_meV_A": force["max_abs_meV_A"],
        "Aprime_RMS_meV_A": aprime["force_error"]["RMS_meV_A"],
        "Aprime_slope_relative_error_abs": abs(aprime["relative_slope_error"]),
        "centered_energy_RMSE_meV_config": energy["RMSE_meV_config"],
        "importance_weight_ESS_fraction": ess,
    }
    checks = {}
    for name, value in observed.items():
        threshold = LIMITS[name]
        passed = value >= threshold if name == "importance_weight_ESS_fraction" else value <= threshold
        checks[name] = {"observed": value, "threshold": threshold, "pass": bool(passed)}
    return {"checks": checks, "passes_all": bool(all(v["pass"] for v in checks.values()))}


def evaluate_seed2(
    reserved: list,
    e50_all: list,
    base_calculator: MACECalculator,
    core_calculator: MACECalculator,
    force_constants: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
    mode_operator: np.ndarray,
) -> dict:
    source_by_key = {
        (int(item.info["trajectory_seed"]), int(item.info["snapshot_index"])): item
        for item in e50_all
    }
    if len(source_by_key) != 60:
        raise ValueError("E50 all60 keys are not unique")
    errors = []
    core_errors = []
    coordinates = []
    projected_predicted = []
    projected_target = []
    core_energies = []
    core_target_energies = []
    replay = {
        "foundation_force_max_abs_eV_A": 0.0,
        "q6_force_max_abs_eV_A": 0.0,
        "stored_core_target_reconstruction_max_abs_eV_A": 0.0,
        "embedded_Aprime_mode_max_abs": 0.0,
        "embedded_Aprime_coordinate_max_abs_A": 0.0,
        "reserved_vs_original_DFT_force_max_abs_eV_A": 0.0,
    }
    records = []
    for index, structure in enumerate(reserved):
        key = (int(structure.info["trajectory_seed"]), int(structure.info["snapshot_index"]))
        if key[0] != 2 or key not in source_by_key:
            raise ValueError(f"invalid reserved seed2 key {key}")
        original = source_by_key[key]
        if geometry_fingerprint(original) != geometry_fingerprint(structure):
            raise ValueError(f"reserved/original E50 geometry differs at {key}")
        base_energy, base_force = predict_one(structure, base_calculator)
        core_energy, core_force = predict_one(structure, core_calculator)
        long_energy, long_force, mapping, displacement = q6_response(
            structure, force_constants, reference, cell
        )
        dft_force = np.asarray(structure.arrays["DFT_TOTAL_forces"], float)
        original_dft_force = np.asarray(original.arrays["REF_forces"], float)
        stored_base = np.asarray(structure.arrays["FOUNDATION_BASE_forces"], float)
        stored_long = np.asarray(structure.arrays["FROZEN_Q6_forces"], float)
        stored_target = np.asarray(structure.arrays["CORE_TARGET_forces"], float)
        replay["foundation_force_max_abs_eV_A"] = max(
            replay["foundation_force_max_abs_eV_A"],
            float(np.max(np.abs(base_force - stored_base))),
        )
        replay["q6_force_max_abs_eV_A"] = max(
            replay["q6_force_max_abs_eV_A"],
            float(np.max(np.abs(long_force - stored_long))),
        )
        replay["stored_core_target_reconstruction_max_abs_eV_A"] = max(
            replay["stored_core_target_reconstruction_max_abs_eV_A"],
            float(np.max(np.abs(stored_target - (dft_force - stored_base - stored_long)))),
        )
        replay["reserved_vs_original_DFT_force_max_abs_eV_A"] = max(
            replay["reserved_vs_original_DFT_force_max_abs_eV_A"],
            float(np.max(np.abs(dft_force - original_dft_force))),
        )
        mode = mode_operator[mapping]
        embedded_mode = np.asarray(structure.arrays["APRIME_mode_real"], float) + 1.0j * np.asarray(
            structure.arrays["APRIME_mode_imag"], float
        )
        coordinate = np.vdot(mode.reshape(-1), displacement.reshape(-1))
        embedded_coordinate = complex(
            float(structure.info["APRIME_coordinate_real_A"]),
            float(structure.info["APRIME_coordinate_imag_A"]),
        )
        replay["embedded_Aprime_mode_max_abs"] = max(
            replay["embedded_Aprime_mode_max_abs"],
            float(np.max(np.abs(mode - embedded_mode))),
        )
        replay["embedded_Aprime_coordinate_max_abs_A"] = max(
            replay["embedded_Aprime_coordinate_max_abs_A"],
            float(abs(coordinate - embedded_coordinate)),
        )
        predicted_total = base_force + core_force + long_force
        error = predicted_total - dft_force
        projected_prediction = np.vdot(mode.reshape(-1), predicted_total.reshape(-1))
        projected_reference = np.vdot(mode.reshape(-1), dft_force.reshape(-1))
        errors.append(error)
        core_errors.append(core_force - stored_target)
        coordinates.append(coordinate)
        projected_predicted.append(projected_prediction)
        projected_target.append(projected_reference)
        core_energies.append(core_energy)
        core_target_energies.append(
            float(original.info["REF_energy"]) - base_energy - long_energy
        )
        records.append(
            {
                "trajectory_seed": key[0],
                "snapshot_index": key[1],
                "force_RMSE_meV_A": force_metrics(error)["RMSE_meV_A"],
                "force_max_abs_meV_A": force_metrics(error)["max_abs_meV_A"],
                "Aprime_force_error_abs_meV_A": float(
                    1000.0 * abs(projected_prediction - projected_reference)
                ),
            }
        )
    strict_replay_keys = (
        "q6_force_max_abs_eV_A",
        "stored_core_target_reconstruction_max_abs_eV_A",
        "embedded_Aprime_mode_max_abs",
        "embedded_Aprime_coordinate_max_abs_A",
        "reserved_vs_original_DFT_force_max_abs_eV_A",
    )
    if any(replay[key] > 2.0e-7 for key in strict_replay_keys):
        raise ValueError(f"seed2 provenance replay failed: {replay}")
    if replay["foundation_force_max_abs_eV_A"] > 2.0e-4:
        raise ValueError(f"seed2 foundation replay exceeds 0.2 meV/A: {replay}")
    errors_array = np.asarray(errors)
    core_errors_array = np.asarray(core_errors)
    force = force_metrics(errors_array)
    aprime = aprime_result(
        np.asarray(coordinates),
        np.asarray(projected_predicted),
        np.asarray(projected_target),
    )
    centered, energy = centered_energy_metrics(
        np.asarray(core_energies), np.asarray(core_target_energies)
    )
    energy.update(
        {
            "target_definition": "E_DFT_total - E_frozen_v11 - E_frozen_q6",
            "interpretation": (
                "force-only conservative core; only within-ensemble energy differences "
                "after one global offset are evaluated"
            ),
        }
    )
    ess = importance_ess_fraction(centered, 450.0)
    for record, value in zip(records, centered, strict=True):
        record["centered_core_energy_error_meV_config"] = float(1000.0 * value)
    return {
        "scope": "opened development; excluded from R2M gradients, scales, and checkpoint selection",
        "n_configurations": len(reserved),
        "force_error_replacement_core_vs_core_target": force_metrics(core_errors_array),
        "force_error_full_v11_plus_core_plus_q6": force,
        "Aprime_projection_full_v11_plus_core_plus_q6": aprime,
        "centered_core_energy_error": energy,
        "importance_weight_ESS_fraction_at_450K": ess,
        "fixed_report_gate": fixed_condition_gate(force, aprime, energy, ess),
        "provenance_replay": replay,
        "records": records,
    }


def evaluate_support9(
    support: list,
    base_calculator: MACECalculator,
    core_calculator: MACECalculator,
    force_constants: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
    mode_operator: np.ndarray,
) -> dict:
    errors = []
    core_errors = []
    coordinates = []
    projected_predicted = []
    projected_target = []
    core_energies = []
    core_target_energies = []
    replay = {
        "foundation_force_max_abs_eV_A": 0.0,
        "foundation_energy_abs_eV": 0.0,
        "q6_force_max_abs_eV_A": 0.0,
        "q6_energy_abs_eV": 0.0,
        "target_force_reconstruction_max_abs_eV_A": 0.0,
        "stored_operator_mapping_max_abs_index": 0,
    }
    records = []
    for structure in support:
        base_energy, base_force = predict_one(structure, base_calculator)
        core_energy, core_force = predict_one(structure, core_calculator)
        long_energy, long_force, mapping, displacement = q6_response(
            structure, force_constants, reference, cell
        )
        stored_mapping = np.asarray(structure.arrays["operator_mapping"], int)
        if sorted(stored_mapping.tolist()) != list(range(len(structure))):
            raise ValueError("support9 stored operator mapping is not a permutation")
        replay["stored_operator_mapping_max_abs_index"] = max(
            replay["stored_operator_mapping_max_abs_index"],
            int(np.max(np.abs(mapping - stored_mapping))),
        )
        stored_base = np.asarray(structure.arrays["BASE_forces"], float)
        stored_long = np.asarray(structure.arrays["LONG_RANGE_forces"], float)
        short_target = np.asarray(structure.arrays["SHORT_RANGE_TARGET_forces"], float)
        total_target = np.asarray(structure.arrays["TOTAL_forces"], float)
        replay["foundation_force_max_abs_eV_A"] = max(
            replay["foundation_force_max_abs_eV_A"],
            float(np.max(np.abs(base_force - stored_base))),
        )
        replay["foundation_energy_abs_eV"] = max(
            replay["foundation_energy_abs_eV"],
            abs(base_energy - float(structure.info["BASE_energy"])),
        )
        replay["q6_force_max_abs_eV_A"] = max(
            replay["q6_force_max_abs_eV_A"],
            float(np.max(np.abs(long_force - stored_long))),
        )
        replay["q6_energy_abs_eV"] = max(
            replay["q6_energy_abs_eV"],
            abs(long_energy - float(structure.info["LONG_RANGE_energy"])),
        )
        replay["target_force_reconstruction_max_abs_eV_A"] = max(
            replay["target_force_reconstruction_max_abs_eV_A"],
            float(np.max(np.abs(total_target - (stored_base + stored_long + short_target)))),
        )
        predicted_total = base_force + core_force + long_force
        error = predicted_total - total_target
        mode = mode_operator[mapping]
        coordinate = np.vdot(mode.reshape(-1), displacement.reshape(-1))
        predicted_mode = np.vdot(mode.reshape(-1), predicted_total.reshape(-1))
        target_mode = np.vdot(mode.reshape(-1), total_target.reshape(-1))
        errors.append(error)
        core_errors.append(core_force - short_target)
        coordinates.append(coordinate)
        projected_predicted.append(predicted_mode)
        projected_target.append(target_mode)
        core_energies.append(core_energy)
        core_target_energies.append(float(structure.info["SHORT_RANGE_TARGET_energy"]))
        records.append(
            {
                "sscha_index": int(structure.info["sscha_index"]),
                "support_level": str(structure.info["corrected_X0_support_level"]),
                "force_RMSE_meV_A": force_metrics(error)["RMSE_meV_A"],
                "force_max_abs_meV_A": force_metrics(error)["max_abs_meV_A"],
                "Aprime_force_error_abs_meV_A": float(
                    1000.0 * abs(predicted_mode - target_mode)
                ),
            }
        )
    if (
        replay["q6_force_max_abs_eV_A"] > 2.0e-7
        or replay["q6_energy_abs_eV"] > 2.0e-7
        or replay["target_force_reconstruction_max_abs_eV_A"] > 2.0e-7
        or replay["stored_operator_mapping_max_abs_index"] != 0
    ):
        raise ValueError(f"support9 q6/target provenance replay failed: {replay}")
    if (
        replay["foundation_force_max_abs_eV_A"] > 2.0e-4
        or replay["foundation_energy_abs_eV"] > 2.0e-4
    ):
        raise ValueError(f"support9 foundation replay exceeds fixed tolerance: {replay}")
    errors_array = np.asarray(errors)
    core_errors_array = np.asarray(core_errors)
    force = force_metrics(errors_array)
    aprime = aprime_result(
        np.asarray(coordinates),
        np.asarray(projected_predicted),
        np.asarray(projected_target),
    )
    centered, energy = centered_energy_metrics(
        np.asarray(core_energies), np.asarray(core_target_energies)
    )
    energy.update(
        {
            "target_definition": "stored R2C SHORT_RANGE_TARGET_energy",
            "interpretation": (
                "support9 uses one fixed T_lat/degauss condition and equal atom counts; "
                "the force-only core gauge is removed by one global offset"
            ),
        }
    )
    ess = importance_ess_fraction(centered, 450.0)
    for record, value in zip(records, centered, strict=True):
        record["centered_core_energy_error_meV_config"] = float(1000.0 * value)
    return {
        "scope": "opened development/support; never used by selected core training or checkpoint selection",
        "n_configurations": len(support),
        "force_error_replacement_core_vs_short_target": force_metrics(core_errors_array),
        "force_error_full_v11_plus_core_plus_q6": force,
        "Aprime_projection_full_v11_plus_core_plus_q6": aprime,
        "centered_core_energy_error": energy,
        "importance_weight_ESS_fraction_at_450K": ess,
        "fixed_report_gate": fixed_condition_gate(force, aprime, energy, ess),
        "provenance_replay": replay,
        "records": records,
    }


def retention_group(
    structures: list,
    core_calculator: MACECalculator,
    rmse_limit: float,
    max_limit: float,
    scope: str,
) -> dict:
    _, predicted = predict_many(structures, core_calculator)
    target = np.asarray([np.asarray(item.arrays["REF_forces"], float) for item in structures])
    errors = predicted - target
    metrics = force_metrics(errors)
    checks = {
        "force_RMSE": {
            "observed_meV_A": metrics["RMSE_meV_A"],
            "threshold_meV_A": rmse_limit,
            "pass": bool(metrics["RMSE_meV_A"] <= rmse_limit),
        },
        "force_max_abs": {
            "observed_meV_A": metrics["max_abs_meV_A"],
            "threshold_meV_A": max_limit,
            "pass": bool(metrics["max_abs_meV_A"] <= max_limit),
        },
    }
    return {
        "scope": scope,
        "n_configurations": len(structures),
        "force_error_replacement_core": metrics,
        "energy_error": {
            "availability": "unavailable",
            "reason": (
                "the R2M preparation deliberately set REF_energy=0 and energy weight=0; "
                "these prepared harmonic/auxiliary files do not carry an audited core-energy target"
            ),
        },
        "fixed_retention_gate": {
            "checks": checks,
            "passes_all": bool(all(item["pass"] for item in checks.values())),
        },
    }


def evaluate_retention(train: list, valid: list, core_calculator: MACECalculator) -> dict:
    harmonic_train = [
        item for item in train if str(item.info["r2m_target_role"]) == "harmonic_train"
    ]
    harmonic_valid = [
        item for item in valid if str(item.info["r2m_target_role"]) == "harmonic_validation"
    ]
    auxiliary_300 = [
        item
        for item in train
        if str(item.info["r2m_target_role"]) == "auxiliary_legacy_thermal"
        and float(item.info["lattice_temperature_K"]) == 300.0
    ]
    auxiliary_600 = [
        item
        for item in train
        if str(item.info["r2m_target_role"]) == "auxiliary_legacy_thermal"
        and float(item.info["lattice_temperature_K"]) == 600.0
    ]
    if tuple(map(len, (harmonic_train, harmonic_valid, auxiliary_300, auxiliary_600))) != (
        72,
        25,
        36,
        36,
    ):
        raise ValueError("prepared harmonic/auxiliary retention group counts changed")
    groups = {
        "harmonic_train": retention_group(
            harmonic_train,
            core_calculator,
            LIMITS["harmonic_force_RMSE_meV_A"],
            LIMITS["harmonic_force_max_abs_meV_A"],
            "training replay; not external",
        ),
        "harmonic_validation": retention_group(
            harmonic_valid,
            core_calculator,
            LIMITS["harmonic_force_RMSE_meV_A"],
            LIMITS["harmonic_force_max_abs_meV_A"],
            "used in the fixed checkpoint gate; not external",
        ),
        "auxiliary_T300": retention_group(
            auxiliary_300,
            core_calculator,
            LIMITS["auxiliary_force_RMSE_meV_A"],
            LIMITS["auxiliary_force_max_abs_meV_A"],
            "legacy mixed-smearing training replay; not fixed-smearing validation",
        ),
        "auxiliary_T600": retention_group(
            auxiliary_600,
            core_calculator,
            LIMITS["auxiliary_force_RMSE_meV_A"],
            LIMITS["auxiliary_force_max_abs_meV_A"],
            "legacy mixed-smearing training replay; not fixed-smearing validation",
        ),
    }
    return {
        "groups": groups,
        "passes_all_available_force_gates": bool(
            all(value["fixed_retention_gate"]["passes_all"] for value in groups.values())
        ),
    }


def local_perturbation(size: int, lattice_constant_A: float):
    atoms = graphene(
        a=lattice_constant_A,
        size=(size, size, 1),
        vacuum=7.5,
    )
    atoms.pbc = (True, True, True)
    scaled = atoms.get_scaled_positions(wrap=True)
    center = int(np.argmin(np.sum((scaled[:, :2] - np.array([0.5, 0.5])) ** 2, axis=1)))
    vectors = atoms.get_distances(center, np.arange(len(atoms)), mic=True, vector=True)
    distances = np.linalg.norm(vectors, axis=1)
    neighbours = np.where((distances > 1.0e-8) & (distances < 1.75))[0]
    if len(neighbours) != 3:
        raise ValueError("ideal graphene perturbation does not have three neighbours")
    angles = np.arctan2(vectors[neighbours, 1], vectors[neighbours, 0])
    neighbours = neighbours[np.argsort(angles)]
    ideal = atoms.copy()
    perturbed = atoms.copy()
    perturbed.positions[center] += np.array([0.070, -0.040, 0.030])
    perturbed.positions[neighbours] += np.array(
        [
            [-0.020, 0.010, -0.010],
            [0.015, -0.012, 0.006],
            [0.005, 0.002, 0.004],
        ]
    )
    matched = np.asarray([center, *neighbours.tolist()], int)
    local_vectors = ideal.get_distances(center, matched, mic=True, vector=True)
    return ideal, perturbed, matched, local_vectors


def defect_response(ideal, perturbed, calculator: MACECalculator) -> dict:
    ideal_energy, ideal_force = predict_one(ideal, calculator)
    perturbed_energy, perturbed_force = predict_one(perturbed, calculator)
    return {
        "defect_energy_eV": float(perturbed_energy - ideal_energy),
        "defect_force_eV_A": perturbed_force - ideal_force,
        "ideal_force_max_abs_eV_A": float(np.max(np.abs(ideal_force))),
    }


def compare_size_component(left: dict, right: dict, left_indices: np.ndarray, right_indices: np.ndarray) -> dict:
    local_difference = (
        np.asarray(left["defect_force_eV_A"])[left_indices]
        - np.asarray(right["defect_force_eV_A"])[right_indices]
    )
    energy_difference = float(left["defect_energy_eV"] - right["defect_energy_eV"])
    checks = {
        "matched_local_force": {
            "observed_max_abs_difference_meV_A": float(1000.0 * np.max(np.abs(local_difference))),
            "threshold_meV_A": LIMITS["size_local_force_max_abs_difference_meV_A"],
            "pass": bool(
                1000.0 * np.max(np.abs(local_difference))
                <= LIMITS["size_local_force_max_abs_difference_meV_A"]
            ),
        },
        "defect_energy": {
            "observed_abs_difference_meV": abs(1000.0 * energy_difference),
            "threshold_meV": LIMITS["size_defect_energy_abs_difference_meV"],
            "pass": bool(
                abs(1000.0 * energy_difference)
                <= LIMITS["size_defect_energy_abs_difference_meV"]
            ),
        },
    }
    return {
        "defect_energy_6x6_eV": float(left["defect_energy_eV"]),
        "defect_energy_8x8_eV": float(right["defect_energy_eV"]),
        "signed_defect_energy_difference_meV": float(1000.0 * energy_difference),
        "matched_local_force_difference": force_metrics(local_difference),
        "checks": checks,
        "passes_all": bool(all(value["pass"] for value in checks.values())),
    }


def evaluate_size_consistency(
    base_calculator: MACECalculator,
    core_calculator: MACECalculator,
    base_model,
    core_model,
    force_constants: np.ndarray,
    reference: np.ndarray,
    operator_cell: np.ndarray,
) -> dict:
    lattice_constant = float(np.linalg.norm(operator_cell[0]) / 6.0)
    systems = {}
    for size in (6, 8):
        ideal, perturbed, matched, local_vectors = local_perturbation(size, lattice_constant)
        foundation = defect_response(ideal, perturbed, base_calculator)
        core = defect_response(ideal, perturbed, core_calculator)
        short = {
            "defect_energy_eV": foundation["defect_energy_eV"] + core["defect_energy_eV"],
            "defect_force_eV_A": foundation["defect_force_eV_A"] + core["defect_force_eV_A"],
            "ideal_force_max_abs_eV_A": max(
                foundation["ideal_force_max_abs_eV_A"], core["ideal_force_max_abs_eV_A"]
            ),
        }
        entry = {
            "n_atoms": len(ideal),
            "matched_atom_indices": matched.tolist(),
            "matched_local_vectors_A": local_vectors.tolist(),
            "foundation": foundation,
            "replacement_core": core,
            "short_composite": short,
        }
        if size == 6:
            q6_ideal_energy, q6_ideal_force, _, _ = q6_response(
                ideal, force_constants, reference, operator_cell
            )
            q6_perturbed_energy, q6_perturbed_force, _, _ = q6_response(
                perturbed, force_constants, reference, operator_cell
            )
            q6 = {
                "availability": "available",
                "defect_energy_eV": float(q6_perturbed_energy - q6_ideal_energy),
                "defect_force_eV_A": q6_perturbed_force - q6_ideal_force,
            }
            entry["q6"] = q6
            entry["full_composite"] = {
                "availability": "available",
                "defect_energy_eV": short["defect_energy_eV"] + q6["defect_energy_eV"],
                "defect_force_eV_A": short["defect_force_eV_A"] + q6["defect_force_eV_A"],
            }
        else:
            entry["q6"] = {
                "availability": "unavailable",
                "reason": "the frozen matched electronic operator is a 6x6 q-grid/supercell operator; no audited 8x8 counterpart is frozen",
            }
            entry["full_composite"] = {
                "availability": "unavailable",
                "reason": "v11+core is evaluated, but a like-for-like frozen q6 term is unavailable at 8x8",
            }
        systems[size] = entry
    vector_difference = float(
        np.max(
            np.abs(
                np.asarray(systems[6]["matched_local_vectors_A"])
                - np.asarray(systems[8]["matched_local_vectors_A"])
            )
        )
    )
    if vector_difference > 1.0e-10:
        raise ValueError("6x6 and 8x8 local perturbation constructions differ")
    comparisons = {
        label: compare_size_component(
            systems[6][label],
            systems[8][label],
            np.asarray(systems[6]["matched_atom_indices"], int),
            np.asarray(systems[8]["matched_atom_indices"], int),
        )
        for label in ("foundation", "replacement_core", "short_composite")
    }
    for entry in systems.values():
        for component in ("foundation", "replacement_core", "short_composite", "q6", "full_composite"):
            value = entry.get(component, {})
            if isinstance(value.get("defect_force_eV_A"), np.ndarray):
                value["defect_force_eV_A"] = value["defect_force_eV_A"].tolist()
    base_diameter = 2.0 * float(base_model.r_max) * int(base_model.num_interactions)
    core_diameter = 2.0 * float(core_model.r_max) * int(core_model.num_interactions)
    return {
        "construction": (
            "identical center plus three-neighbour perturbation embedded in ideal 6x6 "
            "and 8x8 graphene; compare defect forces on the four perturbed atoms and "
            "the whole-cell defect energy"
        ),
        "lattice_constant_A": lattice_constant,
        "local_geometry_max_abs_difference_A": vector_difference,
        "theoretical_interaction_diameter_bound_A": {
            "foundation_v11": base_diameter,
            "replacement_core": core_diameter,
            "minimum_cell_length_6x6_A": float(
                min(np.linalg.norm(vector) for vector in np.asarray(operator_cell) if np.linalg.norm(vector) > 1.0)
            ),
            "note": "the empirical comparison is required even when a theoretical diameter exceeds the 6x6 cell length",
        },
        "systems": {"6x6": systems[6], "8x8": systems[8]},
        "cross_size_comparisons": comparisons,
        "short_composite_size_gate_passed": comparisons["short_composite"]["passes_all"],
        "full_v11_core_q6_size_gate": {
            "status": "unavailable_fail_closed",
            "pass": False,
            "reason": "no frozen and audited 8x8 q6 operator exists; the 6x6 q6 matrix is not extrapolated or zero-padded",
        },
    }


def validate_opened_inputs(
    data_dir: Path,
    support_data: Path,
    e50_all: Path,
    r0_predictions: Path,
    r0_summary_path: Path,
    base_model_path: Path,
    q6_operator_path: Path,
    manifest: dict,
) -> tuple[dict, np.ndarray]:
    """Called only after authorize_before_reserved_open succeeds."""
    paths = {
        "prepared_train": data_dir / "train.xyz",
        "prepared_valid": data_dir / "valid.xyz",
        "reserved_seed2": data_dir / "reserved_e50_seed2.xyz",
        "support_source": support_data,
        "e50_all": e50_all,
        "r0_predictions": r0_predictions,
        "r0_summary": r0_summary_path,
        "base_model": base_model_path,
        "q6_operator": q6_operator_path,
    }
    observed = {
        label: require_hash(path, EXPECTED_HASHES[label], label)
        for label, path in paths.items()
    }
    manifest_expectations = {
        "prepared_train": manifest["outputs"]["train.xyz"]["sha256"],
        "prepared_valid": manifest["outputs"]["valid.xyz"]["sha256"],
        "reserved_seed2": manifest["outputs"]["reserved_e50_seed2.xyz"]["sha256"],
        "support_source": manifest["inputs"]["r2c_train"]["sha256"],
        "e50_all": manifest["inputs"]["e50"]["sha256"],
        "r0_predictions": manifest["inputs"]["r0_npz"]["sha256"],
        "r0_summary": manifest["inputs"]["r0_summary"]["sha256"],
    }
    for label, expected in manifest_expectations.items():
        if observed[label] != expected:
            raise ValueError(f"{label} disagrees with the prepared manifest")
    r0_summary = json.loads(r0_summary_path.read_text(encoding="utf-8"))
    if r0_summary.get("status") != "current_S0_fixed_smearing_audit_complete":
        raise ValueError("R0 summary status changed")
    if r0_summary.get("condition") != {
        "lattice_temperature_K": 450.0,
        "smearing": "fermi-dirac",
        "degauss_Ry": FIXED_DEGAUSS_RY,
    }:
        raise ValueError("R0 fixed condition changed")
    if r0_summary["inputs"]["base_model"]["sha256"] != observed["base_model"]:
        raise ValueError("base model differs from the R0 audit")
    if r0_summary["inputs"]["operator"]["sha256"] != observed["q6_operator"]:
        raise ValueError("q6 operator differs from the R0 audit")
    with np.load(r0_predictions, allow_pickle=False) as data:
        required = {
            "Aprime_mode_operator_order",
            "trajectory_seed",
            "snapshot_index",
        }
        if not required.issubset(data.files):
            raise ValueError("R0 predictions lack A-prime provenance")
        mode = np.asarray(data["Aprime_mode_operator_order"], complex)
        seeds = np.asarray(data["trajectory_seed"], int)
        snapshots = np.asarray(data["snapshot_index"], int)
    if mode.shape != (72, 3) or seeds.shape != (60,) or snapshots.shape != (60,):
        raise ValueError("R0 prediction array shapes changed")
    if Counter(seeds.tolist()) != Counter({0: 20, 1: 20, 2: 20}):
        raise ValueError("R0 prediction seed counts changed")
    return {label: {"path": str(paths[label]), "sha256": value} for label, value in observed.items()}, mode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--selected-core", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--support-data", type=Path, required=True)
    parser.add_argument("--e50-all", type=Path, required=True)
    parser.add_argument("--r0-predictions", type=Path, required=True)
    parser.add_argument("--r0-summary", type=Path, required=True)
    parser.add_argument("--q6-operator", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Authorization boundary: no ASE read of seed2/support (and no output
    # directory creation) occurs before this function returns successfully.
    gate, manifest, selected = authorize_before_reserved_open(
        args.data_dir,
        args.training_dir,
        args.gate_json,
        args.selected_core,
    )
    input_records, mode_operator = validate_opened_inputs(
        args.data_dir,
        args.support_data,
        args.e50_all,
        args.r0_predictions,
        args.r0_summary,
        args.base_model,
        args.q6_operator,
        manifest,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = args.output_dir / "evaluate_post_selection_script_snapshot.py"
    shutil.copy2(Path(__file__).resolve(), snapshot)

    # Reserved/opened data are first parsed here, after the gate and hashes.
    reserved = read(args.data_dir / "reserved_e50_seed2.xyz", index=":")
    source_train = read(args.support_data, index=":")
    e50_all = read(args.e50_all, index=":")
    prepared_train = read(args.data_dir / "train.xyz", index=":")
    prepared_valid = read(args.data_dir / "valid.xyz", index=":")
    support = [
        item
        for item in source_train
        if str(item.info.get("delta_target_role", ""))
        == "fixed_smearing_support_repair"
    ]
    if len(reserved) != 20 or len(support) != 9 or len(e50_all) != 60:
        raise ValueError("reserved/support/E50 counts changed")
    if len({int(item.info["sscha_index"]) for item in support}) != 9:
        raise ValueError("support9 sscha indices are not unique")
    if any(
        int(item.info["trajectory_seed"]) != 2
        or str(item.info["r2m_target_role"]) != "exact_e50_seed2"
        or float(item.info["lattice_temperature_K"]) != 450.0
        or abs(float(item.info["degauss_Ry"]) - FIXED_DEGAUSS_RY) > 5.0e-10
        for item in reserved
    ):
        raise ValueError("reserved E50 seed2 scope changed")
    if any(
        float(item.info["lattice_temperature_K"]) != 450.0
        or abs(float(item.info["degauss_Ry"]) - FIXED_DEGAUSS_RY) > 5.0e-10
        for item in support
    ):
        raise ValueError("support9 fixed condition changed")

    force_constants, reference, operator_cell, operator_degauss = load_operator(
        args.q6_operator
    )
    if abs(operator_degauss - FIXED_DEGAUSS_RY) > 5.0e-10:
        raise ValueError("q6 operator degauss changed")
    base_model = torch_load(args.base_model)
    base_calculator = make_calculator(base_model, args.device)
    core_calculator = make_calculator(selected["model"], args.device)

    seed2 = evaluate_seed2(
        reserved,
        e50_all,
        base_calculator,
        core_calculator,
        force_constants,
        reference,
        operator_cell,
        mode_operator,
    )
    support9 = evaluate_support9(
        support,
        base_calculator,
        core_calculator,
        force_constants,
        reference,
        operator_cell,
        mode_operator,
    )
    retention = evaluate_retention(prepared_train, prepared_valid, core_calculator)
    size_consistency = evaluate_size_consistency(
        base_calculator,
        core_calculator,
        base_model,
        selected["model"],
        force_constants,
        reference,
        operator_cell,
    )

    available_checks_pass = bool(
        seed2["fixed_report_gate"]["passes_all"]
        and support9["fixed_report_gate"]["passes_all"]
        and retention["passes_all_available_force_gates"]
        and size_consistency["short_composite_size_gate_passed"]
    )
    report = {
        "status": "R2M_post_selection_report_complete",
        "selected_checkpoint_changed": False,
        "selection_effect": "none; every metric in this file is post-selection only",
        "opened_data_boundary": {
            "authorization_required_before_ASE_read": "R2M_core_checkpoint_gate_passed",
            "seed2_or_support_used_for_training_scales_or_checkpoint_selection": False,
            "seed2_status": "opened development from prior R2K; not blind/external",
            "support9_status": "opened development/support; not blind/external",
        },
        "model_semantics": {
            "combination": "frozen v11 foundation + selected replacement core + frozen q6 where defined",
            "training_loss_contract": selected["loss_contract"],
            "training_config_type_weights": selected["config_type_weights"],
            "depth3_or_current_S0_deployed": False,
            "temperature_or_smearing_is_core_input": False,
            "full_EPC_spectral_replacement_evaluated_here": False,
        },
        "fixed_thresholds": LIMITS,
        "E50_seed2": seed2,
        "support9": support9,
        "harmonic_and_auxiliary_retention": retention,
        "size_consistency": size_consistency,
        "post_selection_available_checks_pass": available_checks_pass,
        "full_composite_deployment_authorized": False,
        "finite_temperature_validation_closed": False,
        "authorization_limit": (
            "even if all available post-selection checks pass, this report cannot authorize "
            "deployment: a matched 8x8 q6 operator is unavailable and no new order-safe "
            "fixed-smearing finite-temperature holdout or final SSCHA/full-EPC spectrum is evaluated"
        ),
        "inputs": {
            **input_records,
            "core_gate": {"path": str(args.gate_json), "sha256": sha256(args.gate_json)},
            "selected_core": {
                "path": str(args.selected_core),
                "sha256": selected["sha256"],
            },
            "training_freeze": {
                "path": str(args.training_dir / "training_freeze.json"),
                "sha256": sha256(args.training_dir / "training_freeze.json"),
            },
            "training_runtime": {
                "path": str(args.training_dir / "training_runtime.json"),
                "sha256": sha256(args.training_dir / "training_runtime.json"),
            },
            "core_gate_script_snapshot": gate["inputs"]["script_snapshot"],
            "post_selection_script_snapshot": {
                "path": str(snapshot),
                "sha256": sha256(snapshot),
            },
            "audited_q6_replay_dependency": {
                "path": str(HERE / "audit_graphene_current_s0_fixed_smearing.py"),
                "sha256": sha256(HERE / "audit_graphene_current_s0_fixed_smearing.py"),
            },
        },
    }
    strict_json(args.output_dir / "post_selection_report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_checkpoint_changed": False,
                "E50_seed2_pass": seed2["fixed_report_gate"]["passes_all"],
                "support9_pass": support9["fixed_report_gate"]["passes_all"],
                "retention_pass": retention["passes_all_available_force_gates"],
                "short_size_pass": size_consistency["short_composite_size_gate_passed"],
                "full_size_status": size_consistency["full_v11_core_q6_size_gate"]["status"],
                "full_composite_deployment_authorized": False,
            },
            indent=2,
            allow_nan=False,
        )
    )
    del base_calculator, core_calculator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
