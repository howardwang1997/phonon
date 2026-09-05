#!/usr/bin/env python3
"""Evaluate graphene R2C short-range candidates without changing long range.

The nine corrected-X0-supported DFT geometries are development/training data,
not a final holdout.  Candidate selection additionally protects the existing
300/450/600 K development tests and the frozen harmonic replay.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    complex_metrics,
    folded_k_aprime_mode,
    force_metrics,
    load_operator,
    structure_mapping,
)
SUPPORT_LIMITS = {
    "force_component_RMSE_meV_A": 30.0,
    "force_component_max_abs_meV_A": 200.0,
    "Aprime_projected_force_RMS_meV_A": 15.0,
    "Aprime_restoring_slope_relative_error": 0.05,
    "centered_energy_RMSE_meV_config": 19.4,
    "importance_weight_ESS_fraction": 0.3,
}
KB_EV_K = 8.617333262145e-5


def importance_ess_fraction(energy_error_eV: np.ndarray, temperature_K: float) -> float:
    values = np.asarray(energy_error_eV, float)
    centered = values - np.mean(values)
    log_weights = -centered / (KB_EV_K * temperature_K)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return float((np.sum(weights) ** 2 / np.sum(weights**2)) / len(weights))


def restoring_slope(coordinates: np.ndarray, projected_forces: np.ndarray) -> float:
    denominator = float(np.vdot(coordinates, coordinates).real)
    if denominator <= 0.0:
        raise ValueError("all folded-K A-prime coordinates are zero")
    return -float(np.vdot(coordinates, projected_forces).real / denominator)


def labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("candidate/thermal paths use LABEL=/path")
    label, path = specification.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("candidate/thermal paths use LABEL=/path")
    return label, Path(path)


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


def energy_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    error = np.asarray(predicted, float) - np.asarray(target, float)
    offset = float(np.mean(error))
    centered = error - offset
    return centered, {
        "global_offset_eV": offset,
        "RMSE_meV_config": float(np.sqrt(np.mean(centered**2)) * 1000.0),
        "MAE_meV_config": float(np.mean(np.abs(centered)) * 1000.0),
        "max_abs_meV_config": float(np.max(np.abs(centered)) * 1000.0),
    }


def target_arrays(structure, base_force: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    long_force = np.asarray(
        structure.arrays.get("LONG_RANGE_forces", np.zeros_like(base_force)), float
    )
    short_force = np.asarray(
        structure.arrays.get("SHORT_RANGE_TARGET_forces", structure.arrays["REF_forces"]),
        float,
    )
    total_force = np.asarray(
        structure.arrays.get("TOTAL_forces", base_force + long_force + short_force),
        float,
    )
    return total_force, long_force


def target_delta_energy(structure) -> float:
    return float(
        structure.info.get("SHORT_RANGE_TARGET_energy", structure.info["REF_energy"])
    )


def predictions(structures: list, calculator: MACECalculator) -> tuple[np.ndarray, np.ndarray]:
    energies = []
    forces = []
    for source in structures:
        atoms = source.copy()
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), float))
    return np.asarray(energies), np.asarray(forces)


def ordinary_dataset_metrics(
    structures: list,
    base_forces: np.ndarray,
    delta_energies: np.ndarray,
    delta_forces: np.ndarray,
) -> dict:
    total_errors = []
    target_energies = []
    for index, structure in enumerate(structures):
        target_total, long_force = target_arrays(structure, base_forces[index])
        predicted_total = base_forces[index] + long_force + delta_forces[index]
        total_errors.append(predicted_total - target_total)
        target_energies.append(target_delta_energy(structure))
    centered, energy = energy_metrics(delta_energies, np.asarray(target_energies))
    return {
        "n_structures": len(structures),
        "total_force_error": force_metrics(np.asarray(total_errors)),
        "delta_energy_error_after_global_offset": energy,
        "centered_delta_energy_error_eV": centered.tolist(),
    }


def support_metrics(
    structures: list,
    base_forces: np.ndarray,
    delta_energies: np.ndarray,
    delta_forces: np.ndarray,
    mode: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
) -> dict:
    errors = []
    coordinates = []
    predicted_modes = []
    target_modes = []
    target_energies = []
    records = []
    for index, structure in enumerate(structures):
        mapping, displacement = structure_mapping(structure, reference, cell)
        target_total, long_force = target_arrays(structure, base_forces[index])
        predicted_total = base_forces[index] + long_force + delta_forces[index]
        error = predicted_total - target_total
        mode_structure = mode[mapping]
        mode_flat = mode_structure.reshape(-1)
        coordinate = np.vdot(mode_flat, displacement.reshape(-1))
        predicted_mode = np.vdot(mode_flat, predicted_total.reshape(-1))
        target_mode = np.vdot(mode_flat, target_total.reshape(-1))
        errors.append(error)
        coordinates.append(coordinate)
        predicted_modes.append(predicted_mode)
        target_modes.append(target_mode)
        target_energies.append(target_delta_energy(structure))
        records.append(
            {
                "sscha_index": int(structure.info["sscha_index"]),
                "support_level": str(structure.info["corrected_X0_support_level"]),
                "force_RMSE_meV_A": force_metrics(error[None])["RMSE_meV_A"],
                "force_max_abs_meV_A": force_metrics(error[None])["max_abs_meV_A"],
                "Aprime_coordinate_abs_A": float(abs(coordinate)),
                "Aprime_force_error_abs_meV_A": float(
                    abs(predicted_mode - target_mode) * 1000.0
                ),
            }
        )
    errors = np.asarray(errors)
    coordinates = np.asarray(coordinates)
    predicted_modes = np.asarray(predicted_modes)
    target_modes = np.asarray(target_modes)
    centered_energy, energy = energy_metrics(
        delta_energies, np.asarray(target_energies)
    )
    predicted_slope = restoring_slope(coordinates, predicted_modes)
    target_slope = restoring_slope(coordinates, target_modes)
    relative_slope_error = (predicted_slope - target_slope) / target_slope
    observed = {
        "force_component_RMSE_meV_A": force_metrics(errors)["RMSE_meV_A"],
        "force_component_max_abs_meV_A": force_metrics(errors)["max_abs_meV_A"],
        "Aprime_projected_force_RMS_meV_A": complex_metrics(
            predicted_modes - target_modes
        )["RMS_meV_A"],
        "Aprime_restoring_slope_relative_error": abs(relative_slope_error),
        "centered_energy_RMSE_meV_config": energy["RMSE_meV_config"],
        "importance_weight_ESS_fraction": importance_ess_fraction(
            centered_energy, 450.0
        ),
    }
    gates = {}
    ratios = {}
    for name, limit in SUPPORT_LIMITS.items():
        if name == "importance_weight_ESS_fraction":
            passed = observed[name] >= limit
            ratio = limit / max(observed[name], 1.0e-12)
        else:
            passed = observed[name] <= limit
            ratio = observed[name] / limit
        gates[name] = {"observed": observed[name], "threshold": limit, "pass": passed}
        ratios[f"support/{name}"] = ratio
    return {
        "n_structures": len(structures),
        "force_error": force_metrics(errors),
        "Aprime_projection": {
            "force_error": complex_metrics(predicted_modes - target_modes),
            "candidate_restoring_slope_eV_A2": predicted_slope,
            "DFT_restoring_slope_eV_A2": target_slope,
            "relative_slope_error": relative_slope_error,
        },
        "delta_energy_error_after_global_offset": energy,
        "importance_weight_ESS_fraction": importance_ess_fraction(
            centered_energy, 450.0
        ),
        "fixed_gate": gates,
        "normalized_gate_ratios": ratios,
        "records": records,
    }


def evaluate_candidate(
    model: Path,
    device: str,
    support: list,
    support_base: np.ndarray,
    thermal: dict[str, list],
    thermal_base: dict[str, np.ndarray],
    harmonic: list,
    harmonic_base: np.ndarray,
    harmonic_limit: float,
    mode: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
) -> dict:
    calculator = MACECalculator(
        model_paths=str(model), device=device, default_dtype="float32"
    )
    support_energy, support_force = predictions(support, calculator)
    support_result = support_metrics(
        support,
        support_base,
        support_energy,
        support_force,
        mode,
        reference,
        cell,
    )
    thermal_result = {}
    ratios = dict(support_result["normalized_gate_ratios"])
    for label, structures in thermal.items():
        energy, force = predictions(structures, calculator)
        metrics = ordinary_dataset_metrics(
            structures, thermal_base[label], energy, force
        )
        thermal_result[label] = metrics
        ratios[f"{label}/force_RMSE"] = (
            metrics["total_force_error"]["RMSE_meV_A"] / 30.0
        )
        ratios[f"{label}/force_max_abs"] = (
            metrics["total_force_error"]["max_abs_meV_A"] / 200.0
        )
    harmonic_energy, harmonic_force = predictions(harmonic, calculator)
    harmonic_result = ordinary_dataset_metrics(
        harmonic, harmonic_base, harmonic_energy, harmonic_force
    )
    ratios["harmonic/force_RMSE"] = (
        harmonic_result["total_force_error"]["RMSE_meV_A"] / harmonic_limit
    )
    return {
        "model": str(model),
        "model_sha256": sha256(model),
        "support9_development_fit": support_result,
        "thermal_development": thermal_result,
        "harmonic_replay": harmonic_result,
        "normalized_gate_ratios": ratios,
        "normalized_worst_gate_score": float(max(ratios.values())),
        "passes_all_development_gates": bool(all(value <= 1.0 for value in ratios.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=labelled_path, required=True)
    parser.add_argument("--support-data", type=Path, required=True)
    parser.add_argument("--thermal", action="append", type=labelled_path, required=True)
    parser.add_argument("--harmonic", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = dict(args.candidate)
    thermal_paths = dict(args.thermal)
    if len(candidates) != len(args.candidate) or len(thermal_paths) != len(args.thermal):
        raise ValueError("candidate and thermal labels must be unique")
    support = read(args.support_data, index=":")
    thermal = {label: read(path, index=":") for label, path in thermal_paths.items()}
    harmonic = read(args.harmonic, index=":")
    if len(support) != 9:
        raise ValueError("support development set must contain nine structures")

    force_constants, reference, cell, operator_degauss = load_operator(args.operator)
    del force_constants
    if abs(operator_degauss - 0.0019000869380739254) > 5.0e-10:
        raise ValueError("R2C first condition requires the frozen T300 q6 operator")
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )

    base = MACECalculator(
        model_paths=str(args.base_model), device=args.device, default_dtype="float32"
    )
    _, support_base = predictions(support, base)
    thermal_base = {
        label: predictions(structures, base)[1]
        for label, structures in thermal.items()
    }
    _, harmonic_base = predictions(harmonic, base)
    base_harmonic_errors = []
    for index, structure in enumerate(harmonic):
        target_total, long_force = target_arrays(structure, harmonic_base[index])
        base_harmonic_errors.append(harmonic_base[index] + long_force - target_total)
    base_harmonic = force_metrics(np.asarray(base_harmonic_errors))
    harmonic_limit = 1.25 * base_harmonic["RMSE_meV_A"]
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {}
    for label, model in candidates.items():
        results[label] = evaluate_candidate(
            model,
            args.device,
            support,
            support_base,
            thermal,
            thermal_base,
            harmonic,
            harmonic_base,
            harmonic_limit,
            mode,
            reference,
            cell,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ranked = sorted(results, key=lambda label: results[label]["normalized_worst_gate_score"])
    summary = {
        "status": "R2C_candidate_development_evaluation_complete",
        "finite_temperature_validation_closed": False,
        "reason_not_final": (
            "the nine corrected-X0-supported DFT points are included in R2C training; "
            "a new order-safe matched holdout is still required"
        ),
        "fixed_condition": {
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": operator_degauss,
        },
        "fixed_limits": {
            "support9": SUPPORT_LIMITS,
            "each_thermal_force_RMSE_meV_A": 30.0,
            "each_thermal_force_max_abs_meV_A": 200.0,
            "harmonic_replay_RMSE_relative_to_frozen_base": 1.25,
            "harmonic_replay_RMSE_meV_A": harmonic_limit,
        },
        "frozen_base_harmonic_force": base_harmonic,
        "Aprime_mode": mode_provenance,
        "ranking": ranked,
        "best_development_candidate": ranked[0],
        "candidates": results,
        "inputs": {
            "base_model": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
            "support_data": {"path": str(args.support_data), "sha256": sha256(args.support_data)},
            "thermal": {
                label: {"path": str(path), "sha256": sha256(path)}
                for label, path in thermal_paths.items()
            },
            "harmonic": {"path": str(args.harmonic), "sha256": sha256(args.harmonic)},
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {"path": str(args.background), "sha256": sha256(args.background)},
            "corrected_result": {"path": str(args.corrected_result), "sha256": sha256(args.corrected_result)},
        },
    }
    atomic_json(args.output, summary)
    concise = {
        label: {
            "passes": results[label]["passes_all_development_gates"],
            "worst_score": results[label]["normalized_worst_gate_score"],
            "support_gate": results[label]["support9_development_fit"]["fixed_gate"],
            "thermal_force": {
                name: metrics["total_force_error"]
                for name, metrics in results[label]["thermal_development"].items()
            },
            "harmonic_force": results[label]["harmonic_replay"]["total_force_error"],
        }
        for label in ranked
    }
    print(json.dumps({"ranking": ranked, "candidates": concise}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
