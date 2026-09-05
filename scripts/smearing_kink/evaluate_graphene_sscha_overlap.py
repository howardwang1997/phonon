#!/usr/bin/env python3
"""Merge and evaluate the frozen 12-point current-S0 SSCHA overlap test.

The DFT structures are selected and frozen before labeling.  This script
checks the two shard summaries against that freeze manifest, projects the
saved current-model and DFT forces onto the folded-K A' mode, applies the
predefined gates, and reports a diagnostic for whether a global rescaling of
the long-range operator could explain any remaining force error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.units import create_units

from audit_graphene_current_s0_fixed_smearing import (
    complex_metrics,
    folded_k_aprime_mode,
    force_metrics,
    load_operator,
    structure_mapping,
)

KB_EV_K = 8.617333262145e-5
RY_TO_EV = float(create_units("2006")["Ry"])


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


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value) -> float:
    return float(np.asarray(value).reshape(()))


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


def energy_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    error = np.asarray(predicted, float) - np.asarray(target, float)
    centered = error - np.mean(error)
    return centered, {
        "RMSE_meV_config": float(np.sqrt(np.mean(centered**2)) * 1000.0),
        "MAE_meV_config": float(np.mean(np.abs(centered)) * 1000.0),
        "max_abs_meV_config": float(np.max(np.abs(centered)) * 1000.0),
    }


def bootstrap_intervals(
    errors: np.ndarray,
    coordinates: np.ndarray,
    predicted_mode_forces: np.ndarray,
    target_mode_forces: np.ndarray,
    predicted_energies: np.ndarray,
    target_energies: np.ndarray,
    temperature_K: float,
    count: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    nframes = len(errors)
    samples: dict[str, list[float]] = {
        "force_component_RMSE_meV_A": [],
        "Aprime_projected_force_RMS_meV_A": [],
        "Aprime_restoring_slope_relative_error_abs": [],
        "centered_energy_RMSE_meV_config": [],
        "importance_weight_ESS_fraction": [],
    }
    for _ in range(count):
        chosen = rng.integers(0, nframes, size=nframes)
        force_error = errors[chosen]
        mode_error = predicted_mode_forces[chosen] - target_mode_forces[chosen]
        q = coordinates[chosen]
        predicted_slope = restoring_slope(q, predicted_mode_forces[chosen])
        target_slope = restoring_slope(q, target_mode_forces[chosen])
        relative_slope_error = abs((predicted_slope - target_slope) / target_slope)
        centered_energy, energy_summary = energy_metrics(
            predicted_energies[chosen], target_energies[chosen]
        )
        samples["force_component_RMSE_meV_A"].append(
            force_metrics(force_error)["RMSE_meV_A"]
        )
        samples["Aprime_projected_force_RMS_meV_A"].append(
            complex_metrics(mode_error)["RMS_meV_A"]
        )
        samples["Aprime_restoring_slope_relative_error_abs"].append(
            relative_slope_error
        )
        samples["centered_energy_RMSE_meV_config"].append(
            energy_summary["RMSE_meV_config"]
        )
        samples["importance_weight_ESS_fraction"].append(
            importance_ess_fraction(centered_energy, temperature_K)
        )
    intervals = {}
    for name, values in samples.items():
        array = np.asarray(values, float)
        intervals[name] = {
            "median": float(np.median(array)),
            "percentile_2_5": float(np.percentile(array, 2.5)),
            "percentile_97_5": float(np.percentile(array, 97.5)),
        }
    return intervals


def validate_shard(
    name: str,
    summary_path: Path,
    xyz_path: Path,
    manifest: dict,
) -> tuple[list, dict]:
    summary = load_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError(f"shard {name} summary is not complete")
    expected_snapshot_hash = manifest["outputs"][f"shard_{name}_snapshots_sha256"]
    if summary["snapshots"]["sha256"] != expected_snapshot_hash:
        raise ValueError(f"shard {name} snapshot hash differs from freeze manifest")
    if summary["output_extxyz"]["sha256"] != sha256(xyz_path):
        raise ValueError(f"shard {name} extxyz hash differs from its summary")
    condition = manifest["condition"]
    if abs(float(summary["lattice_temperature_K"]) - condition["lattice_temperature_K"]) > 1.0e-12:
        raise ValueError(f"shard {name} lattice temperature differs from manifest")
    if abs(float(summary["degauss_Ry"]) - condition["degauss_Ry"]) > 5.0e-10:
        raise ValueError(f"shard {name} degauss differs from manifest")
    frozen_kgrid = [int(value) for value in manifest["fixed_DFT_settings"]["kgrid"]]
    if frozen_kgrid != [int(summary["kgrid"]), int(summary["kgrid"]), 1]:
        raise ValueError(f"shard {name} k grid differs from manifest")
    expected_indices = {int(value) for value in manifest["shards"][name]}
    if {int(value) for value in summary["sscha_indices"]} != expected_indices:
        raise ValueError(f"shard {name} SSCHA indices differ from manifest")
    structures = read(xyz_path, index=":")
    if len(structures) != len(expected_indices):
        raise ValueError(f"shard {name} has the wrong number of structures")
    if {int(atoms.info["sscha_index"]) for atoms in structures} != expected_indices:
        raise ValueError(f"shard {name} extxyz indices differ from manifest")
    return structures, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit-erratum", type=Path, required=True)
    parser.add_argument("--shard-a-summary", type=Path, required=True)
    parser.add_argument("--shard-a-xyz", type=Path, required=True)
    parser.add_argument("--shard-b-summary", type=Path, required=True)
    parser.add_argument("--shard-b-xyz", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--thermal-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    if manifest.get("status") != "frozen_before_R1_DFT":
        raise ValueError("selection manifest was not frozen before DFT")
    unit_erratum = load_json(args.unit_erratum)
    if unit_erratum.get("status") != "deterministic_unit_interpretation_correction":
        raise ValueError("invalid SSCHA unit interpretation erratum")
    if unit_erratum["freeze_manifest_sha256"] != sha256(args.manifest):
        raise ValueError("unit erratum refers to a different freeze manifest")
    if abs(float(unit_erratum["conversion"]["Ry_to_eV"]) - RY_TO_EV) > 1.0e-12:
        raise ValueError("runtime ASE Rydberg differs from the frozen unit correction")
    if unit_erratum.get("frozen_indices_and_snapshot_hashes_changed") is not False:
        raise ValueError("unit correction must not change the frozen test set")
    shard_a, summary_a = validate_shard(
        "A", args.shard_a_summary, args.shard_a_xyz, manifest
    )
    shard_b, summary_b = validate_shard(
        "B", args.shard_b_summary, args.shard_b_xyz, manifest
    )
    by_index = {
        int(atoms.info["sscha_index"]): atoms for atoms in shard_a + shard_b
    }
    frozen_order = [int(record["sscha_index"]) for record in manifest["selected"]]
    if set(by_index) != set(frozen_order) or len(by_index) != len(frozen_order):
        raise ValueError("merged DFT structures differ from the frozen 12-point set")
    structures = [by_index[index] for index in frozen_order]
    selected_manifest = {
        int(record["sscha_index"]): record for record in manifest["selected"]
    }

    force_constants, reference_positions, cell, operator_degauss = load_operator(
        args.operator
    )
    if abs(operator_degauss - manifest["condition"]["degauss_Ry"]) > 5.0e-10:
        raise ValueError("operator and frozen test degauss differ")
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.thermal_result, reference_positions, cell
    )

    predicted_forces = []
    target_forces = []
    long_forces = []
    displacements = []
    mappings = []
    coordinates = []
    predicted_mode_forces = []
    target_mode_forces = []
    predicted_energies = []
    target_energies = []
    groups = []
    records = []
    for structure in structures:
        sscha_index = int(structure.info["sscha_index"])
        frozen = selected_manifest[sscha_index]
        group = str(structure.info["selection_group"])
        if group != frozen["selection_group"]:
            raise ValueError(f"selection group differs for SSCHA {sscha_index}")
        if abs(float(structure.info["degauss_Ry"]) - operator_degauss) > 5.0e-10:
            raise ValueError(f"degauss differs for SSCHA {sscha_index}")
        mapping, displacement = structure_mapping(
            structure, reference_positions, cell
        )
        reordered_fc = force_constants[mapping][:, mapping]
        long_force = -np.einsum("ijab,jb->ia", reordered_fc, displacement)
        saved_force_Ry_A = np.asarray(structure.arrays["MLIP_forces"], float)
        predicted_force = saved_force_Ry_A * RY_TO_EV
        target_force = np.asarray(structure.arrays["REF_forces"], float)
        expected_saved_force_rms = float(
            frozen["features"]["MLIP_force_RMS_eV_A"]
        )
        observed_saved_force_rms = float(np.sqrt(np.mean(saved_force_Ry_A**2)))
        if abs(observed_saved_force_rms - expected_saved_force_rms) > 2.0e-9:
            raise ValueError(f"saved current-model forces changed for SSCHA {sscha_index}")
        mode_structure = mode[mapping]
        mode_flat = mode_structure.reshape(-1)
        coordinate = np.vdot(mode_flat, displacement.reshape(-1))
        predicted_mode_force = np.vdot(mode_flat, predicted_force.reshape(-1))
        target_mode_force = np.vdot(mode_flat, target_force.reshape(-1))
        saved_energy_Ry = float(structure.info["MLIP_energy"])
        predicted_energy = saved_energy_Ry * RY_TO_EV
        target_energy = float(structure.info["REF_energy"])
        error = predicted_force - target_force
        max_flat = int(np.argmax(np.abs(error)))
        max_atom, max_component = np.unravel_index(max_flat, error.shape)

        predicted_forces.append(predicted_force)
        target_forces.append(target_force)
        long_forces.append(long_force)
        displacements.append(displacement)
        mappings.append(mapping)
        coordinates.append(coordinate)
        predicted_mode_forces.append(predicted_mode_force)
        target_mode_forces.append(target_mode_force)
        predicted_energies.append(predicted_energy)
        target_energies.append(target_energy)
        groups.append(group)
        records.append(
            {
                "sscha_index": sscha_index,
                "selection_group": group,
                "shard": frozen["shard"],
                "force_RMSE_meV_A": force_metrics(error)["RMSE_meV_A"],
                "force_MAE_meV_A": force_metrics(error)["MAE_meV_A"],
                "force_max_abs_meV_A": force_metrics(error)["max_abs_meV_A"],
                "max_error_atom_index": int(max_atom),
                "max_error_component": "xyz"[int(max_component)],
                "Aprime_coordinate_abs_A": float(abs(coordinate)),
                "Aprime_force_error_abs_meV_A": float(
                    abs(predicted_mode_force - target_mode_force) * 1000.0
                ),
                "MLIP_energy_eV": predicted_energy,
                "DFT_energy_eV": target_energy,
            }
        )
        structure.arrays["SSCHA_saved_forces_Ry_A"] = saved_force_Ry_A
        structure.arrays["MLIP_forces"] = predicted_force
        structure.info["SSCHA_saved_energy_Ry"] = saved_energy_Ry
        structure.info["MLIP_energy"] = predicted_energy
        structure.info["SSCHA_unit_conversion_Ry_to_eV"] = RY_TO_EV

    predicted_forces = np.asarray(predicted_forces)
    target_forces = np.asarray(target_forces)
    force_errors = predicted_forces - target_forces
    long_forces = np.asarray(long_forces)
    displacements = np.asarray(displacements)
    mappings = np.asarray(mappings)
    coordinates = np.asarray(coordinates)
    predicted_mode_forces = np.asarray(predicted_mode_forces)
    target_mode_forces = np.asarray(target_mode_forces)
    predicted_energies = np.asarray(predicted_energies)
    target_energies = np.asarray(target_energies)
    groups = np.asarray(groups)

    predicted_slope = restoring_slope(coordinates, predicted_mode_forces)
    target_slope = restoring_slope(coordinates, target_mode_forces)
    relative_slope_error = (predicted_slope - target_slope) / target_slope
    centered_energy_error, energy_summary = energy_metrics(
        predicted_energies, target_energies
    )
    for record, value in zip(records, centered_energy_error):
        record["centered_energy_error_meV_config"] = float(value * 1000.0)
    temperature_K = float(manifest["condition"]["lattice_temperature_K"])
    ess_fraction = importance_ess_fraction(centered_energy_error, temperature_K)

    flattened_error = force_errors.reshape(-1)
    flattened_long = long_forces.reshape(-1)
    extra_lr_scalar = -float(
        flattened_error @ flattened_long / (flattened_long @ flattened_long)
    )
    rescaled_error = force_errors + extra_lr_scalar * long_forces
    removed_fraction = float(
        1.0 - np.sum(rescaled_error**2) / np.sum(force_errors**2)
    )

    thresholds = manifest["fixed_acceptance_thresholds"]
    observed = {
        "force_component_RMSE_meV_A": force_metrics(force_errors)["RMSE_meV_A"],
        "force_component_max_abs_meV_A": force_metrics(force_errors)["max_abs_meV_A"],
        "Aprime_projected_force_RMS_meV_A": complex_metrics(
            predicted_mode_forces - target_mode_forces
        )["RMS_meV_A"],
        "Aprime_restoring_slope_relative_error": abs(relative_slope_error),
        "centered_energy_RMSE_meV_config": energy_summary["RMSE_meV_config"],
        "importance_weight_ESS_fraction": ess_fraction,
    }
    gate = {}
    for name, limit in thresholds.items():
        if name == "importance_weight_ESS_fraction":
            passed = observed[name] >= float(limit)
            comparison = ">="
        else:
            passed = observed[name] <= float(limit)
            comparison = "<="
        gate[name] = {
            "observed": observed[name],
            "comparison": comparison,
            "threshold": float(limit),
            "pass": bool(passed),
        }
    all_pass = all(item["pass"] for item in gate.values())

    group_metrics = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        group_metrics[group] = {
            "n_structures": int(np.sum(mask)),
            "force_error": force_metrics(force_errors[mask]),
            "Aprime_projected_force_error": complex_metrics(
                predicted_mode_forces[mask] - target_mode_forces[mask]
            ),
        }

    if all_pass:
        decision = (
            "R1 passes all frozen gates: keep both current S0 and the q6 long-range "
            "operator fixed, then run the finite-temperature dispersion closure test."
        )
    elif removed_fraction < 0.05 and gate[
        "Aprime_restoring_slope_relative_error"
    ]["pass"]:
        decision = (
            "R1 fails at least one frozen gate, while a global long-range rescale "
            "removes less than 5% of force squared error and the A-prime slope gate "
            "passes: keep the q6 operator fixed and update the finite-temperature "
            "short-range residual using only the frozen overlap labels."
        )
    else:
        decision = (
            "R1 does not support a short-range-only update: diagnose the q-dependent "
            "long-range kernel and mode projection before adding labels or retraining."
        )

    bootstrap = bootstrap_intervals(
        force_errors,
        coordinates,
        predicted_mode_forces,
        target_mode_forces,
        predicted_energies,
        target_energies,
        temperature_K,
        args.bootstrap,
        args.seed,
    )
    summary = {
        "status": "R1_frozen_overlap_evaluated",
        "scope": "current S0 + fixed-smearing q6 operator on 12 post-freeze SSCHA DFT labels",
        "n_structures": len(structures),
        "condition": manifest["condition"],
        "DFT_settings": manifest["fixed_DFT_settings"],
        "SSCHA_saved_value_units": {
            "raw_force": "Ry/angstrom",
            "raw_energy": "Ry",
            "converted_force": "eV/angstrom",
            "converted_energy": "eV",
            "Ry_to_eV": RY_TO_EV,
        },
        "gate": gate,
        "all_frozen_gates_pass": all_pass,
        "force_error": force_metrics(force_errors),
        "force_error_by_selection_group": group_metrics,
        "Aprime_projection": {
            "force_error": complex_metrics(
                predicted_mode_forces - target_mode_forces
            ),
            "MLIP_restoring_slope_eV_A2": predicted_slope,
            "DFT_restoring_slope_eV_A2": target_slope,
            "relative_slope_error": relative_slope_error,
            "mode": mode_provenance,
        },
        "energy_error_after_global_offset": energy_summary,
        "importance_reweighting": {
            "ESS_fraction": ess_fraction,
            "temperature_K": temperature_K,
            "note": "diagnostic on the deliberately stratified 12-point overlap set",
        },
        "long_range_global_rescale_diagnostic": {
            "optimal_extra_scalar": extra_lr_scalar,
            "force_RMSE_after_rescale_meV_A": force_metrics(rescaled_error)[
                "RMSE_meV_A"
            ],
            "squared_error_fraction_removed": removed_fraction,
            "interpretation": "diagnostic only; no DFT point was used to alter the frozen model",
        },
        "bootstrap_95_percent_intervals": {
            "n_resamples": args.bootstrap,
            "seed": args.seed,
            "intervals": bootstrap,
        },
        "decision": decision,
        "inputs": {
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "unit_erratum": {
                "path": str(args.unit_erratum),
                "sha256": sha256(args.unit_erratum),
            },
            "shard_A_summary": {
                "path": str(args.shard_a_summary),
                "sha256": sha256(args.shard_a_summary),
                "wall_seconds": float(summary_a["total_wall_seconds"]),
            },
            "shard_B_summary": {
                "path": str(args.shard_b_summary),
                "sha256": sha256(args.shard_b_summary),
                "wall_seconds": float(summary_b["total_wall_seconds"]),
            },
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {
                "path": str(args.background),
                "sha256": sha256(args.background),
            },
            "thermal_result": {
                "path": str(args.thermal_result),
                "sha256": sha256(args.thermal_result),
            },
        },
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "R1_overlap_merged12.extxyz"
    temporary_merged = merged_path.with_name(merged_path.name + ".tmp")
    write(temporary_merged, structures, format="extxyz")
    os.replace(temporary_merged, merged_path)
    atomic_npz(
        args.output_dir / "R1_overlap_predictions.npz",
        sscha_indices=np.asarray(frozen_order, int),
        selection_groups=groups,
        MLIP_forces_eV_A=predicted_forces,
        DFT_forces_eV_A=target_forces,
        force_errors_eV_A=force_errors,
        long_range_forces_eV_A=long_forces,
        displacement_A=displacements,
        structure_to_operator_mapping=mappings,
        Aprime_mode_operator_order=mode,
        Aprime_coordinate_A=coordinates,
        Aprime_MLIP_force_eV_A=predicted_mode_forces,
        Aprime_DFT_force_eV_A=target_mode_forces,
        MLIP_energy_eV=predicted_energies,
        DFT_energy_eV=target_energies,
        centered_energy_error_eV=centered_energy_error,
    )
    csv_path = args.output_dir / "R1_overlap_records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    atomic_json(args.output_dir / "R1_overlap_gate_summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": gate,
                "all_frozen_gates_pass": all_pass,
                "decision": decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
