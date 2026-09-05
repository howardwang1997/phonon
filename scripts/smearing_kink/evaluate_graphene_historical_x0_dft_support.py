#!/usr/bin/env python3
"""Evaluate completed DFT labels that came from the invalid historical X0.

The current model is taken from an exact frozen-model replay, not from the
historically corrupted SSCHA saved forces.  A geometry-only support record,
fixed before all labels completed, determines which structures overlap the
corrected order-safe X0 ensemble.  Results are diagnostic and cannot by
themselves close the finite-temperature validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from ase.io import write
from ase.units import create_units

from audit_graphene_current_s0_fixed_smearing import (
    complex_metrics,
    folded_k_aprime_mode,
    force_metrics,
    load_operator,
    structure_mapping,
)
from evaluate_graphene_sscha_overlap import (
    atomic_json,
    atomic_npz,
    bootstrap_intervals,
    energy_metrics,
    importance_ess_fraction,
    load_json,
    restoring_slope,
    sha256,
    validate_shard,
)

RY_TO_EV = float(create_units("2006")["Ry"])


def metric_block(
    mask: np.ndarray,
    force_errors: np.ndarray,
    coordinates: np.ndarray,
    predicted_mode_forces: np.ndarray,
    target_mode_forces: np.ndarray,
    predicted_energies: np.ndarray,
    target_energies: np.ndarray,
    long_forces: np.ndarray,
    temperature_K: float,
) -> dict:
    errors = force_errors[mask]
    q = coordinates[mask]
    predicted_mode = predicted_mode_forces[mask]
    target_mode = target_mode_forces[mask]
    predicted_slope = restoring_slope(q, predicted_mode)
    target_slope = restoring_slope(q, target_mode)
    relative_slope_error = (predicted_slope - target_slope) / target_slope
    centered_energy, energy_summary = energy_metrics(
        predicted_energies[mask], target_energies[mask]
    )
    long_values = long_forces[mask]
    flattened_error = errors.reshape(-1)
    flattened_long = long_values.reshape(-1)
    extra_lr_scalar = -float(
        flattened_error @ flattened_long / (flattened_long @ flattened_long)
    )
    rescaled_error = errors + extra_lr_scalar * long_values
    squared_error = float(np.sum(errors**2))
    removed_fraction = (
        float(1.0 - np.sum(rescaled_error**2) / squared_error)
        if squared_error > 0.0
        else 0.0
    )
    return {
        "n_structures": int(np.sum(mask)),
        "force_error": force_metrics(errors),
        "Aprime_projection": {
            "force_error": complex_metrics(predicted_mode - target_mode),
            "MLIP_restoring_slope_eV_A2": predicted_slope,
            "DFT_restoring_slope_eV_A2": target_slope,
            "relative_slope_error": relative_slope_error,
        },
        "energy_error_after_global_offset": energy_summary,
        "importance_weight_ESS_fraction": importance_ess_fraction(
            centered_energy, temperature_K
        ),
        "long_range_global_rescale_diagnostic": {
            "optimal_extra_scalar": extra_lr_scalar,
            "force_RMSE_after_rescale_meV_A": force_metrics(rescaled_error)[
                "RMSE_meV_A"
            ],
            "squared_error_fraction_removed": removed_fraction,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit-erratum", type=Path, required=True)
    parser.add_argument("--atom-order-diagnostic", type=Path, required=True)
    parser.add_argument("--geometry-support", type=Path, required=True)
    parser.add_argument("--model-replay", type=Path, required=True)
    parser.add_argument("--model-replay-summary", type=Path, required=True)
    parser.add_argument("--shard-a-summary", type=Path, required=True)
    parser.add_argument("--shard-a-xyz", type=Path, required=True)
    parser.add_argument("--shard-b-summary", type=Path, required=True)
    parser.add_argument("--shard-b-xyz", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    manifest_hash = sha256(args.manifest)
    unit_erratum = load_json(args.unit_erratum)
    atom_diagnostic = load_json(args.atom_order_diagnostic)
    support = load_json(args.geometry_support)
    replay_summary = load_json(args.model_replay_summary)
    if unit_erratum["freeze_manifest_sha256"] != manifest_hash:
        raise ValueError("unit erratum refers to another manifest")
    if atom_diagnostic.get("status") != "historical_X0_atom_order_bug_reproduced":
        raise ValueError("historical atom-order bug was not reproduced")
    if atom_diagnostic["inputs"]["freeze_manifest"]["sha256"] != manifest_hash:
        raise ValueError("atom-order diagnostic refers to another manifest")
    if support.get("status") != "geometry_only_support_frozen":
        raise ValueError("geometry support was not frozen")
    if support.get("DFT_energy_or_force_read") is not False:
        raise ValueError("geometry support classification read DFT values")
    if replay_summary.get("status") != "current_S0_fixed_smearing_audit_complete":
        raise ValueError("exact frozen-model replay is incomplete")
    if (
        replay_summary["inputs"]["delta_model"]["sha256"]
        != manifest["inputs"]["template_model"]["sha256"]
    ):
        raise ValueError("model replay used another S0 delta model")
    if replay_summary["inputs"]["operator"]["sha256"] != manifest["inputs"]["operator"]["sha256"]:
        raise ValueError("model replay used another q6 operator")

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
    if set(by_index) != set(frozen_order):
        raise ValueError("DFT structures differ from the frozen set")
    structures = [by_index[index] for index in frozen_order]

    with np.load(args.model_replay, allow_pickle=False) as data:
        replay = {key: np.asarray(data[key]) for key in data.files}
    replay_indices = np.asarray(replay["snapshot_index"], int).tolist()
    if replay_indices != frozen_order:
        raise ValueError("model replay order differs from the frozen set")
    if replay["predicted_total_forces_eV_A"].shape != (12, 72, 3):
        raise ValueError("model replay has the wrong force shape")

    support_records = {
        int(record["sscha_index"]): record for record in support["records"]
    }
    if set(support_records) != set(frozen_order):
        raise ValueError("geometry support record differs from the frozen set")
    support_levels = np.asarray(
        [support_records[index]["corrected_X0_support_level"] for index in frozen_order]
    )
    in_support = support_levels != "outside_corrected_X0_support"

    force_constants, reference, cell, operator_degauss = load_operator(args.operator)
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )
    predicted_forces = []
    target_forces = []
    historical_saved_forces = []
    long_forces = []
    displacements = []
    mappings = []
    coordinates = []
    predicted_mode_forces = []
    target_mode_forces = []
    predicted_energies = []
    target_energies = []
    records = []
    for local_index, structure in enumerate(structures):
        sscha_index = frozen_order[local_index]
        mapping, displacement = structure_mapping(structure, reference, cell)
        reordered_fc = force_constants[mapping][:, mapping]
        long_force = -np.einsum("ijab,jb->ia", reordered_fc, displacement)
        replay_long = np.asarray(replay["long_range_forces_eV_A"][local_index], float)
        if float(np.max(np.abs(long_force - replay_long))) > 2.0e-9:
            raise ValueError(f"long-range replay changed for SSCHA {sscha_index}")
        predicted_force = np.asarray(
            replay["predicted_total_forces_eV_A"][local_index], float
        )
        target_force = np.asarray(structure.arrays["REF_forces"], float)
        historical_saved_force = (
            np.asarray(structure.arrays["MLIP_forces"], float) * RY_TO_EV
        )
        mode_structure = mode[mapping]
        mode_flat = mode_structure.reshape(-1)
        coordinate = np.vdot(mode_flat, displacement.reshape(-1))
        predicted_mode = np.vdot(mode_flat, predicted_force.reshape(-1))
        target_mode = np.vdot(mode_flat, target_force.reshape(-1))
        predicted_energy = float(replay["predicted_energy_eV"][local_index])
        target_energy = float(structure.info["REF_energy"])
        error = predicted_force - target_force
        maximum_flat = int(np.argmax(np.abs(error)))
        max_atom, max_component = np.unravel_index(maximum_flat, error.shape)

        predicted_forces.append(predicted_force)
        target_forces.append(target_force)
        historical_saved_forces.append(historical_saved_force)
        long_forces.append(long_force)
        displacements.append(displacement)
        mappings.append(mapping)
        coordinates.append(coordinate)
        predicted_mode_forces.append(predicted_mode)
        target_mode_forces.append(target_mode)
        predicted_energies.append(predicted_energy)
        target_energies.append(target_energy)
        records.append(
            {
                "sscha_index": sscha_index,
                "selection_group": str(structure.info["selection_group"]),
                "corrected_X0_support_level": str(support_levels[local_index]),
                "included_in_corrected_support_diagnostic": bool(in_support[local_index]),
                "force_RMSE_meV_A": force_metrics(error)["RMSE_meV_A"],
                "force_MAE_meV_A": force_metrics(error)["MAE_meV_A"],
                "force_max_abs_meV_A": force_metrics(error)["max_abs_meV_A"],
                "max_error_atom_index": int(max_atom),
                "max_error_component": "xyz"[int(max_component)],
                "Aprime_coordinate_abs_A": float(abs(coordinate)),
                "Aprime_force_error_abs_meV_A": float(
                    abs(predicted_mode - target_mode) * 1000.0
                ),
                "MLIP_energy_eV": predicted_energy,
                "DFT_energy_eV": target_energy,
            }
        )
        structure.arrays["historical_X0_saved_forces_Ry_A"] = np.asarray(
            structure.arrays["MLIP_forces"], float
        )
        structure.arrays["MLIP_forces"] = predicted_force
        structure.info["historical_X0_saved_energy_Ry"] = float(
            structure.info["MLIP_energy"]
        )
        structure.info["MLIP_energy"] = predicted_energy
        structure.info["corrected_X0_support_level"] = str(
            support_levels[local_index]
        )

    predicted_forces = np.asarray(predicted_forces)
    target_forces = np.asarray(target_forces)
    historical_saved_forces = np.asarray(historical_saved_forces)
    force_errors = predicted_forces - target_forces
    long_forces = np.asarray(long_forces)
    displacements = np.asarray(displacements)
    mappings = np.asarray(mappings)
    coordinates = np.asarray(coordinates)
    predicted_mode_forces = np.asarray(predicted_mode_forces)
    target_mode_forces = np.asarray(target_mode_forces)
    predicted_energies = np.asarray(predicted_energies)
    target_energies = np.asarray(target_energies)
    temperature_K = float(manifest["condition"]["lattice_temperature_K"])

    all_metrics = metric_block(
        np.ones(12, bool), force_errors, coordinates, predicted_mode_forces,
        target_mode_forces, predicted_energies, target_energies, long_forces,
        temperature_K,
    )
    support_metrics = metric_block(
        in_support, force_errors, coordinates, predicted_mode_forces,
        target_mode_forces, predicted_energies, target_energies, long_forces,
        temperature_K,
    )
    outside_metrics = metric_block(
        ~in_support, force_errors, coordinates, predicted_mode_forces,
        target_mode_forces, predicted_energies, target_energies, long_forces,
        temperature_K,
    )

    thresholds = manifest["fixed_acceptance_thresholds"]
    support_observed = {
        "force_component_RMSE_meV_A": support_metrics["force_error"]["RMSE_meV_A"],
        "force_component_max_abs_meV_A": support_metrics["force_error"]["max_abs_meV_A"],
        "Aprime_projected_force_RMS_meV_A": support_metrics["Aprime_projection"]["force_error"]["RMS_meV_A"],
        "Aprime_restoring_slope_relative_error": abs(
            support_metrics["Aprime_projection"]["relative_slope_error"]
        ),
        "centered_energy_RMSE_meV_config": support_metrics["energy_error_after_global_offset"]["RMSE_meV_config"],
        "importance_weight_ESS_fraction": support_metrics["importance_weight_ESS_fraction"],
    }
    provisional_gate = {}
    for name, limit in thresholds.items():
        passed = (
            support_observed[name] >= float(limit)
            if name == "importance_weight_ESS_fraction"
            else support_observed[name] <= float(limit)
        )
        provisional_gate[name] = {
            "observed": support_observed[name],
            "threshold": float(limit),
            "pass": bool(passed),
            "scope": "geometry-support diagnostic; not final finite-temperature acceptance",
        }

    force_shape_pass = all(
        provisional_gate[name]["pass"]
        for name in (
            "force_component_RMSE_meV_A",
            "force_component_max_abs_meV_A",
            "Aprime_projected_force_RMS_meV_A",
            "Aprime_restoring_slope_relative_error",
        )
    )
    energy_pass = provisional_gate["centered_energy_RMSE_meV_config"]["pass"]
    lr_removed = support_metrics["long_range_global_rescale_diagnostic"][
        "squared_error_fraction_removed"
    ]
    if force_shape_pass and energy_pass:
        decision = (
            "The nine geometry-supported historical labels show no need to refit either "
            "model component. Select at most three replacement points from corrected X0 "
            "and complete the originally intended 12-point order-safe screen."
        )
    elif lr_removed < 0.05 and provisional_gate[
        "Aprime_restoring_slope_relative_error"
    ]["pass"]:
        decision = (
            "The geometry-supported diagnostic fails, while global q6 rescaling explains "
            "less than 5% of force squared error and the A-prime slope gate passes. Keep "
            "the long-range kernel fixed and repair the finite-temperature short-range residual."
        )
    else:
        decision = (
            "The geometry-supported diagnostic requires a q-dependent/operator audit before "
            "short-range retraining or additional DFT labels."
        )

    bootstrap = bootstrap_intervals(
        force_errors[in_support], coordinates[in_support],
        predicted_mode_forces[in_support], target_mode_forces[in_support],
        predicted_energies[in_support], target_energies[in_support],
        temperature_K, args.bootstrap, args.seed,
    )
    summary = {
        "status": "historical_X0_DFT_support_diagnostic_complete",
        "finite_temperature_validation_closed": False,
        "reason_not_final": (
            "The DFT geometries were selected from the invalid historical X0. Nine were "
            "classified as lying in corrected-X0 geometry support without reading DFT values; "
            "the set remains deliberately stratified rather than an equilibrium sample."
        ),
        "condition": manifest["condition"],
        "counts": {
            "all_labels": 12,
            "corrected_X0_geometry_supported": int(np.sum(in_support)),
            "outside_corrected_X0_support": int(np.sum(~in_support)),
        },
        "provisional_corrected_support_gate": provisional_gate,
        "metrics_all_12": all_metrics,
        "metrics_corrected_X0_geometry_support": support_metrics,
        "metrics_outside_corrected_X0_support": outside_metrics,
        "Aprime_mode": mode_provenance,
        "bootstrap_95_percent_intervals_on_supported_set": {
            "n_resamples": args.bootstrap,
            "seed": args.seed,
            "intervals": bootstrap,
        },
        "decision": decision,
        "inputs": {
            "manifest": {"path": str(args.manifest), "sha256": manifest_hash},
            "unit_erratum": {"path": str(args.unit_erratum), "sha256": sha256(args.unit_erratum)},
            "atom_order_diagnostic": {"path": str(args.atom_order_diagnostic), "sha256": sha256(args.atom_order_diagnostic)},
            "geometry_support": {"path": str(args.geometry_support), "sha256": sha256(args.geometry_support)},
            "model_replay": {"path": str(args.model_replay), "sha256": sha256(args.model_replay)},
            "model_replay_summary": {"path": str(args.model_replay_summary), "sha256": sha256(args.model_replay_summary)},
            "shard_A_wall_seconds": float(summary_a["total_wall_seconds"]),
            "shard_B_wall_seconds": float(summary_b["total_wall_seconds"]),
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {"path": str(args.background), "sha256": sha256(args.background)},
            "corrected_result": {"path": str(args.corrected_result), "sha256": sha256(args.corrected_result)},
        },
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = args.output_dir / "historical_X0_DFT_merged12.extxyz"
    temporary = merged.with_name(merged.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, merged)
    atomic_npz(
        args.output_dir / "historical_X0_DFT_support_predictions.npz",
        sscha_indices=np.asarray(frozen_order, int),
        support_levels=support_levels,
        in_corrected_X0_geometry_support=in_support,
        MLIP_forces_eV_A=predicted_forces,
        DFT_forces_eV_A=target_forces,
        historical_X0_saved_forces_eV_A=historical_saved_forces,
        force_errors_eV_A=force_errors,
        long_range_forces_eV_A=long_forces,
        displacement_A=displacements,
        structure_to_operator_mapping=mappings,
        Aprime_coordinate_A=coordinates,
        Aprime_MLIP_force_eV_A=predicted_mode_forces,
        Aprime_DFT_force_eV_A=target_mode_forces,
        MLIP_energy_eV=predicted_energies,
        DFT_energy_eV=target_energies,
    )
    with (args.output_dir / "historical_X0_DFT_support_records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    atomic_json(args.output_dir / "historical_X0_DFT_support_summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "counts": summary["counts"],
                "provisional_corrected_support_gate": provisional_gate,
                "decision": decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
