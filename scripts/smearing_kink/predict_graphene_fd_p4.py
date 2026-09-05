#!/usr/bin/env python3
"""Materialize the frozen P4 force and static-phonon predictions without targets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
import td_common as tdc  # noqa: E402
from conditioned_mace import conditioned_short_calculator  # noqa: E402
from friedel_calc import fc2_from_calc  # noqa: E402
from graphene_fd_p4_common import (  # noqa: E402
    atomic_json,
    atomic_npz,
    geometry_sha256,
    harmonic_energy_forces,
    p4_key,
    sha256,
)
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 changed: {observed} != {expected}")


def load_operator(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        raw_fc = np.asarray(payload["delta_fc_full"], float)
        raw_reference = np.asarray(payload["reference_positions"], float)
        cell = np.asarray(payload["cell"], float)
        mapping = np.asarray(payload["atom_mapping"], int)
        temperature = float(payload["temperature_K"])
        degauss = float(payload["degauss_Ry"])
    if sorted(mapping.tolist()) != list(range(len(mapping))):
        raise ValueError("operator atom_mapping is not a permutation")
    return raw_fc[mapping][:, mapping], raw_reference[mapping], cell, temperature, degauss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--merge-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--delta-model-300", type=Path, required=True)
    parser.add_argument("--delta-model-600", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=450.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-static", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text())
    merge = json.loads(args.merge_manifest.read_text())
    if freeze.get("status") != "frozen_before_450_holdout":
        raise ValueError("P4 predictor was not frozen before the holdout")
    if merge.get("status") != "complete" or int(merge["n_structures"]) != 60:
        raise ValueError("P4 60-structure merge is incomplete")
    if merge["freeze_manifest"]["sha256"] != sha256(args.freeze_manifest):
        raise ValueError("merge used a different freeze manifest")
    if merge["all_output"]["sha256"] != sha256(args.labels):
        raise ValueError("merged label file changed after merge")

    predictor = freeze["predictor"]
    require_hash(args.base_model, predictor["base_model_sha256"], "base model")
    require_hash(
        args.delta_model_300,
        predictor["delta_model_300_sha256"],
        "300 K delta model",
    )
    require_hash(
        args.delta_model_600,
        predictor["delta_model_600_sha256"],
        "600 K delta model",
    )
    operator_record = freeze["prediction_sampling_operator"]["output"]
    require_hash(args.operator, operator_record["sha256"], "450 K sampling operator")
    if not np.isclose(
        args.temperature,
        float(freeze["T450_on_policy"]["temperature_K"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("prediction temperature differs from the frozen temperature")

    operator_fc, reference_positions, operator_cell, operator_temperature, operator_degauss = (
        load_operator(args.operator)
    )
    if not np.isclose(operator_temperature, args.temperature, rtol=0.0, atol=1.0e-12):
        raise ValueError("operator temperature mismatch")

    base = tdc.get_mace_calc(str(args.base_model), device=args.device)
    delta_300 = tdc.get_mace_calc(str(args.delta_model_300), device=args.device)
    delta_600 = tdc.get_mace_calc(str(args.delta_model_600), device=args.device)
    _, weights = conditioned_short_calculator(
        base, delta_300, delta_600, args.temperature
    )

    structures = read(args.labels, index=":")
    keys = []
    seeds = []
    indices = []
    geometry_hashes = []
    short_forces = []
    long_forces = []
    total_forces = []
    for structure in structures:
        seed = int(structure.info["trajectory_seed"])
        snapshot_index = int(structure.info["snapshot_index"])
        key = p4_key(seed, snapshot_index)
        cell = np.asarray(structure.cell, float)
        if len(structure) != len(reference_positions):
            raise ValueError(f"atom count mismatch in {key}")
        if not np.allclose(cell, operator_cell, rtol=0.0, atol=2.0e-5):
            raise ValueError(f"cell mismatch in {key}")

        atoms = structure.copy()
        atoms.calc = base
        base_force = np.asarray(atoms.get_forces(), float)
        atoms.calc = delta_300
        force_300 = np.asarray(atoms.get_forces(), float)
        atoms.calc = delta_600
        force_600 = np.asarray(atoms.get_forces(), float)
        short_force = (
            base_force
            + weights.delta_300 * force_300
            + weights.delta_600 * force_600
        )
        _, long_force, _ = harmonic_energy_forces(
            operator_fc,
            reference_positions,
            cell,
            np.asarray(structure.positions, float),
        )
        if np.linalg.norm(long_force.sum(axis=0)) > 2.0e-5:
            raise ValueError(f"operator violates translational invariance in {key}")
        keys.append(key)
        seeds.append(seed)
        indices.append(snapshot_index)
        geometry_hashes.append(
            geometry_sha256(structure.numbers, cell, structure.positions)
        )
        short_forces.append(short_force)
        long_forces.append(long_force)
        total_forces.append(short_force + long_force)

    expected_keys = [
        p4_key(seed, index)
        for seed in (0, 1, 2)
        for index in freeze["T450_on_policy"]["dft_label_indices_per_seed"]
    ]
    if keys != expected_keys:
        raise ValueError("merged labels are not in the frozen P4 order")

    atomic_npz(
        args.output_predictions,
        keys=np.asarray(keys),
        trajectory_seed=np.asarray(seeds, int),
        snapshot_index=np.asarray(indices, int),
        geometry_sha256=np.asarray(geometry_hashes),
        predicted_short_forces_eV_A=np.asarray(short_forces, float),
        predicted_long_range_forces_eV_A=np.asarray(long_forces, float),
        predicted_total_forces_eV_A=np.asarray(total_forces, float),
        temperature_K=np.array(args.temperature),
        degauss_Ry=np.array(operator_degauss),
        weight_delta_300=np.array(weights.delta_300),
        weight_delta_600=np.array(weights.delta_600),
        freeze_manifest_sha256=np.array(sha256(args.freeze_manifest)),
        merged_labels_sha256=np.array(sha256(args.labels)),
    )

    phonon = fm.load_ph(args.background)
    ideal = phonopy_to_ase(phonon.supercell)
    ideal.wrap()
    if len(ideal) != len(reference_positions):
        raise ValueError("background supercell atom count differs from operator")
    if not np.allclose(np.asarray(ideal.cell), operator_cell, rtol=0.0, atol=2.0e-5):
        raise ValueError("background supercell cell differs from operator")
    if not np.allclose(
        np.asarray(ideal.positions), reference_positions, rtol=0.0, atol=2.0e-5
    ):
        raise ValueError("background supercell atom order differs from operator")
    conditioned, static_weights = conditioned_short_calculator(
        base, delta_300, delta_600, args.temperature
    )
    rebuilt, short_fc = fc2_from_calc(
        phonon,
        conditioned,
        distance=args.distance,
        subtract_ref=True,
    )
    rebuilt.symmetrize_force_constants()
    short_fc = np.asarray(rebuilt.force_constants, float)
    if short_fc.shape != operator_fc.shape:
        raise ValueError("conditioned static FC2 and sampling operator shapes differ")
    full_fc = short_fc + operator_fc
    atomic_npz(
        args.output_static,
        short_force_constants=short_fc,
        long_range_force_constants=operator_fc,
        full_force_constants=full_fc,
        temperature_K=np.array(args.temperature),
        degauss_Ry=np.array(operator_degauss),
        finite_displacement_A=np.array(args.distance),
        supercell_matrix=np.asarray(rebuilt.supercell_matrix, int),
        primitive_matrix=np.asarray(rebuilt.primitive_matrix, float),
        weight_delta_300=np.array(static_weights.delta_300),
        weight_delta_600=np.array(static_weights.delta_600),
    )

    result = {
        "status": "complete",
        "scope": (
            "frozen 450 K predictions materialized without using REF_forces "
            "or REF_energy values; no 450 K fitting or selection"
        ),
        "temperature_K": args.temperature,
        "degauss_Ry": operator_degauss,
        "weights": {
            "base": 1.0,
            "delta_300": weights.delta_300,
            "delta_600": weights.delta_600,
        },
        "inputs": {
            "freeze_manifest": {
                "path": str(args.freeze_manifest),
                "sha256": sha256(args.freeze_manifest),
            },
            "merged_geometry_container": {
                "path": str(args.labels),
                "sha256": sha256(args.labels),
                "note": "reference arrays were ignored by this prediction-only program",
            },
            "base_model": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
            "delta_model_300": {
                "path": str(args.delta_model_300),
                "sha256": sha256(args.delta_model_300),
            },
            "delta_model_600": {
                "path": str(args.delta_model_600),
                "sha256": sha256(args.delta_model_600),
            },
            "sampling_operator": {
                "path": str(args.operator),
                "sha256": sha256(args.operator),
            },
            "background": {"path": str(args.background), "sha256": sha256(args.background)},
        },
        "n_force_predictions": len(keys),
        "force_predictions": {
            "path": str(args.output_predictions),
            "sha256": sha256(args.output_predictions),
        },
        "static_prediction": {
            "path": str(args.output_static),
            "sha256": sha256(args.output_static),
            "finite_displacement_A": args.distance,
            "short_force_constants_shape": list(short_fc.shape),
            "short_pair_symmetry_max_abs_eV_A2": float(
                np.max(np.abs(short_fc - short_fc.transpose(1, 0, 3, 2)))
            ),
            "full_pair_symmetry_max_abs_eV_A2": float(
                np.max(np.abs(full_fc - full_fc.transpose(1, 0, 3, 2)))
            ),
        },
    }
    atomic_json(args.output_manifest, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
