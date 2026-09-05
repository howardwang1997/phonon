#!/usr/bin/env python3
"""Materialize the three-temperature S0 short-range delta dataset.

For every thermal structure the force target is

    F_delta = F_DFT_total - F_electronic(T) - F_v11,

where ``F_electronic(T)`` is the released E0 Cartesian q6 operator and its
smearing is fixed by ``degauss (Ry) = k_B T / Ry``.  Temperature and degauss
are provenance used to select the physical operator; neither is an input to
the local MACE delta model.  The E1 cross-degauss result fixes the additional
local Mermin term to zero.

The 300/600 K DFT totals and frozen-v11 predictions are recovered from the
audited endpoint delta dataset.  The new 450 K labels are evaluated with the
same frozen v11 model.  Harmonic replay is copied once and is not acted on by
the electronic operator because it is an explicit short-range anchor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read, write
from mace.calculators import MACECalculator
from scipy.constants import Boltzmann, electron_volt, physical_constants


RYDBERG_EV = physical_constants["Rydberg constant times hc in eV"][0]
TEMPERATURES = (300, 450, 600)
SPLITS = ("train", "val", "test")
ENDPOINT_COUNTS = {
    300: {"train": 36, "val": 6, "test": 3},
    600: {"train": 36, "val": 6, "test": 3},
}
REPLAY_COUNTS = {"train": 72, "val": 25, "test": 0}
THERMAL_REPEATS = {
    300: {"train": 2, "val": 4, "test": 1},
    450: {"train": 4, "val": 1, "test": 1},
    600: {"train": 2, "val": 4, "test": 1},
}
SPLIT_LABEL = {"train": "train", "val": "validation", "test": "test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def write_xyz(path: Path, structures: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, path)


def expected_degauss(temperature: int) -> float:
    return float(Boltzmann * temperature / electron_volt / RYDBERG_EV)


def harmonic_energy_forces(
    force_constants: np.ndarray,
    reference_positions: np.ndarray,
    cell: np.ndarray,
    positions: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    displacement = np.asarray(positions, float) - reference_positions
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    displacement = fractional @ cell
    phi_u = np.einsum("ijab,jb->ia", force_constants, displacement)
    energy = 0.5 * float(np.einsum("ia,ia->", displacement, phi_u))
    return energy, -phi_u, displacement


def force_metrics(values: list[np.ndarray]) -> dict:
    if not values:
        return {"RMSE_meV_A": 0.0, "max_abs_meV_A": 0.0}
    joined = np.concatenate([np.asarray(value, float).reshape(-1) for value in values])
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(joined**2)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(joined)) * 1000.0),
    }


def load_and_verify_operators(
    operator_dir: Path, decision_path: Path, e1_gate_path: Path
) -> tuple[dict[int, dict], dict]:
    decision = load_json(decision_path)
    e1_gate = load_json(e1_gate_path)
    if decision.get("status") != "passed" or decision.get("releases_E1") is not True:
        raise ValueError("E0 aggregate operator gate did not pass")
    if decision.get("temperature_law") != "smearing/degauss (Ry) = k_B*T/Ry":
        raise ValueError("unexpected E0 temperature law")
    if decision.get("development_temperatures_K") != list(TEMPERATURES):
        raise ValueError("unexpected E0 development temperatures")
    if decision.get("locked_validation_temperatures_K_not_accessed") != [375, 525]:
        raise ValueError("E0 locked-temperature boundary is missing")
    if e1_gate.get("status") != "passed":
        raise ValueError("E1 cross-degauss gate did not pass")
    if e1_gate.get("decision") != "set_local_Mermin_term_to_zero":
        raise ValueError("E1 does not release a zero local Mermin term")
    if e1_gate.get("development_temperatures_K") != list(TEMPERATURES):
        raise ValueError("unexpected E1 development temperatures")
    if e1_gate.get("locked_validation_temperatures_K_not_accessed") != [375, 525]:
        raise ValueError("E1 locked-temperature boundary is missing")
    if e1_gate.get("checks") != {
        "residual_RMSE_le_10_meV_A": True,
        "raw_le_10_meV_A_or_residual_le_20pct_raw": True,
    }:
        raise ValueError("E1 numerical checks are incomplete")
    e1_decision_hash = e1_gate.get("sources", {}).get("operator_decision", {}).get(
        "sha256"
    )
    if e1_decision_hash != sha256(decision_path):
        raise ValueError("E1 was evaluated with a different operator decision")

    records = {int(record["temperature_K"]): record for record in decision["operators"]}
    if sorted(records) != list(TEMPERATURES):
        raise ValueError("operator decision does not contain exactly 300/450/600 K")
    e1_hashes = e1_gate["sources"]["operator_sha256_by_temperature"]
    operators: dict[int, dict] = {}
    reference_hashes: set[str] = set()
    cell_hashes: set[str] = set()
    for temperature in TEMPERATURES:
        path = operator_dir / f"T{temperature}_operator.npz"
        digest = sha256(path)
        if digest != records[temperature]["operator_sha256"]:
            raise ValueError(f"T{temperature} operator hash differs from E0 decision")
        if digest != e1_hashes[str(temperature)]:
            raise ValueError(f"T{temperature} operator hash differs from E1 gate")
        with np.load(path, allow_pickle=False) as data:
            fc_raw = np.asarray(data["delta_fc_full"], float)
            reference_raw = np.asarray(data["reference_positions"], float)
            cell = np.asarray(data["cell"], float)
            mapping = np.asarray(data["atom_mapping"], int)
            degauss = float(data["degauss_Ry"])
            stored_temperature = float(data["temperature_K"])
            formula = str(data["degauss_formula"])
        if fc_raw.shape != (72, 72, 3, 3):
            raise ValueError(f"invalid T{temperature} force-constant shape")
        if reference_raw.shape != (72, 3) or cell.shape != (3, 3):
            raise ValueError(f"invalid T{temperature} reference geometry")
        if sorted(mapping.tolist()) != list(range(72)):
            raise ValueError(f"T{temperature} atom_mapping is not a permutation")
        if not np.isfinite(fc_raw).all():
            raise ValueError(f"T{temperature} operator contains non-finite values")
        if abs(stored_temperature - temperature) > 1.0e-8:
            raise ValueError(f"T{temperature} stored temperature mismatch")
        if formula != "k_B*T/Ry" or abs(degauss - expected_degauss(temperature)) > 1.0e-12:
            raise ValueError(f"T{temperature} degauss formula mismatch")
        pair_error = float(np.max(np.abs(fc_raw - fc_raw.transpose(1, 0, 3, 2))))
        asr_error = float(np.max(np.abs(fc_raw.sum(axis=1))))
        if pair_error > 1.0e-8 or asr_error > 1.0e-8:
            raise ValueError(f"T{temperature} operator symmetry/ASR gate failed")
        force_constants = fc_raw[mapping][:, mapping]
        reference = reference_raw[mapping]
        reference_hashes.add(hashlib.sha256(reference.tobytes()).hexdigest())
        cell_hashes.add(hashlib.sha256(cell.tobytes()).hexdigest())
        operators[temperature] = {
            "path": path,
            "sha256": digest,
            "force_constants": force_constants,
            "reference_positions": reference,
            "cell": cell,
            "degauss_Ry": degauss,
            "atom_mapping_is_identity": bool(np.array_equal(mapping, np.arange(72))),
            "pair_error": pair_error,
            "asr_error": asr_error,
        }
    if len(reference_hashes) != 1 or len(cell_hashes) != 1:
        raise ValueError("operator reference geometries differ across temperature")
    prerequisites = {
        "operator_decision": str(decision_path),
        "operator_decision_sha256": sha256(decision_path),
        "e1_residual_gate": str(e1_gate_path),
        "e1_residual_gate_sha256": sha256(e1_gate_path),
        "e1_overall": e1_gate["overall"],
        "local_Mermin_term": "zero_by_E1_gate",
    }
    return operators, prerequisites


def read_endpoint_data(endpoint_root: Path, base_model: Path) -> tuple[dict, dict]:
    manifest_path = endpoint_root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("base_model_sha256") != sha256(base_model):
        raise ValueError("endpoint data and requested frozen v11 model hashes differ")
    thermal: dict[int, dict[str, list]] = {300: {}, 600: {}}
    replay: dict[int, dict[str, list]] = {300: {}, 600: {}}
    sources: dict = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "files": {},
    }
    for temperature in (300, 600):
        for split in SPLITS:
            path = endpoint_root / f"T{temperature}" / f"{split}.xyz"
            expected_hash = manifest["lanes"][str(temperature)][split]["output_sha256"]
            if sha256(path) != expected_hash:
                raise ValueError(f"endpoint T{temperature}/{split} hash mismatch")
            structures = read(path, index=":")
            thermal_all = [
                atom for atom in structures if atom.info.get("delta_target_role") == "thermal"
            ]
            selected = [
                atom
                for atom in thermal_all
                if int(atom.info.get("delta_repeat_index", 0)) == 0
            ]
            replay_selected = [
                atom
                for atom in structures
                if atom.info.get("delta_target_role") == "harmonic_replay"
                and int(atom.info.get("delta_repeat_index", 0)) == 0
            ]
            expected_repeat = 2 if split == "train" else 4 if split == "val" else 1
            if len(thermal_all) != ENDPOINT_COUNTS[temperature][split] * expected_repeat:
                raise ValueError(f"unexpected expanded T{temperature}/{split} thermal count")
            if len(selected) != ENDPOINT_COUNTS[temperature][split]:
                raise ValueError(f"unexpected unique T{temperature}/{split} thermal count")
            if len(replay_selected) != REPLAY_COUNTS[split]:
                raise ValueError(f"unexpected T{temperature}/{split} replay count")
            thermal[temperature][split] = selected
            replay[temperature][split] = replay_selected
            sources["files"][f"T{temperature}/{split}.xyz"] = {
                "path": str(path),
                "sha256": expected_hash,
                "n_input": len(structures),
                "n_unique_thermal": len(selected),
                "n_replay": len(replay_selected),
            }
    verify_replay_identity(replay[300], replay[600])
    return {"thermal": thermal, "replay": replay[300]}, sources


def maximum_structure_difference(left, right) -> float:
    if len(left) != len(right) or not np.array_equal(left.numbers, right.numbers):
        return float("inf")
    values = [
        np.max(np.abs(np.asarray(left.positions) - np.asarray(right.positions))),
        np.max(np.abs(np.asarray(left.cell) - np.asarray(right.cell))),
    ]
    for key in ("TOTAL_forces", "BASE_forces", "REF_forces"):
        values.append(
            np.max(np.abs(np.asarray(left.arrays[key]) - np.asarray(right.arrays[key])))
        )
    for key in ("TOTAL_energy", "BASE_energy", "REF_energy"):
        values.append(abs(float(left.info[key]) - float(right.info[key])))
    return float(max(values))


def verify_replay_identity(left: dict[str, list], right: dict[str, list]) -> None:
    for split in SPLITS:
        if len(left[split]) != len(right[split]):
            raise ValueError(f"300/600 K replay counts differ for {split}")
        maximum = max(
            (maximum_structure_difference(a, b) for a, b in zip(left[split], right[split])),
            default=0.0,
        )
        if maximum > 1.0e-10:
            raise ValueError(f"300/600 K replay content differs for {split}: {maximum}")


def read_t450_data(root: Path, manifest_path: Path) -> tuple[dict[str, list], dict]:
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("n_structures") != 60:
        raise ValueError("450 K merge manifest is incomplete")
    if manifest.get("temperature_K") != 450 or manifest.get("n_structures_per_seed") != 20:
        raise ValueError("unexpected 450 K merge manifest")
    by_split: dict[str, list] = {}
    sources = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "files": {},
        "whole_trajectory_split": {"seed0": "train", "seed1": "val", "seed2": "test"},
    }
    combined = []
    for seed, split in ((0, "train"), (1, "val"), (2, "test")):
        path = root / f"seed{seed}.xyz"
        structures = read(path, index=":")
        if len(structures) != 20:
            raise ValueError(f"450 K seed{seed} does not contain 20 structures")
        for atom in structures:
            if int(atom.info.get("trajectory_seed", -1)) != seed:
                raise ValueError(f"450 K seed metadata mismatch in seed{seed}")
            if abs(float(atom.info.get("lattice_temperature_K", 0.0)) - 450.0) > 1.0e-8:
                raise ValueError("450 K lattice-temperature metadata mismatch")
            if abs(float(atom.info.get("degauss_Ry", 0.0)) - expected_degauss(450)) > 1.0e-8:
                raise ValueError("450 K source degauss is inconsistent with k_B*T/Ry")
        by_split[split] = structures
        combined.extend(structures)
        sources["files"][path.name] = {"path": str(path), "sha256": sha256(path)}
    all_path = root / "all60.xyz"
    all_structures = read(all_path, index=":")
    if len(all_structures) != 60:
        raise ValueError("450 K all60.xyz does not contain 60 structures")
    maximum = max(
        maximum_raw_difference(a, b) for a, b in zip(combined, all_structures)
    )
    if maximum > 1.0e-10:
        raise ValueError(f"450 K seed files do not reproduce all60.xyz: {maximum}")
    sources["files"][all_path.name] = {"path": str(all_path), "sha256": sha256(all_path)}
    sources["seed_concat_vs_all60_max_abs"] = maximum
    return by_split, sources


def maximum_raw_difference(left, right) -> float:
    if len(left) != len(right) or not np.array_equal(left.numbers, right.numbers):
        return float("inf")
    return float(
        max(
            np.max(np.abs(np.asarray(left.positions) - np.asarray(right.positions))),
            np.max(np.abs(np.asarray(left.cell) - np.asarray(right.cell))),
            np.max(
                np.abs(
                    np.asarray(left.arrays["REF_forces"])
                    - np.asarray(right.arrays["REF_forces"])
                )
            ),
            abs(float(left.info["REF_energy"]) - float(right.info["REF_energy"])),
        )
    )


def physical_config_type(temperature: int, split: str) -> str:
    suffix = "" if split == "train" else f"_{SPLIT_LABEL[split]}"
    return f"physical_s0_thermal_{temperature}K{suffix}"


def controlled_info(source_info: dict) -> dict:
    excluded = {
        "REF_energy",
        "TOTAL_energy",
        "LONG_RANGE_energy",
        "SHORT_RANGE_TARGET_energy",
        "BASE_energy",
        "config_type",
        "split",
        "force_target",
        "delta_target_role",
        "delta_repeat_index",
        "long_range_degauss_Ry",
        "long_range_temperature_K",
    }
    return {key: value for key, value in source_info.items() if key not in excluded}


def materialize_thermal(
    source,
    temperature: int,
    split: str,
    operator: dict,
    base_forces: np.ndarray,
    base_energy: float,
    source_family: str,
):
    if len(source) != 72:
        raise ValueError("thermal structure does not have 72 atoms")
    cell = np.asarray(source.cell, float)
    if not np.allclose(cell, operator["cell"], atol=2.0e-5, rtol=0.0):
        raise ValueError(f"T{temperature} structure cell does not match operator")
    total_forces = np.asarray(
        source.arrays.get("TOTAL_forces", source.arrays["REF_forces"]), float
    ).copy()
    total_energy = float(source.info.get("TOTAL_energy", source.info["REF_energy"]))
    long_energy, long_forces, displacement = harmonic_energy_forces(
        operator["force_constants"],
        operator["reference_positions"],
        cell,
        np.asarray(source.positions, float),
    )
    net_long = float(np.linalg.norm(long_forces.sum(axis=0)))
    max_displacement = float(np.linalg.norm(displacement, axis=1).max())
    if net_long > 2.0e-5:
        raise ValueError(f"T{temperature} long-range net force exceeds gate: {net_long}")
    if max_displacement > 1.5:
        raise ValueError(f"T{temperature} displacement exceeds 1.5 A gate")
    short_forces = total_forces - long_forces
    short_energy = total_energy - long_energy
    delta_forces = short_forces - np.asarray(base_forces, float)
    delta_energy = short_energy - float(base_energy)

    result = source.copy()
    result.info = controlled_info(source.info)
    result.info.update(
        {
            "REF_energy": delta_energy,
            "TOTAL_energy": total_energy,
            "LONG_RANGE_energy": long_energy,
            "SHORT_RANGE_TARGET_energy": short_energy,
            "BASE_energy": float(base_energy),
            "config_type": physical_config_type(temperature, split),
            "split": SPLIT_LABEL[split],
            "force_target": "DFT_total_minus_physical_electronic_operator_minus_frozen_v11",
            "delta_target_role": "thermal",
            "source_family": source_family,
            "lattice_temperature_K": float(temperature),
            "degauss_Ry": operator["degauss_Ry"],
            "long_range_temperature_K": temperature,
            "long_range_degauss_Ry": operator["degauss_Ry"],
            "local_Mermin_term": "zero_by_E1_gate",
            "temperature_is_model_input": False,
        }
    )
    result.arrays["TOTAL_forces"] = total_forces
    result.arrays["LONG_RANGE_forces"] = long_forces
    result.arrays["SHORT_RANGE_TARGET_forces"] = short_forces
    result.arrays["BASE_forces"] = np.asarray(base_forces, float).copy()
    result.arrays["REF_forces"] = delta_forces
    reconstructed = result.arrays["BASE_forces"] + delta_forces + long_forces
    return result, {
        "total": total_forces,
        "long": long_forces,
        "short": short_forces,
        "base": result.arrays["BASE_forces"],
        "delta": delta_forces,
        "reconstruction": reconstructed - total_forces,
        "net_long": net_long,
        "max_displacement": max_displacement,
    }


def materialize_replay(source, split: str):
    total_forces = np.asarray(source.arrays["TOTAL_forces"], float).copy()
    base_forces = np.asarray(source.arrays["BASE_forces"], float).copy()
    delta_forces = total_forces - base_forces
    total_energy = float(source.info["TOTAL_energy"])
    base_energy = float(source.info["BASE_energy"])
    result = source.copy()
    result.info = controlled_info(source.info)
    for key in ("degauss_Ry", "lattice_temperature_K"):
        result.info.pop(key, None)
    suffix = "" if split == "train" else f"_{SPLIT_LABEL[split]}"
    result.info.update(
        {
            "REF_energy": total_energy - base_energy,
            "TOTAL_energy": total_energy,
            "LONG_RANGE_energy": 0.0,
            "SHORT_RANGE_TARGET_energy": total_energy,
            "BASE_energy": base_energy,
            "config_type": f"physical_s0_harmonic_replay{suffix}",
            "split": SPLIT_LABEL[split],
            "force_target": "short_range_harmonic_replay_minus_frozen_v11",
            "delta_target_role": "harmonic_replay",
            "source_family": "v11_fc_distillation_replay",
            "temperature_is_model_input": False,
        }
    )
    result.arrays["TOTAL_forces"] = total_forces
    result.arrays["LONG_RANGE_forces"] = np.zeros_like(total_forces)
    result.arrays["SHORT_RANGE_TARGET_forces"] = total_forces.copy()
    result.arrays["BASE_forces"] = base_forces
    result.arrays["REF_forces"] = delta_forces
    return result, {
        "total": total_forces,
        "long": np.zeros_like(total_forces),
        "short": total_forces,
        "base": base_forces,
        "delta": delta_forces,
        "reconstruction": base_forces + delta_forces - total_forces,
        "net_long": 0.0,
        "max_displacement": 0.0,
    }


def summarize(records: list[dict], n_output: int, repeat: int) -> dict:
    reconstruction = force_metrics([record["reconstruction"] for record in records])
    return {
        "n_unique_structures": len(records),
        "n_output_structures": n_output,
        "repeat": repeat,
        "total_force": force_metrics([record["total"] for record in records]),
        "physical_long_range_force": force_metrics([record["long"] for record in records]),
        "short_range_target_force": force_metrics([record["short"] for record in records]),
        "frozen_v11_force": force_metrics([record["base"] for record in records]),
        "delta_target_force": force_metrics([record["delta"] for record in records]),
        "reconstruction_max_abs_meV_A": reconstruction["max_abs_meV_A"],
        "long_range_net_force_max_eV_A": float(
            max((record["net_long"] for record in records), default=0.0)
        ),
        "max_displacement_A": float(
            max((record["max_displacement"] for record in records), default=0.0)
        ),
    }


def expanded_copies(structures: list, repeat: int) -> list:
    output = []
    for structure in structures:
        for repeat_index in range(repeat):
            copied = structure.copy()
            copied.info["delta_repeat_index"] = repeat_index
            output.append(copied)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--t450-root", type=Path, required=True)
    parser.add_argument("--t450-manifest", type=Path, required=True)
    parser.add_argument("--operator-dir", type=Path, required=True)
    parser.add_argument("--operator-decision", type=Path, required=True)
    parser.add_argument("--e1-gate", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output}")
    stage = args.output.with_name(f".{args.output.name}.building.{os.getpid()}")
    if stage.exists():
        raise ValueError(f"staging directory already exists: {stage}")
    stage.mkdir(parents=True)

    try:
        operators, prerequisites = load_and_verify_operators(
            args.operator_dir, args.operator_decision, args.e1_gate
        )
        endpoint, endpoint_sources = read_endpoint_data(args.endpoint_root, args.base_model)
        t450, t450_sources = read_t450_data(args.t450_root, args.t450_manifest)
        calculator = MACECalculator(
            model_paths=str(args.base_model),
            device=args.device,
            default_dtype="float32",
        )

        thermal_output: dict[int, dict[str, list]] = {
            temperature: {} for temperature in TEMPERATURES
        }
        metrics: dict[str, dict] = {str(temperature): {} for temperature in TEMPERATURES}
        maximum_reconstruction = 0.0
        maximum_net_long = 0.0
        for temperature in TEMPERATURES:
            for split in SPLITS:
                sources = (
                    endpoint["thermal"][temperature][split]
                    if temperature in (300, 600)
                    else t450[split]
                )
                unique_output = []
                records = []
                for source in sources:
                    if temperature in (300, 600):
                        base_forces = np.asarray(source.arrays["BASE_forces"], float)
                        base_energy = float(source.info["BASE_energy"])
                        source_family = "endpoint_DFT_recovered_from_audited_delta_data"
                    else:
                        evaluated = source.copy()
                        evaluated.calc = calculator
                        base_forces = np.asarray(evaluated.get_forces(), float)
                        base_energy = float(evaluated.get_potential_energy())
                        source_family = "450K_on_policy_P4_DFT"
                    result, record = materialize_thermal(
                        source,
                        temperature,
                        split,
                        operators[temperature],
                        base_forces,
                        base_energy,
                        source_family,
                    )
                    unique_output.append(result)
                    records.append(record)
                repeat = THERMAL_REPEATS[temperature][split]
                expanded = expanded_copies(unique_output, repeat)
                thermal_output[temperature][split] = expanded
                destination = stage / f"T{temperature}" / f"{split}.xyz"
                write_xyz(destination, expanded)
                summary = summarize(records, len(expanded), repeat)
                summary["output"] = str(args.output / f"T{temperature}" / f"{split}.xyz")
                summary["output_sha256"] = sha256(destination)
                metrics[str(temperature)][split] = summary
                maximum_reconstruction = max(
                    maximum_reconstruction, summary["reconstruction_max_abs_meV_A"]
                )
                maximum_net_long = max(
                    maximum_net_long, summary["long_range_net_force_max_eV_A"]
                )

        replay_output: dict[str, list] = {}
        replay_metrics: dict[str, dict] = {}
        for split in ("train", "val"):
            unique_output = []
            records = []
            for source in endpoint["replay"][split]:
                result, record = materialize_replay(source, split)
                unique_output.append(result)
                records.append(record)
            expanded = expanded_copies(unique_output, 1)
            replay_output[split] = expanded
            destination = stage / "replay" / f"{split}.xyz"
            write_xyz(destination, expanded)
            summary = summarize(records, len(expanded), 1)
            summary["output"] = str(args.output / "replay" / f"{split}.xyz")
            summary["output_sha256"] = sha256(destination)
            replay_metrics[split] = summary
            maximum_reconstruction = max(
                maximum_reconstruction, summary["reconstruction_max_abs_meV_A"]
            )

        joint = {
            "train": [
                *thermal_output[300]["train"],
                *thermal_output[450]["train"],
                *thermal_output[600]["train"],
                *replay_output["train"],
            ],
            "val": [
                *thermal_output[300]["val"],
                *thermal_output[450]["val"],
                *thermal_output[600]["val"],
                *replay_output["val"],
            ],
            "test": [
                *thermal_output[300]["test"],
                *thermal_output[450]["test"],
                *thermal_output[600]["test"],
            ],
        }
        joint_manifest = {}
        for split, structures in joint.items():
            destination = stage / f"{split}.xyz"
            write_xyz(destination, structures)
            joint_manifest[split] = {
                "output": str(args.output / f"{split}.xyz"),
                "output_sha256": sha256(destination),
                "n_structures": len(structures),
            }

        operator_output = stage / "operators"
        operator_output.mkdir()
        operator_records = {}
        for temperature, operator in operators.items():
            destination = operator_output / f"T{temperature}_operator.npz"
            shutil.copy2(operator["path"], destination)
            if sha256(destination) != operator["sha256"]:
                raise ValueError(f"T{temperature} operator changed during copy")
            operator_records[str(temperature)] = {
                "output": str(args.output / "operators" / destination.name),
                "sha256": operator["sha256"],
                "degauss_Ry": operator["degauss_Ry"],
                "degauss_formula": "k_B*T/Ry",
                "atom_mapping_is_identity": operator["atom_mapping_is_identity"],
                "force_pair_max_abs_eV_A2": operator["pair_error"],
                "force_ASR_max_abs_eV_A2": operator["asr_error"],
            }

        train_rmse = {
            temperature: metrics[str(temperature)]["train"]["delta_target_force"][
                "RMSE_meV_A"
            ]
            for temperature in TEMPERATURES
        }
        reference_mse = train_rmse[600] ** 2
        temperature_weights = {
            temperature: float(
                np.clip(reference_mse / max(train_rmse[temperature] ** 2, 1.0e-20), 0.25, 4.0)
            )
            for temperature in TEMPERATURES
        }
        config_weights = {"Default": 12.0}
        for temperature in TEMPERATURES:
            for split in SPLITS:
                config_weights[physical_config_type(temperature, split)] = temperature_weights[
                    temperature
                ]
        config_weights["physical_s0_harmonic_replay"] = 12.0
        config_weights["physical_s0_harmonic_replay_validation"] = 12.0

        gates = {
            "E0_operator_gate_passed": True,
            "E1_cross_degauss_gate_passed": True,
            "locked_375_525_not_accessed": True,
            "base_model_hash_matches_endpoint_data": True,
            "endpoint_file_hashes_match_manifest": True,
            "T450_whole_trajectory_split": True,
            "T300_T600_replay_identical": True,
            "force_reconstruction_max_abs_le_1e-8_meV_A": maximum_reconstruction
            <= 1.0e-8,
            "long_range_net_force_max_le_2e-5_eV_A": maximum_net_long <= 2.0e-5,
        }
        if not all(gates.values()):
            raise ValueError(f"S0 data gate failed: {gates}")
        manifest = {
            "status": "passed",
            "scope": "S0 three-temperature physical-operator short-range delta dataset",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_formula": "F_delta = F_DFT_total - F_electronic(T) - F_v11",
            "electronic_smearing_formula": "smearing/degauss (Ry) = k_B*T/Ry",
            "lattice_temperature_role": "selects the released physical electronic operator; not a local-model input",
            "local_Mermin_term": "zero_by_E1_gate",
            "development_temperatures_K": list(TEMPERATURES),
            "locked_validation_temperatures_K_not_accessed": [375, 525],
            "temperature_or_degauss_is_model_input": False,
            "base_model": str(args.base_model),
            "base_model_sha256": sha256(args.base_model),
            "device_used_for_450K_base_predictions": args.device,
            "source_data": {"endpoints": endpoint_sources, "T450": t450_sources},
            "prerequisites": prerequisites,
            "operators": operator_records,
            "thermal_repeat_policy": {
                str(temperature): THERMAL_REPEATS[temperature]
                for temperature in TEMPERATURES
            },
            "metrics_by_temperature": metrics,
            "replay": replay_metrics,
            "joint": joint_manifest,
            "recommended_config_weights": {
                "method": "inverse train delta-target MSE, normalized to T600=1 and clipped to [0.25,4]; replay fixed at 12",
                "train_delta_target_RMSE_meV_A": {
                    str(key): value for key, value in train_rmse.items()
                },
                "temperature_weights": {
                    str(key): value for key, value in temperature_weights.items()
                },
                "config_type_weights": config_weights,
            },
            "aggregate_gates": {
                "checks": gates,
                "force_reconstruction_max_abs_meV_A": maximum_reconstruction,
                "long_range_net_force_max_eV_A": maximum_net_long,
            },
        }
        atomic_json(stage / "manifest.json", manifest)
        os.replace(stage, args.output)
        print(json.dumps(manifest, indent=2))
        return 0
    except Exception:
        failure = stage / "FAILED.json"
        atomic_json(
            failure,
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
