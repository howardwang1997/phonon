#!/usr/bin/env python3
"""Numerically replay the conditioned short model at its 300/600 K endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import td_common as tdc  # noqa: E402
from conditioned_mace import conditioned_short_calculator  # noqa: E402


ENDPOINTS = (300, 600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labelled_path(specification: str) -> tuple[int, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("endpoint data must use TEMPERATURE=/path")
    raw_temperature, raw_path = specification.split("=", 1)
    try:
        temperature = int(raw_temperature)
    except ValueError as error:
        raise argparse.ArgumentTypeError("endpoint temperature must be an integer") from error
    return temperature, Path(raw_path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def evaluate_structures(
    structures,
    temperature: int,
    base,
    delta_300,
    delta_600,
) -> dict:
    conditioned, weights = conditioned_short_calculator(
        base, delta_300, delta_600, temperature
    )
    records = []
    for index, structure in enumerate(structures):
        atoms = structure.copy()
        atoms.calc = base
        base_energy = float(atoms.get_potential_energy())
        base_force = np.asarray(atoms.get_forces(), float)
        atoms.calc = delta_300
        delta_300_energy = float(atoms.get_potential_energy())
        delta_300_force = np.asarray(atoms.get_forces(), float)
        atoms.calc = delta_600
        delta_600_energy = float(atoms.get_potential_energy())
        delta_600_force = np.asarray(atoms.get_forces(), float)

        expected_energy = (
            base_energy
            + weights.delta_300 * delta_300_energy
            + weights.delta_600 * delta_600_energy
        )
        expected_force = (
            base_force
            + weights.delta_300 * delta_300_force
            + weights.delta_600 * delta_600_force
        )
        atoms.calc = conditioned
        replay_energy = float(atoms.get_potential_energy())
        replay_force = np.asarray(atoms.get_forces(), float)
        delta_force = replay_force - expected_force
        records.append(
            {
                "structure_index": index,
                "snapshot_index": (
                    int(structure.info["snapshot_index"])
                    if "snapshot_index" in structure.info
                    else None
                ),
                "energy_difference_eV_atom": float(
                    (replay_energy - expected_energy) / len(structure)
                ),
                "force_RMSE_eV_A": float(np.sqrt(np.mean(delta_force**2))),
                "force_max_abs_eV_A": float(np.max(np.abs(delta_force))),
            }
        )
    return {
        "n_structures": len(structures),
        "weights": {
            "base": 1.0,
            "delta_300": weights.delta_300,
            "delta_600": weights.delta_600,
        },
        "energy_max_abs_difference_eV_atom": max(
            abs(record["energy_difference_eV_atom"]) for record in records
        ),
        "force_max_abs_difference_eV_A": max(
            record["force_max_abs_eV_A"] for record in records
        ),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", type=labelled_path, required=True)
    parser.add_argument("--harmonic", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--delta-model-300", type=Path, required=True)
    parser.add_argument("--delta-model-600", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-tolerance-eV-A", type=float, default=1.0e-6)
    parser.add_argument("--energy-tolerance-eV-atom", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    endpoints = dict(args.endpoint)
    if set(endpoints) != set(ENDPOINTS) or len(args.endpoint) != len(ENDPOINTS):
        parser.error("provide exactly one endpoint dataset for 300 and 600 K")
    for path in (
        args.harmonic,
        args.base_model,
        args.delta_model_300,
        args.delta_model_600,
        *endpoints.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = tdc.get_mace_calc(str(args.base_model), device=args.device)
    delta_300 = tdc.get_mace_calc(str(args.delta_model_300), device=args.device)
    delta_600 = tdc.get_mace_calc(str(args.delta_model_600), device=args.device)
    harmonic_structures = read(args.harmonic, index=":")
    endpoint_results = {}
    passed = True
    for temperature in ENDPOINTS:
        thermal_structures = read(endpoints[temperature], index=":")
        thermal = evaluate_structures(
            thermal_structures, temperature, base, delta_300, delta_600
        )
        harmonic = evaluate_structures(
            harmonic_structures, temperature, base, delta_300, delta_600
        )
        endpoint_passed = bool(
            max(
                thermal["force_max_abs_difference_eV_A"],
                harmonic["force_max_abs_difference_eV_A"],
            )
            <= args.force_tolerance_eV_A
            and max(
                thermal["energy_max_abs_difference_eV_atom"],
                harmonic["energy_max_abs_difference_eV_atom"],
            )
            <= args.energy_tolerance_eV_atom
        )
        passed = passed and endpoint_passed
        endpoint_results[str(temperature)] = {
            "dataset": {
                "path": str(endpoints[temperature]),
                "sha256": sha256(endpoints[temperature]),
            },
            "thermal_replay": thermal,
            "harmonic_replay": harmonic,
            "passes_endpoint_identity": endpoint_passed,
        }

    payload = {
        "status": "passed" if passed else "failed",
        "scope": (
            "implementation replay of the frozen temperature-conditioned short "
            "calculator at 300/600 K before any 450 K targets"
        ),
        "formula": (
            "E_SR(T) = E_v11 + (1-w(T))*E_delta300 + w(T)*E_delta600; "
            "w(T)=(T-300)/300"
        ),
        "thresholds": {
            "force_max_abs_difference_eV_A": args.force_tolerance_eV_A,
            "energy_max_abs_difference_eV_atom": args.energy_tolerance_eV_atom,
        },
        "passes_all_endpoint_identity_checks": passed,
        "models": {
            "base": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
            "delta_300": {
                "path": str(args.delta_model_300),
                "sha256": sha256(args.delta_model_300),
            },
            "delta_600": {
                "path": str(args.delta_model_600),
                "sha256": sha256(args.delta_model_600),
            },
        },
        "harmonic_dataset": {
            "path": str(args.harmonic),
            "sha256": sha256(args.harmonic),
            "n_structures": len(harmonic_structures),
        },
        "endpoints": endpoint_results,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
