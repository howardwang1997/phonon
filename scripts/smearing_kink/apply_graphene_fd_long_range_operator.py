#!/usr/bin/env python3
"""Attach a frozen harmonic long-range operator to raw force labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read, write


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def harmonic_energy_forces(fc, reference_positions, cell, positions):
    displacement = np.asarray(positions, float) - reference_positions
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    displacement = fractional @ cell
    phi_u = np.einsum("ijab,jb->ia", fc, displacement)
    energy = 0.5 * float(np.einsum("ia,ia->", displacement, phi_u))
    return energy, -phi_u, displacement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", default="independent_force_holdout")
    args = parser.parse_args()

    structures = read(args.input, index=":")
    with np.load(args.operator, allow_pickle=False) as data:
        fc_raw = np.asarray(data["delta_fc_full"], float)
        reference_raw = np.asarray(data["reference_positions"], float)
        operator_cell = np.asarray(data["cell"], float)
        mapping = np.asarray(data["atom_mapping"], int)
        degauss = float(data["degauss_Ry"])
        temperature = int(data["temperature_K"])
    if sorted(mapping.tolist()) != list(range(len(mapping))):
        raise ValueError("operator atom_mapping is not a permutation")
    fc = fc_raw[mapping][:, mapping]
    reference = reference_raw[mapping]

    output = []
    long_values = []
    max_displacement = 0.0
    for index, structure in enumerate(structures):
        if len(structure) != len(mapping):
            raise ValueError(f"structure {index} atom count does not match operator")
        cell = np.asarray(structure.cell, float)
        if not np.allclose(cell, operator_cell, atol=2e-5, rtol=0.0):
            raise ValueError(f"structure {index} cell does not match operator")
        total_forces = np.asarray(structure.arrays["REF_forces"], float).copy()
        total_energy = float(structure.info["REF_energy"])
        energy_long, force_long, displacement = harmonic_energy_forces(
            fc, reference, cell, np.asarray(structure.positions, float)
        )
        if np.linalg.norm(force_long.sum(axis=0)) > 2e-5:
            raise ValueError(f"structure {index} long-range force violates translation")
        result = structure.copy()
        result.arrays["TOTAL_forces"] = total_forces
        result.info["TOTAL_energy"] = total_energy
        result.arrays["LONG_RANGE_forces"] = force_long
        result.info["LONG_RANGE_energy"] = energy_long
        result.arrays["SHORT_RANGE_TARGET_forces"] = total_forces - force_long
        result.info["SHORT_RANGE_TARGET_energy"] = total_energy - energy_long
        result.arrays["REF_forces"] = total_forces - force_long
        result.info["REF_energy"] = total_energy - energy_long
        result.info["delta_target_role"] = args.role
        result.info["long_range_degauss_Ry"] = degauss
        result.info["long_range_temperature_K"] = temperature
        output.append(result)
        long_values.append(force_long.reshape(-1))
        max_displacement = max(
            max_displacement, float(np.linalg.norm(displacement, axis=1).max())
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, output, format="extxyz")
    joined = np.concatenate(long_values)
    payload = {
        "status": "complete",
        "role": args.role,
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "operator": str(args.operator),
        "operator_sha256": sha256(args.operator),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "n_structures": len(output),
        "temperature_K": temperature,
        "degauss_Ry": degauss,
        "long_range_force_RMSE_meV_A": float(
            np.sqrt(np.mean(joined**2)) * 1000.0
        ),
        "long_range_force_max_abs_meV_A": float(
            np.max(np.abs(joined)) * 1000.0
        ),
        "max_displacement_A": max_displacement,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_name(args.manifest.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.manifest)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
