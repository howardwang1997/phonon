#!/usr/bin/env python3
"""Check statistical convergence of a checkpointed DFT-MD/TDEP result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import phonopy
from ase.calculators.singlepoint import SinglePointCalculator
from phonopy import Phonopy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import td_phonon as tdp
from phonon_accel.phonons import ase_to_phonopy, phonopy_to_ase


CM_PER_THz = 33.356


def load_snapshots(path: Path, ideal):
    with np.load(path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=float)
        forces = np.asarray(data["forces"], dtype=float)
        energies = (
            np.asarray(data["energies"], dtype=float)
            if "energies" in data.files
            else np.full(len(positions), np.nan)
        )
    if positions.shape != forces.shape or positions.shape[1:] != (len(ideal), 3):
        raise ValueError(
            f"snapshot shape mismatch: positions={positions.shape}, forces={forces.shape}, "
            f"ideal_atoms={len(ideal)}"
        )
    snapshots = []
    for pos, force, energy in zip(positions, forces, energies):
        atoms = ideal.copy()
        atoms.set_positions(pos)
        properties = {"forces": force}
        if np.isfinite(energy):
            properties["energy"] = float(energy)
        atoms.calc = SinglePointCalculator(atoms, **properties)
        snapshots.append(atoms)
    return snapshots


def fit_subset(prim, ideal, sc_matrix, ph0, snapshots, cutoff2, name):
    fc2, rmse, _ = tdp.effective_fc2(prim, ideal, sc_matrix, snapshots, cutoff2)
    dist, freq, label_positions, labels = tdp.band_from_phonopy(ph0, fc2, npoints=60)
    indices = [int(np.argmin(np.abs(dist - position))) for position in label_positions]
    gamma = indices[1]
    k_point = indices[2]
    return {
        "subset": name,
        "n_snapshots": len(snapshots),
        "fit_rmse_meV_A": rmse * 1000.0,
        "minimum_frequency_cm-1": float(np.min(freq) * CM_PER_THz),
        "Gamma_highest_cm-1": float(freq[gamma, -1] * CM_PER_THz),
        "K_highest_cm-1": float(freq[k_point, -1] * CM_PER_THz),
        "_frequency_THz": freq,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--phonopy", type=Path, required=True)
    parser.add_argument("--supercell", default="6,6,1")
    parser.add_argument("--cutoff2", type=float, default=6.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ph_input = phonopy.load(args.phonopy, is_compact_fc=False)
    prim = phonopy_to_ase(ph_input.unitcell)
    prim.wrap()
    sc_matrix = np.diag([int(value) for value in args.supercell.split(",")])
    ph0 = Phonopy(
        ase_to_phonopy(prim),
        supercell_matrix=sc_matrix,
        primitive_matrix=np.eye(3),
    )
    ideal = phonopy_to_ase(ph0.supercell)
    ideal.wrap()
    snapshots = load_snapshots(args.snapshots, ideal)

    selections = []
    for count in (10, 20, 30, 40, 50, len(snapshots)):
        if count <= len(snapshots):
            selections.append((f"prefix_{count}", snapshots[:count]))
    if len(snapshots) >= 30:
        selections.extend(
            [
                ("first_30", snapshots[:30]),
                ("last_30", snapshots[-30:]),
                ("odd_30", snapshots[::2][:30]),
                ("even_30", snapshots[1::2][:30]),
            ]
        )

    records = [
        fit_subset(prim, ideal, sc_matrix, ph0, subset, args.cutoff2, name)
        for name, subset in selections
    ]
    reference = records[[record["subset"] for record in records].index(f"prefix_{len(snapshots)}")]
    reference_frequency = reference.pop("_frequency_THz")
    for record in records:
        frequency = record.pop("_frequency_THz", reference_frequency)
        record["band_MAE_vs_full_cm-1"] = float(
            np.mean(np.abs(frequency - reference_frequency)) * CM_PER_THz
        )
        record["band_max_vs_full_cm-1"] = float(
            np.max(np.abs(frequency - reference_frequency)) * CM_PER_THz
        )

    result = {
        "snapshots": str(args.snapshots),
        "phonopy": str(args.phonopy),
        "n_total": len(snapshots),
        "records": records,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
