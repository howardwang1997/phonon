#!/usr/bin/env python3
"""Fit a physical-FD graphene TDEP reference directly from DFT force labels."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from phonopy import Phonopy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import anomaly_locate as al  # noqa: E402
import td_common as tdc  # noqa: E402
import td_phonon as tdp  # noqa: E402
from phonon_accel.phonons import ase_to_phonopy, phonopy_to_ase  # noqa: E402

CM = 33.35641


def snapshot_index(structure) -> int:
    if "snapshot_index" in structure.info:
        return int(structure.info["snapshot_index"])
    match = re.search(r"snapshot_(\d+)", str(structure.info.get("config_type", "")))
    if not match:
        raise ValueError("cannot recover snapshot index from extxyz metadata")
    return int(match.group(1))


def atomic_savez(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", action="append", type=Path, required=True)
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--degauss", type=float, required=True)
    parser.add_argument("--a", type=float, default=2.4600000087)
    parser.add_argument("--supercell", default="6,6,1")
    parser.add_argument("--cutoff2", type=float, default=6.0)
    parser.add_argument("--npoints", type=int, default=201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    structures = []
    for source in args.labels:
        loaded = read(source, index=":")
        if len(loaded) != 15:
            raise ValueError(f"expected 15 structures in {source}, found {len(loaded)}")
        for structure in loaded:
            observed_temperature = float(structure.info["lattice_temperature_K"])
            observed_degauss = float(structure.info["degauss_Ry"])
            if not np.isclose(observed_temperature, args.temperature):
                raise ValueError(f"temperature mismatch in {source}")
            if not np.isclose(observed_degauss, args.degauss, rtol=0.0, atol=1e-12):
                raise ValueError(f"degauss mismatch in {source}")
            if "REF_forces" not in structure.arrays:
                raise ValueError(f"missing REF_forces in {source}")
            evaluated = structure.copy()
            properties = {"forces": np.asarray(structure.arrays["REF_forces"], float)}
            if "REF_energy" in structure.info:
                properties["energy"] = float(structure.info["REF_energy"])
            evaluated.calc = SinglePointCalculator(evaluated, **properties)
            structures.append(evaluated)
    indices = [snapshot_index(structure) for source in args.labels for structure in read(source, index=":")]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate snapshot indices across physical-FD label waves")

    sc = tuple(int(value) for value in args.supercell.split(","))
    sc_matrix = np.diag(sc)
    primitive = tdc.build_monolayer("graphene", vacuum=7.5, a=args.a)
    primitive.wrap()
    phonon = Phonopy(
        ase_to_phonopy(primitive),
        supercell_matrix=sc_matrix,
        primitive_matrix=np.eye(3),
    )
    ideal = phonopy_to_ase(phonon.supercell)
    ideal.wrap()
    fc2, fit_rmse, n_dof = tdp.effective_fc2(
        primitive, ideal, sc_matrix, structures, args.cutoff2
    )
    distance, frequency, label_positions, labels = tdp.band_from_phonopy(
        phonon, fc2, npoints=args.npoints
    )
    records = {entry["label"]: entry for entry in al.high_sym_kinks(
        distance, frequency, label_positions, labels
    )}
    gamma_index = int(np.argmin(np.abs(distance - label_positions[1])))
    k_index = int(np.argmin(np.abs(distance - label_positions[2])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"T{args.temperature}"
    atomic_savez(
        args.output,
        temperatures=np.array([float(args.temperature)]),
        degauss_Ry=np.array(args.degauss),
        n_structures=np.array(len(structures)),
        snapshot_indices=np.asarray(sorted(indices)),
        fit_rmse_eV_A=np.array(fit_rmse),
        n_dof=np.array(n_dof),
        supercell=np.asarray(sc),
        supercell_matrix=np.asarray(sc_matrix),
        primitive_numbers=np.asarray(primitive.numbers),
        primitive_positions=np.asarray(primitive.positions),
        primitive_cell=np.asarray(primitive.cell),
        label_positions=label_positions,
        labels=labels,
        **{
            f"{prefix}_dist": distance,
            f"{prefix}_freq": frequency,
            f"{prefix}_fc2": fc2,
        },
    )
    row = {
        "temperature_K": args.temperature,
        "degauss_Ry": args.degauss,
        "n_structures": len(structures),
        "fit_RMSE_meV_A": fit_rmse * 1000.0,
        "gamma_top_cm-1": frequency[gamma_index, -1] * CM,
        "k_top_cm-1": frequency[k_index, -1] * CM,
        "gamma_kink": records[r"$\Gamma$"]["kink_strength"],
        "k_kink": records["K"]["kink_strength"],
        "min_frequency_cm-1": float(frequency.min() * CM),
    }
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
