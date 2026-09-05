#!/usr/bin/env python3
"""Diagnose the historical graphene SSCHA q6 atom-order interface.

The historical Q0/X0 runner passed CellConstructor-ordered structures to a
harmonic calculator whose reference positions and Hessian were in different
orders.  This script reproduces that deployed force path from frozen artifacts
and compares it with an order-safe current-model replay.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.units import create_units

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
from audit_graphene_current_s0_fixed_smearing import force_metrics, sha256  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402

RY_TO_EV = float(create_units("2006")["Ry"])


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


def historical_harmonic_force(
    force_constants: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    displacement = positions - reference
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    displacement = fractional @ cell
    return -np.einsum("ijab,jb->ia", force_constants, displacement)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-extxyz", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--model-replay", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    structures = read(args.selected_extxyz, index=":")
    manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    with np.load(args.model_replay, allow_pickle=False) as data:
        replay = {key: np.asarray(data[key]) for key in data.files}
    with np.load(args.operator, allow_pickle=False) as data:
        raw_operator = np.asarray(data["delta_fc_full"], float)
        operator_atom_mapping = np.asarray(data["atom_mapping"], int)
    phonon = fm.load_ph(args.background)
    phonopy_reference = np.asarray(phonopy_to_ase(phonon.supercell).positions, float)
    cell = np.asarray(structures[0].cell, float)

    cc_to_operator = np.asarray(
        manifest["atom_mapping"]["SSCHA_xats_row_to_operator_reference"], int
    )
    operator_to_cc = np.argsort(cc_to_operator)
    if sorted(cc_to_operator.tolist()) != list(range(len(cc_to_operator))):
        raise ValueError("SSCHA-to-operator mapping is not a permutation")
    if len(structures) != len(replay["predicted_total_forces_eV_A"]):
        raise ValueError("selected set and model replay have different sizes")

    saved_forces = []
    correct_total_forces = []
    correct_long_forces = []
    historical_long_forces = []
    reproduced_historical_forces = []
    records = []
    for index, structure in enumerate(structures):
        saved = np.asarray(structure.arrays["MLIP_forces"], float) * RY_TO_EV
        correct_total = np.asarray(replay["predicted_total_forces_eV_A"][index], float)
        correct_long = np.asarray(replay["long_range_forces_eV_A"][index], float)
        short_operator = correct_total - correct_long

        positions_operator = np.asarray(structure.positions, float)
        positions_cc = positions_operator[cc_to_operator]
        short_cc = short_operator[cc_to_operator]
        wrong_long_cc = historical_harmonic_force(
            raw_operator, phonopy_reference, cell, positions_cc
        )
        reproduced_operator = (short_cc + wrong_long_cc)[operator_to_cc]
        wrong_long_operator = wrong_long_cc[operator_to_cc]

        saved_forces.append(saved)
        correct_total_forces.append(correct_total)
        correct_long_forces.append(correct_long)
        historical_long_forces.append(wrong_long_operator)
        reproduced_historical_forces.append(reproduced_operator)
        records.append(
            {
                "sscha_index": int(structure.info["sscha_index"]),
                "historical_force_reproduction_RMSE_meV_A": force_metrics(
                    reproduced_operator - saved
                )["RMSE_meV_A"],
                "correct_model_vs_saved_RMSE_meV_A": force_metrics(
                    correct_total - saved
                )["RMSE_meV_A"],
                "intended_long_range_RMS_meV_A": force_metrics(correct_long)[
                    "RMSE_meV_A"
                ],
                "historically_applied_long_range_RMS_meV_A": force_metrics(
                    wrong_long_operator
                )["RMSE_meV_A"],
            }
        )

    saved_forces = np.asarray(saved_forces)
    correct_total_forces = np.asarray(correct_total_forces)
    correct_long_forces = np.asarray(correct_long_forces)
    historical_long_forces = np.asarray(historical_long_forces)
    reproduced_historical_forces = np.asarray(reproduced_historical_forces)
    reproduction_error = reproduced_historical_forces - saved_forces
    correct_vs_saved = correct_total_forces - saved_forces

    summary = {
        "status": "historical_X0_atom_order_bug_reproduced",
        "n_structures": len(structures),
        "finding": (
            "The saved X0 forces are reproduced by applying the raw q6 Hessian and "
            "phonopy reference elementwise to CellConstructor-ordered atoms. The "
            "historical X0 ensemble and Hessian therefore cannot be used as an "
            "order-safe current-S0 fixed-smearing distribution."
        ),
        "historical_force_reproduction_error": force_metrics(reproduction_error),
        "correct_model_vs_historical_saved_force": force_metrics(correct_vs_saved),
        "intended_long_range_force_magnitude": force_metrics(correct_long_forces),
        "historically_applied_long_range_force_magnitude": force_metrics(
            historical_long_forces
        ),
        "unit_interpretation": {
            "saved_force": "Ry/angstrom",
            "Ry_to_eV": RY_TO_EV,
        },
        "mappings": {
            "SSCHA_CC_row_to_operator_reference": cc_to_operator.tolist(),
            "operator_reference_to_SSCHA_CC_row": operator_to_cc.tolist(),
            "operator_npz_atom_mapping_not_applied_historically": operator_atom_mapping.tolist(),
        },
        "inputs": {
            "selected_extxyz": {
                "path": str(args.selected_extxyz),
                "sha256": sha256(args.selected_extxyz),
            },
            "freeze_manifest": {
                "path": str(args.freeze_manifest),
                "sha256": sha256(args.freeze_manifest),
            },
            "model_replay": {
                "path": str(args.model_replay),
                "sha256": sha256(args.model_replay),
            },
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {
                "path": str(args.background),
                "sha256": sha256(args.background),
            },
        },
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        args.output_dir / "historical_X0_atom_order_diagnostic.npz",
        saved_historical_forces_eV_A=saved_forces,
        correct_model_forces_eV_A=correct_total_forces,
        correct_long_range_forces_eV_A=correct_long_forces,
        historically_applied_long_range_forces_eV_A=historical_long_forces,
        reproduced_historical_forces_eV_A=reproduced_historical_forces,
        reproduction_error_eV_A=reproduction_error,
    )
    atomic_json(args.output_dir / "historical_X0_atom_order_diagnostic.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
