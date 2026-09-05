#!/usr/bin/env python3
"""Deterministic smoke test for the phonopy/CellConstructor q6 atom mapping."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
from audit_graphene_current_s0_fixed_smearing import sha256  # noqa: E402
from run_graphene_physical_q0_sscha import (  # noqa: E402
    operator_fc,
    periodic_reference_max_distance,
    phonopy_fc_to_cc_dyn,
)
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402


def harmonic_energy_force(fc: np.ndarray, displacement: np.ndarray):
    force = -np.einsum("ijab,jb->ia", fc, displacement)
    energy = 0.5 * float(np.einsum("ia,ijab,jb->", displacement, fc, displacement))
    return energy, force


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--operator-temperature", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("run_graphene_physical_q0_sscha.py")
    if sha256(runner) != freeze["scripts"]["q0_sscha"]["sha256"]:
        raise ValueError("runner hash differs from corrected freeze manifest")
    background_path = Path(freeze["source_L0"]["background"]["path"])
    if sha256(background_path) != freeze["source_L0"]["background"]["sha256"]:
        raise ValueError("background hash differs from freeze manifest")
    phonon = fm.load_ph(background_path)
    (
        operator_path,
        operator_phonopy,
        operator_reference,
        operator_cell,
        operator_atom_mapping,
    ) = operator_fc(freeze, args.operator_temperature)
    (
        _,
        _,
        phonopy_for_cc,
        cc_for_phonopy,
        cc_reference,
    ) = phonopy_fc_to_cc_dyn(phonon, operator_phonopy)
    phonopy_reference = phonopy_to_ase(phonon.supercell)
    operator_reference_mismatch = periodic_reference_max_distance(
        operator_reference, np.asarray(phonopy_reference.positions), operator_cell
    )
    cc_reference_mismatch = periodic_reference_max_distance(
        np.asarray(phonopy_reference.positions)[phonopy_for_cc],
        np.asarray(cc_reference.positions),
        operator_cell,
    )

    rng = np.random.default_rng(args.seed)
    displacement_phonopy = rng.normal(0.0, 0.03, size=(len(phonopy_reference), 3))
    displacement_phonopy -= np.mean(displacement_phonopy, axis=0, keepdims=True)
    displacement_cc = displacement_phonopy[phonopy_for_cc]
    operator_cc = operator_phonopy[phonopy_for_cc][:, phonopy_for_cc]
    energy_phonopy, force_phonopy = harmonic_energy_force(
        operator_phonopy, displacement_phonopy
    )
    energy_cc, force_cc = harmonic_energy_force(operator_cc, displacement_cc)
    force_cc_returned_to_phonopy = force_cc[cc_for_phonopy]

    finite_difference_step = 1.0e-6
    atom = 17
    component = 1
    displaced_plus = displacement_cc.copy()
    displaced_minus = displacement_cc.copy()
    displaced_plus[atom, component] += finite_difference_step
    displaced_minus[atom, component] -= finite_difference_step
    energy_plus, _ = harmonic_energy_force(operator_cc, displaced_plus)
    energy_minus, _ = harmonic_energy_force(operator_cc, displaced_minus)
    finite_difference_force = -(energy_plus - energy_minus) / (
        2.0 * finite_difference_step
    )
    analytic_force = float(force_cc[atom, component])

    result = {
        "status": "pass",
        "operator_temperature_K": args.operator_temperature,
        "operator_path": str(operator_path),
        "operator_sha256": sha256(operator_path),
        "operator_atom_mapping": operator_atom_mapping.tolist(),
        "phonopy_for_CellConstructor": phonopy_for_cc.tolist(),
        "CellConstructor_for_phonopy": cc_for_phonopy.tolist(),
        "operator_reference_mismatch_A": operator_reference_mismatch,
        "CellConstructor_reference_mismatch_A": cc_reference_mismatch,
        "energy_order_difference_eV": float(energy_cc - energy_phonopy),
        "force_order_max_abs_difference_eV_A": float(
            np.max(np.abs(force_cc_returned_to_phonopy - force_phonopy))
        ),
        "finite_difference": {
            "atom_CC_order": atom,
            "component": "xyz"[component],
            "step_A": finite_difference_step,
            "analytic_force_eV_A": analytic_force,
            "finite_difference_force_eV_A": finite_difference_force,
            "absolute_difference_eV_A": abs(analytic_force - finite_difference_force),
        },
    }
    tolerances = {
        "reference_A": 2.0e-5,
        "energy_eV": 1.0e-12,
        "force_eV_A": 1.0e-12,
        "finite_difference_eV_A": 2.0e-8,
    }
    passed = (
        operator_reference_mismatch <= tolerances["reference_A"]
        and cc_reference_mismatch <= tolerances["reference_A"]
        and abs(energy_cc - energy_phonopy) <= tolerances["energy_eV"]
        and np.max(np.abs(force_cc_returned_to_phonopy - force_phonopy))
        <= tolerances["force_eV_A"]
        and abs(analytic_force - finite_difference_force)
        <= tolerances["finite_difference_eV_A"]
    )
    result["status"] = "pass" if passed else "fail"
    result["tolerances"] = tolerances
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
