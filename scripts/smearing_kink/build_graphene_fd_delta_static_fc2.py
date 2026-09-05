#!/usr/bin/env python3
"""Build the static short-range FC2 of frozen v11 + frozen delta MACE."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.calculators.calculator import all_changes
from ase.calculators.mixing import SumCalculator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
import td_common as tdc  # noqa: E402
from friedel_calc import fc2_from_calc  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CompatibleSumCalculator(SumCalculator):
    """Give ASE's sum calculator the optional system_changes API used by MACE."""

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        return super().calculate(atoms, properties, system_changes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--delta-model", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    phonon = fm.load_ph(args.background)
    base = tdc.get_mace_calc(str(args.base_model), device=args.device)
    delta = tdc.get_mace_calc(str(args.delta_model), device=args.device)
    calculator = CompatibleSumCalculator([base, delta])
    rebuilt, force_constants = fc2_from_calc(
        phonon,
        calculator,
        distance=args.distance,
        subtract_ref=True,
    )
    if force_constants.ndim != 4 or not np.isfinite(force_constants).all():
        raise ValueError("invalid finite-displacement force constants")
    rebuilt.symmetrize_force_constants()
    force_constants = np.asarray(rebuilt.force_constants, float)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    with temporary_output.open("wb") as handle:
        np.savez(
            handle,
            force_constants=force_constants,
            supercell_matrix=np.asarray(rebuilt.supercell_matrix, int),
            primitive_matrix=np.asarray(rebuilt.primitive_matrix, float),
            unitcell_cell=np.asarray(rebuilt.unitcell.cell, float),
            unitcell_positions=np.asarray(rebuilt.unitcell.positions, float),
            unitcell_numbers=np.asarray(rebuilt.unitcell.numbers, int),
            displacement_A=np.array(args.distance),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_output, args.output)
    payload = {
        "status": "complete",
        "role": "static short-range base + delta FC2 for q-space LR fitting",
        "base_model": str(args.base_model),
        "base_model_sha256": sha256(args.base_model),
        "delta_model": str(args.delta_model),
        "delta_model_sha256": sha256(args.delta_model),
        "background": str(args.background),
        "background_sha256": sha256(args.background),
        "device": args.device,
        "finite_displacement_A": args.distance,
        "force_constants_shape": list(force_constants.shape),
        "force_constants_pair_symmetry_max_abs_eV_A2": float(
            np.max(np.abs(force_constants - force_constants.transpose(1, 0, 3, 2)))
        ),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_name(args.manifest.name + ".tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary_manifest, args.manifest)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
