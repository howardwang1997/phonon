#!/usr/bin/env python3
"""Project a delta model's equilibrium Hessian to recover harmonic retention."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator
from phonopy import Phonopy
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
from friedel_calc import fc2_from_calc  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def force_metrics(values: list[np.ndarray]) -> dict[str, float]:
    joined = np.concatenate([value.reshape(-1) for value in values])
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(joined**2)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(joined)) * 1000.0),
    }


def minimum_image_vectors(left, right, cell):
    delta = left[:, None, :] - right[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def mapping_and_displacement(structure, ideal):
    cell = np.asarray(structure.cell, float)
    if len(structure) != len(ideal) or not np.allclose(
        cell, np.asarray(ideal.cell), atol=2e-5, rtol=0.0
    ):
        raise ValueError("dataset structure and FC2 reference supercell differ")
    vectors = minimum_image_vectors(
        np.asarray(structure.positions, float), np.asarray(ideal.positions, float), cell
    )
    costs = np.linalg.norm(vectors, axis=2)
    row, column = linear_sum_assignment(costs)
    if not np.array_equal(row, np.arange(len(structure))):
        column = column[np.argsort(row)]
    if not np.array_equal(
        np.asarray(structure.numbers), np.asarray(ideal.numbers)[column]
    ):
        raise ValueError("atom mapping changes chemical species")
    reference = np.asarray(ideal.positions, float)[column]
    displacement = np.asarray(structure.positions, float) - reference
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return column.astype(int), fractional @ cell


def reference(structure):
    short = np.asarray(
        structure.arrays.get("SHORT_RANGE_TARGET_forces", structure.arrays["REF_forces"]),
        float,
    )
    total = np.asarray(structure.arrays.get("TOTAL_forces", short), float)
    long_range = np.asarray(
        structure.arrays.get("LONG_RANGE_forces", np.zeros_like(short)), float
    )
    return short, total, long_range


def parse_grid(specification: str) -> list[float]:
    values = specification.split(":")
    if len(values) != 3:
        raise ValueError("--alpha-grid must use START:STOP:STEP")
    start, stop, step = (float(value) for value in values)
    if step <= 0 or stop < start:
        raise ValueError("invalid --alpha-grid")
    count = int(np.floor((stop - start) / step + 0.5))
    return [float(start + index * step) for index in range(count + 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--delta-model", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--harmonic", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--alpha-grid", default="0:1:0.05")
    parser.add_argument("--force-rmse-threshold", type=float, default=50.0)
    parser.add_argument("--force-max-threshold", type=float, default=250.0)
    parser.add_argument("--harmonic-rmse-ratio", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adjustment-output", type=Path, required=True)
    args = parser.parse_args()

    template_phonon = fm.load_ph(args.background)
    base = MACECalculator(
        model_paths=str(args.base_model), device=args.device, default_dtype="float32"
    )
    delta = MACECalculator(
        model_paths=str(args.delta_model), device=args.device, default_dtype="float32"
    )
    datasets = {
        "thermal": read(args.thermal, index=":"),
        "harmonic": read(args.harmonic, index=":"),
    }
    hessian_cache = {}

    def build_hessian(natoms: int):
        if natoms % len(template_phonon.unitcell) != 0:
            raise ValueError(f"cannot infer graphene supercell from {natoms} atoms")
        repeat = int(round(np.sqrt(natoms / len(template_phonon.unitcell))))
        if len(template_phonon.unitcell) * repeat * repeat != natoms:
            raise ValueError(f"expected an n x n graphene supercell, found {natoms} atoms")
        if natoms == len(template_phonon.supercell):
            phonon = template_phonon
        else:
            phonon = Phonopy(
                template_phonon.unitcell,
                supercell_matrix=np.diag((repeat, repeat, 1)),
                primitive_matrix=template_phonon.primitive_matrix,
            )
        rebuilt, force_constants = fc2_from_calc(
            phonon, delta, distance=args.distance, subtract_ref=True
        )
        rebuilt.symmetrize_force_constants()
        force_constants = np.asarray(rebuilt.force_constants, float)
        ideal = phonopy_to_ase(rebuilt.supercell)
        ideal.wrap()
        reference_atoms = ideal.copy()
        reference_atoms.calc = delta
        reference_force = np.asarray(reference_atoms.get_forces(), float)
        return {
            "repeat": repeat,
            "phonon": rebuilt,
            "force_constants": force_constants,
            "ideal": ideal,
            "reference_force": reference_force,
            "pair_symmetry_max_abs_eV_A2": float(
                np.max(
                    np.abs(
                        force_constants
                        - force_constants.transpose(1, 0, 3, 2)
                    )
                )
            ),
        }

    for structures in datasets.values():
        for natoms in sorted({len(structure) for structure in structures}):
            if natoms not in hessian_cache:
                hessian_cache[natoms] = build_hessian(natoms)

    cached = {}
    for label, structures in datasets.items():
        records = []
        for structure in structures:
            hessian = hessian_cache[len(structure)]
            ideal = hessian["ideal"]
            delta_fc = hessian["force_constants"]
            reference_delta_force = hessian["reference_force"]
            atoms = structure.copy()
            atoms.calc = base
            base_force = np.asarray(atoms.get_forces(), float)
            atoms.calc = delta
            delta_force = np.asarray(atoms.get_forces(), float)
            mapping, displacement = mapping_and_displacement(structure, ideal)
            reordered_fc = delta_fc[mapping][:, mapping]
            harmonic_delta_force = -np.einsum(
                "ijab,jb->ia", reordered_fc, displacement
            )
            short, total, long_range = reference(structure)
            records.append(
                {
                    "base_force": base_force,
                    "delta_force": delta_force,
                    "harmonic_delta_force": harmonic_delta_force,
                    "reference_delta_force": reference_delta_force[mapping],
                    "short": short,
                    "total": total,
                    "long_range": long_range,
                }
            )
        cached[label] = records

    base_harmonic_errors = [
        item["base_force"] + item["long_range"] - item["total"]
        for item in cached["harmonic"]
    ]
    base_harmonic = force_metrics(base_harmonic_errors)
    harmonic_limit = args.harmonic_rmse_ratio * base_harmonic["RMSE_meV_A"]
    candidates = []
    for centered in (False, True):
        for alpha in parse_grid(args.alpha_grid):
            by_dataset = {}
            for label, records in cached.items():
                errors = []
                for item in records:
                    adjusted_delta = (
                        item["delta_force"]
                        - alpha * item["harmonic_delta_force"]
                        - (item["reference_delta_force"] if centered else 0.0)
                    )
                    predicted_total = (
                        item["base_force"] + adjusted_delta + item["long_range"]
                    )
                    errors.append(predicted_total - item["total"])
                by_dataset[label] = force_metrics(errors)
            ratios = {
                "thermal_RMSE": by_dataset["thermal"]["RMSE_meV_A"]
                / args.force_rmse_threshold,
                "thermal_max_abs": by_dataset["thermal"]["max_abs_meV_A"]
                / args.force_max_threshold,
                "harmonic_RMSE": by_dataset["harmonic"]["RMSE_meV_A"]
                / harmonic_limit,
            }
            candidates.append(
                {
                    "alpha": alpha,
                    "subtract_reference_delta_force": centered,
                    "passes_fixed_gate": all(value <= 1.0 for value in ratios.values()),
                    "normalized_gate_ratios": ratios,
                    "normalized_worst_gate_score": max(ratios.values()),
                    "thermal_total_force": by_dataset["thermal"],
                    "harmonic_combined_force": by_dataset["harmonic"],
                }
            )
    passing = [item for item in candidates if item["passes_fixed_gate"]]
    pool = passing if passing else candidates
    selected = min(pool, key=lambda item: item["normalized_worst_gate_score"])
    deployment_hessian = hessian_cache[len(datasets["thermal"][0])]
    deployment_delta_fc = deployment_hessian["force_constants"]
    deployment_reference_force = deployment_hessian["reference_force"]
    deployment_ideal = deployment_hessian["ideal"]
    constant_force = (
        -deployment_reference_force
        if selected["subtract_reference_delta_force"]
        else np.zeros_like(deployment_reference_force)
    )
    args.adjustment_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_adjustment = args.adjustment_output.with_name(
        args.adjustment_output.name + ".tmp"
    )
    with temporary_adjustment.open("wb") as handle:
        np.savez(
            handle,
            correction_fc_full=-float(selected["alpha"]) * deployment_delta_fc,
            constant_force=constant_force,
            reference_positions=np.asarray(deployment_ideal.positions, float),
            cell=np.asarray(deployment_ideal.cell, float),
            alpha=np.array(selected["alpha"]),
            subtract_reference_delta_force=np.array(
                selected["subtract_reference_delta_force"]
            ),
            source_delta_model_sha256=np.array(sha256(args.delta_model)),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_adjustment, args.adjustment_output)
    result = {
        "status": "projection_passed" if passing else "no_projection_passed",
        "method": (
            "subtract a frozen fraction of the delta model's equilibrium Hessian; "
            "optional reference-force centering is an affine conservative correction"
        ),
        "fixed_thresholds": {
            "thermal_force_RMSE_meV_A": args.force_rmse_threshold,
            "thermal_force_max_abs_meV_A": args.force_max_threshold,
            "harmonic_RMSE_relative_to_frozen_v11": args.harmonic_rmse_ratio,
            "harmonic_RMSE_meV_A": harmonic_limit,
        },
        "base_model": str(args.base_model),
        "base_model_sha256": sha256(args.base_model),
        "delta_model": str(args.delta_model),
        "delta_model_sha256": sha256(args.delta_model),
        "background": str(args.background),
        "background_sha256": sha256(args.background),
        "thermal_dataset": str(args.thermal),
        "harmonic_dataset": str(args.harmonic),
        "finite_displacement_A": args.distance,
        "hessian_supercells": {
            str(natoms): {
                "repeat": hessian["repeat"],
                "delta_reference_force": force_metrics(
                    [hessian["reference_force"]]
                ),
                "delta_reference_net_force_meV_A": float(
                    np.linalg.norm(hessian["reference_force"].sum(axis=0))
                    * 1000.0
                ),
                "delta_fc_pair_symmetry_max_abs_eV_A2": hessian[
                    "pair_symmetry_max_abs_eV_A2"
                ],
            }
            for natoms, hessian in hessian_cache.items()
        },
        "harmonic_v11_force": base_harmonic,
        "selected": selected,
        "adjustment_output": str(args.adjustment_output),
        "adjustment_output_sha256": sha256(args.adjustment_output),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    temporary_output.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary_output, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))
    del base, delta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
