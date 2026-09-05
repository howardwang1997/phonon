#!/usr/bin/env python3
"""Refit one graphene TDEP spectrum from one or more completed MD checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import anomaly_locate as al  # noqa: E402
import friedel_module as fm  # noqa: E402
import td_phonon as tdp  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402


CM = 33.35641


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subtract_operator(snapshots, ideal, path: Path):
    with np.load(path, allow_pickle=False) as data:
        force_constants = np.asarray(data["delta_fc_full"], float)
        reference_positions = np.asarray(data["reference_positions"], float)
        operator_cell = np.asarray(data["cell"], float)
    if force_constants.shape != (len(ideal), len(ideal), 3, 3):
        raise ValueError("long-range operator shape does not match TDEP supercell")
    if not np.allclose(operator_cell, np.asarray(ideal.cell), atol=2e-5, rtol=0.0):
        raise ValueError("long-range operator cell does not match TDEP supercell")
    if not np.allclose(
        reference_positions, np.asarray(ideal.positions), atol=2e-5, rtol=0.0
    ):
        raise ValueError("long-range operator atom order does not match TDEP supercell")
    inverse_cell = np.linalg.inv(operator_cell)
    output = []
    long_forces = []
    for snapshot in snapshots:
        displacement = np.asarray(snapshot.positions, float) - reference_positions
        fractional = displacement @ inverse_cell
        fractional -= np.round(fractional)
        displacement = fractional @ operator_cell
        force_long = -np.einsum("ijab,jb->ia", force_constants, displacement)
        if np.linalg.norm(force_long.sum(axis=0)) > 2e-5:
            raise ValueError("long-range force violates translation invariance")
        result = snapshot.copy()
        result.calc = SinglePointCalculator(
            result, forces=np.asarray(snapshot.get_forces(), float) - force_long
        )
        output.append(result)
        long_forces.append(force_long.reshape(-1))
    joined = np.concatenate(long_forces)
    return output, {
        "path": str(path),
        "sha256": sha256(path),
        "force_RMSE_meV_A": float(np.sqrt(np.mean(joined**2)) * 1000.0),
        "force_max_abs_meV_A": float(np.max(np.abs(joined)) * 1000.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--subtract-operator",
        type=Path,
        help="subtract this exact harmonic force before fitting the TDEP FC2",
    )
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--cutoff2", type=float, default=6.0)
    parser.add_argument("--npoints", type=int, default=201)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phonon = fm.load_ph(args.background)
    primitive = phonopy_to_ase(phonon.unitcell)
    primitive.wrap()
    ideal = phonopy_to_ase(phonon.supercell)
    ideal.wrap()
    snapshots = []
    counts = []
    for raw_path in args.checkpoint:
        path = raw_path / "snapshots.npz" if raw_path.is_dir() else raw_path
        loaded = tdp._load_checkpoint_snapshots(path, ideal)
        if not loaded:
            raise ValueError(f"no snapshots found in {path}")
        snapshots.extend(loaded)
        counts.append(len(loaded))
    temperatures = np.asarray([snapshot.get_temperature() for snapshot in snapshots], float)
    operator = None
    if args.subtract_operator is not None:
        snapshots, operator = subtract_operator(snapshots, ideal, args.subtract_operator)
    fc2, rmse, ndof = tdp.effective_fc2(
        primitive,
        ideal,
        phonon.supercell_matrix,
        snapshots,
        args.cutoff2,
    )
    distance, frequency, label_positions, labels = tdp.band_from_phonopy(
        phonon, fc2, npoints=args.npoints
    )
    gamma = al.branch_freq_at_label(distance, frequency, label_positions, labels, r"$\Gamma$") * CM
    kpoint = al.branch_freq_at_label(distance, frequency, label_positions, labels, "K") * CM
    prefix = f"T{args.temperature}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        temperatures=np.array([args.temperature], float),
        tag=np.array(args.tag),
        source_checkpoint_counts=np.asarray(counts, int),
        subtracted_operator_path=np.array(operator["path"] if operator else ""),
        subtracted_operator_sha256=np.array(operator["sha256"] if operator else ""),
        subtracted_operator_force_RMSE_meV_A=np.array(
            operator["force_RMSE_meV_A"] if operator else 0.0
        ),
        subtracted_operator_force_max_abs_meV_A=np.array(
            operator["force_max_abs_meV_A"] if operator else 0.0
        ),
        **{
            f"{prefix}_dist": distance,
            f"{prefix}_freq": frequency,
            f"{prefix}_fc2": fc2,
            f"{prefix}_instantaneous_temperatures_K": temperatures,
            f"{prefix}_mean_temperature_K": np.array(float(np.mean(temperatures))),
            f"{prefix}_max_temperature_K": np.array(float(np.max(temperatures))),
            f"{prefix}_fit_rmse_meV_A": np.array(float(rmse * 1000.0)),
            "label_positions": label_positions,
            "labels": labels,
        },
    )
    print(
        f"tag={args.tag} checkpoints={counts} snapshots={len(snapshots)} "
        f"operator_subtracted={args.subtract_operator is not None} "
        f"mean_T={np.mean(temperatures):.2f} K fit_RMSE={rmse * 1000.0:.2f} meV/A "
        f"Gamma_top={gamma:.2f} cm^-1 K_top={kpoint:.2f} cm^-1 ndof={ndof}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
