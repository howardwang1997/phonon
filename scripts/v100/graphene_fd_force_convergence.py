"""Relabel graphene thermal snapshots with physical Fermi-Dirac broadening.

This small restartable campaign determines the supercell k mesh needed for the
DFT force labels used by the finite-temperature residual MLIP.  It deliberately
reuses geometries from the completed 300/600 K trajectories but recomputes
energy and forces; the old trajectory forces used a different smearing setup.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_snapshots(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=float)
        if "cells" in data.files:
            cells = np.asarray(data["cells"], dtype=float)
        elif "cell" in data.files:
            cell = np.asarray(data["cell"], dtype=float)
            cells = np.repeat(cell[None, :, :], len(positions), axis=0)
        else:
            raise KeyError(f"{path} has neither 'cell' nor 'cells'")
        if "numbers" in data.files:
            numbers = np.asarray(data["numbers"], dtype=int)
        else:
            numbers = np.full(positions.shape[1], 6, dtype=int)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"invalid positions shape: {positions.shape}")
    if cells.shape != (len(positions), 3, 3):
        raise ValueError(f"invalid cells shape: {cells.shape}")
    if numbers.shape != (positions.shape[1],):
        raise ValueError(f"invalid numbers shape: {numbers.shape}")
    return positions, cells, numbers


def make_espresso(args, kgrid: int, directory: Path):
    from ase.calculators.espresso import Espresso, EspressoProfile

    profile = EspressoProfile(command=args.pw, pseudo_dir=str(args.pseudo_dir))
    input_data = {
        "control": {
            "calculation": "scf",
            "tprnfor": True,
            "tstress": False,
            "disk_io": args.disk_io,
            "verbosity": "low",
        },
        "system": {
            "ecutwfc": args.ecutwfc,
            "ecutrho": args.ecutrho,
            "occupations": "smearing",
            "smearing": "fd",
            "degauss": args.degauss,
        },
        "electrons": {
            "conv_thr": 1.0e-10,
            "mixing_beta": 0.3,
            "mixing_mode": "plain",
            "electron_maxstep": 300,
            "startingwfc": "atomic",
            "diagonalization": "david",
        },
    }
    return Espresso(
        profile=profile,
        pseudopotentials={"C": "C_ONCV_PBE-1.2.upf"},
        input_data=input_data,
        kpts=(kgrid, kgrid, 1),
        directory=directory,
    )


def valid_result(path: Path, natoms: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            forces = np.asarray(data["forces"])
            energy = float(data["energy"])
        return forces.shape == (natoms, 3) and np.isfinite(forces).all() and np.isfinite(energy)
    except (OSError, KeyError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--indices", default="0,29,59")
    parser.add_argument("--validation-indices", default="")
    parser.add_argument("--kgrids", default="4,6,8")
    parser.add_argument("--degauss", type=float, required=True)
    parser.add_argument("--lattice-temperature", type=float, required=True)
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=None,
        help="optional trajectory seed used to disambiguate equal snapshot indices",
    )
    parser.add_argument("--pw", default="/root/gpupw.sh")
    parser.add_argument("--pseudo-dir", type=Path, default=Path("/root/phonon/pseudo"))
    parser.add_argument("--ecutwfc", type=float, default=60.0)
    parser.add_argument("--ecutrho", type=float, default=240.0)
    parser.add_argument(
        "--disk-io",
        choices=("none", "low", "medium", "high"),
        default="medium",
        help="QE wavefunction I/O policy; dense k meshes require medium/high",
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (args.pseudo_dir / "C_ONCV_PBE-1.2.upf").is_file():
        raise FileNotFoundError(args.pseudo_dir / "C_ONCV_PBE-1.2.upf")
    positions, cells, numbers = load_snapshots(args.snapshots)
    indices = [int(value) for value in args.indices.split(",")]
    validation_indices = {
        int(value) for value in args.validation_indices.split(",") if value.strip()
    }
    kgrids = sorted({int(value) for value in args.kgrids.split(",")})
    if not indices or not kgrids:
        raise ValueError("indices and kgrids must be non-empty")
    if min(indices) < 0 or max(indices) >= len(positions):
        raise IndexError(f"indices {indices} outside 0..{len(positions) - 1}")
    if not validation_indices.issubset(indices):
        raise ValueError("validation indices must be a subset of indices")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    completed: dict[tuple[int, int], Path] = {}
    for snapshot_index in indices:
        for kgrid in kgrids:
            label_path = args.workdir / "labels" / f"snapshot_{snapshot_index:03d}_k{kgrid}.npz"
            completed[(snapshot_index, kgrid)] = label_path
            if valid_result(label_path, len(numbers)):
                print(f"reuse snapshot={snapshot_index} k={kgrid}", flush=True)
                continue
            qe_dir = args.workdir / "qe" / f"snapshot_{snapshot_index:03d}_k{kgrid}"
            qe_dir.mkdir(parents=True, exist_ok=True)
            atoms = Atoms(
                numbers=numbers,
                positions=positions[snapshot_index],
                cell=cells[snapshot_index],
                pbc=True,
            )
            atoms.calc = make_espresso(args, kgrid, qe_dir)
            start = time.perf_counter()
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            label_arrays = dict(
                snapshot_index=snapshot_index,
                lattice_temperature_K=args.lattice_temperature,
                degauss_Ry=args.degauss,
                kgrid=kgrid,
                energy=energy,
                forces=forces,
                positions=positions[snapshot_index],
                cell=cells[snapshot_index],
                numbers=numbers,
            )
            if args.trajectory_seed is not None:
                label_arrays["trajectory_seed"] = args.trajectory_seed
            atomic_savez(label_path, **label_arrays)
            print(
                f"done snapshot={snapshot_index} k={kgrid}: "
                f"E={energy:.10f} eV max|F|={np.abs(forces).max():.5f} eV/A "
                f"wall={time.perf_counter() - start:.1f}s",
                flush=True,
            )

    reference_k = max(kgrids)
    records = []
    training_atoms = []
    for snapshot_index in indices:
        with np.load(completed[(snapshot_index, reference_k)], allow_pickle=False) as data:
            reference_forces = np.asarray(data["forces"], dtype=float)
            reference_energy = float(data["energy"])
        atoms = Atoms(
            numbers=numbers,
            positions=positions[snapshot_index],
            cell=cells[snapshot_index],
            pbc=True,
        )
        atoms.info.update(
            REF_energy=reference_energy,
            config_type=f"fd_force_conv_snapshot_{snapshot_index:03d}",
            snapshot_index=snapshot_index,
            split="validation" if snapshot_index in validation_indices else "train",
            degauss_Ry=args.degauss,
            kgrid=reference_k,
            lattice_temperature_K=args.lattice_temperature,
        )
        if args.trajectory_seed is not None:
            atoms.info["trajectory_seed"] = args.trajectory_seed
        atoms.arrays["REF_forces"] = reference_forces
        training_atoms.append(atoms)
        for kgrid in kgrids:
            with np.load(completed[(snapshot_index, kgrid)], allow_pickle=False) as data:
                forces = np.asarray(data["forces"], dtype=float)
                energy = float(data["energy"])
            delta = forces - reference_forces
            record = {
                    "snapshot_index": snapshot_index,
                    "split": "validation" if snapshot_index in validation_indices else "train",
                    "kgrid": kgrid,
                    "reference_kgrid": reference_k,
                    "force_RMSE_meV_A": float(np.sqrt(np.mean(delta**2)) * 1000.0),
                    "force_max_abs_meV_A": float(np.max(np.abs(delta)) * 1000.0),
                    "energy_difference_meV_atom": float(
                        (energy - reference_energy) * 1000.0 / len(numbers)
                    ),
                }
            if args.trajectory_seed is not None:
                record["trajectory_seed"] = args.trajectory_seed
            records.append(record)

    payload = {
        "snapshots": str(args.snapshots),
        "trajectory_seed": args.trajectory_seed,
        "indices": indices,
        "training_indices": [value for value in indices if value not in validation_indices],
        "validation_indices": sorted(validation_indices),
        "n_atoms": int(len(numbers)),
        "lattice_temperature_K": args.lattice_temperature,
        "smearing": "fermi-dirac",
        "degauss_Ry": args.degauss,
        "ecutwfc_Ry": args.ecutwfc,
        "ecutrho_Ry": args.ecutrho,
        "disk_io": args.disk_io,
        "kgrids": kgrids,
        "reference_kgrid": reference_k,
        "records": records,
        "total_wall_seconds": time.perf_counter() - t0,
    }
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    write(args.output.with_suffix(".xyz"), training_atoms, format="extxyz")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {args.output} and {args.output.with_suffix('.xyz')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
