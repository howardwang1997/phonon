#!/usr/bin/env python3
"""Restartable DFT labels for a frozen current-S0 SSCHA overlap shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_snapshots(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        values = {key: np.asarray(data[key]) for key in data.files}
    required = {
        "positions",
        "cells",
        "numbers",
        "sscha_indices",
        "selection_groups",
        "MLIP_forces",
        "MLIP_energies",
    }
    missing = required - set(values)
    if missing:
        raise KeyError(f"snapshot container is missing {sorted(missing)}")
    nframes, natoms, ndim = values["positions"].shape
    if ndim != 3 or values["cells"].shape != (nframes, 3, 3):
        raise ValueError("invalid positions/cells shape")
    if values["numbers"].shape != (natoms,):
        raise ValueError("invalid numbers shape")
    if values["MLIP_forces"].shape != (nframes, natoms, 3):
        raise ValueError("invalid MLIP force shape")
    for key in ("sscha_indices", "selection_groups", "MLIP_energies"):
        if values[key].shape != (nframes,):
            raise ValueError(f"invalid {key} shape")
    return values


def make_espresso(args, directory: Path):
    from ase.calculators.espresso import Espresso, EspressoProfile

    profile = EspressoProfile(command=args.pw, pseudo_dir=str(args.pseudo_dir))
    input_data = {
        "control": {
            "calculation": "scf",
            "tprnfor": True,
            "tstress": False,
            "disk_io": "none",
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
        kpts=(args.kgrid, args.kgrid, 1),
        directory=directory,
    )


def valid_result(path: Path, natoms: int, sscha_index: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            forces = np.asarray(data["forces"], float)
            energy = float(np.asarray(data["energy"]).reshape(()))
            stored_index = int(np.asarray(data["sscha_index"]).reshape(()))
        return (
            forces.shape == (natoms, 3)
            and np.isfinite(forces).all()
            and np.isfinite(energy)
            and stored_index == sscha_index
        )
    except (OSError, KeyError, ValueError):
        return False


def force_metrics(values: np.ndarray) -> dict[str, float]:
    flattened = np.asarray(values, float).reshape(-1)
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(flattened**2)) * 1000.0),
        "MAE_meV_A": float(np.mean(np.abs(flattened)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(flattened)) * 1000.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--indices", default="0,1,2,3,4,5")
    parser.add_argument("--degauss", type=float, required=True)
    parser.add_argument("--lattice-temperature", type=float, required=True)
    parser.add_argument("--kgrid", type=int, default=8)
    parser.add_argument("--pw", default="/root/gpupw.sh")
    parser.add_argument("--pseudo-dir", type=Path, default=Path("/root/phonon/pseudo"))
    parser.add_argument("--ecutwfc", type=float, default=60.0)
    parser.add_argument("--ecutrho", type=float, default=240.0)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (args.pseudo_dir / "C_ONCV_PBE-1.2.upf").is_file():
        raise FileNotFoundError(args.pseudo_dir / "C_ONCV_PBE-1.2.upf")
    snapshots = load_snapshots(args.snapshots)
    indices = [int(value) for value in args.indices.split(",")]
    if len(indices) != 6 or len(set(indices)) != 6:
        raise ValueError("a frozen R1 shard must contain exactly six unique indices")
    if min(indices) < 0 or max(indices) >= len(snapshots["positions"]):
        raise IndexError("requested shard index is outside the snapshot container")
    if args.kgrid != 8 or abs(args.degauss - 0.0019000869) > 5.0e-11:
        raise ValueError("R1 requires k8 and degauss=0.0019000869 Ry")
    if abs(args.lattice_temperature - 450.0) > 1.0e-12:
        raise ValueError("R1 requires lattice temperature 450 K")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    start_all = time.perf_counter()
    for local_index in indices:
        sscha_index = int(snapshots["sscha_indices"][local_index])
        label_path = args.workdir / "labels" / f"sscha_{sscha_index:03d}_k8.npz"
        completed[local_index] = label_path
        if valid_result(label_path, len(snapshots["numbers"]), sscha_index):
            print(f"reuse local={local_index} sscha={sscha_index}", flush=True)
            continue
        qe_dir = args.workdir / "qe" / f"sscha_{sscha_index:03d}_k8"
        qe_dir.mkdir(parents=True, exist_ok=True)
        atoms = Atoms(
            numbers=snapshots["numbers"],
            positions=snapshots["positions"][local_index],
            cell=snapshots["cells"][local_index],
            pbc=(True, True, False),
        )
        atoms.calc = make_espresso(args, qe_dir)
        start = time.perf_counter()
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), float)
        atomic_savez(
            label_path,
            local_index=np.array(local_index),
            sscha_index=np.array(sscha_index),
            selection_group=np.array(str(snapshots["selection_groups"][local_index])),
            lattice_temperature_K=np.array(args.lattice_temperature),
            degauss_Ry=np.array(args.degauss),
            kgrid=np.array(args.kgrid),
            energy=np.array(energy),
            forces=forces,
            positions=snapshots["positions"][local_index],
            cell=snapshots["cells"][local_index],
            numbers=snapshots["numbers"],
            MLIP_energy=np.array(float(snapshots["MLIP_energies"][local_index])),
            MLIP_forces=snapshots["MLIP_forces"][local_index],
        )
        print(
            f"done local={local_index} sscha={sscha_index}: E={energy:.10f} eV "
            f"max|F|={np.max(np.abs(forces)):.6f} eV/A "
            f"wall={time.perf_counter() - start:.1f}s",
            flush=True,
        )

    structures = []
    records = []
    errors = []
    for local_index in indices:
        label_path = completed[local_index]
        with np.load(label_path, allow_pickle=False) as data:
            sscha_index = int(np.asarray(data["sscha_index"]).reshape(()))
            group = str(np.asarray(data["selection_group"]).reshape(()))
            energy = float(np.asarray(data["energy"]).reshape(()))
            forces = np.asarray(data["forces"], float)
            mlip_energy = float(np.asarray(data["MLIP_energy"]).reshape(()))
            mlip_forces = np.asarray(data["MLIP_forces"], float)
        error = mlip_forces - forces
        errors.append(error)
        atoms = Atoms(
            numbers=snapshots["numbers"],
            positions=snapshots["positions"][local_index],
            cell=snapshots["cells"][local_index],
            pbc=(True, True, False),
        )
        atoms.info.update(
            REF_energy=energy,
            MLIP_energy=mlip_energy,
            config_type=f"s0_sscha_overlap_{sscha_index:03d}",
            split="frozen_overlap_screen",
            sscha_index=sscha_index,
            overlap_local_index=local_index,
            selection_group=group,
            lattice_temperature_K=args.lattice_temperature,
            degauss_Ry=args.degauss,
            kgrid=args.kgrid,
            sscha_random_seed=453005,
        )
        atoms.arrays["REF_forces"] = forces
        atoms.arrays["MLIP_forces"] = mlip_forces
        structures.append(atoms)
        records.append(
            {
                "local_index": local_index,
                "sscha_index": sscha_index,
                "selection_group": group,
                "force_error": force_metrics(error),
                "DFT_energy_eV": energy,
                "MLIP_energy_eV": mlip_energy,
                "label_path": str(label_path),
                "label_sha256": sha256(label_path),
            }
        )

    output_xyz = args.output.with_suffix(".xyz")
    temporary_xyz = output_xyz.with_name(output_xyz.name + ".tmp")
    write(temporary_xyz, structures, format="extxyz")
    os.replace(temporary_xyz, output_xyz)
    payload = {
        "status": "complete",
        "scope": "frozen current-S0 SSCHA overlap DFT shard",
        "snapshots": {"path": str(args.snapshots), "sha256": sha256(args.snapshots)},
        "indices": indices,
        "sscha_indices": [int(snapshots["sscha_indices"][index]) for index in indices],
        "n_atoms": int(len(snapshots["numbers"])),
        "lattice_temperature_K": args.lattice_temperature,
        "smearing": "fermi-dirac",
        "degauss_Ry": args.degauss,
        "kgrid": args.kgrid,
        "ecutwfc_Ry": args.ecutwfc,
        "ecutrho_Ry": args.ecutrho,
        "disk_io": "none",
        "total_wall_seconds": time.perf_counter() - start_all,
        "provisional_saved_MLIP_vs_DFT_force": force_metrics(np.asarray(errors)),
        "records": records,
        "output_extxyz": {"path": str(output_xyz), "sha256": sha256(output_xyz)},
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
