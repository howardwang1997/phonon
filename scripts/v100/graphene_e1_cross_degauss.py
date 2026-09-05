#!/usr/bin/env python3
"""Run one restartable V100 lane of the E1 cross-degauss DFT pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from ase.io import read
from scipy.constants import Boltzmann, electron_volt, physical_constants


RYDBERG_EV = physical_constants["Rydberg constant times hc in eV"][0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def make_espresso(args, degauss: float, directory: Path):
    from ase.calculators.espresso import Espresso, EspressoProfile

    profile = EspressoProfile(command=args.pw, pseudo_dir=str(args.pseudo_dir))
    return Espresso(
        profile=profile,
        pseudopotentials={"C": "C_ONCV_PBE-1.2.upf"},
        input_data={
            "control": {
                "calculation": "scf",
                "tprnfor": True,
                "tstress": False,
                "disk_io": "none",
                "verbosity": "low",
            },
            "system": {
                "ecutwfc": 60.0,
                "ecutrho": 240.0,
                "occupations": "smearing",
                "smearing": "fd",
                "degauss": degauss,
            },
            "electrons": {
                "conv_thr": 1.0e-10,
                "mixing_beta": 0.3,
                "mixing_mode": "plain",
                "electron_maxstep": 300,
                "startingwfc": "atomic",
                "diagonalization": "david",
            },
        },
        kpts=(8, 8, 1),
        directory=directory,
    )


def valid_result(path: Path, task: dict, natoms: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            forces = np.asarray(data["forces"], float)
            energy = float(data["energy"])
            config_id = str(data["config_id"])
            degauss = float(data["degauss_Ry"])
        return (
            forces.shape == (natoms, 3)
            and np.isfinite(forces).all()
            and np.isfinite(energy)
            and config_id == task["config_id"]
            and abs(degauss - task["target_degauss_Ry"]) < 1.0e-12
        )
    except (OSError, KeyError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--lane", choices=("A", "B"), required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--pw", default="/root/gpupw.sh")
    parser.add_argument("--pseudo-dir", type=Path, default=Path("/root/phonon/pseudo"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest["status"] != "frozen_not_released":
        raise ValueError("unexpected E1 manifest state")
    if manifest.get("degauss_formula") != "smearing/degauss (Ry) = k_B*T/Ry":
        raise ValueError("E1 manifest is missing the physical degauss formula")
    for temperature in (300, 450, 600):
        expected = Boltzmann * temperature / electron_volt / RYDBERG_EV
        stored = float(manifest["degauss_Ry"][str(temperature)])
        if abs(stored - expected) > 1.0e-12:
            raise ValueError(f"E1 degauss formula mismatch at {temperature} K")
    if sha256(args.configs) != manifest["configs"]["sha256"]:
        raise ValueError("E1 configs hash does not match the frozen manifest")
    if not (args.pseudo_dir / "C_ONCV_PBE-1.2.upf").is_file():
        raise FileNotFoundError(args.pseudo_dir / "C_ONCV_PBE-1.2.upf")

    frames = read(args.configs, ":")
    by_id = {frame.info["e1_config_id"]: frame for frame in frames}
    tasks = [task for task in manifest["tasks"] if task["lane"] == args.lane]
    if len(tasks) != 15:
        raise ValueError(f"lane {args.lane} expected 15 tasks, found {len(tasks)}")
    args.workdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    for task in tasks:
        atoms = by_id[task["config_id"]].copy()
        label_path = args.workdir / "labels" / f"{task['task_id']}.npz"
        if not valid_result(label_path, task, len(atoms)):
            qe_dir = args.workdir / "qe" / task["task_id"]
            qe_dir.mkdir(parents=True, exist_ok=True)
            atoms.calc = make_espresso(args, task["target_degauss_Ry"], qe_dir)
            task_started = time.perf_counter()
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), float)
            atomic_savez(
                label_path,
                task_id=task["task_id"],
                config_id=task["config_id"],
                lane=args.lane,
                lattice_temperature_K=task["lattice_temperature_K"],
                degauss_Ry=task["target_degauss_Ry"],
                kgrid=8,
                energy=energy,
                forces=forces,
                positions=atoms.positions,
                cell=atoms.cell.array,
                numbers=atoms.numbers,
            )
            print(
                f"done {task['task_id']} E={energy:.10f} eV "
                f"max|F|={np.abs(forces).max():.6f} eV/A "
                f"wall={time.perf_counter() - task_started:.1f}s",
                flush=True,
            )
        else:
            print(f"reuse {task['task_id']}", flush=True)
        with np.load(label_path, allow_pickle=False) as data:
            records.append(
                {
                    **task,
                    "label": str(label_path),
                    "energy_eV": float(data["energy"]),
                    "max_abs_force_eV_A": float(np.max(np.abs(data["forces"]))),
                }
            )

    payload = {
        "status": "complete",
        "lane": args.lane,
        "n_tasks": len(tasks),
        "records": records,
        "wall_seconds": time.perf_counter() - started,
        "manifest_sha256": sha256(args.manifest),
        "configs_sha256": sha256(args.configs),
    }
    atomic_json(args.workdir / "summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
