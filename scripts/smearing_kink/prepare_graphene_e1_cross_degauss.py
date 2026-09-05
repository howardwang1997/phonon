#!/usr/bin/env python3
"""Freeze the 15 configurations and 30 off-diagonal E1 DFT tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from ase import Atoms
from ase.io import read, write
from scipy.constants import Boltzmann, electron_volt, physical_constants


RYDBERG_EV = physical_constants["Rydberg constant times hc in eV"][0]
DEGAUSS = {
    temperature: Boltzmann * temperature / electron_volt / RYDBERG_EV
    for temperature in (300, 450, 600)
}
SNAPSHOT_SELECTION = (0, 13, 29, 42, 59)
P4_SELECTION = (
    "seed0:snapshot003",
    "seed0:snapshot117",
    "seed1:snapshot003",
    "seed1:snapshot117",
    "seed2:snapshot057",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def snapshot_index(atoms: Atoms) -> int:
    match = re.fullmatch(r"fd_force_conv_snapshot_(\d{3})", atoms.info["config_type"])
    if match is None:
        raise ValueError(f"unexpected config_type={atoms.info.get('config_type')}")
    return int(match.group(1))


def clone_for_e1(atoms: Atoms, config_id: str, source_identity: str) -> Atoms:
    cloned = atoms.copy()
    cloned.info["e1_config_id"] = config_id
    cloned.info["e1_source_identity"] = source_identity
    cloned.info["diagonal_degauss_Ry"] = float(atoms.info["degauss_Ry"])
    if "REF_forces" not in cloned.arrays:
        raise ValueError(f"{source_identity} does not contain REF_forces")
    if len(cloned) != 72:
        raise ValueError(f"{source_identity} has {len(cloned)} atoms, expected 72")
    return cloned


def select_snapshot_frames(path: Path, temperature: int) -> list[Atoms]:
    frames = read(path, ":")
    by_index = {snapshot_index(frame): frame for frame in frames}
    if set(SNAPSHOT_SELECTION) - set(by_index):
        raise ValueError(f"{path} is missing selected snapshot indices")
    result = []
    for index in SNAPSHOT_SELECTION:
        frame = by_index[index]
        if int(round(float(frame.info["lattice_temperature_K"]))) != temperature:
            raise ValueError(f"temperature mismatch in {path} index {index}")
        # Existing diagonal labels stored the same formula rounded to 10 digits.
        if abs(float(frame.info["degauss_Ry"]) - DEGAUSS[temperature]) > 1.0e-9:
            raise ValueError(f"degauss mismatch in {path} index {index}")
        result.append(
            clone_for_e1(
                frame,
                f"T{temperature}_snapshot{index:03d}",
                f"T{temperature}:snapshot{index:03d}",
            )
        )
    return result


def select_p4_frames(path: Path) -> list[Atoms]:
    frames = read(path, ":")
    by_key = {frame.info["p4_key"]: frame for frame in frames}
    if set(P4_SELECTION) - set(by_key):
        raise ValueError(f"{path} is missing selected P4 identities")
    result = []
    for key in P4_SELECTION:
        frame = by_key[key]
        if int(round(float(frame.info["lattice_temperature_K"]))) != 450:
            raise ValueError(f"temperature mismatch for {key}")
        if abs(float(frame.info["degauss_Ry"]) - DEGAUSS[450]) > 1.0e-9:
            raise ValueError(f"degauss mismatch for {key}")
        result.append(clone_for_e1(frame, f"T450_{key.replace(':', '_')}", key))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-300", type=Path, required=True)
    parser.add_argument("--labels-450", type=Path, required=True)
    parser.add_argument("--labels-600", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames = [
        *select_snapshot_frames(args.labels_300, 300),
        *select_p4_frames(args.labels_450),
        *select_snapshot_frames(args.labels_600, 600),
    ]
    if len(frames) != 15 or len({frame.info["e1_config_id"] for frame in frames}) != 15:
        raise ValueError("E1 selection must contain 15 unique configurations")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs_path = args.output_dir / "e1_configs.xyz"
    temporary_configs = configs_path.with_name(configs_path.name + ".tmp")
    write(temporary_configs, frames, format="extxyz")
    os.replace(temporary_configs, configs_path)

    tasks = []
    for target_temperature, target_degauss in DEGAUSS.items():
        candidates = [
            frame
            for frame in frames
            if int(round(float(frame.info["lattice_temperature_K"]))) != target_temperature
        ]
        candidates.sort(key=lambda frame: frame.info["e1_config_id"])
        if len(candidates) != 10:
            raise ValueError("each target degauss must have 10 off-diagonal tasks")
        for index, frame in enumerate(candidates):
            config_id = frame.info["e1_config_id"]
            tasks.append(
                {
                    "task_id": f"{config_id}_dgT{target_temperature}",
                    "config_id": config_id,
                    "lattice_temperature_K": int(
                        round(float(frame.info["lattice_temperature_K"]))
                    ),
                    "target_degauss_label": f"d{target_temperature}",
                    "target_degauss_Ry": target_degauss,
                    "lane": "A" if index % 2 == 0 else "B",
                }
            )
    if len(tasks) != 30:
        raise ValueError("E1 must contain exactly 30 off-diagonal tasks")

    configurations = []
    for frame in frames:
        configurations.append(
            {
                "config_id": frame.info["e1_config_id"],
                "source_identity": frame.info["e1_source_identity"],
                "lattice_temperature_K": int(
                    round(float(frame.info["lattice_temperature_K"]))
                ),
                "diagonal_degauss_Ry": float(frame.info["diagonal_degauss_Ry"]),
                "n_atoms": len(frame),
            }
        )
    lane_counts = {
        lane: sum(task["lane"] == lane for task in tasks) for lane in ("A", "B")
    }
    target_counts = {
        f"{degauss:.10f}": {
            lane: sum(
                task["lane"] == lane
                and abs(task["target_degauss_Ry"] - degauss) < 1.0e-12
                for task in tasks
            )
            for lane in ("A", "B")
        }
        for degauss in DEGAUSS.values()
    }
    if lane_counts != {"A": 15, "B": 15}:
        raise ValueError(f"unbalanced lanes: {lane_counts}")
    if any(counts != {"A": 5, "B": 5} for counts in target_counts.values()):
        raise ValueError(f"unbalanced target degauss values: {target_counts}")

    payload = {
        "status": "frozen_not_released",
        "scope": "E1 cross-degauss force pilot; development temperatures only",
        "release_condition": (
            "E0 complete Cartesian Hermitian operator passes q-to-real-to-q replay, "
            "symmetry, ASR, and analytic-gradient gates"
        ),
        "locked_validation_temperatures_K_not_accessed": [375, 525],
        "degauss_formula": "smearing/degauss (Ry) = k_B*T/Ry",
        "degauss_Ry": DEGAUSS,
        "selection": {
            "T300_snapshot_indices": list(SNAPSHOT_SELECTION),
            "T450_p4_keys": list(P4_SELECTION),
            "T600_snapshot_indices": list(SNAPSHOT_SELECTION),
        },
        "configurations": configurations,
        "tasks": tasks,
        "lane_counts": lane_counts,
        "target_degauss_lane_counts": target_counts,
        "dft": {
            "kgrid": [8, 8, 1],
            "ecutwfc_Ry": 60.0,
            "ecutrho_Ry": 240.0,
            "smearing": "fermi-dirac",
        },
        "force_residual_gate": {
            "residual_RMSE_meV_A_max": 10.0,
            "relative_to_raw_max": 0.20,
        },
        "configs": {"path": str(configs_path), "sha256": sha256(configs_path)},
        "sources": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (args.labels_300, args.labels_450, args.labels_600)
        ],
    }
    atomic_json(args.output_dir / "e1_manifest.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
