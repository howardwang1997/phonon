#!/usr/bin/env python3
"""Evaluate thermal-model variants on fixed thermal and harmonic holdouts."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("values must use LABEL=/path")
    label, path = specification.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("values must use LABEL=/path")
    return label, Path(path)


def evaluate(calculator, structures) -> dict:
    deltas = []
    records = []
    for index, structure in enumerate(structures):
        reference = np.asarray(structure.arrays["REF_forces"], dtype=float)
        atoms = structure.copy()
        atoms.calc = calculator
        predicted = np.asarray(atoms.get_forces(), dtype=float)
        delta = predicted - reference
        deltas.append(delta.reshape(-1))
        snapshot_index = structure.info.get("snapshot_index")
        records.append(
            {
                "structure_index": index,
                "snapshot_index": (
                    int(snapshot_index) if snapshot_index is not None else None
                ),
                "n_atoms": len(structure),
                "RMSE_meV_A": float(np.sqrt(np.mean(delta**2)) * 1000.0),
                "max_abs_meV_A": float(np.max(np.abs(delta)) * 1000.0),
            }
        )
    flattened = np.concatenate(deltas)
    return {
        "n_structures": len(structures),
        "n_force_components": len(flattened),
        "RMSE_meV_A": float(np.sqrt(np.mean(flattened**2)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(flattened)) * 1000.0),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=labelled_path, required=True)
    parser.add_argument("--model", action="append", type=labelled_path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    datasets = {}
    for label, path in args.dataset:
        structures = read(path, index=":")
        if not structures:
            raise ValueError(f"empty evaluation dataset: {path}")
        if any("REF_forces" not in structure.arrays for structure in structures):
            raise ValueError(f"missing REF_forces in {path}")
        datasets[label] = {"path": str(path), "structures": structures}

    payload = {
        "units": {"force": "meV/angstrom"},
        "datasets": {
            label: {"path": entry["path"], "n_structures": len(entry["structures"])}
            for label, entry in datasets.items()
        },
        "models": {},
    }
    for label, model_path in args.model:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        calculator = MACECalculator(
            model_paths=str(model_path),
            device=args.device,
            default_dtype="float32",
        )
        payload["models"][label] = {
            "path": str(model_path),
            "sha256": sha256(model_path),
            "datasets": {
                dataset_label: evaluate(calculator, entry["structures"])
                for dataset_label, entry in datasets.items()
            },
        }
        del calculator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
