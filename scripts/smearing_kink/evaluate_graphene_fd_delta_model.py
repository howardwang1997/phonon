#!/usr/bin/env python3
"""Evaluate frozen-v11 + delta-MACE + harmonic long-range reconstruction."""
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
        raise argparse.ArgumentTypeError("datasets must use LABEL=/path")
    label, path = specification.split("=", 1)
    return label, Path(path)


def metrics(values: list[np.ndarray]) -> dict:
    joined = np.concatenate([value.reshape(-1) for value in values])
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(joined**2)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(joined)) * 1000.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=labelled_path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--delta-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = MACECalculator(
        model_paths=str(args.base_model),
        device=args.device,
        default_dtype="float32",
    )
    delta = MACECalculator(
        model_paths=str(args.delta_model),
        device=args.device,
        default_dtype="float32",
    )
    payload = {
        "units": {"force": "meV/angstrom"},
        "base_model": str(args.base_model),
        "base_model_sha256": sha256(args.base_model),
        "delta_model": str(args.delta_model),
        "delta_model_sha256": sha256(args.delta_model),
        "datasets": {},
    }
    for label, path in args.dataset:
        structures = read(path, index=":")
        base_errors = []
        combined_short_errors = []
        reconstructed_total_errors = []
        delta_force_values = []
        records = []
        for index, structure in enumerate(structures):
            atoms = structure.copy()
            atoms.calc = base
            base_force = np.asarray(atoms.get_forces(), float)
            atoms.calc = delta
            delta_force = np.asarray(atoms.get_forces(), float)
            short_reference = np.asarray(
                structure.arrays.get(
                    "SHORT_RANGE_TARGET_forces", structure.arrays["REF_forces"]
                ),
                float,
            )
            total_reference = np.asarray(
                structure.arrays.get("TOTAL_forces", short_reference), float
            )
            long_range = np.asarray(
                structure.arrays.get("LONG_RANGE_forces", np.zeros_like(base_force)),
                float,
            )
            combined_short = base_force + delta_force
            reconstructed_total = combined_short + long_range
            reconstructed_error = reconstructed_total - total_reference
            max_flat = int(np.argmax(np.abs(reconstructed_error)))
            max_atom, max_component = np.unravel_index(
                max_flat, reconstructed_error.shape
            )
            base_errors.append(base_force - short_reference)
            combined_short_errors.append(combined_short - short_reference)
            reconstructed_total_errors.append(reconstructed_error)
            delta_force_values.append(delta_force)
            records.append(
                {
                    "structure_index": index,
                    "snapshot_index": (
                        int(structure.info["snapshot_index"])
                        if "snapshot_index" in structure.info
                        else None
                    ),
                    "combined_total_RMSE_meV_A": float(
                        np.sqrt(np.mean(reconstructed_error**2))
                        * 1000.0
                    ),
                    "combined_total_max_abs_meV_A": float(
                        np.max(np.abs(reconstructed_error)) * 1000.0
                    ),
                    "max_error_atom_index": int(max_atom),
                    "max_error_cartesian_component": "xyz"[int(max_component)],
                    "max_error_signed_meV_A": float(
                        reconstructed_error[max_atom, max_component] * 1000.0
                    ),
                }
            )
        payload["datasets"][label] = {
            "path": str(path),
            "n_structures": len(structures),
            "base_only_short_range_error": metrics(base_errors),
            "combined_short_range_error": metrics(combined_short_errors),
            "reconstructed_total_error": metrics(reconstructed_total_errors),
            "delta_prediction_magnitude": metrics(delta_force_values),
            "records": records,
        }

    del base, delta
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
