#!/usr/bin/env python3
"""Evaluate a proxy delta-MACE on grouped 300/600 K development forces."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True, help="label=path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    datasets: list[tuple[str, Path]] = []
    for spec in args.dataset:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label:
            raise ValueError(f"invalid dataset specification: {spec}")
        path = Path(raw_path)
        if any(token in str(path) for token in ("T450", "P4", "p4")):
            raise ValueError(f"refusing 450 K P4 input: {path}")
        datasets.append((label, path))

    calculator = MACECalculator(
        model_paths=str(args.model), device=args.device, default_dtype="float32"
    )
    errors: dict[str, list[np.ndarray]] = defaultdict(list)
    n_structures: dict[str, int] = defaultdict(int)
    for dataset_label, path in datasets:
        for structure in read(path, index=":"):
            config_type = str(structure.info.get("config_type", "Default"))
            if "450" in config_type or "P4" in config_type or "p4" in config_type:
                raise ValueError(f"refusing P4-tagged structure: {config_type}")
            key = f"{dataset_label}:{config_type}"
            reference = np.asarray(structure.arrays["REF_forces"], dtype=float)
            probe = structure.copy()
            probe.calc = calculator
            prediction = np.asarray(probe.get_forces(), dtype=float)
            errors[key].append((prediction - reference).reshape(-1))
            n_structures[key] += 1

    grouped = {}
    for key in sorted(errors):
        values = np.concatenate(errors[key])
        grouped[key] = {
            "n_structures": n_structures[key],
            "force_RMSE_meV_A": float(np.sqrt(np.mean(values**2)) * 1000.0),
            "force_MAE_meV_A": float(np.mean(np.abs(values)) * 1000.0),
            "force_max_abs_meV_A": float(np.max(np.abs(values)) * 1000.0),
        }
    payload = {
        "status": "complete",
        "scope": "300/600 K development residual-force proxy evaluation",
        "scientific_use": "batch/LR convergence sizing only",
        "target_leakage": {
            "p4_450K_inputs_read": False,
            "allowed_input_temperatures_K": [300, 600],
        },
        "model": {
            "path": str(args.model),
            "sha256": sha256(args.model),
        },
        "datasets": [
            {"label": label, "path": str(path), "sha256": sha256(path)}
            for label, path in datasets
        ],
        "grouped_metrics": grouped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
