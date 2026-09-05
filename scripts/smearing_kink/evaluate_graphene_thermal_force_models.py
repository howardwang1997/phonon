#!/usr/bin/env python3
"""Evaluate graphene thermal fine-tunes on untouched physical-FD labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def parse_model(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("models must use LABEL=/path/to/model")
    label, path = specification.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("models must use LABEL=/path/to/model")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-a", type=Path, required=True)
    parser.add_argument("--validation-b", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = {
        "300": read(args.validation_a, index=":"),
        "600": read(args.validation_b, index=":"),
    }
    for temperature, structures in validation.items():
        if len(structures) < 3 or len(structures) % 3:
            raise ValueError(
                f"T={temperature} validation must contain complete three-structure "
                f"waves; found {len(structures)} structures"
            )

    payload = {"units": {"force": "meV/angstrom"}, "models": {}}
    for label, model_path in args.model:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        calculator = MACECalculator(
            model_paths=str(model_path),
            device=args.device,
            default_dtype="float32",
        )
        model_result = {"path": str(model_path), "by_temperature": {}}
        for temperature, structures in validation.items():
            all_delta = []
            records = []
            for structure in structures:
                reference = np.asarray(structure.arrays["REF_forces"], dtype=float)
                evaluated = structure.copy()
                evaluated.calc = calculator
                predicted = np.asarray(evaluated.get_forces(), dtype=float)
                delta = predicted - reference
                all_delta.append(delta.reshape(-1))
                records.append(
                    {
                        "snapshot_index": int(structure.info["snapshot_index"]),
                        "RMSE_meV_A": float(np.sqrt(np.mean(delta**2)) * 1000.0),
                        "max_abs_meV_A": float(np.max(np.abs(delta)) * 1000.0),
                    }
                )
            flattened = np.concatenate(all_delta)
            model_result["by_temperature"][temperature] = {
                "n_validation_structures": len(structures),
                "RMSE_meV_A": float(np.sqrt(np.mean(flattened**2)) * 1000.0),
                "max_abs_meV_A": float(np.max(np.abs(flattened)) * 1000.0),
                "records": records,
            }
        payload["models"][label] = model_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
