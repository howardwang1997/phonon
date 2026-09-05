#!/usr/bin/env python3
"""Measure whether an additive adapter learned direction or only small amplitude."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def metrics(target: list[np.ndarray], predicted: list[np.ndarray]) -> dict[str, float]:
    y = np.concatenate([np.asarray(value, float).reshape(-1) for value in target])
    p = np.concatenate([np.asarray(value, float).reshape(-1) for value in predicted])
    denominator = float(np.dot(p, p))
    alpha = float(np.dot(p, y) / denominator) if denominator > 0.0 else 0.0
    error = p - y
    scaled_error = alpha * p - y
    target_squared = float(np.dot(y, y))
    cosine_denominator = float(np.linalg.norm(y) * np.linalg.norm(p))
    return {
        "target_RMS_meV_A": float(np.sqrt(np.mean(y**2)) * 1000.0),
        "prediction_RMS_meV_A": float(np.sqrt(np.mean(p**2)) * 1000.0),
        "error_RMS_meV_A": float(np.sqrt(np.mean(error**2)) * 1000.0),
        "cosine_similarity": (
            float(np.dot(p, y) / cosine_denominator)
            if cosine_denominator > 0.0
            else 0.0
        ),
        "squared_error_fraction_removed": (
            float(1.0 - np.dot(error, error) / target_squared)
            if target_squared > 0.0
            else 0.0
        ),
        "least_squares_amplitude": alpha,
        "scaled_error_RMS_meV_A": float(
            np.sqrt(np.mean(scaled_error**2)) * 1000.0
        ),
        "scaled_squared_error_fraction_removed": (
            float(1.0 - np.dot(scaled_error, scaled_error) / target_squared)
            if target_squared > 0.0
            else 0.0
        ),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    structures = read(args.data, index=":")
    calculator = MACECalculator(
        model_paths=str(args.model), device=args.device, default_dtype="float32"
    )
    target_by_type: dict[str, list[np.ndarray]] = defaultdict(list)
    predicted_by_type: dict[str, list[np.ndarray]] = defaultdict(list)
    records = []
    for source in structures:
        atoms = source.copy()
        target = np.asarray(source.arrays["REF_forces"], float)
        atoms.calc = calculator
        predicted = np.asarray(atoms.get_forces(), float)
        config_type = str(source.info.get("config_type", "Default"))
        target_by_type[config_type].append(target)
        predicted_by_type[config_type].append(predicted)
        record = {
            "config_type": config_type,
            "target_RMS_meV_A": float(np.sqrt(np.mean(target**2)) * 1000.0),
            "prediction_RMS_meV_A": float(
                np.sqrt(np.mean(predicted**2)) * 1000.0
            ),
            "error_RMS_meV_A": float(
                np.sqrt(np.mean((predicted - target) ** 2)) * 1000.0
            ),
        }
        if "sscha_index" in source.info:
            record["sscha_index"] = int(source.info["sscha_index"])
        records.append(record)

    all_target = [value for values in target_by_type.values() for value in values]
    all_predicted = [value for values in predicted_by_type.values() for value in values]
    summary = {
        "n_structures": len(structures),
        "all": metrics(all_target, all_predicted),
        "by_config_type": {
            config_type: metrics(
                target_by_type[config_type], predicted_by_type[config_type]
            )
            for config_type in sorted(target_by_type)
        },
        "configuration_records": records,
    }
    if args.output:
        atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
