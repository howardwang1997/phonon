#!/usr/bin/env python3
"""Combine accepted 300/600 K frozen-v11 delta-model force gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-300", type=Path, required=True)
    parser.add_argument("--metrics-600", type=Path, required=True)
    parser.add_argument("--selection-300", type=Path, required=True)
    parser.add_argument("--selection-600", type=Path, required=True)
    parser.add_argument("--force-output", type=Path, required=True)
    parser.add_argument("--stack-output", type=Path, required=True)
    args = parser.parse_args()

    metrics = {
        300: json.loads(args.metrics_300.read_text()),
        600: json.loads(args.metrics_600.read_text()),
    }
    selections = {
        300: json.loads(args.selection_300.read_text()),
        600: json.loads(args.selection_600.read_text()),
    }
    if not all(item["passes_force_and_replay_gate"] for item in selections.values()):
        raise RuntimeError("cannot combine a delta model that failed its force gate")

    force_payload = {"units": {"force": "meV/angstrom"}, "models": {}}
    stack_payload = {
        "method": "frozen v11 + additive delta MACE + harmonic Fermi-Dirac correction",
        "status": "provisional_operator_pilot",
        "by_temperature": {},
    }
    for temperature in (300, 600):
        thermal_label = f"thermal{temperature}"
        total = metrics[temperature]["datasets"][thermal_label][
            "reconstructed_total_error"
        ]
        force_payload["models"][f"fd{temperature}"] = {
            "path": metrics[temperature]["delta_model"],
            "sha256": metrics[temperature]["delta_model_sha256"],
            "by_temperature": {
                str(temperature): {
                    "n_validation_structures": metrics[temperature]["datasets"][
                        thermal_label
                    ]["n_structures"],
                    **total,
                }
            },
        }
        stack_payload["by_temperature"][str(temperature)] = {
            "base_model": metrics[temperature]["base_model"],
            "base_model_sha256": metrics[temperature]["base_model_sha256"],
            "delta_model": metrics[temperature]["delta_model"],
            "delta_model_sha256": metrics[temperature]["delta_model_sha256"],
            "force_gate": selections[temperature],
        }
    atomic_json(args.force_output, force_payload)
    atomic_json(args.stack_output, stack_payload)
    print(json.dumps(stack_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
