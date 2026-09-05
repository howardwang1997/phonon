#!/usr/bin/env python3
"""Materialize the frozen joint-model endpoint force gate for later stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    if selection["status"] != "joint_endpoint_candidate_found":
        raise ValueError("joint endpoint selection did not pass its fixed gate")
    selected = selection["selected"]
    if not selected["passes_fixed_endpoint_gate"]:
        raise ValueError("selected joint model is not marked as endpoint-passing")

    sweep_path = Path(selected["sweep_path"])
    sweep = json.loads(sweep_path.read_text())
    if sweep["status"] != "checkpoint_passed":
        raise ValueError("selected checkpoint sweep did not pass")
    source_model = Path(sweep["selected_model"])
    source_model_sha = sha256(source_model)
    frozen_model_sha = sha256(args.model)
    if source_model_sha != frozen_model_sha:
        raise ValueError("copied frozen model differs from the selected checkpoint model")

    record = sweep["selected"]
    thermals = record["thermal_total_force"]
    if set(thermals) != {"thermal300", "thermal600"}:
        raise ValueError(f"unexpected endpoint labels: {sorted(thermals)}")
    thresholds = sweep["fixed_thresholds"]
    thermal_pass = all(
        metric["RMSE_meV_A"] <= thresholds["thermal_force_RMSE_meV_A"]
        and metric["max_abs_meV_A"] <= thresholds["thermal_force_max_abs_meV_A"]
        for metric in thermals.values()
    )
    harmonic_pass = (
        record["harmonic_combined_force"]["RMSE_meV_A"]
        <= thresholds["harmonic_RMSE_meV_A"]
    )
    passed = bool(record["passes_fixed_gate"] and thermal_pass and harmonic_pass)
    if not passed:
        raise ValueError("selected checkpoint metrics do not reproduce the fixed gate")

    result = {
        "status": "passed",
        "scope": (
            "300/600 K endpoint development selection completed before generating "
            "or labeling any 450 K configurations"
        ),
        "passes_force_and_replay_gate": True,
        "passes_both_endpoint_development_gates": True,
        "thresholds": thresholds,
        "thermal_total_force": {
            "300": thermals["thermal300"],
            "600": thermals["thermal600"],
        },
        "harmonic_combined_force": record["harmonic_combined_force"],
        "harmonic_v11_force": sweep["harmonic_v11_force"],
        "normalized_worst_gate_score": record["normalized_worst_gate_score"],
        "selected_epoch": record["epoch"],
        "frozen_model": str(args.model),
        "frozen_model_sha256": frozen_model_sha,
        "inputs": {
            "checkpoint_selection": {
                "path": str(args.selection),
                "sha256": sha256(args.selection),
            },
            "checkpoint_sweep": {
                "path": str(sweep_path),
                "sha256": sha256(sweep_path),
            },
            "selected_checkpoint": {
                "path": record["checkpoint"],
                "sha256": record["checkpoint_sha256"],
            },
            "selected_source_model": {
                "path": str(source_model),
                "sha256": source_model_sha,
            },
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
