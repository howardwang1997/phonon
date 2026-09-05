#!/usr/bin/env python3
"""Evaluate joint-delta checkpoints on both fixed endpoint force gates."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


def labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("thermal datasets must use LABEL=/path")
    label, path = specification.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("thermal datasets must use LABEL=/path")
    return label, Path(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_load(path: Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"_epoch-(\d+)(?:_swa)?\.pt$", path.name)
    if not match:
        raise ValueError(f"cannot parse checkpoint epoch from {path}")
    return int(match.group(1))


def force_metrics(values) -> dict[str, float]:
    joined = np.concatenate([value.reshape(-1) for value in values])
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(joined**2)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(joined)) * 1000.0),
    }


def reference(structure, base_force):
    short = np.asarray(
        structure.arrays.get("SHORT_RANGE_TARGET_forces", structure.arrays["REF_forces"]),
        float,
    )
    total = np.asarray(structure.arrays.get("TOTAL_forces", short), float)
    long_range = np.asarray(
        structure.arrays.get("LONG_RANGE_forces", np.zeros_like(base_force)), float
    )
    return total, long_range


def base_predictions(structures, calculator):
    values = []
    for structure in structures:
        atoms = structure.copy()
        atoms.calc = calculator
        values.append(np.asarray(atoms.get_forces(), float))
    return values


def evaluate_delta(structures, base_forces, calculator):
    errors = []
    for structure, base_force in zip(structures, base_forces):
        atoms = structure.copy()
        atoms.calc = calculator
        delta_force = np.asarray(atoms.get_forces(), float)
        total, long_range = reference(structure, base_force)
        errors.append(base_force + delta_force + long_range - total)
    return force_metrics(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--template-model", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--thermal", action="append", type=labelled_path, required=True)
    parser.add_argument("--harmonic", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-rmse-threshold", type=float, default=50.0)
    parser.add_argument("--force-max-threshold", type=float, default=250.0)
    parser.add_argument("--harmonic-rmse-ratio", type=float, default=2.0)
    parser.add_argument("--scope", default="joint endpoint development checkpoint selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-model", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = sorted(args.checkpoint_dir.glob("*_epoch-*.pt"), key=checkpoint_epoch)
    if not checkpoints:
        raise ValueError(f"no checkpoints found in {args.checkpoint_dir}")
    thermal = {label: read(path, index=":") for label, path in args.thermal}
    if len(thermal) != len(args.thermal):
        raise ValueError("duplicate thermal dataset labels")
    harmonic = read(args.harmonic, index=":")

    base = MACECalculator(
        model_paths=str(args.base_model), device=args.device, default_dtype="float32"
    )
    thermal_base = {
        label: base_predictions(structures, base) for label, structures in thermal.items()
    }
    harmonic_base = base_predictions(harmonic, base)
    harmonic_v11 = force_metrics(
        [
            base_force + reference(structure, base_force)[1] - reference(structure, base_force)[0]
            for structure, base_force in zip(harmonic, harmonic_base)
        ]
    )
    harmonic_limit = args.harmonic_rmse_ratio * harmonic_v11["RMSE_meV_A"]
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    template = torch_load(args.template_model, "cpu")
    records = []
    for path in checkpoints:
        checkpoint = torch_load(path, "cpu")
        candidate = copy.deepcopy(template)
        candidate.load_state_dict(checkpoint["model"], strict=True)
        delta = MACECalculator(models=candidate, device=args.device, default_dtype="float32")
        by_temperature = {
            label: evaluate_delta(thermal[label], thermal_base[label], delta)
            for label in thermal
        }
        harmonic_metrics = evaluate_delta(harmonic, harmonic_base, delta)
        ratios = {
            f"{label}_RMSE": metrics["RMSE_meV_A"] / args.force_rmse_threshold
            for label, metrics in by_temperature.items()
        }
        ratios.update(
            {
                f"{label}_max_abs": metrics["max_abs_meV_A"] / args.force_max_threshold
                for label, metrics in by_temperature.items()
            }
        )
        ratios["harmonic_RMSE"] = harmonic_metrics["RMSE_meV_A"] / harmonic_limit
        records.append(
            {
                "checkpoint": str(path),
                "checkpoint_sha256": sha256(path),
                "epoch": checkpoint_epoch(path),
                "passes_fixed_gate": all(value <= 1.0 for value in ratios.values()),
                "normalized_gate_ratios": ratios,
                "normalized_worst_gate_score": max(ratios.values()),
                "thermal_total_force": by_temperature,
                "harmonic_combined_force": harmonic_metrics,
            }
        )
        del delta, candidate, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    passing = [record for record in records if record["passes_fixed_gate"]]
    selected = min(
        passing if passing else records,
        key=lambda record: record["normalized_worst_gate_score"],
    )
    selected_checkpoint = torch_load(Path(selected["checkpoint"]), "cpu")
    selected_model = copy.deepcopy(template)
    selected_model.load_state_dict(selected_checkpoint["model"], strict=True)
    args.selected_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected_model.cpu(), args.selected_model)

    result = {
        "status": "checkpoint_passed" if passing else "no_checkpoint_passed",
        "selection_scope": args.scope,
        "fixed_thresholds": {
            "thermal_force_RMSE_meV_A": args.force_rmse_threshold,
            "thermal_force_max_abs_meV_A": args.force_max_threshold,
            "harmonic_RMSE_relative_to_frozen_v11": args.harmonic_rmse_ratio,
            "harmonic_RMSE_meV_A": harmonic_limit,
        },
        "base_model": str(args.base_model),
        "base_model_sha256": sha256(args.base_model),
        "template_model": str(args.template_model),
        "thermal_datasets": {label: str(path) for label, path in args.thermal},
        "harmonic_dataset": str(args.harmonic),
        "harmonic_v11_force": harmonic_v11,
        "n_checkpoints": len(records),
        "selected": selected,
        "selected_model": str(args.selected_model),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
