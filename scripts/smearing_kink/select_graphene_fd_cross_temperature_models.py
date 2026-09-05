#!/usr/bin/env python3
"""Apply the fixed endpoint gates to existing graphene delta models.

Each input metrics file must be produced by
``evaluate_graphene_fd_delta_model.py`` with both thermal endpoint datasets and
the same harmonic replay set.  This script does not evaluate forces itself; it
only makes the pre-holdout decision whether an existing short model can be
reused at both 300 and 600 K.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("candidates must use LABEL=/path")
    label, path = specification.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("candidates must use LABEL=/path")
    return label, Path(path)


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
    parser.add_argument("--candidate", action="append", type=labelled_path, required=True)
    parser.add_argument("--thermal-300-label", default="thermal300")
    parser.add_argument("--thermal-600-label", default="thermal600")
    parser.add_argument("--harmonic-label", default="harmonic")
    parser.add_argument("--force-rmse-threshold", type=float, default=50.0)
    parser.add_argument("--force-max-threshold", type=float, default=250.0)
    parser.add_argument("--harmonic-rmse-ratio", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates: dict[str, dict] = {}
    reference_base_hash = None
    reference_harmonic_baseline = None
    thermal_labels = {
        "300": args.thermal_300_label,
        "600": args.thermal_600_label,
    }
    for label, path in args.candidate:
        if label in candidates:
            raise ValueError(f"duplicate candidate label: {label}")
        payload = json.loads(path.read_text())
        missing = [
            dataset
            for dataset in (*thermal_labels.values(), args.harmonic_label)
            if dataset not in payload["datasets"]
        ]
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")

        base_hash = payload["base_model_sha256"]
        harmonic_dataset = payload["datasets"][args.harmonic_label]
        harmonic = harmonic_dataset["reconstructed_total_error"]
        harmonic_baseline = harmonic_dataset["base_only_short_range_error"]
        if reference_base_hash is None:
            reference_base_hash = base_hash
            reference_harmonic_baseline = harmonic_baseline
        elif base_hash != reference_base_hash:
            raise ValueError("candidate evaluations used different frozen base models")
        elif not all(
            math.isclose(
                harmonic_baseline[key],
                reference_harmonic_baseline[key],
                rel_tol=1e-6,
                abs_tol=1e-5,
            )
            for key in ("RMSE_meV_A", "max_abs_meV_A")
        ):
            raise ValueError("candidate evaluations used different harmonic replay baselines")

        by_temperature = {}
        scores = [
            harmonic["RMSE_meV_A"]
            / (args.harmonic_rmse_ratio * harmonic_baseline["RMSE_meV_A"])
        ]
        passes_thermal = True
        for temperature, dataset_label in thermal_labels.items():
            dataset = payload["datasets"][dataset_label]
            thermal = dataset["reconstructed_total_error"]
            passed = bool(
                thermal["RMSE_meV_A"] <= args.force_rmse_threshold
                and thermal["max_abs_meV_A"] <= args.force_max_threshold
            )
            passes_thermal = passes_thermal and passed
            scores.extend(
                (
                    thermal["RMSE_meV_A"] / args.force_rmse_threshold,
                    thermal["max_abs_meV_A"] / args.force_max_threshold,
                )
            )
            by_temperature[temperature] = {
                "dataset_label": dataset_label,
                "dataset_path": dataset["path"],
                "n_structures": dataset["n_structures"],
                "reconstructed_total_force": thermal,
                "passes_thermal_force_gate": passed,
            }

        passes_harmonic = bool(
            harmonic["RMSE_meV_A"]
            <= args.harmonic_rmse_ratio * harmonic_baseline["RMSE_meV_A"]
        )
        passes_all = bool(passes_thermal and passes_harmonic)
        candidates[label] = {
            "metrics_path": str(path),
            "metrics_sha256": sha256(path),
            "base_model": payload["base_model"],
            "base_model_sha256": base_hash,
            "delta_model": payload["delta_model"],
            "delta_model_sha256": payload["delta_model_sha256"],
            "by_temperature": by_temperature,
            "harmonic_replay": {
                "dataset_path": harmonic_dataset["path"],
                "n_structures": harmonic_dataset["n_structures"],
                "combined_force": harmonic,
                "frozen_v11_force": harmonic_baseline,
                "passes_harmonic_retention_gate": passes_harmonic,
            },
            "normalized_worst_gate_score": max(scores),
            "passes_both_endpoint_development_gates": passes_all,
        }

    eligible = sorted(
        (label for label, item in candidates.items() if item["passes_both_endpoint_development_gates"]),
        key=lambda label: (candidates[label]["normalized_worst_gate_score"], label),
    )
    selected = eligible[0] if eligible else None
    result = {
        "status": "existing_model_found" if selected else "joint_model_required",
        "scope": (
            "300/600 K endpoint development cross-evaluation before any 450 K "
            "labels; not a lattice-temperature holdout"
        ),
        "thresholds": {
            "thermal_force_RMSE_meV_A": args.force_rmse_threshold,
            "thermal_force_max_abs_meV_A": args.force_max_threshold,
            "harmonic_RMSE_relative_to_frozen_v11": args.harmonic_rmse_ratio,
        },
        "candidates": candidates,
        "eligible_existing_models": eligible,
        "selected_existing_model": selected,
        "requires_joint_model": selected is None,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
