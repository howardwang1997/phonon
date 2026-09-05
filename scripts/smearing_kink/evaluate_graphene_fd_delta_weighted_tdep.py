#!/usr/bin/env python3
"""Evaluate the frozen 600 K weighted-delta model on three full-force TDEP runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np


CM = 33.35641


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spectrum(path: Path, temperature: int) -> dict:
    prefix = f"T{temperature}"
    with np.load(path, allow_pickle=False) as data:
        frequency = np.asarray(data[f"{prefix}_freq"], float) * CM
        return {
            "path": str(path),
            "sha256": sha256(path),
            "distance": np.asarray(data[f"{prefix}_dist"], float),
            "frequency_cm-1": frequency,
            "label_positions": np.asarray(data["label_positions"], float),
            "labels": [str(value) for value in data["labels"]],
            "mean_temperature_K": (
                float(data[f"{prefix}_mean_temperature_K"])
                if f"{prefix}_mean_temperature_K" in data
                else None
            ),
            "max_temperature_K": (
                float(data[f"{prefix}_max_temperature_K"])
                if f"{prefix}_max_temperature_K" in data
                else None
            ),
            "minimum_frequency_cm-1": float(np.min(frequency)),
        }


def normalized_labels(spectrum: dict) -> list[str]:
    return [label.replace("$", "").replace("\\", "") for label in spectrum["labels"]]


def interpolate(reference: dict, candidate: dict) -> np.ndarray:
    if candidate["frequency_cm-1"].shape[1] != reference["frequency_cm-1"].shape[1]:
        raise ValueError("candidate and reference have different branch counts")
    return np.stack(
        [
            np.interp(
                reference["distance"],
                candidate["distance"],
                candidate["frequency_cm-1"][:, branch],
            )
            for branch in range(candidate["frequency_cm-1"].shape[1])
        ],
        axis=1,
    )


def spectrum_metrics(reference: dict, candidate: dict) -> dict[str, float]:
    candidate_frequency = interpolate(reference, candidate)
    delta = candidate_frequency - reference["frequency_cm-1"]
    labels = normalized_labels(reference)
    if "Gamma" not in labels or "K" not in labels:
        raise ValueError(f"reference labels do not contain Gamma and K: {labels}")
    gamma_position = reference["label_positions"][labels.index("Gamma")]
    k_position = reference["label_positions"][labels.index("K")]
    gamma_index = int(np.argmin(np.abs(reference["distance"] - gamma_position)))
    k_index = int(np.argmin(np.abs(reference["distance"] - k_position)))
    return {
        "full_band_MAE_cm-1": float(np.mean(np.abs(delta))),
        "full_band_max_abs_cm-1": float(np.max(np.abs(delta))),
        "Gamma_top_abs_error_cm-1": float(abs(delta[gamma_index, -1])),
        "K_top_abs_error_cm-1": float(abs(delta[k_index, -1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=int, default=600)
    parser.add_argument("--dft-reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--force-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        default=(
            "frozen delta model evaluated after its force gate; TDEP uses the full "
            "base + delta + provisional real-space LR force"
        ),
    )
    parser.add_argument("--mlip-mae-threshold", type=float, default=10.0)
    parser.add_argument("--high-symmetry-threshold", type=float, default=15.0)
    parser.add_argument("--seed-mae-threshold", type=float, default=5.0)
    parser.add_argument("--mean-temperature-relative-threshold", type=float, default=0.20)
    args = parser.parse_args()

    if len(args.candidate) != 3:
        parser.error("exactly three --candidate paths are required")
    force = json.loads(args.force_selection.read_text())
    force_passed = bool(force["passes_force_and_replay_gate"])
    reference = load_spectrum(args.dft_reference, args.temperature)
    candidates = [load_spectrum(path, args.temperature) for path in args.candidate]
    if any(item["mean_temperature_K"] is None for item in candidates):
        raise ValueError("candidate TDEP result is missing trajectory-temperature diagnostics")

    versus_dft = [spectrum_metrics(reference, candidate) for candidate in candidates]
    seed_pairs = []
    for (left_index, left), (right_index, right) in combinations(enumerate(candidates), 2):
        seed_pairs.append(
            {
                "seeds": [left_index, right_index],
                **spectrum_metrics(left, right),
            }
        )
    mean_temperature_relative_errors = [
        abs(float(item["mean_temperature_K"]) - args.temperature) / args.temperature
        for item in candidates
    ]
    finite = all(
        np.isfinite(
            [
                item["mean_temperature_K"],
                item["max_temperature_K"],
                item["minimum_frequency_cm-1"],
            ]
        ).all()
        for item in candidates
    )
    spectrum_passed = (
        max(item["full_band_MAE_cm-1"] for item in versus_dft)
        < args.mlip_mae_threshold
        and max(
            max(item["Gamma_top_abs_error_cm-1"], item["K_top_abs_error_cm-1"])
            for item in versus_dft
        )
        < args.high_symmetry_threshold
        and max(item["full_band_MAE_cm-1"] for item in seed_pairs)
        < args.seed_mae_threshold
        and max(mean_temperature_relative_errors)
        <= args.mean_temperature_relative_threshold
        and finite
    )
    result = {
        "status": "passed" if force_passed and spectrum_passed else "failed",
        "temperature_K": args.temperature,
        "scope": args.scope,
        "force_gate_passed": force_passed,
        "force_selection": str(args.force_selection),
        "force_selection_sha256": sha256(args.force_selection),
        "dft_reference": {
            "path": reference["path"],
            "sha256": reference["sha256"],
        },
        "thresholds": {
            "MLIP_vs_DFT_full_band_MAE_cm-1": args.mlip_mae_threshold,
            "Gamma_K_top_abs_error_cm-1": args.high_symmetry_threshold,
            "seed_pair_full_band_MAE_cm-1": args.seed_mae_threshold,
            "MD_mean_temperature_relative_error": args.mean_temperature_relative_threshold,
        },
        "candidate_files": [
            {"path": item["path"], "sha256": item["sha256"]} for item in candidates
        ],
        "candidate_vs_DFT": versus_dft,
        "seed_pair_spread": seed_pairs,
        "trajectory": [
            {
                "seed": index,
                "mean_temperature_K": item["mean_temperature_K"],
                "max_temperature_K": item["max_temperature_K"],
                "mean_temperature_relative_error": mean_temperature_relative_errors[index],
                "minimum_TDEP_frequency_cm-1": item["minimum_frequency_cm-1"],
            }
            for index, item in enumerate(candidates)
        ],
        "finite_trajectory_and_spectrum_diagnostics": finite,
        "passes_TDEP_gate": spectrum_passed,
        "passes_force_and_TDEP_gate": force_passed and spectrum_passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
