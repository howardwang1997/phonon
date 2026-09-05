#!/usr/bin/env python3
"""Gate the three 450 K conditioned-model trajectories before DFT labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np


CM_PER_THz = 33.35641


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(data, key: str):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} is not scalar")
    return value.reshape(()).item()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_tdep(path: Path, temperature: int) -> dict:
    prefix = f"T{temperature}"
    with np.load(path, allow_pickle=False) as data:
        distance = np.asarray(data[f"{prefix}_dist"], float)
        frequency = np.asarray(data[f"{prefix}_freq"], float) * CM_PER_THz
        force_constants = np.asarray(data[f"{prefix}_fc2"], float)
        result = {
            "path": str(path),
            "sha256": sha256(path),
            "distance": distance,
            "frequency_cm-1": frequency,
            "force_constants_shape": list(force_constants.shape),
            "source_checkpoint_counts": np.asarray(
                data["source_checkpoint_counts"], int
            ).tolist(),
            "mean_temperature_K": float(scalar(data, f"{prefix}_mean_temperature_K")),
            "max_temperature_K": float(scalar(data, f"{prefix}_max_temperature_K")),
            "fit_RMSE_meV_A": float(scalar(data, f"{prefix}_fit_rmse_meV_A")),
            "subtracted_operator_sha256": str(
                scalar(data, "subtracted_operator_sha256")
            ),
            "subtracted_operator_force_RMSE_meV_A": float(
                scalar(data, "subtracted_operator_force_RMSE_meV_A")
            ),
            "subtracted_operator_force_max_abs_meV_A": float(
                scalar(data, "subtracted_operator_force_max_abs_meV_A")
            ),
        }
    if distance.ndim != 1 or frequency.ndim != 2:
        raise ValueError(f"invalid band arrays in {path}")
    if not np.isfinite(frequency).all() or not np.isfinite(force_constants).all():
        raise ValueError(f"non-finite TDEP values in {path}")
    return result


def interpolated_mae(left: dict, right: dict) -> float:
    right_frequency = np.stack(
        [
            np.interp(
                left["distance"],
                right["distance"],
                right["frequency_cm-1"][:, branch],
            )
            for branch in range(right["frequency_cm-1"].shape[1])
        ],
        axis=1,
    )
    return float(np.mean(np.abs(right_frequency - left["frequency_cm-1"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=int, default=450)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--seed-tdep", type=Path, action="append", required=True)
    parser.add_argument("--pooled-tdep", type=Path, required=True)
    parser.add_argument("--seed-mae-threshold", type=float, default=5.0)
    parser.add_argument("--mean-temperature-relative-threshold", type=float, default=0.05)
    parser.add_argument("--expected-snapshots-per-seed", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.checkpoint) != 3 or len(args.seed_tdep) != 3:
        parser.error("exactly three checkpoints and three seed TDEP files are required")
    freeze = json.loads(args.freeze_manifest.read_text())
    if freeze.get("status") != "frozen_before_450_holdout":
        raise ValueError("invalid conditional predictor freeze manifest")
    if freeze["T450_on_policy"]["temperature_K"] != args.temperature:
        raise ValueError("sampling temperature differs from the frozen holdout")
    operator_sha = sha256(args.operator)
    if freeze["prediction_sampling_operator"]["output"]["sha256"] != operator_sha:
        raise ValueError("sampling operator differs from the frozen predictor")

    seeds = [load_tdep(path, args.temperature) for path in args.seed_tdep]
    pooled = load_tdep(args.pooled_tdep, args.temperature)
    for seed in seeds:
        if seed["subtracted_operator_sha256"] != operator_sha:
            raise ValueError("seed TDEP subtracted a different provisional operator")
        if seed["source_checkpoint_counts"] != [args.expected_snapshots_per_seed]:
            raise ValueError(
                "each seed TDEP must contain exactly "
                f"{args.expected_snapshots_per_seed} snapshots"
            )
    if pooled["subtracted_operator_sha256"] != operator_sha:
        raise ValueError("pooled TDEP subtracted a different provisional operator")
    expected_pooled = [args.expected_snapshots_per_seed] * 3
    if pooled["source_checkpoint_counts"] != expected_pooled:
        raise ValueError(
            "pooled TDEP source counts differ from "
            f"{expected_pooled}"
        )

    seed_pairs = [
        {
            "seeds": [left_index, right_index],
            "full_band_MAE_cm-1": interpolated_mae(left, right),
        }
        for (left_index, left), (right_index, right) in combinations(
            enumerate(seeds), 2
        )
    ]
    temperature_diagnostics = [
        {
            "seed": index,
            "mean_temperature_K": seed["mean_temperature_K"],
            "max_temperature_K": seed["max_temperature_K"],
            "mean_temperature_relative_error": abs(
                seed["mean_temperature_K"] - args.temperature
            )
            / args.temperature,
        }
        for index, seed in enumerate(seeds)
    ]
    spread_pass = max(item["full_band_MAE_cm-1"] for item in seed_pairs) < args.seed_mae_threshold
    temperature_pass = max(
        item["mean_temperature_relative_error"] for item in temperature_diagnostics
    ) <= args.mean_temperature_relative_threshold
    passed = bool(spread_pass and temperature_pass)

    checkpoint_records = []
    for index, checkpoint in enumerate(args.checkpoint):
        snapshot_file = checkpoint / "snapshots.npz" if checkpoint.is_dir() else checkpoint
        with np.load(snapshot_file, allow_pickle=False) as data:
            n_snapshots = len(np.asarray(data["positions"]))
        if n_snapshots != args.expected_snapshots_per_seed:
            raise ValueError(
                f"seed {index} checkpoint has {n_snapshots} snapshots; "
                f"expected {args.expected_snapshots_per_seed}"
            )
        checkpoint_records.append(
            {
                "seed": index,
                "path": str(checkpoint),
                "snapshots_path": str(snapshot_file),
                "snapshots_sha256": sha256(snapshot_file),
                "n_snapshots": n_snapshots,
            }
        )
    seed_records = []
    for index, seed in enumerate(seeds):
        seed_records.append(
            {
                key: value
                for key, value in {"seed": index, **seed}.items()
                if key not in {"distance", "frequency_cm-1"}
            }
        )
    pooled_record = {
        key: value
        for key, value in pooled.items()
        if key not in {"distance", "frequency_cm-1"}
    }
    payload = {
        "status": "passed" if passed else "failed",
        "scope": (
            "450 K frozen conditioned-model on-policy sampling gate before any "
            "450 K DFT force labels"
        ),
        "temperature_K": args.temperature,
        "degauss_Ry": 0.0019000869 * args.temperature / 300.0,
        "thresholds": {
            "seed_pair_full_band_MAE_cm-1": args.seed_mae_threshold,
            "MD_mean_temperature_relative_error": args.mean_temperature_relative_threshold,
        },
        "expected_snapshots_per_seed": args.expected_snapshots_per_seed,
        "passes_seed_spread_gate": spread_pass,
        "passes_temperature_gate": temperature_pass,
        "passes_sampling_gate": passed,
        "seed_pair_spread": seed_pairs,
        "seed_temperature_diagnostics": temperature_diagnostics,
        "freeze_manifest": {
            "path": str(args.freeze_manifest),
            "sha256": sha256(args.freeze_manifest),
        },
        "sampling_operator": {"path": str(args.operator), "sha256": operator_sha},
        "checkpoints": checkpoint_records,
        "seed_tdeps": seed_records,
        "pooled_tdep": pooled_record,
        "next_stage": (
            "release the fixed 60-structure DFT force-label shards"
            if passed
            else "keep all 450 K DFT force labels locked"
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
