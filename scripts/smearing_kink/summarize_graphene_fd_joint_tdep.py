#!/usr/bin/env python3
"""Audit one endpoint of the joint-model three-seed short-range TDEP run."""
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


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def scalar(data, key: str):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} is not scalar")
    return value.reshape(()).item()


def load_tdep(path: Path, temperature: int) -> dict:
    prefix = f"T{temperature}"
    with np.load(path, allow_pickle=False) as data:
        distance = np.asarray(data[f"{prefix}_dist"], float)
        frequency = np.asarray(data[f"{prefix}_freq"], float) * CM_PER_THz
        force_constants = np.asarray(data[f"{prefix}_fc2"], float)
        counts = np.asarray(data["source_checkpoint_counts"], int).tolist()
        result = {
            "path": str(path),
            "sha256": sha256(path),
            "distance": distance,
            "frequency_cm-1": frequency,
            "force_constants_shape": list(force_constants.shape),
            "source_checkpoint_counts": counts,
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
    parser.add_argument("--temperature", type=int, choices=(300, 600), required=True)
    parser.add_argument("--degauss", type=float, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--delta-model", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--static-manifest", type=Path, required=True)
    parser.add_argument("--force-gate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--seed-tdep", type=Path, action="append", required=True)
    parser.add_argument("--pooled-tdep", type=Path, required=True)
    parser.add_argument("--seed-mae-threshold", type=float, default=5.0)
    parser.add_argument(
        "--mean-temperature-relative-threshold", type=float, default=0.20
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.checkpoint) != 3 or len(args.seed_tdep) != 3:
        parser.error("exactly three checkpoints and three seed TDEP files are required")
    expected_degauss = 0.0019000869 * args.temperature / 300.0
    if not np.isclose(args.degauss, expected_degauss, rtol=0.0, atol=1e-12):
        raise ValueError("degauss is not on the fixed physical-FD temperature path")

    model_sha = sha256(args.delta_model)
    operator_sha = sha256(args.operator)
    static = json.loads(args.static_manifest.read_text())
    force = json.loads(args.force_gate.read_text())
    if static["delta_model_sha256"] != model_sha:
        raise ValueError("static FC2 manifest uses a different delta model")
    if force["frozen_model_sha256"] != model_sha:
        raise ValueError("force gate uses a different delta model")
    if not force["passes_force_and_replay_gate"]:
        raise ValueError("joint force gate is not passed")
    static_output = Path(static["output"])
    if sha256(static_output) != static["output_sha256"]:
        raise ValueError("static FC2 hash no longer matches its manifest")

    seeds = [load_tdep(path, args.temperature) for path in args.seed_tdep]
    pooled = load_tdep(args.pooled_tdep, args.temperature)
    for seed in seeds:
        if seed["subtracted_operator_sha256"] != operator_sha:
            raise ValueError("seed TDEP subtracted a different provisional operator")
        if seed["source_checkpoint_counts"] != [120]:
            raise ValueError("each seed TDEP must contain exactly 120 snapshots")
    if pooled["subtracted_operator_sha256"] != operator_sha:
        raise ValueError("pooled TDEP subtracted a different provisional operator")
    if pooled["source_checkpoint_counts"] != [120, 120, 120]:
        raise ValueError("pooled TDEP must contain 120 snapshots from each seed")

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
        checkpoint_records.append(
            {
                "seed": index,
                "path": str(checkpoint),
                "snapshots_path": str(snapshot_file),
                "snapshots_sha256": sha256(snapshot_file),
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
    result = {
        "status": "passed" if passed else "failed",
        "scope": (
            "endpoint MLIP trajectory sampling and short-range TDEP development "
            "before any 450 K labels"
        ),
        "temperature_K": args.temperature,
        "smearing_degauss_Ry": args.degauss,
        "thresholds": {
            "seed_pair_full_band_MAE_cm-1": args.seed_mae_threshold,
            "MD_mean_temperature_relative_error": (
                args.mean_temperature_relative_threshold
            ),
        },
        "passes_seed_spread_gate": spread_pass,
        "passes_temperature_gate": temperature_pass,
        "passes_tdep_sampling_gate": passed,
        "seed_pair_spread": seed_pairs,
        "seed_temperature_diagnostics": temperature_diagnostics,
        "base_model": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
        "delta_model": {"path": str(args.delta_model), "sha256": model_sha},
        "background": {"path": str(args.background), "sha256": sha256(args.background)},
        "provisional_sampling_operator": {
            "path": str(args.operator),
            "sha256": operator_sha,
            "role": (
                "used only during endpoint MD sampling and subtracted exactly from "
                "every saved force before short-range TDEP fitting"
            ),
        },
        "static_short_fc2": {
            "path": str(static_output),
            "sha256": static["output_sha256"],
            "manifest_path": str(args.static_manifest),
            "manifest_sha256": sha256(args.static_manifest),
        },
        "force_gate": {
            "path": str(args.force_gate),
            "sha256": sha256(args.force_gate),
        },
        "checkpoints": checkpoint_records,
        "seed_tdep": seed_records,
        "pooled_tdep": pooled_record,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
