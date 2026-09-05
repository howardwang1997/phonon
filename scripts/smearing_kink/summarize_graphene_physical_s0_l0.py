#!/usr/bin/env python3
"""Build and gate physical L0 spectra from three short-range TDEP fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
import td_phonon as tdp  # noqa: E402


CM_PER_THZ = 33.35641


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


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_short_tdep(path: Path, temperature: int, operator_hash: str) -> dict:
    prefix = f"T{temperature}"
    with np.load(path, allow_pickle=False) as data:
        result = {
            "path": str(path),
            "sha256": sha256(path),
            "fc2": np.asarray(data[f"{prefix}_fc2"], float),
            "mean_temperature_K": float(scalar(data, f"{prefix}_mean_temperature_K")),
            "max_temperature_K": float(scalar(data, f"{prefix}_max_temperature_K")),
            "fit_RMSE_meV_A": float(scalar(data, f"{prefix}_fit_rmse_meV_A")),
            "source_checkpoint_counts": np.asarray(
                data["source_checkpoint_counts"], int
            ).tolist(),
            "subtracted_operator_sha256": str(
                scalar(data, "subtracted_operator_sha256")
            ),
        }
    if result["subtracted_operator_sha256"] != operator_hash:
        raise ValueError(f"{path} subtracted a different physical operator")
    if not np.isfinite(result["fc2"]).all():
        raise ValueError(f"non-finite short TDEP force constants in {path}")
    return result


def spectrum(phonon, force_constants: np.ndarray, npoints: int):
    distance, frequency, label_positions, labels = tdp.band_from_phonopy(
        phonon, force_constants, npoints=npoints
    )
    frequency = np.asarray(frequency, float) * CM_PER_THZ
    if not np.isfinite(frequency).all():
        raise ValueError("physical spectrum contains non-finite frequencies")
    return np.asarray(distance, float), frequency, label_positions, labels


def full_band_mae(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("seed spectrum shapes differ")
    return float(np.mean(np.abs(left - right)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--short-tdep", type=Path, action="append", required=True)
    parser.add_argument("--pooled-short-tdep", type=Path, required=True)
    parser.add_argument("--expected-snapshots-per-seed", type=int, required=True)
    parser.add_argument("--mode", choices=("pilot", "final"), required=True)
    parser.add_argument("--npoints", type=int, default=201)
    parser.add_argument("--physical-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.checkpoint) != 3 or len(args.short_tdep) != 3:
        parser.error("exactly three checkpoints and short-TDEP files are required")
    freeze = json.loads(args.freeze_manifest.read_text())
    if freeze.get("status") != "frozen_for_L0_development":
        raise ValueError("invalid L0 freeze manifest")
    if args.temperature not in freeze["development_temperatures_K"]:
        raise ValueError("temperature is outside the frozen development set")
    expected = (
        freeze["trajectory_protocol"]["pilot_snapshots_per_seed"]
        if args.mode == "pilot"
        else freeze["trajectory_protocol"]["final_snapshots_per_seed"]
    )
    if args.expected_snapshots_per_seed != expected:
        raise ValueError("requested snapshot count differs from the frozen protocol")

    operator_record = freeze["operators"][str(args.temperature)]
    operator_path = Path(operator_record["path"])
    operator_hash = sha256(operator_path)
    if operator_hash != operator_record["sha256"]:
        raise ValueError("physical operator hash changed after L0 freeze")
    with np.load(operator_path, allow_pickle=False) as data:
        operator_fc = np.asarray(data["delta_fc_full"], float)
        mapping = np.asarray(data["atom_mapping"], int)
    if not np.array_equal(mapping, np.arange(len(mapping))):
        raise ValueError("L0 physical-spectrum assembly requires identity atom mapping")

    background_path = Path(freeze["background"]["path"])
    if sha256(background_path) != freeze["background"]["sha256"]:
        raise ValueError("background hash changed after L0 freeze")
    phonon = fm.load_ph(background_path)
    seeds = [
        load_short_tdep(path, args.temperature, operator_hash)
        for path in args.short_tdep
    ]
    pooled = load_short_tdep(args.pooled_short_tdep, args.temperature, operator_hash)
    for seed in seeds:
        if seed["source_checkpoint_counts"] != [expected]:
            raise ValueError("seed short-TDEP source count differs from frozen protocol")
    if pooled["source_checkpoint_counts"] != [expected, expected, expected]:
        raise ValueError("pooled short-TDEP source counts differ from frozen protocol")

    checkpoint_records = []
    for seed, path in enumerate(args.checkpoint):
        snapshots = path / "snapshots.npz" if path.is_dir() else path
        with np.load(snapshots, allow_pickle=False) as data:
            count = len(np.asarray(data["positions"]))
        if count != expected:
            raise ValueError(f"seed {seed} checkpoint contains {count}, expected {expected}")
        checkpoint_records.append(
            {
                "seed": seed,
                "path": str(path),
                "snapshots_path": str(snapshots),
                "snapshots_sha256": sha256(snapshots),
                "n_snapshots": count,
            }
        )

    physical_arrays = {
        "temperature_K": np.array(args.temperature),
        "degauss_Ry": np.array(operator_record["degauss_Ry"]),
        "operator_sha256": np.array(operator_hash),
        "freeze_manifest_sha256": np.array(sha256(args.freeze_manifest)),
        "mode": np.array(args.mode),
    }
    physical_frequencies = []
    label_positions = labels = None
    for seed, record in enumerate(seeds):
        physical_fc2 = record["fc2"] + operator_fc
        distance, frequency, label_positions, labels = spectrum(
            phonon, physical_fc2, args.npoints
        )
        physical_arrays[f"seed{seed}_dist"] = distance
        physical_arrays[f"seed{seed}_frequency_cm-1"] = frequency
        physical_arrays[f"seed{seed}_fc2"] = physical_fc2
        physical_frequencies.append(frequency)
    pooled_physical_fc2 = pooled["fc2"] + operator_fc
    pooled_distance, pooled_frequency, label_positions, labels = spectrum(
        phonon, pooled_physical_fc2, args.npoints
    )
    physical_arrays["pooled_dist"] = pooled_distance
    physical_arrays["pooled_frequency_cm-1"] = pooled_frequency
    physical_arrays["pooled_fc2"] = pooled_physical_fc2
    physical_arrays["label_positions"] = label_positions
    physical_arrays["labels"] = labels
    atomic_npz(args.physical_output, **physical_arrays)

    pair_records = [
        {
            "seeds": [left, right],
            "full_band_MAE_cm-1": full_band_mae(
                physical_frequencies[left], physical_frequencies[right]
            ),
        }
        for left, right in combinations(range(3), 2)
    ]
    temperature_records = [
        {
            "seed": seed,
            "mean_temperature_K": record["mean_temperature_K"],
            "max_temperature_K": record["max_temperature_K"],
            "mean_temperature_relative_error": abs(
                record["mean_temperature_K"] - args.temperature
            )
            / args.temperature,
        }
        for seed, record in enumerate(seeds)
    ]
    thresholds = freeze["fixed_acceptance_thresholds"]
    temperature_threshold = (
        freeze["trajectory_protocol"][
            "pilot_stability_mean_temperature_relative_error_max"
        ]
        if args.mode == "pilot"
        else thresholds["MD_mean_temperature_relative_error"]
    )
    temperature_pass = max(
        record["mean_temperature_relative_error"] for record in temperature_records
    ) <= temperature_threshold
    spread_pass = max(record["full_band_MAE_cm-1"] for record in pair_records) < float(
        thresholds["seed_pair_full_band_MAE_cm-1"]
    )
    passed = bool(temperature_pass and (spread_pass if args.mode == "final" else True))
    payload = {
        "status": "passed" if passed else "failed",
        "scope": (
            f"S0 physical L0 {args.temperature} K "
            + ("120-snapshot stability pilot" if args.mode == "pilot" else "fixed 3000-snapshot point gate")
        ),
        "mode": args.mode,
        "temperature_K": args.temperature,
        "degauss_Ry": operator_record["degauss_Ry"],
        "degauss_formula": "k_B*T/Ry",
        "expected_snapshots_per_seed": expected,
        "thresholds": {
            "seed_pair_full_band_MAE_cm-1": thresholds[
                "seed_pair_full_band_MAE_cm-1"
            ],
            "MD_mean_temperature_relative_error": temperature_threshold,
            "seed_spread_applied_as_gate": args.mode == "final",
        },
        "passes_seed_spread_gate": spread_pass,
        "passes_temperature_gate": temperature_pass,
        "passes_sampling_gate": passed,
        "seed_pair_spread": pair_records,
        "seed_temperature_diagnostics": temperature_records,
        "l0_freeze_manifest": {
            "path": str(args.freeze_manifest),
            "sha256": sha256(args.freeze_manifest),
        },
        "sampling_operator": {"path": str(operator_path), "sha256": operator_hash},
        "checkpoints": checkpoint_records,
        "seed_tdeps": [
            {
                key: value
                for key, value in {"seed": seed, **record}.items()
                if key != "fc2"
            }
            for seed, record in enumerate(seeds)
        ],
        "pooled_short_tdep": {
            key: value for key, value in pooled.items() if key != "fc2"
        },
        "physical_tdep": {
            "path": str(args.physical_output),
            "sha256": sha256(args.physical_output),
            "assembly": "short-range TDEP FC2 + exact physical electronic operator FC2",
        },
        "next_stage": (
            "run fixed 3000-snapshot extension"
            if args.mode == "pilot" and passed
            else "run the frozen 1000-replicate block bootstrap"
            if args.mode == "final" and passed
            else "stop and diagnose; do not append samples automatically"
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
