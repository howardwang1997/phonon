#!/usr/bin/env python3
"""Summarize development-only MACE throughput across fixed batch sizes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def training_steps(path: Path, batch_size: int, n_train: int) -> dict[str, object]:
    times: list[float] = []
    epochs: set[int] = set()
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("mode") != "opt":
                continue
            step_time = float(record["time"])
            if step_time <= 0:
                continue
            times.append(step_time)
            epochs.add(int(record["epoch"]))
    if not times:
        return {"optimizer_steps": 0, "epochs_observed": []}
    return {
        "optimizer_steps": len(times),
        "epochs_observed": sorted(epochs),
        "median_step_seconds": statistics.median(times),
        "mean_step_seconds": statistics.fmean(times),
        "configs_per_second_from_step_times": n_train * len(epochs) / sum(times),
    }


def gpu_samples(path: Path) -> dict[str, object]:
    utilization: list[float] = []
    memory_mib: list[float] = []
    power_w: list[float] = []
    if not path.is_file():
        return {"n_samples": 0}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                utilization.append(float(row["utilization_gpu_percent"]))
                memory_mib.append(float(row["memory_used_MiB"]))
                power_w.append(float(row["power_draw_W"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not utilization:
        return {"n_samples": 0}
    return {
        "n_samples": len(utilization),
        "mean_utilization_percent": statistics.fmean(utilization),
        "max_utilization_percent": max(utilization),
        "max_memory_used_MiB": max(memory_mib),
        "max_power_draw_W": max(power_w),
    }


def read_status(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        record = next(csv.DictReader(handle, delimiter="\t"))
    return {
        "train_exit_code": int(record["train_exit_code"]),
        "start": record["start_iso"],
        "end": record["end_iso"],
        "elapsed_seconds": int(record["elapsed_seconds"]),
    }


def candidate(directory: Path, batch_size: int, n_train: int) -> dict[str, object]:
    status_path = directory / "status.tsv"
    train_files = sorted((directory / "results").glob("*_train.txt"))
    if not status_path.is_file() or len(train_files) != 1:
        return {
            "batch_size": batch_size,
            "status": "incomplete",
            "directory": str(directory),
        }
    status = read_status(status_path)
    result = {
        "batch_size": batch_size,
        "status": "complete" if status["train_exit_code"] == 0 else "failed",
        "timing": status,
        "training": training_steps(train_files[0], batch_size, n_train),
        "gpu": gpu_samples(directory / "gpu_samples.csv"),
        "artifacts": {
            "status": artifact(status_path),
            "train_metrics": artifact(train_files[0]),
            "gpu_samples": artifact(directory / "gpu_samples.csv"),
        },
    }
    models = sorted(directory.glob("*.model"))
    result["models"] = [artifact(path) for path in models]
    return result


def baseline(directory: Path, n_train: int) -> dict[str, object]:
    summary_path = directory / "benchmark_summary.json"
    summary = json.loads(summary_path.read_text())
    train_files = sorted((directory / "results").glob("*_train.txt"))
    if len(train_files) != 1:
        raise RuntimeError("expected exactly one batch-1 train metrics file")
    return {
        "batch_size": 1,
        "status": summary["status"],
        "timing": summary["timing"],
        "training": training_steps(train_files[0], 1, n_train),
        "gpu": summary["gpu"],
        "artifacts": {
            "summary": artifact(summary_path),
            "train_metrics": artifact(train_files[0]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-head", default="unknown")
    args = parser.parse_args()

    data_manifest = json.loads(args.data_manifest.read_text())
    n_train = int(data_manifest["splits"]["train"]["n_output_structures"])
    results = [baseline(args.baseline_dir, n_train)]
    for batch_size in (4, 8, 16, 32):
        results.append(
            candidate(args.sweep_root / f"batch{batch_size}", batch_size, n_train)
        )
    complete = [
        item
        for item in results
        if item["status"] == "complete"
        and item["training"].get("configs_per_second_from_step_times") is not None
    ]
    baseline_rate = float(results[0]["training"]["configs_per_second_from_step_times"])
    for item in complete:
        item["throughput_speedup_vs_batch1"] = (
            float(item["training"]["configs_per_second_from_step_times"]) / baseline_rate
        )
    best = max(
        complete,
        key=lambda item: float(item["training"]["configs_per_second_from_step_times"]),
    )
    payload = {
        "status": "complete" if len(complete) == 5 else "partial",
        "scope": "V100-B MACE batch-size throughput sizing on 300/600 K development data",
        "scientific_use": "resource sizing only; batch choice is not a scientific model-selection result",
        "target_leakage": {
            "p4_450K_inputs_read": False,
            "allowed_input_temperatures_K": [300, 600],
        },
        "configuration": {
            "candidate_batches": [1, 4, 8, 16, 32],
            "new_candidate_epochs": 3,
            "n_train_structures": n_train,
            "architecture": "MACE, 2 interactions, 32x0e+32x1o, r_max=5.0",
            "seed": 83,
        },
        "candidates": results,
        "throughput_best": {
            "batch_size": best["batch_size"],
            "configs_per_second_from_step_times": best["training"][
                "configs_per_second_from_step_times"
            ],
            "speedup_vs_batch1": best["throughput_speedup_vs_batch1"],
            "max_memory_used_MiB": best["gpu"].get("max_memory_used_MiB"),
            "limitation": (
                "Before formal S0 training, freeze batch size together with learning-rate "
                "and gradient-equivalence checks on the complete development dataset."
            ),
        },
        "provenance": {
            "git_head": args.git_head,
            "data_manifest": artifact(args.data_manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
