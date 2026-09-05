#!/usr/bin/env python3
"""Write a provenance-rich summary for the bounded MACE throughput benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
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


def parse_gpu_samples(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        return {"n_samples": 0}
    utilization: list[float] = []
    memory_mib: list[float] = []
    power_w: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
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
        "mean_utilization_percent": sum(utilization) / len(utilization),
        "max_utilization_percent": max(utilization),
        "max_memory_used_MiB": max(memory_mib),
        "max_power_draw_W": max(power_w),
    }


def parse_resource_usage(path: Path) -> dict[str, object]:
    output: dict[str, object] = {}
    if not path.is_file():
        return output
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if normalized in {
            "elapsed_(wall_clock)_time_(h:mm:ss_or_m:ss)",
            "maximum_resident_set_size_(kbytes)",
            "percent_of_cpu_this_job_got",
            "file_system_inputs",
            "file_system_outputs",
        }:
            output[normalized] = value
    return output


def checkpoint_epochs(checkpoint_dir: Path) -> list[int]:
    epochs: set[int] = set()
    for path in checkpoint_dir.glob("*.pt"):
        match = re.search(r"_epoch-(\d+)\.pt$", path.name)
        if match:
            epochs.add(int(match.group(1)))
    return sorted(epochs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--resource-usage", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--start-iso", required=True)
    parser.add_argument("--end-iso", required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    parser.add_argument("--train-exit-code", type=int, required=True)
    parser.add_argument("--git-head", default="unknown")
    parser.add_argument("--configured-epochs", type=int, default=10)
    args = parser.parse_args()

    status = {
        0: "complete",
        124: "bounded_timeout",
        137: "bounded_timeout",
    }.get(args.train_exit_code, "failed")
    models = sorted(args.output_dir.glob("*.model"))
    payload = {
        "status": status,
        "scope": "V100-B 300/600 K development-only MACE throughput benchmark",
        "scientific_use": "runtime and resource sizing only; not a model-selection or acceptance result",
        "target_leakage": {
            "p4_450K_inputs_read": False,
            "allowed_input_temperatures_K": [300, 600],
            "excluded_scope": "all 450 K P4 structures, labels, DFPT values, and acceptance outputs",
        },
        "configuration": {
            "configured_epochs": args.configured_epochs,
            "seed": 83,
            "architecture": "MACE, 2 interactions, 32x0e+32x1o, r_max=5.0",
            "batch_size": 1,
            "eval_interval": 5,
            "hard_walltime_seconds": 7200,
        },
        "timing": {
            "start": args.start_iso,
            "end": args.end_iso,
            "elapsed_seconds": args.elapsed_seconds,
            "train_exit_code": args.train_exit_code,
        },
        "gpu": parse_gpu_samples(args.gpu_samples),
        "resource_usage": parse_resource_usage(args.resource_usage),
        "checkpoint_epochs": checkpoint_epochs(args.output_dir / "checkpoints"),
        "provenance": {
            "git_head": args.git_head,
            "joint_manifest": artifact(args.joint_manifest),
            "source_manifest": artifact(args.source_manifest),
            "run_log": artifact(args.run_log),
            "gpu_samples": artifact(args.gpu_samples),
        },
        "models": [artifact(path) for path in models],
    }
    destination = args.output_dir / "benchmark_summary.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
