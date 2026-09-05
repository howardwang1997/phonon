#!/usr/bin/env python3
"""Select a speed/quality-equivalent batch configuration on development data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path


CANDIDATES = (
    ("batch1_lr0p001", 1, 0.001),
    ("batch8_lrsqrt", 8, 0.00282842712474619),
    ("batch16_lrsqrt", 16, 0.004),
    ("batch32_lrsqrt", 32, 0.00565685424949238),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def status(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    return {
        "train_exit_code": int(row["train_exit_code"]),
        "start": row["start_iso"],
        "end": row["end_iso"],
        "elapsed_seconds": int(row["elapsed_seconds"]),
    }


def gpu(path: Path) -> dict[str, object]:
    utilization: list[float] = []
    memory: list[float] = []
    power: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                utilization.append(float(row["utilization_gpu_percent"]))
                memory.append(float(row["memory_used_MiB"]))
                power.append(float(row["power_draw_W"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not utilization:
        return {"n_samples": 0}
    return {
        "n_samples": len(utilization),
        "mean_utilization_percent": statistics.fmean(utilization),
        "max_utilization_percent": max(utilization),
        "max_memory_used_MiB": max(memory),
        "max_power_draw_W": max(power),
    }


def load_candidate(root: Path, tag: str, batch_size: int, lr: float) -> dict[str, object]:
    lane = root / tag
    status_path = lane / "status.tsv"
    evaluation_path = lane / "development_evaluation.json"
    gpu_path = lane / "gpu_samples.csv"
    if not all(path.is_file() for path in (status_path, evaluation_path, gpu_path)):
        return {"tag": tag, "batch_size": batch_size, "lr": lr, "status": "incomplete"}
    timing = status(status_path)
    evaluation = json.loads(evaluation_path.read_text())
    return {
        "tag": tag,
        "batch_size": batch_size,
        "lr": lr,
        "status": "complete" if timing["train_exit_code"] == 0 else "failed",
        "timing": timing,
        "gpu": gpu(gpu_path),
        "grouped_metrics": evaluation["grouped_metrics"],
        "artifacts": {
            "status": artifact(status_path),
            "evaluation": artifact(evaluation_path),
            "gpu_samples": artifact(gpu_path),
        },
    }


def relevant_metrics(candidate: dict[str, object]) -> dict[str, float]:
    grouped = candidate["grouped_metrics"]
    keys = {
        "thermal300_test": "test:physical_fd_thermal_300K_test",
        "thermal600_test": "test:physical_fd_thermal_600K_test",
        "harmonic_replay_val": "val:v11_fc_distillation_replay_validation",
    }
    missing = [source for source in keys.values() if source not in grouped]
    if missing:
        raise KeyError(f"missing grouped metrics: {missing}")
    return {
        name: float(grouped[source]["force_RMSE_meV_A"])
        for name, source in keys.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-head", default="unknown")
    args = parser.parse_args()

    candidates = [load_candidate(args.root, *spec) for spec in CANDIDATES]
    complete = [item for item in candidates if item["status"] == "complete"]
    if len(complete) != len(CANDIDATES):
        overall_status = "partial"
    else:
        overall_status = "complete"
    control = next(item for item in complete if item["batch_size"] == 1)
    control_metrics = relevant_metrics(control)
    control_wall = float(control["timing"]["elapsed_seconds"])
    for item in complete:
        metrics = relevant_metrics(item)
        endpoint_limits = {
            name: max(control_metrics[name] * 1.05, control_metrics[name] + 5.0)
            for name in ("thermal300_test", "thermal600_test")
        }
        replay_limit = max(
            control_metrics["harmonic_replay_val"] * 1.05,
            control_metrics["harmonic_replay_val"] + 2.0,
        )
        item["comparison_metrics"] = metrics
        item["equivalence_limits_meV_A"] = {
            **endpoint_limits,
            "harmonic_replay_val": replay_limit,
        }
        item["quality_equivalent_to_batch1"] = bool(
            all(metrics[name] <= endpoint_limits[name] for name in endpoint_limits)
            and metrics["harmonic_replay_val"] <= replay_limit
        )
        item["walltime_speedup_vs_batch1"] = control_wall / float(
            item["timing"]["elapsed_seconds"]
        )
    equivalent = [item for item in complete if item["quality_equivalent_to_batch1"]]
    selected = min(equivalent, key=lambda item: float(item["timing"]["elapsed_seconds"]))
    payload = {
        "status": overall_status,
        "scope": "30-epoch batch/LR convergence equivalence on 300/600 K development data",
        "scientific_use": "training-configuration sizing only; no P4 or validation claim",
        "target_leakage": {"p4_450K_inputs_read": False},
        "fixed_protocol": {
            "epochs": 30,
            "seed": 83,
            "learning_rate_rule": "lr(batch)=0.001*sqrt(batch)",
            "thermal_equivalence": "RMSE no worse than batch1 by both >5% and >5 meV/A",
            "replay_equivalence": "RMSE no worse than batch1 by both >5% and >2 meV/A",
        },
        "candidates": candidates,
        "selected_for_next_development_benchmark": {
            "tag": selected["tag"],
            "batch_size": selected["batch_size"],
            "lr": selected["lr"],
            "walltime_speedup_vs_batch1": selected["walltime_speedup_vs_batch1"],
            "scope_limit": (
                "This does not freeze the formal S0 training configuration; repeat on the "
                "complete physics-subtracted S0 development set before validation generation."
            ),
        },
        "provenance": {
            "git_head": args.git_head,
            "data_manifest": artifact(args.data_manifest),
        },
    }
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
