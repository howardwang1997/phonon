#!/usr/bin/env python3
"""Summarize the 300/600 K development-only MACE epoch learning curve."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MILESTONES = (60, 120, 240)
GROUPS = {
    "thermal300_test": "test:physical_fd_thermal_300K_test",
    "thermal600_test": "test:physical_fd_thermal_600K_test",
    "harmonic_replay_val": "val:v11_fc_distillation_replay_validation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def extract_metrics(evaluation: dict[str, object]) -> dict[str, dict[str, float]]:
    grouped = evaluation["grouped_metrics"]
    missing = [source for source in GROUPS.values() if source not in grouped]
    if missing:
        raise KeyError(f"missing grouped metrics: {missing}")
    return {
        name: {
            "RMSE_meV_A": float(grouped[source]["force_RMSE_meV_A"]),
            "max_abs_meV_A": float(grouped[source]["force_max_abs_meV_A"]),
        }
        for name, source in GROUPS.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-head", default="unknown")
    args = parser.parse_args()

    control_payload = json.loads(args.control_summary.read_text())
    if control_payload.get("status") != "complete":
        raise ValueError("30-epoch control summary is not complete")
    if control_payload.get("target_leakage", {}).get("p4_450K_inputs_read") is not False:
        raise ValueError("30-epoch control does not prove P4 isolation")
    control = next(
        item
        for item in control_payload["candidates"]
        if item["tag"] == "batch1_lr0p001" and item["status"] == "complete"
    )
    control_metrics = control["grouped_metrics"]
    baseline = {
        name: {
            "RMSE_meV_A": float(control_metrics[source]["force_RMSE_meV_A"]),
            "max_abs_meV_A": float(control_metrics[source]["force_max_abs_meV_A"]),
        }
        for name, source in GROUPS.items()
    }
    replay_limit = float(
        control["equivalence_limits_meV_A"]["harmonic_replay_val"]
    )

    points = []
    for epoch in MILESTONES:
        evaluation_path = args.root / "evaluations" / f"epoch{epoch:03d}.json"
        model_path = args.root / "models" / f"epoch{epoch:03d}.model"
        if not evaluation_path.is_file() or not model_path.is_file():
            points.append({"epoch": epoch, "status": "incomplete"})
            continue
        metrics = extract_metrics(json.loads(evaluation_path.read_text()))
        thermal_pass = all(
            metrics[name]["RMSE_meV_A"] <= 50.0
            and metrics[name]["max_abs_meV_A"] <= 250.0
            for name in ("thermal300_test", "thermal600_test")
        )
        replay_pass = metrics["harmonic_replay_val"]["RMSE_meV_A"] <= replay_limit
        points.append(
            {
                "epoch": epoch,
                "status": "complete",
                "metrics": metrics,
                "relative_RMSE_change_vs_epoch30": {
                    name: metrics[name]["RMSE_meV_A"] / baseline[name]["RMSE_meV_A"] - 1.0
                    for name in GROUPS
                },
                "passes_proxy_force_gate": bool(thermal_pass and replay_pass),
                "artifacts": {
                    "model": artifact(model_path),
                    "evaluation": artifact(evaluation_path),
                },
            }
        )

    complete = [point for point in points if point["status"] == "complete"]
    passing = [point for point in complete if point["passes_proxy_force_gate"]]
    payload = {
        "status": "complete" if len(complete) == len(MILESTONES) else "partial",
        "scope": "epoch learning curve on 300/600 K development data",
        "scientific_use": "S0 feasibility and walltime sizing only; no P4 or validation claim",
        "target_leakage": {"p4_450K_inputs_read": False},
        "fixed_protocol": {
            "epochs_evaluated": list(MILESTONES),
            "batch_size": 1,
            "learning_rate": 0.001,
            "seed": 83,
            "thermal_force_gate": "each 300/600 K test split RMSE <=50 and max <=250 meV/A",
            "harmonic_replay_gate": f"RMSE <= {replay_limit:.12g} meV/A",
        },
        "epoch30_control": {
            "metrics": baseline,
            "artifact": artifact(args.control_summary),
        },
        "learning_curve": points,
        "earliest_passing_epoch": min((point["epoch"] for point in passing), default=None),
        "scope_limit": (
            "The 300/600 K test groups contain three structures each. Repeat the gate on the "
            "complete physics-subtracted S0 development set before freezing a production model."
        ),
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
