#!/usr/bin/env python3
"""Release wave 3 early when wave-2 held-out forces already fail its gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rmse-threshold", type=float, default=50.0)
    parser.add_argument("--max-threshold", type=float, default=250.0)
    parser.add_argument("--final-wave", action="store_true")
    parser.add_argument("--trajectory-error", action="append", default=[])
    args = parser.parse_args()

    force = json.loads(args.force_validation.read_text())
    by_temperature = {}
    failed = False
    for temperature, label in ((300, "fd300"), (600, "fd600")):
        metrics = force["models"][label]["by_temperature"][str(temperature)]
        passed = (
            float(metrics["RMSE_meV_A"]) <= args.rmse_threshold
            and float(metrics["max_abs_meV_A"]) <= args.max_threshold
        )
        failed = failed or not passed
        by_temperature[str(temperature)] = {
            "force_validation": metrics,
            "passes_force_gate": passed,
        }

    failed = failed or bool(args.trajectory_error)
    if not failed:
        print("wave2 force gate passes; waiting for the complete TDEP gate")
        return 0
    payload = {
        "status": (
            "final_short_range_failure" if args.final_wave else "early_force_gate"
        ),
        "thresholds": {
            "force_RMSE_meV_A": args.rmse_threshold,
            "force_max_abs_meV_A": args.max_threshold,
        },
        "by_temperature": by_temperature,
        "passes_short_range_gate": False,
        "trajectory_stability_errors": args.trajectory_error,
        "wave3_required": not args.final_wave,
        "next_step": (
            "compare training, replay, and model variants before long-range fitting"
            if args.final_wave
            else (
                "run conditional third physical-FD label wave while wave2 TDEP "
                "diagnostics finish"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
