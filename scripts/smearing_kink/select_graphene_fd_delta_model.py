#!/usr/bin/env python3
"""Apply independent thermal-force and harmonic-retention gates to a delta model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--thermal-label", required=True)
    parser.add_argument("--harmonic-label", default="harmonic")
    parser.add_argument("--force-rmse-threshold", type=float, default=50.0)
    parser.add_argument("--force-max-threshold", type=float, default=250.0)
    parser.add_argument("--harmonic-rmse-ratio", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.metrics.read_text())
    thermal = payload["datasets"][args.thermal_label]["reconstructed_total_error"]
    harmonic_dataset = payload["datasets"][args.harmonic_label]
    harmonic = harmonic_dataset["reconstructed_total_error"]
    baseline = harmonic_dataset["base_only_short_range_error"]
    passed = (
        thermal["RMSE_meV_A"] <= args.force_rmse_threshold
        and thermal["max_abs_meV_A"] <= args.force_max_threshold
        and harmonic["RMSE_meV_A"]
        <= args.harmonic_rmse_ratio * baseline["RMSE_meV_A"]
    )
    result = {
        "passes_force_and_replay_gate": passed,
        "thresholds": {
            "thermal_force_RMSE_meV_A": args.force_rmse_threshold,
            "thermal_force_max_abs_meV_A": args.force_max_threshold,
            "harmonic_RMSE_relative_to_frozen_v11": args.harmonic_rmse_ratio,
        },
        "thermal_total_force": thermal,
        "harmonic_combined_force": harmonic,
        "harmonic_v11_force": baseline,
        "metrics_path": str(args.metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
