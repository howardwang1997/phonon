#!/usr/bin/env python3
"""Select temperature-specific models using fixed force and replay gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--candidate-300", action="append")
    parser.add_argument("--candidate-600", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-rmse-threshold", type=float, default=50.0)
    parser.add_argument("--force-max-threshold", type=float, default=250.0)
    parser.add_argument("--harmonic-rmse-ratio", type=float, default=2.0)
    args = parser.parse_args()
    if not args.candidate_300 and not args.candidate_600:
        parser.error("provide at least one temperature's candidate list")

    payload = json.loads(args.metrics.read_text())
    baseline_harmonic = payload["models"]["v11"]["datasets"]["harmonic"][
        "RMSE_meV_A"
    ]
    result = {
        "thresholds": {
            "thermal_force_RMSE_meV_A": args.force_rmse_threshold,
            "thermal_force_max_abs_meV_A": args.force_max_threshold,
            "harmonic_RMSE_relative_to_v11": args.harmonic_rmse_ratio,
        },
        "v11_harmonic_RMSE_meV_A": baseline_harmonic,
        "by_temperature": {},
    }
    all_pass = True
    requested = [
        (temperature, specifications)
        for temperature, specifications in (
            (300, args.candidate_300),
            (600, args.candidate_600),
        )
        if specifications
    ]
    for temperature, specifications in requested:
        dataset = f"thermal{temperature}"
        records = []
        for label in specifications:
            model = payload["models"][label]
            thermal = model["datasets"][dataset]
            harmonic = model["datasets"]["harmonic"]
            passes = (
                thermal["RMSE_meV_A"] <= args.force_rmse_threshold
                and thermal["max_abs_meV_A"] <= args.force_max_threshold
                and harmonic["RMSE_meV_A"]
                <= args.harmonic_rmse_ratio * baseline_harmonic
            )
            score = (
                thermal["RMSE_meV_A"] / args.force_rmse_threshold
                + thermal["max_abs_meV_A"] / args.force_max_threshold
                + 0.25 * harmonic["RMSE_meV_A"] / max(baseline_harmonic, 1e-12)
            )
            records.append(
                {
                    "label": label,
                    "path": model["path"],
                    "sha256": model["sha256"],
                    "thermal_force": thermal,
                    "harmonic_force": harmonic,
                    "passes_force_and_replay_gate": passes,
                    "ranking_score": score,
                }
            )
        passing = [record for record in records if record["passes_force_and_replay_gate"]]
        selected = min(passing or records, key=lambda record: record["ranking_score"])
        all_pass = all_pass and bool(passing)
        result["by_temperature"][str(temperature)] = {
            "selected": selected,
            "candidates": records,
            "has_passing_candidate": bool(passing),
        }
    result["passes_all_requested_temperature_gates"] = all_pass
    result["passes_both_temperature_gates"] = all_pass and len(requested) == 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
