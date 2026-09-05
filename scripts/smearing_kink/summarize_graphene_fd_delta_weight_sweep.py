#!/usr/bin/env python3
"""Summarize a fixed-weight 600 K delta-MACE development sweep.

The three-structure thermal set used here has already been inspected during
the earlier delta pilot.  It is therefore a development gate, not a new final
holdout.  This script preserves that distinction in the machine-readable
output while keeping the original numerical thresholds unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def normalized_score(payload: dict) -> float:
    thresholds = payload["thresholds"]
    thermal = payload["thermal_total_force"]
    harmonic = payload["harmonic_combined_force"]
    harmonic_limit = (
        thresholds["harmonic_RMSE_relative_to_frozen_v11"]
        * payload["harmonic_v11_force"]["RMSE_meV_A"]
    )
    return max(
        thermal["RMSE_meV_A"] / thresholds["thermal_force_RMSE_meV_A"],
        thermal["max_abs_meV_A"] / thresholds["thermal_force_max_abs_meV_A"],
        harmonic["RMSE_meV_A"] / harmonic_limit,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for specification in args.candidate:
        if "=" not in specification:
            parser.error("--candidate must use TAG=/path/to/gate_selection.json")
        tag, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        records.append(
            {
                "tag": tag,
                "selection_path": str(path),
                "passes_fixed_development_gate": bool(
                    payload["passes_force_and_replay_gate"]
                ),
                "normalized_worst_gate_score": normalized_score(payload),
                "thermal_total_force": payload["thermal_total_force"],
                "harmonic_combined_force": payload["harmonic_combined_force"],
                "harmonic_v11_force": payload["harmonic_v11_force"],
                "thresholds": payload["thresholds"],
            }
        )
    records.sort(key=lambda item: item["normalized_worst_gate_score"])
    passing = [item for item in records if item["passes_fixed_development_gate"]]
    selected = passing[0] if passing else records[0]
    result = {
        "status": "development_candidate_found" if passing else "no_candidate_passed",
        "selection_scope": (
            "development only: the three thermal structures were used in the prior "
            "delta-pilot model comparison; an independent physical-FD force holdout "
            "is required before final acceptance"
        ),
        "fixed_thresholds_unchanged": True,
        "selected_tag": selected["tag"],
        "selected_normalized_worst_gate_score": selected[
            "normalized_worst_gate_score"
        ],
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
