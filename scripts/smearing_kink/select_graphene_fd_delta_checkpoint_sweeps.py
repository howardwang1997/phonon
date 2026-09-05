#!/usr/bin/env python3
"""Select one pre-holdout model across weighted checkpoint sweeps."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for specification in args.candidate:
        if "=" not in specification:
            parser.error("--candidate must use TAG=/path/to/checkpoint_sweep.json")
        tag, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        records.append(
            {
                "tag": tag,
                "sweep_path": str(path),
                "sweep_status": payload["status"],
                "passes_fixed_development_gate": payload["status"]
                == "checkpoint_passed",
                "selected_model": payload["selected_model"],
                "selected_checkpoint": payload["selected"]["checkpoint"],
                "selected_epoch": payload["selected"]["epoch"],
                "normalized_worst_gate_score": payload["selected"][
                    "normalized_worst_gate_score"
                ],
                "thermal_total_force": payload["selected"]["thermal_total_force"],
                "harmonic_combined_force": payload["selected"][
                    "harmonic_combined_force"
                ],
            }
        )
    passing = [item for item in records if item["passes_fixed_development_gate"]]
    pool = passing if passing else records
    selected = min(pool, key=lambda item: item["normalized_worst_gate_score"])
    result = {
        "status": "development_candidate_found" if passing else "no_candidate_passed",
        "selection_scope": (
            "model, replay weight, and epoch frozen before reading the new 15-structure "
            "600 K physical-FD force holdout"
        ),
        "fixed_thresholds_unchanged": True,
        "selected": selected,
        "candidates": sorted(
            records, key=lambda item: item["normalized_worst_gate_score"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
