#!/usr/bin/env python3
"""Select the frozen joint endpoint model from the predefined replay sweeps."""
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

    candidates = []
    for specification in args.candidate:
        if "=" not in specification:
            parser.error("--candidate must use LABEL=/path")
        label, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        candidates.append(
            {
                "label": label,
                "sweep_path": str(path),
                "sweep_status": payload["status"],
                "passes_fixed_endpoint_gate": payload["status"] == "checkpoint_passed",
                "selected_model": payload["selected_model"],
                "selected_checkpoint": payload["selected"]["checkpoint"],
                "selected_epoch": payload["selected"]["epoch"],
                "normalized_worst_gate_score": payload["selected"][
                    "normalized_worst_gate_score"
                ],
                "thermal_total_force": payload["selected"]["thermal_total_force"],
                "harmonic_combined_force": payload["selected"]["harmonic_combined_force"],
            }
        )
    passing = [item for item in candidates if item["passes_fixed_endpoint_gate"]]
    selected = min(
        passing if passing else candidates,
        key=lambda item: item["normalized_worst_gate_score"],
    )
    result = {
        "status": "joint_endpoint_candidate_found" if passing else "no_joint_candidate_passed",
        "selection_scope": (
            "300/600 K endpoint development data only; frozen before generating or "
            "labeling any 450 K configurations"
        ),
        "selected": selected,
        "candidates": sorted(candidates, key=lambda item: item["normalized_worst_gate_score"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
