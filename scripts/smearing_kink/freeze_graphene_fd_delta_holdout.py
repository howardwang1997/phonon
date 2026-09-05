#!/usr/bin/env python3
"""Record a frozen 600 K model before the independent holdout is exposed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--expected-indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exposed_markers = [
        args.holdout_root / "RAW_READY",
        args.holdout_root / "shard_A" / "RAW_READY",
        args.holdout_root / "shard_B" / "RAW_READY",
    ]
    exposed = [str(path) for path in exposed_markers if path.exists()]
    if exposed:
        raise RuntimeError(f"holdout was already exposed before freeze: {exposed}")
    selection = json.loads(args.selection.read_text())
    if selection["status"] != "development_candidate_found":
        raise ValueError("selection did not produce a passing development candidate")
    selected = selection["selected"]
    selected_checkpoint = Path(selected["selected_checkpoint"])
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(selected_checkpoint)
    expected_indices = [int(value) for value in args.expected_indices.split(",")]
    if len(expected_indices) != len(set(expected_indices)):
        raise ValueError("expected holdout indices contain duplicates")

    payload = {
        "status": "frozen_before_independent_holdout_exposure",
        "frozen_at": datetime.now().astimezone().isoformat(),
        "selection": str(args.selection),
        "selection_sha256": sha256(args.selection),
        "selected_tag": selected["tag"],
        "selected_epoch": selected["selected_epoch"],
        "selected_checkpoint": selected["selected_checkpoint"],
        "selected_checkpoint_sha256": sha256(selected_checkpoint),
        "development_metrics": {
            "thermal_total_force": selected["thermal_total_force"],
            "harmonic_combined_force": selected["harmonic_combined_force"],
            "normalized_worst_gate_score": selected["normalized_worst_gate_score"],
        },
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "fixed_thresholds": {
            "thermal_force_RMSE_meV_A": 50.0,
            "thermal_force_max_abs_meV_A": 250.0,
            "harmonic_RMSE_relative_to_frozen_v11": 2.0,
        },
        "holdout_root": str(args.holdout_root),
        "holdout_expected_indices": expected_indices,
        "holdout_raw_ready_at_freeze": False,
        "selection_scope": (
            "model, replay weight, epoch, and thresholds fixed before either remote "
            "holdout shard was relayed to this machine"
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
