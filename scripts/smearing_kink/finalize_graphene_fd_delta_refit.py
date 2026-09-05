#!/usr/bin/env python3
"""Record a final-refit model frozen before the independent force holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--gate-selection", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay-weight", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = json.loads(args.checkpoint_selection.read_text())
    data = json.loads(args.data_manifest.read_text())
    gate = json.loads(args.gate_selection.read_text())
    result = {
        "status": "frozen_before_independent_holdout",
        "selection_scope": (
            "replay weight and epoch selected on development data; final refit then "
            "uses those development thermal structures as training data; the new "
            "15-structure force holdout has not been read"
        ),
        "replay_weight": args.replay_weight,
        "frozen_checkpoint_epoch": args.epoch,
        "source_checkpoint_selection": str(args.checkpoint_selection),
        "source_checkpoint_selection_sha256": sha256(args.checkpoint_selection),
        "source_development_candidate": checkpoint["selected"],
        "data_manifest": data,
        "development_refit_metrics_are_in_sample_for_thermal": True,
        "development_refit_gate_selection": gate,
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "next_valid_thermal_result": "one-shot independent 15-structure holdout",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
