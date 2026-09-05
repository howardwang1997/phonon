#!/usr/bin/env python3
"""Combine independently gated 300 and 600 K model selections."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-300", type=Path, required=True)
    parser.add_argument("--selection-600", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source300 = json.loads(args.selection_300.read_text())
    source600 = json.loads(args.selection_600.read_text())
    result = {
        "thresholds": source300["thresholds"],
        "v11_harmonic_RMSE_meV_A": source300["v11_harmonic_RMSE_meV_A"],
        "by_temperature": {
            "300": source300["by_temperature"]["300"],
            "600": source600["by_temperature"]["600"],
        },
        "passes_both_temperature_gates": bool(
            source300["passes_all_requested_temperature_gates"]
            and source600["passes_all_requested_temperature_gates"]
        ),
        "sources": {
            "300": str(args.selection_300),
            "600": str(args.selection_600),
        },
    }
    if source600["thresholds"] != source300["thresholds"]:
        raise ValueError("300 and 600 K selections used different thresholds")
    if abs(
        source600["v11_harmonic_RMSE_meV_A"]
        - source300["v11_harmonic_RMSE_meV_A"]
    ) > 1e-9:
        raise ValueError("300 and 600 K selections used different v11 retention baselines")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
