#!/usr/bin/env python3
"""Check the frozen k=120 to k=144 DFPT convergence gate at 450 K degauss."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


EXPECTED_QPOINTS = (
    ("G", 0.000),
    ("G", 0.015),
    ("G", 0.030),
    ("K", 0.980),
    ("K", 0.990),
    ("K", 1.000),
    ("K", 1.010),
    ("K", 1.020),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--degauss", type=float, default=0.00285013035)
    parser.add_argument("--lower-k", type=int, default=120)
    parser.add_argument("--upper-k", type=int, default=144)
    parser.add_argument("--threshold-cm-1", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {}
    sources = []
    for path in args.csv:
        sources.append({"path": str(path), "sha256": sha256(path)})
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                degauss = float(row["degauss_Ry"])
                if not np.isclose(degauss, args.degauss, rtol=0.0, atol=1e-12):
                    continue
                kgrid = int(row["kgrid"])
                if kgrid not in (args.lower_k, args.upper_k):
                    continue
                region = str(row["region"])
                t_value = float(row["t_GK"])
                key = (kgrid, region, round(t_value, 12))
                if key in records:
                    raise ValueError(f"duplicate DFPT row {key}")
                frequencies = np.asarray(
                    [float(row[f"f{index}_cm-1"]) for index in range(1, 7)], float
                )
                if not np.isfinite(frequencies).all():
                    raise ValueError(f"non-finite frequencies for {key}")
                records[key] = frequencies

    comparisons = []
    for region, t_value in EXPECTED_QPOINTS:
        lower_key = (args.lower_k, region, round(t_value, 12))
        upper_key = (args.upper_k, region, round(t_value, 12))
        if lower_key not in records or upper_key not in records:
            raise ValueError(f"missing convergence pair for {region} t={t_value:.3f}")
        lower_top = float(records[lower_key][-1])
        upper_top = float(records[upper_key][-1])
        comparisons.append(
            {
                "region": region,
                "t_GK": t_value,
                f"k{args.lower_k}_top_cm-1": lower_top,
                f"k{args.upper_k}_top_cm-1": upper_top,
                "top_abs_difference_cm-1": abs(upper_top - lower_top),
            }
        )
    maximum = max(item["top_abs_difference_cm-1"] for item in comparisons)
    passed = maximum < args.threshold_cm_1
    payload = {
        "status": "passed" if passed else "failed",
        "scope": "450 K electronic-smearing static-DFPT k-grid convergence",
        "degauss_Ry": args.degauss,
        "lower_kgrid": args.lower_k,
        "upper_kgrid": args.upper_k,
        "threshold_top_max_abs_difference_cm-1": args.threshold_cm_1,
        "maximum_top_abs_difference_cm-1": maximum,
        "passes_convergence_gate": passed,
        "comparisons": comparisons,
        "sources": sources,
        "next_stage": "run FD450_LINE at k=144" if passed else "keep FD450_LINE locked",
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
