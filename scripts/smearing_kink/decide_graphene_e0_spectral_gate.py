#!/usr/bin/env python3
"""Materialize the five-metric E0 spectral decision as an atomic marker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-temperature-K", type=float, default=450.0)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    metrics = summary["metrics"]["transferred_static_top_cm-1"]
    checks = {
        "gamma_line_MAE_lt_10_cm-1": metrics["G"]["line_MAE_cm-1"] < 10.0,
        "gamma_point_abs_lt_15_cm-1": (
            metrics["G"]["high_symmetry_abs_error_cm-1"] < 15.0
        ),
        "K_line_MAE_lt_10_cm-1": metrics["K"]["line_MAE_cm-1"] < 10.0,
        "K_point_abs_lt_15_cm-1": (
            metrics["K"]["high_symmetry_abs_error_cm-1"] < 15.0
        ),
        "K_kink_relative_error_lt_0p20": metrics["K"]["kink_relative_error"] < 0.20,
    }
    passed = all(checks.values())
    decision = {
        "status": "passed" if passed else "failed",
        "scope": (
            "E0-P2 k18/q9 "
            f"{float(args.target_temperature_K):g} K nine-point static spectral gate"
        ),
        "target_temperature_K": float(args.target_temperature_K),
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "line_MAE_cm-1": 10.0,
            "high_symmetry_abs_error_cm-1": 15.0,
            "K_kink_relative_error": 0.20,
        },
        "checks": checks,
        "metrics": {
            "gamma_line_MAE_cm-1": metrics["G"]["line_MAE_cm-1"],
            "gamma_point_abs_cm-1": metrics["G"]["high_symmetry_abs_error_cm-1"],
            "K_line_MAE_cm-1": metrics["K"]["line_MAE_cm-1"],
            "K_point_abs_cm-1": metrics["K"]["high_symmetry_abs_error_cm-1"],
            "K_kink_relative_error": metrics["K"]["kink_relative_error"],
        },
        "summary": {"path": str(args.summary), "sha256": sha256(args.summary)},
        "next_stage": (
            "release_complete_q6_static_grid_for_6x6_operator"
            if passed
            else "stop_epw_expansion_and_diagnose_electronic_response"
        ),
        "does_not_release_E1": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = args.output_dir / "spectral_gate_decision.json"
    atomic_text(decision_path, json.dumps(decision, indent=2) + "\n")
    marker = args.output_dir / (
        "SPECTRAL_GATE_PASS" if passed else "SPECTRAL_GATE_FAIL"
    )
    atomic_text(marker, decision["decided_at_utc"] + "\n")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
