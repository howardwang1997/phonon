#!/usr/bin/env python3
"""Freeze the synchronized post-P4 development inputs without modifying them."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from graphene_fd_p4_common import atomic_json, sha256


REQUIRED = (
    "on_policy/transferability_acceptance.json",
    "on_policy/freeze_manifest.json",
    "on_policy/temperature_law.json",
    "on_policy/dft_tdep_60.json",
    "on_policy/dft_tdep_60.npz",
    "on_policy/dft_tdep_bootstrap.json",
    "on_policy/dft_tdep_bootstrap.npz",
    "on_policy/evaluation/force_acceptance.json",
    "on_policy/evaluation/prediction_acceptance.json",
    "on_policy/evaluation/frozen_predictions_vs_targets.csv",
    "on_policy/frozen_prediction/frozen_force_predictions.npz",
    "on_policy/frozen_prediction/frozen_static_fc2.npz",
    "on_policy/frozen_prediction/manifest.json",
    "labels/merged/all60.xyz",
    "labels/merged/manifest.json",
    "dfpt/FD450_LINE/graphene_FD450_LINE_dfpt.csv",
    "dfpt/FD450_CONV/convergence_acceptance.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_root.resolve()
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required synchronized inputs: {missing}")

    acceptance_path = source / "on_policy/transferability_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance.get("status") != "failed_C1":
        raise ValueError("expected the archived P4 result to have status=failed_C1")
    if acceptance.get("passes_C1") is not False:
        raise ValueError("expected passes_C1=false")
    if acceptance.get("C2_status") != "not_run":
        raise ValueError("expected C2_status=not_run")

    forbidden_tokens = ("T375", "T525", "375K", "525K", "validation_locked")
    forbidden_paths = [
        str(path.relative_to(source))
        for path in source.rglob("*")
        if any(token.lower() in path.name.lower() for token in forbidden_tokens)
    ]
    if forbidden_paths:
        raise ValueError(f"locked validation artifacts found in source: {forbidden_paths}")

    files = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        stat = path.stat()
        files.append(
            {
                "path": str(path.relative_to(source)),
                "size_bytes": int(stat.st_size),
                "sha256": sha256(path),
            }
        )

    payload = {
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": (
            "immutable local inventory of the opened 450 K failed holdout for "
            "post-P4 development diagnostics"
        ),
        "source": {
            "host": "RTX 2060 via Tailscale 100.105.21.7",
            "remote_project_root": "/home/howardwang/phonon",
            "local_source_root": str(source),
        },
        "registered_outcome": {
            "status": acceptance["status"],
            "passes_C1": acceptance["passes_C1"],
            "passes_C2": acceptance.get("passes_C2"),
            "C2_status": acceptance["C2_status"],
            "component_gates": acceptance.get("component_gates", {}),
            "transferability_acceptance_sha256": sha256(acceptance_path),
        },
        "data_domains": {
            "development": {
                "temperatures_K": [300, 450, 600],
                "degauss_Ry": [0.0019000869, 0.00285013035, 0.0038001738],
                "note": "450 K was opened after the failed one-shot P4 evaluation",
            },
            "validation_locked": {
                "temperatures_K": [375, 525],
                "degauss_Ry": [0.002375108625, 0.003325152075],
                "targets_present_in_archive": False,
            },
        },
        "required_inputs": list(REQUIRED),
        "n_files": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
