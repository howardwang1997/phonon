"""Pull the 300/600 K R2B label batches from the V100 boxes.

One stem per box, full 24-config batches, identity local index (the remote
npz order IS the frozen order), so labels land directly as
``R2B_{T}k_tagged_pairs/labels/snapshot_{i:03d}_k8.npz``.

Remote layout: /data/graphene_450k_closure/T{300,600}/r2b_T{T}/labels/

Usage:
    pull_graphene_closure_r2b_labels.py [--temperature 300] [--temperature 600] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
REMOTE_STEM = "/data/graphene_450k_closure/T{temp}/r2b_T{temp}/labels/"

HOST_TEMP = {300: "v100ts", 600: "v100bts"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=int, action="append",
                        choices=sorted(HOST_TEMP), help="temperatures to pull (default: both)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    temps = args.temperature or sorted(HOST_TEMP)

    pulled = {}
    for temp in temps:
        host = HOST_TEMP[temp]
        outdir = BASE / f"R2B_{temp}k_tagged_pairs" / "labels"
        outdir.mkdir(parents=True, exist_ok=True)
        remote = f"{host}:{REMOTE_STEM.format(temp=temp)}"
        command = ["rsync", "-av", "--include=*/", "--include=snapshot_*_k8.npz",
                   "--exclude=*", remote, str(outdir) + "/"]
        if args.dry_run:
            command.insert(1, "--dry-run")
        result = subprocess.run(command, check=False)
        if result.returncode not in (0, 23):  # 23 = remote dir not created yet
            raise RuntimeError(f"rsync {remote} failed with {result.returncode}")
        files = sorted(outdir.glob("snapshot_*_k8.npz"))
        pulled[f"{host}:T{temp}"] = {
            "canonical_dir": str(outdir),
            "labels": len(files),
            "expected": 24,
        }

    print(json.dumps(pulled, indent=2))
    for temp in temps:
        report = BASE / f"R2B_{temp}k_tagged_pairs" / "pull_report.json"
        report.write_text(
            json.dumps({"pulled": pulled, "dry_run": args.dry_run}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
