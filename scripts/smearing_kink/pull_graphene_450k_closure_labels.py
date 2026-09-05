"""Pull DFT labels from the V100 boxes into the canonical local label tree.

Mapping rules:
  * ``shard_A`` / ``shard_B`` labels land as ``labels/<stem>/snapshot_{i:03d}_k8.npz``
    with the identity local index (the shard npz order IS the frozen order).
  * ``r2b_half1`` / ``r2b_half2`` labels are remapped through the half npz
    ``sscha_indices`` (frozen label ids) into
    ``labels/r2b_tagged_pairs/snapshot_{id - 1000:03d}_k8.npz`` so they index
    the FULL frozen ``r2b_tagged_pairs_snapshots.npz``.

Usage:
    pull_graphene_450k_closure_labels.py [--host v100ts] [--host v100bts] [--dry-run]

Remote layout: /data/graphene_450k_closure/T450/<stem>/labels/snapshot_*_k8.npz
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R1DIR = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R1_450k_matched_overlap_screen"
)
REMOTE_BASE = "/data/graphene_450k_closure/T450"

# host -> stems that run there
HOST_STEMS = {
    "v100ts": ["shard_A", "r2b_half1"],
    "v100bts": ["shard_B", "r2b_half2"],
}


def rsync(host: str, stem: str, destination: Path, dry_run: bool) -> list[Path]:
    remote = f"{host}:{REMOTE_BASE}/{stem}/labels/"
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-av", "--include=*/", "--include=snapshot_*_k8.npz",
               "--exclude=*", remote, str(destination) + "/"]
    if dry_run:
        command.insert(1, "--dry-run")
    subprocess.run(command, check=True)
    return sorted(destination.glob("snapshot_*_k8.npz"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", action="append", choices=sorted(HOST_STEMS),
                        help="hosts to pull from (default: both)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    hosts = args.host or list(HOST_STEMS)

    pulled = {}
    for host in hosts:
        for stem in HOST_STEMS[host]:
            if stem.startswith("r2b_half"):
                half_npz = R1DIR / f"{stem}_snapshots.npz"
                if not half_npz.exists():
                    raise FileNotFoundError(half_npz)
                with np.load(half_npz, allow_pickle=False) as data:
                    label_ids = np.asarray(data["sscha_indices"], int)
                staging = R1DIR / "labels" / stem
                files = rsync(host, stem, staging, args.dry_run)
                canonical = R1DIR / "labels" / "r2b_tagged_pairs"
                canonical.mkdir(parents=True, exist_ok=True)
                count = 0
                for path in files:
                    local = int(path.name.split("_")[1])
                    target = canonical / f"snapshot_{label_ids[local] - 1000:03d}_k8.npz"
                    if not args.dry_run:
                        target.write_bytes(path.read_bytes())
                    count += 1
                pulled[f"{host}:{stem}"] = {
                    "staged": str(staging),
                    "canonical_dir": str(canonical),
                    "labels": count,
                    "expected": len(label_ids),
                }
            else:
                destination = R1DIR / "labels" / stem
                files = rsync(host, stem, destination, args.dry_run)
                pulled[f"{host}:{stem}"] = {
                    "canonical_dir": str(destination),
                    "labels": len(files),
                }

    print(json.dumps(pulled, indent=2))
    (R1DIR / "labels" / "pull_report.json").write_text(
        json.dumps({"pulled": pulled, "dry_run": args.dry_run}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
