"""Build the A3 fine-tuning dataset by distilling MDR DFPT force constants.

Train on one set of materials, hold out another for generalization testing.
Writes train.xyz / val.xyz (extxyz, MACE/MatterSim-consumable).

Usage:
    conda run -n phonon python scripts/make_finetune_data.py \
        --out data/finetune --n-configs 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phonon_accel.finetune.dataset import build_dataset

# 14 benchmarked crystals split into train / held-out (generalization) sets.
TRAIN = ["mp-149", "mp-1265", "mp-22862", "mp-661", "mp-7140", "mp-9946", "mp-2172", "mp-1986"]
HOLDOUT = ["mp-804", "mp-1138", "mp-2605", "mp-20351", "mp-984", "mp-390"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/finetune")
    ap.add_argument("--n-configs", type=int, default=50)
    ap.add_argument("--n-single-sites", type=int, default=6)
    ap.add_argument("--rattle-std", type=float, default=0.02)
    ap.add_argument("--train", nargs="+", default=TRAIN)
    args = ap.parse_args()

    print(f"building fine-tune data from {len(args.train)} materials: {args.train}")
    summary = build_dataset(
        args.train, out_dir=args.out, n_configs=args.n_configs,
        rattle_std=args.rattle_std, n_single_sites=args.n_single_sites,
    )
    summary["train_materials"] = args.train
    summary["holdout_materials"] = HOLDOUT
    print(json.dumps(summary, indent=2))
    Path(args.out, "split.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
