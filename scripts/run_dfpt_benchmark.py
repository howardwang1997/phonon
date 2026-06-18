"""Line A full-band benchmark: foundation MLIP vs genuine DFPT (MDR/PhononDB).

For each MDR material, runs the MLIP on the DFT reference cell with the same
supercell matrix and compares the full phonon band structure + thermal
properties against the VASP+phonopy DFPT reference (no API key needed).

Usage:
    conda run -n phonon python scripts/run_dfpt_benchmark.py \
        --models mattersim --materials mp-149 mp-2657 mp-2534 \
        --device cpu --out results/dfpt_benchmark.csv

    # or sample N random MDR materials of a given formula/spacegroup:
    conda run -n phonon python scripts/run_dfpt_benchmark.py \
        --models mattersim --sample 30 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phonon_accel import reference
from phonon_accel.benchmark import run_dfpt_benchmark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mattersim"])
    ap.add_argument("--materials", nargs="+", default=None, help="explicit mp-ids")
    ap.add_argument("--sample", type=int, default=0, help="sample N from MDR index")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/dfpt_benchmark.csv")
    args = ap.parse_args()

    if args.materials:
        mp_ids = args.materials
    elif args.sample:
        import random

        rows = list(reference._index().values())
        random.Random(args.seed).shuffle(rows)
        mp_ids = [r["mp_id"] for r in rows[: args.sample]]
    else:
        mp_ids = ["mp-149"]  # Si worked example

    print(f"models={args.models}  n_materials={len(mp_ids)}  device={args.device}")
    df = run_dfpt_benchmark(args.models, mp_ids, device=args.device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    ok = df[df["error"].fillna("") == ""] if "error" in df else df
    if len(ok):
        print("\nper-model summary (successful materials):")
        g = ok.groupby("model").agg(
            n=("freq_mae", "size"),
            freq_mae=("freq_mae", "mean"),
            freq_rmse=("freq_rmse", "mean"),
            wmax_err=("omega_max_error", "mean"),
            n_imag_pred=("n_imaginary_pred", "sum"),
        )
        print(g.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
