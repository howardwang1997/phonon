"""Run the Line A MLIP phonon benchmark over the built-in worked-example set.

Produces a softening / dynamical-stability table (CSV + console) across the
installed foundation MLIPs. Swap in Petretto/MDR materials + per-q DFPT
reference for the full study.

Usage:
    conda run -n phonon python scripts/run_benchmark.py \
        --models mace --device cpu --out results/benchmark_builtin.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phonon_accel.benchmark import REFERENCE_OMEGA_MAX, run_benchmark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mace"])
    ap.add_argument("--materials", nargs="+", default=list(REFERENCE_OMEGA_MAX))
    ap.add_argument("--device", default=None)
    ap.add_argument("--supercell", type=int, default=3)
    ap.add_argument("--out", default="results/benchmark_builtin.csv")
    args = ap.parse_args()

    print(f"models={args.models} materials={args.materials} device={args.device}")
    df = run_benchmark(
        args.models,
        args.materials,
        device=args.device,
        supercell=(args.supercell,) * 3,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    cols = ["model", "material", "omega_max", "omega_max_ref", "softening_pct",
            "n_imaginary", "asr_residual"]
    print("\n" + df[cols].to_string(index=False))

    ok = df["error"].eq("").all()
    if ok:
        soft = df.loc[df["softening_pct"].notna(), "softening_pct"]
        print(f"\nmean softening: {soft.mean():+.1f}%   "
              f"materials with imaginary modes: {(df['n_imaginary'] > 0).sum()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
