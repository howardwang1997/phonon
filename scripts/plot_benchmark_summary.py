"""Summary figure for a Line-A benchmark CSV: per-material freq MAE + softening.

Usage:
    conda run -n phonon python scripts/plot_benchmark_summary.py \
        --csv results/dfpt_mattersim_curated.csv --out results/figures/mattersim_summary.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/dfpt_mattersim_curated.csv")
    ap.add_argument("--out", default="results/figures/benchmark_summary.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["error"].fillna("") == ""].copy()
    df["softening_pct"] = 100 * df["omega_max_error"] / df["omega_max_ref"]
    df["label"] = df["formula"] + "\n" + df["mp_id"]
    df = df.sort_values("freq_mae")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    x = range(len(df))

    ax1.bar(x, df["freq_mae"], color="tab:blue", alpha=0.85)
    ax1.axhline(df["freq_mae"].mean(), color="k", ls="--", lw=1,
                label=f"mean {df['freq_mae'].mean():.2f} THz")
    ax1.set_ylabel("phonon freq MAE (THz)")
    ax1.set_title(f"{args.csv} — MLIP vs DFPT ({len(df)} materials)")
    ax1.legend()

    colors = ["tab:red" if s < 0 else "tab:green" for s in df["softening_pct"]]
    ax2.bar(x, df["softening_pct"], color=colors, alpha=0.85)
    ax2.axhline(df["softening_pct"].mean(), color="k", ls="--", lw=1,
                label=f"mean {df['softening_pct'].mean():+.1f}%")
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_ylabel(r"$\omega_{max}$ softening (%)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(df["label"], rotation=0, fontsize=8)
    ax2.legend()

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
