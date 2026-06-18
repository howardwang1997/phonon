"""Paper figures from the A3 ablation CSVs:
  (1) anti-forgetting three-method comparison (train MAE + unseen-elem imaginary)
  (2) data-efficiency curve (in-domain MAE vs configs/material)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "ablation"
OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _metrics(csv):
    df = pd.read_csv(csv)
    df = df[df["error"].fillna("") == ""]
    f = df[df.variant == "finetuned"].set_index("mp_id")
    tr = f[f.split == "train"]
    return (tr.freq_mae.mean(),
            int(f.loc["mp-7140"].n_imaginary_pred),   # SiC (C unseen)
            int(f.loc["mp-984"].n_imaginary_pred))     # BN  (B unseen)


# ---- Figure 1: anti-forgetting comparison ----
methods = [
    ("single-head\n(full FT)", "eval_pt0.csv"),
    ("replay\npt=1000", "eval_pt1000.csv"),
    ("replay\npt=5000", "eval_pt5000.csv"),
    ("LoRA\nrank=8", "eval_lora8.csv"),
    ("LoRA\nrank=32", "eval_lora32.csv"),
]
labels, maes, imags = [], [], []
for name, csv in methods:
    p = R / csv
    if not p.exists():
        continue
    mae, sic, bn = _metrics(p)
    labels.append(name); maes.append(mae); imags.append(sic + bn)

fig, ax1 = plt.subplots(figsize=(8, 4.5))
x = range(len(labels))
ax1.bar([i - 0.2 for i in x], maes, width=0.4, color="tab:blue", label="train MAE (THz)")
ax1.set_ylabel("in-domain MAE (THz)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=9)
ax2 = ax1.twinx()
ax2.bar([i + 0.2 for i in x], [v + 0.5 for v in imags], width=0.4, color="tab:red",
        label="unseen-elem imag (SiC+BN)")
ax2.set_yscale("log")
ax2.set_ylabel("unseen-element imaginary modes (log)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax1.set_title("Anti-forgetting: in-domain accuracy vs unseen-element robustness")
fig.tight_layout()
fig.savefig(OUT / "antiforgetting_comparison.png", dpi=150)
plt.close(fig)
print("wrote", OUT / "antiforgetting_comparison.png")

# ---- Figure 2: data-efficiency curve ----
ns, de_mae = [], []
for N in [5, 15, 30, 60]:
    p = R / f"eval_ncfg{N}.csv"
    if not p.exists():
        continue
    mae, _, _ = _metrics(p)
    ns.append(N); de_mae.append(mae)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ns, de_mae, "o-", color="tab:green", lw=2, ms=8)
for n, m in zip(ns, de_mae):
    ax.annotate(f"{m:.2f}", (n, m), textcoords="offset points", xytext=(0, 8), fontsize=9)
ax.set_xlabel("distilled configs per material")
ax.set_ylabel("in-domain MAE (THz)")
ax.set_title("Data efficiency: in-domain phonon MAE vs data/material")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "data_efficiency.png", dpi=150)
plt.close(fig)
print("wrote", OUT / "data_efficiency.png")
