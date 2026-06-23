"""Supplementary Figure S1 — anti-forgetting strategies (held-out transfer, seed
error bars), restyled to match the main-figure publication style.

Regenerates results/figures/antiforgetting_errorbars.png from the ablation eval CSVs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_style import C, set_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

set_style()
R = ROOT / "results" / "ablation"
OUT = ROOT / "results" / "figures" / "antiforgetting_errorbars.png"


def metrics(job):
    p = R / f"eval_{job}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p); d = d[d["error"].fillna("") == ""]
    f = d[(d.variant == "finetuned") & (d.split == "holdout")]
    return f.freq_mae.mean() if len(f) else None


methods = [("single", "eval_pt0", "single-head"), ("pt1000", "eval_pt1000", "replay pt1k"),
           ("pt5000", "eval_pt5000", "replay pt5k"), ("lora8", "eval_lora8", "LoRA r8"),
           ("lora32", "eval_lora32", "LoRA r32")]
labels, means, stds = [], [], []
for mode, seed1csv, lab in methods:
    jobs = []
    if (R / f"{seed1csv}.csv").exists():
        jobs.append(seed1csv.replace("eval_", ""))
    jobs += [f"eb_{mode}_s{s}" for s in (2, 3, 4)]
    vals = [metrics(j) for j in jobs if metrics(j)]
    if vals:
        labels.append(lab); means.append(np.mean(vals)); stds.append(np.std(vals))

# colour LoRA (best family) green, replay neutral, single-head red to echo the story
colour = {"single-head": C["mace"], "replay pt1k": "#6a8caf", "replay pt5k": "#6a8caf",
          "LoRA r8": C["finetuned"], "LoRA r32": C["finetuned"]}
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=5,
              color=[colour.get(l, "tab:blue") for l in labels], alpha=0.9,
              error_kw=dict(lw=1.2))
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_ylabel("held-out transfer MAE (THz)")
ax.set_title("Anti-forgetting: replay and LoRA prevent catastrophic forgetting")
ax.set_ylim(0, max(np.array(means) + np.array(stds)) + 0.35)
ax.grid(True, axis="y")
fig.savefig(OUT)
print("wrote", OUT, list(zip(labels, [round(m, 2) for m in means])))
