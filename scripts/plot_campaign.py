"""Figures from the 1-day campaign eval CSVs (results/ablation/eval_*.csv).
Defensive: plots whatever subset of jobs has finished, skips the rest.

  (1) depth_vs_breadth.png  -- held-out MAE vs #materials (G1, mean+-std over
      seeds) overlaid with held-out MAE vs configs/material (data-efficiency).
  (2) antiforgetting_errorbars.png -- method comparison with seed error bars (G2).
  (3) replay_x_breadth.png  -- G3.   (4) small_vs_medium.png -- G5.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "ablation"
OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NMAT = {"B4": 4, "B8": 8, "B16": 16, "B20": 20, "B24": 24, "B28": 28, "B32": 32, "B48": 48, "B64": 64}


def breadth_jobs(key):
    """all seed runs present for a breadth set, e.g. br_B16_s1..s5"""
    return sorted(p.stem.replace("eval_", "") for p in R.glob(f"eval_br_{key}_s*.csv"))


def metrics(job):
    """(_in-domain MAE, holdout MAE, holdout imaginary count) for eval_<job>.csv."""
    p = R / f"eval_{job}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["error"].fillna("") == ""]
    f = df[df.variant == "finetuned"]
    tr, ho = f[f.split == "train"], f[f.split == "holdout"]
    if not len(ho):
        return None
    return (tr.freq_mae.mean(), ho.freq_mae.mean(), int(ho.n_imaginary_pred.sum()))


def agg(jobs):
    """mean+-std of holdout MAE over a list of jobs (seeds)."""
    vals = [metrics(j)[1] for j in jobs if metrics(j)]
    return (np.mean(vals), np.std(vals), len(vals)) if vals else None


# ---- (1) depth vs breadth ----
fig, ax = plt.subplots(figsize=(7, 4.5))
xs, ys, es = [], [], []
for key, n in NMAT.items():
    a = agg(breadth_jobs(key))
    if a:
        xs.append(n); ys.append(a[0]); es.append(a[1])
if xs:
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="tab:purple", lw=2, ms=8, capsize=4,
                label="breadth: vary #materials (30 cfg each)")
# overlay depth (data-efficiency) curve, same held-out set, B16 fixed
dxs, dys = [], []
for nc in (5, 15, 30, 60):
    m = metrics(f"ncfg{nc}")
    if m:
        dxs.append(nc); dys.append(m[1])
if dxs:
    ax.plot(dxs, dys, "s--", color="tab:green", lw=2, ms=7,
            label="depth: vary cfg/material (16 materials)")
ax.set_xscale("log", base=2)
ax.set_xlabel("# training materials  /  # configs per material (log2)")
ax.set_ylabel("held-out transfer MAE (THz)")
ax.set_title("Breadth fixes transfer where depth cannot")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "depth_vs_breadth.png", dpi=150)
plt.close(fig); print("wrote depth_vs_breadth.png  breadth=", list(zip(xs, [round(y,3) for y in ys])))

# ---- (2) anti-forgetting with seed error bars ----
methods = [("single", "eval_pt0", "single-head"), ("pt1000", "eval_pt1000", "replay pt1k"),
           ("pt5000", "eval_pt5000", "replay pt5k"), ("lora8", "eval_lora8", "LoRA r8"),
           ("lora32", "eval_lora32", "LoRA r32")]
labels, means, stds = [], [], []
for mode, seed1csv, lab in methods:
    jobs = []
    s1 = seed1csv.replace("eval_", "")
    if (R / f"{seed1csv}.csv").exists():
        jobs.append(s1)
    jobs += [f"eb_{mode}_s{s}" for s in (2, 3, 4)]
    vals = [metrics(j)[1] for j in jobs if metrics(j)]
    if vals:
        labels.append(lab); means.append(np.mean(vals)); stds.append(np.std(vals))
if labels:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=5, color="tab:blue", alpha=0.8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("held-out MAE (THz)")
    ax.set_title("Anti-forgetting methods (held-out transfer, seed error bars)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "antiforgetting_errorbars.png", dpi=150)
    plt.close(fig); print("wrote antiforgetting_errorbars.png", list(zip(labels, [round(m,3) for m in means])))

# ---- (3) replay x breadth (G3, at B32) ----
rb = []
for mode in ["single", "pt500", "pt1000", "pt2500"]:
    a = agg([f"rb_{mode}_s{s}" for s in (1, 2)])
    if a:
        rb.append((mode, a[0], a[1]))
if rb:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(range(len(rb)), [r[1] for r in rb], yerr=[r[2] for r in rb],
                fmt="o-", capsize=4, color="tab:red")
    ax.set_xticks(range(len(rb))); ax.set_xticklabels([r[0] for r in rb])
    ax.set_ylabel("held-out MAE (THz)"); ax.set_title("Replay strength x breadth (B32)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "replay_x_breadth.png", dpi=150); plt.close(fig)
    print("wrote replay_x_breadth.png", [(r[0], round(r[1], 3)) for r in rb])

# ---- (4) small vs medium foundation (G5) ----
sm = []
for key, n in NMAT.items():
    s = metrics(f"br_{key}_s1")   # small, seed1
    m = metrics(f"med_{key}")     # medium
    if s and m:
        sm.append((n, s[1], m[1]))
if sm:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[0] for r in sm], [r[1] for r in sm], "o-", label="small base")
    ax.plot([r[0] for r in sm], [r[2] for r in sm], "s-", label="medium base")
    ax.set_xscale("log", base=2); ax.set_xlabel("# training materials")
    ax.set_ylabel("held-out MAE (THz)"); ax.set_title("Foundation size vs data need")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "small_vs_medium.png", dpi=150); plt.close(fig)
    print("wrote small_vs_medium.png", sm)

print("done.")
