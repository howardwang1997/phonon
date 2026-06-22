"""Figure 3 — Data-efficiency laws for FC distillation.

  (a) Transfer is governed by breadth, not depth (held-out MAE vs #materials,
      with the depth curve overlaid).
  (b) Depth x breadth surface: breadth helps (down), depth hurts (across).
  (c) Acquisition: coverage beats random; naive uncertainty is worst.

All from results/ablation/eval_*.csv. Defensive to missing jobs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_style import C, panel, set_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

set_style()
R = ROOT / "results" / "ablation"
FIG = ROOT / "results" / "figures"
NMAT = {"B4": 4, "B8": 8, "B16": 16, "B20": 20, "B24": 24, "B28": 28, "B32": 32, "B48": 48, "B64": 64}


def _hold(job, stat):
    p = R / f"eval_{job}.csv"
    if not p.exists():
        return np.nan
    d = pd.read_csv(p); d = d[d["error"].fillna("") == ""]
    f = d[(d.variant == "finetuned") & (d.split == "holdout")]
    if not len(f):
        return np.nan
    return f.freq_mae.mean() if stat == "mean" else f.freq_mae.median()


def hold_mae(job):
    return _hold(job, "mean")


def agg(jobs, stat="mean"):
    v = [_hold(j, stat) for j in jobs]
    v = [x for x in v if x == x]
    return (np.mean(v), np.std(v)) if v else None


fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(13.5, 4.1))
fig.subplots_adjust(wspace=0.34)

# ---- (a) breadth vs depth (median emphasised; mean is BN-inflated) ----
xs, ymed, emed, ymean = [], [], [], []
for key, n in NMAT.items():
    md = agg([f"br_{key}_s{s}" for s in range(1, 6)], "median")
    mn = agg([f"br_{key}_s{s}" for s in range(1, 6)], "mean")
    if md and mn:
        xs.append(n); ymed.append(md[0]); emed.append(md[1]); ymean.append(mn[0])
axa.errorbar(xs, ymed, yerr=emed, fmt="o-", color=C["breadth"], capsize=4, ms=7,
             label="breadth (median)")
axa.plot(xs, ymean, "^--", color=C["breadth"], ms=5, lw=1.2, alpha=0.55,
         label="breadth (mean, BN-inflated)")
dxs, dys = [], []
for nc in (5, 15, 30, 60):
    v = hold_mae(f"ncfg{nc}")
    if v == v:
        dxs.append(nc); dys.append(v)
if dxs:
    axa.plot(dxs, dys, "s--", color=C["depth"], ms=6, label="depth: vary configs/material")
axa.set_xscale("log", base=2)
axa.set_xlabel("# training materials  (breadth)")
axa.set_ylabel("held-out transfer MAE (THz)")
axa.set_title("Breadth fixes transfer; depth cannot")
axa.grid(True); axa.legend()
panel(axa, "a")

# ---- (b) depth x breadth surface ----
DEPTHS = [("nc15", 15), ("nc30", 30), ("nc60", 60)]
BR = [("B16", 16), ("B24", 24), ("B32", 32), ("B48", 48), ("B64", 64), ("B79", 79)]
grid = np.full((len(BR), len(DEPTHS)), np.nan)
for i, (k, _) in enumerate(BR):
    for j, (dl, _) in enumerate(DEPTHS):
        grid[i, j] = hold_mae(f"br_{k}_s1") if dl == "nc30" else hold_mae(f"db_{k}_{dl}")
im = axb.imshow(grid, cmap="viridis_r", aspect="auto")
axb.set_xticks(range(len(DEPTHS))); axb.set_xticklabels([d[1] for d in DEPTHS])
axb.set_yticks(range(len(BR))); axb.set_yticklabels([b[1] for b in BR])
axb.set_xlabel("configs / material  (depth)")
axb.set_ylabel("# training materials  (breadth)")
axb.set_title("Breadth helps, depth hurts")
thr = np.nanmean(grid)
for i in range(len(BR)):
    for j in range(len(DEPTHS)):
        if np.isfinite(grid[i, j]):
            axb.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=8.5,
                     color="white" if grid[i, j] > thr else "black")
cb = fig.colorbar(im, ax=axb, fraction=0.046, pad=0.04)
cb.set_label("held-out MAE (THz)", fontsize=9)
panel(axb, "b")

# ---- (c) acquisition ----
NS = [8, 16, 24, 32, 40, 48, 64, 79]
arms = [
    ("random", C["random"], "o-", lambda n: agg([f"rnd{c}_N{n}_s1" for c in "ABCDEF"])),
    ("coverage (greedy)", C["coverage"], "o-", lambda n: agg([f"gdiv_N{n}_s{s}" for s in (1, 2)])),
    ("naive uncertainty", C["uncertainty"], "o--", lambda n: agg([f"al_unc_N{n}"])),
]
for label, color, fmt, fn in arms:
    xs, ys, es = [], [], []
    for n in NS:
        a = fn(n)
        if a:
            xs.append(n); ys.append(a[0]); es.append(a[1])
    if xs:
        axc.errorbar(xs, ys, yerr=es, fmt=fmt, color=color, capsize=4, ms=6, label=label)
axc.set_xscale("log", base=2)
axc.set_xlabel("# materials queried (DFT budget)")
axc.set_ylabel("held-out transfer MAE (THz)")
axc.set_title("Coverage > random > uncertainty")
axc.grid(True); axc.legend()
panel(axc, "c")

out = FIG / "fig3_data_laws.png"
fig.savefig(out)
print("wrote", out)
