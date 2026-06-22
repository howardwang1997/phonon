"""Figure 2 — Foundation MLIPs soften phonons; FC distillation repairs them.

  (a) MACE per-material omega_max softening on canonical crystals (the model we
      fine-tune).               <- results/benchmark_builtin_mace.csv
  (b) Softening is universal across foundation MLIPs (MatterSim, SevenNet).
                                <- results/xmodel_baseline.csv
  (c) Si phonon dispersion: DFPT vs MACE-baseline (softened) vs FC-distilled.
                                <- results/figures/data/si_dispersion.npz
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
from matplotlib.lines import Line2D  # noqa: E402

set_style()
FIG = ROOT / "results" / "figures"

fig = plt.figure(figsize=(13, 4.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.5], wspace=0.32)
axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

# ---- (a) MACE softening ----
m = pd.read_csv(ROOT / "results" / "benchmark_builtin_mace.csv")
m = m[m["error"].fillna("") == ""].sort_values("softening_pct")
axa.bar(range(len(m)), m["softening_pct"], color=C["mace"], alpha=0.9)
med = m["softening_pct"].median()
axa.axhline(med, color="k", ls="--", lw=1, label=f"median {med:+.0f}%")
axa.axhline(0, color="0.6", lw=0.8)
axa.set_xticks(range(len(m)))
axa.set_xticklabels(m["material"])
axa.set_ylabel(r"MACE $\omega_{\max}$ softening vs DFPT (%)")
axa.set_title("MACE baseline softens every crystal")
axa.legend(loc="lower right")
panel(axa, "a")

# ---- (b) cross-model universality ----
x = pd.read_csv(ROOT / "results" / "xmodel_baseline.csv")
piv = {mdl: x[x.model == mdl].set_index("material")["softening_pct"]
       for mdl in ("mattersim", "sevennet")}
mats = ["C", "Si", "Ge", "GaAs", "MgO", "NaCl", "Al"]
xs = np.arange(len(mats)); w = 0.38
ms = [piv["mattersim"].get(k, np.nan) for k in mats]
sn = [piv["sevennet"].get(k, np.nan) for k in mats]
axb.bar(xs - w / 2, ms, w, color=C["mattersim"], label=f"MatterSim (med {np.nanmedian(ms):+.0f}%)")
axb.bar(xs + w / 2, sn, w, color=C["sevennet"], label=f"SevenNet (med {np.nanmedian(sn):+.0f}%)")
axb.axhline(0, color="k", lw=0.8)
axb.set_xticks(xs); axb.set_xticklabels(mats)
axb.set_ylabel(r"$\omega_{\max}$ softening vs DFPT (%)")
axb.set_title("Softening is model-universal")
axb.legend(loc="lower left")
panel(axb, "b")

# ---- (c) Si dispersion before/after ----
d = np.load(FIG / "data" / "si_dispersion.npz", allow_pickle=True)


def _bands(ax, prefix, color, ls="-", lw=1.5, z=2):
    for dist, freq in zip(d[f"{prefix}_dist"], d[f"{prefix}_freq"]):
        ax.plot(np.asarray(dist), np.asarray(freq), color=color, ls=ls, lw=lw, zorder=z)


_bands(axc, "dfpt", C["dfpt"], ls="--", lw=1.3, z=3)
_bands(axc, "base", C["mace"], lw=1.4, z=2)
_bands(axc, "ft", C["finetuned"], lw=1.6, z=2)
for b in d["boundaries"]:
    axc.axvline(float(b), color="0.85", lw=0.7, zorder=0)
axc.axhline(0, color="0.6", lw=0.7)
axc.set_xlim(float(d["boundaries"][0]), float(d["boundaries"][-1]))
axc.set_ylabel("frequency (THz)")
axc.set_xlabel("wave vector")
axc.set_xticks([])
axc.set_title("Si: softened baseline, repaired by distillation")
handles = [Line2D([], [], color=C["dfpt"], ls="--", lw=1.3, label="DFPT (reference)"),
           Line2D([], [], color=C["mace"], lw=1.4, label="MACE-MP-0 (baseline)"),
           Line2D([], [], color=C["finetuned"], lw=1.6, label="FC-distilled")]
axc.legend(handles=handles, loc="lower center")
panel(axc, "c")

out = FIG / "fig2_failure_repair.png"
fig.savefig(out)
print("wrote", out)
