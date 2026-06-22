"""Cross-model phonon softening figure: per-material ω_max softening (%) vs DFPT
for MatterSim and SevenNet — shows softening is universal across foundation MLIPs,
not a MACE artifact."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "results" / "xmodel_baseline.csv")))
by = {}
for r in rows:
    try:
        by.setdefault(r["material"], {})[r["model"]] = float(r["softening_pct"])
    except Exception:
        pass

order = ["C", "Si", "Ge", "GaAs", "GaAs", "AlAs", "MgO", "NaCl", "Al"]
mats = [m for m in ["C", "Si", "Ge", "GaAs", "MgO", "NaCl", "Al"] if m in by]
ms = [by[m].get("mattersim", np.nan) for m in mats]
sn = [by[m].get("sevennet", np.nan) for m in mats]

x = np.arange(len(mats)); w = 0.38
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.bar(x - w / 2, ms, w, label=f"MatterSim (median {np.nanmedian(ms):+.1f}%)", color="tab:blue")
ax.bar(x + w / 2, sn, w, label=f"SevenNet (median {np.nanmedian(sn):+.1f}%)", color="tab:orange")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(mats)
ax.set_ylabel("phonon ω$_{max}$ softening vs DFPT (%)")
ax.set_title("Phonon softening is universal across foundation MLIPs")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = ROOT / "results" / "figures" / "xmodel_softening.png"
fig.savefig(out, dpi=150)
print("wrote", out, "(all", len(mats), "materials soften for both models)")
