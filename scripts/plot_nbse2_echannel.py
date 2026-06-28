"""E1 figure: NbSe2 CDW soft-mode frequency vs electronic temperature (the (E)-channel).
The soft mode hardens through zero as Fermi-Dirac smearing (electronic T) rises -> the
CDW is electronically melted; omega^2=0 crossing = electronic T_CDW. A ground-state-PES
MLIP structurally cannot produce this T_el dependence.

    python scripts/plot_nbse2_echannel.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "results/vq3f/nbse2_echannel.csv")))
T = np.array([float(r["T_el_K"]) for r in rows])
w2 = np.array([float(r["omega2_eVA2amu"]) for r in rows])
fr = np.array([float(r["freq_cm"]) for r in rows])

# omega^2 zero-crossing (linear interp between the soft and first stable point)
i = int(np.where(fr < 0)[0][-1])
Tc = T[i] + (T[i+1]-T[i]) * (0 - w2[i])/(w2[i+1]-w2[i])

fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(0, color="0.5", lw=0.8, ls="--")
ax.axvline(Tc, color="tab:red", lw=1.0, ls=":")
ax.plot(T, fr, "o-", color="tab:blue", lw=1.8, ms=8)
# shade soft region
ax.fill_between(T, fr, 0, where=(fr < 0), color="tab:blue", alpha=0.15)
ax.annotate(f"electronic $T_{{CDW}}$\n$\\approx${Tc:.0f} K",
            xy=(Tc, 0), xytext=(Tc+600, -35),
            arrowprops=dict(arrowstyle="->", color="tab:red"), color="tab:red", fontsize=10)
for t, f in zip(T, fr):
    ax.annotate(f"{f:+.0f}", (t, f), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("electronic temperature  $T_{el}$ = degauss·157887 (K)")
ax.set_ylabel("CDW soft-mode frequency (cm$^{-1}$)   [<0 = soft]")
ax.set_title("NbSe$_2$ (E)-channel: CDW soft mode melts with electronic temperature\n"
             "(frozen-phonon DFT, Fermi-Dirac smearing) — a channel the MLIP cannot produce")
fig.tight_layout()
out = ROOT / "results/figures/nbse2_echannel.png"
fig.savefig(out, dpi=150)
print(f"electronic T_CDW = {Tc:.0f} K; saved {out}")
