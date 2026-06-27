"""C figure: NbSe2 CDW soft-mode thermal stabilization. TDEP (distilled-FT MLIP,
fixed a=3.44) min phonon frequency vs T -- the soft mode (negative at low T)
renormalizes to stable above ~150-200 K = the (L)-channel CDW order-disorder
signature."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "results" / "td_phonon" / "td_nbse2_ft_Tfix.csv")))
T = np.array([float(r["T_K"]) for r in rows])
mf = np.array([float(r["minfreq_thz"]) for r in rows])

fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.axhspan(min(mf.min() * 1.2, -0.7), 0, color="tab:red", alpha=0.06)
ax.axhline(0, color="gray", lw=0.9)
ax.plot(T, mf, "o-", color="tab:purple", lw=1.8, ms=7, label="TDEP min freq (distilled-FT)")
ax.annotate("soft (CDW unstable)", xy=(60, -0.45), fontsize=8, color="tab:red")
ax.annotate("stabilized", xy=(300, 0.04), fontsize=8, color="tab:green")
# mark the crossing region
ax.axvspan(100, 200, color="tab:green", alpha=0.06)
ax.text(150, mf.min() * 0.8, "stabilizes\n~150 K", fontsize=8, ha="center", color="tab:green")
ax.set_xlabel("temperature (K)")
ax.set_ylabel("min phonon frequency (THz)")
ax.set_title("NbSe$_2$ CDW soft mode thermally stabilizes (TDEP, L-channel)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
out = ROOT / "results" / "figures" / "nbse2_Tevolution.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out, "min freq:", list(zip(T.astype(int), mf)))
