"""#3 figure: graphene Gamma-E2g phonon vs electronic temperature. DFPT (ph.x,
linear response) captures the Kohn-anomaly softening at low electronic T that the
frozen-phonon (MLIP-style) approach structurally misses -- the (E) channel."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# #3 DFPT (ph.x linear response), graphene Gamma-E2g
T = np.array([789, 1579, 3158, 6315])
dfpt = np.array([1472.9, 1531.3, 1560.4, 1574.5])
# V-Q2 frozen-phonon (DFT, fixed 5x5, Fermi-Dirac) -- frequency ~insensitive
frozen = np.array([1559, 1568, 1567, 1553])

fig, ax = plt.subplots(figsize=(6.2, 4.3))
ax.plot(T, dfpt, "o-", color="tab:red", lw=2, ms=8, label="DFPT (ph.x, linear response) — (E) channel")
ax.plot(T, frozen, "s--", color="tab:gray", lw=1.5, ms=7, label="frozen-phonon (MLIP-style) — misses it")
ax.annotate("Kohn anomaly: E$_{2g}$ softens ~100 cm$^{-1}$\nat low electronic T (sharp Fermi surface)",
            xy=(789, 1473), xytext=(1700, 1485), fontsize=8, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
ax.set_xlabel("electronic temperature T$_{el}$ (K)  [degauss · 157887]")
ax.set_ylabel("graphene Γ-E$_{2g}$ frequency (cm$^{-1}$)")
ax.set_title("DFPT captures the (E)-channel Kohn softening; frozen-phonon misses it")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
out = ROOT / "results" / "figures" / "dfpt_smearing.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out)
