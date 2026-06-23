"""Figure 4 — Downstream: FC distillation recovers lattice thermal conductivity.

  (a) kappa vs experiment (log-log), baseline vs FC-distilled, converged sc4.
  (b) Si kappa supercell convergence: FC-distilled -> experiment; baseline stays soft.

Panel (a) from results/kappa/sc4_*.json; (b) from the recorded convergence runs.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_style import C, panel, set_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

set_style()
R = ROOT / "results" / "kappa"
FIG = ROOT / "results" / "figures"
EXP = {"Si": 140, "Ge": 60, "C": 2200, "Sn": 11, "SiC": 430, "BN": 760, "BP": 400,
       "BAs": 1300, "AlP": 90, "AlAs": 91, "GaP": 100, "GaAs": 45, "InP": 68}

fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.6))
fig.subplots_adjust(wspace=0.27)

# ---- (a) kappa scatter ----
r = {}
for f in glob.glob(str(R / "sc4_*.json")):
    try:
        dd = json.load(open(f)); mm = dd["material"]; k = dd["kappa"].get("300")
        if k:
            r.setdefault(mm, {})["ft" if f.endswith("_ft.json") else "base"] = k
    except Exception:
        pass
lim = [5, 4000]
axa.plot(lim, lim, "--", color="0.5", lw=1, label="y = x (experiment)")
# hand-tuned label offsets (points) to avoid collisions in the crowded mid-low cluster
OFF = {"Si": (-17, 5), "C": (-13, 4), "BN": (1, 8), "SiC": (-24, 5), "BP": (3, -12),
       "BAs": (7, -3), "AlAs": (-31, 6), "GaP": (8, 2), "AlP": (3, -12),
       "InP": (7, 5), "Ge": (-8, -13), "GaAs": (-31, 0), "Sn": (-19, 6)}
for mm, v in r.items():
    e = EXP.get(mm)
    if not e:
        continue
    if "base" in v and "ft" in v:
        axa.plot([e, e], [v["base"], v["ft"]], color="0.8", lw=0.8, zorder=1)
    if "base" in v:
        axa.plot(e, v["base"], "o", mfc="none", mec=C["baseline"], ms=7, zorder=2)
    if "ft" in v:
        axa.plot(e, v["ft"], "o", color=C["finetuned"], ms=7, zorder=3)
        axa.annotate(mm, (e, v["ft"]), textcoords="offset points",
                     xytext=OFF.get(mm, (5, 3)), fontsize=7.5, zorder=4)
axa.plot([], [], "o", mfc="none", mec=C["baseline"], label="baseline MLIP")
axa.plot([], [], "o", color=C["finetuned"], label="FC-distilled")
axa.set_xscale("log"); axa.set_yscale("log"); axa.set_xlim(lim); axa.set_ylim(lim)
axa.set_xlabel(r"experimental $\kappa$  (W m$^{-1}$K$^{-1}$)")
axa.set_ylabel(r"MLIP $\kappa$  (W m$^{-1}$K$^{-1}$)")
axa.set_title("κ pulled toward experiment (converged sc4)")
axa.legend(loc="upper left"); axa.grid(True, which="both")
panel(axa, "a")

# ---- (b) Si convergence ----
sc = [2, 3, 4]
base = [52.8, 45.2, 45.5]; ft = [109.9, 114.0, 143.2]
axb.plot(sc, ft, "o-", color=C["finetuned"], ms=8, label="FC-distilled")
axb.plot(sc, base, "o-", mfc="none", color=C["baseline"], ms=8, label="baseline")
axb.axhline(140, ls="--", color=C["dfpt"], lw=1.2, label="experiment (~140)")
axb.set_xticks(sc); axb.set_xticklabels([f"{n}×{n}×{n}" for n in sc])
axb.set_xlabel("supercell")
axb.set_ylabel(r"Si $\kappa$(300 K)  (W m$^{-1}$K$^{-1}$)")
axb.set_title("Si κ converges to experiment")
axb.legend(loc="center right"); axb.grid(True)
axb.set_ylim(0, 160)
panel(axb, "b")

out = FIG / "fig4_kappa.png"
fig.savefig(out)
print("wrote", out)
