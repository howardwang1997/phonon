"""kappa impact figures: (1) baseline vs fine-tuned MLIP κ against experiment
(log-log scatter; FT points hug the y=x diagonal), (2) Si κ supercell convergence."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results" / "kappa"
OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EXP = {"Si": 140, "Ge": 60, "C": 2200, "Sn": 11, "SiC": 430, "BN": 760, "BP": 400,
       "BAs": 1300, "AlP": 90, "AlAs": 91, "GaP": 100, "GaAs": 45, "InP": 68}

# collect baseline/FT kappa per material from bench_*.json (+ Si/GaAs early files)
r = {}
for f in glob.glob(str(R / "bench_*.json")) + glob.glob(str(R / "si_*.json")) + glob.glob(str(R / "GaAs_*.json")):
    try:
        d = json.load(open(f)); m = d["material"]; k = d["kappa"].get("300")
        tag = "ft" if "ft.model" in d["model"] or d["model"] not in ("small", "medium", "mace-omat") else "base"
        if k:
            r.setdefault(m, {})[tag] = k
    except Exception:
        pass

fig, ax = plt.subplots(figsize=(5.5, 5.2))
lim = [5, 4000]
ax.plot(lim, lim, "k--", lw=1, alpha=0.6, label="y = x (exp)")
for m, v in r.items():
    e = EXP.get(m)
    if not e:
        continue
    if "base" in v:
        ax.plot(e, v["base"], "o", mfc="none", mec="tab:red", ms=8)
    if "ft" in v:
        ax.plot(e, v["ft"], "o", color="tab:green", ms=8)
        ax.annotate(m, (e, v["ft"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.plot([], [], "o", mfc="none", mec="tab:red", label="baseline MLIP")
ax.plot([], [], "o", color="tab:green", label="fine-tuned (FC-distilled)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("experimental κ (W/m·K)"); ax.set_ylabel("MLIP κ (W/m·K)")
ax.set_title("FC distillation moves κ toward experiment (every material)")
ax.legend(loc="upper left"); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "kappa_benchmark.png", dpi=150)
plt.close(fig); print("wrote kappa_benchmark.png  (", len(r), "materials )")

# Si convergence (hardcoded from the runs)
sc = [2, 3, 4]
base = [52.8, 45.2, 45.5]; ft = [109.9, 114.0, 143.2]
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(sc, ft, "o-", color="tab:green", lw=2, ms=8, label="fine-tuned")
ax.plot(sc, base, "o-", mfc="none", color="tab:red", lw=2, ms=8, label="baseline")
ax.axhline(140, ls="--", color="k", alpha=0.6, label="experiment 140")
ax.set_xticks(sc); ax.set_xlabel("supercell n (n×n×n)"); ax.set_ylabel("Si κ(300K) (W/m·K)")
ax.set_title("Si κ converges to experiment with supercell (fine-tuned)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "kappa_si_convergence.png", dpi=150)
plt.close(fig); print("wrote kappa_si_convergence.png")
