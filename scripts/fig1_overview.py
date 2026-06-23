"""Figure 1 — concept / overview schematic for the whole paper.

Left-to-right narrative with embedded real-data insets:
  [A] the problem  ->  [B] FC distillation (zero new DFT)  ->  [D] payoff
  [C] closed-loop data engine (breadth>=depth, coverage>=uncertainty) feeds [B]

Insets [A] PES curvature sketch (illustrative), [B] a Phi force-constant matrix
(illustrative), [D] mini Si dispersion + mini kappa scatter (REAL data, reused
from results/figures/data/si_dispersion.npz and results/kappa/sc4_*.json).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_style import C, set_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

set_style()
FIG = ROOT / "results" / "figures"

# soft fills for the four story blocks
FILL = {"prob": "#fdecea", "meth": "#eaf3fb", "eng": "#f1ecf7", "pay": "#eaf6ee"}
EDGE = {"prob": C["mace"], "meth": C["mattersim"], "eng": C["breadth"], "pay": C["finetuned"]}

fig = plt.figure(figsize=(13.5, 6.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def block(x0, y0, x1, y1, key, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.006,rounding_size=0.02",
                 fc=FILL[key], ec=EDGE[key], lw=1.8, zorder=1))
    ax.text((x0 + x1) / 2, y1 - 0.035, title, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color=EDGE[key], zorder=3)


def arrow(x0, y0, x1, y1, color="0.25", lw=2.4, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=22, lw=lw, color=color, ls=ls, zorder=4))


# ===================== top row: problem -> method -> payoff =====================
PY0, PY1 = 0.40, 0.93
block(0.015, PY0, 0.265, PY1, "prob", "1. The problem")
block(0.355, PY0, 0.665, PY1, "meth", "2. FC distillation — zero new DFT")
block(0.735, PY0, 0.985, PY1, "pay", "4. Payoff")

# ---- [A] problem: PES curvature inset (top) + caption stacked below ----
axp = ax.inset_axes([0.065, 0.635, 0.15, 0.18])
u = np.linspace(-1, 1, 100)
axp.plot(u, 0.5 * 8.5 * u ** 2, color=C["dfpt"], lw=2)
axp.plot(u, 0.5 * 3.6 * u ** 2, color=C["mace"], lw=2)
axp.set_xticks([]); axp.set_yticks([]); axp.set_ylim(0, 4.2)
axp.set_title("PES curvature", fontsize=8.5, pad=2)
axp.text(-0.52, 3.35, "DFT", color=C["dfpt"], fontsize=7.5, ha="center")
axp.text(0.66, 0.65, "MLIP\n(too soft)", color=C["mace"], fontsize=7.5, ha="center", va="center")
for s in axp.spines.values():
    s.set_linewidth(0.8)
ax.text(0.03, 0.59, "Foundation MLIPs under-predict\nthe PES curvature:",
        fontsize=9, va="top", ha="left")
ax.text(0.03, 0.50,
        r"$\Rightarrow$ phonons soften $-$10 to $-$30%" "\n"
        r"$\Rightarrow$ spurious imaginary modes" "\n"
        r"$\Rightarrow$ $\kappa \approx \frac{1}{2}\times$ experiment",
        fontsize=8.8, va="top", ha="left")

# ---- [B] FC distillation: Phi matrix inset + equations ----
axm = ax.inset_axes([0.375, 0.575, 0.115, 0.245])
rng = np.add.outer(np.arange(9), np.arange(9))
phi = np.cos(0.9 * (rng % 9)) * np.exp(-0.18 * np.abs(np.subtract.outer(np.arange(9), np.arange(9))))
axm.imshow(phi, cmap="RdBu_r", vmin=-1, vmax=1)
axm.set_xticks([]); axm.set_yticks([])
axm.set_title(r"DFPT $\Phi$", fontsize=9, pad=2)
arrow(0.495, 0.70, 0.545, 0.70, color="0.35", lw=2.0)
ax.text(0.55, 0.755,
        r"$E=\frac{1}{2}\,\mathbf{u}^\top\Phi\,\mathbf{u}$" "\n"
        r"$\mathbf{F}=-\Phi\,\mathbf{u}$",
        fontsize=11, va="center", ha="left")
ax.text(0.55, 0.625,
        "harmonic labels from\n"
        "already-computed force\n"
        "constants — no new DFT",
        fontsize=8.8, va="center", ha="left")
ax.text(0.51, 0.475,
        r"fine-tune foundation MLIP $\rightarrow$ in-domain phonon MAE $\approx$ 0.10 THz",
        fontsize=9.2, va="center", ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=EDGE["meth"], lw=1))

# ---- [D] payoff: mini dispersion + mini kappa scatter (REAL data) ----
axd1 = ax.inset_axes([0.755, 0.575, 0.10, 0.255])
d = np.load(FIG / "data" / "si_dispersion.npz", allow_pickle=True)
for pre, col, lw, ls in [("dfpt", C["dfpt"], 0.9, "--"), ("base", C["mace"], 0.9, "-"),
                          ("ft", C["finetuned"], 1.0, "-")]:
    for dist, freq in zip(d[f"{pre}_dist"], d[f"{pre}_freq"]):
        axd1.plot(np.asarray(dist), np.asarray(freq), color=col, lw=lw, ls=ls)
axd1.set_xticks([]); axd1.set_yticks([0, 5, 10, 15]); axd1.tick_params(labelsize=6)
axd1.set_title("Si phonons", fontsize=8.5, pad=2); axd1.set_ylabel("THz", fontsize=7)
axd1.margins(x=0)

axd2 = ax.inset_axes([0.895, 0.575, 0.082, 0.255])
EXP = {"Si": 140, "Ge": 60, "C": 2200, "Sn": 11, "SiC": 430, "BN": 760, "BP": 400,
       "BAs": 1300, "AlP": 90, "AlAs": 91, "GaP": 100, "GaAs": 45, "InP": 68}
r = {}
for f in glob.glob(str(ROOT / "results" / "kappa" / "sc4_*.json")):
    dd = json.load(open(f)); k = dd["kappa"].get("300")
    if k:
        r.setdefault(dd["material"], {})["ft" if f.endswith("_ft.json") else "base"] = k
lim = [5, 4000]
axd2.plot(lim, lim, "--", color="0.6", lw=0.8)
for mm, v in r.items():
    e = EXP.get(mm)
    if not e:
        continue
    if "base" in v:
        axd2.plot(e, v["base"], "o", mfc="none", mec=C["baseline"], ms=3.2, mew=0.8)
    if "ft" in v:
        axd2.plot(e, v["ft"], "o", color=C["finetuned"], ms=3.2)
axd2.set_xscale("log"); axd2.set_yscale("log"); axd2.set_xlim(lim); axd2.set_ylim(lim)
axd2.set_xticks([]); axd2.set_yticks([])
axd2.set_title(r"$\kappa$ vs exp", fontsize=8.5, pad=2)
for s in axd2.spines.values():
    s.set_linewidth(0.8)
ax.text(0.86, 0.49,
        "near-DFT phonons  +  " r"$\kappa\,\rightarrow$ experiment" "\n"
        r"(Si $\kappa$ 143 vs 140, ${\sim}$2%; every covalent $\uparrow$)",
        fontsize=8.8, va="center", ha="center")

# top-row flow arrows
arrow(0.265, 0.665, 0.355, 0.665, lw=2.6)
arrow(0.665, 0.665, 0.735, 0.665, lw=2.6)

# ===================== bottom band: closed-loop data engine =====================
EY0, EY1 = 0.06, 0.305
block(0.355, EY0, 0.665, EY1, "eng", "3. Closed-loop data engine")
ax.text(0.51, 0.205,
        r"$\bullet$  spend DFT on BREADTH, not depth",
        fontsize=9.4, va="center", ha="center")
ax.text(0.51, 0.115,
        r"$\bullet$  acquire by COVERAGE, not uncertainty",
        fontsize=9.4, va="center", ha="center")

# loop arrows: engine feeds method (up); payoff/model feeds engine (down, curved)
arrow(0.45, 0.305, 0.45, 0.40, color=EDGE["eng"], lw=2.4)
ax.text(0.456, 0.355, "targeted\nbreadth", fontsize=7.8, va="center", ha="left", color=EDGE["eng"])
# closed loop: from payoff back down and into the engine top edge
arrow(0.86, 0.40, 0.60, 0.307, color=EDGE["eng"], lw=2.0, rad=0.30, ls="-")
ax.text(0.74, 0.355, "iterate", fontsize=7.8, va="center", ha="center",
        color=EDGE["eng"], style="italic")

# left label tying problem -> engine motivation
arrow(0.14, 0.40, 0.355, 0.20, color="0.55", lw=1.6, rad=-0.18, ls=(0, (4, 3)))
ax.text(0.225, 0.30, "what to\ncompute?", fontsize=7.8, va="center", ha="center", color="0.45")

fig.text(0.5, 0.985,
         "Small targeted DFT, distilled as curvature, large MLIP at scale: "
         "near-DFT phonons & thermal conductivity",
         ha="center", va="top", fontsize=12.5, fontweight="bold")

out = FIG / "fig1_overview.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
