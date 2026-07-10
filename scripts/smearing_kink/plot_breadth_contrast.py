"""(E)-breadth contrast figure: gapped MoS2 (FLAT) vs CDW metals (melt with T_el).
Proves the (E)-smearing-dependence is a Fermi-surface / metallic-screening effect:
no FS -> no smearing dependence (MoS2 flat line = the negative control)."""
import csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SK = Path("results/smearing_kink")
fam = list(csv.DictReader(open(SK / "family_melting.csv")))      # 5 CDW metals
brd = list(csv.DictReader(open(SK / "breadth_melting.csv")))     # MoS2(gapped), 1T-TaS2, 1T-TiS2

# palette: gapped=gray, CDW metals=blues/reds, 1T breadth=light
CDW = {"NbS2": "#0072B2", "2H-TaS2": "#56B4E9", "2H-TaSe2": "#009E73",
       "NbSe2": "#CC79A7", "1T-VSe2": "#D55E00"}
fig, ax = plt.subplots(figsize=(7.5, 5.0))

# CDW metals (melt)
for m, c in CDW.items():
    pts = sorted([(float(r["T_el_K"]), float(r["minfreq_THz"])) for r in fam if r["material"] == m])
    if not pts: continue
    T = np.array([p[0] for p in pts]) / 1000; f = [p[1] for p in pts]
    ax.plot(T, f, "-o", color=c, ms=5, mec="white", mew=0.8, lw=1.4, label=m, alpha=0.9)

# MoS2 gapped control (FLAT) — bold black dashed
pts = sorted([(float(r["T_el_K"]), float(r["minfreq_THz"])) for r in brd if r["material"] == "MoS2"])
T = np.array([p[0] for p in pts]) / 1000; f = [p[1] for p in pts]
ax.plot(T, f, "--s", color="#222", ms=7, mec="#222", mfc="white", mew=1.8, lw=1.8,
        label="MoS$_2$ (gapped ctrl)", zorder=5)

# 1T-TaS2 / 1T-TiS2 (metallic, lighter)
for m, c in [("1T-TaS2", "#E69F00"), ("1T-TiS2", "#999")]:
    pts = sorted([(float(r["T_el_K"]), float(r["minfreq_THz"])) for r in brd if r["material"] == m])
    if not pts: continue
    T = np.array([p[0] for p in pts]) / 1000; f = [p[1] for p in pts]
    ax.plot(T, f, ":^", color=c, ms=5, lw=1.2, label=m, alpha=0.8)

ax.axhline(0, color="#333", lw=0.8, ls="-", alpha=0.5)
ax.text(3.05, 0.06, "melted", fontsize=8, color="#555")
ax.set_xlabel(r"electronic temperature  $T_{\rm el}$ (10$^3$ K)", fontsize=11)
ax.set_ylabel(r"soft-mode min freq  (THz)", fontsize=11)
ax.set_title("(E)-channel is metallic-screening-driven: gapped MoS$_2$ stays flat\nwhile CDW metals melt with smearing", fontsize=10.5)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.12, bottom=0.12, right=0.72, top=0.90)
fig.savefig(SK / "breadth_contrast.png", dpi=150, bbox_inches="tight")
print("saved", SK / "breadth_contrast.png")
