"""Family (L)-channel: SSCHA soft-mode vs lattice temperature — the Part-II origin figure.

Shows the polytype split:
  2H (NbSe2/NbS2/TaS2/TaSe2): soft-mode ~0 across all T_lat  -> (L)-inert = electronic origin
  1T (VSe2/TiSe2):            deep imaginary soft mode, heals to 0 AT T_CDW -> lattice/anharmonic origin

The crossover landing on the experimental T_CDW (VSe2 110K, TiSe2 200K) is the headline.
Run: conda run --no-capture-output -n phonon python scripts/smearing_kink/plot_family_L.py
"""
from __future__ import annotations
import csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "results" / "td_phonon"
OUT_CSV = ROOT / "results" / "smearing_kink" / "family_L_crossover.csv"
FIG = ROOT / "results" / "smearing_kink" / "family_L_crossover.png"

T_CDW = {"1T-VSe2": 110.0, "1T-TiSe2": 200.0}  # experimental CDW T (K); 2H inert -> n/a

def load(name):
    rows = sorted(csv.DictReader(open(TD / name)), key=lambda r: float(r["T_K"]))
    return np.array([[float(r["T_K"]), float(r["sscha_minfreq_cm"])] for r in rows])

# 1T (deep soft mode) — VSe2 uses the fine sweep + coarse points merged
vse2 = load("1T-VSe2_fine.csv")
vse2_c = load("1T-VSe2_L.csv")
vse2 = np.vstack([vse2, vse2_c])
vse2 = vse2[np.argsort(vse2[:, 0])]
tise2 = load("1T-TiSe2_L.csv")
# 2H (inert)
twoH = {"NbSe2": load("nbse2_sscha_pathp.csv"),
        "NbS2": load("NbS2_L.csv"),
        "2H-TaS2": load("2H-TaS2_L.csv"),
        "2H-TaSe2": load("2H-TaSe2_L.csv")}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_1T, COL_2H = "#c0504d", "#2f6f9f"
fig, ax = plt.subplots(figsize=(7.8, 5.4))

# 2H: all ~0 at 3x3 -> faint markers, one shared label band
for i, (m, d) in enumerate(twoH.items()):
    ax.scatter(d[:, 0], d[:, 1], s=42, c=COL_2H, alpha=0.45, marker="o",
               edgecolor="white", linewidth=0.5, zorder=2)
ax.plot([], [], "o", color=COL_2H, alpha=0.6, label="2H 3$\\times$3 (NbSe$_2$/NbS$_2$/TaS$_2$/TaSe$_2$) — (L)-inert")

# A2 honesty test: NbS2 4x4 reveals a low-T (L) soft mode the 3x3 missed (convergence caveat)
nbs2_4x4 = load("NbS2_4x4.csv")
ax.plot(nbs2_4x4[:, 0], nbs2_4x4[:, 1], "-^", color="#e08a2e", lw=1.8, ms=8, mfc="#e08a2e",
        mec="white", label="2H-NbS$_2$ 4$\\times$4 (A2: 3$\\times$3 under-converged)", zorder=3)

# 1T: prominent lines
ax.plot(vse2[:, 0], vse2[:, 1], "-o", color=COL_1T, lw=2.2, ms=7, mfc=COL_1T,
        mec="white", label="1T-VSe$_2$", zorder=4)
ax.plot(tise2[:, 0], tise2[:, 1], "--s", color=COL_1T, lw=2.0, ms=7, mfc="white",
        mec=COL_1T, mew=1.8, label="1T-TiSe$_2$", zorder=4)

# T_CDW markers
for m, T in T_CDW.items():
    ax.axvline(T, color="#bbb", ls=":", lw=1.0, zorder=1)
    ax.text(T, -560, f"$T_{{\\rm CDW}}$={int(T)}K\n({m[3:]})",
            ha="center", va="bottom", fontsize=8.5, color="#666")

ax.axhline(0, color="#888", lw=0.8, zorder=1)
ax.set_xlabel("lattice temperature  $T_{\\rm lat}$  [K]", fontsize=11)
ax.set_ylabel("SSCHA soft-mode min freq  [cm$^{-1}$]", fontsize=11)
ax.set_ylim(-620, 80)
ax.set_xlim(0, 320)
ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.set_title("(L)-channel: 1T soft mode heals at $T_{\\rm CDW}$; 2H inert (3$\\times$3)", fontsize=11)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG, dpi=150)

# csv summary: per-material (L)-depth + crossover T
rows = [["material", "polytype", "L_depth_cm", "crossover_K", "origin"]]
def crossover(d):
    neg = d[d[:, 1] < -1.0]
    pos = d[d[:, 1] >= -1.0]
    return float(pos[:, 0].min()) if len(pos) else float("nan")
for m, d in [("1T-VSe2", vse2), ("1T-TiSe2", tise2)]:
    rows.append([m, "1T", f"{-d[:,1].min():.1f}", f"{crossover(d):.0f}", "lattice"])
for m, d in twoH.items():
    rows.append([m, "2H", f"{-d[:,1].min():.1f}", "n/a", "electronic"])
csv.writer(open(OUT_CSV, "w")).writerows(rows)
print("wrote", FIG, "and", OUT_CSV)
for r in rows[1:]:
    print(f"  {r[0]:12s} {r[1]}  depth={r[2]:>7s} cm^-1  crossover={r[3]:>5s}  -> {r[4]}")
