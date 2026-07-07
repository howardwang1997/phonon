"""(E)/(L) origin-map: the Part-II headline figure.

Each material plotted as (E)-channel response (x) vs (L)-channel instability depth (y).
- (E)-response = |dminfreq/dT_el| from family_melting.csv (how fast smearing melts the soft mode).
- (L)-depth    = most-negative SSCHA min-freq at low T (cm^-1) -> magnitude of lattice instability.
Color by polytype. Headline: 2H cluster high-(E)/~0-(L) (electronic origin);
1T-VSe2 sits at strong-(L) (lattice origin).
Run: conda run --no-capture-output -n phonon python scripts/smearing_kink/plot_origin_map.py
"""
from __future__ import annotations
import csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SK = ROOT / "results" / "smearing_kink"
TD = ROOT / "results" / "td_phonon"
OUT = SK / "origin_map.csv"
FIG = SK / "origin_map.png"

# --- (E)-response per material: slope of minfreq(T_el) from family_melting.csv ---
E = {}  # mat -> (polytype, slope THz/K)
melting = list(csv.DictReader(open(SK / "family_melting.csv")))
for m in sorted(set(r["material"] for r in melting)):
    rows = sorted([r for r in melting if r["material"] == m], key=lambda r: float(r["T_el_K"]))
    T = np.array([float(r["T_el_K"]) for r in rows])
    f = np.array([float(r["minfreq_THz"]) for r in rows])
    poly = rows[0]["polytype"]
    slope = abs(np.polyfit(T, f, 1)[0]) * 1000.0  # THz/K -> 1e-3 THz/K (responsivity)
    E[m] = (poly, slope)

# --- (L)-depth per material: most-negative SSCHA min-freq (cm^-1) ---
L_DEPTH = {  # material -> |min SSCHA freq| (cm^-1); 0 = no (L)-instability
    "1T-VSe2": 411.9,   # fine sweep, 50K deepest
    "2H-NbS2": 0.0, "NbS2": 0.0,
    "2H-TaS2": 0.0, "2H-TaSe2": 0.0,
    "NbSe2": 0.0,       # pathp SSCHA ~0 all T (electronic origin; Stage-D old model marginal)
}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated-ish 2-category palette: 2H (cool) vs 1T (warm); direct labels (no legend needed for <5)
COL = {"2H": "#2f6f9f", "1T": "#c0504d"}
rows_out = []
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for m, (poly, slope) in E.items():
    ld = L_DEPTH.get(m, 0.0)
    # nudge duplicates off x-axis floor for visibility (2H all ~0 (L))
    y = ld if ld > 0 else 8.0
    c = COL.get(poly, "#888")
    ax.scatter(slope, y, s=160, c=c, edgecolor="white", linewidth=1.2, zorder=3)
    ax.annotate(m, (slope, y), xytext=(6, 6), textcoords="offset points", fontsize=10, color="#222")
    rows_out.append({"material": m, "polytype": poly, "E_response_mThz_per_K": f"{slope:.3f}",
                     "L_depth_cm": f"{ld:.1f}", "origin": "lattice" if ld > 50 else "electronic"})

csv.writer(open(OUT, "w")).writerows([["material","polytype","E_response_mThz_per_K","L_depth_cm","origin"]])
with open(OUT, "a", newline="") as fh:
    w = csv.writer(fh)
    for r in rows_out:
        w.writerow([r["material"], r["polytype"], r["E_response_mThz_per_K"], r["L_depth_cm"], r["origin"]])

ax.set_yscale("symlog", linthresh=50)
ax.set_ylim(0, 1000)
ax.set_xlabel(r"(E)-channel response  $|\partial\omega_{\rm soft}/\partial T_{\rm el}|$  [$10^{-3}$ THz/K]", fontsize=11)
ax.set_ylabel(r"(L)-channel instability depth  $|\min\,\omega_{\rm SSCHA}|$  [cm$^{-1}$]", fontsize=11)
ax.axhspan(0, 50, color="#f4f4f4", zorder=0)  # "no (L)-instability" floor band
ax.text(0.02, 0.96, "no (L)-instability\n(electronic origin)", transform=ax.transAxes,
        va="top", ha="left", fontsize=9, color="#777")
ax.text(0.98, 0.96, "strong (L)-instability\n(lattice origin)", transform=ax.transAxes,
        va="top", ha="right", fontsize=9, color="#c0504d")
# polytype legend (2 entries -> small, legal)
for p, lab in [("2H", "2H (electronic)"), ("1T", "1T (lattice)")]:
    ax.scatter([], [], s=120, c=COL[p], label=lab, edgecolor="white")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_title("(E)/(L) origin map: 2H = electronic, 1T-VSe$_2$ = lattice", fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG, dpi=150)
print("wrote", OUT, "and", FIG)
for r in rows_out:
    print(f"  {r['material']:12s} {r['polytype']}  E={r['E_response_mThz_per_K']}  L={r['L_depth_cm']}  -> {r['origin']}")
