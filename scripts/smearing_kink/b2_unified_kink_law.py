"""B2 step 1 — unified kink(T_el,T_lat) law: do the (E) smearing-melting and (L)
temperature-crossover soft-mode curves share a functional form?

Both are "soft mode heals to 0 at a characteristic T*". Normalize x=T/T*, y=depth/|depth_0|
and compare shapes. If they collapse onto one curve, the kink on the two axes is ONE law
=> the unified MLIP+LR target. (E) from family_melting.csv (THz); (L) from SSCHA csv (cm-1)."""
import csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SK = Path("results/smearing_kink")
CM2THz = 1.0/33.356   # cm-1 -> THz

def load_E(mat):
    rows = [(float(r["T_el_K"]), float(r["minfreq_THz"])) for r in csv.DictReader(open(SK/"family_melting.csv")) if r["material"]==mat]
    return np.array(sorted(rows))

def load_L(path):
    rows = [(float(r["T_K"]), float(r["sscha_minfreq_cm"])*CM2THz) for r in csv.DictReader(open(path))]
    return np.array(sorted(rows))

def heal_T(d):
    neg = d[d[:,1] < -0.02]; pos = d[d[:,1] >= -0.02]
    if len(neg)==0: return None, None
    depth = float(neg[:,1].min())  # most negative
    Tstar = float(pos[:,0].min()) if len(pos) else float(d[-1,0])
    return Tstar, depth

def norm(d, Tstar, depth):
    x = d[:,0]/Tstar; y = np.minimum(d[:,1]/depth, 1.0)  # clamp positive
    return x, np.where(d[:,1] < -0.02, y, 0.0)

E = {"1T-VSe2": load_E("1T-VSe2")}
L = {"1T-VSe2": load_L("/tmp/1T-VSe2_fine.csv"), "1T-TiSe2": load_L("/tmp/1T-TiSe2_L.csv")}

print(f"{'material':10s} {'axis':4s} {'T*':>7s} {'depth_THz':>10s} {'depth_cm':>9s}")
pts=[]
for mat in ["1T-VSe2"]:
    Te, de = heal_T(E[mat]); Tl, dl = heal_T(L[mat])
    print(f"{mat:10s} (E)  {Te:7.0f} {de:10.3f} {de/CM2THz:9.0f}")
    print(f"{mat:10s} (L)  {Tl:7.0f} {dl:10.3f} {dl/CM2THz:9.0f}")
    xe, ye = norm(E[mat], Te, de); xl, yl = norm(L[mat], Tl, dl)
    pts.append((mat+" (E), T*=%d"%Te, xe, ye, "#0072B2", "o"))
    pts.append((mat+" (L), T*=%d"%Tl, xl, yl, "#D55E00", "s"))

fig, ax = plt.subplots(figsize=(6.5,4.6))
for lab,x,y,c,m in pts:
    ax.plot(x, y, m, color=c, ms=9, mec="white", mew=1.2, label=lab, zorder=3)
# reference shared form: mean-field healing y=(1-x^2)^0.5 for x<1
xx = np.linspace(0,1.4,80); ax.plot(xx, np.sqrt(np.maximum(0,1-xx**2)), "k--", lw=1, alpha=0.5, label="mean-field $(1{-}x^2)^{1/2}$")
ax.axhline(0, color="#333", lw=0.6); ax.axvline(1, color="#999", lw=0.6, ls=":")
ax.set_xlabel(r"reduced temperature  $x = T/T^*$   ($T^*$=heal point: $T_{el}^*$ or $T_{\rm CDW}$)", fontsize=10)
ax.set_ylabel(r"normalized soft-mode depth  $|\omega|/|\omega_0|$", fontsize=10)
ax.set_title("Unified kink law? (E) smearing-melt vs (L) T-crossover — VSe$_2$\nboth axes heal through the same reduced-T window", fontsize=10)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(SK/"b2_unified_kink_law.png", dpi=150)
print("\nsaved", SK/"b2_unified_kink_law.png")
# quantitative: compare (E) and (L) y at matched x
print("\n=== shape comparison at matched reduced-T x ===")
for xq in [0.5, 0.7, 0.9]:
    ye_q = float(np.interp(xq, xe, ye)); yl_q = float(np.interp(xq, xl, yl))
    print(f"  x={xq}: (E) y={ye_q:.2f}  (L) y={yl_q:.2f}  diff={abs(ye_q-yl_q):.2f}")
