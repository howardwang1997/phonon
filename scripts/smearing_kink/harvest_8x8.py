"""Harvest: does the Kohn kink change 6x6 -> 8x8? (supercell convergence of the
smearing-dependent phonon spectrum). 8x8 reaches ~9.8A clean vs 6x6 ~7A."""
import sys, csv, numpy as np
sys.path.insert(0, "scripts/smearing_kink")
import friedel_module as fm

# existing 6x6 kink(T_el)
k6 = {}
for r in csv.DictReader(open("results/smearing_kink/graphene_kink_Tel_6x6.csv")):
    k6[r["degauss_Ry"]] = float(r["kinkK"])
DG = ["0.005","0.01","0.02","0.04","0.08"]
DG6 = {"0.005":"0.005","0.01":"0.010","0.02":"0.020","0.04":"0.040","0.08":"0.08"}
TEL = {d: float(d)*157888 for d in DG}

print(f"{'dg':>7} {'T_el':>6} {'kink_6x6':>9} {'kink_8x8':>9} {'delta':>7}")
rows=[]
for d in DG:
    ph = fm.load_ph(f"results/sc_conv/graphene_sc8_dg{d}_phonopy.yaml")
    k8 = fm.kink_of(ph, ph.force_constants)[0]
    k6v = k6.get(DG6[d], float("nan"))
    rows.append((d, TEL[d], k6v, k8))
    print(f"{d:>7} {TEL[d]:>6.0f} {k6v:>9.2f} {k8:>9.2f} {k8-k6v:>+7.2f}")
md = np.mean([abs(r[3]-r[2]) for r in rows])
print(f"\n=> mean |kink_8x8 - kink_6x6| = {md:.2f} cm^-1  (supercell not converged if >~model MAE 1.7)")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T=[r[1] for r in rows]; a=[r[2] for r in rows]; b=[r[3] for r in rows]
fig,ax=plt.subplots(figsize=(6.2,4.3))
ax.plot(T,a,"-o",color="#0072B2",lw=2,ms=6,label="6×6 (clean ~7 Å)")
ax.plot(T,b,"-s",color="#D55E00",lw=2,ms=6,label="8×8 (clean ~9.8 Å)")
for t,x,y in zip(T,a,b): ax.plot([t,t],[x,y],color="#bbb",lw=.8,zorder=0)
ax.set_xlabel("electronic temperature T_el (K)"); ax.set_ylabel("Kohn kink at K (cm$^{-1}$)")
ax.set_title(f"Supercell convergence of the smearing-dependent kink\n6×6 vs 8×8 (mean Δ={md:.1f} cm⁻¹)",fontsize=10)
ax.legend(frameon=False,fontsize=9); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/smearing_kink/kink_6x6_vs_8x8.png",dpi=150)
print("saved results/smearing_kink/kink_6x6_vs_8x8.png")
