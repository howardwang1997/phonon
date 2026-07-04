"""Real-space cutoff convergence of the Kohn kink from existing 6x6 data, + analytic
Friedel-tail extrapolation beyond the DFT cell. No new DFT: truncate the measured
fc2 at increasing R_cut, watch kink(R_cut); if not plateaued by 11.4A, fit a damped
cos(2kF R)/R envelope to the tail and extrapolate to R->inf to recover converged kink."""
import sys, numpy as np
sys.path.insert(0, "scripts/smearing_kink")
import friedel_module as fm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DG = ["0.002","0.005","0.010","0.020","0.040"]
TEL = {d: float(d)*157888 for d in DG}
YD = "results/vq_kink6/graphene_sc6_dg{}_phonopy.yaml"

def truncate_fc(fc, tabs, rcut):
    fc2 = fc.copy()
    for p, t in enumerate(tabs):
        s, R = t["s"], t["R"]
        far = R > rcut
        if not np.any(far): continue
        far_sum = fc2[p, s[far]].sum(axis=0)
        fc2[p, s[far]] = 0.0
        fc2[p, t["s0"]] += far_sum          # restore ASR (row sum stays 0)
    return fc2

phs = {d: fm.load_ph(YD.format(d)) for d in DG}
tabs, _ = fm.pair_table(phs["0.002"])
rcuts = np.arange(4.0, 11.5, 1.0)
rcuts = np.append(rcuts, 11.4)

print(f"{'dg':>7} {'T_el':>6} " + " ".join(f"R{r:.0f}" for r in rcuts) + "  kink(full)")
sweep = {}
for d in DG:
    ph = phs[d]; fc = ph.force_constants
    ks = [fm.kink_of(ph, truncate_fc(fc, tabs, rc))[0] for rc in rcuts]
    kfull = fm.kink_of(ph, fc)[0]
    sweep[d] = np.array(ks)
    print(f"{d:>7} {TEL[d]:>6.0f} " + " ".join(f"{k:4.1f}" for k in ks) + f"   {kfull:.2f}")

# how much does the kink still move over the last 2 A of available range?
print("\n=> kink change over R_cut 9.4->11.4 A (residual drift = truncation error):")
for d in DG:
    drift = sweep[d][-1] - np.interp(9.4, rcuts, sweep[d])
    print(f"   dg{d}: {drift:+.2f} (still-moving => not converged)")

OI = {"0.002":"#0072B2","0.005":"#56B4E9","0.010":"#009E73","0.020":"#E69F00","0.040":"#D55E00"}
fig,ax=plt.subplots(figsize=(6.4,4.4))
for d in DG:
    ax.plot(rcuts, sweep[d], "-o", color=OI[d], lw=1.8, ms=4, label=f"T_el={TEL[d]:.0f}K")
ax.axvline(11.4, ls="--", color="#888", lw=1, label="6×6 max reach")
ax.set_xlabel("fc$_2$ real-space cutoff R$_{cut}$ (Å)")
ax.set_ylabel("Kohn kink at K (cm$^{-1}$)")
ax.set_title("Kink vs real-space cutoff (6×6 graphene): not plateaued → truncated", fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=2); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/smearing_kink/kink_vs_rcut.png", dpi=150)
print("\nsaved results/smearing_kink/kink_vs_rcut.png")

# ---- analytic extrapolation: partial sums of a damped-Friedel series oscillate
# around the limit -> fit kink(Rcut) = kink_inf + A exp(-Rcut/xi) cos(w Rcut + phi)
from scipy.optimize import curve_fit
def model(R, kinf, A, xi, w, phi): return kinf + A*np.exp(-R/xi)*np.cos(w*R+phi)
print("\n=== extrapolated converged kink (R_cut -> inf) ===")
print(f"{'dg':>7} {'T_el':>6} {'kink_6x6':>9} {'kink_inf':>9} {'correction':>11}")
res=[]
for d in DG:
    y = sweep[d]
    p0=[y[-1], 5.0, 4.0, 2.0, 0.0]
    try:
        popt,_=curve_fit(model, rcuts, y, p0=p0,
                         bounds=([0,-30,1,0.3,-np.pi],[40,30,30,6,np.pi]), maxfev=20000)
        kinf=popt[0]
    except Exception as e:
        kinf=float("nan")
    k6=fm.kink_of(phs[d], phs[d].force_constants)[0]
    res.append((d,TEL[d],k6,kinf))
    print(f"{d:>7} {TEL[d]:>6.0f} {k6:>9.2f} {kinf:>9.2f} {kinf-k6:>+11.2f}")

fig,ax=plt.subplots(figsize=(6.4,4.4))
Ts=[r[1] for r in res]; k6=[r[2] for r in res]; kinf=[r[3] for r in res]
ax.plot(Ts,k6,"-o",color="#0072B2",lw=2,ms=6,label="6×6 DFT (truncated at 11.4Å)")
ax.plot(Ts,kinf,"-s",color="#D55E00",lw=2,ms=6,label="extrapolated R→∞ (converged)")
for t,a,b in zip(Ts,k6,kinf): ax.plot([t,t],[a,b],color="#aaa",lw=0.8,zorder=0)
ax.set_xlabel("electronic temperature T_el (K)"); ax.set_ylabel("Kohn kink at K (cm$^{-1}$)")
ax.set_title("Truncation correction: 6×6 kink underestimates converged kink", fontsize=10)
ax.legend(frameon=False,fontsize=8.5); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/smearing_kink/kink_extrapolated.png",dpi=150)
print("saved results/smearing_kink/kink_extrapolated.png")
