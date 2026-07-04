import sys, numpy as np
sys.path.insert(0, "scripts/smearing_kink")
import friedel_module as fm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ph_bg = fm.load_ph("results/vq_kink6/graphene_sc6_dg0.080_phonopy.yaml")
fc_bg = ph_bg.force_constants
fc_sharp = fm.load_ph("results/vq_kink6/graphene_sc6_dg0.002_phonopy.yaml").force_constants
tabs, _ = fm.pair_table(ph_bg)
D0 = fm.template_delta(fc_sharp, fc_bg, tabs)   # smearing-induced dfc2 per pair

Rs, norms = [], []
for p, t in enumerate(tabs):
    R = t["R"]
    fro = np.linalg.norm(D0[p].reshape(len(R), 9), axis=1)  # ||dfc2|| per pair
    Rs.append(R); norms.append(fro)
R = np.concatenate(Rs); N = np.concatenate(norms)
Rmax_6x6 = R.max()
# bin by distance
bins = np.arange(0, Rmax_6x6+0.6, 0.5)
idx = np.digitize(R, bins)
binR = 0.5*(bins[:-1]+bins[1:])
rms = np.array([np.sqrt(np.mean(N[idx==i+1]**2)) if np.any(idx==i+1) else 0 for i in range(len(bins)-1)])
peak = rms.max()
print(f"6x6 max pair distance = {Rmax_6x6:.2f} A")
print(f"{'R(A)':>6} {'RMS|dfc2|':>10} {'% of peak':>9}")
for r, v in zip(binR, rms):
    if v>0: print(f"{r:>6.2f} {v:>10.4f} {100*v/peak:>8.1f}%")
# tail fraction near the cell edge
edge = rms[binR > Rmax_6x6-1.5]
print(f"\n=> Delta-fc2 at edge (R>{Rmax_6x6-1.5:.1f}A) = {100*edge.max()/peak:.1f}% of peak")
print(f"=> {'TRUNCATED: tail still significant at 6x6 edge -> need bigger cell' if edge.max()/peak>0.05 else 'CONVERGED: tail decayed within 6x6 -> 6x6 sufficient'}")

fig,ax=plt.subplots(figsize=(6.2,4.2))
ax.semilogy(binR, rms/peak, "-o", color="#0072B2", lw=2, ms=5)
ax.axvline(Rmax_6x6, ls="--", color="#D55E00", label=f"6×6 cutoff {Rmax_6x6:.1f} Å")
ax.axhline(0.05, ls=":", color="#555", label="5% threshold")
ax.set_xlabel("pair distance R (Å)"); ax.set_ylabel(r"RMS $|\Delta$fc$_2|$ / peak (smearing-induced)")
ax.set_title("Friedel tail decay in 6×6 graphene (dg0.002 − dg0.080)", fontsize=10)
ax.legend(frameon=False, fontsize=8.5); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/smearing_kink/friedel_tail_decay.png", dpi=150)
print("saved results/smearing_kink/friedel_tail_decay.png")
