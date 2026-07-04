import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T=np.array([789,1579,3158,6316,12631])/1000
k6k4=[18.32,13.66,9.75,3.92,0.10]; k8k4=[6.12,6.35,6.45,3.40,0.10]; k6k6=[15.72,13.33,9.75,3.91,0.22]
fig,ax=plt.subplots(figsize=(6.4,4.4))
ax.plot(T,k6k4,"-o",color="#0072B2",lw=2,ms=6,label="6×6  (clean ~7 Å)")
ax.plot(T,k8k4,"-s",color="#D55E00",lw=2,ms=6,label="8×8  (clean ~9.8 Å)")
ax.plot(T,k6k6,":^",color="#0072B2",lw=1.2,ms=5,alpha=.5,label="6×6 @ denser k (≈overlaps → k not the issue)")
for t,a,b in zip(T,k6k4,k8k4): ax.plot([t,t],[a,b],color="#bbb",lw=.8,zorder=0)
ax.set_xlabel(r"electronic temperature $T_{\rm el}$ (10$^3$ K)"); ax.set_ylabel("Kohn kink at K (cm$^{-1}$)")
ax.set_title("Kink truncation is SUPERCELL, not k-density (matched kpts4)\n6×6 overestimates by up to +12 cm⁻¹ at sharp smearing",fontsize=9.5)
ax.legend(frameon=False,fontsize=8.5); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/smearing_kink/supercell_disentangle.png",dpi=150); print("saved fig")
