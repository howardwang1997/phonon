import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
d=np.load(ROOT/"results/td_phonon/nbse2_pathp_analysis.npz", allow_pickle=True)
A=d["A"]
fig,ax=plt.subplots(figsize=(7,5))
ax.axhline(0,color="0.6",lw=0.7,ls="--")
if "dft_A" in d: ax.plot(d["dft_A"],d["dft_E"],"o-",color="k",lw=2,ms=6,label="DFT (truth): −3 meV well, stiff",zorder=5)
ax.plot(A,d["E_harmonicFT"],"s-",color="tab:orange",lw=1.8,ms=5,label="harmonic-FT: −19 meV well, 5× too soft (699 meV/Å)")
ax.plot(A,d["E_anharmFT"],"^-",color="tab:green",lw=1.8,ms=5,label="anharmonic-FT (Path-P): ≈DFT (151 meV/Å)")
ax.set_xlabel("amplitude A along CDW soft eigenvector (Å, RMS/atom)")
ax.set_ylabel("energy along soft mode (meV/cell, ref A=0)")
ax.set_title("NbSe$_2$ CDW double-well: anharmonic distillation reproduces the (stiff,\nmarginal) DFT landscape; harmonic distillation distorts it")
ax.legend(fontsize=8.5,loc="upper center")
fig.tight_layout(); fig.savefig(ROOT/"results/figures/nbse2_doublewell.png",dpi=150)
print("saved results/figures/nbse2_doublewell.png")
