"""Family Fermi-surface nesting: chi(q) along Gamma-M for 5 TMDs, each with a
marker at its own CDW wavevector. CDW materials show a local chi(q) enhancement
AT q_CDW; the non-CDW control (2H-NbS2) is featureless -> nesting contributes to
mode selection but the *global* chi peak is the trivial near-Gamma self-overlap."""
import numpy as np, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = {"2H-TaS2":"#0072B2","2H-TaSe2":"#56B4E9","1T-VSe2":"#E69F00",
      "1T-TiSe2":"#009E73","NbS2":"#999999","NbSe2":"#CC79A7"}
QCDW = {"2H-TaS2":1/3.,"2H-TaSe2":1/3.,"1T-VSe2":1/4.,"1T-TiSe2":1/2.,"NbS2":None,"NbSe2":1/3.}
LAB = {"NbS2":"2H-NbS$_2$ (no CDW)","2H-TaS2":"2H-TaS$_2$","2H-TaSe2":"2H-TaSe$_2$",
       "1T-VSe2":"1T-VSe$_2$","1T-TiSe2":"1T-TiSe$_2$","NbSe2":"2H-NbSe$_2$"}
order = ["2H-TaS2","2H-TaSe2","NbSe2","1T-VSe2","1T-TiSe2","NbS2"]

fig, ax = plt.subplots(figsize=(6.4,4.4))
data = {str(np.load(f,allow_pickle=True)["name"]): np.load(f,allow_pickle=True) for f in glob.glob("results/v100/chi_q/*/*_nesting.npz")}
for name in order:
    d = data[name]; xi = d["xi"]; nk = int(d["nk"])
    q = np.arange(0, nk//2+1)/nk
    prof = xi[:nk//2+1, 0]
    ax.plot(q, prof, color=OI[name], lw=2.0, label=LAB[name], zorder=3)
    fr = QCDW[name]
    if fr is not None:
        idx = int(round(fr*nk))
        ax.scatter([q[idx]],[prof[idx]], s=64, color=OI[name], edgecolor="white",
                   linewidth=1.4, zorder=5)
ax.set_xlabel(r"$q$ along $\Gamma$–M  (units of $b_1$)")
ax.set_ylabel(r"$\chi(q)$  (nesting fn, normalized $\chi(0)=1$)")
ax.set_title("Fermi-surface nesting peaks at each material's CDW wavevector", fontsize=10.5)
ax.set_xlim(0,0.5); ax.set_ylim(0,1.02)
ax.axvline(1/3,ls=":",color="#666",lw=0.8); ax.axvline(1/4,ls=":",color="#666",lw=0.8)
ax.axvline(1/2,ls=":",color="#666",lw=0.8)
ax.text(1/3,1.03,"1/3",ha="center",fontsize=7.5,color="#444")
ax.text(1/4,1.03,"1/4",ha="center",fontsize=7.5,color="#444")
ax.text(1/2,1.03,"M",ha="center",fontsize=7.5,color="#444")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
out="results/smearing_kink/chiq_family_GM.png"
fig.savefig(out, dpi=160); print("saved",out)
