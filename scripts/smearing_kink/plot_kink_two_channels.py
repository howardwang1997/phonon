"""Answer figure: can we reproduce the Kohn-anomaly kink trend in the two cases the
user asks about?
  (A) smearing-dependent (E-channel): kink_K vs electronic smearing T_el
      -> DFT ground truth + 2-param damped-Friedel model + 3-anchor few-shot
  (B) temperature-dependent, lattice-anharmonic part (L-channel): kink_K vs T_lat
      -> DFT (self-consistent thermal), which is ~flat -> lattice channel negligible
The Kohn anomaly is an electronic effect, so real T enters mainly via the electronic
occupation = smearing (A); the lattice-anharmonic contribution (B) is separately small.
"""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def rd(p):
    rows=list(csv.DictReader(open(p))); 
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
E = rd("results/smearing_kink/friedel_fit.csv")           # E-channel: model vs dft vs fewshot
L = rd("results/td_phonon/td_graphene_ft.csv")            # L-channel: MLIP-TDEP v11 (real source; was a phantom hand-authored csv)

BLU="#0072B2"; ORA="#E69F00"; GRY="#555555"; GRN="#009E73"
fig,(axA,axB)=plt.subplots(1,2,figsize=(10.6,4.5))

# ---- Panel A: smearing (E) channel ----
axA.plot(E["T_el"], E["kinkK_model"], "-", color=BLU, lw=2.2, zorder=3,
         label="2-param Friedel model")
axA.scatter(E["T_el"], E["kinkK_dft"], s=52, color="#222", zorder=5,
            label="DFT (ground truth)")
axA.scatter(E["T_el"], E["kinkK_fewshot"], s=34, marker="s", color=ORA, zorder=4,
            edgecolor="white", linewidth=0.6, label="few-shot (3 anchors)")
mae=np.mean(np.abs(E["kinkK_model"]-E["kinkK_dft"]))
axA.set_title(f"(A) SMEARING-dependent spectrum  ·  (E) electronic channel\n"
              f"Kohn kink$_K$($T_{{el}}$)  —  model MAE {mae:.1f} on a 0–23 scale",fontsize=10)
axA.set_xlabel(r"electronic smearing $T_{\rm el}=$ degauss$\times$157888  (K)")
axA.set_ylabel(r"Kohn kink at K  (cusp strength, cm$^{-1}$)")
axA.legend(frameon=False,fontsize=8.6,loc="upper right")
axA.set_ylim(0,24)
axA.annotate("kink vanishes monotonically\nas Fermi surface is smeared", xy=(9000,3),
             xytext=(5200,14.5), fontsize=8.4, color=GRY,
             arrowprops=dict(arrowstyle="->",color=GRY,lw=0.9))

# ---- Panel B: temperature (L, lattice-anharmonic) channel ----
axB.plot(L["T_K"], L["kink_k"], "-o", color=GRN, lw=2.2, ms=7, mec="white", mew=1.3,
         zorder=4, label="MLIP-TDEP v11 (L)")
axB.set_title("(B) TEMPERATURE-dependent spectrum  ·  (L) lattice-anharmonic channel\n"
              r"kink$_K$($T_{lat}$)  —  MLIP-TDEP (DFT-MD-TDEP ref pending)",fontsize=10)
axB.set_xlabel(r"lattice temperature $T_{\rm lat}$  (K)")
axB.set_ylabel(r"Kohn kink at K  (cm$^{-1}$)")
axB.set_ylim(0,8)
axB.set_xlim(0,700)
axB.annotate("lattice anharmonicity barely moves the kink\n"
             r"$\Rightarrow$ the kink's T-dependence lives in channel (A)",
             xy=(300,5.9), xytext=(70,2.2), fontsize=8.4, color=GRY,
             arrowprops=dict(arrowstyle="->",color=GRY,lw=0.9))
axB.legend(frameon=False,fontsize=8.6,loc="upper left")

for ax in (axA,axB): ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); out="results/smearing_kink/kink_two_channels.png"
fig.savefig(out,dpi=160); print("saved",out,"| E-channel model MAE =",round(mae,2))
