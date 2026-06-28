import csv, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
T={"Gamma":([],[]),"K":([],[])}
for r in csv.DictReader(open(ROOT/"results/td_phonon/graphene_K_echannel.csv")):
    if r["qlabel"] in T:
        top=float(r["freqs_cm"].split()[-1])
        T[r["qlabel"]][0].append(float(r["T_el_K"])); T[r["qlabel"]][1].append(top)
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(T["Gamma"][0],T["Gamma"][1],"o-",color="tab:blue",lw=2,ms=7,label="Γ-E$_{2g}$ (Δ≈40 cm$^{-1}$)")
ax.plot(T["K"][0],T["K"][1],"s-",color="tab:red",lw=2,ms=7,label="K-A$_1'$ (Δ≈103 cm$^{-1}$, sharper anomaly)")
ax.set_xlabel("electronic temperature $T_{el}$ (K)  [degauss = $k_B T_{el}$]")
ax.set_ylabel("phonon frequency (cm$^{-1}$)")
ax.set_title("Graphene (E)-channel at both Kohn anomalies (E7):\nΓ-E$_{2g}$ and K-A$_1'$ both stiffen with electronic T; K-A$_1'$ far more")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(ROOT/"results/figures/graphene_K_echannel.png",dpi=150)
print("saved results/figures/graphene_K_echannel.png")
