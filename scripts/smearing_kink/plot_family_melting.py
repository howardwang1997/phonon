"""Family (E)-melting: soft-mode min-freq vs T_el for all CDW members (incl NbSe2).
Reads results/smearing_kink/family_melting.csv."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=list(csv.DictReader(open("results/smearing_kink/family_melting.csv")))
mats={}
for r in rows: mats.setdefault(r["material"],[]).append((float(r["T_el_K"]),float(r["minfreq_THz"])))
OI={"NbS2":"#D55E00","2H-TaS2":"#0072B2","2H-TaSe2":"#56B4E9","1T-VSe2":"#E69F00","NbSe2":"#CC79A7"}
LAB={"NbS2":"2H-NbS$_2$","2H-TaS2":"2H-TaS$_2$","2H-TaSe2":"2H-TaSe$_2$","1T-VSe2":"1T-VSe$_2$","NbSe2":"2H-NbSe$_2$"}
order=["NbS2","2H-TaS2","2H-TaSe2","NbSe2","1T-VSe2"]
fig,ax=plt.subplots(figsize=(6.6,4.5))
for m in order:
    if m not in mats: continue
    pts=sorted(mats[m]); T=np.array([p[0] for p in pts])/1000; f=[p[1] for p in pts]
    ax.plot(T,f,"-o",color=OI[m],lw=2,ms=6,mec="white",mew=1.2,label=LAB[m])
ax.axhline(0,color="#333",lw=1,ls="--"); ax.text(4.4,.12,"CDW melted",fontsize=8,ha="right",color="#333")
ax.set_xlabel(r"electronic temperature $T_{\rm el}$ (10$^3$ K)"); ax.set_ylabel("soft-mode min freq (THz)")
ax.set_title("(E)-channel CDW melting across the TMD family (incl. NbSe$_2$)",fontsize=10)
ax.legend(frameon=False,fontsize=9,loc="lower right"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/smearing_kink/family_melting_Tel.png",dpi=150); print("saved melting fig")
