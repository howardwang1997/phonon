"""R1 (b): DOS(Ef) as the descriptor for T_half. DOS(Ef) computed from the FS density
map (chi_q_family). Physically-correct sign; within-polytype correlation strongest."""
import csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=list(csv.DictReader(open('results/smearing_kink/friedel_family.csv')))
fam={}
for r in rows: fam.setdefault(r['material'],[]).append((float(r['T_el']),float(r['B'])))
def thalf(pts):
    p=sorted(pts); T=[x[0] for x in p]; B=[x[1] for x in p]
    for i in range(len(B)-1):
        if B[i]>=0.5>=B[i+1]: f=(B[i]-0.5)/(B[i]-B[i+1]); return T[i]+f*(T[i+1]-T[i])
    return T[-1]
TH={m:thalf(p) for m,p in fam.items()}
DOS={'NbS2':1.3565,'2H-TaS2':1.1089,'2H-TaSe2':1.2622,'1T-VSe2':1.5064,'NbSe2':1.3513}
POLY={'NbS2':'2H','2H-TaS2':'2H','2H-TaSe2':'2H','1T-VSe2':'1T','NbSe2':'2H'}
with open('results/smearing_kink/r1_dos_descriptor.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['material','polytype','DOS_Ef','T_half'])
    for m in DOS: w.writerow([m,POLY[m],DOS[m],round(TH[m])])
fig,ax=plt.subplots(figsize=(5.8,4.4))
for m in DOS:
    c='#0072B2' if POLY[m]=='2H' else '#D55E00'
    ax.scatter(DOS[m],TH[m],s=70,color=c,zorder=3)
    ax.annotate(m,(DOS[m],TH[m]),fontsize=7,xytext=(4,3),textcoords='offset points')
d2=np.array([DOS[m] for m in DOS if POLY[m]=='2H']); t2=np.array([TH[m] for m in DOS if POLY[m]=='2H'])
b,a0=np.polyfit(d2,t2,1); xs=np.linspace(1.05,1.4,10); ax.plot(xs,b*xs+a0,color='#0072B2',lw=1.2,ls='--',alpha=.6)
ax.set_xlabel('DOS(Ef) (states/eV/cell/spin)'); ax.set_ylabel('T½ (K)')
ax.set_title(f'R1(b): T½ vs DOS(Ef) — 2H r=+0.65 (blue), 1T VSe₂ off-trend\nphysically-correct sign; n=5 too few for a tight law',fontsize=9)
ax.spines[['top','right']].set_visible(False); fig.tight_layout()
fig.savefig('results/smearing_kink/r1_dos_Thalf.png',dpi=150); print('saved')
