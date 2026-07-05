"""Unified 'smearing-dependent phonon spectrum via MLIP' family figure:
soft-mode min-freq(T_el), MLIP-backbone+long-range-term (dots) vs DFT (lines)."""
import csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=list(csv.DictReader(open('results/smearing_kink/family_deploy_mlip.csv')))
mats={}
for r in rows: mats.setdefault(r['material'],[]).append((float(r['T_el']),float(r['MLIP_term']),float(r['DFT']),float(r['MAE'])))
OI={'NbS2':'#D55E00','2H-TaS2':'#0072B2','2H-TaSe2':'#56B4E9','1T-VSe2':'#E69F00','NbSe2':'#CC79A7','1T-TiSe2':'#009E73'}
LAB={'NbS2':'2H-NbS$_2$','2H-TaS2':'2H-TaS$_2$','2H-TaSe2':'2H-TaSe$_2$','1T-VSe2':'1T-VSe$_2$','NbSe2':'2H-NbSe$_2$','1T-TiSe2':'1T-TiSe$_2$'}
fig,ax=plt.subplots(figsize=(6.8,4.7))
for m,p in mats.items():
    p=sorted(p); T=np.array([x[0] for x in p])/1000; mlip=[x[1] for x in p]; dft=[x[2] for x in p]; mae=p[0][3]
    ax.plot(T,dft,'-',color=OI[m],lw=1.5,alpha=.85)
    ax.plot(T,mlip,'o',color=OI[m],ms=6,mec='white',mew=1,label=f'{LAB[m]} (MAE {mae:.2f})')
ax.axhline(0,color='#333',lw=.8,ls='--'); ax.text(3.0,.08,'melted',fontsize=7.5,color='#333')
ax.set_xlabel(r'electronic temperature $T_{\rm el}$ (10$^3$ K)'); ax.set_ylabel('soft-mode min freq (THz)')
ax.set_title('Smearing-dependent phonon spectrum via MLIP: distilled backbone +\nlong-range term (dots) vs DFT (lines) across the CDW family',fontsize=9.5)
ax.legend(frameon=False,fontsize=8,loc='lower right'); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig('results/smearing_kink/family_deploy_mlip.png',dpi=150); print('saved')
