"""Extra figures for the smearing-phonon-via-MLIP weekly:
(A) deployment MAE per material (bar), (B) 'smearing-blind backbone vs +long-range-term'
for a representative CDW material (NbS2) — why the term is needed."""
import csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=list(csv.DictReader(open('results/smearing_kink/family_deploy_mlip.csv')))
mats={}
for r in rows: mats.setdefault(r['material'],[]).append((float(r['T_el']),float(r['MLIP_term']),float(r['DFT']),float(r['MAE'])))
OI={'NbS2':'#D55E00','2H-TaS2':'#0072B2','2H-TaSe2':'#56B4E9','1T-VSe2':'#E69F00','NbSe2':'#CC79A7'}
LAB={'NbS2':'2H-NbS$_2$','2H-TaS2':'2H-TaS$_2$','2H-TaSe2':'2H-TaSe$_2$','1T-VSe2':'1T-VSe$_2$','NbSe2':'2H-NbSe$_2$'}

# (A) MAE bar chart
order=sorted(mats, key=lambda m: mats[m][0][3])
mae=[mats[m][0][3] for m in order]
fig,ax=plt.subplots(figsize=(6.0,4.0))
bars=ax.bar([LAB[m] for m in order],mae,color=[OI[m] for m in order])
for b,v in zip(bars,mae): ax.text(b.get_x()+b.get_width()/2,v+0.01,f'{v:.2f}',ha='center',fontsize=8)
ax.axhline(0.1,ls=':',color='#888',lw=1); ax.text(0.1,0.11,'0.1 THz',fontsize=7,color='#888')
ax.set_ylabel('deployment min-freq MAE (THz)')
ax.set_title('Smearing-phonon-via-MLIP: deployment accuracy across CDW family\n(graphene kink: 0.31 cm⁻¹, shown separately)',fontsize=9.5)
ax.spines[['top','right']].set_visible(False); plt.xticks(rotation=15,fontsize=8)
fig.tight_layout(); fig.savefig('results/smearing_kink/deploy_mae_bar.png',dpi=150); print('saved deploy_mae_bar.png')

# (B) smearing-blind backbone vs +term (NbS2)
m='NbS2'; p=sorted(mats[m]); T=np.array([x[0] for x in p])/1000; mlip=[x[1] for x in p]; dft=[x[2] for x in p]
bb=-0.677  # MACE backbone alone (one fc2 -> constant vs T_el)
fig,ax=plt.subplots(figsize=(6.0,4.2))
ax.axhline(bb,color='#999',lw=2,ls=':',label='MLIP backbone alone (smearing-blind)')
ax.plot(T,dft,'-',color=OI[m],lw=1.6,label='DFT')
ax.plot(T,mlip,'o',color=OI[m],ms=7,mec='white',label='MLIP backbone + long-range term')
ax.axhline(0,color='#333',lw=.7,ls='--')
ax.annotate('term adds the\nT$_{el}$-dependence',xy=(1.0,-4.5),xytext=(1.6,-3.5),fontsize=8.5,color='#555',
            arrowprops=dict(arrowstyle='->',color='#555'))
ax.set_xlabel(r'electronic temperature $T_{\rm el}$ (10$^3$ K)'); ax.set_ylabel('soft-mode min freq (THz)')
ax.set_title('Why the long-range term: 2H-NbS$_2$\nshort-range MLIP alone is smearing-blind; term restores it',fontsize=9.5)
ax.legend(frameon=False,fontsize=8.3,loc='lower right'); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig('results/smearing_kink/deploy_why_term.png',dpi=150); print('saved deploy_why_term.png')
