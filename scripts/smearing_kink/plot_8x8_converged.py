import sys, numpy as np; sys.path.insert(0,'scripts/smearing_kink')
import friedel_module as fm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def truncate(fc, tabs, rc):
    fc2=fc.copy()
    for p,t in enumerate(tabs):
        s,R=t['s'],t['R']; far=R>rc
        if np.any(far): fc2[p,t['s0']]+=fc2[p,s[far]].sum(axis=0); fc2[p,s[far]]=0.0
    return fc2
rc=np.array([4,5,6,7,8,9,9.8]); OI={'0.005':'#0072B2','0.01':'#009E73','0.02':'#E69F00'}
fig,ax=plt.subplots(figsize=(6.2,4.3))
for d in ['0.005','0.01','0.02']:
    ph=fm.load_ph(f'results/sc_conv/graphene_sc8_dg{d}_phonopy.yaml'); tabs,_=fm.pair_table(ph)
    ks=[fm.kink_of(ph,truncate(ph.force_constants,tabs,r)[0] if False else truncate(ph.force_constants,tabs,r))[0] for r in rc]
    ax.plot(rc,ks,'-o',color=OI[d],lw=2,ms=5,label=f'T_el={float(d)*157888:.0f}K')
ax.axhspan(5.7,6.6,color='#ccc',alpha=.3,zorder=0); ax.text(4.3,6.1,'converged ~6 cm⁻¹',fontsize=8,color='#555')
ax.set_xlabel('fc$_2$ real-space cutoff R$_{cut}$ (Å)'); ax.set_ylabel('Kohn kink at K (cm$^{-1}$)')
ax.set_title('8×8 converges: kink plateaus ~9 Å at ~6 cm⁻¹\n(6×6 gave spurious 15–18 — truncation artifact)',fontsize=9.5)
ax.legend(frameon=False,fontsize=8.5); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig('results/smearing_kink/kink_8x8_converged.png',dpi=150); print('saved')
