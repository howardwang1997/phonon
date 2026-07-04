import sys, numpy as np; sys.path.insert(0,'scripts/smearing_kink'); sys.path.insert(0,'src')
import friedel_module as fm, td_common as tdc
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
CM=33.35641
def bands(ph, fc):
    ph.force_constants=fc
    q,conn,lab=tdc.make_band_path("MGKM",npoints=60)
    ph.run_band_structure(q,path_connections=conn,labels=lab,with_eigenvectors=False)
    d=ph.get_band_structure_dict()
    dist=np.concatenate([np.asarray(x) for x in d["distances"]])
    frq=np.concatenate([np.asarray(x) for x in d["frequencies"]],axis=0)
    seg=[np.asarray(x) for x in d["distances"]]; lp=[seg[0][0]]+[s[-1] for s in seg]
    return dist, frq*CM, lp
phbg=fm.load_ph('results/sc_conv/graphene_sc8_dg0.08_phonopy.yaml')
fig,ax=plt.subplots(figsize=(6.6,4.6))
for dgt,T,c in [('0.005',789,'#0072B2'),('0.02',3158,'#D55E00')]:
    phD=fm.load_ph(f'results/sc_conv/graphene_sc8_dg{dgt}_phonopy.yaml')
    dD,fD,lp=bands(phD,phD.force_constants.copy())
    fcdep=np.load(f'results/smearing_kink/deploy_fc2_T{T}.npz')['fc2']
    dM,fM,_=bands(phbg,fcdep)
    ax.plot(dD,fD,color=c,lw=1.3,alpha=.85,label=f'DFT  T_el={T}K')
    ax.plot(dM,fM,color=c,lw=0,marker='o',ms=1.8,alpha=.5,label=f'MACE+term  {T}K')
for x in lp: ax.axvline(x,color='#ddd',lw=.6)
ax.set_xticks(lp); ax.set_xticklabels(['M','Γ','K','M']); ax.set_ylabel('frequency (cm$^{-1}$)')
ax.set_title('STEP 4 full dispersion: MACE+long-range term (dots) vs DFT (lines)',fontsize=9.5)
ax.legend(frameon=False,fontsize=7.5,ncol=2,loc='lower center'); ax.set_ylim(0,1750)
ax.spines[['top','right']].set_visible(False); fig.tight_layout()
fig.savefig('results/smearing_kink/step4_dispersion.png',dpi=150); print('SAVED dispersion')
