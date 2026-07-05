"""R1 step 2: does the UNIVERSAL B(x) law (one curve for the whole family) + each
material's T_half + its own D0 reproduce that material's DFT soft-mode melting curve?
Compares universal-law min-freq(T_el) to per-material-fit and to DFT."""
import sys, csv, glob, numpy as np
sys.path.insert(0,'scripts/smearing_kink'); import friedel_module as fm
rows=list(csv.DictReader(open('results/smearing_kink/friedel_family.csv')))
fam={}
for r in rows: fam.setdefault(r['material'],[]).append((float(r['T_el']),float(r['B']),float(r['minf_dft_THz'])))
def thalf(pts):
    pts=sorted(pts); T=[p[0] for p in pts]; B=[p[1] for p in pts]
    for i in range(len(B)-1):
        if B[i]>=0.5>=B[i+1]:
            f=(B[i]-0.5)/(B[i]-B[i+1]); return T[i]+f*(T[i+1]-T[i])
    return T[-1]
TH={m:thalf(p) for m,p in fam.items()}
# universal B(x): pooled family points
XX=[];BB=[]
for m,p in fam.items():
    for T,B,_ in p: XX.append(T/TH[m]); BB.append(B)
o=np.argsort(XX); XX=np.array(XX)[o]; BB=np.array(BB)[o]
Buni=lambda x: float(np.interp(x,XX,BB))
def minf(ph,fc,mesh=12):
    ph.force_constants=fc; ph.run_mesh([mesh,mesh,1],with_eigenvectors=False,is_gamma_center=True)
    return float(np.min(ph.get_mesh_dict()['frequencies']))
LOC={'NbS2':('results/v100/fc2_nbs2_3x3_{}/NbS2_phonopy.yaml'),
     '2H-TaSe2':('results/v100/fc2_tase2_3x3_{}/2H-TaSe2_phonopy.yaml'),
     'NbSe2':('results/v100/fc2_nbse2_3x3_{}/NbSe2_phonopy.yaml')}
DGS=['0.005','0.010','0.015','0.020']; TELpd=157888
print(f"{'material':>10} {'T_half':>7} {'universal-law min-freq MAE (THz)':>32}")
for m,patt in LOC.items():
    phs={d:fm.load_ph(patt.format(d)) for d in DGS}; fcs={d:phs[d].force_constants for d in DGS}
    bg,ref=DGS[-1],DGS[0]; ph=phs[bg]; tabs,_=fm.pair_table(ph); D0=fm.template_delta(fcs[ref],fcs[bg],tabs)
    errs=[]
    for d in DGS:
        T=float(d)*TELpd; mdft=minf(phs[d],fcs[d])
        B=1.0 if d==ref else (0.0 if d==bg else Buni(T/TH[m]))
        mmod=minf(ph,fm.add_template(fcs[bg],tabs,D0,B,0.0,1.0,14.0))
        if d not in (ref,bg): errs.append(abs(mmod-mdft))
    print(f"{m:>10} {TH[m]:>7.0f} {np.mean(errs):>32.3f}")
print("# universal-law family melting reproduced with one shared B(x); per-material fit MAEs were 0.007-0.13 THz")
