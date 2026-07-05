"""R1 (a): leave-one-out FEW-SHOT family transfer. Since no descriptor predicts T_half,
pin it from ONE anchor smearing. For each held-out material: build universal B(x) from
the OTHER 4, fit B at one anchor (dg0.015) to invert -> T_half, then universal law
predicts the FULL soft-mode melting. Compares few-shot MAE to the full-curve fit."""
import sys, csv, numpy as np
sys.path.insert(0,'scripts/smearing_kink'); import friedel_module as fm
rows=list(csv.DictReader(open('results/smearing_kink/friedel_family.csv')))
fam={}
for r in rows: fam.setdefault(r['material'],[]).append((float(r['T_el']),float(r['B']),float(r['dg']),float(r['minf_dft_THz'])))
def thalf(pts):
    p=sorted(pts); T=[x[0] for x in p]; B=[x[1] for x in p]
    for i in range(len(B)-1):
        if B[i]>=0.5>=B[i+1]: f=(B[i]-0.5)/(B[i]-B[i+1]); return T[i]+f*(T[i+1]-T[i])
    return T[-1]
TH={m:thalf(p) for m,p in fam.items()}
def minf(ph,fc,mesh=12):
    ph.force_constants=fc; ph.run_mesh([mesh,mesh,1],with_eigenvectors=False,is_gamma_center=True)
    return float(np.min(ph.get_mesh_dict()['frequencies']))
LOC={'NbS2':'results/v100/fc2_nbs2_3x3_{}/NbS2_phonopy.yaml',
     '2H-TaSe2':'results/v100/fc2_tase2_3x3_{}/2H-TaSe2_phonopy.yaml',
     'NbSe2':'results/v100/fc2_nbse2_3x3_{}/NbSe2_phonopy.yaml'}
DGS=['0.005','0.010','0.015','0.020']; TELpd=157888
print(f"{'held-out':>10} {'T½_true':>8} {'T½_1shot':>9} {'fewshot MAE':>11} {'(step2 MAE)':>11}")
for held,patt in LOC.items():
    # universal B(x) from the OTHER 4
    XX,BB=[],[]
    for m,p in fam.items():
        if m==held: continue
        for T,B,dg,mf in p: XX.append(T/TH[m]); BB.append(B)
    XX,BB=np.array(XX),np.array(BB); o=np.argsort(XX); Xs,Bs=XX[o],BB[o]
    Buni=lambda x: float(np.interp(x,Xs,Bs))
    # inverse: x* where Buni=Banchor (sort by B ascending)
    ob=np.argsort(BB); Binv,Xinv=BB[ob],XX[ob]
    # anchor = held material's fitted B at dg0.015
    Banch=[B for T,B,dg,mf in fam[held] if abs(dg-0.015)<1e-6][0]; Tanch=0.015*TELpd
    xstar=float(np.interp(Banch,Binv,Xinv)); Th_fs=Tanch/xstar
    phs={d:fm.load_ph(patt.format(d)) for d in DGS}; fcs={d:phs[d].force_constants for d in DGS}
    bg,ref=DGS[-1],DGS[0]; ph=phs[bg]; tabs,_=fm.pair_table(ph); D0=fm.template_delta(fcs[ref],fcs[bg],tabs)
    errs=[]
    for d in DGS:
        if d in (ref,bg): continue
        T=float(d)*TELpd; B=Buni(T/Th_fs)
        mmod=minf(ph,fm.add_template(fcs[bg],tabs,D0,B,0.0,1.0,14.0)); mdft=minf(phs[d],fcs[d])
        errs.append(abs(mmod-mdft))
    s2={'NbS2':0.62,'2H-TaSe2':0.16,'NbSe2':0.33}[held]
    print(f"{held:>10} {TH[held]:>8.0f} {Th_fs:>9.0f} {np.mean(errs):>11.3f} {s2:>11.2f}")
print("# few-shot: T_half from ONE anchor smearing + universal B(x) from the other 4 materials")
