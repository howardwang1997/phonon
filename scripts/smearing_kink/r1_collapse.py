"""R1 step 1: does the family B(T_el) collapse onto a universal law under a
material-descriptor rescaling? Descriptor = T_half (T_el where B=0.5), a proxy for
CDW robustness. If B vs T_el/T_half collapses -> family-general conditioned term."""
import csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=list(csv.DictReader(open('results/smearing_kink/friedel_family.csv')))
mats={}
for r in rows: mats.setdefault(r['material'],[]).append((float(r['T_el']),float(r['B'])))
OI={'NbS2':'#D55E00','2H-TaS2':'#0072B2','2H-TaSe2':'#56B4E9','1T-VSe2':'#E69F00','NbSe2':'#CC79A7'}
def thalf(pts):  # T_el where B crosses 0.5
    pts=sorted(pts); T=[p[0] for p in pts]; B=[p[1] for p in pts]
    for i in range(len(B)-1):
        if B[i]>=0.5>=B[i+1]:
            f=(B[i]-0.5)/(B[i]-B[i+1]); return T[i]+f*(T[i+1]-T[i])
    return T[-1]
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,4.4))
spread_raw=[]; spread_scaled=[]
xg=np.linspace(0,2,50); allsc=[]
for m,pts in mats.items():
    pts=sorted(pts); T=np.array([p[0] for p in pts])/1000; B=[p[1] for p in pts]
    th=thalf(pts)/1000
    a1.plot(T,B,'-o',color=OI.get(m,'#888'),ms=6,label=f'{m} (T½={th*1000:.0f}K)')
    a2.plot(T/th,B,'-o',color=OI.get(m,'#888'),ms=6,label=m)
    allsc.append((T/th,B))
a1.set_title('raw B(T_el) — spread across family',fontsize=10); a1.set_xlabel(r'$T_{el}$ (10³K)'); a1.set_ylabel('B')
a2.set_title('rescaled B(T_el/T½) — collapse?',fontsize=10); a2.set_xlabel(r'$T_{el}/T_{1/2}$')
a2.axhline(0.5,ls=':',color='#999'); a2.axvline(1,ls=':',color='#999')
for a in (a1,a2): a.legend(frameon=False,fontsize=7.5); a.spines[['top','right']].set_visible(False)
# quantify collapse: std of B at fixed x-bins after rescaling
xb=np.linspace(0.3,1.7,8); resid=[]
for xc in xb:
    vals=[np.interp(xc,x,B) for x,B in allsc if x.min()<=xc<=x.max()]
    if len(vals)>=3: resid.append(np.std(vals))
print(f'# collapse residual (mean std of B across family after T/T-half rescaling) = {np.mean(resid):.3f}')
fig.tight_layout(); fig.savefig('results/smearing_kink/r1_B_collapse.png',dpi=150); print('saved r1_B_collapse.png')
