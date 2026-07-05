"""R1 step 3: can a COMPUTABLE descriptor predict T_half (the family-collapse scale)?
Test T_half vs: chi(q_CDW) nesting strength, |soft-mode depth at dg0.005|, q_CDW.
5 CDW materials -> report Pearson r (thin stats, look for a clear winner)."""
import csv, numpy as np, glob
rows=list(csv.DictReader(open('results/smearing_kink/friedel_family.csv')))
fam={}
for r in rows: fam.setdefault(r['material'],[]).append((float(r['T_el']),float(r['B']),float(r['dg']),float(r['minf_dft_THz'])))
def thalf(pts):
    p=sorted(pts); T=[x[0] for x in p]; B=[x[1] for x in p]
    for i in range(len(B)-1):
        if B[i]>=0.5>=B[i+1]: f=(B[i]-0.5)/(B[i]-B[i+1]); return T[i]+f*(T[i+1]-T[i])
    return T[-1]
QCDW={'1T-VSe2':0.25,'2H-TaS2':1/3,'2H-TaSe2':1/3,'NbS2':1/3,'NbSe2':1/3}
NPZ={'1T-VSe2':'1T-VSe2','2H-TaS2':'2H-TaS2','2H-TaSe2':'2H-TaSe2','NbS2':'NbS2','NbSe2':'NbSe2'}
def xi_qcdw(m):
    f=glob.glob(f'results/v100/chi_q/{NPZ[m]}/*_nesting.npz')
    if not f: return np.nan
    d=np.load(f[0],allow_pickle=True); xi=d['xi']; nk=int(d['nk'])
    return float(xi[round(QCDW[m]*nk),0])
data=[]
for m,p in fam.items():
    th=thalf(p); minf005=min(x[3] for x in p if abs(x[2]-0.005)<1e-6)  # deepest (sharpest)
    data.append((m,th,xi_qcdw(m),abs(minf005),QCDW[m]))
print(f"{'material':>10} {'T_half':>7} {'chi(qCDW)':>9} {'|minf.005|':>10} {'q_CDW':>6}")
for m,th,xi,mf,q in data: print(f"{m:>10} {th:>7.0f} {xi:>9.3f} {mf:>10.2f} {q:>6.3f}")
T=np.array([d[1] for d in data])
for name,idx in [('chi(q_CDW)',2),('|minf(0.005)|',3),('q_CDW',4)]:
    x=np.array([d[idx] for d in data]); ok=~np.isnan(x)
    r=np.corrcoef(x[ok],T[ok])[0,1]
    print(f"# T_half vs {name:>14}: Pearson r = {r:+.2f}")

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
labs=[d[0] for d in data]
fig,axs=plt.subplots(1,3,figsize=(11,3.6))
for ax,(name,idx) in zip(axs,[('chi(q_CDW)',2),('|soft-mode depth| (THz)',3),('q_CDW',4)]):
    x=[d[idx] for d in data]
    ax.scatter(x,T,s=60,color='#0072B2',zorder=3)
    for xi,ti,l in zip(x,T,labs): ax.annotate(l,(xi,ti),fontsize=6.5,xytext=(3,3),textcoords='offset points')
    r=np.corrcoef(x,T)[0,1]
    ax.set_xlabel(name); ax.set_ylabel('T½ (K)'); ax.set_title(f'r={r:+.2f}',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('R1 step 3: no simple descriptor predicts T½ (all |r|<0.5) — needs electronic energy scale or few-shot',fontsize=9.5)
fig.tight_layout(); fig.savefig('results/smearing_kink/r1_descriptor_scatter.png',dpi=150); print('saved scatter')
