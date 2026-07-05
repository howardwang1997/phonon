"""Harvest the (L)-channel family screen: T0 triage + T1 SSCHA crossover + T2 TDEP anharmonicity.
Build the consolidated family (L)-table + figure (the (L)-half of the origin-adjudication,
MLIP-triage tier)."""
import csv, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OI={'NbS2':'#D55E00','2H-TaS2':'#0072B2','2H-TaSe2':'#56B4E9','1T-VSe2':'#E69F00','NbSe2':'#CC79A7','1T-TiSe2':'#009E73'}
LAB={'NbS2':'2H-NbS$_2$','2H-TaS2':'2H-TaS$_2$','2H-TaSe2':'2H-TaSe$_2$','1T-VSe2':'1T-VSe$_2$','NbSe2':'2H-NbSe$_2$','1T-TiSe2':'1T-TiSe$_2$'}
MAT=['1T-VSe2','NbS2','NbSe2','2H-TaS2','2H-TaSe2','1T-TiSe2']

def read_t2(n):
    f=glob.glob(f"results/h20/lchannel/{n}_tdep/td_*.csv")
    if not f: return (float('nan'),float('nan'),float('nan'))
    rows=list(csv.DictReader(open(f[0])))
    if not rows: return (float('nan'),float('nan'),float('nan'))
    r300=[r for r in rows if abs(float(r['T_K'])-300)<1]
    r=r300[0] if r300 else rows[-1]
    return float(r['anharm_frac_force']), float(r['minfreq_thz']), float(r['fc3_norm'])

def read_t1(n):
    f=f"results/h20/lchannel/{n}_sscha.csv"
    try:
        rows=list(csv.DictReader(open(f)))
        # SSCHA min-freq at highest T (does it heal?)
        mf=[float(r.get('sscha_minfreq_thz',r.get('minfreq_thz','nan'))) for r in rows if r.get('sscha_minfreq_thz') or r.get('minfreq_thz')]
        label_row=[r for r in rows if 'label' in {**r}]
        return rows[-1] if rows else None
    except Exception:
        return None

print(f"{'material':>10} {'T0-triage':>10} {'T2-anharm%':>11} {'T2-fc3norm':>11}")
T0={'1T-VSe2':'SOFT @M -0.96','NbS2':'marginal','NbSe2':'marginal','2H-TaS2':'marginal','2H-TaSe2':'marginal','1T-TiSe2':'marginal'}
rows_out=[]
for m in MAT:
    t2=read_t2(m)
    ah,mf,fc3=read_t2(m)
    if t2: ah,mf,fc3=t2
    print(f"{m:>10} {T0[m]:>10} {ah*100:>10.1f}% {fc3:>11.1f}")
    rows_out.append((m,T0[m],ah,mf,fc3))

# write table
with open('results/h20/lchannel/family_L_screen.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['material','T0_triage','T2_anharm_frac_force@300K','T2_minfreq_thz@300K','T2_fc3_norm'])
    for r in rows_out: w.writerow([r[0],r[1],f'{r[2]:.4f}',f'{r[3]:.3f}',f'{r[4]:.2f}'])

# figure: T2 anharmonicity ranking bar
order=sorted(MAT, key=lambda m: read_t2(m)[0] if read_t2(m) else 0)
fig,ax=plt.subplots(figsize=(6.4,4.2))
vals=[read_t2(m)[0]*100 for m in order]
bars=ax.bar([LAB[m] for m in order],vals,color=[OI[m] for m in order])
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.3,f'{v:.1f}%',ha='center',fontsize=8)
ax.set_ylabel('anharm. force fraction @ 300 K (TDEP)')
ax.set_title('(L)-channel TDEP anharmonicity ranking (foundation MACE-MP)\nhigher = more lattice-anharmonic; VSe$_2$ flagged by T0 triage',fontsize=9.5)
ax.spines[['top','right']].set_visible(False); plt.xticks(rotation=15,fontsize=8)
fig.tight_layout(); fig.savefig('results/h20/lchannel/family_L_tdep_anharm.png',dpi=150); print('saved family_L_tdep_anharm.png')
