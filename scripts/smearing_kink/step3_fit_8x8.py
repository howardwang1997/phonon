"""STEP 3: fit the T_el-conditioned long-range term on the CONVERGED 8x8 data
(the prior fit used truncated 6x6). Backbone = most-smeared (dg0.08); template D0 =
sharpest (dg0.005). Fit B(T_el), kappa(T_el) so backbone + B*exp(-kappa*R)*D0
reproduces each smearing's Kohn kink. Reports held-out kink MAE vs 8x8 DFT."""
import sys, numpy as np
sys.path.insert(0,'scripts/smearing_kink'); import friedel_module as fm
DG=['0.005','0.01','0.02','0.04','0.08']; TEL={d:float(d)*157888 for d in DG}
Y='results/sc_conv/graphene_sc8_dg{}_phonopy.yaml'
phs={d:fm.load_ph(Y.format(d)) for d in DG}
fcs={d:phs[d].force_constants for d in DG}
ph=phs['0.08']; fc_bg=fcs['0.08']                       # backbone = most smeared
tabs,_=fm.pair_table(ph); D0=fm.template_delta(fcs['0.005'],fc_bg,tabs)  # D0 = sharpest
kappas=np.linspace(0,2.0,161); rmin,rmax=1.0,9.8
print(f"{'dg':>7} {'T_el':>6} {'B':>7} {'kappa':>7} {'kink_model':>10} {'kink_DFT':>9} {'err':>6}")
rows=[]
for d in DG:
    kdft=fm.kink_of(phs[d],fcs[d])[0]
    if d=='0.005': B,kap=1.0,0.0
    elif d=='0.08': B,kap=0.0,0.0
    else: B,kap,_=fm.fit_template_env(fcs[d],fc_bg,tabs,D0,rmin,rmax,kappas)
    fcm=fm.add_template(fc_bg,tabs,D0,B,kap,rmin,rmax)
    kmod=fm.kink_of(ph,fcm)[0]
    rows.append((d,TEL[d],B,kap,kmod,kdft))
    print(f"{d:>7} {TEL[d]:>6.0f} {B:>7.3f} {kap:>7.3f} {kmod:>10.2f} {kdft:>9.2f} {kmod-kdft:>+6.2f}")
heldout=[r for r in rows if r[0] not in ('0.005','0.08')]
mae=np.mean([abs(r[4]-r[5]) for r in heldout])
print(f"# held-out kink MAE (dg 0.01/0.02/0.04) = {mae:.2f} cm-1  (prior 6x6 fit MAE was 1.7 on truncated data)")

# save laws + figure
import csv
with open('results/smearing_kink/step3_longrange_8x8.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['dg','T_el','B','kappa','kink_model','kink_DFT'])
    for r in rows: w.writerow([r[0],f'{r[1]:.0f}',f'{r[2]:.4f}',f'{r[3]:.4f}',f'{r[4]:.3f}',f'{r[5]:.3f}'])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T=[r[1]/1000 for r in rows]; km=[r[4] for r in rows]; kd=[r[5] for r in rows]
fig,ax=plt.subplots(figsize=(6.2,4.3))
ax.plot(T,kd,'-o',color='#222',ms=7,lw=1.5,label='8×8 DFT (converged)',zorder=3)
ax.plot(T,km,'--s',color='#D55E00',ms=6,lw=2,label='backbone + 2-param long-range term',zorder=4)
ax.set_xlabel(r'electronic temperature $T_{\rm el}$ (10$^3$ K)'); ax.set_ylabel('Kohn kink at K (cm$^{-1}$)')
ax.set_title(f'STEP 3: T_el-conditioned long-range term vs converged 8×8 kink\nheld-out MAE = {mae:.2f} cm⁻¹ (was 1.7 on truncated 6×6)',fontsize=9.5)
ax.legend(frameon=False,fontsize=8.5); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig('results/smearing_kink/step3_kink_8x8.png',dpi=150); print('saved fig + csv')
