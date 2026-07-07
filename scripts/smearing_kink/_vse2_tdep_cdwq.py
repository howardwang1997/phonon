"""Re-read B1 VSe2 TDEP npz at the CDW q (Γ-M midpoint = (1/4,1/4)) to get the
(L) soft-mode / kink(T_lat). The csv readout only had Γ/K; the CDW soft mode lives
between them. Also locates the globally-softest q per T."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
d = np.load('/tmp/td_1T-VSe2_tdep.npz', allow_pickle=True)
temps = d['temperatures']
labels = list(d['labels']); lpos = d['label_positions']
print("labels:", [(l, round(float(p),2)) for l,p in zip(labels,lpos)])
# Γ-M segment: find M position
gm_end = float(lpos[1])  # M
gm_mid = gm_end/2.0      # CDW q ~ (1/4,1/4) midpoint
dist = d['T50_dist']
i_mid = int(np.argmin(np.abs(dist - gm_mid)))
print(f"Γ-M midpoint (CDW q≈1/4) at dist index {i_mid}/{len(dist)} (dist={dist[i_mid]:.2f})\n")
print(f"{'T(K)':>5} {'min_freq_path':>14} {'@CDW_q(¼,¼)':>13} {'softest_q_dist':>15}")
rows=[]
for T in temps:
    f = d[f'T{int(T)}_freq']  # (601,9) in THz
    imin = int(np.unravel_index(np.argmin(f), f.shape)[0])
    mn = float(f[imin].min()); cdw = float(f[i_mid].min())
    rows.append((T, mn, cdw, dist[imin]))
    print(f"{int(T):>5} {mn:>14.3f} {cdw:>13.3f} {dist[imin]:>15.2f}")
# does the CDW soft mode harden with T (the (L) kink signature)?
import numpy as np
cdw_T = [float(d[f'T{int(T)}_freq'][i_mid].min()) for T in temps]
print(f"\nCDW-q soft mode vs T: {dict(zip([int(t) for t in temps], [round(x,3) for x in cdw_T]))}")
g = np.polyfit(temps, cdw_T, 1)[0]
print(f"linear trend (THz/K): {g*1000:.3f} mTHz/K  ({'hardens↑' if g>0 else 'softens↓'} with T)")
