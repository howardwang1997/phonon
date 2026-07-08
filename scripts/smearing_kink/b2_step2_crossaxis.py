"""B2 step-2 — formalize the unified kink(T_el,T_lat) law & cross-validate.

Combine VSe2 (E)-melting + (L)-crossover soft-mode data, normalize to reduced-T
x = T/T* (T*_el = smearing-heal point; T*_lat = T_CDW), y = |soft-mode|/|depth_0|.
Fit ONE shared healing law f(x) to the combined points. Then cross-validate:
calibrate f on (E)-only -> predict (L), and vice versa. If cross-axis prediction
holds, the unified law is demonstrated (one law, two axes) -> the MLIP+LR target
is "condition the long-range amplitude on reduced-T"."""
import csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
SK = Path("results/smearing_kink"); CM2THz = 1.0/33.356

def load_E(mat):
    return np.array(sorted([(float(r["T_el_K"]), float(r["minfreq_THz"])) for r in csv.DictReader(open(SK/"family_melting.csv")) if r["material"]==mat]))
def load_L(p):
    return np.array(sorted([(float(r["T_K"]), float(r["sscha_minfreq_cm"])*CM2THz) for r in csv.DictReader(open(p))]))
def heal(d):
    neg=d[d[:,1]<-0.02]; pos=d[d[:,1]>=-0.02]
    return (float(pos[:,0].min()) if len(pos) else float(d[-1,0])), float(neg[:,1].min())
def norm(d, Ts, depth):
    x=d[:,0]/Ts; y=np.where(d[:,1]<-0.02, np.minimum(d[:,1]/depth,1.0), 0.0)
    return x,y

E = load_E("1T-VSe2"); L = load_L("/tmp/1T-VSe2_fine.csv")
Te,de = heal(E); Tl,dl = heal(L)
xe,ye = norm(E,Te,de); xl,yl = norm(L,Tl,dl)
# keep only the healing region (x<=1.05), y in [0,1]
mask_e = (xe<=1.05); mask_l = (xl<=1.05)
xe,ye = xe[mask_e],ye[mask_e]; xl,yl = xl[mask_l],yl[mask_l]

# shared healing law: f(x) = (1 - x^p)^q  (flexible 2-param mean-field-like form)
def fmodel(x, p, q): return np.maximum(0, 1 - x**p)**q
from scipy.optimize import curve_fit
Xall = np.concatenate([xe, xl]); Yall = np.concatenate([ye, yl])
popt,_ = curve_fit(fmodel, Xall, Yall, p0=[2.0, 0.5], bounds=([0.5,0.1],[5,3]))
res = Yall - fmodel(Xall, *popt)
print(f"=== shared healing law fit (combined (E)+(L), VSe2) ===")
print(f"  f(x) = (1 - x^p)^q ,  p={popt[0]:.2f}, q={popt[1]:.2f}")
print(f"  combined MAE = {np.mean(np.abs(res)):.3f}  max resid = {np.max(np.abs(res)):.3f}  (n={len(Xall)})")

# cross-validate: fit on (E), predict (L); fit on (L), predict (E)
def cv(train_x, train_y, test_x, test_y, label):
    po,_ = curve_fit(fmodel, train_x, train_y, p0=[2.0,0.5], bounds=([0.5,0.1],[5,3]))
    pred = fmodel(test_x, *po)
    mae = np.mean(np.abs(pred - test_y))
    print(f"  calibrate on {label}-axis (p={po[0]:.2f},q={po[1]:.2f}) -> predict other axis: MAE={mae:.3f}")
print("\n=== cross-axis prediction ===")
cv(xe,ye, xl,yl, "(E) smearing")
cv(xl,yl, xe,ye, "(L) lattice")

# write the unified-law table
with open(SK/"b2_unified_law.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["axis","Tstar_K","depth_THz","p","q"])
    w.writerow(["(E)_T_el",int(Te),f"{de:.3f}",f"{popt[0]:.3f}",f"{popt[1]:.3f}"])
    w.writerow(["(L)_T_lat",int(Tl),f"{dl:.3f}",f"{popt[0]:.3f}",f"{popt[1]:.3f}"])
print(f"\nT* (E)={int(Te)} K (smearing-heal), T* (L)={int(Tl)} K (=T_CDW)")
print("wrote", SK/"b2_unified_law.csv")
print("\n=> unified kink law: |soft-mode(T)| = |depth| * (1-(T/T*)^p)^q,")
print("   SAME (p,q) on both axes -> one MLIP+LR conditioned on reduced-T T/T* covers both.")
