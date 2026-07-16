"""Fit a HEALING B(T_el) law (the project's B2 unified kink law) to the graphene Kohn
kink melt on the DFT backbone, to capture the SHARP melt ("突变") that a smooth
kappa(T) power-law misses.

B(T_el) = B0 * max(0, 1-(T/T*)^p)^q   -> amplitude heals sharply to 0 at the melt T*.
kappa held at a small constant. Invert kink vs B (binary search) per T, then fit (B0,T*,p,q).
    conda run -n phonon python scripts/smearing_kink/fit_healing_law.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json
sys.path.insert(0, "scripts/smearing_kink")
import friedel_module as fm
import numpy as np
from scipy.optimize import least_squares

FD = "results/graphene_kohn_fd"
DGS = ["0.01", "0.015", "0.02", "0.03", "0.04", "0.06", "0.08", "0.10", "0.14"]
BG, REF = "0.08", "0.01"
KAP = 0.05   # const mild radial damping; the SHARP melt comes from B(T) healing

phs = {dg: fm.load_ph(f"{FD}/graphene_sc6_dg{dg}_phonopy.yaml") for dg in DGS}
fcs = {dg: phs[dg].force_constants for dg in DGS}
ph = phs[BG]; tabs, _ = fm.pair_table(ph)
D0 = fm.template_delta(fcs[REF], fcs[BG], tabs)
DFT = {dg: fm.kink_of(phs[dg], fcs[dg])[0] for dg in DGS}


def kink_of_B(B):
    fc = fm.add_template(fcs[BG], tabs, D0, B, KAP, 1.0, 12.0)
    return fm.kink_of(ph, fc)[0]


def B_for_kink(target):
    lo, hi = 0.0, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if kink_of_B(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


fit_dgs = [dg for dg in DGS if dg != BG]
print("# B(T) needed to match each DFT kink (kappa const):")
Bs = {}
for dg in fit_dgs:
    Bs[dg] = B_for_kink(DFT[dg])
    print(f"  dg{dg} T={float(dg)*157887:>6.0f}  DFT_kink={DFT[dg]:6.2f}  -> B={Bs[dg]:5.3f}")

Tarr = np.array([float(dg) * 157887 for dg in fit_dgs])
Barr = np.array([Bs[dg] for dg in fit_dgs])


def heal(T, B0, Tstar, p, q):
    return B0 * np.maximum(0.0, 1.0 - (T / Tstar) ** p) ** q


sol = least_squares(lambda x: heal(Tarr, *x) - Barr, x0=[1.2, 12000, 3, 2],
                    bounds=([0.3, 4000, 1.0, 0.5], [4.0, 25000, 8.0, 5.0]))
B0, Tstar, p, q = sol.x
print(f"\n# HEALING law: B0={B0:.3f}, T*={Tstar:.0f} K, p={p:.2f}, q={q:.2f}, kappa={KAP}")

print(f"\n{'dg':>6}{'T':>7}{'DFT':>8}{'healing':>9}")
err = 0.0
for dg in DGS:
    T = float(dg) * 157887
    k = 0.0 if dg == BG else kink_of_B(float(heal(T, B0, Tstar, p, q)))
    if dg != BG:
        err += abs(k - DFT[dg])
    print(f"{dg:>6}{T:>7.0f}{DFT[dg]:>8.2f}{k:>9.2f}")
print(f"# MAE kink_K (healing-B law vs DFT) = {err/len(fit_dgs):.2f} cm^-1")

json.dump({"B0": B0, "Tstar": Tstar, "p": p, "q": q, "kappa": KAP},
          open(FD + "/healing_law.json", "w"), indent=2)
print("# wrote healing_law.json")
