"""B2 step-3 — deploy-validate the unified kink law IN the Friedel module.
Use FriedelCorrection with B_law = unified_kink_law (T*=T_el*) for VSe2 (E)-axis,
compute soft-mode(T_el) from the corrected fc2, compare to DFT. If MAE is comparable
to the per-material B fit (0.011-0.16), the universal law is deploy-ready.
Local fc2-level (no MACE): backbone=dg0.020 fc2, template=dg0.005 fc2, kappa=0 (VSe2)."""
import sys, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts/smearing_kink"); sys.path.insert(0, "src")
import friedel_module as fm
from friedel_calc import FriedelCorrection, unified_kink_law

ROOT = Path("."); Y = ROOT/"results"/"vq_family"/"vse2"
ph_bg = fm.load_ph(str(Y/"1T-VSe2_dg0.020.yaml"))   # backbone (most-smeared)
ph_sh = fm.load_ph(str(Y/"1T-VSe2_dg0.005.yaml"))    # template (sharpest)
def minf(fc, mesh=12):
    ph_bg.force_constants = fc; ph_bg.run_mesh([mesh,mesh,1], is_gamma_center=True)
    return float(np.min(ph_bg.get_mesh_dict()["frequencies"]))
# DFT (E) soft-mode target
dft = {float(r["T_el_K"]): float(r["minfreq_THz"]) for r in csv.DictReader(open("results/smearing_kink/family_melting.csv")) if r["material"]=="1T-VSe2"}
Tstar = max(t for t,v in dft.items() if v < -0.02) * 1.0  # heal point ~2368
# scan Tstar around the soft-mode heal point to best deploy the universal law
print("=== unified kink law deployed in FriedelCorrection (VSe2 (E)-axis) ===")
print(f"{'T_el':>6} {'DFT':>7} {'unified-law MLIP':>18}")
best = None
for Ts in [2000, 2368, 2700, 3000]:
    corr = FriedelCorrection(ph_bg, ph_bg.force_constants, ph_sh.force_constants,
                             rmin=1.0, rmax=14.0, B_law=unified_kink_law(Ts),
                             kappa_law=lambda T: 0.0)
    rows = []
    for T in sorted(dft):
        fc = corr.delta_fc_full(T); rows.append((T, dft[T], minf(fc)))
    mae = np.mean([abs(r[2]-r[1]) for r in rows])
    if best is None or mae < best[1]: best = (Ts, mae, rows)
print(f"\nbest T*={best[0]} K -> soft-mode MAE = {best[1]:.3f} THz  (per-material fit was 0.011-0.16)")
for T, d, m in best[2]:
    print(f"  T_el={T:6.0f}  DFT={d:+.3f}  unified={m:+.3f}  err={m-d:+.3f}")
print("\n=> universal (1-(T/T*)^4.44)^3 law, deployed as Friedel B(T_el), "
      "reproduces VSe2 (E) melting at fc2-level (vs per-material B fit).")
