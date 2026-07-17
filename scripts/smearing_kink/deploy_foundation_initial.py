"""Deploy initial MACE-MP-0 (no fine-tune) + healing Friedel — ablation baseline.
Shows whether the raw foundation captures the Kohn anomaly vs FC-distillation."""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts/smearing_kink")
import friedel_module as fm, numpy as np
from friedel_calc import FriedelCorrection, FriedelMACECalculator, fc2_from_calc

FD = "results/graphene_kohn_fd"
DGS = ["0.01", "0.015", "0.02", "0.03", "0.04", "0.06", "0.08", "0.10", "0.14"]
REF = "0.01"

ph = fm.load_ph(FD + "/graphene_sc6_dg0.08_phonopy.yaml")
fc_dft = {dg: fm.load_ph(FD + f"/graphene_sc6_dg{dg}_phonopy.yaml").force_constants for dg in DGS}
hl = json.load(open(FD + "/healing_law.json"))

from mace.calculators import mace_mp
mace = mace_mp(model="small", device="cuda", default_dtype="float32")
from phonon_accel.phonons import phonopy_to_ase
ref_atoms = phonopy_to_ase(ph.supercell)
_, fc_bb = fc2_from_calc(ph, mace, distance=0.03, subtract_ref=True)
corr = FriedelCorrection(ph, fc_dft["0.08"], fc_dft[REF],
    B_law=lambda T: hl["B0"] * max(0.0, 1.0 - (T / hl["Tstar"]) ** hl["p"]) ** hl["q"],
    kappa_law=lambda T: hl["kappa"])
kB = fm.kink_of(ph, fc_bb)[0]

print("# INITIAL foundation (MACE-MP-0, no FT) + healing Friedel:")
print(f"{'dg':>6} {'T_el':>7} {'DFT':>7} {'found':>8}")
err = 0; n = 0
for dg in DGS:
    T = float(dg) * 157887
    dK = fm.kink_of(ph, fc_dft[dg])[0]
    if dg == REF:
        mK = dK
    else:
        calc = FriedelMACECalculator(mace, ref_atoms, corr, T)
        _, fc_t = fc2_from_calc(ph, calc, distance=0.03, subtract_ref=True)
        mK = fm.kink_of(ph, fc_t)[0]
    err += abs(mK - dK); n += 1
    print(f"{dg:>6} {T:>7.0f} {dK:>7.2f} {mK:>8.2f}")
print(f"# MAE kink_K (foundation no-FT vs DFT) = {err/n:.2f}  backbone_kink={kB:.2f}")
print("# compare: FC-distillation reuse=0.20 retrain=0.19")
