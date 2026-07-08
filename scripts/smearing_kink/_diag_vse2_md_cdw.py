"""Diagnose B1: why does VSe2 MD-TDEP give a stable 1.9 THz CDW mode (no T_crossover)
when SSCHA found -412 cm-1 @ 50K? Two checks:
1. Path-P model BARE fc2 soft mode on the SYMMETRIC 4x4 -> is the instability even captured?
2. Short 300K MD from symmetric start -> CDW order param (rms disp) trajectory; does it collapse?
If bare soft mode is imaginary AND MD rms jumps -> MD collapses to CDW-ordered (model over-stabilizes
the CDW) => B1 (L)-kink data invalid; fix = start/constrain to symmetric, or use SSCHA fc2(T)."""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts/smearing_kink"); sys.path.insert(0, "src")
import friedel_module as fm
from phonon_accel.phonons import phonopy_to_ase
from friedel_calc import fc2_from_calc

ROOT = Path(".")
ph = fm.load_ph("results/vq_family/vse2/1T-VSe2_dg0.020.yaml")
def minf(fc, mesh=12):
    ph.force_constants = fc; ph.run_mesh([mesh,mesh,1], is_gamma_center=True)
    return float(np.min(ph.get_mesh_dict()["frequencies"]))

from mace.calculators import MACECalculator
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = MACECalculator(model_paths="results/finetune_path_p_1T-VSe2/ft.model", device=dev, default_dtype="float32")

# --- Step 1: bare soft mode on symmetric 4x4 ---
_, fc2 = fc2_from_calc(ph, m, subtract_ref=False)
bare = minf(fc2)
print(f"STEP 1  Path-P bare soft mode on SYMMETRIC 4x4 = {bare:+.3f} THz")
print(f"        ({'IMAGINARY -> instability captured' if bare < -0.05 else 'stable -> NO instability in model'})")

# --- Step 2: short 300K MD from symmetric, track rms displacement (CDW collapse proxy) ---
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
atoms = phonopy_to_ase(ph.supercell); atoms.calc = m
x0 = atoms.positions.copy()
MaxwellBoltzmannDistribution(atoms, 300*units.kB)
dyn = Langevin(atoms, 1*units.fs, friction=0.05, temperature_K=300)
rms=[]
for step in range(60):
    dyn.run(5)
    disp = atoms.positions - x0; disp -= disp.mean(axis=0)
    rms.append(float(np.sqrt((disp**2).sum(axis=1).mean())))
rms = np.array(rms)
print(f"\nSTEP 2  300K MD rms-displacement (A), 60 pts x 5fs = 300fs:")
print(f"        first 8: {np.round(rms[:8],3)}")
print(f"        last  8: {np.round(rms[-8:],3)}")
print(f"        mean first half {rms[:30].mean():.3f} A | mean last half {rms[30:].mean():.3f} A")
jump = rms[30:].mean() - rms[:30].mean()
print(f"        drift (last-first half): {jump:+.3f} A  ({'COLLAPSE to CDW-ordered' if jump > 0.04 else 'stays symmetric (thermal only)'})")
