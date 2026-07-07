"""Probe: does the VSe2 v2 backbone know the CDW soft mode?
Compare backbone-bare min-freq vs DFT sharp (dg0.005) and melted (dg0.020)."""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))
sys.path.insert(0, str(ROOT / "src"))
import friedel_module as fm
from friedel_calc import fc2_from_calc

def minf(ph, fc, mesh=12):
    ph.force_constants = fc
    ph.run_mesh([mesh, mesh, 1], with_eigenvectors=False, is_gamma_center=True)
    return float(np.min(ph.get_mesh_dict()["frequencies"]))

Y = ROOT / "results" / "vq_family" / "vse2"
ph_dft_sharp = fm.load_ph(str(Y / "1T-VSe2_dg0.005.yaml"))
ph_dft_melt  = fm.load_ph(str(Y / "1T-VSe2_dg0.020.yaml"))
print(f"DFT sharp  (dg0.005) min-freq: {minf(ph_dft_sharp, ph_dft_sharp.force_constants):+.3f} THz")
print(f"DFT melted (dg0.020) min-freq: {minf(ph_dft_melt,  ph_dft_melt.force_constants):+.3f} THz")

# v2 backbone bare fc2 (on the melted-reference supercell, as deploy does)
from mace.calculators import MACECalculator
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = MACECalculator(model_paths=str(ROOT / "results" / "fam_backbone" / "1T-VSe2_v2" / "ft_1T-VSe2_v2_compiled.model"),
                  device=dev, default_dtype="float32")
_, fcm = fc2_from_calc(ph_dft_melt, m, subtract_ref=False)
print(f"v2 backbone BARE min-freq (on melted supercell): {minf(ph_dft_melt, fcm):+.3f} THz")
# also on the sharp supercell geometry (in case backbone soft-mode shows only there)
_, fcm2 = fc2_from_calc(ph_dft_sharp, m, subtract_ref=False)
print(f"v2 backbone BARE min-freq (on sharp supercell):  {minf(ph_dft_sharp, fcm2):+.3f} THz")
