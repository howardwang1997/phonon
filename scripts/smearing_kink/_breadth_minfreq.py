"""Quick (E)-breadth contrast: min-freq(T_el) for MoS2 (gapped) vs 1T-TaS2/1T-TiS2 (metallic).
Reads the 12 Box-A fc2 phonopy yamls. MoS2 should be FLAT (no Fermi surface -> no smearing dep);
metallic should melt. No MLIP needed (pure DFT fc2 readout)."""
import sys, glob, re, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import yaml
import phonopy

SK = Path("results/v100")
TEL = 157888.0
def min_freq(yaml_path, mesh=12):
    ph = phonopy.load(yaml_path)
    ph.run_mesh([mesh, mesh, 1], is_gamma_center=True)
    return float(np.min(ph.get_mesh_dict()["frequencies"]))

rows = []
for mat in ["MoS2", "1T-TaS2", "1T-TiS2"]:
    for dg in [0.005, 0.010, 0.015, 0.020]:
        dgs = f"{dg:.3f}"
        cands = glob.glob(str(SK / f"fc2_{mat}_3x3_{dgs}" / "*_phonopy.yaml"))
        if not cands:
            print(f"{mat:10s} dg{dg}: MISSING"); continue
        mf = min_freq(cands[0])
        rows.append((mat, dg, dg * TEL, mf))
        print(f"{mat:10s} dg{dg:<5} T_el={dg*TEL:7.0f}K  min-freq={mf:+.3f} THz")

# contrast: slope of min-freq vs T_el per material
print("\n=== (E)-response (melting slope, mTHz/K) ===")
import csv
Path("results/smearing_kink").mkdir(parents=True, exist_ok=True)
with open("results/smearing_kink/breadth_melting.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["material", "degauss", "T_el_K", "minfreq_THz"])
    for mat, dg, tel, mf in rows: w.writerow([mat, dg, f"{tel:.0f}", f"{mf:.4f}"])
for mat in ["MoS2", "1T-TaS2", "1T-TiS2"]:
    pts = [(r[2], r[3]) for r in rows if r[0] == mat]
    if len(pts) >= 2:
        T = np.array([p[0] for p in pts]); f = np.array([p[1] for p in pts])
        slope = abs(np.polyfit(T, f, 1)[0]) * 1000
        tag = "FLAT (gapped, no smearing dep)" if slope < 0.15 else "melts (metallic)"
        print(f"  {mat:10s} slope={slope:.3f}  -> {tag}")
print("\nwrote results/smearing_kink/breadth_melting.csv")
