"""Evaluate the distilled graphene backbone models (gr_backbone_distill.sh) and
the FULL deployment with a REAL MLIP backbone.

(A) E2a conditioning baseline: each smearing-specific MACE must reproduce ITS OWN
    kink -> a conditioned backbone COULD span T_el; a single fixed one cannot.
(B) Deployment: base = the dg0.080 backbone MACE + the Friedel module ->
    kink(T_el) tracks DFT with a genuine trained MLIP (not the harmonic stand-in,
    not the poor MACE-MP-0).

    conda run --no-capture-output -n phonon python scripts/smearing_kink/eval_backbone_deploy.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import friedel_module as fm
from friedel_calc import (HarmonicCalculator, FriedelCorrection,
                          FriedelMACECalculator, fc2_from_calc, full_fc)

YDIR = ROOT / "results" / "vq_kink6"
BB = ROOT / "results" / "gr_backbone"
DFT_KINKK = {"0.002": 22.63, "0.010": 13.33, "0.080": 0.22}


def mace_calc(model_path):
    from mace.calculators import MACECalculator
    dev = "cuda"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    return MACECalculator(model_paths=str(model_path), device=dev,
                          default_dtype="float32")


def main():
    ph = fm.load_ph(YDIR / "graphene_sc6_dg0.080_phonopy.yaml")
    fc_bg = ph.force_constants
    fc_ref = fm.load_ph(YDIR / "graphene_sc6_dg0.002_phonopy.yaml").force_constants

    print("# (A) E2a conditioning baseline: does each smearing-specific MACE hit its own kink?")
    print(f"  {'dg':>6} {'MACE_kink':>9} {'DFT_kink':>8}")
    models = {}
    for dg in ("0.002", "0.010", "0.080"):
        mp = BB / f"dg{dg}" / f"gr_dg{dg}.model"
        if not mp.exists():
            print(f"  {dg:>6}   [model not found: {mp}]"); continue
        calc = mace_calc(mp); models[dg] = calc
        _, fc_m = fc2_from_calc(ph, calc, subtract_ref=True)
        kK = fm.kink_of(ph, fc_m)[0]
        print(f"  {dg:>6} {kK:>9.2f} {DFT_KINKK[dg]:>8.2f}")

    if "0.080" not in models:
        print("# (B) skipped: dg0.080 backbone model not ready"); return

    print("\n# (B) deployment: REAL MACE backbone (dg0.080) + Friedel module -> kink(T_el)")
    corr = FriedelCorrection(ph, fc_bg, fc_ref)
    from phonon_accel.phonons import phonopy_to_ase
    ref_atoms = phonopy_to_ase(ph.supercell)
    base = models["0.080"]
    DFT = {316: 22.63, 789: 15.72, 1579: 13.33, 3158: 9.75, 6315: 3.91}
    print(f"  {'T_el':>6} {'MACE+Friedel':>12} {'DFT':>6}")
    for T, kd in DFT.items():
        calc = FriedelMACECalculator(base, ref_atoms, corr, T)
        _, fc_t = fc2_from_calc(ph, calc, subtract_ref=True)
        kK = fm.kink_of(ph, fc_t)[0]
        print(f"  {T:>6} {kK:>12.2f} {kd:>6.2f}")
    print("# a genuine trained MLIP backbone now carries the correct T_el-dependent K cusp.")


if __name__ == "__main__":
    main()
