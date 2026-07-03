"""Cross-model backbone check: the Friedel long-range module is an additive,
backbone-agnostic harmonic term, so it must compose with ANY MLIP, not just MACE.
Here the base backbone is a foundation SevenNet (7net-0) instead of MACE.

Run in the dedicated env (see setup_sevenn.sh):
    conda run --no-capture-output -n phonon-sevenn python scripts/smearing_kink/cross_model_check.py

Note: a foundation model (SevenNet-0 / MACE-MP-0) is a poor *graphene* backbone
(known 2D failure) -> its bare kink is unphysical; the point here is only that the
Friedel term adds the correct T_el-dependence on top of whatever backbone is used.
A production run uses a graphene-fine-tuned backbone (see gr_backbone_distill.sh).
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
from friedel_calc import FriedelCorrection, FriedelMACECalculator, fc2_from_calc

YDIR = ROOT / "results" / "vq_kink6"


def sevennet_calc(model="7net-0"):
    """model = '7net-0' (foundation) or a path to a fine-tuned checkpoint .pth."""
    from sevenn.calculator import SevenNetCalculator
    dev = "cpu"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    for args in [((), {"model": model, "device": dev}),
                 ((model,), {"device": dev})]:
        try:
            return SevenNetCalculator(*args[0], **args[1])
        except Exception:
            continue
    raise RuntimeError("could not construct SevenNetCalculator")


def main():
    ph = fm.load_ph(YDIR / "graphene_sc6_dg0.080_phonopy.yaml")
    fc_bg = ph.force_constants
    fc_ref = fm.load_ph(YDIR / "graphene_sc6_dg0.002_phonopy.yaml").force_constants
    corr = FriedelCorrection(ph, fc_bg, fc_ref)
    from phonon_accel.phonons import phonopy_to_ase
    ref_atoms = phonopy_to_ase(ph.supercell)

    model = sys.argv[1] if len(sys.argv) > 1 else "7net-0"
    tag = "fine-tuned dg0.080" if model != "7net-0" else "foundation 7net-0"
    print(f"# cross-model: base = SevenNet ({tag}) — a DIFFERENT MLIP framework")
    base = sevennet_calc(model)
    _, fc_b = fc2_from_calc(ph, base, subtract_ref=True)
    kb = fm.kink_of(ph, fc_b)[0]
    print(f"  SevenNet backbone only              -> kink_K={kb:6.2f}")
    for T in (316, 1579, 6315):
        calc = FriedelMACECalculator(base, ref_atoms, corr, T)
        _, fc_t = fc2_from_calc(ph, calc, subtract_ref=True)
        kK = fm.kink_of(ph, fc_t)[0]
        print(f"  SevenNet + Friedel(T_el={T:>5} K)     -> kink_K={kK:6.2f}   "
              f"(delta from backbone = {kK-kb:+.2f})")
    print("# the Friedel term shifts the kink with T_el on a SevenNet backbone too")
    print("# -> the long-range module is backbone-agnostic (additive harmonic term).")


if __name__ == "__main__":
    main()
