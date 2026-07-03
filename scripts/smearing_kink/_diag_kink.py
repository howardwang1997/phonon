"""Diagnostic: where is the kink lost? Test reconstruction fidelity step by step.

  T0  fc(0.002) itself                              -> must be 22.63 (kink machinery OK)
  T1  backbone + EXACT full-tensor delta            -> identity, must be 22.63
  T2  backbone + EXACT longitudinal-only delta      -> is longitudinal enough?
  T3  backbone + EXACT delta but only tail R>=rmin   -> does the tail carry the kink?
  T4  backbone + full-tensor delta scaled by A only  -> pure-amplitude (rank-1) trend
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import friedel_module as fm
ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "results" / "vq_kink6"
BG = "0.080"

def load(dg):
    return fm.load_ph(YDIR / f"graphene_sc6_dg{dg}_phonopy.yaml")

def main():
    ph0 = load("0.002"); ph80 = load(BG)
    fc0 = ph0.force_constants; fc80 = ph80.force_constants
    tabs, p2s = fm.pair_table(ph80)
    ph = ph80

    print("T0 fc(0.002) raw:            kink_K=%.2f (want 22.63)" % fm.kink_of(ph0, fc0)[0])
    print("T0 fc(0.080) raw:            kink_K=%.2f (want 0.22)" % fm.kink_of(ph80, fc80)[0])

    # T1 backbone + exact full-tensor delta == fc0
    fc = fc80 + (fc0 - fc80)
    print("T1 bg + exact FULL delta:    kink_K=%.2f (want 22.63)" % fm.kink_of(ph, fc)[0])

    # T2 backbone + exact longitudinal-only delta
    fc = fc80.copy()
    for prow, t in enumerate(tabs):
        n, s = t["n"], t["s"]
        db = fc0[prow, s] - fc80[prow, s]
        dL = np.einsum("ki,kij,kj->k", n, db, n)
        corr = np.einsum("k,ki,kj->kij", dL, n, n)
        fc[prow, s] += corr
        si = np.where(s == t["s0"])[0][0]
        fc[prow, si] -= corr.sum(axis=0)
    print("T2 bg + exact LONGIT delta:  kink_K=%.2f  (is longitudinal enough?)" % fm.kink_of(ph, fc)[0])

    # T3 backbone + exact FULL delta only for tail R>=rmin
    for rmin in (3.5, 4.5, 5.5):
        fc = fc80.copy()
        for prow, t in enumerate(tabs):
            s, R = t["s"], t["R"]
            m = R >= rmin
            db = (fc0[prow, s] - fc80[prow, s])
            db[~m] = 0.0
            fc[prow, s] += db
            si = np.where(s == t["s0"])[0][0]
            fc[prow, si] -= db.sum(axis=0)
        print("T3 bg + exact FULL delta (R>=%.1f): kink_K=%.2f" % (rmin, fm.kink_of(ph, fc)[0]))

    # T4 pure amplitude scaling of full delta (rank-1): A in {1,.75,.5,.25,0}
    full = fc0 - fc80
    print("T4 bg + A*(full delta), rank-1 amplitude sweep:")
    for A in (1.0, 0.75, 0.5, 0.25, 0.0):
        fc = fc80 + A * full
        print("     A=%.2f  kink_K=%.2f" % (A, fm.kink_of(ph, fc)[0]))

if __name__ == "__main__":
    main()
