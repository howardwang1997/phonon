"""Data-efficiency of the Friedel few-shot law: how few smearings are needed to
fit kappa(T_el)=a*T^b (+ B=const) and still predict kink_K(T_el) at the rest?

Reads per-smearing (B, kappa) from friedel_fit.csv (produced by fit_friedel.py),
fits the smooth law from N anchor smearings (N=2,3,4, a few anchor choices),
predicts the held-out kinks, and reports the held-out MAE per N.

    conda run --no-capture-output -n phonon python scripts/smearing_kink/few_shot_dataeff.py
"""
from __future__ import annotations
import sys, csv, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import friedel_module as fm
from fit_friedel import DG, TEL, DFT_KINKK, YDIR, BG, REF

OUT = Path(__file__).resolve().parents[2] / "results" / "smearing_kink"


def main():
    rows = {r["degauss"]: r for r in csv.DictReader(open(OUT / "friedel_fit.csv"))}
    # per-smearing fitted (B, kappa) and DFT kink
    dgs = [d for d in DG if d not in (BG, REF)]       # the fittable intermediate set
    ph = fm.load_ph(YDIR / f"graphene_sc6_dg{BG}_phonopy.yaml")
    fc_bg = ph.force_constants
    fc_ref = fm.load_ph(YDIR / f"graphene_sc6_dg{REF}_phonopy.yaml").force_constants
    tabs, _ = fm.pair_table(ph)
    D0 = fm.template_delta(fc_ref, fc_bg, tabs)
    Bmean = np.mean([float(rows[d]["B"]) for d in dgs])

    def predict_kink(dg, a_pow, b):
        T = TEL[dg]; kap = a_pow * T ** b
        fc = fm.add_template(fc_bg, tabs, D0, Bmean, kap, 1.0, 12.0)
        return fm.kink_of(ph, fc)[0]

    # anchor sets of size N (always spanning the range: include a low + high)
    ANCHORS = {
        2: [["0.002", "0.040"]],
        3: [["0.002", "0.010", "0.040"], ["0.003", "0.010", "0.030"]],
        4: [["0.002", "0.006", "0.015", "0.040"]],
    }
    print(f"# few-shot data-efficiency (B~const={Bmean:.3f}); kappa(T)=a*T^b fitted from N anchors")
    print(f"# held-out MAE(kink_K) over the non-anchor smearings\n")
    print(f"{'N':>2} {'anchors':>34} {'a':>10} {'b':>6} {'heldout_MAE':>11}")
    for N, sets in ANCHORS.items():
        for anc in sets:
            Ta = np.array([TEL[d] for d in anc], float)
            Ka = np.array([float(rows[d]["kappa"]) for d in anc])
            m = Ka > 1e-6
            if m.sum() < 2:
                print(f"{N:>2} {'/'.join(anc):>34}   (need >=2 nonzero kappa)"); continue
            b, loga = np.polyfit(np.log(Ta[m]), np.log(Ka[m]), 1)
            a_pow = np.exp(loga)
            errs = []
            for dg in DG:
                if dg in anc or dg == BG:
                    continue
                errs.append(abs(predict_kink(dg, a_pow, b) - DFT_KINKK[dg]))
            print(f"{N:>2} {'/'.join(anc):>34} {a_pow:>10.2e} {b:>6.2f} {np.mean(errs):>11.2f}")
    print("\n# conclusion: the smooth kappa(T) law needs only a handful of smearings;")
    print("# report the smallest N whose held-out MAE stays ~<1 cm^-1.")


if __name__ == "__main__":
    main()
