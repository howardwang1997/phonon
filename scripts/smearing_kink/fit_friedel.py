"""Fit the 2-parameter damped-Friedel long-range module to graphene fc2(T_el) and
validate it reproduces kink_K(T_el); then few-shot from 3 smearings.

    conda run --no-capture-output -n phonon python scripts/smearing_kink/fit_friedel.py

Model (see friedel_module.py):
    fc2(T_el) = fc2_backbone(dg0.080)  +  B(T_el) * exp(-kappa(T_el) R) * D0(R)
D0 = Fermi-surface Friedel waveform (full 3x3 tensor) measured at the sharpest
smearing (dg0.002 - dg0.080); the two thermal parameters are the amplitude
B(T_el) and the EXTRA damping rate kappa(T_el) = 1/xi - 1/xi_ref.  Fit on the
fc2 tail R in [rmin,rmax]; readout = kink_K via phonopy band on M-G-K-M.
"""
from __future__ import annotations
import sys, argparse, csv, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import friedel_module as fm

ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "results" / "vq_kink6"
OUT = ROOT / "results" / "smearing_kink"

DG = ["0.002", "0.003", "0.004", "0.005", "0.006", "0.007",
      "0.010", "0.015", "0.020", "0.030", "0.040", "0.080"]
TEL = {"0.002": 316, "0.003": 474, "0.004": 632, "0.005": 789, "0.006": 947,
       "0.007": 1105, "0.010": 1579, "0.015": 2368, "0.020": 3158,
       "0.030": 4737, "0.040": 6315, "0.080": 12631}
DFT_KINKK = {"0.002": 22.63, "0.003": 18.51, "0.004": 16.74, "0.005": 15.72,
             "0.006": 15.04, "0.007": 14.52, "0.010": 13.33, "0.015": 11.56,
             "0.020": 9.75, "0.030": 6.42, "0.040": 3.91, "0.080": 0.22}
BG = "0.080"    # backbone = maximally smeared (Friedel damped out)
REF = "0.002"   # template = sharpest smearing (least-damped Friedel waveform)


def load_all():
    phs, fcs = {}, {}
    for dg in DG:
        ph = fm.load_ph(YDIR / f"graphene_sc6_dg{dg}_phonopy.yaml")
        phs[dg], fcs[dg] = ph, ph.force_constants
    tabs, p2s = fm.pair_table(phs[BG])
    return phs, fcs, tabs, p2s


def fit_all(fcs, tabs, D0, rmin, rmax, kappas, wexp=0.0):
    fits = {}
    for dg in DG:
        if dg == BG:
            fits[dg] = (0.0, 0.0, 0.0)          # backbone: no correction
        elif dg == REF:
            fits[dg] = (1.0, 0.0, 0.0)          # template anchor: exact
        else:
            fits[dg] = fm.fit_template_env(fcs[dg], fcs[BG], tabs, D0, rmin, rmax, kappas, wexp)
    return fits


def reconstruct_kinks(ph, fcbg, tabs, D0, fits, rmin, rmax):
    rows = []
    for dg in DG:
        B, kap, _ = fits[dg]
        fc = fm.add_template(fcbg, tabs, D0, B, kap, rmin, rmax)
        kK, wK, kG = fm.kink_of(ph, fc)
        rows.append((dg, TEL[dg], B, kap, kK, DFT_KINKK[dg], wK, kG))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmin", type=float, default=1.0)
    ap.add_argument("--rmax", type=float, default=12.0)
    ap.add_argument("--wexp", type=float, default=0.0, help="pair weight R**wexp (tail emphasis)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"# 12 fc2(T_el); backbone=dg{BG}, template=dg{REF}; tail R in [{a.rmin},{a.rmax}] A; wexp={a.wexp}")
    phs, fcs, tabs, p2s = load_all()
    D0 = fm.template_delta(fcs[REF], fcs[BG], tabs)
    ph = phs[BG]

    kappas = np.linspace(0.0, 2.0, 161)   # extra damping rate (1/A)
    fits = fit_all(fcs, tabs, D0, a.rmin, a.rmax, kappas, a.wexp)
    rows = reconstruct_kinks(ph, fcs[BG], tabs, D0, fits, a.rmin, a.rmax)

    print(f"\n{'dg':>6} {'T_el':>6} {'B':>8} {'kappa':>7} {'xi_ext':>7} "
          f"{'kinkK_mod':>10} {'kinkK_DFT':>10} {'err':>7} {'resid%':>7}")
    errs = []
    for dg, T, B, kap, kK, kd, wK, kG in rows:
        e = kK - kd
        errs.append(abs(e))
        xi = (1.0 / kap) if kap > 1e-6 else np.inf
        rp = fits[dg][2] * 100
        print(f"{dg:>6} {T:>6} {B:>8.4f} {kap:>7.3f} {xi:>7.2f} "
              f"{kK:>10.2f} {kd:>10.2f} {e:>7.2f} {rp:>7.2f}")
    print(f"# MAE(kink_K) = {np.mean(errs):.2f} cm^-1/qunit   "
          f"(backbone flat=0.22, full-recon=exact)")

    # ---- few-shot: fit smooth B, kappa laws from 3 anchor smearings, predict all
    anchors = ["0.002", "0.010", "0.040"]
    Ta = np.array([TEL[d] for d in anchors], float)
    Ba = np.array([fits[d][0] for d in anchors])
    Ka = np.array([fits[d][1] for d in anchors])
    # B(T) ~ constant (physical: Friedel amplitude ~ T_el-independent) -> mean
    Bfun = lambda T: np.full_like(np.atleast_1d(T), Ba.mean(), dtype=float)
    # kappa(T) ~ a * T^b  (extra thermal damping rate); fit log-log on the two
    # non-zero anchors (kappa(ref)=0 by construction) + the mid anchor
    msk = Ka > 1e-6
    b, loga = np.polyfit(np.log(Ta[msk]), np.log(Ka[msk]), 1)
    a_pow = np.exp(loga)
    Kfun = lambda T: a_pow * np.asarray(T, float) ** b
    print(f"\n# FEW-SHOT from {anchors}: B~const={Ba.mean():.3f}, "
          f"kappa(T)={a_pow:.3e}*T^{b:.2f}")
    fs_rows = []
    for dg in DG:
        T = TEL[dg]
        B = float(Bfun(T)[0]); kap = float(Kfun(T))
        if dg == BG:
            B, kap = 0.0, 0.0
        fc = fm.add_template(fcs[BG], tabs, D0, B, kap, a.rmin, a.rmax)
        kK = fm.kink_of(ph, fc)[0]
        fs_rows.append((dg, T, B, kap, kK, DFT_KINKK[dg]))
    fs_err = [abs(r[4] - r[5]) for r in fs_rows if r[0] not in anchors + [BG]]
    print(f"{'dg':>6} {'T_el':>6} {'kinkK_fewshot':>13} {'kinkK_DFT':>10}")
    for dg, T, B, kap, kK, kd in fs_rows:
        tag = " (anchor)" if dg in anchors else ""
        print(f"{dg:>6} {T:>6} {kK:>13.2f} {kd:>10.2f}{tag}")
    print(f"# few-shot MAE on the 8 held-out smearings = {np.mean(fs_err):.2f}")

    with open(OUT / "friedel_fit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["degauss", "T_el", "B", "kappa", "xi_extra",
                    "kinkK_model", "kinkK_fewshot", "kinkK_dft", "wK_cm",
                    "kinkG_model", "resid_pct"])
        fsd = {r[0]: r[4] for r in fs_rows}
        for dg, T, B, kap, kK, kd, wK, kG in rows:
            xi = (1.0 / kap) if kap > 1e-6 else 0.0
            w.writerow([dg, T, B, kap, xi, kK, fsd[dg], kd, wK, kG, fits[dg][2] * 100])
    print(f"# wrote {OUT/'friedel_fit.csv'}")


if __name__ == "__main__":
    main()
