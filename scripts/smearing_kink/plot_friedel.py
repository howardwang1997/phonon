"""Figure: the 2-parameter damped-Friedel long-range module.

(A) kink_K(T_el) collapse: DFT vs backbone(smearing-blind MLIP) vs full 2-param
    fit vs few-shot-from-3.   (B) the mechanism: longitudinal fc2 tail
    delta_phi_L(R) = a thermally-damped Friedel oscillation that shortens with T_el.

    conda run --no-capture-output -n phonon python scripts/smearing_kink/plot_friedel.py
"""
from __future__ import annotations
import sys, csv, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import friedel_module as fm
ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "results" / "vq_kink6"
OUT = ROOT / "results" / "smearing_kink"

# Okabe-Ito colorblind-safe palette
CB = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7",
          grey="#999999")


def main():
    rows = list(csv.DictReader(open(OUT / "friedel_fit.csv")))
    T = np.array([float(r["T_el"]) for r in rows])
    kd = np.array([float(r["kinkK_dft"]) for r in rows])
    km = np.array([float(r["kinkK_model"]) for r in rows])
    kf = np.array([float(r["kinkK_fewshot"]) for r in rows])
    xi = np.array([float(r["xi_extra"]) for r in rows])
    anchors_T = [316, 1579, 6315]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    # --- panel A: kink collapse ---------------------------------------------
    axA.axhline(0.22, color=CB["grey"], lw=2, ls=":",
                label="backbone (smearing-blind MLIP)")
    axA.plot(T, km, "-", color=CB["purple"], lw=2,
             label="Friedel module, 2 param / smearing")
    axA.plot(T, kf, "--", color=CB["orange"], lw=2,
             label="few-shot from 3 smearings")
    axA.plot(T, kd, "o", color=CB["black"], ms=7, mfc="white", mew=1.6,
             label="DFT (ground truth)", zorder=5)
    for at in anchors_T:
        i = int(np.argmin(np.abs(T - at)))
        axA.plot(T[i], kf[i], "s", color=CB["orange"], ms=9, zorder=6)
    axA.set_xscale("log")
    axA.set_xlabel(r"electronic temperature  $T_{\mathrm{el}} = \mathrm{degauss}\times157{,}888$  (K)")
    axA.set_ylabel(r"K-A$_1'$ Kohn kink  $|\Delta v|$  (THz / q-unit)")
    axA.set_title("(A)  smearing collapse of the K cusp", loc="left", fontsize=11)
    axA.legend(fontsize=8.3, frameon=False, loc="upper right")
    axA.grid(alpha=0.25, lw=0.5)
    axA.text(0.03, 0.05, "squares = the 3 few-shot anchors",
             transform=axA.transAxes, fontsize=8, color=CB["orange"])

    # --- panel B: the Friedel tail vs R for 3 smearings ---------------------
    show = [("0.002", CB["blue"], r"$T_{\rm el}$=316 K (sharp)"),
            ("0.010", CB["green"], r"$T_{\rm el}$=1579 K"),
            ("0.040", CB["verm"], r"$T_{\rm el}$=6315 K (broad)")]
    ph80 = fm.load_ph(YDIR / "graphene_sc6_dg0.080_phonopy.yaml")
    fc80 = ph80.force_constants
    tabs, _ = fm.pair_table(ph80)
    for dg, col, lab in show:
        ph = fm.load_ph(YDIR / f"graphene_sc6_dg{dg}_phonopy.yaml")
        rows_d = fm.delta_long(ph.force_constants, fc80, tabs, 3.0, 12.0)
        R = np.concatenate([r[0] for r in rows_d])
        dL = np.concatenate([r[1] for r in rows_d])
        o = np.argsort(R)
        # bin-average symmetry-equivalent pairs at same R for a clean curve
        Ru, idx = np.unique(np.round(R[o], 2), return_inverse=True)
        dLu = np.array([dL[o][idx == k].mean() for k in range(len(Ru))])
        axB.plot(Ru, dLu, "-o", color=col, ms=3.5, lw=1.4, label=lab)
    axB.axhline(0, color=CB["grey"], lw=0.8)
    axB.set_xlabel(r"pair distance  $R$  ($\mathrm{\AA}$)")
    axB.set_ylabel(r"$\Delta\phi_L(R)=\phi_L(T_{\rm el})-\phi_L^{\rm backbone}$  (eV/$\mathrm{\AA}^2$)")
    axB.set_title(r"(B)  the mechanism: damped Friedel oscillation in fc$_2$",
                  loc="left", fontsize=11)
    axB.legend(fontsize=8.3, frameon=False)
    axB.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    fig.savefig(OUT / "friedel_module.png", dpi=155)
    print("wrote", OUT / "friedel_module.png")


if __name__ == "__main__":
    main()
