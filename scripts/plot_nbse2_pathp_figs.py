"""Stage B figures for the Path-P NbSe2 story:
  (a) omega(q,T) dispersion overlay  -- the soft CDW branch filling in with T (TDEP).
  (b) soft-mode min-freq vs T: TDEP (perturbative) vs SSCHA (rigorous, #1)
      [extensible: the Path-P anharmonic-FT SSCHA curve is added at Stage E].

    python scripts/plot_nbse2_pathp_figs.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TD = ROOT / "results" / "td_phonon"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
THZ_TO_CM = 33.356410


def fig_omega_qT():
    d = np.load(TD / "td_nbse2_ft_Tfix.npz", allow_pickle=True)
    temps = [int(t) for t in d["temperatures"]]
    labels = [str(x) for x in d["labels"]]
    lp = d["label_positions"]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(temps)))
    for T, c in zip(temps, colors):
        dist = d[f"T{T}_dist"]; freq = d[f"T{T}_freq"] * THZ_TO_CM  # cm^-1
        # plot the lowest 3 branches (acoustic + CDW soft); label once
        for b in range(3):
            ax.plot(dist, freq[:, b], color=c, lw=1.3,
                    label=(f"{T} K" if b == 0 else None))
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(lp); ax.set_xticklabels(labels)
    for x in lp:
        ax.axvline(x, color="0.85", lw=0.6)
    ax.set_ylabel("frequency (cm$^{-1}$)")
    ax.set_title("NbSe$_2$ $\\omega(q,T)$ (distilled-FT + TDEP, fixed $a$): soft CDW branch vs T")
    ax.set_ylim(-90, None)
    ax.legend(title="electronic/lattice T", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "nbse2_omega_qT.png", dpi=150)
    print("saved", FIG / "nbse2_omega_qT.png")


def fig_softmode_T():
    # TDEP (Tfix.csv): minfreq_thz vs T
    rows = list(csv.DictReader(open(TD / "td_nbse2_ft_Tfix.csv")))
    T_t = np.array([float(r["T_K"]) for r in rows])
    mf_t = np.array([float(r["minfreq_thz"]) for r in rows]) * THZ_TO_CM
    # SSCHA (#1): sscha_minfreq_cm vs T
    srows = list(csv.DictReader(open(TD / "nbse2_sscha.csv")))
    T_s = np.array([float(r["T_K"]) for r in srows])
    mf_s = np.array([float(r["sscha_minfreq_cm"]) for r in srows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.plot(T_t, mf_t, "o-", color="tab:purple", lw=1.8, ms=7,
            label="TDEP (perturbative), harmonic-FT")
    ax.plot(T_s, mf_s, "s-", color="tab:green", lw=1.8, ms=7,
            label="SSCHA (rigorous, #1), harmonic-FT")
    # extensible: Path-P anharmonic-FT SSCHA curve, if Stage D wrote it
    pathp = TD / "nbse2_sscha_pathp.csv"
    if pathp.exists():
        pr = list(csv.DictReader(open(pathp)))
        T_p = np.array([float(r["T_K"]) for r in pr])
        mf_p = np.array([float(r["sscha_minfreq_cm"]) for r in pr])
        ax.plot(T_p, mf_p, "D-", color="tab:red", lw=2.0, ms=7,
                label="SSCHA, Path-P anharmonic-FT")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("min phonon frequency (cm$^{-1}$)   [<0 = soft mode]")
    ax.set_title("NbSe$_2$ CDW soft mode vs T: TDEP vs SSCHA")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "nbse2_softmode_TDEP_vs_SSCHA.png", dpi=150)
    print("saved", FIG / "nbse2_softmode_TDEP_vs_SSCHA.png")


if __name__ == "__main__":
    fig_omega_qT()
    fig_softmode_T()
