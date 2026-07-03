"""Figure: transfer of the thermal Friedel damping law across lattice constant.
Reads friedel_transfer.csv (from transfer_friedel.py).

    conda run --no-capture-output -n phonon python scripts/smearing_kink/plot_transfer.py
"""
from __future__ import annotations
import csv, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "smearing_kink"
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00",
          purple="#CC79A7", grey="#999999", black="#000000")
ACOL = {"2.44": CB["blue"], "2.46": CB["black"], "2.48": CB["verm"], "2.50": CB["green"]}


def main():
    rows = list(csv.DictReader(open(OUT / "friedel_transfer.csv")))
    byA = {}
    for r in rows:
        byA.setdefault(r["a"], []).append(r)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for a, rs in sorted(byA.items()):
        rs.sort(key=lambda r: float(r["T_el"]))
        T = np.array([float(r["T_el"]) for r in rs])
        kd = np.array([float(r["kink_dft"]) for r in rs])
        kp = np.array([float(r["kink_pred"]) for r in rs])
        held = np.array([int(r["held_out"]) for r in rs], bool)
        c = ACOL.get(a, CB["grey"])
        train = " (train)" if a == "2.46" else " (transfer)"
        ax.plot(T, kp, "-", color=c, lw=1.8, alpha=0.9,
                label=f"a={a} A{train}: a=2.46 law")
        ax.plot(T[~held], kd[~held], "s", color=c, ms=9, mfc=c, zorder=5)  # anchors
        ax.plot(T[held], kd[held], "o", color=c, ms=8, mfc="white", mew=1.7, zorder=6)  # DFT held-out
    ax.set_xscale("log")
    ax.set_xlabel(r"electronic temperature  $T_{\rm el}$  (K)")
    ax.set_ylabel(r"K-A$_1'$ Kohn kink  $|\Delta v|$  (THz / q-unit)")
    ax.set_title("Transfer: a=2.46 damping law predicts held-out lattice constants",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.6, frameon=False, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(0.02, 0.06,
            "squares = 2 measured anchors (template+backbone)\n"
            "open circles = DFT at held-out smearings\n"
            "lines = prediction from the a=2.46 thermal law",
            transform=ax.transAxes, fontsize=7.8, color=CB["grey"], va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "friedel_transfer.png", dpi=155)
    print("wrote", OUT / "friedel_transfer.png")


if __name__ == "__main__":
    main()
