"""M1.1 convergence figure: graphene Gamma-E2g / K-A1' frequency and top-branch
kink vs supercell size, foundation vs FC-distilled. Shows the softening is a
genuine PES effect (frequencies plateau below the literature anchors) and that
the weak Gamma cusp is not a truncation artifact (kink_Gamma stays flat while
kink_K converges high).

    conda run -n phonon python scripts/plot_sc_convergence.py \
        --csv results/td_phonon/graphene_sc_convergence.csv
"""
from __future__ import annotations

import argparse
import csv as csvmod
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

LIT = {"gamma": 1600.0, "k": 1300.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/td_phonon/graphene_sc_convergence.csv")
    ap.add_argument("--out", default="results/figures/graphene_sc_convergence.png")
    a = ap.parse_args()

    rows = defaultdict(list)
    with open(ROOT / a.csv) as f:
        for r in csvmod.DictReader(f):
            rows[r["model"]].append(r)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from plot_style import C, set_style
        set_style()
    except Exception:
        C = {"dfpt": "#000000", "mace": "#d1495b", "finetuned": "#2a9d4a"}

    color = {"foundation": C["mace"], "FC-distilled": C["finetuned"]}
    fig, (axf, axk) = plt.subplots(1, 2, figsize=(8.4, 3.6))

    for model, rs in rows.items():
        rs = sorted(rs, key=lambda r: int(r["natoms"]))
        n = [int(r["natoms"]) for r in rs]
        col = color.get(model, "0.4")
        axf.plot(n, [float(r["w_gamma_cm"]) for r in rs], "o-", color=col,
                 label=f"{model} $\\Gamma$")
        axf.plot(n, [float(r["w_k_cm"]) for r in rs], "s--", color=col, alpha=0.7,
                 label=f"{model} K")
        axk.plot(n, [float(r["kink_gamma"]) for r in rs], "o-", color=col,
                 label=f"{model} $\\Gamma$")
        axk.plot(n, [float(r["kink_k"]) for r in rs], "s--", color=col, alpha=0.7,
                 label=f"{model} K")

    axf.axhline(LIT["gamma"], color=C["dfpt"], lw=1.0, ls=":")
    axf.axhline(LIT["k"], color=C["dfpt"], lw=1.0, ls=":")
    axf.annotate("lit $\\Gamma$-E$_{2g}$", (axf.get_xlim()[1], LIT["gamma"]),
                 ha="right", va="bottom", fontsize=8)
    axf.annotate("lit K-A$_1'$", (axf.get_xlim()[1], LIT["k"]),
                 ha="right", va="bottom", fontsize=8)
    axf.set_xlabel("supercell atoms")
    axf.set_ylabel("Frequency (cm$^{-1}$)")
    axf.set_title("Top-optical frequency vs supercell")
    axf.legend(frameon=False, fontsize=7, ncol=2)

    axk.set_xlabel("supercell atoms")
    axk.set_ylabel("top-branch kink |dv| (THz / q-unit)")
    axk.set_title("Cusp sharpness vs supercell")
    axk.legend(frameon=False, fontsize=7, ncol=2)

    fig.tight_layout()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
