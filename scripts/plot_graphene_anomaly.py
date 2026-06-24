"""M1.1 figure: graphene dispersion, foundation MACE vs FC-distilled, with the
Kohn-anomaly locator's q* marked and literature anchors (Gamma-E2g ~1600 cm^-1,
K-A1' ~1300 cm^-1 with the sharp cusp).

    conda run -n phonon python scripts/plot_graphene_anomaly.py \
        --base results/td_phonon/disp_graphene_base.npz \
        --ft   results/td_phonon/disp_graphene_ft.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import anomaly_locate as al

CM = 33.35641  # THz -> cm^-1
# literature graphene Kohn-anomaly anchors (cm^-1)
LIT = {"Gamma_E2g": 1600.0, "K_A1p": 1300.0}


def _load(p):
    return dict(np.load(p, allow_pickle=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", default="results/figures/graphene_kohn_anomaly.png")
    ap.add_argument("--units", default="cm", choices=["cm", "thz"])
    a = ap.parse_args()

    base, ft = _load(a.base), _load(a.ft)
    scale = CM if a.units == "cm" else 1.0
    ylab = "Frequency (cm$^{-1}$)" if a.units == "cm" else "Frequency (THz)"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from plot_style import C, set_style
        set_style()
    except Exception:
        C = {"dfpt": "#000000", "mace": "#d1495b", "finetuned": "#2a9d4a"}

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    for d, color, name in ((base, C["mace"], "foundation MACE"),
                           (ft, C["finetuned"], "FC-distilled")):
        dist = d["distances"]
        freq = d["frequencies"] * scale
        ax.plot(dist, freq[:, 0], color=color, lw=1.3, alpha=0.9, label=name)
        ax.plot(dist, freq[:, 1:], color=color, lw=1.3, alpha=0.9)

    # high-symmetry guides
    labels = [str(x) for x in base["labels"]]
    for x in base["label_positions"]:
        ax.axvline(x, color="0.6", lw=0.6, zorder=0)
    ax.set_xticks(base["label_positions"])
    ax.set_xticklabels(labels)
    ax.set_xlim(base["distances"][0], base["distances"][-1])

    # literature anchors at Gamma and K
    gpos = base["label_positions"][0]
    kpos = base["label_positions"][2]
    for (xpos, key, txt) in ((gpos, "Gamma_E2g", "E$_{2g}$"),
                             (kpos, "K_A1p", "A$_1'$")):
        yv = LIT[key] if a.units == "cm" else LIT[key] / CM
        ax.plot([xpos], [yv], marker="_", ms=16, mew=2.0, color=C["dfpt"], zorder=5)
        ax.annotate(f"lit {txt}", (xpos, yv), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=C["dfpt"])

    # locate + mark cusps for the FC-distilled model (the one meant to recover them)
    an = al.locate_anomalies(ft["distances"], ft["frequencies"],
                             ft["label_positions"], ft["labels"],
                             qfrac=ft.get("qpoints_frac"))
    for r in an[:6]:
        ax.scatter([r["distance"]], [r["freq_thz"] * scale], s=55,
                   facecolors="none", edgecolors=C["finetuned"], lw=1.6, zorder=6)

    ax.set_ylabel(ylab)
    ax.set_ylim(bottom=min(0, ft["frequencies"].min() * scale))
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)
    ax.set_title("Graphene phonon dispersion + Kohn-anomaly locator")
    fig.tight_layout()

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))

    # numeric summary
    print(f"{'quantity':28s} {'foundation':>12s} {'FC-distill':>12s} {'lit':>8s}")
    for xpos, key, name in ((gpos, "Gamma_E2g", "top optical @ Gamma"),
                            (kpos, "K_A1p", "top optical @ K")):
        labname = "$\\Gamma$" if "Gamma" in key else "K"
        wb = al.branch_freq_at_label(base["distances"], base["frequencies"],
                                     base["label_positions"], base["labels"], labname) * CM
        wf = al.branch_freq_at_label(ft["distances"], ft["frequencies"],
                                     ft["label_positions"], ft["labels"], labname) * CM
        print(f"{name:28s} {wb:9.0f}    {wf:9.0f}    {LIT[key]:6.0f}  cm^-1")
    print(f"\nFC-distilled cusps (top {min(6,len(an))}):")
    for r in an[:6]:
        print(f"  {r['nearest_label']:>8s} branch {r['branch']:2d} "
              f"omega={r['freq_thz']*CM:6.0f} cm^-1 kink|dv|={r['kink_strength']:6.2f} "
              f"at_label={r['at_label']}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
