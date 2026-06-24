"""M1.1 figure: graphene dispersion, foundation MACE vs FC-distilled, with the
Kohn-anomaly locator's top-branch kinks marked and literature anchors
(Gamma-E2g ~1600 cm^-1, K-A1' ~1300 cm^-1, both with a sharp cusp).

Use a path with Gamma and K interior (e.g. M-Gamma-K-M) so both anomalies carry
a two-sided kink.

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
LIT = {r"$\Gamma$": 1600.0, "K": 1300.0}  # graphene Kohn-anomaly anchors, cm^-1
LIT_TXT = {r"$\Gamma$": "E$_{2g}$", "K": "A$_1'$"}


def _load(p):
    return dict(np.load(p, allow_pickle=True))


def _label_pos(d, name):
    j = [str(x) for x in d["labels"]].index(name)
    return float(d["label_positions"][j])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", default="results/figures/graphene_kohn_anomaly.png")
    a = ap.parse_args()

    base, ft = _load(a.base), _load(a.ft)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from plot_style import C, set_style
        set_style()
    except Exception:
        C = {"dfpt": "#000000", "mace": "#d1495b", "finetuned": "#2a9d4a"}

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    for d, color, name in ((base, C["mace"], "foundation MACE"),
                           (ft, C["finetuned"], "FC-distilled")):
        freq = d["frequencies"] * CM
        ax.plot(d["distances"], freq[:, 0], color=color, lw=1.3, alpha=0.9, label=name)
        ax.plot(d["distances"], freq[:, 1:], color=color, lw=1.3, alpha=0.9)

    labels = [str(x) for x in base["labels"]]
    for x in base["label_positions"]:
        ax.axvline(x, color="0.6", lw=0.6, zorder=0)
    ax.set_xticks(base["label_positions"])
    ax.set_xticklabels(labels)
    ax.set_xlim(base["distances"][0], base["distances"][-1])

    # literature anchors + top-branch kink markers at Gamma and K
    for name in (r"$\Gamma$", "K"):
        if name not in labels:
            continue
        xpos = _label_pos(base, name)
        ax.plot([xpos], [LIT[name]], marker="_", ms=18, mew=2.2,
                color=C["dfpt"], zorder=6)
        ax.annotate(f"lit {LIT_TXT[name]}", (xpos, LIT[name]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    color=C["dfpt"])

    # mark the FC-distilled top-branch frequency at interior high-sym points
    for k in al.high_sym_kinks(ft["distances"], ft["frequencies"],
                               ft["label_positions"], ft["labels"]):
        if k["interior"]:
            ax.scatter([k["distance"]], [k["freq_thz"] * CM], s=60,
                       facecolors="none", edgecolors=C["finetuned"], lw=1.7, zorder=7)

    ax.set_ylabel("Frequency (cm$^{-1}$)")
    ax.set_ylim(bottom=min(0, ft["frequencies"].min() * CM))
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)
    ax.set_title("Graphene phonon dispersion + Kohn-anomaly locator")
    fig.tight_layout()

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))

    # numeric summary: softening + cusp-sharpness recovery
    print(f"{'quantity':26s} {'foundation':>11s} {'FC-distill':>11s} {'lit':>7s}")
    for name in (r"$\Gamma$", "K"):
        if name not in labels:
            continue
        wb = al.branch_freq_at_label(base["distances"], base["frequencies"],
                                     base["label_positions"], base["labels"], name) * CM
        wf = al.branch_freq_at_label(ft["distances"], ft["frequencies"],
                                     ft["label_positions"], ft["labels"], name) * CM
        nm = "Gamma" if "Gamma" in name else name
        print(f"  top optical @ {nm:8s} {wb:9.0f}   {wf:9.0f}   {LIT[name]:6.0f}  cm^-1")
    kb = {k["label"]: k for k in al.high_sym_kinks(base["distances"],
          base["frequencies"], base["label_positions"], base["labels"])}
    kf = {k["label"]: k for k in al.high_sym_kinks(ft["distances"],
          ft["frequencies"], ft["label_positions"], ft["labels"])}
    print("  --- top-branch kink |dv| (interior points; larger = sharper cusp) ---")
    for name in (r"$\Gamma$", "K"):
        if name in kb and kb[name]["interior"]:
            nm = "Gamma" if "Gamma" in name else name
            print(f"  kink @ {nm:8s}        {kb[name]['kink_strength']:9.1f}   "
                  f"{kf[name]['kink_strength']:9.1f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
