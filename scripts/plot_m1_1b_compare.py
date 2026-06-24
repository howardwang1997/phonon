"""M1.1b figure: does graphene-specific FC distillation recover the Gamma-E2g
(and K-A1') Kohn cusp that the bulk-trained model misses?

Overlays the graphene dispersion from: foundation MACE, the general (bulk) FC-
distilled model, the NEW graphene-specific FC-distilled model, and the graphene
DFT reference -- against the literature anchors. The headline number is how far
the graphene-specific model closes the foundation->DFT gap at Gamma.

    conda run -n phonon python scripts/plot_m1_1b_compare.py \
        --base results/td_phonon/disp_graphene_base.npz \
        --ft   results/td_phonon/disp_graphene_ft.npz \
        --ftg  results/m1_1b/disp_graphene_ftgraphene.npz \
        --dft  results/m1_1b/dft/disp_graphene_dft.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import anomaly_locate as al

CM = 33.35641
LIT = {r"$\Gamma$": 1600.0, "K": 1300.0}


def _load(p):
    return dict(np.load(ROOT / p if not Path(p).is_absolute() else p, allow_pickle=True))


def _wlabel(d, name):
    return al.branch_freq_at_label(d["distances"], d["frequencies"],
                                   d["label_positions"], d["labels"], name) * CM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--ftg", required=True)
    ap.add_argument("--dft", required=True)
    ap.add_argument("--out", default="results/figures/m1_1b_graphene_compare.png")
    a = ap.parse_args()

    series = [("foundation MACE", _load(a.base), "#d1495b", "-"),
              ("FC-distilled (bulk)", _load(a.ft), "#e8a33d", "-"),
              ("FC-distilled (graphene)", _load(a.ftg), "#2a9d4a", "-"),
              ("DFT (QE)", _load(a.dft), "#000000", "--")]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, d, c, ls in series:
        f = d["frequencies"] * CM
        ax.plot(d["distances"], f[:, 0], color=c, ls=ls, lw=1.4, alpha=0.9, label=name)
        ax.plot(d["distances"], f[:, 1:], color=c, ls=ls, lw=1.4, alpha=0.9)

    base = series[0][1]
    labels = [str(x) for x in base["labels"]]
    for x in base["label_positions"]:
        ax.axvline(x, color="0.7", lw=0.5, zorder=0)
    ax.set_xticks(base["label_positions"]); ax.set_xticklabels(labels)
    ax.set_xlim(base["distances"][0], base["distances"][-1])
    for name in (r"$\Gamma$", "K"):
        if name in labels:
            j = labels.index(name)
            ax.plot([base["label_positions"][j]], [LIT[name]], marker="*", ms=13,
                    color="#3b6ea5", zorder=6)
    ax.set_ylabel("Frequency (cm$^{-1}$)")
    ax.set_ylim(bottom=min(0, base["frequencies"].min() * CM))
    ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
    ax.set_title("M1.1b: graphene-specific distillation vs the Gamma/K cusp")
    fig.tight_layout()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf"))

    # table + gap-closing
    print(f"{'model':26s} {'Gamma-E2g':>10s} {'K-A1p':>8s}")
    vals = {}
    for name, d, _, _ in series:
        wG, wK = _wlabel(d, r"$\Gamma$"), _wlabel(d, "K")
        vals[name] = (wG, wK)
        print(f"{name:26s} {wG:9.0f} {wK:8.0f}")
    print(f"{'literature':26s} {LIT[chr(0)] if False else 1600:9.0f} {1300:8.0f}")
    fG = vals["foundation MACE"][0]; dG = vals["DFT (QE)"][0]
    gG = vals["FC-distilled (graphene)"][0]; bG = vals["FC-distilled (bulk)"][0]
    print("\n--- Gamma-E2g gap closing (foundation -> DFT) ---")
    print(f"  bulk FT closed:     {100*(bG-fG)/(dG-fG):5.0f}%   ({fG:.0f} -> {bG:.0f}, DFT {dG:.0f})")
    print(f"  graphene FT closed: {100*(gG-fG)/(dG-fG):5.0f}%   ({fG:.0f} -> {gG:.0f}, DFT {dG:.0f})")

    # cusp sharpness (top-branch two-sided kink |dv|) -- shape recovery, not just freq
    print("\n--- top-branch kink |dv| at Gamma/K (cusp sharpness; closer to DFT = sharper) ---")
    print(f"{'model':26s} {'kink_Gamma':>11s} {'kink_K':>8s}")
    for name, d, _, _ in series:
        ks = {k["label"]: k for k in al.high_sym_kinks(
            d["distances"], d["frequencies"], d["label_positions"], d["labels"])}
        kG = ks.get(r"$\Gamma$", {}).get("kink_strength", float("nan"))
        kK = ks.get("K", {}).get("kink_strength", float("nan"))
        print(f"{name:26s} {kG:11.1f} {kK:8.1f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
