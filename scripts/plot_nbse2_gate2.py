"""V-Q3 Gate #2 figure: NbSe2 monolayer phonon dispersion (Gamma-M-K-Gamma) for
DFT, foundation MACE, and the distilled-FT MACE. The CDW soft mode (negative
frequencies along Gamma-M) is present in DFT and recovered by the distilled-FT,
but missed by the foundation model -> the Gate #2 result in one panel."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(f):
    d = np.load(ROOT / f, allow_pickle=True)
    return d["distances"], d["frequencies"], d["label_positions"], d["labels"]


series = [
    ("DFT (this work)", "results/vq3/disp_nbse2_dft.npz", "k", "-", 1.4),
    ("foundation MACE", "results/vq3/disp_nbse2_base_sc3.npz", "tab:red", "--", 1.0),
    ("distilled-FT (NbSe$_2$ DFT fc$_2$)", "results/vq3/disp_nbse2_ft_sc3.npz", "tab:green", "-", 1.0),
]

fig, ax = plt.subplots(figsize=(6.6, 4.3))
ref = None
for lab, f, c, ls, lw in series:
    try:
        dist, freq, lp, labs = load(f)
        if ref is None:
            ref = (lp, labs)
        for b in range(freq.shape[1]):
            ax.plot(dist, freq[:, b], color=c, ls=ls, lw=lw, alpha=0.75,
                    label=lab if b == 0 else None)
        print(f"{lab}: min freq = {freq.min():.2f} THz")
    except Exception as e:  # noqa
        print(f"skip {f}: {e}")

ax.axhline(0.0, color="gray", lw=0.8, zorder=0)
ax.axhspan(ax.get_ylim()[0], 0.0, color="tab:blue", alpha=0.05, zorder=0)
lp, labs = ref
ax.set_xticks(lp)
ax.set_xticklabels([l.replace(r"$\Gamma$", "Γ") for l in labs])
for x in lp:
    ax.axvline(x, color="gray", lw=0.4, alpha=0.5)
ax.set_xlim(float(lp[0]), float(lp[-1]))
ax.set_ylabel("frequency (THz)")
ax.set_title("NbSe$_2$ Gate #2: DFT distillation recovers the CDW soft mode")
ax.annotate("CDW soft mode\n(DFT & FT capture it,\nfoundation misses it)",
            xy=(0.16, 0.04), xycoords="axes fraction", fontsize=8,
            color="tab:blue", va="bottom")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
fig.tight_layout()
out = ROOT / "results" / "figures" / "nbse2_gate2.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out)
