"""V-Q1 figure: graphene K-A1' Kohn anomaly emerges as the DFT supercell is
converged. 5x5 (K interpolated) looks smooth & high; the K-commensurate 6x6
reveals a sharp cusp at K~1292 cm-1. Top two optical branches on M-Gamma-K-M."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
cm = 33.35641


def load(f):
    d = np.load(ROOT / f, allow_pickle=True)
    return d["distances"], d["frequencies"] * cm, d["label_positions"], d["labels"]


series = [
    ("5×5 (K interp., kink 0.8)", "results/m1_1b/dft/disp_graphene_dft.npz", "tab:orange", "--"),
    ("7×7 (K interp., kink 8.3)", "results/vq1/disp_graphene_sc7_k5.npz", "tab:blue", "-."),
    ("6×6 (K commensurate, kink 14.4)", "results/vq1/disp_graphene_sc6_k6.npz", "tab:green", "-"),
]
fig, ax = plt.subplots(figsize=(6.6, 4.3))
ref = None
for lab, f, c, ls in series:
    try:
        dist, freq, lp, labs = load(f)
        ref = (lp, labs)
        # top two optical branches
        top = np.sort(freq, axis=1)[:, -2:]
        for b in range(top.shape[1]):
            ax.plot(dist, top[:, b], color=c, ls=ls, lw=1.4,
                    label=lab if b == 1 else None)
    except Exception as e:  # noqa
        print(f"skip {f}: {e}")
lp, labs = ref
ax.axhline(1300, color="gray", lw=0.7, ls=":")
ax.text(dist.max() * 0.5, 1305, "lit. K-A₁′ ~1300", fontsize=7, color="gray")
ax.set_xticks(lp)
ax.set_xticklabels([l.replace(r"$\Gamma$", "Γ") for l in labs])
for x in lp:
    ax.axvline(x, color="gray", lw=0.4, alpha=0.4)
ax.set_xlim(float(lp[0]), float(lp[-1]))
ax.set_ylim(1250, 1620)
ax.set_ylabel("frequency (cm$^{-1}$)")
ax.set_title("Graphene K-A$_1'$ Kohn anomaly: real, under-resolved by 5×5 DFT")
ax.legend(loc="lower center", fontsize=8, framealpha=0.9)
fig.tight_layout()
out = ROOT / "results" / "figures" / "vq1_converged_K.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out)
