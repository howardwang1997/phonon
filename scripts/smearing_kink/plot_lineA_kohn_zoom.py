"""Line A Kohn-anomaly ZOOM: high-frequency optical region only, near the anomaly
points (Gamma E2g/G-band ~1575 and K iTO/A1' ~1300). No low-frequency clutter.
DFT vs MLIP+long-range at 3 electronic temperatures, showing the anomaly + its melting.

    conda run -n phonon python scripts/smearing_kink/plot_lineA_kohn_zoom.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FD = ROOT / "results" / "graphene_kohn_fd"
OUT = ROOT / "results" / "smearing_kink"
LABELS = ["Γ", "M", "K", "Γ"]

d = np.load(FD / "deploy_lineA_bands.npz", allow_pickle=True)
x = d["x"]; tick = list(d["tick"])
dgs = [str(s) for s in d["dgs"]]
dft_b = d["dft_bands"]; mlip_b = d["mlip_bands"]
idx = {dg: i for i, dg in enumerate(dgs)}
pick = [dg for dg in ["0.01", "0.04", "0.08"] if dg in idx]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
for ax, dg in zip(axes, pick):
    i = idx[dg]; T = float(dg) * 157887
    # high-frequency optical region only: the top branches hosting the Kohn anomaly
    ax.plot(x, dft_b[i], color="#1f1f1f", lw=1.6, alpha=0.9, label="DFT (PBE fc₂)")
    ax.plot(x, mlip_b[i], color="#c0392b", lw=1.3, ls="--", alpha=0.95,
            label="MLIP + long-range (reuse v11)")
    for xt in tick[1:-1]:
        ax.axvline(xt, color="#e4e4e4", lw=0.7)
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_xticks(tick); ax.set_xticklabels(LABELS)
    ax.set_ylim(1240, 1620)              # high-freq optical only — no acoustic
    ax.set_xlim(tick[0], tick[-1])
    ax.set_title(f"degauss = {dg}   (T$_{{el}}$ = {T:.0f} K)", fontsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if dg == pick[0]:
        ax.set_ylabel("frequency [cm$^{-1}$]", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    # annotate the two Kohn-anomaly points
    ax.annotate("G-band\n(E2g, Γ)", xy=(tick[0], 1575), xytext=(tick[0] + 0.18 * (tick[-1] - tick[0]), 1500),
                fontsize=7, color="#444", ha="left")
    ax.annotate("Kohn anomaly\n(iTO, K)", xy=(tick[2], 1290), xytext=(tick[2] - 0.30 * (tick[-1] - tick[0]), 1330),
                fontsize=7, color="#444", ha="left")

fig.suptitle("graphene Kohn anomaly (high-frequency zoom): Γ E2g + K iTO  —  "
             "DFT vs MLIP+long-range, melting with T$_{el}$",
             fontsize=12, y=1.01)
fig.tight_layout()
out = OUT / "lineA_kohn_zoom.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
