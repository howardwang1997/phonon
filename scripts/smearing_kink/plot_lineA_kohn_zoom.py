"""Line A Kohn-anomaly overlay: high-frequency optical region, all smearings overlaid.
Two panels — DFT (all T_el overlaid) and MLIP+long-range (all T_el overlaid) — so the
Kohn anomaly and its sharp melt with electronic temperature is visible as a family of
curves (dark = sharp Fermi surface / anomaly present; light = broad / anomaly melted).

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
Tel = np.array([float(dg) * 157887 for dg in dgs])
cmap = plt.cm.Blues
n = len(dgs)


def style(ax):
    for xt in tick[1:-1]:
        ax.axvline(xt, color="#e6e6e6", lw=0.7)
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_xticks(tick); ax.set_xticklabels(LABELS)
    ax.set_ylim(1240, 1620); ax.set_xlim(tick[0], tick[-1])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, (axD, axM) = plt.subplots(1, 2, figsize=(13, 5.6))
for i, dg in enumerate(dgs):
    col = cmap(0.85 - 0.55 * i / max(n - 1, 1))   # dark=sharp, light=broad
    lab = f"dg={dg} (T$_{{el}}$={Tel[i]:.0f} K)" if i in (0, n - 1) else None
    axD.plot(x, dft_b[i], color=col, lw=1.3, alpha=0.9, label=lab)
    axM.plot(x, mlip_b[i], color=col, lw=1.3, alpha=0.9, ls="--", label=lab)
for ax in (axD, axM):
    style(ax)
axD.set_title("DFT (PBE fc₂, Fermi-Dirac)", fontsize=11)
axM.set_title("MLIP + long-range (reuse v11 + healing Friedel)", fontsize=11)
axD.set_ylabel("frequency [cm$^{-1}$]", fontsize=10)
axD.legend(frameon=False, fontsize=7.5, loc="lower right")
axM.legend(frameon=False, fontsize=7.5, loc="lower right")
# annotate the two Kohn-anomaly points on the DFT panel
axD.annotate("G-band (E2g, Γ)", xy=(tick[0], 1575), xytext=(tick[0] + 0.05 * (tick[-1] - tick[0]), 1480),
             fontsize=7.5, color="#333")
axD.annotate("Kohn anomaly\n(iTO, K)", xy=(tick[2], 1290), xytext=(tick[2] - 0.22 * (tick[-1] - tick[0]), 1335),
             fontsize=7.5, color="#333")
mae = float(np.mean(np.abs(d["mlip_kink"] - d["dft_kink"])))
fig.suptitle("graphene Kohn anomaly (high-frequency overlay): sharp melt with T$_{el}$  —  "
             f"DFT vs MLIP+long-range, kink MAE {mae:.2f} cm$^{{-1}}$  (dark=sharp / anomaly present, light=broad / melted)",
             fontsize=11, y=1.01)
fig.tight_layout()
out = OUT / "lineA_kohn_zoom.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out, "| kink MAE", round(mae, 2))
