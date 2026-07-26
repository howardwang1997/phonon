"""Compare the Line-A reuse and Line-B retrain deployments with DFT.

The figure uses Fermi--Dirac smearing in Ry directly; no electronic-temperature
conversion is shown.

    conda run -n phonon python scripts/smearing_kink/plot_reuse_vs_retrain_vs_dft.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FD = ROOT / "results" / "graphene_kohn_fd"
OUT = ROOT / "results" / "smearing_kink"
LABELS = ["Γ", "M", "K", "Γ"]


line_a = np.load(FD / "deploy_lineA_bands.npz", allow_pickle=True)
line_b = np.load(FD / "deploy_lineB_bands.npz", allow_pickle=True)

x = line_a["x"]
tick = line_a["tick"]
dgs = [str(value) for value in line_a["dgs"]]
np.testing.assert_allclose(x, line_b["x"])
np.testing.assert_allclose(tick, line_b["tick"])
if dgs != [str(value) for value in line_b["dgs"]]:
    raise ValueError("Line A and Line B use different smearing grids")
np.testing.assert_allclose(line_a["dft_bands"], line_b["dft_bands"])
np.testing.assert_allclose(line_a["dft_kink"], line_b["dft_kink"])

dft_bands = line_a["dft_bands"]
reuse_bands = line_a["mlip_bands"]
retrain_bands = line_b["mlip_bands"]
dft_kink = line_a["dft_kink"]
reuse_kink = line_a["mlip_kink"]
retrain_kink = line_b["mlip_kink"]

idx = {dg: i for i, dg in enumerate(dgs)}
pick = [dg for dg in ("0.01", "0.04", "0.08") if dg in idx]
if len(pick) != 3:
    raise ValueError(f"Expected 0.01/0.04/0.08 Ry data, found {dgs}")

fig = plt.figure(figsize=(17, 5.8))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15])

for panel, dg in enumerate(pick):
    ax = fig.add_subplot(gs[0, panel])
    i = idx[dg]
    # Plot branches without labels; the legend is a single figure-level legend.
    ax.plot(x, dft_bands[i], color="#222", lw=1.3, alpha=0.9)
    ax.plot(x, reuse_bands[i], color="#c0392b", lw=1.1, ls="--", alpha=0.95)
    ax.plot(x, retrain_bands[i], color="#2471a3", lw=1.0, ls=":", alpha=0.95)
    for xt in tick[1:-1]:
        ax.axvline(xt, color="#e7e7e7", lw=0.7)
    ax.set_xticks(tick)
    ax.set_xticklabels(LABELS)
    ax.set_xlim(tick[0], tick[-1])
    ax.set_ylim(1240, 1620)
    ax.set_title(f"smearing = {dg} Ry", fontsize=10)
    if panel == 0:
        ax.set_ylabel("frequency [cm$^{-1}$]", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

smearing = np.array([float(dg) for dg in dgs])
ax = fig.add_subplot(gs[0, 3])
ax.plot(smearing, dft_kink, "o-", color="#222", lw=1.4, ms=4)
ax.plot(smearing, reuse_kink, "s--", color="#c0392b", lw=1.2, ms=4)
ax.plot(smearing, retrain_kink, "^:", color="#2471a3", lw=1.2, ms=4)
ax.set_xscale("log")
smearing_ticks = [0.01, 0.02, 0.04, 0.08, 0.14]
ax.set_xticks(smearing_ticks)
ax.set_xticklabels([f"{value:g}" for value in smearing_ticks])
ax.set_xlim(0.009, 0.155)
ax.set_xlabel("Fermi–Dirac smearing (Ry)", fontsize=9)
ax.set_ylabel("K-point Kohn kink [cm$^{-1}$]", fontsize=9)
ax.set_title("Kohn anomaly melts with smearing", fontsize=10)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

reuse_mae = float(np.mean(np.abs(reuse_kink - dft_kink)))
retrain_mae = float(np.mean(np.abs(retrain_kink - dft_kink)))
fig.suptitle(
    "Reuse and retrain both reproduce the finite-supercell DFT Kohn anomaly "
    f"(kink MAE {reuse_mae:.2f}/{retrain_mae:.2f} cm$^{{-1}}$)",
    fontsize=11.5,
    y=0.97,
)
legend_handles = [
    Line2D([0], [0], color="#222", lw=1.5, marker="o", ms=4, label="DFT"),
    Line2D([0], [0], color="#c0392b", lw=1.2, ls="--", marker="s", ms=4,
           label="Line A: reuse v11"),
    Line2D([0], [0], color="#2471a3", lw=1.2, ls=":", marker="^", ms=4,
           label="Line B: retrain on fd data"),
]
fig.legend(handles=legend_handles, frameon=False, fontsize=8.5, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 0.91))
fig.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.78, wspace=0.28)

OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "reuse_vs_retrain_vs_dft.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(
    "wrote",
    out,
    "| kink MAE reuse/retrain",
    f"{reuse_mae:.2f}/{retrain_mae:.2f}",
)
