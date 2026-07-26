"""Line A (reuse v11 + tuned Friedel long-range) vs DFT, Piscanec-2004-style:
the graphene Kohn anomaly at Gamma (E2g/G-band) and K (iTO/A1'), and its melting
with Fermi--Dirac smearing. Reads deploy_lineA_bands.npz.

    conda run -n phonon python scripts/smearing_kink/plot_lineA_vs_dft.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
FD = ROOT / "results" / "graphene_kohn_fd"
OUT = ROOT / "results" / "smearing_kink"
LABELS = ["Γ", "M", "K", "Γ"]

d = np.load(FD / "deploy_lineA_bands.npz", allow_pickle=True)
x = d["x"]; tick = d["tick"]
dgs = [str(s) for s in d["dgs"]]
dft_b = d["dft_bands"]; mlip_b = d["mlip_bands"]
dft_k = d["dft_kink"]; mlip_k = d["mlip_kink"]; bb_k = float(d["backbone_kink"])
idx = {dg: i for i, dg in enumerate(dgs)}
pick = [dg for dg in ["0.01", "0.04", "0.08"] if dg in idx]

fig = plt.figure(figsize=(17, 5.8))
# row: 3 dispersion panels (sharp/mid/broad smearing) + 1 kink-vs-smearing panel
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15])
for j, dg in enumerate(pick):
    ax = fig.add_subplot(gs[0, j]); i = idx[dg]
    # Plot the six branches without labels.  Labelling a 2-D array makes
    # Matplotlib add one legend entry per branch and obscures the spectrum.
    ax.plot(x, dft_b[i], color="#1f1f1f", lw=1.3, alpha=0.9)
    ax.plot(x, mlip_b[i], color="#c0392b", lw=1.0, ls="--", alpha=0.95)
    for xt in tick[1:-1]:
        ax.axvline(xt, color="#e2e2e2", lw=0.6)
    ax.axhline(0, color="#bbb", lw=0.6)
    ax.set_xticks(tick); ax.set_xticklabels(LABELS)
    ax.set_ylim(-15, 1620); ax.set_xlim(tick[0], tick[-1])
    ax.set_title(f"smearing = {dg} Ry", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if j == 0:
        ax.set_ylabel("frequency [cm$^{-1}$]", fontsize=9)
# kink_K vs smearing panel
ax = fig.add_subplot(gs[0, 3])
smearing = np.array([float(dg) for dg in dgs])
ax.plot(smearing, dft_k, "o-", color="#1f1f1f", lw=1.3, ms=4)
ax.plot(smearing, mlip_k, "s--", color="#c0392b", lw=1.1, ms=4)
ax.axhline(bb_k, color="#888", ls=":", lw=1.0)
ax.set_xscale("log")
smearing_ticks = [0.01, 0.02, 0.04, 0.08, 0.14]
ax.set_xticks(smearing_ticks)
ax.set_xticklabels([f"{value:g}" for value in smearing_ticks])
ax.set_xlim(0.009, 0.155)
ax.set_xlabel("Fermi–Dirac smearing (Ry)", fontsize=9)
ax.set_ylabel("K-point Kohn kink [cm$^{-1}$]", fontsize=9)
ax.set_title("Kohn anomaly melts with smearing", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
mae = float(np.mean(np.abs(mlip_k - dft_k)))
fig.suptitle("Line A: MLIP + long-range reproduces the graphene Kohn anomaly "
             f"(reuse-v11 + tuned Friedel)  —  kink MAE {mae:.2f} cm$^{{-1}}$",
             fontsize=11.5, y=0.97)
legend_handles = [
    Line2D([0], [0], color="#1f1f1f", lw=1.5, marker="o", ms=4,
           label="DFT (PBE finite-displacement fc₂)"),
    Line2D([0], [0], color="#c0392b", lw=1.2, ls="--", marker="s", ms=4,
           label="MLIP + long-range (reuse v11)"),
    Line2D([0], [0], color="#888", lw=1.1, ls=":",
           label="backbone only (smearing-blind)"),
]
fig.legend(handles=legend_handles, frameon=False, fontsize=8.5, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 0.91))
fig.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.78, wspace=0.28)
out = OUT / "lineA_vs_dft_kohn.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out, "| kink MAE", round(mae, 2))
