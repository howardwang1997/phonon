"""Line A (reuse v11 + tuned Friedel long-range) vs DFT, Piscanec-2004-style:
the graphene Kohn anomaly at Gamma (E2g/G-band) and K (iTO/A1'), and its melting
with electronic temperature (smearing = T_el). Reads deploy_lineA_bands.npz.

    conda run -n phonon python scripts/smearing_kink/plot_lineA_vs_dft.py
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
x = d["x"]; tick = d["tick"]
dgs = [str(s) for s in d["dgs"]]
dft_b = d["dft_bands"]; mlip_b = d["mlip_bands"]
dft_k = d["dft_kink"]; mlip_k = d["mlip_kink"]; bb_k = float(d["backbone_kink"])
idx = {dg: i for i, dg in enumerate(dgs)}
pick = [dg for dg in ["0.01", "0.04", "0.08"] if dg in idx]

fig = plt.figure(figsize=(17, 5.6))
# row: 3 dispersion panels (sharp/mid/broad T_el) + 1 kink-vs-T_el panel
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15])
for j, dg in enumerate(pick):
    ax = fig.add_subplot(gs[0, j]); i = idx[dg]; T = float(dg) * 157887
    ax.plot(x, dft_b[i], color="#1f1f1f", lw=1.3, alpha=0.9, label="DFT (PBE fc₂)")
    ax.plot(x, mlip_b[i], color="#c0392b", lw=1.0, ls="--", alpha=0.95,
            label="MLIP+long-range (reuse v11)")
    for xt in tick[1:-1]:
        ax.axvline(xt, color="#e2e2e2", lw=0.6)
    ax.axhline(0, color="#bbb", lw=0.6)
    ax.set_xticks(tick); ax.set_xticklabels(LABELS)
    ax.set_ylim(-15, 1620); ax.set_xlim(tick[0], tick[-1])
    ax.set_title(f"degauss={dg}  (T$_{{el}}$={T:.0f} K)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if j == 0:
        ax.set_ylabel("frequency [cm$^{-1}$]", fontsize=9)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
# kink_K vs T_el panel
ax = fig.add_subplot(gs[0, 3])
Tel = np.array([float(dg) * 157887 for dg in dgs])
ax.plot(Tel, dft_k, "o-", color="#1f1f1f", lw=1.3, ms=4, label="DFT")
ax.plot(Tel, mlip_k, "s--", color="#c0392b", lw=1.1, ms=4, label="MLIP+long-range")
ax.axhline(bb_k, color="#888", ls=":", lw=1.0, label="backbone alone (smearing-blind)")
ax.set_xscale("log")
ax.set_xlabel("T$_{el}$ [K]  (degauss × 157887)", fontsize=9)
ax.set_ylabel("K-point Kohn kink [cm$^{-1}$]", fontsize=9)
ax.set_title("Kohn anomaly melts with T$_{el}$", fontsize=10)
ax.legend(frameon=False, fontsize=7.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
mae = float(np.mean(np.abs(mlip_k - dft_k)))
fig.suptitle("Line A: MLIP + long-range reproduces the graphene Kohn anomaly "
             f"(reuse-v11 + tuned Friedel)  —  kink MAE {mae:.2f} cm$^{{-1}}$",
             fontsize=11.5, y=1.01)
fig.tight_layout()
out = OUT / "lineA_vs_dft_kohn.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out, "| kink MAE", round(mae, 2))
