"""Line A Kohn-anomaly overlay (FIXED legend): high-frequency optical region, all
smearings overlaid as a colour gradient (colorbar = degauss/T_el), no in-axes legend
that overlaps the curves. Two panels — DFT (solid) | MLIP+long-range healing (dashed).

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
dg_arr = np.array([float(dg) for dg in dgs])     # degauss (Ry), for the colourbar
cmap = plt.cm.viridis
norm = plt.Normalize(dg_arr.min(), dg_arr.max())


def style(ax):
    for xt in tick[1:-1]:
        ax.axvline(xt, color="#ececec", lw=0.7)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.set_xticks(tick); ax.set_xticklabels(LABELS)
    ax.set_ylim(1240, 1620); ax.set_xlim(tick[0], tick[-1])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, (axD, axM) = plt.subplots(1, 2, figsize=(15, 5.6))
for i, dg in enumerate(dgs):
    col = cmap(norm(float(dg)))                  # dark(viridis)=sharp, yellow=broad
    axD.plot(x, dft_b[i], color=col, lw=1.4, alpha=0.92)
    axM.plot(x, mlip_b[i], color=col, lw=1.3, alpha=0.92, ls="--")
for ax in (axD, axM):
    style(ax)
axD.set_title("DFT (PBE fc₂, Fermi-Dirac)", fontsize=11)
axM.set_title("MLIP + long-range (reuse v11 + healing B)", fontsize=11)
axD.set_ylabel("frequency [cm$^{-1}$]", fontsize=10)

# single shared colourbar (degauss -> T_el), off to the right; replaces the in-axes legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, ax=axM, pad=0.06, fraction=0.03, aspect=30)
cbar.set_label("degauss (Ry)  →  T$_{el}$ (K)", fontsize=9)
ticks_dg = np.array([0.01, 0.03, 0.06, 0.10, 0.14])
cbar.set_ticks(ticks_dg)
cbar.set_ticklabels([f"{g:.2f}\n({g*157887:.0f} K)" for g in ticks_dg])

mae = float(np.mean(np.abs(d["mlip_kink"] - d["dft_kink"])))
fig.suptitle("graphene Kohn anomaly (high-frequency overlay): sharp melt with T$_{el}$  —  "
             f"DFT vs MLIP+long-range, kink MAE {mae:.2f} cm$^{{-1}}$",
             fontsize=11.5, y=1.01)
fig.tight_layout()
out = OUT / "lineA_kohn_zoom.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out, "| kink MAE", round(mae, 2))
