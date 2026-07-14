"""Combined graphene: 0K (DFT vs MLIP+LR) + 300K (DFT-MD-TDEP vs MLIP-TDEP) side-by-side."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SK = ROOT / "results" / "smearing_kink"
TD = ROOT / "results" / "td_phonon"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["$\\Gamma$", "M", "K", "$\\Gamma$"]

# ---- 0K panel: DFT fc2 vs MLIP backbone+LR (from deploy_vs_dft_graphene_v11.npz) ----
dep = np.load(SK / "deploy_vs_dft_graphene_v11.npz", allow_pickle=True)
qs0 = dep["qs"]
seglen = np.linalg.norm(np.diff(qs0, axis=0), axis=1)
x0 = np.concatenate([[0], np.cumsum(seglen)])
seg = len(x0) // 3
tick0 = [x0[0], x0[seg-1], x0[2*seg-1], x0[-1]]

# 0K = T_el=789K (dg0.005, sharpest = closest to 0K harmonic)
dft_0K = dep["dft_T789"]   # DFT fc2 band structure (cm⁻¹)
mlip_0K = dep["mlip_T789"]  # MLIP backbone + Friedel LR (cm⁻¹)
bb_0K = dep["backbone"]      # MLIP backbone alone (smearing-blind)

# ---- 300K panel: DFT-MD-TDEP vs MLIP-TDEP ----
dft300 = np.load(TD / "graphene_dft_tdep.npz", allow_pickle=True)
mlip300 = np.load(TD / "td_graphene_ft.npz", allow_pickle=True)
x300_dft = dft300["T300_dist"]
dft_300K = dft300["T300_freq"] * 33.356   # THz -> cm⁻¹
x300_mlip = mlip300["T300_dist"]
mlip_300K = mlip300["T300_freq"] * 33.356
ticks300 = dft300["label_positions"]
labels300 = [str(l) for l in dft300["labels"]]

# ---- plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

# Panel 1: 0K
ax1.plot(x0, dft_0K, color="#222", lw=2.0, alpha=0.9, label="DFT fc₂ (target)")
ax1.plot(x0, mlip_0K, color="#c0504d", lw=1.5, ls="--", alpha=0.95, label="MLIP + long-range Friedel")
ax1.plot(x0, bb_0K, color="#2f6f9f", lw=1.0, alpha=0.4, label="backbone (smearing-blind)")
ax1.set_xticks(tick0); ax1.set_xticklabels(LABELS)
for xt in tick0[1:-1]: ax1.axvline(xt, color="#ccc", lw=0.6)
ax1.axhline(0, color="#aaa", lw=0.6)
ax1.set_ylabel("frequency [cm$^{-1}$]", fontsize=11)
ax1.set_ylim(-30, 1650)
ax1.set_xlim(tick0[0], tick0[-1])
mae0 = np.mean(np.abs(mlip_0K - dft_0K))
ax1.set_title(f"graphene 0K (harmonic)  MAE = {mae0:.1f} cm$^{{-1}}$", fontsize=11)
ax1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
for s in ("top", "right"): ax1.spines[s].set_visible(False)

# Panel 2: 300K
ax2.plot(x300_dft, dft_300K, color="#222", lw=2.0, alpha=0.9, label="DFT-MD-TDEP (300K)")
ax2.plot(x300_mlip, mlip_300K, color="#c0504d", lw=1.5, ls="--", alpha=0.95, label="MLIP-TDEP (300K)")
ax2.set_xticks(ticks300); ax2.set_xticklabels(labels300)
for xt in ticks300[1:-1]: ax2.axvline(xt, color="#ccc", lw=0.6)
ax2.axhline(0, color="#aaa", lw=0.6)
ax2.set_ylim(-30, 1650)
ax2.set_xlim(min(x300_dft[0], x300_mlip[0]), max(x300_dft[-1], x300_mlip[-1]))
# MAE needs interpolation to common grid
from numpy import interp
dft_interp = np.array([np.interp(x300_mlip, x300_dft, dft_300K[:, b]) for b in range(dft_300K.shape[1])]).T
mae300 = np.nanmean(np.abs(mlip_300K - dft_interp))
ax2.set_title(f"graphene 300K (anharmonic)  MAE = {mae300:.0f} cm$^{{-1}}$", fontsize=11)
ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
for s in ("top", "right"): ax2.spines[s].set_visible(False)

fig.suptitle("graphene: DFT vs MLIP+fine-tuning — 0K harmonic vs 300K anharmonic phonon spectrum", fontsize=12, y=1.01)
fig.tight_layout()
out = SK / "graphene_0K_vs_300K.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
