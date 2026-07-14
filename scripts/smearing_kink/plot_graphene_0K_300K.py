"""Combined graphene: (E) multi-smearing (DFT vs MLIP+LR) + 300K (DFT-MD-TDEP vs MLIP-TDEP)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
SK = ROOT / "results" / "smearing_kink"
TD = ROOT / "results" / "td_phonon"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["$\\Gamma$", "M", "K", "$\\Gamma$"]
POINTS = [np.array([0., 0., 0.]), np.array([.5, 0., 0.]),
          np.array([1/3, 1/3, 0.]), np.array([0., 0., 0.])]

def band_structure(yaml_path, nseg=40):
    import phonopy
    ph = phonopy.load(str(yaml_path), is_compact_fc=False)
    qs = []
    for i in range(len(POINTS) - 1):
        q0, q1 = POINTS[i], POINTS[i + 1]
        for j in range(1, nseg + 1):
            qs.append(q0 + (q1 - q0) * j / nseg)
    qs = np.array(qs)
    ph.run_qpoints(qs, with_dynamical_matrices=False)
    freq = np.array(ph.get_qpoints_dict()["frequencies"]) * 33.356
    seglen = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(seglen)])
    seg = len(x) // 3
    tick = [x[0], x[seg - 1], x[2 * seg - 1], x[-1]]
    return x, freq, tick

# ---- Left: (E) multi-smearing DFT vs MLIP+LR ----
dft_yamls = {
    0.002: ROOT / "results" / "vq_kink6" / "graphene_sc6_dg0.002_phonopy.yaml",
    0.005: ROOT / "results" / "vq_kink6" / "graphene_sc6_dg0.005_phonopy.yaml",
    0.015: ROOT / "results" / "vq_kink6" / "graphene_sc6_dg0.015_phonopy.yaml",
    0.030: ROOT / "results" / "vq_kink6" / "graphene_sc6_dg0.030_phonopy.yaml",
}
dft_data = {}
for dg, yml in sorted(dft_yamls.items()):
    if yml.exists():
        x, f, tick = band_structure(yml)
        dft_data[dg] = (x, f, tick)
        print(f"  DFT dg{dg}: shape {f.shape}")

# MLIP+LR deploy (8x8, T_el=789=dg0.005)
dep = np.load(SK / "deploy_vs_dft_graphene_v11.npz", allow_pickle=True)
qs8 = dep["qs"]
seglen8 = np.linalg.norm(np.diff(qs8, axis=0), axis=1)
x8 = np.concatenate([[0], np.cumsum(seglen8)])
seg8 = len(x8) // 3
tick8 = [x8[0], x8[seg8 - 1], x8[2 * seg8 - 1], x8[-1]]

# ---- Right: 300K ----
dft300 = np.load(TD / "graphene_dft_tdep.npz", allow_pickle=True)
mlip300 = np.load(TD / "td_graphene_ft.npz", allow_pickle=True)
x300_dft = dft300["T300_dist"]
dft_300K = dft300["T300_freq"] * 33.356
x300_mlip = mlip300["T300_dist"]
mlip_300K = mlip300["T300_freq"] * 33.356
ticks300 = dft300["label_positions"]
labels300 = [str(l) for l in dft300["labels"]]

# ---- plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

cmap = plt.cm.Blues
dgs = sorted(dft_data.keys())
nd = len(dgs)
for i, dg in enumerate(dgs):
    x, f, tick = dft_data[dg]
    Tel = dg * 157888
    col = cmap(0.30 + 0.60 * i / max(nd - 1, 1))
    lw = 2.0 if dg == min(dgs) else 1.2
    ls = "-" if dg == min(dgs) else "--"
    ax1.plot(x, f, color=col, lw=lw, ls=ls, alpha=0.85,
            label=f"DFT dg={dg} (T$_{{el}}$={Tel:.0f} K)")

ax1.plot(x8, dep["mlip_T789"], color="#c0504d", lw=1.5, ls="-.", alpha=0.9,
        label="MLIP v11 + LR (dg0.005, 8x8)")

ax1.set_xticks(tick8 if dgs else tick)
ax1.set_xticklabels(LABELS)
for xt in (tick8 if dgs else tick)[1:-1]:
    ax1.axvline(xt, color="#ccc", lw=0.6)
ax1.axhline(0, color="#aaa", lw=0.6)
ax1.set_ylabel("frequency [cm$^{-1}$]", fontsize=11)
ax1.set_ylim(-30, 1650)   # full graphene dispersion (optical G-band ~1615; was 360, cut off)
ax1.set_xlim(0, tick8[-1])
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
           frameon=False, fontsize=7.5, ncol=3)
ax1.set_title("(E): graphene smearing 谱 (DFT multi-smearing + MLIP+LR)", fontsize=10.5)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

ax2.plot(x300_dft, dft_300K, color="#222", lw=2.0, alpha=0.9, label="DFT-MD-TDEP (300K)")
ax2.plot(x300_mlip, mlip_300K, color="#c0504d", lw=1.5, ls="--", alpha=0.95, label="MLIP-TDEP (300K)")
ax2.set_xticks(ticks300)
ax2.set_xticklabels(labels300)
for xt in ticks300[1:-1]:
    ax2.axvline(xt, color="#ccc", lw=0.6)
ax2.axhline(0, color="#aaa", lw=0.6)
ax2.set_ylim(-30, 1650)
xmax300 = max(float(x300_dft[-1]), float(x300_mlip[-1]))
ax2.set_xlim(0, xmax300)
dft_interp = np.array([np.interp(x300_mlip, x300_dft, dft_300K[:, b]) for b in range(dft_300K.shape[1])]).T
mae300 = np.nanmean(np.abs(mlip_300K - dft_interp))
ax2.set_title(f"(L): graphene 300K (anharmonic)  MAE = {mae300:.0f} cm$^{{-1}}$", fontsize=10.5)
ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.suptitle("graphene: DFT vs MLIP+fine-tuning — smearing 谱 vs temperature 谱", fontsize=12, y=1.01)
fig.tight_layout()
out = SK / "graphene_0K_vs_300K.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
