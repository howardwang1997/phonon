"""Graphene Kohn anomaly: 15-point fd smearing analysis + 4 figures.

Figures:
1. kink_K vs T_el (15 pts, DFT + MLIP+healing + backbone)
2. High-freq dispersion overlay (15 curves, DFT | MLIP)
3. K-iTO frequency vs T_el
4. Per-branch smearing sensitivity heatmap
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FD = ROOT / "results" / "graphene_kohn_fd"
OUT = ROOT / "results" / "smearing_kink"
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))
import friedel_module as fm
import phonopy

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

LABELS = ["Γ", "M", "K", "Γ"]
PTS = [np.array([0., 0., 0.]), np.array([.5, 0., 0.]),
      np.array([1/3, 1/3, 0.]), np.array([0., 0., 0.])]
CM = 33.35641

# --- load all 15 yamls ---
all_dgs = sorted([float(f.stem.split("_dg")[1].split("_")[0])
                  for f in FD.glob("graphene_sc6_dg*_phonopy.yaml")])
print(f"Found {len(all_dgs)} smearing points: {all_dgs}")

data = {}  # dg -> {kink_K, kink_G, K_iTO, G_E2g, band_x, band_freq}
for dg in all_dgs:
    dg_str = f"{dg}"
    yml = FD / f"graphene_sc6_dg{dg_str}_phonopy.yaml"
    if not yml.exists():
        # try alternate formatting
        for alt in [f"{dg:.3f}", f"{dg:.2f}"]:
            yml = FD / f"graphene_sc6_dg{alt}_phonopy.yaml"
            if yml.exists():
                break
    if not yml.exists():
        print(f"  skip dg{dg}: file not found"); continue
    ph = fm.load_ph(str(yml))
    kK, wK, kG = fm.kink_of(ph, ph.force_constants)

    # K-iTO and G-E2g
    ph2 = phonopy.load(str(yml), is_compact_fc=False)
    ph2.run_qpoints([np.array([1/3, 1/3, 0.]), np.array([0., 0., 0.])],
                    with_dynamical_matrices=False)
    f = np.array(ph2.get_qpoints_dict()["frequencies"]) * CM
    K_iTO = float(f[0].max())
    G_E2g = float(f[1].max())

    # full band
    qs = []
    for i in range(len(PTS) - 1):
        for j in range(1, 41):
            qs.append(PTS[i] + (PTS[i + 1] - PTS[i]) * j / 40)
    qs = np.array(qs)
    ph2.run_qpoints(qs, with_dynamical_matrices=False)
    freq = np.array(ph2.get_qpoints_dict()["frequencies"]) * CM
    seglen = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(seglen)])
    seg = len(x) // 3
    tick = [x[0], x[seg - 1], x[2 * seg - 1], x[-1]]

    data[dg] = dict(kink_K=kK, kink_G=kG, K_iTO=K_iTO, G_E2g=G_E2g,
                    x=x, freq=freq, tick=tick)
    print(f"  dg{dg}: kink_K={kK:.2f}  K_iTO={K_iTO:.1f}  G_E2g={G_E2g:.1f}")

dgs = sorted(data.keys())
Tel = np.array([dg * 157887 for dg in dgs])

# --- MLIP+healing deploy data (from existing deploy_lineA_bands.npz) ---
bands_npz = FD / "deploy_lineA_bands.npz"
mlip_data = None
if bands_npz.exists():
    d = np.load(bands_npz, allow_pickle=True)
    mlip_dgs = [str(s) for s in d["dgs"]]
    mlip_kink = d["mlip_kink"]
    mlip_dft_kink = d["dft_kink"]
    mlip_bands = d["mlip_bands"]
    mlip_dft_bands = d["dft_bands"]
    mlip_x = d["x"]
    mlip_data = True

# --- healing law ---
hl = json.loads((FD / "healing_law.json").read_text())

cmap = plt.cm.viridis
norm = Normalize(min(dgs), max(dgs))

# ================================================================ Figure 1
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(Tel, [data[dg]["kink_K"] for dg in dgs], "o-", color="#222", lw=1.8,
        ms=5, label="DFT (PBE fc₂, Fermi-Dirac)", zorder=5)
if mlip_data:
    mlip_T = np.array([float(dg) * 157887 for dg in mlip_dgs])
    ax.plot(mlip_T, mlip_kink, "s--", color="#c0392b", lw=1.4, ms=4,
            label=f"MLIP + long-range (healing B, MAE {np.mean(np.abs(mlip_kink - mlip_dft_kink)):.2f})")
    ax.axhline(float(d["backbone_kink"]), color="#888", ls=":", lw=1,
              label="backbone alone (smearing-blind)")
# annotate T*
Tstar = hl["Tstar"]
ax.axvline(Tstar, color="#2980b9", ls="--", lw=0.8, alpha=0.7)
ax.text(Tstar * 1.05, 10, f"T* ≈ {Tstar:.0f} K", color="#2980b9", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("T$_{el}$ [K]  (degauss × 157887)", fontsize=11)
ax.set_ylabel("K-point Kohn kink [cm$^{-1}$]", fontsize=11)
ax.set_title("Graphene Kohn anomaly: sharp melt with T$_{el}$ (15 fd smearing points)", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "gr_kink_melt_15pt.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT / 'gr_kink_melt_15pt.png'}")

# ================================================================ Figure 2
fig, (axD, axM) = plt.subplots(1, 2, figsize=(15, 5.8))
for dg in dgs:
    col = cmap(norm(dg))
    d = data[dg]
    axD.plot(d["x"], d["freq"], color=col, lw=1.3, alpha=0.9)
    if mlip_data:
        dg_str = f"{dg}"
        if dg_str in mlip_dgs:
            idx = mlip_dgs.index(dg_str)
            axM.plot(mlip_x, mlip_bands[idx], color=col, lw=1.2, alpha=0.9, ls="--")
for ax in (axD, axM):
    tick = data[dgs[0]]["tick"]
    for xt in tick[1:-1]:
        ax.axvline(xt, color="#ececec", lw=0.7)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.set_xticks(tick); ax.set_xticklabels(LABELS)
    ax.set_ylim(1240, 1620)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axD.set_title("DFT (PBE fc₂, Fermi-Dirac)", fontsize=11)
axM.set_title("MLIP + long-range (healing B)", fontsize=11)
axD.set_ylabel("frequency [cm$^{-1}$]", fontsize=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, ax=[axD, axM], pad=0.06, fraction=0.03, aspect=28)
cbar.set_label("degauss (Ry)", fontsize=9)
fig.suptitle("Graphene Kohn anomaly (high-freq overlay): 15 fd smearings — sharp melt with T$_{el}$",
             fontsize=11.5, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "gr_hf_overlay_15pt.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT / 'gr_hf_overlay_15pt.png'}")

# ================================================================ Figure 3
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(Tel, [data[dg]["K_iTO"] for dg in dgs], "o-", color="#222", lw=1.8,
        ms=5, label="K-iTO (DFT)")
# bare estimate: the highest-T (most smeared) K-iTO ≈ bare
bare = data[max(dgs)]["K_iTO"]
ax.axhline(bare, color="#888", ls=":", label=f"bare K-iTO ≈ {bare:.0f} cm⁻¹ (high-T$_{{el}}$)")
# annotate softening at low T
soft = data[min(dgs)]["K_iTO"]
ax.annotate(f"Kohn softening\n{soft:.0f} → {bare:.0f} cm⁻¹",
            xy=(Tel[0], soft), xytext=(Tel[0] * 2, soft - 40),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="#666"))
ax.set_xscale("log")
ax.set_xlabel("T$_{el}$ [K]", fontsize=11)
ax.set_ylabel("K-point iTO frequency [cm$^{-1}$]", fontsize=11)
ax.set_title("K-point optical phonon: Kohn softening vs T$_{el}$", fontsize=11)
ax.legend(frameon=False, fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "gr_kito_vs_tel.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT / 'gr_kito_vs_tel.png'}")

# ================================================================ Figure 4
# Per-branch smearing sensitivity: heatmap of freq(dg) for each branch at K
fig, ax = plt.subplots(figsize=(8, 5))
# collect K-point frequencies per branch
branch_freqs = np.zeros((len(dgs), 6))  # 6 branches
for i, dg in enumerate(dgs):
    branch_freqs[i] = np.sort(data[dg]["freq"][40])  # K = q index 40
# plot as line per branch
colors_b = plt.cm.tab10(np.linspace(0, 0.85, 6))
for b in range(6):
    ax.plot(Tel, branch_freqs[:, b], "o-", color=colors_b[b], lw=1.5, ms=3,
            label=f"branch {b+1}")
ax.set_xscale("log")
ax.set_xlabel("T$_{el}$ [K]", fontsize=11)
ax.set_ylabel("K-point frequency [cm$^{-1}$]", fontsize=11)
ax.set_title("All 6 graphene branches @ K vs T$_{el}$: only Kohn channels respond", fontsize=10.5)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
ax.axhline(0, color="#aaa", lw=0.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "gr_branches_vs_tel.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT / 'gr_branches_vs_tel.png'}")

print(f"\nAll 4 figures written to {OUT}/")
