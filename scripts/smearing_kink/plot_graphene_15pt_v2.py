"""Graphene smearing figures with DFT vs MLIP + long-range comparison.
Uses the all-point deployment when available and falls back to the older
9-point MLIP archive while that deployment is still running.
All electronic broadening axes are reported directly as smearing in Ry.
No new compute needed — extracts K-iTO + K-branches from band data."""
import sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FD = ROOT / "results" / "graphene_kohn_fd"
OUT = ROOT / "results" / "smearing_kink"
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))
import friedel_module as fm, phonopy

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

LABELS = ["Γ", "M", "K", "Γ"]
PTS = [np.array([0.,0,0]), np.array([.5,0,0]), np.array([1/3,1/3,0]), np.array([0.,0,0])]
CM = 33.35641

# --- 15-pt DFT data ---
# Keep the original path for each parsed value.  Reconstructing the name from a
# float turns 0.10 into 0.1 and silently drops that data point.
dg_files = {
    float(path.stem.split("_dg")[1].split("_")[0]): path
    for path in FD.glob("graphene_sc6_dg*_phonopy.yaml")
}
all_dgs = sorted(dg_files)
dft = {}
for dg in all_dgs:
    yml = dg_files[dg]
    ph = fm.load_ph(str(yml)); kK = fm.kink_of(ph, ph.force_constants)[0]
    ph2 = phonopy.load(str(yml), is_compact_fc=False)
    qs = []
    for i in range(len(PTS)-1):
        for j in range(1,41): qs.append(PTS[i]+(PTS[i+1]-PTS[i])*j/40)
    qs = np.array(qs)
    ph2.run_qpoints(qs, with_dynamical_matrices=False)
    band = np.array(ph2.get_qpoints_dict()["frequencies"]) * CM
    ph2.run_qpoints([np.array([1/3,1/3,0.])], with_dynamical_matrices=False)
    K_br = np.sort(np.array(ph2.get_qpoints_dict()["frequencies"])[0]) * CM
    dft[dg] = dict(kK=kK, band=band, K_iTO=float(K_br[-1]), K_br=K_br)

# helper: match mlip dg string to dft key
def dft_lookup(dg_str):
    f = float(dg_str)
    for k in dft:
        if abs(k - f) < 1e-6: return dft[k]
    return None

# --- MLIP data: use the 15-point deployment once it is available ---
deploy_all = FD / "deploy_15pt.npz"
if deploy_all.is_file():
    d = np.load(deploy_all, allow_pickle=False)
    mlip_dgs = [str(s) for s in d["dgs"]]
    mlip_bands = np.asarray(d["mlip_bands"])
    mlip_kink = np.asarray(d["mlip_kink_K"])
    dft_kink_matched = np.asarray(d["dft_kink_K"])
    mlip_K_iTO = np.asarray(d["mlip_K_iTO"])
    mlip_K_br = np.asarray(d["mlip_K_branches"])
else:
    d = np.load(FD / "deploy_lineA_bands.npz", allow_pickle=True)
    mlip_dgs = [str(s) for s in d["dgs"]]
    mlip_bands = np.asarray(d["mlip_bands"])
    mlip_kink = np.asarray(d["mlip_kink"])
    dft_kink_matched = np.asarray(d["dft_kink"])
    # Each segment contains 40 points and omits its initial endpoint; K is the
    # final point of the second segment, at zero-based index 79.
    K_idx = 79
    mlip_K_iTO = np.asarray(
        [float(np.sort(mlip_bands[i, K_idx])[-1]) for i in range(len(mlip_dgs))]
    )
    mlip_K_br = np.asarray(
        [np.sort(mlip_bands[i, K_idx]) for i in range(len(mlip_dgs))]
    )

dgs = sorted(dft.keys())
smearing_dft = np.array(dgs)
smearing_mlip = np.array([float(dg) for dg in mlip_dgs])
hl = json.loads((FD / "healing_law.json").read_text())
smearing_star = hl["Tstar"] / 157887
cmap = plt.cm.viridis; norm = Normalize(min(dgs), max(dgs))


def set_smearing_axis(ax, upper):
    """Use readable direct smearing values on a logarithmic axis."""
    ticks = [value for value in (0.01, 0.02, 0.04, 0.08, 0.16) if value <= upper + 1e-12]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{value:g}" for value in ticks])
    ax.set_xlim(0.009, upper * 1.08)
    ax.set_xlabel("Fermi–Dirac smearing (Ry)", fontsize=11)

# ============ Figure 1: kink_K vs smearing ============
fig, ax = plt.subplots(figsize=(8.4, 5.7))
ax.plot(smearing_dft, [dft[dg]["kK"] for dg in dgs], "o-", color="#222", lw=1.8, ms=5, zorder=5,
        label="DFT (15 fd points)")
ax.plot(smearing_mlip, mlip_kink, "s--", color="#c0392b", lw=1.4, ms=5,
        label=f"MLIP + long-range ({len(mlip_dgs)} points)")
mae = np.mean(np.abs(mlip_kink - dft_kink_matched))
ax.text(0.02, 0.02, f"kink_K MAE = {mae:.2f} cm$^{{-1}}$", transform=ax.transAxes, fontsize=9, color="#c0392b")
ax.axvline(smearing_star, color="#2980b9", ls="--", lw=0.8, alpha=0.7)
ax.text(smearing_star * 1.03, 8, f"smearing*≈{smearing_star:.3f} Ry", color="#2980b9", fontsize=9)
set_smearing_axis(ax, max(dgs))
ax.set_ylabel("K-point Kohn kink [cm$^{-1}$]", fontsize=11)
fig.suptitle("Graphene Kohn anomaly: sharp melt with electronic smearing", fontsize=11, y=0.96)
ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.76)
fig.savefig(OUT/"gr_fig1_kink_melt.png", dpi=150, bbox_inches="tight")
print("wrote fig1")

# ============ Figure 2: high-frequency overlay ============
fig, (axD, axM) = plt.subplots(1, 2, figsize=(15, 5.8))
# recompute x properly
sl = np.linalg.norm(np.diff(qs, axis=0), axis=1)
x = np.concatenate([[0.], np.cumsum(sl)]); seg=len(x)//3
tick=[float(x[0]),float(x[seg-1]),float(x[2*seg-1]),float(x[-1])]
for dg in dgs:
    col=cmap(norm(dg)); axD.plot(x, dft[dg]["band"], color=col, lw=1.2, alpha=0.9)
for i,dg_str in enumerate(mlip_dgs):
    col=cmap(norm(float(dg_str))); axM.plot(x, mlip_bands[i], color=col, lw=1.1, alpha=0.9, ls="--")
for ax in (axD,axM):
    for xt in tick[1:-1]: ax.axvline(xt, color="#ececec", lw=0.7)
    ax.axhline(0, color="#ccc", lw=0.5); ax.set_xticks(tick); ax.set_xticklabels(LABELS); ax.set_ylim(1240,1620)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axD.set_title(f"DFT ({len(dgs)} points)", fontsize=11)
axM.set_title(f"MLIP + long-range ({len(mlip_dgs)} points)", fontsize=11)
axD.set_ylabel("frequency [cm$^{-1}$]", fontsize=10)
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
# Reserve a dedicated right margin for the colorbar so it cannot cover axM.
fig.subplots_adjust(left=0.065, right=0.89, bottom=0.12, top=0.82, wspace=0.12)
cax = fig.add_axes([0.915, 0.18, 0.015, 0.58])
cbar=fig.colorbar(sm,cax=cax); cbar.set_label("Fermi–Dirac smearing (Ry)",fontsize=9)
fig.suptitle("Graphene Kohn anomaly (high-frequency): DFT vs MLIP + long-range across smearing",
             fontsize=11.5, y=0.96)
fig.savefig(OUT/"gr_fig2_hf_overlay.png", dpi=150, bbox_inches="tight")
print("wrote fig2")

# ============ Figure 3: K-iTO vs smearing ============
fig, ax = plt.subplots(figsize=(8.4, 5.7))
ax.plot(smearing_dft, [dft[dg]["K_iTO"] for dg in dgs], "o-", color="#222", lw=1.8, ms=5,
        label=f"DFT ({len(dgs)} points)")
ax.plot(smearing_mlip, mlip_K_iTO, "s--", color="#c0392b", lw=1.4, ms=5,
        label=f"MLIP + long-range ({len(mlip_dgs)} points)")
mae_iTO_pairs = [(m, dft_lookup(dg)["K_iTO"]) for m, dg in zip(mlip_K_iTO, mlip_dgs) if dft_lookup(dg) is not None]
mae_iTO = np.mean(np.abs(np.array([p[0] for p in mae_iTO_pairs]) - np.array([p[1] for p in mae_iTO_pairs])))
ax.text(0.98, 0.04, f"K-iTO MAE = {mae_iTO:.1f} cm$^{{-1}}$", transform=ax.transAxes,
        fontsize=9, color="#c0392b", ha="right")
bare = dft[max(dgs)]["K_iTO"]
ax.axhline(bare, color="#888", ls=":", label=f"bare ≈ {bare:.0f} cm⁻¹")
soft = dft[min(dgs)]["K_iTO"]
ax.annotate(f"Kohn softening = {bare-soft:.0f} cm⁻¹", xy=(smearing_dft[0], soft),
            xytext=(0.013, 1375), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#666"))
set_smearing_axis(ax, max(dgs))
ax.set_ylabel("K-point iTO frequency [cm$^{-1}$]", fontsize=11)
fig.suptitle("K-point optical phonon: Kohn softening vs electronic smearing", fontsize=10.5, y=0.96)
ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.76)
fig.savefig(OUT/"gr_fig3_kito.png", dpi=150, bbox_inches="tight")
print("wrote fig3")

# ============ Figure 4: K branches vs smearing ============
fig, ax = plt.subplots(figsize=(9.0, 5.8))
colors_b = plt.cm.tab10(np.linspace(0, 0.85, 6))
for b in range(6):
    ax.plot(smearing_dft, [dft[dg]["K_br"][b] for dg in dgs], "o-", color=colors_b[b], lw=1.5, ms=3,
            label=f"branch {b+1}")
    if b >= 4:  # top 2 branches = Kohn channels, show MLIP
        mlip_b_vals = [mlip_K_br[i][b] for i in range(len(mlip_dgs))]
        ax.plot(smearing_mlip, mlip_b_vals, "s--", color=colors_b[b], lw=1.0, ms=3, alpha=0.7)
set_smearing_axis(ax, max(dgs))
ax.set_ylabel("K-point frequency [cm$^{-1}$]", fontsize=11)
fig.suptitle("All 6 branches at K vs electronic smearing: only Kohn channels respond",
             fontsize=10.5, y=0.96)
handles, legend_labels = ax.get_legend_handles_labels()
handles.append(
    Line2D(
        [0], [0], color="#555555", marker="s", ls="--", lw=1.0, ms=4,
        label="MLIP overlay: branches 5–6",
    )
)
legend_labels.append("MLIP overlay: branches 5–6")
ax.legend(
    handles, legend_labels, frameon=False, fontsize=8, ncol=4,
    loc="lower center", bbox_to_anchor=(0.5, 1.02),
)
ax.axhline(0, color="#aaa", lw=0.5)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.11, right=0.97, bottom=0.14, top=0.72)
fig.savefig(OUT/"gr_fig4_branches.png", dpi=150, bbox_inches="tight")
print("wrote fig4")
print(f"\nAll 4 figures: DFT {len(dgs)}-pt + MLIP {len(mlip_dgs)}-pt")
print(f"kink_K MAE={mae:.2f}  K-iTO MAE={mae_iTO:.1f}")
