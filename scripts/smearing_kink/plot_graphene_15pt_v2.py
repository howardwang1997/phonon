"""Graphene 15-pt figures with DFT vs MLIP+healing comparison.
Uses existing deploy_lineA_bands.npz (9-pt MLIP) + 15-pt DFT fc2 yamls.
No new compute needed — extracts K-iTO + K-branches from band data."""
import sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

ROOT = Path("/Users/howardwang/Desktop/playground/phonon")
FD = ROOT / "results" / "graphene_kohn_fd"
OUT = ROOT / "results" / "smearing_kink"
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))
import friedel_module as fm, phonopy

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

LABELS = ["Γ", "M", "K", "Γ"]
PTS = [np.array([0.,0,0]), np.array([.5,0,0]), np.array([1/3,1/3,0]), np.array([0.,0,0])]
CM = 33.35641

# --- 15-pt DFT data ---
all_dgs = sorted([float(f.stem.split("_dg")[1].split("_")[0]) for f in FD.glob("graphene_sc6_dg*_phonopy.yaml")])
dft = {}
for dg in all_dgs:
    yml = FD / f"graphene_sc6_dg{dg}_phonopy.yaml"
    if not yml.exists():
        for alt in [f"{dg:.3f}"]:
            yml = FD / f"graphene_sc6_dg{alt}_phonopy.yaml"
            if yml.exists(): break
    if not yml.exists(): continue
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

# --- 9-pt MLIP data from existing deploy ---
d = np.load(FD / "deploy_lineA_bands.npz", allow_pickle=True)
mlip_dgs = [str(s) for s in d["dgs"]]
mlip_bands = d["mlip_bands"]  # (9, nq, 6)
mlip_kink = d["mlip_kink"]
dft_kink_9 = d["dft_kink"]
# K point index in the band: 40 per segment, K is at index 80 (start of 3rd segment)
K_idx = 80
mlip_K_iTO = [float(np.sort(mlip_bands[i, K_idx])[-1]) for i in range(len(mlip_dgs))]
mlip_K_br = [np.sort(mlip_bands[i, K_idx]) for i in range(len(mlip_dgs))]

dgs = sorted(dft.keys())
Tel_dft = np.array([dg * 157887 for dg in dgs])
Tel_mlip = np.array([float(dg) * 157887 for dg in mlip_dgs])
hl = json.loads((FD / "healing_law.json").read_text())
cmap = plt.cm.viridis; norm = Normalize(min(dgs), max(dgs))

# ============ Figure 1: kink_K vs T_el (DFT 15pt + MLIP 9pt) ============
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(Tel_dft, [dft[dg]["kK"] for dg in dgs], "o-", color="#222", lw=1.8, ms=5, zorder=5,
        label="DFT (15 fd points)")
ax.plot(Tel_mlip, mlip_kink, "s--", color="#c0392b", lw=1.4, ms=5,
        label=f"MLIP + long-range healing ({len(mlip_dgs)} pts)")
mae = np.mean(np.abs(mlip_kink - dft_kink_9))
ax.text(0.02, 0.02, f"kink_K MAE = {mae:.2f} cm$^{{-1}}$", transform=ax.transAxes, fontsize=9, color="#c0392b")
ax.axvline(hl["Tstar"], color="#2980b9", ls="--", lw=0.8, alpha=0.7)
ax.text(hl["Tstar"]*1.05, 8, f"T*≈{hl['Tstar']:.0f} K", color="#2980b9", fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("T$_{el}$ [K]", fontsize=11)
ax.set_ylabel("K-point Kohn kink [cm$^{-1}$]", fontsize=11)
ax.set_title("Graphene Kohn anomaly: sharp melt with T$_{el}$", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper right")
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT/"gr_fig1_kink_melt.png", dpi=150, bbox_inches="tight")
print("wrote fig1")

# ============ Figure 2: high-freq overlay (DFT 15 | MLIP 9) ============
fig, (axD, axM) = plt.subplots(1, 2, figsize=(15, 5.8))
for dg in dgs:
    col = cmap(norm(dg)); axD.plot(dft[dg]["band"][:,0]*0+dft[dg]["band"][:,0], dft[dg]["band"], color=col, lw=1.2, alpha=0.9) if False else None
    # use x from first band
    pass
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
axD.set_title("DFT (15 points)", fontsize=11); axM.set_title("MLIP + long-range (9 points)", fontsize=11)
axD.set_ylabel("frequency [cm$^{-1}$]", fontsize=10)
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
cbar=fig.colorbar(sm,ax=[axD,axM],pad=0.06,fraction=0.03,aspect=28); cbar.set_label("degauss (Ry)",fontsize=9)
fig.suptitle("Graphene Kohn anomaly (high-freq): DFT vs MLIP+healing — melt with T$_{el}$", fontsize=11.5, y=1.01)
fig.tight_layout(); fig.savefig(OUT/"gr_fig2_hf_overlay.png", dpi=150, bbox_inches="tight")
print("wrote fig2")

# ============ Figure 3: K-iTO vs T_el (DFT 15 + MLIP 9) ============
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(Tel_dft, [dft[dg]["K_iTO"] for dg in dgs], "o-", color="#222", lw=1.8, ms=5,
        label="DFT (15 points)")
ax.plot(Tel_mlip, mlip_K_iTO, "s--", color="#c0392b", lw=1.4, ms=5,
        label="MLIP + long-range (9 points)")
mae_iTO_pairs = [(m, dft_lookup(dg)["K_iTO"]) for m, dg in zip(mlip_K_iTO, mlip_dgs) if dft_lookup(dg) is not None]
mae_iTO = np.mean(np.abs(np.array([p[0] for p in mae_iTO_pairs]) - np.array([p[1] for p in mae_iTO_pairs])))
ax.text(0.02, 0.02, f"K-iTO MAE = {mae_iTO:.1f} cm$^{{-1}}$", transform=ax.transAxes, fontsize=9, color="#c0392b")
bare = dft[max(dgs)]["K_iTO"]
ax.axhline(bare, color="#888", ls=":", label=f"bare ≈ {bare:.0f} cm⁻¹")
soft = dft[min(dgs)]["K_iTO"]
ax.annotate(f"Kohn softening\n{soft:.0f}→{bare:.0f} cm⁻¹", xy=(Tel_dft[0],soft),
            xytext=(Tel_dft[0]*3, soft-35), fontsize=9, arrowprops=dict(arrowstyle="->",color="#666"))
ax.set_xscale("log"); ax.set_xlabel("T$_{el}$ [K]", fontsize=11)
ax.set_ylabel("K-point iTO frequency [cm$^{-1}$]", fontsize=11)
ax.set_title("K-point optical phonon: Kohn softening vs T$_{el}$ (DFT vs MLIP)", fontsize=10.5)
ax.legend(frameon=False, fontsize=9)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT/"gr_fig3_kito.png", dpi=150, bbox_inches="tight")
print("wrote fig3")

# ============ Figure 4: K-branches vs T_el (DFT 15 + MLIP 9) ============
fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors_b = plt.cm.tab10(np.linspace(0, 0.85, 6))
for b in range(6):
    ax.plot(Tel_dft, [dft[dg]["K_br"][b] for dg in dgs], "o-", color=colors_b[b], lw=1.5, ms=3,
            label=f"branch {b+1}")
    if b >= 4:  # top 2 branches = Kohn channels, show MLIP
        mlip_b_vals = [mlip_K_br[i][b] for i in range(len(mlip_dgs))]
        ax.plot(Tel_mlip, mlip_b_vals, "s--", color=colors_b[b], lw=1.0, ms=3, alpha=0.7)
ax.set_xscale("log"); ax.set_xlabel("T$_{el}$ [K]", fontsize=11)
ax.set_ylabel("K-point frequency [cm$^{-1}$]", fontsize=11)
ax.set_title("All 6 branches @ K vs T$_{el}$: only Kohn channels respond (□=MLIP)", fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
ax.axhline(0, color="#aaa", lw=0.5)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT/"gr_fig4_branches.png", dpi=150, bbox_inches="tight")
print("wrote fig4")
print(f"\nAll 4 figures: DFT {len(dgs)}-pt + MLIP {len(mlip_dgs)}-pt")
print(f"kink_K MAE={mae:.2f}  K-iTO MAE={mae_iTO:.1f}")
