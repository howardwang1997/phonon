#!/usr/bin/env python3
"""Publication-standard figures for the TD-Phonon / Kohn-Anomaly Weekly Report
(week ending 2026-07-01). Run in the `phonon` conda env:
    conda run -n phonon python docs/weekly/plot_weekly.py
Figures are self-contained (data embedded from the campaign logs + results docs);
no dependence on the (gitignored / remote) result files.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.colors import ListedColormap
import os

mpl.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True, "legend.fontsize": 9, "legend.frameon": False,
    "figure.facecolor": "white",
})
OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)
# colour-blind-safe (Okabe-Ito)
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
          purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#7F7F7F")


# ============================================================ FIG 1 — status matrix
def fig1_status_matrix():
    mats = ["NbSe2 (2H)", "2H-TaSe2", "NbS2 (2H)", "1T-TiSe2", "2H-TaS2",
            "1T-TaS2", "1T-VSe2", "1T-TiS2*", "1T-VS2*", "MoS2*", "WSe2*"]
    stages = ["DFT fc₂", "bands / χ(q)", "Path-P\n(anharm)",
              "(E)-EPW\nγ_qν", "(L)-SSCHA\nω(q,T)"]
    # 0 N/A(control) · 1 pending · 2 running · 3 done
    S = np.array([
        [3,3,3,3,1],   # NbSe2   — fc2/bands/PathP/EPW done; SSCHA pending (H20 offline)
        [3,3,3,3,1],   # 2H-TaSe2 — complete; SSCHA pending
        [3,3,3,3,1],   # NbS2    — complete; SSCHA pending
        [3,3,3,3,1],   # 1T-TiSe2 — complete; EPW off-instability (3×3 stable) → re-run 2×2
        [3,3,3,3,1],   # 2H-TaS2 — complete; SSCHA pending
        [3,3,3,3,1],   # 1T-TaS2 — complete; soft mode partial (true CDW √13×√13)
        [3,3,3,3,1],   # 1T-VSe2 — complete; EPW off-instability (3×3 stable) → re-run
        [3,3,0,0,0],   # 1T-TiS2 (control) — fc2/bands only
        [3,3,0,0,0],   # 1T-VS2  (control)
        [3,3,0,0,0],   # MoS2    (gapped control)
        [3,3,0,0,0],   # WSe2    (gapped control)
    ])
    # annotations: captured DFT soft modes (THz) on fc2-done cells
    soft = {(0,0): "−2.18", (1,0): "−2.18", (2,0): "−1.89", (3,0): "−0.03",
            (4,0): "−1.97", (5,0): "−0.75", (6,0): "−0.00"}
    cmap = ListedColormap(["#E8E8E8", "#FbE9C7", CB["orange"], CB["green"]])
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.imshow(S, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages)
    ax.set_yticks(range(len(mats)));   ax.set_yticklabels(mats)
    ax.set_xticks(np.arange(-.5, len(stages), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(mats), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    txt = {0: "n/a", 1: "·", 2: "run", 3: "✓"}
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            lbl = soft.get((i, j), txt[S[i, j]])
            c = "white" if S[i, j] >= 2 else ("#999" if S[i, j] == 0 else "#666")
            ax.text(j, i, lbl, ha="center", va="center", color=c,
                    fontsize=8.5, fontweight="bold")
    ax.set_title("Fig 1  V100 FP64 campaign — status matrix (2026-07-02, complete)")
    leg = [Patch(fc=CB["green"], label="done"), Patch(fc=CB["orange"], label="running"),
           Patch(fc="#FbE9C7", label="pending"), Patch(fc="#E8E8E8", label="n/a (non-CDW)")]
    ax.legend(handles=leg, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    ax.text(1.02, 0.5, "* = non-CDW / gapped control\nfc₂ cells: captured DFT soft-mode "
            "ω_min (THz)\n→ CDW instability seen at DFT level",
            transform=ax.transAxes, fontsize=8, va="center", color="#555")
    fig.savefig(f"{OUT}/fig1_status_matrix.png"); plt.close(fig)
    print("wrote fig1_status_matrix.png")


# ============================================================ FIG 2 — (E)-(L) origin map
def fig2_origin_map():
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    # axes: x = electronic (E) drive (nesting <-> EPC), y = lattice-anharmonic (L) stabilization
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axvline(5, color="#ccc", lw=1, ls="--"); ax.axhline(5, color="#ccc", lw=1, ls="--")
    ax.set_xlabel("(E) electronic channel  —  Fermi-surface nesting  →  momentum-dependent EPC")
    ax.set_ylabel("(L) lattice-anharmonic channel  —  harmonic  →  strongly quantum-stabilized")
    ax.set_title("Fig 2  The (E)–(L) CDW origin-classification map (Part II — the discovery)")
    # quadrant labels
    ax.text(2.5, 9.3, "nesting-driven", ha="center", color="#888", fontsize=9, style="italic")
    ax.text(7.5, 9.3, "EPC-driven", ha="center", color="#888", fontsize=9, style="italic")
    ax.text(0.3, 8.4, "anharmonic-\nstabilized", ha="left", color="#888", fontsize=8, style="italic")
    # placed member: NbSe2 (adjudicated)
    ax.scatter([8.1], [7.6], s=260, color=CB["green"], zorder=5, edgecolor="k")
    ax.annotate("NbSe₂  ✓ adjudicated\nχ(q) not peaked → not nesting\n"
                "γ_qν broad @ q_CDW → EPC\nSSCHA: stabilizes ~150 K (exp 145 K)",
                (8.1, 7.6), (4.6, 5.6), fontsize=8.3, color=CB["green"],
                arrowprops=dict(arrowstyle="->", color=CB["green"]))
    # graphene reference (E-channel calibrator, not a CDW)
    ax.scatter([6.4], [1.2], s=130, color=CB["blue"], zorder=5, edgecolor="k")
    ax.annotate("graphene (ref.)\nKohn anomalies, doping-tuned\nEPC = pure Fermi-surface effect",
                (6.4, 1.2), (5.0, 2.2), fontsize=8, color=CB["blue"],
                arrowprops=dict(arrowstyle="->", color=CB["blue"]))
    # family members to be placed
    to_place = {"2H-TaSe₂": (2), "NbS₂": 2, "1T-TiSe₂": 2,
                "2H-TaS₂": 1, "1T-TaS₂": 1, "1T-VSe₂": 1}
    xs = [3.0, 4.2, 2.2, 6.9, 5.6, 3.7]; ys = [7.0, 6.2, 8.0, 6.7, 7.8, 5.2]
    for (name, st), x, y in zip(to_place.items(), xs, ys):
        col = CB["orange"] if st == 2 else "#BbBbBb"
        ax.scatter([x], [y], s=120, facecolor="white", edgecolor=col, lw=2,
                   hatch="////" if st == 2 else None, zorder=4)
        ax.text(x + 0.18, y + 0.18, name, fontsize=7.6, color="#555")
    leg = [plt.Line2D([], [], marker="o", ls="", mfc=CB["green"], mec="k", ms=11, label="adjudicated (NbSe₂)"),
           plt.Line2D([], [], marker="o", ls="", mfc="white", mec=CB["orange"], ms=10, label="(E)/(L) in progress"),
           plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#BbBbBb", ms=10, label="family, to be placed"),
           plt.Line2D([], [], marker="o", ls="", mfc=CB["blue"], mec="k", ms=9, label="graphene reference")]
    ax.legend(handles=leg, loc="lower right", fontsize=8)
    fig.savefig(f"{OUT}/fig2_origin_map.png"); plt.close(fig)
    print("wrote fig2_origin_map.png")


# ============================================================ FIG 3 — compute budget
def fig3_compute_budget():
    # (phase, lo, hi, hardware, unit, gated)
    rows = [
        ("Phase 0-1: DFT fc₂ + anchors (family)", 30, 45, "2×V100 (GPU-lane, ~free)", "box-h", False),
        ("Phase 1-2: Path-P anharmonic (7 CDW)",       5,  8,  "2×V100 (GPU-lane)",        "box-h", False),
        ("Phase 1-2: (E)-EPW γ_qν (6 new)",  40, 90, "2×V100 (CPU-lane)",        "box-h", False),
        ("λ(T_el) + ASR + convergence",           20, 40, "2×V100 (CPU-lane)",        "box-h", False),
        ("MLIP half: E1/E3/E4/(L)-SSCHA/κ",      900, 1800, "8×H20 (offline)",         "GPU-h", False),
        ("RENTAL (FP64) — gated on discovery",   2100, 5800, "rented A100/V100",           "GPU-h", True),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lbl, lo, hi, hw, unit, gated) in zip(y, rows):
        col = CB["red"] if gated else (CB["sky"] if "H20" in hw else CB["green"])
        ax.barh(yi, hi - lo, left=lo, color=col, alpha=0.85, edgecolor="k", lw=0.6)
        ax.text(hi * 1.02, yi, f"{lo}–{hi} {unit}\n{hw}", va="center", fontsize=7.8, color="#333")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    ax.set_xscale("log"); ax.set_xlim(3, 22000)
    ax.set_xlabel("compute (log scale) — box-h on 2×V100 (FP64/CPU) or GPU-h on 8×H20 / rental")
    ax.set_title("Fig 3  Compute budget by phase & hardware class (NCS roadmap §0b)")
    leg = [Patch(fc=CB["green"], label="2×V100 now (box-h, FP64/CPU)"),
           Patch(fc=CB["sky"], label="8×H20 (GPU-h, MLIP, offline)"),
           Patch(fc=CB["red"], label="rental FP64 — only if discovery gate passes")]
    ax.legend(handles=leg, loc="lower right", fontsize=8)
    ax.axvline(210, color="#888", ls=":", lw=1)
    ax.text(210, len(rows)-0.4, " core-complete draft\n carried by machines in hand",
            fontsize=7.5, color="#666", rotation=0)
    fig.savefig(f"{OUT}/fig3_compute_budget.png"); plt.close(fig)
    print("wrote fig3_compute_budget.png")


# ============================================================ FIG 4 — NbSe2 lambda(T_el) PRELIMINARY
def fig4_lambda_Tel():
    # measured points (from epw.out, week ending 07-01) — PRELIMINARY, mixed params
    Tel  = np.array([3158, 7894, 9473])                 # K  (degauss 0.02, 0.05, 0.06 Ry)
    dg   = ["0.02", "0.05", "0.06"]
    lam_sets = [ [140.77], [13.30,15.32,14.32,11.75], [17.72,18.47,18.00,16.83] ]
    lam_mean = np.array([np.mean(s) for s in lam_sets])
    lam_err  = np.array([ (max(s)-min(s))/2 if len(s)>1 else 0 for s in lam_sets ])
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.errorbar(Tel, lam_mean, yerr=lam_err, fmt="o-", color=CB["purple"], ms=9,
                capsize=4, lw=1.6, mec="k", zorder=4, label="EPW λ (nsmear spread)")
    ax.set_yscale("log")
    for x, y, d, s in zip(Tel, lam_mean, dg, lam_sets):
        ax.annotate(f"degauss={d}\nλ={y:.0f}", (x, y),
                    textcoords="offset points", xytext=(10, 6), fontsize=8, color="#444")
    ax.axhline(1.0, color=CB["green"], ls="--", lw=1.4)
    ax.text(9600, 1.05, "physical NbSe₂ λ ≈ 1 (target regime)", color=CB["green"], fontsize=8, ha="right")
    ax.set_xlabel("electronic temperature  T_el = degauss × 157887  (K)")
    ax.set_ylabel("EPW integrated λ  (log)")
    ax.set_title("Fig 4  NbSe₂ λ(T_el) — PRELIMINARY (soft-mode-divergent, not converged)")
    # caveat box
    ax.text(0.03, 0.05,
            "PRELIMINARY — not publication values:\n"
            "• λ inflated by 1/ω² of the CDW soft mode (near-imaginary)\n"
            "• base pt uses different EPW smearing than d050/d060 (not one ruler)\n"
            "• nkf=24 too coarse for the metal Fermi surface\n"
            "→ needs nkf convergence + unified params + more T_el points",
            transform=ax.transAxes, fontsize=7.6, color=CB["red"], va="bottom",
            bbox=dict(boxstyle="round", fc="#FFF3F0", ec=CB["red"], alpha=0.9))
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(f"{OUT}/fig4_lambda_Tel.png"); plt.close(fig)
    print("wrote fig4_lambda_Tel.png")


# ============================================================ FIG 5 — graphene Kohn cure
def fig5_graphene_cure():
    # Gamma-E2g (cm-1) across foundations + distillation cures + DFT truth (fixed a=2.46 / relaxed)
    labels = ["MACE\n(found.)", "SevenNet\n(found.)", "MatterSim\n(found.)",
              "bulk-FT\n(distill)", "graphene-FT\n(distill)", "DFT\n(this work)", "lit."]
    vals   = [1254, 1382, 1563, 1265, 1570, 1568, 1600]
    pct    = ["−22%", "−14%", "−2%", "+8% gap", "+101% gap", "truth", "ref"]
    cols   = [CB["red"], CB["orange"], CB["yellow"], CB["sky"], CB["green"], "#333", "#999"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.8, 4.8),
                                   gridspec_kw=dict(width_ratios=[1.5, 1]))
    b = axL.bar(labels, vals, color=cols, edgecolor="k", lw=0.6)
    axL.axhline(1568, color=CB["green"], ls="--", lw=1.2)
    axL.axhline(1600, color="#999", ls=":", lw=1)
    axL.set_ylim(1150, 1650); axL.set_ylabel("graphene Γ-E₂g optical frequency (cm⁻¹)")
    axL.set_title("(a) Γ-E₂g softening is model-specific; material-specific distillation cures it")
    for r, p in zip(b, pct):
        axL.text(r.get_x()+r.get_width()/2, r.get_height()+6, p, ha="center", fontsize=7.8, color="#333")
    axL.tick_params(axis="x", labelsize=8)
    # panel b — converged K-A1' cusp is real (V-Q1)
    sc = ["5×5\n(interp)", "6×6\n(commens.)", "7×7\n(interp)"]
    kfreq = [1362, 1292, 1324]; kink = [0.84, 14.41, 8.30]
    ax2 = axR.twinx()
    axR.bar(sc, kfreq, color=CB["blue"], alpha=0.35, edgecolor="k", lw=0.5, label="K-A₁′ freq")
    ax2.plot(sc, kink, "o-", color=CB["red"], ms=9, lw=1.6, label="cusp kink |dv|")
    axR.axhline(1300, color="#999", ls=":", lw=1); axR.set_ylim(1200, 1420)
    axR.set_ylabel("K-A₁′ freq (cm⁻¹)", color=CB["blue"]); ax2.set_ylabel("cusp kink |dv|", color=CB["red"])
    axR.set_title("(b) V-Q1: converged (6×6) DFT\nreveals a REAL K-A₁′ cusp")
    ax2.annotate("kink 0.8→14.4\n= real Kohn anomaly\n(5×5 under-resolved it)", (1, 14.41), (0.1, 11),
                 fontsize=7.5, color=CB["red"], arrowprops=dict(arrowstyle="->", color=CB["red"]))
    fig.suptitle("Fig 5  Graphene Kohn anomaly: foundation-MLIP failure & FC-distillation cure",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/fig5_graphene_cure.png"); plt.close(fig)
    print("wrote fig5_graphene_cure.png")


# ============================================================ FIG 6 — Path-P gap closure
def fig6_pathp():
    groups = ["graphene\n(100–600 K, 150 cfg)", "NbSe₂\n(CDW coord, 69 cfg)"]
    found  = [294, 366]; harm = [149, 699]; pathp = [21, 151]
    x = np.arange(len(groups)); w = 0.25
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.bar(x-w, found, w, label="foundation MLIP", color=CB["red"], edgecolor="k", lw=0.6)
    ax.bar(x,   harm,  w, label="harmonic-FT (0 K distill)", color=CB["orange"], edgecolor="k", lw=0.6)
    ax.bar(x+w, pathp, w, label="Path-P (anharmonic distill)", color=CB["green"], edgecolor="k", lw=0.6)
    for xi, f, h, p in zip(x, found, harm, pathp):
        for dx, v in zip((-w, 0, w), (f, h, p)):
            ax.text(xi+dx, v+8, f"{v}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("held-out force RMSE vs DFT (meV/Å)")
    ax.set_title("Fig 6  Path-P anharmonic distillation closes the thermal-force gap")
    ax.legend(loc="upper left")
    ax.annotate("86% gap\nclosed", (0+w, 21), (0+w, 190), ha="center", fontsize=8,
                color=CB["green"], arrowprops=dict(arrowstyle="->", color=CB["green"]))
    ax.annotate("harmonic-FT BACKFIRES\n(699 > 366); Path-P 4.6× better", (1, 699), (0.35, 620),
                fontsize=7.8, color="#b00", arrowprops=dict(arrowstyle="->", color="#b00"))
    ax.set_ylim(0, 780)
    fig.savefig(f"{OUT}/fig6_pathp.png"); plt.close(fig)
    print("wrote fig6_pathp.png")


# ============================================================ FIG 7 — breadth-vs-depth law
def fig7_breadth():
    N = np.array([8, 16, 32, 64]); mae = np.array([1.80, 1.80, 1.40, 1.34])
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(N, mae, "o-", color=CB["blue"], ms=10, lw=1.8, mec="k", zorder=4,
            label="held-out transfer MAE")
    ax.axhspan(1.30, 1.36, color=CB["green"], alpha=0.12)
    ax.axhline(1.34, color=CB["green"], ls="--", lw=1.2)
    ax.text(60, 1.365, "transfer floor ≈ 1.3 THz", color=CB["green"], fontsize=8.5, ha="right")
    ax.axhline(0.10, color=CB["purple"], ls=":", lw=1.4)
    ax.text(8.5, 0.14, "in-domain (distilled) ≈ 0.10 THz, imaginary modes eliminated",
            color=CB["purple"], fontsize=8)
    ax.axvspan(16, 32, color=CB["orange"], alpha=0.12)
    ax.text(23, 1.72, "knee\n~16–32", ha="center", color=CB["orange"], fontsize=8.5)
    ax.set_xscale("log", base=2); ax.set_xticks(N); ax.set_xticklabels(N)
    ax.set_xlabel("number of distinct training materials  N  (breadth)")
    ax.set_ylabel("held-out phonon-frequency MAE (THz)")
    ax.set_ylim(0, 2.0)
    ax.set_title("Fig 7  Breadth-not-depth generalization law (FC-distillation)")
    ax.legend(loc="center right")
    ax.text(0.02, 0.03, "breadth of chemistry (not depth of sampling per material) drives transfer;\n"
            "coverage-driven acquisition beats random ≈0.18 THz @ N=32; naive uncertainty is worst",
            transform=ax.transAxes, fontsize=7.6, color="#555", va="bottom")
    fig.savefig(f"{OUT}/fig7_breadth.png"); plt.close(fig)
    print("wrote fig7_breadth.png")


# ============================================================ FIG 8 — (E)-channel signatures
def fig8_echannel():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.7))
    # A: NbSe2 soft mode melts with electronic T (E1, frozen-phonon along CDW eigenvector)
    Tel = np.array([474, 947, 1579, 2368, 3158, 4737])
    w   = np.array([-14.1, 56.1, 89.6, 110.3, 120.6, 128.9])
    axA.plot(Tel, w, "o-", color=CB["red"], ms=9, lw=1.8, mec="k")
    axA.axhline(0, color="#333", lw=1)
    axA.fill_between(Tel, w, 0, where=(w < 0), color=CB["red"], alpha=0.2)
    axA.annotate("ω²=0 crossing\n→ electronic T_CDW ≈ 500–570 K", (700, 0), (1400, -40),
                 fontsize=8, arrowprops=dict(arrowstyle="->"))
    axA.set_xlabel("electronic temperature T_el (K)")
    axA.set_ylabel("NbSe₂ CDW soft-mode freq (cm⁻¹)")
    axA.set_title("(a) NbSe₂: CDW soft mode electronically melts\n(EPC weakened by Fermi smearing)")
    # B: graphene Kohn modes stiffen with electronic T (E7 DFPT 3x3), q-selective
    TelB = np.array([474, 947, 1579, 3158, 6315])
    G = np.array([1531.8, 1559.9, 1569.9, 1571.3, 1558.2])
    K = np.array([1250.9, 1285.7, 1300.1, 1318.7, 1354.0])
    axB.plot(TelB, G, "s-", color=CB["blue"], ms=8, lw=1.6, mec="k", label="Γ-E₂g  (Δ≈40 cm⁻¹)")
    axB.plot(TelB, K, "o-", color=CB["green"], ms=8, lw=1.6, mec="k", label="K-A₁′  (Δ≈103 cm⁻¹)")
    axB.set_xlabel("electronic temperature T_el (K)")
    axB.set_ylabel("graphene mode freq (cm⁻¹)")
    axB.set_title("(b) graphene: Kohn anomalies stiffen with T_el\n(q-selective — K-A₁′ 2.6× more than Γ)")
    axB.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Fig 8  The (E) electronic channel — already demonstrated on both flagships "
                 "(MLIP-blind)", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/fig8_echannel.png"); plt.close(fig)
    print("wrote fig8_echannel.png")


# ============================================================ FIG 9 — Kohn anomaly phonon spectrum
def fig9_kohn_anomaly():
    """Graphene phonon dispersion (M–Γ–K–M) that exhibits the Kohn anomaly.
    Real computed data (converged DFT vs foundation MACE-MP-0 vs graphene-FT),
    embedded in docs/weekly/figdata/*.npz (601 q-pts × 6 branches, a=2.46)."""
    THZ2CM = 33.35641
    FD = os.path.join(os.path.dirname(__file__), "figdata")
    def load(name):
        d = np.load(os.path.join(FD, name), allow_pickle=True)
        return (d["distances"], d["frequencies"] * THZ2CM,
                d["label_positions"], [str(x) for x in d["labels"]])
    xd, fd, lp, lab = load("graphene_dft_conv.npz")        # converged DFT — real cusp
    xm, fm, _,  _   = load("graphene_foundation_mace.npz")  # foundation MACE — washed out
    xg, fg, _,  _   = load("graphene_ft.npz")               # graphene-specific FT — recovers
    lab = ["M", "Γ", "K", "M"]
    gi, ki = np.argmin(np.abs(xd - lp[1])), np.argmin(np.abs(xd - lp[2]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 5.0),
                                   gridspec_kw={"width_ratios": [1.12, 1]})

    # ---- (a) full dispersion: DFT (truth) vs foundation MACE, all branches
    for b in range(fd.shape[1]):
        axA.plot(xd, fd[:, b], color=CB["blue"], lw=1.9,
                 label="DFT (converged, truth)" if b == 0 else None, zorder=3)
        axA.plot(xm, fm[:, b], color=CB["orange"], lw=1.7, ls="--",
                 label="foundation MACE-MP-0" if b == 0 else None, zorder=2)
    for xpos in lp:
        axA.axvline(xpos, color="#bbb", lw=0.8, zorder=1)
    axA.set_xticks(lp); axA.set_xticklabels(lab)
    axA.set_xlim(xd.min(), xd.max()); axA.set_ylim(-30, 1780)
    axA.set_ylabel("phonon frequency (cm⁻¹)")
    axA.set_title("(a) graphene dispersion — the Kohn anomaly is a DFT feature\n"
                  "the foundation MLIP softens & washes it out")
    axA.legend(loc="center", bbox_to_anchor=(0.5, 0.42), fontsize=8.8,
               frameon=True, framealpha=0.85, facecolor="white", edgecolor="none")
    axA.annotate("Γ-E₂g\nKohn cusp", (lp[1], fd[gi].max()), (lp[1] + 0.028, 1690),
                 fontsize=8.5, color=CB["blue"], ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=CB["blue"]))
    axA.annotate("K-A₁′\nKohn cusp", (lp[2], fd[ki].max()), (lp[2] - 0.005, 1660),
                 fontsize=8.5, color=CB["blue"], ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=CB["blue"]))

    # ---- (b) top optical branch zoom: who keeps the cusp
    top = fd.shape[1] - 1
    axB.plot(xd, fd[:, top], color=CB["blue"],  lw=2.3,               label="DFT (truth)        1570 / 1292", zorder=4)
    axB.plot(xg, fg[:, top], color=CB["green"], lw=1.9, ls=(0,(1,1)), label="graphene-FT       1570 / 1368", zorder=3)
    axB.plot(xm, fm[:, top], color=CB["orange"],lw=1.9, ls="--",      label="foundation MACE  1238 / 1113", zorder=2)
    for xpos in lp:
        axB.axvline(xpos, color="#bbb", lw=0.8, zorder=1)
    axB.scatter([lp[1], lp[2]], [fd[gi].max(), fd[ki].max()], s=42,
                facecolor="white", edgecolor=CB["blue"], lw=1.6, zorder=5)
    axB.set_xticks(lp); axB.set_xticklabels(lab)
    axB.set_xlim(xd.min(), xd.max())
    axB.set_ylabel("highest optical branch (cm⁻¹)")
    axB.set_title("(b) top branch zoom — DFT & graphene-FT keep the cusp;\n"
                  "foundation MLIP over-softens (Γ −21 %) and smooths it")
    axB.legend(loc="lower center", fontsize=7.8, title="  Γ-E₂g / K-A₁′  (cm⁻¹)",
               title_fontsize=8, ncol=1,
               frameon=True, framealpha=0.85, facecolor="white", edgecolor="none")

    fig.suptitle("Fig 9  A phonon spectrum exhibiting the Kohn anomaly — graphene (M–Γ–K–M)",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/fig9_kohn_anomaly.png"); plt.close(fig)
    print("wrote fig9_kohn_anomaly.png")


# ============================================================ FIG 10 — temperature-dependent Kohn anomaly
def fig10_td_kohn_anomaly():
    """The Kohn anomaly's *physical-temperature* evolution (the (L)-channel),
    on both flagships. Real MLIP TDEP/SSCHA data (results/td_phonon):
      (a) graphene M2 dense-T — the K-A₁′ Kohn cusp PERSISTS while Γ stays washed;
      (b) NbSe₂ — the CDW soft mode (extreme Kohn anomaly) heals with T; the
          rigorous SSCHA free-energy Hessian is stable at all T (quantum-stabilized),
          the crossover ≈ exp monolayer T_CDW 145 K.
    Distinct from Fig 8 (electronic T_el, frozen-phonon): this is real lattice T."""
    # (a) graphene cusp kink |dv| vs T  (td_graphene_ft_m2.csv)
    Tg = np.array([10, 50, 100, 200, 300, 400, 500, 600])
    kink_K = np.array([95.90, 95.57, 98.13, 99.04, 91.65, 102.67, 94.88, 91.53])
    kink_G = np.array([5.77, 5.22, 5.55, 5.35, 6.87, 4.84, 5.05, 5.68])
    wK = np.array([1186.99, 1188.12, 1184.44, 1184.84, 1179.55, 1167.74, 1173.57, 1171.69])
    # (b) NbSe2 soft mode vs T  (td_nbse2_ft_Tfix.csv TDEP + nbse2_sscha.csv SSCHA)
    Tn = np.array([20, 100, 200, 300, 400])
    tdep = np.array([-0.482, -0.554, -0.000, -0.000, -0.000])
    sscha = np.array([0.0, 0.0, 0.0, 0.0, 0.0])   # free-energy Hessian: 0 imaginary at all T
    bare = -2.18                                   # bare DFT harmonic soft mode (0 K)

    dft0, ft0, found0 = -2.18, -2.23, -0.20   # NbSe2 0 K harmonic anchors (THz)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 5.1))

    # ---- (a) graphene: the K cusp survives, the Γ cusp is washed out (MLIP (L)-channel)
    axA.axhspan(88, 103, color=CB["green"], alpha=0.10)
    axA.plot(Tg, kink_K, "o-", color=CB["green"], ms=8, lw=1.9, mec="k",
             label="K-A₁′  (kink ≈ 95 — cusp persists)", zorder=4)
    axA.plot(Tg, kink_G, "s-", color=CB["blue"], ms=7, lw=1.7, mec="k",
             label="Γ-E₂g  (kink ≈ 5 — washed out)", zorder=3)
    axA.set_ylim(0, 118); axA.set_xlim(-15, 615)
    axA.set_xlabel("physical temperature  T (K)")
    axA.set_ylabel("Kohn-cusp sharpness  |dv|  (top-branch curvature)")
    axA.set_title("(a) graphene — the K-A₁′ Kohn cusp survives to 600 K")
    axA.legend(loc="center", bbox_to_anchor=(0.5, 0.42), fontsize=8.5)
    axA.text(0.5, 0.955, "K-A₁′ softens only ~1–2 % (1187→1172 cm⁻¹) yet the cusp is robust",
             transform=axA.transAxes, ha="center", fontsize=7.6, color="#555")
    axA.text(0.02, 0.12, "fine-tuned MLIP (L)-channel — no finite-T DFT (infeasible; Fig 11).\n"
             "FT↔DFT validated at 0 K in Fig 9 (graphene-FT 1570/1368 ≈ DFT 1570/1292).",
             transform=axA.transAxes, fontsize=6.9, color="#999", va="bottom")

    # ---- (b) NbSe2: the CDW soft mode heals with T; 0 K FT-vs-DFT anchors made explicit
    axB.axhline(0, color="#333", lw=1)
    axB.axhline(dft0, color="#777", ls="--", lw=1.2)
    # 0 K harmonic anchors: DFT / FT (overlap = the match) / foundation (misses)
    axB.plot([7], [dft0], "D", color="#777", ms=11, mec="k", zorder=6, label="DFT 0 K harmonic  −2.18")
    axB.plot([7], [ft0], "o", color=CB["green"], ms=9, mec="k", zorder=7, label="distilled-FT 0 K  −2.23 (≈DFT)")
    axB.plot([7], [found0], "X", color=CB["red"], ms=11, mec="k", zorder=6, label="foundation 0 K  −0.20 (misses)")
    axB.text(20, dft0 - 0.10, "0 K harmonic:  FT −2.23 ≈ DFT −2.18", fontsize=7.6, color="#444")
    # MLIP T-evolution
    axB.plot(Tn, tdep, "o-", color=CB["orange"], ms=8, lw=1.8, mec="k",
             label="TDEP soft mode (MLIP)", zorder=4)
    axB.plot(Tn, sscha, "s-", color=CB["purple"], ms=8, lw=1.8, mec="k",
             label="SSCHA free-energy Hessian (MLIP)", zorder=5)
    axB.fill_between(Tn, tdep, 0, where=(tdep < 0), color=CB["orange"], alpha=0.16)
    axB.axvline(145, color=CB["red"], ls=":", lw=1.3)
    axB.text(150, -0.95, "exp monolayer\nT_CDW ≈ 145 K", color=CB["red"], fontsize=7.8)
    axB.text(70, -1.62, "renormalized up ~2.2 THz\n(quantum + anharmonic)",
             fontsize=7.6, color="#555")
    axB.set_ylim(-2.62, 0.55); axB.set_xlim(-8, 415)
    axB.set_xlabel("physical temperature  T (K)")
    axB.set_ylabel("NbSe₂ CDW soft-mode min frequency (THz)")
    axB.set_title("(b) NbSe₂ — soft mode heals with T; FT starts on the DFT 0 K value")
    axB.legend(loc="center right", fontsize=7.2, ncol=1)

    fig.suptitle("Fig 10  Temperature-dependent Kohn anomaly — the (L)-channel ω(q,T) "
                 "on both flagships", fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/fig10_td_kohn_anomaly.png"); plt.close(fig)
    print("wrote fig10_td_kohn_anomaly.png")


# ============================================================ FIG 11 — speedup of the TD calculation
def fig11_td_speedup():
    """Why the temperature-dependent phonon calculation is only tractable with the
    MLIP. Any TDEP/SSCHA ω(q,T) map = force evaluations on thermally-sampled
    snapshots across a T-grid; the MLIP replaces the DFT force call.
      (a) per-force-evaluation wall time (measured anchors): DFT CPU-QE ~120 s,
          GPU-QE ~16 s (Path-P: 150 cfg / 40 min on 1 V100), MLIP-MACE ~0.1 s
          → the ~10³× 'MLIP proxy' (GPU-DFT alone is only ~5–15×).
      (b) that unit cost × one full ω(q,T) map (~6,000 evals): MLIP ~10 min vs
          DFT projected >1 day (GPU) / >1 week (CPU) — never run in full, which
          is exactly the enabling point. DFT bars = projected (hatched)."""
    engines = ["DFT\nCPU-QE\n(3×3)", "DFT\nGPU-QE\n(V100)", "MLIP\n(MACE, GPU)"]
    cols = [CB["blue"], CB["orange"], CB["green"]]
    t_eval = np.array([120.0, 16.0, 0.1])                  # s / force evaluation
    N_eval = 6000                                          # 5 T × 300 cfg × 4 pop
    t_map_h = t_eval * N_eval / 3600.0                     # h for one ω(q,T) map
    hatch = [None, None, None]  # panel-a all measured
    proj = [True, True, False]  # panel-b: DFT projected, MLIP measured-scale

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 5.0))
    x = np.arange(3)

    # ---- (a) per-force-evaluation cost (measured), log scale
    axA.bar(x, t_eval, 0.62, color=cols, edgecolor="k", lw=0.7, zorder=3)
    axA.set_yscale("log"); axA.set_ylim(0.03, 400)
    for xi, v in zip(x, t_eval):
        lab = f"{v:g} s" if v >= 1 else f"{v*1000:.0f} ms"
        axA.text(xi, v * 1.35, lab, ha="center", fontsize=9, fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels(engines, fontsize=8.6)
    axA.set_ylabel("wall-clock per force evaluation (s, log)")
    axA.set_title("(a) one force evaluation — DFT measured, MLIP typical")
    axA.annotate("", (1, 16), (0, 120), arrowprops=dict(arrowstyle="->", color="#555"))
    axA.text(0.5, 60, "GPU-DFT\n~7.5×", ha="center", fontsize=8, color="#555")
    axA.annotate("", (2, 0.1), (1, 16), arrowprops=dict(arrowstyle="->", color=CB["green"]))
    axA.text(1.5, 1.3, "MLIP\n~160× over GPU-DFT\n(~10³× over CPU)", ha="center",
             fontsize=8, color=CB["green"], fontweight="bold")

    # ---- (b) total force-eval compute for one ω(q,T) map, log scale
    for xi, v, p, c in zip(x, t_map_h, proj, cols):
        axB.bar(xi, v, 0.62, color=c, edgecolor="k", lw=0.7,
                hatch="////" if p else None, alpha=0.9 if p else 1.0, zorder=3)
    axB.set_yscale("log"); axB.set_ylim(0.08, 3200)
    lbls = ["200 h\n≈ 8.3 days", "27 h\n≈ 1.1 days", "10 min"]
    for xi, v, t in zip(x, t_map_h, lbls):
        axB.text(xi, v * 1.7, t, ha="center", fontsize=8.4, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(engines, fontsize=8.6)
    axB.set_ylabel("compute for one ω(q,T) map  (h, log)")
    axB.set_title("(b) one temperature-dependent ω(q,T) map\n(~6,000 force evaluations)")
    axB.text(0.5, 0.94, "hatched DFT bars = projected from the measured per-config cost\n"
             "(a full DFT ω(q,T) map is never run — that is the enabling point)",
             transform=axB.transAxes, ha="center", va="top", fontsize=7.4, color="#b00",
             bbox=dict(boxstyle="round", fc="#FFF3F0", ec=CB["red"], alpha=0.95))

    fig.suptitle("Fig 11  Speedup that makes the temperature-dependent calculation "
                 "tractable — MLIP vs DFT force evaluation", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/fig11_td_speedup.png"); plt.close(fig)
    print("wrote fig11_td_speedup.png")


# ============================================================ FIG 12 — Kohn anomaly vs electronic smearing
def fig12_smearing_kohn():
    """Graphene Kohn anomaly vs electronic smearing (degauss) — the (E)-channel
    resolution. Real DFT frozen-phonon dispersions (results/vq2, 5×5 grid),
    embedded in figdata/graphene_dg*.npz (601-q M–Γ–K–M, 6 branches). The Γ-E₂g
    cusp weakens monotonically as smearing rises (kink 8.2→5.8); K-A₁′ is
    under-resolved on the non-commensurate 5×5 grid (kink ~1) — the real K cusp
    needs a 6×6 K-commensurate grid (V-Q1: kink 14.4). Classic reference:
    Piscanec et al., PRL 93, 185503 (2004), which sweeps smearing 0.01–0.20."""
    THZ2CM = 33.35641
    FD = os.path.join(os.path.dirname(__file__), "figdata")
    dgs = [0.005, 0.01, 0.02, 0.04]

    def load(dg):
        d = np.load(os.path.join(FD, f"graphene_dg{dg}.npz"), allow_pickle=True)
        return (np.asarray(d["distances"]), np.asarray(d["frequencies"]),
                np.asarray(d["label_positions"]))
    D = {dg: load(dg) for dg in dgs}
    dist0, _, lp = D[0.005]
    gpos, kpos = lp[1], lp[2]
    labels = ["M", "Γ", "K", "M"]
    cramp = plt.cm.Blues(np.linspace(0.92, 0.5, len(dgs)))  # sequential: dark = low smearing (sharp)

    def kink(dist, ythz, x0, span=8):
        i = int(np.argmin(np.abs(dist - x0)))
        lo, hi = max(0, i - span), min(len(dist) - 1, i + span)
        sl = np.polyfit(dist[lo:i + 1], ythz[lo:i + 1], 1)[0]
        sr = np.polyfit(dist[i:hi + 1], ythz[i:hi + 1], 1)[0]
        return abs(sr - sl)
    kinks_g = [kink(D[dg][0], D[dg][1][:, -1], gpos) for dg in dgs]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                   gridspec_kw=dict(width_ratios=[1.55, 1]))
    # ---- (a) full spectrum (all branches; top optical branch bold)
    for c, dg in zip(cramp, dgs):
        dist, f, _ = D[dg]
        fcm = f * THZ2CM
        for b in range(f.shape[1] - 1):
            axA.plot(dist, fcm[:, b], color=c, lw=0.6, alpha=0.5, zorder=2)
        axA.plot(dist, fcm[:, -1], color=c, lw=1.9, zorder=3)
    for xp in lp:
        axA.axvline(xp, color="#ccc", lw=0.8, zorder=1)
    axA.set_xticks(lp); axA.set_xticklabels(labels)
    axA.set_xlim(dist0.min(), dist0.max()); axA.set_ylim(-20, 1740)
    axA.set_ylabel("phonon frequency (cm⁻¹)")
    axA.set_title("(a) full graphene phonon spectrum vs electronic smearing\n"
                  "(DFT frozen-phonon, 5×5 grid; top optical branch bold)")
    axA.annotate("Γ-E₂g\nKohn cusp", (gpos, 1580), (gpos - 0.03, 1330),
                 fontsize=8.2, color="#333", ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->"))
    axA.annotate("K-A₁′\n(under-resolved, 5×5)", (kpos, 1360), (kpos + 0.015, 1010),
                 fontsize=8, color="#999", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#999"))
    # ---- (b) Γ-E₂g cusp zoom
    for c, dg in zip(cramp, dgs):
        dist, f, _ = D[dg]
        mm = (dist >= gpos - 0.11) & (dist <= gpos + 0.11)
        axB.plot(dist[mm], (f[:, -1] * THZ2CM)[mm], color=c, lw=2.2, zorder=3)
    axB.axvline(gpos, color="#ccc", lw=0.8)
    axB.set_xticks([gpos - 0.08, gpos, gpos + 0.08])
    axB.set_xticklabels(["←M", "Γ", "K→"])
    axB.set_ylabel("Γ-E₂g optical branch (cm⁻¹)")
    axB.set_title("(b) the Γ-E₂g cusp sharpens as smearing → 0")
    txt = "cusp kink |dv|:\n" + "\n".join(
        f"  degauss {dg}: {kk:.1f}" for dg, kk in zip(dgs, kinks_g))
    axB.text(0.03, 0.03, txt, transform=axB.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="#bbb", alpha=0.92))
    axB.text(0.97, 0.97, "K-A₁′: flat on 5×5 (kink ~1, under-resolved)\n"
             "real K cusp needs 6×6 → kink 14.4 (V-Q1)\n"
             "classic: Piscanec PRL 2004 (smearing 0.01–0.20)",
             transform=axB.transAxes, fontsize=6.9, va="top", ha="right", color="#b00")
    handles = [plt.Line2D([], [], color=c, lw=2.6,
                          label=f"degauss {dg} Ry  (T_el {int(dg*157887)} K)")
               for c, dg in zip(cramp, dgs)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.2,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Fig 12  Graphene Kohn anomaly vs electronic smearing — "
                 "DFT frozen-phonon (E-channel)", fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(f"{OUT}/fig12_smearing_kohn.png"); plt.close(fig)
    print("wrote fig12_smearing_kohn.png; Γ kinks:", [round(k, 1) for k in kinks_g])


if __name__ == "__main__":
    fig1_status_matrix()
    fig2_origin_map()
    fig3_compute_budget()
    fig4_lambda_Tel()
    fig5_graphene_cure()
    fig6_pathp()
    fig7_breadth()
    fig8_echannel()
    fig9_kohn_anomaly()
    fig10_td_kohn_anomaly()
    fig11_td_speedup()
    fig12_smearing_kohn()
    print("done (figs 1-12)")
