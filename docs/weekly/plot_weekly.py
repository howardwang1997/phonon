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
        [3,3,3,3,3],   # NbSe2  — all done (Part-I flagship)
        [3,3,3,2,1],   # 2H-TaSe2 — fc2/bands/PathP done, EPW running (box B), SSCHA pending
        [3,3,2,2,1],   # NbS2 — fc2/bands done, PathP running, EPW running (box A)
        [2,1,1,1,1],   # 1T-TiSe2 — fc2 running (box B)
        [1,1,1,1,1],   # 2H-TaS2
        [1,1,1,1,1],   # 1T-TaS2
        [1,1,1,1,1],   # 1T-VSe2
        [1,1,0,0,0],   # 1T-TiS2 (control) — fc2/bands only
        [1,1,0,0,0],   # 1T-VS2  (control)
        [1,1,0,0,0],   # MoS2    (gapped control)
        [1,1,0,0,0],   # WSe2    (gapped control)
    ])
    # annotations: captured DFT soft modes (THz) on fc2-done cells
    soft = {(0,0): "−2.18", (1,0): "−2.18", (2,0): "−1.89"}
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
    ax.set_title("Fig 1  V100 FP64 campaign — status matrix (2026-07-01)")
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


if __name__ == "__main__":
    fig1_status_matrix()
    fig2_origin_map()
    fig3_compute_budget()
    fig4_lambda_Tel()
    fig5_graphene_cure()
    fig6_pathp()
    fig7_breadth()
    fig8_echannel()
    print("done (figs 1-8)")
