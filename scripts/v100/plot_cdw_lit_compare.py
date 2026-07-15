"""Side-by-side: OUR PBE fc2 CDW imaginary soft mode vs literature, 4 TMD monolayers.

Plots our Gamma-M-K-Gamma dispersion (dg0.005, experimental-a fc2 — the
literature-comparable set) zoomed to the acoustic/CDW region, marks the literature
CDW wavevector q_CDW, and annotates the literature source + our soft-mode depth.

Output: results/smearing_kink/cdw_lit_compare.png  (for WEEKLY_2026-07-15.md §5)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phonopy

LABELS = ["Γ", "M", "K", "Γ"]
POINTS = [np.array([0., 0., 0.]), np.array([.5, 0., 0.]),
          np.array([1/3, 1/3, 0.]), np.array([0., 0., 0.])]
CM = 33.35641

def bands(yaml_path, nseg=40):
    ph = phonopy.load(str(yaml_path), is_compact_fc=False)
    qs = []
    for i in range(len(POINTS) - 1):
        q0, q1 = POINTS[i], POINTS[i + 1]
        for j in range(1, nseg + 1):
            qs.append(q0 + (q1 - q0) * j / nseg)
    qs = np.array(qs)
    ph.run_qpoints(qs, with_dynamical_matrices=False)
    freq = np.array(ph.get_qpoints_dict()["frequencies"]) * CM   # cm^-1
    seglen = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(seglen)])
    seg = len(x) // 3
    tick = [x[0], x[seg - 1], x[2 * seg - 1], x[-1]]
    return x, freq, tick

# material: (display, yaml, q_CDW fraction along Gamma-M, our-depth, short lit citation)
MATS = [
    ("1T-VSe₂", ROOT/"results"/"vq_family"/"vse2"/"1T-VSe2_dg0.005.yaml", 0.50,
     "−73", "Lit: Fumega 2023 (PBE-DFPT) — imag ½ΓM(4×4) & ⅗ΓK(3×7)"),
    ("1H-NbSe₂", ROOT/"results"/"v100"/"fc2_nbse2_3x3_0.005"/"NbSe2_phonopy.yaml", 2/3,
     "−99", "Lit: Calandra 2009 + Weber PRL 107,107403 (PBE) — imag @ q_CDW≈⅔ΓM (EPC)"),
    ("1H-NbS₂", ROOT/"results"/"v100"/"fc2_nbs2_3x3_0.005"/"NbS2_phonopy.yaml", 2/3,
     "−165", "Lit: Wu 2022 (PRB 105 174105) — imag @ ⅔ΓM→3×3 (mono)"),
    ("2H-TaSe₂", ROOT/"results"/"v100"/"fc2_tase2_3x3_0.005"/"2H-TaSe2_phonopy.yaml", 2/3,
     "−97", "Lit: Shen 2023 (Nat.Commun. 14 7282, LDA-DFPT) — Fig 6a imag LA @ q≈0.35ΓM"),
]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
for ax, (name, yml, qfrac, ours, lit) in zip(axes.flat, MATS):
    x, freq, tick = bands(yml)
    ax.plot(x, freq, color="#1f5fa8", lw=1.1, alpha=0.9)
    ax.axhline(0, color="#aaa", lw=0.6)
    for xt in tick[1:-1]:
        ax.axvline(xt, color="#ddd", lw=0.6)
    ax.set_xticks(tick); ax.set_xticklabels(LABELS)
    # mark literature q_CDW on the Gamma-M segment (first segment)
    seglen = tick[1] - tick[0]
    qcdw_x = tick[0] + qfrac * seglen
    ax.axvline(qcdw_x, color="#c0392b", lw=1.3, ls="--", alpha=0.8)
    ymin = float(freq.min()); ymax = float(freq.max())
    ax.set_ylim(ymin * 1.12 - 5, ymax * 1.05)            # FULL range — nothing cut
    ax.set_xlim(tick[0], tick[-1])
    # shade the imaginary region
    ax.axhspan(ymin * 1.12 - 5, 0, color="#fdecea", alpha=0.5, zorder=0)
    ax.set_title(f"{name}    ours: {ours} cm⁻¹ (PBE fc₂, dg0.005, exp-a)", fontsize=10.5)
    ax.text(qcdw_x, ymax * 1.00, "q_CDW (lit)", color="#c0392b", fontsize=7.8,
            ha="center", va="top")
    # literature citation as a clean caption below the panel (no on-axes box -> no overlap)
    ax.text(0.5, -0.16, lit, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.2, color="#333", style="italic")
    ax.set_ylabel("frequency [cm⁻¹]", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("CDW imaginary soft mode: our PBE fc₂ vs literature  (pink = imaginary; red dashed = lit q_CDW)",
             fontsize=12.5, y=0.995)
fig.tight_layout(rect=[0, 0.02, 1, 0.985])

fig.suptitle("CDW imaginary soft mode: our PBE fc₂ vs literature  (pink = imaginary; red dashed = lit q_CDW)",
             fontsize=12.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = ROOT / "results" / "smearing_kink" / "cdw_lit_compare.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
