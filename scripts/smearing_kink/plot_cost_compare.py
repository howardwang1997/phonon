"""Compute cost comparison: DFT vs MLIP+fine-tuning for 0K / smearing / temperature phonon spectra.
Bar chart + table. Based on actual timings from this campaign (NbSe2 3×3 family + graphene 8×8).
Run: conda run -n phonon python scripts/smearing_kink/plot_cost_compare.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SK = Path(__file__).resolve().parents[2] / "results" / "smearing_kink"

# Timings (hours). Based on actual campaign measurements.
# NbSe2 3×3 (27 atoms) — representative CDW family material
# Graphene 8×8 (128 atoms) — large non-CDW

tasks = ["0K fc₂\n(1 spectrum)", "Smearing\n(4 degauss)", "Temperature\n(4 T_lat)"]

# DFT times (V100 GPU)
dft_nbse2 = [1.5, 6.0, 34.0]       # 1×fc2; 4×fc2; 4×DFT-MD(8.5h each)
dft_graphene = [3.0, 6.0, 34.0]    # 1×fc2(8×8); 2×fc2; 4×DFT-MD

# MLIP times: training (one-time) + inference (per query)
# NbSe2: backbone distill ~1h train; inference ~10min/fc2, ~2min/deploy, ~5min/TDEP
mlip_nbse2_train = [1.0, 0.0, 0.0]     # one-time; amortized for subsequent
mlip_nbse2_infer = [0.17, 0.07, 0.13]  # 10min fc2; 2min Friedel deploy ×4; 2min TDEP ×4
# Graphene: backbone distill ~4h (v11); inference similar
mlip_graphene_train = [4.0, 0.0, 0.0]
mlip_graphene_infer = [0.17, 0.07, 0.13]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, (mat, dft, train, infer, title) in zip(axes, [
    ("NbSe₂ (2H CDW, 3×3)", dft_nbse2, mlip_nbse2_train, mlip_nbse2_infer, "NbSe$_2$ (2H CDW, 3×3, 27 atoms)"),
    ("graphene (non-CDW, 8×8)", dft_graphene, mlip_graphene_train, mlip_graphene_infer, "graphene (non-CDW, 8×8, 128 atoms)")
]):
    x = np.arange(len(tasks))
    w = 0.25
    bars_dft = ax.bar(x - w, dft, w, label="DFT (V100)", color="#888", edgecolor="white")
    bars_tr = ax.bar(x, train, w, label="MLIP train (one-time, 2060)", color="#2f6f9f", edgecolor="white")
    bars_inf = ax.bar(x + w, infer, w, label="MLIP inference (2060)", color="#c0504d", edgecolor="white")

    for i, (d, t, inf) in enumerate(zip(dft, train, infer)):
        speedup = d / (t + inf) if (t + inf) > 0 else float('inf')
        txt = f"{speedup:.0f}×" if speedup < 1000 else f"{speedup/1000:.0f}k×"
        ax.annotate(txt, (x[i], d + 0.5), ha="center", fontsize=9, fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("time [hours]" if ax == axes[0] else "")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_yscale("log")
axes[0].set_ylim(0.01, 100)
fig.suptitle("Compute cost: DFT vs MLIP+fine-tuning (speedup annotated)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(SK / "compute_cost_compare.png", dpi=150, bbox_inches="tight")
print("wrote", SK / "compute_cost_compare.png")

# Also print a summary table
print("\n=== Compute Cost Summary ===")
print(f"{'Task':<25} {'DFT (h)':<10} {'MLIP train (h)':<15} {'MLIP infer (h)':<15} {'Speedup':<10}")
print("-" * 75)
for mat, dft_arr, tr_arr, inf_arr in [("NbSe₂", dft_nbse2, mlip_nbse2_train, mlip_nbse2_infer),
                                       ("graphene", dft_graphene, mlip_graphene_train, mlip_graphene_infer)]:
    for i, t in enumerate(tasks):
        d, tr, inf = dft_arr[i], tr_arr[i], inf_arr[i]
        su = d / (tr + inf) if (tr + inf) > 0 else float('inf')
        print(f"{mat+' '+t.replace(chr(10),' '):<25} {d:<10.1f} {tr:<15.1f} {inf:<15.2f} {su:<10.0f}×")
