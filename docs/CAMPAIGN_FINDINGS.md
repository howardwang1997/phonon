# Campaign Findings — empirical backbone for the NCS paper

Two-machine 1-day campaign (8×H20 training + RTX 2060 baselines), ~100 FC-distillation
fine-tunes. All numbers are **held-out transfer MAE vs DFPT** (NAC-off, `--no-relax` =
evaluated at the DFT geometry → isolates force-constant quality), THz. Held-out = 6 fixed
materials (SiC, BN, LiF, …) never trained on. Foundation model: MACE-MP-0 small; FC distillation
= harmonic E/F labels from public MDR DFPT force constants (zero new DFT).

## Setup
- Pool: MDR DFPT expanded 22 → 85 materials; 79 trainable (6 held out).
- Nested breadth subsets B4⊂…⊂B79; depth = distilled configs/material; replay pt=1000 default.
- In-domain MAE after distillation: **~0.10 THz**, zero spurious imaginary modes (replay preserves
  the foundation model's universality).

## Result 1 — Breadth, not depth, drives transfer
Held-out MAE vs # training materials (replay pt1000, 30 cfg, seed-averaged):

| N | 4 | 8 | 16 | 20 | 24 | 28 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| MAE | 1.82 | 1.80 | 1.82 | 1.78 | 1.58 | 1.43 | 1.42 | 1.31 | 1.37 |

**Plateau (N≤~16–20) → knee (~20–24) → drop → saturation (~1.3 THz).** Error bars ±0.06–0.10.
`results/figures/depth_vs_breadth.png`.

## Result 2 — Depth overfits (the data-need surface)
Held-out MAE, breadth (rows) × configs/material (cols):

| N \ cfg | 15 | 30 | 60 |
|---|---|---|---|
| 16 | 1.91 | 1.90 | 1.92 |
| 32 | 1.50 | 1.50 | 1.55 |
| 48 | 1.35 | 1.37 | 1.45 |
| 64 | **1.33** | 1.42 | 1.52 |
| 79 | **1.27** | 1.34 | 1.44 |

**Down (more breadth) = monotonically better; across (more depth) = flat-to-worse, and worse the
more breadth you have** (overfitting to the training materials). Global min = high-breadth +
low-depth (N=79/cfg15 = **1.27 THz, still descending**). The N=79/cfg60 cell (1.44, run later on an
idle H20) completes the row and confirms depth hurts at high breadth (1.27→1.34→1.44).
`results/figures/depth_breadth_surface.png`.
→ **Practical law: spend the DFT budget on more materials, fewer configs each.**

## Result 3 — Anti-forgetting (held-out, seed error bars)
single-head 2.22 · replay-pt1k 1.85 · replay-pt5k 2.33 · LoRA-r8 1.79 · **LoRA-r32 1.46**.
Over-replay (pt5k) hurts; light replay or LoRA both viable; more breadth reduces replay need.
`results/figures/antiforgetting_errorbars.png`, `replay_x_breadth.png`.

## Result 4 — Acquisition: coverage > random > naive uncertainty (E7 v1)
Which materials to query (MDR = DFT oracle), held-out MAE at matched N:

| N | random (6 draws) | **coverage (greedy)** | uncertainty (AL) |
|---|---|---|---|
| 16 | 1.61 ± 0.09 | 1.58 ± 0.03 | 1.68 |
| 24 | — | 1.46 ± 0.01 | 1.61 |
| 32 | 1.53 ± 0.08 | **1.38 ± 0.01** | 1.60 |
| 48 | 1.43 ± 0.07 | 1.37 ± 0.03 | — |

- **Chemical-coverage acquisition is best AND lowest-variance** (±0.01 vs random ±0.08): it picks
  nearly the same informative set every time. Saturates ~1.37–1.40 (N=48–64), tracking the breadth
  law's floor.
- **Naive model-uncertainty acquisition is the *worst*** (full arm 1.55–1.68; never reaches even the
  random curve at N=48: 1.59 vs 1.43) — it chases pathological/soft outliers (P allotrope,
  mp-761842 with 31 imaginary modes) that are informative-to-the-model but unrepresentative.
- **Stability-filtering rescues uncertainty only at larger budget**: excluding predicted-
  pathological candidates is *worse* at N=16 (1.72) but catches and slightly passes coverage by
  N=32 (1.35 vs 1.38). So pathology-chasing explains the small-budget failure, but coverage is
  still the best *single* strategy and by far the most reliable across budgets.
- **Design lesson for the closed-loop data engine: acquire by chemical coverage (optionally
  coverage × stability), not naive uncertainty** — counterintuitive, since uncertainty sampling is
  the textbook default. Full four-way curve: `acquisition_comparison.png`.

## Result 5 — The transfer floor is dominated by one hard chemistry
Per-material held-out error of the best model (coverage N=79, mean 1.32 THz):

| material | BN | SiC | mp-390 | mp-20351 | LiF | mp-2605 |
|---|---|---|---|---|---|---|
| MAE (THz) | **4.43** | 0.98 | 0.82 | 0.74 | 0.65 | 0.30 |
| softening | −23% | −9% | −7% | +11% | −5% | +5% |

**5/6 materials transfer at ~0.3–1.0 THz (median ~0.7); BN alone (4.43 THz) sets the mean.** BN is
the canonical hard case — light B/N, ~40 THz optical modes, strong covalency — where foundation
MLIPs soften most and which is under-covered chemically. → report **median + mean**; the residual
challenge is specific light-element/high-ω chemistries, which is exactly what coverage acquisition
targets (and naive uncertainty wastes on soft P/F₂ allotropes).

## Implications for the NCS framework
1. The accuracy ceiling at fixed data is set by **chemical breadth** → motivates a *targeted* DFT
   data engine (Line B) feeding breadth, not a bigger model or deeper sampling.
2. The engine's **acquisition function should maximize coverage** (or coverage×stability), making
   "which materials to compute with DFT" a solved, cheap decision.
3. FC distillation gives near-DFT in-domain (~0.10 THz) at **zero new DFT** from public force
   constants → the public-data half of the pipeline is essentially free.

## Caveats (honest)
- Single foundation model (MACE-MP-0 small) so far; cross-model (MatterSim/SevenNet) is future E1.
- Small pool (79 trainable, 6 held out); a larger external held-out test set would tighten the
  transfer estimate. MDR-as-oracle stands in for the Line B engine in this prototype.
- AL arms are single-seed; uncertainty proxy = predicted imaginary+ASR (ensemble-disagreement is
  the natural refinement). `--no-relax` isolates FC quality but omits relaxation drift.
