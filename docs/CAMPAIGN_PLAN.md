# 1-Day Two-Machine Campaign Plan (Line A: breadth + statistics)

**Date:** 2026-06-19 · **Branch:** `exp-replay-ablation`

## Hardware
| Box | GPU | Role |
|---|---|---|
| `root@100.91.194.14` | **8× NVIDIA H20 96 GB** | Fine-tuning campaign (48 jobs, 8-way parallel) |
| `howardwang@100.105.21.7` | **1× RTX 2060 SUPER 8 GB** | Baseline foundation-model phonon benchmark (inference, fits 8 GB) |

8 GB can't train MACE (fixed ~7 GiB force double-backprop) → 2060 does inference-only baselines; all training on the H20s.

## Why these experiments
Prior ablations (replay / LoRA / data-efficiency) established:
- **Depth (configs/material) fixes in-domain but does NOT transfer.** in-domain MAE 0.45→0.27 THz as configs 5→60, but held-out MAE stays ~2.1–3.3 THz and the deepest run (ncfg60) is the *worst* held-out (3.27, overfitting).
- Replay pt=1000 is the robustness winner (held-out imaginary modes → 0 at zero in-domain cost).
- **Open question = breadth/transfer.** Nothing tried so far moves held-out MAE below ~2.1 THz.

So this campaign tests the **breadth axis** (number of distinct training materials) and puts **error bars** on every prior headline number.

## Material design
Cached MDR DFPT pool expanded 22 → **85 materials**. Nested subsets (greedy max-element-coverage):
`B4 ⊂ B8 ⊂ B16 ⊂ B32 ⊂ B64` (element coverage 9→16→20→42→43). `B16` = the original 16-material TRAIN set (continuity with prior results). **HOLD** = 10 fixed held-out materials, never trained on.

## Job groups (48 jobs, `scripts/jobs.tsv`)
| Grp | What | Jobs | Output figure |
|---|---|---|---|
| **G1** | **Breadth headline**: N∈{B4,B8,B16,B32,B64} × seed{1,2,3}, pt1000, 30cfg | 15 | depth-vs-breadth (the headline) |
| **G2** | **Seed error bars**: {single,pt1000,pt5000,lora8,lora32} × seed{2,3,4} on B16 | 15 | error bars on anti-forgetting + data-efficiency |
| **G3** | **Replay × breadth**: B32 × {single,pt500,pt1000,pt2500} × seed{1,2} | 8 | does breadth reduce the replay needed? |
| **G4** | **Depth × breadth**: {B16,B32,B64} × ncfg{15,60}, pt1000 | 6 | 2-D data-need surface |
| **G5** | **Bigger foundation**: {B8,B16,B32,B64}, model=medium, pt1000 | 4 | does a larger base need less data? |

Metric per job: in-domain (train) + **held-out transfer** freq MAE and imaginary-mode count vs DFPT (NAC-off, apples-to-apples).

## Execution
- `scripts/gpu_driver.sh` — 8 workers, each pinned to one GPU, round-robin over `jobs.tsv`, sequential per GPU. **Resumable** (skips jobs whose `eval_<job>.csv` exists). Detached via `nohup … & disown`. Logs → `results/ablation/driver.log`; per-job logs → `results/ablation/<job>.joblog`.
- `scripts/run_one_job.sh` — one fine-tune+eval, own data dir (no races), cleans bulky checkpoints after.
- **2060 track** — un-fine-tuned MatterSim / MACE-MP-0 / MACE-OMAT phonons across the 85-material pool → baseline "before" table.

Est. ~30 GPU-hours of training / 8 GPUs ≈ **4–6 h wall** for all 48; remaining day budget → extra seeds + larger benchmark breadth.

## Deliverables
1. `depth_vs_breadth.png` — held-out MAE vs #materials (G1) overlaid with vs configs/material (data-efficiency).
2. Error-bar versions of `antiforgetting_comparison.png` + `data_efficiency.png` (G2).
3. Replay×breadth and depth×breadth grids (G3,G4).
4. small-vs-medium foundation comparison (G5).
5. Expanded baseline benchmark table (2060).
