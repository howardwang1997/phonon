# Findings log

## 2026-06-18 — M0 + first Line-A result

### Pipeline validated (Si worked example)
Unified `phonopy + ASE` engine runs end-to-end on Si with MACE-MP-0:
fully converged across supercell size (2³/3³/4³) and displacement (0.01/0.03 Å)
— Γ optical = **11.19–11.21 THz**, acoustic modes at Γ = 0 (ASR satisfied,
residual ~1.6e-7 THz), no imaginary modes, C_v(300 K) = 43.5 J/K/mol.
The engine is correct; the frequency offset below is a *model* property.

### Systematic phonon softening of MACE-MP-0 (the Line-A gap)
Built-in benchmark (`scripts/run_benchmark.py`, 3³ supercell, Γ-centered 24³
mesh), MACE-MP-0 *medium* (2023) vs literature/DFT highest phonon frequency:

| material | ω_max MLIP (THz) | ω_max ref (THz) | softening |
|----------|-----------------:|----------------:|----------:|
| Si       | 11.20 | 15.5 | −27.7% |
| Ge       |  5.40 |  9.1 | −40.7% |
| C (diam) | 35.10 | 39.9 | −12.0% |
| GaAs     |  6.37 |  8.8 | −27.6% |
| NaCl     |  6.44 |  7.9 | −18.5% |
| MgO      | 16.67 | 21.4 | −22.1% |
| Al       |  6.62 |  9.9 | −33.1% |

**Mean softening −26%**, 0 materials with imaginary modes, all ASR-clean.

Consistent with the literature (Loew et al., *Overcoming systematic softening
in uMLIPs by fine-tuning*). This is the precise, quantitative motivation for
Line A: fine-tune to remove the systematic frequency underestimation while
keeping the 50–1000× speedup.

### Genuine DFPT reference wired in (MDR / PhononDB, no API key)
`src/phonon_accel/reference.py` + `data/benchmark/mdr_index.csv` index the
**10,034-material** MDR phonon database (Togo/NIMS, VASP+phonopy DFPT). Each
material's `phonopy_params.yaml.xz` downloads from
`https://mdr.nims.go.jp/download_all/<dataset_id>.zip` and loads directly with
`phonopy.load` — giving true per-q DFPT force constants for full-band
comparison. This is the reference set used by the foundation-MLIP phonon
literature (Loew et al. 2025).

### Full-band MLIP-vs-DFPT benchmark — MatterSim-v1 (14 crystals)
`scripts/run_dfpt_benchmark.py`, MLIP run on the DFT-reference cell with the
*same* supercell matrix → identical seekpath q-path → point-for-point compare.

| metric | value |
|--------|-------|
| freq MAE (mean / median / max) | **0.58 / 0.53 / 1.20 THz** |
| freq RMSE (mean) | 0.97 THz |
| ω_max error (mean) | −1.31 THz |
| softening (mean, range) | **−7.2%** (−19.9 … +0.7) |
| C_v(300 K) error (mean abs) | 1.01 J/K/mol |
| S(300 K) error (mean abs) | 2.08 J/K/mol |
| dynamically stable (predicted) | **14/14** |

Materials: Si, MgO, NaCl, AlN, GaN, SiC, ZnS, BN, LiF, CaO, AlAs, InP, ZnO,
TiO₂ (`results/dfpt_mattersim_curated.csv`). MatterSim is ~4× more accurate
than MACE-MP-0 and softens far less (−7% vs −26%) — matches the literature
ordering. Note: 2 DFPT references (BN, TiO₂) **themselves** carry imaginary
modes (reference-quality caveat for the benchmark).

Verification figure: `results/figures/Si_mattersim_vs_dfpt.png` — acoustic
branches essentially exact, optical branches mildly softened.

### Environment strategy (resolved e3nn conflict)
MACE 0.3.16 needs `e3nn==0.4.4`; MatterSim pulls `e3nn 0.6.0` → they cannot
share one env. Solution: **per-model conda envs**, pipeline code shared via
`sys.path` (env-agnostic, lazy model import):
- `phonon`      — MatterSim (+ core stack)  ← workhorse / fine-tuning
- `phonon-mace` — MACE (e3nn 0.4.4), isolated baseline

## 2026-06-18 — A3 fine-tuning (FC distillation): method + key findings

### Method (zero new DFT)
Distill the DFPT second-order force constants Φ from MDR into the foundation
model: per training material, generate rattled supercells labeled with exact
harmonic forces `F = -Φ·u` and energy `E = ½uᵀΦu` (`finetune/dataset.py`).
Force-matching these injects the DFPT curvature. **Round-trip verified exact**:
distilled forces fed through the finite-displacement pipeline reproduce the
DFPT dispersion (Si 15.287 → 15.287 THz). Fine-tune MACE-MP-0 (small) with
forces weighted 100×, energy 0.01× (distilled energies are relative).

### Result — Si (training material, controlled)
| | ω_max (THz) | softening | freq MAE |
|---|---|---|---|
| baseline MACE-MP-0 | 11.47 | −25.0% | 2.04 THz |
| **fine-tuned** | 15.17 | **−0.8%** | **0.30 THz** |
| DFPT | 15.29 | — | — |
Validation force RMSE 42 → ~7 meV/Å (≈6×). FC distillation essentially
removes the softening for Si; AlN/ZnS/ZnO also improved.

### Two methodological findings (both matter for the paper)
1. **Element coverage** — naively restricting `--E0s` to the training
   elements collapses MACE's element table, so the fine-tuned model *cannot
   run* on held-out materials with unseen elements. Fix:
   `--foundation_model_elements True` (keep the foundation's full periodic
   table). Essential for any generalization claim.
2. **NAC / Born charges** — short-range MLIPs have no Born effective charges,
   so they cannot reproduce LO–TO splitting. The MDR DFPT reference has it on
   by default. Comparing NAC-on reference vs NAC-off MLIP unfairly penalizes
   every polar crystal and is *unfixable* by short-range fine-tuning
   (MgO Γ-region 20.8 vs 11.7 THz!). Fix: load reference NAC-off
   (`reference.load_reference_phonon(is_nac=False)`) → isolates the
   short-range force constants the fine-tuning actually targets. (No effect on
   non-polar Si.) Full-dispersion comparison would instead graft DFT Born
   charges onto the MLIP — a documented future option.

### Training cost reality (CPU)
float64 MACE on CPU ≈ 16 min/epoch (impractical). Switched to float32
(accurate enough for finite-displacement FCs: noise ~1e-6 « FC scale)
→ ~90 s/epoch. The GPU at 100.105.21.7 would cut this ~10–50×; it remains the
key unblock for scaling fine-tuning and Line B.

### Corrected before/after (full elements + NAC-off) — HONEST mixed result
Final training force RMSE 6.6 meV/Å. `results/finetune_eval.csv`,
figure `results/figures/Si_before_after.png` (fine-tuned overlaps DFPT).

| split | material | MAE before→after (THz) | softening before→after |
|---|---|---|---|
| train | Si | 2.04→**0.30** | −25.0→**−0.8%** ✓✓ |
| train | AlN | 1.22→**0.14** | −10.4→**−2.9%** ✓✓ |
| train | ZnS | 0.83→0.36 | −13.8→−7.9% ✓ |
| train | MgO | 0.40→1.07 | −6.3→−7.6% ✗ |
| train | NaCl | 0.18→0.61 | −2.6→−11.3% ✗ |
| holdout | ZnO | 0.67→**0.24** | −3.2→−4.0% ✓ |
| holdout | InP | 0.45→0.40 | −2.6→−0.3% ✓ |
| holdout | SiC | 1.61→1.40 | −11.9→−9.5% ✓ |
| holdout | AlAs | 1.03→0.88 | −20→−20% ~ |
| holdout | CaO | 0.42→0.42 | +4.1→−6.6% ~ |
| holdout | GaN | 0.84→1.67 | −8.5→−4.9% ✗ |
| holdout | TiO₂ | 1.08→1.43 | −0.8→−7.9% ✗ |
| holdout | LiF | 0.88→**2.59** | −12.1→**−32.9%** ✗✗ |
| holdout | BN | 4.65→**6.90** | −24→**−36%** ✗✗ |

Aggregate: **train** MAE 0.93→0.49, softening −11.6→−6.1% (improved);
**holdout** MAE 1.29→1.77, softening −8.8→−13.6%, and **50 spurious
imaginary modes** appear (baseline had 0).

**Interpretation (this is the science).** FC distillation *works* — for
in-distribution materials it snaps the dispersion onto DFPT (Si −25→−0.8%,
AlN −10→−3%; the figure shows blue≈black). Held-out tetrahedral semiconductors
(ZnO, InP, SiC, AlAs) also improve → generalizes *within a structural family*.
BUT naive single-head fine-tuning on only 5 materials causes **catastrophic
forgetting**: ionic/OOD crystals (LiF, BN, MgO, NaCl) degrade and spurious
imaginary modes appear out-of-distribution. This is the expected small-data
failure mode and motivates the established fix:
**multihead fine-tuning with foundation-data replay**
(`--multiheads_finetuning True` / `--pt_train_file`), **more & more diverse
training materials**, and **LoRA/PEFT** (`--lora_rank`) to bound drift — all
supported by the toolchain, all practical once the **GPU** is online.

### Caveats / next steps

### Caveats / next steps
- Scale benchmark: random N from MDR (not just easy simple crystals) for an
  honest distribution; add SevenNet / ORB-conservative / CHGNet / M3GNet
  (own envs) and the MACE baseline on the *same* DFPT materials.
- Plot polish: x-axis high-symmetry labels (Γ/X/K/L) not yet wired.
- A3 idea: MDR force constants (Hessian) for 10k materials = ready-made
  fine-tuning target for the phonon/Hessian loss — fine-tune MatterSim,
  hold out materials for validation.
- Run on remote CUDA GPU (float64) for throughput once reachable.

## 2026-06-18 — A3 full-scale on H20 (16 train + 6 held-out): forgetting localized to UNSEEN elements

Proper full-diverse fine-tune on an H20 (96 GB) — no size filter, full
16-material distilled dataset (912 configs), single-head + full element
coverage, batch 32, ~6 s/epoch. (Multihead MP-replay intended but its replay
download is blocked from the China host; NIMS/pytorch.org also blocked →
relayed MDR cache via tar-pipe, pinned cu124 torch via SJTU mirror.)
`results/finetune_eval_h20.csv`.

**Training set (16) — FC distillation works spectacularly:**
| | mean MAE | mean \|soft\| | total imaginary |
|---|---|---|---|
| baseline MACE-MP-0 | 0.82 THz | 9.6% | 291 (SrTiO₃) |
| **fine-tuned** | **0.28 THz** | **3.6%** | **4** |
SrTiO₃ 291→4 imag; Li₂O MAE 2.01→0.12; Al₂O₃ 0.34→0.10; Si −25→−2.7%.

**Held-out (6) — failure precisely localized to UNSEEN elements:**
| material | coverage | MAE before→after | imag |
|---|---|---|---|
| LiF | all-trained | 0.88→0.77 ✓ | 0→0 |
| CaO | all-trained | 0.42→0.53 ~ | 0→0 |
| TiO₂ | all-trained | 1.08→0.75 ✓ | 0→0 |
| InP | In unseen | 0.45→0.62 ✗ | 0→0 |
| SiC | C unseen | 1.61→5.53 ✗✗ | 0→542 |
| BN | B unseen | 4.65→14.1 ✗✗ | 0→858 |

**The science.** Diverse-data FC distillation (a) dramatically fixes phonons for
trained materials (3× MAE↓, softening 9.6→3.6%, imaginary 291→4); (b)
**generalizes to held-out materials whose elements were all in training**
(LiF/CaO/TiO₂ — LiF *improved* here vs *degraded* in the 5-material CPU PoC →
more diverse training fixed within-element generalization); (c)
**catastrophically forgets elements NOT seen in fine-tuning** (C in SiC, B in BN
→ hundreds of spurious imaginary modes). Fix = multihead-replay (needs non-China
GPU or staged replay data) and/or element-coverage extension / LoRA.

## 2026-06-19 — A3 multihead-replay vs single-head (H20): robustness/accuracy trade-off

Ran the proper **multihead-replay** fine-tune on the H20 (the anti-forgetting
method the 8GB card + China-blocked replay download both prevented). Worked
around the block: downloaded the 595 MB `mp_traj_combined.xyz` on the Mac,
**subsampled to 13k structures with guaranteed C/B/In coverage** (54 MB),
relayed to the H20's MACE cache (`~/.cache/mace/mp_traj_combinedxyz`) so MACE
skips the download. Two heads: `Default` (FC distillation, 912 configs) +
`pt_head` (MP replay, 4,500 structures). `results/finetune_eval_h20_mh.csv`.

**KEY WIN — replay eliminates the catastrophic imaginary-mode forgetting on
unseen elements:**
| held-out (unseen elem) | baseline imag | single-head imag | multihead imag |
|---|---|---|---|
| SiC (C) | 0 | **542** | **0** ✓ |
| BN (B) | 0 | **858** | **0** ✓ |
(BN MAE also recovered 14.1→7.7.) The replay head anchoring C/B/In removed the
spurious dynamical instabilities — exactly its purpose.

**TRADE-OFF — replay dampens peak accuracy + beneficial transfer:**
| | train MAE | train imag | held-out all-seen (LiF/CaO/TiO₂ MAE) |
|---|---|---|---|
| single-head | 0.28 | 4 | 0.77 / 0.53 / 0.75 (improved) |
| multihead   | 0.30 | 15 | 1.56 / 0.83 / 2.33 (worse) |

**Takeaway (publishable nuance):** replay buys **robustness** (no catastrophic
instabilities on unseen chemistry — essential for high-throughput screening) at
the cost of some **accuracy/transfer**. Neither uniformly best: single-head wins
on known chemistries, multihead wins on safety. The replay/Default balance
(sample counts, loss weights) is the obvious next tuning knob — and would make a
clean ablation axis for the paper.

## 2026-06-19 — A3 replay-strength ablation (exp-replay-ablation branch)

Swept `num_samples_pt ∈ {0,1000,2500,5000}` (single-head → full multihead),
50 epochs, identical otherwise. `results/ablation/eval_pt*.csv`.

| config | train MAE | SiC imag (C unseen) | BN imag (B unseen) | held-out-seen MAE (LiF/CaO/TiO₂) |
|---|---|---|---|---|
| baseline | 0.82 | 0 | 0 | 0.88/0.42/1.08 |
| single-head (pt=0) | 0.33 | **1269** | **1421** | 1.18/0.53/0.77 |
| **replay pt=1000** | **0.32** | **0** | **0** | 1.27/0.70/1.59 |
| replay pt=2500 | 0.32 | 0 | 0 | 1.35/0.85/2.15 |
| replay pt=5000 | 0.33 | 0 | 0 | 1.41/0.78/2.22 |

**Headline:** (1) **minimal replay (pt=1000, ~20% of data) already fully fixes
the catastrophic unseen-element forgetting** (SiC 1269→0, BN 1421→0); (2) at
**zero in-domain cost** (train MAE flat ~0.32); (3) **more replay only hurts
held-out transfer** (TiO₂ 1.59→2.22). → Sweet spot ≈ pt=1000; "a little replay
goes a long way" is a clean paper message.

Open issue: held-out-but-seen materials (LiF/CaO/TiO₂) stay *worse than baseline*
even at the sweet spot → FC distillation on 16 materials transfers poorly to new
materials (even seen-element ones); needs more **training materials** (breadth),
which the data-efficiency / material-count sweep probes next.
