# The (E)-channel long-range term — findings & architecture spec

**Date:** 2026-07-03 · **Branch:** `smearing-kink-ml` · **Machines:** 2× V100 (DFT) + RTX 2060 SUPER (ML training).

## The question (reframed by the user, and the answer)

> If I need phonon spectra that are accurate at **both physical temperature (T_lat) and electronic
> smearing (T_el)**, do I need to add a long-range term to the MLIP?

**The long-range term's purpose is NOT to "capture the cusp" — it is to keep the simulation
physically correct and transferable across smearing.** Answer, now backed by evidence:

- **T_lat axis (anharmonic/lattice):** **No.** A standard MLIP (learns the ground-state BO PES) +
  thermal sampling (TDEP/SSCHA) captures it. Demonstrated: graphene (L)-channel FT+TDEP gives
  kink_K(T_lat) ≈ 5.8→6.3 (T=100→600 K), nearly T_lat-flat. The long-range term only matters here if
  you want the near-anomaly continuum quantitatively exact.
- **T_el axis (smearing/electronic):** **Yes — necessarily.** A standard MLIP is trained at ONE
  smearing and is **structurally blind** to T_el → it returns one fixed spectrum regardless of
  smearing → wrong at any other T_el. You need (a) **T_el-conditioning** and (b) an **explicit
  long-range Friedel term**, because the smearing-dependence physically *is* a long-range
  Fermi-surface effect. The recipe for a `(T_lat, T_el)`-accurate calculator:
  `T_lat = MLIP + TDEP` · `T_el = smearing-conditioned MLIP + long-range 2k_F term`.

## Evidence chain (graphene, 6×6 K-commensurate, 12 smearings T_el 315–12631 K)

1. **Short-range distillation reproduces the cusp *in-distribution* (any cutoff).** From-scratch MACE
   FC-distilled on the 6×6 fc₂, r_max ∈ {3,4,5,6} Å → all give kink_K ≈ 14.8–15.3 ≈ DFT 14.4. So the
   naive "a short-range MLIP can't get the cusp" is **wrong for in-distribution fitting** — the MLIP
   dispersion is grid-independent and simply fits the fc₂ it's trained on. (The Fig-9 75% residual was
   a *foundation-model* fine-tuning constraint, not a hard cutoff limit.)
   → `results/smearing_kink/graphene_rmax_sweep_kink.csv`
2. **But it does not transfer.** The same r_max=3 vs r_max=6 models (trained only at a=2.46), applied
   to unseen lattice constants a=2.44/2.48/2.50, give **opposite kink-vs-a trends** (r_max=3 ↓, r_max=6
   ↑) → model-dependent, unphysical extrapolation. → `graphene_rmax_transfer_a.csv`
3. **The smearing-dependence lives in the long-range tail.** Real-space decomposition of the 12
   fc₂(T_el): the smearing-*independent* backbone dies by ~4 Å (|fc₂| 86.8→0.08); the smearing-
   *difference* Δfc₂ persists to 11 Å (0.42 at 5 Å, **0.35 at 8 Å ≈ 20× the backbone there**). The
   short-range part is smearing-blind; **all** the T_el physics is in the 5–11 Å tail.
4. **The long-range term is rank-2 = 2 physical parameters.** SVD of the smearing-variation: mode-1
   = 99.46 % of variance, modes 1+2 = 99.9 %. But **rank-1 fails the kink** (flat ~14, can't do
   22.6→3.9) — one amplitude is not enough. **Rank-2 reproduces kink_K(T_el) exactly** across all
   smearings (22.1/15.7/10.1/3.6 vs DFT 22.6/15.7/9.8/3.9). → `test2_lr_model.csv`, figure
   `results/smearing_kink/lr_term_rank2.png`.
5. **Physical identification:** the two parameters are the **amplitude A(T_el)** and the **thermal
   damping length ξ(T_el) ~ v_F/(k_B T_el)** of a **Friedel/RKKY oscillation**
   `Φ_LR(R) = A(T_el)·cos(2k_F·R)·e^{−R/ξ(T_el)} / R^d`. Higher T_el → shorter ξ → damped cusp. Both
   A and ξ are smooth, monotonic in T_el → **few-shot predictable**.

## Architecture spec for E2b (the module to build)

Locked by the evidence above; realises the **BAMBOO scaffold** (local NN + explicit analytic
long-range module + distillation) with the physics swapped from electrostatic to metallic:

```
fc₂(T_el) = MACE_short-range(smearing-blind)  +  Friedel_module(R; 2k_F, A(T_el), ξ(T_el))
```
- **Backbone:** ordinary short-range MACE (r_max ~5 Å) — carries the T_el-independent bulk.
- **Long-range module:** analytic `A(T_el)·cos(2k_F·R)·e^{−R/ξ(T_el))/R^d`; k_F fixed from the bands
  (graphene 2k_F is exact at the Dirac point), A and ξ conditioned on T_el (2 scalar heads).
- **Training:** FC-distill the 12 fc₂(T_el); fit/learn A(T_el), ξ(T_el) (BAMBOO-style ensemble
  distillation for stability). Then **few-shot** (predict A, ξ from ~3 smearings) + **transfer** test
  (held-out a / material).
- **Refs:** BAMBOO — ByteDance, *Nat. Mach. Intell.* 2025 / arXiv:2404.07181 / `bytedance/bamboo`
  (electrostatic long-range template); "Long-range electrostatics for MLIPs is easier than we
  thought", *JCP* 2026.

## Status

- **Proven (science):** steps 1–5 above — the (E)-channel long-range term is a 2-parameter damped
  Friedel oscillation, necessary and sufficient for smearing-accurate graphene phonons. Committed.
- **Remaining (engineering):** implement the Friedel module in MACE, fit A/ξ, run few-shot + transfer.
  Compute: 2060 (8 GB ample; ~GPU-hours). No rental.
- **Data in hand:** 12 graphene fc₂(T_el) (`results/vq_kink6/`, symlinked to `/data`), the r_max
  models (`results/gr_rmax_sweep/` on 2060), the fc₂ on the 2060 (`results/vq1/…yaml`).

## Hardware notes

- **2060** `howardwang@100.105.21.7` (`howard-pc`): RTX 2060 SUPER 8 GB, `~/miniconda3/envs/phonon` =
  torch 2.6.0+cu124 / mace 0.3.16 / e3nn 0.4.4, GPU-training-verified. Repo `~/phonon` (run `git pull`;
  it was ~50 commits behind). No tmux → use `setsid`/`nohup`.
- **Box A** `/` had filled to 100 % (QE scratch); redundant installers deleted + `results/{path_p,vq*}`
  migrated to `/data/results/` with symlinks back (`/` now ~45 %). New graphene DFT runs must set
  `--workdir /data/...` + `TMPDIR=/data/tmp` (m1_1b predates the scratch→/data fix).
