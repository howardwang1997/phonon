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

## Module BUILT & VALIDATED (2026-07-03)

Implemented and validated the module on the 12 graphene fc₂(T_el) (CPU numerics, local `phonon`
env; no GPU/rental needed). Code: `scripts/smearing_kink/friedel_module.py` (geometry + kernel +
fc assembly + kink readout), `fit_friedel.py` (fit + few-shot), `transfer_friedel.py` (cross-a),
`plot_friedel.py` / `plot_transfer.py`.

**Realized model** (the physically-correct form, after a decisive diagnostic):
```
fc₂(R; T_el) = fc₂_backbone(dg0.080)  +  B(T_el) · exp(−κ(T_el)·R) · D0(R)
```
- `D0(R)` = the **full 3×3-tensor** Fermi-surface Friedel waveform, measured once at the sharpest
  smearing (`fc₂(dg0.002) − fc₂(dg0.080)`). Correct directionality + tensor structure baked in.
- Two thermal parameters per smearing: amplitude `B(T_el)` and **extra damping rate**
  `κ(T_el) = 1/ξ(T_el) − 1/ξ_ref`. Applied with the ASR self-term → a genuine, ASR-preserving
  force-constant correction (plugs into phonopy; as an additive pair term, into an MLIP).

**Decisive diagnostic (`_diag_kink.py`) — why the naive form failed first:** the Kohn cusp lives in
the **off-diagonal / transverse** fc₂ components, NOT the bond-longitudinal one. A longitudinal-only
(bond-stretch) correction reconstructs kink_K = 0.27 (dead); the **full tensor** is required. And the
cusp is genuinely long-range: exact-Δ truncated to R≥3.5/5.5 Å gives kink 17.2/12.1 of the full 22.6.

**Results (kink_K in THz/q-unit; DFT range 22.63 → 0.22):**
- **Fit:** the 2-param envelope reproduces the entire kink collapse across all 12 smearings,
  MAE ≈ 1.7 (vs the smearing-blind backbone which is flat at 0.22). `B ≈ 1.1` (≈T_el-independent
  amplitude), `κ` monotonically increasing (thermal damping). A mild systematic +2 overshoot at
  intermediate smearings is the one limitation — a single global waveform can't follow the small
  drift of 2k_F with T_el (future: T_el-dependent 2k_F / rank-2 shape term). → `results/smearing_kink/friedel_fit.csv`
- **Few-shot:** fitting smooth laws `B ≈ const` and `κ(T_el) = a·T_el^b` (b ≈ 0.6–0.7) from **just 3
  anchor smearings** predicts the kink at the **8 held-out smearings with MAE 0.87** — the smooth law
  regularises away the per-smearing overshoot. ⇒ `ξ(T_el) = 1/κ ∝ T_el^(−0.6…−0.7)` thermal damping length.
- **Transfer across lattice constant:** the law fit at **a=2.46** (`κ ∝ T_el^0.74`) predicts the
  held-out-smearing kinks at **a=2.44 with MAE 0.27**, measuring only 2 DFT points at the new `a`
  (template + backbone). ⇒ the damping *physics* transfers; only the Fermi-surface waveform D0 is
  re-measured once per material. (Second lattice constant a=2.48 pulling to confirm.) →
  `results/smearing_kink/friedel_transfer.csv`
- **Figures:** `results/smearing_kink/friedel_module.png` (collapse + few-shot + the fc₂ Friedel
  tail), `friedel_transfer.png`.

**What this establishes:** the (E)-channel long-range term is quantitatively a **2-parameter
thermally-damped Friedel oscillation** — amplitude ~T_el-independent, damping length ξ ∝ T_el^(−0.6..−0.7)
— that is both **few-shot** (3 smearings) and **cross-lattice transferable** (2 anchors at a new `a`).
This is the metallic BAMBOO long-range module, validated at the force-constant level.

**Deployment DONE — `scripts/smearing_kink/friedel_calc.py`.** The module is now an **ASE calculator**
(`FriedelMACECalculator`) that wraps *any* base MLIP and adds the Friedel term as an additive
**harmonic** long-range correction `E_Friedel(u) = ½·uᵀ·Δfc(T_el)·u`, `F = −Δfc(T_el)·u` (u = displacement
from the reference supercell; exact in the finite-displacement phonon regime the term targets). T_el is a
calculator argument; B(T_el), κ(T_el) come from the few-shot laws. Validation: driving a
`HarmonicCalculator(DFT dg0.080 backbone)` + Friedel through phonopy finite-displacement recovers the
backbone kink (0.22) and, at each T_el, a kink that **matches the direct `add_template` result to ~0.1**
(round-trip is faithful) and tracks DFT via the few-shot law. It also **composes with a real foundation
MACE** (`mace_mp` medium): the Friedel term adds the correct T_el-dependence on top — though MACE-MP-0 is
itself a poor graphene backbone (bare kink ~101 vs DFT ~14, a known foundation-model graphene failure), so
a graphene-fine-tuned backbone is what production would use.

**Analytic-template attempt (`analytic_template.py`) — NEGATIVE, informative.** Tried to replace the one
measured waveform D0 with a fully analytic Fermi-surface form `W(R)=Σ_{K∈star}cos(K·R+φ)·e^{−R/ξ₀}/R^p`
times an in-plane tensor `(cL n̂n̂ + cT t̂t̂ + cz ẑẑ)`, even sublattice-resolved (independent A-A / A-B
coeffs). It captures only ~19% of D0 (rel-resid 0.90) and the kink goes dead. Reason: graphene's Friedel/
RKKY is genuinely harder than a K-cosine sum — the real-space fc oscillation period is **~2.3 Å**, *shorter*
than the naïve K-period 2π/|K|=3.69 Å, and it is sublattice-dependent (√3×√3). A correct analytic waveform
needs the proper nesting vector + sublattice-resolved RKKY tensor (Bessel radial envelopes), a research
problem of its own. **Conclusion:** keep the **measured 1-smearing template per material** — it is cheap
(one DFT fc₂), and the module already delivers few-shot (3 smearings) + cross-lattice transfer on top of it.

## Data in hand
- 12 graphene fc₂(T_el) at a=2.46: `results/vq_kink6/graphene_sc6_dg*_phonopy.yaml` (also on Box A `/data`).
- Held-out lattice constants (2D-surface DFT): `results/vq_surface/gr_a{2.44,2.48,2.50}_dg*_phonopy.yaml`.
- r_max from-scratch models on the 2060 (`results/gr_rmax_sweep/`).

## Hardware notes

- **2060** `howardwang@100.105.21.7` (`howard-pc`): RTX 2060 SUPER 8 GB, `~/miniconda3/envs/phonon` =
  torch 2.6.0+cu124 / mace 0.3.16 / e3nn 0.4.4, GPU-training-verified. Repo `~/phonon` (run `git pull`;
  it was ~50 commits behind). No tmux → use `setsid`/`nohup`.
- **Box A** `/` had filled to 100 % (QE scratch); redundant installers deleted + `results/{path_p,vq*}`
  migrated to `/data/results/` with symlinks back (`/` now ~45 %). New graphene DFT runs must set
  `--workdir /data/...` + `TMPDIR=/data/tmp` (m1_1b predates the scratch→/data fix).
