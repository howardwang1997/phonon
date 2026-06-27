# TD-Phonon / Kohn-Anomaly Study — Consolidated Summary & Paper Outline

A synthesis of the whole temperature-dependent-phonon / Kohn-anomaly study (branch
`td-phonon-anomaly`). Detailed records + every number live in
[`TD_PHONON_M1_RESULTS.md`](./TD_PHONON_M1_RESULTS.md); the plan/thesis in
[`TD_PHONON_ANOMALY_PLAN.en.md`](./TD_PHONON_ANOMALY_PLAN.en.md). This file is the
executive summary and a paper skeleton.

---

## Thesis (one paragraph)

Foundation machine-learning interatomic potentials (MLIPs) systematically soften — and
sometimes entirely miss — the **phonon Kohn anomalies** of 2D materials. We show that a
cheap, **material-specific force-constant (FC) distillation** (harmonic labels from the
material's own DFT fc₂, *zero new DFT beyond one fc₂*) recovers them — from graphene's
textbook Γ-E₂g / K-A₁′ cusps to a *metallic charge-density-wave (CDW) soft mode* in
monolayer NbSe₂. This enables high-throughput MLIP temperature-dependent dispersions ω(q,T)
for the **lattice-anharmonic (L)** channel (TDEP / SSCHA), while the **electronic (E)**
channel — the Fermi-smearing dependence of the anomaly, which a ground-state-PES MLIP
structurally cannot produce — is supplied by **targeted DFPT-with-smearing**. The same
pipeline cleanly **distinguishes nesting-driven (graphene, q\*=2k_F) from electron–phonon-
coupling-driven (NbSe₂) anomalies**, and the cure is **universal across MLIP backbones**.

---

## 1. The physics & the (L)/(E) decomposition (the intellectual core)

A Kohn anomaly is a near-non-analytic kink/softening of ω(q) where q connects two Fermi-
surface points (q\* ≈ 2k_F). MLIPs learn the ground-state Born–Oppenheimer PES, so:
- **(L) lattice-anharmonic channel** — phonon population, thermal expansion, phonon–phonon
  renormalization. *In the PES → the MLIP sees it.* Tooling: MLIP + TDEP/SSCHA (cheap, many T).
- **(E) electronic channel** — Fermi–Dirac smearing of the 2k_F anomaly with electronic
  temperature. *Not in the ground-state PES → the MLIP cannot make it.* Tooling: DFPT-with-smearing.

Every reported T-trend is labelled by channel; this provenance is the paper's honesty backbone.

---

## 2. Results by material (the three-material arc)

### 2.1 Graphene — validation ✅
- **Softening is model-specific** (Γ-E₂g vs lit ~1600): MACE −22%, SevenNet −14%,
  **MatterSim −2%** (≈DFT). Corrects an earlier "universal softening" reading.
- **Bulk distillation fails the Γ cusp; graphene-specific distillation recovers it** (M1.1b):
  at the fixed DFT geometry bulk-FT closes ~8%, **graphene-FT closes ~101%** (Γ 1570 vs DFT 1568).
- **The K-A₁′ Kohn anomaly is real** (V-Q1 converged-K): the 5×5 DFT under-resolved it (1362,
  kink 0.8, "looks smooth"); the **K-commensurate 6×6 gives K=1292 cm⁻¹ (≈lit 1300) with a sharp
  cusp (kink 14.4)**. Γ converged ~1570. (9×9 needs 45 GB > the V100's 32 GB → 6×6 is the converged point.)
- **(E) channel, two ways:** frozen-phonon (V-Q2) shows the Γ-E₂g cusp *sharpness* drops with
  electronic T; the **rigorous DFPT (#3)** shows the Γ-E₂g *frequency* hardens **1473→1575 cm⁻¹**
  over T_el 790→6300 K (the Kohn softening fills in as the Fermi surface smears) — a trend the
  frozen-phonon / MLIP **structurally misses** (its frequency is electronic-T-insensitive).
- **q\* = 2k_F** (M1.3): the Dirac points at K are the 2k_F connectors → graphene's anomalies are
  **nesting-driven**.

### 2.2 Monolayer MoS₂ — negative control ✅
Gapped (no Fermi surface) → the anomaly locator finds **no cusp** for any backbone (kink ≤ 3) —
it does not false-positive. (Optical modes still soften, but cusp-free.)

### 2.3 Monolayer NbSe₂ — the dramatic CDW case ✅ (the centerpiece)
- **Preview:** foundation *and* general-distilled MLIPs (MACE/SevenNet/MatterSim) all show NbSe₂
  as dynamically stable — the **CDW soft mode is missed** by every backbone.
- **Gate #2 — distillation recovers the CDW soft mode** (V-Q3): DFT fc₂ on a 3×3 supercell
  **captures the soft mode (−2.18 THz)**; distilling it → fine-tune MACE → at the same geometry the
  foundation is stable (−0.2 THz) while the **distilled-FT shows −2.23 THz ≈ DFT**. Material-specific
  DFT distillation **transfers a structural instability** into the MLIP.
- **Robust to geometry** (A): at the DFT-relaxed a=3.474 Å the soft mode survives (−2.05 THz).
- **Backbone-universal** (#4): distilling into **SevenNet** also recovers it (−0.04 → −1.99 THz) —
  not a MACE artifact.
- **Mechanism: NOT Fermi-surface nesting** (B + #2): the Γ-M k_F check and the rigorous **nesting
  function ξ(q)** (FFT autocorrelation of the FS density on a 36×36 grid) **do not peak at q_CDW**
  (ξ(q_CDW)=0.53 vs 1.0 at Γ) → NbSe₂'s CDW is driven by **momentum-dependent e-ph coupling**, the
  textbook Johannes–Mazin result. (Contrast graphene, q\*=2k_F.) So **q\*≈2k_F is material-dependent.**
- **(L)-channel T-evolution, two tiers:**
  - **TDEP (C):** soft mode −0.48 THz (20 K) → 0 (≥200 K) — apparent stabilization ~150 K
    (≈ exp T_CDW 145 K), but TDEP's perturbative footing is weak near an instability.
  - **SSCHA (#1, rigorous):** the free-energy Hessian has **no imaginary modes at any T (20–400 K)** —
    the harmonic CDW soft mode is **fully anharmonically/quantum-stabilized** in the FT MLIP. This
    **refines TDEP-C** (SSCHA > TDEP near an instability, as the plan warned) and exposes that
    **harmonic distillation reproduces the soft fc₂ but not the anharmonic CDW double-well**;
    capturing the CDW *landscape* needs Path-P-style anharmonic distillation.

---

## 3. The four rigor follow-ups (hardening the soft spots)

| # | experiment | result | commit |
|---|---|---|---|
| **#1** | NbSe₂ **SSCHA** (rigorous (L) T-evolution) | free-energy Hessian: no imaginary modes 20–400 K; soft mode anharmonically stabilized; refines TDEP-C | `3e4960b` |
| **#2** | NbSe₂ **χ(q) nesting function** | ξ(q) does not peak at q_CDW → CDW not nesting-driven | `ebcb846` |
| **#3** | graphene **DFPT-with-smearing** | Γ-E₂g hardens 1473→1575 cm⁻¹ with T_el → rigorous (E) channel the MLIP misses | `f716637` |
| **#4** | **cross-model** distillation | SevenNet also recovers the CDW soft mode → cure is backbone-universal | `563d365` |

(#1 was a deep environment-bisection; the working `sscha14` recipe — CC 1.4.1 + python-sscha 1.4.1
+ numpy 1.23 + spglib 1.16 + phonopy 2.18 + mace 0.3.16, all via the Tsinghua mirror — is recorded
in `TD_PHONON_M1_RESULTS.md`. The actual blocker was trivial: CC wants the supercell *diagonal*
`[nx,ny,nz]`, not the 3×3 matrix.)

---

## 4. Methods (brief)

- **DFT engine:** Quantum ESPRESSO (GPU pw.x ~19× on a rented V100; conda CPU ph.x for DFPT),
  ONCV-PBE pseudopotentials, the same `PhononCalculation` pipeline as the MLIPs (fair comparison).
- **FC distillation:** harmonic labels F=−Φ₂u, E=½uᵀΦ₂u from the material's DFT fc₂ → fine-tune the
  foundation MLIP (force-dominated). Anharmonic (Path P): thermal-config DFT-force distillation.
- **(L) channel:** hiPhive-TDEP effective fc₂(T) and **SSCHA** free-energy Hessian (the rigorous
  tool near instabilities), both with MLIP forces — thousands of evals, trivial for an MLIP.
- **(E) channel:** DFPT (ph.x) phonon vs electronic smearing (degauss = k_B·T_el).
- **Anomaly mechanism:** electronic bands + the Fermi-surface nesting function ξ(q).

---

## 5. Limitations & honest caveats

- The **harmonic** FC distillation captures the soft fc₂ but **not the anharmonic CDW double-well**
  (revealed by SSCHA #1) — the MLIP doesn't sustain the CDW under quantum/thermal fluctuations.
- NbSe₂ DFT used a fixed (then DFT-relaxed) geometry; the soft-mode magnitude is geometry-sensitive
  (the *same-geometry* foundation→FT contrast is robust). The CDW is incommensurate in reality.
- The **(E)-channel #3** reports DFPT frequencies; the full e-ph **linewidth** (the canonical
  observable) needs a 2-pass dvscf / EPW — deferred.
- Absolute T_CDW is order-of-magnitude only; "q\*≈2k_F" is material-dependent (holds for graphene,
  fails for NbSe₂).

---

## 6. Flagship future work (needs 8×V100)

A **systematic 2D Kohn-anomaly / CDW family survey** — graphene + h-BN (sp² controls),
MoS₂/MoSe₂/WS₂ (gapped controls), and the CDW metals NbSe₂/NbS₂/TaS₂/TaSe₂/TiSe₂/1T-VSe₂ — running,
per material: (a) DFT fc₂(+fc₃) → distillation, (b) DFPT-with-smearing, (c) **EPW** (α²F, mode
linewidths γ_qν, rigorous mechanism), (d) MLIP+TDEP/SSCHA ω(q,T). EPW on a converged fine k/q grid is
FP64-heavy (~hours/material × ~10 materials × T/smearing) → days on 8×V100. This turns three case
studies into a benchmarked, mechanistic law.

---

## 7. One-line takeaway

> Cheap material-specific FC distillation makes foundation MLIPs reproduce phonon Kohn anomalies — even
> a metallic CDW soft mode, universally across backbones; the lattice-anharmonic temperature evolution
> comes free from MLIP+SSCHA while the electronic channel is supplied by targeted DFPT, and the pipeline
> separates nesting-driven from e-ph-coupling-driven anomalies.

**Commit trail:** V-Q DFT queue `9069ecc` · A/B/C `98a688c` · rigor #2 `ebcb846` · #3 `f716637` ·
#4 `563d365` · #1 unblock `3db43d3` · #1 done `3e4960b`.
