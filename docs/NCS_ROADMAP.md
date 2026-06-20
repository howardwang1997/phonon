# Roadmap → Nature Computational Science
## A closed-loop, GPU-accelerated framework for near-DFT phonons at scale

**One-line thesis.** Foundation MLIPs are fast but phonon-blind; high-throughput phonon DFT is
accurate but the bottleneck. We close the gap by *co-designing the data engine and the model in a
loop*: a GPU-accelerated finite-displacement DFT workflow (Line B) generates curvature-rich
reference data cheaply; force-constant (FC) distillation injects that curvature into a foundation
MLIP (Line A); and active learning lets MLIP uncertainty decide where the expensive DFT is spent —
so the framework self-extends across the inorganic space at a fixed, small DFT budget.

---

## 1. What problem we solve

**Phonons gate a huge slice of materials science** — lattice thermal conductivity (thermal
management, thermoelectrics), thermal expansion, free energies & phase stability, electron–phonon
coupling (superconductivity, transport, mobility), and IR/Raman/INS spectra. Yet **high-throughput
phonon prediction is the standing bottleneck**:

1. **DFT phonons are 10–100× costlier than electronic structure.** Finite-displacement / DFPT needs
   many self-consistent calculations on symmetry-inequivalent displaced supercells, at second-
   derivative precision (≈ N³–N⁴, far tighter convergence than relaxation).
2. **Foundation MLIPs promise ~10³× speed but systematically fail on phonons.** Across the public
   DFPT benchmark they show *frequency softening* (>90 % of materials underestimate ω), *spurious
   imaginary modes* (5–20 % of materials → falsely "unstable"), and *acoustic-sum-rule violation* —
   because they are trained on energies/forces, not curvature (the Hessian). **Low force MAE ≠
   accurate phonons.**

The field is stuck between two half-solutions — *faster DFT* (only ~5–15× per SCF) and *MLIPs*
(fast but inaccurate for curvature). **No route today delivers near-DFT phonon accuracy at MLIP
speed, at scale, for arbitrary inorganic crystals.** That is the gap this work closes.

---

## 2. How the two lines fuse (the methodological core)

The two research lines are complementary halves of **one engine**, coupled in a loop:

```
            ┌─────────────────────  ACTIVE-LEARNING LOOP  ─────────────────────┐
            │                                                                  │
   ┌────────▼─────────┐   curvature-rich      ┌──────────────────┐   uncertainty / errors
   │  LINE B           │   reference FC data   │  LINE A           │   on new chemistries
   │  GPU finite-disp  │ ────────────────────► │  FC-distillation  │ ──────────┐
   │  DFT engine (~50×)│                       │  fine-tuned MLIP  │           │
   │                   │ ◄──────────────────── │  (near-DFT, 10³×) │           │
   └────────▲──────────┘  warm-start density / └──────────────────┘           │
            │             geometry + pre-screen                                │
            └──────────  select which materials/displacements need DFT  ◄──────┘
```

- **Line B → Line A (data):** the GPU engine produces reference phonons (forces + force constants)
  ~50× cheaper, and **FC distillation reuses the *already-computed* force constants as harmonic
  energy/force labels** (`F=−Φu`, `E=½uᵀΦu`) — curvature supervision at **zero extra DFT**.
- **Line A → Line B (acceleration):** the improving MLIP supplies converged-density / relaxed-
  geometry **warm starts** and **pre-screens** candidates, cutting SCF iterations and displaced-
  supercell counts — feeding Line B's own speedup.
- **The loop (the novelty):** MLIP **uncertainty / ASR residual** ranks where the model is still
  wrong; Line B computes DFT *only there*; new FC data re-distills the model. DFT is never spent on
  chemistries the model already nails. This is what lets one small DFT budget cover a huge space.

This closed-loop co-design — not either line alone — is the contribution that lifts the work from a
method paper (PRB-class) to a **framework + tool + dataset** of the scope NCS publishes.

---

## 3. The story we tell (narrative arc)

> *"Phonons are the missing piece of the foundation-model materials revolution — and the fix is to
> co-design the data engine with the model."*

- **Act 1 — Quantify the blind spot.** A large, stratified benchmark shows universal MLIPs are *not
  ready* for phonons: softening, imaginary modes, ASR violation, with the failure correlated to
  curvature, not energy error.
- **Act 2 — Diagnose.** It is a *Hessian-supervision gap*, not a capacity gap: force MAE and phonon
  MAE decouple; conservative (energy-derivative) models beat direct-force ones; the error lives in
  the second derivative.
- **Act 3 — Fix it cheaply.** FC distillation turns existing DFPT force constants into curvature
  labels. *Preliminary result (this work):* in-domain phonon MAE drops to **~0.10 THz** with
  imaginary modes eliminated — at zero new DFT. Anti-forgetting (replay / LoRA) preserves the base
  model's universality.
- **Act 4 — Make it generalize and scale.** Naively adding *more configs per material* fixes only
  in-domain accuracy; it does **not** transfer. *Preliminary result (this work):* held-out transfer
  MAE is flat (~1.80 THz) up to 16 training materials, then falls to **1.40 (32) → 1.34 (64)** — so
  **breadth of chemistry, not depth of sampling, drives generalization, with a threshold near
  16–32 materials.** This is exactly why a *targeted* DFT data engine + active learning is needed:
  to buy chemical breadth at minimal DFT cost. Line B + the loop deliver it.
- **Act 5 — Pay it off downstream.** Near-DFT lattice thermal conductivity (phono3py), thermo-
  electric power factors, and dynamical-stability screening at ~10³× speed — validated against DFT
  and experiment — with an open dataset and tool released to the community.

---

## 4. What experiments to run (the program)

### Phase I — Establish accuracy science (Line A) · *mostly public data*
- **E1 Benchmark atlas.** ≥6 foundation MLIPs (MACE-MP/MPA/OMAT, MatterSim, SevenNet, ORB-
  conservative, eSEN, CHGNet, M3GNet baseline) × a stratified **~1,500-material** subset of MDR/
  Petretto DFPT. Full metric suite: freq MAE/RMSE, ω_max error, **imaginary-mode count**, **ASR
  residual**, DOS error, thermodynamics (C_v, S, F). → the failure-mode atlas. *(2060 box is already
  computing baselines across the 85-material pilot pool.)*
- **E2 Diagnosis.** Force-MAE vs phonon-MAE decoupling; conservative vs direct-force; displacement-
  amplitude sensitivity; ASR before/after symmetrization.
- **E3 FC-distillation design** *(largely done — pilot on 8×H20):* anti-forgetting comparison
  (single-head / replay / LoRA, with seed error bars), **data-efficiency: depth vs breadth**,
  replay×breadth interaction, foundation-size (small vs medium). *Pilot outcome:* in-domain ~0.10
  THz; breadth threshold ~16–32 materials; light replay/LoRA both viable.
- **E4 Generalization stress test.** Held-out materials, **unseen elements**, **unseen structure
  prototypes**; full-band (not just mesh) comparison; NAC/LO-TO for polar crystals with Born charges.

### Phase II — Accelerate the data engine (Line B) · *self-generated, small*
- **E5 GPU finite-displacement workflow**, honestly timed end-to-end against a CPU baseline:
  (i) GPU SCF (QE-GPU/OpenACC, ~5–15×) × (ii) cross-displacement density reuse/extrapolation
  (~2–4×) × (iii) symmetry + non-diagonal supercells (~2–10×) → **target ~50× workflow-level** on a
  benchmark set; report each factor separately and label "workflow-level vs single-SCF."
- **E6 (stretch) ML-accelerated SCF.** Predict converged density / one-shot Harris forces as SCF
  warm start; *evaluate whether the force accuracy suffices for phonons* (the known open risk).

### Phase III — Fuse: the closed loop (the novelty) · *self-generated, budgeted*
- **E7 Active-learning loop.** Acquisition function selects materials for Line B DFT → FC-distill →
  repeat. **Headline claim:** reach a target phonon accuracy with **N× less DFT** than random.
  *Preliminary 3-way result (this work, MDR-as-oracle):* at matched budget, **chemical-coverage
  acquisition beats random** (~0.18 THz lower held-out MAE at N=32), while **naive model-uncertainty
  acquisition is the *worst*** — it chases pathological/soft outliers (P allotrope, 30+ imaginary
  modes) that are informative-to-the-model but unrepresentative of the target distribution. → **the
  data engine's acquisition function must be coverage-driven (or stability-filtered), not naive
  uncertainty** — a concrete, counterintuitive design lesson (uncertainty sampling is the textbook
  default). Next: coverage×stability-filtered hybrid; ensemble-disagreement uncertainty.
- **E8 Scale-out demonstration.** Run the loop to produce a **near-DFT phonon dataset for 10³–10⁴
  materials** at a fixed, small DFT budget; full cost accounting (public reuse vs new DFT).
- **E9 Downstream payoff.** Lattice **κ** (phono3py 3rd-order), thermoelectric power factor / zT,
  and a dynamical-stability classifier — fine-tuned MLIP vs baseline vs DFT vs experiment.
- **E10 Reverse coupling.** Quantify MLIP-warm-started DFT speedup folded back into Line B.

### Validation throughout
Si gold standard vs neutron INS + DFPT; convergence (supercell, displacement amplitude, k/q mesh,
random seeds); seed error bars on every headline number; one-command reproducibility; ablations.

---

## 5. Compute & data requirements

### 5.1 Data — what is public vs self-generated
| Need | Source | Public? | Use |
|---|---|---|---|
| Harmonic DFPT phonons (10,034 materials) | **MDR/PhononDB** (Togo/NIMS) | ✅ free | Benchmark (E1) + **FC-distillation labels** (E3) — zero new DFT |
| Harmonic DFPT phonons (~1,521) | **Petretto 2018** | ✅ free | Cross-check / second reference (E1,E4) |
| Lattice κ / 3rd-order FC (~100s) | **Togo phono3py DB** | ✅ free | Downstream κ public reference (E9) |
| Pre-training trajectories | **MPtrj** | ✅ free | Replay anti-forgetting (E3) |
| GPU-DFT **timing** baselines | self (QE-GPU + CPU) | ❌ | E5 (engine speedup) |
| **New-chemistry** phonons (not in public DBs) | self (Line B engine) | ❌ | E7/E8 active-learning expansion |
| Anharmonic κ validation (selected) | self (phono3py) | ❌ | E9 (where no public κ exists) |

**Key leverage:** the entire accuracy science (E1–E4) and the distillation labels run on **public
data with zero new DFT** — self-generated DFT is needed *only* for the engine timing (E5) and the
active-learning expansion (E7–E9), which is exactly the budget the loop minimizes.

### 5.2 Compute — estimated GPU-hours
| Item | Workload | Est. GPU-hours | Hardware |
|---|---|---|---|
| E1 benchmark + E3/E4 fine-tuning | MLIP inference + training | ~tens–low hundreds | **8×H20** (training fleet) + 2060 (inference) |
| E5 GPU-DFT engine timing | QE-GPU + CPU baselines, ~10–30 materials | ~300–800 | A100/V100 (FP64) — rent on demand |
| E7/E8 active-learning DFT | harmonic phonons, new chemistries: ~3 GPU-hr/material (simple) → ~200–800 (complex); target ~10–30 materials/round × few rounds | ~1,000–3,000 | A100/A800 80 GB or V100 32 GB |
| E9 anharmonic κ validation | phono3py 3rd-order, ~10–15 materials (10–50× harmonic) | ~500–1,500 (lean on public Togo set) | A100/A800 |
| **Total self-generated DFT** | | **~2,000–5,000 GPU-hr** | mix: fleet + on-demand cloud |

**Cost-effective hardware split (established):** MLIP work (≈90 % of the science) on a single
mid-range GPU is sufficient (fine-tuning needs only ~7 GiB) — the 8×H20 fleet makes the ablations
*fast*, not *possible*. DFT (FP64) is a one-time burst → **rent A100/V100 by the hour** rather than
own; avoid consumer cards (crippled FP64) for DFT. Multi-GPU is embarrassingly parallel over
displacements (wall-clock = GPU-hours / N_GPU).

---

## 6. Deliverables & impact
1. A **released fine-tuned foundation MLIP** with validated near-DFT phonon accuracy (softening &
   imaginary modes fixed).
2. An **open near-DFT phonon dataset** for 10³–10⁴ materials produced by the loop.
3. An **open-source closed-loop framework**: GPU-DFT data engine + FC-distillation + active learning.
4. **High-throughput demonstrations**: κ, zT, dynamical-stability screening at ~10³× speed.

---

## 7. Differentiation (why NCS, not PRB)
- vs **MLIP-phonon benchmarks** (e.g. Loew 2025): they *measure* the failure; we *fix* it and *scale*.
- vs **direct MLIP fine-tuning**: we add **curvature (FC/Hessian) supervision** and quantify the
  breadth-not-depth generalization law.
- vs **GPU-DFT acceleration papers**: they speed a single SCF; we embed DFT in a **loop with an MLIP**
  so the speedup compounds and the DFT budget is *targeted*.
- **The novelty = the closed-loop co-design of data engine and model**, delivered as a general
  framework + tool + dataset with cross-property impact — NCS's remit, not a single-method result.

---

## 7b. Status & gap to NCS (as of 2026-06)

**Done — publishable PRB/npj-grade backbone:**
- **Line A laws** (`CAMPAIGN_FINDINGS.md`): breadth > depth for transfer (knee ~20–24 materials, floor
  ~1.3 THz); depth overfits; anti-forgetting with error bars; **E7 acquisition — coverage > random >
  naive-uncertainty** (counterintuitive design lesson); transfer floor dominated by BN.
- **Line B engine** (`LINE_B_FINDINGS.md`): QE finite-displacement validated (Si ~1–5%); honest
  speedup decomposition (symmetry 48–384× *standard*; density+wfc reuse ~1.5× wall, iteration savings
  scale with SCF difficulty Si 1.0×→MgO 1.78×; GPU-SCF deferred); first **self-DFT phonon dataset**
  (11 materials, mean 1.4% vs MDR).

**Two hard bones remaining for NCS:**
1. **Real GPU acceleration** — QE-GPU on rented A100/V100 for the GPU-SCF factor (~5–15×). H20 FP64 is
   crippled, so "GPU-accelerated" is literature-only until this runs. *~3–5 d; needs rented GPU.*
2. **Downstream impact** — lattice thermal conductivity κ (phono3py 3rd-order): fine-tuned MLIP vs DFT
   vs experiment. NCS needs a *useful-property* payoff, not just phonon accuracy. *~hundreds–thousands
   GPU-hr.*

**Strengthening (needed but not blocking the thesis):**
- Loop closure (self-DFT → distill) — **immediate, ~40 min**.
- Real active-learning loop with **Line-B-generated** DFT on *new* chemistry (E7→E8) — ~1–2 wk.
- Cross-model generality (MatterSim/SevenNet/ORB) — ~3–5 d (new envs + non-MACE fine-tune path).

**Resource decisions gating the above (need user):**
- **Broad SSSP pseudopotentials** — required for diverse self-DFT + the real loop; current 4-element
  (Si/Al/Mg/O) coverage is the binding constraint on Line B chemistry. (materialscloud / GBRV / aiida
  URLs all failed from the China box → download on an open-internet machine.)
- **A100/V100 rental** — required for both hard bones (GPU-SCF and κ).

**Honest framing:** the *method backbone is strong now*; reaching NCS is **~2–4 weeks of real compute**
(rented GPU + thousands of GPU-hr) gated on the two resource decisions above — not on more ideas.

## 8. Milestones (indicative, ~6–9 months)
- **M1 (1–2 mo):** E1 benchmark atlas + E2 diagnosis (public data). Submit-quality failure-mode figure.
- **M2 (2–3 mo):** E3/E4 finalized with breadth law + generalization (pilot already done).
- **M3 (3–5 mo):** E5 GPU-DFT engine ~50× demonstrated; E6 scoped.
- **M4 (4–7 mo):** E7 active-learning loop beats random selection; E8 scale-out dataset.
- **M5 (6–9 mo):** E9 downstream (κ/zT/stability) + E10; assemble, write, release tool+dataset.

> Honesty clauses to keep in the paper (credibility): pure GPU-DFT single-SCF is only ~5–15×; the
> ~50× is workflow-level; the ~10³× is the MLIP proxy. Every speedup is reported with its scope and
> every accuracy number with seed error bars.
