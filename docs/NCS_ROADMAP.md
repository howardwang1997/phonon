# Roadmap → Nature Computational Science
## A foundation-model framework that repairs the phonon blind spot — and turns it into a discovery engine for electronically-driven lattice instabilities

> **⚠ Strategy update (2026-07-01) — now ONE complete flagship paper (not two). Read §0.** The engine
> (method) and the (E)/(L) CDW-origin decomposition (discovery) merge into a **single high-level paper** —
> **Part I (method) + Part II (discovery)** — no npj保底 / NCS冲刺 split. *Scope is unchanged; only the
> packaging merges.* Target: a single flagship submission (NCS / Nature Materials class).
>
> **⚠ Strategy update (2026-06-30) — reframed; read §0 first.** From *"near-DFT phonons at scale
> (a tool)"* → *"a tool that enables a discovery."* A pure tool paper is a strong **npj** but a
> low-probability **NCS** (the components — MLIP phonons, distillation, active learning — are each
> well-trodden); the fix is to **lead with the science the engine uniquely enables**: high-throughput
> **decomposition of lattice instabilities into electronic vs lattice-anharmonic origins** (CDW origin
> in 2D materials). The accuracy engine (§1–§8) becomes the *method*; the instability-origin study
> becomes the *headline*. **Ballistic transport is explicitly out of scope.**

**One-line thesis.** Foundation MLIPs are fast but **blind precisely to the physics that defines
quantum materials** — the *electronically-driven* phonon anomalies (Kohn anomalies, charge-density-
wave soft modes, soft-mode instabilities). We (i) repair this blind spot by *co-designing a data
engine and the model in a loop* — a GPU finite-displacement DFT workflow (Line B) feeds force-
constant (FC) distillation into a foundation MLIP (Line A), with active learning targeting the DFT;
and (ii) **because the repaired framework is cheap *and* accurate on these hardest cases, it makes
origin-decomposition of lattice instabilities high-throughput for the first time** — which we use to
map and adjudicate CDW instabilities across the 2D transition-metal-dichalcogenide (TMD) family.

---

## 0. Strategy update (2026-06-30) — the merged story (READ FIRST)

**Why this update.** §1–§8 describe the *accuracy engine* (Line A distillation + Line B GPU-DFT +
active-learning loop). That engine is real and largely built — but **as a standalone "tool" paper it
is a strong npj and a low-probability NCS** (~10–20%): MLIP phonons, FC distillation, and active
learning are each well-trodden by well-resourced groups, so the framework risks reading as
incremental, and NCS increasingly wants a *scientific payoff*, not a faster tool.

**The fix — invert the framing.** Lead with the **science the engine uniquely enables**, not the engine:

> **Headline capability.** Foundation MLIPs systematically fail on the *electronically-driven* phonon
> anomalies (Kohn anomalies, CDW soft modes) that underlie quantum materials. The repaired framework
> is the first that is **both cheap and accurate enough on these hardest cases to make high-throughput
> *origin-decomposition* of lattice instabilities feasible** — separating each instability into its
> **electronic (Fermi-surface / EPC)** vs **lattice-anharmonic** channel — which we use to **adjudicate
> the long-debated origin of charge-density waves across the 2D-TMD family.**

**Why merged > either half alone:**
- the engine's novelty is *justified by the application it enables* (not "yet another phonon framework");
- the discovery is made *tractable by the engine* — the cheap (L)-channel (distilled MLIP + SSCHA)
  screens a whole family that full DFPT/EPW could never afford; (E)/EPW deep-dives only the flagships.

**Already done toward the discovery half (this work — the "B-line" / anomaly thread, `TD_PHONON_*`):**
- **(E)/(L) decomposition defined + validated on NbSe₂.** (L) = MLIP+SSCHA: the soft mode is
  *quantum-stabilized* (bare DFT overestimates the instability; **Path-P** anharmonic distillation
  reproduces the true landscape, 4.6× better forces than harmonic). (E) = DFPT-vs-smearing + EPW:
  χ(q) only weakly peaked (#2) **and** EPW **γ_qν broad at q_CDW** (E6) → **EPC-driven, not nesting**
  (Johannes–Mazin) — the **first family member adjudicated.**
- **graphene** as the (E)-channel reference (Γ-E₂g + K-A₁′ vs electronic T; undoped EPW λ≈0 at the
  Dirac point → doped EPW λ turns on — coupling is a pure Fermi-surface effect).
- **Methodological maturity to sell as rigor, not weakness:** integrated λ is ill-defined at a soft
  mode (it *diverges* — that divergence **is** the instability signal) → report q-resolved γ_qν/λ_qν;
  2D acoustic-ASR subtleties (q2r `zasr='crystal'`); bare-DFT overestimation cured by SSCHA.

**What the discovery half still needs (the NCS gate):**
1. **Family breadth** — extend (E)/(L) from NbSe₂ to NbS₂, TaS₂, TaSe₂, TiSe₂, VSe₂, … (cheap (L)
   screen over all; (E)/EPW on 3–5) → the **origin-classification map** (each material in the (E)–(L)
   plane).
2. **A discovery** — a *non-trivial* outcome: a material re-classified (assumed-nesting → EPC or
   vice-versa), a predicted-but-unmeasured instability, or a clean (E)/(L)→T_CDW trend validated vs
   experiment. **This is make-or-break; NCS probability ~25–40% even if it lands.**

**Explicitly OUT of scope:** **ballistic transport (NEGF/Landauer)** — different method family, no
shared backbone; bolting it on reads as scope creep and *lowers* acceptance.

**Single-flagship discipline (updated 2026-07-01) — ONE complete paper, tracked to stop drift:**
The plan is now **one complete, high-level paper** (no longer two), whose arc is *method → the discovery
it enables*:
- **Part I — the method (~2–4 wk to submittable):** graphene+NbSe₂ = the engine (Line A/B backbone) +
  (E)/(L) decomposition on the two flagships. Lockable from current results + the ASR fix.
- **Part II — the discovery (+2–4 mo):** the TMD-family origin-map + the non-trivial CDW-origin finding
  + open tool/dataset. **Integral, not optional** — the paper is not complete until it lands.
- **Target:** a single flagship submission (Nature Computational Science / Nature Materials class) — the
  complete method + discovery.
- **Honest risk:** one flagship = higher variance, longer runway, **no npj保底 fallback**; needs both
  halves airtight (the method half stays independently strong as an implicit floor).
- **Anti-drift:** **scope is unchanged from the two-paper plan — only the packaging merges.** Every
  experiment below (E1–E10 engine + the family origin-map/EPW/SSCHA/κ) is retained.

**Execution rule (this update's real purpose):** from here, work is tracked against *this* merged
plan; revisit it before every compute campaign and log advance-vs-drift. The prior drift — a session
spent deep on the B-line without reconciling against this roadmap — is exactly what §0 fixes.

---

## 0b. Execution plan — data · compute · experiments (merged, 3-machine)

> Companion to §0. The full **engine** experiment list (E1–E10), public-data table, and engine
> GPU-hour budget stay in §4 / §5 (unchanged). **This section is the actionable, resource-mapped plan
> for the merged single-flagship-paper push, given the three machines actually in hand** and what is gated on rental.

**Machines in hand**

| Machine | Capability | Role |
|---|---|---|
| **Box A / Box B** — 2× V100, 16-core, conda **CPU-QE** (+ GPU pw.x) | DFT / DFPT / EPW (CPU; ~8.5 h per NbSe₂-class material) + MLIP | DFT/EPW **workhorses** |
| **2060** — 8 GB | MLIP fine-tune (~7 GB) + SSCHA — **no QE/DFT** | distillation + (L)-channel |

DFT/DFPT/EPW is the bottleneck (CPU, only the 2 V100s); MLIP / distillation / SSCHA is cheap (2060).

### 1 · Data

**Public (zero new DFT):**
- MDR/PhononDB (10,034) + Petretto (1,521) harmonic DFPT → benchmark + FC-distillation labels (engine).
- Togo phono3py κ DB → downstream κ reference. MPtrj → replay anti-forgetting.
- **Experimental T_CDW / INS for the TMD family (literature)** → Part-II (discovery) validation.

**Self-generated (the 2 V100s):**
- graphene + NbSe₂ DFT fc₂ / DFPT / dvscf / EPW — **done** (Part-I flagships).
- **TMD family** (NbS₂, TaS₂, TaSe₂, TiSe₂, VSe₂, …) fc₂ + DFPT + dvscf — **to generate** (Part II).

**Data ENABLERS (gating, low-cost, do now — not rental):**
- **Broad pseudopotentials**: have C/Nb/Se(+few); the TMD family needs **Ta, Ti, V, S**. **Download** on
  an open-internet machine. *Without these the family work cannot start.*
- **q2r.x / matdyn.x**: missing from the minimal conda QE build; needed for the **crystal-ASR** fix
  (clean absolute λ, 2D ZA mode). **conda-install full QE** on Box A/B (small).

### 2 · Compute requirements — GPU-hour estimates by hardware class

**Measured anchors (this project):** NbSe₂ DFPT ≈ 6 h, EPW ≈ 2.5 h (→ one NbSe₂-class (E)/EPW material
≈ **8.5 box-h**, CPU); TMD fc₂ (finite-displacement) ≈ 1–3 h; SSCHA ≈ 15–30 min/(material·T); MLIP
fine-tune ≈ 30–60 min/run; MLIP phonon inference ≈ ~1 min/material.
*Unit note:* H20 numbers are true **GPU-h** (MLIP/FP32); V100 DFT runs on **CPU (16-core box)** → quoted
as **box-h**, not GPU-h.

**(a) 8×H20 — the MLIP / (L)-channel / benchmark half** *(FP32; mostly zero new DFT):*
| Workload | single-GPU GPU-h |
|---|---|
| Benchmark atlas (E1, ~1,500 mat × 6 models) | ~200–400 |
| Diagnosis (E2) | ~50–100 |
| FC-distillation + ablations (E3/E4) | ~100–200 |
| Active-learning loop (E7→E8, MLIP side) | ~150–300 |
| **(L)-channel SSCHA family screen** (Part II) | ~50–100 |
| 10³–10⁴ MLIP phonon dataset (E8) | ~200–400 |
| Cross-model generality (MatterSim/SevenNet/ORB) | ~50–100 |
| κ — MLIP-force part (E9) | ~100–200 |
| **Total** | **~900–1,800 GPU-h** → on 8 cards **~5–10 parallel-days** (w/ iteration ~2–3 wk) |

**(b) 2×V100 + 2060 — the DFT/EPW half, NOW** *(no rental; FP64 = box-h):*
| Task | box-h |
|---|---|
| λ(T_el) campaign (running; mostly spent) | ~35 |
| ASR re-pass + convergence (graphene/NbSe₂) | ~20–40 |
| TMD-family fc₂ labels (~12 mat) | ~15–45 |
| (E)/EPW deep-dive on 3–5 flagship TMDs | ~40–90 |
| **NOW subtotal (2×V100)** | **~110–210 box-h → ~1–2 wk on 2 boxes** |

*(2060 is redundant once an H20 fleet exists — anything it does, H20 does faster.)*

**(c) Rented V100/A100 — the FP64 scale half** *(Phase 3; only after the discovery gate):*
| Task | GPU-h |
|---|---|
| GPU-QE ~50× engine demo (E5; needs A100/V100 FP64) | ~300–800 |
| family-wide **converged** EPW (dense grids) | ~300–500 |
| κ-at-scale DFT reference (3rd-order, E9) | ~500–1,500 |
| 10³–10⁴ dataset DFT labels (active-learning chemistries) | ~1,000–3,000 |
| **Rental subtotal (FP64)** | **~2,100–5,800 GPU-h** |

**Summary.** ~900–1,800 GPU-h on **8×H20** (mostly zero new DFT) **+** ~110–210 **box-h** on the **2 V100s
now** carries the **single paper to a core-complete draft (Part I + the Part II origin-map)**. Only the **~2,100–5,800 FP64 GPU-h**
of scale (engine demo + converged family EPW + κ-at-scale + 10³–10⁴ DFT labels) needs **rental — and only
if the Phase-2 discovery gate passes**. *Ranges are wide; the two biggest unknowns are MLIP-inference
speed (material size) and converged-EPW grid cost (could be 2–3× higher). A 3–5-material TMD pilot on the
current machines would re-calibrate every number above.*

### 3 · Experiment plan (ordered, resource-tagged)

**Phase 0 — enablers** *(now · ~1 day · no rental)*
- **E-0a** conda-install full QE (q2r.x/matdyn.x) on Box A/B → crystal-ASR fix.
- **E-0b** download Ta/Ti/V/S pseudopotentials (open-internet machine).

**Phase 1 — Part I (method) lock** *(now · ~2–4 wk · no rental)*
- **E-1a** finish λ(T_el) campaign (running) + ASR re-pass on graphene/NbSe₂ → clean absolute λ.
- **E-1b** convergence (k/q grid, smearing, supercell) on the flagships.
- **E-1c** assemble graphene+NbSe₂ (engine + (E)/(L)) → **Part-I draft**.

**Phase 2 — Part II (discovery) attempt** *(now · ~1–2 wk · no rental · THE COMPLETENESS BAR)*
- **E-2a** TMD-family triage: foundation-MLIP SSCHA on 2060 (no DFT) → rank likely-unstable.
- **E-2b** fc₂ (finite-displacement) on candidates @ Box A/B [needs E-0b pseudos] → distill + **(L)-channel SSCHA** @ 2060.
- **E-2c** **(E)-channel** — DFPT-vs-smearing + EPW γ_qν — on 3–5 flagship TMDs @ Box A/B.
- **E-2d** build the **(E)–(L) origin-classification map**; hunt the non-trivial discovery (re-classification / predicted instability / T_CDW trend vs experiment).
- **★ the discovery** (integral to the paper): origin-map + ≥1 non-trivial result → then rent for the Phase-3 scale-up. If it underdelivers, strengthen/reposition — there is no npj fallback.

**Phase 3 — scale for NCS** *(rental · +2–4 mo · only if the Phase-2 gate passes)*
- **E-3a** family-wide converged EPW + κ-at-scale + benchmark atlas + 10³–10⁴ dataset + GPU-DFT engine demo.
- **E-3b** open-source the instability-origin pipeline + dataset.
- **E-3c** assemble the complete flagship paper (Part I method + Part II discovery).

**Golden rule:** *don't rent until the Phase-2 discovery gate shows signal.* The 3 machines can carry
the single paper to a core-complete draft (Part I + the Part II origin-map); rental buys only **scale**, after the
science is de-risked.

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
- **B-line / anomaly thread** (`TD_PHONON_M1_RESULTS.md`) — **the seed of Part II, the discovery (§0):** (E)/(L)
  decomposition validated on graphene + NbSe₂ — SSCHA quantum-stabilization, Path-P anharmonic
  distillation, EPW **γ_qν broad at q_CDW** (EPC-driven, not nesting), (E)-channel vs electronic T,
  doped-graphene λ turn-on. NbSe₂ = the **first TMD-family member adjudicated**.

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

**Honest framing (updated 2026-07-01 — see §0):** the *method backbone is strong now* — the accuracy
engine + (E)/(L) decomposition on graphene+NbSe₂ is **~2–4 weeks from a submittable Part I (method)**.
But the **complete flagship paper is *not* 2–4 weeks away** — the prior estimate conflated a "ready
method backbone" with the whole paper. The paper also needs the §0 **Part II discovery**: the TMD-family
origin-map + a *non-trivial discovery* + open tool/dataset (**+2–4 months; the discovery is ~25–40%
even if it lands**). The two resource decisions below gate the *engine*; the **discovery is the real
completeness bar for the single paper** (higher variance, no npj fallback).

## 8. Milestones (indicative, ~6–9 months)
- **M1 (1–2 mo):** E1 benchmark atlas + E2 diagnosis (public data). Submit-quality failure-mode figure.
- **M2 (2–3 mo):** E3/E4 finalized with breadth law + generalization (pilot already done).
- **M3 (3–5 mo):** E5 GPU-DFT engine ~50× demonstrated; E6 scoped.
- **M4 (4–7 mo):** E7 active-learning loop beats random selection; E8 scale-out dataset.
- **M5 (6–9 mo) — Part I complete:** E9 downstream (κ/zT/stability) + E10; the engine + (E)/(L) on the
  two flagships assembled as **Part I (the method)** of the single paper.
- **M6 — Part II, the discovery (the §0 headline; integral to the same paper):** TMD-family (E)/(L)
  **origin-map** (cheap (L)-channel screen of NbS₂/TaS₂/TaSe₂/TiSe₂/VSe₂ + (E)/EPW deep-dive on 3–5) →
  the origin-classification figure → **a non-trivial discovery** → open instability-origin pipeline +
  dataset. *The discovery is the completeness bar; ~+2–4 mo after Part I is drafted; single flagship =
  higher variance, ~25–40% if the discovery lands, no npj fallback.*

> Honesty clauses to keep in the paper (credibility): pure GPU-DFT single-SCF is only ~5–15×; the
> ~50× is workflow-level; the ~10³× is the MLIP proxy. Every speedup is reported with its scope and
> every accuracy number with seed error bars.
