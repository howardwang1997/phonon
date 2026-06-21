# Line B Findings — GPU/CPU DFT phonon engine (honest speedup decomposition)

Quantum ESPRESSO 7.5 (conda-forge, CPU) on the 192-core H20 box, driven through the *same*
`PhononCalculation` pipeline as the MLIPs. Goal: a finite-displacement DFT phonon engine that (a)
validates as ground truth, (b) produces reference data for Line A's FC distillation, and (c) honestly
characterizes the workflow-level acceleration.

## Engine validation (the milestone)
Si, QE PBE, 2×2×2 supercell (64 atoms), ecutwfc 40 Ry:
- **ω_max = 15.358 THz** vs MDR DFPT / neutron experiment ≈ 15.5 → **~1% error** ✓
- n_imaginary = 0 (stable), ASR residual 1.4e-7 ✓
The DFT engine reproduces the gold-standard Si phonon. It is the "DFT ground truth" half of the
framework, computed identically to the MLIP phonons (fair speed + accuracy comparison).

## Speedup decomposition (honest, measured)
| Factor | Si value | Novelty | Notes |
|---|---|---|---|
| **Symmetry reduction of displacements** | **48–384×** | standard (phonopy) | naive 6N force-evals → symmetry-inequivalent set. Si 384× (1 disp), MgO/LiF/CaO 192×, BN 96×, SiC 48×. *Dominant, but free to any phonopy user.* |
| **Charge-density + wavefunction reuse** | **~1.5× wall** (all); iter savings scale with difficulty | **engine** | displaced SCF restarted from the equilibrium density+wfc (`startingpot/startingwfc=file`, `nosym` so wfc transfers across symmetry breaking). Reuse starts **300× closer** in SCF accuracy. Measured: **Si 8→8 iters (1.0×), Al 7→6 (1.17×), MgO 16→9 (1.78×)** — the harder the system (more from-scratch iterations), the bigger the saving; wall ~1.46–1.48× consistently. |
| **GPU SCF (QE-GPU/OpenACC)** | ~5–15× (lit.) | engine | **deferred** — H20 FP64 is crippled (inference card), so GPU-QE won't beat 192 CPU cores here; needs A100/V100. Rent on demand. |
| **Non-diagonal supercells** | ~2–8× | engine | fewer atoms for the same q-resolution — future. |

## Honest conclusion (important for framing)
- The headline "~50× workflow-level" is **dominated by symmetry reduction (48–384×), which is
  standard practice**, not a GPU/engine novelty.
- The engine's **genuine** contributions beyond standard phonopy — density/wfc reuse (~1.5× on Si)
  + non-diagonal supercells + GPU SCF (deferred) — are individually modest on CPU and stack to
  roughly **~3× (CPU now) → ~25× (with GPU SCF on A100)**.
- **This reinforces the project's central thesis:** pure workflow-DFT acceleration is bounded
  (~tens×, dominated by free symmetry); the **MLIP path (50–1000×) is the real high-throughput win**.
  Line B's value is therefore (i) producing **reference data efficiently** to feed Line A's FC
  distillation (the closed loop), and (ii) a validated GPU-ready DFT engine — *not* an outsized
  standalone speedup. We report every factor with its scope and label "standard vs engine."

## What Line B delivers to the NCS framework
1. Validated DFT ground-truth phonons (Si ~1%); extensible to any material with a pseudopotential.
2. The data engine for E7/E8 active learning — and Result-4 already told us *which* materials to
   query (chemical coverage), so the engine's DFT budget is spent optimally.
3. Honest speedup accounting that strengthens (not inflates) the paper's credibility.

## Self-DFT phonon dataset (Line B output, 2026-06)
First self-generated DFT phonon set: **11 materials** (Si phases, MgO₂, multiple SiO₂ polymorphs),
each via QE finite-displacement (adaptive supercell ≥9 Å/axis, ecutwfc 50 Ry), force constants saved
to `data/dft_ref/selfdft-*_phonopy_params.yaml`, validated vs MDR DFPT:
- **mean |ω_max error| ≈ 1.4%** (Si ~5%, SiO₂ polymorphs <1%, MgO₂ +0.1%).
- Chemistry is narrow (Si/O/Mg only) — limited by the 4 reliable pseudopotentials (Si/Al/Mg/O);
  **broad SSSP coverage is the gating blocker** to diversify.
- Sequential CPU generation (~2 min–1.5 h/material by cell size); large SiO₂ polymorphs dominate cost.
This is the data-engine output that feeds Line A FC distillation (loop closure) and a stand-alone
near-DFT phonon set.

## Remaining Line B experiments (priority + blocker)
| experiment | gives | est. | blocker |
|---|---|---|---|
| **GPU-SCF factor** | the GPU acceleration the project is premised on (~5–15×) | ~3–5 d | **needs A100/V100 — H20 FP64 crippled** |
| **non-diagonal supercells** | engine factor ~2–8× (fewer atoms per q-resolution) | ~1–2 d | none |
| **end-to-end composed timing** | honest workflow-level total (symmetry×reuse×non-diag×GPU) | ~1 d | depends on above |
| **broaden chemistry** | diverse self-DFT set (oxides/metals/nitrides/chalcogenides) | ~1–3 d CPU | **broad SSSP pseudopotentials** |
| DFPT (ph.x) cross-check (optional) | validate finite-displacement vs DFPT | ~1 d | none |

## Downstream impact: lattice thermal conductivity κ (anharmonic, no rental)
MLIP + phono3py (3rd-order FC, RTA) → κ; **no DFT** (MLIP forces on GPU, ~6 s/material). Si, sc 3³,
mesh 21³:

κ(300 K), W/m·K, sc 3³ mesh 21³, ~6–16 s/material on one GPU:

| material | baseline (MACE small) | fine-tuned (FC-distilled) | DFT/exp | FT effect |
|---|---|---|---|---|
| **Si** (non-polar covalent) | 45 | **113** | ~140 | ✓ big fix (3× → ~0.85×) |
| **GaAs** (covalent) | 13.7 | **28.5** | ~45 | ✓ improved (still low) |
| **MgO** (polar ionic) | 60.7 | 152 | ~55–60 | ✗ over-stiffened |

**Honest, nuanced finding:** FC distillation **recovers the κ of softening-prone covalent
semiconductors — exactly the failure-mode materials** (Si 45→113, GaAs 13.7→28.5; baselines were 3×
low because MACE softens their phonons → low group velocities → low κ). But it is **not a uniform
win**: the already-good ionic MgO is **over-stiffened (61→152)**, and polar MgO κ is anyway
unreliable here because the MLIP κ omits **NAC/Born charges** (LO-TO splitting). → curvature
supervision propagates to the anharmonic downstream property where it matters most (softened
covalent crystals), with two honest caveats (over-correction; NAC for polar). Targets the **3rd-order
FC distillation** + NAC as next steps. Runs on public data + MLIP, **no rental**.

### κ benchmark across covalent semiconductors (Task 1: baseline vs fine-tuned, sc 3³ mesh 21³)
| material | baseline | fine-tuned | exp | FT/base |
|---|---|---|---|---|
| Si | 45 | 113 | 140 | 2.5× |
| AlAs | 32 | 73 | 91 | 2.3× |
| GaP | 34 | 70 | 100 | 2.1× |
| AlP | 31 | 61 | 90 | 2.0× |
| GaAs | 14 | 29 | 45 | 2.0× |
| BP | 173 | 332 | 400 | 1.9× |
| BN | 362 | 595 | 760 | 1.6× |
| InP | 16 | 25 | 68 | 1.6× |
| Ge | 14 | 19 | 60 | 1.3× |
| BAs | 109 | 145 | 1300 | 1.3× |
| SiC | 328 | 389 | 430 | 1.2× |
| C (diamond) | 1055 | 2781 | 2200 | 2.6× (overshoot) |

**Headline:** FC distillation **improves κ for every covalent material (1.2–2.6×), always toward
experiment** — systematically correcting the *universal softening-driven κ underestimate*. Many land
near experiment (AlAs 73/91, BP 332/400, SiC 389/430, BN 595/760). **Honest residuals:** pure-transfer
materials not in training stay low (Ge 19/60, InP 25/68); exotic high-κ BAs is badly underestimated
(MLIP+RTA misses its weak 3-phonon scattering); the highest-κ diamond **overshoots** (2781/2200).
Absolute κ is limited by mesh(21)+RTA+transfer — the *consistent improvement* is the robust result,
and per-material fix-vs-overshoot tracks the breadth law (in-distribution → full fix, transfer →
partial). Next: per-material vs public Togo DFT-κ (same settings); 3rd-order distillation for the residual.

## Reproduce
- `scripts/bootstrap_qe.sh` — QE + pseudopotentials. `scripts/dft_phonon.py` — DFT phonon via the
  shared pipeline. `scripts/dft_density_reuse.py` — density/wfc reuse benchmark.
- Next: more pseudopotentials (SSSP) → density-reuse on metals/oxides; non-diagonal supercells;
  QE-GPU on rented A100 for the GPU-SCF factor; generate a small self-DFT reference set for Line A.
