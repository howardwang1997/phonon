# Results Summary — phonon MLIP + GPU-DFT framework (2026-06)

Consolidated inventory of all results. Detail in `CAMPAIGN_FINDINGS.md` (Line A),
`LINE_B_FINDINGS.md` (Line B), `NCS_ROADMAP.md` (framing), `NCS_EXECUTION_PLAN.md` (compute/data).

## Line A — accuracy science (foundation MLIP phonons + FC distillation) ✅ solid
| result | finding | figure |
|---|---|---|
| Failure modes | foundation MLIPs soften phonons (>90%), spurious imaginary (5–20%), ASR violation | `mattersim_summary.png` |
| FC distillation | harmonic E/F from public DFPT force constants, **zero new DFT** → in-domain **~0.10 THz**, imaginary→0 | `Si_before_after.png` |
| Anti-forgetting (error bars) | LoRA-r32 **1.46**, replay-pt1k 1.85, single-head 2.22 (held-out) | `antiforgetting_errorbars.png` |
| **Breadth law** ★ | transfer MAE flat ~1.81 (N≤16–20) → knee ~20–24 → **1.27–1.37 (N=48–79)**. *Breadth > depth.* | `depth_vs_breadth.png` |
| Depth × breadth surface | depth **overfits**; at high breadth more depth *hurts*; min = high-breadth+low-depth | `depth_breadth_surface.png` |
| **E7 acquisition** ★ | **coverage > random > naive-uncertainty** (counterintuitive); uncertainty chases pathological outliers | `acquisition_comparison.png` |
| Transfer floor | dominated by BN (light elements, high-ω) — report mean+median | — |

## Line B — GPU/CPU DFT engine + downstream impact
| result | finding | status |
|---|---|---|
| DFT engine validated | QE finite-displacement: Si ω_max ~1–5% vs DFPT/exp | ✅ |
| Speedup decomposition (honest) | symmetry **48–384×** (standard), density+wfc reuse **~1.5×** (scales w/ SCF difficulty), **non-diagonal supercells 141–1152×** (exact fine-q), GPU-SCF deferred | ✅ |
| Self-DFT harmonic dataset | validated vs MDR (**~1.4%** mean); SG15 pseudos (69 elem) unblock chemistry | ✅ (broad CPU set slow, ~partial) |
| Loop closure | self-DFT FC → MLIP distillation works (MAE 0.12–0.21) | ✅ verified |
| **NAC (Task 2)** | QE Born charges (ε=3.19, Z\*=±1.97) → MgO κ baseline+NAC **51** (exp ~55–60) | ✅ |
| **κ benchmark (Task 1)** ★★ | **FC distillation improves κ for *every* covalent material 1.2–2.6× toward exp** (13 systems); **at converged supercell recovers κ to ~2% (Si FT 143 vs exp 140, baseline 45)** | ✅ impact main result |
| κ vs same-settings DFT | DFT-RTA at sc2 itself under-converged (Si 48 vs exp 140) → κ is supercell-sensitive; relative FT improvement + convergence-to-exp is the robust claim | ✅ |
| **3rd-order distillation (Task 3)** | **NEGATIVE**: regresses κ (113→42–46). Not forgetting, not soft ref (fc2 ω_max 15.42 ✓). Cause: large-displacement anharmonic training degrades the phonon **Hessian**; needs a **curvature-aware loss** | ✅ instructive negative |
| DFPT (ph.x) cross-check | engine works; 2060 too slow (desktop CPU) | 🟡 partial |

## Fusion / framework
- **Closed-loop thesis** (`NCS_ROADMAP.md`): small DFT + large MLIP; data engine ⟷ model; acquisition = coverage.
- The E7 finding (coverage acquisition) **and** the Task-3 negative (curvature loss needed) directly shape the data-engine + method design.

## Honest scorecard
- **Strong/robust:** breadth law, depth×breadth, E7 acquisition, anti-forgetting, FC-distillation accuracy, **κ benchmark (the impact)**, honest speedup decomposition.
- **Instructive negatives (keep — they strengthen credibility):** naive-uncertainty acquisition fails; naive 3rd-order distillation regresses κ.
- **Caveats:** κ absolute values limited by mesh(21)+RTA+transfer (compare vs Togo DFT-κ next); polar κ needs NAC (done for MgO); broad self-DFT set slow on H20 CPU; H20 FP64 too weak for GPU-SCF.

## Compute reality
- Everything above ran on **public data + the existing machines (8×H20, 2×H20, 2060), no rental.**
- The **only** hard rental gate is the **GPU-SCF factor** (needs V100/A100; ~2–3 days lean).
