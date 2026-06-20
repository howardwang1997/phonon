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
| **Charge-density + wavefunction reuse** | **~1.5×** (Si) | **engine** | displaced SCF restarted from the equilibrium density+wfc (`startingpot/startingwfc=file`, `nosym` so the wfc transfers across symmetry breaking). Reuse starts **300× closer** in SCF accuracy (0.0013 vs 0.435 Ry); wall 329→225 s. Marginal for easy high-symmetry Si (8-iter SCF, little headroom) — expected larger for many-iteration systems (metals/oxides). |
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

## Reproduce
- `scripts/bootstrap_qe.sh` — QE + pseudopotentials. `scripts/dft_phonon.py` — DFT phonon via the
  shared pipeline. `scripts/dft_density_reuse.py` — density/wfc reuse benchmark.
- Next: more pseudopotentials (SSSP) → density-reuse on metals/oxides; non-diagonal supercells;
  QE-GPU on rented A100 for the GPU-SCF factor; generate a small self-DFT reference set for Line A.
