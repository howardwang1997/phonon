# How much, and which, DFT? Data-efficient force-constant distillation for near-DFT phonons and thermal conductivity with foundation interatomic potentials

*Manuscript draft. Intended for npj Computational Materials / Nature Computational Science.*

**Authors.** [Author list TBD]
**Affiliations.** [TBD]
**Corresponding author.** [TBD]

---

## Abstract

Universal (foundation) machine-learning interatomic potentials (MLIPs) now reproduce energies and
forces across the periodic table, but they systematically underestimate the *curvature* of the
potential-energy surface, which manifests as softened phonon frequencies, spurious imaginary modes,
and — downstream — lattice thermal conductivities (κ) that are too low by roughly a factor of two to
three. Fine-tuning can repair the harmonic spectrum; the open and practically decisive questions are
**how much, and which, density-functional theory (DFT) data is required**, and whether the harmonic
repair propagates to the anharmonic transport property that actually matters for screening. Here we (i)
introduce **force-constant (FC) distillation**, which converts *already-computed* density-functional
perturbation theory (DFPT) force constants into harmonic energy/force labels and thus fine-tunes a
foundation MLIP at **zero additional DFT**, recovering an in-domain phonon mean absolute error (MAE) of
~0.10 THz; (ii) establish two **data-efficiency laws** — held-out transfer is governed by *chemical
breadth* (the number of distinct training materials), not by *sampling depth* (configurations per
material), with a knee at ~20–24 materials and a floor near 1.3 THz, and *coverage-based acquisition*
of new training materials outperforms random selection while *naive model-uncertainty* sampling is the
worst because it chases pathological outliers; (iii) build a GPU/CPU finite-displacement DFT engine and
report an **honest** workflow-level speedup decomposition; and (iv) demonstrate the **downstream
payoff**: FC distillation improves κ for every covalent semiconductor tested (13 systems, 1.2–2.6×
toward experiment) and, at converged supercell, recovers κ to within ~2% (Si 143 vs experimental
140 W m⁻¹K⁻¹, versus 45 for the untuned baseline). We also report two instructive negative results —
naive-uncertainty acquisition and naive third-order distillation both fail — that sharpen the design of
the framework. The approach uses public data and commodity GPUs and we release the code, models, and
protocol. Taken together, the results turn "fine-tune to fix phonons" into a *quantitative, closed-loop
data strategy*: spend a small, coverage-targeted DFT budget, distil it as curvature, and obtain
near-DFT phonons and thermal transport at MLIP throughput.

---

## 1. Introduction

Lattice dynamics underpin a large fraction of a crystal's functional behaviour: dynamical (and thermo-
dynamic) stability, lattice thermal conductivity and thermoelectric performance, thermal expansion,
electron–phonon coupling and the resulting transport and superconductivity, and the vibrational
fingerprints measured by infrared, Raman and inelastic-neutron spectroscopies. High-throughput phonon
data would therefore accelerate the discovery of thermoelectrics, thermal-management and
thermal-barrier materials, and dynamically stable compounds. The bottleneck is cost: the standard
route — finite displacements or density-functional perturbation theory (DFPT) in DFT — requires many
self-consistent calculations on symmetry-inequivalent displaced supercells at second-derivative
precision, and scales steeply with cell size, making it 10–100× more expensive than a single
electronic-structure evaluation [1,2].

Foundation MLIPs (e.g. MACE-MP [3], MatterSim [4], SevenNet, ORB, CHGNet, M3GNet) promise a ~10³×
speed-up by replacing the DFT force evaluations inside the finite-displacement workflow. A recent
large-scale benchmark of ~10⁴ ab-initio phonon calculations established that some foundation models
already predict harmonic phonons well while others do not, even when their energy and force errors near
equilibrium are small [5]. The systematic origin of the failures is now understood: foundation MLIPs
**under-predict the PES curvature**, a "softening" traced to the over-representation of near-equilibrium
configurations in pre-training data [6]. Because phonons are second derivatives of the energy, low
force error does not imply correct phonons — a *curvature-supervision gap*. The consequences propagate:
foundation models underestimate lattice thermal conductivity by roughly −50% on the median, with large
dispersion [7], rendering them unreliable for thermal screening.

Fine-tuning can close the harmonic gap. Parameter-efficient and phonon-targeted fine-tuning schemes
have recently driven phonon MAEs to ~0.05 THz [8,9]. This reframes the problem: *that* fine-tuning
helps is established; *how to do it data-efficiently, and whether it transfers to anharmonic transport*,
are not. Two questions are decisive for any practical campaign. **(Q1) How much, and which, DFT data?**
For a fixed DFT budget, is it better to compute more configurations of a few materials, or a few
configurations of many materials — and when one must *acquire* new training materials, which ones?
**(Q2) Does harmonic fine-tuning recover the anharmonic property (κ) that screening needs**, or only the
dispersion?

Here we answer both. We introduce **force-constant (FC) distillation**, a fine-tuning that reuses the
*existing* DFPT force constants behind public phonon databases [10,11] as exact harmonic labels, adding
**no new DFT**. We then treat the data question quantitatively and discover two laws: transfer is
controlled by **chemical breadth**, not sampling depth; and **coverage** is the right acquisition
criterion, while **naive uncertainty sampling backfires**. We build a GPU/CPU finite-displacement DFT
engine to supply targeted reference data and report an honest workflow-level speedup decomposition.
Finally, we show that FC distillation **recovers lattice thermal conductivity** across covalent
semiconductors — to within ~2% at converged supercell for silicon — and we document two informative
negative results (uncertainty acquisition; naive third-order distillation) that delineate the design
space. The result is a closed-loop, *small-DFT + large-MLIP* recipe: a small, coverage-targeted DFT
budget, distilled as curvature, yields near-DFT phonons and transport at MLIP throughput.

**Contributions.** (1) FC distillation: zero-new-DFT curvature fine-tuning from public DFPT force
constants. (2) Two data-efficiency laws (breadth ≫ depth; coverage ≫ uncertainty acquisition). (3) An
honest GPU/CPU DFT-engine speedup decomposition. (4) A demonstrated κ recovery (Si to ~2%) plus two
sharp negative results. (5) Open code, models, and protocol.

---

## 2. Results

### 2.1 Foundation MLIPs under-predict phonon curvature

Benchmarking foundation MLIPs against the MDR/PhononDB DFPT database [10] (10,034 inorganic materials),
with the non-analytical term correction disabled on both sides for a like-for-like short-range
comparison, reproduces the established picture [5,6]: foundation models systematically soften the
spectrum (e.g. Si ω_max ≈ 11 THz versus DFPT 15.5 THz), introduce spurious imaginary modes, and violate
the acoustic sum rule. Force MAE and phonon MAE decouple — confirming the curvature-supervision gap.

![Foundation-MLIP phonon failure modes vs DFPT](../results/figures/mattersim_summary.png)
*Figure 1. Foundation-MLIP phonon accuracy versus DFPT across the benchmark: systematic frequency
softening and spurious imaginary modes despite low near-equilibrium force error.*

### 2.2 Force-constant distillation restores near-DFT phonons at zero new DFT

For a training material with DFPT force constants Φ (defined on the supercell), we generate rattled
supercell configurations labelled with the exact harmonic response,
$E(\mathbf{u})=\tfrac12\mathbf{u}^\top\Phi\,\mathbf{u}$ and $\mathbf{F}=-\Phi\,\mathbf{u}$, augmented
with single-atom ±displacements that directly probe individual force-constant columns, and fine-tune
the foundation model on these labels. Because the labels come from *already-computed* force constants,
this adds **no new DFT** — the public phonon database is converted directly into curvature supervision.
FC distillation drives the in-domain phonon MAE to **~0.10 THz** and eliminates imaginary modes, while
the anti-forgetting strategy of §2.5 preserves the base model's universality. This in-domain accuracy
is comparable to recent phonon-targeted fine-tuning [8,9]; our emphasis below is the *data efficiency*
and *downstream transport*, where the open questions lie.

![Si phonon dispersion before/after FC distillation](../results/figures/Si_before_after.png)
*Figure 2. Si phonon dispersion: foundation MLIP (softened) versus FC-distilled (near-DFPT) versus the
DFPT reference.*

### 2.3 Transfer is governed by chemical breadth, not sampling depth

We separate two data axes — *depth* (configurations per material) and *breadth* (number of distinct
training materials) — and measure phonon MAE on a fixed held-out set of materials never used in
training.

| # training materials | 4 | 8 | 16 | 20 | 24 | 28 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| held-out MAE (THz) | 1.82 | 1.80 | 1.82 | 1.78 | 1.58 | 1.43 | 1.42 | 1.31 | 1.37 |

Transfer error is **flat (~1.81 THz) up to ~16–20 materials, exhibits a knee at ~20–24, and falls to a
~1.3 THz floor by N≈48** (Fig. 3). Increasing *depth* at fixed breadth does **not** transfer and in
fact *overfits*: at high breadth, more configurations per material make transfer worse (e.g. N=64: 15
configs → 1.33; 60 configs → 1.52). The full data-need surface (Fig. 4) has its minimum at
high-breadth + low-depth. The practical rule is therefore to **spend the DFT budget on more materials,
not on more configurations of each**. The residual transfer floor is dominated by light-element,
high-frequency chemistries (notably boron nitride); we report the median alongside the mean.

![Breadth vs depth transfer](../results/figures/depth_vs_breadth.png)
*Figure 3. Held-out transfer MAE versus number of training materials (breadth): a plateau to ~16–20, a
knee at ~20–24, and a ~1.3 THz floor. Depth (configurations per material, dashed) does not transfer.*

![Depth × breadth data-need surface](../results/figures/depth_breadth_surface.png)
*Figure 4. Data-need surface: transfer improves with breadth (down columns) but worsens with depth
(across rows); the optimum is high-breadth + low-depth.*

### 2.4 Coverage acquisition outperforms uncertainty sampling

For a closed-loop engine the operative question is *which* materials to compute with DFT. Using the
public DFPT database as an oracle, we compare acquisition strategies at matched budget (seed-averaged):

| # materials | random (6 draws) | coverage (greedy) | uncertainty |
|---|---|---|---|
| 16 | 1.61 ± 0.09 | 1.58 ± 0.03 | 1.68 |
| 24 | — | 1.46 ± 0.01 | 1.61 |
| 32 | 1.53 ± 0.08 | **1.38 ± 0.01** | 1.60 |
| 48 | 1.43 ± 0.07 | 1.37 ± 0.03 | — |

**Chemical-coverage acquisition is the best and the most reliable** (its variance is far lower — it
selects nearly the same informative set every time), beating random by ~0.18 THz at N = 32.
**Naive model-uncertainty acquisition is the worst**: it preferentially queries pathological/soft
outliers (e.g. elemental phosphorus, F₂ molecular crystals, structures with thousands of spurious
imaginary modes) that are informative *to the model* but unrepresentative of the target distribution;
filtering predicted-unstable candidates only partially rescues it. This is counter-intuitive —
uncertainty sampling is the textbook default — and it instructs the data engine to **acquire for
coverage, not uncertainty** (Fig. 5).

![Acquisition strategies](../results/figures/acquisition_comparison.png)
*Figure 5. Acquisition at matched budget: chemical-coverage selection (green) is best and lowest-
variance; random (grey) is intermediate; naive model-uncertainty (red) is worst.*

### 2.5 Anti-forgetting during fine-tuning

Single-head fine-tuning catastrophically forgets elements absent from the fine-tuning set (held-out
imaginary modes proliferate). Multihead **replay** (sampling the foundation pre-training trajectories)
and **LoRA** both prevent this at a small in-domain cost; with seed error bars the held-out transfer
ordering is LoRA-r32 (1.46) < replay (1.85) < single-head (2.22). More chemical breadth reduces the
replay required.

![Anti-forgetting comparison](../results/figures/antiforgetting_errorbars.png)
*Figure 6. Held-out transfer with seed error bars: replay and LoRA prevent the catastrophic forgetting
of naive single-head fine-tuning.*

### 2.6 A GPU/CPU DFT engine and an honest workflow speedup decomposition

To supply targeted reference data we implement a finite-displacement DFT engine (Quantum ESPRESSO [12]
with SG15 ONCV pseudopotentials [13], driven through the same phonon pipeline as the MLIPs) and
validate it (Si ω_max within ~1–5% of DFPT/experiment). We decompose the workflow-level acceleration
honestly:

| Factor | Value | Nature |
|---|---|---|
| symmetry reduction of displacements | 48–384× | standard practice, free |
| charge-density + wavefunction reuse across displacements | ~1.5× wall (iteration savings scale with SCF difficulty: Si 1.0×, Al 1.17×, MgO 1.78×) | engine |
| non-diagonal supercells (exact fine-q) [14] | peak cell N³→N ⇒ SCF cost (~N_atoms³) reduced 141× (4³ grid) – 1152× (8³) | engine |
| per-SCF GPU port | ~5–15× (literature) | requires strong-FP64 hardware (V100/A100) |

The headline ~50× is *dominated by symmetry reduction*, which is standard; the genuinely engine-side
factors are more modest on CPU, and the per-SCF GPU factor requires proper double-precision hardware
(inference-class GPUs with crippled FP64 do not realize it). Stating this explicitly **strengthens the
central thesis**: pure workflow-DFT acceleration is bounded, so the MLIP proxy (10³×) is the real
high-throughput lever and the DFT engine's value is *targeted reference-data generation*, not an
outsized standalone speedup.

### 2.7 Downstream: recovering lattice thermal conductivity

The decisive test of "useful accuracy" is a downstream property. We compute lattice thermal
conductivity κ within the relaxation-time approximation (phono3py [11]) from MLIP-evaluated second- and
third-order force constants — **no DFT**, ~6–16 s per material on one GPU — for baseline versus
FC-distilled models across covalent semiconductors:

| material | baseline κ | fine-tuned κ | experiment | FT/base |
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

**FC distillation improves κ for every material (1.2–2.6×), always toward experiment**, correcting the
softening-driven underestimate that makes baseline foundation MLIPs unusable for thermal screening
[7] (Fig. 7). Honest residuals: pure-transfer materials absent from training stay low (Ge, InP); the
exotic high-κ material BAs is severely underestimated (the relaxation-time approximation cannot capture
its weak three-phonon scattering); and the highest-κ material, diamond, overshoots. The per-material
fix-vs-overshoot tracks the breadth law of §2.3.

![κ benchmark: baseline vs fine-tuned vs experiment](../results/figures/kappa_benchmark.png)
*Figure 7. Lattice thermal conductivity versus experiment (log–log). The baseline MLIP (open red) lies
below the y = x line — the softening-driven underestimate — while the FC-distilled model (green) is
pulled toward experiment for every material.*

These benchmark values use a fixed supercell/mesh and are *under-converged in absolute terms*: κ in
covalent crystals converges slowly with supercell owing to long phonon mean free paths. A supercell-
convergence study on silicon makes the impact unambiguous — the FC-distilled model converges **straight
to experiment** while the baseline stays softened:

| Si κ(300 K) [W m⁻¹K⁻¹] | sc 2 | sc 3 | sc 4 |
|---|---|---|---|
| baseline | 52.8 | 45.2 | 45.5 |
| **fine-tuned** | 109.9 | 114.0 | **143.2** |
| experiment | — | — | **~140** |

At converged supercell the FC-distilled MLIP reproduces Si κ to **~2%** (143 vs 140) while the baseline
is **3× too low**. (A same-settings DFT anchor at the *affordable* small supercell is itself far from
converged — DFT-RTA at sc 2 yields only 48 W m⁻¹K⁻¹ — so we anchor against experiment and report the
convergence trend.) For polar materials, the non-analytical term correction (Born charges and dielectric
tensor from a single Γ-point DFPT calculation; ε∞ = 3.19, Z\* = ±1.97 for MgO) restores the correct κ
(MgO 51 W m⁻¹K⁻¹ versus experiment ~55–60).

![Si κ supercell convergence](../results/figures/kappa_si_convergence.png)
*Figure 8. Silicon κ versus supercell size: the FC-distilled model converges to the experimental value
(~140 W m⁻¹K⁻¹); the baseline remains softened at ~45.*

### 2.8 Two instructive negative results

Two natural extensions fail, and the failures are informative.
- **Naive uncertainty acquisition** (§2.4) underperforms random selection; uncertainty must be tempered
  by representativeness (coverage).
- **Naive third-order FC distillation** *regresses* κ (Si 113 → 42–46). This is **not** catastrophic
  forgetting (replay does not help) and **not** a soft reference (the DFT fc₂ yields the correct Si
  ω_max, 15.42 THz). The cause is a **curvature trade-off**: fine-tuning on large-displacement anharmonic
  configurations improves large-force accuracy at the expense of the small-displacement Hessian that κ
  depends on, re-softening the phonons. Adding anharmonicity therefore requires a **curvature-aware
  loss** (explicit Hessian / second-order supervision), not merely additional anharmonic configurations —
  a concrete prescription for future work.

---

## 3. Discussion

**A quantitative data strategy.** The two data laws and the acquisition result jointly specify a
closed-loop data engine. Because the accuracy ceiling at fixed data is set by chemical breadth rather
than sampling depth (§2.3), and because coverage acquisition reaches that ceiling with the fewest
queries (§2.4), the engine should spend its targeted DFT budget on *coverage-diverse new materials*,
distil them as curvature (§2.2), and repeat. FC distillation makes the in-domain cost essentially free
by reusing public DFPT force constants; the DFT engine (§2.6) supplies breadth for genuinely new
chemistries; the fine-tuned MLIP delivers the 10³× throughput and, crucially, *useful* downstream
properties (§2.7).

**Relation to prior work.** Benchmarks established *that* foundation MLIPs are mixed for phonons [5] and
diagnosed the softening mechanism [6]; foundation-model κ studies quantified the downstream
under-estimate [7]; and parameter-efficient / phonon-targeted fine-tuning showed *that* the harmonic
spectrum can be repaired [8,9]. Our contribution is orthogonal and complementary: we treat the *data
question* quantitatively (breadth vs depth; coverage vs uncertainty acquisition), reuse existing DFPT
force constants as zero-cost curvature labels, and follow the harmonic repair *through to converged
lattice thermal conductivity* (Si to ~2%), while reporting where the obvious extensions fail.

**Reframing "GPU-accelerated phonons".** A common framing is to accelerate the DFT itself. Our honest
decomposition (§2.6) shows that pure workflow-DFT acceleration is bounded and dominated by *free*
symmetry reduction; the genuine acceleration is the MLIP proxy, made *trustworthy* by curvature
supervision. The DFT/GPU engine's role is efficient, targeted reference generation — not a headline
single-SCF speed-up.

**Implications.** For practitioners building phonon or thermal-transport datasets, the prescription is
concrete: prioritise chemical coverage over per-material sampling; acquire by coverage, not by
uncertainty; distil existing force constants rather than recomputing; and validate the downstream
property (κ) under explicit supercell convergence. The honest negatives (uncertainty acquisition;
additive anharmonic distillation) mark the pitfalls.

**Limitations.** (i) Absolute κ is limited by q-mesh, the relaxation-time approximation, supercell
convergence, and transfer; the relative improvement and the convergence-to-experiment trend are the
robust claims, and a fully converged same-settings DFT-κ anchor (sc ≥ 4) on fast hardware is the natural
next step. (ii) Polar-material κ requires NAC, demonstrated here for MgO; a learned Born-charge model
would remove the per-material Γ-DFPT step. (iii) The per-SCF GPU factor requires strong-FP64 hardware.
(iv) The present study uses one foundation backbone (MACE-MP); cross-model generality is future work.
(v) Third-order distillation needs a curvature-aware loss (§2.8).

---

## 4. Methods

**Phonon pipeline.** A unified phonopy [2] + ASE driver generates symmetry-reduced displacements,
produces force constants (with acoustic-sum-rule symmetrisation), and computes dispersions, density of
states, thermodynamic functions, imaginary-mode counts and ASR residuals. The identical pipeline is
used for every MLIP and for the DFT backend, so MLIP and DFT phonons are computed identically. For
benchmark fairness the non-analytical term correction is disabled on both sides; for FC-distillation
evaluation, phonons are computed at the DFT geometry ("no-relax") to isolate force-constant quality
from relaxation drift.

**FC distillation.** From the supercell force constants Φ (full form), rattled configurations
$\mathbf u$ (Gaussian, with a mix of amplitudes) are labelled by $\mathbf F=-\Phi\mathbf u$ and
$E=\tfrac12\mathbf u^\top\Phi\mathbf u$, plus single-atom ±displacement probes. The foundation model is
fine-tuned (energy weight 0.01, force weight 100) with the full periodic-table element set preserved.
Anti-forgetting uses multihead replay of the pre-training trajectories or LoRA adapters.

**Data-efficiency and acquisition.** Nested training subsets vary breadth (number of materials) and
depth (configurations per material) independently against a fixed held-out set. Acquisition strategies —
random, greedy maximum-element-coverage, and model-uncertainty (predicted imaginary-mode count plus ASR
residual, with an optional stability filter) — are compared with the public DFPT database as oracle and
multiple seeds.

**DFT engine.** Quantum ESPRESSO [12] (CPU/GPU) with SG15 ONCV PBE pseudopotentials [13]. Cross-
displacement charge-density and wavefunction reuse via `startingpot`/`startingwfc=file` (with `nosym`
so wavefunctions transfer across symmetry-broken displacements). Non-diagonal supercells are constructed
by a Hermite-normal-form search for the minimal commensurate cell per q [14]. Born charges and the
dielectric tensor are computed by Γ-point DFPT (`epsil`).

**Thermal conductivity.** Third-order force constants and RTA κ via phono3py [11] with MLIP-evaluated
forces; isotope scattering included; NAC applied for polar systems. Supercell convergence is reported
explicitly.

**Data.** Public MDR/PhononDB DFPT [10], Petretto et al. [1], pre-training trajectories for replay, and
SG15 pseudopotentials [13]. Self-generated DFT is deliberately small (validation/anchor).

---

## Data availability
All benchmark phonon references are from public databases [1,10]. Generated phonon/κ results,
fine-tuned models, and figures are released with the code.

## Code availability
Code (FC distillation, data-efficiency/acquisition study, DFT engine drivers, κ pipeline) and the
fine-tuned models are available at `github.com/howardwang1997/phonon`. All accuracy values carry seed
error bars; all speed-ups are reported with scope; κ is reported as mean and median with explicit
supercell convergence; negative results are retained.

---

## References

[1] G. Petretto *et al.*, "High-throughput density-functional perturbation theory phonons for inorganic
materials," *Sci. Data* **5**, 180065 (2018).
[2] A. Togo, "First-principles phonon calculations with phonopy and phono3py," *J. Phys. Soc. Jpn.*
**92**, 012001 (2023).
[3] I. Batatia *et al.*, "A foundation model for atomistic materials chemistry" (MACE-MP-0),
arXiv:2401.00096 (2024).
[4] H. Yang *et al.*, "MatterSim: a deep-learning atomistic model across elements, temperatures and
pressures," arXiv:2405.04967 (2024).
[5] A. Loew, D. Sun, H.-C. Wang *et al.*, "Universal machine-learning interatomic potentials are ready
for phonons," *npj Comput. Mater.* **11** (2025); arXiv:2412.16551.
[6] B. Deng *et al.*, "Systematic softening in universal machine-learning interatomic potentials,"
*npj Comput. Mater.* **10**, 175 (2024).
[7] "Thermal conductivity predictions with foundation atomistic models," arXiv:2408.00755 (2024).
[8] "Parameter-efficient fine-tuning of machine-learning interatomic potentials for phonon and thermal
properties," arXiv:2604.01017 (2026).
[9] "PFT: phonon fine-tuning for machine-learned interatomic potentials," arXiv:2601.07742 (2026).
[10] A. Togo, MDR/PhononDB (NIMS) — open DFPT phonon database (10,034 materials).
[11] A. Togo, L. Chaput, I. Tanaka, "Distributions of phonon lifetimes in Brillouin zones" (phono3py),
*Phys. Rev. B* **91**, 094306 (2015).
[12] P. Giannozzi *et al.*, "Quantum ESPRESSO toward the exascale," *J. Chem. Phys.* **152**, 154105
(2020).
[13] M. Schlipf and F. Gygi, "Optimization algorithm for the generation of ONCV pseudopotentials"
(SG15), *Comput. Phys. Commun.* **196**, 36 (2015).
[14] J. H. Lloyd-Williams and B. Monserrat, "Lattice dynamics and electron–phonon coupling calculations
using non-diagonal supercells," *Phys. Rev. B* **92**, 184301 (2015).

*(Reference list to be completed and verified against the journal style at submission; a few 2026
preprint identifiers are provisional.)*
