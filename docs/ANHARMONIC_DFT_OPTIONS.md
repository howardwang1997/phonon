# Making converged anharmonic DFT (fc3) feasible — research note

**Problem.** Supercell finite-displacement fc3 cost = (# displaced supercells) × (supercell SCF), and
the supercell SCF scales steeply with cell size. Measured on H20 CPU: Si sc3 (54-atom) ≈ **31 min/SCF**
(bandwidth-bound), 181 configs ⇒ ~4 days; sc4 (128-atom) ⇒ ~weeks. Infeasible — which is *why the
project uses MLIPs*. To get a **converged DFT fc3 reference** (needed to turn the 3rd-order-distillation
negative into a positive) without renting A100s, two routes:

---

## Option A — D3Q / thermal2 : third-order DFPT (supercell-FREE) ★ biggest win
Refs: Paulatto, Mauri, Lazzeri, *Phys. Rev. B* **87**, 214303 (2013); https://anharmonic.github.io.

**How.** Computes the third-order dynamical matrix D³(q₁,q₂,q₃) **directly by DFPT** (2n+1 theorem:
3rd-order energy from 1st-order wavefunction responses) on a q-point grid, **in the primitive cell** —
no supercells. Pipeline: `pw.x` (scf) → `ph.x` (fc2/DFPT) → `d3q.x` (D³ on the grid) → `d3_q2r.x`
(→ real-space fc3) → `thermal2` (`d3_lw.x`/`d3_tk.x`) for linewidths + κ.

**Why it kills the cost.** Cost = (# irreducible q-triplets) × (primitive-cell DFPT), NOT
N_configs × (128-atom SCF). For **Si the primitive cell is 2 atoms** → each DFPT is tiny, and the
H20's weak FP64 doesn't matter (small cell). A converged 3rd-order grid (e.g. 4×4×4, symmetry-reduced
to ~tens of triplets) ⇒ **~hours, supercell-free** → turns "weeks" into "hours." Scales to any
material and is reusable — effectively a *supercell-free anharmonic DFT engine* (a Line-B asset).

**Hurdle.** d3q is **not** in conda-forge QE; needs `make d3q` from QE **source** (compile, ~30–60 min
+ deps). For distillation we also need the fc3 in phono3py format → write a small `d3_q2r → phono3py
fc3.hdf5` converter (or use thermal2 directly for the DFT-κ validation).

**Fit for us.** Best route to a *truly converged* Si fc3 cheaply and repeatedly; Si's tiny primitive
cell is the ideal case. Bonus: a reusable engine + a methods contribution.

---

## Option B — phono3py cutoff + random displacements + compressed sensing ★ smallest change
Refs: phono3py `--cutoff-pair`, `--rd`, symfc/ALM (Togo, *JPSJ* 92, 012001, 2023); compressive-sensing
lattice dynamics (Zhou *et al.*, arXiv:1805.08904 / PRB Mater. 2019); hiphive (Eriksson–Fransson–Erhart).

**How.** (i) `--cutoff-pair Rc` truncates fc3 to atom triplets within Rc → **fewer** displaced
supercells *and* allows a **smaller** supercell (need only ≳ 2·Rc). (ii) `--rd N` random-displacement
configs + **regression fit** (symfc / ALM, or ℓ1-LASSO compressed sensing) recovers fc3 from a *small*
dataset by exploiting that only a few IFCs are non-negligible.

**Why partial.** Random displacements *alone* need ~comparable config count (~100–150). The real
reduction is **cutoff + regression**: for Si (short-range anharmonicity, Rc ~4–5 Å) a cutoff lets you
use **sc3 (54-atom) or even smaller** and ~**20–50** random configs + one symfc fit. Each config is
still a supercell SCF, but far fewer + smaller cells ⇒ ~hours (not weeks) — especially if we **drop
concurrent workers to 1–2** (the 31-min/SCF was bandwidth contention from 4×24 ranks; a clean single
worker on 54 atoms is ~minutes).

**Hurdle.** Minimal — stays in our finite-displacement framework. `pip install symfc`; modify
`dft_fc3.py` to emit `--rd` + `--cutoff-pair` displacements and fit fc3. No compilation.

**Caveat.** Accuracy depends on Rc + #configs → needs a convergence check (κ vs Rc).

> Note: phono3py also recommends **pypolymlp** (train an MLP on a few configs, generate fc3 from it).
> We deliberately **avoid** that here — it would make the *reference itself* an MLP, defeating the
> point of validating *DFT*-fc3 distillation. Compressed sensing keeps the reference pure-DFT.

---

## Recommendation
Do **both**, in order of effort:

1. **Now (low effort, retry the curvature validation today):** Option B — `dft_fc3.py` with
   `--cutoff-pair ~4.5 Å` + `--rd ~40` + symfc fit, on **sc3, single/dual worker** (no bandwidth
   contention). Estimate ~**2–5 h** for a converged-ish Si fc3 → re-distil → does κ hold ~140?
2. **Parallel (rigorous + reusable):** Option A — compile **QE+d3q** on one freed 4×H20 box →
   supercell-free Si fc3 in ~**hours** → the *truly* converged reference + the DFT-κ anchor via
   thermal2, and a Line-B "supercell-free anharmonic DFT engine."

Both are **no-rental** (run on the existing H20s; D3Q especially, since Si's primitive cell is tiny so
weak FP64 is irrelevant). Either one lets us **complete the converged-fc3 curvature validation** and,
if κ lands ~140, turn the 3rd-order-distillation negative into a positive — strengthening Line B and
removing the one "needs rental" caveat from §2.8/§5 of the paper.

**Paper angle.** This also reframes the cost story: the supercell-finite-displacement wall (weeks) is
real, but **third-order DFPT (D3Q) sidesteps it in the primitive cell** — so the converged anharmonic
reference *is* obtainable cheaply, just not via supercells. That is a cleaner, stronger narrative than
"needs A100s."
