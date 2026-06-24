# TD-phonon project — M0 + M1.1 results (graphene Kohn-anomaly fuse)

Executable record for [`TD_PHONON_ANOMALY_EXECPLAN.en.md`](./TD_PHONON_ANOMALY_EXECPLAN.en.md)
milestones **M0** (environment + structures) and **M1.1** (graphene harmonic
baseline — *can the MLIP see the Kohn anomaly at all?*). All compute on the RTX
2060 SUPER (8 GB), total ≈ a few minutes of GPU — well inside the ≲5 GPU-hr
Gate-#1 budget.

## M0 — environment + structures  ✅
- `phonon` conda env on the 2060: `hiphive 1.5` installed (TDEP, for M1.2); the
  e3nn 0.4.4 / mace 0.3.16 / phonopy 4.1.0 / ase 3.28.0 stack loads the
  FC-distilled model cleanly (`results/finetune_mace/ft_phonon.model`, scp'd —
  it is gitignored).
- Monolayers built (ASE) + relaxed in-plane with foundation MACE (vacuum held):
  graphene `a = 2.471 Å` (planar), MoS₂ `a = 3.183 Å, t = 3.143 Å`,
  NbSe₂ `a = 3.492 Å, t = 3.379 Å`. Cached to `data/td_phonon/*.xyz`.
- Smoke test (graphene single-point F + one Γ–M–K–Γ dispersion) passes.

Scripts: `td_common.py` (builders, in-plane relaxer, explicit-path finite-
displacement dispersion driver), `td_structures.py`.

## M1.1 — graphene harmonic dispersion + Kohn-anomaly locator  ✅
**Method.** Per model: relax → phonopy finite displacement on an `n×n×1`
supercell → dispersion on **M–Γ–K–M** (both Γ and K interior, so each Kohn cusp
is two-sided). The locator (`anomaly_locate.py`) reports the top-branch frequency
and the two-sided slope discontinuity `|dv|` = kink strength at each high-symmetry
point. Foundation MACE-MP ("medium") vs the general FC-distilled MACE; both share
the supercell, so their *difference* is purely the MLIP (cell truncation cancels).

**Headline numbers** (converged 11×11 = 242-atom supercell; literature = graphene
DFPT / inelastic X-ray):

| quantity | foundation | FC-distilled | literature |
|---|---|---|---|
| Γ-E₂g (top optical) | 1262 cm⁻¹ (−21%) | 1406 cm⁻¹ (−12%) | ~1600 cm⁻¹ |
| K-A₁′ (top optical)  | 1159 cm⁻¹ (−11%) | 1194 cm⁻¹ (−8%)  | ~1300 cm⁻¹ |
| kink `|dv|` at Γ | 14.9 | 6.6 | (sharp cusp) |
| kink `|dv|` at K | 101.5 | 97.7 | (sharp cusp) |

**Convergence** (`graphene_sc_convergence.csv`, fig `graphene_sc_convergence.png`):
both frequencies and the K-kink plateau by the 50-atom (5×5) cell; the Γ-kink
stays flat and small across 18→242 atoms. So the numbers below are PES/MLIP
properties, **not** supercell-truncation artifacts.

### Interpretation (honest)
1. **Softening confirmed, model-universal in sign** — both optical modes are
   under-predicted by foundation MACE (Γ −21%, K −11%), exactly the §2.1 failure.
2. **FC-distillation recovers part of the *frequency*** (Γ −21%→−12%, K −11%→−8%)
   but the model was distilled on the paper's *bulk* materials, not graphene, so
   the cure is partial (and it contracts the lattice 2.471→2.421 Å).
3. **The K-A₁′ cusp *shape* survives** in both MLIPs — a sharp, converged kink
   (`|dv|`≈100) sits right at K. The MLIP softens the K frequency but does **not**
   wash out the K kink.
4. **The Γ-E₂g cusp is washed out** — `|dv|`≈7–15 ≪ K's ≈100, flat in supercell,
   and FC-distillation makes it *slightly worse* (6.6 < 14.9). The most
   non-analytic anomaly (the Γ LO Kohn cusp) is the casualty of MLIP smoothing,
   and bulk-material distillation does not recover it.

> **Caveat:** a finite-displacement supercell with no explicit electronic
> structure can only show a Kohn anomaly if the MLIP's fc₂ inherited it from the
> training DFT. The converged K `|dv|` is therefore *consistent with* the K-A₁′
> Kohn anomaly, but distinguishing a genuine e-ph cusp from generic
> zone-boundary curvature needs the **M1.3** DFPT / `q* ≈ 2k_F` cross-check.

### Go/No-Go gate #1 — **conditional PASS**
The pipeline works and resolves graphene's optical anomalies in ≪5 GPU-hr; the
MLIP *sees* the K cusp; we have cleanly localized where it fails (**Γ**). Two
forward branches, both consistent with the main paper:
- **M1.1b** (graphene-specific DFPT fc₂ distillation) — the decisive test of
  whether *any* distillation recovers the Γ cusp. FP64-blocked → literature fc₂
  first, own DFPT batched into the V100/A100 rental.
- **M1.2** (hiPhive-TDEP) — proceed to the (L)-channel ω(q,T); the K-anomaly is a
  good handle to track q*(T).

## Artifacts
- code: `scripts/{td_common,td_structures,harmonic_dispersion_2d,anomaly_locate,graphene_sc_convergence,plot_graphene_anomaly,plot_sc_convergence}.py`
- data: `data/td_phonon/{graphene,mos2,nbse2}.xyz`, `results/td_phonon/disp_graphene_{base,ft}.npz`, `results/td_phonon/graphene_sc_convergence.csv`
- figures: `results/figures/graphene_kohn_anomaly.png`, `results/figures/graphene_sc_convergence.png`
