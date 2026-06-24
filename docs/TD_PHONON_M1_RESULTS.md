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

## M1.2 — temperature-dependent omega(q,T) via hiPhive-TDEP  ✅ (L-channel)
**Method.** Per T: Langevin MD (FC-distilled MLIP forces) on a 6×6×1 = 72-atom
supercell, 2 ps equilibration + 150 decorrelated snapshots (every 30 fs) →
hiPhive effective harmonic fc₂(T) → phonopy dispersion on M-Γ-K-M → top-branch
frequency + Kohn-kink at Γ/K. Atom order follows phonopy's supercell so the
hiphive fc₂ maps straight on; the ClusterSpace is built from a freshly-rebuilt
*symmetric* primitive at the model's relaxed `a` (the relaxer's in-plane shear
otherwise trips hiphive's orbit enumeration). All on the RTX 2060, ≈ 20 min total
(≈ 400 s / T). Thermalization is on target (T_inst = 103 / 308 / 609 K).

| T (K) | Γ-E₂g (cm⁻¹) | K-A₁′ (cm⁻¹) | kink_Γ | kink_K | fit rmse (meV/Å) |
|---|---|---|---|---|---|
| 0 (M1.1 fc) | 1399 | 1192 | 6.6 | ~98 | — |
| 100 | 1397 | 1181 | 5.7 | 95.3 | 64 |
| 300 | 1381 | 1170 | 6.0 | 95.4 | 174 |
| 600 | 1364 | 1158 | 5.8 | 92.4 | 325 |

**Findings (all (L)-channel).**
1. **Monotonic thermal softening** of both optical modes: Γ-E₂g −33 cm⁻¹ (−2.4%)
   and K-A₁′ −34 cm⁻¹ (−2.9%) over 0→600 K — the phonon-population / anharmonic
   renormalization the MLIP *can* see (fig `graphene_td_dispersion.png`).
2. **The K-A₁′ cusp persists across T** (kink_K ≈ 95, flat to 300 K, slight
   broadening to 92 at 600 K) — in the (L) channel the anomaly is only weakly
   T-dependent.
3. **The Γ cusp stays washed out at every T** (kink_Γ ≈ 6), consistent with M1.1.
4. **Fit rmse grows with T** (64→174→325 meV/Å) — expected: larger displacements
   → more fc₃⁺ anharmonicity outside the effective fc₂ (motivates adding fc₃ for
   linewidths later).

> **Honesty (the plan's core tension).** This is the **(L) lattice-anharmonic**
> channel ONLY. The **(E) electronic** channel — Fermi-Dirac smearing that
> broadens/moves the Kohn anomaly with electronic temperature — is **not in the
> MLIP** (ground-state BO PES). Real graphene's K-anomaly would broaden more
> strongly with T from (E); quantifying that needs DFPT-with-smearing (M1.3,
> FP64/rental). So the reportable M1.2 result is: *the (L) channel gives a small
> monotonic softening and a weakly-T-broadening K cusp; the electronic broadening
> is a separate, MLIP-inaccessible channel.*

## M2 (L-channel) — dense-T q*(T) map  ✅
Extended M1.2 to **8 temperatures (10–600 K)**, FC-distilled MLIP, 6×6×1, in one
MD pass (`td_anharmonic.py`, ~44 min on the 2060). K-A₁′ softens monotonically
~1187→1172 cm⁻¹; the K cusp persists (kink_K ≈ 95±5); Γ-E₂g shows the same downward
trend but **noisier** — its ~20 cm⁻¹ T-shift sits inside the ±~10 cm⁻¹ per-T
stochastic-TDEP scatter (each T is an independent stochastic fit), so a clean Γ(T)
needs more snapshots / ensemble averaging. Figure `graphene_td_m2.png`.

## Anharmonicity diagnostic — quantifying the model's (L) anharmonicity  ✅
Same MD snapshots, fit **fc₂-only vs fc₂+fc₃** (hiPhive) → how anharmonic the MLIP's
PES sampling is, and how much is cubic. This is a **quantitative target for Path P's
DFT distillation** (after distillation, compare these same numbers to DFT). Figure
`graphene_anharmonicity.png`.

| T (K) | total \|F\| | resid. after fc₂ | resid. after fc₂+fc₃ | anharmonic share of \|F\| | cubic share | ‖fc₃‖ |
|---|---|---|---|---|---|---|
| 10  | 182  | 7   | 1  | 3.8%  | 85% | 7498 |
| 100 | 581  | 62  | 14 | 10.6% | 78% | 6536 |
| 300 | 1026 | 190 | 54 | 18.5% | 72% | 5236 |
| 600 | 1465 | 305 | 95 | 20.8% | 69% | 4871 |

(meV/Å for forces; ‖fc₃‖ in the hiPhive fc₃ array norm.)

**Findings.**
1. **Anharmonic force fraction grows 3.8%→~21%** over 10→600 K — at 600 K ~1/5 of the
   thermal force is beyond any harmonic fit.
2. **Cubic share falls 85%→69%** — at low T the anharmonicity is almost all cubic
   (fc₃); at high T the quartic+ (fc₄…) contribution grows, as it physically should.
3. **Effective ‖fc₃‖ drops 7498→4871** — the cubic coupling itself thermally
   renormalizes (effective fc₃(T) softens, like effective fc₂(T)).

> **Honest scope.** These are the MLIP's *self*-anharmonicity (no DFT reference yet);
> the diagnostic *defines the metric and the target* — Path P (P-A fc₃ distillation /
> P-B thermal-config DFT-force distillation) is what supplies the DFT comparison and
> the actual fix. Still (L)-channel only.

## Artifacts
- code: `scripts/{td_common,td_structures,harmonic_dispersion_2d,anomaly_locate,graphene_sc_convergence,td_phonon,td_anharmonic,plot_graphene_anomaly,plot_sc_convergence,plot_td_dispersion,plot_anharm_diag}.py`
- data: `data/td_phonon/{graphene,mos2,nbse2}.xyz`, `results/td_phonon/disp_graphene_{base,ft}.npz`, `graphene_sc_convergence.csv`, `td_graphene_ft.{npz,csv}`, `td_graphene_ft_m2.{npz,csv}`
- figures: `results/figures/graphene_kohn_anomaly.png`, `graphene_sc_convergence.png`, `graphene_td_dispersion.png`, `graphene_td_m2.png`, `graphene_anharmonicity.png`
