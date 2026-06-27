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
  whether *any* distillation recovers the Γ cusp. **DONE — see below (PASS).**
- **M1.2** (hiPhive-TDEP) — proceed to the (L)-channel ω(q,T); the K-anomaly is a
  good handle to track q*(T).

## M1.1b — graphene-specific DFT distillation: the Γ-cusp cure  ✅
The decisive follow-up to M1.1's open question (*does any distillation recover the
washed-out Γ-E₂g cusp?*). Full chain on a **rented 1×V100** (Quantum ESPRESSO + MACE,
2026-06-25):

1. **DFT fc₂** (`m1_1b_graphene_dft.py`): graphene finite-displacement fc₂ via QE
   (ONCV-PBE, 5×5×1 = 50-atom supercell, ecutwfc 60 Ry, 6×6 supercell k-mesh,
   conv_thr 1e-9; ~32 min on 8 CPU cores) → **Γ-E₂g = 1568, K-A₁′ = 1362 cm⁻¹**
   (lit ~1600/~1300), ZA≈0 at Γ. The graphene-specific distillation target.
2. **Distillation data** (`m1_1b_make_graphene_data.py`): 153 harmonic-labelled
   rattled supercells (F=−Φu, E=½uΦu) from that DFT fc₂.
3. **Fine-tune** (`finetune_graphene.sh`): *same single-head recipe as the bulk FT*
   (`small` foundation, energy_weight 0.01 / forces_weight 100, 60 epochs) →
   force RMSE **18 meV/Å** → `ft_graphene.model`.
4. **Eval** (`harmonic_dispersion_2d.py` + `plot_m1_1b_compare.py`).

| model | Γ-E₂g (cm⁻¹) | K-A₁′ (cm⁻¹) |
|---|---|---|
| foundation MACE | 1254 | 1151 |
| FC-distilled (bulk) | 1399 | 1192 |
| **FC-distilled (graphene)** | **1556** | **1355** |
| DFT (QE, this work) | 1568 | 1362 |
| literature | ~1600 | ~1300 |

**Result — the Γ cusp comes back.** Of the foundation→DFT softening gap at Γ-E₂g,
the **bulk** FC-distilled model closed **46%**; the **graphene-specific** model
closed **96%** (1254→1556, DFT 1568) — essentially matching DFT, and K-A₁′ likewise
(1355 vs 1362). In `m1_1b_graphene_compare.png` the graphene-FT optical branches sit
on top of the DFT dashed curve.

**Interpretation.** M1.1's "Γ cusp washed out *even after distillation*" was a
limitation of the **bulk-trained** distillation, **not** of FC-distillation itself:
targeting the material's own DFPT fc₂ recovers the cusp (the §2.2 cure works when
material-specific). This **resolves Gate #1's open branch — PASS**.

> **Caveats.** (i) harmonic / 0 K cure only; the anharmonic (L) and electronic (E)
> channels are unchanged (→ Path P, M1.3). (ii) graphene-FT relaxed to a=2.497 Å
> (vs DFT 2.46) because the distillation weights forces ≫ energy — the *curvature*
> (hence the cusp) is right even though equilibrium a drifts; tightenable with more
> energy weight or fixed-cell distillation.

### Fixed-geometry refinement (apples-to-apples at a=2.46, no relax)
Re-evaluating every model at the *same* DFT geometry removes the lattice confound and
sharpens the verdict (`harmonic_dispersion_2d.py --a 2.46 --no-relax`;
`m1_1b_fixed_geom_compare.png`):

| at a=2.46 | Γ-E₂g | K-A₁′ | kink_Γ | kink_K |
|---|---|---|---|---|
| foundation (small) | 1238 | 1113 | 10.9 | 85.7 |
| FC-distilled (bulk) | 1265 | 1107 | 8.9 | 87.4 |
| **FC-distilled (graphene)** | **1570** | **1368** | **7.0** | **0.9** |
| DFT (QE) | 1568 | 1362 | 7.0 | 0.8 |

- **graphene-FT reproduces DFT to ~1% at *both* Γ and K** and matches DFT's cusp shape
  (kink_Γ 7.0=7.0, kink_K 0.9≈0.8) — the green curve overlies the DFT dashed curve
  across the whole M-Γ-K-M path. Γ gap closed = **101%**.
- **bulk-FT's Γ recovery collapses to 8%** at the true geometry — the relaxed-eval 46%
  was a lattice-contraction artifact (bulk-FT relaxes to a=2.42, which spuriously
  stiffens its phonons). So bulk distillation essentially does **nothing** for graphene's Γ.
- **Revises the M1.1/M1.2 "K cusp survives in the MLIPs" reading:** the large
  foundation/bulk K kink (~86) is a *softening artifact* — their top branch dips to
  ~1110 at K — **not** a faithful Kohn anomaly; DFT and graphene-FT keep K high (~1365)
  and smooth. *Caveat (now resolved — see V-Q1):* the 5×5 DFT **did** under-resolve a true
  K-A₁′ Kohn dip — the K-commensurate 6×6 DFT gives K=1292 cm⁻¹ with a sharp cusp (kink 14.4),
  so converged DFT is *not* smooth at K; the foundation MLIP still over-softens it (~1110).

## M1.3 — (E)-channel: Dirac point + q* ≈ 2k_F  ✅
DFT electronic bands of graphene (QE, 18×18 SCF + M-Γ-K-M bands;
`m1_3_graphene_bands.py`, ~17 min CPU) confirm the **Dirac point at K** (|E−E_F| = 20
meV ≈ 0, limited by the discrete k-path) and pin the Kohn-anomaly geometry:
**Γ-E₂g = intra-valley (q→0); K-A₁′ = inter-valley (q=K connects K↔K′)** — both are the
2k_F connectors of the Dirac points. So the phonon anomalies are **electron–phonon in
origin** (the (E) channel). The part the MLIP structurally cannot make — the
*electronic-temperature* broadening of the anomaly (Fermi smearing) — needs a
finite-T_el DFPT scan, deferred (heavier DFT). Figure `m1_3_graphene_ebands.png`.

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

## Path P (one-shot) — anharmonic distillation, proof-of-concept  ✅
Tests whether **thermal-config DFT-force distillation** fixes the large-displacement
(anharmonic) PES that the harmonic M1.1b distillation leaves wrong. On the V100
(QE + MACE, 3×3 graphene, 300 K):
1. graphene-FT (M1.1b) runs Langevin MD → 40 thermal snapshots (`path_p_make_data.py`);
2. DFT single-point (QE) each → REF forces (~2 min/config, ~81 min total);
3. fine-tune from foundation on the thermal DFT forces, **force-only** (energy_weight 0
   — DFT total energies are huge absolute values that otherwise swamp the loss; force
   RMSE *diverged* at energy_weight 0.01) → `ft_path_p.model`;
4. eval: force RMSE vs DFT on 10 held-out thermal configs (`path_p_eval.py`).

| model | force RMSE vs DFT (thermal configs) |
|---|---|
| foundation MACE | 280 meV/Å |
| FC-distilled harmonic (M1.1b) | 138 meV/Å |
| **Path-P (anharmonic)** | **17 meV/Å** |

(DFT force magnitude ⟨\|F\|⟩_rms = 1183 meV/Å.)

**Path P closes the anharmonic gap 138→17 meV/Å (87% reduction)** — from ~12% to ~1.4%
of the force magnitude. Confirms the principle: the harmonic distillation (M1.1b) is
accurate only for small displacements; labelling the *thermally-sampled* configurations
with DFT forces makes the model accurate where the simulation actually samples.

> **Scope.** Proof-of-concept: small 3×3 cell, single T, 30 train configs, force-only.
> A production Path P wants a larger cell, multiple T, an isolated-atom E0 reference, and
> downstream validation (ω(q,T) / linewidths). CPU-QE made this feasible (~2 min/config);
> scaling to 100s of configs wants GPU-QE.

### Path P PRODUCTION — multi-T, 150 configs on GPU-QE  ✅
With GPU-QE built (~19× faster DFT; see `docs/GPU_QE_BUILD.md`), scaled Path P to **3
temperatures (100/300/600 K) × 50 snapshots = 150 DFT-labelled configs** (train 97 / val
16 / **held-out test 37**, spanning all T). Force-only fine-tune (energy_weight 0) from
the MACE-small foundation. The anharmonic gap *grows with T* (harmonic-FT force error 92
meV/Å at 300 K → 180 at 600 K; mean 128).

| model | force RMSE vs DFT (held-out, 100–600 K) |
|---|---|
| foundation (MACE-small) | 294 meV/Å |
| harmonic-FT (M1.1b, 0 K-distilled) | 149 meV/Å |
| **Path-P production** | **21 meV/Å** |

**Gap closed 149 → 21 meV/Å (86%)**, now across the full 100–600 K range and on a *held-out*
test set (vs the one-shot's 138→17 at 300 K only). The 21 (vs 17) is *because* the test
includes 600 K configs where anharmonicity is strongest — so this is the stronger result:
**one model accurate across the whole thermal range**, not just near 300 K. The DFT
labelling (150 single-points) took ~40 min on the one V100 GPU — would have been ~5 h on
CPU. Data `data/path_p/{train,val,test}.xyz` + `split.json`; model `ft_path_p.model`
(gitignored). The original GPU-QE motivation (scale Path P) is realised.

## MoS₂ negative control — the locator does NOT false-positive  ✅
A gapped semiconductor has no Fermi surface, so the locator should find *nothing*.
MoS₂ monolayer dispersion (5×5×1, M-Γ-K-M) + the anomaly locator, on the 2060:

| top-branch kink \|dv\| | Γ | K |
|---|---|---|
| MoS₂ foundation MACE | 2.1 | 0.3 |
| MoS₂ FC-distilled | 0.7 | 0.3 |
| *graphene (contrast)* | ~11 | **~86–100** |

MoS₂ shows **no cusp** (kink ≤ 2 at Γ/K) vs graphene's strong K-A₁′ cusp (~86–100) —
confirming the locator flags anomalies only where the e-ph physics actually produces
them, not numerical artifacts. (MoS₂ optical modes: foundation ~348, FT ~368–381 cm⁻¹,
both softened vs exp ~400 — the §2.1 softening is present here too, but cusp-free.)

## Cross-model (3 backbones × 3 systems) — a more nuanced picture  ✅
Re-ran the harmonic dispersion with **SevenNet** (7net-0) and **MatterSim** (v1.0.0-5M)
in isolated envs on the 2060 (CPU), alongside MACE, on all three monolayers.

**Graphene Γ-E₂g — softening is *model-dependent*, NOT universal:**

| model | Γ-E₂g (cm⁻¹) | vs lit ~1600 | kink_Γ | kink_K |
|---|---|---|---|---|
| MACE foundation | 1254 | **−22%** | ~11 | ~86 |
| SevenNet (7net-0) | 1382 | **−14%** | 24 | 0.3 |
| **MatterSim (5M)** | **1563** | **−2%** | 8 | 1.5 |
| MACE-FT graphene (M1.1b) | 1556 | −3% | 7 | 0.9 |
| DFT (QE) | 1568 | −2% | 7 | 0.8 |

> **Correction to the earlier "softening is universal" reading:** with all three backbones
> the severity clearly varies — MACE softens hardest (−22%), SevenNet middle (−14%), and
> **MatterSim barely softens graphene's E₂g at all (−2%, essentially DFT)**. So the Γ-E₂g
> softening is **model-specific**; §2.1's −10.7% MatterSim figure is a *median over bulk
> materials* — graphene is a case where MatterSim happens to be excellent. Also: only MACE
> shows the spurious dipped K (kink ~86); SevenNet/MatterSim/DFT all keep K smooth (kink
> ≤1.5) — confirming the MACE "K cusp" is a softening artifact, model-specifically.

**MoS₂ negative control — universal (all 3 cusp-free):** kink_Γ/K ≤ 3 for every backbone
(MACE, SevenNet 2.2/1.3, MatterSim 2.7/0.0) — the locator never false-positives. Optical
mode: MACE 368 (softened), SevenNet 431, MatterSim 458 (vs exp ~400) — again model-specific.

**NbSe₂ soft mode — the MISS is universal (all 3 stable, 0 imaginary):** MACE −0.013,
SevenNet −0.00, MatterSim −0.00 THz; n_imaginary = 0 for all. **Every backbone misses the
CDW soft mode** → Gate #2 is robust across models: **M3 needs NbSe₂-specific DFT
distillation regardless of the MLIP backbone.**

## NbSe₂ MLIP preview — Gate #2: the soft mode is missed  ⚠️
Monolayer NbSe₂ (metallic, strong Kohn anomaly → CDW) harmonic dispersion with the
foundation and general-FC-distilled MLIP (6×6×1, Γ-M-K-Γ; 2060, MLIP-only):

| | min freq (THz) | n_imaginary (<−0.1 THz) | soft mode? |
|---|---|---|---|
| NbSe₂ foundation | −0.013 | 0 | none |
| NbSe₂ FC-distilled | −0.005 | 0 | none |

**Both MLIPs show NbSe₂ as dynamically stable** — no soft/imaginary mode along Γ-M,
where real monolayer NbSe₂ has a CDW instability. The d-electron-driven soft mode is
**missed** by the bulk-trained MLIPs (plan risk R4). **Gate #2 preview = the soft mode
needs DFT distillation**: M3 (NbSe₂) cannot ride cheap MLIP compute; it requires NbSe₂'s
own DFPT fc₂/fc₃ as a distillation target (FP64) — the heavy path the plan flagged. (Top
optical ~210–230 cm⁻¹, also softened.) *Cheap MLIP check that de-risks the M3 budget.*
**→ RESOLVED in V-Q3 (Gate #2 PASS):** the NbSe₂-specific DFT fc₂ *does* capture the CDW soft
mode and distilling it recovers the mode in the MLIP (foundation −0.2 THz → distilled-FT −2.23,
matching DFT −2.18). See the V-Q DFT-queue section below.

## V-Q DFT queue (rented 1×V100, GPU-QE, 2026-06-27) — converged-K, (E)-channel, NbSe₂ Gate #2
Three follow-up DFT experiments, run as an autonomous single-card queue on the V100
(GPU-QE pw.x, finite-displacement). They close the three open questions left by M1.1b /
M1.3 / the NbSe₂ preview. Graphene at fixed a=2.46; NbSe₂ at literature a=3.44.

### V-Q1 — converged-K graphene: the K-A₁′ Kohn anomaly is **real**  ✅
M1.1b's 5×5 DFT made K look high and smooth (1362 cm⁻¹, kink 0.8) — but K=(⅓,⅓) is **not
commensurate** with a 5×5 supercell, so that value is *interpolated*. Re-running at larger
supercells (GPU-QE finite-disp, matched electronic k-density) resolves it:

| supercell | Γ-E₂g (cm⁻¹) | K-A₁′ (cm⁻¹) | kink_Γ | kink_K | K sampled |
|---|---|---|---|---|---|
| 5×5 (M1.1b) | 1568 | 1362 | 6.96 | 0.84 | interpolated |
| **6×6 (V-Q1b)** | 1571 | **1292** | 9.29 | **14.41** | **direct (commensurate)** |
| 7×7 (V-Q1) | 1570 | 1324 | 11.41 | 8.30 | interpolated |

**Finding.** Γ-E₂g is fully converged (1568→1571, ≤0.2%). But at the **K-commensurate 6×6**,
where K is sampled directly, **K-A₁′ = 1292 cm⁻¹ (right on lit ~1300) with a strong cusp
(kink 14.4)** — the 5×5 under-resolved a genuine Kohn anomaly into a smooth-looking 1362.
The 7×7 (also interpolated) gives 1324/8.3, consistent. **This revises the M1.1b/M1.2
reading**: the converged DFT K-A₁′ is *not* smooth — it has a real, sharp cusp at ~1292.
So the truth is intermediate: DFT has a genuine K Kohn cusp (kink ~14 resolved), the foundation
MACE grossly *over*-softens it (dips to ~1110, kink ~86), and bulk-FC-distillation sits between.
9×9 was attempted but needs **45 GB > V100's 32 GB** (hard VRAM wall) → 6×6 is the converged
commensurate point. Files `results/vq1/disp_graphene_sc{6_k6,7_k5}.npz`.

### V-Q2 — (E)-channel: electronic-temperature scan of the anomaly  ✅
Frozen-phonon Γ/K at fixed 5×5 geometry with **Fermi-Dirac smearing** (degauss = k_B·T_el),
sweeping the electronic temperature — a GPU-affordable proxy for the (E)-channel that the
MLIP structurally cannot see (heavy DFPT-with-smearing was the alternative):

| T_el (K) | degauss (Ry) | Γ-E₂g | K-A₁′ | **kink_Γ** | kink_K |
|---|---|---|---|---|---|
| 789 | 0.005 | 1559 | 1362 | **8.18** | 0.75 |
| 1579 | 0.010 | 1568 | 1361 | **6.99** | 0.91 |
| 3158 | 0.020 | 1567 | 1356 | **6.71** | 1.22 |
| 6315 | 0.040 | 1553 | 1361 | **5.81** | 1.25 |

**Finding.** The mode *frequencies* are electronic-T-insensitive (≲1%, within k-noise), but the
**Γ-E₂g cusp sharpness decreases monotonically (kink 8.2→5.8) as electronic T rises 790→6300 K**
— the frozen-phonon fingerprint of Fermi-surface smearing **broadening the Kohn anomaly**, i.e.
the (E) channel acts on the *cusp lineshape*, not the zone-centre frequency. (K's kink is
noise-level here because 5×5 under-resolves K, per V-Q1.) Honest scope: frozen-phonon at
commensurate q captures the *trend* in cusp sharpness; the full anomaly linewidth still needs
DFPT linear response. Files `results/vq2/disp_graphene_dg*.npz`.

### V-Q3 — NbSe₂ CDW **Gate #2: distillation recovers the soft mode**  ✅✅ (headline)
The decisive metallic-CDW test. (1) DFT fc₂ of monolayer NbSe₂ on a **3×3×1 supercell** (hosts
the 3×3 CDW), GPU-QE, ONCV-PBE Nb/Se, dense k, degauss 0.015. (2) Distil 133 harmonic configs
from that fc₂ → fine-tune MACE-small (force RMSE 11.5 meV/Å). (3) Re-evaluate the MLIP dispersion
at the **same geometry (a=3.44, no-relax) and supercell** as the DFT — foundation vs distilled-FT:

| | min freq, 3×3 (THz) | min freq, 6×6 (THz) | top optical Γ/K (cm⁻¹) |
|---|---|---|---|
| **DFT (this work, 3×3)** | **−2.18** (232 imag.) | — | — |
| foundation MACE-small | **−0.20** | −0.33 | 249 / 236 |
| **distilled-FT (NbSe₂ DFT fc₂)** | **−2.23** | −3.22 | 294 / 268 |

**Finding — Gate #2 PASS.** DFT **captures the CDW soft mode** (−2.18 THz, strongly imaginary)
that every foundation/general MLIP backbone misses. The foundation MACE at the *same* geometry
is essentially stable (−0.2 THz — no CDW). **Distilling the NbSe₂-specific DFT fc₂ transfers the
instability into the MLIP**: the FT min freq is −2.23 THz at 3×3, **matching the DFT −2.18**.
So material-specific DFT distillation recovers even a *structural instability* (a metallic CDW
soft mode) — the M1.1b graphene-cusp cure repeated for the much harder d-electron CDW case, and
the resolution of the NbSe₂ preview's Gate #2. **Honest caveats**: (i) literature geometry
a=3.44 (not DFT-relaxed) — the absolute −2.2 THz magnitude is geometry-sensitive, but the
*same-geometry* foundation(−0.2)→FT(−2.2) contrast and FT≈DFT agreement are robust; (ii) the FT
faithfully reproduces its fc₂ training target (as expected) — the non-trivial part is that an
unstable/soft fc₂ *can* be distilled into a foundation MLIP that otherwise enforces stability;
(iii) full CDW physics (incommensuration, fc₃, electronic-T) is beyond this harmonic 3×3 pass.
Scripts `scripts/{vq3_nbse2_dft,vq3_downstream.sh,vq3_eval.sh}`; data `results/vq3/*`; model
`results/finetune_nbse2/ft_nbse2.model` (gitignored, on the V100).

## Artifacts
- code: `scripts/{td_common,td_structures,harmonic_dispersion_2d,anomaly_locate,graphene_sc_convergence,td_phonon,td_anharmonic,plot_graphene_anomaly,plot_sc_convergence,plot_td_dispersion,plot_anharm_diag}.py`
- V-Q code: `scripts/{m1_1b_graphene_dft (now --degauss/--smearing),vq3_nbse2_dft}.py`, `scripts/{vq_queue,vq3_downstream,vq3_eval}.sh`
- M1.3 / Path-P code: `scripts/{m1_3_graphene_bands,path_p_make_data,path_p_eval}.py`
- M1.1b code: `scripts/{m1_1b_graphene_dft,m1_1b_make_graphene_data,plot_m1_1b_compare}.py`, `scripts/finetune_graphene.sh`
- data: `data/td_phonon/{graphene,mos2,nbse2}.xyz`, `results/td_phonon/disp_graphene_{base,ft}.npz`, `graphene_sc_convergence.csv`, `td_graphene_ft.{npz,csv}`, `td_graphene_ft_m2.{npz,csv}`
- M1.1b data: `results/m1_1b/dft/{graphene_dft_phonopy.yaml,disp_graphene_dft.npz}`, `results/m1_1b/disp_graphene_ftgraphene.npz`, `results/m1_1b/fixed/disp_graphene_{base,ftbulk,ftg}_a246.npz` (model `ft_graphene.model` gitignored, on the V100)
- figures: `results/figures/{graphene_kohn_anomaly,graphene_sc_convergence,graphene_td_dispersion,graphene_td_m2,graphene_anharmonicity,m1_1b_graphene_compare,m1_1b_fixed_geom_compare}.png`
- V-Q data + figures: `results/vq1/disp_graphene_sc{6_k6,7_k5}.npz`, `results/vq2/disp_graphene_dg*.npz`, `results/vq3/{nbse2_dft_phonopy.yaml,disp_nbse2_{dft,base,ft}_sc*.npz}`; plotters `scripts/plot_{nbse2_gate2,vq1_convergence}.py`; figures `results/figures/{nbse2_gate2,vq1_converged_K}.png`
