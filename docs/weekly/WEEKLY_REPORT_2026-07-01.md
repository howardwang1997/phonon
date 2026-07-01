# TD-Phonon / Kohn-Anomaly — Weekly Report

**Week ending 2026-07-01**  ·  branch `td-phonon-anomaly`  ·  machines: 2×V100 (FP64 DFT/EPW) + 1×RTX 2060 (MLIP/proxy); 8×H20 fleet offline this week.

> **One-line status.** The engine (foundation-MLIP failure atlas → material-specific FC-distillation → anharmonic Path-P) and the (E)/(L) origin-decomposition are **validated on both flagships (graphene + NbSe₂)** — the paper's **Part I (method)** is data-complete pending a convergence/ASR pass. This week we launched the **TMD-family FP64 campaign** on the two V100 boxes that builds the **Part II (discovery)** origin-map; **2 of 11 materials (NbSe₂, 2H-TaSe₂) are DFT-complete**, a third (NbS₂) is finishing, and the first family (E)-channel EPW is running.

---

## 0. Executive summary

This project builds a cheap-and-accurate phonon engine and uses it for a science payoff foundation MLIPs cannot reach: **decomposing lattice instabilities (Kohn anomalies, CDW soft modes) into their electronic (E) and lattice-anharmonic (L) origins**, across the 2D-TMD family. **One complete, high-level paper** merges the two halves into a single arc: **Part I (method)** = the engine + (E)/(L) on graphene & NbSe₂; **Part II (discovery)** = the TMD-family origin-map + a non-trivial CDW-origin finding.

**Banked this reporting period and before (all validated):**

| Result | System | Headline number | Status |
|---|---|---|---|
| Foundation-MLIP Kohn-anomaly failure is **model-specific** | graphene Γ-E₂g | MACE −22%, SevenNet −14%, MatterSim −2% vs DFT 1568 cm⁻¹ | ✅ |
| **Material-specific FC-distillation cures the cusp** | graphene Γ-E₂g | graphene-FT 1570 vs DFT 1568 (**101 % gap closed**); bulk-FT only 8 % | ✅ |
| Converged DFT reveals a **real K-A₁′ Kohn cusp** | graphene | 6×6 commensurate K-A₁′ = 1292 cm⁻¹, cusp kink 0.8→**14.4** | ✅ |
| **Distillation transfers a structural instability** (CDW soft mode) | NbSe₂ | foundation −0.20 → distilled-FT **−2.23 THz** ≈ DFT −2.18 (backbone-universal) | ✅ |
| NbSe₂ CDW is **EPC-driven, not nesting** (first family member adjudicated) | NbSe₂ | χ(q) ξ=0.53 at q_CDW (no peak) + EPW γ_qν 40–52 meV **broad** at q_CDW | ✅ |
| **Anharmonic Path-P** closes the thermal-force gap | graphene / NbSe₂ | 149→**21 meV/Å** (86 %); NbSe₂ **4.6×** better than harmonic-FT | ✅ |
| **Breadth-not-depth** generalization law | cross-material | transfer MAE flat ~1.80 THz to N=16, → 1.34 at N=64; floor ~1.3 THz | ✅ |

**In flight this week (live):** the two-lane V100 campaign (Fig 1) — GPU lane (DFT fc₂ → bands/χ(q) → Path-P) + CPU lane ((E)-channel DFPT-smearing → EPW γ_qν) across all 11 TMDs. **NbSe₂ and 2H-TaSe₂ are DFT-complete with soft modes captured (−2.18 THz); NbS₂ soft mode captured (−1.89 THz), Path-P + EPW running; 1T-TiSe₂ fc₂ running.**

![Fig 1 — campaign status matrix](figs/fig1_status_matrix.png)

*Fig 1. V100 FP64 campaign status (2026-07-01). Rows = the 11-member TMD family (★ starred = non-CDW / gapped controls); columns = the five per-material pipeline stages. fc₂ cells annotate the captured DFT soft-mode ω_min (THz) — negative = CDW instability seen at the DFT level, the ground truth the MLIP half is calibrated against.*

---

## 1. Framing — one complete, high-level paper (method → discovery)

**Strategy (2026-07-01): merged into a single flagship paper.** The plan is no longer two papers (an npj 保底 + an NCS 冲刺). It is **one complete, high-level paper** whose arc is *method → the discovery it enables*:

- **Part I — the method (engine).** Foundation MLIPs are blind precisely to the electronically-driven phonon anomalies (Kohn anomalies, CDW soft modes) that define quantum materials (blind-spot atlas → Hessian-supervision diagnosis). We repair it by co-designing a GPU finite-displacement DFT data engine (Line B) with material-specific FC-distillation + anharmonic Path-P into a foundation MLIP (Line A) — near-DFT accuracy on exactly the hardest cases, a breadth-not-depth generalization law, and a downstream κ payoff.
- **Part II — the discovery it enables.** Because the repaired framework is cheap *and* accurate on these hardest cases, high-throughput **origin-decomposition of lattice instabilities** becomes feasible for the first time: separate each CDW into its **electronic (Fermi-surface / EPC)** vs **lattice-anharmonic** channel, build the **(E)–(L) origin-classification map** across the 2D-TMD family, and adjudicate the long-debated origin of these CDWs (NbSe₂ = first member: EPC-driven, not nesting).
- **Why one paper, not two:** the method's novelty is *justified by* the discovery it enables (not "yet another phonon framework"); the discovery is *tractable only because of* the method (the cheap (L)-channel screens a whole family full DFPT/EPW could never afford). The two halves are one argument, not two packages.
- **Target:** a single flagship submission (Nature Computational Science / Nature Materials class) — the complete method + discovery.
- **Honest risk:** one flagship = higher variance and a longer runway than an npj — it needs *both* the method airtight *and* the discovery to land; there is **no npj保底 fallback**. (The method half remains independently strong — an implicit floor if repositioning were ever forced.)
- **Out of scope (explicit):** ballistic transport (NEGF/Landauer) — different method family, reads as scope creep.

> **Anti-drift note (per the roadmap's track-vs-drift rule).** This merge changes *packaging, not scope*: **every experiment in the NCS roadmap is retained** — the E1–E10 engine program *and* the family (E)/(L) origin-map + EPW + SSCHA + κ — now as sections of one paper. The long-term goal is unchanged: *origin-decomposition of lattice instabilities, made high-throughput by the engine*. Tracked against `docs/NCS_ROADMAP.md` §0; revisit before each compute campaign and log advance-vs-drift.

![Fig 2 — the (E)–(L) origin map](figs/fig2_origin_map.png)

*Fig 2. The paper's Part-II headline object: every CDW material placed in the **(E) electronic** (Fermi-surface nesting → momentum-dependent EPC) × **(L) lattice-anharmonic** (harmonic → quantum-stabilized) plane. Only **NbSe₂ is adjudicated so far** (EPC-driven, not nesting; anharmonically stabilized ~150 K ≈ exp 145 K); graphene is the (E)-channel calibrator. Filling this plane for NbS₂/TaS₂/TaSe₂/TiSe₂/VSe₂ is the paper's discovery.*

---

## 2. Completed results (the science banked)

### 2.1 Engine on flagship #1 — graphene Kohn anomaly + FC-distillation

Foundation MLIPs under-predict the zone-centre optical modes and **wash out the most non-analytic Kohn cusp** (Γ-E₂g), and the size/sign of the failure is *model-specific*. Distilling the material's own DFPT fc₂ recovers it; a converged commensurate DFT grid shows the K-A₁′ cusp is physically real.

| Quantity (graphene) | Foundation | Distilled (bulk) | Distilled (graphene) | DFT (this work) | Lit. |
|---|---|---|---|---|---|
| Γ-E₂g @ a=2.46 (cm⁻¹) | 1238 | 1265 | **1570** | 1568 | ~1600 |
| K-A₁′ @ a=2.46 (cm⁻¹) | 1113 | 1107 | **1368** | 1362 | ~1300 |
| Γ-cusp gap closed | — | 8 % | **101 %** | truth | — |
| cross-model Γ-E₂g (relaxed) | MACE 1254 / SevenNet 1382 / MatterSim 1563 | | | 1568 | ~1600 |
| converged K-A₁′ (6×6, commens.) | — | — | — | **1292, kink 14.4** | ~1300 |

![Fig 5 — graphene Kohn cure](figs/fig5_graphene_cure.png)

*Fig 5. (a) Γ-E₂g softening is model-specific (−22 % to −2 %); bulk-trained distillation barely helps (+8 %), but graphene-specific distillation lands on DFT (+101 %). (b) V-Q1: at the K-commensurate 6×6 grid the K-A₁′ cusp sharpens (kink 0.8→14.4) — a **real Kohn anomaly** the 5×5 grid under-resolved; foundation MLIPs instead over-soften K to ~1110 cm⁻¹ (a spurious "cusp").*

Force-fit quality: graphene DFT-distill force RMSE **18 meV/Å** (harmonic configs). Gate #1 (graphene fuse) = **PASS**.

### 2.2 Engine on flagship #2 — NbSe₂ CDW soft-mode capture

Every foundation/bulk-distilled MLIP reports NbSe₂ as dynamically **stable** (min freq ≈ 0, 0 imaginary modes) — the d-electron CDW soft mode is missed. Distilling NbSe₂'s **own** 3×3 DFT fc₂ transfers the instability into the MLIP, and this is backbone-universal.

| Quantity (NbSe₂, 3×3, a=3.44 no-relax) | Foundation | Distilled-FT | DFT (this work) |
|---|---|---|---|
| min freq — MACE (THz) | −0.20 | **−2.23** | −2.18 |
| min freq — SevenNet (THz) | −0.04 | **−1.99** | −2.18 |
| distill force RMSE (meV/Å) | — | 11.5 (MACE) / 12.7 (SevenNet) | — |
| geometry robustness | DFT vc-relax a=3.474 Å (+1 %) → soft mode −2.05 THz | | |

- **T-evolution (L-channel):** TDEP soft mode −0.48 THz (20 K) → 0 (≥200 K); **SSCHA free-energy Hessian has 0 imaginary modes at all T (20–400 K)** — the soft mode is fully quantum/anharmonically stabilized. Stabilization window ~150 K ≈ **exp monolayer T_CDW 145 K**.
- Gate #2 (CDW) = **PASS**.

### 2.3 The (E) electronic channel — demonstrated on both flagships (MLIP-blind)

The (E) channel is what foundation MLIPs structurally cannot see. Two independent DFT probes make it quantitative:

- **NbSe₂ (E1, frozen-phonon vs Fermi-Dirac T_el):** the CDW soft mode hardens monotonically through zero — −14.1 cm⁻¹ (474 K) → +129 (4737 K) → **electronic T_CDW ≈ 500–570 K**; rising T_el smears the Fermi surface and melts the CDW ⇒ EPC-driven.
- **graphene (E7 DFPT-smearing, 3×3 q):** both Kohn modes stiffen with T_el but **q-selectively** — K-A₁′ by Δ≈103 cm⁻¹ vs Γ-E₂g Δ≈40; the (E) channel is strongest where the bare anomaly is sharpest.
- **graphene EPW:** undoped λ = λ_tr ≈ 1×10⁻⁷ ≈ 0 (E_F at the Dirac point, no Fermi surface); doping to E_F=−0.94 eV turns on **λ = 0.96**, max γ_qν neV→**1.0 meV** — EPC is a pure Fermi-surface effect.
- **NbSe₂ EPW (E6):** soft-mode γ_qν = **40–52 meV, broad** around q_CDW≈(⅓,0) and the M–K boundary (not a sharp nesting spike); total λ **diverges (≈140, ill-defined)** at the soft mode → report q-resolved γ_qν only.

![Fig 8 — (E)-channel signatures](figs/fig8_echannel.png)

*Fig 8. (a) NbSe₂ CDW soft mode melts with electronic temperature (ω²=0 near 500–570 K). (b) graphene Kohn anomalies stiffen with T_el, K-A₁′ 2.6× more than Γ-E₂g. Both are frozen-phonon/DFPT effects that MLIPs miss entirely.*

**Nesting cross-check (Rigor #2):** the full Fermi-surface nesting function ξ(q) on a 36×36 grid does **not** peak at q_CDW (ξ=0.53 there vs 1.0 at Γ) → NbSe₂ CDW is **not nesting-driven** (Johannes–Mazin). Contrast graphene, where q*=2k_F is clean (M1.3, Dirac point at K).

### 2.4 Anharmonic Path-P — beyond harmonic distillation

Harmonic distillation is accurate only for small displacements; DFT-force labelling of thermally-sampled configs makes the model accurate where the simulation samples.

| Force RMSE vs DFT (meV/Å) | Foundation | Harmonic-FT | Path-P |
|---|---|---|---|
| graphene (150 cfg, 100–600 K) | 294 | 149 | **21** (86 % closed) |
| NbSe₂ (69 cfg, CDW coordinate) | 366 | **699** (backfires) | **151** (4.6× better) |

- NbSe₂ landscape (along CDW eigenvector): DFT edge +2047 meV (stiff), well −3 meV (marginally unstable); harmonic-FT edge +390 (5× too soft), well −19 meV (spurious); **Path-P reproduces DFT** (+2073, no real well). Re-SSCHA confirms stabilization at all T.

![Fig 6 — Path-P gap closure](figs/fig6_pathp.png)

*Fig 6. Path-P (anharmonic-distilled, green) closes the held-out thermal-force gap in both systems. Note NbSe₂ **harmonic**-FT (699) is worse than the foundation model (366) — a 0 K-distilled model actively misleads on the anharmonic CDW landscape; anharmonic labelling is required.*

### 2.5 Breadth-vs-depth transfer law + controls

![Fig 7 — breadth law](figs/fig7_breadth.png)

*Fig 7. Held-out transfer MAE is flat (~1.80 THz) up to N≈16 training materials, then falls to 1.40 (32) → 1.34 (64) — breadth of chemistry, not depth of sampling, drives generalization (knee ~16–32, floor ~1.3 THz). In-domain distilled MAE ≈ 0.10 THz with imaginary modes eliminated. Coverage-driven acquisition beats random by ≈0.18 THz at N=32; naive-uncertainty acquisition is the worst (chases pathological soft outliers).*

**Controls / negatives (locator does not false-positive):** MoS₂ (gapped) — cusp kink ≤ 2 at Γ/K for every backbone (vs graphene ~86–100); MoS₂ & graphene contrast confirms the anomaly detector flags only real e-ph physics. NbSe₂ 2k_F check: k_F=0.46 Γ-M ⇒ 2k_F=0.46 b₁ ≠ q_CDW=0.33 b₁ (qualified negative — correct answer).

---

## 3. In-progress this week (live)

### 3.1 V100 FP64 campaign (see Fig 1)

Two single-V100 boxes, two non-competing lanes each. Scope: full-family DFT fc₂ + bands/χ(q) (11 materials) + Path-P (7 CDW) + (E)-channel EPW γ_qν (7 CDW). Live positions (2026-07-01):

| Box | GPU lane (fc₂→bands→Path-P) | CPU lane ((E)-EPW) |
|---|---|---|
| **A** (v100-rental) | NbSe₂ ✅ → **NbS₂ Path-P (4/69)** → queued: 2H-TaS₂, 1T-TaS₂, 1T-TiS₂, MoS₂ | **NbS₂ EPW** — DFPT on last q-point (rep 5/9) |
| **B** (v100b) | 2H-TaSe₂ ✅ → **1T-TiSe₂ fc₂ (disp 2/3)** → queued: 1T-VSe₂, 1T-VS₂, WSe₂ | **2H-TaSe₂ EPW** DFPT started; queued 1T-TiSe₂, 1T-VSe₂ |

**DFT ground truth captured so far:** NbSe₂ −2.18 THz, 2H-TaSe₂ −2.175 THz, NbS₂ −1.893 THz (all show the CDW soft mode at the DFT level — the labels the H20 (L)-screen will be calibrated against).

### 3.2 NbSe₂ λ(T_el) — PRELIMINARY (not publication values yet)

![Fig 4 — NbSe2 lambda(T_el) preliminary](figs/fig4_lambda_Tel.png)

*Fig 4. NbSe₂ integrated EPW λ vs electronic temperature (degauss sweep). The qualitative physics is right — raising T_el hardens the soft mode and tames the λ divergence (140 → O(10–18)). **This is NOT a converged result:** (i) λ is inflated by the 1/ω² weight of the near-imaginary soft mode; (ii) the base point uses different EPW smearing (nsmear=1, degaussw=0.2) than d050/d060 (nsmear=4, 0.1) — not one ruler; (iii) nkf=24 is too coarse for the metallic Fermi surface. Convergence plan in §4.1.*

### 3.3 Infrastructure fixes this week

- **CPU oversubscription cured:** EPW lane ran `-np 8 × OMP=2` = 16 threads on 8 physical cores (load 21). Set OMP→1 across the EPW + GPU lanes (load 21→12 and falling; future materials run un-oversubscribed).
- **Silent throughput bug fixed:** the GPU/CPU lane loops used `cmd | while read MAT`, and the inner `mpirun/pw.x/epw.x` consumed the piped material list → each lane processed **only its first material** then declared "DONE" (11-material campaign would have produced 2). Fixed by redirecting inner commands `</dev/null`; **verified end-to-end** (Box B GPU lane correctly advanced 2H-TaSe₂ → 1T-TiSe₂ via the fix).
- **Self-healing relaunchers** installed so running (un-patchable in-memory) lanes are taken over by the patched version on exit, idempotently (skips finished materials via `EPW_DONE` markers + `JOB DONE` step-guards).
- ⚠️ *Open:* Box B briefly hit load 27 when two legacy EPW queues (`jconv`, `jlq6`) overlapped the new 2H-TaSe₂ DFPT — transient oversubscription to watch/serialize.

> **Note (not yet in git):** the three lane-script fixes are deployed to the boxes by scp, not committed. GitHub `td-phonon-anomaly` still carries the buggy versions — a re-clone would reintroduce both bugs. Recommend committing (separate follow-up).

---

## 4. Experiments still needed

### 4.1 Near-term — close the current campaign & lock Part I (no rental)

| Task | What it produces | Est. compute | Data needed |
|---|---|---|---|
| Finish family DFT fc₂ + bands/χ(q) (9 remaining) | ground-truth soft modes + nesting per material | ~15–45 box-h (GPU lane, ~free) | Ta/Ti/V/S ONCV pseudos (present) |
| Finish Path-P (6 remaining CDW) | anharmonic (L) labels | ~5–8 box-h (GPU lane) | fc₂ soft eigenvectors |
| Finish (E)-EPW γ_qν (5 remaining CDW) | q-resolved EPC per material | ~40–90 box-h (CPU lane, bottleneck) | dvscf + Wannier per material |
| **λ(T_el) / λ_q convergence (NbSe₂ first, then flagships)** | defensible λ: **nkf 24→48(→60)** convergence + unified smearing + ≥5 T_el points | ~1 day (nkf, reuse DFPT) + ~½ day/new-T_el point | — |
| **ASR re-pass** (q2r `zasr='crystal'`) on graphene/NbSe₂ | clean absolute λ + correct 2D ZA acoustic mode | ~20–40 box-h | conda-install full QE (q2r/matdyn) |
| Convergence sweeps (k/q, smearing, supercell) on flagships | seed error bars on every Part-I headline number | included above | — |

*Expected outcome:* the family DFT truth table + a converged NbSe₂ λ(T_el)/γ_q, bringing Part I (the method) to a submittable state.

### 4.2 Part II — the discovery half (from the NCS roadmap)

| Roadmap step | Expected result | Compute (hardware) | Data |
|---|---|---|---|
| **E-2a** family (L)-triage — foundation-MLIP SSCHA, all 11 | rank likely-unstable; catch what MLIP mis-calls | ~50–100 GPU-h (8×H20) / 2060 | MLIP only, no DFT |
| **E-2b** fc₂ + material-specific distill + (L)-SSCHA on candidates | quantum-stabilization T per material (L-channel of the map) | fc₂ in §4.1; SSCHA 8×90 min | family fc₂ labels |
| **E-2c** (E)-channel DFPT-smearing + EPW on 3–5 flagships | γ_qν breadth / nesting verdict (E-channel of the map) | in §4.1 EPW budget | dvscf + Wannier |
| **E-2d** build (E)–(L) origin map + hunt the discovery | **a material re-classified, or a predicted instability, or a clean (E)/(L)→T_CDW trend vs experiment** | synthesis | exp T_CDW / INS (literature) |
| ★ **the discovery** (integral, not optional) | origin-map + ≥1 non-trivial result = the paper's Part II; then rent for the scale-up | — | exp T_CDW/INS |

**H20 MLIP half (offline this week — resume when fleet returns):** E1 benchmark atlas (≥6 foundation MLIPs × ~1,500-material stratified MDR; 72 jobs), E3/E4 distillation + anti-forgetting/LoRA/breadth ablations (7 finetunes; canon = b64/pt1000), (L)-SSCHA family screen (8 SSCHA, supercell 4×4×1, T={50,100,150,200,300}), E9 κ (phono3py, 5 covalent references) → **~900–1,800 GPU-h ≈ 5–10 parallel-days**.

**Phase 3 — scale-up (rental FP64, after the discovery lands):** family-wide converged EPW (dense grids) + κ-at-scale 3rd-order + GPU-DFT ~50× engine demo + 10³–10⁴ active-learning dataset → **~2,100–5,800 GPU-h** on rented A100/V100.

---

## 5. The three planning dimensions

### 5.1 Expected results (per experiment program)

| Program | Predicted result (the claim it lands) |
|---|---|
| E1 benchmark atlas | universal MLIPs are *not ready* for phonons — softening, spurious imaginary modes, ASR violation; failure correlates with **curvature (Hessian)**, not energy error |
| E2 diagnosis | it is a **Hessian-supervision gap**, not capacity — force-MAE and phonon-MAE decouple; conservative > direct-force models |
| E3/E4 distillation | in-domain phonon MAE **~0.10 THz**, imaginary modes eliminated, at **zero new DFT**; **breadth-not-depth** law (knee ~16–32, floor ~1.3 THz) |
| E5 GPU-DFT engine | **~50× workflow-level** speedup (SCF ~5–15× × density-reuse ~2–4× × non-diagonal supercells) — each factor reported separately |
| E7/E8 active learning | reach target accuracy with **N× less DFT** than random; coverage-driven acquisition wins, naive uncertainty loses |
| E9 downstream κ/zT | near-DFT **κ** and stability screening at ~10³× speed |
| **(E)/(L) origin-map (Part II — the discovery)** | each TMD placed in the (E)–(L) plane; **≥1 non-trivial reclassification/prediction** = the paper's headline discovery |

### 5.2 Compute required

![Fig 3 — compute budget](figs/fig3_compute_budget.png)

*Fig 3. Budget by phase and hardware class. Machines in hand (2×V100 now + 8×H20) carry the complete paper to a core-complete draft (Part I + the Part II origin-map); only the FP64 scale-up needs rental, and only after the discovery lands.*

| Bucket | Workload | Estimate | Hardware | Unit note |
|---|---|---|---|---|
| V100 now — DFT fc₂ + anchors | family harmonic truth | ~30–45 | 2×V100 GPU lane | box-h (rides idle GPU ~free) |
| V100 now — Path-P (7 CDW) | anharmonic (L) labels | ~5–8 | 2×V100 GPU lane | box-h |
| V100 now — (E)-EPW (6 new) | q-resolved EPC | ~40–90 | 2×V100 CPU lane | box-h (bottleneck) |
| V100 now — λ(T_el)+ASR+conv | clean Part-I numbers | ~20–40 | 2×V100 CPU lane | box-h |
| **V100 now subtotal** | | **~110–210 box-h ≈ 1–2 wk** | 2×V100 | — |
| H20 — MLIP half (E1/E3/E4/(L)-SSCHA/κ) | benchmark + distill + screen | **~900–1,800** | 8×H20 | GPU-h (≈5–10 parallel-days) |
| **Rental — FP64 scale (gated)** | dense EPW + κ-at-scale + engine demo + dataset | **~2,100–5,800** | rented A100/V100 | GPU-h (only if gate passes) |

*Measured anchors:* NbSe₂-class DFPT ≈ 6 h + EPW ≈ 2.5 h = **~8.5 box-h/material**; TMD fc₂ ≈ 1–3 h; SSCHA ≈ 15–30 min/(material·T); MLIP fine-tune ≈ 30–60 min; MLIP phonon inference ≈ ~1 min/material. Two biggest unknowns: MLIP-inference speed vs material size, and converged-EPW grid cost (**could be 2–3× higher**).

### 5.3 Data required

| Need | Source | Public? | Use | Status |
|---|---|---|---|---|
| Harmonic DFPT phonons (**10,034**) | MDR/PhononDB (Togo/NIMS) | ✅ | E1 benchmark + E3 distillation labels | index committed; cache regenerable |
| Harmonic DFPT (**1,521**) | Petretto 2018 | ✅ | second reference (E1/E4) | public |
| Lattice κ + 3rd-order FC | Togo phono3py DB | ✅ | E9 κ reference | public |
| Pre-training trajectories | MPtrj (`mp_traj_combinedxyz`, **~56 MB**) | ✅ | replay anti-forgetting (E3) | gitignored; manual relay to H20 |
| Foundation pseudopotentials | **SG15 ONCV-PBE (69 elem)** | ✅ | self-DFT across the periodic table | blocker solved |
| **TMD-family pseudos: Ta, Ti, V, S** | PseudoDojo ONCV-PBE | ✅ | family DFT — *gating* | present on both boxes ✅ |
| Experimental T_CDW / INS (TMD family) | literature | ✅ | Part-II validation | to compile |
| graphene + NbSe₂ DFT fc₂/DFPT/dvscf/EPW | self (2×V100) | ❌ | Part-I flagships | **done** |
| TMD-family fc₂ + DFPT + dvscf + EPW | self (2×V100) | ❌ | Part-II origin-map | **in progress (2/11)** |
| `q2r.x` / `matdyn.x` (crystal-ASR) | conda full-QE | ✅ | clean absolute λ + 2D ZA mode | to install on boxes |

**Enablers still to do (low-cost, now):** conda-install full QE (q2r/matdyn) on Box A/B; compile the literature T_CDW / INS table for the family.

---

## 6. Risks, blockers, next-week plan

**Risks / blockers**
- **EPW is the throughput bottleneck** (CPU-bound, ~8.5 box-h/material, 5 more (E)-channel materials queued) — the campaign is a ~1–2 week job on 2 boxes.
- **λ diverges at soft modes** (physics, not a bug): report q-resolved γ_qν/λ_qν, never absolute λ, for CDW materials.
- **8×H20 fleet offline 7 days** → the MLIP half (E1/E3/E4/(L)-SSCHA/κ) is paused; the Part-I MLIP tables and the family (L)-screen wait on its return.
- **Fixes not in git** (§3.3); Box B transient oversubscription **resolved** this session (legacy jconv/jlq6 queues paused + serialized behind the campaign).
- **The discovery is the completeness bar** — not compute: the single paper is not submittable until the origin-map yields ≥1 non-trivial result. **Single-flagship = higher variance** (no npj保底 fallback; the method half stays independently strong as a floor).

**Next week**
1. Let the family DFT/EPW campaign run (self-healing); land NbS₂ + 2H-TaSe₂ (E)-channel EPW as the first two family γ_qν points.
2. Run the NbSe₂ λ(T_el) **convergence** (nkf 24→48, unified smearing) → upgrade Fig 4 from preliminary to defensible.
3. Commit the lane fixes to `td-phonon-anomaly`; serialize the Box B EPW queues.
4. On H20 return: kick E1 + E3/E4 + (L)-SSCHA triage for the family map.

---

## Appendix A — figure & table index

| # | Figure | Data basis |
|---|---|---|
| Fig 1 | Campaign status matrix | live campaign logs (2026-07-01) |
| Fig 2 | (E)–(L) origin-classification map | Rigor #2, E6, E1, SSCHA (NbSe₂ adjudicated) |
| Fig 3 | Compute budget by phase/hardware | NCS_ROADMAP §0b |
| Fig 4 | NbSe₂ λ(T_el) — **preliminary** | box-B epw.out (base/d050/d060) |
| Fig 5 | Graphene Kohn cure + converged-K cusp | M1.1b, V-Q1, cross-model |
| Fig 6 | Path-P anharmonic gap closure | Path-P graphene production + NbSe₂ |
| Fig 7 | Breadth-not-depth transfer law | E3/E4 |
| Fig 8 | (E)-channel signatures (both flagships) | E1 (NbSe₂), E7 (graphene) |

Tables: §0 banked results · §2.1–2.4 per-flagship number tables · §4.1–4.2 remaining/roadmap experiments · §5.1 expected results · §5.2 compute · §5.3 data.

## Appendix B — honest-framing clauses (for the paper)

- Pure GPU-DFT single-SCF speedup is only ~5–15×; "~50×" is **workflow-level**; "~10³×" is the **MLIP proxy** — every speedup quoted with its scope.
- Every accuracy headline carries **seed error bars**; κ reported as **mean + median**; negative results kept (naive-uncertainty acquisition; NbSe₂ non-nesting; harmonic-FT backfire).
- Absolute EPW λ is **ill-defined at a soft mode** — the divergence *is* the instability signal; q-resolved γ_qν is the reportable quantity.
- The current NbSe₂ λ(T_el) (Fig 4) is **explicitly preliminary** pending the §4.1 convergence pass.

---

*Generated 2026-07-01. Figures reproducible via `conda run -n phonon python docs/weekly/plot_weekly.py` (data embedded, no external files). Numbers sourced from `docs/TD_PHONON_M1_RESULTS.md`, `docs/NCS_ROADMAP.md`, `configs/{v100,h20}_campaign.yaml`, and the live campaign logs.*
