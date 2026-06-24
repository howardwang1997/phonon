# TD-phonon / Kohn-anomaly project: detailed execution plan (goal · data · method · expected result · compute · risk)

This is the **executable refinement** of [`TD_PHONON_ANOMALY_PLAN.en.md`](./TD_PHONON_ANOMALY_PLAN.en.md)
(the high-level plan). Every step carries the same six fields: **goal · experiment data · methodology ·
expected result · compute (amount + type) · risk**. A **risk register** and **Go/No-Go gates** close it out.

## Project goal & overall expected outcome

- **Overall goal:** use an FC-distilled MLIP, on **cheap compute**, to obtain **temperature-dependent phonon
  dispersions ω(q,T)** of 2D materials, locate the **slope-discontinuities / kinks ("jumps," i.e. Kohn
  anomalies)** in the dispersion, track how they move with temperature, and test their link to electron–phonon
  coupling (q\* ≈ 2k_F Fermi-surface nesting) / ballistic transport.
- **Overall expected outcome:** q\*(T) anomaly maps for ≥2 systems (graphene + NbSe₂); a proof (or disproof)
  that FC-distillation lets an MLIP reproduce the Kohn cusp; firm **(L)-channel** temperature conclusions,
  with DFT pinning down the **(E)-electronic channel** the MLIP cannot reach.
- **Core tension:** the MLIP learns the *ground-state* BO PES → the **(L) lattice-anharmonic** channel is
  cheap, but the **(E) electronic** channel (Fermi smearing that moves the Kohn anomaly) is **structurally
  un-computable** by it → DFT. And §2.1 shows MLIPs **smooth sharp cusps** — a Kohn anomaly *is* a cusp.
  **So M1 first tests "can the MLIP even see the cusp" — the fuse for the whole plan.**

## Milestone overview

| milestone | goal (one line) | main compute | scale | gate |
|---|---|---|---|---|
| **M0** | install tools, prep structures | CPU / 2060 | ~1–2 h | — |
| **M1** | can the MLIP resolve Γ/K Kohn anomaly + run TDEP | **RTX 2060** | ~3–5 GPU-hr | **gate #1** |
| **M2** | full-T q\*(T) anomaly map + 2k_F cross-check | 2060 / H20 + a little V100 | MLIP ~10 GPU-hr | — |
| **M3** | NbSe₂ SSCHA soft-mode T-evolution (hardest) | **H20** (+ V100) | MLIP ~10–40; DFT ~200–500 | **gate #2** |
| **M4** | ballistic-transport reading + (L)/(E) writeup (optional EPW) | CPU (+ optional V100) | analysis ~0; EPW +100–300 | — |

**Compute-type cheat sheet:** MLIP small cells (graphene/MoS₂) → **RTX 2060** (8 GB, the only box online);
MLIP large cells (NbSe₂ CDW supercell / large cells for a sharp anomaly) → **H20** (needs to come back on
tailnet); **any real DFT (FP64: DFPT, Fermi surface, EPW) → V100/A100 rental** (2060 & H20 FP64 both
crippled).

---

## M0 — environment + structures

- **Goal:** tooling and structures ready so M1 can start.
- **Data:** crystal structures — graphene (ASE `graphene` / from `mp-48` + vacuum), monolayer MoS₂ (mp-2815),
  monolayer NbSe₂ (C2DB / mp-derived). No DFT.
- **Method:** on the 2060's `phonon` env `pip install hiphive` (TDEP fit); for M3 put `python-sscha` in a
  *separate* env to dodge the mace e3nn/torch clash (known e3nn 0.4.4 vs 0.6.0 trap). `git pull` the repo.
- **Expected result:** `import hiphive/phonopy/mace` all succeed; the three structures relax; a smoke-test
  (graphene single-point forces + one phonopy dispersion) runs.
- **Compute:** (a) ~1–2 h; (b) **CPU + RTX 2060** (local, no rental).
- **Risk:** *low.* Only trap is hiphive/sscha vs mace dependency conflict → isolate in a dedicated env.

---

## M1 — graphene de-risk (the fuse; Go/No-Go gate #1)

### M1.1 Harmonic baseline: can the MLIP see the Kohn anomaly at all?
- **Goal:** settle the global make-or-break question — does the FC-distilled MLIP reproduce graphene's Kohn
  cusp, or does the §2.1 softening wash it out?
- **Data:** graphene structure (M0); **reference dispersion from published graphene DFPT / inelastic
  scattering** (Γ-E₂g ≈ 1600 cm⁻¹, the sharp K-A₁′ kink — textbook). MLIP: (a) foundation MACE-MP and
  (b) FC-distilled MACE.
- **Method:** phonopy finite-displacement + MLIP forces → dense Γ–M–K–Γ dispersion; run the **anomaly
  locator** (discontinuity in group velocity v=dω/dq, peak in |d²ω/dq²|). Compare baseline vs distilled vs
  literature; quantify cusp softening and whether distillation recovers it.
- **Expected result:** foundation MACE shows the kink **smoothed/absent** (softened ω); FC-distilled
  **recovers the kink** at K (and Γ) to within ≲ a few % of literature; the locator flags q\*=K / Γ.
  *(Or a clean negative: even distilled can't recover it → see gate #1.)*
- **Compute:** (a) ~0.5–1 GPU-hr (graphene primitive + ~5×5 supercell finite-displacement, a few dozen
  structures); (b) **RTX 2060**.
- **Risk:** **highest, and global.** (i) MLIP smooths the cusp → no anomaly (mitigate: distill first; if
  insufficient → M1.1b). (ii) locator mistakes numerical noise for a cusp → smoothed deriv + threshold +
  multi-supercell convergence.

### M1.1b (conditional) graphene DFPT fc₂ as a distillation target
- **Goal:** provide a graphene-specific distillation target if M1.1's model is insufficient.
- **Data:** graphene's own **DFPT force constants** ("basic fc₂," not the dense-k anomaly-resolving calc →
  **cheap**).
- **Method:** QE graphene fc₂ DFPT on a moderate k-mesh → distill a graphene-specific MLIP per §2.2.
- **Expected result:** distilled MLIP fc₂ matches DFPT; dispersion (incl. the kink) tracks DFPT.
- **Compute:** (a) ~20–50 FP64 GPU-hr; (b) **V100 rental** (2-atom but FP64; H20 too weak). *Deferrable via
  literature force constants.*
- **Risk:** *medium.* FP64-blocked; bypass with literature fc₂ / public phonopy data first.

### M1.2 Level-3 TDEP: temperature-dependent ω(q,T)
- **Goal:** obtain the (L)-channel T-dependent frequencies and validate the TDEP pipeline.
- **Data:** MLIP-generated thermal snapshots (no external data). 3–4 temperatures (100/300/600 K).
- **Method:** NVT/Langevin MD or stochastic ensemble (MLIP forces) → hiPhive fit of effective fc₂(T) →
  ω(q,T); locate q\*(T).
- **Expected result:** ω(q,T) at 3–4 T; modes soften/shift with T (anharmonicity visible); q\*(T) extracted;
  the kink position/strength shows a **measurable** T-shift.
- **Compute:** (a) ~1–2 GPU-hr (~6×6=72- or 8×8≈128-atom supercell × ~2k snapshots × 4 T ≈ 15–20k force
  evals); (b) **RTX 2060** (8 GB holds ≲300-atom MACE inference).
- **Risk:** *medium.* (i) 8 GB VRAM: too-big cell → drop to 6×6 or CPU fallback (slow). (ii) TDEP fit
  convergence: decorrelate snapshots (~every 0.2 ps), enough of them; check fit residual + windows.

### M1.3 DFT validation: is q\* ≈ 2k_F? ((E)-channel reference)
- **Goal:** confirm the anomaly is electron–phonon in origin (q\* matches Fermi-surface nesting).
- **Data:** DFT — graphene Fermi surface / Dirac-point geometry (one SCF), ideally a finite-smearing DFPT
  dispersion at one T. In M1 prefer **literature DFPT/experiment**; own-DFPT deferred to rental.
- **Method:** take nesting vectors from the Fermi surface; verify K-anomaly ↔ inter-valley q≈K, Γ-anomaly ↔
  intra-valley.
- **Expected result:** the located q\* **geometrically matches** the 2k_F nesting vector (inter-valley K);
  agreement with literature DFPT.
- **Compute:** (a) own calc ~50–100 FP64 GPU-hr (semimetal/Dirac point needs ~100×100+ dense k); (b)
  **V100 rental**. With literature, (a)≈0 for M1.
- **Risk:** *medium.* FP64-blocked → M1 passes its gate on literature; own-DFPT rolls into M2/rental.

> **Go/No-Go gate #1 (end of M1):** does the FC-distilled MLIP reproduce graphene's K/Γ Kohn-anomaly kink
> (consistent with literature) in ≲5 GPU-hr, with TDEP running and giving ω(q,T)? **Pass** → M2/M3. **Fail**
> → a clean **negative result** ("MLIPs miss Kohn anomalies," consistent with §2.1); pivot to DFT-led with
> the MLIP only as a sampling accelerator.

---

## M2 — q\*(T) tracking across the full T range + 2k_F cross-check

- **Goal:** produce the full-temperature anomaly map and cross-check it against the Fermi-surface 2k_F.
- **Data:** extend M1.2 to ~8 temperatures (10–600 K); the Fermi surface / 2k_F from M1.3 (one DFT).
- **Method:** dense-T TDEP → q\*(T) and kink-strength vs T map; compare q\* against 2k_F nesting. **Key
  honesty point:** the MLIP gives (L); if literature/DFPT show the (E) channel also moving over the same
  T-range, label them separately.
- **Expected result:** q\*(T) and kink-strength(T) curves over 10–600 K with a monotone/interpretable trend;
  the (L)-channel T-shift quantified; an (L) vs (E) decomposition where DFT is available.
- **Compute:** (a) MLIP ~5–10 GPU-hr (8 T × larger cells; more with fc₃ linewidths); DFT 2k_F per M1.3; (b)
  MLIP → **RTX 2060/H20**; DFT → **V100 rental**.
- **Risk:** *medium.* (i) (L)/(E) hard to separate → must pair with a DFT-with-smearing reference, else only
  an (L)-channel claim is defensible (state it). (ii) sharp anomaly needs a large cell → may exceed 2060
  VRAM → H20.

---

## M3 — monolayer NbSe₂: CDW / soft-mode T-evolution (hardest; Go/No-Go gate #2)

- **Goal:** obtain NbSe₂'s soft-mode stabilization vs T and the CDW instability temperature, tracking the
  strong Kohn anomaly.
- **Data:** NbSe₂ monolayer structure; **the MLIP's anharmonic/CDW fidelity for NbSe₂ is doubtful** → likely
  needs NbSe₂'s own DFPT fc₂/fc₃ as a distillation target (FP64, expensive). Experimental/literature CDW T
  (~33 K bulk, higher in monolayer) for comparison.
- **Method:** `python-sscha` + MLIP forces → free-energy Hessian → soft-mode frequency vs T → CDW instability
  T; track the anomaly. SSCHA suits the near-instability regime (TDEP fails there).
- **Expected result:** the soft-mode frequency **hardens** with T; a CDW instability T extracted and compared
  to experiment; the strong anomaly's T-evolution. *(Or: the foundation MLIP yields no soft mode → triggers
  gate #2, needing DFT distillation.)*
- **Compute:** (a) MLIP ~10–40 GPU-hr (SSCHA: several populations × iterations; CDW needs a 3×3 commensurate
  supercell, many atoms); DFT validation ~200–500 FP64 GPU-hr; (b) MLIP → **H20** (8 GB 2060 too small); DFT
  → **V100/A100 rental**.
- **Risk:** **high.** (i) foundation MLIP likely misses the d-electron soft mode → needs DFT distillation,
  dragging M3 into heavy FP64. (ii) SSCHA convergence near the instability is delicate / ensemble-sensitive.
  (iii) a genuinely e-ph-dominated system, where the MLIP's no-electronic-channel blind spot bites hardest.

---

## M4 — ballistic-transport reading + (L)/(E) writeup

- **Goal:** translate the phonon anomaly into transport language and honestly bound what the phonon side vs
  electronic side can claim.
- **Data:** ω(q,T), q\*(T) from M1–M3; electronic structure (Fermi surface; e-ph matrix elements if EPW).
- **Method:** phonon side — group velocity from ω(q,T) → Landauer ballistic phonon conductance. Electronic
  side (**honest**) — q\*(T) only *locates* the Fermi-surface feature; the actual *electronic* ballistic
  window needs EPW; the phonon anomaly is the signature, not the conductivity.
- **Expected result:** ballistic phonon conductance vs T; q\*(T) geometric match to 2k_F; **if EPW is done**,
  the electronic ballistic window; otherwise the transport link is reported as a **controlled inference**.
- **Compute:** (a) analysis ~0 (CPU-minutes); optional EPW +100–300 FP64 GPU-hr; (b) **CPU**; EPW →
  **V100/A100 rental**.
- **Risk:** *medium.* Treating "phonon anomaly ↔ electronic transport window" as a conclusion rather than a
  hypothesis is the biggest narrative risk → back it with EPW/electronic structure, else report only "the
  T-dependent phonon-anomaly map + its geometric agreement with 2k_F."

---

## Risk register (consolidated)

| # | risk | impact | likelihood | mitigation | fallback if triggered |
|---|---|---|---|---|---|
| R1 | **MLIP smooths the Kohn cusp** (§2.1) | fatal (whole premise) | med–high | FC-distill first; test early in ≲5 GPU-hr (M1) | pivot DFT-led; MLIP for sampling; the negative is publishable |
| R2 | **(E) electronic channel un-computable by MLIP** | high (credibility) | certain (structural) | label (L)/(E) separately; DFT-with-smearing as (E) reference | report (L)-only + 2k_F geometric match |
| R3 | **No MDR DFPT reference for 2D monolayers** | med (distill target/validation) | high | literature force constants; basic fc₂ DFPT is cheap | compute own DFPT (V100 rental) |
| R4 | **NbSe₂ d-electron soft mode missed by MLIP** | high (M3) | med–high | DFT-distill fc₂/fc₃; SSCHA | M3 DFT-led; or swap to an easier 2D anomaly system |
| R5 | **8 GB VRAM limits large cells** | med (sharp anomaly needs big cells) | med | small cells first; big cells on H20 | CPU fallback (slow) / wait for H20 |
| R6 | **All FP64 depends on rental** | med (M1.3/M3/M4) | certain | literature first; batch DFT into one rental block | defer DFT validation, ship (L)-channel results first |
| R7 | **GPU boxes off tailnet** | med (M2 big cells, M3) | current | M0/M1 run locally on the 2060 | ask someone to `tailscale up` an H20 when needed |
| R8 | **SSCHA/hiphive dependency conflicts** | low | med | isolate in dedicated conda env | pin e3nn/torch versions (experience exists) |
| R9 | **anomaly locator false pos/neg** | med | med | smoothed deriv + threshold + multi-supercell convergence; benchmark vs literature | manual check of key q-points |

## Go/No-Go gates
- **Gate #1 (after M1):** can the (distilled) MLIP reproduce graphene's Kohn anomaly in ≲5 GPU-hr? **No →
  redirect the whole plan**, avoiding hundreds of GPU-hr on a false premise.
- **Gate #2 (before M3):** does NbSe₂'s soft mode need DFT distillation to appear? **Yes → M3's budget shifts
  to FP64 rental** (~hundreds of GPU-hr); decide then whether to do M3 or close on a cheaper 2D anomaly system.

## Doable right now (no rental, no H20)
**M0 + M1.1 + M1.2 all run on the only currently-online GPU, the RTX 2060** (~3–5 GPU-hr), with M1.3 against
literature — enough to clear **Gate #1**. Highest-leverage step: ~half a day of GPU time decides whether the
following few-hundred-GPU-hr program is worth funding.

---

### One line
**M0–M1 on the 2060 in ≲5 GPU-hr answers "can the MLIP even see the cusp" (Gate #1)**; if it passes, the
(L)-channel T-dependent dispersion and q\*(T) tracking (M2) keep riding cheap MLIP compute (2060/H20), while
**all FP64 electronic-side validation (M1.3/M3/M4) is batched into a single V100/A100 rental** — the same wall
as the main paper's B-series. NbSe₂ (M3) is the hardest, most likely DFT-led part; **Gate #2** decides it
before the spend.
