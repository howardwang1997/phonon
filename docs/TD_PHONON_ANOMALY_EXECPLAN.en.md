# TD-phonon / Kohn-anomaly project: detailed execution plan (data · method · compute · risk)

This is the **executable refinement** of [`TD_PHONON_ANOMALY_PLAN.en.md`](./TD_PHONON_ANOMALY_PLAN.en.md)
(the high-level plan): milestones M0–M4 broken into concrete steps, each with **① data · ② methodology
(what method solves what problem) · ③ compute ((a) amount (b) type) · ④ risk**. A **risk register** and
**Go/No-Go gates** close it out.

> **The tension that shapes the whole plan:** an MLIP learns the *ground-state* Born–Oppenheimer PES, so the
> **(L) lattice-anharmonic** temperature channel is cheap for it, while the **(E) electronic** channel
> (Fermi-surface smearing that moves the Kohn anomaly) it *structurally cannot* produce — that needs **DFT
> (DFPT + electronic smearing)**. Worse, §2.1 shows foundation MLIPs **smooth sharp curvature** — and a Kohn
> anomaly *is* that sharp cusp. So **M1's first job is to prove whether the MLIP can even see the cusp** —
> the "fuse" protecting the whole few-hundred-GPU-hr program.

## Overview

| milestone | one-liner | main compute | scale | blocking? |
|---|---|---|---|---|
| **M0** env + structures | install hiphive/sscha, prep structures | CPU / 2060 | ~1–2 h | no |
| **M1** graphene de-risk | can the MLIP resolve Γ/K Kohn anomaly + run TDEP | **RTX 2060** | ~3–5 GPU-hr | **Go/No-Go gate #1** |
| **M2** q\*(T) tracking | full-T anomaly map + 2k_F cross-check | 2060 / H20 + a little V100 | MLIP ~10 GPU-hr; DFT see M1.3 | no |
| **M3** NbSe₂ CDW | SSCHA soft-mode T-evolution (hardest) | **H20** (+ V100 validation) | MLIP ~10–40; DFT ~200–500 | **Go/No-Go gate #2** |
| **M4** ballistic transport + writeup | group velocity/conductance + (L)/(E) (optional EPW) | CPU (+ optional V100) | analysis ~0; EPW +100–300 | no |

**Compute-type cheat sheet:** MLIP small cells (graphene/MoS₂) → **RTX 2060** (8 GB, the only box currently
online); MLIP large cells (NbSe₂ CDW supercell / large cells for a sharp anomaly) → **H20** (needs to come
back on tailnet); **any real DFT (FP64: DFPT, Fermi surface, EPW) → V100/A100 rental** (the 2060 and H20
both have crippled FP64 — not viable).

---

## M0 — environment + structures

**① Data:** crystal structures. Graphene (ASE `graphene` / from `mp-48` + vacuum), monolayer MoS₂ (mp-2815),
monolayer NbSe₂ (C2DB / mp-derived). No DFT.
**② Method:** "tooling in place." On the 2060's `phonon` env, `pip install hiphive` (TDEP fitting); for M3
pre-install `python-sscha` in a *separate* env (avoid the mace e3nn/torch clash — the known e3nn 0.4.4 vs
0.6.0 trap). `git pull` the repo.
**③ Compute:** (a) ~1–2 h; (b) **CPU + RTX 2060** (local, no rental).
**④ Risk:** *low.* Only trap is hiphive/sscha vs mace dependency conflict → isolate in a dedicated env.

---

## M1 — graphene de-risk (the fuse; Go/No-Go gate #1)

### M1.1 Harmonic baseline: can the MLIP see the Kohn anomaly at all?
**① Data:** graphene structure (M0); **reference dispersion from published graphene DFPT / inelastic
scattering** (Γ-E₂g ≈ 1600 cm⁻¹ and the sharp K-A₁′ kink are textbook and well-measured) — *don't* compute
DFT yet. MLIP: (a) foundation MACE-MP and (b) FC-distilled MACE.
**② Method:** answers the **make-or-break** question "does the MLIP smooth the cusp?" phonopy
finite-displacement + MLIP forces → dense Γ–M–K–Γ dispersion; run the **anomaly locator** (discontinuity in
group velocity v=dω/dq, peak in |d²ω/dq²|). Compare baseline vs distilled vs literature; quantify how much
the cusp is softened and whether distillation recovers it.
**③ Compute:** (a) ~0.5–1 GPU-hr (graphene primitive + ~5×5 supercell finite-displacement, only a few dozen
structures); (b) **RTX 2060**.
**④ Risk:** **highest, and global.** (i) MLIP smooths the cusp → no anomaly. Mitigate: use the FC-distilled
model first; if its carbon coverage is too weak, distill on graphene's *own* DFPT fc₂ (M1.1b). (ii) the
locator mistakes numerical noise for a cusp → smoothed derivative + threshold + multi-supercell convergence.

### M1.1b (conditional) graphene DFPT fc₂ as a distillation target
**① Data:** if the distilled model is insufficient in M1.1, need graphene's *own* **DFPT force constants** as
the distillation target (note: this is the "basic fc₂," *not* the dense-k anomaly-resolving calc → **cheap**).
**② Method:** QE graphene fc₂ DFPT on a moderate k-mesh → distill a graphene-specific MLIP per §2.2.
**③ Compute:** (a) ~20–50 FP64 GPU-hr; (b) **V100 rental** (2-atom cell but FP64; H20 too weak). *Can be
deferred by using literature force constants.*
**④ Risk:** *medium.* FP64-blocked; bypass with literature fc₂ / public graphene phonopy data first.

### M1.2 Level-3 TDEP: temperature-dependent ω(q,T)
**① Data:** MLIP-generated thermal snapshots (no external data). 3–4 temperatures (100/300/600 K).
**② Method:** gets the **(L)-channel** T-dependent frequencies. NVT/Langevin MD or stochastic ensemble (MLIP
forces) → hiPhive fit of effective fc₂(T) → ω(q,T); locate q\*(T).
**③ Compute:** (a) ~1–2 GPU-hr (~6×6=72-atom or 8×8≈128-atom supercell × ~2k snapshots × 4 T ≈ 15–20k force
evals); (b) **RTX 2060** (8 GB holds ≲300-atom MACE inference).
**④ Risk:** *medium.* (i) 8 GB VRAM: if the cell is too big → drop to 6×6 or CPU fallback (slow). (ii) TDEP
fit convergence: decorrelate snapshots (~every 0.2 ps), enough of them; check via fit residual + windows.

### M1.3 DFT validation: is q\* ≈ 2k_F? ((E)-channel reference)
**① Data:** DFT — graphene Fermi surface / Dirac-point geometry (one SCF), and (ideally) a finite-electronic-
smearing DFPT dispersion at one T. In M1, **prefer literature DFPT/experiment** for the cross-check; defer
own-DFPT to the rental phase.
**② Method:** answers "is the anomaly electron–phonon in origin?" Take nesting vectors from the Fermi surface;
verify the K-anomaly maps to inter-valley q≈K and the Γ-anomaly to intra-valley.
**③ Compute:** (a) own calc ~50–100 FP64 GPU-hr (semimetal/Dirac point needs very dense k, ~100×100+); (b)
**V100 rental**. With literature, (a)≈0 for M1.
**④ Risk:** *medium.* FP64-blocked → M1 passes its gate on literature alone; own-DFPT rolls into M2/rental.

> **Go/No-Go gate #1 (end of M1):** does the FC-distilled MLIP reproduce graphene's K/Γ Kohn-anomaly kink
> (consistent with literature) in ≲5 GPU-hr, with TDEP running and giving ω(q,T)? **Pass** → proceed to
> M2/M3. **Fail** (MLIP smooths it, distillation can't recover) → that is itself a clean **negative result**
> ("MLIPs miss Kohn anomalies," consistent with §2.1); pivot to a DFT-led study with the MLIP only as a
> sampling accelerator.

---

## M2 — q\*(T) tracking across the full T range + 2k_F cross-check

**① Data:** extend M1.2 to ~8 temperatures (10–600 K); the Fermi surface / 2k_F from M1.3 (one DFT).
**② Method:** answers "how does the kink move with T?" dense-T TDEP → q\*(T) and kink-strength vs T map;
compare q\* against 2k_F nesting. **Key honesty point:** the MLIP gives the (L) channel; if literature/DFPT
show the (E) channel (electronic smearing) also moving over the same T-range, label them separately.
**③ Compute:** (a) MLIP ~5–10 GPU-hr (8 T × larger cells; more if fc₃ linewidths); DFT 2k_F per M1.3; (b)
MLIP → **RTX 2060/H20**; DFT → **V100 rental**.
**④ Risk:** *medium.* (i) (L)/(E) hard to separate → must pair with a DFT-with-smearing reference, else only
an (L)-channel claim is defensible (still valuable, but state it). (ii) sharp anomaly needs a large cell →
may exceed 2060 VRAM → use H20.

---

## M3 — monolayer NbSe₂: CDW / soft-mode T-evolution (hardest; Go/No-Go gate #2)

**① Data:** NbSe₂ monolayer structure; **the MLIP's anharmonic/CDW fidelity for NbSe₂ is doubtful** → likely
needs NbSe₂'s *own* DFPT fc₂/fc₃ as a distillation target (expensive, FP64). Experimental/literature CDW
transition T (~33 K bulk, higher in monolayer) for comparison.
**② Method:** answers "soft-mode anharmonic stabilization vs T." `python-sscha` + MLIP forces → free-energy
Hessian → soft-mode frequency vs T → CDW instability temperature; track the anomaly. SSCHA suits the
near-instability regime (TDEP's perturbative footing fails there).
**③ Compute:** (a) MLIP ~10–40 GPU-hr (SSCHA: several populations × iterations; CDW needs a 3×3 commensurate
supercell, many atoms); DFT validation ~200–500 FP64 GPU-hr; (b) MLIP → **H20** (CDW supercell + ensembles;
8 GB 2060 too small); DFT → **V100/A100 rental**.
**④ Risk:** **high.** (i) the foundation MLIP likely misses the d-electron-driven soft mode → needs DFT
distillation, dragging M3 into heavy FP64. (ii) SSCHA convergence near the instability is delicate and
ensemble-size-sensitive. (iii) this is a genuinely *e-ph-dominated* system, where the MLIP's structural blind
spot (no electronic channel) bites hardest → DFT must lead, MLIP only accelerates sampling.

---

## M4 — ballistic-transport reading + (L)/(E) writeup

**① Data:** ω(q,T), q\*(T) from M1–M3; electronic structure (Fermi surface; e-ph matrix elements if EPW).
**② Method:** translates the phonon anomaly into transport language. Phonon side: group velocity from ω(q,T)
→ Landauer ballistic phonon conductance. Electronic side (**honest**): q\*(T) only *locates* the Fermi-surface
feature; the actual *electronic* ballistic window needs EPW — the phonon anomaly is the signature, not the
conductivity.
**③ Compute:** (a) analysis ~0 (CPU-minutes); optional EPW +100–300 FP64 GPU-hr; (b) **CPU**; EPW →
**V100/A100 rental**.
**④ Risk:** *medium.* Treating "phonon anomaly ↔ electronic transport window" as a conclusion rather than a
hypothesis is the biggest narrative risk; must be backed by EPW/electronic structure, else report only "the
T-dependent phonon-anomaly map + its geometric agreement with 2k_F," leaving the transport link a controlled
inference.

---

## Risk register (consolidated)

| # | risk | impact | likelihood | mitigation | fallback if triggered |
|---|---|---|---|---|---|
| R1 | **MLIP smooths the Kohn cusp** (§2.1) | fatal (whole premise) | med–high | FC-distill first; test early in ≲5 GPU-hr (M1) | pivot to DFT-led; MLIP for sampling; the negative is publishable |
| R2 | **(E) electronic channel un-computable by MLIP** | high (credibility) | certain (structural) | label (L)/(E) separately; DFT-with-smearing as (E) reference | report (L)-only + 2k_F geometric match |
| R3 | **No MDR DFPT reference for 2D monolayers** | med (distill target/validation) | high | use literature force constants; basic fc₂ DFPT is cheap | compute own DFPT (V100 rental) |
| R4 | **NbSe₂ d-electron soft mode missed by MLIP** | high (M3) | med–high | DFT-distill fc₂/fc₃; SSCHA | M3 goes DFT-led; or swap to an easier 2D anomaly system |
| R5 | **8 GB VRAM limits large cells** | med (sharp anomaly needs big cells) | med | small cells first; big cells on H20 | CPU fallback (slow) / wait for H20 |
| R6 | **All FP64 depends on rental** | med (M1.3/M3/M4) | certain | literature first; batch DFT into one rental block | defer DFT validation, ship (L)-channel MLIP results first |
| R7 | **GPU boxes are off tailnet** | med (M2 big cells, M3) | current | M0/M1 run locally on the 2060 | ask someone to `tailscale up` an H20 when needed |
| R8 | **SSCHA/hiphive dependency conflicts** | low | med | isolate in dedicated conda env | pin e3nn/torch versions (experience exists) |
| R9 | **anomaly locator false pos/neg** | med | med | smoothed deriv + threshold + multi-supercell convergence; benchmark vs literature | manual check of key q-points |

## Go/No-Go gates
- **Gate #1 (after M1):** can the (distilled) MLIP reproduce graphene's Kohn anomaly in ≲5 GPU-hr? **No →
  redirect the whole plan** (DFT-led, or a negative-result writeup) — *avoid burning hundreds of GPU-hr on a
  false premise.*
- **Gate #2 (before M3):** does NbSe₂'s soft mode require DFT distillation to appear at all? **Yes → M3's
  budget shifts to FP64 rental** (~hundreds of GPU-hr); decide then whether to do M3 or close on a cheaper 2D
  anomaly system.

## Doable right now (no rental, no H20)
**M0 + M1.1 + M1.2 all run on the only currently-online GPU, the RTX 2060** (~3–5 GPU-hr), with M1.3 done
against literature — enough to clear **Gate #1**. This is the highest-leverage step: ~half a day of GPU time
decides whether the following few-hundred-GPU-hr program is worth funding.

---

### One line
**M0–M1 on the 2060 in ≲5 GPU-hr answers "can the MLIP even see the cusp" (Gate #1)**; if it passes, the
(L)-channel T-dependent dispersion and q\*(T) tracking (M2) keep riding cheap MLIP compute (2060/H20), while
**all FP64 electronic-side validation (M1.3/M3/M4) is batched into a single V100/A100 rental** — the same wall
as the main paper's B-series. NbSe₂ (M3) is the hardest, most likely DFT-led part; **Gate #2** decides it
before the spend.
