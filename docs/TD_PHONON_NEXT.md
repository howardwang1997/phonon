# TD-Phonon — Next Experiments Plan (post-rigor)

Grounded in: the execution plan (`TD_PHONON_ANOMALY_EXECPLAN.en.md`, M0–M4 + Path P),
the long-term goal (a publishable "FC-distilled MLIP reproduces phonon Kohn anomalies +
(L)/(E) channel decomposition" paper), and the **latest results** (`TD_PHONON_SUMMARY.md` —
the three-material arc + the four rigor follow-ups, all done).

## Where we stand
- **M0–M3 done**, both Go/No-Go gates **passed** (graphene cusp recovered by distillation;
  NbSe₂ CDW soft mode needs + responds to distillation).
- **Four rigor follow-ups done**: #1 SSCHA, #2 χ(q) nesting, #3 DFPT-with-smearing, #4 cross-model.
- **Roadmap remainder:** **M4** (ballistic/(L)-(E) writeup, optional EPW) + **Path P** (anharmonic
  distillation). Path-P infra already exists and ran on **graphene** (commit `7414c94`: multi-T
  150 configs on GPU-QE, force gap 149→21 meV/Å). It has **not** been applied to NbSe₂.

## The one open question the latest result created (this drives the plan)
SSCHA #1 found: **harmonic FC distillation reproduces NbSe₂'s soft fc₂, but the MLIP does NOT
sustain the CDW under SSCHA fluctuations at any T** — because F=−Φ₂u carries *no anharmonicity*,
so the MLIP lacks the CDW **double-well**. This is exactly risk **R10** ("mistaking harmonic fc₂
distillation for an anharmonic fix") made concrete, and **Path-P (finite-T DFT-force distillation)
is its direct, pre-planned answer.** This is the single highest-value next experiment.

---

## Priority 1 — Path-P anharmonic distillation on NbSe₂ → re-SSCHA  ⭐ (the centerpiece)
**Question:** does *anharmonic* (finite-T DFT-force, Path-P-B) distillation give the MLIP the CDW
double-well that harmonic distillation missed — so that re-running SSCHA recovers a *real* CDW
T-evolution (soft/ordered at low T, melting toward T_CDW)?

**Method** (reuse graphene Path-P infra, swap material):
1. Sample thermal configs of the 3×3 NbSe₂ supercell from the FT MLIP at 2–3 T (e.g. 50/150/300 K),
   ~100–150 configs (`path_p_make_data.py`, NbSe₂ + metallic smearing).
2. DFT forces on each (GPU pw.x, ONCV Nb/Se from vq3, `degauss` for the metal) → anharmonic labels.
3. Fine-tune the foundation MLIP on these forces (`path_p_eval.py` recipe; 2060).
4. **Re-run SSCHA (`vq3e_nbse2_sscha.py`) with the anharmonic-FT MLIP**, same T grid 20–400 K.

**Expected outcomes (both publishable):**
- *(positive)* anharmonic-FT sustains the soft/imaginary mode at low T and it stabilizes near T_CDW
  → "anharmonic distillation captures the CDW landscape that harmonic distillation can't" — closes
  the thesis loop and upgrades #1 from a limitation into a *demonstrated cure*.
- *(boundary)* it still doesn't (small commensurate cell can't host the incommensurate CDW) →
  honest, well-scoped boundary on FC distillation.

**Compute:** DFT ~3–6 GPU-hr on the rented V100 (27-atom metallic SCF ×~150 ÷ 19×); fine-tune ~1 hr
on the 2060; SSCHA minutes (working). **Wall-clock ~half a day. Runnable now.**

---

## Priority 2 — free diagnostics + the deliverable figures (M4-cheap)  (near-zero compute)
Do regardless; produces the paper's money figures and a free quantitative anharmonicity target.
- **fc₃ anharmonicity diagnostic** (the plan's "free on the 2060 now"): fit fc₂+fc₃ from existing MD
  snapshots → fc₃ norm / linewidth / softening slope = a *number* for the MLIP's current anharmonicity
  (the exact gap SSCHA #1 found). Also a baseline P1 must beat.
- **ω(q,T) money figures**: turn the point-wise soft-mode-vs-T data into full **dispersion-vs-T**
  curves for graphene (Γ-E₂g, K-A₁′) and NbSe₂ (soft branch) from FT-MLIP + TDEP/SSCHA — the figure
  a reader expects but we haven't drawn end-to-end.
- **Ballistic phonon conductance G(T)** (M4): from ω(q,T) (CPU-minutes) — the transport hook the plan
  names; report as the (L)-channel observable (note the (E) caveat).

**Compute:** MLIP/CPU only, ~0 GPU. A few hours of scripting/plotting.

---

## Priority 3 — EPW e-ph linewidths for ONE material (the rigor anchor)  (bigger setup)
The **canonical Kohn-anomaly observable** is the e-ph linewidth γ_qν / α²F(ω) / λ — #3 gives only the
DFPT *frequency* trend. One material done rigorously (recommend **graphene**: well-characterized α²F,
simplest Wannierization) upgrades the (E)-channel from "frequency hardens with T_el" to "actual mode
linewidths + λ", the reviewer-proof version.
- **Lift:** install Wannier90 + EPW on the V100 (FP64); coarse DFPT grid → Wannier interpolation →
  fine q-grid γ_qν. ~tens of GPU-hr + a day of setup. This is the flagship's core machinery, proven
  on one material first.

---

## Deferred (needs 8×V100) — the flagship family survey
The systematic 2D Kohn-anomaly/CDW family (graphene/h-BN controls + MoS₂/MoSe₂/WS₂ gapped +
NbSe₂/NbS₂/TaS₂/TaSe₂/TiSe₂/VSe₂ CDW), each with DFT fc₂(+fc₃)→distill, DFPT-with-smearing, EPW,
MLIP+SSCHA ω(q,T). Turns 3 case studies into a benchmarked law. Stays deferred until 8×V100.

---

## Recommended sequence
1. **Start P1 now** (Path-P NbSe₂ DFT campaign on the V100) — it's the experiment the latest result
   demands, reuses existing infra, and is the deepest novel result.
2. **In parallel, P2** on the 2060/CPU (no GPU contention) — free diagnostics + the deliverable figures.
3. **Then P3** (EPW on graphene) as the rigor anchor / flagship dry-run, if the V100 rental window allows.
4. Flagship family survey stays in the doc until 8×V100.

**Immediate next action:** kick off P1 step 1–2 (sample NbSe₂ thermal configs + queue DFT forces on
the V100), while running P2's fc₃ diagnostic on the 2060.
