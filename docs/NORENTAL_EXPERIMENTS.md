# No-rental experiments (A) — plan + live status

Runs entirely on the existing H20s (MLIP train/eval + small-cell DFT, FP64-light). Goal: close
Limitations (ii)(iv)(vi) + add engine rigor, **without renting**. Rental items are in
`RENTAL_EXPERIMENTS.md`.

**Boxes**: `100.80.123.104` (2×H20, fully set up: phonon env + MDR cache + replay + B79 models) is the
workhorse; `100.87.77.7` (4×H20, needs repo/data sync) is spare capacity; `100.91.139.56` (2×H20) often
busy; 2060 desktop = ignore for heavy work.

## Status tracker
| # | experiment | Limitation | box | status |
|---|---|---|---|---|
| A1 | cross-model FC distillation | (iv) | 104 | ☐ env installing |
| A2 | larger external held-out | (vi) | 104 | ☐ queued |
| A3 | multi-material NAC κ | (ii) | 104 | ☐ queued |
| A4 | DFPT (ph.x) cross-check | engine rigor | 104 | ☐ queued |

---

## A1 — Cross-model FC distillation fine-tune
- **内容**: FC-distill fine-tune MatterSim (and SevenNet if its fine-tune API allows) on the same
  distillation data; eval in-domain + held-out; optional mini breadth sweep (N=8/24/48). (Baseline
  softening already done → Fig 2b / Table S1.)
- **预期结果**: in-domain MAE <0.3 THz and held-out transfer improves, mirroring MACE → the *cure* is
  model-universal, not just the failure.
- **解决的问题**: **Limitation (iv)** — distillation demonstrated only on MACE-MP.
- **算力**: 8×H20 GPU (pure MLIP) + a fresh `xmodel` conda env (mattersim, sevenn, matched torch).
- **时间**: env ~30 min; fine-tune ~2–4 h per model.
- **Steps**: (1) `conda create -n xmodel`; pip install mattersim sevenn. (2) wrap each model in the phonon
  pipeline calculator. (3) reuse `make_finetune_data.py` distillation set; fine-tune via each model's API.
  (4) `eval_finetune.py` on held-out. **Risk**: non-MACE fine-tune plumbing differs; fall back to
  baseline-only (already proves softening universal) if a model's fine-tune is blocked.

## A2 — Larger external held-out set
- **内容**: evaluate the high-breadth FC-distilled model (B79) on **all cached MDR materials not in its
  training set** (≈30 beyond the current 6), after vetting out pathological/soft cells.
- **预期结果**: the breadth/coverage transfer laws hold on a ~20–30-material held-out set; the median is
  no longer dominated by a single hard chemistry (BN).
- **解决的问题**: **Limitation (vi)** — the laws rest on a 6-material held-out set, BN-dominated.
- **算力**: 1×H20 (pure MLIP phonon eval, cached MDR).
- **时间**: ~2–4 h.
- **Steps**: held-out = `ls data/benchmark/mdr` minus B79 train list; `eval_finetune.py --ft-model
  <B79 ft.model> --holdout <list> --no-relax`; report mean+median distribution. *Acceptance*: median
  transfer reported on ≥20 materials; trend consistent with §2.3.

## A3 — Multi-material NAC κ (beyond MgO)
- **内容**: for 3–5 polar materials (e.g. GaN, ZnO, AlN, BeO), one Γ-point DFPT each (Born charges Z*,
  ε∞) → phono3py κ with NAC, baseline vs FC-distilled.
- **预期结果**: several polar κ values land within ~10–20% of experiment after NAC → MgO's N=1 becomes
  N≈4–6.
- **解决的问题**: **Limitation (ii)** — polar-κ NAC demonstrated for a single material.
- **算力**: H20 CPU (small-cell Γ-DFPT, FP64-light) + MLIP κ.
- **时间**: ~1–3 h/material → ~1 day.
- **Steps**: `pw.x` scf + `ph.x` (`epsil`) at Γ per material → feed Z*/ε∞ to phono3py NAC; reuse the κ
  pipeline. *Acceptance*: ≥4 polar materials with NAC κ vs experiment tabulated.

## A4 — DFPT (ph.x) linear-response cross-check
- **内容**: write `dft_dfpt.py` (pw.x scf → ph.x at Γ + a q-path) for Si; compare ω(q) to the
  finite-displacement engine and MDR.
- **预期结果**: Γ-optical + zone-boundary frequencies agree to ≲2% → the finite-displacement engine ≡
  linear-response DFPT.
- **解决的问题**: engine rigor (§2.6 / Methods credibility; a likely reviewer question).
- **算力**: H20 CPU (Si small cell).
- **时间**: ~1 h (1 material).
- **Steps**: `bootstrap_qe.sh` (if ph.x not built) → `dft_dfpt.py --mp-id mp-149` → overlay vs engine.
  *Acceptance*: ≤2% on Γ-optical and zone-boundary modes.

---

## Execution order (this session)
1. **A1 env install** — long pole, kick off in background first.
2. **A2** — cheapest, fastest concrete result; run while A1 installs.
3. **A3 / A4** — small QE runs once A2 is in; A4 needs ph.x built (shared with A3).
4. A1 fine-tune once env is ready.

## Considered but excluded (no padding)
- ensemble-disagreement uncertainty acquisition — optional refinement; §2.4 negative already stands.
- 3rd-order joint-training hedge — **superseded** by the corrected §2.8 diagnosis (fix is the converged
  reference = B2, not a different loss).
- learned Born-charge model — future-work direction, not gating; A3 computes Born charges directly.
