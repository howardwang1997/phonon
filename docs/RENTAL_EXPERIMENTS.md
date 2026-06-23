# Rental-gated experiments (V100/A100) — execution-ready spec

These are the **only** experiments that need rented strong-FP64 GPUs. Everything else runs on the
existing H20s (see `NORENTAL_EXPERIMENTS.md`). The wall is FP64: **V100 ~7 TFLOPS, A100 ~9.7, H20 ~1**
(inference card, crippled). QE-DFT SCF on H20 is therefore the bottleneck; renting V100/A100 is the point.

> **B1, B2, B3 share ONE DFT run** — a converged silicon fc₂+fc₃ at sc4. Computing it once yields the κ
> anchor (B1), the distillation reference (B2), and the per-SCF timing (B3). Do **not** budget them
> separately. B4 is a separate, much larger scale-out.

## Sizing (8×V100, embarrassingly parallel → wall = GPU-hr ÷ 8)

| tier | wall | GPU-hr | delivers |
|---|---|---|---|
| **Lean** | ~2–3 d | ~50–200 | B3 + a couple κ spot-checks only |
| **Core week** | **~5–7 d** | ~600–1000 | **B1 + B2 + B3** → closes Limitations (i)(iii)(v), *submittable* |
| **Competitive month** | **~30 d** | ~5760 | + **B4** (scale) → *NCS-competitive* |

Cost reference: competitive month ≈ ¥2000–8000 cloud (China rates).

---

## B1 — Converged Si DFT-κ anchor (sc≥4)
- **内容**: QE finite-displacement fc₂+fc₃ for Si at sc3 (54-atom) **and** sc4 (128-atom); phono3py RTA κ
  at mesh 21³; report κ vs supercell to show convergence. fc₃ at sc4 ≈ **800 SCFs** (cf. SiC sc4
  `n_disp3=834`); sc3 ≈ 180 SCFs; fc₂ negligible.
- **预期结果**: same-settings DFT κ(sc4) reported, expected ≈ MLIP-FT 143 and experiment 140.
- **解决的问题**: **Limitation (i)** — κ currently anchored only to experiment (possible RTA error
  cancellation); no converged same-settings DFT anchor.
- **算力**: 8×V100, strong FP64. (H20: sc4 ⇒ "weeks" at ~31 min/SCF CPU — infeasible.)
- **时间**: ~2–4 d pure SCF; **~5–7 d** with the sc3/sc4 convergence study + k-mesh/cutoff checks + reruns.
- **Steps**: `scripts/bootstrap_qe.sh` (QE-GPU + SG15) → `scripts/dft_fc3.py --mp-id mp-149 --sc 3` and
  `--sc 4` → `scripts/run_mlip_kappa.py`-style phono3py κ on the DFT FCs. *Acceptance*: κ(sc4) within
  ~10–15% of 140 and monotone toward it across sc2→sc3→sc4.
- **Faster route (optional)**: compressed-sensing fc₃ (`dft_fc3_cs`, symfc, ~100–200 random
  displacements) cuts SCF count ~4–8× → ~1–1.5 d, at the cost of a small fc₃ accuracy check.

## B2 — Converged fc₃ reference + re-validated 3rd-order distillation
- **内容**: take B1's converged Si fc₃ as the distillation target; regenerate cubic-label data
  (`F=−Φ₂u−½Φ₃uu`), re-fine-tune the 3rd-order FC-distilled model, recompute κ.
- **预期结果**: with a *converged* fc₃ reference the distilled κ no longer regresses to ~45; it returns to
  ~143 → turns the §2.8 negative into a positive.
- **解决的问题**: **Limitation (v)** — §2.8 showed the model faithfully reproduces the fc₃ it is given but
  the *affordable sc2 reference is under-converged*; this supplies the converged reference.
- **算力**: reuses B1's DFT (V100); the re-fine-tune is MLIP-only (any H20).
- **时间**: **shares B1's DFT** (not additive) + MLIP retrain ~2–4 h.
- **Steps**: after B1, `scripts/make_finetune_data.py` with the converged fc₃ → `run_one_job.sh`-style
  3rd-order distill → phono3py κ. *Acceptance*: distilled κ(sc4) ≈ DFT κ(sc4) and ≈ 143 (not ~45).

## B3 — per-SCF GPU speedup factor (measured)
- **内容**: time the same QE SCFs on V100 (QE-GPU/OpenACC) vs H20/CPU; report single-SCF speedup and the
  composed end-to-end ~50× workflow timing.
- **预期结果**: measured **5–15×** per-SCF GPU factor — fills the one "literature, not realised" row in
  Table 4 (§2.6).
- **解决的问题**: **Limitation (iii)** — the per-SCF GPU factor was never measured (H20 FP64 too weak).
- **算力**: 8×V100, ~30–50 GPU-hr.
- **时间**: ~½ d, piggy-backed on B1's SCFs (just instrument them).
- **Steps**: run a representative Si supercell SCF under `pw.x` GPU vs CPU build; record wall + iteration
  counts; combine with the §2.6 symmetry/reuse/non-diagonal factors. *Acceptance*: factor reported with
  hardware scope; Table 4 (iii) row updated from "literature" → "measured".

## B4 — κ + 3rd-order distillation at scale (competitive upgrade, NOT gating)
- **内容**: anharmonic self-DFT κ benchmark across **30–50 materials** + broad harmonic set (300–500
  materials) + a real **multi-round** closed-loop data engine (acquire→DFT→distil→repeat with self-DFT).
- **预期结果**: κ benchmark extended from MLIP-only to self-generated DFT at scale; a demonstrated real
  closed loop (not the MDR-oracle stand-in of §2.4).
- **解决的问题**: not a limitation — a **competitiveness upgrade** (submittable → NCS-competitive).
- **算力**: 8×V100, ~5760 GPU-hr (anharmonic 3rd-order self-DFT is the long pole, 10–50× harmonic cost).
- **时间**: **~1 month**.
- **Steps**: scale `dft_fc3` + `dft_dataset` over the breadth-selected material set; drive the loop with
  the coverage acquisition function (§2.4). *Acceptance*: ≥30-material DFT-κ benchmark + ≥2 real loop
  rounds with transfer improving.

---

## Run order when cards arrive
1. `bootstrap_qe.sh` on the rented box (QE-GPU + SG15 pseudos).
2. **B1** Si sc3 then sc4 fc₃ (instrument SCF timings → **B3** for free).
3. **B2** re-fine-tune on the converged fc₃ → κ.
4. Stop here for *submittable* (core week). **B4** only if committing to the competitive month.
