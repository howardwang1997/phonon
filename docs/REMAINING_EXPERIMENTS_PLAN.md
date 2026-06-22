# Remaining-experiments execution plan (no-rental first)

Goal: finish everything that does **not** need rented GPUs, in priority order, to make the
manuscript (`PAPER.md`) reviewer-proof. The single rental gate (GPU-SCF factor) is deferred.

Machines: **8×H20** `100.91.194.14` (GPU fine-tune + κ; CPU QE), **2×H20** `100.80.123.104`
(broad self-DFT, grinding), **2060** `100.105.21.7` (weak, ignore for heavy work).

---

## Priority 1 — Cross-model generality ★ (highest reviewer value)
**Claim to establish:** the softening, FC distillation, and the breadth law are **not MACE-specific**.
**Steps**
1. Env: `conda create -n xmodel`; install `mattersim`, `sevenn` (+ torch matching CUDA) on 8×H20.
2. **Baseline benchmark** (inference only, cheap): run MatterSim + SevenNet through the *same*
   phonon pipeline on the held-out set → confirm softening/imaginary modes are model-universal
   (supports §2.1 across models). *Acceptance:* ω_max softening + imaginary counts reported for ≥2
   extra models.
3. **FC-distill fine-tune** at least MatterSim (its fine-tune API) on the same distillation data →
   in-domain MAE + held-out transfer; if feasible, a mini breadth sweep (N = 8/24/48).
   *Acceptance:* in-domain MAE drops to <0.3 THz and transfer improves, mirroring MACE.
**Machine:** 8×H20 GPU. **Est.:** env ~30 min; baseline ~1 h; fine-tune ~2–4 h. **Rental:** no.
**Risk:** non-MACE fine-tune plumbing differs; fall back to *baseline-only* (still proves softening is
universal) if fine-tuning is blocked.

## Priority 2 — Converged fc₃ reference for 3rd-order distillation ★ (turn the negative positive)
> **Diagnosis corrected (2026-06):** the 3rd-order κ regression is **not** a degraded harmonic Hessian or
> a missing "curvature-aware loss." The distilled model *faithfully reproduces* the fc₃ it was given; the
> *affordable sc2 reference is itself under-converged* (a same-settings DFT-RTA from the 2×2×2 fc₂+fc₃
> also gives κ ≈ 48). **The real fix is a converged fc₃ reference (→ Priority 4), not a different loss.**
> The joint-training idea below is retained only as a *secondary hedge* if a converged reference stays
> out of reach.

**Claim (secondary hedge):** anharmonicity can be added *without* degrading the harmonic Hessian → κ stays converged.
**Approach (no native Hessian loss in MACE):** **joint training** — fine-tune from the foundation on
the *union* of (a) the harmonic FC-distillation data (anchors the Hessian, gave κ→143) and (b) the
anharmonic cubic-label data, with the harmonic set up-weighted; sweep the harmonic:anharmonic ratio.
**Acceptance:** Si κ(sc4) stays ≈140 (does not regress to ~45) while 3rd-order info is present;
ideally improved phonon lifetimes. **Machine:** 8×H20 GPU. **Est.:** ~1–2 h. **Rental:** no.

## Priority 3 — DFPT (ph.x) cross-check (cheap engine-rigor)
**Claim:** the finite-displacement engine agrees with linear-response DFPT.
**Steps:** write `dft_dfpt.py` (pw.x scf → ph.x at Γ + a q-path) for Si; compare ω(q) to the
finite-displacement result and to MDR. **Acceptance:** Γ-optical + zone-boundary frequencies agree to
≲2%. **Machine:** 8×H20 CPU (~1 h, 1 material). **Rental:** no.

## Priority 4 — Converged DFT-κ anchor for Si
**Claim:** a clean same-settings DFT κ to anchor §2.7 (currently anchored to experiment + convergence).
**Steps:** `dft_fc3` at sc 4 (128-atom 3rd-order supercells) → `dft_kappa` at mesh 21 → compare to
MLIP-FT sc4 (143). **Acceptance:** DFT κ(sc4) reported; expected ≈ MLIP-FT and experiment.
**Machine:** 8×H20 CPU (slow — hundreds of 128-atom SCFs). **Rental:** *borderline* — much faster on a
rented A100/V100; do on H20 only if idle, else defer with GPU-SCF.

## Priority 5 — Broaden self-DFT chemistry (ongoing)
Already grinding on 2×H20 (~24 materials, narrow Si/O/Mg). Let it continue; it demonstrates the data
engine but is not gating (κ benchmark already covers chemical breadth). **Rental:** no.

---

## Deferred (rental gate)
- **GPU-SCF per-SCF factor** — the only experiment that *requires* V100/A100 (H20 FP64 too weak).
  When rented: GPU-SCF speedup + the converged DFT-κ anchor (P4) in the same session.

## Execution order (this session)
1. Kick off P1 env install (long pole, background) **and** P3 DFPT (cheap) in parallel.
2. P1 baseline benchmark once env ready.
3. P2 curvature-aware joint training.
4. P1 fine-tune; P4 if a box is idle.
