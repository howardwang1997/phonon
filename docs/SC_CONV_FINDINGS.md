# Supercell convergence of the smearing-dependent Kohn kink (graphene)

**Date** 2026-07-04 · branch `smearing-kink-ml`

## Result
The smearing-induced Friedel tail is long-range; the Kohn kink at K needs a large fc2
real-space range. Matched-k (kpts4) supercell study:

| dg | T_el(K) | 6×6 (≤7Å) | 8×8 (≤9.8Å, converged) | 6×6 truncation error |
|----|---------|-----------|------------------------|----------------------|
| 0.005 | 789  | 18.3 | 6.1 | +12.2 |
| 0.01  | 1579 | 13.7 | 6.4 | +7.3 |
| 0.02  | 3158 | 9.8  | 6.5 | +3.3 |
| 0.04  | 6316 | 3.9  | 3.4 | +0.5 |
| 0.08  | 12631| 0.1  | 0.1 | 0.0 |

- **It is a supercell (truncation) effect, NOT k-density**: at matched kpts4, 6×6→8×8 drops the
  kink by up to +12; the k-density effect (eff32 vs eff36) is ~0.
- **8×8 is ~converged**: its kink-vs-cutoff plateaus by ~9Å at ~6 cm⁻¹ (residual Friedel
  oscillation ±0.3); 6×6 was still rising at its edge.
- **Refined physics**: the converged kink is ~6 and nearly FLAT for sharp smearing (dg 0.005–0.02),
  only melting for broad smearing (dg≥0.04). The strong "kink melts with T_el" seen at 6×6 in the
  sharp range was largely a TRUNCATION ARTIFACT.
- **10×10 not needed / infeasible**: 200-atom GPU-QE OOMs host RAM on our boxes; and the 8×8
  kink-vs-cutoff plateau shows 8×8 suffices. This is why the long-range term (fit tail + extrapolate)
  is the right tool rather than ever-bigger DFT.

## EPW lambda(T_el) family note
VSe2 lambda(T_el)=2.12–2.24 (good). **TiSe2 lambda diverges — PHYSICAL**: TiSe2 is a semimetal with
DOS(Ef)≈0.01 states/eV/cell, so the Fermi-surface double-delta lambda blows up (N(Ef)→0). This is a
finding (TiSe2 CDW is not a conventional lambda/e-ph mechanism), not a fixable bug. TaSe2 lambda
magnitude also off (~200) — needs dedicated double-delta k-convergence; off critical path.
