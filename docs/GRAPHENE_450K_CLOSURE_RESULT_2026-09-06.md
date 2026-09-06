# 450 K Fixed-Smearing Closure — Result Record (2026-09-06)

Minimal independent-DFT closure of the deployed fixed-smearing full-EPC 450 K
K-A′ curve (`R2AT_paired_full_epc_acceptance_20260827`,
`r2ao_s0_fixed_smearing_full_epc_matrices.npz`). Protocol frozen in
`results/graphene_physics_temperature/post_p4_feasibility/R1_450k_matched_overlap_screen/freeze_manifest.json`
(R1′) and `r2b_pair_manifest.json` (R2B) **before any DFT label was read**.
Execution plan: `docs/GRAPHENE_FIXED_SMEARING_FINITE_T_CLOSURE_PLAN_2026-08-22.md`.

## Composition under test (replay-validated)

```
short_fc = Φ_SSCHA(T_lat=450, q6-sharp) − Φ_q6        (operator delta_fc_full)
D(q)     = D[short_fc](q) + Π_full-EPC(q)|e_A′⟩⟨e_A′|  (mode-projected, A′-tracked)
```

Replayed bit-exactly on all 6 model×T stored matrices (max |Δω| ≤ 1e-6 cm⁻¹)
before any comparison — the closure measures model-vs-DFT, never code-path drift.

Key deployed numbers (K = q-index 120): short ω(K) = 1301.4709,
full S0 = 1269.7964, R2AO = 1266.2849, finite_q6 = 1270.3794 cm⁻¹;
Π(K) = −81443.6 cm⁻²; scale = 2.719318e5; m·λ_short(S0) = 74.8130 eV/Å².

## Why the reference estimator changed (R2A → R2B)

Synthetic study on the *stored* R2AO forces, run before any DFT label existed:

| estimator | verdict |
|---|---|
| cutoff-6 hiPhive global-fit K eigenvalue | **unusable**: scatters ~60 cm⁻¹ across 12-config subsets (batch_3 +67.9; random-12 spread 62.7) — forces are ~17% anharmonic |
| A′ force-regression slope | **biased**: ≈ +30 cm⁻¹ above the SSCHA eigenvalue (regression estimates ⟨V″⟩+asymmetric terms, not the stationarity object) |
| tagged-pair thermal curvature | **exact** on linear forces (2.5e-14 spread); measures ⟨∂²V/∂a²⟩_T — the SSCHA free-energy Hessian's stationarity object |

R2B therefore labels 12 stratified frozen-reserve seeds at u_i ± 0.04 Å·P,
P = verified K-A′ eigendisplacement of the deployed S0@450 short Hessian
(Γ-eigenspace cluster of the 216×216 problem, eigen-residual 4e-16).
Projected curvature, q6 operator force subtracted, + shared Π(K) → absolute K.
Full synthetic end-to-end (labels emulated as −(short_S0+q6)·u): all gates
pass, curvature K exact to 5.5e-12 cm⁻¹, shape RMSE 0.41 / max 0.89 cm⁻¹.

## DFT labeling

ASE-Espresso scf, 60/240 Ry, fd degauss 0.0019000869380739254 Ry, 8×8×1 k,
conv_thr 1e-10, C_ONCV_PBE-1.2.upf, 6×6 graphene (72 atoms).
48 labels total: shard_A/B (12 plain, R1′ overlap screen) + r2b_half1/2
(24 tagged pairs), two V100 boxes. Runner
`run_graphene_450k_closure_batch.sh`; canonical label tree
`R1_450k_matched_overlap_screen/labels/` (tagged ids 1000+2i±1 → local = id−1000).

## Result 1 — absolute K A′ (R2B curvature, 24/24 labels) ✅ measured

| quantity | value |
|---|---|
| λ_short,DFT | 6.51630 eV/Å²/amu (m·λ = 78.269 eV/Å²) |
| ω_short,DFT | 1331.161 cm⁻¹ (deployed 1301.471) |
| **ω_full,DFT (composed with shared Π)** | **1300.209 cm⁻¹** |
| deployed S0 full | 1269.796 cm⁻¹ |
| **abs diff** | **+30.41 cm⁻¹** (+2.4%) — frozen gate ≤ 5 → **FAIL** |
| deployed R2AO full | 1266.285 → diff +33.92 cm⁻¹ |
| per-seed curvature | −75.99 … −80.12 eV/Å², std 1.08 → systematic, not noise |
| statistical error of the mean | SEM = 1.08/√12 = 0.31 eV/Å² → ±1.5 cm⁻¹ on ω (result: **+30.4 ± 1.5 cm⁻¹**) |
| split-half (6/6) | 1297.17 / 1303.24 → spread 6.08 cm⁻¹ (gate ≤ 5 → marginal FAIL; = 2.3×SEM of a half-mean, seed-ordering luck, mean itself tight) |

Cross-reads:
- No-decomposition check: raw DFT total curvature (q6 not subtracted) gives
  ω_total ≈ 1301.2 vs deployed finite_q6 object 1270.4 — the same ~31 cm⁻¹,
  so the gap sits in the Born–Oppenheimer total, not in the short/Π split.
- The offset exceeds the cusp depth itself at d=0.025 (17.3 cm⁻¹): in
  absolute-frequency terms the deployed composition's K-accuracy floor is
  larger than the Kohn-anomaly signal. Shape/cusp gates (below) test the
  *spectral morphology* claim separately.

Interpretation (frozen-protocol): holding the EPC response Π(K) shared, the
deployed SSCHA short Hessian is systematically **~3.5 eV/Å² (0.287 eV/Å²/amu)
soft at K** relative to the independent DFT thermal curvature at the same
450 K ensemble — an honest accuracy bound for the MLIP-SSCHA leg of the
composition, measured with seed-level tightness.

## Result 2 — R1′ force-overlap gates (12 plain labels) ✅ measured

| metric | value | frozen gate | verdict |
|---|---|---|---|
| force RMSE | 153.1 meV/Å | ≤ 30 | FAIL |
| force max abs | 3101.0 meV/Å | ≤ 200 | FAIL |
| A′ diff RMS | 91.5 meV/Å | ≤ 15 | FAIL |
| A′ slope ratio R2AO/DFT | 1.0322 | [0.9, 1.1] | **PASS** |
| A′ slope rel err vs Φ | 10.3% | ≤ 5% | FAIL |

Context: DFT force RMS on these 450 K thermal configs is 2061 meV/Å → the
R2AO-vs-DFT RMSE is a **7.4% relative** force agreement. The 30/200 meV gates
were a matched-quality screen (near-label-noise); the deployed MLIP sits at
several-percent force accuracy on hot thermal configs — the same several
percent that integrates to +30 cm⁻¹ at K (Result 1). `r1_evaluation.json`.

## Result 3 — composed shape + cusp gates (fitter v2, all 48 labels) ✅ measured

| gate | value | threshold | verdict |
|---|---|---|---|
| K-referenced shape RMSE | 0.634 cm⁻¹ | ≤ 1 | **PASS** |
| K-referenced shape max | 1.210 cm⁻¹ | ≤ 2 | **PASS** |
| cusp depth rel err, d=0.003 | 2.95% | ≤ 15% | **PASS** |
| cusp depth rel err, d=0.025 | 3.75% | ≤ 15% | **PASS** |
| K abs diff (curvature) | 30.41 cm⁻¹ | ≤ 5 | FAIL |
| split-half spread | 6.08 cm⁻¹ | ≤ 5 | FAIL (marginal) |

Cusp depths: DFT-fit 0.664 / 16.653 cm⁻¹ vs deployed 0.684 / 17.301 (d = 0.003
/ 0.025) — the deployed cusp is ~3–4% deeper, comfortably inside 15%.
Composed-fit K = 1301.74 cm⁻¹ independently corroborates the curvature
reference (1300.21) to 1.5 cm⁻¹ on real DFT. Fit: cutoff-6 hiPhive, 36
structures (q6-subtracted), RMSE 399 meV/Å, branch tracking overlap 0.9999996.
`R2A_450k_dft_tdep_reference/acceptance.json` + `dft_reference.npz`.

## Closure verdict

1. **Composition machinery: validated.** Bit-exact replay on stored matrices;
   curvature estimator exact on synthetic linear labels; on real DFT it is
   corroborated by the independent composed fit to 1.5 cm⁻¹.
2. **Spectral-morphology claim: CONFIRMED at 450 K.** Kohn-anomaly cusp
   shape/depth reproduced by the deployed composition to ~1 cm⁻¹ / ~4%
   against fully independent DFT.
3. **Absolute-frequency claim: NOT confirmed at 450 K.** Deployed S0 K-A′ is
   systematically soft by **+30.4 ± 1.5 cm⁻¹ (2.4%)** vs the independent DFT
   thermal curvature; force-level MLIP-vs-DFT agreement is 7.4% RMSE (R1′).
   The offset exceeds the d=0.025 cusp depth (17.3 cm⁻¹), so absolute K
   frequencies from the composition should carry a ~±30 cm⁻¹ model-error bar
   at 450 K, while cusp *depths* and line shapes are good to a few percent.

Accuracy envelope of the deployed composition at 450 K, as measured by this
closure: **morphology ~1 cm⁻¹; absolute K ~30 cm⁻¹.**

## Incident log (execution honesty)

- v100ts ran shard_A + r2b_half1 concurrently on one V100: CUDA-context
  time-slicing starved shard_A (config 001: 1h26m CPU / **7h04m wall**, SCF
  itself normal at 15 iterations). half1 unaffected (10.5 h for 12 configs).
- Fix: SIGSTOP shard_A's tree, let half1 finish solo, auto-SIGCONT via tmux
  watcher. On resume the pw.x CUDA context wedged (D-state, 0% GPU, no SCF
  output for 18 min) → killed; label script's `valid_result` reuse meant the
  rerun cost only config 002. (Lesson: never SIGSTOP a GPU-QE process that
  shares a device; serialize batches on one GPU instead.)
