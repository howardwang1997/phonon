# Graphene low-smearing Kohn anomaly: q-space correction experiment plan

**Date:** 2026-07-23  
**Status:** started  
**Scope:** harmonic phonon spectra. A force/MD-capable infinite-lattice implementation is a
separate follow-up and is not required for the first acceptance gate.

## 1. Problem and falsifiable objective

The frozen 8x8 MLIP + real-space Friedel-template deployment reproduces its
finite-displacement DFT target, but it underestimates the converged direct-DFPT K cusp at
low Fermi-Dirac smearing:

- at `degauss=0.01 Ry`, direct DFPT gives a K slope jump of
  `21.985 THz/t`, while the primary frozen deployment gives `4.707 THz/t`;
- the K-centre error is `+37.692 cm^-1`;
- the direct `32x32` and `64x64` electron-grid line cuts agree to
  `0.093 cm^-1` MAE, so the discrepancy is not a k-grid artefact.

The working hypothesis is that the measured 8x8 template `D0(R)` is truncated before the
long-range Friedel tail has converged. Multiplying that same finite template by new
`B(T)` or `kappa(T)` values cannot create the missing tail.

**Objective:** add a symmetry-preserving non-analytic correction in reciprocal space and
pass a completely new direct-DFPT holdout without fitting to that holdout.

## 2. Locked data split

### Development data (may be inspected and fitted)

The existing direct-DFPT line cuts are development data:

- `degauss={0.01,0.04,0.08} Ry`, electron grid `32x32x1`;
- Gamma ray `t={0,0.015,0.030,0.050,0.080}`;
- K crossing `t={0.920,0.950,0.970,0.985,1.000,1.015,1.030,1.050,1.080}`;
- the existing `degauss=0.01 Ry`, `64x64x1` data are convergence evidence only.

The complete QE `gr.dyn` files, not only the printed frequencies, will be retained so that
the correction can be constructed and audited at the dynamical-matrix/eigenvector level.

The first three-temperature development fit showed that the q dependence can be
reproduced, but interpolation from `0.01/0.08 Ry` to `0.04 Ry` remained too inaccurate
(`11.00 cm^-1` at Gamma and `6.92 cm^-1` at K). Therefore two additional points are
declared **development data before any final holdout is run**:

- `degauss={0.02,0.06} Ry`, electron grid `32x32x1`;
- the same five Gamma and nine K q points as the existing development grid.

Candidate selection will use q-group holdout cross-validation at all five smearings and
leave-one-smearing-out interpolation tests at `0.02`, `0.04`, and `0.06 Ry`.

### Final direct-DFPT holdout (must not be inspected before freeze)

Use smearings not present in the current direct-DFPT campaign:

- `degauss={0.013,0.027,0.055} Ry`, electron grid `32x32x1`;
- shifted Gamma ray `t={0,0.012,0.025,0.045,0.075}`;
- shifted K crossing
  `t={0.925,0.955,0.975,0.990,1.000,1.010,1.025,1.045,1.075}`.

This is 42 primary q points. Add a five-point `64x64x1` low-smearing K check at
`degauss=0.013 Ry`, `t={0.975,0.990,1.000,1.010,1.025}`.

The holdout runner may be prepared before freeze, but the calculations are launched only
after a freeze manifest records the code hash, parameters, predictions and acceptance
thresholds.

## 3. Model to test

Start from the already frozen primary finite-displacement deployment and add

```text
D(q,T) = D_short(q,T) + Delta_D_K(q,T) + Delta_D_Gamma(q,T).
```

The correction is defined in squared-frequency/dynamical-matrix space, not by drawing a
frequency curve after diagonalisation.

1. Build mode projectors for the K `A1'` and Gamma `E2g` channels from the development
   `gr.dyn` eigenvectors and symmetrise over the K star / degenerate Gamma subspace.
2. Use a thermally rounded cusp kernel. Its small-q width is constrained to scale with
   electronic smearing, while a small number of amplitude/background parameters vary
   smoothly with `log(T_el)`.
3. Preserve Hermiticity, time-reversal symmetry, lattice point-group symmetry and the
   acoustic sum rule.
4. Compare at least these development candidates:
   - rounded-cusp projector correction;
   - a smooth even-polynomial residual control;
   - the uncorrected frozen deployment.
5. Select the form using only development-data q-point cross-validation and
   leave-one-smearing-out interpolation tests at `0.02`, `0.04`, and `0.06 Ry`.
   Complexity is selected before the final holdout is launched.

The existing scalar `tail_extrapolate.py` remains a diagnostic only. It is not accepted as
the final method because it predicts one kink number rather than a dynamical matrix.

## 4. Pre-registered acceptance criteria

The new holdout is a pass only if all of the following hold for the primary correction:

1. K-line highest-optical-branch MAE `< 5 cm^-1` at each holdout smearing.
2. K-centre absolute error `< 5 cm^-1` at each holdout smearing.
3. At `degauss=0.013 Ry`, K slope-jump relative error `< 20%`.
4. Gamma-line highest-optical-branch MAE `< 5 cm^-1` at each holdout smearing.
5. The correction does not increase the corresponding uncorrected line MAE by more than
   `1 cm^-1` at any holdout smearing.
6. The five-point `32x32`/`64x64` low-smearing K comparison differs by
   `< 1 cm^-1` MAE.

All metrics are reported whether the gate passes or fails. Thresholds are not changed
after the holdout starts.

## 5. Execution order and compute

1. Sync and checksum the existing development `gr.dyn` files.
2. Implement a strict QE-dynamical-matrix parser and reproduce all archived frequencies.
3. Run the supplemental `degauss=0.02/0.06 Ry` development DFPT points.
4. Implement the candidate q-space corrections and five-temperature cross-validation.
5. Freeze code, model parameters and predictions in
   `results/p0_graphene_qspace/freeze_manifest.json`.
6. Launch the 42-point `32x32` holdout and five-point `64x64` convergence subset under a
   restartable supervisor.
7. Evaluate once, write the pass/fail summary and update the weekly report.

Development and fitting run locally or on the idle RTX 2060. Direct DFPT uses the primitive
cell and can run at low CPU priority beside the checkpointed V100 DFT-MD jobs; it must not
pause or kill experiment 3. Based on the previous campaign, the holdout is expected to take
roughly 8--15 hours wall time when split over both V100 boxes, with some slowdown to DFT-MD.

## 6. Interpretation boundary

- Passing this plan supports a quantitative claim for harmonic graphene phonon spectra.
- It does not by itself make the correction transferable to arbitrary distorted structures
  or finite-temperature MD.
- A force-capable version requires an infinite-lattice/Ewald-like real-space
  implementation of the same analytic kernel plus off-equilibrium validation.
- A 10x10/12x12 finite-displacement campaign is an independent large-memory benchmark, not
  a prerequisite for the q-space harmonic correction.

## 7. Execution record

- `2026-07-23 20:11 +08`: launched `DEV_A` (`degauss=0.02 Ry`) on V100-A and
  `DEV_B` (`degauss=0.06 Ry`) on V100-B under independent restartable supervisors.
- Both supervisors use completed-file validation before declaring `DONE`; interrupted q
  points are rerun while completed q points and SCF output are reused.
- The final `HOLD_A/HOLD_B` runners have a hard guard and refuse to start until a
  `status=frozen_before_holdout` manifest explicitly authorises them.
- Existing experiment 3 jobs remain running; the new primitive-cell DFPT jobs use eight
  CPU ranks at `nice=10` and do not stop or restart experiment 3.

### Development revision R2 (before final holdout)

The supplemental `0.02/0.06 Ry` campaigns completed with 28/28 valid q points. Their
`gr.dyn` matrices pass Hermiticity, eigenvector-orthogonality, CSV-frequency, and
matrix-frequency replay checks.

The first five-temperature implementation interpolated the correction *residual* to the
frozen baseline. Although each q line fitted well, its leave-one-smearing-out MAE remained
`5.09 cm^-1` at Gamma and `5.87 cm^-1` at K. Inspection showed that the direct-DFPT
frequencies are smooth in smearing, while the residual is artificially non-smooth because
the frozen baseline has a piecewise interpolation anchor at `0.04 Ry`.

Before reading or launching any final holdout, the primary representation was therefore
changed to:

1. fit the absolute direct-DFPT highest-optical squared frequency with the
   thermally-rounded cusp;
2. use `epsilon=4*degauss` for both Gamma and K, selected within the two-parameter physical
   cusp family;
3. fit each cusp coefficient with a degree-three polynomial in `degauss`;
4. at runtime, subtract the frozen baseline eigenvalue and apply that difference as the
   Hermitian mode-projector correction.

This is algebraically still an additive correction to the frozen deployment, but avoids
forcing a smooth temperature law onto an artificial residual discontinuity. A smooth
quadratic q-control is reported but is not eligible as the primary model because it cannot
represent the Kohn-cusp claim.

Development-only validation after this revision:

| region | grouped-q CV MAE | leave-one-smearing-out MAE |
|---|---:|---:|
| Gamma | `0.069 cm^-1` | `0.917 cm^-1` |
| K | `0.400 cm^-1` | `0.309 cm^-1` |

The maximum disagreement among cubic, PCHIP, linear-`degauss`, and linear-`log(degauss)`
blind predictions at the three final smearings is `<3 cm^-1`. No final-holdout DFT result
was available or inspected during this revision.
