"""Absolute-frequency closure: K-channel recalibration of the deployed S0@450
composition, calibrated on 6 seeds, verified per the frozen design
(WEEKLY_2026-09-07 §4 / GRAPHENE_450K_CLOSURE_RESULT_2026-09-06.md).

Calibration scalar (frozen):
    dml = -(mean curvature_short over the first 6 seeds) - m*lambda_short(deployed)

Design v1 (frozen form, rank-1 real space):
    short_fc' = short_fc + dml * P (x) P / ||P||^2
EXECUTED AND FAILED the primary gate, with a clean physical diagnosis: a
rank-1 pattern that is a supercell Gamma-eigenvector is a pure K-star Bloch
state whose Fourier weight spreads over ~1/L (L = 6 cells, ~0.17 in 2pi/a
units) — wider than the whole +-0.04 dense window.  Both K-star members
absorb ~half the shift (Delta omega^2 ~ dml/2), the entire window lifts
~+13 cm-1, and the corrected K misses by 16.3 cm-1.  The same run showed
the true discrepancy profile: DFT-fit minus deployed is UNIFORM across the
window (+31.9 at K, RMS deviation from constant 0.63 — exactly what
K-referenced shape RMSE 0.63 means), i.e. the softness is a branch-uniform
A' offset, not a K-local one.

Design v2 (revision, documented as such — evidence-driven pivot, gates
unchanged): apply the calibration as a uniform mode-projected addition
along the tracked A' branch through the deployed response channel:
    corrected = apply_mode_projected_correction(
        short sequence, ..., response_cm2 + dml/m * scale_cm2)
The deployed composition's effective A' branch stiffness in the K window is
uniformly soft by dml; the recalibrated deployment carries Pi + dPi_calib
with dPi_calib = dml/m*scale.  No real-space fc object is modified, so
Gamma/M and the non-A' branches are untouched by construction; the
correction is scoped to this K-window A' composition.

Frozen verification (applied to v2; v1 recorded as failed):
  primary   |K_corrected - K_ref(12 seeds)| <= 5 cm-1   (original gate verbatim)
  general.  |K_corrected - K_ref(held-out 6)| vs split-half noise floor 6.08
  cusp      corrected cusp depths within 15% of deployed depths
  shape     non-K-referenced RMSE(corrected vs DFT fit) <= RMSE(uncorrected vs DFT fit)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import fit_graphene_450k_closure_tdep as fitter  # noqa: E402

R1DIR = fitter.R1DIR
TAGGED_NPZ = fitter.TAGGED_NPZ
OUTDIR = fitter.OUT
N_CALIB_SEEDS = 6
PRIMARY_GATE_CM1 = 5.0
CUSP_GATE = 0.15
SPLIT_HALF_NOISE_CM1 = 6.076162691321997  # measured, frozen upstream


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    # --- calibration data: per-seed q6-subtracted curvatures (24 labels) ---
    with np.load(fitter.OPERATOR, allow_pickle=False) as data:
        operator = {key: np.asarray(data[key]) for key in data.files}
    delta_fc = np.asarray(operator["delta_fc_full"], float)
    reference = np.asarray(operator["reference_positions"], float)
    cell = np.asarray(operator["cell"], float)

    curvature, _ = fitter.curvature_reference(delta_fc, reference, cell, None)
    per_seed = curvature["per_seed_curvature_eV_A2"]
    with np.load(TAGGED_NPZ, allow_pickle=False) as data:
        tagged = {key: np.asarray(data[key]) for key in data.files}
    pattern = np.asarray(tagged["k_aprime_pattern"], float)
    pattern_norm = float(tagged["pattern_norm2"].reshape(()))
    seeds = [int(s) for s in np.asarray(tagged["seed_sscha_indices"], int)]
    if len(seeds) != 12 or curvature["n_seeds"] != 12:
        raise ValueError("expected 12 tagged seeds")
    calib_seeds = seeds[:N_CALIB_SEEDS]
    heldout_seeds = seeds[N_CALIB_SEEDS:]
    calib_curv = np.array([per_seed[str(s)] for s in calib_seeds])

    # --- deployed S0@450 short fc and its projected K stiffness ---
    with np.load(fitter.DEPLOYED, allow_pickle=False) as data:
        deployed = {key: np.asarray(data[key]) for key in data.files}
    models = [str(m) for m in deployed["short_model"]]
    temperatures = np.asarray(deployed["lattice_temperature_K"], int)
    short_fc = np.asarray(deployed["short_force_constants_eV_A2"], float)[
        models.index("S0"), list(temperatures).index(450)
    ].copy()
    qpoints = np.asarray(deployed["qpoints"], float)
    signed_q = np.asarray(deployed["signed_q_2pi_over_a"], float)
    response = np.asarray(deployed["full_EPC_response_cm2"], float)
    deployed_s0 = np.asarray(deployed["full_EPC_frequency_cm1"], float)[
        models.index("S0"), list(temperatures).index(450)
    ]

    projected = float(
        (pattern * np.einsum("ijab,jb->ia", short_fc, pattern)).sum() / pattern_norm
    )
    dml = float(-calib_curv.mean() - projected)
    corrected_fc = short_fc + dml * np.einsum("ia,jb->ijab", pattern, pattern) / pattern_norm
    projected_corr = float(
        (pattern * np.einsum("ijab,jb->ia", corrected_fc, pattern)).sum() / pattern_norm
    )
    if abs(projected_corr - (projected + dml)) > 1.0e-10:
        raise ValueError("rank-1 update did not shift the projected stiffness exactly")

    # --- recompose through the deployed code path ---
    phonon, geometry_error = fitter.a0.make_phonopy(operator)
    if geometry_error > 1.0e-8:
        raise ValueError(f"operator geometry error {geometry_error:.2e} A")
    corrected_curve, diag = fitter.compose_aprime(
        phonon, corrected_fc, qpoints, signed_q, response
    )
    uncorrected_curve, diag = fitter.compose_aprime(
        phonon, short_fc, qpoints, signed_q, response
    )
    replay_err = float(np.max(np.abs(uncorrected_curve - deployed_s0)))
    if replay_err > 1.0e-6:
        raise ValueError(f"uncorrected replay mismatch {replay_err:.2e} cm-1")

    k_index = int(np.argmin(np.abs(signed_q)))
    k_ref_12 = curvature["omega_full_cm1"]
    k_ref_heldout = curvature["split_half_omega_full_cm1"][1]
    k_ref_calib = curvature["split_half_omega_full_cm1"][0]

    # --- design v2: uniform mode-projected addition along the tracked A' branch ---
    scale = float(tagged["scale_cm2"].reshape(()))
    mass = fitter.MASS_C_AMU
    dpi_calib_cm2 = dml / mass * scale
    corrected_curve_v2, diag_v2 = fitter.compose_aprime(
        phonon, short_fc, qpoints, signed_q, np.asarray(response, float) + dpi_calib_cm2
    )
    if diag_v2 != diag:  # same short sequence -> identical branch tracking
        raise ValueError("branch tracking changed under the response shift")

    def cusp_depths(curve: np.ndarray) -> dict[str, float]:
        depths = {}
        for offset in fitter.CUSP_OFFSETS:
            left = np.interp(-offset, signed_q, curve)
            right = np.interp(offset, signed_q, curve)
            depths[f"d{offset}"] = float(0.5 * (left + right) - curve[k_index])
        return depths

    corr_depths = cusp_depths(corrected_curve_v2)
    dep_depths = cusp_depths(deployed_s0)

    with np.load(OUTDIR / "dft_reference.npz", allow_pickle=False) as data:
        fit_curve = np.asarray(data["aprime_frequency_cm1"], float)

    def shape_rmse_nonk(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(((a - b) ** 2).mean()))

    def evaluate(curve: np.ndarray) -> dict:
        return {
            "K_cm1": float(curve[k_index]),
            "primary_abs_diff_vs_12seed_cm1": float(abs(curve[k_index] - k_ref_12)),
            "heldout_abs_diff_cm1": float(abs(curve[k_index] - k_ref_heldout)),
            "window_edge_shift_cm1": float(
                max(abs(curve[i] - deployed_s0[i]) for i in (0, len(signed_q) - 1))
            ),
            "cusp_rel_error": {
                key: abs(cusp_depths(curve)[key] - dep_depths[key]) / abs(dep_depths[key])
                for key in dep_depths
            },
            "rmse_nonkreferenced_vs_dft_fit": shape_rmse_nonk(curve, fit_curve),
        }

    v1_eval = evaluate(corrected_curve)
    v2_eval = evaluate(corrected_curve_v2)
    metrics = {
        "calibration": {
            "calib_seeds": calib_seeds,
            "heldout_seeds": heldout_seeds,
            "calib_mean_curvature_eV_A2": float(calib_curv.mean()),
            "calib_seed_std_eV_A2": float(calib_curv.std()),
            "deployed_projected_m_lambda_eV_A2": projected,
            "dml_eV_A2": dml,
            "dpi_calib_cm2": dpi_calib_cm2,
            "corrected_projected_m_lambda_eV_A2": projected_corr,
        },
        "K_references_cm1": {
            "uncorrected_deployed": float(deployed_s0[k_index]),
            "reference_12seed": k_ref_12,
            "reference_calib6": k_ref_calib,
            "reference_heldout6": k_ref_heldout,
        },
        "split_half_noise_floor_cm1": SPLIT_HALF_NOISE_CM1,
        "design_v1_rank1_real_space": {
            "status": "failed_primary_gate",
            "diagnosis": (
                "K-star Bloch Fourier width ~1/L (~0.17 in 2pi/a) exceeds the "
                "+-0.04 window; both star members absorb ~half the shift and "
                "the whole window lifts; corrected K misses by 16.3 cm-1"
            ),
            **v1_eval,
        },
        "design_v2_branch_uniform_response": {
            "status": "final",
            "note": (
                "uniform mode-projected addition along the tracked A' branch; "
                "Gamma/M and non-A' branches untouched by construction (no fc "
                "object modified)"
            ),
            **v2_eval,
        },
        "shape_vs_dft_fit": {
            "rmse_nonkreferenced_uncorrected": shape_rmse_nonk(deployed_s0, fit_curve),
        },
        "branch_tracking": {
            **diag,
            "note": "shared by v1/v2/uncorrected (same short sequence)",
        },
        "uncorrected_replay_max_abs_cm1": replay_err,
    }
    gates = {
        "primary_abs_diff_vs_12seed_le_5": v2_eval["primary_abs_diff_vs_12seed_cm1"] <= PRIMARY_GATE_CM1,
        "heldout_within_split_half_noise": v2_eval["heldout_abs_diff_cm1"] <= SPLIT_HALF_NOISE_CM1,
        "cusp_depth_le_15pct": all(v <= CUSP_GATE for v in v2_eval["cusp_rel_error"].values()),
        "shape_not_degraded": (
            v2_eval["rmse_nonkreferenced_vs_dft_fit"]
            <= metrics["shape_vs_dft_fit"]["rmse_nonkreferenced_uncorrected"]
        ),
    }

    payload = {
        "status": "all_gates_passed" if all(gates.values()) else "gate_failure",
        "scope": "K-channel recalibration of the deployed S0@450 composition; absolute-frequency closure",
        "frozen_design": "docs/WEEKLY_2026-09-07.md section 4 (committed 3cb7675 before execution)",
        "design_revision": (
            "v1 rank-1 real-space form failed (see metrics.design_v1_*); revised to v2 "
            "uniform branch-response form after diagnosing the supercell Fourier-width "
            "limit and the branch-uniform discrepancy profile; gates unchanged"
        ),
        "condition": {
            "deployed_matrices": str(fitter.DEPLOYED),
            "deployed_matrices_sha256": sha256(fitter.DEPLOYED),
            "tagged_pairs": str(TAGGED_NPZ),
            "tagged_pairs_sha256": sha256(TAGGED_NPZ),
            "labels": str(R1DIR / "labels" / "r2b_tagged_pairs"),
        },
        "metrics": metrics,
        "gates": gates,
    }
    np.savez(
        OUTDIR / "k_channel_calibration.npz",
        corrected_short_fc_v1_rank1_eV_A2=corrected_fc,
        dml_eV_A2=np.array(dml),
        dpi_calib_cm2=np.array(dpi_calib_cm2),
        corrected_aprime_v1_rank1_cm1=corrected_curve,
        corrected_aprime_v2_branch_uniform_cm1=corrected_curve_v2,
        qpoints=qpoints,
        signed_q_2pi_over_a=signed_q,
    )
    (OUTDIR / "k_channel_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"metrics": {k: v for k, v in metrics.items() if k != "branch_tracking"}, "gates": gates}, indent=2))
    print(f"wrote {OUTDIR / 'k_channel_calibration.json'}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
