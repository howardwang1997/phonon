"""Absolute-frequency closure at 300/600 K: curvature reference + K-channel
calibration + the dPi_calib(T_lat) transferability readout.

Temperature-parametric sibling of ``calibrate_graphene_450k_k_channel.py``
(v2 branch-uniform form only — the v1 rank-1 real-space form was executed,
failed, and retired at 450 K; see that script's docstring).  Reads the frozen
``R2B_{T}k_tagged_pairs`` design and its pulled DFT labels, measures the
deployed-vs-DFT absolute-K discrepancy, applies the calib-6 scalar through
the deployed compose path, and evaluates the frozen gate set:

  primary  |K_corrected - K_ref(12 seeds)| <= 5 cm-1
  heldout  |K_corrected - K_ref(held-out 6)| vs that temperature's split-half
           spread (consistency readout, not pass/fail)
  cusp     corrected cusp depths (d = 0.003/0.025) within 15% of deployed
  readout  dPi_calib(T) vs the 450 K constant +70261.60 cm-2

Synthetic-label invariant (verified): with linear labels of the deployed
constituents the calibration is a no-op (dml = 0, corrected == deployed).
"""
from __future__ import annotations

import argparse
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

BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
N_CALIB_SEEDS = 6
PRIMARY_GATE_CM1 = 5.0
CUSP_GATE = 0.15
DPI_450K_CM2 = 70261.60368031562  # measured 2026-09-07, commit f23d33a


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(temperature: int, labels_dir: Path | None) -> int:
    outdir = BASE / f"R2B_{temperature}k_tagged_pairs"
    tagged_npz = outdir / "r2b_tagged_pairs_snapshots.npz"
    label_dir = labels_dir or (outdir / "labels")

    with np.load(fitter.OPERATOR, allow_pickle=False) as data:
        operator = {key: np.asarray(data[key]) for key in data.files}
    delta_fc = np.asarray(operator["delta_fc_full"], float)
    reference = np.asarray(operator["reference_positions"], float)
    cell = np.asarray(operator["cell"], float)

    curvature, _ = fitter.curvature_reference(
        delta_fc, reference, cell, label_dir, tagged_npz=tagged_npz
    )
    per_seed = curvature["per_seed_curvature_eV_A2"]
    with np.load(tagged_npz, allow_pickle=False) as data:
        tagged = {key: np.asarray(data[key]) for key in data.files}
    pattern = np.asarray(tagged["k_aprime_pattern"], float)
    pattern_norm = float(tagged["pattern_norm2"].reshape(()))
    scale = float(tagged["scale_cm2"].reshape(()))
    seeds = [int(s) for s in np.asarray(tagged["seed_sscha_indices"], int)]
    if len(seeds) != 12 or curvature["n_seeds"] != 12:
        raise ValueError("expected 12 tagged seeds")
    calib_seeds, heldout_seeds = seeds[:N_CALIB_SEEDS], seeds[N_CALIB_SEEDS:]
    calib_curv = np.array([per_seed[str(s)] for s in calib_seeds])

    with np.load(fitter.DEPLOYED, allow_pickle=False) as data:
        deployed = {key: np.asarray(data[key]) for key in data.files}
    models = [str(m) for m in deployed["short_model"]]
    temperatures = [int(t) for t in deployed["lattice_temperature_K"]]
    mi, ti = models.index("S0"), temperatures.index(temperature)
    short_fc = np.asarray(deployed["short_force_constants_eV_A2"], float)[mi, ti].copy()
    qpoints = np.asarray(deployed["qpoints"], float)
    signed_q = np.asarray(deployed["signed_q_2pi_over_a"], float)
    response = np.asarray(deployed["full_EPC_response_cm2"], float)
    deployed_s0 = np.asarray(deployed["full_EPC_frequency_cm1"], float)[mi, ti]

    projected = float(
        (pattern * np.einsum("ijab,jb->ia", short_fc, pattern)).sum() / pattern_norm
    )
    dml = float(-calib_curv.mean() - projected)
    dpi_calib_cm2 = dml / fitter.MASS_C_AMU * scale

    phonon, geometry_error = fitter.a0.make_phonopy(operator)
    if geometry_error > 1.0e-8:
        raise ValueError(f"operator geometry error {geometry_error:.2e} A")
    uncorrected_curve, diag = fitter.compose_aprime(
        phonon, short_fc, qpoints, signed_q, response
    )
    replay_err = float(np.max(np.abs(uncorrected_curve - deployed_s0)))
    if replay_err > 1.0e-6:
        raise ValueError(f"uncorrected replay mismatch {replay_err:.2e} cm-1")
    corrected_curve, diag_corr = fitter.compose_aprime(
        phonon, short_fc, qpoints, signed_q, np.asarray(response, float) + dpi_calib_cm2
    )
    if diag_corr != diag:
        raise ValueError("branch tracking changed under the response shift")

    k_index = int(np.argmin(np.abs(signed_q)))

    def cusp_depths(curve: np.ndarray) -> dict[str, float]:
        depths = {}
        for offset in fitter.CUSP_OFFSETS:
            left = np.interp(-offset, signed_q, curve)
            right = np.interp(offset, signed_q, curve)
            depths[f"d{offset}"] = float(0.5 * (left + right) - curve[k_index])
        return depths

    dep_depths = cusp_depths(deployed_s0)
    corr_depths = cusp_depths(corrected_curve)
    cusp_rel = {k: abs(corr_depths[k] - dep_depths[k]) / abs(dep_depths[k]) for k in dep_depths}

    k_corr = float(corrected_curve[k_index])
    k_ref_12 = curvature["omega_full_cm1"]
    k_ref_calib = curvature["split_half_omega_full_cm1"][0]
    k_ref_heldout = curvature["split_half_omega_full_cm1"][1]
    split_half = curvature["split_half_spread_cm1"]

    metrics = {
        "lattice_temperature_K": temperature,
        "calibration": {
            "calib_seeds": calib_seeds,
            "heldout_seeds": heldout_seeds,
            "calib_mean_curvature_eV_A2": float(calib_curv.mean()),
            "calib_seed_std_eV_A2": float(calib_curv.std()),
            "deployed_projected_m_lambda_eV_A2": projected,
            "dml_eV_A2": dml,
            "dpi_calib_cm2": dpi_calib_cm2,
        },
        "K_cm1": {
            "uncorrected_deployed": float(deployed_s0[k_index]),
            "reference_12seed": k_ref_12,
            "reference_calib6": k_ref_calib,
            "reference_heldout6": k_ref_heldout,
            "corrected": k_corr,
            "discrepancy_before_calibration_cm1": float(k_ref_12 - deployed_s0[k_index]),
        },
        "gates_raw": {
            "primary_abs_diff_vs_12seed_cm1": float(abs(k_corr - k_ref_12)),
            "heldout_abs_diff_cm1": float(abs(k_corr - k_ref_heldout)),
            "split_half_spread_cm1": split_half,
            "cusp_rel_error": cusp_rel,
        },
        "transferability_readout": {
            "dpi_calib_450K_cm2": DPI_450K_CM2,
            "ratio_T_over_450K": dpi_calib_cm2 / DPI_450K_CM2,
            "would_450K_constant_give_K_cm1": None,
            "question": "does the 450 K calibration constant transfer, or is dPi_calib(T) needed per temperature",
        },
        "branch_tracking": diag,
        "uncorrected_replay_max_abs_cm1": replay_err,
    }

    # counterfactual: apply the 450 K constant at this temperature
    cf_curve, _ = fitter.compose_aprime(
        phonon, short_fc, qpoints, signed_q, np.asarray(response, float) + DPI_450K_CM2
    )
    metrics["transferability_readout"]["would_450K_constant_give_K_cm1"] = float(cf_curve[k_index])
    metrics["transferability_readout"]["450K_constant_abs_diff_vs_12seed_cm1"] = float(
        abs(cf_curve[k_index] - k_ref_12)
    )

    gates = {
        "primary_abs_diff_vs_12seed_le_5": metrics["gates_raw"]["primary_abs_diff_vs_12seed_cm1"] <= PRIMARY_GATE_CM1,
        "cusp_depth_le_15pct": all(v <= CUSP_GATE for v in cusp_rel.values()),
    }
    # frozen-design statistical hierarchy: the held-out residual is a
    # consistency READOUT against the split-half noise floor, not a gate
    # (a 6-seed reference's own spread exceeds the 5 cm-1 primary gate;
    # at zero synthetic noise even floating-point dust exceeds a ~1e-13 floor)
    metrics["heldout_consistency_readout"] = {
        "heldout_abs_diff_cm1": metrics["gates_raw"]["heldout_abs_diff_cm1"],
        "split_half_floor_cm1": split_half,
        "within_floor": metrics["gates_raw"]["heldout_abs_diff_cm1"] <= split_half,
    }

    payload = {
        "status": "all_gates_passed" if all(gates.values()) else "gate_failure",
        "scope": f"K-channel recalibration of the deployed S0@{temperature} composition; absolute-frequency closure, second/third temperature point",
        "frozen_design": f"R2B_{temperature}k_tagged_pairs/r2b_pair_manifest.json (committed 0034239 before any DFT label was read)",
        "condition": {
            "deployed_matrices": str(fitter.DEPLOYED),
            "deployed_matrices_sha256": sha256(fitter.DEPLOYED),
            "tagged_pairs": str(tagged_npz),
            "tagged_pairs_sha256": sha256(tagged_npz),
            "labels": str(label_dir),
        },
        "metrics": metrics,
        "gates": gates,
    }
    np.savez(
        outdir / "k_channel_calibration.npz",
        corrected_aprime_branch_uniform_cm1=corrected_curve,
        counterfactual_450K_constant_cm1=cf_curve,
        dml_eV_A2=np.array(dml),
        dpi_calib_cm2=np.array(dpi_calib_cm2),
        qpoints=qpoints,
        signed_q_2pi_over_a=signed_q,
    )
    (outdir / "k_channel_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: metrics[k] for k in ("K_cm1", "gates_raw", "transferability_readout")}, indent=2))
    print(json.dumps({"gates": gates}, indent=2))
    print(f"wrote {outdir / 'k_channel_calibration.json'}")
    return 0 if all(gates.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=int, required=True, choices=(300, 600))
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="override the label directory (smoke tests)")
    args = parser.parse_args()
    return run(args.temperature, args.labels_dir)


if __name__ == "__main__":
    raise SystemExit(main())
