"""Absolute-frequency closure at 300 K with the EXTENDED 24-seed reference.

Reads the frozen ``R2B_300k_tagged_pairs_ext24`` design (commit 8c3d79b, before
any new DFT label) plus the 24 reused labels of the 12-seed design, measures the
24-seed curvature reference, applies the amplitude-block-interleaved calib-12
scalar through the deployed compose path, and evaluates the frozen gate set:

  primary  |K_corrected - K_ref(24 seeds)| <= 5 cm-1
  heldout  |K_corrected - K_ref(held-out 12)| vs the interleaved split-half
           spread -- consistency readout, not pass/fail
  cusp     corrected cusp depths (d=0.003/0.025) within 15% of deployed S0@300
  in-force primary AND cusp pass -> ext24 dPi_calib IN FORCE at 300 K;
           on failure the deployment stays un-recalibrated

Extra readouts (frozen in the manifest): reference continuity vs the old
12-seed reference, dPi_ext24 vs the old 12-seed dPi and the 450 K constant,
and the 450 K-constant counterfactual against the 24-seed reference.

Synthetic smoke (``--smoke``): emulate BOTH label sets as the deployed
constituents' exact linear forces -- every per-seed curvature collapses to
-<pattern|short_fc|pattern>, the calibration is a no-op (dml ~ 0), the
corrected curve is bit-close to the deployed curve, primary/split-half ~ 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import fit_graphene_450k_closure_tdep as fitter  # noqa: E402

BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
TEMPERATURE = 300
PRIMARY_GATE_CM1 = 5.0
CUSP_GATE = 0.15
DPI_450K_CM2 = 70261.60368031562  # measured 2026-09-07, commit f23d33a


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_synthetic_labels(tagged_npz: Path, out_dir: Path, short_fc, delta_fc, reference, cell) -> None:
    """Emulate labels as the exact linear forces of the deployed constituents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(tagged_npz, allow_pickle=False) as data:
        positions_all = np.asarray(data["positions"], float)
    for local_index, positions in enumerate(positions_all):
        u = fitter.displacement_of(positions, reference, cell)
        forces = -(np.einsum("ijab,jb->ia", short_fc, u) + np.einsum("ijab,jb->ia", delta_fc, u))
        np.savez(out_dir / f"snapshot_{local_index:03d}_k8.npz", positions=positions, forces=forces)


def run(old_labels: Path, new_labels: Path, smoke: bool = False) -> int:
    old_dir = BASE / f"R2B_{TEMPERATURE}k_tagged_pairs"
    ext_dir = BASE / f"R2B_{TEMPERATURE}k_tagged_pairs_ext24"
    old_npz = old_dir / "r2b_tagged_pairs_snapshots.npz"
    ext_npz = ext_dir / "r2b_tagged_pairs_snapshots.npz"
    manifest = json.loads((ext_dir / "r2b_pair_ext24_manifest.json").read_text())

    with np.load(fitter.OPERATOR, allow_pickle=False) as data:
        operator = {key: np.asarray(data[key]) for key in data.files}
    delta_fc = np.asarray(operator["delta_fc_full"], float)
    reference = np.asarray(operator["reference_positions"], float)
    cell = np.asarray(operator["cell"], float)

    with np.load(fitter.DEPLOYED, allow_pickle=False) as data:
        deployed = {key: np.asarray(data[key]) for key in data.files}
    models = [str(m) for m in deployed["short_model"]]
    temperatures = [int(t) for t in deployed["lattice_temperature_K"]]
    mi, ti = models.index("S0"), temperatures.index(TEMPERATURE)
    short_fc = np.asarray(deployed["short_force_constants_eV_A2"], float)[mi, ti].copy()
    qpoints = np.asarray(deployed["qpoints"], float)
    signed_q = np.asarray(deployed["signed_q_2pi_over_a"], float)
    response = np.asarray(deployed["full_EPC_response_cm2"], float)
    deployed_s0 = np.asarray(deployed["full_EPC_frequency_cm1"], float)[mi, ti]

    old_labels = old_labels or (old_dir / "labels")
    new_labels = new_labels or (ext_dir / "labels")

    old_curvature, _ = fitter.curvature_reference(
        delta_fc, reference, cell, old_labels, tagged_npz=old_npz
    )
    ext_curvature, _ = fitter.curvature_reference(
        delta_fc, reference, cell, new_labels, tagged_npz=ext_npz
    )
    per_seed = dict(old_curvature["per_seed_curvature_eV_A2"])
    for seed, curvature in ext_curvature["per_seed_curvature_eV_A2"].items():
        if seed in per_seed:
            raise ValueError(f"seed {seed} appears in both label sets")
        per_seed[seed] = curvature

    seeds24 = [int(s) for s in manifest["seeds_amplitude_rank_order"]]
    calib = [int(s) for s in manifest["calib_split"]["calib_12"]]
    heldout = [int(s) for s in manifest["calib_split"]["heldout_12"]]
    if sorted(seeds24) != sorted(int(s) for s in per_seed) or len(seeds24) != 24:
        raise ValueError("the 24 manifest seeds do not match the measured labels")

    with np.load(ext_npz, allow_pickle=False) as data:
        pattern = np.asarray(data["k_aprime_pattern"], float)
        pattern_norm = float(data["pattern_norm2"].reshape(()))
        scale = float(data["scale_cm2"].reshape(()))
        pi_k = float(data["pi_K_cm2"].reshape(()))
    if not np.array_equal(pattern, np.asarray(
        np.load(old_npz, allow_pickle=False)["k_aprime_pattern"], float
    )):
        raise ValueError("P differs between the 12-seed and ext24 designs")

    def omega_full(seed_list: list[int]) -> float:
        lam = -float(np.mean([per_seed[str(s)] for s in seed_list])) / fitter.MASS_C_AMU
        arg = lam * scale + pi_k
        return float(np.sign(arg) * np.sqrt(abs(arg)))

    k_ref_24 = omega_full(seeds24)
    k_ref_calib = omega_full(calib)
    k_ref_heldout = omega_full(heldout)
    split_half = abs(k_ref_calib - k_ref_heldout)

    calib_curv = np.array([per_seed[str(s)] for s in calib])
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

    # counterfactual: the 450 K constant and the OLD 12-seed dPi at this temperature
    cf_450, _ = fitter.compose_aprime(
        phonon, short_fc, qpoints, signed_q, np.asarray(response, float) + DPI_450K_CM2
    )
    old_calibration = json.loads((old_dir / "k_channel_calibration.json").read_text())
    dpi_old12 = float(old_calibration["metrics"]["calibration"]["dpi_calib_cm2"])
    cf_old12, _ = fitter.compose_aprime(
        phonon, short_fc, qpoints, signed_q, np.asarray(response, float) + dpi_old12
    )

    k_corr = float(corrected_curve[k_index])
    metrics = {
        "lattice_temperature_K": TEMPERATURE,
        "calibration": {
            "calib_seeds": calib,
            "heldout_seeds": heldout,
            "calib_mean_curvature_eV_A2": float(calib_curv.mean()),
            "calib_seed_std_eV_A2": float(calib_curv.std()),
            "deployed_projected_m_lambda_eV_A2": projected,
            "dml_eV_A2": dml,
            "dpi_calib_cm2": dpi_calib_cm2,
        },
        "K_cm1": {
            "uncorrected_deployed": float(deployed_s0[k_index]),
            "reference_24seed": k_ref_24,
            "reference_calib12": k_ref_calib,
            "reference_heldout12": k_ref_heldout,
            "corrected": k_corr,
            "discrepancy_before_calibration_cm1": float(k_ref_24 - deployed_s0[k_index]),
        },
        "gates_raw": {
            "primary_abs_diff_vs_24seed_cm1": float(abs(k_corr - k_ref_24)),
            "heldout_abs_diff_cm1": float(abs(k_corr - k_ref_heldout)),
            "interleaved_split_half_spread_cm1": split_half,
            "cusp_rel_error": cusp_rel,
        },
        "reference_continuity": {
            "reference_12seed_old": float(old_calibration["metrics"]["K_cm1"]["reference_12seed"]),
            "reference_24seed_ext": k_ref_24,
            "shift_cm1": float(k_ref_24 - old_calibration["metrics"]["K_cm1"]["reference_12seed"]),
            "per_seed_std_eV_A2": float(np.std([per_seed[str(s)] for s in seeds24])),
        },
        "transferability_readout": {
            "dpi_calib_450K_cm2": DPI_450K_CM2,
            "dpi_calib_old12_cm2": dpi_old12,
            "ratio_ext24_over_450K": dpi_calib_cm2 / DPI_450K_CM2,
            "would_450K_constant_give_K_cm1": float(cf_450[k_index]),
            "450K_constant_abs_diff_vs_24seed_cm1": float(abs(cf_450[k_index] - k_ref_24)),
            "would_old12_dPi_give_K_cm1": float(cf_old12[k_index]),
            "old12_dPi_abs_diff_vs_24seed_cm1": float(abs(cf_old12[k_index] - k_ref_24)),
        },
        "branch_tracking": diag,
        "uncorrected_replay_max_abs_cm1": replay_err,
        "synthetic_smoke": smoke,
    }

    gates = {
        "primary_abs_diff_vs_24seed_le_5": metrics["gates_raw"]["primary_abs_diff_vs_24seed_cm1"] <= PRIMARY_GATE_CM1,
        "cusp_depth_le_15pct": all(v <= CUSP_GATE for v in cusp_rel.values()),
    }
    metrics["heldout_consistency_readout"] = {
        "heldout_abs_diff_cm1": metrics["gates_raw"]["heldout_abs_diff_cm1"],
        "interleaved_split_half_floor_cm1": split_half,
        "within_floor": metrics["gates_raw"]["heldout_abs_diff_cm1"] <= split_half,
    }
    in_force = all(gates.values()) and not smoke

    if smoke:
        if abs(dml) > 1.0e-9 or metrics["gates_raw"]["primary_abs_diff_vs_24seed_cm1"] > 1.0e-6:
            raise ValueError("synthetic smoke failed the no-op invariant")
        print("SMOKE PASS: dml = %.2e, primary = %.2e, split-half = %.2e"
              % (dml, metrics["gates_raw"]["primary_abs_diff_vs_24seed_cm1"], split_half))
        return 0

    payload = {
        "status": "in_force_300K_ext24" if in_force else "gate_failure_stays_unrecalibrated",
        "scope": (
            "K-channel recalibration of the deployed S0@300 composition against the "
            "extended 24-seed curvature reference (12 reused + 12 new labels, "
            "amplitude-block interleaved halves)"
        ),
        "frozen_design": (
            "R2B_300k_tagged_pairs_ext24/r2b_pair_ext24_manifest.json "
            "(committed 8c3d79b before any new DFT label was read)"
        ),
        "in_force_rule": manifest["frozen_gates"]["in_force_rule"],
        "condition": {
            "deployed_matrices": str(fitter.DEPLOYED),
            "deployed_matrices_sha256": sha256(fitter.DEPLOYED),
            "ext24_tagged_pairs": str(ext_npz),
            "ext24_tagged_pairs_sha256": sha256(ext_npz),
            "reused_labels": str(old_labels or (old_dir / "labels")),
            "new_labels": str(new_labels or (ext_dir / "labels")),
        },
        "metrics": metrics,
        "gates": gates,
    }
    np.savez(
        ext_dir / "k_channel_calibration.npz",
        corrected_aprime_branch_uniform_cm1=corrected_curve,
        counterfactual_450K_constant_cm1=cf_450,
        counterfactual_old12_dPi_cm1=cf_old12,
        dml_eV_A2=np.array(dml),
        dpi_calib_cm2=np.array(dpi_calib_cm2),
        per_seed_curvature_eV_A2=np.array([per_seed[str(s)] for s in seeds24]),
        seeds_amplitude_rank_order=np.asarray(seeds24, int),
        qpoints=qpoints,
        signed_q_2pi_over_a=signed_q,
    )
    (ext_dir / "k_channel_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: metrics[k] for k in ("K_cm1", "gates_raw", "reference_continuity", "transferability_readout")}, indent=2))
    print(json.dumps({"gates": gates, "in_force": in_force}, indent=2))
    print(f"wrote {ext_dir / 'k_channel_calibration.json'}")
    return 0 if in_force else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-labels", type=Path, default=None,
                        help="override the reused 12-seed label dir (smoke tests)")
    parser.add_argument("--new-labels", type=Path, default=None,
                        help="override the ext24 label dir (smoke tests)")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic end-to-end: emulate both label sets as the "
                             "deployed constituents' exact linear forces")
    args = parser.parse_args()

    if not args.smoke and (args.old_labels or args.new_labels):
        raise SystemExit("--old-labels/--new-labels are only valid together with --smoke")

    old_labels, new_labels = args.old_labels, args.new_labels
    if args.smoke:
        with np.load(fitter.OPERATOR, allow_pickle=False) as data:
            operator = {key: np.asarray(data[key]) for key in data.files}
        delta_fc = np.asarray(operator["delta_fc_full"], float)
        reference = np.asarray(operator["reference_positions"], float)
        cell = np.asarray(operator["cell"], float)
        with np.load(fitter.DEPLOYED, allow_pickle=False) as data:
            models = [str(m) for m in data["short_model"]]
            temperatures = [int(t) for t in data["lattice_temperature_K"]]
            mi, ti = models.index("S0"), temperatures.index(TEMPERATURE)
            short_fc = np.asarray(data["short_force_constants_eV_A2"], float)[mi, ti].copy()
        tmp = Path(tempfile.mkdtemp(prefix="ext24_smoke_"))
        try:
            old_dir = BASE / f"R2B_{TEMPERATURE}k_tagged_pairs"
            write_synthetic_labels(old_dir / "r2b_tagged_pairs_snapshots.npz",
                                   tmp / "old", short_fc, delta_fc, reference, cell)
            write_synthetic_labels(BASE / f"R2B_{TEMPERATURE}k_tagged_pairs_ext24/r2b_tagged_pairs_snapshots.npz",
                                   tmp / "new", short_fc, delta_fc, reference, cell)
            return run(tmp / "old", tmp / "new", smoke=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return run(old_labels, new_labels)


if __name__ == "__main__":
    raise SystemExit(main())
