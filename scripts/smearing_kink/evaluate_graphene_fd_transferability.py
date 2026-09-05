#!/usr/bin/env python3
"""One-shot acceptance of the frozen 450 K P4 predictor against holdout targets."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
from bootstrap_graphene_fd_conditioned_sampling import moving_block_indices  # noqa: E402
from graphene_fd_p4_common import (  # noqa: E402
    CM_PER_THZ,
    P4_LINE_QPOINTS,
    apply_top_projector,
    atomic_json,
    atomic_npz,
    geometry_sha256,
    line_metrics,
    p4_key,
    percentile_higher,
    qpoint_fractional,
    rounded_quadratic_features,
    sha256,
)


def force_metrics(errors: np.ndarray) -> dict:
    values = np.asarray(errors, float).reshape(-1)
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(values**2)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(values)) * 1000.0),
    }


def qpoint_data(background: Path, force_constants: np.ndarray):
    phonon = fm.load_ph(background)
    t_values = [value for _, value in P4_LINE_QPOINTS]
    phonon.force_constants = np.asarray(force_constants, float)
    phonon.run_qpoints(qpoint_fractional(t_values), with_dynamical_matrices=True)
    result = phonon.get_qpoints_dict()
    frequencies = np.sort(np.asarray(result["frequencies"], float), axis=1) * CM_PER_THZ
    matrices = np.asarray(result["dynamical_matrices"], complex)
    return frequencies, matrices


def load_direct_dfpt(path: Path, expected_degauss: float) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(P4_LINE_QPOINTS):
        raise ValueError(f"expected 19 FD450_LINE rows, found {len(rows)}")
    lookup = {}
    for row in rows:
        if int(row["kgrid"]) != 144:
            raise ValueError("FD450_LINE contains a non-k144 row")
        if not np.isclose(float(row["degauss_Ry"]), expected_degauss, rtol=0.0, atol=1e-12):
            raise ValueError("FD450_LINE degauss differs from freeze")
        lookup[(row["region"], round(float(row["t_GK"]), 3))] = float(row["f6_cm-1"])
    expected = [(region, round(value, 3)) for region, value in P4_LINE_QPOINTS]
    if set(lookup) != set(expected):
        raise ValueError("FD450_LINE q points differ from the frozen list")
    return np.asarray([lookup[key] for key in expected], float)


def region_payload(prediction: np.ndarray, target: np.ndarray) -> dict:
    payload = {}
    for region in ("G", "K"):
        indices = [
            index for index, (observed, _) in enumerate(P4_LINE_QPOINTS) if observed == region
        ]
        t = [P4_LINE_QPOINTS[index][1] for index in indices]
        payload[region] = line_metrics(
            region, t, prediction[indices], target[indices]
        )
    return payload


def point_qspace_pass(regions: dict, thresholds: dict) -> bool:
    line_threshold = float(thresholds["Gamma_K_line_MAE_cm-1"])
    anchor_threshold = float(thresholds["Gamma_K_top_abs_error_cm-1"])
    kink_threshold = float(thresholds["K_kink_relative_error"])
    return bool(
        all(regions[region]["line_MAE_cm-1"] < line_threshold for region in ("G", "K"))
        and all(
            regions[region]["high_symmetry_abs_error_cm-1"] < anchor_threshold
            for region in ("G", "K")
        )
        and all(
            regions[region]["central_block_MAE_cm-1"] < line_threshold
            and regions[region]["central_block_high_symmetry_abs_error_cm-1"]
            < anchor_threshold
            for region in ("G", "K")
        )
        and regions["K"]["kink_relative_error"] < kink_threshold
    )


def qspace_bootstrap_metrics(prediction: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(targets, float)
    if target.ndim != 2 or target.shape[1] != len(P4_LINE_QPOINTS):
        raise ValueError("invalid P4 DFT-TDEP bootstrap target array")
    g = np.arange(0, 8)
    k = np.arange(8, 19)
    g_central = np.arange(0, 3)
    k_central = np.arange(12, 15)
    target_kink = np.abs(
        (target[:, 14] - target[:, 13]) / 0.008
        - (target[:, 13] - target[:, 12]) / 0.008
    )
    predicted_kink = abs(
        (prediction[14] - prediction[13]) / 0.008
        - (prediction[13] - prediction[12]) / 0.008
    )
    return {
        "G_line_MAE_cm_1": np.mean(np.abs(target[:, g] - prediction[g]), axis=1),
        "K_line_MAE_cm_1": np.mean(np.abs(target[:, k] - prediction[k]), axis=1),
        "G_high_symmetry_abs_error_cm_1": np.abs(target[:, 0] - prediction[0]),
        "K_high_symmetry_abs_error_cm_1": np.abs(target[:, 13] - prediction[13]),
        "G_central_block_MAE_cm_1": np.mean(
            np.abs(target[:, g_central] - prediction[g_central]), axis=1
        ),
        "K_central_block_MAE_cm_1": np.mean(
            np.abs(target[:, k_central] - prediction[k_central]), axis=1
        ),
        "K_kink_relative_error": np.abs(predicted_kink - target_kink)
        / np.maximum(np.abs(target_kink), 1.0e-12),
    }


def bootstrap_acceptance(raw: dict[str, np.ndarray], thresholds: dict) -> tuple[dict, bool]:
    threshold_by_metric = {
        "G_line_MAE_cm_1": float(thresholds["Gamma_K_line_MAE_cm-1"]),
        "K_line_MAE_cm_1": float(thresholds["Gamma_K_line_MAE_cm-1"]),
        "G_high_symmetry_abs_error_cm_1": float(
            thresholds["Gamma_K_top_abs_error_cm-1"]
        ),
        "K_high_symmetry_abs_error_cm_1": float(
            thresholds["Gamma_K_top_abs_error_cm-1"]
        ),
        "G_central_block_MAE_cm_1": float(thresholds["Gamma_K_line_MAE_cm-1"]),
        "K_central_block_MAE_cm_1": float(thresholds["Gamma_K_line_MAE_cm-1"]),
        "K_kink_relative_error": float(thresholds["K_kink_relative_error"]),
    }
    records = {}
    passed = True
    for name, values in raw.items():
        upper = percentile_higher(values)
        threshold = threshold_by_metric[name]
        metric_passed = upper < threshold
        passed = passed and metric_passed
        records[name] = {
            "median": float(np.median(values)),
            "bootstrap_95pct_upper": upper,
            "threshold": threshold,
            "passes_95pct_upper": metric_passed,
        }
    return records, passed


def add_failure(failures: list[dict], code: str, observed, threshold, comparison: str):
    failures.append(
        {
            "code": code,
            "observed": float(observed) if np.isscalar(observed) else observed,
            "threshold": float(threshold),
            "comparison": comparison,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--merge-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--static-prediction", type=Path, required=True)
    parser.add_argument("--dft-tdep", type=Path, required=True)
    parser.add_argument("--dft-tdep-summary", type=Path, required=True)
    parser.add_argument("--dft-bootstrap", type=Path, required=True)
    parser.add_argument("--dft-bootstrap-summary", type=Path, required=True)
    parser.add_argument("--short-tdep", type=Path, required=True)
    parser.add_argument("--temperature-law", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--sampling-acceptance", type=Path, required=True)
    parser.add_argument("--sampling-bootstrap", type=Path, required=True)
    parser.add_argument("--dfpt-convergence", type=Path, required=True)
    parser.add_argument("--dfpt-line", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=450003)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(args.freeze_manifest.read_text())
    merge = json.loads(args.merge_manifest.read_text())
    prediction_manifest = json.loads(args.prediction_manifest.read_text())
    dft_summary = json.loads(args.dft_tdep_summary.read_text())
    dft_bootstrap_summary = json.loads(args.dft_bootstrap_summary.read_text())
    sampling = json.loads(args.sampling_acceptance.read_text())
    sampling_bootstrap = json.loads(args.sampling_bootstrap.read_text())
    convergence = json.loads(args.dfpt_convergence.read_text())
    law = json.loads(args.temperature_law.read_text())
    thresholds = freeze["fixed_acceptance_thresholds"]
    n_replicates = int(thresholds["bootstrap_replicates"])
    temperature = int(freeze["T450_on_policy"]["temperature_K"])
    degauss = float(freeze["T450_on_policy"]["degauss_Ry"])

    freeze_sha = sha256(args.freeze_manifest)
    merge_sha = sha256(args.merge_manifest)
    if merge["freeze_manifest"]["sha256"] != freeze_sha:
        raise ValueError("merge/freeze mismatch")
    if prediction_manifest["inputs"]["freeze_manifest"]["sha256"] != freeze_sha:
        raise ValueError("prediction/freeze mismatch")
    if prediction_manifest["force_predictions"]["sha256"] != sha256(args.predictions):
        raise ValueError("force prediction file changed")
    if prediction_manifest["static_prediction"]["sha256"] != sha256(args.static_prediction):
        raise ValueError("static prediction file changed")
    if dft_summary["output"]["sha256"] != sha256(args.dft_tdep):
        raise ValueError("DFT-TDEP target file changed")
    if dft_summary["merge_manifest_sha256"] != merge_sha:
        raise ValueError("DFT-TDEP target used a different P4 merge")
    if dft_bootstrap_summary["raw_distributions"]["sha256"] != sha256(args.dft_bootstrap):
        raise ValueError("DFT-TDEP bootstrap file changed")
    if dft_bootstrap_summary["merge_manifest_sha256"] != merge_sha:
        raise ValueError("DFT-TDEP bootstrap used a different P4 merge")
    if dft_bootstrap_summary["point_tdep"]["sha256"] != sha256(args.dft_tdep):
        raise ValueError("DFT-TDEP bootstrap used a different point target")
    if freeze["temperature_law"]["sha256"] != sha256(args.temperature_law):
        raise ValueError("temperature law differs from freeze")
    if law.get("status") != "frozen_before_450_holdout":
        raise ValueError("temperature law was not frozen before P4")
    if int(law["prediction_temperature_K"]) != temperature:
        raise ValueError("temperature-law prediction temperature mismatch")
    if not np.isclose(law["prediction_degauss_Ry"], degauss, rtol=0.0, atol=1e-12):
        raise ValueError("temperature-law degauss mismatch")
    if sampling.get("status") != "passed" or not sampling.get("passes_sampling_gate"):
        raise ValueError("point-estimate sampling prerequisite did not pass")
    if sampling_bootstrap.get("status") != "passed" or not sampling_bootstrap.get(
        "passes_sampling_bootstrap_gate"
    ):
        raise ValueError("sampling bootstrap prerequisite did not pass")
    if sampling["freeze_manifest"]["sha256"] != freeze_sha:
        raise ValueError("sampling acceptance used a different freeze manifest")
    if sampling_bootstrap["freeze_manifest_sha256"] != freeze_sha:
        raise ValueError("sampling bootstrap used a different freeze manifest")
    if sampling_bootstrap["sampling_acceptance_sha256"] != sha256(
        args.sampling_acceptance
    ):
        raise ValueError("sampling bootstrap used a different point acceptance")
    if sampling["pooled_tdep"]["sha256"] != sha256(args.short_tdep):
        raise ValueError("frozen pooled short TDEP changed after sampling acceptance")
    if not np.isclose(convergence["degauss_Ry"], degauss, rtol=0.0, atol=1e-12):
        raise ValueError("DFPT convergence degauss differs from freeze")

    structures = read(args.labels, index=":")
    if len(structures) != 60 or merge["all_output"]["sha256"] != sha256(args.labels):
        raise ValueError("merged 60-label target changed")
    with np.load(args.predictions, allow_pickle=False) as data:
        predicted_keys = [str(value) for value in data["keys"]]
        predicted_geometry_hashes = [str(value) for value in data["geometry_sha256"]]
        predicted_total = np.asarray(data["predicted_total_forces_eV_A"], float)
        predicted_short = np.asarray(data["predicted_short_forces_eV_A"], float)
        predicted_long = np.asarray(data["predicted_long_range_forces_eV_A"], float)
    reference_forces = []
    observed_keys = []
    geometry_hashes = []
    for structure in structures:
        seed = int(structure.info["trajectory_seed"])
        index = int(structure.info["snapshot_index"])
        observed_keys.append(p4_key(seed, index))
        geometry_hashes.append(
            geometry_sha256(structure.numbers, structure.cell, structure.positions)
        )
        reference_forces.append(np.asarray(structure.arrays["REF_forces"], float))
    if predicted_keys != observed_keys or predicted_geometry_hashes != geometry_hashes:
        raise ValueError("frozen predictions do not match holdout geometries")
    reference_forces = np.asarray(reference_forces, float)
    errors = predicted_total - reference_forces
    primary_indices = set(
        int(value) for value in freeze["T450_on_policy"]["primary_force_indices_per_seed"]
    )
    primary_mask = np.asarray(
        [int(structure.info["snapshot_index"]) in primary_indices for structure in structures]
    )
    if int(np.count_nonzero(primary_mask)) != 15:
        raise ValueError("frozen P4 primary force subset is not 15 structures")
    point_primary = force_metrics(errors[primary_mask])
    point_all = force_metrics(errors)
    force_records = []
    for structure, error in zip(structures, errors, strict=True):
        force_records.append(
            {
                "key": p4_key(
                    int(structure.info["trajectory_seed"]),
                    int(structure.info["snapshot_index"]),
                ),
                **force_metrics(error[None, ...]),
            }
        )

    sampling_blocks = {
        int(item["seed"]): int(item["block_length_snapshots"])
        for item in sampling_bootstrap["block_bootstrap"]["trajectory_diagnostics"]
    }
    primary_order = sorted(primary_indices)
    primary_stride = int(np.gcd.reduce(np.diff(primary_order)))
    force_boot_rmse = np.full(n_replicates, np.nan)
    force_boot_max = np.full(n_replicates, np.nan)
    errors_by_seed = {}
    force_block_records = []
    for seed in (0, 1, 2):
        selected = [
            error
            for structure, error in zip(structures, errors, strict=True)
            if int(structure.info["trajectory_seed"]) == seed
            and int(structure.info["snapshot_index"]) in primary_indices
        ]
        errors_by_seed[seed] = np.asarray(selected, float)
        label_block = min(
            len(selected), max(1, math.ceil(sampling_blocks[seed] / primary_stride))
        )
        force_block_records.append(
            {
                "seed": seed,
                "sampling_block_length_saved_snapshots": sampling_blocks[seed],
                "primary_label_stride_saved_snapshots": primary_stride,
                "primary_label_block_length": label_block,
            }
        )
    for replicate in range(n_replicates):
        sampled = []
        for seed in (0, 1, 2):
            values = errors_by_seed[seed]
            block = force_block_records[seed]["primary_label_block_length"]
            indices = moving_block_indices(
                len(values), block, args.random_seed + replicate * 10 + seed
            )
            sampled.append(values[indices])
        metrics = force_metrics(np.concatenate(sampled, axis=0))
        force_boot_rmse[replicate] = metrics["RMSE_meV_A"]
        force_boot_max[replicate] = metrics["max_abs_meV_A"]
    force_bootstrap = {
        "RMSE_meV_A": {
            "median": float(np.median(force_boot_rmse)),
            "bootstrap_95pct_upper": percentile_higher(force_boot_rmse),
            "threshold": float(thresholds["force_RMSE_meV_A"]),
        },
        "max_abs_meV_A": {
            "median": float(np.median(force_boot_max)),
            "bootstrap_95pct_upper": percentile_higher(force_boot_max),
            "threshold": float(thresholds["force_max_abs_meV_A"]),
        },
    }
    for item in force_bootstrap.values():
        item["passes_95pct_upper"] = item["bootstrap_95pct_upper"] <= item["threshold"]
    force_point_pass = bool(
        point_primary["RMSE_meV_A"] <= thresholds["force_RMSE_meV_A"]
        and point_primary["max_abs_meV_A"] <= thresholds["force_max_abs_meV_A"]
    )
    force_bootstrap_pass = all(item["passes_95pct_upper"] for item in force_bootstrap.values())

    with np.load(args.short_tdep, allow_pickle=False) as data:
        short_fc = np.asarray(data[f"T{temperature}_fc2"], float)
    with np.load(args.static_prediction, allow_pickle=False) as data:
        static_full_fc = np.asarray(data["full_force_constants"], float)
    short_frequency, short_matrices = qpoint_data(args.background, short_fc)
    static_frequency, _ = qpoint_data(args.background, static_full_fc)
    finite_prediction = np.zeros(len(P4_LINE_QPOINTS), float)
    max_projector_error = 0.0
    for region in ("G", "K"):
        indices = [
            index for index, (observed, _) in enumerate(P4_LINE_QPOINTS) if observed == region
        ]
        t = np.asarray([P4_LINE_QPOINTS[index][1] for index in indices], float)
        distance = t if region == "G" else np.abs(t - 1.0)
        coefficients = np.asarray(
            law["predicted_coefficients_delta_lambda_cm-2"][region], float
        )
        delta_lambda = rounded_quadratic_features(distance, degauss) @ coefficients
        for local_index, global_index in enumerate(indices):
            corrected, scalar = apply_top_projector(
                short_matrices[global_index],
                delta_lambda[local_index],
                short_frequency[global_index],
                gamma=(region == "G"),
            )
            finite_prediction[global_index] = float(corrected[-1])
            max_projector_error = max(
                max_projector_error, abs(float(corrected[-1]) - scalar)
            )

    with np.load(args.dft_tdep, allow_pickle=False) as data:
        dft_point = np.asarray(data["T450_line_top_cm_1"], float)
        dft_regions = [str(value) for value in data["line_regions"]]
        dft_t = np.asarray(data["line_t_GK"], float)
    expected_regions = [region for region, _ in P4_LINE_QPOINTS]
    expected_t = np.asarray([value for _, value in P4_LINE_QPOINTS])
    if dft_regions != expected_regions or not np.allclose(dft_t, expected_t, atol=1e-12):
        raise ValueError("DFT-TDEP q-point order differs from freeze")
    finite_regions = region_payload(finite_prediction, dft_point)
    finite_point_pass = point_qspace_pass(finite_regions, thresholds)
    with np.load(args.dft_bootstrap, allow_pickle=False) as data:
        if int(data["replicates"]) != n_replicates:
            raise ValueError("DFT-TDEP bootstrap replicate count differs from freeze")
        dft_boot_targets = np.asarray(data["line_top_cm_1"], float)
    q_raw = qspace_bootstrap_metrics(finite_prediction, dft_boot_targets)
    q_bootstrap, finite_bootstrap_pass = bootstrap_acceptance(q_raw, thresholds)

    static_prediction = static_frequency[:, -1]
    direct_target = load_direct_dfpt(args.dfpt_line, degauss)
    static_regions = region_payload(static_prediction, direct_target)
    static_point_pass = point_qspace_pass(static_regions, thresholds)
    convergence_pass = bool(
        convergence.get("status") == "passed"
        and convergence.get("passes_convergence_gate")
    )
    sampling_point_pass = bool(
        sampling.get("status") == "passed" and sampling.get("passes_sampling_gate")
    )
    sampling_bootstrap_pass = bool(
        sampling_bootstrap.get("status") == "passed"
        and sampling_bootstrap.get("passes_sampling_bootstrap_gate")
    )
    matrix_pass = bool(
        max_projector_error < thresholds["matrix_projector_replay_max_abs_cm-1"]
    )

    force_status = (
        "passed"
        if force_point_pass and force_bootstrap_pass
        else ("insufficient_sampling" if force_point_pass else "failed")
    )
    force_acceptance = {
        "status": force_status,
        "scope": "frozen 450 K P4 total-force prediction on the fixed primary subset",
        "temperature_K": temperature,
        "degauss_Ry": degauss,
        "thresholds": {
            "force_RMSE_meV_A": thresholds["force_RMSE_meV_A"],
            "force_max_abs_meV_A": thresholds["force_max_abs_meV_A"],
        },
        "primary_subset": {
            "indices_per_seed": primary_order,
            "n_structures": 15,
            "point_estimate": point_primary,
            "bootstrap": force_bootstrap,
            "passes_point_gate": force_point_pass,
            "passes_bootstrap_gate": force_bootstrap_pass,
        },
        "all60_diagnostic": point_all,
        "bootstrap_method": {
            "replicates": n_replicates,
            "random_seed": args.random_seed,
            "trajectory_blocks": force_block_records,
        },
        "records": force_records,
        "prediction_decomposition_RMSE_meV_A": {
            "short": float(np.sqrt(np.mean(predicted_short**2)) * 1000.0),
            "long_range": float(np.sqrt(np.mean(predicted_long**2)) * 1000.0),
        },
        "passes_force_gate": bool(force_point_pass and force_bootstrap_pass),
        "inputs": {
            "labels": {"path": str(args.labels), "sha256": sha256(args.labels)},
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256(args.predictions),
            },
            "freeze_manifest": {
                "path": str(args.freeze_manifest),
                "sha256": freeze_sha,
            },
        },
    }
    atomic_json(args.output_dir / "force_acceptance.json", force_acceptance)

    prediction_status = (
        "passed"
        if finite_point_pass
        and finite_bootstrap_pass
        and static_point_pass
        and convergence_pass
        and matrix_pass
        else (
            "insufficient_sampling"
            if finite_point_pass
            and static_point_pass
            and convergence_pass
            and matrix_pass
            else "failed"
        )
    )
    prediction_acceptance = {
        "status": prediction_status,
        "scope": (
            "frozen 450 K q-space prediction: finite-lattice DFT-TDEP holdout plus "
            "an independent static-degauss DFPT diagnostic"
        ),
        "temperature_K": temperature,
        "degauss_Ry": degauss,
        "thresholds": {
            "Gamma_K_line_MAE_cm-1": thresholds["Gamma_K_line_MAE_cm-1"],
            "Gamma_K_top_abs_error_cm-1": thresholds["Gamma_K_top_abs_error_cm-1"],
            "K_kink_relative_error": thresholds["K_kink_relative_error"],
            "matrix_projector_replay_max_abs_cm-1": thresholds[
                "matrix_projector_replay_max_abs_cm-1"
            ],
        },
        "finite_lattice_temperature_prediction": {
            "method": (
                "apply the frozen 450 K rounded-cusp-plus-quadratic coefficients "
                "to the frozen pooled short-range MLIP-TDEP; no 450 K coefficient fit"
            ),
            "regions": finite_regions,
            "bootstrap": q_bootstrap,
            "passes_point_gate": finite_point_pass,
            "passes_bootstrap_gate": finite_bootstrap_pass,
        },
        "static_degauss_prediction": {
            "method": (
                "conditioned short-model finite-displacement FC2 plus the frozen "
                "interpolated sampling operator, compared with k=144 direct DFPT"
            ),
            "regions": static_regions,
            "passes_kgrid_convergence_gate": convergence_pass,
            "passes_static_degauss_gate": static_point_pass,
        },
        "matrix_projector_replay_max_abs_cm-1": max_projector_error,
        "passes_matrix_replay_gate": matrix_pass,
        "passes_prediction_gate": bool(
            finite_point_pass
            and finite_bootstrap_pass
            and static_point_pass
            and convergence_pass
            and matrix_pass
        ),
        "inputs": {
            "temperature_law": {
                "path": str(args.temperature_law),
                "sha256": sha256(args.temperature_law),
            },
            "short_tdep": {"path": str(args.short_tdep), "sha256": sha256(args.short_tdep)},
            "dft_tdep": {"path": str(args.dft_tdep), "sha256": sha256(args.dft_tdep)},
            "dft_tdep_bootstrap": {
                "path": str(args.dft_bootstrap),
                "sha256": sha256(args.dft_bootstrap),
            },
            "static_prediction": {
                "path": str(args.static_prediction),
                "sha256": sha256(args.static_prediction),
            },
            "dfpt_line": {"path": str(args.dfpt_line), "sha256": sha256(args.dfpt_line)},
            "dfpt_convergence": {
                "path": str(args.dfpt_convergence),
                "sha256": sha256(args.dfpt_convergence),
            },
        },
    }
    atomic_json(args.output_dir / "prediction_acceptance.json", prediction_acceptance)

    rows = []
    for index, (region, t_value) in enumerate(P4_LINE_QPOINTS):
        rows.append(
            {
                "temperature_K": temperature,
                "degauss_Ry": degauss,
                "region": region,
                "t_GK": t_value,
                "static_frozen_prediction_cm-1": static_prediction[index],
                "direct_DFPT_target_cm-1": direct_target[index],
                "finite_T_frozen_prediction_cm-1": finite_prediction[index],
                "DFT_TDEP_target_cm-1": dft_point[index],
            }
        )
    with (args.output_dir / "frozen_predictions_vs_targets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_npz(
        args.output_dir / "p4_metric_bootstrap.npz",
        force_RMSE_meV_A=force_boot_rmse,
        force_max_abs_meV_A=force_boot_max,
        **q_raw,
        replicates=np.array(n_replicates),
        force_random_seed=np.array(args.random_seed),
        dft_tdep_bootstrap_sha256=np.array(sha256(args.dft_bootstrap)),
    )

    failures = []
    if not sampling_point_pass:
        add_failure(failures, "sampling_point", 1, 0, "must pass")
    if not sampling_bootstrap_pass:
        add_failure(failures, "sampling_bootstrap", 1, 0, "must pass")
    if not convergence_pass:
        add_failure(failures, "dfpt_kgrid_convergence", 1, 0, "must pass")
    if point_primary["RMSE_meV_A"] > thresholds["force_RMSE_meV_A"]:
        add_failure(
            failures,
            "force_point_RMSE_meV_A",
            point_primary["RMSE_meV_A"],
            thresholds["force_RMSE_meV_A"],
            "<=",
        )
    if point_primary["max_abs_meV_A"] > thresholds["force_max_abs_meV_A"]:
        add_failure(
            failures,
            "force_point_max_abs_meV_A",
            point_primary["max_abs_meV_A"],
            thresholds["force_max_abs_meV_A"],
            "<=",
        )
    for name, record in force_bootstrap.items():
        if not record["passes_95pct_upper"]:
            add_failure(
                failures,
                f"force_bootstrap_{name}",
                record["bootstrap_95pct_upper"],
                record["threshold"],
                "<=",
            )
    for name, record in q_bootstrap.items():
        if not record["passes_95pct_upper"]:
            add_failure(
                failures,
                f"qspace_bootstrap_{name}",
                record["bootstrap_95pct_upper"],
                record["threshold"],
                "<",
            )
    if not finite_point_pass:
        add_failure(failures, "finite_T_qspace_point_gate", 1, 0, "must pass")
    if not static_point_pass:
        add_failure(failures, "static_degauss_point_gate", 1, 0, "must pass")
    if not matrix_pass:
        add_failure(
            failures,
            "matrix_projector_replay_max_abs_cm-1",
            max_projector_error,
            thresholds["matrix_projector_replay_max_abs_cm-1"],
            "<",
        )

    passes_c1 = bool(
        sampling_point_pass
        and sampling_bootstrap_pass
        and convergence_pass
        and force_point_pass
        and force_bootstrap_pass
        and finite_point_pass
        and finite_bootstrap_pass
        and static_point_pass
        and matrix_pass
    )
    insufficient = bool(
        not passes_c1
        and sampling_point_pass
        and convergence_pass
        and force_point_pass
        and finite_point_pass
        and static_point_pass
        and matrix_pass
    )
    overall_status = "passed_C1" if passes_c1 else (
        "insufficient_sampling" if insufficient else "failed_C1"
    )
    result = {
        "status": overall_status,
        "scope": "450 K frozen conditioned-model on-policy transferability holdout",
        "passes_C1": passes_c1,
        "passes_C2": False,
        "C2_status": "not_run",
        "component_gates": {
            "sampling_point": sampling_point_pass,
            "sampling_bootstrap": sampling_bootstrap_pass,
            "DFPT_kgrid_convergence": convergence_pass,
            "force_point": force_point_pass,
            "force_bootstrap": force_bootstrap_pass,
            "finite_T_qspace_point": finite_point_pass,
            "finite_T_qspace_bootstrap": finite_bootstrap_pass,
            "static_degauss_point": static_point_pass,
            "matrix_projector_replay": matrix_pass,
        },
        "failed_or_insufficient_metrics": failures,
        "artifacts": {
            "force_acceptance": {
                "path": str(args.output_dir / "force_acceptance.json"),
                "sha256": sha256(args.output_dir / "force_acceptance.json"),
            },
            "prediction_acceptance": {
                "path": str(args.output_dir / "prediction_acceptance.json"),
                "sha256": sha256(args.output_dir / "prediction_acceptance.json"),
            },
            "metric_bootstrap": {
                "path": str(args.output_dir / "p4_metric_bootstrap.npz"),
                "sha256": sha256(args.output_dir / "p4_metric_bootstrap.npz"),
            },
        },
        "allowed_conclusion": (
            "The frozen conditioned predictor passes the fixed 450 K on-policy C1 "
            "force, DFT-TDEP, and static-DFPT gates. This does not establish C2 on "
            "an independent physical-FD DFT-MD trajectory and does not validate the "
            "new physics-based degauss/lattice-temperature formulation."
            if passes_c1
            else (
                "Point estimates pass, but at least one fixed 95% bootstrap upper bound "
                "crosses its threshold; C1 is not registered."
                if insufficient
                else (
                    "At least one fixed 450 K P4 point-estimate gate fails; C1 is not "
                    "registered. The 450 K targets may be used only as development data "
                    "after this one-shot result is archived."
                )
            )
        ),
        "limitations": [
            "P4 structures come from frozen-model on-policy trajectories, not independent DFT-MD.",
            "The current predictor uses empirical endpoint interpolation; this is not the planned physical formula.",
            "C2 remains untested.",
        ],
        "freeze_manifest": {"path": str(args.freeze_manifest), "sha256": freeze_sha},
    }
    atomic_json(args.output_dir / "transferability_acceptance.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
