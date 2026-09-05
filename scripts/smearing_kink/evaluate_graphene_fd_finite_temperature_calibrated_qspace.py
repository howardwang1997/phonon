#!/usr/bin/env python3
"""Cross-validate a lattice-temperature-calibrated q-space correction.

The static direct-DFPT transfer remains a separate control.  This evaluator is
used when that control demonstrates that its amplitude or intercept cannot be
transferred unchanged to a finite-lattice-temperature TDEP background.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_graphene_fd_finite_temperature_qspace as transfer  # noqa: E402


def calibration_features(
    distance: np.ndarray, degauss: float, include_quadratic: bool
) -> np.ndarray:
    """Rounded cusp plus an analytic finite-temperature curvature term."""
    distance = np.asarray(distance, float)
    rounded = transfer.features(distance, degauss)[:, 1]
    columns = [np.ones_like(distance), rounded]
    if include_quadratic:
        columns.append(distance**2)
    return np.column_stack(columns)


def fit_calibrated_delta_lambda(
    x, target_frequency, baseline_frequency, degauss, include_quadratic
):
    coefficients, *_ = np.linalg.lstsq(
        calibration_features(
            np.asarray(x, float), degauss, include_quadratic
        ),
        np.asarray(target_frequency, float) ** 2
        - np.asarray(baseline_frequency, float) ** 2,
        rcond=None,
    )
    return coefficients


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--degauss", type=float, required=True)
    parser.add_argument("--dfpt-line", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--thermal-short-tdep", type=Path, required=True)
    parser.add_argument("--dft-tdep", type=Path, required=True)
    parser.add_argument("--thermal-seed", type=Path, action="append", required=True)
    parser.add_argument("--force-selection", type=Path, required=True)
    parser.add_argument("--static-transfer-acceptance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope",
        default="finite-lattice-temperature calibrated q-space correction",
    )
    parser.add_argument("--thermal-mae-threshold", type=float, default=10.0)
    parser.add_argument("--high-symmetry-threshold", type=float, default=15.0)
    parser.add_argument("--kink-relative-threshold", type=float, default=0.20)
    parser.add_argument("--seed-mae-threshold", type=float, default=5.0)
    parser.add_argument("--mean-temperature-relative-threshold", type=float, default=0.20)
    parser.add_argument("--matrix-replay-threshold", type=float, default=1e-5)
    parser.add_argument(
        "--rounded-only-control",
        action="store_true",
        help="omit the analytic q^2 term for a non-selecting basis ablation",
    )
    args = parser.parse_args()
    include_quadratic = not args.rounded_only_control

    if len(args.thermal_seed) != 3:
        parser.error("exactly three --thermal-seed paths are required")
    force = json.loads(args.force_selection.read_text())
    force_passed = bool(force["passes_force_and_replay_gate"])
    static_control = json.loads(args.static_transfer_acceptance.read_text())
    static_regions = static_control["regions"]
    static_thresholds = static_control["thresholds"]
    static_support_passed = bool(
        all(
            static_regions[region]["direct_DFPT_group_CV_MAE_cm-1"]
            < static_thresholds["direct_DFPT_group_CV_MAE_cm-1"]
            for region in ("G", "K")
        )
        and static_regions["K"]["static_group_CV_kink_relative_error"]
        < static_thresholds["K_kink_relative_error"]
        and static_control["matrix_projector_replay_max_abs_cm-1"]
        < static_thresholds["matrix_projector_replay_max_abs_cm-1"]
    )

    rows = transfer.load_rows(args.dfpt_line)
    thermal_fc2 = transfer.load_fc2(
        args.thermal_short_tdep, f"T{args.temperature}_fc2"
    )
    dft_fc2 = transfer.load_fc2(args.dft_tdep, f"T{args.temperature}_fc2")
    seeds = [transfer.load_seed(path, args.temperature) for path in args.thermal_seed]
    seed_pairs = [
        {
            "seeds": [left_index, right_index],
            "full_band_MAE_cm-1": transfer.interpolated_mae(left, right),
        }
        for (left_index, left), (right_index, right) in combinations(
            enumerate(seeds), 2
        )
    ]
    temperature_errors = [
        abs(seed["mean_temperature_K"] - args.temperature) / args.temperature
        for seed in seeds
    ]
    seed_gate = bool(
        max(item["full_band_MAE_cm-1"] for item in seed_pairs)
        < args.seed_mae_threshold
        and max(temperature_errors) <= args.mean_temperature_relative_threshold
    )

    output_rows: list[dict[str, float | str]] = []
    metrics: dict[str, dict[str, float | int | bool | list[float]]] = {}
    max_projector_error = 0.0
    for region in ("G", "K"):
        selected = sorted(
            (row for row in rows if row["region"] == region),
            key=lambda row: float(row["t_GK"]),
        )
        t = np.asarray([float(row["t_GK"]) for row in selected], float)
        x = t if region == "G" else np.abs(t - 1.0)
        thermal_frequency, thermal_matrices = transfer.qpoint_data(
            args.background, thermal_fc2, t
        )
        dft_frequency, _ = transfer.qpoint_data(args.background, dft_fc2, t)
        thermal_top = thermal_frequency[:, -1]
        dft_top = dft_frequency[:, -1]
        calibrated_cv = np.zeros_like(dft_top)
        cv_delta_lambda = np.zeros_like(dft_top)

        for held_distance in sorted(set(np.round(x, 12))):
            held = np.isclose(x, held_distance)
            coefficients = fit_calibrated_delta_lambda(
                x[~held],
                dft_top[~held],
                thermal_top[~held],
                args.degauss,
                include_quadratic,
            )
            cv_delta_lambda[held] = (
                calibration_features(
                    x[held], args.degauss, include_quadratic
                )
                @ coefficients
            )
            for index in np.flatnonzero(held):
                scalar = np.sqrt(
                    max(thermal_top[index] ** 2 + cv_delta_lambda[index], 0.0)
                )
                projected = transfer.apply_projector(
                    thermal_matrices[index],
                    cv_delta_lambda[index],
                    thermal_frequency[index],
                    gamma=(region == "G"),
                )[-1]
                calibrated_cv[index] = projected
                max_projector_error = max(
                    max_projector_error, abs(float(projected) - float(scalar))
                )

        final_coefficients = fit_calibrated_delta_lambda(
            x, dft_top, thermal_top, args.degauss, include_quadratic
        )
        final_delta_lambda = (
            calibration_features(x, args.degauss, include_quadratic)
            @ final_coefficients
        )
        calibrated_final = np.asarray(
            [
                transfer.apply_projector(
                    thermal_matrices[index],
                    final_delta_lambda[index],
                    thermal_frequency[index],
                    gamma=(region == "G"),
                )[-1]
                for index in range(len(t))
            ],
            float,
        )
        central_limit = 0.010 if region == "G" else 0.008
        central_block = x <= central_limit + 1e-12
        central_coefficients = fit_calibrated_delta_lambda(
            x[~central_block],
            dft_top[~central_block],
            thermal_top[~central_block],
            args.degauss,
            include_quadratic,
        )
        central_delta_lambda = (
            calibration_features(
                x[central_block], args.degauss, include_quadratic
            )
            @ central_coefficients
        )
        central_predictions = np.asarray(
            [
                transfer.apply_projector(
                    thermal_matrices[index],
                    central_delta_lambda[local_index],
                    thermal_frequency[index],
                    gamma=(region == "G"),
                )[-1]
                for local_index, index in enumerate(np.flatnonzero(central_block))
            ],
            float,
        )
        central_target = dft_top[central_block]
        for index in range(len(t)):
            output_rows.append(
                {
                    "temperature_K": args.temperature,
                    "degauss_Ry": args.degauss,
                    "region": region,
                    "t_GK": float(t[index]),
                    "thermal_short_TDEP_cm-1": float(thermal_top[index]),
                    "thermal_calibrated_group_CV_cm-1": float(
                        calibrated_cv[index]
                    ),
                    "thermal_calibrated_final_fit_cm-1": float(
                        calibrated_final[index]
                    ),
                    "DFT_TDEP_cm-1": float(dft_top[index]),
                    "group_CV_delta_lambda_cm-2": float(cv_delta_lambda[index]),
                }
            )

        static_coefficients = np.asarray(
            static_regions[region]["rounded_cusp_coefficients_delta_lambda_cm-2"],
            float,
        )
        region_metric: dict[str, float | int | bool | list[float]] = {
            "n_points": len(t),
            "calibration_basis": (
                "rounded_cusp_plus_quadratic"
                if include_quadratic
                else "rounded_cusp_only_control"
            ),
            "calibration_coefficients_delta_lambda_cm-2": final_coefficients.tolist(),
            "static_control_rounded_cusp_coefficients_delta_lambda_cm-2": (
                static_coefficients.tolist()
            ),
            "finite_T_to_static_intercept_ratio": float(
                final_coefficients[0] / static_coefficients[0]
            ),
            "finite_T_to_static_rounded_cusp_ratio": float(
                final_coefficients[1] / static_coefficients[1]
            ),
            "calibration_target": "finite-lattice-temperature DFT-TDEP",
            "DFT_TDEP_thermal_short_MAE_cm-1": float(
                np.mean(np.abs(thermal_top - dft_top))
            ),
            "DFT_TDEP_thermal_calibrated_group_CV_MAE_cm-1": float(
                np.mean(np.abs(calibrated_cv - dft_top))
            ),
            "DFT_TDEP_thermal_calibrated_group_CV_max_abs_cm-1": float(
                np.max(np.abs(calibrated_cv - dft_top))
            ),
            "DFT_TDEP_thermal_calibrated_final_fit_MAE_cm-1": float(
                np.mean(np.abs(calibrated_final - dft_top))
            ),
            "central_block_CV_max_distance_from_anchor": central_limit,
            "central_block_CV_n_points": int(np.count_nonzero(central_block)),
            "central_block_CV_MAE_cm-1": float(
                np.mean(np.abs(central_predictions - central_target))
            ),
            "central_block_CV_max_abs_cm-1": float(
                np.max(np.abs(central_predictions - central_target))
            ),
        }
        anchor_t = 0.0 if region == "G" else 1.0
        anchor_index = int(np.argmin(np.abs(t - anchor_t)))
        region_metric["high_symmetry_DFT_TDEP_cm-1"] = float(dft_top[anchor_index])
        region_metric["high_symmetry_thermal_calibrated_group_CV_cm-1"] = float(
            calibrated_cv[anchor_index]
        )
        region_metric["high_symmetry_abs_error_cm-1"] = float(
            abs(calibrated_cv[anchor_index] - dft_top[anchor_index])
        )
        central_indices = np.flatnonzero(central_block)
        central_anchor_local_index = int(
            np.argmin(np.abs(t[central_indices] - anchor_t))
        )
        region_metric["central_block_CV_high_symmetry_abs_error_cm-1"] = float(
            abs(
                central_predictions[central_anchor_local_index]
                - dft_top[anchor_index]
            )
        )
        if region == "K":
            dft_lookup = {round(value, 3): freq for value, freq in zip(t, dft_top)}
            calibrated_lookup = {
                round(value, 3): freq for value, freq in zip(t, calibrated_cv)
            }
            dft_kink = transfer.kink(dft_lookup)
            calibrated_kink = transfer.kink(calibrated_lookup)
            central_lookup = {
                round(float(t[index]), 3): float(central_predictions[local_index])
                for local_index, index in enumerate(central_indices)
            }
            central_kink = transfer.kink(central_lookup)
            region_metric.update(
                {
                    "DFT_TDEP_kink_cm-1_per_t": dft_kink,
                    "thermal_calibrated_group_CV_kink_cm-1_per_t": calibrated_kink,
                    "thermal_calibrated_group_CV_kink_relative_error": abs(
                        calibrated_kink - dft_kink
                    )
                    / max(dft_kink, 1e-12),
                    "central_block_CV_kink_cm-1_per_t": central_kink,
                    "central_block_CV_kink_relative_error": abs(
                        central_kink - dft_kink
                    )
                    / max(dft_kink, 1e-12),
                }
            )
        metrics[region] = region_metric

    calibrated_qspace_pass = bool(
        all(
            metrics[region]["DFT_TDEP_thermal_calibrated_group_CV_MAE_cm-1"]
            < args.thermal_mae_threshold
            for region in ("G", "K")
        )
        and all(
            metrics[region]["high_symmetry_abs_error_cm-1"]
            < args.high_symmetry_threshold
            for region in ("G", "K")
        )
        and all(
            metrics[region]["central_block_CV_MAE_cm-1"]
            < args.thermal_mae_threshold
            and metrics[region]["central_block_CV_high_symmetry_abs_error_cm-1"]
            < args.high_symmetry_threshold
            for region in ("G", "K")
        )
        and metrics["K"]["thermal_calibrated_group_CV_kink_relative_error"]
        < args.kink_relative_threshold
        and metrics["K"]["central_block_CV_kink_relative_error"]
        < args.kink_relative_threshold
        and max_projector_error < args.matrix_replay_threshold
    )
    passes_all = bool(
        force_passed and seed_gate and static_support_passed and calibrated_qspace_pass
    )
    basis_description = (
        "rounded-cusp plus analytic quadratic"
        if include_quadratic
        else "rounded-cusp-only control"
    )
    result = {
        "status": "passed" if passes_all else "failed",
        "scope": args.scope,
        "method": (
            f"fit a lattice-temperature-specific squared-frequency {basis_description} "
            "correction from short-range MLIP-TDEP to DFT-TDEP with equal-distance "
            "q-group leave-out cross-validation and "
            "a separate central-three-point block extrapolation; static direct-DFPT "
            "transfer is retained as a control and static-basis support check"
        ),
        "temperature_K": args.temperature,
        "degauss_Ry": args.degauss,
        "thresholds": {
            "DFT_TDEP_thermal_calibrated_group_CV_MAE_cm-1": (
                args.thermal_mae_threshold
            ),
            "Gamma_K_top_abs_error_cm-1": args.high_symmetry_threshold,
            "K_kink_relative_error": args.kink_relative_threshold,
            "central_block_CV_MAE_cm-1": args.thermal_mae_threshold,
            "central_block_Gamma_K_top_abs_error_cm-1": (
                args.high_symmetry_threshold
            ),
            "central_block_K_kink_relative_error": args.kink_relative_threshold,
            "seed_pair_full_band_MAE_cm-1": args.seed_mae_threshold,
            "MD_mean_temperature_relative_error": (
                args.mean_temperature_relative_threshold
            ),
            "matrix_projector_replay_max_abs_cm-1": args.matrix_replay_threshold,
        },
        "inputs": {
            "dfpt_line": {
                "path": str(args.dfpt_line),
                "sha256": transfer.sha256(args.dfpt_line),
            },
            "background": {
                "path": str(args.background),
                "sha256": transfer.sha256(args.background),
            },
            "thermal_short_tdep": {
                "path": str(args.thermal_short_tdep),
                "sha256": transfer.sha256(args.thermal_short_tdep),
            },
            "dft_tdep_calibration_target": {
                "path": str(args.dft_tdep),
                "sha256": transfer.sha256(args.dft_tdep),
            },
            "force_selection": {
                "path": str(args.force_selection),
                "sha256": transfer.sha256(args.force_selection),
            },
            "static_transfer_control": {
                "path": str(args.static_transfer_acceptance),
                "sha256": transfer.sha256(args.static_transfer_acceptance),
            },
        },
        "force_gate_passed": force_passed,
        "static_direct_DFPT_basis_support_passed": static_support_passed,
        "static_transfer_control_passed": bool(
            static_control["passes_qspace_transfer_gate"]
        ),
        "thermal_seed_files": [
            {"path": seed["path"], "sha256": seed["sha256"]} for seed in seeds
        ],
        "thermal_seed_pair_spread": seed_pairs,
        "thermal_seed_temperatures": [
            {
                "seed": index,
                "mean_temperature_K": seed["mean_temperature_K"],
                "max_temperature_K": seed["max_temperature_K"],
                "mean_temperature_relative_error": temperature_errors[index],
            }
            for index, seed in enumerate(seeds)
        ],
        "passes_seed_stability_gate": seed_gate,
        "regions": metrics,
        "matrix_projector_replay_max_abs_cm-1": max_projector_error,
        "passes_finite_temperature_calibration_gate": calibrated_qspace_pass,
        "passes_all_force_seed_calibrated_qspace_gates": passes_all,
        "limitation": (
            "The DFT-TDEP force constants at this lattice temperature are a "
            "calibration target. Group CV tests interpolation across held q groups, "
            "not transfer to an independent DFT-MD trajectory or lattice temperature."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "finite_temperature_calibrated_predictions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary_path = args.output_dir / "acceptance.json"
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, summary_path)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for axis, region in zip(axes, ("G", "K")):
        selected_rows = sorted(
            (row for row in output_rows if row["region"] == region),
            key=lambda row: float(row["t_GK"]),
        )
        t = [float(row["t_GK"]) for row in selected_rows]
        axis.plot(
            t,
            [float(row["DFT_TDEP_cm-1"]) for row in selected_rows],
            "o-",
            color="#0072B2",
            label="DFT-TDEP target",
        )
        axis.plot(
            t,
            [float(row["thermal_short_TDEP_cm-1"]) for row in selected_rows],
            ":",
            color="#D55E00",
            label="short-range MLIP-TDEP",
        )
        axis.plot(
            t,
            [
                float(row["thermal_calibrated_group_CV_cm-1"])
                for row in selected_rows
            ],
            "--",
            color="#009E73",
            label="finite-T LR calibration (group CV)",
        )
        axis.set_title(r"$\Gamma$ line" if region == "G" else "K line")
        axis.set_xlabel("reduced Γ–K coordinate")
        axis.set_ylabel("top optical frequency (cm$^{-1}$)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=1,
    )
    fig.suptitle(
        f"{args.temperature} K finite-temperature calibrated q-space correction"
    )
    fig.tight_layout(rect=(0.0, 0.20, 1.0, 0.94))
    fig.savefig(
        args.output_dir / "finite_temperature_calibrated_comparison.png", dpi=180
    )
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
