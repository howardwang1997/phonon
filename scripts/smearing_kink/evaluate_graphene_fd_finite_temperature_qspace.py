#!/usr/bin/env python3
"""Validate static q-space LR transfer onto a finite-lattice-temperature TDEP background."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import friedel_module as fm  # noqa: E402


CM = 33.35641


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def features(distance: np.ndarray, degauss: float) -> np.ndarray:
    epsilon = 4.0 * degauss
    rounded = np.sqrt(distance * distance + epsilon * epsilon) - epsilon
    return np.column_stack((np.ones_like(distance), rounded))


def fit_delta_lambda(x, direct_frequency, static_short_frequency, degauss):
    coefficients, *_ = np.linalg.lstsq(
        features(np.asarray(x, float), degauss),
        np.asarray(direct_frequency, float) ** 2
        - np.asarray(static_short_frequency, float) ** 2,
        rcond=None,
    )
    return coefficients


def corrected_frequency(baseline, x, coefficients, degauss):
    squared = np.asarray(baseline, float) ** 2 + features(
        np.asarray(x, float), degauss
    ) @ coefficients
    return np.sqrt(np.maximum(squared, 0.0))


def apply_projector(matrix, delta_lambda_cm2, frequencies_cm, gamma):
    hermitian = (matrix + matrix.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    mask = (eigenvalues > 1e-10) & (frequencies_cm > 1e-6)
    scale = float(np.median(frequencies_cm[mask] ** 2 / eigenvalues[mask]))
    correction = np.zeros_like(hermitian)
    for index in ((-2, -1) if gamma else (-1,)):
        vector = eigenvectors[:, index]
        correction += (delta_lambda_cm2 / scale) * np.outer(vector, vector.conj())
    corrected_eigenvalues = np.linalg.eigvalsh(hermitian + correction)
    return np.sign(corrected_eigenvalues) * np.sqrt(
        np.abs(corrected_eigenvalues) * scale
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 19:
        raise ValueError(f"expected 19 dense-line rows in {path}, found {len(rows)}")
    return rows


def load_fc2(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        result = np.asarray(data[key], float)
    if result.ndim != 4 or not np.isfinite(result).all():
        raise ValueError(f"invalid {key} in {path}")
    return result


def qpoint_data(background: Path, force_constants, t_values):
    phonon = fm.load_ph(background)
    phonon.force_constants = np.asarray(force_constants, float)
    qpoints = np.array([[value / 3.0, value / 3.0, 0.0] for value in t_values])
    phonon.run_qpoints(qpoints, with_dynamical_matrices=True)
    result = phonon.get_qpoints_dict()
    frequencies = np.sort(np.asarray(result["frequencies"], float), axis=1) * CM
    matrices = np.asarray(result["dynamical_matrices"], complex)
    return frequencies, matrices


def load_seed(path: Path, temperature: int) -> dict:
    prefix = f"T{temperature}"
    with np.load(path, allow_pickle=False) as data:
        return {
            "path": str(path),
            "sha256": sha256(path),
            "distance": np.asarray(data[f"{prefix}_dist"], float),
            "frequency": np.asarray(data[f"{prefix}_freq"], float) * CM,
            "mean_temperature_K": float(data[f"{prefix}_mean_temperature_K"]),
            "max_temperature_K": float(data[f"{prefix}_max_temperature_K"]),
        }


def interpolated_mae(left: dict, right: dict) -> float:
    right_frequency = np.stack(
        [
            np.interp(left["distance"], right["distance"], right["frequency"][:, branch])
            for branch in range(right["frequency"].shape[1])
        ],
        axis=1,
    )
    return float(np.mean(np.abs(right_frequency - left["frequency"])))


def kink(values: dict[float, float]) -> float:
    dx = 0.008
    return abs(
        (values[1.008] - values[1.000]) / dx
        - (values[1.000] - values[0.992]) / dx
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--degauss", type=float, required=True)
    parser.add_argument("--dfpt-line", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--static-short-fc2", type=Path, required=True)
    parser.add_argument("--thermal-short-tdep", type=Path, required=True)
    parser.add_argument("--dft-tdep", type=Path, required=True)
    parser.add_argument("--thermal-seed", type=Path, action="append", required=True)
    parser.add_argument("--force-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope",
        default="finite-temperature q-space transfer validation",
    )
    parser.add_argument("--direct-cv-mae-threshold", type=float, default=5.0)
    parser.add_argument("--thermal-mae-threshold", type=float, default=10.0)
    parser.add_argument("--high-symmetry-threshold", type=float, default=15.0)
    parser.add_argument("--kink-relative-threshold", type=float, default=0.20)
    parser.add_argument("--seed-mae-threshold", type=float, default=5.0)
    parser.add_argument("--mean-temperature-relative-threshold", type=float, default=0.20)
    args = parser.parse_args()

    if len(args.thermal_seed) != 3:
        parser.error("exactly three --thermal-seed paths are required")
    force = json.loads(args.force_selection.read_text())
    force_passed = bool(force["passes_force_and_replay_gate"])
    rows = load_rows(args.dfpt_line)
    static_fc2 = load_fc2(args.static_short_fc2, "force_constants")
    thermal_fc2 = load_fc2(args.thermal_short_tdep, f"T{args.temperature}_fc2")
    dft_fc2 = load_fc2(args.dft_tdep, f"T{args.temperature}_fc2")

    seeds = [load_seed(path, args.temperature) for path in args.thermal_seed]
    seed_pairs = [
        {
            "seeds": [left_index, right_index],
            "full_band_MAE_cm-1": interpolated_mae(left, right),
        }
        for (left_index, left), (right_index, right) in combinations(
            enumerate(seeds), 2
        )
    ]
    temperature_errors = [
        abs(seed["mean_temperature_K"] - args.temperature) / args.temperature
        for seed in seeds
    ]
    seed_gate = (
        max(item["full_band_MAE_cm-1"] for item in seed_pairs)
        < args.seed_mae_threshold
        and max(temperature_errors) <= args.mean_temperature_relative_threshold
    )

    output_rows = []
    metrics = {}
    max_projector_error = 0.0
    for region in ("G", "K"):
        selected = sorted(
            (row for row in rows if row["region"] == region),
            key=lambda row: float(row["t_GK"]),
        )
        t = np.asarray([float(row["t_GK"]) for row in selected], float)
        x = t if region == "G" else np.abs(t - 1.0)
        direct = np.asarray([float(row["f6_cm-1"]) for row in selected], float)
        static_frequency, static_matrices = qpoint_data(args.background, static_fc2, t)
        thermal_frequency, thermal_matrices = qpoint_data(args.background, thermal_fc2, t)
        dft_frequency, _ = qpoint_data(args.background, dft_fc2, t)
        static_top = static_frequency[:, -1]
        thermal_top = thermal_frequency[:, -1]
        dft_top = dft_frequency[:, -1]

        direct_cv = np.zeros_like(direct)
        thermal_cv = np.zeros_like(direct)
        thermal_anchor_cv = np.zeros_like(direct)
        cv_delta_lambda = np.zeros_like(direct)
        cv_transfer_delta_lambda = np.zeros_like(direct)
        for held_distance in sorted(set(np.round(x, 12))):
            held = np.isclose(x, held_distance)
            coefficients = fit_delta_lambda(
                x[~held], direct[~held], static_top[~held], args.degauss
            )
            direct_cv[held] = corrected_frequency(
                static_top[held], x[held], coefficients, args.degauss
            )
            cv_delta_lambda[held] = (
                features(x[held], args.degauss) @ coefficients
            )
            cv_transfer_delta_lambda[held] = (
                cv_delta_lambda[held] - coefficients[0]
            )
            for index in np.flatnonzero(held):
                scalar_frequency = np.sqrt(
                    max(
                        thermal_top[index] ** 2
                        + cv_delta_lambda[index],
                        0.0,
                    )
                )
                projected_frequency = apply_projector(
                    thermal_matrices[index],
                    cv_delta_lambda[index],
                    thermal_frequency[index],
                    gamma=(region == "G"),
                )[-1]
                thermal_cv[index] = projected_frequency
                thermal_anchor_cv[index] = apply_projector(
                    thermal_matrices[index],
                    cv_transfer_delta_lambda[index],
                    thermal_frequency[index],
                    gamma=(region == "G"),
                )[-1]
                max_projector_error = max(
                    max_projector_error,
                    abs(float(projected_frequency) - float(scalar_frequency)),
                )
        final_coefficients = fit_delta_lambda(
            x, direct, static_top, args.degauss
        )
        direct_final = corrected_frequency(
            static_top, x, final_coefficients, args.degauss
        )
        final_delta_lambda = features(x, args.degauss) @ final_coefficients
        thermal_final = np.asarray(
            [
                apply_projector(
                    thermal_matrices[index],
                    final_delta_lambda[index],
                    thermal_frequency[index],
                    gamma=(region == "G"),
                )[-1]
                for index in range(len(t))
            ],
            float,
        )
        for index in range(len(t)):
            replay = apply_projector(
                static_matrices[index],
                direct_cv[index] ** 2 - static_top[index] ** 2,
                static_frequency[index],
                gamma=(region == "G"),
            )
            max_projector_error = max(
                max_projector_error, abs(float(replay[-1]) - float(direct_cv[index]))
            )
            output_rows.append(
                {
                    "temperature_K": args.temperature,
                    "degauss_Ry": args.degauss,
                    "region": region,
                    "t_GK": float(t[index]),
                    "direct_DFPT_cm-1": float(direct[index]),
                    "static_short_MLIP_cm-1": float(static_top[index]),
                    "static_qspace_group_CV_cm-1": float(direct_cv[index]),
                    "static_qspace_final_fit_cm-1": float(direct_final[index]),
                    "thermal_short_TDEP_cm-1": float(thermal_top[index]),
                    "thermal_qspace_group_CV_cm-1": float(thermal_cv[index]),
                    "thermal_qspace_anchor_preserving_control_group_CV_cm-1": float(
                        thermal_anchor_cv[index]
                    ),
                    "thermal_qspace_final_fit_cm-1": float(thermal_final[index]),
                    "DFT_TDEP_cm-1": float(dft_top[index]),
                    "group_CV_delta_lambda_cm-2": float(cv_delta_lambda[index]),
                    "anchor_preserving_control_delta_lambda_cm-2": float(
                        cv_transfer_delta_lambda[index]
                    ),
                }
            )
        region_metric = {
            "n_points": len(t),
            "rounded_cusp_coefficients_delta_lambda_cm-2": final_coefficients.tolist(),
            "thermal_transfer_includes_static_intercept": True,
            "direct_DFPT_static_short_MAE_cm-1": float(
                np.mean(np.abs(static_top - direct))
            ),
            "direct_DFPT_group_CV_MAE_cm-1": float(
                np.mean(np.abs(direct_cv - direct))
            ),
            "direct_DFPT_group_CV_max_abs_cm-1": float(
                np.max(np.abs(direct_cv - direct))
            ),
            "DFT_TDEP_thermal_short_MAE_cm-1": float(
                np.mean(np.abs(thermal_top - dft_top))
            ),
            "DFT_TDEP_thermal_group_CV_MAE_cm-1": float(
                np.mean(np.abs(thermal_cv - dft_top))
            ),
            "DFT_TDEP_thermal_group_CV_max_abs_cm-1": float(
                np.max(np.abs(thermal_cv - dft_top))
            ),
            "control_anchor_preserving_transfer_MAE_cm-1": float(
                np.mean(np.abs(thermal_anchor_cv - dft_top))
            ),
            "control_anchor_preserving_transfer_max_abs_cm-1": float(
                np.max(np.abs(thermal_anchor_cv - dft_top))
            ),
        }
        anchor_t = 0.0 if region == "G" else 1.0
        anchor_index = int(np.argmin(np.abs(t - anchor_t)))
        region_metric["high_symmetry_DFT_TDEP_cm-1"] = float(dft_top[anchor_index])
        region_metric["high_symmetry_thermal_group_CV_cm-1"] = float(
            thermal_cv[anchor_index]
        )
        region_metric["high_symmetry_abs_error_cm-1"] = float(
            abs(thermal_cv[anchor_index] - dft_top[anchor_index])
        )
        region_metric["control_anchor_preserving_high_symmetry_abs_error_cm-1"] = float(
            abs(thermal_anchor_cv[anchor_index] - dft_top[anchor_index])
        )
        if region == "K":
            direct_lookup = {round(value, 3): freq for value, freq in zip(t, direct)}
            direct_cv_lookup = {
                round(value, 3): freq for value, freq in zip(t, direct_cv)
            }
            dft_lookup = {round(value, 3): freq for value, freq in zip(t, dft_top)}
            thermal_cv_lookup = {
                round(value, 3): freq for value, freq in zip(t, thermal_cv)
            }
            thermal_anchor_cv_lookup = {
                round(value, 3): freq for value, freq in zip(t, thermal_anchor_cv)
            }
            direct_kink = kink(direct_lookup)
            direct_cv_kink = kink(direct_cv_lookup)
            dft_kink = kink(dft_lookup)
            thermal_cv_kink = kink(thermal_cv_lookup)
            thermal_anchor_cv_kink = kink(thermal_anchor_cv_lookup)
            region_metric.update(
                {
                    "direct_DFPT_kink_cm-1_per_t": direct_kink,
                    "static_group_CV_kink_cm-1_per_t": direct_cv_kink,
                    "static_group_CV_kink_relative_error": abs(
                        direct_cv_kink - direct_kink
                    )
                    / max(direct_kink, 1e-12),
                    "DFT_TDEP_kink_cm-1_per_t": dft_kink,
                    "thermal_group_CV_kink_cm-1_per_t": thermal_cv_kink,
                    "thermal_group_CV_kink_relative_error": abs(
                        thermal_cv_kink - dft_kink
                    )
                    / max(dft_kink, 1e-12),
                    "control_anchor_preserving_transfer_kink_cm-1_per_t": thermal_anchor_cv_kink,
                    "control_anchor_preserving_transfer_kink_relative_error": abs(
                        thermal_anchor_cv_kink - dft_kink
                    )
                    / max(dft_kink, 1e-12),
                }
            )
        metrics[region] = region_metric

    qspace_pass = (
        all(
            metrics[region]["direct_DFPT_group_CV_MAE_cm-1"]
            < args.direct_cv_mae_threshold
            for region in ("G", "K")
        )
        and all(
            metrics[region]["DFT_TDEP_thermal_group_CV_MAE_cm-1"]
            < args.thermal_mae_threshold
            for region in ("G", "K")
        )
        and all(
            metrics[region]["high_symmetry_abs_error_cm-1"]
            < args.high_symmetry_threshold
            for region in ("G", "K")
        )
        and metrics["K"]["static_group_CV_kink_relative_error"]
        < args.kink_relative_threshold
        and metrics["K"]["thermal_group_CV_kink_relative_error"]
        < args.kink_relative_threshold
        and max_projector_error < 1e-5
    )
    result = {
        "status": "passed" if force_passed and seed_gate and qspace_pass else "failed",
        "scope": args.scope,
        "method": (
            "fit missing squared-frequency LR response against static direct DFPT "
            "using a static base+delta background; transfer the complete held-group "
            "correction, including the LR contribution at the high-symmetry anchor, "
            "to a short-range finite-temperature TDEP dynamical matrix; retain an "
            "anchor-preserving transfer as a non-selecting control"
        ),
        "temperature_K": args.temperature,
        "degauss_Ry": args.degauss,
        "thresholds": {
            "direct_DFPT_group_CV_MAE_cm-1": args.direct_cv_mae_threshold,
            "DFT_TDEP_thermal_group_CV_MAE_cm-1": args.thermal_mae_threshold,
            "Gamma_K_top_abs_error_cm-1": args.high_symmetry_threshold,
            "K_kink_relative_error": args.kink_relative_threshold,
            "seed_pair_full_band_MAE_cm-1": args.seed_mae_threshold,
            "MD_mean_temperature_relative_error": args.mean_temperature_relative_threshold,
            "matrix_projector_replay_max_abs_cm-1": 1e-5,
        },
        "inputs": {
            "dfpt_line": {"path": str(args.dfpt_line), "sha256": sha256(args.dfpt_line)},
            "background": {"path": str(args.background), "sha256": sha256(args.background)},
            "static_short_fc2": {"path": str(args.static_short_fc2), "sha256": sha256(args.static_short_fc2)},
            "thermal_short_tdep": {"path": str(args.thermal_short_tdep), "sha256": sha256(args.thermal_short_tdep)},
            "dft_tdep": {"path": str(args.dft_tdep), "sha256": sha256(args.dft_tdep)},
            "force_selection": {"path": str(args.force_selection), "sha256": sha256(args.force_selection)},
        },
        "force_gate_passed": force_passed,
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
        "passes_qspace_transfer_gate": qspace_pass,
        "passes_all_force_seed_qspace_gates": force_passed and seed_gate and qspace_pass,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "finite_temperature_qspace_predictions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary_path = args.output_dir / "acceptance.json"
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, summary_path)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for axis, region in zip(axes, ("G", "K")):
        selected_rows = [row for row in output_rows if row["region"] == region]
        selected_rows.sort(key=lambda row: row["t_GK"])
        t = [row["t_GK"] for row in selected_rows]
        axis.plot(t, [row["direct_DFPT_cm-1"] for row in selected_rows], "o-", color="#222222", label="static direct DFPT")
        axis.plot(t, [row["static_qspace_group_CV_cm-1"] for row in selected_rows], "--", color="#0072B2", label="static short + LR (group CV)")
        axis.plot(t, [row["DFT_TDEP_cm-1"] for row in selected_rows], "o-", color="#009E73", label=f"DFT-TDEP {args.temperature} K")
        axis.plot(t, [row["thermal_short_TDEP_cm-1"] for row in selected_rows], ":", color="#D55E00", label="short-range MLIP-TDEP")
        axis.plot(t, [row["thermal_qspace_anchor_preserving_control_group_CV_cm-1"] for row in selected_rows], "--", color="#999999", label="MLIP-TDEP + anchor-preserving correction (control)")
        axis.plot(t, [row["thermal_qspace_group_CV_cm-1"] for row in selected_rows], "-.", color="#CC79A7", label="MLIP-TDEP + full LR (group CV)")
        axis.set_title(f"{args.temperature} K · {region}")
        axis.set_xlabel("t along q=tK")
        axis.set_ylabel(r"top optical frequency (cm$^{-1}$)")
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.savefig(args.output_dir / "finite_temperature_qspace_comparison.png", dpi=200)
    fig.savefig(args.output_dir / "finite_temperature_qspace_comparison.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
