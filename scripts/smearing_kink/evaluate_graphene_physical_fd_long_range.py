#!/usr/bin/env python3
"""Validate physical-FD q-space long-range correction at 300 and 600 K.

The short-range background is the final physical-FD thermal fine-tuned TDEP
model.  At each matching Fermi-Dirac smearing, a two-parameter rounded cusp is
fit in squared-frequency space.  Equal-distance q groups are left out together
during validation, then the fitted shift is applied as a Hermitian projector to
the MLIP dynamical matrix.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from phonopy import Phonopy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import td_common as tdc  # noqa: E402
from phonon_accel.phonons import ase_to_phonopy  # noqa: E402

CM = 33.35641


def features(x: np.ndarray, degauss: float) -> np.ndarray:
    epsilon = 4.0 * degauss
    rounded = np.sqrt(x * x + epsilon * epsilon) - epsilon
    return np.column_stack((np.ones_like(x), rounded))


def fit_predict(
    x_train,
    target_train_cm,
    baseline_train_cm,
    x_target,
    baseline_target_cm,
    degauss,
):
    # Fit only the missing non-local contribution.  The thermal MLIP/TDEP
    # spectrum remains the short-range background at every held-out q point.
    coefficients, *_ = np.linalg.lstsq(
        features(np.asarray(x_train), degauss),
        np.asarray(target_train_cm) ** 2 - np.asarray(baseline_train_cm) ** 2,
        rcond=None,
    )
    squared = (
        np.asarray(baseline_target_cm) ** 2
        + features(np.asarray(x_target), degauss) @ coefficients
    )
    return np.sqrt(np.maximum(squared, 0.0)), coefficients


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
    corrected = np.sign(corrected_eigenvalues) * np.sqrt(
        np.abs(corrected_eigenvalues) * scale
    )
    return corrected


def load_rows(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 19:
        raise ValueError(f"expected 19 dense-line rows in {path}, found {len(rows)}")
    return rows


def final_wave(wave2: Path, wave3: Path | None) -> str:
    wave2_payload = json.loads(wave2.read_text())
    if wave2_payload["passes_short_range_gate"]:
        return "wave2"
    if wave3 is None or not wave3.is_file():
        raise RuntimeError("wave 2 failed and final wave-3 acceptance is unavailable")
    wave3_payload = json.loads(wave3.read_text())
    if not wave3_payload["passes_short_range_gate"]:
        raise RuntimeError(
            "final wave 3 failed the short-range gate; long-range fitting is invalid"
        )
    return "wave3"


def load_background(path: Path, temperature: int):
    with np.load(path, allow_pickle=False) as data:
        fc2 = np.asarray(data[f"T{temperature}_fc2"], float)
    primitive = tdc.build_monolayer("graphene", vacuum=7.5, a=2.4600000087)
    primitive.wrap()
    phonon = Phonopy(
        ase_to_phonopy(primitive),
        supercell_matrix=np.diag((6, 6, 1)),
        primitive_matrix=np.eye(3),
    )
    phonon.force_constants = fc2
    return phonon


def evaluate_temperature(rows, phonon, temperature, degauss):
    output_rows, region_metrics = [], {}
    max_matrix_error = 0.0
    for region in ("G", "K"):
        selected = sorted(
            (row for row in rows if row["region"] == region),
            key=lambda row: float(row["t_GK"]),
        )
        t = np.array([float(row["t_GK"]) for row in selected])
        target = np.array([float(row["f6_cm-1"]) for row in selected])
        x = t if region == "G" else np.abs(t - 1.0)
        qpoints = np.array([[value / 3.0, value / 3.0, 0.0] for value in t])
        phonon.run_qpoints(qpoints, with_dynamical_matrices=True)
        qdata = phonon.get_qpoints_dict()
        baseline = np.sort(np.asarray(qdata["frequencies"], float), axis=1) * CM
        matrices = np.asarray(qdata["dynamical_matrices"], complex)

        cv_prediction = np.zeros_like(target)
        for held_distance in sorted(set(np.round(x, 12))):
            held = np.isclose(x, held_distance)
            predicted, _ = fit_predict(
                x[~held],
                target[~held],
                baseline[~held, -1],
                x[held],
                baseline[held, -1],
                degauss,
            )
            cv_prediction[held] = predicted
        final_prediction, coefficients = fit_predict(
            x, target, baseline[:, -1], x, baseline[:, -1], degauss
        )

        for index, row in enumerate(selected):
            delta_lambda = final_prediction[index] ** 2 - baseline[index, -1] ** 2
            corrected = apply_projector(
                matrices[index], delta_lambda, baseline[index], gamma=(region == "G")
            )
            max_matrix_error = max(
                max_matrix_error, abs(float(corrected[-1]) - float(final_prediction[index]))
            )
            output_rows.append(
                {
                    "temperature_K": temperature,
                    "degauss_Ry": degauss,
                    "region": region,
                    "t_GK": t[index],
                    "direct_DFPT_cm-1": target[index],
                    "short_range_MLIP_TDEP_cm-1": baseline[index, -1],
                    "long_range_group_CV_cm-1": cv_prediction[index],
                    "long_range_final_fit_cm-1": final_prediction[index],
                }
            )
        errors = np.abs(cv_prediction - target)
        region_metrics[region] = {
            "n_points": len(selected),
            "rounded_cusp_coefficients_cm2": coefficients.tolist(),
            "short_range_MAE_cm-1": float(np.mean(np.abs(baseline[:, -1] - target))),
            "group_CV_MAE_cm-1": float(np.mean(errors)),
            "group_CV_max_abs_cm-1": float(np.max(errors)),
        }

        if region == "K":
            lookup_target = {round(value, 3): freq for value, freq in zip(t, target)}
            lookup_cv = {round(value, 3): freq for value, freq in zip(t, cv_prediction)}
            dx = 0.008
            dft_kink = abs(
                (lookup_target[1.008] - lookup_target[1.000]) / dx
                - (lookup_target[1.000] - lookup_target[0.992]) / dx
            )
            cv_kink = abs(
                (lookup_cv[1.008] - lookup_cv[1.000]) / dx
                - (lookup_cv[1.000] - lookup_cv[0.992]) / dx
            )
            region_metrics[region]["DFPT_kink_cm-1_per_t"] = float(dft_kink)
            region_metrics[region]["group_CV_kink_cm-1_per_t"] = float(cv_kink)
            region_metrics[region]["group_CV_kink_relative_error"] = float(
                abs(cv_kink - dft_kink) / max(dft_kink, 1e-12)
            )
    passed = (
        all(region_metrics[region]["group_CV_MAE_cm-1"] < 5.0 for region in ("G", "K"))
        and region_metrics["K"]["group_CV_kink_relative_error"] < 0.20
        and max_matrix_error < 1e-5
    )
    return output_rows, {
        "temperature_K": temperature,
        "degauss_Ry": degauss,
        "regions": region_metrics,
        "matrix_projector_replay_max_abs_cm-1": max_matrix_error,
        "passes_long_range_gate": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfpt-300", type=Path, required=True)
    parser.add_argument("--dfpt-600", type=Path, required=True)
    parser.add_argument("--td-root", type=Path, required=True)
    parser.add_argument("--wave2-acceptance", type=Path)
    parser.add_argument("--wave3-acceptance", type=Path)
    parser.add_argument(
        "--short-range-acceptance",
        type=Path,
        help="accepted replacement short-range result (for example model ablation)",
    )
    parser.add_argument(
        "--short-range-tag",
        help="TDEP filename tag paired with --short-range-acceptance",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.short_range_acceptance is not None:
        if not args.short_range_tag:
            parser.error("--short-range-tag is required with --short-range-acceptance")
        acceptance = json.loads(args.short_range_acceptance.read_text())
        if not acceptance["passes_short_range_gate"]:
            raise RuntimeError("replacement short-range result failed its gate")
        wave = args.short_range_tag
    else:
        if args.wave2_acceptance is None:
            parser.error(
                "provide --wave2-acceptance or --short-range-acceptance with its tag"
            )
        wave = final_wave(args.wave2_acceptance, args.wave3_acceptance)
    all_rows, metrics = [], []
    for temperature, degauss, dfpt_path in (
        (300, 0.0019000869, args.dfpt_300),
        (600, 0.0038001738, args.dfpt_600),
    ):
        background_path = (
            args.td_root
            / f"td_graphene_v11_fd{temperature}_{wave}_short_range_seed0.npz"
        )
        rows, result = evaluate_temperature(
            load_rows(dfpt_path),
            load_background(background_path, temperature),
            temperature,
            degauss,
        )
        all_rows.extend(rows)
        metrics.append(result)
    summary = {
        "final_short_range_model_tag": wave,
        "final_short_range_wave": wave,
        "smearing": "fermi-dirac",
        "acceptance": {
            "Gamma_and_K_group_CV_MAE_cm-1": 5.0,
            "K_kink_relative_error": 0.20,
            "matrix_projector_replay_max_abs_cm-1": 1e-5,
        },
        "by_temperature": metrics,
        "passes_all_long_range_gates": all(
            item["passes_long_range_gate"] for item in metrics
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "physical_fd_long_range_predictions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = args.output_dir / "physical_fd_long_range_summary.json"
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, summary_path)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for column, temperature in enumerate((300, 600)):
        for row_index, region in enumerate(("G", "K")):
            axis = axes[row_index, column]
            data = [row for row in all_rows if row["temperature_K"] == temperature and row["region"] == region]
            data.sort(key=lambda row: row["t_GK"])
            axis.plot([row["t_GK"] for row in data], [row["direct_DFPT_cm-1"] for row in data], "o-", color="#222222", label="direct DFPT")
            axis.plot([row["t_GK"] for row in data], [row["short_range_MLIP_TDEP_cm-1"] for row in data], "--", color="#D55E00", label="thermal fine-tune")
            axis.plot([row["t_GK"] for row in data], [row["long_range_group_CV_cm-1"] for row in data], "-.", color="#0072B2", label="fine-tune + q-space LR")
            axis.set_title(f"{temperature} K · {region}")
            axis.set_xlabel("t along q=tK")
            axis.set_ylabel(r"top optical frequency (cm$^{-1}$)")
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "physical_fd_long_range_comparison.png", dpi=200)
    fig.savefig(args.output_dir / "physical_fd_long_range_comparison.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
