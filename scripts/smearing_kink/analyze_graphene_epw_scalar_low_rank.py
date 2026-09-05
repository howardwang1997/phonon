#!/usr/bin/env python3
"""Compress existing dense graphene EPW K shapes across smearing.

The dynamical-matrix correction is already known to be rank one in the A'
phonon subspace.  This script diagnoses the separate rank of its scalar
q/smearing dependence after removing the K anchor.  It uses no new DFT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEMPERATURES = (300, 450, 600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def symmetrized_curves(path: Path) -> dict:
    raw = [row for row in read_csv(path) if row["channel"] == "L0"]
    corrections = []
    backgrounds = []
    asymmetry = []
    smearings = []
    distances = np.arange(15, dtype=float) / 240.0
    for temperature in TEMPERATURES:
        block = [row for row in raw if int(row["temperature_K"]) == temperature]
        if len(block) != 29:
            raise ValueError(f"expected 29 dense rows at {temperature} K")
        by_index: dict[int, list[dict[str, str]]] = {}
        for row in block:
            index = int(round(abs(float(row["t_GK"]) - 1.0) * 240.0))
            by_index.setdefault(index, []).append(row)
        if set(by_index) != set(range(15)):
            raise ValueError(f"unexpected dense distance grid at {temperature} K")
        correction_row = []
        background_row = []
        maximum_asymmetry = 0.0
        for index in range(15):
            group = by_index[index]
            values = np.asarray(
                [float(row["dense_delta_lambda_top_cm-2"]) for row in group]
            )
            baseline = np.asarray(
                [float(row["finite_lattice_short_background_cm-1"]) for row in group]
            )
            correction_row.append(float(np.mean(values)))
            background_row.append(float(np.mean(baseline)))
            if len(values) == 2:
                maximum_asymmetry = max(
                    maximum_asymmetry, float(abs(values[0] - values[1]))
                )
        corrections.append(correction_row)
        backgrounds.append(background_row)
        asymmetry.append(maximum_asymmetry)
        smearings.append(float(block[0]["degauss_Ry"]))
    return {
        "distance": distances,
        "correction": np.asarray(corrections),
        "background": np.asarray(backgrounds),
        "maximum_left_right_asymmetry_cm-2": np.asarray(asymmetry),
        "degauss_Ry": np.asarray(smearings),
    }


def signed_frequency(background: np.ndarray, correction: np.ndarray) -> np.ndarray:
    squared = np.asarray(background, float) ** 2 + np.asarray(correction, float)
    return np.sign(squared) * np.sqrt(np.abs(squared))


def reconstruction_metrics(data: dict, reconstructed_centered: np.ndarray) -> dict:
    correction = data["correction"]
    reconstructed = correction[:, [0]] + reconstructed_centered
    target_frequency = signed_frequency(data["background"], correction)
    predicted_frequency = signed_frequency(data["background"], reconstructed)
    by_temperature = {}
    for index, temperature in enumerate(TEMPERATURES):
        target_depth = float(target_frequency[index, 2] - target_frequency[index, 0])
        predicted_depth = float(predicted_frequency[index, 2] - predicted_frequency[index, 0])
        by_temperature[str(temperature)] = {
            "frequency_RMSE_cm-1": float(
                np.sqrt(np.mean((predicted_frequency[index] - target_frequency[index]) ** 2))
            ),
            "frequency_max_abs_cm-1": float(
                np.max(np.abs(predicted_frequency[index] - target_frequency[index]))
            ),
            "target_d1_over_120_cusp_depth_cm-1": target_depth,
            "predicted_d1_over_120_cusp_depth_cm-1": predicted_depth,
            "cusp_depth_relative_error": float(
                abs(predicted_depth - target_depth) / abs(target_depth)
            ),
        }
    return {
        "correction_max_abs_cm-2": float(np.max(np.abs(reconstructed - correction))),
        "frequency_RMSE_cm-1": float(
            np.sqrt(np.mean((predicted_frequency - target_frequency) ** 2))
        ),
        "frequency_max_abs_cm-1": float(
            np.max(np.abs(predicted_frequency - target_frequency))
        ),
        "maximum_cusp_depth_relative_error": max(
            row["cusp_depth_relative_error"] for row in by_temperature.values()
        ),
        "by_temperature": by_temperature,
    }


def affine_coefficient_loso(
    data: dict, scalar_bases: np.ndarray, smearing_scores: np.ndarray
) -> dict:
    smear = np.asarray(data["degauss_Ry"], float)
    parameters = np.column_stack(
        (data["correction"][:, 0], np.asarray(smearing_scores, float))
    )
    folds = {}
    for held, temperature in enumerate(TEMPERATURES):
        training = [index for index in range(len(TEMPERATURES)) if index != held]
        design = np.column_stack((np.ones(len(training)), smear[training]))
        law = np.linalg.solve(design, parameters[training])
        predicted_parameters = np.asarray([1.0, smear[held]]) @ law
        predicted_centered = predicted_parameters[1:] @ scalar_bases
        predicted_correction = np.concatenate(
            (
                np.asarray([predicted_parameters[0]]),
                predicted_parameters[0] + predicted_centered,
            )
        )
        target_correction = data["correction"][held]
        predicted_frequency = signed_frequency(
            data["background"][held], predicted_correction
        )
        target_frequency = signed_frequency(
            data["background"][held], target_correction
        )
        target_depth = float(target_frequency[2] - target_frequency[0])
        predicted_depth = float(predicted_frequency[2] - predicted_frequency[0])
        folds[str(temperature)] = {
            "training_temperatures_K": [TEMPERATURES[index] for index in training],
            "predicted_parameters_cm-2": predicted_parameters.tolist(),
            "target_parameters_cm-2": parameters[held].tolist(),
            "correction_RMSE_cm-2": float(
                np.sqrt(np.mean((predicted_correction - target_correction) ** 2))
            ),
            "frequency_RMSE_cm-1": float(
                np.sqrt(np.mean((predicted_frequency - target_frequency) ** 2))
            ),
            "frequency_max_abs_cm-1": float(
                np.max(np.abs(predicted_frequency - target_frequency))
            ),
            "K_anchor_abs_error_cm-1": float(
                abs(predicted_frequency[0] - target_frequency[0])
            ),
            "cusp_depth_relative_error": float(
                abs(predicted_depth - target_depth) / abs(target_depth)
            ),
        }
    full_design = np.column_stack((np.ones(len(smear)), smear))
    full_law, *_ = np.linalg.lstsq(full_design, parameters, rcond=None)
    return {
        "scope": (
            "two scalar q bases are held fixed from the opened three-smearing SVD; "
            "only K anchor and two scores are fitted affinely from the other two smearings"
        ),
        "full_affine_law_intercept_and_slope": full_law.tolist(),
        "folds": folds,
        "maximum_frequency_RMSE_cm-1": max(
            row["frequency_RMSE_cm-1"] for row in folds.values()
        ),
        "maximum_frequency_abs_error_cm-1": max(
            row["frequency_max_abs_cm-1"] for row in folds.values()
        ),
        "maximum_K_anchor_abs_error_cm-1": max(
            row["K_anchor_abs_error_cm-1"] for row in folds.values()
        ),
        "maximum_cusp_depth_relative_error": max(
            row["cusp_depth_relative_error"] for row in folds.values()
        ),
        "limitation": (
            "the q bases use all three opened smearings, so this tests coefficient-law "
            "interpolation/extrapolation rather than a fully independent spectral holdout"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results/graphene_kohn_cusp_two_methods/B0_dense_k29/b0_dense_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E5_epw_scalar_low_rank"
        ),
    )
    args = parser.parse_args()

    data = symmetrized_curves(args.input)
    centered = data["correction"] - data["correction"][:, [0]]
    matrix = centered[:, 1:]
    left, singular, right_h = np.linalg.svd(matrix, full_matrices=False)
    capture = np.cumsum(singular**2) / np.sum(singular**2)
    reconstructions = {}
    metrics = {}
    for rank in (1, 2):
        reconstructed = (left[:, :rank] * singular[:rank]) @ right_h[:rank]
        full = np.column_stack((np.zeros(len(TEMPERATURES)), reconstructed))
        reconstructions[rank] = full
        metrics[f"rank{rank}"] = reconstruction_metrics(data, full)

    two_basis_scores = left[:, :2] * singular[:2]
    coefficient_loso = affine_coefficient_loso(
        data, right_h[:2], two_basis_scores
    )

    checks = {
        "rank1_centered_shape_capture_ge_0p995": float(capture[0]) >= 0.995,
        "rank2_centered_shape_capture_ge_0p9999": float(capture[1]) >= 0.9999,
        "rank2_frequency_RMSE_lt_0p1_cm-1": metrics["rank2"]["frequency_RMSE_cm-1"] < 0.1,
        "rank2_max_cusp_depth_relative_error_lt_0p03": (
            metrics["rank2"]["maximum_cusp_depth_relative_error"] < 0.03
        ),
        "third_singular_component_below_left_right_asymmetry": (
            float(singular[2])
            < float(np.max(data["maximum_left_right_asymmetry_cm-2"]))
        ),
        "affine_score_LOSO_frequency_RMSE_lt_0p3_cm-1": (
            coefficient_loso["maximum_frequency_RMSE_cm-1"] < 0.3
        ),
        "affine_score_LOSO_cusp_depth_relative_error_lt_0p10": (
            coefficient_loso["maximum_cusp_depth_relative_error"] < 0.10
        ),
    }
    status = "two_scalar_q_bases_sufficient" if all(checks.values()) else "more_scalar_q_bases_required"
    summary = {
        "status": status,
        "scope": "finite-smearing dense EPW A-prime scalar correction after exact K-anchor removal; no new DFT",
        "interpretation": (
            "use one Hermitian A-prime projector, a separate K anchor, and two shared q-shape coefficients per smearing"
            if all(checks.values())
            else "increase the scalar q basis before transfer fitting"
        ),
        "singular_values_cm-2": singular.tolist(),
        "cumulative_frobenius_capture": capture.tolist(),
        "reconstruction_metrics": metrics,
        "affine_smearing_coefficient_law": coefficient_loso,
        "maximum_left_right_asymmetry_cm-2": {
            str(temperature): float(value)
            for temperature, value in zip(
                TEMPERATURES,
                data["maximum_left_right_asymmetry_cm-2"],
                strict=True,
            )
        },
        "checks": checks,
        "inputs": {"path": str(args.input), "sha256": sha256(args.input)},
        "limitations": [
            "The bases are distilled from three opened graphene smearings and are not yet a cross-material basis.",
            "The exact zero-smearing row is excluded because its present k-grid convergence uncertainty is larger than the second finite-smearing singular component.",
            "A future transferable parameterization should predict the two coefficients from band velocity, filling, and electron-phonon coupling rather than tabulate them by smearing.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "epw_scalar_low_rank_summary.json", summary)
    model_path = args.output_dir / "epw_scalar_two_basis_development.npz"
    with model_path.open("wb") as handle:
        np.savez(
            handle,
            status=np.asarray(status),
            distances_from_K=data["distance"],
            degauss_Ry=data["degauss_Ry"],
            K_anchor_delta_lambda_cm2=data["correction"][:, 0],
            scalar_q_bases=right_h[:2],
            smearing_scores=two_basis_scores,
            affine_smearing_law_intercept_and_slope=np.asarray(
                coefficient_loso["full_affine_law_intercept_and_slope"]
            ),
            singular_values_cm2=singular,
            cumulative_frobenius_capture=capture,
        )

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), sharey=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for index, (axis, temperature, color) in enumerate(
        zip(axes, TEMPERATURES, colors, strict=True)
    ):
        axis.plot(
            data["distance"],
            centered[index],
            "o",
            color="#202020",
            ms=3.5,
            label="dense EPW",
        )
        axis.plot(
            data["distance"],
            reconstructions[1][index],
            "--",
            color="#D55E00",
            lw=1.4,
            label="one scalar q basis",
        )
        axis.plot(
            data["distance"],
            reconstructions[2][index],
            "-",
            color=color,
            lw=1.7,
            label="two scalar q bases",
        )
        axis.set_title(f"degauss = {data['degauss_Ry'][index]:.7f} Ry", fontsize=9.4)
        axis.set_xlabel("distance from K")
        axis.grid(axis="y", color="#E6E6E6", lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$\Delta\lambda(q)-\Delta\lambda(K)$ (cm$^{-2}$)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Dense EPW K response is low rank in q and smearing")
    fig.subplots_adjust(left=0.08, right=0.79, top=0.82, bottom=0.18, wspace=0.10)
    fig.savefig(
        args.output_dir / "epw_scalar_low_rank_comparison.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
