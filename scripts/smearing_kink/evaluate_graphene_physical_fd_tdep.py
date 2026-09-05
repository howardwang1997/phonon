#!/usr/bin/env python3
"""Evaluate the expanded physical-FD thermal fine-tune and decide wave 3."""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

CM = 33.35641


def load_spectrum(path: Path, temperature: int):
    with np.load(path, allow_pickle=False) as data:
        prefix = f"T{temperature}"
        mean_key = f"{prefix}_mean_temperature_K"
        max_key = f"{prefix}_max_temperature_K"
        return {
            "distance": np.asarray(data[f"{prefix}_dist"], float),
            "frequency_cm": np.asarray(data[f"{prefix}_freq"], float) * CM,
            "label_positions": np.asarray(data["label_positions"], float),
            "labels": [str(value) for value in data["labels"]],
            "trajectory_temperature": (
                {
                    "mean_K": float(data[mean_key]),
                    "max_K": float(data[max_key]),
                }
                if mean_key in data and max_key in data
                else None
            ),
        }


def interpolate(reference, candidate):
    return np.stack(
        [
            np.interp(
                reference["distance"],
                candidate["distance"],
                candidate["frequency_cm"][:, branch],
            )
            for branch in range(candidate["frequency_cm"].shape[1])
        ],
        axis=1,
    )


def spectrum_metrics(reference, candidate):
    candidate_frequency = interpolate(reference, candidate)
    delta = candidate_frequency - reference["frequency_cm"]
    normalized = [label.replace("$", "").replace("\\", "") for label in reference["labels"]]
    gamma_label = "Gamma" if "Gamma" in normalized else "Gamma"
    gamma_position = reference["label_positions"][normalized.index(gamma_label)]
    k_position = reference["label_positions"][normalized.index("K")]
    gamma_index = int(np.argmin(np.abs(reference["distance"] - gamma_position)))
    k_index = int(np.argmin(np.abs(reference["distance"] - k_position)))
    return {
        "full_band_MAE_cm-1": float(np.mean(np.abs(delta))),
        "full_band_max_abs_cm-1": float(np.max(np.abs(delta))),
        "gamma_top_abs_error_cm-1": float(abs(delta[gamma_index, -1])),
        "k_top_abs_error_cm-1": float(abs(delta[k_index, -1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--force-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave", default="wave2")
    parser.add_argument("--previous-wave", default="wave1")
    parser.add_argument("--model-tag", default="wave2")
    parser.add_argument("--final-wave", action="store_true")
    parser.add_argument("--force-rmse-threshold", type=float, default=50.0)
    parser.add_argument("--force-max-threshold", type=float, default=250.0)
    parser.add_argument("--dft-wave-mae-threshold", type=float, default=5.0)
    parser.add_argument("--mlip-mae-threshold", type=float, default=10.0)
    parser.add_argument("--high-symmetry-threshold", type=float, default=15.0)
    parser.add_argument("--seed-mae-threshold", type=float, default=5.0)
    parser.add_argument(
        "--mean-temperature-relative-threshold", type=float, default=0.20
    )
    args = parser.parse_args()

    force = json.loads(args.force_validation.read_text())
    result = {
        "thresholds": {
            "force_RMSE_meV_A": args.force_rmse_threshold,
            "force_max_abs_meV_A": args.force_max_threshold,
            "DFT_wave1_to_wave2_full_band_MAE_cm-1": args.dft_wave_mae_threshold,
            "MLIP_vs_DFT_full_band_MAE_cm-1": args.mlip_mae_threshold,
            "Gamma_K_top_abs_error_cm-1": args.high_symmetry_threshold,
            "seed_pair_full_band_MAE_cm-1": args.seed_mae_threshold,
            "MD_mean_temperature_relative_error": (
                args.mean_temperature_relative_threshold
            ),
        },
        "by_temperature": {},
    }
    passed = True
    for temperature, trained_label in ((300, "fd300"), (600, "fd600")):
        dft_wave1 = load_spectrum(
            args.root
            / f"graphene_physical_fd_dft_{temperature}K_{args.previous_wave}.npz",
            temperature,
        )
        dft_wave2 = load_spectrum(
            args.root / f"graphene_physical_fd_dft_{temperature}K_{args.wave}.npz",
            temperature,
        )
        dft_convergence = spectrum_metrics(dft_wave2, dft_wave1)
        seeds = [
            load_spectrum(
                args.root
                / f"td_graphene_v11_fd{temperature}_{args.model_tag}_short_range_seed{seed}.npz",
                temperature,
            )
            for seed in (0, 1, 2)
        ]
        mlip_metrics = [spectrum_metrics(dft_wave2, seed) for seed in seeds]
        seed_pairs = [spectrum_metrics(left, right) for left, right in combinations(seeds, 2)]
        temperature_statistics = [seed["trajectory_temperature"] for seed in seeds]
        if any(statistics is None for statistics in temperature_statistics):
            raise ValueError("MLIP TDEP result is missing trajectory temperature diagnostics")
        max_temperature_relative_error = max(
            abs(statistics["mean_K"] - temperature) / temperature
            for statistics in temperature_statistics
        )
        force_metrics = force["models"][trained_label]["by_temperature"][str(temperature)]
        temperature_pass = (
            float(force_metrics["RMSE_meV_A"]) <= args.force_rmse_threshold
            and float(force_metrics["max_abs_meV_A"]) <= args.force_max_threshold
            and dft_convergence["full_band_MAE_cm-1"] <= args.dft_wave_mae_threshold
            and max(item["full_band_MAE_cm-1"] for item in mlip_metrics)
            <= args.mlip_mae_threshold
            and max(
                max(item["gamma_top_abs_error_cm-1"], item["k_top_abs_error_cm-1"])
                for item in mlip_metrics
            ) <= args.high_symmetry_threshold
            and max(item["full_band_MAE_cm-1"] for item in seed_pairs)
            <= args.seed_mae_threshold
            and max_temperature_relative_error
            <= args.mean_temperature_relative_threshold
        )
        passed = passed and temperature_pass
        result["by_temperature"][str(temperature)] = {
            "force_validation": force_metrics,
            f"DFT_{args.previous_wave}_to_{args.wave}": dft_convergence,
            f"MLIP_seed_vs_DFT_{args.wave}": mlip_metrics,
            "MLIP_seed_pair_spread": seed_pairs,
            "MLIP_trajectory_temperatures": temperature_statistics,
            "MLIP_max_mean_temperature_relative_error": (
                max_temperature_relative_error
            ),
            "passes_short_range_gate": temperature_pass,
        }

    result["passes_short_range_gate"] = passed
    result["wave3_required"] = (not passed) and (not args.final_wave)
    result["next_step"] = (
        "fit and validate physical-FD q-space long-range correction"
        if passed
        else (
            "run conditional third physical-FD label wave, then refit"
            if not args.final_wave
            else "stop automatic data expansion and diagnose the short-range model"
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
