#!/usr/bin/env python3
"""One-shot seed1 and actual-small-H confirmation of frozen R2X."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from graphene_r2r0_formal import load_endpoint
from graphene_r2x_paired_readout import (
    load_paired_readout_checkpoint,
    production_paired_readout_energy_force,
)
from train_graphene_r2s_conditional_mlp import recommended_inputs


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/graphene_r2o_taylor_null_core"
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_FREEZE_ROOT = BASE / "R2X_paired_readout_freeze_20260826"
DEFAULT_MECHANICS_ROOT = BASE / "R2X_paired_readout_mechanics_20260826"
DEFAULT_OUTPUT = BASE / "R2X_paired_readout_development_20260826"
EXPECTED_CHECKPOINT_SHA256 = (
    "0aae2ef1ee871406c347843f1c5c6155faac8df816bf9fb3addcf21bc0edcfd0"
)
EXPECTED_MECHANICS_SUMMARY_SHA256 = (
    "3ea364fe606e3f7f0cf6c0dc6323550eeefaa7db235f827197d86fd1df915abc"
)
SEED1_PATH = DATA / "valid_e50_seed1.xyz"
SMALL_PATH = DATA / "harmonic_lambda1_small_gate.xyz"


def force_metrics(error: np.ndarray) -> dict[str, float | int]:
    flattened = np.asarray(error, dtype=np.float64).reshape(-1)
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(np.square(flattened)))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(flattened))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(flattened))),
        "n_force_components": int(flattened.size),
    }


def restoring_slope(coordinates: np.ndarray, projected_force: np.ndarray) -> float:
    denominator = float(np.vdot(coordinates, coordinates).real)
    if denominator <= 0.0:
        raise ValueError("development A-prime coordinates vanished")
    return -float(np.vdot(coordinates, projected_force).real / denominator)


def predict(model, structures, reference, checkpoint, device):
    output = np.empty((len(structures), len(structures[0]), 3), dtype=np.float64)
    energy = np.empty(len(structures), dtype=np.float64)
    for index, structure in enumerate(structures):
        with torch.enable_grad():
            probe = production_paired_readout_energy_force(
                model, structure, reference, checkpoint, device=device
            )
        energy[index] = float(probe.energy_eV.detach().cpu())
        output[index] = probe.force_source_order_eV_A.detach().cpu().numpy()
        if index == 0 or (index + 1) % 5 == 0 or index + 1 == len(structures):
            print(
                json.dumps(
                    {
                        "predicted_count": index + 1,
                        "requested_count": len(structures),
                        "atom_count": len(structure),
                    }
                ),
                flush=True,
            )
    return energy, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freeze-root", type=Path, default=DEFAULT_FREEZE_ROOT)
    parser.add_argument("--mechanics-root", type=Path, default=DEFAULT_MECHANICS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    mechanics_summary_path = args.mechanics_root / "summary.json"
    if file_sha256(mechanics_summary_path) != EXPECTED_MECHANICS_SUMMARY_SHA256:
        raise ValueError("R2X mechanics summary differs from frozen passing receipt")
    mechanics = json.loads(mechanics_summary_path.read_text())
    if mechanics.get("status") != "R2X_PAIRED_READOUT_MECHANICS_PASSED" or not all(
        mechanics.get("checks", {}).values()
    ):
        raise ValueError("R2X development requires all mechanics checks to pass")
    checkpoint_path = args.freeze_root / "frozen_readout.npz"
    checkpoint = load_paired_readout_checkpoint(
        checkpoint_path, expected_sha256=EXPECTED_CHECKPOINT_SHA256
    )
    inputs = recommended_inputs()
    model = load_endpoint(inputs, args.device)
    reference6 = read(inputs.reference_6x6, index=0)
    reference8 = read(inputs.reference_8x8, index=0)
    seed1 = read(SEED1_PATH, index=":")
    small = read(SMALL_PATH, index=":")
    if len(seed1) != 20 or any(len(item) != 72 for item in seed1):
        raise ValueError("R2X seed1 development set schema changed")
    if len(small) != 12 or any(len(item) != 128 for item in small):
        raise ValueError("R2X actual-small-H development set schema changed")

    seed1_energy, seed1_prediction = predict(
        model, seed1, reference6, checkpoint, args.device
    )
    small_energy, small_prediction = predict(
        model, small, reference8, checkpoint, args.device
    )
    seed1_reference = np.stack([item.arrays["REF_forces"] for item in seed1])
    small_reference = np.stack([item.arrays["REF_forces"] for item in small])
    seed1_force = force_metrics(seed1_prediction - seed1_reference)
    small_force = force_metrics(small_prediction - small_reference)

    mode = np.stack(
        [
            item.arrays["APRIME_mode_real"]
            + 1.0j * item.arrays["APRIME_mode_imag"]
            for item in seed1
        ]
    )
    base = np.stack(
        [
            item.arrays["FOUNDATION_BASE_forces"]
            + item.arrays["FROZEN_Q6_forces"]
            for item in seed1
        ]
    )
    error_modes = np.einsum(
        "nat,nat->n", np.conj(mode), seed1_prediction - seed1_reference
    )
    predicted_modes = np.einsum(
        "nat,nat->n", np.conj(mode), base + seed1_prediction
    )
    target_modes = np.einsum(
        "nat,nat->n", np.conj(mode), base + seed1_reference
    )
    coordinates = np.asarray(
        [
            complex(
                float(item.info["APRIME_coordinate_real_A"]),
                float(item.info["APRIME_coordinate_imag_A"]),
            )
            for item in seed1
        ]
    )
    predicted_slope = restoring_slope(coordinates, predicted_modes)
    target_slope = restoring_slope(coordinates, target_modes)
    slope_relative_error = (predicted_slope - target_slope) / target_slope
    seed1_aprime = {
        "RMS_meV_A": float(1000.0 * np.sqrt(np.mean(np.abs(error_modes) ** 2))),
        "predicted_restoring_slope_eV_A2": predicted_slope,
        "target_restoring_slope_eV_A2": target_slope,
        "slope_relative_error": float(slope_relative_error),
        "count": 20,
    }
    checks = {
        "seed1_force_RMSE": seed1_force["RMSE_meV_A"] <= 30.0,
        "seed1_force_max": seed1_force["max_abs_meV_A"] <= 200.0,
        "seed1_Aprime_RMS": seed1_aprime["RMS_meV_A"] <= 15.0,
        "seed1_Aprime_slope": abs(slope_relative_error) <= 0.05,
        "actual_small_H_force_RMSE": small_force["RMSE_meV_A"] <= 0.5,
        "actual_small_H_force_max": small_force["max_abs_meV_A"] <= 10.0,
    }
    if not all(math.isfinite(float(value)) for value in seed1_energy) or not all(
        math.isfinite(float(value)) for value in small_energy
    ):
        raise ValueError("R2X development predicted a non-finite energy")
    passed = all(checks.values())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "development_predictions.npz"
    np.savez(
        arrays_path,
        seed1_energy_eV=np.asarray(seed1_energy, dtype="<f8"),
        seed1_predicted_force_eV_A=np.asarray(seed1_prediction, dtype="<f8"),
        seed1_reference_force_eV_A=np.asarray(seed1_reference, dtype="<f8"),
        small_energy_eV=np.asarray(small_energy, dtype="<f8"),
        small_predicted_force_eV_A=np.asarray(small_prediction, dtype="<f8"),
        small_reference_force_eV_A=np.asarray(small_reference, dtype="<f8"),
    )
    summary = {
        "format": "graphene_r2x_paired_readout_development_confirmation_v1",
        "status": "R2X_DEVELOPMENT_PASSED" if passed else "R2X_DEVELOPMENT_FAILED",
        "deployable": False,
        "checkpoint_or_hyperparameters_modified": False,
        "checks": checks,
        "seed1": {"force": seed1_force, "Aprime": seed1_aprime},
        "actual_small_H": {"force": small_force},
        "input_sha256": {
            "checkpoint": file_sha256(checkpoint_path),
            "freeze_summary": file_sha256(args.freeze_root / "summary.json"),
            "mechanics_summary": file_sha256(mechanics_summary_path),
            "seed1": file_sha256(SEED1_PATH),
            "actual_small_H": file_sha256(SMALL_PATH),
            "reference6": file_sha256(inputs.reference_6x6),
            "reference8": file_sha256(inputs.reference_8x8),
        },
        "arrays_sha256": file_sha256(arrays_path),
        "seed1_and_actual_small_are_development_not_blind": True,
        "unseen_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = output / "summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
