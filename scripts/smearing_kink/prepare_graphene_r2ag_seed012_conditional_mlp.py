#!/usr/bin/env python3
"""Freeze the seed012 conditional-MLP training package and linear skips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_graphene_r2z_forward_bilinear as r2z
import fit_graphene_r2ad_seed012_step32 as r2ad
import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_BOUNDARY_ROOT = BASE / "R2AE_seed012_step32_boundary_20260826"
DEFAULT_OUTPUT = BASE / "R2AG_seed012_conditional_mlp_package_20260826"


def solve_coefficient(
    statistics: tuple[r2ad.FoldRaw, ...],
    folds: tuple[int, ...],
    best: dict,
) -> np.ndarray:
    seed_weights = tuple(
        float(best["seed_projection_weights"][name]) for name in r2ad.SEED_RANGES
    )
    normalizer, gram, rhs = r2ad.normal_equations(
        statistics,
        folds,
        float(best["projection_mass"]),
        seed_weights,
        float(best["bilinear_penalty"]),
        float(best["T600_force_fraction"]),
    )
    alpha = float(best["alpha"])
    normalized = np.linalg.solve(gram + alpha * np.eye(len(rhs)), rhs)
    return r2ad.FORCE_SCALE * normalized / normalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=r2z.DEFAULT_AGGREGATE)
    parser.add_argument("--failed-root", type=Path, default=r2z.DEFAULT_R2Y_FAILED)
    parser.add_argument(
        "--seed1-selected-root", type=Path, default=r2z.DEFAULT_SEED1_SELECTED
    )
    parser.add_argument("--thermal-full-root", type=Path, default=r2z.DEFAULT_THERMAL_FULL)
    parser.add_argument("--seed1-full-root", type=Path, default=r2z.DEFAULT_SEED1_FULL)
    parser.add_argument("--seed2-labels", type=Path, default=r2ad.DEFAULT_SEED2_LABELS)
    parser.add_argument("--seed2-base-root", type=Path, default=r2ad.DEFAULT_SEED2_BASE)
    parser.add_argument("--seed2-full-root", type=Path, default=r2ad.DEFAULT_SEED2_FULL)
    parser.add_argument("--path-root", type=Path, default=r2ad.DEFAULT_PATH_ROOT)
    parser.add_argument("--boundary-root", type=Path, default=DEFAULT_BOUNDARY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data, input_receipt = r2ad.load_training(args)
    statistics = r2ad.fold_raw_statistics(data)
    boundary_summary_path = args.boundary_root / "summary.json"
    boundary_prediction_path = (
        args.boundary_root / "best_OOF_predicted_force_eV_A.npy"
    )
    boundary = json.loads(boundary_summary_path.read_text())
    if boundary.get("status") != "R2AE_FIXED_STEP32_DEVELOPMENT_GATE_FAILED":
        raise ValueError("R2AG expects the terminal fixed-step32 boundary audit")
    if file_sha256(boundary_prediction_path) != boundary.get("prediction_sha256"):
        raise ValueError("R2AG boundary prediction differs from receipt")
    best = boundary["best"]
    saved_oof = np.load(boundary_prediction_path, allow_pickle=False)
    if saved_oof.shape != (132, 72, 3):
        raise ValueError("R2AG boundary OOF shape changed")

    coefficients = np.empty((4, 129), dtype=np.float64)
    replay_oof = np.full((132, 72, 3), np.nan)
    for hold_id, fold_tuple in enumerate(r2ad.FOLDS):
        coefficients[hold_id] = solve_coefficient(
            statistics,
            tuple(value for value in range(4) if value != hold_id),
            best,
        )
        hold = np.asarray(fold_tuple, dtype=int)
        replay_oof[hold] = data["fixed"][hold] + np.einsum(
            "natk,k->nat",
            data["design"][hold],
            coefficients[hold_id],
            optimize=False,
        )
    replay_max = float(np.max(np.abs(replay_oof - saved_oof)))
    if replay_max > 2.0e-10:
        raise ValueError(f"R2AG boundary OOF replay failed: {replay_max:.3e}")
    all_coefficient = solve_coefficient(statistics, tuple(range(4)), best)

    with np.load(args.failed_root / "frozen_readout.npz", allow_pickle=False) as arrays:
        bilinear_mean = np.asarray(arrays["feature_mean"], dtype=np.float64)
        bilinear_scale = np.asarray(arrays["feature_scale"], dtype=np.float64)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package_path = output / "training_package.npz"
    np.savez(
        package_path,
        fixed_force_eV_A=np.asarray(data["fixed"], dtype="<f8"),
        reference_force_eV_A=np.asarray(data["reference"], dtype="<f8"),
        Aprime_mode_real=np.asarray(data["modes"].real, dtype="<f8"),
        Aprime_mode_imag=np.asarray(data["modes"].imag, dtype="<f8"),
        Aprime_coordinates_real_A=np.asarray(
            data["coordinates"].real, dtype="<f8"
        ),
        Aprime_coordinates_imag_A=np.asarray(
            data["coordinates"].imag, dtype="<f8"
        ),
        total_base_force_eV_A=np.asarray(data["base"], dtype="<f8"),
        selected_bilinear_indices=np.asarray(
            data["selected_bilinear_indices"], dtype="<i8"
        ),
        bilinear_feature_mean=np.asarray(bilinear_mean, dtype="<f8"),
        bilinear_feature_scale=np.asarray(bilinear_scale, dtype="<f8"),
        fold_hold_indices=np.asarray(r2ad.FOLDS, dtype="<i8"),
        fold_linear_skip_coefficient=np.asarray(coefficients, dtype="<f8"),
        all132_linear_skip_coefficient=np.asarray(all_coefficient, dtype="<f8"),
        fixed_step32_OOF_predicted_force_eV_A=np.asarray(
            replay_oof, dtype="<f8"
        ),
    )
    receipt = {
        "format": "graphene_r2ag_seed012_conditional_mlp_training_package_v1",
        "status": "R2AG_SEED012_CONDITIONAL_MLP_PACKAGE_FROZEN",
        "deployable": False,
        "data_roles": {
            "seed0": "training",
            "seed1": "promoted_training",
            "seed2": "promoted_training_from_opened_development",
            "new_525K_off_policy": "unopened_true_unseen",
        },
        "structure_order": "seed0[20], seed1[20], seed2[20], T300[36], T600[36]",
        "folds": [list(item) for item in r2ad.FOLDS],
        "fixed_linear_skip_hyperparameters": {
            key: best[key]
            for key in (
                "projection_mass",
                "seed_projection_weights",
                "bilinear_penalty",
                "T600_force_fraction",
                "group_masses",
                "alpha",
            )
        },
        "linear_skip_width": 129,
        "selected_bilinear_indices": data[
            "selected_bilinear_indices"
        ].tolist(),
        "fold_coefficient_raw_sha256": r2r1.raw_array_sha256(
            coefficients, "<f8"
        ),
        "all132_coefficient_raw_sha256": r2r1.raw_array_sha256(
            all_coefficient, "<f8"
        ),
        "fixed_step32_OOF_replay_max_abs_eV_A": replay_max,
        "fixed_step32_OOF_raw_sha256": r2r1.raw_array_sha256(
            replay_oof, "<f8"
        ),
        "input_receipt": input_receipt,
        "input_sha256": {
            "boundary_summary": file_sha256(boundary_summary_path),
            "boundary_prediction": file_sha256(boundary_prediction_path),
            "failed_R2Y_checkpoint_scaler_source": file_sha256(
                args.failed_root / "frozen_readout.npz"
            ),
        },
        "package_sha256": file_sha256(package_path),
        "energy_labels_used": False,
        "unseen_525K_access": False,
    }
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    (output / "DONE").write_text(receipt["status"] + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
