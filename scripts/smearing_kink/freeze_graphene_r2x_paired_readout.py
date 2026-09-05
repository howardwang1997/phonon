#!/usr/bin/env python3
"""Freeze the conditional R2X paired-bilinear readout after its fixed OOF gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import (
    DEFAULT_AGGREGATE,
    _parse_whitelist,
    canonical_json_bytes,
    file_sha256,
)
from evaluate_graphene_r2t_bilinear_nested_objective import (
    bilinear_indices,
    design_audit,
    fold_statistics,
    predict,
    ridge_system,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_DESIGN_ROOT = BASE / "R2T_full_bilinear_materialization_20260826"
DEFAULT_FEATURE_SCALER = (
    BASE / "R2T_full_bilinear_shard_0_20260826/feature_scaler.npz"
)
DEFAULT_OUTPUT = BASE / "R2X_paired_readout_freeze_20260826"
PROJECTION_MASS = 0.45
RIDGE_ALPHA = 1.0e-3
PAIRED_BILINEAR_INDICES = np.asarray([127, 383], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--design-root", type=Path, default=DEFAULT_DESIGN_ROOT)
    parser.add_argument("--feature-scaler", type=Path, default=DEFAULT_FEATURE_SCALER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()

    reference, aprime, label_hashes = _parse_whitelist(
        r2r1.RECOMMENDED_THERMAL92
    )
    design_receipt_path = args.design_root / "receipt.json"
    design_receipt = json.loads(design_receipt_path.read_text())
    energy_path = args.design_root / "bilinear_energy_design_eV.npy"
    force_path = args.design_root / "bilinear_force_design_eV_A.npy"
    scaler_path = args.feature_scaler.resolve()
    bilinear_energy = np.load(energy_path, allow_pickle=False)
    bilinear_force = np.load(force_path, allow_pickle=False)
    if r2r1.raw_array_sha256(bilinear_energy, "<f8") != design_receipt[
        "array_raw_sha256"
    ]["energy"]:
        raise ValueError("R2X bilinear energy design differs from its receipt")
    if r2r1.raw_array_sha256(bilinear_force, "<f8") != design_receipt[
        "array_raw_sha256"
    ]["force"]:
        raise ValueError("R2X bilinear force design differs from its receipt")
    if file_sha256(args.aggregate.resolve()) != r2r1.ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("attempt3 aggregate SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        base_energy = np.asarray(
            arrays["thermal_parameter_energy_design_eV"], dtype=np.float64
        )
        base_force = np.asarray(
            arrays["thermal_parameter_force_design_eV_A"], dtype=np.float64
        )
        fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], dtype=np.float64)
    with np.load(scaler_path, allow_pickle=False) as scaler:
        feature_mean = np.asarray(scaler["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(scaler["feature_scale"], dtype=np.float64)
    if feature_mean.shape != (34,) or feature_scale.shape != (34,):
        raise ValueError("R2X feature scaler shape changed")

    selected_bilinear = np.concatenate(
        (bilinear_indices("cross32"), PAIRED_BILINEAR_INDICES)
    )
    energy_design = np.column_stack(
        (base_energy, np.take(bilinear_energy, selected_bilinear, axis=-1))
    )
    force_design = np.concatenate(
        (base_force, np.take(bilinear_force, selected_bilinear, axis=-1)), axis=3
    )
    audit = design_audit(force_design)
    if not audit["pass"] or audit["width"] != 99:
        raise ValueError(f"R2X frozen design audit failed: {audit}")
    statistics = fold_statistics(force_design, fixed, reference, aprime)

    oof = np.full_like(reference, np.nan)
    outer_records = []
    for hold_id, hold_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
        hold = np.asarray(hold_tuple, dtype=int)
        train_folds = tuple(fold for fold in range(4) if fold != hold_id)
        coefficient = ridge_system(
            statistics, train_folds, PROJECTION_MASS
        ).coefficient(RIDGE_ALPHA)
        oof[hold] = predict(force_design[hold], fixed[hold], coefficient)
        outer_records.append(
            {
                "outer_fold": hold_id,
                "outer_hold_global_indices": hold.tolist(),
                "physical_coefficient_raw_sha256": r2r1.raw_array_sha256(
                    coefficient, "<f8"
                ),
            }
        )
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("R2X fixed-hyperparameter OOF does not cover thermal92")
    all_indices = np.arange(92, dtype=int)
    oof_metrics = r2r1.gate_metrics(oof, reference, aprime, all_indices)
    if not oof_metrics["passes_fixed_gate"]:
        raise ValueError(f"R2X fixed OOF gate failed: {oof_metrics}")

    final_system = ridge_system(
        statistics, tuple(range(4)), PROJECTION_MASS
    )
    coefficient = final_system.coefficient(RIDGE_ALPHA)
    train_prediction = predict(force_design, fixed, coefficient)
    train_metrics = r2r1.gate_metrics(
        train_prediction, reference, aprime, all_indices
    )
    if not train_metrics["passes_fixed_gate"]:
        raise ValueError(f"R2X final all92 train gate failed: {train_metrics}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "frozen_readout.npz"
    np.savez(
        arrays_path,
        physical_coefficient=np.asarray(coefficient, dtype="<f8"),
        base65_physical_coefficient=np.asarray(coefficient[:65], dtype="<f8"),
        bilinear34_physical_coefficient=np.asarray(coefficient[65:], dtype="<f8"),
        selected_bilinear_indices=np.asarray(selected_bilinear, dtype="<i8"),
        feature_mean=np.asarray(feature_mean, dtype="<f8"),
        feature_scale=np.asarray(feature_scale, dtype="<f8"),
        OOF_predicted_force_eV_A=np.asarray(oof, dtype="<f8"),
        train_predicted_force_eV_A=np.asarray(train_prediction, dtype="<f8"),
        thermal_energy_design_eV=np.asarray(energy_design, dtype="<f8"),
        thermal_force_design_eV_A=np.asarray(force_design, dtype="<f8"),
    )
    summary = {
        "format": "graphene_r2x_paired_bilinear_frozen_readout_v1",
        "status": "R2X_TRAIN_GATE_PASSED_MECHANICS_PENDING",
        "deployable": False,
        "conditional_representation_development": True,
        "representation": (
            "R2R 65 + 32 interaction1_i*interaction2_i diagonal columns + "
            "plain/amplitude-modulated interaction1_07*interaction2_15 pair"
        ),
        "base_column_count": 65,
        "bilinear_column_count": 34,
        "total_column_count": 99,
        "selected_bilinear_indices": selected_bilinear.tolist(),
        "projection_mass": PROJECTION_MASS,
        "ridge_alpha": RIDGE_ALPHA,
        "hyperparameter_policy": (
            "fixed after thermal92 representation diagnostics; no independent "
            "development or unseen labels accessed"
        ),
        "design_audit": audit,
        "fixed_hyperparameter_OOF": {
            "outer_records": outer_records,
            "pooled_metrics": oof_metrics,
            "prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
        },
        "final_all92_fit": {
            "metrics": train_metrics,
            "coefficient_raw_sha256": r2r1.raw_array_sha256(coefficient, "<f8"),
            "base65_coefficient_raw_sha256": r2r1.raw_array_sha256(
                coefficient[:65], "<f8"
            ),
            "bilinear34_coefficient_raw_sha256": r2r1.raw_array_sha256(
                coefficient[65:], "<f8"
            ),
        },
        "input_sha256": {
            "thermal92": file_sha256(r2r1.RECOMMENDED_THERMAL92),
            "aggregate_arrays": file_sha256(args.aggregate),
            "bilinear_design_receipt": file_sha256(design_receipt_path),
            "bilinear_energy_design": file_sha256(energy_path),
            "bilinear_force_design": file_sha256(force_path),
            "feature_scaler": file_sha256(scaler_path),
        },
        "label_raw_sha256": label_hashes,
        "output_arrays_sha256": file_sha256(arrays_path),
        "energy_labels_used": False,
        "development_or_unseen_access": False,
        "mechanics_pending": True,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": (
            "the fixed-hyperparameter OOF passes, but the paired representation and "
            "hyperparameters were developed on thermal92; mechanics and separately "
            "frozen development/unseen checks remain mandatory"
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
