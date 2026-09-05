#!/usr/bin/env python3
"""Freeze the all-132 R2AE step-32 readout for spectral sensitivity runs.

The development OOF force gate did not pass, so this artifact is explicitly
not a force-gate-approved deployment model.  It is frozen because its three
A-prime restoring slopes are stable and because the final research acceptance
is the finite-lattice-temperature Kohn cusp.  SSCHA results made with it must
be reported together with model-sensitivity and the failed OOF metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import fit_graphene_r2ad_seed012_step32 as r2ad
import graphene_r2r1_linear_readout as r2r1
import train_graphene_r2ag_seed012_conditional_mlp as r2ag
from diagnose_graphene_r2r1_aprime_objective import (
    canonical_json_bytes,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_OUTPUT = BASE / "R2AO_step32_all132_spectral_candidate_20260827"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=r2ad.r2z.DEFAULT_AGGREGATE)
    parser.add_argument("--failed-root", type=Path, default=r2ad.r2z.DEFAULT_R2Y_FAILED)
    parser.add_argument(
        "--seed1-selected-root", type=Path, default=r2ad.r2z.DEFAULT_SEED1_SELECTED
    )
    parser.add_argument(
        "--thermal-full-root", type=Path, default=r2ad.r2z.DEFAULT_THERMAL_FULL
    )
    parser.add_argument(
        "--seed1-full-root", type=Path, default=r2ad.r2z.DEFAULT_SEED1_FULL
    )
    parser.add_argument("--seed2-labels", type=Path, default=r2ad.DEFAULT_SEED2_LABELS)
    parser.add_argument("--seed2-base-root", type=Path, default=r2ad.DEFAULT_SEED2_BASE)
    parser.add_argument("--seed2-full-root", type=Path, default=r2ad.DEFAULT_SEED2_FULL)
    parser.add_argument("--path-root", type=Path, default=r2ad.DEFAULT_PATH_ROOT)
    parser.add_argument("--package-root", type=Path, default=r2ag.DEFAULT_PACKAGE_ROOT)
    parser.add_argument(
        "--boundary-root",
        type=Path,
        default=BASE / "R2AE_seed012_step32_boundary_20260826",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()

    package, package_receipt = r2ag.load_package(args.package_root.resolve())
    data, data_receipt = r2ad.load_training(args)
    coefficient = np.asarray(
        package["all132_linear_skip_coefficient"], dtype=np.float64
    )
    prediction = data["fixed"] + np.einsum(
        "natk,k->nat", data["design"], coefficient, optimize=True
    )
    thermal92_prediction = np.concatenate((prediction[:20], prediction[60:]), axis=0)
    if thermal92_prediction.shape != (92, 72, 3):
        raise ValueError("R2AO thermal92 runtime replay order changed")
    train_metrics = r2ag.scalar_metrics(prediction, package)
    oof_prediction = package["fixed_step32_OOF_predicted_force_eV_A"]
    oof_metrics = r2ag.scalar_metrics(oof_prediction, package)
    boundary_path = args.boundary_root / "summary.json"
    boundary = json.loads(boundary_path.read_text())
    if boundary.get("status") != "R2AE_FIXED_STEP32_DEVELOPMENT_GATE_FAILED":
        raise ValueError("R2AO requires the terminal R2AE boundary receipt")
    if abs(
        float(boundary["best"]["full_gate"]["raw_gate_score"])
        - float(oof_metrics["raw_gate_score"])
    ) > 1.0e-10:
        raise ValueError("R2AO package/boundary OOF metrics disagree")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "spectral_candidate_readout.npz"
    np.savez(
        checkpoint_path,
        physical_coefficient=np.asarray(coefficient, dtype="<f8"),
        base65_physical_coefficient=np.asarray(coefficient[:65], dtype="<f8"),
        bilinear64_physical_coefficient=np.asarray(coefficient[65:], dtype="<f8"),
        selected_bilinear_indices=np.asarray(
            package["selected_bilinear_indices"], dtype="<i8"
        ),
        feature_mean=np.asarray(package["bilinear_feature_mean"], dtype="<f8"),
        feature_scale=np.asarray(package["bilinear_feature_scale"], dtype="<f8"),
        train_predicted_force_eV_A=np.asarray(thermal92_prediction, dtype="<f8"),
    )
    summary = {
        "format": "graphene_r2ao_step32_all132_spectral_candidate_v1",
        "status": "R2AO_FROZEN_FOR_SPECTRAL_SENSITIVITY_MECHANICS_PENDING",
        "deployable": False,
        "force_gate_approved": False,
        "authorized_use": (
            "300/450/600 K SSCHA spectral sensitivity and quantitative Kohn-cusp "
            "validation only; not a generally validated MLIP"
        ),
        "representation": "R2R 65 + fixed forward-selected 64 bilinear columns",
        "total_column_count": 129,
        "selected_bilinear_indices": package[
            "selected_bilinear_indices"
        ].tolist(),
        "all132_train_metrics": train_metrics,
        "development_OOF_metrics": oof_metrics,
        "development_OOF_gate_passed": oof_metrics["passes_fixed_gate"],
        "selection_boundary": {
            "path": str(boundary_path),
            "sha256": file_sha256(boundary_path),
            "status": boundary["status"],
        },
        "coefficient_raw_sha256": r2r1.raw_array_sha256(coefficient, "<f8"),
        "train_prediction_raw_sha256": r2r1.raw_array_sha256(prediction, "<f8"),
        "runtime_thermal92_prediction_raw_sha256": r2r1.raw_array_sha256(
            thermal92_prediction, "<f8"
        ),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "input_sha256": {
            "training_package": package_receipt["package_sha256"],
            "training_package_receipt": file_sha256(args.package_root / "receipt.json"),
        },
        "design_input_receipts": data_receipt,
        "energy_labels_used": False,
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "The fixed all-132 development OOF force/A-prime RMS gate failed.",
            "Stable restoring slopes motivate a spectral sensitivity run but do not erase that failure.",
            "A final claim requires model-sensitivity plus quantitative Kohn-cusp acceptance.",
        ],
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
