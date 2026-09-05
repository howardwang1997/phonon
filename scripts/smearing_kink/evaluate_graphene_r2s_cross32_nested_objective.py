#!/usr/bin/env python3
"""Nested mass+alpha OOF for the fixed R2S cross32 representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import (
    DEFAULT_AGGREGATE,
    _parse_whitelist,
    canonical_json_bytes,
    file_sha256,
)
from diagnose_graphene_r2s_nonlinear_readout_basis import (
    _predict,
    _ridge_system,
    _selection_key,
    design_audit,
)


ROOT = Path(__file__).resolve().parents[2]
DESIGN_ROOT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/R2S_node_nonlinear_basis_diagnostic_20260826"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/R2S_cross32_nested_mass_alpha_20260826"
)
PROJECTION_MASS_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
CROSS32_EXTRA_INDICES = np.arange(64, 96, dtype=int)


def nested_mass_alpha(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    aprime: r2r1.AprimeData,
) -> tuple[np.ndarray, dict[str, Any]]:
    all_indices = np.arange(92, dtype=int)
    oof = np.full_like(reference, np.nan)
    outer_records = []
    for outer_id, outer_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
        outer_hold = np.asarray(outer_tuple, dtype=int)
        outer_train = np.setdiff1d(all_indices, outer_hold)
        predictions = {
            (mass, alpha): np.full_like(reference, np.nan)
            for mass in PROJECTION_MASS_GRID
            for alpha in r2r1.ALPHA_GRID
        }
        for inner_id, inner_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
            if inner_id == outer_id:
                continue
            inner_hold = np.asarray(inner_tuple, dtype=int)
            inner_train = np.setdiff1d(outer_train, inner_hold)
            for mass in PROJECTION_MASS_GRID:
                system = _ridge_system(
                    design,
                    fixed,
                    reference,
                    aprime,
                    inner_train,
                    mass,
                )
                for alpha in r2r1.ALPHA_GRID:
                    predictions[(mass, alpha)][inner_hold] = _predict(
                        design[inner_hold],
                        fixed[inner_hold],
                        system.coefficient(alpha),
                    )
        candidates = []
        for mass in PROJECTION_MASS_GRID:
            for alpha in r2r1.ALPHA_GRID:
                metrics = r2r1.gate_metrics(
                    predictions[(mass, alpha)], reference, aprime, outer_train
                )
                candidates.append(
                    {
                        "projection_mass": mass,
                        "alpha": float(alpha),
                        "metrics": metrics,
                    }
                )
        # Lowest rounded gate score; then lower projection mass (less targeted
        # intervention); then larger alpha, matching the R2R-1 ridge tie rule.
        selected = min(
            candidates,
            key=lambda item: (
                float(item["metrics"]["selection_score_rounded_12"]),
                float(item["projection_mass"]),
                -float(item["alpha"]),
            ),
        )
        system = _ridge_system(
            design,
            fixed,
            reference,
            aprime,
            outer_train,
            float(selected["projection_mass"]),
        )
        coefficient = system.coefficient(float(selected["alpha"]))
        oof[outer_hold] = _predict(
            design[outer_hold], fixed[outer_hold], coefficient
        )
        outer_records.append(
            {
                "outer_fold": outer_id,
                "outer_hold_global_indices": outer_hold.tolist(),
                "selected_projection_mass": selected["projection_mass"],
                "selected_alpha": selected["alpha"],
                "selected_inner_metrics": selected["metrics"],
                "candidate_count": len(candidates),
                "outer_coefficient_raw_sha256": r2r1.raw_array_sha256(
                    coefficient, "<f8"
                ),
            }
        )
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("nested mass+alpha OOF does not cover thermal92")
    pooled = r2r1.gate_metrics(oof, reference, aprime, all_indices)
    return oof, {"outer_records": outer_records, "pooled_OOF_metrics": pooled}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument(
        "--design-npz",
        type=Path,
        default=DESIGN_ROOT / "node_nonlinear_design_and_oof.npz",
    )
    parser.add_argument(
        "--design-summary", type=Path, default=DESIGN_ROOT / "summary.json"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference, aprime, label_hashes = _parse_whitelist(r2r1.RECOMMENDED_THERMAL92)
    design_summary = json.loads(args.design_summary.read_text())
    with np.load(args.design_npz, allow_pickle=False) as arrays:
        extra_force = np.asarray(arrays["nonlinear_force_design"], np.float64)
    if r2r1.raw_array_sha256(extra_force, "<f8") != design_summary["materialization"][
        "force_raw_sha256"
    ]:
        raise ValueError("fixed nonlinear force design differs from its receipt")
    if file_sha256(args.aggregate.resolve()) != r2r1.ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("attempt3 aggregate SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        base_force = np.asarray(
            arrays["thermal_parameter_force_design_eV_A"], np.float64
        )
        fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], np.float64)
    design = np.concatenate(
        (base_force, extra_force[..., CROSS32_EXTRA_INDICES]), axis=3
    )
    audit = design_audit(design)
    if audit != {
        "width": 97,
        "rank": 97,
        "condition": 23754.407134984984,
        "min_relative_column_RMS": 0.0028402039950859805,
        "pass": True,
    }:
        raise ValueError("cross32 design audit differs from frozen diagnostic")

    oof, nested = nested_mass_alpha(design, fixed, reference, aprime)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "OOF_predictions.npz", OOF_predicted_force_eV_A=oof)
    passed = bool(nested["pooled_OOF_metrics"]["passes_fixed_gate"])
    summary = {
        "format": "graphene_r2s_cross32_nested_projection_mass_alpha_oof_v1",
        "status": (
            "R2S_CROSS32_NESTED_OOF_PASSED_DEVELOPMENT_ONLY"
            if passed
            else "R2S_CROSS32_NESTED_OOF_FAILED"
        ),
        "deployable": False,
        "representation": (
            "frozen R2R 65 plus sum_i a_i phi_i^(interaction1) "
            "phi_i^(interaction2), with and without c_i, 32 added columns"
        ),
        "projection_mass_grid": list(PROJECTION_MASS_GRID),
        "alpha_grid": list(r2r1.ALPHA_GRID),
        "selection_tie_break": "lower projection mass, then larger alpha",
        "design_audit": audit,
        "nested_OOF": nested,
        "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
        "input_sha256": {
            "thermal92": file_sha256(r2r1.RECOMMENDED_THERMAL92),
            "aggregate_arrays": file_sha256(args.aggregate),
            "node_nonlinear_design_npz": file_sha256(args.design_npz),
            "node_nonlinear_design_summary": file_sha256(args.design_summary),
        },
        "label_raw_sha256": label_hashes,
        "energy_labels_used": False,
        "development_or_held_access": False,
        "interpretation": (
            "conditional nested readout OOF; representation was developed on thermal92 "
            "and therefore still requires independent data"
        ),
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
