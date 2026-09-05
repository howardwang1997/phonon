#!/usr/bin/env python3
"""Evaluate a conservative global-quadratic readout on R2Y training data.

Geometry-only PCA coordinates are formed from frozen scalar-energy columns.
Pairwise products of those coordinates are scalar energies, with forces built
analytically from the frozen conservative force columns.  The construction
therefore remains conservative and inherits the Taylor-null reference jet.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_graphene_r2z_forward_bilinear as r2z
import fit_graphene_r2y_seed01_paired_readout as r2y
import graphene_r2r1_linear_readout as r2r1
import refine_graphene_r2z_step28 as refine
import scan_graphene_r2z_step28_full_gate as full_scan
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_PATH_ROOT = BASE / "R2Z_forward_bilinear_development_32step_20260826"
DEFAULT_OUTPUT = BASE / "R2AA_global_quadratic_development_20260826"
PCA_WIDTHS = (4, 8, 12, 16)
MASSES = (0.50, 0.70, 0.85, 0.95)
SEED1_WEIGHTS = (0.35, 0.45, 0.55)
TAIL_PENALTIES = (0.10, 0.30, 1.00, 3.00)
T600_FORCE_FRACTIONS = (0.25, 0.50, 0.75)
ALPHAS = tuple(float(value) for value in np.logspace(-4.0, 0.0, 17))
FORCE_SCALE = 0.030


def canonicalize_sign(vectors: np.ndarray) -> np.ndarray:
    output = np.array(vectors, dtype=np.float64, copy=True)
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0.0:
            output[:, column] *= -1.0
    return output


def load_energy_design(
    args: argparse.Namespace, selected_bilinear: list[int]
) -> tuple[np.ndarray, dict[str, str]]:
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        base92 = np.asarray(
            arrays["thermal_parameter_energy_design_eV"], dtype=np.float64
        )
    seed1_selected_path = args.seed1_selected_root / "seed1_selected_design.npz"
    with np.load(seed1_selected_path, allow_pickle=False) as arrays:
        base1 = np.asarray(arrays["parameter_energy_design_eV"], np.float64)[
            :, :65
        ]
    thermal_energy_path = args.thermal_full_root / "bilinear_energy_design_eV.npy"
    seed1_energy_path = args.seed1_full_root / "bilinear_energy_design_eV.npy"
    thermal_full = np.load(thermal_energy_path, allow_pickle=False)
    seed1_full = np.load(seed1_energy_path, allow_pickle=False)
    full = np.concatenate((thermal_full[:20], seed1_full, thermal_full[20:]))
    base = np.concatenate((base92[:20], base1, base92[20:]))
    energy = np.concatenate((base, full[:, selected_bilinear]), axis=1)
    return energy, {
        "aggregate": file_sha256(args.aggregate),
        "seed1_selected": file_sha256(seed1_selected_path),
        "thermal_full_energy": file_sha256(thermal_energy_path),
        "seed1_full_energy": file_sha256(seed1_energy_path),
    }


def quadratic_basis(
    energy: np.ndarray,
    force: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = np.mean(energy, axis=0)
    scale = np.sqrt(np.mean(np.square(energy - mean), axis=0))
    if np.any(scale <= np.max(scale) * 1.0e-12):
        raise ValueError("R2AA energy coordinate contains a zero column")
    standardized = (energy - mean) / scale
    _u, singular, vectors_t = np.linalg.svd(standardized, full_matrices=False)
    vectors = canonicalize_sign(vectors_t.T[:, :width])
    transform = vectors / scale[:, None]
    coordinate = standardized @ vectors
    coordinate_reference = (-mean / scale) @ vectors
    coordinate_force = np.einsum(
        "natk,kp->natp", force, transform, optimize=True
    )
    pairs = [(left, right) for left in range(width) for right in range(left, width)]
    q_energy = np.empty((len(energy), len(pairs)))
    q_force = np.empty((len(energy), 72, 3, len(pairs)))
    for column, (left, right) in enumerate(pairs):
        q_energy[:, column] = (
            coordinate[:, left] * coordinate[:, right]
            - coordinate_reference[left] * coordinate_reference[right]
        )
        q_force[..., column] = (
            coordinate[:, left, None, None] * coordinate_force[..., right]
            + coordinate[:, right, None, None] * coordinate_force[..., left]
        )
    receipt = {
        "PCA_width": width,
        "quadratic_column_count": len(pairs),
        "energy_scale_min_relative": float(np.min(scale) / np.max(scale)),
        "singular_values_first": singular[:width].tolist(),
        "transform_raw_sha256": r2r1.raw_array_sha256(transform, "<f8"),
        "coordinate_reference": coordinate_reference.tolist(),
        "pair_order": [list(pair) for pair in pairs],
        "analytic_force_product_rule": True,
        "reference_value_force_Hessian_zero": True,
    }
    return q_energy, q_force, receipt


def coefficient_matrix(
    system: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    normalizer, gram, rhs = system
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo(float).eps * len(normalizer) * max(
        float(eigenvalues[-1]), 1.0
    )
    if eigenvalues[0] < -10.0 * tolerance:
        raise ValueError("R2AA Gram is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ rhs
    normalized = eigenvectors @ (
        projected_rhs[:, None] / (eigenvalues[:, None] + np.asarray(ALPHAS)[None, :])
    )
    return FORCE_SCALE * normalized / normalizer[:, None]


def scan_representation(
    data: dict[str, np.ndarray], design: np.ndarray, width: int
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    statistics = r2z.fold_raw_statistics(
        design, data["fixed"], data["reference"], data["modes"]
    )
    columns = np.arange(design.shape[-1])
    records = []
    best_score = np.inf
    best_prediction: np.ndarray | None = None
    best_record: dict[str, Any] | None = None
    for mass in MASSES:
        for seed1_weight in SEED1_WEIGHTS:
            for penalty in TAIL_PENALTIES:
                for t600_fraction in T600_FORCE_FRACTIONS:
                    oof = np.full((112, 72, 3, len(ALPHAS)), np.nan)
                    for hold_id, fold_tuple in enumerate(r2y.FOLDS):
                        hold = np.asarray(fold_tuple)
                        train_folds = tuple(
                            value for value in range(4) if value != hold_id
                        )
                        coefficient = coefficient_matrix(
                            refine.normal_equations(
                                statistics,
                                train_folds,
                                columns,
                                mass,
                                seed1_weight,
                                penalty,
                                t600_fraction,
                            )
                        )
                        matrix = design[hold].reshape(-1, design.shape[-1])
                        prediction = (
                            data["fixed"][hold].reshape(-1, 1)
                            + matrix @ coefficient
                        )
                        oof[hold] = prediction.reshape(
                            len(hold), 72, 3, len(ALPHAS)
                        )
                    scores = full_scan.vectorized_full_scores(oof, data)
                    alpha_id = int(np.argmin(scores))
                    record = {
                        "PCA_width": width,
                        "total_column_count": int(design.shape[-1]),
                        "raw_gate_score": float(scores[alpha_id]),
                        "projection_mass": mass,
                        "seed1_projection_weight": seed1_weight,
                        "tail_penalty": penalty,
                        "T600_force_fraction": t600_fraction,
                        "group_masses": refine.group_masses(t600_fraction),
                        "alpha": ALPHAS[alpha_id],
                    }
                    records.append(record)
                    if scores[alpha_id] < best_score:
                        best_score = float(scores[alpha_id])
                        best_prediction = oof[..., alpha_id].copy()
                        best_record = record
    if best_prediction is None or best_record is None:
        raise RuntimeError("R2AA scan produced no candidate")
    metrics = r2y.metrics(
        best_prediction,
        data["reference"],
        data["modes"],
        data["coordinates"],
        data["base"],
        np.arange(112),
    )
    if abs(metrics["raw_gate_score"] - best_score) > 1.0e-10:
        raise ValueError("R2AA vectorized/scalar score mismatch")
    records.sort(key=lambda item: (item["raw_gate_score"], item["alpha"]))
    return {**best_record, "full_gate": metrics}, best_prediction, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=r2z.DEFAULT_AGGREGATE)
    parser.add_argument("--failed-root", type=Path, default=r2z.DEFAULT_R2Y_FAILED)
    parser.add_argument(
        "--seed1-selected-root", type=Path, default=r2z.DEFAULT_SEED1_SELECTED
    )
    parser.add_argument("--thermal-full-root", type=Path, default=r2z.DEFAULT_THERMAL_FULL)
    parser.add_argument("--seed1-full-root", type=Path, default=r2z.DEFAULT_SEED1_FULL)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    data, input_receipt = r2z.load_full_training(args)
    path_summary_path = args.path_root / "summary.json"
    path_summary = json.loads(path_summary_path.read_text())
    step = path_summary["steps"][31]
    selected = list(step["starting_bilinear_indices"]) + [
        int(step["best"]["selected_bilinear_index"])
    ]
    if len(selected) != 64 or len(set(selected)) != 64:
        raise ValueError("R2AA step-32 representation changed")
    linear_columns = np.r_[np.arange(65), 65 + np.asarray(selected)]
    linear_force = data["design"][..., linear_columns]
    linear_energy, energy_receipt = load_energy_design(args, selected)
    if linear_energy.shape != (112, 129) or linear_force.shape != (112, 72, 3, 129):
        raise ValueError("R2AA linear scalar basis shape changed")

    representation_records = []
    all_system_records = []
    best_global: dict[str, Any] | None = None
    best_global_prediction: np.ndarray | None = None
    pca_receipts = []
    for width in PCA_WIDTHS:
        _q_energy, q_force, pca_receipt = quadratic_basis(
            linear_energy, linear_force, width
        )
        design = np.concatenate((linear_force, q_force), axis=-1)
        best, prediction, records = scan_representation(data, design, width)
        pca_receipts.append(pca_receipt)
        representation_records.append(best)
        all_system_records.extend(records[:50])
        if best_global is None or (
            best["full_gate"]["raw_gate_score"], best["total_column_count"]
        ) < (
            best_global["full_gate"]["raw_gate_score"],
            best_global["total_column_count"],
        ):
            best_global = best
            best_global_prediction = prediction
        print(
            json.dumps(
                {
                    "PCA_width_complete": width,
                    "raw_gate_score": best["full_gate"]["raw_gate_score"],
                    "passes_fixed_gate": best["full_gate"]["passes_fixed_gate"],
                }
            ),
            flush=True,
        )
    if best_global is None or best_global_prediction is None:
        raise RuntimeError("R2AA produced no representation")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.save(
        output / "best_OOF_predicted_force_eV_A.npy",
        np.asarray(best_global_prediction, dtype="<f8"),
        allow_pickle=False,
    )
    summary = {
        "format": "graphene_r2aa_global_quadratic_development_v1",
        "status": (
            "R2AA_GLOBAL_QUADRATIC_DEVELOPMENT_GATE_PASSED"
            if best_global["full_gate"]["passes_fixed_gate"]
            else "R2AA_GLOBAL_QUADRATIC_DEVELOPMENT_GATE_FAILED"
        ),
        "scope": "seed0 + seed1 + T300 + T600 development; 525 K unopened",
        "construction": "geometry-only PCA of conservative scalar energies followed by pairwise scalar products",
        "selected_linear_bilinear_indices": selected,
        "PCA_widths": list(PCA_WIDTHS),
        "PCA_receipts": pca_receipts,
        "representation_best": representation_records,
        "best": best_global,
        "top_system_candidates": sorted(
            all_system_records,
            key=lambda item: (item["raw_gate_score"], item["total_column_count"]),
        )[:200],
        "best_OOF_raw_sha256": r2r1.raw_array_sha256(
            best_global_prediction, "<f8"
        ),
        "energy_design_input_sha256": energy_receipt,
        "path_summary_sha256": file_sha256(path_summary_path),
        "input_receipt": input_receipt,
        "energy_labels_used": False,
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps({"status": summary["status"], "best": best_global}, indent=2))


if __name__ == "__main__":
    main()
