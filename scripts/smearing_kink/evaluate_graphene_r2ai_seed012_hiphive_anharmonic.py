#!/usr/bin/env python3
"""Evaluate symmetry-reduced cubic/quartic force-constant tails on seed012.

The frozen R2AE fold-specific readout supplies the baseline.  This script
uses hiphive to construct P6/mmm- and ASR-constrained force designs containing
only third- and, optionally, fourth-order force constants.  Since no
second-order parameter is fitted, the added energy, force, and Hessian vanish
at the reference while the finite-amplitude thermal forces can change.

All hyperparameters are evaluated by the same four complete-configuration
folds used by R2AD/R2AG.  The opened seed0/1/2 trajectories are development
data.  The planned 525 K off-policy set is not accessed.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from hiphive import ClusterSpace, StructureContainer
from hiphive.utilities import prepare_structure

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
DEFAULT_OUTPUT = BASE / "R2AI_seed012_hiphive_anharmonic_20260827"
FORCE_SCALE_EV_A = 0.030
APRIME_SCALE_EV_A = 0.015
ALPHAS = tuple(float(value) for value in np.logspace(-6.0, 2.0, 25))

REPRESENTATIONS: tuple[dict[str, Any], ...] = (
    {"name": "fc3_r2p5", "cutoffs_A": (4.5, 2.5), "orders": (3,)},
    {"name": "fc3_r3p0", "cutoffs_A": (4.5, 3.0), "orders": (3,)},
    {"name": "fc3_r4p0", "cutoffs_A": (4.5, 4.0), "orders": (3,)},
    {
        "name": "fc3_r3p0_fc4_r2p0",
        "cutoffs_A": (4.5, 3.0, 2.0),
        "orders": (3, 4),
    },
    {
        "name": "fc3_r3p0_fc4_r2p5",
        "cutoffs_A": (4.5, 3.0, 2.5),
        "orders": (3, 4),
    },
)

PROJECTION_MASSES = (0.50, 0.70, 0.85, 0.95, 0.99)
SEED_WEIGHT_PROFILES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("equal", (1.0 / 3.0,) * 3),
    ("seed2_half", (0.25, 0.25, 0.50)),
    ("seed0_40", (0.40, 0.30, 0.30)),
)
FORCE_GROUP_PROFILES: tuple[tuple[str, dict[str, float]], ...] = (
    (
        "equal",
        {
            "E50_seed0": 0.20,
            "E50_seed1": 0.20,
            "E50_seed2": 0.20,
            "T300": 0.20,
            "T600": 0.20,
        },
    ),
    (
        "thermal_focus",
        {
            "E50_seed0": 0.05,
            "E50_seed1": 0.05,
            "E50_seed2": 0.05,
            "T300": 0.20,
            "T600": 0.65,
        },
    ),
    (
        "T600_focus",
        {
            "E50_seed0": 0.0375,
            "E50_seed1": 0.0375,
            "E50_seed2": 0.0375,
            "T300": 0.0375,
            "T600": 0.85,
        },
    ),
)


def prepare_geometries() -> tuple[list[Any], Any, dict[str, str], str]:
    structures, reference, hashes = r2ag.structures_and_reference()
    prepared = []
    for structure in structures:
        current = structure.copy()
        current.arrays["forces"] = np.zeros((72, 3), dtype=np.float64)
        converted = prepare_structure(current, reference, check_permutation=False)
        # Extxyz positions are printed to finite precision.  Remove the
        # resulting sub-micro-Angstrom rigid shift; ASR makes this operation
        # physically null and it matches the production alignment convention.
        converted.arrays["displacements"] -= np.mean(
            converted.arrays["displacements"], axis=0, keepdims=True
        )
        prepared.append(converted)
    displacements = np.stack(
        [np.asarray(item.arrays["displacements"], dtype=np.float64) for item in prepared]
    )
    if displacements.shape != (132, 72, 3):
        raise ValueError("R2AI prepared displacement schema changed")
    mean_translation = np.max(np.abs(np.mean(displacements, axis=1)))
    if mean_translation > 2.0e-15:
        raise ValueError("R2AI prepared structures contain a large translation")
    return prepared, reference, hashes, r2r1.raw_array_sha256(displacements, "<f8")


def materialize_design(
    prepared: Sequence[Any], reference: Any, specification: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    cluster_space = ClusterSpace(
        reference,
        list(specification["cutoffs_A"]),
        symprec=1.0e-5,
        acoustic_sum_rules=True,
    )
    container = StructureContainer(cluster_space)
    for index, structure in enumerate(prepared):
        container.add_structure(structure, global_index=index)
    fit_matrix, target = container.get_fit_data()
    if fit_matrix.shape != (132 * 72 * 3, cluster_space.n_dofs):
        raise ValueError("R2AI hiphive fit-matrix shape changed")
    if target.shape != (132 * 72 * 3,) or np.max(np.abs(target)) != 0.0:
        raise ValueError("R2AI dummy hiphive target changed")
    parameter_indices = np.asarray(
        [
            index
            for order in specification["orders"]
            for index in cluster_space.get_parameter_indices(order)
        ],
        dtype=int,
    )
    if len(parameter_indices) == 0 or np.any(
        np.isin(parameter_indices, cluster_space.get_parameter_indices(2))
    ):
        raise ValueError("R2AI tail unexpectedly contains second-order parameters")
    design = np.asarray(
        fit_matrix[:, parameter_indices].reshape(132, 72, 3, -1),
        dtype=np.float64,
    )
    flat = design.reshape(-1, design.shape[-1])
    scale = np.sqrt(np.mean(np.square(flat), axis=0))
    normalized = flat / scale[None, :]
    singular = np.linalg.svd(normalized, compute_uv=False)
    net_force = np.sum(design, axis=1)
    receipt = {
        "name": specification["name"],
        "cutoffs_A": list(specification["cutoffs_A"]),
        "included_orders": list(specification["orders"]),
        "excluded_order2": True,
        "cluster_space_total_dofs": int(cluster_space.n_dofs),
        "dofs_by_order": {
            str(order): int(cluster_space.get_n_dofs_by_order(order))
            for order in range(2, len(specification["cutoffs_A"]) + 2)
        },
        "tail_parameter_indices": parameter_indices.tolist(),
        "tail_width": int(design.shape[-1]),
        "normalized_design_rank": int(np.linalg.matrix_rank(normalized)),
        "normalized_design_condition": float(singular[0] / singular[-1]),
        "column_RMS_min_relative": float(np.min(scale) / np.max(scale)),
        "max_net_force_column_eV_A_per_parameter": float(np.max(np.abs(net_force))),
        "force_design_raw_sha256": r2r1.raw_array_sha256(design, "<f8"),
        "reference_energy_force_Hessian_zero": True,
        "space_group": "P6/mmm (191)",
        "acoustic_sum_rules": True,
        "hiphive_version": __import__("hiphive").__version__,
    }
    return design, receipt


def baseline_by_fold(
    data: dict[str, np.ndarray], package: dict[str, np.ndarray]
) -> tuple[list[np.ndarray], float]:
    output = []
    replay = 0.0
    for fold, hold_values in enumerate(package["fold_hold_indices"]):
        coefficient = package["fold_linear_skip_coefficient"][fold]
        prediction = data["fixed"] + np.einsum(
            "natk,k->nat", data["design"], coefficient, optimize=True
        )
        hold = np.asarray(hold_values, dtype=int)
        replay = max(
            replay,
            float(
                np.max(
                    np.abs(
                        prediction[hold]
                        - package["fixed_step32_OOF_predicted_force_eV_A"][hold]
                    )
                )
            ),
        )
        output.append(prediction)
    if replay > 2.0e-12:
        raise ValueError("R2AI frozen fold baseline replay changed")
    return output, replay


def fold_statistics(
    design: np.ndarray,
    baseline: np.ndarray,
    reference: np.ndarray,
    modes: np.ndarray,
    train: np.ndarray,
) -> dict[str, Any]:
    width = design.shape[-1]
    normalizer = np.sqrt(np.mean(np.square(design[train]), axis=(0, 1, 2)))
    if np.any(normalizer <= np.max(normalizer) * 1.0e-12):
        raise ValueError("R2AI fold design contains a zero column")
    normalized = design / normalizer[None, None, None, :]
    residual = reference - baseline
    group_gram: dict[str, np.ndarray] = {}
    group_rhs: dict[str, np.ndarray] = {}
    for name, bounds in r2ad.GROUP_RANGES.items():
        members = np.intersect1d(train, r2ad.indices_for(bounds))
        matrix = normalized[members].reshape(-1, width) / FORCE_SCALE_EV_A
        target = residual[members].reshape(-1) / FORCE_SCALE_EV_A
        group_gram[name] = matrix.T @ matrix / float(matrix.shape[0])
        group_rhs[name] = matrix.T @ target / float(matrix.shape[0])
    projection_gram: dict[str, np.ndarray] = {}
    projection_rhs: dict[str, np.ndarray] = {}
    for name, bounds in r2ad.SEED_RANGES.items():
        members = np.intersect1d(train, r2ad.indices_for(bounds))
        matrix = np.einsum(
            "nat,natk->nk",
            np.conj(modes[members]),
            normalized[members],
            optimize=True,
        ) / APRIME_SCALE_EV_A
        target = np.einsum(
            "nat,nat->n",
            np.conj(modes[members]),
            residual[members],
            optimize=True,
        ) / APRIME_SCALE_EV_A
        projection_gram[name] = (
            matrix.real.T @ matrix.real + matrix.imag.T @ matrix.imag
        ) / (2.0 * len(members))
        projection_rhs[name] = (
            matrix.real.T @ target.real + matrix.imag.T @ target.imag
        ) / (2.0 * len(members))
    return {
        "normalizer": normalizer,
        "group_gram": group_gram,
        "group_rhs": group_rhs,
        "projection_gram": projection_gram,
        "projection_rhs": projection_rhs,
    }


def coefficient_matrix(
    statistics: dict[str, Any],
    projection_mass: float,
    seed_weights: Sequence[float],
    force_group_masses: dict[str, float],
) -> np.ndarray:
    width = len(statistics["normalizer"])
    force_gram = np.zeros((width, width))
    force_rhs = np.zeros(width)
    for name, mass in force_group_masses.items():
        force_gram += mass * statistics["group_gram"][name]
        force_rhs += mass * statistics["group_rhs"][name]
    projection_gram = np.zeros((width, width))
    projection_rhs = np.zeros(width)
    for name, weight in zip(r2ad.SEED_RANGES, seed_weights, strict=True):
        projection_gram += weight * statistics["projection_gram"][name]
        projection_rhs += weight * statistics["projection_rhs"][name]
    gram = (1.0 - projection_mass) * force_gram + projection_mass * projection_gram
    rhs = (1.0 - projection_mass) * force_rhs + projection_mass * projection_rhs
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    tolerance = np.finfo(float).eps * width * max(float(eigenvalues[-1]), 1.0)
    if eigenvalues[0] < -10.0 * tolerance:
        raise ValueError("R2AI normal matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ rhs
    normalized_coefficient = eigenvectors @ (
        projected_rhs[:, None]
        / (eigenvalues[:, None] + np.asarray(ALPHAS)[None, :])
    )
    return normalized_coefficient / statistics["normalizer"][:, None]


def scan_representation(
    design: np.ndarray,
    data: dict[str, np.ndarray],
    package: dict[str, np.ndarray],
    baselines: Sequence[np.ndarray],
    representation: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    all_indices = np.arange(132, dtype=int)
    fold_stats = []
    for fold, hold_values in enumerate(package["fold_hold_indices"]):
        hold = np.asarray(hold_values, dtype=int)
        train = np.setdiff1d(all_indices, hold)
        fold_stats.append(
            fold_statistics(
                design,
                baselines[fold],
                data["reference"],
                data["modes"],
                train,
            )
        )
    best_score = np.inf
    best_prediction: np.ndarray | None = None
    best_record: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    for projection_mass in PROJECTION_MASSES:
        for seed_profile, seed_weights in SEED_WEIGHT_PROFILES:
            for force_profile, force_masses in FORCE_GROUP_PROFILES:
                oof = np.full((132, 72, 3, len(ALPHAS)), np.nan)
                for fold, hold_values in enumerate(package["fold_hold_indices"]):
                    hold = np.asarray(hold_values, dtype=int)
                    coefficient = coefficient_matrix(
                        fold_stats[fold],
                        projection_mass,
                        seed_weights,
                        force_masses,
                    )
                    oof[hold] = baselines[fold][hold][..., None] + np.einsum(
                        "natk,kq->natq", design[hold], coefficient, optimize=True
                    )
                if not np.all(np.isfinite(oof)):
                    raise RuntimeError("R2AI OOF coverage failed")
                scores = r2ad.vectorized_scores(oof, data)
                alpha_id = int(np.argmin(scores))
                record = {
                    "representation": representation["name"],
                    "projection_mass": projection_mass,
                    "seed_weight_profile": seed_profile,
                    "seed_weights": {
                        name: float(weight)
                        for name, weight in zip(
                            r2ad.SEED_RANGES, seed_weights, strict=True
                        )
                    },
                    "force_group_profile": force_profile,
                    "force_group_masses": force_masses,
                    "alpha": ALPHAS[alpha_id],
                    "raw_gate_score": float(scores[alpha_id]),
                }
                records.append(record)
                if scores[alpha_id] < best_score:
                    best_score = float(scores[alpha_id])
                    best_prediction = oof[..., alpha_id].copy()
                    best_record = record
    if best_prediction is None or best_record is None:
        raise RuntimeError("R2AI representation scan produced no candidate")
    metrics = r2ad.scalar_metrics(best_prediction, data)
    if abs(metrics["raw_gate_score"] - best_score) > 1.0e-10:
        raise ValueError("R2AI vectorized/scalar score mismatch")
    records.sort(key=lambda item: (item["raw_gate_score"], item["alpha"]))
    return {**best_record, "full_gate": metrics}, best_prediction, records


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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    logging.getLogger("hiphive").setLevel(logging.WARNING)

    package, package_receipt = r2ag.load_package(args.package_root.resolve())
    data, data_receipt = r2ad.load_training(args)
    prepared, reference, geometry_hashes, displacement_hash = prepare_geometries()
    baselines, baseline_replay = baseline_by_fold(data, package)

    representation_records = []
    all_candidates = []
    best_global: dict[str, Any] | None = None
    best_global_prediction: np.ndarray | None = None
    best_global_design: np.ndarray | None = None
    best_global_receipt: dict[str, Any] | None = None
    for specification in REPRESENTATIONS:
        design, design_receipt = materialize_design(
            prepared, reference, specification
        )
        best, prediction, candidates = scan_representation(
            design, data, package, baselines, specification
        )
        representation_records.append(
            {"design": design_receipt, "best": best}
        )
        all_candidates.extend(candidates[:50])
        if best_global is None or best["raw_gate_score"] < best_global["raw_gate_score"]:
            best_global = best
            best_global_prediction = prediction
            best_global_design = design.copy()
            best_global_receipt = design_receipt
        print(
            json.dumps(
                {
                    "representation_complete": specification["name"],
                    "tail_width": design.shape[-1],
                    "raw_gate_score": best["raw_gate_score"],
                    "passes_fixed_gate": best["full_gate"]["passes_fixed_gate"],
                    "Aprime_by_seed": best["full_gate"]["Aprime_by_seed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del design, prediction

    if (
        best_global is None
        or best_global_prediction is None
        or best_global_design is None
        or best_global_receipt is None
    ):
        raise RuntimeError("R2AI global representation scan produced no candidate")
    status = (
        "R2AI_HIPHIVE_ANHARMONIC_OOF_GATE_PASSED_FINAL_FIT_PENDING"
        if best_global["full_gate"]["passes_fixed_gate"]
        else "R2AI_HIPHIVE_ANHARMONIC_OOF_GATE_FAILED"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "best_OOF_predicted_force_eV_A.npy"
    design_path = output / "best_tail_force_design_eV_A_per_parameter.npy"
    np.save(
        prediction_path,
        np.asarray(best_global_prediction, dtype="<f8"),
        allow_pickle=False,
    )
    np.save(
        design_path,
        np.asarray(best_global_design, dtype="<f8"),
        allow_pickle=False,
    )
    summary = {
        "format": "graphene_r2ai_seed012_hiphive_anharmonic_OOF_v1",
        "status": status,
        "deployable": False,
        "scope": "seed012 + T300 + T600 development; 525 K unopened",
        "construction": "P6/mmm + acoustic-sum-rule constrained hiphive force-constant tail; orders >=3 only",
        "analytic_reference_jet": {
            "energy_zero": True,
            "force_zero": True,
            "Hessian_zero": True,
            "second_order_parameters_fitted": False,
        },
        "representations": representation_records,
        "best": best_global,
        "best_design_receipt": best_global_receipt,
        "top_candidates": sorted(
            all_candidates,
            key=lambda item: (item["raw_gate_score"], item["alpha"]),
        )[:200],
        "alphas": list(ALPHAS),
        "projection_masses": list(PROJECTION_MASSES),
        "seed_weight_profiles": {
            name: list(values) for name, values in SEED_WEIGHT_PROFILES
        },
        "force_group_profiles": {
            name: values for name, values in FORCE_GROUP_PROFILES
        },
        "fold_linear_skip_replay_max_abs_eV_A": baseline_replay,
        "input_sha256": {
            "training_package": package_receipt["package_sha256"],
            "training_package_receipt": file_sha256(
                args.package_root / "receipt.json"
            ),
            "prepared_displacements_raw": displacement_hash,
            **geometry_hashes,
        },
        "design_input_receipts": data_receipt,
        "best_prediction_sha256": file_sha256(prediction_path),
        "best_prediction_raw_sha256": r2r1.raw_array_sha256(
            best_global_prediction, "<f8"
        ),
        "best_design_sha256": file_sha256(design_path),
        "best_design_raw_sha256": r2r1.raw_array_sha256(
            best_global_design, "<f8"
        ),
        "energy_labels_used": False,
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_boundary": [
            "This is development OOF selection, not independent validation.",
            "A pass still requires an all132 fit, force-constant runtime/mechanics audit, and the unopened 525 K test.",
        ],
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(status + "\n")
    print(json.dumps({"status": status, "best": best_global}, indent=2))


if __name__ == "__main__":
    main()
