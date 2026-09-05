#!/usr/bin/env python3
"""Four-fold capacity diagnostic for a conservative global cubic A-prime tail.

The tail energy is a linear combination of

    q_r z_j z_k  and  q_i z_j z_k,

where ``q`` is the frozen folded-K A-prime coordinate and ``z`` are
geometry-only principal coordinates of the translation-free 6x6
displacement.  Every term is cubic, so its value, force, and Hessian vanish
at the reference.  The representation is deliberately fixed-cell and is a
capacity diagnostic only; passing here does not make it deployable.

The frozen R2AE fold-specific linear skip is retained.  Only the cubic tail
is fitted, using A-prime force projections from the three promoted
development trajectories.  The 525 K off-policy set is never opened.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
DEFAULT_OUTPUT = BASE / "R2AH_seed012_global_cubic_diagnostic_20260827"
PCA_WIDTHS = (4, 8, 12, 16, 20, 24)
ALPHAS = tuple(float(value) for value in np.logspace(-5.0, 3.0, 25))
TAIL_SCALES = (0.25, 0.50, 0.75, 1.00)
APRIME_SCALE_EV_A = 0.015


def canonicalize_sign(vectors: np.ndarray) -> np.ndarray:
    output = np.array(vectors, dtype=np.float64, copy=True)
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0.0:
            output[:, column] *= -1.0
    return output


def translation_free_displacements() -> tuple[np.ndarray, dict[str, str]]:
    structures, reference, hashes = r2ag.structures_and_reference()
    reference_positions = np.asarray(reference.positions, dtype=np.float64)
    displacement = np.empty((132, 72, 3), dtype=np.float64)
    for index, structure in enumerate(structures):
        cell = np.asarray(structure.cell.array, dtype=np.float64)
        delta = np.asarray(structure.positions, dtype=np.float64) - reference_positions
        fractional = delta @ np.linalg.inv(cell)
        fractional[:, :2] -= np.rint(fractional[:, :2])
        aligned = fractional @ cell
        aligned -= np.mean(aligned, axis=0, keepdims=True)
        displacement[index] = aligned
    if np.max(np.abs(np.mean(displacement, axis=1))) > 2.0e-15:
        raise ValueError("R2AH translation removal failed")
    return displacement, hashes


def principal_vectors(
    flat_displacement: np.ndarray, train: np.ndarray, width: int
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(flat_displacement[train], dtype=np.float64)
    _left, singular, vectors_t = np.linalg.svd(matrix, full_matrices=False)
    vectors = canonicalize_sign(vectors_t[:width].T)
    # Numerical projection keeps the fixed basis exactly translation-free.
    shaped = vectors.reshape(72, 3, width)
    shaped -= np.mean(shaped, axis=0, keepdims=True)
    vectors, triangular = np.linalg.qr(shaped.reshape(216, width))
    vectors = canonicalize_sign(vectors)
    if np.max(np.abs(vectors.T @ vectors - np.eye(width))) > 2.0e-12:
        raise ValueError("R2AH PCA basis lost orthonormality")
    translation_sum = np.sum(vectors.reshape(72, 3, width), axis=0)
    if np.max(np.abs(translation_sum)) > 2.0e-12:
        raise ValueError("R2AH PCA basis is not translation-free")
    return vectors, {
        "train_geometry_count": int(len(train)),
        "PCA_width": int(width),
        "singular_values_first": singular[:width].tolist(),
        "pre_QR_raw_sha256": r2r1.raw_array_sha256(
            shaped.reshape(216, width), "<f8"
        ),
        "post_QR_raw_sha256": r2r1.raw_array_sha256(vectors, "<f8"),
        "QR_abs_diagonal_min": float(np.min(np.abs(np.diag(triangular)))),
        "max_translation_sum": float(np.max(np.abs(translation_sum))),
        "train_geometry_only": True,
    }


def cubic_force_design(
    flat_displacement: np.ndarray,
    q_gradient: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    count, components = flat_displacement.shape
    width = vectors.shape[1]
    if components != 216 or q_gradient.shape != (2, 216):
        raise ValueError("R2AH cubic design input shape changed")
    coordinate = flat_displacement @ vectors
    q_components = flat_displacement @ q_gradient.T
    pairs = [(left, right) for left in range(width) for right in range(left, width)]
    labels = [
        (component, left, right)
        for component in range(2)
        for left, right in pairs
    ]
    design = np.empty((count, components, len(labels)), dtype=np.float64)
    column = 0
    for component in range(2):
        q_value = q_components[:, component]
        q_grad = q_gradient[component]
        for left, right in pairs:
            z_left = coordinate[:, left]
            z_right = coordinate[:, right]
            gradient = (
                (z_left * z_right)[:, None] * q_grad[None, :]
                + (q_value * z_right)[:, None] * vectors[:, left][None, :]
                + (q_value * z_left)[:, None] * vectors[:, right][None, :]
            )
            design[..., column] = -gradient
            column += 1
    if column != len(labels) or not np.all(np.isfinite(design)):
        raise RuntimeError("R2AH cubic design materialization failed")
    return design, labels


def ridge_coefficients(
    projected_design: np.ndarray,
    target: np.ndarray,
    full_design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if projected_design.ndim != 2 or np.iscomplexobj(full_design):
        raise ValueError("R2AH ridge input schema changed")
    normalizer = np.sqrt(np.mean(np.square(full_design), axis=(0, 1)))
    tolerance = np.max(normalizer) * 1.0e-12
    if np.any(normalizer <= tolerance):
        raise ValueError("R2AH cubic force design contains a zero column")
    normalized = projected_design / normalizer[None, :]
    matrix = np.concatenate((normalized.real, normalized.imag), axis=0)
    response = np.concatenate((target.real, target.imag), axis=0)
    matrix /= APRIME_SCALE_EV_A
    response /= APRIME_SCALE_EV_A
    kernel = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (kernel + kernel.T))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_response = eigenvectors.T @ response
    dual = eigenvectors @ (
        projected_response[:, None]
        / (eigenvalues[:, None] + np.asarray(ALPHAS)[None, :])
    )
    normalized_coefficient = matrix.T @ dual
    coefficient = normalized_coefficient / normalizer[:, None]
    return coefficient, normalizer, {
        "real_equation_count": int(matrix.shape[0]),
        "column_count": int(matrix.shape[1]),
        "normalizer_min_relative": float(np.min(normalizer) / np.max(normalizer)),
        "kernel_rank": int(np.linalg.matrix_rank(matrix)),
        "kernel_eigenvalue_range": [float(eigenvalues[0]), float(eigenvalues[-1])],
    }


def fold_baseline(
    data: dict[str, np.ndarray], package: dict[str, np.ndarray], fold: int
) -> np.ndarray:
    coefficient = package["fold_linear_skip_coefficient"][fold]
    return data["fixed"] + np.einsum(
        "natk,k->nat", data["design"], coefficient, optimize=True
    )


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

    package, package_receipt = r2ag.load_package(args.package_root.resolve())
    data, data_receipt = r2ad.load_training(args)
    displacement, geometry_hashes = translation_free_displacements()
    flat = displacement.reshape(132, 216)
    modes = package["Aprime_mode_real"] + 1.0j * package["Aprime_mode_imag"]
    coordinates = (
        package["Aprime_coordinates_real_A"]
        + 1.0j * package["Aprime_coordinates_imag_A"]
    )
    fixed_mode = modes[0]
    mode_real = fixed_mode.real - np.mean(fixed_mode.real, axis=0, keepdims=True)
    mode_imag = fixed_mode.imag - np.mean(fixed_mode.imag, axis=0, keepdims=True)
    q_gradient = np.stack((mode_real.reshape(-1), -mode_imag.reshape(-1)))
    replay = flat[:60] @ q_gradient.T
    replay_complex = replay[:, 0] + 1.0j * replay[:, 1]
    replay_error = float(np.max(np.abs(replay_complex - coordinates)))
    if replay_error > 1.0e-7:
        raise ValueError("R2AH fixed K coordinate does not replay seed012")

    # Verify the fold-specific skips exactly reproduce the frozen baseline OOF.
    baseline_by_fold = []
    replay_max = 0.0
    for fold, hold in enumerate(package["fold_hold_indices"]):
        current = fold_baseline(data, package, fold)
        baseline_by_fold.append(current)
        replay_max = max(
            replay_max,
            float(
                np.max(
                    np.abs(
                        current[np.asarray(hold, dtype=int)]
                        - package["fixed_step32_OOF_predicted_force_eV_A"][hold]
                    )
                )
            ),
        )
    if replay_max > 2.0e-12:
        raise ValueError("R2AH fold baseline replay changed")

    width_records: list[dict[str, Any]] = []
    best_prediction: np.ndarray | None = None
    best_record: dict[str, Any] | None = None
    best_score = np.inf
    all_indices = np.arange(132, dtype=int)
    for width in PCA_WIDTHS:
        candidate_oof = np.repeat(
            package["fixed_step32_OOF_predicted_force_eV_A"][None, None, ...],
            len(ALPHAS),
            axis=0,
        )
        candidate_oof = np.repeat(candidate_oof, len(TAIL_SCALES), axis=1)
        pca_receipts = []
        fold_ridge_receipts = []
        labels_reference: list[tuple[int, int, int]] | None = None
        for fold, hold_values in enumerate(package["fold_hold_indices"]):
            hold = np.asarray(hold_values, dtype=int)
            train = np.setdiff1d(all_indices, hold)
            seed_train = train[train < 60]
            vectors, pca_receipt = principal_vectors(flat, train, width)
            design, labels = cubic_force_design(flat, q_gradient, vectors)
            if labels_reference is None:
                labels_reference = labels
            elif labels != labels_reference:
                raise RuntimeError("R2AH cubic label order changed across folds")
            baseline = baseline_by_fold[fold]
            target_projection = np.einsum(
                "nat,nat->n",
                np.conj(modes[seed_train]),
                data["reference"][seed_train] - baseline[seed_train],
                optimize=True,
            )
            projected_design = np.einsum(
                "nat,natk->nk",
                np.conj(modes[seed_train]),
                design[seed_train].reshape(len(seed_train), 72, 3, -1),
                optimize=True,
            )
            coefficient, _normalizer, ridge_receipt = ridge_coefficients(
                projected_design,
                target_projection,
                design[train],
            )
            tail = np.einsum(
                "nck,ka->nca", design[hold], coefficient, optimize=True
            ).reshape(len(hold), 72, 3, len(ALPHAS))
            for alpha_id in range(len(ALPHAS)):
                for scale_id, scale in enumerate(TAIL_SCALES):
                    candidate_oof[alpha_id, scale_id, hold] += (
                        scale * tail[..., alpha_id]
                    )
            pca_receipts.append({"fold": fold, **pca_receipt})
            fold_ridge_receipts.append({"fold": fold, **ridge_receipt})
            del design, tail, coefficient

        width_best: dict[str, Any] | None = None
        for alpha_id, alpha in enumerate(ALPHAS):
            for scale_id, scale in enumerate(TAIL_SCALES):
                prediction = candidate_oof[alpha_id, scale_id]
                metrics = r2ag.scalar_metrics(prediction, package)
                record = {
                    "PCA_width": width,
                    "cubic_column_count": width * (width + 1),
                    "alpha": alpha,
                    "tail_scale": scale,
                    "full_gate": metrics,
                }
                score = float(metrics["raw_gate_score"])
                if width_best is None or score < width_best["full_gate"]["raw_gate_score"]:
                    width_best = record
                if score < best_score:
                    best_score = score
                    best_record = record
                    best_prediction = prediction.copy()
        if width_best is None:
            raise RuntimeError("R2AH width scan produced no candidate")
        width_records.append(
            {
                "best": width_best,
                "PCA_receipts": pca_receipts,
                "ridge_receipts": fold_ridge_receipts,
            }
        )
        print(
            json.dumps(
                {
                    "PCA_width_complete": width,
                    "best_score": width_best["full_gate"]["raw_gate_score"],
                    "Aprime_by_seed": width_best["full_gate"]["Aprime_by_seed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del candidate_oof

    if best_record is None or best_prediction is None:
        raise RuntimeError("R2AH global scan produced no candidate")
    status = (
        "R2AH_FIXED_CELL_GLOBAL_CUBIC_GATE_PASSED_PRODUCTION_IMPLEMENTATION_PENDING"
        if best_record["full_gate"]["passes_fixed_gate"]
        else "R2AH_FIXED_CELL_GLOBAL_CUBIC_GATE_FAILED"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "best_OOF_predicted_force_eV_A.npy"
    np.save(prediction_path, np.asarray(best_prediction, dtype="<f8"), allow_pickle=False)
    summary = {
        "format": "graphene_r2ah_seed012_fixed_cell_global_cubic_v1",
        "status": status,
        "deployable": False,
        "scope": "seed012 + T300 + T600 development; 525 K unopened",
        "construction": "q_r*z_j*z_k and q_i*z_j*z_k cubic scalar energies; fold-train geometry-only displacement PCA",
        "analytic_reference_jet": {
            "energy_zero": True,
            "force_zero": True,
            "Hessian_zero": True,
            "reason": "every scalar-energy column has total displacement degree three",
        },
        "translation_invariant_fixed_cell": True,
        "fixed_cell_representation_only": True,
        "K_coordinate_replay_max_abs_A": replay_error,
        "fold_linear_skip_replay_max_abs_eV_A": replay_max,
        "PCA_widths": list(PCA_WIDTHS),
        "alphas": list(ALPHAS),
        "tail_scales": list(TAIL_SCALES),
        "width_records": width_records,
        "best": best_record,
        "input_sha256": {
            "training_package": package_receipt["package_sha256"],
            "training_package_receipt": file_sha256(
                args.package_root / "receipt.json"
            ),
            **geometry_hashes,
        },
        "design_input_receipts": data_receipt,
        "best_prediction_sha256": file_sha256(prediction_path),
        "best_prediction_raw_sha256": r2r1.raw_array_sha256(
            best_prediction, "<f8"
        ),
        "energy_labels_used": False,
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_boundary": [
            "This is a fixed-6x6 capacity diagnostic, not a deployable potential.",
            "A pass requires a K-star-symmetric production implementation and an unseen 525 K validation.",
        ],
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(status + "\n")
    print(json.dumps({"status": status, "best": best_record}, indent=2))


if __name__ == "__main__":
    main()
