#!/usr/bin/env python3
"""Test the full bond-frame tensor extension of the R2AK K-star triad.

R2AK used only ``|du|^2`` for each translation orbit.  Here the quadratic
bond field is resolved into the six symmetric products of longitudinal,
in-plane-transverse, and out-of-plane relative displacement.  Contracting
each folded-K field with the frozen A-prime coordinate gives conservative
cubic energy columns with an exact zero two-jet at the reference.

This remains a fixed-6x6 development diagnostic.  It never opens the planned
525 K off-policy set and cannot be deployed without point-group tying.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import diagnose_graphene_r2ak_kstar_bond_triad as r2ak
import fit_graphene_r2ad_seed012_step32 as r2ad
import graphene_r2r1_linear_readout as r2r1
import train_graphene_r2ag_seed012_conditional_mlp as r2ag
from diagnose_graphene_r2ah_seed012_global_cubic import (
    fold_baseline,
    translation_free_displacements,
)
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
DEFAULT_OUTPUT = BASE / "R2AL_seed012_kstar_bond_tensor_diagnostic_20260827"
CUTOFFS_A = (7.5, float("inf"))
FRAME_PRODUCTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
FRAME_NAMES = ("LL", "TT", "ZZ", "LT", "LZ", "TZ")


def minimum_image_vector(
    orbit: r2ak.BondOrbit, primitive_cell: np.ndarray
) -> np.ndarray:
    basis = np.asarray(((0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0)))
    raw = (
        np.asarray(orbit.shift, dtype=np.float64)
        + basis[orbit.target_sublattice]
        - basis[orbit.source_sublattice]
    )
    candidates = []
    for first in (-1, 0, 1):
        for second in (-1, 0, 1):
            reduced = raw + r2ak.GRID * np.asarray((first, second), dtype=np.float64)
            vector = reduced @ primitive_cell
            candidates.append((float(np.linalg.norm(vector)), vector))
    distance, vector = min(candidates, key=lambda item: item[0])
    if abs(distance - orbit.reference_distance_A) > 2.0e-10:
        raise ValueError("R2AL orbit distance/vector mismatch")
    return np.asarray(vector, dtype=np.float64)


def orbit_frames(
    orbits: list[r2ak.BondOrbit], supercell: np.ndarray
) -> np.ndarray:
    primitive = np.asarray(supercell[:2], dtype=np.float64) / r2ak.GRID
    normal = np.cross(primitive[0], primitive[1])
    normal /= np.linalg.norm(normal)
    frames = []
    for orbit in orbits:
        longitudinal = minimum_image_vector(orbit, primitive)
        longitudinal /= np.linalg.norm(longitudinal)
        transverse = np.cross(normal, longitudinal)
        transverse /= np.linalg.norm(transverse)
        frame = np.stack((longitudinal, transverse, normal))
        if np.max(np.abs(frame @ frame.T - np.eye(3))) > 2.0e-12:
            raise ValueError("R2AL bond frame is not orthonormal")
        frames.append(frame)
    return np.asarray(frames)


def tensor_force_design(
    displacement: np.ndarray,
    q_gradient: np.ndarray,
    orbits: list[r2ak.BondOrbit],
    frames: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    count = len(displacement)
    flat = displacement.reshape(count, 216)
    q_value = flat @ q_gradient
    phases = np.empty((r2ak.GRID, r2ak.GRID), dtype=np.complex128)
    for x in range(r2ak.GRID):
        for y in range(r2ak.GRID):
            phases[x, y] = np.exp(
                -2.0j * np.pi * np.dot(r2ak.K_REDUCED, (x, y))
            ) / np.sqrt(r2ak.NCELL)

    column_count = 2 * len(orbits) * len(FRAME_PRODUCTS)
    design = np.empty((count, 216, column_count), dtype=np.float64)
    labels = []
    column = 0
    for orbit, frame in zip(orbits, frames, strict=True):
        dx, dy = orbit.shift
        for product_name, (left, right) in zip(
            FRAME_NAMES, FRAME_PRODUCTS, strict=True
        ):
            field = np.zeros(count, dtype=np.complex128)
            field_gradient = np.zeros((count, 72, 3), dtype=np.complex128)
            for x in range(r2ak.GRID):
                for y in range(r2ak.GRID):
                    source = r2ak._cell_atom(orbit.source_sublattice, x, y)
                    target = r2ak._cell_atom(
                        orbit.target_sublattice, x + dx, y + dy
                    )
                    difference = displacement[:, target] - displacement[:, source]
                    left_value = difference @ frame[left]
                    right_value = difference @ frame[right]
                    phase = phases[x, y] * orbit.self_inverse_weight
                    field += phase * left_value * right_value
                    derivative = (
                        right_value[:, None] * frame[left][None, :]
                        + left_value[:, None] * frame[right][None, :]
                    )
                    derivative = phase * derivative
                    field_gradient[:, source] -= derivative
                    field_gradient[:, target] += derivative
            complex_gradient = (
                np.conj(q_gradient)[None, :] * field[:, None]
                + np.conj(q_value)[:, None]
                * field_gradient.reshape(count, 216)
            )
            design[..., column] = -complex_gradient.real
            design[..., column + 1] = -complex_gradient.imag
            common = {
                "source_sublattice": orbit.source_sublattice,
                "target_sublattice": orbit.target_sublattice,
                "shift": list(orbit.shift),
                "reference_distance_A": orbit.reference_distance_A,
                "bond_frame_product": product_name,
            }
            labels.extend(
                [
                    {**common, "component": "Re_conj_q_B"},
                    {**common, "component": "Im_conj_q_B"},
                ]
            )
            column += 2
    if column != column_count or not np.all(np.isfinite(design)):
        raise RuntimeError("R2AL tensor design materialization failed")
    net_force = np.sum(design.reshape(count, 72, 3, -1), axis=1)
    audit = {
        "uniform_translation_force_sum_max_abs": float(np.max(np.abs(net_force))),
        "finite_max_abs_force_column": float(np.max(np.abs(design))),
    }
    if audit["uniform_translation_force_sum_max_abs"] > 3.0e-12:
        raise ValueError("R2AL bond-tensor force is not translation invariant")
    return design, labels, audit


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
    modes = package["Aprime_mode_real"] + 1.0j * package["Aprime_mode_imag"]
    coordinates = (
        package["Aprime_coordinates_real_A"]
        + 1.0j * package["Aprime_coordinates_imag_A"]
    )
    fixed_mode = modes[0]
    q_gradient = (
        fixed_mode.real
        - np.mean(fixed_mode.real, axis=0, keepdims=True)
        - 1.0j
        * (fixed_mode.imag - np.mean(fixed_mode.imag, axis=0, keepdims=True))
    ).reshape(-1)
    replay = displacement[:60].reshape(60, 216) @ q_gradient
    replay_error = float(np.max(np.abs(replay - coordinates)))
    if replay_error > 1.0e-7:
        raise ValueError("R2AL fixed K coordinate does not replay seed012")

    structures, reference_atoms, _hashes = r2ag.structures_and_reference()
    del structures
    supercell = np.asarray(reference_atoms.cell.array, dtype=np.float64)
    orbits = r2ak.bond_orbits(supercell)
    frames = orbit_frames(orbits, supercell)
    full_design, labels, design_audit = tensor_force_design(
        displacement, q_gradient, orbits, frames
    )

    all_indices = np.arange(132, dtype=int)
    range_records = []
    best_record: dict[str, Any] | None = None
    best_prediction: np.ndarray | None = None
    for cutoff in CUTOFFS_A:
        selected_columns = np.asarray(
            [label["reference_distance_A"] <= cutoff for label in labels], dtype=bool
        )
        design = full_design[..., selected_columns]
        candidate_oof = np.repeat(
            package["fixed_step32_OOF_predicted_force_eV_A"][..., None],
            len(r2ak.ALPHAS),
            axis=-1,
        )
        fold_receipts = []
        for fold, hold_values in enumerate(package["fold_hold_indices"]):
            hold = np.asarray(hold_values, dtype=int)
            train = np.setdiff1d(all_indices, hold)
            baseline = fold_baseline(data, package, fold)
            coefficients, receipt = r2ak.ridge_path(
                design, baseline, data["reference"], modes, train
            )
            candidate_oof[hold] += np.einsum(
                "nck,ka->nca", design[hold], coefficients, optimize=True
            ).reshape(len(hold), 72, 3, len(r2ak.ALPHAS))
            fold_receipts.append({"fold": fold, **receipt})

        cutoff_best = None
        for alpha_index, alpha in enumerate(r2ak.ALPHAS):
            prediction = candidate_oof[..., alpha_index]
            metrics = r2ag.scalar_metrics(prediction, package)
            record = {
                "cutoff_A": None if np.isinf(cutoff) else cutoff,
                "bond_orbit_count": int(len(orbits)),
                "frame_product_count": len(FRAME_PRODUCTS),
                "force_column_count": int(design.shape[-1]),
                "alpha": alpha,
                "full_gate": metrics,
            }
            if cutoff_best is None or (
                metrics["raw_gate_score"]
                < cutoff_best["full_gate"]["raw_gate_score"]
            ):
                cutoff_best = record
            if best_record is None or (
                metrics["raw_gate_score"]
                < best_record["full_gate"]["raw_gate_score"]
            ):
                best_record = record
                best_prediction = prediction.copy()
        if cutoff_best is None:
            raise RuntimeError("R2AL cutoff scan produced no candidate")
        range_records.append(
            {"best": cutoff_best, "fold_ridge_receipts": fold_receipts}
        )
        print(
            json.dumps(
                {
                    "cutoff_complete_A": cutoff_best["cutoff_A"],
                    "column_count": cutoff_best["force_column_count"],
                    "best_score": cutoff_best["full_gate"]["raw_gate_score"],
                    "Aprime_by_seed": cutoff_best["full_gate"]["Aprime_by_seed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if best_record is None or best_prediction is None:
        raise RuntimeError("R2AL scan produced no candidate")
    status = (
        "R2AL_KSTAR_BOND_TENSOR_GATE_PASSED_PRODUCTION_IMPLEMENTATION_PENDING"
        if best_record["full_gate"]["passes_fixed_gate"]
        else "R2AL_KSTAR_BOND_TENSOR_GATE_FAILED"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "best_OOF_predicted_force_eV_A.npy"
    np.save(prediction_path, np.asarray(best_prediction, dtype="<f8"), allow_pickle=False)
    summary = {
        "format": "graphene_r2al_seed012_kstar_bond_tensor_v1",
        "status": status,
        "deployable": False,
        "scope": "seed012 + T300 + T600 development; 525 K unopened",
        "construction": (
            "six bond-frame tensor components of B_d(K), coupled to q_K as "
            "real cubic scalar energies"
        ),
        "frame_products": list(FRAME_NAMES),
        "analytic_reference_jet": {
            "energy_zero": True,
            "force_zero": True,
            "Hessian_zero": True,
        },
        "fixed_cell_representation_only": True,
        "point_group_tying_complete": False,
        "K_coordinate_replay_max_abs_A": replay_error,
        "design_audit": design_audit,
        "objective": {
            "projection_mass": r2ak.PROJECTION_MASS,
            "seed_weights": r2ak.SEED_WEIGHTS,
            "group_masses_within_force_loss": r2ak.GROUP_MASSES,
        },
        "cutoffs_A": [None if np.isinf(x) else x for x in CUTOFFS_A],
        "alphas": list(r2ak.ALPHAS),
        "range_records": range_records,
        "best": best_record,
        "input_sha256": {
            "training_package": package_receipt["package_sha256"],
            "training_package_receipt": file_sha256(args.package_root / "receipt.json"),
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
            "A pass requires point-group tying, size-general mechanics, and unseen 525 K validation.",
        ],
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(status + "\n")
    print(json.dumps({"status": status, "best": best_record}, indent=2))


if __name__ == "__main__":
    main()
