#!/usr/bin/env python3
"""Diagnose a conservative nonlocal K-star/bond-triad correction.

For each translation orbit of a reference bond, form the complex folded-K
quadratic field

    B_d(K) = Ncell**(-1/2) sum_R exp(-i K.R) |u_(R+d,t)-u_(R,s)|**2.

The real and imaginary parts of ``conj(q_K) B_d(K)`` are cubic scalar-energy
columns.  They are invariant to a uniform displacement and have an exact zero
value, force, and Hessian at the reference.  This fixed-6x6 script is only a
capacity/range diagnostic; a passing representation still requires explicit
point-group tying and a size-general production implementation.

All seed0/1/2 and T300/T600 structures are development data.  The planned
525 K off-policy trajectory remains unopened.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
DEFAULT_OUTPUT = BASE / "R2AK_seed012_kstar_bond_triad_diagnostic_20260827"
GRID = 6
NCELL = GRID * GRID
K_REDUCED = np.asarray((1.0 / 3.0, 1.0 / 3.0), dtype=np.float64)
FORCE_SCALE_EV_A = 0.030
APRIME_SCALE_EV_A = 0.015
PROJECTION_MASS = 0.85
SEED_WEIGHTS = {name: 1.0 / 3.0 for name in r2ad.SEED_RANGES}
GROUP_MASSES = {
    "E50_seed0": 0.075,
    "E50_seed1": 0.075,
    "E50_seed2": 0.075,
    "T300": 0.075,
    "T600": 0.70,
}
RANGE_CUTOFFS_A = (3.0, 5.0, 7.5, float("inf"))
ALPHAS = tuple(float(value) for value in np.logspace(-6.0, 3.0, 31))


@dataclass(frozen=True)
class BondOrbit:
    source_sublattice: int
    target_sublattice: int
    shift: tuple[int, int]
    reference_distance_A: float
    self_inverse_weight: float


def _cell_atom(sublattice: int, x: int, y: int) -> int:
    return sublattice * NCELL + (y % GRID) * GRID + (x % GRID)


def _canonical_same_sublattice_shifts() -> list[tuple[int, int]]:
    shifts = []
    for dx in range(GRID):
        for dy in range(GRID):
            if dx == 0 and dy == 0:
                continue
            shift = (dx, dy)
            inverse = ((-dx) % GRID, (-dy) % GRID)
            if shift <= inverse:
                shifts.append(shift)
    if len(shifts) != 19:
        raise RuntimeError("R2AK same-sublattice orbit count changed")
    return shifts


def _minimum_image_distance(
    shift: tuple[int, int], source: int, target: int, primitive_cell: np.ndarray
) -> float:
    basis = np.asarray(((0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0)))
    raw = np.asarray(shift, dtype=np.float64) + basis[target] - basis[source]
    candidates = []
    for first in (-1, 0, 1):
        for second in (-1, 0, 1):
            reduced = raw + GRID * np.asarray((first, second), dtype=np.float64)
            candidates.append(float(np.linalg.norm(reduced @ primitive_cell)))
    return min(candidates)


def bond_orbits(supercell: np.ndarray) -> list[BondOrbit]:
    primitive = np.asarray(supercell[:2, :], dtype=np.float64) / GRID
    output = []
    same_shifts = _canonical_same_sublattice_shifts()
    for sublattice in (0, 1):
        for shift in same_shifts:
            inverse = ((-shift[0]) % GRID, (-shift[1]) % GRID)
            output.append(
                BondOrbit(
                    sublattice,
                    sublattice,
                    shift,
                    _minimum_image_distance(
                        shift, sublattice, sublattice, primitive
                    ),
                    0.5 if shift == inverse else 1.0,
                )
            )
    for dx in range(GRID):
        for dy in range(GRID):
            output.append(
                BondOrbit(
                    0,
                    1,
                    (dx, dy),
                    _minimum_image_distance((dx, dy), 0, 1, primitive),
                    1.0,
                )
            )
    if len(output) != 74:
        raise RuntimeError("R2AK bond-orbit count changed")
    return output


def kstar_bond_force_design(
    displacement: np.ndarray,
    q_gradient: np.ndarray,
    orbits: list[BondOrbit],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    count = displacement.shape[0]
    if displacement.shape != (count, 72, 3) or q_gradient.shape != (216,):
        raise ValueError("R2AK displacement/mode shape changed")
    flat = displacement.reshape(count, 216)
    q_value = flat @ q_gradient
    phases = np.empty((GRID, GRID), dtype=np.complex128)
    for x in range(GRID):
        for y in range(GRID):
            phases[x, y] = np.exp(
                -2.0j * np.pi * np.dot(K_REDUCED, (x, y))
            ) / np.sqrt(NCELL)

    design = np.empty((count, 216, 2 * len(orbits)), dtype=np.float64)
    labels: list[dict[str, Any]] = []
    for orbit_index, orbit in enumerate(orbits):
        field = np.zeros(count, dtype=np.complex128)
        field_gradient = np.zeros((count, 72, 3), dtype=np.complex128)
        dx, dy = orbit.shift
        for x in range(GRID):
            for y in range(GRID):
                source = _cell_atom(orbit.source_sublattice, x, y)
                target = _cell_atom(orbit.target_sublattice, x + dx, y + dy)
                difference = displacement[:, target] - displacement[:, source]
                phase = phases[x, y] * orbit.self_inverse_weight
                field += phase * np.sum(np.square(difference), axis=1)
                gradient = 2.0 * phase * difference
                field_gradient[:, source] -= gradient
                field_gradient[:, target] += gradient
        complex_gradient = (
            np.conj(q_gradient)[None, :, None] * field[:, None, None]
            + np.conj(q_value)[:, None, None]
            * field_gradient.reshape(count, 216, 1)
        )
        # The singleton final dimension above keeps the expression explicit;
        # remove it before storing the two real scalar-energy gradients.
        complex_gradient = complex_gradient[..., 0]
        design[..., 2 * orbit_index] = -complex_gradient.real
        design[..., 2 * orbit_index + 1] = -complex_gradient.imag
        common = {
            "source_sublattice": orbit.source_sublattice,
            "target_sublattice": orbit.target_sublattice,
            "shift": list(orbit.shift),
            "reference_distance_A": orbit.reference_distance_A,
        }
        labels.extend(
            [
                {**common, "component": "Re_conj_q_B"},
                {**common, "component": "Im_conj_q_B"},
            ]
        )
    net_force = np.sum(design.reshape(count, 72, 3, -1), axis=1)
    audit = {
        "uniform_translation_force_sum_max_abs": float(np.max(np.abs(net_force))),
        "finite_max_abs_force_column": float(np.max(np.abs(design))),
    }
    if audit["uniform_translation_force_sum_max_abs"] > 2.0e-12:
        raise ValueError("R2AK bond-triad force is not translation invariant")
    if not np.all(np.isfinite(design)):
        raise ValueError("R2AK bond-triad design is non-finite")
    return design, labels, audit


def _members(train: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
    return np.intersect1d(train, np.arange(bounds[0], bounds[1], dtype=int))


def ridge_path(
    design: np.ndarray,
    baseline: np.ndarray,
    reference: np.ndarray,
    modes: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalizer = np.sqrt(np.mean(np.square(design[train]), axis=(0, 1)))
    tolerance = float(np.max(normalizer) * 1.0e-12)
    active = normalizer > tolerance
    if np.count_nonzero(active) < 2:
        raise ValueError("R2AK bond-triad design has insufficient active columns")
    normalized = design[..., active] / normalizer[active][None, None, :]
    width = normalized.shape[-1]
    gram = np.zeros((width, width), dtype=np.float64)
    rhs = np.zeros(width, dtype=np.float64)
    residual = reference - baseline

    force_mass = 1.0 - PROJECTION_MASS
    for name, bounds in r2ad.GROUP_RANGES.items():
        selected = _members(train, bounds)
        matrix = normalized[selected].reshape(-1, width) / FORCE_SCALE_EV_A
        target = residual[selected].reshape(-1) / FORCE_SCALE_EV_A
        weight = force_mass * GROUP_MASSES[name] / matrix.shape[0]
        gram += weight * (matrix.T @ matrix)
        rhs += weight * (matrix.T @ target)

    for name, bounds in r2ad.SEED_RANGES.items():
        selected = _members(train, bounds)
        matrix = np.einsum(
            "nat,natk->nk",
            np.conj(modes[selected]),
            normalized[selected].reshape(len(selected), 72, 3, width),
            optimize=True,
        ) / APRIME_SCALE_EV_A
        target = np.einsum(
            "nat,nat->n",
            np.conj(modes[selected]),
            residual[selected],
            optimize=True,
        ) / APRIME_SCALE_EV_A
        weight = PROJECTION_MASS * SEED_WEIGHTS[name] / (2.0 * len(selected))
        gram += weight * (
            matrix.real.T @ matrix.real + matrix.imag.T @ matrix.imag
        )
        rhs += weight * (
            matrix.real.T @ target.real + matrix.imag.T @ target.imag
        )

    gram = 0.5 * (gram + gram.T)
    ridge_scale = float(np.trace(gram) / width)
    values, vectors = np.linalg.eigh(gram)
    values = np.maximum(values, 0.0)
    projected_rhs = vectors.T @ rhs
    normalized_coefficients = vectors @ (
        projected_rhs[:, None]
        / (values[:, None] + ridge_scale * np.asarray(ALPHAS)[None, :])
    )
    coefficients = np.zeros((design.shape[-1], len(ALPHAS)), dtype=np.float64)
    coefficients[active] = normalized_coefficients / normalizer[active, None]
    return coefficients, {
        "active_columns": int(np.count_nonzero(active)),
        "total_columns": int(len(active)),
        "normalizer_min_relative": float(
            np.min(normalizer[active]) / np.max(normalizer[active])
        ),
        "dimensionless_ridge_scale": ridge_scale,
        "Gram_eigenvalue_range": [float(values[0]), float(values[-1])],
    }


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
        raise ValueError("R2AK fixed K coordinate does not replay seed012")

    structures, reference_atoms, _hashes = r2ag.structures_and_reference()
    del structures
    orbits = bond_orbits(np.asarray(reference_atoms.cell.array, dtype=np.float64))
    full_design, labels, design_audit = kstar_bond_force_design(
        displacement, q_gradient, orbits
    )

    all_indices = np.arange(132, dtype=int)
    range_records = []
    best_record: dict[str, Any] | None = None
    best_prediction: np.ndarray | None = None
    for cutoff in RANGE_CUTOFFS_A:
        selected_columns = np.asarray(
            [label["reference_distance_A"] <= cutoff for label in labels], dtype=bool
        )
        design = full_design[..., selected_columns]
        if design.shape[-1] < 2:
            continue
        candidate_oof = np.repeat(
            package["fixed_step32_OOF_predicted_force_eV_A"][..., None],
            len(ALPHAS),
            axis=-1,
        )
        fold_receipts = []
        for fold, hold_values in enumerate(package["fold_hold_indices"]):
            hold = np.asarray(hold_values, dtype=int)
            train = np.setdiff1d(all_indices, hold)
            baseline = fold_baseline(data, package, fold)
            coefficients, receipt = ridge_path(
                design, baseline, data["reference"], modes, train
            )
            candidate_oof[hold] += np.einsum(
                "nck,ka->nca", design[hold], coefficients, optimize=True
            ).reshape(len(hold), 72, 3, len(ALPHAS))
            fold_receipts.append({"fold": fold, **receipt})

        cutoff_best: dict[str, Any] | None = None
        for alpha_index, alpha in enumerate(ALPHAS):
            prediction = candidate_oof[..., alpha_index]
            metrics = r2ag.scalar_metrics(prediction, package)
            record = {
                "cutoff_A": None if np.isinf(cutoff) else cutoff,
                "bond_orbit_count": int(design.shape[-1] // 2),
                "force_column_count": int(design.shape[-1]),
                "alpha": alpha,
                "full_gate": metrics,
            }
            if (
                cutoff_best is None
                or metrics["raw_gate_score"]
                < cutoff_best["full_gate"]["raw_gate_score"]
            ):
                cutoff_best = record
            if (
                best_record is None
                or metrics["raw_gate_score"]
                < best_record["full_gate"]["raw_gate_score"]
            ):
                best_record = record
                best_prediction = prediction.copy()
        if cutoff_best is None:
            raise RuntimeError("R2AK cutoff scan produced no candidate")
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
        raise RuntimeError("R2AK range scan produced no candidate")
    status = (
        "R2AK_KSTAR_BOND_TRIAD_GATE_PASSED_PRODUCTION_IMPLEMENTATION_PENDING"
        if best_record["full_gate"]["passes_fixed_gate"]
        else "R2AK_KSTAR_BOND_TRIAD_GATE_FAILED"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "best_OOF_predicted_force_eV_A.npy"
    np.save(prediction_path, np.asarray(best_prediction, dtype="<f8"), allow_pickle=False)
    summary = {
        "format": "graphene_r2ak_seed012_kstar_bond_triad_v1",
        "status": status,
        "deployable": False,
        "scope": "seed012 + T300 + T600 development; 525 K unopened",
        "construction": (
            "Re/Im[conj(q_K) B_d(K)] cubic energy columns, where B_d is the "
            "folded-K transform of squared translation-invariant bond displacement"
        ),
        "analytic_reference_jet": {
            "energy_zero": True,
            "force_zero": True,
            "Hessian_zero": True,
            "reason": "q_K is degree one and every B_d is degree two",
        },
        "fixed_cell_representation_only": True,
        "point_group_tying_complete": False,
        "K_coordinate_replay_max_abs_A": replay_error,
        "design_audit": design_audit,
        "objective": {
            "projection_mass": PROJECTION_MASS,
            "seed_weights": SEED_WEIGHTS,
            "group_masses_within_force_loss": GROUP_MASSES,
        },
        "range_cutoffs_A": [None if np.isinf(x) else x for x in RANGE_CUTOFFS_A],
        "alphas": list(ALPHAS),
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
            "This is a fixed-6x6 capacity/range diagnostic, not a deployable potential.",
            "A pass requires point-group tying, size-general mechanics, and unseen 525 K validation.",
        ],
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(status + "\n")
    print(json.dumps({"status": status, "best": best_record}, indent=2))


if __name__ == "__main__":
    main()
