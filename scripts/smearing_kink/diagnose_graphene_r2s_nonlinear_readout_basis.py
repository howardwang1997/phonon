#!/usr/bin/env python3
"""Screen conservative nonlinear readout bases after the terminal R2R-1 failure.

The frozen R2R energy basis B and force basis X=-dB/dx permit conservative
product columns without another encoder pass.  This script compares two
zero-2-jet expansions on thermal92 conditional OOF:

* C-modulated: C B_k / N, where C is the existing sum_i a_i c_i column;
* diagonal square: B_k^2 / N.

Both remain scalar-energy models and are at least fourth order beyond the
already Taylor-null B columns.  The scan is representation-development
evidence only and never accesses seed1, small-H, support, or held data.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import (
    DEFAULT_AGGREGATE,
    DEFAULT_FIT,
    _parse_whitelist,
    canonical_json_bytes,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/R2S_nonlinear_readout_basis_diagnostic_retry1_20260826"
)
PROJECTION_MASSES = (0.0, 0.50, 0.90)
ATOM_COUNT = 72.0


@dataclass(frozen=True)
class RidgeSystem:
    scale: np.ndarray
    u: np.ndarray
    singular: np.ndarray
    vt: np.ndarray
    weighted_response: np.ndarray

    def coefficient(self, alpha: float) -> np.ndarray:
        normalized = self.vt.T @ (
            self.singular
            / (np.square(self.singular) + float(alpha))
            * (self.u.T @ self.weighted_response)
        )
        return r2r1.FORCE_SCALE_EV_A * normalized / self.scale


def _project(mode: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.vdot(one_mode.reshape(-1), one_value.reshape(-1))
            for one_mode, one_value in zip(mode, values, strict=True)
        ],
        dtype=np.complex128,
    )


def build_basis(
    energy: np.ndarray, force: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    base_energy = np.asarray(energy, dtype=np.float64)
    base_force = np.asarray(force, dtype=np.float64)
    if base_energy.shape != (92, 65) or base_force.shape != (92, 72, 3, 65):
        raise ValueError("frozen R2R design shape changed")
    labels = [f"linear_{index:02d}" for index in range(65)]
    energies = [base_energy]
    forces = [base_force]

    if kind in {"C_modulated", "C_modulated_plus_diagonal"}:
        c_energy = base_energy[:, 32]
        c_force = base_force[..., 32]
        modulated_energy = c_energy[:, None] * base_energy / ATOM_COUNT
        modulated_force = (
            c_energy[:, None, None, None] * base_force
            + base_energy[:, None, None, :] * c_force[..., None]
        ) / ATOM_COUNT
        energies.append(modulated_energy)
        forces.append(modulated_force)
        labels.extend(f"C_times_linear_{index:02d}_per_atom" for index in range(65))

    if kind in {"diagonal_square", "C_modulated_plus_diagonal"}:
        square_energy = np.square(base_energy) / ATOM_COUNT
        square_force = (
            2.0 * base_energy[:, None, None, :] * base_force / ATOM_COUNT
        )
        energies.append(square_energy)
        forces.append(square_force)
        labels.extend(f"linear_square_{index:02d}_per_atom" for index in range(65))

    if kind not in {"linear", "C_modulated", "diagonal_square", "C_modulated_plus_diagonal"}:
        raise ValueError(f"unknown nonlinear basis: {kind}")
    output_energy = np.concatenate(energies, axis=1)
    output_force = np.concatenate(forces, axis=3)
    if output_energy.shape[1] != len(labels) or output_force.shape[3] != len(labels):
        raise AssertionError("nonlinear basis labels and columns disagree")
    return output_energy, output_force, labels


def _ridge_system(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    aprime: r2r1.AprimeData,
    train_indices: Sequence[int],
    projection_mass: float,
) -> RidgeSystem:
    indices = np.asarray(sorted(set(int(value) for value in train_indices)), dtype=int)
    width = design.shape[-1]
    selected = design[indices]
    scale = np.sqrt(np.mean(np.square(selected), axis=(0, 1, 2)))
    relative = scale / np.max(scale)
    if np.any(relative <= 1.0e-12):
        raise ValueError("nonlinear readout contains a zero/near-zero force column")
    normalized_design = selected / scale
    response = (reference[indices] - fixed[indices]) / r2r1.FORCE_SCALE_EV_A
    components = int(np.prod(response.shape[1:]))
    counts = {
        group: sum(r2r1.group_for_global_index(int(index)) == group for index in indices)
        for group in r2r1.THERMAL_GROUP_RANGES
    }
    force_weights: list[float] = []
    for index in indices:
        group = r2r1.group_for_global_index(int(index))
        force_weights.extend(
            [
                (1.0 - projection_mass)
                * r2r1.GROUP_MASSES[group]
                / (counts[group] * components)
            ]
            * components
        )
    matrix_parts = [normalized_design.reshape(-1, width)]
    response_parts = [response.reshape(-1)]
    weight_parts = [np.asarray(force_weights, dtype=np.float64)]

    if projection_mass > 0.0:
        e50 = indices[indices < 20]
        positions = np.searchsorted(indices, e50)
        mode = aprime.mode_real[e50] + 1.0j * aprime.mode_imag[e50]
        projected_design = np.empty((len(e50), width), dtype=np.complex128)
        for row, (one_mode, columns) in enumerate(
            zip(mode, normalized_design[positions], strict=True)
        ):
            projected_design[row] = np.asarray(
                [
                    np.vdot(one_mode.reshape(-1), columns[..., column].reshape(-1))
                    for column in range(width)
                ]
            )
        projected_target = _project(mode, reference[e50] - fixed[e50])
        projected_design *= r2r1.FORCE_SCALE_EV_A / 0.015
        projected_target /= 0.015
        matrix_parts.append(
            np.concatenate((projected_design.real, projected_design.imag), axis=0)
        )
        response_parts.append(
            np.concatenate((projected_target.real, projected_target.imag), axis=0)
        )
        weight_parts.append(
            np.full(2 * len(e50), projection_mass / (2 * len(e50)), np.float64)
        )

    matrix = np.concatenate(matrix_parts, axis=0)
    target = np.concatenate(response_parts)
    weights = np.concatenate(weight_parts)
    if not math.isclose(float(np.sum(weights)), 1.0, abs_tol=3.0e-15, rel_tol=0.0):
        raise RuntimeError("nonlinear objective weights do not sum to one")
    root = np.sqrt(weights)
    u, singular, vt = np.linalg.svd(matrix * root[:, None], full_matrices=False)
    return RidgeSystem(
        scale=scale,
        u=u,
        singular=singular,
        vt=vt,
        weighted_response=target * root,
    )


def _predict(design: np.ndarray, fixed: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return fixed + np.einsum("natk,k->nat", design, coefficient, optimize=False)


def _selection_key(item: dict[str, Any]) -> tuple[float, float]:
    return (float(item["metrics"]["selection_score_rounded_12"]), -float(item["alpha"]))


def nested_oof(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    aprime: r2r1.AprimeData,
    projection_mass: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    all_indices = np.arange(92, dtype=int)
    oof = np.full_like(reference, np.nan)
    records = []
    for outer_id, outer_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
        outer_hold = np.asarray(outer_tuple, dtype=int)
        outer_train = np.setdiff1d(all_indices, outer_hold)
        inner_predictions = {
            alpha: np.full_like(reference, np.nan) for alpha in r2r1.ALPHA_GRID
        }
        for inner_id, inner_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
            if inner_id == outer_id:
                continue
            inner_hold = np.asarray(inner_tuple, dtype=int)
            inner_train = np.setdiff1d(outer_train, inner_hold)
            system = _ridge_system(
                design,
                fixed,
                reference,
                aprime,
                inner_train,
                projection_mass,
            )
            for alpha in r2r1.ALPHA_GRID:
                inner_predictions[alpha][inner_hold] = _predict(
                    design[inner_hold], fixed[inner_hold], system.coefficient(alpha)
                )
        candidates = []
        for alpha in r2r1.ALPHA_GRID:
            metrics = r2r1.gate_metrics(
                inner_predictions[alpha], reference, aprime, outer_train
            )
            candidates.append({"alpha": float(alpha), "metrics": metrics})
        selected = min(candidates, key=_selection_key)
        outer_system = _ridge_system(
            design,
            fixed,
            reference,
            aprime,
            outer_train,
            projection_mass,
        )
        coefficient = outer_system.coefficient(float(selected["alpha"]))
        oof[outer_hold] = _predict(design[outer_hold], fixed[outer_hold], coefficient)
        records.append(
            {
                "outer_fold": outer_id,
                "selected_alpha": selected["alpha"],
                "selected_inner_score": selected["metrics"]["raw_selection_score"],
                "selected_inner_Aprime_RMS_meV_A": selected["metrics"]["E50_Aprime"][
                    "RMS_meV_A"
                ],
                "outer_train_min_weighted_singular_value": float(
                    outer_system.singular[-1]
                ),
                "outer_train_condition": float(
                    outer_system.singular[0] / outer_system.singular[-1]
                ),
            }
        )
    metrics = r2r1.gate_metrics(oof, reference, aprime, all_indices)
    return oof, {"outer_records": records, "pooled_metrics": metrics}


def design_audit(design: np.ndarray) -> dict[str, Any]:
    width = design.shape[-1]
    scale = np.sqrt(np.mean(np.square(design), axis=(0, 1, 2)))
    normalized = (design / scale).reshape(-1, width)
    singular = np.linalg.svd(normalized, full_matrices=False, compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(normalized.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    return {
        "width": width,
        "rank": rank,
        "condition": float(singular[0] / singular[-1]),
        "min_relative_column_RMS": float(np.min(scale) / np.max(scale)),
        "pass": bool(rank == width and np.all(scale > np.max(scale) * 1.0e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thermal", type=Path, default=r2r1.RECOMMENDED_THERMAL92)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--fit-root", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference, aprime, label_hashes = _parse_whitelist(args.thermal.resolve())
    if file_sha256(args.aggregate.resolve()) != r2r1.ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("attempt3 aggregate SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        energy = np.asarray(arrays["thermal_parameter_energy_design_eV"], np.float64)
        force = np.asarray(arrays["thermal_parameter_force_design_eV_A"], np.float64)
        fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], np.float64)

    basis_kinds = (
        "C_modulated",
        "diagonal_square",
        "C_modulated_plus_diagonal",
    )
    results: list[dict[str, Any]] = []
    arrays_to_save: dict[str, np.ndarray] = {}
    for kind in basis_kinds:
        _, design, labels = build_basis(energy, force, kind)
        audit = design_audit(design)
        if not audit["pass"]:
            results.append(
                {
                    "basis": kind,
                    "column_labels": labels,
                    "design_audit": audit,
                    "status": "DESIGN_AUDIT_FAILED",
                }
            )
            continue
        for projection_mass in PROJECTION_MASSES:
            oof, nested = nested_oof(
                design, fixed, reference, aprime, projection_mass
            )
            key = f"{kind}_projection_mass_{projection_mass:.2f}"
            arrays_to_save[key] = oof
            results.append(
                {
                    "basis": kind,
                    "projection_mass": projection_mass,
                    "force_objective_mass": 1.0 - projection_mass,
                    "column_count": len(labels),
                    "design_audit": audit,
                    "nested_OOF": nested,
                    "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
                    "status": (
                        "CONDITIONAL_OOF_GATE_PASSED"
                        if nested["pooled_metrics"]["passes_fixed_gate"]
                        else "CONDITIONAL_OOF_GATE_FAILED"
                    ),
                }
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "nonlinear_basis_oof_arrays.npz", **arrays_to_save)
    summary = {
        "format": "graphene_r2s_nonlinear_readout_basis_diagnostic_v1",
        "status": "REPRESENTATION_DIAGNOSTIC_COMPLETE_NOT_DEPLOYABLE",
        "scope": "thermal92 whitelist and frozen R2R design only",
        "input_sha256": {
            "thermal92": file_sha256(args.thermal),
            "aggregate_arrays": file_sha256(args.aggregate),
            "terminal_R2R1_fit_receipt": file_sha256(args.fit_root / "fit_receipt.json"),
        },
        "label_raw_sha256": label_hashes,
        "basis_definitions": {
            "C_modulated": "[B, C*B/N], C=B[:,32], conservative product rule",
            "diagonal_square": "[B, B^2/N], conservative product rule",
            "C_modulated_plus_diagonal": "[B, C*B/N, B^2/N]",
            "atom_count_N": int(ATOM_COUNT),
            "reference_semantics": (
                "all added products inherit exact zero value/Jacobian/Hessian because "
                "each frozen B column has an exact zero 2-jet"
            ),
        },
        "projection_masses": list(PROJECTION_MASSES),
        "results": results,
        "interpretation_boundary": [
            "This scan reuses thermal92 and is conditional representation development.",
            "Passing here does not authorize final fit, mechanics, development data, or SSCHA.",
            "A deployable candidate needs a separately frozen basis and independent gates.",
        ],
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    compact = [
        {
            "basis": item["basis"],
            "projection_mass": item.get("projection_mass"),
            "status": item["status"],
            "design_audit": item["design_audit"],
            "pooled_metrics": item.get("nested_OOF", {}).get("pooled_metrics"),
        }
        for item in results
    ]
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
