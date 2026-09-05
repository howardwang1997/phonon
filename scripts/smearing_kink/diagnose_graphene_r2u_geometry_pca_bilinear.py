#!/usr/bin/env python3
"""Screen geometry-only PCA reductions of the R2T off-diagonal basis.

The 97-column R2S core is retained exactly.  The remaining 480 bilinear force
columns are standardized and residualized against that core using all92
geometry only.  Principal directions of the residual Gram matrix are then
frozen before force labels enter the nested OOF fits.
"""

from __future__ import annotations

import argparse
import json
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
    nested_mass_alpha,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_DESIGN_ROOT = BASE / "R2T_full_bilinear_materialization_20260826"
DEFAULT_OUTPUT = BASE / "R2U_geometry_PCA_bilinear_diagnostic_20260826"
PCA_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
DENSE_ALPHA_GRID = tuple(10.0 ** (exponent / 2.0) for exponent in range(-20, 5))


def canonicalize_eigenvector_sign(vectors: np.ndarray) -> np.ndarray:
    output = np.array(vectors, dtype=np.float64, copy=True)
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0.0:
            output[:, column] *= -1.0
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--design-root", type=Path, default=DEFAULT_DESIGN_ROOT)
    parser.add_argument("--geometry-scope", choices=("all92", "E50"), default="all92")
    parser.add_argument("--pca-width", type=int, action="append")
    parser.add_argument("--projection-mass", type=float, action="append")
    parser.add_argument("--dense-alpha-grid", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pca_widths = tuple(args.pca_width) if args.pca_width else PCA_WIDTHS
    if not pca_widths or any(width <= 0 or width > 64 for width in pca_widths):
        raise ValueError("PCA widths must be unique integers in 1..64")
    if len(set(pca_widths)) != len(pca_widths):
        raise ValueError("PCA widths must be unique")
    projection_masses = (
        tuple(args.projection_mass) if args.projection_mass else (0.50,)
    )
    if any(not 0.0 <= mass < 1.0 for mass in projection_masses):
        raise ValueError("projection masses must be in [0, 1)")
    alpha_grid = DENSE_ALPHA_GRID if args.dense_alpha_grid else r2r1.ALPHA_GRID

    reference, aprime, label_hashes = _parse_whitelist(r2r1.RECOMMENDED_THERMAL92)
    receipt_path = args.design_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    bilinear_energy_path = args.design_root / "bilinear_energy_design_eV.npy"
    bilinear_force_path = args.design_root / "bilinear_force_design_eV_A.npy"
    bilinear_energy = np.load(bilinear_energy_path, allow_pickle=False)
    bilinear_force = np.load(bilinear_force_path, allow_pickle=False)
    if r2r1.raw_array_sha256(bilinear_energy, "<f8") != receipt[
        "array_raw_sha256"
    ]["energy"]:
        raise ValueError("R2T bilinear energy differs from its receipt")
    if r2r1.raw_array_sha256(bilinear_force, "<f8") != receipt[
        "array_raw_sha256"
    ]["force"]:
        raise ValueError("R2T bilinear force differs from its receipt")
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

    diagonal = bilinear_indices("cross32")
    diagonal_set = set(int(value) for value in diagonal)
    offdiagonal = np.asarray(
        [value for value in range(512) if value not in diagonal_set], dtype=int
    )
    core_energy = np.concatenate(
        (base_energy, np.take(bilinear_energy, diagonal, axis=-1)), axis=1
    )
    core_force = np.concatenate(
        (base_force, np.take(bilinear_force, diagonal, axis=-1)), axis=3
    )
    off_energy = np.take(bilinear_energy, offdiagonal, axis=-1)
    off_force = np.take(bilinear_force, offdiagonal, axis=-1)
    core_scale = np.sqrt(np.mean(np.square(core_force), axis=(0, 1, 2)))
    off_scale = np.sqrt(np.mean(np.square(off_force), axis=(0, 1, 2)))
    geometry_indices = (
        np.arange(92, dtype=int)
        if args.geometry_scope == "all92"
        else np.arange(20, dtype=int)
    )
    component_count = len(geometry_indices) * 72 * 3
    core_matrix = (core_force[geometry_indices] / core_scale).reshape(
        component_count, -1
    )
    off_matrix = (off_force[geometry_indices] / off_scale).reshape(
        component_count, -1
    )
    core_gram = core_matrix.T @ core_matrix
    core_off_gram = core_matrix.T @ off_matrix
    off_gram = off_matrix.T @ off_matrix
    core_projection = np.linalg.solve(core_gram, core_off_gram)
    residual_gram = off_gram - core_off_gram.T @ core_projection
    residual_gram = 0.5 * (residual_gram + residual_gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(residual_gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = canonicalize_eigenvector_sign(eigenvectors[:, order])
    negative_tolerance = (
        1.0e-9 * max(float(eigenvalues[0]), np.finfo(np.float64).tiny)
    )
    if float(eigenvalues[-1]) < -negative_tolerance:
        raise ValueError("geometry-only residual Gram matrix is not PSD")

    maximum_width = max(pca_widths)
    raw_projection = eigenvectors[:, :maximum_width] / off_scale[:, None]
    projected_energy = off_energy @ raw_projection
    projected_force = np.einsum(
        "natk,kp->natp", off_force, raw_projection, optimize=True
    )
    results = []
    arrays_to_save: dict[str, np.ndarray] = {
        "offdiagonal_bilinear_indices": offdiagonal,
        "offdiagonal_force_RMS_scale_eV_A": off_scale,
        "residual_PCA_eigenvalues": eigenvalues,
        "raw_offdiagonal_projection_first64": raw_projection,
        "projected_energy_first64_eV": projected_energy,
        "projected_force_first64_eV_A": projected_force,
    }
    for width in pca_widths:
        energy = np.concatenate((core_energy, projected_energy[:, :width]), axis=1)
        design = np.concatenate((core_force, projected_force[..., :width]), axis=3)
        audit = design_audit(design)
        if not audit["pass"]:
            results.append(
                {"PCA_width": width, "status": "DESIGN_AUDIT_FAILED", "design_audit": audit}
            )
            continue
        statistics = fold_statistics(design, fixed, reference, aprime)
        penalty = np.ones(design.shape[-1], dtype=np.float64)
        oof, nested = nested_mass_alpha(
            design,
            fixed,
            reference,
            aprime,
            statistics,
            projection_masses,
            ((1.0, penalty),),
            alpha_grid,
        )
        arrays_to_save[f"OOF_PCA_width_{width}"] = oof
        passed = bool(nested["pooled_OOF_metrics"]["passes_fixed_gate"])
        results.append(
            {
                "PCA_width": width,
                "total_column_count": int(design.shape[-1]),
                "status": (
                    "CONDITIONAL_NESTED_OOF_PASSED" if passed else "CONDITIONAL_NESTED_OOF_FAILED"
                ),
                "design_audit": audit,
                "nested_OOF": nested,
                "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
                "energy_design_raw_sha256": r2r1.raw_array_sha256(energy, "<f8"),
            }
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "geometry_PCA_basis_and_OOF.npz"
    np.savez(arrays_path, **arrays_to_save)
    any_passed = any(
        item["status"] == "CONDITIONAL_NESTED_OOF_PASSED" for item in results
    )
    summary = {
        "format": "graphene_r2u_geometry_only_residual_PCA_bilinear_diagnostic_v1",
        "status": (
            "GEOMETRY_PCA_REPRESENTATION_FOUND_DEVELOPMENT_ONLY"
            if any_passed
            else "GEOMETRY_PCA_REPRESENTATION_DIAGNOSTIC_FAILED"
        ),
        "deployable": False,
        "geometry_only_representation_construction": True,
        "geometry_scope": args.geometry_scope,
        "geometry_global_indices": geometry_indices.tolist(),
        "PCA_widths": list(pca_widths),
        "core_column_count": 97,
        "offdiagonal_candidate_count": 480,
        "projection_mass_grid": list(projection_masses),
        "alpha_grid": list(alpha_grid),
        "residual_eigenvalue_first10": eigenvalues[:10].tolist(),
        "residual_eigenvalue_min": float(eigenvalues[-1]),
        "results": results,
        "label_raw_sha256": label_hashes,
        "energy_labels_used": False,
        "development_or_held_access": False,
        "input_sha256": {
            "thermal92": file_sha256(r2r1.RECOMMENDED_THERMAL92),
            "aggregate_arrays": file_sha256(args.aggregate),
            "bilinear_design_receipt": file_sha256(receipt_path),
            "bilinear_energy_design": file_sha256(bilinear_energy_path),
            "bilinear_force_design": file_sha256(bilinear_force_path),
        },
        "output_arrays_sha256": file_sha256(arrays_path),
        "interpretation": (
            "PCA directions use geometry only, but width screening uses thermal92 labels; "
            "a passing width still requires a separate freeze and independent data"
        ),
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    compact = [
        {
            "PCA_width": item["PCA_width"],
            "status": item["status"],
            "design_audit": item["design_audit"],
            "pooled_metrics": item.get("nested_OOF", {}).get("pooled_OOF_metrics"),
        }
        for item in results
    ]
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
