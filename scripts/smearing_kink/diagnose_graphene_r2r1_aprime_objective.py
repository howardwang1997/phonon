#!/usr/bin/env python3
"""Diagnose the terminal R2R-1 A-prime OOF failure without held-data access.

Only the frozen thermal92 force/A-prime whitelist is converted to numbers.  The
script reproduces the published OOF metrics, decomposes the collective-mode
residual, and evaluates a diagnostic Pareto curve obtained by adding an
explicit A-prime projection term to the existing 65-column ridge objective.
The Pareto scan is development evidence, not a deployable model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import graphene_r2r1_linear_readout as r2r1


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/formal_r2r0_three_host_seed83_attempt3"
    / "aggregate/arrays.npz"
)
DEFAULT_FIT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
    / "R2R1_conditional_linear_readout_fit_seed83_attempt2_retry1_20260826"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/R2R1_Aprime_objective_diagnostic_20260826"
)
PROJECTION_MASSES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_whitelist(path: Path) -> tuple[np.ndarray, r2r1.AprimeData, dict[str, str]]:
    if file_sha256(path) != r2r1.THERMAL92_FILE_SHA256:
        raise ValueError("thermal92 SHA256 differs from the frozen R2R-1 input")

    reference: list[np.ndarray] = []
    mode_real: list[np.ndarray] = []
    mode_imag: list[np.ndarray] = []
    coordinates: list[list[float]] = []
    foundation: list[np.ndarray] = []
    q6: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        global_index = 0
        while True:
            count_line = handle.readline()
            if not count_line:
                break
            if not count_line.strip():
                continue
            if int(count_line) != 72:
                raise ValueError("thermal92 atom count changed")
            header = r2r1._header_fields(handle.readline())
            schema = r2r1._parse_properties(header["Properties"])
            group = r2r1.group_for_global_index(global_index)
            expected = (
                r2r1.E50_PROPERTY_SCHEMA
                if group == "E50_seed0"
                else r2r1.AUXILIARY_PROPERTY_SCHEMA
            )
            if schema != expected:
                raise ValueError(f"thermal92 schema changed at index {global_index}")
            if header.get("config_type") != r2r1.THERMAL_CONFIG_TYPES[group]:
                raise ValueError(f"thermal92 order changed at index {global_index}")
            offsets, row_width = r2r1._property_offsets(schema)
            names = ["REF_forces"]
            if group == "E50_seed0":
                names.extend(
                    [
                        "FOUNDATION_BASE_forces",
                        "FROZEN_Q6_forces",
                        "APRIME_mode_real",
                        "APRIME_mode_imag",
                    ]
                )
                coordinates.append(
                    [
                        float(header["APRIME_coordinate_real_A"]),
                        float(header["APRIME_coordinate_imag_A"]),
                    ]
                )
            converted = {name: [] for name in names}
            for _ in range(72):
                tokens = handle.readline().split()
                if len(tokens) != row_width:
                    raise ValueError("thermal92 atom-row width changed")
                for name in names:
                    start, width = offsets[name]
                    converted[name].append(
                        [float(value) for value in tokens[start : start + width]]
                    )
            reference.append(np.asarray(converted["REF_forces"], dtype=np.float64))
            if group == "E50_seed0":
                foundation.append(
                    np.asarray(converted["FOUNDATION_BASE_forces"], dtype=np.float64)
                )
                q6.append(np.asarray(converted["FROZEN_Q6_forces"], dtype=np.float64))
                mode_real.append(
                    np.asarray(converted["APRIME_mode_real"], dtype=np.float64)
                )
                mode_imag.append(
                    np.asarray(converted["APRIME_mode_imag"], dtype=np.float64)
                )
            global_index += 1
    if global_index != 92:
        raise ValueError(f"thermal92 count is {global_index}, expected 92")

    force = np.asarray(reference, dtype="<f8", order="C")
    aprime = r2r1.AprimeData(
        mode_real=np.asarray(mode_real, dtype="<f8", order="C"),
        mode_imag=np.asarray(mode_imag, dtype="<f8", order="C"),
        coordinates=np.asarray(coordinates, dtype="<f8", order="C"),
        foundation_base_force_eV_A=np.asarray(foundation, dtype="<f8", order="C"),
        frozen_q6_force_eV_A=np.asarray(q6, dtype="<f8", order="C"),
    )
    raw = {
        "REF_forces_all92": r2r1.raw_array_sha256(force, "<f8"),
        "REF_forces_E50": r2r1.raw_array_sha256(force[:20], "<f8"),
        "REF_forces_T300": r2r1.raw_array_sha256(force[20:56], "<f8"),
        "REF_forces_T600": r2r1.raw_array_sha256(force[56:92], "<f8"),
        "APRIME_mode_real": r2r1.raw_array_sha256(aprime.mode_real, "<f8"),
        "APRIME_mode_imag": r2r1.raw_array_sha256(aprime.mode_imag, "<f8"),
        "APRIME_coordinates": r2r1.raw_array_sha256(aprime.coordinates, "<f8"),
        "FOUNDATION_BASE_forces": r2r1.raw_array_sha256(
            aprime.foundation_base_force_eV_A, "<f8"
        ),
        "FROZEN_Q6_forces": r2r1.raw_array_sha256(
            aprime.frozen_q6_force_eV_A, "<f8"
        ),
    }
    if raw != dict(r2r1.EXPECTED_LABEL_RAW_SHA256):
        raise ValueError("whitelisted thermal92 arrays differ from R2R-1 hashes")
    return force, aprime, raw


def _project(mode: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.vdot(one_mode.reshape(-1), one_value.reshape(-1))
            for one_mode, one_value in zip(mode, values, strict=True)
        ],
        dtype=np.complex128,
    )


def _column_scale(design: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = design[indices]
    return np.sqrt(np.mean(np.square(selected), axis=(0, 1, 2)))


def _ridge(
    design: np.ndarray,
    fixed: np.ndarray,
    reference: np.ndarray,
    aprime: r2r1.AprimeData,
    train_indices: Sequence[int],
    alpha: float,
    projection_mass: float,
) -> np.ndarray:
    indices = np.asarray(sorted(set(int(value) for value in train_indices)), dtype=int)
    scale = _column_scale(design, indices)
    if np.any(scale <= np.max(scale) * r2r1.ZERO_COLUMN_RELATIVE_RMS):
        raise ValueError("diagnostic ridge encountered a zero design column")
    normalized_design = design[indices] / scale
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
    matrix_parts = [normalized_design.reshape(-1, r2r1.LINEAR_WIDTH)]
    response_parts = [response.reshape(-1)]
    weight_parts = [np.asarray(force_weights, dtype=np.float64)]

    if projection_mass > 0.0:
        e50 = indices[indices < 20]
        local = e50
        complex_mode = aprime.mode_real[local] + 1.0j * aprime.mode_imag[local]
        projected_columns = np.empty((len(e50), r2r1.LINEAR_WIDTH), np.complex128)
        for row, (mode, columns) in enumerate(
            zip(complex_mode, normalized_design[np.searchsorted(indices, e50)], strict=True)
        ):
            for column in range(r2r1.LINEAR_WIDTH):
                projected_columns[row, column] = np.vdot(
                    mode.reshape(-1), columns[..., column].reshape(-1)
                )
        projected_target = _project(
            complex_mode, reference[e50] - fixed[e50]
        )
        # q predicts 0.030 * <mode,D>q; normalize the collective residual by
        # its fixed 0.015 eV/A gate before adding real and imaginary rows.
        projected_columns *= r2r1.FORCE_SCALE_EV_A / 0.015
        projected_target /= 0.015
        projection_matrix = np.concatenate(
            (projected_columns.real, projected_columns.imag), axis=0
        )
        projection_response = np.concatenate(
            (projected_target.real, projected_target.imag), axis=0
        )
        matrix_parts.append(projection_matrix)
        response_parts.append(projection_response)
        weight_parts.append(
            np.full(2 * len(e50), projection_mass / (2 * len(e50)), np.float64)
        )

    matrix = np.concatenate(matrix_parts, axis=0)
    target = np.concatenate(response_parts)
    weights = np.concatenate(weight_parts)
    if not math.isclose(float(np.sum(weights)), 1.0, abs_tol=3.0e-15, rel_tol=0.0):
        raise RuntimeError("diagnostic objective weights do not sum to one")
    root_weight = np.sqrt(weights)
    u, singular, vt = np.linalg.svd(matrix * root_weight[:, None], full_matrices=False)
    normalized = vt.T @ (
        singular
        / (np.square(singular) + float(alpha))
        * (u.T @ (target * root_weight))
    )
    return r2r1.FORCE_SCALE_EV_A * normalized / scale


def _predict(design: np.ndarray, fixed: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return fixed + np.einsum("natk,k->nat", design, coefficient, optimize=False)


def _score(metrics: dict[str, Any]) -> tuple[float, float]:
    return (
        float(metrics["selection_score_rounded_12"]),
        -float(metrics.get("diagnostic_alpha", 0.0)),
    )


def _nested_oof(
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
        candidates = []
        for alpha in r2r1.ALPHA_GRID:
            inner_prediction = np.full_like(reference, np.nan)
            for inner_id, inner_tuple in enumerate(r2r1.FOLD_GLOBAL_INDICES):
                if inner_id == outer_id:
                    continue
                inner_hold = np.asarray(inner_tuple, dtype=int)
                inner_train = np.setdiff1d(outer_train, inner_hold)
                coefficient = _ridge(
                    design,
                    fixed,
                    reference,
                    aprime,
                    inner_train,
                    alpha,
                    projection_mass,
                )
                inner_prediction[inner_hold] = _predict(
                    design[inner_hold], fixed[inner_hold], coefficient
                )
            metrics = r2r1.gate_metrics(
                inner_prediction, reference, aprime, outer_train
            )
            metrics["diagnostic_alpha"] = float(alpha)
            candidates.append(metrics)
        selected = min(candidates, key=_score)
        alpha = float(selected["diagnostic_alpha"])
        coefficient = _ridge(
            design,
            fixed,
            reference,
            aprime,
            outer_train,
            alpha,
            projection_mass,
        )
        oof[outer_hold] = _predict(
            design[outer_hold], fixed[outer_hold], coefficient
        )
        records.append(
            {
                "outer_fold": outer_id,
                "selected_alpha": alpha,
                "inner_selection_score": float(selected["raw_selection_score"]),
                "inner_Aprime_RMS_meV_A": float(
                    selected["E50_Aprime"]["RMS_meV_A"]
                ),
            }
        )
    metrics = r2r1.gate_metrics(oof, reference, aprime, all_indices)
    return oof, {"outer_records": records, "pooled_metrics": metrics}


def _complex_residual_decomposition(
    oof: np.ndarray, reference: np.ndarray, aprime: r2r1.AprimeData
) -> dict[str, Any]:
    mode = aprime.mode_real + 1.0j * aprime.mode_imag
    error = _project(mode, oof[:20] - reference[:20])
    coordinate = aprime.coordinates[:, 0] + 1.0j * aprime.coordinates[:, 1]
    coefficient = np.vdot(coordinate, error) / np.vdot(coordinate, coordinate)
    remainder = error - coefficient * coordinate
    constant = np.mean(error)
    centered = error - constant
    return {
        "RMS_meV_A": float(1000.0 * np.sqrt(np.mean(np.abs(error) ** 2))),
        "coordinate_linear_coefficient_eV_A2": {
            "real": float(coefficient.real),
            "imag": float(coefficient.imag),
        },
        "RMS_after_best_complex_coordinate_linear_term_meV_A": float(
            1000.0 * np.sqrt(np.mean(np.abs(remainder) ** 2))
        ),
        "mean_complex_error_meV_A": {
            "real": float(1000.0 * constant.real),
            "imag": float(1000.0 * constant.imag),
        },
        "RMS_after_complex_mean_removal_meV_A": float(
            1000.0 * np.sqrt(np.mean(np.abs(centered) ** 2))
        ),
        "coordinate_abs_A_range": [
            float(np.min(np.abs(coordinate))),
            float(np.max(np.abs(coordinate))),
        ],
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
        raise ValueError("attempt3 aggregate NPZ SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], dtype=np.float64)
        design = np.asarray(
            arrays["thermal_parameter_force_design_eV_A"], dtype=np.float64
        )
    with np.load(args.fit_root / "fit_arrays.npz", allow_pickle=False) as arrays:
        published_oof = np.asarray(arrays["OOF_predicted_force_eV_A"], dtype=np.float64)
    receipt = json.loads((args.fit_root / "fit_receipt.json").read_text())
    published_metrics = r2r1.gate_metrics(
        published_oof, reference, aprime, np.arange(92)
    )
    expected_metrics = receipt["pipeline_receipt"]["nested_OOF"]["pooled_OOF_metrics"]
    if canonical_json_bytes(published_metrics) != canonical_json_bytes(expected_metrics):
        raise ValueError("published R2R-1 OOF metrics do not replay exactly")

    scans: list[dict[str, Any]] = []
    per_mass_oof: dict[str, np.ndarray] = {}
    for mass in PROJECTION_MASSES:
        oof, result = _nested_oof(design, fixed, reference, aprime, mass)
        key = f"projection_mass_{mass:.2f}"
        per_mass_oof[key] = oof
        scans.append(
            {
                "projection_mass": mass,
                "force_objective_mass": 1.0 - mass,
                **result,
                "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
            }
        )

    decomposition = _complex_residual_decomposition(published_oof, reference, aprime)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(
        output / "diagnostic_oof_arrays.npz",
        published_force_only_oof=published_oof,
        **per_mass_oof,
    )
    with (output / "projection_mass_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "projection_mass",
                "E50_Aprime_RMS_meV_A",
                "E50_slope_relative_error",
                "E50_force_RMSE_meV_A",
                "T300_force_RMSE_meV_A",
                "T600_force_RMSE_meV_A",
                "force_max_abs_meV_A",
                "passes_fixed_gate",
            ],
        )
        writer.writeheader()
        for scan in scans:
            metrics = scan["pooled_metrics"]
            writer.writerow(
                {
                    "projection_mass": scan["projection_mass"],
                    "E50_Aprime_RMS_meV_A": metrics["E50_Aprime"]["RMS_meV_A"],
                    "E50_slope_relative_error": metrics["E50_Aprime"][
                        "slope_relative_error"
                    ],
                    "E50_force_RMSE_meV_A": metrics["force_by_group"]["E50_seed0"][
                        "RMSE_meV_A"
                    ],
                    "T300_force_RMSE_meV_A": metrics["force_by_group"]["T300"][
                        "RMSE_meV_A"
                    ],
                    "T600_force_RMSE_meV_A": metrics["force_by_group"]["T600"][
                        "RMSE_meV_A"
                    ],
                    "force_max_abs_meV_A": max(
                        value["max_abs_meV_A"]
                        for value in metrics["force_by_group"].values()
                    ),
                    "passes_fixed_gate": metrics["passes_fixed_gate"],
                }
            )

    summary = {
        "format": "graphene_r2r1_Aprime_objective_failure_diagnostic_v1",
        "status": "DIAGNOSTIC_COMPLETE_NOT_A_MODEL_SELECTION",
        "scope": "thermal92 force/A-prime whitelist only; no development or held data",
        "input_sha256": {
            "thermal92": file_sha256(args.thermal),
            "aggregate_arrays": file_sha256(args.aggregate),
            "fit_arrays": file_sha256(args.fit_root / "fit_arrays.npz"),
            "fit_receipt": file_sha256(args.fit_root / "fit_receipt.json"),
        },
        "label_raw_sha256": label_hashes,
        "published_force_only_OOF_metrics": published_metrics,
        "published_residual_decomposition": decomposition,
        "projection_mass_scan": scans,
        "interpretation_boundary": [
            "The projection-mass scan reuses thermal92 and is development evidence only.",
            "It cannot authorize a checkpoint, mechanics, SSCHA, or spectra.",
            "Any next candidate must be frozen separately and pass independent data gates.",
        ],
    }
    raw = canonical_json_bytes(summary) + b"\n"
    (output / "summary.json").write_bytes(raw)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
