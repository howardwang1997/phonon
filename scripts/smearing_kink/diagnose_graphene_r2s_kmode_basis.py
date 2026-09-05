#!/usr/bin/env python3
"""Test explicit conservative folded-K mode invariants after R2R-1.

This is a fixed-6x6 representation diagnostic.  It constructs scalar energy
invariants of the frozen complex A-prime coordinate q and the translation-free
mean-square displacement S, differentiates them analytically, and appends the
resulting force columns to the frozen 65-column R2R design.  Every added term
has polynomial degree >=3, so its value, force, and Hessian vanish at the
reference.  No development or held labels are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

import graphene_r2r1_linear_readout as r2r1
import graphene_r2r_multipolar_background as r2r
from diagnose_graphene_r2r1_aprime_objective import (
    DEFAULT_AGGREGATE,
    DEFAULT_FIT,
    _parse_whitelist,
    canonical_json_bytes,
    file_sha256,
)
from diagnose_graphene_r2s_nonlinear_readout_basis import design_audit, nested_oof


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data/graphene_r2o_taylor_null_core/reference_6x6.xyz"
DEFAULT_OUTPUT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/R2S_Kmode_basis_diagnostic_20260826"
)
PROJECTION_MASSES = (0.0, 0.50, 0.90)


def parse_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if file_sha256(path) != r2r1.THERMAL92_FILE_SHA256:
        raise ValueError("thermal92 SHA256 changed")
    positions: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        index = 0
        while True:
            count_line = handle.readline()
            if not count_line:
                break
            if not count_line.strip():
                continue
            if int(count_line) != 72:
                raise ValueError("thermal atom count changed")
            header = r2r1._header_fields(handle.readline())
            schema = r2r1._parse_properties(header["Properties"])
            expected = (
                r2r1.E50_PROPERTY_SCHEMA
                if index < 20
                else r2r1.AUXILIARY_PROPERTY_SCHEMA
            )
            if schema != expected:
                raise ValueError(f"thermal schema changed at {index}")
            offsets, row_width = r2r1._property_offsets(schema)
            start, width = offsets["pos"]
            if width != 3:
                raise ValueError("position width changed")
            one = []
            for _ in range(72):
                tokens = handle.readline().split()
                if len(tokens) != row_width:
                    raise ValueError("thermal row width changed")
                one.append([float(value) for value in tokens[start : start + 3]])
            lattice = [float(value) for value in header["Lattice"].split()]
            positions.append(np.asarray(one, dtype=np.float64))
            cells.append(np.asarray(lattice, dtype=np.float64).reshape(3, 3))
            index += 1
    if index != 92:
        raise ValueError("thermal structure count changed")
    return np.asarray(positions), np.asarray(cells)


def mic_displacement(
    positions: np.ndarray, cells: np.ndarray, reference_positions: np.ndarray
) -> np.ndarray:
    output = np.empty_like(positions)
    for index, (one, cell) in enumerate(zip(positions, cells, strict=True)):
        delta = one - reference_positions
        fractional = delta @ np.linalg.inv(cell)
        fractional[:, :2] -= np.rint(fractional[:, :2])
        aligned = fractional @ cell
        aligned -= np.mean(aligned, axis=0, keepdims=True)
        output[index] = aligned
    return output


def choose_effective_mode(
    mode: np.ndarray, displacement: np.ndarray, expected_coordinate: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = {
        "vdot_mode": mode,
        "vdot_conjugate_mode": np.conjugate(mode),
        "minus_vdot_mode": -mode,
        "minus_vdot_conjugate_mode": -np.conjugate(mode),
    }
    records = []
    for name, candidate in candidates.items():
        coordinate = np.asarray(
            [np.vdot(candidate.reshape(-1), one.reshape(-1)) for one in displacement[:20]]
        )
        records.append(
            {
                "name": name,
                "max_abs_error_A": float(np.max(np.abs(coordinate - expected_coordinate))),
                "mode": candidate,
            }
        )
    selected = min(records, key=lambda item: item["max_abs_error_A"])
    if selected["max_abs_error_A"] > 1.0e-7:
        raise ValueError("no frozen-mode convention replays the E50 coordinates")
    return np.asarray(selected["mode"]), {
        "selected": selected["name"],
        "selected_max_abs_error_A": selected["max_abs_error_A"],
        "candidate_errors_A": {
            item["name"]: item["max_abs_error_A"] for item in records
        },
    }


def kmode_columns(
    displacement: np.ndarray, effective_mode: np.ndarray, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    count, atoms, _ = displacement.shape
    mode_real = effective_mode.real
    mode_imag = effective_mode.imag
    grad_qr = mode_real - np.mean(mode_real, axis=0, keepdims=True)
    grad_qi = -mode_imag + np.mean(mode_imag, axis=0, keepdims=True)
    coordinate = np.asarray(
        [np.vdot(effective_mode.reshape(-1), one.reshape(-1)) for one in displacement]
    )
    qr = coordinate.real
    qi = coordinate.imag
    q2 = np.square(qr) + np.square(qi)
    s = np.mean(np.sum(np.square(displacement), axis=2), axis=1)
    grad_s = 2.0 * displacement / float(atoms)

    energy_columns = []
    force_columns = []
    degrees: dict[str, int] = {}
    for name in names:
        if name == "Re_q3":
            energy = np.power(qr, 3) - 3.0 * qr * np.square(qi)
            d_qr = 3.0 * (np.square(qr) - np.square(qi))
            d_qi = -6.0 * qr * qi
            d_s = np.zeros(count)
            degree = 3
        elif name == "Im_q3":
            energy = 3.0 * np.square(qr) * qi - np.power(qi, 3)
            d_qr = 6.0 * qr * qi
            d_qi = 3.0 * (np.square(qr) - np.square(qi))
            d_s = np.zeros(count)
            degree = 3
        elif name == "q4":
            energy = np.square(q2)
            d_qr = 4.0 * qr * q2
            d_qi = 4.0 * qi * q2
            d_s = np.zeros(count)
            degree = 4
        elif name == "q2_S":
            energy = q2 * s
            d_qr = 2.0 * qr * s
            d_qi = 2.0 * qi * s
            d_s = q2
            degree = 4
        elif name == "S2":
            energy = np.square(s)
            d_qr = np.zeros(count)
            d_qi = np.zeros(count)
            d_s = 2.0 * s
            degree = 4
        elif name == "Re_q3_S":
            q3 = np.power(qr, 3) - 3.0 * qr * np.square(qi)
            energy = q3 * s
            d_qr = 3.0 * (np.square(qr) - np.square(qi)) * s
            d_qi = -6.0 * qr * qi * s
            d_s = q3
            degree = 5
        elif name == "Im_q3_S":
            q3 = 3.0 * np.square(qr) * qi - np.power(qi, 3)
            energy = q3 * s
            d_qr = 6.0 * qr * qi * s
            d_qi = 3.0 * (np.square(qr) - np.square(qi)) * s
            d_s = q3
            degree = 5
        elif name == "q4_S":
            energy = np.square(q2) * s
            d_qr = 4.0 * qr * q2 * s
            d_qi = 4.0 * qi * q2 * s
            d_s = np.square(q2)
            degree = 6
        elif name == "q2_S2":
            energy = q2 * np.square(s)
            d_qr = 2.0 * qr * np.square(s)
            d_qi = 2.0 * qi * np.square(s)
            d_s = 2.0 * q2 * s
            degree = 6
        elif name == "S3":
            energy = np.power(s, 3)
            d_qr = np.zeros(count)
            d_qi = np.zeros(count)
            d_s = 3.0 * np.square(s)
            degree = 6
        else:
            raise ValueError(f"unknown K-mode invariant: {name}")
        gradient = (
            d_qr[:, None, None] * grad_qr[None, ...]
            + d_qi[:, None, None] * grad_qi[None, ...]
            + d_s[:, None, None] * grad_s
        )
        energy_columns.append(energy)
        force_columns.append(-gradient)
        degrees[name] = degree
    energy_design = np.stack(energy_columns, axis=1)
    force_design = np.stack(force_columns, axis=3)
    receipt = {
        "coordinate_abs_A_range": [
            float(np.min(np.abs(coordinate))),
            float(np.max(np.abs(coordinate))),
        ],
        "translation_free_mean_square_displacement_A2_range": [
            float(np.min(s)),
            float(np.max(s)),
        ],
        "polynomial_degree": degrees,
        "analytic_reference_value_force_Hessian_zero": all(
            degree >= 3 for degree in degrees.values()
        ),
    }
    return energy_design, force_design, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thermal", type=Path, default=r2r1.RECOMMENDED_THERMAL92)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--fit-root", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference_force, aprime, label_hashes = _parse_whitelist(args.thermal.resolve())
    positions, cells = parse_positions(args.thermal.resolve())
    if file_sha256(args.reference.resolve()) != r2r.EXPECTED_INPUTS["reference_6x6"]["sha256"]:
        raise ValueError("reference_6x6 SHA256 changed")
    reference_atoms = read(args.reference, index=0)
    displacement = mic_displacement(
        positions, cells, np.asarray(reference_atoms.positions, dtype=np.float64)
    )
    stored_coordinate = aprime.coordinates[:, 0] + 1.0j * aprime.coordinates[:, 1]
    raw_mode = aprime.mode_real[0] + 1.0j * aprime.mode_imag[0]
    effective_mode, mode_receipt = choose_effective_mode(
        raw_mode, displacement, stored_coordinate
    )

    if file_sha256(args.aggregate.resolve()) != r2r1.ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("attempt3 aggregate SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        base_energy = np.asarray(arrays["thermal_parameter_energy_design_eV"], np.float64)
        base_force = np.asarray(arrays["thermal_parameter_force_design_eV_A"], np.float64)
        fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], np.float64)

    basis_definitions = {
        "K_even6": ("q4", "q2_S", "S2", "q4_S", "q2_S2", "S3"),
        "K_low5": ("Re_q3", "Im_q3", "q4", "q2_S", "S2"),
        "K_full10": (
            "Re_q3",
            "Im_q3",
            "q4",
            "q2_S",
            "S2",
            "Re_q3_S",
            "Im_q3_S",
            "q4_S",
            "q2_S2",
            "S3",
        ),
    }
    results = []
    arrays_to_save: dict[str, np.ndarray] = {}
    basis_receipts: dict[str, Any] = {}
    for basis_name, names in basis_definitions.items():
        k_energy, k_force, receipt = kmode_columns(displacement, effective_mode, names)
        energy = np.concatenate((base_energy, k_energy), axis=1)
        force = np.concatenate((base_force, k_force), axis=3)
        audit = design_audit(force)
        basis_receipts[basis_name] = {
            "invariants": list(names),
            "Kmode_receipt": receipt,
            "design_audit": audit,
            "energy_design_shape": list(energy.shape),
            "force_design_shape": list(force.shape),
        }
        if not audit["pass"]:
            results.append(
                {"basis": basis_name, "status": "DESIGN_AUDIT_FAILED", "audit": audit}
            )
            continue
        for projection_mass in PROJECTION_MASSES:
            oof, nested = nested_oof(
                force, fixed, reference_force, aprime, projection_mass
            )
            key = f"{basis_name}_projection_mass_{projection_mass:.2f}"
            arrays_to_save[key] = oof
            results.append(
                {
                    "basis": basis_name,
                    "projection_mass": projection_mass,
                    "status": (
                        "CONDITIONAL_OOF_GATE_PASSED"
                        if nested["pooled_metrics"]["passes_fixed_gate"]
                        else "CONDITIONAL_OOF_GATE_FAILED"
                    ),
                    "nested_OOF": nested,
                    "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
                }
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "Kmode_basis_oof_arrays.npz", **arrays_to_save)
    summary = {
        "format": "graphene_r2s_folded_Kmode_basis_diagnostic_v1",
        "status": "FIXED_CELL_REPRESENTATION_DIAGNOSTIC_COMPLETE_NOT_DEPLOYABLE",
        "scope": "thermal92 whitelist + frozen design + reference geometry only",
        "input_sha256": {
            "thermal92": file_sha256(args.thermal),
            "aggregate_arrays": file_sha256(args.aggregate),
            "reference_6x6": file_sha256(args.reference),
            "terminal_R2R1_fit_receipt": file_sha256(args.fit_root / "fit_receipt.json"),
        },
        "label_raw_sha256": label_hashes,
        "mode_coordinate_replay": mode_receipt,
        "basis_receipts": basis_receipts,
        "projection_masses": list(PROJECTION_MASSES),
        "results": results,
        "interpretation_boundary": [
            "The added mode basis is tied to the fixed 6x6 reference and is not yet O(3)/size deployable.",
            "This thermal92 scan is representation development, not an independent gate.",
            "Passing requires a separate K-star-symmetric production implementation and unseen validation.",
        ],
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    compact = [
        {
            "basis": item["basis"],
            "projection_mass": item.get("projection_mass"),
            "status": item["status"],
            "pooled_metrics": item.get("nested_OOF", {}).get("pooled_metrics"),
        }
        for item in results
    ]
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
