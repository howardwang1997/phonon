#!/usr/bin/env python3
"""Audit one dense K-line EPW static-self-energy calculation for B0."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


MEV_TO_CM1 = 8.06554393734921


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_qpoints(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(rows[0]) != 2 or rows[0][1] != "crystal":
        raise ValueError("q-point header must be '<count> crystal'")
    count = int(rows[0][0])
    if len(rows) != count + 1:
        raise ValueError(f"q-point row count mismatch: {len(rows) - 1} vs {count}")
    qpoints = np.asarray([[float(value) for value in row[:3]] for row in rows[1:]])
    weights = np.asarray([float(row[3]) for row in rows[1:]])
    return qpoints, weights


def load_static(path: Path, n_qpoints: int) -> dict:
    rows = {}
    all_noncomment = 0
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        all_noncomment += 1
        fields = stripped.split()
        if len(fields) != 9:
            raise ValueError(f"unexpected specfun_sup.phon row: {line}")
        iq, mode = int(fields[0]), int(fields[1])
        values = [float(value) for value in fields[2:]]
        if abs(values[3]) <= 1.0e-12:
            key = (iq, mode)
            if key in rows:
                raise ValueError(f"duplicate zero-frequency row {key}")
            rows[key] = {
                "temperature_K": values[0],
                "broadening_eV": values[1],
                "bare_eV": values[2],
                "pi_low_meV": values[4],
                "pi_high_meV": values[5],
                "imag_pi_low_meV": values[6],
            }
    expected = n_qpoints * 6
    if len(rows) != expected:
        raise ValueError(f"zero-frequency rows: {len(rows)}, expected {expected}")
    if set(rows) != {
        (iq, mode)
        for iq in range(1, n_qpoints + 1)
        for mode in range(1, 7)
    }:
        raise ValueError("static-self-energy iq/mode coverage is incomplete")
    return {"rows": rows, "all_noncomment_rows": all_noncomment}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpoints", type=Path, required=True)
    parser.add_argument("--static-self-energy", type=Path, required=True)
    parser.add_argument("--epw-output", type=Path, required=True)
    parser.add_argument("--temperature-K", type=float, required=True)
    parser.add_argument("--expected-nq", type=int, default=29)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qpoints, weights = load_qpoints(args.qpoints)
    static = load_static(args.static_self_energy, len(qpoints))
    rows = static["rows"]
    t_values = 3.0 * qpoints[:, 0]
    expected_t = np.arange(226, 255, dtype=float) / 240.0
    if args.expected_nq != len(expected_t):
        raise ValueError(
            "the frozen k720-commensurate K line contains exactly 29 q points"
        )
    temperatures = np.asarray([row["temperature_K"] for row in rows.values()])
    broadenings = np.asarray([row["broadening_eV"] for row in rows.values()])
    finite_values = np.asarray(
        [
            value
            for row in rows.values()
            for value in (
                row["bare_eV"],
                row["pi_low_meV"],
                row["pi_high_meV"],
                row["imag_pi_low_meV"],
            )
        ]
    )

    corrected_top = []
    delta_lambda_top = []
    for iq in range(1, len(qpoints) + 1):
        row = rows[(iq, 6)]
        bare_cm1 = row["bare_eV"] * 1000.0 * MEV_TO_CM1
        delta_lambda = (
            2.0
            * row["bare_eV"]
            * 1000.0
            * (row["pi_low_meV"] - row["pi_high_meV"])
            * MEV_TO_CM1**2
        )
        corrected_top.append(np.sqrt(max(bare_cm1**2 + delta_lambda, 0.0)))
        delta_lambda_top.append(delta_lambda)
    corrected_top = np.asarray(corrected_top)
    delta_lambda_top = np.asarray(delta_lambda_top)
    symmetry_frequency_error = float(
        np.max(np.abs(corrected_top - corrected_top[::-1]))
    )
    symmetry_shift_error = float(
        np.max(np.abs(delta_lambda_top - delta_lambda_top[::-1]))
    )

    checks = {
        "EPW_completed": "Total program execution" in args.epw_output.read_text(
            errors="replace"
        ),
        "qpoint_count_matches": len(qpoints) == args.expected_nq,
        "qpoints_are_symmetric_K_line": (
            qpoints.shape == (args.expected_nq, 3)
            and np.allclose(qpoints[:, 0], qpoints[:, 1], atol=1.0e-12, rtol=0.0)
            and np.allclose(qpoints[:, 2], 0.0, atol=1.0e-12, rtol=0.0)
            and np.allclose(t_values, expected_t, atol=2.0e-10, rtol=0.0)
        ),
        "weights_sum_to_one": abs(float(weights.sum()) - 1.0) <= 1.0e-9,
        "zero_frequency_iq_mode_coverage_complete": len(rows) == len(qpoints) * 6,
        "target_temperature_matches": float(
            np.max(np.abs(temperatures - args.temperature_K))
        )
        <= 1.0e-8,
        "numerical_broadening_is_0p005_eV": float(
            np.max(np.abs(broadenings - 0.005))
        )
        <= 1.0e-12,
        "all_values_finite": bool(np.isfinite(finite_values).all()),
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "scope": f"B0 dense K-line EPW integrity audit at {args.temperature_K:g} K",
        "temperature_K": args.temperature_K,
        "n_qpoints": len(qpoints),
        "t_min": float(t_values.min()),
        "t_max": float(t_values.max()),
        "t_step": float(np.median(np.diff(t_values))),
        "n_static_rows": len(rows),
        "n_all_noncomment_specfun_rows": static["all_noncomment_rows"],
        "metrics": {
            "mirror_corrected_top_frequency_max_abs_cm-1": symmetry_frequency_error,
            "mirror_top_delta_lambda_max_abs_cm-2": symmetry_shift_error,
            "top_delta_lambda_min_cm-2": float(delta_lambda_top.min()),
            "top_delta_lambda_max_cm-2": float(delta_lambda_top.max()),
        },
        "checks": checks,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
