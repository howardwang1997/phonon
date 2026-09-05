#!/usr/bin/env python3
"""Evaluate an extended electronic-smearing ladder on one lattice background.

Only the full Wannier-EPC electronic response changes with smearing/degauss.
Every curve uses the same static short-range MLIP dynamical matrix and the
same absolute long-range reference constant as the accepted zero-smearing
curve.  No finite-lattice-temperature or TDEP background enters the sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate_graphene_zero_finite_full_kpath as full  # noqa: E402
from extract_graphene_epw_dynamical_matrices import RYDBERG_CM1  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
DEFAULT_SMEARINGS_RY = (
    0.0019000869,
    0.00285013035,
    0.0038001738,
    0.005,
    0.010,
    0.020,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-nk", type=int, default=144)
    parser.add_argument("--patch-n", type=int, default=180)
    parser.add_argument("--patch-halfwidth", type=float, default=0.04)
    parser.add_argument("--eta-eV", type=float, default=0.005)
    parser.add_argument("--maximum-distance", type=float, default=0.06)
    parser.add_argument("--points-per-direction", type=int, default=121)
    parser.add_argument("--fermi-eV", type=float, default=-1.7187)
    parser.add_argument("--reference-degauss-Ry", type=float, default=0.020000)
    parser.add_argument(
        "--smearings-Ry",
        type=float,
        nargs="+",
        default=DEFAULT_SMEARINGS_RY,
    )
    parser.add_argument(
        "--zero-curve",
        type=Path,
        default=BASE / "E38_epc_zero_extended_curve/zero_dense_curve.csv",
    )
    parser.add_argument(
        "--hr",
        type=Path,
        default=BASE
        / "E0_epw_matched/wannier_diagnostics/matched_k18q9_ex1_pifroz_hr.dat",
    )
    parser.add_argument(
        "--epwdata",
        type=Path,
        default=BASE / "E0_epw_matched/k18_q9_ex1_pifroz/restart_small/epwdata.fmt",
    )
    parser.add_argument(
        "--crystal",
        type=Path,
        default=BASE / "E0_epw_matched/k18_q9_ex1_pifroz/restart_small/crystal.fmt",
    )
    parser.add_argument(
        "--epmatwp",
        type=Path,
        default=BASE / "E9_epc_bandsum_assets/graphene.epmatwp",
    )
    parser.add_argument(
        "--overlap-reference",
        type=Path,
        default=BASE
        / "E41_fixed_lattice_extended_smearing_overlay"
        / "fixed_lattice_extended_smearing_overlay.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E42_extended_fixed_lattice_smearing_sweep",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def point_index(
    records: list[tuple[str, float]], direction: str, distance: float
) -> int:
    matches = [
        index
        for index, (label, value) in enumerate(records)
        if label == direction and np.isclose(value, distance, atol=1.0e-14)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {direction} d={distance} point, found {len(matches)}"
        )
    return matches[0]


def series_metrics(
    smearings: np.ndarray,
    records: list[tuple[str, float]],
    frequencies: np.ndarray,
) -> list[dict]:
    output = []
    K_index = point_index(records, "K", 0.0)
    for series_index, smearing in enumerate(smearings):
        row = {
            "smearing_degauss_Ry": float(smearing),
            "K_frequency_cm-1": float(frequencies[series_index, K_index]),
        }
        for direction in ("KG", "KM"):
            for distance in (0.003, 0.023, 0.060):
                index = point_index(records, direction, distance)
                row[f"{direction}_d{distance:.3f}_rise_cm-1"] = float(
                    frequencies[series_index, index]
                    - frequencies[series_index, K_index]
                )
        output.append(row)
    return output


def overlap_error(
    path: Path,
    smearings: np.ndarray,
    records: list[tuple[str, float]],
    frequencies: np.ndarray,
) -> dict:
    if not path.exists():
        return {"available": False}
    rows = read_csv(path)
    reference = {
        (
            round(float(row["smearing_degauss_Ry"]), 12),
            row["direction"],
            round(float(row["distance"]), 12),
        ): float(row["controlled_MLIP_plus_full_EPC_cm-1"])
        for row in rows
    }
    errors = []
    matched_smearings = []
    for series_index, smearing in enumerate(smearings):
        smearing_key = round(float(smearing), 12)
        if not any(key[0] == smearing_key for key in reference):
            continue
        matched_smearings.append(float(smearing))
        for q_index, (direction, distance) in enumerate(records):
            key = (smearing_key, direction, round(float(distance), 12))
            if key not in reference:
                continue
            errors.append(frequencies[series_index, q_index] - reference[key])
    error = np.asarray(errors, float)
    return {
        "available": True,
        "matched_smearings_Ry": matched_smearings,
        "number_of_values": len(error),
        "maximum_absolute_error_cm-1": float(np.max(np.abs(error))),
        "RMSE_cm-1": float(np.sqrt(np.mean(error**2))),
    }


def main() -> int:
    args = parse_args()
    if args.coarse_nk % 3 != 0:
        raise ValueError("coarse_nk must be divisible by 3")
    finite_smearings = np.asarray(sorted(set(args.smearings_Ry)), float)
    if len(finite_smearings) != len(args.smearings_Ry):
        raise ValueError("smearing/degauss values must be unique")
    if np.any(finite_smearings <= 0.0):
        raise ValueError("finite smearing/degauss values must be positive")
    if finite_smearings[-1] > args.reference_degauss_Ry + 1.0e-14:
        raise ValueError("requested smearing exceeds the frozen 0.020 Ry reference")

    started = time.perf_counter()
    expected_qpoints = 2 * args.points_per_direction - 1
    zero_rows = full.load_zero_curve(args.zero_curve, expected_qpoints)
    records, qpoints = full.make_records(
        args.maximum_distance, args.points_per_direction
    )
    zero_qpoints = np.asarray(
        [[float(row["q1"]), float(row["q2"]), 0.0] for row in zero_rows],
        float,
    )
    if not np.allclose(qpoints, zero_qpoints, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("zero curve and requested extended q path differ")

    finite_response_Ry2 = full.evaluate_finite_response(
        args, qpoints, finite_smearings
    )
    zero_response_Ry2 = np.asarray(
        [float(row["full_EPC_response_Ry2"]) for row in zero_rows], float
    )
    common_background = np.asarray(
        [float(row["short_range_MLIP_cm-1"]) for row in zero_rows], float
    )
    zero_correction_cm2 = np.asarray(
        [float(row["long_range_correction_cm-2"]) for row in zero_rows], float
    )
    zero_response_cm2 = zero_response_Ry2 * RYDBERG_CM1**2
    constant_samples = zero_correction_cm2 - zero_response_cm2
    common_constant = float(np.mean(constant_samples))
    common_constant_spread = float(np.ptp(constant_samples))
    if common_constant_spread > 1.0e-5:
        raise RuntimeError("zero curve has no unique long-range reference constant")

    smearings = np.concatenate(([0.0], finite_smearings))
    responses_Ry2 = np.vstack((zero_response_Ry2, finite_response_Ry2))
    corrections_cm2 = common_constant + responses_Ry2 * RYDBERG_CM1**2
    squared_frequency = common_background[None, :] ** 2 + corrections_cm2
    frequencies = np.sign(squared_frequency) * np.sqrt(np.abs(squared_frequency))
    stored_zero = np.asarray(
        [float(row["MLIP_plus_full_EPC_cm-1"]) for row in zero_rows], float
    )
    zero_replay_error = float(np.max(np.abs(frequencies[0] - stored_zero)))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "extended_fixed_lattice_smearing_sweep.csv"
    fields = [
        "smearing_degauss_Ry",
        "lattice_temperature_K",
        "lattice_condition",
        "direction",
        "distance",
        "signed_distance",
        "q1",
        "q2",
        "short_range_background_cm-1",
        "full_EPC_response_Ry2",
        "long_range_correction_cm-2",
        "MLIP_plus_full_EPC_cm-1",
    ]
    signed = full.signed_axis(records)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for series_index, smearing in enumerate(smearings):
            for q_index, ((direction, distance), qpoint) in enumerate(
                zip(records, qpoints)
            ):
                writer.writerow(
                    {
                        "smearing_degauss_Ry": float(smearing),
                        "lattice_temperature_K": "",
                        "lattice_condition": "fixed_static_lattice_reference",
                        "direction": direction,
                        "distance": distance,
                        "signed_distance": signed[q_index],
                        "q1": qpoint[0],
                        "q2": qpoint[1],
                        "short_range_background_cm-1": common_background[q_index],
                        "full_EPC_response_Ry2": responses_Ry2[
                            series_index, q_index
                        ],
                        "long_range_correction_cm-2": corrections_cm2[
                            series_index, q_index
                        ],
                        "MLIP_plus_full_EPC_cm-1": frequencies[
                            series_index, q_index
                        ],
                    }
                )

    metrics = series_metrics(smearings, records, frequencies)
    mean_d003 = np.asarray(
        [
            0.5
            * (
                row["KG_d0.003_rise_cm-1"]
                + row["KM_d0.003_rise_cm-1"]
            )
            for row in metrics
        ],
        float,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "status": "extended_fixed_lattice_smearing_sweep_generated",
        "scope": (
            "zero plus six electronic smearings on one common static-lattice "
            "MLIP background and one common full-EPC reference constant"
        ),
        "lattice_control": {
            "condition": "fixed_static_lattice_reference",
            "same_short_range_MLIP_Dq_for_every_smearing": True,
            "thermal_lattice_or_TDEP_background_used": False,
            "numeric_lattice_temperature_K": None,
        },
        "integration": {
            "coarse_nk": args.coarse_nk,
            "patch_quadrature": "triangle_centroid",
            "patch_n": args.patch_n,
            "patch_halfwidth_fractional": args.patch_halfwidth,
            "finite_eta_eV": args.eta_eV,
            "reference_degauss_Ry": args.reference_degauss_Ry,
            "number_of_qpoints": len(qpoints),
            "number_of_finite_smearings": len(finite_smearings),
            "wall_time_seconds": elapsed,
            "new_DFT_used": False,
        },
        "smearings_degauss_Ry": [float(value) for value in smearings],
        "series_metrics": metrics,
        "checks": {
            "zero_curve_replay_max_abs_cm-1": zero_replay_error,
            "common_reference_constant_spread_cm-2": common_constant_spread,
            "all_frequencies_finite": bool(np.isfinite(frequencies).all()),
            "K_frequency_increases_monotonically_with_smearing": bool(
                np.all(np.diff(frequencies[:, 0]) > 0.0)
            ),
            "mean_d003_depth_decreases_monotonically_with_smearing": bool(
                np.all(np.diff(mean_d003) < 0.0)
            ),
            "overlap_with_previous_four_curve_sweep": overlap_error(
                args.overlap_reference, smearings, records, frequencies
            ),
        },
        "limitations": [
            (
                "the 0.005, 0.010, and 0.020 Ry curves are full-EPC model "
                "predictions without new direct DFPT points"
            ),
            (
                "this isolates electronic smearing on a static lattice and is "
                "not a finite-lattice-temperature spectrum"
            ),
        ],
        "outputs": {"csv": str(csv_path.resolve())},
    }
    summary_path = output / "extended_fixed_lattice_smearing_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
