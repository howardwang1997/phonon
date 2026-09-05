#!/usr/bin/env python3
"""Transfer the finite-smearing constant-vertex band sum to zero smearing.

The A' matrix projector and K anchor remain separate.  The sigma_z response
amplitude is read from the finite-smearing ablation and is not refitted to the
zero-smearing q-shape.  This is a development diagnostic on existing data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ablate_graphene_constant_vertex_bandsum import (  # noqa: E402
    VERTICES,
    hamiltonian_grid,
    response_for_q,
)
from diagnose_graphene_epw_wannier import read_hr  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_zero_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["group"] == "zero"]
    if len(rows) != 13:
        raise ValueError(f"expected 13 zero-smearing rows, found {len(rows)}")
    return rows


def phonon_qpoint(direction: str, distance: float) -> np.ndarray:
    if direction == "K":
        h = k = 1.0 / 3.0
    elif direction == "KG":
        h = k = (1.0 - distance) / 3.0
    elif direction == "KM":
        h = (1.0 + distance) / 3.0
        k = (1.0 - 2.0 * distance) / 3.0
    else:
        raise ValueError(direction)
    return np.asarray([h, k, 0.0])


def zero_response(
    translations: np.ndarray,
    matrices: np.ndarray,
    nk: int,
    qpoints: np.ndarray,
    fermi_eV: float,
) -> np.ndarray:
    initial_h = hamiltonian_grid(
        translations, matrices, nk, np.zeros(3, float)
    )
    initial_energy, initial_vectors = np.linalg.eigh(initial_h)
    values = []
    vertex = {"sigma_z": VERTICES["sigma_z"]}
    for qpoint in qpoints:
        final_h = hamiltonian_grid(translations, matrices, nk, qpoint)
        final_energy, final_vectors = np.linalg.eigh(final_h)
        response = response_for_q(
            initial_energy,
            initial_vectors,
            final_energy,
            final_vectors,
            vertex,
            fermi_eV,
            np.asarray([0.0]),
        )["sigma_z"][0]
        values.append(float(response))
    return np.asarray(values, float)


def shape_metrics(rows: list[dict], prediction: np.ndarray) -> dict:
    target = np.asarray([float(row["target_total_cm-1"]) for row in rows])
    result = {
        "RMSE_cm-1": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "MAE_cm-1": float(np.mean(np.abs(prediction - target))),
        "max_abs_cm-1": float(np.max(np.abs(prediction - target))),
    }
    target_lookup = {
        row["direction"] + f"_{float(row['distance']):.3f}": float(
            row["target_total_cm-1"]
        )
        for row in rows
    }
    predicted_lookup = {
        row["direction"] + f"_{float(row['distance']):.3f}": float(prediction[index])
        for index, row in enumerate(rows)
    }
    target_K = target_lookup["K_0.000"]
    predicted_K = predicted_lookup["K_0.000"]
    by_direction = {}
    for direction in ("KG", "KM"):
        target_inner = (target_lookup[f"{direction}_0.003"] - target_K) / 0.003
        predicted_inner = (
            predicted_lookup[f"{direction}_0.003"] - predicted_K
        ) / 0.003
        target_outer = (
            target_lookup[f"{direction}_0.019"]
            - target_lookup[f"{direction}_0.011"]
        ) / 0.008
        predicted_outer = (
            predicted_lookup[f"{direction}_0.019"]
            - predicted_lookup[f"{direction}_0.011"]
        ) / 0.008
        target_ratio = target_inner / target_outer
        predicted_ratio = predicted_inner / predicted_outer
        by_direction[direction] = {
            "target_d003_cusp_depth_cm-1": float(
                target_lookup[f"{direction}_0.003"] - target_K
            ),
            "predicted_d003_cusp_depth_cm-1": float(
                predicted_lookup[f"{direction}_0.003"] - predicted_K
            ),
            "d003_cusp_depth_relative_error": float(
                abs(predicted_inner - target_inner) / abs(target_inner)
            ),
            "target_inner_to_outer_slope_ratio": float(target_ratio),
            "predicted_inner_to_outer_slope_ratio": float(predicted_ratio),
            "slope_ratio_relative_error": float(
                abs(predicted_ratio - target_ratio) / abs(target_ratio)
            ),
        }
    result["by_direction"] = by_direction
    result["maximum_d003_cusp_depth_relative_error"] = max(
        value["d003_cusp_depth_relative_error"] for value in by_direction.values()
    )
    result["maximum_slope_ratio_relative_error"] = max(
        value["slope_ratio_relative_error"] for value in by_direction.values()
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=288)
    parser.add_argument("--fermi-eV", type=float, default=-1.7187)
    parser.add_argument(
        "--hr",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E0_epw_matched/wannier_diagnostics/matched_k18q9_ex1_pifroz_hr.dat"
        ),
    )
    parser.add_argument(
        "--finite-summary",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E7_constant_vertex_bandsum_nk288/constant_vertex_bandsum_summary.json"
        ),
    )
    parser.add_argument(
        "--zero-data",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E4_joint_zero_finite_rank1/joint_zero_finite_rank1_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E8_bandsum_zero_transfer"
        ),
    )
    args = parser.parse_args()
    if args.nk % 3 != 0:
        raise ValueError("nk must be divisible by 3")

    finite = json.loads(args.finite_summary.read_text(encoding="utf-8"))
    amplitude = float(
        finite["vertices"]["sigma_z"]["global_scale_cm-2_per_response_unit"]
    )
    rows = read_zero_rows(args.zero_data)
    qpoints = np.asarray(
        [phonon_qpoint(row["direction"], float(row["distance"])) for row in rows]
    )
    translations, matrices = read_hr(args.hr)
    response = zero_response(
        translations, matrices, args.nk, qpoints, args.fermi_eV
    )
    K_index = next(index for index, row in enumerate(rows) if row["direction"] == "K")
    centered_response = response - response[K_index]
    target_delta = np.asarray(
        [float(row["target_delta_lambda_cm-2"]) for row in rows]
    )
    background = np.asarray([float(row["background_cm-1"]) for row in rows])
    predicted_delta = target_delta[K_index] + amplitude * centered_response
    prediction = np.sqrt(np.maximum(background**2 + predicted_delta, 0.0))
    metrics = shape_metrics(rows, prediction)
    status = (
        "zero_transfer_shape_gate_passed"
        if metrics["maximum_d003_cusp_depth_relative_error"] < 0.20
        and metrics["maximum_slope_ratio_relative_error"] < 0.20
        else "zero_transfer_shape_gate_failed"
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "bandsum_zero_transfer_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(rows[0]) + [
            "bandsum_response_eV-1",
            "bandsum_centered_response_eV-1",
            "prediction_cm-1",
            "signed_error_cm-1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            output_row = dict(row)
            output_row.update(
                {
                    "bandsum_response_eV-1": response[index],
                    "bandsum_centered_response_eV-1": centered_response[index],
                    "prediction_cm-1": prediction[index],
                    "signed_error_cm-1": prediction[index]
                    - float(row["target_total_cm-1"]),
                }
            )
            writer.writerow(output_row)

    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    target_K = float(rows[K_index]["target_total_cm-1"])
    axis.scatter([0.0], [target_K], color="black", s=32, zorder=4, label="DFPT reference")
    axis.scatter([0.0], [prediction[K_index]], color="#d55e00", s=20, zorder=5, label="band-sum transfer")
    colors = {"KG": "#2676b8", "KM": "#7a3db8"}
    for direction in ("KG", "KM"):
        indices = [index for index, row in enumerate(rows) if row["direction"] == direction]
        distance = np.asarray([float(rows[index]["distance"]) for index in indices])
        target = np.asarray([float(rows[index]["target_total_cm-1"]) for index in indices])
        axis.plot(distance, target, "o", color=colors[direction], ms=4, label=f"DFPT {direction}")
        axis.plot(distance, prediction[indices], "-", color=colors[direction], lw=1.8, label=f"band-sum {direction}")
    axis.set_xlabel("distance from K")
    axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    axis.set_title("Zero-smearing transfer from finite-smearing band sum")
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        bbox_to_anchor=(0.78, 0.5),
        frameon=False,
    )
    figure.subplots_adjust(left=0.12, right=0.76, bottom=0.14, top=0.89)
    figure.savefig(
        output / "bandsum_zero_transfer_comparison.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "status": status,
        "scope": (
            "sigma_z constant-vertex band-sum amplitude fixed on opened finite-smearing "
            "EPW curves, separate exact K anchor, no zero-shape refit and no new DFT"
        ),
        "nk": args.nk,
        "fermi_eV": args.fermi_eV,
        "finite_fitted_amplitude_cm-2_per_response_unit": amplitude,
        "metrics": metrics,
        "checks": {
            "maximum_d003_cusp_depth_relative_error_lt_0p20": metrics[
                "maximum_d003_cusp_depth_relative_error"
            ]
            < 0.20,
            "maximum_slope_ratio_relative_error_lt_0p20": metrics[
                "maximum_slope_ratio_relative_error"
            ]
            < 0.20,
        },
        "limitations": [
            "zero-temperature uniform-grid integration must be checked against nk convergence",
            "sigma_z is a constant vertex in the present Wannier gauge, not a reconstructed EPW A-prime vertex",
            "the zero reference mixes a k288 K/d=0.003 anchor with k192 outer shape and remains numerically uncertain",
        ],
        "inputs": {
            "hr": {"path": str(args.hr), "sha256": sha256(args.hr)},
            "finite_summary": {
                "path": str(args.finite_summary),
                "sha256": sha256(args.finite_summary),
            },
            "zero_data": {
                "path": str(args.zero_data),
                "sha256": sha256(args.zero_data),
            },
        },
    }
    atomic_json(output / "bandsum_zero_transfer_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
