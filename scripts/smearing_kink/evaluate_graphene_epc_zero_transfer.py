#!/usr/bin/env python3
"""Evaluate the stored full EPC vertex against zero-smearing graphene DFPT.

The response uses the same EPW Wannier vertex validated at finite smearing.  A
separate K anchor is allowed by the long-range model; no zero-q shape parameter
is fitted.  Uniform-mesh results are explicitly treated as convergence tests
for the later zero-temperature triangle/tetrahedron integration.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ablate_graphene_constant_vertex_bandsum import hamiltonian_grid  # noqa: E402
from diagnose_graphene_epw_wannier import read_hr  # noqa: E402
from extract_graphene_epw_dynamical_matrices import (  # noqa: E402
    RYDBERG_CM1,
    load_crystal,
    load_rdw,
    zone_centered_wigner,
)
from replay_graphene_epc_bandsum import (  # noqa: E402
    dynamical_matrix_and_mode,
    epc_wannier_grid,
    load_epmatwp,
    response_difference,
    signed_frequency,
)


def read_zero_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["group"] == "zero"]
    if len(rows) != 13:
        raise ValueError(f"expected 13 zero-smearing rows, found {len(rows)}")
    return rows


def qpoint(direction: str, distance: float) -> np.ndarray:
    if direction == "K":
        return np.asarray([1.0 / 3.0, 1.0 / 3.0, 0.0])
    if direction == "KG":
        value = (1.0 - distance) / 3.0
        return np.asarray([value, value, 0.0])
    if direction == "KM":
        return np.asarray(
            [(1.0 + distance) / 3.0, (1.0 - 2.0 * distance) / 3.0, 0.0]
        )
    raise ValueError(direction)


def shape_metrics(rows: list[dict], prediction: np.ndarray) -> dict:
    target = np.asarray([float(row["target_total_cm-1"]) for row in rows])
    lookup_target = {
        row["direction"] + f"_{float(row['distance']):.3f}": float(
            row["target_total_cm-1"]
        )
        for row in rows
    }
    lookup_prediction = {
        row["direction"] + f"_{float(row['distance']):.3f}": float(prediction[index])
        for index, row in enumerate(rows)
    }
    target_K = lookup_target["K_0.000"]
    prediction_K = lookup_prediction["K_0.000"]
    directions = {}
    for direction in ("KG", "KM"):
        target_inner = (lookup_target[f"{direction}_0.003"] - target_K) / 0.003
        prediction_inner = (
            lookup_prediction[f"{direction}_0.003"] - prediction_K
        ) / 0.003
        target_outer = (
            lookup_target[f"{direction}_0.019"]
            - lookup_target[f"{direction}_0.011"]
        ) / 0.008
        prediction_outer = (
            lookup_prediction[f"{direction}_0.019"]
            - lookup_prediction[f"{direction}_0.011"]
        ) / 0.008
        target_ratio = target_inner / target_outer
        prediction_ratio = prediction_inner / prediction_outer
        directions[direction] = {
            "target_d003_cusp_depth_cm-1": float(
                lookup_target[f"{direction}_0.003"] - target_K
            ),
            "prediction_d003_cusp_depth_cm-1": float(
                lookup_prediction[f"{direction}_0.003"] - prediction_K
            ),
            "d003_cusp_depth_relative_error": float(
                abs(prediction_inner - target_inner) / abs(target_inner)
            ),
            "target_inner_to_outer_slope_ratio": float(target_ratio),
            "prediction_inner_to_outer_slope_ratio": float(prediction_ratio),
            "slope_ratio_relative_error": float(
                abs(prediction_ratio - target_ratio) / abs(target_ratio)
            ),
        }
    return {
        "frequency_RMSE_cm-1": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "frequency_MAE_cm-1": float(np.mean(np.abs(prediction - target))),
        "frequency_max_abs_cm-1": float(np.max(np.abs(prediction - target))),
        "by_direction": directions,
        "maximum_d003_cusp_depth_relative_error": max(
            row["d003_cusp_depth_relative_error"] for row in directions.values()
        ),
        "maximum_slope_ratio_relative_error": max(
            row["slope_ratio_relative_error"] for row in directions.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=720)
    parser.add_argument("--target-fermi-eV", type=float, default=-1.7005)
    parser.add_argument("--reference-fermi-eV", type=float, default=-1.7187)
    parser.add_argument("--reference-degauss-Ry", type=float, default=0.020000)
    parser.add_argument("--eta-eV", type=float, default=0.005)
    parser.add_argument(
        "--hr",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility/E0_epw_matched"
            / "wannier_diagnostics/matched_k18q9_ex1_pifroz_hr.dat"
        ),
    )
    parser.add_argument(
        "--epwdata",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility/E0_epw_matched"
            / "k18_q9_ex1_pifroz/restart_small/epwdata.fmt"
        ),
    )
    parser.add_argument(
        "--crystal",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility/E0_epw_matched"
            / "k18_q9_ex1_pifroz/restart_small/crystal.fmt"
        ),
    )
    parser.add_argument(
        "--epmatwp",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E9_epc_bandsum_assets/graphene.epmatwp"
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
            / "E10_epc_zero_transfer"
        ),
    )
    args = parser.parse_args()
    if args.nk % 3 != 0:
        raise ValueError("nk must be divisible by 3")

    rows = read_zero_rows(args.zero_data)
    qpoints = np.asarray(
        [qpoint(row["direction"], float(row["distance"])) for row in rows]
    )
    crystal = load_crystal(args.crystal, natoms=2)
    rdw, dimensions = load_rdw(args.epwdata)
    vectors_k, degeneracies_k, _ = zone_centered_wigner(
        crystal["at"], (18, 18, 1)
    )
    vectors_q, degeneracies_q, _ = zone_centered_wigner(
        crystal["at"], (9, 9, 1)
    )
    translations, hamiltonian_matrices = read_hr(args.hr)
    if not np.array_equal(translations, vectors_k):
        raise ValueError("Hamiltonian and EPC electronic WS orders differ")
    epmatwp = load_epmatwp(
        args.epmatwp,
        dimensions["nbndsub"],
        dimensions["nrr_k"],
        dimensions["nmodes"],
        dimensions["nrr_g"],
    )

    initial_h = hamiltonian_grid(
        translations, hamiltonian_matrices, args.nk, np.zeros(3)
    )
    initial_energy, initial_vectors = np.linalg.eigh(initial_h)
    correction_Ry2 = np.empty(len(rows), float)
    for index, q in enumerate(qpoints):
        _, _, physical_mode = dynamical_matrix_and_mode(
            rdw,
            q,
            vectors_q,
            degeneracies_q,
            crystal["masses_au"],
            crystal["atom_types"],
        )
        vertex = epc_wannier_grid(
            epmatwp,
            q,
            vectors_q,
            degeneracies_q,
            physical_mode,
            vectors_k,
            degeneracies_k,
            args.nk,
        )
        final_h = hamiltonian_grid(
            translations, hamiltonian_matrices, args.nk, q
        )
        final_energy, final_vectors = np.linalg.eigh(final_h)
        correction_Ry2[index] = response_difference(
            initial_energy,
            initial_vectors,
            final_energy,
            final_vectors,
            vertex,
            np.asarray([0.0]),
            args.reference_degauss_Ry,
            args.target_fermi_eV,
            args.eta_eV,
            reference_fermi_eV=args.reference_fermi_eV,
        )[0]

    correction_cm2 = correction_Ry2 * RYDBERG_CM1**2
    target_correction = np.asarray(
        [float(row["target_delta_lambda_cm-2"]) for row in rows]
    )
    background = np.asarray([float(row["background_cm-1"]) for row in rows])
    K_index = next(index for index, row in enumerate(rows) if row["direction"] == "K")
    raw_prediction = signed_frequency(background, correction_cm2)
    anchored_correction = (
        target_correction[K_index] + correction_cm2 - correction_cm2[K_index]
    )
    anchored_prediction = signed_frequency(background, anchored_correction)
    raw_metrics = shape_metrics(rows, raw_prediction)
    anchored_metrics = shape_metrics(rows, anchored_prediction)
    converged_shape_gate = (
        anchored_metrics["maximum_d003_cusp_depth_relative_error"] < 0.20
        and anchored_metrics["maximum_slope_ratio_relative_error"] < 0.20
    )
    status = (
        "uniform_grid_zero_shape_gate_passed_requires_nk_convergence"
        if converged_shape_gate
        else "uniform_grid_zero_shape_not_yet_converged_or_failed"
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "epc_zero_transfer_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(rows[0]) + [
            "epc_correction_cm-2",
            "raw_prediction_cm-1",
            "K_anchored_prediction_cm-1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            current = dict(row)
            current.update(
                {
                    "epc_correction_cm-2": correction_cm2[index],
                    "raw_prediction_cm-1": raw_prediction[index],
                    "K_anchored_prediction_cm-1": anchored_prediction[index],
                }
            )
            writer.writerow(current)

    figure, axis = plt.subplots(figsize=(7.0, 4.3))
    target_K = float(rows[K_index]["target_total_cm-1"])
    axis.scatter([0.0], [target_K], color="black", s=28, zorder=5)
    colors = {"KG": "#2676b8", "KM": "#7a3db8"}
    for direction in ("KG", "KM"):
        indices = [index for index, row in enumerate(rows) if row["direction"] == direction]
        distance = np.asarray([float(rows[index]["distance"]) for index in indices])
        target_frequency = np.asarray(
            [float(rows[index]["target_total_cm-1"]) for index in indices]
        )
        axis.plot(
            distance,
            target_frequency,
            "o",
            color=colors[direction],
            ms=4,
            label=f"DFPT {direction}",
        )
        axis.plot(
            distance,
            anchored_prediction[indices],
            "-",
            color=colors[direction],
            lw=1.8,
            label=f"EPC band-sum {direction}",
        )
    axis.set_xlabel("distance from K")
    axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    axis.set_title("Zero-smearing EPC transfer (separate K anchor)")
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.79, 0.5),
        frameon=False,
    )
    figure.subplots_adjust(left=0.12, right=0.77, bottom=0.14, top=0.89)
    figure.savefig(
        output / "epc_zero_transfer_comparison.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "status": status,
        "scope": (
            "full stored A-prime EPC vertex, target zero-temperature occupations, "
            "T_high=0.02 Ry reference, separate K anchor, no zero-shape fit and no new DFT"
        ),
        "integration": "uniform nk x nk mesh with exact step occupations",
        "nk": args.nk,
        "target_fermi_eV": args.target_fermi_eV,
        "reference_fermi_eV": args.reference_fermi_eV,
        "reference_degauss_Ry": args.reference_degauss_Ry,
        "numerical_broadening_eV": args.eta_eV,
        "raw_absolute_metrics": raw_metrics,
        "separate_K_anchor_metrics": anchored_metrics,
        "checks": {
            "anchored_max_d003_cusp_depth_relative_error_lt_0p20": anchored_metrics[
                "maximum_d003_cusp_depth_relative_error"
            ]
            < 0.20,
            "anchored_max_slope_ratio_relative_error_lt_0p20": anchored_metrics[
                "maximum_slope_ratio_relative_error"
            ]
            < 0.20,
        },
        "limitations": [
            "an exact zero-temperature occupation step converges slowly on uniform meshes; compare multiple nk before interpreting failure",
            "the zero DFPT reference uses optimized tetrahedra, so production transfer should use a 2D triangle/tetrahedron or adaptive valley integral",
            "the outer zero reference shape is k192 while K and d=0.003 use k288",
        ],
    }
    (output / "epc_zero_transfer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
