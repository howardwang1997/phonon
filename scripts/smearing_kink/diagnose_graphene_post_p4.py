#!/usr/bin/env python3
"""Attribute the failed graphene P4 force and q-space errors."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from graphene_fd_p4_common import (  # noqa: E402
    atomic_json,
    geometry_sha256,
    harmonic_energy_forces,
    line_metrics,
    sha256,
)


AXES = ("x", "y", "z")


def metrics(error: np.ndarray) -> dict[str, float]:
    values = np.asarray(error, float).ravel()
    return {
        "RMSE_meV_A": float(np.sqrt(np.mean(values * values)) * 1000.0),
        "MAE_meV_A": float(np.mean(np.abs(values)) * 1000.0),
        "max_abs_meV_A": float(np.max(np.abs(values)) * 1000.0),
    }


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    if len(x) < 2 or np.std(x) < 1.0e-15 or np.std(y) < 1.0e-15:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def scalar_projection(reference: np.ndarray, short: np.ndarray, long: np.ndarray):
    residual = np.asarray(reference, float) - np.asarray(short, float)
    vector = np.asarray(long, float)
    denominator = float(np.sum(vector * vector))
    if denominator < 1.0e-20:
        return math.nan, math.nan
    alpha = float(np.sum(vector * residual) / denominator)
    residual_norm = float(np.linalg.norm(residual.ravel()))
    vector_norm = float(np.linalg.norm(vector.ravel()))
    cosine = (
        float(np.dot(vector.ravel(), residual.ravel()) / (vector_norm * residual_norm))
        if vector_norm > 0 and residual_norm > 0
        else math.nan
    )
    return alpha, cosine


def load_operator_reference(path: Path):
    with np.load(path, allow_pickle=False) as data:
        raw_reference = np.asarray(data["reference_positions"], float)
        cell = np.asarray(data["cell"], float)
        mapping = np.asarray(data["atom_mapping"], int)
        force_constants = np.asarray(data["delta_fc_full"], float)
    return raw_reference[mapping], cell, force_constants[mapping][:, mapping]


def force_diagnostics(args, output_dir: Path) -> dict:
    structures = read(args.labels, ":")
    with np.load(args.predictions, allow_pickle=False) as data:
        keys = [str(value) for value in data["keys"]]
        geometry_hashes = [str(value) for value in data["geometry_sha256"]]
        short = np.asarray(data["predicted_short_forces_eV_A"], float)
        long = np.asarray(data["predicted_long_range_forces_eV_A"], float)
        total = np.asarray(data["predicted_total_forces_eV_A"], float)
    if len(structures) != len(keys):
        raise ValueError("label/prediction structure counts differ")

    reference_positions, operator_cell, operator_fc = load_operator_reference(args.operator)
    observed_keys, observed_hashes, reference = [], [], []
    component_rows, structure_rows = [], []
    displacement_all, abs_error_all = [], []
    reference_magnitude_all, long_magnitude_all = [], []

    for index, structure in enumerate(structures):
        key = str(structure.info["p4_key"])
        observed_keys.append(key)
        observed_hashes.append(
            geometry_sha256(structure.numbers, structure.cell.array, structure.positions)
        )
        reference_force = np.asarray(structure.arrays["REF_forces"], float)
        reference.append(reference_force)
        if not np.allclose(structure.cell.array, operator_cell, rtol=0.0, atol=2.0e-5):
            raise ValueError(f"operator/label cell mismatch for {key}")
        if len(structure) != len(reference_positions):
            raise ValueError(f"operator/label atom count mismatch for {key}")

        _, reconstructed_long, displacement = harmonic_energy_forces(
            operator_fc,
            reference_positions,
            operator_cell,
            structure.positions,
        )
        reconstruction_error = float(np.max(np.abs(reconstructed_long - long[index])))
        error_total = total[index] - reference_force
        error_short = short[index] - reference_force
        alpha, cosine = scalar_projection(reference_force, short[index], long[index])
        oracle = short[index] + alpha * long[index] if np.isfinite(alpha) else short[index]
        pair_distances = structure.get_all_distances(mic=True)
        pair_distances[pair_distances < 1.0e-12] = np.inf
        max_component = np.unravel_index(np.argmax(np.abs(error_total)), error_total.shape)

        row = {
            "key": key,
            "trajectory_seed": int(structure.info["trajectory_seed"]),
            "snapshot_index": int(structure.info["snapshot_index"]),
            "displacement_RMS_A": float(np.sqrt(np.mean(displacement * displacement))),
            "displacement_max_A": float(np.max(np.linalg.norm(displacement, axis=1))),
            "min_pair_distance_A": float(np.min(pair_distances)),
            "reference_force_RMS_meV_A": float(
                np.sqrt(np.mean(reference_force * reference_force)) * 1000.0
            ),
            "short_force_RMS_meV_A": float(np.sqrt(np.mean(short[index] ** 2)) * 1000.0),
            "long_force_RMS_meV_A": float(np.sqrt(np.mean(long[index] ** 2)) * 1000.0),
            "total_force_RMS_meV_A": float(np.sqrt(np.mean(total[index] ** 2)) * 1000.0),
            "short_only_error_RMSE_meV_A": metrics(error_short)["RMSE_meV_A"],
            "total_error_RMSE_meV_A": metrics(error_total)["RMSE_meV_A"],
            "total_error_max_abs_meV_A": metrics(error_total)["max_abs_meV_A"],
            "oracle_alpha": alpha,
            "long_vs_required_residual_cosine": cosine,
            "oracle_scaled_error_RMSE_meV_A": metrics(oracle - reference_force)[
                "RMSE_meV_A"
            ],
            "max_error_atom": int(max_component[0]),
            "max_error_axis": AXES[int(max_component[1])],
            "operator_force_replay_max_abs_eV_A": reconstruction_error,
            "reference_net_force_meV_A": float(
                np.linalg.norm(reference_force.sum(axis=0)) * 1000.0
            ),
            "prediction_net_force_meV_A": float(
                np.linalg.norm(total[index].sum(axis=0)) * 1000.0
            ),
        }
        structure_rows.append(row)

        atom_reference_magnitude = np.linalg.norm(reference_force, axis=1)
        atom_long_magnitude = np.linalg.norm(long[index], axis=1)
        atom_displacement = np.linalg.norm(displacement, axis=1)
        for atom in range(len(structure)):
            for axis in range(3):
                component_rows.append(
                    {
                        "key": key,
                        "trajectory_seed": row["trajectory_seed"],
                        "snapshot_index": row["snapshot_index"],
                        "atom": atom,
                        "axis": AXES[axis],
                        "displacement_A": float(displacement[atom, axis]),
                        "displacement_norm_A": float(atom_displacement[atom]),
                        "reference_force_eV_A": float(reference_force[atom, axis]),
                        "short_force_eV_A": float(short[index, atom, axis]),
                        "long_force_eV_A": float(long[index, atom, axis]),
                        "total_force_eV_A": float(total[index, atom, axis]),
                        "short_only_error_meV_A": float(error_short[atom, axis] * 1000.0),
                        "total_error_meV_A": float(error_total[atom, axis] * 1000.0),
                        "reference_force_norm_eV_A": float(atom_reference_magnitude[atom]),
                        "long_force_norm_eV_A": float(atom_long_magnitude[atom]),
                    }
                )
        displacement_all.extend(np.repeat(atom_displacement, 3))
        abs_error_all.extend(np.abs(error_total).ravel())
        reference_magnitude_all.extend(np.repeat(atom_reference_magnitude, 3))
        long_magnitude_all.extend(np.repeat(atom_long_magnitude, 3))

    if keys != observed_keys:
        raise ValueError("prediction and label key order differs")
    hash_mismatches = [
        key
        for key, expected, observed in zip(keys, geometry_hashes, observed_hashes, strict=True)
        if expected != observed
    ]
    if hash_mismatches:
        raise ValueError(f"geometry hash mismatch: {hash_mismatches}")

    reference_array = np.asarray(reference, float)
    alpha_global, cosine_global = scalar_projection(reference_array, short, long)
    oracle_global = short + alpha_global * long
    primary_mask = np.asarray(
        [
            int(structure.info["snapshot_index"]) in (3, 27, 51, 75, 99)
            for structure in structures
        ],
        bool,
    )
    sorted_rows = sorted(
        structure_rows, key=lambda item: item["total_error_max_abs_meV_A"], reverse=True
    )
    result = {
        "status": "complete",
        "integrity": {
            "n_structures": len(structures),
            "n_atoms_per_structure": len(structures[0]),
            "keys_match": True,
            "geometry_hashes_match": True,
            "operator_force_replay_max_abs_eV_A": max(
                row["operator_force_replay_max_abs_eV_A"] for row in structure_rows
            ),
        },
        "all60": {
            "frozen_total": metrics(total - reference_array),
            "short_only_against_total_DFT": metrics(short - reference_array),
            "global_oracle_scaled_long_range": metrics(oracle_global - reference_array),
        },
        "primary15": {
            "frozen_total": metrics(total[primary_mask] - reference_array[primary_mask]),
            "short_only_against_total_DFT": metrics(
                short[primary_mask] - reference_array[primary_mask]
            ),
        },
        "long_range_attribution": {
            "global_oracle_alpha": alpha_global,
            "global_long_vs_required_residual_cosine": cosine_global,
            "n_structures_oracle_alpha_negative": int(
                sum(row["oracle_alpha"] < 0 for row in structure_rows)
            ),
            "n_structures_frozen_total_better_than_short_only": int(
                sum(
                    row["total_error_RMSE_meV_A"]
                    < row["short_only_error_RMSE_meV_A"]
                    for row in structure_rows
                )
            ),
        },
        "correlations": {
            "abs_component_error_vs_atom_displacement": correlation(
                abs_error_all, displacement_all
            ),
            "abs_component_error_vs_reference_force_norm": correlation(
                abs_error_all, reference_magnitude_all
            ),
            "abs_component_error_vs_long_force_norm": correlation(
                abs_error_all, long_magnitude_all
            ),
        },
        "threshold_counts": {
            "components_abs_error_gt_250_meV_A": int(
                np.count_nonzero(np.abs(total - reference_array) * 1000.0 > 250.0)
            ),
            "structures_max_abs_error_gt_250_meV_A": int(
                sum(row["total_error_max_abs_meV_A"] > 250.0 for row in structure_rows)
            ),
        },
        "worst_structures": sorted_rows[:10],
        "fixed_outlier_records": {
            key: next((row for row in structure_rows if row["key"] == key), None)
            for key in ("seed0:snapshot099", "seed0:snapshot063")
        },
        "inputs": {
            "labels": {"path": str(args.labels), "sha256": sha256(args.labels)},
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256(args.predictions),
            },
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
        },
    }

    for filename, rows in (
        ("force_component_records.csv", component_rows),
        ("force_structure_summary.csv", structure_rows),
    ):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(output_dir / "force_diagnostics.json", result)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    ref_flat = reference_array.ravel() * 1000.0
    axes[0].scatter(ref_flat, short.ravel() * 1000.0, s=5, alpha=0.22, label="short only")
    axes[0].scatter(ref_flat, total.ravel() * 1000.0, s=5, alpha=0.22, label="frozen total")
    limits = [float(min(ref_flat.min(), total.min() * 1000.0)), float(max(ref_flat.max(), total.max() * 1000.0))]
    axes[0].plot(limits, limits, color="black", linewidth=1)
    axes[0].set_xlabel("DFT force component (meV/Å)")
    axes[0].set_ylabel("prediction (meV/Å)")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2)

    order = np.arange(len(sorted_rows))
    axes[1].bar(order, [row["total_error_RMSE_meV_A"] for row in sorted_rows], color="#0072B2")
    axes[1].axhline(50.0, color="#D55E00", linestyle="--", linewidth=1)
    axes[1].set_xlabel("structures ordered by max component error")
    axes[1].set_ylabel("force RMSE (meV/Å)")

    axes[2].scatter(
        [row["long_force_RMS_meV_A"] for row in structure_rows],
        [row["total_error_RMSE_meV_A"] for row in structure_rows],
        c=[row["displacement_RMS_A"] for row in structure_rows],
        cmap="viridis",
        s=28,
    )
    axes[2].set_xlabel("long-range force RMS (meV/Å)")
    axes[2].set_ylabel("total error RMSE (meV/Å)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "force_error_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return result


def qspace_diagnostics(args, output_dir: Path) -> dict:
    with args.qspace.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 19:
        raise ValueError(f"expected 19 q-space rows, found {len(rows)}")
    output_rows = []
    summary = {"status": "complete", "static": {}, "finite_lattice": {}}
    for region in ("G", "K"):
        selected = [row for row in rows if row["region"] == region]
        selected.sort(key=lambda row: float(row["t_GK"]))
        t = np.asarray([float(row["t_GK"]) for row in selected])
        static_prediction = np.asarray(
            [float(row["static_frozen_prediction_cm-1"]) for row in selected]
        )
        static_target = np.asarray(
            [float(row["direct_DFPT_target_cm-1"]) for row in selected]
        )
        finite_prediction = np.asarray(
            [float(row["finite_T_frozen_prediction_cm-1"]) for row in selected]
        )
        finite_target = np.asarray(
            [float(row["DFT_TDEP_target_cm-1"]) for row in selected]
        )
        summary["static"][region] = line_metrics(
            region, t, static_prediction, static_target
        )
        summary["finite_lattice"][region] = line_metrics(
            region, t, finite_prediction, finite_target
        )
        for index, row in enumerate(selected):
            output_rows.append(
                {
                    "region": region,
                    "t_GK": t[index],
                    "static_prediction_cm-1": static_prediction[index],
                    "static_target_cm-1": static_target[index],
                    "static_residual_cm-1": static_prediction[index] - static_target[index],
                    "finite_prediction_cm-1": finite_prediction[index],
                    "finite_target_cm-1": finite_target[index],
                    "finite_residual_cm-1": finite_prediction[index] - finite_target[index],
                }
            )
    summary["dominant_point_failure"] = (
        "static_electronic_response"
        if sum(summary["static"][r]["line_MAE_cm-1"] for r in ("G", "K"))
        > sum(summary["finite_lattice"][r]["line_MAE_cm-1"] for r in ("G", "K"))
        else "finite_lattice_response"
    )
    summary["input"] = {"path": str(args.qspace), "sha256": sha256(args.qspace)}
    with (output_dir / "qspace_residuals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    atomic_json(output_dir / "qspace_diagnostics.json", summary)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey="row")
    for column, channel in enumerate(("static", "finite")):
        for row_index, region in enumerate(("G", "K")):
            axis = axes[row_index, column]
            selected = [row for row in output_rows if row["region"] == region]
            t = [float(row["t_GK"]) for row in selected]
            prefix = "static" if channel == "static" else "finite"
            prediction = [float(row[f"{prefix}_prediction_cm-1"]) for row in selected]
            target = [float(row[f"{prefix}_target_cm-1"]) for row in selected]
            axis.plot(t, target, "o-", color="#222222", label="DFT target")
            axis.plot(t, prediction, "s--", color="#D55E00", label="frozen prediction")
            axis.set_title(f"{region} · {'static' if channel == 'static' else 'finite lattice'}")
            axis.set_xlabel("t along q=tK")
            axis.set_ylabel("top optical frequency (cm$^{-1}$)")
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.23), ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "qspace_residuals.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--qspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    force = force_diagnostics(args, args.output_dir)
    qspace = qspace_diagnostics(args, args.output_dir)
    combined = {
        "status": "complete",
        "force": force,
        "qspace": qspace,
        "implementation_integrity_passes": bool(
            force["integrity"]["keys_match"]
            and force["integrity"]["geometry_hashes_match"]
            and force["integrity"]["operator_force_replay_max_abs_eV_A"] < 1.0e-10
        ),
    }
    atomic_json(args.output_dir / "diagnostic_summary.json", combined)
    print(json.dumps(combined, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
