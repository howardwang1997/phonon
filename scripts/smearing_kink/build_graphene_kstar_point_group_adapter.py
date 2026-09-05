#!/usr/bin/env python3
"""Validate the full graphene C3/C6 K-star matrix representation.

The archived long-range q6 operators provide an exact matrix-level reference
at K and K'.  An arbitrary off-centre K-patch point is also expanded to its
full two-dimensional orbit to verify q geometry.  No new electronic-structure
calculation is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spglib

from kstar_equivariant import (
    K_REDUCED,
    PointGroupOrbitAdapter,
    hermitian,
    periodic_delta,
)


ROOT = Path(__file__).resolve().parents[2]
TEMPERATURES = (300, 450, 600)
THRESHOLDS = {
    "long_range_point_group_relative_error": 1.0e-12,
    "long_range_point_group_max_abs": 1.0e-18,
    "shared_projector_duplicate_orbit_max_abs": 1.0e-12,
    "shared_projector_time_reversal_max_abs": 1.0e-12,
    "generated_hermitian_max_abs": 1.0e-12,
}


def periodic_index(qpoints: np.ndarray, target: np.ndarray) -> int:
    distance = np.linalg.norm(
        np.asarray([periodic_delta(qpoint, target) for qpoint in qpoints]), axis=1
    )
    index = int(np.argmin(distance))
    if distance[index] > 1.0e-10:
        raise ValueError(f"q point {target} is absent")
    return index


def qkey(qpoint: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.round(np.mod(qpoint, 1.0), 10).tolist())


def grouped_spread(qpoints: np.ndarray, matrices: np.ndarray) -> float:
    groups: dict[tuple[float, float, float], list[np.ndarray]] = {}
    for qpoint, matrix in zip(qpoints, matrices):
        groups.setdefault(qkey(qpoint), []).append(matrix)
    maximum = 0.0
    for group in groups.values():
        mean = np.mean(group, axis=0)
        maximum = max(
            maximum,
            max(float(np.max(np.abs(matrix - mean))) for matrix in group),
        )
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E0_epw_matched/k18_q9_ex1_pifroz/operator_q6_450K"
        ),
    )
    parser.add_argument(
        "--rank1-adapter",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E23_kstar_equivariant_adapter/graphene_kstar_rank1_adapter.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E28_kstar_point_group_adapter"
        ),
    )
    args = parser.parse_args()

    bare_path = args.input_dir / "matdyn/epw_q6_dynamical_matrices.npz"
    with np.load(bare_path, allow_pickle=False) as payload:
        qpoints = np.asarray(payload["qpoints_crystal"], float)
        bare = np.asarray(payload["dynamical_matrices"], complex)
    with np.load(args.rank1_adapter, allow_pickle=False) as payload:
        projector_k = np.asarray(payload["projector_K"], complex)

    # These positions are the Fourier-gauge basis used by q6.dyn.  They differ
    # by an atom/origin convention from the MLIP supercell input; mixing the
    # two conventions causes an order-one matrix symmetry error.
    lattice = np.asarray(
        [
            [2.46, 0.0, 0.0],
            [-1.23, 2.46 * np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 15.0],
        ]
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]])
    symmetry = spglib.get_symmetry((lattice, positions, [6, 6]), symprec=1.0e-8)
    if symmetry is None:
        raise RuntimeError("spglib did not find graphene symmetries")
    adapter = PointGroupOrbitAdapter(
        lattice,
        positions,
        symmetry["rotations"],
        symmetry["translations"],
    )
    index_k = periodic_index(qpoints, K_REDUCED)

    records = []
    relative_errors = []
    absolute_errors = []
    for temperature in TEMPERATURES:
        with np.load(
            args.input_dir / f"cartesian/T{temperature}_operator.npz",
            allow_pickle=False,
        ) as payload:
            correction = np.asarray(payload["delta_dynamical_q"], complex)
            smearing = float(payload["degauss_Ry"])
        orbit_q, orbit_matrix = adapter.orbit(correction[index_k], K_REDUCED)
        operation_errors = []
        for qpoint, generated in zip(orbit_q, orbit_matrix):
            target = correction[periodic_index(qpoints, qpoint)]
            absolute = float(np.max(np.abs(generated - target)))
            relative = float(
                np.linalg.norm(generated - target)
                / max(np.linalg.norm(target), 1.0e-30)
            )
            absolute_errors.append(absolute)
            relative_errors.append(relative)
            operation_errors.append(relative)
        records.append(
            {
                "temperature_K": temperature,
                "smearing_degauss_Ry": smearing,
                "maximum_operation_relative_error": max(operation_errors),
            }
        )

    projector_q, projector_orbit = adapter.orbit(projector_k, K_REDUCED)
    projector_spread = grouped_spread(projector_q, projector_orbit)
    projector_by_q = {}
    for qpoint, matrix in zip(projector_q, projector_orbit):
        projector_by_q.setdefault(qkey(qpoint), []).append(matrix)
    projector_k_mean = hermitian(np.mean(projector_by_q[qkey(K_REDUCED)], axis=0))
    k_prime = np.mod(-K_REDUCED, 1.0)
    projector_k_prime_mean = hermitian(
        np.mean(projector_by_q[qkey(k_prime)], axis=0)
    )
    projector_tr = float(
        np.max(np.abs(projector_k_mean - projector_k_prime_mean.conj()))
    )
    hermitian_error = float(
        np.max(
            np.abs(
                projector_orbit
                - np.swapaxes(projector_orbit.conj(), -1, -2)
            )
        )
    )

    # A generic two-dimensional displacement has twelve distinct in-plane
    # images; horizontal-mirror partners duplicate the matrices/q points.
    offcenter = K_REDUCED + np.asarray([0.012, -0.007, 0.0])
    offcenter_q, _ = adapter.orbit(projector_k, offcenter)
    unique_offcenter = np.unique(np.round(offcenter_q, 10), axis=0)
    offcenter_unfolded = np.asarray(
        [
            offcenter @ np.linalg.inv(rotation)
            for rotation in symmetry["rotations"]
        ]
    )
    unique_offcenter_unfolded = np.unique(
        np.round(offcenter_unfolded, 10), axis=0
    )

    observed = {
        "long_range_point_group_relative_error": max(relative_errors),
        "long_range_point_group_max_abs": max(absolute_errors),
        "shared_projector_duplicate_orbit_max_abs": projector_spread,
        "shared_projector_time_reversal_max_abs": projector_tr,
        "generated_hermitian_max_abs": hermitian_error,
    }
    gates = {
        name: {
            "observed": observed[name],
            "threshold": threshold,
            "comparison": "<=",
            "pass": bool(observed[name] <= threshold),
        }
        for name, threshold in THRESHOLDS.items()
    }

    # Quantify the known point-group residual of the archived bare EPW matrix;
    # it is not part of the long-range-adapter gate.
    bare_q, bare_orbit = adapter.orbit(bare[index_k], K_REDUCED)
    bare_relative = []
    for qpoint, generated in zip(bare_q, bare_orbit):
        target = bare[periodic_index(qpoints, qpoint)]
        bare_relative.append(
            float(
                np.linalg.norm(generated - target)
                / max(np.linalg.norm(target), 1.0e-30)
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "graphene_kstar_point_group_adapter.npz",
        lattice=lattice,
        positions_fractional=positions,
        rotations=symmetry["rotations"],
        translations=symmetry["translations"],
        projector_K=projector_k_mean,
        projector_K_prime=projector_k_prime_mean,
        offcenter_seed=offcenter,
        offcenter_orbit=offcenter_q,
        offcenter_unique_orbit=unique_offcenter,
        offcenter_unique_unfolded_orbit=unique_offcenter_unfolded,
    )
    summary = {
        "status": "PASS" if all(gate["pass"] for gate in gates.values()) else "FAIL",
        "scope": (
            "C3/C6 space-group lifting of existing graphene K/K' long-range matrices; "
            "three smearings; no new DFT"
        ),
        "number_of_space_group_operations": len(symmetry["rotations"]),
        "K_star_unique_reduced_qpoints": sorted(
            [list(key) for key in projector_by_q]
        ),
        "generic_offcenter_unique_orbit_size": len(unique_offcenter),
        "generic_offcenter_unique_qpoints": unique_offcenter.tolist(),
        "generic_offcenter_unique_unfolded_qpoints": (
            unique_offcenter_unfolded.tolist()
        ),
        "gates": gates,
        "records": records,
        "bare_EPW_K_star_maximum_relative_residual_not_gated": max(bare_relative),
        "fourier_gauge": {
            "positions_fractional": positions.tolist(),
            "warning": (
                "the EPW q6 cell-gauge atom convention must be retained when "
                "constructing Bloch phases"
            ),
        },
        "limitations": [
            "matrix covariance is directly validated at K/K' where archived q6 references exist",
            "the generic off-centre test validates orbit geometry and representation unitarity; dense two-dimensional EPC references are not available",
            "a physical off-centre generator must predict the canonical-wedge projector/vertex before this exact symmetry lifting is applied",
        ],
    }
    (args.output_dir / "kstar_point_group_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    cartesian = unique_offcenter_unfolded @ reciprocal
    fig, axis = plt.subplots(figsize=(5.8, 5.0))
    angles = np.arctan2(cartesian[:, 1], cartesian[:, 0])
    order = np.argsort(angles)
    axis.plot(
        np.r_[cartesian[order, 0], cartesian[order[0], 0]],
        np.r_[cartesian[order, 1], cartesian[order[0], 1]],
        color="#B8CCE0",
        lw=0.9,
        zorder=1,
    )
    axis.scatter(
        cartesian[:, 0], cartesian[:, 1], s=44, color="#2676B8", zorder=2
    )
    axis.scatter([0.0], [0.0], marker="+", s=55, color="#555555", zorder=3)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$q_x$ (Å$^{-1}$)")
    axis.set_ylabel(r"$q_y$ (Å$^{-1}$)")
    axis.set_title("Graphene generic K-patch point: 12-member C3/C6 orbit")
    axis.grid(color="#E6E6E6", lw=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "generic_offcenter_kstar_orbit.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
