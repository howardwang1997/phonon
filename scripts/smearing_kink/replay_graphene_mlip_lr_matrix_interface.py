#!/usr/bin/env python3
"""Replay the zero-smearing graphene curve through the reusable matrix API.

The inputs are the frozen MLIP-derived short-range force constants and the
already computed full-EPC long-range squared-frequency correction from E15.
No curve is refitted.  Passing this test means the plotted solid curve can be
reproduced by an explicit Hermitian matrix addition rather than a post-hoc
frequency edit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
sys.path.insert(0, str(ROOT / "src"))

import analyze_graphene_k_cusp_b0_dense as b0  # noqa: E402
import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402
import fit_graphene_joint_zero_finite_rank1 as joint  # noqa: E402
from phonon_accel.long_range import apply_mode_projected_correction  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
THRESHOLDS = {
    "short_range_frequency_replay_max_abs_cm-1": 1.0e-8,
    "combined_frequency_replay_max_abs_cm-1": 1.0e-8,
    "total_hermitian_max_abs": 1.0e-12,
    "projector_idempotency_max_abs": 1.0e-12,
    "orthogonal_correction_leakage_max_abs": 1.0e-12,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve",
        type=Path,
        default=BASE / "E15_epc_zero_dense_curve/zero_dense_curve.csv",
    )
    parser.add_argument(
        "--static-short",
        type=Path,
        default=BASE
        / "source_p4_450/on_policy/frozen_prediction/frozen_static_fc2.npz",
    )
    parser.add_argument(
        "--operator-geometry",
        type=Path,
        default=a0.operator_path(450),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E26_mlip_lr_matrix_interface",
    )
    args = parser.parse_args()

    with args.curve.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    qpoints = np.asarray(
        [[float(row["q1"]), float(row["q2"]), 0.0] for row in rows], float
    )
    signed_distance = np.asarray([float(row["signed_distance"]) for row in rows])
    archived_short = np.asarray(
        [float(row["short_range_MLIP_cm-1"]) for row in rows]
    )
    delta_lambda = np.asarray(
        [float(row["long_range_correction_cm-2"]) for row in rows]
    )
    archived_total = np.asarray(
        [float(row["MLIP_plus_full_EPC_cm-1"]) for row in rows]
    )

    phonon, force_constants = joint.make_static_short_phonon(
        args.operator_geometry, args.static_short
    )
    sequence = b0.dynamical_sequence(phonon, force_constants, qpoints)
    mode_indices = np.argmax(sequence["frequencies_cm1"], axis=1)
    selected_short = sequence["frequencies_cm1"][
        np.arange(len(qpoints)), mode_indices
    ]
    result = apply_mode_projected_correction(
        sequence["matrices"],
        sequence["scale_cm2"],
        sequence["eigenvectors"],
        mode_indices,
        delta_lambda,
    )

    observed = {
        "short_range_frequency_replay_max_abs_cm-1": float(
            np.max(np.abs(selected_short - archived_short))
        ),
        "combined_frequency_replay_max_abs_cm-1": float(
            np.max(np.abs(result.tracked_frequency_cm1 - archived_total))
        ),
        **result.diagnostics,
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "matrix_interface_replay.npz",
        qpoints=qpoints,
        signed_distance=signed_distance,
        short_matrices=sequence["matrices"],
        correction_matrices=result.correction_matrices,
        total_matrices=result.total_matrices,
        projectors=result.projectors,
        delta_lambda_cm2=delta_lambda,
        short_frequency_cm1=selected_short,
        combined_frequency_cm1=result.tracked_frequency_cm1,
    )

    order = np.argsort(signed_distance)
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    axes[0].plot(
        signed_distance[order],
        selected_short[order],
        "--",
        color="#777777",
        lw=1.6,
        label="short-range MLIP",
    )
    axes[0].plot(
        signed_distance[order],
        result.tracked_frequency_cm1[order],
        "-",
        color="#2676B8",
        lw=2.0,
        label="MLIP + full EPC LR",
    )
    axes[0].set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.25), frameon=False)
    axes[1].plot(
        signed_distance[order],
        delta_lambda[order],
        color="#C84630",
        lw=1.9,
    )
    axes[1].set_ylabel(r"long-range $\Delta\lambda$ (cm$^{-2}$)")
    for axis in axes:
        axis.axvline(0.0, color="#BBBBBB", lw=0.8, zorder=0)
        axis.set_xlabel("signed distance from K")
        axis.grid(color="#E6E6E6", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Graphene, smearing/degauss = 0 Ry: matrix-level MLIP + LR assembly")
    figure.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.27, wspace=0.31)
    figure.savefig(
        args.output_dir / "matrix_interface_replay.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(figure)

    summary = {
        "status": "PASS" if all(gate["pass"] for gate in gates.values()) else "FAIL",
        "scope": (
            "matrix-level replay of the existing zero-smearing 121-point curve: "
            "frozen short-range MLIP force constants plus full-EPC long-range projector; "
            "no refit and no new DFT"
        ),
        "number_of_qpoints": len(qpoints),
        "assembly_formula": (
            "D_total(q,s)=D_MLIP_short(q)+Delta_lambda_EPC(q,s) "
            "|e_A'(q)><e_A'(q)|"
        ),
        "gates": gates,
        "inputs": {
            "zero_curve": {"path": str(args.curve), "sha256": sha256(args.curve)},
            "short_force_constants": {
                "path": str(args.static_short),
                "sha256": sha256(args.static_short),
            },
            "operator_geometry": {
                "path": str(args.operator_geometry),
                "sha256": sha256(args.operator_geometry),
            },
        },
        "interpretation": (
            "the solid zero-smearing curve is an actual Hermitian MLIP+LR dynamical-matrix "
            "calculation; the long-range response is not a cosmetic smoothing spline"
        ),
        "limitations": [
            "the full-EPC scalar response is reused from E15 and is not recomputed here",
            "this replay validates the graphene rank-one interface; mixed-mode materials use the adaptive rank-R<=4 vertex adapter before the electronic band sum",
            "absolute zero-smearing accuracy remains limited by the existing K anchor and sparse k-converged DFPT checks",
        ],
    }
    (args.output_dir / "matrix_interface_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
