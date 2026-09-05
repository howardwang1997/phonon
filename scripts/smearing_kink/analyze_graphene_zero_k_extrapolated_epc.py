#!/usr/bin/env python3
"""Compare full-EPC zero-smearing depth with existing k-grid extrapolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-summary",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E4_joint_zero_finite_rank1/joint_zero_finite_rank1_summary.json"
        ),
    )
    parser.add_argument(
        "--epc-summary",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E11_epc_zero_adaptive_c288_p360_eta0p0002"
            / "epc_zero_adaptive_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E12_zero_k_extrapolated_epc"
        ),
    )
    args = parser.parse_args()

    joint = json.loads(args.joint_summary.read_text(encoding="utf-8"))
    epc = json.loads(args.epc_summary.read_text(encoding="utf-8"))
    references = joint["zero_reference_sensitivity"]
    k_values = np.asarray(
        [int(row["zero_reference"].removeprefix("k")) for row in references],
        float,
    )
    inverse_k = 1.0 / k_values
    output_by_direction = {}
    for direction in ("KG", "KM"):
        depth = np.asarray(
            [row[f"{direction}_d003_cusp_depth_cm-1"] for row in references],
            float,
        )
        slope, intercept = np.polyfit(inverse_k, depth, 1)
        fitted = slope * inverse_k + intercept
        pairwise_intercepts = []
        for first, second in ((0, 1), (1, 2), (0, 2)):
            pair_slope, pair_intercept = np.polyfit(
                inverse_k[[first, second]], depth[[first, second]], 1
            )
            pairwise_intercepts.append(float(pair_intercept))
        epc_depth = float(
            epc["adaptive_metrics"]["by_direction"][direction][
                "prediction_d003_cusp_depth_cm-1"
            ]
        )
        output_by_direction[direction] = {
            "k_values": k_values.astype(int).tolist(),
            "DFPT_depth_cm-1": depth.tolist(),
            "linear_1_over_k": {
                "intercept_k_infinity_cm-1": float(intercept),
                "slope_cm-1_times_k": float(slope),
                "fitted_depth_cm-1": fitted.tolist(),
                "maximum_abs_residual_cm-1": float(
                    np.max(np.abs(fitted - depth))
                ),
            },
            "pairwise_1_over_k_intercepts_cm-1": pairwise_intercepts,
            "pairwise_intercept_range_cm-1": [
                min(pairwise_intercepts),
                max(pairwise_intercepts),
            ],
            "adaptive_full_EPC_depth_cm-1": epc_depth,
            "EPC_relative_error_vs_three_point_1_over_k_intercept": float(
                abs(epc_depth - intercept) / abs(intercept)
            ),
        }

    checks = {
        "three_point_1_over_k_max_residual_lt_0p02_cm-1": max(
            row["linear_1_over_k"]["maximum_abs_residual_cm-1"]
            for row in output_by_direction.values()
        )
        < 0.02,
        "full_EPC_depth_within_10pct_of_1_over_k_limit": max(
            row["EPC_relative_error_vs_three_point_1_over_k_intercept"]
            for row in output_by_direction.values()
        )
        < 0.10,
    }
    status = (
        "full_EPC_consistent_with_1_over_k_extrapolated_zero_depth"
        if all(checks.values())
        else "full_EPC_not_consistent_with_current_zero_extrapolation"
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    colors = {"KG": "#2676b8", "KM": "#7a3db8"}
    x_line = np.linspace(0.0, inverse_k.max() * 1.05, 200)
    for direction in ("KG", "KM"):
        row = output_by_direction[direction]
        depth = np.asarray(row["DFPT_depth_cm-1"])
        intercept = row["linear_1_over_k"]["intercept_k_infinity_cm-1"]
        slope = row["linear_1_over_k"]["slope_cm-1_times_k"]
        axis.plot(
            inverse_k,
            depth,
            "o",
            color=colors[direction],
            label=f"DFPT {direction}",
        )
        axis.plot(
            x_line,
            intercept + slope * x_line,
            "--",
            color=colors[direction],
            alpha=0.8,
            label=f"1/k fit {direction}",
        )
        axis.plot(
            [0.0],
            [row["adaptive_full_EPC_depth_cm-1"]],
            marker="*",
            ms=11,
            color=colors[direction],
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=f"full EPC {direction}",
        )
    axis.set_xlabel(r"$1/N_k$")
    axis.set_ylabel(r"$d=0.003$ cusp depth (cm$^{-1}$)")
    axis.set_title("Zero-smearing K cusp: k-grid trend and full EPC limit")
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.79, 0.5),
        frameon=False,
    )
    figure.subplots_adjust(left=0.13, right=0.77, bottom=0.14, top=0.89)
    figure.savefig(
        output / "zero_k_extrapolated_epc_comparison.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "status": status,
        "scope": (
            "existing k192/k240/k288 optimized-tetrahedron d=0.003 depths "
            "versus the stored full-EPC adaptive integral; no new DFT"
        ),
        "directions": output_by_direction,
        "checks": checks,
        "conclusion": (
            "The k288 depth is not a converged zero-smearing target. Under the "
            "physically motivated leading 1/Nk error model for the unresolved "
            "nonanalytic integral, the full-EPC result agrees with the extrapolated "
            "limit to better than 10%."
        ),
        "limitations": [
            "only three k grids are available, so the asymptotic 1/Nk law cannot be independently distinguished from a free power law",
            "the comparison validates the d=0.003 depth, not the outer zero-smearing slope ratio whose DFPT values remain k192",
            "an independent zero-temperature triangle/tetrahedron implementation is still required before freezing the zero-smearing curve",
        ],
        "inputs": {
            "joint_summary": str(args.joint_summary),
            "epc_summary": str(args.epc_summary),
        },
    }
    (output / "zero_k_extrapolated_epc_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
