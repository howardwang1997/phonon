#!/usr/bin/env python3
"""Make a simple absolute plot and a matched-direction DFPT comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
COLORS = {
    0.0: "#111111",
    0.0019000869: "#0072B2",
    0.00285013035: "#009E73",
    0.0038001738: "#D55E00",
}
FINITE_DFPT = {
    0.0019000869: ROOT
    / "results/graphene_physical_fd_dfpt/campaigns/FD300_LINE/graphene_FD300_LINE_dfpt.csv",
    0.00285013035: BASE
    / "source_p4_450/dfpt/FD450_LINE/graphene_FD450_LINE_dfpt.csv",
    0.0038001738: ROOT
    / "results/graphene_physical_fd_dfpt/campaigns/FD600_LINE/graphene_FD600_LINE_dfpt.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-sweep",
        type=Path,
        default=BASE
        / "E31_fixed_background_smearing_sweep/fixed_background_smearing_sweep_Aprime.csv",
    )
    parser.add_argument(
        "--zero-summary",
        type=Path,
        default=BASE
        / "E14_zero_quadrature_convergence/zero_quadrature_convergence_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E32_absolute_and_DFPT_comparison",
    )
    return parser.parse_args()


def load_controlled(path: Path) -> tuple[np.ndarray, dict[float, list[dict[str, str]]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    smearings = np.asarray(
        sorted({float(row["smearing_degauss_Ry"]) for row in rows}), float
    )
    blocks = {
        float(smearing): [
            row
            for row in rows
            if np.isclose(
                float(row["smearing_degauss_Ry"]), smearing, atol=1.0e-14
            )
        ]
        for smearing in smearings
    }
    if len(smearings) != 4 or any(len(block) != 121 for block in blocks.values()):
        raise RuntimeError("expected four 121-point controlled curves")
    return smearings, blocks


def frequency(block: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [float(row["controlled_MLIP_plus_full_EPC_cm-1"]) for row in block],
        float,
    )


def signed_distance(block: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([float(row["signed_distance"]) for row in block], float)


def direction_curve(
    block: list[dict[str, str]], direction: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row for row in block if row["direction"] in {"K", direction}
    ]
    distance = np.asarray([float(row["distance"]) for row in selected], float)
    values = frequency(selected)
    order = np.argsort(distance)
    return distance[order], values[order]


def load_finite_dfpt(path: Path, maximum_distance: float = 0.03) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["region"] == "K"]
    rows = [
        row
        for row in rows
        if float(row["t_GK"]) <= 1.0 + 1.0e-12
        and 1.0 - float(row["t_GK"]) <= maximum_distance + 1.0e-12
    ]
    rows.sort(key=lambda row: 1.0 - float(row["t_GK"]))
    return {
        "distance": np.asarray([1.0 - float(row["t_GK"]) for row in rows], float),
        "frequency_cm1": np.asarray([float(row["f6_cm-1"]) for row in rows], float),
        "degauss_Ry": float(rows[0]["degauss_Ry"]),
        "kgrid": int(rows[0]["kgrid"]),
    }


def make_absolute_figure(
    png_path: Path,
    pdf_path: Path,
    smearings: np.ndarray,
    blocks: dict[float, list[dict[str, str]]],
) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.1))
    for smearing in smearings:
        block = blocks[float(smearing)]
        x = signed_distance(block)
        values = frequency(block)
        order = np.argsort(x)
        label = "0 Ry" if smearing == 0.0 else f"{smearing:.7f} Ry"
        axis.plot(
            x[order],
            values[order],
            color=COLORS[float(smearing)],
            lw=2.7 if smearing == 0.0 else 2.1,
            label=label,
        )
    axis.axvline(0.0, color="#AAAAAA", lw=0.9, zorder=0)
    axis.set_xlim(-0.03, 0.03)
    axis.set_xlabel("signed distance from K (K→Γ < 0; K→M > 0)")
    axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    axis.set_title("Graphene K-neighbourhood A′: fixed-background absolute frequency")
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.77, 0.58),
        frameon=False,
        title="smearing/degauss (Ry)",
    )
    figure.text(
        0.77,
        0.30,
        "Same static MLIP background.\n"
        "One common EPC reference constant.\n"
        "No lattice-temperature background.\n"
        "No per-smearing K re-anchoring.",
        fontsize=9.1,
        color="#555555",
        ha="left",
        va="top",
    )
    figure.subplots_adjust(left=0.11, right=0.75, bottom=0.14, top=0.88)
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def make_dfpt_figure(
    png_path: Path,
    pdf_path: Path,
    smearings: np.ndarray,
    blocks: dict[float, list[dict[str, str]]],
    zero_reference: dict,
    finite_metrics: dict,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 7.8), sharey=True)
    axes_flat = axes.flat

    zero_block = blocks[0.0]
    zero_x = signed_distance(zero_block)
    zero_frequency = frequency(zero_block)
    zero_order = np.argsort(zero_x)
    axes_flat[0].plot(
        zero_x[zero_order],
        zero_frequency[zero_order],
        color=COLORS[0.0],
        lw=2.4,
    )
    zero_K = float(zero_frequency[0])
    zero_dfpt_x = np.asarray([-0.003, 0.0, 0.003])
    zero_dfpt_y = np.asarray(
        [
            zero_K
            + float(
                zero_reference["directions"]["KG"][
                    "DFPT_three_point_1_over_k_limit_cm-1"
                ]
            ),
            zero_K,
            zero_K
            + float(
                zero_reference["directions"]["KM"][
                    "DFPT_three_point_1_over_k_limit_cm-1"
                ]
            ),
        ]
    )
    axes_flat[0].plot(
        zero_dfpt_x,
        zero_dfpt_y,
        "s",
        color="#333333",
        ms=5.2,
        zorder=3,
    )
    axes_flat[0].set_xlim(-0.0045, 0.0045)
    axes_flat[0].set_xticks(
        [-0.003, 0.0, 0.003],
        ["−0.003\nK→Γ", "K", "+0.003\nK→M"],
    )
    axes_flat[0].set_title("smearing/degauss = 0 Ry")
    axes_flat[0].set_xlabel("signed distance from K")

    for axis, smearing in zip(axes_flat[1:], smearings[1:]):
        block = blocks[float(smearing)]
        model_distance, model_frequency = direction_curve(block, "KG")
        direct = load_finite_dfpt(FINITE_DFPT[float(smearing)])
        axis.plot(
            model_distance,
            model_frequency,
            color=COLORS[float(smearing)],
            lw=2.2,
        )
        axis.plot(
            direct["distance"],
            direct["frequency_cm1"],
            "o",
            color="#333333",
            ms=4.8,
            zorder=3,
        )
        axis.set_xlim(0.0, 0.03)
        axis.set_title(
            f"smearing/degauss = {smearing:.7f} Ry; DFPT k{direct['kgrid']}"
        )
        axis.set_xlabel("distance from K toward Γ")

    for axis in axes_flat:
        axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
        axis.grid(alpha=0.18)
    figure.suptitle(
        "Graphene A′: fixed-background MLIP + full EPC LR versus DFPT",
        fontsize=15.5,
        y=0.98,
    )
    legend_handles = [
        Line2D([0], [0], color="#555555", lw=2.3, label="MLIP + full EPC LR"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#333333",
            markeredgecolor="#333333",
            label="direct DFPT",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#333333",
            markeredgecolor="#333333",
            label=r"zero-smearing DFPT $1/N_k$ extrapolation",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.79, 0.73),
        frameon=False,
    )
    finite_lines = ["Finite-smearing K→Γ absolute errors:"]
    for smearing in smearings[1:]:
        metric = finite_metrics[f"{smearing:.10f}"]
        finite_lines.append(
            f"{smearing:.7f}: RMSE {metric['RMSE_cm-1']:.2f}, "
            f"K offset {metric['K_error_cm-1']:+.2f} cm⁻¹"
        )
    finite_lines.extend(
        [
            "",
            "Zero-smearing d=0.003:",
            "quadrature-extrapolated depth errors",
            (
                f"K→Γ {zero_reference['directions']['KG']['relative_difference_vs_DFPT_limit']:.2%}; "
                f"K→M {zero_reference['directions']['KM']['relative_difference_vs_DFPT_limit']:.2%}"
            ),
            "",
            "Finite DFPT is available only on",
            "the common K→Γ direction.",
        ]
    )
    figure.text(
        0.79,
        0.54,
        "\n".join(finite_lines),
        fontsize=8.9,
        color="#555555",
        ha="left",
        va="top",
    )
    figure.subplots_adjust(
        left=0.08, right=0.77, bottom=0.09, top=0.90, wspace=0.22, hspace=0.32
    )
    figure.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    smearings, blocks = load_controlled(args.controlled_sweep)
    zero_reference = json.loads(args.zero_summary.read_text(encoding="utf-8"))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    absolute_png = output / "fixed_background_absolute_Aprime.png"
    absolute_pdf = output / "fixed_background_absolute_Aprime.pdf"
    make_absolute_figure(absolute_png, absolute_pdf, smearings, blocks)

    comparison_rows = []
    finite_metrics = {}
    for smearing in smearings[1:]:
        block = blocks[float(smearing)]
        model_distance, model_frequency = direction_curve(block, "KG")
        interpolator = PchipInterpolator(model_distance, model_frequency)
        direct = load_finite_dfpt(FINITE_DFPT[float(smearing)])
        prediction = np.asarray(interpolator(direct["distance"]), float)
        error = prediction - direct["frequency_cm1"]
        K_index = int(np.argmin(direct["distance"]))
        metric = {
            "smearing_degauss_Ry": float(smearing),
            "DFPT_kgrid": direct["kgrid"],
            "number_of_points": len(error),
            "RMSE_cm-1": float(np.sqrt(np.mean(error**2))),
            "maximum_absolute_error_cm-1": float(np.max(np.abs(error))),
            "K_error_cm-1": float(error[K_index]),
            "K_aligned_shape_RMSE_cm-1": float(
                np.sqrt(np.mean(((prediction - prediction[K_index]) - (direct["frequency_cm1"] - direct["frequency_cm1"][K_index])) ** 2))
            ),
        }
        finite_metrics[f"{smearing:.10f}"] = metric
        for distance, target, predicted, signed_error in zip(
            direct["distance"], direct["frequency_cm1"], prediction, error
        ):
            comparison_rows.append(
                {
                    "smearing_degauss_Ry": float(smearing),
                    "DFPT_kgrid": direct["kgrid"],
                    "direction": "KG",
                    "distance": float(distance),
                    "DFPT_frequency_cm-1": float(target),
                    "MLIP_plus_full_EPC_LR_cm-1": float(predicted),
                    "signed_error_cm-1": float(signed_error),
                }
            )

    zero_block = blocks[0.0]
    zero_values = frequency(zero_block)
    zero_K = float(zero_values[0])
    zero_metrics = {}
    for direction in ("KG", "KM"):
        row = zero_reference["directions"][direction]
        model_index = next(
            index
            for index, source in enumerate(zero_block)
            if source["direction"] == direction
            and np.isclose(float(source["distance"]), 0.003, atol=1.0e-14)
        )
        model_depth = float(zero_values[model_index] - zero_K)
        target_depth = float(row["DFPT_three_point_1_over_k_limit_cm-1"])
        zero_metrics[direction] = {
            "dense_curve_d003_depth_cm-1": model_depth,
            "quadrature_extrapolated_d003_depth_cm-1": float(
                row["recommended_depth_cm-1"]
            ),
            "DFPT_1_over_k_extrapolated_d003_depth_cm-1": target_depth,
            "dense_curve_relative_error": float(
                abs(model_depth - target_depth) / abs(target_depth)
            ),
            "quadrature_extrapolated_relative_error": float(
                row["relative_difference_vs_DFPT_limit"]
            ),
        }
        comparison_rows.append(
            {
                "smearing_degauss_Ry": 0.0,
                "DFPT_kgrid": "1_over_k_extrapolated",
                "direction": direction,
                "distance": 0.003,
                "DFPT_frequency_cm-1": zero_K + target_depth,
                "MLIP_plus_full_EPC_LR_cm-1": zero_values[model_index],
                "signed_error_cm-1": model_depth - target_depth,
            }
        )

    comparison_csv = output / "MLIP_full_EPC_LR_vs_DFPT_points.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    dfpt_png = output / "MLIP_full_EPC_LR_vs_DFPT_Aprime.png"
    dfpt_pdf = output / "MLIP_full_EPC_LR_vs_DFPT_Aprime.pdf"
    make_dfpt_figure(
        dfpt_png,
        dfpt_pdf,
        smearings,
        blocks,
        zero_reference,
        finite_metrics,
    )
    summary = {
        "status": "simple_absolute_and_matched_direction_DFPT_comparison_generated",
        "interpretation_of_user_term": "DPT treated as DFPT",
        "absolute_plot": {
            "fixed_static_MLIP_background": True,
            "common_EPC_reference_constant": True,
            "lattice_temperature_background_used": False,
            "per_smearing_K_reanchoring": False,
        },
        "zero_smearing_DFPT_comparison": zero_metrics,
        "finite_smearing_DFPT_comparison": finite_metrics,
        "comparison_scope": {
            "zero": "K and d=0.003 on K-to-Gamma/K-to-M, using 1/Nk extrapolated DFPT depths",
            "finite": "common K-to-Gamma direction only; direct static-lattice DFPT",
        },
        "limitations": [
            "the common absolute EPC reference constant is calibrated once from the zero-smearing K anchor",
            "finite-smearing direct DFPT uses the available k144, k144, and k120 calculations and is not a new k-grid extrapolation",
            "no finite-smearing K-to-M DFPT points are available, so that direction is not shown as validated",
        ],
        "outputs": {
            "absolute_png": str(absolute_png.resolve()),
            "absolute_pdf": str(absolute_pdf.resolve()),
            "DFPT_png": str(dfpt_png.resolve()),
            "DFPT_pdf": str(dfpt_pdf.resolve()),
            "comparison_csv": str(comparison_csv.resolve()),
        },
    }
    summary_path = output / "absolute_and_DFPT_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
