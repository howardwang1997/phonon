#!/usr/bin/env python3
"""Merge and gate the two-machine lowest-degauss dense K-line DFPT result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE = (
    ROOT
    / "results"
    / "graphene_physical_fd_dfpt"
    / "campaigns"
    / "FD0_SCAN"
    / "graphene_FD0_SCAN_dfpt.csv"
)


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


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for row in source:
        if row["region"] != "K":
            continue
        rows.append(
            {
                "source": str(path),
                "campaign": row["campaign"],
                "degauss_Ry": float(row["degauss_Ry"]),
                "kgrid": int(row["kgrid"]),
                "t_GK": float(row["t_GK"]),
                "frequencies_cm1": np.asarray(
                    [float(row[f"f{mode}_cm-1"]) for mode in range(1, 7)]
                ),
            }
        )
    if not rows:
        raise ValueError(f"no K rows in {path}")
    return rows


def row_at(rows: list[dict], t_value: float, tolerance: float = 1.0e-8) -> dict:
    matches = [row for row in rows if abs(row["t_GK"] - t_value) <= tolerance]
    if len(matches) != 1:
        raise ValueError(f"expected one row at t={t_value}, found {len(matches)}")
    return matches[0]


def pair_metrics(t: np.ndarray, frequency: np.ndarray) -> list[dict]:
    center_index = int(np.argmin(np.abs(t - 1.0)))
    center_frequency = float(frequency[center_index])
    result = []
    for left_index in range(center_index - 1, -1, -1):
        delta = 1.0 - float(t[left_index])
        right_index = int(np.argmin(np.abs(t - (1.0 + delta))))
        if abs(float(t[right_index]) - (1.0 + delta)) > 1.0e-8:
            continue
        left_frequency = float(frequency[left_index])
        right_frequency = float(frequency[right_index])
        depth = 0.5 * (left_frequency + right_frequency) - center_frequency
        jump = abs(
            (right_frequency - center_frequency) / delta
            - (center_frequency - left_frequency) / delta
        )
        result.append(
            {
                "delta_t": delta,
                "left_t": float(t[left_index]),
                "right_t": float(t[right_index]),
                "left_cm-1": left_frequency,
                "center_cm-1": center_frequency,
                "right_cm-1": right_frequency,
                "mirror_abs_cm-1": abs(left_frequency - right_frequency),
                "cusp_depth_cm-1": depth,
                "slope_jump_cm-1_per_t": jump,
            }
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    combined_rows: list[dict],
    baseline_rows: list[dict],
    pairs: list[dict],
    output_dir: Path,
) -> list[Path]:
    t = np.asarray([row["t_GK"] for row in combined_rows])
    top = np.asarray([row["frequencies_cm1"][-1] for row in combined_rows])
    center = float(top[np.argmin(np.abs(t - 1.0))])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    source_colors = {"baseline_B": "#222222", "lane_A": "#0072B2", "lane_B": "#D55E00"}
    source_labels = {
        "baseline_B": "existing B baseline",
        "lane_A": "new V100-A",
        "lane_B": "new V100-B",
    }
    axes[0].plot(t, top, color="#555555", lw=1.1, zorder=1)
    for source in ("baseline_B", "lane_A", "lane_B"):
        selected = [row for row in combined_rows if row["merge_source"] == source]
        axes[0].scatter(
            [row["t_GK"] for row in selected],
            [row["frequencies_cm1"][-1] for row in selected],
            s=28,
            color=source_colors[source],
            label=source_labels[source],
            zorder=3,
        )
    axes[0].axvline(1.0, color="#BBBBBB", lw=0.8)
    axes[0].set_xlabel(r"$t$ along $\mathbf{q}=t\mathbf{K}$")
    axes[0].set_ylabel(r"highest optical frequency (cm$^{-1}$)")
    axes[0].set_title("Lowest finite degauss, k192")
    axes[0].legend(frameon=False, fontsize=7.7)

    axes[1].plot(t, top - center, "o-", color="#0072B2", lw=1.4, ms=3.5)
    axes[1].axvline(1.0, color="#BBBBBB", lw=0.8)
    axes[1].set_xlabel(r"$t$ along $\mathbf{q}=t\mathbf{K}$")
    axes[1].set_ylabel(r"$\omega(t)-\omega(K)$ (cm$^{-1}$)")
    axes[1].set_title("K-centered cusp")

    delta = np.asarray([row["delta_t"] for row in pairs])
    depth = np.asarray([row["cusp_depth_cm-1"] for row in pairs])
    jump = np.asarray([row["slope_jump_cm-1_per_t"] for row in pairs])
    left_axis = axes[2]
    right_axis = left_axis.twinx()
    left_axis.plot(delta, depth, "o-", color="#D55E00", label="cusp depth")
    right_axis.plot(delta, jump, "s--", color="#009E73", label="slope jump")
    left_axis.set_xlabel(r"symmetric half-window $\delta t$")
    left_axis.set_ylabel(r"cusp depth (cm$^{-1}$)", color="#D55E00")
    right_axis.set_ylabel(r"slope jump (cm$^{-1}$/t)", color="#009E73")
    left_axis.tick_params(axis="y", colors="#D55E00")
    right_axis.tick_params(axis="y", colors="#009E73")
    left_axis.set_title("Window dependence")

    for axis in axes:
        axis.grid(axis="y", color="#E6E6E6", lw=0.55)
        axis.spines["top"].set_visible(False)
        axis.tick_params(direction="out", length=3.5, width=0.8)
    right_axis.spines["top"].set_visible(False)
    fig.suptitle(
        "Graphene direct DFPT K cusp at degauss=0.0006333623 Ry",
        fontsize=12.0,
        y=0.99,
    )
    fig.subplots_adjust(left=0.07, right=0.94, top=0.88, bottom=0.16, wspace=0.34)
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"graphene_fd0_k192_dense_cusp.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--lane-a",
        type=Path,
        default=ROOT
        / "results"
        / "graphene_kohn_cusp_two_methods"
        / "C0_remote_dfpt"
        / "A"
        / "graphene_FD0_K_DENSE_A_dfpt.csv",
    )
    parser.add_argument(
        "--lane-b",
        type=Path,
        default=ROOT
        / "results"
        / "graphene_kohn_cusp_two_methods"
        / "C0_remote_dfpt"
        / "B"
        / "graphene_FD0_K_DENSE_B_dfpt.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "graphene_kohn_cusp_two_methods"
        / "C0_fd0_k192_dense",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_all = load_rows(args.baseline)
    baseline = [
        row
        for row in baseline_all
        if abs(row["degauss_Ry"] - 0.0006333623) < 1.0e-12
        and row["kgrid"] == 192
    ]
    lane_a = load_rows(args.lane_a)
    lane_b = load_rows(args.lane_b)
    for name, rows in (("baseline", baseline), ("lane A", lane_a), ("lane B", lane_b)):
        if any(abs(row["degauss_Ry"] - 0.0006333623) > 1.0e-12 for row in rows):
            raise ValueError(f"{name} contains a different degauss")
        if any(row["kgrid"] != 192 for row in rows):
            raise ValueError(f"{name} contains a different k grid")

    center_baseline = row_at(baseline, 1.0)
    center_a = row_at(lane_a, 1.0)
    duplicate_difference = np.abs(
        center_a["frequencies_cm1"] - center_baseline["frequencies_cm1"]
    )

    selected: dict[float, dict] = {}
    for source_name, rows in (
        ("baseline_B", baseline),
        ("lane_A", lane_a),
        ("lane_B", lane_b),
    ):
        for row in rows:
            key = round(row["t_GK"], 6)
            if key in selected:
                if key == 1.0 and source_name == "lane_A":
                    continue
                raise ValueError(f"unexpected duplicate q point t={key}")
            selected[key] = {**row, "merge_source": source_name}
    combined = [selected[key] for key in sorted(selected)]
    expected_t = np.asarray(
        [
            0.977, 0.981, 0.985, 0.989, 0.993, 0.997,
            1.000,
            1.003, 1.007, 1.011, 1.015, 1.019, 1.023,
        ]
    )
    t_values = np.asarray([row["t_GK"] for row in combined])
    if not np.allclose(t_values, expected_t, atol=1.0e-10, rtol=0.0):
        raise ValueError(f"merged C0 q list differs from frozen list: {t_values}")
    top = np.asarray([row["frequencies_cm1"][-1] for row in combined])
    pairs = pair_metrics(t_values, top)
    if len(pairs) != 6:
        raise ValueError(f"expected six symmetric cusp windows, found {len(pairs)}")

    x = np.abs(t_values - 1.0)
    design = np.column_stack((np.ones_like(x), x, x**2))
    coefficients, _, _, _ = np.linalg.lstsq(design, top, rcond=None)
    fitted = design @ coefficients
    fit_rmse = float(np.sqrt(np.mean((fitted - top) ** 2)))
    fit_slope_jump = float(2.0 * coefficients[1])
    max_mirror = float(max(row["mirror_abs_cm-1"] for row in pairs))
    positive_depths = all(row["cusp_depth_cm-1"] > 0.0 for row in pairs)
    center_is_minimum = bool(top[np.argmin(np.abs(t_values - 1.0))] == top.min())
    passed = bool(
        float(duplicate_difference.max()) <= 0.1
        and max_mirror <= 1.0
        and positive_depths
        and center_is_minimum
        and fit_rmse <= 1.0
        and fit_slope_jump > 0.0
    )

    prediction_rows = []
    for row in combined:
        prediction_rows.append(
            {
                "merge_source": row["merge_source"],
                "campaign": row["campaign"],
                "degauss_Ry": row["degauss_Ry"],
                "kgrid": row["kgrid"],
                "t_GK": row["t_GK"],
                **{
                    f"f{mode}_cm-1": float(row["frequencies_cm1"][mode - 1])
                    for mode in range(1, 7)
                },
            }
        )
    write_csv(args.output_dir / "c0_merged_dfpt.csv", prediction_rows)
    write_csv(args.output_dir / "c0_cusp_window_metrics.csv", pairs)
    figures = make_figure(combined, baseline, pairs, args.output_dir)
    summary = {
        "status": "passed_q_density" if passed else "failed",
        "scope": "two-machine lowest-finite-degauss k192 direct-DFPT K-line densification",
        "degauss_Ry": 0.0006333623,
        "kgrid": 192,
        "n_unique_qpoints": len(combined),
        "cross_machine_duplicate_K": {
            "V100_A_cm-1": center_a["frequencies_cm1"].tolist(),
            "V100_B_cm-1": center_baseline["frequencies_cm1"].tolist(),
            "all_mode_max_abs_cm-1": float(duplicate_difference.max()),
            "top_mode_abs_cm-1": float(duplicate_difference[-1]),
        },
        "cusp_windows": pairs,
        "absolute_value_cusp_fit": {
            "formula": "omega(t)=c0+c1*abs(t-1)+c2*abs(t-1)^2",
            "coefficients": coefficients.tolist(),
            "fit_RMSE_cm-1": fit_rmse,
            "slope_jump_2c1_cm-1_per_t": fit_slope_jump,
        },
        "checks": {
            "cross_machine_all_mode_max_le_0p1_cm-1": float(duplicate_difference.max()) <= 0.1,
            "mirror_top_max_le_1_cm-1": max_mirror <= 1.0,
            "all_symmetric_cusp_depths_positive": positive_depths,
            "K_is_local_minimum": center_is_minimum,
            "absolute_value_fit_RMSE_le_1_cm-1": fit_rmse <= 1.0,
            "fitted_slope_jump_positive": fit_slope_jump > 0.0,
        },
        "sources": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (args.baseline, args.lane_a, args.lane_b)
        ],
        "outputs": [str(path.relative_to(ROOT)) for path in figures]
        + [
            str((args.output_dir / "c0_merged_dfpt.csv").relative_to(ROOT)),
            str((args.output_dir / "c0_cusp_window_metrics.csv").relative_to(ROOT)),
        ],
        "limitation": (
            "This gate establishes q-density and cross-machine reproducibility at k192. "
            "A separate fixed-degauss k-grid subset is required before the near-zero reference is final."
        ),
    }
    atomic_json(args.output_dir / "c0_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
