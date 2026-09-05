#!/usr/bin/env python3
"""Gate the fixed-degauss k-grid subset around graphene K."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEGAUSS = 0.0006333623
T_EXPECTED = np.asarray([0.993, 0.997, 1.000, 1.003, 1.007])


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


def ordered_grid(rows: list[dict], kgrid: int) -> list[dict]:
    selected = [
        row
        for row in rows
        if row["kgrid"] == kgrid
        and abs(row["degauss_Ry"] - DEGAUSS) <= 1.0e-12
        and any(abs(row["t_GK"] - value) <= 1.0e-10 for value in T_EXPECTED)
    ]
    by_t: dict[float, dict] = {}
    for row in selected:
        key = round(row["t_GK"], 6)
        if key in by_t:
            raise ValueError(f"duplicate k={kgrid}, t={key}")
        by_t[key] = row
    expected_keys = [round(value, 6) for value in T_EXPECTED]
    if sorted(by_t) != sorted(expected_keys):
        raise ValueError(
            f"k={kgrid} q list mismatch: got {sorted(by_t)}, expected {expected_keys}"
        )
    return [by_t[key] for key in expected_keys]


def cusp_windows(t: np.ndarray, top: np.ndarray) -> list[dict]:
    center = int(np.argmin(np.abs(t - 1.0)))
    result = []
    for delta in (0.003, 0.007):
        left = int(np.argmin(np.abs(t - (1.0 - delta))))
        right = int(np.argmin(np.abs(t - (1.0 + delta))))
        depth = 0.5 * (top[left] + top[right]) - top[center]
        jump = abs(
            (top[right] - top[center]) / delta
            - (top[center] - top[left]) / delta
        )
        result.append(
            {
                "delta_t": delta,
                "left_cm-1": float(top[left]),
                "center_cm-1": float(top[center]),
                "right_cm-1": float(top[right]),
                "mirror_abs_cm-1": float(abs(top[left] - top[right])),
                "cusp_depth_cm-1": float(depth),
                "slope_jump_cm-1_per_t": float(jump),
            }
        )
    return result


def fit_metrics(t: np.ndarray, top: np.ndarray) -> dict:
    x = np.abs(t - 1.0)
    design = np.column_stack((np.ones_like(x), x, x**2))
    coefficients, _, _, _ = np.linalg.lstsq(design, top, rcond=None)
    fitted = design @ coefficients
    return {
        "coefficients": coefficients.tolist(),
        "fit_RMSE_cm-1": float(np.sqrt(np.mean((fitted - top) ** 2))),
        "slope_jump_2c1_cm-1_per_t": float(2.0 * coefficients[1]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(grids: dict[int, np.ndarray], output_dir: Path) -> list[Path]:
    colors = {168: "#D55E00", 192: "#222222", 216: "#0072B2"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for kgrid in (168, 192, 216):
        top = grids[kgrid][:, -1]
        axes[0].plot(
            T_EXPECTED, top, "o-", color=colors[kgrid], lw=1.45, ms=4,
            label=f"k{kgrid}",
        )
        axes[1].plot(
            T_EXPECTED, top - top[2], "o-", color=colors[kgrid], lw=1.45, ms=4,
        )
    for kgrid in (168, 216):
        difference = grids[kgrid][:, -1] - grids[192][:, -1]
        axes[2].plot(
            T_EXPECTED, difference, "o-", color=colors[kgrid], lw=1.45, ms=4,
            label=f"k{kgrid} − k192",
        )
    axes[0].set_ylabel(r"highest optical frequency (cm$^{-1}$)")
    axes[0].set_title("Absolute frequency")
    axes[1].set_ylabel(r"$\omega(t)-\omega(K)$ (cm$^{-1}$)")
    axes[1].set_title("K-centered cusp")
    axes[2].axhline(0.0, color="#AAAAAA", lw=0.8)
    axes[2].set_ylabel(r"difference from k192 (cm$^{-1}$)")
    axes[2].set_title("k-grid difference")
    for axis in axes:
        axis.axvline(1.0, color="#BBBBBB", lw=0.8, zorder=0)
        axis.grid(axis="y", color="#E6E6E6", lw=0.55)
        axis.set_xlabel(r"$t$ along $\mathbf{q}=t\mathbf{K}$")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.5, width=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Graphene K cusp k-grid check at degauss=0.0006333623 Ry",
        fontsize=12.0,
        y=0.99,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.16, wspace=0.34)
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"graphene_fd0_kgrid_cusp.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Graphene K cusp C1 k 网格复核",
        "",
        f"**状态：**`{summary['status']}`  ",
        "**参数：**固定 `degauss=0.0006333623 Ry`，中心五个 q 点。",
        "",
        "| k 网格 | 顶支 MAE vs k192 | 顶支最大差 | K 频率 | δ=0.003 cusp depth | δ=0.007 cusp depth |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["grid_metrics"]:
        lines.append(
            f"| {row['kgrid']} | {row['top_MAE_vs_k192_cm-1']:.4f} cm⁻¹ | "
            f"{row['top_max_abs_vs_k192_cm-1']:.4f} cm⁻¹ | "
            f"{row['K_top_cm-1']:.4f} cm⁻¹ | "
            f"{row['cusp_windows'][0]['cusp_depth_cm-1']:.4f} cm⁻¹ | "
            f"{row['cusp_windows'][1]['cusp_depth_cm-1']:.4f} cm⁻¹ |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "本复核只改变 k 网格，电子展宽、结构、赝势、截断能和 q 点均保持不变。通过后可把 k192 的 13 点结果作为最低已直接计算有限展宽下的参考线；仍不能写成 `smearing=0.00`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    base = ROOT / "results" / "graphene_kohn_cusp_two_methods"
    parser.add_argument(
        "--baseline",
        type=Path,
        default=base / "C0_fd0_k192_dense" / "c0_merged_dfpt.csv",
    )
    parser.add_argument("--a-k168", type=Path, default=base / "C1_remote_kgrid" / "A" / "k168" / "graphene_FD0_K_DENSE_A_dfpt.csv")
    parser.add_argument("--a-k216", type=Path, default=base / "C1_remote_kgrid" / "A" / "k216" / "graphene_FD0_K_DENSE_A_dfpt.csv")
    parser.add_argument("--b-k168", type=Path, default=base / "C1_remote_kgrid" / "B" / "k168" / "graphene_FD0_K_DENSE_B_dfpt.csv")
    parser.add_argument("--b-k216", type=Path, default=base / "C1_remote_kgrid" / "B" / "k216" / "graphene_FD0_K_DENSE_B_dfpt.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "C1_fd0_kgrid")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_rows(args.baseline)
    new_rows = []
    for path in (args.a_k168, args.a_k216, args.b_k168, args.b_k216):
        new_rows.extend(load_rows(path))
    ordered = {
        168: ordered_grid(new_rows, 168),
        192: ordered_grid(baseline, 192),
        216: ordered_grid(new_rows, 216),
    }
    grids = {
        kgrid: np.asarray([row["frequencies_cm1"] for row in rows])
        for kgrid, rows in ordered.items()
    }

    grid_metrics = []
    prediction_rows = []
    for kgrid in (168, 192, 216):
        frequencies = grids[kgrid]
        top = frequencies[:, -1]
        difference = frequencies - grids[192]
        windows = cusp_windows(T_EXPECTED, top)
        fit = fit_metrics(T_EXPECTED, top)
        metric = {
            "kgrid": kgrid,
            "top_MAE_vs_k192_cm-1": float(np.mean(np.abs(difference[:, -1]))),
            "top_max_abs_vs_k192_cm-1": float(np.max(np.abs(difference[:, -1]))),
            "all_mode_MAE_vs_k192_cm-1": float(np.mean(np.abs(difference))),
            "all_mode_max_abs_vs_k192_cm-1": float(np.max(np.abs(difference))),
            "K_top_cm-1": float(top[2]),
            "K_is_minimum": bool(top[2] == top.min()),
            "cusp_windows": windows,
            "fit": fit,
        }
        grid_metrics.append(metric)
        for index, t_value in enumerate(T_EXPECTED):
            prediction_rows.append(
                {
                    "degauss_Ry": DEGAUSS,
                    "kgrid": kgrid,
                    "t_GK": float(t_value),
                    **{
                        f"f{mode}_cm-1": float(frequencies[index, mode - 1])
                        for mode in range(1, 7)
                    },
                    "top_difference_vs_k192_cm-1": float(difference[index, -1]),
                }
            )

    metrics_by_grid = {row["kgrid"]: row for row in grid_metrics}
    k216 = metrics_by_grid[216]
    k168 = metrics_by_grid[168]
    checks = {
        "k216_top_MAE_vs_k192_le_1_cm-1": k216["top_MAE_vs_k192_cm-1"] <= 1.0,
        "k216_top_max_abs_vs_k192_le_2_cm-1": k216["top_max_abs_vs_k192_cm-1"] <= 2.0,
        "k216_not_worse_than_k168_top_MAE": k216["top_MAE_vs_k192_cm-1"] <= k168["top_MAE_vs_k192_cm-1"],
        "all_grids_K_are_minima": all(row["K_is_minimum"] for row in grid_metrics),
        "all_cusp_depths_positive": all(
            window["cusp_depth_cm-1"] > 0.0
            for row in grid_metrics
            for window in row["cusp_windows"]
        ),
        "all_mirror_differences_le_1_cm-1": all(
            window["mirror_abs_cm-1"] <= 1.0
            for row in grid_metrics
            for window in row["cusp_windows"]
        ),
        "all_fit_RMSE_le_1_cm-1": all(
            row["fit"]["fit_RMSE_cm-1"] <= 1.0 for row in grid_metrics
        ),
    }
    passed = all(checks.values())
    write_csv(args.output_dir / "c1_kgrid_predictions.csv", prediction_rows)
    figures = make_figure(grids, args.output_dir)
    summary = {
        "status": "passed_kgrid" if passed else "failed",
        "scope": "fixed-lowest-degauss five-point k168/k192/k216 convergence gate",
        "degauss_Ry": DEGAUSS,
        "t_values": T_EXPECTED.tolist(),
        "grid_metrics": grid_metrics,
        "checks": checks,
        "outputs": [str(path.relative_to(ROOT)) for path in figures]
        + [str((args.output_dir / "c1_kgrid_predictions.csv").relative_to(ROOT))],
        "limitation": "The smallest directly computed smearing remains finite; this is not a smearing=0.00 calculation.",
    }
    atomic_json(args.output_dir / "c1_summary.json", summary)
    write_report(summary, args.output_dir / "C1_RESULT.md")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
