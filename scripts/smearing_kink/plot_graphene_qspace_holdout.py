#!/usr/bin/env python3
"""Create Weekly-Report figures for the frozen graphene q-space holdout."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INDIR = ROOT / "results" / "p0_graphene_qspace"
CM_PER_THz = 33.356
DGS = (0.013, 0.027, 0.055)
COLORS = {
    "DFT": "#222222",
    "qspace": "#0072B2",
    "baseline": "#D55E00",
}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def load_rows():
    with (INDIR / "holdout_evaluation_points.csv").open() as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "degauss_Ry": float(raw["degauss_Ry"]),
                    "region": raw["region"],
                    "t_GK": float(raw["t_GK"]),
                    "DFT": float(raw["DFPT_f6_cm"]),
                    "baseline": float(raw["frozen_baseline_f6_cm"]),
                    "qspace": float(raw["qspace_prediction_f6_cm"]),
                }
            )
    summary = json.loads((INDIR / "holdout_evaluation_summary.json").read_text())
    if summary["status"] != "PASS":
        raise RuntimeError("refusing report plot: final holdout did not pass")
    return rows, summary


def line(rows, dg, region):
    selected = [
        row for row in rows
        if np.isclose(row["degauss_Ry"], dg) and row["region"] == region
    ]
    return sorted(selected, key=lambda row: row["t_GK"])


def slope_jump(data, field):
    t = np.array([row["t_GK"] for row in data])
    f = np.array([row[field] for row in data])
    left = np.polyfit(t[t < 1.0], f[t < 1.0], 1)
    right = np.polyfit(t[t > 1.0], f[t > 1.0], 1)
    return abs(right[0] - left[0]) / CM_PER_THz


def gamma_slope(data, field):
    t = np.array([row["t_GK"] for row in data])
    f = np.array([row[field] for row in data])
    mask = t <= 0.045
    return np.polyfit(t[mask], f[mask], 1)[0] / CM_PER_THz


def comparison_figure(rows, summary):
    metric_lookup = {
        (float(item["degauss_Ry"]), item["region"]): item
        for item in summary["line_metrics"]
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.7))
    for column, dg in enumerate(DGS):
        for row_index, region in enumerate(("G", "K")):
            ax = axes[row_index, column]
            data = line(rows, dg, region)
            t = np.array([item["t_GK"] for item in data])
            ax.plot(
                t,
                [item["baseline"] for item in data],
                "--",
                color=COLORS["baseline"],
                lw=1.5,
                label="frozen 8x8 baseline",
                zorder=1,
            )
            ax.plot(
                t,
                [item["qspace"] for item in data],
                "-",
                color=COLORS["qspace"],
                lw=1.8,
                label="frozen q-space method",
                zorder=2,
            )
            ax.scatter(
                t,
                [item["DFT"] for item in data],
                s=22,
                facecolor="white",
                edgecolor=COLORS["DFT"],
                linewidth=1.0,
                label="unseen direct DFPT",
                zorder=3,
            )
            if region == "K":
                ax.axvline(1.0, color="#bbbbbb", lw=0.8, zorder=0)
            metric = metric_lookup[(dg, region)]
            ax.text(
                0.04,
                0.94,
                rf"MAE = {metric['qspace_MAE_cm-1']:.3f} cm$^{{-1}}$",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color=COLORS["qspace"],
            )
            region_title = r"$\Gamma$ ray" if region == "G" else "K crossing"
            ax.set_title(
                f"{region_title}, " rf"$\delta={dg:g}$ Ry",
                fontsize=9.5,
            )
            ax.set_xlabel(r"$t$ along $\mathbf{q}=t\mathbf{K}$")
            if column == 0:
                ax.set_ylabel(r"highest optical mode (cm$^{-1}$)")
            style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        "Graphene Kohn anomaly: frozen q-space method vs unseen direct-DFPT holdout",
        y=0.995,
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    for suffix in ("png", "pdf"):
        fig.savefig(INDIR / f"qspace_holdout_comparison.{suffix}", dpi=220)
    plt.close(fig)


def physics_figure(rows, summary):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.45))
    fields = ("DFT", "qspace", "baseline")
    styles = {
        "DFT": ("o-", "direct DFPT"),
        "qspace": ("s-", "q-space method"),
        "baseline": ("^--", "frozen baseline"),
    }
    for field in fields:
        style_spec, label = styles[field]
        axes[0].plot(
            DGS,
            [slope_jump(line(rows, dg, "K"), field) for dg in DGS],
            style_spec,
            color=COLORS[field],
            lw=1.5,
            ms=4.5,
            label=label,
        )
        axes[1].plot(
            DGS,
            [gamma_slope(line(rows, dg, "G"), field) for dg in DGS],
            style_spec,
            color=COLORS[field],
            lw=1.5,
            ms=4.5,
            label=label,
        )
    axes[0].set_title("K cusp melts with smearing", fontsize=9.5)
    axes[0].set_ylabel(r"K slope jump (THz per $t$)")
    axes[1].set_title(r"$\Gamma$ anomaly weakens", fontsize=9.5)
    axes[1].set_ylabel(r"near-$\Gamma$ slope (THz per $t$)")
    for ax in axes[:2]:
        ax.set_xlabel("Fermi–Dirac degauss (Ry)")
        style(ax)

    gate = summary["acceptance_gates"]["k32_k64_low_smearing_K_MAE"]
    t_values = np.array((0.975, 0.990, 1.000, 1.010, 1.025))
    differences = np.asarray(gate["signed_differences_cm-1"])
    axes[2].axhline(0.0, color="#999999", lw=0.8)
    axes[2].plot(
        t_values,
        differences,
        "o-",
        color="#009E73",
        lw=1.4,
        ms=4.5,
    )
    axes[2].text(
        0.05,
        0.94,
        rf"MAE = {gate['observed']:.3f} cm$^{{-1}}$",
        transform=axes[2].transAxes,
        va="top",
        fontsize=8,
    )
    axes[2].set_title(r"DFPT $32^2$ vs $64^2$ convergence", fontsize=9.5)
    axes[2].set_xlabel(r"$t$ near K")
    axes[2].set_ylabel(r"$\omega_{32}-\omega_{64}$ (cm$^{-1}$)")
    style(axes[2])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8.3,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for suffix in ("png", "pdf"):
        fig.savefig(INDIR / f"qspace_holdout_physics.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    rows, summary = load_rows()
    comparison_figure(rows, summary)
    physics_figure(rows, summary)
    print("wrote qspace_holdout_comparison.{png,pdf}")
    print("wrote qspace_holdout_physics.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
