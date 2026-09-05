#!/usr/bin/env python3
"""Plot the predeclared rounded-cusp model against its smooth control."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_dfpt_linecuts as dfpt  # noqa: E402
import qspace_kohn_development as dev  # noqa: E402


OUTDIR = ROOT / "results" / "p0_graphene_qspace"


def candidate(summary: dict, region: str, kind: str) -> dict:
    matches = [
        item
        for item in summary["candidate_scores"][region]
        if item["kind"] == kind
        and (kind == "poly2" or np.isclose(item["width_scale_per_Ry"], 4.0))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {region}/{kind} candidate, found {matches}")
    return matches[0]


def main() -> int:
    archived = json.loads((OUTDIR / "qspace_development_summary.json").read_text())
    if "final holdout is not read" not in archived["target_leakage"]:
        raise RuntimeError("development-only provenance is missing")
    rows = dev.load_development_rows()
    predictions = []
    in_sample = []

    for region in ("G", "K"):
        rounded_meta = candidate(archived, region, "rounded_cusp")
        smooth_meta = candidate(archived, region, "poly2")
        for dg in dev.DEVELOPMENT_DGS:
            data = dev.grouped(rows, region, dg)
            rounded_coeff = dev.fit_coefficients(
                data, "rounded_cusp", rounded_meta["width_scale_per_Ry"]
            )
            smooth_coeff = dev.fit_coefficients(data, "poly2", None)
            rounded = dev.corrected_frequency(
                data,
                "rounded_cusp",
                rounded_meta["width_scale_per_Ry"],
                rounded_coeff,
            )
            smooth = dev.corrected_frequency(data, "poly2", None, smooth_coeff)
            direct = np.asarray([float(row["dft_cm"]) for row in data])
            in_sample.append(
                {
                    "region": region,
                    "degauss_Ry": dg,
                    "rounded_in_sample_MAE_cm-1": float(
                        np.mean(np.abs(rounded - direct))
                    ),
                    "smooth_in_sample_MAE_cm-1": float(
                        np.mean(np.abs(smooth - direct))
                    ),
                }
            )
            for row, rounded_value, smooth_value in zip(data, rounded, smooth):
                predictions.append(
                    {
                        "region": region,
                        "degauss_Ry": dg,
                        "t_GK": float(row["t_GK"]),
                        "direct_DFPT_cm-1": float(row["dft_cm"]),
                        "frozen_baseline_cm-1": float(row["base_cm"]),
                        "rounded_cusp_cm-1": float(rounded_value),
                        "smooth_poly2_cm-1": float(smooth_value),
                    }
                )

    comparison = {}
    for region in ("G", "K"):
        rounded = candidate(archived, region, "rounded_cusp")
        smooth = candidate(archived, region, "poly2")
        comparison[region] = {
            "rounded_cusp": {
                "q_group_CV_MAE_cm-1": rounded["q_group_CV_MAE_cm-1"],
                "smearing_LOOCV_MAE_cm-1": rounded["smearing_LOOCV_MAE_cm-1"],
                "selection_score": rounded["selection_score"],
            },
            "smooth_poly2_control": {
                "q_group_CV_MAE_cm-1": smooth["q_group_CV_MAE_cm-1"],
                "smearing_LOOCV_MAE_cm-1": smooth["smearing_LOOCV_MAE_cm-1"],
                "selection_score": smooth["selection_score"],
            },
        }

    summary = {
        "status": "complete",
        "scope": (
            "development data only; smooth control was predeclared before the "
            "final holdout; this post-hoc plot does not refit to holdout data"
        ),
        "interpretation": (
            "rounded cusp is the physics-constrained primary model; the smooth "
            "control is numerically competitive at Gamma but loses the K "
            "cross-validation comparison"
        ),
        "cross_validation": comparison,
        "in_sample_metrics": in_sample,
    }

    with (OUTDIR / "qspace_ablation_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (OUTDIR / "qspace_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    fig, axes = plt.subplots(2, len(dev.DEVELOPMENT_DGS), figsize=(18.0, 6.8))
    for column, dg in enumerate(dev.DEVELOPMENT_DGS):
        for row_index, region in enumerate(("G", "K")):
            ax = axes[row_index, column]
            data = sorted(
                [
                    row
                    for row in predictions
                    if row["region"] == region and np.isclose(row["degauss_Ry"], dg)
                ],
                key=lambda row: row["t_GK"],
            )
            t = [row["t_GK"] for row in data]
            ax.plot(
                t,
                [row["direct_DFPT_cm-1"] for row in data],
                "-o",
                color="#222222",
                lw=1.5,
                ms=3.5,
                label="direct DFPT",
            )
            ax.plot(
                t,
                [row["frozen_baseline_cm-1"] for row in data],
                "--",
                color="#999999",
                lw=1.2,
                label="frozen baseline",
            )
            ax.plot(
                t,
                [row["rounded_cusp_cm-1"] for row in data],
                "-",
                color="#0072B2",
                lw=1.5,
                label="rounded cusp",
            )
            ax.plot(
                t,
                [row["smooth_poly2_cm-1"] for row in data],
                ":",
                color="#D55E00",
                lw=1.6,
                label="smooth polynomial",
            )
            if region == "K":
                ax.axvline(1.0, color="#cccccc", lw=0.7)
            ax.set_title(f"{region}, dg={dg:g} Ry", fontsize=9)
            ax.set_xlabel(r"$t$ along $q=tK$")
            if column == 0:
                ax.set_ylabel(r"highest optical frequency (cm$^{-1}$)")
            dfpt.style_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=7.0)
    fig.suptitle(
        "Graphene q-space ablation: rounded Kohn cusp vs smooth control "
        "(development data only)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTDIR / f"qspace_ablation.{suffix}", dpi=200)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
