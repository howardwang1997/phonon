#!/usr/bin/env python3
"""Audit whether the no-degauss d=0.003 K-cusp points are k-grid converged."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_k_cusp_nosmear_method as method  # noqa: E402


EXPECTED_DIAGNOSTIC = {"KG_d003", "KM_d003"}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def k192_values(path: Path) -> tuple[float, dict[str, float]]:
    rows = read_csv(path)
    selected = {
        row["label"]: float(row["frequency_cm-1"])
        for row in rows
        if row["label"] in EXPECTED_DIAGNOSTIC
    }
    if set(selected) != EXPECTED_DIAGNOSTIC:
        raise ValueError("k192 holdout CSV lacks the two d=0.003 points")
    anchors = {float(row["K_frequency_cm-1"]) for row in rows}
    if len(anchors) != 1:
        raise ValueError("k192 holdout CSV has inconsistent K anchors")
    return anchors.pop(), selected


def k240_values(paths: list[Path]) -> tuple[float, dict[str, float], float, dict]:
    points = method.load_sources(paths)
    rows, minimum_overlap, provenance = method.track_points(points)
    if provenance["kgrid"] != 240 or provenance["qe_tag"] != "qe75_npk120k":
        raise ValueError("diagnostic sources are not the released k240/QE dataset")
    selected = {
        row["label"]: float(row["frequency_cm-1"])
        for row in rows
        if row["label"] in EXPECTED_DIAGNOSTIC
    }
    if set(selected) != EXPECTED_DIAGNOSTIC:
        raise ValueError("k240 sources lack the two d=0.003 points")
    k_value = next(
        float(row["frequency_cm-1"])
        for row in rows
        if row["label"] == "K"
    )
    provenance.pop("K_reference_eigenvector")
    return k_value, selected, minimum_overlap, provenance


def make_figure(comparisons: dict[str, dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.8, 3.8))
    colors = {"KG": "#1F77B4", "KM": "#D55E00"}
    for direction, label in (("KG", "K→Γ"), ("KM", "K→M")):
        record = comparisons[direction]
        axis.plot(
            [0.0, 0.003],
            [0.0, record["k192_cusp_depth_cm-1"]],
            "o--",
            color=colors[direction],
            lw=1.25,
            ms=4.5,
            alpha=0.65,
            label=f"{label} k192",
        )
        axis.plot(
            [0.0, 0.003],
            [0.0, record["k240_cusp_depth_cm-1"]],
            "s-",
            color=colors[direction],
            lw=1.5,
            ms=4.5,
            label=f"{label} k240",
        )
    axis.set_xlabel(r"equal-distance coordinate $d$ from K")
    axis.set_ylabel(r"frequency relative to K (cm$^{-1}$)")
    axis.set_title("No-degauss d=0.003 k-grid diagnostic")
    axis.grid(axis="y", color="#E6E6E6", lw=0.55)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.70, top=0.89, bottom=0.16)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k192-holdout-csv", type=Path, required=True)
    parser.add_argument("--k240-source", type=Path, action="append", required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    release = json.loads(args.release.read_text(encoding="utf-8"))
    if int(release["diagnostic_kgrid"]) != 240:
        raise ValueError("release does not authorize k240")
    k192_k, k192 = k192_values(args.k192_holdout_csv)
    k240_k, k240, minimum_overlap, provenance = k240_values(args.k240_source)
    comparisons = {}
    checks = {"minimum_mode_overlap_ge_0p95": minimum_overlap >= 0.95}
    for direction in ("KG", "KM"):
        label = f"{direction}_d003"
        depth192 = k192[label] - k192_k
        depth240 = k240[label] - k240_k
        frequency_change = k240[label] - k192[label]
        depth_relative_change = abs(depth240 - depth192) / max(abs(depth240), 1.0e-12)
        comparisons[direction] = {
            "k192_K_frequency_cm-1": k192_k,
            "k240_K_frequency_cm-1": k240_k,
            "k192_frequency_cm-1": k192[label],
            "k240_frequency_cm-1": k240[label],
            "signed_frequency_change_cm-1": frequency_change,
            "frequency_abs_change_cm-1": abs(frequency_change),
            "k192_cusp_depth_cm-1": depth192,
            "k240_cusp_depth_cm-1": depth240,
            "cusp_depth_relative_change": depth_relative_change,
        }
        checks[f"{direction}_frequency_change_le_1_cm-1"] = abs(frequency_change) <= 1.0
        checks[f"{direction}_depth_change_le_10pct"] = depth_relative_change <= 0.10

    passed = all(checks.values())
    status = "k192_d003_converged_model_shape_failure" if passed else "requires_k288_d003"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_figure(comparisons, args.output_dir / "d003_kgrid_diagnostic.png")
    summary = {
        "status": status,
        "electronic_integration": "tetrahedra_opt (no degauss)",
        "diagnostic": "k192 versus k240 at the two d=0.003 blind-failure points",
        "comparisons": comparisons,
        "minimum_mode_overlap": minimum_overlap,
        "checks": checks,
        "thresholds": {
            "frequency_abs_change_cm-1": 1.0,
            "cusp_depth_relative_change": 0.10,
            "minimum_mode_overlap": 0.95,
        },
        "sources": {
            "k192_holdout_csv": {
                "path": str(args.k192_holdout_csv),
                "sha256": method.digest(args.k192_holdout_csv),
            },
            "release": {
                "path": str(args.release),
                "sha256": method.digest(args.release),
            },
            "k240": provenance,
        },
        "next_action": (
            "fit a two-scale q-space cusp law, then use a new untouched holdout"
            if passed
            else "do not refit; compute the same K and d=0.003 points at k288"
        ),
    }
    atomic_json(args.output_dir / "d003_kgrid_summary.json", summary)
    marker = args.output_dir / ("PASS" if passed else "NEEDS_K288")
    marker.write_text(status + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
