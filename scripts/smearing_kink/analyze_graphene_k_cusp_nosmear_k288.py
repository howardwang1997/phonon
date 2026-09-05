#!/usr/bin/env python3
"""Compare the no-degauss d=0.003 K cusp at k240 and k288."""
from __future__ import annotations

import argparse
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


EXPECTED = {"KG_d003", "KM_d003"}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def grid_values(
    paths: list[Path], expected_grid: int
) -> tuple[float, dict[str, float], float, np.ndarray, dict]:
    points = method.load_sources(paths)
    rows, minimum_overlap, provenance = method.track_points(points)
    if provenance["kgrid"] != expected_grid or provenance["qe_tag"] != "qe75_npk120k":
        raise ValueError(
            f"expected k{expected_grid}/qe75_npk120k, got "
            f"k{provenance['kgrid']}/{provenance['qe_tag']}"
        )
    selected = {
        row["label"]: float(row["frequency_cm-1"])
        for row in rows
        if row["label"] in EXPECTED
    }
    if set(selected) != EXPECTED:
        raise ValueError(f"k{expected_grid} sources lack the two d=0.003 points")
    k_rows = [row for row in rows if row["label"] == "K"]
    if len(k_rows) != 1:
        raise ValueError(f"k{expected_grid} sources do not define one tracked K anchor")
    reference = np.asarray(provenance.pop("K_reference_eigenvector"))
    return (
        float(k_rows[0]["frequency_cm-1"]),
        selected,
        minimum_overlap,
        reference,
        provenance,
    )


def make_figure(comparisons: dict[str, dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.8, 3.8))
    colors = {"KG": "#1F77B4", "KM": "#D55E00"}
    for direction, label in (("KG", "K→Γ"), ("KM", "K→M")):
        record = comparisons[direction]
        axis.plot(
            [0.0, 0.003],
            [0.0, record["k240_cusp_depth_cm-1"]],
            "o--",
            color=colors[direction],
            lw=1.25,
            ms=4.5,
            alpha=0.65,
            label=f"{label} k240",
        )
        axis.plot(
            [0.0, 0.003],
            [0.0, record["k288_cusp_depth_cm-1"]],
            "s-",
            color=colors[direction],
            lw=1.5,
            ms=4.5,
            label=f"{label} k288",
        )
    axis.set_xlabel(r"equal-distance coordinate $d$ from K")
    axis.set_ylabel(r"frequency relative to K (cm$^{-1}$)")
    axis.set_title("No-degauss d=0.003: k240 vs k288")
    axis.grid(axis="y", color="#E6E6E6", lw=0.55)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.70, top=0.89, bottom=0.16)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k240-source", type=Path, action="append", required=True)
    parser.add_argument("--k288-source", type=Path, action="append", required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    release = json.loads(args.release.read_text(encoding="utf-8"))
    if int(release["diagnostic_kgrid"]) != 288:
        raise ValueError("release does not authorize k288")

    k240_k, k240, overlap240, reference240, provenance240 = grid_values(
        args.k240_source, 240
    )
    k288_k, k288, overlap288, reference288, provenance288 = grid_values(
        args.k288_source, 288
    )
    cross_grid_k_overlap = method.vector_overlap(reference240, reference288)

    comparisons = {}
    checks = {
        "k240_minimum_mode_overlap_ge_0p95": overlap240 >= 0.95,
        "k288_minimum_mode_overlap_ge_0p95": overlap288 >= 0.95,
        "k240_to_k288_K_mode_overlap_ge_0p95": cross_grid_k_overlap >= 0.95,
        "k288_cross_host_K_all_mode_max_abs_le_1e-5_cm-1": (
            provenance288["cross_host_K_all_mode_max_abs_cm-1"] <= 1.0e-5
        ),
    }
    for direction in ("KG", "KM"):
        label = f"{direction}_d003"
        depth240 = k240[label] - k240_k
        depth288 = k288[label] - k288_k
        frequency_change = k288[label] - k240[label]
        depth_relative_change = abs(depth288 - depth240) / max(abs(depth288), 1.0e-12)
        comparisons[direction] = {
            "k240_K_frequency_cm-1": k240_k,
            "k288_K_frequency_cm-1": k288_k,
            "k240_frequency_cm-1": k240[label],
            "k288_frequency_cm-1": k288[label],
            "signed_frequency_change_cm-1": frequency_change,
            "frequency_abs_change_cm-1": abs(frequency_change),
            "k240_cusp_depth_cm-1": depth240,
            "k288_cusp_depth_cm-1": depth288,
            "cusp_depth_relative_change": depth_relative_change,
        }
        checks[f"{direction}_frequency_change_le_1_cm-1"] = abs(frequency_change) <= 1.0
        checks[f"{direction}_depth_change_le_10pct"] = depth_relative_change <= 0.10

    passed = all(checks.values())
    status = "k240_d003_converged_at_k288" if passed else "k288_still_unconverged"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_figure(comparisons, args.output_dir / "d003_k240_k288_diagnostic.png")
    summary = {
        "status": status,
        "electronic_integration": "tetrahedra_opt (no degauss)",
        "diagnostic": "k240 versus k288 at K and the two d=0.003 points",
        "comparisons": comparisons,
        "mode_tracking": {
            "k240_minimum_overlap": overlap240,
            "k288_minimum_overlap": overlap288,
            "k240_to_k288_K_mode_overlap": cross_grid_k_overlap,
        },
        "checks": checks,
        "thresholds": {
            "frequency_abs_change_cm-1": 1.0,
            "cusp_depth_relative_change": 0.10,
            "minimum_mode_overlap": 0.95,
            "cross_host_K_all_mode_max_abs_cm-1": 1.0e-5,
        },
        "sources": {
            "release": {"path": str(args.release), "sha256": method.digest(args.release)},
            "k240": provenance240,
            "k288": provenance288,
        },
        "next_action": (
            "use the k288 K/d=0.003 values as the highest-grid reference and freeze the finite-slope long-range model"
            if passed
            else "do not refit; design a denser-grid or explicit convergence-extrapolation study"
        ),
    }
    atomic_json(args.output_dir / "d003_k240_k288_summary.json", summary)
    marker = args.output_dir / ("PASS" if passed else "STILL_UNCONVERGED")
    marker.write_text(status + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
