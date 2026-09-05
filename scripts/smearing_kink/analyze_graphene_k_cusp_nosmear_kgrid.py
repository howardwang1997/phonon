#!/usr/bin/env python3
"""Select the production k grid for the no-degauss graphene K cusp."""
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


EXPECTED = {"K", "KG_d007", "KM_d007"}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def dataset(paths: list[Path], expected_kgrid: int, expected_tag: str | None = None) -> dict:
    points = method.load_sources(paths)
    rows, minimum_overlap, provenance = method.track_points(points)
    if {row["label"] for row in rows} != EXPECTED:
        raise ValueError(f"k-grid set does not contain {EXPECTED}")
    if provenance["kgrid"] != expected_kgrid:
        raise ValueError(f"expected k={expected_kgrid}, found {provenance['kgrid']}")
    if expected_tag is not None and provenance["qe_tag"] != expected_tag:
        raise ValueError(f"expected QE tag {expected_tag}, found {provenance['qe_tag']}")
    values = {row["label"]: float(row["frequency_cm-1"]) for row in rows}
    values["KG_cusp_depth_cm-1"] = values["KG_d007"] - values["K"]
    values["KM_cusp_depth_cm-1"] = values["KM_d007"] - values["K"]
    return {
        "values": values,
        "minimum_mode_overlap": minimum_overlap,
        "provenance": {key: value for key, value in provenance.items() if key != "K_reference_eigenvector"},
    }


def comparison(coarse: dict, fine: dict) -> dict:
    labels = ("K", "KG_d007", "KM_d007")
    differences = np.asarray(
        [fine["values"][label] - coarse["values"][label] for label in labels]
    )
    relative_depth = {}
    for direction in ("KG", "KM"):
        key = f"{direction}_cusp_depth_cm-1"
        relative_depth[direction] = float(
            abs(fine["values"][key] - coarse["values"][key])
            / max(abs(fine["values"][key]), 1.0e-12)
        )
    return {
        "signed_differences_cm-1": dict(zip(labels, differences.tolist(), strict=True)),
        "MAE_cm-1": float(np.mean(np.abs(differences))),
        "max_abs_cm-1": float(np.max(np.abs(differences))),
        "K_abs_cm-1": float(abs(differences[0])),
        "cusp_depth_relative_changes": relative_depth,
        "pass": bool(
            np.mean(np.abs(differences)) <= 1.0
            and abs(differences[0]) <= 1.0
            and max(relative_depth.values()) <= 0.10
        ),
    }


def make_figure(data: dict[int, dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.9, 3.9))
    x = np.asarray([-0.007, 0.0, 0.007])
    for kgrid, record in sorted(data.items()):
        values = record["values"]
        y = [values["KM_d007"], values["K"], values["KG_d007"]]
        axis.plot(x, y, "o-", lw=1.35, ms=4.5, label=f"k{kgrid}")
    axis.set_xticks(x, ["K→M", "K", "K→Γ"])
    axis.set_ylabel(r"tracked $A_1'$ frequency (cm$^{-1}$)")
    axis.set_title("No-degauss k-grid convergence")
    axis.grid(axis="y", color="#E6E6E6", lw=0.55)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.79, top=0.90, bottom=0.14)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference192", type=Path, action="append", required=True)
    parser.add_argument("--patched192", type=Path, action="append", required=True)
    parser.add_argument("--k240", type=Path, action="append", required=True)
    parser.add_argument("--k288", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference192 = dataset(args.reference192, 192)
    patched192 = dataset(args.patched192, 192, "qe75_npk120k")
    k240 = dataset(args.k240, 240, "qe75_npk120k")
    replay = comparison(reference192, patched192)
    replay["pass"] = bool(replay["max_abs_cm-1"] <= 0.2)
    k192_to_k240 = comparison(patched192, k240)
    data = {192: patched192, 240: k240}
    k240_to_k288 = None
    if k192_to_k240["pass"]:
        status = "passed_selected_k192"
        selected = 192
        exit_code = 0
    elif args.k288:
        k288 = dataset(args.k288, 288, "qe75_npk120k")
        data[288] = k288
        k240_to_k288 = comparison(k240, k288)
        if k240_to_k288["pass"]:
            status = "passed_selected_k240"
            selected = 240
            exit_code = 0
        else:
            status = "failed_kgrid_not_converged"
            selected = None
            exit_code = 2
    else:
        status = "requires_k288"
        selected = None
        exit_code = 3
    if not replay["pass"]:
        status = "failed_QE_cross_version_replay"
        selected = None
        exit_code = 2
    if min(record["minimum_mode_overlap"] for record in data.values()) < 0.95:
        status = "failed_mode_tracking"
        selected = None
        exit_code = 2

    make_figure(data, args.output_dir / "kgrid_convergence.png")
    summary = {
        "status": status,
        "selected_kgrid": selected,
        "electronic_integration": "tetrahedra_opt (no degauss)",
        "QE_cross_version_k192": replay,
        "k192_to_k240": k192_to_k240,
        "k240_to_k288": k240_to_k288,
        "datasets": data,
        "thresholds": {
            "QE_cross_version_max_abs_cm-1": 0.2,
            "grid_MAE_cm-1": 1.0,
            "grid_K_abs_cm-1": 1.0,
            "cusp_depth_relative_change": 0.10,
            "minimum_mode_overlap": 0.95,
        },
    }
    atomic_json(args.output_dir / "kgrid_summary.json", summary)
    if selected is not None:
        (args.output_dir / "SELECTED_KGRID").write_text(f"{selected}\n", encoding="utf-8")
        (args.output_dir / "PASS").write_text(status + "\n", encoding="utf-8")
    elif exit_code == 3:
        (args.output_dir / "NEEDS_K288").write_text(status + "\n", encoding="utf-8")
    else:
        (args.output_dir / "FAILED").write_text(status + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
