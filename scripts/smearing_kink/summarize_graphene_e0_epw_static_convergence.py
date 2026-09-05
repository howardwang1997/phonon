#!/usr/bin/env python3
"""Summarize EPW static k-grid and numerical-broadening convergence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graphene_fd_p4_common import atomic_json, sha256


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "t_GK",
            "target_top_cm-1",
            "transferred_static_top_cm-1",
        ):
            row[key] = float(row[key])
    return rows


def difference_metrics(left: list[dict], right: list[dict]) -> dict:
    left_lookup = {
        (row["region"], round(row["t_GK"], 10)): row for row in left
    }
    records = []
    for row in right:
        key = (row["region"], round(row["t_GK"], 10))
        reference = left_lookup[key]
        records.append(
            {
                "region": row["region"],
                "t_GK": row["t_GK"],
                "difference_cm-1": row["transferred_static_top_cm-1"]
                - reference["transferred_static_top_cm-1"],
            }
        )
    result = {"records": records}
    for region in ("all", "G", "K"):
        values = np.asarray(
            [
                record["difference_cm-1"]
                for record in records
                if region == "all" or record["region"] == region
            ],
            float,
        )
        result[region] = {
            "MAE_cm-1": float(np.mean(np.abs(values))),
            "max_abs_cm-1": float(np.max(np.abs(values))),
        }
    result["passes_inherited_1_cm-1_pointwise_gate"] = (
        result["all"]["max_abs_cm-1"] < 1.0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("k360_eta010", "k720_eta010", "k720_eta005"):
        parser.add_argument(f"--{name.replace('_', '-')}-summary", type=Path, required=True)
        parser.add_argument(f"--{name.replace('_', '-')}-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    names = ("k360_eta010", "k720_eta010", "k720_eta005")
    summaries = {
        name: json.loads(getattr(args, f"{name}_summary").read_text()) for name in names
    }
    rows = {name: load_csv(getattr(args, f"{name}_csv")) for name in names}
    kgrid = difference_metrics(rows["k360_eta010"], rows["k720_eta010"])
    broadening = difference_metrics(rows["k720_eta010"], rows["k720_eta005"])

    run_metrics = {}
    for name in names:
        metric = summaries[name]["metrics"]["transferred_static_top_cm-1"]
        run_metrics[name] = {
            "fine_k": summaries[name]["mesh"]["fine_k"],
            "numerical_broadening_eV": summaries[name]["metadata"][
                "gaussian_broadening_eV"
            ],
            "G_line_MAE_cm-1": metric["G"]["line_MAE_cm-1"],
            "G_high_symmetry_abs_error_cm-1": metric["G"][
                "high_symmetry_abs_error_cm-1"
            ],
            "K_line_MAE_cm-1": metric["K"]["line_MAE_cm-1"],
            "K_high_symmetry_abs_error_cm-1": metric["K"][
                "high_symmetry_abs_error_cm-1"
            ],
            "K_kink_relative_error": metric["K"]["kink_relative_error"],
            "passes_static_spectral_gate": summaries[name][
                "passes_static_spectral_gate"
            ],
        }

    final_pass = bool(summaries["k720_eta005"]["passes_static_spectral_gate"])
    numerically_converged = bool(
        kgrid["passes_inherited_1_cm-1_pointwise_gate"]
        and broadening["passes_inherited_1_cm-1_pointwise_gate"]
    )
    if final_pass and numerically_converged:
        conclusion = "static_spectral_gate_passed"
    elif numerically_converged:
        conclusion = "current_epw3_representation_fails_static_spectral_gate"
    else:
        conclusion = "static_band_sum_not_yet_numerically_converged"

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete",
        "scope": "450 K EPW zero-frequency static self-energy convergence",
        "run_metrics": run_metrics,
        "kgrid_360_to_720_at_0.010eV": kgrid,
        "numerical_broadening_0.010_to_0.005eV_at_k720": broadening,
        "numerically_converged_under_inherited_pointwise_gate": numerically_converged,
        "passes_final_static_spectral_gate": final_pass,
        "conclusion": conclusion,
        "limitations": [
            "The 1 cm-1 pointwise convergence threshold is inherited from the direct-DFPT k-grid check; it was not introduced as a new spectral acceptance gate.",
            "The reused EPW3 representation has coarse k=12, q=6 and a 29.5 A vacuum, whereas the target direct-DFPT cell has a 15 A vacuum.",
            "A replayable force operator still requires a complete Hermitian q-mesh matrix and a real-space transform even if the sparse spectral gate passes.",
        ],
        "sources": [
            {
                "path": str(getattr(args, f"{name}_{kind}")),
                "sha256": sha256(getattr(args, f"{name}_{kind}")),
            }
            for name in names
            for kind in ("summary", "csv")
        ],
    }
    atomic_json(output / "static_convergence_summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    styles = {
        "k360_eta010": ("s--", "k=360, numerical broadening 0.010 eV"),
        "k720_eta010": ("^--", "k=720, numerical broadening 0.010 eV"),
        "k720_eta005": ("v--", "k=720, numerical broadening 0.005 eV"),
    }
    for axis, region in zip(axes, ("G", "K"), strict=True):
        selected_target = [row for row in rows["k720_eta005"] if row["region"] == region]
        axis.plot(
            [row["t_GK"] for row in selected_target],
            [row["target_top_cm-1"] for row in selected_target],
            "o-",
            label="DFPT 450 K",
        )
        for name in names:
            selected = [row for row in rows[name] if row["region"] == region]
            style, label = styles[name]
            axis.plot(
                [row["t_GK"] for row in selected],
                [row["transferred_static_top_cm-1"] for row in selected],
                style,
                label=label,
            )
        axis.set_title(f"{region} region")
        axis.set_xlabel("t along Γ–K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("top branch (cm$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    fig.savefig(output / "static_convergence.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
