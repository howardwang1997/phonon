#!/usr/bin/env python3
"""Build a degauss-to-zero graphene reference from the low-smearing scan."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows in {path}, found {len(rows)}")
    return rows


def positive_frequency(squared: float) -> float:
    return float(np.sqrt(max(float(squared), 0.0)))


def extrapolate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["region"], float(row["t_GK"]))].append(row)

    records: list[dict[str, object]] = []
    for (region, t_value), group in sorted(grouped.items()):
        group.sort(key=lambda row: float(row["degauss_Ry"]))
        if len(group) != 3:
            raise ValueError(f"{region} t={t_value} has {len(group)} smearing values")
        degauss = np.array([float(row["degauss_Ry"]) for row in group])
        frequency = np.array([float(row["f6_cm-1"]) for row in group])
        squared = frequency**2
        linear = np.polyfit(degauss, squared, 1)
        even = np.polyfit(degauss**2, squared, 1)
        intercept_linear = positive_frequency(linear[-1])
        intercept_even = positive_frequency(even[-1])
        estimate = 0.5 * (intercept_linear + intercept_even)
        records.append(
            {
                "region": region,
                "t_GK": t_value,
                "degauss_Ry": degauss.tolist(),
                "direct_DFPT_cm-1": frequency.tolist(),
                "kgrids": [int(row["kgrid"]) for row in group],
                "zero_linear_in_degauss_cm-1": intercept_linear,
                "zero_linear_in_degauss_squared_cm-1": intercept_even,
                "zero_central_cm-1": estimate,
                "extrapolation_model_spread_cm-1": abs(
                    intercept_linear - intercept_even
                ),
            }
        )
    return records


def line_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, float], float]:
    return {
        (row["region"], round(float(row["t_GK"]), 6)): float(row["f6_cm-1"])
        for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd0", type=Path, required=True)
    parser.add_argument("--fd300", type=Path, required=True)
    parser.add_argument("--fd600", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scan = load_rows(args.fd0, 15)
    line300 = load_rows(args.fd300, 19)
    line600 = load_rows(args.fd600, 19)
    records = extrapolate(scan)
    lookup300 = line_lookup(line300)
    lookup600 = line_lookup(line600)
    overlap_errors = []
    for record in records:
        key = (str(record["region"]), round(float(record["t_GK"]), 6))
        record["physical_300K_smearing_cm-1"] = lookup300[key]
        record["physical_600K_smearing_cm-1"] = lookup600[key]
        # The largest smearing point in FD0_SCAN has the 300 K degauss but a
        # k=120 mesh.  Comparing it with FD300_LINE (k=144) checks that the
        # extrapolation is not hiding a material k-grid error.
        scan_300 = float(record["direct_DFPT_cm-1"][-1])
        overlap_error = abs(scan_300 - lookup300[key])
        record["same_smearing_k120_to_k144_abs_cm-1"] = overlap_error
        overlap_errors.append(overlap_error)

    zero_lookup = {
        (str(record["region"]), round(float(record["t_GK"]), 3)): float(
            record["zero_central_cm-1"]
        )
        for record in records
    }
    dx = 0.015
    zero_kink = abs(
        (zero_lookup[("K", 1.015)] - zero_lookup[("K", 1.000)]) / dx
        - (zero_lookup[("K", 1.000)] - zero_lookup[("K", 0.985)]) / dx
    )
    max_spread = max(float(record["extrapolation_model_spread_cm-1"]) for record in records)
    max_overlap = max(overlap_errors)
    summary = {
        "method": (
            "central estimate is the mean of squared-frequency intercepts from "
            "linear-in-degauss and linear-in-degauss-squared fits"
        ),
        "smearing": "fermi-dirac",
        "input_degauss_Ry": sorted({float(row["degauss_Ry"]) for row in scan}),
        "zero_K_kink_cm-1_per_t": zero_kink,
        "max_extrapolation_model_spread_cm-1": max_spread,
        "max_same_smearing_k120_to_k144_abs_cm-1": max_overlap,
        "diagnostic_thresholds_cm-1": {
            "extrapolation_model_spread": 20.0,
            "same_smearing_kgrid_overlap": 5.0,
        },
        "passes_extrapolation_diagnostics": max_spread <= 20.0 and max_overlap <= 5.0,
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "graphene_fd0_extrapolation.json"
    temporary = json_path.with_name(json_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, json_path)
    csv_path = args.output_dir / "graphene_fd0_extrapolation.csv"
    fieldnames = (
        "region",
        "t_GK",
        "zero_central_cm-1",
        "zero_linear_in_degauss_cm-1",
        "zero_linear_in_degauss_squared_cm-1",
        "extrapolation_model_spread_cm-1",
        "physical_300K_smearing_cm-1",
        "physical_600K_smearing_cm-1",
        "same_smearing_k120_to_k144_abs_cm-1",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    colors = ("#999999", "#56B4E9", "#0072B2")
    degauss_values = summary["input_degauss_Ry"]
    for axis, region in zip(axes, ("G", "K")):
        selected = sorted(
            (record for record in records if record["region"] == region),
            key=lambda record: float(record["t_GK"]),
        )
        for index, degauss in enumerate(degauss_values):
            axis.plot(
                [record["t_GK"] for record in selected],
                [record["direct_DFPT_cm-1"][index] for record in selected],
                "o-",
                color=colors[index],
                label=f"smearing={degauss:.10f} Ry",
            )
        axis.plot(
            [record["t_GK"] for record in selected],
            [record["zero_central_cm-1"] for record in selected],
            "s--",
            color="#D55E00",
            label="degauss→0 extrapolation",
        )
        axis.set_title(f"{region} line")
        axis.set_xlabel("t along q=tK")
        axis.set_ylabel(r"top optical frequency (cm$^{-1}$)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "graphene_fd0_extrapolation.png", dpi=200)
    fig.savefig(args.output_dir / "graphene_fd0_extrapolation.pdf")
    plt.close(fig)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
