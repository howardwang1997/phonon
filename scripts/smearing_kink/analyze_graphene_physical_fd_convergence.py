"""Analyze k-mesh convergence of physical-smearing graphene direct-DFPT."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(paths: list[Path]) -> tuple[list[dict], list[str]]:
    rows = []
    frequency_columns: list[str] | None = None
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            current_columns = [
                name for name in (reader.fieldnames or []) if name.startswith("f") and "cm" in name
            ]
            if len(current_columns) != 6:
                raise ValueError(f"{path}: expected six frequency columns, got {current_columns}")
            if frequency_columns is None:
                frequency_columns = current_columns
            elif len(frequency_columns) != len(current_columns):
                raise ValueError("inconsistent frequency columns")
            for raw in reader:
                values = np.array([float(raw[name]) for name in current_columns], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}: non-finite frequencies in {raw}")
                rows.append(
                    {
                        "source": str(path),
                        "campaign": raw.get("campaign", raw.get("lane", "unknown")),
                        "degauss_Ry": float(raw["degauss_Ry"]),
                        "kgrid": int(raw["kgrid"]),
                        "region": raw["region"],
                        "t_GK": float(raw["t_GK"]),
                        "frequencies_cm-1": values,
                    }
                )
    if not rows or frequency_columns is None:
        raise ValueError("no DFPT rows loaded")
    return rows, frequency_columns


def comparison(lower: dict, upper: dict, region: str) -> dict:
    keys = sorted(set(lower) & set(upper))
    keys = [key for key in keys if key[0] == region]
    if not keys:
        return {"n_q": 0}
    delta = np.stack([upper[key] - lower[key] for key in keys])
    top = delta[:, -1]
    return {
        "n_q": len(keys),
        "frequency_RMSE_cm-1": float(np.sqrt(np.mean(delta**2))),
        "frequency_max_abs_cm-1": float(np.max(np.abs(delta))),
        "top_branch_RMSE_cm-1": float(np.sqrt(np.mean(top**2))),
        "top_branch_max_abs_cm-1": float(np.max(np.abs(top))),
        "top_branch_delta_by_t_cm-1": {
            f"{key[1]:.6f}": float(delta[index, -1]) for index, key in enumerate(keys)
        },
    }


def nearest_value(mapping: dict[tuple[str, float], np.ndarray], region: str, t: float):
    candidates = [(abs(key_t - t), values) for (key_region, key_t), values in mapping.items() if key_region == region]
    if not candidates:
        return None
    distance, values = min(candidates, key=lambda item: item[0])
    return values if distance < 1.0e-7 else None


def anomaly_metrics(mapping: dict[tuple[str, float], np.ndarray]) -> dict:
    result = {}
    gamma = [nearest_value(mapping, "G", t) for t in (0.0, 0.015, 0.030)]
    if all(value is not None for value in gamma):
        f0, f15, f30 = [value[-1] for value in gamma]
        result["Gamma_top_discrete_curvature_cm-1"] = float(f0 - 2.0 * f15 + f30)
        result["Gamma_top_0_to_0.030_change_cm-1"] = float(f30 - f0)
    k_values = [nearest_value(mapping, "K", t) for t in (0.990, 1.000, 1.010)]
    if all(value is not None for value in k_values):
        fm, f0, fp = [value[-1] for value in k_values]
        result["K_top_kink_cm-1"] = float(0.5 * (fm + fp) - f0)
        result["K_top_left_right_asymmetry_cm-1"] = float((fp - f0) - (f0 - fm))
    return result


def anomaly_comparison(lower: dict, upper: dict, thresholds: dict[str, float]) -> dict:
    deltas = {
        name: float(upper[name] - lower[name])
        for name in sorted(set(lower) & set(upper))
    }
    checks = {
        "Gamma_top_discrete_curvature_cm-1": "Gamma_curvature_abs_change",
        "Gamma_top_0_to_0.030_change_cm-1": "Gamma_0_to_0.030_abs_change",
        "K_top_kink_cm-1": "K_kink_abs_change",
    }
    passed = bool(deltas) and all(
        name in deltas and abs(deltas[name]) <= thresholds[threshold_name]
        for name, threshold_name in checks.items()
    )
    return {
        "delta_upper_minus_lower_cm-1": deltas,
        "passes_anomaly_thresholds": passed,
    }


def analyze(rows: list[dict]) -> dict:
    by_degauss: dict[float, dict[int, dict[tuple[str, float], np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        key = (row["region"], row["t_GK"])
        target = by_degauss[row["degauss_Ry"]][row["kgrid"]]
        if key in target:
            raise ValueError(
                f"duplicate row degauss={row['degauss_Ry']} k={row['kgrid']} key={key}"
            )
        target[key] = row["frequencies_cm-1"]

    result = {
        "thresholds_cm-1": {
            "all_branch_max_abs": 3.0,
            "top_branch_max_abs": 2.0,
            "Gamma_curvature_abs_change": 0.3,
            "Gamma_0_to_0.030_abs_change": 0.5,
            "K_kink_abs_change": 0.3,
        },
        "by_degauss": {},
    }
    for degauss, by_k in sorted(by_degauss.items()):
        kgrids = sorted(by_k)
        entry = {
            "kgrids": kgrids,
            "n_q_by_k": {str(kgrid): len(by_k[kgrid]) for kgrid in kgrids},
            "anomaly_metrics_by_k": {
                str(kgrid): anomaly_metrics(by_k[kgrid]) for kgrid in kgrids
            },
            "adjacent_k_comparisons": [],
        }
        for lower_k, upper_k in zip(kgrids[:-1], kgrids[1:]):
            regions = {
                region: comparison(by_k[lower_k], by_k[upper_k], region)
                for region in ("G", "K")
            }
            valid = [value for value in regions.values() if value.get("n_q", 0)]
            passed = bool(valid) and all(
                value["frequency_max_abs_cm-1"]
                <= result["thresholds_cm-1"]["all_branch_max_abs"]
                and value["top_branch_max_abs_cm-1"]
                <= result["thresholds_cm-1"]["top_branch_max_abs"]
                for value in valid
            )
            anomaly_delta = anomaly_comparison(
                entry["anomaly_metrics_by_k"][str(lower_k)],
                entry["anomaly_metrics_by_k"][str(upper_k)],
                result["thresholds_cm-1"],
            )
            entry["adjacent_k_comparisons"].append(
                {
                    "lower_kgrid": lower_k,
                    "upper_kgrid": upper_k,
                    "regions": regions,
                    "passes_frequency_thresholds": passed,
                    "anomaly_comparison": anomaly_delta,
                }
            )
        latest = entry["adjacent_k_comparisons"][-1] if entry["adjacent_k_comparisons"] else None
        entry["largest_kgrid"] = max(kgrids)
        entry["current_candidate_converged"] = bool(
            latest
            and latest["passes_frequency_thresholds"]
            and latest["anomaly_comparison"]["passes_anomaly_thresholds"]
            and max(kgrids) >= 120
        )
        entry["k144_confirmation_required"] = bool(
            max(kgrids) < 144 and not entry["current_candidate_converged"]
        )
        entry["recommended_next_campaign"] = (
            "dense line cut"
            if entry["current_candidate_converged"]
            else (
                "k144 anchor confirmation"
                if max(kgrids) < 144
                else "increase k mesh beyond 144"
            )
        )
        result["by_degauss"][f"{degauss:.10f}"] = entry
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, _ = load_rows(args.input)
    payload = analyze(rows)
    payload["inputs"] = [str(path) for path in args.input]
    payload["n_rows"] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
