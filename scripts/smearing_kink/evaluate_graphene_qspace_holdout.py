#!/usr/bin/env python3
"""Audit and evaluate the frozen graphene q-space final holdout exactly once."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from qe_dyn import load_qe_dyn


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results" / "p0_graphene_qspace"
HOLDOUT_DIR = OUTDIR / "holdout_dfpt"
CM_PER_THz = 33.356
PRIMARY_DGS = (0.013, 0.027, 0.055)
EXPECTED_T = {
    "G": (0.000, 0.012, 0.025, 0.045, 0.075),
    "K": (0.925, 0.955, 0.975, 0.990, 1.000, 1.010, 1.025, 1.045, 1.075),
}
K64_T = (0.975, 0.990, 1.000, 1.010, 1.025)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(dg: float, kgrid: int, region: str, t_value: float):
    return round(float(dg), 12), int(kgrid), region, round(float(t_value), 12)


def load_direct_rows():
    rows = []
    for campaign in ("HOLD_A", "HOLD_B"):
        campaign_dir = HOLDOUT_DIR / campaign
        csv_path = campaign_dir / f"dfpt_{campaign}.csv"
        if not (campaign_dir / "DONE").is_file() or not csv_path.is_file():
            raise RuntimeError(f"incomplete holdout mirror: {campaign_dir}")
        with csv_path.open() as handle:
            for raw in csv.DictReader(handle):
                if raw["campaign"] != campaign:
                    raise RuntimeError(f"campaign mismatch in {csv_path}: {raw}")
                frequencies = np.array(
                    [float(raw[f"f{i}_cm"]) for i in range(1, 7)]
                )
                if not np.isfinite(frequencies).all():
                    raise RuntimeError(f"non-finite direct row: {raw}")
                rows.append(
                    {
                        "campaign": campaign,
                        "degauss_Ry": float(raw["degauss_Ry"]),
                        "kgrid": int(raw["kgrid"]),
                        "region": raw["region"],
                        "t_GK": float(raw["t_GK"]),
                        "frequencies_cm": frequencies,
                    }
                )
    lookup = {
        key(
            row["degauss_Ry"],
            row["kgrid"],
            row["region"],
            row["t_GK"],
        ): row
        for row in rows
    }
    if len(lookup) != len(rows) or len(rows) != 47:
        raise RuntimeError(
            f"expected 47 unique direct holdout rows, found {len(rows)}"
        )
    expected = set()
    for dg in PRIMARY_DGS:
        for region, t_values in EXPECTED_T.items():
            expected.update(key(dg, 32, region, t) for t in t_values)
    expected.update(key(0.013, 64, "K", t) for t in K64_T)
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise RuntimeError(f"holdout grid mismatch; missing={missing}, extra={extra}")
    return rows, lookup


def audit_dyn(direct_lookup):
    records = []
    paths = sorted(HOLDOUT_DIR.glob("HOLD_*/*_k*/[GK]_t*/gr.dyn"))
    if len(paths) != 47:
        raise RuntimeError(f"expected 47 holdout gr.dyn files, found {len(paths)}")
    for path in paths:
        campaign = path.parents[2].name
        set_match = re.fullmatch(r"dg(.+)_k(\d+)", path.parents[1].name)
        q_match = re.fullmatch(r"([GK])_t(.+)", path.parent.name)
        if campaign not in ("HOLD_A", "HOLD_B") or not set_match or not q_match:
            raise RuntimeError(f"unexpected holdout path: {path}")
        dg = float(set_match.group(1))
        kgrid = int(set_match.group(2))
        region = q_match.group(1)
        t_value = float(q_match.group(2).replace("p", "."))
        row_key = key(dg, kgrid, region, t_value)
        if row_key not in direct_lookup:
            raise RuntimeError(f"matrix has no direct CSV row: {path}")
        dyn = load_qe_dyn(path)
        csv_error = float(
            np.max(
                np.abs(
                    dyn.frequencies_cm
                    - direct_lookup[row_key]["frequencies_cm"]
                )
            )
        )
        matrix_error = float(
            np.max(
                np.abs(dyn.frequencies_from_matrix_cm() - dyn.frequencies_cm)
            )
        )
        records.append(
            {
                "campaign": campaign,
                "degauss_Ry": dg,
                "kgrid": kgrid,
                "region": region,
                "t_GK": t_value,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "hermitian_max_abs": dyn.hermitian_error,
                "eigenvector_orthogonality_max_abs": (
                    dyn.eigenvector_orthogonality_error
                ),
                "csv_frequency_max_abs_cm-1": csv_error,
                "matrix_frequency_max_abs_cm-1": matrix_error,
            }
        )
    summary = {
        "status": "complete",
        "n_files": len(records),
        "n_primary_k32": sum(row["kgrid"] == 32 for row in records),
        "n_convergence_k64": sum(row["kgrid"] == 64 for row in records),
        "max_hermitian_error": max(row["hermitian_max_abs"] for row in records),
        "max_eigenvector_orthogonality_error": max(
            row["eigenvector_orthogonality_max_abs"] for row in records
        ),
        "max_csv_frequency_error_cm-1": max(
            row["csv_frequency_max_abs_cm-1"] for row in records
        ),
        "max_matrix_frequency_error_cm-1": max(
            row["matrix_frequency_max_abs_cm-1"] for row in records
        ),
    }
    if summary["max_hermitian_error"] > 5e-7:
        raise RuntimeError(f"holdout Hermiticity audit failed: {summary}")
    if summary["max_eigenvector_orthogonality_error"] > 5e-5:
        raise RuntimeError(f"holdout eigenvector audit failed: {summary}")
    if summary["max_csv_frequency_error_cm-1"] > 1e-6:
        raise RuntimeError(f"holdout CSV replay failed: {summary}")
    if summary["max_matrix_frequency_error_cm-1"] > 0.1:
        raise RuntimeError(f"holdout matrix replay failed: {summary}")
    return records, summary


def load_prediction_rows(path: Path, frequency_field: str):
    rows = {}
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            row_key = key(
                float(raw["degauss_Ry"]),
                32,
                raw["region"],
                float(raw["t_GK"]),
            )
            if row_key in rows:
                raise RuntimeError(f"duplicate prediction row in {path}: {row_key}")
            rows[row_key] = {
                "frequency_cm": float(raw[frequency_field]),
                "raw": raw,
            }
    if len(rows) != 42:
        raise RuntimeError(f"expected 42 rows in {path}, found {len(rows)}")
    return rows


def slope_jump(t_values: np.ndarray, frequencies: np.ndarray) -> float:
    left = np.polyfit(t_values[t_values < 1.0], frequencies[t_values < 1.0], 1)
    right = np.polyfit(
        t_values[t_values > 1.0], frequencies[t_values > 1.0], 1
    )
    return float(abs(right[0] - left[0]) / CM_PER_THz)


def main() -> int:
    freeze_path = OUTDIR / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    if (
        freeze["status"] != "frozen_before_holdout"
        or freeze["holdout_evaluated"] is not False
        or freeze["direct_holdout_read"] is not False
    ):
        raise RuntimeError("invalid pre-holdout freeze manifest")
    hash_mismatches = {}
    for relative, expected in freeze["file_sha256"].items():
        path = ROOT / relative
        observed = sha256(path)
        if observed != expected:
            hash_mismatches[relative] = {"expected": expected, "observed": observed}
    if hash_mismatches:
        raise RuntimeError(f"frozen artifact hash mismatch: {hash_mismatches}")

    direct_rows, direct = load_direct_rows()
    audit_records, audit_summary = audit_dyn(direct)
    predictions = load_prediction_rows(
        OUTDIR / "holdout_blind_predictions.csv", "predicted_f6_cm"
    )
    baselines = load_prediction_rows(
        OUTDIR / "holdout_baseline_predictions.csv", "baseline_f6_cm"
    )
    if set(predictions) != {
        row_key for row_key in direct if row_key[1] == 32
    } or set(baselines) != set(predictions):
        raise RuntimeError("prediction/direct primary-key mismatch")

    point_rows = []
    for row_key in sorted(predictions):
        dft = float(direct[row_key]["frequencies_cm"][-1])
        method = predictions[row_key]["frequency_cm"]
        baseline = baselines[row_key]["frequency_cm"]
        point_rows.append(
            {
                "degauss_Ry": row_key[0],
                "region": row_key[2],
                "t_GK": row_key[3],
                "DFPT_f6_cm": dft,
                "frozen_baseline_f6_cm": baseline,
                "qspace_prediction_f6_cm": method,
                "baseline_signed_error_cm-1": baseline - dft,
                "qspace_signed_error_cm-1": method - dft,
                "baseline_abs_error_cm-1": abs(baseline - dft),
                "qspace_abs_error_cm-1": abs(method - dft),
            }
        )

    line_metrics = []
    for dg in PRIMARY_DGS:
        for region in ("G", "K"):
            selected = [
                row for row in point_rows
                if np.isclose(row["degauss_Ry"], dg)
                and row["region"] == region
            ]
            baseline_mae = float(
                np.mean([row["baseline_abs_error_cm-1"] for row in selected])
            )
            method_mae = float(
                np.mean([row["qspace_abs_error_cm-1"] for row in selected])
            )
            metric = {
                "degauss_Ry": dg,
                "region": region,
                "n_q": len(selected),
                "baseline_MAE_cm-1": baseline_mae,
                "qspace_MAE_cm-1": method_mae,
                "qspace_max_abs_error_cm-1": float(
                    max(row["qspace_abs_error_cm-1"] for row in selected)
                ),
                "MAE_degradation_vs_baseline_cm-1": method_mae - baseline_mae,
            }
            if region == "K":
                center = next(row for row in selected if np.isclose(row["t_GK"], 1.0))
                metric["K_center_abs_error_cm-1"] = center[
                    "qspace_abs_error_cm-1"
                ]
            line_metrics.append(metric)

    dg13_k = sorted(
        [
            row for row in point_rows
            if np.isclose(row["degauss_Ry"], 0.013) and row["region"] == "K"
        ],
        key=lambda row: row["t_GK"],
    )
    k_t = np.array([row["t_GK"] for row in dg13_k])
    direct_k = np.array([row["DFPT_f6_cm"] for row in dg13_k])
    method_k = np.array([row["qspace_prediction_f6_cm"] for row in dg13_k])
    direct_jump = slope_jump(k_t, direct_k)
    method_jump = slope_jump(k_t, method_k)
    jump_relative_error = abs(method_jump - direct_jump) / abs(direct_jump)

    kgrid_differences = []
    for t_value in K64_T:
        f32 = float(direct[key(0.013, 32, "K", t_value)]["frequencies_cm"][-1])
        f64 = float(direct[key(0.013, 64, "K", t_value)]["frequencies_cm"][-1])
        kgrid_differences.append(f32 - f64)
    kgrid_mae = float(np.mean(np.abs(kgrid_differences)))

    thresholds = freeze["acceptance_thresholds"]
    gates = {
        "K_line_MAE_each_smearing": {
            "threshold": thresholds["K_line_MAE_each_smearing_cm-1"],
            "observed": {
                f"{row['degauss_Ry']:g}": row["qspace_MAE_cm-1"]
                for row in line_metrics if row["region"] == "K"
            },
        },
        "K_center_abs_error_each_smearing": {
            "threshold": thresholds[
                "K_center_abs_error_each_smearing_cm-1"
            ],
            "observed": {
                f"{row['degauss_Ry']:g}": row["K_center_abs_error_cm-1"]
                for row in line_metrics if row["region"] == "K"
            },
        },
        "K_slope_jump_relative_error_at_dg0.013": {
            "threshold": thresholds[
                "K_slope_jump_relative_error_at_dg0.013"
            ],
            "observed": jump_relative_error,
            "direct_THz_per_t": direct_jump,
            "predicted_THz_per_t": method_jump,
        },
        "Gamma_line_MAE_each_smearing": {
            "threshold": thresholds["Gamma_line_MAE_each_smearing_cm-1"],
            "observed": {
                f"{row['degauss_Ry']:g}": row["qspace_MAE_cm-1"]
                for row in line_metrics if row["region"] == "G"
            },
        },
        "maximum_MAE_degradation_vs_uncorrected": {
            "threshold": thresholds[
                "maximum_MAE_degradation_vs_uncorrected_cm-1"
            ],
            "observed": {
                f"{row['region']}_{row['degauss_Ry']:g}": row[
                    "MAE_degradation_vs_baseline_cm-1"
                ]
                for row in line_metrics
            },
        },
        "k32_k64_low_smearing_K_MAE": {
            "threshold": thresholds["k32_k64_low_smearing_K_MAE_cm-1"],
            "observed": kgrid_mae,
            "signed_differences_cm-1": kgrid_differences,
        },
    }
    for name, gate in gates.items():
        observed = gate["observed"]
        values = observed.values() if isinstance(observed, dict) else [observed]
        gate["pass"] = all(float(value) < float(gate["threshold"]) for value in values)
    overall_pass = all(gate["pass"] for gate in gates.values())

    with (OUTDIR / "holdout_dyn_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_records[0]))
        writer.writeheader()
        writer.writerows(audit_records)
    (OUTDIR / "holdout_dyn_audit.json").write_text(
        json.dumps(audit_summary, indent=2) + "\n"
    )
    with (OUTDIR / "holdout_evaluation_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    with (OUTDIR / "holdout_evaluation_metrics.csv").open("w", newline="") as handle:
        fieldnames = sorted(set().union(*(row.keys() for row in line_metrics)))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(line_metrics)

    summary = {
        "status": "PASS" if overall_pass else "FAIL",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest_sha256": sha256(freeze_path),
        "frozen_artifact_hashes_verified": True,
        "direct_holdout_rows": len(direct_rows),
        "dynamical_matrix_audit": audit_summary,
        "line_metrics": line_metrics,
        "acceptance_gates": gates,
    }
    (OUTDIR / "holdout_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
