#!/usr/bin/env python3
"""Freeze the graphene q-space model and blind holdout predictions.

This script refuses to run if a final-HOLD result directory is present.  It
reads only development data and the previously generated frozen-baseline
predictions, verifies the real dynamical-matrix projector, then writes the
manifest that authorizes the remote HOLD_A/HOLD_B runners.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import graphene_p0_deploy as p0  # noqa: E402
from qspace_kohn_development import (  # noqa: E402
    DEVELOPMENT_DGS,
    TEMPERATURE_POLYNOMIAL_DEGREE,
    apply_top_projector,
    coefficient_law,
    distance_from_anomaly,
    features,
)
from qspace_kohn_model import build_primary_baseline  # noqa: E402


OUTDIR = ROOT / "results" / "p0_graphene_qspace"
FREEZE = OUTDIR / "freeze_manifest.json"
PREDICTIONS = OUTDIR / "holdout_blind_predictions.csv"
HOLDOUT_DGS = (0.013, 0.027, 0.055)
EXPECTED_T = {
    "G": (0.000, 0.012, 0.025, 0.045, 0.075),
    "K": (0.925, 0.955, 0.975, 0.990, 1.000, 1.010, 1.025, 1.045, 1.075),
}
CM_PER_THz = 33.356


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def assert_holdout_absent() -> None:
    forbidden = [
        OUTDIR / "HOLD_A",
        OUTDIR / "HOLD_B",
        OUTDIR / "holdout_dfpt",
        OUTDIR / "final_dfpt",
        OUTDIR / "development_dfpt" / "HOLD_A",
        OUTDIR / "development_dfpt" / "HOLD_B",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"refusing to freeze after holdout appeared: {present}")


def matrix_frequencies(
    matrix: np.ndarray,
    reference_matrix: np.ndarray,
    reference_cm: np.ndarray,
) -> np.ndarray:
    herm = (matrix + matrix.conj().T) / 2
    eig = np.linalg.eigvalsh(herm)
    reference_eig = np.linalg.eigvalsh(
        (reference_matrix + reference_matrix.conj().T) / 2
    )
    scale_mask = (reference_eig > 1e-10) & (reference_cm > 1e-6)
    scale = float(
        np.median(reference_cm[scale_mask] ** 2 / reference_eig[scale_mask])
    )
    return np.sign(eig) * np.sqrt(np.abs(eig) * scale)


def line_metrics(rows: list[dict[str, object]], dg: float) -> dict[str, float]:
    gamma = sorted(
        [
            row for row in rows
            if row["region"] == "G" and np.isclose(row["degauss_Ry"], dg)
        ],
        key=lambda row: row["t_GK"],
    )
    k_rows = sorted(
        [
            row for row in rows
            if row["region"] == "K" and np.isclose(row["degauss_Ry"], dg)
        ],
        key=lambda row: row["t_GK"],
    )
    tg = np.array([row["t_GK"] for row in gamma], float)
    fg = np.array([row["predicted_f6_cm"] for row in gamma], float)
    tk = np.array([row["t_GK"] for row in k_rows], float)
    fk = np.array([row["predicted_f6_cm"] for row in k_rows], float)
    left = np.polyfit(tk[tk < 1.0], fk[tk < 1.0], 1)
    right = np.polyfit(tk[tk > 1.0], fk[tk > 1.0], 1)
    k_index = int(np.argmin(np.abs(tk - 1.0)))
    gamma_near = tg <= 0.045
    return {
        "degauss_Ry": dg,
        "predicted_Gamma_center_cm-1": float(fg[0]),
        "predicted_Gamma_near_slope_THz_per_t": float(
            np.polyfit(tg[gamma_near], fg[gamma_near], 1)[0] / CM_PER_THz
        ),
        "predicted_K_center_cm-1": float(fk[k_index]),
        "predicted_K_slope_jump_THz_per_t": float(
            abs(right[0] - left[0]) / CM_PER_THz
        ),
        "predicted_K_local_cusp_depth_cm-1": float(
            0.5 * (fk[k_index - 1] + fk[k_index + 1]) - fk[k_index]
        ),
    }


def main() -> int:
    if FREEZE.exists():
        raise FileExistsError(f"frozen manifest already exists: {FREEZE}")
    assert_holdout_absent()

    development_path = OUTDIR / "qspace_development_summary.json"
    development = load_json(development_path)
    campaign_audit_path = (
        OUTDIR / "development_dfpt" / "development_campaign_audit.json"
    )
    campaign_audit = load_json(campaign_audit_path)
    baseline_summary_path = OUTDIR / "holdout_baseline_summary.json"
    baseline_summary = load_json(baseline_summary_path)
    baseline_csv_path = OUTDIR / "holdout_baseline_predictions.csv"

    if development["status"] != "development_only_not_frozen":
        raise RuntimeError("development summary has unexpected status")
    if "final holdout is not read" not in development["target_leakage"]:
        raise RuntimeError("development leakage declaration missing")
    if campaign_audit["status"] != "complete" or campaign_audit["n_files"] != 28:
        raise RuntimeError("supplemental campaign audit is incomplete")
    if baseline_summary["direct_holdout_read"] is not False:
        raise RuntimeError("baseline provenance indicates holdout leakage")
    if baseline_summary["development_control_optical_replay_max_abs_cm-1"] >= 0.01:
        raise RuntimeError("frozen baseline optical replay failed")

    selected = development["selected"]
    for region in ("G", "K"):
        model = selected[region]
        if model["kind"] != "rounded_cusp" or not np.isclose(
            model["width_scale_per_Ry"], 4.0
        ):
            raise RuntimeError(f"unexpected primary model for {region}: {model}")
        if model["q_group_CV_MAE_cm-1"] >= 1.0:
            raise RuntimeError(f"{region} q-point CV freeze gate failed")
        if model["smearing_LOOCV_MAE_cm-1"] >= 2.0:
            raise RuntimeError(f"{region} smearing CV freeze gate failed")

    coeff_by_region = {
        region: {
            float(dg): np.asarray(values, float)
            for dg, values in development["coefficients_by_degauss"][region].items()
        }
        for region in ("G", "K")
    }
    for region in ("G", "K"):
        if set(coeff_by_region[region]) != set(DEVELOPMENT_DGS):
            raise RuntimeError(f"incomplete coefficient anchors for {region}")

    with baseline_csv_path.open() as handle:
        baseline_rows = list(csv.DictReader(handle))
    if len(baseline_rows) != 42:
        raise RuntimeError(f"expected 42 baseline rows, found {len(baseline_rows)}")
    baseline_lookup = {
        (
            float(row["degauss_Ry"]),
            row["region"],
            round(float(row["t_GK"]), 12),
        ): row
        for row in baseline_rows
    }
    if len(baseline_lookup) != 42:
        raise RuntimeError("duplicate frozen-baseline prediction rows")

    frozen_baseline = build_primary_baseline()
    predictions: list[dict[str, object]] = []
    baseline_replay_error = 0.0
    projector_replay_error = 0.0
    hermitian_error = 0.0
    law_coefficients = {}

    for region in ("G", "K"):
        anchors = np.array(sorted(coeff_by_region[region]), float)
        values = np.stack([coeff_by_region[region][dg] for dg in anchors])
        law_coefficients[region] = [
            np.polyfit(
                anchors, values[:, column], TEMPERATURE_POLYNOMIAL_DEGREE
            ).tolist()
            for column in range(values.shape[1])
        ]

    for dg in HOLDOUT_DGS:
        fc_path = (
            OUTDIR / "baseline_cache"
            / f"graphene_8x8_dg{p0.dg_slug(dg)}_baseline_fc2.npz"
        )
        with np.load(fc_path, allow_pickle=False) as data:
            fc = np.asarray(data["fc2"], float)
        frozen_baseline.ph.force_constants = fc
        for region, t_values_raw in EXPECTED_T.items():
            t_values = np.asarray(t_values_raw, float)
            qpoints = np.array(
                # Phonopy uses reciprocal reduced coordinates, where this
                # Gamma--K ray is (t/3,t/3,0). QE's Cartesian 2pi/a input uses
                # (t/3,t/sqrt(3),0); do not mix the two conventions.
                [[value / 3.0, value / 3.0, 0.0] for value in t_values]
            )
            frozen_baseline.ph.run_qpoints(
                qpoints, with_dynamical_matrices=True
            )
            result = frozen_baseline.ph.get_qpoints_dict()
            base_frequencies = (
                np.sort(np.asarray(result["frequencies"], float), axis=1)
                * p0.CM
            )
            matrices = np.asarray(result["dynamical_matrices"], complex)
            model = selected[region]
            target_coeff = coefficient_law(coeff_by_region[region], dg)
            x = distance_from_anomaly(region, t_values)
            target_lambda = features(
                model["kind"],
                x,
                dg,
                model["width_scale_per_Ry"],
            ) @ target_coeff
            if np.any(target_lambda <= 0):
                raise RuntimeError(f"non-positive blind target for {region}, {dg}")
            predicted = np.sqrt(target_lambda)

            for index, (t_value, matrix) in enumerate(zip(t_values, matrices)):
                key = (dg, region, round(float(t_value), 12))
                archived_base = float(baseline_lookup[key]["baseline_f6_cm"])
                baseline_replay_error = max(
                    baseline_replay_error,
                    abs(archived_base - float(base_frequencies[index, -1])),
                )
                delta_lambda = (
                    float(target_lambda[index])
                    - float(base_frequencies[index, -1]) ** 2
                )
                corrected_matrix = apply_top_projector(
                    matrix,
                    delta_lambda,
                    base_frequencies[index],
                    gamma_degenerate=(region == "G"),
                )
                hermitian_error = max(
                    hermitian_error,
                    float(
                        np.max(
                            np.abs(corrected_matrix - corrected_matrix.conj().T)
                        )
                    ),
                )
                projected = matrix_frequencies(
                    corrected_matrix, matrix, base_frequencies[index]
                )
                projector_replay_error = max(
                    projector_replay_error,
                    abs(float(projected[-1]) - float(predicted[index])),
                )
                predictions.append(
                    {
                        "degauss_Ry": dg,
                        "region": region,
                        "t_GK": float(t_value),
                        "baseline_f6_cm": float(base_frequencies[index, -1]),
                        "predicted_f6_cm": float(predicted[index]),
                        "delta_lambda_cm-2": delta_lambda,
                        "model_kind": model["kind"],
                        "width_scale_per_Ry": model["width_scale_per_Ry"],
                    }
                )

    if baseline_replay_error >= 0.01:
        raise RuntimeError(
            f"holdout baseline replay failed: {baseline_replay_error}"
        )
    if projector_replay_error >= 1e-6 or hermitian_error >= 1e-10:
        raise RuntimeError(
            "blind projector replay failed: "
            f"frequency={projector_replay_error}, hermitian={hermitian_error}"
        )

    predictions.sort(
        key=lambda row: (row["degauss_Ry"], row["region"], row["t_GK"])
    )
    with PREDICTIONS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    frozen_files = [
        ROOT / "docs" / "GRAPHENE_KOHN_QSPACE_FIX_2026-07-23.md",
        ROOT / "scripts" / "smearing_kink" / "qspace_kohn_development.py",
        ROOT / "scripts" / "smearing_kink" / "qspace_kohn_model.py",
        ROOT / "scripts" / "smearing_kink" / "freeze_graphene_qspace_model.py",
        ROOT / "scripts" / "smearing_kink" / "qe_dyn.py",
        ROOT / "scripts" / "v100" / "run_graphene_qspace_dfpt.sh",
        ROOT / "scripts" / "v100" / "watch_graphene_qspace_dfpt.sh",
        ROOT / "results" / "p0_graphene_dfpt" / "dfpt_linecuts_B.csv",
        OUTDIR / "development_dfpt" / "DEV_A" / "dfpt_DEV_A.csv",
        OUTDIR / "development_dfpt" / "DEV_B" / "dfpt_DEV_B.csv",
        campaign_audit_path,
        development_path,
        baseline_summary_path,
        baseline_csv_path,
        PREDICTIONS,
    ]
    file_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in frozen_files
    }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metrics = [line_metrics(predictions, dg) for dg in HOLDOUT_DGS]
    manifest = {
        "status": "frozen_before_holdout",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "content_hashes_are_authoritative_for_uncommitted_files": True,
        "authorized_holdout_campaigns": ["HOLD_A", "HOLD_B"],
        "holdout_evaluated": False,
        "direct_holdout_read": False,
        "development_degauss_Ry": list(DEVELOPMENT_DGS),
        "holdout_degauss_Ry": list(HOLDOUT_DGS),
        "model": {
            "target": development["target_definition"],
            "selected": selected,
            "temperature_polynomial_degree": TEMPERATURE_POLYNOMIAL_DEGREE,
            "temperature_polynomial_coefficients_high_to_low": law_coefficients,
            "development_coefficients_by_degauss": (
                development["coefficients_by_degauss"]
            ),
        },
        "development_validation": {
            region: {
                "q_group_CV_MAE_cm-1": selected[region][
                    "q_group_CV_MAE_cm-1"
                ],
                "smearing_LOOCV_MAE_cm-1": selected[region][
                    "smearing_LOOCV_MAE_cm-1"
                ],
            }
            for region in ("G", "K")
        },
        "blind_prediction_audit": {
            "n_primary_points": len(predictions),
            "baseline_replay_max_abs_cm-1": baseline_replay_error,
            "projector_replay_max_abs_cm-1": projector_replay_error,
            "projector_hermitian_max_abs": hermitian_error,
            "predicted_line_metrics": metrics,
        },
        "acceptance_thresholds": {
            "K_line_MAE_each_smearing_cm-1": 5.0,
            "K_center_abs_error_each_smearing_cm-1": 5.0,
            "K_slope_jump_relative_error_at_dg0.013": 0.20,
            "Gamma_line_MAE_each_smearing_cm-1": 5.0,
            "maximum_MAE_degradation_vs_uncorrected_cm-1": 1.0,
            "k32_k64_low_smearing_K_MAE_cm-1": 1.0,
        },
        "file_sha256": file_hashes,
    }
    FREEZE.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
