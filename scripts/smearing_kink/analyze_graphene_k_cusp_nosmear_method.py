#!/usr/bin/env python3
"""Freeze and evaluate the no-degauss graphene K-cusp q-space model.

``freeze`` reads only the predeclared development q points, selects one of
three direction-aware cusp bases by leave-one-distance-out validation, and
writes an immutable model artifact.  It refuses to run if synced holdout data
already exist under the default result root.

``holdout`` reads the frozen artifact and the six blind q points.  It does not
refit.  In addition to scalar frequency metrics, it realizes the prediction as
a Hermitian rank-one correction on the frozen 450 K static short-range MLIP
matrix and checks exact matrix replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_k_cusp_b0_dense as b0  # noqa: E402
import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402
from qe_dyn import QEDyn, load_qe_dyn  # noqa: E402


MODEL_NAMES = (
    "linear_directional",
    "quadratic_shared",
    "quadratic_directional",
)
EXPECTED_DEVELOPMENT = {"K", "KG_d007", "KM_d007", "KG_d015", "KM_d015", "KG_d023", "KM_d023"}
EXPECTED_HOLDOUT = {"KG_d003", "KM_d003", "KG_d011", "KM_d011", "KG_d019", "KM_d019"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass
class DirectPoint:
    label: str
    direction: str
    delta: float
    role: str
    lane: str
    kgrid: int
    qe_tag: str
    q_reduced: np.ndarray
    frequencies_cm: np.ndarray
    dyn: QEDyn
    dyn_sha256: str
    source_manifest: Path
    source_manifest_sha256: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_source(path: Path) -> list[DirectPoint]:
    manifest_path = path / "manifest.json"
    csv_path = path / "dfpt_points.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete source: {path}")
    if manifest.get("electronic_integration") != "occupations='tetrahedra_opt'":
        raise ValueError(f"source is not tetrahedra_opt: {path}")
    if manifest.get("smearing") is not None or manifest.get("degauss_Ry") is not None:
        raise ValueError(f"source records finite smearing: {path}")
    if digest(csv_path) != manifest["outputs"]["csv_sha256"]:
        raise ValueError(f"CSV hash mismatch: {path}")
    audit_by_label = {row["label"]: row for row in manifest["qpoint_audits"]}
    result = []
    manifest_hash = digest(manifest_path)
    for row in read_csv(csv_path):
        label = row["label"]
        dyn_path = path / "bundle" / label / "gr.dyn"
        dyn_hash = digest(dyn_path)
        if dyn_hash != audit_by_label[label]["gr_dyn_sha256"]:
            raise ValueError(f"gr.dyn hash mismatch: {path}/{label}")
        result.append(
            DirectPoint(
                label=label,
                direction=row["direction"],
                delta=float(row["delta_equal_distance"]),
                role=row["role"],
                lane=row["lane"],
                kgrid=int(row["kgrid"]),
                qe_tag=row["qe_tag"],
                q_reduced=np.asarray(
                    [float(row["q_reduced_h"]), float(row["q_reduced_k"]), 0.0]
                ),
                frequencies_cm=np.asarray(
                    [float(row[f"f{index}_cm-1"]) for index in range(1, 7)]
                ),
                dyn=load_qe_dyn(dyn_path),
                dyn_sha256=dyn_hash,
                source_manifest=manifest_path,
                source_manifest_sha256=manifest_hash,
            )
        )
    return result


def load_sources(paths: list[Path]) -> list[DirectPoint]:
    points = [point for path in paths for point in load_source(path)]
    if not points:
        raise ValueError("no direct DFPT points")
    kgrids = {point.kgrid for point in points}
    qe_tags = {point.qe_tag for point in points}
    if len(kgrids) != 1 or len(qe_tags) != 1:
        raise ValueError(f"mixed k grids or QE tags: k={kgrids}, tags={qe_tags}")
    return points


def vector_overlap(left: np.ndarray, right: np.ndarray) -> float:
    return float(abs(np.vdot(left.reshape(-1), right.reshape(-1))) ** 2)


def track_points(points: list[DirectPoint]) -> tuple[list[dict], float, dict]:
    k_points = [point for point in points if point.direction == "K"]
    if not k_points:
        raise ValueError("missing exact K anchor")
    cross_host = 0.0
    for left in k_points:
        for right in k_points:
            cross_host = max(
                cross_host,
                float(np.max(np.abs(left.frequencies_cm - right.frequencies_cm))),
            )
    reference = k_points[0].dyn.eigenvectors[-1]
    k_frequency = float(np.mean([point.frequencies_cm[-1] for point in k_points]))
    rows = [
        {
            "label": "K",
            "direction": "K",
            "delta": 0.0,
            "frequency_cm-1": k_frequency,
            "mode_1based": 6,
            "overlap": 1.0,
            "q_reduced": k_points[0].q_reduced,
            "source_lanes": "+".join(sorted({point.lane for point in k_points})),
        }
    ]
    minimum_overlap = 1.0
    for direction in ("KG", "KM"):
        previous = reference
        selected = sorted(
            (point for point in points if point.direction == direction),
            key=lambda point: point.delta,
        )
        for point in selected:
            candidates = range(3, 6)
            overlaps = np.asarray(
                [vector_overlap(previous, point.dyn.eigenvectors[index]) for index in candidates]
            )
            mode = int(np.argmax(overlaps)) + 3
            overlap = float(np.max(overlaps))
            minimum_overlap = min(minimum_overlap, overlap)
            previous = point.dyn.eigenvectors[mode]
            rows.append(
                {
                    "label": point.label,
                    "direction": direction,
                    "delta": point.delta,
                    "frequency_cm-1": float(point.frequencies_cm[mode]),
                    "mode_1based": mode + 1,
                    "overlap": overlap,
                    "q_reduced": point.q_reduced,
                    "source_lanes": point.lane,
                }
            )
    provenance = {
        "kgrid": points[0].kgrid,
        "qe_tag": points[0].qe_tag,
        "cross_host_K_all_mode_max_abs_cm-1": cross_host,
        "source_manifests": sorted(
            {
                (str(point.source_manifest), point.source_manifest_sha256)
                for point in points
            }
        ),
        "source_dyn_sha256": {point.label + "@" + point.lane: point.dyn_sha256 for point in points},
        "K_reference_eigenvector": reference,
    }
    return rows, minimum_overlap, provenance


def design(model: str, rows: list[dict]) -> np.ndarray:
    columns = []
    for row in rows:
        direction = row["direction"]
        r = float(row["delta"])
        kg = 1.0 if direction == "KG" else 0.0
        km = 1.0 if direction == "KM" else 0.0
        values = [1.0, kg * r, km * r]
        if model == "quadratic_shared":
            values.append(r * r)
        elif model == "quadratic_directional":
            values.extend([kg * r * r, km * r * r])
        elif model != "linear_directional":
            raise ValueError(model)
        columns.append(values)
    return np.asarray(columns, float)


def fit_model(model: str, rows: list[dict]) -> np.ndarray:
    target = np.asarray([float(row["frequency_cm-1"]) ** 2 for row in rows])
    matrix = design(model, rows)
    k_targets = [
        float(row["frequency_cm-1"]) ** 2
        for row in rows
        if row["direction"] == "K"
    ]
    if not k_targets:
        raise ValueError("the shared K intercept must be present in every fit")
    intercept = float(np.mean(k_targets))
    remaining, *_ = np.linalg.lstsq(
        matrix[:, 1:], target - intercept, rcond=None
    )
    return np.concatenate(([intercept], remaining))


def predict_frequency(model: str, coefficients: np.ndarray, rows: list[dict]) -> np.ndarray:
    squared = design(model, rows) @ coefficients
    return np.sign(squared) * np.sqrt(np.abs(squared))


def development_scores(rows: list[dict]) -> list[dict]:
    distances = sorted({float(row["delta"]) for row in rows if row["direction"] != "K"})
    scores = []
    for model in MODEL_NAMES:
        errors = []
        by_distance = {}
        for held_distance in distances:
            training = [
                row
                for row in rows
                if row["direction"] == "K" or not np.isclose(row["delta"], held_distance)
            ]
            held = [row for row in rows if np.isclose(row["delta"], held_distance)]
            coefficients = fit_model(model, training)
            prediction = predict_frequency(model, coefficients, held)
            target = np.asarray([float(row["frequency_cm-1"]) for row in held])
            held_errors = np.abs(prediction - target)
            errors.extend(held_errors.tolist())
            by_distance[f"{held_distance:.3f}"] = float(np.mean(held_errors))
        n_parameters = design(model, rows[:1]).shape[1]
        cv_mae = float(np.mean(errors))
        penalty = 0.10 * max(0, n_parameters - 3)
        scores.append(
            {
                "model": model,
                "n_parameters": n_parameters,
                "leave_one_distance_out_MAE_cm-1": cv_mae,
                "MAE_by_held_distance_cm-1": by_distance,
                "complexity_penalty_cm-1": penalty,
                "selection_score": cv_mae + penalty,
            }
        )
    return sorted(scores, key=lambda row: row["selection_score"])


def model_rows(direction: str, distances: np.ndarray) -> list[dict]:
    return [{"direction": direction, "delta": float(value)} for value in distances]


def write_csv(path: Path, rows: list[dict]) -> None:
    serializable = []
    for row in rows:
        serializable.append(
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in row.items()
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def make_freeze_figure(rows: list[dict], model: str, coefficients: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.1, 4.0))
    colors = {"KG": "#1F77B4", "KM": "#D55E00"}
    dense = np.linspace(0.0, 0.024, 161)
    for direction, label in (("KG", "K→Γ"), ("KM", "K→M")):
        selected = [row for row in rows if row["direction"] in ("K", direction)]
        axis.plot(
            dense,
            predict_frequency(model, coefficients, model_rows(direction, dense)),
            color=colors[direction],
            lw=1.5,
            label=f"{label} frozen model",
        )
        axis.plot(
            [float(row["delta"]) for row in selected],
            [float(row["frequency_cm-1"]) for row in selected],
            "o",
            color=colors[direction],
            ms=4.5,
            label=f"{label} development",
        )
    axis.grid(axis="y", color="#E6E6E6", lw=0.55)
    axis.set_xlabel(r"equal-distance coordinate $d$ from K")
    axis.set_ylabel(r"tracked $A_1'$ frequency (cm$^{-1}$)")
    axis.set_title("No-degauss K-cusp development freeze")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.70, top=0.90, bottom=0.14)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def freeze(args: argparse.Namespace) -> int:
    default_holdout = ROOT / "results" / "graphene_k_cusp_nosmear" / "raw" / "holdout"
    if default_holdout.exists() and any(default_holdout.rglob("dfpt_points.csv")):
        raise RuntimeError("refusing to freeze after holdout data were synced")
    points = load_sources(args.source)
    rows, minimum_overlap, provenance = track_points(points)
    labels = {row["label"] for row in rows}
    if labels != EXPECTED_DEVELOPMENT:
        raise ValueError(f"development point mismatch: missing={EXPECTED_DEVELOPMENT-labels}, extra={labels-EXPECTED_DEVELOPMENT}")
    scores = development_scores(rows)
    selected = scores[0]
    coefficients = fit_model(selected["model"], rows)
    k_frequency = next(float(row["frequency_cm-1"]) for row in rows if row["direction"] == "K")
    probe_distances = np.asarray([0.003, 0.007, 0.011, 0.015, 0.019, 0.023])
    direction_depths = {}
    direction_positive = True
    for direction in ("KG", "KM"):
        predicted = predict_frequency(
            selected["model"], coefficients, model_rows(direction, probe_distances)
        )
        depths = predicted - k_frequency
        direction_depths[direction] = depths.tolist()
        direction_positive = direction_positive and bool(np.all(depths > 0.0))
    checks = {
        "development_points_exact": True,
        "minimum_mode_overlap_ge_0p95": minimum_overlap >= 0.95,
        "cross_host_K_le_0p2_cm-1": provenance["cross_host_K_all_mode_max_abs_cm-1"] <= 0.2,
        "development_CV_MAE_le_3_cm-1": selected["leave_one_distance_out_MAE_cm-1"] <= 3.0,
        "both_directions_have_positive_cusp_depth": direction_positive,
    }
    passed = all(checks.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "frozen_model.npz"
    atomic_npz(
        model_path,
        model_name=np.asarray(selected["model"]),
        coefficients_cm2=np.asarray(coefficients),
        K_reference_eigenvector=np.asarray(provenance.pop("K_reference_eigenvector")),
        kgrid=np.asarray(provenance["kgrid"]),
        qe_tag=np.asarray(provenance["qe_tag"]),
    )
    write_csv(args.output_dir / "development_points.csv", rows)
    make_freeze_figure(
        rows, selected["model"], coefficients, args.output_dir / "development_freeze.png"
    )
    summary = {
        "status": "frozen_before_holdout" if passed else "development_gate_failed",
        "frozen_before_holdout": True,
        "target": "absolute no-degauss direct-DFPT A1' squared frequency",
        "selected": selected,
        "candidate_scores": scores,
        "coefficients_cm-2": coefficients.tolist(),
        "development_metrics": {
            "minimum_mode_overlap": minimum_overlap,
            "K_frequency_cm-1": k_frequency,
            "probe_distances": probe_distances.tolist(),
            "predicted_cusp_depths_cm-1": direction_depths,
        },
        "provenance": provenance,
        "checks": checks,
        "holdout_acceptance_frozen": {
            "line_MAE_cm-1": 10.0,
            "max_abs_error_cm-1": 15.0,
            "cusp_depth_relative_error_each_direction": 0.20,
            "one_sided_slope_relative_error_each_direction": 0.20,
            "slope_jump_relative_error": 0.20,
            "matrix_Hermiticity_max_abs": 1.0e-10,
            "rank_one_replay_max_abs_cm-1": 1.0e-6,
        },
        "model_artifact": {"path": model_path.name, "sha256": digest(model_path)},
    }
    atomic_json(args.output_dir / "freeze_manifest.json", summary)
    marker = args.output_dir / ("PASS" if passed else "FAILED")
    marker.write_text(summary["status"] + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


def fixed_slope(distances: np.ndarray, values: np.ndarray, center: float) -> float:
    delta = np.asarray(distances, float)
    return float(np.dot(delta, np.asarray(values, float) - center) / np.dot(delta, delta))


def static_projector_replay(
    rows: list[dict], prediction: np.ndarray
) -> dict:
    operator_path = a0.operator_path(450)
    with np.load(operator_path, allow_pickle=False) as payload:
        operator = {key: np.asarray(payload[key]) for key in payload.files}
    phonon, _ = a0.make_phonopy(operator)
    static_path = (
        ROOT
        / "results/graphene_physics_temperature/post_p4_feasibility/source_p4_450"
        / "on_policy/frozen_prediction/frozen_static_fc2.npz"
    )
    with np.load(static_path, allow_pickle=False) as payload:
        static_short = np.asarray(payload["short_force_constants"], float)
    qpoints = np.asarray([row["q_reduced"] for row in rows], float)
    sequence = b0.dynamical_sequence(phonon, static_short, qpoints)
    selected = np.asarray(
        [int(np.argmax(sequence["frequencies_cm1"][index, -3:])) + 3 for index in range(len(rows))]
    )
    baseline = np.asarray(
        [sequence["frequencies_cm1"][index, mode] for index, mode in enumerate(selected)]
    )
    delta_lambda = prediction**2 - baseline**2
    applied = b0.apply_rank_one(sequence, selected, delta_lambda)
    matrix_corrected = np.asarray(applied["corrected_frequencies_cm1"][:, -1], float)
    return {
        "baseline_cm-1": baseline,
        "corrected_cm-1": matrix_corrected,
        "delta_lambda_cm-2": delta_lambda,
        "Hermiticity_max_abs": applied["Hermiticity_max_abs"],
        "rank_one_replay_max_abs_cm-1": float(
            np.max(np.abs(matrix_corrected - prediction))
        ),
        "static_short_fc2": str(static_path),
        "static_short_fc2_sha256": digest(static_path),
        "operator_geometry": str(operator_path),
        "operator_geometry_sha256": digest(operator_path),
    }


def make_holdout_figure(rows: list[dict], prediction: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.1, 4.0))
    colors = {"KG": "#1F77B4", "KM": "#D55E00"}
    k_value = float(rows[0]["K_frequency_cm-1"])
    for direction, label in (("KG", "K→Γ"), ("KM", "K→M")):
        indices = [index for index, row in enumerate(rows) if row["direction"] == direction]
        x = [0.0] + [float(rows[index]["delta"]) for index in indices]
        direct = [k_value] + [float(rows[index]["frequency_cm-1"]) for index in indices]
        predicted = [k_value] + [float(prediction[index]) for index in indices]
        axis.plot(x, direct, "o", color=colors[direction], ms=5, label=f"{label} blind DFPT")
        axis.plot(x, predicted, "--", color=colors[direction], lw=1.5, label=f"{label} frozen prediction")
    axis.grid(axis="y", color="#E6E6E6", lw=0.55)
    axis.set_xlabel(r"equal-distance coordinate $d$ from K")
    axis.set_ylabel(r"tracked $A_1'$ frequency (cm$^{-1}$)")
    axis.set_title("No-degauss K-cusp blind holdout")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.69, top=0.90, bottom=0.14)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def holdout(args: argparse.Namespace) -> int:
    freeze_manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze_manifest.get("status") != "frozen_before_holdout":
        raise ValueError("development model did not pass before holdout")
    model_path = Path(freeze_manifest["model_artifact"]["path"])
    if not model_path.is_absolute():
        model_path = args.freeze_manifest.parent / model_path.name
    if digest(model_path) != freeze_manifest["model_artifact"]["sha256"]:
        raise ValueError("frozen model hash mismatch")
    with np.load(model_path, allow_pickle=False) as payload:
        model = str(payload["model_name"].item())
        coefficients = np.asarray(payload["coefficients_cm2"], float)
        k_reference = np.asarray(payload["K_reference_eigenvector"], complex)
        frozen_kgrid = int(payload["kgrid"])
        frozen_qe_tag = str(payload["qe_tag"].item())

    points = load_sources(args.source)
    if points[0].kgrid != frozen_kgrid or points[0].qe_tag != frozen_qe_tag:
        raise ValueError("holdout grid/QE tag does not match frozen model")
    if {point.label for point in points} != EXPECTED_HOLDOUT:
        raise ValueError("holdout point set differs from the frozen plan")
    rows = []
    minimum_overlap = 1.0
    for direction in ("KG", "KM"):
        previous = k_reference
        for point in sorted(
            (item for item in points if item.direction == direction),
            key=lambda item: item.delta,
        ):
            candidates = range(3, 6)
            overlaps = np.asarray(
                [vector_overlap(previous, point.dyn.eigenvectors[index]) for index in candidates]
            )
            mode = int(np.argmax(overlaps)) + 3
            overlap = float(np.max(overlaps))
            previous = point.dyn.eigenvectors[mode]
            minimum_overlap = min(minimum_overlap, overlap)
            rows.append(
                {
                    "label": point.label,
                    "direction": point.direction,
                    "delta": point.delta,
                    "frequency_cm-1": float(point.frequencies_cm[mode]),
                    "mode_1based": mode + 1,
                    "overlap": overlap,
                    "q_reduced": point.q_reduced,
                    "K_frequency_cm-1": freeze_manifest["development_metrics"]["K_frequency_cm-1"],
                }
            )
    prediction = predict_frequency(model, coefficients, rows)
    target = np.asarray([float(row["frequency_cm-1"]) for row in rows])
    errors = prediction - target
    k_frequency = float(freeze_manifest["development_metrics"]["K_frequency_cm-1"])
    by_direction = {}
    depth_errors = []
    slope_errors = []
    direct_slopes = {}
    predicted_slopes = {}
    for direction in ("KG", "KM"):
        indices = np.asarray([index for index, row in enumerate(rows) if row["direction"] == direction])
        distances = np.asarray([rows[index]["delta"] for index in indices], float)
        direct = target[indices]
        pred = prediction[indices]
        direct_depth = direct - k_frequency
        pred_depth = pred - k_frequency
        relative_depth = np.abs(pred_depth - direct_depth) / np.maximum(np.abs(direct_depth), 1.0e-12)
        direct_slope = fixed_slope(distances, direct, k_frequency)
        predicted_slope = fixed_slope(distances, pred, k_frequency)
        slope_error = abs(predicted_slope - direct_slope) / max(abs(direct_slope), 1.0e-12)
        depth_errors.extend(relative_depth.tolist())
        slope_errors.append(slope_error)
        direct_slopes[direction] = direct_slope
        predicted_slopes[direction] = predicted_slope
        by_direction[direction] = {
            "MAE_cm-1": float(np.mean(np.abs(errors[indices]))),
            "max_abs_error_cm-1": float(np.max(np.abs(errors[indices]))),
            "direct_cusp_depths_cm-1": direct_depth.tolist(),
            "predicted_cusp_depths_cm-1": pred_depth.tolist(),
            "cusp_depth_relative_errors": relative_depth.tolist(),
            "direct_one_sided_slope_cm-1_per_d": direct_slope,
            "predicted_one_sided_slope_cm-1_per_d": predicted_slope,
            "one_sided_slope_relative_error": slope_error,
        }
    direct_jump = direct_slopes["KG"] + direct_slopes["KM"]
    predicted_jump = predicted_slopes["KG"] + predicted_slopes["KM"]
    jump_error = abs(predicted_jump - direct_jump) / max(abs(direct_jump), 1.0e-12)
    projector = static_projector_replay(rows, prediction)
    thresholds = freeze_manifest["holdout_acceptance_frozen"]
    checks = {
        "holdout_point_set_exact": True,
        "minimum_mode_overlap_ge_0p95": minimum_overlap >= 0.95,
        "line_MAE_lt_10_cm-1": float(np.mean(np.abs(errors))) < thresholds["line_MAE_cm-1"],
        "max_abs_error_lt_15_cm-1": float(np.max(np.abs(errors))) < thresholds["max_abs_error_cm-1"],
        "all_cusp_depth_relative_errors_lt_20pct": max(depth_errors) < thresholds["cusp_depth_relative_error_each_direction"],
        "both_one_sided_slope_errors_lt_20pct": max(slope_errors) < thresholds["one_sided_slope_relative_error_each_direction"],
        "slope_jump_error_lt_20pct": jump_error < thresholds["slope_jump_relative_error"],
        "Hermiticity_le_1e-10": projector["Hermiticity_max_abs"] <= thresholds["matrix_Hermiticity_max_abs"],
        "rank_one_replay_le_1e-6_cm-1": projector["rank_one_replay_max_abs_cm-1"] <= thresholds["rank_one_replay_max_abs_cm-1"],
    }
    passed = all(checks.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            {
                **row,
                "q_reduced": row["q_reduced"].tolist(),
                "frozen_prediction_cm-1": float(prediction[index]),
                "signed_error_cm-1": float(errors[index]),
                "static_short_baseline_cm-1": float(projector["baseline_cm-1"][index]),
                "matrix_corrected_cm-1": float(projector["corrected_cm-1"][index]),
                "delta_lambda_cm-2": float(projector["delta_lambda_cm-2"][index]),
            }
        )
    write_csv(args.output_dir / "holdout_predictions.csv", output_rows)
    make_holdout_figure(rows, prediction, args.output_dir / "blind_holdout.png")
    summary = {
        "status": "passed_blind_holdout" if passed else "failed_blind_holdout",
        "freeze_manifest": {"path": str(args.freeze_manifest), "sha256": digest(args.freeze_manifest)},
        "model_artifact_sha256": digest(model_path),
        "kgrid": frozen_kgrid,
        "qe_tag": frozen_qe_tag,
        "metrics": {
            "line_MAE_cm-1": float(np.mean(np.abs(errors))),
            "max_abs_error_cm-1": float(np.max(np.abs(errors))),
            "minimum_mode_overlap": minimum_overlap,
            "by_direction": by_direction,
            "direct_slope_jump_cm-1_per_d": direct_jump,
            "predicted_slope_jump_cm-1_per_d": predicted_jump,
            "slope_jump_relative_error": jump_error,
            "matrix_projector": {
                key: value
                for key, value in projector.items()
                if not isinstance(value, np.ndarray)
            },
        },
        "checks": checks,
        "interpretation": (
            "static no-degauss q-space method passed on six unseen q points"
            if passed
            else "frozen q-space method did not satisfy the predeclared blind gates"
        ),
    }
    atomic_json(args.output_dir / "holdout_summary.json", summary)
    marker = args.output_dir / ("PASS" if passed else "FAILED")
    marker.write_text(summary["status"] + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--source", type=Path, action="append", required=True)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    freeze_parser.set_defaults(function=freeze)
    holdout_parser = subparsers.add_parser("holdout")
    holdout_parser.add_argument("--source", type=Path, action="append", required=True)
    holdout_parser.add_argument("--freeze-manifest", type=Path, required=True)
    holdout_parser.add_argument("--output-dir", type=Path, required=True)
    holdout_parser.set_defaults(function=holdout)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
