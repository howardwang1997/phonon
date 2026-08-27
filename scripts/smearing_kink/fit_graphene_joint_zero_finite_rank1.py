#!/usr/bin/env python3
"""Fit a shared zero/finite-smearing rank-one graphene K correction.

The short-range MLIP force constants and A' projector are kept fixed.  The
fit changes only the scalar long-range self-energy.  Finite-smearing shapes
come from the already-computed dense EPW correction, anchored to direct DFPT
at K; the zero-smearing shape comes from the opened tetrahedra-opt DFPT data.
No new electronic-structure calculation is launched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_k_cusp_b0_dense as b0  # noqa: E402
import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402
from graphene_fd_p4_common import line_metrics, qpoint_fractional  # noqa: E402


RY_TO_EV = 13.605693122994
A_GRAPHENE = 2.4600000087
K_MAGNITUDE_A_INV = 4.0 * math.pi / (3.0 * A_GRAPHENE)
# Replaced at runtime from the production k18/q9 Wannier diagnostic.
V_F_EVA_FIXED = 5.450391548607145
TEMPERATURES = (300, 450, 600)
GROUPS = ("zero", "300", "450", "600")
CANDIDATES = (
    "rounded_one_scale",
    "rounded_one_scale_curvature",
    "rounded_two_scale",
    "rounded_two_scale_curvature",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_static_short_phonon(operator_path: Path, static_path: Path):
    with np.load(operator_path, allow_pickle=False) as payload:
        operator = {key: np.asarray(payload[key]) for key in payload.files}
    phonon, _ = a0.make_phonopy(operator)
    with np.load(static_path, allow_pickle=False) as payload:
        force_constants = np.asarray(payload["short_force_constants"], float)
    return phonon, force_constants


def tracked_static_background(phonon, force_constants, qpoints: np.ndarray) -> np.ndarray:
    sequence = b0.dynamical_sequence(phonon, force_constants, qpoints)
    # The A' branch is isolated and is the highest mode throughout this window.
    selected = np.argmax(sequence["frequencies_cm1"], axis=1)
    return np.asarray(
        [sequence["frequencies_cm1"][index, mode] for index, mode in enumerate(selected)],
        float,
    )


def zero_qpoint(direction: str, distance: float) -> np.ndarray:
    if direction == "K":
        h = k = 1.0 / 3.0
    elif direction == "KG":
        h = k = (1.0 - distance) / 3.0
    elif direction == "KM":
        h, k = (1.0 + distance) / 3.0, (1.0 - 2.0 * distance) / 3.0
    else:
        raise ValueError(direction)
    return np.asarray([h, k, 0.0])


def load_zero_rows(
    path: Path,
    kgrid_path: Path,
    phonon,
    force_constants: np.ndarray,
) -> list[dict]:
    raw = read_csv(path)
    if len(raw) != 13:
        raise ValueError(f"expected 13 opened zero-smearing rows, found {len(raw)}")
    k192 = next(row for row in raw if row["direction"] == "K")
    k192_frequency = float(k192["frequency_cm-1"])
    convergence = json.loads(kgrid_path.read_text(encoding="utf-8"))
    if convergence.get("status") != "k288_still_unconverged":
        raise ValueError("unexpected zero-smearing convergence record")
    comparison = convergence["comparisons"]
    k288_anchor = float(comparison["KG"]["k288_K_frequency_cm-1"])
    qpoints = np.asarray(
        [zero_qpoint(row["direction"], float(row["delta"])) for row in raw],
        float,
    )
    background = tracked_static_background(phonon, force_constants, qpoints)
    rows = []
    for index, row in enumerate(raw):
        direction = row["direction"]
        distance = float(row["delta"])
        if direction == "K":
            target = k288_anchor
            source = "k288_anchor"
        elif np.isclose(distance, 0.003):
            target = float(comparison[direction]["k288_frequency_cm-1"])
            source = "k288_d003"
        else:
            # Preserve each k192 shape depth while aligning its absolute anchor
            # to the best existing k288 K value.  This is development data, not
            # a claim that the full k288 line has been calculated.
            target = k288_anchor + float(row["frequency_cm-1"]) - k192_frequency
            source = "k192_depth_aligned_to_k288_anchor"
        signed = 0.0 if direction == "K" else (-distance if direction == "KG" else distance)
        rows.append(
            {
                "group": "zero",
                "temperature_K": 0,
                "degauss_Ry": 0.0,
                "signed_distance": signed,
                "distance": distance,
                "direction": direction,
                "background_cm-1": float(background[index]),
                "target_total_cm-1": target,
                "target_delta_lambda_cm-2": target**2 - float(background[index]) ** 2,
                "source": source,
            }
        )
    return rows


def direct_finite_by_temperature(path: Path) -> dict[int, list[dict]]:
    rows = read_csv(path)
    result = {}
    for temperature in TEMPERATURES:
        block = [
            row
            for row in rows
            if int(row["temperature_K"]) == temperature and row["region"] == "K"
        ]
        if len(block) != 11:
            raise ValueError(f"expected 11 direct K rows at {temperature} K")
        result[temperature] = block
    return result


def load_finite_teacher_rows(
    dense_path: Path,
    direct_path: Path,
    phonon,
    force_constants: np.ndarray,
) -> tuple[list[dict], dict[int, list[dict]]]:
    dense = [row for row in read_csv(dense_path) if row["channel"] == "L0"]
    direct = direct_finite_by_temperature(direct_path)
    rows = []
    for temperature in TEMPERATURES:
        block = [row for row in dense if int(row["temperature_K"]) == temperature]
        if len(block) != 29:
            raise ValueError(f"expected 29 dense EPW rows at {temperature} K")
        t = np.asarray([float(row["t_GK"]) for row in block], float)
        background = tracked_static_background(
            phonon, force_constants, qpoint_fractional(t)
        )
        center = int(np.argmin(np.abs(t - 1.0)))
        epw_delta = np.asarray(
            [float(row["dense_delta_lambda_top_cm-2"]) for row in block], float
        )
        direct_k = min(
            direct[temperature], key=lambda row: abs(float(row["t_GK"]) - 1.0)
        )
        anchor = float(direct_k["target_cm-1"]) ** 2 - float(background[center]) ** 2
        target_delta = anchor + epw_delta - epw_delta[center]
        target_total = np.sqrt(np.maximum(background**2 + target_delta, 0.0))
        for index, row in enumerate(block):
            rows.append(
                {
                    "group": str(temperature),
                    "temperature_K": temperature,
                    "degauss_Ry": float(row["degauss_Ry"]),
                    "signed_distance": float(row["t_GK"]) - 1.0,
                    "distance": abs(float(row["t_GK"]) - 1.0),
                    "direction": "line_GK",
                    "background_cm-1": float(background[index]),
                    "target_total_cm-1": float(target_total[index]),
                    "target_delta_lambda_cm-2": float(target_delta[index]),
                    "source": "direct_K_anchor_plus_dense_EPW_shape",
                }
            )
    return rows, direct


def response_coordinate(degauss: np.ndarray, response_scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.asarray(degauss, float) / float(response_scale))


def rounded_distance(
    distance: np.ndarray, degauss: np.ndarray, width_scale: float
) -> np.ndarray:
    distance = np.asarray(distance, float)
    degauss = np.asarray(degauss, float)
    width = (
        2.0
        * float(width_scale)
        * degauss
        * RY_TO_EV
        / (V_F_EVA_FIXED * K_MAGNITUDE_A_INV)
    )
    return np.sqrt(distance * distance + width * width) - width


def feature_names(candidate: str) -> list[str]:
    names = ["anchor_constant", "anchor_zero_response", "rounded", "zero_response_x_rounded"]
    if candidate in ("rounded_two_scale", "rounded_two_scale_curvature"):
        names.extend(["narrow_cusp", "zero_response_x_narrow_cusp"])
    if candidate in ("rounded_one_scale_curvature", "rounded_two_scale_curvature"):
        names.extend(["distance_squared", "zero_response_x_distance_squared"])
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    return names


def design(
    rows: list[dict],
    candidate: str,
    width_scale: float,
    response_scale: float,
    narrow_scale: float,
) -> np.ndarray:
    distance = np.asarray([row["distance"] for row in rows], float)
    degauss = np.asarray([row["degauss_Ry"] for row in rows], float)
    response = response_coordinate(degauss, response_scale)
    rounded = rounded_distance(distance, degauss, width_scale)
    columns = [np.ones_like(distance), response, rounded, response * rounded]
    if candidate in ("rounded_two_scale", "rounded_two_scale_curvature"):
        narrow = float(narrow_scale) * rounded / (rounded + float(narrow_scale))
        columns.extend([narrow, response * narrow])
    if candidate in ("rounded_one_scale_curvature", "rounded_two_scale_curvature"):
        columns.extend([distance * distance, response * distance * distance])
    return np.column_stack(columns)


def solve_at_parameters(
    rows: list[dict],
    train_groups: set[str],
    candidate: str,
    width_scale: float,
    response_scale: float,
    narrow_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    train = [row for row in rows if row["group"] in train_groups]
    matrix = design(
        train, candidate, width_scale, response_scale, narrow_scale
    )
    target = np.asarray([row["target_delta_lambda_cm-2"] for row in train], float)
    frequency = np.asarray([row["target_total_cm-1"] for row in train], float)
    counts = {group: sum(row["group"] == group for row in train) for group in train_groups}
    weight = np.asarray(
        [1.0 / math.sqrt(counts[row["group"]]) / (2.0 * f)
         for row, f in zip(train, frequency, strict=True)],
        float,
    )
    scaled = matrix * weight[:, None]
    scaled_target = target * weight
    column_norm = np.linalg.norm(scaled, axis=0)
    if np.any(column_norm <= 0.0):
        raise ValueError("zero design column")
    normalized, *_ = np.linalg.lstsq(
        scaled / column_norm[None, :], scaled_target, rcond=1.0e-10
    )
    coefficients = normalized / column_norm
    all_matrix = design(
        rows, candidate, width_scale, response_scale, narrow_scale
    )
    delta = all_matrix @ coefficients
    background = np.asarray([row["background_cm-1"] for row in rows], float)
    prediction = np.sqrt(np.maximum(background**2 + delta, 0.0))
    losses = []
    for group in train_groups:
        indices = [index for index, row in enumerate(rows) if row["group"] == group]
        observed = np.asarray([rows[index]["target_total_cm-1"] for index in indices])
        losses.append(float(np.mean((prediction[indices] - observed) ** 2)))
    balanced_rmse = math.sqrt(float(np.mean(losses)))
    return coefficients, prediction, balanced_rmse


def fit_model(
    rows: list[dict], train_groups: set[str], candidate: str, grids: dict
) -> dict:
    best = None
    narrow_values = grids["narrow"] if "two_scale" in candidate else [0.003]
    for width_scale in grids["width"]:
        for response_scale in grids["response"]:
            for narrow_scale in narrow_values:
                coefficients, prediction, loss = solve_at_parameters(
                    rows,
                    train_groups,
                    candidate,
                    float(width_scale),
                    float(response_scale),
                    float(narrow_scale),
                )
                item = (
                    loss,
                    float(width_scale),
                    float(response_scale),
                    float(narrow_scale),
                    coefficients,
                    prediction,
                )
                if best is None or item[0] < best[0]:
                    best = item
    if best is None:
        raise RuntimeError("no fitted candidate")
    return {
        "train_balanced_RMSE_cm-1": best[0],
        "width_scale": best[1],
        "response_scale_Ry": best[2],
        "narrow_scale": best[3],
        "coefficients_cm-2": best[4],
        "prediction": best[5],
    }


def group_metrics(rows: list[dict], prediction: np.ndarray, groups: set[str]) -> dict:
    result = {}
    for group in GROUPS:
        if group not in groups:
            continue
        indices = [index for index, row in enumerate(rows) if row["group"] == group]
        observed = np.asarray([rows[index]["target_total_cm-1"] for index in indices])
        predicted = np.asarray(prediction)[indices]
        result[group] = {
            "n_points": len(indices),
            "RMSE_cm-1": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "MAE_cm-1": float(np.mean(np.abs(predicted - observed))),
            "max_abs_cm-1": float(np.max(np.abs(predicted - observed))),
        }
    return result


def zero_shape_metrics(rows: list[dict], prediction: np.ndarray) -> dict:
    selected = [(index, row) for index, row in enumerate(rows) if row["group"] == "zero"]
    lookup_target = {row["direction"] + f"_{row['distance']:.3f}": row["target_total_cm-1"] for _, row in selected}
    lookup_pred = {row["direction"] + f"_{row['distance']:.3f}": float(prediction[index]) for index, row in selected}
    target_k = lookup_target["K_0.000"]
    predicted_k = lookup_pred["K_0.000"]
    output = {}
    for direction in ("KG", "KM"):
        target_inner = (lookup_target[f"{direction}_0.003"] - target_k) / 0.003
        pred_inner = (lookup_pred[f"{direction}_0.003"] - predicted_k) / 0.003
        target_outer = (
            lookup_target[f"{direction}_0.019"] - lookup_target[f"{direction}_0.011"]
        ) / 0.008
        pred_outer = (
            lookup_pred[f"{direction}_0.019"] - lookup_pred[f"{direction}_0.011"]
        ) / 0.008
        target_ratio = target_inner / target_outer
        pred_ratio = pred_inner / pred_outer
        output[direction] = {
            "target_d003_cusp_depth_cm-1": lookup_target[f"{direction}_0.003"] - target_k,
            "predicted_d003_cusp_depth_cm-1": lookup_pred[f"{direction}_0.003"] - predicted_k,
            "d003_cusp_depth_relative_error": abs(pred_inner - target_inner) / abs(target_inner),
            "target_inner_to_outer_slope_ratio": target_ratio,
            "predicted_inner_to_outer_slope_ratio": pred_ratio,
            "slope_ratio_relative_error": abs(pred_ratio - target_ratio) / abs(target_ratio),
        }
    return output


def predict_rows(rows: list[dict], fit: dict, candidate: str) -> np.ndarray:
    delta = design(
        rows,
        candidate,
        fit["width_scale"],
        fit["response_scale_Ry"],
        fit["narrow_scale"],
    ) @ np.asarray(fit["coefficients_cm-2"], float)
    background = np.asarray([row["background_cm-1"] for row in rows], float)
    return np.sqrt(np.maximum(background**2 + delta, 0.0))


def evaluate_direct_finite(
    direct: dict[int, list[dict]],
    phonon,
    force_constants: np.ndarray,
    fit: dict,
    candidate: str,
) -> tuple[dict, list[dict]]:
    metrics = {}
    output = []
    for temperature in TEMPERATURES:
        block = direct[temperature]
        t = np.asarray([float(row["t_GK"]) for row in block], float)
        background = tracked_static_background(
            phonon, force_constants, qpoint_fractional(t)
        )
        rows = [
            {
                "group": str(temperature),
                "temperature_K": temperature,
                "degauss_Ry": float(row["degauss_Ry"]),
                "signed_distance": float(row["t_GK"]) - 1.0,
                "distance": abs(float(row["t_GK"]) - 1.0),
                "direction": "line_GK",
                "background_cm-1": float(base),
                "target_total_cm-1": float(row["target_cm-1"]),
            }
            for row, base in zip(block, background, strict=True)
        ]
        prediction = predict_rows(rows, fit, candidate)
        target = np.asarray([float(row["target_cm-1"]) for row in block], float)
        metrics[str(temperature)] = line_metrics("K", t, prediction, target)
        for row, predicted in zip(rows, prediction, strict=True):
            output.append({**row, "model_cm-1": float(predicted)})
    return metrics, output


def candidate_evaluation(
    rows: list[dict], candidate: str, grids: dict
) -> tuple[dict, np.ndarray]:
    all_groups = set(GROUPS)
    full = fit_model(rows, all_groups, candidate, grids)
    folds = {}
    loso_prediction = np.zeros(len(rows), float)
    for held in GROUPS:
        fitted = fit_model(rows, all_groups - {held}, candidate, grids)
        mask = np.asarray([row["group"] == held for row in rows], bool)
        loso_prediction[mask] = fitted["prediction"][mask]
        folds[held] = {
            "heldout_group": held,
            "train_balanced_RMSE_cm-1": fitted["train_balanced_RMSE_cm-1"],
            "width_scale": fitted["width_scale"],
            "response_scale_Ry": fitted["response_scale_Ry"],
            "narrow_scale": fitted["narrow_scale"],
            "metrics": group_metrics(rows, fitted["prediction"], {held})[held],
        }
    target = np.asarray([row["target_total_cm-1"] for row in rows], float)
    loso_group_rmse = [folds[group]["metrics"]["RMSE_cm-1"] for group in GROUPS]
    full_summary = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in full.items()
        if key != "prediction"
    }
    n_coefficients = len(feature_names(candidate))
    summary = {
        "candidate": candidate,
        "feature_names": feature_names(candidate),
        "n_linear_coefficients": n_coefficients,
        "full_fit": {
            **full_summary,
            "group_metrics": group_metrics(rows, full["prediction"], all_groups),
            "zero_shape": zero_shape_metrics(rows, full["prediction"]),
        },
        "LOSO_balanced_RMSE_cm-1": float(
            math.sqrt(np.mean(np.asarray(loso_group_rmse) ** 2))
        ),
        "LOSO_all_point_RMSE_cm-1": float(
            np.sqrt(np.mean((loso_prediction - target) ** 2))
        ),
        "folds": folds,
        "complexity_penalty_cm-1": 0.03 * max(0, n_coefficients - 4),
    }
    summary["selection_score"] = (
        summary["LOSO_balanced_RMSE_cm-1"] + summary["complexity_penalty_cm-1"]
    )
    return summary, full["prediction"]


def narrow_scale_profile(
    rows: list[dict],
    direct: dict[int, list[dict]],
    phonon,
    force_constants: np.ndarray,
    candidate: str,
    grids: dict,
) -> list[dict]:
    if "two_scale" not in candidate:
        return []
    values = (0.003, 0.004, 0.005, 0.0075, 0.010, 0.015, 0.025, 0.050)
    output = []
    for narrow_scale in values:
        fitted = fit_model(
            rows,
            set(GROUPS),
            candidate,
            {
                "width": grids["width"],
                "response": grids["response"],
                "narrow": np.asarray([narrow_scale]),
            },
        )
        zero_shape = zero_shape_metrics(rows, fitted["prediction"])
        serializable = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in fitted.items()
            if key != "prediction"
        }
        direct_metrics, _ = evaluate_direct_finite(
            direct, phonon, force_constants, serializable, candidate
        )
        output.append(
            {
                "narrow_scale": narrow_scale,
                "balanced_RMSE_cm-1": fitted["train_balanced_RMSE_cm-1"],
                "width_scale": fitted["width_scale"],
                "response_scale_Ry": fitted["response_scale_Ry"],
                "max_zero_d003_cusp_depth_relative_error": max(
                    float(zero_shape[direction]["d003_cusp_depth_relative_error"])
                    for direction in ("KG", "KM")
                ),
                "max_zero_slope_ratio_relative_error": max(
                    float(zero_shape[direction]["slope_ratio_relative_error"])
                    for direction in ("KG", "KM")
                ),
                "max_direct_finite_K_kink_relative_error": max(
                    float(metrics["kink_relative_error"])
                    for metrics in direct_metrics.values()
                ),
            }
        )
    return output


def zero_reference_sensitivity(
    rows: list[dict],
    raw_zero_path: Path,
    kgrid_path: Path,
    direct: dict[int, list[dict]],
    phonon,
    force_constants: np.ndarray,
    candidate: str,
    grids: dict,
) -> list[dict]:
    raw = read_csv(raw_zero_path)
    lookup = {
        (row["direction"], float(row["delta"])): float(row["frequency_cm-1"])
        for row in raw
    }
    k192_anchor = lookup[("K", 0.0)]
    convergence = json.loads(kgrid_path.read_text(encoding="utf-8"))["comparisons"]
    references = {
        "k192": {
            "anchor": k192_anchor,
            "KG_d003": lookup[("KG", 0.003)],
            "KM_d003": lookup[("KM", 0.003)],
        },
        "k240": {
            "anchor": float(convergence["KG"]["k240_K_frequency_cm-1"]),
            "KG_d003": float(convergence["KG"]["k240_frequency_cm-1"]),
            "KM_d003": float(convergence["KM"]["k240_frequency_cm-1"]),
        },
        "k288": {
            "anchor": float(convergence["KG"]["k288_K_frequency_cm-1"]),
            "KG_d003": float(convergence["KG"]["k288_frequency_cm-1"]),
            "KM_d003": float(convergence["KM"]["k288_frequency_cm-1"]),
        },
    }
    output = []
    for label, reference in references.items():
        adjusted = [dict(row) for row in rows]
        for row in adjusted:
            if row["group"] != "zero":
                continue
            if row["direction"] == "K":
                target = reference["anchor"]
            elif np.isclose(row["distance"], 0.003):
                target = reference[f"{row['direction']}_d003"]
            else:
                target = (
                    reference["anchor"]
                    + lookup[(row["direction"], row["distance"])]
                    - k192_anchor
                )
            row["target_total_cm-1"] = target
            row["target_delta_lambda_cm-2"] = (
                target**2 - row["background_cm-1"] ** 2
            )
        fitted = fit_model(adjusted, set(GROUPS), candidate, grids)
        zero_shape = zero_shape_metrics(adjusted, fitted["prediction"])
        serializable = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in fitted.items()
            if key != "prediction"
        }
        direct_metrics, _ = evaluate_direct_finite(
            direct, phonon, force_constants, serializable, candidate
        )
        output.append(
            {
                "zero_reference": label,
                "K_frequency_cm-1": reference["anchor"],
                "KG_d003_cusp_depth_cm-1": (
                    reference["KG_d003"] - reference["anchor"]
                ),
                "KM_d003_cusp_depth_cm-1": (
                    reference["KM_d003"] - reference["anchor"]
                ),
                "balanced_RMSE_cm-1": fitted["train_balanced_RMSE_cm-1"],
                "width_scale": fitted["width_scale"],
                "response_scale_Ry": fitted["response_scale_Ry"],
                "narrow_scale": fitted["narrow_scale"],
                "max_zero_d003_cusp_depth_relative_error": max(
                    float(zero_shape[direction]["d003_cusp_depth_relative_error"])
                    for direction in ("KG", "KM")
                ),
                "max_zero_slope_ratio_relative_error": max(
                    float(zero_shape[direction]["slope_ratio_relative_error"])
                    for direction in ("KG", "KM")
                ),
                "max_direct_finite_K_kink_relative_error": max(
                    float(metrics["kink_relative_error"])
                    for metrics in direct_metrics.values()
                ),
            }
        )
    return output


def write_predictions(path: Path, rows: list[dict], prediction: np.ndarray) -> None:
    fields = list(rows[0]) + ["model_cm-1", "signed_error_cm-1"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, predicted in zip(rows, prediction, strict=True):
            writer.writerow(
                {
                    **row,
                    "model_cm-1": float(predicted),
                    "signed_error_cm-1": float(predicted - row["target_total_cm-1"]),
                }
            )


def write_direct_predictions(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) + ["signed_error_cm-1"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "signed_error_cm-1": float(
                        row["model_cm-1"] - row["target_total_cm-1"]
                    ),
                }
            )


def matrix_replay_gate(
    rows: list[dict],
    prediction: np.ndarray,
    phonon,
    force_constants: np.ndarray,
) -> dict:
    qpoints = []
    for row in rows:
        if row["group"] == "zero":
            qpoints.append(zero_qpoint(row["direction"], row["distance"]))
        else:
            t_value = 1.0 + row["signed_distance"]
            qpoints.append(np.asarray([t_value / 3.0, t_value / 3.0, 0.0]))
    sequence = b0.dynamical_sequence(
        phonon, force_constants, np.asarray(qpoints, float)
    )
    selected = np.argmax(sequence["frequencies_cm1"], axis=1)
    baseline = np.asarray(
        [sequence["frequencies_cm1"][index, mode] for index, mode in enumerate(selected)]
    )
    expected_background = np.asarray([row["background_cm-1"] for row in rows], float)
    background_replay = float(np.max(np.abs(baseline - expected_background)))
    delta_lambda = np.asarray(prediction, float) ** 2 - baseline**2
    applied = b0.apply_rank_one(sequence, selected, delta_lambda)
    corrected = np.asarray(applied["corrected_frequencies_cm1"][:, -1], float)
    sorted_frequency = np.sort(sequence["frequencies_cm1"], axis=1)
    return {
        "background_frequency_replay_max_abs_cm-1": background_replay,
        "Hermiticity_max_abs": float(applied["Hermiticity_max_abs"]),
        "rank_one_frequency_replay_max_abs_cm-1": float(
            np.max(np.abs(corrected - prediction))
        ),
        "minimum_top_mode_isolation_cm-1": float(
            np.min(sorted_frequency[:, -1] - sorted_frequency[:, -2])
        ),
    }


def make_figure(
    path: Path,
    rows: list[dict],
    prediction: np.ndarray,
    direct_rows: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.3), sharex=True)
    for axis, group in zip(axes.flat, GROUPS, strict=True):
        indices = [index for index, row in enumerate(rows) if row["group"] == group]
        x = np.asarray([rows[index]["signed_distance"] for index in indices], float)
        target = np.asarray([rows[index]["target_total_cm-1"] for index in indices], float)
        predicted = np.asarray(prediction)[indices]
        order = np.argsort(x)
        dense_x = np.linspace(float(x.min()), float(x.max()), 801)
        dense_model = PchipInterpolator(x[order], predicted[order])(dense_x)
        axis.plot(
            dense_x,
            dense_model,
            color="#0072B2",
            lw=1.8,
            label="MLIP + joint long-range adapter",
        )
        if group == "zero":
            axis.plot(x, target, "o", color="#202020", ms=4.0, label="no-smearing DFPT")
            title = "smearing/degauss = 0 (tetrahedra_opt)"
        else:
            axis.plot(x, target, ".", color="#999999", ms=3.2, label="EPW shape + DFPT K anchor")
            direct = [row for row in direct_rows if row["group"] == group]
            axis.plot(
                [row["signed_distance"] for row in direct],
                [row["target_total_cm-1"] for row in direct],
                "o",
                color="#202020",
                ms=3.8,
                label="direct DFPT",
            )
            title = f"smearing/degauss = {rows[indices[0]]['degauss_Ry']:.7f} Ry"
        axis.set_title(title, fontsize=9.8)
        axis.grid(axis="y", color="#E6E6E6", lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(-0.026, 0.026)
        axis.set_xlabel(r"signed distance from K")
        axis.set_ylabel(r"tracked $A_1'$ frequency (cm$^{-1}$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    finite_handles, finite_labels = axes[0, 1].get_legend_handles_labels()
    for handle, label in zip(finite_handles, finite_labels, strict=True):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.80, 0.5), frameon=False)
    fig.suptitle("Graphene K A′: one shared zero/finite-smearing long-range kernel", y=0.98)
    fig.subplots_adjust(left=0.09, right=0.78, top=0.92, bottom=0.09, hspace=0.28, wspace=0.20)
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zero-data",
        type=Path,
        default=ROOT / "results/graphene_k_cusp_nosmear/S4b_v2_development/v2_development_predictions.csv",
    )
    parser.add_argument(
        "--zero-kgrid",
        type=Path,
        default=ROOT / "results/graphene_k_cusp_nosmear/S4c_d003_k288/d003_k240_k288_summary.json",
    )
    parser.add_argument(
        "--dense-finite",
        type=Path,
        default=ROOT / "results/graphene_kohn_cusp_two_methods/B0_dense_k29/b0_dense_predictions.csv",
    )
    parser.add_argument(
        "--direct-finite",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E0_shared_dirac/shared_dirac_predictions.csv"
        ),
    )
    parser.add_argument(
        "--static-short",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility/source_p4_450"
            / "on_policy/frozen_prediction/frozen_static_fc2.npz"
        ),
    )
    parser.add_argument(
        "--operator-geometry", type=Path, default=a0.operator_path(450)
    )
    parser.add_argument(
        "--wannier-diagnostic",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility/E0_epw_matched"
            / "wannier_diagnostics/ex1_pifroz_k18q9/wannier_diagnostic_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E4_joint_zero_finite_rank1"
        ),
    )
    args = parser.parse_args()

    wannier_diagnostic = json.loads(
        args.wannier_diagnostic.read_text(encoding="utf-8")
    )
    production_wannier = wannier_diagnostic["matched_15A_k12q6"]
    global V_F_EVA_FIXED
    V_F_EVA_FIXED = float(
        0.5
        * (
            production_wannier["vF_conduction_eVA"]
            + production_wannier["vF_valence_eVA"]
        )
    )

    phonon, force_constants = make_static_short_phonon(
        args.operator_geometry, args.static_short
    )
    zero_rows = load_zero_rows(
        args.zero_data, args.zero_kgrid, phonon, force_constants
    )
    finite_rows, direct = load_finite_teacher_rows(
        args.dense_finite, args.direct_finite, phonon, force_constants
    )
    rows = zero_rows + finite_rows
    grids = {
        "width": np.linspace(0.50, 6.00, 23),
        "response": np.geomspace(5.0e-5, 3.0e-2, 23),
        "narrow": np.linspace(0.003, 0.050, 20),
    }

    evaluations = []
    predictions = {}
    for candidate in CANDIDATES:
        evaluation, prediction = candidate_evaluation(rows, candidate, grids)
        evaluations.append(evaluation)
        predictions[candidate] = prediction
    selected = min(evaluations, key=lambda item: item["selection_score"])["candidate"]
    selected_evaluation = next(item for item in evaluations if item["candidate"] == selected)
    selected_fit = selected_evaluation["full_fit"]
    resolution_profile = narrow_scale_profile(
        rows, direct, phonon, force_constants, selected, grids
    )
    reference_sensitivity = zero_reference_sensitivity(
        rows,
        args.zero_data,
        args.zero_kgrid,
        direct,
        phonon,
        force_constants,
        selected,
        grids,
    )
    direct_metrics, direct_rows = evaluate_direct_finite(
        direct, phonon, force_constants, selected_fit, selected
    )
    max_direct_kink_error = max(
        float(metrics["kink_relative_error"]) for metrics in direct_metrics.values()
    )
    zero_shape = selected_fit["zero_shape"]
    max_zero_depth_error = max(
        float(zero_shape[direction]["d003_cusp_depth_relative_error"])
        for direction in ("KG", "KM")
    )
    matrix_gate = matrix_replay_gate(
        rows, predictions[selected], phonon, force_constants
    )
    gates = {
        "finite_direct_K_kink_relative_error_lt_0p20": max_direct_kink_error < 0.20,
        "finite_direct_central_MAE_lt_1_cm-1": max(
            float(metrics["central_block_MAE_cm-1"])
            for metrics in direct_metrics.values()
        ) < 1.0,
        "zero_d003_cusp_depth_relative_error_lt_0p20": max_zero_depth_error < 0.20,
        "zero_shape_ratio_relative_error_lt_0p20": max(
            float(zero_shape[direction]["slope_ratio_relative_error"])
            for direction in ("KG", "KM")
        ) < 0.20,
        "narrow_scale_not_below_data_resolution": (
            "two_scale" not in selected
            or selected_fit["narrow_scale"] >= 0.003
        ),
        "matrix_Hermiticity_max_le_1e-10": (
            matrix_gate["Hermiticity_max_abs"] <= 1.0e-10
        ),
        "rank_one_frequency_replay_max_le_1e-6_cm-1": (
            matrix_gate["rank_one_frequency_replay_max_abs_cm-1"] <= 1.0e-6
        ),
    }
    parameter_boundary = {
        "width_scale_at_grid_boundary": bool(
            np.isclose(selected_fit["width_scale"], grids["width"][0])
            or np.isclose(selected_fit["width_scale"], grids["width"][-1])
        ),
        "response_scale_at_grid_boundary": bool(
            np.isclose(selected_fit["response_scale_Ry"], grids["response"][0])
            or np.isclose(selected_fit["response_scale_Ry"], grids["response"][-1])
        ),
        "narrow_scale_at_grid_boundary": bool(
            "two_scale" in selected
            and (
                np.isclose(selected_fit["narrow_scale"], grids["narrow"][0])
                or np.isclose(selected_fit["narrow_scale"], grids["narrow"][-1])
            )
        ),
    }
    if not all(gates.values()):
        status = "joint_development_shape_gate_failed"
    elif any(parameter_boundary.values()):
        status = "joint_development_shape_gate_passed_parameter_boundary"
    else:
        status = "joint_development_shape_gate_passed"
    summary = {
        "status": status,
        "scope": (
            "development-only shared zero/finite-smearing rank-one scalar kernel; "
            "fixed short-range MLIP and fixed A-prime projector; no new DFT"
        ),
        "selected_candidate": selected,
        "selection_rule": "minimum balanced LOSO RMSE plus 0.03 cm-1 per coefficient beyond four",
        "physical_form": {
            "operator": "D = D_short_MLIP + U_Aprime(q) DeltaLambda(r,s) U_Aprime(q)^dagger",
            "rounded_coordinate": "R=sqrt(r^2+xi(s)^2)-xi(s), xi proportional to smearing/v_F",
            "zero_limit": "R(r,0)=|r| gives a finite one-sided cusp slope",
            "finite_smearing_limit": "R is quadratic at K and linear outside the thermal width",
            "anchor": "separate saturating function of smearing",
            "narrow_cusp": "ell*R/(R+ell) supplies the observed zero-smearing inner/outer slope change",
            "fixed_v_F_eV_A": V_F_EVA_FIXED,
        },
        "candidates": evaluations,
        "selected_direct_finite_metrics": direct_metrics,
        "selected_max_direct_finite_K_kink_relative_error": max_direct_kink_error,
        "selected_max_zero_d003_cusp_depth_relative_error": max_zero_depth_error,
        "matrix_replay": matrix_gate,
        "gates": gates,
        "parameter_boundary": parameter_boundary,
        "narrow_scale_resolution_profile": resolution_profile,
        "zero_reference_sensitivity": reference_sensitivity,
        "nonlinear_search_grid": {
            key: [float(values[0]), float(values[-1]), int(len(values))]
            for key, values in grids.items()
        },
        "inputs": {
            "zero_data": {"path": str(args.zero_data), "sha256": sha256(args.zero_data)},
            "zero_kgrid": {"path": str(args.zero_kgrid), "sha256": sha256(args.zero_kgrid)},
            "dense_finite": {"path": str(args.dense_finite), "sha256": sha256(args.dense_finite)},
            "direct_finite": {"path": str(args.direct_finite), "sha256": sha256(args.direct_finite)},
            "static_short": {"path": str(args.static_short), "sha256": sha256(args.static_short)},
            "operator_geometry": {"path": str(args.operator_geometry), "sha256": sha256(args.operator_geometry)},
            "production_wannier_diagnostic": {
                "path": str(args.wannier_diagnostic),
                "sha256": sha256(args.wannier_diagnostic),
                "Hamiltonian_source": production_wannier["source"],
                "v_F_conduction_eV_A": production_wannier["vF_conduction_eVA"],
                "v_F_valence_eV_A": production_wannier["vF_valence_eVA"],
            },
        },
        "limitations": [
            "All four smearing groups are opened development data; this is not an independent validation.",
            "The zero-smearing d=0.003 k288 depth is still changing by about 12% from k240, so the 20% shape gate is commensurate with present reference uncertainty.",
            "Finite-smearing dense shapes are distilled from the existing EPW operator and anchored at direct-DFPT K; transfer to a new material still requires a band-response parameterization or a few calibration matrices.",
            "The fitted response scale between zero and the smallest finite smearing is weakly identified because no existing data lie in that interval.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "joint_zero_finite_rank1_summary.json", summary)
    write_predictions(
        args.output_dir / "joint_zero_finite_rank1_predictions.csv",
        rows,
        predictions[selected],
    )
    write_direct_predictions(
        args.output_dir / "joint_zero_finite_direct_dfpt_evaluation.csv",
        direct_rows,
    )
    model_path = args.output_dir / "joint_rank1_development_model.npz"
    with model_path.open("wb") as handle:
        np.savez(
            handle,
            status=np.asarray(status),
            candidate=np.asarray(selected),
            feature_names=np.asarray(feature_names(selected)),
            coefficients_cm2=np.asarray(selected_fit["coefficients_cm-2"], float),
            width_scale=np.asarray(selected_fit["width_scale"]),
            response_scale_Ry=np.asarray(selected_fit["response_scale_Ry"]),
            narrow_scale=np.asarray(selected_fit["narrow_scale"]),
            fixed_v_F_eV_A=np.asarray(V_F_EVA_FIXED),
        )
    make_figure(
        args.output_dir / "joint_zero_finite_rank1_comparison.png",
        rows,
        predictions[selected],
        direct_rows,
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
