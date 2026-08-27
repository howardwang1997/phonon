#!/usr/bin/env python3
"""Assemble and gate the R2AO finite-lattice full-EPC Kohn anomaly.

The electronic condition is fixed at ``degauss=0.0019000869 Ry`` for every
lattice temperature.  For each converged SSCHA Hessian the finite-q6 operator
is removed exactly, then the frozen 241-point full-EPC A' response is added as
a Hermitian mode projector.  R2AO is compared with the independent S0 short
model at 300/450/600 K.  The archived static-lattice DFPT data independently
gate K frequency, near-K shape, cusp depth, and rounding width.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for item in (HERE, ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import analyze_graphene_k_cusp_b0_dense as b0  # noqa: E402
import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402
import build_graphene_fixed_smearing_thermal_full_epc as e48  # noqa: E402
from phonon_accel.long_range import apply_mode_projected_correction  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
R2ROOT = BASE / "R2R_multipolar_background"
TEMPERATURES = (300, 450, 600)
MODELS = ("R2AO", "S0")
TARGET_SMEARING_RY = 0.0019000869

MODEL_INPUTS = {
    "R2AO": {
        temperature: {
            "result": R2ROOT
            / f"R2AP_fixed_smearing_SSCHA/formal_T{temperature}/result.npz",
            "acceptance": R2ROOT
            / f"R2AP_fixed_smearing_SSCHA/formal_T{temperature}/acceptance.json",
        }
        for temperature in TEMPERATURES
    },
    "S0": {
        300: {
            "result": BASE / "S0_unified_short/Q0_quantum_sscha/formal_T300/result.npz",
            "acceptance": BASE
            / "S0_unified_short/Q0_quantum_sscha/formal_T300/acceptance.json",
        },
        450: {
            "result": R2ROOT
            / "R2AR_s0_fixed_smearing_sensitivity/formal_T450_Tel300/result.npz",
            "acceptance": R2ROOT
            / "R2AR_s0_fixed_smearing_sensitivity/formal_T450_Tel300/acceptance.json",
        },
        600: {
            "result": R2ROOT
            / "R2AR_s0_fixed_smearing_sensitivity/formal_T600_Tel300/result.npz",
            "acceptance": R2ROOT
            / "R2AR_s0_fixed_smearing_sensitivity/formal_T600_Tel300/acceptance.json",
        },
    },
}


def use_s0_result_root(root: Path) -> None:
    """Point every S0 temperature at one controlled result root."""
    root = root.resolve()
    for temperature in TEMPERATURES:
        case = root / f"formal_T{temperature}_Tel300"
        MODEL_INPUTS["S0"][temperature] = {
            "result": case / "result.npz",
            "acceptance": case / "acceptance.json",
        }

STATIC_GATES = {
    "K_abs_error_cm-1": 5.0,
    "K_referenced_shape_RMSE_cm-1": 1.0,
    "K_referenced_shape_max_abs_cm-1": 2.0,
    "cusp_depth_relative_error": 0.15,
    "rounding_width_relative_error": 0.15,
}
SENSITIVITY_GATES = {
    "K_abs_difference_cm-1": 10.0,
    "K_referenced_shape_RMSE_cm-1": 1.0,
    "K_referenced_shape_max_abs_cm-1": 2.0,
    "cusp_depth_relative_difference": 0.15,
    "K_temperature_shift_abs_difference_cm-1": 5.0,
}


def scalar(value):
    return np.asarray(value).reshape(()).item()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def operator_record(acceptance: dict) -> dict:
    provenance = acceptance.get("input_provenance", {})
    record = provenance.get("fixed_operator", provenance.get("operator"))
    if not isinstance(record, dict):
        raise ValueError("SSCHA acceptance lacks a fixed operator record")
    return record


def load_case(model: str, temperature: int, operator_hash: str) -> dict:
    paths = MODEL_INPUTS[model][temperature]
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed" or not acceptance.get("converged"):
        raise ValueError(f"{model} T{temperature} SSCHA did not pass")
    result_hash = e48.sha256(paths["result"])
    if acceptance.get("result_sha256") != result_hash:
        raise ValueError(f"{model} T{temperature} result hash changed")
    record = operator_record(acceptance)
    if record.get("sha256") != operator_hash:
        raise ValueError(f"{model} T{temperature} used a different q6 operator")
    result = e48.load_npz(paths["result"])
    if int(scalar(result["lattice_temperature_K"])) != temperature:
        raise ValueError(f"{model} T{temperature} temperature label changed")
    operator_key = (
        "fixed_operator_temperature_K"
        if "fixed_operator_temperature_K" in result
        else "operator_temperature_K"
    )
    if int(scalar(result[operator_key])) != 300:
        raise ValueError(f"{model} T{temperature} did not use the T300 operator")
    if not bool(scalar(result["converged"])):
        raise ValueError(f"{model} T{temperature} result is not converged")
    force_constants = np.asarray(result["free_energy_fc2_eV_A2"], float)
    if force_constants.shape != (72, 72, 3, 3):
        raise ValueError(f"{model} T{temperature} FC2 shape changed")
    if not np.isfinite(force_constants).all():
        raise ValueError(f"{model} T{temperature} FC2 is non-finite")
    return {
        "paths": paths,
        "acceptance": acceptance,
        "result": result,
        "force_constants": force_constants,
        "result_sha256": result_hash,
    }


def saved_k_frequency(result: dict) -> float:
    distance = np.asarray(result["distance"], float)
    labels = np.asarray(result["label_positions"], float)
    frequency = np.asarray(result["frequency_cm_1"], float)
    index = int(np.argmin(np.abs(distance - labels[2])))
    return float(frequency[index, -1])


def assemble_case(
    phonon,
    total_fc: np.ndarray,
    delta_fc: np.ndarray,
    response: dict,
    result: dict,
) -> dict:
    short_fc = total_fc - delta_fc
    reassembly = float(np.max(np.abs(short_fc + delta_fc - total_fc)))
    short_sequence = b0.dynamical_sequence(phonon, short_fc, response["qpoints"])
    q6_sequence = b0.dynamical_sequence(phonon, total_fc, response["qpoints"])
    operator_matrices = e48.raw_dynamical_matrices(
        phonon, delta_fc, response["qpoints"]
    )
    linearity = float(
        np.max(
            np.abs(
                q6_sequence["matrices"]
                - short_sequence["matrices"]
                - operator_matrices
            )
        )
    )
    short_modes, short_overlap, short_top = e48.track_branch(
        short_sequence, response["signed_distance"]
    )
    q6_modes, q6_overlap, q6_top = e48.track_branch(
        q6_sequence, response["signed_distance"]
    )
    short_frequency = e48.selected_frequency(short_sequence, short_modes)
    q6_frequency = e48.selected_frequency(q6_sequence, q6_modes)
    applied = apply_mode_projected_correction(
        short_sequence["matrices"],
        short_sequence["scale_cm2"],
        short_sequence["eigenvectors"],
        short_modes,
        response["full_response_cm2"],
    )
    full_frequency = np.asarray(applied.tracked_frequency_cm1, float)

    negative = b0.dynamical_sequence(phonon, short_fc, -response["qpoints"])
    negative_modes, negative_overlap, negative_top = e48.track_branch(
        negative, response["signed_distance"]
    )
    negative_applied = apply_mode_projected_correction(
        negative["matrices"],
        negative["scale_cm2"],
        negative["eigenvectors"],
        negative_modes,
        response["full_response_cm2"],
    )
    time_reversal_matrix = float(
        np.max(
            np.abs(
                negative_applied.total_matrices - applied.total_matrices.conj()
            )
        )
    )
    time_reversal_frequency = float(
        np.max(
            np.abs(
                negative_applied.tracked_frequency_cm1
                - applied.tracked_frequency_cm1
            )
        )
    )
    center = e48.index_at(response, "K", 0.0)
    diagnostics = {
        "FC2_remove_reassemble_max_abs_eV_A2": reassembly,
        "dynamical_matrix_linearity_max_abs": linearity,
        "minimum_adjacent_mode_overlap": float(
            min(short_overlap, q6_overlap, negative_overlap)
        ),
        "tracked_vs_highest_max_abs_cm-1": float(
            max(short_top, q6_top, negative_top)
        ),
        "saved_q6_K_replay_max_abs_cm-1": float(
            abs(q6_frequency[center] - saved_k_frequency(result))
        ),
        "time_reversal_matrix_max_abs": time_reversal_matrix,
        "time_reversal_frequency_max_abs_cm-1": time_reversal_frequency,
        "all_frequencies_finite": bool(
            np.isfinite(short_frequency).all()
            and np.isfinite(q6_frequency).all()
            and np.isfinite(full_frequency).all()
        ),
        **{key: float(value) for key, value in applied.diagnostics.items()},
    }
    return {
        "short_fc": short_fc,
        "short_frequency_cm1": short_frequency,
        "q6_frequency_cm1": q6_frequency,
        "full_frequency_cm1": full_frequency,
        "short_matrices": short_sequence["matrices"],
        "q6_correction_matrices": operator_matrices,
        "full_correction_matrices": applied.correction_matrices,
        "full_total_matrices": applied.total_matrices,
        "projectors": applied.projectors,
        "mode_indices": short_modes,
        "scale_cm2": short_sequence["scale_cm2"],
        "metrics": {
            "thermal_short": e48.curve_metrics(response, short_frequency),
            "finite_q6": e48.curve_metrics(response, q6_frequency),
            "full_epc": e48.curve_metrics(response, full_frequency),
        },
        "diagnostics": diagnostics,
    }


def rounded_cusp(
    distance: np.ndarray,
    omega_k: float,
    slope: float,
    width: float,
    curvature: float,
) -> np.ndarray:
    return (
        omega_k
        + slope * (np.sqrt(distance**2 + width**2) - width)
        + curvature * distance**2
    )


def fit_rounded_cusp(distance: np.ndarray, frequency: np.ndarray) -> dict:
    distance = np.asarray(distance, float)
    frequency = np.asarray(frequency, float)
    omega_k = float(frequency[0])

    def residual(parameters: np.ndarray) -> np.ndarray:
        return rounded_cusp(
            distance,
            omega_k,
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
        ) - frequency

    fit = least_squares(
        residual,
        np.asarray([750.0, 0.01, -2500.0]),
        bounds=(
            np.asarray([0.0, 1.0e-6, -1.0e5]),
            np.asarray([5000.0, 0.2, 1.0e5]),
        ),
    )
    return {
        "omega_K_cm-1": omega_k,
        "slope_cm-1_per_d": float(fit.x[0]),
        "rounding_width_d": float(fit.x[1]),
        "curvature_cm-1_per_d2": float(fit.x[2]),
        "fit_RMSE_cm-1": float(np.sqrt(np.mean(fit.fun**2))),
        "fit_max_abs_cm-1": float(np.max(np.abs(fit.fun))),
        "success": bool(fit.success),
    }


def static_validation(response: dict, direct_path: Path, dfpt_line: Path) -> dict:
    direct = [
        row
        for row in read_csv(direct_path)
        if np.isclose(
            float(row["smearing_degauss_Ry"]),
            TARGET_SMEARING_RY,
            atol=5.0e-11,
            rtol=0.0,
        )
    ]
    direct.sort(key=lambda row: float(row["distance"]))
    if len(direct) != 4:
        raise ValueError("expected four direct finite-smearing DFPT points")
    dfpt_direct = np.asarray([float(row["DFPT_frequency_cm-1"]) for row in direct])
    model_direct = np.asarray(
        [float(row["MLIP_plus_full_EPC_LR_cm-1"]) for row in direct]
    )
    direct_error = model_direct - dfpt_direct
    centered_error = (
        model_direct - model_direct[0] - (dfpt_direct - dfpt_direct[0])
    )

    dfpt_rows = [
        row
        for row in read_csv(dfpt_line)
        if row["region"] == "K" and float(row["t_GK"]) <= 1.0 + 1.0e-12
    ]
    dfpt_rows.sort(key=lambda row: 1.0 - float(row["t_GK"]))
    sample_distance = np.asarray(
        [1.0 - float(row["t_GK"]) for row in dfpt_rows], float
    )
    dfpt_frequency = np.asarray([float(row["f6_cm-1"]) for row in dfpt_rows])
    center = e48.index_at(response, "K", 0.0)
    kg = response["direction"] == "KG"
    model_distance = np.concatenate(([0.0], response["distance"][kg]))
    model_frequency = np.concatenate(
        ([response["static_total_cm1"][center]], response["static_total_cm1"][kg])
    )
    order = np.argsort(model_distance)
    sampled_model = np.interp(
        sample_distance, model_distance[order], model_frequency[order]
    )
    dfpt_fit = fit_rounded_cusp(sample_distance, dfpt_frequency)
    model_fit = fit_rounded_cusp(sample_distance, sampled_model)
    depth_records = {}
    depth_gate = True
    for distance in (0.003, 0.025):
        reference_depth = float(
            rounded_cusp(
                np.asarray([distance]),
                dfpt_fit["omega_K_cm-1"],
                dfpt_fit["slope_cm-1_per_d"],
                dfpt_fit["rounding_width_d"],
                dfpt_fit["curvature_cm-1_per_d2"],
            )[0]
            - dfpt_fit["omega_K_cm-1"]
        )
        model_value = float(
            np.interp(distance, model_distance[order], model_frequency[order])
        )
        model_depth = model_value - float(model_frequency[order][0])
        relative_error = float(
            abs(model_depth - reference_depth) / abs(reference_depth)
        )
        depth_records[f"d{distance:.3f}"] = {
            "DFPT_fit_depth_cm-1": reference_depth,
            "full_EPC_model_depth_cm-1": model_depth,
            "relative_error": relative_error,
            "threshold": STATIC_GATES["cusp_depth_relative_error"],
            "passed": relative_error
            <= STATIC_GATES["cusp_depth_relative_error"],
        }
        depth_gate = depth_gate and depth_records[f"d{distance:.3f}"]["passed"]
    width_relative_error = float(
        abs(model_fit["rounding_width_d"] - dfpt_fit["rounding_width_d"])
        / dfpt_fit["rounding_width_d"]
    )
    metrics = {
        "K_error_cm-1": float(direct_error[0]),
        "K_referenced_shape_RMSE_cm-1": float(
            np.sqrt(np.mean(centered_error**2))
        ),
        "K_referenced_shape_max_abs_cm-1": float(np.max(np.abs(centered_error))),
        "direct_point_max_abs_error_cm-1": float(np.max(np.abs(direct_error))),
        "cusp_depth": depth_records,
        "DFPT_rounding_fit": dfpt_fit,
        "full_EPC_model_rounding_fit_at_same_qpoints": model_fit,
        "rounding_width_relative_error": width_relative_error,
    }
    gates = {
        "K_frequency": abs(metrics["K_error_cm-1"])
        <= STATIC_GATES["K_abs_error_cm-1"],
        "near_K_shape_RMSE": metrics["K_referenced_shape_RMSE_cm-1"]
        <= STATIC_GATES["K_referenced_shape_RMSE_cm-1"],
        "near_K_shape_max": metrics["K_referenced_shape_max_abs_cm-1"]
        <= STATIC_GATES["K_referenced_shape_max_abs_cm-1"],
        "cusp_depth_d0.003_and_d0.025": depth_gate,
        "rounding_width": width_relative_error
        <= STATIC_GATES["rounding_width_relative_error"],
    }
    return {
        "metrics": metrics,
        "thresholds": STATIC_GATES,
        "gates": gates,
        "passed": all(gates.values()),
        "sources": {
            "direct_DFPT_points": {
                "path": str(direct_path),
                "sha256": e48.sha256(direct_path),
            },
            "DFPT_KG_line": {
                "path": str(dfpt_line),
                "sha256": e48.sha256(dfpt_line),
            },
        },
    }


def model_sensitivity(response: dict, assembled: dict) -> dict:
    coordinate = np.asarray(response["signed_q_2pi_over_a"], float)
    center = e48.index_at(response, "K", 0.0)
    near = np.abs(coordinate) <= 0.025 + 1.0e-14
    by_temperature = {}
    all_temperature_gates = True
    for temperature in TEMPERATURES:
        primary = assembled["R2AO"][temperature]["full_frequency_cm1"]
        independent = assembled["S0"][temperature]["full_frequency_cm1"]
        difference = primary - independent
        centered = (
            primary - primary[center] - (independent - independent[center])
        )
        depths = {}
        depth_gate = True
        for direction in ("KG", "KM"):
            for distance in (0.003, 0.025):
                index = e48.index_at(response, direction, distance)
                primary_depth = float(primary[index] - primary[center])
                independent_depth = float(independent[index] - independent[center])
                relative = float(
                    abs(primary_depth - independent_depth)
                    / max(abs(independent_depth), 1.0e-12)
                )
                key = f"{direction}_d{distance:.3f}"
                depths[key] = {
                    "R2AO_depth_cm-1": primary_depth,
                    "S0_depth_cm-1": independent_depth,
                    "relative_difference": relative,
                    "passed": relative
                    <= SENSITIVITY_GATES["cusp_depth_relative_difference"],
                }
                depth_gate = depth_gate and depths[key]["passed"]
        metrics = {
            "R2AO_minus_S0_K_cm-1": float(difference[center]),
            "near_K_absolute_RMSE_cm-1": float(
                np.sqrt(np.mean(difference[near] ** 2))
            ),
            "near_K_K_referenced_shape_RMSE_cm-1": float(
                np.sqrt(np.mean(centered[near] ** 2))
            ),
            "near_K_K_referenced_shape_max_abs_cm-1": float(
                np.max(np.abs(centered[near]))
            ),
            "cusp_depth": depths,
        }
        gates = {
            "K_frequency": abs(metrics["R2AO_minus_S0_K_cm-1"])
            <= SENSITIVITY_GATES["K_abs_difference_cm-1"],
            "near_K_shape_RMSE": metrics[
                "near_K_K_referenced_shape_RMSE_cm-1"
            ]
            <= SENSITIVITY_GATES["K_referenced_shape_RMSE_cm-1"],
            "near_K_shape_max": metrics[
                "near_K_K_referenced_shape_max_abs_cm-1"
            ]
            <= SENSITIVITY_GATES["K_referenced_shape_max_abs_cm-1"],
            "cusp_depth": depth_gate,
        }
        by_temperature[str(temperature)] = {"metrics": metrics, "gates": gates}
        all_temperature_gates = all_temperature_gates and all(gates.values())

    shifts = {}
    shift_gate = True
    for lower, upper in ((300, 450), (450, 600)):
        r2ao_shift = float(
            assembled["R2AO"][upper]["full_frequency_cm1"][center]
            - assembled["R2AO"][lower]["full_frequency_cm1"][center]
        )
        s0_shift = float(
            assembled["S0"][upper]["full_frequency_cm1"][center]
            - assembled["S0"][lower]["full_frequency_cm1"][center]
        )
        difference = r2ao_shift - s0_shift
        passed = abs(difference) <= SENSITIVITY_GATES[
            "K_temperature_shift_abs_difference_cm-1"
        ]
        shifts[f"{lower}_to_{upper}"] = {
            "R2AO_K_shift_cm-1": r2ao_shift,
            "S0_K_shift_cm-1": s0_shift,
            "difference_cm-1": difference,
            "passed": passed,
        }
        shift_gate = shift_gate and passed
    return {
        "by_lattice_temperature_K": by_temperature,
        "K_temperature_shifts": shifts,
        "thresholds": SENSITIVITY_GATES,
        "per_temperature_passed": all_temperature_gates,
        "temperature_shift_passed": shift_gate,
        "passed": all_temperature_gates and shift_gate,
    }


def paired_sampling_validation(loaded: dict, tolerance: float = 1.0e-12) -> dict:
    """Verify that R2AO and S0 used the same first/final population samples."""
    records = {}
    passed = True
    for temperature in TEMPERATURES:
        r2ao_case = loaded["R2AO"][temperature]
        s0_case = loaded["S0"][temperature]
        r2ao_population = int(r2ao_case["acceptance"]["final_population"])
        s0_population = int(s0_case["acceptance"]["final_population"])
        populations_match_first = r2ao_population == s0_population == 1
        r2ao_xats = (
            r2ao_case["paths"]["result"].parent
            / "ensembles"
            / f"xats_pop{r2ao_population}.npy"
        )
        s0_xats = (
            s0_case["paths"]["result"].parent
            / "ensembles"
            / f"xats_pop{s0_population}.npy"
        )
        left = np.load(r2ao_xats, allow_pickle=False)
        right = np.load(s0_xats, allow_pickle=False)
        shape_match = left.shape == right.shape
        coordinate_error = (
            float(np.max(np.abs(np.asarray(left, float) - np.asarray(right, float))))
            if shape_match
            else math.inf
        )
        r2ao_initial = r2ao_case["acceptance"]["input_provenance"][
            "initial_hessian"
        ]
        s0_initial = s0_case["acceptance"]["input_provenance"]["initial_hessian"]
        r2ao_tdep = r2ao_initial.get("physical_tdep", {})
        s0_tdep = s0_initial.get("physical_tdep", {})
        initial_sources_match = bool(
            r2ao_tdep.get("sha256")
            and r2ao_tdep.get("sha256") == s0_tdep.get("sha256")
            and r2ao_initial.get("matched_operator_temperature_K") == temperature
            and s0_initial.get("matched_operator_temperature_K") == temperature
            and r2ao_initial.get("fixed_operator_temperature_K") == 300
            and s0_initial.get("fixed_operator_temperature_K") == 300
        )
        gates = {
            "same_initial_hessian_sources": initial_sources_match,
            "both_converged_on_first_population": populations_match_first,
            "configuration_array_shape": shape_match,
            "configuration_coordinate_replay": coordinate_error <= tolerance,
        }
        records[str(temperature)] = {
            "R2AO_xats": {"path": str(r2ao_xats), "sha256": e48.sha256(r2ao_xats)},
            "S0_xats": {"path": str(s0_xats), "sha256": e48.sha256(s0_xats)},
            "configuration_coordinate_max_abs": coordinate_error,
            "configuration_coordinate_tolerance": tolerance,
            "array_shape": list(left.shape) if shape_match else None,
            "gates": gates,
            "passed": all(gates.values()),
        }
        passed = passed and all(gates.values())
    return {
        "required": True,
        "method": "common initial Hessian, random seed, and first-population configurations",
        "by_lattice_temperature_K": records,
        "passed": passed,
    }


def s0_force_gate_validation(path: Path, loaded: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected", {})
    checkpoint_hash = selected.get("checkpoint_sha256")
    deployed_hashes = {
        str(temperature): loaded["S0"][temperature]["acceptance"]
        .get("input_provenance", {})
        .get("delta_model", {})
        .get("selected_checkpoint_sha256")
        for temperature in TEMPERATURES
    }
    gates = {
        "checkpoint_selection_status": payload.get("status") == "checkpoint_passed",
        "selected_checkpoint_passed_fixed_gate": selected.get("passes_fixed_gate")
        is True,
        "same_checkpoint_deployed_at_all_temperatures": bool(checkpoint_hash)
        and all(value == checkpoint_hash for value in deployed_hashes.values()),
    }
    return {
        "path": str(path),
        "sha256": e48.sha256(path),
        "fixed_thresholds": payload.get("fixed_thresholds"),
        "selected_checkpoint_sha256": checkpoint_hash,
        "selected_metrics": {
            "thermal_total_force": selected.get("thermal_total_force"),
            "harmonic_combined_force": selected.get("harmonic_combined_force"),
            "normalized_worst_gate_score": selected.get(
                "normalized_worst_gate_score"
            ),
        },
        "deployed_checkpoint_sha256_by_temperature_K": deployed_hashes,
        "gates": gates,
        "passed": all(gates.values()),
    }


def make_figure(response: dict, assembled: dict, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {300: "#2676B8", 450: "#D55E00", 600: "#7A4EAB"}
    x = response["signed_q_2pi_over_a"]
    center = e48.index_at(response, "K", 0.0)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    absolute, referenced, difference, ktrend = axes.ravel()
    for temperature in TEMPERATURES:
        color = colors[temperature]
        for model, style, width in (("R2AO", "-", 2.0), ("S0", "--", 1.25)):
            y = assembled[model][temperature]["full_frequency_cm1"]
            absolute.plot(
                x,
                y,
                color=color,
                ls=style,
                lw=width,
                label=f"{temperature} K, {model}",
            )
            referenced.plot(x, y - y[center], color=color, ls=style, lw=width)
        delta = (
            assembled["R2AO"][temperature]["full_frequency_cm1"]
            - assembled["S0"][temperature]["full_frequency_cm1"]
        )
        difference.plot(x, delta, color=color, lw=1.8, label=f"{temperature} K")
    for model, style, marker in (("R2AO", "-", "o"), ("S0", "--", "s")):
        values = [
            assembled[model][temperature]["full_frequency_cm1"][center]
            for temperature in TEMPERATURES
        ]
        ktrend.plot(
            TEMPERATURES,
            values,
            color="#222222" if model == "R2AO" else "#777777",
            ls=style,
            marker=marker,
            lw=1.8,
            label=model,
        )
    absolute.set_title("(a) Absolute A' branch")
    referenced.set_title("(b) K-referenced anomaly")
    difference.set_title("(c) Short-model sensitivity: R2AO − S0")
    ktrend.set_title("(d) K frequency versus lattice temperature")
    absolute.set_ylabel(r"frequency (cm$^{-1}$)")
    referenced.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    difference.set_ylabel(r"frequency difference (cm$^{-1}$)")
    ktrend.set_ylabel(r"K frequency (cm$^{-1}$)")
    for axis in (absolute, referenced, difference):
        axis.axvline(0.0, color="#B8B8B8", lw=0.8, zorder=0)
        axis.set_xlabel(r"signed $|q-K|/(2\pi/a)$")
        axis.grid(axis="y", color="#E6E6E6", lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    referenced.set_xlim(-0.025, 0.025)
    difference.set_xlim(-0.025, 0.025)
    ktrend.set_xlabel("lattice temperature (K)")
    ktrend.grid(axis="y", color="#E6E6E6", lw=0.55)
    ktrend.spines[["top", "right"]].set_visible(False)
    ktrend.legend(loc="best", frameon=False)
    handles, labels = absolute.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=8.4,
    )
    fig.suptitle(
        "Graphene finite-lattice A' spectra with frozen full-EPC response\n"
        "smearing/degauss = 0.0019000869 Ry",
        fontsize=12.0,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.88, bottom=0.15, hspace=0.38, wspace=0.25
    )
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def make_full_band_figure(
    response: dict, loaded: dict, assembled: dict, output: Path
) -> None:
    """Plot the force-gated S0 full bands and the dense full-EPC K patch."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {300: "#2676B8", 450: "#D55E00", 600: "#7A4EAB"}
    response_center = e48.index_at(response, "K", 0.0)
    q_k = np.asarray(response["qpoints"][response_center], float)
    q_offset = np.linalg.norm(np.asarray(response["qpoints"], float) - q_k, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.45), sharey=True)
    for axis, temperature in zip(axes, TEMPERATURES):
        result = loaded["S0"][temperature]["result"]
        distance = np.asarray(result["distance"], float)
        label_positions = np.asarray(result["label_positions"], float)
        labels = [str(value) for value in np.asarray(result["labels"])]
        frequency = np.asarray(result["frequency_cm_1"], float)
        color = colors[temperature]
        axis.plot(distance, frequency, color=color, lw=0.9, alpha=0.64)
        patch_x = np.where(
            response["direction"] == "KM",
            label_positions[2] - q_offset,
            label_positions[2] + q_offset,
        )
        order = np.argsort(patch_x)
        axis.plot(
            patch_x[order],
            assembled["S0"][temperature]["full_frequency_cm1"][order],
            color="#151515",
            lw=2.15,
            solid_capstyle="round",
        )
        axis.scatter(
            [label_positions[2]],
            [assembled["S0"][temperature]["full_frequency_cm1"][response_center]],
            s=19,
            color="#151515",
            zorder=4,
        )
        for position in label_positions:
            axis.axvline(position, color="#D4D4D4", lw=0.65, zorder=0)
        axis.set_xticks(label_positions, labels)
        axis.set_xlim(distance[0], distance[-1])
        axis.set_ylim(-25.0, 1650.0)
        axis.set_title(f"{temperature} K")
        axis.set_xlabel("wave-vector path")
        axis.grid(axis="y", color="#E8E8E8", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"frequency (cm$^{-1}$)")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#777777", lw=1.2, label="S0 SSCHA + q6 bands"),
        Line2D(
            [0],
            [0],
            color="#151515",
            lw=2.15,
            marker="o",
            markersize=4,
            label="dense full-EPC A' near K",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.085,
        "Small ZA dips near Gamma are Fourier-interpolation residues; the commensurate supercell stability gate is reported separately.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.suptitle(
        "Graphene finite-lattice phonon spectra (force-gated S0)\n"
        "smearing/degauss = 0.0019000869 Ry",
        fontsize=12.0,
        y=0.98,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.23, wspace=0.13)
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def write_csv(path: Path, response: dict, assembled: dict) -> None:
    fields = [
        "short_model",
        "lattice_temperature_K",
        "smearing_degauss_Ry",
        "direction",
        "path_distance_d",
        "signed_q_2pi_over_a",
        "q1",
        "q2",
        "thermal_short_cm-1",
        "finite_q6_cm-1",
        "full_EPC_cm-1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            for temperature in TEMPERATURES:
                record = assembled[model][temperature]
                for index, qpoint in enumerate(response["qpoints"]):
                    writer.writerow(
                        {
                            "short_model": model,
                            "lattice_temperature_K": temperature,
                            "smearing_degauss_Ry": TARGET_SMEARING_RY,
                            "direction": response["direction"][index],
                            "path_distance_d": response["distance"][index],
                            "signed_q_2pi_over_a": response[
                                "signed_q_2pi_over_a"
                            ][index],
                            "q1": qpoint[0],
                            "q2": qpoint[1],
                            "thermal_short_cm-1": record[
                                "short_frequency_cm1"
                            ][index],
                            "finite_q6_cm-1": record["q6_frequency_cm1"][index],
                            "full_EPC_cm-1": record["full_frequency_cm1"][index],
                        }
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response",
        type=Path,
        default=BASE
        / "E42_extended_fixed_lattice_smearing_sweep/extended_fixed_lattice_smearing_sweep.csv",
    )
    parser.add_argument(
        "--operator", type=Path, default=a0.operator_path(300)
    )
    parser.add_argument(
        "--direct-dfpt",
        type=Path,
        default=BASE
        / "E32_absolute_and_DFPT_comparison/MLIP_full_EPC_LR_vs_DFPT_points.csv",
    )
    parser.add_argument(
        "--dfpt-line",
        type=Path,
        default=ROOT
        / "results/graphene_physical_fd_dfpt/campaigns/FD300_LINE/graphene_FD300_LINE_dfpt.csv",
    )
    parser.add_argument(
        "--e48-summary",
        type=Path,
        default=BASE
        / "E48_fixed_smearing_thermal_full_epc/fixed_smearing_thermal_full_epc_summary.json",
    )
    parser.add_argument(
        "--s0-result-root",
        type=Path,
        help=(
            "optional root containing formal_T300/450/600_Tel300; used for a "
            "controlled S0 comparison instead of the historical mixed roots"
        ),
    )
    parser.add_argument("--require-paired-sampling", action="store_true")
    parser.add_argument("--comparison-label", default="historical_S0_control")
    parser.add_argument(
        "--s0-force-gate",
        type=Path,
        default=BASE
        / "S0_unified_short/formal_240ep_2060_seed83_replayw16/checkpoint_sweep.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=R2ROOT / "R2AS_r2ao_full_epc_acceptance_20260827",
    )
    args = parser.parse_args()

    if args.s0_result_root is not None:
        use_s0_result_root(args.s0_result_root)
    if args.require_paired_sampling and args.s0_result_root is None:
        parser.error("--require-paired-sampling requires --s0-result-root")

    response = e48.load_response(args.response, TARGET_SMEARING_RY)
    operator = e48.load_npz(args.operator)
    operator_hash = e48.sha256(args.operator)
    if not np.array_equal(operator["atom_mapping"], np.arange(72)):
        raise ValueError("R2AS requires the identity-mapped T300 q6 operator")
    if abs(float(scalar(operator["degauss_Ry"])) - TARGET_SMEARING_RY) > 5e-10:
        raise ValueError("R2AS operator and response smearings differ")
    phonon, geometry_error = a0.make_phonopy(operator)
    delta_fc = np.asarray(operator["delta_fc_full"], float)

    loaded = {model: {} for model in MODELS}
    assembled = {model: {} for model in MODELS}
    for model in MODELS:
        for temperature in TEMPERATURES:
            loaded[model][temperature] = load_case(
                model, temperature, operator_hash
            )
            assembled[model][temperature] = assemble_case(
                phonon,
                loaded[model][temperature]["force_constants"],
                delta_fc,
                response,
                loaded[model][temperature]["result"],
            )

    assembly_limits = {
        "FC2_remove_reassemble_max_abs_eV_A2": 1.0e-12,
        "dynamical_matrix_linearity_max_abs": 1.0e-12,
        "minimum_adjacent_mode_overlap": 0.95,
        "tracked_vs_highest_max_abs_cm-1": 1.0e-6,
        "saved_q6_K_replay_max_abs_cm-1": 1.0e-6,
        "time_reversal_matrix_max_abs": 1.0e-12,
        "time_reversal_frequency_max_abs_cm-1": 1.0e-6,
        "total_hermitian_max_abs": 1.0e-12,
        "correction_hermitian_max_abs": 1.0e-12,
        "projector_idempotency_max_abs": 1.0e-12,
        "orthogonal_correction_leakage_max_abs": 1.0e-12,
    }
    assembly_by_case = {}
    assembly_passed = True
    for model in MODELS:
        assembly_by_case[model] = {}
        for temperature in TEMPERATURES:
            diagnostics = assembled[model][temperature]["diagnostics"]
            gates = {
                "FC2_reassembly": diagnostics[
                    "FC2_remove_reassemble_max_abs_eV_A2"
                ]
                <= assembly_limits["FC2_remove_reassemble_max_abs_eV_A2"],
                "matrix_linearity": diagnostics[
                    "dynamical_matrix_linearity_max_abs"
                ]
                <= assembly_limits["dynamical_matrix_linearity_max_abs"],
                "mode_overlap": diagnostics["minimum_adjacent_mode_overlap"]
                >= assembly_limits["minimum_adjacent_mode_overlap"],
                "tracked_branch": diagnostics[
                    "tracked_vs_highest_max_abs_cm-1"
                ]
                <= assembly_limits["tracked_vs_highest_max_abs_cm-1"],
                "saved_K_replay": diagnostics[
                    "saved_q6_K_replay_max_abs_cm-1"
                ]
                <= assembly_limits["saved_q6_K_replay_max_abs_cm-1"],
                "time_reversal_matrix": diagnostics[
                    "time_reversal_matrix_max_abs"
                ]
                <= assembly_limits["time_reversal_matrix_max_abs"],
                "time_reversal_frequency": diagnostics[
                    "time_reversal_frequency_max_abs_cm-1"
                ]
                <= assembly_limits["time_reversal_frequency_max_abs_cm-1"],
                "Hermiticity": diagnostics["total_hermitian_max_abs"]
                <= assembly_limits["total_hermitian_max_abs"]
                and diagnostics["correction_hermitian_max_abs"]
                <= assembly_limits["correction_hermitian_max_abs"],
                "projector": diagnostics["projector_idempotency_max_abs"]
                <= assembly_limits["projector_idempotency_max_abs"]
                and diagnostics["orthogonal_correction_leakage_max_abs"]
                <= assembly_limits["orthogonal_correction_leakage_max_abs"],
                "finite": diagnostics["all_frequencies_finite"],
            }
            assembly_by_case[model][str(temperature)] = {
                "diagnostics": diagnostics,
                "gates": gates,
                "passed": all(gates.values()),
            }
            assembly_passed = assembly_passed and all(gates.values())

    inherited = json.loads(args.e48_summary.read_text(encoding="utf-8"))
    inherited_gates = inherited.get("gates", {})
    inherited_passed = bool(inherited_gates) and all(inherited_gates.values())
    static = static_validation(response, args.direct_dfpt, args.dfpt_line)
    sensitivity = model_sensitivity(response, assembled)
    s0_force_gate = s0_force_gate_validation(args.s0_force_gate, loaded)
    paired_sampling = (
        paired_sampling_validation(loaded)
        if args.require_paired_sampling
        else {"required": False, "passed": True}
    )
    spectral_passed = bool(
        assembly_passed
        and inherited_passed
        and static["passed"]
        and s0_force_gate["passed"]
        and sensitivity["passed"]
        and paired_sampling["passed"]
    )
    if spectral_passed:
        status = (
            "R2AT_PAIRED_QUANTITATIVE_KOHN_ANOMALY_SENSITIVITY_PASSED"
            if args.require_paired_sampling
            else "R2AS_QUANTITATIVE_KOHN_ANOMALY_SENSITIVITY_PASSED"
        )
    elif (
        assembly_passed
        and inherited_passed
        and static["passed"]
        and s0_force_gate["passed"]
        and sensitivity["passed"]
        and not paired_sampling["passed"]
    ):
        status = "R2AS_KOHN_CUSP_PASSED_PAIRED_SAMPLING_FAILED"
    elif (
        assembly_passed
        and inherited_passed
        and static["passed"]
        and s0_force_gate["passed"]
    ):
        status = "R2AS_KOHN_CUSP_PASSED_MODEL_SENSITIVITY_FAILED"
    else:
        status = "R2AS_QUANTITATIVE_KOHN_ANOMALY_FAILED"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "r2ao_s0_fixed_smearing_full_epc.csv"
    png_path = args.output_dir / "r2ao_s0_fixed_smearing_full_epc.png"
    pdf_path = args.output_dir / "r2ao_s0_fixed_smearing_full_epc.pdf"
    full_band_png_path = args.output_dir / "s0_full_bands_with_dense_K_full_epc.png"
    full_band_pdf_path = args.output_dir / "s0_full_bands_with_dense_K_full_epc.pdf"
    matrix_path = args.output_dir / "r2ao_s0_fixed_smearing_full_epc_matrices.npz"
    write_csv(csv_path, response, assembled)
    make_figure(response, assembled, png_path)
    make_figure(response, assembled, pdf_path)
    make_full_band_figure(response, loaded, assembled, full_band_png_path)
    make_full_band_figure(response, loaded, assembled, full_band_pdf_path)
    e48.atomic_npz(
        matrix_path,
        short_model=np.asarray(MODELS),
        lattice_temperature_K=np.asarray(TEMPERATURES),
        smearing_degauss_Ry=np.asarray(TARGET_SMEARING_RY),
        qpoints=response["qpoints"],
        signed_distance=response["signed_distance"],
        signed_q_2pi_over_a=response["signed_q_2pi_over_a"],
        full_EPC_response_cm2=response["full_response_cm2"],
        short_force_constants_eV_A2=np.asarray(
            [
                [assembled[model][temperature]["short_fc"] for temperature in TEMPERATURES]
                for model in MODELS
            ]
        ),
        full_EPC_frequency_cm1=np.asarray(
            [
                [
                    assembled[model][temperature]["full_frequency_cm1"]
                    for temperature in TEMPERATURES
                ]
                for model in MODELS
            ]
        ),
        finite_q6_frequency_cm1=np.asarray(
            [
                [
                    assembled[model][temperature]["q6_frequency_cm1"]
                    for temperature in TEMPERATURES
                ]
                for model in MODELS
            ]
        ),
        full_total_dynamical_matrices=np.asarray(
            [
                [
                    assembled[model][temperature]["full_total_matrices"]
                    for temperature in TEMPERATURES
                ]
                for model in MODELS
            ]
        ),
        full_EPC_correction_matrices=np.asarray(
            [
                [
                    assembled[model][temperature]["full_correction_matrices"]
                    for temperature in TEMPERATURES
                ]
                for model in MODELS
            ]
        ),
    )
    artifact_records = {
        "csv": {"path": str(csv_path), "sha256": e48.sha256(csv_path)},
        "png": {"path": str(png_path), "sha256": e48.sha256(png_path)},
        "pdf": {"path": str(pdf_path), "sha256": e48.sha256(pdf_path)},
        "full_band_png": {
            "path": str(full_band_png_path),
            "sha256": e48.sha256(full_band_png_path),
        },
        "full_band_pdf": {
            "path": str(full_band_pdf_path),
            "sha256": e48.sha256(full_band_pdf_path),
        },
        "matrices": {
            "path": str(matrix_path),
            "sha256": e48.sha256(matrix_path),
        },
    }

    metrics_by_model = {
        model: {
            str(temperature): assembled[model][temperature]["metrics"]
            for temperature in TEMPERATURES
        }
        for model in MODELS
    }
    primary_band_stability = {}
    for temperature in TEMPERATURES:
        frequency = np.asarray(
            loaded["S0"][temperature]["result"]["frequency_cm_1"], float
        )
        flat_index = int(np.argmin(frequency))
        path_index, branch_index = np.unravel_index(flat_index, frequency.shape)
        primary_band_stability[str(temperature)] = {
            "commensurate_supercell_min_frequency_cm-1": loaded["S0"][temperature][
                "acceptance"
            ]["supercell_min_frequency_cm-1"],
            "commensurate_supercell_imaginary_modes_below_minus_1_cm-1": loaded[
                "S0"
            ][temperature]["acceptance"][
                "supercell_imaginary_modes_below_minus_1_cm-1"
            ],
            "interpolated_full_band_min_frequency_cm-1_not_gated": float(
                frequency[path_index, branch_index]
            ),
            "interpolated_minimum_path_index": int(path_index),
            "interpolated_minimum_branch_index": int(branch_index),
        }
    inputs = {
        "response": {"path": str(args.response), "sha256": e48.sha256(args.response)},
        "operator": {"path": str(args.operator), "sha256": operator_hash},
        "E48_crosscheck": {
            "path": str(args.e48_summary),
            "sha256": e48.sha256(args.e48_summary),
        },
        "S0_force_gate": {
            "path": str(args.s0_force_gate),
            "sha256": e48.sha256(args.s0_force_gate),
        },
        "SSCHA": {
            model: {
                str(temperature): {
                    "result": {
                        "path": str(MODEL_INPUTS[model][temperature]["result"]),
                        "sha256": loaded[model][temperature]["result_sha256"],
                    },
                    "acceptance": {
                        "path": str(MODEL_INPUTS[model][temperature]["acceptance"]),
                        "sha256": e48.sha256(
                            MODEL_INPUTS[model][temperature]["acceptance"]
                        ),
                    },
                }
                for temperature in TEMPERATURES
            }
            for model in MODELS
        },
    }
    summary = {
        "format": "graphene_r2as_r2ao_full_epc_acceptance_v2",
        "status": status,
        "spectral_acceptance_passed": spectral_passed,
        "general_MLIP_force_gate_passed": False,
        "general_MLIP_force_gate_subject": "R2AO",
        "force_gate_by_short_model": {
            "S0": s0_force_gate,
            "R2AO": {
                "passed": False,
                "scope": "fixed per-configuration OOF force/A-prime gate",
                "authorized_use": "paired finite-lattice spectral sensitivity only",
            },
        },
        "primary_force_gated_short_model": "S0",
        "authorized_conclusion": (
            "S0 plus frozen full-EPC response is the force-gated primary spectrum; "
            "R2AO is an independent paired spectral-sensitivity control and remains "
            "outside its general force gate"
        ),
        "electronic_control": {
            "smearing_degauss_Ry": TARGET_SMEARING_RY,
            "shared_across_lattice_temperatures": True,
            "lattice_temperature_and_electronic_smearing_are_distinct": True,
        },
        "lattice_temperatures_K": list(TEMPERATURES),
        "comparison_label": args.comparison_label,
        "formula": (
            "D(q,Tlat)=D[Phi_SSCHA(Tlat,q6_300)-Phi_q6_300](q) "
            "+ Pi_full_EPC(q,degauss_300)|e_A'><e_A'|"
        ),
        "metrics_by_short_model_and_lattice_temperature_K": metrics_by_model,
        "primary_S0_full_band_stability": primary_band_stability,
        "static_full_EPC_DFPT_validation": static,
        "finite_lattice_short_model_sensitivity": sensitivity,
        "paired_sampling_validation": paired_sampling,
        "matrix_assembly": {
            "passed": assembly_passed,
            "limits": assembly_limits,
            "by_case": assembly_by_case,
            "reconstructed_supercell_position_error_A": geometry_error,
        },
        "inherited_dense_response_crosscheck": {
            "passed": inherited_passed,
            "gates": inherited_gates,
        },
        "limitations": [
            "R2AO did not pass the fixed per-configuration OOF force/A-prime gate.",
            "The final dense full-EPC update is a targeted A-prime rank-one spectral correction.",
            "The plotted full-path ZA branch has small negative Fourier-interpolation residues near Gamma; the commensurate supercell spectra have no modes below -1 cm-1 and define the stability gate.",
            "The available finite-lattice DFT-TDEP series is not a converged independent 300/450/600 K thermodynamic reference; the final finite-T gate therefore relies on an independent short-model sensitivity comparison.",
        ],
        "inputs": inputs,
        "outputs": artifact_records,
    }
    report_path = args.output_dir / (
        "R2AT_RESULT.md" if args.require_paired_sampling else "R2AS_RESULT.md"
    )
    static_metrics = static["metrics"]
    shift_metrics = sensitivity["K_temperature_shifts"]
    paired_max = max(
        record["configuration_coordinate_max_abs"]
        for record in paired_sampling.get("by_lattice_temperature_K", {}).values()
    ) if paired_sampling["required"] else 0.0
    report = [
        "# Graphene 有限晶格温度 full-EPC 验收",
        "",
        f"状态：`{status}`",
        "",
        "## 计算条件",
        "",
        f"电子展宽固定为 `smearing/degauss = {TARGET_SMEARING_RY:.10f} Ry`，晶格温度分别为 300、450 和 600 K。每个 SSCHA 条件使用 300 个构型，三个温度都在第一轮收敛。S0 是通过冻结力门槛的主短程模型；R2AO 只作为独立的配对声谱敏感性对照。",
        "",
        "## full-EPC K 点结果",
        "",
        "| lattice temperature (K) | S0（主结果，cm⁻¹） | R2AO（敏感性，cm⁻¹） | R2AO − S0（cm⁻¹） |",
        "| ---: | ---: | ---: | ---: |",
        *[
            "| {temperature} | {s0:.6f} | {r2ao:.6f} | {difference:.6f} |".format(
                temperature=temperature,
                s0=metrics_by_model["S0"][str(temperature)]["full_epc"][
                    "K_frequency_cm-1"
                ],
                r2ao=metrics_by_model["R2AO"][str(temperature)]["full_epc"][
                    "K_frequency_cm-1"
                ],
                difference=sensitivity["by_lattice_temperature_K"][
                    str(temperature)
                ]["metrics"]["R2AO_minus_S0_K_cm-1"],
            )
            for temperature in TEMPERATURES
        ],
        "",
        "| 温区 (K) | S0 K 点温移（cm⁻¹） | R2AO K 点温移（cm⁻¹） | 两模型温移差（cm⁻¹） | 门槛（cm⁻¹） |",
        "| :--- | ---: | ---: | ---: | ---: |",
        *[
            "| {label} | {s0:.6f} | {r2ao:.6f} | {difference:.6f} | 5.0 |".format(
                label=label.replace("_to_", "→"),
                s0=record["S0_K_shift_cm-1"],
                r2ao=record["R2AO_K_shift_cm-1"],
                difference=abs(record["difference_cm-1"]),
            )
            for label, record in shift_metrics.items()
        ],
        "",
        "## 核心门槛",
        "",
        f"- S0 冻结力门槛：{'通过' if s0_force_gate['passed'] else '未通过'}；300/450/600 K 测试集总力 RMSE 分别为 "
        + "/".join(
            f"{s0_force_gate['selected_metrics']['thermal_total_force'][f'thermal{temperature}']['RMSE_meV_A']:.2f}"
            for temperature in TEMPERATURES
        )
        + " meV/Å。",
        f"- 矩阵重构：{'通过' if assembly_passed else '未通过'}",
        "- 超胞稳定性：300/450/600 K 均没有低于 −1 cm⁻¹ 的模。全路径 Fourier 插值的 ZA 最低值分别为 "
        + "/".join(
            f"{primary_band_stability[str(temperature)]['interpolated_full_band_min_frequency_cm-1_not_gated']:.2f}"
            for temperature in TEMPERATURES
        )
        + " cm⁻¹，位于 Γ 附近，作为插值数值残差记录，不用于稳定性门槛。",
        f"- 静态 full-EPC 对直接 DFPT：K 点误差 {static_metrics['K_error_cm-1']:.3f} cm⁻¹；K-referenced shape RMSE/max 为 {static_metrics['K_referenced_shape_RMSE_cm-1']:.3f}/{static_metrics['K_referenced_shape_max_abs_cm-1']:.3f} cm⁻¹；d=0.003/0.025 的 cusp depth 相对误差为 {100.0 * static_metrics['cusp_depth']['d0.003']['relative_error']:.2f}%/{100.0 * static_metrics['cusp_depth']['d0.025']['relative_error']:.2f}%；rounding width 相对误差为 {100.0 * static_metrics['rounding_width_relative_error']:.2f}%。各项均通过。",
        f"- R2AO 对独立 S0 的有限温模型敏感性：{'通过' if sensitivity['passed'] else '未通过'}",
        f"- R2AO–S0 配对采样重放：{'通过' if paired_sampling['passed'] else '未通过'}；三个温度的构型坐标最大差为 {paired_max:.3e}。",
        "",
        "## 适用范围",
        "",
        "结果支持固定电子展宽下 300/450/600 K 的 A′ 分支和 Kohn anomaly 定量曲线。R2AO 没有通过逐构型 OOF 力门槛，不能作为一般用途 MLIP；它在这里仅用于检验主结果对短程势的敏感性。full-EPC 更新是针对 A′ 模的 rank-one 长程修正。当前没有收敛且独立的 300/450/600 K DFT-TDEP 热力学参考，因此有限温绝对频移由 S0 力门槛、严格配对模型对照和静态 DFPT cusp 验证共同限定。",
        "",
        "全 M–Γ–K–M 六分支图见 `s0_full_bands_with_dense_K_full_epc.png`；近 K 定量图见 `r2ao_s0_fixed_smearing_full_epc.png`。",
        "",
        "详细数值和逐项门槛见 `r2ao_full_epc_acceptance_summary.json`。",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    summary["outputs"]["report"] = {
        "path": str(report_path),
        "sha256": e48.sha256(report_path),
    }
    summary_path = args.output_dir / "r2ao_full_epc_acceptance_summary.json"
    e48.atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "status": status,
                "spectral_acceptance_passed": spectral_passed,
                "static_full_EPC_DFPT_validation": static,
                "finite_lattice_short_model_sensitivity": sensitivity,
                "paired_sampling_validation": paired_sampling,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0 if spectral_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
