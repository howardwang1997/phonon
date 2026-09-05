#!/usr/bin/env python3
"""Aggregate nine provenance-locked R2M routed-tail outer-LOCO folds."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from graphene_r2m_routed_tail import (
    FORMAT,
    sha256,
    strict_json,
    torch_load,
    validate_invariant_schema,
)


POLICY = "fixed_epoch_240_no_support_checkpoint_selection"
EXPECTED_R2M_MANIFEST_SHA256 = (
    "bb20a86af073e344c38e39ab14d3a80af9e39fa051591bed2b039ae643ffe31e"
)
EXPECTED_R2M_TRAIN_SHA256 = (
    "789b65e1a2c3b260e24d80f008407c7ee55e02676ff3bcf9a875fe9de294519c"
)
EXPECTED_R2M_VALID_SHA256 = (
    "92ad1508e1067265445b79679cfe5142d23b2507dab2aa44397578cee8c7a6da"
)
EXPECTED_SUPPORT9_SHA256 = (
    "505cf1a22ec05f83b1cdc610809e5930652536a9b295bf45c4c1221fa4183a88"
)
FOLD_LOCAL_LIMITS = {
    "support_force_RMSE_meV_A": 30.0,
    "support_force_max_abs_meV_A": 200.0,
    "support_centered_energy_abs_meV_config": 19.4,
    "seed1_force_RMSE_meV_A": 30.0,
    "seed1_force_max_abs_meV_A": 200.0,
    "seed1_Aprime_RMS_meV_A": 15.0,
    "seed1_Aprime_slope_relative_error_abs": 0.05,
    "harmonic_force_RMSE_meV_A": 9.044673057081592,
    "harmonic_force_max_abs_meV_A": 200.0,
    "harmonic_gate_mean": 0.05,
    "each_harmonic_gate_mean": 0.05,
    "finite_difference_max_abs_eV_A": 1.0e-5,
    "Hessian_antisymmetry_max_abs_eV_A2": 1.0e-7,
    "O3_energy_max_abs_eV": 1.0e-6,
    "O3_force_equivariance_max_abs_eV_A": 1.0e-5,
    "size_tail_force_max_abs_eV_A": 1.0e-5,
    "size_tail_node_energy_max_abs_eV": 1.0e-7,
    "size_core_plus_tail_force_max_abs_eV_A": 1.0e-5,
}
SUPPORT_LIMITS = {
    "force_RMSE_meV_A": 30.0,
    "force_max_abs_meV_A": 200.0,
    "Aprime_RMS_meV_A": 15.0,
    "Aprime_slope_relative_error_abs": 0.05,
    "train_gauge_energy_RMSE_meV_config": 19.4,
    "importance_weight_ESS_fraction": 0.30,
}
KB_EV_K = 8.617333262145e-5
EXPECTED_OPTIMIZER = {
    "name": "AdamW_amsgrad",
    "learning_rate": 1.0e-3,
    "constant_schedule": True,
    "weight_decay": 1.0e-6,
    "EMA_decay": 0.99,
    "gradient_clip": 20.0,
}
EXPECTED_LOSS = {
    "group_mass": {
        "support": 0.50,
        "harmonic": 0.25,
        "e50": 0.15,
        "auxiliary": 0.10,
    },
    "force_scale_eV_A": {
        "support": 0.030,
        "harmonic": 0.009044673057081592,
        "e50": 0.030,
        "auxiliary": 0.030,
    },
    "support_centered_energy_weight": 0.25,
    "support_centered_energy_scale_eV": 0.0194,
    "support_centered_energy_scale_source": (
        "pre-existing fixed 19.4 meV support gate from the R2C/R2G "
        "development contract; frozen before all R2M outer folds and "
        "not tuned on any outer-held configuration"
    ),
    "harmonic_gate_off_weight": 0.05,
    "e50_gate_off_weight": 0.02,
    "gate_off_scale": 0.05,
}
EXPECTED_SPECIFICATION = {
    "input_dimension": 64,
    "tail_hidden_1": 64,
    "tail_hidden_2": 32,
    "router_hidden_1": 32,
    "router_hidden_2": 16,
    "gate_score_off": 0.0,
    "gate_score_on": 1.0,
}


def reject_seed2_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in ("seed2", "reserved_e50", "reserved-e50")):
        raise ValueError(f"outer-LOCO aggregate rejects seed2 path: {path}")


def exact_file(path: Path, *, allow_empty: bool = False) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or symlinked artifact: {path}")
    if not allow_empty and path.stat().st_size == 0:
        raise ValueError(f"empty artifact: {path}")


def same_path(left: Path | str, right: Path | str) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def checked_artifact(item: dict, label: str, *, allow_empty: bool = False) -> Path:
    path = Path(str(item.get("path", "")))
    reject_seed2_path(path)
    exact_file(path, allow_empty=allow_empty)
    if sha256(path) != item.get("sha256"):
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def force_metrics(errors: np.ndarray) -> dict:
    flat = np.asarray(errors, float).reshape(-1)
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(flat**2))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(flat))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(flat))),
        "n_force_components": int(flat.size),
    }


def restoring_slope(coordinates: np.ndarray, projected_forces: np.ndarray) -> float:
    denominator = float(np.vdot(coordinates, coordinates).real)
    if denominator <= 1.0e-16:
        raise ValueError("outer-LOCO A-prime coordinates have zero norm")
    return -float(np.vdot(coordinates, projected_forces).real / denominator)


def importance_ess_fraction(errors_eV: np.ndarray, temperature_K: float) -> float:
    values = np.asarray(errors_eV, float)
    centered = values - np.mean(values)
    log_weights = -centered / (KB_EV_K * temperature_K)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return float((np.sum(weights) ** 2 / np.sum(weights**2)) / len(weights))


def prepared_contract(prepared_path: Path) -> tuple[dict, dict[int, dict]]:
    exact_file(prepared_path)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if (
        prepared.get("status")
        != "R2M_routed_tail_nine_outer_folds_frozen_before_training"
        or prepared.get("n_outer_folds") != 9
        or prepared.get("selection_policy") != POLICY
        or prepared.get("E50_seed2_read_or_accepted_as_input") is not False
        or prepared.get("core_gate_status") != "R2M_core_checkpoint_gate_passed"
        or prepared.get("source_r2m_manifest_sha256")
        != EXPECTED_R2M_MANIFEST_SHA256
    ):
        raise ValueError("wrong prepared outer-LOCO manifest")
    validate_invariant_schema(prepared["feature_schema"])
    folds = prepared.get("folds", [])
    if len(folds) != 9 or {int(item["outer_fold"]) for item in folds} != set(range(9)):
        raise ValueError("prepared manifest must map exactly folds 0..8")
    support_order = list(map(int, prepared.get("support_order_sscha_index", [])))
    if len(support_order) != 9 or len(set(support_order)) != 9:
        raise ValueError("prepared support order is not nine unique configurations")
    expected: dict[int, dict] = {}
    core_hashes = set()
    for item in folds:
        fold_index = int(item["outer_fold"])
        fold_dir = Path(item["directory"])
        fold_path = fold_dir / "fold_manifest.json"
        exact_file(fold_path)
        if sha256(fold_path) != item["manifest_sha256"]:
            raise ValueError(f"prepared fold {fold_index} manifest changed")
        fold = json.loads(fold_path.read_text(encoding="utf-8"))
        held = int(item["held_sscha_index"])
        if (
            fold.get("status")
            != "R2M_routed_tail_outer_fold_frozen_before_training"
            or int(fold.get("outer_fold", -1)) != fold_index
            or int(fold.get("held_sscha_index", -1)) != held
            or support_order[fold_index] != held
            or fold.get("selection_policy") != POLICY
            or fold.get("strict_nested_inner_CV_run") is not False
            or fold.get("leakage", {}).get("E50_seed2_read") is not False
            or fold.get("leakage", {}).get(
                "held_geometry_or_label_in_scaler_gradient_or_selection"
            )
            is not False
        ):
            raise ValueError(f"prepared fold {fold_index} identity/policy changed")
        if fold["feature_schema"]["schema_sha256"] != prepared["feature_schema"][
            "schema_sha256"
        ]:
            raise ValueError("fold feature schema differs from prepared manifest")
        validate_invariant_schema(fold["feature_schema"])
        if (
            fold["inputs"]["support9"]["sha256"] != EXPECTED_SUPPORT9_SHA256
            or fold["inputs"]["replay_train"]["sha256"]
            != EXPECTED_R2M_TRAIN_SHA256
            or fold["inputs"]["replay_valid_post_freeze_only"]["sha256"]
            != EXPECTED_R2M_VALID_SHA256
        ):
            raise ValueError("fold is not bound to the frozen support/replay roots")
        core_hashes.add(fold["inputs"]["core"]["sha256"])
        expected[fold_index] = {
            "entry": item,
            "manifest": fold,
            "manifest_path": fold_path,
            "manifest_sha256": sha256(fold_path),
        }
    if len(core_hashes) != 1:
        raise ValueError("prepared folds do not share one frozen core")
    return prepared, expected


def recompute_fold_local_gates(
    summary: dict, raw_observed: dict[str, float] | None = None
) -> bool:
    gates = summary.get("fixed_gates", {})
    if set(gates) != set(FOLD_LOCAL_LIMITS):
        raise ValueError("fold-local fixed gate set changed")
    results = []
    for name, limit in FOLD_LOCAL_LIMITS.items():
        item = gates[name]
        observed = float(item["observed"])
        threshold = float(item["threshold"])
        if not np.isfinite(observed) or threshold != limit:
            raise ValueError(f"fold-local gate {name} threshold/value changed")
        if raw_observed is not None:
            raw_value = float(raw_observed[name])
            tolerance = 1.0e-11 * max(1.0, abs(raw_value))
            if not np.isfinite(raw_value) or abs(observed - raw_value) > tolerance:
                raise ValueError(f"fold-local gate {name} differs from raw diagnostics")
        passed = observed <= limit
        if item.get("pass") is not passed:
            raise ValueError(f"fold-local gate {name} boolean is inconsistent")
        results.append(passed)
    recomputed = bool(all(results))
    if summary.get("passes_fold_local_fixed_gates") is not recomputed:
        raise ValueError("fold-local aggregate boolean is inconsistent")
    return recomputed


def finite_array(data, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(data[name])
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"raw fold array {name} has wrong shape or non-finite values")
    return value


def raw_fold_observed(summary: dict, data) -> tuple[dict[str, float], dict]:
    held_force = finite_array(data, "held_force_error_eV_A", (72, 3)).astype(float)
    held_tail = finite_array(data, "held_tail_force_eV_A", (72, 3)).astype(float)
    held_gate = finite_array(data, "held_gate", (72,)).astype(float)
    seed1_force = finite_array(data, "seed1_force_errors_eV_A", (20, 72, 3)).astype(float)
    seed1_coordinate = finite_array(data, "seed1_Aprime_coordinates", (20,)).astype(complex)
    seed1_predicted = finite_array(
        data, "seed1_Aprime_predicted_forces", (20,)
    ).astype(complex)
    seed1_target = finite_array(data, "seed1_Aprime_target_forces", (20,)).astype(complex)
    harmonic_force = finite_array(
        data, "harmonic_force_errors_eV_A", (25, 128, 3)
    ).astype(float)
    harmonic_gate = finite_array(data, "harmonic_gate_means", (25,)).astype(float)
    for name in (
        "held_energy_error_eV",
        "support_Aprime_coordinate",
        "support_Aprime_predicted_force",
        "support_Aprime_target_force",
    ):
        value = np.asarray(data[name])
        if value.shape != () or not bool(np.isfinite(value.reshape(()))):
            raise ValueError(f"raw fold scalar {name} is non-finite or non-scalar")
    if np.any((held_gate < 0.0) | (held_gate > 1.0)) or np.any(
        (harmonic_gate < 0.0) | (harmonic_gate > 1.0)
    ):
        raise ValueError("raw routed-tail gates fall outside [0,1]")
    seed1_predicted_slope = restoring_slope(seed1_coordinate, seed1_predicted)
    seed1_target_slope = restoring_slope(seed1_coordinate, seed1_target)
    mechanics = summary.get("mechanics", {})
    finite_difference = mechanics.get("finite_difference", {})
    hessian = mechanics.get("Hessian_symmetry", {})
    o3 = mechanics.get("O3", {})
    size = mechanics.get("size_consistency", {})
    fd_records = finite_difference.get("records", [])
    if (
        len(fd_records) != 3
        or float(finite_difference.get("step_A", float("nan"))) != 1.0e-4
        or not np.isfinite(
            float(finite_difference.get("tail_energy_eV", float("nan")))
        )
    ):
        raise ValueError("finite-difference mechanics gate must contain three probes")
    fd_errors = []
    for item in fd_records:
        analytic = float(item["analytic_eV_A"])
        numerical = float(item["numerical_eV_A"])
        reported_error = float(item["absolute_error_eV_A"])
        if not all(np.isfinite(value) for value in (analytic, numerical, reported_error)):
            raise ValueError("finite-difference mechanics record is non-finite")
        recomputed_error = abs(analytic - numerical)
        if abs(recomputed_error - reported_error) > 1.0e-12:
            raise ValueError("finite-difference record has inconsistent error")
        fd_errors.append(recomputed_error)
    fd_max = max(fd_errors)
    if abs(
        fd_max - float(finite_difference["maximum_absolute_error_eV_A"])
    ) > 1.0e-12:
        raise ValueError("finite-difference maximum is inconsistent with records")
    hessian_max = float(hessian["maximum_absolute_antisymmetry_eV_A2"])
    hessian_rms = float(hessian["RMS_antisymmetry_eV_A2"])
    if (
        int(hessian.get("matrix_dimension", -1)) != 216
        or not np.isfinite(hessian_max)
        or not np.isfinite(hessian_rms)
        or hessian_max < 0.0
        or hessian_rms < 0.0
        or hessian_rms > hessian_max + 1.0e-15
    ):
        raise ValueError("Hessian mechanics diagnostics are malformed")
    o3_records = o3.get("records", [])
    if [item.get("transformation") for item in o3_records] != [
        "rotation",
        "reflection",
    ]:
        raise ValueError("O(3) mechanics records changed")
    if not np.allclose(
        [float(item.get("determinant", float("nan"))) for item in o3_records],
        [1.0, -1.0],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("O(3) mechanics determinants changed")
    o3_energy_values = [float(item["energy_absolute_error_eV"]) for item in o3_records]
    o3_force_values = [
        float(item["force_equivariance_max_abs_eV_A"]) for item in o3_records
    ]
    if not all(
        np.isfinite(value) and value >= 0.0
        for value in o3_energy_values + o3_force_values
    ):
        raise ValueError("O(3) mechanics records are malformed")
    o3_energy_max = max(o3_energy_values)
    o3_force_max = max(o3_force_values)
    if (
        abs(o3_energy_max - float(o3["energy_max_abs_eV"])) > 1.0e-12
        or abs(o3_force_max - float(o3["force_equivariance_max_abs_eV_A"]))
        > 1.0e-12
    ):
        raise ValueError("O(3) maxima are inconsistent with records")
    if (
        size.get("test")
        != "same centered local displacement in pristine 6x6 and 8x8 cells"
        or not isinstance(size.get("center_indices"), list)
        or len(size["center_indices"]) != 2
    ):
        raise ValueError("size-consistency mechanics definition changed")
    size_gate_difference = float(size.get("tail_gate_abs_difference", float("nan")))
    if not np.isfinite(size_gate_difference) or size_gate_difference < 0.0:
        raise ValueError("size-consistency gate difference is malformed")
    held_metrics = force_metrics(held_force[None])
    seed1_metrics = force_metrics(seed1_force)
    harmonic_metrics = force_metrics(harmonic_force)
    observed = {
        "support_force_RMSE_meV_A": held_metrics["RMSE_meV_A"],
        "support_force_max_abs_meV_A": held_metrics["max_abs_meV_A"],
        "support_centered_energy_abs_meV_config": 1000.0
        * abs(float(np.asarray(data["held_energy_error_eV"]).reshape(()))),
        "seed1_force_RMSE_meV_A": seed1_metrics["RMSE_meV_A"],
        "seed1_force_max_abs_meV_A": seed1_metrics["max_abs_meV_A"],
        "seed1_Aprime_RMS_meV_A": float(
            1000.0 * np.sqrt(np.mean(np.abs(seed1_predicted - seed1_target) ** 2))
        ),
        "seed1_Aprime_slope_relative_error_abs": abs(
            (seed1_predicted_slope - seed1_target_slope) / seed1_target_slope
        ),
        "harmonic_force_RMSE_meV_A": harmonic_metrics["RMSE_meV_A"],
        "harmonic_force_max_abs_meV_A": harmonic_metrics["max_abs_meV_A"],
        "harmonic_gate_mean": float(np.mean(harmonic_gate)),
        "each_harmonic_gate_mean": float(np.max(harmonic_gate)),
        "finite_difference_max_abs_eV_A": fd_max,
        "Hessian_antisymmetry_max_abs_eV_A2": hessian_max,
        "O3_energy_max_abs_eV": o3_energy_max,
        "O3_force_equivariance_max_abs_eV_A": o3_force_max,
        "size_tail_force_max_abs_eV_A": float(
            size["tail_force_max_abs_difference_eV_A"]
        ),
        "size_tail_node_energy_max_abs_eV": float(
            size["tail_node_energy_abs_difference_eV"]
        ),
        "size_core_plus_tail_force_max_abs_eV_A": float(
            size["core_plus_tail_force_max_abs_difference_eV_A"]
        ),
    }
    if set(observed) != set(FOLD_LOCAL_LIMITS) or not all(
        np.isfinite(value) for value in observed.values()
    ):
        raise ValueError("raw fold diagnostics are incomplete or non-finite")
    return observed, {
        "held_force": held_force,
        "held_tail": held_tail,
        "held_gate": held_gate,
        "seed1_force": seed1_force,
        "harmonic_force": harmonic_force,
        "harmonic_gate": harmonic_gate,
    }


def verify_fold_evaluation(path: Path, expected: dict) -> tuple[dict, dict]:
    exact_file(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    fold = expected["manifest"]
    fold_index = int(fold["outer_fold"])
    held = int(fold["held_sscha_index"])
    if (
        summary.get("status")
        != "R2M_routed_tail_outer_fold_post_freeze_evaluation_complete"
        or int(summary.get("outer_fold", -1)) != fold_index
        or int(summary.get("held_sscha_index", -1)) != held
        or summary.get("selection_policy") != POLICY
        or summary.get("held_opened_only_after_epoch240_EMA_hash_freeze") is not True
        or summary.get("training_provenance_verified_before_held_read") is not True
        or summary.get("E50_seed2_read") is not False
        or summary.get("full_composite_deployment_authorized") is not False
    ):
        raise ValueError(f"unsafe or incomplete fold evaluation: {path}")
    artifacts = summary.get("artifacts", {})
    required = {
        "core",
        "fold_manifest",
        "tail",
        "arrays",
        "training_summary",
        "training_freeze",
        "training_metrics",
        "TRAINING_DONE",
        "launcher_freeze",
    }
    if not required.issubset(artifacts):
        raise ValueError("fold evaluation artifact provenance is incomplete")
    artifact_paths = {
        name: checked_artifact(
            artifacts[name], name, allow_empty=(name == "TRAINING_DONE")
        )
        for name in required
    }
    if (
        not same_path(artifact_paths["fold_manifest"], expected["manifest_path"])
        or artifacts["fold_manifest"]["sha256"] != expected["manifest_sha256"]
        or artifacts["core"]["sha256"] != fold["inputs"]["core"]["sha256"]
        or not same_path(artifact_paths["core"], fold["inputs"]["core"]["path"])
    ):
        raise ValueError("evaluation is bound to another fold/core")
    train_dir = artifact_paths["training_summary"].parent
    canonical = {
        "tail": train_dir / "r2m_routed_tail_epoch240_ema.pt",
        "training_freeze": train_dir / "training_freeze.json",
        "training_metrics": train_dir / "training_metrics.jsonl",
        "TRAINING_DONE": train_dir / "TRAINING_DONE",
    }
    if any(not same_path(artifact_paths[name], target) for name, target in canonical.items()):
        raise ValueError("evaluation training artifacts are not in one canonical directory")
    if artifact_paths["TRAINING_DONE"].stat().st_size != 0:
        raise ValueError("TRAINING_DONE marker is not empty")

    training = json.loads(artifact_paths["training_summary"].read_text(encoding="utf-8"))
    if (
        training.get("status")
        != "R2M_routed_tail_outer_fold_training_complete_pending_held_evaluation"
        or int(training.get("outer_fold", -1)) != fold_index
        or int(training.get("held_sscha_index", -1)) != held
        or training.get("selection_policy") != POLICY
        or training.get("selected_epoch") != 240
        or training.get("selected_state") != "EMA"
        or training.get("outer_held_read_for_training_scaler_or_selection") is not False
        or training.get("E50_seed2_read") is not False
        or training.get("artifact", {}).get("sha256") != artifacts["tail"]["sha256"]
        or training.get("training_freeze_sha256")
        != artifacts["training_freeze"]["sha256"]
        or training.get("training_metrics_sha256")
        != artifacts["training_metrics"]["sha256"]
        or training.get("fold_manifest", {}).get("sha256")
        != expected["manifest_sha256"]
    ):
        raise ValueError("training summary provenance/identity changed")
    freeze = json.loads(artifact_paths["training_freeze"].read_text(encoding="utf-8"))
    leakage = freeze.get("leakage", {})
    model_contract = freeze.get("model", {})
    if (
        freeze.get("status") != "frozen_before_R2M_routed_tail_outer_fold_training"
        or int(freeze.get("outer_fold", -1)) != fold_index
        or int(freeze.get("held_sscha_index", -1)) != held
        or freeze.get("selection_policy") != POLICY
        or freeze.get("epochs") != 240
        or freeze.get("seed") != 83
        or freeze.get("optimizer") != EXPECTED_OPTIMIZER
        or freeze.get("loss") != EXPECTED_LOSS
        or any(
            leakage.get(name) is not False
            for name in (
                "outer_held_file_read",
                "outer_held_geometry_label_scaler_gradient_or_selection",
                "E50_seed2_read",
                "validation_file_read",
                "support_checkpoint_selection",
            )
        )
        or freeze.get("inputs", {}).get("core", {}).get("sha256")
        != fold["inputs"]["core"]["sha256"]
        or freeze.get("inputs", {}).get("fold_manifest", {}).get("sha256")
        != expected["manifest_sha256"]
        or freeze.get("inputs", {}).get("launcher_freeze", {}).get("sha256")
        != artifacts["launcher_freeze"]["sha256"]
        or model_contract.get("formula")
        != "sum_i smootherstep(score_i)*(epsilon_i-c_C)"
        or model_contract.get("full_energy_autograd") is not True
        or model_contract.get("g_times_force_shortcut") is not False
        or model_contract.get("query_feature_score_or_node_energy_detached")
        is not False
        or model_contract.get("specification") != EXPECTED_SPECIFICATION
        or model_contract.get("trainable_parameters") != 8898
        or model_contract.get("core_trainable_parameters") != 0
        or model_contract.get("feature_schema", {}).get("schema_sha256")
        != fold["feature_schema"]["schema_sha256"]
    ):
        raise ValueError("training freeze provenance or fixed contract changed")
    metric_lines = [
        line
        for line in artifact_paths["training_metrics"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(metric_lines) != 240:
        raise ValueError("training metrics do not contain exactly 240 epochs")

    launcher = json.loads(artifact_paths["launcher_freeze"].read_text(encoding="utf-8"))
    runtime = launcher.get("runtime_environment", {})
    required_runtime_keys = {
        "python_version",
        "numpy_version",
        "torch_version",
        "torch_cuda_build",
        "torch_cudnn_version",
        "mace_torch_version",
        "CUDA_VISIBLE_DEVICES",
        "cuda_available",
        "visible_cuda_device_count",
        "visible_device_0_name",
        "visible_device_0_capability",
        "torch_default_dtype",
    }
    if (
        launcher.get("status")
        != "R2M_routed_tail_outer_fold_launcher_frozen_before_training"
        or int(launcher.get("outer_fold", -1)) != fold_index
        or int(launcher.get("held_sscha_index", -1)) != held
        or launcher.get("selection_policy") != POLICY
        or launcher.get("E50_seed2_read_or_accepted_as_input") is not False
        or launcher.get("inputs", {}).get("fold_manifest", {}).get("sha256")
        != expected["manifest_sha256"]
        or launcher.get("inputs", {}).get("core", {}).get("sha256")
        != fold["inputs"]["core"]["sha256"]
        or set(runtime) != required_runtime_keys
        or runtime.get("mace_torch_version") != "0.3.16"
        or runtime.get("CUDA_VISIBLE_DEVICES") != "0"
        or runtime.get("cuda_available") is not True
        or runtime.get("visible_cuda_device_count") != 1
        or not str(runtime.get("visible_device_0_name", ""))
        or not isinstance(runtime.get("visible_device_0_capability"), list)
        or len(runtime["visible_device_0_capability"]) != 2
        or tuple(map(int, runtime["visible_device_0_capability"])) < (7, 0)
        or runtime.get("torch_default_dtype") != "torch.float32"
    ):
        raise ValueError("launcher freeze provenance/identity changed")
    for name, item in launcher.get("inputs", {}).items():
        checked_artifact(item, f"launcher frozen input {name}")
    for name, item in launcher.get("code_snapshots", {}).items():
        checked_artifact(item, f"launcher code snapshot {name}")
    expected_code_names = {
        "run_graphene_r2m_routed_tail_outer_fold.sh",
        "graphene_r2m_run_provenance.py",
        "graphene_r2m_routed_tail.py",
        "train_graphene_r2m_routed_tail_outer_fold.py",
        "evaluate_graphene_r2m_routed_tail_outer_fold.py",
        "graphene_r2m_aprime_eval.py",
    }
    if set(launcher.get("code_snapshots", {})) != expected_code_names:
        raise ValueError("launcher code snapshot set changed")
    if launcher["inputs"]["launcher_source"]["sha256"] != launcher[
        "code_snapshots"
    ]["run_graphene_r2m_routed_tail_outer_fold.sh"]["sha256"]:
        raise ValueError("executed launcher differs from its code snapshot")
    evaluation_snapshots = summary.get("artifacts", {}).get("code_snapshots", {})
    expected_evaluation_names = {
        "graphene_r2m_routed_tail.py",
        "train_graphene_r2m_routed_tail_outer_fold.py",
        "evaluate_graphene_r2m_routed_tail_outer_fold.py",
        "graphene_r2m_aprime_eval.py",
    }
    if set(evaluation_snapshots) != expected_evaluation_names:
        raise ValueError("evaluation code snapshot set changed")
    for name, item in evaluation_snapshots.items():
        checked_artifact(item, f"evaluation code snapshot {name}")
        if item["sha256"] != launcher["code_snapshots"][name]["sha256"]:
            raise ValueError(f"evaluation executed a different {name}")
    training_snapshots = freeze.get("snapshots", {})
    if set(training_snapshots) != {
        "graphene_r2m_routed_tail.py",
        "train_graphene_r2m_routed_tail_outer_fold.py",
    }:
        raise ValueError("training code snapshot set changed")
    if any(
        digest != launcher["code_snapshots"][name]["sha256"]
        for name, digest in training_snapshots.items()
    ):
        raise ValueError("training executed code outside the launcher snapshot")
    for name, item in summary.get("artifacts", {}).get(
        "frozen_post_training_inputs", {}
    ).items():
        checked_artifact(item, f"post-training input {name}")

    payload = torch_load(artifact_paths["tail"], map_location="cpu")
    metadata = payload.get("metadata", {})
    if (
        payload.get("format") != FORMAT
        or payload.get("core_sha256") != fold["inputs"]["core"]["sha256"]
        or int(metadata.get("outer_fold", -1)) != fold_index
        or int(metadata.get("held_sscha_index", -1)) != held
        or int(metadata.get("epoch", -1)) != 240
        or metadata.get("state") != "EMA"
        or metadata.get("selection_policy") != POLICY
        or metadata.get("outer_held_read_before_freeze") is not False
        or metadata.get("E50_seed2_read") is not False
    ):
        raise ValueError("tail artifact metadata is not bound to this fold")
    validate_invariant_schema(payload["feature_schema"])
    if payload["feature_schema"]["schema_sha256"] != fold["feature_schema"][
        "schema_sha256"
    ]:
        raise ValueError("tail artifact feature schema differs from fold")

    arrays_path = artifact_paths["arrays"]
    if not same_path(arrays_path, path.with_suffix(".npz")):
        raise ValueError("fold arrays are not beside the evaluation JSON")
    with np.load(arrays_path, allow_pickle=False) as data:
        raw_observed, _ = raw_fold_observed(summary, data)
        force_error = np.asarray(data["held_force_error_eV_A"], float)
        energy_error = float(np.asarray(data["held_energy_error_eV"]).reshape(()))
        coordinate = complex(np.asarray(data["support_Aprime_coordinate"]).reshape(()))
        predicted_mode = complex(
            np.asarray(data["support_Aprime_predicted_force"]).reshape(())
        )
        target_mode = complex(np.asarray(data["support_Aprime_target_force"]).reshape(()))
    local_pass = recompute_fold_local_gates(summary, raw_observed)
    recomputed = force_metrics(force_error[None])
    reported = summary["held_support"]["force_error"]
    if (
        abs(recomputed["RMSE_meV_A"] - float(reported["RMSE_meV_A"])) > 1.0e-9
        or abs(recomputed["max_abs_meV_A"] - float(reported["max_abs_meV_A"]))
        > 1.0e-9
        or abs(
            1000.0 * energy_error
            - float(summary["held_support"]["centered_energy_error_meV_config"])
        )
        > 1.0e-9
    ):
        raise ValueError("fold JSON/NPZ held metrics differ")
    return summary, {
        "force_error": force_error,
        "energy_error": energy_error,
        "coordinate": coordinate,
        "predicted_mode": predicted_mode,
        "target_mode": target_mode,
        "local_pass": local_pass,
        "tail_sha256": artifacts["tail"]["sha256"],
        "launcher_freeze_sha256": artifacts["launcher_freeze"]["sha256"],
        "code_snapshot_hashes": {
            name: item["sha256"]
            for name, item in sorted(launcher["code_snapshots"].items())
        },
        "runtime_environment": runtime,
        "software_environment": {
            name: runtime[name]
            for name in (
                "python_version",
                "numpy_version",
                "torch_version",
                "torch_cuda_build",
                "torch_cudnn_version",
                "mace_torch_version",
                "CUDA_VISIBLE_DEVICES",
                "torch_default_dtype",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--fold-evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in [args.prepared_manifest, args.output, *args.fold_evaluation]:
        reject_seed2_path(path)
    prepared, expected = prepared_contract(args.prepared_manifest)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite outer-LOCO result: {args.output}")
    if len(args.fold_evaluation) != 9:
        raise ValueError("exactly nine fold evaluations are required")

    records = []
    force_errors = []
    energy_errors = []
    coordinates = []
    predicted_modes = []
    target_modes = []
    cross_fold_code_hashes = None
    cross_fold_software_environment = None
    observed_folds = set()
    for path in args.fold_evaluation:
        exact_file(path)
        preliminary = json.loads(path.read_text(encoding="utf-8"))
        fold_index = int(preliminary.get("outer_fold", -1))
        if fold_index not in expected or fold_index in observed_folds:
            raise ValueError("unknown or duplicate outer fold evaluation")
        observed_folds.add(fold_index)
        summary, values = verify_fold_evaluation(path, expected[fold_index])
        if cross_fold_code_hashes is None:
            cross_fold_code_hashes = values["code_snapshot_hashes"]
        elif values["code_snapshot_hashes"] != cross_fold_code_hashes:
            raise ValueError("outer folds were run with different code snapshots")
        if cross_fold_software_environment is None:
            cross_fold_software_environment = values["software_environment"]
        elif values["software_environment"] != cross_fold_software_environment:
            raise ValueError("outer folds used different software environments")
        force_errors.append(values["force_error"])
        energy_errors.append(values["energy_error"])
        coordinates.append(values["coordinate"])
        predicted_modes.append(values["predicted_mode"])
        target_modes.append(values["target_mode"])
        fold_force = force_metrics(values["force_error"][None])
        records.append(
            {
                "outer_fold": fold_index,
                "held_sscha_index": int(summary["held_sscha_index"]),
                "fold_manifest_sha256": expected[fold_index]["manifest_sha256"],
                "evaluation": {"path": str(path), "sha256": sha256(path)},
                "tail_sha256": values["tail_sha256"],
                "launcher_freeze_sha256": values["launcher_freeze_sha256"],
                "runtime_environment": values["runtime_environment"],
                "passes_fold_local_fixed_gates_recomputed": values["local_pass"],
                "held_force_RMSE_meV_A": fold_force["RMSE_meV_A"],
                "held_force_max_abs_meV_A": fold_force["max_abs_meV_A"],
                "held_energy_error_meV_config": 1000.0 * values["energy_error"],
            }
        )
    if observed_folds != set(range(9)):
        raise ValueError("outer fold evaluations do not cover exactly 0..8")
    records.sort(key=lambda item: item["outer_fold"])
    script_path = Path(__file__).resolve()
    module_path = script_path.with_name("graphene_r2m_routed_tail.py")
    if cross_fold_code_hashes is None or sha256(module_path) != cross_fold_code_hashes[
        module_path.name
    ]:
        raise ValueError("aggregate imported a routed-tail module different from folds")
    aggregate_snapshot_dir = args.output.parent / f"{args.output.stem}_code_snapshots"
    if aggregate_snapshot_dir.exists():
        existing = {item.name for item in aggregate_snapshot_dir.iterdir()}
        if existing != {script_path.name, module_path.name}:
            raise ValueError("aggregate code snapshot directory has unexpected contents")
    else:
        aggregate_snapshot_dir.mkdir(parents=True)
    aggregate_code = {}
    for source in (script_path, module_path):
        target = aggregate_snapshot_dir / source.name
        if target.exists():
            if target.is_symlink() or sha256(target) != sha256(source):
                raise ValueError(f"existing aggregate snapshot changed: {target}")
        else:
            shutil.copy2(source, target)
        aggregate_code[source.name] = {
            "executed_path": str(source),
            "executed_sha256": sha256(source),
            "snapshot_path": str(target),
            "snapshot_sha256": sha256(target),
        }

    force = force_metrics(np.asarray(force_errors))
    energy_errors_array = np.asarray(energy_errors)
    coordinate_array = np.asarray(coordinates)
    predicted_array = np.asarray(predicted_modes)
    target_array = np.asarray(target_modes)
    predicted_slope = restoring_slope(coordinate_array, predicted_array)
    target_slope = restoring_slope(coordinate_array, target_array)
    relative_slope = (predicted_slope - target_slope) / target_slope
    observed = {
        "force_RMSE_meV_A": force["RMSE_meV_A"],
        "force_max_abs_meV_A": force["max_abs_meV_A"],
        "Aprime_RMS_meV_A": float(
            1000.0 * np.sqrt(np.mean(np.abs(predicted_array - target_array) ** 2))
        ),
        "Aprime_slope_relative_error_abs": float(abs(relative_slope)),
        "train_gauge_energy_RMSE_meV_config": float(
            1000.0 * np.sqrt(np.mean(energy_errors_array**2))
        ),
        "importance_weight_ESS_fraction": importance_ess_fraction(
            energy_errors_array, 450.0
        ),
    }
    gates = {}
    for name, threshold in SUPPORT_LIMITS.items():
        if name == "importance_weight_ESS_fraction":
            passed = observed[name] >= threshold
            ratio = threshold / max(observed[name], 1.0e-15)
        else:
            passed = observed[name] <= threshold
            ratio = observed[name] / threshold
        gates[name] = {
            "observed": observed[name],
            "threshold": threshold,
            "pass": bool(passed),
            "normalized_ratio": float(ratio),
        }
    each_support_force_gate = all(
        item["held_force_RMSE_meV_A"] <= SUPPORT_LIMITS["force_RMSE_meV_A"]
        and item["held_force_max_abs_meV_A"] <= SUPPORT_LIMITS["force_max_abs_meV_A"]
        for item in records
    )
    all_fold_local = all(
        item["passes_fold_local_fixed_gates_recomputed"] for item in records
    )
    pass_outer = bool(
        all(item["pass"] for item in gates.values())
        and each_support_force_gate
        and all_fold_local
    )
    result = {
        "status": (
            "R2M_routed_tail_outer_LOCO_passed"
            if pass_outer
            else "R2M_routed_tail_outer_LOCO_failed"
        ),
        "routed_tail_final_training_authorized": pass_outer,
        "full_composite_deployment_authorized": False,
        "finite_temperature_validation_closed": False,
        "selection_policy": POLICY,
        "strict_nested_inner_CV_run": False,
        "support9_status": (
            "opened development data; outer LOCO is a within-stage transfer test, "
            "not a blind or external test"
        ),
        "E50_seed2_read": False,
        "support_outer_LOCO": {
            "force_error": force,
            "train_gauge_energy_errors_meV_config": (
                1000.0 * energy_errors_array
            ).tolist(),
            "Aprime_predicted_restoring_slope_eV_A2": predicted_slope,
            "Aprime_DFT_restoring_slope_eV_A2": target_slope,
            "Aprime_restoring_slope_relative_error": float(relative_slope),
            "observed": observed,
            "fixed_gates": gates,
            "each_held_configuration_force_gate_passes": each_support_force_gate,
            "all_fold_seed1_harmonic_mechanics_gates_pass_recomputed": all_fold_local,
        },
        "records": records,
        "scope": {
            "core": "frozen support-free passing R2M core",
            "foundation_and_q6": "outside routed tail and unchanged",
            "size_consistency": "strict local core+tail gate only",
            "whole_foundation_composite_theoretical_no_wrap_claimed": False,
            "next_if_pass": "train one final all-support model with the same fixed 240-epoch contract, then run independent full-composite gates",
            "next_if_fail": "do not train final model and do not open seed2 for selection",
        },
        "inputs": {
            "prepared_manifest": {
                "path": str(args.prepared_manifest),
                "sha256": sha256(args.prepared_manifest),
            },
            "prepared_feature_schema_sha256": prepared["feature_schema"][
                "schema_sha256"
            ],
        },
        "aggregate_code": aggregate_code,
    }
    strict_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "observed": observed,
                "final_training_authorized": pass_outer,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
