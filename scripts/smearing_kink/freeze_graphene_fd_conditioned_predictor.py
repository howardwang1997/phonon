#!/usr/bin/env python3
"""Freeze the conditional short model and two-endpoint q-space law before 450 K."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ENDPOINTS = (300, 600)
COEFFICIENT_NAMES = ("c0_intercept", "c1_rounded_cusp", "c2_quadratic")
QPOINTS = (
    "G:0.000", "G:0.005", "G:0.010", "G:0.015", "G:0.025",
    "G:0.040", "G:0.060", "G:0.080", "K:0.940", "K:0.960",
    "K:0.975", "K:0.985", "K:0.992", "K:1.000", "K:1.008",
    "K:1.015", "K:1.025", "K:1.040", "K:1.060",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def labelled_path(specification: str) -> tuple[int, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("value must use TEMPERATURE=/path")
    raw_temperature, raw_path = specification.split("=", 1)
    try:
        temperature = int(raw_temperature)
    except ValueError as error:
        raise argparse.ArgumentTypeError("temperature must be an integer") from error
    return temperature, Path(raw_path)


def exact_endpoints(values: list[tuple[int, Path]], label: str) -> dict[int, Path]:
    result = dict(values)
    if set(result) != set(ENDPOINTS) or len(values) != len(ENDPOINTS):
        raise ValueError(f"provide exactly one {label} for 300 and 600 K")
    return result


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def close(left: float, right: float, tolerance: float = 1.0e-8) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-selection", type=Path, required=True)
    parser.add_argument("--p2-selection", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--endpoint-replay", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--delta-model", action="append", type=labelled_path, required=True)
    parser.add_argument("--force-selection", action="append", type=labelled_path, required=True)
    parser.add_argument("--force-metrics", action="append", type=labelled_path, required=True)
    parser.add_argument("--endpoint-acceptance", action="append", type=labelled_path, required=True)
    parser.add_argument("--pooled-tdep", action="append", type=labelled_path, required=True)
    parser.add_argument("--operator", action="append", type=labelled_path, required=True)
    parser.add_argument("--prediction-operator-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--code-path", action="append", type=Path, default=[])
    parser.add_argument("--absence-path", action="append", type=Path, default=[])
    parser.add_argument("--prediction-temperature", type=int, default=450)
    parser.add_argument("--temperature-law-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.temperature_law_output.exists():
        raise RuntimeError("refusing to overwrite an existing frozen predictor")
    if not 300 < args.prediction_temperature < 600:
        parser.error("prediction temperature must lie strictly between 300 and 600 K")
    models = exact_endpoints(args.delta_model, "delta model")
    force_selections = exact_endpoints(args.force_selection, "force selection")
    force_metrics = exact_endpoints(args.force_metrics, "force metrics")
    acceptances = exact_endpoints(args.endpoint_acceptance, "endpoint acceptance")
    pooled_tdeps = exact_endpoints(args.pooled_tdep, "pooled TDEP")
    operators = exact_endpoints(args.operator, "sampling operator")
    fixed_files = [
        args.p1_selection,
        args.p2_selection,
        args.source_inventory,
        args.endpoint_replay,
        args.base_model,
        args.prediction_operator_manifest,
        args.plan,
        *args.code_path,
        *models.values(),
        *force_selections.values(),
        *force_metrics.values(),
        *acceptances.values(),
        *pooled_tdeps.values(),
        *operators.values(),
    ]
    for path in fixed_files:
        require_file(path)
    exposed = [str(path) for path in args.absence_path if path.exists()]
    if exposed:
        raise RuntimeError(f"450 K target artifacts existed before freeze: {exposed}")

    p1 = json.loads(args.p1_selection.read_text())
    p2 = json.loads(args.p2_selection.read_text())
    inventory = json.loads(args.source_inventory.read_text())
    replay = json.loads(args.endpoint_replay.read_text())
    if p1.get("status") != "joint_model_required" or not p1.get("requires_joint_model"):
        raise ValueError("P1 does not require the predeclared fallback path")
    if p2.get("status") != "no_joint_candidate_passed":
        raise ValueError("P2 did not record the required joint-model failure")
    if not replay.get("passes_all_endpoint_identity_checks"):
        raise ValueError("conditioned endpoint implementation replay failed")

    base_sha = sha256(args.base_model)
    model_shas = {temperature: sha256(path) for temperature, path in models.items()}
    expected_inventory = {
        300: ("t300_model", "t300_short_pooled360", "t300_calibrated"),
        600: ("t600_model", "t600_short_pooled360", "t600_calibrated"),
    }
    endpoint_records: dict[str, dict] = {}
    coefficient_values: dict[str, dict[int, list[float]]] = {"G": {}, "K": {}}
    force_thresholds = None
    qspace_thresholds = None
    for temperature in ENDPOINTS:
        model_key, pooled_key, calibrated_key = expected_inventory[temperature]
        artifacts = inventory["artifacts"]
        if artifacts[model_key]["sha256"] != model_shas[temperature]:
            raise ValueError(f"source inventory model mismatch at {temperature} K")
        pooled_sha = sha256(pooled_tdeps[temperature])
        acceptance_sha = sha256(acceptances[temperature])
        if artifacts[pooled_key]["sha256"] != pooled_sha:
            raise ValueError(f"source inventory pooled TDEP mismatch at {temperature} K")
        if artifacts[calibrated_key]["sha256"] != acceptance_sha:
            raise ValueError(f"source inventory acceptance mismatch at {temperature} K")

        selection = json.loads(force_selections[temperature].read_text())
        metrics = json.loads(force_metrics[temperature].read_text())
        acceptance = json.loads(acceptances[temperature].read_text())
        selection_sha = sha256(force_selections[temperature])
        if not selection.get("passes_force_and_replay_gate"):
            raise ValueError(f"force gate failed at {temperature} K")
        if metrics["base_model_sha256"] != base_sha:
            raise ValueError(f"base-model force metrics mismatch at {temperature} K")
        if metrics["delta_model_sha256"] != model_shas[temperature]:
            raise ValueError(f"delta-model force metrics mismatch at {temperature} K")
        thermal_labels = [label for label in metrics["datasets"] if label.startswith("thermal")]
        if len(thermal_labels) != 1:
            raise ValueError(f"unexpected thermal force datasets at {temperature} K")
        thermal_metric = metrics["datasets"][thermal_labels[0]]["reconstructed_total_error"]
        for key in ("RMSE_meV_A", "max_abs_meV_A"):
            if not close(thermal_metric[key], selection["thermal_total_force"][key], 1.0e-5):
                raise ValueError(f"force selection does not replay metrics at {temperature} K")
        harmonic_metric = metrics["datasets"]["harmonic"]["reconstructed_total_error"]
        if not close(
            harmonic_metric["RMSE_meV_A"],
            selection["harmonic_combined_force"]["RMSE_meV_A"],
            1.0e-5,
        ):
            raise ValueError(f"harmonic selection does not replay metrics at {temperature} K")
        if force_thresholds is None:
            force_thresholds = selection["thresholds"]
        elif selection["thresholds"] != force_thresholds:
            raise ValueError("endpoint force thresholds differ")

        expected_degauss = 0.0019000869 * temperature / 300.0
        if acceptance.get("status") != "passed" or not acceptance.get(
            "passes_all_force_seed_calibrated_qspace_gates"
        ):
            raise ValueError(f"calibrated q-space endpoint failed at {temperature} K")
        if acceptance["temperature_K"] != temperature or not close(
            acceptance["degauss_Ry"], expected_degauss, 1.0e-12
        ):
            raise ValueError(f"endpoint metadata mismatch at {temperature} K")
        if acceptance["inputs"]["force_selection"]["sha256"] != selection_sha:
            raise ValueError(f"endpoint force-selection hash mismatch at {temperature} K")
        if acceptance["inputs"]["thermal_short_tdep"]["sha256"] != pooled_sha:
            raise ValueError(f"endpoint pooled-TDEP hash mismatch at {temperature} K")
        if qspace_thresholds is None:
            qspace_thresholds = acceptance["thresholds"]
        elif acceptance["thresholds"] != qspace_thresholds:
            raise ValueError("endpoint q-space thresholds differ")
        for region in ("G", "K"):
            region_payload = acceptance["regions"][region]
            if region_payload["calibration_basis"] != "rounded_cusp_plus_quadratic":
                raise ValueError(f"unexpected q-space basis at {temperature} K {region}")
            values = region_payload["calibration_coefficients_delta_lambda_cm-2"]
            if len(values) != 3 or not np.isfinite(values).all():
                raise ValueError(f"invalid q-space coefficients at {temperature} K {region}")
            coefficient_values[region][temperature] = [float(value) for value in values]

        endpoint_records[str(temperature)] = {
            "model": {"path": str(models[temperature]), "sha256": model_shas[temperature]},
            "force_selection": {
                "path": str(force_selections[temperature]),
                "sha256": selection_sha,
                "scope": (
                    "endpoint development" if temperature == 300 else "frozen 15-structure holdout"
                ),
                "thermal_total_force": selection["thermal_total_force"],
                "harmonic_combined_force": selection["harmonic_combined_force"],
            },
            "force_metrics": {
                "path": str(force_metrics[temperature]),
                "sha256": sha256(force_metrics[temperature]),
            },
            "sampling_operator": {
                "path": str(operators[temperature]),
                "sha256": sha256(operators[temperature]),
            },
            "pooled_short_tdep": {
                "path": str(pooled_tdeps[temperature]),
                "sha256": pooled_sha,
            },
            "calibrated_acceptance": {
                "path": str(acceptances[temperature]),
                "sha256": acceptance_sha,
                "scope": acceptance["scope"],
            },
        }

    replay_models = replay["models"]
    if replay_models["base"]["sha256"] != base_sha:
        raise ValueError("endpoint replay used a different base model")
    if replay_models["delta_300"]["sha256"] != model_shas[300]:
        raise ValueError("endpoint replay used a different 300 K delta model")
    if replay_models["delta_600"]["sha256"] != model_shas[600]:
        raise ValueError("endpoint replay used a different 600 K delta model")

    operator_manifest = json.loads(args.prediction_operator_manifest.read_text())
    if operator_manifest.get("status") != "complete":
        raise ValueError("prediction sampling-operator interpolation is incomplete")
    if not close(
        operator_manifest["temperature_K"], args.prediction_temperature, 1.0e-12
    ):
        raise ValueError("prediction operator has the wrong temperature")
    for temperature in ENDPOINTS:
        if operator_manifest["sources"][str(temperature)]["sha256"] != sha256(
            operators[temperature]
        ):
            raise ValueError(f"prediction operator source mismatch at {temperature} K")
    prediction_operator = Path(operator_manifest["output"]["path"])
    require_file(prediction_operator)
    if sha256(prediction_operator) != operator_manifest["output"]["sha256"]:
        raise ValueError("prediction operator hash does not match its manifest")

    prediction_temperature = args.prediction_temperature
    weight_600 = (prediction_temperature - 300.0) / 300.0
    laws: dict[str, dict[str, dict]] = {"G": {}, "K": {}}
    predictions: dict[str, list[float]] = {"G": [], "K": []}
    for region in ("G", "K"):
        for index, name in enumerate(COEFFICIENT_NAMES):
            value_300 = coefficient_values[region][300][index]
            value_600 = coefficient_values[region][600][index]
            slope = (value_600 - value_300) / 300.0
            prediction = value_300 + (prediction_temperature - 300.0) * slope
            laws[region][name] = {
                "value_at_300K": value_300,
                "value_at_600K": value_600,
                "slope_per_K": slope,
                f"value_at_{prediction_temperature}K": prediction,
            }
            predictions[region].append(prediction)

    stack_definition = {
        "formula": (
            "E_SR(T)=E_v11+(1-w(T))*E_delta300+w(T)*E_delta600; "
            "w(T)=(T-300)/300"
        ),
        "interval_K": [300, 600],
        "base_model_sha256": base_sha,
        "delta_model_300_sha256": model_shas[300],
        "delta_model_600_sha256": model_shas[600],
    }
    law = {
        "status": "frozen_before_450_holdout",
        "scope": (
            "300--600 K physical-FD interpolation fixed only from endpoint "
            "development calibrations; no 450 K target was read"
        ),
        "short_model_stack": stack_definition,
        "short_model_stack_sha256": canonical_sha256(stack_definition),
        "basis": {
            "delta_lambda": (
                "c0(T) + c1(T)*(sqrt(x^2 + (4*d(T))^2) - 4*d(T)) + c2(T)*x^2"
            ),
            "d_of_T_Ry": "0.0019000869*T/300",
            "coefficient_interpolation": (
                "c(T)=((600-T)/300)*c(300)+((T-300)/300)*c(600)"
            ),
        },
        "prediction_temperature_K": prediction_temperature,
        "prediction_degauss_Ry": 0.0019000869 * prediction_temperature / 300.0,
        "interpolation_weight_600": weight_600,
        "coefficient_laws": laws,
        "predicted_coefficients_delta_lambda_cm-2": predictions,
        "endpoints": endpoint_records,
        "fixed_thresholds": {
            "force": force_thresholds,
            "qspace": qspace_thresholds,
        },
        "limitation": (
            "The two endpoints define but do not validate the temperature law; "
            "450 K and an independent DFT-MD trajectory remain unseen holdouts."
        ),
    }
    atomic_json(args.temperature_law_output, law)

    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unavailable"
    manifest = {
        "status": "frozen_before_450_holdout",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "conditional short-model stack and linear q-space law frozen before "
            "any 450 K DFT force, DFT-TDEP, or direct-DFPT target"
        ),
        "direct_450_targets_read": False,
        "absence_paths_checked": [str(path) for path in args.absence_path],
        "p1_selection": {"path": str(args.p1_selection), "sha256": sha256(args.p1_selection)},
        "p2_failed_selection": {"path": str(args.p2_selection), "sha256": sha256(args.p2_selection)},
        "endpoint_identity_replay": {
            "path": str(args.endpoint_replay),
            "sha256": sha256(args.endpoint_replay),
        },
        "predictor": {
            **stack_definition,
            "stack_sha256": canonical_sha256(stack_definition),
            "base_model": {"path": str(args.base_model), "sha256": base_sha},
            "delta_model_300": {
                "path": str(models[300]),
                "sha256": model_shas[300],
            },
            "delta_model_600": {
                "path": str(models[600]),
                "sha256": model_shas[600],
            },
        },
        "temperature_law": {
            "path": str(args.temperature_law_output),
            "sha256": sha256(args.temperature_law_output),
        },
        "prediction_sampling_operator": operator_manifest,
        "endpoint_evidence": endpoint_records,
        "T450_on_policy": {
            "temperature_K": 450,
            "smearing": "fermi-dirac",
            "degauss_Ry": 0.00285013035,
            "supercell": [6, 6, 1],
            "seeds": {
                "0": {"dt_fs": 0.5, "equilibration_steps": 3000, "stride": 80},
                "1": {"dt_fs": 0.25, "equilibration_steps": 6000, "stride": 160},
                "2": {"dt_fs": 0.5, "equilibration_steps": 3000, "stride": 80},
            },
            "snapshots_per_seed": 120,
            "dft_label_indices_per_seed": list(range(3, 118, 6)),
            "primary_force_indices_per_seed": [3, 27, 51, 75, 99],
            "dft_shards": {
                "V100-A": "seed0 all 20 plus seed2 first 10",
                "V100-B": "seed1 all 20 plus seed2 last 10",
            },
            "dft_force_settings": {
                "kgrid": [8, 8, 1],
                "ecutwfc_Ry": 60,
                "ecutrho_Ry": 240,
            },
            "dfpt": {
                "convergence_kgrids": [120, 144],
                "line_kgrid_if_converged": 144,
                "qpoints": list(QPOINTS),
            },
        },
        "fixed_acceptance_thresholds": {
            "force_RMSE_meV_A": 50.0,
            "force_max_abs_meV_A": 250.0,
            "seed_pair_full_band_MAE_cm-1": 5.0,
            "MD_mean_temperature_relative_error": 0.05,
            "Gamma_K_line_MAE_cm-1": 10.0,
            "Gamma_K_top_abs_error_cm-1": 15.0,
            "K_kink_relative_error": 0.20,
            "matrix_projector_replay_max_abs_cm-1": 1.0e-5,
            "bootstrap_replicates": 1000,
            "bootstrap_block_rule": "max(5, ceil(2*tau_int))",
        },
        "release_rules": {
            "DFT_force_labels": (
                "only after all three 450 K MLIP trajectories pass temperature "
                "and short-TDEP seed-spread gates"
            ),
            "independent_DFT_MD": "only after the complete on-policy P4 gate passes",
        },
        "authorized_next_stages": ["P4_on_policy_MLIP", "FD450_CONV"],
        "git_head": git_head,
        "files": {
            str(path): sha256(path)
            for path in [args.plan, args.source_inventory, *args.code_path]
        },
    }
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
