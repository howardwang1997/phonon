#!/usr/bin/env python3
"""Freeze the two-endpoint linear q-space coefficient law before 450 K."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ENDPOINTS = (300, 600)
COEFFICIENT_NAMES = ("c0_intercept", "c1_rounded_cusp", "c2_quadratic")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labelled_path(specification: str) -> tuple[int, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("value must use TEMPERATURE=/path")
    raw_temperature, raw_path = specification.split("=", 1)
    try:
        temperature = int(raw_temperature)
    except ValueError as error:
        raise argparse.ArgumentTypeError("temperature must be an integer") from error
    return temperature, Path(raw_path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", type=labelled_path, required=True)
    parser.add_argument(
        "--tdep-manifest", action="append", type=labelled_path, required=True
    )
    parser.add_argument("--force-gate", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--delta-model", type=Path, required=True)
    parser.add_argument("--prediction-temperature", type=int, default=450)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    endpoints = dict(args.endpoint)
    manifests = dict(args.tdep_manifest)
    if set(endpoints) != set(ENDPOINTS) or len(args.endpoint) != len(ENDPOINTS):
        parser.error("provide exactly one endpoint acceptance for 300 and 600 K")
    if set(manifests) != set(ENDPOINTS) or len(args.tdep_manifest) != len(ENDPOINTS):
        parser.error("provide exactly one TDEP manifest for 300 and 600 K")
    if not 300 <= args.prediction_temperature <= 600:
        parser.error("prediction temperature must lie within 300--600 K")

    force = json.loads(args.force_gate.read_text())
    delta_sha = sha256(args.delta_model)
    if not force["passes_force_and_replay_gate"]:
        raise ValueError("joint endpoint force gate is not passed")
    if force["frozen_model_sha256"] != delta_sha:
        raise ValueError("force gate and frozen delta model differ")
    force_sha = sha256(args.force_gate)

    endpoint_payloads: dict[int, dict] = {}
    manifest_payloads: dict[int, dict] = {}
    coefficients: dict[str, dict[int, list[float]]] = {"G": {}, "K": {}}
    provenance: dict[str, dict] = {}
    for temperature in ENDPOINTS:
        acceptance_path = endpoints[temperature]
        manifest_path = manifests[temperature]
        acceptance = json.loads(acceptance_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        expected_degauss = 0.0019000869 * temperature / 300.0
        if acceptance["temperature_K"] != temperature:
            raise ValueError(f"endpoint acceptance temperature mismatch at {temperature} K")
        if not np.isclose(
            acceptance["degauss_Ry"], expected_degauss, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"endpoint degauss is off the physical-FD path at {temperature} K")
        if not acceptance["passes_all_force_seed_calibrated_qspace_gates"]:
            raise ValueError(f"calibrated q-space endpoint failed at {temperature} K")
        if manifest["temperature_K"] != temperature:
            raise ValueError(f"TDEP manifest temperature mismatch at {temperature} K")
        if not manifest["passes_tdep_sampling_gate"]:
            raise ValueError(f"TDEP sampling gate failed at {temperature} K")
        if manifest["delta_model"]["sha256"] != delta_sha:
            raise ValueError(f"TDEP endpoint uses a different delta model at {temperature} K")
        if (
            acceptance["inputs"]["thermal_short_tdep"]["sha256"]
            != manifest["pooled_tdep"]["sha256"]
        ):
            raise ValueError(f"q-space endpoint uses a different pooled TDEP at {temperature} K")
        if acceptance["inputs"]["force_selection"]["sha256"] != force_sha:
            raise ValueError(f"q-space endpoint uses a different force gate at {temperature} K")
        for region in ("G", "K"):
            region_payload = acceptance["regions"][region]
            if region_payload["calibration_basis"] != "rounded_cusp_plus_quadratic":
                raise ValueError(f"unexpected q-space basis for {region} at {temperature} K")
            values = region_payload[
                "calibration_coefficients_delta_lambda_cm-2"
            ]
            if len(values) != 3 or not np.isfinite(values).all():
                raise ValueError(f"invalid q-space coefficients for {region} at {temperature} K")
            coefficients[region][temperature] = [float(value) for value in values]
        endpoint_payloads[temperature] = acceptance
        manifest_payloads[temperature] = manifest
        provenance[str(temperature)] = {
            "acceptance": {
                "path": str(acceptance_path),
                "sha256": sha256(acceptance_path),
            },
            "tdep_manifest": {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
            },
            "pooled_tdep": manifest["pooled_tdep"],
            "static_short_fc2": manifest["static_short_fc2"],
            "static_transfer_acceptance": acceptance["inputs"][
                "static_transfer_control"
            ],
            "dft_tdep_calibration_target": acceptance["inputs"][
                "dft_tdep_calibration_target"
            ],
        }

    prediction_temperature = args.prediction_temperature
    weight_600 = (prediction_temperature - 300.0) / 300.0
    laws: dict[str, dict[str, dict]] = {"G": {}, "K": {}}
    predicted: dict[str, list[float]] = {"G": [], "K": []}
    for region in ("G", "K"):
        for index, name in enumerate(COEFFICIENT_NAMES):
            value_300 = coefficients[region][300][index]
            value_600 = coefficients[region][600][index]
            slope = (value_600 - value_300) / 300.0
            prediction = value_300 + (prediction_temperature - 300.0) * slope
            laws[region][name] = {
                "units": "cm^-2" if index == 0 else (
                    "cm^-2 per reduced-q" if index == 1 else "cm^-2 per reduced-q^2"
                ),
                "value_at_300K": value_300,
                "value_at_600K": value_600,
                "slope_per_K": slope,
                f"value_at_{prediction_temperature}K": prediction,
            }
            predicted[region].append(prediction)

    result = {
        "status": "ready_for_freeze",
        "scope": (
            "300--600 K physical-FD interpolation law fixed from the two endpoint "
            "development calibrations; no intermediate-temperature labels were read"
        ),
        "basis": {
            "delta_lambda": (
                "c0(T) + c1(T) * (sqrt(x^2 + (4*d(T))^2) - 4*d(T)) "
                "+ c2(T) * x^2"
            ),
            "d_of_T_Ry": "0.0019000869 * T / 300",
            "coefficient_interpolation": (
                "c(T) = ((600-T)/300)*c(300) + ((T-300)/300)*c(600)"
            ),
            "regions": ["G", "K"],
        },
        "endpoint_temperatures_K": list(ENDPOINTS),
        "prediction_temperature_K": prediction_temperature,
        "prediction_degauss_Ry": 0.0019000869 * prediction_temperature / 300.0,
        "interpolation_weight_600": weight_600,
        "coefficient_laws": laws,
        "predicted_coefficients_delta_lambda_cm-2": predicted,
        "base_model": {
            "path": str(args.base_model),
            "sha256": sha256(args.base_model),
        },
        "delta_model": {"path": str(args.delta_model), "sha256": delta_sha},
        "force_gate": {"path": str(args.force_gate), "sha256": force_sha},
        "endpoints": provenance,
        "fixed_thresholds": endpoint_payloads[300]["thresholds"],
        "limitation": (
            "Two endpoint calibrations define a linear interpolation; they do not "
            "test the functional form. The prediction remains provisional until an "
            "unseen intermediate lattice temperature and an independent DFT-MD "
            "trajectory pass their fixed gates."
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
