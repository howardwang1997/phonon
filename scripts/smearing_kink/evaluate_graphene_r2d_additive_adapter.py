#!/usr/bin/env python3
"""Evaluate frozen depth-3 plus additive conservative local adapters."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    folded_k_aprime_mode,
    force_metrics,
    load_operator,
)
from evaluate_graphene_r2c_candidates import (  # noqa: E402
    ordinary_dataset_metrics,
    predictions,
    support_metrics,
    target_arrays,
)


def labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("model/data paths use LABEL=/path")
    label, path = specification.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("model/data paths use LABEL=/path")
    return label, Path(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def unload(calculator: MACECalculator) -> None:
    del calculator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_predictions(
    model: Path, device: str, datasets: dict[str, list]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    calculator = MACECalculator(
        model_paths=str(model), device=device, default_dtype="float32"
    )
    result = {
        label: predictions(structures, calculator)
        for label, structures in datasets.items()
    }
    unload(calculator)
    return result


def evaluate_combination(
    datasets: dict[str, list],
    base_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    depth_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    adapter_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    mode: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
    harmonic_limit: float,
) -> dict:
    combined = {
        label: (
            depth_predictions[label][0] + adapter_predictions[label][0],
            depth_predictions[label][1] + adapter_predictions[label][1],
        )
        for label in datasets
    }
    support = support_metrics(
        datasets["support"],
        base_predictions["support"][1],
        combined["support"][0],
        combined["support"][1],
        mode,
        reference,
        cell,
    )
    ratios = dict(support["normalized_gate_ratios"])
    thermal = {}
    for label in ("T300", "T450", "T600"):
        metrics = ordinary_dataset_metrics(
            datasets[label],
            base_predictions[label][1],
            combined[label][0],
            combined[label][1],
        )
        thermal[label] = metrics
        ratios[f"{label}/force_RMSE"] = (
            metrics["total_force_error"]["RMSE_meV_A"] / 30.0
        )
        ratios[f"{label}/force_max_abs"] = (
            metrics["total_force_error"]["max_abs_meV_A"] / 200.0
        )
    harmonic = ordinary_dataset_metrics(
        datasets["harmonic"],
        base_predictions["harmonic"][1],
        combined["harmonic"][0],
        combined["harmonic"][1],
    )
    ratios["harmonic/force_RMSE"] = (
        harmonic["total_force_error"]["RMSE_meV_A"] / harmonic_limit
    )
    return {
        "support9_development_fit": support,
        "thermal_development": thermal,
        "harmonic_replay": harmonic,
        "normalized_gate_ratios": ratios,
        "normalized_worst_gate_score": float(max(ratios.values())),
        "passes_all_development_gates": bool(
            all(value <= 1.0 for value in ratios.values())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--depth3-model", type=Path, required=True)
    parser.add_argument("--adapter", action="append", type=labelled_path, required=True)
    parser.add_argument("--support-data", type=Path, required=True)
    parser.add_argument("--thermal", action="append", type=labelled_path, required=True)
    parser.add_argument("--harmonic", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    adapters = dict(args.adapter)
    thermal_paths = dict(args.thermal)
    if len(adapters) != len(args.adapter):
        raise ValueError("adapter labels must be unique")
    if set(thermal_paths) != {"T300", "T450", "T600"}:
        raise ValueError("thermal inputs must be labelled T300, T450, and T600")

    datasets = {
        "support": read(args.support_data, index=":"),
        **{label: read(path, index=":") for label, path in thermal_paths.items()},
        "harmonic": read(args.harmonic, index=":"),
    }
    if len(datasets["support"]) != 9:
        raise ValueError("support development set must contain nine structures")

    _, reference, cell, operator_degauss = load_operator(args.operator)
    if abs(operator_degauss - 0.0019000869380739254) > 5.0e-10:
        raise ValueError("R2D uses the frozen T300 q6 operator")
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )

    base = model_predictions(args.base_model, args.device, datasets)
    depth = model_predictions(args.depth3_model, args.device, datasets)
    base_harmonic_errors = []
    for index, structure in enumerate(datasets["harmonic"]):
        target_total, long_force = target_arrays(
            structure, base["harmonic"][1][index]
        )
        base_harmonic_errors.append(
            base["harmonic"][1][index] + long_force - target_total
        )
    base_harmonic = force_metrics(np.asarray(base_harmonic_errors))
    harmonic_limit = 1.25 * base_harmonic["RMSE_meV_A"]

    zero_adapter = {
        label: (np.zeros_like(value[0]), np.zeros_like(value[1]))
        for label, value in depth.items()
    }
    results = {
        "depth3_no_adapter": evaluate_combination(
            datasets,
            base,
            depth,
            zero_adapter,
            mode,
            reference,
            cell,
            harmonic_limit,
        )
    }
    for label, path in adapters.items():
        adapter = model_predictions(path, args.device, datasets)
        results[label] = evaluate_combination(
            datasets,
            base,
            depth,
            adapter,
            mode,
            reference,
            cell,
            harmonic_limit,
        )
        results[label]["adapter_model"] = {
            "path": str(path),
            "sha256": sha256(path),
        }

    ranking = sorted(
        results, key=lambda label: results[label]["normalized_worst_gate_score"]
    )
    summary = {
        "status": "R2D_additive_local_adapter_development_evaluation_complete",
        "finite_temperature_validation_closed": False,
        "reason_not_final": (
            "the nine corrected-X0-supported DFT points were used for adapter "
            "training; a new order-safe matched holdout remains required"
        ),
        "fixed_condition": {
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": operator_degauss,
        },
        "frozen_components": {
            "base_model": {
                "path": str(args.base_model),
                "sha256": sha256(args.base_model),
            },
            "depth3_model": {
                "path": str(args.depth3_model),
                "sha256": sha256(args.depth3_model),
            },
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
        },
        "frozen_base_harmonic_force": base_harmonic,
        "harmonic_limit_meV_A": harmonic_limit,
        "Aprime_mode": mode_provenance,
        "ranking": ranking,
        "best_development_candidate": ranking[0],
        "best_passes_all_development_gates": results[ranking[0]][
            "passes_all_development_gates"
        ],
        "candidates": results,
        "inputs": {
            "support_data": {
                "path": str(args.support_data),
                "sha256": sha256(args.support_data),
            },
            "thermal": {
                label: {"path": str(path), "sha256": sha256(path)}
                for label, path in thermal_paths.items()
            },
            "harmonic": {"path": str(args.harmonic), "sha256": sha256(args.harmonic)},
            "background": {
                "path": str(args.background),
                "sha256": sha256(args.background),
            },
            "corrected_result": {
                "path": str(args.corrected_result),
                "sha256": sha256(args.corrected_result),
            },
        },
    }
    atomic_json(args.output, summary)
    concise = {
        label: {
            "passes": results[label]["passes_all_development_gates"],
            "worst_score": results[label]["normalized_worst_gate_score"],
            "support_force": results[label]["support9_development_fit"]["force_error"],
            "support_Aprime": results[label]["support9_development_fit"]
            ["Aprime_projection"]["force_error"],
            "support_energy": results[label]["support9_development_fit"]
            ["delta_energy_error_after_global_offset"],
            "harmonic_force": results[label]["harmonic_replay"]["total_force_error"],
        }
        for label in ranking
    }
    print(json.dumps({"ranking": ranking, "candidates": concise}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
