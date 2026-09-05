#!/usr/bin/env python3
"""Evaluate frozen short-range base plus compact conservative bond experts."""
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
from graphene_short_bond_expert import (  # noqa: E402
    activation_envelope,
    expert_prediction,
    load_expert,
)


def labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("paths use LABEL=/path")
    label, path = specification.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("paths use LABEL=/path")
    return label, Path(path)


def indexed_path(specification: str) -> tuple[int, Path]:
    label, path = labelled_path(specification)
    try:
        return int(label), path
    except ValueError as error:
        raise argparse.ArgumentTypeError("LOCO labels must be sscha indices") from error


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


def model_predictions(
    model_path: Path, device: str, datasets: dict[str, list]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    calculator = MACECalculator(
        model_paths=str(model_path), device=device, default_dtype="float32"
    )
    result = {
        label: predictions(structures, calculator)
        for label, structures in datasets.items()
    }
    del calculator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def expert_predictions(
    expert_path: Path, device: torch.device, datasets: dict[str, list]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    model, metadata = load_expert(expert_path, device)
    result = {}
    for label, structures in datasets.items():
        energy = []
        forces = []
        for structure in structures:
            one_energy, one_force = expert_prediction(model, structure, device)
            energy.append(one_energy)
            forces.append(one_force)
        result[label] = (np.asarray(energy), np.asarray(forces))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, metadata


def evaluate_combination(
    datasets: dict[str, list],
    base_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    short_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    correction_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    mode: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
    harmonic_limit: float,
) -> dict:
    combined = {
        label: (
            short_predictions[label][0] + correction_predictions[label][0],
            short_predictions[label][1] + correction_predictions[label][1],
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


def envelope_boundary_check(model_path: Path) -> dict:
    model, _ = load_expert(model_path, torch.device("cpu"))
    specification = model.specification
    points = [
        specification.lower_off_A,
        specification.lower_on_A,
        specification.upper_on_A,
        specification.upper_off_A,
    ]
    rows = []
    for point in points:
        distance = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        value = activation_envelope(distance, specification)
        first = torch.autograd.grad(value, distance, create_graph=True)[0]
        second = torch.autograd.grad(first, distance)[0]
        rows.append(
            {
                "distance_A": point,
                "envelope": float(value.detach()),
                "first_derivative_A-1": float(first.detach()),
                "second_derivative_A-2": float(second.detach()),
            }
        )
    return {"boundaries": rows}


def finite_difference_check(
    model_path: Path,
    structure,
    device: torch.device,
    atom: int = 48,
    component: int = 0,
    step_A: float = 1.0e-6,
) -> dict:
    model, _ = load_expert(model_path, device)
    analytic_energy, analytic_force = expert_prediction(model, structure, device)
    plus = structure.copy()
    minus = structure.copy()
    plus.positions[atom, component] += step_A
    minus.positions[atom, component] -= step_A
    plus_energy, _ = expert_prediction(model, plus, device)
    minus_energy, _ = expert_prediction(model, minus, device)
    numerical = -(plus_energy - minus_energy) / (2.0 * step_A)
    return {
        "atom": atom,
        "component": component,
        "step_A": step_A,
        "expert_energy_eV": analytic_energy,
        "analytic_force_eV_A": float(analytic_force[atom, component]),
        "finite_difference_force_eV_A": float(numerical),
        "absolute_error_eV_A": float(abs(analytic_force[atom, component] - numerical)),
        "maximum_net_force_component_eV_A": float(
            np.max(np.abs(np.sum(analytic_force, axis=0)))
        ),
    }


def loco_support_metrics(
    specifications: list[tuple[int, Path]],
    support: list,
    base_support: tuple[np.ndarray, np.ndarray],
    short_support: tuple[np.ndarray, np.ndarray],
    mode: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
    device: torch.device,
) -> dict | None:
    if not specifications:
        return None
    by_index = dict(specifications)
    support_indices = {int(item.info["sscha_index"]) for item in support}
    if set(by_index) != support_indices or len(by_index) != len(specifications):
        raise ValueError("LOCO experts must cover each support sscha index exactly once")
    expert_energy = []
    expert_force = []
    provenance = {}
    for structure in support:
        index = int(structure.info["sscha_index"])
        path = by_index[index]
        model, metadata = load_expert(path, device)
        energy, force = expert_prediction(model, structure, device)
        expert_energy.append(energy)
        expert_force.append(force)
        provenance[str(index)] = {
            "path": str(path),
            "sha256": sha256(path),
            "metadata": metadata,
        }
    result = support_metrics(
        support,
        base_support[1],
        short_support[0] + np.asarray(expert_energy),
        short_support[1] + np.asarray(expert_force),
        mode,
        reference,
        cell,
    )
    return {"metrics": result, "experts": provenance}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--short-base-model", type=Path, required=True)
    parser.add_argument("--expert", action="append", type=labelled_path, required=True)
    parser.add_argument("--loco-expert", action="append", type=indexed_path, default=[])
    parser.add_argument("--support-data", type=Path, required=True)
    parser.add_argument("--thermal", action="append", type=labelled_path, required=True)
    parser.add_argument("--harmonic", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    experts = dict(args.expert)
    thermal_paths = dict(args.thermal)
    if len(experts) != len(args.expert):
        raise ValueError("expert labels must be unique")
    if set(thermal_paths) != {"T300", "T450", "T600"}:
        raise ValueError("thermal inputs must be T300, T450, and T600")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    datasets = {
        "support": read(args.support_data, index=":"),
        **{label: read(path, index=":") for label, path in thermal_paths.items()},
        "harmonic": read(args.harmonic, index=":"),
    }
    if len(datasets["support"]) != 9:
        raise ValueError("support development set must contain nine structures")

    _, reference, cell, operator_degauss = load_operator(args.operator)
    if abs(operator_degauss - 0.0019000869380739254) > 5.0e-10:
        raise ValueError("R2G uses the frozen T300 q6 operator")
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )

    base = model_predictions(args.base_model, args.device, datasets)
    short = model_predictions(args.short_base_model, args.device, datasets)
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

    zero = {
        label: (np.zeros_like(values[0]), np.zeros_like(values[1]))
        for label, values in short.items()
    }
    results = {
        "short_base_no_expert": evaluate_combination(
            datasets, base, short, zero, mode, reference, cell, harmonic_limit
        )
    }
    expert_metadata = {}
    for label, path in experts.items():
        correction, metadata = expert_predictions(path, device, datasets)
        results[label] = evaluate_combination(
            datasets,
            base,
            short,
            correction,
            mode,
            reference,
            cell,
            harmonic_limit,
        )
        results[label]["expert"] = {
            "path": str(path),
            "sha256": sha256(path),
            "metadata": metadata,
        }
        expert_metadata[label] = metadata

    ranking = sorted(
        results, key=lambda label: results[label]["normalized_worst_gate_score"]
    )
    best_expert_label = next((label for label in ranking if label in experts), None)
    diagnostics = {}
    if best_expert_label is not None:
        best_path = experts[best_expert_label]
        sscha99 = next(
            item
            for item in datasets["support"]
            if int(item.info["sscha_index"]) == 99
        )
        diagnostics = {
            "candidate": best_expert_label,
            "activation_boundary": envelope_boundary_check(best_path),
            "energy_force_finite_difference": finite_difference_check(
                best_path, sscha99, device
            ),
        }

    loco = loco_support_metrics(
        args.loco_expert,
        datasets["support"],
        base["support"],
        short["support"],
        mode,
        reference,
        cell,
        device,
    )
    summary = {
        "status": "R2G_short_bond_expert_development_evaluation_complete",
        "finite_temperature_validation_closed": False,
        "reason_not_final": (
            "the nine corrected-X0 support configurations are development data; "
            "a new order-safe matched holdout is required after all gates pass"
        ),
        "new_DFT_labels": 0,
        "long_range_model_modified": False,
        "fixed_condition": {
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": operator_degauss,
        },
        "frozen_components": {
            "base_model": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
            "short_base_model": {"path": str(args.short_base_model), "sha256": sha256(args.short_base_model)},
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
        },
        "frozen_base_harmonic_force": base_harmonic,
        "harmonic_limit_meV_A": harmonic_limit,
        "Aprime_mode": mode_provenance,
        "ranking": ranking,
        "best_development_candidate": ranking[0],
        "best_passes_all_development_gates": results[ranking[0]]["passes_all_development_gates"],
        "candidates": results,
        "diagnostics": diagnostics,
        "leave_one_supported_configuration_out": loco,
        "inputs": {
            "support_data": {"path": str(args.support_data), "sha256": sha256(args.support_data)},
            "thermal": {
                label: {"path": str(path), "sha256": sha256(path)}
                for label, path in thermal_paths.items()
            },
            "harmonic": {"path": str(args.harmonic), "sha256": sha256(args.harmonic)},
            "background": {"path": str(args.background), "sha256": sha256(args.background)},
            "corrected_result": {"path": str(args.corrected_result), "sha256": sha256(args.corrected_result)},
        },
    }
    atomic_json(args.output, summary)
    concise = {
        label: {
            "passes": results[label]["passes_all_development_gates"],
            "worst_score": results[label]["normalized_worst_gate_score"],
            "support_force": results[label]["support9_development_fit"]["force_error"],
            "support_Aprime": results[label]["support9_development_fit"]["Aprime_projection"]["force_error"],
            "support_energy": results[label]["support9_development_fit"]["delta_energy_error_after_global_offset"],
            "harmonic_force": results[label]["harmonic_replay"]["total_force_error"],
        }
        for label in ranking
    }
    print(json.dumps({"ranking": ranking, "candidates": concise, "diagnostics": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
