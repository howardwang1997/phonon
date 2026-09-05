#!/usr/bin/env python3
"""Fit an environment-conditioned conservative graphene bond-order residual.

The energy is linear in smooth radial basis functions multiplied by symmetric
first-shell environment invariants.  Forces are exact negative derivatives of
that scalar energy, computed with torch autograd.  A small fixed ridge grid is
ranked with the frozen support, thermal, and harmonic development gates.
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

import numpy as np
import torch
from ase.io import read


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    complex_metrics,
    folded_k_aprime_mode,
    force_metrics,
    load_operator,
    structure_mapping,
)
from evaluate_graphene_r2c_candidates import (  # noqa: E402
    SUPPORT_LIMITS,
    energy_metrics,
    importance_ess_fraction,
    restoring_slope,
)


RIDGE_GRID = [1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0]
HARMONIC_LIMIT_MEV_A = 9.044634650009101
THERMAL_LABELS = {
    "physical_s0_thermal_300K_test": "T300",
    "physical_s0_thermal_450K_test": "T450",
    "physical_s0_thermal_600K_test": "T600",
}


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


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class Topology:
    left: np.ndarray
    right: np.ndarray
    shift_cart_A: np.ndarray
    other_left: np.ndarray
    other_left_sign: np.ndarray
    other_right: np.ndarray
    other_right_sign: np.ndarray


@dataclass(frozen=True)
class BasisSpecification:
    r_min_A: float = 1.15
    r_max_A: float = 1.75
    n_radial: int = 10
    radial_width_A: float = 0.075
    neighbour_cutoff_A: float = 1.75

    @property
    def centers_A(self) -> np.ndarray:
        return np.linspace(1.18, 1.72, self.n_radial)

    @property
    def n_environment(self) -> int:
        return 7

    @property
    def n_basis(self) -> int:
        return self.n_radial * self.n_environment


def build_topology(structure, specification: BasisSpecification) -> Topology:
    positions = np.asarray(structure.positions, float)
    cell = np.asarray(structure.cell, float)
    delta = positions[None, :, :] - positions[:, None, :]
    fractional = delta @ np.linalg.inv(cell)
    image_shift = -np.round(fractional)
    vectors = delta + image_shift @ cell
    distances = np.linalg.norm(vectors, axis=2)
    left, right = np.where(
        np.triu(
            (distances > 1.0e-8)
            & (distances < specification.neighbour_cutoff_A),
            k=1,
        )
    )
    expected = 3 * len(structure) // 2
    if len(left) != expected:
        raise ValueError(
            f"expected {expected} first-neighbour bonds, found {len(left)}"
        )
    shift_cart = image_shift[left, right] @ cell

    incident: list[list[tuple[int, float]]] = [[] for _ in structure]
    for bond, (atom_i, atom_j) in enumerate(zip(left, right, strict=True)):
        incident[int(atom_i)].append((bond, 1.0))
        incident[int(atom_j)].append((bond, -1.0))
    if any(len(items) != 3 for items in incident):
        raise ValueError("first-shell coordination is not exactly three")

    other_left = []
    other_left_sign = []
    other_right = []
    other_right_sign = []
    for bond, (atom_i, atom_j) in enumerate(zip(left, right, strict=True)):
        left_items = [(index, sign) for index, sign in incident[int(atom_i)] if index != bond]
        right_items = [(index, sign) for index, sign in incident[int(atom_j)] if index != bond]
        other_left.append([item[0] for item in left_items])
        other_left_sign.append([item[1] for item in left_items])
        other_right.append([item[0] for item in right_items])
        other_right_sign.append([item[1] for item in right_items])
    return Topology(
        left=np.asarray(left, int),
        right=np.asarray(right, int),
        shift_cart_A=np.asarray(shift_cart, float),
        other_left=np.asarray(other_left, int),
        other_left_sign=np.asarray(other_left_sign, float),
        other_right=np.asarray(other_right, int),
        other_right_sign=np.asarray(other_right_sign, float),
    )


def torch_topology(topology: Topology, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "left": torch.as_tensor(topology.left, dtype=torch.long, device=device),
        "right": torch.as_tensor(topology.right, dtype=torch.long, device=device),
        "shift": torch.as_tensor(topology.shift_cart_A, dtype=torch.float64, device=device),
        "other_left": torch.as_tensor(topology.other_left, dtype=torch.long, device=device),
        "other_left_sign": torch.as_tensor(
            topology.other_left_sign, dtype=torch.float64, device=device
        ),
        "other_right": torch.as_tensor(topology.other_right, dtype=torch.long, device=device),
        "other_right_sign": torch.as_tensor(
            topology.other_right_sign, dtype=torch.float64, device=device
        ),
    }


def raw_environment(
    positions: torch.Tensor, topology: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = (
        positions[topology["right"]]
        - positions[topology["left"]]
        + topology["shift"]
    )
    distance = torch.linalg.vector_norm(vectors, dim=1)
    left_other_distance = distance[topology["other_left"]]
    right_other_distance = distance[topology["other_right"]]
    left_mean = torch.mean(left_other_distance, dim=1)
    right_mean = torch.mean(right_other_distance, dim=1)

    left_other_vectors = (
        vectors[topology["other_left"]]
        * topology["other_left_sign"][..., None]
    )
    right_other_vectors = (
        vectors[topology["other_right"]]
        * topology["other_right_sign"][..., None]
    )
    left_cosine = torch.sum(vectors[:, None, :] * left_other_vectors, dim=2) / (
        distance[:, None] * left_other_distance
    )
    right_cosine = torch.sum((-vectors)[:, None, :] * right_other_vectors, dim=2) / (
        distance[:, None] * right_other_distance
    )
    left_angle_distortion = torch.mean((left_cosine + 0.5) ** 2, dim=1)
    right_angle_distortion = torch.mean((right_cosine + 0.5) ** 2, dim=1)

    raw = torch.stack(
        [
            0.5 * (left_mean + right_mean),
            0.5 * ((left_mean - distance) + (right_mean - distance)),
            (left_mean - right_mean) ** 2,
            0.25
            * (
                (left_other_distance[:, 0] - left_other_distance[:, 1]) ** 2
                + (right_other_distance[:, 0] - right_other_distance[:, 1]) ** 2
            ),
            0.5 * (left_angle_distortion + right_angle_distortion),
            (left_angle_distortion - right_angle_distortion) ** 2,
        ],
        dim=1,
    )
    return distance, raw


def environment_statistics(
    structures: list,
    specification: BasisSpecification,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    with torch.no_grad():
        for structure in structures:
            topology = torch_topology(build_topology(structure, specification), device)
            positions = torch.as_tensor(
                np.asarray(structure.positions, float),
                dtype=torch.float64,
                device=device,
            )
            _, raw = raw_environment(positions, topology)
            values.append(raw.cpu().numpy())
    joined = np.concatenate(values, axis=0)
    mean = np.mean(joined, axis=0)
    scale = np.std(joined, axis=0)
    scale = np.maximum(scale, np.maximum(np.abs(mean) * 1.0e-5, 1.0e-10))
    return mean, scale


def energy_basis(
    positions: torch.Tensor,
    topology: dict[str, torch.Tensor],
    specification: BasisSpecification,
    environment_mean: torch.Tensor,
    environment_scale: torch.Tensor,
) -> torch.Tensor:
    distance, raw = raw_environment(positions, topology)
    standardized = (raw - environment_mean[None, :]) / environment_scale[None, :]
    environment = torch.cat(
        [
            torch.ones((len(distance), 1), dtype=torch.float64, device=positions.device),
            standardized,
        ],
        dim=1,
    )
    span = specification.r_max_A - specification.r_min_A
    reduced = (distance - specification.r_min_A) / span
    envelope = 16.0 * reduced**2 * (1.0 - reduced) ** 2
    inside = (distance > specification.r_min_A) & (distance < specification.r_max_A)
    envelope = torch.where(inside, envelope, torch.zeros_like(envelope))
    centers = torch.as_tensor(
        specification.centers_A, dtype=torch.float64, device=positions.device
    )
    radial = envelope[:, None] * torch.exp(
        -0.5
        * ((distance[:, None] - centers[None, :]) / specification.radial_width_A)
        ** 2
    )
    return torch.sum(radial[:, :, None] * environment[:, None, :], dim=0).reshape(-1)


def configuration_design(
    structure,
    specification: BasisSpecification,
    environment_mean: np.ndarray,
    environment_scale: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    topology = torch_topology(build_topology(structure, specification), device)
    mean = torch.as_tensor(environment_mean, dtype=torch.float64, device=device)
    scale = torch.as_tensor(environment_scale, dtype=torch.float64, device=device)
    positions = torch.tensor(
        np.asarray(structure.positions, float),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        return energy_basis(value, topology, specification, mean, scale)

    energy = evaluate(positions)
    jacobian = torch.autograd.functional.jacobian(
        evaluate, positions, vectorize=True, create_graph=False
    )
    force = -jacobian.permute(1, 2, 0)
    return energy.detach().cpu().numpy(), force.detach().cpu().numpy()


def residual_force(structure) -> np.ndarray:
    return np.asarray(structure.arrays["REF_forces"], float)


def residual_energy(structure) -> float:
    return float(structure.info["REF_energy"])


def config_weight(structure, weights: dict[str, float]) -> float:
    config_type = str(structure.info.get("config_type", "Default"))
    return float(weights.get(config_type, weights.get("Default", 1.0)))


def fit_coefficients(
    structures: list,
    designs: list[tuple[np.ndarray, np.ndarray]],
    weights: dict[str, float],
    ridge: float,
    energy_balance: float,
    excluded_sscha_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    for index, structure in enumerate(structures):
        is_excluded = (
            excluded_sscha_index is not None
            and str(structure.info.get("config_type")) == "r2c_fixed_smearing_supported"
            and int(structure.info.get("sscha_index", -1)) == excluded_sscha_index
        )
        if not is_excluded:
            selected.append(index)

    rows = []
    targets = []
    for index in selected:
        scale = np.sqrt(config_weight(structures[index], weights))
        rows.append(designs[index][1].reshape(-1, designs[index][1].shape[-1]) * scale)
        targets.append(residual_force(structures[index]).reshape(-1) * scale)
    force_matrix = np.concatenate(rows, axis=0)
    force_target = np.concatenate(targets)
    column_scale = np.sqrt(np.mean(force_matrix**2, axis=0))
    column_scale = np.maximum(column_scale, np.max(column_scale) * 1.0e-10)

    matrix_rows = [force_matrix]
    target_rows = [force_target]
    support_indices = [
        index
        for index in selected
        if str(structures[index].info.get("config_type"))
        == "r2c_fixed_smearing_supported"
    ]
    if energy_balance > 0.0 and len(support_indices) >= 2:
        energy_matrix = np.asarray([designs[index][0] for index in support_indices])
        energy_target = np.asarray(
            [residual_energy(structures[index]) for index in support_indices]
        )
        energy_matrix -= np.mean(energy_matrix, axis=0, keepdims=True)
        energy_target -= np.mean(energy_target)
        energy_scale = np.sqrt(energy_balance * weights["r2c_fixed_smearing_supported"])
        matrix_rows.append(energy_matrix * energy_scale)
        target_rows.append(energy_target * energy_scale)

    matrix = np.concatenate(matrix_rows, axis=0)
    target = np.concatenate(target_rows)
    normalized = matrix / column_scale[None, :]
    gram = normalized.T @ normalized / len(target)
    right = normalized.T @ target / len(target)
    normalized_coefficients = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]), right
    )
    coefficients = normalized_coefficients / column_scale
    return coefficients, column_scale


def predictions(
    designs: list[tuple[np.ndarray, np.ndarray]], coefficients: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    energy = np.asarray([design[0] @ coefficients for design in designs])
    force = [
        np.tensordot(design[1], coefficients, axes=(2, 0)) for design in designs
    ]
    return energy, force


def support_metrics(
    structures: list,
    predicted_energy: np.ndarray,
    predicted_force: list[np.ndarray],
    mode: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
) -> dict:
    errors = []
    coordinates = []
    target_modes = []
    predicted_modes = []
    target_energies = []
    records = []
    for structure, energy, force in zip(
        structures, predicted_energy, predicted_force, strict=True
    ):
        error = np.asarray(force, float) - residual_force(structure)
        mapping, displacement = structure_mapping(structure, reference, cell)
        mode_flat = mode[mapping].reshape(-1)
        coordinate = np.vdot(mode_flat, displacement.reshape(-1))
        target_total = np.asarray(structure.arrays["TOTAL_forces"], float)
        target_mode = np.vdot(mode_flat, target_total.reshape(-1))
        error_mode = np.vdot(mode_flat, error.reshape(-1))
        errors.append(error)
        coordinates.append(coordinate)
        target_modes.append(target_mode)
        predicted_modes.append(target_mode + error_mode)
        target_energies.append(residual_energy(structure))
        records.append(
            {
                "sscha_index": int(structure.info["sscha_index"]),
                "force_RMSE_meV_A": force_metrics(error[None])["RMSE_meV_A"],
                "force_max_abs_meV_A": force_metrics(error[None])["max_abs_meV_A"],
                "predicted_adapter_energy_eV": float(energy),
                "Aprime_force_error_abs_meV_A": float(abs(error_mode) * 1000.0),
            }
        )
    errors_array = np.asarray(errors)
    coordinates_array = np.asarray(coordinates)
    target_modes_array = np.asarray(target_modes)
    predicted_modes_array = np.asarray(predicted_modes)
    centered_energy, energy_result = energy_metrics(
        predicted_energy, np.asarray(target_energies)
    )
    target_slope = restoring_slope(coordinates_array, target_modes_array)
    predicted_slope = restoring_slope(coordinates_array, predicted_modes_array)
    relative_slope_error = (predicted_slope - target_slope) / target_slope
    observed = {
        "force_component_RMSE_meV_A": force_metrics(errors_array)["RMSE_meV_A"],
        "force_component_max_abs_meV_A": force_metrics(errors_array)["max_abs_meV_A"],
        "Aprime_projected_force_RMS_meV_A": complex_metrics(
            predicted_modes_array - target_modes_array
        )["RMS_meV_A"],
        "Aprime_restoring_slope_relative_error": abs(float(relative_slope_error)),
        "centered_energy_RMSE_meV_config": energy_result["RMSE_meV_config"],
        "importance_weight_ESS_fraction": importance_ess_fraction(
            centered_energy, 450.0
        ),
    }
    ratios = {}
    gates = {}
    for name, limit in SUPPORT_LIMITS.items():
        if name == "importance_weight_ESS_fraction":
            ratio = limit / max(observed[name], 1.0e-12)
            passed = observed[name] >= limit
        else:
            ratio = observed[name] / limit
            passed = observed[name] <= limit
        ratios[f"support/{name}"] = float(ratio)
        gates[name] = {
            "observed": observed[name],
            "threshold": limit,
            "pass": bool(passed),
        }
    return {
        "force_error": force_metrics(errors_array),
        "Aprime_projection": {
            "force_error": complex_metrics(predicted_modes_array - target_modes_array),
            "candidate_restoring_slope_eV_A2": predicted_slope,
            "DFT_restoring_slope_eV_A2": target_slope,
            "relative_slope_error": float(relative_slope_error),
        },
        "delta_energy_error_after_global_offset": energy_result,
        "importance_weight_ESS_fraction": importance_ess_fraction(
            centered_energy, 450.0
        ),
        "fixed_gate": gates,
        "normalized_gate_ratios": ratios,
        "records": records,
    }


def force_only_metrics(
    structures: list, predicted_force: list[np.ndarray]
) -> dict:
    errors = [
        np.asarray(force, float) - residual_force(structure)
        for structure, force in zip(structures, predicted_force, strict=True)
    ]
    return force_metrics(np.concatenate([error.reshape(-1) for error in errors])[None])


def candidate_metrics(
    support: list,
    support_designs: list,
    thermal: dict[str, list],
    thermal_designs: dict[str, list],
    harmonic: list,
    harmonic_designs: list,
    coefficients: np.ndarray,
    mode: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
) -> dict:
    support_energy, support_force = predictions(support_designs, coefficients)
    support_result = support_metrics(
        support, support_energy, support_force, mode, reference, cell
    )
    ratios = dict(support_result["normalized_gate_ratios"])
    thermal_result = {}
    for label, structures in thermal.items():
        _, force = predictions(thermal_designs[label], coefficients)
        metrics = force_only_metrics(structures, force)
        thermal_result[label] = {"total_force_error": metrics}
        ratios[f"{label}/force_RMSE"] = metrics["RMSE_meV_A"] / 30.0
        ratios[f"{label}/force_max_abs"] = metrics["max_abs_meV_A"] / 200.0
    _, harmonic_force = predictions(harmonic_designs, coefficients)
    harmonic_metrics = force_only_metrics(harmonic, harmonic_force)
    ratios["harmonic/force_RMSE"] = (
        harmonic_metrics["RMSE_meV_A"] / HARMONIC_LIMIT_MEV_A
    )
    return {
        "support9_development_fit": support_result,
        "thermal_development": thermal_result,
        "harmonic_replay": {"total_force_error": harmonic_metrics},
        "normalized_gate_ratios": ratios,
        "normalized_worst_gate_score": float(max(ratios.values())),
        "passes_all_development_gates": bool(
            all(value <= 1.0 for value in ratios.values())
        ),
    }


def finite_difference_check(
    structure,
    coefficients: np.ndarray,
    specification: BasisSpecification,
    environment_mean: np.ndarray,
    environment_scale: np.ndarray,
    device: torch.device,
    atom: int = 48,
    component: int = 0,
    step_A: float = 1.0e-6,
) -> dict:
    topology_np = build_topology(structure, specification)
    topology = torch_topology(topology_np, device)
    mean = torch.as_tensor(environment_mean, dtype=torch.float64, device=device)
    scale = torch.as_tensor(environment_scale, dtype=torch.float64, device=device)
    coefficient_tensor = torch.as_tensor(coefficients, dtype=torch.float64, device=device)

    def energy(position_array: np.ndarray) -> float:
        positions = torch.as_tensor(position_array, dtype=torch.float64, device=device)
        basis = energy_basis(positions, topology, specification, mean, scale)
        return float(torch.dot(basis, coefficient_tensor).detach().cpu())

    design_energy, design_force = configuration_design(
        structure, specification, environment_mean, environment_scale, device
    )
    del design_energy
    analytic = float(design_force[atom, component] @ coefficients)
    plus = np.asarray(structure.positions, float).copy()
    minus = plus.copy()
    plus[atom, component] += step_A
    minus[atom, component] -= step_A
    numerical = -(energy(plus) - energy(minus)) / (2.0 * step_A)
    return {
        "atom": atom,
        "component": component,
        "step_A": step_A,
        "analytic_force_eV_A": analytic,
        "finite_difference_force_eV_A": numerical,
        "absolute_error_eV_A": abs(analytic - numerical),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--support-data", type=Path, required=True)
    parser.add_argument("--thermal-data", type=Path, required=True)
    parser.add_argument("--harmonic-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--energy-balance", type=float, default=516.5)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("new_DFT_labels") != 0:
        raise ValueError("R2E must not add DFT labels")
    if manifest.get("long_range_model_modified") is not False:
        raise ValueError("long-range model must remain frozen")
    weights = dict(manifest["config_type_weights"])

    train = read(args.train_data, index=":")
    support = read(args.support_data, index=":")
    thermal_all = read(args.thermal_data, index=":")
    harmonic_all = read(args.harmonic_data, index=":")
    harmonic = [
        structure
        for structure in harmonic_all
        if str(structure.info.get("delta_target_role")) == "harmonic_replay"
    ]
    thermal = {
        label: [
            structure
            for structure in thermal_all
            if str(structure.info.get("config_type")) == config_type
        ]
        for config_type, label in THERMAL_LABELS.items()
    }
    if len(train) != 81 or len(support) != 9 or len(harmonic) != 25:
        raise ValueError("unexpected R2E train/support/harmonic counts")
    if {label: len(values) for label, values in thermal.items()} != {
        "T300": 3,
        "T450": 20,
        "T600": 3,
    }:
        raise ValueError("unexpected thermal development counts")

    specification = BasisSpecification()
    environment_mean, environment_scale = environment_statistics(
        train, specification, device
    )
    datasets = {"train": train, "support": support, "harmonic": harmonic, **thermal}
    all_designs = {}
    for label, structures in datasets.items():
        print(f"building {label} designs: {len(structures)}", flush=True)
        all_designs[label] = []
        for index, structure in enumerate(structures):
            all_designs[label].append(
                configuration_design(
                    structure,
                    specification,
                    environment_mean,
                    environment_scale,
                    device,
                )
            )
            if (index + 1) % 10 == 0 or index + 1 == len(structures):
                print(f"  {label}: {index + 1}/{len(structures)}", flush=True)

    force_constants, reference, cell, degauss = load_operator(args.operator)
    del force_constants
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )
    coefficients_by_label = {"depth3_no_bond_order": np.zeros(specification.n_basis)}
    column_scales = {}
    for ridge in RIDGE_GRID:
        label = f"bond_order_ridge_{ridge:.0e}"
        coefficients, column_scale = fit_coefficients(
            train,
            all_designs["train"],
            weights,
            ridge,
            args.energy_balance,
        )
        coefficients_by_label[label] = coefficients
        column_scales[label] = column_scale

    results = {}
    for label, coefficients in coefficients_by_label.items():
        results[label] = candidate_metrics(
            support,
            all_designs["support"],
            thermal,
            {name: all_designs[name] for name in thermal},
            harmonic,
            all_designs["harmonic"],
            coefficients,
            mode,
            reference,
            cell,
        )
        results[label]["coefficient_L2"] = float(np.linalg.norm(coefficients))
    ranking = sorted(
        results, key=lambda label: results[label]["normalized_worst_gate_score"]
    )
    best_label = ranking[0]
    best_coefficients = coefficients_by_label[best_label]

    loco_energy = []
    loco_force = []
    if best_label == "depth3_no_bond_order":
        loco_coefficients = {
            int(structure.info["sscha_index"]): np.zeros(specification.n_basis)
            for structure in support
        }
    else:
        ridge_text = best_label.removeprefix("bond_order_ridge_")
        selected_ridge = float(ridge_text)
        loco_coefficients = {}
        for structure in support:
            sscha_index = int(structure.info["sscha_index"])
            coefficients, _ = fit_coefficients(
                train,
                all_designs["train"],
                weights,
                selected_ridge,
                args.energy_balance,
                excluded_sscha_index=sscha_index,
            )
            loco_coefficients[sscha_index] = coefficients
    for structure, design in zip(support, all_designs["support"], strict=True):
        coefficients = loco_coefficients[int(structure.info["sscha_index"])]
        loco_energy.append(float(design[0] @ coefficients))
        loco_force.append(np.tensordot(design[1], coefficients, axes=(2, 0)))
    loco_result = support_metrics(
        support,
        np.asarray(loco_energy),
        loco_force,
        mode,
        reference,
        cell,
    )

    sscha99 = next(
        structure for structure in support if int(structure.info["sscha_index"]) == 99
    )
    finite_difference = finite_difference_check(
        sscha99,
        best_coefficients,
        specification,
        environment_mean,
        environment_scale,
        device,
    )
    _, best_support_force = predictions(all_designs["support"], best_coefficients)
    maximum_net_force = max(
        float(np.max(np.abs(np.sum(force, axis=0)))) for force in best_support_force
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        args.output_dir / "best_bond_order_model.npz",
        coefficients=best_coefficients,
        environment_mean=environment_mean,
        environment_scale=environment_scale,
        radial_centers_A=specification.centers_A,
        radial_width_A=np.asarray(specification.radial_width_A),
        r_min_A=np.asarray(specification.r_min_A),
        r_max_A=np.asarray(specification.r_max_A),
    )
    candidate_rows = [
        {
            "candidate": label,
            "worst_gate_score": results[label]["normalized_worst_gate_score"],
            "passes": results[label]["passes_all_development_gates"],
            "support_RMSE_meV_A": results[label]["support9_development_fit"][
                "force_error"
            ]["RMSE_meV_A"],
            "support_max_meV_A": results[label]["support9_development_fit"][
                "force_error"
            ]["max_abs_meV_A"],
            "Aprime_RMS_meV_A": results[label]["support9_development_fit"][
                "Aprime_projection"
            ]["force_error"]["RMS_meV_A"],
            "energy_RMSE_meV_config": results[label]["support9_development_fit"][
                "delta_energy_error_after_global_offset"
            ]["RMSE_meV_config"],
            "harmonic_RMSE_meV_A": results[label]["harmonic_replay"][
                "total_force_error"
            ]["RMSE_meV_A"],
        }
        for label in ranking
    ]
    atomic_csv(args.output_dir / "candidate_gate_summary.csv", candidate_rows)
    summary = {
        "status": "R2E_environment_conditioned_conservative_bond_order_complete",
        "finite_temperature_validation_closed": False,
        "reason_not_final": (
            "support structures are development/training data; a new matched "
            "holdout is required after all gates pass"
        ),
        "fixed_condition": {
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": degauss,
        },
        "basis": {
            "energy_form": "sum_b radial_RBF(r_b) times symmetric endpoint-environment invariants",
            "n_radial": specification.n_radial,
            "n_environment": specification.n_environment,
            "n_basis": specification.n_basis,
            "r_min_A": specification.r_min_A,
            "r_max_A": specification.r_max_A,
            "radial_width_A": specification.radial_width_A,
            "environment_features": [
                "constant",
                "mean endpoint other-bond length",
                "central-versus-other bond contrast",
                "squared endpoint mean asymmetry",
                "other-bond splitting",
                "mean 120-degree angle distortion",
                "squared endpoint angle-distortion asymmetry",
            ],
            "forces": "exact negative autograd derivative of scalar energy",
        },
        "fit": {
            "ridge_grid": RIDGE_GRID,
            "energy_balance": args.energy_balance,
            "train_counts": {
                "all": len(train),
                "support": sum(
                    str(structure.info.get("config_type"))
                    == "r2c_fixed_smearing_supported"
                    for structure in train
                ),
                "harmonic": sum(
                    str(structure.info.get("delta_target_role")) == "harmonic_replay"
                    for structure in train
                ),
                "thermal_gradient": 0,
            },
            "config_type_weights": weights,
        },
        "ranking": ranking,
        "best_development_candidate": best_label,
        "best_passes_all_development_gates": results[best_label][
            "passes_all_development_gates"
        ],
        "candidates": results,
        "leave_one_supported_configuration_out": loco_result,
        "energy_force_finite_difference": finite_difference,
        "maximum_support_net_force_component_eV_A": maximum_net_force,
        "Aprime_mode": mode_provenance,
        "inputs": {
            "train_data": {"path": str(args.train_data), "sha256": sha256(args.train_data)},
            "support_data": {"path": str(args.support_data), "sha256": sha256(args.support_data)},
            "thermal_data": {"path": str(args.thermal_data), "sha256": sha256(args.thermal_data)},
            "harmonic_data": {"path": str(args.harmonic_data), "sha256": sha256(args.harmonic_data)},
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {"path": str(args.background), "sha256": sha256(args.background)},
            "corrected_result": {"path": str(args.corrected_result), "sha256": sha256(args.corrected_result)},
        },
        "outputs": {
            "model": str(args.output_dir / "best_bond_order_model.npz"),
            "candidate_csv": str(args.output_dir / "candidate_gate_summary.csv"),
        },
    }
    atomic_json(args.output_dir / "bond_order_fit_summary.json", summary)
    print(
        json.dumps(
            {
                "ranking": ranking,
                "best": best_label,
                "passes": summary["best_passes_all_development_gates"],
                "best_metrics": results[best_label],
                "LOCO": loco_result,
                "finite_difference": finite_difference,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
