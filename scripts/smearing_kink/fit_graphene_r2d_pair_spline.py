#!/usr/bin/env python3
"""Fit a conservative C--C pair-spline residual on top of graphene depth-3.

The spline is an energy model, so its forces are conservative by construction.
Hyperparameters are fixed to a small ridge/loss grid.  Candidate ranking uses
the same support, thermal, and harmonic development gates as R2C.  A final
leave-one-supported-configuration-out audit tests whether improvements transfer
instead of merely memorising the nine R2C DFT structures.
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

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
from scipy.interpolate import BSpline


ROOT = Path(__file__).resolve().parents[2]
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
    ordinary_dataset_metrics,
    support_metrics,
    target_arrays,
    target_delta_energy,
)


RIDGE_GRID = [1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0]
ENERGY_BALANCE_GRID = {
    "forces_only": 0.0,
    # One centered 19.4 meV/config energy residual contributes similarly to
    # 216 force components each at the 30 meV/A support gate.
    "gate_scaled_energy": 516.5,
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


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def minimum_image_vectors(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = positions[None, :, :] - positions[:, None, :]
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


@dataclass(frozen=True)
class PairSplineBasis:
    r_min_A: float
    r_max_A: float
    n_basis: int
    degree: int = 3

    def __post_init__(self) -> None:
        if self.n_basis <= self.degree + 1:
            raise ValueError("n_basis is too small for the spline degree")
        if self.r_min_A >= self.r_max_A:
            raise ValueError("invalid radial interval")

    @property
    def knots(self) -> np.ndarray:
        n_internal = self.n_basis - self.degree - 1
        internal = np.linspace(
            self.r_min_A, self.r_max_A, n_internal + 2
        )[1:-1]
        return np.concatenate(
            [
                np.repeat(self.r_min_A, self.degree + 1),
                internal,
                np.repeat(self.r_max_A, self.degree + 1),
            ]
        )

    def raw_splines(self) -> list[BSpline]:
        identity = np.eye(self.n_basis)
        return [
            BSpline(self.knots, identity[index], self.degree, extrapolate=False)
            for index in range(self.n_basis)
        ]

    def evaluate(self, distance_A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance = np.asarray(distance_A, float)
        values = np.zeros((len(distance), self.n_basis), float)
        derivatives = np.zeros_like(values)
        inside = (distance > self.r_min_A) & (distance < self.r_max_A)
        if not np.any(inside):
            return values, derivatives
        selected = distance[inside]
        span = self.r_max_A - self.r_min_A
        t = (selected - self.r_min_A) / span
        envelope = 16.0 * t**2 * (1.0 - t) ** 2
        envelope_derivative = 32.0 * t * (1.0 - t) * (1.0 - 2.0 * t) / span
        for column, spline in enumerate(self.raw_splines()):
            raw = spline(selected)
            raw_derivative = spline.derivative()(selected)
            values[inside, column] = envelope * raw
            derivatives[inside, column] = (
                envelope_derivative * raw + envelope * raw_derivative
            )
        return values, derivatives

    def design(self, structure) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(structure.positions, float)
        cell = np.asarray(structure.cell, float)
        vectors = minimum_image_vectors(positions, cell)
        distances = np.linalg.norm(vectors, axis=2)
        left, right = np.where(
            np.triu(
                (distances > self.r_min_A) & (distances < self.r_max_A),
                k=1,
            )
        )
        pair_distance = distances[left, right]
        energy_values, energy_derivatives = self.evaluate(pair_distance)
        energy_design = np.sum(energy_values, axis=0)
        force_design = np.zeros((len(structure), 3, self.n_basis), float)
        for pair, (atom_i, atom_j) in enumerate(zip(left, right, strict=True)):
            unit = vectors[atom_i, atom_j] / pair_distance[pair]
            contribution = unit[:, None] * energy_derivatives[pair][None, :]
            force_design[atom_i] += contribution
            force_design[atom_j] -= contribution
        return energy_design, force_design

    def roughness(self) -> np.ndarray:
        grid = np.linspace(self.r_min_A, self.r_max_A, 500)[1:-1]
        step = grid[1] - grid[0]
        values, _ = self.evaluate(grid)
        second = np.gradient(np.gradient(values, step, axis=0), step, axis=0)
        matrix = second.T @ second / len(grid)
        matrix += np.eye(self.n_basis) * max(float(np.trace(matrix)), 1.0) * 1.0e-8
        return matrix


def mace_predictions(
    structures: list, calculator: MACECalculator
) -> tuple[np.ndarray, np.ndarray | list[np.ndarray]]:
    energy = []
    force = []
    for source in structures:
        structure = source.copy()
        structure.calc = calculator
        energy.append(float(structure.get_potential_energy()))
        force.append(np.asarray(structure.get_forces(), float))
    if len({array.shape for array in force}) == 1:
        force_output: np.ndarray | list[np.ndarray] = np.stack(force, axis=0)
    else:
        force_output = force
    return np.asarray(energy), force_output


def pair_predictions(
    designs: list[tuple[np.ndarray, np.ndarray]], coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    energy = np.asarray([design[0] @ coefficients for design in designs])
    force = np.asarray(
        [np.tensordot(design[1], coefficients, axes=(2, 0)) for design in designs]
    )
    return energy, force


def config_weight(structure, weights: dict[str, float]) -> float:
    kind = str(structure.info.get("config_type", "Default"))
    return float(weights.get(kind, weights.get("Default", 1.0)))


def assemble_fit(
    structures: list,
    designs: list[tuple[np.ndarray, np.ndarray]],
    predicted_energy: np.ndarray,
    predicted_force: np.ndarray,
    weights: dict[str, float],
    energy_balance: float,
    exclude_sscha_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    for index, structure in enumerate(structures):
        is_excluded = (
            exclude_sscha_index is not None
            and str(structure.info.get("config_type")) == "r2c_fixed_smearing_supported"
            and int(structure.info.get("sscha_index", -1)) == exclude_sscha_index
        )
        if not is_excluded:
            selected.append(index)

    rows = []
    targets = []
    for index in selected:
        structure = structures[index]
        scale = np.sqrt(config_weight(structure, weights))
        target = np.asarray(structure.arrays["SHORT_RANGE_TARGET_forces"], float)
        rows.append(designs[index][1].reshape(-1, designs[index][1].shape[-1]) * scale)
        targets.append((target - predicted_force[index]).reshape(-1) * scale)

    if energy_balance > 0.0:
        groups: dict[str, list[int]] = {}
        for index in selected:
            structure = structures[index]
            if float(structure.info.get("config_energy_weight", 1.0)) <= 0.0:
                continue
            groups.setdefault(str(structure.info.get("config_type", "Default")), []).append(index)
        for indices in groups.values():
            design = np.asarray([designs[index][0] for index in indices])
            target = np.asarray(
                [target_delta_energy(structures[index]) - predicted_energy[index] for index in indices]
            )
            sample_weight = np.asarray(
                [config_weight(structures[index], weights) for index in indices]
            )
            mean_design = np.average(design, axis=0, weights=sample_weight)
            mean_target = float(np.average(target, weights=sample_weight))
            row_scale = np.sqrt(sample_weight * energy_balance)
            rows.append((design - mean_design) * row_scale[:, None])
            targets.append((target - mean_target) * row_scale)

    return np.concatenate(rows, axis=0), np.concatenate(targets, axis=0)


def fit_coefficients(
    matrix: np.ndarray,
    target: np.ndarray,
    roughness: np.ndarray,
    ridge: float,
) -> np.ndarray:
    gram = matrix.T @ matrix
    relative_scale = max(float(np.trace(gram) / len(gram)), 1.0e-16)
    penalty = ridge * relative_scale * roughness / max(float(np.trace(roughness) / len(roughness)), 1.0e-16)
    return np.linalg.solve(gram + penalty, matrix.T @ target)


def candidate_gate(
    coefficients: np.ndarray,
    basis: PairSplineBasis,
    eval_sets: dict[str, list],
    eval_designs: dict[str, list],
    depth_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    base_forces: dict[str, np.ndarray],
    harmonic_limit: float,
    mode: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
) -> dict:
    combined = {}
    for label in eval_sets:
        pair_energy, pair_force = pair_predictions(eval_designs[label], coefficients)
        depth_energy, depth_force = depth_predictions[label]
        combined[label] = (depth_energy + pair_energy, depth_force + pair_force)

    support_energy, support_force = combined["support"]
    support = support_metrics(
        eval_sets["support"],
        base_forces["support"],
        support_energy,
        support_force,
        mode,
        reference,
        cell,
    )
    ratios = dict(support["normalized_gate_ratios"])
    thermal = {}
    for label in ["T300", "T450", "T600"]:
        energy, force = combined[label]
        metrics = ordinary_dataset_metrics(
            eval_sets[label], base_forces[label], energy, force
        )
        thermal[label] = metrics
        ratios[f"{label}/force_RMSE"] = metrics["total_force_error"]["RMSE_meV_A"] / 30.0
        ratios[f"{label}/force_max_abs"] = metrics["total_force_error"]["max_abs_meV_A"] / 200.0
    harmonic_energy, harmonic_force = combined["harmonic"]
    harmonic = ordinary_dataset_metrics(
        eval_sets["harmonic"],
        base_forces["harmonic"],
        harmonic_energy,
        harmonic_force,
    )
    ratios["harmonic/force_RMSE"] = harmonic["total_force_error"]["RMSE_meV_A"] / harmonic_limit
    return {
        "pair_spline": {
            "r_min_A": basis.r_min_A,
            "r_max_A": basis.r_max_A,
            "n_basis": basis.n_basis,
            "degree": basis.degree,
            "coefficients_eV": coefficients.tolist(),
        },
        "support9_development_fit": support,
        "thermal_development": thermal,
        "harmonic_replay": harmonic,
        "normalized_gate_ratios": ratios,
        "normalized_worst_gate_score": float(max(ratios.values())),
        "passes_all_development_gates": bool(all(value <= 1.0 for value in ratios.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--depth-model", type=Path, required=True)
    parser.add_argument("--support-data", type=Path, required=True)
    parser.add_argument("--thermal", action="append", required=True)
    parser.add_argument("--harmonic", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--r-min-A", type=float, default=1.15)
    parser.add_argument("--r-max-A", type=float, default=1.75)
    parser.add_argument("--n-basis", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    thermal_paths = {}
    for specification in args.thermal:
        if "=" not in specification:
            raise ValueError("thermal inputs use LABEL=/path")
        label, raw_path = specification.split("=", 1)
        thermal_paths[label] = Path(raw_path)
    if sorted(thermal_paths) != ["T300", "T450", "T600"]:
        raise ValueError("thermal inputs must define T300, T450, and T600")

    train = read(args.train, index=":")
    support = read(args.support_data, index=":")
    eval_sets = {
        "support": support,
        **{label: read(path, index=":") for label, path in thermal_paths.items()},
        "harmonic": read(args.harmonic, index=":"),
    }
    manifest = json.loads(args.manifest.read_text())
    weights = {key: float(value) for key, value in manifest["config_type_weights"].items()}
    basis = PairSplineBasis(args.r_min_A, args.r_max_A, args.n_basis)

    train_designs = [basis.design(structure) for structure in train]
    eval_designs = {
        label: [basis.design(structure) for structure in structures]
        for label, structures in eval_sets.items()
    }

    depth = MACECalculator(
        model_paths=str(args.depth_model), device=args.device, default_dtype="float32"
    )
    train_depth_energy, train_depth_force = mace_predictions(train, depth)
    depth_predictions = {
        label: mace_predictions(structures, depth)
        for label, structures in eval_sets.items()
    }
    del depth

    base = MACECalculator(
        model_paths=str(args.base_model), device=args.device, default_dtype="float32"
    )
    base_forces = {
        label: mace_predictions(structures, base)[1]
        for label, structures in eval_sets.items()
    }
    base_harmonic_errors = []
    for index, structure in enumerate(eval_sets["harmonic"]):
        target_total, long_force = target_arrays(structure, base_forces["harmonic"][index])
        base_harmonic_errors.append(base_forces["harmonic"][index] + long_force - target_total)
    base_harmonic_force = force_metrics(np.asarray(base_harmonic_errors))
    harmonic_limit = 1.25 * base_harmonic_force["RMSE_meV_A"]
    del base

    _, reference, cell, degauss = load_operator(args.operator)
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )

    roughness = basis.roughness()
    candidates = {}
    fit_records = []
    zero = np.zeros(args.n_basis)
    candidates["depth3_no_pair"] = candidate_gate(
        zero,
        basis,
        eval_sets,
        eval_designs,
        depth_predictions,
        base_forces,
        harmonic_limit,
        mode,
        reference,
        cell,
    )
    for loss_name, energy_balance in ENERGY_BALANCE_GRID.items():
        matrix, target = assemble_fit(
            train,
            train_designs,
            train_depth_energy,
            train_depth_force,
            weights,
            energy_balance,
        )
        for ridge in RIDGE_GRID:
            coefficients = fit_coefficients(matrix, target, roughness, ridge)
            label = f"pair_{loss_name}_ridge_{ridge:.0e}".replace("+", "")
            result = candidate_gate(
                coefficients,
                basis,
                eval_sets,
                eval_designs,
                depth_predictions,
                base_forces,
                harmonic_limit,
                mode,
                reference,
                cell,
            )
            residual = matrix @ coefficients - target
            result["fit"] = {
                "loss_name": loss_name,
                "energy_balance": energy_balance,
                "ridge": ridge,
                "n_equations": int(len(target)),
                "weighted_equation_RMSE": float(np.sqrt(np.mean(residual**2))),
                "coefficient_norm_eV": float(np.linalg.norm(coefficients)),
            }
            candidates[label] = result
            support_gate = result["support9_development_fit"]["fixed_gate"]
            fit_records.append(
                {
                    "candidate": label,
                    "loss_name": loss_name,
                    "energy_balance": energy_balance,
                    "ridge": ridge,
                    "worst_gate_score": result["normalized_worst_gate_score"],
                    "passes_all": result["passes_all_development_gates"],
                    "support_force_RMSE_meV_A": support_gate["force_component_RMSE_meV_A"]["observed"],
                    "support_force_max_meV_A": support_gate["force_component_max_abs_meV_A"]["observed"],
                    "support_Aprime_RMS_meV_A": support_gate["Aprime_projected_force_RMS_meV_A"]["observed"],
                    "support_energy_RMSE_meV_config": support_gate["centered_energy_RMSE_meV_config"]["observed"],
                    "harmonic_RMSE_meV_A": result["harmonic_replay"]["total_force_error"]["RMSE_meV_A"],
                }
            )

    ranking = sorted(candidates, key=lambda name: candidates[name]["normalized_worst_gate_score"])
    best_label = ranking[0]
    best = candidates[best_label]
    best_coefficients = np.asarray(best["pair_spline"]["coefficients_eV"])

    # Leave each supported DFT structure out of the fit, using the chosen fixed
    # hyperparameters, and predict that configuration once.
    if best_label == "depth3_no_pair":
        best_loss_name = "forces_only"
        best_energy_balance = 0.0
        best_ridge = 0.0
    else:
        best_loss_name = best["fit"]["loss_name"]
        best_energy_balance = float(best["fit"]["energy_balance"])
        best_ridge = float(best["fit"]["ridge"])
    loco_errors = []
    loco_aprime = []
    loco_records = []
    support_depth_energy, support_depth_force = depth_predictions["support"]
    for support_index, structure in enumerate(support):
        sscha_index = int(structure.info["sscha_index"])
        if best_label == "depth3_no_pair":
            coefficients = zero
        else:
            matrix, target = assemble_fit(
                train,
                train_designs,
                train_depth_energy,
                train_depth_force,
                weights,
                best_energy_balance,
                exclude_sscha_index=sscha_index,
            )
            coefficients = fit_coefficients(matrix, target, roughness, best_ridge)
        _, pair_force = pair_predictions([eval_designs["support"][support_index]], coefficients)
        predicted = support_depth_force[support_index] + pair_force[0]
        target_force = np.asarray(structure.arrays["SHORT_RANGE_TARGET_forces"], float)
        error = predicted - target_force
        mapping, _ = structure_mapping(structure, reference, cell)
        aprime = np.vdot(mode[mapping].reshape(-1), error.reshape(-1))
        loco_errors.append(error)
        loco_aprime.append(aprime)
        metrics = force_metrics(error[None])
        loco_records.append(
            {
                "sscha_index": sscha_index,
                "force_RMSE_meV_A": metrics["RMSE_meV_A"],
                "force_max_abs_meV_A": metrics["max_abs_meV_A"],
                "Aprime_force_error_abs_meV_A": float(abs(aprime) * 1000.0),
            }
        )
    loco_errors = np.asarray(loco_errors)
    loco_aprime = np.asarray(loco_aprime)
    loco = {
        "selection": best_label,
        "protocol": "refit after excluding one r2c_fixed_smearing_supported configuration",
        "force_error": force_metrics(loco_errors),
        "Aprime_force_error": complex_metrics(loco_aprime),
        "records": loco_records,
    }

    payload = {
        "status": "R2D_conservative_pair_spline_development_complete",
        "finite_temperature_validation_closed": False,
        "reason_not_final": (
            "pair residual is selected on development data; a new order-safe matched holdout remains required"
        ),
        "fixed_condition": {
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": degauss,
        },
        "basis": {
            "species_pair": "C-C",
            "r_min_A": basis.r_min_A,
            "r_max_A": basis.r_max_A,
            "n_basis": basis.n_basis,
            "degree": basis.degree,
            "boundary_envelope": "16*t^2*(1-t)^2; energy and force vanish at both bounds",
            "conservative": True,
        },
        "fixed_search": {
            "ridge_grid": RIDGE_GRID,
            "energy_balance_grid": ENERGY_BALANCE_GRID,
            "ranking_metric": "maximum normalized frozen development gate ratio",
        },
        "frozen_base_harmonic_force": base_harmonic_force,
        "harmonic_limit_meV_A": harmonic_limit,
        "ranking": ranking,
        "best_development_candidate": best_label,
        "best_passes_all_development_gates": best["passes_all_development_gates"],
        "leave_one_supported_configuration_out": loco,
        "Aprime_mode": mode_provenance,
        "candidates": candidates,
        "inputs": {
            "train": {"path": str(args.train), "sha256": sha256(args.train)},
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "base_model": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
            "depth_model": {"path": str(args.depth_model), "sha256": sha256(args.depth_model)},
            "support_data": {"path": str(args.support_data), "sha256": sha256(args.support_data)},
            "thermal": {label: {"path": str(path), "sha256": sha256(path)} for label, path in thermal_paths.items()},
            "harmonic": {"path": str(args.harmonic), "sha256": sha256(args.harmonic)},
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {"path": str(args.background), "sha256": sha256(args.background)},
            "corrected_result": {"path": str(args.corrected_result), "sha256": sha256(args.corrected_result)},
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "pair_spline_fit_summary.json", payload)
    atomic_csv(args.output_dir / "pair_spline_candidate_table.csv", list(fit_records[0]), fit_records)
    atomic_csv(
        args.output_dir / "pair_spline_loco_records.csv",
        list(loco_records[0]),
        loco_records,
    )
    atomic_npz(
        args.output_dir / "best_pair_spline.npz",
        coefficients_eV=best_coefficients,
        knots_A=basis.knots,
        r_min_A=np.asarray(basis.r_min_A),
        r_max_A=np.asarray(basis.r_max_A),
        degree=np.asarray(basis.degree),
    )

    grid = np.linspace(basis.r_min_A, basis.r_max_A, 500)
    value, derivative = basis.evaluate(grid)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    axes[0].plot(grid, value @ best_coefficients, color="#1f77b4")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_xlabel("C-C distance (Å)")
    axes[0].set_ylabel("pair residual energy (eV)")
    axes[0].set_title(best_label)
    axes[1].plot(grid, derivative @ best_coefficients, color="#d62728")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_xlabel("C-C distance (Å)")
    axes[1].set_ylabel("pair force amplitude φ′(r) (eV/Å)")
    figure.savefig(args.output_dir / "best_pair_spline.png", dpi=200)
    figure.savefig(args.output_dir / "best_pair_spline.pdf")
    plt.close(figure)

    concise = {
        "ranking": ranking,
        "best": best_label,
        "passes": best["passes_all_development_gates"],
        "worst_score": best["normalized_worst_gate_score"],
        "support_gate": best["support9_development_fit"]["fixed_gate"],
        "thermal_force": {
            label: result["total_force_error"]
            for label, result in best["thermal_development"].items()
        },
        "harmonic_force": best["harmonic_replay"]["total_force_error"],
        "LOCO": loco,
    }
    print(json.dumps(concise, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
