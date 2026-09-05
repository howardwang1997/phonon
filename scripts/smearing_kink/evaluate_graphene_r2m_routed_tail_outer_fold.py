#!/usr/bin/env python3
"""Post-freeze evaluation of one R2M routed-tail outer fold.

This is the first program in the fold workflow that opens ``support_held1``.
It verifies the frozen epoch-240 EMA artifact before doing so, evaluates the
held support configuration and the support-free seed1/harmonic validation
set, and runs conservative-mechanics checks.  It never reads E50 seed2.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator

from graphene_r2m_aprime_eval import (
    folded_k_aprime_mode,
    load_operator,
    structure_mapping,
)
from graphene_r2m_routed_tail import (
    correction_energy_and_forces,
    load_routed_tail,
    sha256,
    strict_json,
    torch_load,
)
from train_graphene_r2m_routed_tail_outer_fold import (
    batch_data,
    geometry_fingerprint,
    pristine_supercell,
)


LIMITS = {
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


def reject_seed2_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in ("seed2", "reserved_e50", "reserved-e50")):
        raise ValueError(f"outer-fold evaluator rejects seed2 path: {path}")


def exact_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"missing, empty, or symlinked {label}: {path}")


def same_resolved_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def validate_training_metrics(path: Path) -> None:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 240:
        raise ValueError("training metrics must contain exactly 240 epochs")
    if [int(item.get("epoch", -1)) for item in records] != list(range(1, 241)):
        raise ValueError("training metrics epochs are not exactly 1..240")
    if any(int(item.get("n_optimizer_steps", -1)) != 8 for item in records):
        raise ValueError("training metrics do not record eight steps per epoch")
    for record in records:
        for name, value in record.items():
            if name not in {"epoch", "n_optimizer_steps"} and not np.isfinite(
                float(value)
            ):
                raise ValueError(f"non-finite training metric {name!r}")


def force_metrics(errors: list[np.ndarray] | np.ndarray) -> dict:
    array = np.asarray(errors, float)
    flat = array.reshape(-1)
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(flat**2))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(flat))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(flat))),
        "n_force_components": int(flat.size),
    }


def restoring_slope(coordinates: np.ndarray, projected_forces: np.ndarray) -> float:
    denominator = float(np.vdot(coordinates, coordinates).real)
    if denominator <= 1.0e-16:
        raise ValueError("A-prime coordinates have zero norm")
    return -float(np.vdot(coordinates, projected_forces).real / denominator)


def core_predictions(
    structures: list[Atoms], core_path: Path, device: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    calculator = MACECalculator(
        model_paths=str(core_path), device=device, default_dtype="float32"
    )
    energies = []
    forces = []
    for source in structures:
        atoms = source.copy()
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), float))
    del calculator
    return np.asarray(energies), forces


def tail_predictions(
    structures: list[Atoms],
    core: torch.nn.Module,
    tail,
    schema: dict,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    energies = []
    forces = []
    gates = []
    node_energies = []
    tail.eval()
    for source in structures:
        data = batch_data([source], core, device)
        energy, force, diagnostics = correction_energy_and_forces(
            core, tail, data, schema, create_graph=False
        )
        energies.append(float(energy[0].detach().cpu()))
        forces.append(force.detach().cpu().numpy())
        gates.append(diagnostics["gate"].detach().cpu().numpy())
        node_energies.append(diagnostics["node_energy"].detach().cpu().numpy())
    return np.asarray(energies), forces, gates, node_energies


def seed1_metrics(
    structures: list[Atoms], predicted_core_tail: list[np.ndarray]
) -> tuple[dict, dict[str, np.ndarray]]:
    errors = []
    coordinates = []
    predicted_modes = []
    target_modes = []
    error_modes = []
    for source, short_force in zip(structures, predicted_core_tail, strict=True):
        error = short_force - np.asarray(source.arrays["REF_forces"], float)
        mode = np.asarray(source.arrays["APRIME_mode_real"], float) + 1.0j * np.asarray(
            source.arrays["APRIME_mode_imag"], float
        )
        coordinate = complex(
            float(source.info["APRIME_coordinate_real_A"]),
            float(source.info["APRIME_coordinate_imag_A"]),
        )
        baseline = np.asarray(source.arrays["FOUNDATION_BASE_forces"], float) + np.asarray(
            source.arrays["FROZEN_Q6_forces"], float
        )
        target_total = np.asarray(source.arrays["DFT_TOTAL_forces"], float)
        predicted_total = baseline + short_force
        predicted_mode = np.vdot(mode.reshape(-1), predicted_total.reshape(-1))
        target_mode = np.vdot(mode.reshape(-1), target_total.reshape(-1))
        errors.append(error)
        coordinates.append(coordinate)
        predicted_modes.append(predicted_mode)
        target_modes.append(target_mode)
        error_modes.append(predicted_mode - target_mode)
    coordinates_array = np.asarray(coordinates)
    predicted_array = np.asarray(predicted_modes)
    target_array = np.asarray(target_modes)
    predicted_slope = restoring_slope(coordinates_array, predicted_array)
    target_slope = restoring_slope(coordinates_array, target_array)
    relative = (predicted_slope - target_slope) / target_slope
    result = {
        "force_error": force_metrics(errors),
        "Aprime_RMS_meV_A": float(
            1000.0 * np.sqrt(np.mean(np.abs(np.asarray(error_modes)) ** 2))
        ),
        "predicted_restoring_slope_eV_A2": predicted_slope,
        "DFT_restoring_slope_eV_A2": target_slope,
        "restoring_slope_relative_error": float(relative),
    }
    arrays = {
        "coordinates": coordinates_array,
        "predicted_modes": predicted_array,
        "target_modes": target_array,
        "force_errors": np.asarray(errors),
    }
    return result, arrays


def finite_difference_gate(
    structure: Atoms,
    core: torch.nn.Module,
    tail,
    schema: dict,
    device: torch.device,
    step_A: float = 1.0e-4,
) -> dict:
    base_energy, base_force, _, _ = tail_predictions(
        [structure], core, tail, schema, device
    )
    records = []
    for atom, axis in ((0, 0), (0, 1), (1, 2)):
        plus = structure.copy()
        minus = structure.copy()
        plus.positions[atom, axis] += step_A
        minus.positions[atom, axis] -= step_A
        plus_energy = tail_predictions([plus], core, tail, schema, device)[0][0]
        minus_energy = tail_predictions([minus], core, tail, schema, device)[0][0]
        numerical = -(plus_energy - minus_energy) / (2.0 * step_A)
        analytic = float(base_force[0][atom, axis])
        records.append(
            {
                "atom": atom,
                "axis": axis,
                "analytic_eV_A": analytic,
                "numerical_eV_A": float(numerical),
                "absolute_error_eV_A": float(abs(analytic - numerical)),
            }
        )
    return {
        "step_A": step_A,
        "tail_energy_eV": float(base_energy[0]),
        "records": records,
        "maximum_absolute_error_eV_A": max(
            item["absolute_error_eV_A"] for item in records
        ),
    }


def hessian_symmetry_gate(
    structure: Atoms,
    core: torch.nn.Module,
    tail,
    schema: dict,
    device: torch.device,
) -> dict:
    template = batch_data([structure], core, device)
    initial = template["positions"].detach().clone().requires_grad_(True)

    def force_function(positions: torch.Tensor) -> torch.Tensor:
        data = dict(template)
        data["positions"] = positions
        _, force, _ = correction_energy_and_forces(
            core, tail, data, schema, create_graph=True
        )
        return force

    jacobian = torch.autograd.functional.jacobian(
        force_function, initial, create_graph=False, vectorize=False
    )
    size = initial.numel()
    matrix = jacobian.reshape(size, size)
    antisymmetric = matrix - matrix.T
    return {
        "matrix_dimension": size,
        "maximum_absolute_antisymmetry_eV_A2": float(
            torch.max(torch.abs(antisymmetric)).detach().cpu()
        ),
        "RMS_antisymmetry_eV_A2": float(
            torch.sqrt(torch.mean(antisymmetric.square())).detach().cpu()
        ),
    }


def transform_structure(structure: Atoms, matrix: np.ndarray) -> Atoms:
    transformed = structure.copy()
    transformed.positions = np.asarray(structure.positions, float) @ matrix.T
    transformed.cell = np.asarray(structure.cell, float) @ matrix.T
    return transformed


def o3_gate(
    structure: Atoms,
    core: torch.nn.Module,
    tail,
    schema: dict,
    device: torch.device,
) -> dict:
    angle = 0.371
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    reflection = np.diag([-1.0, 1.0, 1.0])
    base_energy, base_force, _, _ = tail_predictions(
        [structure], core, tail, schema, device
    )
    records = []
    for label, matrix in (("rotation", rotation), ("reflection", reflection)):
        candidate = transform_structure(structure, matrix)
        energy, force, _, _ = tail_predictions([candidate], core, tail, schema, device)
        records.append(
            {
                "transformation": label,
                "determinant": float(np.linalg.det(matrix)),
                "energy_absolute_error_eV": float(abs(energy[0] - base_energy[0])),
                "force_equivariance_max_abs_eV_A": float(
                    np.max(np.abs(force[0] - base_force[0] @ matrix.T))
                ),
            }
        )
    return {
        "records": records,
        "energy_max_abs_eV": max(item["energy_absolute_error_eV"] for item in records),
        "force_equivariance_max_abs_eV_A": max(
            item["force_equivariance_max_abs_eV_A"] for item in records
        ),
    }


def centered_perturbed_supercell(primitive: Atoms, repeat: int) -> tuple[Atoms, int]:
    result = primitive.repeat((repeat, repeat, 1))
    primitive_cell = np.asarray(primitive.cell, float)
    target = (
        np.asarray(primitive.positions[0], float)
        + (repeat // 2) * primitive_cell[0]
        + (repeat // 2) * primitive_cell[1]
    )
    index = int(np.argmin(np.linalg.norm(np.asarray(result.positions) - target, axis=1)))
    if float(np.linalg.norm(np.asarray(result.positions[index]) - target)) > 1.0e-8:
        raise ValueError("could not identify the central pristine atom")
    result.positions[index] += np.asarray([0.015, -0.010, 0.005])
    return result, index


def size_consistency_gate(
    pristine_path: Path,
    core_path: Path,
    core: torch.nn.Module,
    tail,
    schema: dict,
    device: torch.device,
    device_name: str,
) -> dict:
    primitive = read(pristine_path, index=0)
    six, center6 = centered_perturbed_supercell(primitive, 6)
    eight, center8 = centered_perturbed_supercell(primitive, 8)
    tail_energy, tail_force, gates, node_energy = tail_predictions(
        [six, eight], core, tail, schema, device
    )
    del tail_energy
    _, core_force = core_predictions([six, eight], core_path, device_name)
    return {
        "test": "same centered local displacement in pristine 6x6 and 8x8 cells",
        "center_indices": [center6, center8],
        "tail_force_max_abs_difference_eV_A": float(
            np.max(np.abs(tail_force[0][center6] - tail_force[1][center8]))
        ),
        "tail_node_energy_abs_difference_eV": float(
            abs(node_energy[0][center6] - node_energy[1][center8])
        ),
        "tail_gate_abs_difference": float(abs(gates[0][center6] - gates[1][center8])),
        "core_plus_tail_force_max_abs_difference_eV_A": float(
            np.max(
                np.abs(
                    core_force[0][center6]
                    + tail_force[0][center6]
                    - core_force[1][center8]
                    - tail_force[1][center8]
                )
            )
        ),
        "scope": "strict local gate for frozen core plus routed tail; foundation and q6 remain outside",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--pristine", type=Path, required=True)
    parser.add_argument("--core-model", type=Path, required=True)
    parser.add_argument("--core-sha256", required=True)
    parser.add_argument("--launcher-freeze", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--thermal-result", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.fold_dir,
        args.training_dir,
        args.valid,
        args.pristine,
        args.core_model,
        args.launcher_freeze,
        args.operator,
        args.background,
        args.thermal_result,
        args.output,
    ):
        reject_seed2_path(path)
    if sha256(args.core_model) != args.core_sha256:
        raise ValueError("frozen core SHA-256 mismatch")
    fold_manifest_path = args.fold_dir / "fold_manifest.json"
    train_summary_path = args.training_dir / "training_summary.json"
    training_freeze_path = args.training_dir / "training_freeze.json"
    training_metrics_path = args.training_dir / "training_metrics.jsonl"
    training_done_path = args.training_dir / "TRAINING_DONE"
    expected_artifact = args.training_dir / "r2m_routed_tail_epoch240_ema.pt"
    for path, label in (
        (fold_manifest_path, "fold manifest"),
        (train_summary_path, "training summary"),
        (training_freeze_path, "training freeze"),
        (training_metrics_path, "training metrics"),
        (expected_artifact, "epoch-240 EMA artifact"),
        (args.launcher_freeze, "launcher freeze"),
    ):
        exact_regular_file(path, label)
    if (
        not training_done_path.is_file()
        or training_done_path.is_symlink()
        or training_done_path.stat().st_size != 0
    ):
        raise ValueError("missing or invalid empty TRAINING_DONE marker")
    fold = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    if (
        fold.get("status") != "R2M_routed_tail_outer_fold_frozen_before_training"
        or fold.get("selection_policy")
        != "fixed_epoch_240_no_support_checkpoint_selection"
        or fold.get("strict_nested_inner_CV_run") is not False
        or fold.get("leakage", {}).get("E50_seed2_read") is not False
        or fold.get("leakage", {}).get(
            "held_geometry_or_label_in_scaler_gradient_or_selection"
        )
        is not False
    ):
        raise ValueError("wrong or unsafe frozen outer-fold manifest")
    if fold["inputs"]["core"]["sha256"] != args.core_sha256:
        raise ValueError("fold and supplied core SHA-256 differ")
    if not same_resolved_path(Path(fold["inputs"]["core"]["path"]), args.core_model):
        raise ValueError("fold was prepared with a different core path")
    launcher = json.loads(args.launcher_freeze.read_text(encoding="utf-8"))
    if (
        launcher.get("status")
        != "R2M_routed_tail_outer_fold_launcher_frozen_before_training"
        or int(launcher.get("outer_fold", -1)) != int(fold["outer_fold"])
        or int(launcher.get("held_sscha_index", -1))
        != int(fold["held_sscha_index"])
        or launcher.get("selection_policy")
        != "fixed_epoch_240_no_support_checkpoint_selection"
        or launcher.get("E50_seed2_read_or_accepted_as_input") is not False
        or launcher.get("inputs", {}).get("core", {}).get("sha256")
        != args.core_sha256
        or launcher.get("inputs", {}).get("fold_manifest", {}).get("sha256")
        != sha256(fold_manifest_path)
    ):
        raise ValueError("launcher freeze is not bound to this outer fold/core")
    executed_dependencies = [
        Path(__file__).resolve(),
        Path(__file__).with_name("graphene_r2m_routed_tail.py").resolve(),
        Path(__file__).with_name(
            "train_graphene_r2m_routed_tail_outer_fold.py"
        ).resolve(),
        Path(__file__).with_name("graphene_r2m_aprime_eval.py").resolve(),
    ]
    for path in executed_dependencies:
        record = launcher.get("code_snapshots", {}).get(path.name, {})
        if (
            record.get("sha256") != sha256(path)
            or Path(str(record.get("path", ""))).resolve() != path
        ):
            raise ValueError(f"launcher did not freeze the executed {path.name}")
    training = json.loads(train_summary_path.read_text(encoding="utf-8"))
    if (
        training.get("status")
        != "R2M_routed_tail_outer_fold_training_complete_pending_held_evaluation"
        or training.get("selection_policy")
        != "fixed_epoch_240_no_support_checkpoint_selection"
        or training.get("selected_epoch") != 240
        or training.get("selected_state") != "EMA"
        or training.get("outer_held_read_for_training_scaler_or_selection") is not False
        or training.get("E50_seed2_read") is not False
    ):
        raise ValueError("training artifact is not frozen for outer-held evaluation")
    if (
        int(training["outer_fold"]) != int(fold["outer_fold"])
        or int(training["held_sscha_index"]) != int(fold["held_sscha_index"])
    ):
        raise ValueError("training and fold identity differ")
    artifact = Path(str(training.get("artifact", {}).get("path", "")))
    if not same_resolved_path(artifact, expected_artifact):
        raise ValueError("training summary points outside the canonical artifact path")
    if sha256(expected_artifact) != training.get("artifact", {}).get("sha256"):
        raise ValueError("frozen routed-tail artifact changed")
    if sha256(training_freeze_path) != training.get("training_freeze_sha256"):
        raise ValueError("training freeze SHA-256 mismatch")
    if sha256(training_metrics_path) != training.get("training_metrics_sha256"):
        raise ValueError("training metrics SHA-256 mismatch")
    if (
        not same_resolved_path(
            Path(str(training.get("training_freeze_path", ""))),
            training_freeze_path,
        )
        or not same_resolved_path(
            Path(str(training.get("training_metrics_path", ""))),
            training_metrics_path,
        )
        or training.get("fold_manifest", {}).get("sha256")
        != sha256(fold_manifest_path)
        or not same_resolved_path(
            Path(str(training.get("fold_manifest", {}).get("path", ""))),
            fold_manifest_path,
        )
    ):
        raise ValueError("training summary paths are not canonical for this fold")
    training_freeze = json.loads(training_freeze_path.read_text(encoding="utf-8"))
    if (
        training_freeze.get("status")
        != "frozen_before_R2M_routed_tail_outer_fold_training"
        or int(training_freeze.get("outer_fold", -1)) != int(fold["outer_fold"])
        or int(training_freeze.get("held_sscha_index", -1))
        != int(fold["held_sscha_index"])
        or training_freeze.get("selection_policy")
        != "fixed_epoch_240_no_support_checkpoint_selection"
        or int(training_freeze.get("epochs", -1)) != 240
    ):
        raise ValueError("training freeze identity or fixed endpoint changed")
    leakage = training_freeze.get("leakage", {})
    required_false_leakage = {
        "outer_held_file_read",
        "outer_held_geometry_label_scaler_gradient_or_selection",
        "E50_seed2_read",
        "validation_file_read",
        "support_checkpoint_selection",
    }
    if any(leakage.get(name) is not False for name in required_false_leakage):
        raise ValueError("training freeze does not prove leakage-safe training")
    freeze_inputs = training_freeze.get("inputs", {})
    if (
        freeze_inputs.get("core", {}).get("sha256") != args.core_sha256
        or freeze_inputs.get("fold_manifest", {}).get("sha256")
        != sha256(fold_manifest_path)
        or not same_resolved_path(
            Path(str(freeze_inputs.get("core", {}).get("path", ""))),
            args.core_model,
        )
        or not same_resolved_path(
            Path(str(freeze_inputs.get("fold_manifest", {}).get("path", ""))),
            fold_manifest_path,
        )
        or freeze_inputs.get("launcher_freeze", {}).get("sha256")
        != sha256(args.launcher_freeze)
        or not same_resolved_path(
            Path(str(freeze_inputs.get("launcher_freeze", {}).get("path", ""))),
            args.launcher_freeze,
        )
    ):
        raise ValueError(
            "training freeze is not bound to this core/fold/launcher freeze"
        )
    validate_training_metrics(training_metrics_path)
    frozen_post_training_inputs = {
        "replay_valid_post_freeze_only": args.valid,
        "pristine": args.pristine,
        "operator_post_freeze_only": args.operator,
        "background_post_freeze_only": args.background,
        "thermal_result_post_freeze_only": args.thermal_result,
        "aprime_evaluation_helper": Path(__file__).with_name(
            "graphene_r2m_aprime_eval.py"
        ),
    }
    for name, path in frozen_post_training_inputs.items():
        if sha256(path) != fold["inputs"][name]["sha256"]:
            raise ValueError(f"post-freeze evaluation input changed: {name}")

    device = torch.device(args.device)
    tail, schema, metadata = load_routed_tail(
        expected_artifact, device, args.core_sha256
    )
    if schema.get("schema_sha256") != fold["feature_schema"]["schema_sha256"]:
        raise ValueError("tail feature schema differs from prepared fold")
    if (
        int(metadata.get("outer_fold", -1)) != int(fold["outer_fold"])
        or int(metadata.get("held_sscha_index", -1))
        != int(fold["held_sscha_index"])
        or int(metadata.get("epoch", -1)) != 240
        or metadata.get("state") != "EMA"
        or metadata.get("selection_policy")
        != "fixed_epoch_240_no_support_checkpoint_selection"
        or metadata.get("outer_held_read_before_freeze") is not False
        or metadata.get("E50_seed2_read") is not False
    ):
        raise ValueError("routed-tail artifact metadata violates fold freeze")

    valid = read(args.valid, index=":")
    valid_roles = Counter(
        str(item.info.get("r2m_target_role", "")) for item in valid
    )
    if len(valid) != 45 or valid_roles != Counter(
        {"exact_e50_seed1": 20, "harmonic_validation": 25}
    ):
        raise ValueError("valid must be exactly seed1=20 plus harmonic=25")
    if any("seed2" in str(item.info).lower() for item in valid):
        raise ValueError("seed2 metadata entered post-freeze validation")
    seed1 = [
        item
        for item in valid
        if str(item.info.get("r2m_target_role")) == "exact_e50_seed1"
    ]
    harmonic = [
        item
        for item in valid
        if str(item.info.get("r2m_target_role")) == "harmonic_validation"
    ]
    if any(int(item.info.get("trajectory_seed", -1)) != 1 for item in seed1):
        raise ValueError("non-seed1 E50 record entered validation")

    held_path = args.fold_dir / "support_held1.xyz"
    if sha256(held_path) != fold["outputs"]["support_held1.xyz"]:
        raise ValueError("outer-held file changed")
    held = read(held_path, index=":")
    if len(held) != 1:
        raise ValueError("outer-held file must contain exactly one configuration")
    if geometry_fingerprint(held[0]) != fold["leakage"]["held_support_fingerprint"]:
        raise ValueError("outer-held geometry fingerprint changed")

    core = torch_load(args.core_model, map_location=device)
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    all_structures = held + seed1 + harmonic
    core_energy, core_force = core_predictions(all_structures, args.core_model, args.device)
    tail_energy, tail_force, gates, _ = tail_predictions(
        all_structures, core, tail, schema, device
    )
    held_core_force = np.asarray(held[0].arrays["R2M_CORE_forces"], float)
    if float(np.max(np.abs(held_core_force - core_force[0]))) > 2.0e-5:
        raise ValueError("held core replay differs from fold preparation")
    held_predicted_short = core_force[0] + tail_force[0]
    held_target_short = np.asarray(held[0].arrays["R2M_SHORT_TARGET_forces"], float)
    held_error = held_predicted_short - held_target_short
    held_energy_error = (
        float(tail_energy[0])
        - float(held[0].info["R2M_TAIL_TARGET_energy_centered"])
    )

    seed1_start = 1
    harmonic_start = seed1_start + len(seed1)
    seed1_short = [
        core_force[index] + tail_force[index]
        for index in range(seed1_start, harmonic_start)
    ]
    seed1_result, seed1_arrays = seed1_metrics(seed1, seed1_short)
    harmonic_errors = [
        core_force[index] + tail_force[index] - np.asarray(source.arrays["REF_forces"], float)
        for index, source in zip(
            range(harmonic_start, len(all_structures)), harmonic, strict=True
        )
    ]
    harmonic_gate_means = [
        float(np.mean(gates[index]))
        for index in range(harmonic_start, len(all_structures))
    ]
    harmonic_result = {
        "force_error": force_metrics(harmonic_errors),
        "pooled_gate_mean": float(
            np.mean(np.concatenate(gates[harmonic_start:], axis=0))
        ),
        "per_configuration_gate_mean": harmonic_gate_means,
        "maximum_configuration_gate_mean": max(harmonic_gate_means),
    }

    force_constants, operator_reference, operator_cell, operator_degauss = load_operator(
        args.operator
    )
    del force_constants
    if abs(float(held[0].info["degauss_Ry"]) - operator_degauss) > 5.0e-10:
        raise ValueError("support and q6 operator degauss differ")
    support_mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.thermal_result, operator_reference, operator_cell
    )
    mapping, displacement = structure_mapping(
        held[0], operator_reference, operator_cell
    )
    mode = support_mode[mapping]
    coordinate = np.vdot(mode.reshape(-1), displacement.reshape(-1))
    predicted_total = (
        np.asarray(held[0].arrays["BASE_forces"], float)
        + np.asarray(held[0].arrays["LONG_RANGE_forces"], float)
        + held_predicted_short
    )
    target_total = np.asarray(held[0].arrays["TOTAL_forces"], float)
    predicted_mode = np.vdot(mode.reshape(-1), predicted_total.reshape(-1))
    target_mode = np.vdot(mode.reshape(-1), target_total.reshape(-1))

    finite_difference = finite_difference_gate(held[0], core, tail, schema, device)
    hessian = hessian_symmetry_gate(held[0], core, tail, schema, device)
    o3 = o3_gate(held[0], core, tail, schema, device)
    size = size_consistency_gate(
        args.pristine,
        args.core_model,
        core,
        tail,
        schema,
        device,
        args.device,
    )
    observed = {
        "support_force_RMSE_meV_A": force_metrics([held_error])["RMSE_meV_A"],
        "support_force_max_abs_meV_A": force_metrics([held_error])["max_abs_meV_A"],
        "support_centered_energy_abs_meV_config": 1000.0 * abs(held_energy_error),
        "seed1_force_RMSE_meV_A": seed1_result["force_error"]["RMSE_meV_A"],
        "seed1_force_max_abs_meV_A": seed1_result["force_error"]["max_abs_meV_A"],
        "seed1_Aprime_RMS_meV_A": seed1_result["Aprime_RMS_meV_A"],
        "seed1_Aprime_slope_relative_error_abs": abs(
            seed1_result["restoring_slope_relative_error"]
        ),
        "harmonic_force_RMSE_meV_A": harmonic_result["force_error"]["RMSE_meV_A"],
        "harmonic_force_max_abs_meV_A": harmonic_result["force_error"]["max_abs_meV_A"],
        "harmonic_gate_mean": harmonic_result["pooled_gate_mean"],
        "each_harmonic_gate_mean": harmonic_result["maximum_configuration_gate_mean"],
        "finite_difference_max_abs_eV_A": finite_difference[
            "maximum_absolute_error_eV_A"
        ],
        "Hessian_antisymmetry_max_abs_eV_A2": hessian[
            "maximum_absolute_antisymmetry_eV_A2"
        ],
        "O3_energy_max_abs_eV": o3["energy_max_abs_eV"],
        "O3_force_equivariance_max_abs_eV_A": o3[
            "force_equivariance_max_abs_eV_A"
        ],
        "size_tail_force_max_abs_eV_A": size[
            "tail_force_max_abs_difference_eV_A"
        ],
        "size_tail_node_energy_max_abs_eV": size[
            "tail_node_energy_abs_difference_eV"
        ],
        "size_core_plus_tail_force_max_abs_eV_A": size[
            "core_plus_tail_force_max_abs_difference_eV_A"
        ],
    }
    gate_results = {
        name: {"observed": observed[name], "threshold": limit, "pass": observed[name] <= limit}
        for name, limit in LIMITS.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.output.parent / "evaluation_code_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    script_path = Path(__file__).resolve()
    dependency_paths = [
        script_path,
        script_path.with_name("graphene_r2m_routed_tail.py"),
        script_path.with_name("train_graphene_r2m_routed_tail_outer_fold.py"),
        script_path.with_name("graphene_r2m_aprime_eval.py"),
    ]
    for path in dependency_paths:
        shutil.copy2(path, snapshot_dir / path.name)
    arrays_path = args.output.with_suffix(".npz")
    temporary_arrays = arrays_path.with_name(arrays_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary_arrays,
        held_force_error_eV_A=held_error,
        held_tail_force_eV_A=tail_force[0],
        held_gate=gates[0],
        held_energy_error_eV=np.asarray(held_energy_error),
        support_Aprime_coordinate=np.asarray(coordinate),
        support_Aprime_predicted_force=np.asarray(predicted_mode),
        support_Aprime_target_force=np.asarray(target_mode),
        seed1_force_errors_eV_A=seed1_arrays["force_errors"],
        seed1_Aprime_coordinates=seed1_arrays["coordinates"],
        seed1_Aprime_predicted_forces=seed1_arrays["predicted_modes"],
        seed1_Aprime_target_forces=seed1_arrays["target_modes"],
        harmonic_force_errors_eV_A=np.asarray(harmonic_errors),
        harmonic_gate_means=np.asarray(harmonic_gate_means),
    )
    os.replace(temporary_arrays, arrays_path)
    summary = {
        "status": "R2M_routed_tail_outer_fold_post_freeze_evaluation_complete",
        "outer_fold": int(fold["outer_fold"]),
        "held_sscha_index": int(fold["held_sscha_index"]),
        "held_opened_only_after_epoch240_EMA_hash_freeze": True,
        "training_provenance_verified_before_held_read": True,
        "selection_policy": "fixed_epoch_240_no_support_checkpoint_selection",
        "E50_seed2_read": False,
        "full_composite_deployment_authorized": False,
        "held_support": {
            "force_error": force_metrics([held_error]),
            "centered_energy_error_meV_config": 1000.0 * held_energy_error,
            "Aprime_coordinate": {"real": float(coordinate.real), "imag": float(coordinate.imag)},
            "Aprime_predicted_force": {
                "real": float(predicted_mode.real), "imag": float(predicted_mode.imag)
            },
            "Aprime_target_force": {
                "real": float(target_mode.real), "imag": float(target_mode.imag)
            },
            "gate_mean": float(np.mean(gates[0])),
        },
        "seed1_post_freeze_preservation": seed1_result,
        "harmonic_post_freeze_preservation": harmonic_result,
        "mechanics": {
            "finite_difference": finite_difference,
            "Hessian_symmetry": hessian,
            "O3": o3,
            "size_consistency": size,
        },
        "fixed_gates": gate_results,
        "passes_fold_local_fixed_gates": bool(
            all(item["pass"] for item in gate_results.values())
        ),
        "support_mode_provenance": mode_provenance,
        "scope": {
            "foundation_and_q6_outside_tail": True,
            "local_size_gate": "frozen core plus routed tail",
            "whole_foundation_composite_theoretical_no_wrap_claimed": False,
            "q6_size_gate": "unchanged frozen Fourier audit; not rerun here",
        },
        "artifacts": {
            "core": {"path": str(args.core_model), "sha256": sha256(args.core_model)},
            "fold_manifest": {
                "path": str(fold_manifest_path),
                "sha256": sha256(fold_manifest_path),
            },
            "tail": {
                "path": str(expected_artifact),
                "sha256": sha256(expected_artifact),
            },
            "arrays": {"path": str(arrays_path), "sha256": sha256(arrays_path)},
            "training_summary": {
                "path": str(train_summary_path), "sha256": sha256(train_summary_path)
            },
            "training_freeze": {
                "path": str(training_freeze_path),
                "sha256": sha256(training_freeze_path),
            },
            "training_metrics": {
                "path": str(training_metrics_path),
                "sha256": sha256(training_metrics_path),
            },
            "TRAINING_DONE": {
                "path": str(training_done_path),
                "sha256": sha256(training_done_path),
            },
            "launcher_freeze": {
                "path": str(args.launcher_freeze),
                "sha256": sha256(args.launcher_freeze),
            },
            "code_snapshots": {
                path.name: {
                    "path": str(snapshot_dir / path.name),
                    "sha256": sha256(snapshot_dir / path.name),
                }
                for path in dependency_paths
            },
            "frozen_post_training_inputs": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in frozen_post_training_inputs.items()
            },
        },
    }
    strict_json(args.output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "mechanics"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
