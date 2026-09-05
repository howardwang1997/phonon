#!/usr/bin/env python3
"""Fixed R2O checkpoint gate using seed1, small harmonic and null tests only."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from graphene_r2o_taylor_null import (
    adapt_reference_cell,
    cutoff_c2_metrics,
    evaluate_structure,
    fixed_reference_graph,
    mace_interaction_energy,
    model_dtype,
    reordered_structure,
    sha256,
    solve_assignment,
    source_order_forces,
    state_dict_sha256,
    strict_json,
    taylor_remainder_energy_and_forces,
    torch_load,
    validate_mace_architecture,
)


FORMAT = "graphene_r2o_fixed_checkpoint_gate_v1"
CM1_PER_SQRT_EV_A2_AMU = 521.4708983725066


def force_metrics(errors: list[np.ndarray]) -> dict:
    joined = np.concatenate([np.asarray(value, float).reshape(-1) for value in errors])
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(np.square(joined)))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(joined))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(joined))),
        "n_force_components": int(joined.size),
    }


def restoring_slope(coordinates: np.ndarray, force: np.ndarray) -> float:
    denominator = float(np.vdot(coordinates, coordinates).real)
    if denominator <= 0.0:
        raise ValueError("A-prime coordinates are all zero")
    return -float(np.vdot(coordinates, force).real / denominator)


def predict(
    model: torch.nn.Module,
    structures: list,
    references: dict[int, object],
    device: str,
) -> list[np.ndarray]:
    values = []
    for source in structures:
        result, assignment = evaluate_structure(
            model,
            source,
            references[len(source)],
            device=device,
            create_graph=False,
        )
        values.append(
            source_order_forces(result.forces_reference_order, assignment)
            .detach()
            .cpu()
            .numpy()
        )
    return values


def endpoint_metrics(model, e50, harmonic_small, harmonic_full, references, device):
    structures = e50 + harmonic_small + harmonic_full
    predicted = predict(model, structures, references, device)
    e50_predicted = predicted[: len(e50)]
    small_predicted = predicted[len(e50) : len(e50) + len(harmonic_small)]
    full_predicted = predicted[len(e50) + len(harmonic_small) :]
    e50_errors = [
        prediction - np.asarray(item.arrays["REF_forces"], float)
        for item, prediction in zip(e50, e50_predicted, strict=True)
    ]
    small_errors = [
        prediction - np.asarray(item.arrays["REF_forces"], float)
        for item, prediction in zip(harmonic_small, small_predicted, strict=True)
    ]
    full_errors = [
        prediction - np.asarray(item.arrays["REF_forces"], float)
        for item, prediction in zip(harmonic_full, full_predicted, strict=True)
    ]
    coordinates = np.asarray(
        [
            complex(
                float(item.info["APRIME_coordinate_real_A"]),
                float(item.info["APRIME_coordinate_imag_A"]),
            )
            for item in e50
        ]
    )
    predicted_modes = []
    target_modes = []
    error_modes = []
    for item, tail, error in zip(e50, e50_predicted, e50_errors, strict=True):
        mode = np.asarray(item.arrays["APRIME_mode_real"], float) + 1.0j * np.asarray(
            item.arrays["APRIME_mode_imag"], float
        )
        baseline = np.asarray(item.arrays["FOUNDATION_BASE_forces"], float) + np.asarray(
            item.arrays["FROZEN_Q6_forces"], float
        )
        total = np.asarray(item.arrays["DFT_TOTAL_forces"], float)
        predicted_modes.append(np.vdot(mode.reshape(-1), (baseline + tail).reshape(-1)))
        target_modes.append(np.vdot(mode.reshape(-1), total.reshape(-1)))
        error_modes.append(np.vdot(mode.reshape(-1), error.reshape(-1)))
    predicted_slope = restoring_slope(coordinates, np.asarray(predicted_modes))
    target_slope = restoring_slope(coordinates, np.asarray(target_modes))
    relative_slope_error = (predicted_slope - target_slope) / target_slope
    return {
        "E50_seed1": {
            "force_error": force_metrics(e50_errors),
            "Aprime_force_error_RMS_meV_A": float(
                1000.0 * np.sqrt(np.mean(np.abs(error_modes) ** 2))
            ),
            "predicted_restoring_slope_eV_A2": predicted_slope,
            "DFT_restoring_slope_eV_A2": target_slope,
            "relative_slope_error": float(relative_slope_error),
        },
        "harmonic_lambda1_small_gate": {
            "force_error": force_metrics(small_errors),
            "per_configuration_force": [force_metrics([value]) for value in small_errors],
        },
        "harmonic_full25_report_only": {
            "used_for_checkpoint_selection": False,
            "force_error": force_metrics(full_errors),
            "per_configuration_force": [force_metrics([value]) for value in full_errors],
        },
    }


def reference_null(model, reference, device: str) -> dict:
    dtype = model_dtype(model)
    graph = fixed_reference_graph(reference, device=device, dtype=dtype)
    position = torch.as_tensor(
        np.asarray(reference.positions, float), dtype=dtype, device=device
    ).clone().requires_grad_(True)
    image = torch.zeros_like(position, dtype=torch.int64)
    result = taylor_remainder_energy_and_forces(
        model, graph, position, position.detach(), image, create_graph=True
    )
    rows = []
    flattened = result.forces_reference_order.reshape(-1)
    for value in flattened:
        rows.append(
            -torch.autograd.grad(value, position, retain_graph=True)[0].reshape(-1)
        )
    hessian = torch.stack(rows)
    hessian_numpy = hessian.detach().cpu().numpy()
    mass_weighted = 0.5 * (hessian_numpy + hessian_numpy.T) / 12.011
    maximum_absolute_eigenvalue = float(
        np.max(np.abs(np.linalg.eigvalsh(mass_weighted)))
    )
    frequency_bound = (
        CM1_PER_SQRT_EV_A2_AMU * math.sqrt(maximum_absolute_eigenvalue)
    )
    return {
        "energy_abs_eV": float(torch.abs(result.energy).max().detach().cpu()),
        "force_max_abs_eV_A": float(
            result.forces_reference_order.abs().max().detach().cpu()
        ),
        "Hessian_max_abs_eV_A2": float(np.max(np.abs(hessian_numpy))),
        "Hessian_symmetry_max_abs_eV_A2": float(
            np.max(np.abs(hessian_numpy - hessian_numpy.T))
        ),
        "Hessian_ASR_row_sum_max_abs_eV_A2": float(
            np.max(np.abs(hessian_numpy.sum(axis=1)))
        ),
        "Gamma_K_frequency_drift_upper_bound_cm-1": frequency_bound,
        "frequency_bound_definition": "Weyl bound from the largest absolute eigenvalue of the full 6x6 mass-weighted Taylor-tail Hessian; therefore bounds Gamma and folded K",
    }


def raw_energy_parity(model, structure, template, device: str) -> dict:
    reference = adapt_reference_cell(template, structure)
    assignment = solve_assignment(structure, reference)
    ordered = reordered_structure(structure, assignment)
    aligned_positions = (
        np.asarray(ordered.positions, float)
        - assignment.image_integer_reference_order @ np.asarray(ordered.cell, float)
    )
    dtype = model_dtype(model)
    graph = fixed_reference_graph(reference, device=device, dtype=dtype)

    def compare(candidate) -> dict:
        position = torch.as_tensor(
            aligned_positions, dtype=dtype, device=device
        ).clone().requires_grad_(True)
        manual = mace_interaction_energy(candidate, graph, position)
        direct_data = dict(graph)
        direct_data["positions"] = position
        direct = candidate(
            direct_data, training=False, compute_force=False
        )["interaction_energy"]
        manual_gradient = torch.autograd.grad(
            manual.sum(), position, retain_graph=True
        )[0]
        direct_gradient = torch.autograd.grad(direct.sum(), position)[0]
        scale = torch.atleast_1d(candidate.scale_shift.scale).detach().cpu().numpy()
        shift = torch.atleast_1d(candidate.scale_shift.shift).detach().cpu().numpy()
        return {
            "manual_interaction_energy_eV": float(manual.detach().cpu()),
            "ScaleShiftMACE_interaction_energy_eV": float(direct.detach().cpu()),
            "absolute_difference_eV": float(
                torch.abs(manual - direct).max().detach().cpu()
            ),
            "position_gradient_max_abs_difference_eV_A": float(
                torch.max(torch.abs(manual_gradient - direct_gradient)).detach().cpu()
            ),
            "manual_position_gradient_max_abs_eV_A": float(
                torch.max(torch.abs(manual_gradient)).detach().cpu()
            ),
            "ScaleShiftMACE_position_gradient_max_abs_eV_A": float(
                torch.max(torch.abs(direct_gradient)).detach().cpu()
            ),
            "scale": scale.tolist(),
            "shift_eV_per_atom": shift.tolist(),
        }

    actual = compare(model)
    semantic_probe = copy.deepcopy(model).to(device)
    with torch.no_grad():
        semantic_probe.scale_shift.scale.fill_(2.25)
        semantic_probe.scale_shift.shift.fill_(-0.375)
    nontrivial = compare(semantic_probe)
    return {
        "actual_checkpoint": actual,
        "nontrivial_in_memory_semantic_probe": {
            **nontrivial,
            "injected_scale": 2.25,
            "injected_shift_eV_per_atom": -0.375,
            "saved_or_used_for_prediction": False,
            "Taylor_constant_shift_cancels_exactly": True,
        },
        "optional_pair_repulsion_present": False,
        "formal_pair_repulsion_policy": "forbidden",
        "scale_shift_included": True,
    }


def implementation_checks(model, probe, references, device: str) -> dict:
    original_result, original_assignment = evaluate_structure(
        model, probe, references[len(probe)], device=device, create_graph=True
    )
    original_force_live = source_order_forces(
        original_result.forces_reference_order, original_assignment
    )
    original_force = original_force_live.detach().cpu().numpy()
    original_energy = float(original_result.energy.detach().cpu())

    # The reference Hessian is identically zero by construction, so it cannot
    # establish conservative mechanics away from x0.  Audit the complete
    # 216x216 Hessian at an ordinary E50 configuration instead.
    hessian_rows = []
    for force_component in original_result.forces_reference_order.reshape(-1):
        force_jacobian_row = torch.autograd.grad(
            force_component,
            original_result.current_positions_reference_order,
            retain_graph=True,
        )[0]
        hessian_rows.append(-force_jacobian_row.reshape(-1).detach().cpu())
    hessian = torch.stack(hessian_rows).numpy()
    atom_count = len(probe)
    translational_row_sums = hessian.reshape(
        atom_count, 3, atom_count, 3
    ).sum(axis=2)

    generator = np.random.default_rng(83)
    proper, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(proper) < 0.0:
        proper[:, 0] *= -1.0
    improper = proper.copy()
    improper[:, 0] *= -1.0
    o3 = {}
    for name, orthogonal in (("proper", proper), ("improper", improper)):
        transformed = probe.copy()
        transformed.positions = np.asarray(probe.positions) @ orthogonal.T
        transformed.set_cell(np.asarray(probe.cell) @ orthogonal.T, scale_atoms=False)
        transformed_result, transformed_assignment = evaluate_structure(
            model,
            transformed,
            references[len(probe)],
            device=device,
            create_graph=False,
        )
        transformed_force = source_order_forces(
            transformed_result.forces_reference_order, transformed_assignment
        ).detach().cpu().numpy()
        o3[name] = {
            "determinant": float(np.linalg.det(orthogonal)),
            "energy_abs_difference_eV": abs(
                float(transformed_result.energy.detach().cpu()) - original_energy
            ),
            "force_max_abs_difference_eV_A": float(
                np.max(np.abs(transformed_force - original_force @ orthogonal.T))
            ),
        }

    translation_A = np.asarray([0.031, -0.027, 0.019])
    translated = probe.copy()
    translated.positions = np.asarray(probe.positions) + translation_A
    translated_result, translated_assignment = evaluate_structure(
        model,
        translated,
        references[len(probe)],
        device=device,
        create_graph=False,
    )
    translated_force = source_order_forces(
        translated_result.forces_reference_order, translated_assignment
    ).detach().cpu().numpy()

    permutation = generator.permutation(len(probe))
    permuted = probe[permutation]
    permuted.set_cell(probe.cell, scale_atoms=False)
    permuted.pbc = probe.pbc
    permuted.positions[0] += np.asarray(permuted.cell[0])
    permuted_result, permuted_assignment = evaluate_structure(
        model, permuted, references[len(probe)], device=device, create_graph=False
    )
    permuted_force = source_order_forces(
        permuted_result.forces_reference_order, permuted_assignment
    ).detach().cpu().numpy()
    expected_permuted_force = original_force[permutation]

    coordinate = (2, 1)
    step = 1.0e-4
    displaced_energies = []
    displaced_force_components = []
    for sign in (-1.0, 1.0):
        displaced = probe.copy()
        displaced.positions[coordinate] += sign * step
        value, displaced_assignment = evaluate_structure(
            model, displaced, references[len(probe)], device=device, create_graph=False
        )
        displaced_energies.append(float(value.energy.detach().cpu()))
        displaced_force = source_order_forces(
            value.forces_reference_order, displaced_assignment
        )
        displaced_force_components.append(
            float(displaced_force[coordinate].detach().cpu())
        )
    fd_force = -(displaced_energies[1] - displaced_energies[0]) / (2.0 * step)
    reference_atom = int(original_assignment.source_to_reference[coordinate[0]])
    analytic_force_jacobian = torch.autograd.grad(
        original_force_live[coordinate],
        original_result.current_positions_reference_order,
    )[0][reference_atom, coordinate[1]]
    fd_force_jacobian = (
        displaced_force_components[1] - displaced_force_components[0]
    ) / (2.0 * step)

    localized = {}
    for count, template in references.items():
        item = template.copy()
        item.positions[0, 2] += 0.03
        value, assignment = evaluate_structure(
            model, item, template, device=device, create_graph=False
        )
        forces = source_order_forces(value.forces_reference_order, assignment)
        localized[count] = {
            "energy_eV": float(value.energy.detach().cpu()),
            "central_force_eV_A": forces[0].detach().cpu().numpy(),
        }
    size_energy_difference = abs(localized[72]["energy_eV"] - localized[128]["energy_eV"])
    size_force_difference = float(
        np.max(
            np.abs(
                localized[72]["central_force_eV_A"]
                - localized[128]["central_force_eV_A"]
            )
        )
    )
    return {
        "O3": o3,
        "global_translation": {
            "translation_A": translation_A.tolist(),
            "energy_abs_difference_eV": abs(
                float(translated_result.energy.detach().cpu()) - original_energy
            ),
            "force_max_abs_difference_eV_A": float(
                np.max(np.abs(translated_force - original_force))
            ),
        },
        "permutation_image_energy_abs_difference_eV": abs(
            float(permuted_result.energy.detach().cpu())
            - original_energy
        ),
        "permutation_image_force_max_abs_difference_eV_A": float(
            np.max(np.abs(permuted_force - expected_permuted_force))
        ),
        "finite_difference": {
            "coordinate": list(coordinate),
            "step_A": step,
            "analytic_force_eV_A": float(original_force[coordinate]),
            "finite_difference_force_eV_A": fd_force,
            "absolute_difference_eV_A": abs(fd_force - float(original_force[coordinate])),
        },
        "force_Hessian_finite_difference": {
            "force_coordinate": list(coordinate),
            "position_coordinate": list(coordinate),
            "step_A": step,
            "analytic_dF_dx_eV_A2": float(analytic_force_jacobian.detach().cpu()),
            "finite_difference_dF_dx_eV_A2": fd_force_jacobian,
            "absolute_difference_eV_A2": abs(
                fd_force_jacobian - float(analytic_force_jacobian.detach().cpu())
            ),
        },
        "nonreference_complete_Hessian": {
            "shape": list(hessian.shape),
            "antisymmetry_max_abs_eV_A2": float(
                np.max(np.abs(hessian - hessian.T))
            ),
            "translation_ASR_max_abs_eV_A2": float(
                np.max(np.abs(translational_row_sums))
            ),
        },
        "size_6x6_8x8": {
            "localized_displacement_A": 0.03,
            "energy_abs_difference_eV": size_energy_difference,
            "central_force_max_abs_difference_eV_A": size_force_difference,
        },
    }


def materialize_candidate(template, checkpoint_path: Path, expected_contract: dict, epoch: int):
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != "graphene_r2o_whole_energy_taylor2_training_v1":
        raise ValueError("wrong R2O checkpoint format")
    if checkpoint.get("contract") != expected_contract:
        raise ValueError("R2O candidate checkpoint belongs to another frozen run")
    if int(checkpoint.get("epoch", -1)) != int(epoch):
        raise ValueError("R2O candidate checkpoint epoch/label mismatch")
    model = copy.deepcopy(template).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    state = model.state_dict()
    shadow = checkpoint["ema_state_dict"]["shadow"]
    parameter_names = {name for name, _ in model.named_parameters()}
    if set(shadow) != parameter_names:
        raise ValueError("EMA checkpoint parameter set changed")
    for name, value in shadow.items():
        state[name] = value.detach().cpu()
    model.load_state_dict(state, strict=True)
    return model


def save_bundle(path: Path, model, metadata: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "format": "graphene_r2o_deployment_bundle_v1",
            "raw_MACE_must_not_be_deployed_without_Taylor_wrapper": True,
            "model": model.cpu(),
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-bundle", type=Path, required=True)
    parser.add_argument("--wrapper-snapshot", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    for path in (
        args.data,
        args.training_dir,
        args.output,
        args.selected_bundle,
        args.wrapper_snapshot,
    ):
        if any(token in str(path).lower() for token in ("seed2", "reserved", "support")):
            raise ValueError(f"R2O evaluator rejects forbidden path {path}")
    if not args.wrapper_snapshot.is_file():
        raise FileNotFoundError(args.wrapper_snapshot)
    imported_wrapper = Path(__file__).with_name("graphene_r2o_taylor_null.py")
    if sha256(args.wrapper_snapshot) != sha256(imported_wrapper):
        raise ValueError("executed Taylor wrapper differs from the frozen snapshot")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_R2O_training":
        raise ValueError("wrong R2O data status")
    if manifest["counts"]["seed2_opened"] != 0 or manifest["counts"]["support_opened"] != 0:
        raise ValueError("R2O forbidden split count changed")
    for name, record in manifest["outputs"].items():
        path = args.data / name
        if name == "prepare_script_snapshot.py":
            path = args.data / name
        if sha256(path) != record["sha256"]:
            raise ValueError(f"R2O frozen data hash changed: {name}")
    stage2_contract = json.loads((args.training_dir / "stage2_contract.json").read_text())
    runtime = json.loads((args.training_dir / "stage2_runtime.json").read_text())
    training_freeze_path = args.training_dir.parent / "training_freeze.json"
    if not training_freeze_path.is_file():
        raise FileNotFoundError(training_freeze_path)
    training_freeze = json.loads(training_freeze_path.read_text())
    if stage2_contract.get("launcher_freeze_sha256") != sha256(training_freeze_path):
        raise ValueError("stage2 contract is not bound to this training freeze")
    if training_freeze.get("data_manifest_sha256") != sha256(manifest_path):
        raise ValueError("training freeze is not bound to this data manifest")
    if training_freeze.get("source_sha256", {}).get("wrapper") != sha256(
        args.wrapper_snapshot
    ):
        raise ValueError("training freeze wrapper hash changed")
    if runtime.get("status") != "R2O_stage2_training_complete_pending_fixed_gate":
        raise ValueError("R2O stage2 is not ready for evaluation")
    if stage2_contract.get("selection_data_read") is not False:
        raise ValueError("selection data entered training")
    if stage2_contract.get("seed1_seed2_support_read") is not False:
        raise ValueError("forbidden data entered training")
    if runtime.get("seed1_seed2_support_read") is not False:
        raise ValueError("training runtime leakage flag changed")
    if runtime.get("stage2_contract_sha256") != sha256(
        args.training_dir / "stage2_contract.json"
    ):
        raise ValueError("training runtime is not bound to this stage2 contract")

    raw_path = Path(runtime["raw_model"]["path"])
    if not raw_path.is_absolute():
        raw_path = args.training_dir / raw_path.name
    if sha256(raw_path) != runtime["raw_model"]["sha256"]:
        raise ValueError("R2O final raw model hash changed")
    template = torch_load(raw_path, map_location="cpu")
    validate_mace_architecture(template)
    if runtime.get("raw_state_sha256") != state_dict_sha256(template):
        raise ValueError("R2O raw model semantic state hash changed")
    checkpoint_epochs = list(stage2_contract["checkpoint_epochs"])
    runtime_checkpoints = runtime.get("fixed_checkpoints", {})
    if set(runtime_checkpoints) != {str(epoch) for epoch in checkpoint_epochs}:
        raise ValueError("training runtime fixed-checkpoint set changed")
    for epoch in checkpoint_epochs:
        expected_path = args.training_dir / "checkpoints" / f"r2o_epoch-{epoch}.pt"
        record = runtime_checkpoints[str(epoch)]
        recorded_path = Path(record["path"])
        if not recorded_path.is_absolute():
            recorded_path = args.training_dir / "checkpoints" / recorded_path.name
        if recorded_path.resolve() != expected_path.resolve():
            raise ValueError("training runtime fixed-checkpoint path changed")
        if sha256(expected_path) != record["sha256"]:
            raise ValueError("training runtime fixed-checkpoint hash changed")
    candidate_epochs = [checkpoint_epochs[-1]] if args.smoke else checkpoint_epochs
    candidates = {
        f"epoch{epoch}": args.training_dir / "checkpoints" / f"r2o_epoch-{epoch}.pt"
        for epoch in candidate_epochs
    }
    if any(not path.is_file() for path in candidates.values()):
        raise FileNotFoundError("one or more fixed R2O checkpoints are missing")

    e50 = read(args.data / "valid_e50_seed1.xyz", index=":")
    harmonic_small = read(args.data / "harmonic_lambda1_small_gate.xyz", index=":")
    harmonic_full = read(args.data / "harmonic_full_report.xyz", index=":")
    reference6 = read(args.data / "reference_6x6.xyz", index=0)
    reference8 = read(args.data / "reference_8x8.xyz", index=0)
    references = {72: reference6, 128: reference8}
    if len(e50) != 20 or len(harmonic_full) != 25:
        raise ValueError("R2O fixed evaluation counts changed")
    if len(harmonic_small) != manifest["counts"]["harmonic_lambda1_small_gate"]:
        raise ValueError("R2O small harmonic gate count changed")

    thresholds = manifest["fixed_endpoint_thresholds"]
    records = []
    models = {}
    for label, path in candidates.items():
        epoch = int(label.removeprefix("epoch"))
        model = materialize_candidate(
            template, path, stage2_contract, epoch
        ).to(args.device)
        endpoint = endpoint_metrics(
            model, e50, harmonic_small, harmonic_full, references, args.device
        )
        null = reference_null(model, reference6, args.device)
        observed = {
            "e50_seed1_force_RMSE_meV_A": endpoint["E50_seed1"]["force_error"]["RMSE_meV_A"],
            "e50_seed1_force_max_abs_meV_A": endpoint["E50_seed1"]["force_error"]["max_abs_meV_A"],
            "e50_seed1_Aprime_RMS_meV_A": endpoint["E50_seed1"]["Aprime_force_error_RMS_meV_A"],
            "e50_seed1_Aprime_slope_relative_error_abs": abs(endpoint["E50_seed1"]["relative_slope_error"]),
            "harmonic_small_force_RMSE_meV_A": endpoint["harmonic_lambda1_small_gate"]["force_error"]["RMSE_meV_A"],
            "harmonic_small_force_max_abs_meV_A": endpoint["harmonic_lambda1_small_gate"]["force_error"]["max_abs_meV_A"],
            "Taylor_remainder_reference_E_abs_eV": null["energy_abs_eV"],
            "Taylor_remainder_reference_force_max_abs_eV_A": null["force_max_abs_eV_A"],
            "Taylor_remainder_reference_Hessian_max_abs_eV_A2": null["Hessian_max_abs_eV_A2"],
            "Taylor_remainder_reference_Hessian_ASR_row_sum_max_abs_eV_A2": null[
                "Hessian_ASR_row_sum_max_abs_eV_A2"
            ],
            "Gamma_K_frequency_drift_cm-1": null["Gamma_K_frequency_drift_upper_bound_cm-1"],
        }
        ratios = {name: observed[name] / float(thresholds[name]) for name in observed}
        passed = all(value <= 1.0 for value in ratios.values())
        records.append(
            {
                "label": label,
                "path": str(path),
                "sha256": sha256(path),
                "model_state_sha256": state_dict_sha256(model),
                "passes_fixed_gate": bool(passed),
                "normalized_gate_ratios": ratios,
                "normalized_worst_gate_score": float(max(ratios.values())),
                **endpoint,
                "reference_null": null,
            }
        )
        models[label] = model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    choice = min(records, key=lambda record: record["normalized_worst_gate_score"])
    chosen_model = models[choice["label"]]
    implementation = implementation_checks(
        chosen_model.to(args.device), e50[0], references, args.device
    )
    parity = raw_energy_parity(
        chosen_model, e50[0], reference6, args.device
    )
    cutoff = cutoff_c2_metrics(chosen_model)
    implementation_limits = {
        "actual_raw_energy_parity_eV": 1.0e-10,
        "actual_raw_position_gradient_parity_eV_A": 1.0e-9,
        "nontrivial_probe_raw_energy_parity_eV": 1.0e-10,
        "nontrivial_probe_raw_position_gradient_parity_eV_A": 1.0e-9,
        "O3_proper_energy_eV": 1.0e-6,
        "O3_proper_force_eV_A": 1.0e-5,
        "O3_improper_energy_eV": 1.0e-6,
        "O3_improper_force_eV_A": 1.0e-5,
        "translation_energy_eV": 1.0e-9,
        "translation_force_eV_A": 1.0e-8,
        "permutation_energy_eV": 1.0e-6,
        "permutation_force_eV_A": 1.0e-5,
        "finite_difference_force_eV_A": 1.0e-5,
        "force_Hessian_finite_difference_eV_A2": 1.0e-5,
        "nonreference_Hessian_antisymmetry_eV_A2": 1.0e-7,
        "nonreference_Hessian_translation_ASR_eV_A2": 1.0e-7,
        "size_energy_eV": 1.0e-7,
        "size_force_eV_A": 1.0e-5,
    }
    implementation_observed = {
        "actual_raw_energy_parity_eV": parity["actual_checkpoint"][
            "absolute_difference_eV"
        ],
        "actual_raw_position_gradient_parity_eV_A": parity["actual_checkpoint"][
            "position_gradient_max_abs_difference_eV_A"
        ],
        "nontrivial_probe_raw_energy_parity_eV": parity[
            "nontrivial_in_memory_semantic_probe"
        ]["absolute_difference_eV"],
        "nontrivial_probe_raw_position_gradient_parity_eV_A": parity[
            "nontrivial_in_memory_semantic_probe"
        ]["position_gradient_max_abs_difference_eV_A"],
        "O3_proper_energy_eV": implementation["O3"]["proper"][
            "energy_abs_difference_eV"
        ],
        "O3_proper_force_eV_A": implementation["O3"]["proper"][
            "force_max_abs_difference_eV_A"
        ],
        "O3_improper_energy_eV": implementation["O3"]["improper"][
            "energy_abs_difference_eV"
        ],
        "O3_improper_force_eV_A": implementation["O3"]["improper"][
            "force_max_abs_difference_eV_A"
        ],
        "translation_energy_eV": implementation["global_translation"][
            "energy_abs_difference_eV"
        ],
        "translation_force_eV_A": implementation["global_translation"][
            "force_max_abs_difference_eV_A"
        ],
        "permutation_energy_eV": implementation["permutation_image_energy_abs_difference_eV"],
        "permutation_force_eV_A": implementation["permutation_image_force_max_abs_difference_eV_A"],
        "finite_difference_force_eV_A": implementation["finite_difference"]["absolute_difference_eV_A"],
        "force_Hessian_finite_difference_eV_A2": implementation[
            "force_Hessian_finite_difference"
        ]["absolute_difference_eV_A2"],
        "nonreference_Hessian_antisymmetry_eV_A2": implementation[
            "nonreference_complete_Hessian"
        ]["antisymmetry_max_abs_eV_A2"],
        "nonreference_Hessian_translation_ASR_eV_A2": implementation[
            "nonreference_complete_Hessian"
        ]["translation_ASR_max_abs_eV_A2"],
        "size_energy_eV": implementation["size_6x6_8x8"]["energy_abs_difference_eV"],
        "size_force_eV_A": implementation["size_6x6_8x8"]["central_force_max_abs_difference_eV_A"],
    }
    implementation_pass = all(
        implementation_observed[name] <= implementation_limits[name]
        for name in implementation_limits
    )
    cutoff_pass = (
        abs(cutoff["value"]) <= 1.0e-12
        and abs(cutoff["first_derivative_A-1"]) <= 1.0e-11
        and abs(cutoff["second_derivative_A-2"]) <= 1.0e-10
        and abs(cutoff["third_derivative_A-3"]) > 1.0e-6
    )
    null_keys = (
        "Taylor_remainder_reference_E_abs_eV",
        "Taylor_remainder_reference_force_max_abs_eV_A",
        "Taylor_remainder_reference_Hessian_max_abs_eV_A2",
        "Taylor_remainder_reference_Hessian_ASR_row_sum_max_abs_eV_A2",
        "Gamma_K_frequency_drift_cm-1",
    )
    null_gate_ratios = {
        name: choice["normalized_gate_ratios"][name] for name in null_keys
    }
    null_pass = all(value <= 1.0 for value in null_gate_ratios.values())
    if args.smoke:
        authorized = bool(implementation_pass and cutoff_pass and null_pass)
        status = "R2O_two_epoch_smoke_passed" if authorized else "R2O_two_epoch_smoke_failed"
    else:
        authorized = bool(
            choice["passes_fixed_gate"]
            and implementation_pass
            and cutoff_pass
            and null_pass
        )
        status = "R2O_core_checkpoint_gate_passed" if authorized else "R2O_core_checkpoint_gate_failed"
    bundle_kind = (
        "selected_R2O_Taylor_bundle_for_postcore_validation"
        if authorized and not args.smoke
        else "R2O_diagnostic_bundle_not_authorized_for_deployment"
    )
    metadata = {
        "kind": bundle_kind,
        "model_state_sha256": state_dict_sha256(chosen_model),
        "wrapper_snapshot": str(args.wrapper_snapshot),
        "wrapper_sha256": sha256(args.wrapper_snapshot),
        "data_manifest_sha256": sha256(manifest_path),
        "reference_6x6_sha256": sha256(args.data / "reference_6x6.xyz"),
        "reference_8x8_sha256": sha256(args.data / "reference_8x8.xyz"),
        "raw_MACE_direct_deployment_forbidden": True,
        "evaluation_recovery": "bundle is deterministically rematerialized from the contract-bound chosen checkpoint and atomically replaced before the gate JSON is published",
    }
    save_bundle(args.selected_bundle, chosen_model, metadata)
    payload = {
        "format": FORMAT,
        "status": status,
        "smoke": bool(args.smoke),
        "inputs": {
            "data_manifest_sha256": sha256(manifest_path),
            "training_freeze_sha256": sha256(training_freeze_path),
            "stage2_contract_sha256": sha256(
                args.training_dir / "stage2_contract.json"
            ),
            "stage2_runtime_sha256": sha256(
                args.training_dir / "stage2_runtime.json"
            ),
            "source_sha256": training_freeze["source_sha256"],
        },
        "postcore_or_deployment_authorized": bool(authorized and not args.smoke),
        "selection_scope": "E50 seed1 + bond-RMS<=0.003 A small harmonic + exact reference null; full25 harmonic is report-only",
        "seed2_or_support_read": False,
        "fixed_thresholds": thresholds,
        "records": records,
        "checkpoint_choice": choice,
        "implementation_checks": implementation,
        "raw_energy_parity": parity,
        "cutoff_C2_inside_limit": cutoff,
        "implementation_limits": implementation_limits,
        "implementation_observed": implementation_observed,
        "implementation_pass": implementation_pass,
        "cutoff_C2_pass": cutoff_pass,
        "reference_null_gate_ratios": null_gate_ratios,
        "reference_null_pass": null_pass,
        "bundle": {
            "kind": bundle_kind,
            "path": str(args.selected_bundle),
            "sha256": sha256(args.selected_bundle),
            **metadata,
        },
    }
    strict_json(args.output, payload)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
