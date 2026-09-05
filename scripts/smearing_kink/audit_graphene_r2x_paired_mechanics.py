#!/usr/bin/env python3
"""Audit mechanics and runtime replay for the frozen R2X paired readout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

import graphene_r2r_multipolar_background as r2r
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from graphene_r2r0_formal import load_endpoint
from graphene_r2x_paired_readout import (
    load_paired_readout_checkpoint,
    production_paired_readout_energy_force,
    production_paired_readout_energy_force_hessian,
)
from train_graphene_r2s_conditional_mlp import recommended_inputs


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_FREEZE_ROOT = BASE / "R2X_paired_readout_freeze_20260826"
DEFAULT_OUTPUT = BASE / "R2X_paired_readout_mechanics_20260826"
EXPECTED_CHECKPOINT_SHA256 = (
    "0aae2ef1ee871406c347843f1c5c6155faac8df816bf9fb3addcf21bc0edcfd0"
)
FD_STEP_A = 5.0e-5
FD_COORDINATE = (2, 1)


def evaluate_ef(model, structure, reference, checkpoint, device, **kwargs):
    with torch.enable_grad():
        probe = production_paired_readout_energy_force(
            model,
            structure,
            reference,
            checkpoint,
            device=device,
            **kwargs,
        )
    return (
        float(probe.energy_eV.detach().cpu()),
        probe.force_source_order_eV_A.detach().cpu().numpy(),
    )


def evaluate_efh(model, structure, reference, checkpoint, device):
    with torch.enable_grad():
        probe = production_paired_readout_energy_force_hessian(
            model, structure, reference, checkpoint, device=device
        )
    return (
        float(probe.energy_eV.detach().cpu()),
        probe.force_source_order_eV_A.detach().cpu().numpy(),
        probe.Hessian_source_order_eV_A2.detach().cpu().numpy(),
        {
            "antisymmetry_max_abs_eV_A2": probe.Hessian_antisymmetry_max_abs_eV_A2,
            "translation_ASR_max_abs_eV_A2": (
                probe.Hessian_translation_ASR_max_abs_eV_A2
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freeze-root", type=Path, default=DEFAULT_FREEZE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    checkpoint_path = args.freeze_root / "frozen_readout.npz"
    checkpoint = load_paired_readout_checkpoint(
        checkpoint_path, expected_sha256=EXPECTED_CHECKPOINT_SHA256
    )
    inputs = recommended_inputs()
    model = load_endpoint(inputs, args.device)
    reference6 = read(inputs.reference_6x6, index=0)
    reference8 = read(inputs.reference_8x8, index=0)
    thermal = read(inputs.thermal92, index=":")
    thermal0 = thermal[0]

    reference6_energy, reference6_force, reference6_hessian, reference6_hdiag = (
        evaluate_efh(
            model, reference6, reference6, checkpoint, args.device
        )
    )
    print(json.dumps({"stage": "reference6_full_H_complete", "elapsed_seconds": time.perf_counter() - started}), flush=True)
    reference8_energy, reference8_force = evaluate_ef(
        model, reference8, reference8, checkpoint, args.device
    )
    print(json.dumps({"stage": "reference8_EF_complete", "elapsed_seconds": time.perf_counter() - started}), flush=True)
    thermal0_energy, thermal0_force, thermal0_hessian, thermal0_hdiag = evaluate_efh(
        model, thermal0, reference6, checkpoint, args.device
    )
    print(json.dumps({"stage": "thermal0_full_H_complete", "elapsed_seconds": time.perf_counter() - started}), flush=True)

    with np.load(checkpoint_path, allow_pickle=False) as arrays:
        expected_train = np.asarray(
            arrays["train_predicted_force_eV_A"], dtype=np.float64
        )
    replay_indices = (0, 19, 20, 55, 56, 91)
    replay_force_max = 0.0
    replay_records = []
    for index in replay_indices:
        energy, force = evaluate_ef(
            model, thermal[index], reference6, checkpoint, args.device
        )
        difference = float(np.max(np.abs(force - expected_train[index])))
        replay_force_max = max(replay_force_max, difference)
        replay_records.append(
            {
                "global_index": index,
                "energy_eV": energy,
                "force_replay_max_abs_eV_A": difference,
            }
        )
    print(json.dumps({"stage": "runtime_replay_complete", "elapsed_seconds": time.perf_counter() - started}), flush=True)

    generator = np.random.default_rng(83)
    proper, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(proper) < 0.0:
        proper[:, 0] *= -1.0
    improper = proper.copy()
    improper[:, 0] *= -1.0
    o3 = {}
    for name, transformation in (("proper", proper), ("improper", improper)):
        transformed_reference = reference6.copy()
        transformed_reference.positions = (
            np.asarray(reference6.positions) @ transformation.T
        )
        transformed_reference.set_cell(
            np.asarray(reference6.cell) @ transformation.T, scale_atoms=False
        )
        transformed_structure = thermal0.copy()
        transformed_structure.positions = (
            np.asarray(thermal0.positions) @ transformation.T
        )
        transformed_structure.set_cell(
            np.asarray(thermal0.cell) @ transformation.T, scale_atoms=False
        )
        energy, force = evaluate_ef(
            model,
            transformed_structure,
            transformed_reference,
            checkpoint,
            args.device,
            graph_mode="rigid_transform_probe",
            baseline_reference_template=reference6,
            baseline_structure_template=thermal0,
            rigid_transform=transformation,
        )
        o3[name] = {
            "determinant": float(np.linalg.det(transformation)),
            "energy_abs_difference_eV": abs(energy - thermal0_energy),
            "force_covariance_max_abs_difference_eV_A": float(
                np.max(np.abs(force - thermal0_force @ transformation.T))
            ),
        }
    print(json.dumps({"stage": "O3_complete", "elapsed_seconds": time.perf_counter() - started}), flush=True)

    translation = np.asarray([0.031, -0.027, 0.019], dtype=np.float64)
    translated = thermal0.copy()
    translated.positions = np.asarray(translated.positions) + translation
    translated_energy, translated_force = evaluate_ef(
        model, translated, reference6, checkpoint, args.device
    )
    permutation = generator.permutation(len(thermal0))
    permuted = thermal0[permutation]
    permuted.set_cell(thermal0.cell, scale_atoms=False)
    permuted.pbc = thermal0.pbc
    permuted_energy, permuted_force = evaluate_ef(
        model, permuted, reference6, checkpoint, args.device
    )
    wrapped = thermal0.copy()
    wrapped.positions[5] += np.asarray(wrapped.cell[0])
    wrapped_energy, wrapped_force = evaluate_ef(
        model, wrapped, reference6, checkpoint, args.device
    )

    flat_coordinate = 3 * FD_COORDINATE[0] + FD_COORDINATE[1]
    finite = {}
    endpoint_values = []
    for side, sign in (("minus", -1.0), ("plus", 1.0)):
        displaced = thermal0.copy()
        displaced.positions[FD_COORDINATE] += sign * FD_STEP_A
        energy, force = evaluate_ef(
            model, displaced, reference6, checkpoint, args.device
        )
        finite[side] = {
            "energy_eV": energy,
            "coordinate_force_eV_A": float(force[FD_COORDINATE]),
        }
        endpoint_values.append((energy, float(force[FD_COORDINATE])))
    force_from_energy = -(
        endpoint_values[1][0] - endpoint_values[0][0]
    ) / (2.0 * FD_STEP_A)
    force_derivative = (
        endpoint_values[1][1] - endpoint_values[0][1]
    ) / (2.0 * FD_STEP_A)
    finite.update(
        {
            "step_A": FD_STEP_A,
            "coordinate": list(FD_COORDINATE),
            "force_from_energy_eV_A": force_from_energy,
            "autograd_force_eV_A": float(thermal0_force[FD_COORDINATE]),
            "force_abs_difference_eV_A": abs(
                force_from_energy - float(thermal0_force[FD_COORDINATE])
            ),
            "force_derivative_eV_A2": force_derivative,
            "negative_Hessian_eV_A2": float(
                -thermal0_hessian[flat_coordinate, flat_coordinate]
            ),
            "force_Hessian_abs_difference_eV_A2": abs(
                force_derivative
                + float(thermal0_hessian[flat_coordinate, flat_coordinate])
            ),
        }
    )

    thresholds = r2r.CANONICAL_CONTRACT["fixed_gates"]
    reference_limit = thresholds["reference_Taylor_remainder"]
    checks = {
        "runtime_replay": replay_force_max <= 1.0e-10,
        "reference6_zero_2jet": (
            abs(reference6_energy) <= reference_limit["energy_abs_eV"]
            and float(np.max(np.abs(reference6_force)))
            <= reference_limit["force_max_abs_eV_A"]
            and float(np.max(np.abs(reference6_hessian)))
            <= reference_limit["Hessian_max_abs_eV_A2"]
            and reference6_hdiag["antisymmetry_max_abs_eV_A2"]
            <= reference_limit["Hessian_antisymmetry_max_abs_eV_A2"]
            and reference6_hdiag["translation_ASR_max_abs_eV_A2"]
            <= reference_limit["Hessian_translation_ASR_max_abs_eV_A2"]
        ),
        "reference8_zero_EF": (
            abs(reference8_energy) <= reference_limit["energy_abs_eV"]
            and float(np.max(np.abs(reference8_force)))
            <= reference_limit["force_max_abs_eV_A"]
        ),
        "nonreference_Hessian": (
            thermal0_hdiag["antisymmetry_max_abs_eV_A2"]
            <= thresholds["nonreference_complete_Hessian"][
                "antisymmetry_max_abs_eV_A2"
            ]
            and thermal0_hdiag["translation_ASR_max_abs_eV_A2"]
            <= thresholds["nonreference_complete_Hessian"][
                "translation_ASR_max_abs_eV_A2"
            ]
        ),
        "O3": all(
            item["energy_abs_difference_eV"]
            <= thresholds["O3_proper_and_improper"]["energy_abs_eV"]
            and item["force_covariance_max_abs_difference_eV_A"]
            <= thresholds["O3_proper_and_improper"][
                "force_covariance_max_abs_eV_A"
            ]
            for item in o3.values()
        ),
        "translation": (
            abs(translated_energy - thermal0_energy)
            <= thresholds["translation"]["energy_abs_eV"]
            and float(np.max(np.abs(translated_force - thermal0_force)))
            <= thresholds["translation"]["force_max_abs_eV_A"]
        ),
        "permutation": (
            abs(permuted_energy - thermal0_energy)
            <= thresholds["permutation"]["energy_abs_eV"]
            and float(
                np.max(np.abs(permuted_force - thermal0_force[permutation]))
            )
            <= thresholds["permutation"]["force_max_abs_eV_A"]
        ),
        "native_wrap_MIC": (
            abs(wrapped_energy - thermal0_energy)
            <= thresholds["native_cell_wrap_order_MIC_combined_probe"][
                "energy_abs_eV"
            ]
            and float(np.max(np.abs(wrapped_force - thermal0_force)))
            <= thresholds["native_cell_wrap_order_MIC_combined_probe"][
                "force_max_abs_eV_A"
            ]
        ),
        "finite_difference": (
            finite["force_abs_difference_eV_A"]
            <= thresholds["finite_difference"]["force_abs_eV_A"]
            and finite["force_Hessian_abs_difference_eV_A2"]
            <= thresholds["finite_difference"]["force_Hessian_abs_eV_A2"]
        ),
    }
    passed = all(checks.values())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "mechanics_arrays.npz"
    np.savez(
        arrays_path,
        reference6_force_eV_A=np.asarray(reference6_force, dtype="<f8"),
        reference6_Hessian_eV_A2=np.asarray(reference6_hessian, dtype="<f8"),
        reference8_force_eV_A=np.asarray(reference8_force, dtype="<f8"),
        thermal0_force_eV_A=np.asarray(thermal0_force, dtype="<f8"),
        thermal0_Hessian_eV_A2=np.asarray(thermal0_hessian, dtype="<f8"),
    )
    summary = {
        "format": "graphene_r2x_paired_readout_mechanics_v1",
        "status": (
            "R2X_PAIRED_READOUT_MECHANICS_PASSED"
            if passed
            else "R2X_PAIRED_READOUT_MECHANICS_FAILED"
        ),
        "deployable": False,
        "device": args.device,
        "checks": checks,
        "runtime_replay": {
            "indices": list(replay_indices),
            "records": replay_records,
            "force_max_abs_difference_eV_A": replay_force_max,
        },
        "reference6": {
            "energy_abs_eV": abs(reference6_energy),
            "force_max_abs_eV_A": float(np.max(np.abs(reference6_force))),
            "Hessian_max_abs_eV_A2": float(np.max(np.abs(reference6_hessian))),
            **reference6_hdiag,
        },
        "reference8": {
            "energy_abs_eV": abs(reference8_energy),
            "force_max_abs_eV_A": float(np.max(np.abs(reference8_force))),
        },
        "thermal0_complete_Hessian": thermal0_hdiag,
        "O3": o3,
        "translation": {
            "energy_abs_difference_eV": abs(translated_energy - thermal0_energy),
            "force_max_abs_difference_eV_A": float(
                np.max(np.abs(translated_force - thermal0_force))
            ),
        },
        "permutation": {
            "energy_abs_difference_eV": abs(permuted_energy - thermal0_energy),
            "force_max_abs_difference_eV_A": float(
                np.max(np.abs(permuted_force - thermal0_force[permutation]))
            ),
        },
        "native_wrap_MIC": {
            "energy_abs_difference_eV": abs(wrapped_energy - thermal0_energy),
            "force_max_abs_difference_eV_A": float(
                np.max(np.abs(wrapped_force - thermal0_force))
            ),
        },
        "finite_difference": finite,
        "input_sha256": {
            "checkpoint": file_sha256(checkpoint_path),
            "freeze_summary": file_sha256(args.freeze_root / "summary.json"),
            "endpoint": file_sha256(inputs.endpoint_checkpoint),
            "reference6": file_sha256(inputs.reference_6x6),
            "reference8": file_sha256(inputs.reference_8x8),
            "thermal92": file_sha256(inputs.thermal92),
        },
        "arrays_sha256": file_sha256(arrays_path),
        "development_or_unseen_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = output / "summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
