#!/usr/bin/env python3
"""Materialize the frozen 512-column bilinear basis on seed1 geometry only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

import graphene_r2o_taylor_null as r2o
import graphene_r2r1_linear_readout as r2r1
import graphene_r2r_multipolar_background as r2r
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from graphene_r2r0_formal import load_endpoint
from materialize_graphene_r2t_full_bilinear_basis import (
    TOTAL_WIDTH,
    force_jacobian,
    full_bilinear_energy_columns,
)
from materialize_graphene_r2y_seed1_design import (
    DEFAULT_STRUCTURES,
    read_authorized_geometry_only_extxyz,
)
from train_graphene_r2s_conditional_mlp import SEED, recommended_inputs


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_SCALER = BASE / "R2T_full_bilinear_shard_0_20260826/feature_scaler.npz"
DEFAULT_OUTPUT = BASE / "R2Y_seed1_full_bilinear_20260826"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gradient-mode", choices=("batched", "loop"), default="batched"
    )
    parser.add_argument("--gradient-batch-size", type=int, default=64)
    parser.add_argument("--structures", type=Path, default=DEFAULT_STRUCTURES)
    parser.add_argument("--feature-scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.gradient_batch_size <= 0:
        raise ValueError("gradient batch size must be positive")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    inputs = recommended_inputs()
    endpoint = load_endpoint(inputs, args.device)
    for parameter in endpoint.parameters():
        parameter.requires_grad_(False)
    endpoint.eval()
    structures = read_authorized_geometry_only_extxyz(args.structures)
    if len(structures) != 20 or any(len(item) != 72 for item in structures):
        raise ValueError("R2Y seed1 geometry schema changed")
    reference = read(inputs.reference_6x6, index=0)
    with np.load(args.feature_scaler.resolve(), allow_pickle=False) as arrays:
        mean_array = np.asarray(arrays["feature_mean"], dtype=np.float64)
        scale_array = np.asarray(arrays["feature_scale"], dtype=np.float64)
    if mean_array.shape != (34,) or scale_array.shape != (34,):
        raise ValueError("R2Y frozen feature scaler schema changed")
    mean = torch.as_tensor(mean_array, dtype=torch.float64, device=args.device)
    scale = torch.as_tensor(scale_array, dtype=torch.float64, device=args.device)

    energy = np.empty((20, TOTAL_WIDTH), dtype=np.float64)
    force = np.empty((20, 72, 3, TOTAL_WIDTH), dtype=np.float64)
    labels: list[str] | None = None
    started = time.perf_counter()
    for row, structure in enumerate(structures):
        context = r2r._verified_production_context(
            endpoint, structure, reference, device=args.device
        )
        positions = context.current_positions_reference_order
        aligned = positions - context.image_shift.detach()
        observables = context.node_observables_fn(aligned)
        fields = context.background_fields_fn(aligned)
        features = torch.cat(
            (
                observables.signed_l0,
                observables.scale_shift_node_energy_eV[:, None],
                fields.amplitude_gate[:, None],
            ),
            dim=1,
        )
        normalized = (features - mean) / scale
        columns, current_labels = full_bilinear_energy_columns(normalized, fields)
        if labels is None:
            labels = current_labels
        elif current_labels != labels:
            raise RuntimeError("R2Y bilinear labels changed between structures")
        reference_force = force_jacobian(
            columns,
            positions,
            batch_size=args.gradient_batch_size,
            mode=args.gradient_mode,
        )
        source_force = torch.empty_like(reference_force)
        reference_to_source = torch.as_tensor(
            context.assignment.reference_to_source,
            dtype=torch.long,
            device=positions.device,
        )
        source_force[reference_to_source] = reference_force
        energy[row] = columns.detach().cpu().numpy()
        force[row] = source_force.detach().cpu().numpy()
        print(
            json.dumps(
                {
                    "materialized_count": row + 1,
                    "requested_count": 20,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    if labels is None:
        raise RuntimeError("R2Y full bilinear materialization produced no columns")
    if r2o.state_dict_sha256(endpoint) != r2r.R2Q_ENDPOINT_STATE_SHA256:
        raise RuntimeError("R2Y materialization changed the frozen endpoint")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    energy_path = output / "bilinear_energy_design_eV.npy"
    force_path = output / "bilinear_force_design_eV_A.npy"
    np.save(energy_path, np.asarray(energy, dtype="<f8"), allow_pickle=False)
    np.save(force_path, np.asarray(force, dtype="<f8"), allow_pickle=False)
    receipt = {
        "format": "graphene_r2y_seed1_full_bilinear_geometry_design_v1",
        "status": "R2Y_SEED1_FULL_BILINEAR_COMPLETE",
        "device": args.device,
        "gradient_mode": args.gradient_mode,
        "gradient_batch_size": args.gradient_batch_size,
        "structure_count": 20,
        "column_count": TOTAL_WIDTH,
        "column_order": "plain 16x16 row-major, then amplitude-modulated 16x16 row-major",
        "column_labels": labels,
        "geometry_only_materialization": True,
        "force_or_energy_labels_accessed": False,
        "feature_scaler_refit": False,
        "input_sha256": {
            "endpoint": file_sha256(inputs.endpoint_checkpoint),
            "reference6": file_sha256(inputs.reference_6x6),
            "seed1_geometry_container": file_sha256(args.structures),
            "feature_scaler": file_sha256(args.feature_scaler),
        },
        "array_raw_sha256": {
            "energy": r2r1.raw_array_sha256(energy, "<f8"),
            "force": r2r1.raw_array_sha256(force, "<f8"),
        },
        "array_file_sha256": {
            "energy": file_sha256(energy_path),
            "force": file_sha256(force_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    (output / "DONE").write_text(receipt["status"] + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
