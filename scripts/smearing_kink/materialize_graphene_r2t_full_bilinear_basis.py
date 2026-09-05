#!/usr/bin/env python3
"""Materialize the frozen full interaction1 x interaction2 node basis.

The 16 signed scalar channels from each of the two frozen interaction blocks
form a 16 x 16 outer product at every node.  Summing that product with the
R2R multipolar gate, with and without the frozen amplitude field, gives 512
conservative scalar-energy columns.  Every column inherits the exact R2R
zero-2-jet and rank-1 null because it is multiplied by the multipolar gate.

This is a geometry-only materialization step.  It never reads force, energy,
development, support, or held labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read

import graphene_r2o_taylor_null as r2o
import graphene_r2r1_linear_readout as r2r1
import graphene_r2r_multipolar_background as r2r
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from graphene_r2r0_formal import load_endpoint, read_geometry_only_extxyz
from train_graphene_r2s_conditional_mlp import (
    SEED,
    _build_contexts,
    _feature_scaler,
    _node_features,
    recommended_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/R2T_full_bilinear_materialization_20260826"
)
CHANNEL_WIDTH = 16
PLAIN_WIDTH = CHANNEL_WIDTH * CHANNEL_WIDTH
TOTAL_WIDTH = 2 * PLAIN_WIDTH


def full_bilinear_energy_columns(
    normalized: torch.Tensor, fields: Any
) -> tuple[torch.Tensor, list[str]]:
    signed = normalized[:, : 2 * CHANNEL_WIDTH]
    first = signed[:, :CHANNEL_WIDTH]
    second = signed[:, CHANNEL_WIDTH : 2 * CHANNEL_WIDTH]
    gate = fields.multipolar_gate
    amplitude = fields.amplitude_gate
    plain = torch.einsum("n,nj,nk->jk", gate, first, second).reshape(-1)
    modulated = torch.einsum(
        "n,n,nj,nk->jk", gate, amplitude, first, second
    ).reshape(-1)
    output = torch.cat((plain, modulated))
    labels = [
        f"sum_a_interaction1_{left:02d}_interaction2_{right:02d}"
        for left in range(CHANNEL_WIDTH)
        for right in range(CHANNEL_WIDTH)
    ] + [
        f"sum_a_c_interaction1_{left:02d}_interaction2_{right:02d}"
        for left in range(CHANNEL_WIDTH)
        for right in range(CHANNEL_WIDTH)
    ]
    if output.shape != (TOTAL_WIDTH,) or len(labels) != TOTAL_WIDTH:
        raise AssertionError("R2T full bilinear column layout changed")
    return output, labels


def force_jacobian(
    columns: torch.Tensor,
    positions: torch.Tensor,
    *,
    batch_size: int,
    mode: str,
) -> torch.Tensor:
    """Return -d columns / d positions with columns on the final axis."""
    width = int(columns.numel())
    if mode == "loop":
        gradients = []
        for column in range(width):
            gradients.append(
                -torch.autograd.grad(
                    columns[column],
                    positions,
                    retain_graph=column + 1 < width,
                    create_graph=False,
                )[0]
            )
        return torch.stack(gradients, dim=2)
    if mode != "batched":
        raise ValueError(f"unknown gradient mode: {mode}")
    pieces = []
    identity = torch.eye(width, dtype=columns.dtype, device=columns.device)
    for start in range(0, width, batch_size):
        stop = min(start + batch_size, width)
        gradient = torch.autograd.grad(
            columns,
            positions,
            grad_outputs=identity[start:stop],
            retain_graph=stop < width,
            create_graph=False,
            is_grads_batched=True,
        )[0]
        pieces.append(-gradient.permute(1, 2, 0))
    return torch.cat(pieces, dim=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gradient-mode", choices=("batched", "loop"), default="batched"
    )
    parser.add_argument("--gradient-batch-size", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=92)
    parser.add_argument("--feature-scaler", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.gradient_batch_size <= 0:
        raise ValueError("gradient batch size must be positive")
    if not 0 <= args.start < 92:
        raise ValueError("start must be between 0 and 91")
    if not 1 <= args.limit <= 92 - args.start:
        raise ValueError("limit must keep the selected range within 0..91")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    inputs = recommended_inputs()
    endpoint = load_endpoint(inputs, args.device)
    for parameter in endpoint.parameters():
        parameter.requires_grad_(False)
    endpoint.eval()
    structures = read_geometry_only_extxyz(inputs.thermal92)
    reference_atoms = read(inputs.reference_6x6, index=0)
    all_indices = np.arange(92, dtype=int)
    selected_indices = np.arange(args.start, args.start + args.limit, dtype=int)
    context_indices = all_indices if args.feature_scaler is None else selected_indices
    contexts = _build_contexts(
        endpoint, structures, reference_atoms, context_indices, args.device
    )
    if args.feature_scaler is None:
        mean, scale, scaler_receipt = _feature_scaler(contexts, all_indices)
        scaler_source = "computed from all92 geometry in this run"
    else:
        with np.load(args.feature_scaler.resolve(), allow_pickle=False) as scaler:
            mean_array = np.asarray(scaler["feature_mean"], dtype=np.float64)
            scale_array = np.asarray(scaler["feature_scale"], dtype=np.float64)
        if mean_array.shape != (34,) or scale_array.shape != (34,):
            raise ValueError("frozen R2T feature scaler shape changed")
        mean = torch.as_tensor(mean_array, dtype=torch.float64, device=args.device)
        scale = torch.as_tensor(scale_array, dtype=torch.float64, device=args.device)
        scaler_receipt = {
            "sample_count": 6624,
            "feature_width": 34,
            "min_relative_RMS": float(np.min(scale_array) / np.max(scale_array)),
            "mean_raw_sha256": r2r1.raw_array_sha256(mean_array, "<f8"),
            "scale_raw_sha256": r2r1.raw_array_sha256(scale_array, "<f8"),
            "train_geometry_only": True,
        }
        scaler_source = str(args.feature_scaler.resolve())

    energy = np.empty((args.limit, TOTAL_WIDTH), dtype=np.float64)
    force = np.empty((args.limit, 72, 3, TOTAL_WIDTH), dtype=np.float64)
    labels: list[str] | None = None
    started = time.perf_counter()
    for row, index in enumerate(selected_indices):
        context = contexts[index].context
        positions = context.current_positions_reference_order
        observables = context.node_observables_fn(positions)
        fields = context.background_fields_fn(positions)
        features = _node_features(observables, fields)
        normalized = (features - mean) / scale
        columns, current_labels = full_bilinear_energy_columns(normalized, fields)
        if labels is None:
            labels = current_labels
        elif current_labels != labels:
            raise RuntimeError("R2T bilinear labels changed between structures")
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
            device=reference_force.device,
        )
        source_force[reference_to_source] = reference_force
        energy[row] = columns.detach().cpu().numpy()
        force[row] = source_force.detach().cpu().numpy()
        if row == 0 or (row + 1) % 10 == 0 or row + 1 == args.limit:
            print(
                json.dumps(
                    {
                        "materialized_count": row + 1,
                        "requested_count": args.limit,
                        "last_global_index": int(index),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    if labels is None:
        raise AssertionError("R2T materialization produced no labels")
    if r2o.state_dict_sha256(endpoint) != r2r.R2Q_ENDPOINT_STATE_SHA256:
        raise RuntimeError("R2T materialization changed the frozen endpoint")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    energy_path = output / "bilinear_energy_design_eV.npy"
    force_path = output / "bilinear_force_design_eV_A.npy"
    scaler_path = output / "feature_scaler.npz"
    np.save(energy_path, energy, allow_pickle=False)
    np.save(force_path, force, allow_pickle=False)
    np.savez(
        scaler_path,
        feature_mean=mean.detach().cpu().numpy(),
        feature_scale=scale.detach().cpu().numpy(),
    )
    receipt = {
        "format": "graphene_r2t_full_bilinear_geometry_design_v1",
        "status": (
            "FULL_GEOMETRY_DESIGN_COMPLETE"
            if args.start == 0 and args.limit == 92
            else "PARTIAL_PERFORMANCE_DIAGNOSTIC"
        ),
        "device": args.device,
        "gradient_mode": args.gradient_mode,
        "gradient_batch_size": args.gradient_batch_size,
        "structure_count": args.limit,
        "structure_indices": selected_indices.tolist(),
        "column_count": TOTAL_WIDTH,
        "column_order": "plain 16x16 row-major, then amplitude-modulated 16x16 row-major",
        "column_labels": labels,
        "feature_scaler": scaler_receipt,
        "feature_scaler_source": scaler_source,
        "label_access": False,
        "input_sha256": {
            "endpoint": file_sha256(inputs.endpoint_checkpoint),
            "thermal92_geometry": file_sha256(inputs.thermal92),
            "reference_6x6": file_sha256(inputs.reference_6x6),
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
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
