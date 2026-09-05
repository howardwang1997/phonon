#!/usr/bin/env python3
"""Materialize the frozen 65-column base design on promoted seed2 geometry."""

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
from materialize_graphene_r2ac_seed2_full_bilinear import (
    DEFAULT_STRUCTURES,
    STRUCTURE_COUNT,
    read_geometry_only_extxyz,
)
from train_graphene_r2s_conditional_mlp import SEED, recommended_inputs


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_OUTPUT = BASE / "R2AC_seed2_base_design_shard_0_20260826"
BASE_WIDTH = 65


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=STRUCTURE_COUNT)
    parser.add_argument("--structures", type=Path, default=DEFAULT_STRUCTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0 <= args.start < STRUCTURE_COUNT:
        raise ValueError("start must be between 0 and 19")
    if not 1 <= args.limit <= STRUCTURE_COUNT - args.start:
        raise ValueError("limit must keep the selected range within 0..19")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    inputs = recommended_inputs()
    endpoint = load_endpoint(inputs, args.device)
    for parameter in endpoint.parameters():
        parameter.requires_grad_(False)
    endpoint.eval()
    structures = read_geometry_only_extxyz(args.structures)
    if len(structures) != STRUCTURE_COUNT or any(
        len(item) != 72 for item in structures
    ):
        raise ValueError("R2AC seed2 geometry schema changed")
    selected_indices = np.arange(args.start, args.start + args.limit, dtype=int)
    reference = read(inputs.reference_6x6, index=0)

    fixed_energy = np.empty(args.limit, dtype=np.float64)
    fixed_force = np.empty((args.limit, 72, 3), dtype=np.float64)
    parameter_energy = np.empty((args.limit, BASE_WIDTH), dtype=np.float64)
    parameter_force = np.empty(
        (args.limit, 72, 3, BASE_WIDTH), dtype=np.float64
    )
    started = time.perf_counter()
    for row, index in enumerate(selected_indices):
        query = r2r.production_linear_design_query(
            endpoint,
            structures[index],
            reference,
            device=args.device,
            create_graph=False,
        )
        fixed_energy[row] = float(query.fixed_offset_energy_eV.detach().cpu()[0])
        fixed_force[row] = (
            query.fixed_offset_force_source_order_eV_A.detach()
            .cpu()
            .numpy()
            .reshape(72, 3)
        )
        parameter_energy[row] = query.parameter_energy_design_eV.detach().cpu().numpy()
        parameter_force[row] = (
            query.parameter_force_design_source_order_eV_A.detach()
            .cpu()
            .numpy()
            .reshape(72, 3, BASE_WIDTH)
        )
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
    if r2o.state_dict_sha256(endpoint) != r2r.R2Q_ENDPOINT_STATE_SHA256:
        raise RuntimeError("R2AC base materialization changed the frozen endpoint")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "seed2_base_design.npz"
    np.savez(
        arrays_path,
        structure_indices=np.asarray(selected_indices, dtype="<i8"),
        fixed_energy_eV=np.asarray(fixed_energy, dtype="<f8"),
        fixed_force_eV_A=np.asarray(fixed_force, dtype="<f8"),
        parameter_energy_design_eV=np.asarray(parameter_energy, dtype="<f8"),
        parameter_force_design_eV_A=np.asarray(parameter_force, dtype="<f8"),
    )
    receipt = {
        "format": "graphene_r2ac_seed2_base_geometry_design_shard_v1",
        "status": (
            "R2AC_SEED2_BASE_DESIGN_COMPLETE"
            if args.start == 0 and args.limit == STRUCTURE_COUNT
            else "R2AC_SEED2_BASE_DESIGN_SHARD_COMPLETE"
        ),
        "data_role": "promoted_training_from_opened_development",
        "device": args.device,
        "structure_count": args.limit,
        "structure_indices": selected_indices.tolist(),
        "column_count": BASE_WIDTH,
        "geometry_only_materialization": True,
        "force_or_energy_labels_accessed": False,
        "input_sha256": {
            "endpoint": file_sha256(inputs.endpoint_checkpoint),
            "reference6": file_sha256(inputs.reference_6x6),
            "seed2_geometry_container": file_sha256(args.structures),
        },
        "array_raw_sha256": {
            "fixed_energy": r2r1.raw_array_sha256(fixed_energy, "<f8"),
            "fixed_force": r2r1.raw_array_sha256(fixed_force, "<f8"),
            "parameter_energy": r2r1.raw_array_sha256(parameter_energy, "<f8"),
            "parameter_force": r2r1.raw_array_sha256(parameter_force, "<f8"),
        },
        "arrays_file_sha256": file_sha256(arrays_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    (output / "DONE").write_text(receipt["status"] + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
