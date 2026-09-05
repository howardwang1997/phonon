#!/usr/bin/env python3
"""Materialize the frozen 99-column R2X representation on E50 seed1 geometry."""

from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read

import graphene_r2r1_linear_readout as r2r1
import graphene_r2r_multipolar_background as r2r
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from graphene_r2r0_formal import load_endpoint
from graphene_r2x_paired_readout import (
    SELECTED_BILINEAR_INDICES,
    selected_bilinear_energy_columns,
)
from train_graphene_r2s_conditional_mlp import recommended_inputs


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_STRUCTURES = (
    ROOT / "data/graphene_r2o_taylor_null_core/valid_e50_seed1.xyz"
)
DEFAULT_SCALER = BASE / "R2T_full_bilinear_shard_0_20260826/feature_scaler.npz"
DEFAULT_OUTPUT = BASE / "R2Y_seed1_selected_design_20260826"


def read_authorized_geometry_only_extxyz(path: Path) -> list[Atoms]:
    """Parse only species/positions/cell/PBC after seed1 became training data."""
    lines = Path(path).resolve(strict=True).read_text(encoding="utf-8").splitlines()
    structures: list[Atoms] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        count = int(lines[cursor].strip())
        cursor += 1
        header = shlex.split(lines[cursor], posix=True)
        cursor += 1
        fields = {}
        for token in header:
            if "=" in token:
                key, value = token.split("=", 1)
                if key in {"Properties", "Lattice", "pbc", "config_type"}:
                    fields[key] = value
        properties = fields.get("Properties", "").split(":")
        if len(properties) % 3:
            raise ValueError("R2Y extxyz Properties schema changed")
        width = 0
        species_offset = None
        position_offset = None
        for index in range(0, len(properties), 3):
            name, _kind, item_width_text = properties[index : index + 3]
            item_width = int(item_width_text)
            if name == "species":
                species_offset = width
            if name == "pos":
                position_offset = width
            width += item_width
        if species_offset is None or position_offset is None:
            raise ValueError("R2Y geometry container lacks species/pos")
        symbols = []
        positions = []
        for _ in range(count):
            tokens = lines[cursor].split()
            cursor += 1
            if len(tokens) != width:
                raise ValueError("R2Y extxyz atom row width changed")
            symbols.append(tokens[species_offset])
            positions.append(
                [float(tokens[position_offset + component]) for component in range(3)]
            )
        lattice = np.asarray(
            [float(value) for value in fields["Lattice"].split()], dtype=np.float64
        ).reshape(3, 3)
        pbc = tuple(value == "T" for value in fields.get("pbc", "F F F").split())
        atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=pbc)
        atoms.info["config_type"] = fields.get("config_type")
        structures.append(atoms)
    return structures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--structures", type=Path, default=DEFAULT_STRUCTURES)
    parser.add_argument("--feature-scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = recommended_inputs()
    model = load_endpoint(inputs, args.device)
    structures = read_authorized_geometry_only_extxyz(args.structures)
    if len(structures) != 20 or any(len(item) != 72 for item in structures):
        raise ValueError("R2Y seed1 geometry set schema changed")
    reference = read(inputs.reference_6x6, index=0)
    with np.load(args.feature_scaler.resolve(), allow_pickle=False) as arrays:
        mean_array = np.asarray(arrays["feature_mean"], dtype=np.float64)
        scale_array = np.asarray(arrays["feature_scale"], dtype=np.float64)
    if mean_array.shape != (34,) or scale_array.shape != (34,):
        raise ValueError("R2Y feature scaler schema changed")
    mean = torch.as_tensor(mean_array, dtype=torch.float64, device=args.device)
    scale = torch.as_tensor(scale_array, dtype=torch.float64, device=args.device)
    selected = torch.as_tensor(
        SELECTED_BILINEAR_INDICES, dtype=torch.long, device=args.device
    )

    fixed_energy = np.empty(20, dtype=np.float64)
    fixed_force = np.empty((20, 72, 3), dtype=np.float64)
    design_energy = np.empty((20, 99), dtype=np.float64)
    design_force = np.empty((20, 72, 3, 99), dtype=np.float64)
    for row, structure in enumerate(structures):
        context = r2r._verified_production_context(
            model, structure, reference, device=args.device
        )
        live = context.current_positions_reference_order
        fixed, base_columns = r2r._production_energy_components(context)
        aligned = live - context.image_shift.detach()
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
        bilinear = selected_bilinear_energy_columns(
            normalized,
            fields.multipolar_gate,
            fields.amplitude_gate,
            selected,
        )
        columns = torch.cat((base_columns, bilinear))
        if columns.shape != (99,):
            raise ValueError("R2Y selected energy design width changed")
        fixed_reference = -torch.autograd.grad(
            fixed, live, retain_graph=True
        )[0]
        gradients = []
        for column in range(99):
            gradients.append(
                -torch.autograd.grad(
                    columns[column], live, retain_graph=column + 1 < 99
                )[0]
            )
        force_reference = torch.stack(gradients, dim=2)
        reference_to_source = torch.as_tensor(
            context.assignment.reference_to_source,
            dtype=torch.long,
            device=live.device,
        )
        fixed_source = torch.empty_like(fixed_reference)
        force_source = torch.empty_like(force_reference)
        fixed_source[reference_to_source] = fixed_reference
        force_source[reference_to_source] = force_reference
        fixed_energy[row] = float(fixed.detach().cpu())
        fixed_force[row] = fixed_source.detach().cpu().numpy()
        design_energy[row] = columns.detach().cpu().numpy()
        design_force[row] = force_source.detach().cpu().numpy()
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

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "seed1_selected_design.npz"
    np.savez(
        arrays_path,
        fixed_energy_eV=np.asarray(fixed_energy, dtype="<f8"),
        fixed_force_eV_A=np.asarray(fixed_force, dtype="<f8"),
        parameter_energy_design_eV=np.asarray(design_energy, dtype="<f8"),
        parameter_force_design_eV_A=np.asarray(design_force, dtype="<f8"),
        selected_bilinear_indices=np.asarray(
            SELECTED_BILINEAR_INDICES, dtype="<i8"
        ),
    )
    receipt = {
        "format": "graphene_r2y_seed1_selected_geometry_design_v1",
        "status": "R2Y_SEED1_SELECTED_DESIGN_COMPLETE",
        "device": args.device,
        "structure_count": 20,
        "column_count": 99,
        "selected_bilinear_indices": SELECTED_BILINEAR_INDICES.tolist(),
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
            "fixed_energy": r2r1.raw_array_sha256(fixed_energy, "<f8"),
            "fixed_force": r2r1.raw_array_sha256(fixed_force, "<f8"),
            "parameter_energy_design": r2r1.raw_array_sha256(design_energy, "<f8"),
            "parameter_force_design": r2r1.raw_array_sha256(design_force, "<f8"),
        },
        "arrays_sha256": file_sha256(arrays_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    (output / "DONE").write_text(receipt["status"] + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
