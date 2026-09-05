#!/usr/bin/env python3
"""Materialize and screen fixed nonlinear node bases for R2S.

All nonlinear maps are frozen from geometry and seed 83 before force labels are
used.  Only their final conservative linear coefficients are fit.  Every new
node energy is multiplied by the multipolar gate, preserving the exact zero
2-jet/reference and rank-1 null of R2R while adding local nonlinear capacity.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read

import graphene_r2o_taylor_null as r2o
import graphene_r2r1_linear_readout as r2r1
import graphene_r2r_multipolar_background as r2r
from diagnose_graphene_r2r1_aprime_objective import (
    DEFAULT_AGGREGATE,
    DEFAULT_FIT,
    _parse_whitelist,
    canonical_json_bytes,
    file_sha256,
)
from diagnose_graphene_r2s_nonlinear_readout_basis import design_audit, nested_oof
from graphene_r2r0_formal import load_endpoint, read_geometry_only_extxyz
from train_graphene_r2s_conditional_mlp import (
    INPUT_WIDTH,
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
    / "R2R_multipolar_background/R2S_node_nonlinear_basis_diagnostic_20260826"
)
PROJECTION_MASSES = (0.0, 0.50, 0.90)


def fixed_random_weights(device: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    width1, width2 = 32, 16
    bound1 = math.sqrt(6.0 / (INPUT_WIDTH + width1))
    bound2 = math.sqrt(6.0 / (width1 + width2))
    weight1 = torch.empty((width1, INPUT_WIDTH), dtype=torch.float64)
    weight2 = torch.empty((width2, width1), dtype=torch.float64)
    weight1.uniform_(-bound1, bound1, generator=generator)
    weight2.uniform_(-bound2, bound2, generator=generator)
    return {
        "weight1": weight1.to(device),
        "weight2": weight2.to(device),
    }


def nonlinear_energy_columns(
    normalized: torch.Tensor,
    fields: Any,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, slice], list[str]]:
    signed = normalized[:, :32]
    first = signed[:, :16]
    second = signed[:, 16:32]
    gate = fields.multipolar_gate[:, None]
    amplitude = fields.amplitude_gate[:, None]
    columns = []
    labels: list[str] = []
    slices: dict[str, slice] = {}

    def append(name: str, values: torch.Tensor, names: list[str]) -> None:
        start = sum(item.shape[0] for item in columns)
        columns.append(values)
        slices[name] = slice(start, start + values.shape[0])
        labels.extend(names)

    square = torch.sum(gate * torch.square(signed), dim=0)
    append("square", square, [f"sum_a_signed2_{index:02d}" for index in range(32)])
    c_square = torch.sum(gate * amplitude * torch.square(signed), dim=0)
    append(
        "c_square",
        c_square,
        [f"sum_a_c_signed2_{index:02d}" for index in range(32)],
    )
    cross = torch.sum(gate * first * second, dim=0)
    append(
        "cross",
        cross,
        [f"sum_a_interaction_cross_{index:02d}" for index in range(16)],
    )
    c_cross = torch.sum(gate * amplitude * first * second, dim=0)
    append(
        "c_cross",
        c_cross,
        [f"sum_a_c_interaction_cross_{index:02d}" for index in range(16)],
    )
    hidden1 = torch.nn.functional.silu(normalized @ weights["weight1"].T)
    hidden2 = torch.nn.functional.silu(hidden1 @ weights["weight2"].T)
    random1 = torch.sum(gate * hidden1, dim=0)
    append(
        "random1",
        random1,
        [f"sum_a_fixed_silu1_{index:02d}" for index in range(32)],
    )
    random2 = torch.sum(gate * hidden2, dim=0)
    append(
        "random2",
        random2,
        [f"sum_a_fixed_silu2_{index:02d}" for index in range(16)],
    )
    output = torch.cat(columns)
    if output.shape != (144,) or len(labels) != 144:
        raise AssertionError("R2S nonlinear node basis width changed")
    return output, slices, labels


def materialize(
    contexts: dict[int, Any],
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    weights = fixed_random_weights(device)
    energy = np.empty((92, 144), dtype=np.float64)
    force = np.empty((92, 72, 3, 144), dtype=np.float64)
    canonical_slices = None
    canonical_labels = None
    start = time.perf_counter()
    for index in range(92):
        context = contexts[index].context
        positions = context.current_positions_reference_order
        observables = context.node_observables_fn(positions)
        fields = context.background_fields_fn(positions)
        features = _node_features(observables, fields)
        normalized = (features - mean) / scale
        columns, slices, labels = nonlinear_energy_columns(
            normalized, fields, weights
        )
        if canonical_slices is None:
            canonical_slices = slices
            canonical_labels = labels
        elif slices != canonical_slices or labels != canonical_labels:
            raise RuntimeError("R2S nonlinear column layout changed between structures")
        force_columns = []
        for column in range(columns.numel()):
            force_columns.append(
                -torch.autograd.grad(
                    columns[column],
                    positions,
                    retain_graph=column + 1 < columns.numel(),
                )[0]
            )
        reference_force = torch.stack(force_columns, dim=2)
        source_force = torch.empty_like(reference_force)
        reference_to_source = torch.as_tensor(
            context.assignment.reference_to_source,
            dtype=torch.long,
            device=reference_force.device,
        )
        source_force[reference_to_source] = reference_force
        energy[index] = columns.detach().cpu().numpy()
        force[index] = source_force.detach().cpu().numpy()
        if index in {0, 19, 55, 91}:
            print(
                json.dumps(
                    {
                        "materialized_through_global_index": index,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    assert canonical_slices is not None and canonical_labels is not None
    receipt = {
        "format": "graphene_r2s_fixed_node_nonlinear_design_v1",
        "width": 144,
        "column_labels": canonical_labels,
        "column_slices": {
            key: [value.start, value.stop] for key, value in canonical_slices.items()
        },
        "feature_scaler_scope": "all92 geometry only before label fit",
        "random_seed": SEED,
        "random_map": "FP64 Xavier-uniform 34->32->16 with SiLU; biases zero",
        "energy_raw_sha256": r2r1.raw_array_sha256(energy, "<f8"),
        "force_raw_sha256": r2r1.raw_array_sha256(force, "<f8"),
        "elapsed_seconds": time.perf_counter() - start,
    }
    return energy, force, receipt


def selected_extra_indices(receipt: dict[str, Any], kind: str) -> np.ndarray:
    slices = receipt["column_slices"]
    groups = {
        "squares64": ("square", "c_square"),
        "cross32": ("cross", "c_cross"),
        "polynomial96": ("square", "c_square", "cross", "c_cross"),
        "random48": ("random1", "random2"),
        "polynomial_plus_random144": (
            "square",
            "c_square",
            "cross",
            "c_cross",
            "random1",
            "random2",
        ),
    }
    if kind not in groups:
        raise ValueError(f"unknown R2S node basis kind: {kind}")
    values = []
    for group in groups[kind]:
        start, stop = slices[group]
        values.extend(range(start, stop))
    return np.asarray(values, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--fit-root", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

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
    contexts = _build_contexts(
        endpoint, structures, reference_atoms, np.arange(92), args.device
    )
    mean, scale, scaler_receipt = _feature_scaler(contexts, np.arange(92))
    extra_energy, extra_force, materialization_receipt = materialize(
        contexts, mean, scale, args.device
    )
    if r2o.state_dict_sha256(endpoint) != r2r.R2Q_ENDPOINT_STATE_SHA256:
        raise RuntimeError("R2S materialization changed the frozen endpoint")

    reference_force, aprime, label_hashes = _parse_whitelist(inputs.thermal92)
    if file_sha256(args.aggregate.resolve()) != r2r1.ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("attempt3 aggregate SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        base_energy = np.asarray(arrays["thermal_parameter_energy_design_eV"], np.float64)
        base_force = np.asarray(arrays["thermal_parameter_force_design_eV_A"], np.float64)
        fixed = np.asarray(arrays["thermal_fixed_force_eV_A"], np.float64)

    kinds = (
        "squares64",
        "cross32",
        "polynomial96",
        "random48",
        "polynomial_plus_random144",
    )
    results = []
    arrays_to_save: dict[str, np.ndarray] = {
        "nonlinear_energy_design": extra_energy,
        "nonlinear_force_design": extra_force,
    }
    for kind in kinds:
        selected = selected_extra_indices(materialization_receipt, kind)
        energy = np.concatenate((base_energy, extra_energy[:, selected]), axis=1)
        force = np.concatenate((base_force, extra_force[..., selected]), axis=3)
        audit = design_audit(force)
        if not audit["pass"]:
            results.append(
                {"basis": kind, "status": "DESIGN_AUDIT_FAILED", "design_audit": audit}
            )
            continue
        for projection_mass in PROJECTION_MASSES:
            oof, nested = nested_oof(
                force, fixed, reference_force, aprime, projection_mass
            )
            key = f"{kind}_projection_mass_{projection_mass:.2f}"
            arrays_to_save[key] = oof
            results.append(
                {
                    "basis": kind,
                    "projection_mass": projection_mass,
                    "column_count": int(force.shape[-1]),
                    "status": (
                        "CONDITIONAL_OOF_GATE_PASSED"
                        if nested["pooled_metrics"]["passes_fixed_gate"]
                        else "CONDITIONAL_OOF_GATE_FAILED"
                    ),
                    "design_audit": audit,
                    "nested_OOF": nested,
                    "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
                }
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "node_nonlinear_design_and_oof.npz", **arrays_to_save)
    summary = {
        "format": "graphene_r2s_fixed_node_nonlinear_basis_diagnostic_v1",
        "status": "NODE_NONLINEAR_REPRESENTATION_DIAGNOSTIC_COMPLETE_NOT_DEPLOYABLE",
        "scope": "thermal92 geometry/force whitelist only; no development or held access",
        "input_sha256": {
            "endpoint": file_sha256(inputs.endpoint_checkpoint),
            "thermal92": file_sha256(inputs.thermal92),
            "reference_6x6": file_sha256(inputs.reference_6x6),
            "aggregate_arrays": file_sha256(args.aggregate),
            "terminal_R2R1_fit_receipt": file_sha256(args.fit_root / "fit_receipt.json"),
        },
        "feature_scaler": scaler_receipt,
        "materialization": materialization_receipt,
        "label_raw_sha256": label_hashes,
        "projection_masses": list(PROJECTION_MASSES),
        "results": results,
        "interpretation_boundary": [
            "Fixed nonlinear maps are geometry-derived; only final coefficients enter OOF fits.",
            "The encoder saw related thermal data, so this remains conditional readout OOF.",
            "A passing representation still requires separate mechanics and independent-data gates.",
        ],
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    compact = [
        {
            "basis": item["basis"],
            "projection_mass": item.get("projection_mass"),
            "status": item["status"],
            "design_audit": item["design_audit"],
            "pooled_metrics": item.get("nested_OOF", {}).get("pooled_metrics"),
        }
        for item in results
    ]
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
