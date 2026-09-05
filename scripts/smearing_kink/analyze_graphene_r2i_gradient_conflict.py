#!/usr/bin/env python3
"""Audit gradient conflicts among graphene short-range development objectives.

The audited depth-3 model is loaded unchanged and only gradients with respect
to its terminal interaction/product/readout block are measured.  No optimizer
step is taken.  The output determines whether an explicit A-prime/top-component
loss is locally compatible with the harmonic replay objective before another
fine-tuning run is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.data import AtomicData, config_from_atoms
from mace.data.utils import KeySpecification
from mace.tools import AtomicNumberTable, torch_geometric


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    folded_k_aprime_mode,
    load_operator,
    structure_mapping,
)


TRAINABLE_PREFIXES = ("interactions.2.", "products.2.", "readouts.2.")
FIXED_DEGAUSS_RY = 0.0019000869380739254
SUPPORT_FORCE_SCALE_EV_A = 0.030
APRIME_SCALE_EV_A = 0.015
TOP_COMPONENT_SCALE_EV_A = 0.200
HARMONIC_SCALE_EV_A = 0.00904466916312638


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


def prepare_records(path: Path, cutoff: float) -> tuple[list, list, list[str]]:
    structures = read(path, index=":")
    key_specification = KeySpecification(
        info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
    )
    table = AtomicNumberTable([6])
    records = []
    roles = []
    for structure in structures:
        configuration = config_from_atoms(
            structure,
            key_specification=key_specification,
            config_type_weights=None,
            head_name="Default",
        )
        records.append(
            AtomicData.from_config(
                configuration, z_table=table, cutoff=cutoff, heads=["Default"]
            )
        )
        roles.append(str(structure.info.get("delta_target_role", "unknown")))
    return structures, records, roles


def flatten_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    parts = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        parts.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
    return torch.cat(parts).to(dtype=torch.float64, device="cpu")


def objective_gradient(
    model: torch.nn.Module,
    parameters: list[torch.nn.Parameter],
    records: list,
    kind: str,
    device: torch.device,
    mode_vectors: list[np.ndarray | None],
    top_k: int,
) -> tuple[float, torch.Tensor, list[dict]]:
    model.train()
    loader = torch_geometric.dataloader.DataLoader(
        records, batch_size=1, shuffle=False, drop_last=False
    )
    accumulated = torch.zeros(
        sum(parameter.numel() for parameter in parameters), dtype=torch.float64
    )
    losses = []
    per_record = []
    for index, batch in enumerate(loader):
        batch = batch.to(device)
        output = model(batch.to_dict(), training=True, compute_force=True)
        error = output["forces"] - batch.forces
        if kind == "harmonic_force":
            loss = torch.mean((error / HARMONIC_SCALE_EV_A).square())
        elif kind == "support_force":
            loss = torch.mean((error / SUPPORT_FORCE_SCALE_EV_A).square())
        elif kind == "support_topk":
            count = min(top_k, error.numel())
            values = torch.topk(error.abs().reshape(-1), k=count).values
            loss = torch.mean((values / TOP_COMPONENT_SCALE_EV_A).square())
        elif kind == "support_Aprime":
            mode = mode_vectors[index]
            if mode is None:
                raise ValueError("A-prime objective requires one mode per record")
            mode_tensor = torch.as_tensor(mode, dtype=torch.complex64, device=device)
            projection = torch.vdot(
                mode_tensor.reshape(-1), error.to(torch.complex64).reshape(-1)
            )
            loss = (projection.real.square() + projection.imag.square()) / (
                APRIME_SCALE_EV_A**2
            )
        else:
            raise ValueError(f"unknown objective {kind}")
        gradients = torch.autograd.grad(
            loss / len(records), parameters, allow_unused=True
        )
        flat = flatten_gradients(gradients, parameters)
        accumulated.add_(flat)
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        per_record.append(
            {
                "record_index": index,
                "scaled_loss": loss_value,
                "gradient_norm": float(torch.linalg.vector_norm(flat)),
                "gradient": flat,
            }
        )
        del output, error, loss, gradients, flat
    return float(np.mean(losses)), accumulated, per_record


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = torch.load(args.initial_model, map_location="cpu", weights_only=False)
    if int(model.num_interactions) != 3:
        raise ValueError("gradient audit expects the audited depth-3 model")
    model = model.to(dtype=torch.float32, device=device)
    named_parameters = list(model.named_parameters())
    for name, parameter in named_parameters:
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
    parameters = [parameter for _, parameter in named_parameters if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no terminal-block parameters are trainable")

    structures, records, roles = prepare_records(args.train_file, float(model.r_max))
    if len(records) != 81 or Counter(roles) != Counter(
        {"harmonic_replay": 72, "fixed_smearing_support_repair": 9}
    ):
        raise ValueError(f"unexpected training roles: {Counter(roles)}")
    harmonic_indices = [index for index, role in enumerate(roles) if role == "harmonic_replay"]
    support_indices = [
        index for index, role in enumerate(roles) if role == "fixed_smearing_support_repair"
    ]

    _, reference, cell, operator_degauss = load_operator(args.operator)
    if abs(operator_degauss - FIXED_DEGAUSS_RY) > 5.0e-10:
        raise ValueError("gradient audit requires the frozen T300 q6 operator")
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )
    support_modes = []
    support_ids = []
    for index in support_indices:
        mapping, _ = structure_mapping(structures[index], reference, cell)
        support_modes.append(mode[mapping])
        support_ids.append(int(structures[index].info["sscha_index"]))

    groups = {
        "harmonic_force": (
            [records[index] for index in harmonic_indices],
            [None] * len(harmonic_indices),
        ),
        "support_force": (
            [records[index] for index in support_indices],
            [None] * len(support_indices),
        ),
        "support_Aprime": (
            [records[index] for index in support_indices],
            support_modes,
        ),
        "support_topk": (
            [records[index] for index in support_indices],
            [None] * len(support_indices),
        ),
    }
    gradients = {}
    objectives = {}
    support_Aprime_records = None
    for name, (selected_records, selected_modes) in groups.items():
        print(f"computing {name} gradient", flush=True)
        loss, gradient, per_record = objective_gradient(
            model,
            parameters,
            selected_records,
            name,
            device,
            selected_modes,
            args.top_k,
        )
        gradients[name] = gradient
        objectives[name] = {
            "mean_scaled_loss": loss,
            "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        }
        if name == "support_Aprime":
            support_Aprime_records = per_record

    names = list(groups)
    cosine_matrix = {
        left: {right: cosine(gradients[left], gradients[right]) for right in names}
        for left in names
    }
    if support_Aprime_records is None:
        raise AssertionError("missing per-configuration A-prime gradients")
    per_configuration = []
    for support_id, record in zip(support_ids, support_Aprime_records, strict=True):
        per_configuration.append(
            {
                "sscha_index": support_id,
                "scaled_loss": record["scaled_loss"],
                "gradient_norm": record["gradient_norm"],
            }
        )
    pairwise = []
    for left in range(len(support_Aprime_records)):
        for right in range(left + 1, len(support_Aprime_records)):
            pairwise.append(
                {
                    "left_sscha_index": support_ids[left],
                    "right_sscha_index": support_ids[right],
                    "cosine": cosine(
                        support_Aprime_records[left]["gradient"],
                        support_Aprime_records[right]["gradient"],
                    ),
                }
            )
    finite_pairwise = [item["cosine"] for item in pairwise if np.isfinite(item["cosine"])]

    summary = {
        "status": "R2I_terminal_block_gradient_conflict_audit_complete",
        "optimizer_steps_taken": 0,
        "new_DFT_labels": 0,
        "long_range_model_modified": False,
        "fixed_condition": {
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": operator_degauss,
        },
        "scales": {
            "support_force_eV_A": SUPPORT_FORCE_SCALE_EV_A,
            "support_Aprime_eV_A": APRIME_SCALE_EV_A,
            "support_top_component_eV_A": TOP_COMPONENT_SCALE_EV_A,
            "harmonic_force_eV_A": HARMONIC_SCALE_EV_A,
            "top_k_components_per_configuration": args.top_k,
        },
        "model": {
            "path": str(args.initial_model),
            "sha256": sha256(args.initial_model),
            "trainable_prefixes": list(TRAINABLE_PREFIXES),
            "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        },
        "train_data": {"path": str(args.train_file), "sha256": sha256(args.train_file)},
        "objectives": objectives,
        "gradient_cosine": cosine_matrix,
        "support_Aprime_per_configuration": per_configuration,
        "support_Aprime_pairwise_gradient_cosine": {
            "mean": float(np.mean(finite_pairwise)),
            "minimum": float(np.min(finite_pairwise)),
            "maximum": float(np.max(finite_pairwise)),
            "pairs": pairwise,
        },
        "Aprime_mode": mode_provenance,
        "inputs": {
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {"path": str(args.background), "sha256": sha256(args.background)},
            "corrected_result": {
                "path": str(args.corrected_result),
                "sha256": sha256(args.corrected_result),
            },
        },
    }
    atomic_json(args.output, summary)
    concise = {
        "objectives": objectives,
        "gradient_cosine": cosine_matrix,
        "support_Aprime_pairwise_gradient_cosine": {
            key: summary["support_Aprime_pairwise_gradient_cosine"][key]
            for key in ("mean", "minimum", "maximum")
        },
    }
    print(json.dumps(concise, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
