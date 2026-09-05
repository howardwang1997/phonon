#!/usr/bin/env python3
"""Fine-tune graphene MACE blocks with gate-balanced common descent.

Each optimizer step pairs one fixed-smearing support configuration with eight
harmonic replay configurations.  Harmonic and support gradients are normalized
separately and combined along their equal-angle common descent direction.  The
support direction itself balances component RMSE, folded-K A-prime projection,
and the largest force components.  The model remains a scalar-energy MACE; the
procedure changes the training objective, not the force definition.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.tools import torch_geometric


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    folded_k_aprime_mode,
    load_operator,
    structure_mapping,
)
from train_graphene_r2f_last_block import (  # noqa: E402
    EMA,
    atomic_json,
    atomic_model,
    dataset,
    force_metrics,
    sha256,
)


SCOPES = {
    "last_block": ("interactions.2.", "products.2.", "readouts.2."),
    "last_two_blocks": (
        "interactions.1.",
        "interactions.2.",
        "products.1.",
        "products.2.",
        "readouts.1.",
        "readouts.2.",
    ),
}
FIXED_DEGAUSS_RY = 0.0019000869380739254
SUPPORT_FORCE_SCALE_EV_A = 0.030
APRIME_SCALE_EV_A = 0.015
TOP_COMPONENT_SCALE_EV_A = 0.200
HARMONIC_SCALE_EV_A = 0.00904466916312638


Gradient = tuple[torch.Tensor, ...]


def batches(records: list, device: torch.device) -> list:
    result = []
    for record in records:
        loader = torch_geometric.dataloader.DataLoader(
            [record], batch_size=1, shuffle=False, drop_last=False
        )
        result.append(next(iter(loader)).to(device))
    return result


def gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool = False,
) -> Gradient:
    values = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter) if value is None else value.detach()
        for value, parameter in zip(values, parameters, strict=True)
    )


def zero_gradient(parameters: list[torch.nn.Parameter]) -> Gradient:
    return tuple(torch.zeros_like(parameter) for parameter in parameters)


def add(left: Gradient, right: Gradient) -> Gradient:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def scale(values: Gradient, factor: float | torch.Tensor) -> Gradient:
    return tuple(value * factor for value in values)


def norm(values: Gradient) -> torch.Tensor:
    return torch.sqrt(sum(torch.sum(value.square()) for value in values))


def unit(values: Gradient) -> Gradient:
    magnitude = norm(values)
    if float(magnitude.detach().cpu()) <= 1.0e-20:
        raise RuntimeError("encountered a zero objective gradient")
    return scale(values, 1.0 / magnitude)


def dot(left: Gradient, right: Gradient) -> torch.Tensor:
    return sum(torch.sum(a * b) for a, b in zip(left, right, strict=True))


def harmonic_gradient(
    model: torch.nn.Module,
    parameters: list[torch.nn.Parameter],
    selected_batches: list,
) -> tuple[Gradient, float]:
    total = zero_gradient(parameters)
    losses = []
    for batch in selected_batches:
        output = model(batch.to_dict(), training=True, compute_force=True)
        error = output["forces"] - batch.forces
        loss = torch.mean((error / HARMONIC_SCALE_EV_A).square())
        total = add(total, gradients(loss / len(selected_batches), parameters))
        losses.append(float(loss.detach().cpu()))
    return total, float(np.mean(losses))


def support_gradient(
    model: torch.nn.Module,
    parameters: list[torch.nn.Parameter],
    batch,
    mode: torch.Tensor,
    top_k: int,
) -> tuple[Gradient, dict]:
    output = model(batch.to_dict(), training=True, compute_force=True)
    error = output["forces"] - batch.forces
    force_loss = torch.mean((error / SUPPORT_FORCE_SCALE_EV_A).square())
    projection = torch.vdot(
        mode.reshape(-1), error.to(torch.complex64).reshape(-1)
    )
    Aprime_loss = (projection.real.square() + projection.imag.square()) / (
        APRIME_SCALE_EV_A**2
    )
    count = min(top_k, error.numel())
    largest = torch.topk(error.abs().reshape(-1), k=count).values
    topk_loss = torch.mean((largest / TOP_COMPONENT_SCALE_EV_A).square())

    force_gradient = unit(gradients(force_loss, parameters, retain_graph=True))
    Aprime_gradient = unit(gradients(Aprime_loss, parameters, retain_graph=True))
    topk_gradient = unit(gradients(topk_loss, parameters))
    balanced = unit(add(add(force_gradient, Aprime_gradient), topk_gradient))
    return balanced, {
        "support_force_scaled_loss": float(force_loss.detach().cpu()),
        "support_Aprime_scaled_loss": float(Aprime_loss.detach().cpu()),
        "support_topk_scaled_loss": float(topk_loss.detach().cpu()),
        "support_internal_cosines": {
            "force_Aprime": float(dot(force_gradient, Aprime_gradient).detach().cpu()),
            "force_topk": float(dot(force_gradient, topk_gradient).detach().cpu()),
            "Aprime_topk": float(dot(Aprime_gradient, topk_gradient).detach().cpu()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", choices=sorted(SCOPES), required=True)
    parser.add_argument("--max-epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-8)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument(
        "--checkpoint-epochs", default="1,5,10,20,40,80,120,160,200,240"
    )
    args = parser.parse_args()

    if args.max_epochs <= 0 or args.learning_rate <= 0.0 or args.top_k <= 0:
        raise ValueError("epochs, learning rate, and top-k must be positive")
    checkpoint_epochs = {
        int(value) for value in args.checkpoint_epochs.split(",") if value
    }
    if any(epoch <= 0 for epoch in checkpoint_epochs):
        raise ValueError("checkpoint epochs must be positive")

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
        raise ValueError("R2J expects the audited depth-3 model")
    prefixes = SCOPES[args.scope]
    model = model.to(dtype=torch.float32, device=device)
    named_parameters = list(model.named_parameters())
    for name, parameter in named_parameters:
        parameter.requires_grad_(name.startswith(prefixes))
    parameters = [parameter for _, parameter in named_parameters if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no parameters selected by the requested scope")

    train_structures = read(args.train_file, index=":")
    train_records, train_roles = dataset(args.train_file, float(model.r_max))
    valid_records, valid_roles = dataset(args.valid_file, float(model.r_max))
    test_records, test_roles = dataset(args.test_file, float(model.r_max))
    if (len(train_records), len(valid_records), len(test_records)) != (81, 34, 26):
        raise ValueError("unexpected R2J dataset counts")
    if Counter(train_roles) != Counter(
        {"harmonic_replay": 72, "fixed_smearing_support_repair": 9}
    ):
        raise ValueError(f"unexpected train roles: {Counter(train_roles)}")
    if Counter(valid_roles) != Counter(
        {"harmonic_replay": 25, "fixed_smearing_support_repair": 9}
    ):
        raise ValueError(f"unexpected validation roles: {Counter(valid_roles)}")
    if Counter(test_roles) != Counter({"thermal": 26}):
        raise ValueError(f"unexpected test roles: {Counter(test_roles)}")

    _, reference, cell, operator_degauss = load_operator(args.operator)
    if abs(operator_degauss - FIXED_DEGAUSS_RY) > 5.0e-10:
        raise ValueError("R2J requires the frozen T300 q6 operator")
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )
    harmonic_indices = [
        index for index, role in enumerate(train_roles) if role == "harmonic_replay"
    ]
    support_indices = [
        index
        for index, role in enumerate(train_roles)
        if role == "fixed_smearing_support_repair"
    ]
    support_modes = {}
    support_ids = {}
    for index in support_indices:
        mapping, _ = structure_mapping(train_structures[index], reference, cell)
        support_modes[index] = torch.as_tensor(
            mode[mapping], dtype=torch.complex64, device=device
        )
        support_ids[index] = int(train_structures[index].info["sscha_index"])

    train_batches = batches(train_records, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint_models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    ema = EMA(named_parameters, args.ema_decay)
    generator = torch.Generator().manual_seed(args.seed)

    backup = ema.apply(named_parameters)
    initial_valid = force_metrics(model, valid_records, valid_roles, device)
    initial_test = force_metrics(model, test_records, test_roles, device)
    EMA.restore(named_parameters, backup)
    print(
        json.dumps({"epoch": 0, "valid": initial_valid, "test": initial_test}),
        flush=True,
    )

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        harmonic_order = torch.randperm(
            len(harmonic_indices), generator=generator
        ).tolist()
        support_order = torch.randperm(
            len(support_indices), generator=generator
        ).tolist()
        step_records = []
        for step, support_position in enumerate(support_order):
            chosen_harmonic = [
                harmonic_indices[position]
                for position in harmonic_order[8 * step : 8 * (step + 1)]
            ]
            support_index = support_indices[support_position]
            optimizer.zero_grad(set_to_none=True)
            harmonic, harmonic_loss = harmonic_gradient(
                model,
                parameters,
                [train_batches[index] for index in chosen_harmonic],
            )
            harmonic_unit = unit(harmonic)
            support_unit, support_record = support_gradient(
                model,
                parameters,
                train_batches[support_index],
                support_modes[support_index],
                args.top_k,
            )
            objective_cosine = float(
                dot(harmonic_unit, support_unit).detach().cpu()
            )
            common = unit(add(harmonic_unit, support_unit))
            harmonic_descent = float(dot(common, harmonic_unit).detach().cpu())
            support_descent = float(dot(common, support_unit).detach().cpu())
            if harmonic_descent <= 0.0 or support_descent <= 0.0:
                raise RuntimeError("failed to construct a strict common descent direction")
            for parameter, value in zip(parameters, common, strict=True):
                parameter.grad = value
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            ema.update(named_parameters)
            step_records.append(
                {
                    "support_sscha_index": support_ids[support_index],
                    "harmonic_scaled_loss": harmonic_loss,
                    "objective_gradient_cosine": objective_cosine,
                    "common_descent_dot_harmonic_unit": harmonic_descent,
                    "common_descent_dot_support_unit": support_descent,
                    **support_record,
                }
            )

        should_evaluate = (
            epoch == 1
            or epoch == args.max_epochs
            or epoch % args.eval_interval == 0
            or epoch in checkpoint_epochs
        )
        record_out = {
            "epoch": epoch,
            "mean_harmonic_scaled_loss": float(
                np.mean([record["harmonic_scaled_loss"] for record in step_records])
            ),
            "mean_support_force_scaled_loss": float(
                np.mean(
                    [record["support_force_scaled_loss"] for record in step_records]
                )
            ),
            "mean_support_Aprime_scaled_loss": float(
                np.mean(
                    [record["support_Aprime_scaled_loss"] for record in step_records]
                )
            ),
            "mean_support_topk_scaled_loss": float(
                np.mean(
                    [record["support_topk_scaled_loss"] for record in step_records]
                )
            ),
            "mean_objective_gradient_cosine": float(
                np.mean(
                    [record["objective_gradient_cosine"] for record in step_records]
                )
            ),
            "minimum_common_descent_dot": float(
                min(
                    min(
                        record["common_descent_dot_harmonic_unit"],
                        record["common_descent_dot_support_unit"],
                    )
                    for record in step_records
                )
            ),
        }
        if should_evaluate:
            backup = ema.apply(named_parameters)
            record_out["valid"] = force_metrics(
                model, valid_records, valid_roles, device
            )
            record_out["test"] = force_metrics(
                model, test_records, test_roles, device
            )
            if epoch in checkpoint_epochs:
                atomic_model(checkpoint_dir / f"epoch{epoch}.model", model)
            EMA.restore(named_parameters, backup)
            print(json.dumps(record_out), flush=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record_out) + "\n")

    backup = ema.apply(named_parameters)
    final_valid = force_metrics(model, valid_records, valid_roles, device)
    final_test = force_metrics(model, test_records, test_roles, device)
    final_model = args.output_dir / f"{args.name}.model"
    atomic_model(final_model, model)
    EMA.restore(named_parameters, backup)
    summary = {
        "status": "R2J_gate_balanced_common_descent_training_complete",
        "energy_conservative": True,
        "new_DFT_labels": 0,
        "long_range_model_modified": False,
        "thermal_structures_used_for_gradient_updates": 0,
        "fixed_condition": {
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": operator_degauss,
        },
        "initial_model": {
            "path": str(args.initial_model),
            "sha256": sha256(args.initial_model),
        },
        "datasets": {
            "train": {"path": str(args.train_file), "sha256": sha256(args.train_file)},
            "valid": {"path": str(args.valid_file), "sha256": sha256(args.valid_file)},
            "test": {"path": str(args.test_file), "sha256": sha256(args.test_file)},
        },
        "training": {
            "scope": args.scope,
            "trainable_prefixes": list(prefixes),
            "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
            "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "max_epochs": args.max_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ema_decay": args.ema_decay,
            "seed": args.seed,
            "top_k": args.top_k,
            "steps_per_epoch": 9,
            "harmonic_configurations_per_step": 8,
            "support_configurations_per_step": 1,
            "gradient_combination": (
                "unit(harmonic_gradient) + unit(gate-balanced_support_gradient)"
            ),
            "scales": {
                "harmonic_force_eV_A": HARMONIC_SCALE_EV_A,
                "support_force_eV_A": SUPPORT_FORCE_SCALE_EV_A,
                "support_Aprime_eV_A": APRIME_SCALE_EV_A,
                "support_top_component_eV_A": TOP_COMPONENT_SCALE_EV_A,
            },
        },
        "initial_validation": initial_valid,
        "initial_test": initial_test,
        "final_validation": final_valid,
        "final_test": final_test,
        "checkpoint_models": {
            str(epoch): str(checkpoint_dir / f"epoch{epoch}.model")
            for epoch in sorted(checkpoint_epochs)
            if (checkpoint_dir / f"epoch{epoch}.model").is_file()
        },
        "final_model": {"path": str(final_model), "sha256": sha256(final_model)},
        "Aprime_mode": mode_provenance,
    }
    atomic_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
