#!/usr/bin/env python3
"""Train a conservative compact short-bond expert on frozen graphene data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

from graphene_short_bond_expert import (
    ExpertSpecification,
    ShortBondExpert,
    build_topology,
    environment_statistics,
    expert_prediction,
    save_expert,
    torch_topology,
)


FORCE_SCALE_EV_A = 0.030
ENERGY_SCALE_EV_CONFIG = 0.0194


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


def mace_predictions(
    structures: list, model_path: Path, device: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    calculator = MACECalculator(
        model_paths=str(model_path), device=device, default_dtype="float32"
    )
    energies = []
    forces = []
    for source in structures:
        structure = source.copy()
        structure.calc = calculator
        energies.append(float(structure.get_potential_energy()))
        forces.append(np.asarray(structure.get_forces(), float))
    del calculator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(energies), forces


@dataclass
class PreparedRecord:
    structure: object
    role: str
    positions: torch.Tensor
    topology: dict[str, torch.Tensor]
    residual_force: torch.Tensor
    residual_energy: torch.Tensor
    base_energy: float
    base_force: np.ndarray


def prepare_records(
    structures: list,
    base_energy: np.ndarray,
    base_force: list[np.ndarray],
    specification: ExpertSpecification,
    device: torch.device,
) -> list[PreparedRecord]:
    records = []
    for structure, energy, force in zip(
        structures, base_energy, base_force, strict=True
    ):
        role = str(structure.info.get("delta_target_role", "unknown"))
        topology = torch_topology(
            build_topology(structure, specification), device, torch.float64
        )
        positions = torch.as_tensor(
            np.asarray(structure.positions, float),
            dtype=torch.float64,
            device=device,
        )
        target_force = np.asarray(structure.arrays["REF_forces"], float)
        target_energy = float(structure.info["REF_energy"])
        records.append(
            PreparedRecord(
                structure=structure,
                role=role,
                positions=positions,
                topology=topology,
                residual_force=torch.as_tensor(
                    target_force - force, dtype=torch.float64, device=device
                ),
                residual_energy=torch.as_tensor(
                    target_energy - energy, dtype=torch.float64, device=device
                ),
                base_energy=float(energy),
                base_force=np.asarray(force, float),
            )
        )
    return records


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(
                    parameter.detach(), alpha=1.0 - self.decay
                )

    def apply(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(backup[name])


def force_summary(values: list[np.ndarray]) -> dict:
    flat = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(flat**2))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(flat))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(flat))),
        "n_force_components": int(flat.size),
    }


def evaluate_records(
    model: ShortBondExpert,
    records: list[PreparedRecord],
    device: torch.device,
) -> dict:
    errors_by_role: dict[str, list[np.ndarray]] = defaultdict(list)
    support_energy_error = []
    expert_force_rms = []
    model.eval()
    for record in records:
        energy, force = expert_prediction(model, record.structure, device)
        combined_error = record.base_force + force - np.asarray(
            record.structure.arrays["REF_forces"], float
        )
        errors_by_role[record.role].append(combined_error)
        expert_force_rms.append(np.asarray(force, float))
        if record.role == "fixed_smearing_support_repair":
            support_energy_error.append(
                record.base_energy
                + energy
                - float(record.structure.info["REF_energy"])
            )
    result = {
        "all": force_summary(
            [value for values in errors_by_role.values() for value in values]
        ),
        "by_role": {
            role: force_summary(values)
            for role, values in sorted(errors_by_role.items())
        },
        "expert_force": force_summary(expert_force_rms),
    }
    if support_energy_error:
        values = np.asarray(support_energy_error)
        centered = values - np.mean(values)
        result["support_centered_energy_RMSE_meV_config"] = float(
            1000.0 * np.sqrt(np.mean(centered**2))
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--environment-mode", choices=["nonlinear", "constant"], default="nonlinear")
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-7)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--energy-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--checkpoint-epochs", default="1,5,10,20,40,80,120,160,240,320,400")
    parser.add_argument("--exclude-support-index", type=int)
    args = parser.parse_args()

    if args.max_epochs <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs and learning rate must be positive")
    checkpoint_epochs = {
        int(value) for value in args.checkpoint_epochs.split(",") if value
    }
    if any(value <= 0 for value in checkpoint_epochs):
        raise ValueError("checkpoint epochs must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("long_range_model_modified") is not False:
        raise ValueError("long-range model must remain frozen")
    if manifest.get("new_DFT_labels_for_this_stage") != 0:
        raise ValueError("R2G must not add DFT labels")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    train_all = read(args.train_file, index=":")
    valid = read(args.valid_file, index=":")
    test = read(args.test_file, index=":")
    if (len(train_all), len(valid), len(test)) != (81, 34, 26):
        raise ValueError("unexpected R2G dataset counts")
    expected_train = Counter(
        {"harmonic_replay": 72, "fixed_smearing_support_repair": 9}
    )
    train_roles_all = Counter(
        str(item.info.get("delta_target_role")) for item in train_all
    )
    if train_roles_all != expected_train:
        raise ValueError(f"unexpected training roles: {train_roles_all}")

    excluded = []
    train = []
    for structure in train_all:
        is_excluded = (
            args.exclude_support_index is not None
            and str(structure.info.get("delta_target_role"))
            == "fixed_smearing_support_repair"
            and int(structure.info.get("sscha_index", -1))
            == args.exclude_support_index
        )
        (excluded if is_excluded else train).append(structure)
    if args.exclude_support_index is not None and len(excluded) != 1:
        raise ValueError("LOCO exclusion must match exactly one support structure")

    specification = ExpertSpecification(environment_mode=args.environment_mode)
    environment_mean, environment_scale = environment_statistics(
        train, specification
    )
    print("precomputing frozen short-range base predictions", flush=True)
    all_structures = train + valid + test
    base_energy, base_force = mace_predictions(all_structures, args.base_model, args.device)
    n_train = len(train)
    n_valid = len(valid)
    train_records = prepare_records(
        train,
        base_energy[:n_train],
        base_force[:n_train],
        specification,
        device,
    )
    valid_records = prepare_records(
        valid,
        base_energy[n_train : n_train + n_valid],
        base_force[n_train : n_train + n_valid],
        specification,
        device,
    )
    test_records = prepare_records(
        test,
        base_energy[n_train + n_valid :],
        base_force[n_train + n_valid :],
        specification,
        device,
    )

    model = ShortBondExpert(
        specification, environment_mean, environment_scale
    ).to(device=device, dtype=torch.float64)
    energy_offset = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device=device))
    trainable = list(model.parameters()) + [energy_offset]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    ema = EMA(model, args.ema_decay)
    generator = torch.Generator().manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint_experts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    initial_validation = evaluate_records(model, valid_records, device)
    initial_test = evaluate_records(model, test_records, device)
    print(json.dumps({"epoch": 0, "valid": initial_validation, "test": initial_test}), flush=True)

    history = []
    for epoch in range(1, args.max_epochs + 1):
        order = torch.randperm(len(train_records), generator=generator).tolist()
        model.train()
        summed_loss = 0.0
        summed_force_squared = 0.0
        force_components = 0
        for index in order:
            record = train_records[index]
            positions = record.positions.detach().clone().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            energy, force = model.energy_and_forces(
                positions, record.topology, create_graph=True
            )
            force_error = force - record.residual_force
            force_loss = torch.mean(force_error.square()) / FORCE_SCALE_EV_A**2
            loss = force_loss
            if record.role == "fixed_smearing_support_repair":
                energy_error = energy - record.residual_energy - energy_offset
                loss = loss + args.energy_weight * (
                    energy_error / ENERGY_SCALE_EV_CONFIG
                ).square()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=20.0)
            optimizer.step()
            ema.update(model)
            summed_loss += float(loss.detach().cpu())
            summed_force_squared += float(force_error.detach().square().sum().cpu())
            force_components += int(force_error.numel())

        should_evaluate = (
            epoch == 1
            or epoch == args.max_epochs
            or epoch % args.eval_interval == 0
            or epoch in checkpoint_epochs
        )
        record_out = {
            "epoch": epoch,
            "mean_scaled_loss": summed_loss / len(train_records),
            "train_online_force_RMSE_meV_A": float(
                1000.0 * np.sqrt(summed_force_squared / force_components)
            ),
            "energy_offset_eV": float(energy_offset.detach().cpu()),
        }
        if should_evaluate:
            backup = ema.apply(model)
            record_out["valid"] = evaluate_records(model, valid_records, device)
            record_out["test"] = evaluate_records(model, test_records, device)
            if epoch in checkpoint_epochs:
                save_expert(
                    checkpoint_dir / f"epoch{epoch}.pt",
                    model,
                    {
                        "epoch": epoch,
                        "base_model_sha256": sha256(args.base_model),
                        "exclude_support_index": args.exclude_support_index,
                    },
                )
            EMA.restore(model, backup)
            print(json.dumps(record_out), flush=True)
        history.append(record_out)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record_out) + "\n")

    backup = ema.apply(model)
    final_validation = evaluate_records(model, valid_records, device)
    final_test = evaluate_records(model, test_records, device)
    final_path = args.output_dir / f"{args.name}.pt"
    save_expert(
        final_path,
        model,
        {
            "epoch": args.max_epochs,
            "base_model_sha256": sha256(args.base_model),
            "exclude_support_index": args.exclude_support_index,
        },
    )
    EMA.restore(model, backup)

    summary = {
        "status": "R2G_short_bond_expert_training_complete",
        "energy_conservative": True,
        "activation_is_C2": True,
        "new_DFT_labels": 0,
        "long_range_model_modified": False,
        "thermal_structures_used_for_gradient_updates": 0,
        "base_model": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
        "datasets": {
            "train": {"path": str(args.train_file), "sha256": sha256(args.train_file)},
            "valid": {"path": str(args.valid_file), "sha256": sha256(args.valid_file)},
            "test": {"path": str(args.test_file), "sha256": sha256(args.test_file)},
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
        },
        "training": {
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ema_decay": args.ema_decay,
            "energy_weight": args.energy_weight,
            "force_gate_scale_eV_A": FORCE_SCALE_EV_A,
            "energy_gate_scale_eV_config": ENERGY_SCALE_EV_CONFIG,
            "exclude_support_index": args.exclude_support_index,
            "n_train": len(train),
            "train_role_counts": dict(Counter(record.role for record in train_records)),
            "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        "specification": specification.__dict__,
        "environment_statistics": {
            "mean": environment_mean.tolist(),
            "scale": environment_scale.tolist(),
        },
        "initial_validation": initial_validation,
        "initial_test": initial_test,
        "final_validation": final_validation,
        "final_test": final_test,
        "checkpoint_experts": {
            str(epoch): str(checkpoint_dir / f"epoch{epoch}.pt")
            for epoch in sorted(checkpoint_epochs)
            if (checkpoint_dir / f"epoch{epoch}.pt").is_file()
        },
        "final_expert": {"path": str(final_path), "sha256": sha256(final_path)},
    }
    atomic_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
