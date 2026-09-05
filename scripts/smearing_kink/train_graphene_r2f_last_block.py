#!/usr/bin/env python3
"""Directly fine-tune the terminal block of the frozen graphene depth-3 MACE.

MACE 0.3.16 cannot reconstruct this three-interaction single-element model
through its generic foundation-model element remapper.  This trainer loads the
audited model without reconstruction, freezes every parameter except the last
interaction/product/readout, and keeps the model energy-conservative.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.data import AtomicData, config_from_atoms
from mace.data.utils import KeySpecification
from mace.tools import AtomicNumberTable, torch_geometric


TRAINABLE_PREFIXES = ("interactions.2.", "products.2.", "readouts.2.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_model(path: Path, model: torch.nn.Module) -> None:
    temporary = path.with_name(path.name + ".tmp")
    cpu_model = copy.deepcopy(model).to("cpu")
    torch.save(cpu_model, temporary)
    os.replace(temporary, path)
    del cpu_model


def dataset(path: Path, cutoff: float) -> tuple[list, list[str]]:
    structures = read(path, index=":")
    key_specification = KeySpecification(
        info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
    )
    table = AtomicNumberTable([6])
    records = []
    roles = []
    for structure in structures:
        if set(structure.numbers) != {6}:
            raise ValueError(f"non-carbon structure in {path}")
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
    return records, roles


class EMA:
    def __init__(self, named_parameters: list[tuple[str, torch.nn.Parameter]], decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in named_parameters
            if parameter.requires_grad
        }

    def update(self, named_parameters: list[tuple[str, torch.nn.Parameter]]) -> None:
        with torch.no_grad():
            for name, parameter in named_parameters:
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        parameter.detach(), alpha=1.0 - self.decay
                    )

    def apply(self, named_parameters: list[tuple[str, torch.nn.Parameter]]) -> dict:
        backup = {}
        with torch.no_grad():
            for name, parameter in named_parameters:
                if name in self.shadow:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        return backup

    @staticmethod
    def restore(
        named_parameters: list[tuple[str, torch.nn.Parameter]], backup: dict
    ) -> None:
        with torch.no_grad():
            for name, parameter in named_parameters:
                if name in backup:
                    parameter.copy_(backup[name])


def force_metrics(
    model: torch.nn.Module,
    records: list,
    roles: list[str],
    device: torch.device,
) -> dict:
    model.eval()
    loader = torch_geometric.dataloader.DataLoader(
        records, batch_size=1, shuffle=False, drop_last=False
    )
    errors_by_role: dict[str, list[np.ndarray]] = defaultdict(list)
    for batch, role in zip(loader, roles, strict=True):
        batch = batch.to(device)
        output = model(batch.to_dict(), training=False, compute_force=True)
        error = (output["forces"] - batch.forces).detach().cpu().numpy()
        errors_by_role[role].append(error.reshape(-1))

    def summarize(parts: list[np.ndarray]) -> dict:
        values = np.concatenate(parts)
        return {
            "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(values**2))),
            "MAE_meV_A": float(1000.0 * np.mean(np.abs(values))),
            "max_abs_meV_A": float(1000.0 * np.max(np.abs(values))),
            "n_force_components": int(values.size),
        }

    all_errors = [part for parts in errors_by_role.values() for part in parts]
    return {
        "all": summarize(all_errors),
        "by_role": {
            role: summarize(parts) for role, parts in sorted(errors_by_role.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-8)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--checkpoint-epochs", default="40,80,120,160")
    args = parser.parse_args()

    if args.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
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
        raise ValueError("R2F expects the audited three-interaction depth-3 model")
    if list(model.heads) != ["Default"]:
        raise ValueError("R2F expects one Default head")
    if [int(value) for value in model.atomic_numbers.tolist()] != [6]:
        raise ValueError("R2F expects a carbon-only model")
    model = model.to(dtype=torch.float32, device=device)
    named_parameters = list(model.named_parameters())
    for name, parameter in named_parameters:
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
    trainable = [parameter for _, parameter in named_parameters if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable terminal-block parameters")

    cutoff = float(model.r_max)
    train_records, train_roles = dataset(args.train_file, cutoff)
    valid_records, valid_roles = dataset(args.valid_file, cutoff)
    test_records, test_roles = dataset(args.test_file, cutoff)
    if (len(train_records), len(valid_records), len(test_records)) != (81, 34, 26):
        raise ValueError("unexpected R2F dataset counts")
    if Counter(train_roles) != Counter(
        {"harmonic_replay": 72, "fixed_smearing_support_repair": 9}
    ):
        raise ValueError(f"unexpected training roles: {Counter(train_roles)}")
    if Counter(valid_roles) != Counter(
        {"harmonic_replay": 25, "fixed_smearing_support_repair": 9}
    ):
        raise ValueError(f"unexpected validation roles: {Counter(valid_roles)}")
    if Counter(test_roles) != Counter({"thermal": 26}):
        raise ValueError(f"unexpected test roles: {Counter(test_roles)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint_models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    optimizer = torch.optim.Adam(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    ema = EMA(named_parameters, args.ema_decay)
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch_geometric.dataloader.DataLoader(
        train_records,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    backup = ema.apply(named_parameters)
    initial_valid = force_metrics(model, valid_records, valid_roles, device)
    initial_test = force_metrics(model, test_records, test_roles, device)
    EMA.restore(named_parameters, backup)
    history = []
    metrics_path.write_text("", encoding="utf-8")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        sum_squared = 0.0
        n_components = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.to_dict(), training=True, compute_force=True)
            error = output["forces"] - batch.forces
            loss = torch.mean(error.square())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
            optimizer.step()
            ema.update(named_parameters)
            sum_squared += float(error.detach().square().sum().cpu())
            n_components += int(error.numel())

        should_evaluate = (
            epoch == 1
            or epoch == args.max_epochs
            or epoch % args.eval_interval == 0
            or epoch in checkpoint_epochs
        )
        record = {
            "epoch": epoch,
            "train_force_RMSE_meV_A": float(
                1000.0 * np.sqrt(sum_squared / n_components)
            ),
        }
        if should_evaluate:
            backup = ema.apply(named_parameters)
            record["valid"] = force_metrics(
                model, valid_records, valid_roles, device
            )
            record["test"] = force_metrics(model, test_records, test_roles, device)
            if epoch in checkpoint_epochs:
                atomic_model(checkpoint_dir / f"epoch{epoch}.model", model)
            EMA.restore(named_parameters, backup)
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if should_evaluate:
            print(json.dumps(record), flush=True)

    backup = ema.apply(named_parameters)
    final_valid = force_metrics(model, valid_records, valid_roles, device)
    final_test = force_metrics(model, test_records, test_roles, device)
    final_model = args.output_dir / f"{args.name}.model"
    atomic_model(final_model, model)
    EMA.restore(named_parameters, backup)

    summary = {
        "status": "R2F_terminal_block_training_complete",
        "energy_conservative": True,
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
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ema_decay": args.ema_decay,
            "trainable_prefixes": list(TRAINABLE_PREFIXES),
            "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
            "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
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
    }
    atomic_json(args.output_dir / "direct_training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
