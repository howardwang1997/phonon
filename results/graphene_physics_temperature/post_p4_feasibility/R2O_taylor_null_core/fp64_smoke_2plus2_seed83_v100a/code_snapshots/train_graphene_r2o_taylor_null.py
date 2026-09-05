#!/usr/bin/env python3
"""Train the full-parameter R2O whole-energy Taylor2 remainder.

The input MACE is the exact final state of the 80-epoch thermal-only pretrain.
Stage 2 continues *all* encoder and readout parameters.  Each force is an
autograd derivative of the complete live Taylor-remainder scalar energy; the
reference gradient and HVP remain differentiable with respect to theta.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from graphene_r2o_taylor_null import (
    Assignment,
    adapt_reference_cell,
    finite_parameters_and_gradients,
    fixed_reference_graph,
    model_dtype,
    reordered_structure,
    sha256,
    solve_assignment,
    state_dict_sha256,
    strict_json,
    taylor_remainder_energy_and_forces,
    torch_load,
    validate_current_edge_set,
    validate_mace_architecture,
)


FORMAT = "graphene_r2o_whole_energy_taylor2_training_v1"
SEED = 83
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
EMA_DECAY = 0.99
GRADIENT_CLIP = 20.0
GROUP_MASS = {
    "E50": 0.50,
    "T300": 0.20,
    "T600": 0.20,
    "harmonic_zero": 0.10,
}
GROUP_SCALE_EV_A = {
    "E50": 0.030,
    "T300": 0.030,
    "T600": 0.030,
    "harmonic_zero": 0.0005,
}


@dataclass
class Prepared:
    role: str
    source_index: int
    positions_reference_order: np.ndarray
    target_forces_reference_order: np.ndarray
    image_integer_reference_order: np.ndarray
    reference_positions: np.ndarray
    graph_size: int


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

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, payload: dict, device: torch.device) -> None:
        if float(payload["decay"]) != self.decay:
            raise ValueError("EMA decay changed across resume")
        if set(payload["shadow"]) != set(self.shadow):
            raise ValueError("EMA parameter names changed across resume")
        self.shadow = {
            name: value.to(device=device) for name, value in payload["shadow"].items()
        }

    def materialized(self, model: torch.nn.Module) -> torch.nn.Module:
        result = copy.deepcopy(model).cpu()
        state = result.state_dict()
        for name, value in self.shadow.items():
            state[name] = value.detach().cpu()
        result.load_state_dict(state, strict=True)
        return result


def role(structure: Atoms) -> str:
    value = str(structure.info.get("r2o_role", ""))
    if value == "exact_e50_seed0_train":
        return "E50"
    if value == "auxiliary_T300_train":
        return "T300"
    if value == "auxiliary_T600_train":
        return "T600"
    if value == "harmonic_lambda1_small_zero_train":
        return "harmonic_zero"
    raise ValueError(f"unexpected R2O training role {value!r}")


def prepare_records(
    thermal: list[Atoms],
    harmonic: list[Atoms],
    reference6: Atoms,
    reference8: Atoms,
) -> tuple[list[Prepared], dict[int, Atoms]]:
    records = []
    references = {72: reference6, 128: reference8}
    for index, structure in enumerate(thermal + harmonic):
        template = references.get(len(structure))
        if template is None:
            raise ValueError("R2O training record has unsupported size")
        reference = adapt_reference_cell(template, structure)
        assignment = solve_assignment(structure, reference)
        ordered = reordered_structure(structure, assignment)
        aligned = ordered.copy()
        aligned.positions = (
            np.asarray(ordered.positions, float)
            - assignment.image_integer_reference_order @ np.asarray(ordered.cell, float)
        )
        validate_current_edge_set(aligned, reference)
        target = np.asarray(structure.arrays["REF_forces"], float)[
            assignment.reference_to_source
        ]
        records.append(
            Prepared(
                role=role(structure),
                source_index=index,
                positions_reference_order=np.asarray(ordered.positions, float),
                target_forces_reference_order=target,
                image_integer_reference_order=assignment.image_integer_reference_order,
                reference_positions=np.asarray(reference.positions, float),
                graph_size=len(reference),
            )
        )
    counts = {name: sum(item.role == name for item in records) for name in GROUP_MASS}
    if counts["E50"] != 20 or counts["T300"] != 36 or counts["T600"] != 36:
        raise ValueError(f"thermal group counts changed: {counts}")
    if counts["harmonic_zero"] != len(harmonic) or not harmonic:
        raise ValueError("harmonic-zero count changed")
    return records, references


def atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    contract: dict,
    epoch_metrics: dict,
) -> dict:
    return {
        "format": FORMAT,
        "epoch": int(epoch),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "contract": contract,
        "epoch_metrics": epoch_metrics,
    }


def restore_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    contract: dict,
) -> int:
    payload = torch_load(path, map_location=device)
    if payload.get("format") != FORMAT or payload.get("contract") != contract:
        raise ValueError("R2O resume checkpoint contract mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    ema.load_state_dict(payload["ema_state_dict"], device)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])
    return int(payload["epoch"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thermal-train", type=Path, required=True)
    parser.add_argument("--harmonic-zero-train", type=Path, required=True)
    parser.add_argument("--reference-6x6", type=Path, required=True)
    parser.add_argument("--reference-8x8", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--initial-model-sha256", required=True)
    parser.add_argument("--launcher-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--checkpoint-epochs", default="40,80,120,160,200,235,240")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in (
        args.thermal_train,
        args.harmonic_zero_train,
        args.reference_6x6,
        args.reference_8x8,
        args.initial_model,
        args.launcher_freeze,
    ):
        lowered = str(path).lower()
        if any(token in lowered for token in ("seed2", "reserved", "support")):
            raise ValueError(f"R2O trainer rejects forbidden path {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(args.initial_model) != args.initial_model_sha256:
        raise ValueError("80-epoch pretrain model hash changed")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    checkpoint_epochs = sorted(
        {int(value) for value in args.checkpoint_epochs.split(",") if value}
    )
    if not checkpoint_epochs or max(checkpoint_epochs) > args.epochs:
        raise ValueError("checkpoint epoch is outside the frozen training run")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)
    model = torch_load(args.initial_model, map_location=device).to(device)
    architecture = validate_mace_architecture(model)
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("stage2 must continue all MACE parameters")
    thermal = read(args.thermal_train, index=":")
    harmonic = read(args.harmonic_zero_train, index=":")
    if len(thermal) != 92:
        raise ValueError("R2O thermal train count changed")
    reference6 = read(args.reference_6x6, index=0)
    reference8 = read(args.reference_8x8, index=0)
    records, references = prepare_records(
        thermal, harmonic, reference6, reference8
    )
    counts = {name: sum(item.role == name for item in records) for name in GROUP_MASS}
    dtype = model_dtype(model)
    graphs = {
        size: fixed_reference_graph(reference, device=device, dtype=dtype)
        for size, reference in references.items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    ema = EMA(model, EMA_DECAY)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": FORMAT,
        "thermal_train_sha256": sha256(args.thermal_train),
        "harmonic_zero_train_sha256": sha256(args.harmonic_zero_train),
        "reference_6x6_sha256": sha256(args.reference_6x6),
        "reference_8x8_sha256": sha256(args.reference_8x8),
        "initial_model_sha256": args.initial_model_sha256,
        "launcher_freeze_sha256": sha256(args.launcher_freeze),
        "architecture": architecture,
        "epochs": args.epochs,
        "checkpoint_epochs": checkpoint_epochs,
        "seed": SEED,
        "optimizer": {
            "name": "AdamW",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "scheduler": "ExponentialLR",
            "gamma": 1.0,
            "ema_decay": EMA_DECAY,
        },
        "group_counts": counts,
        "group_mass": GROUP_MASS,
        "group_scale_eV_A": GROUP_SCALE_EV_A,
        "selection_data_read": False,
        "seed1_seed2_support_read": False,
    }
    strict_json(args.output_dir / "stage2_contract.json", contract)

    start_epoch = 0
    latest = checkpoint_dir / "latest.pt"
    metrics_path = args.output_dir / "training_metrics.jsonl"
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError("--resume requested without latest.pt")
        start_epoch = restore_checkpoint(
            latest, model, ema, optimizer, scheduler, device, contract
        )
        latest_payload = torch_load(latest, map_location="cpu")
        metrics_records = []
        if metrics_path.exists():
            metrics_records = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if len(metrics_records) > start_epoch:
            metrics_records = metrics_records[:start_epoch]
            temporary_metrics = metrics_path.with_name(metrics_path.name + ".tmp")
            temporary_metrics.write_text(
                "".join(json.dumps(item, allow_nan=False) + "\n" for item in metrics_records),
                encoding="utf-8",
            )
            os.replace(temporary_metrics, metrics_path)
        if len(metrics_records) == start_epoch - 1:
            recovered_metric = latest_payload.get("epoch_metrics")
            if not isinstance(recovered_metric, dict) or int(recovered_metric.get("epoch", -1)) != start_epoch:
                raise ValueError("latest checkpoint lacks the missing epoch metric")
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(recovered_metric, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            metrics_records.append(recovered_metric)
        if len(metrics_records) != start_epoch or [
            int(item.get("epoch", -1)) for item in metrics_records
        ] != list(range(1, start_epoch + 1)):
            raise ValueError("training_metrics.jsonl is inconsistent with latest.pt")
        fixed_from_latest = checkpoint_dir / f"r2o_epoch-{start_epoch}.pt"
        if start_epoch in checkpoint_epochs and not fixed_from_latest.exists():
            temporary_fixed = fixed_from_latest.with_name(fixed_from_latest.name + ".tmp")
            shutil.copy2(latest, temporary_fixed)
            os.replace(temporary_fixed, fixed_from_latest)
    elif latest.exists():
        raise ValueError("refusing implicit restart over an existing R2O checkpoint")

    if start_epoch == 0 and metrics_path.exists():
        raise ValueError("stale metrics exist without an exact resume")
    total = len(records)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        permutation = np.random.default_rng(SEED * 1000003 + epoch).permutation(total)
        sums = {name: 0.0 for name in GROUP_MASS}
        model.train()
        epoch_started = time.monotonic()
        for record_index in permutation:
            record = records[int(record_index)]
            positions = torch.as_tensor(
                record.positions_reference_order, dtype=dtype, device=device
            ).clone().requires_grad_(True)
            reference_positions = torch.as_tensor(
                record.reference_positions, dtype=dtype, device=device
            )
            image = torch.as_tensor(
                record.image_integer_reference_order,
                dtype=torch.int64,
                device=device,
            )
            target = torch.as_tensor(
                record.target_forces_reference_order, dtype=dtype, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            result = taylor_remainder_energy_and_forces(
                model,
                graphs[record.graph_size],
                positions,
                reference_positions,
                image,
                create_graph=True,
            )
            scaled_mse = torch.mean(
                torch.square(result.forces_reference_order - target)
            ) / (GROUP_SCALE_EV_A[record.role] ** 2)
            coefficient = GROUP_MASS[record.role] * total / counts[record.role]
            loss = coefficient * scaled_mse
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite R2O loss")
            loss.backward()
            if not finite_parameters_and_gradients(model):
                raise FloatingPointError("non-finite R2O parameter or gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite R2O gradient norm")
            optimizer.step()
            ema.update(model)
            sums[record.role] += float(scaled_mse.detach().cpu())
        scheduler.step()
        payload = {
            "epoch": epoch,
            "wall_seconds": time.monotonic() - epoch_started,
            "mean_gate_normalized_MSE": {
                name: sums[name] / counts[name] for name in GROUP_MASS
            },
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        checkpoint = checkpoint_payload(
            epoch=epoch,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            contract=contract,
            epoch_metrics=payload,
        )
        if epoch in checkpoint_epochs:
            atomic_torch_save(checkpoint_dir / f"r2o_epoch-{epoch}.pt", checkpoint)
        atomic_torch_save(latest, checkpoint)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(payload, allow_nan=False), flush=True)

    raw_model = copy.deepcopy(model).cpu()
    ema_model = ema.materialized(model)
    raw_path = args.output_dir / "r2o_stage2_raw.model"
    ema_path = args.output_dir / "r2o_stage2_ema.model"
    atomic_torch_save(raw_path, raw_model)
    atomic_torch_save(ema_path, ema_model)
    runtime = {
        "status": "R2O_stage2_training_complete_pending_fixed_gate",
        "wall_time_seconds": time.monotonic() - started,
        "epochs": args.epochs,
        "raw_model": {"path": str(raw_path), "sha256": sha256(raw_path)},
        "ema_model": {"path": str(ema_path), "sha256": sha256(ema_path)},
        "raw_state_sha256": state_dict_sha256(raw_model),
        "ema_state_sha256": state_dict_sha256(ema_model),
        "stage2_contract_sha256": sha256(args.output_dir / "stage2_contract.json"),
        "fixed_checkpoints": {
            str(epoch): {
                "path": str(checkpoint_dir / f"r2o_epoch-{epoch}.pt"),
                "sha256": sha256(checkpoint_dir / f"r2o_epoch-{epoch}.pt"),
            }
            for epoch in checkpoint_epochs
        },
        "peak_cuda_memory_MiB": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
        "all_parameters_trained": True,
        "theta_differentiable_reference_gradient_and_HVP": True,
        "seed1_seed2_support_read": False,
    }
    strict_json(args.output_dir / "stage2_runtime.json", runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
