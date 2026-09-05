#!/usr/bin/env python3
"""Train one fixed-epoch R2M routed-tail outer fold.

There is intentionally no checkpoint selection in this program.  It reads
only the eight outer-training support configurations and the support-free R2M
training replay, runs exactly 240 epochs, and freezes the EMA state at epoch
240.  The outer-held configuration and E50 seed2 are not accepted as inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from mace.data import AtomicData, KeySpecification, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric

from graphene_r2m_routed_tail import (
    EXPECTED_INVARIANT_WIDTH,
    RoutedTail,
    RoutedTailSpecification,
    core_invariant_schema,
    correction_energy_and_forces,
    ensure_frozen_core,
    node_invariants,
    save_routed_tail,
    sha256,
    strict_json,
    torch_load,
)


EPOCHS = 240
SEED = 83
SELECTION_POLICY = "fixed_epoch_240_no_support_checkpoint_selection"
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
EMA_DECAY = 0.99
GRADIENT_CLIP = 20.0
ENERGY_WEIGHT = 0.25
ENERGY_SCALE_EV = 0.0194
GATE_OFF_SCALE = 0.05
GATE_HARMONIC_WEIGHT = 0.05
GATE_E50_WEIGHT = 0.02
GROUP_MASS = {
    "support": 0.50,
    "harmonic": 0.25,
    "e50": 0.15,
    "auxiliary": 0.10,
}
FORCE_SCALE_EV_A = {
    "support": 0.030,
    "harmonic": 0.009044673057081592,
    "e50": 0.030,
    "auxiliary": 0.030,
}


def reject_seed2_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in ("seed2", "reserved_e50", "reserved-e50")):
        raise ValueError(f"outer-fold trainer rejects seed2 path: {path}")


def geometry_fingerprint(structure: Atoms) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.asarray(structure.numbers, np.dtype("<i8")).tobytes())
    digest.update(np.round(np.asarray(structure.cell, float), 10).astype("<f8").tobytes())
    digest.update(
        np.round(np.asarray(structure.positions, float), 10).astype("<f8").tobytes()
    )
    return digest.hexdigest()


def model_dtype(core: torch.nn.Module) -> torch.dtype:
    for parameter in core.parameters():
        return parameter.dtype
    raise ValueError("core has no parameters")


def dummy_labelled(structure: Atoms) -> Atoms:
    atoms = structure.copy()
    atoms.calc = None
    atoms.info["REF_energy"] = float(atoms.info.get("REF_energy", 0.0))
    if "REF_forces" not in atoms.arrays:
        atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3), float)
    return atoms


def atomic_record(structure: Atoms, cutoff: float) -> AtomicData:
    specification = KeySpecification(
        info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
    )
    configuration = config_from_atoms(
        dummy_labelled(structure),
        key_specification=specification,
        config_type_weights=None,
        head_name="Default",
    )
    return AtomicData.from_config(
        configuration,
        z_table=AtomicNumberTable([6]),
        cutoff=cutoff,
        heads=["Default"],
    )


def batch_data(
    structures: list[Atoms], core: torch.nn.Module, device: torch.device
) -> dict[str, torch.Tensor]:
    records = [atomic_record(item, float(core.r_max)) for item in structures]
    loader = torch_geometric.dataloader.DataLoader(
        records, batch_size=len(records), shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to(device)
    data = batch.to_dict()
    dtype = model_dtype(core)
    for key, value in list(data.items()):
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.to(dtype=dtype)
    return data


def extract_invariants(
    structures: list[Atoms],
    core: torch.nn.Module,
    schema: dict,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    values = []
    with torch.no_grad():
        for start in range(0, len(structures), batch_size):
            selected = structures[start : start + batch_size]
            data = batch_data(selected, core, device)
            output = core(data, training=False, compute_force=False)
            invariants = node_invariants(output["node_feats"], schema)
            values.append(invariants.detach().cpu().numpy())
    return np.concatenate(values, axis=0)


def feature_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != EXPECTED_INVARIANT_WIDTH:
        raise ValueError("training-only feature matrix has the wrong shape")
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    rms = np.sqrt(np.mean(values**2, axis=0))
    floor = np.maximum(1.0e-6 * rms, 1.0e-8)
    scale = np.maximum(std, floor)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("non-finite training-only feature statistics")
    if np.any(scale <= 0.0):
        raise ValueError("non-positive training-only feature scale")
    return mean, scale, {
        "definition": "mean and max(std, 1e-6*RMS, 1e-8) over train8 plus support-free replay-train atoms",
        "input_dimension": EXPECTED_INVARIANT_WIDTH,
        "output_dimension": EXPECTED_INVARIANT_WIDTH,
        "feature_pruning": False,
        "minimum_std": float(np.min(std)),
        "minimum_RMS": float(np.min(rms)),
        "minimum_scale": float(np.min(scale)),
    }


def pristine_supercell(path: Path) -> Atoms:
    primitive = read(path, index=0)
    if len(primitive) != 2 or set(primitive.numbers) != {6}:
        raise ValueError("pristine reference must be the two-carbon graphene cell")
    pristine = primitive.repeat((6, 6, 1))
    in_plane = np.asarray(pristine.cell.lengths(), float)[:2]
    if float(np.min(in_plane)) <= 8.0:
        raise ValueError("pristine gauge cell is too small for the R2M force diameter")
    return pristine


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
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


@dataclass
class ReplayGroups:
    harmonic: list[Atoms]
    e50: list[Atoms]
    auxiliary: list[Atoms]


def split_replay(structures: list[Atoms]) -> ReplayGroups:
    groups = ReplayGroups(harmonic=[], e50=[], auxiliary=[])
    for item in structures:
        role = str(item.info.get("r2m_target_role", ""))
        if role == "harmonic_train":
            groups.harmonic.append(item)
        elif role == "exact_e50_seed0":
            if int(item.info.get("trajectory_seed", -1)) != 0:
                raise ValueError("non-seed0 E50 record entered replay-train")
            groups.e50.append(item)
        elif role == "auxiliary_legacy_thermal":
            groups.auxiliary.append(item)
        else:
            raise ValueError(f"unexpected replay role {role!r}")
    if (len(groups.harmonic), len(groups.e50), len(groups.auxiliary)) != (72, 20, 72):
        raise ValueError("unexpected support-free replay group counts")
    return groups


def epoch_permutation(length: int, epoch: int, salt: int) -> np.ndarray:
    generator = np.random.default_rng(SEED * 1000003 + epoch * 101 + salt)
    return generator.permutation(length)


def balanced_replay_indices(
    length: int, epoch: int, n_steps: int, salt: int
) -> np.ndarray:
    """Cycle through one fixed permutation; total exposure differs by at most one."""
    if length <= 0 or epoch <= 0 or n_steps <= 0:
        raise ValueError("balanced replay schedule requires positive inputs")
    generator = np.random.default_rng(SEED * 1000003 + salt)
    permutation = generator.permutation(length)
    start = ((epoch - 1) * n_steps) % length
    return np.asarray(
        [permutation[(start + step) % length] for step in range(n_steps)],
        dtype=int,
    )


def graph_mask(batch: torch.Tensor, graph: int) -> torch.Tensor:
    return batch == graph


def training_step(
    core: torch.nn.Module,
    tail: RoutedTail,
    schema: dict,
    structures: list[Atoms],
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    if len(structures) != 4:
        raise ValueError("one step must contain support/harmonic/E50/auxiliary")
    data = batch_data(structures, core, device)
    energies, predicted_force, diagnostics = correction_energy_and_forces(
        core, tail, data, schema, create_graph=True
    )
    batch = data["batch"]
    group_names = ("support", "harmonic", "e50", "auxiliary")
    target_force = []
    for graph, structure in enumerate(structures):
        if graph == 0:
            target_force.append(
                torch.as_tensor(
                    np.asarray(structure.arrays["R2M_TAIL_TARGET_forces"], float),
                    dtype=predicted_force.dtype,
                    device=device,
                )
            )
        else:
            target_force.append(
                torch.zeros(
                    (len(structure), 3), dtype=predicted_force.dtype, device=device
                )
            )
    target_force_joined = torch.cat(target_force, dim=0)
    force_losses = {}
    loss = energies.new_zeros(())
    for graph, group in enumerate(group_names):
        mask = graph_mask(batch, graph)
        error = predicted_force[mask] - target_force_joined[mask]
        scaled = torch.mean(error.square()) / FORCE_SCALE_EV_A[group] ** 2
        force_losses[group] = scaled
        loss = loss + GROUP_MASS[group] * scaled
    target_energy = torch.as_tensor(
        float(structures[0].info["R2M_TAIL_TARGET_energy_centered"]),
        dtype=energies.dtype,
        device=device,
    )
    energy_loss = ((energies[0] - target_energy) / ENERGY_SCALE_EV) ** 2
    loss = loss + ENERGY_WEIGHT * energy_loss
    harmonic_gate = diagnostics["gate"][graph_mask(batch, 1)].mean()
    e50_gate = diagnostics["gate"][graph_mask(batch, 2)].mean()
    loss = loss + GATE_HARMONIC_WEIGHT * (harmonic_gate / GATE_OFF_SCALE) ** 2
    loss = loss + GATE_E50_WEIGHT * (e50_gate / GATE_OFF_SCALE) ** 2
    metrics = {
        "loss": float(loss.detach().cpu()),
        "support_force_scaled": float(force_losses["support"].detach().cpu()),
        "harmonic_force_scaled": float(force_losses["harmonic"].detach().cpu()),
        "e50_force_scaled": float(force_losses["e50"].detach().cpu()),
        "auxiliary_force_scaled": float(force_losses["auxiliary"].detach().cpu()),
        "support_energy_scaled": float(energy_loss.detach().cpu()),
        "harmonic_gate_mean": float(harmonic_gate.detach().cpu()),
        "e50_gate_mean": float(e50_gate.detach().cpu()),
    }
    return loss, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--replay-train", type=Path, required=True)
    parser.add_argument("--pristine", type=Path, required=True)
    parser.add_argument("--core-model", type=Path, required=True)
    parser.add_argument("--core-sha256", required=True)
    parser.add_argument("--launcher-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for path in (
        args.fold_dir,
        args.replay_train,
        args.pristine,
        args.core_model,
        args.launcher_freeze,
        args.output_dir,
    ):
        reject_seed2_path(path)
    if sha256(args.core_model) != args.core_sha256:
        raise ValueError("frozen core SHA-256 mismatch")
    fold_manifest_path = args.fold_dir / "fold_manifest.json"
    fold = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    if fold.get("status") != "R2M_routed_tail_outer_fold_frozen_before_training":
        raise ValueError("wrong outer-fold manifest status")
    if fold.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("outer fold changed its fixed-epoch selection policy")
    if fold["inputs"]["core"]["sha256"] != args.core_sha256:
        raise ValueError("fold was prepared against another core")
    if Path(fold["inputs"]["core"]["path"]).resolve() != args.core_model.resolve():
        raise ValueError("fold was prepared against another core path")
    launcher = json.loads(args.launcher_freeze.read_text(encoding="utf-8"))
    if (
        launcher.get("status")
        != "R2M_routed_tail_outer_fold_launcher_frozen_before_training"
        or int(launcher.get("outer_fold", -1)) != int(fold["outer_fold"])
        or int(launcher.get("held_sscha_index", -1))
        != int(fold["held_sscha_index"])
        or launcher.get("selection_policy") != SELECTION_POLICY
        or launcher.get("E50_seed2_read_or_accepted_as_input") is not False
        or launcher.get("inputs", {}).get("core", {}).get("sha256")
        != args.core_sha256
        or launcher.get("inputs", {}).get("fold_manifest", {}).get("sha256")
        != sha256(fold_manifest_path)
    ):
        raise ValueError("launcher freeze is not bound to this outer fold/core")
    script_path = Path(__file__).resolve()
    module_path = script_path.with_name("graphene_r2m_routed_tail.py")
    for path in (script_path, module_path):
        record = launcher.get("code_snapshots", {}).get(path.name, {})
        if (
            record.get("sha256") != sha256(path)
            or Path(str(record.get("path", ""))).resolve() != path.resolve()
        ):
            raise ValueError(f"launcher did not freeze the executed {path.name}")
    train_path = args.fold_dir / "support_train8.xyz"
    if sha256(train_path) != fold["outputs"]["support_train8.xyz"]:
        raise ValueError("support train8 file changed")
    if sha256(args.replay_train) != fold["inputs"]["replay_train"]["sha256"]:
        raise ValueError("support-free replay-train file changed")
    if sha256(args.pristine) != fold["inputs"]["pristine"]["sha256"]:
        raise ValueError("pristine reference changed")

    support = read(train_path, index=":")
    replay = read(args.replay_train, index=":")
    if len(support) != 8 or len(replay) != 164:
        raise ValueError("unexpected outer-train/replay counts")
    train_fingerprints = [geometry_fingerprint(item) for item in support]
    if train_fingerprints != fold["leakage"]["train_support_fingerprints"]:
        raise ValueError("support train8 geometry/order changed")
    if fold["leakage"]["held_support_fingerprint"] in train_fingerprints:
        raise ValueError("outer-held support leaked into train8")
    if any("seed2" in str(item.info).lower() for item in replay):
        raise ValueError("seed2 metadata entered replay-train")
    groups = split_replay(replay)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    core = torch_load(args.core_model, map_location=device)
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    ensure_frozen_core(core)
    schema = core_invariant_schema(core)
    if schema["schema_sha256"] != fold["feature_schema"]["schema_sha256"]:
        raise ValueError("core feature schema differs from fold preparation")

    print("fitting training-only 64-D scaler", flush=True)
    scaler_values = extract_invariants(support + replay, core, schema, device)
    mean, scale, scaler_record = feature_statistics(scaler_values)
    pristine = pristine_supercell(args.pristine)
    pristine_values = extract_invariants([pristine], core, schema, device)
    pristine_center = np.mean(pristine_values, axis=0, keepdims=True)
    pristine_spread = float(np.max(np.abs(pristine_values - pristine_center)))
    pristine_scale = max(float(np.max(np.abs(pristine_values))), 1.0e-8)
    pristine_relative_spread = pristine_spread / pristine_scale
    if pristine_relative_spread > 5.0e-5:
        raise ValueError(
            "pristine carbon invariant rows are not symmetry-equivalent: "
            f"relative spread {pristine_relative_spread:g}"
        )
    dtype = model_dtype(core)
    tail = RoutedTail(
        RoutedTailSpecification(),
        torch.as_tensor(mean, dtype=dtype),
        torch.as_tensor(scale, dtype=dtype),
        torch.as_tensor(pristine_values, dtype=dtype),
    ).to(device=device)
    optimizer = torch.optim.AdamW(
        tail.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True
    )
    ema = EMA(tail, EMA_DECAY)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = args.output_dir / "code_snapshots"
    snapshots.mkdir()
    shutil.copy2(script_path, snapshots / script_path.name)
    shutil.copy2(module_path, snapshots / module_path.name)
    freeze = {
        "status": "frozen_before_R2M_routed_tail_outer_fold_training",
        "outer_fold": int(fold["outer_fold"]),
        "held_sscha_index": int(fold["held_sscha_index"]),
        "selection_policy": SELECTION_POLICY,
        "epochs": EPOCHS,
        "seed": SEED,
        "optimizer": {
            "name": "AdamW_amsgrad",
            "learning_rate": LEARNING_RATE,
            "constant_schedule": True,
            "weight_decay": WEIGHT_DECAY,
            "EMA_decay": EMA_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "loss": {
            "group_mass": GROUP_MASS,
            "force_scale_eV_A": FORCE_SCALE_EV_A,
            "support_centered_energy_weight": ENERGY_WEIGHT,
            "support_centered_energy_scale_eV": ENERGY_SCALE_EV,
            "support_centered_energy_scale_source": (
                "pre-existing fixed 19.4 meV support gate from the R2C/R2G "
                "development contract; frozen before all R2M outer folds and "
                "not tuned on any outer-held configuration"
            ),
            "harmonic_gate_off_weight": GATE_HARMONIC_WEIGHT,
            "e50_gate_off_weight": GATE_E50_WEIGHT,
            "gate_off_scale": GATE_OFF_SCALE,
        },
        "leakage": {
            "outer_held_file_read": False,
            "outer_held_geometry_label_scaler_gradient_or_selection": False,
            "E50_seed2_read": False,
            "validation_file_read": False,
            "support_checkpoint_selection": False,
        },
        "model": {
            "formula": "sum_i smootherstep(score_i)*(epsilon_i-c_C)",
            "full_energy_autograd": True,
            "g_times_force_shortcut": False,
            "query_feature_score_or_node_energy_detached": False,
            "feature_schema": schema,
            "specification": tail.specification.__dict__,
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in tail.parameters())
            ),
            "core_trainable_parameters": 0,
        },
        "feature_scaler": scaler_record,
        "pristine_carbon_gauge": {
            "n_atoms": len(pristine),
            "maximum_invariant_spread": pristine_spread,
            "relative_invariant_spread": pristine_relative_spread,
            "relative_spread_limit": 5.0e-5,
            "definition": "mean tail node energy over symmetry-equivalent pristine 6x6 carbon atoms",
        },
        "inputs": {
            "core": {"path": str(args.core_model), "sha256": args.core_sha256},
            "fold_manifest": {
                "path": str(fold_manifest_path),
                "sha256": sha256(fold_manifest_path),
            },
            "support_train8": {"path": str(train_path), "sha256": sha256(train_path)},
            "replay_train": {
                "path": str(args.replay_train),
                "sha256": sha256(args.replay_train),
            },
            "pristine": {"path": str(args.pristine), "sha256": sha256(args.pristine)},
            "launcher_freeze": {
                "path": str(args.launcher_freeze),
                "sha256": sha256(args.launcher_freeze),
            },
        },
        "snapshots": {
            script_path.name: sha256(snapshots / script_path.name),
            module_path.name: sha256(snapshots / module_path.name),
        },
    }
    strict_json(args.output_dir / "training_freeze.json", freeze)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    for epoch in range(1, EPOCHS + 1):
        support_order = epoch_permutation(len(support), epoch, 11)
        harmonic_order = balanced_replay_indices(
            len(groups.harmonic), epoch, len(support), 23
        )
        e50_order = balanced_replay_indices(
            len(groups.e50), epoch, len(support), 37
        )
        auxiliary_order = balanced_replay_indices(
            len(groups.auxiliary), epoch, len(support), 53
        )
        accumulated: dict[str, float] = {}
        tail.train()
        for step, support_index in enumerate(support_order):
            chosen = [
                support[int(support_index)],
                groups.harmonic[int(harmonic_order[step])],
                groups.e50[int(e50_order[step])],
                groups.auxiliary[int(auxiliary_order[step])],
            ]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = training_step(core, tail, schema, chosen, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tail.parameters(), GRADIENT_CLIP)
            optimizer.step()
            ema.update(tail)
            for name, value in metrics.items():
                accumulated[name] = accumulated.get(name, 0.0) + value
        epoch_record = {
            "epoch": epoch,
            "n_optimizer_steps": len(support),
            **{name: value / len(support) for name, value in accumulated.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, allow_nan=False) + "\n")
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(json.dumps(epoch_record, allow_nan=False), flush=True)

    backup = ema.apply(tail)
    final_path = args.output_dir / "r2m_routed_tail_epoch240_ema.pt"
    save_routed_tail(
        final_path,
        tail,
        schema,
        args.core_sha256,
        {
            "outer_fold": int(fold["outer_fold"]),
            "held_sscha_index": int(fold["held_sscha_index"]),
            "epoch": EPOCHS,
            "state": "EMA",
            "selection_policy": SELECTION_POLICY,
            "outer_held_read_before_freeze": False,
            "E50_seed2_read": False,
        },
    )
    EMA.restore(tail, backup)
    summary = {
        "status": "R2M_routed_tail_outer_fold_training_complete_pending_held_evaluation",
        "outer_fold": int(fold["outer_fold"]),
        "held_sscha_index": int(fold["held_sscha_index"]),
        "selection_policy": SELECTION_POLICY,
        "selected_epoch": EPOCHS,
        "selected_state": "EMA",
        "outer_held_read_for_training_scaler_or_selection": False,
        "E50_seed2_read": False,
        "training_role_counts": {
            "support": len(support),
            "harmonic": len(groups.harmonic),
            "e50_seed0": len(groups.e50),
            "auxiliary": len(groups.auxiliary),
        },
        "artifact": {"path": str(final_path), "sha256": sha256(final_path)},
        "fold_manifest": {
            "path": str(fold_manifest_path),
            "sha256": sha256(fold_manifest_path),
        },
        "training_freeze_path": str(args.output_dir / "training_freeze.json"),
        "training_freeze_sha256": sha256(args.output_dir / "training_freeze.json"),
        "training_metrics_path": str(metrics_path),
        "training_metrics_sha256": sha256(metrics_path),
    }
    strict_json(args.output_dir / "training_summary.json", summary)
    (args.output_dir / "TRAINING_DONE").touch()
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
