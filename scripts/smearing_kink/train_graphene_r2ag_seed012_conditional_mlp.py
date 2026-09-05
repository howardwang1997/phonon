#!/usr/bin/env python3
"""Train and aggregate conservative seed012 conditional-MLP OOF folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ase.io import read

import graphene_r2o_taylor_null as r2o
import graphene_r2r1_linear_readout as r2r1
import graphene_r2r_multipolar_background as r2r
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from graphene_r2r0_formal import load_endpoint
from materialize_graphene_r2ac_seed2_full_bilinear import (
    read_geometry_only_extxyz,
)
from materialize_graphene_r2t_full_bilinear_basis import (
    full_bilinear_energy_columns,
)
from train_graphene_r2s_conditional_mlp import (
    INPUT_WIDTH,
    SEED,
    _node_features,
    recommended_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_PACKAGE_ROOT = BASE / "R2AG_seed012_conditional_mlp_package_20260826"
SEED1_PATH = ROOT / "data/graphene_r2o_taylor_null_core/valid_e50_seed1.xyz"
SEED2_PATH = (
    ROOT / "data/graphene_r2m_support_free_core/reserved_e50_seed2.xyz"
)
EXPECTED_INPUT_SHA256 = {
    "thermal92": "cc2e9c68d418fba996e8ac89c8156c8adf00768406f6171aaa57f0a9bd2720c1",
    "seed1": "bf67ac601c97ca8de21f7b4966c1cd88ea6e8439e560a05713ef07a681f3078d",
    "seed2": "ab8fb629501134ba9e94d11fcc67a6363971c2ccbf2bbae0e2d236a8c23988b1",
    "reference6": "2793b514768c6d875831a1b62fcb2aa5f858830cb2de7b1b2d5492dabeb8da90",
    "endpoint": "31dd053a01c700ee73eedff35e85c02cf7765de2299ff418259c270b3f823eb5",
}
FORMAT = "graphene_r2ag_seed012_gated_conditional_mlp_v1"
HIDDEN_WIDTHS = (32, 16)
ENERGY_SCALE_EV = 0.030
FORCE_SCALE_EV_A = 0.030
APRIME_SCALE_EV_A = 0.015
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-6
GRADIENT_CLIP = 10.0
GROUP_RANGES = {
    "E50_seed0": (0, 20),
    "E50_seed1": (20, 40),
    "E50_seed2": (40, 60),
    "T300": (60, 96),
    "T600": (96, 132),
}
SEED_RANGES = {
    "seed0": (0, 20),
    "seed1": (20, 40),
    "seed2": (40, 60),
}


class ConditionalReadout(torch.nn.Module):
    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        bilinear_mean: torch.Tensor,
        bilinear_scale: torch.Tensor,
        selected_bilinear: torch.Tensor,
        linear_skip: torch.Tensor,
    ) -> None:
        super().__init__()
        if feature_mean.shape != (34,) or feature_scale.shape != (34,):
            raise ValueError("R2AG MLP scaler shape changed")
        if bilinear_mean.shape != (34,) or bilinear_scale.shape != (34,):
            raise ValueError("R2AG bilinear scaler shape changed")
        if selected_bilinear.shape != (64,) or linear_skip.shape != (129,):
            raise ValueError("R2AG frozen linear skip schema changed")
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        self.register_buffer("bilinear_mean", bilinear_mean.detach().clone())
        self.register_buffer("bilinear_scale", bilinear_scale.detach().clone())
        self.register_buffer(
            "selected_bilinear", selected_bilinear.detach().clone()
        )
        self.register_buffer("linear_skip", linear_skip.detach().clone())
        self.linear1 = torch.nn.Linear(INPUT_WIDTH, HIDDEN_WIDTHS[0])
        self.linear2 = torch.nn.Linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1])
        self.linear3 = torch.nn.Linear(HIDDEN_WIDTHS[1], 1)
        self.activation = torch.nn.SiLU()
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)
        torch.nn.init.zeros_(self.linear3.weight)
        torch.nn.init.zeros_(self.linear3.bias)

    def conditional_node_energy(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        hidden = self.activation(self.linear1(normalized))
        hidden = self.activation(self.linear2(hidden))
        return ENERGY_SCALE_EV * self.linear3(hidden).squeeze(-1)


def indices_for(bounds: tuple[int, int]) -> np.ndarray:
    return np.arange(bounds[0], bounds[1], dtype=int)


def group_for(index: int) -> str:
    for name, (start, stop) in GROUP_RANGES.items():
        if start <= index < stop:
            return name
    raise ValueError(f"R2AG global index is outside 0..131: {index}")


def seed_for(index: int) -> str | None:
    for name, (start, stop) in SEED_RANGES.items():
        if start <= index < stop:
            return name
    return None


def state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_package(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    receipt_path = root / "receipt.json"
    package_path = root / "training_package.npz"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "R2AG_SEED012_CONDITIONAL_MLP_PACKAGE_FROZEN":
        raise ValueError("R2AG training package is not frozen")
    if file_sha256(package_path) != receipt.get("package_sha256"):
        raise ValueError("R2AG training package differs from receipt")
    with np.load(package_path, allow_pickle=False) as arrays:
        data = {key: np.array(arrays[key], copy=True) for key in arrays.files}
    expected_shapes = {
        "fixed_force_eV_A": (132, 72, 3),
        "reference_force_eV_A": (132, 72, 3),
        "Aprime_mode_real": (60, 72, 3),
        "Aprime_mode_imag": (60, 72, 3),
        "Aprime_coordinates_real_A": (60,),
        "Aprime_coordinates_imag_A": (60,),
        "total_base_force_eV_A": (60, 72, 3),
        "selected_bilinear_indices": (64,),
        "bilinear_feature_mean": (34,),
        "bilinear_feature_scale": (34,),
        "fold_hold_indices": (4, 33),
        "fold_linear_skip_coefficient": (4, 129),
        "all132_linear_skip_coefficient": (129,),
        "fixed_step32_OOF_predicted_force_eV_A": (132, 72, 3),
    }
    if set(data) != set(expected_shapes):
        raise ValueError("R2AG training package member set changed")
    for name, shape in expected_shapes.items():
        if data[name].shape != shape or not np.all(np.isfinite(data[name])):
            raise ValueError(f"R2AG package array changed: {name}")
    if not np.array_equal(
        np.sort(data["fold_hold_indices"].reshape(-1)), np.arange(132)
    ):
        raise ValueError("R2AG package folds do not cover all132 exactly once")
    return data, receipt


def structures_and_reference() -> tuple[list[Any], Any, dict[str, str]]:
    inputs = recommended_inputs()
    paths = {
        "thermal92": inputs.thermal92,
        "seed1": SEED1_PATH,
        "seed2": SEED2_PATH,
        "reference6": inputs.reference_6x6,
        "endpoint": inputs.endpoint_checkpoint,
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("R2AG geometry or frozen endpoint input changed")
    thermal = read_geometry_only_extxyz(inputs.thermal92)
    seed1 = read_geometry_only_extxyz(SEED1_PATH)
    seed2 = read_geometry_only_extxyz(SEED2_PATH)
    structures = thermal[:20] + seed1 + seed2 + thermal[20:]
    if len(structures) != 132 or any(len(item) != 72 for item in structures):
        raise ValueError("R2AG structure assembly changed")
    return structures, read(inputs.reference_6x6, index=0), hashes


def build_contexts(
    endpoint: torch.nn.Module,
    structures: Sequence[Any],
    reference: Any,
    device: str,
) -> dict[int, Any]:
    return {
        index: r2r._verified_production_context(
            endpoint, structure, reference, device=device
        )
        for index, structure in enumerate(structures)
    }


def feature_scaler(
    contexts: dict[int, Any], train_indices: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    values = []
    for index in train_indices:
        context = contexts[int(index)]
        aligned = (
            context.current_positions_reference_order - context.image_shift.detach()
        )
        observables = context.node_observables_fn(aligned)
        fields = context.background_fields_fn(aligned)
        values.append(_node_features(observables, fields).detach())
    joined = torch.cat(values, dim=0)
    mean = joined.mean(dim=0)
    scale = torch.sqrt(torch.mean(torch.square(joined - mean), dim=0))
    if torch.any(scale <= torch.max(scale) * 1.0e-10):
        raise ValueError("R2AG fold-train feature scaler contains a zero column")
    return mean, scale, {
        "sample_count": int(joined.shape[0]),
        "feature_width": int(joined.shape[1]),
        "min_relative_RMS": float((torch.min(scale) / torch.max(scale)).cpu()),
        "mean_raw_sha256": r2r1.raw_array_sha256(
            mean.detach().cpu().numpy(), "<f8"
        ),
        "scale_raw_sha256": r2r1.raw_array_sha256(
            scale.detach().cpu().numpy(), "<f8"
        ),
        "train_geometry_only": True,
    }


def tail_energy_force(
    readout: ConditionalReadout,
    context: Any,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = context.current_positions_reference_order
    positions.grad = None
    aligned = positions - context.image_shift.detach()
    observables = context.node_observables_fn(aligned)
    fields = context.background_fields_fn(aligned)
    features = _node_features(observables, fields)
    base_columns = context.parameter_basis_fn(observables, fields)
    normalized_bilinear = (
        features - readout.bilinear_mean
    ) / readout.bilinear_scale
    full_bilinear, _labels = full_bilinear_energy_columns(
        normalized_bilinear, fields
    )
    selected_bilinear = full_bilinear[readout.selected_bilinear]
    energy = (
        torch.dot(base_columns, readout.linear_skip[:65])
        + torch.dot(selected_bilinear, readout.linear_skip[65:])
        + torch.sum(
            fields.multipolar_gate * readout.conditional_node_energy(features)
        )
    )
    force_reference = -torch.autograd.grad(
        energy,
        positions,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    force_source = torch.empty_like(force_reference)
    reference_to_source = torch.as_tensor(
        context.assignment.reference_to_source,
        dtype=torch.long,
        device=force_reference.device,
    )
    force_source[reference_to_source] = force_reference
    return energy, force_source


def predict_indices(
    readout: ConditionalReadout,
    contexts: dict[int, Any],
    indices: Sequence[int],
    fixed: torch.Tensor,
) -> np.ndarray:
    readout.eval()
    output = np.empty((len(indices), 72, 3), dtype=np.float64)
    for row, index in enumerate(indices):
        with torch.enable_grad():
            _energy, tail_force = tail_energy_force(
                readout, contexts[int(index)], create_graph=False
            )
        output[row] = (fixed[int(index)] + tail_force).detach().cpu().numpy()
    return output


def train_epoch(
    readout: ConditionalReadout,
    endpoint: torch.nn.Module,
    contexts: dict[int, Any],
    train_indices: Sequence[int],
    fixed: torch.Tensor,
    target: torch.Tensor,
    modes: torch.Tensor,
    force_mass: float,
    projection_mass: float,
    group_masses: dict[str, float],
    seed_weights: dict[str, float],
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
) -> dict[str, float]:
    readout.train()
    optimizer.zero_grad(set_to_none=True)
    order = [int(value) for value in train_indices]
    rng.shuffle(order)
    group_counts = {
        name: sum(group_for(index) == name for index in order)
        for name in GROUP_RANGES
    }
    seed_counts = {
        name: sum(seed_for(index) == name for index in order) for name in SEED_RANGES
    }
    force_total = 0.0
    projection_total = 0.0
    for index in order:
        _energy, tail_force = tail_energy_force(
            readout, contexts[index], create_graph=True
        )
        error = fixed[index] + tail_force - target[index]
        group = group_for(index)
        force_loss = torch.mean(torch.square(error / FORCE_SCALE_EV_A))
        weighted_force = (
            force_mass * group_masses[group] / group_counts[group] * force_loss
        )
        loss = weighted_force
        force_total += float(weighted_force.detach().cpu())
        seed = seed_for(index)
        if seed is not None:
            projected = torch.sum(torch.conj(modes[index]) * error.to(torch.complex128))
            projection_loss = (
                torch.square(projected.real) + torch.square(projected.imag)
            ) / (2.0 * APRIME_SCALE_EV_A**2)
            weighted_projection = (
                projection_mass
                * seed_weights[seed]
                / seed_counts[seed]
                * projection_loss
            )
            loss = loss + weighted_projection
            projection_total += float(weighted_projection.detach().cpu())
        loss.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(readout.parameters(), GRADIENT_CLIP)
        .detach()
        .cpu()
    )
    optimizer.step()
    if r2o.state_dict_sha256(endpoint) != r2r.R2Q_ENDPOINT_STATE_SHA256:
        raise RuntimeError("R2AG training changed the frozen endpoint")
    return {
        "force_objective": force_total,
        "Aprime_objective": projection_total,
        "total_objective": force_total + projection_total,
        "gradient_norm_before_clip": gradient_norm,
    }


def run_fold(args: argparse.Namespace) -> None:
    if args.fold not in range(4):
        raise ValueError("R2AG fold must be 0..3")
    snapshot_epochs = tuple(sorted(set(args.snapshot_epochs)))
    if not snapshot_epochs or snapshot_epochs[0] != 0:
        raise ValueError("R2AG snapshot epochs must include zero")
    if snapshot_epochs[-1] != args.epochs or any(
        not 0 <= value <= args.epochs for value in snapshot_epochs
    ):
        raise ValueError("R2AG final snapshot must equal --epochs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.manual_seed(SEED + args.fold)
    np.random.seed(SEED + args.fold)
    random.seed(SEED + args.fold)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    package, package_receipt = load_package(args.package_root.resolve())
    structures, reference_atoms, input_hashes = structures_and_reference()
    inputs = recommended_inputs()
    endpoint = load_endpoint(inputs, args.device)
    for parameter in endpoint.parameters():
        parameter.requires_grad_(False)
    endpoint.eval()
    contexts = build_contexts(endpoint, structures, reference_atoms, args.device)
    hold = np.asarray(package["fold_hold_indices"][args.fold], dtype=int)
    train = np.setdiff1d(np.arange(132, dtype=int), hold)
    mean, scale, scaler_receipt = feature_scaler(contexts, train)
    readout = ConditionalReadout(
        mean,
        scale,
        torch.as_tensor(
            package["bilinear_feature_mean"], dtype=torch.float64, device=args.device
        ),
        torch.as_tensor(
            package["bilinear_feature_scale"], dtype=torch.float64, device=args.device
        ),
        torch.as_tensor(
            package["selected_bilinear_indices"],
            dtype=torch.long,
            device=args.device,
        ),
        torch.as_tensor(
            package["fold_linear_skip_coefficient"][args.fold],
            dtype=torch.float64,
            device=args.device,
        ),
    ).to(device=args.device, dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        readout.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    fixed = torch.as_tensor(
        package["fixed_force_eV_A"], dtype=torch.float64, device=args.device
    )
    target = torch.as_tensor(
        package["reference_force_eV_A"], dtype=torch.float64, device=args.device
    )
    modes = torch.as_tensor(
        package["Aprime_mode_real"] + 1.0j * package["Aprime_mode_imag"],
        dtype=torch.complex128,
        device=args.device,
    )
    hyper = package_receipt["fixed_linear_skip_hyperparameters"]
    projection_mass = float(hyper["projection_mass"])
    force_mass = 1.0 - projection_mass
    group_masses = {
        name: float(value) for name, value in hyper["group_masses"].items()
    }
    seed_weights = {
        name: float(value)
        for name, value in hyper["seed_projection_weights"].items()
    }
    initial = predict_indices(readout, contexts, hold, fixed)
    expected_initial = package["fixed_step32_OOF_predicted_force_eV_A"][hold]
    initial_replay = float(np.max(np.abs(initial - expected_initial)))
    if initial_replay > 2.0e-9:
        raise ValueError(f"R2AG runtime/materialized replay failed: {initial_replay:.3e}")

    predictions = [initial]
    state_snapshots = {0: {key: value.detach().cpu() for key, value in readout.state_dict().items()}}
    state_hashes = {0: state_sha256(readout)}
    history = []
    rng = random.Random(SEED + args.fold)
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        record = train_epoch(
            readout,
            endpoint,
            contexts,
            train,
            fixed,
            target,
            modes,
            force_mass,
            projection_mass,
            group_masses,
            seed_weights,
            optimizer,
            rng,
        )
        record.update({"epoch": epoch, "wall_seconds": time.perf_counter() - tick})
        history.append(record)
        if epoch in snapshot_epochs:
            predictions.append(predict_indices(readout, contexts, hold, fixed))
            state_snapshots[epoch] = {
                key: value.detach().cpu() for key, value in readout.state_dict().items()
            }
            state_hashes[epoch] = state_sha256(readout)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps(record, sort_keys=True), flush=True)
    if len(predictions) != len(snapshot_epochs):
        raise RuntimeError("R2AG snapshot prediction count changed")
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {
            "format": FORMAT,
            "kind": "conditional_development_fold_not_deployable",
            "fold": args.fold,
            "snapshot_epochs": snapshot_epochs,
            "state_dict_by_epoch": state_snapshots,
            "architecture": {
                "input_width": INPUT_WIDTH,
                "hidden_widths": HIDDEN_WIDTHS,
                "activation": "SiLU",
                "energy_scale_eV": ENERGY_SCALE_EV,
                "gate": "multipolar_gate",
                "frozen_step32_linear_skip": True,
            },
        },
        checkpoint_path,
    )
    predictions_path = output / "predictions.npz"
    np.savez(
        predictions_path,
        snapshot_epochs=np.asarray(snapshot_epochs, dtype="<i8"),
        hold_global_indices=np.asarray(hold, dtype="<i8"),
        hold_predicted_force_eV_A=np.asarray(predictions, dtype="<f8"),
    )
    receipt = {
        "format": FORMAT,
        "status": "R2AG_CONDITIONAL_MLP_FOLD_COMPLETE",
        "deployable": False,
        "fold": args.fold,
        "device": args.device,
        "epochs": args.epochs,
        "snapshot_epochs": list(snapshot_epochs),
        "train_global_indices": train.tolist(),
        "hold_global_indices": hold.tolist(),
        "objective": {
            "force_mass": force_mass,
            "Aprime_mass": projection_mass,
            "group_masses_within_force": group_masses,
            "seed_weights_within_Aprime": seed_weights,
            "force_scale_eV_A": FORCE_SCALE_EV_A,
            "Aprime_scale_eV_A": APRIME_SCALE_EV_A,
        },
        "optimizer": {
            "kind": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "feature_scaler": scaler_receipt,
        "initial_step32_runtime_replay_max_abs_eV_A": initial_replay,
        "readout_state_sha256_by_epoch": {
            str(epoch): value for epoch, value in state_hashes.items()
        },
        "history": history,
        "endpoint_state_sha256": r2o.state_dict_sha256(endpoint),
        "input_sha256": {
            **input_hashes,
            "training_package": file_sha256(
                args.package_root / "training_package.npz"
            ),
            "training_package_receipt": file_sha256(
                args.package_root / "receipt.json"
            ),
        },
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "predictions_sha256": file_sha256(predictions_path),
        "energy_labels_used": False,
        "unseen_525K_access": False,
        "elapsed_seconds": time.time() - started,
    }
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    (output / "DONE").write_text(receipt["status"] + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


def force_metrics(error: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(error).reshape(-1)
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(np.square(flat)))),
        "MAE_meV_A": float(1000.0 * np.mean(np.abs(flat))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(flat))),
        "n_force_components": int(flat.size),
    }


def restoring_slope(coordinates: np.ndarray, values: np.ndarray) -> float:
    return -float(
        np.vdot(coordinates, values).real
        / float(np.vdot(coordinates, coordinates).real)
    )


def scalar_metrics(prediction: np.ndarray, package: dict[str, np.ndarray]) -> dict[str, Any]:
    reference = package["reference_force_eV_A"]
    modes = package["Aprime_mode_real"] + 1.0j * package["Aprime_mode_imag"]
    coordinates = (
        package["Aprime_coordinates_real_A"]
        + 1.0j * package["Aprime_coordinates_imag_A"]
    )
    base = package["total_base_force_eV_A"]
    output: dict[str, Any] = {"force_by_group": {}, "Aprime_by_seed": {}}
    ratios = []
    for name, bounds in GROUP_RANGES.items():
        members = indices_for(bounds)
        current = force_metrics(prediction[members] - reference[members])
        output["force_by_group"][name] = current
        ratios.extend(
            (current["RMSE_meV_A"] / 30.0, current["max_abs_meV_A"] / 200.0)
        )
    all_mode_error = []
    for name, bounds in SEED_RANGES.items():
        members = indices_for(bounds)
        mode_error = np.einsum(
            "nat,nat->n",
            np.conj(modes[members]),
            prediction[members] - reference[members],
        )
        predicted_modes = np.einsum(
            "nat,nat->n", np.conj(modes[members]), base[members] + prediction[members]
        )
        target_modes = np.einsum(
            "nat,nat->n", np.conj(modes[members]), base[members] + reference[members]
        )
        predicted_slope = restoring_slope(coordinates[members], predicted_modes)
        target_slope = restoring_slope(coordinates[members], target_modes)
        slope_error = (predicted_slope - target_slope) / target_slope
        rms = float(1000.0 * np.sqrt(np.mean(np.abs(mode_error) ** 2)))
        output["Aprime_by_seed"][name] = {
            "RMS_meV_A": rms,
            "predicted_restoring_slope_eV_A2": predicted_slope,
            "target_restoring_slope_eV_A2": target_slope,
            "slope_relative_error": float(slope_error),
            "count": len(members),
        }
        ratios.extend((rms / 15.0, abs(slope_error) / 0.05))
        all_mode_error.append(mode_error)
    combined = float(
        1000.0 * np.sqrt(np.mean(np.abs(np.concatenate(all_mode_error)) ** 2))
    )
    output["Aprime_combined_RMS_meV_A"] = combined
    ratios.append(combined / 15.0)
    output["raw_gate_score"] = float(max(ratios))
    output["passes_fixed_gate"] = bool(all(value <= 1.0 for value in ratios))
    return output


def aggregate(args: argparse.Namespace) -> None:
    if len(args.fold_roots) != 4:
        raise ValueError("R2AG aggregate requires four ordered fold roots")
    package, package_receipt = load_package(args.package_root.resolve())
    expected_package_sha = package_receipt["package_sha256"]
    snapshot_epochs: tuple[int, ...] | None = None
    oof_by_epoch: np.ndarray | None = None
    fold_records = []
    seen: set[int] = set()
    for expected_fold, root in enumerate(args.fold_roots):
        root = root.resolve()
        receipt_path = root / "receipt.json"
        predictions_path = root / "predictions.npz"
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("status") != "R2AG_CONDITIONAL_MLP_FOLD_COMPLETE"
            or receipt.get("fold") != expected_fold
            or receipt["input_sha256"]["training_package"] != expected_package_sha
            or file_sha256(predictions_path) != receipt.get("predictions_sha256")
        ):
            raise ValueError(f"R2AG fold receipt changed: {expected_fold}")
        with np.load(predictions_path, allow_pickle=False) as arrays:
            epochs = tuple(int(value) for value in arrays["snapshot_epochs"])
            hold = np.asarray(arrays["hold_global_indices"], dtype=int)
            prediction = np.asarray(
                arrays["hold_predicted_force_eV_A"], dtype=np.float64
            )
        if not np.array_equal(hold, package["fold_hold_indices"][expected_fold]):
            raise ValueError(f"R2AG fold hold indices changed: {expected_fold}")
        if seen.intersection(int(value) for value in hold):
            raise ValueError("R2AG fold predictions overlap")
        seen.update(int(value) for value in hold)
        if snapshot_epochs is None:
            snapshot_epochs = epochs
            oof_by_epoch = np.full((len(epochs), 132, 72, 3), np.nan)
        elif epochs != snapshot_epochs:
            raise ValueError("R2AG fold snapshot epochs disagree")
        if prediction.shape != (len(epochs), 33, 72, 3):
            raise ValueError(f"R2AG fold prediction shape changed: {expected_fold}")
        assert oof_by_epoch is not None
        oof_by_epoch[:, hold] = prediction
        fold_records.append(
            {
                "fold": expected_fold,
                "receipt_sha256": file_sha256(receipt_path),
                "predictions_sha256": file_sha256(predictions_path),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }
        )
    if seen != set(range(132)) or snapshot_epochs is None or oof_by_epoch is None:
        raise ValueError("R2AG OOF folds do not cover all132")
    if not np.all(np.isfinite(oof_by_epoch)):
        raise ValueError("R2AG aggregated OOF contains non-finite values")
    snapshots = []
    for row, epoch in enumerate(snapshot_epochs):
        metrics = scalar_metrics(oof_by_epoch[row], package)
        snapshots.append({"epoch": epoch, "full_gate": metrics})
    snapshots.sort(
        key=lambda item: (item["full_gate"]["raw_gate_score"], item["epoch"])
    )
    best = snapshots[0]
    best_row = snapshot_epochs.index(int(best["epoch"]))
    status = (
        "R2AG_CONDITIONAL_MLP_OOF_GATE_PASSED_FINAL_FIT_PENDING"
        if best["full_gate"]["passes_fixed_gate"]
        else "R2AG_CONDITIONAL_MLP_OOF_GATE_FAILED"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "OOF_predictions.npz"
    np.savez(
        arrays_path,
        snapshot_epochs=np.asarray(snapshot_epochs, dtype="<i8"),
        OOF_predicted_force_eV_A=np.asarray(oof_by_epoch, dtype="<f8"),
        selected_epoch=np.asarray([best["epoch"]], dtype="<i8"),
        selected_OOF_predicted_force_eV_A=np.asarray(
            oof_by_epoch[best_row], dtype="<f8"
        ),
    )
    summary = {
        "format": "graphene_r2ag_seed012_conditional_mlp_outer4_OOF_v1",
        "status": status,
        "deployable": False,
        "scope": "seed0 + seed1 + promoted opened-development seed2 + T300 + T600 development; 525 K unopened",
        "selection_rule": "one shared epoch minimizes the complete fixed OOF gate score",
        "best": best,
        "snapshots_by_score": snapshots,
        "folds": fold_records,
        "training_package_sha256": expected_package_sha,
        "arrays_sha256": file_sha256(arrays_path),
        "selected_OOF_raw_sha256": r2r1.raw_array_sha256(
            oof_by_epoch[best_row], "<f8"
        ),
        "energy_labels_used": False,
        "unseen_525K_access": False,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(status + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    fold = sub.add_parser("fold")
    fold.add_argument("--fold", type=int, required=True)
    fold.add_argument("--epochs", type=int, default=40)
    fold.add_argument(
        "--snapshot-epochs", type=int, nargs="+", default=(0, 5, 10, 20, 40)
    )
    fold.add_argument("--device", default="cpu")
    fold.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    fold.add_argument("--output", type=Path, required=True)
    fold.add_argument("--log-every", type=int, default=5)
    combined = sub.add_parser("aggregate")
    combined.add_argument("--fold-roots", type=Path, nargs=4, required=True)
    combined.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    combined.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "fold":
        run_fold(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
