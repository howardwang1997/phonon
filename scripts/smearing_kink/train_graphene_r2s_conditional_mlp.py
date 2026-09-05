#!/usr/bin/env python3
"""Train/evaluate a gated nonlinear R2R readout on thermal92 only.

The R2Q encoder is frozen.  A small FP64 MLP consumes its signed even-l=0
node channels, ScaleShift node energy, and the multipolar amplitude field.
Its node energy is multiplied by the frozen multipolar gate before summation,
so the added scalar energy has an exact zero value/Jacobian/Hessian at the
reference and vanishes on exact rank-1 displacement fields.

This file implements conditional four-fold development OOF.  It does not read
seed1, small-H, support, energy labels, or any held trajectory, and it does not
publish a deployment checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ase.io import read

import graphene_r2o_taylor_null as r2o
import graphene_r2r1_linear_readout as r2r1
import graphene_r2r_multipolar_background as r2r
from diagnose_graphene_r2r1_aprime_objective import (
    DEFAULT_AGGREGATE,
    DEFAULT_FIT,
    _ridge,
    _parse_whitelist,
    canonical_json_bytes,
    file_sha256,
)
from graphene_r2r0_formal import FormalInputs, load_endpoint, read_geometry_only_extxyz


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/graphene_r2o_taylor_null_core"
ENDPOINT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2Q_four_step_trust_region/formal_4step_seed83_rtx"
)
DEFAULT_ROOT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background/R2S_conditional_mlp_development_20260826"
)

FORMAT = "graphene_r2s_gated_conditional_mlp_thermal92_development_v2_linear_skip"
SEED = 83
INPUT_WIDTH = 34
HIDDEN_WIDTHS = (32, 16)
ENERGY_SCALE_EV = 0.030
FORCE_SCALE_EV_A = 0.030
APRIME_SCALE_EV_A = 0.015
FORCE_OBJECTIVE_MASS = 0.50
APRIME_OBJECTIVE_MASS = 0.50
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-6
GRADIENT_CLIP = 10.0


def recommended_inputs() -> FormalInputs:
    return FormalInputs(
        endpoint_checkpoint=ENDPOINT / "endpoint.pt",
        endpoint_receipt=ENDPOINT / "endpoint_receipt.json",
        endpoint_marker=ENDPOINT / "ENDPOINT_FROZEN",
        reference_6x6=DATA / "reference_6x6.xyz",
        reference_8x8=DATA / "reference_8x8.xyz",
        thermal92=DATA / "train_thermal.xyz",
        harmonic_zero32=DATA / "train_harmonic_lambda1_small_zero.xyz",
    )


class ConditionalMLP(torch.nn.Module):
    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        linear_coefficient: torch.Tensor,
    ) -> None:
        super().__init__()
        if feature_mean.shape != (INPUT_WIDTH,) or feature_scale.shape != (INPUT_WIDTH,):
            raise ValueError("R2S feature scaler shape changed")
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        if linear_coefficient.shape != (r2r1.LINEAR_WIDTH,):
            raise ValueError("R2S frozen linear skip width changed")
        self.register_buffer(
            "linear_coefficient", linear_coefficient.detach().clone()
        )
        self.linear1 = torch.nn.Linear(INPUT_WIDTH, HIDDEN_WIDTHS[0], bias=True)
        self.linear2 = torch.nn.Linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1], bias=True)
        self.linear3 = torch.nn.Linear(HIDDEN_WIDTHS[1], 1, bias=True)
        self.activation = torch.nn.SiLU()
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)
        # Zero output makes the initial physical correction exactly zero while
        # allowing the last layer to receive a gradient on the first update.
        torch.nn.init.zeros_(self.linear3.weight)
        torch.nn.init.zeros_(self.linear3.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        hidden = self.activation(self.linear1(normalized))
        hidden = self.activation(self.linear2(hidden))
        return ENERGY_SCALE_EV * self.linear3(hidden).squeeze(-1)


@dataclass
class CachedContext:
    global_index: int
    context: Any


def _node_features(observables: Any, fields: Any) -> torch.Tensor:
    output = torch.cat(
        (
            observables.signed_l0,
            observables.scale_shift_node_energy_eV[:, None],
            fields.amplitude_gate[:, None],
        ),
        dim=1,
    )
    if output.shape[1] != INPUT_WIDTH:
        raise ValueError("R2S conditional feature width changed")
    return output


def _build_contexts(
    endpoint: torch.nn.Module,
    structures: Sequence[Any],
    reference: Any,
    indices: Sequence[int],
    device: str,
) -> dict[int, CachedContext]:
    output: dict[int, CachedContext] = {}
    for global_index in indices:
        context = r2r._verified_production_context(
            endpoint,
            structures[int(global_index)],
            reference,
            device=device,
        )
        output[int(global_index)] = CachedContext(int(global_index), context)
    return output


def _feature_scaler(
    contexts: dict[int, CachedContext], train_indices: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    values = []
    with torch.enable_grad():
        for index in train_indices:
            context = contexts[int(index)].context
            observables = context.node_observables_fn(
                context.current_positions_reference_order
            )
            fields = context.background_fields_fn(
                context.current_positions_reference_order
            )
            values.append(_node_features(observables, fields).detach())
    joined = torch.cat(values, dim=0)
    mean = joined.mean(dim=0)
    scale = torch.sqrt(torch.mean(torch.square(joined - mean), dim=0))
    floor = torch.max(scale) * 1.0e-10
    if torch.any(scale <= floor):
        raise ValueError("R2S train-only feature scaler has a zero column")
    receipt = {
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
    return mean, scale, receipt


def _tail_energy_force(
    readout: ConditionalMLP,
    cached: CachedContext,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = cached.context
    positions = context.current_positions_reference_order
    positions.grad = None
    observables = context.node_observables_fn(positions)
    fields = context.background_fields_fn(positions)
    features = _node_features(observables, fields)
    linear_basis = context.parameter_basis_fn(observables, fields)
    energy = torch.dot(linear_basis, readout.linear_coefficient) + torch.sum(
        fields.multipolar_gate * readout(features)
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


def _group_counts(indices: Sequence[int]) -> dict[str, int]:
    return {
        group: sum(r2r1.group_for_global_index(int(index)) == group for index in indices)
        for group in r2r1.THERMAL_GROUP_RANGES
    }


def _train_epoch(
    readout: ConditionalMLP,
    endpoint: torch.nn.Module,
    contexts: dict[int, CachedContext],
    indices: Sequence[int],
    fixed: torch.Tensor,
    reference: torch.Tensor,
    aprime_mode: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
) -> dict[str, float]:
    readout.train()
    optimizer.zero_grad(set_to_none=True)
    counts = _group_counts(indices)
    order = list(int(index) for index in indices)
    rng.shuffle(order)
    force_total = 0.0
    aprime_total = 0.0
    for index in order:
        _, tail_force = _tail_energy_force(
            readout, contexts[index], create_graph=True
        )
        predicted = fixed[index] + tail_force
        error = predicted - reference[index]
        group = r2r1.group_for_global_index(index)
        force_loss = torch.mean(torch.square(error / FORCE_SCALE_EV_A))
        weighted_force = (
            FORCE_OBJECTIVE_MASS
            * r2r1.GROUP_MASSES[group]
            / counts[group]
            * force_loss
        )
        loss = weighted_force
        force_total += float(weighted_force.detach().cpu())
        if index < 20:
            projected = torch.sum(torch.conj(aprime_mode) * error.to(torch.complex128))
            aprime_loss = (
                torch.square(projected.real) + torch.square(projected.imag)
            ) / (APRIME_SCALE_EV_A**2)
            weighted_aprime = APRIME_OBJECTIVE_MASS / counts["E50_seed0"] * aprime_loss
            loss = loss + weighted_aprime
            aprime_total += float(weighted_aprime.detach().cpu())
        loss.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(readout.parameters(), GRADIENT_CLIP).detach().cpu()
    )
    optimizer.step()
    if r2o.state_dict_sha256(endpoint) != r2r.R2Q_ENDPOINT_STATE_SHA256:
        raise RuntimeError("R2S training changed the frozen endpoint")
    return {
        "force_objective": force_total,
        "Aprime_objective": aprime_total,
        "total_objective": force_total + aprime_total,
        "gradient_norm_before_clip": gradient_norm,
    }


def _predict_indices(
    readout: ConditionalMLP,
    contexts: dict[int, CachedContext],
    indices: Sequence[int],
    fixed: torch.Tensor,
) -> np.ndarray:
    readout.eval()
    output = np.full((92, 72, 3), np.nan, dtype=np.float64)
    for index in indices:
        with torch.enable_grad():
            _, tail_force = _tail_energy_force(
                readout, contexts[int(index)], create_graph=False
            )
        output[int(index)] = (
            fixed[int(index)] + tail_force
        ).detach().cpu().numpy()
    return output


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_fold(args: argparse.Namespace) -> None:
    if args.fold not in range(4):
        raise ValueError("fold must be 0..3")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    inputs = recommended_inputs()
    endpoint = load_endpoint(inputs, args.device)
    for parameter in endpoint.parameters():
        parameter.requires_grad_(False)
    endpoint.eval()
    structures = read_geometry_only_extxyz(inputs.thermal92)
    reference_atoms = read(inputs.reference_6x6, index=0)
    reference_force, aprime, label_hashes = _parse_whitelist(inputs.thermal92)
    if file_sha256(args.aggregate.resolve()) != r2r1.ATTEMPT3_AGGREGATE_ARRAYS_SHA256:
        raise ValueError("R2S aggregate SHA256 changed")
    with np.load(args.aggregate, allow_pickle=False) as arrays:
        fixed_np = np.asarray(arrays["thermal_fixed_force_eV_A"], np.float64)
        design_np = np.asarray(
            arrays["thermal_parameter_force_design_eV_A"], np.float64
        )

    hold = np.asarray(r2r1.FOLD_GLOBAL_INDICES[args.fold], dtype=int)
    train = np.setdiff1d(np.arange(92, dtype=int), hold)
    required = np.sort(np.concatenate((train, hold)))
    contexts = _build_contexts(endpoint, structures, reference_atoms, required, args.device)
    mean, scale, scaler_receipt = _feature_scaler(contexts, train)
    r2r1_receipt = json.loads((args.r2r1_fit_root / "fit_receipt.json").read_text())
    selected_alpha = float(
        r2r1_receipt["pipeline_receipt"]["nested_OOF"]["outer_records"][args.fold][
            "selected_alpha"
        ]
    )
    linear_coefficient_np = _ridge(
        design_np,
        fixed_np,
        reference_force,
        aprime,
        train,
        selected_alpha,
        0.0,
    )
    linear_coefficient = torch.as_tensor(
        linear_coefficient_np, device=args.device, dtype=torch.float64
    )
    readout = ConditionalMLP(mean, scale, linear_coefficient).to(
        device=args.device, dtype=torch.float64
    )
    optimizer = torch.optim.AdamW(
        readout.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    fixed = torch.as_tensor(fixed_np, device=args.device, dtype=torch.float64)
    target = torch.as_tensor(reference_force, device=args.device, dtype=torch.float64)
    mode = torch.as_tensor(
        aprime.mode_real[0] + 1.0j * aprime.mode_imag[0],
        device=args.device,
        dtype=torch.complex128,
    )
    initial_hold_prediction = _predict_indices(readout, contexts, hold, fixed)
    with np.load(args.r2r1_fit_root / "fit_arrays.npz", allow_pickle=False) as arrays:
        published_oof = np.asarray(
            arrays["OOF_predicted_force_eV_A"], dtype=np.float64
        )
    initial_replay_max_abs = float(
        np.max(np.abs(initial_hold_prediction[hold] - published_oof[hold]))
    )
    if initial_replay_max_abs > 2.0e-10:
        raise ValueError("R2S frozen linear skip does not replay terminal R2R-1 OOF")
    initial_hold_metrics = r2r1.gate_metrics(
        initial_hold_prediction, reference_force, aprime, hold
    )
    rng = random.Random(SEED + args.fold)
    history = []
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        record = _train_epoch(
            readout,
            endpoint,
            contexts,
            train,
            fixed,
            target,
            mode,
            optimizer,
            rng,
        )
        record.update({"epoch": epoch, "wall_seconds": time.perf_counter() - tick})
        history.append(record)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps(record, sort_keys=True), flush=True)

    train_prediction = _predict_indices(readout, contexts, train, fixed)
    hold_prediction = _predict_indices(readout, contexts, hold, fixed)
    train_metrics = r2r1.gate_metrics(
        train_prediction, reference_force, aprime, train
    )
    hold_metrics = r2r1.gate_metrics(
        hold_prediction, reference_force, aprime, hold
    )
    checkpoint = {
        "format": FORMAT,
        "kind": "conditional_development_fold_not_deployable",
        "fold": args.fold,
        "epochs": args.epochs,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in readout.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": {
            "input_width": INPUT_WIDTH,
            "hidden_widths": HIDDEN_WIDTHS,
            "activation": "SiLU",
            "energy_scale_eV": ENERGY_SCALE_EV,
            "gate": "multipolar_gate_a",
            "frozen_linear_skip": True,
        },
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    np.savez(
        output / "predictions.npz",
        hold_global_indices=hold,
        hold_predicted_force_eV_A=hold_prediction[hold],
        train_global_indices=train,
        train_predicted_force_eV_A=train_prediction[train],
    )
    receipt = {
        "format": FORMAT,
        "status": "R2S_CONDITIONAL_MLP_FOLD_COMPLETE",
        "deployable": False,
        "fold": args.fold,
        "epochs": args.epochs,
        "seed": SEED,
        "device": args.device,
        "train_global_indices": train.tolist(),
        "hold_global_indices": hold.tolist(),
        "objective": {
            "force_mass": FORCE_OBJECTIVE_MASS,
            "Aprime_mass": APRIME_OBJECTIVE_MASS,
            "group_masses_within_force": r2r1.GROUP_MASSES,
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
        "frozen_linear_skip": {
            "selected_alpha_from_terminal_nested_OOF": selected_alpha,
            "coefficient_raw_sha256": r2r1.raw_array_sha256(
                linear_coefficient_np, "<f8"
            ),
            "initial_terminal_OOF_replay_max_abs_eV_A": initial_replay_max_abs,
            "initial_hold_metrics": initial_hold_metrics,
        },
        "train_metrics": train_metrics,
        "hold_metrics": hold_metrics,
        "history": history,
        "readout_state_sha256": _state_sha256(readout),
        "endpoint_state_sha256": r2o.state_dict_sha256(endpoint),
        "input_sha256": {
            "endpoint": file_sha256(inputs.endpoint_checkpoint),
            "thermal92": file_sha256(inputs.thermal92),
            "aggregate_arrays": file_sha256(args.aggregate),
            "reference_6x6": file_sha256(inputs.reference_6x6),
            "terminal_R2R1_fit_receipt": file_sha256(
                args.r2r1_fit_root / "fit_receipt.json"
            ),
            "terminal_R2R1_fit_arrays": file_sha256(
                args.r2r1_fit_root / "fit_arrays.npz"
            ),
        },
        "label_raw_sha256": label_hashes,
        "energy_labels_used": False,
        "development_or_held_access": False,
        "elapsed_seconds": time.time() - started,
    }
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    (output / "DONE").write_text("R2S_CONDITIONAL_MLP_FOLD_COMPLETE\n")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    if len(args.fold_roots) != 4:
        raise ValueError("aggregate requires exactly four fold roots")
    reference_force, aprime, label_hashes = _parse_whitelist(
        recommended_inputs().thermal92
    )
    oof = np.full_like(reference_force, np.nan)
    receipts = []
    seen: set[int] = set()
    for expected_fold, root in enumerate(args.fold_roots):
        root = root.resolve()
        if (root / "DONE").read_text() != "R2S_CONDITIONAL_MLP_FOLD_COMPLETE\n":
            raise ValueError(f"fold {expected_fold} is not terminal")
        receipt = json.loads((root / "receipt.json").read_text())
        if receipt.get("fold") != expected_fold or receipt.get("status") != "R2S_CONDITIONAL_MLP_FOLD_COMPLETE":
            raise ValueError(f"fold receipt order/status mismatch: {expected_fold}")
        with np.load(root / "predictions.npz", allow_pickle=False) as arrays:
            indices = np.asarray(arrays["hold_global_indices"], dtype=int)
            prediction = np.asarray(arrays["hold_predicted_force_eV_A"], np.float64)
        if tuple(indices.tolist()) != r2r1.FOLD_GLOBAL_INDICES[expected_fold]:
            raise ValueError(f"fold indices changed: {expected_fold}")
        if seen.intersection(int(value) for value in indices):
            raise ValueError("fold predictions overlap")
        seen.update(int(value) for value in indices)
        oof[indices] = prediction
        receipts.append(receipt)
    if seen != set(range(92)) or not np.all(np.isfinite(oof)):
        raise ValueError("fold predictions do not cover thermal92 exactly")
    metrics = r2r1.gate_metrics(oof, reference_force, aprime, np.arange(92))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "OOF_predictions.npz", OOF_predicted_force_eV_A=oof)
    summary = {
        "format": "graphene_r2s_conditional_mlp_outer4_OOF_v1",
        "status": (
            "R2S_CONDITIONAL_MLP_OOF_PASSED_DEVELOPMENT_ONLY"
            if metrics["passes_fixed_gate"]
            else "R2S_CONDITIONAL_MLP_OOF_FAILED"
        ),
        "deployable": False,
        "pooled_OOF_metrics": metrics,
        "OOF_prediction_raw_sha256": r2r1.raw_array_sha256(oof, "<f8"),
        "fold_receipt_sha256": [file_sha256(root / "receipt.json") for root in args.fold_roots],
        "fold_readout_state_sha256": [item["readout_state_sha256"] for item in receipts],
        "label_raw_sha256": label_hashes,
        "energy_labels_used": False,
        "development_or_held_access": False,
        "interpretation": "conditional readout OOF; frozen encoder saw related corpus",
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    fold = sub.add_parser("fold")
    fold.add_argument("--fold", type=int, required=True)
    fold.add_argument("--epochs", type=int, default=240)
    fold.add_argument("--device", default="cpu")
    fold.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    fold.add_argument("--r2r1-fit-root", type=Path, default=DEFAULT_FIT)
    fold.add_argument("--output", type=Path, required=True)
    fold.add_argument("--log-every", type=int, default=10)
    combined = sub.add_parser("aggregate")
    combined.add_argument("--fold-roots", type=Path, nargs=4, required=True)
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
