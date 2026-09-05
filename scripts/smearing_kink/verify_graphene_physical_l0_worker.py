#!/usr/bin/env python3
"""Verify an isolated V100 worker deployment against the frozen L0 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def require_hash(path: Path, expected: str, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")
    return {"path": str(path), "sha256": actual}


def scalar(data, key: str):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} is not scalar")
    return value.reshape(()).item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--short-tdep", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    s0 = (
        root
        / "results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short"
    )
    freeze_path = s0 / "L0_classical_tdep/freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "frozen_for_L0_development":
        raise ValueError("invalid L0 freeze status")
    if args.temperature not in freeze["development_temperatures_K"]:
        raise ValueError("worker temperature is outside the frozen development set")
    if args.seed not in freeze["trajectory_protocol"]["seeds"]:
        raise ValueError("worker seed is outside the frozen protocol")
    if freeze["locked_validation_temperatures_K_not_accessed"] != [375, 525]:
        raise ValueError("validation boundary changed")

    formal = s0 / "formal_240ep_2060_seed83_replayw16"
    assets = {
        "freeze_manifest": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
        "base_model": require_hash(
            root / "results/gr_backbone_v11/ft_graphene.model",
            freeze["base_model"]["sha256"],
            "base model",
        ),
        "delta_model": require_hash(
            formal / "selected_checkpoint.model",
            freeze["delta_model"]["sha256"],
            "selected delta model",
        ),
        "background": require_hash(
            root / "results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml",
            freeze["background"]["sha256"],
            "phonon background",
        ),
    }
    operator_record = freeze["operators"][str(args.temperature)]
    if operator_record["degauss_formula"] != "k_B*T/Ry":
        raise ValueError("operator does not use the frozen degauss formula")
    operator_path = (
        root / f"data/graphene_physical_s0/operators/T{args.temperature}_operator.npz"
    )
    assets["operator"] = require_hash(
        operator_path, operator_record["sha256"], "physical operator"
    )

    local_scripts = {
        "sampling": root / "scripts/smearing_kink/td_phonon_friedel.py",
        "short_tdep_recompute": (
            root / "scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py"
        ),
        "point_gate": root / "scripts/smearing_kink/summarize_graphene_physical_s0_l0.py",
        "bootstrap_gate": (
            root / "scripts/smearing_kink/bootstrap_graphene_fd_conditioned_sampling.py"
        ),
    }
    assets["scripts"] = {
        label: require_hash(path, freeze["scripts"][label]["sha256"], label)
        for label, path in local_scripts.items()
    }

    snapshots = args.checkpoint / "snapshots.npz"
    state = args.checkpoint / "state.npz"
    if not snapshots.is_file() or not state.is_file():
        raise FileNotFoundError("checkpoint requires snapshots.npz and state.npz")
    with np.load(snapshots, allow_pickle=False) as data:
        count = len(np.asarray(data["positions"]))
    if count < freeze["trajectory_protocol"]["pilot_snapshots_per_seed"]:
        raise ValueError("worker checkpoint is shorter than the accepted pilot")
    if count > freeze["trajectory_protocol"]["final_snapshots_per_seed"]:
        raise ValueError("worker checkpoint exceeds the frozen final count")
    if args.expected_count is not None and count != args.expected_count:
        raise ValueError(f"checkpoint has {count} snapshots, expected {args.expected_count}")

    result = {
        "status": "passed",
        "scope": "isolated V100 L0 worker deployment/result verification",
        "temperature_K": args.temperature,
        "seed": args.seed,
        "snapshot_count": count,
        "checkpoint": {
            "path": str(args.checkpoint),
            "snapshots_sha256": sha256(snapshots),
            "state_sha256": sha256(state),
        },
        "assets": assets,
    }
    if args.short_tdep is not None:
        if not args.short_tdep.is_file():
            raise FileNotFoundError(args.short_tdep)
        with np.load(args.short_tdep, allow_pickle=False) as data:
            prefix = f"T{args.temperature}"
            source_counts = np.asarray(data["source_checkpoint_counts"], int).tolist()
            operator_hash = str(scalar(data, "subtracted_operator_sha256"))
            mean_temperature = float(scalar(data, f"{prefix}_mean_temperature_K"))
            fit_rmse = float(scalar(data, f"{prefix}_fit_rmse_meV_A"))
        if source_counts != [count]:
            raise ValueError("short-TDEP source count differs from the checkpoint")
        if operator_hash != operator_record["sha256"]:
            raise ValueError("short-TDEP subtracted a different physical operator")
        result["short_tdep"] = {
            "path": str(args.short_tdep),
            "sha256": sha256(args.short_tdep),
            "source_checkpoint_counts": source_counts,
            "mean_temperature_K": mean_temperature,
            "fit_RMSE_meV_A": fit_rmse,
        }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
