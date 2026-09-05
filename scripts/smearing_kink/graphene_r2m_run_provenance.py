#!/usr/bin/env python3
"""Fail-closed launcher and completion provenance for R2M outer folds."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


STATUS = "R2M_routed_tail_outer_fold_launcher_frozen_before_training"
POLICY = "fixed_epoch_240_no_support_checkpoint_selection"
CODE_NAMES = (
    "run_graphene_r2m_routed_tail_outer_fold.sh",
    "graphene_r2m_run_provenance.py",
    "graphene_r2m_routed_tail.py",
    "train_graphene_r2m_routed_tail_outer_fold.py",
    "evaluate_graphene_r2m_routed_tail_outer_fold.py",
    "graphene_r2m_aprime_eval.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def reject_seed2(path: Path) -> None:
    if any(
        token in str(path).lower()
        for token in ("seed2", "reserved_e50", "reserved-e50")
    ):
        raise ValueError(f"routed-tail provenance rejects seed2 path: {path}")


def exact_file(path: Path, *, allow_empty: bool = False) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or symlinked file: {path}")
    if not allow_empty and path.stat().st_size == 0:
        raise ValueError(f"empty file: {path}")


def record(path: Path) -> dict:
    exact_file(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def same_path(recorded: dict, path: Path) -> bool:
    return Path(str(recorded.get("path", ""))).resolve() == path.resolve()


def read_json(path: Path) -> dict:
    exact_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def current_environment() -> dict:
    mace_version = importlib.metadata.version("mace-torch")
    if mace_version != "0.3.16":
        raise ValueError(f"R2M routed tail requires mace-torch==0.3.16, got {mace_version}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("R2M routed tail requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("R2M routed tail requires exactly one visible CUDA device")
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "mace_torch_version": mace_version,
        "CUDA_VISIBLE_DEVICES": "0",
        "cuda_available": True,
        "visible_cuda_device_count": 1,
        "visible_device_0_name": torch.cuda.get_device_name(0),
        "visible_device_0_capability": list(torch.cuda.get_device_capability(0)),
        "torch_default_dtype": str(torch.get_default_dtype()),
    }


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "launcher_source": args.launcher_source,
        "prepared_manifest": args.prepared_manifest,
        "fold_manifest": args.fold_dir / "fold_manifest.json",
        "core": args.core_model,
        "replay_train": args.replay_train,
        "replay_valid_post_freeze_only": args.replay_valid,
        "pristine": args.pristine,
        "operator_post_freeze_only": args.operator,
        "background_post_freeze_only": args.background,
        "thermal_result_post_freeze_only": args.thermal_result,
    }


def validate_source_contract(args: argparse.Namespace) -> tuple[dict, dict]:
    paths = source_paths(args)
    for path in [*paths.values(), args.fold_dir, args.code_dir]:
        reject_seed2(path)
    if len(args.core_sha256) != 64 or args.core_sha256.lower() != args.core_sha256:
        raise ValueError("core SHA-256 must be 64 lowercase hexadecimal characters")
    int(args.core_sha256, 16)
    for path in paths.values():
        exact_file(path)
    prepared = read_json(args.prepared_manifest)
    fold_path = args.fold_dir / "fold_manifest.json"
    fold = read_json(fold_path)
    if (
        prepared.get("status")
        != "R2M_routed_tail_nine_outer_folds_frozen_before_training"
        or prepared.get("n_outer_folds") != 9
        or prepared.get("selection_policy") != POLICY
        or prepared.get("E50_seed2_read_or_accepted_as_input") is not False
    ):
        raise ValueError("wrong prepared outer-LOCO manifest")
    entries = prepared.get("folds", [])
    if len(entries) != 9 or {int(item["outer_fold"]) for item in entries} != set(
        range(9)
    ):
        raise ValueError("prepared manifest does not contain exactly folds 0..8")
    if (
        fold.get("status") != "R2M_routed_tail_outer_fold_frozen_before_training"
        or fold.get("selection_policy") != POLICY
        or fold.get("strict_nested_inner_CV_run") is not False
        or fold.get("leakage", {}).get("E50_seed2_read") is not False
        or fold.get("leakage", {}).get(
            "held_geometry_or_label_in_scaler_gradient_or_selection"
        )
        is not False
    ):
        raise ValueError("wrong or unsafe fold manifest")
    outer_fold = int(fold["outer_fold"])
    held_sscha = int(fold["held_sscha_index"])
    expected_entry = next(item for item in entries if int(item["outer_fold"]) == outer_fold)
    if (
        int(expected_entry["held_sscha_index"]) != held_sscha
        or not same_path({"path": expected_entry["directory"]}, args.fold_dir)
        or expected_entry["manifest_sha256"] != sha256(fold_path)
        or int(prepared["support_order_sscha_index"][outer_fold]) != held_sscha
    ):
        raise ValueError("fold directory/hash/held identity differs from prepared manifest")
    if sha256(args.core_model) != args.core_sha256:
        raise ValueError("core SHA-256 mismatch")
    if (
        fold["inputs"]["core"]["sha256"] != args.core_sha256
        or not same_path(fold["inputs"]["core"], args.core_model)
    ):
        raise ValueError("fold was prepared against a different core")
    fold_to_arg = {
        "replay_train": args.replay_train,
        "replay_valid_post_freeze_only": args.replay_valid,
        "pristine": args.pristine,
        "operator_post_freeze_only": args.operator,
        "background_post_freeze_only": args.background,
        "thermal_result_post_freeze_only": args.thermal_result,
    }
    for name, path in fold_to_arg.items():
        item = fold["inputs"][name]
        if item["sha256"] != sha256(path) or not same_path(item, path):
            raise ValueError(f"fold input changed: {name}")
    if set(path.name for path in args.code_dir.iterdir()) != set(CODE_NAMES):
        raise ValueError("launcher code snapshot directory has unexpected contents")
    for name in CODE_NAMES:
        exact_file(args.code_dir / name)
    if sha256(args.launcher_source) != sha256(
        args.code_dir / "run_graphene_r2m_routed_tail_outer_fold.sh"
    ):
        raise ValueError("executed launcher source differs from its frozen snapshot")
    helper = args.code_dir / "graphene_r2m_aprime_eval.py"
    if sha256(helper) != fold["inputs"]["aprime_evaluation_helper"]["sha256"]:
        raise ValueError("A-prime evaluation helper differs from prepared definition")
    return prepared, fold


def freeze_launcher(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to replace launcher freeze: {args.output}")
    _, fold = validate_source_contract(args)
    payload = {
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": int(fold["outer_fold"]),
        "held_sscha_index": int(fold["held_sscha_index"]),
        "selection_policy": POLICY,
        "fixed_endpoint": {"epochs": 240, "state": "EMA"},
        "conda_environment": args.conda_env,
        "runtime_environment": current_environment(),
        "E50_seed2_read_or_accepted_as_input": False,
        "inputs": {name: record(path) for name, path in source_paths(args).items()},
        "code_snapshots": {
            name: record(args.code_dir / name) for name in CODE_NAMES
        },
    }
    strict_json(args.output, payload)


def validate_freeze_record(path: Path) -> dict:
    freeze = read_json(path)
    if (
        freeze.get("status") != STATUS
        or freeze.get("selection_policy") != POLICY
        or freeze.get("fixed_endpoint") != {"epochs": 240, "state": "EMA"}
        or freeze.get("E50_seed2_read_or_accepted_as_input") is not False
        or not str(freeze.get("conda_environment", ""))
        or freeze.get("runtime_environment") != current_environment()
    ):
        raise ValueError("invalid launcher freeze contract")
    if set(freeze.get("code_snapshots", {})) != set(CODE_NAMES):
        raise ValueError("launcher freeze has the wrong code snapshot set")
    for collection in (freeze["inputs"], freeze["code_snapshots"]):
        for name, item in collection.items():
            item_path = Path(item["path"])
            reject_seed2(item_path)
            exact_file(item_path)
            if sha256(item_path) != item["sha256"]:
                raise ValueError(f"frozen launcher input changed: {name}")
    prepared = read_json(Path(freeze["inputs"]["prepared_manifest"]["path"]))
    fold_path = Path(freeze["inputs"]["fold_manifest"]["path"])
    fold = read_json(fold_path)
    outer_fold = int(freeze["outer_fold"])
    matches = [
        item for item in prepared.get("folds", []) if int(item["outer_fold"]) == outer_fold
    ]
    if (
        len(matches) != 1
        or matches[0]["manifest_sha256"] != sha256(fold_path)
        or int(matches[0]["held_sscha_index"]) != int(freeze["held_sscha_index"])
        or int(fold["outer_fold"]) != outer_fold
        or int(fold["held_sscha_index"]) != int(freeze["held_sscha_index"])
        or fold["inputs"]["core"]["sha256"]
        != freeze["inputs"]["core"]["sha256"]
    ):
        raise ValueError("launcher freeze no longer matches prepared fold")
    helper = freeze["code_snapshots"]["graphene_r2m_aprime_eval.py"]
    if helper["sha256"] != fold["inputs"]["aprime_evaluation_helper"]["sha256"]:
        raise ValueError("launcher freeze changed the prepared A-prime definition")
    return freeze


def validate_expected(args: argparse.Namespace) -> dict:
    freeze = validate_freeze_record(args.freeze)
    expected = source_paths(args)
    for name, path in expected.items():
        if not same_path(freeze["inputs"][name], path):
            raise ValueError(f"launcher environment path differs from freeze: {name}")
    if freeze["inputs"]["core"]["sha256"] != args.core_sha256:
        raise ValueError("launcher environment core hash differs from freeze")
    if Path(freeze["code_snapshots"][CODE_NAMES[0]]["path"]).parent != args.code_dir.resolve():
        raise ValueError("launcher environment code directory differs from freeze")
    if freeze["conda_environment"] != args.conda_env:
        raise ValueError("launcher Conda environment differs from freeze")
    return freeze


def validate_evaluation(path: Path, freeze: dict) -> dict:
    summary = read_json(path)
    if (
        summary.get("status")
        != "R2M_routed_tail_outer_fold_post_freeze_evaluation_complete"
        or int(summary.get("outer_fold", -1)) != int(freeze["outer_fold"])
        or int(summary.get("held_sscha_index", -1))
        != int(freeze["held_sscha_index"])
        or summary.get("selection_policy") != POLICY
        or summary.get("held_opened_only_after_epoch240_EMA_hash_freeze") is not True
        or summary.get("training_provenance_verified_before_held_read") is not True
        or summary.get("E50_seed2_read") is not False
        or summary.get("full_composite_deployment_authorized") is not False
    ):
        raise ValueError("evaluation is incomplete or not bound to launcher freeze")
    artifacts = summary.get("artifacts", {})
    required = {
        "core",
        "fold_manifest",
        "tail",
        "arrays",
        "training_summary",
        "training_freeze",
        "training_metrics",
        "TRAINING_DONE",
        "launcher_freeze",
    }
    if not required.issubset(artifacts):
        raise ValueError("evaluation provenance is incomplete")
    for name in required:
        item = artifacts[name]
        item_path = Path(item["path"])
        exact_file(item_path, allow_empty=(name == "TRAINING_DONE"))
        if sha256(item_path) != item["sha256"]:
            raise ValueError(f"evaluation artifact changed: {name}")
    expected_launcher_freeze = (
        Path(
            freeze["code_snapshots"]["graphene_r2m_run_provenance.py"]["path"]
        ).parent.parent
        / "launcher_freeze.json"
    )
    if (
        not same_path(artifacts["launcher_freeze"], expected_launcher_freeze)
        or artifacts["launcher_freeze"]["sha256"]
        != sha256(expected_launcher_freeze)
    ):
        raise ValueError("evaluation points to a different launcher freeze")
    return summary


def write_completion(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to replace completion manifest: {args.output}")
    freeze = validate_freeze_record(args.freeze)
    evaluation = validate_evaluation(args.evaluation, freeze)
    training_summary = Path(evaluation["artifacts"]["training_summary"]["path"])
    payload = {
        "status": "R2M_routed_tail_outer_fold_execution_complete",
        "outer_fold": int(freeze["outer_fold"]),
        "held_sscha_index": int(freeze["held_sscha_index"]),
        "launcher_freeze": record(args.freeze),
        "training_summary": record(training_summary),
        "evaluation": record(args.evaluation),
    }
    strict_json(args.output, payload)


def validate_complete(args: argparse.Namespace) -> None:
    freeze = validate_freeze_record(args.freeze)
    if args.failed.exists():
        raise ValueError("FAILED marker coexists with DONE")
    if args.running.exists():
        raise ValueError("RUNNING marker coexists with DONE")
    exact_file(args.done, allow_empty=True)
    if args.done.stat().st_size != 0:
        raise ValueError("DONE marker must be empty")
    exact_file(args.exit_code)
    if args.exit_code.read_text(encoding="utf-8").strip() != "0":
        raise ValueError("prior launcher exit status was not zero")
    validate_completion_manifest(args.completion, args.freeze, freeze)


def validate_completion_manifest(
    completion_path: Path, freeze_path: Path, freeze: dict | None = None
) -> dict:
    if freeze is None:
        freeze = validate_freeze_record(freeze_path)
    completion = read_json(completion_path)
    if (
        completion.get("status") != "R2M_routed_tail_outer_fold_execution_complete"
        or int(completion.get("outer_fold", -1)) != int(freeze["outer_fold"])
        or int(completion.get("held_sscha_index", -1))
        != int(freeze["held_sscha_index"])
        or completion.get("launcher_freeze", {}).get("sha256")
        != sha256(freeze_path)
        or not same_path(completion.get("launcher_freeze", {}), freeze_path)
    ):
        raise ValueError("completion manifest identity differs from launcher freeze")
    evaluation_path = Path(completion["evaluation"]["path"])
    if sha256(evaluation_path) != completion["evaluation"]["sha256"]:
        raise ValueError("completed evaluation changed")
    evaluation = validate_evaluation(evaluation_path, freeze)
    training_summary = Path(evaluation["artifacts"]["training_summary"]["path"])
    if (
        not same_path(completion.get("training_summary", {}), training_summary)
        or completion.get("training_summary", {}).get("sha256")
        != sha256(training_summary)
    ):
        raise ValueError("completed training summary changed")
    return completion


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--launcher-source", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--core-model", type=Path, required=True)
    parser.add_argument("--core-sha256", required=True)
    parser.add_argument("--replay-train", type=Path, required=True)
    parser.add_argument("--replay-valid", type=Path, required=True)
    parser.add_argument("--pristine", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--thermal-result", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--conda-env", required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    add_source_arguments(freeze_parser)
    freeze_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    add_source_arguments(validate_parser)
    validate_parser.add_argument("--freeze", type=Path, required=True)
    completion_parser = commands.add_parser("write-completion")
    completion_parser.add_argument("--freeze", type=Path, required=True)
    completion_parser.add_argument("--evaluation", type=Path, required=True)
    completion_parser.add_argument("--output", type=Path, required=True)
    done_parser = commands.add_parser("validate-complete")
    done_parser.add_argument("--freeze", type=Path, required=True)
    done_parser.add_argument("--completion", type=Path, required=True)
    done_parser.add_argument("--done", type=Path, required=True)
    done_parser.add_argument("--exit-code", type=Path, required=True)
    done_parser.add_argument("--failed", type=Path, required=True)
    done_parser.add_argument("--running", type=Path, required=True)
    recovery_parser = commands.add_parser("validate-completion")
    recovery_parser.add_argument("--freeze", type=Path, required=True)
    recovery_parser.add_argument("--completion", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_launcher(args)
    elif args.command == "validate":
        validate_expected(args)
    elif args.command == "write-completion":
        write_completion(args)
    elif args.command == "validate-complete":
        validate_complete(args)
    else:
        validate_completion_manifest(args.completion, args.freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
