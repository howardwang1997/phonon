#!/usr/bin/env python3
"""CLI for one R2R-0 shard, V100 mechanics audit, or bounded preflight."""
from __future__ import annotations

import argparse
from pathlib import Path

from graphene_r2r0_formal import (
    FormalInputs,
    run_bounded_preflight,
    run_mechanics,
    run_shard,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/graphene_r2o_taylor_null_core"
ENDPOINT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2Q_four_step_trust_region/formal_4step_seed83_rtx"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("mode", choices=("shard", "mechanics", "preflight"))
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--device", default="cpu")
    value.add_argument("--shard-id", type=int)
    value.add_argument("--preflight-kind", choices=("synthetic", "real"))
    value.add_argument("--freeze-manifest", type=Path)
    value.add_argument("--authorization-marker", type=Path)
    value.add_argument("--endpoint-checkpoint", type=Path, default=ENDPOINT / "endpoint.pt")
    value.add_argument("--endpoint-receipt", type=Path, default=ENDPOINT / "endpoint_receipt.json")
    value.add_argument("--endpoint-marker", type=Path, default=ENDPOINT / "ENDPOINT_FROZEN")
    value.add_argument("--reference-6x6", type=Path, default=DATA / "reference_6x6.xyz")
    value.add_argument("--reference-8x8", type=Path, default=DATA / "reference_8x8.xyz")
    value.add_argument("--thermal92", type=Path, default=DATA / "train_thermal.xyz")
    value.add_argument(
        "--harmonic-zero32",
        type=Path,
        default=DATA / "train_harmonic_lambda1_small_zero.xyz",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    inputs = FormalInputs(
        endpoint_checkpoint=args.endpoint_checkpoint,
        endpoint_receipt=args.endpoint_receipt,
        endpoint_marker=args.endpoint_marker,
        reference_6x6=args.reference_6x6,
        reference_8x8=args.reference_8x8,
        thermal92=args.thermal92,
        harmonic_zero32=args.harmonic_zero32,
    )
    if args.mode == "shard":
        if args.shard_id is None or args.preflight_kind is not None:
            raise ValueError("shard mode requires --shard-id and forbids --preflight-kind")
        if args.freeze_manifest is None or args.authorization_marker is None:
            raise ValueError("formal shard requires freeze manifest and authorization marker")
        run_shard(
            inputs,
            args.output,
            args.shard_id,
            device=args.device,
            freeze_manifest=args.freeze_manifest,
            authorization_marker=args.authorization_marker,
        )
    elif args.mode == "mechanics":
        if args.shard_id is not None or args.preflight_kind is not None:
            raise ValueError("mechanics mode forbids shard/preflight arguments")
        if args.freeze_manifest is None or args.authorization_marker is None:
            raise ValueError("formal mechanics requires freeze manifest and authorization marker")
        run_mechanics(
            inputs,
            args.output,
            device=args.device,
            freeze_manifest=args.freeze_manifest,
            authorization_marker=args.authorization_marker,
        )
    else:
        if args.preflight_kind is None or args.shard_id is not None:
            raise ValueError("preflight mode requires --preflight-kind and forbids --shard-id")
        if args.freeze_manifest is not None or args.authorization_marker is not None:
            raise ValueError("bounded preflight does not consume formal GO authorization")
        run_bounded_preflight(
            inputs,
            args.output,
            device=args.device,
            kind=args.preflight_kind,
        )


if __name__ == "__main__":
    main()
