#!/usr/bin/env python3
"""CLI for the independent, non-adjudicating graphene R2R-0D package."""
from __future__ import annotations

import argparse
from pathlib import Path

from graphene_r2r0_diagnostic import (
    prepare_freeze_manifest,
    run_cpu_preflight,
    run_diagnostic,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "mode", choices=("prepare-freeze-manifest", "cpu-preflight", "diagnostic")
    )
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--device", default=None)
    value.add_argument("--freeze-manifest", type=Path)
    value.add_argument("--authorization-marker", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.mode == "prepare-freeze-manifest":
        if any(
            value is not None
            for value in (
                args.device,
                args.freeze_manifest,
                args.authorization_marker,
            )
        ):
            raise ValueError("manifest preparation accepts only --output")
        prepare_freeze_manifest(args.output)
        return
    if args.freeze_manifest is None or args.authorization_marker is None:
        raise ValueError(
            "R2R-0D execution requires freeze manifest and authorization marker"
        )
    if args.mode == "cpu-preflight":
        if args.device not in (None, "cpu"):
            raise ValueError("CPU preflight accepts only --device cpu")
        receipt = run_cpu_preflight(
            args.output,
            freeze_manifest=args.freeze_manifest,
            authorization_marker=args.authorization_marker,
        )
    else:
        if args.device is None or not args.device.startswith("cuda"):
            raise ValueError("full diagnostic requires an explicit CUDA device")
        receipt = run_diagnostic(
            args.output,
            device=args.device,
            freeze_manifest=args.freeze_manifest,
            authorization_marker=args.authorization_marker,
        )
    if receipt.get("stopped_early") is True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
