#!/usr/bin/env python3
"""Aggregate exactly three R2R-0 shards plus the independent mechanics run."""
from __future__ import annotations

import argparse
from pathlib import Path

from graphene_r2r0_formal import aggregate_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--mechanics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--authorization-marker", type=Path, required=True)
    args = parser.parse_args()
    if len(args.shard) != 3:
        raise ValueError("aggregate CLI requires exactly three ordered --shard values")
    aggregate_shards(
        args.shard,
        args.mechanics,
        args.output,
        freeze_manifest=args.freeze_manifest,
        authorization_marker=args.authorization_marker,
    )


if __name__ == "__main__":
    main()
