#!/usr/bin/env python3
"""Write a reviewed, plan-only Tailscale dual-node smoke launch receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphene_r2o_tail_smoke import build_dual_node_launch_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--nodes-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nodes = json.loads(args.nodes_json.read_text(encoding="utf-8"))
    if not isinstance(nodes, list):
        raise ValueError("nodes JSON must be a list")
    plan = build_dual_node_launch_plan(
        args.portable_root,
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        nodes=nodes,
        output_path=args.output,
    )
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

