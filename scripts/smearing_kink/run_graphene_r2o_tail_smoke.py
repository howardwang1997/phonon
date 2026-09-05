#!/usr/bin/env python3
"""Run one hash-bound R2O no-support actual-tail smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphene_r2o_tail_smoke import run_node_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--node-label", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--formal-go-marker")
    parser.add_argument("--expected-formal-go-marker-sha256")
    args = parser.parse_args()
    result = run_node_smoke(
        args.portable_root,
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        node_label=args.node_label,
        output_root=args.output_root,
        device=args.device,
        formal_go_marker_relative_path=args.formal_go_marker,
        expected_formal_go_marker_sha256=(
            args.expected_formal_go_marker_sha256
        ),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
