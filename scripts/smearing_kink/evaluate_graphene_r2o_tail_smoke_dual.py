#!/usr/bin/env python3
"""Aggregate exactly two completed R2O actual-tail node smokes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphene_r2o_tail_smoke import evaluate_dual_node_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--first-result-sha256", required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--second-result-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_dual_node_results(
        [
            (args.first_root, args.first_result_sha256),
            (args.second_root, args.second_result_sha256),
        ],
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

