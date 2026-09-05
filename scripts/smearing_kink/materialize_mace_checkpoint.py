#!/usr/bin/env python3
"""Materialize a MACE training checkpoint as a standalone model."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = copy.deepcopy(
        torch.load(args.template_model, map_location="cpu", weights_only=False)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"checkpoint has no model state: {args.checkpoint}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to("cpu")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    torch.save(model, temporary)
    os.replace(temporary, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
