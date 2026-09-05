#!/usr/bin/env python3
"""Write a checksum inventory for graphene transferability source artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("artifacts must use LABEL=/path")
    label, path = specification.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("artifacts must use LABEL=/path")
    return label, Path(path)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", type=labelled_path, required=True)
    parser.add_argument("--source-host", default="local")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifacts = {}
    for label, path in args.artifact:
        if label in artifacts:
            raise ValueError(f"duplicate artifact label: {label}")
        if not path.is_file():
            raise FileNotFoundError(path)
        record = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            record["json_summary"] = {
                key: payload[key]
                for key in (
                    "status",
                    "scope",
                    "temperature_K",
                    "degauss_Ry",
                    "passes_force_and_replay_gate",
                    "passes_all_force_seed_qspace_gates",
                    "passes_all_force_seed_calibrated_qspace_gates",
                )
                if key in payload
            }
        artifacts[label] = record

    result = {
        "status": "complete",
        "scope": args.scope,
        "source_host": args.source_host,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
