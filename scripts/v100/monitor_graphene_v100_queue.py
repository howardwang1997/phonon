#!/usr/bin/env python3
"""Low-overhead V100 resource monitor with a validation-safe held queue."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


KB_EV_K = 8.617333262145e-5
RYDBERG_EV = 13.605693122994


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def command(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )


def gpu() -> dict:
    query = command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if query.returncode:
        return {"status": "unavailable", "error": query.stderr.strip()}
    name, used, total, utilization = [value.strip() for value in query.stdout.split(",")]
    applications = command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = []
    for line in applications.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            continue
        pid = int(fields[0])
        details = command(
            ["ps", "-p", str(pid), "-o", "etime=,lstart=,args="]
        ).stdout.strip()
        processes.append(
            {
                "pid": pid,
                "process_name": fields[1],
                "used_memory_MiB": int(fields[2]),
                "process_details": details,
            }
        )
    return {
        "status": "available",
        "name": name,
        "memory_used_MiB": int(used),
        "memory_total_MiB": int(total),
        "utilization_percent": int(utilization),
        "compute_processes": processes,
        "idle_for_new_primary_task": int(used) < 1000 and not processes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("A", "B"), required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    temperature = 375 if args.lane == "A" else 525
    degauss = KB_EV_K * temperature / RYDBERG_EV
    root = Path("/data/graphene_physics_temperature/automation")
    output = root / f"V100-{args.lane}_status.json"
    release = root / f"RELEASE_V0_{temperature}K"
    while True:
        try:
            disk = shutil.disk_usage("/data")
            gpu_record = gpu()
            released = release.exists()
            payload = {
                "status": "active",
                "updated_at_iso": datetime.now().astimezone().isoformat(),
                "machine": f"V100-{args.lane}",
                "resources": {
                    "gpu": gpu_record,
                    "data_free_GiB": disk.free / 1024**3,
                    "new_QE_start_allowed_by_storage": disk.free >= 50 * 1024**3,
                },
                "held_queue": {
                    "task": f"V0 static DFPT at T={temperature} K",
                    "lattice_temperature_K": temperature,
                    "degauss_Ry": degauss,
                    "degauss_formula": "k_B*T/Ry",
                    "estimated_box_hours": [67, 77],
                    "release_marker": str(release),
                    "state": (
                        "released_waiting_for_resource_and_submission"
                        if released
                        else "validation_locked_until_L0_Q0_X0_freeze"
                    ),
                    "automatic_start_before_release": False,
                },
            }
            atomic_json(output, payload)
        except Exception as error:
            atomic_json(
                output,
                {
                    "status": "monitor_error",
                    "updated_at_iso": datetime.now().astimezone().isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
