#!/usr/bin/env python3
"""Persistent status monitor and gate-aware S0 -> L0 -> Q0 -> X0 dispatcher."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


ROOT = Path.home() / "phonon"
S0 = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "S0_unified_short"
)
FORMAL = S0 / "formal_240ep_2060_seed83_replayw16"
L0 = S0 / "L0_classical_tdep"
Q0 = S0 / "Q0_quantum_sscha"
X0 = S0 / "X0_cross_development"
AUTOMATION = S0 / "automation"
STATUS = AUTOMATION / "status.json"
HISTORY = AUTOMATION / "rate_history.json"
FORMAL_UNIT = "graphene-physical-s0-replayw16.service"
L0_UNIT = "graphene-physical-s0-l0.service"
Q0_UNIT = "graphene-physical-q0.service"
X0_UNIT = "graphene-physical-x0.service"
TEMPERATURE_ORDER = (450, 300, 600)
SEEDS = (0, 1, 2)
DEFAULT_SNAPSHOT_RATE_PER_HOUR = {0: 350.0, 1: 190.0, 2: 350.0}
EPOCH_PATTERN = re.compile(
    r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ INFO: "
    r"Epoch (?P<epoch>\d+):.*RMSE_F=\s*(?P<rmse>[0-9.]+) meV / A"
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def command(args: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def service_state(name: str) -> str:
    result = command(["systemctl", "--user", "is-active", name])
    return result.stdout.strip() or "unknown"


def gpu_state() -> dict:
    result = command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode:
        return {"status": "unavailable", "error": result.stderr.strip()}
    name, used, total, utilization = [value.strip() for value in result.stdout.split(",")]
    return {
        "status": "available",
        "name": name,
        "memory_used_MiB": int(used),
        "memory_total_MiB": int(total),
        "utilization_percent": int(utilization),
    }


def formal_progress(now: datetime) -> dict:
    records = []
    try:
        for line in (FORMAL / "run.log").read_text(errors="replace").splitlines():
            match = EPOCH_PATTERN.search(line)
            if match:
                records.append(
                    {
                        "timestamp": datetime.strptime(
                            match.group("time"), "%Y-%m-%d %H:%M:%S"
                        ).astimezone(),
                        "epoch": int(match.group("epoch")),
                        "validation_force_RMSE_meV_A": float(match.group("rmse")),
                    }
                )
    except FileNotFoundError:
        pass
    result = {
        "service": service_state(FORMAL_UNIT),
        "requested_epochs": 240,
        "checkpoint_evaluation_included_after_training": True,
    }
    if records:
        latest = records[-1]
        result.update(
            {
                "latest_epoch": latest["epoch"],
                "latest_validation_force_RMSE_meV_A": latest[
                    "validation_force_RMSE_meV_A"
                ],
            }
        )
        if len(records) > 1:
            elapsed = (records[-1]["timestamp"] - records[0]["timestamp"]).total_seconds()
            epoch_delta = records[-1]["epoch"] - records[0]["epoch"]
            seconds_per_epoch = elapsed / epoch_delta if epoch_delta > 0 else None
            if seconds_per_epoch:
                remaining_epochs = max(0, 240 - (latest["epoch"] + 5))
                eta = now + timedelta(seconds=remaining_epochs * seconds_per_epoch + 600)
                result["measured_seconds_per_epoch"] = seconds_per_epoch
                result["estimated_completion_iso"] = eta.isoformat()
    sweep = read_json(FORMAL / "checkpoint_sweep.json")
    if (FORMAL / "DONE").exists() and sweep:
        result["state"] = (
            "passed" if sweep.get("status") == "checkpoint_passed" else "blocked_force_gate"
        )
        result["gate_status"] = sweep.get("status")
        result["selected_epoch"] = sweep.get("selected", {}).get("epoch")
    elif (FORMAL / "FAILED").exists() or (FORMAL / "FAILED_INFRASTRUCTURE").exists():
        result["state"] = "failed_infrastructure"
    else:
        result["state"] = "running_or_initializing"
    return result


def checkpoint_path(temperature: int, seed: int) -> Path:
    dt_tag = "dt0p25" if seed == 1 else "dt0p5"
    return L0 / f"T{temperature}/checkpoints/seed{seed}_{dt_tag}/T{temperature}/snapshots.npz"


def snapshot_count(path: Path) -> int:
    try:
        with np.load(path, allow_pickle=False) as data:
            return int(len(np.asarray(data["positions"])))
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return 0


def l0_progress(now: datetime, previous: dict) -> tuple[dict, dict]:
    counts = {
        f"T{temperature}_seed{seed}": snapshot_count(checkpoint_path(temperature, seed))
        for temperature in TEMPERATURE_ORDER
        for seed in SEEDS
    }
    prior_counts = previous.get("counts", {})
    prior_time = float(previous.get("observed_at_epoch", now.timestamp()))
    elapsed_hours = max((now.timestamp() - prior_time) / 3600.0, 0.0)
    observed_rates = dict(previous.get("rates_per_hour", {}))
    update_history_baseline = not prior_counts or elapsed_hours >= 0.05
    if prior_counts and elapsed_hours >= 0.05:
        for key, count in counts.items():
            delta = count - int(prior_counts.get(key, count))
            if delta > 0:
                measured = delta / elapsed_hours
                old = observed_rates.get(key)
                observed_rates[key] = measured if old is None else 0.5 * old + 0.5 * measured

    remaining_hours = 0.0
    active_lane = None
    for temperature in TEMPERATURE_ORDER:
        for seed in SEEDS:
            key = f"T{temperature}_seed{seed}"
            count = counts[key]
            if count < 3000:
                active_lane = active_lane or key
                rate = observed_rates.get(key, DEFAULT_SNAPSHOT_RATE_PER_HOUR[seed])
                remaining_hours += (3000 - count) / max(rate, 1.0)
    # Fixed allowance for three point fits and 1000-replicate bootstraps.
    remaining_analysis_hours = sum(
        0.0 if (L0 / f"T{temperature}/FINAL_GATE_PASSED").exists() else 2.0
        for temperature in TEMPERATURE_ORDER
    )
    remaining_hours += remaining_analysis_hours
    markers = {
        "pilot_done": (L0 / "PILOT_DONE").exists(),
        "all_temperatures_passed": (L0 / "L0_ALL_TEMPERATURES_PASSED").exists(),
        "blocked_pilot": (L0 / "BLOCKED_PILOT_STABILITY").exists(),
        "blocked_point_gate": (L0 / "BLOCKED_FINAL_POINT_GATE").exists(),
        "blocked_bootstrap_gate": (L0 / "BLOCKED_FINAL_BOOTSTRAP_GATE").exists(),
        "waiting_storage": (L0 / "WAITING_STORAGE").exists(),
        "waiting_gpu": (L0 / "WAITING_GPU").exists(),
        "failed_infrastructure": (L0 / "FAILED_INFRASTRUCTURE").exists(),
    }
    if markers["all_temperatures_passed"]:
        state = "passed"
    elif any(
        markers[key]
        for key in ("blocked_pilot", "blocked_point_gate", "blocked_bootstrap_gate")
    ):
        state = "blocked_scientific_gate"
    elif markers["failed_infrastructure"]:
        state = "failed_infrastructure"
    elif markers["waiting_storage"]:
        state = "waiting_storage"
    elif markers["waiting_gpu"]:
        state = "waiting_gpu"
    elif service_state(L0_UNIT) == "active":
        state = "running"
    else:
        state = "not_started_or_between_services"
    result = {
        "state": state,
        "service": service_state(L0_UNIT),
        "execution_order_K": list(TEMPERATURE_ORDER),
        "snapshot_counts": counts,
        "target_snapshots_per_seed": 3000,
        "active_or_next_lane": active_lane,
        "estimated_remaining_hours": remaining_hours,
        "estimated_completion_iso": (now + timedelta(hours=remaining_hours)).isoformat(),
        "observed_snapshot_rates_per_hour": observed_rates,
        "markers": markers,
    }
    history = (
        {
            "observed_at_epoch": now.timestamp(),
            "counts": counts,
            "rates_per_hour": observed_rates,
        }
        if update_history_baseline
        else previous
    )
    return result, history


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip()
    except FileNotFoundError:
        return None


def q0_progress() -> dict:
    cases = {}
    for name in ("benchmark_T450", "formal_T450", "formal_T300", "formal_T600"):
        acceptance = read_json(Q0 / name / "acceptance.json")
        if acceptance:
            cases[name] = {
                "status": acceptance.get("status"),
                "converged": acceptance.get("converged"),
                "final_population": acceptance.get("final_population"),
                "maximum_populations": acceptance.get("maximum_populations"),
                "supercell_min_frequency_cm-1": acceptance.get(
                    "supercell_min_frequency_cm-1"
                ),
            }
    markers = {
        "benchmark_passed": (Q0 / "BENCHMARK_PASSED").exists(),
        "all_temperatures_passed": (Q0 / "Q0_ALL_TEMPERATURES_PASSED").exists(),
        "blocked_benchmark": (Q0 / "BLOCKED_Q0_BENCHMARK").exists(),
        "blocked_formal": (Q0 / "BLOCKED_Q0_FORMAL").exists(),
        "failed_infrastructure": (Q0 / "FAILED_INFRASTRUCTURE").exists(),
    }
    service = service_state(Q0_UNIT)
    if markers["all_temperatures_passed"]:
        state = "passed"
    elif markers["blocked_benchmark"] or markers["blocked_formal"]:
        state = "blocked_convergence_gate"
    elif markers["failed_infrastructure"]:
        state = "failed_infrastructure"
    elif service == "active":
        state = "running"
    elif (L0 / "L0_ALL_TEMPERATURES_PASSED").exists():
        state = "ready_or_between_services"
    else:
        state = "held_until_L0_passes"
    return {
        "state": state,
        "service": service,
        "running_case": read_text(Q0 / "RUNNING_CASE"),
        "cases": cases,
        "markers": markers,
        "budget_hours": {"benchmark_450K": [2, 4], "formal_300_450_600K": [10, 20]},
    }


def x0_progress() -> dict:
    sscha = read_json(X0 / "analysis/sscha_acceptance.json")
    md = read_json(X0 / "analysis/md_acceptance.json")
    fallback_snapshots = snapshot_count(
        X0 / "fallback_md/checkpoints/seed0_dt0p5/T450/snapshots.npz"
    )
    markers = {
        "development_passed": (X0 / "X0_DEVELOPMENT_PASSED").exists(),
        "sscha_cross_exceeds_gate": (X0 / "X0_SSCHA_CROSS_EXCEEDS_GATE").exists(),
        "blocked_sscha_convergence": (X0 / "BLOCKED_X0_SSCHA_CONVERGENCE").exists(),
        "blocked_cross_term": (X0 / "BLOCKED_X0_CROSS_TERM").exists(),
        "failed_infrastructure": (X0 / "FAILED_INFRASTRUCTURE").exists(),
        "ready_for_validation_freeze_review": (
            S0 / "READY_FOR_VALIDATION_FREEZE_REVIEW"
        ).exists(),
    }
    service = service_state(X0_UNIT)
    if markers["development_passed"]:
        state = "passed_formula_reconstruction"
    elif markers["blocked_sscha_convergence"]:
        state = "blocked_convergence_gate"
    elif markers["blocked_cross_term"]:
        state = "blocked_cross_term_after_fixed_seed_diagnostic"
    elif markers["failed_infrastructure"]:
        state = "failed_infrastructure"
    elif service == "active":
        state = "running"
    elif (Q0 / "Q0_ALL_TEMPERATURES_PASSED").exists():
        state = "ready_or_between_services"
    else:
        state = "held_until_Q0_passes"
    return {
        "state": state,
        "service": service,
        "running_case": read_text(X0 / "RUNNING_CASE"),
        "first_cross_SSCHA": sscha,
        "fallback_MD": {
            "snapshot_count": fallback_snapshots,
            "target_snapshots": 3000,
            "acceptance": md,
        },
        "markers": markers,
        "fixed_first_condition": {
            "lattice_temperature_K": 450,
            "operator_temperature_K": 300,
            "top_branch_gate_cm-1": 2.0,
            "K_kink_relative_change_gate": 0.20,
        },
    }


def resources_allow_start(gpu: dict, free_gib: float) -> str | None:
    if free_gib < 50.0:
        return "RTX_free_space_below_50_GiB"
    if gpu.get("status") != "available" or gpu.get("memory_used_MiB", 99999) > 1000:
        return "RTX_GPU_not_idle"
    return None


def start_unit(unit: str, action: str) -> dict:
    result = command(["systemctl", "--user", "start", unit], timeout=30)
    return {
        "action": action if result.returncode == 0 else f"{action}_failed",
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
    }


def maybe_dispatch(formal: dict, gpu: dict, free_gib: float) -> dict:
    decision = {"action": "none"}
    if formal.get("state") != "passed":
        decision["reason"] = "waiting_for_passing_S0_force_gate"
        return decision
    if not (L0 / "L0_ALL_TEMPERATURES_PASSED").exists():
        if (L0 / "DONE").exists():
            decision["reason"] = "L0_stopped_at_scientific_gate"
            return decision
        if service_state(L0_UNIT) == "active":
            decision["reason"] = "L0_already_running"
            return decision
        blocked = resources_allow_start(gpu, free_gib)
        if blocked:
            decision["reason"] = blocked
            return decision
        return start_unit(L0_UNIT, "start_L0")

    if not (Q0 / "Q0_ALL_TEMPERATURES_PASSED").exists():
        if (Q0 / "DONE").exists():
            decision["reason"] = "Q0_stopped_at_convergence_gate"
            return decision
        if service_state(Q0_UNIT) == "active":
            decision["reason"] = "Q0_already_running"
            return decision
        if service_state(L0_UNIT) == "active":
            decision["reason"] = "waiting_for_L0_service_to_exit"
            return decision
        blocked = resources_allow_start(gpu, free_gib)
        if blocked:
            decision["reason"] = blocked
            return decision
        return start_unit(Q0_UNIT, "start_Q0")

    if not (X0 / "X0_DEVELOPMENT_PASSED").exists():
        if (X0 / "DONE").exists():
            decision["reason"] = "X0_stopped_at_fixed_cross_gate"
            return decision
        if service_state(X0_UNIT) == "active":
            decision["reason"] = "X0_already_running"
            return decision
        if service_state(Q0_UNIT) == "active":
            decision["reason"] = "waiting_for_Q0_service_to_exit"
            return decision
        blocked = resources_allow_start(gpu, free_gib)
        if blocked:
            decision["reason"] = blocked
            return decision
        return start_unit(X0_UNIT, "start_X0")

    decision["reason"] = "development_ready_for_explicit_validation_freeze_review"
    return decision


def main() -> int:
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    while True:
        now = datetime.now().astimezone()
        try:
            disk = shutil.disk_usage(ROOT)
            free_gib = disk.free / 1024**3
            gpu = gpu_state()
            formal = formal_progress(now)
            dispatch = maybe_dispatch(formal, gpu, free_gib)
            previous = read_json(HISTORY) or {}
            l0, history = l0_progress(now, previous)
            q0 = q0_progress()
            x0 = x0_progress()
            atomic_json(HISTORY, history)
            payload = {
                "status": "active",
                "updated_at_iso": now.isoformat(),
                "host": "RTX 2060",
                "resources": {"gpu": gpu, "free_disk_GiB": free_gib},
                "S0_formal": formal,
                "automatic_dispatch": dispatch,
                "L0_classical_TDEP": l0,
                "Q0_quantum_SSCHA": q0,
                "X0_cross_development": x0,
                "held_queue": [
                    {
                        "machine": "RTX 2060",
                        "task": "Q0 450 K quantum benchmark",
                        "state": q0["state"],
                        "budget_hours": [2, 4],
                    },
                    {
                        "machine": "RTX 2060",
                        "task": "X0 first non-diagonal SSCHA cross gate",
                        "state": x0["state"],
                        "budget_hours": [2, 5],
                    },
                    {
                        "machine": "V100-A",
                        "task": "V0 static DFPT at 375 K formula degauss",
                        "state": "validation_locked_until_L0_Q0_X0_freeze",
                        "budget_hours": [67, 77],
                    },
                    {
                        "machine": "V100-B",
                        "task": "V0 static DFPT at 525 K formula degauss",
                        "state": "validation_locked_until_L0_Q0_X0_freeze",
                        "budget_hours": [67, 77],
                    },
                ],
                "automation_boundary": {
                    "starts_L0_only_after_all_S0_force_gates_pass": True,
                    "stops_on_any_fixed_L0_scientific_gate_failure": True,
                    "starts_Q0_only_after_all_L0_gates_pass": True,
                    "starts_X0_only_after_all_Q0_convergence_gates_pass": True,
                    "X0_failure_runs_only_one_fixed_seed_MD_diagnostic": True,
                    "does_not_unlock_375_525_K": True,
                    "does_not_preempt_other_GPU_jobs": True,
                    "reboot_note": "user linger is disabled; after a full host reboot, start the autopilot user service once",
                },
            }
            atomic_json(STATUS, payload)
        except Exception as error:  # keep the monitor alive and make the failure visible
            atomic_json(
                STATUS,
                {
                    "status": "monitor_error",
                    "updated_at_iso": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
