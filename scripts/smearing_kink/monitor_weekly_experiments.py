#!/usr/bin/env python3
"""Monitor the active experiment lanes and update the latest weekly report.

The remote services own computation and checkpoint recovery.  This monitor
adds a second, conservative layer: it restarts a lane only when its completion
marker is absent and its service is inactive, then syncs validated artifacts
after completion and reruns the local analysis/plotting scripts.
"""
from __future__ import annotations

import csv
import fcntl
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "results" / "monitor"
STATE_PATH = STATE_DIR / "weekly_experiments_state.json"
LOCK_PATH = Path("/tmp/phonon_weekly_experiments.lock")
SSH = Path("/usr/bin/ssh")
SCP = Path("/usr/bin/scp")
CONDA = Path("/Users/howardwang/miniconda3/bin/conda")

V100_A = "root@100.80.236.112"
V100_B = "root@100.123.220.57"
RTX_2060 = "howardwang@100.105.21.7"
V100_TARGETS = {V100_A, V100_B}

START_MARKER = "<!-- AUTO-EXPERIMENT-STATUS:START -->"
END_MARKER = "<!-- AUTO-EXPERIMENT-STATUS:END -->"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="minutes")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}", flush=True)


def run(command: list[str], *, timeout: int = 1800, check: bool = True):
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-4000:]}"
        )
    return completed


def ssh(target: str, command: str, *, timeout: int = 35, check: bool = True):
    arguments = [
        str(SSH),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
    ]
    if target in V100_TARGETS:
        # Both hops use 100.x addresses.  This avoids the currently unreliable
        # Mac -> Tokyo DERP TCP path while keeping all traffic in Tailscale.
        arguments.extend(["-J", RTX_2060])
    arguments.extend([target, command])
    return run(arguments, timeout=timeout, check=check)


def parse_key_values(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def probe(target: str, command: str) -> dict[str, str]:
    try:
        result = ssh(target, command, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"reachable": "0", "error": str(exc)}
    if result.returncode != 0:
        return {
            "reachable": "0",
            "error": result.stdout.strip()[-500:],
        }
    values = parse_key_values(result.stdout)
    values["reachable"] = "1"
    return values


def probe_v100_a() -> dict[str, str]:
    return probe(
        V100_A,
        r'''
echo "fullpath_done=$(test -e /data/graphene_fullpath_pilot/DONE && echo 1 || echo 0)"
echo "fullpath_service=$(systemctl is-active phonon-graphene-fullpath-pilot-v2.service 2>/dev/null || true)"
ph_count=$(grep -l "JOB DONE" /data/graphene_fullpath_pilot/dg*_k*/*/ph.out 2>/dev/null | wc -l | tr -d ' ')
echo "fullpath_points=$ph_count"
echo "cold_done=$(test -e /root/phonon/results/tmd_exp_a_recovery/COLD_PAIRS_DONE && echo 1 || echo 0)"
echo "cold_service=$(systemctl is-active phonon-tmd-cold-pairs.service 2>/dev/null || true)"
echo "cold_verifier=$(systemctl is-active phonon-tmd-cold-pairs-verifier.service 2>/dev/null || true)"
cold_count=$(find /root/phonon/results/tmd_exp_a_recovery -maxdepth 1 -type f -name 'disp_*_cold.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "cold_pairs=$cold_count"
echo "fd_done=$(test -e /data/graphene_physical_fd_dfpt/campaigns/FD300_CONV/DONE && echo 1 || echo 0)"
echo "fd_service=$(systemctl is-active phonon-graphene-physical-fd@FD300_CONV.service 2>/dev/null || true)"
fd_scf=$(grep -l 'JOB DONE' /data/graphene_physical_fd_dfpt/sets/dg0.0019000869_k*/scf.out 2>/dev/null | wc -l | tr -d ' ')
fd_ph=$(grep -l 'JOB DONE' /data/graphene_physical_fd_dfpt/sets/dg0.0019000869_k*/*/ph.out 2>/dev/null | wc -l | tr -d ' ')
echo "fd_scf=$fd_scf"
echo "fd_ph=$fd_ph"
echo "force_done=$(test -e /data/graphene_fd_force_convergence/A/DONE && echo 1 || echo 0)"
echo "force_service=$(systemctl is-active phonon-graphene-fd-force@A.service 2>/dev/null || true)"
force_count=$(find /data/graphene_fd_force_convergence/A/labels -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "force_count=$force_count"
echo "k144_done=$(test -e /data/graphene_physical_fd_dfpt/campaigns/FD300_K144/DONE && echo 1 || echo 0)"
echo "k144_service=$(systemctl is-active phonon-graphene-physical-fd@FD300_K144.service 2>/dev/null || true)"
k144_ph=0
for slug in G_t0p000 G_t0p015 G_t0p030 K_t0p980 K_t0p990 K_t1p000 K_t1p010 K_t1p020; do
  if grep -q 'JOB DONE' "/data/graphene_physical_fd_dfpt/sets/dg0.0019000869_k144/$slug/ph.out" 2>/dev/null; then
    k144_ph=$((k144_ph + 1))
  fi
done
echo "k144_ph=$k144_ph"
echo "line_done=$(test -e /data/graphene_physical_fd_dfpt/campaigns/FD300_LINE/DONE && echo 1 || echo 0)"
echo "line_service=$(systemctl is-active phonon-graphene-physical-fd@FD300_LINE.service 2>/dev/null || true)"
line_ph=0
for slug in G_t0p000 G_t0p005 G_t0p010 G_t0p015 G_t0p025 G_t0p040 G_t0p060 G_t0p080 K_t0p940 K_t0p960 K_t0p975 K_t0p985 K_t0p992 K_t1p000 K_t1p008 K_t1p015 K_t1p025 K_t1p040 K_t1p060; do
  if grep -q 'JOB DONE' "/data/graphene_physical_fd_dfpt/sets/dg0.0019000869_k144/$slug/ph.out" 2>/dev/null; then
    line_ph=$((line_ph + 1))
  fi
done
echo "line_ph=$line_ph"
echo "thermal_done=$(test -e /data/graphene_fd_thermal_labels/A/DONE && echo 1 || echo 0)"
echo "thermal_service=$(systemctl is-active phonon-graphene-fd-labels@A.service 2>/dev/null || true)"
thermal_count=$(find /data/graphene_fd_thermal_labels/A/labels -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "thermal_count=$thermal_count"
echo "thermal2_done=$(test -e /data/graphene_fd_thermal_labels_wave2/A/DONE && echo 1 || echo 0)"
echo "thermal2_service=$(systemctl is-active phonon-graphene-fd-labels-wave2@A.service 2>/dev/null || true)"
thermal2_count=$(find /data/graphene_fd_thermal_labels_wave2/A/labels -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "thermal2_count=$thermal2_count"
echo "thermal3_done=$(test -e /data/graphene_fd_thermal_labels_wave3/A/DONE && echo 1 || echo 0)"
echo "thermal3_skipped=$(test -e /data/graphene_fd_thermal_labels_wave3/A/SKIPPED_GATE_PASSED && echo 1 || echo 0)"
echo "thermal3_service=$(systemctl is-active phonon-graphene-fd-labels-wave3@A.service 2>/dev/null || true)"
thermal3_count=$(find /data/graphene_fd_thermal_labels_wave3/A/labels -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "thermal3_count=$thermal3_count"
''',
    )


def probe_v100_b() -> dict[str, str]:
    return probe(
        V100_B,
        r'''
echo "done=$(test -e /data/gr_dftmd_recovery/B/DONE && echo 1 || echo 0)"
echo "service=$(systemctl is-active phonon-recovery-run@B.service 2>/dev/null || true)"
echo "watchdog=$(systemctl is-active phonon-recovery-watch@B.timer 2>/dev/null || true)"
state=/data/gr_dftmd_recovery/B/T600/md_checkpoint/state.npz
progress=""
if test -s "$state"; then
  progress=$(/root/miniconda3/envs/phonon/bin/python -c '
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    print(
        "MD checkpoint: snap={}/{}, stride={}/{}".format(
            int(data["sample_done"]), int(data["n_snap"]),
            int(data["stride_done"]), int(data["stride"]),
        )
    )
' "$state" 2>/dev/null || true)
fi
if test -z "$progress"; then
  progress=$(grep -E 'MD checkpoint: snap=|MD snapshot committed:' /data/gr_dftmd_recovery/B/recovery.log 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
fi
echo "progress=$progress"
echo "fd_done=$(test -e /data/graphene_physical_fd_dfpt/campaigns/FD600_CONV/DONE && echo 1 || echo 0)"
echo "fd_service=$(systemctl is-active phonon-graphene-physical-fd@FD600_CONV.service 2>/dev/null || true)"
fd_scf=$(grep -l 'JOB DONE' /data/graphene_physical_fd_dfpt/sets/dg0.0038001738_k*/scf.out 2>/dev/null | wc -l | tr -d ' ')
fd_ph=$(grep -l 'JOB DONE' /data/graphene_physical_fd_dfpt/sets/dg0.0038001738_k*/*/ph.out 2>/dev/null | wc -l | tr -d ' ')
echo "fd_scf=$fd_scf"
echo "fd_ph=$fd_ph"
echo "force_done=$(test -e /data/graphene_fd_force_convergence/B/DONE && echo 1 || echo 0)"
echo "force_service=$(systemctl is-active phonon-graphene-fd-force@B.service 2>/dev/null || true)"
force_count=$(find /data/graphene_fd_force_convergence/B/labels -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "force_count=$force_count"
echo "line_done=$(test -e /data/graphene_physical_fd_dfpt/campaigns/FD600_LINE/DONE && echo 1 || echo 0)"
echo "line_service=$(systemctl is-active phonon-graphene-physical-fd@FD600_LINE.service 2>/dev/null || true)"
line_ph=0
for slug in G_t0p000 G_t0p005 G_t0p010 G_t0p015 G_t0p025 G_t0p040 G_t0p060 G_t0p080 K_t0p940 K_t0p960 K_t0p975 K_t0p985 K_t0p992 K_t1p000 K_t1p008 K_t1p015 K_t1p025 K_t1p040 K_t1p060; do
  if grep -q 'JOB DONE' "/data/graphene_physical_fd_dfpt/sets/dg0.0038001738_k120/$slug/ph.out" 2>/dev/null; then
    line_ph=$((line_ph + 1))
  fi
done
echo "line_ph=$line_ph"
echo "fd0_done=$(test -e /data/graphene_physical_fd_dfpt/campaigns/FD0_SCAN/DONE && echo 1 || echo 0)"
echo "fd0_service=$(systemctl is-active phonon-graphene-physical-fd@FD0_SCAN.service 2>/dev/null || true)"
fd0_ph=0
for set_slug in dg0.0006333623_k192 dg0.0012667246_k144 dg0.0019000869_k120; do
  for q_slug in G_t0p000 G_t0p015 K_t0p985 K_t1p000 K_t1p015; do
    if grep -q 'JOB DONE' "/data/graphene_physical_fd_dfpt/sets/$set_slug/$q_slug/ph.out" 2>/dev/null; then
      fd0_ph=$((fd0_ph + 1))
    fi
  done
done
echo "fd0_ph=$fd0_ph"
echo "thermal_done=$(test -e /data/graphene_fd_thermal_labels/B/DONE && echo 1 || echo 0)"
echo "thermal_service=$(systemctl is-active phonon-graphene-fd-labels@B.service 2>/dev/null || true)"
thermal_count=$(find /data/graphene_fd_thermal_labels/B/labels -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "thermal_count=$thermal_count"
echo "thermal2_done=$(test -e /data/graphene_fd_thermal_labels_wave2/B/DONE && echo 1 || echo 0)"
echo "thermal2_service=$(systemctl is-active phonon-graphene-fd-labels-wave2@B.service 2>/dev/null || true)"
thermal2_count=$(find /data/graphene_fd_thermal_labels_wave2/B/labels -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "thermal2_count=$thermal2_count"
echo "thermal3_done=$(test -e /data/graphene_fd_thermal_labels_wave3/B/DONE && echo 1 || echo 0)"
echo "thermal3_skipped=$(test -e /data/graphene_fd_thermal_labels_wave3/B/SKIPPED_GATE_PASSED && echo 1 || echo 0)"
echo "thermal3_service=$(systemctl is-active phonon-graphene-fd-labels-wave3@B.service 2>/dev/null || true)"
thermal3_count=$(find /data/graphene_fd_thermal_labels_wave3/B/labels -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "thermal3_count=$thermal3_count"
ablation_out=/root/phonon/results/graphene_fd_model_ablation/T600
echo "ablation_done=$(test -e $ablation_out/DONE && echo 1 || echo 0)"
echo "ablation_service=$(systemctl is-active phonon-graphene-fd-model-ablation-600.service 2>/dev/null || true)"
ablation_progress=$(grep -E 'variant=.*(start|complete)|Epoch|COMPLETE' "$ablation_out/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "ablation_progress=${ablation_progress:-waiting to start}"
tdep_out=/root/phonon/results/graphene_fd_model_ablation/T600_TDEP
echo "ablation_tdep_done=$(test -e $tdep_out/DONE && echo 1 || echo 0)"
echo "ablation_tdep_failed=$(test -e $tdep_out/FAILED_STABILITY && echo 1 || echo 0)"
echo "ablation_tdep_service=$(systemctl is-active phonon-graphene-fd-model-ablation-tdep600.service 2>/dev/null || true)"
ablation_tdep_progress=$(grep -E 'waiting for|T=600 K:|wG=|COMPLETE|failed' "$tdep_out/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "ablation_tdep_progress=${ablation_tdep_progress:-waiting for model selection}"
v2_out=/root/phonon/results/graphene_fd_model_ablation_v2/T600
echo "v2_done=$(test -e $v2_out/DONE && echo 1 || echo 0)"
echo "v2_passed=$(test -e $v2_out/CANDIDATE_PASSED && echo 1 || echo 0)"
echo "v2_blocked=$(test -e $v2_out/BLOCKED_FORCE_GATE && echo 1 || echo 0)"
echo "v2_service=$(systemctl is-active phonon-graphene-fd-model-ablation-v2-600.service 2>/dev/null || true)"
v2_progress=$(grep -E 'variant=.*(start|failed)|Epoch|selected variant|COMPLETE' "$v2_out/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "v2_progress=${v2_progress:-waiting to start}"
v2_variant=$(grep -oE 'v2 variant=[^ ]+' "$v2_out/run.log" 2>/dev/null | tail -1 | cut -d= -f2)
echo "v2_variant=${v2_variant:-pending}"
v2_tdep_out=/root/phonon/results/graphene_fd_model_ablation_v2/T600_TDEP
echo "v2_tdep_done=$(test -e $v2_tdep_out/DONE && echo 1 || echo 0)"
echo "v2_tdep_failed=$(test -e $v2_tdep_out/FAILED_STABILITY && echo 1 || echo 0)"
echo "v2_tdep_service=$(systemctl is-active phonon-graphene-fd-model-ablation-v2-tdep600.service 2>/dev/null || true)"
v2_tdep_progress=$(grep -E 'waiting for|T=600 K:|wG=|COMPLETE|failed' "$v2_tdep_out/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "v2_tdep_progress=${v2_tdep_progress:-waiting for combined selection}"
''',
    )


def probe_2060() -> dict[str, str]:
    return probe(
        RTX_2060,
        r'''
echo "file_ready=$(test -s /home/howardwang/phonon/results/graphene_kohn_fd/deploy_15pt.npz && echo 1 || echo 0)"
echo "service=$(systemctl --user is-active phonon-deploy-15pt-v3.service 2>/dev/null || true)"
control_base=/home/howardwang/phonon/results/td_phonon
echo "control_done=$(test -e $control_base/graphene_v11_no_long_range_physical_geom.DONE && echo 1 || echo 0)"
echo "control_service=$(systemctl --user is-active phonon-graphene-v11-control.service 2>/dev/null || true)"
control_progress="not started"
for temperature in 300 600; do
  state="$control_base/graphene_v11_no_long_range_physical_geom_checkpoint/T${temperature}/state.npz"
  if test -s "$state"; then
    control_progress=$(/home/howardwang/miniconda3/bin/conda run -n phonon python -c '
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    print("T{} equil={}/{} snap={}/{} stride={}/{}".format(
        int(data["temperature"]), int(data["equil_done"]), int(data["n_equil"]),
        int(data["sample_done"]), int(data["n_snap"]),
        int(data["stride_done"]), int(data["stride"])))
' "$state" 2>/dev/null | tail -1)
  fi
done
echo "control_progress=$control_progress"
echo "control_seed_service=$(systemctl --user is-active phonon-graphene-v11-control-seeds.service 2>/dev/null || true)"
control_seed_count=$(find "$control_base" -maxdepth 1 -type f -name 'td_graphene_v11_no_long_range_seed*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "control_seed_count=$control_seed_count"
training_source=/home/howardwang/phonon/data/graphene_fd_thermal_labels
training_out=/home/howardwang/phonon/results/graphene_fd_thermal_finetune
echo "fd_train_ready=$(test -e $training_source/.READY && echo 1 || echo 0)"
echo "fd_train_done=$(test -e $training_out/DONE && echo 1 || echo 0)"
echo "fd_train_service=$(systemctl --user is-active phonon-graphene-fd-thermal-finetune.service 2>/dev/null || true)"
fd_train_progress=$(grep -E 'waiting for both|fine-tune|Epoch|short-range TDEP|MD checkpoint|COMPLETE' "$training_out/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "fd_train_progress=${fd_train_progress:-waiting for validated labels}"
wave2_source=/home/howardwang/phonon/data/graphene_fd_thermal_labels
wave2_out=/home/howardwang/phonon/results/graphene_fd_thermal_finetune_wave2
echo "wave2_ready=$(test -e $wave2_source/.READY_WAVE2 && echo 1 || echo 0)"
echo "wave2_done=$(test -e $wave2_out/DONE && echo 1 || echo 0)"
echo "wave2_service=$(systemctl --user is-active phonon-graphene-fd-thermal-refine-wave2.service 2>/dev/null || true)"
wave2_progress=$(grep -E 'waiting for|fine-tune|Epoch|TDEP|COMPLETE' "$wave2_out/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "wave2_progress=${wave2_progress:-waiting for wave2 labels}"
wave3_out=/home/howardwang/phonon/results/graphene_fd_thermal_finetune_wave3
echo "wave3_ready=$(test -e $wave2_source/.READY_WAVE3 && echo 1 || echo 0)"
echo "wave3_done=$(test -e $wave3_out/DONE && echo 1 || echo 0)"
echo "wave3_skipped=$(test -e $wave3_out/SKIPPED_GATE_PASSED && echo 1 || echo 0)"
echo "wave3_failed=$(test -e $wave3_out/FAILED_GATE && echo 1 || echo 0)"
echo "wave3_service=$(systemctl --user is-active phonon-graphene-fd-thermal-refine-wave3.service 2>/dev/null || true)"
wave3_progress=$(grep -E 'waiting for|fine-tune|Epoch|TDEP|COMPLETE|skipped' "$wave3_out/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "wave3_progress=${wave3_progress:-conditional queue waiting}"
ablation_out=/home/howardwang/phonon/results/graphene_fd_model_ablation
echo "ablation_300_done=$(test -e $ablation_out/T300/DONE && echo 1 || echo 0)"
echo "ablation_600_done=$(test -e $ablation_out/T600/REMOTE_DONE && echo 1 || echo 0)"
echo "ablation_300_service=$(systemctl --user is-active phonon-graphene-fd-model-ablation-300.service 2>/dev/null || true)"
ablation_300_progress=$(grep -E 'variant=.*(start|complete)|Epoch|COMPLETE' "$ablation_out/T300/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "ablation_300_progress=${ablation_300_progress:-waiting to start}"
echo "ablation_done=$(test -e $ablation_out/DONE && echo 1 || echo 0)"
echo "ablation_selection=$(test -s $ablation_out/selection.json && echo 1 || echo 0)"
echo "ablation_tdep600_done=$(test -e $ablation_out/TDEP600_REMOTE_DONE && echo 1 || echo 0)"
echo "ablation_tdep600_failed=$(test -e $ablation_out/TDEP600_FAILED && echo 1 || echo 0)"
echo "ablation_force_blocked=$(test -e $ablation_out/BLOCKED_FORCE_GATE && echo 1 || echo 0)"
echo "ablation_stability_blocked=$(test -e $ablation_out/BLOCKED_STABILITY && echo 1 || echo 0)"
echo "ablation_tdep_blocked=$(test -e $ablation_out/BLOCKED_TDEP_GATE && echo 1 || echo 0)"
echo "ablation_passed=$(test -e $ablation_out/PASSED_SHORT_RANGE && echo 1 || echo 0)"
echo "ablation_service=$(systemctl --user is-active phonon-graphene-fd-model-ablation-aggregate.service 2>/dev/null || true)"
ablation_progress=$(grep -E 'waiting for|no candidate|TDEP|COMPLETE' "$ablation_out/aggregate.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "ablation_progress=${ablation_progress:-waiting for both temperature lanes}"
v2_out=/home/howardwang/phonon/results/graphene_fd_model_ablation_v2
echo "v2_started=$(test -d $v2_out/T300 && echo 1 || echo 0)"
echo "v2_diagnostic_only=$(test -e $v2_out/DIAGNOSTIC_ONLY && echo 1 || echo 0)"
echo "v2_300_done=$(test -e $v2_out/T300/DONE && echo 1 || echo 0)"
echo "v2_300_passed=$(test -e $v2_out/T300/CANDIDATE_PASSED && echo 1 || echo 0)"
echo "v2_300_blocked=$(test -e $v2_out/T300/BLOCKED_FORCE_GATE && echo 1 || echo 0)"
echo "v2_300_service=$(systemctl --user is-active phonon-graphene-fd-model-ablation-v2-300.service 2>/dev/null || true)"
v2_300_progress=$(grep -E 'variant=.*(start|failed)|Epoch|selected variant|COMPLETE' "$v2_out/T300/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "v2_300_progress=${v2_300_progress:-waiting to start}"
v2_300_variant=$(grep -oE 'v2 variant=[^ ]+' "$v2_out/T300/run.log" 2>/dev/null | tail -1 | cut -d= -f2)
echo "v2_300_variant=${v2_300_variant:-pending}"
echo "v2_600_done=$(test -e $v2_out/T600/REMOTE_DONE && echo 1 || echo 0)"
echo "v2_600_blocked=$(test -e $v2_out/T600/BLOCKED_FORCE_GATE && echo 1 || echo 0)"
echo "v2_done=$(test -e $v2_out/DONE && echo 1 || echo 0)"
echo "v2_passed=$(test -e $v2_out/PASSED_SHORT_RANGE && echo 1 || echo 0)"
echo "v2_force_blocked=$(test -e $v2_out/BLOCKED_FORCE_GATE && echo 1 || echo 0)"
echo "v2_stability_blocked=$(test -e $v2_out/BLOCKED_STABILITY && echo 1 || echo 0)"
echo "v2_tdep_blocked=$(test -e $v2_out/BLOCKED_TDEP_GATE && echo 1 || echo 0)"
echo "v2_tdep600_done=$(test -e $v2_out/TDEP600_REMOTE_DONE && echo 1 || echo 0)"
echo "v2_tdep600_failed=$(test -e $v2_out/TDEP600_FAILED && echo 1 || echo 0)"
echo "v2_aggregate_service=$(systemctl --user is-active phonon-graphene-fd-model-ablation-v2-aggregate.service 2>/dev/null || true)"
v2_aggregate_progress=$(grep -E 'waiting for|TDEP|failed|COMPLETE' "$v2_out/aggregate.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "v2_aggregate_progress=${v2_aggregate_progress:-waiting for temperature lanes}"
long_out=/home/howardwang/phonon/results/graphene_physical_fd_long_range
zero_out=/home/howardwang/phonon/results/graphene_physical_fd0
weighted_out=/home/howardwang/phonon/results/graphene_fd_delta_weighted
echo "long_done=$(test -e $long_out/DONE && echo 1 || echo 0)"
echo "long_blocked=$(test -e $long_out/BLOCKED_SHORT_RANGE_GATE && echo 1 || echo 0)"
echo "post_done=$(test -e $long_out/POSTPROCESS_DONE && echo 1 || echo 0)"
echo "zero_done=$(test -e $zero_out/DONE && echo 1 || echo 0)"
echo "post_service=$(systemctl --user is-active phonon-graphene-physical-fd-postprocess.service 2>/dev/null || true)"
post_progress=$(grep -E 'waiting for|failed|complete|COMPLETE' "$long_out/run.log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "post_progress=${post_progress:-waiting for prerequisites}"
echo "weighted_force_done=$(test -e $weighted_out/holdout600/DONE && echo 1 || echo 0)"
echo "weighted_force_passed=$(test -e $weighted_out/holdout600/INDEPENDENT_FORCE_GATE_PASSED && echo 1 || echo 0)"
echo "weighted_tdep_done=$(test -s $weighted_out/T600_TDEP/acceptance.json && echo 1 || echo 0)"
echo "weighted_static_passed=$(grep -q '\"passes_all_force_seed_qspace_gates\": true' $weighted_out/T600_TDEP/acceptance.json 2>/dev/null && echo 1 || echo 0)"
echo "weighted_calibrated_done=$(test -e $weighted_out/T600_TDEP/CALIBRATED_REMOTE_DONE && echo 1 || echo 0)"
echo "weighted_calibrated_passed=$(grep -q '\"passes_all_force_seed_calibrated_qspace_gates\": true' $weighted_out/T600_TDEP/calibrated_acceptance.json 2>/dev/null && echo 1 || echo 0)"
weighted_progress=$(grep -E 'MD checkpoint:|wrote td_|short_seed|COMPLETE|failed' $weighted_out/T600_TDEP/run.log 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
echo "weighted_progress=${weighted_progress:-waiting for frozen model and force holdout}"
''',
    )


def copy_remote(target: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_name(local_path.name + ".partial")
    arguments = [
        str(SCP),
        "-p",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
    ]
    if target in V100_TARGETS:
        arguments.extend(["-o", f"ProxyJump={RTX_2060}"])
    arguments.extend([f"{target}:{remote_path}", str(temporary)])
    run(arguments, timeout=600)
    temporary.replace(local_path)
    log(f"synced {target}:{remote_path} -> {local_path.relative_to(ROOT)}")


def copy_to_remote(local_path: Path, target: str, remote_path: str) -> None:
    remote = PurePosixPath(remote_path)
    ssh(target, f"mkdir -p {shlex.quote(str(remote.parent))}")
    remote_partial = str(remote) + ".partial"
    arguments = [
        str(SCP),
        "-p",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
    ]
    if target in V100_TARGETS:
        arguments.extend(["-o", f"ProxyJump={RTX_2060}"])
    arguments.extend([str(local_path), f"{target}:{remote_partial}"])
    run(arguments, timeout=600)
    ssh(
        target,
        f"mv {shlex.quote(remote_partial)} {shlex.quote(str(remote))}",
    )
    log(f"staged {local_path.relative_to(ROOT)} -> {target}:{remote_path}")


def conda_python(script: Path, *arguments: str, timeout: int = 3600) -> None:
    completed = run(
        [
            str(CONDA),
            "run",
            "--no-capture-output",
            "-n",
            "phonon",
            "python",
            str(script),
            *map(str, arguments),
        ],
        timeout=timeout,
    )
    if completed.stdout.strip():
        log(completed.stdout.strip()[-1200:])


def sync_v100_b() -> None:
    td = ROOT / "results" / "td_phonon"
    provenance = td / "recovery_provenance" / "B"
    changed = False
    targets = (
        (
            "/root/phonon/results/td_phonon/graphene_dft_tdep_600K_recovery.npz",
            td / "graphene_dft_tdep_600K_recovery.npz",
        ),
        (
            "/root/phonon/results/td_phonon/graphene_dft_tdep_600K_recovery.csv",
            td / "graphene_dft_tdep_600K_recovery.csv",
        ),
        (
            "/data/gr_dftmd_recovery/B/T600/md_checkpoint/snapshots.npz",
            provenance / "snapshots.npz",
        ),
        (
            "/data/gr_dftmd_recovery/B/T600/md_checkpoint/state.npz",
            provenance / "state.npz",
        ),
        (
            "/data/gr_dftmd_recovery/B/recovery.log",
            provenance / "recovery.log",
        ),
    )
    for remote, local in targets:
        if not local.is_file():
            copy_remote(V100_B, remote, local)
            changed = True

    convergence = td / "graphene_dft_tdep_600K_recovery_convergence.json"
    if not convergence.is_file():
        conda_python(
            ROOT / "scripts" / "smearing_kink" / "analyze_dftmd_recovery.py",
            "--snapshots",
            provenance / "snapshots.npz",
            "--phonopy",
            ROOT / "results" / "vq_kink6" / "graphene_sc6_dg0.005_phonopy.yaml",
            "--output",
            convergence,
        )
        changed = True
    summary_path = td / "graphene_lattice_temperature_summary.json"
    summary_has_600 = False
    if summary_path.is_file():
        summary_has_600 = 600 in json.loads(summary_path.read_text()).get(
            "dft_lattice_temperatures_K", []
        )
    if changed or not summary_has_600:
        conda_python(
            ROOT / "scripts" / "smearing_kink" / "plot_graphene_dftmd_recovery.py"
        )


def sync_fullpath() -> None:
    destination = (
        ROOT
        / "results"
        / "p1_graphene_fullpath_pilot"
        / "graphene_fullpath_pilot_dfpt.csv"
    )
    if not destination.is_file():
        copy_remote(
            V100_A,
            "/root/phonon/results/p1_graphene_fullpath_pilot/graphene_fullpath_pilot_dfpt.csv",
            destination,
        )
    log_path = destination.parent / "run.log"
    if not log_path.is_file():
        copy_remote(V100_A, "/data/graphene_fullpath_pilot/run.log", log_path)


def sync_cold_pairs() -> None:
    output = ROOT / "results" / "tmd_exp_a_recovery"
    changed = False
    names = (
        "1T-VSe2_exp_a3.340_dg0.005_cold",
        "NbSe2_exp_a3.440_dg0.005_cold",
        "NbS2_exp_a3.320_dg0.005_cold",
        "2H-TaSe2_exp_a3.436_dg0.005_cold",
        "1T-TiSe2_exp_a3.540_dg0.005_cold",
    )
    for name in names:
        for prefix, suffix in (("disp_", ".npz"), ("", "_phonopy.yaml")):
            filename = f"{prefix}{name}{suffix}"
            destination = output / filename
            if not destination.is_file():
                copy_remote(
                    V100_A,
                    f"/root/phonon/results/tmd_exp_a_recovery/{filename}",
                    destination,
                )
                changed = True
    summary = output / "tmd_cold_fd_pairs.json"
    if changed or not summary.is_file():
        conda_python(
            ROOT / "scripts" / "smearing_kink" / "analyze_tmd_cold_fd_pairs.py"
        )


def sync_15point_deployment() -> None:
    destination = ROOT / "results" / "graphene_kohn_fd" / "deploy_15pt.npz"
    changed = False
    if not destination.is_file():
        copy_remote(
            RTX_2060,
            "/home/howardwang/phonon/results/graphene_kohn_fd/deploy_15pt.npz",
            destination,
        )
        changed = True
    expected = (
        ROOT / "results" / "smearing_kink" / "gr_fig1_kink_melt.png",
        ROOT / "results" / "smearing_kink" / "gr_fig2_hf_overlay.png",
        ROOT / "results" / "smearing_kink" / "gr_fig3_kito.png",
        ROOT / "results" / "smearing_kink" / "gr_fig4_branches.png",
    )
    if changed or not all(path.is_file() for path in expected):
        conda_python(
            ROOT / "scripts" / "smearing_kink" / "plot_graphene_15pt_v2.py"
        )


def sync_2060_v11_control() -> None:
    destination = ROOT / "results" / "td_phonon"
    tag = "graphene_v11_no_long_range_physical_geom"
    for remote_name, local_name in (
        (f"td_{tag}.npz", f"td_{tag}.npz"),
        (f"td_{tag}.csv", f"td_{tag}.csv"),
        (f"{tag}.log", f"{tag}.log"),
    ):
        local_path = destination / local_name
        if not local_path.is_file():
            copy_remote(
                RTX_2060,
                f"/home/howardwang/phonon/results/td_phonon/{remote_name}",
                local_path,
            )


def sync_physical_fd(target: str, campaign: str) -> None:
    destination = ROOT / "results" / "graphene_physical_fd_dfpt" / "campaigns" / campaign
    filename = f"graphene_{campaign}_dfpt.csv"
    for remote_name, local_name in ((filename, filename), ("run.log", "run.log")):
        local_path = destination / local_name
        if not local_path.is_file():
            copy_remote(
                target,
                f"/data/graphene_physical_fd_dfpt/campaigns/{campaign}/{remote_name}",
                local_path,
            )
    qpoints_conv = (
        "G_t0p000", "G_t0p015", "G_t0p030",
        "K_t0p980", "K_t0p990", "K_t1p000", "K_t1p010", "K_t1p020",
    )
    qpoints_line = (
        "G_t0p000", "G_t0p005", "G_t0p010", "G_t0p015", "G_t0p025",
        "G_t0p040", "G_t0p060", "G_t0p080", "K_t0p940", "K_t0p960",
        "K_t0p975", "K_t0p985", "K_t0p992", "K_t1p000", "K_t1p008",
        "K_t1p015", "K_t1p025", "K_t1p040", "K_t1p060",
    )
    qpoints_zero = ("G_t0p000", "G_t0p015", "K_t0p985", "K_t1p000", "K_t1p015")
    dyn_specs = {
        "FD300_K144": (("dg0.0019000869_k144", qpoints_conv),),
        "FD300_LINE": (("dg0.0019000869_k144", qpoints_line),),
        "FD600_LINE": (("dg0.0038001738_k120", qpoints_line),),
        "FD0_SCAN": (
            ("dg0.0006333623_k192", qpoints_zero),
            ("dg0.0012667246_k144", qpoints_zero),
            ("dg0.0019000869_k120", qpoints_zero),
        ),
    }
    for set_slug, qpoints in dyn_specs.get(campaign, ()):
        set_destination = destination / "dynamical_matrices" / set_slug
        scf_input = set_destination / "scf.in"
        if not scf_input.is_file():
            copy_remote(
                target,
                f"/data/graphene_physical_fd_dfpt/sets/{set_slug}/scf.in",
                scf_input,
            )
        for qpoint in qpoints:
            for filename in ("ph.in", "gr.dyn"):
                local_path = set_destination / qpoint / filename
                if not local_path.is_file():
                    copy_remote(
                        target,
                        f"/data/graphene_physical_fd_dfpt/sets/{set_slug}/{qpoint}/{filename}",
                        local_path,
                    )


def sync_force_convergence(target: str, lane: str, expected_reference_k: int = 12) -> None:
    destination = ROOT / "results" / "graphene_fd_force_convergence" / lane
    summary = destination / "summary.json"
    if summary.is_file():
        try:
            if int(json.loads(summary.read_text())["reference_kgrid"]) >= expected_reference_k:
                return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    for filename in ("summary.json", "summary.xyz", "run.log"):
        local_path = destination / filename
        copy_remote(
            target,
            f"/data/graphene_fd_force_convergence/{lane}/{filename}",
            local_path,
        )


def sync_thermal_labels(target: str, lane: str) -> None:
    destination = ROOT / "results" / "graphene_fd_thermal_labels" / lane
    summary = destination / "summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text())
            if len(payload["indices"]) == 15 and len(payload["training_indices"]) == 12:
                return
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    for filename in ("summary.json", "summary.xyz", "run.log"):
        copy_remote(
            target,
            f"/data/graphene_fd_thermal_labels/{lane}/{filename}",
            destination / filename,
        )


def sync_thermal_label_wave(target: str, lane: str, wave: int) -> None:
    remote_root = f"/data/graphene_fd_thermal_labels_wave{wave}/{lane}"
    destination = ROOT / "results" / f"graphene_fd_thermal_labels_wave{wave}" / lane
    summary = destination / "summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text())
            if len(payload["indices"]) == 15:
                return
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    for filename in ("summary.json", "summary.xyz", "run.log"):
        copy_remote(target, f"{remote_root}/{filename}", destination / filename)


def stage_thermal_finetune_if_ready() -> None:
    local_base = ROOT / "results" / "graphene_fd_thermal_labels"
    sources = {}
    for lane in ("A", "B"):
        summary = local_base / lane / "summary.json"
        xyz = local_base / lane / "summary.xyz"
        if not (summary.is_file() and xyz.is_file()):
            return
        payload = json.loads(summary.read_text())
        if len(payload.get("training_indices", [])) != 12:
            raise ValueError(f"lane {lane} does not contain 12 thermal training labels")
        if len(payload.get("validation_indices", [])) != 3:
            raise ValueError(f"lane {lane} does not contain 3 thermal validation labels")
        sources[lane] = (summary, xyz)

    remote_base = "/home/howardwang/phonon/data/graphene_fd_thermal_labels"
    ready = ssh(RTX_2060, f"test -e {remote_base}/.READY", check=False)
    if ready.returncode == 0:
        return
    for lane, (summary, xyz) in sources.items():
        copy_to_remote(summary, RTX_2060, f"{remote_base}/{lane}.json")
        copy_to_remote(xyz, RTX_2060, f"{remote_base}/{lane}.xyz")
    verification = ssh(
        RTX_2060,
        rf'''/home/howardwang/miniconda3/bin/conda run -n phonon python -c '
import sys
from ase.io import read
for path in sys.argv[1:]:
    atoms = read(path, index=":")
    assert len(atoms) == 15
    assert sum(atom.info.get("split") == "train" for atom in atoms) == 12
    assert sum(atom.info.get("split") == "validation" for atom in atoms) == 3
' {remote_base}/A.xyz {remote_base}/B.xyz''',
        check=False,
    )
    if verification.returncode != 0:
        raise RuntimeError(f"2060 thermal-label validation failed: {verification.stdout[-1000:]}")
    ssh(
        RTX_2060,
        f"touch {remote_base}/.READY; "
        "systemctl --user restart phonon-graphene-fd-thermal-finetune.service",
    )
    log("both physical-FD thermal label sets staged; 2060 fine-tune queue released")


def sync_2060_thermal_finetune() -> None:
    destination = ROOT / "results" / "graphene_fd_thermal_finetune"
    targets = (
        (
            "/home/howardwang/phonon/results/graphene_fd_thermal_finetune/force_validation.json",
            destination / "force_validation.json",
        ),
        (
            "/home/howardwang/phonon/data/graphene_fd_thermal_finetune/manifest.json",
            destination / "data_manifest.json",
        ),
        (
            "/home/howardwang/phonon/results/graphene_fd_thermal_finetune/run.log",
            destination / "run.log",
        ),
    )
    td = ROOT / "results" / "td_phonon"
    for temperature in (300, 600):
        tag = f"graphene_v11_fd{temperature}_short_range"
        for suffix in ("npz", "csv"):
            targets += (
                (
                    f"/home/howardwang/phonon/results/td_phonon/td_{tag}.{suffix}",
                    td / f"td_{tag}.{suffix}",
                ),
            )
    for remote, local in targets:
        if not local.is_file():
            copy_remote(RTX_2060, remote, local)


def sync_2060_refine(wave: int) -> None:
    tag = f"wave{wave}"
    remote_out = f"/home/howardwang/phonon/results/graphene_fd_thermal_finetune_{tag}"
    destination = ROOT / "results" / f"graphene_fd_thermal_finetune_{tag}"
    targets: list[tuple[str, Path]] = [
        (f"{remote_out}/acceptance.json", destination / "acceptance.json"),
        (f"{remote_out}/force_validation.json", destination / "force_validation.json"),
        (f"{remote_out}/run.log", destination / "run.log"),
        (
            f"/home/howardwang/phonon/data/graphene_fd_thermal_finetune_{tag}/manifest.json",
            destination / "data_manifest.json",
        ),
    ]
    td = ROOT / "results" / "td_phonon"
    reference_waves = (1, 2) if wave == 2 else (3,)
    for temperature in (300, 600):
        for reference_wave in reference_waves:
            stem = f"graphene_physical_fd_dft_{temperature}K_wave{reference_wave}"
            for suffix in ("npz", "csv", "json"):
                targets.append(
                    (
                        f"/home/howardwang/phonon/results/td_phonon/{stem}.{suffix}",
                        td / f"{stem}.{suffix}",
                    )
                )
        stem = f"td_graphene_v11_fd{temperature}_{tag}_short_range_seed"
        for seed in (0, 1, 2):
            for suffix in ("npz", "csv"):
                targets.append(
                    (
                        f"/home/howardwang/phonon/results/td_phonon/{stem}{seed}.{suffix}",
                        td / f"{stem}{seed}.{suffix}",
                    )
                )
    for remote, local in targets:
        if not local.is_file():
            copy_remote(RTX_2060, remote, local)


def sync_2060_refine_failure() -> None:
    remote_out = "/home/howardwang/phonon/results/graphene_fd_thermal_finetune_wave3"
    destination = ROOT / "results" / "graphene_fd_thermal_finetune_wave3"
    targets: list[tuple[str, Path]] = [
        (f"{remote_out}/acceptance.json", destination / "acceptance.json"),
        (f"{remote_out}/force_validation.json", destination / "force_validation.json"),
        (f"{remote_out}/run.log", destination / "run.log"),
        (
            "/home/howardwang/phonon/data/graphene_fd_thermal_finetune_wave3/manifest.json",
            destination / "data_manifest.json",
        ),
    ]
    td = ROOT / "results" / "td_phonon"
    for temperature in (300, 600):
        stem = f"graphene_physical_fd_dft_{temperature}K_wave3"
        for suffix in ("npz", "csv", "json"):
            targets.append(
                (
                    f"/home/howardwang/phonon/results/td_phonon/{stem}.{suffix}",
                    td / f"{stem}.{suffix}",
                )
            )
    for seed in (0, 1, 2):
        stem = f"td_graphene_v11_fd300_wave3_short_range_seed{seed}"
        for suffix in ("npz", "csv"):
            targets.append(
                (
                    f"/home/howardwang/phonon/results/td_phonon/{stem}.{suffix}",
                    td / f"{stem}.{suffix}",
                )
            )
    for remote, local in targets:
        if not local.is_file():
            copy_remote(RTX_2060, remote, local)


def sync_2060_model_ablation() -> None:
    remote_out = "/home/howardwang/phonon/results/graphene_fd_model_ablation"
    destination = ROOT / "results" / "graphene_fd_model_ablation"
    targets: list[tuple[str, Path]] = [
        (f"{remote_out}/selection.json", destination / "selection.json"),
        (
            f"{remote_out}/variant_force_metrics.json",
            destination / "variant_force_metrics.json",
        ),
        (f"{remote_out}/force_validation.json", destination / "force_validation.json"),
        (f"{remote_out}/acceptance.json", destination / "acceptance.json"),
        (f"{remote_out}/aggregate.log", destination / "aggregate.log"),
        (f"{remote_out}/T300/run.log", destination / "T300_run.log"),
        (f"{remote_out}/T600/run.log", destination / "T600_run.log"),
        (
            "/home/howardwang/phonon/data/graphene_fd_model_ablation/replay0/manifest.json",
            destination / "replay0_manifest.json",
        ),
        (
            "/home/howardwang/phonon/data/graphene_fd_model_ablation/replay12/manifest.json",
            destination / "replay12_manifest.json",
        ),
    ]
    td = ROOT / "results" / "td_phonon"
    for temperature in (300, 600):
        for seed in (0, 1, 2):
            stem = f"td_graphene_v11_fd{temperature}_ablation_short_range_seed{seed}"
            for suffix in ("npz", "csv"):
                targets.append(
                    (
                        f"/home/howardwang/phonon/results/td_phonon/{stem}.{suffix}",
                        td / f"{stem}.{suffix}",
                    )
                )
    for remote, local in targets:
        if local.is_file():
            continue
        if ssh(RTX_2060, f"test -s {remote}", check=False).returncode == 0:
            copy_remote(RTX_2060, remote, local)


def sync_2060_model_ablation_v2() -> None:
    remote_out = "/home/howardwang/phonon/results/graphene_fd_model_ablation_v2"
    destination = ROOT / "results" / "graphene_fd_model_ablation_v2"
    targets: list[tuple[str, Path]] = [
        (f"{remote_out}/selection.json", destination / "selection.json"),
        (f"{remote_out}/force_validation.json", destination / "force_validation.json"),
        (f"{remote_out}/acceptance.json", destination / "acceptance.json"),
        (f"{remote_out}/aggregate.log", destination / "aggregate.log"),
        (f"{remote_out}/T300/selection.json", destination / "T300_selection.json"),
        (
            f"{remote_out}/T300/selected_gate_metrics.json",
            destination / "T300_gate_metrics.json",
        ),
        (f"{remote_out}/T300/run.log", destination / "T300_run.log"),
        (f"{remote_out}/T600/selection.json", destination / "T600_selection.json"),
        (
            f"{remote_out}/T600/selected_gate_metrics.json",
            destination / "T600_gate_metrics.json",
        ),
        (f"{remote_out}/T600/run.log", destination / "T600_run.log"),
        (
            "/home/howardwang/phonon/data/graphene_fd_model_ablation_v2/replay72_joint/manifest.json",
            destination / "data_manifest.json",
        ),
    ]
    td = ROOT / "results" / "td_phonon"
    for temperature in (300, 600):
        for seed in (0, 1, 2):
            stem = (
                f"td_graphene_v11_fd{temperature}_ablation_v2_short_range_seed{seed}"
            )
            for suffix in ("npz", "csv"):
                targets.append(
                    (
                        f"/home/howardwang/phonon/results/td_phonon/{stem}.{suffix}",
                        td / f"{stem}.{suffix}",
                    )
                )
    for remote, local in targets:
        if local.is_file():
            continue
        if ssh(RTX_2060, f"test -s {remote}", check=False).returncode == 0:
            copy_remote(RTX_2060, remote, local)


def sync_2060_final_postprocess() -> None:
    groups = (
        (
            "/home/howardwang/phonon/results/graphene_physical_fd_long_range",
            ROOT / "results" / "graphene_physical_fd_long_range",
            (
                "physical_fd_long_range_predictions.csv",
                "physical_fd_long_range_summary.json",
                "physical_fd_long_range_comparison.png",
                "physical_fd_long_range_comparison.pdf",
                "run.log",
            ),
        ),
        (
            "/home/howardwang/phonon/results/graphene_physical_fd0",
            ROOT / "results" / "graphene_physical_fd0",
            (
                "graphene_fd0_extrapolation.csv",
                "graphene_fd0_extrapolation.json",
                "graphene_fd0_extrapolation.png",
                "graphene_fd0_extrapolation.pdf",
            ),
        ),
    )
    for remote_root, destination, filenames in groups:
        for filename in filenames:
            local = destination / filename
            if not local.is_file():
                copy_remote(RTX_2060, f"{remote_root}/{filename}", local)


def analyze_physical_fd_if_ready() -> None:
    base = ROOT / "results" / "graphene_physical_fd_dfpt" / "campaigns"
    required = [
        base / campaign / f"graphene_{campaign}_dfpt.csv"
        for campaign in ("FD300_CONV", "FD600_CONV")
    ]
    if not all(path.is_file() for path in required):
        return
    inputs = list(required)
    k144 = base / "FD300_K144" / "graphene_FD300_K144_dfpt.csv"
    if k144.is_file():
        inputs.append(k144)
    output = ROOT / "results" / "graphene_physical_fd_dfpt" / "convergence_summary.json"
    if output.is_file() and output.stat().st_mtime >= max(path.stat().st_mtime for path in inputs):
        return
    input_arguments = []
    for path in inputs:
        input_arguments.extend(("--input", path))
    conda_python(
        ROOT / "scripts" / "smearing_kink" / "analyze_graphene_physical_fd_convergence.py",
        *input_arguments,
        "--output",
        output,
    )


def maybe_restart(
    name: str,
    target: str,
    command: str,
    restart_counts: dict[str, int],
    maximum: int = 3,
) -> None:
    count = int(restart_counts.get(name, 0))
    if count >= maximum:
        log(f"{name}: local restart cap {maximum} reached; leaving evidence intact")
        return
    result = ssh(target, command, check=False)
    if result.returncode == 0:
        restart_counts[name] = count + 1
        log(f"{name}: requested restart ({count + 1}/{maximum})")
    else:
        log(f"{name}: restart request failed: {result.stdout.strip()[-500:]}")


def latest_weekly() -> Path:
    candidates = sorted((ROOT / "docs").glob("WEEKLY_????-??-??.md"))
    if not candidates:
        raise FileNotFoundError("no docs/WEEKLY_YYYY-MM-DD.md found")
    return candidates[-1]


def status_text(statuses: dict[str, dict[str, str]]) -> dict[str, str]:
    a, b, gpu = statuses["v100_a"], statuses["v100_b"], statuses["rtx_2060"]

    def compact_training_progress(value: str) -> str:
        match = re.search(r"Epoch\s+(\d+)", value or "")
        if match:
            return f"epoch {match.group(1)}"
        if "selected variant=" in (value or ""):
            return "模型筛选完成"
        return value or "等待启动"
    if b.get("reachable") != "1":
        b_text = "暂时无法连接；远端 watchdog 继续独立运行"
    elif b.get("done") == "1":
        b_text = "完成；60/60 snapshots"
    else:
        progress = b.get("progress") or "尚未读到 checkpoint"
        b_text = f"{b.get('service', 'unknown')}；{progress}"

    if a.get("reachable") != "1":
        fullpath_text = cold_text = "暂时无法连接"
    else:
        fullpath_text = (
            "完成；8/8 q points"
            if a.get("fullpath_done") == "1"
            else f"{a.get('fullpath_service', 'unknown')}；"
            f"{a.get('fullpath_points', '?')}/8 q points 完成"
        )
        cold_text = (
            "完成；5/5 exact-setting pairs"
            if a.get("cold_done") == "1"
            else f"主队列 {a.get('cold_service', 'unknown')}，"
            f"verifier {a.get('cold_verifier', 'unknown')}；"
            f"{a.get('cold_pairs', '?')}/5 pairs 完成"
        )

    if gpu.get("reachable") != "1":
        deploy_text = control_text = fd_train_text = "暂时无法连接"
        wave2_text = wave3_text = ablation_text = v2_text = post_text = "暂时无法连接"
        weighted600_text = "暂时无法连接"
    elif gpu.get("file_ready") == "1" and gpu.get("service") != "active":
        deploy_text = "完成；15/15 smearing points"
    else:
        deploy_text = f"{gpu.get('service', 'unknown')}；15 点 MLIP 部署"
    if gpu.get("reachable") == "1":
        if gpu.get("control_done") == "1":
            if (
                gpu.get("control_seed_service") == "active"
                and gpu.get("fd_train_ready") != "1"
            ):
                control_text = (
                    "主对照完成；独立 seed "
                    f"{gpu.get('control_seed_count', '?')}/8，低优先级运行"
                )
            else:
                control_text = "完成；300/600 K 同底座、无长程项 TDEP"
        else:
            control_text = (
                f"{gpu.get('control_service', 'unknown')}；"
                f"{gpu.get('control_progress', '尚未读到 checkpoint')}"
            )
        if gpu.get("fd_train_done") == "1":
            fd_train_text = "完成；300/600 K 微调、验证集 force、TDEP"
        elif gpu.get("fd_train_ready") == "1":
            fd_train_text = (
                f"{gpu.get('fd_train_service', 'unknown')}；"
                f"{gpu.get('fd_train_progress', '已收到 labels')}"
            )
        else:
            fd_train_text = (
                f"{gpu.get('fd_train_service', 'unknown')}；"
                "等待 V100-A/B 的 physical-FD labels，完成后自动训练"
            )
        if gpu.get("wave2_done") == "1":
            wave2_text = "完成；24 train + 6 validation，三组 TDEP seeds"
        elif gpu.get("wave2_ready") == "1":
            wave2_text = (
                f"{gpu.get('wave2_service', 'unknown')}；"
                f"{gpu.get('wave2_progress', '已收到 wave2 labels')}"
            )
        else:
            wave2_text = "队列已启用；等待两台 V100 的第二批 labels"
        if gpu.get("wave3_failed") == "1":
            wave3_text = "最终短程门槛未通过；停止使用不稳定的 600 K 轨迹"
        elif gpu.get("wave3_skipped") == "1":
            wave3_text = "未触发；wave2 已达到固定门槛"
        elif gpu.get("wave3_done") == "1":
            wave3_text = "完成；wave2 未达标后补充到 36 train + 9 validation"
        elif gpu.get("wave3_ready") == "1":
            wave3_text = (
                f"{gpu.get('wave3_service', 'unknown')}；"
                f"{gpu.get('wave3_progress', '已收到条件 labels')}"
            )
        else:
            wave3_text = "条件队列已启用；仅在 wave2 未达标时运行"
        if gpu.get("ablation_done") == "1":
            if gpu.get("ablation_passed") == "1":
                ablation_text = "完成并通过短程门槛；长程修正队列已接续"
            elif gpu.get("ablation_force_blocked") == "1":
                ablation_text = "完成；没有模型通过独立受力与谐波保持门槛"
            elif gpu.get("ablation_stability_blocked") == "1":
                ablation_text = "完成；独立受力通过，但 600 K 轨迹仍不稳定"
            else:
                ablation_text = "完成；TDEP 与 DFT 的定量门槛未通过"
        else:
            if b.get("reachable") != "1":
                b_lane = "暂时无法连接 V100-B"
            elif gpu.get("ablation_selection") == "1":
                b_lane = (
                    "600 K 三种子 TDEP 完成"
                    if b.get("ablation_tdep_done") == "1"
                    else f"600 K TDEP {b.get('ablation_tdep_service', 'unknown')}，"
                    f"{b.get('ablation_tdep_progress', '等待模型筛选')}"
                )
            else:
                b_lane = (
                    "600 K 模型训练完成，等待统一筛选"
                    if b.get("ablation_done") == "1"
                    else f"600 K {b.get('ablation_service', 'unknown')}，"
                    f"{b.get('ablation_progress', '等待启动')}"
                )
            ablation_text = (
                f"300 K {gpu.get('ablation_300_service', 'unknown')}，"
                f"{gpu.get('ablation_300_progress', '等待启动')}；{b_lane}；"
                f"筛选/TDEP {gpu.get('ablation_service', 'unknown')}"
            )
        if gpu.get("v2_started") == "1":
            ablation_text = (
                "已停止；300 K 三种旧配比均明显破坏 v11 谐波曲率，"
                "已转入回放 + 联合早停的原始总力诊断"
            )
        if gpu.get("v2_started") == "1":
            if gpu.get("v2_done") == "1":
                if gpu.get("v2_force_blocked") == "1":
                    v2_text = "诊断完成；至少一个温度没有模型同时通过热受力与谐波保持门槛"
                else:
                    v2_text = (
                        "诊断完成；原始总力模型不作为短程背景，"
                        "已跳过 TDEP，等待长程力扣除后的残差训练"
                    )
            else:
                lane300 = (
                    "300 K 原始总力诊断已选出候选"
                    if gpu.get("v2_300_passed") == "1"
                    else (
                        "300 K 三种方案均未通过"
                        if gpu.get("v2_300_blocked") == "1"
                        else f"300 K {gpu.get('v2_300_service', 'unknown')}，"
                        f"{gpu.get('v2_300_variant', 'pending')}，"
                        f"{compact_training_progress(gpu.get('v2_300_progress', ''))}"
                    )
                )
                if b.get("reachable") != "1":
                    lane600 = "600 K 暂时无法连接"
                elif b.get("v2_passed") == "1":
                    lane600 = "600 K 原始总力诊断已选出候选"
                elif b.get("v2_blocked") == "1":
                    lane600 = "600 K 三种方案均未通过"
                else:
                    lane600 = (
                        f"600 K {b.get('v2_service', 'unknown')}，"
                        f"{b.get('v2_variant', 'pending')}，"
                        f"{compact_training_progress(b.get('v2_progress', ''))}"
                    )
                v2_text = (
                    f"{lane300}；{lane600}；诊断聚合 "
                    f"{gpu.get('v2_aggregate_service', 'unknown')}"
                )
        else:
            v2_text = "等待原始总力诊断队列启动"
        if gpu.get("v2_started") == "1" and gpu.get("v2_done") != "1":
            post_text = "等待物理 smearing 长程力标定及残差短程模型"
        elif gpu.get("long_blocked") == "1":
            post_text = "短程 wave3 仍未达标；已保留结果，未做无效长程拟合"
        elif gpu.get("post_done") == "1":
            post_text = "完成；长程组留出验证和 degauss→0 参考"
        elif gpu.get("long_done") == "1":
            post_text = (
                "长程验证完成；等待 FD0_SCAN 后自动做 degauss→0 外推"
            )
        else:
            post_text = (
                f"{gpu.get('post_service', 'unknown')}；"
                f"{gpu.get('post_progress', '等待 dense DFPT 和短程验收')}"
            )
        if gpu.get("weighted_calibrated_done") == "1":
            if gpu.get("weighted_calibrated_passed") == "1":
                weighted600_text = (
                    "完成；独立 force、三 seed spread 与 600 K 有限温校准门槛通过；"
                    "静态修正原样迁移对照未通过"
                )
                if gpu.get("post_done") != "1":
                    post_text = (
                        "旧 wave3 路径不再用于 600 K 结论；"
                        "degauss→0 参考仍由独立队列继续"
                    )
            else:
                weighted600_text = "完成；有限温校准门槛未通过"
        elif gpu.get("weighted_tdep_done") == "1":
            weighted600_text = "三 seed 与静态迁移对照完成；等待有限温校准"
        elif gpu.get("weighted_force_done") == "1":
            if gpu.get("weighted_force_passed") == "1":
                weighted600_text = (
                    "独立 force gate 通过；"
                    f"{gpu.get('weighted_progress', '三 seed TDEP 运行中')}"
                )
            else:
                weighted600_text = "独立 force gate 未通过；未启动三 seed TDEP"
        else:
            weighted600_text = gpu.get(
                "weighted_progress", "等待冻结模型和独立 force holdout"
            )

    def physical_fd_text(peer: dict[str, str]) -> str:
        if peer.get("reachable") != "1":
            return "暂时无法通过 Tailscale 跳板连接"
        if peer.get("fd_done") == "1":
            return "完成；3 个 k 网格、24 个 q points"
        return (
            f"{peer.get('fd_service', 'unknown')}；"
            f"SCF {peer.get('fd_scf', '?')}/3，"
            f"DFPT {peer.get('fd_ph', '?')}/24"
        )

    def force_text(peer: dict[str, str]) -> str:
        if peer.get("reachable") != "1":
            return "暂时无法通过 Tailscale 跳板连接"
        if peer.get("force_done") == "1":
            return "完成；12/12 thermal-snapshot force labels"
        return (
            f"{peer.get('force_service', 'unknown')}；"
            f"{peer.get('force_count', '?')}/12 labels"
        )

    def campaign_text(
        peer: dict[str, str],
        *,
        done_key: str,
        service_key: str,
        count_key: str,
        expected: int,
        unit: str = "q points",
        waiting_for: str | None = None,
    ) -> str:
        if peer.get("reachable") != "1":
            return "暂时无法通过 Tailscale 跳板连接"
        if peer.get(done_key) == "1":
            return f"完成；{expected}/{expected} {unit}"
        service = peer.get(service_key, "unknown")
        if waiting_for and service != "active":
            return f"等待 {waiting_for} 完成后自动启动"
        return f"{service}；{peer.get(count_key, '?')}/{expected} {unit}"

    a_k144 = campaign_text(
        a,
        done_key="k144_done",
        service_key="k144_service",
        count_key="k144_ph",
        expected=8,
    )
    a_line = campaign_text(
        a,
        done_key="line_done",
        service_key="line_service",
        count_key="line_ph",
        expected=19,
        waiting_for="300 K k=144 确认" if a.get("k144_done") != "1" else None,
    )
    b_line = campaign_text(
        b,
        done_key="line_done",
        service_key="line_service",
        count_key="line_ph",
        expected=19,
    )
    b_fd0 = campaign_text(
        b,
        done_key="fd0_done",
        service_key="fd0_service",
        count_key="fd0_ph",
        expected=15,
        waiting_for="600 K 加密线" if b.get("line_done") != "1" else None,
    )
    a_thermal = campaign_text(
        a,
        done_key="thermal_done",
        service_key="thermal_service",
        count_key="thermal_count",
        expected=15,
        unit="labels",
    )
    b_thermal = campaign_text(
        b,
        done_key="thermal_done",
        service_key="thermal_service",
        count_key="thermal_count",
        expected=15,
        unit="labels",
    )

    def thermal_expansion_text(peer: dict[str, str], wave: int) -> str:
        if peer.get("reachable") != "1":
            return "暂时无法通过 Tailscale 跳板连接"
        if wave == 3 and peer.get("thermal3_skipped") == "1":
            return "未触发；wave2 已达到固定门槛"
        prefix = f"thermal{wave}"
        if peer.get(f"{prefix}_done") == "1":
            return "完成；15/15 labels"
        if wave == 2:
            waiting = "等待第一批 labels 完成后自动启动"
        else:
            waiting = "条件队列已启用；仅在 wave2 未达标时运行"
        count = peer.get(f"{prefix}_count", "?")
        service = peer.get(f"{prefix}_service", "unknown")
        if count == "0":
            return waiting
        return f"{service}；{count}/15 labels"

    return {
        "V100-A · 300 K k=144 确认": a_k144,
        "V100-B · 600 K 加密 Γ/K 线": b_line,
        "V100-A · 300 K 加密 Γ/K 线": a_line,
        "V100-B · degauss→0 扫描": b_fd0,
        "V100-A · 300 K physical-FD thermal labels": a_thermal,
        "V100-B · 600 K physical-FD thermal labels": b_thermal,
        "V100-A · 第二批 300 K thermal labels": thermal_expansion_text(a, 2),
        "V100-B · 第二批 600 K thermal labels": thermal_expansion_text(b, 2),
        "V100-A/B · 条件第三批 thermal labels": (
            f"A: {thermal_expansion_text(a, 3)}；B: {thermal_expansion_text(b, 3)}"
        ),
        "V100-A · physical FD 0.0019000869 Ry direct-DFPT": physical_fd_text(a),
        "V100-B · physical FD 0.0038001738 Ry direct-DFPT": physical_fd_text(b),
        "V100-A · 300 K physical-FD force k-grid": force_text(a),
        "V100-B · 600 K physical-FD force k-grid": force_text(b),
        "RTX 2060 · v11 同底座无长程项对照": control_text,
        "RTX 2060 · physical-FD thermal fine-tune": fd_train_text,
        "RTX 2060 · wave2 扩充训练与 DFT-label TDEP": wave2_text,
        "RTX 2060 · 条件 wave3": wave3_text,
        "2060/V100-B · 固定数据模型消融与三种子 TDEP": ablation_text,
        "2060/V100-B · 保守微调（回放 + 联合早停）": v2_text,
        "V100-A/RTX 2060 · 600 K weighted replay 与有限温 q-space": weighted600_text,
        "RTX 2060 · q-space 长程项与 degauss→0": post_text,
        "V100-B · 600 K DFT-MD/TDEP": b_text,
        "V100-A · graphene 完整路径 pilot": fullpath_text,
        "V100-A · TMD cold/fd 配对": cold_text,
        "RTX 2060 · graphene 15 点 MLIP": deploy_text,
    }


def local_result_lines() -> tuple[list[str], dict]:
    lines = []
    fingerprint = {}
    physical_rows = {}
    physical_campaigns = {
        "FD300_CONV": 24,
        "FD600_CONV": 24,
        "FD300_K144": 8,
        "FD300_LINE": 19,
        "FD600_LINE": 19,
        "FD0_SCAN": 15,
    }
    for campaign, expected_rows in physical_campaigns.items():
        path = (
            ROOT
            / "results"
            / "graphene_physical_fd_dfpt"
            / "campaigns"
            / campaign
            / f"graphene_{campaign}_dfpt.csv"
        )
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        dgs = sorted({float(row["degauss_Ry"]) for row in rows})
        kgrids = sorted({int(row["kgrid"]) for row in rows})
        physical_rows[campaign] = len(rows)
        lines.append(
            f"{campaign} direct-DFPT：`{len(rows)}/{expected_rows}` 个 q points；"
            f"Fermi–Dirac smearing `{dgs[0]:.10f} Ry`，"
            f"k 网格 `{','.join(map(str, kgrids))}`。"
        )
    if physical_rows:
        fingerprint["physical_fd_dfpt_rows"] = physical_rows
    convergence_path = (
        ROOT / "results" / "graphene_physical_fd_dfpt" / "convergence_summary.json"
    )
    if convergence_path.is_file():
        convergence = json.loads(convergence_path.read_text())
        for degauss, entry in convergence["by_degauss"].items():
            comparisons = entry.get("adjacent_k_comparisons", [])
            if not comparisons:
                continue
            latest = comparisons[-1]
            region_max = max(
                values["top_branch_max_abs_cm-1"]
                for values in latest["regions"].values()
                if values.get("n_q", 0)
            )
            lines.append(
                f"Fermi–Dirac smearing `{float(degauss):.10f} Ry` 的 k 网格检查："
                f"`k={latest['lower_kgrid']}→{latest['upper_kgrid']}` 时，"
                f"Γ/K 顶支最大变化 `{region_max:.2f} cm^-1`；"
                f"下一步为 `{entry['recommended_next_campaign']}`。"
            )
        fingerprint["physical_fd_dfpt_convergence"] = convergence

    force_metrics = {}
    for lane in ("A", "B"):
        path = ROOT / "results" / "graphene_fd_force_convergence" / lane / "summary.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        reference_k = int(payload["reference_kgrid"])
        lower_k = sorted(int(value) for value in payload["kgrids"])[-2]
        comparisons = [
            row for row in payload["records"] if int(row["kgrid"]) == lower_k
        ]
        max_rmse = max(float(row["force_RMSE_meV_A"]) for row in comparisons)
        max_component = max(float(row["force_max_abs_meV_A"]) for row in comparisons)
        lines.append(
            f"{int(payload['lattice_temperature_K'])} K thermal snapshots 的 physical-FD force 收敛："
            f"`k={lower_k}` 相对 `k={reference_k}` 的最大构型 RMSE "
            f"`{max_rmse:.1f} meV/Å`，最大分量差 `{max_component:.1f} meV/Å`。"
        )
        force_metrics[lane] = payload
    if force_metrics:
        fingerprint["physical_fd_force_convergence"] = force_metrics

    thermal_metrics = {}
    for lane in ("A", "B"):
        path = ROOT / "results" / "graphene_fd_thermal_labels" / lane / "summary.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        lines.append(
            f"{int(payload['lattice_temperature_K'])} K physical-FD thermal labels："
            f"`{len(payload['training_indices'])}` 个训练构型、"
            f"`{len(payload['validation_indices'])}` 个 validation 构型；"
            f"Fermi–Dirac smearing `{float(payload['degauss_Ry']):.10f} Ry`，"
            f"k 网格 `{int(payload['reference_kgrid'])}`。"
        )
        thermal_metrics[lane] = payload
    if thermal_metrics:
        fingerprint["physical_fd_thermal_labels"] = thermal_metrics

    validation_path = (
        ROOT
        / "results"
        / "graphene_fd_thermal_finetune"
        / "force_validation.json"
    )
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text())
        for temperature, trained_label in (("300", "fd300"), ("600", "fd600")):
            baseline = validation["models"]["v11"]["by_temperature"][temperature]
            trained = validation["models"][trained_label]["by_temperature"][temperature]
            lines.append(
                f"{temperature} K physical-FD validation-set force：同一 v11 backbone 微调前/后 "
                f"RMSE `{baseline['RMSE_meV_A']:.1f}→{trained['RMSE_meV_A']:.1f} meV/Å`，"
                f"微调后最大分量误差 `{trained['max_abs_meV_A']:.1f} meV/Å`。"
            )
        fingerprint["physical_fd_thermal_finetune_validation"] = validation

    for wave in (2, 3):
        acceptance_path = (
            ROOT
            / "results"
            / f"graphene_fd_thermal_finetune_wave{wave}"
            / "acceptance.json"
        )
        if not acceptance_path.is_file():
            continue
        acceptance = json.loads(acceptance_path.read_text())
        if acceptance.get("status") == "final_short_range_failure":
            temperature_parts = []
            for temperature in ("300", "600"):
                metrics = acceptance["by_temperature"][temperature][
                    "force_validation"
                ]
                temperature_parts.append(
                    f"{temperature} K force RMSE {metrics['RMSE_meV_A']:.1f} meV/Å，"
                    f"最大分量 {metrics['max_abs_meV_A']:.1f} meV/Å"
                )
            lines.append(
                "physical-FD wave3 最终短程门槛未通过："
                + "；".join(temperature_parts)
                + "；600 K 轨迹稳定性检查未通过。"
            )
            fingerprint[f"physical_fd_wave{wave}_acceptance"] = acceptance
            continue
        outcome = "通过" if acceptance["passes_short_range_gate"] else "未通过"
        temperature_parts = []
        for temperature in ("300", "600"):
            item = acceptance["by_temperature"][temperature]
            trained = item["force_validation"]
            mlip_key = next(key for key in item if key.startswith("MLIP_seed_vs_DFT_"))
            max_mlip_mae = max(
                metric["full_band_MAE_cm-1"] for metric in item[mlip_key]
            )
            temperature_parts.append(
                f"{temperature} K force RMSE {trained['RMSE_meV_A']:.1f} meV/Å，"
                f"MLIP–DFT 全谱 MAE 最大 {max_mlip_mae:.1f} cm^-1"
            )
        lines.append(
            f"physical-FD wave{wave} 固定门槛{outcome}："
            + "；".join(temperature_parts)
            + "。"
        )
        fingerprint[f"physical_fd_wave{wave}_acceptance"] = acceptance

    ablation_root = ROOT / "results" / "graphene_fd_model_ablation"
    selection_path = ablation_root / "selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        parts = []
        for temperature in ("300", "600"):
            item = selection["by_temperature"][temperature]["selected"]
            thermal = item["thermal_force"]
            parts.append(
                f"{temperature} K `{item['label']}`，test force RMSE/最大分量 "
                f"{thermal['RMSE_meV_A']:.1f}/{thermal['max_abs_meV_A']:.1f} meV/Å"
            )
        lines.append("固定数据模型配比比较：" + "；".join(parts) + "。")
        fingerprint["physical_fd_model_ablation_selection"] = selection
    ablation_acceptance_path = ablation_root / "acceptance.json"
    if ablation_acceptance_path.is_file():
        acceptance = json.loads(ablation_acceptance_path.read_text())
        outcome = "通过" if acceptance["passes_short_range_gate"] else "未通过"
        details = []
        for temperature in ("300", "600"):
            if temperature not in acceptance.get("by_temperature", {}):
                continue
            item = acceptance["by_temperature"][temperature]
            mlip_key = next(
                (key for key in item if key.startswith("MLIP_seed_vs_DFT_")),
                None,
            )
            if mlip_key:
                maximum = max(
                    metric["full_band_MAE_cm-1"] for metric in item[mlip_key]
                )
                details.append(f"{temperature} K 最大全谱 MAE {maximum:.1f} cm^-1")
        lines.append(
            f"模型配比复查后的三种子 TDEP {outcome}短程门槛"
            + ("：" + "；".join(details) if details else "")
            + "。"
        )
        fingerprint["physical_fd_model_ablation_acceptance"] = acceptance

    conservative_root = ROOT / "results" / "graphene_fd_model_ablation_v2"
    conservative_selection_path = conservative_root / "selection.json"
    if conservative_selection_path.is_file():
        selection = json.loads(conservative_selection_path.read_text())
        parts = []
        for temperature in ("300", "600"):
            selected = selection["by_temperature"][temperature]["selected"]
            thermal = selected["thermal_force"]
            harmonic = selected["harmonic_force"]
            parts.append(
                f"{temperature} K `{selected['label']}`：test force RMSE/最大分量 "
                f"{thermal['RMSE_meV_A']:.1f}/{thermal['max_abs_meV_A']:.1f} meV/Å，"
                f"谐波保持 RMSE {harmonic['RMSE_meV_A']:.1f} meV/Å"
            )
        lines.append("保守微调模型：" + "；".join(parts) + "。")
        fingerprint["physical_fd_conservative_selection"] = selection
    conservative_acceptance_path = conservative_root / "acceptance.json"
    if conservative_acceptance_path.is_file():
        acceptance = json.loads(conservative_acceptance_path.read_text())
        outcome = "通过" if acceptance["passes_short_range_gate"] else "未通过"
        lines.append(f"保守微调模型的三种子 TDEP {outcome}短程背景门槛。")
        fingerprint["physical_fd_conservative_acceptance"] = acceptance

    long_range_path = (
        ROOT
        / "results"
        / "graphene_physical_fd_long_range"
        / "physical_fd_long_range_summary.json"
    )
    if long_range_path.is_file():
        payload = json.loads(long_range_path.read_text())
        parts = []
        for item in payload["by_temperature"]:
            parts.append(
                f"{item['temperature_K']} K Γ/K 组留出 MAE "
                f"{item['regions']['G']['group_CV_MAE_cm-1']:.1f}/"
                f"{item['regions']['K']['group_CV_MAE_cm-1']:.1f} cm^-1"
            )
        lines.append(
            "physical-FD q-space 长程校正："
            + "；".join(parts)
            + ("；通过固定门槛。" if payload["passes_all_long_range_gates"] else "；未通过固定门槛。")
        )
        fingerprint["physical_fd_long_range"] = payload

    fd0_path = (
        ROOT
        / "results"
        / "graphene_physical_fd0"
        / "graphene_fd0_extrapolation.json"
    )
    if fd0_path.is_file():
        payload = json.loads(fd0_path.read_text())
        lines.append(
            "degauss→0 参考：外推模型最大差 "
            f"`{payload['max_extrapolation_model_spread_cm-1']:.1f} cm^-1`，"
            "同 smearing 的 k=120→144 最大变化 "
            f"`{payload['max_same_smearing_k120_to_k144_abs_cm-1']:.1f} cm^-1`。"
        )
        fingerprint["physical_fd0_extrapolation"] = payload

    td = ROOT / "results" / "td_phonon"
    v11_control = td / "td_graphene_v11_no_long_range_physical_geom.csv"
    if v11_control.is_file():
        with v11_control.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        values = ", ".join(
            f"{int(float(row['T_K']))} K: K={float(row['w_k_cm']):.1f} cm^-1, "
            f"kink={float(row['kink_k']):.2f}"
            for row in rows
        )
        lines.append(f"v11 同底座、不加长程项的 TDEP 对照：{values}。")
        fingerprint["v11_no_long_range_control"] = rows

    metric_path = td / "graphene_lattice_temperature_summary.json"
    if metric_path.is_file():
        metrics = json.loads(metric_path.read_text())
        temperatures = metrics.get("dft_lattice_temperatures_K", [])
        lines.append(
            "300 K DFT-MD/TDEP：与匹配的 0 K harmonic DFT 参考谱的全谱 MAE "
            f"`{metrics['harmonic_0K_vs_DFTMD_300K_full_band_MAE_cm-1']:.1f} cm^-1`；"
            "plain fine-tuned MLIP 方案为 "
            f"`{metrics['MLIP_300K_vs_DFTMD_300K_full_band_MAE_cm-1']:.1f} cm^-1`，"
            "v11 backbone + 长程方案为 "
            f"`{metrics['MLIP_long_range_300K_vs_DFTMD_300K_full_band_MAE_cm-1']:.1f} cm^-1`；"
            "两者 backbone 不同，不作为单变量 ablation。"
        )
        if 600 in temperatures:
            lines.append(
                "600 K DFT-MD/TDEP 已同步并完成收敛检查；300→600 K 全谱 MAE "
                f"`{metrics['DFTMD_300K_vs_600K_full_band_MAE_cm-1']:.1f} cm^-1`。"
            )
        fingerprint["lattice_metrics"] = metrics

    fullpath = (
        ROOT
        / "results"
        / "p1_graphene_fullpath_pilot"
        / "graphene_fullpath_pilot_dfpt.csv"
    )
    if fullpath.is_file():
        with fullpath.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        lines.append(
            f"完整路径 pilot：`{len(rows)}/8` 个 direct-DFPT q points 已同步。"
        )
        fingerprint["fullpath_rows"] = len(rows)

    cold = ROOT / "results" / "tmd_exp_a_recovery" / "tmd_cold_fd_pairs.json"
    if cold.is_file():
        payload = json.loads(cold.read_text())
        deltas = [row["cold_minus_fd_min_cm-1"] for row in payload["rows"]]
        lines.append(
            f"TMD cold/fd：`{payload['n_pairs']}/5` 个完全同设置配对通过检查；"
            f"最低频率 cold−fd 范围为 `{min(deltas):.1f}` 到 `{max(deltas):.1f} cm^-1`。"
        )
        fingerprint["cold_pairs"] = payload

    deployment = ROOT / "results" / "graphene_kohn_fd" / "deploy_15pt.npz"
    weighted_calibrated = (
        ROOT
        / "results"
        / "graphene_fd_delta_weighted"
        / "T600_TDEP"
        / "calibrated_acceptance.json"
    )
    if weighted_calibrated.is_file():
        payload = json.loads(weighted_calibrated.read_text())
        max_seed_spread = max(
            row["full_band_MAE_cm-1"]
            for row in payload["thermal_seed_pair_spread"]
        )
        g_mae = payload["regions"]["G"][
            "DFT_TDEP_thermal_calibrated_group_CV_MAE_cm-1"
        ]
        k_mae = payload["regions"]["K"][
            "DFT_TDEP_thermal_calibrated_group_CV_MAE_cm-1"
        ]
        kink_error = payload["regions"]["K"][
            "thermal_calibrated_group_CV_kink_relative_error"
        ]
        passed = payload["passes_all_force_seed_calibrated_qspace_gates"]
        lines.append(
            "600 K weighted replay + 有限温 q-space：三 seed 全谱差异最大 "
            f"`{max_seed_spread:.2f} cm^-1`，Γ/K line 留组 MAE "
            f"`{g_mae:.4f}/{k_mae:.4f} cm^-1`，K kink 相对误差 "
            f"`{100.0 * kink_error:.2f}%`；"
            + ("通过固定门槛。" if passed else "未通过固定门槛。")
        )
        fingerprint["weighted_600K_calibrated_qspace"] = {
            "sha256_inputs": payload["inputs"],
            "max_seed_spread_cm-1": max_seed_spread,
            "G_group_CV_MAE_cm-1": g_mae,
            "K_group_CV_MAE_cm-1": k_mae,
            "K_kink_relative_error": kink_error,
            "passed": passed,
        }
    if deployment.is_file():
        with np.load(deployment, allow_pickle=False) as data:
            count = len(data["dgs"])
        lines.append(f"graphene MLIP + 长程项：`{count}/15` 个 smearing points 已完成。")
        fingerprint["deploy_points"] = count
    return lines, fingerprint


def render_block(
    status_rows: dict[str, str], result_lines: list[str], changed_at: str
) -> str:
    rows = [
        f"| {task} | {status.replace('|', '/')} |" for task, status in status_rows.items()
    ]
    results = (
        "\n".join(f"- {line}" for line in result_lines)
        if result_lines
        else "- 尚无新的完成结果需要写回。"
    )
    return "\n".join(
        [
            START_MARKER,
            "## 实验监控",
            "",
            f"**最近状态变化：**{changed_at}",
            "",
            "| 任务 | 状态 |",
            "|---|---|",
            *rows,
            "",
            "**已经同步并复核的结果：**",
            "",
            results,
            "",
            "监控脚本只改动本区块；任务完成后会同步结果、重跑分析和图片。",
            END_MARKER,
        ]
    )


def update_report(block: str) -> Path:
    report = latest_weekly()
    text = report.read_text()
    if START_MARKER in text and END_MARKER in text:
        pattern = re.compile(
            re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL
        )
        updated = pattern.sub(block, text, count=1)
    else:
        divider = "\n---\n"
        position = text.find(divider)
        if position < 0:
            updated = text.rstrip() + "\n\n" + block + "\n"
        else:
            insertion = position + len(divider)
            updated = text[:insertion] + "\n" + block + "\n\n---\n" + text[insertion:]
    if updated != text:
        report.write_text(updated)
        log(f"updated {report.relative_to(ROOT)}")
    return report


def load_state() -> dict:
    if not STATE_PATH.is_file():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(payload: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(STATE_PATH)


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another monitor pass is still running")
            return 0

        previous = load_state()
        restart_counts = dict(previous.get("restart_counts", {}))
        statuses = {
            "v100_a": probe_v100_a(),
            "v100_b": probe_v100_b(),
            "rtx_2060": probe_2060(),
        }

        a, b, gpu = statuses["v100_a"], statuses["v100_b"], statuses["rtx_2060"]
        if a.get("reachable") == "1":
            if a.get("fd_done") != "1" and a.get("fd_service") != "active":
                maybe_restart(
                    "v100_a_physical_fd",
                    V100_A,
                    "systemctl restart phonon-graphene-physical-fd@FD300_CONV.service",
                    restart_counts,
                )
            if a.get("force_done") != "1" and a.get("force_service") != "active":
                maybe_restart(
                    "v100_a_physical_fd_force",
                    V100_A,
                    "systemctl restart phonon-graphene-fd-force@A.service",
                    restart_counts,
                )
            if a.get("k144_done") != "1" and a.get("k144_service") != "active":
                maybe_restart(
                    "v100_a_fd300_k144",
                    V100_A,
                    "systemctl reset-failed phonon-graphene-physical-fd@FD300_K144.service; "
                    "systemctl restart phonon-graphene-physical-fd@FD300_K144.service",
                    restart_counts,
                )
            if (
                a.get("k144_done") == "1"
                and a.get("line_done") != "1"
                and a.get("line_service") != "active"
            ):
                maybe_restart(
                    "v100_a_fd300_line",
                    V100_A,
                    "systemctl reset-failed phonon-graphene-physical-fd@FD300_LINE.service; "
                    "systemctl restart phonon-graphene-physical-fd@FD300_LINE.service",
                    restart_counts,
                )
            if a.get("thermal_done") != "1" and a.get("thermal_service") != "active":
                maybe_restart(
                    "v100_a_fd_thermal_labels",
                    V100_A,
                    "systemctl reset-failed phonon-graphene-fd-labels@A.service; "
                    "systemctl restart phonon-graphene-fd-labels@A.service",
                    restart_counts,
                )
            if a.get("thermal2_done") != "1" and a.get("thermal2_service") != "active":
                maybe_restart(
                    "v100_a_fd_thermal_labels_wave2",
                    V100_A,
                    "systemctl reset-failed phonon-graphene-fd-labels-wave2@A.service; "
                    "systemctl restart phonon-graphene-fd-labels-wave2@A.service",
                    restart_counts,
                )
            if (
                a.get("thermal3_done") != "1"
                and a.get("thermal3_skipped") != "1"
                and a.get("thermal3_service") != "active"
            ):
                maybe_restart(
                    "v100_a_fd_thermal_labels_wave3",
                    V100_A,
                    "systemctl reset-failed phonon-graphene-fd-labels-wave3@A.service; "
                    "systemctl restart phonon-graphene-fd-labels-wave3@A.service",
                    restart_counts,
                )
        if b.get("reachable") == "1":
            if b.get("fd_done") != "1" and b.get("fd_service") != "active":
                maybe_restart(
                    "v100_b_physical_fd",
                    V100_B,
                    "systemctl restart phonon-graphene-physical-fd@FD600_CONV.service",
                    restart_counts,
                )
            if b.get("force_done") != "1" and b.get("force_service") != "active":
                maybe_restart(
                    "v100_b_physical_fd_force",
                    V100_B,
                    "systemctl restart phonon-graphene-fd-force@B.service",
                    restart_counts,
                )
            if b.get("line_done") != "1" and b.get("line_service") != "active":
                maybe_restart(
                    "v100_b_fd600_line",
                    V100_B,
                    "systemctl reset-failed phonon-graphene-physical-fd@FD600_LINE.service; "
                    "systemctl restart phonon-graphene-physical-fd@FD600_LINE.service",
                    restart_counts,
                )
            if (
                b.get("line_done") == "1"
                and b.get("fd0_done") != "1"
                and b.get("fd0_service") != "active"
            ):
                maybe_restart(
                    "v100_b_fd0_scan",
                    V100_B,
                    "systemctl reset-failed phonon-graphene-physical-fd@FD0_SCAN.service; "
                    "systemctl restart phonon-graphene-physical-fd@FD0_SCAN.service",
                    restart_counts,
                )
            if b.get("thermal_done") != "1" and b.get("thermal_service") != "active":
                maybe_restart(
                    "v100_b_fd_thermal_labels",
                    V100_B,
                    "systemctl reset-failed phonon-graphene-fd-labels@B.service; "
                    "systemctl restart phonon-graphene-fd-labels@B.service",
                    restart_counts,
                )
            if b.get("thermal2_done") != "1" and b.get("thermal2_service") != "active":
                maybe_restart(
                    "v100_b_fd_thermal_labels_wave2",
                    V100_B,
                    "systemctl reset-failed phonon-graphene-fd-labels-wave2@B.service; "
                    "systemctl restart phonon-graphene-fd-labels-wave2@B.service",
                    restart_counts,
                )
            if (
                b.get("thermal3_done") != "1"
                and b.get("thermal3_skipped") != "1"
                and b.get("thermal3_service") != "active"
            ):
                maybe_restart(
                    "v100_b_fd_thermal_labels_wave3",
                    V100_B,
                    "systemctl reset-failed phonon-graphene-fd-labels-wave3@B.service; "
                    "systemctl restart phonon-graphene-fd-labels-wave3@B.service",
                    restart_counts,
                )
            if (
                gpu.get("reachable") == "1"
                and gpu.get("v2_started") != "1"
                and gpu.get("ablation_600_done") != "1"
                and b.get("ablation_service") != "active"
            ):
                maybe_restart(
                    "v100_b_fd_model_ablation_600",
                    V100_B,
                    "systemctl reset-failed phonon-graphene-fd-model-ablation-600.service; "
                    "systemctl restart phonon-graphene-fd-model-ablation-600.service",
                    restart_counts,
                )
            if (
                gpu.get("reachable") == "1"
                and gpu.get("v2_started") != "1"
                and gpu.get("ablation_selection") == "1"
                and gpu.get("ablation_tdep600_done") != "1"
                and gpu.get("ablation_tdep600_failed") != "1"
                and b.get("ablation_tdep_service") != "active"
            ):
                maybe_restart(
                    "v100_b_fd_model_ablation_tdep600",
                    V100_B,
                    "systemctl reset-failed phonon-graphene-fd-model-ablation-tdep600.service; "
                    "systemctl restart phonon-graphene-fd-model-ablation-tdep600.service",
                    restart_counts,
                )
            if (
                gpu.get("reachable") == "1"
                and gpu.get("v2_started") == "1"
                and b.get("v2_done") != "1"
                and b.get("v2_service") != "active"
            ):
                maybe_restart(
                    "v100_b_fd_model_ablation_v2_600",
                    V100_B,
                    "systemctl reset-failed phonon-graphene-fd-model-ablation-v2-600.service; "
                    "systemctl restart phonon-graphene-fd-model-ablation-v2-600.service",
                    restart_counts,
                )
        if b.get("reachable") == "1" and b.get("done") != "1":
            if b.get("service") != "active" and b.get("watchdog") != "active":
                maybe_restart(
                    "v100_b_600K",
                    V100_B,
                    "systemctl start phonon-recovery-watch@B.timer phonon-recovery-run@B.service",
                    restart_counts,
                )
        if a.get("reachable") == "1":
            if a.get("fullpath_done") != "1" and a.get("fullpath_service") != "active":
                maybe_restart(
                    "v100_a_fullpath",
                    V100_A,
                    "systemctl reset-failed phonon-graphene-fullpath-pilot-v2.service; "
                    "systemctl restart phonon-graphene-fullpath-pilot-v2.service",
                    restart_counts,
                )
            if (
                a.get("cold_done") != "1"
                and a.get("cold_service") != "active"
                and a.get("cold_verifier") != "active"
            ):
                maybe_restart(
                    "v100_a_cold_pairs",
                    V100_A,
                    "systemctl reset-failed phonon-tmd-cold-pairs-verifier.service; "
                    "systemctl restart phonon-tmd-cold-pairs-verifier.service",
                    restart_counts,
                )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("file_ready") != "1"
            and gpu.get("service") != "active"
        ):
            maybe_restart(
                "rtx_2060_deploy_15pt",
                RTX_2060,
                "systemctl --user reset-failed phonon-deploy-15pt-v3.service; "
                "systemctl --user restart phonon-deploy-15pt-v3.service",
                restart_counts,
                maximum=2,
            )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("control_done") != "1"
            and gpu.get("control_service") != "active"
        ):
            maybe_restart(
                "rtx_2060_v11_no_long_range_control",
                RTX_2060,
                "systemctl --user reset-failed phonon-graphene-v11-control.service; "
                "systemctl --user restart phonon-graphene-v11-control.service",
                restart_counts,
            )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("ablation_300_done") != "1"
            and gpu.get("ablation_300_service") != "active"
        ):
            maybe_restart(
                "rtx_2060_fd_model_ablation_300",
                RTX_2060,
                "systemctl --user reset-failed phonon-graphene-fd-model-ablation-300.service; "
                "systemctl --user restart phonon-graphene-fd-model-ablation-300.service",
                restart_counts,
            )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("v2_started") != "1"
            and gpu.get("ablation_done") != "1"
            and gpu.get("ablation_service") != "active"
        ):
            maybe_restart(
                "rtx_2060_fd_model_ablation_aggregate",
                RTX_2060,
                "systemctl --user reset-failed phonon-graphene-fd-model-ablation-aggregate.service; "
                "systemctl --user restart phonon-graphene-fd-model-ablation-aggregate.service",
                restart_counts,
            )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("v2_started") == "1"
            and gpu.get("v2_300_done") != "1"
            and gpu.get("v2_300_service") != "active"
        ):
            maybe_restart(
                "rtx_2060_fd_model_ablation_v2_300",
                RTX_2060,
                "systemctl --user reset-failed phonon-graphene-fd-model-ablation-v2-300.service; "
                "systemctl --user restart phonon-graphene-fd-model-ablation-v2-300.service",
                restart_counts,
            )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("v2_started") == "1"
            and gpu.get("v2_done") != "1"
            and gpu.get("v2_aggregate_service") != "active"
        ):
            maybe_restart(
                "rtx_2060_fd_model_ablation_v2_aggregate",
                RTX_2060,
                "systemctl --user reset-failed phonon-graphene-fd-model-ablation-v2-aggregate.service; "
                "systemctl --user restart phonon-graphene-fd-model-ablation-v2-aggregate.service",
                restart_counts,
            )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("fd_train_done") != "1"
            and gpu.get("fd_train_service") != "active"
        ):
            maybe_restart(
                "rtx_2060_fd_thermal_finetune",
                RTX_2060,
                "systemctl --user reset-failed phonon-graphene-fd-thermal-finetune.service; "
                "systemctl --user restart phonon-graphene-fd-thermal-finetune.service",
                restart_counts,
            )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("wave2_done") != "1"
            and gpu.get("wave2_service") != "active"
        ):
            maybe_restart(
                "rtx_2060_fd_thermal_wave2",
                RTX_2060,
                "systemctl --user reset-failed phonon-graphene-fd-thermal-refine-wave2.service; "
                "systemctl --user restart phonon-graphene-fd-thermal-refine-wave2.service",
                restart_counts,
            )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("wave3_done") != "1"
            and gpu.get("wave3_skipped") != "1"
            and gpu.get("wave3_failed") != "1"
            and gpu.get("wave3_service") != "active"
        ):
            maybe_restart(
                "rtx_2060_fd_thermal_wave3",
                RTX_2060,
                "systemctl --user reset-failed phonon-graphene-fd-thermal-refine-wave3.service; "
                "systemctl --user restart phonon-graphene-fd-thermal-refine-wave3.service",
                restart_counts,
            )
        if (
            gpu.get("reachable") == "1"
            and gpu.get("post_done") != "1"
            and gpu.get("long_blocked") != "1"
            and gpu.get("post_service") != "active"
        ):
            maybe_restart(
                "rtx_2060_physical_fd_postprocess",
                RTX_2060,
                "systemctl --user reset-failed phonon-graphene-physical-fd-postprocess.service; "
                "systemctl --user restart phonon-graphene-physical-fd-postprocess.service",
                restart_counts,
            )

        sync_jobs = []
        if a.get("fd_done") == "1":
            sync_jobs.append(
                (
                    "V100-A physical FD direct-DFPT",
                    lambda: sync_physical_fd(V100_A, "FD300_CONV"),
                )
            )
        if b.get("fd_done") == "1":
            sync_jobs.append(
                (
                    "V100-B physical FD direct-DFPT",
                    lambda: sync_physical_fd(V100_B, "FD600_CONV"),
                )
            )
        if a.get("force_done") == "1":
            sync_jobs.append(
                (
                    "V100-A physical FD force convergence",
                    lambda: sync_force_convergence(V100_A, "A"),
                )
            )
        if b.get("force_done") == "1":
            sync_jobs.append(
                (
                    "V100-B physical FD force convergence",
                    lambda: sync_force_convergence(V100_B, "B"),
                )
            )
        if a.get("k144_done") == "1":
            sync_jobs.append(
                (
                    "V100-A 300 K k=144",
                    lambda: sync_physical_fd(V100_A, "FD300_K144"),
                )
            )
        if a.get("line_done") == "1":
            sync_jobs.append(
                (
                    "V100-A 300 K dense line",
                    lambda: sync_physical_fd(V100_A, "FD300_LINE"),
                )
            )
        if b.get("line_done") == "1":
            sync_jobs.append(
                (
                    "V100-B 600 K dense line",
                    lambda: sync_physical_fd(V100_B, "FD600_LINE"),
                )
            )
        if b.get("fd0_done") == "1":
            sync_jobs.append(
                (
                    "V100-B degauss-to-zero scan",
                    lambda: sync_physical_fd(V100_B, "FD0_SCAN"),
                )
            )
        if a.get("thermal_done") == "1":
            sync_jobs.append(
                (
                    "V100-A physical-FD thermal labels",
                    lambda: sync_thermal_labels(V100_A, "A"),
                )
            )
        if b.get("thermal_done") == "1":
            sync_jobs.append(
                (
                    "V100-B physical-FD thermal labels",
                    lambda: sync_thermal_labels(V100_B, "B"),
                )
            )
        if a.get("thermal2_done") == "1":
            sync_jobs.append(
                (
                    "V100-A physical-FD thermal labels wave2",
                    lambda: sync_thermal_label_wave(V100_A, "A", 2),
                )
            )
        if b.get("thermal2_done") == "1":
            sync_jobs.append(
                (
                    "V100-B physical-FD thermal labels wave2",
                    lambda: sync_thermal_label_wave(V100_B, "B", 2),
                )
            )
        if a.get("thermal3_done") == "1":
            sync_jobs.append(
                (
                    "V100-A physical-FD thermal labels wave3",
                    lambda: sync_thermal_label_wave(V100_A, "A", 3),
                )
            )
        if b.get("thermal3_done") == "1":
            sync_jobs.append(
                (
                    "V100-B physical-FD thermal labels wave3",
                    lambda: sync_thermal_label_wave(V100_B, "B", 3),
                )
            )
        if b.get("done") == "1":
            sync_jobs.append(("V100-B 600 K", sync_v100_b))
        if a.get("fullpath_done") == "1":
            sync_jobs.append(("V100-A fullpath", sync_fullpath))
        if a.get("cold_done") == "1":
            sync_jobs.append(("V100-A cold/fd", sync_cold_pairs))
        if gpu.get("file_ready") == "1" and gpu.get("service") != "active":
            sync_jobs.append(("RTX 2060 15-point deployment", sync_15point_deployment))
        if gpu.get("control_done") == "1":
            sync_jobs.append(("RTX 2060 v11 no-long-range control", sync_2060_v11_control))
        if gpu.get("fd_train_done") == "1":
            sync_jobs.append(("RTX 2060 thermal fine-tune", sync_2060_thermal_finetune))
        if gpu.get("wave2_done") == "1":
            sync_jobs.append(("RTX 2060 thermal refine wave2", lambda: sync_2060_refine(2)))
        if gpu.get("wave3_done") == "1":
            sync_jobs.append(("RTX 2060 thermal refine wave3", lambda: sync_2060_refine(3)))
        if gpu.get("wave3_failed") == "1":
            sync_jobs.append(
                ("RTX 2060 thermal refine wave3 failure", sync_2060_refine_failure)
            )
        if gpu.get("ablation_done") == "1":
            sync_jobs.append(
                ("RTX 2060 fixed-data model ablation", sync_2060_model_ablation)
            )
        if gpu.get("v2_done") == "1":
            sync_jobs.append(
                ("RTX 2060 conservative model ablation", sync_2060_model_ablation_v2)
            )
        if gpu.get("post_done") == "1":
            sync_jobs.append(("RTX 2060 final physical-FD postprocess", sync_2060_final_postprocess))
        for name, function in sync_jobs:
            try:
                function()
            except Exception as exc:  # keep the other lanes monitored
                log(f"{name}: completion handling failed: {exc}")
        try:
            stage_thermal_finetune_if_ready()
        except Exception as exc:
            log(f"thermal fine-tune staging failed: {exc}")
        try:
            analyze_physical_fd_if_ready()
        except Exception as exc:
            log(f"physical FD convergence analysis failed: {exc}")

        rows = status_text(statuses)
        result_lines, result_fingerprint = local_result_lines()
        semantic = {"status": rows, "results": result_fingerprint}
        changed = semantic != previous.get("semantic")
        report = latest_weekly()
        if changed or START_MARKER not in report.read_text():
            changed_at = now_text()
            update_report(render_block(rows, result_lines, changed_at))
        else:
            changed_at = previous.get("changed_at", now_text())

        save_state(
            {
                "last_check": now_text(),
                "changed_at": changed_at,
                "semantic": semantic,
                "raw_status": statuses,
                "restart_counts": restart_counts,
            }
        )
        log("monitor pass complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
