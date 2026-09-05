#!/usr/bin/env bash
# Coordinate frozen L0 lanes with the Mac's existing authenticated SSH access.
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/Users/howardwang/Desktop/playground/phonon}"
IDENTITY="${IDENTITY:-/Users/howardwang/.ssh/id_ed25519}"
RTX="${RTX:-howardwang@100.105.21.7}"
V100_A="${V100_A:-root@100.80.236.112}"
V100_B="${V100_B:-root@100.123.220.57}"
RTX_ROOT="/home/howardwang/phonon"
V100_ROOT="/data/graphene_physical_s0_l0_dist/project"
S0_REL="results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short"
OUT_REL="$S0_REL/L0_classical_tdep"
RTX_OUT="$RTX_ROOT/$OUT_REL"
V100_OUT="$V100_ROOT/$OUT_REL"
STATE="$LOCAL_ROOT/results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short/L0_classical_tdep/distributed_local_coordinator"
FREEZE_SHA256="14f4688f4f0eb542ae5b5f74c79584ad406928253acd2a319d5575d2e1389b3a"
SSH=(
    /usr/bin/ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -i "$IDENTITY"
)
SCP=(
    /usr/bin/scp -q -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10
    -i "$IDENTITY"
)

mkdir -p "$STATE/imports"
if [[ -e "$STATE/FAILED" ]]; then
    mv "$STATE/FAILED" "$STATE/FAILED_RECOVERED_$(/bin/date -u '+%Y%m%dT%H%M%SZ')"
fi
exec > >(tee -a "$STATE/coordinator.log") 2>&1
echo $$ > "$STATE/pid"
touch "$STATE/RUNNING"

finish() {
    local code=$?
    rm -f "$STATE/RUNNING" "$STATE/pid"
    echo "$code" > "$STATE/exit_code"
    if (( code != 0 )); then touch "$STATE/FAILED"; fi
}
trap finish EXIT

rtx() { "${SSH[@]}" "$RTX" "$1"; }
v100a() { "${SSH[@]}" "$V100_A" "$1"; }
v100b() { "${SSH[@]}" "$V100_B" "$1"; }
now() { /bin/date -u '+%Y-%m-%dT%H:%M:%SZ'; }

remote_gate_state() {
    local temperature="$1"
    rtx "if test -e '$RTX_OUT/T${temperature}/FINAL_GATE_PASSED'; then echo passed; elif test -e '$RTX_OUT/BLOCKED_FINAL_POINT_GATE' || test -e '$RTX_OUT/BLOCKED_FINAL_BOOTSTRAP_GATE' || test -e '$RTX_OUT/FAILED_INFRASTRUCTURE'; then echo blocked; else echo waiting; fi" \
        2>/dev/null || echo unreachable
}

ensure_remote_layout() {
    rtx "install -d -m 0755 '$RTX_ROOT/coordination'; test -x '$RTX_ROOT/scripts/v100/run_graphene_physical_l0_worker.sh'; test -x '$RTX_ROOT/scripts/smearing_kink/verify_graphene_physical_l0_worker.py'"
    v100a "install -d -m 0755 '$V100_ROOT/coordination'; test -x '$V100_ROOT/scripts/v100/run_graphene_physical_l0_worker.sh'; test -x '$V100_ROOT/scripts/smearing_kink/verify_graphene_physical_l0_worker.py'"
    v100b "install -d -m 0755 '$V100_ROOT/coordination'; test -x '$V100_ROOT/scripts/v100/run_graphene_physical_l0_worker.sh'; test -x '$V100_ROOT/scripts/smearing_kink/verify_graphene_physical_l0_worker.py'"
    echo "three-host coordination layout verified at $(now)"
}

wait_for_gate() {
    local temperature="$1" state attempt=0
    while true; do
        state="$(remote_gate_state "$temperature")"
        if (( attempt % 20 == 0 )); then
            echo "T=$temperature gate state=$state ($(now))"
        fi
        case "$state" in
            passed) echo "T=$temperature gate passed at $(now)"; return ;;
            blocked)
                echo "T=$temperature gate blocked; distributed queue will not advance"
                touch "$STATE/BLOCKED_T${temperature}"
                exit 0
                ;;
            *) sleep 15 ;;
        esac
        attempt=$((attempt + 1))
    done
}

stop_serial_l0() {
    rtx "touch '$RTX_OUT/DISTRIBUTED_TRANSITION'; systemctl --user stop graphene-physical-s0-autopilot.service || true; systemctl --user stop graphene-physical-s0-l0.service || true; while systemctl --user is-active --quiet graphene-physical-s0-l0.service; do sleep 1; done; rm -f '$RTX_OUT/FAILED_INFRASTRUCTURE' '$RTX_OUT/EXIT_CODE' '$RTX_OUT/RUNNING'"
    echo "RTX autopilot and serial L0 stopped for distributed lanes at $(now)"
}

make_gate_token() {
    local temperature="$1" previous_temperature="$2"
    local previous_hash token="$STATE/ALLOW_T${temperature}"
    previous_hash="$(rtx "sha256sum '$RTX_OUT/T${previous_temperature}/final_bootstrap_acceptance.json' | cut -d ' ' -f1")"
    [[ "$previous_hash" =~ ^[0-9a-f]{64}$ ]]
    printf '%s\n' \
        "freeze_sha256=$FREEZE_SHA256" \
        "temperature_K=$temperature" \
        "previous_temperature_K=$previous_temperature" \
        "previous_bootstrap_acceptance_sha256=$previous_hash" > "$token"
    "${SCP[@]}" "$token" "$RTX:$RTX_ROOT/coordination/ALLOW_T${temperature}"
    "${SCP[@]}" "$token" "$V100_A:$V100_ROOT/coordination/ALLOW_T${temperature}"
    "${SCP[@]}" "$token" "$V100_B:$V100_ROOT/coordination/ALLOW_T${temperature}"
    echo "published authenticated T=$temperature gate token at $(now)"
}

start_v100_worker() {
    local target="$1" environment="$2" temperature="$3" seed="$4" unit
    unit="graphene-physical-l0-t${temperature}-seed${seed}"
    local result="$V100_OUT/T${temperature}/worker_seed${seed}_result.json"
    if "${SSH[@]}" "$target" "test -s '$result'"; then return; fi
    local active
    active="$("${SSH[@]}" "$target" "systemctl is-active '$unit.service' 2>/dev/null || true")"
    if [[ "$active" == active || "$active" == activating ]]; then return; fi
    "${SSH[@]}" "$target" "systemctl reset-failed '$unit.service' 2>/dev/null || true"
    "${SSH[@]}" "$target" "systemd-run --unit='$unit' --collect --property=Type=exec --setenv=DIST_ROOT='$V100_ROOT' --setenv=PHONON_ENV='$environment' '$V100_ROOT/scripts/v100/run_graphene_physical_l0_worker.sh' '$temperature' '$seed' formal"
}

start_rtx_worker() {
    local temperature="$1" seed=2 unit="graphene-physical-l0-t${temperature}-seed2"
    local result="$RTX_OUT/T${temperature}/worker_seed2_result.json"
    if rtx "test -s '$result'"; then return; fi
    local active
    active="$(rtx "systemctl --user is-active '$unit.service' 2>/dev/null || true")"
    if [[ "$active" == active || "$active" == activating ]]; then return; fi
    rtx "systemctl --user reset-failed '$unit.service' 2>/dev/null || true"
    rtx "systemd-run --user --unit='$unit' --collect --property=Type=exec --setenv=DIST_ROOT='$RTX_ROOT' --setenv=CONDA=/home/howardwang/miniconda3/bin/conda --setenv=PHONON_ENV=phonon '$RTX_ROOT/scripts/v100/run_graphene_physical_l0_worker.sh' '$temperature' '$seed' formal"
}

worker_state() {
    local target="$1" scope="$2" result="$3" marker="$4" unit="$5"
    if "${SSH[@]}" "$target" "test -s '$result' && test -e '$marker'"; then
        echo complete
        return
    fi
    local command="systemctl is-active '$unit.service' 2>/dev/null || true"
    if [[ "$scope" == user ]]; then command="systemctl --user is-active '$unit.service' 2>/dev/null || true"; fi
    local active
    active="$("${SSH[@]}" "$target" "$command" 2>/dev/null || true)"
    case "$active" in
        active|activating) echo running ;;
        *) echo failed ;;
    esac
}

wait_workers() {
    local temperature="$1" a b r
    while true; do
        a="$(worker_state "$V100_A" system \
            "$V100_OUT/T${temperature}/worker_seed0_result.json" \
            "$V100_OUT/T${temperature}/WORKER_SEED0_FORMAL_DONE" \
            "graphene-physical-l0-t${temperature}-seed0")"
        b="$(worker_state "$V100_B" system \
            "$V100_OUT/T${temperature}/worker_seed1_result.json" \
            "$V100_OUT/T${temperature}/WORKER_SEED1_FORMAL_DONE" \
            "graphene-physical-l0-t${temperature}-seed1")"
        r="$(worker_state "$RTX" user \
            "$RTX_OUT/T${temperature}/worker_seed2_result.json" \
            "$RTX_OUT/T${temperature}/WORKER_SEED2_FORMAL_DONE" \
            "graphene-physical-l0-t${temperature}-seed2")"
        echo "T=$temperature workers: V100-A=$a V100-B=$b RTX=$r ($(now))"
        if [[ "$a" == failed || "$b" == failed || "$r" == failed ]]; then
            return 1
        fi
        if [[ "$a" == complete && "$b" == complete && "$r" == complete ]]; then
            return
        fi
        sleep 60
    done
}

safe_archive_listing() {
    local archive="$1" temperature="$2" seed="$3"
    local prefix="$OUT_REL/T${temperature}/"
    while IFS= read -r member; do
        case "$member" in "$prefix"*) ;; *) return 1 ;; esac
        [[ "$member" != /* && "$member" != *"../"* && "$member" != ".." ]]
    done < <(/usr/bin/tar -tzf "$archive")
    /usr/bin/tar -tzf "$archive" | /usr/bin/grep -Fxq \
        "${prefix}worker_seed${seed}_result.json"
}

import_worker() {
    local label="$1" target="$2" temperature="$3" seed="$4" dt_tag="$5"
    local archive="$STATE/imports/${label}_T${temperature}_seed${seed}.tar.gz"
    "${SSH[@]}" "$target" \
        "tar -C '$V100_ROOT' -czf - '$OUT_REL/T${temperature}/checkpoints/seed${seed}_${dt_tag}/T${temperature}' '$OUT_REL/T${temperature}/final_short_seed${seed}.npz' '$OUT_REL/T${temperature}/worker_seed${seed}_result.json'" \
        > "$archive.partial"
    mv "$archive.partial" "$archive"
    safe_archive_listing "$archive" "$temperature" "$seed"
    "${SSH[@]}" "$RTX" "tar -C '$RTX_ROOT' -xzf -" < "$archive"
    rtx "'/home/howardwang/miniconda3/bin/conda' run --no-capture-output -n phonon python '$RTX_ROOT/scripts/smearing_kink/verify_graphene_physical_l0_worker.py' --root '$RTX_ROOT' --temperature '$temperature' --seed '$seed' --checkpoint '$RTX_OUT/T${temperature}/checkpoints/seed${seed}_${dt_tag}/T${temperature}' --expected-count 3000 --short-tdep '$RTX_OUT/T${temperature}/final_short_seed${seed}.npz' --output '$RTX_OUT/distributed_coordination/${label}_T${temperature}_seed${seed}_verified.json'"
    echo "imported and verified $label T=$temperature seed=$seed at $(now)"
}

run_distributed_temperature() {
    local temperature="$1" previous_temperature="$2"
    make_gate_token "$temperature" "$previous_temperature"
    start_v100_worker "$V100_A" phonon-mlip "$temperature" 0
    start_v100_worker "$V100_B" phonon "$temperature" 1
    start_rtx_worker "$temperature"
    if ! wait_workers "$temperature"; then
        echo "T=$temperature worker failure detected; coordinator is stopping without automatic restart"
        touch "$STATE/FAILED_WORKER_T${temperature}"
        return 1
    fi
    import_worker v100a "$V100_A" "$temperature" 0 dt0p5
    import_worker v100b "$V100_B" "$temperature" 1 dt0p25
    rtx "rm -f '$RTX_OUT/FAILED_INFRASTRUCTURE' '$RTX_OUT/EXIT_CODE'; systemctl --user reset-failed graphene-physical-s0-l0.service || true; systemctl --user start graphene-physical-s0-l0.service"
    wait_for_gate "$temperature"
}

echo "=== local three-host L0 coordinator START $(now) ==="
ensure_remote_layout
wait_for_gate 450
stop_serial_l0
if [[ "$(remote_gate_state 300)" != passed ]]; then
    if ! run_distributed_temperature 300 450; then exit 0; fi
    stop_serial_l0
fi
if [[ "$(remote_gate_state 600)" != passed ]]; then
    if ! run_distributed_temperature 600 300; then exit 0; fi
fi
while ! rtx "test -e '$RTX_OUT/L0_ALL_TEMPERATURES_PASSED'"; do
    state="$(remote_gate_state 600)"
    if [[ "$state" == blocked ]]; then touch "$STATE/BLOCKED_T600"; exit 0; fi
    sleep 15
done
rtx "rm -f '$RTX_OUT/DISTRIBUTED_TRANSITION'; touch '$RTX_OUT/DISTRIBUTED_COMPLETE'; systemctl --user start graphene-physical-s0-autopilot.service"
rm -f "$STATE/FAILED"
touch "$STATE/COMPLETE"
echo "=== local three-host L0 coordinator COMPLETE $(now) ==="
