#!/usr/bin/env bash
# Gated, restartable Q0 benchmark and formal three-temperature SSCHA queue.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
S0="$ROOT/results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short"
L0="$S0/L0_classical_tdep"
Q0="$S0/Q0_quantum_sscha"
X0="$S0/X0_cross_development"
FREEZE="$S0/Q0_X0_freeze_manifest.json"
LOG="$Q0/run.log"

cd "$ROOT"
mkdir -p "$Q0" "$X0"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$Q0/DONE" ]]; then
    echo "Q0 already reached a terminal state"
    exit 0
fi
if [[ ! -e "$L0/L0_ALL_TEMPERATURES_PASSED" ]]; then
    echo "refusing Q0: L0 has not passed all fixed gates"
    exit 2
fi
for path in \
    scripts/smearing_kink/freeze_graphene_physical_q0_x0.py \
    scripts/smearing_kink/run_graphene_physical_q0_sscha.py \
    scripts/smearing_kink/analyze_graphene_physical_x0.py \
    scripts/smearing_kink/td_phonon_friedel.py \
    scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py; do
    [[ -s "$path" ]]
done

date -Is > "$Q0/STARTED_AT"
touch "$Q0/RUNNING"
START_EPOCH="$(date +%s)"
finish() {
    local exit_code=$?
    trap - EXIT
    rm -f "$Q0/RUNNING"
    echo "$exit_code" > "$Q0/EXIT_CODE"
    if (( exit_code != 0 )); then
        touch "$Q0/FAILED_INFRASTRUCTURE"
    fi
    exit "$exit_code"
}
trap finish EXIT

if [[ ! -s "$FREEZE" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/freeze_graphene_physical_q0_x0.py \
        --l0-root "$L0" --q0-root "$Q0" --x0-root "$X0" \
        --q0-script scripts/smearing_kink/run_graphene_physical_q0_sscha.py \
        --x0-analysis-script scripts/smearing_kink/analyze_graphene_physical_x0.py \
        --sampling-script scripts/smearing_kink/td_phonon_friedel.py \
        --recompute-script scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
        --output "$FREEZE"
fi

run_case() {
    local phase="$1" lattice_temperature="$2" operator_temperature="$3"
    local nconfigs="$4" maxpop="$5" random_seed="$6" outdir="$7"
    mkdir -p "$outdir"
    if [[ -e "$outdir/PASSED" ]]; then
        return 0
    fi
    printf '%s\n' "$phase Tlat=$lattice_temperature Tel=$operator_temperature" > "$Q0/RUNNING_CASE"
    echo "=== Q0 phase=$phase Tlat=$lattice_temperature Tel=$operator_temperature START $(date -Is) ==="
    if "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/run_graphene_physical_q0_sscha.py \
        --freeze-manifest "$FREEZE" --phase "$phase" \
        --lattice-temperature "$lattice_temperature" \
        --operator-temperature "$operator_temperature" \
        --n-configs "$nconfigs" --max-populations "$maxpop" \
        --random-seed "$random_seed" --nsegments 60 --device cuda \
        --output-dir "$outdir"; then
        touch "$outdir/PASSED"
        echo "=== Q0 phase=$phase Tlat=$lattice_temperature PASSED $(date -Is) ==="
        return 0
    fi
    touch "$outdir/NOT_CONVERGED"
    return 1
}

if [[ ! -e "$Q0/BENCHMARK_PASSED" ]]; then
    if ! run_case benchmark 450 450 200 4 450004 "$Q0/benchmark_T450"; then
        touch "$Q0/BLOCKED_Q0_BENCHMARK" "$Q0/DONE"
        rm -f "$Q0/RUNNING_CASE"
        echo "Q0 450 K benchmark did not converge; formal quantum queue remains blocked"
        exit 0
    fi
    touch "$Q0/BENCHMARK_PASSED"
fi

for temperature in 450 300 600; do
    seed="$((temperature * 1000 + 5))"
    if ! run_case formal "$temperature" "$temperature" 300 5 "$seed" "$Q0/formal_T${temperature}"; then
        touch "$Q0/BLOCKED_Q0_FORMAL" "$Q0/DONE"
        rm -f "$Q0/RUNNING_CASE"
        echo "Q0 formal T=$temperature K did not converge; later temperatures remain blocked"
        exit 0
    fi
done

END_EPOCH="$(date +%s)"
"$CONDA" run -n phonon python -c \
    'import json,pathlib,sys; pathlib.Path(sys.argv[3]).write_text(json.dumps({"status":"complete","wall_time_seconds":int(sys.argv[2])-int(sys.argv[1]),"temperatures_K":[300,450,600],"benchmark_nconfigs":200,"formal_nconfigs":300},indent=2)+"\n")' \
    "$START_EPOCH" "$END_EPOCH" "$Q0/runtime.json"
rm -f "$Q0/RUNNING_CASE"
touch "$Q0/Q0_ALL_TEMPERATURES_PASSED" "$Q0/DONE"
date -Is > "$Q0/COMPLETED_AT"
echo "=== Q0 ALL TEMPERATURES PASSED $(date -Is) ==="
