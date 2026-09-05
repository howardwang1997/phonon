#!/usr/bin/env bash
# Gated X0 first-cross SSCHA test with one fixed-seed MD diagnostic on gate failure.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
S0="$ROOT/results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short"
Q0="$S0/Q0_quantum_sscha"
X0="$S0/X0_cross_development"
FREEZE="$S0/Q0_X0_freeze_manifest.json"
FORMAL="$S0/formal_240ep_2060_seed83_replayw16"
BASE="$ROOT/results/gr_backbone_v11/ft_graphene.model"
DELTA="$FORMAL/selected_checkpoint.model"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
OPERATOR="$ROOT/data/graphene_physical_s0/operators/T300_operator.npz"
LOG="$X0/run.log"

cd "$ROOT"
mkdir -p "$X0"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$X0/DONE" ]]; then
    echo "X0 already reached a terminal state"
    exit 0
fi
if [[ ! -e "$Q0/Q0_ALL_TEMPERATURES_PASSED" ]]; then
    echo "refusing X0: Q0 has not passed all fixed convergence gates"
    exit 2
fi
for path in "$FREEZE" "$BASE" "$DELTA" "$BG" "$OPERATOR" \
    scripts/smearing_kink/run_graphene_physical_q0_sscha.py \
    scripts/smearing_kink/analyze_graphene_physical_x0.py \
    scripts/smearing_kink/td_phonon_friedel.py \
    scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py; do
    [[ -s "$path" ]]
done

date -Is > "$X0/STARTED_AT"
touch "$X0/RUNNING"
START_EPOCH="$(date +%s)"
finish() {
    local exit_code=$?
    trap - EXIT
    rm -f "$X0/RUNNING"
    echo "$exit_code" > "$X0/EXIT_CODE"
    if (( exit_code != 0 )); then
        touch "$X0/FAILED_INFRASTRUCTURE"
    fi
    exit "$exit_code"
}
trap finish EXIT

SSCHA_OUT="$X0/sscha_Tlat450_Tel300"
mkdir -p "$SSCHA_OUT" "$X0/analysis"
if [[ ! -e "$SSCHA_OUT/PASSED" ]]; then
    printf '%s\n' "SSCHA Tlat=450 Tel=300" > "$X0/RUNNING_CASE"
    echo "=== X0 first non-diagonal SSCHA START $(date -Is) ==="
    if "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/run_graphene_physical_q0_sscha.py \
        --freeze-manifest "$FREEZE" --phase x0_first \
        --lattice-temperature 450 --operator-temperature 300 \
        --n-configs 300 --max-populations 5 --random-seed 453005 \
        --nsegments 60 --device cuda --output-dir "$SSCHA_OUT"; then
        touch "$SSCHA_OUT/PASSED"
    else
        touch "$X0/BLOCKED_X0_SSCHA_CONVERGENCE" "$X0/DONE"
        rm -f "$X0/RUNNING_CASE"
        echo "X0 first non-diagonal SSCHA did not converge; stopping before MD expansion"
        exit 0
    fi
fi

printf '%s\n' "analyze SSCHA cross term" > "$X0/RUNNING_CASE"
if "$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/analyze_graphene_physical_x0.py \
    --freeze-manifest "$FREEZE" --mode sscha \
    --actual-result "$SSCHA_OUT/result.npz" --output-dir "$X0/analysis"; then
    END_EPOCH="$(date +%s)"
    "$CONDA" run -n phonon python -c \
        'import json,pathlib,sys; pathlib.Path(sys.argv[3]).write_text(json.dumps({"status":"complete","wall_time_seconds":int(sys.argv[2])-int(sys.argv[1]),"branch":"formula_reconstruction_passed"},indent=2)+"\n")' \
        "$START_EPOCH" "$END_EPOCH" "$X0/runtime.json"
    touch "$X0/X0_FIRST_CROSS_PASSED" "$X0/X0_DEVELOPMENT_PASSED" \
        "$S0/READY_FOR_VALIDATION_FREEZE_REVIEW" "$X0/DONE"
    rm -f "$X0/RUNNING_CASE"
    date -Is > "$X0/COMPLETED_AT"
    echo "=== X0 PASSED; 3x3 development grid reconstructed by formula $(date -Is) ==="
    exit 0
fi

touch "$X0/X0_SSCHA_CROSS_EXCEEDS_GATE"
FALLBACK="$X0/fallback_md"
CHECKPOINT_ROOT="$FALLBACK/checkpoints/seed0_dt0p5"
CHECKPOINT="$CHECKPOINT_ROOT/T450"
SHORT_TDEP="$FALLBACK/coupled_short_seed0.npz"
mkdir -p "$FALLBACK"
printf '%s\n' "fallback classical MD Tlat=450 Tel=300 seed=0" > "$X0/RUNNING_CASE"
echo "=== X0 fixed-seed MD diagnostic START $(date -Is) ==="
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/td_phonon_friedel.py \
    --model "$BASE" --delta-model "$DELTA" --bg "$BG" \
    --operator "$OPERATOR" --friedel-tel 300 --device cuda \
    --tag graphene_physical_X0_Tlat450_Tel300_seed0 \
    --temperatures 450 --dt 0.5 --equil 3000 --nsnap 3000 --stride 80 \
    --npoints 201 --seed 0 --checkpoint-root "$CHECKPOINT_ROOT" \
    --checkpoint-every 25 --max-temperature-factor 5 \
    --min-pair-distance 0.8 --max-force 100

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
    --background "$BG" --checkpoint "$CHECKPOINT" \
    --subtract-operator "$OPERATOR" --temperature 450 \
    --tag graphene_physical_X0_Tlat450_Tel300_seed0_short \
    --output "$SHORT_TDEP"

set +e
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/analyze_graphene_physical_x0.py \
    --freeze-manifest "$FREEZE" --mode md \
    --coupled-short-tdep "$SHORT_TDEP" --output-dir "$X0/analysis"
MD_GATE_EXIT=$?
set -e
echo "$MD_GATE_EXIT" > "$FALLBACK/DIAGNOSTIC_GATE_EXIT_CODE"
touch "$FALLBACK/DIAGNOSTIC_COMPLETE" "$X0/BLOCKED_X0_CROSS_TERM" "$X0/DONE"
rm -f "$X0/RUNNING_CASE"
date -Is > "$X0/COMPLETED_AT"
echo "X0 cross term exceeded the fixed SSCHA gate; fixed-seed MD diagnostic is complete"
echo "automatic three-seed expansion and 375/525 K validation remain blocked"
