#!/usr/bin/env bash
# Run one already-prepared R2M routed-tail outer fold.  This remains inert
# until a passing support-free core path and exact SHA-256 are supplied.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
LAUNCHER_SOURCE="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon-mlip}"
FOLD_INDEX="${FOLD_INDEX:?set FOLD_INDEX=0..8}"
PREPARED_ROOT="${PREPARED_ROOT:?set PREPARED_ROOT to the frozen nine-fold data}"
CORE_MODEL="${CORE_MODEL:?set CORE_MODEL to selected_core.model}"
CORE_SHA256="${CORE_SHA256:?set CORE_SHA256 to the passing artifact hash}"
R2M_DATA="${R2M_DATA:-$ROOT/data/graphene_r2m_support_free_core}"
PRISTINE="${PRISTINE:-$ROOT/data/td_phonon/graphene.xyz}"
OPERATOR="${OPERATOR:?set OPERATOR to the frozen T300 q6 operator}"
BACKGROUND="${BACKGROUND:?set BACKGROUND to the frozen graphene phonopy background}"
THERMAL_RESULT="${THERMAL_RESULT:?set THERMAL_RESULT to the frozen corrected T450 result}"
OUT="${OUT:-/data/graphene_r2m_routed_tail/outer$(printf '%02d' "$FOLD_INDEX")_seed83}"
TRAIN_OUT="$OUT/train"
EVALUATION="$OUT/outer_fold_evaluation.json"
PREPARED_MANIFEST="$PREPARED_ROOT/manifest.json"
CODE_DIR="$OUT/code_snapshots"
LAUNCHER_FREEZE="$OUT/launcher_freeze.json"
COMPLETION_MANIFEST="$OUT/completion_manifest.json"

if (( FOLD_INDEX < 0 || FOLD_INDEX > 8 )); then
    echo "FOLD_INDEX must be in 0..8" >&2
    exit 2
fi
case "$CORE_SHA256" in
    (*[!0-9a-f]*|'') echo "CORE_SHA256 must be lowercase hexadecimal" >&2; exit 2 ;;
esac
if (( ${#CORE_SHA256} != 64 )); then
    echo "CORE_SHA256 must have 64 characters" >&2
    exit 2
fi
for path in "$PREPARED_ROOT" "$OUT" "$CORE_MODEL" "$R2M_DATA" \
    "$PRISTINE" "$OPERATOR" "$BACKGROUND" "$THERMAL_RESULT"; do
    case "${path,,}" in
        (*seed2*|*reserved_e50*|*reserved-e50*)
            echo "seed2 paths are forbidden in routed-tail outer LOCO: $path" >&2
            exit 2
            ;;
    esac
done

shopt -s nullglob
FOLD_MATCHES=("$PREPARED_ROOT"/outer"$(printf '%02d' "$FOLD_INDEX")"_sscha*)
shopt -u nullglob
if (( ${#FOLD_MATCHES[@]} != 1 )); then
    echo "expected exactly one prepared directory for outer fold $FOLD_INDEX" >&2
    exit 2
fi
FOLD_DIR="${FOLD_MATCHES[0]}"
for path in \
    "$PREPARED_MANIFEST" \
    "$FOLD_DIR/fold_manifest.json" \
    "$FOLD_DIR/support_train8.xyz" \
    "$FOLD_DIR/support_held1.xyz" \
    "$R2M_DATA/train.xyz" \
    "$R2M_DATA/valid.xyz" \
    "$PRISTINE" \
    "$CORE_MODEL" \
    "$OPERATOR" \
    "$BACKGROUND" \
    "$THERMAL_RESULT"; do
    [[ -s "$path" ]]
done

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r2m-routed-tail}"

if [[ ! -e "$LAUNCHER_FREEZE" ]]; then
    if [[ -e "$CODE_DIR" ]]; then
        echo "refusing to overwrite an unfrozen code snapshot directory: $CODE_DIR" >&2
        exit 3
    fi
    mkdir "$CODE_DIR"
    cp "$ROOT/scripts/v100/run_graphene_r2m_routed_tail_outer_fold.sh" \
        "$CODE_DIR/run_graphene_r2m_routed_tail_outer_fold.sh"
    for script in \
        graphene_r2m_run_provenance.py \
        graphene_r2m_routed_tail.py \
        train_graphene_r2m_routed_tail_outer_fold.py \
        evaluate_graphene_r2m_routed_tail_outer_fold.py \
        graphene_r2m_aprime_eval.py; do
        cp "$ROOT/scripts/smearing_kink/$script" "$CODE_DIR/$script"
    done
    "$CONDA" run -n "$CONDA_ENV" python \
        "$CODE_DIR/graphene_r2m_run_provenance.py" freeze \
        --launcher-source "$LAUNCHER_SOURCE" \
        --prepared-manifest "$PREPARED_MANIFEST" \
        --fold-dir "$FOLD_DIR" \
        --core-model "$CORE_MODEL" \
        --core-sha256 "$CORE_SHA256" \
        --replay-train "$R2M_DATA/train.xyz" \
        --replay-valid "$R2M_DATA/valid.xyz" \
        --pristine "$PRISTINE" \
        --operator "$OPERATOR" \
        --background "$BACKGROUND" \
        --thermal-result "$THERMAL_RESULT" \
        --code-dir "$CODE_DIR" \
        --conda-env "$CONDA_ENV" \
        --output "$LAUNCHER_FREEZE"
fi

"$CONDA" run -n "$CONDA_ENV" python \
    "$CODE_DIR/graphene_r2m_run_provenance.py" validate \
    --launcher-source "$LAUNCHER_SOURCE" \
    --prepared-manifest "$PREPARED_MANIFEST" \
    --fold-dir "$FOLD_DIR" \
    --core-model "$CORE_MODEL" \
    --core-sha256 "$CORE_SHA256" \
    --replay-train "$R2M_DATA/train.xyz" \
    --replay-valid "$R2M_DATA/valid.xyz" \
    --pristine "$PRISTINE" \
    --operator "$OPERATOR" \
    --background "$BACKGROUND" \
    --thermal-result "$THERMAL_RESULT" \
    --code-dir "$CODE_DIR" \
    --conda-env "$CONDA_ENV" \
    --freeze "$LAUNCHER_FREEZE"

if [[ -e "$OUT/DONE" ]]; then
    "$CONDA" run -n "$CONDA_ENV" python \
        "$CODE_DIR/graphene_r2m_run_provenance.py" validate-complete \
        --freeze "$LAUNCHER_FREEZE" \
        --completion "$COMPLETION_MANIFEST" \
        --done "$OUT/DONE" \
        --exit-code "$OUT/EXIT_CODE" \
        --failed "$OUT/FAILED" \
        --running "$OUT/RUNNING"
    echo "outer fold $FOLD_INDEX already complete and provenance-valid"
    exit 0
fi
if [[ -e "$COMPLETION_MANIFEST" ]]; then
    "$CONDA" run -n "$CONDA_ENV" python \
        "$CODE_DIR/graphene_r2m_run_provenance.py" validate-completion \
        --freeze "$LAUNCHER_FREEZE" \
        --completion "$COMPLETION_MANIFEST"
    date -Is > "$OUT/COMPLETED_AT"
    echo 0 > "$OUT/EXIT_CODE"
    rm -f "$OUT/RUNNING"
    rm -f "$OUT/FAILED"
    touch "$OUT/DONE"
    echo "recovered outer fold $FOLD_INDEX from a valid completion manifest"
    exit 0
fi
if [[ -d "$TRAIN_OUT" && ! -e "$TRAIN_OUT/TRAINING_DONE" ]]; then
    echo "refusing implicit restart of partial outer-fold training: $TRAIN_OUT" >&2
    exit 3
fi

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB already used" >&2
    exit 4
fi

if [[ ! -e "$OUT/STARTED_AT" ]]; then
    date -Is > "$OUT/STARTED_AT"
fi
touch "$OUT/RUNNING"
if [[ ! -e "$OUT/gpu_samples.csv" ]]; then
    echo "timestamp_iso,gpu_util_percent,memory_used_MiB,memory_total_MiB" > "$OUT/gpu_samples.csv"
fi
(
    while [[ -e "$OUT/RUNNING" ]]; do
        GPU_SAMPLE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n 1)"
        echo "$(date -Is),$GPU_SAMPLE" >> "$OUT/gpu_samples.csv"
        sleep 10
    done
) &
MONITOR_PID=$!

finish() {
    EXIT_STATUS=$?
    trap - EXIT
    rm -f "$OUT/RUNNING"
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    echo "$EXIT_STATUS" > "$OUT/EXIT_CODE"
    if (( EXIT_STATUS != 0 )); then touch "$OUT/FAILED"; fi
    exit "$EXIT_STATUS"
}
trap finish EXIT

if [[ ! -e "$TRAIN_OUT/TRAINING_DONE" ]]; then
    "$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
        "$CODE_DIR/train_graphene_r2m_routed_tail_outer_fold.py" \
        --fold-dir "$FOLD_DIR" \
        --replay-train "$R2M_DATA/train.xyz" \
        --pristine "$PRISTINE" \
        --core-model "$CORE_MODEL" \
        --core-sha256 "$CORE_SHA256" \
        --launcher-freeze "$LAUNCHER_FREEZE" \
        --output-dir "$TRAIN_OUT" \
        --device cuda
fi

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$CODE_DIR/evaluate_graphene_r2m_routed_tail_outer_fold.py" \
    --fold-dir "$FOLD_DIR" \
    --training-dir "$TRAIN_OUT" \
    --valid "$R2M_DATA/valid.xyz" \
    --pristine "$PRISTINE" \
    --core-model "$CORE_MODEL" \
    --core-sha256 "$CORE_SHA256" \
    --launcher-freeze "$LAUNCHER_FREEZE" \
    --operator "$OPERATOR" \
    --background "$BACKGROUND" \
    --thermal-result "$THERMAL_RESULT" \
    --device cuda \
    --output "$EVALUATION"

"$CONDA" run -n "$CONDA_ENV" python \
    "$CODE_DIR/graphene_r2m_run_provenance.py" write-completion \
    --freeze "$LAUNCHER_FREEZE" \
    --evaluation "$EVALUATION" \
    --output "$COMPLETION_MANIFEST"
date -Is > "$OUT/COMPLETED_AT"
echo 0 > "$OUT/EXIT_CODE"
rm -f "$OUT/RUNNING"
rm -f "$OUT/FAILED"
touch "$OUT/DONE"
echo "R2M routed-tail outer fold $FOLD_INDEX complete"
