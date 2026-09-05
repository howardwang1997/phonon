#!/usr/bin/env bash
# Wait for both frozen P4 target lanes, then perform the one-shot C1 evaluation.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
RESULTS="$ROOT/results/graphene_fd_transferability"
OUT="$RESULTS/T450/on_policy"
LABELS="$ROOT/data/graphene_fd_transferability/T450/on_policy/dft_force_labels"
MERGED="$LABELS/merged"
EVAL="$OUT/evaluation"
PREDICT="$OUT/frozen_prediction"
FREEZE="$RESULTS/freeze_manifest.json"
SAMPLING="$OUT/sampling_acceptance_n3000.json"
SAMPLING_BOOTSTRAP="$OUT/sampling_bootstrap_acceptance_n3000.json"
DFPT_ROOT="$ROOT/results/graphene_physical_fd_dfpt/campaigns"
DFPT_CONV="$DFPT_ROOT/FD450_CONV/convergence_acceptance.json"
DFPT_LINE="$DFPT_ROOT/FD450_LINE/graphene_FD450_LINE_dfpt.csv"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
BASE="$ROOT/results/gr_backbone_v11/ft_graphene.model"
DELTA300="$ROOT/results/graphene_fd_delta_pilot/T300/delta32/gr_fd300_delta32.model"
DELTA600="$ROOT/results/graphene_fd_delta_weighted/T600/selected_checkpoint.model"
OPERATOR="$RESULTS/conditioned_short/T450_long_range_operator.npz"
LAW="$RESULTS/temperature_law.json"
LOG="$OUT/run_p4c_evaluation.log"

cd "$ROOT"
mkdir -p "$OUT" "$EVAL" "$PREDICT" "$MERGED"
exec 9>"$OUT/.p4c_evaluation.lock"
if ! flock -n 9; then
    echo "P4c evaluation is already running; refusing a duplicate"
    exit 0
fi
exec > >(tee -a "$LOG") 2>&1

if [[ -e "$OUT/P4_COMPLETE" && -s "$RESULTS/transferability_acceptance.json" ]]; then
    echo "P4c evaluation already complete"
    exit 0
fi

export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_round=0
while [[ ! -e "$LABELS/shard_A/RAW_READY" \
      || ! -e "$LABELS/shard_B/RAW_READY" \
      || ! -e "$DFPT_ROOT/FD450_LINE/REMOTE_DONE" \
      || ! -s "$DFPT_CONV" \
      || ! -s "$DFPT_LINE" ]]; do
    if (( wait_round % 10 == 0 )); then
        echo "waiting for P4 shard A/B and FD450_LINE: $(date -Is)"
    fi
    wait_round=$((wait_round + 1))
    sleep 60
done

for path in "$FREEZE" "$SAMPLING" "$SAMPLING_BOOTSTRAP" "$BG" \
    "$BASE" "$DELTA300" "$DELTA600" "$OPERATOR" "$LAW" \
    "$LABELS/shard_A/seed0/summary.json" "$LABELS/shard_A/seed0/summary.xyz" \
    "$LABELS/shard_A/seed2/summary.json" "$LABELS/shard_A/seed2/summary.xyz" \
    "$LABELS/shard_B/seed1/summary.json" "$LABELS/shard_B/seed1/summary.xyz" \
    "$LABELS/shard_B/seed2/summary.json" "$LABELS/shard_B/seed2/summary.xyz"; do
    [[ -s "$path" ]]
done

available_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if [[ -z "$available_kb" || "$available_kb" -lt 52428800 ]]; then
    echo "refusing P4c evaluation: less than 50 GiB free on RTX filesystem" >&2
    exit 75
fi

echo "=== graphene transferability P4c START $(date -Is) ==="
if [[ ! -s "$MERGED/manifest.json" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/merge_graphene_fd_p4_labels.py \
        --labels-root "$LABELS" --freeze-manifest "$FREEZE" \
        --sampling-acceptance "$SAMPLING" \
        --sampling-bootstrap "$SAMPLING_BOOTSTRAP" --output-dir "$MERGED"
fi

if [[ ! -s "$PREDICT/manifest.json" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/predict_graphene_fd_p4.py \
        --labels "$MERGED/all60.xyz" --merge-manifest "$MERGED/manifest.json" \
        --freeze-manifest "$FREEZE" --base-model "$BASE" \
        --delta-model-300 "$DELTA300" --delta-model-600 "$DELTA600" \
        --operator "$OPERATOR" --background "$BG" --temperature 450 --device cuda \
        --output-predictions "$PREDICT/frozen_force_predictions.npz" \
        --output-static "$PREDICT/frozen_static_fc2.npz" \
        --output-manifest "$PREDICT/manifest.json"
fi

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/bootstrap_graphene_fd_p4_dft_tdep.py \
    --seed-label "0=$MERGED/seed0.xyz" --seed-label "1=$MERGED/seed1.xyz" \
    --seed-label "2=$MERGED/seed2.xyz" --merge-manifest "$MERGED/manifest.json" \
    --freeze-manifest "$FREEZE" --sampling-bootstrap "$SAMPLING_BOOTSTRAP" \
    --background "$BG" --temperature 450 --replicates 1000 --checkpoint-every 20 \
    --output-tdep "$OUT/dft_tdep_60.npz" \
    --output-summary "$OUT/dft_tdep_60.json" \
    --bootstrap-output "$OUT/dft_tdep_bootstrap.npz" \
    --bootstrap-summary "$OUT/dft_tdep_bootstrap.json"

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_fd_transferability.py \
    --labels "$MERGED/all60.xyz" --merge-manifest "$MERGED/manifest.json" \
    --predictions "$PREDICT/frozen_force_predictions.npz" \
    --prediction-manifest "$PREDICT/manifest.json" \
    --static-prediction "$PREDICT/frozen_static_fc2.npz" \
    --dft-tdep "$OUT/dft_tdep_60.npz" --dft-tdep-summary "$OUT/dft_tdep_60.json" \
    --dft-bootstrap "$OUT/dft_tdep_bootstrap.npz" \
    --dft-bootstrap-summary "$OUT/dft_tdep_bootstrap.json" \
    --short-tdep "$OUT/short_pooled_n9000.npz" --temperature-law "$LAW" \
    --freeze-manifest "$FREEZE" --sampling-acceptance "$SAMPLING" \
    --sampling-bootstrap "$SAMPLING_BOOTSTRAP" \
    --dfpt-convergence "$DFPT_CONV" --dfpt-line "$DFPT_LINE" \
    --background "$BG" --output-dir "$EVAL"

cp -p "$EVAL/transferability_acceptance.json" \
    "$RESULTS/transferability_acceptance.json.partial"
mv "$RESULTS/transferability_acceptance.json.partial" \
    "$RESULTS/transferability_acceptance.json"
if "$CONDA" run -n phonon python -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passes_C1"] else 1)' \
    "$RESULTS/transferability_acceptance.json"; then
    touch "$OUT/P4_C1_PASSED"
else
    touch "$OUT/P4_C1_FAILED"
fi
touch "$OUT/P4_COMPLETE"
echo "=== graphene transferability P4c COMPLETE $(date -Is) ==="

