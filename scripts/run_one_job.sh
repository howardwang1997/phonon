#!/usr/bin/env bash
# ONE fine-tune+eval job, pinned to a GPU by the caller (CUDA_VISIBLE_DEVICES).
# Params via env: JOB SEED NCFG MODE MODEL TRAIN HOLD [EPOCHS]
#   MODE = single | pt<N> | lora<R>
# Each job uses its OWN data dir so 8 run concurrently without clobbering.
# Resumable: if $OUT/ft.model already exists, skips data+train and only evals.
set -uo pipefail
cd ~/phonon
# cap BLAS/OMP threads so 8 concurrent evals don't thrash the 192-core node
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
PY=$HOME/miniconda3/envs/phonon/bin/python
MT=$HOME/miniconda3/envs/phonon/bin/mace_run_train
JOB=${JOB:?need JOB}; SEED=${SEED:-1}; NCFG=${NCFG:-30}; MODE=${MODE:-pt1000}
MODEL=${MODEL:-small}; EPOCHS=${EPOCHS:-50}
: "${TRAIN:?need TRAIN}" "${HOLD:?need HOLD}"

DD=data/finetune_$JOB
OUT=results/ablation/$JOB
LOG=results/ablation/$JOB.joblog
mkdir -p results/ablation "$OUT"

if [ -f "$OUT/ft.model" ]; then
  echo "=== [$JOB] ft.model exists -> eval only $(date) ===" | tee "$LOG"
else
  rm -rf "$DD"; mkdir -p "$DD"
  echo "=== [$JOB] gpu=${CUDA_VISIBLE_DEVICES:-?} seed=$SEED ncfg=$NCFG mode=$MODE model=$MODEL $(date) ===" | tee "$LOG"
  # 1. distilled dataset
  $PY scripts/make_finetune_data.py --out "$DD" --train $TRAIN --n-configs "$NCFG" --n-single-sites 4 >>"$LOG" 2>&1 || { echo "[$JOB] DATA FAIL" | tee -a "$LOG"; exit 2; }
  E0S=$($PY -c "from ase.io import read; zs=sorted({int(z) for a in read('$DD/train.xyz',':') for z in a.numbers}); print('{'+','.join(f'{z}:0.0' for z in zs)+'}')")
  # 2. anti-forgetting mode -> MACE flags
  case "$MODE" in
    single|pt0) MH="--multiheads_finetuning False" ;;
    pt*)        MH="--multiheads_finetuning True --num_samples_pt ${MODE#pt} --subselect_pt random" ;;
    lora*)      R=${MODE#lora}; MH="--multiheads_finetuning False --lora True --lora_rank $R --lora_alpha $R" ;;
    *) echo "[$JOB] bad MODE=$MODE" | tee -a "$LOG"; exit 3 ;;
  esac
  # 3. fine-tune
  $MT --name ft --foundation_model "$MODEL" $MH --foundation_model_elements True \
    --train_file "$DD/train.xyz" --valid_file "$DD/val.xyz" \
    --energy_key REF_energy --forces_key REF_forces --E0s "$E0S" \
    --energy_weight 0.01 --forces_weight 100.0 --max_num_epochs "$EPOCHS" \
    --batch_size 32 --valid_batch_size 32 --eval_interval 10 --lr 0.001 \
    --default_dtype float32 --device cuda --seed "$SEED" \
    --model_dir "$OUT" --results_dir "$OUT" --log_dir "$OUT" \
    --checkpoints_dir "$OUT" --save_cpu >>"$LOG" 2>&1 || { echo "[$JOB] TRAIN FAIL" | tee -a "$LOG"; exit 4; }
fi

# 4. eval vs DFPT at the DFT geometry (--no-relax): faster AND a cleaner FC-quality
#    metric (relaxation drift removed). Held-out = full fixed set (headline); the
#    in-domain probe is capped to the first 8 (nested -> same B8 for every N).
EVALTRAIN=$(echo $TRAIN | tr ' ' '\n' | head -8 | tr '\n' ' ')
$PY scripts/eval_finetune.py --baseline "$MODEL" --ft-model "$OUT/ft.model" --device cuda --ft-only --no-relax \
  --train $EVALTRAIN --holdout $HOLD --out "results/ablation/eval_$JOB.csv" >>"$LOG" 2>&1 || { echo "[$JOB] EVAL FAIL" | tee -a "$LOG"; exit 5; }

# 5. free disk (keep ft.model + eval csv)
rm -rf "$DD" "$OUT"/checkpoints "$OUT"/*.pt 2>/dev/null
echo "=== [$JOB] DONE $(date) ===" | tee -a "$LOG"
