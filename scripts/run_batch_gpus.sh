#!/usr/bin/env bash
# Run a jobs.tsv across an explicit GPU list (round-robin, sequential per GPU).
# Lets a batch use GPUs 1-7 while another job (e.g. the AL loop) holds GPU 0.
#   GPULIST="1 2 3 4 5 6 7" JOBS=scripts/jobs5p.tsv bash scripts/run_batch_gpus.sh
set -uo pipefail
cd ~/phonon
source scripts/campaign_materials.sh
JOBS=${JOBS:?need JOBS}
read -ra GPUS <<< "${GPULIST:-0 1 2 3 4 5 6 7}"
NG=${#GPUS[@]}
DLOG=results/ablation/batch.log
mkdir -p results/ablation
echo "=== batch start $(date): $(grep -cvE '^#|^$' "$JOBS") jobs over GPUs ${GPUS[*]} ===" | tee "$DLOG"

run_slot() {                       # $1 = slot index (0..NG-1)
  local slot=$1 g=${GPUS[$1]} i=0 job seed ncfg mode model trainkey tr
  while IFS=$'\t' read -r job seed ncfg mode model trainkey; do
    [[ -z "${job:-}" || "$job" == \#* ]] && continue
    if (( i % NG == slot )); then
      if [ -f "results/ablation/eval_$job.csv" ]; then echo "[gpu$g] skip $job" >>"$DLOG"; else
        tr="${!trainkey}"
        echo "[gpu$g] >> $job ($trainkey seed=$seed) $(date)" >>"$DLOG"
        CUDA_VISIBLE_DEVICES=$g JOB="$job" SEED="$seed" NCFG="$ncfg" MODE="$mode" MODEL="$model" \
          TRAIN="$tr" HOLD="$HOLD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/run_one_job.sh >>"$DLOG" 2>&1 \
          && echo "[gpu$g] OK $job $(date)" >>"$DLOG" || echo "[gpu$g] ERR $job $(date)" >>"$DLOG"
      fi
    fi
    ((i++))
  done < "$JOBS"
}

for s in $(seq 0 $((NG-1))); do run_slot "$s" & done
wait
echo "=== batch DONE $(date) ===" | tee -a "$DLOG"
