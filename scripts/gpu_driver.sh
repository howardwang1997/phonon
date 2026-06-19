#!/usr/bin/env bash
# Autonomous 8-GPU campaign driver. Each GPU worker scans jobs.tsv and runs every
# NGPU-th job (round-robin), sequentially. Resumable: skips jobs whose eval CSV
# already exists. Detach with: nohup bash scripts/gpu_driver.sh & disown
set -uo pipefail
cd ~/phonon
source scripts/campaign_materials.sh      # defines HOLD, B4 B8 B16 B32 B64, TRAIN16
JOBS=${JOBS:-scripts/jobs.tsv}
NGPU=${NGPU:-8}
DLOG=results/ablation/driver.log
mkdir -p results/ablation
echo "=== driver start $(date): $(grep -cvE '^#|^$' "$JOBS") jobs over $NGPU gpus ===" | tee "$DLOG"

run_gpu() {              # $1 = gpu id
  local g=$1 i=0 job seed ncfg mode model trainkey tr
  while IFS=$'\t' read -r job seed ncfg mode model trainkey; do
    [[ -z "${job:-}" || "$job" == \#* ]] && continue
    if (( i % NGPU == g )); then
      if [ -f "results/ablation/eval_$job.csv" ]; then echo "[gpu$g] skip $job (done)" >>"$DLOG"; else
        tr="${!trainkey}"
        echo "[gpu$g] >> START $job (seed=$seed ncfg=$ncfg mode=$mode model=$model train=$trainkey) $(date)" >>"$DLOG"
        CUDA_VISIBLE_DEVICES=$g JOB="$job" SEED="$seed" NCFG="$ncfg" MODE="$mode" MODEL="$model" \
          TRAIN="$tr" HOLD="$HOLD" bash scripts/run_one_job.sh >>"$DLOG" 2>&1 \
          && echo "[gpu$g] << OK   $job $(date)" >>"$DLOG" \
          || echo "[gpu$g] << ERR  $job (rc=$?) $(date)" >>"$DLOG"
      fi
    fi
    ((i++))
  done < "$JOBS"
  echo "[gpu$g] drained $(date)" >>"$DLOG"
}

for g in $(seq 0 $((NGPU-1))); do run_gpu "$g" & done
wait
echo "=== driver ALL DONE $(date) ===" | tee -a "$DLOG"
