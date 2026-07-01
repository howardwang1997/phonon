#!/usr/bin/env bash
# =============================================================================
# (L)-channel SSCHA follow-up — persistent autonomous re-run of the TMD CDW
# family with improved convergence settings (more populations + smaller minimizer
# step), since the campaign-default (200/5) diverged on the noisy MLIP forces.
# Round-robin across GPUs, idempotent (skip done), overwrites lchannel/*.csv so
# the convergence-guarded aggregate.py picks up the best data.
#
#   nohup bash scripts/h20/run_sscha_followup.sh > results/h20/sscha_followup.log 2>&1 & disown
# =============================================================================
set -uo pipefail
cd "$HOME/phonon"
SPY="$HOME/miniconda3/envs/sscha14/bin/python"
CANON="results/ablation/b64_s1/ft.model"
DONE="results/h20/lchannel_v2_done"; mkdir -p "$DONE" results/h20/lchannel results/h20/joblogs
NGPU="${NGPU:-8}"
NCFG="${NCFG:-400}"; MAXPOP="${MAXPOP:-12}"; MINSTEP="${MINSTEP:-0.02}"
TEMPS="100,200,300"

# 7 CDW family members (sscha=true in config) x {foundation medium, distilled canon}
# fields: name formula polytype a thickness cdw_exp_K
MATS=(
  "NbSe2 NbSe2 2H 3.44 3.34 145"
  "NbS2 NbS2 2H 3.33 3.00 null"
  "2H-TaS2 TaS2 2H 3.31 3.00 75"
  "1T-TaS2 TaS2 1T 3.36 2.90 350"
  "2H-TaSe2 TaSe2 2H 3.43 3.30 122"
  "1T-TiSe2 TiSe2 1T 3.53 2.90 200"
  "1T-VSe2 VSe2 1T 3.34 3.00 110"
)
# build job list: each line = tag \t model-arg
JOBS=()
for m in "${MATS[@]}"; do
  read -r name formula poly a th cdw <<<"$m"
  JOBS+=("$name|$formula|$poly|$a|$th|$cdw|found|medium")
  JOBS+=("$name|$formula|$poly|$a|$th|$cdw|distilled|$CANON")
done
NJ=${#JOBS[@]}
echo "=== sscha followup start $(date): $NJ jobs over $NGPU GPUs (ncfg=$NCFG maxpop=$MAXPOP minstep=$MINSTEP) ==="

run_one(){   # $1 = job "name|formula|poly|a|th|cdw|tag|model"
  local job="$1" name formula poly a th cdw tag model out marker
  IFS='|' read -r name formula poly a th cdw tag model <<<"$job"
  out="results/h20/lchannel/${name}_sscha_${tag}.csv"
  marker="$DONE/${name}_${tag}.done"
  if [ -f "$marker" ]; then echo "[skip] $name/$tag (done)"; return; fi
  local jid="${name}_sscha_${tag}_v2"
  echo "[run] $name/$tag (model=${model##*/}) $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$2" "$SPY" scripts/h20/lchannel_sscha.py --mode sscha \
    --name "$name" --formula "$formula" --polytype "$poly" --a "$a" --thickness "$th" \
    --model-type mace --model "$model" --tag "$tag" --supercell 4,4,1 --displacement 0.03 \
    --nconfigs "$NCFG" --maxpop "$MAXPOP" --min-step "$MINSTEP" --temperatures "$TEMPS" \
    --cdw-exp-K "$cdw" --out "$out" > "results/h20/joblogs/${jid}.log" 2>&1 \
    && { touch "$marker"; echo "[OK] $name/$tag $(date +%H:%M:%S)"; } \
    || echo "[FAIL] $name/$tag $(date +%H:%M:%S) (see joblogs/${jid}.log)"
}

# round-robin: each GPU slot consumes every NGPU-th job
worker(){  # $1 = slot
  local slot=$1 i=0
  for job in "${JOBS[@]}"; do
    if (( i % NGPU == slot )); then run_one "$job" "$slot"; fi
    i=$((i+1))
  done
}
for s in $(seq 0 $((NGPU-1))); do [ $s -lt $NJ ] && worker "$s" & done
wait
echo "=== sscha followup DONE $(date) ==="
