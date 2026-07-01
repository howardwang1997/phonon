#!/usr/bin/env bash
# =============================================================================
# NEW experiment: κ-vs-distillation-breadth curve.
# Does κ converge as the FC-distillation breadth grows (b16 -> b32 -> b64 -> B79)?
# Reuses existing breadth models (no new training). Strong paper figure: pairs
# with the held-out MAE breadth law to show breadth buys downstream κ too.
# Round-robin over GPUs, idempotent.
#   nohup bash scripts/h20/run_kappa_breadth.sh > results/h20/kappa_breadth.log 2>&1 & disown
# =============================================================================
set -uo pipefail
cd "$HOME/phonon"
PY="$HOME/miniconda3/envs/phonon/bin/python"
mkdir -p results/h20/e9 results/h20/joblogs
NGPU="${NGPU:-8}"

# materials: name structure a kappa_exp
MATS=( "Si diamond 5.43 140" "Ge diamond 5.658 60" "C diamond 3.567 2200" \
       "GaAs zincblende 5.653 45" "BAs zincblende 4.777 1300" )
# breadth models: modeldir tag
MODS=( "b16_s1 b16" "b32_s1 b32" "b64_s1 b64" "br_B79_s1 B79" )

JOBS=()
for m in "${MATS[@]}"; do
  read -r name st a kexp <<<"$m"
  for md in "${MODS[@]}"; do
    read -r mdir mtag <<<"$md"
    JOBS+=("$name|$st|$a|$kexp|$mdir|$mtag")
  done
done
NJ=${#JOBS[@]}
echo "=== kappa-breadth start $(date): $NJ jobs over $NGPU GPUs ==="

run_one(){  # $1=slot $2=job "name|st|a|kexp|modeldir|mtag"
  local slot=$1 job=$2 name st a kexp mdir mtag out marker
  IFS='|' read -r name st a kexp mdir mtag <<<"$job"
  out="results/h20/e9/${name}_breadth${mtag}_sc3.json"
  marker="results/h20/e9/${name}_breadth${mtag}_sc3.done"
  [ -f "$marker" ] && { echo "[skip] $name/$mtag"; return; }
  local jid="${name}_breadth${mtag}_sc3"
  echo "[gpu$slot] $name / $mtag $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$slot" "$PY" scripts/h20/e9_kappa_mlip.py \
    --name "$name" --structure "$st" --a "$a" --model-type mace \
    --model "results/ablation/$mdir/ft.model" --tag "breadth_$mtag" \
    --supercell 3,3,3 --mesh 19 --kappa-exp "$kexp" --out "$out" \
    > "results/h20/joblogs/${jid}.log" 2>&1 \
    && { touch "$marker"; echo "[OK] $name/$mtag $(date +%H:%M:%S)"; } \
    || echo "[FAIL] $name/$mtag $(date +%H:%M:%S)"
}
worker(){ local slot=$1 i=0; for j in "${JOBS[@]}"; do (( i % NGPU == slot )) && run_one "$slot" "$j"; i=$((i+1)); done; }
for s in $(seq 0 $((NGPU-1))); do [ $s -lt $NJ ] && worker "$s" & done
wait
echo "=== kappa-breadth DONE $(date) ==="
