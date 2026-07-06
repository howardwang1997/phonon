#!/usr/bin/env bash
# =============================================================================
# NEW experiment (phase 2): κ cell-convergence at sc[4,4,4] for the materials
# that were UNDERESTIMATED at sc3 (Ge/GaAs/BAs/C), using the best two breadth
# models (b64, B79). Si sc4 was already measured (=139.8). Tests whether the
# underestimate is a cell-convergence artifact vs a model-fc3 limit.
#   nohup bash scripts/h20/run_kappa_sc4.sh > results/h20/kappa_sc4.log 2>&1 & disown
# =============================================================================
set -uo pipefail
cd "$HOME/phonon"
PY="$HOME/miniconda3/envs/phonon/bin/python"
mkdir -p results/h20/e9 results/h20/joblogs
NGPU="${NGPU:-8}"
MATS=( "Ge diamond 5.658 60" "GaAs zincblende 5.653 45" "BAs zincblende 4.777 1300" "C diamond 3.567 2200" )
MODS=( "b64_s1 b64" "br_B79_s1 B79" )
JOBS=()
for m in "${MATS[@]}"; do read -r name st a kexp <<<"$m"; for md in "${MODS[@]}"; do
  read -r mdir mtag <<<"$md"; JOBS+=("$name|$st|$a|$kexp|$mdir|$mtag")
done; done
NJ=${#JOBS[@]}; echo "=== kappa-sc4 start $(date): $NJ jobs over $NGPU GPUs ==="
run_one(){ local slot=$1 job=$2 name st a kexp mdir mtag out marker jid
  IFS='|' read -r name st a kexp mdir mtag <<<"$job"
  out="results/h20/e9/${name}_breadth${mtag}_sc4.json"; marker="${out}.done"
  [ -f "$marker" ] && { echo "[skip] $name/$mtag/sc4"; return; }
  jid="${name}_breadth${mtag}_sc4"
  echo "[gpu$slot] $name / $mtag / sc4 $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$slot" "$PY" scripts/h20/e9_kappa_mlip.py \
    --name "$name" --structure "$st" --a "$a" --model-type mace \
    --model "results/ablation/$mdir/ft.model" --tag "breadth_$mtag" \
    --supercell 4,4,4 --mesh 19 --kappa-exp "$kexp" --out "$out" \
    > "results/h20/joblogs/${jid}.log" 2>&1 \
    && { touch "$marker"; echo "[OK] $name/$mtag/sc4 $(date +%H:%M:%S)"; } \
    || echo "[FAIL] $name/$mtag/sc4 $(date +%H:%M:%S)"
}
worker(){ local slot=$1 i=0; for j in "${JOBS[@]}"; do (( i % NGPU == slot )) && run_one "$slot" "$j"; i=$((i+1)); done; }
for s in $(seq 0 $((NGPU-1))); do [ $s -lt $NJ ] && worker "$s" & done
wait
echo "=== kappa-sc4 DONE $(date) ==="
