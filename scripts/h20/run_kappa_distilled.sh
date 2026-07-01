#!/usr/bin/env bash
# E9 κ re-run with the distilled canon model (b64_s1) across supercell sizes.
# Primary: sc[3,3,3] for all 5 materials. Convergence curve: Si sc2 + sc4.
#   nohup bash scripts/h20/run_kappa_distilled.sh > results/h20/kappa_distilled.log 2>&1 & disown
set -uo pipefail
cd "$HOME/phonon"
PY="$HOME/miniconda3/envs/phonon/bin/python"
CANON="results/ablation/b64_s1/ft.model"
mkdir -p results/h20/e9

run(){ # gpu name structure a kexp supercell outtag
  local g=$1 name=$2 struct=$3 a=$4 kexp=$5 sc=$6 tag=$7
  local out="results/h20/e9/${name}_distilled_${tag}.json"
  echo "[gpu$g] $name sc=$tag -> $out $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$g "$PY" scripts/h20/e9_kappa_mlip.py \
    --name "$name" --structure "$struct" --a "$a" --model-type mace \
    --model "$CANON" --tag distilled --supercell "$sc" --mesh 19 \
    --kappa-exp "$kexp" --out "$out" >> "results/h20/e9/${name}_distilled_${tag}.log" 2>&1 \
    && echo "[gpu$g] OK $name/$tag $(date +%H:%M:%S)" \
    || echo "[gpu$g] FAIL $name/$tag $(date +%H:%M:%S)"
}

# primary: sc[3,3,3] for all 5 (GPUs 0-4)
run 0 Si   diamond    5.43  140  3,3,3 sc3 &
run 1 Ge   diamond    5.658 60   3,3,3 sc3 &
run 2 C    diamond    3.567 2200 3,3,3 sc3 &
run 3 GaAs zincblende 5.653 45   3,3,3 sc3 &
run 4 BAs  zincblende 4.777 1300 3,3,3 sc3 &
# Si convergence curve: sc2 + sc4 (GPUs 6-7)
run 6 Si   diamond    5.43  140  2,2,2 sc2 &
run 7 Si   diamond    5.43  140  4,4,4 sc4 &
wait
echo "=== kappa distilled batch DONE $(date) ==="
