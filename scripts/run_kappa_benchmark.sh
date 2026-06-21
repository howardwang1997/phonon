#!/usr/bin/env bash
# Task 1: kappa benchmark across non-polar/covalent materials, baseline vs
# fine-tuned MLIP. GPU MLIP inference (no DFT). Uses GPUs 2-7 (leaves 0-1 free
# for the anharm chain). Compare vs known experimental kappa (analysis script).
cd ~/phonon
PY=~/miniconda3/envs/phonon/bin/python
FT=${FT:-results/ablation/br_B64_s1/ft.model}
MATS=${MATS:-"Ge SiC BN BP BAs AlP AlAs GaP InP C Sn"}
read -ra GPUS <<< "${GPUS:-2 3 4 5 6 7}"
NG=${#GPUS[@]}
mkdir -p results/kappa
i=0
for mat in $MATS; do
  for tm in "small=small" "ft=$FT"; do
    tag=${tm%%=*}; model=${tm#*=}
    out=results/kappa/bench_${mat}_${tag}.json
    [ -f "$out" ] && { echo "skip $mat/$tag"; continue; }
    g=${GPUS[$((i % NG))]}
    CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/run_mlip_kappa.py --material "$mat" --model "$model" \
      --sc 3 --mesh 21 --device cuda --out "$out" > /tmp/kb_${mat}_${tag}.out 2>&1 &
    i=$((i+1))
    (( i % NG == 0 )) && wait
  done
done
wait
echo "KAPPA BENCHMARK DONE ($i runs)"
