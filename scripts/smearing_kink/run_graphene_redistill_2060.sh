#!/usr/bin/env bash
# 2060: finetune graphene backbone v2 (8x8 data, 200 epochs) after Box B relays it.
set -uo pipefail
cd ~/phonon
PY=/home/howardwang/miniconda3/envs/phonon/bin/python
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=data/finetune_graphene8
echo "[redistill-2060] $(date) waiting for $DATA/.relayed ..."
while [ ! -f "$DATA/.relayed" ]; do sleep 60; done
echo "[redistill-2060] $(date) data arrived -> finetune 200ep"
DATA_DIR=$DATA OUT_DIR=results/gr_backbone_v2 EPOCHS=200 bash scripts/finetune_graphene.sh 200 cuda 2>&1 | tail -8
mkdir -p results/gr_backbone_v2; touch results/gr_backbone_v2/.done
echo "[redistill-2060] DONE $(date) -> results/gr_backbone_v2/"
