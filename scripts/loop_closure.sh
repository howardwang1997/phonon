#!/usr/bin/env bash
# Line B -> Line A LOOP CLOSURE: FC-distill an MLIP on SELF-GENERATED DFT force
# constants (not public MDR), proving the data engine drives the model.
# Waits for the broad dataset to produce >=MIN self-DFT FC files, copies them into
# the MDR cache dir (so reference.fetch finds them by id), then trains+evals via
# the proven run_one_job.sh. Round-trip: does the MLIP reproduce the self-DFT phonons?
set -uo pipefail
cd ~/phonon
MIN=${MIN:-8}
echo "[loop] waiting for >=$MIN self-DFT FC files ..."
while [ "$(ls data/dft_ref/selfdft-*.yaml 2>/dev/null | wc -l)" -lt "$MIN" ]; do sleep 120; done

# expose self-DFT force constants under reference.fetch's lookup dir
cp data/dft_ref/selfdft-*.yaml data/benchmark/mdr/ 2>/dev/null
IDS=$(ls data/dft_ref/selfdft-*.yaml | sed 's#.*/##; s/_phonopy_params.yaml//')
N=$(echo "$IDS" | wc -l)
TRAIN=$(echo "$IDS" | head -n $((N-2)) | tr '\n' ' ')
HOLD=$(echo "$IDS" | tail -2 | tr '\n' ' ')
echo "[loop] self-DFT materials: $N  (train $((N-2)), holdout 2)"
echo "[loop] TRAIN=$TRAIN"
echo "[loop] HOLD=$HOLD"

# fine-tune MACE on self-DFT FCs (single-head; small self-set) on GPU 0
CUDA_VISIBLE_DEVICES=0 JOB=loopclose SEED=1 NCFG=30 MODE=single MODEL=small \
  TRAIN="$TRAIN" HOLD="$HOLD" bash scripts/run_one_job.sh
echo "[loop] DONE -> results/ablation/eval_loopclose.csv (self-DFT -> MLIP round-trip)"
