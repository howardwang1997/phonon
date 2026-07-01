#!/usr/bin/env bash
# =============================================================================
# 8×H20 campaign — ONE command, runs the whole menu to completion.
#   git pull && nohup bash scripts/h20/run_campaign.sh > results/h20/campaign.log 2>&1 & disown
#   bash scripts/h20/status.sh        # watch progress
#
# Self-healing + idempotent + resumable: safe to Ctrl-C and re-launch; finished
# jobs are skipped, crashed jobs are retried. Wave A = all model-independent
# work (E1 / fine-tunes / foundation L-channel / κ); Wave B = the distilled-model
# re-runs (gated on Wave A's canon model existing).
# =============================================================================
set -uo pipefail
cd "$HOME/phonon"
H=results/h20; mkdir -p "$H"
CONDA="$HOME/miniconda3"; PHPY="$CONDA/envs/phonon/bin/python"
NGPU="${NGPU:-8}"

echo "############ H20 campaign start $(date) (NGPU=$NGPU) ############"

# 1. self-healing bootstrap: envs + data (won't abort the campaign on a single
#    optional-env failure; those jobs just get skipped).
bash scripts/h20/bootstrap_h20_full.sh || echo "!! bootstrap reported issues — see $H/BOOTSTRAP_STATUS.txt (continuing)"

# 2. (re)generate the manifest from the config
"$PHPY" scripts/h20/gen_h20_manifest.py --config configs/h20_campaign.yaml --out "$H/jobs.jsonl" || {
  echo "!! manifest generation failed — aborting"; exit 1; }

# 3. free claims left by a crashed prior run so they get retried
"$PHPY" scripts/h20/hqueue.py release-stale

run_wave() {
  local wave="$1" pids=()
  echo "==== Wave $wave: launch $NGPU workers $(date) ===="
  for g in $(seq 0 $((NGPU-1))); do
    bash scripts/h20/queue_worker.sh "$g" "$wave" &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "==== Wave $wave drained $(date) ===="
}

# 4. Wave A
run_wave A

# 5. Wave B — only if every Wave A job produced its done-marker (canon model ready)
if "$PHPY" scripts/h20/hqueue.py wave-ready --wave B; then
  run_wave B
else
  echo "!! Wave A has unfinished/failed jobs -> SKIPPING Wave B."
  echo "   Inspect $H/joblogs/, fix, then re-run run_campaign.sh to retry."
fi

# 6. aggregate everything into master tables + the origin map
"$PHPY" scripts/h20/aggregate.py || echo "!! aggregate failed (results still in $H/)"

echo "############ H20 campaign done $(date) ############"
"$PHPY" scripts/h20/hqueue.py status
