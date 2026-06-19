"""Active-learning UNCERTAINTY arm of the acquisition experiment (E7).

Start from a fixed seed set, then each round add the K candidates the CURRENT
fine-tuned model is most *uncertain* about (predicted imaginary modes / ASR), via
score_candidates.py; retrain+eval on the grown set (reusing run_one_job.sh).
Held-out transfer MAE vs #materials is the curve we compare against:
  - RANDOM acquisition (round-4: rndA/B/C),
  - DIVERSITY acquisition (the greedy B-curve: B8/B16/B24/B32/B48).
MDR plays the DFT oracle (FC distillation is free once a material is "queried").

    python scripts/al_loop.py --gpu 0 --seed-key B8 --pool-key B79 --k 8 --rounds 5
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_sets():
    V = {}
    for line in (ROOT / "scripts" / "campaign_materials.sh").read_text().splitlines():
        m = re.match(r'(\w+)="(.*)"', line)
        if m:
            V[m.group(1)] = m.group(2).split()
    return V


def run_one_job(job, train, hold, gpu, ncfg=30, mode="pt1000"):
    env = dict(os.environ, JOB=job, SEED="1", NCFG=str(ncfg), MODE=mode, MODEL="small",
               TRAIN=" ".join(train), HOLD=" ".join(hold), CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    subprocess.run(["bash", "scripts/run_one_job.sh"], cwd=ROOT, env=env, check=True)


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed-key", default="B8")
    ap.add_argument("--pool-key", default="B79")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    V = load_sets()
    SET = list(V[args.seed_key])
    POOL = [m for m in V[args.pool_key] if m not in SET]
    HOLD = V["HOLD"]
    PY = f"{os.environ['HOME']}/miniconda3/envs/phonon/bin/python"

    # round 0: seed model (reuse the existing greedy-B8 model if present, else train)
    n = len(SET)
    seed_model = ROOT / "results" / "ablation" / f"br_{args.seed_key}_s1" / "ft.model"
    if not seed_model.exists():
        run_one_job(f"al_unc_N{n}", SET, HOLD, args.gpu)
        cur_model = ROOT / "results" / "ablation" / f"al_unc_N{n}" / "ft.model"
    else:
        cur_model = seed_model
    print(f"[AL] round 0: N={n} seed set, model={cur_model.name}", flush=True)

    for r in range(1, args.rounds + 1):
        remain = [m for m in POOL if m not in SET]
        if not remain:
            break
        scores = ROOT / "results" / "ablation" / f"al_scores_N{len(SET)}.csv"
        subprocess.run([PY, "scripts/score_candidates.py", "--model", str(cur_model),
                        "--candidates", *remain, "--device", "cuda", "--out", str(scores)],
                       cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                                          OMP_NUM_THREADS="4"), check=True)
        sc = pd.read_csv(scores)
        sc = sc[sc.get("uncertainty").notna()] if "uncertainty" in sc else sc
        picks = list(sc.sort_values("uncertainty", ascending=False).head(args.k).mp_id)
        SET += picks
        n = len(SET)
        print(f"[AL] round {r}: +{len(picks)} by uncertainty -> N={n}; picks={picks}", flush=True)
        run_one_job(f"al_unc_N{n}", SET, HOLD, args.gpu)
        cur_model = ROOT / "results" / "ablation" / f"al_unc_N{n}" / "ft.model"

    print("[AL] uncertainty arm done. sizes ->", [len(V[args.seed_key]) + args.k * i for i in range(args.rounds + 1)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
