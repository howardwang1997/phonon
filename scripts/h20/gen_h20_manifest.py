"""Generate the 8×H20 job manifest (results/h20/jobs.jsonl) from
configs/h20_campaign.yaml.

Every line is one atomic, idempotent job:
    {"id","group","wave","env","gpu","cmd","done","est_min"}
``done`` is a marker-file path; the job is complete iff it exists. Jobs are
emitted sorted long-first WITHIN each wave so the work-stealing queue overlaps
the long pole (fine-tunes, SSCHA) with the many short jobs instead of stranding
idle GPUs at the tail.

    python scripts/h20/gen_h20_manifest.py --config configs/h20_campaign.yaml \
        --out results/h20/jobs.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_set(name: str) -> str:
    """Resolve a breadth set (B16/B32/...) from scripts/campaign_materials.sh."""
    r = subprocess.run(
        ["bash", "-c", f"source {ROOT}/scripts/campaign_materials.sh && echo ${name}"],
        capture_output=True, text=True)
    ids = r.stdout.strip()
    if not ids:
        raise SystemExit(f"[gen] set {name!r} resolved empty from campaign_materials.sh")
    return ids


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/h20_campaign.yaml")
    ap.add_argument("--out", default="results/h20/jobs.jsonl")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / a.config).read_text())

    rt = cfg["runtime"]
    dev = rt["device"]
    jobs: list[dict] = []

    def add(jid, group, wave, env, cmd, done, est, gpu=True):
        jobs.append({"id": jid, "group": group, "wave": wave, "env": env,
                     "gpu": gpu, "cmd": cmd, "done": done, "est_min": est})

    # ---- E1: foundation models × MDR shards (Wave A) -----------------------
    e1 = cfg["e1"]
    pool, ns = e1["pool_file"], e1["nshards"]
    e1_models = [m for m in cfg["models"] if m.get("e1")]
    for m in e1_models:
        for s in range(ns):
            out = f"results/h20/e1/{m['name']}_s{s:02d}.csv"
            cmd = (f"python scripts/h20/e1_benchmark.py --calc {m['key']} "
                   f"--tag {m['name']} --pool {pool} --shard {s} --nshards {ns} "
                   f"--mesh {e1['mesh']} --displacement {e1['displacement']} "
                   f"--device {dev} --out {out}")
            add(f"e1_{m['name']}_s{s:02d}", "e1", "A", m["env"], cmd, out + ".done", 15)

    # ---- E3/E4: FC-distillation fine-tunes (Wave A) ------------------------
    ft = cfg["finetune"]
    canon = ft["canon"]
    canon_job = f"{canon['tag']}_s{canon['seed']}"
    canon_model = f"results/ablation/{canon_job}/ft.model"
    canon_seen = False
    for j in ft["jobs"]:
        train_ids = resolve_set(j["train"])
        for seed in j["seeds"]:
            JOB = f"{j['tag']}_s{seed}"
            if JOB == canon_job:
                canon_seen = True
            done = f"results/ablation/eval_{JOB}.csv"
            cmd = (f"JOB={JOB} SEED={seed} NCFG={ft['ncfg']} MODE={j['mode']} "
                   f"MODEL={ft['base_model']} EPOCHS={ft['epochs']} "
                   f"TRAIN='{train_ids}' HOLD='{ft['hold']}' "
                   f"bash scripts/run_one_job.sh")
            add(f"ft_{JOB}", "finetune", "A", "phonon", cmd, done, 50)
    if not canon_seen:
        raise SystemExit(f"[gen] canon job {canon_job!r} not in finetune.jobs/seeds")

    # ---- (L)-channel: TMD-family screen (Wave A foundation; Wave B distilled)
    lc = cfg["lchannel"]
    sc = ",".join(str(x) for x in lc["supercell"])
    temps = ",".join(str(x) for x in lc["sscha_temps"])
    found = lc["foundation_model"]

    def lchannel_jobs(model, tag, wave):
        for mat in lc["materials"]:
            base = (f"--name {mat['name']} --formula {mat['formula']} "
                    f"--polytype {mat['polytype']} --a {mat['a']} "
                    f"--thickness {mat['thickness']} --model-type {lc['model_type']} "
                    f"--model {model} --tag {tag} --supercell {sc} "
                    f"--displacement {lc['displacement']} --device {dev}")
            # triage for every material (incl. controls) — phonopy+MLIP only (phonon env)
            tout = f"results/h20/lchannel/{mat['name']}_triage_{tag}.csv"
            add(f"lch_triage_{mat['name']}_{tag}", "lchannel", wave, "phonon",
                f"python scripts/h20/lchannel_sscha.py --mode triage {base} --out {tout}",
                tout, 8)
            # SSCHA only for the known/likely-CDW subset — needs the sscha14 env
            if mat.get("sscha"):
                sout = f"results/h20/lchannel/{mat['name']}_sscha_{tag}.csv"
                cdw = mat.get("cdw_exp_K")
                add(f"lch_sscha_{mat['name']}_{tag}", "lchannel", wave, "sscha14",
                    f"python scripts/h20/lchannel_sscha.py --mode sscha {base} "
                    f"--temperatures {temps} --nconfigs {lc['sscha_nconfigs']} "
                    f"--maxpop {lc['sscha_maxpop']} --cdw-exp-K {cdw} --out {sout}",
                    sout, 90)

    lchannel_jobs(found, "found", "A")
    if cfg["waveB"].get("lchannel_distilled"):
        lchannel_jobs(canon_model, "distilled", "B")

    # ---- E9: κ via MLIP (Wave A foundation; optional Wave B distilled) ------
    e9 = cfg["e9_kappa"]
    e9sc = ",".join(str(x) for x in e9["supercell"])
    e9t = ",".join(str(x) for x in e9["temps"])

    def e9_jobs(model, tag, wave):
        for mat in e9["materials"]:
            cflag = f" --c {mat['c']}" if mat.get("c") else ""
            out = f"results/h20/e9/{mat['name']}_{tag}.json"
            cmd = (f"python scripts/h20/e9_kappa_mlip.py --name {mat['name']} "
                   f"--structure {mat['structure']} --a {mat['a']}{cflag} "
                   f"--model-type {e9['model_type']} --model {model} --tag {tag} "
                   f"--supercell {e9sc} --mesh {e9['mesh']} --temps {e9t} "
                   f"--kappa-exp {mat['kappa_exp_300K']} --device {dev} --out {out}")
            add(f"e9_{mat['name']}_{tag}", "e9", wave, "phonon", cmd, out, 30)

    e9_jobs(e9["foundation_model"], "found", "A")
    if cfg["waveB"].get("e9_distilled"):
        e9_jobs(canon_model, "distilled", "B")

    # ---- E1 "after": full-pool benchmark of the distilled model (Wave B) ----
    if cfg["waveB"].get("e1_after"):
        for s in range(ns):
            out = f"results/h20/e1/distilled_s{s:02d}.csv"
            cmd = (f"python scripts/h20/e1_benchmark.py --calc mace --tag distilled "
                   f"--model-path {canon_model} --pool {pool} --shard {s} "
                   f"--nshards {ns} --mesh {e1['mesh']} --device {dev} --out {out}")
            add(f"e1after_distilled_s{s:02d}", "e1", "B", "phonon", cmd, out + ".done", 15)

    # sort long-first within each wave (overlap long pole with short jobs)
    order = {"A": 0, "B": 1}
    jobs.sort(key=lambda j: (order[j["wave"]], -j["est_min"]))

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")

    from collections import Counter
    byw = Counter((j["wave"], j["group"]) for j in jobs)
    print(f"[gen] wrote {len(jobs)} jobs -> {out.relative_to(ROOT)}")
    for (w, g), n in sorted(byw.items()):
        print(f"   wave {w}  {g:<10} {n:>3}")
    print(f"[gen] canon distilled model -> {canon_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
