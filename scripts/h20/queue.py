"""Work-stealing job queue for the 8×H20 campaign.

The manifest (results/h20/jobs.jsonl) is the shared queue. Each GPU worker
repeatedly calls ``queue.py claim`` to atomically grab the next runnable job;
the atomic primitive is ``os.mkdir(claims/<id>)`` (POSIX-atomic, NFS-safe). A
job is *done* when its ``done`` marker file exists; *claimed* when its claim dir
exists. This gives us, for free:

  * no GPU idles while any job is unclaimed (pure pull model);
  * resumability — a re-run skips jobs whose ``done`` marker exists;
  * crash safety — ``release-stale`` drops claims whose ``done`` is absent so a
    fresh campaign retries jobs that were claimed by a process that died.

Jobs carry a ``wave`` (A or B). Wave B only becomes claimable after every Wave A
job is done (Wave B re-runs depend on Wave A's distilled model). Within a wave
the manifest is pre-sorted long-job-first so the long pole overlaps the short
jobs instead of stranding 7 idle GPUs at the end.

Subcommands
-----------
    claim   --wave A   -> atomically claim+print the next job as TSV, or exit 1
    release-stale      -> rm claim dirs whose done-marker is absent
    status             -> per-group done/total + running, to stdout
    wave-ready --wave B -> exit 0 iff all Wave A jobs are done (gate)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H20 = ROOT / "results" / "h20"
MANIFEST = H20 / "jobs.jsonl"
CLAIMS = H20 / "claims"


def load_jobs(manifest: Path) -> list[dict]:
    jobs = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                jobs.append(json.loads(line))
    return jobs


def is_done(job: dict) -> bool:
    return bool(job.get("done")) and (ROOT / job["done"]).exists()


def all_wave_done(jobs: list[dict], wave: str) -> bool:
    return all(is_done(j) for j in jobs if j.get("wave") == wave)


def cmd_claim(args) -> int:
    jobs = load_jobs(Path(args.manifest))
    CLAIMS.mkdir(parents=True, exist_ok=True)
    # Wave gate: Wave B not claimable until all Wave A jobs are done.
    if args.wave == "B" and not all_wave_done(jobs, "A"):
        return 2  # not ready yet
    for job in jobs:
        if job.get("wave") != args.wave:
            continue
        if is_done(job):
            continue
        claim = CLAIMS / job["id"]
        try:
            claim.mkdir()  # atomic: only one worker wins
        except FileExistsError:
            continue  # someone else has it (or it failed earlier this run)
        # write provenance for `status`/debugging
        (claim / "info").write_text(
            f"gpu={os.environ.get('CLAIM_GPU','?')} pid={os.getpid()}\n"
            f"group={job.get('group')} env={job.get('env')}\n"
        )
        # emit fields the worker needs, TSV (cmd is last; may contain spaces)
        sys.stdout.write(
            "\t".join([job["id"], job.get("env", "phonon"),
                       "1" if job.get("gpu", True) else "0",
                       job.get("group", ""), str(job.get("done", "")),
                       job["cmd"]]) + "\n")
        return 0
    return 1  # nothing left in this wave


def cmd_release_stale(args) -> int:
    jobs = {j["id"]: j for j in load_jobs(Path(args.manifest))}
    if not CLAIMS.exists():
        return 0
    freed = 0
    for claim in CLAIMS.iterdir():
        if not claim.is_dir():
            continue
        job = jobs.get(claim.name)
        if job is None or not is_done(job):
            # claimed but never finished -> release so this run retries it
            for p in claim.iterdir():
                p.unlink()
            claim.rmdir()
            freed += 1
    print(f"[queue] released {freed} stale claim(s)")
    return 0


def cmd_wave_ready(args) -> int:
    jobs = load_jobs(Path(args.manifest))
    return 0 if all_wave_done(jobs, "A") else 1


def cmd_status(args) -> int:
    jobs = load_jobs(Path(args.manifest))
    from collections import defaultdict
    tot = defaultdict(int)
    done = defaultdict(int)
    for j in jobs:
        key = (j.get("wave", "?"), j.get("group", "?"))
        tot[key] += 1
        if is_done(j):
            done[key] += 1
    running = []
    if CLAIMS.exists():
        for claim in sorted(CLAIMS.iterdir()):
            jid = claim.name
            j = next((x for x in jobs if x["id"] == jid), None)
            if j is not None and not is_done(j):
                running.append(jid)
    print(f"{'WAVE/GROUP':<22} {'DONE':>6} {'TOTAL':>6}")
    grand_d = grand_t = 0
    for key in sorted(tot):
        d, t = done[key], tot[key]
        grand_d += d
        grand_t += t
        bar = "#" * int(20 * d / t) if t else ""
        print(f"{key[0]+'/'+key[1]:<22} {d:>6} {t:>6}  {bar}")
    print(f"{'TOTAL':<22} {grand_d:>6} {grand_t:>6}")
    if running:
        print(f"\nrunning now ({len(running)}): {' '.join(running)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="sub", required=True)
    for name in ("claim", "release-stale", "status", "wave-ready"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", default=str(MANIFEST))
        if name in ("claim", "wave-ready"):
            p.add_argument("--wave", default="A", choices=["A", "B"])
    args = ap.parse_args()
    return {
        "claim": cmd_claim,
        "release-stale": cmd_release_stale,
        "status": cmd_status,
        "wave-ready": cmd_wave_ready,
    }[args.sub](args)


if __name__ == "__main__":
    raise SystemExit(main())
