"""E1 — full-band MLIP-vs-DFPT benchmark for ONE model over a shard of the MDR
pool. Idempotent at the material level: appends one CSV row per mp-id and skips
mp-ids already present, so a crashed/re-run shard only does what's missing.

Each (model, shard) is one queue job. Sharding is round-robin over the pool so
every shard gets a balanced mix of cell sizes.

    python scripts/h20/e1_benchmark.py --calc mace --tag mace \
        --pool data/benchmark/e1_pool.txt --shard 0 --nshards 12 \
        --out results/h20/e1/mace_s00.csv --device cuda
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phonon_accel.benchmark import run_material_vs_dfpt   # noqa: E402
from phonon_accel.mlip_calc import get_calculator         # noqa: E402

# keys match phonon_accel.metrics.PhononMetrics.as_row() (+ softening_pct we add)
FIELDS = ["model", "mp_id", "freq_mae", "freq_rmse", "omega_max_pred",
          "omega_max_ref", "omega_max_error", "softening_pct",
          "n_imaginary_pred", "n_imaginary_ref", "asr_residual_pred",
          "cv_error_300k", "s_error_300k", "f_error_300k",
          "n_displacements", "error"]


def load_pool(path: Path) -> list[str]:
    ids = []
    for tok in path.read_text().split():
        tok = tok.strip()
        if tok.startswith("mp-"):
            ids.append(tok)
    return ids


def already_done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    done = set()
    with open(out) as f:
        for row in csv.DictReader(f):
            # a row counts as done whether it succeeded or hard-errored
            done.add(row["mp_id"])
    return done


def make_calc(calc_name: str, model_path: str | None, device: str):
    if calc_name.startswith("mace"):
        model = model_path if model_path else "medium"
        return get_calculator(calc_name, device=device, model=model)
    return get_calculator(calc_name, device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", required=True,
                    help="get_calculator key: mace|mace-omat|sevennet|mattersim|orb|chgnet")
    ap.add_argument("--tag", required=True, help="label for the 'model' column")
    ap.add_argument("--model-path", default=None,
                    help="for mace: a fine-tuned .model file (the 'after' row)")
    ap.add_argument("--pool", default="data/benchmark/e1_pool.txt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--mesh", type=int, default=24)
    ap.add_argument("--displacement", type=float, default=0.03)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pool = load_pool(ROOT / a.pool if not Path(a.pool).is_absolute() else Path(a.pool))
    mine = pool[a.shard::a.nshards]
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(out) + ".done")     # queue done-marker: shard fully finished
    done = already_done(out)
    todo = [m for m in mine if m not in done]
    print(f"[e1:{a.tag}] shard {a.shard}/{a.nshards}: {len(mine)} assigned, "
          f"{len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        marker.write_text("complete\n")
        print(f"[e1:{a.tag}] shard already complete", flush=True)
        return 0

    calc = make_calc(a.calc, a.model_path, a.device)
    mesh = (a.mesh,) * 3
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, mp in enumerate(todo, 1):
            t0 = time.perf_counter()
            r = run_material_vs_dfpt(a.tag, mp, calc=calc, device=a.device,
                                     displacement=a.displacement, mesh=mesh)
            # softening % of omega_max (the headline failure-mode number)
            if not r.get("error") and r.get("omega_max_ref"):
                r["softening_pct"] = 100.0 * r["omega_max_error"] / r["omega_max_ref"]
            row = {k: r.get(k, "") for k in FIELDS}
            row["model"] = a.tag
            row["mp_id"] = mp
            w.writerow(row)
            fh.flush()
            if r.get("error"):
                print(f"  [{i}/{len(todo)}] {mp:<11} ERR {r['error'][:60]}", flush=True)
            else:
                print(f"  [{i}/{len(todo)}] {mp:<11} "
                      f"MAE={r.get('freq_mae', float('nan')):.3f} THz  "
                      f"soft={100*r.get('omega_max_error',0)/max(r.get('omega_max_ref',1),1e-9):+5.1f}%  "
                      f"imag={r.get('n_imaginary_pred','?')}  "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    marker.write_text("complete\n")
    print(f"[e1:{a.tag}] shard {a.shard} wrote {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
