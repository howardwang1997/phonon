"""Baseline (un-fine-tuned) foundation-MLIP phonon benchmark across the full
cached MDR pool -> the "before fine-tuning" reference for the breadth story.
Inference-only, fits 8 GB; runs on the RTX 2060. Writes incrementally so a
partial run is never lost.

    python scripts/run_baseline_benchmark.py --model small  --out results/baseline/small.csv
    python scripts/run_baseline_benchmark.py --model medium --out results/baseline/medium.csv
    python scripts/run_baseline_benchmark.py --model mace-omat --out results/baseline/omat.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonon_accel.benchmark import run_material_vs_dfpt
from phonon_accel.mlip_calc import get_calculator


def cached_pool():
    return sorted(p.name.split("_")[0] for p in (ROOT / "data" / "benchmark" / "mdr").glob("*_phonopy_params.yaml"))


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="small|medium|mace-omat|<path>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--materials", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    relax = not args.no_relax
    mats = args.materials or cached_pool()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # resume: skip materials already in the CSV
    done = set()
    if out.exists():
        try:
            done = set(pd.read_csv(out)["mp_id"])
        except Exception:  # noqa: BLE001
            pass

    name = "mace-omat" if args.model == "mace-omat" else "mace"
    kw = {} if args.model == "mace-omat" else {"model": args.model}
    calc = get_calculator(name, device=args.device, **kw)

    rows = []
    if out.exists() and done:
        rows = pd.read_csv(out).to_dict("records")
    for i, mp in enumerate(mats):
        if mp in done:
            continue
        r = run_material_vs_dfpt("mace", mp, calc=calc, device=args.device, relax=relax)
        if r.get("error"):
            print(f"  [{args.model}] {mp}: ERR {r['error']}", flush=True)
            r = {"mp_id": mp, "error": r["error"]}
        else:
            soft = 100 * r["omega_max_error"] / r["omega_max_ref"]
            r.update(model=args.model, softening_pct=soft)
            print(f"  [{args.model}] {mp:<11} ({i+1}/{len(mats)}) MAE={r['freq_mae']:.3f} THz "
                  f"soft={soft:+5.1f}% imag={r['n_imaginary_pred']}", flush=True)
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)  # incremental save

    df = pd.DataFrame(rows)
    ok = df[df.get("error").isna()] if "error" in df else df
    print(f"\n[{args.model}] {len(ok)}/{len(mats)} ok, mean MAE={ok['freq_mae'].mean():.3f} THz, "
          f"total imag={int(ok['n_imaginary_pred'].sum())}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
