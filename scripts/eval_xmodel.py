"""A1 cross-model eval: phonon accuracy BEFORE vs AFTER FC distillation for a
non-MACE foundation model (SevenNet / MatterSim). Mirrors eval_finetune.py but
model-agnostic via --model-type. Run in the env with that model (e.g. xmodel):

    python scripts/eval_xmodel.py --model-type sevennet --baseline 7net-0 \
        --ft-model /tmp/a1_sevennet_run/checkpoint_best.pth --device cuda --no-relax \
        --train mp-149 ... --holdout mp-7140 ... --out results/xmodel_sevennet_eval.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phonon_accel.benchmark import run_material_vs_dfpt
from phonon_accel.mlip_calc import get_calculator


def rows_for(calc, mtype, tag, train, holdout, device, relax):
    out = []
    for split, mats in (("train", train), ("holdout", holdout)):
        for mp in mats:
            try:
                r = run_material_vs_dfpt(mtype, mp, calc=calc, device=device, relax=relax)
            except Exception as e:
                print(f"  [{tag}] {split} {mp}: EXC {str(e)[:50]}"); continue
            if r.get("error"):
                print(f"  [{tag}] {split} {mp}: ERR {r['error']}"); continue
            r.update(variant=tag, split=split,
                     softening_pct=100 * r["omega_max_error"] / r["omega_max_ref"])
            out.append(r)
            print(f"  [{tag:>9}] {split:<7} {mp:<9} MAE={r['freq_mae']:.3f} THz  "
                  f"soft={r['softening_pct']:+5.1f}%")
    return out


def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="sevennet")
    ap.add_argument("--baseline", default="7net-0")
    ap.add_argument("--ft-model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--holdout", nargs="+", required=True)
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--ft-only", action="store_true")
    ap.add_argument("--out", default="results/xmodel_finetune_eval.csv")
    a = ap.parse_args()
    relax = not a.no_relax

    rows = []
    if not a.ft_only:
        print(f"== baseline {a.model_type}:{a.baseline} ==")
        rows += rows_for(get_calculator(a.model_type, device=a.device, model=a.baseline),
                         a.model_type, "baseline", a.train, a.holdout, a.device, relax)
    print(f"== fine-tuned {a.model_type} ==")
    rows += rows_for(get_calculator(a.model_type, device=a.device, model=a.ft_model),
                     a.model_type, "finetuned", a.train, a.holdout, a.device, relax)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}\n")
    print(df.groupby(["variant", "split"]).agg(
        n=("freq_mae", "size"), freq_mae=("freq_mae", "mean"),
        softening=("softening_pct", "mean")).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
