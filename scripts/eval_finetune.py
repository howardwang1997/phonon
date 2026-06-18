"""A3 evaluation: phonon accuracy BEFORE vs AFTER FC-distillation fine-tuning.

Re-runs the full-band DFPT benchmark with (a) the baseline foundation model
and (b) the fine-tuned model, over the training materials AND the held-out
materials, so we can see both the in-domain fix and generalization.

Run in the env that has the model (phonon-mace for MACE):
    conda run -n phonon-mace python scripts/eval_finetune.py \
        --baseline small --ft-model results/finetune_mace/ft_phonon.model \
        --device cpu --out results/finetune_eval.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phonon_accel.benchmark import run_material_vs_dfpt
from phonon_accel.mlip_calc import get_calculator

# Materials actually used to fine-tune (5) vs held-out (generalization).
# Held-out split spans both in-domain elements (ZnO) and entirely new
# elements (Ga, Ti, Li, B, C, As, P, Ca, In) — a strict generalization test.
TRAIN = ["mp-149", "mp-1265", "mp-22862", "mp-661", "mp-9946"]
HOLDOUT = ["mp-1986", "mp-7140", "mp-2172", "mp-804", "mp-1138", "mp-2605", "mp-20351", "mp-984", "mp-390"]


def _rows_for(calc, tag, device, relax=True):
    out = []
    for split, mats in (("train", TRAIN), ("holdout", HOLDOUT)):
        for mp in mats:
            r = run_material_vs_dfpt("mace", mp, calc=calc, device=device, relax=relax)
            if r.get("error"):
                print(f"  [{tag}] {split} {mp}: ERR {r['error']}")
                continue
            soft = 100 * r["omega_max_error"] / r["omega_max_ref"]
            r.update(variant=tag, split=split, softening_pct=soft)
            out.append(r)
            print(f"  [{tag:>8}] {split:<7} {mp:<9} MAE={r['freq_mae']:.3f} THz  "
                  f"soft={soft:+5.1f}%  imag={r['n_imaginary_pred']}")
    return out


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="small")
    ap.add_argument("--ft-model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train", nargs="+", default=None, help="override train mp-ids")
    ap.add_argument("--holdout", nargs="+", default=None, help="override holdout mp-ids")
    ap.add_argument("--no-relax", action="store_true",
                    help="evaluate at the DFT geometry (isolates FC quality from relaxation drift)")
    ap.add_argument("--out", default="results/finetune_eval.csv")
    args = ap.parse_args()
    relax = not args.no_relax
    global TRAIN, HOLDOUT
    if args.train:
        TRAIN = args.train
    if args.holdout:
        HOLDOUT = args.holdout

    print(f"== baseline == (relax={relax})")
    base_calc = get_calculator("mace", device=args.device, model=args.baseline)
    rows = _rows_for(base_calc, "baseline", args.device, relax=relax)

    print("== fine-tuned ==")
    ft_calc = get_calculator("mace", device=args.device, model=args.ft_model)
    rows += _rows_for(ft_calc, "finetuned", args.device, relax=relax)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}\n")

    # summary: mean |MAE| and mean softening by variant x split
    g = df.groupby(["variant", "split"]).agg(
        n=("freq_mae", "size"),
        freq_mae=("freq_mae", "mean"),
        softening=("softening_pct", "mean"),
        imag=("n_imaginary_pred", "sum"),
    )
    print(g.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
