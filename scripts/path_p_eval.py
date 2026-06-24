"""Path P step C: does the anharmonic distillation close the large-displacement gap?

Force RMSE of each model against DFT (REF_forces) on the held-out *thermal* test
configs. The harmonic-distilled graphene-FT (M1.1b) should have a large error there
(it only saw small displacements); the Path-P model (fine-tuned on thermal DFT forces)
should match DFT much better.

    conda run -n phonon python scripts/path_p_eval.py --test data/path_p/test.xyz \
        --model harmonic=results/finetune_graphene/ft_graphene.model \
        --model pathP=results/finetune_path_p/ft_path_p.model \
        --model foundation=small
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from ase.io import read

import td_common as tdc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", action="append", default=[],
                    help="name=modelpath (repeatable)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    configs = read(ROOT / a.test if not Path(a.test).is_absolute() else a.test, ":")
    Fref = [np.asarray(at.arrays["REF_forces"]) for at in configs]
    fref_rms = float(np.sqrt(np.mean(np.concatenate([f.ravel() for f in Fref]) ** 2)))
    print(f"# {len(configs)} thermal test configs; <|F_dft|>_rms = {fref_rms*1000:.0f} meV/A")
    print(f"{'model':14s} {'force RMSE vs DFT':>18s}")

    rows = []
    for spec in a.model:
        name, _, mp = spec.partition("=")
        calc = tdc.get_mace_calc(mp, device=a.device)
        errs = []
        for at, fr in zip(configs, Fref):
            at2 = at.copy(); at2.calc = calc
            errs.append(np.sqrt(np.mean((at2.get_forces() - fr) ** 2)))
        rmse = float(np.mean(errs)) * 1000
        rows.append((name, rmse))
        print(f"{name:14s} {rmse:13.1f} meV/A")

    if len(rows) >= 2:
        d = dict(rows)
        if "harmonic" in d and "pathP" in d:
            print(f"\nPath P closed the anharmonic gap: {d['harmonic']:.0f} -> "
                  f"{d['pathP']:.0f} meV/A "
                  f"({100*(d['harmonic']-d['pathP'])/d['harmonic']:.0f}% reduction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
