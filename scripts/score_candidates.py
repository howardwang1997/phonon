"""Predict-only uncertainty score for candidate materials, for the active-
learning loop. Runs the CURRENT fine-tuned MLIP on each candidate's (DFT-relaxed)
cell -- NO DFPT frequencies used -- and reports an instability/inconsistency
signal the acquisition function ranks on:

    uncertainty = n_imaginary_mesh + LAMBDA * asr_residual(THz)

The model is least trustworthy where it predicts spurious soft/imaginary modes
or violates the acoustic sum rule. Small supercell/mesh keeps scoring fast
(ranking only; absolute accuracy not needed).

    python scripts/score_candidates.py --model <ft.model> --candidates mp-a mp-b ... --out scores.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonon_accel import reference
from phonon_accel.mlip_calc import get_calculator
from phonon_accel.phonons import PhononCalculation

LAMBDA = 5.0  # weight of ASR residual (THz) relative to an imaginary-mode count


def score_one(mp, calc, supercell, mesh, disp):
    atoms = reference.reference_atoms(mp)  # DFT cell; freqs NOT used
    phon = PhononCalculation(atoms, supercell_matrix=supercell, displacement=disp)
    res = phon.run_all(calculator=calc, mesh=mesh)
    nimag = int(res.n_imaginary_mesh)
    asr = float(res.asr_residual) if res.asr_residual == res.asr_residual else 0.0
    return nimag, asr, nimag + LAMBDA * abs(asr)


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--supercell", type=int, default=2)
    ap.add_argument("--mesh", type=int, default=12)
    ap.add_argument("--disp", type=float, default=0.03)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sc = [[args.supercell, 0, 0], [0, args.supercell, 0], [0, 0, args.supercell]]
    mesh = (args.mesh,) * 3
    calc = get_calculator("mace", device=args.device, model=args.model)

    rows = []
    for mp in args.candidates:
        try:
            nimag, asr, u = score_one(mp, calc, sc, mesh, args.disp)
            rows.append(dict(mp_id=mp, n_imaginary=nimag, asr_residual=asr, uncertainty=u))
            print(f"  {mp:<12} imag={nimag:<5} asr={asr:+.3f}  U={u:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(mp_id=mp, error=f"{type(exc).__name__}: {exc}"))
            print(f"  {mp:<12} ERR {exc}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
