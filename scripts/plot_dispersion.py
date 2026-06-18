"""Plot an MLIP-vs-DFPT phonon dispersion overlay for one MDR material.

Usage:
    conda run -n phonon python scripts/plot_dispersion.py \
        --model mattersim --mp-id mp-149 --device cpu \
        --out results/figures/Si_mattersim_vs_dfpt.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phonon_accel import reference, structures
from phonon_accel.mlip_calc import get_calculator
from phonon_accel.phonons import PhononCalculation
from phonon_accel.plot import plot_dispersion_comparison


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mattersim")
    ap.add_argument("--mp-id", default="mp-149")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref_res, ref_ph = reference.reference_result(args.mp_id)
    atoms = reference.reference_atoms(args.mp_id)
    sc = ref_ph.supercell_matrix.tolist()

    calc = get_calculator(args.model, device=args.device)
    atoms = structures.relax(atoms, calc, fmax=1e-4)
    phon = PhononCalculation(atoms, supercell_matrix=sc, displacement=0.03)
    pred = phon.run_all(calculator=calc)

    out = args.out or f"results/figures/{ref_res.formula}_{args.model}_vs_dfpt.png"
    path = plot_dispersion_comparison(
        pred, ref_res, out=out,
        title=f"{ref_res.formula} ({args.mp_id}) — {args.model} vs DFPT",
        pred_label=args.model, ref_label="DFPT (MDR)",
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
