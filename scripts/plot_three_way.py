"""A3 before/after figure: DFPT vs baseline vs fine-tuned dispersion overlay.

Run in the model's env (phonon-mace for MACE):
    conda run -n phonon-mace python scripts/plot_three_way.py \
        --mp-id mp-22862 --baseline small \
        --ft-model results/finetune_mace/ft_phonon.model --device cpu
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
from phonon_accel.plot import plot_three_way


def _mlip_result(model, atoms0, sc, device):
    calc = get_calculator("mace", device=device, model=model)
    atoms = structures.relax(atoms0.copy(), calc, fmax=1e-4)
    phon = PhononCalculation(atoms, supercell_matrix=sc, displacement=0.03)
    return phon.run_all(calculator=calc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp-id", default="mp-22862")  # NaCl (strong softening)
    ap.add_argument("--baseline", default="small")
    ap.add_argument("--ft-model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref_res, ref_ph = reference.reference_result(args.mp_id)
    atoms0 = reference.reference_atoms(args.mp_id)
    sc = ref_ph.supercell_matrix.tolist()

    base = _mlip_result(args.baseline, atoms0, sc, args.device)
    ft = _mlip_result(args.ft_model, atoms0, sc, args.device)

    out = args.out or f"results/figures/{ref_res.formula}_{args.mp_id}_before_after.png"
    path = plot_three_way(
        ref_res, base, ft, out=out,
        title=f"{ref_res.formula} ({args.mp_id}) — phonon dispersion before/after FC-distillation",
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
