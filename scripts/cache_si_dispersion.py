"""Compute Si (mp-149) phonon bands for DFPT / MACE-baseline / FC-distilled and
cache them to an npz, so the Fig-2 composite can render the dispersion panel with
the shared style without re-running MLIP calculations.

    conda run -n phonon-mace python scripts/cache_si_dispersion.py \
        --ft-model results/finetune_mace/ft_phonon.model
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from phonon_accel import reference, structures
from phonon_accel.mlip_calc import get_calculator
from phonon_accel.phonons import PhononCalculation


def _mlip_result(model, atoms0, sc, device):
    calc = get_calculator("mace", device=device, model=model)
    atoms = structures.relax(atoms0.copy(), calc, fmax=1e-4)
    phon = PhononCalculation(atoms, supercell_matrix=sc, displacement=0.03)
    return phon.run_all(calculator=calc)


def _pack(prefix, res, d):
    d[f"{prefix}_dist"] = np.array([np.asarray(x) for x in res.band_distances], dtype=object)
    d[f"{prefix}_freq"] = np.array([np.asarray(x) for x in res.band_frequencies], dtype=object)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp-id", default="mp-149")
    ap.add_argument("--baseline", default="small")
    ap.add_argument("--ft-model", default="results/finetune_mace/ft_phonon.model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/figures/data/si_dispersion.npz")
    args = ap.parse_args()

    ref_res, ref_ph = reference.reference_result(args.mp_id)
    atoms0 = reference.reference_atoms(args.mp_id)
    sc = ref_ph.supercell_matrix.tolist()

    base = _mlip_result(args.baseline, atoms0, sc, args.device)
    ft = _mlip_result(args.ft_model, atoms0, sc, args.device)

    d = {}
    _pack("dfpt", ref_res, d)
    _pack("base", base, d)
    _pack("ft", ft, d)
    boundaries = [np.asarray(x)[0] for x in ref_res.band_distances]
    boundaries.append(np.asarray(ref_res.band_distances[-1])[-1])
    d["boundaries"] = np.asarray(boundaries)
    d["labels"] = np.array(ref_res.band_labels if ref_res.band_labels else [], dtype=object)
    d["formula"] = np.array(ref_res.formula)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    print("wrote", out, "| segments:", len(d["dfpt_dist"]),
          "| DFPT ωmax:", round(max(np.asarray(f).max() for f in d["dfpt_freq"]), 2),
          "base ωmax:", round(max(np.asarray(f).max() for f in d["base_freq"]), 2),
          "ft ωmax:", round(max(np.asarray(f).max() for f in d["ft_freq"]), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
