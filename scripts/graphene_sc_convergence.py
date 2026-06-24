"""M1.1 convergence check: how the graphene Gamma-E2g / K-A1' frequencies and
the top-branch kink strengths move with supercell size, for one model.

A Kohn anomaly is non-analytic, so a finite-displacement supercell Fourier-
truncates the cusp; this separates that truncation from genuine MLIP smoothing.
Both models share whichever supercell we settle on, so their *difference* is the
MLIP -- this run tells us how converged the *absolute* numbers are.

    conda run -n phonon python scripts/graphene_sc_convergence.py \
        --model medium --sizes 3,5,7,9
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from ase.io import read

import anomaly_locate as al
import td_common as tdc

CM = 33.35641


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--structure", default="data/td_phonon/graphene.xyz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sizes", default="3,5,7,9")
    ap.add_argument("--npoints", type=int, default=201)
    ap.add_argument("--csv", default="", help="append rows to this CSV path")
    ap.add_argument("--label", default="", help="model label for the CSV")
    a = ap.parse_args()

    calc = tdc.get_mace_calc(a.model, device=a.device)
    at0 = read(ROOT / a.structure)
    at, info = tdc.relax_monolayer(at0, calc)
    print(f"model={a.model}  a={info['a']:.4f} A")
    print(f"{'sc':>4s} {'natoms':>7s} {'wGamma':>9s} {'wK':>9s} "
          f"{'kinkGamma':>10s} {'kinkK':>8s} {'minfreq':>9s}")

    csv_rows = []
    label = a.label or a.model
    for n in [int(x) for x in a.sizes.split(",")]:
        disp = tdc.dispersion(at, calc, supercell=(n, n, 1),
                              displacement=0.03, path="MGKM", npoints=a.npoints)
        dist, freq = disp["distances"], disp["frequencies"]
        lp, labs = disp["label_positions"], disp["labels"]
        wG = al.branch_freq_at_label(dist, freq, lp, labs, r"$\Gamma$") * CM
        wK = al.branch_freq_at_label(dist, freq, lp, labs, "K") * CM
        kinks = {k["label"]: k for k in al.high_sym_kinks(dist, freq, lp, labs)}
        kG = kinks[r"$\Gamma$"]["kink_strength"]
        kK = kinks["K"]["kink_strength"]
        print(f"{n:>3d}x{n:<1d} {n*n*2:>7d} {wG:8.1f}  {wK:8.1f}  "
              f"{kG:9.1f}  {kK:7.1f}  {freq.min():8.2f}", flush=True)
        csv_rows.append(f"{label},{n},{n*n*2},{wG:.2f},{wK:.2f},{kG:.3f},{kK:.3f},{freq.min():.3f}")

    if a.csv:
        p = ROOT / a.csv
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with open(p, "a") as f:
            if new:
                f.write("model,sc,natoms,w_gamma_cm,w_k_cm,kink_gamma,kink_k,minfreq_thz\n")
            f.write("\n".join(csv_rows) + "\n")
        print("appended", len(csv_rows), "rows ->", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
