"""M1.1: harmonic (0 K) phonon dispersion of a 2D monolayer with one MLIP.

Relaxes the structure with the given model (its own equilibrium), runs phonopy
finite-displacement forces on a supercell, and saves the Gamma-M-K-Gamma bands
to npz. Run once per model (foundation vs FC-distilled) to test whether the
foundation MLIP smooths graphene's Kohn cusp and whether distillation recovers
it (Gate #1).

    conda run -n phonon python scripts/harmonic_dispersion_2d.py \
        --structure data/td_phonon/graphene.xyz --model medium --tag graphene_base
    conda run -n phonon python scripts/harmonic_dispersion_2d.py \
        --structure data/td_phonon/graphene.xyz \
        --model results/finetune_mace/ft_phonon.model --tag graphene_ft
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from ase.io import read

import td_common as tdc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--model", required=True, help="foundation tag or model path")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--supercell", default="5,5,1")
    ap.add_argument("--disp", type=float, default=0.03)
    ap.add_argument("--npoints", type=int, default=201)
    ap.add_argument("--path", default="GMKG")
    ap.add_argument("--outdir", default="results/td_phonon")
    a = ap.parse_args()

    sc = tuple(int(x) for x in a.supercell.split(","))
    at0 = read(ROOT / a.structure)
    print(f"[{a.tag}] loaded {a.structure} ({len(at0)} atoms); "
          f"loading model={a.model} on {a.device} ...", flush=True)
    calc = tdc.get_mace_calc(a.model, device=a.device)

    t0 = time.perf_counter()
    at, info = tdc.relax_monolayer(at0, calc)
    print(f"[{a.tag}] relaxed: a={info['a']:.4f} A  thickness={info['thickness']:.3f} A "
          f"fmax={info['fmax']:.1e} conv={info['converged']}", flush=True)

    disp = tdc.dispersion(at, calc, supercell=sc, displacement=a.disp,
                          path=a.path, npoints=a.npoints)
    disp["model"] = np.array(str(a.model))
    disp["tag"] = np.array(str(a.tag))
    disp["a_relaxed"] = np.array(info["a"])
    disp["structure"] = np.array(str(a.structure))

    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"disp_{a.tag}.npz"
    np.savez(out, **disp)

    freq = disp["frequencies"]
    cm = 33.35641
    gi = int(np.argmin(np.abs(disp["distances"] - disp["label_positions"][0])))
    ki = int(np.argmin(np.abs(disp["distances"] - disp["label_positions"][2])))
    print(f"[{a.tag}] {disp['n_displacements']} displacements, "
          f"{freq.shape[0]}x{freq.shape[1]} bands in {time.perf_counter()-t0:.1f}s",
          flush=True)
    print(f"[{a.tag}] top optical: Gamma={freq[gi].max():.2f} THz "
          f"({freq[gi].max()*cm:.0f} cm^-1)  K={freq[ki].max():.2f} THz "
          f"({freq[ki].max()*cm:.0f} cm^-1)", flush=True)
    print(f"[{a.tag}] min freq on path = {freq.min():.2f} THz -> {out.relative_to(ROOT)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
