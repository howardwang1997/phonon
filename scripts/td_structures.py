"""M0: build, relax, and cache the 2D monolayers for the TD-phonon study.

Builds graphene / MoS2 / NbSe2 at literature geometries, relaxes each in-plane
with the chosen MLIP (vacuum held fixed), writes extxyz to data/td_phonon/, and
runs a graphene smoke test (single-point forces + one Gamma-M-K-Gamma dispersion)
so M1.1 starts from a verified pipeline.

    conda run -n phonon python scripts/td_structures.py --model medium --device cuda
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
from ase.io import write

import td_common as tdc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium", help="foundation tag or model path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--which", default="graphene,mos2,nbse2")
    ap.add_argument("--outdir", default="data/td_phonon")
    ap.add_argument("--no-smoke", action="store_true")
    a = ap.parse_args()

    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"loading MACE calculator (model={a.model}, device={a.device}) ...", flush=True)
    calc = tdc.get_mace_calc(a.model, device=a.device)

    relaxed = {}
    for name in [s.strip() for s in a.which.split(",") if s.strip()]:
        at0 = tdc.build_monolayer(name)
        at, info = tdc.relax_monolayer(at0, calc)
        relaxed[name] = at
        out = outdir / f"{name}.xyz"
        write(out, at)
        print(
            f"  {name:9s} a={info['a']:.4f} A  thickness={info['thickness']:.3f} A  "
            f"fmax={info['fmax']:.1e}  conv={info['converged']}  "
            f"({info['nsteps']} steps, {len(at)} atoms) -> {out.relative_to(ROOT)}",
            flush=True,
        )

    if a.no_smoke or "graphene" not in relaxed:
        return 0

    print("smoke test: graphene single-point forces + Gamma-M-K-Gamma dispersion ...",
          flush=True)
    g = relaxed["graphene"].copy()
    g.calc = calc
    f = g.get_forces()
    print(f"  single-point: max|F| on relaxed cell = {np.abs(f).max():.2e} eV/A", flush=True)

    disp = tdc.dispersion(relaxed["graphene"], calc, supercell=(4, 4, 1),
                          displacement=0.03, path="GMKG", npoints=101)
    freq = disp["frequencies"]
    cm = 33.35641
    i_top = np.argmin(np.abs(disp["distances"] - disp["label_positions"][0]))  # Gamma
    print(f"  dispersion ok: {disp['n_displacements']} displacements, "
          f"{freq.shape[0]} q-points x {freq.shape[1]} branches", flush=True)
    print(f"  graphene top optical at Gamma = {freq[i_top].max():.2f} THz "
          f"= {freq[i_top].max()*cm:.0f} cm^-1  (lit ~1600)", flush=True)
    print(f"  min frequency on path = {freq.min():.2f} THz "
          f"(slightly imaginary ZA near Gamma is expected)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
