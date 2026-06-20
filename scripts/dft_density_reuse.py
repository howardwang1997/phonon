"""Line B speedup factor #1: cross-displacement charge-density reuse.

Displaced supercells in a finite-displacement phonon differ by ~0.01 Angstrom, so
the equilibrium converged charge density is a near-perfect SCF starting guess.
This measures the SCF-iteration and wall-clock reduction of starting each
displaced SCF from the equilibrium density (`startingpot='file'`) vs from scratch
(`startingpot='atomic'`). Drives pw.x directly for full control of outdir/prefix.

    python scripts/dft_density_reuse.py --supercell 2 --ecutwfc 40 --kpts 3 --nproc 24
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ase.build import bulk
from ase.io.espresso import write_espresso_in

HOME = Path.home()
DFT = HOME / "miniconda3" / "envs" / "dft" / "bin"


def run_pw(atoms, workdir, pseudos, ecutwfc, ecutrho, kpts, nproc, startingpot, conv_thr=1e-8):
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    outdir = workdir / "out"
    pwi = workdir / "pw.in"; pwo = workdir / "pw.out"
    input_data = {
        "control": {"calculation": "scf", "tprnfor": True, "disk_io": "low",
                    "outdir": str(outdir), "prefix": "ph", "verbosity": "high"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "gaussian", "degauss": 0.01},
        "electrons": {"conv_thr": conv_thr, "mixing_beta": 0.7, "startingpot": startingpot},
    }
    with open(pwi, "w") as f:
        write_espresso_in(f, atoms, input_data=input_data, pseudopotentials=pseudos,
                          kpts=(kpts, kpts, kpts))
    cmd = [str(DFT / "mpirun"), "--allow-run-as-root", "-np", str(nproc), str(DFT / "pw.x"),
           "-in", str(pwi)]
    t0 = time.perf_counter()
    with open(pwo, "w") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, ESPRESSO_PSEUDO=str(HOME / "pseudo")))
    wall = time.perf_counter() - t0
    txt = pwo.read_text()
    m = re.search(r"convergence has been achieved in\s+(\d+)\s+iterations", txt)
    iters = int(m.group(1)) if m else -1
    return wall, iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supercell", type=int, default=2)
    ap.add_argument("--ecutwfc", type=float, default=40.0)
    ap.add_argument("--ecutrho", type=float, default=320.0)
    ap.add_argument("--kpts", type=int, default=3)
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--disp", type=float, default=0.01)
    args = ap.parse_args()
    pseudos = {"Si": "Si.upf"}
    n = args.supercell
    eq = bulk("Si", "diamond", a=5.43) * (n, n, n)
    base = Path("/tmp/dre")
    if base.exists():
        shutil.rmtree(base)

    # 1. equilibrium SCF (produces the reusable charge density)
    w_eq = base / "eq"
    t_eq, i_eq = run_pw(eq, w_eq, pseudos, args.ecutwfc, args.ecutrho, args.kpts, args.nproc, "atomic")
    print(f"equilibrium: {i_eq} iters, {t_eq:.1f}s", flush=True)

    # displaced structure (move atom 0 by disp along x)
    disp = eq.copy(); disp.positions[0, 0] += args.disp

    # 2a. displaced from scratch
    t_s, i_s = run_pw(disp, base / "scratch", pseudos, args.ecutwfc, args.ecutrho, args.kpts, args.nproc, "atomic")
    print(f"displaced from scratch:  {i_s} iters, {t_s:.1f}s", flush=True)

    # 2b. displaced reusing equilibrium density (copy eq charge-density into reuse outdir)
    w_re = base / "reuse"; (w_re / "out").mkdir(parents=True, exist_ok=True)
    shutil.copytree(w_eq / "out" / "ph.save", w_re / "out" / "ph.save")
    t_r, i_r = run_pw(disp, w_re, pseudos, args.ecutwfc, args.ecutrho, args.kpts, args.nproc, "file")
    print(f"displaced reuse-density: {i_r} iters, {t_r:.1f}s", flush=True)

    print("\n=== density-reuse speedup (displaced SCF) ===")
    print(f"  iterations: {i_s} -> {i_r}  ({i_s/max(i_r,1):.2f}x fewer)")
    print(f"  wall:       {t_s:.1f}s -> {t_r:.1f}s  ({t_s/max(t_r,1e-9):.2f}x faster)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
