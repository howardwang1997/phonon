from __future__ import annotations

import argparse
import numpy as np

from .structure import Structure
from .pipeline import PhononPipeline
from .scheduler import MultiGPUExecutor
from .io.poscar import read_poscar
from .phonon import freqs_to_thz


def _build_prim(args):
    if args.poscar:
        return read_poscar(args.poscar)
    a = args.a
    lattice = a * np.eye(3)
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    return Structure(lattice, np.vstack([fcc, fcc + 0.25]), ["Si"] * 8)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="phonongpu", description="GPU phonon pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run finite-displacement phonon calculation")
    pr.add_argument("--poscar", help="primitive POSCAR path (default: demo diamond Si 8-atom)")
    pr.add_argument("--a", type=float, default=5.43, help="cubic lattice const for demo cell")
    pr.add_argument("--supercell", type=int, default=2, help="diagonal supercell factor")
    pr.add_argument("--backend", default="analytic", help="analytic|qe-gpu|vasp-gpu")
    pr.add_argument("--distance", type=float, default=0.01)
    pr.add_argument("--num-gpus", type=int, default=None)
    pr.add_argument("--device", default="auto")
    pr.add_argument("--gamma", action="store_true", help="print Gamma frequencies")
    pr.add_argument("--dos-mesh", type=int, default=0, help="DOS q-mesh (0=skip)")

    bm = sub.add_parser("benchmark", help="CPU vs GPU solver benchmark")
    bm.add_argument("--nq", type=int, default=8000)
    bm.add_argument("--sc", type=int, default=2)
    bm.add_argument("--prim-reps", type=int, default=1)

    args = ap.parse_args(argv)

    if args.cmd == "benchmark":
        from .benchmark import main as bmain
        bmain(["--nq", str(args.nq), "--sc", str(args.sc), "--prim-reps", str(args.prim_reps)])
        return

    prim = _build_prim(args)
    executor = MultiGPUExecutor(num_gpus=args.num_gpus) if args.num_gpus else None
    pipe = PhononPipeline(prim, supercell_scale=args.supercell, backend=args.backend,
                          executor=executor, distance=args.distance, device=args.device)
    result = pipe.run()
    if args.gamma:
        g = np.sort(result.gamma_frequencies)
        print("Gamma frequencies (sqrt(eV/Å²/amu)):", np.round(g, 5))
        print("Gamma frequencies (THz)           :", np.round(freqs_to_thz(g), 4))
    if args.dos_mesh > 0:
        grid, dos, _ = result.dos.dos(mesh=args.dos_mesh, device=args.device)
        print(f"DOS computed on {args.dos_mesh}^3 mesh; peak={dos.max():.3f}")


if __name__ == "__main__":
    main()
