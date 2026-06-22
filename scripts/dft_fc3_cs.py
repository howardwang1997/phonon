"""Compressed-sensing fc3 (Option B): converged third-order force constants from
a SMALL number of random-displacement DFT supercells + symfc regression fit,
optionally with a real-space cutoff. Turns the systematic finite-displacement
fc3 (hundreds of configs) into ~tens — feasible on H20 CPU when the systematic
route is ~days.

    python scripts/dft_fc3_cs.py --material Si --sc 3 --rd 40 --cutoff 4.5 \
        --nproc 24 --workers 2

Refs: phono3py --rd + symfc (Togo JPSJ 92, 012001); compressive-sensing lattice
dynamics (Zhou et al., PRB Mater. 2019). The reference stays pure-DFT (no MLP).
"""
from __future__ import annotations

import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms
    from ase import Atoms
    from ase.build import bulk
    sys.path.insert(0, str(ROOT / "scripts"))
    from dft_fc3 import make_espresso, pseudos_for, PSEUDO  # reuse engine setup

    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Si")
    ap.add_argument("--sc", type=int, default=3)
    ap.add_argument("--rd", type=int, default=40, help="number of random-displacement supercells")
    ap.add_argument("--cutoff", type=float, default=0.0, help="fc3 real-space cutoff (Ang); 0 = none")
    ap.add_argument("--disp", type=float, default=0.03)
    ap.add_argument("--ecutwfc", type=float, default=40.0)
    ap.add_argument("--ecutrho", type=float, default=160.0)
    ap.add_argument("--kpts", type=int, default=3)
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/fc3_cs")
    args = ap.parse_args()

    a = {"Si": bulk("Si", "diamond", a=5.43)}[args.material]
    uc = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=a.cell.array,
                      scaled_positions=a.get_scaled_positions())
    n = args.sc
    ph3 = Phono3py(uc, supercell_matrix=[[n, 0, 0], [0, n, 0], [0, 0, n]], primitive_matrix="auto")
    # RANDOM displacements (compressed-sensing dataset) instead of systematic 6N set
    ph3.generate_displacements(distance=args.disp, number_of_snapshots=args.rd, random_seed=1)
    scells = ph3.supercells_with_displacements
    print(f"[{args.material}] sc={n}^3 ({len(ph3.supercell)} atoms): {len(scells)} RANDOM-disp "
          f"supercells (systematic would be ~hundreds)", flush=True)

    pseudos = pseudos_for(a)
    t0 = time.perf_counter(); done = [0]; lock = threading.Lock()

    def one(arg):
        i, sc = arg
        calc = make_espresso(PSEUDO, pseudos, args.ecutwfc, args.ecutrho, args.kpts, args.nproc,
                             directory=f"/tmp/fc3cs_{i}")
        at = Atoms(symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        at.calc = calc
        f = at.get_forces()
        with lock:
            done[0] += 1
            print(f"  {done[0]}/{len(scells)} SCFs ({time.perf_counter()-t0:.0f}s)", flush=True)
        return i, f

    print(f"computing {len(scells)} SCFs with {args.workers} workers x {args.nproc} ranks", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = sorted(ex.map(one, enumerate(scells)), key=lambda x: x[0])
    ph3.forces = np.array([f for _, f in res])

    # symfc regression fit (compressed sensing), with optional cutoff
    opts = {"cutoff": {3: args.cutoff}} if args.cutoff > 0 else None
    ph3.produce_fc2(fc_calculator="symfc")
    ph3.produce_fc3(fc_calculator="symfc", fc_calculator_options=opts)
    print(f"[{args.material}] fc2 {ph3.fc2.shape}, fc3 {ph3.fc3.shape} (symfc fit from {args.rd} configs)",
          flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import h5py
    with h5py.File(out / f"{args.material}_fc2.hdf5", "w") as f:
        f.create_dataset("force_constants", data=ph3.fc2)
    with h5py.File(out / f"{args.material}_fc3.hdf5", "w") as f:
        f.create_dataset("fc3", data=ph3.fc3)
    sc = ph3.supercell
    np.savez(out / f"{args.material}_cells.npz", sc_matrix=[[n,0,0],[0,n,0],[0,0,n]],
             sc_numbers=sc.numbers, sc_positions=sc.scaled_positions, sc_cell=sc.cell)
    print(f"saved -> {out}  (compressed-sensing fc3, {args.rd} configs, cutoff {args.cutoff or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
