"""Generate 3rd-order force constants (Phi_3) via QE (phono3py) for a material ->
the anharmonic reference for 3rd-order FC distillation. Saves fc2 + fc3 to hdf5.
Small DFT (one material, ~tens of supercell SCFs on CPU).

    python scripts/dft_fc3.py --material Si --sc 2 --ecutwfc 50 --nproc 32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dft_phonon import make_espresso, pseudos_for  # QE calc setup

PSEUDO = Path.home() / "pseudo"


def get_atoms(material):
    if material.startswith("mp-"):
        from phonon_accel import reference
        return reference.reference_atoms(material)
    from ase.build import bulk
    return {"Si": bulk("Si", "diamond", a=5.43), "MgO": bulk("MgO", "rocksalt", a=4.21)}[material]


def to_phonopyatoms(a):
    from phonopy.structure.atoms import PhonopyAtoms
    return PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=a.cell.array,
                        scaled_positions=a.get_scaled_positions())


def main() -> int:
    from phono3py import Phono3py
    from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
    from ase import Atoms

    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Si")
    ap.add_argument("--sc", type=int, default=2)
    ap.add_argument("--ecutwfc", type=float, default=50.0)
    ap.add_argument("--ecutrho", type=float, default=200.0)
    ap.add_argument("--kpts", type=int, default=4)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--disp", type=float, default=0.03)
    ap.add_argument("--out-dir", default="results/fc3")
    args = ap.parse_args()

    a = get_atoms(args.material)
    pseudos = pseudos_for(a, PSEUDO)
    n = args.sc
    ph3 = Phono3py(to_phonopyatoms(a), supercell_matrix=[[n, 0, 0], [0, n, 0], [0, 0, n]],
                   primitive_matrix="auto")
    ph3.generate_displacements(distance=args.disp)
    scells = ph3.supercells_with_displacements
    print(f"[{args.material}] sc={n}^3: {len(scells)} 3rd-order displaced supercells "
          f"({len(scells[0])} atoms each)", flush=True)

    calc = make_espresso(PSEUDO, pseudos, args.ecutwfc, args.ecutrho, args.kpts, args.nproc,
                         directory="/tmp/fc3qe")
    t0 = time.perf_counter()
    forces = []
    for i, sc in enumerate(scells):
        at = Atoms(symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        at.calc = calc
        forces.append(at.get_forces())
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(scells)} SCFs done ({time.perf_counter()-t0:.0f}s)", flush=True)
    ph3.forces = np.array(forces)
    ph3.produce_fc3(is_compact_fc=False)   # FULL fc (N_super, N_super, ...) for distillation einsum
    ph3.produce_fc2(is_compact_fc=False)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_fc2_to_hdf5(ph3.fc2, filename=str(out / f"{args.material}_fc2.hdf5"))
    write_fc3_to_hdf5(ph3.fc3, filename=str(out / f"{args.material}_fc3.hdf5"))
    # also save the supercell structure (needed to apply the FCs)
    np.savez(out / f"{args.material}_cells.npz",
             sc_positions=ph3.supercell.scaled_positions, sc_cell=ph3.supercell.cell,
             sc_numbers=ph3.supercell.numbers, sc_matrix=np.diag([n, n, n]))
    print(f"[{args.material}] fc2 {ph3.fc2.shape}, fc3 {ph3.fc3.shape} -> {out}  "
          f"({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
