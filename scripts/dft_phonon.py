"""Line B: GPU/CPU finite-displacement phonons with Quantum ESPRESSO, driven
through the SAME PhononCalculation pipeline used for the MLIPs (so DFT and MLIP
phonons are computed identically -> a fair acceleration + accuracy comparison).

Reuses ASE's Espresso calculator: PhononCalculation builds the symmetry-reduced
displaced supercells, QE computes forces on each, phonopy builds the force
constants. Per-stage wall-clock is recorded for the speedup study.

    python scripts/dft_phonon.py --material mp-149 --supercell 2 \
        --ecutwfc 40 --kpts 4 --nproc 32 --out results/dft/si.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonon_accel import reference
from phonon_accel.phonons import PhononCalculation


def make_espresso(pseudo_dir, pseudopotentials, ecutwfc, ecutrho, kpts, nproc, conv_thr=1e-8):
    from ase.calculators.espresso import Espresso, EspressoProfile

    pw = f"{Path.home()}/miniconda3/envs/dft/bin/pw.x"
    mpi = f"mpirun -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=mpi, pseudo_dir=str(pseudo_dir))
    input_data = {
        "control": {"calculation": "scf", "tprnfor": True, "tstress": False, "disk_io": "low"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "gaussian", "degauss": 0.01},
        "electrons": {"conv_thr": conv_thr, "mixing_beta": 0.7},
    }
    return Espresso(profile=profile, pseudopotentials=pseudopotentials,
                    input_data=input_data, kpts=(kpts, kpts, kpts))


def pseudos_for(atoms, pseudo_dir):
    """Map each element to the first matching UPF in pseudo_dir."""
    pd = Path(pseudo_dir)
    out = {}
    for sym in sorted(set(atoms.get_chemical_symbols())):
        cands = list(pd.glob(f"{sym}.*UPF")) + list(pd.glob(f"{sym}_*UPF")) + list(pd.glob(f"{sym}.*upf"))
        if not cands:
            raise FileNotFoundError(f"no UPF for {sym} in {pd}")
        out[sym] = cands[0].name
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="mp-149")
    ap.add_argument("--supercell", type=int, default=2)
    ap.add_argument("--ecutwfc", type=float, default=40.0)
    ap.add_argument("--ecutrho", type=float, default=320.0)
    ap.add_argument("--kpts", type=int, default=4)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--disp", type=float, default=0.01)
    ap.add_argument("--pseudo-dir", default=f"{Path.home()}/pseudo")
    ap.add_argument("--out", default="results/dft/phonon.json")
    args = ap.parse_args()

    atoms = reference.reference_atoms(args.material)
    pseudos = pseudos_for(atoms, args.pseudo_dir)
    print(f"[{args.material}] {atoms.get_chemical_formula()} natoms={len(atoms)} pseudos={pseudos}", flush=True)

    sc = [[args.supercell, 0, 0], [0, args.supercell, 0], [0, 0, args.supercell]]
    phon = PhononCalculation(atoms, supercell_matrix=sc, displacement=args.disp)
    ndisp = phon.n_displacements
    print(f"symmetry-reduced displacements: {ndisp} (supercell {args.supercell}^3, {len(atoms)*args.supercell**3} atoms)", flush=True)

    calc = make_espresso(args.pseudo_dir, pseudos, args.ecutwfc, args.ecutrho, args.kpts, args.nproc)
    t0 = time.perf_counter()
    res = phon.run_all(calculator=calc, mesh=(12, 12, 12))
    wall = time.perf_counter() - t0

    out = dict(material=args.material, formula=atoms.get_chemical_formula(), natoms=len(atoms),
               supercell=args.supercell, n_displacements=ndisp, ecutwfc=args.ecutwfc, kpts=args.kpts,
               nproc=args.nproc, wall_s=wall, per_disp_s=wall / max(ndisp, 1),
               omega_max=float(res.mesh_frequencies.max()), n_imaginary=int(res.n_imaginary_mesh),
               asr_residual=float(res.asr_residual))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
