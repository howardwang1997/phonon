"""Anharmonic downstream (the NCS impact piece): lattice thermal conductivity
kappa from a foundation/fine-tuned MLIP via phono3py (3rd-order force constants +
RTA). No DFT — MLIP forces on the 3rd-order displaced supercells. Validate vs
public DFT/experimental kappa (Si ~130 W/mK exp, ~140 DFT). Runs on a GPU.

    python scripts/run_mlip_kappa.py --material Si --model small --sc 2 --mesh 19
    python scripts/run_mlip_kappa.py --material mp-149 --model results/.../ft.model --sc 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phonon_accel.mlip_calc import get_calculator


def get_unitcell(material):
    from phonopy.structure.atoms import PhonopyAtoms
    if material.startswith("mp-"):
        from phonon_accel import reference
        a = reference.reference_atoms(material)
    else:
        from ase.build import bulk
        db = {
            "Si": ("diamond", 5.43), "Ge": ("diamond", 5.658), "C": ("diamond", 3.567),
            "Sn": ("diamond", 6.489),
            "SiC": ("zincblende", 4.36), "BN": ("zincblende", 3.615), "BP": ("zincblende", 4.538),
            "BAs": ("zincblende", 4.777), "AlP": ("zincblende", 5.463), "AlAs": ("zincblende", 5.661),
            "GaP": ("zincblende", 5.451), "GaAs": ("zincblende", 5.653), "InP": ("zincblende", 5.869),
            "MgO": ("rocksalt", 4.21), "NaCl": ("rocksalt", 5.64),
        }
        struct, lat = db[material]
        a = bulk(material, struct, a=lat)
    return PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=a.cell.array,
                        scaled_positions=a.get_scaled_positions())


def forces_on(supercells, calc):
    from ase import Atoms
    out = []
    for sc in supercells:
        at = Atoms(symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        at.calc = calc
        out.append(at.get_forces())
    return out


def main() -> int:
    from phono3py import Phono3py

    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Si")
    ap.add_argument("--model", default="small")
    ap.add_argument("--sc", type=int, default=2, help="supercell n (n x n x n)")
    ap.add_argument("--mesh", type=int, default=19, help="q-mesh for BTE")
    ap.add_argument("--disp", type=float, default=0.03)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tmin", type=int, default=200)
    ap.add_argument("--tmax", type=int, default=600)
    ap.add_argument("--born", default=None, help="phonopy BORN file (Z*+dielectric) -> NAC for polar κ")
    ap.add_argument("--out", default="results/kappa/kappa.json")
    args = ap.parse_args()

    uc = get_unitcell(args.material)
    n = args.sc
    ph3 = Phono3py(uc, supercell_matrix=[[n, 0, 0], [0, n, 0], [0, 0, n]], primitive_matrix="auto")
    ph3.generate_displacements(distance=args.disp)
    n3 = len(ph3.supercells_with_displacements)
    print(f"[{args.material}] model={args.model} sc={n}^3 mesh={args.mesh}: {n3} 3rd-order displacements", flush=True)

    calc = get_calculator("mace", device=args.device, model=args.model)
    t0 = time.perf_counter()
    ph3.forces = np.array(forces_on(ph3.supercells_with_displacements, calc))
    ph3.produce_fc3()
    ph3.produce_fc2()  # fc2 from the same supercell's single-displacement subset
    t_forces = time.perf_counter() - t0

    if args.born:  # non-analytical term correction (LO-TO) for polar materials
        from phonopy.file_IO import parse_BORN
        try:
            ph3.nac_params = parse_BORN(ph3.phonon_primitive, filename=args.born)
        except Exception as _e:
            # dft_born.py writes Z* for every primitive atom; parse_BORN expects only
            # the symmetry-inequivalent set. Build nac_params from the full array instead.
            ls = [l.split() for l in open(args.born).read().strip().split("\n")]
            ph3.nac_params = {
                "born": np.array([np.array(l, float).reshape(3, 3) for l in ls[2:]]),
                "dielectric": np.array(ls[1], float).reshape(3, 3),
                "factor": float(ls[0][0]),
            }
            print(f"NAC fallback (full Z* array, parse_BORN said: {_e})", flush=True)
        print(f"NAC enabled from {args.born}", flush=True)
    ph3.mesh_numbers = [args.mesh] * 3
    ph3.init_phph_interaction()
    temps = list(range(args.tmin, args.tmax + 1, 100))
    ph3.run_thermal_conductivity(temperatures=temps, is_isotope=True)
    tc = ph3.thermal_conductivity
    kappa = tc.kappa[0]  # shape (n_temp, 6): xx,yy,zz,yz,xz,xy
    out = {"material": args.material, "model": args.model, "mesh": args.mesh, "n_disp3": n3,
           "t_forces_s": round(t_forces, 1),
           "kappa": {str(T): round(float(np.mean(kappa[i, :3])), 2) for i, T in enumerate(temps)}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(__import__("json").dumps(out, indent=2))
    print(__import__("json").dumps(out, indent=2))
    k300 = out["kappa"].get("300") or list(out["kappa"].values())[0]
    print(f"\nkappa(~300K) = {k300} W/m·K  ({n3} MLIP force evals in {t_forces:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
