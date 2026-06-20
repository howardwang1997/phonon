"""Line B DATASET generator: self-produced DFT finite-displacement phonons over a
set of materials -> saved force constants (data/dft_ref/<id>_phonopy_params.yaml)
+ a validation manifest against MDR DFPT. This is the data-engine output that
feeds Line A's FC distillation (closing the loop), and a stand-alone near-DFT
phonon dataset.

Auto-selects MDR materials whose elements are covered by the available
pseudopotentials and whose supercell stays small enough for CPU DFT.

    python scripts/dft_dataset.py --max-atoms 80 --ecutwfc 50 --nproc 24 --limit 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonon_accel import reference
from phonon_accel.phonons import PhononCalculation
from dft_phonon import make_espresso, pseudos_for  # reuse the QE calc setup

PSEUDO_DIR = Path.home() / "pseudo"


def available_elements():
    els = set()
    for p in PSEUDO_DIR.glob("*.upf"):
        els.add(p.stem)
    return els


def kpts_for(atoms, kspacing=0.20):
    """k-grid from a target spacing (1/Angstrom) -> coarser for larger supercells."""
    recip = np.linalg.norm(np.linalg.inv(atoms.cell.array).T, axis=1) * 2 * np.pi
    return tuple(max(1, int(np.ceil(r / (kspacing * 2 * np.pi)))) for r in recip)


def to_primitive(atoms):
    """Reduce to the primitive cell (fewer atoms -> cheaper supercell DFT). ω is cell-invariant."""
    import spglib
    from ase import Atoms
    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    prim = spglib.find_primitive(cell, symprec=1e-4)
    if prim is None:
        return atoms
    lat, scaled, nums = prim
    return Atoms(numbers=nums, scaled_positions=scaled, cell=lat, pbc=True)


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-atoms", type=int, default=80, help="max supercell atoms (DFT cost cap)")
    ap.add_argument("--min-length", type=float, default=9.0,
                    help="target min supercell lattice length (Angstrom) -> adaptive supercell, "
                         "converged phonons at minimal cost (fixed 2x2x2 under-converges small cells)")
    ap.add_argument("--ecutwfc", type=float, default=50.0)
    ap.add_argument("--ecutrho", type=float, default=400.0)
    ap.add_argument("--kspacing", type=float, default=0.22)
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--disp", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--materials", nargs="+", default=None)
    ap.add_argument("--out-dir", default="data/dft_ref")
    ap.add_argument("--manifest", default="results/dft/dataset_manifest.csv")
    ap.add_argument("--workdir", default=None, help="unique QE working dir (for parallel instances)")
    args = ap.parse_args()

    have = available_elements()
    idx = reference._index()
    cached = sorted(p.name.split("_")[0] for p in (ROOT / "data" / "benchmark" / "mdr").glob("*_phonopy_params.yaml"))

    def els(mp):
        return set(re.findall(r"[A-Z][a-z]?", idx.get(mp, {}).get("formula", "")))

    # candidate materials: covered by pseudos, small supercell, prefer simple
    if args.materials:
        cands = args.materials
    else:
        cands = [mp for mp in cached if els(mp) and els(mp) <= have]
        cands.sort(key=lambda mp: len(idx[mp]["formula"]))
    print(f"available pseudos: {sorted(have)}")
    print(f"candidate materials (pseudo-covered): {len(cands)}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    done = 0
    for mp in cands:
        if done >= args.limit:
            break
        try:
            atoms = to_primitive(reference.reference_atoms(mp))
            # adaptive supercell: each axis multiplied until its length >= min-length
            lengths = np.linalg.norm(atoms.cell.array, axis=1)
            mult = [max(1, int(np.ceil(args.min_length / L))) for L in lengths]
            nat = len(atoms) * mult[0] * mult[1] * mult[2]
            if nat > args.max_atoms:
                print(f"  skip {mp} ({nat} atoms = {len(atoms)}*{mult} > {args.max_atoms})", flush=True)
                continue
            sc = [[mult[0], 0, 0], [0, mult[1], 0], [0, 0, mult[2]]]
            pseudos = pseudos_for(atoms, PSEUDO_DIR)
            phon = PhononCalculation(atoms, supercell_matrix=sc, displacement=args.disp)
            kpts = max(kpts_for(atoms * tuple(mult), args.kspacing))
            calc = make_espresso(PSEUDO_DIR, pseudos, args.ecutwfc, args.ecutrho, kpts, args.nproc,
                                 directory=args.workdir)
            t0 = time.perf_counter()
            res = phon.run_all(calculator=calc, mesh=(12, 12, 12))
            wall = time.perf_counter() - t0
            # save self-DFT force constants (MDR/phonopy format) -> FC distillation
            fc_path = out_dir / f"selfdft-{mp}_phonopy_params.yaml"
            phon.phonon.save(filename=str(fc_path), settings={"force_constants": True})
            # validate vs MDR DFPT
            ref_res, _ = reference.reference_result(mp)
            om_self = float(res.mesh_frequencies.max()); om_ref = float(ref_res.mesh_frequencies.max())
            row = dict(mp_id=mp, formula=atoms.get_chemical_formula(), natoms=nat,
                       n_disp=phon.n_displacements, kpts=kpts, wall_s=round(wall, 1),
                       omega_max_self=round(om_self, 3), omega_max_mdr=round(om_ref, 3),
                       omega_err_pct=round(100 * (om_self - om_ref) / om_ref, 1),
                       n_imag_self=int(res.n_imaginary_mesh), fc=str(fc_path))
            rows.append(row)
            done += 1
            print(f"  [{done}] {mp} {row['formula']}: omega_max self={om_self:.2f} mdr={om_ref:.2f} "
                  f"({row['omega_err_pct']:+.1f}%) {wall:.0f}s -> {fc_path.name}", flush=True)
            pd.DataFrame(rows).to_csv(args.manifest, index=False)  # incremental
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {mp}: {type(e).__name__}: {str(e)[:80]}", flush=True)

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.manifest, index=False)
        print(f"\n=== Line B dataset: {len(rows)} materials, mean |omega err| = "
              f"{df.omega_err_pct.abs().mean():.1f}% vs MDR DFPT ===")
        print("manifest:", args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
