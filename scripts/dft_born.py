"""Born effective charges + dielectric tensor via QE DFPT at Gamma (epsil) ->
phonopy BORN file. This is the small "DFT input" that enables NAC (LO-TO
splitting) for polar-material lattice thermal conductivity, while the MLIP
provides the (cheap) force constants. Generalizes the harmonic story's
"public DFT FC + MLIP" to the polar/NAC term.

    python scripts/dft_born.py --material MgO --ecutwfc 60 --nproc 24 --out results/kappa/MgO.BORN
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ase.io.espresso import write_espresso_in

HOME = Path.home()
DFT = HOME / "miniconda3" / "envs" / "dft" / "bin"
PSEUDO = HOME / "pseudo"


def get_atoms(material):
    if material.startswith("mp-"):
        from phonon_accel import reference
        return reference.reference_atoms(material)
    from ase.build import bulk
    return {"MgO": bulk("MgO", "rocksalt", a=4.21), "GaAs": bulk("GaAs", "zincblende", a=5.65),
            "NaCl": bulk("NaCl", "rocksalt", a=5.64)}[material]


def pseudos_for(atoms):
    out = {}
    for s in sorted(set(atoms.get_chemical_symbols())):
        c = list(PSEUDO.glob(f"{s}.*UPF")) + list(PSEUDO.glob(f"{s}.*upf")) + list(PSEUDO.glob(f"{s}_*upf"))
        if not c:
            raise FileNotFoundError(f"no UPF for {s}")
        out[s] = c[0].name
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="MgO")
    ap.add_argument("--ecutwfc", type=float, default=60.0)
    ap.add_argument("--ecutrho", type=float, default=240.0)
    ap.add_argument("--kpts", type=int, default=8)
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    atoms = get_atoms(args.material)
    pseudos = pseudos_for(atoms)
    wd = Path(f"/tmp/born_{args.material}"); shutil.rmtree(wd, ignore_errors=True); (wd / "out").mkdir(parents=True)
    mpi = [str(DFT / "mpirun"), "--allow-run-as-root", "-np", str(args.nproc)]

    # 1. SCF
    pwi = wd / "scf.in"
    with open(pwi, "w") as f:
        write_espresso_in(f, atoms, pseudopotentials=pseudos, kpts=(args.kpts,) * 3,
                          input_data={"control": {"calculation": "scf", "prefix": "born",
                                                   "outdir": str(wd / "out"), "pseudo_dir": str(PSEUDO)},
                                      "system": {"ecutwfc": args.ecutwfc, "ecutrho": args.ecutrho},
                                      "electrons": {"conv_thr": 1e-12}})
    subprocess.run(mpi + [str(DFT / "pw.x"), "-in", str(pwi)], cwd=wd,
                   stdout=open(wd / "scf.out", "w"), stderr=subprocess.STDOUT, check=True)
    # 2. ph.x at Gamma with epsil -> dielectric + Born charges
    phi = wd / "ph.in"
    phi.write_text(f"born\n&inputph\n  prefix='born'\n  outdir='{wd/'out'}'\n  fildyn='born.dyn'\n"
                   f"  epsil=.true.\n  trans=.true.\n  asr=.true.\n/\n0.0 0.0 0.0\n")
    subprocess.run(mpi + [str(DFT / "ph.x"), "-in", str(phi)], cwd=wd,
                   stdout=open(wd / "ph.out", "w"), stderr=subprocess.STDOUT, check=True)
    txt = (wd / "ph.out").read_text()

    # 3. parse dielectric tensor (3 lines "( a b c )" after the header)
    eps = None
    m = re.search(r"Dielectric constant in cartesian axis\s*\n\s*\n((?:\s*\([^\)]*\)\s*\n){3})", txt)
    if m:
        eps = np.array(re.findall(r"[-0-9.]+", m.group(1)), float).reshape(3, 3)
    # 4. Born effective charges: per atom "Mean Z*:" then Ex/Ey/Ez ( a b c )
    born = []
    for blk in re.finditer(
        r"atom\s+\d+\s+\w+\s+Mean Z\*:.*?\n"
        r"\s*Ex\s*\(([-0-9.\s]+)\)\s*\n\s*Ey\s*\(([-0-9.\s]+)\)\s*\n\s*Ez\s*\(([-0-9.\s]+)\)", txt):
        z = np.array([[float(x) for x in re.findall(r"[-0-9.]+", g)] for g in blk.groups()])
        born.append(z)
    # de-dup: QE prints the same Z* block twice (with/without ASR); keep first natom
    nat = len(get_atoms(args.material))
    born = born[:nat]
    if eps is None or not born:
        print("PARSE FAILED — dumping ph.out tail:")
        print("\n".join(txt.splitlines()[-30:]))
        return 1

    # 5. write phonopy BORN file
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("14.399652\n")
        f.write(" ".join(f"{x:.6f}" for x in eps.flatten()) + "\n")
        for z in born:
            f.write(" ".join(f"{x:.6f}" for x in z.flatten()) + "\n")
    print(f"dielectric (trace/3) = {np.trace(eps)/3:.3f}")
    for i, z in enumerate(born):
        print(f"Z*[{i}] (trace/3) = {np.trace(z)/3:+.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
