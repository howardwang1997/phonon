"""Line B cross-check: DFPT (QE ph.x, density-functional perturbation theory) vs
the finite-displacement workflow. Validates the finite-displacement phonons and
gives the DFPT cost baseline. Drives pw.x (SCF) then ph.x (linear response) on a
small q-grid, parses the highest phonon frequency, compares to MDR DFPT.

    python scripts/dft_dfpt.py --material mp-149 --ecutwfc 60 --kpts 8 --nq 2 --nproc 16
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ase.io.espresso import write_espresso_in
from phonon_accel import reference

HOME = Path.home()
DFT = HOME / "miniconda3" / "envs" / "dft" / "bin"
PSEUDO = HOME / "pseudo"


def pseudos_for(atoms):
    out = {}
    for s in sorted(set(atoms.get_chemical_symbols())):
        c = list(PSEUDO.glob(f"{s}.*UPF")) + list(PSEUDO.glob(f"{s}.*upf")) + list(PSEUDO.glob(f"{s}_*upf"))
        if not c:
            raise FileNotFoundError(f"no UPF for {s}")
        out[s] = c[0].name
    return out


def run(cmd, cwd, log):
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="mp-149")
    ap.add_argument("--ecutwfc", type=float, default=60.0)
    ap.add_argument("--ecutrho", type=float, default=240.0)
    ap.add_argument("--kpts", type=int, default=8)
    ap.add_argument("--nq", type=int, default=2, help="DFPT q-grid nq^3")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--out", default="results/dft/dfpt.json")
    args = ap.parse_args()

    from phonon_accel.phonons import phonopy_to_ase  # noqa
    atoms = reference.reference_atoms(args.material)
    pseudos = pseudos_for(atoms)
    wd = Path("/tmp/dfpt"); shutil.rmtree(wd, ignore_errors=True); (wd / "out").mkdir(parents=True)
    mpirun = [str(DFT / "mpirun"), "--allow-run-as-root", "-np", str(args.nproc)]

    # 1. pw.x SCF (tight, DFPT needs a good ground state)
    pwi = wd / "scf.in"
    with open(pwi, "w") as f:
        write_espresso_in(f, atoms, pseudopotentials=pseudos, kpts=(args.kpts,) * 3,
                          input_data={"control": {"calculation": "scf", "prefix": "dfpt",
                                                   "outdir": str(wd / "out"), "disk_io": "low"},
                                      "system": {"ecutwfc": args.ecutwfc, "ecutrho": args.ecutrho,
                                                 "occupations": "smearing", "smearing": "gaussian",
                                                 "degauss": 0.01},
                                      "electrons": {"conv_thr": 1e-10}})
    t0 = time.perf_counter()
    run(mpirun + [str(DFT / "pw.x"), "-in", str(pwi)], wd, wd / "scf.out")
    t_scf = time.perf_counter() - t0

    # 2. ph.x DFPT on nq^3 q-grid
    phi = wd / "ph.in"
    phi.write_text(
        f"DFPT\n&inputph\n  prefix='dfpt'\n  outdir='{wd/'out'}'\n  fildyn='dfpt.dyn'\n"
        f"  ldisp=.true., nq1={args.nq}, nq2={args.nq}, nq3={args.nq}\n  tr2_ph=1.0d-15\n/\n")
    t0 = time.perf_counter()
    run(mpirun + [str(DFT / "ph.x"), "-in", str(phi)], wd, wd / "ph.out")
    t_ph = time.perf_counter() - t0

    # 3. parse highest frequency over all q (THz) from ph.x output
    freqs = [float(m) for m in re.findall(r"freq \(.*?\) =\s*([-0-9.]+)\s*\[THz\]", (wd / "ph.out").read_text())]
    omega_max = max(freqs) if freqs else float("nan")
    n_imag = sum(1 for f in freqs if f < -0.1)

    ref_res, _ = reference.reference_result(args.material)
    om_ref = float(ref_res.mesh_frequencies.max())
    out = dict(material=args.material, method="DFPT(ph.x)", nq=args.nq,
               omega_max_dfpt=round(omega_max, 3), omega_max_mdr=round(om_ref, 3),
               err_pct=round(100 * (omega_max - om_ref) / om_ref, 1), n_imaginary=n_imag,
               t_scf_s=round(t_scf, 1), t_ph_s=round(t_ph, 1), t_total_s=round(t_scf + t_ph, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(__import__("json").dumps(out, indent=2))
    print(__import__("json").dumps(out, indent=2))
    print(f"\nDFPT omega_max={omega_max:.2f} THz vs MDR {om_ref:.2f} ({out['err_pct']:+.1f}%); "
          f"SCF {t_scf:.0f}s + ph.x {t_ph:.0f}s = {t_scf+t_ph:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
