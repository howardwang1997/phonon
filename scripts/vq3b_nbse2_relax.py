"""A (V-Q3 refinement): DFT in-plane relaxation of monolayer NbSe2 -> the PBE
equilibrium lattice constant, so the Gate#2 fc2 can be recomputed at the DFT
geometry (removing the literature-a=3.44 caveat on the soft-mode magnitude).

vc-relax with cell_dofree='2Dxy' relaxes a,b (+ atoms) holding the c vacuum
fixed. Prints A_DFT / THICKNESS_DFT (parseable) for the downstream fc2 run.

    python scripts/vq3b_nbse2_relax.py --pw /root/gpupw.sh --nproc 1 \
        --pseudo-dir pseudo --ecutwfc 70 --kpts 12 --degauss 0.015
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import td_common as tdc

PSEUDOS = {"Nb": "Nb_ONCV_PBE-1.2.upf", "Se": "Se_ONCV_PBE-1.2.upf"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--pseudo-dir", default="pseudo")
    ap.add_argument("--a", type=float, default=3.44, help="starting in-plane a (A)")
    ap.add_argument("--thickness", type=float, default=3.34)
    ap.add_argument("--ecutwfc", type=float, default=70.0)
    ap.add_argument("--ecutrho", type=float, default=280.0)
    ap.add_argument("--kpts", type=int, default=12, help="primitive k-mesh (k,k,1); dense for metal")
    ap.add_argument("--degauss", type=float, default=0.015)
    ap.add_argument("--workdir", default="results/vq3b")
    a = ap.parse_args()

    from ase.calculators.espresso import Espresso, EspressoProfile
    from ase.io import read

    pdir = (ROOT / a.pseudo_dir) if not Path(a.pseudo_dir).is_absolute() else Path(a.pseudo_dir)
    work = ROOT / a.workdir
    work.mkdir(parents=True, exist_ok=True)
    atoms = tdc.build_monolayer("nbse2", a=a.a, thickness=a.thickness)
    atoms.wrap()

    cmd = f"{a.mpirun or 'mpirun'} --allow-run-as-root -np {a.nproc} {a.pw}" if a.nproc > 1 else a.pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pdir))
    input_data = {
        "control": {"calculation": "vc-relax", "tprnfor": True, "tstress": True,
                    "forc_conv_thr": 1e-4, "disk_io": "low", "verbosity": "low",
                    "nstep": 120},
        "system": {"ecutwfc": a.ecutwfc, "ecutrho": a.ecutrho, "occupations": "smearing",
                   "smearing": "cold", "degauss": a.degauss},
        "electrons": {"conv_thr": 1e-9, "mixing_beta": 0.3, "electron_maxstep": 250,
                      "diago_david_ndim": 4, "startingwfc": "atomic+random"},
        "ions": {"ion_dynamics": "bfgs"},
        "cell": {"cell_dofree": "2Dxy", "cell_dynamics": "bfgs", "press_conv_thr": 0.5},
    }
    calc = Espresso(profile=profile, pseudopotentials=PSEUDOS, input_data=input_data,
                    kpts=(a.kpts, a.kpts, 1), directory=work)
    print(f"[relax] NbSe2 vc-relax (2Dxy) from a={a.a}; ecutwfc={a.ecutwfc} kpts={a.kpts} "
          f"degauss={a.degauss} ...", flush=True)
    t0 = time.perf_counter()
    atoms.calc = calc
    atoms.get_potential_energy()  # runs the full QE relaxation
    relaxed = read(work / "espresso.pwo", index=-1)  # last frame = relaxed
    a_dft = float(np.linalg.norm(relaxed.cell[0]))
    z = relaxed.get_positions()[:, 2]
    thick = float(z.max() - z.min())
    print(f"[relax] done ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"[relax] A_DFT={a_dft:.4f} THICKNESS_DFT={thick:.4f} (started a={a.a}, t={a.thickness})",
          flush=True)
    from ase.io import write
    write(work / "nbse2_relaxed.xyz", relaxed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
