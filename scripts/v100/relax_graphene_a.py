"""vc-relax graphene in-plane lattice constant a (PBE, Fermi-Dirac, cell_dofree='2Dxy').

Graphene is planar (atoms at z=0) with only the in-plane a as a degree of freedom.
Writes the relaxed a to <workdir>/a_relaxed.txt and prints A_RELAXED=<a> on the last
line (for shell capture). Used before the FD fc2 smearing scan so every smearing uses
the SAME properly-relaxed a (Kohn-anomaly depth is a-sensitive).

    python scripts/v100/relax_graphene_a.py --pw /root/gpupw.sh --workdir results/graphene_kohn_fd/relax
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def main():
    import argparse
    from ase.build import graphene
    from ase.calculators.espresso import Espresso, EspressoProfile
    from ase.io import read
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", required=True)
    ap.add_argument("--pseudo-dir", default="/root/phonon/pseudo")
    ap.add_argument("--pseudo", default="C_ONCV_PBE-1.2.upf")
    ap.add_argument("--a", type=float, default=2.46, help="starting graphene a (A)")
    ap.add_argument("--ecutwfc", type=float, default=60.0)
    ap.add_argument("--ecutrho", type=float, default=240.0)
    ap.add_argument("--kpts", type=int, default=24, help="primitive k-mesh (k,k,1)")
    ap.add_argument("--workdir", default="results/graphene_kohn_fd/relax")
    a = ap.parse_args()
    work = ROOT / a.workdir
    work.mkdir(parents=True, exist_ok=True)
    at = graphene(formula="C2", a=a.a, vacuum=7.5)
    at.pbc = True
    profile = EspressoProfile(command=a.pw, pseudo_dir=str(a.pseudo_dir))
    input_data = {
        "control": {"calculation": "vc-relax", "tprnfor": True, "tstress": True,
                    "disk_io": "low", "verbosity": "low"},
        "system": {"ecutwfc": a.ecutwfc, "ecutrho": a.ecutrho,
                   "occupations": "smearing", "smearing": "fd", "degauss": 0.01},
        "electrons": {"conv_thr": 1e-9, "mixing_beta": 0.3, "electron_maxstep": 250,
                      "diago_david_ndim": 4, "startingwfc": "atomic+random"},
        "cell": {"cell_dofree": "2Dxy", "cell_dynamics": "bfgs", "press_conv_thr": 0.5},
    }
    at.calc = Espresso(profile=profile, pseudopotentials={"C": a.pseudo},
                       input_data=input_data, kpts=(a.kpts, a.kpts, 1), directory=str(work))
    at.get_potential_energy()
    relaxed = read(work / "espresso.pwo", index=-1)
    a_rel = float(np.linalg.norm(relaxed.cell[0]))
    (work / "a_relaxed.txt").write_text(f"{a_rel:.6f}\n")
    print(f"A_RELAXED={a_rel:.6f}")


if __name__ == "__main__":
    main()
