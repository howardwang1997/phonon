"""TMD-family DFT fc2 (finite-displacement + GPU pw.x) — generalizes
scripts/vq3_nbse2_dft.py to any 2H/1T monolayer in configs/v100_campaign.yaml.

Produces the per-material DFT force constants = the GROUND TRUTH that calibrates
the H20 (L)-channel MLIP+SSCHA screen, AND the material-specific FC-distillation
target (the V-Q3 recipe that transferred NbSe2's CDW into the MLIP). Saves the
phonopy object (full fc2, for distillation) + the DFT dispersion on G-M-K-G.

    python scripts/v100/tmd_dft_fc2.py --name NbS2 \
        --pw /root/gpupw.sh --nproc 1 --pseudo-dir /root/phonon/pseudo
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "v100"))

import numpy as np   # noqa: E402

import tmd_common as tc   # noqa: E402
import td_common as tdc   # noqa: E402 (make_band_path)


def make_espresso(pw, mpirun, nproc, pseudo_dir, pseudos, ecutwfc, ecutrho,
                  kpts, degauss, directory):
    from ase.calculators.espresso import Espresso, EspressoProfile
    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pseudo_dir))
    input_data = {
        # disk_io='none': forces only (tprnfor) -> no wfc/charge written. The
        # finite-displacement scratch would otherwise be ~GBs/disp and fill the
        # root fs. Scratch dirs live on /data (see --scratch) and are cleaned after.
        "control": {"calculation": "scf", "tprnfor": True, "tstress": False,
                    "disk_io": "none", "verbosity": "low"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho,
                   "occupations": "smearing", "smearing": "cold", "degauss": degauss},
        "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3,
                      "electron_maxstep": 250, "diago_david_ndim": 4,
                      "startingwfc": "atomic+random"},
    }
    return Espresso(profile=profile, pseudopotentials=pseudos,
                    input_data=input_data, kpts=(kpts, kpts, 1),
                    directory=Path(directory))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="configs/v100_campaign.yaml")
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--pseudo-dir", default="/root/phonon/pseudo")
    ap.add_argument("--workdir", default="results/v100/fc2")
    ap.add_argument("--scratch", default="/data/v100scratch",
                    help="big QE scratch root (keep OFF the root fs)")
    a = ap.parse_args()

    import shutil
    from phonon_accel.phonons import PhononCalculation

    cfg = tc.load_config(a.config)
    mat = tc.material(cfg, a.name)
    d = cfg["dft"]
    out_yaml = ROOT / a.workdir / f"{a.name}_phonopy.yaml"
    if out_yaml.exists():
        print(f"[fc2:{a.name}] {out_yaml.name} exists -> skip", flush=True)
        return 0
    work = Path(a.scratch) / "fc2" / a.name
    work.mkdir(parents=True, exist_ok=True)
    pseudos = tc.pseudo_map(mat, a.pseudo_dir)

    atoms = tc.build_tmd(mat["formula"], mat["polytype"], mat["a"], mat["thickness"])
    atoms.wrap()
    print(f"[fc2:{a.name}] {mat['polytype']}-{mat['formula']} a={mat['a']} "
          f"sc {d['fc2_supercell']}x{d['fc2_supercell']}x1 ecut={d['ecutwfc']} "
          f"k={d['fc2_kpts']} degauss={d['degauss']} pseudos={pseudos}", flush=True)

    n = d["fc2_supercell"]
    phon = PhononCalculation(atoms, supercell_matrix=np.diag([n, n, 1]),
                             primitive_matrix=np.eye(3), displacement=d["fc2_disp"])
    nd = phon.n_displacements
    print(f"[fc2:{a.name}] {nd} displaced supercell(s), {len(phon.displaced_supercells[0])} atoms; running QE ...", flush=True)
    t0 = time.perf_counter()
    forces = []
    for i, scell in enumerate(phon.displaced_supercells):
        dd = work / f"disp-{i:03d}"; dd.mkdir(parents=True, exist_ok=True)
        scell.calc = make_espresso(a.pw, a.mpirun, a.nproc, a.pseudo_dir, pseudos,
                                   d["ecutwfc"], d["ecutrho"], d["fc2_kpts"],
                                   d["degauss"], dd)
        forces.append(scell.get_forces())
        print(f"[fc2:{a.name}]   disp {i+1}/{nd}: max|F|={np.abs(forces[-1]).max():.4f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    phon.set_forces(np.array(forces))
    phon.produce_force_constants(symmetrize=True)
    ph = phon.phonon
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    ph.save(filename=str(out_yaml), settings={"force_constants": True})

    qpoints, conn, labels = tdc.make_band_path("GMKG", npoints=201)
    ph.run_band_structure(qpoints, path_connections=conn, labels=labels,
                          with_eigenvectors=False)
    bsd = ph.get_band_structure_dict()
    dist = np.concatenate([np.asarray(x) for x in bsd["distances"]])
    freq = np.concatenate([np.asarray(x) for x in bsd["frequencies"]], axis=0)
    seg = [np.asarray(x) for x in bsd["distances"]]
    lp = np.array([seg[0][0]] + [s[-1] for s in seg])
    keep = np.concatenate([[True], np.diff(dist) > 1e-9])
    dist, freq = dist[keep], freq[keep]
    np.savez(ROOT / a.workdir / f"disp_{a.name}.npz", distances=dist, frequencies=freq,
             label_positions=lp, labels=np.array(labels), a=mat["a"])
    fmin = float(freq.min())
    n_imag = int((freq < -0.1).sum())
    print(f"[fc2:{a.name}] min freq = {fmin:.3f} THz, n_imag(<-0.1) = {n_imag} -> "
          f"{'SOFT MODE (CDW captured)' if fmin < -0.1 else 'stable'}; saved {out_yaml.name}", flush=True)
    shutil.rmtree(work, ignore_errors=True)     # free the QE scratch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
