"""M1.1b step 1: graphene DFT fc2 via finite-displacement + Quantum ESPRESSO.

Drives QE through the same ``PhononCalculation`` pipeline as the MLIPs (fair,
identical settings), on a 2D graphene monolayer, then saves the phonopy object
(full force constants) so the FC-distillation can synthesise harmonic training
data from the *graphene-specific* DFPT-quality fc2 -- the test of whether a
graphene-targeted distillation finally recovers the Gamma-E2g Kohn cusp that the
bulk-trained model misses (paper section 2.1).

Also writes the DFT dispersion on M-Gamma-K-M (npz) for a direct MLIP-vs-DFT
comparison.

    conda run -n qe ... (pw.x in PATH) ; or pass --pw / --mpirun explicitly
    python scripts/m1_1b_graphene_dft.py --pw $CONDA_PREFIX/bin/pw.x \
        --mpirun $CONDA_PREFIX/bin/mpirun --nproc 16 \
        --pseudo-dir pseudo --pseudo C_ONCV_PBE-1.2.upf \
        --a 2.46 --supercell 5 --ecutwfc 60 --kpts 4
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
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


def make_espresso(pw, mpirun, nproc, pseudo_dir, pseudo, ecutwfc, ecutrho,
                  kpts, degauss=0.02, smearing="cold", conv_thr=1e-9,
                  mixing_beta=0.4, mixing_mode="plain", electron_maxstep=200,
                  diagonalization="david", startingwfc="atomic+random",
                  diago_thr_init=None, disk_io="low", prefix="pwscf", directory=None):
    from ase.calculators.espresso import Espresso, EspressoProfile

    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pseudo_dir))
    input_data = {
        "control": {"calculation": "scf", "tprnfor": True, "tstress": False,
                    "disk_io": disk_io, "verbosity": "low", "prefix": prefix},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho,
                   "occupations": "smearing", "smearing": smearing, "degauss": degauss},
        "electrons": {"conv_thr": conv_thr, "mixing_beta": mixing_beta,
                      "mixing_mode": mixing_mode, "electron_maxstep": electron_maxstep,
                      "diagonalization": diagonalization, "diago_david_ndim": 4,
                      "startingwfc": startingwfc},
    }
    if diago_thr_init is not None:
        input_data["electrons"]["diago_thr_init"] = diago_thr_init
    kw = {"directory": Path(directory)} if directory else {}
    return Espresso(profile=profile, pseudopotentials={"C": pseudo},
                    input_data=input_data, kpts=(kpts, kpts, 1), **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--pseudo-dir", default="pseudo")
    ap.add_argument("--pseudo", default="C_ONCV_PBE-1.2.upf")
    ap.add_argument("--a", type=float, default=2.46, help="graphene lattice const (A)")
    ap.add_argument("--supercell", type=int, default=5)
    ap.add_argument("--disp", type=float, default=0.03)
    ap.add_argument("--ecutwfc", type=float, default=60.0)
    ap.add_argument("--ecutrho", type=float, default=240.0)
    ap.add_argument("--kpts", type=int, default=4, help="supercell k-mesh (k,k,1)")
    ap.add_argument("--degauss", type=float, default=0.02, help="electronic smearing (Ry); V-Q2 scans this")
    ap.add_argument("--smearing", default="cold", help="cold|fermi-dirac (V-Q2 uses fermi-dirac: degauss=k_B*T_el)")
    ap.add_argument("--npoints", type=int, default=201)
    ap.add_argument("--workdir", default="results/m1_1b/dft")
    ap.add_argument("--tag", default="graphene_dft")
    ap.add_argument("--scratch", default=None,
                    help="QE scratch root; each tag gets an isolated subdirectory")
    ap.add_argument("--clean-scratch-on-success", action="store_true")
    ap.add_argument("--conv-thr", type=float, default=1e-9)
    ap.add_argument("--mixing-beta", type=float, default=0.4)
    ap.add_argument("--mixing-mode", default="plain")
    ap.add_argument("--electron-maxstep", type=int, default=200)
    ap.add_argument("--diagonalization", default="david")
    ap.add_argument("--startingwfc", default="atomic+random")
    ap.add_argument("--diago-thr-init", type=float, default=None)
    ap.add_argument("--disk-io", default="low")
    a = ap.parse_args()

    from phonon_accel.phonons import PhononCalculation
    import anomaly_locate as al

    workdir = ROOT / a.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", a.tag):
        raise ValueError(f"unsafe --tag {a.tag!r}; use letters, numbers, '.', '_', '+', or '-'")
    pseudo_dir = (ROOT / a.pseudo_dir) if not Path(a.pseudo_dir).is_absolute() else Path(a.pseudo_dir)
    scratch = Path(a.scratch) / a.tag if a.scratch else workdir / "qe" / a.tag

    atoms = tdc.build_monolayer("graphene", a=a.a)
    atoms.wrap()
    print(f"[{a.tag}] graphene a={a.a} A, supercell {a.supercell}x{a.supercell}x1, "
          f"ecutwfc={a.ecutwfc} kpts={a.kpts} degauss={a.degauss} nproc={a.nproc} "
          f"conv_thr={a.conv_thr} mixing={a.mixing_mode}/{a.mixing_beta} "
          f"diag={a.diagonalization} maxstep={a.electron_maxstep}", flush=True)

    sc = np.diag([a.supercell, a.supercell, 1])
    phon = PhononCalculation(atoms, supercell_matrix=sc, primitive_matrix=np.eye(3),
                             displacement=a.disp)
    print(f"[{a.tag}] {phon.n_displacements} displaced supercell(s) "
          f"({len(phon.displaced_supercells[0])} atoms each); running QE ...", flush=True)
    t0 = time.perf_counter()
    # run each displaced supercell in its own QE workdir
    forces = []
    for i, scell in enumerate(phon.displaced_supercells):
        d = scratch / f"disp-{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        scell.calc = make_espresso(a.pw, a.mpirun or "mpirun", a.nproc, pseudo_dir,
                                   a.pseudo, a.ecutwfc, a.ecutrho, a.kpts,
                                   degauss=a.degauss, smearing=a.smearing,
                                   conv_thr=a.conv_thr, mixing_beta=a.mixing_beta,
                                   mixing_mode=a.mixing_mode,
                                   electron_maxstep=a.electron_maxstep,
                                   diagonalization=a.diagonalization,
                                   startingwfc=a.startingwfc,
                                   diago_thr_init=a.diago_thr_init,
                                   disk_io=a.disk_io, prefix=a.tag, directory=d)
        forces.append(scell.get_forces())
        print(f"[{a.tag}]   disp {i}: max|F|={np.abs(forces[-1]).max():.4f} eV/A "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    phon.set_forces(np.array(forces))
    phon.produce_force_constants(symmetrize=True)
    ph = phon.phonon

    # save phonopy (full FC) for distillation
    save = workdir / f"{a.tag}_phonopy.yaml"
    save_tmp = save.with_name(save.name + ".tmp")
    ph.save(filename=str(save_tmp), settings={"force_constants": True})

    # DFT dispersion on M-Gamma-K-M
    qpoints, conn, labels = tdc.make_band_path("MGKM", npoints=a.npoints)
    ph.run_band_structure(qpoints, path_connections=conn, labels=labels,
                          with_eigenvectors=False)
    bsd = ph.get_band_structure_dict()
    dist = np.concatenate([np.asarray(x) for x in bsd["distances"]])
    freq = np.concatenate([np.asarray(x) for x in bsd["frequencies"]], axis=0)
    seg = [np.asarray(x) for x in bsd["distances"]]
    lp = np.array([seg[0][0]] + [s[-1] for s in seg])
    keep = np.concatenate([[True], np.diff(dist) > 1e-9])
    dist, freq = dist[keep], freq[keep]
    out = workdir / f"disp_{a.tag}.npz"
    out_tmp = out.with_name(out.name + ".tmp")
    with out_tmp.open("wb") as handle:
        np.savez(handle, distances=dist, frequencies=freq, label_positions=lp,
                 labels=np.array(labels), a=a.a, tag=np.array(a.tag),
                 degauss=a.degauss, smearing=np.array(a.smearing),
                 supercell=a.supercell, displacement=a.disp,
                 ecutwfc=a.ecutwfc, ecutrho=a.ecutrho, kpts=a.kpts,
                 conv_thr=a.conv_thr, mixing_beta=a.mixing_beta,
                 mixing_mode=np.array(a.mixing_mode),
                 electron_maxstep=a.electron_maxstep,
                 diagonalization=np.array(a.diagonalization),
                 startingwfc=np.array(a.startingwfc))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(out_tmp, out)
    os.replace(save_tmp, save)
    print(f"[{a.tag}] saved phonopy -> {save.relative_to(ROOT)}", flush=True)
    cm = 33.35641
    wG = al.branch_freq_at_label(dist, freq, lp, np.array(labels), r"$\Gamma$") * cm
    wK = al.branch_freq_at_label(dist, freq, lp, np.array(labels), "K") * cm
    print(f"[{a.tag}] DFT top optical: Gamma={wG:.0f} cm^-1  K={wK:.0f} cm^-1 "
          f"(lit ~1600 / ~1300)", flush=True)
    print(f"[{a.tag}] min freq = {freq.min():.2f} THz -> {out.relative_to(ROOT)}", flush=True)
    if a.clean_scratch_on_success:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
