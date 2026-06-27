"""V-Q3 (Gate #2): NbSe2 monolayer DFT fc2 via finite-displacement + Quantum
ESPRESSO -- the distillation target that tests whether a *material-specific* DFT
distillation can recover the CDW soft mode that every foundation/general MLIP
backbone misses (TD_PHONON_M1_RESULTS.md: NbSe2 preview, Gate #2).

Monolayer 2H-NbSe2 is metallic with a 3x3 charge-density-wave (CDW) instability.
A frozen-phonon fc2 on a 3x3x1 supercell hosts the CDW wavevector, so the soft
(imaginary) mode appears along Gamma-M *iff* the DFT captures it. The electronic
smearing (degauss) is the (E)-channel knob: too-large degauss (high electronic T)
suppresses the CDW, so we use a modest degauss + dense k.

Saves the phonopy object (full fc2) for FC-distillation + the DFT dispersion on
Gamma-M-K-Gamma (npz). The downstream chain (distill -> fine-tune MACE -> re-eval
dispersion) then asks: does the distilled MLIP finally show the soft mode?

    python scripts/vq3_nbse2_dft.py --pw /root/gpupw.sh --nproc 1 \
        --pseudo-dir pseudo --supercell 3 --ecutwfc 70 --kpts 6 --degauss 0.015
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


def make_espresso(pw, mpirun, nproc, pseudo_dir, pseudos, ecutwfc, ecutrho,
                  kpts, degauss=0.015, conv_thr=1e-8, directory=None):
    from ase.calculators.espresso import Espresso, EspressoProfile

    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pseudo_dir))
    input_data = {
        "control": {"calculation": "scf", "tprnfor": True, "tstress": False,
                    "disk_io": "low", "verbosity": "low"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho,
                   "occupations": "smearing", "smearing": "cold", "degauss": degauss},
        # metallic + transition-metal d states: robust diag + generous maxstep
        "electrons": {"conv_thr": conv_thr, "mixing_beta": 0.3,
                      "electron_maxstep": 250, "diago_david_ndim": 4,
                      "startingwfc": "atomic+random"},
    }
    kw = {"directory": Path(directory)} if directory else {}
    return Espresso(profile=profile, pseudopotentials=pseudos,
                    input_data=input_data, kpts=(kpts, kpts, 1), **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--pseudo-dir", default="pseudo")
    ap.add_argument("--a", type=float, default=3.44, help="NbSe2 in-plane a (A); lit 3.44")
    ap.add_argument("--thickness", type=float, default=3.34, help="Se-Se thickness (A); lit 3.34")
    ap.add_argument("--supercell", type=int, default=3, help="n x n x 1 (3 hosts the 3x3 CDW)")
    ap.add_argument("--disp", type=float, default=0.03)
    ap.add_argument("--ecutwfc", type=float, default=70.0)
    ap.add_argument("--ecutrho", type=float, default=280.0)
    ap.add_argument("--kpts", type=int, default=6, help="supercell k-mesh (k,k,1)")
    ap.add_argument("--degauss", type=float, default=0.015, help="smearing (Ry); CDW (E)-knob")
    ap.add_argument("--npoints", type=int, default=201)
    ap.add_argument("--workdir", default="results/vq3")
    ap.add_argument("--tag", default="nbse2_dft")
    a = ap.parse_args()

    from phonon_accel.phonons import PhononCalculation
    import anomaly_locate as al  # noqa: F401 (kept for parity / future cusp scan)

    workdir = ROOT / a.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    pseudo_dir = (ROOT / a.pseudo_dir) if not Path(a.pseudo_dir).is_absolute() else Path(a.pseudo_dir)

    atoms = tdc.build_monolayer("nbse2", a=a.a, thickness=a.thickness)
    atoms.wrap()
    print(f"[{a.tag}] NbSe2 a={a.a} t={a.thickness} A, supercell {a.supercell}x{a.supercell}x1, "
          f"ecutwfc={a.ecutwfc} kpts={a.kpts} degauss={a.degauss} nproc={a.nproc}", flush=True)

    sc = np.diag([a.supercell, a.supercell, 1])
    phon = PhononCalculation(atoms, supercell_matrix=sc, primitive_matrix=np.eye(3),
                             displacement=a.disp)
    nd = phon.n_displacements
    print(f"[{a.tag}] {nd} displaced supercell(s) "
          f"({len(phon.displaced_supercells[0])} atoms each); running QE ...", flush=True)
    t0 = time.perf_counter()
    forces = []
    for i, scell in enumerate(phon.displaced_supercells):
        d = workdir / f"disp-{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        scell.calc = make_espresso(a.pw, a.mpirun or "mpirun", a.nproc, pseudo_dir,
                                   PSEUDOS, a.ecutwfc, a.ecutrho, a.kpts,
                                   degauss=a.degauss, directory=d)
        forces.append(scell.get_forces())
        print(f"[{a.tag}]   disp {i+1}/{nd}: max|F|={np.abs(forces[-1]).max():.4f} eV/A "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    phon.set_forces(np.array(forces))
    phon.produce_force_constants(symmetrize=True)
    ph = phon.phonon

    save = workdir / f"{a.tag}_phonopy.yaml"
    ph.save(filename=str(save), settings={"force_constants": True})
    print(f"[{a.tag}] saved phonopy -> {save.relative_to(ROOT)}", flush=True)

    # DFT dispersion on Gamma-M-K-Gamma (CDW soft mode would dip along Gamma-M)
    qpoints, conn, labels = tdc.make_band_path("GMKG", npoints=a.npoints)
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
    np.savez(out, distances=dist, frequencies=freq, label_positions=lp,
             labels=np.array(labels), a=a.a)
    fmin = float(freq.min())
    n_imag = int((freq < -0.1).sum())  # modes below -0.1 THz = genuine soft modes
    print(f"[{a.tag}] min freq = {fmin:.3f} THz, n_imaginary(<-0.1 THz) = {n_imag} "
          f"-> {'SOFT MODE PRESENT (CDW captured)' if fmin < -0.1 else 'no soft mode (stable)'}",
          flush=True)
    print(f"[{a.tag}] saved dispersion -> {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
