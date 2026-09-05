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
import os
import re
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
                  kpts, degauss, directory, smearing="fd"):
    from ase.calculators.espresso import Espresso, EspressoProfile
    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pseudo_dir))
    input_data = {
        # disk_io='none': forces only (tprnfor) -> no wfc/charge written. The
        # finite-displacement scratch would otherwise be ~GBs/disp and fill the
        # root fs. Scratch dirs live on /data (see --scratch) and are cleaned after.
        "control": {"calculation": "scf", "tprnfor": True, "tstress": False,
                    "disk_io": "none", "verbosity": "low"},
        # smearing='fd' (Fermi-Dirac) by default: for the (E)-channel T_el axis
        # degauss = k_B*T_el EXACTLY (cold-smearing degauss is only a broadener).
        # Use --smearing cold to reproduce the earlier cold-smearing runs.
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho,
                   "occupations": "smearing", "smearing": smearing, "degauss": degauss},
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
    ap.add_argument("--degauss", type=float, default=None,
                    help="override config dft.degauss (Ry) — e.g. 0.002 for electronic-0K check")
    ap.add_argument("--supercell", type=int, default=None,
                    help="override config dft.fc2_supercell")
    ap.add_argument("--a", type=float, default=None,
                    help="override the config lattice constant (Angstrom)")
    ap.add_argument("--thickness", type=float, default=None,
                    help="override the config chalcogen-to-chalcogen thickness (Angstrom)")
    ap.add_argument("--tag", default=None,
                    help="artifact/scratch tag (defaults to --name); useful for provenance-safe reruns")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when both output artifacts already exist")
    ap.add_argument("--smearing", default="fd",
                    help="fd (Fermi-Dirac, default; degauss=k_B*T_el) | cold | gauss | mp")
    a = ap.parse_args()

    import shutil
    from phonon_accel.phonons import PhononCalculation

    cfg = tc.load_config(a.config)
    mat = tc.material(cfg, a.name)
    d = dict(cfg["dft"])
    if a.degauss is not None:
        d["degauss"] = a.degauss
    if a.supercell is not None:
        d["fc2_supercell"] = a.supercell
    lattice_a = float(a.a if a.a is not None else mat["a"])
    thickness = float(a.thickness if a.thickness is not None else mat["thickness"])
    tag = a.tag or a.name
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", tag):
        raise ValueError(f"unsafe --tag {tag!r}; use letters, numbers, '.', '_', '+', or '-'")
    outdir = ROOT / a.workdir
    out_yaml = outdir / f"{tag}_phonopy.yaml"
    out_npz = outdir / f"disp_{tag}.npz"
    if (not a.force and out_yaml.is_file() and out_yaml.stat().st_size > 0
            and out_npz.is_file() and out_npz.stat().st_size > 0):
        print(f"[fc2:{tag}] validated artifacts exist -> skip", flush=True)
        return 0
    if out_yaml.exists() or out_npz.exists():
        print(f"[fc2:{tag}] partial artifact detected -> recomputing both outputs", flush=True)
    work = Path(a.scratch) / "fc2" / tag
    work.mkdir(parents=True, exist_ok=True)
    pseudos = tc.pseudo_map(mat, a.pseudo_dir)

    atoms = tc.build_tmd(mat["formula"], mat["polytype"], lattice_a, thickness)
    atoms.wrap()
    print(f"[fc2:{tag}] material={a.name} {mat['polytype']}-{mat['formula']} a={lattice_a} "
          f"thickness={thickness} "
          f"sc {d['fc2_supercell']}x{d['fc2_supercell']}x1 ecut={d['ecutwfc']} "
          f"k={d['fc2_kpts']} degauss={d['degauss']} smearing={a.smearing} "
          f"pseudos={pseudos}", flush=True)

    n = d["fc2_supercell"]
    phon = PhononCalculation(atoms, supercell_matrix=np.diag([n, n, 1]),
                             primitive_matrix=np.eye(3), displacement=d["fc2_disp"])
    nd = phon.n_displacements
    print(f"[fc2:{tag}] {nd} displaced supercell(s), {len(phon.displaced_supercells[0])} atoms; running QE ...", flush=True)
    t0 = time.perf_counter()
    forces = []
    for i, scell in enumerate(phon.displaced_supercells):
        dd = work / f"disp-{i:03d}"; dd.mkdir(parents=True, exist_ok=True)
        scell.calc = make_espresso(a.pw, a.mpirun, a.nproc, a.pseudo_dir, pseudos,
                                   d["ecutwfc"], d["ecutrho"], d["fc2_kpts"],
                                   d["degauss"], dd, a.smearing)
        forces.append(scell.get_forces())
        print(f"[fc2:{tag}]   disp {i+1}/{nd}: max|F|={np.abs(forces[-1]).max():.4f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    phon.set_forces(np.array(forces))
    phon.produce_force_constants(symmetrize=True)
    ph = phon.phonon
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml_tmp = out_yaml.with_name(out_yaml.name + ".tmp")
    npz_tmp = out_npz.with_name(out_npz.name + ".tmp")
    ph.save(filename=str(yaml_tmp), settings={"force_constants": True})

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
    with npz_tmp.open("wb") as handle:
        np.savez(handle, distances=dist, frequencies=freq,
                 label_positions=lp, labels=np.array(labels), a=lattice_a,
                 thickness=thickness, material=np.array(a.name), tag=np.array(tag),
                 formula=np.array(mat["formula"]), polytype=np.array(mat["polytype"]),
                 degauss=float(d["degauss"]), smearing=np.array(a.smearing),
                 supercell=int(d["fc2_supercell"]), ecutwfc=float(d["ecutwfc"]),
                 ecutrho=float(d["ecutrho"]), kpts=int(d["fc2_kpts"]),
                 displacement=float(d["fc2_disp"]))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(npz_tmp, out_npz)
    os.replace(yaml_tmp, out_yaml)
    fmin = float(freq.min())
    n_imag = int((freq < -0.1).sum())
    print(f"[fc2:{tag}] min freq = {fmin:.3f} THz, n_imag(<-0.1) = {n_imag} -> "
          f"{'SOFT MODE (CDW captured)' if fmin < -0.1 else 'stable'}; saved {out_yaml.name}", flush=True)
    shutil.rmtree(work, ignore_errors=True)     # free the QE scratch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
