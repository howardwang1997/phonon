"""Verify: does proper structural relaxation + reference-force subtraction
reduce the deep imaginary CDW soft modes in the TMD DFT fc2?

The production fc2 (tmd_dft_fc2.py) is computed on an *ideal* ASE monolayer at
the config lattice constants, with NO vc-relax and the undisplaced (reference)
forces NOT subtracted. Both can inflate the imaginary-mode depth. This script
recomputes fc2 the "correct" way and compares:

  1. tight vc-relax (atoms + in-plane cell, cell_dofree='2Dxy') at dg0.015;
  2. rebuild a clean symmetric primitive at the RELAXED (a, thickness);
  3. fc2 on the SAME supercell as the existing data, at dg0.005 + dg0.020,
     WITH the undisplaced-supercell reference forces F0 subtracted;
  4. soft-mode metrics (min freq on Gamma-M-K-Gamma, mesh imag %, Gamma ASR),
     compared side-by-side against the existing UNRELAXED fc2.

    python scripts/v100/verify_relax_fc2.py --name NbSe2 \
        --pw /root/gpupw.sh --nproc 1 --pseudo-dir /root/phonon/pseudo \
        --smearings 0.005 0.020
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "v100"))

import numpy as np  # noqa: E402
import tmd_common as tc  # noqa: E402

CM = 33.35641
POINTS = [np.array([0., 0., 0.]), np.array([.5, 0., 0.]),
          np.array([1/3, 1/3, 0.]), np.array([0., 0., 0.])]

# (supercell_n, existing-unrelaxed yaml template) per config material name.
# supercell matches the existing data so relaxed vs unrelaxed are comparable.
REF = {
    "NbSe2":    (3, "results/v100/fc2_nbse2_3x3_{dg}/NbSe2_phonopy.yaml"),
    "NbS2":     (3, "results/v100/fc2_nbs2_3x3_{dg}/NbS2_phonopy.yaml"),
    "2H-TaSe2": (3, "results/v100/fc2_tase2_3x3_{dg}/2H-TaSe2_phonopy.yaml"),
    "1T-VSe2":  (4, "results/vq_family/vse2/1T-VSe2_dg{dg}.yaml"),
}


def make_espresso(pw, mpirun, nproc, pseudo_dir, pseudos, ecutwfc, ecutrho,
                  kpts, degauss, directory, calculation="scf", conv_thr=1e-8,
                  disk_io="none"):
    """QE calculator. calculation='vc-relax' adds the cell block + tstress +
    disk_io='low' (relax must read/write charge between ionic steps)."""
    from ase.calculators.espresso import Espresso, EspressoProfile
    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pseudo_dir))
    control = {"calculation": calculation, "tprnfor": True, "tstress": False,
               "disk_io": disk_io, "verbosity": "low"}
    input_data = {
        "control": control,
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho,
                   "occupations": "smearing", "smearing": "cold", "degauss": degauss},
        "electrons": {"conv_thr": conv_thr, "mixing_beta": 0.3,
                      "electron_maxstep": 250, "diago_david_ndim": 4,
                      "startingwfc": "atomic+random"},
    }
    if calculation == "vc-relax":
        control.update(tstress=True, forc_conv_thr=1e-4)
        input_data["cell"] = {"cell_dofree": "2Dxy", "cell_dynamics": "bfgs",
                              "press_conv_thr": 0.5}
    return Espresso(profile=profile, pseudopotentials=pseudos,
                    input_data=input_data, kpts=(kpts, kpts, 1),
                    directory=Path(directory))


def vc_relax(atoms, calc_factory, work):
    """Run QE vc-relax; return relaxed ASE atoms (last frame of espresso.pwo)."""
    from ase.io import read
    d = work / "relax"; d.mkdir(parents=True, exist_ok=True)
    atoms = atoms.copy(); atoms.calc = calc_factory(d)
    atoms.get_potential_energy()             # drives the full vc-relax
    relaxed = read(d / "espresso.pwo", index=-1)
    return relaxed


def rebuild_clean(mat, relaxed):
    """Symmetric primitive at the RELAXED (a, thickness), per the project rebuild
    convention (avoids the vc-relax in-plane shear tripping downstream tools)."""
    cell = np.array(relaxed.cell)
    a = float(np.linalg.norm(cell[0]))
    zs = relaxed.get_positions()[:, 2]
    syms = relaxed.get_chemical_symbols()
    x = [z for z, s in zip(zs, syms) if s != mat["M"]]
    thickness = float(max(x) - min(x))
    print(f"  relaxed a={a:.4f} (start {mat['a']})  thickness={thickness:.4f} "
          f"(start {mat['thickness']})", flush=True)
    return tc.build_tmd(mat["formula"], mat["polytype"], a, thickness), a, thickness


def fc2_with_refsub(primitive_atoms, calc_factory, n, disp, work):
    """Finite-displacement fc2 with the undisplaced-supercell reference forces
    subtracted (F0), ASR-symmetrized. Returns the phonopy object."""
    from phonon_accel.phonons import PhononCalculation, phonopy_to_ase
    phon = PhononCalculation(primitive_atoms, supercell_matrix=np.diag([n, n, 1]),
                             primitive_matrix=np.eye(3), displacement=disp)
    # F0 on the undisplaced supercell
    at0 = phonopy_to_ase(phon.phonon.supercell)
    rd = work / "ref"; rd.mkdir(parents=True, exist_ok=True)
    at0.calc = calc_factory(rd)
    F0 = np.array(at0.get_forces())
    print(f"  F0 (ref) max|F|={np.abs(F0).max():.4f} eV/A", flush=True)
    forces = []
    for i, scell in enumerate(phon.displaced_supercells):
        dd = work / f"disp-{i:03d}"; dd.mkdir(parents=True, exist_ok=True)
        scell.calc = calc_factory(dd)
        forces.append(np.array(scell.get_forces()) - F0)
    phon.set_forces(np.array(forces))
    phon.produce_force_constants(symmetrize=True)
    return phon.phonon


def metrics(ph):
    """(soft_min_cm on Gamma-M-K-Gamma, mesh_imag%, gamma 3-lowest cm^-1)."""
    qs = []
    for i in range(len(POINTS) - 1):
        for j in range(1, 41):
            qs.append(POINTS[i] + (POINTS[i + 1] - POINTS[i]) * j / 40)
    qs = np.array(qs)
    ph.run_qpoints(qs, with_dynamical_matrices=False)
    freq = np.array(ph.get_qpoints_dict()["frequencies"]) * CM
    soft_min = float(freq.min())
    ph.run_mesh([12, 12, 1])
    md = ph.get_mesh_dict()["frequencies"] * CM
    mesh_imag_pct = float((md < -0.5).mean() * 100)
    ph.run_qpoints([np.zeros(3)], with_dynamical_matrices=False)
    g = np.sort(np.array(ph.get_qpoints_dict()["frequencies"])[0] * CM)
    return soft_min, mesh_imag_pct, [float(x) for x in g[:3]]


def metrics_from_yaml(path):
    import phonopy
    ph = phonopy.load(str(path), is_compact_fc=False)
    return metrics(ph)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="configs/v100_campaign.yaml")
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--pseudo-dir", default="/root/phonon/pseudo")
    ap.add_argument("--smearings", nargs="+", default=["0.005", "0.020"])
    ap.add_argument("--scratch", default="/data/v100scratch")
    ap.add_argument("--out-dir", default="results/v100/verify_relax")
    a = ap.parse_args()

    from phonon_accel.phonons import phonopy_to_ase  # noqa: F401 (import check)

    cfg = tc.load_config(a.config)
    mat = tc.material(cfg, a.name)
    d = cfg["dft"]
    pseudos = tc.pseudo_map(mat, a.pseudo_dir)
    n_sup, ref_tmpl = REF[a.name]
    smearings = [float(s) for s in a.smearings]
    out_dir = ROOT / a.out_dir / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(a.scratch) / "verify_relax" / a.name

    print(f"[verify:{a.name}] {mat['polytype']}-{mat['formula']} sc={n_sup}x "
          f"ecut={d['ecutwfc']} k={d['fc2_kpts']} smearings={smearings}", flush=True)

    def scf_factory(dg):
        def f(directory):
            return make_espresso(a.pw, a.mpirun, a.nproc, a.pseudo_dir, pseudos,
                                 d["ecutwfc"], d["ecutrho"], d["fc2_kpts"], dg,
                                 directory)
        return f

    # ---- 1. vc-relax (atoms + 2Dxy cell) at the campaign degauss ----
    t0 = time.perf_counter()
    atoms0 = tc.build_tmd(mat["formula"], mat["polytype"], mat["a"], mat["thickness"])
    atoms0.wrap()
    relaxed = vc_relax(atoms0, lambda dd: make_espresso(
        a.pw, a.mpirun, a.nproc, a.pseudo_dir, pseudos, d["ecutwfc"], d["ecutrho"],
        d["fc2_kpts"], d["degauss"], dd, calculation="vc-relax", conv_thr=1e-9,
        disk_io="low"), scratch)
    prim_clean, a_rel, th_rel = rebuild_clean(mat, relaxed)
    prim_clean.wrap()
    print(f"[verify:{a.name}] relax done ({time.perf_counter()-t0:.0f}s)", flush=True)

    summary = {"name": a.name, "relaxed_a": a_rel, "relaxed_thickness": th_rel,
               "start_a": mat["a"], "start_thickness": mat["thickness"],
               "smearings": {}}

    # ---- 2. fc2 (ref-subtracted) per smearing + compare to existing unrelaxed ----
    for dg in smearings:
        print(f"[verify:{a.name}] === fc2 dg={dg} (ref-subtracted) ===", flush=True)
        t1 = time.perf_counter()
        work = scratch / f"fc2_dg{dg}"; work.mkdir(parents=True, exist_ok=True)
        ph = fc2_with_refsub(prim_clean, scf_factory(dg), n_sup, d["fc2_disp"], work)
        r_soft, r_mesh, r_g = metrics(ph)
        ph.save(filename=str(out_dir / f"dg{dg}_relaxed_phonopy.yaml"),
                settings={"force_constants": True})
        ref_yaml = ROOT / ref_tmpl.format(dg=dg)
        if ref_yaml.exists():
            u_soft, u_mesh, u_g = metrics_from_yaml(ref_yaml)
        else:
            u_soft = u_mesh = None; u_g = None
        summary["smearings"][str(dg)] = {
            "relaxed": {"soft_min_cm": r_soft, "mesh_imag_pct": r_mesh,
                        "gamma_acoustic_cm": r_g},
            "unrelaxed": {"soft_min_cm": u_soft, "mesh_imag_pct": u_mesh,
                          "gamma_acoustic_cm": u_g},
        }
        print(f"  dg{dg}: relaxed soft_min={r_soft:7.1f} cm^-1  mesh_imag={r_mesh:4.1f}%  "
              f"| unrelaxed soft_min={u_soft}  mesh_imag={u_mesh}  "
              f"({time.perf_counter()-t1:.0f}s)", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[verify:{a.name}] wrote {out_dir/'summary.json'}", flush=True)
    print(f"[verify:{a.name}] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
