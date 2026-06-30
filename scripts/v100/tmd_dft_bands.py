"""TMD-family electronic anchor — generalizes scripts/vq3c_nbse2_bands.py.

Two (E)-channel diagnostics per material, both cheap GPU pw.x:
  1. bands on G-M-K-G + Fermi crossings on G-M -> 2k_F vs the CDW q (q* ~ 2k_F?)
  2. nesting function xi(q) along G-M from a dense regular k-grid -> does the
     Fermi-surface nesting peak AT q_CDW (nesting-driven) or NOT (EPC-driven)?
     (For NbSe2 it does NOT peak at q_CDW -> EPC-driven; see Rigor #2.)

Together they adjudicate the CDW mechanism per family member — the (E)-channel
half of the Paper-2 origin map.

    python scripts/v100/tmd_dft_bands.py --name NbS2 \
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


def espresso(pw, mpirun, nproc, pdir, pseudos, ecutwfc, ecutrho, degauss,
             calculation, directory, kpts, nbnd, prefix):
    from ase.calculators.espresso import Espresso, EspressoProfile
    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pdir))
    input_data = {
        "control": {"calculation": calculation, "prefix": prefix, "disk_io": "low",
                    "verbosity": "high"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "cold", "degauss": degauss, "nbnd": nbnd},
        "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3, "electron_maxstep": 250,
                      "diago_david_ndim": 4, "startingwfc": "atomic+random"},
    }
    return Espresso(profile=profile, pseudopotentials=pseudos, input_data=input_data,
                    kpts=kpts, directory=Path(directory))


def nesting_along_GM(E, nk, efermi, width=0.05, nq=None):
    """xi(q) for q=(j/nk,0,0) along G-M from eigenvalues E on an nk×nk grid.
    E shape (nk*nk, nbnd), already referenced so E_F=0. Gaussian double-delta."""
    nbnd = E.shape[1]
    Eg = E.reshape(nk, nk, nbnd)
    g = np.exp(-(Eg / width) ** 2) / (np.sqrt(np.pi) * width)   # delta(eps-EF)
    nq = nq or (nk // 2 + 1)
    xi = np.zeros(nq)
    for j in range(nq):
        gshift = np.roll(g, -j, axis=0)              # k -> k+q along b1
        xi[j] = float((g * gshift).sum() / (nk * nk))
    qfrac = np.arange(nq) / nk                        # fraction of b1 (G..M is 0..0.5)
    return qfrac, xi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="configs/v100_campaign.yaml")
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--pseudo-dir", default="/root/phonon/pseudo")
    ap.add_argument("--nk", type=int, default=18, help="scf k-mesh")
    ap.add_argument("--nk-dense", type=int, default=24, help="dense grid for nesting")
    ap.add_argument("--npoints", type=int, default=200)
    ap.add_argument("--nbnd", type=int, default=28)
    ap.add_argument("--workdir", default="results/v100/bands")
    ap.add_argument("--scratch", default="/data/v100scratch",
                    help="big QE scratch root (keep OFF the root fs)")
    a = ap.parse_args()
    import shutil

    cfg = tc.load_config(a.config)
    mat = tc.material(cfg, a.name)
    dft = cfg["dft"]
    out = ROOT / a.workdir / f"{a.name}_ebands.npz"
    if out.exists():
        print(f"[bands:{a.name}] {out.name} exists -> skip", flush=True)
        return 0
    work = Path(a.scratch) / "bands" / a.name
    work.mkdir(parents=True, exist_ok=True)
    pseudos = tc.pseudo_map(mat, a.pseudo_dir)
    pref = a.name.replace("-", "")
    atoms = tc.build_tmd(mat["formula"], mat["polytype"], mat["a"], mat["thickness"])
    atoms.wrap()
    ec, er, dg = dft["ecutwfc"], dft["ecutrho"], dft["degauss"]

    # 1) SCF
    print(f"[bands:{a.name}] SCF {a.nk}x{a.nk} ...", flush=True)
    t0 = time.perf_counter()
    scf = espresso(a.pw, a.mpirun, a.nproc, a.pseudo_dir, pseudos, ec, er, dg, "scf",
                   work / "scf", (a.nk, a.nk, 1), a.nbnd, pref)
    atoms.calc = scf
    atoms.get_potential_energy()
    efermi = scf.get_fermi_level()
    print(f"[bands:{a.name}] SCF done ({time.perf_counter()-t0:.0f}s), E_F={efermi:.3f} eV", flush=True)

    # 2) bands on G-M-K-G
    G, M, K = [0., 0., 0.], [0.5, 0., 0.], [1/3, 1/3, 0.]
    pts = [G, M, K, G]
    seg = []
    for p0, p1 in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, a.npoints, endpoint=False):
            seg.append([p0[i] + t * (p1[i] - p0[i]) for i in range(3)])
    seg.append(G)
    kpath = np.array(seg)
    kpath4 = np.column_stack([kpath, np.ones(len(kpath))])
    bands = espresso(a.pw, a.mpirun, a.nproc, a.pseudo_dir, pseudos, ec, er, dg, "bands",
                     work / "scf", kpath4, a.nbnd, pref)
    atoms.calc = bands
    from ase.calculators.calculator import PropertyNotImplementedError
    try:
        atoms.get_potential_energy()
    except PropertyNotImplementedError:
        pass
    nk = len(kpath)
    E = np.array([bands.get_eigenvalues(spin=0, kpt=i) for i in range(nk)]) - efermi
    rec = atoms.cell.reciprocal() * 2 * np.pi
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(kpath @ rec, axis=0), axis=1))])
    labpos = [dist[0], dist[a.npoints], dist[2 * a.npoints], dist[-1]]

    nseg = a.npoints
    crossings = []
    for b in range(E.shape[1]):
        eb = E[:nseg + 1, b]
        for i in range(nseg):
            if eb[i] == 0 or (eb[i] < 0) != (eb[i + 1] < 0):
                f = (i + eb[i] / (eb[i] - eb[i + 1])) / nseg if eb[i + 1] != eb[i] else i / nseg
                crossings.append(round(f, 3))
    crossings = sorted(set(crossings))
    print(f"[bands:{a.name}] E_F crossings on G-M (frac |GM|): {crossings}; "
          f"q_CDW~0.667 -> 2k_F match? " +
          ", ".join(f"{c}->{2*c:.2f}{'*' if abs(min(2*c,2-2*c)-0.667)<0.12 else ''}" for c in crossings), flush=True)

    # 3) nesting xi(q) along G-M from a dense regular grid (best-effort)
    qfrac = xi = None
    try:
        print(f"[bands:{a.name}] nesting: NSCF {a.nk_dense}x{a.nk_dense} ...", flush=True)
        nscf = espresso(a.pw, a.mpirun, a.nproc, a.pseudo_dir, pseudos, ec, er, dg, "nscf",
                        work / "scf", (a.nk_dense, a.nk_dense, 1), a.nbnd, pref)
        atoms.calc = nscf
        try:
            atoms.get_potential_energy()
        except PropertyNotImplementedError:
            pass
        nkd = a.nk_dense
        Ed = np.array([nscf.get_eigenvalues(spin=0, kpt=i) for i in range(nkd * nkd)]) - efermi
        win = np.abs(Ed).max(axis=0) < 2.0       # bands within ±2 eV of E_F
        qfrac, xi = nesting_along_GM(Ed[:, win], nkd, efermi, width=2 * 0.0136 * 1.0)
        ipk = int(np.argmax(xi[1:]) + 1)
        print(f"[bands:{a.name}] nesting xi(q) peaks at q={qfrac[ipk]:.3f} GM "
              f"(q_CDW~0.333 of b1); {'NESTING-DRIVEN' if abs(qfrac[ipk]-1/3)<0.06 else 'NOT nesting-driven (EPC?)'}", flush=True)
    except Exception as exc:
        print(f"[bands:{a.name}] nesting step skipped ({type(exc).__name__}: {exc})", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, dist=dist, E=E, label_positions=np.array(labpos),
             labels=np.array(["G", "M", "K", "G"]), efermi=efermi,
             kF_GM=np.array(crossings),
             nest_q=(qfrac if qfrac is not None else np.array([])),
             nest_xi=(xi if xi is not None else np.array([])), a=mat["a"])
    shutil.rmtree(work, ignore_errors=True)     # free the QE scratch
    print(f"[bands:{a.name}] saved -> {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
