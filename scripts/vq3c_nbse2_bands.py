"""B (V-Q3 cross-check): NbSe2 electronic bands -> Fermi-surface 2k_F vs the CDW
soft-mode q. Tests the plan's core hypothesis q* ~ 2k_F: the monolayer 2H-NbSe2
CDW is 3x3, i.e. ordering wavevector q_CDW = (1/3,0) = 2/3 of Gamma-M. If a band
crosses E_F at k_F ~ 1/3 of Gamma-M, then 2k_F ~ 2/3 Gamma-M = q_CDW -> the soft
mode is a genuine Fermi-surface-nesting (Kohn/CDW) anomaly, not a generic artifact.

    python scripts/vq3c_nbse2_bands.py --pw /root/gpupw.sh --nproc 1 \
        --pseudo-dir pseudo --a 3.44 --nk 18 --npoints 200
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


def espresso(pw, mpirun, nproc, pdir, ecutwfc, ecutrho, degauss, calculation,
             directory, kpts=None, nbnd=28):
    from ase.calculators.espresso import Espresso, EspressoProfile
    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pdir))
    input_data = {
        "control": {"calculation": calculation, "prefix": "nbse2", "disk_io": "low",
                    "verbosity": "high"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "cold", "degauss": degauss, "nbnd": nbnd},
        "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3, "electron_maxstep": 250,
                      "diago_david_ndim": 4, "startingwfc": "atomic+random"},
    }
    return Espresso(profile=profile, pseudopotentials=PSEUDOS, input_data=input_data,
                    kpts=kpts, directory=Path(directory))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--pseudo-dir", default="pseudo")
    ap.add_argument("--a", type=float, default=3.44)
    ap.add_argument("--thickness", type=float, default=3.34)
    ap.add_argument("--ecutwfc", type=float, default=70.0)
    ap.add_argument("--ecutrho", type=float, default=280.0)
    ap.add_argument("--degauss", type=float, default=0.015)
    ap.add_argument("--nk", type=int, default=18, help="scf k-mesh (nk,nk,1)")
    ap.add_argument("--npoints", type=int, default=200, help="k-points per Gamma-M-K-Gamma segment")
    ap.add_argument("--nbnd", type=int, default=28)
    ap.add_argument("--workdir", default="results/vq3c")
    a = ap.parse_args()
    mp = a.mpirun or "mpirun"
    pdir = (ROOT / a.pseudo_dir) if not Path(a.pseudo_dir).is_absolute() else Path(a.pseudo_dir)
    work = ROOT / a.workdir
    work.mkdir(parents=True, exist_ok=True)
    atoms = tdc.build_monolayer("nbse2", a=a.a, thickness=a.thickness)
    atoms.wrap()

    # 1) SCF
    print(f"[B] NbSe2 a={a.a}; SCF {a.nk}x{a.nk} k, degauss={a.degauss} ...", flush=True)
    t0 = time.perf_counter()
    scf = espresso(a.pw, mp, a.nproc, pdir, a.ecutwfc, a.ecutrho, a.degauss, "scf",
                   work / "scf", kpts=(a.nk, a.nk, 1), nbnd=a.nbnd)
    atoms.calc = scf
    atoms.get_potential_energy()
    efermi = scf.get_fermi_level()
    print(f"[B] SCF done ({time.perf_counter()-t0:.0f}s), E_F={efermi:.3f} eV", flush=True)

    # 2) bands on Gamma-M-K-Gamma (explicit reduced-coord path)
    G, M, K = [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0 / 3, 1.0 / 3, 0.0]
    pts = [G, M, K, G]
    seg = []
    for p0, p1 in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, a.npoints, endpoint=False):
            seg.append([p0[i] + t * (p1[i] - p0[i]) for i in range(3)])
    seg.append(G)
    kpath = np.array(seg)
    kpath4 = np.column_stack([kpath, np.ones(len(kpath))])
    print(f"[B] bands on {len(kpath)} k-points (G-M-K-G) ...", flush=True)
    bands = espresso(a.pw, mp, a.nproc, pdir, a.ecutwfc, a.ecutrho, a.degauss, "bands",
                     work / "scf", kpts=kpath4, nbnd=a.nbnd)
    atoms.calc = bands
    from ase.calculators.calculator import PropertyNotImplementedError
    try:
        atoms.get_potential_energy()
    except PropertyNotImplementedError:
        pass
    nk = len(kpath)
    E = np.array([bands.get_eigenvalues(spin=0, kpt=i) for i in range(nk)]) - efermi

    rec = atoms.cell.reciprocal() * 2 * np.pi
    kc = kpath @ rec
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
    labpos = [dist[0], dist[a.npoints], dist[2 * a.npoints], dist[-1]]

    # --- k_F on the Gamma-M segment: sign changes of (E - E_F) per band ---
    nseg = a.npoints
    seg_frac = np.linspace(0, 1, nseg, endpoint=False)  # fraction of Gamma-M
    crossings = []
    for b in range(E.shape[1]):
        eb = E[:nseg + 1, b]  # Gamma..M inclusive
        for i in range(nseg):
            if eb[i] == 0 or (eb[i] < 0) != (eb[i + 1] < 0):
                # linear interp fraction of Gamma-M where E crosses 0
                f = i / nseg
                if eb[i + 1] != eb[i]:
                    f = (i + eb[i] / (eb[i] - eb[i + 1])) / nseg
                crossings.append(round(f, 3))
    crossings = sorted(set(crossings))
    print(f"[B] E_F crossings along Gamma-M (fraction of |GM|): {crossings}", flush=True)
    print("[B] q_CDW (3x3 monolayer NbSe2) = 1/3 b1 = 2/3 of Gamma-M = 0.667", flush=True)
    for kf in crossings:
        print(f"[B]   k_F={kf:.3f} GM  ->  2k_F={2*kf:.3f} GM   (match 0.667 = CDW q?  "
              f"{'YES' if abs(2*kf-0.667)<0.12 or abs(min(2*kf,2-2*kf)-0.667)<0.12 else 'no'})",
              flush=True)

    out = work / "nbse2_ebands.npz"
    np.savez(out, dist=dist, E=E, label_positions=np.array(labpos),
             labels=np.array(["$\\Gamma$", "M", "K", "$\\Gamma$"]), efermi=efermi,
             kF_GM=np.array(crossings), a=a.a)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for b in range(E.shape[1]):
        ax.plot(dist, E[:, b], color="#3b6ea5", lw=1.0)
    ax.axhline(0, color="0.5", ls=":", lw=0.9)
    for x in labpos:
        ax.axvline(x, color="0.7", lw=0.5)
    for kf in crossings:
        ax.scatter([dist[int(kf * nseg)]], [0], s=35, color="#d1495b", zorder=5)
    ax.set_xticks(labpos); ax.set_xticklabels(["Γ", "M", "K", "Γ"])
    ax.set_xlim(dist[0], dist[-1]); ax.set_ylim(-6, 6)
    ax.set_ylabel("E − E$_F$ (eV)")
    ax.set_title("NbSe$_2$ bands: Fermi crossings on Γ-M → 2k$_F$ vs CDW q (2/3 ΓM)")
    fig.tight_layout()
    figp = ROOT / "results/figures/nbse2_ebands.png"
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=180)
    print(f"[B] saved -> {out.relative_to(ROOT)}, {figp.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
