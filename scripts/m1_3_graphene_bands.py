"""M1.3 (E-channel): graphene electronic band structure via QE -> confirm the
Dirac point at K and the q* ~ 2k_F geometry of the phonon Kohn anomalies.

For graphene the Fermi surface collapses to the two Dirac points (K, K'). The two
phonon Kohn anomalies map to electron-phonon scattering across it:
  - Gamma-E2g  <-> intra-valley  (q -> 0, within one Dirac cone)
  - K-A1'      <-> inter-valley  (q = K, connecting K <-> K')
So q* = {Gamma, K} are exactly the 2k_F (here k_F at the Dirac point) connectors.
This confirms the anomalies are electron-phonon in origin -- the (E) channel the
MLIP cannot itself produce.

    python scripts/m1_3_graphene_bands.py --pw .../pw.x --mpirun .../mpirun \
        --nproc 8 --pseudo-dir pseudo --a 2.46 --nk 24 --npoints 180
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


def espresso(pw, mpirun, nproc, pseudo_dir, pseudo, ecutwfc, ecutrho, degauss,
             calculation, directory, kpts=None, koffset=(0, 0, 0)):
    from ase.calculators.espresso import Espresso, EspressoProfile

    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pseudo_dir))
    input_data = {
        "control": {"calculation": calculation, "prefix": "gr", "disk_io": "low",
                    "verbosity": "high"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "cold", "degauss": degauss, "nbnd": 12},
        "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.4},
    }
    return Espresso(profile=profile, pseudopotentials={"C": pseudo},
                    input_data=input_data, kpts=kpts, koffset=koffset,
                    directory=Path(directory))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--pseudo-dir", default="pseudo")
    ap.add_argument("--pseudo", default="C_ONCV_PBE-1.2.upf")
    ap.add_argument("--a", type=float, default=2.46)
    ap.add_argument("--ecutwfc", type=float, default=60.0)
    ap.add_argument("--ecutrho", type=float, default=240.0)
    ap.add_argument("--degauss", type=float, default=0.01)
    ap.add_argument("--nk", type=int, default=24, help="scf k-mesh (nk,nk,1)")
    ap.add_argument("--npoints", type=int, default=180)
    ap.add_argument("--workdir", default="results/m1_3")
    a = ap.parse_args()

    pdir = (ROOT / a.pseudo_dir) if not Path(a.pseudo_dir).is_absolute() else Path(a.pseudo_dir)
    work = ROOT / a.workdir
    work.mkdir(parents=True, exist_ok=True)
    atoms = tdc.build_monolayer("graphene", a=a.a)
    atoms.wrap()

    # 1) SCF
    print(f"[m1.3] graphene a={a.a} A; SCF on {a.nk}x{a.nk} k ...", flush=True)
    t0 = time.perf_counter()
    scf = espresso(a.pw, a.mpirun, a.nproc, pdir, a.pseudo, a.ecutwfc, a.ecutrho,
                   a.degauss, "scf", work / "scf", kpts=(a.nk, a.nk, 1))
    atoms.calc = scf
    atoms.get_potential_energy()
    efermi = scf.get_fermi_level()
    print(f"[m1.3] SCF done ({time.perf_counter()-t0:.0f}s), E_F={efermi:.3f} eV", flush=True)

    # 2) bands along M-Gamma-K-M  (explicit reduced-coord path)
    M, G, K = [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0 / 3, 1.0 / 3, 0.0]
    seg = []
    pts = [M, G, K, M]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, a.npoints, endpoint=False):
            seg.append([p0[i] + t * (p1[i] - p0[i]) for i in range(3)])
    seg.append(M)
    kpath = np.array(seg)
    kpath4 = np.column_stack([kpath, np.ones(len(kpath))])  # ASE wants Nx4 (coords+weight)
    print(f"[m1.3] bands on {len(kpath)} k-points (M-G-K-M) ...", flush=True)
    bands = espresso(a.pw, a.mpirun, a.nproc, pdir, a.pseudo, a.ecutwfc, a.ecutrho,
                     a.degauss, "bands", work / "scf", kpts=kpath4)
    atoms.calc = bands
    from ase.calculators.calculator import PropertyNotImplementedError
    try:
        atoms.get_potential_energy()  # 'bands' calc has no total energy -> expected
    except PropertyNotImplementedError:
        pass

    # parse eigenvalues
    nk = len(kpath)
    E = np.array([bands.get_eigenvalues(spin=0, kpt=i) for i in range(nk)]) - efermi

    # cumulative k-distance + label positions (M,G,K,M segment ends)
    rec = atoms.cell.reciprocal() * 2 * np.pi
    kc = kpath @ rec
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
    nseg = a.npoints
    labpos = [dist[0], dist[nseg], dist[2 * nseg], dist[-1]]

    # gap at K (Dirac): min over bands of |E| at the K index (end of 2nd segment)
    iK = 2 * nseg
    gapK = float(np.min(np.abs(E[iK])))
    iG = nseg
    print(f"[m1.3] |E| closest-to-EF at K = {gapK*1000:.1f} meV (Dirac point ~ 0)", flush=True)
    print(f"[m1.3] q* geometry: Gamma-E2g = intra-valley (q->0); K-A1' = inter-valley "
          f"(q=K connects K<->K'); both = 2k_F connectors of the Dirac points", flush=True)

    out = work / "graphene_ebands.npz"
    np.savez(out, dist=dist, E=E, label_positions=np.array(labpos),
             labels=np.array(["M", "$\\Gamma$", "K", "M"]), efermi=efermi, gapK=gapK, a=a.a)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for b in range(E.shape[1]):
        ax.plot(dist, E[:, b], color="#3b6ea5", lw=1.2)
    ax.axhline(0, color="0.5", ls=":", lw=0.9)
    for x in labpos:
        ax.axvline(x, color="0.7", lw=0.5)
    ax.scatter([labpos[2]], [0], s=60, color="#d1495b", zorder=5, label="Dirac point (K)")
    ax.annotate("K-A$_1'$ anomaly\n= inter-valley", (labpos[2], 0),
                textcoords="offset points", xytext=(6, 12), fontsize=8, color="#d1495b")
    ax.annotate("$\\Gamma$-E$_{2g}$\n= intra-valley", (labpos[1], 0),
                textcoords="offset points", xytext=(6, 12), fontsize=8, color="#2a9d4a")
    ax.set_xticks(labpos); ax.set_xticklabels(["M", "$\\Gamma$", "K", "M"])
    ax.set_xlim(dist[0], dist[-1]); ax.set_ylim(-12, 12)
    ax.set_ylabel("E - E$_F$ (eV)")
    ax.set_title("Graphene electronic bands: Dirac point at K = 2k$_F$ of the anomalies")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    figp = ROOT / "results/figures/m1_3_graphene_ebands.png"
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=200); fig.savefig(figp.with_suffix(".pdf"))
    print(f"[m1.3] gap at K = {gapK*1000:.1f} meV -> {out.relative_to(ROOT)}, {figp.relative_to(ROOT)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
