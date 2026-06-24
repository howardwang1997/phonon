"""M1.2: temperature-dependent phonon dispersion omega(q,T) of a 2D monolayer
via hiPhive-TDEP with MLIP forces -- the (L) lattice-anharmonic channel.

Per temperature: Langevin MD (MLIP forces) on a supercell -> decorrelated thermal
snapshots -> hiPhive effective harmonic fc2(T) -> phonopy dispersion on M-Gamma-K-M
-> track the top-branch frequency and Kohn-kink at Gamma/K. The MLIP learns the
*ground-state* BO PES, so this captures phonon-population / anharmonic
renormalization (L) but NOT the electronic Fermi-smearing of the anomaly (E);
every T-trend here is the (L) channel only.

Atom ordering follows phonopy's supercell so hiPhive fc2 maps straight onto the
phonopy object (the classic hiphive<->phonopy ordering trap).

    conda run -n phonon python scripts/td_phonon.py \
        --structure data/td_phonon/graphene.xyz \
        --model results/finetune_mace/ft_phonon.model \
        --temperatures 100,300,600 --supercell 6,6,1 --tag graphene_ft
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
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

import anomaly_locate as al
import td_common as tdc

CM = 33.35641


def sample_md(ideal, calc, T, dt_fs, n_equil, n_snap, stride, seed=0, log=None):
    """Langevin MD at T on a copy of ``ideal``; return decorrelated snapshots
    carrying their MLIP forces (SinglePointCalculator)."""
    at = ideal.copy()
    at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=T, rng=np.random.default_rng(seed))
    Stationary(at)
    ZeroRotation(at)
    dyn = Langevin(at, dt_fs * units.fs, temperature_K=T, friction=0.02,
                   rng=np.random.default_rng(seed + 1))
    dyn.run(n_equil)
    snaps, temps = [], []
    for _ in range(n_snap):
        dyn.run(stride)
        s = at.copy()
        s.calc = SinglePointCalculator(s, forces=at.get_forces())
        snaps.append(s)
        temps.append(at.get_temperature())
    if log is not None:
        log(f"    sampled {len(snaps)} snapshots, <T_inst>={np.mean(temps):.0f} K")
    return snaps


def effective_fc2(prim, ideal, sc_matrix, snaps, cutoff2):
    """hiPhive effective harmonic fit -> fc2 array in phonopy supercell order."""
    from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
    from hiphive.utilities import prepare_structures
    from trainstation import Optimizer

    cs = ClusterSpace(prim, [cutoff2])
    structures = prepare_structures(snaps, ideal)
    sc = StructureContainer(cs)
    for s in structures:
        sc.add_structure(s)
    opt = Optimizer(sc.get_fit_data(), train_size=1.0)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)
    fcs = fcp.get_force_constants(ideal)
    fc2 = fcs.get_fc_array(order=2)  # (Nsc, Nsc, 3, 3), phonopy order
    return fc2, float(opt.rmse_train), cs.number_of_dofs


def dispersion_from_fc2(prim_ase, sc_matrix, fc2, path="MGKM", npoints=201):
    from phonon_accel.phonons import ase_to_phonopy
    from phonopy import Phonopy

    ph = Phonopy(ase_to_phonopy(prim_ase), supercell_matrix=sc_matrix,
                 primitive_matrix=np.eye(3))
    ph.force_constants = fc2
    qpoints, connections, labels = tdc.make_band_path(path, npoints=npoints)
    ph.run_band_structure(qpoints, path_connections=connections, labels=labels,
                          with_eigenvectors=False)
    bsd = ph.get_band_structure_dict()
    dist = np.concatenate([np.asarray(d) for d in bsd["distances"]])
    freq = np.concatenate([np.asarray(f) for f in bsd["frequencies"]], axis=0)
    seg_d = [np.asarray(d) for d in bsd["distances"]]
    label_pos = np.array([seg_d[0][0]] + [s[-1] for s in seg_d])
    keep = np.concatenate([[True], np.diff(dist) > 1e-9])
    return dist[keep], freq[keep], label_pos, np.array(labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperatures", default="100,300,600")
    ap.add_argument("--supercell", default="6,6,1")
    ap.add_argument("--cutoff2", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=1.0, help="MD timestep (fs)")
    ap.add_argument("--equil", type=int, default=1500, help="equilibration steps")
    ap.add_argument("--nsnap", type=int, default=120, help="snapshots per T")
    ap.add_argument("--stride", type=int, default=40, help="MD steps between snapshots")
    ap.add_argument("--npoints", type=int, default=201)
    ap.add_argument("--outdir", default="results/td_phonon")
    a = ap.parse_args()

    def log(m):
        print(m, flush=True)

    sc = tuple(int(x) for x in a.supercell.split(","))
    sc_matrix = np.diag(sc)
    temps = [float(x) for x in a.temperatures.split(",")]

    at0 = read(ROOT / a.structure)
    log(f"[{a.tag}] {a.structure} ({len(at0)} atoms); model={a.model} on {a.device}")
    calc = tdc.get_mace_calc(a.model, device=a.device)

    # relax -> primitive; build phonopy supercell ordering as the MD reference
    prim, info = tdc.relax_monolayer(at0, calc)
    log(f"[{a.tag}] relaxed a={info['a']:.4f} A")
    from phonon_accel.phonons import ase_to_phonopy, phonopy_to_ase
    from phonopy import Phonopy
    ph0 = Phonopy(ase_to_phonopy(prim), supercell_matrix=sc_matrix,
                  primitive_matrix=np.eye(3))
    ideal = phonopy_to_ase(ph0.supercell)
    prim_for_cs = phonopy_to_ase(ph0.primitive)
    log(f"[{a.tag}] MD supercell {sc} = {len(ideal)} atoms, cutoff2={a.cutoff2} A")

    out = {"temperatures": np.array(temps), "supercell": np.asarray(sc),
           "model": np.array(str(a.model)), "tag": np.array(str(a.tag))}
    rows = ["T_K,w_gamma_cm,w_k_cm,kink_gamma,kink_k,minfreq_thz,fit_rmse_meVA"]

    for T in temps:
        t0 = time.perf_counter()
        log(f"[{a.tag}] T={T:.0f} K: MD sampling ...")
        snaps = sample_md(ideal, calc, T, a.dt, a.equil, a.nsnap, a.stride, log=log)
        fc2, rmse, ndof = effective_fc2(prim_for_cs, ideal, sc_matrix, snaps, a.cutoff2)
        dist, freq, lp, labs = dispersion_from_fc2(prim_for_cs, sc_matrix, fc2,
                                                   npoints=a.npoints)
        wG = al.branch_freq_at_label(dist, freq, lp, labs, r"$\Gamma$") * CM
        wK = al.branch_freq_at_label(dist, freq, lp, labs, "K") * CM
        kinks = {k["label"]: k for k in al.high_sym_kinks(dist, freq, lp, labs)}
        kG, kK = kinks[r"$\Gamma$"]["kink_strength"], kinks["K"]["kink_strength"]
        key = f"T{int(T)}"
        out[f"{key}_dist"], out[f"{key}_freq"] = dist, freq
        out["label_positions"], out["labels"] = lp, labs
        rows.append(f"{T:.0f},{wG:.2f},{wK:.2f},{kG:.3f},{kK:.3f},"
                    f"{freq.min():.3f},{rmse*1000:.2f}")
        log(f"[{a.tag}] T={T:.0f} K: wG={wG:.1f} wK={wK:.1f} cm^-1  "
            f"kinkG={kG:.1f} kinkK={kK:.1f}  rmse={rmse*1000:.1f} meV/A  "
            f"({ndof} dofs, {time.perf_counter()-t0:.0f}s)")

    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    npz = outdir / f"td_{a.tag}.npz"
    np.savez(npz, **out)
    csv = outdir / f"td_{a.tag}.csv"
    csv.write_text("\n".join(rows) + "\n")
    log(f"[{a.tag}] wrote {npz.relative_to(ROOT)} + {csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
