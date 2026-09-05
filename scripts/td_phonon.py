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
import json
import os
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


class MDStabilityError(RuntimeError):
    """Raised before an unphysical trajectory is allowed into the fc2 fit."""


def _atomic_savez(path: Path, **arrays) -> None:
    """Write an NPZ without ever exposing a half-written checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load_checkpoint_snapshots(path: Path, ideal):
    if not path.is_file():
        return []
    with np.load(path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=float)
        forces = np.asarray(data["forces"], dtype=float)
        energies = np.asarray(data["energies"], dtype=float)
        cells = np.asarray(data["cells"], dtype=float)
        momenta = np.asarray(data["momenta"], dtype=float)
    if not (len(positions) == len(forces) == len(energies) == len(cells) == len(momenta)):
        raise RuntimeError(f"inconsistent snapshot checkpoint arrays in {path}")
    snaps = []
    for pos, force, energy, cell in zip(positions, forces, energies, cells):
        snap = ideal.copy()
        snap.set_cell(cell, scale_atoms=False)
        snap.set_positions(pos)
        snap.set_momenta(momenta[len(snaps)])
        properties = {"forces": force}
        if np.isfinite(energy):
            properties["energy"] = float(energy)
        snap.calc = SinglePointCalculator(snap, **properties)
        snaps.append(snap)
    return snaps


def _validate_md_state(
    atoms,
    target_temperature,
    max_temperature_factor,
    min_pair_distance,
    max_force,
    *,
    check_geometry=False,
):
    temperature = float(atoms.get_temperature())
    if not np.isfinite(temperature) or temperature > max_temperature_factor * target_temperature:
        raise MDStabilityError(
            f"unphysical instantaneous temperature {temperature:.6g} K "
            f"(target {target_temperature:.6g} K, limit "
            f"{max_temperature_factor * target_temperature:.6g} K)"
        )
    if not np.all(np.isfinite(atoms.positions)) or not np.all(
        np.isfinite(atoms.get_momenta())
    ):
        raise MDStabilityError("non-finite MD positions or momenta")
    if check_geometry:
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        observed_distance = float(np.min(distances))
        if not np.isfinite(observed_distance) or observed_distance < min_pair_distance:
            raise MDStabilityError(
                f"unphysical minimum pair distance {observed_distance:.6g} A "
                f"(limit {min_pair_distance:.6g} A)"
            )
        forces = np.asarray(atoms.get_forces(), dtype=float)
        observed_force = float(np.max(np.abs(forces)))
        if not np.all(np.isfinite(forces)) or observed_force > max_force:
            raise MDStabilityError(
                f"unphysical maximum force component {observed_force:.6g} eV/A "
                f"(limit {max_force:.6g} eV/A)"
            )
    return temperature


def sample_md(
    ideal,
    calc,
    T,
    dt_fs,
    n_equil,
    n_snap,
    stride,
    seed=0,
    log=None,
    checkpoint_dir=None,
    resume=True,
    checkpoint_every=1,
    max_temperature_factor=5.0,
    min_pair_distance=0.8,
    max_force=100.0,
):
    """Langevin MD at T on a copy of ``ideal``; return decorrelated snapshots
    carrying their forces (SinglePointCalculator).

    If ``checkpoint_dir`` is supplied, the atomic state, momenta, Langevin RNG
    state, phase counters, and completed snapshots are saved atomically.  A
    rerun with the same parameters resumes at the next MD step.  This is
    intentionally calculator-agnostic and is especially important for DFT-MD,
    where one force evaluation can take minutes.
    """
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    at = ideal.copy()
    at.calc = calc
    if checkpoint_dir is None:
        MaxwellBoltzmannDistribution(at, temperature_K=T, rng=np.random.default_rng(seed))
        Stationary(at)
        ZeroRotation(at)
        dyn = Langevin(at, dt_fs * units.fs, temperature_K=T, friction=0.02,
                       rng=np.random.default_rng(seed + 1))
        dyn.run(n_equil)
        _validate_md_state(
            at, T, max_temperature_factor, min_pair_distance, max_force,
            check_geometry=True,
        )
        snaps, temps = [], []
        for _ in range(n_snap):
            dyn.run(stride)
            temperature = _validate_md_state(
                at, T, max_temperature_factor, min_pair_distance, max_force,
                check_geometry=True,
            )
            s = at.copy()
            s.calc = SinglePointCalculator(s, forces=at.get_forces())
            snaps.append(s)
            temps.append(temperature)
        if log is not None:
            log(f"    sampled {len(snaps)} snapshots, <T_inst>={np.mean(temps):.0f} K")
        return snaps

    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "state.npz"
    snaps_path = checkpoint_dir / "snapshots.npz"
    rng = np.random.default_rng(seed + 1)
    equil_done = sample_done = stride_done = 0
    snaps = []

    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"checkpoint exists but resume=False: {state_path}")
        with np.load(state_path, allow_pickle=False) as state:
            expected = {
                "n_atoms": len(ideal), "temperature": float(T), "dt_fs": float(dt_fs),
                "n_equil": int(n_equil), "n_snap": int(n_snap),
                "stride": int(stride), "seed": int(seed),
            }
            observed = {key: state[key].item() for key in expected}
            completed_snapshots = int(state["sample_done"].item())
            for key, value in expected.items():
                if key == "n_snap":
                    stored_n_snap = int(observed[key])
                    if int(value) < stored_n_snap or int(value) < completed_snapshots:
                        raise ValueError(
                            "checkpoint n_snap can only be extended: "
                            f"stored={stored_n_snap}, requested={int(value)}, "
                            f"completed={completed_snapshots}"
                        )
                    if int(value) > stored_n_snap and log is not None:
                        log(
                            "    extending MD checkpoint target: "
                            f"n_snap={stored_n_snap}->{int(value)}"
                        )
                    continue
                if isinstance(value, float):
                    matches = np.isclose(float(observed[key]), value, rtol=0.0, atol=1e-12)
                else:
                    matches = int(observed[key]) == value
                if not matches:
                    raise ValueError(
                        f"checkpoint parameter mismatch for {key}: "
                        f"stored={observed[key]!r}, requested={value!r}"
                    )
            at.set_cell(np.asarray(state["cell"], dtype=float), scale_atoms=False)
            at.set_positions(np.asarray(state["positions"], dtype=float))
            at.set_momenta(np.asarray(state["momenta"], dtype=float))
            equil_done = int(state["equil_done"].item())
            sample_done = int(state["sample_done"].item())
            stride_done = int(state["stride_done"].item())
            rng.bit_generator.state = json.loads(str(state["rng_state"].item()))
        at.calc = calc
        snaps = _load_checkpoint_snapshots(snaps_path, ideal)
        if len(snaps) != sample_done:
            raise RuntimeError(
                f"snapshot checkpoint count mismatch: state={sample_done}, file={len(snaps)}"
            )
        if log is not None:
            log(f"    resumed MD checkpoint: equil={equil_done}/{n_equil}, "
                f"snap={sample_done}/{n_snap}, stride={stride_done}/{stride}")
        _validate_md_state(
            at, T, max_temperature_factor, min_pair_distance, max_force,
            check_geometry=True,
        )
        for snapshot in snaps:
            _validate_md_state(
                snapshot, T, max_temperature_factor, min_pair_distance, max_force,
                check_geometry=True,
            )
    else:
        MaxwellBoltzmannDistribution(at, temperature_K=T, rng=np.random.default_rng(seed))
        Stationary(at)
        ZeroRotation(at)

    dyn = Langevin(at, dt_fs * units.fs, temperature_K=T, friction=0.02, rng=rng)

    def save_state() -> None:
        _atomic_savez(
            state_path, version=1, n_atoms=len(ideal), temperature=float(T),
            dt_fs=float(dt_fs), n_equil=int(n_equil), n_snap=int(n_snap),
            stride=int(stride), seed=int(seed), positions=np.asarray(at.positions),
            cell=np.asarray(at.cell), momenta=np.asarray(at.get_momenta()),
            equil_done=equil_done, sample_done=sample_done,
            stride_done=stride_done, rng_state=np.array(json.dumps(rng.bit_generator.state)),
        )

    def save_snapshots() -> None:
        energies = []
        for snap in snaps:
            try:
                energies.append(float(snap.get_potential_energy()))
            except Exception:
                energies.append(float("nan"))
        _atomic_savez(
            snaps_path,
            positions=np.asarray([snap.positions for snap in snaps]),
            forces=np.asarray([snap.get_forces() for snap in snaps]),
            energies=np.asarray(energies),
            cells=np.asarray([snap.cell for snap in snaps]),
            momenta=np.asarray([snap.get_momenta() for snap in snaps]),
        )

    # Ensure there is a restart point even if the first force evaluation dies.
    if not state_path.exists():
        save_state()

    total_since_checkpoint = 0
    while equil_done < n_equil:
        dyn.run(1)
        _validate_md_state(
            at, T, max_temperature_factor, min_pair_distance, max_force,
        )
        equil_done += 1
        total_since_checkpoint += 1
        if total_since_checkpoint >= checkpoint_every or equil_done == n_equil:
            _validate_md_state(
                at, T, max_temperature_factor, min_pair_distance, max_force,
                check_geometry=True,
            )
            save_state()
            total_since_checkpoint = 0
            if log is not None:
                log(f"    MD checkpoint: equil={equil_done}/{n_equil}")

    temps = [snap.get_temperature() for snap in snaps]
    while sample_done < n_snap:
        while stride_done < stride:
            dyn.run(1)
            _validate_md_state(
                at, T, max_temperature_factor, min_pair_distance, max_force,
            )
            stride_done += 1
            total_since_checkpoint += 1
            if total_since_checkpoint >= checkpoint_every:
                _validate_md_state(
                    at, T, max_temperature_factor, min_pair_distance, max_force,
                    check_geometry=True,
                )
                save_state()
                total_since_checkpoint = 0
                if log is not None:
                    log(f"    MD checkpoint: snap={sample_done}/{n_snap}, "
                        f"stride={stride_done}/{stride}")
        temperature = _validate_md_state(
            at, T, max_temperature_factor, min_pair_distance, max_force,
            check_geometry=True,
        )
        force = np.asarray(at.get_forces()).copy()
        try:
            energy = float(at.get_potential_energy())
        except Exception:
            energy = float("nan")
        snap = at.copy()
        properties = {"forces": force}
        if np.isfinite(energy):
            properties["energy"] = energy
        snap.calc = SinglePointCalculator(snap, **properties)
        snaps.append(snap)
        temps.append(temperature)
        sample_done += 1
        stride_done = 0
        save_snapshots()
        save_state()
        if log is not None:
            log(f"    MD snapshot committed: {sample_done}/{n_snap}, "
                f"T_inst={temps[-1]:.0f} K")
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
    rmse = getattr(opt, "rmse_train", None)
    return fc2, float(rmse) if rmse is not None else float("nan"), len(opt.parameters)


def band_from_phonopy(ph, fc2, path="MGKM", npoints=201):
    """Run the dispersion on an existing phonopy object whose supercell ordering
    matches the hiphive ``ideal`` used to build ``fc2``."""
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
    ap.add_argument("--seed", type=int, default=0, help="independent MD random seed")
    ap.add_argument("--npoints", type=int, default=201)
    ap.add_argument("--outdir", default="results/td_phonon")
    ap.add_argument(
        "--checkpoint-root",
        default="",
        help="optional restart directory; one atomic MD checkpoint is kept per temperature",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="MD steps between restart checkpoints when --checkpoint-root is set",
    )
    ap.add_argument(
        "--max-temperature-factor",
        type=float,
        default=5.0,
        help="abort before fitting when instantaneous T exceeds this multiple of target T",
    )
    ap.add_argument(
        "--min-pair-distance",
        type=float,
        default=0.8,
        help="abort before fitting when any periodic pair is closer than this distance (A)",
    )
    ap.add_argument(
        "--max-force",
        type=float,
        default=100.0,
        help="abort before fitting when any force component exceeds this value (eV/A)",
    )
    ap.add_argument("--no-relax", action="store_true",
                    help="skip MLIP relax; build the clean primitive at --a/--thickness "
                         "(use to pin an unstable/soft-mode material at a fixed DFT geometry)")
    ap.add_argument("--a", type=float, default=0.0, help="fixed in-plane a (A) when --no-relax")
    ap.add_argument("--thickness", type=float, default=0.0, help="fixed thickness (A) when --no-relax")
    a = ap.parse_args()

    def log(m):
        print(m, flush=True)

    sc = tuple(int(x) for x in a.supercell.split(","))
    sc_matrix = np.diag(sc)
    temps = [float(x) for x in a.temperatures.split(",")]

    at0 = read(ROOT / a.structure)
    log(f"[{a.tag}] {a.structure} ({len(at0)} atoms); model={a.model} on {a.device}")
    calc = tdc.get_mace_calc(a.model, device=a.device)

    # relax to get the model's equilibrium (a, thickness), then rebuild a clean
    # symmetric primitive at those values -- the relaxer's in-plane shear leaves
    # the cell slightly off-hexagonal and trips hiphive's orbit enumeration.
    name = Path(a.structure).stem
    if a.no_relax:
        aa = a.a if a.a > 0 else float(np.linalg.norm(at0.cell[0]))
        tt = a.thickness if a.thickness > 0 else None
        prim = tdc.build_monolayer(name, vacuum=7.5, a=aa, thickness=tt)
        prim.wrap()
        log(f"[{a.tag}] NO-RELAX: clean {name} primitive at a={aa:.4f} A "
            f"thickness={a.thickness or 'lit'}")
    else:
        prim, info = tdc.relax_monolayer(at0, calc)
        prim = tdc.build_monolayer(name, vacuum=7.5, a=info["a"], thickness=info["thickness"])
        prim.wrap()
        log(f"[{a.tag}] relaxed a={info['a']:.4f} A thickness={info['thickness']:.3f} A "
            f"-> clean {name} primitive")
    from phonon_accel.phonons import ase_to_phonopy, phonopy_to_ase
    from phonopy import Phonopy
    ph0 = Phonopy(ase_to_phonopy(prim), supercell_matrix=sc_matrix,
                  primitive_matrix=np.eye(3))
    # Use phonopy's supercell (its atom ordering) as the MD reference so the
    # hiphive fc2 maps straight onto ph0; build the ClusterSpace from the clean
    # ASE primitive (phonopy's round-tripped primitive trips a hiphive orbit bug).
    ideal = phonopy_to_ase(ph0.supercell)
    ideal.wrap()
    log(f"[{a.tag}] MD supercell {sc} = {len(ideal)} atoms, cutoff2={a.cutoff2} A")

    out = {
        "temperatures": np.array(temps),
        "supercell": np.asarray(sc),
        "supercell_matrix": np.asarray(sc_matrix),
        "primitive_numbers": np.asarray(prim.numbers),
        "primitive_positions": np.asarray(prim.positions),
        "primitive_cell": np.asarray(prim.cell),
        "model": np.array(str(a.model)),
        "tag": np.array(str(a.tag)),
        "seed": np.array(a.seed),
        "dt_fs": np.array(a.dt),
        "n_equil": np.array(a.equil),
        "n_snap": np.array(a.nsnap),
        "stride": np.array(a.stride),
    }
    rows = [
        "T_K,w_gamma_cm,w_k_cm,kink_gamma,kink_k,minfreq_thz,"
        "fit_rmse_meVA,mean_T_K,max_T_K"
    ]

    for T in temps:
        t0 = time.perf_counter()
        log(f"[{a.tag}] T={T:.0f} K: MD sampling ...")
        checkpoint_dir = None
        if a.checkpoint_root:
            checkpoint_root = Path(a.checkpoint_root)
            if not checkpoint_root.is_absolute():
                checkpoint_root = ROOT / checkpoint_root
            checkpoint_dir = checkpoint_root / f"T{int(T)}"
        snaps = sample_md(
            ideal,
            calc,
            T,
            a.dt,
            a.equil,
            a.nsnap,
            a.stride,
            seed=a.seed,
            log=log,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=a.checkpoint_every,
            max_temperature_factor=a.max_temperature_factor,
            min_pair_distance=a.min_pair_distance,
            max_force=a.max_force,
        )
        instantaneous_temperatures = np.asarray(
            [snapshot.get_temperature() for snapshot in snaps], dtype=float
        )
        key = f"T{int(T)}"
        out[f"{key}_instantaneous_temperatures_K"] = instantaneous_temperatures
        out[f"{key}_mean_temperature_K"] = np.array(
            float(np.mean(instantaneous_temperatures))
        )
        out[f"{key}_max_temperature_K"] = np.array(
            float(np.max(instantaneous_temperatures))
        )
        fc2, rmse, ndof = effective_fc2(prim, ideal, sc_matrix, snaps, a.cutoff2)
        dist, freq, lp, labs = band_from_phonopy(ph0, fc2, npoints=a.npoints)
        wG = al.branch_freq_at_label(dist, freq, lp, labs, r"$\Gamma$") * CM
        wK = al.branch_freq_at_label(dist, freq, lp, labs, "K") * CM
        kinks = {k["label"]: k for k in al.high_sym_kinks(dist, freq, lp, labs)}
        kG, kK = kinks[r"$\Gamma$"]["kink_strength"], kinks["K"]["kink_strength"]
        out[f"{key}_dist"], out[f"{key}_freq"] = dist, freq
        out[f"{key}_fc2"] = fc2
        out["label_positions"], out["labels"] = lp, labs
        rows.append(f"{T:.0f},{wG:.2f},{wK:.2f},{kG:.3f},{kK:.3f},"
                    f"{freq.min():.3f},{rmse*1000:.2f},"
                    f"{np.mean(instantaneous_temperatures):.2f},"
                    f"{np.max(instantaneous_temperatures):.2f}")
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
