"""Generate a fine-tuning dataset by distilling DFPT force constants.

For a training material we load the MDR DFPT *full* force constants Φ (shape
(N, N, 3, 3) over the supercell), then sample rattled supercell configurations
and label them with the harmonic response:

    F[i,a] = -Σ_{j,b} Φ[i,j,a,b] u[j,b]            (forces, eV/Å)
    E      =  ½ Σ u[i,a] Φ[i,j,a,b] u[j,b]         (energy, eV; 0 at equilibrium)

Within the small-displacement regime that sets the phonons, these harmonic
labels ARE the DFPT forces — so force-matching on them injects the exact DFPT
curvature into the MLIP. No new DFT is required.

Configs are emitted as ASE ``Atoms`` with ``info['energy']`` and
``arrays['forces']`` (+ ``info['REF_energy']`` / ``arrays['REF_forces']`` aliases
for MACE), and can be written to extxyz consumable by MACE / MatterSim / NequIP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import phonopy
from ase import Atoms

from .. import reference
from ..phonons import phonopy_to_ase


def harmonic_forces(fc_full: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Harmonic forces (eV/Å) for displacement field ``u`` (N,3)."""
    return -np.einsum("ijab,jb->ia", fc_full, u)


def harmonic_energy(fc_full: np.ndarray, u: np.ndarray) -> float:
    """Harmonic energy (eV) relative to equilibrium."""
    return 0.5 * float(np.einsum("ia,ijab,jb->", u, fc_full, u))


def _supercell_atoms(ph: phonopy.Phonopy) -> Atoms:
    return phonopy_to_ase(ph.supercell)


def make_training_configs(
    mp_id: str,
    n_configs: int = 40,
    rattle_std: float = 0.02,
    include_single_atom: bool = True,
    n_single_sites: int = 6,
    single_deltas=(0.01, -0.01, 0.03),
    seed: int = 0,
) -> list[Atoms]:
    """Return a list of labeled rattled supercells for one MDR material.

    Parameters
    ----------
    n_configs : number of random-rattle configs (Gaussian, std ``rattle_std`` Å).
    include_single_atom : also add phonopy-style single-atom ±displacement
        configs (0.01/0.03 Å) which most directly probe individual FC columns.
    """
    ph = reference.load_reference_phonon(mp_id)
    # reload with FULL force constants for the harmonic response
    path = reference.fetch(mp_id)
    ph = phonopy.load(str(path), is_compact_fc=False)
    fc = ph.force_constants  # (N, N, 3, 3)
    base = _supercell_atoms(ph)
    N = len(base)
    pos0 = base.get_positions()
    rng = np.random.default_rng(seed)

    configs: list[Atoms] = []

    def _add(u: np.ndarray):
        atoms = base.copy()
        atoms.set_positions(pos0 + u)
        F = harmonic_forces(fc, u)
        E = harmonic_energy(fc, u)
        atoms.info["energy"] = E
        atoms.info["REF_energy"] = E
        atoms.arrays["forces"] = F
        atoms.arrays["REF_forces"] = F
        atoms.info["mp_id"] = mp_id
        configs.append(atoms)

    # equilibrium anchor (E=0, F=0)
    _add(np.zeros((N, 3)))

    # random rattles
    for _ in range(n_configs):
        _add(rng.normal(0.0, rattle_std, (N, 3)))

    # single-atom probes along +/- x,y,z for a few atoms (most directly
    # probe individual force-constant columns)
    if include_single_atom:
        for i in rng.choice(N, size=min(N, n_single_sites), replace=False):
            for a in range(3):
                for delta in single_deltas:
                    u = np.zeros((N, 3))
                    u[i, a] = delta
                    _add(u)

    return configs


def build_dataset(
    mp_ids: Iterable[str],
    out_dir: str = "data/finetune",
    val_fraction: float = 0.15,
    n_configs: int = 40,
    rattle_std: float = 0.02,
    n_single_sites: int = 6,
    seed: int = 0,
) -> dict:
    """Build train/val extxyz from a set of training materials."""
    from ase.io import write

    rng = np.random.default_rng(seed)
    all_configs: list[Atoms] = []
    per_material = {}
    for k, mp_id in enumerate(mp_ids):
        cfgs = make_training_configs(
            mp_id, n_configs=n_configs, rattle_std=rattle_std,
            n_single_sites=n_single_sites, seed=seed + k,
        )
        per_material[mp_id] = len(cfgs)
        all_configs.extend(cfgs)

    rng.shuffle(all_configs)
    n_val = int(len(all_configs) * val_fraction)
    val, train = all_configs[:n_val], all_configs[n_val:]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "train.xyz", train, format="extxyz")
    write(out / "val.xyz", val, format="extxyz")
    summary = dict(
        n_train=len(train), n_val=len(val), per_material=per_material,
        train_path=str(out / "train.xyz"), val_path=str(out / "val.xyz"),
    )
    return summary
