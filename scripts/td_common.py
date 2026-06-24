"""Shared helpers for the TD-phonon / Kohn-anomaly study (M0–M2).

2D-monolayer builders (graphene / MoS2 / NbSe2), an in-plane-only relaxer that
preserves the vacuum gap, and a phonopy finite-displacement dispersion driver
restricted to an explicit Gamma-M-K-Gamma path (the auto/seekpath path drags in
the meaningless k_z = 1/2 points for a slab). Built on the repo's existing
``phonon_accel`` pipeline so the bands share settings with the main paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------------- #
# Monolayer structures (literature geometry; relaxed per-model downstream)
# --------------------------------------------------------------------------- #
_LIT = {  # literature (a, thickness) per monolayer
    "graphene": (2.46, 0.0),
    "mos2": (3.18, 3.19),
    "nbse2": (3.44, 3.34),
}


def build_monolayer(name: str, vacuum: float = 7.5, a: float | None = None,
                    thickness: float | None = None) -> Atoms:
    """Return a *perfectly symmetric* monolayer (ASE builder).

    With ``a``/``thickness`` None, uses the literature values; pass the relaxed
    values to rebuild a clean primitive at a model's equilibrium (the relaxer's
    in-plane shear can leave the cell slightly off-hexagonal, which trips
    hiphive's orbit enumeration -- always rebuild before TDEP). ``vacuum`` is the
    half-gap along c; pbc in all three directions.
    """
    from ase.build import graphene, mx2

    name = name.lower()
    la, lt = _LIT.get(name, (None, None))
    a = la if a is None else a
    thickness = lt if thickness is None else thickness
    if name == "graphene":
        at = graphene(formula="C2", a=a, vacuum=vacuum)
    elif name == "mos2":
        at = mx2(formula="MoS2", kind="2H", a=a, thickness=thickness, vacuum=vacuum)
    elif name == "nbse2":
        at = mx2(formula="NbSe2", kind="2H", a=a, thickness=thickness, vacuum=vacuum)
    else:
        raise ValueError(f"unknown monolayer {name!r} (graphene|mos2|nbse2)")
    at.pbc = True
    return at


def relax_monolayer(atoms: Atoms, calc, fmax: float = 1e-3, steps: int = 200):
    """Relax in-plane lattice (a, b, in-plane shear) + all atomic positions,
    holding the c-axis vacuum fixed. Returns (relaxed_atoms, info)."""
    from ase.optimize import BFGS

    try:
        from ase.filters import FrechetCellFilter as CellFilter
    except Exception:  # older ASE
        from ase.constraints import ExpCellFilter as CellFilter

    at = atoms.copy()
    at.calc = calc
    # Voigt mask [xx, yy, zz, yz, xz, xy]: relax in-plane only, freeze vacuum.
    flt = CellFilter(at, mask=[True, True, False, False, False, True])
    opt = BFGS(flt, logfile=None)
    opt.run(fmax=fmax, steps=steps)

    a = float(np.linalg.norm(at.cell[0]))
    z = at.get_positions()[:, 2]
    info = {
        "a": a,
        "thickness": float(z.max() - z.min()),
        "fmax": float(np.abs(at.get_forces()).max()),
        "converged": bool(opt.converged()),
        "nsteps": opt.get_number_of_steps(),
    }
    at.calc = None
    return at, info


# --------------------------------------------------------------------------- #
# Band path  (hexagonal primitive: Setyawan-Curtarolo HEX reduced coords)
# --------------------------------------------------------------------------- #
HIGH_SYM = {
    "G": [0.0, 0.0, 0.0],
    "M": [0.5, 0.0, 0.0],
    "K": [1.0 / 3.0, 1.0 / 3.0, 0.0],
}
_LABEL = {"G": r"$\Gamma$", "M": "M", "K": "K"}


def make_band_path(names: str = "GMKG", npoints: int = 201):
    """One connected path through the named high-symmetry points.

    Returns (qpoints, connections, labels) for ``phonon.run_band_structure``.
    """
    from phonopy.phonon.band_structure import (
        get_band_qpoints_and_path_connections,
    )

    pts = [HIGH_SYM[c] for c in names]
    qpoints, connections = get_band_qpoints_and_path_connections(
        [pts], npoints=npoints
    )
    labels = [_LABEL[c] for c in names]
    return qpoints, connections, labels


# --------------------------------------------------------------------------- #
# Dispersion driver
# --------------------------------------------------------------------------- #
def dispersion(
    atoms: Atoms,
    calc,
    supercell=(5, 5, 1),
    displacement: float = 0.03,
    path: str = "GMKG",
    npoints: int = 201,
):
    """Finite-displacement phonon dispersion on an explicit path.

    Uses the input cell as the primitive (identity primitive_matrix) so the
    band reduced coordinates map directly onto ``HIGH_SYM``. Returns a dict of
    flat numpy arrays ready to cache to npz.
    """
    from phonon_accel.phonons import PhononCalculation

    sc = np.diag(supercell) if np.asarray(supercell).ndim == 1 else np.asarray(supercell)
    phon = PhononCalculation(
        atoms,
        supercell_matrix=sc,
        primitive_matrix=np.eye(3),
        displacement=displacement,
    )
    phon.compute_forces(calc)
    phon.produce_force_constants(symmetrize=True)
    ph = phon.phonon

    qpoints, connections, labels = make_band_path(path, npoints=npoints)
    ph.run_band_structure(
        qpoints, path_connections=connections, labels=labels,
        with_eigenvectors=False,
    )
    bsd = ph.get_band_structure_dict()

    dist = np.concatenate([np.asarray(d) for d in bsd["distances"]])
    freq = np.concatenate([np.asarray(f) for f in bsd["frequencies"]], axis=0)
    qfrac = np.concatenate([np.asarray(q) for q in bsd["qpoints"]], axis=0)

    # distance of each labelled high-symmetry point (segment endpoints)
    seg_d = [np.asarray(d) for d in bsd["distances"]]
    label_pos = [seg_d[0][0]] + [s[-1] for s in seg_d]

    # Drop duplicate join points: phonopy repeats the q-distance of each
    # high-symmetry point between adjacent segments, which makes np.gradient
    # divide by ~0 and manufacture an infinite spurious cusp there.
    keep = np.concatenate([[True], np.diff(dist) > 1e-9])
    dist, freq, qfrac = dist[keep], freq[keep], qfrac[keep]

    return {
        "distances": dist,
        "frequencies": freq,            # (n_q, n_band), THz
        "qpoints_frac": qfrac,
        "labels": np.array(labels),
        "label_positions": np.array(label_pos),
        "n_displacements": phon.n_displacements,
        "supercell": np.asarray(sc),
    }


def get_mace_calc(model: str, device: str = "cuda"):
    """Foundation (``medium``/``small``/...) or a path to an FC-distilled model."""
    from phonon_accel.mlip_calc import get_calculator

    return get_calculator("mace", device=device, model=model)
