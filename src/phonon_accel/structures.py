"""Structure providers for the benchmark.

Sources:
* built-in reference cells (Si and a few standard crystals) — no network,
  used for the worked example and smoke tests;
* Materials Project via ``mp-api`` (optional, needs MP_API_KEY);
* any file ASE can read (CIF/POSCAR/...).

For the worked example we use Si in the diamond structure with the
experimental lattice constant a = 5.431 Angstrom.
"""
from __future__ import annotations

from typing import Optional

from ase import Atoms
from ase.build import bulk


# Experimental / well-converged PBE lattice constants (Angstrom) for quick cells.
_BUILTIN = {
    "Si": dict(name="Si", crystalstructure="diamond", a=5.431),
    "Ge": dict(name="Ge", crystalstructure="diamond", a=5.658),
    "C": dict(name="C", crystalstructure="diamond", a=3.567),
    "GaAs": dict(name="GaAs", crystalstructure="zincblende", a=5.653),
    "NaCl": dict(name="NaCl", crystalstructure="rocksalt", a=5.640),
    "MgO": dict(name="MgO", crystalstructure="rocksalt", a=4.212),
    "Al": dict(name="Al", crystalstructure="fcc", a=4.050),
}


def builtin(name: str) -> Atoms:
    """Return a built-in reference crystal cell."""
    if name not in _BUILTIN:
        raise KeyError(f"No built-in cell for {name!r}; have {list(_BUILTIN)}")
    return bulk(**_BUILTIN[name])


def from_file(path: str) -> Atoms:
    from ase.io import read

    return read(path)


def from_materials_project(mp_id: str, api_key: Optional[str] = None) -> Atoms:
    """Fetch a relaxed structure from Materials Project as ASE ``Atoms``."""
    import os

    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor

    key = api_key or os.environ.get("MP_API_KEY")
    with MPRester(key) as mpr:
        struct = mpr.get_structure_by_material_id(mp_id)
    return AseAtomsAdaptor.get_atoms(struct)


def relax(atoms: Atoms, calculator, fmax: float = 1e-4, steps: int = 300) -> Atoms:
    """Relax cell+positions with an ASE calculator before phonons.

    Phonons require a well-relaxed cell (residual forces bias the Hessian).
    """
    from ase.optimize import BFGS

    try:  # ASE >= 3.23
        from ase.filters import FrechetCellFilter as CellFilter
    except ImportError:  # older ASE
        from ase.constraints import ExpCellFilter as CellFilter

    atoms = atoms.copy()
    atoms.calc = calculator
    BFGS(CellFilter(atoms), logfile=None).run(fmax=fmax, steps=steps)
    return atoms
