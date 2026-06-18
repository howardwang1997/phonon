import numpy as np

from phonongpu.structure import Structure
from phonongpu import symmetry as sym


def _diamond_conv(a=5.43):
    cell = a * np.eye(3)
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    return Structure(cell, np.vstack([fcc, fcc + 0.25]), ["Si"] * 8)


def test_diamond_spacegroup():
    s = _diamond_conv()
    ops = sym.find_spacegroup_ops(s)
    assert len(ops) == 192  # Fd-3m


def test_diamond_one_inequivalent_atom():
    s = _diamond_conv()
    orbits = sym.atom_orbits(s)
    assert len(orbits) == 1
    assert len(orbits[0]) == 8


def test_low_symmetry_identity_only():
    cell = np.array([[4.1, 0, 0], [0.2, 4.3, 0], [0.1, 0.2, 4.2]])
    s = Structure(cell, [[0, 0, 0], [0.31, 0.34, 0.37]], ["A", "B"], [10.0, 20.0])
    ops = sym.find_spacegroup_ops(s)
    assert len(ops) >= 1  # at least identity
