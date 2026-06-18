import numpy as np
import pytest

from phonongpu.structure import Structure
from phonongpu.io.poscar import write_poscar, read_poscar


def _diamond_conv(a=5.43):
    cell = a * np.eye(3)
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    return Structure(cell, np.vstack([fcc, fcc + 0.25]), ["Si"] * 8)


def test_supercell_diagonal():
    s = _diamond_conv()
    assert s.make_supercell(2).natoms == 64
    assert s.make_supercell([2, 2, 2]).natoms == 64


def test_supercell_rejects_matrix():
    s = _diamond_conv()
    with pytest.raises(ValueError):
        s.make_supercell(np.diag([2, 2, 2]))


def test_reciprocal():
    s = Structure(5.0 * np.eye(3), [[0, 0, 0]], ["Si"])
    assert np.allclose(np.diag(s.reciprocal()), 2 * np.pi / 5.0)


def test_poscar_roundtrip(tmp_path):
    s = _diamond_conv()
    p = tmp_path / "POSCAR"
    write_poscar(s, str(p))
    s2 = read_poscar(str(p))
    assert s2.natoms == s.natoms
    assert np.allclose(s2.cell, s.cell)
    assert np.allclose(s2.positions, s.positions, atol=1e-8)
