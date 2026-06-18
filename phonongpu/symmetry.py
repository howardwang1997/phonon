from __future__ import annotations

import itertools
import numpy as np

try:
    import spglib as _spg
    _HAS_SPGLIB = True
except Exception:
    _HAS_SPGLIB = False


def _integer_point_group_candidates():
    mats = []
    for vals in itertools.product((-1, 0, 1), repeat=9):
        R = np.array(vals, dtype=int).reshape(3, 3)
        det = int(round(np.linalg.det(R)))
        if det in (1, -1):
            mats.append(R)
    return mats


def find_spacegroup_ops(structure, symprec=1e-5):
    numbers = _species_to_numbers(structure.species)
    cell = (structure.cell, structure.positions, numbers)
    if _HAS_SPGLIB:
        ops = _spg_get_symmetry(cell, symprec)
        if ops:
            return ops
    return _numpy_spacegroup_ops(
        (structure.cell, structure.cartesian(), list(structure.species)), symprec)


def _species_to_numbers(species):
    unique = {}
    out = []
    n = 1
    for s in species:
        if s not in unique:
            unique[s] = n
            n += 1
        out.append(unique[s])
    return out


def _spg_get_symmetry(cell, symprec):
    try:
        d = _spg.get_symmetry(cell, symprec=symprec)
        out = []
        for R, t in zip(d["rotations"], d["translations"]):
            out.append((np.array(R, dtype=int), np.array(t, dtype=float)))
        return out
    except Exception:
        return []


def _numpy_spacegroup_ops(cell, symprec):
    lattice, cart, species = cell
    G = lattice.T @ lattice
    frac = (np.linalg.inv(lattice) @ cart.T).T
    species = list(species)
    n = len(species)

    def reduced(x):
        x = np.atleast_2d(x)
        return x - np.floor(x)

    rots = [R for R in _integer_point_group_candidates()
            if np.allclose(R.T @ G @ R, G, atol=max(symprec, 1e-7) * 10)]

    frac_r = reduced(frac)
    ops = []
    for R in rots:
        rotated = (R @ frac.T).T
        rot_red = reduced(rotated)
        i0 = 0
        candidates = [frac_r[j] - rot_red[i0] for j in range(n)
                      if species[j] == species[i0]]
        for tau in candidates:
            tau = reduced(tau)[0]
            full = reduced(rot_red + tau)
            ok = True
            for k in range(n):
                found = False
                for j in range(n):
                    if species[j] != species[k]:
                        continue
                    d = full[k] - frac_r[j]
                    d = d - np.round(d)
                    if np.linalg.norm(d) < symprec:
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok:
                ops.append((R.copy(), tau.copy()))
    return ops


def site_symmetry(structure, atom_index, symprec=1e-5):
    ops = find_spacegroup_ops(structure, symprec)
    out = []
    for R, tau in ops:
        f = structure.positions[atom_index]
        img = R @ f + tau
        diff = img - f
        diff = diff - np.round(diff)
        if np.linalg.norm(diff) < symprec:
            out.append((R, tau))
    return out


def map_atom(structure, R, tau, atom_index, symprec=1e-5):
    frac = structure.positions
    f = R @ frac[atom_index] + tau
    for j in range(structure.natoms):
        if structure.species[j] != structure.species[atom_index]:
            continue
        diff = f - frac[j]
        diff = diff - np.round(diff)
        if np.linalg.norm(diff) < symprec:
            return j
    return None


def atom_orbits(structure, symprec=1e-5):
    ops = find_spacegroup_ops(structure, symprec)
    n = structure.natoms
    I3 = np.eye(3, dtype=int)
    z = np.zeros(3)
    assigned = [False] * n
    orbits = []
    for i in range(n):
        if assigned[i]:
            continue
        members = [(i, I3.copy(), z.copy())]
        assigned[i] = True
        for R, tau in ops:
            j = map_atom(structure, R, tau, i, symprec)
            if j is not None and not assigned[j]:
                members.append((j, R.copy(), tau.copy()))
                assigned[j] = True
        orbits.append(members)
    return orbits


def num_ops(structure, symprec=1e-5):
    return len(find_spacegroup_ops(structure, symprec))
