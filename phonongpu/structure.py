from __future__ import annotations

import numpy as np

_KEPT_TYPES = ("all",)


class Structure:
    def __init__(self, cell, positions, species, masses=None):
        self.cell = np.asarray(cell, dtype=float).reshape(3, 3)
        self.positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        n = self.positions.shape[0]
        if isinstance(species, str):
            species = [species] * n
        self.species = list(species)
        if len(self.species) != n:
            raise ValueError("species length must match number of positions")
        if masses is None:
            self.masses = np.array(
                [_DEFAULT_MASS.get(s, 1.0) for s in self.species], dtype=float
            )
        else:
            self.masses = np.asarray(masses, dtype=float).reshape(-1)
        if self.masses.shape[0] != n:
            raise ValueError("masses length must match number of positions")

    @property
    def natoms(self):
        return self.positions.shape[0]

    @property
    def volume(self):
        return float(abs(np.linalg.det(self.cell)))

    def reciprocal(self):
        return 2.0 * np.pi * np.linalg.inv(self.cell).T

    def cartesian(self):
        return self.positions @ self.cell

    def make_supercell(self, scale):
        scale = np.asarray(scale)
        if scale.ndim == 0:
            s = np.array([int(scale)] * 3)
        elif scale.size == 3:
            s = scale.astype(int).reshape(3)
        else:
            raise ValueError("scale must be scalar or length-3 (diagonal) for phonon supercells")
        if (s <= 0).any():
            raise ValueError("diagonal supercell factors must be positive")
        nx, ny, nz = s
        new_cell = self.cell * np.array([nx, ny, nz])[None, :]
        denom = np.array([nx, ny, nz], dtype=float)
        pos, spec, mass = [], [], []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shift = np.array([ix, iy, iz], dtype=float)
                    for i in range(self.natoms):
                        frac = (self.positions[i] + shift) / denom
                        frac = frac - np.floor(frac)
                        pos.append(frac)
                        spec.append(self.species[i])
                        mass.append(self.masses[i])
        return Structure(new_cell, np.array(pos), spec, mass)

    def neighbors(self, cutoff, self_image=True):
        cart = self.cartesian()
        n = self.natoms
        a = self.cell
        lens = np.linalg.norm(a, axis=1)
        r_ext = [int(np.ceil(cutoff / lens[i])) + 1 for i in range(3)]
        shifts = []
        for i in range(-r_ext[0], r_ext[0] + 1):
            for j in range(-r_ext[1], r_ext[1] + 1):
                for k in range(-r_ext[2], r_ext[2] + 1):
                    shifts.append((i, j, k))
        shifts = np.array(shifts, dtype=float)
        shift_cart = shifts @ a
        pairs = []
        for i in range(n):
            for j in range(n):
                for sidx, sc in enumerate(shift_cart):
                    d = cart[j] + sc - cart[i]
                    dist = np.linalg.norm(d)
                    if dist > cutoff + 1e-8:
                        continue
                    if dist < 1e-8 and not self_image:
                        continue
                    frac_shift = shifts[sidx]
                    pairs.append((i, j, frac_shift, dist, d))
        return pairs

    def make_supercell_with_map(self, scale):
        scale = np.asarray(scale)
        if scale.ndim == 0:
            s = np.array([int(scale)] * 3)
        elif scale.size == 3:
            s = scale.astype(int).reshape(3)
        elif scale.size == 9:
            raise NotImplementedError("3x3 supercell mapping not supported; use diagonal scale")
        else:
            raise ValueError("scale must be scalar or length-3 for mapped supercell")
        if (s <= 0).any():
            raise ValueError("supercell factors must be positive")
        nx, ny, nz = s
        denom = np.array([nx, ny, nz], dtype=float)
        pos, spec, mass = [], [], []
        cell_map, prim_map = [], []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shift = np.array([ix, iy, iz], dtype=float)
                    for i in range(self.natoms):
                        frac = (self.positions[i] + shift) / denom
                        frac = frac - np.floor(frac)
                        pos.append(frac)
                        spec.append(self.species[i])
                        mass.append(self.masses[i])
                        cell_map.append((ix, iy, iz))
                        prim_map.append(i)
        new_cell = self.cell * np.array([nx, ny, nz])[None, :]
        sc = Structure(new_cell, np.array(pos), spec, mass)
        return sc, cell_map, prim_map

    def copy(self):
        return Structure(self.cell.copy(), self.positions.copy(), list(self.species), self.masses.copy())

    @classmethod
    def from_poscar(cls, path):
        from .io.poscar import read_poscar
        return read_poscar(path)

    def __repr__(self):
        return f"Structure(natoms={self.natoms}, species={sorted(set(self.species))}, vol={self.volume:.3f})"


_DEFAULT_MASS = {
    "H": 1.00794, "He": 4.0026, "Li": 6.941, "Be": 9.0122, "B": 10.811, "C": 12.0107,
    "N": 14.0067, "O": 15.9994, "F": 18.9984, "Ne": 20.1797, "Na": 22.9898, "Mg": 24.305,
    "Al": 26.9815, "Si": 28.0855, "P": 30.9738, "S": 32.065, "Cl": 35.453, "Ar": 39.948,
    "K": 39.0983, "Ca": 40.078, "Sc": 44.9559, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.9332, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.39,
    "Ga": 69.723, "Ge": 72.64, "As": 74.9216, "Se": 78.96, "Br": 79.904, "Kr": 83.8,
    "Rb": 85.4678, "Sr": 87.62, "Y": 88.9059, "Zr": 91.224, "Nb": 92.9064, "Mo": 95.94,
    "Tc": 98.0, "Ru": 101.07, "Rh": 102.9055, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411,
    "In": 114.818, "Sn": 118.71, "Sb": 121.76, "Te": 127.6, "I": 126.9045, "Xe": 131.293,
    "Cs": 132.9055, "Ba": 137.327, "La": 138.9055, "W": 183.84, "Pt": 195.084, "Au": 196.9665,
    "Pb": 207.2, "Bi": 208.9804,
}
