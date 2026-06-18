from __future__ import annotations

import numpy as np

from ..structure import Structure


def write_poscar(structure, path, title="phonongpu", scale=1.0):
    species = structure.species
    unique = []
    counts = []
    for s in species:
        if s in unique:
            counts[unique.index(s)] += 1
        else:
            unique.append(s)
            counts.append(1)
    lines = [title, f"{scale:.10g}"]
    for row in structure.cell:
        lines.append(f"{row[0]: .16f} {row[1]: .16f} {row[2]: .16f}")
    lines.append("  " + "  ".join(unique))
    lines.append("  " + "  ".join(str(c) for c in counts))
    lines.append("Direct")
    for p in structure.positions:
        lines.append(f"{p[0]: .16f} {p[1]: .16f} {p[2]: .16f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_poscar(path):
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    title = lines[0]
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[2 + i].split()] for i in range(3)])
    cell = cell * scale
    sp_block = lines[5].split()
    try:
        counts = [int(float(x)) for x in sp_block]
        species_line = None
    except ValueError:
        species_line = sp_block
        counts = [int(float(x)) for x in lines[6].split()]
        offset = 7
    else:
        offset = 6
    coord_type = lines[offset].strip()[:1].lower() or "d"
    offset += 1
    positions = []
    species = []
    for c, cnt in enumerate(counts):
        for _ in range(cnt):
            parts = lines[offset].split()
            positions.append([float(x) for x in parts[:3]])
            sp = species_line[c] if species_line else str(c)
            species.append(sp)
            offset += 1
    positions = np.array(positions)
    if coord_type == "c":
        positions = (np.linalg.inv(cell) @ positions.T).T
    return Structure(cell, positions, species)
