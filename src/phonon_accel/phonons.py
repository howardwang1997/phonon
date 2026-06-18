"""Unified phonon pipeline built on phonopy + ASE.

Backend-agnostic: any ASE calculator (MLIP or DFT) plugs into the same
flow. Given a primitive cell and a supercell matrix, the pipeline

1. generates symmetry-reduced displacements,
2. computes forces on each displaced supercell with the supplied calculator,
3. builds force constants (optionally ASR-symmetrized),
4. exposes band structure, DOS, and thermal properties.

The ``compute_forces`` step optionally supports a "density/state reuse"
ordering hook used by Line B (DFT) — see ``forces_from_calculator``'s
``order`` argument — but for MLIPs the per-supercell evaluations are
independent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


# --------------------------------------------------------------------------- #
# ASE <-> phonopy conversion
# --------------------------------------------------------------------------- #
def ase_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    """Convert an ASE ``Atoms`` to a ``PhonopyAtoms``."""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell().array,
    )


def phonopy_to_ase(ph_atoms: PhonopyAtoms) -> Atoms:
    """Convert a ``PhonopyAtoms`` to an ASE ``Atoms`` (periodic)."""
    return Atoms(
        symbols=ph_atoms.symbols,
        scaled_positions=ph_atoms.scaled_positions,
        cell=ph_atoms.cell,
        pbc=True,
    )


# --------------------------------------------------------------------------- #
# Results container
# --------------------------------------------------------------------------- #
@dataclass
class PhononResult:
    """Lightweight, picklable summary of a phonon calculation."""

    formula: str
    supercell_matrix: list
    n_displacements: int
    # band structure on the auto (seekpath) path
    band_qpoints: Optional[list] = None
    band_frequencies: Optional[list] = None  # THz, list of (n_q, n_band) arrays
    band_distances: Optional[list] = None
    band_labels: Optional[list] = None
    # mesh-sampled frequencies (THz) for DOS / global metrics
    mesh_frequencies: Optional[np.ndarray] = None
    dos_frequencies: Optional[np.ndarray] = None
    dos: Optional[np.ndarray] = None
    # thermal properties
    temperatures: Optional[np.ndarray] = None
    free_energy: Optional[np.ndarray] = None  # kJ/mol
    entropy: Optional[np.ndarray] = None  # J/K/mol
    heat_capacity: Optional[np.ndarray] = None  # J/K/mol
    # diagnostics
    min_frequency: float = np.nan  # THz (most negative = imaginary)
    n_imaginary_mesh: int = 0
    asr_residual: float = np.nan  # THz, acoustic freq at Gamma
    timing: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
class PhononCalculation:
    """Phonopy-backed finite-displacement phonon calculation.

    Parameters
    ----------
    atoms : ase.Atoms
        The (already relaxed) primitive/unit cell.
    supercell_matrix : array-like, shape (3,) or (3, 3)
        Supercell expansion for force constants.
    primitive_matrix : "auto" | array-like
        Passed to phonopy. "auto" lets phonopy/spglib find the primitive.
    displacement : float
        Finite-displacement distance in Angstrom.
    """

    def __init__(
        self,
        atoms: Atoms,
        supercell_matrix=(2, 2, 2),
        primitive_matrix="auto",
        displacement: float = 0.01,
        symprec: float = 1e-5,
    ):
        sc = np.asarray(supercell_matrix)
        if sc.shape == (3,):
            sc = np.diag(sc)
        self.displacement = displacement
        self.formula = atoms.get_chemical_formula()
        self.supercell_matrix = sc.tolist()
        self.phonon = Phonopy(
            ase_to_phonopy(atoms),
            supercell_matrix=sc,
            primitive_matrix=primitive_matrix,
            symprec=symprec,
        )
        self.phonon.generate_displacements(distance=displacement)
        self._forces_set = False

    # -- step 2: forces ---------------------------------------------------- #
    @property
    def displaced_supercells(self) -> list[Atoms]:
        return [phonopy_to_ase(c) for c in self.phonon.supercells_with_displacements]

    @property
    def n_displacements(self) -> int:
        return len(self.phonon.supercells_with_displacements)

    def compute_forces(self, calculator, batch: bool = False) -> float:
        """Attach ``calculator`` to each displaced supercell and store forces.

        Returns wall-clock seconds spent in force evaluation. ``calculator``
        is an ASE calculator instance (shared across supercells).
        """
        t0 = time.perf_counter()
        supercells = self.displaced_supercells
        forces = []
        for sc in supercells:
            sc.calc = calculator
            forces.append(sc.get_forces())
        self.phonon.forces = np.array(forces)
        self._forces_set = True
        return time.perf_counter() - t0

    def set_forces(self, forces: np.ndarray) -> None:
        """Set forces directly (e.g. parsed from an external DFT run)."""
        self.phonon.forces = np.asarray(forces)
        self._forces_set = True

    # -- step 3: force constants ------------------------------------------ #
    def produce_force_constants(self, symmetrize: bool = True) -> None:
        if not self._forces_set:
            raise RuntimeError("Call compute_forces/set_forces first.")
        self.phonon.produce_force_constants()
        if symmetrize:
            # enforce acoustic sum rule / symmetry on force constants
            self.phonon.symmetrize_force_constants()

    # -- step 4: observables ---------------------------------------------- #
    def run_band_structure(self, npoints: int = 101):
        self.phonon.auto_band_structure(
            npoints=npoints, with_eigenvectors=False, plot=False, write_yaml=False
        )
        return self.phonon.get_band_structure_dict()

    def run_mesh_and_dos(self, mesh=(20, 20, 20)):
        # Gamma-centered mesh keeps full point-group symmetry (no half-shift).
        self.phonon.run_mesh(mesh, is_gamma_center=True)
        self.phonon.run_total_dos()
        return self.phonon.get_mesh_dict(), self.phonon.get_total_dos_dict()

    def run_thermal_properties(self, t_min=0, t_max=1000, t_step=10):
        self.phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
        return self.phonon.get_thermal_properties_dict()

    # -- one-shot driver -------------------------------------------------- #
    def run_all(
        self,
        calculator=None,
        mesh=(20, 20, 20),
        band_npoints: int = 101,
        symmetrize: bool = True,
    ) -> PhononResult:
        """Run the full pipeline and return a ``PhononResult``.

        If ``calculator`` is None, forces must already have been set via
        ``set_forces`` (DFT path).
        """
        timing = {}
        if calculator is not None:
            timing["forces_s"] = self.compute_forces(calculator)
        t0 = time.perf_counter()
        self.produce_force_constants(symmetrize=symmetrize)
        timing["fc_s"] = time.perf_counter() - t0

        return build_result_from_phonon(
            self.phonon,
            formula=self.formula,
            supercell_matrix=self.supercell_matrix,
            n_displacements=self.n_displacements,
            mesh=mesh,
            band_npoints=band_npoints,
            timing=timing,
        )


def gamma_acoustic_residual(phonon: Phonopy) -> float:
    """|max acoustic frequency at Gamma| in THz (should be ~0 by ASR)."""
    try:
        phonon.run_qpoints([[0.0, 0.0, 0.0]])
        freqs = phonon.get_qpoints_dict()["frequencies"][0]
        acoustic = np.sort(np.abs(freqs))[:3]  # three acoustic branches
        return float(np.max(acoustic))
    except Exception:
        return float("nan")


def build_result_from_phonon(
    phonon: Phonopy,
    formula: str,
    supercell_matrix,
    n_displacements: int = -1,
    mesh=(20, 20, 20),
    band_npoints: int = 101,
    timing: Optional[dict] = None,
) -> PhononResult:
    """Build a ``PhononResult`` from a phonopy object that already has
    force constants. Shared by the MLIP path and the DFT-reference path so
    both are computed on identical band/mesh settings.
    """
    phonon.auto_band_structure(
        npoints=band_npoints, with_eigenvectors=False, plot=False, write_yaml=False
    )
    band = phonon.get_band_structure_dict()
    phonon.run_mesh(mesh, is_gamma_center=True)
    phonon.run_total_dos()
    mesh_dict = phonon.get_mesh_dict()
    dos_dict = phonon.get_total_dos_dict()
    phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)
    thermal = phonon.get_thermal_properties_dict()

    mesh_freqs = np.asarray(mesh_dict["frequencies"])  # (n_q, n_band) THz
    return PhononResult(
        formula=formula,
        supercell_matrix=list(supercell_matrix),
        n_displacements=n_displacements,
        band_qpoints=[q.tolist() for q in band["qpoints"]],
        band_frequencies=[f.tolist() for f in band["frequencies"]],
        band_distances=[d.tolist() for d in band["distances"]],
        band_labels=band.get("labels"),
        mesh_frequencies=mesh_freqs,
        dos_frequencies=np.asarray(dos_dict["frequency_points"]),
        dos=np.asarray(dos_dict["total_dos"]),
        temperatures=np.asarray(thermal["temperatures"]),
        free_energy=np.asarray(thermal["free_energy"]),
        entropy=np.asarray(thermal["entropy"]),
        heat_capacity=np.asarray(thermal["heat_capacity"]),
        min_frequency=float(np.min(mesh_freqs)),
        n_imaginary_mesh=int(np.sum(mesh_freqs < -1e-3)),
        asr_residual=gamma_acoustic_residual(phonon),
        timing=timing or {},
    )
