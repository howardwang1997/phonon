"""Deployment: the damped-Friedel long-range term as an ASE calculator that
wraps any base MLIP (e.g. MACE), so a finite-displacement phonon run *through*
it reproduces the smearing-dependent fc2(T_el).

    total_calc = FriedelMACECalculator(base_calc, ref_supercell, corr, T_el)

The base MLIP provides the smearing-blind backbone (its own accuracy); the
Friedel module adds the correct electronic-temperature dependence on top, as an
additive HARMONIC long-range correction

    E_Friedel(u) = 1/2  u^T . Delta_fc(T_el) . u ,      F = - Delta_fc(T_el) . u

where u = displacement from the reference supercell and Delta_fc(T_el) =
B(T_el) exp(-kappa(T_el) R) D0(R) is the validated 2-parameter Friedel fc
correction (friedel_module.add_template, expanded to full fc).  This is exact in
the harmonic (finite-displacement phonon) regime the term is meant for, and it
composes additively with the MLIP energy -- the BAMBOO short-range-NN +
analytic-long-range-module pattern, realised for the metallic 2k_F channel.

Run the self-contained validation:
    conda run --no-capture-output -n phonon python scripts/smearing_kink/friedel_calc.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import friedel_module as fm
from ase.calculators.calculator import Calculator, all_changes
from phonopy.harmonic.force_constants import compact_fc_to_full_fc

ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "results" / "vq_kink6"


# --------------------------------------------------------------- harmonic core
def full_fc(ph, compact):
    """Expand phonopy compact fc (nprim,nsuper,3,3) -> full (nsuper,nsuper,3,3)."""
    if compact.shape[0] == compact.shape[1]:
        return compact                      # already full
    return compact_fc_to_full_fc(ph.primitive, compact)


def _harm_EF(fc_full, ref_pos, cell, pos):
    """Energy/forces of a harmonic PES 1/2 u^T Phi u about ref_pos (min-image u)."""
    d = pos - ref_pos
    inv = np.linalg.inv(cell)
    frac = d @ inv
    frac -= np.round(frac)                # minimum-image displacement
    u = frac @ cell
    Fu = np.einsum("ijab,jb->ia", fc_full, u)   # (natoms,3)
    E = 0.5 * float(np.einsum("ia,ia->", u, Fu))
    return E, -Fu


class HarmonicCalculator(Calculator):
    """A harmonic PES from a phonopy fc (the smearing-blind backbone stand-in)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, ref_atoms, fc_full, **kw):
        super().__init__(**kw)
        self.ref_pos = np.array(ref_atoms.positions)
        self.cell = np.array(ref_atoms.cell)
        self.fc_full = fc_full

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        E, F = _harm_EF(self.fc_full, self.ref_pos, self.cell, np.array(atoms.positions))
        self.results = {"energy": E, "forces": F}


# ------------------------------------------------ the Friedel long-range module
class FriedelCorrection:
    """Builds Delta_fc(T_el) (full fc) from the validated template-envelope model.

    B_law, kappa_law: callables T_el(K) -> scalar (default = the graphene fit).
    """
    def __init__(self, ph_backbone, fc_backbone, fc_template, rmin=1.0, rmax=12.0,
                 B_law=None, kappa_law=None):
        self.ph = ph_backbone
        self.fc_bg = fc_backbone
        self.tabs, self.p2s = fm.pair_table(ph_backbone)
        self.D0 = fm.template_delta(fc_template, fc_backbone, self.tabs)
        self.rmin, self.rmax = rmin, rmax
        # graphene defaults (from fit_friedel few-shot law)
        self.B_law = B_law or (lambda T: 1.08)
        self.kappa_law = kappa_law or (lambda T: 8.318e-04 * T ** 0.61)

    def delta_fc_full(self, T_el):
        B, kap = float(self.B_law(T_el)), float(self.kappa_law(T_el))
        comp = fm.add_template(self.fc_bg, self.tabs, self.D0,
                               B, kap, self.rmin, self.rmax) - self.fc_bg
        return full_fc(self.ph, comp)


class FriedelMACECalculator(Calculator):
    """base_calc (any ASE calculator: MACE, ...) + additive harmonic Friedel term
    at electronic temperature T_el.  ref_atoms = the reference supercell."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, ref_atoms, corr: FriedelCorrection, T_el, **kw):
        super().__init__(**kw)
        self.base = base_calc
        self.ref_pos = np.array(ref_atoms.positions)
        self.cell = np.array(ref_atoms.cell)
        self.dfc = corr.delta_fc_full(T_el)
        self.T_el = T_el

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.base.calculate(atoms, ["energy", "forces"], system_changes)
        Eb = self.base.results["energy"]; Fb = np.array(self.base.results["forces"])
        Ec, Fc = _harm_EF(self.dfc, self.ref_pos, self.cell, np.array(atoms.positions))
        self.results = {"energy": Eb + Ec, "forces": Fb + Fc}


# --------------------------------------------------------------- fc from a calc
def fc2_from_calc(ph, calc, distance=0.01, subtract_ref=False):
    """Run phonopy finite displacement THROUGH an ASE calculator -> compact fc.

    subtract_ref: subtract the undisplaced-supercell forces (needed when the base
    MLIP is not exactly at the reference geometry, so F_ref != 0)."""
    from phonopy import Phonopy
    ph2 = Phonopy(ph.unitcell, supercell_matrix=ph.supercell_matrix,
                  primitive_matrix=ph.primitive_matrix)
    ph2.generate_displacements(distance=distance)
    from phonon_accel.phonons import phonopy_to_ase  # ordering-safe converter
    F0 = 0.0
    if subtract_ref:
        at0 = phonopy_to_ase(ph2.supercell)
        calc.calculate(at0, ["forces"])
        F0 = np.array(calc.results["forces"])
    forces = []
    for sc in ph2.supercells_with_displacements:
        at = phonopy_to_ase(sc)
        calc.calculate(at, ["forces"])
        forces.append(np.array(calc.results["forces"]) - F0)
    ph2.forces = np.array(forces)
    ph2.produce_force_constants()
    return ph2, ph2.force_constants


def main():
    sys.path.insert(0, str(ROOT / "src"))
    print("# validate: HarmonicCalculator(dg0.080 backbone) + Friedel(T_el)")
    ph_bg = fm.load_ph(YDIR / "graphene_sc6_dg0.080_phonopy.yaml")
    fc_bg = ph_bg.force_constants
    ph_ref = fm.load_ph(YDIR / "graphene_sc6_dg0.002_phonopy.yaml")
    corr = FriedelCorrection(ph_bg, fc_bg, ph_ref.force_constants)

    ref_sc = ph_bg.supercell
    from phonon_accel.phonons import phonopy_to_ase
    ref_atoms = phonopy_to_ase(ref_sc)
    base = HarmonicCalculator(ref_atoms, full_fc(ph_bg, fc_bg))

    # (0) base alone through the calculator -> must recover backbone kink 0.22
    ph_out, fc_out = fc2_from_calc(ph_bg, base)
    kK0 = fm.kink_of(ph_out, fc_out)[0]
    print(f"  base only            -> kink_K={kK0:5.2f}  (backbone DFT 0.22)")

    # (1) base + Friedel at several T_el -> kink(T_el), compare to DFT + direct
    DFT = {316: 22.63, 789: 15.72, 1579: 13.33, 3158: 9.75, 6315: 3.91}
    print(f"  {'T_el':>6} {'calc_kink':>9} {'direct_kink':>11} {'DFT':>6}")
    for T, kdft in DFT.items():
        calc = FriedelMACECalculator(base, ref_atoms, corr, T)
        ph_out, fc_out = fc2_from_calc(ph_bg, calc)
        kK = fm.kink_of(ph_out, fc_out)[0]
        # direct add_template reference (no calculator round-trip)
        comp = fm.add_template(fc_bg, corr.tabs, corr.D0,
                               corr.B_law(T), corr.kappa_law(T), 1.0, 12.0)
        kdir = fm.kink_of(ph_bg, comp)[0]
        print(f"  {T:>6} {kK:>9.2f} {kdir:>11.2f} {kdft:>6.2f}")
    print("# calc_kink should match direct_kink (calculator round-trip is exact);")
    print("# both approximate DFT via the few-shot law (backbone here = DFT dg0.080).")

    demo_mace_backbone(ph_bg, corr, ref_atoms)


def demo_mace_backbone(ph_bg, corr, ref_atoms):
    """Genuine deployment: base = a foundation MACE MLIP; the Friedel module adds
    the (E)-channel smearing dependence the MLIP is structurally blind to."""
    try:
        from mace.calculators import mace_mp
    except Exception as e:
        print(f"\n# [MACE demo skipped: {e}]"); return
    print("\n# MACE-backbone deployment: base = MACE-MP-0 (medium), + Friedel(T_el)")
    try:
        mace_calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
    except Exception as e:
        print(f"# [MACE load failed: {e}]"); return
    # MACE backbone alone (smearing-blind); subtract ref forces (MACE geom != DFT)
    _, fc_m = fc2_from_calc(ph_bg, mace_calc, subtract_ref=True)
    kb = fm.kink_of(ph_bg, fc_m)[0]
    print(f"  MACE backbone only                 -> kink_K={kb:5.2f}  (smearing-blind)")
    for T in (316, 6315):
        calc = FriedelMACECalculator(mace_calc, ref_atoms, corr, T)
        _, fc_t = fc2_from_calc(ph_bg, calc, subtract_ref=True)
        kK = fm.kink_of(ph_bg, fc_t)[0]
        print(f"  MACE + Friedel(T_el={T:>5} K)          -> kink_K={kK:5.2f}")
    print("# the SAME MACE backbone now yields a T_el-dependent K cusp -- the "
          "smearing physics the bare MLIP cannot represent, added analytically.")


if __name__ == "__main__":
    main()
