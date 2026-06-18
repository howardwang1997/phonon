"""Line A benchmark harness: foundation MLIP x material -> phonon metrics.

Two reference modes:

* ``REFERENCE_OMEGA_MAX`` — quick literature/experimental highest-frequency
  values (THz) for the built-in crystals, used for the worked example and
  fast softening tables before the full DFPT reference is wired in;
* full per-q DFPT reference (Petretto/MDR) — compared via ``metrics.compare``
  once reference ``PhononResult`` objects are available.

A "row" is one (model, material) evaluation: converged ω_max, the softening
vs reference, dynamical-stability flags, and timing.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

from . import structures
from .mlip_calc import get_calculator
from .phonons import PhononCalculation


# Experimental / DFT-PBE highest phonon frequency (THz) for built-in cells.
# Sources: standard phonon literature (LO/TO at Gamma or zone-boundary max).
# These are coarse references for the worked example; the full benchmark
# replaces them with the Petretto 2018 DFPT database per-q dispersions.
REFERENCE_OMEGA_MAX = {
    "Si": 15.5,    # LTO(Gamma) 15.53 THz
    "Ge": 9.1,     # LTO(Gamma) ~9.0 THz
    "C": 39.9,     # 1332 cm^-1 diamond Raman
    "GaAs": 8.8,   # LO ~292 cm^-1
    "NaCl": 7.9,   # LO ~264 cm^-1
    "MgO": 21.4,   # LO ~718 cm^-1
    "Al": 9.9,     # max phonon ~9.9 THz
}


@dataclass
class BenchRow:
    model: str
    material: str
    omega_max: float = float("nan")
    omega_max_ref: float = float("nan")
    softening_pct: float = float("nan")
    min_frequency: float = float("nan")
    n_imaginary: int = -1
    asr_residual: float = float("nan")
    cv_300k: float = float("nan")
    n_displacements: int = -1
    force_time_s: float = float("nan")
    error: str = ""

    def as_row(self) -> dict:
        d = self.__dict__.copy()
        return d


def run_material(
    model: str,
    material: str,
    calc=None,
    device: Optional[str] = None,
    supercell=(3, 3, 3),
    displacement: float = 0.03,
    mesh=(24, 24, 24),
    relax_fmax: float = 1e-4,
) -> BenchRow:
    """Run one (model, material) phonon evaluation; never raises."""
    row = BenchRow(model=model, material=material)
    row.omega_max_ref = REFERENCE_OMEGA_MAX.get(material, float("nan"))
    try:
        if calc is None:
            calc = get_calculator(model, device=device)
        atoms = structures.builtin(material)
        atoms = structures.relax(atoms, calc, fmax=relax_fmax)
        phon = PhononCalculation(
            atoms, supercell_matrix=supercell, displacement=displacement
        )
        row.n_displacements = phon.n_displacements
        res = phon.run_all(calculator=calc, mesh=mesh)
        row.omega_max = float(res.mesh_frequencies.max())
        row.min_frequency = float(res.min_frequency)
        row.n_imaginary = int(res.n_imaginary_mesh)
        row.asr_residual = float(res.asr_residual)
        row.cv_300k = float(res.heat_capacity[30])
        row.force_time_s = float(res.timing.get("forces_s", float("nan")))
        if row.omega_max_ref == row.omega_max_ref:  # not nan
            row.softening_pct = 100.0 * (row.omega_max - row.omega_max_ref) / row.omega_max_ref
    except Exception as exc:  # keep the table going on failures
        row.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


def run_material_vs_dfpt(
    model: str,
    mp_id: str,
    calc=None,
    device: Optional[str] = None,
    displacement: float = 0.03,
    mesh=(24, 24, 24),
    relax: bool = True,
    relax_fmax: float = 1e-4,
):
    """Full-band MLIP-vs-DFPT comparison for one MDR material.

    Runs the MLIP on the reference (DFT) unit cell with the *same* supercell
    matrix so band paths match point-for-point, then compares against the
    MDR/PhononDB DFPT reference. Returns a ``PhononMetrics`` row (or a dict
    with an ``error`` key on failure).
    """
    from . import reference, structures
    from .metrics import compare
    from .phonons import PhononCalculation, ase_to_phonopy

    try:
        ref_res, ref_ph = reference.reference_result(mp_id, mesh=mesh)
        atoms = reference.reference_atoms(mp_id)
        sc_matrix = ref_ph.supercell_matrix.tolist()

        if calc is None:
            calc = get_calculator(model, device=device)
        if relax:
            atoms = structures.relax(atoms, calc, fmax=relax_fmax)

        phon = PhononCalculation(atoms, supercell_matrix=sc_matrix, displacement=displacement)
        pred_res = phon.run_all(calculator=calc, mesh=mesh)

        m = compare(pred_res, ref_res)
        row = m.as_row()
        row.update(model=model, mp_id=mp_id, n_displacements=phon.n_displacements, error="")
        return row
    except Exception as exc:
        traceback.print_exc()
        return dict(model=model, mp_id=mp_id, error=f"{type(exc).__name__}: {exc}")


def run_dfpt_benchmark(models, mp_ids, device=None, **kwargs):
    """Return a DataFrame of full-band MLIP-vs-DFPT metrics over MDR materials."""
    import pandas as pd

    rows = []
    for model in models:
        calc = None
        try:
            calc = get_calculator(model, device=device)
        except Exception as exc:
            print(f"[skip] could not load {model}: {exc}")
            continue
        for mp_id in mp_ids:
            row = run_material_vs_dfpt(model, mp_id, calc=calc, **kwargs)
            if row.get("error"):
                print(f"  {model:>10} / {mp_id:<10} ERR {row['error']}")
            else:
                print(
                    f"  {model:>10} / {mp_id:<10} "
                    f"freq_MAE={row['freq_mae']:.3f} THz  "
                    f"wmax_err={row['omega_max_error']:+.2f}  "
                    f"imag(pred/ref)={row['n_imaginary_pred']}/{row['n_imaginary_ref']}"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def run_benchmark(
    models: list[str],
    materials: list[str],
    device: Optional[str] = None,
    **kwargs,
):
    """Return a pandas DataFrame of (model x material) rows."""
    import pandas as pd

    rows = []
    for model in models:
        calc = None
        try:
            calc = get_calculator(model, device=device)
        except Exception as exc:
            print(f"[skip] could not load {model}: {exc}")
            for mat in materials:
                r = BenchRow(model=model, material=mat, error=f"load failed: {exc}")
                rows.append(r.as_row())
            continue
        for mat in materials:
            t0 = time.perf_counter()
            r = run_material(model, mat, calc=calc, **kwargs)
            print(
                f"  {model:>10} / {mat:<5} "
                f"omega_max={r.omega_max:6.2f} THz  "
                f"soft={r.softening_pct:+6.1f}%  "
                f"imag={r.n_imaginary:>3}  "
                f"({time.perf_counter()-t0:.1f}s)"
                + (f"  ERR {r.error}" if r.error else "")
            )
            rows.append(r.as_row())
    return pd.DataFrame(rows)
