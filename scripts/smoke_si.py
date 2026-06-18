"""M0 smoke test: compute the Si phonon dispersion with a foundation MLIP.

Verifies the whole pipeline end-to-end (structure -> relax -> displacements ->
forces -> force constants -> bands/DOS/thermal/diagnostics) on the worked
example. Si is the canonical benchmark: two atoms, well-known dispersion with
TO/LO ~ 15.5 THz at Gamma, no imaginary modes when done right.

Usage:
    conda run -n phonon python scripts/smoke_si.py --model mace --supercell 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phonon_accel import PhononCalculation
from phonon_accel import structures
from phonon_accel.metrics import stability_flags
from phonon_accel.mlip_calc import get_calculator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace")
    ap.add_argument("--device", default=None)
    ap.add_argument("--supercell", type=int, default=3)
    ap.add_argument("--displacement", type=float, default=0.03)
    args = ap.parse_args()

    print(f"[1/4] building Si and {args.model} calculator (device auto)...")
    atoms = structures.builtin("Si")
    calc = get_calculator(args.model, device=args.device)

    print("[2/4] relaxing cell+positions...")
    atoms = structures.relax(atoms, calc, fmax=1e-4)
    # bulk() returns the primitive cell: |a_prim| = a_conv / sqrt(2)
    print(f"      |a_prim| = {atoms.cell.lengths()[0]:.4f} Ang  (exp 3.840 = 5.431/sqrt2)")

    print(f"[3/4] phonons, {args.supercell}x{args.supercell}x{args.supercell} supercell...")
    t0 = time.perf_counter()
    phon = PhononCalculation(
        atoms,
        supercell_matrix=(args.supercell,) * 3,
        displacement=args.displacement,
    )
    print(f"      n_displacements = {phon.n_displacements}")
    result = phon.run_all(calculator=calc)
    wall = time.perf_counter() - t0

    SI_OMEGA_MAX_REF = 15.5  # THz, experimental/DFT-PBE LO/TO at Gamma
    omega_max = float(result.mesh_frequencies.max())
    soft_pct = 100.0 * (omega_max - SI_OMEGA_MAX_REF) / SI_OMEGA_MAX_REF

    print("[4/4] results:")
    print(f"      omega_max                      : {omega_max:.2f} THz "
          f"(ref ~{SI_OMEGA_MAX_REF}; softening {soft_pct:+.0f}%)")
    print(f"      min frequency                  : {result.min_frequency:.3f} THz")
    print(f"      imaginary modes on mesh        : {result.n_imaginary_mesh}")
    print(f"      ASR residual at Gamma          : {result.asr_residual:.2e} THz")
    print(f"      C_v(300K)                      : {result.heat_capacity[30]:.2f} J/K/mol")
    print(f"      force eval time                : {result.timing.get('forces_s', float('nan')):.2f} s")
    print(f"      total wall time                : {wall:.2f} s")
    print("      stability:", stability_flags(result))

    # The smoke test validates the *pipeline*, not DFT-accuracy of the model:
    # acoustic sum rule satisfied, Si dynamically stable, sensible heat capacity.
    pipeline_ok = (
        result.asr_residual < 1e-3          # ASR enforced
        and result.min_frequency > -0.2     # Si has no real imaginary modes
        and 30.0 < result.heat_capacity[30] < 50.0  # Cv(300K) per 2-atom cell
        and omega_max > 5.0
    )
    print("\nPIPELINE:", "PASS" if pipeline_ok else "FAIL")
    if abs(soft_pct) > 8:
        print(f"NOTE: {args.model} softens Si by {soft_pct:+.0f}% vs reference "
              f"-> exactly the failure mode Line A studies/fixes.")
    return 0 if pipeline_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
