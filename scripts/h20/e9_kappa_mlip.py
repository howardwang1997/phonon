"""E9 — lattice thermal conductivity κ from MLIP forces (phono3py RTA).

The downstream "impact" line: does the (FC-distilled) MLIP give DFT-quality κ?
This computes fc2 + fc3 with MLIP forces, runs the RTA solver, and writes
κ(T) plus the % error vs a known experimental value. The apples-to-apples DFT
reference at the same (supercell, mesh) is produced off-box by scripts/dft_kappa.py.

    python scripts/h20/e9_kappa_mlip.py --name Si --structure diamond --a 5.43 \
        --model medium --supercell 2,2,2 --mesh 19 --kappa-exp 140 \
        --out results/h20/e9/Si_medium.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np   # noqa: E402


def build_bulk(name, structure, a, c=None):
    from ase.build import bulk
    if structure in ("zincblende", "rocksalt", "wurtzite"):
        # two-species: name like "GaAs"; ase parses the formula
        kw = dict(crystalstructure=structure, a=a)
        if c:
            kw["c"] = c
        return bulk(name, **kw)
    return bulk(name, structure, a=a)


def forces_on(supercells, calc):
    from phonon_accel.phonons import phonopy_to_ase
    out = []
    for sc in supercells:
        at = phonopy_to_ase(sc)
        at.calc = calc
        out.append(at.get_forces())
    return np.array(out)


def main() -> int:
    import td_common as tdc
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--structure", default="diamond")
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--model-type", dest="model_type", default="mace")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--supercell", default="2,2,2")
    ap.add_argument("--mesh", type=int, default=19)
    ap.add_argument("--temps", default="200,300,400,500,600")
    ap.add_argument("--kappa-exp", dest="kappa_exp", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tag = a.tag or (Path(a.model).stem if a.model.endswith(".model") else a.model)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[e9:{a.name}/{tag}] {out.name} exists -> skip", flush=True)
        return 0

    calc = tdc.get_calc(a.model_type, a.model, device=a.device)
    sc = [int(x) for x in a.supercell.split(",")]
    smat = [[sc[0], 0, 0], [0, sc[1], 0], [0, 0, sc[2]]]

    at = build_bulk(a.name, a.structure, a.a, a.c)
    at.calc = calc
    # light relax of positions/cell would shift κ; bulk reference cells are near
    # equilibrium already and we want the SAME geometry as the DFT reference.
    uc = PhonopyAtoms(symbols=at.get_chemical_symbols(), cell=at.cell.array,
                      scaled_positions=at.get_scaled_positions())
    ph3 = Phono3py(uc, supercell_matrix=smat, phonon_supercell_matrix=smat,
                   primitive_matrix="auto")
    ph3.generate_displacements()

    t0 = time.perf_counter()
    ph3.forces = forces_on(ph3.supercells_with_displacements, calc)
    n_fc3 = len(ph3.supercells_with_displacements)
    if ph3.phonon_supercells_with_displacements:
        ph3.phonon_forces = forces_on(ph3.phonon_supercells_with_displacements, calc)
    ph3.produce_fc2()
    ph3.produce_fc3()
    print(f"[e9:{a.name}/{tag}] {n_fc3} fc3 force sets in {time.perf_counter()-t0:.0f}s", flush=True)

    ph3.mesh_numbers = [a.mesh] * 3
    ph3.init_phph_interaction()
    temps = [int(x) for x in a.temps.split(",")]
    ph3.run_thermal_conductivity(temperatures=temps, is_isotope=True)
    kappa = ph3.thermal_conductivity.kappa[0]    # (n_T, 6) Voigt
    kdict = {str(T): round(float(np.mean(kappa[i, :3])), 3) for i, T in enumerate(temps)}
    k300 = kdict.get("300")
    err = (round(100 * (k300 - a.kappa_exp) / a.kappa_exp, 1)
           if (a.kappa_exp and k300 is not None) else None)
    res = {"material": a.name, "model": tag, "method": "MLIP(fc2+fc3)",
           "supercell": sc, "mesh": a.mesh, "kappa": kdict,
           "kappa_exp_300K": a.kappa_exp, "kappa_err_pct_300K": err,
           "n_fc3_forces": n_fc3}
    out.write_text(json.dumps(res, indent=2))
    print(f"[e9:{a.name}/{tag}] κ(300K)={k300} W/mK (exp {a.kappa_exp}, err {err}%) "
          f"-> {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
