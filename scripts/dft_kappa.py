"""DFT lattice thermal conductivity from saved fc2/fc3 (phono3py RTA) — the
apples-to-apples DFT reference at the SAME settings the MLIP κ uses (sc, mesh).
Lets us compare MLIP κ vs DFT κ vs experiment without the converged-DFT/experiment
confound.

    python scripts/dft_kappa.py --material Si --fc-dir results/fc3 --mesh 21
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_fc(path, keys):
    with h5py.File(path, "r") as f:
        for k in keys:
            if k in f:
                return f[k][:]
    raise KeyError(f"none of {keys} in {path}")


def main() -> int:
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms
    from ase.build import bulk

    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Si")
    ap.add_argument("--fc-dir", default="results/fc3")
    ap.add_argument("--mesh", type=int, default=21)
    ap.add_argument("--out", default="results/kappa/Si_dft.json")
    args = ap.parse_args()
    d = Path(args.fc_dir)
    fc2 = load_fc(d / f"{args.material}_fc2.hdf5", ["fc2", "force_constants"])
    fc3 = load_fc(d / f"{args.material}_fc3.hdf5", ["fc3"])
    cells = np.load(d / f"{args.material}_cells.npz")
    n = int(round(cells["sc_matrix"][0][0]))

    a = {"Si": bulk("Si", "diamond", a=5.43)}[args.material]
    uc = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=a.cell.array,
                      scaled_positions=a.get_scaled_positions())
    ph3 = Phono3py(uc, supercell_matrix=[[n, 0, 0], [0, n, 0], [0, 0, n]], primitive_matrix="auto")
    ph3.fc2 = fc2
    ph3.fc3 = fc3
    ph3.mesh_numbers = [args.mesh] * 3
    ph3.init_phph_interaction()
    temps = list(range(200, 601, 100))
    ph3.run_thermal_conductivity(temperatures=temps, is_isotope=True)
    kappa = ph3.thermal_conductivity.kappa[0]
    out = {"material": args.material, "method": "DFT(fc2+fc3)", "sc": n, "mesh": args.mesh,
           "kappa": {str(T): round(float(np.mean(kappa[i, :3])), 2) for i, T in enumerate(temps)}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(__import__("json").dumps(out, indent=2))
    print(f"DFT kappa(300K) [sc {n}^3, mesh {args.mesh}] = {out['kappa'].get('300')} W/mK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
