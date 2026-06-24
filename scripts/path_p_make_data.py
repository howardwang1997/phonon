"""Path P (one-shot) step A: thermal-config DFT-force data for anharmonic distillation.

The harmonic-distilled graphene model (M1.1b) is only accurate for *small*
displacements. Path P fixes the anharmonic PES by labelling the configurations the
thermal simulation actually visits with real DFT forces. Here:
  1. run Langevin MD with the graphene-FT (harmonic) model on an NxN graphene cell,
  2. DFT single-point (QE) each thermal snapshot -> REF energy + forces,
  3. also store the harmonic-model forces, so we can report its large-displacement
     error (the gap Path P closes),
  4. write train/test extxyz (DFT-labelled).

    conda run -n phonon python scripts/path_p_make_data.py \
        --model results/finetune_graphene/ft_graphene.model \
        --pw .../pw.x --mpirun .../mpirun --nproc 8 --supercell 3 \
        --T 300 --nsnap 40 --kpts 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

import td_common as tdc
from m1_1b_graphene_dft import make_espresso


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="graphene-FT (harmonic) MACE model")
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--pseudo-dir", default="pseudo")
    ap.add_argument("--pseudo", default="C_ONCV_PBE-1.2.upf")
    ap.add_argument("--a", type=float, default=2.46)
    ap.add_argument("--supercell", type=int, default=3)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--nsnap", type=int, default=40)
    ap.add_argument("--equil", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--ecutwfc", type=float, default=60.0)
    ap.add_argument("--ecutrho", type=float, default=240.0)
    ap.add_argument("--kpts", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/path_p")
    a = ap.parse_args()

    pdir = (ROOT / a.pseudo_dir) if not Path(a.pseudo_dir).is_absolute() else Path(a.pseudo_dir)
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = ROOT / "results/path_p/dft"
    work.mkdir(parents=True, exist_ok=True)

    # 1) MD with the harmonic graphene-FT model
    calc = tdc.get_mace_calc(a.model, device=a.device)
    prim = tdc.build_monolayer("graphene", a=a.a)
    cell = prim.repeat((a.supercell, a.supercell, 1))
    cell.wrap()
    print(f"[pathP] graphene {a.supercell}x{a.supercell} = {len(cell)} atoms; "
          f"MD {a.T:.0f} K with {a.model} ...", flush=True)
    cell.calc = calc
    MaxwellBoltzmannDistribution(cell, temperature_K=a.T, rng=np.random.default_rng(0))
    Stationary(cell); ZeroRotation(cell)
    dyn = Langevin(cell, a.dt * units.fs, temperature_K=a.T, friction=0.02,
                   rng=np.random.default_rng(1))
    dyn.run(a.equil)
    snaps = []
    for _ in range(a.nsnap):
        dyn.run(a.stride)
        s = cell.copy()
        s.info["mace_forces_rms"] = float(np.sqrt(np.mean(cell.get_forces() ** 2)))
        s.arrays["mace_forces"] = cell.get_forces()
        snaps.append(s)
    print(f"[pathP] sampled {len(snaps)} snapshots", flush=True)

    # 2) DFT-label each snapshot
    cm_err = []
    labelled = []
    t0 = time.perf_counter()
    for i, s in enumerate(snaps):
        d = work / f"cfg-{i:03d}"
        d.mkdir(exist_ok=True)
        s.calc = make_espresso(a.pw, a.mpirun, a.nproc, pdir, a.pseudo,
                               a.ecutwfc, a.ecutrho, a.kpts, directory=d)
        Fdft = s.get_forces()
        Edft = s.get_potential_energy()
        at = s.copy()
        at.calc = None
        at.info["energy"] = Edft
        at.info["REF_energy"] = Edft
        at.arrays["forces"] = Fdft
        at.arrays["REF_forces"] = Fdft
        # harmonic-model error on this thermal config (the gap Path P closes)
        Fmace = s.arrays["mace_forces"]
        rmse = float(np.sqrt(np.mean((Fmace - Fdft) ** 2))) * 1000  # meV/A
        cm_err.append(rmse)
        labelled.append(at)
        print(f"[pathP] cfg {i}: |F_dft|={np.abs(Fdft).max():.2f} eV/A  "
              f"harmonic-model err={rmse:.0f} meV/A  ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    # 3) split + write
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(labelled))
    ntest = max(2, int(len(labelled) * a.test_frac))
    test = [labelled[i] for i in idx[:ntest]]
    train = [labelled[i] for i in idx[ntest:]]
    nval = max(1, int(len(train) * 0.15))
    write(out / "train.xyz", train[nval:], format="extxyz")
    write(out / "val.xyz", train[:nval], format="extxyz")
    write(out / "test.xyz", test, format="extxyz")
    summary = {
        "n_train": len(train) - nval, "n_val": nval, "n_test": len(test),
        "supercell": a.supercell, "T": a.T, "natoms": len(cell),
        "harmonic_model_force_rmse_meVA": round(float(np.mean(cm_err)), 1),
        "source_model": a.model,
    }
    (out / "split.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[pathP] harmonic graphene-FT force error on thermal configs = "
          f"{np.mean(cm_err):.0f} meV/A (the anharmonic gap) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
