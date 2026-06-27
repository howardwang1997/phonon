"""Path-P NbSe2 analysis: (1) the CDW double-well E(A) along the soft eigenvector for
each MLIP model (vs DFT if a scan-energy npz is given), (2) force RMSE on a held-out
thermal test set vs DFT.

The money plot: E(A) for DFT vs harmonic-FT (flat/curved-up = no well) vs anharmonic-FT
(should reproduce the DFT double-well if Path-P worked).

    python scripts/path_p_nbse2_analyze.py --yaml results/vq3/nbse2_dft_phonopy.yaml \
        --model harmonicFT=results/finetune_nbse2/ft_nbse2.model \
        --model anharmFT=results/finetune_path_p_nbse2/ft.model \
        --device cpu --amax 0.18 --n-scan 21 \
        --test data/path_p_nbse2/test.xyz --out results/td_phonon/nbse2_pathp_analysis.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from path_p_nbse2_make_data import soft_eigen


def get_calc(model_path, device):
    try:
        import td_common as tdc
        return tdc.get_calc("mace", model_path, device=device)
    except Exception:
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=model_path, device=device, default_dtype="float64")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="results/vq3/nbse2_dft_phonopy.yaml")
    ap.add_argument("--model", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amax", type=float, default=0.18)
    ap.add_argument("--n-scan", type=int, default=21)
    ap.add_argument("--test", default="")
    ap.add_argument("--out", default="results/td_phonon/nbse2_pathp_analysis.npz")
    a = ap.parse_args()

    import phonopy
    ph = phonopy.load(a.yaml if Path(a.yaml).is_absolute() else ROOT / a.yaml,
                      is_compact_fc=False)
    atoms0, e_cart, w2, soft_cm = soft_eigen(ph)
    x0 = atoms0.get_positions()
    N = len(atoms0)
    A = np.linspace(-a.amax, a.amax, a.n_scan)
    mid = a.n_scan // 2
    print(f"# soft mode {soft_cm:.1f} cm^-1; double-well scan A in [{-a.amax},{a.amax}] x {a.n_scan}")

    out = {"A": A, "soft_cm": soft_cm}
    print(f"\n{'model':14s} E(A) along soft eigenvector (meV/cell, ref A=0):")
    for spec in a.model:
        name, _, mp = spec.partition("=")
        calc = get_calc(mp if Path(mp).is_absolute() else str(ROOT / mp), a.device)
        E = np.empty(a.n_scan)
        for i, amp in enumerate(A):
            at = atoms0.copy(); at.set_positions(x0 + amp * e_cart); at.calc = calc
            E[i] = at.get_potential_energy()
        E = (E - E[mid]) * 1000.0    # meV, ref A=0
        out[f"E_{name}"] = E
        # is there a double-well? min away from center
        imin = int(np.argmin(E)); well = E[imin]
        tag = (f"DOUBLE-WELL min {well:.0f} meV @ A={A[imin]:+.3f}"
               if imin != mid and well < -1.0 else "no well (stable / curved up)")
        print(f"{name:14s} center->edge {E[0]:+.0f} .. {E[mid]:+.0f} .. {E[-1]:+.0f} | {tag}")

    # DFT E(A) from the scan configs (if present in train/test)
    dft_A, dft_E = [], []
    for src in ("data/path_p_nbse2/train.xyz", "data/path_p_nbse2/test.xyz"):
        p = ROOT / src
        if p.exists():
            from ase.io import read
            for at in read(p, ":"):
                ct = at.info.get("config_type", "")
                if ct.startswith("scan_A"):
                    dft_A.append(float(ct.split("scan_A")[1]))
                    dft_E.append(float(at.info["REF_energy"]))
    if dft_A:
        dft_A = np.array(dft_A); dft_E = np.array(dft_E)
        o = np.argsort(dft_A); dft_A, dft_E = dft_A[o], dft_E[o]
        # ref to the A closest to 0
        dft_E = (dft_E - dft_E[np.argmin(np.abs(dft_A))]) * 1000.0
        out["dft_A"] = dft_A; out["dft_E"] = dft_E
        imin = int(np.argmin(dft_E))
        print(f"{'DFT':14s} center->edge {dft_E[0]:+.0f} .. {dft_E[-1]:+.0f} | "
              f"min {dft_E[imin]:+.0f} meV @ A={dft_A[imin]:+.3f}")

    # force RMSE on held-out thermal test set
    if a.test and (ROOT / a.test if not Path(a.test).is_absolute() else Path(a.test)).exists():
        from ase.io import read
        tp = a.test if Path(a.test).is_absolute() else ROOT / a.test
        configs = read(tp, ":")
        Fref = [np.asarray(at.arrays["REF_forces"]) for at in configs]
        frms = float(np.sqrt(np.mean(np.concatenate([f.ravel() for f in Fref]) ** 2)))
        print(f"\n# {len(configs)} thermal test configs; <|F_dft|>_rms = {frms*1000:.0f} meV/A")
        out["test_fref_rms_meV"] = frms * 1000
        for spec in a.model:
            name, _, mp = spec.partition("=")
            calc = get_calc(mp if Path(mp).is_absolute() else str(ROOT / mp), a.device)
            err = []
            for at, fr in zip(configs, Fref):
                at2 = at.copy(); at2.calc = calc
                err.append((at2.get_forces() - fr).ravel())
            rmse = float(np.sqrt(np.mean(np.concatenate(err) ** 2))) * 1000
            out[f"frmse_{name}"] = rmse
            print(f"{name:14s} force RMSE vs DFT = {rmse:.0f} meV/A")

    op = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    np.savez(op, **out)
    print(f"\nsaved -> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
