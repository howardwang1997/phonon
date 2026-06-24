"""Si (mp-149) phonon density of states — DFPT vs MACE-baseline vs FC-distilled:
the DOS counterpart to the Fig-2 dispersion panel. Reuses the same phonopy 20^3
mesh that ``build_result_from_phonon`` already computes (the dispersion cache just
never saved the DOS arrays), so the bands and the DOS share identical settings.
Computes once, caches to npz, then overlays the three with the shared figure style.

    conda run -n phonon python scripts/plot_si_dos.py \
        --ft-model results/finetune_mace/ft_phonon.model
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np


def _mlip_dos(model, atoms0, sc, device):
    from phonon_accel import structures
    from phonon_accel.mlip_calc import get_calculator
    from phonon_accel.phonons import PhononCalculation

    calc = get_calculator("mace", device=device, model=model)
    atoms = structures.relax(atoms0.copy(), calc, fmax=1e-4)
    phon = PhononCalculation(atoms, supercell_matrix=sc, displacement=0.03)
    res = phon.run_all(calculator=calc)
    return np.asarray(res.dos_frequencies), np.asarray(res.dos)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp-id", default="mp-149")
    ap.add_argument("--baseline", default="small")
    ap.add_argument("--ft-model", default="results/finetune_mace/ft_phonon.model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache", default="results/figures/data/si_dos.npz")
    ap.add_argument("--out", default="results/figures/si_dos.png")
    a = ap.parse_args()

    cache = ROOT / a.cache
    if cache.exists():
        d = {k: v for k, v in np.load(cache, allow_pickle=True).items()}
        print("loaded cache", cache)
    else:
        from phonon_accel import reference

        ref_res, ref_ph = reference.reference_result(a.mp_id)
        atoms0 = reference.reference_atoms(a.mp_id)
        sc = ref_ph.supercell_matrix.tolist()
        print("computing phonon DOS (20^3 mesh): DFPT / baseline / FC-distilled ...")
        d = {
            "dfpt_f": np.asarray(ref_res.dos_frequencies),
            "dfpt_dos": np.asarray(ref_res.dos),
            "formula": np.array(str(ref_res.formula)),
        }
        d["base_f"], d["base_dos"] = _mlip_dos(a.baseline, atoms0, sc, a.device)
        d["ft_f"], d["ft_dos"] = _mlip_dos(a.ft_model, atoms0, sc, a.device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **d)
        print("wrote cache", cache)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from plot_style import C, set_style

        set_style()
    except Exception:
        C = {"dfpt": "#000000", "mace": "#d1495b", "finetuned": "#2a9d4a"}

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.fill_between(d["dfpt_f"], d["dfpt_dos"], color=C["dfpt"], alpha=0.07, zorder=0)
    ax.plot(d["dfpt_f"], d["dfpt_dos"], color=C["dfpt"], ls="--", lw=1.4,
            label="DFPT (reference)", zorder=3)
    ax.plot(d["base_f"], d["base_dos"], color=C["mace"], lw=1.5,
            label="MACE baseline", zorder=2)
    ax.plot(d["ft_f"], d["ft_dos"], color=C["finetuned"], lw=1.7,
            label="FC-distilled", zorder=2)

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phonon DOS (states / THz / cell)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(f"{str(d['formula'])} phonon density of states")
    fig.tight_layout()

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))

    # report omega_max from the DOS support (shows softening + cure)
    for k, name in (("dfpt", "DFPT"), ("base", "baseline"), ("ft", "FC-distilled")):
        f, dos = np.asarray(d[f"{k}_f"]), np.asarray(d[f"{k}_dos"])
        wmax = float(f[np.nonzero(dos > 1e-3 * dos.max())[0][-1]])
        print(f"  {name:13s} omega_max (DOS support) = {wmax:.2f} THz")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
