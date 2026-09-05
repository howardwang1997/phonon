"""Evaluate a raw or fine-tuned MACE backbone with the healing Friedel term.

The original script only supported a downloaded ``small`` foundation and had
no machine-readable output.  ``--model`` now also accepts a local ``.model``
file, which makes the interrupted 2060 fine-tune a verifiable ablation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import friedel_module as fm  # noqa: E402
from friedel_calc import FriedelCorrection, FriedelMACECalculator, fc2_from_calc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small",
                    help="foundation tag (e.g. small) or a fine-tuned .model path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--default-dtype", default="float32")
    ap.add_argument("--dgs", default="0.01,0.015,0.02,0.03,0.04,0.06,0.08,0.10,0.14")
    ap.add_argument("--reference", default="0.01")
    ap.add_argument("--label", default="foundation")
    ap.add_argument("--output", default="results/smearing_kink/deploy_foundation.csv")
    args = ap.parse_args()

    fd = ROOT / "results" / "graphene_kohn_fd"
    dgs = [value.strip() for value in args.dgs.split(",") if value.strip()]
    required = [fd / f"graphene_sc6_dg{dg}_phonopy.yaml" for dg in dgs]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing DFT reference(s): {missing}")

    ph = fm.load_ph(str(fd / "graphene_sc6_dg0.08_phonopy.yaml"))
    fc_dft = {dg: fm.load_ph(str(path)).force_constants for dg, path in zip(dgs, required)}
    healing = json.loads((fd / "healing_law.json").read_text())

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if model_path.is_file():
        from mace.calculators import MACECalculator
        mace = MACECalculator(model_paths=str(model_path), device=args.device,
                              default_dtype=args.default_dtype)
        try:
            model_description = str(model_path.relative_to(ROOT))
        except ValueError:
            model_description = str(model_path)
    else:
        from mace.calculators import mace_mp
        mace = mace_mp(model=args.model, device=args.device,
                       default_dtype=args.default_dtype)
        model_description = args.model

    from phonon_accel.phonons import phonopy_to_ase
    ref_atoms = phonopy_to_ase(ph.supercell)
    _, fc_backbone = fc2_from_calc(ph, mace, distance=0.03, subtract_ref=True)
    correction = FriedelCorrection(
        ph, fc_dft["0.08"], fc_dft[args.reference],
        B_law=lambda temp: healing["B0"] * max(
            0.0, 1.0 - (temp / healing["Tstar"]) ** healing["p"]
        ) ** healing["q"],
        kappa_law=lambda _temp: healing["kappa"],
    )
    backbone_kink = float(fm.kink_of(ph, fc_backbone)[0])

    print(f"# {args.label}: model={model_description}, backbone_kink={backbone_kink:.3f}")
    print(f"{'dg':>6} {'T_el':>7} {'DFT':>8} {'model':>8} {'abs_err':>8}")
    rows = []
    for dg in dgs:
        temperature = float(dg) * 157887
        dft_kink = float(fm.kink_of(ph, fc_dft[dg])[0])
        if dg == args.reference:
            model_kink = dft_kink
        else:
            calc = FriedelMACECalculator(mace, ref_atoms, correction, temperature)
            _, fc_temp = fc2_from_calc(ph, calc, distance=0.03, subtract_ref=True)
            model_kink = float(fm.kink_of(ph, fc_temp)[0])
        abs_error = abs(model_kink - dft_kink)
        rows.append({
            "label": args.label, "model": model_description, "degauss": dg,
            "T_el_K": f"{temperature:.3f}", "dft_kink_K": f"{dft_kink:.8f}",
            "model_kink_K": f"{model_kink:.8f}", "abs_error": f"{abs_error:.8f}",
            "backbone_kink_K": f"{backbone_kink:.8f}",
        })
        print(f"{dg:>6} {temperature:7.0f} {dft_kink:8.2f} {model_kink:8.2f} {abs_error:8.2f}")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output)
    mae = float(np.mean([float(row["abs_error"]) for row in rows]))
    print(f"# MAE kink_K={mae:.3f}; wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
