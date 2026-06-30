"""Aggregate the V100 FP64 outputs into results/v100/SUMMARY.md — the (E)-channel
+ DFT-fc2 half of the origin map, designed to sit next to H20's (L)-channel
SUMMARY. Tolerant: skips anything not yet produced (useful mid-campaign).

    python scripts/v100/aggregate.py
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v100"))
V = ROOT / "results" / "v100"


def _f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main() -> int:
    import numpy as np
    import tmd_common as tc
    cfg = tc.load_config()
    lines = ["# V100 FP64 campaign — results summary\n",
             "_DFT fc2 (ground truth for the H20 (L)-screen) + (E)-channel "
             "(DFPT-smearing EPW γ_qν + nesting). Pairs with results/h20/SUMMARY.md._\n"]

    # --- DFT fc2: did DFT capture a soft mode per material? -----------------
    lines.append("## DFT fc2 — harmonic soft mode (ground truth)\n")
    lines.append("| material | poly | min freq (THz) | soft mode? |")
    lines.append("|---|---|---|---|")
    for m in cfg["materials"]:
        npz = V / "fc2" / f"disp_{m['name']}.npz"
        if not npz.exists():
            continue
        fr = np.load(npz)["frequencies"]
        fmin = float(fr.min())
        lines.append(f"| {m['name']} | {m['polytype']} | {fmin:.2f} | "
                     f"{'YES (CDW captured)' if fmin < -0.1 else 'no'} |")
    lines.append("")

    # --- (E)-channel: bands k_F (2k_F vs q_CDW) + nesting peak --------------
    lines.append("## (E)-channel electronic anchors — 2k_F & nesting\n")
    lines.append("| material | k_F on Γ-M (frac) | 2k_F≈q_CDW(0.67)? | nesting peak q | nesting-driven? |")
    lines.append("|---|---|---|---|---|")
    for m in cfg["materials"]:
        npz = V / "bands" / f"{m['name']}_ebands.npz"
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=True)
        kf = list(z["kF_GM"])
        two = any(abs(min(2 * c, 2 - 2 * c) - 0.667) < 0.12 for c in kf)
        nq, nx = z["nest_q"], z["nest_xi"]
        if len(nx) > 1:
            ipk = int(np.argmax(nx[1:]) + 1)
            qpk = f"{nq[ipk]:.2f}"
            nest = "YES" if abs(nq[ipk] - 1 / 3) < 0.06 else "no (EPC?)"
        else:
            qpk, nest = "—", "—"
        lines.append(f"| {m['name']} | {kf} | {'YES' if two else 'no'} | {qpk} | {nest} |")
    lines.append("")

    # --- (E)-channel EPW: lambda / gamma per material ----------------------
    csvp = Path("/root/tmd_epw_results.csv")
    if not csvp.exists():
        csvp = V / "tmd_epw_results.csv"
    if csvp.exists():
        lines.append("## (E)-channel EPW — λ / γ_qν\n")
        lines.append("| material | degauss | T_el (K) | min freq (cm⁻¹) | λ | λ_tr | max γ (meV) | NQ | NKF |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        with open(csvp) as f:
            for r in csv.DictReader(f):
                lines.append("| " + " | ".join([
                    r.get("material", "?"), r.get("degauss", ""), r.get("T_el", ""),
                    r.get("minfreq_cm", ""), r.get("lambda", ""), r.get("lambda_tr", ""),
                    r.get("maxgamma_meV", ""), r.get("NQ", ""), r.get("NKF", "")]) + " |")
        lines.append("\n> λ diverges at soft modes (1/ω²) — the robust observable is **γ_qν** "
                     "(max γ) and whether it peaks at q_CDW (EPC-driven) vs the nesting peak.")
    lines.append("")

    # --- Path-P datasets ready --------------------------------------------
    pp = sorted(glob.glob(str(ROOT / "data/v100/path_p/*/train.xyz")))
    if pp:
        mats = [Path(p).parent.name for p in pp]
        lines.append(f"## Path-P anharmonic DFT labels ready: {', '.join(mats)}\n")

    V.mkdir(parents=True, exist_ok=True)
    (V / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {V / 'SUMMARY.md'}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
