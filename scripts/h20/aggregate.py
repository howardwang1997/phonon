"""Aggregate every H20 job output into master tables + the (L)-channel origin
map, and write results/h20/SUMMARY.md. Tolerant: any group with no files is
simply skipped, so this is useful mid-campaign too.

    python scripts/h20/aggregate.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "results" / "h20"


def _f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# E1 benchmark atlas
# --------------------------------------------------------------------------- #
def agg_e1(lines: list[str]):
    rows = []
    for p in sorted(glob.glob(str(H / "e1" / "*.csv"))):
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    if not rows:
        return
    # write merged master table
    keys = sorted({k for r in rows for k in r})
    with open(H / "e1_master.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # per-model summary
    by = {}
    for r in rows:
        m = r.get("model", "?")
        d = by.setdefault(m, {"n": 0, "mae": [], "soft": [], "imag": 0, "err": 0})
        d["n"] += 1
        if r.get("error"):
            d["err"] += 1
            continue
        d["mae"].append(_f(r.get("freq_mae")))
        d["soft"].append(_f(r.get("softening_pct")))
        d["imag"] += int(_f(r.get("n_imaginary_pred"), 0))

    def mean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    lines.append("## E1 — benchmark atlas (MLIP vs DFPT, MDR)\n")
    lines.append("| model | n | mean freq-MAE (THz) | mean softening (%) | Σ imaginary | errors |")
    lines.append("|---|---|---|---|---|---|")
    for m in sorted(by):
        d = by[m]
        lines.append(f"| {m} | {d['n']} | {mean(d['mae']):.3f} | "
                     f"{mean(d['soft']):+.1f} | {d['imag']} | {d['err']} |")
    if "distilled" in by:
        base = "mace"
        if base in by:
            lines.append(f"\n**Distillation effect (full pool):** mace {mean(by[base]['mae']):.3f} → "
                         f"distilled {mean(by['distilled']['mae']):.3f} THz freq-MAE; "
                         f"imaginary {by[base]['imag']} → {by['distilled']['imag']}.")
    lines.append("")


# --------------------------------------------------------------------------- #
# E3/E4 fine-tune evals (results/ablation/eval_*.csv)
# --------------------------------------------------------------------------- #
def agg_finetune(lines: list[str]):
    files = sorted(glob.glob(str(ROOT / "results" / "ablation" / "eval_*.csv")))
    if not files:
        return
    lines.append("## E3/E4 — FC-distillation fine-tunes (held-out generalization)\n")
    lines.append("| job | variant | split | freq-MAE (THz) | softening (%) | Σ imaginary |")
    lines.append("|---|---|---|---|---|---|")
    for p in files:
        job = Path(p).stem.replace("eval_", "")
        agg = {}
        with open(p) as f:
            for r in csv.DictReader(f):
                k = (r.get("variant", "?"), r.get("split", "?"))
                a = agg.setdefault(k, {"mae": [], "soft": [], "imag": 0})
                a["mae"].append(_f(r.get("freq_mae")))
                a["soft"].append(_f(r.get("softening_pct")))
                a["imag"] += int(_f(r.get("n_imaginary_pred"), 0))
        for (variant, split), a in sorted(agg.items()):
            xs = [x for x in a["mae"] if x == x]
            sx = [x for x in a["soft"] if x == x]
            lines.append(f"| {job} | {variant} | {split} | "
                         f"{(sum(xs)/len(xs) if xs else float('nan')):.3f} | "
                         f"{(sum(sx)/len(sx) if sx else float('nan')):+.1f} | {a['imag']} |")
    lines.append("")


# --------------------------------------------------------------------------- #
# (L)-channel TMD-family origin map
# --------------------------------------------------------------------------- #
def _parse_sscha(p):
    meta, tmin = {}, float("inf")
    rows = []
    for line in Path(p).read_text().splitlines():
        if line.startswith("#"):
            for tok in line.lstrip("# ").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
        elif line and not line.startswith("T_K"):
            parts = line.split(",")
            if len(parts) >= 2:
                # (T, sscha_minfreq_cm, n_imag)
                rows.append((_f(parts[0]), _f(parts[1]), _f(parts[2]) if len(parts) > 2 else None))
    minf = min((m for _, m, _ in rows), default=float("nan"))
    return meta, minf, rows


# 4x4x1 TMD supercell of a 3-atom MX2 primitive => 48 atoms => 144 modes.
# SSCHA free-energy-Hessian regression on a noisy MLIP can diverge (all 144
# modes imaginary, |minfreq| > 1e3 cm^-1). Detect that so the origin map does
# not report garbage min-freqs as physical "L-unstable" signals.
_SSCHA_TOTAL_MODES_441 = 144
_SSCHA_ABSURD_CM = 500.0   # |minfreq| above this at any T => non-converged


def _sscha_relabel(bare, rows):
    """Override the run-time label if the SSCHA Hessian diverged. Returns
    (label, converged_topT_minf_cm). Falls back to the bare harmonic fc2 as the
    only reliable signal when SSCHA did not converge."""
    if not rows:
        return "?", float("nan")
    topT = max(rows, key=lambda r: r[0])
    top_minf, top_nimag = topT[1], topT[2]
    absurd = any(abs(m) > _SSCHA_ABSURD_CM for _, m, _ in rows)
    half_imag = (top_nimag is not None and top_nimag > 0.5 * _SSCHA_TOTAL_MODES_441)
    if absurd or half_imag:
        # SSCHA diverged: trust only the bare harmonic fc2.
        return ("L-stable" if (bare == bare and bare >= -1.0)
                else "L-unstable(nh,SSCHA-diverged)"), top_minf
    return None, top_minf   # None => keep the run-time label


def agg_lchannel(lines: list[str]):
    tri = sorted(glob.glob(str(H / "lchannel" / "*_triage_*.csv")))
    ssc = sorted(glob.glob(str(H / "lchannel" / "*_sscha_*.csv")))
    if not tri and not ssc:
        return
    # collect per (material, modeltag)
    mat = {}
    for p in tri:
        with open(p) as f:
            for r in csv.DictReader(f):
                key = (r["name"], r["model"])
                mat.setdefault(key, {})["harm"] = _f(r.get("harm_minfreq_THz"))
                mat[key]["q"] = f"({_f(r.get('qmin_a')):.2f},{_f(r.get('qmin_b')):.2f})"
                mat[key]["polytype"] = r.get("polytype", "")
                mat[key]["label_tri"] = r.get("label", "")
    for p in ssc:
        meta, minf, rows = _parse_sscha(p)
        key = (meta.get("name", Path(p).stem), meta.get("model", "?"))
        d = mat.setdefault(key, {})
        bare = _f(meta.get("bare_minfreq_cm"))
        override, top_minf = _sscha_relabel(bare, rows)
        d["sscha_label"] = override if override is not None else meta.get("label", "?")
        # show the highest-T (most physical) min-freq, not the divergent low-T one
        d["sscha_minf"] = top_minf
        d["bare"] = bare
        d["cdw"] = meta.get("cdw_exp_K", "null")
        d["polytype"] = d.get("polytype", meta.get("polytype", ""))

    lines.append("## (L)-channel — TMD-family origin map (MLIP+SSCHA screen)\n")
    lines.append("> Triage min-freq < 0 ⇒ MLIP sees harmonic softening. SSCHA label "
                 "`L-stable` for a *known-CDW* material ⇒ the instability is NOT "
                 "lattice-anharmonic ⇒ **electronic-driven candidate → (E)-channel rental**.\n")
    lines.append("| material | poly | model | harm min-freq (THz) @q | bare fc2 (cm⁻¹) | SSCHA min (cm⁻¹) | SSCHA label | exp T_CDW (K) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (name, model) in sorted(mat):
        d = mat[(name, model)]
        harm = d.get("harm", float("nan"))
        lines.append(f"| {name} | {d.get('polytype','')} | {model} | "
                     f"{harm:.3f} {d.get('q','')} | "
                     f"{d.get('bare', float('nan')):.1f} | "
                     f"{d.get('sscha_minf', float('nan')):.1f} | "
                     f"{d.get('sscha_label','—')} | {d.get('cdw','')} |")
    # the actionable shortlist
    elec = sorted({name for (name, model) in mat
                   if mat[(name, model)].get("sscha_label", "").startswith("L-stable")
                   and str(mat[(name, model)].get("cdw", "null")) not in ("null", "None", "")})
    if elec:
        lines.append(f"\n**→ Electronic-driven shortlist (rent FP64 (E)-channel for these): "
                     f"{', '.join(elec)}**")
    lines.append("")


# --------------------------------------------------------------------------- #
# E9 kappa
# --------------------------------------------------------------------------- #
def agg_e9(lines: list[str]):
    files = sorted(glob.glob(str(H / "e9" / "*.json")))
    if not files:
        return
    lines.append("## E9 — κ (MLIP forces + phono3py RTA)\n")
    lines.append("| material | model | κ(300K) W/mK | κ_exp(300K) | error % |")
    lines.append("|---|---|---|---|---|")
    for p in files:
        d = json.loads(Path(p).read_text())
        k = d.get("kappa", {}).get("300", "—")
        lines.append(f"| {d.get('material')} | {d.get('model')} | {k} | "
                     f"{d.get('kappa_exp_300K')} | {d.get('kappa_err_pct_300K')} |")
    lines.append("")


def main() -> int:
    lines = [f"# H20 campaign — results summary\n",
             "_Auto-generated by scripts/h20/aggregate.py. See "
             "docs/H20_EXPERIMENT_PLAN.md §3 for how each table feeds the next decision._\n"]
    agg_e1(lines)
    agg_finetune(lines)
    agg_lchannel(lines)
    agg_e9(lines)
    H.mkdir(parents=True, exist_ok=True)
    (H / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {(H / 'SUMMARY.md')}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
