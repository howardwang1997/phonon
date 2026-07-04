"""Family generalization of the damped-Friedel (E)-channel finding: does the
smearing-dependence of a CDW material's fc2 also decompose as a 2-parameter
thermally-damped Friedel term, reproducing its soft-mode(T_el) melting curve?

Unlike graphene (a KINK on the top branch), a CDW material has an imaginary SOFT
MODE; the observable is the minimum phonon frequency vs T_el. The core
decomposition (backbone + B(T_el)*exp(-kappa(T_el)*R)*D0) is material-agnostic
(operates on the fc2 tensor + pair geometry), so we reuse friedel_module and just
swap the readout to min-freq over a q-mesh.

    conda run -n phonon python scripts/smearing_kink/friedel_family.py \
        --tag 1T-VSe2 --glob 'results/vq_family/fc2_vse2_4x4_*/1T-VSe2_phonopy.yaml' \
        --tel-per-dg 157888 --mesh 12

backbone = most-smeared (largest degauss, Friedel most damped); template =
sharpest (smallest degauss, least-damped waveform).
"""
from __future__ import annotations
import sys, re, glob, argparse, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import friedel_module as fm
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def min_freq(ph, fc, mesh):
    """Minimum phonon frequency (THz) over an NxNx1 q-mesh (the soft mode)."""
    ph.force_constants = fc
    ph.run_mesh([mesh, mesh, 1], with_eigenvectors=False, is_gamma_center=True)
    return float(np.min(ph.get_mesh_dict()["frequencies"]))


def parse_dg(path):
    m = re.search(r"_([0-9]*\.?[0-9]+)[/_]", path) or re.search(r"([0-9]\.[0-9]{3})", path)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--glob", required=True, help="glob for the per-smearing phonopy yamls")
    ap.add_argument("--tel-per-dg", type=float, default=157888.0)
    ap.add_argument("--mesh", type=int, default=12)
    ap.add_argument("--rmin", type=float, default=1.0)
    ap.add_argument("--rmax", type=float, default=14.0)
    a = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / a.glob)) or glob.glob(a.glob))
    items = [(parse_dg(f), f) for f in files]
    items = sorted([(d, f) for d, f in items if d is not None])
    if len(items) < 3:
        print(f"# need >=3 smearings, found {len(items)}: {items}"); return
    print(f"# {a.tag}: {len(items)} smearings {[d for d,_ in items]}")

    phs, fcs = {}, {}
    for dg, f in items:
        ph = fm.load_ph(f); phs[dg] = ph; fcs[dg] = ph.force_constants
    dgs = [d for d, _ in items]
    bg, ref = dgs[-1], dgs[0]                     # backbone=most-smeared, template=sharpest
    tabs, _ = fm.pair_table(phs[bg])
    D0 = fm.template_delta(fcs[ref], fcs[bg], tabs)
    kappas = np.linspace(0.0, 2.0, 161)
    ph = phs[bg]

    print(f"# backbone=dg{bg}, template=dg{ref}; mesh {a.mesh}x{a.mesh}; tail [{a.rmin},{a.rmax}] A")
    print(f"{'dg':>7} {'T_el':>6} {'B':>7} {'kappa':>7} {'minf_model':>10} {'minf_DFT':>9} {'err':>6}")
    rows = []
    for dg in dgs:
        T = dg * a.tel_per_dg
        mf_dft = min_freq(phs[dg], fcs[dg], a.mesh)
        if dg == bg:
            B, kap = 0.0, 0.0
        elif dg == ref:
            B, kap = 1.0, 0.0
        else:
            B, kap, _ = fm.fit_template_env(fcs[dg], fcs[bg], tabs, D0, a.rmin, a.rmax, kappas)
        fc_m = fm.add_template(fcs[bg], tabs, D0, B, kap, a.rmin, a.rmax)
        mf_mod = min_freq(ph, fc_m, a.mesh)
        rows.append((dg, T, B, kap, mf_mod, mf_dft))
        print(f"{dg:>7.3f} {T:>6.0f} {B:>7.3f} {kap:>7.3f} {mf_mod:>10.3f} {mf_dft:>9.3f} {mf_mod-mf_dft:>6.2f}")
    err = np.mean([abs(r[4]-r[5]) for r in rows if r[0] not in (bg, ref)])
    print(f"# soft-mode min-freq MAE (held-out smearings) = {err:.3f} THz")
    print(f"# verdict: if the 2-param Friedel tracks min-freq(T_el), the (E)-channel")
    print(f"#   damped-Friedel law GENERALIZES from graphene to {a.tag}.")


if __name__ == "__main__":
    main()
