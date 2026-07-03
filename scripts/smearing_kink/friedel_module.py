"""Analytic damped-Friedel long-range fc2 module (BAMBOO scaffold, metallic variant).

Physical model for the smearing (electronic-temperature) dependence of the
harmonic force constants of a metal / semimetal:

    fc2(R; T_el)  =  fc2_backbone(R)                       [smearing-blind, short-range]
                  +  Sum_pairs  g(R; A(T_el), xi(T_el))  n^ n^   [long-range Friedel term]

    g(R) = A * cos(2 k_F R + phi) * exp(-R / xi) / R^p

The backbone is what an ordinary (single-smearing) MLIP learns; the analytic
long-range term carries ALL the T_el dependence with just two smooth, physical
parameters -- the amplitude A(T_el) and the thermal damping length
xi(T_el) ~ v_F / (k_B T_el) of a Friedel/RKKY oscillation. (k_F, phi, p are
smearing-independent globals fixed by the Fermi surface.)  This is the metallic
analogue of BAMBOO's analytic electrostatic long-range module (arXiv:2404.07181):
short-range NN backbone + explicit analytic long-range physics + distillation.

The correction is applied as a longitudinal (bond-stretch) force constant
g(R) n^ n^ with the acoustic-sum-rule self term -- a genuine, ASR-preserving
force-constant model, so it plugs straight into a phonopy dispersion (and, as an
additive pair energy, into an MLIP calculator).

This module = geometry + kernel + fc assembly + kink readout.  The fit/validation
driver is fit_friedel.py.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import phonopy
from td_phonon import band_from_phonopy
from anomaly_locate import high_sym_kinks

CM = 33.35641  # THz -> cm^-1


# ---------------------------------------------------------------- data loading
def load_ph(yaml_path):
    """phonopy object with force_constants read from the yaml (no re-derivation)."""
    return phonopy.load(str(yaml_path), produce_fc=False)


def pair_table(ph):
    """Minimum-image pair geometry for each compact-fc row (primitive atom).

    Returns a list (one per primitive row p) of dicts with parallel arrays over
    supercell atoms s: s index, R (A), unit vector n (3,), and R vector.
    Row p corresponds to supercell atom p2s[p] (the compact-fc convention).
    """
    sc = ph.supercell
    pos = np.array(sc.positions)
    cell = np.array(sc.cell)
    inv = np.linalg.inv(cell)
    p2s = list(np.array(ph.primitive.p2s_map))
    tabs = []
    for p, s0 in enumerate(p2s):
        r0 = pos[s0]
        S, R, N, V = [], [], [], []
        for s in range(len(sc)):
            d = pos[s] - r0
            frac = d @ inv
            frac -= np.round(frac)
            dm = frac @ cell
            r = float(np.linalg.norm(dm))
            S.append(s); R.append(r)
            N.append(dm / r if r > 1e-6 else np.zeros(3)); V.append(dm)
        tabs.append({"s": np.array(S), "R": np.array(R),
                     "n": np.array(N), "vec": np.array(V), "s0": s0})
    return tabs, p2s


# ---------------------------------------------------------------- Friedel kernel
def kernel_unit(R, twokF, phi, p, xi):
    """g/A : cos(2kF R + phi) exp(-R/xi) / R^p  (0 where R==0)."""
    R = np.asarray(R, float)
    out = np.zeros_like(R)
    m = R > 1e-6
    out[m] = np.cos(twokF * R[m] + phi) * np.exp(-R[m] / xi) / R[m] ** p
    return out


def longitudinal(fc_block, n):
    """n . Phi . n for a stack of 3x3 blocks and matching unit vectors."""
    return np.einsum("i,kij,j->k", n[0], fc_block, n[0]) if fc_block.ndim == 3 and n.ndim == 1 \
        else np.einsum("ki,kij,kj->k", n, fc_block, n)


def delta_long(fc, fc_bg, tabs, rmin, rmax):
    """Longitudinal fc2 difference (fc - backbone) vs pair, per row, tail mask.

    Returns per row: (R[mask], dL[mask], mask indices) for pairs in [rmin,rmax].
    """
    rows = []
    for p, t in enumerate(tabs):
        n, s, R = t["n"], t["s"], t["R"]
        dblock = fc[p, s] - fc_bg[p, s]           # (nsuper,3,3)
        dL = np.einsum("ki,kij,kj->k", n, dblock, n)
        m = (R >= rmin) & (R <= rmax)
        rows.append((R[m], dL[m], np.where(m)[0]))
    return rows


# ---------------------------------------------------------------- fc assembly
def add_friedel(fc_bg, tabs, p2s, A, xi, twokF, phi, p, rmin, rmax):
    """fc_model = fc_bg + Sum g(R) n^ n^  (longitudinal), ASR self term restored.

    A, xi are scalars for this smearing; globals twokF/phi/p shared. Returns a
    new compact fc array (copy of fc_bg + correction).
    """
    fc = fc_bg.copy()
    for prow, t in enumerate(tabs):
        n, s, R = t["n"], t["s"], t["R"]
        g = A * kernel_unit(R, twokF, phi, p, xi)
        m = (R >= rmin) & (R <= rmax)
        # cross blocks: g(R) * n outer n
        corr = np.einsum("k,ki,kj->kij", g * m, n, n)   # (nsuper,3,3)
        fc[prow, s] += corr
        # ASR: self block (s == s0 for this row) gets -sum of cross corrections
        self_idx = np.where(s == t["s0"])[0][0]
        fc[prow, self_idx] -= corr.sum(axis=0)
    return fc


def kink_of(ph, fc, path="MGKM"):
    """(kink_K, wK_cm, kink_G) from a compact fc via band + high_sym_kinks."""
    dist, freq, lp, labels = band_from_phonopy(ph, fc, path=path, npoints=201)
    d = {k["label"]: k for k in high_sym_kinks(dist, freq, lp, labels)}
    K, G = d["K"], d["$\\Gamma$"]
    return K["kink_strength"], K["freq_thz"] * CM, G["kink_strength"]


# ------------------------------------------- full-tensor template-envelope model
# The Kohn cusp lives in the *off-diagonal / transverse* fc2 components, not just
# the bond-longitudinal one (diagnostic T2). So the long-range term must carry the
# full 3x3 oscillation waveform.  We take that waveform from the Fermi surface:
# the reference (sharpest-smearing) fc2 difference D0 = fc(ref) - fc(backbone) is
# the least-damped Friedel oscillation, with correct directionality + tensor
# structure.  Its thermal evolution is then a 2-parameter envelope
#
#     Delta(R; T_el) = B(T_el) * exp(-kappa(T_el) * R) * D0(R)
#
# B = amplitude, kappa = 1/xi(T_el) - 1/xi_ref = EXTRA thermal damping rate.
# At the reference smearing B=1, kappa=0 (exact); higher T_el -> B down, kappa up
# (shorter Friedel range).  This is the metallic BAMBOO long-range module: an
# analytic Fermi-surface envelope on the explicit long-range oscillation.

def template_delta(fc_ref, fc_bg, tabs):
    """D0 per primitive row: full-tensor (nsuper,3,3) fc difference ref - backbone."""
    return [fc_ref[p, t["s"]] - fc_bg[p, t["s"]] for p, t in enumerate(tabs)]


def add_template(fc_bg, tabs, D0_rows, B, kappa, rmin, rmax):
    """fc = fc_bg + B exp(-kappa R) D0  on tail pairs; ASR self term restored."""
    fc = fc_bg.copy()
    for prow, t in enumerate(tabs):
        s, R = t["s"], t["R"]
        env = B * np.exp(-kappa * R) * ((R >= rmin) & (R <= rmax))
        corr = env[:, None, None] * D0_rows[prow]      # (nsuper,3,3)
        fc[prow, s] += corr
        si = np.where(s == t["s0"])[0][0]
        fc[prow, si] -= corr.sum(axis=0)
    return fc


def fit_template_env(fc_dg, fc_bg, tabs, D0_rows, rmin, rmax, kappas, wexp=0.0):
    """Best (B, kappa, rel_resid) so B exp(-kappa R) D0 ~ (fc_dg - fc_bg) on the
    tail, over the full 3x3 tensor.  kappa scanned; B projected (linear).

    wexp>0 weights each pair's tensor residual by R**wexp -- the Kohn cusp is
    long-range dominated (diagnostic T3), so up-weighting the tail aligns the
    fc-fit objective with the kink observable it must reproduce.
    """
    tgt, tmpl_R, wts = [], [], []
    for prow, t in enumerate(tabs):
        s, R = t["s"], t["R"]
        m = (R >= rmin) & (R <= rmax)
        tgt.append((fc_dg[prow, s][m] - fc_bg[prow, s][m]).reshape(-1))
        tmpl_R.append((D0_rows[prow][m], R[m]))
        w = np.repeat(R[m] ** wexp, 9)              # 9 tensor comps per pair
        wts.append(w)
    tgt = np.concatenate(tgt)
    w = np.concatenate(wts)
    denom = float((w * tgt) @ tgt) + 1e-12
    best = (0.0, kappas[0], 1.0)
    for kap in kappas:
        cols = [ (np.exp(-kap * Rm)[:, None, None] * D0m).reshape(-1)
                 for D0m, Rm in tmpl_R ]
        u = np.concatenate(cols)
        uu = float((w * u) @ u)
        if uu < 1e-12:
            continue
        B = float((w * u) @ tgt) / uu
        resid = float(np.sum(w * (tgt - B * u) ** 2)) / denom
        if resid < best[2]:
            best = (B, kap, resid)
    return best
