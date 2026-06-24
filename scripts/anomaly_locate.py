"""Kohn-anomaly / dispersion-kink locator.

A Kohn anomaly is a sharp, near-non-analytic cusp in omega(q) at q* ~ 2k_F:
locally a discontinuity in the group velocity v = domega/dq and a spike in the
curvature |d2omega/dq2|. In graphene it lives on the *highest optical branch*
(E2g at Gamma, A1' at K). This module reports

  1. high_sym_kinks: the two-sided slope discontinuity |dv| of the top branch at
     each interior high-symmetry point -- the clean, artifact-free Kohn metric;
  2. locate_anomalies: a generic curvature-cusp scan restricted to the top
     optical branches (so it does not fire on acoustic band crossings).

Frequency-sorted branches swap identity at crossings, which fakes a cusp; the
top branch is the upper envelope and crosses nothing above it, so it is the safe
one to differentiate. Path endpoints are skipped (a slope change there is
geometric, not anomalous).

CLI:  python scripts/anomaly_locate.py results/td_phonon/disp_graphene_ft.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CM = 33.35641  # THz -> cm^-1


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Odd-window moving average with edge replication (no scipy dependency)."""
    if window < 3:
        return y
    window = window + 1 if window % 2 == 0 else window
    pad = window // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    ker = np.ones(window) / window
    return np.convolve(ypad, ker, mode="valid")


def _slopes_about(dist: np.ndarray, y: np.ndarray, i: int, span: int = 8):
    """Least-squares slope just left and just right of index ``i``."""
    lo, hi = max(0, i - span), min(len(dist) - 1, i + span)
    left = right = np.nan
    if i - lo >= 2:
        left = np.polyfit(dist[lo : i + 1], y[lo : i + 1], 1)[0]
    if hi - i >= 2:
        right = np.polyfit(dist[i : hi + 1], y[i : hi + 1], 1)[0]
    return left, right


def nearest_label(d: float, label_pos, labels):
    j = int(np.argmin(np.abs(np.asarray(label_pos) - d)))
    return str(labels[j]), float(label_pos[j])


def branch_freq_at_label(dist, freq, label_pos, labels, label_name, branch="top"):
    """omega of a branch at a named high-symmetry point. branch='top' = highest."""
    j = [str(x) for x in labels].index(label_name)
    i = int(np.argmin(np.abs(np.asarray(dist) - label_pos[j])))
    col = freq.shape[1] - 1 if branch == "top" else int(branch)
    return float(freq[i, col])


def high_sym_kinks(dist, freq, label_pos, labels, branch="top", span=8):
    """Two-sided slope discontinuity of one branch at each high-symmetry point.

    Returns a list of dicts (one per label). ``interior`` flags points that are
    not path endpoints -- only those carry a meaningful (geometric-free) kink.
    ``kink_strength`` = |slope_right - slope_left| in THz / (q-distance unit).
    """
    dist = np.asarray(dist, float)
    freq = np.asarray(freq, float)
    col = freq.shape[1] - 1 if branch == "top" else int(branch)
    y = freq[:, col]
    names = [str(x) for x in labels]
    lp = np.asarray(label_pos, float)
    out = []
    for j, (d0, nm) in enumerate(zip(lp, names)):
        i = int(np.argmin(np.abs(dist - d0)))
        left, right = _slopes_about(dist, y, i, span)
        kink = abs(right - left) if np.isfinite(left + right) else np.nan
        out.append({
            "label": nm,
            "distance": float(d0),
            "branch": col,
            "freq_thz": float(y[i]),
            "slope_left": float(left),
            "slope_right": float(right),
            "kink_strength": float(kink),
            "interior": 0 < j < len(lp) - 1,
        })
    return out


def locate_anomalies(
    dist: np.ndarray,
    freq: np.ndarray,
    label_pos,
    labels,
    qfrac=None,
    n_top: int = 2,
    nsig: float = 3.0,
    smooth_window: int = 5,
    edge_frac: float = 0.03,
):
    """Curvature cusps on the top ``n_top`` (optical) branches.

    Returns a list of dicts sorted by descending kink strength. Restricting to
    the highest branches avoids the acoustic-crossing false positives; the
    duplicate-join fix in ``td_common.dispersion`` removes the at-label spikes.
    """
    dist = np.asarray(dist, float)
    freq = np.asarray(freq, float)
    n_q, n_band = freq.shape
    span_d = (dist[-1] - dist[0]) * edge_frac
    out = []

    for b in range(max(0, n_band - n_top), n_band):
        y = _smooth(freq[:, b], smooth_window)
        v = np.gradient(y, dist)
        a = np.gradient(v, dist)
        absa = np.abs(a)
        thr = absa.mean() + nsig * absa.std()
        for i in range(2, n_q - 2):
            if absa[i] < thr:
                continue
            if not (absa[i] >= absa[i - 1] and absa[i] >= absa[i + 1]):
                continue
            if dist[i] - dist[0] < span_d or dist[-1] - dist[i] < span_d:
                continue  # skip endpoints (geometric)
            left, right = _slopes_about(dist, freq[:, b], i)
            kink = abs(right - left) if np.isfinite(left + right) else float(absa[i])
            lab, lab_d = nearest_label(dist[i], label_pos, labels)
            rec = {
                "distance": float(dist[i]),
                "branch": int(b),
                "freq_thz": float(freq[i, b]),
                "curvature": float(absa[i]),
                "kink_strength": float(kink),
                "nearest_label": lab,
                "at_label": bool(abs(dist[i] - lab_d) < span_d),
            }
            if qfrac is not None:
                rec["q_frac"] = [float(x) for x in np.asarray(qfrac)[i]]
            out.append(rec)

    out.sort(key=lambda r: -r["kink_strength"])
    dedup, seen = [], []
    for r in out:
        if any(r["branch"] == b and abs(r["distance"] - d) < span_d for b, d in seen):
            continue
        seen.append((r["branch"], r["distance"]))
        dedup.append(r)
    return dedup


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    d = dict(np.load(argv[0], allow_pickle=True))
    dist, freq = d["distances"], d["frequencies"]
    label_pos, labels = d["label_positions"], d["labels"]

    print(f"# {Path(argv[0]).name}: {freq.shape[1]} branches, {freq.shape[0]} q-points")
    print("# top-branch frequency + two-sided kink at each high-symmetry point:")
    for k in high_sym_kinks(dist, freq, label_pos, labels):
        tag = "interior" if k["interior"] else "endpoint"
        ks = "  n/a" if not np.isfinite(k["kink_strength"]) else f"{k['kink_strength']:7.1f}"
        print(f"  {k['label']:>9s} ({tag:8s})  omega={k['freq_thz']*CM:7.1f} cm^-1  "
              f"kink|dv|={ks}")
    print("# generic cusp scan (top 2 optical branches):")
    for r in locate_anomalies(dist, freq, label_pos, labels,
                              qfrac=d.get("qpoints_frac"))[:6]:
        print(f"  near {r['nearest_label']:>9s}  branch {r['branch']:2d}  "
              f"omega={r['freq_thz']*CM:7.1f} cm^-1  kink|dv|={r['kink_strength']:7.1f}  "
              f"at_label={r['at_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
