"""Kohn-anomaly / dispersion-kink locator.

A Kohn anomaly is a sharp, near-non-analytic cusp in omega(q) at q* ~ 2k_F:
locally a discontinuity in the group velocity v = domega/dq and a spike in the
curvature |d2omega/dq2|. This module turns a cached dispersion (from
``td_common.dispersion``) into a list of such cusps, with a quantitative
"kink strength" = |slope_right - slope_left| at each one.

The detector deliberately ignores the path *endpoints* (Gamma at the ends) where
a slope change is geometric, not anomalous, and reports the nearest labelled
high-symmetry point so graphene's Gamma-E2g and K-A1' anomalies are easy to read.

CLI:  python scripts/anomaly_locate.py results/td_phonon/disp_graphene_ft.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Odd-window moving average with edge replication (no scipy dependency)."""
    if window < 3:
        return y
    window = window + 1 if window % 2 == 0 else window
    pad = window // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    ker = np.ones(window) / window
    return np.convolve(ypad, ker, mode="valid")


def _slopes_about(dist: np.ndarray, y: np.ndarray, i: int, span: int = 6):
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


def locate_anomalies(
    dist: np.ndarray,
    freq: np.ndarray,
    label_pos,
    labels,
    qfrac=None,
    nsig: float = 4.0,
    smooth_window: int = 5,
    min_freq_thz: float = 5.0,
    edge_frac: float = 0.02,
):
    """Find curvature cusps per branch.

    Returns a list of dicts sorted by descending kink strength. ``nsig`` is the
    per-branch curvature threshold in units of (mean + nsig*std). Branches and
    points below ``min_freq_thz`` (acoustic near Gamma, flexural noise) are
    skipped so the optical Kohn anomalies dominate.
    """
    dist = np.asarray(dist, float)
    freq = np.asarray(freq, float)
    n_q, n_band = freq.shape
    span_d = (dist[-1] - dist[0]) * edge_frac
    out = []

    for b in range(n_band):
        y = _smooth(freq[:, b], smooth_window)
        v = np.gradient(y, dist)                 # group velocity
        a = np.gradient(v, dist)                 # curvature
        absa = np.abs(a)
        thr = absa.mean() + nsig * absa.std()
        # interior local maxima of |curvature| above threshold
        for i in range(2, n_q - 2):
            if absa[i] < thr:
                continue
            if not (absa[i] >= absa[i - 1] and absa[i] >= absa[i + 1]):
                continue
            if freq[i, b] < min_freq_thz:
                continue
            if dist[i] - dist[0] < span_d or dist[-1] - dist[i] < span_d:
                continue  # skip path endpoints (geometric, not anomalous)
            left, right = _slopes_about(dist, freq[:, b], i)
            kink = abs((right - left)) if np.isfinite(left + right) else float(absa[i])
            lab, lab_d = nearest_label(dist[i], label_pos, labels)
            rec = {
                "distance": float(dist[i]),
                "branch": int(b),
                "freq_thz": float(freq[i, b]),
                "curvature": float(absa[i]),
                "kink_strength": float(kink),
                "slope_left": float(left),
                "slope_right": float(right),
                "nearest_label": lab,
                "label_distance": lab_d,
                "at_label": bool(abs(dist[i] - lab_d) < span_d),
            }
            if qfrac is not None:
                rec["q_frac"] = [float(x) for x in np.asarray(qfrac)[i]]
            out.append(rec)

    # collapse near-duplicate detections (same label + branch, adjacent points)
    out.sort(key=lambda r: -r["kink_strength"])
    dedup, seen = [], set()
    for r in out:
        key = (r["branch"], round(r["distance"], 3))
        if any(r["branch"] == s[0] and abs(r["distance"] - s[1]) < span_d for s in seen):
            continue
        seen.add((r["branch"], r["distance"]))
        dedup.append(r)
    return dedup


def branch_freq_at_label(dist, freq, label_pos, labels, label_name, branch="top"):
    """omega of a branch at a named high-symmetry point. branch='top' = highest."""
    j = [str(x) for x in labels].index(label_name)
    i = int(np.argmin(np.abs(np.asarray(dist) - label_pos[j])))
    col = freq.shape[1] - 1 if branch == "top" else int(branch)
    return float(freq[i, col])


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    d = dict(np.load(argv[0], allow_pickle=True))
    dist, freq = d["distances"], d["frequencies"]
    label_pos, labels = d["label_positions"], d["labels"]
    qfrac = d.get("qpoints_frac")
    anomalies = locate_anomalies(dist, freq, label_pos, labels, qfrac=qfrac)

    print(f"# {Path(argv[0]).name}: {freq.shape[1]} branches, {freq.shape[0]} q-points")
    cm = 33.35641  # THz -> cm^-1
    for lab in ("$\\Gamma$", "K", "M"):
        if lab in [str(x) for x in labels]:
            w = branch_freq_at_label(dist, freq, label_pos, labels, lab)
            print(f"  top optical at {lab:9s}: {w:7.3f} THz = {w*cm:7.1f} cm^-1")
    print(f"# detected {len(anomalies)} cusp(s):")
    for r in anomalies[:8]:
        print(
            f"  {r['nearest_label']:>8s}  branch {r['branch']:2d}  "
            f"omega={r['freq_thz']:6.2f} THz  kink|dv|={r['kink_strength']:7.2f}  "
            f"curv={r['curvature']:8.1f}  at_label={r['at_label']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
