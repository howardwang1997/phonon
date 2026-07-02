"""Few-shot emulator for the Kohn-anomaly kink as a function of temperature —
the roadmap §9 · E11 baseline of the `smearing-kink-ml` sub-line.

kink(T) is a *smooth* function of electronic smearing T_el (and lattice T_lat):
the smearing regularizes the 2k_F singularity, so — unlike the cusp itself — the
kink-vs-T curve is safe for a smooth regressor. A Gaussian process is the right
tool for the "few-shot / small-data finetune" goal: fit on a handful of DFT/EPW
anchors, predict the whole curve (incl. the σ→0 limit), and report the
leave-k-out learning curve that quantifies "how few points are enough."

This is numpy-only (no sklearn) so it runs in the base `phonon` env. It ships
with the measured graphene kink_Γ(T_el) (weekly-report Fig 12) as a runnable
proof-of-concept; point `--data` at gen_kink_dataset.py's CSV for the real set.

    conda run --no-capture-output -n phonon python scripts/smearing_kink/kink_emulator.py
    conda run --no-capture-output -n phonon python scripts/smearing_kink/kink_emulator.py \
        --data results/smearing_kink/kink_dataset_graphene.csv --target kink_gamma
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Measured graphene Γ-E₂g kink vs electronic smearing (DFT frozen-phonon, 5×5;
# results/vq2, weekly-report Fig 12). T_el = degauss × 157887 K.
SEED_TEL = np.array([789.0, 1579.0, 3158.0, 6315.0])
SEED_KINK = np.array([8.2, 7.0, 6.7, 5.8])


# --------------------------------------------------------------------------- #
# Minimal Gaussian process (RBF kernel, homoscedastic noise)
# --------------------------------------------------------------------------- #
class GP:
    def __init__(self, length=None, sigma_f=None, sigma_n=0.05):
        self.length = length
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n

    def _k(self, A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return self.sigma_f ** 2 * np.exp(-0.5 * d2 / self.length ** 2)

    def fit(self, X, y):
        X = np.atleast_2d(X).astype(float)
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.xmu) / self.xsd
        self.ymu = y.mean()
        yc = y - self.ymu
        if self.length is None:                       # median-distance heuristic
            dif = Xs[:, None, :] - Xs[None, :, :]
            dd = np.sqrt(np.sum(dif ** 2, -1))
            self.length = np.median(dd[dd > 0]) if np.any(dd > 0) else 1.0
        if self.sigma_f is None:
            self.sigma_f = np.std(yc) + 1e-6
        self.Xs = Xs
        K = self._k(Xs, Xs) + self.sigma_n ** 2 * np.eye(len(Xs))
        self.alpha = np.linalg.solve(K, yc)
        return self

    def predict(self, X):
        Xs = (np.atleast_2d(X).astype(float) - self.xmu) / self.xsd
        return self._k(Xs, self.Xs) @ self.alpha + self.ymu


# --------------------------------------------------------------------------- #
def load_data(path, target):
    """CSV with columns T_el_K[, T_lat_K], <target>. Returns X (n, d), y, cols."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get(target) in (None, "", "nan"):
                continue
            rows.append(r)
    tel = np.array([float(r["T_el_K"]) for r in rows])
    has_lat = any(r.get("T_lat_K") not in (None, "", "nan") for r in rows)
    if has_lat:
        tlat = np.array([float(r.get("T_lat_K") or 0.0) for r in rows])
        X = np.column_stack([tel, tlat]); cols = ["T_el_K", "T_lat_K"]
    else:
        X = tel[:, None]; cols = ["T_el_K"]
    y = np.array([float(r[target]) for r in rows])
    return X, y, cols


def learning_curve(X, y):
    """Leave-k-out few-shot curve: train on k range-spanning points, predict rest."""
    n = len(y)
    order = np.argsort(X[:, 0])
    Xo, yo = X[order], y[order]
    print(f"\nfew-shot learning curve ({n} points total):")
    print(f"  {'#train':>6} {'train idx':<18} {'held-out MAE':>12}")
    for k in range(2, n):
        idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
        tr = np.zeros(n, bool); tr[idx] = True
        if tr.all():
            continue
        gp = GP().fit(Xo[tr], yo[tr])
        pred = gp.predict(Xo[~tr])
        mae = float(np.mean(np.abs(pred - yo[~tr])))
        print(f"  {k:>6} {str(list(idx)):<18} {mae:>12.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV from gen_kink_dataset.py (else use graphene seed)")
    ap.add_argument("--target", default="kink_gamma")
    ap.add_argument("--extrap-to", type=float, default=100.0,
                    help="predict the kink at this T_el (K) — the σ→0 sharp-anomaly limit probe")
    ap.add_argument("--out", default="results/smearing_kink/kink_emulator_pred.csv")
    a = ap.parse_args()

    if a.data:
        X, y, cols = load_data(a.data, a.target)
        src = a.data
    else:
        X, y, cols = SEED_TEL[:, None], SEED_KINK, ["T_el_K"]
        src = "graphene seed (Fig 12: kink_Γ vs T_el)"
    print(f"[kink-emulator] {len(y)} points from {src}; features={cols}; target={a.target}")

    gp = GP().fit(X, y)
    learning_curve(X, y)

    # dense prediction along T_el at T_lat=0 (the (E)-channel curve) + σ→0 probe
    tel_grid = np.linspace(min(a.extrap_to, X[:, 0].min()), X[:, 0].max(), 60)
    Xg = (np.column_stack([tel_grid, np.zeros_like(tel_grid)]) if len(cols) == 2
          else tel_grid[:, None])
    yg = gp.predict(Xg)
    lim = float(gp.predict([[a.extrap_to, 0.0]] if len(cols) == 2 else [[a.extrap_to]])[0])
    print(f"\nGP fit: length={gp.length:.3f} σ_f={gp.sigma_f:.3f} σ_n={gp.sigma_n}")
    print(f"extrapolated kink at T_el={a.extrap_to:.0f} K (toward σ→0): {lim:.2f}"
          f"   [graphene σ→0 truth ≈ 14.4 on 6×6; 5×5 undershoots — see plan §5]")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["T_el_K", f"{a.target}_pred"])
        for t, v in zip(tel_grid, yg):
            w.writerow([f"{t:.1f}", f"{v:.4f}"])
    print(f"[kink-emulator] wrote dense prediction -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
