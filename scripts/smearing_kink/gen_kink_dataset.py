"""Assemble the kink(T_el, T_lat) dataset for the `smearing-kink-ml` sub-line.

Every row is one (temperature) point with the Kohn-anomaly kink at Γ and K,
computed with ONE estimator (anomaly_locate.high_sym_kinks, span=8) so the (E)
and (L) channels live on a comparable scale. Columns:

    T_el_K, T_lat_K, channel, model, kink_gamma, kink_k, w_gamma_cm, w_k_cm, source

Seed (runs now, no new DFT):
  * (E) electronic-smearing channel — graphene DFT frozen-phonon at 4 degauss
    (results/vq2/disp_graphene_dg*.npz, 5×5), T_lat=0 (frozen geometry);
  * (L) lattice-temperature channel — graphene MLIP TDEP (results/td_phonon/
    td_graphene_ft_m2.csv), T_el marked NaN (the model's implicit smearing).

To GROW the (E) set (roadmap S1): run the DFT degauss sweep, e.g.
    python scripts/m1_1b_graphene_dft.py --pw <pw.x> --degauss 0.08 --smearing fermi-dirac \\
        --supercell 6 --kpts 6 --tag graphene_dg0.08     # 6×6 K-commensurate
then add its disp_*.npz to E_NPZ below (T_el = degauss × 157887).

    conda run --no-capture-output -n phonon python scripts/smearing_kink/gen_kink_dataset.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from anomaly_locate import high_sym_kinks  # noqa: E402

CM = 33.35641
DEG2TEL = 157887.0

# (E) channel: DFT frozen-phonon dispersions at each degauss (5×5). Extend here.
E_NPZ = {
    0.005: "results/vq2/disp_graphene_dg0.005.npz",
    0.010: "results/vq2/disp_graphene_dg0.01.npz",
    0.020: "results/vq2/disp_graphene_dg0.02.npz",
    0.040: "results/vq2/disp_graphene_dg0.04.npz",
}
# (L) channel: MLIP-TDEP kink vs lattice T (already reduced to kinks in this CSV).
L_CSV = "results/td_phonon/td_graphene_ft_m2.csv"


def kinks_from_npz(path):
    """Return (kink_Γ, kink_K, ω_Γ_cm, ω_K_cm) via the shared estimator."""
    d = np.load(ROOT / path, allow_pickle=True)
    ks = {k["label"]: k for k in high_sym_kinks(
        d["distances"], d["frequencies"], d["label_positions"], d["labels"])}
    g = next(v for k, v in ks.items() if "G" in k or "Γ" in k or "gamma" in k.lower())
    kk = next(v for k, v in ks.items() if k.strip().endswith("K") or k.strip() == "$K$")
    return g["kink_strength"], kk["kink_strength"], g["freq_thz"] * CM, kk["freq_thz"] * CM


def main() -> int:
    rows = []
    # ---- (E) electronic-smearing rows (T_lat = 0, frozen geometry) ----
    for dg, path in sorted(E_NPZ.items()):
        if not (ROOT / path).exists():
            print(f"[skip] {path} missing"); continue
        kg, kk, wg, wk = kinks_from_npz(path)
        rows.append(dict(T_el_K=round(dg * DEG2TEL), T_lat_K=0, channel="E",
                         model="DFT-5x5-frozen", kink_gamma=round(kg, 3), kink_k=round(kk, 3),
                         w_gamma_cm=round(wg, 1), w_k_cm=round(wk, 1), source=path))
    # ---- (L) lattice-temperature rows (T_el = NaN, MLIP-TDEP scale) ----
    lcsv = ROOT / L_CSV
    if lcsv.exists():
        for r in csv.DictReader(open(lcsv)):
            rows.append(dict(T_el_K="nan", T_lat_K=int(float(r["T_K"])), channel="L",
                             model="MLIP-TDEP-ft", kink_gamma=round(float(r["kink_gamma"]), 3),
                             kink_k=round(float(r["kink_k"]), 3),
                             w_gamma_cm=round(float(r["w_gamma_cm"]), 1),
                             w_k_cm=round(float(r["w_k_cm"]), 1), source=L_CSV))
    else:
        print(f"[skip] {L_CSV} missing")

    cols = ["T_el_K", "T_lat_K", "channel", "model", "kink_gamma", "kink_k",
            "w_gamma_cm", "w_k_cm", "source"]
    out = ROOT / "results/smearing_kink/kink_dataset_graphene.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    ne = sum(r["channel"] == "E" for r in rows)
    nl = sum(r["channel"] == "L" for r in rows)
    print(f"[gen] wrote {len(rows)} rows ({ne} E + {nl} L) -> {out.relative_to(ROOT)}")
    print(f"\n{'chan':<4} {'T_el':>6} {'T_lat':>6} {'kinkΓ':>7} {'kinkK':>7}  model")
    for r in rows:
        print(f"{r['channel']:<4} {str(r['T_el_K']):>6} {str(r['T_lat_K']):>6} "
              f"{r['kink_gamma']:>7} {r['kink_k']:>7}  {r['model']}")
    print("\nNOTE: E-rows (DFT 5×5) and L-rows (MLIP-TDEP) are on different model "
          "scales — see EXECPLAN §5. Recompute both from one distilled model before "
          "fitting a single kink(T_el,T_lat) surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
