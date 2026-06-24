"""Anharmonicity diagnostic figure: how anharmonic the MLIP's PES is vs T, and
how much of it is cubic -- a quantitative target for the Path-P DFT distillation.

Left:  fraction of the thermal force that is anharmonic (rmse of the fc2-only fit
       / total |F|), and the cubic share of that anharmonicity
       ((rmse_fc2 - rmse_fc23)/rmse_fc2), both vs T.
Right: the residuals themselves -- total |F|, fc2-only residual, fc2+fc3 residual
       (meV/A) vs T.

    conda run -n phonon python scripts/plot_anharm_diag.py \
        --csv results/td_phonon/td_graphene_ft_m2.csv
"""
from __future__ import annotations

import argparse
import csv as csvmod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/td_phonon/td_graphene_ft_m2.csv")
    ap.add_argument("--out", default="results/figures/graphene_anharmonicity.png")
    a = ap.parse_args()

    rows = list(csvmod.DictReader(open(ROOT / a.csv)))
    T = [float(r["T_K"]) for r in rows]
    frac_force = [float(r["anharm_frac_force"]) for r in rows]
    frac_cubic = [float(r["anharm_frac_cubic"]) for r in rows]
    frms = [float(r["force_rms_meVA"]) for r in rows]
    rmse2 = [float(r["rmse_fc2_meVA"]) for r in rows]
    rmse23 = [float(r["rmse_fc23_meVA"]) for r in rows]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"mace": "#d1495b", "finetuned": "#2a9d4a", "dfpt": "#000000"}
    fig, (axf, axr) = plt.subplots(1, 2, figsize=(8.6, 3.7))

    axf.plot(T, [100 * x for x in frac_force], "o-", color=C["mace"],
             label="anharmonic share of |F|")
    axf.plot(T, [100 * x for x in frac_cubic], "s--", color=C["finetuned"],
             label="cubic share of anharmonicity")
    axf.set_xlabel("T (K)")
    axf.set_ylabel("percent (%)")
    axf.set_title("MLIP anharmonicity vs T")
    axf.set_ylim(0, 100)
    axf.legend(frameon=False, fontsize=8, loc="center right")

    axr.plot(T, frms, "o-", color=C["dfpt"], label="total |F|")
    axr.plot(T, rmse2, "s-", color=C["mace"], label="residual after fc$_2$")
    axr.plot(T, rmse23, "^-", color=C["finetuned"], label="residual after fc$_2$+fc$_3$")
    axr.set_xlabel("T (K)")
    axr.set_ylabel("force RMS (meV/Å)")
    axr.set_title("force decomposition vs T")
    axr.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))

    print(f"{'T(K)':>5s} {'|F|':>7s} {'res_fc2':>8s} {'res_fc23':>9s} "
          f"{'anh%':>6s} {'cubic%':>7s} {'||fc3||':>9s}")
    for r in rows:
        print(f"{float(r['T_K']):5.0f} {float(r['force_rms_meVA']):7.0f} "
              f"{float(r['rmse_fc2_meVA']):8.0f} {float(r['rmse_fc23_meVA']):9.0f} "
              f"{100*float(r['anharm_frac_force']):6.1f} "
              f"{100*float(r['anharm_frac_cubic']):7.1f} {float(r['fc3_norm']):9.1f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
