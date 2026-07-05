"""M2 dense-T q*(T) map + anharmonicity diagnostic, in one MD-sampling pass.

Per temperature: Langevin MD (MLIP forces) once, then
  (M2)         fit effective harmonic fc2(T) -> dispersion on M-Gamma-K-M ->
               omega_Gamma/omega_K, Kohn-kink, i.e. the dense-T q*(T) map;
  (diagnostic) fit fc2+fc3 on the SAME snapshots -> how much of the thermal force
               is anharmonic (rmse of fc2-only fit), how much of THAT the cubic
               term captures, and the fc3 norm -- a quantitative target for the
               Path-P DFT distillation. (L)-channel only; no electronic channel.

Reuses the validated helpers in td_phonon.py; the fc3 diagnostic is guarded so a
fit hiccup never loses the M2 dispersion. All on the RTX 2060.

    conda run --no-capture-output -n phonon python scripts/td_anharmonic.py \
        --structure data/td_phonon/graphene.xyz \
        --model results/finetune_mace/ft_phonon.model \
        --temperatures 10,50,100,200,300,400,500,600 --tag graphene_ft
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from ase.io import read

import anomaly_locate as al
import td_common as tdc
import td_phonon as tdp

CM = 33.35641


def fit_fc3(prim, ideal, snaps, cutoff2, cutoff3):
    """fc2+fc3 hiPhive fit on the same snapshots; return (fc3_array, rmse, ndof)."""
    from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
    from hiphive.utilities import prepare_structures
    from trainstation import Optimizer

    cs = ClusterSpace(prim, [cutoff2, cutoff3])
    structures = prepare_structures(snaps, ideal)
    sc = StructureContainer(cs)
    for s in structures:
        sc.add_structure(s)
    opt = Optimizer(sc.get_fit_data(), train_size=1.0)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)
    fc3 = fcp.get_force_constants(ideal).get_fc_array(order=3)
    rmse = getattr(opt, "rmse_train", None)
    return fc3, float(rmse) if rmse is not None else float("nan"), len(opt.parameters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperatures", default="10,50,100,200,300,400,500,600")
    ap.add_argument("--supercell", default="6,6,1")
    ap.add_argument("--cutoff2", type=float, default=6.0)
    ap.add_argument("--cutoff3", type=float, default=3.5)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--equil", type=int, default=1500)
    ap.add_argument("--nsnap", type=int, default=120)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--npoints", type=int, default=201)
    ap.add_argument("--no-fc3", action="store_true")
    ap.add_argument("--outdir", default="results/td_phonon")
    a = ap.parse_args()

    def log(m):
        print(m, flush=True)

    sc = tuple(int(x) for x in a.supercell.split(","))
    sc_matrix = np.diag(sc)
    temps = [float(x) for x in a.temperatures.split(",")]

    at0 = read(ROOT / a.structure)
    log(f"[{a.tag}] {a.structure} ({len(at0)} atoms); model={a.model} on {a.device}")
    calc = tdc.get_mace_calc(a.model, device=a.device)

    prim0, info = tdc.relax_monolayer(at0, calc)
    name = Path(a.structure).stem
    try:
        prim = tdc.build_monolayer(name, vacuum=7.5, a=info["a"], thickness=info["thickness"])
    except (ValueError, KeyError):
        # family generalization: unknown monolayer (any TMD) -> use the relaxed primitive directly
        prim = prim0
    prim.wrap()
    from phonon_accel.phonons import ase_to_phonopy, phonopy_to_ase
    from phonopy import Phonopy
    ph0 = Phonopy(ase_to_phonopy(prim), supercell_matrix=sc_matrix, primitive_matrix=np.eye(3))
    ideal = phonopy_to_ase(ph0.supercell)
    ideal.wrap()
    log(f"[{a.tag}] relaxed a={info['a']:.4f} A -> clean {name} primitive; "
        f"MD cell {sc}={len(ideal)} atoms; cutoff2={a.cutoff2} cutoff3={a.cutoff3} A")

    out = {"temperatures": np.array(temps), "supercell": np.asarray(sc),
           "model": np.array(str(a.model)), "tag": np.array(str(a.tag))}
    rows = ["T_K,w_gamma_cm,w_k_cm,kink_gamma,kink_k,minfreq_thz,"
            "force_rms_meVA,rmse_fc2_meVA,rmse_fc23_meVA,anharm_frac_cubic,"
            "anharm_frac_force,fc3_norm"]

    for T in temps:
        t0 = time.perf_counter()
        log(f"[{a.tag}] T={T:.0f} K: MD ...")
        snaps = tdp.sample_md(ideal, calc, T, a.dt, a.equil, a.nsnap, a.stride,
                              seed=int(T), log=log)
        F = np.concatenate([s.get_forces().ravel() for s in snaps])
        frms = float(np.sqrt(np.mean(F ** 2)))

        try:
            fc2, rmse2, _ = tdp.effective_fc2(prim, ideal, sc_matrix, snaps, a.cutoff2)
            dist, freq, lp, labs = tdp.band_from_phonopy(ph0, fc2, npoints=a.npoints)
        except Exception as e:
            log(f"[{a.tag}] T={T:.0f} K: fc2 fit failed ({type(e).__name__}: {e}); skipping T")
            continue
        minf = float(freq.min())  # THz; the key TMD (soft-mode) observable
        # graphene-specific readout (w_gamma/w_k, Kohn kink) — NaN for non-hexagonal/TMD paths
        try:
            wG = al.branch_freq_at_label(dist, freq, lp, labs, r"$\Gamma$") * CM
            wK = al.branch_freq_at_label(dist, freq, lp, labs, "K") * CM
            kinks = {k["label"]: k for k in al.high_sym_kinks(dist, freq, lp, labs)}
            kG, kK = kinks[r"$\Gamma$"]["kink_strength"], kinks["K"]["kink_strength"]
        except (KeyError, ValueError, IndexError):
            wG = wK = kG = kK = float("nan")

        rmse23, fc3n, frac_cubic = float("nan"), float("nan"), float("nan")
        if not a.no_fc3:
            try:
                fc3, rmse23, ndof3 = fit_fc3(prim, ideal, snaps, a.cutoff2, a.cutoff3)
                fc3n = float(np.linalg.norm(fc3))
                frac_cubic = (rmse2 - rmse23) / rmse2 if rmse2 > 0 else float("nan")
            except Exception as e:
                log(f"[{a.tag}] T={T:.0f} K: fc3 diagnostic failed: {type(e).__name__}: {e}")

        frac_force = rmse2 / frms if frms > 0 else float("nan")
        key = f"T{int(T)}"
        out[f"{key}_dist"], out[f"{key}_freq"] = dist, freq
        out["label_positions"], out["labels"] = lp, labs
        rows.append(f"{T:.0f},{wG:.2f},{wK:.2f},{kG:.3f},{kK:.3f},{freq.min():.3f},"
                    f"{frms*1000:.2f},{rmse2*1000:.2f},{rmse23*1000:.2f},"
                    f"{frac_cubic:.4f},{frac_force:.4f},{fc3n:.4f}")
        log(f"[{a.tag}] T={T:.0f}K  wG={wG:.1f} wK={wK:.1f}  kinkK={kK:.1f}  "
            f"|F|={frms*1000:.0f}  rmse2={rmse2*1000:.0f} rmse23={rmse23*1000:.0f} meV/A  "
            f"cubic-frac={frac_cubic:.2f}  ||fc3||={fc3n:.2f}  ({time.perf_counter()-t0:.0f}s)")

    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    npz = outdir / f"td_{a.tag}.npz"
    np.savez(npz, **out)
    csv = outdir / f"td_{a.tag}.csv"  # superset of plot_td_dispersion's columns
    csv.write_text("\n".join(rows) + "\n")
    log(f"[{a.tag}] wrote {npz.relative_to(ROOT)} + {csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
