"""TMD-family Path-P anharmonic DFT labels — generalizes
scripts/path_p_nbse2_make_data.py to any CDW member (needs that material's DFT
fc2 from tmd_dft_fc2.py, which provides the soft eigenvector = CDW distortion).

Reuses the proven crash-safe sampler+labeller (soft-mode double-well scan +
thermal rattle, each DFT single-pointed on GPU pw.x); only swaps in the
material's pseudo map and fc2 yaml. CDW members only — the soft eigenvector is
meaningless for a dynamically-stable monolayer.

    python scripts/v100/tmd_path_p.py --name NbS2 \
        --pw /root/gpupw.sh --pseudo-dir /root/phonon/pseudo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "v100"))

import tmd_common as tc                  # noqa: E402
import path_p_nbse2_make_data as pp      # noqa: E402  (reuse soft_eigen/build_configs/loop)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="configs/v100_campaign.yaml")
    ap.add_argument("--pw", required=True)
    ap.add_argument("--pseudo-dir", default="/root/phonon/pseudo")
    ap.add_argument("--yaml", default=None, help="DFT fc2 phonopy (default: results/v100/fc2/<name>_phonopy.yaml)")
    a = ap.parse_args()

    cfg = tc.load_config(a.config)
    mat = tc.material(cfg, a.name)
    if not mat.get("cdw"):
        print(f"[pathP:{a.name}] not a CDW member -> skip (no soft eigenvector)")
        return 0
    yaml = a.yaml or f"results/v100/fc2/{a.name}_phonopy.yaml"
    if not (ROOT / yaml).exists():
        print(f"[pathP:{a.name}] fc2 yaml missing ({yaml}) -> run tmd_dft_fc2 first")
        return 2
    outdir = f"data/v100/path_p/{a.name}"
    if (ROOT / outdir / "train.xyz").exists():
        print(f"[pathP:{a.name}] {outdir}/train.xyz exists -> skip")
        return 0

    d, pp.PSEUDOS = cfg["dft"], tc.pseudo_map(mat, a.pseudo_dir)
    pcfg = cfg["path_p"]
    # drive the reused script via its own argparse
    sys.argv = [
        "tmd_path_p", "--yaml", str(ROOT / yaml), "--pw", a.pw,
        "--pseudo-dir", a.pseudo_dir, "--ecutwfc", str(d["ecutwfc"]),
        "--ecutrho", str(d["ecutrho"]), "--kpts", str(d["fc2_kpts"]),
        "--degauss", str(d["degauss"]), "--n-scan", str(pcfg["n_scan"]),
        "--amax", str(pcfg["amax"]), "--temps", str(pcfg["temps"]),
        "--n-therm", str(pcfg["n_therm"]), "--n-well", str(pcfg["n_well"]),
        "--workdir", f"results/v100/path_p/{a.name}", "--outdir", outdir,
    ]
    print(f"[pathP:{a.name}] pseudos={pp.PSEUDOS} fc2={yaml}", flush=True)
    return pp.main()


if __name__ == "__main__":
    raise SystemExit(main())
