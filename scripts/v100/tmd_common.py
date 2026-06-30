"""Shared helpers for the V100 TMD FP64 campaign: a general 2H/1T monolayer
builder, the config/material loader, and an ONCV-pseudo resolver (glob so the
SG15 version suffix doesn't matter).

Reused by tmd_dft_fc2.py / tmd_dft_bands.py / tmd_path_p.py. The EPW bash
pipeline (tmd_epw_full.sh) reads the same material fields via env.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_config(path="configs/v100_campaign.yaml") -> dict:
    import yaml
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    return yaml.safe_load(p.read_text())


def material(cfg: dict, name: str) -> dict:
    for m in cfg["materials"]:
        if m["name"] == name:
            return m
    raise KeyError(f"material {name!r} not in config ({[m['name'] for m in cfg['materials']]})")


def build_tmd(formula: str, polytype: str, a: float, thickness: float,
              vacuum: float = 7.5):
    """General 2H/1T TMD monolayer (ase.build.mx2)."""
    from ase.build import mx2
    at = mx2(formula=formula, kind=polytype, a=a, thickness=max(thickness, 0.5),
             vacuum=vacuum)
    at.pbc = True
    return at


def pseudo_for(element: str, pseudo_dir) -> str:
    """Resolve {element}_ONCV_PBE*.upf in pseudo_dir -> basename. Glob avoids
    hardcoding the SG15 version suffix (varies per element)."""
    pd = Path(pseudo_dir)
    hits = sorted(glob.glob(str(pd / f"{element}_ONCV_PBE*.upf")))
    if not hits:
        # case-insensitive / looser fallback
        hits = sorted(glob.glob(str(pd / f"{element}_ONCV*.upf"))
                      + glob.glob(str(pd / f"{element}.*upf")))
    if not hits:
        raise FileNotFoundError(
            f"no ONCV pseudo for {element} in {pd} (run scripts/v100/fetch_pseudos.sh)")
    return Path(hits[0]).name


def pseudo_map(mat: dict, pseudo_dir) -> dict:
    """{element: upf basename} for a material's M and X."""
    return {mat["M"]: pseudo_for(mat["M"], pseudo_dir),
            mat["X"]: pseudo_for(mat["X"], pseudo_dir)}


# --------------------------------------------------------------------------- #
# tiny CLI so the bash lanes can read the YAML without re-parsing it
#   python tmd_common.py boxgpu A   -> "<name> <cdw 0|1>" per GPU-lane material
#   python tmd_common.py boxepw A   -> EPW-lane material names (one per line)
#   python tmd_common.py epwenv NbS2 -> "MAT=.. A_ANG=.. .." for tmd_epw_full.sh
# --------------------------------------------------------------------------- #
def _main(argv):
    cfg = load_config()
    sub = argv[1] if len(argv) > 1 else ""
    if sub == "boxgpu":
        for n in cfg["boxes"][argv[2]]["gpu"]:
            print(f"{n} {1 if material(cfg, n).get('cdw') else 0}")
    elif sub == "boxepw":
        for n in cfg["boxes"][argv[2]]["epw"]:
            print(n)
    elif sub == "epwenv":
        m = material(cfg, argv[2])
        e = cfg["epw"]
        kv = {
            "MAT": m["name"], "A_ANG": m["a"], "COA": 10.0,
            "THICK_ANG": m["thickness"], "M": m["M"], "X": m["X"],
            "MMASS": m["Mmass"], "XMASS": m["Xmass"], "POLY": m["polytype"],
            "DEGAUSS": e["degauss"], "NQ": e["nq"], "NK": e["nk"], "NKF": e["nkf"],
            "NBNDSUB": e["nbndsub"], "DISWINMAX": e["dis_win_max"],
        }
        print(" ".join(f"{k}={v}" for k, v in kv.items()))
    else:
        raise SystemExit("usage: tmd_common.py boxgpu|boxepw|epwenv ...")


if __name__ == "__main__":
    import sys
    _main(sys.argv)
