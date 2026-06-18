"""Genuine DFPT phonon reference data from the MDR / PhononDB (Togo, NIMS).

~10,034 inorganic materials computed with VASP + phonopy. Each material's
``phonopy_params.yaml.xz`` (force constants + cell) loads directly with
``phonopy.load`` — no API key required. This is the reference used by the
foundation-MLIP phonon benchmark literature (Loew et al., npj Comput. Mater.
2025).

The mp-id -> MDR dataset-id map lives in
``data/benchmark/mdr_index.csv`` (parsed from atztogo/phonondb).

Usage
-----
    from phonon_accel import reference
    ref_result, ref_phonon = reference.reference_result("mp-149")  # Si
    atoms = reference.reference_atoms("mp-149")  # DFT-relaxed unit cell
"""
from __future__ import annotations

import csv
import io
import lzma
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import phonopy

from .phonons import PhononResult, build_result_from_phonon, phonopy_to_ase

_PKG_ROOT = Path(__file__).resolve().parents[2]
INDEX_CSV = _PKG_ROOT / "data" / "benchmark" / "mdr_index.csv"
CACHE_DIR = _PKG_ROOT / "data" / "benchmark" / "mdr"
DOWNLOAD_URL = "https://mdr.nims.go.jp/download_all/{dataset_id}.zip"


@lru_cache(maxsize=1)
def _index() -> dict:
    """mp_id -> row dict from the MDR index CSV."""
    out = {}
    with open(INDEX_CSV) as f:
        for row in csv.DictReader(f):
            out[row["mp_id"]] = row
    return out


def list_materials(formula: Optional[str] = None, spacegroup_number: Optional[int] = None):
    """Filter the MDR index. Returns list of row dicts."""
    rows = list(_index().values())
    if formula is not None:
        rows = [r for r in rows if r["formula"] == formula]
    if spacegroup_number is not None:
        rows = [r for r in rows if r["spacegroup_number"] == str(spacegroup_number)]
    return rows


def dataset_id(mp_id: str) -> str:
    idx = _index()
    if mp_id not in idx:
        raise KeyError(f"{mp_id} not in MDR index ({len(idx)} materials)")
    return idx[mp_id]["dataset_id"]


def fetch(mp_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Download + extract ``phonopy_params.yaml.xz`` for ``mp_id``; cache it.

    Returns the path to the extracted (decompressed) phonopy yaml.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{mp_id}_phonopy_params.yaml"
    if out.exists():
        return out

    url = DOWNLOAD_URL.format(dataset_id=dataset_id(mp_id))
    with urllib.request.urlopen(url, timeout=90) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = next(n for n in zf.namelist() if n.endswith("phonopy_params.yaml.xz"))
        raw = lzma.decompress(zf.read(name))
    out.write_bytes(raw)
    return out


def load_reference_phonon(mp_id: str, is_nac: bool = False) -> phonopy.Phonopy:
    """Load the DFPT reference as a phonopy object (force constants set).

    ``is_nac=False`` by default: short-range MLIPs carry no Born effective
    charges, so for an apples-to-apples comparison we disable the
    non-analytical term correction (LO-TO splitting) on the reference too.
    This isolates the short-range force constants — exactly what the
    FC-distillation fine-tuning targets. (NAC changes nothing for non-polar
    crystals like Si, but shifts polar LO modes substantially, e.g. MgO
    20.8 -> 11.7 THz near Gamma.)
    """
    path = fetch(mp_id)
    return phonopy.load(str(path), is_nac=is_nac)


def reference_atoms(mp_id: str):
    """The DFT-relaxed unit cell of the reference, as ASE ``Atoms``."""
    ph = load_reference_phonon(mp_id)
    return phonopy_to_ase(ph.unitcell)


def reference_result(
    mp_id: str, mesh=(24, 24, 24), band_npoints: int = 101
) -> tuple[PhononResult, phonopy.Phonopy]:
    """Build a reference ``PhononResult`` from the DFPT force constants."""
    ph = load_reference_phonon(mp_id)
    formula = "".join(sorted(set(ph.unitcell.symbols)))
    res = build_result_from_phonon(
        ph,
        formula=_index().get(mp_id, {}).get("formula", formula),
        supercell_matrix=ph.supercell_matrix.tolist(),
        n_displacements=-1,  # reference: force constants come precomputed
        mesh=mesh,
        band_npoints=band_npoints,
    )
    return res, ph
