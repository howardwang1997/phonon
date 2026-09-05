"""Reusable pieces of the frozen 8x8 baseline and q-space mode correction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import friedel_module as fm  # noqa: E402
import graphene_p0_deploy as p0  # noqa: E402
from friedel_calc import HarmonicCalculator, fc2_from_calc, full_fc  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402


def load_cached_fc(cache_dir: Path, degauss_ry: float) -> np.ndarray:
    path = cache_dir / (
        f"graphene_8x8_dg{p0.dg_slug(degauss_ry)}_p0_fc2.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        fc = np.asarray(data["fc2"], float)
    if fc.ndim != 4 or not np.isfinite(fc).all():
        raise ValueError(f"invalid cached force constants: {path}")
    return fc


@dataclass
class Frozen8x8Baseline:
    """Independent additive replay of the frozen log-interpolation baseline.

    The original deployment runs finite displacements through
    ``MACE + harmonic correction``.  Since the correction is exactly harmonic,
    its compact force constants can be added to the archived MACE-backbone
    force constants without rerunning MACE.  Existing cache replay validates
    that equivalence to finite-displacement numerical precision before this
    object is used for unseen smearings.
    """

    ph: object
    fc_dft_background: np.ndarray
    fc_dft_template: np.ndarray
    tabs: list[dict[str, np.ndarray]]
    d0_rows: list[np.ndarray]
    fc_mace_backbone: np.ndarray
    b_law: object
    kappa_law: object
    rmin: float
    rmax: float
    anchors: dict[float, tuple[float, float, float]]

    def envelope(self, degauss_ry: float) -> tuple[float, float]:
        temperature = float(degauss_ry) * p0.K_PER_RY
        return float(self.b_law(temperature)), float(self.kappa_law(temperature))

    def force_constants_algebraic(self, degauss_ry: float) -> np.ndarray:
        """Fast direct full-FC addition (diagnostic, not the frozen replay)."""
        b_value, kappa = self.envelope(degauss_ry)
        with_correction = fm.add_template(
            self.fc_dft_background,
            self.tabs,
            self.d0_rows,
            b_value,
            kappa,
            self.rmin,
            self.rmax,
        )
        compact_delta = with_correction - self.fc_dft_background
        return self.fc_mace_backbone + full_fc(self.ph, compact_delta)

    def force_constants(self, degauss_ry: float, distance: float = 0.03) -> np.ndarray:
        """Replay the harmonic correction through the production FD operator.

        ``phonopy.produce_force_constants`` applies the same symmetry/displacement
        reconstruction used by the original combined MACE+correction run.  The
        finite-difference operator is linear, so replaying only the harmonic
        correction and adding the archived MACE-backbone FC preserves the
        frozen target modes to finite-difference precision without rerunning
        MACE.
        """
        b_value, kappa = self.envelope(degauss_ry)
        with_correction = fm.add_template(
            self.fc_dft_background,
            self.tabs,
            self.d0_rows,
            b_value,
            kappa,
            self.rmin,
            self.rmax,
        )
        compact_delta = with_correction - self.fc_dft_background
        delta_full = full_fc(self.ph, compact_delta)
        reference_atoms = phonopy_to_ase(self.ph.supercell)
        calculator = HarmonicCalculator(reference_atoms, delta_full)
        _, recovered_delta = fc2_from_calc(
            self.ph,
            calculator,
            distance=distance,
            subtract_ref=False,
        )
        return self.fc_mace_backbone + recovered_delta

    def qpoint_data(self, degauss_ry: float, t_values: np.ndarray):
        qpoints = np.array([[t / 3.0, t / 3.0, 0.0] for t in t_values])
        self.ph.force_constants = self.force_constants(degauss_ry)
        self.ph.run_qpoints(qpoints, with_dynamical_matrices=True)
        result = self.ph.get_qpoints_dict()
        frequencies = np.sort(np.asarray(result["frequencies"], float), axis=1) * p0.CM
        matrices = np.asarray(result["dynamical_matrices"], complex)
        return frequencies, matrices


def build_primary_baseline() -> Frozen8x8Baseline:
    cfg = p0.DATASETS["8x8"]
    paths = p0.discover_yamls(cfg)
    ref_dg = p0.require_point(paths, cfg.reference_dg)
    bg_dg = p0.require_point(paths, cfg.background_dg)
    anchor_keys = [p0.require_point(paths, value) for value in cfg.anchor_dgs]
    ph_by_dg = {dg: fm.load_ph(path) for dg, path in paths.items()}
    fc_dft = {dg: ph.force_constants.copy() for dg, ph in ph_by_dg.items()}
    ph = ph_by_dg[bg_dg]
    tabs, _ = fm.pair_table(ph)
    d0_rows = fm.template_delta(fc_dft[ref_dg], fc_dft[bg_dg], tabs)
    kappas = np.linspace(0.0, 2.0, 321)
    fitted: dict[float, tuple[float, float, float]] = {}
    for dg in anchor_keys:
        if p0.same_dg(dg, ref_dg):
            fitted[dg] = (1.0, 0.0, 0.0)
        elif p0.same_dg(dg, bg_dg):
            fitted[dg] = (0.0, 0.0, 0.0)
        else:
            fitted[dg] = fm.fit_template_env(
                fc_dft[dg],
                fc_dft[bg_dg],
                tabs,
                d0_rows,
                1.0,
                cfg.rmax,
                kappas,
            )
    temperatures = np.asarray(anchor_keys) * p0.K_PER_RY
    b_law = p0.log_interp_law(
        temperatures, np.asarray([fitted[dg][0] for dg in anchor_keys])
    )
    kappa_law = p0.log_interp_law(
        temperatures, np.asarray([fitted[dg][1] for dg in anchor_keys])
    )
    cache = ROOT / "results" / "p0_graphene_prospective" / "cache"
    backbone_path = cache / "graphene_8x8_v11_backbone_fc2.npz"
    with np.load(backbone_path, allow_pickle=False) as data:
        fc_backbone = np.asarray(data["fc2"], float)
    if fc_backbone.ndim != 4 or not np.isfinite(fc_backbone).all():
        raise ValueError(f"invalid archived MACE backbone: {backbone_path}")
    return Frozen8x8Baseline(
        ph=ph,
        fc_dft_background=fc_dft[bg_dg],
        fc_dft_template=fc_dft[ref_dg],
        tabs=tabs,
        d0_rows=d0_rows,
        fc_mace_backbone=fc_backbone,
        b_law=b_law,
        kappa_law=kappa_law,
        rmin=1.0,
        rmax=cfg.rmax,
        anchors=fitted,
    )


def audit_primary_cache_replay(model: Frozen8x8Baseline) -> dict[str, float]:
    cache = ROOT / "results" / "p0_graphene_prospective" / "cache"
    pattern = re.compile(r"graphene_8x8_dg(.+)_p0_fc2\.npz$")
    errors = {}
    for path in sorted(cache.glob("graphene_8x8_dg*_p0_fc2.npz")):
        match = pattern.fullmatch(path.name)
        if not match or "v11_backbone" in path.name:
            continue
        dg = float(match.group(1))
        with np.load(path, allow_pickle=False) as data:
            archived = np.asarray(data["fc2"], float)
        errors[f"{dg:g}"] = float(
            np.max(np.abs(model.force_constants(dg) - archived))
        )
    if not errors:
        raise RuntimeError(f"no archived primary caches in {cache}")
    return errors
