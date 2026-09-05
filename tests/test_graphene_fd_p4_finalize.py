from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase import Atoms
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "scripts" / "smearing_kink"
sys.path.insert(0, str(MODULES))

from graphene_fd_p4_common import (  # noqa: E402
    apply_top_projector,
    atomic_npz,
    geometry_sha256,
    line_metrics,
    p4_key,
    rounded_quadratic_features,
    sha256,
)
import bootstrap_graphene_fd_p4_dft_tdep as p4_tdep  # noqa: E402
import evaluate_graphene_fd_transferability as p4_evaluate  # noqa: E402
import friedel_module as fm  # noqa: E402
import merge_graphene_fd_p4_labels as merge  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402


def _write_lane(path: Path, seed: int, indices: list[int]) -> None:
    path.mkdir(parents=True)
    structures = []
    records = []
    for index in indices:
        atoms = Atoms(
            "C2",
            positions=[[0.0, 0.0, 0.0], [1.2 + seed * 0.01, 0.0, 0.0]],
            cell=[4.0, 4.0, 8.0],
            pbc=True,
        )
        atoms.arrays["REF_forces"] = np.full((2, 3), seed + index / 1000.0)
        atoms.info.update(
            {
                "REF_energy": -10.0 - seed,
                "snapshot_index": index,
                "trajectory_seed": seed,
                "split": "validation",
                "config_type": f"fd_force_conv_snapshot_{index:03d}",
                "degauss_Ry": 0.00285013035,
                "lattice_temperature_K": 450.0,
                "kgrid": 8,
            }
        )
        structures.append(atoms)
        records.append(
            {
                "snapshot_index": index,
                "trajectory_seed": seed,
                "split": "validation",
                "kgrid": 8,
                "reference_kgrid": 8,
            }
        )
    write(path / "summary.xyz", structures, format="extxyz")
    summary = {
        "trajectory_seed": seed,
        "indices": indices,
        "n_atoms": 2,
        "lattice_temperature_K": 450.0,
        "smearing": "fermi-dirac",
        "degauss_Ry": 0.00285013035,
        "ecutwfc_Ry": 60.0,
        "ecutrho_Ry": 240.0,
        "disk_io": "none",
        "kgrids": [8],
        "reference_kgrid": 8,
        "records": records,
        "total_wall_seconds": 1.0,
    }
    (path / "summary.json").write_text(json.dumps(summary) + "\n")


class P4FinalizeTests(unittest.TestCase):
    def test_common_qspace_metrics_and_projector_replay(self):
        features = rounded_quadratic_features(np.array([0.0, 0.01]), 0.0025)
        self.assertEqual(features.shape, (2, 3))
        self.assertTrue(np.allclose(features[0], [1.0, 0.0, 0.0]))

        eigenvalues = np.arange(1.0, 7.0)
        matrix = np.diag(eigenvalues)
        frequencies = np.sqrt(eigenvalues * 100.0)
        corrected, scalar = apply_top_projector(
            matrix, 25.0, frequencies, gamma=False
        )
        self.assertLess(abs(corrected[-1] - scalar), 1.0e-12)

        t = np.array([0.992, 1.000, 1.008])
        target = np.array([1292.0, 1289.0, 1293.0])
        metrics = line_metrics("K", t, target.copy(), target)
        self.assertEqual(metrics["line_MAE_cm-1"], 0.0)
        self.assertEqual(metrics["high_symmetry_abs_error_cm-1"], 0.0)
        self.assertEqual(metrics["kink_relative_error"], 0.0)

    def test_merge_uses_seed_and_index_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            freeze_path = tmp_path / "freeze.json"
            freeze_path.write_text(
                json.dumps(
                    {
                        "T450_on_policy": {
                            "temperature_K": 450,
                            "degauss_Ry": 0.00285013035,
                            "dft_label_indices_per_seed": [3, 9],
                        }
                    }
                )
                + "\n"
            )
            freeze_sha = sha256(freeze_path)
            sampling_path = tmp_path / "sampling.json"
            sampling_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "passes_sampling_gate": True,
                        "freeze_manifest": {"sha256": freeze_sha},
                    }
                )
                + "\n"
            )
            sampling_sha = sha256(sampling_path)
            bootstrap_path = tmp_path / "bootstrap.json"
            bootstrap_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "passes_sampling_bootstrap_gate": True,
                        "freeze_manifest_sha256": freeze_sha,
                        "sampling_acceptance_sha256": sampling_sha,
                    }
                )
                + "\n"
            )
            bootstrap_sha = sha256(bootstrap_path)

            labels_root = tmp_path / "labels"
            lanes = {
                ("A", 0): [3, 9],
                ("A", 2): [3],
                ("B", 1): [3, 9],
                ("B", 2): [9],
            }
            for (shard, seed), indices in lanes.items():
                _write_lane(labels_root / f"shard_{shard}" / f"seed{seed}", seed, indices)
            for shard in ("A", "B"):
                shard_path = labels_root / f"shard_{shard}"
                (shard_path / "RAW_READY").touch()
                (shard_path / "input_manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "validated_for_dft_force_labels",
                            "shard": shard,
                            "n_selected_structures": 30,
                            "freeze_manifest_sha256": freeze_sha,
                            "sampling_acceptance_sha256": sampling_sha,
                            "sampling_bootstrap_sha256": bootstrap_sha,
                        }
                    )
                    + "\n"
                )

            output = tmp_path / "merged"
            arguments = [
                "merge_graphene_fd_p4_labels.py",
                "--labels-root",
                str(labels_root),
                "--freeze-manifest",
                str(freeze_path),
                "--sampling-acceptance",
                str(sampling_path),
                "--sampling-bootstrap",
                str(bootstrap_path),
                "--output-dir",
                str(output),
            ]
            with mock.patch.object(sys, "argv", arguments):
                self.assertEqual(merge.main(), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["identity_key"], "(trajectory_seed, snapshot_index)")
            self.assertEqual(manifest["n_structures"], 6)
            structures = read(output / "all60.xyz", index=":")
            identities = [
                (int(atoms.info["trajectory_seed"]), int(atoms.info["snapshot_index"]))
                for atoms in structures
            ]
            self.assertEqual(
                identities,
                [(0, 3), (0, 9), (1, 3), (1, 9), (2, 3), (2, 9)],
            )

    def test_dft_tdep_two_replicate_smoke(self):
        background = ROOT / "results" / "vq_kink6_fd" / "graphene_sc6_dg0.040_phonopy.yaml"
        if not background.is_file():
            self.skipTest("graphene 6x6 background is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            indices = list(range(3, 118, 6))
            freeze_path = tmp_path / "freeze.json"
            freeze_path.write_text(
                json.dumps(
                    {
                        "T450_on_policy": {
                            "temperature_K": 450,
                            "degauss_Ry": 0.00285013035,
                            "dft_label_indices_per_seed": indices,
                        },
                        "fixed_acceptance_thresholds": {
                            "bootstrap_replicates": 2,
                            "bootstrap_block_rule": "max(5, ceil(2*tau_int))",
                        },
                    }
                )
                + "\n"
            )
            sampling_bootstrap_path = tmp_path / "sampling_bootstrap.json"
            sampling_bootstrap_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "passes_sampling_bootstrap_gate": True,
                        "block_bootstrap": {
                            "trajectory_diagnostics": [
                                {"seed": seed, "block_length_snapshots": 5}
                                for seed in (0, 1, 2)
                            ]
                        },
                    }
                )
                + "\n"
            )

            phonon = fm.load_ph(background)
            ideal = phonopy_to_ase(phonon.supercell)
            ideal.wrap()
            rng = np.random.default_rng(450)
            seed_outputs = {}
            all_structures = []
            for seed in (0, 1, 2):
                structures = []
                for snapshot_index in indices:
                    structure = ideal.copy()
                    structure.positions += rng.normal(scale=0.01, size=structure.positions.shape)
                    displacement = structure.positions - ideal.positions
                    forces = -5.0 * displacement
                    forces -= np.mean(forces, axis=0, keepdims=True)
                    energy = 2.5 * float(np.sum(displacement**2))
                    structure.arrays["REF_forces"] = forces
                    structure.info.update(
                        {
                            "REF_energy": energy,
                            "trajectory_seed": seed,
                            "snapshot_index": snapshot_index,
                            "split": "validation",
                        }
                    )
                    structures.append(structure)
                    all_structures.append(structure)
                seed_path = tmp_path / f"seed{seed}.xyz"
                write(seed_path, structures, format="extxyz")
                seed_outputs[str(seed)] = {
                    "path": str(seed_path),
                    "sha256": sha256(seed_path),
                }
            merge_path = tmp_path / "merge.json"
            merge_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "n_structures": 60,
                        "freeze_manifest": {"sha256": sha256(freeze_path)},
                        "sampling_bootstrap": {
                            "sha256": sha256(sampling_bootstrap_path)
                        },
                        "seed_outputs": seed_outputs,
                    }
                )
                + "\n"
            )
            output_tdep = tmp_path / "dft_tdep.npz"
            bootstrap_output = tmp_path / "bootstrap.npz"
            arguments = [
                "bootstrap_graphene_fd_p4_dft_tdep.py",
                "--seed-label",
                f"0={tmp_path / 'seed0.xyz'}",
                "--seed-label",
                f"1={tmp_path / 'seed1.xyz'}",
                "--seed-label",
                f"2={tmp_path / 'seed2.xyz'}",
                "--merge-manifest",
                str(merge_path),
                "--freeze-manifest",
                str(freeze_path),
                "--sampling-bootstrap",
                str(sampling_bootstrap_path),
                "--background",
                str(background),
                "--replicates",
                "2",
                "--checkpoint-every",
                "1",
                "--output-tdep",
                str(output_tdep),
                "--output-summary",
                str(tmp_path / "dft_tdep.json"),
                "--bootstrap-output",
                str(bootstrap_output),
                "--bootstrap-summary",
                str(tmp_path / "bootstrap.json"),
            ]
            with mock.patch.object(sys, "argv", arguments):
                self.assertEqual(p4_tdep.main(), 0)
            with np.load(output_tdep, allow_pickle=False) as data:
                self.assertEqual(data["T450_fc2"].shape, (72, 72, 3, 3))
                self.assertTrue(np.isfinite(data["T450_line_top_cm_1"]).all())
            with np.load(bootstrap_output, allow_pickle=False) as data:
                self.assertEqual(data["line_top_cm_1"].shape, (2, 19))
                self.assertTrue(np.isfinite(data["line_top_cm_1"]).all())

            # Build a zero-error synthetic P4 bundle around the fitted point target.
            # This checks only the evaluator and provenance logic; it does not tune a model.
            all_labels = tmp_path / "all60.xyz"
            write(all_labels, all_structures, format="extxyz")
            evaluation_structures = read(all_labels, index=":")
            law_path = tmp_path / "temperature_law.json"
            law_path.write_text(
                json.dumps(
                    {
                        "status": "frozen_before_450_holdout",
                        "prediction_temperature_K": 450,
                        "prediction_degauss_Ry": 0.00285013035,
                        "predicted_coefficients_delta_lambda_cm-2": {
                            "G": [0.0, 0.0, 0.0],
                            "K": [0.0, 0.0, 0.0],
                        },
                    }
                )
                + "\n"
            )
            eval_freeze = tmp_path / "eval_freeze.json"
            eval_freeze.write_text(
                json.dumps(
                    {
                        "status": "frozen_before_450_holdout",
                        "temperature_law": {"sha256": sha256(law_path)},
                        "T450_on_policy": {
                            "temperature_K": 450,
                            "degauss_Ry": 0.00285013035,
                            "dft_label_indices_per_seed": indices,
                            "primary_force_indices_per_seed": [3, 27, 51, 75, 99],
                        },
                        "fixed_acceptance_thresholds": {
                            "force_RMSE_meV_A": 50.0,
                            "force_max_abs_meV_A": 250.0,
                            "Gamma_K_line_MAE_cm-1": 10.0,
                            "Gamma_K_top_abs_error_cm-1": 15.0,
                            "K_kink_relative_error": 0.2,
                            "matrix_projector_replay_max_abs_cm-1": 1.0e-5,
                            "bootstrap_replicates": 2,
                            "bootstrap_block_rule": "max(5, ceil(2*tau_int))",
                        },
                    }
                )
                + "\n"
            )
            eval_freeze_sha = sha256(eval_freeze)
            sampling_acceptance = tmp_path / "sampling_acceptance.json"
            sampling_acceptance.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "passes_sampling_gate": True,
                        "freeze_manifest": {"sha256": eval_freeze_sha},
                        "pooled_tdep": {"sha256": sha256(output_tdep)},
                    }
                )
                + "\n"
            )
            eval_sampling_bootstrap = tmp_path / "eval_sampling_bootstrap.json"
            eval_sampling_bootstrap.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "passes_sampling_bootstrap_gate": True,
                        "freeze_manifest_sha256": eval_freeze_sha,
                        "sampling_acceptance_sha256": sha256(sampling_acceptance),
                        "block_bootstrap": {
                            "trajectory_diagnostics": [
                                {"seed": seed, "block_length_snapshots": 5}
                                for seed in (0, 1, 2)
                            ]
                        },
                    }
                )
                + "\n"
            )
            eval_merge = tmp_path / "eval_merge.json"
            eval_merge.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "n_structures": 60,
                        "freeze_manifest": {"sha256": eval_freeze_sha},
                        "all_output": {"sha256": sha256(all_labels)},
                    }
                )
                + "\n"
            )
            eval_merge_sha = sha256(eval_merge)

            keys = []
            geometry_hashes = []
            references = []
            for structure in evaluation_structures:
                seed = int(structure.info["trajectory_seed"])
                snapshot_index = int(structure.info["snapshot_index"])
                keys.append(p4_key(seed, snapshot_index))
                geometry_hashes.append(
                    geometry_sha256(structure.numbers, structure.cell, structure.positions)
                )
                references.append(np.asarray(structure.arrays["REF_forces"], float))
            predictions = tmp_path / "predictions.npz"
            references = np.asarray(references, float)
            atomic_npz(
                predictions,
                keys=np.asarray(keys),
                geometry_sha256=np.asarray(geometry_hashes),
                predicted_total_forces_eV_A=references,
                predicted_short_forces_eV_A=references,
                predicted_long_range_forces_eV_A=np.zeros_like(references),
            )
            with np.load(output_tdep, allow_pickle=False) as data:
                fitted_fc = np.asarray(data["T450_fc2"], float)
                point_top = np.asarray(data["T450_line_top_cm_1"], float)
            static_prediction = tmp_path / "static.npz"
            atomic_npz(static_prediction, full_force_constants=fitted_fc)
            prediction_manifest = tmp_path / "prediction_manifest.json"
            prediction_manifest.write_text(
                json.dumps(
                    {
                        "inputs": {
                            "freeze_manifest": {"sha256": eval_freeze_sha}
                        },
                        "force_predictions": {"sha256": sha256(predictions)},
                        "static_prediction": {"sha256": sha256(static_prediction)},
                    }
                )
                + "\n"
            )
            dft_summary = tmp_path / "eval_dft_summary.json"
            dft_summary.write_text(
                json.dumps(
                    {
                        "output": {"sha256": sha256(output_tdep)},
                        "merge_manifest_sha256": eval_merge_sha,
                    }
                )
                + "\n"
            )
            eval_dft_bootstrap = tmp_path / "eval_dft_bootstrap.npz"
            atomic_npz(
                eval_dft_bootstrap,
                replicates=np.array(2),
                line_top_cm_1=np.stack([point_top, point_top]),
            )
            eval_dft_bootstrap_summary = tmp_path / "eval_dft_bootstrap.json"
            eval_dft_bootstrap_summary.write_text(
                json.dumps(
                    {
                        "raw_distributions": {
                            "sha256": sha256(eval_dft_bootstrap)
                        },
                        "merge_manifest_sha256": eval_merge_sha,
                        "point_tdep": {"sha256": sha256(output_tdep)},
                    }
                )
                + "\n"
            )
            convergence = tmp_path / "convergence.json"
            convergence.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "passes_convergence_gate": True,
                        "degauss_Ry": 0.00285013035,
                    }
                )
                + "\n"
            )
            dfpt_line = tmp_path / "dfpt_line.csv"
            with dfpt_line.open("w", newline="") as handle:
                fieldnames = [
                    "campaign",
                    "degauss_Ry",
                    "kgrid",
                    "region",
                    "t_GK",
                    "f6_cm-1",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for (region, t_value), top in zip(
                    p4_evaluate.P4_LINE_QPOINTS, point_top, strict=True
                ):
                    writer.writerow(
                        {
                            "campaign": "FD450_LINE",
                            "degauss_Ry": 0.00285013035,
                            "kgrid": 144,
                            "region": region,
                            "t_GK": t_value,
                            "f6_cm-1": top,
                        }
                    )
            eval_output = tmp_path / "evaluation"
            eval_arguments = [
                "evaluate_graphene_fd_transferability.py",
                "--labels",
                str(all_labels),
                "--merge-manifest",
                str(eval_merge),
                "--predictions",
                str(predictions),
                "--prediction-manifest",
                str(prediction_manifest),
                "--static-prediction",
                str(static_prediction),
                "--dft-tdep",
                str(output_tdep),
                "--dft-tdep-summary",
                str(dft_summary),
                "--dft-bootstrap",
                str(eval_dft_bootstrap),
                "--dft-bootstrap-summary",
                str(eval_dft_bootstrap_summary),
                "--short-tdep",
                str(output_tdep),
                "--temperature-law",
                str(law_path),
                "--freeze-manifest",
                str(eval_freeze),
                "--sampling-acceptance",
                str(sampling_acceptance),
                "--sampling-bootstrap",
                str(eval_sampling_bootstrap),
                "--dfpt-convergence",
                str(convergence),
                "--dfpt-line",
                str(dfpt_line),
                "--background",
                str(background),
                "--output-dir",
                str(eval_output),
            ]
            with mock.patch.object(sys, "argv", eval_arguments):
                self.assertEqual(p4_evaluate.main(), 0)
            acceptance = json.loads(
                (eval_output / "transferability_acceptance.json").read_text()
            )
            self.assertTrue(acceptance["passes_C1"])


if __name__ == "__main__":
    unittest.main()
