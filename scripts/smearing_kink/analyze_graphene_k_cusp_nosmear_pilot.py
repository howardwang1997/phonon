#!/usr/bin/env python3
"""Audit the two-host no-degauss graphene K-cusp pilot.

The pilot contains an independently repeated exact K point and one equal-
distance point in each of the K->Gamma and K->M directions.  It decides only
whether optimized-tetrahedron DFPT is numerically usable and whether the local
cusp has the expected direction; it does not fit the repository correction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qe_dyn import QEDyn, load_qe_dyn  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_lane(path: Path, expected_lane: str) -> dict:
    manifest_path = path / "manifest.json"
    csv_path = path / "dfpt_points.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete manifest: {manifest_path}")
    if manifest.get("stage") != "pilot" or manifest.get("lane") != expected_lane:
        raise ValueError(f"unexpected stage/lane in {manifest_path}")
    if manifest.get("electronic_integration") != "occupations='tetrahedra_opt'":
        raise ValueError(f"wrong electronic integration in {manifest_path}")
    if manifest.get("smearing") is not None or manifest.get("degauss_Ry") is not None:
        raise ValueError(f"finite smearing recorded in {manifest_path}")
    if digest(csv_path) != manifest["outputs"]["csv_sha256"]:
        raise ValueError(f"CSV hash mismatch in {path}")
    if digest(ROOT / "configs/graphene_k_cusp_nosmear/qpoints.tsv") != manifest["config"]["sha256"]:
        raise ValueError(f"q-point freeze hash mismatch in {path}")

    rows = read_csv(csv_path)
    if len(rows) != 2 or int(manifest["n_qpoints"]) != 2:
        raise ValueError(f"pilot lane {expected_lane} must contain two q points")
    audit_by_label = {row["label"]: row for row in manifest["qpoint_audits"]}
    dyn = {}
    for row in rows:
        label = row["label"]
        dyn_path = path / "bundle" / label / "gr.dyn"
        audit = audit_by_label[label]
        if digest(dyn_path) != audit["gr_dyn_sha256"]:
            raise ValueError(f"gr.dyn hash mismatch for {expected_lane}/{label}")
        dyn[label] = load_qe_dyn(dyn_path)
    return {
        "path": path,
        "manifest_path": manifest_path,
        "manifest_sha256": digest(manifest_path),
        "manifest": manifest,
        "rows": rows,
        "dyn": dyn,
    }


def select_by_overlap(reference: QEDyn, target: QEDyn) -> tuple[int, float]:
    reference_vector = reference.eigenvectors[-1].reshape(-1)
    candidates = np.arange(len(target.frequencies_cm) - 3, len(target.frequencies_cm))
    overlaps = np.asarray(
        [
            abs(np.vdot(reference_vector, target.eigenvectors[index].reshape(-1))) ** 2
            for index in candidates
        ]
    )
    selected = int(candidates[int(np.argmax(overlaps))])
    return selected, float(np.max(overlaps))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(records: list[dict], output: Path) -> None:
    by_direction = {row["direction"]: row for row in records}
    k_frequency = by_direction["K"]["tracked_A1prime_cm-1"]
    x = np.asarray(
        [
            -by_direction["KM"]["delta_equal_distance"],
            0.0,
            by_direction["KG"]["delta_equal_distance"],
        ]
    )
    y = np.asarray(
        [
            by_direction["KM"]["tracked_A1prime_cm-1"],
            k_frequency,
            by_direction["KG"]["tracked_A1prime_cm-1"],
        ]
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    fig, axis = plt.subplots(figsize=(5.6, 3.8))
    axis.plot(x, y, "o-", color="#1F5A94", lw=1.5, ms=5)
    axis.axvline(0.0, color="#B8B8B8", lw=0.8, zorder=0)
    axis.grid(axis="y", color="#E6E6E6", lw=0.55)
    axis.set_xticks(x, ["K→M", "K", "K→Γ"])
    axis.set_xlabel("equal-distance K-window pilot")
    axis.set_ylabel(r"tracked $A_1'$ frequency (cm$^{-1}$)")
    axis.set_title("Graphene no-degauss direct-DFPT pilot")
    axis.text(
        0.5,
        -0.22,
        "electronic integration: tetrahedra_opt (no degauss)",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.27)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def write_report(summary: dict, path: Path) -> None:
    checks = summary["checks"]
    metrics = summary["metrics"]
    lines = [
        "# Graphene K cusp 无 degauss pilot",
        "",
        f"**状态：**`{summary['status']}`  ",
        "**电子积分：**`tetrahedra_opt (no degauss)`  ",
        f"**k 网格：**`{summary['kgrid']}×{summary['kgrid']}×1`",
        "",
        "## 数值",
        "",
        f"- K 点跨机器六模最大差：{metrics['cross_host_K_all_mode_max_abs_cm-1']:.6f} cm⁻¹；",
        f"- K→Γ，d=0.007 的 A1′ 上升量：{metrics['KG_cusp_depth_cm-1']:.6f} cm⁻¹；",
        f"- K→M，等距离 d=0.007 的 A1′ 上升量：{metrics['KM_cusp_depth_cm-1']:.6f} cm⁻¹；",
        f"- 最小本征矢重叠：{metrics['minimum_mode_overlap']:.6f}。",
        "",
        "## 检查",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "该阶段只判断四面体积分和 cusp 方向是否可用；k 网格收敛、形状拟合和 blind holdout 仍需后续阶段完成。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-a", type=Path, required=True)
    parser.add_argument("--lane-b", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/graphene_k_cusp_nosmear/S1_pilot",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lane_a = load_lane(args.lane_a, "A")
    lane_b = load_lane(args.lane_b, "B")
    if lane_a["manifest"]["kgrid"] != lane_b["manifest"]["kgrid"]:
        raise ValueError("pilot lanes use different k grids")
    if lane_a["manifest"]["qe_tag"] != lane_b["manifest"]["qe_tag"]:
        raise ValueError("pilot lanes use different QE tags")
    same_qe_hashes = bool(
        lane_a["manifest"]["quantum_espresso"]["pw_sha256"]
        == lane_b["manifest"]["quantum_espresso"]["pw_sha256"]
        and lane_a["manifest"]["quantum_espresso"]["ph_sha256"]
        == lane_b["manifest"]["quantum_espresso"]["ph_sha256"]
    )
    same_pseudo_hash = bool(
        lane_a["manifest"]["pseudopotential"]["sha256"]
        == lane_b["manifest"]["pseudopotential"]["sha256"]
    )

    k_a, k_b = lane_a["dyn"]["K"], lane_b["dyn"]["K"]
    cross_host = float(np.max(np.abs(k_a.frequencies_cm - k_b.frequencies_cm)))
    kg = lane_a["dyn"]["KG_d007"]
    km = lane_b["dyn"]["KM_d007"]
    kg_mode, kg_overlap = select_by_overlap(k_a, kg)
    km_mode, km_overlap = select_by_overlap(k_b, km)
    k_frequency = float(0.5 * (k_a.frequencies_cm[-1] + k_b.frequencies_cm[-1]))
    kg_frequency = float(kg.frequencies_cm[kg_mode])
    km_frequency = float(km.frequencies_cm[km_mode])
    kg_depth = kg_frequency - k_frequency
    km_depth = km_frequency - k_frequency

    aggregate_hermiticity = max(
        lane_a["manifest"]["aggregate_audit"]["Hermiticity_max_abs"],
        lane_b["manifest"]["aggregate_audit"]["Hermiticity_max_abs"],
    )
    aggregate_replay = max(
        lane_a["manifest"]["aggregate_audit"]["matrix_frequency_replay_max_abs_cm-1"],
        lane_b["manifest"]["aggregate_audit"]["matrix_frequency_replay_max_abs_cm-1"],
    )
    checks = {
        "no_smearing_or_degauss": True,
        "same_kgrid_and_QE_tag": True,
        "same_QE_binary_hashes": same_qe_hashes,
        "same_pseudopotential_hash": same_pseudo_hash,
        "Hermiticity_le_5e-7": aggregate_hermiticity <= 5.0e-7,
        "matrix_replay_le_0p1_cm-1": aggregate_replay <= 0.1,
        "cross_host_K_le_0p2_cm-1": cross_host <= 0.2,
        "minimum_mode_overlap_ge_0p95": min(kg_overlap, km_overlap) >= 0.95,
        "KG_neighbor_above_K": kg_depth > 0.0,
        "KM_neighbor_above_K": km_depth > 0.0,
    }
    passed = all(checks.values())
    records = [
        {
            "direction": "KM",
            "delta_equal_distance": 0.007,
            "tracked_mode_1based": km_mode + 1,
            "tracked_A1prime_cm-1": km_frequency,
            "relative_to_K_cm-1": km_depth,
            "mode_overlap": km_overlap,
            "source_lane": "B",
        },
        {
            "direction": "K",
            "delta_equal_distance": 0.0,
            "tracked_mode_1based": 6,
            "tracked_A1prime_cm-1": k_frequency,
            "relative_to_K_cm-1": 0.0,
            "mode_overlap": 1.0,
            "source_lane": "A+B mean",
        },
        {
            "direction": "KG",
            "delta_equal_distance": 0.007,
            "tracked_mode_1based": kg_mode + 1,
            "tracked_A1prime_cm-1": kg_frequency,
            "relative_to_K_cm-1": kg_depth,
            "mode_overlap": kg_overlap,
            "source_lane": "A",
        },
    ]
    write_csv(args.output_dir / "pilot_points.csv", records)
    make_figure(records, args.output_dir / "graphene_nosmear_pilot.png")
    summary = {
        "status": "passed_pilot" if passed else "failed_pilot",
        "scope": "two-host k192 optimized-tetrahedron direct-DFPT pilot",
        "kgrid": lane_a["manifest"]["kgrid"],
        "qe_tag": lane_a["manifest"]["qe_tag"],
        "electronic_integration": "tetrahedra_opt (no degauss)",
        "inputs": {
            "lane_A_manifest": str(lane_a["manifest_path"]),
            "lane_A_manifest_sha256": lane_a["manifest_sha256"],
            "lane_B_manifest": str(lane_b["manifest_path"]),
            "lane_B_manifest_sha256": lane_b["manifest_sha256"],
        },
        "metrics": {
            "K_A_cm-1": float(k_a.frequencies_cm[-1]),
            "K_B_cm-1": float(k_b.frequencies_cm[-1]),
            "cross_host_K_all_mode_max_abs_cm-1": cross_host,
            "KG_d007_cm-1": kg_frequency,
            "KM_d007_cm-1": km_frequency,
            "KG_cusp_depth_cm-1": kg_depth,
            "KM_cusp_depth_cm-1": km_depth,
            "minimum_mode_overlap": min(kg_overlap, km_overlap),
            "Hermiticity_max_abs": aggregate_hermiticity,
            "matrix_frequency_replay_max_abs_cm-1": aggregate_replay,
        },
        "checks": checks,
        "next_stage": "build/replay npk=120000 QE and run k240 convergence" if passed else "stop before k-grid expansion",
    }
    atomic_json(args.output_dir / "pilot_summary.json", summary)
    write_report(summary, args.output_dir / "S1_RESULT.md")
    marker = args.output_dir / ("PASS" if passed else "FAILED")
    marker.write_text(summary["status"] + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
