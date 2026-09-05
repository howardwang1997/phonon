#!/usr/bin/env python3
"""Aggregate graphene/TMD evidence for the five-q EPC adapter decision."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
OUTPUT = BASE / "E27_low_data_epc_transfer_summary"


def read(relative: str) -> dict:
    return json.loads((BASE / relative).read_text(encoding="utf-8"))


def main() -> int:
    graphene = read(
        "E17_epc_vertex_rank_response_nk720/finite_vertex_rank_response_summary.json"
    )
    vse2_repair = read("E19_VSe2_rank_repair/VSe2_rank_repair_summary.json")
    vse2_response = read("E20_VSe2_rank_response/VSe2_rank_response_summary.json")
    tas2_vertex = read(
        "E21_TaS2_rank4_five_q_blind/rank3_five_q_blind_summary.json"
    )
    tas2_response = read("E22_TaS2_rank_response/rank_response_summary.json")
    nbs2_vertex = read(
        "E24_NbS2_rank4_five_q_blind/rank3_five_q_blind_summary.json"
    )
    nbs2_response = read("E25_NbS2_rank_response/rank_response_summary.json")

    graphene_metric = graphene["metrics_vs_full_EPC"]["rank_3_five_q_blind_basis"]
    vse2_vertex_metric = vse2_repair["rank4_five_q_interpolation_models"][
        "degree4_Chebyshev"
    ]
    vse2_response_metric = vse2_response["metrics_vs_full_EPC"][
        "blind_rank4_Chebyshev"
    ]
    rows = [
        {
            "material": "graphene",
            "evidence_type": "within-system q holdout",
            "selected_rank": 3,
            "rank_capture": graphene["rank_capture"]["3"],
            "vertex_RMSE_relative": graphene_metric["vertex_RMSE_relative_error"],
            "vertex_max_relative": graphene_metric["maximum_q_relative_vertex_error"],
            "frequency_RMSE_cm-1": graphene_metric["frequency_RMSE_cm-1"],
            "frequency_max_abs_cm-1": graphene_metric["frequency_max_abs_cm-1"],
            "shape_relative_error": graphene_metric[
                "maximum_cusp_depth_relative_error"
            ],
            "passed_fixed_repaired_model": None,
        },
        {
            "material": "1T-VSe2",
            "evidence_type": "development failure and repair",
            "selected_rank": 4,
            "rank_capture": vse2_repair["rank_results"]["4"]["full_data_capture"],
            "vertex_RMSE_relative": vse2_vertex_metric[
                "heldout_RMSE_relative_vertex_error"
            ],
            "vertex_max_relative": vse2_vertex_metric[
                "heldout_maximum_relative_vertex_error"
            ],
            "frequency_RMSE_cm-1": vse2_response_metric["frequency_RMSE_cm-1"],
            "frequency_max_abs_cm-1": vse2_response_metric[
                "frequency_max_abs_cm-1"
            ],
            "shape_relative_error": vse2_response_metric[
                "maximum_window_depth_relative_error"
            ],
            "passed_fixed_repaired_model": None,
        },
        {
            "material": "2H-TaS2",
            "evidence_type": "independent blind transfer",
            "selected_rank": 4,
            "rank_capture": tas2_vertex["full_data_selected_rank_capture"],
            "vertex_RMSE_relative": tas2_vertex["blind_five_q_selected_rank"][
                "heldout_RMSE_relative_vertex_error"
            ],
            "vertex_max_relative": tas2_vertex["blind_five_q_selected_rank"][
                "heldout_maximum_relative_vertex_error"
            ],
            "frequency_RMSE_cm-1": tas2_response["metrics_vs_full_EPC"][
                "blind_rank4_Chebyshev"
            ]["frequency_RMSE_cm-1"],
            "frequency_max_abs_cm-1": tas2_response["metrics_vs_full_EPC"][
                "blind_rank4_Chebyshev"
            ]["frequency_max_abs_cm-1"],
            "shape_relative_error": tas2_response["metrics_vs_full_EPC"][
                "blind_rank4_Chebyshev"
            ]["maximum_window_depth_relative_error"],
            "passed_fixed_repaired_model": tas2_response[
                "status"
            ] == "low_rank_response_comparison_passed",
        },
        {
            "material": "2H-NbS2",
            "evidence_type": "independent blind transfer",
            "selected_rank": 4,
            "rank_capture": nbs2_vertex["full_data_selected_rank_capture"],
            "vertex_RMSE_relative": nbs2_vertex["blind_five_q_selected_rank"][
                "heldout_RMSE_relative_vertex_error"
            ],
            "vertex_max_relative": nbs2_vertex["blind_five_q_selected_rank"][
                "heldout_maximum_relative_vertex_error"
            ],
            "frequency_RMSE_cm-1": nbs2_response["metrics_vs_full_EPC"][
                "blind_rank4_Chebyshev"
            ]["frequency_RMSE_cm-1"],
            "frequency_max_abs_cm-1": nbs2_response["metrics_vs_full_EPC"][
                "blind_rank4_Chebyshev"
            ]["frequency_max_abs_cm-1"],
            "shape_relative_error": nbs2_response["metrics_vs_full_EPC"][
                "blind_rank4_Chebyshev"
            ]["maximum_window_depth_relative_error"],
            "passed_fixed_repaired_model": nbs2_response[
                "status"
            ] == "low_rank_response_comparison_passed",
        },
    ]
    independent = [
        row for row in rows if row["evidence_type"] == "independent blind transfer"
    ]
    fixed_model_pass = all(
        row["passed_fixed_repaired_model"] is True for row in independent
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": (
            "freeze_five_q_rank4_Chebyshev_adapter"
            if fixed_model_pass
            else "do_not_freeze_adapter"
        ),
        "fixed_representation": {
            "labels_per_material": 5,
            "label_type": "complex mode-projected coarse-q EPC vertex",
            "rank_policy": "material-adaptive, capped at 4",
            "q_latent": "degree-4 Chebyshev over the tested local 1D window",
            "smearing_policy": (
                "no separate EPC labels per smearing; use one vertex/Hamiltonian "
                "with explicit Fermi-Dirac occupations in the band sum"
            ),
        },
        "systems": rows,
        "independent_blind_transfer": {
            "systems": [row["material"] for row in independent],
            "passed": sum(row["passed_fixed_repaired_model"] is True for row in independent),
            "tested": len(independent),
            "maximum_vertex_RMSE_relative": max(
                row["vertex_RMSE_relative"] for row in independent
            ),
            "maximum_frequency_RMSE_cm-1": max(
                row["frequency_RMSE_cm-1"] for row in independent
            ),
            "maximum_frequency_error_cm-1": max(
                row["frequency_max_abs_cm-1"] for row in independent
            ),
            "maximum_shape_relative_error": max(
                row["shape_relative_error"] for row in independent
            ),
        },
        "decision": {
            "freeze_adapter_architecture": fixed_model_pass,
            "claim_pretrained_cross_material_generator": False,
            "launch_dense_DFPT": False,
            "next_model_work": (
                "define gauge-invariant inputs for a generator of the material-specific "
                "rank basis/latent prior; keep the five-label adapter as the correction layer"
            ),
        },
        "limitations": [
            "only TaS2 and NbS2 are independent blind transfers of the repaired representation; VSe2 selected the repair and graphene is a within-system q holdout",
            "the five labels currently learn a material-specific basis; a pretrained generator of that basis has not yet been demonstrated",
            "all TMD response tests compare compressed and full stored EPC at the same 24x24 integration mesh, not against an absolutely converged DFT spectrum",
            "the tested Chebyshev latent is one-dimensional; full two-dimensional point-group equivariance away from the K/K' time-reversal pair remains to be implemented",
        ],
    }
    (OUTPUT / "low_data_epc_transfer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    names = [row["material"] for row in rows]
    colors = ["#7F7F7F", "#D55E00", "#2676B8", "#5A9E45"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    x = np.arange(len(rows))
    axes[0].bar(
        x,
        [100.0 * row["vertex_RMSE_relative"] for row in rows],
        color=colors,
    )
    axes[0].axhline(2.0, color="#222222", ls="--", lw=0.9, label="fixed 2% gate")
    axes[0].set_ylabel("held-out EPC vertex RMSE (%)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.26), frameon=False)
    axes[1].bar(
        x,
        [row["frequency_RMSE_cm-1"] for row in rows],
        color=colors,
    )
    axes[1].axhline(0.05, color="#222222", ls="--", lw=0.9, label=r"fixed 0.05 cm$^{-1}$ gate")
    axes[1].set_ylabel(r"frequency RMSE vs full EPC (cm$^{-1}$)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, -0.26), frameon=False)
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xticks(x, names, rotation=18, ha="right")
        axis.grid(axis="y", color="#E6E6E6", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Five-q EPC adapter: development and independent transfer evidence")
    figure.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.32, wspace=0.30)
    figure.savefig(
        OUTPUT / "low_data_epc_transfer_summary.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    return 0 if fixed_model_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
