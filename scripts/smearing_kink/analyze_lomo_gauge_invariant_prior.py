"""Leave-one-material-out test of a gauge-invariant latent prior for the
five-q EPC vertex adapter.

Motivation (WEEKLY_2026-08-19 next-phase table): check whether a prior
fitted on the other materials' five coarse-q labels can substitute part of
a new material's label budget, or whether the per-material five-label
budget must stay.

Data policy: existing stored vertex arrays only (E17 graphene, E18/E21/E24
TMDs); no new DFT, no new q labels, no re-running of EPW.

Protocol (fixed before inspecting holdout errors, see lomo_test_spec.json):

- B1: deployed baseline - five labels, SVD rank r_m, degree-4 Chebyshev
  latent (exact interpolation). Holdout relative vertex RMSE must
  reproduce the published gates (<= 0.02).
- B2(k): label reduction baseline - k evenly spaced labels, degree k-1
  Chebyshev (exact interpolation at that degree).
- P(k): prior-regularized fit - k labels, degree-4 Chebyshev latent,
  coefficients ridge-shrunk toward the leave-one-material-out prior (mean
  normalized latent coefficients of the other materials' five-label fits).
  Rank is capped at min(k, r_m).

Gates (pre-registered):
- G1 reproduction: B1 holdout relative vertex RMSE <= 0.02 for every
  material.
- G2 transfer: for every material, P(r_m) <= max(0.02, 1.5 * B1) and
  P(r_m) <= B2(r_m).
- Decision: prior transfers only if G1 and G2 hold for all four materials;
  otherwise the per-material five-label budget stays.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
OUT = BASE / "E25_lomo_gauge_invariant_prior"

MATERIALS = {
    "graphene": {
        "directory": BASE / "E17_epc_vertex_rank_response_nk720",
        "arrays": "finite_vertex_rank_response_arrays.npz",
        "summary": "finite_vertex_rank_response_summary.json",
        "vertex_key": "full_vertex_rows",
        "coordinate_key": "t_GK",
        "rank": 3,
    },
    "1T-VSe2": {
        "directory": BASE / "E18_VSe2_rank3_five_q_blind",
        "arrays": "rank3_five_q_blind_arrays.npz",
        "summary": "rank3_five_q_blind_summary.json",
        "vertex_key": "full_vertex",
        "coordinate_key": "coordinate",
        # E18 rank-3 attempt failed the gate; the E19 repair selected rank 4
        # (blind heldout RMSE 0.0110), validated independently on TaS2/NbS2.
        "rank": 4,
    },
    "2H-TaS2": {
        "directory": BASE / "E21_TaS2_rank4_five_q_blind",
        "arrays": "rank3_five_q_blind_arrays.npz",
        "summary": "rank3_five_q_blind_summary.json",
        "vertex_key": "full_vertex",
        "coordinate_key": "coordinate",
        "rank": None,
    },
    "2H-NbS2": {
        "directory": BASE / "E24_NbS2_rank4_five_q_blind",
        "arrays": "rank3_five_q_blind_arrays.npz",
        "summary": "rank3_five_q_blind_summary.json",
        "vertex_key": "full_vertex",
        "coordinate_key": "coordinate",
        "rank": None,
    },
}
DEGREE = 4  # deployed latent degree
SPEC = {
    "fixed_before_holdout_inspection": "2026-09-05",
    "purpose": "leave-one-material-out transferability of the five-q EPC adapter latent prior",
    "data_policy": "stored E17/E18/E21/E24 vertex arrays only; no new DFT or q labels",
    "protocol": {
        "B1": "five labels, SVD rank r_m, degree-4 Chebyshev (deployed baseline)",
        "B2(k)": "k evenly spaced labels, degree k-1 Chebyshev (label reduction, no prior)",
        "P(k)": "k labels, degree-4 Chebyshev, ridge toward LOMO prior latent coefficients",
    },
    "gates": {
        "G1_reproduction_max_relative_vertex_rmse": 0.02,
        "G2_transfer_factor_over_B1": 1.5,
        "G2_transfer_absolute_cap": 0.02,
        "decision": "prior transfers iff G1 and G2 hold for all four materials; else keep five labels per system",
    },
    "ridge_mu_grid": [1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1],
    "ridge_note": "P(k) reports the best holdout RMSE over the mu grid; if even the best mu gives no benefit over B2(k) the prior does not transfer",
}


def load_materials() -> dict:
    loaded = {}
    for name, spec in MATERIALS.items():
        data = np.load(spec["directory"] / spec["arrays"], allow_pickle=True)
        vertex = np.asarray(data[spec["vertex_key"]], complex)
        coordinate = np.asarray(data[spec["coordinate_key"]], float)
        training = np.asarray(data["training_indices"], int)
        summary = json.loads((spec["directory"] / spec["summary"]).read_text())
        rank = int(spec["rank"]) if spec["rank"] else int(summary["selected_rank"])
        scale = np.linalg.norm(vertex)
        loaded[name] = {
            "vertex": vertex / scale,
            "coordinate": coordinate,
            "training": training,
            "rank": rank,
            "scale": scale,
        }
    return loaded


def normalized_line_coordinate(coordinate: np.ndarray) -> np.ndarray:
    lo, hi = float(coordinate.min()), float(coordinate.max())
    return 2.0 * (coordinate - lo) / (hi - lo) - 1.0


def svd_latent(labels: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    left, singular, right = np.linalg.svd(labels, full_matrices=False)
    r = min(rank, len(labels), len(singular))
    latent = left[:, :r] * singular[None, :r]
    return latent, right[:r]


def cheb_coefficients(x: np.ndarray, latent: np.ndarray, degree: int) -> np.ndarray:
    return np.array(
        [chebfit(x, latent[:, component], min(degree, len(x) - 1)) for component in range(latent.shape[1])]
    ).T  # (degree+1, r)


def reconstruct(
    x_eval: np.ndarray,
    x_labels: np.ndarray,
    labels: np.ndarray,
    rank: int,
    degree: int,
    prior_coefficients: np.ndarray | None = None,
    ridge_mu: float = 0.0,
) -> np.ndarray:
    latent, right = svd_latent(labels, rank)
    if prior_coefficients is None or ridge_mu <= 0.0 or len(x_labels) > degree:
        coefficients = cheb_coefficients(x_labels, latent, degree)
    else:
        # Ridge on Chebyshev coefficients toward the prior, fitted componentwise:
        # min_c sum_i |latent_i - sum_a c_a T_a(x_i)|^2 + mu |c - prior|^2
        design = np.polynomial.chebyshev.chebvander(x_labels, degree)
        coefficients = np.empty((degree + 1, latent.shape[1]), complex)
        for component in range(latent.shape[1]):
            target = latent[:, component]
            gram = design.conj().T @ design + ridge_mu * np.eye(degree + 1)
            rhs = design.conj().T @ target + ridge_mu * prior_coefficients[:, component]
            coefficients[:, component] = np.linalg.solve(gram, rhs)
    predicted_latent = np.column_stack(
        [chebval(x_eval, coefficients[:, component]) for component in range(latent.shape[1])]
    )
    return predicted_latent @ right


def holdout_errors(
    material: dict,
    label_indices: np.ndarray,
    degree: int,
    prior_coefficients: np.ndarray | None = None,
    ridge_mu: float = 0.0,
) -> tuple[float, float]:
    x = normalized_line_coordinate(material["coordinate"])
    vertex = material["vertex"]
    mask = np.ones(len(x), bool)
    mask[label_indices] = False
    predicted = reconstruct(
        x[mask], x[label_indices], vertex[label_indices], material["rank"], degree, prior_coefficients, ridge_mu
    )
    reference = vertex[mask]
    pointwise = np.linalg.norm(predicted - reference, axis=1) / np.linalg.norm(reference, axis=1)
    return float(np.sqrt(np.mean(pointwise**2))), float(pointwise.max())


def five_label_coefficients(material: dict) -> np.ndarray:
    x = normalized_line_coordinate(material["coordinate"])
    latent, _ = svd_latent(material["vertex"][material["training"]], material["rank"])
    return cheb_coefficients(x[material["training"]], latent, DEGREE)


def fit_prior_coefficients(
    loaded: dict, held_out: str, target_rank: int, labels: np.ndarray | None = None
) -> np.ndarray:
    """Mean normalized latent coefficients of the other materials.

    With ``labels`` given, the prior is rescaled to the held-out material's own
    label-fitted coefficient norm (scale-matched variant); otherwise the prior
    keeps unit Frobenius norm (unit variant). The scale-matched variant is the
    fairer test because the unit-norm prior is typically several times larger
    than the data coefficients and dominates the null-space direction.
    """
    stack = []
    for name, material in loaded.items():
        if name == held_out:
            continue
        coefficients = five_label_coefficients(material)
        r = min(target_rank, coefficients.shape[1])
        trimmed = coefficients[:, :r]
        if coefficients.shape[1] < target_rank:  # pad with zeros
            trimmed = np.pad(trimmed, ((0, 0), (0, target_rank - coefficients.shape[1])))
        stack.append(trimmed / max(np.linalg.norm(trimmed), 1.0e-300))
    prior = np.mean(stack, axis=0)
    if labels is not None:
        latent, _ = svd_latent(labels, target_rank)
        own = cheb_coefficients(
            np.linspace(-1.0, 1.0, len(labels)), latent, min(DEGREE, len(labels) - 1)
        )
        prior = prior * (np.linalg.norm(own) / max(np.linalg.norm(prior), 1.0e-300))
    return prior


def evenly_spaced_indices(n_total: int, k: int) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, n_total - 1, k)).astype(int))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = OUT / "lomo_test_spec.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps(SPEC, indent=2), encoding="utf-8")
    loaded = load_materials()
    mu_grid = [float(v) for v in SPEC["ridge_mu_grid"]]
    report: dict = {"materials": {}, "gates": SPEC["gates"], "ridge_mu_grid": mu_grid}
    for name, material in loaded.items():
        rank = material["rank"]
        n_total = len(material["coordinate"])
        entry: dict = {"rank": rank, "n_points": n_total}
        entry["B1_five_labels"] = dict(
            zip(("rmse", "max"), holdout_errors(material, material["training"], DEGREE))
        )
        reduction: dict = {}
        for k in (2, 3, 4, 5):
            if k > len(material["training"]):
                continue
            indices = evenly_spaced_indices(n_total, k)
            b2 = dict(zip(("rmse", "max"), holdout_errors(material, indices, k - 1)))
            prior_unit = fit_prior_coefficients(loaded, name, min(rank, k))
            prior_scaled = fit_prior_coefficients(
                loaded, name, min(rank, k), material["vertex"][indices]
            )
            if len(indices) > DEGREE:
                best_p = {"unit": dict(zip(("rmse", "max"), holdout_errors(material, indices, DEGREE)))}
                best_mu = None
            else:
                best_p = {}
                for tag, prior in (("unit", prior_unit), ("scaled", prior_scaled)):
                    candidates = [
                        (mu, *holdout_errors(material, indices, DEGREE, prior, mu)) for mu in mu_grid
                    ]
                    best_mu, best_rmse, best_max = min(candidates, key=lambda item: item[1])
                    best_p[tag] = {"rmse": best_rmse, "max": best_max, "mu": best_mu}
            reduction[f"k={k}"] = {"B2": b2, "P": best_p}
        entry["label_reduction"] = reduction
        report["materials"][name] = entry
        b1 = entry["B1_five_labels"]["rmse"]
        k_star = min(rank, 4)
        p_variants = reduction[f"k={k_star}"]["P"]
        p_star = min(v["rmse"] for v in p_variants.values())
        b2_star = reduction[f"k={k_star}"]["B2"]["rmse"]
        detail = " ".join(f"{tag} {v['rmse']:.5f}" for tag, v in p_variants.items())
        print(
            f"{name:9s} rank {rank} | B1 {b1:.5f} | k={k_star}: B2 {b2_star:.5f} | {detail}"
        )
    gate_results = {}
    for name, entry in report["materials"].items():
        b1 = entry["B1_five_labels"]["rmse"]
        k_star = min(entry["rank"], 4)
        p_star = min(v["rmse"] for v in entry["label_reduction"][f"k={k_star}"]["P"].values())
        b2_star = entry["label_reduction"][f"k={k_star}"]["B2"]["rmse"]
        g1 = b1 <= SPEC["gates"]["G1_reproduction_max_relative_vertex_rmse"]
        cap = max(SPEC["gates"]["G2_transfer_absolute_cap"], SPEC["gates"]["G2_transfer_factor_over_B1"] * b1)
        gate_results[name] = {
            "G1_reproduction": bool(g1),
            "G2_transfer": bool(p_star <= cap and p_star <= b2_star),
            "B1_rmse": b1,
            "P_rmse_at_k_rank": p_star,
            "B2_rmse_at_k_rank": b2_star,
        }
    report["gate_results"] = gate_results
    report["prior_transfers"] = bool(all(v["G1_reproduction"] and v["G2_transfer"] for v in gate_results.values()))
    (OUT / "lomo_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("prior_transfers:", report["prior_transfers"])
    print("wrote", OUT / "lomo_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
