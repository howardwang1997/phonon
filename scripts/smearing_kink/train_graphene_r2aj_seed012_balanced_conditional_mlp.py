#!/usr/bin/env python3
"""Run the R2AG conditional MLP with one fixed, gate-aligned objective.

R2AG inherited the best linear-readout objective, which assigned only 10% of
the A-prime loss to promoted seed2 even though every trajectory has the same
15 meV/A acceptance threshold.  This controlled follow-up changes only the
loss masses: equal A-prime mass for seed0/1/2, 85% total A-prime loss, and a
70% T600 share within the remaining ordinary-force loss.  Architecture,
features, frozen linear skip, optimizer, folds, seeds, and snapshots remain
the R2AG implementation.
"""

from __future__ import annotations

import copy

import train_graphene_r2ag_seed012_conditional_mlp as base


FORMAT = "graphene_r2aj_seed012_balanced_conditional_mlp_v1"
ORIGINAL_LOAD_PACKAGE = base.load_package


def balanced_load_package(root):
    data, receipt = ORIGINAL_LOAD_PACKAGE(root)
    receipt = copy.deepcopy(receipt)
    receipt["fixed_linear_skip_hyperparameters"] = {
        "projection_mass": 0.85,
        "seed_projection_weights": {
            "seed0": 1.0 / 3.0,
            "seed1": 1.0 / 3.0,
            "seed2": 1.0 / 3.0,
        },
        "group_masses": {
            "E50_seed0": 0.075,
            "E50_seed1": 0.075,
            "E50_seed2": 0.075,
            "T300": 0.075,
            "T600": 0.70,
        },
        "source": "R2AJ fixed gate-aligned objective; frozen linear skip unchanged",
    }
    return data, receipt


def main() -> None:
    base.FORMAT = FORMAT
    base.load_package = balanced_load_package
    base.main()


if __name__ == "__main__":
    main()
