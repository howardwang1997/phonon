#!/usr/bin/env python3
"""Run the R2AI protocol for the evidence-motivated 5--7 A FC3 range.

R2AK showed monotonic seed2 improvement as a conservative K-star cubic bond
field was extended beyond the previous 4 A hiphive limit.  This wrapper keeps
the R2AI folds, objectives, ridge grid, metrics, and data boundary unchanged;
it changes only the symmetry-constrained third-order cluster cutoff.
"""

from __future__ import annotations

import evaluate_graphene_r2ai_seed012_hiphive_anharmonic as base


base.DEFAULT_OUTPUT = (
    base.BASE / "R2AN_seed012_hiphive_longrange_fc3_20260827"
)
base.REPRESENTATIONS = (
    {"name": "fc3_r5p0", "cutoffs_A": (5.0, 5.0), "orders": (3,)},
    {"name": "fc3_r6p0", "cutoffs_A": (6.0, 6.0), "orders": (3,)},
    {"name": "fc3_r7p0", "cutoffs_A": (7.0, 7.0), "orders": (3,)},
)


if __name__ == "__main__":
    base.main()
