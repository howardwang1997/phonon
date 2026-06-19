"""Expand the local MDR DFPT cache to a chemically/structurally diverse pool
for the breadth sweep. Stratifies the 10,034-material index by spacegroup and
leading element, samples candidates, downloads each (skips failures), and prints
the final list of successfully-cached mp-ids.

Run on a machine that can reach NIMS (the Mac), then relay data/benchmark/mdr/.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phonon_accel import reference  # noqa: E402

TARGET_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 55

# deterministic "diverse" ordering: round-robin across spacegroup buckets,
# preferring materials whose leading element is under-represented so far.
rows = list(reference._index().values())
already = {p.name.split("_")[0] for p in reference.CACHE_DIR.glob("*_phonopy_params.yaml")}
print(f"already cached: {len(already)}")

by_sg = defaultdict(list)
for r in rows:
    if r["mp_id"] in already:
        continue
    by_sg[r["spacegroup_number"]].append(r)
# sort each bucket by formula length (prefer simpler/smaller cells first)
for sg in by_sg:
    by_sg[sg].sort(key=lambda r: (len(r["formula"]), r["formula"]))

# interleave buckets (round-robin) for spacegroup diversity
order = []
sgs = sorted(by_sg, key=lambda s: int(s))
i = 0
while any(by_sg[s] for s in sgs):
    s = sgs[i % len(sgs)]
    if by_sg[s]:
        order.append(by_sg[s].pop(0))
    i += 1

ok, fail = [], []
for r in order:
    if len(ok) >= TARGET_NEW:
        break
    mp = r["mp_id"]
    try:
        reference.fetch(mp)
        ok.append(mp)
        print(f"  OK  {mp:12s} {r['formula']:14s} sg={r['spacegroup_number']}  ({len(ok)}/{TARGET_NEW})", flush=True)
    except Exception as e:  # noqa: BLE001
        fail.append(mp)
        if len(fail) <= 20:
            print(f"  FAIL {mp:12s} {str(e)[:50]}", flush=True)

allids = sorted(already | set(ok))
print(f"\n=== done: +{len(ok)} new, {len(fail)} failed, pool now {len(allids)} ===")
print("POOL=" + " ".join(allids))
