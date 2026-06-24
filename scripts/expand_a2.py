"""A2 expansion: select chemically diverse, high-symmetry, small-cell binaries/
ternaries from the MDR index that are NOT already cached (hence not in the B79
training pool), fetch their DFPT references, and print the fetched mp-id list for
a larger held-out transfer test (Limitation vi). Stability is filtered post-hoc
after eval (n_imaginary_ref). Run locally (needs internet to mdr.nims.go.jp).

    python scripts/expand_a2.py --n 35 --out-list /tmp/a2_new.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phonon_accel import reference

CACHE = ROOT / "data" / "benchmark" / "mdr"


def parse(formula):
    toks = [(a, int(b or 1)) for a, b in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if a]
    return {a for a, _ in toks}, sum(n for _, n in toks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--min-sg", type=int, default=150, help="higher symmetry -> likelier stable")
    ap.add_argument("--max-atoms", type=int, default=12)
    ap.add_argument("--spacegroups", default="", help="comma list; if set, restrict to exactly these sg numbers")
    ap.add_argument("--out-list", default="/tmp/a2_new.txt")
    args = ap.parse_args()
    only_sg = {int(x) for x in args.spacegroups.split(",") if x.strip()}

    idx = reference._index()
    cached = {p.name.split("_")[0] for p in CACHE.glob("*_phonopy_params.yaml")}
    cand = []
    for mp, row in idx.items():
        if mp in cached:
            continue
        el, nat = parse(row["formula"])
        sg = int(row["spacegroup_number"] or 0)
        if not (2 <= len(el) <= 3):
            continue
        if nat > args.max_atoms:
            continue
        if only_sg and sg not in only_sg:
            continue
        if not only_sg and sg < args.min_sg:
            continue
        if el <= {"Si", "O"}:  # skip silica polymorphs
            continue
        cand.append((mp, row["formula"], frozenset(el), sg, nat))

    # greedy: one material per distinct element-set, prefer small cells + high symmetry
    seen, picked = set(), []
    for mp, fm, el, sg, nat in sorted(cand, key=lambda x: (x[4], -x[3])):
        if el in seen:
            continue
        seen.add(el); picked.append((mp, fm, sg, nat))
        if len(picked) >= args.n:
            break

    print(f"candidates: {len(cand)} | selected {len(picked)} distinct-chemistry materials")
    fetched = []
    for mp, fm, sg, nat in picked:
        try:
            reference.fetch(mp); fetched.append(mp)
            print(f"  ok  {mp:12s} {fm:10s} sg{sg} ({nat} atoms)")
        except Exception as e:
            print(f"  FAIL {mp:12s} {fm:10s}: {str(e)[:50]}")
    Path(args.out_list).write_text(" ".join(fetched))
    print(f"\nFETCHED {len(fetched)} -> {args.out_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
