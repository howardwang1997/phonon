#!/usr/bin/env bash
# Launch N parallel self-DFT-dataset workers (each its own QE workdir) over the
# SG15-covered MDR materials -> diverse-chemistry Line B dataset.
cd ~/phonon
PY=~/miniconda3/envs/dft/bin/python
N=${N:-4}
NPROC=${NPROC:-44}
$PY - "$N" <<'PYEOF'
import sys, re, glob, os, collections
sys.path.insert(0, "src")
from phonon_accel import reference
n = int(sys.argv[1])
have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.expanduser("~/pseudo/*.upf"))}
idx = reference._index()
cached = sorted(os.path.basename(p).split("_")[0] for p in glob.glob("data/benchmark/mdr/*_phonopy_params.yaml"))
def els(mp): return set(re.findall(r"[A-Z][a-z]?", idx.get(mp, {}).get("formula", "")))
cov = [mp for mp in cached if els(mp) and els(mp) <= have]
cov.sort(key=lambda mp: len(idx[mp]["formula"]))
ch = collections.defaultdict(list)
for i, mp in enumerate(cov):
    ch[i % n].append(mp)
for i in range(n):
    open(f"/tmp/chunk{i}.txt", "w").write(" ".join(ch[i]))
print(f"split {len(cov)} materials into {n} chunks")
PYEOF
for i in $(seq 0 $((N-1))); do
  mkdir -p /tmp/ds$i
  nohup $PY scripts/dft_dataset.py --materials $(cat /tmp/chunk$i.txt) --workdir /tmp/ds$i \
    --nproc $NPROC --ecutwfc 55 --ecutrho 220 --kspacing 0.30 --min-length 8 --max-atoms 64 \
    --limit 100 --manifest results/dft/manifest_$i.csv > /root/broad_$i.out 2>&1 &
done
disown -a
echo "launched $N broad-dataset workers (nproc=$NPROC each)"
