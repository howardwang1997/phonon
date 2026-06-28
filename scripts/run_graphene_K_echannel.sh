#!/usr/bin/env bash
# E7: graphene (E)-channel at the K point. DFPT on a 3x3 q-grid (includes Gamma AND
# K=(1/3,1/3)) vs Fermi-Dirac smearing (degauss = k_B*T_el). Extends #3 (Gamma-E2g
# vs smearing) to the K-A1' Kohn anomaly. conda qe ph.x (CPU). Run in tmux (pty).
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate qe 2>/dev/null || conda activate phonon
WORK="${WORK:-/data/graphene_K_ech}"; PSEUDO="${PSEUDO:-/root/phonon/pseudo}"
NP="${NP:-8}"; MPI="mpirun --allow-run-as-root -np $NP"
A=4.6488; COA=6.0
mkdir -p "$WORK"; cd "$WORK"; export OMP_NUM_THREADS=1
PY=/root/miniconda3/envs/phonon/bin/python
echo "degauss_Ry,T_el_K,qlabel,qx,qy,freqs_cm" > result.csv

for DG in 0.003 0.006 0.010 0.020 0.040; do
  TEL=$(awk "BEGIN{printf \"%.0f\", $DG*157887}")
  cat > scf.in <<EOF
&control
 calculation='scf', prefix='gr', outdir='./tmp', pseudo_dir='$PSEUDO'
/
&system
 ibrav=4, celldm(1)=$A, celldm(3)=$COA, nat=2, ntyp=1, ecutwfc=60, ecutrho=240
 occupations='smearing', smearing='fd', degauss=$DG
/
&electrons
 conv_thr=1.0d-11, mixing_beta=0.7
/
ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.0 0.0 0.0
 C 0.333333333 0.666666667 0.0
K_POINTS automatic
 24 24 1 0 0 0
EOF
  $MPI pw.x -in scf.in > scf.out 2>&1
  cat > ph.in <<EOF
graphene DFPT 3x3 (Gamma + K)
&inputph
 prefix='gr', outdir='./tmp', fildyn='gr.dyn', ldisp=.true., nq1=3, nq2=3, nq3=1
 tr2_ph=1.0d-15
/
EOF
  rm -f gr.dyn*; $MPI ph.x -in ph.in > ph.out 2>&1
  echo "[K-ech] degauss=$DG (T_el=$TEL K): parsing dyn files ..."
  # parse each dyn file: q-point + frequencies (cm-1)
  $PY - "$DG" "$TEL" <<'PY'
import sys, glob, re
dg, tel = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob("gr.dyn[1-9]*")):
    txt = open(f).read()
    m = re.search(r"q\s*=\s*\(\s*([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)", txt)
    if not m: continue
    qx, qy = float(m.group(1)), float(m.group(2))
    fr = [float(x) for x in re.findall(r"freq.*?=\s*([-0-9.]+)\s*\[cm-1\]", txt)]
    if not fr: fr = [float(x) for x in re.findall(r"([-0-9.]+)\s*\[cm-1\]", txt)]
    lab = "Gamma" if abs(qx)<1e-4 and abs(qy)<1e-4 else ("K" if abs(qx)>0.3 else "M/other")
    frs = " ".join(f"{x:.1f}" for x in fr)
    print(f"{dg},{tel},{lab},{qx:.4f},{qy:.4f},{frs}")
    open("result.csv","a").write(f"{dg},{tel},{lab},{qx:.4f},{qy:.4f},{frs}\n")
PY
done
echo "GRAPHENE_K_ECH_DONE"; echo "=== result.csv ==="; cat result.csv
