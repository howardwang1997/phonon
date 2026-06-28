#!/usr/bin/env bash
# Graphene EPW: alpha2F(w), lambda, mode linewidths gamma_qv (Kohn anomalies at
# Gamma-E2g and K-A1'). Runs on conda CPU QE (graphene is 2 atoms) -- parallel-safe
# with a GPU job on the same box.
#
#   bash run_graphene_epw.sh prep   # scf -> ph DFPT(6x6) -> gather dvscf  [the long pole]
#   bash run_graphene_epw.sh epw    # nscf(full 12x12 k) -> epw.x          [after E_F known]
#
# Env: conda 'phonon' (pw.x/ph.x/epw.x/wannier90.x/pw2wannier90.x/epw_pp.py).
set -e
STAGE="${1:-prep}"
WORK="${WORK:-/data/graphene_epw}"
PSEUDO="${PSEUDO:-/root/phonon/pseudo}"
NP="${NP:-8}"
MPI="mpirun --allow-run-as-root -np $NP"
A_BOHR=4.6488          # graphene a=2.46 A in bohr
COA=6.0                # c/a (vacuum)
NK=12; NQ=6            # coarse k / q grids
NKF=48; NQF=48         # fine k / q grids
mkdir -p "$WORK"; cd "$WORK"
export OMP_NUM_THREADS=1

write_scf () {
cat > graphene.scf.in <<EOF
&control
  calculation='scf', prefix='graphene', outdir='./tmp', pseudo_dir='$PSEUDO'
  verbosity='high', tprnfor=.true.
/
&system
  ibrav=4, celldm(1)=$A_BOHR, celldm(3)=$COA, nat=2, ntyp=1
  ecutwfc=60, ecutrho=240
  occupations='smearing', smearing='fd', degauss=0.02
/
&electrons
  conv_thr=1.0d-12, mixing_beta=0.7
/
ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.000000000 0.000000000 0.000000000
 C 0.333333333 0.666666667 0.000000000
K_POINTS automatic
 $NK $NK 1 0 0 0
EOF
}

write_ph () {
cat > graphene.ph.in <<EOF
graphene phonons (DFPT, coarse q for EPW)
&inputph
  prefix='graphene', outdir='./tmp'
  fildyn='graphene.dyn', fildvscf='dvscf'
  ldisp=.true., nq1=$NQ, nq2=$NQ, nq3=1
  tr2_ph=1.0d-14
  electron_phonon='dvscf'
/
EOF
}

if [ "$STAGE" = prep ]; then
  echo "[epw-prep] scf ..."; write_scf
  $MPI pw.x -in graphene.scf.in > graphene.scf.out 2>&1
  grep -q "JOB DONE" graphene.scf.out && echo "[epw-prep] SCF_DONE Ef=$(grep -i 'Fermi energy' graphene.scf.out | tail -1)"
  echo "[epw-prep] ph.x DFPT ${NQ}x${NQ}x1 (the long pole) ..."; write_ph
  $MPI ph.x -in graphene.ph.in > graphene.ph.out 2>&1
  grep -q "JOB DONE" graphene.ph.out && echo "[epw-prep] PH_DONE"
  echo "[epw-prep] gather dvscf -> save/ ..."
  python "$(command -v epw_pp.py)" graphene > epw_pp.out 2>&1 || epw_pp.py graphene > epw_pp.out 2>&1
  ls save/ | head; echo "[epw-prep] PREP_COMPLETE"
fi

if [ "$STAGE" = epw ]; then
  echo "[epw] nscf full ${NK}x${NK}x1 k-grid ..."
  # full uniform k-list (crystal, weight 1) via python
  python - "$NK" > kpts.txt <<'PY'
import sys
nk=int(sys.argv[1]); pts=[(i/nk,j/nk,0.0) for i in range(nk) for j in range(nk)]
print(len(pts))
for x,y,z in pts: print(f"{x:.10f} {y:.10f} {z:.10f} 1.0")
PY
  { cat <<EOF
&control
  calculation='nscf', prefix='graphene', outdir='./tmp', pseudo_dir='$PSEUDO'
  verbosity='high'
/
&system
  ibrav=4, celldm(1)=$A_BOHR, celldm(3)=$COA, nat=2, ntyp=1
  ecutwfc=60, ecutrho=240, nbnd=20
  occupations='smearing', smearing='fd', degauss=0.02
/
&electrons
  conv_thr=1.0d-12, mixing_beta=0.7, diago_full_acc=.true.
/
ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.000000000 0.000000000 0.000000000
 C 0.333333333 0.666666667 0.000000000
K_POINTS crystal
EOF
    cat kpts.txt; } > graphene.nscf.in
  $MPI pw.x -in graphene.nscf.in > graphene.nscf.out 2>&1
  grep -q "JOB DONE" graphene.nscf.out && echo "[epw] NSCF_DONE"
  echo "[epw] epw.x (uses epw.in -- finalize windows from Ef first) ..."
  $MPI epw.x -in graphene.epw.in > graphene.epw.out 2>&1
  echo "[epw] EPW_DONE (check graphene.epw.out for lambda / a2f / linewidths)"
fi
