#!/usr/bin/env bash
# Graphene EPW end-to-end (scf -> ph DFPT -> gather -> nscf -> epw), conda CPU QE.
# c/a=3.0 (not 6.0): with large vacuum the out-of-plane k-vector |b3| is ~degenerate
# with graphene's in-plane 2nd shell, breaking wannier90's b-vector shell search
# (no kmesh_tol works: loose->"too many neighbours", tight->"not enough bvectors").
# A less anisotropic cell makes the shell search well-conditioned.
set -e
WORK="${WORK:-/data/graphene_epw3}"
PSEUDO="${PSEUDO:-/root/phonon/pseudo}"
NP="${NP:-8}"; MPI="mpirun --allow-run-as-root -np $NP"
A=4.6488; COA=3.0; NK=12; NQ=6; NKF=60; NQF=60
mkdir -p "$WORK"; cd "$WORK"; export OMP_NUM_THREADS=1

cat > scf.in <<EOF
&control
 calculation='scf', prefix='graphene', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'
/
&system
 ibrav=4, celldm(1)=$A, celldm(3)=$COA, nat=2, ntyp=1, ecutwfc=60, ecutrho=240
 occupations='smearing', smearing='fd', degauss=0.02
/
&electrons
 conv_thr=1.0d-12, mixing_beta=0.7
/
ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.0 0.0 0.0
 C 0.333333333 0.666666667 0.0
K_POINTS automatic
 $NK $NK 1 0 0 0
EOF
echo "[full] scf ..."; $MPI pw.x -in scf.in > scf.out 2>&1; grep -q "JOB DONE" scf.out && echo "  SCF ok Ef=$(grep -i 'the Fermi energy' scf.out | tail -1 | grep -oE '[-0-9.]+ ev')"

cat > ph.in <<EOF
graphene DFPT
&inputph
 prefix='graphene', outdir='./tmp', fildyn='graphene.dyn', fildvscf='dvscf'
 ldisp=.true., nq1=$NQ, nq2=$NQ, nq3=1, tr2_ph=1.0d-14, electron_phonon='dvscf'
/
EOF
echo "[full] ph DFPT ${NQ}x${NQ} ..."; $MPI ph.x -in ph.in > ph.out 2>&1; grep -q "JOB DONE" ph.out && echo "  PH ok"

echo "[full] gather dvscf -> save/ (QE7.5 naming) ..."
rm -rf save; mkdir -p save; nq=$(sed -n '2p' graphene.dyn0 | tr -dc 0-9)
cp -r tmp/_ph0/graphene.phsave save/
for iq in $(seq 1 "$nq"); do
  cp "graphene.dyn${iq}" "save/graphene.dyn_q${iq}"
  if [ "$iq" -eq 1 ]; then src=$(ls tmp/_ph0/graphene.dvscf${iq}_* | head -1)
  else src=$(ls tmp/_ph0/graphene.q_${iq}/graphene.dvscf${iq}_* | head -1); fi
  cp "$src" "save/graphene.dvscf_q${iq}"
done
echo "  gathered $nq q-points"

echo "[full] nscf full ${NK}x${NK} k ..."
python - "$NK" > kpts.txt <<'PY'
import sys
nk=int(sys.argv[1]); pts=[(i/nk,j/nk,0.0) for i in range(nk) for j in range(nk)]
print(len(pts))
for x,y,z in pts: print(f"{x:.10f} {y:.10f} {z:.10f} 1.0")
PY
{ cat <<EOF
&control
 calculation='nscf', prefix='graphene', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'
/
&system
 ibrav=4, celldm(1)=$A, celldm(3)=$COA, nat=2, ntyp=1, ecutwfc=60, ecutrho=240, nbnd=20
 occupations='smearing', smearing='fd', degauss=0.02
/
&electrons
 conv_thr=1.0d-12, diago_full_acc=.true.
/
ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.0 0.0 0.0
 C 0.333333333 0.666666667 0.0
K_POINTS crystal
EOF
  cat kpts.txt; } > nscf.in
$MPI pw.x -in nscf.in > nscf.out 2>&1; grep -q "JOB DONE" nscf.out && echo "  NSCF ok"

cat > epw.in <<EOF
--
&inputepw
 prefix='graphene', outdir='./tmp'
 elph=.true., epbwrite=.true., epwwrite=.true.
 wannierize=.true., nbndsub=2, num_iter=400, proj(1)='C:pz'
 dis_win_max=4.0, dis_froz_max=0.0
 phonselfen=.true., a2f=.true., elecselfen=.false.
 fsthick=6.0, degaussw=0.05, nsmear=1, delta_smear=0.01
 dvscf_dir='./save'
 nk1=$NK, nk2=$NK, nk3=1, nq1=$NQ, nq2=$NQ, nq3=1
 nkf1=$NKF, nkf2=$NKF, nkf3=1, nqf1=$NQF, nqf2=$NQF, nqf3=1
/
EOF
echo "[full] epw.x ..."; mpirun --allow-run-as-root -np 1 epw.x -in epw.in > epw.out 2>&1
echo "  epw exit $?"
grep -aiE "Phonon linewidth|lambda|a2f|electron-phonon coupling|EPW.*WALL" epw.out | tail -8
echo "GRAPHENE_EPW_FULL_DONE"
