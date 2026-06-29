#!/usr/bin/env bash
# Doped-graphene EPW: undoped graphene gave lambda~0 (E_F at the Dirac point, DOS->0).
# Doping (tot_charge<0 => n-type) shifts E_F into the pi* band -> a real Fermi surface
# (circle radius k_F around K) -> FINITE lambda + the 2k_F Kohn-anomaly linewidths.
# Reuses the debugged graphene EPW pipeline (c/a=12 b-vectors, tmux/pty). conda CPU QE.
set -e
source /root/miniconda3/etc/profile.d/conda.sh && { conda activate qe 2>/dev/null || conda activate phonon; }
WORK="${WORK:-/data/graphene_doped_epw}"; PSEUDO="${PSEUDO:-/root/phonon/pseudo}"
NP="${NP:-8}"; MPI="mpirun --allow-run-as-root -np $NP"
A=4.6488; COA=12.0; NK=12; NQ=6; NKF="${NKF:-36}"; NQF="${NQF:-36}"
TOT="${TOT_CHARGE:--0.04}"        # -0.04 e/cell n-doping -> E_F ~0.8 eV into pi*
mkdir -p "$WORK"; cd "$WORK"; export OMP_NUM_THREADS=1
SYS="ibrav=4, celldm(1)=$A, celldm(3)=$COA, nat=2, ntyp=1, ecutwfc=60, ecutrho=240
 occupations='smearing', smearing='fd', degauss=0.02, tot_charge=$TOT"
ATOMS="ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.0 0.0 0.0
 C 0.333333333 0.666666667 0.0"

cat > scf.in <<EOF
&control
 calculation='scf', prefix='graphene', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'
/
&system
 $SYS
/
&electrons
 conv_thr=1.0d-12, mixing_beta=0.7
/
$ATOMS
K_POINTS automatic
 $NK $NK 1 0 0 0
EOF
echo "[doped] scf tot_charge=$TOT ..."; grep -q "JOB DONE" scf.out 2>/dev/null || $MPI pw.x -in scf.in > scf.out 2>&1
grep -q "JOB DONE" scf.out || { echo "SCF FAIL"; tail -5 scf.out; exit 1; }
EF=$(grep -i 'the Fermi energy' scf.out | tail -1 | grep -oE '[-0-9.]+' | head -1)
echo "  SCF ok, doped E_F = $EF eV"

cat > ph.in <<EOF
doped graphene DFPT
&inputph
 prefix='graphene', outdir='./tmp', fildyn='graphene.dyn', fildvscf='dvscf'
 ldisp=.true., nq1=$NQ, nq2=$NQ, nq3=1, tr2_ph=1.0d-14, electron_phonon='dvscf'
/
EOF
echo "[doped] ph DFPT ${NQ}x${NQ} (metallic now) ..."; grep -q "JOB DONE" ph.out 2>/dev/null || $MPI ph.x -in ph.in > ph.out 2>&1
grep -q "JOB DONE" ph.out || { echo "PH FAIL"; tail -6 ph.out; exit 1; }; echo "  PH ok"

echo "[doped] gather dvscf -> save/ ..."
rm -rf save; mkdir -p save; nq=$(sed -n '2p' graphene.dyn0 | tr -dc 0-9)
cp -r tmp/_ph0/graphene.phsave save/
for iq in $(seq 1 "$nq"); do
  cp "graphene.dyn${iq}" "save/graphene.dyn_q${iq}"
  if [ "$iq" -eq 1 ]; then src=$(ls tmp/_ph0/graphene.dvscf${iq}_* | head -1)
  else src=$(ls tmp/_ph0/graphene.q_${iq}/graphene.dvscf${iq}_* | head -1); fi
  cp "$src" "save/graphene.dvscf_q${iq}"
done
echo "  gathered $nq q-points"

echo "[doped] nscf full ${NK}x${NK} k ..."
awk -v nk="$NK" 'BEGIN{print nk*nk; for(i=0;i<nk;i++)for(j=0;j<nk;j++)printf "%.10f %.10f 0.0 1.0\n", i/nk, j/nk}' > kpts.txt
{ cat <<EOF
&control
 calculation='nscf', prefix='graphene', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'
/
&system
 $SYS
 nbnd=20
/
&electrons
 conv_thr=1.0d-12, diago_full_acc=.true.
/
$ATOMS
K_POINTS crystal
EOF
  cat kpts.txt; } > nscf.in
grep -q "JOB DONE" nscf.out 2>/dev/null || $MPI pw.x -in nscf.in > nscf.out 2>&1; grep -q "JOB DONE" nscf.out && echo "  NSCF ok"

cat > epw.in <<EOF
--
&inputepw
 prefix='graphene', outdir='./tmp'
 elph=.true., epbwrite=.true., epwwrite=.true.
 wannierize=.true., nbndsub=2, num_iter=400, proj(1)='C:pz'
 dis_win_max=4.0
 phonselfen=.true., a2f=.true., elecselfen=.false.
 efermi_read=.true., fermi_energy=$EF
 fsthick=6.0, degaussw=0.05, nsmear=1, delta_smear=0.01
 dvscf_dir='./save'
 nk1=$NK, nk2=$NK, nk3=1, nq1=$NQ, nq2=$NQ, nq3=1
 nkf1=$NKF, nkf2=$NKF, nkf3=1, nqf1=$NQF, nqf2=$NQF, nqf3=1
/
EOF
echo "[doped] epw.x (pty/tmux) ..."; mpirun --allow-run-as-root -np 1 epw.x -in epw.in > epw.out 2>&1
echo "  epw exit $?"
grep -aiE "Phonon linewidth|lambda :|a2f|electron-phonon coupling|EPW.*WALL" epw.out | tail -8
echo "GRAPHENE_DOPED_EPW_DONE (E_F=$EF eV, tot_charge=$TOT)"
