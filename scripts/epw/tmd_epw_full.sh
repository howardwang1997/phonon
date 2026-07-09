#!/usr/bin/env bash
# Generalized TMD EPW (scf -> DFPT NQxNQ -> nscf -> Wannier M:d+X:p -> epw) for any
# 2H/1T monolayer TMD. NbSe2-specific values of nbse2_epw_full.sh factored into env knobs.
# Family (E)-arbiter: lambda(T_el) per material at incommensurate NQ (no soft-mode divergence).
#
# Required env: MAT (prefix), M_ELM, X_ELM, M_MASS, X_MASS, POLYTYPE (2H|1T),
#   ABOHR (celldm1), COA (celldm3=c/a), DZ (X internal z), NP.
# Optional env: DEGAUSS, NQ, NK, NKF, NQF, VACUUM_ANG, NBND, NBNDSUB, WORK, RESULT_CSV, PSEUDO.
set -u
source /root/miniconda3/etc/profile.d/conda.sh && { conda activate qe 2>/dev/null || conda activate phonon; }
DEGAUSS="${DEGAUSS:-0.02}"; NQ="${NQ:-4}"; NK="${NK:-12}"; NKF="${NKF:-24}"; NQF="${NQF:-$NKF}"
MAT="${MAT:?set MAT}"; M_ELM="${M_ELM:?set M_ELM}"; X_ELM="${X_ELM:?set X_ELM}"
M_MASS="${M_MASS:?set M_MASS}"; X_MASS="${X_MASS:?set X_MASS}"; POLYTYPE="${POLYTYPE:?set POLYTYPE}"
ABOHR="${ABOHR:?set ABOHR}"; COA="${COA:?set COA}"; DZ="${DZ:?set DZ}"
NBND="${NBND:-30}"; NBNDSUB="${NBNDSUB:-11}"
WORK="${WORK:-/data/${MAT}_d${DEGAUSS}_q${NQ}}"
PSEUDO="${PSEUDO:-/root/phonon/pseudo}"
RESULT_CSV="${RESULT_CSV:-/root/tmd_lambda_results.csv}"
NP="${NP:-8}"; MPI="mpirun --allow-run-as-root -np $NP"; export OMP_NUM_THREADS=2
M_PSEUDO="${M_ELM}_ONCV_PBE-1.2.upf"; X_PSEUDO="${X_ELM}_ONCV_PBE-1.2.upf"
TEL=$(awk "BEGIN{printf \"%.0f\", $DEGAUSS*157887}")
mkdir -p "$WORK"; cd "$WORK"
echo "===== ${MAT} (${POLYTYPE}-${M_ELM}${X_ELM}) EPW degauss=$DEGAUSS (T_el=$TEL K) NQ=$NQ NKF=$NKF ====="

SYS="ibrav=4, celldm(1)=$ABOHR, celldm(3)=$COA, nat=3, ntyp=2, ecutwfc=70, ecutrho=280
 occupations='smearing', smearing='cold', degauss=$DEGAUSS"
# 2H: M@(0,0,1/2), X@(1/3,2/3,1/2+-dz).  1T: M@(0,0,0), X@(2/3,1/3,+-dz).
if [ "$POLYTYPE" = "1T" ]; then
  XP=$(awk "BEGIN{print 2.0/3.0}")
  ATOMS="ATOMIC_SPECIES
 ${M_ELM} ${M_MASS} ${M_PSEUDO}
 ${X_ELM} ${X_MASS} ${X_PSEUDO}
ATOMIC_POSITIONS (crystal)
 ${M_ELM} 0.000000000 0.000000000 0.000000000
 ${X_ELM} ${XP} ${XP} $(awk "BEGIN{print $DZ}")
 ${X_ELM} ${XP} ${XP} $(awk "BEGIN{print -$DZ}")"
else
  XP=$(awk "BEGIN{print 1.0/3.0}"); YP=$(awk "BEGIN{print 2.0/3.0}"); H=$(awk "BEGIN{print 0.5}")
  ATOMS="ATOMIC_SPECIES
 ${M_ELM} ${M_MASS} ${M_PSEUDO}
 ${X_ELM} ${X_MASS} ${X_PSEUDO}
ATOMIC_POSITIONS (crystal)
 ${M_ELM} 0.000000000 0.000000000 ${H}
 ${X_ELM} ${XP} ${YP} $(awk "BEGIN{print 0.5+$DZ}")
 ${X_ELM} ${XP} ${YP} $(awk "BEGIN{print 0.5-$DZ}")"
fi

cat > scf.in <<EOF
&control
 calculation='scf', prefix='${MAT}', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'
/
&system
 $SYS
/
&electrons
 conv_thr=1.0d-10, mixing_beta=0.3, electron_maxstep=200, diago_david_ndim=4
/
$ATOMS
K_POINTS automatic
 $NK $NK 1 0 0 0
EOF
echo "[${MAT} d$DEGAUSS] scf ..."; grep -q "JOB DONE" scf.out 2>/dev/null || $MPI pw.x -in scf.in > scf.out 2>&1
grep -q "JOB DONE" scf.out || { echo "SCF FAIL"; exit 1; }
EF=$(grep -i 'the Fermi energy' scf.out | tail -1 | grep -oE '[-0-9.]+' | head -1)
echo "  E_F=$EF eV"

cat > ph.in <<EOF
${MAT} DFPT
&inputph
 prefix='${MAT}', outdir='./tmp', fildyn='${MAT}.dyn', fildvscf='dvscf'
 ldisp=.true., nq1=$NQ, nq2=$NQ, nq3=1, tr2_ph=1.0d-14, electron_phonon='dvscf', alpha_mix(1)=0.3
/
EOF
echo "[${MAT} d$DEGAUSS] DFPT ${NQ}x${NQ} (long, metal) ..."; grep -q "JOB DONE" ph.out 2>/dev/null || $MPI ph.x -in ph.in > ph.out 2>&1
grep -q "JOB DONE" ph.out || { echo "PH FAIL"; exit 1; }
MINFREQ=$(grep -aoE "freq \(.*\) = *[-0-9.]+ *\[cm-1\]" ph.out | grep -oE "[-0-9.]+ *\[cm-1\]" | grep -oE "[-0-9.]+" | sort -g | head -1)
echo "  PH ok, min phonon freq=$MINFREQ cm-1"

echo "[${MAT} d$DEGAUSS] gather dvscf ..."
rm -rf save; mkdir -p save; nq=$(sed -n '2p' ${MAT}.dyn0 | tr -dc 0-9)
cp -r tmp/_ph0/${MAT}.phsave save/
for iq in $(seq 1 "$nq"); do
  cp "${MAT}.dyn${iq}" "save/${MAT}.dyn_q${iq}"
  if [ "$iq" -eq 1 ]; then src=$(ls tmp/_ph0/${MAT}.dvscf${iq}_* | head -1)
  else src=$(ls tmp/_ph0/${MAT}.q_${iq}/${MAT}.dvscf${iq}_* | head -1); fi
  cp "$src" "save/${MAT}.dvscf_q${iq}"
done

awk -v nk="$NK" 'BEGIN{print nk*nk; for(i=0;i<nk;i++)for(j=0;j<nk;j++)printf "%.10f %.10f 0.0 1.0\n", i/nk, j/nk}' > kpts.txt
{ echo "&control"; echo " calculation='nscf', prefix='${MAT}', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'"; echo "/"
  echo "&system"; echo " $SYS"; echo " nbnd=$NBND"; echo "/"; echo "&electrons"; echo " conv_thr=1.0d-10, diago_full_acc=.true."; echo "/"
  echo "$ATOMS"; echo "K_POINTS crystal"; cat kpts.txt; } > nscf.in
echo "[${MAT} d$DEGAUSS] nscf ..."; grep -q "JOB DONE" nscf.out 2>/dev/null || $MPI pw.x -in nscf.in > nscf.out 2>&1
grep -q "JOB DONE" nscf.out || { echo "NSCF FAIL"; exit 1; }

cat > epw.in <<EOF
--
&inputepw
 prefix='${MAT}', outdir='./tmp'
 elph=.true., epbwrite=.true., epwwrite=.true.
 wannierize=.true., nbndsub=$NBNDSUB, num_iter=600, proj(1)='${M_ELM}:d', proj(2)='${X_ELM}:p'
 dis_win_max=8.0
 phonselfen=.true., a2f=.true., elecselfen=.false.
 efermi_read=.true., fermi_energy=$EF
 fsthick=4.0, degaussw=0.1, nsmear=4, delta_smear=0.05
 dvscf_dir='./save'
 nk1=$NK, nk2=$NK, nk3=1, nq1=$NQ, nq2=$NQ, nq3=1
 nkf1=$NKF, nk2=$NKF, nk3=1, nqf1=$NQF, nqf2=$NQF, nqf3=1
/
EOF
echo "[${MAT} d$DEGAUSS] epw.x (serial, pty) ..."; epw.x -in epw.in > epw.out 2>&1
LAM=$(grep -aE "lambda :" epw.out | tail -1 | grep -oE "[-0-9.]+" | head -1)
LAMTR=$(grep -aE "lambda_tr :" epw.out | tail -1 | grep -oE "[-0-9.]+" | head -1)
MAXG=$(grep -avE "^\s*#" linewidth.phself.* 2>/dev/null | awk '{print $NF}' | sort -g | tail -1)
echo "${MAT},${POLYTYPE},${M_ELM}${X_ELM},$DEGAUSS,$TEL,$EF,$MINFREQ,${LAM:-NA},${LAMTR:-NA},${MAXG:-NA},$NQ,$NKF" >> "$RESULT_CSV"
echo "[${MAT} d$DEGAUSS] DONE: lambda=${LAM:-NA}  minfreq=$MINFREQ  T_el=$TEL K  maxgamma=${MAXG:-NA} meV"
