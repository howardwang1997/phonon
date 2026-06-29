#!/usr/bin/env bash
# Parametrized full NbSe2 EPW (scf -> DFPT NQxNQ -> nscf -> Wannier Nb:d+Se:p -> epw).
# Knobs via env: DEGAUSS (electronic smearing, Ry; T_el=DEGAUSS*157887 K), NQ (coarse q),
# NKF (fine grid), WORK, RESULT_CSV. Used for the lambda(T_el) sweep (vary DEGAUSS to
# harden the soft mode so integrated lambda is finite) and q-grid convergence (vary NQ).
# Incorporates all session fixes: no dis_froz_max, efermi_read=scf E_F, awk kpts, step
# guards (re-run-safe), run epw.x serially inside tmux (pty). Appends one CSV row.
set -u
source /root/miniconda3/etc/profile.d/conda.sh && { conda activate qe 2>/dev/null || conda activate phonon; }
DEGAUSS="${DEGAUSS:-0.02}"; NQ="${NQ:-3}"; NK="${NK:-12}"; NKF="${NKF:-24}"; NQF="${NQF:-$NKF}"
WORK="${WORK:-/data/nbse2_d${DEGAUSS}_q${NQ}}"; PSEUDO="${PSEUDO:-/root/phonon/pseudo}"
RESULT_CSV="${RESULT_CSV:-/root/nbse2_lambda_results.csv}"
NP="${NP:-8}"; MPI="mpirun --allow-run-as-root -np $NP"
A=6.5021; COA=10.0; DZ=0.0485
mkdir -p "$WORK"; cd "$WORK"; export OMP_NUM_THREADS=2
TEL=$(awk "BEGIN{printf \"%.0f\", $DEGAUSS*157887}")
echo "===== NbSe2 EPW degauss=$DEGAUSS Ry (T_el=$TEL K) NQ=$NQ NKF=$NKF WORK=$WORK ====="

SYS="ibrav=4, celldm(1)=$A, celldm(3)=$COA, nat=3, ntyp=2, ecutwfc=70, ecutrho=280
 occupations='smearing', smearing='cold', degauss=$DEGAUSS"
ATOMS="ATOMIC_SPECIES
 Nb 92.906 Nb_ONCV_PBE-1.2.upf
 Se 78.971 Se_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 Nb 0.000000000 0.000000000 0.500000000
 Se 0.333333333 0.666666667 $(awk "BEGIN{print 0.5+$DZ}")
 Se 0.333333333 0.666666667 $(awk "BEGIN{print 0.5-$DZ}")"

cat > scf.in <<EOF
&control
 calculation='scf', prefix='nbse2', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'
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
echo "[d$DEGAUSS] scf ..."; grep -q "JOB DONE" scf.out 2>/dev/null || $MPI pw.x -in scf.in > scf.out 2>&1
grep -q "JOB DONE" scf.out || { echo "SCF FAIL"; exit 1; }
EF=$(grep -i 'the Fermi energy' scf.out | tail -1 | grep -oE '[-0-9.]+' | head -1)
echo "  E_F=$EF eV"

cat > ph.in <<EOF
NbSe2 DFPT
&inputph
 prefix='nbse2', outdir='./tmp', fildyn='nbse2.dyn', fildvscf='dvscf'
 ldisp=.true., nq1=$NQ, nq2=$NQ, nq3=1, tr2_ph=1.0d-14, electron_phonon='dvscf', alpha_mix(1)=0.3
/
EOF
echo "[d$DEGAUSS] DFPT ${NQ}x${NQ} (long, metal) ..."; grep -q "JOB DONE" ph.out 2>/dev/null || $MPI ph.x -in ph.in > ph.out 2>&1
grep -q "JOB DONE" ph.out || { echo "PH FAIL"; exit 1; }
MINFREQ=$(grep -aoE "freq \(.*\) = *[-0-9.]+ *\[cm-1\]" ph.out | grep -oE "[-0-9.]+ *\[cm-1\]" | grep -oE "[-0-9.]+" | sort -g | head -1)
echo "  PH ok, min phonon freq=$MINFREQ cm-1"

echo "[d$DEGAUSS] gather dvscf ..."
rm -rf save; mkdir -p save; nq=$(sed -n '2p' nbse2.dyn0 | tr -dc 0-9)
cp -r tmp/_ph0/nbse2.phsave save/
for iq in $(seq 1 "$nq"); do
  cp "nbse2.dyn${iq}" "save/nbse2.dyn_q${iq}"
  if [ "$iq" -eq 1 ]; then src=$(ls tmp/_ph0/nbse2.dvscf${iq}_* | head -1)
  else src=$(ls tmp/_ph0/nbse2.q_${iq}/nbse2.dvscf${iq}_* | head -1); fi
  cp "$src" "save/nbse2.dvscf_q${iq}"
done

awk -v nk="$NK" 'BEGIN{print nk*nk; for(i=0;i<nk;i++)for(j=0;j<nk;j++)printf "%.10f %.10f 0.0 1.0\n", i/nk, j/nk}' > kpts.txt
{ echo "&control"; echo " calculation='nscf', prefix='nbse2', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'"; echo "/"
  echo "&system"; echo " $SYS"; echo " nbnd=30"; echo "/"; echo "&electrons"; echo " conv_thr=1.0d-10, diago_full_acc=.true."; echo "/"
  echo "$ATOMS"; echo "K_POINTS crystal"; cat kpts.txt; } > nscf.in
echo "[d$DEGAUSS] nscf ..."; grep -q "JOB DONE" nscf.out 2>/dev/null || $MPI pw.x -in nscf.in > nscf.out 2>&1
grep -q "JOB DONE" nscf.out || { echo "NSCF FAIL"; exit 1; }

cat > epw.in <<EOF
--
&inputepw
 prefix='nbse2', outdir='./tmp'
 elph=.true., epbwrite=.true., epwwrite=.true.
 wannierize=.true., nbndsub=11, num_iter=600, proj(1)='Nb:d', proj(2)='Se:p'
 dis_win_max=8.0
 phonselfen=.true., a2f=.true., elecselfen=.false.
 efermi_read=.true., fermi_energy=$EF
 fsthick=4.0, degaussw=0.1, nsmear=4, delta_smear=0.05
 dvscf_dir='./save'
 nk1=$NK, nk2=$NK, nk3=1, nq1=$NQ, nq2=$NQ, nq3=1
 nkf1=$NKF, nkf2=$NKF, nkf3=1, nqf1=$NQF, nqf2=$NQF, nqf3=1
/
EOF
echo "[d$DEGAUSS] epw.x (serial, pty) ..."; epw.x -in epw.in > epw.out 2>&1
LAM=$(grep -aE "lambda :" epw.out | tail -1 | grep -oE "[-0-9.]+" | head -1)
LAMTR=$(grep -aE "lambda_tr :" epw.out | tail -1 | grep -oE "[-0-9.]+" | head -1)
MAXG=$(grep -avE "^\s*#" linewidth.phself.* 2>/dev/null | awk '{print $NF}' | sort -g | tail -1)
echo "$DEGAUSS,$TEL,$EF,$MINFREQ,${LAM:-NA},${LAMTR:-NA},${MAXG:-NA},$NQ,$NKF" >> "$RESULT_CSV"
echo "[d$DEGAUSS] DONE: lambda=${LAM:-NA}  minfreq=$MINFREQ cm-1  T_el=$TEL K  maxgamma=${MAXG:-NA} meV"
