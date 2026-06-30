#!/usr/bin/env bash
# Generalized TMD (E)-channel: scf -> DFPT NQxNQ (electron_phonon=dvscf) -> gather
# -> nscf -> Wannier(M:d + X:p) -> epw.x  ->  gamma_qv / lambda. Generalizes
# scripts/epw/nbse2_epw_full.sh to any 2H/1T monolayer via env (set by cpu_lane.sh
# from configs/v100_campaign.yaml). All session fixes kept: no dis_froz_max,
# efermi_read=scf E_F, awk kpts, step guards (re-run-safe), epw.x serial in pty.
#
# Required env: MAT A_ANG COA THICK_ANG M X MMASS XMASS POLY(2H|1T)
#               DEGAUSS NQ NK NKF NBNDSUB DISWINMAX PSEUDO WORK RESULT_CSV NP
set -u
source /root/miniconda3/etc/profile.d/conda.sh && { conda activate qe 2>/dev/null || conda activate phonon; }
: "${MAT:?} ${A_ANG:?} ${M:?} ${X:?} ${POLY:?}"
COA="${COA:-10.0}"; THICK_ANG="${THICK_ANG:-3.3}"
MMASS="${MMASS:-1.0}"; XMASS="${XMASS:-1.0}"
DEGAUSS="${DEGAUSS:-0.02}"; NQ="${NQ:-3}"; NK="${NK:-12}"; NKF="${NKF:-24}"; NQF="${NQF:-$NKF}"
NBNDSUB="${NBNDSUB:-11}"; DISWINMAX="${DISWINMAX:-8.0}"
PSEUDO="${PSEUDO:-/root/phonon/pseudo}"
WORK="${WORK:-/data/${MAT}_epw}"; RESULT_CSV="${RESULT_CSV:-/root/tmd_epw_results.csv}"
NP="${NP:-8}"; MPI="mpirun --allow-run-as-root -np $NP"
PREF=$(echo "$MAT" | tr -d '-')          # QE prefix: no hyphens

MPS=$(basename "$(ls "$PSEUDO/${M}"_ONCV_PBE*.upf 2>/dev/null | head -1)")
XPS=$(basename "$(ls "$PSEUDO/${X}"_ONCV_PBE*.upf 2>/dev/null | head -1)")
[ -z "$MPS" ] || [ -z "$XPS" ] && { echo "[$MAT] missing pseudo ($M:$MPS $X:$XPS) — run fetch_pseudos.sh"; exit 1; }

A_BOHR=$(awk "BEGIN{printf \"%.5f\", $A_ANG*1.88972598858}")
DZ=$(awk "BEGIN{printf \"%.5f\", $THICK_ANG/(2*$COA*$A_ANG)}")
ZP=$(awk "BEGIN{printf \"%.6f\", 0.5+$DZ}"); ZM=$(awk "BEGIN{printf \"%.6f\", 0.5-$DZ}")
TEL=$(awk "BEGIN{printf \"%.0f\", $DEGAUSS*157887}")
mkdir -p "$WORK"; cd "$WORK" || exit 1; export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
echo "===== $MAT ($POLY) EPW degauss=$DEGAUSS (T_el=$TEL K) NQ=$NQ NKF=$NKF a=$A_ANG WORK=$WORK ====="

# --- per-polytype X in-plane positions (2H: both at 1/3,2/3 ; 1T: 1/3,2/3 & 2/3,1/3)
if [ "$POLY" = "1T" ]; then X1="0.333333333 0.666666667"; X2="0.666666667 0.333333333"
else X1="0.333333333 0.666666667"; X2="0.333333333 0.666666667"; fi

SYS="ibrav=4, celldm(1)=$A_BOHR, celldm(3)=$COA, nat=3, ntyp=2, ecutwfc=70, ecutrho=280
 occupations='smearing', smearing='cold', degauss=$DEGAUSS"
ATOMS="ATOMIC_SPECIES
 $M $MMASS $MPS
 $X $XMASS $XPS
ATOMIC_POSITIONS (crystal)
 $M 0.000000000 0.000000000 0.500000000
 $X $X1 $ZP
 $X $X2 $ZM"

cat > scf.in <<EOF
&control
 calculation='scf', prefix='$PREF', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'
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
echo "[$MAT] scf ..."; grep -q "JOB DONE" scf.out 2>/dev/null || $MPI pw.x -in scf.in > scf.out 2>&1
grep -q "JOB DONE" scf.out || { echo "[$MAT] SCF FAIL"; exit 1; }
EF=$(grep -i 'the Fermi energy' scf.out | tail -1 | grep -oE '[-0-9.]+' | head -1)
echo "  E_F=$EF eV"

cat > ph.in <<EOF
$MAT DFPT
&inputph
 prefix='$PREF', outdir='./tmp', fildyn='$PREF.dyn', fildvscf='dvscf'
 ldisp=.true., nq1=$NQ, nq2=$NQ, nq3=1, tr2_ph=1.0d-14, electron_phonon='dvscf', alpha_mix(1)=0.3
/
EOF
echo "[$MAT] DFPT ${NQ}x${NQ} (long, metal) ..."; grep -q "JOB DONE" ph.out 2>/dev/null || $MPI ph.x -in ph.in > ph.out 2>&1
grep -q "JOB DONE" ph.out || { echo "[$MAT] PH FAIL"; exit 1; }
MINFREQ=$(grep -aoE "freq \(.*\) = *[-0-9.]+ *\[cm-1\]" ph.out | grep -oE "[-0-9.]+ *\[cm-1\]" | grep -oE "[-0-9.]+" | sort -g | head -1)
echo "  PH ok, min phonon freq=$MINFREQ cm-1"

echo "[$MAT] gather dvscf ..."
rm -rf save; mkdir -p save; nq=$(sed -n '2p' "$PREF.dyn0" | tr -dc 0-9)
cp -r tmp/_ph0/$PREF.phsave save/
for iq in $(seq 1 "$nq"); do
  cp "$PREF.dyn${iq}" "save/$PREF.dyn_q${iq}"
  if [ "$iq" -eq 1 ]; then src=$(ls tmp/_ph0/$PREF.dvscf${iq}_* | head -1)
  else src=$(ls tmp/_ph0/$PREF.q_${iq}/$PREF.dvscf${iq}_* | head -1); fi
  cp "$src" "save/$PREF.dvscf_q${iq}"
done

awk -v nk="$NK" 'BEGIN{print nk*nk; for(i=0;i<nk;i++)for(j=0;j<nk;j++)printf "%.10f %.10f 0.0 1.0\n", i/nk, j/nk}' > kpts.txt
{ echo "&control"; echo " calculation='nscf', prefix='$PREF', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'"; echo "/"
  echo "&system"; echo " $SYS"; echo " nbnd=30"; echo "/"; echo "&electrons"; echo " conv_thr=1.0d-10, diago_full_acc=.true."; echo "/"
  echo "$ATOMS"; echo "K_POINTS crystal"; cat kpts.txt; } > nscf.in
echo "[$MAT] nscf ..."; grep -q "JOB DONE" nscf.out 2>/dev/null || $MPI pw.x -in nscf.in > nscf.out 2>&1
grep -q "JOB DONE" nscf.out || { echo "[$MAT] NSCF FAIL"; exit 1; }

cat > epw.in <<EOF
--
&inputepw
 prefix='$PREF', outdir='./tmp'
 elph=.true., epbwrite=.true., epwwrite=.true.
 wannierize=.true., nbndsub=$NBNDSUB, num_iter=600, proj(1)='$M:d', proj(2)='$X:p'
 dis_win_max=$DISWINMAX
 phonselfen=.true., a2f=.true., elecselfen=.false.
 efermi_read=.true., fermi_energy=$EF
 fsthick=4.0, degaussw=0.1, nsmear=4, delta_smear=0.05
 dvscf_dir='./save'
 nk1=$NK, nk2=$NK, nk3=1, nq1=$NQ, nq2=$NQ, nq3=1
 nkf1=$NKF, nkf2=$NKF, nkf3=1, nqf1=$NQF, nqf2=$NQF, nqf3=1
/
EOF
echo "[$MAT] epw.x (serial, pty) ..."; epw.x -in epw.in > epw.out 2>&1
LAM=$(grep -aE "lambda :" epw.out | tail -1 | grep -oE "[-0-9.]+" | head -1)
LAMTR=$(grep -aE "lambda_tr :" epw.out | tail -1 | grep -oE "[-0-9.]+" | head -1)
MAXG=$(grep -avE "^\s*#" linewidth.phself.* 2>/dev/null | awk '{print $NF}' | sort -g | tail -1)
echo "$MAT,$DEGAUSS,$TEL,$EF,$MINFREQ,${LAM:-NA},${LAMTR:-NA},${MAXG:-NA},$NQ,$NKF" >> "$RESULT_CSV"
touch "$WORK/EPW_DONE"
echo "[$MAT] DONE: lambda=${LAM:-NA}  minfreq=$MINFREQ cm-1  T_el=$TEL K  maxgamma=${MAXG:-NA} meV"
