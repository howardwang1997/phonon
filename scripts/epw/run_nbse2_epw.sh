#!/usr/bin/env bash
# NbSe2 EPW prep (scf -> DFPT 3x3 q [captures q_CDW] -> gather), conda CPU QE.
# The deep (E)-channel: lambda_q / gamma_qv peaking at q_CDW = the momentum-dependent
# e-ph coupling that drives the CDW (ties to #2 not-nesting). 2D metal -> c/a=10 large
# vacuum (same 2D-wannier b-vector reason as graphene). Soft mode at q_CDW is expected
# (imaginary); EPW's phonon self-energy / lambda_q are still meaningful.
#   bash run_nbse2_epw.sh prep   # scf + DFPT + gather  (the multi-hour long pole)
#   bash run_nbse2_epw.sh epw    # nscf + epw.x         (after prep + Wannier tuning)
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate phonon
STAGE="${1:-prep}"
WORK="${WORK:-/data/nbse2_epw}"; PSEUDO="${PSEUDO:-/root/phonon/pseudo}"
NP="${NP:-8}"; MPI="mpirun --allow-run-as-root -np $NP"
A=6.5021; COA=10.0          # NbSe2 a=3.44 A in bohr; c/a=10 vacuum
DZ=0.0485                   # Se z-offset (thickness 3.34 A / 2 / c)
NK=12; NQ=3                 # coarse k 12x12, coarse q 3x3 (includes q_CDW=(1/3,1/3))
NKF=24; NQF=24
mkdir -p "$WORK"; cd "$WORK"; export OMP_NUM_THREADS=2

write_struct () { cat <<EOF
&system
 ibrav=4, celldm(1)=$A, celldm(3)=$COA, nat=3, ntyp=2, ecutwfc=70, ecutrho=280
 occupations='smearing', smearing='cold', degauss=0.02
$1
/
&electrons
 conv_thr=1.0d-10, mixing_beta=0.3, electron_maxstep=200, diago_david_ndim=4
/
ATOMIC_SPECIES
 Nb 92.906 Nb_ONCV_PBE-1.2.upf
 Se 78.971 Se_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 Nb 0.000000000 0.000000000 0.500000000
 Se 0.333333333 0.666666667 $(awk "BEGIN{print 0.5+$DZ}")
 Se 0.333333333 0.666666667 $(awk "BEGIN{print 0.5-$DZ}")
EOF
}

if [ "$STAGE" = prep ]; then
  { echo "&control"; echo " calculation='scf', prefix='nbse2', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'"; echo "/"; write_struct ""; echo "K_POINTS automatic"; echo " $NK $NK 1 0 0 0"; } > scf.in
  echo "[nbse2-epw] scf ..."; $MPI pw.x -in scf.in > scf.out 2>&1
  grep -q "JOB DONE" scf.out && echo "  SCF ok Ef=$(grep -i 'the Fermi energy' scf.out | tail -1 | grep -oE '[-0-9.]+ ev')" || { echo "  SCF FAIL"; tail -5 scf.out; exit 1; }

  cat > ph.in <<EOF
NbSe2 DFPT ${NQ}x${NQ} (metal, captures q_CDW soft mode)
&inputph
 prefix='nbse2', outdir='./tmp', fildyn='nbse2.dyn', fildvscf='dvscf'
 ldisp=.true., nq1=$NQ, nq2=$NQ, nq3=1
 tr2_ph=1.0d-14, electron_phonon='dvscf'
 alpha_mix(1)=0.3
/
EOF
  echo "[nbse2-epw] ph.x DFPT ${NQ}x${NQ} (LONG POLE - metal) ..."; $MPI ph.x -in ph.in > ph.out 2>&1
  grep -q "JOB DONE" ph.out && echo "  PH ok" || { echo "  PH FAIL"; tail -8 ph.out; exit 1; }

  echo "[nbse2-epw] gather dvscf -> save/ ..."
  rm -rf save; mkdir -p save; nq=$(sed -n '2p' nbse2.dyn0 | tr -dc 0-9)
  cp -r tmp/_ph0/nbse2.phsave save/
  for iq in $(seq 1 "$nq"); do
    cp "nbse2.dyn${iq}" "save/nbse2.dyn_q${iq}"
    if [ "$iq" -eq 1 ]; then src=$(ls tmp/_ph0/nbse2.dvscf${iq}_* | head -1)
    else src=$(ls tmp/_ph0/nbse2.q_${iq}/nbse2.dvscf${iq}_* | head -1); fi
    cp "$src" "save/nbse2.dvscf_q${iq}"
  done
  echo "  gathered $nq q-points; PREP_COMPLETE"
fi

if [ "$STAGE" = epw ]; then
  echo "[nbse2-epw] nscf full ${NK}x${NK} ..."
  python - "$NK" > kpts.txt <<'PY'
import sys; nk=int(sys.argv[1]); p=[(i/nk,j/nk,0.0) for i in range(nk) for j in range(nk)]
print(len(p)); [print(f"{x:.10f} {y:.10f} {z:.10f} 1.0") for x,y,z in p]
PY
  { echo "&control"; echo " calculation='nscf', prefix='nbse2', outdir='./tmp', pseudo_dir='$PSEUDO', verbosity='high'"; echo "/"; write_struct " nbnd=30"; echo "K_POINTS crystal"; cat kpts.txt; } > nscf.in
  grep -q "JOB DONE" nscf.out 2>/dev/null || $MPI pw.x -in nscf.in > nscf.out 2>&1; grep -q "JOB DONE" nscf.out && echo "  NSCF ok" || { echo "NSCF FAIL"; exit 1; }
  cat > epw.in <<EOF
--
&inputepw
 prefix='nbse2', outdir='./tmp'
 elph=.true., epbwrite=.true., epwwrite=.true.
 wannierize=.true., nbndsub=11, num_iter=600
 proj(1)='Nb:d', proj(2)='Se:p'
 dis_win_max=8.0
 phonselfen=.true., a2f=.true., elecselfen=.false.
 efermi_read=.true., fermi_energy=-2.7613
 fsthick=4.0, degaussw=0.2, nsmear=1, delta_smear=0.1
 dvscf_dir='./save'
 nk1=$NK, nk2=$NK, nk3=1, nq1=$NQ, nq2=$NQ, nq3=1
 nkf1=$NKF, nkf2=$NKF, nkf3=1, nqf1=$NQF, nqf2=$NQF, nqf3=1
/
EOF
  echo "[nbse2-epw] epw.x (MUST run in tmux/pty) ..."; epw.x -in epw.in > epw.out 2>&1
  echo "  EPW done"; grep -aiE "lambda|a2f|linewidth" epw.out | tail -5
fi
echo "NBSE2_EPW_${STAGE}_DONE"
