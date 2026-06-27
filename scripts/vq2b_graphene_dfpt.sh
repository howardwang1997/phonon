#!/bin/bash
# #3 (rigorous (E)-channel): graphene DFPT (ph.x) Gamma-E2g phonon vs electronic
# smearing. The frozen-phonon V-Q2 showed the cusp sharpness changes with T_el;
# this is the gold-standard linear-response version: ph.x phonon frequency +
# (if available) the electron-phonon linewidth of the E2g Kohn-anomaly mode vs
# degauss (= electronic temperature). Both pw.x and ph.x from the SAME conda-QE
# (7.5) so the save files are compatible (the GPU build is 7.3.1, pw-only).
set -uo pipefail
QEBIN=/root/miniconda3/envs/qe/bin
PSEUDO=/root/phonon/pseudo
WORK=/root/phonon/results/vq2b_dfpt
mkdir -p "$WORK"; cd "$WORK"
A_BOHR=4.64864       # 2.46 A in bohr
COVERA=6.0976        # c = 15 A (vacuum)
RES="$WORK/dfpt_smearing.csv"
echo "degauss_Ry,T_el_K,E2g_cm,top_gamma_line" > "$RES"
for DG in 0.005 0.01 0.02 0.04; do
  D="$WORK/dg$DG"; rm -rf "$D"; mkdir -p "$D/tmp"; cd "$D"
  cat > scf.in <<EOF
&control
 calculation='scf', prefix='gr', outdir='./tmp', pseudo_dir='$PSEUDO',
 tprnfor=.true., verbosity='low'
/
&system
 ibrav=4, celldm(1)=$A_BOHR, celldm(3)=$COVERA, nat=2, ntyp=1,
 ecutwfc=60, ecutrho=240, occupations='smearing', smearing='mp', degauss=$DG
/
&electrons
 conv_thr=1d-11, mixing_beta=0.4
/
ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.000000000 0.000000000 0.0
 C 0.333333333 0.666666667 0.0
K_POINTS automatic
 36 36 1 0 0 0
EOF
  cat > ph.in <<EOF
graphene Gamma E2g phonon (frequencies only; el-ph linewidth needs a 2-pass dvscf)
&inputph
 prefix='gr', outdir='./tmp', fildyn='gr.dyn',
 tr2_ph=1d-16, ldisp=.false., trans=.true., epsil=.false.
/
0.0 0.0 0.0
EOF
  echo "[dfpt] dg=$DG: pw.x scf ..."
  "$QEBIN/mpirun" --allow-run-as-root -np 8 "$QEBIN/pw.x" -in scf.in > scf.out 2>&1 || { echo "pw FAIL dg=$DG"; continue; }
  echo "[dfpt] dg=$DG: ph.x Gamma ..."
  "$QEBIN/mpirun" --allow-run-as-root -np 8 "$QEBIN/ph.x" -in ph.in > ph.out 2>&1 || echo "ph nonzero dg=$DG (frequencies may still be present)"
  # highest Gamma frequency (E2g) in cm-1
  E2G=$(grep -F "[cm-1]" ph.out | sed -E 's/.*=\s*(-?[0-9.]+)\s*\[cm-1\].*/\1/' | sort -g | tail -1)
  GAM=$(grep -iE "lambda|gamma|broadening|linewidth" ph.out | tail -1 | tr ',' ' ')
  TEL=$(awk "BEGIN{printf \"%.0f\", $DG*157887}")
  echo "$DG,$TEL,$E2G,$GAM" >> "$RES"
  echo "[dfpt] dg=$DG (T_el=${TEL}K): E2g=$E2G cm-1"
done
echo "=== DFPT-smearing done ==="
cat "$RES"
