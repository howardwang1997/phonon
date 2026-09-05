#!/usr/bin/env bash
# Export the complete q6 static self-energy grid after the nine-point gate passes.
# q6 is commensurate with the 6x6, 72-atom E1 configurations.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
SOURCE="${SOURCE:-/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz}"
TARGET_TEMPERATURE_K="${TARGET_TEMPERATURE_K:-450}"
WORK="${WORK:-$SOURCE/full_q6_${TARGET_TEMPERATURE_K}K}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
NQ_FULL="${NQ_FULL:-6}"
RELEASE_MARKER="${RELEASE_MARKER:-$SOURCE/analysis/SPECTRAL_GATE_PASS}"

test -s "$RELEASE_MARKER"
test -s "$SOURCE/EPW_SEED_DONE"
if [[ -s "$WORK/DONE" && -s "$WORK/audit/FULL_Q_DATA_READY" ]]; then
    echo "[e0-full-q] already complete"
    exit 0
fi

mkdir -p "$WORK/tmp" "$WORK/audit"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
    echo "[e0-full-q] another q${NQ_FULL} export is already running"
    exit 0
fi
for source in \
    epwdata.fmt dmedata.fmt vmedata.fmt crystal.fmt decay.H decay.P \
    decay.dynmat decay.epmate decay.epmatp decay.r decay.v \
    graphene.ukk graphene.kmap graphene.kgmap; do
    test -s "$SOURCE/$source"
    [[ -s "$WORK/$source" ]] || cp -p "$SOURCE/$source" "$WORK/$source"
done
test -s "$SOURCE/tmp/graphene.epmatwp"
[[ -e "$WORK/tmp/graphene.epmatwp" ]] || \
    ln -s "$SOURCE/tmp/graphene.epmatwp" "$WORK/tmp/graphene.epmatwp"

awk -v nq="$NQ_FULL" 'BEGIN {
    print nq * nq, "crystal"
    weight = 1.0 / (nq * nq)
    for (i = 0; i < nq; i++)
        for (j = 0; j < nq; j++)
            printf "%.12f %.12f 0.000000000000 %.12f\n", i / nq, j / nq, weight
}' > "$WORK/qpoints_full.dat"

FERMI="$(awk '/the Fermi energy is/{value=$(NF-1)} END{print value}' "$SOURCE/scf.out")"
test -n "$FERMI"
cat > "$WORK/epw.in" <<EOF
--
&inputepw
 prefix='graphene', outdir='./tmp',
 elph=.true., ep_coupling=.true., epwread=.true., epwwrite=.false.,
 wannierize=.false., nbndsub=2,
 bands_skipped='exclude_bands = 1',
 phonselfen=.true., specfun_ph=.true.,
 wmin_specfun=0.0, wmax_specfun=0.001, nw_specfun=2,
 a2f=.false., elecselfen=.false.,
 efermi_read=.true., fermi_energy=$FERMI,
 fsthick=6.0, degaussw=0.005, nsmear=1, delta_smear=0.005,
 temps(1)=$TARGET_TEMPERATURE_K, etf_mem=1, iverbosity=2,
 filqf='./qpoints_full.dat',
 nk1=18, nk2=18, nk3=1, nq1=9, nq2=9, nq3=1,
 nkf1=720, nkf2=720, nkf3=1, nqf1=1, nqf2=1, nqf3=1
/
EOF

source /root/miniconda3/etc/profile.d/conda.sh
if [[ -x /root/miniconda3/envs/qe/bin/epw.x ]]; then
    conda activate qe
else
    conda activate phonon
fi
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$WORK"
echo "[e0-full-q] start=$(date -Is) T=${TARGET_TEMPERATURE_K}K q=${NQ_FULL}x${NQ_FULL} fine-k=720"
set +e
mpirun --allow-run-as-root -np 1 epw.x -in epw.in > epw.out.partial 2>&1
status=$?
set -e
mv epw.out.partial epw.out
[[ "$status" -eq 0 ]]
grep -q "Total program execution" epw.out
test -s specfun_sup.phon

"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/audit_graphene_e0_full_q.py" \
    --qpoints "$WORK/qpoints_full.dat" \
    --static-self-energy "$WORK/specfun_sup.phon" \
    --ngrid "$NQ_FULL" --temperature-K "$TARGET_TEMPERATURE_K" \
    --output-dir "$WORK/audit"
test -s "$WORK/audit/FULL_Q_DATA_READY"
date -Is > "$WORK/DONE.tmp"
mv "$WORK/DONE.tmp" "$WORK/DONE"
echo "[e0-full-q] complete=$(date -Is)"
