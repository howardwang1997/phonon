#!/usr/bin/env bash
# Run the k720-commensurate dense K-line EPW response at 300/450/600 K.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
SOURCE="${SOURCE:-/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz}"
WORK_ROOT="${WORK_ROOT:-/data/graphene_k_cusp_b0/dense_k29_v2}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
NQ="${NQ:-29}"

test -s "$SOURCE/EPW_SEED_DONE"
test -s "$SOURCE/analysis_multitemp/MULTITEMP_SPECTRAL_GATE_PASS"
test "$NQ" -eq 29
mkdir -p "$WORK_ROOT"
exec 9>"$WORK_ROOT/.lock"
if ! flock -n 9; then
    echo "[cusp-b0] another dense-line process is running"
    exit 0
fi
if [[ -s "$WORK_ROOT/DONE" ]]; then
    echo "[cusp-b0] already complete"
    exit 0
fi

awk -v nq="$NQ" 'BEGIN {
    print nq, "crystal"
    for (i = 0; i < nq; i++) {
        m = 226 + i
        q = m / 720.0
        printf "%.12f %.12f 0.000000000000 %.12f\n", q, q, 1.0 / nq
    }
}' > "$WORK_ROOT/qpoints_K29.dat"

FERMI="$(awk '/the Fermi energy is/{value=$(NF-1)} END{print value}' "$SOURCE/scf.out")"
test -n "$FERMI"

run_temperature() {
    local temperature="$1"
    local work="$WORK_ROOT/T${temperature}"
    mkdir -p "$work/tmp"
    exec {temperature_lock}>"$work/.lock"
    if ! flock -n "$temperature_lock"; then
        echo "[cusp-b0] T=${temperature} K is already running"
        return 0
    fi
    if [[ -s "$work/DONE" ]]; then
        echo "[cusp-b0] T=${temperature} K already complete"
        return 0
    fi

    local artifact destination
    for artifact in \
        epwdata.fmt dmedata.fmt vmedata.fmt crystal.fmt decay.H decay.P \
        decay.dynmat decay.epmate decay.epmatp decay.r decay.v \
        graphene.ukk graphene.kmap graphene.kgmap; do
        test -s "$SOURCE/$artifact"
        destination="$work/$artifact"
        [[ -e "$destination" ]] || ln -s "$SOURCE/$artifact" "$destination"
    done
    test -s "$SOURCE/tmp/graphene.epmatwp"
    [[ -e "$work/tmp/graphene.epmatwp" ]] || \
        ln -s "$SOURCE/tmp/graphene.epmatwp" "$work/tmp/graphene.epmatwp"
    [[ -e "$work/qpoints.dat" ]] || ln -s "$WORK_ROOT/qpoints_K29.dat" "$work/qpoints.dat"

    cat > "$work/epw.in" <<EOF
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
 temps(1)=$temperature, etf_mem=1, iverbosity=2,
 filqf='./qpoints.dat',
 nk1=18, nk2=18, nk3=1, nq1=9, nq2=9, nq3=1,
 nkf1=720, nkf2=720, nkf3=1, nqf1=1, nqf2=1, nqf3=1
/
EOF
    date -Is > "$work/STARTED_AT"
    echo "[cusp-b0] T=${temperature} K start=$(cat "$work/STARTED_AT")"
    set +e
    (
        cd "$work"
        mpirun --allow-run-as-root -np 1 epw.x -in epw.in > epw.out.partial 2>&1
    )
    local status=$?
    set -e
    if [[ -e "$work/epw.out.partial" ]]; then
        mv "$work/epw.out.partial" "$work/epw.out"
    fi
    [[ "$status" -eq 0 ]]
    grep -q "Total program execution" "$work/epw.out"
    test -s "$work/specfun_sup.phon"
    "$CONDA" run --no-capture-output -n phonon python \
        "$ROOT/scripts/smearing_kink/audit_graphene_k_cusp_b0_epw.py" \
        --qpoints "$work/qpoints.dat" \
        --static-self-energy "$work/specfun_sup.phon" \
        --epw-output "$work/epw.out" \
        --temperature-K "$temperature" --expected-nq "$NQ" \
        --output "$work/audit.json"
    date -Is > "$work/COMPLETED_AT"
    cp "$work/COMPLETED_AT" "$work/DONE.tmp"
    mv "$work/DONE.tmp" "$work/DONE"
    echo "[cusp-b0] T=${temperature} K complete=$(cat "$work/COMPLETED_AT")"
}

source /root/miniconda3/etc/profile.d/conda.sh
if [[ -x /root/miniconda3/envs/qe/bin/epw.x ]]; then
    conda activate qe
else
    conda activate phonon
fi
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_temperature 300 &
pid_300=$!
run_temperature 600 &
pid_600=$!
parallel_status=0
wait "$pid_300" || parallel_status=1
wait "$pid_600" || parallel_status=1
[[ "$parallel_status" -eq 0 ]]
run_temperature 450

for temperature in 300 450 600; do
    test -s "$WORK_ROOT/T${temperature}/DONE"
    test -s "$WORK_ROOT/T${temperature}/audit.json"
done
date -Is > "$WORK_ROOT/COMPLETED_AT"
cp "$WORK_ROOT/COMPLETED_AT" "$WORK_ROOT/DONE.tmp"
mv "$WORK_ROOT/DONE.tmp" "$WORK_ROOT/DONE"
echo "[cusp-b0] all temperatures complete=$(cat "$WORK_ROOT/COMPLETED_AT")"
