#!/usr/bin/env bash
# Validate the q18/q9 representation at 300 and 600 K before full-q export.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
SOURCE="${SOURCE:-/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz}"
INPUT_ROOT="${INPUT_ROOT:-/data/graphene_e0_epw_matched/e0_postprocess_inputs}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$SOURCE/analysis_multitemp}"

test -s "$SOURCE/analysis/SPECTRAL_GATE_PASS"
test -s "$SOURCE/full_q6_450K/DONE"
test -s "$SOURCE/EPW_SEED_DONE"
mkdir -p "$ANALYSIS_ROOT"
exec 9>"$ANALYSIS_ROOT/.lock"
if ! flock -n 9; then
    echo "[e0-multitemp] another gate process is already running"
    exit 0
fi
if [[ -s "$ANALYSIS_ROOT/MULTITEMP_SPECTRAL_GATE_PASS" || \
      -s "$ANALYSIS_ROOT/MULTITEMP_SPECTRAL_GATE_FAIL" ]]; then
    echo "[e0-multitemp] aggregate decision already complete"
    exit 0
fi

FERMI="$(awk '/the Fermi energy is/{value=$(NF-1)} END{print value}' "$SOURCE/scf.out")"
test -n "$FERMI"

run_temperature() {
    local temperature="$1" work="$SOURCE/gate_${1}K_q18q9" analysis="$ANALYSIS_ROOT/T${1}"
    local reference="$INPUT_ROOT/graphene_FD${1}_LINE_dfpt.csv"
    test -s "$reference"
    mkdir -p "$work/tmp" "$analysis"
    if [[ ! -s "$work/STATIC_GATE_DONE" ]]; then
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
        cp -p "$ROOT/configs/graphene_e0_epw_pilot/qpoints_GK9_k360.dat" \
            "$work/qpoints.dat"
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
        echo "[e0-multitemp] T=${temperature}K nine-point start=$(date -Is)"
        set +e
        (
            cd "$work"
            mpirun --allow-run-as-root -np 1 epw.x -in epw.in > epw.out.partial 2>&1
        )
        local status=$?
        set -e
        mv "$work/epw.out.partial" "$work/epw.out"
        [[ "$status" -eq 0 ]]
        grep -q "Total program execution" "$work/epw.out"
        test -s "$work/specfun_sup.phon"
        date -Is > "$work/STATIC_GATE_DONE"
    fi

    if [[ ! -s "$analysis/spectral_gate_decision.json" ]]; then
        "$CONDA" run --no-capture-output -n phonon python \
            "$ROOT/scripts/smearing_kink/analyze_graphene_e0_epw_static.py" \
            --epw-output "$work/epw.out" \
            --static-self-energy "$work/specfun_sup.phon" \
            --dfpt-target "$reference" --target-temperature-K "$temperature" \
            --static-reference "$INPUT_ROOT/graphene_sc6_dg0.020_phonopy.yaml" \
            --output-dir "$analysis"
        "$CONDA" run --no-capture-output -n phonon python \
            "$ROOT/scripts/smearing_kink/decide_graphene_e0_spectral_gate.py" \
            --summary "$analysis/static_pilot_summary.json" \
            --target-temperature-K "$temperature" --output-dir "$analysis"
    fi
}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate phonon
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

if [[ -s "$ANALYSIS_ROOT/T300/SPECTRAL_GATE_PASS" && \
      -s "$ANALYSIS_ROOT/T600/SPECTRAL_GATE_PASS" ]]; then
    date -Is > "$ANALYSIS_ROOT/MULTITEMP_SPECTRAL_GATE_PASS.tmp"
    mv "$ANALYSIS_ROOT/MULTITEMP_SPECTRAL_GATE_PASS.tmp" \
        "$ANALYSIS_ROOT/MULTITEMP_SPECTRAL_GATE_PASS"
    echo "[e0-multitemp] 300/600 K spectral gates passed"
else
    date -Is > "$ANALYSIS_ROOT/MULTITEMP_SPECTRAL_GATE_FAIL.tmp"
    mv "$ANALYSIS_ROOT/MULTITEMP_SPECTRAL_GATE_FAIL.tmp" \
        "$ANALYSIS_ROOT/MULTITEMP_SPECTRAL_GATE_FAIL"
    echo "[e0-multitemp] at least one 300/600 K spectral gate failed; stopping"
fi
