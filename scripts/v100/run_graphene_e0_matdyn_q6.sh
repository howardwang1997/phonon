#!/usr/bin/env bash
# Build the ASR-corrected q6 Cartesian dynamical matrices used by the E0 operator gate.
set -euo pipefail

SOURCE="${SOURCE:-/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz}"
WORK="${WORK:-$SOURCE/operator_q6_450K/matdyn}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
NQ="${NQ:-6}"

test -s "$SOURCE/graphene.dyn0"
mkdir -p "$WORK"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
    echo "[e0-matdyn] another process is already running"
    exit 0
fi

for path in "$SOURCE"/graphene.dyn*; do
    test -s "$path"
    destination="$WORK/$(basename "$path")"
    [[ -s "$destination" ]] || cp -p "$path" "$destination"
done

cat > "$WORK/q2r.in" <<'EOF'
&input
 fildyn='graphene.dyn', flfrc='graphene.fc', zasr='no'
/
EOF

awk -v nq="$NQ" 'BEGIN {
    print nq * nq
    for (i = 0; i < nq; i++)
        for (j = 0; j < nq; j++)
            printf "%.12f %.12f 0.000000000000\n", i / nq, j / nq
}' > "$WORK/qpoints_q6.dat"

cat > "$WORK/matdyn.in" <<'EOF'
&input
 flfrc='graphene.fc', asr='crystal', write_frc=.true.,
 flfrq='q6.freq', fleig='q6.eig', fldyn='q6.dyn',
 q_in_band_form=.false., q_in_cryst_coord=.true.,
 nosym=.true., loto_disable=.true.
/
EOF
cat "$WORK/qpoints_q6.dat" >> "$WORK/matdyn.in"

cat > "$WORK/matdyn_raw.in" <<'EOF'
&input
 flfrc='graphene.fc', asr='no', write_frc=.false.,
 flfrq='q6_raw.freq', fleig='q6_raw.eig', fldyn='q6_raw.dyn',
 q_in_band_form=.false., q_in_cryst_coord=.true.,
 nosym=.true., loto_disable=.true.
/
EOF
cat "$WORK/qpoints_q6.dat" >> "$WORK/matdyn_raw.in"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$WORK"
"$CONDA" run --no-capture-output -n phonon q2r.x -in q2r.in > q2r.out.partial 2>&1
mv q2r.out.partial q2r.out
grep -q "JOB DONE" q2r.out
test -s graphene.fc

"$CONDA" run --no-capture-output -n phonon matdyn.x -in matdyn_raw.in > matdyn_raw.out.partial 2>&1
mv matdyn_raw.out.partial matdyn_raw.out
grep -q "JOB DONE" matdyn_raw.out
test -s q6_raw.freq
test -s q6_raw.eig

"$CONDA" run --no-capture-output -n phonon matdyn.x -in matdyn.in > matdyn.out.partial 2>&1
mv matdyn.out.partial matdyn.out
grep -q "JOB DONE" matdyn.out
test -s q6.freq
test -s q6.eig
date -Is > MATDYN_Q6_DONE
echo "[e0-matdyn] complete=$(date -Is)"
