#!/usr/bin/env bash
# Build a graphene EPW representation in the same 15 A cell used by the
# physical-FD DFPT target.  Every expensive stage has a completion marker so
# a systemd restart resumes at the last completed stage.
set -euo pipefail

MODE="${1:-full}"
ROOT="${ROOT:-/root/phonon}"
WORK="${WORK:-/data/graphene_e0_epw_matched/k12_q6}"
PSEUDO="${PSEUDO:-/root/phonon/pseudo}"
NK="${NK:-12}"
NQ="${NQ:-6}"
NP="${NP:-8}"
RUN_STATIC_GATE="${RUN_STATIC_GATE:-1}"
SOURCE_PREP="${SOURCE_PREP:-}"
BANDS_SKIPPED="${BANDS_SKIPPED:-}"
DIS_WIN_MIN="${DIS_WIN_MIN:-}"
DIS_WIN_MAX="${DIS_WIN_MAX:-}"
DIS_FROZ_MIN="${DIS_FROZ_MIN:-}"
DIS_FROZ_MAX="${DIS_FROZ_MAX:-}"

if [[ -n "$DIS_FROZ_MIN" || -n "$DIS_FROZ_MAX" ]]; then
    if [[ -z "$DIS_FROZ_MIN" || -z "$DIS_FROZ_MAX" ]]; then
        echo "DIS_FROZ_MIN and DIS_FROZ_MAX must be set together" >&2
        exit 2
    fi
fi

case "$MODE" in
    prep|full) ;;
    *)
        echo "usage: $0 prep|full" >&2
        exit 2
        ;;
esac

if (( NK % NQ != 0 )); then
    echo "coarse electronic grid NK=$NK must be commensurate with NQ=$NQ" >&2
    exit 2
fi
if (( NK % 3 != 0 || NQ % 3 != 0 )); then
    echo "NK=$NK and NQ=$NQ must include graphene K" >&2
    exit 2
fi
test -s "$PSEUDO/C_ONCV_PBE-1.2.upf"
test -s "$ROOT/configs/graphene_e0_epw_pilot/qpoints_GK9_k360.dat"

source /root/miniconda3/etc/profile.d/conda.sh
if [[ -x /root/miniconda3/envs/qe/bin/epw.x ]]; then
    conda activate qe
else
    conda activate phonon
fi
for executable in pw.x ph.x epw.x wannier90.x; do
    command -v "$executable" >/dev/null
done

mkdir -p "$WORK"
if [[ -n "$SOURCE_PREP" && ! -s "$WORK/PREP_DONE" ]]; then
    test "$SOURCE_PREP" != "$WORK"
    for marker in SCF_DONE PH_DONE GATHER_DONE NSCF_DONE PREP_DONE; do
        test -s "$SOURCE_PREP/$marker"
    done
    echo "[matched] copying completed coarse preparation from $SOURCE_PREP"
    cp --reflink=auto -a "$SOURCE_PREP/tmp" "$WORK/"
    cp --reflink=auto -a "$SOURCE_PREP/save" "$WORK/"
    rm -f "$WORK/tmp/graphene.epmatwp" "$WORK/tmp"/graphene.epb*
    cp -p "$SOURCE_PREP"/graphene.dyn* "$WORK/"
    for artifact in \
        scf.in scf.out ph.in ph.out nscf.in nscf.out kpoints_full.dat \
        IRREDUCIBLE_Q_COUNT SCF_DONE PH_DONE GATHER_DONE NSCF_DONE PREP_DONE; do
        cp -p "$SOURCE_PREP/$artifact" "$WORK/$artifact"
    done
    printf '%s\n' "$SOURCE_PREP" > "$WORK/REUSED_PREP_FROM"
fi
mkdir -p "$WORK/tmp" "$WORK/save"
cd "$WORK"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
MPI=(mpirun --allow-run-as-root -np "$NP")

A_BOHR=4.648726286
COA=6.097560976
DEGAUSS_RY=0.020

write_parameters() {
    {
        echo "cell_a_bohr=$A_BOHR"
        echo "cell_c_over_a=$COA"
        echo "vacuum_c_angstrom=15.0"
        echo "ecutwfc_Ry=60"
        echo "ecutrho_Ry=240"
        echo "coarse_k=${NK}x${NK}x1"
        echo "coarse_q=${NQ}x${NQ}x1"
        echo "reference_smearing_degauss_Ry=$DEGAUSS_RY"
        echo "target_lattice_temperature_K=450"
        echo "bands_skipped=${BANDS_SKIPPED:-none}"
        echo "dis_win_min_eV=${DIS_WIN_MIN:-none}"
        echo "dis_win_max_eV=${DIS_WIN_MAX:-none}"
        echo "dis_froz_min_eV=${DIS_FROZ_MIN:-none}"
        echo "dis_froz_max_eV=${DIS_FROZ_MAX:-none}"
        echo "source_prep=${SOURCE_PREP:-none}"
        echo "pseudo=$(sha256sum "$PSEUDO/C_ONCV_PBE-1.2.upf" | awk '{print $1}')"
        echo "qe=$(pw.x -help 2>&1 | awk '/Program PWSCF/{print $3; exit}')"
        echo "host=$(hostname)"
    } > PARAMETERS.txt
}

run_scf() {
    if [[ -s SCF_DONE ]] && grep -q "JOB DONE" scf.out; then
        echo "[matched] SCF already complete"
        return
    fi
    cat > scf.in <<EOF
&control
 calculation='scf', prefix='graphene', outdir='./tmp', pseudo_dir='$PSEUDO',
 verbosity='high', disk_io='low'
/
&system
 ibrav=4, celldm(1)=$A_BOHR, celldm(3)=$COA, nat=2, ntyp=1,
 ecutwfc=60, ecutrho=240,
 occupations='smearing', smearing='fd', degauss=$DEGAUSS_RY
/
&electrons
 conv_thr=1.0d-12, mixing_beta=0.3, electron_maxstep=300,
 diagonalization='david'
/
ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.000000000 0.000000000 0.0
 C 0.333333333 0.666666667 0.0
K_POINTS automatic
 $NK $NK 1 0 0 0
EOF
    echo "[matched] SCF k=$NK start=$(date -Is)"
    "${MPI[@]}" pw.x -in scf.in > scf.out.partial 2>&1
    mv scf.out.partial scf.out
    grep -q "JOB DONE" scf.out
    date -Is > SCF_DONE
}

run_ph() {
    if [[ -s PH_DONE ]] && grep -q "JOB DONE" ph.out; then
        echo "[matched] DFPT already complete"
        return
    fi
    cat > ph.in <<EOF
graphene matched-cell EPW DFPT
&inputph
 prefix='graphene', outdir='./tmp', fildyn='graphene.dyn', fildvscf='dvscf',
 ldisp=.true., nq1=$NQ, nq2=$NQ, nq3=1, tr2_ph=1.0d-14,
 electron_phonon='dvscf', alpha_mix(1)=0.3
/
EOF
    echo "[matched] DFPT q=$NQ start=$(date -Is)"
    "${MPI[@]}" ph.x -in ph.in > ph.out.partial 2>&1
    mv ph.out.partial ph.out
    grep -q "JOB DONE" ph.out
    date -Is > PH_DONE
}

gather_dvscf() {
    if [[ -s GATHER_DONE ]]; then
        echo "[matched] dvscf gather already complete"
        return
    fi
    local nq_ir iq source
    nq_ir="$(sed -n '2p' graphene.dyn0 | tr -dc '0-9')"
    test -n "$nq_ir"
    mkdir -p save/graphene.phsave
    cp -a tmp/_ph0/graphene.phsave/. save/graphene.phsave/
    for iq in $(seq 1 "$nq_ir"); do
        cp -p "graphene.dyn${iq}" "save/graphene.dyn_q${iq}"
        if [[ "$iq" -eq 1 ]]; then
            source="$(find tmp/_ph0 -maxdepth 1 -type f -name "graphene.dvscf${iq}_*" -print -quit)"
        else
            source="$(find "tmp/_ph0/graphene.q_${iq}" -maxdepth 1 -type f -name "graphene.dvscf${iq}_*" -print -quit)"
        fi
        test -s "$source"
        cp -p "$source" "save/graphene.dvscf_q${iq}"
    done
    printf '%s\n' "$nq_ir" > IRREDUCIBLE_Q_COUNT
    date -Is > GATHER_DONE
}

run_nscf() {
    if [[ -s NSCF_DONE ]] && grep -q "JOB DONE" nscf.out; then
        echo "[matched] NSCF already complete"
        return
    fi
    awk -v nk="$NK" 'BEGIN {
        print nk * nk
        for (i = 0; i < nk; i++)
            for (j = 0; j < nk; j++)
                printf "%.10f %.10f 0.0 1.0\n", i / nk, j / nk
    }' > kpoints_full.dat
    cat > nscf.in <<EOF
&control
 calculation='nscf', prefix='graphene', outdir='./tmp', pseudo_dir='$PSEUDO',
 verbosity='high', disk_io='low'
/
&system
 ibrav=4, celldm(1)=$A_BOHR, celldm(3)=$COA, nat=2, ntyp=1,
 ecutwfc=60, ecutrho=240, nbnd=14,
 occupations='smearing', smearing='fd', degauss=$DEGAUSS_RY
/
&electrons
 conv_thr=1.0d-12, diago_full_acc=.true., diago_david_ndim=8,
 electron_maxstep=300, diago_thr_init=1.0d-5
/
ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.000000000 0.000000000 0.0
 C 0.333333333 0.666666667 0.0
K_POINTS crystal
EOF
    cat kpoints_full.dat >> nscf.in
    echo "[matched] NSCF full k=$NK start=$(date -Is)"
    "${MPI[@]}" pw.x -in nscf.in > nscf.out.partial 2>&1
    mv nscf.out.partial nscf.out
    grep -q "JOB DONE" nscf.out
    date -Is > NSCF_DONE
}

run_epw_seed() {
    if [[ -s EPW_SEED_DONE ]] && grep -q "Total program execution" epw_seed.out; then
        echo "[matched] EPW seed already complete"
        return
    fi
    local ef fine bands_skipped_line dis_win_min_line dis_win_max_line dis_froz_line
    ef="$(awk '/the Fermi energy is/{value=$(NF-1)} END{print value}' scf.out)"
    test -n "$ef"
    fine=$(( NK < 18 ? 18 : NK ))
    bands_skipped_line=""
    if [[ -n "$BANDS_SKIPPED" ]]; then
        bands_skipped_line=" bands_skipped='$BANDS_SKIPPED',"
    fi
    dis_win_min_line=""
    dis_win_max_line=""
    dis_froz_line=""
    [[ -n "$DIS_WIN_MIN" ]] && dis_win_min_line=" dis_win_min=$DIS_WIN_MIN,"
    [[ -n "$DIS_WIN_MAX" ]] && dis_win_max_line=" dis_win_max=$DIS_WIN_MAX,"
    if [[ -n "$DIS_FROZ_MIN" ]]; then
        dis_froz_line=" dis_froz_min=$DIS_FROZ_MIN, dis_froz_max=$DIS_FROZ_MAX,"
    fi
    cat > epw_seed.in <<EOF
--
&inputepw
 prefix='graphene', outdir='./tmp',
 elph=.true., epbwrite=.true., epwwrite=.true.,
 wannierize=.true., nbndsub=2, num_iter=400, proj(1)='C:pz',
$bands_skipped_line
$dis_win_min_line
$dis_win_max_line
$dis_froz_line
 phonselfen=.true., a2f=.false., elecselfen=.false.,
 fsthick=6.0, degaussw=0.05, nsmear=1, delta_smear=0.01,
 efermi_read=.true., fermi_energy=$ef,
 dvscf_dir='./save',
 nk1=$NK, nk2=$NK, nk3=1, nq1=$NQ, nq2=$NQ, nq3=1,
 nkf1=$fine, nkf2=$fine, nkf3=1, nqf1=$fine, nqf2=$fine, nqf3=1
/
EOF
    echo "[matched] Wannier/EPW seed k=$NK q=$NQ start=$(date -Is)"
    set +e
    mpirun --allow-run-as-root -np 1 epw.x -in epw_seed.in > epw_seed.out.partial 2>&1
    local status=$?
    set -e
    mv epw_seed.out.partial epw_seed.out
    [[ "$status" -eq 0 ]]
    grep -q "Total program execution" epw_seed.out
    for required in epwdata.fmt dmedata.fmt vmedata.fmt crystal.fmt graphene.ukk graphene.kmap graphene.kgmap tmp/graphene.epmatwp; do
        test -s "$required"
    done
    date -Is > EPW_SEED_DONE
}

run_static_gate() {
    [[ "$RUN_STATIC_GATE" == "1" ]] || return
    if [[ -s STATIC_GATE_DONE ]] && grep -q "Total program execution" gate_450K/epw.out; then
        echo "[matched] 450 K static gate already complete"
        return
    fi
    local ef source destination bands_skipped_line
    ef="$(awk '/the Fermi energy is/{value=$(NF-1)} END{print value}' scf.out)"
    bands_skipped_line=""
    if [[ -n "$BANDS_SKIPPED" ]]; then
        bands_skipped_line=" bands_skipped='$BANDS_SKIPPED',"
    fi
    mkdir -p gate_450K/tmp
    for source in \
        epwdata.fmt dmedata.fmt vmedata.fmt crystal.fmt decay.H decay.P \
        decay.dynmat decay.epmate decay.epmatp decay.r decay.v \
        graphene.ukk graphene.kmap graphene.kgmap; do
        test -s "$source"
        destination="gate_450K/$(basename "$source")"
        [[ -s "$destination" ]] || cp -p "$source" "$destination"
    done
    [[ -s gate_450K/tmp/graphene.epmatwp ]] || cp -p tmp/graphene.epmatwp gate_450K/tmp/graphene.epmatwp
    cp -p "$ROOT/configs/graphene_e0_epw_pilot/qpoints_GK9_k360.dat" gate_450K/qpoints.dat
    cat > gate_450K/epw.in <<EOF
--
&inputepw
 prefix='graphene', outdir='./tmp',
 elph=.true., ep_coupling=.true., epwread=.true., epwwrite=.false.,
 wannierize=.false., nbndsub=2,
$bands_skipped_line
 phonselfen=.true., specfun_ph=.true.,
 wmin_specfun=0.0, wmax_specfun=0.001, nw_specfun=2,
 a2f=.false., elecselfen=.false.,
 efermi_read=.true., fermi_energy=$ef,
 fsthick=6.0, degaussw=0.005, nsmear=1, delta_smear=0.005,
 temps(1)=450.0, etf_mem=1, iverbosity=2,
 filqf='./qpoints.dat',
 nk1=$NK, nk2=$NK, nk3=1, nq1=$NQ, nq2=$NQ, nq3=1,
 nkf1=720, nkf2=720, nkf3=1, nqf1=1, nqf2=1, nqf3=1
/
EOF
    echo "[matched] 450 K static nine-point gate start=$(date -Is)"
    set +e
    (
        cd gate_450K
        mpirun --allow-run-as-root -np 1 epw.x -in epw.in > epw.out.partial 2>&1
    )
    local status=$?
    set -e
    mv gate_450K/epw.out.partial gate_450K/epw.out
    [[ "$status" -eq 0 ]]
    grep -q "Total program execution" gate_450K/epw.out
    date -Is > STATIC_GATE_DONE
}

write_parameters
run_scf
run_ph
gather_dvscf
run_nscf
date -Is > PREP_DONE

if [[ "$MODE" == "full" ]]; then
    run_epw_seed
    run_static_gate
    date -Is > DONE
fi

echo "[matched] mode=$MODE NK=$NK NQ=$NQ complete=$(date -Is)"
