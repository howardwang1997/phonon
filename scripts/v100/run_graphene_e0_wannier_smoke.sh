#!/usr/bin/env bash
# Wannier-only smoke test for the graphene pi manifold.  It reads a completed
# QE NSCF outdir but does not transform or write electron-phonon matrices.
set -euo pipefail

WORK="${WORK:-/data/graphene_e0_epw_matched/wannier_smoke_k18q9_pifroz}"
SOURCE_OUTDIR="${SOURCE_OUTDIR:-/data/graphene_e0_epw_matched/k18_q9_pi45/tmp}"
SOURCE_SCF="${SOURCE_SCF:-/data/graphene_e0_epw_matched/k18_q9/scf.out}"
NK="${NK:-18}"
NQ="${NQ:-9}"
BANDS_SKIPPED="${BANDS_SKIPPED:-exclude_bands = 1-3}"
DIS_FROZ_MIN="${DIS_FROZ_MIN:--2.2187}"
DIS_FROZ_MAX="${DIS_FROZ_MAX:--1.2187}"
DIS_WIN_MIN="${DIS_WIN_MIN:-}"
DIS_WIN_MAX="${DIS_WIN_MAX:-}"

test -d "$SOURCE_OUTDIR/graphene.save"
test -s "$SOURCE_SCF"
source /root/miniconda3/etc/profile.d/conda.sh
if [[ -x /root/miniconda3/envs/qe/bin/epw.x ]]; then
    conda activate qe
else
    conda activate phonon
fi
command -v epw.x >/dev/null

mkdir -p "$WORK"
cd "$WORK"
if [[ -s DONE ]] && grep -q "Total program execution" epw_wannier.out; then
    echo "[wannier-smoke] already complete"
    exit 0
fi

FERMI="$(awk '/the Fermi energy is/{value=$(NF-1)} END{print value}' "$SOURCE_SCF")"
test -n "$FERMI"
dis_win_min_line=""
dis_win_max_line=""
[[ -n "$DIS_WIN_MIN" ]] && dis_win_min_line=" dis_win_min=$DIS_WIN_MIN,"
[[ -n "$DIS_WIN_MAX" ]] && dis_win_max_line=" dis_win_max=$DIS_WIN_MAX,"
cat > epw_wannier.in <<EOF
--
&inputepw
 prefix='graphene', outdir='$SOURCE_OUTDIR',
  elph=.false., ep_coupling=.true.,
 epbwrite=.false., epwwrite=.true., epwread=.false.,
 wannierize=.true., nbndsub=2, num_iter=400, proj(1)='C:pz',
 bands_skipped='$BANDS_SKIPPED',
$dis_win_min_line
$dis_win_max_line
 dis_froz_min=$DIS_FROZ_MIN, dis_froz_max=$DIS_FROZ_MAX,
 wdata(1)='dis_num_iter = 1000',
 wdata(2)='guiding_centres = true',
 phonselfen=.false., a2f=.false., elecselfen=.false.,
 efermi_read=.true., fermi_energy=$FERMI,
 fsthick=6.0, degaussw=0.05,
 nk1=$NK, nk2=$NK, nk3=1, nq1=$NQ, nq2=$NQ, nq3=1,
 nkf1=1, nkf2=1, nkf3=1, nqf1=1, nqf2=1, nqf3=1
/
EOF

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "[wannier-smoke] start=$(date -Is) NK=$NK frozen=[$DIS_FROZ_MIN,$DIS_FROZ_MAX] eV"
set +e
mpirun --allow-run-as-root -np 1 epw.x -in epw_wannier.in > epw_wannier.out.partial 2>&1
status=$?
set -e
mv epw_wannier.out.partial epw_wannier.out
[[ "$status" -eq 0 ]]
grep -q "Total program execution" epw_wannier.out
test -s graphene_hr.dat
date -Is > DONE
echo "[wannier-smoke] complete=$(date -Is)"
