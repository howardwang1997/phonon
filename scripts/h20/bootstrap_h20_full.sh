#!/usr/bin/env bash
# =============================================================================
# Self-healing bootstrap for the 8×H20 campaign.
# Detects what is present and installs ONLY what is missing. Never aborts the
# campaign: a model env that fails to build just means its jobs get skipped.
# Writes a human-readable summary to results/h20/BOOTSTRAP_STATUS.txt and lists
# anything that must be relayed by hand (no SSH from the author -> GitHub only).
#
# Envs built (each conda env carries its model + the phonopy/ase stack):
#   phonon    MACE-MP/OMAT  (the base; built by scripts/bootstrap_h20.sh)
#   sevennet  SevenNet
#   mattersim MatterSim
#   orb       ORB
#   chgnet    CHGNet
#   sscha14   cellconstructor 1.4.1 + python-sscha 1.4.1 (the (L)-channel SSCHA)
# =============================================================================
set -uo pipefail
cd "$HOME/phonon"
CONDA="$HOME/miniconda3"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
H=results/h20; mkdir -p "$H"
STATUS="$H/BOOTSTRAP_STATUS.txt"
: > "$STATUS"
log() { echo "$@" | tee -a "$STATUS"; }

cf() { "$CONDA/bin/conda" create -y -n "$1" "python=$2" -c conda-forge --override-channels; }
penv() { echo "$CONDA/envs/$1/bin/python"; }
have() { [ -x "$(penv "$1")" ]; }
imp()  { "$(penv "$1")" -c "$2" >/dev/null 2>&1; }   # imp ENV "import x"

log "=== H20 bootstrap $(date) ==="

# --------------------------------------------------------------------------- #
# 1. base phonon (MACE) env — delegate to the existing recipe, then top up
# --------------------------------------------------------------------------- #
if imp phonon "import mace, phonopy, ase"; then
  log "[phonon]   present (MACE stack ok)"
else
  log "[phonon]   building via scripts/bootstrap_h20.sh ..."
  bash scripts/bootstrap_h20.sh >>"$STATUS" 2>&1 || log "[phonon]   WARNING: bootstrap_h20.sh returned nonzero"
fi
# extras the campaign needs in the base env
PHPY="$(penv phonon)"
imp phonon "import yaml"   || "$PHPY" -m pip install pyyaml   -i "$MIRROR" >>"$STATUS" 2>&1
imp phonon "import h5py"   || "$PHPY" -m pip install h5py     -i "$MIRROR" >>"$STATUS" 2>&1
imp phonon "import phono3py" || "$PHPY" -m pip install phono3py -i "$MIRROR" >>"$STATUS" 2>&1
imp phonon "import mace, phonopy, phono3py, ase, yaml, h5py" \
  && log "[phonon]   OK (mace+phonopy+phono3py+ase+yaml+h5py)" \
  || log "[phonon]   INCOMPLETE — check $STATUS"

# --------------------------------------------------------------------------- #
# 2. per-model envs (light: torch + model + phonopy + ase + pandas)
# --------------------------------------------------------------------------- #
build_model_env() {           # $1 env  $2 import-probe  $3 pip-spec
  local env="$1" probe="$2" spec="$3" py
  if imp "$env" "$probe"; then log "[$env]   present"; return; fi
  log "[$env]   building ($spec) ..."
  have "$env" || cf "$env" 3.11 >>"$STATUS" 2>&1
  py="$(penv "$env")"
  "$py" -m ensurepip --upgrade >>"$STATUS" 2>&1 || true
  "$py" -m pip install --upgrade pip -i "$MIRROR" >>"$STATUS" 2>&1
  "$py" -m pip install torch==2.6.0 -i "$MIRROR" >>"$STATUS" 2>&1
  "$py" -m pip install $spec phonopy ase pandas numpy -i "$MIRROR" >>"$STATUS" 2>&1
  if imp "$env" "$probe"; then
    log "[$env]   OK ($("$py" -c 'import torch;print("cuda",torch.cuda.is_available())' 2>/dev/null))"
  else
    log "[$env]   FAILED — its E1 jobs will be skipped (campaign continues)"
  fi
}

build_model_env sevennet  "import sevenn, phonopy, ase"      "sevenn"
build_model_env mattersim "import mattersim, phonopy, ase"   "mattersim"
build_model_env orb       "import orb_models, phonopy, ase"  "orb-models"
build_model_env chgnet    "import chgnet, phonopy, ase"      "chgnet"

# --------------------------------------------------------------------------- #
# 3. sscha14 env (the finicky (L)-channel SSCHA — exact pins from the gotchas)
# --------------------------------------------------------------------------- #
if imp sscha14 "import cellconstructor, sscha, mace, phonopy, ase"; then
  log "[sscha14]  present"
else
  log "[sscha14]  building (cellconstructor 1.4.1 + python-sscha 1.4.1) ..."
  have sscha14 || cf sscha14 3.10 >>"$STATUS" 2>&1
  SPY="$(penv sscha14)"
  # CC's f90 extension needs lapack + a fortran compiler at build time
  "$CONDA/bin/conda" install -y -n sscha14 -c conda-forge --override-channels \
    fortran-compiler liblapack libblas lapack openblas >>"$STATUS" 2>&1 || true
  "$SPY" -m ensurepip --upgrade >>"$STATUS" 2>&1 || true
  "$SPY" -m pip install --upgrade pip -i "$MIRROR" >>"$STATUS" 2>&1
  export LIBRARY_PATH="$CONDA/envs/sscha14/lib:${LIBRARY_PATH:-}"
  "$SPY" -m pip install torch==2.6.0 -i "$MIRROR" >>"$STATUS" 2>&1
  # pinned chain: numpy 1.23 (CC symph f90), spglib<2, phonopy 2.18, mace 0.3.16
  "$SPY" -m pip install "numpy==1.23" "spglib==1.16" "phonopy==2.18" \
    "mace-torch==0.3.16" ase -i "$MIRROR" >>"$STATUS" 2>&1
  "$SPY" -m pip install "cellconstructor==1.4.1" "python-sscha==1.4.1" \
    -i "$MIRROR" >>"$STATUS" 2>&1
  if imp sscha14 "import cellconstructor, sscha, mace, phonopy, ase"; then
    log "[sscha14]  OK"
  else
    log "[sscha14]  FAILED — SSCHA jobs skipped; harmonic triage still runs"
  fi
fi

# --------------------------------------------------------------------------- #
# 4. data: E1 MDR pool + anti-forgetting replay file
# --------------------------------------------------------------------------- #
POOL=data/benchmark/e1_pool.txt
NPOOL=$(awk '{ for(i=1;i<=NF;i++) if($i ~ /^mp-/) n++ } END{print n+0}' "$POOL" 2>/dev/null || echo 0)
WANT=$("$PHPY" -c "import yaml;print(yaml.safe_load(open('configs/h20_campaign.yaml'))['e1']['pool_size'])" 2>/dev/null || echo 200)
if [ "${NPOOL:-0}" -ge "$WANT" ]; then
  log "[data]     E1 pool present ($NPOOL mp-ids)"
else
  log "[data]     building stratified E1 pool (target $WANT; downloads MDR DFPT, resumable) ..."
  "$PHPY" - "$WANT" "$POOL" <<'PY' >>"$STATUS" 2>&1 || log "[data]     pool build had errors (partial pool usable)"
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "src")
from phonon_accel import reference
want, outp = int(sys.argv[1]), Path(sys.argv[2])
rows = list(reference._index().values())
cached = {p.name.split("_")[0] for p in reference.CACHE_DIR.glob("*_phonopy_params.yaml")}
by_sg = defaultdict(list)
for r in rows:
    if r["mp_id"] in cached: continue
    by_sg[r["spacegroup_number"]].append(r)
for sg in by_sg: by_sg[sg].sort(key=lambda r:(len(r["formula"]), r["formula"]))
order, sgs, i = [], sorted(by_sg, key=lambda s:int(s)), 0
while any(by_sg[s] for s in sgs):
    s = sgs[i % len(sgs)]
    if by_sg[s]: order.append(by_sg[s].pop(0))
    i += 1
ok = sorted(cached)
for r in order:
    if len(ok) >= want: break
    try:
        reference.fetch(r["mp_id"]); ok.append(r["mp_id"])
        print(f"  OK {r['mp_id']} {r['formula']} ({len(ok)}/{want})", flush=True)
    except Exception as e:
        print(f"  FAIL {r['mp_id']} {str(e)[:40]}", flush=True)
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text("\n".join(sorted(set(ok))) + "\n")
print(f"pool now {len(set(ok))} -> {outp}")
PY
  NPOOL=$(awk '{ for(i=1;i<=NF;i++) if($i ~ /^mp-/) n++ } END{print n+0}' "$POOL" 2>/dev/null || echo 0)
  log "[data]     E1 pool now $NPOOL mp-ids"
fi

REPLAY="$HOME/.cache/mace/mp_traj_combinedxyz"
if [ -f "$REPLAY" ]; then
  log "[data]     MACE replay present ($(du -h "$REPLAY" | cut -f1))"
else
  log "[data]     MACE replay MISSING -> multihead (pt*) fine-tunes will try to"
  log "           auto-download; if the link is blocked, RELAY this file by hand:"
  log "           $REPLAY   (~56 MB, gitignored)"
fi

# --------------------------------------------------------------------------- #
# 5. summary
# --------------------------------------------------------------------------- #
log ""
log "=== summary $(date) ==="
for e in phonon sevennet mattersim orb chgnet sscha14; do
  if   [ "$e" = sscha14 ]; then P="import cellconstructor, sscha, mace";
  elif [ "$e" = phonon  ]; then P="import mace, phono3py, phonopy";
  else P="import phonopy, ase"; fi
  imp "$e" "$P" && log "  env $e: READY" || log "  env $e: NOT READY (its jobs will be skipped)"
done
log "  E1 pool: ${NPOOL:-0} materials"
log "=== bootstrap end ==="
echo "[bootstrap] summary -> $STATUS"
