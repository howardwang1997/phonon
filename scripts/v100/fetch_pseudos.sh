#!/usr/bin/env bash
# Phase-0 enabler: fetch ONCV-PBE (SG15) pseudopotentials for the TMD family.
# Nb/Se are usually already present; this adds Ta/Ti/V/S/Mo/W. Idempotent:
# skips elements already resolvable by glob. Tries the SG15 individual-UPF host;
# logs anything that fails so you can relay it by hand.
#   bash scripts/v100/fetch_pseudos.sh [PSEUDO_DIR]
set -uo pipefail
PSEUDO="${1:-/root/phonon/pseudo}"
mkdir -p "$PSEUDO"
ELEMENTS=(Nb Se S Ta Ti V Mo W)
# SG15 ONCV PBE individual UPFs. Version suffix varies per element -> try a few.
BASE="http://www.quantum-simulation.org/potentials/sg15_oncv/upf"
VERS=(1.2 1.1 1.0)
echo "=== fetch_pseudos -> $PSEUDO  $(date) ==="

have() { ls "$PSEUDO/$1"_ONCV_PBE*.upf >/dev/null 2>&1 || ls "$PSEUDO/$1"_ONCV*.upf >/dev/null 2>&1; }

miss=()
for el in "${ELEMENTS[@]}"; do
  if have "$el"; then echo "[$el] present: $(basename "$(ls "$PSEUDO/$el"_ONCV*.upf | head -1)")"; continue; fi
  got=""
  for v in "${VERS[@]}"; do
    f="${el}_ONCV_PBE-${v}.upf"
    if curl -fsSL "$BASE/$f" -o "$PSEUDO/$f" 2>/dev/null && [ -s "$PSEUDO/$f" ]; then
      echo "[$el] downloaded $f"; got=1; break
    else rm -f "$PSEUDO/$f"; fi
  done
  [ -z "$got" ] && { echo "[$el] FAILED to download"; miss+=("$el"); }
done

echo "=== summary ==="
for el in "${ELEMENTS[@]}"; do have "$el" && echo "  $el OK" || echo "  $el MISSING"; done
if [ "${#miss[@]}" -gt 0 ]; then
  echo "!! could not fetch: ${miss[*]}"
  echo "   RELAY these by hand into $PSEUDO/ (any ONCV-PBE .upf, named <El>_ONCV_PBE-*.upf),"
  echo "   e.g. from PseudoDojo (http://www.pseudo-dojo.org) ONCV PBE standard."
  exit 1
fi
echo "=== all TMD-family pseudos present ==="
