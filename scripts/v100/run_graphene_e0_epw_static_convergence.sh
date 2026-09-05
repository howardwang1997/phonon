#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
"$ROOT/scripts/v100/run_graphene_e0_epw_restart_pilot.sh" static360
"$ROOT/scripts/v100/run_graphene_e0_epw_restart_pilot.sh" static005
