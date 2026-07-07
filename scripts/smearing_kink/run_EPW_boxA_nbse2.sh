#!/usr/bin/env bash
# Box A: WAIT for (E)-breadth fc2 (Ebreadth tmux) to finish, then run the tested
# NbSe2 EPW queue (degauss 0.03/0.04 -> lambda(T_el) first-principles arbiter).
# Family EPW stage 1: NbSe2 (script is NbSe2-hardcoded; generalize to other mats later).
set -u
cd /root/phonon
echo "[epwA] $(date) waiting for Ebreadth fc2 to finish ..."
while tmux has-session -t Ebreadth 2>/dev/null; do sleep 120; done
echo "[epwA] $(date) Ebreadth done -> NbSe2 EPW queue (degauss 0.03, 0.04)"
bash scripts/epw/queue_boxA.sh 2>&1
echo "[epwA] NBSE2 EPW DONE $(date)"
