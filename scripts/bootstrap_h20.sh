#!/usr/bin/env bash
# Bootstrap the phonon conda env on a fresh H20 box (China mirrors -> fast).
# Idempotent-ish: safe to re-run. Logs to ~/bootstrap.log.
set -uo pipefail
CONDA=$HOME/miniconda3
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
ENV=phonon
PY=$CONDA/envs/$ENV/bin/python

echo "=== [$(date)] bootstrap start ==="
# 1. create env (conda-forge override to dodge ToS prompt)
if [ ! -x "$PY" ]; then
  $CONDA/bin/conda create -n $ENV python=3.11 -c conda-forge --override-channels -y
  # conda-forge python may ship without pip
  $PY -m ensurepip --upgrade 2>/dev/null || true
fi
$PY -m pip install --upgrade pip -i $MIRROR

# 2. torch 2.6.0 (PyPI linux wheel == cu124 build) from Tsinghua mirror
$PY -c "import torch" 2>/dev/null || $PY -m pip install torch==2.6.0 -i $MIRROR

# 3. MLIP + phonon stack. MACE needs e3nn==0.4.4 (pin BEFORE mace-torch).
$PY -m pip install "e3nn==0.4.4" -i $MIRROR
$PY -m pip install mace-torch phonopy phono3py ase pymatgen spglib seekpath pandas -i $MIRROR

# 4. verify
echo "=== verify ==="
$PY -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"
$PY -c "import mace, e3nn, phonopy, ase, pymatgen; print('mace ok, e3nn', e3nn.__version__)"
ls -la $CONDA/envs/$ENV/bin/mace_run_train && echo "mace_run_train present"
echo "=== [$(date)] bootstrap done ==="
