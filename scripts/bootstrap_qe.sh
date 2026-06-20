#!/usr/bin/env bash
# Install Quantum ESPRESSO (CPU, conda-forge) + ASE/phonopy in a dedicated `dft`
# env on the 192-core H20 box, and fetch SSSP-efficiency pseudopotentials.
# Logs to ~/bootstrap_qe.log. China mirror for pip; conda-forge for QE.
set -uo pipefail
CONDA=$HOME/miniconda3
ENV=dft
echo "=== [$(date)] QE bootstrap start ==="
if [ ! -x "$CONDA/envs/$ENV/bin/pw.x" ]; then
  $CONDA/bin/conda create -n $ENV -c conda-forge --override-channels -y \
    "qe>=7.2" ase phonopy spglib numpy scipy
fi
PW=$CONDA/envs/$ENV/bin/pw.x
echo "pw.x: $($PW -h 2>&1 | head -1 || echo MISSING)"
$CONDA/envs/$ENV/bin/python -c "import ase, phonopy; print('ase', ase.__version__, 'phonopy ok')"

# SSSP efficiency pseudopotentials (PBE) -> ~/pseudo
PSDIR=$HOME/pseudo
mkdir -p "$PSDIR"
if [ ! -f "$PSDIR/Si.pbe-n-rrkjus_psl.1.0.0.UPF" ] && [ -z "$(ls "$PSDIR"/*.UPF 2>/dev/null)" ]; then
  cd "$PSDIR"
  # SSSP efficiency PBE bundle (small, ~all elements)
  curl -sL -o sssp.tar.gz "https://archive.materialscloud.org/record/file?record_id=1732&filename=SSSP_1.3.0_PBE_efficiency.tar.gz" 2>/dev/null || \
  curl -sL -o Si.UPF "https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF"
  [ -f sssp.tar.gz ] && tar xzf sssp.tar.gz 2>/dev/null
  echo "pseudo files: $(ls *.UPF 2>/dev/null | wc -l)"
fi
echo "=== [$(date)] QE bootstrap done ==="
