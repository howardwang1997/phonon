"""Test whether this cellconstructor/python-sscha version can build a CC dyn from
the NbSe2 DFT fc2 phonopy yaml (CC 1.6.2 SIGFPE-core-dumps; testing an older pin).
Tries load_phonopy first (may work in CC 1.4), then the tensor route."""
import sys
import numpy as np
if not hasattr(np, "int"):
    np.int = int
    np.float = float
RY_TO_CM = 109736.75
yaml = "results/vq3/nbse2_dft_phonopy.yaml"

import cellconstructor as CC
import cellconstructor.Phonons

# Route 1: load_phonopy directly
try:
    dyn = CC.Phonons.Phonons()
    dyn.load_phonopy(yaml)
    dyn.ForcePositiveDefinite()
    w, _ = dyn.DiagonalizeSupercell()
    print("ROUTE1 load_phonopy OK: supercell", dyn.GetSupercell(),
          "minfreq_cm", round(float(w.min()) * RY_TO_CM, 1))
    sys.exit(0)
except SystemExit:
    raise
except Exception as e:
    print("ROUTE1 load_phonopy failed:", repr(e)[:120])

# Route 2: phonopy_fc2_to_tensor2 -> Tensor2 -> GeneratePhonons
try:
    import phonopy
    import cellconstructor.Methods
    import cellconstructor.ForceTensor
    import cellconstructor.Structure
    from ase import Atoms
    ph = phonopy.load(yaml)
    prim = ph.primitive
    uc = Atoms(numbers=prim.numbers, scaled_positions=prim.scaled_positions,
               cell=prim.cell, pbc=True)
    struc = CC.Structure.Structure()
    struc.generate_from_ase_atoms(uc)
    res = CC.Methods.phonopy_fc2_to_tensor2(ph.force_constants, ph)
    tens = res[0] if isinstance(res, tuple) else res
    scmat = np.ascontiguousarray(ph.supercell_matrix, dtype=np.intc)
    sc = struc.generate_supercell(scmat)
    t2 = CC.ForceTensor.Tensor2(struc, sc, np.array(ph.supercell_matrix))
    t2.SetupFromTensor(tens)
    dyn = t2.GeneratePhonons(np.array(ph.supercell_matrix))
    dyn.ForcePositiveDefinite()
    w, _ = dyn.DiagonalizeSupercell()
    print("ROUTE2 tensor OK: supercell", dyn.GetSupercell(),
          "minfreq_cm", round(float(w.min()) * RY_TO_CM, 1))
except Exception as e:
    import traceback
    print("ROUTE2 failed:", repr(e)[:120])
    traceback.print_exc()
