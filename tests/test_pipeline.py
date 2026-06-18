import numpy as np

from phonongpu.structure import Structure
from phonongpu.pipeline import PhononPipeline
from phonongpu.scheduler import MultiGPUExecutor
from phonongpu.displacements import flatten_evals
from phonongpu.backends import QEGPUBackend, VASPGPUBackend, AnalyticForceBackend


def _diamond_conv(a=5.43):
    cell = a * np.eye(3)
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    return Structure(cell, np.vstack([fcc, fcc + 0.25]), ["Si"] * 8)


def test_pipeline_analytic():
    prim = _diamond_conv()
    pipe = PhononPipeline(prim, supercell_scale=2, backend="analytic", verbose=False)
    res = pipe.run()
    g = np.sort(res.gamma_frequencies)
    assert np.max(np.abs(g[:3])) < 1e-6
    assert res.force_constants.shape == (3 * res.supercell.natoms,) * 2


def test_executor_detects_gpus():
    ex = MultiGPUExecutor()
    assert ex.num_gpus >= 1
    assert ex.max_workers >= 1


def test_qe_dry_run_inputs(tmp_path):
    prim = _diamond_conv()
    sc = prim.make_supercell(2)
    from phonongpu.displacements import displacement_plan
    plan = displacement_plan(sc, distance=0.01)
    evals = flatten_evals(plan)
    qe = QEGPUBackend(pseudos={"Si": "Si.UPF"},
                      options={"workdir": str(tmp_path), "dry_run": True})
    F = qe.evaluate(sc, evals)
    assert F.shape == (len(evals), sc.natoms, 3)
    import os
    assert len(os.listdir(tmp_path)) == len(evals)
    inp = os.path.join(str(tmp_path), "disp_00000", "scf.in")
    assert "tprnfor" in open(inp).read()


def test_qe_parse_forces():
    qe = QEGPUBackend()
    sample = (
        "     number of atoms/cell =            8\n"
        "     atom    1 type  1   force =     0.1000   -0.2000    0.3000\n"
        "     atom    2 type  1   force =     1.0000    2.0000    3.0000\n"
    )
    p = "/tmp/_qe_parse_test.txt"
    open(p, "w").write(sample)
    f = qe.parse_forces(p)
    import numpy as np
    assert f.shape == (2, 3)
    assert np.allclose(f[0], [0.1, -0.2, 0.3])
    assert np.allclose(f[1], [1.0, 2.0, 3.0])


def test_vasp_dry_run_inputs(tmp_path):
    prim = _diamond_conv()
    sc = prim.make_supercell(2)
    from phonongpu.displacements import displacement_plan
    plan = displacement_plan(sc, distance=0.01)
    evals = flatten_evals(plan)
    vasp = VASPGPUBackend(options={"workdir": str(tmp_path), "dry_run": True})
    F = vasp.evaluate(sc, evals)
    assert F.shape == (len(evals), sc.natoms, 3)
    import os, glob
    assert len(glob.glob(str(tmp_path / "disp_*" / "INCAR"))) == len(evals)
