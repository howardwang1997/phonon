# phonon — GPU-accelerated phonon spectra (MLIP benchmark + DFT acceleration)

Goal: compute phonon spectra **≥50× faster** than conventional DFPT/DFT, with
controlled accuracy, toward a publishable high-throughput phonon framework.

Two research lines share one backend-agnostic phonon engine
(`src/phonon_accel/phonons.py`):

| Line | What | Speed | The science |
|------|------|-------|-------------|
| **A — MLIP benchmark + fine-tune** | foundation MLIPs (MACE, MatterSim, SevenNet, ORB, eSEN, CHGNet) compute forces via `phonopy` | 50–1000× "for free" | **accuracy**: fix softening / imaginary modes / ASR violations by fine-tuning |
| **B — GPU-DFT workflow acceleration** | Quantum ESPRESSO GPU finite-displacement, staged | ~50× (B1), >100× (B2 stretch) | GPU SCF × cross-displacement density reuse × symmetry/non-diagonal supercells |

Key reframe from the literature survey: **pure GPU porting of DFT only gives
~5–15×**; the 50×+ target is reached by MLIP surrogates (Line A) and by
workflow-level acceleration of the finite-displacement DFT pipeline (Line B).

## Status

- [x] M0 — env + unified pipeline validated on Si (`scripts/smoke_si.py`)
- [x] A2 — genuine DFPT reference (MDR, 10,034 materials) wired in; full-band
  benchmark harness + first real results (MatterSim, 14 crystals)
- [ ] A2 — scale to more models (own envs) + larger MDR sample
- [ ] M0b — remote GPU `100.105.21.7` probe (currently unreachable from dev box)
- [ ] B1 — QE-GPU workflow acceleration
- [ ] A3 — fine-tuning to fix failure modes

**First results (Line A).** Full-band MLIP-vs-DFPT (MDR/PhononDB reference):

| model | freq MAE | mean softening | dyn. stable |
|-------|---------:|---------------:|:-----------:|
| MatterSim-v1 | **0.58 THz** (14 crystals) | −7.2% | 14/14 |
| MACE-MP-0 (2023) | ~1–4 THz (softening-dominated) | −26% | — |

MatterSim tracks DFPT acoustic branches almost exactly; optical branches mildly
softened. MACE-MP-0 softens ~4× more — the gap Line A's fine-tuning targets.
See `results/dfpt_mattersim_curated.csv`, `results/figures/`, `docs/findings.md`.

### Environments (per-model, to avoid e3nn conflicts)
- `phonon` — MatterSim + core stack (workhorse / fine-tuning)
- `phonon-mace` — MACE (pinned `e3nn==0.4.4`), isolated baseline

The pipeline code is env-agnostic (lazy model import); run the same scripts
under whichever env has the target model.

## Layout

```
src/phonon_accel/
  phonons.py      # core: structure -> displacements -> forces -> FC -> bands/DOS/thermal
  mlip_calc.py    # unified ASE-calculator factory for foundation MLIPs (lazy imports)
  structures.py   # built-in cells (Si, ...), Materials Project, file, relax()
  metrics.py      # freq MAE/RMSE, imaginary-mode count, ASR residual, thermal errors
  benchmark.py    # Line A: model x material -> metrics table
scripts/          # smoke_si.py, run_benchmark.py
```

## Quickstart

```bash
conda env create -f environment.yml          # or: conda install -n phonon python=3.11 && pip install ...
conda run -n phonon python scripts/smoke_si.py --model mace --device cpu
```

On the GPU box use `--device cuda` (float64 is required for force constants;
Apple MPS does not support float64, so the local dev box runs on CPU).

See `docs/`/the plan file for the full research design and milestones.

Current graphene plan: [physical-FD 可迁移预测模型实验计划](docs/GRAPHENE_FD_TRANSFERABILITY_PLAN.md).

## Weekly reports

- [2026-07-29 — graphene 600 K force gate 与有限温度长程项](docs/WEEKLY_2026-07-29.md)
- [2026-07-22 — graphene q-space 盲测结果与 `(L)` 通道补充](docs/WEEKLY_2026-07-22.md)
- [2026-07-15 — 主要关注石墨烯：Kohn 反常的 DFT 盲测与声子谱复现](docs/WEEKLY_2026-07-15.md)
