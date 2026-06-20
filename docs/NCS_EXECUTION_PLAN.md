# NCS Execution Plan — compute, data, progress, and the path to submission

Central planning + status doc. Companion to `NCS_ROADMAP.md` (framing), `CAMPAIGN_FINDINGS.md`
(Line A results), `LINE_B_FINDINGS.md` (Line B results). Updated 2026-06-20.

**Core principle (the whole thesis in one line):** *small DFT + large MLIP.* The expensive DFT is a
small training/validation set; the fine-tuned MLIP does the 10³× high-throughput at scale. So almost
everything runs on **public data + the machines we already have**; renting GPUs is needed only to
*substantiate the GPU-SCF factor* and *speed up a small anharmonic validation set*.

---

## 1. 算力需求 (compute requirements)

### 1a. No-rental — runs on current hardware
| machine | role | what runs here |
|---|---|---|
| **2×H20 (100.80.123.104)** 192 cores | CPU-DFT engine | broad harmonic self-DFT set, DFPT cross-check, small 3rd-order DFT, AL-loop DFT |
| **8×H20 (100.91.194.14)** 8 GPU + 192 cores | MLIP fleet + CPU-DFT | MLIP fine-tuning, scale-out MLIP phonon/κ inference, 3rd-order FC distillation, extra CPU-DFT |
| **2060 (100.105.21.7)** 1 GPU (desktop) | MLIP inference only | baseline benchmark, eval (DFT is hopeless here — desktop CPU ~850 s/SCF-iter) |

H20 FP64 is crippled (~1 TFLOP) → H20 is a CPU-DFT box + an MLIP-GPU box, **not** a DFT-GPU box.

### 1b. Rental — only for two things
| task | why rental | GPU-hours | on 8×V100 |
|---|---|---|---|
| **GPU-SCF factor** | needs strong FP64 (V100 ~7 TFLOPS vs H20 ~1) | ~30–50 | core |
| **anharmonic (3rd-order) self-DFT** | 10–50× harmonic cost; V100 faster | ~150–3000 (scales with #materials) | the long pole |

**V100 vs H20 for QE-GPU SCF:** V100 ≈ **3–7× faster** (FP64-dominated; H20's bandwidth helps FFTs but
loses on diagonalization). A100/A800 ≈ 5–10× H20.

**Rental sizing (8×V100, embarrassingly parallel → wall = GPU-hr ÷ 8):**
- **Lean (~2–3 days):** GPU-SCF demo + a few κ spot-checks (everything else public + MLIP).
- **Core week (~5–7 days, ~600–1000 GPU-hr):** GPU-SCF + harmonic set + small κ validation → *submittable*.
- **Competitive month (~5760 GPU-hr):** + **κ at scale (30–50 self-DFT materials) + 3rd-order FC
  distillation + real multi-round closed loop + broad harmonic set (300–500 materials)** → *NCS-competitive*.
  This is the sweet spot; beyond a month → diminishing returns. (≈ ¥2000–8000 cloud, China rates.)

---

## 2. 数据需求 (data requirements)

| need | source | public? | use |
|---|---|---|---|
| Harmonic DFPT phonons (10,034) | MDR/PhononDB (Togo/NIMS) | ✅ | benchmark + FC-distillation labels |
| Harmonic DFPT (~1,521) | Petretto 2018 | ✅ | second reference |
| **Lattice κ + 3rd-order FC** | **Togo phono3py DB** | ✅ | **anharmonic validation (avoids most self-DFT κ)** |
| Pre-training trajectories | MPtrj | ✅ | replay anti-forgetting |
| Foundation pseudopotentials | **SG15 ONCV PBE (69 elem)** | ✅ | self-DFT across the periodic table (blocker solved) |
| MP structures (10³–10⁴) | Materials Project | ✅ | scale-out MLIP phonon/κ database |
| Harmonic self-DFT set | self (QE, CPU now) | ❌ | engine validation + training |
| Small anharmonic κ set | self (QE, V100 later) | ❌ | κ benchmark anchor |
| GPU-SCF timing | self (QE-GPU, V100) | ❌ | speedup factor |

**Principle:** the DFT we self-generate stays **small** (validation/anchor); the *scale* comes from
public DBs + the MLIP. No large self-DFT dataset is required.

---

## 3. 现在进展 (current progress) — done

- **Line A laws** (`CAMPAIGN_FINDINGS.md`): breadth > depth for transfer (knee ~20–24, floor ~1.3 THz);
  depth overfits; anti-forgetting (LoRA-r32/replay best); **acquisition: coverage > random > naive
  uncertainty**; transfer floor dominated by BN. In-domain ~0.10 THz, zero imaginary.
- **Line B engine** (`LINE_B_FINDINGS.md`): QE finite-displacement validated (Si ~1–5%); honest speedup
  decomposition — symmetry 48–384× (standard), density+wfc reuse ~1.5× (scales with SCF difficulty),
  **non-diagonal supercells 141–1152× for exact fine-q**, GPU-SCF deferred.
- **Pseudopotential blocker solved:** SG15 ONCV PBE, 69 elements.
- **Loop closure mechanics verified:** self-DFT force constants build valid harmonic distillation configs.

---

## 4. 正在做的实验 (in progress)

- **Broad harmonic self-DFT dataset** (2×H20, 3 workers): diverse chemistry via SG15, validated vs MDR.
- **DFPT (ph.x) cross-check** (2×H20): Si, pending free cores.
- **Loop closure** (2×H20 GPU): armed — auto-trains on self-DFT FCs at ≥8 materials.
- **Foundation-MLIP baseline benchmark** (2060): 109 materials, "before fine-tuning" reference.

---

## 5. 未来需要做的实验

### 5a. No-rental (start now on the 3 machines)
1. **κ via public Togo + MLIP** ★impact: fine-tuned MLIP → phono3py 3rd-order FC → κ_lattice, validate
   vs public Togo κ. *The downstream payoff, with no rental.* (8×H20 GPU + 2×H20 CPU.)
2. **3rd-order FC distillation**: extend FC distillation to anharmonic (distill Togo 3rd-order FC) →
   methodological novelty. (8×H20 GPU.)
3. **Scale-out MLIP phonon/κ database**: fine-tuned MLIP over 10³–10⁴ MP materials → near-DFT phonon +
   stability + κ screening. (8×H20 / 2060 GPU inference.)
4. **Cross-model generality**: MatterSim / SevenNet / ORB benchmark (+ fine-tune where feasible) →
   laws are model-general. (GPU inference.)
5. **Finish**: broad harmonic set, DFPT, loop closure (in progress).
6. **Real closed loop (CPU-DFT)**: AL acquisition → Line-B self-DFT → distill → repeat, on new chemistry.

### 5b. Rental (V100/A100) — when cards are available
7. **GPU-SCF factor** + composed end-to-end ~50× workflow timing.
8. **Anharmonic self-DFT κ benchmark** (~30–50 materials, the competitive-month centerpiece).

### Timeline
- **No-rental program:** ~1–2 weeks on current machines gets κ-via-public + scale-out + cross-model +
  Line-B finish → most of the NCS story without spending on rental.
- **+ Core rental week:** GPU-SCF + κ anchor → submittable.
- **+ Competitive month rental:** κ at scale + 3rd-order distillation + real loop → NCS-competitive.

---

## Honest framing (keep in the paper)
Pure GPU-DFT single-SCF ~5–15× (V100-class); the ~50× is workflow-level (dominated by *standard*
symmetry); the ~10³× is the MLIP proxy. Every speedup reported with scope; every accuracy with seed
error bars; κ reported as mean + median; negative results (e.g. naive-uncertainty acquisition) kept.
