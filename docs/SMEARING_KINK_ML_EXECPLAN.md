# Smearing-Kink-ML — execution plan (sub-line `smearing-kink-ml`)

**中文 TL;DR.** **这是"科学发现"的一部分,不是独立方法**:用 Kohn 反常的 **kink 如何随温度变化** 来**判定晶格不稳定性的起源**。温度有两半:电子 smearing **T_el**((E) 通道)+ 物理/晶格温度 **T_lat**((L) 通道);物理上存在一个 **kink 变化最剧烈的温度窗口**。用 ML 建模、并用**少量数据 finetune** 就能准确预测整条曲线(乃至迁移到新材料)。两个引擎:**(1) ML 加速的 EPW**(物理忠实,尖点由公式给)、**(2) T-条件化自学习 MLIP**(把温度当输入、少样本微调、直接预测 kink(T))。先 **graphene** 定方法,后 **NbSe₂** 出剧烈-transition 科学。**硬约束**:反常是非解析尖点,ML 不直接学尖点 —— (E) 走 ML-EPW / 光滑 emulator,(L) 走曲率监督的蒸馏 MLIP + TDEP。

> **Positioning (decided 2026-07-02) — this sub-line is PART OF THE SCIENTIFIC DISCOVERY, not a standalone method.** The discovery is the **origin of lattice instabilities (Kohn anomalies / CDW soft modes) across the 2D-TMD family**, read from **how each anomaly's kink evolves with electronic (T_el) and lattice (T_lat) temperature**. `kink(T_el, T_lat)` — and its sharp-transition window — is the **discriminating observable of origin**: Fermi-surface / EPC-driven instabilities melt with T_el; lattice-anharmonic-driven ones are governed by T_lat. The ML (ML-EPW, conditioned / long-range MLIP, few-shot) is the **instrument** that makes this origin-mapping tractable at family scale — reported with scope, **subordinate to the science**; the headline is *what the kink(T) surface reveals*, not the model. (Even Fig 9's K-failure is not a "method bug" but the physical finding that K-A₁′ is a **long-range electronic** anomaly — itself an origin statement.) Implements roadmap `docs/NCS_ROADMAP.md` **§9** as the **(E)-channel instrument of the flagship's Part-II discovery**.

---

## 1. Problem & physics

The Kohn-anomaly **kink** |dv| — the two-sided slope discontinuity of the top optical branch at 2k_F (Γ-E₂g, K-A₁′ in graphene; the CDW soft mode in NbSe₂) — measures the cusp sharpness, which is set by the **Fermi-surface sharpness**. Temperature acts through **two physically distinct channels**:

- **(E) electronic — smearing T_el.** Fermi–Dirac occupation broadens the Fermi surface → the 2k_F log-singularity that produces the cusp is cut off → the kink **washes out**. This acts at *fixed geometry* (frozen phonon) and is **DFT/DFPT/EPW-only — a standard MLIP is structurally blind** (no Fermi surface). Measured: graphene kink_Γ 8.2→5.8 over T_el 789→6315 K (weekly-report Fig 12); NbSe₂ CDW soft mode melts through zero near electronic T_CDW ≈ 500–570 K (Fig 8a).
- **(L) lattice — physical temperature T_lat.** Thermal displacements + phonon–phonon anharmonicity + thermal expansion renormalize the branch → the kink shifts. This is **MLIP + TDEP/SSCHA** (Fig 10). Measured: graphene (L)-kink ≈ T-flat (K-A₁′ ~95); NbSe₂ soft mode anharmonically heals ~150 K ≈ exp T_CDW 145 K.

**Total physical kink(T) = kink(T_el = T, T_lat = T)** — in a real material at temperature T both the electrons *and* the lattice are at T. We **decompose** by varying T_el and T_lat independently (the project's (E)/(L) decomposition), then **recombine** into the physical curve.

**Sharp-transition hypothesis (the scientific target).** There is a temperature window where **d(kink)/dT is largest**. It is strongest for CDW / nesting systems — NbSe₂: the (E) soft-mode healing near electronic T_CDW ≈ 500–570 K *and* the (L) anharmonic healing near ~145 K. For graphene the sharpest (E) change is the low-T_el **anomaly turn-on** (below ~789 K / degauss < 0.005 Ry, which Fig 12 did not sample).

**Hard constraint (from Fig 5/9 — never violate).** The kink is a *non-analytic cusp*; smooth ML regressors smooth it (exactly how foundation MLIPs fail). Therefore **no route learns the cusp directly**: (E) via analytic ML-EPW or a *smooth* kink(T_el) emulator; (L) via a *curvature-supervised* distilled MLIP.

---

## 2. Goal (discovery-first)

**Map the origin of lattice instabilities across the 2D-TMD family** by resolving each Kohn-anomaly / soft-mode kink in **(T_el, T_lat)** — adjudicating **electronic (Fermi-surface / EPC)** vs **lattice-anharmonic** drivers, locating the **sharp-transition windows**, and delivering **≥1 non-trivial origin finding** (a re-classified material, a predicted instability, or a clean (E)/(L) → exp-T_CDW trend). The ML engines (§3) are the **instrument** that makes this *family-scale* and *few-shot* (predict a new material's kink(T) origin from a handful of DFT/EPW + TDEP points) — they are means, not the message. Every method claim is subordinate to, and validated by, the science.

---

## 3. Two engines + a safe baseline

### Engine 1 — ML-accelerated EPW · the (E) channel, physics-faithful  [roadmap §9 · E12]
ML electronic Hamiltonian H(R) (DeepH / HamGNN) → Wannier-interpolated EPC g_qν → phonon self-energy / γ_qν, with the occupation **f(ε; T_el) applied analytically** → `kink_E(T_el)` at family scale. **The cusp comes from the Fermi-surface formula, not the net** — faithful iff bands/EPC are right.

### Engine 2 — T-conditioned self-learned MLIP · "the MLIP simulates it"  [roadmap §9 · Route ③]
A FC-distilled MLIP with **T_el as a global input feature**, trained on DFT forces computed at several electronic temperatures. Then:
- frozen-phonon @ (T_el, 0 K lattice) → `kink_E(T_el)` (the electronic effect);
- TDEP / SSCHA @ (T_el, T_lat) → the **combined** `kink(T_el, T_lat)` (electronic ⊕ lattice).

Few-shot: fine-tune from the material-specific distilled model with a handful of T_el-labelled configs.

> **Open risk, stated up front.** The T_el effect is *nonlocal* (Fermi surface); injecting it as a global scalar into a short-range MLIP is a strong approximation — it may capture the *smooth kink(T_el) trend* but not the true nonlocal physics. **Engine 1 is the faithful cross-check**; if Engine 2 tracks Engine 1 on graphene/NbSe₂ it is validated, otherwise it is reported as an effective (trend-level) surrogate.

#### 3.1 Engine-2 architecture — the long-range problem (why cutoff alone fails)

The Kohn cusp lives in **long-range, oscillating force constants** ~cos(2k_F·R)/R^d (Friedel / RKKY type); enlarging the MLIP cutoff to brute-force it must span many oscillation periods → cell blow-up, never converges. **Empirical proof (this work):** a distilled MLIP's K frequency is *grid-independent* — 5×5 / 6×6 / 3×3 all give ≈ 1106 cm⁻¹ — because a short-range MLIP's FC are fully contained in any supercell ≥ cutoff. So **"distill 5×5, infer 6×6" changes nothing**; the K value is baked in by *training + architecture*, not by the evaluation grid (contrast DFT: 1362 → 1292 across the same grids). The lever is the model, not the inference cell.

**Crucial distinction — standard long-range terms ≠ Kohn's.** MLIP long-range machinery targets *electrostatic / dispersion* physics (insulators / polar), **not** the metallic Fermi-surface singularity:

| mechanism | targets | captures Kohn? |
|---|---|:--:|
| Ewald / charge-equilibration (QEq, 4G-HDNNP) | long-range electrostatics, LO-TO / NAC | ✗ |
| LODE (long-distance equivariant) | electrostatics / polarization field | ✗ |
| D3 / MBD | dispersion / vdW | ✗ |
| Ewald-based message passing | reciprocal-space periodic coupling | partial — still not Fermi-surface |

The metallic **2k_F** singularity is the near-empty frontier (long-range MLIP work is overwhelmingly molecules / insulators). A Kohn-relevant long-range term must carry **electronic / Fermi-surface information**, three routes (speculative → faithful):

1. **Explicit Friedel / RKKY kernel** — an energy term ∝ cos(2k_F·R)/R^d with **k_F (or Fermi-surface descriptors) as a learned / input parameter** that moves with filling and T_el. Principled (it *is* the kernel that produces the anomaly); highest novelty, highest risk.
2. **Electronic-state conditioning** — feed **T_el / Fermi-surface descriptors** as model input (= Engine 2). Learns the T_el trend, but encoding a nonlocal 2k_F response via local features + a global scalar is a strong approximation (the §3 open risk).
3. **ML electronic structure + analytic response** (DeepH / HamGNN → ML-EPW = Engine 1) — the long-range Fermi-surface physics is done by the formula; ML only learns the smooth electronic structure. **Most faithful.**

**Static / non-adiabatic ceiling (a harder limit).** The *static* Kohn anomaly (frozen-phonon at fixed smearing) **is in the Born-Oppenheimer PES** → a Kohn-aware long-range MLIP can in principle target it. But the **T_el-dependence** and the **non-adiabatic** part (Lazzeri–Mauri: the Γ-E₂g phonon outruns electronic relaxation → BO breaks) are **not in any single static PES** → no PES-based MLIP, long-range or not, can represent them; they require **T_el conditioning (Engine 2) or a dynamic EPW treatment (Engine 1)**. This caps how far Engine 2 can go alone, and is why Engine 1 stays the arbiter.

### Baseline — kink(T_el, T_lat) emulator (GP / Δ-ML) · ships first  [roadmap §9 · E11]
A Gaussian-process over (T_el, T_lat, descriptors) → kink. Ideal for **few-shot** smooth regression; it learns only the smooth surface (never the cusp). It (a) validates the "few-shot predicts the curve" claim on day one, (b) is the **oracle / active-learning selector** telling Engines 1–2 which (T_el, T_lat) points to spend DFT on. Implemented and runnable now (`scripts/smearing_kink/kink_emulator.py`).

---

## 4. Milestones (S-series) — discovery-first

**The target is the origin finding (S6–S7, ★).** The engines (S1–S5) are the instrument that gets there; each row leads with the *science it produces*, then the engine that enables it.

| # | Science deliverable (headline) | Engine / how | Compute |
|---|---|---|---|
| **S0** *(now)* | the origin observable exists — seed the (E) & (L) kink(T) axes from banked data (Fig 12 / Fig 10) | branch + plan + GP emulator PoC | local |
| **S1** | graphene: the **real** kink_{Γ,K}(T_el) turn-on curve (σ→0 anomaly resolved, K-commensurate) | 6×6 DFT degauss sweep 0.002–0.20 | V100 |
| **S2** | graphene: **few-shot** reconstruction of kink(T_el) from ≤4 anchors → σ→0 truth (1292 / kink 14.4) | GP / Δ-ML emulator + learning curve | local |
| **S3** | can an MLIP *carry* the electronic origin? kink_E(T_el) predicted, not just fit | T_el-conditioned / long-range MLIP (Engine 2) | 2060/H20 + V100 labels |
| **S4** | graphene **(E)/(L) kink(T_el, T_lat) surface** + its sharp-transition window = the origin map for one system | (L) via TDEP/SSCHA on the conditioned model | 2060/H20 |
| **S5** | faithful (E) origin from first principles: γ_qν(T_el) reproduced (arbiter for S2–S4) | ML-EPW (Engine 1) | rental/V100 |
| **S6 ★** | **NbSe₂ origin adjudicated in (T_el, T_lat)** — soft mode melts with T_el (~500–570 K, EPC) vs lattice healing (~145 K); the dramatic kink(T) transition mapped | Engines 1–2 on the CDW flagship | V100 |
| **S7 ★★ = the discovery** | **family origin-map**: few-shot predict a *new* TMD's kink(T) origin from a handful of points → **≥1 non-trivial finding** (re-classification / predicted instability / clean (E)/(L) → exp-T_CDW trend) | few-shot transfer of the validated engine | — |

---

## 5. Compute & data requirements

**Small by design** — a *targeted deep-dive*, not a campaign. Measured anchors (report §5.2): graphene 6×6 frozen-phonon DFT ≈ 1–3 box-h/point; NbSe₂-class DFPT ≈ 6 h + EPW ≈ 2.5 h; SSCHA ≈ 15–30 min/(mat·T); MLIP fine-tune ≈ 30–60 min; MLIP dispersion ≈ ~1 min.

### 5.1 Compute by task
| Task (milestone) | Workload | Estimate | Hardware |
|---|---|---|---|
| Fig-9 fix: 6×6 DFT fc₂ + re-distill (prereq) | 1× 6×6 frozen-phonon DFT + distill + recompute | ~3–6 box-h + ~1 h | V100 + 2060 |
| (E) graphene kink(T_el) sweep (S1) | ~8–10 × 6×6 frozen-phonon DFT (degauss 0.002–0.20) | ~15–40 box-h | V100 |
| (E) NbSe₂ kink/γ(T_el) (S6) | ~6–8 × frozen-phonon + EPW γ_qν(T_el) | ~40–90 box-h | V100 (CPU-lane) |
| (L) kink(T_lat) TDEP/SSCHA (S4) | ~5 T × 2 materials, MLIP forces | ~5–15 GPU-h | 2060 / H20 |
| Engine-2 conditioning labels (E×L) | DFT forces, thermal × 3–4 T_el × 2 mat | ~10–30 box-h | V100 |
| Baseline emulator + few-shot (S0/S2) | GP / Δ-ML | minutes | local |
| Engine-2 dev (conditioned + long-range MLIP) (S3/S4) | architecture + training + few-shot sweeps | ~50–150 GPU-h | 2060 / H20 (+rental for cutoff-scaling) |
| Engine-1 ML-EPW (DeepH / HamGNN) (S5) | H(R) training data + model + Wannier / EPC | ~500–1500 GPU-h | rental A100 (the stretch) |

**Read.** Everything through **S4** (graphene method + Engine-2 PoC + the few-shot claim) fits on **machines in hand** (2×V100 + 2060/H20): ~**100–200 box-h V100 + ~100–200 GPU-h**, *no rental*. Only **Engine-1 ML-EPW (S5)** and long-range-MLIP **cutoff-scaling** need rental — gated on S2–S4 showing signal.

### 5.2 Data
- **(E)** graphene frozen-phonon DFT at ~8–10 degauss ∈ [0.002, 0.20] Ry on a **6×6 K-commensurate** grid (reuse `m1_1b_graphene_dft.py --degauss --smearing`, patched to also save fc₂). NbSe₂ analogous + EPW γ_qν(T_el).
- **(L)** thermal snapshots + TDEP/SSCHA at ~5 lattice T on the material-specific distilled MLIP.
- **(E×L, Engine 2)** DFT forces on thermal configs at ~3–4 T_el (the conditioning labels).
- **Fermi-surface descriptors (for the long-range term)** — k_F / nesting / DOS(E_F) per material, extracted from the **existing** DFT band calcs (V-Q1 / vq3c), cheap.
- **Engine-1** DFT H(R) + EPC — **reuse the running EPW campaign's dvscf / Wannier** (no new dvscf for graphene/NbSe₂).
- **Consistency (critical).** ALL kink values via **one estimator** (`anomaly_locate.high_sym_kinks`, span = 8) on a **fixed path/grid**, and (E)/(L) kinks referenced to **the same** distilled model so they share a scale — the M2-vs-DFT frequency-scale mismatch (weekly-report Fig 10 note) must **not** recur when the channels are combined.

---

## 6. Scaffold (`scripts/smearing_kink/`)

- `gen_kink_dataset.py` — assemble `(T_el, T_lat, kink_Γ, kink_K, ω_Γ, ω_K)` rows from dispersion `.npz` (via `anomaly_locate.high_sym_kinks`); hook to launch the DFT degauss sweep. Produces the seed dataset from existing `results/vq2` + `results/td_phonon` now.
- `kink_emulator.py` — GP few-shot emulator over (T_el[, T_lat]) → kink; leave-k-out **learning curve**. Runnable PoC on the existing graphene data (numpy-only).
- *(planned)* `cond_mlip_train.py` — T_el-conditioned distillation (Engine 2); `ml_epw/` — Engine-1 interface (H(R) → EPC → γ_qν).
- `README.md` — how to run + extend.

---

## 7. Validation & honesty clauses

- Graphene σ→0 truth (K-A₁′ 1292 cm⁻¹ / kink 14.4, V-Q1); NbSe₂ γ_qν 40–52 meV (E6); seed error bars on every kink.
- **Few-shot learning curves** (kink-MAE vs #training points) are the headline metric for the "少量数据" claim.
- Every speedup reported with scope; the **non-analyticity** and **nonlocal-T_el** caveats stated wherever Engine 2 is used; Engine 1 is the faithful arbiter.

---

## 8. Integration with the wider program

Not standalone — it **reuses banked data**, **consumes running-campaign output**, and is the concrete execution of roadmap §9.

### 8.1 Current results it builds on (zero new cost)
| Banked result | Role here |
|---|---|
| **Fig 12** graphene kink_Γ(T_el), 4 pts (V-Q2) | the (E)-axis **seed dataset** (emulator PoC runs on it today) |
| **Fig 10** graphene kink(T_lat) M2 + NbSe₂ TDEP/SSCHA | the (L)-axis **seed dataset** |
| **V-Q1** 6×6 σ→0 truth (K-A₁′ 1292 / kink 14.4) | the **validation target** + the fc₂ to re-distill (Fig-9 fix) |
| **E6** NbSe₂ EPW γ_qν 40–52 meV; graphene/NbSe₂ DFT fc₂ | Engine-1 (ML-EPW) **training / validation** anchors |
| **Fig 9** FT K-cusp failure | the concrete **motivating defect** this sub-line diagnoses & fixes |
| **Path-P** anharmonic distillation | the (L)-channel **conditioning precedent** for Engine 2 |

### 8.2 Running experiments it couples to (coordinate, don't duplicate)
- **EPW convergence campaign** (NbSe₂ λ(T_el) dual-machine; `jlq6` NQ=6; graphene λ(nkf)) — *this **is** an (E)-vs-T_el measurement*. Its λ(T_el) / γ(T_el) points **are** the sub-line's NbSe₂ (E) data; the nkf/NQ convergence pins the grid the sub-line inherits. **Action:** tag these as sub-line (E)-data and reuse γ_qν directly — do not re-run EPW.
- **V100 family campaign** (fc₂ + EPW γ_qν, 11 mat) — supplies the **family (E) data** Engine-1 would train/validate on; the family bands give the **Fermi-surface descriptors** for the long-range term.

### 8.3 Planned experiments it extends / feeds
- **Roadmap §9 E11/E12** = this sub-line's baseline + two engines; §9 now carries the long-range-architecture pointer (→ §3.1 here). This doc is §9's execution with the T-axis + few-shot.
- **H20 MLIP campaign** ((L)-SSCHA screen, distillation infra) — shares the FC-distillation + SSCHA machinery Engine 2 needs; the conditioned MLIP is a variant of the same pipeline.
- **Part-II family origin-map** — if Engine-1 (ML-EPW) lands, it makes the **(E)-channel of the origin-map cheap** → "a few affordable EPW deep-dives" become high-throughput. **This is the sub-line's payoff for the flagship paper.**
- **Phase-3 rental scale** — Engine-1 ML-EPW + long-range-MLIP cutoff-scaling live here, gated on S2–S4 signal.

### 8.4 Sequencing (how it slots into the timeline)
1. **Now / local:** S0 emulator PoC + this plan (done); patch `m1_1b` to save fc₂.
2. **Near-term / running V100:** Fig-9 fix (6×6 re-distill) + the (E) kink(T_el) sweep — *piggyback the EPW convergence campaign's V100 time*.
3. **Weeks / 2060 + H20:** Engine-2 conditioned + long-range MLIP; few-shot learning curves (the "少量数据" headline).
4. **Months / rental:** Engine-1 ML-EPW — the faithful arbiter + the family-scale (E) accelerator.
