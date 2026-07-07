# Compute & Data Master Plan — TD-phonon / Kohn-anomaly project

**Date:** 2026-07-03 · **Scope:** all remaining experiments across **Part I (method)**, **Part II
(discovery)**, the **`smearing-kink-ml` sub-line**, and the **long-range-MLIP training** — with the
data required, the compute cost (computed from *measured* throughput), and which hardware runs each.
Supersedes the single-V100 draft. Hardware specs below are **verified 2026-07-03**.

---

## 0. Hardware inventory (verified)

| Machine | GPU / VRAM | CUDA / driver | Role | Torch |
|---|---|---|---|---|
| **Box A** `v100ts` / `v100` (root) | Tesla V100-SXM2 **32 GB** | 12.4 / 550 | graphene+NbSe₂ **DFT+EPW** | **CPU-only** (phonon env) |
| **Box B** `v100bts` / `v100b` (root) | Tesla V100 **32 GB** | 12.4 / 550 | **full TMD family** DFT+EPW | CPU-only |
| **2060** `howardwang@100.105.21.7` (`howard-pc`) | RTX **2060 SUPER 8 GB** (free) | 12.5 / 555 | **MLIP training / eval** | ✅ **GPU** — `~/miniconda3/envs/phonon`: torch 2.6.0+cu124, mace 0.3.16, e3nn 0.4.4 (verified: 200 train steps 0.26 s). 16 cores / 46 GB / 91 GB disk. Repo `~/phonon` (behind → needs `git pull`). |
| **H20** 8× (96 GB) | — | — | MLIP **mass** train/benchmark | **OFFLINE this week** (resume when fleet returns) |

**Lane model:** each V100 runs a **GPU-DFT lane** (`pw.x` via `gpupw.sh`) **∥** a **CPU lane**
(`ph.x`/`epw.x`; MACE-CPU) concurrently (verified). The 2060 is **one 8 GB GPU-training lane**.
The **TMD family is Box-B-bound** (pseudos + `tmd_*` scripts). Box A carries graphene/NbSe₂ + MLIP-CPU.

**Key consequence for training:** the V100 phonon envs are **CPU-torch** → MLIP *training* there is
impractical. Training belongs on the **2060 (GPU-torch, verified)**; the V100 32 GB GPUs *could* train
(install CUDA-torch) but are saturated by the DFT/EPW critical path. **No rental is needed for MLIP
training** — the 2060 suffices for every fine-tune below; only Engine-1 (ML-EPW, GPU-weeks, >8 GB)
needs H20/rental.

## 0.1 Measured per-unit cost (2026-07-03, V100 + 2060)

| Task | Cost | Machine | Note |
|---|---|---|---|
| fc₂ **2×2** (k12) | ~14 min | V100 GPU-DFT | TiSe₂ scan 3 pts / 42 min |
| fc₂ **4×4** (k8) | ~1–2 h | V100 GPU-DFT | 48 atoms |
| graphene **6×6** fc₂ (k6) | ~8 min/pt | V100 GPU-DFT | 12-pt kink sweep ~1.6 h |
| graphene 6×6 fc₂ (k12, low degauss) | **stuck >2 h** ⚠ | — | semimetal+low smearing — **avoid** |
| **on-instability EPW**, strong soft mode | **~6–9 h** ⚠ | V100 CPU | TiSe₂ @0.005 (22 imag) >9 h |
| EPW, mild/no soft mode | ~3–5 h | V100 CPU | NbSe₂/graphene ref |
| (L)-channel TDEP (3 T) | ~0.5–1.5 h | V100 CPU (MACE) | running now |
| **MACE fine-tune / FC-distill** | **~10–30 min** | **2060 GPU** | vs ~40 min on V100-CPU; 8 GB fits batch 1–4 |
| SSCHA per material | ~1.5 h | V100/2060 | roadmap anchor |
| χ(q) nesting (bands) | ~20–40 min | V100 GPU-DFT | per material |

*Unit = "box-h" (one lane busy 1 h). Wall-clock assumes continuous running.*

---

## 1. Part I — Method lock (V100, no rental)

Flagship rigor (NbSe₂ + graphene) to make the **method** submittable.

| # | Experiment | Materials | Lane | Cost | Data needed (have?) |
|---|---|---|---|---|---|
| P1-a | **λ(T_el)/λ_q convergence** nkf 24→48→60, unified smearing, ≥5 T_el | NbSe₂ | CPU-EPW | ~18–30 h | reuse DFPT (✅) |
| P1-b | **ASR re-pass** (q2r `zasr='crystal'`) | graphene, NbSe₂ | GPU+CPU | ~5–8 h | full-QE conda install (❌ needed) |
| P1-c | convergence/error-bar sweeps (k,q,smearing,cell) | 2 flagships | GPU-DFT | ~10–18 h | — (✅) |

**Part I:** **~33–56 box-h** = ~20–35 h CPU-EPW + ~15–25 h GPU-DFT. **Box A. Wall ≈ 1.5–2.5 d.**

## 2. Part II — Discovery / (E)–(L) origin-map (V100, no rental)

Family = 7 CDW (NbSe₂✓, TiSe₂✓, VSe₂◐, **NbS₂, 2H-TaS₂, 1T-TaS₂, 2H-TaSe₂**) + 4 controls.

| # | Experiment | Remaining | Lane | Cost | Data needed (have?) |
|---|---|---|---|---|---|
| P2-a | **fc₂ + re-commensuration** (2×2→4×4/√3) | NbS₂, TaS₂×2, TaSe₂ (+VSe₂ 4×4 running) | GPU-DFT | ~12–18 h | Ta/Ti/V/S ONCV (✅ Box B) |
| P2-b | **χ(q) nesting** | same 4 | GPU-DFT | ~2–3 h | bands (✅) |
| P2-c | control fc₂ (confirm stable) | TiS₂,VS₂,MoS₂,WSe₂ | GPU-DFT | ~1–2 h | (✅) |
| P2-d | **(E)-EPW γ_qν on-instability** ⚠ **bottleneck** | 5 CDW | **CPU-EPW** | **~30–55 h** | dvscf+Wannier per mat (gen'd) |
| P2-e | **Path-P (L) labels** + distill | 6 CDW | GPU+CPU/2060 | ~10–15 h | fc₂ soft eigvecs (from P2-a) |
| P2-f | **origin-map synthesis** + hunt re-classification | — | — | ~0 | exp T_CDW/INS (literature) |

**Part II:** **~55–90 box-h** = ~30–55 h CPU-EPW (P2-d critical) + ~15–23 h GPU-DFT + ~7 h MLIP.

> **(L)-channel rigor + (E) ceiling are rental (C-series, `RENTAL_EXPERIMENTS.md`):**
> P2-e (Path-P) on V100 gives MLIP-triage (L)-curves; the **DFT-validated rigorous (L)** for the
> VSe₂ anomaly + 1 control = **C1** (rental, ~5–7 d). The long-range-term convergence ceiling
> (graphene 10×10/12×12) = **C2** (needs an 80 GB A100/H100 — 10×10 OOMs on 32 GB V100). The dynamic
> (E) ceiling (Engine-1 ML-EPW) = **C3** (H20 train + V100 H(R) dump, ~2–4 wk). All gated on the
> no-rental screens. See `docs/LCHANNEL_FAMILY_PLAN.md`.

**All Box-B-bound. Wall ≈ 3–5 d**, set by 5 serial family EPWs.
> ⚠ **Cost driver + mitigation:** strongly-soft EPW is ~6–9 h & fragile (TiSe₂ >9 h). For each material
> find the **marginal degauss** (just-soft, few imag) via a cheap 2×2 scan first → marginal EPWs
> converge far faster. If a member still won't converge, its (E)-verdict falls back to fc₂-softness +
> χ(q)-nesting (as TiSe₂/VSe₂ already show).

## 3. Sub-line — the `kink(T_el, T_lat)` surface (V100 + 2060)

**MISSION (refined 2026-07-08):** ONE unified MLIP+long-range-fine-tuned model reproduces the
Kohn-anomaly **kink** in BOTH (E) smearing (T_el) and (L) temperature (T_lat) spectra. (E) kink(T_el)
✅ done (graphene 0.31, 5 CDW 0.011–0.16); **(L) kink(T_lat) via MLIP+LR = central remaining deliverable**
(extend Friedel LR term to T_lat-conditioning, fed by MD-TDEP fc₂(T_lat)). Strategic plan =
`docs/UNIFIED_KINK_GOAL_2026-07-08.md`; ops = `docs/THREE_MACHINES_PHASE2_2026-07-07.md`.

S1/S2 already banked for graphene (Fig 13). Remaining single-V100 + 2060 work:

| # | Experiment | Materials | Lane | Cost | Data needed (have?) |
|---|---|---|---|---|---|
| S-a | **(L)-axis kink(T_lat)** via FT+TDEP → feeds T_lat-LR term | VSe₂ 🔵(B1 running), TiSe₂, graphene, NbSe₂ | 2060 | ~11 h | Path-P FT (✅ VSe₂/TiSe₂) |
| S-b | **NbSe₂ (E) T_el finer sweep** (S6★ member) | NbSe₂ | GPU-DFT | ~2.5 h | (gen'd) |
| S-c | **`kink(T_el,T_lat)` 2D surface** via unified MLIP+LR | graphene (+ VSe₂) | 2060+local | ~5 h + code | B1 fc₂(T_lat) (⏳) |
| S-d | **few-shot emulator** per material | all done | 2060/local | ~0 | kink CSVs (✅) |

**Sub-line:** **~10–12 box-h** = ~6–7 h GPU-DFT + ~5 h MLIP. Mostly Box A + 2060. **Wall ≈ 0.5–1 d**
(hidden under Part I). **S1 ✓, S2 ✓ done.**

---

## 4. Long-range-MLIP training (the "put long-range into the MLIP") — **2060 / H20**

The physics: the Kohn cusp is a **long-range oscillating FC ~cos(2k_F·R)/R^d** (Friedel/RKKY); a
short-range MLIP structurally recovers only ~75 % of it (Fig 9). This section = the compute to
close the residual. **Two engines + the done baseline.**

| Engine | What it learns | Training data (have?) | Compute | Machine | Status |
|---|---|---|---|---|---|
| **Baseline: GP few-shot emulator** | kink(T) *scalar* | kink CSVs (✅) | ~free (CPU) | any | ✅ **DONE (S2)** |
| **E2b: explicit long-range term** (Friedel 2k_F kernel) | oscillating FC tail w/ k_F | 12 fc₂(T_el) (✅) + k_F from bands | ~1 day (mostly impl.) | 2060/local | ✅ **DONE 2026-07-03** — see §4.2 |
| **E2c: cutoff-scaling ceiling** (r_max 3→6, eff 6→12 Å) | proves short-range ceiling | reference fc₂ (✅) | done | 2060 | ✅ **DONE** (r_max sweep + fc₂-truncation) — see §4.2 |
| **E2a: T_el-conditioned MLIP** | smearing-specific backbone reproduces its own kink | 12 fc₂(T_el) (✅) | ~1 GPU-h (3 smearings) | 2060 | ✅ **DONE 2026-07-03** — see §4.2 |
| **E1: ML-EPW** (DeepH/HamGNN → EPC → γ_qν) | DFT Hamiltonian → EPC; smearing analytic | family DFT **H(R)** dump (❌ — extra ~10–30 box-h V100 DFT) | **~1–4 GPU-weeks** + new pipeline; needs **>8 GB** | **H20 / rental** | gated |

**Engine-2 total (the long-range term): ~3–8 GPU-days, all on the 2060** (8 GB is ample — the
fine-tunes use <2 GB; the cost is research iteration, not VRAM). **No rental.** Data for the **PoC is
already in hand** (12 fc₂). **Engine-1 stays H20/rental** (GPU-weeks, needs H(R) dump first).

### 4.1 Architecture reference for E2b — BAMBOO

**BAMBOO** (ByteDance; *Nat. Mach. Intell.* 2025 / arXiv:2404.07181; open-source `bytedance/bamboo`)
is the template for the long-range term. It bolts an **explicit, analytic long-range module**
(predicted atomic charges → Ewald/Coulomb `1/r`) onto a **local graph-equivariant-transformer (GET)**
backbone, and stabilises training with **density-based ensemble knowledge distillation**. What we take
vs. adapt:

- **Adopt (the scaffold):** local equivariant GNN **+ a separate analytic long-range module + ensemble
  distillation** — a proven, forkable pattern for reaching *beyond the local cutoff analytically*
  (exactly the mechanism our fc₂-truncation result demands: the K-A₁′ cusp needs ~8–12 Å, §sub-line).
- **Swap (the physics):** BAMBOO's long-range is **electrostatic** (net charges, Coulomb — for
  *insulating* electrolytes). Graphene/TMD metals are ~charge-neutral; the Kohn cusp is instead a
  **metallic Friedel/RKKY 2k_F oscillation** `~cos(2k_F·R)/R^d`. So **E2b = BAMBOO's scaffold with a
  Fermi-surface 2k_F kernel** (k_F from the bands) in place of the Coulomb module — a long-range term
  no existing MLIP has, and the genuine methodological novelty of the sub-line.
- **Reuse (the training trick):** BAMBOO's ensemble distillation maps onto our FC-distillation of the
  12 fc₂(T_el) → stability without a huge dataset.
- **Compute note:** GET is heavier than MACE-small; the graphene PoC still fits the 2060's 8 GB, but a
  full multi-material BAMBOO-style train may want H20 on return.

**S3 test (the near-term deliverable):** FC-distill the 12 graphene fc₂(T_el) into a
T_el-conditioned / long-range architecture and measure whether the K-A₁′ kink residual (~25 %, i.e.
DFT 14.4 vs FT 10.8) closes. **Runnable on the 2060 today** once the 12 fc₂ are copied over
(V100 → 2060, ~small).

### 4.2 Status & results (2026-07-03)

**E2b — DONE (full writeup `docs/LONGRANGE_TERM_FINDINGS.md`).** The (E)-channel long-range term is a
**2-parameter thermally-damped Friedel oscillation** `fc₂(R;T_el)=backbone+B(T_el)·e^{−κ(T_el)R}·D0(R)`,
D0 = full-tensor Fermi-surface waveform. Results on graphene 6×6, 12 smearings (kink_K, DFT 22.63→0.22):
- **Fit** reproduces the whole smearing collapse, MAE ≈1.7 (backbone flat = 0.22); B≈1.1 (≈T_el-indep), κ↑.
- **Few-shot:** smooth B, κ(T)=a·T^0.61 from **3 smearings** → 8 held-out kinks **MAE 0.87** ⇒ ξ∝T_el^(−0.6..−0.7).
- **Transfer:** a=2.46 law → held-out kinks at **a=2.44 (MAE 0.27), a=2.48 (MAE 0.52)** from 2 anchors.
- **Deployment:** `FriedelMACECalculator` (`scripts/smearing_kink/friedel_calc.py`) wraps any MLIP + adds the
  Friedel harmonic term; validated round-trip (calc==direct add_template); composes with `mace_mp`.
- **Analytic-template (drop the measured D0):** attempted, **negative** — graphene RKKY is sublattice-
  dependent with fc-period ~2.3 Å (< 2π/|K|=3.69 Å); simple `cos(q*·R)` captures only ~19%. Keep the
  measured 1-smearing template (cheap).
- Code: `friedel_module/fit_friedel/transfer_friedel/friedel_calc/plot_*/analytic_template/_diag_kink.py`.
  Figs: `results/smearing_kink/{friedel_module,friedel_transfer}.png`.

**E2c — DONE.** From-scratch MACE r_max 3→6 Å (eff range 6→12) all reproduce kink_K ≈15 **in-distribution**
(`graphene_rmax_sweep_kink.csv`) → no cutoff ceiling for *fitting*; but the two models give **opposite
kink-vs-a transfer trends** (`graphene_rmax_transfer_a.csv`) → short-range MLIPs do **not** transfer, and
fc₂-truncation kills the cusp below ~7–8 Å (`graphene_fc2_truncation.csv`). This is exactly the evidence
that the explicit long-range term (E2b) is needed — which then transfers (MAE 0.27).

**E2a — DONE on the 2060** (`gr_backbone_distill.sh` + `eval_backbone_deploy.py`). FC-distilled a real
graphene MACE at dg{0.002,0.010,0.080} (~1 GPU-h). Results:
- **Conditioning baseline:** each smearing-specific MACE reproduces its own kink — dg0.002 → **23.74** (DFT
  22.63), dg0.010 → **13.66** (13.33), dg0.080 → **0.22** (0.22, exact). ⇒ the backbone is expressive
  enough for any single smearing; a single *fixed* model gives one kink, so T_el needs conditioning + the
  long-range term.
- **Quantitative deployment (real MLIP):** base = the dg0.080 MACE backbone + Friedel module → kink(T_el) =
  20.0/16.9/14.0/10.7/7.4 vs DFT 22.6/15.7/13.3/9.8/3.9 (MAE ~1.5) — same quality as the harmonic-base
  validation, now with a genuine trained MACE (vs MACE-MP-0's broken bare kink=101). Models in
  `results/gr_backbone/dg{0.002,0.010,0.080}/` on the 2060.

### 4.3 2060 queue — status

1. **E2a backbone distillation** — ✅ **DONE** (see §4.2).
2. **Data-efficiency of the few-shot law** — ✅ **DONE** (`few_shot_dataeff.py`): the κ(T)=a·T^b law fit from
   **3 anchor smearings** predicts held-out kinks to MAE ~1.6 → the few-shot law is data-cheap.
3. **Cross-model backbone check** — ✅ **DONE**. Built a dedicated `phonon-sevenn` env (torch 2.6+cu124 GPU +
   sevenn 0.13), fine-tuned **SevenNet** (7net-0 → graphene dg0.080, 50 ep) → backbone kink 0.23 (≈DFT 0.22).
   **SevenNet + Friedel = 20.1/14.0/7.3** at T_el 316/1579/6315 — *identical* to MACE + Friedel (20.0/14.0/7.4).
   ⇒ the Friedel module is **backbone-agnostic** across two MLIP frameworks (given a reference-matched
   backbone; an off-the-shelf foundation model gives artifacts — the reference-mismatch, not the module).
   `cross_model_check.py <checkpoint>`; env recipe in `setup_sevenn.sh` + `fix_sevenn_torch.sh` (2060).
4. **(L)-axis kink(T_lat)** family (S-a, = the refined goal's central gap): MD-TDEP fc₂(T_lat) →
   feeds the T_lat-conditioned Friedel LR term (B2). **B1 VSe₂ MD-TDEP 🔵 running** (2060,
   `run_B1_2060_vse2_tdep.sh`, across T_CDW=110K); TiSe₂ + graphene next. NbSe₂ NQ=4 clean-λ EPW
   queued on Box A (`run_EPW_boxA_nbse2_nq4.sh`); NbSe₂ NQ=3 λ=17 vetted as soft-mode divergence.
5. **Family Friedel generalization** — ✅ **DONE** (`friedel_family.py`, on 2060 + Box B CPU lane). The
   (E)-channel damped-Friedel law **generalizes from graphene to real CDW materials**: the 2-parameter
   envelope reproduces the soft-mode **min-freq(T_el) melting curve** to **MAE 0.018 THz for 1T-VSe₂**
   (−2.18→−1.20→0→0 THz) and **0.010 THz for 2H-TaS₂** (−3.60→−2.88→−1.94→−0.46) — across polytypes.
   (For these CDW soft modes the amplitude B(T_el) alone carries the melting, κ≈0 — simpler than graphene's
   kink which needed both.) Adds NbS₂/2H-TaSe₂ once their scans finish.
6. **E1 ML-EPW** — 🔒 gated → H20/rental (needs H(R) dump).

---

## 5. Data requirements & gaps (per line)

| Line | Data it needs | Have | Missing → cost to fill |
|---|---|---|---|
| **Part I** | NbSe₂ DFPT (reuse), full-QE (q2r) | DFPT ✅ | conda-install full QE (~1 h) |
| **Part II** | family fc₂ + dvscf + Wannier + χ(q) | NbSe₂/TiSe₂ ✅, VSe₂◐ | 4–5 CDW fc₂/EPW → in §2 (~50–75 box-h) |
| **Sub-line** | kink(T_el) + kink(T_lat) per material | graphene both ✅ | family (E)/(L) sweeps → in §3 |
| **Long-range E2 (PoC)** | fc₂(T_el) spanning T_el, ≥1 material | **12 graphene fc₂ ✅** | none — **PoC-ready** |
| **Long-range E2 (robust)** | diverse configs × degauss forces | ~1 config/degauss (sparse) ❌ | **~50–90 GPU-h V100 DFT** — reuse (L)-TDEP thermal snapshots (~360 configs) × 4–6 degauss |
| **Long-range E1 (ML-EPW)** | family DFT H(R) + EPC | ❌ | H(R) dump ~10–30 box-h V100 + H20 training |

**The one data gap that matters:** the robust long-range MLIP needs a proper **(E)-channel force
dataset** (diverse configs × several degauss). **Smart fill:** the (L)-channel TDEP already emits
~120 thermal snapshots/T × 3 T ≈ **360 physically-relevant configs**; re-run SCF on them at 4–6
degauss → a combined **(E)×(L) training set for ~50–90 GPU-h** on a V100 GPU-DFT lane. This is the
one item to **add to the V100 queue** if we go past the S3 PoC.

---

## 6. Totals, wall-clock, critical path

| Line | GPU-DFT (V100) | CPU-EPW (V100) | MLIP (2060/CPU) | Total |
|---|---|---|---|---|
| Part I | 15–25 | 20–35 | — | 33–56 box-h |
| Part II | 15–23 | 30–55 | 7 | 55–90 box-h |
| Sub-line | 6–7 | — | 5 | 10–12 box-h |
| **V100 subtotal** | **36–55** | **50–90** | **12** | **~100–160 box-h** |
| Long-range E2 (2060) | — | — | ~3–8 GPU-**days** | (2060, parallel) |
| Long-range robust-data (V100) | +50–90 | — | — | optional add-on |
| Long-range E1 (H20/rental) | H(R) +10–30 | — | ~1–4 GPU-**weeks** | gated |

**Critical path = V100 CPU-EPW (~50–90 box-h), Box-B-serial → ~3–5 days.** Everything else hides under it:
- Box A: Part I + sub-line → ~2 d.
- **2060: Engine-2 long-range training runs fully in parallel** (independent machine) → ~3–8 d of iteration, gated only on the 12 fc₂ (already have) for the PoC.

**Two nearly-free accelerators:**
1. **Copy TMD pseudos+scripts → Box A** (~15 min) = 2nd parallel CPU-EPW lane → family EPW wall **~halved** → whole V100 plan **~2.5–3 d**.
2. **2060 starts the long-range PoC now** — it's idle and PoC-data-ready, so it costs nothing against the V100 critical path.

---

## 7. Suggested ordering (saturate all three machines)

1. **Box B (now→t+2 d):** GPU-lane runs family fc₂ + re-commensuration (P2-a/b/c) — each finished fc₂
   unblocks an EPW; CPU-lane drains the family EPW queue (P2-d) at the **marginal degauss**.
2. **Box A (parallel):** Part I rigor (P1-a/c) + sub-line (S-a running, S-b, S-c); add ASR (P1-b) after
   full-QE install. Optionally become 2nd EPW lane (accelerator #1) once Part I frees its CPU.
3. **2060 (parallel, independent):** `git pull`; copy the **12 graphene fc₂** from Box A → run the
   **S3 long-range/conditioned FC-distill PoC** (E2a→E2c). If it closes the residual, escalate to the
   robust dataset (fill via TDEP-snapshots×degauss on a V100 lane, §5).
4. **t+3–5 d:** all fc₂/EPW/Path-P in → build the **(E)/(L) origin-map (P2-f)** + graphene 2D surface →
   Part I submittable, Part II discovery in hand, long-range verdict (does the term recover the cusp?).
5. **Gated on S2–S4 signal:** Engine-1 ML-EPW + family-scale dense EPW → **H20 (on return) / rental**.

**Out of single-machine scope (H20 mass / rental):** E1 ML-EPW GPU-weeks, family-wide dense-grid EPW
scale-up, the E1/E3/E4 MLIP benchmark-atlas + ablations + (L)-SSCHA family screen + E9 κ (~900–1,800 GPU-h).
