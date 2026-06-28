# TD-Phonon — Pre-submission experiment plan across two V100s + the 2060

Two rented V100-32GB boxes + the local 2060. This plan allocates the remaining
single-card experiments needed before submission.

## Machines
- **Box A** = `v100ts` (tailnet) / `v100` (public:24116), ubuntu22. **GPU-QE READY**
  (`/root/gpupw.sh`, NVHPC 24.5, QE 7.3.1 pw.x, ~19×). 500G `/data`. Currently running Path-P Stage A.
- **Box B** = `v100bts` (100.123.220.57) / `v100b` (public:13940), ubuntu22. Fresh; conda `phonon`
  env (ase/phonopy/qe-CPU) installed, 500G `/data`. **Needs GPU pw.x** — conda CPU is far too slow
  for the NbSe₂ metal (~6 min/SCF *iteration* → hours/SCF). Fix = **copy Box A's 14.5 GB GPU-QE
  toolchain over tailscale** (NVHPC `/opt/nvidia/hpc_sdk` 14G + `/root/q-e-qe-7.3.1` 526M; identical
  V100/driver-550/ubuntu22 → relocatable). ~15 min vs a ~3 h fresh build.
- **2060** = `howardwang@100.105.21.7`. MLIP only (SSCHA/TDEP/fine-tune/figures), no FP64 DFT.

## Status correction
Box B is **NOT** running EPW. It was set up for the NbSe₂ (E)-channel frozen-phonon scan (E1 below),
which stalled on CPU pw.x and was killed. EPW (E2) is a *separate, heavier* experiment proposed below.

## The remaining experiment menu (single-V100 feasible)

| # | experiment | why it matters for the paper | box | compute (GPU) | prio |
|---|---|---|---|---|---|
| **E1** | **NbSe₂ (E)-channel**: soft-mode ω²(degauss) vs electronic smearing (frozen-phonon) | fills the **one gap** in the (E)/(L) decomposition — we have NbSe₂ (L)=SSCHA/TDEP but no (E); ω²→0 gives electronic T_CDW; the MLIP structurally can't make this | **B** | ~12–18 SCF, **~30–60 min** | ⭐⭐⭐ |
| **E3** | **Path-P NbSe₂ → re-SSCHA** *(running on A)* | does *anharmonic* distillation capture the CDW double-well the harmonic one misses (SSCHA #1)? | **A** | Stage A ~8 h + C/D/E | ⭐⭐⭐ |
| **E2** | **EPW graphene**: α²F(ω), mode linewidths γ_qν, λ | the **canonical** Kohn-anomaly observable; upgrades (E)-channel from "frequency trend" (#3) to reviewer-proof linewidths | **B** | setup ~1–2 h + run ~1–2 h | ⭐⭐ |
| **E4** | **NbSe₂ DFPT at q_CDW** (direct soft mode, smearing-resolved) | rigorous DFPT cross-check of the frozen-phonon fc₂ (E1) + a cleaner (E)-channel | A or B | ~2–4 h | ⭐⭐ |
| **E5** | **Graphene SSCHA** (rigorous (L)-channel) | symmetry with NbSe₂ SSCHA #1; rigorous ω(q,T) for graphene | **2060** | MLIP ~1 h | ⭐⭐ |
| **E6** | **EPW NbSe₂** at q_CDW (λ_q, mode-resolved EPC) | the momentum-dependent e-ph coupling that *drives* the CDW (ties to #2 not-nesting) | A or B | heavy, ~½–1 day | ⭐ (stretch) |
| **E7** | **Graphene K-point (E)-channel** (A₁′ anomaly vs smearing) | extends #3 from Γ-E₂g to the K-A₁′ anomaly | A or B | ~1 h | ⭐ |

## Two-box allocation + timeline

**Box A (GPU-QE ready) — the Path-P centerpiece, then DFPT/EPW:**
1. *(running, ~8 h)* Path-P Stage A → C (fine-tune) → D (re-SSCHA on 2060) → E (writeup). [E3]
2. *then* E4 (NbSe₂ DFPT q_CDW cross-check) — rigorous companion to E1; optionally E6 (EPW NbSe₂).

**Box B (after the 15-min GPU-QE copy) — the (E)-channel + the EPW anchor:**
1. **Copy GPU-QE from A** (tailscale rsync, ~15 min) → Box B becomes a GPU-DFT node.
2. **E1** NbSe₂ (E)-channel soft-mode vs smearing (~30–60 min) — *the immediate high-value run.*
3. **E2** EPW graphene (α²F/γ_qν/λ) — the rigor anchor; set up Wannier90+EPW, run on graphene (cheap, 2 atoms).

**2060 — MLIP track (parallel, no GPU contention):**
- E5 graphene SSCHA; Path-P Stage D re-SSCHA (when A's data is ready); all figures.

## Bottom line for submission
The two **must-haves** are **E3** (Path-P, running) and **E1** (NbSe₂ (E)-channel, ~1 h on Box B after
the copy) — together they complete the (E)/(L) decomposition + the anharmonic-distillation test for the
centerpiece. **E2** (EPW graphene) is the high-value rigor upgrade (reviewer-proof linewidths) if the
EPW setup cooperates. E4–E7 are confirmatory/symmetry add-ons. Realistically: **E1 today (~1 h),
E3 finishing overnight, E2 next** — the core pre-submission slate lands within ~1 day.
