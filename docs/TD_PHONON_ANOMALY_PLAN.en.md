# Temperature-dependent phonon dispersions and dispersion anomalies in 2D materials via MLIP — plan

**Goal.** Compute *temperature-dependent* phonon dispersions ω(**q**, T) for 2D materials with a
foundation/FC-distilled MLIP (the "Level 3" anharmonic regime), then **locate the slope-discontinuities /
kinks ("jumps") in the dispersion and track how they move with temperature**, and connect those anomalies
to the electron–phonon physics this is really about.

---

## 0. The physics, named precisely

What the brief describes — *flat 2D lattice in real space, but the vibrational modes show abrupt
slope-changes and jumps in the phonon band structure, driven by the electron-gas / ion-core interaction,
plausibly tied to a transport window* — is, in the standard language:

- **Kohn anomalies.** The conduction electrons screen the ionic motion. That screening (the phonon
  self-energy from electron–phonon coupling) becomes **singular when the phonon wavevector q connects two
  points on the Fermi surface**, i.e. at **q\* ≈ 2k_F** (Fermi-surface *nesting*). The result is a
  **cusp / kink / local softening** of ω(**q**) at q\* — a non-analytic feature in the dispersion.
- **Dimensionality.** The singularity sharpens as dimension drops: in 1D it diverges and drives a
  Peierls / charge-density-wave (CDW) distortion; in 2D it survives as **sharp kinks** (graphene's E₂g at
  Γ and A₁′ at K; the soft-mode anomalies of monolayer NbSe₂ / TaS₂ that condense into a CDW).
- **Transport link (the hypothesis to test, not assume).** q\* marks a Fermi-surface feature, so the
  anomaly is a *fingerprint* of where the lattice is strongly coupled to the electrons — which is the
  intuition behind a "ballistic-transport conductivity window." **The phonon anomaly is a locator/signature
  of that Fermi-surface geometry; it is not itself the electronic conductivity.** Establishing the link
  requires the electronic structure (below), and is a result to be *demonstrated*, not presumed.

So the concrete program is: **ω(q, T) → detect kinks q\*(T) → test q\* ≈ 2k_F → relate to transport.**

---

## 1. Honest scope — what an MLIP can and cannot capture (read first; it shapes everything)

An MLIP (MACE-MP, MatterSim, …) learns the **Born–Oppenheimer potential-energy surface** E({R}) — energies
and forces with the electrons integrated out *at their ground state*. Its force constants (fc₂, fc₃, …)
*implicitly* contain electron–phonon effects **only to the extent the training DFT captured them and the
model is expressive enough to reproduce the resulting non-analyticity**. Two hard limits follow, and they
are exactly the failure modes this repository's main paper documents:

1. **A Kohn anomaly is a sharp, near-non-analytic kink in fc₂(q). MLIPs systematically *smooth/soften*
   curvature** — the central finding of §2.1 (foundation MLIPs under-predict curvature, soften ω_max by
   ~10–28%). A localized 2k_F cusp is precisely the feature most at risk of being **washed out**. ⇒ A
   bare foundation MLIP may *miss or blur* the anomaly; the **FC-distillation cure (fc₂ from public DFPT,
   §2.2) is likely *necessary*** to recover it — and even then a perfectly sharp cusp is hard for any
   smooth interpolator. *(If the MLIP cannot resolve it even after distillation, that is itself a clean,
   publishable negative — "MLIPs miss Kohn anomalies" — fully consistent with §2.1.)*

2. **The *electronic* temperature dependence of the anomaly is not in the MLIP.** Raising T smears the
   Fermi–Dirac occupation → the Fermi surface blurs → the 2k_F singularity broadens and the kink washes
   out. This is an **electronic** effect. The BO-PES (hence the MLIP) is evaluated at the electronic
   *ground state*; it carries **no electronic-temperature smearing**.

### The decomposition this forces (the intellectual core of the plan)
Split the temperature dependence into two channels and never conflate them:

| channel | mechanism | does the MLIP see it? | how we get it |
|---|---|---|---|
| **(L) lattice / anharmonic** | phonon population, thermal expansion, phonon–phonon renormalization | **yes** (it's all in the BO-PES) | MLIP + SSCHA/TDEP (Level 3) — cheap |
| **(E) electronic** | Fermi-surface smearing of the 2k_F e-ph anomaly | **no** (ground-state PES) | DFPT with finite electronic smearing T_el (or EPW) — DFT, expensive |

**Operating principle:** use the MLIP for channel (L) and to cheaply scan many T and large supercells; use
**targeted DFPT with electronic smearing** as the reference for channel (E) and to confirm q\* ≈ 2k_F. Every
reported T-trend must be labelled by which channel it comes from. *(For a CDW/soft-mode material the two
channels couple — the anharmonic stabilization (L) and the electronic anomaly (E) jointly set the
transition — so there both must be modelled and the honesty about provenance matters most.)*

---

## 2. Method for Level-3 T-dependent phonons (the MLIP / channel-L workhorse)

| option | what it yields | when to use | tooling |
|---|---|---|---|
| **QHA** (quasi-harmonic) | frequency shift from thermal expansion only | cheap baseline / sanity | phonopy (already in repo) |
| **TDEP** (temp-dependent effective potential) | effective fc₂(T) → ω(q,T); fc₃(T) → linewidths | default workhorse, robust | **hiPhive** (ASE+phonopy native) or TDEP |
| **SSCHA** (stochastic self-consistent harmonic) | free-energy-Hessian frequencies; handles instabilities | strong anharmonicity / near CDW | python-sscha |

**Recommendation:** **hiPhive-TDEP with the FC-distilled MLIP** as the workhorse (clean ASE/phonopy
integration, gives fc₂(T) on a fine q-mesh plus fc₃ for linewidths); **SSCHA** for the CDW/soft-mode case
where TDEP's perturbative footing weakens. Both need only forces on thermally-sampled supercells — thousands
of evaluations, infeasible in DFT but trivial with the MLIP. *This cheapness is the entire reason to use an
MLIP here.*

---

## 3. Materials (tiered, validation-first)

1. **Graphene** — Kohn anomalies at Γ (E₂g) and K (A₁′) are textbook-sharp and well-measured; semimetal →
   a clean electron–phonon test and a known target to validate the anomaly locator and the MLIP-smoothing
   question.
2. **Monolayer MoS₂** — gapped, weak/absent anomaly → negative control (the locator should find *nothing*).
3. **Monolayer NbSe₂ (or TaS₂)** — strong Kohn anomaly / soft mode that condenses into a CDW → the dramatic
   case and the clearest "transport-window" candidate; SSCHA territory, and where channel (E) is essential.

---

## 4. Pipeline (concrete, on the repo's infrastructure)

```
for each material:
  1. relax (ASE + FC-distilled MLIP)                          # structures.relax
  2. for T in T_grid:                                         # e.g. 10–600 K
       a. canonical sampling: NVT/Langevin MD or SSCHA ensemble in a converged supercell  (MLIP forces)
       b. fit effective fc2(T) [+ fc3(T)]                     # hiPhive TDEP  (or SSCHA Hessian)
  3. dispersion ω(q,T) on a dense q-path                      # phonopy from fc2(T)  (reuse phonons.py)
  4. ANOMALY DETECTION  (new):
       v(q)   = dω/dq            per branch (finite diff on dense path)   # group velocity
       kinks  = peaks in |d²ω/dq²| and discontinuities in v(q)           # robust: smoothed-deriv + threshold
       output q*(T), kink strength Δ(slope), branch index
  5. TEMPERATURE TRACKING: q*(T) and kink-strength vs T  →  the dispersion-anomaly map
  6. ELECTRONIC CROSS-LINK:
       - DFT Fermi surface + 2k_F nesting vectors (one DFT calc); test q* ≈ 2k_F
       - (heavier) DFPT at finite electronic smearing T_el → the channel-(E) anomaly the MLIP can't make
  7. BALLISTIC-TRANSPORT READING:
       - phonon side: group velocities + Landauer ballistic phonon conductance from ω(q,T)
       - electronic side (honest): q*(T) locates the Fermi-surface feature; the *electronic* ballistic
         window needs e-ph (EPW) — phonon anomaly is the signature, not the conductivity
```

Planned code:
- `scripts/td_phonon.py` — sampling (MD/SSCHA) + effective-fc₂(T) fit + dispersion per T.
- `scripts/anomaly_locate.py` — group-velocity / curvature kink detector → q*(T), Δslope.
- `scripts/plot_td_dispersion.py` — ω(q,T) overlay with anomalies marked; q*(T) tracking panel.

---

## 5. Validation & honesty checks
- Reproduce graphene Γ/K anomalies at low T with the MLIP; compare to DFPT → **quantify how much the
  foundation MLIP softens the cusp and whether FC-distillation recovers it** (directly extends §2.1/§2.2).
- Confirm q\* ≈ 2k_F from the DFT Fermi surface (the anomaly hypothesis).
- Keep channels (L) and (E) separate in every figure/claim; never attribute electronic-smearing physics to
  an MLIP-only result.
- Convergence: supercell size, q-path density, MD length / ensemble size, electronic smearing (DFT side).

## 6. Milestones
- **M1** — graphene: hiPhive-TDEP fc₂(T) with the FC-distilled MLIP; ω(q,T); anomaly locator working;
  validated vs DFPT at one T (and vs the known graphene Kohn anomaly).
- **M2** — q\*(T) tracking + kink-strength map; 2k_F cross-check from the DFT Fermi surface.
- **M3** — monolayer NbSe₂ (SSCHA): strong-anomaly / soft-mode T-evolution.
- **M4** — ballistic-transport reading + the channel-(L)/(E) provenance write-up.

## 7. Risks / open questions
- **Primary risk:** the MLIP smooths the very cusp we hunt. Mitigate with FC-distillation + DFPT
  validation; if it still fails, report it as a negative result (it strengthens, not weakens, the paper).
- TDEP's perturbative footing breaks near a CDW instability → use SSCHA there.
- The "ballistic-transport window" link (q\* ↔ an electronic transport feature) is a **hypothesis to test**,
  not a premise — it needs the electronic structure / e-ph, which the MLIP does not provide.
- **Compute:** serious supercell MD / SSCHA wants the GPU boxes (currently off the tailnet); small graphene
  cells are feasible locally. DFPT + Fermi-surface validation needs the strong-FP64 (rental) hardware.

## 8. Dependencies
- Python: `hiphive`, `phonopy`, `phono3py`, `ase`, (`python-sscha` for the CDW case), the repo's MLIP
  calculators (`phonon-mace` env / FC-distilled model).
- DFT (validation only): Quantum ESPRESSO (DFPT with electronic smearing + Fermi surface; `epw` optional).

---

### One-line summary
Use the MLIP (FC-distilled, to undo the §2.1 softening) + TDEP/SSCHA to get **lattice-anharmonic** ω(q,T)
cheaply across temperature, build a robust **kink/Kohn-anomaly locator** to extract q\*(T), cross-check
q\* ≈ 2k_F and the *electronic* temperature broadening against **DFPT-with-smearing** (the part the MLIP
structurally cannot produce) — and report the ballistic-transport connection as a tested hypothesis with
honest (L)/(E) provenance, not an assumption.
