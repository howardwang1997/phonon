# Graphene R2R-0D numerical diagnostic contract

R2R-0D is an independent post-attempt2 diagnostic. It may explain the two
failed numerical checks, but it does not re-evaluate the frozen formal result.
The attempt2 scientific status remains
`R2R0_FORMAL_REPRESENTATION_PRECHECK_FAILED` for every possible R2R-0D output.
No R2R-0D status is a formal pass or permission to fit, train, deploy, access
held/support data, or claim a phonon result.

The canonical machine-readable contract is `DIAGNOSTIC_CONTRACT` in
`scripts/smearing_kink/graphene_r2r0_diagnostic.py`. Its semantic SHA-256 is
stored as `DIAGNOSTIC_CONTRACT_SHA256` and is recomputed by the tests.
Hardened diagnostic and CPU-preflight receipts use their respective v2 format
identifiers; v1 output is not eligible for DONE recovery.

Execution additionally requires an external
`graphene_r2r0d_external_freeze_manifest_v2` manifest and a separate
authorization marker. The manifest records the exact SHA-256 of the diagnostic
core, CLI, tests, and this document, plus the canonical contract, v5
formal/primitive/input snapshot, and the complete attempt2 binding described
below. None of the four source hashes is embedded in a source file, so this
scheme has no self-referential hash. The marker contains exactly the manifest
SHA-256 followed by one newline. Its filename must not contain `GO`.

`prepare_freeze_manifest` is a local-only preparation function. It creates only
the candidate manifest and explicitly records
`authorization_marker_created=false`; it cannot create the authorization
marker or a GO marker. Both manifest and marker, and all four current sources,
are checked before an output directory is created. They are checked again after
the calculation and the two identical authorization receipts are stored in the
final receipt.

## Immutable inputs and isolation

The runner validates the exact v5 manifest
`R2R0_formal_train_only_candidate_attempt2_v5_20260825/freeze_manifest.json`
with SHA-256
`f2abdb5e997db69e0f406458c2dd3b0331e22d0e5d8828b4986b6f964c8753a6`.
It verifies every v5 formal/primitive source and input hash, the canonical R2R
hash, the attempt2 control manifest and GO snapshot, the launch receipt, and
every collected shard/mechanics/aggregate artifact manifest. In particular it
binds:

- root `DONE` and `EXIT_CODE`:
  `665cf8bbf78f9cb37587150c7358bb0d5e7fb8fac460ef5e1e4a4830140d1587`,
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`;
- aggregate receipt and arrays:
  `63c89ff68e4799e4d1ccc3e207835830df9144928e49db3b9b8a3b3fe1830d6e`,
  `e93646f5d18218e8a7283fa8835a23a37c1248d8a802e6f63245814dc53e0a77`;
- mechanics receipt and arrays:
  `6aa8e6cd059a8eb55aa1def117e73147d89d572bd65ac4a33782806a4daa9187`,
  `f84607c41d7879798701cb77979564779ae680a59909f98723c37337f0ae65c7`;
- launch receipt:
  `eeff934de678904eb43ce9d24acf7c72594228f974a30e5a4abc1bd27a1f384a`.

R2R-0D writes only to a strict descendant of the fixed
`RECOMMENDED_OUTPUT_ROOT`. Both the caller's raw absolute path and its resolved
path must remain strict descendants. The root itself, `..`, every existing or
dangling symlink component, and names containing `seed1`, `seed2`, `support`,
`held`, or `holdout` are rejected. It never modifies or creates a marker under
attempt2 and never creates a file named GO. There is no R2R-0D remote launcher
in this package.

The only live structure is `train_thermal.xyz` global index 0, loaded through
the frozen geometry-only extxyz parser. The reference structures are
`reference_6x6.xyz` and `reference_8x8.xyz`. The parser converts only species,
positions, lattice, PBC, and `config_type`; force/energy/stress/virial fields are
not converted or attached to `Atoms`. The harmonic file is byte-hashed as part
of the frozen seven-input manifest but its records are not parsed. Seed1,
seed2, held-small, and support data are forbidden.

## Fixed finite-difference diagnostic

The probe is atom 2, Cartesian y, using zero-based coordinate `(2,1)` and flat
coordinate 7. The steps are evaluated exactly in this order:

`[8e-4, 4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5] Å`.

For every step, minus is evaluated before plus. The base and all displaced
energies/forces go through the unchanged public combined E/F adapter
`graphene_r2r0_formal._combined_ef`, which calls
`production_combined_energy_force`. Full raw plus/minus force arrays are stored
in the diagnostic NPZ, while the receipt stores raw energies and the selected
force coordinate. Every base/plus/minus force reference includes its exact NPZ
key, shape, dtype, and raw-byte SHA-256. A global array schema is recomputed
before write; every nested receipt reference must resolve exactly, every schema
key must be referenced exactly once, and missing/duplicate keys stop the run. Every
point also records the formal graph hash and baseline-match flag,
assignment/MIC identity hash, maximum assignment distance, and minimum
uniqueness gap.

The base energy and live force are evaluated once. The frozen targets are model
outputs from the hash-bound attempt2 mechanics array, not dataset labels:

- `F0 = thermal0_force_eV_A[2,1]`;
- `J0 = -thermal0_Hessian_eV_A2[7,7]`.

The runner records the live-base/artifact-force difference. It does not recompute
a full Hessian.

For each step `h`, the reported quantities are

- `D_E(h) = -(E(+h)-E(-h))/(2h)` and `abs(D_E-F0)`;
- `D_F(h) = (F_y(+h)-F_y(-h))/(2h)` and `abs(D_F-J0)`;
- `L(h) = (F0-F(-h))/h`, `R(h) = (F(+h)-F0)/h`, and
  `jump(h)=abs(R-L)`;
- adjacent-step empirical orders `log2(err(2h)/err(h))`; if either error is
  exactly zero, the order is null and the zero case is explicit;
- `R_F(h)=D_F(h)+(D_F(h)-D_F(2h))/3` and its absolute error to `J0`.

`FD_TRUNCATION_CONFIRMED` requires all of the following without choosing a step
after seeing the result: the six force errors strictly decrease; all five force
orders are in `[1.8,2.2]`; all five jump orders are in `[0.8,1.2]`; the raw
force-Hessian error at `h=5e-5 Å` is at most `1e-5 eV/Å²`; and the final three
Richardson errors are each at most `1e-7 eV/Å²`.

This classification is deliberately scoped to the failed force-to-Hessian
subgate. The hash-bound attempt2 identity is recorded explicitly: its
energy-to-force error was `1.8687856714e-6 eV/Å`, already below `1e-5`, while
its force-to-Hessian error was `1.3696790006e-5 eV/Å²`, above `1e-5`. Energy
multistep errors and orders remain report-only because they are not the failed
subgate and the finest energy differences can enter cancellation before the
force differences do.
The receipt nevertheless reports whether all six energy errors decrease and
all five energy orders lie in `[1.8,2.2]` as
`energy_path_consistent_report_only`; a false value is never described as an
overall smooth-path result and does not silently become an additional gate on
the already scoped Hessian attribution.

Any graph/assignment topology change, or two adjacent jump orders outside
`[0.5,1.5]`, gives `NONSMOOTH_CANDIDATE`. With unchanged topology, a smallest-h
force error at least 0.8 times the preceding error gives
`FLOAT_FLOOR_CANDIDATE`. These classifications do not relax the formal gate.

## Fixed q and zero-jet diagnostic

For both 6×6 and 8×8 references, the runner independently constructs the
production `fixed_reference_neighborhood` on CPU and CUDA in FP64. It stores
contiguous little-endian arrays for receiver, sender, periodic image,
reference distance, quintic weight, normalization, center q, neighbor q, and
per-node production source order. Shapes, dtypes, and raw byte hashes are in
the receipt; the arrays are in the NPZ.

CPU/CUDA comparison reports exact-different count, maximum absolute and
relative differences for every floating array. For nonnegative arrays it also
reports maximum, p99, and nonzero-median ULP differences. Per-node comparison
reports sorted-q maximum difference and the nodes whose stable q sort induces a
different sender sequence.

For decimal depths 12 through 16 the payload is
`concat(center_q in node order, neighbor_q in production edge order)`. Each
payload is rounded with NumPy, converted to contiguous little-endian FP64, and
stored and hashed. CPU and CUDA each run the actual full local-Hessian
`background_rank0_zero_jet_audit`; global value/Jacobian, every-node local
value/Jacobian/Hessian, finiteness, coverage, sender uniqueness, source-order
hash, and the nonzero local-production probes are recorded.

`Q_FLOAT_SERIALIZATION_ONLY` requires exact CPU/CUDA discrete topology and
source order, maximum normalized-q difference at most `5e-13`, elementwise and
hash equality at both 12 and 13 decimals, and exact-zero finite physical
value/Jacobian/full-Hessian zero jets on both devices and both sizes. The
audit's nonzero local-to-production value/gradient parity must also pass, so the
independent local Hessian is bound back to the production implementation. A raw
FP64 q-orbit hash difference is explicitly diagnostic only and is never a gate.

## Statuses, stops, and budgets

If both frozen attributions hold, the status is `ATTRIBUTED_BOTH`. FD alone is
`DIAGNOSTIC_TRUNCATION`. An FD nonsmooth or float-floor candidate takes
precedence over a q-only attribution; otherwise q alone is
`NUMERICAL_IDENTITY`. Remaining statuses are `INCONCLUSIVE`,
`DIAGNOSTIC_BUDGET_EXCEEDED`, and the non-attributing
`DIAGNOSTIC_PREFLIGHT_ONLY`. This precedence prevents a q serialization result
from hiding a stronger FD counter-diagnosis.

The run stops fail-closed on any source/input/artifact hash difference, changed
attempt2 status, label contamination, nonfinite value, or CPU/CUDA discrete
topology/source-order difference. It does not automatically retry, change the
probe, change a step, relax a threshold, or switch precision.
The full diagnostic accepts only a normalized CUDA index whose exact PyTorch
device name is in the frozen allowlist, currently
`Tesla V100-SXM2-32GB`. Other CUDA GPUs, including the RTX 2060, stop before any
scientific evaluation. The default PyTorch dtype must originate as FP32. The
CPU preflight remains CPU-only.

Every new run begins with the sole terminal marker `RUNNING`. A successful
completion atomically writes the receipt, writes `EXIT_CODE=0`, removes
`RUNNING`, and writes `DONE` as `status + newline + receipt_SHA256 + newline`.
An early stop atomically writes its receipt, writes `EXIT_CODE=2`, removes
`RUNNING`, and writes an identically bound `FAILED` marker. Thus terminal
markers satisfy XOR. A directory containing `RUNNING`, `FAILED`, or partial
files is never resumed or retried.

A pre-existing `DONE` directory is recoverable without recomputation only after
checking exact root inventory, terminal XOR and exit code, receipt/DONE binding,
manifest and marker, all four live sources, canonical/v5/input/attempt2
bindings, NPZ file hash, every key/shape/dtype/raw-array hash, complete receipt
reference coverage, and the recorded artifact manifest. The returned in-memory
object states `completion_recovered_without_recompute=true`; the frozen receipt
is not rewritten.

The fixed upper bounds are 120 seconds of recorded CPU wall time and 120 seconds
of recorded GPU wall time. An overrun gives
`DIAGNOSTIC_BUDGET_EXCEEDED` and no formal marker. The CPU preflight performs
only provenance, geometry-label audit, and CPU q materialization; it cannot make
an attribution.

Example local preflight:

```bash
conda run -n phonon python scripts/smearing_kink/run_graphene_r2r0_diagnostic.py \
  prepare-freeze-manifest \
  --output results/graphene_physics_temperature/post_p4_feasibility/R2R0D_candidate/freeze_manifest.json

# An independent authorizer writes R2R0D_DIAGNOSTIC_AUTH as exactly:
#   sha256(freeze_manifest.json) + "\\n"

conda run -n phonon python scripts/smearing_kink/run_graphene_r2r0_diagnostic.py \
  cpu-preflight \
  --output results/graphene_physics_temperature/post_p4_feasibility/R2R_multipolar_background/R2R0D_numerical_diagnostic/cpu_preflight \
  --freeze-manifest results/graphene_physics_temperature/post_p4_feasibility/R2R0D_candidate/freeze_manifest.json \
  --authorization-marker results/graphene_physics_temperature/post_p4_feasibility/R2R0D_candidate/R2R0D_DIAGNOSTIC_AUTH
```

A full diagnostic additionally requires an explicit CUDA device and
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Authorization for such a run is separate
from this implementation; this package itself performs no remote or GPU launch.
The requested CUDA index is normalized once; every synchronization and runtime
hardware query uses that exact index, including when it differs from PyTorch's
current device. CUDA is synchronized immediately before and after every timed
GPU block. CPU and GPU recorded wall budgets are independently inclusive at
120 seconds; the first value above either boundary produces the bound FAILED
terminal and `DIAGNOSTIC_BUDGET_EXCEEDED` receipt.
