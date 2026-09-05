# Graphene R2R-0 formal representation precheck contract

v8 is the fresh attempt3 recursive-receipt contract. It retains the v5 formal combined-E/F
adapter binding and changes only the finite-difference adjudication point, the
q portability/zero-jet gate, and launcher log/freshness closure justified by
the completed R2R-0D `ATTRIBUTED_BOTH` diagnostic. The adapter must
return exactly `(float energy, FP64 source-order force array, unmodified query
receipt)` from the production combined path. A mock control-flow regression and
a real thermal0 proper-O(3) regression exercise this adapter directly; the O(3)
provenance predicate is tested independently so a misplaced or unreachable
return cannot be hidden by a receipt-only test.

This stage is train-only, label-blind and fit-free. It materializes the corrected
live-weighted carrier, the fixed 65-column force design, the harmonic rank-1
null receipts, and an independent mechanics receipt. A passing aggregate is
only a representation precheck; it does not authorize fitting, training,
held-data access, deployment, or a phonon claim.

The extxyz geometry loader retains only `Properties`, `Lattice`, `pbc`, and
`config_type` header fields; energy, virial, stress, and all other header labels
are discarded immediately. It requires species width 1, position width 3,
Lattice width 9, and pbc width 3. Other atom-row tokens are opaque skips: they
are neither converted to numbers nor retained. Receipts state
`header_label_values_retained=false` and
`numeric_label_columns_converted=false`.

The thermal partition is fixed by `global_index % 3`: shard 0 has 31 structures,
shard 1 has 31, and shard 2 has 30. Harmonic-zero uses the same rule and has
11/11/10 structures. Every shard additionally evaluates thermal indices
0, 20, 56, and 91 as cross-hardware sentinels without adding them to its matrix.
Sentinel arrays use the symmetric comparison
`|x-y| <= atol + 5e-7 max(|x|,|y|)` elementwise. The fixed and 65 parameter
energy columns use `atol=1e-9 eV`; force columns use `atol=1e-8 eV/A`; `b`,
`a`, and `c` use `1e-18 A^4`, `1e-12`, and `1e-12`. The canonical combined
probe additionally uses `1e-6 eV` and `1e-5 eV/A`.

Mechanics consumes the canonical R2R thresholds without redefining them. The
finite-difference coordinate remains zero-based `(2,1)`. It evaluates the fixed
steps `8e-4, 4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5 A`, always minus then plus. Both
formal finite-difference decisions use only the preselected `h=5e-5 A` row and
the unchanged limits `1e-5 eV/A` for energy-to-force and `1e-5 eV/A^2` for
force-to-Hessian. All six raw energy/coordinate-force pairs are stored in the
mechanics NPZ. Empirical orders, left/right slope jumps, Richardson values and
the energy multistep path remain recomputed report-only diagnostics; they can
neither choose another step nor add a post-result numerical gate. Exact formal
graph and assignment/MIC provenance is required at the base and every
minus/plus point.

For each 6x6 and 8x8 reference, mechanics independently constructs the actual
production neighborhood on CPU and CUDA in FP64. The formal gate requires
receiver, sender, periodic image and the complete per-node production source
order to be array-exact; the CPU discrete hashes and source-order hash must
also match the R2R-0D-derived canonical constants. The numeric comparison is
`max(maxabs(center_q_CPU-center_q_CUDA),
maxabs(neighbor_q_CPU-neighbor_q_CUDA)) <= 5e-13`. The quantized payload is
`concat(center_q in node order, neighbor_q in production edge order)`, rounded
with `np.round(..., decimals=12)` and converted to contiguous little-endian
FP64. CPU and CUDA payloads must be elementwise equal and have the frozen
canonical SHA-256. Raw FP64 q-orbit hashes are recorded only as diagnostics.

Both devices and both references run the full physical zero-jet audit. The
aggregate recomputes from the mechanics NPZ and receipt that value, Jacobian
and every-node local Hessian are exact zero and finite, with exact coverage,
unique senders, 40 local samples, the reference-specific node count and full
`[40,3,40,3]` Hessian shape. It also recomputes the nonzero local-to-production
value difference directly from raw `local_a`/`production_a`, and reconstructs
the mapped and outside gradients from raw actual-device local/global gradients
plus the corresponding NPZ source order. Stored difference scalars and `pass`
Booleans are diagnostic only and cannot satisfy a threshold. The zero-jet
source-order hash must equal the one independently reconstructed from its
corresponding NPZ arrays.

Apart from these fixed attempt3 rules, mechanics
checks reference and thermal-index-0 full Hessians, corrected-carrier parity to
the frozen R2Q whole Taylor remainder, node parity, proper/improper O(3),
translation, pure permutation, native-cell wrapping/order-MIC, finite
differences, deterministic 6x6-to-8x8 source-site locality, no-wrap geometry,
the background 6 A quintic-smootherstep C2 limit, the separate MACE r=3.2 A
`PolynomialCutoff(p=5)` C2 limit, the slow per-node zero-jet audit, and the R2O Weyl
Gamma/K drift upper bound. The Weyl value is a bound, not an actual phonon
frequency.

Reference semantic payloads use the R2R v3 signed-zero rule: after the ordered
fractional coordinates and cell metric are rounded to 10 decimal places, exact
IEEE zeros are canonicalized to `+0.0` before JSON hashing. This fixes rigid O(3)
equivalence across BLAS implementations without clamping any nonzero value;
reference tampering remains fail-closed. Failed attempt1 and attempt2 artifacts
are retained unchanged. Attempt2 failed only the `h=1e-4 A`
force-to-Hessian subgate (`1.3696790006e-5 eV/A^2`) and the raw q-orbit hash
comparison; its energy-to-force error was `1.8687856714e-6 eV/A`, while both
physical zero-jet audits and source-order hashes passed. R2R-0D subsequently
reported `ATTRIBUTED_BOTH`: at `h=5e-5 A` the energy and Hessian errors were
`4.6717795266e-7 eV/A` and `3.4242256781e-6 eV/A^2`; both references had exact
CPU/CUDA topology/source order, normalized-q maximum difference below
`1.813e-16`, round12/13 equality and exact physical zero jets. R2R-0D remains a
non-adjudicating diagnostic and is not a runtime formal input. Any authorized
continuation uses a new manifest-derived run id and fresh `attempt_0003` paths.

The O(3) mechanics probe uses the R2R v4 covariant graph path. It first rebuilds
and exact-hash verifies the frozen 6x6 baseline formal graph. Only `positions`,
`shifts`, and `cell` are multiplied by `Q.T`; `edge_index`, `unit_shifts`,
`node_attrs`, `batch`, `ptr`, `head`, and `pbc` remain byte exact. The exact
baseline `ReferenceNeighborhood` topology, distances, quintic weights, and
normalization are reused, while the live centered displacement vectors rotate
with the synchronized structure/reference. The public call also binds thermal
global index 0 as an explicit baseline structure template, the reference6 full
geometry identity, and all three assignment/MIC arrays. Native rotated graph
rebuilding is retained only in a `diagnostic_only` receipt/API; its graph-order
and combined E/F sensitivity cannot enter the O(3) pass decision. This changes
neither baseline 6x6/8x8 graph hashes nor any shard design semantics or gate.
The supplied reference and live structure positions/cell must be array-exact to
their baseline templates rotated by `Q.T`. Internal ASE ordered/adapted arrays
are bounded separately by the canonical named absolute tolerance `1e-12 A`
with `rtol=0`, and their three maxima and gate Booleans are recorded. Native MACE/background rebuilds can
veto only when their exact physical edge identity-key sets differ. Their vector,
length, distance, weight, normalization, and cell differences remain diagnostic
scalars and do not carry a hidden numerical threshold.

Aggregation reloads all three independent thermal/harmonic/sentinel artifacts
and the mechanics artifact, verifies their hashes, schemas, markers, partitions,
configuration identities, endpoint/reference/graph/column/coefficient hashes,
and recomputes all numerical gates. Mechanics FD and q decisions are recomputed
from the hash-bound NPZ arrays; zero-jet physical/parity predicates are
recomputed from their numeric receipts and bound back to those arrays. It never
trusts a stored pass Boolean. Fresh aggregation and completed-DONE recovery use
the same recomputation function for canonical aggregate arrays, sentinels,
train geometry gates, SVD rank/condition, borderline classification, mechanics
and final status. Recovery additionally requires every stored aggregate array
to be dtype-, shape- and value-exact to reconstruction from the current three
shards, and requires every stored decision field and status to equal that same
reconstruction.

Fresh construction and every completed recovery path first validate a complete
recursive key-path schema, independently of key spelling or value. Dictionary
containers and keys are encoded as `D` and `K` full paths; list containers use
`L`, every element is normalized to the `[]` path, and all elements of a list
must have an identical recursive schema. Scalar values are outside this
structural hash. Any extra or missing key at any receipt depth, including an
unrelated key whose value is `true`, `false`, or `0`, therefore fails before a
stored receipt hash or DONE marker can be accepted. The frozen core path
counts/SHA-256 values are: synthetic preflight 77/
`ea0aa220598dc4c6517c70fcf9d75fd53aa3fe31a172719db1bc9cad8375e394`;
real preflight 266/
`6f19a3efdb9793a54cc56ef42b0f0a9db1281261d0850b8c1d7a27d07c69cc07`;
shard 654/
`673acd629b9d7f6df15ab4a26a290674b3eebfcb0c03b3fbc900677a6db2f23e`;
mechanics 1815/
`a87b3d99c9c83c5436b425431efceba51bdf021a3e7e0daababd596e806f39fa`;
and aggregate 455/
`c64f92d287f98564865512943ff2341e703e0388a5ddf65f7a909ac4f8b75aaa`.
Core receipts have no wildcard mapping path.

Exact versioned formats, top-level key sets, and safety semantics are checked
again after the structural gate. Declared no-label, no-fit, no-training and
no-held/support/seed fields must be the JSON Boolean `false`, not a false-like
number or string. The geometry-only loader map has its own exact key set and
exact Boolean `false` values for header retention, numeric label conversion,
and force/energy label parsing, loading and use. The path-aware alias scanner
remains defense in depth: it splits camelCase, lowercases keys and also examines
their delimiter-free compact form. Launcher DONE recovery additionally requires
`fit_or_training=false` and `held_or_support_access=false` and revalidates the
collected aggregate safety closure. The launcher receipt has 211 paths and
schema SHA-256
`a6eb1777c1504987eba4f590f10f428cf2741ce08983587c7f25e40599633145`.
Artifact and control manifest file names are part of that exact schema. Only
`log_manifest.files` uses a controlled dynamic-key path, whose complete 660-file
keyset and contents are independently reconstructed by the command inventory.
Rewriting a receipt hash and DONE marker cannot bypass these checks.
All 65 columns must be active, rank 65, and have scaled condition number at most
the canonical `1e8`; no reduced design is allowed. A result within the fixed
1% borderline band is numerically inconclusive and requires the separately
authorized complete V100-A replay.

Harmonic-zero materialization keeps two independent energy/force pairs: the
corrected fixed carrier and the canonical fixed-plus-65-column probe. Each must
separately satisfy `|E| <= 1e-10 eV` and `max|F| <= 1e-9 eV/A`; cancellation
between them cannot authorize the null gate. No harmonic 65-column matrix is
computed or fitted.

Formal execution is fail-closed behind an external freeze manifest and a
separate authorization marker. The manifest binds the live formal core, shard
CLI, aggregate CLI, launcher, tests, this document, the frozen R2R primitive,
the exact R2O wrapper and evaluator, formal
contract, and all seven input content hashes. The authorization marker contains
exactly the manifest SHA-256 plus a newline. All execution CLIs require both
paths and verify their own and their dependencies' hashes before work. No GO
marker is created by this implementation/review stage.

Raw paths containing `..`, forbidden held/seed/support tokens, or a symlink in
any existing path component are rejected before path resolution. Successful
recovery requires exactly one terminal state (`DONE`, not `FAILED` or
`RUNNING`), exit code zero, an exact receipt-bound marker, current contract and
source/input provenance, and exact artifact schemas/hashes. Failure and
inconclusive states are diagnostic only.

Formal node allocation is fixed to `root@100.80.236.112` for V100-A/shard0
(`phonon-mlip`), `root@100.123.220.57` for V100-B/shard1 and mechanics
(`phonon-mlip`), and `howardwang@100.105.21.7` for RTX-2060/shard2 (`phonon`).
No other SSH/SCP target is accepted. A/B use
`/root/miniconda3/bin/conda`; RTX uses
`/home/howardwang/miniconda3/bin/conda`. Every execution sets
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Transport is Tailscale-only. The launcher
is executable only after a separately supplied valid GO marker; this stage does
not launch it. It stages an immutable, hash-verified bundle on each node,
including all source snapshots and all seven inputs. Each node's complete
manifest-derived bundle root (and therefore its control root) must be absent
before the first write; a pre-existing ordinary file or directory fails the
same prewrite guard without changing the command inventory. The remote output
path is derived only from the manifest SHA prefix and frozen attempt number 3.
Existing RUNNING/FAILED outputs are rejected; a validated DONE can be
recovered only for the already completed, identical attempt3 collection; it is
not scientific artifact reuse. Before any role starts, shard0, shard1, shard2
and mechanics output roots must all be absent, including DONE roots. No
attempt1/attempt2 shard or mechanics artifact can be copied or recovered.

The successful launcher inventory is fixed at 220 commands: 67 prewrite
symlink guards, four fresh-output checks, nine mkdir commands, 60 staging SCPs,
60 SHA checks, three final symlink finds, eight role marker prechecks, four role
runs, four collections and one local aggregate. Every command, including an
empty-output `test` or `find`, has one command JSON and separate stdout/stderr
files, for exactly 660 flat ordinary log files. Every command JSON has the same
ten keys: argv/hash/return code, command/stdout/stderr basenames, both stream
hashes, and expected/observed staged hashes. The last pair is JSON null except
for a staging SHA check, where both are the same verified 64-hex digest.
Completion and recovery both
require all 220 return codes to be zero, unique ordinary stream basenames,
exact stream hashes and an exact no-orphan file inventory. All concurrent shard
children are joined and their logs are closed before a failure decision. The
launch receipt additionally binds four collected artifact manifests and the
aggregate scientific PASS/FAIL/INCONCLUSIVE status.

The fixed workload is 92 unique thermal structures, 32 harmonic-zero
structures, four sentinels on each of three nodes (12 evaluations), and one
complete V100-B mechanics run. The expected total is at most 0.75 GPU-hour. An
overrun is recorded with actual elapsed time and peak memory for manual review;
it never permits lower precision, fewer probes, weaker thresholds, or fitting,
and does not create a fourth scientific status. CUDA receipts bind the
hostname, exact `nvidia-smi` driver query/stdout hash, CUDA runtime, and cuDNN.
