# v2 parity freeze — September 2026

This note records the final HomotopyContinuation.jl v3 catch-up against upstream
HomotopyContinuation.jl v2.22.4 and the implementation decisions made in the September 2026 PR
tranche.

During the rewrite the v3 package was temporarily named `HomotopyContinuationNext` so it could be
loaded beside v2. PR #12 restores the registered `HomotopyContinuation` package name and UUID for
the v3 release line. That identity transition belongs to the parity/release freeze rather than to
a new numerical feature.

It is intentionally a freeze record rather than another roadmap. `02_status.md` is the current
project-state document and `03_v3_vs_v2.md` is the architectural/capability comparison.

## Freeze verdict

As of 2026-09-14:

> v3 is feature/capability complete relative to upstream v2.22.4.

No known v2 subsystem or post-audit v2 feature remains to be ported. Remaining work belongs to API
stabilization, release documentation, v3-specific cleanup, performance characterization or new v3
features.

The parity claim rests on the earlier testset-by-testset v2 audit plus the late-v2 catch-up below.
The resulting merged v3 tree was validated as one integrated tree by the permanent CI matrix.

## Package identity restoration

v3 is the next major version of the registered HomotopyContinuation.jl package. Its release
identity is therefore restored to:

```text
name = HomotopyContinuation
uuid = f213a82b-91d6-5c5d-acf7-10f1c761b327
version = 3.0.0-DEV
```

The temporary `HomotopyContinuationNext` name and UUID were development scaffolding only. The core
entrypoint is now `src/HomotopyContinuation.jl`, the extension modules are `DistributedExt` and
`SemialgebraicSetsExt`, and the split certification package is named
`HomotopyContinuationCertification`.

This changes the comparison topology. v2 and v3 now intentionally share one package identity, so
they cannot be installed as two package dependencies in one Julia environment. The historical
same-process comparison files were removed rather than naively renamed: doing so could resolve
both sides to v3 and produce a false parity result. Live v2/v3 comparisons must use isolated Julia
processes/environments.

The fixed parity regressions (`v2_parity_test.jl`, `monodromy_v2_parity_test.jl`, and the many
ported v2 cases in ordinary tests) remain active. The executable source, tests, benchmarks,
extensions and certification package all use the restored identities directly; no old-name
forwarding package or backing source tree remains in the active repository.

## Final PR tranche

### PR #1 — predictor trust region at zero crossings

The predictor now uses a finite unit-radius fallback when no valid derivative trust radius exists,
rejects non-finite/non-positive radii, and keeps an available derivative-derived radius
authoritative.

### PR #2 — LinearSubspace endpoint dimensions and promotion

The implementation now supports the complete `0 <= dim <= n` / `0 <= codim <= n` range,
affine point/full-space slices, safe zero-point redraws, and coefficient promotion when translating
a real subspace by a complex displacement.

### PR #3 — successful Result representatives for certification

`certify(F, result)` certifies one successful representative per `Result` cluster rather than
calling `solutions(result)`. A tracker singularity label is not proof of rigorous
non-certifiability; failed paths remain excluded and raw duplicates remain collapsed.

### PR #4 — principal-branch logarithm

`log` is a first-class operation through the full stack:

```text
Expression -> SExpr -> instruction sequence -> interpreter / RGF
          -> symbolic derivative -> Taylor -> DoubleF64 / ComplexDF64
          -> interval arithmetic -> Acb certification
```

Interval and Acb backends reject enclosures crossing the negative-real-axis branch cut.

### PR #5 — witness-set ambient-coordinate identity

Witness-set intersections require the same ordered ambient variables. Positional renaming is
rejected because equal dimension does not imply equal ambient geometry.

### PR #6 — regeneration ordering and tolerance ownership

Final semantics:

- `EquationSorting.BY_DEGREE` sorts in increasing degree;
- `EquationSorting.RANDOMIZED` keeps decreasing-degree pre-order before triangular randomization;
- outer regeneration/intersection `atol` / `rtol` own witness-point identity;
- internal monodromy runs with `equivalence_classes = false`;
- final physical witness deduplication does not pass `group_actions` to `unique_points`.

Symmetry may help discover connectivity but must not redefine witness cardinality.

### PR #7 — preserve unresolved NID witness sets

Trace-test monodromy discoveries are absorbed into persistent point identity. Only trace-certified
orbits are marked irreducible; budget exhaustion returns remaining orbits as
`WitnessSet(...; irreducibility = Irreducibility.UNKNOWN)`.

Public semantics:

- `irreducible_components` — proven components;
- `unresolved_witness_sets` — unresolved geometry;
- `unresolved_degree` — unresolved witness degree;
- `ncomponents` / `degrees` — proven irreducible components only.

### PR #8 — monodromy endpoint admission

Monodromy now rejects singular endpoints before future admission, revalidates genuinely new
heuristic endpoints at the base problem, preserves the rigorous certified-duplicate path, shares
identity semantics across executors, and zero-pads incomplete permutation histories.

### PR #9 — fixed-parameter custom homotopies

`FixedParameterHomotopy` binds external parameters of a user-defined `AbstractHomotopy` before it
enters the monomorphic evaluator firewall. It validates parameter count, delegates all evaluation
modes, forwards metadata/transforms, supports cloning, and presents zero free parameters to the
ordinary solve route.

### PR #10 / #11 — permanent v3 development CI

Permanent CI covers:

- core tests on Julia 1.10 and current Julia;
- certification tests on Julia 1.10 and current Julia;
- package-level JET analysis;
- Runic formatting;
- steady-state benchmark tracking on pinned Julia 1.13;
- representative fresh-process TTFX tracking;
- full TTFX inventory on pushes to `v3`.

Benchmark baselines live in the Actions cache; workload failures are fatal; benchmark thresholds
are explicit; TTFX alerts are informational while TTFX workload failures are fatal.

## Post-merge integration correction

A pre-merge review of PR #6 found that one intermediate version passed `group_actions` to the final
regeneration `unique_points` call, which would have collapsed physical witness points by symmetry.
The merged integrated tree does **not** contain that defect. Final deduplication retains the outer
distance/triangle/tolerance policy with no group-action quotient.

## Upstream changes covered by the freeze

| Upstream change | v3 disposition |
|---|---|
| threaded early-stop result preservation | covered by v3 executor/early-stop implementation |
| polyhedral progress suppression/plumbing | equivalent behavior; different mixed-cell path |
| NID randomization | `EquationSorting.RANDOMIZED` |
| NID tolerance propagation | outer regeneration/decomposition tolerance ownership |
| witness/intersection fixes | ambient-variable contract/current intersection implementation |
| full/zero-dimensional subspaces | PR #2 |
| Taylor integer power with zero constant term | division-free repeated squaring path |
| principal-branch `log` | PR #4 |
| predictor non-finite trust-region fallback | PR #1 |
| v2 compiled-cache locking | inapplicable: v3 has no global per-system compile table |

## Deliberate policy differences

### Decomposition iteration budget

v3 defaults to `max_iters = 50`, while late v2 substantially increased its retry budget. Because
v3 preserves unresolved witness sets, this is a work-budget policy rather than a silent
correctness risk.

### Public API architecture

Feature parity does not imply exact API spelling. v3 uses typed algorithms, scoped enums,
executors, explicit fixed-parameter objects and a separate certification package. The release
freeze must declare stable public extension points without recreating obsolete v2 internals.

## Integrated validation

The September catch-up was not declared complete from individually green PRs. The combined `v3`
tree passed:

- core Julia 1.10;
- core current Julia;
- certification Julia 1.10;
- certification current Julia;
- JET;
- Runic;
- benchmark tracking;
- TTFX tracking;
- full TTFX inventory.

The identity-restoration work in PR #12 must satisfy the same rule: the final exact rename head is
not merge-ready until its integrated matrix is green.

## What remains after parity

- finish/certify the package identity restoration in PR #12;
- formal public API/stability declaration;
- release and migration documentation;
- rebuild optional live v2/v3 comparison tooling around isolated environments;
- direct final compile-mode performance characterization;
- v3-specific executor reach/cleanup in internal witness operations;
- consolidation of duplicated orchestration/progress/interface machinery where useful.

New work should not reopen a generic "v2 parity" bucket unless a concrete missing v2 capability is
identified.
