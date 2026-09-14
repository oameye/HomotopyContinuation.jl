# v2 parity freeze — September 2026

This note records the final HomotopyContinuationNext.jl (v3) catch-up against upstream
HomotopyContinuation.jl v2.22.4 and the implementation decisions made in the September 2026 PR
tranche.

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

## Final PR tranche

### PR #1 — predictor trust region at zero crossings

Problem: the predictor could derive a non-finite or non-positive trust radius around a zero
crossing when the usual derivative-based scale was unavailable.

Resolution:

- use a finite unit-radius fallback when no valid derivative radius exists;
- guard regular and Hermite predictor modes against non-finite/non-positive radii;
- keep a valid derivative-derived radius authoritative.

This closes the late v2 predictor fallback behavior without restoring v2's broader implementation
architecture.

### PR #2 — LinearSubspace endpoint dimensions and promotion

Resolution:

- allow the complete dimension range `0 <= dim <= n` and `0 <= codim <= n`;
- support affine point slices and full-space slices;
- reject a zero-dimensional *linear* subspace through a nonzero point;
- avoid the zero-point normalization singularity in `rand_subspace!(...; affine = false)`;
- promote coefficient types when a complex translation is applied to a real subspace;
- preserve the original subspace under translation.

These endpoint cases are important because witness/NID and projective code naturally construct
full- and zero-dimensional intermediate slices.

### PR #3 — successful Result representatives for certification

`certify(F, result)` now certifies one successful representative per `Result` cluster rather than
calling `solutions(result)`, which filters numerically singular solutions.

The invariant is:

```
tracker singularity label != rigorous non-certifiability
```

Failed paths remain excluded and the cluster partition still prevents raw duplicate
re-certification.

### PR #4 — principal-branch logarithm

`log` is a first-class operation through the complete v3 stack:

```
Expression
  -> SExpr
  -> instruction sequence
  -> interpreter / RGF codegen
  -> symbolic derivative
  -> Taylor recurrence
  -> DoubleF64 / ComplexDF64
  -> interval arithmetic
  -> Acb certification
```

Taylor coefficients use logarithmic differentiation (`(log a)' = a'/a`). Interval and Acb
backends reject enclosures crossing the negative-real-axis branch cut rather than accepting a
discontinuous principal branch enclosure in a Krawczyk proof.

### PR #5 — witness-set ambient-coordinate identity

Witness-set intersections require the same ordered ambient variables.

The previous positional-renaming idea is unsafe: two systems with different variable identities or
orders can have the same dimension but represent different ambient geometry. v3 therefore checks
exact variable identity/order and rejects mismatches.

### PR #6 — regeneration ordering and tolerance ownership

The final semantics are:

- `EquationSorting.BY_DEGREE` sorts in increasing degree;
- `EquationSorting.RANDOMIZED` retains decreasing-degree pre-order before the triangular random
  combination;
- the outer regeneration/intersection `atol` and `rtol` determine witness-point identity and are
  propagated into internal monodromy;
- internal monodromy is an orbit-discovery engine and runs with `equivalence_classes = false`;
- final physical witness deduplication does not pass `group_actions` to `unique_points`.

The last point is crucial. Symmetry may help discover connectivity but must not redefine witness
cardinality. A degree-d witness set remains degree d even if several points lie in one supplied
symmetry orbit.

### PR #7 — preserve unresolved NID witness sets

Numerical irreducible decomposition is now lossless with respect to its current witness-point
knowledge.

During decomposition:

- points discovered by trace-test monodromy are absorbed into one persistent point-identity
  structure;
- orbit connectivity survives index changes and solution-set growth;
- only trace-certified orbits are marked irreducible;
- when the iteration budget is exhausted, remaining connected orbits are returned as
  `WitnessSet(...; irreducibility = Irreducibility.UNKNOWN)`.

Public result semantics:

- `irreducible_components` — proven components;
- `unresolved_witness_sets` — unresolved geometry;
- `unresolved_degree` — total unresolved witness degree;
- `ncomponents` / `degrees` — proven irreducible components only.

This makes the v3 `max_iters` default a work-budget policy rather than a silent-correctness risk.

### PR #8 — monodromy endpoint admission

Monodromy now treats a point becoming a future start as an admission decision, not merely as a
successful loop endpoint.

The driver:

- rejects singular endpoints as future starts;
- revalidates genuinely new heuristic endpoints at the base problem;
- does not revalidate a duplicate that will not be stored as a new start;
- preserves the rigorous certified-duplicate path;
- owns the shared identity state for serial/threaded/distributed execution;
- records incomplete permutation histories with zero padding rather than dropping columns when the
  solution set grows.

### PR #9 — fixed-parameter custom homotopies

`FixedParameterHomotopy` binds external parameters of a user-defined `AbstractHomotopy` before it
enters the monomorphic `HomotopyEvaluator` firewall.

It:

- validates the parameter count;
- stores a concrete bound parameter vector;
- delegates value/Jacobian/Taylor evaluation to the wrapped parameter-aware homotopy;
- forwards variables, variable groups and coordinate transforms;
- supports independent cloning for threaded/distributed workers;
- presents zero free parameters to the ordinary solve route.

The same tranche exports the `AbstractHomotopy` interface methods required to implement such a
custom homotopy without depending on private evaluator internals.

### PR #10 / #11 — permanent v3 development CI

The temporary branch-specific validation used during development is replaced by persistent
repository CI.

The matrix includes:

- core tests on Julia 1.10 and current Julia;
- certification tests on Julia 1.10 and current Julia;
- package-level JET analysis;
- Runic formatting;
- steady-state benchmark tracking on pinned Julia 1.13;
- representative fresh-process TTFX tracking;
- the full TTFX workload inventory on pushes to `v3`.

Benchmark baselines are stored in the Actions cache rather than a gh-pages history branch. Workload
failures are fatal; benchmark regression thresholds are explicit; TTFX performance alerts are
informational while workload failure remains fatal.

## Post-merge integration correction

A pre-merge review of PR #6 found that one intermediate version passed `group_actions` to the final
regeneration `unique_points` call, which would have collapsed physical witness points by symmetry.

The merged integrated tree does **not** contain that defect. The final deduplication retains the
outer distance/triangle/tolerance policy but no group-action quotient.

This distinction is recorded here because it is easy to reconstruct the wrong conclusion from an
old PR-head review rather than the merged tree.

## Upstream changes covered by the freeze

The re-audit included the post-July upstream v2 changes through v2.22.4. Relevant changes are
accounted for as follows:

| Upstream change | v3 disposition |
|---|---|
| threaded early-stop result preservation | covered by v3 executor/early-stop implementation |
| polyhedral progress suppression/plumbing | equivalent user behavior; v3 has a different mixed-cell construction path |
| NID randomization | `EquationSorting.RANDOMIZED` |
| NID tolerance propagation | outer regeneration/decomposition tolerance ownership |
| witness/intersection fixes | ambient-variable contract and current intersection implementation |
| full/zero-dimensional subspaces | PR #2 |
| Taylor integer power with zero constant term | division-free repeated squaring path already in v3 |
| principal-branch `log` | PR #4 |
| predictor non-finite trust-region fallback | PR #1 |
| v2 compiled-cache locking | deliberately inapplicable: v3 has no global per-system compile table |

## Deliberate policy differences

### Decomposition iteration budget

v3 currently defaults to `max_iters = 50`, while late v2 substantially increased its retry budget.
The two defaults are not semantically equivalent because v3 preserves unresolved witness sets.

The release decision is therefore not "copy v2's number for parity". It is to choose the most
useful v3 work budget given that incompleteness is explicit and lossless.

### Public API architecture

Feature parity does not imply name-level API identity. v3 deliberately uses typed algorithms,
scoped enums, executors, explicit fixed-parameter objects and a separate certification package.
The release freeze must declare which v3 names are stable public extension points; it should not
recreate obsolete v2 internals merely to match exports.

## Integrated validation

The September tranche was not declared complete from individually green PRs alone. After the PRs
were merged, the combined `v3` tree passed the permanent validation matrix:

- core Julia 1.10;
- core current Julia;
- certification Julia 1.10;
- certification current Julia;
- JET;
- Runic;
- benchmark tracking;
- TTFX tracking;
- full TTFX inventory.

That integrated-tree requirement is part of the parity-freeze criterion for future large tranches
as well.

## What remains after parity

The remaining project work is intentionally outside this parity note:

- formal public API/stability declaration;
- release and migration documentation;
- direct final compile-mode performance characterization;
- v3-specific executor reach/cleanup in internal witness operations;
- consolidation of duplicated orchestration/progress/interface machinery where it improves
  maintainability without changing numerical semantics.

New work should not reopen a generic "v2 parity" bucket unless a concrete missing v2 capability is
identified.