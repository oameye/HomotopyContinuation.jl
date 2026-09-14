# Status

Last updated: 2026-09-14.

This file is the current implementation-status snapshot for HomotopyContinuationNext.jl (v3).
The detailed development history remains available in git history and in `01_decisions.md`; the
September 2026 parity-freeze audit is recorded in `09_v2_parity_freeze.md`.

## Reproduce

- `make test` — core plus certification test suites
- `make test-cert` — certification subpackage only
- `make benchmark` — steady-state timings
- `make compare` — v3/v2 benchmark ratios
- `make ttfx` — first-use workload inventory
- `julia --project=benchmark benchmark/compare/tracking.jl` — end-to-end solve comparison

The permanent GitHub Actions matrix on `v3` covers core and certification on Julia 1.10 and
current Julia, JET, Runic formatting, steady-state benchmark tracking, representative TTFX
tracking, and the full TTFX workload inventory on pushes to `v3`.

## Current state

**Feature/capability parity with upstream HomotopyContinuation.jl v2.22.4 is complete.**

The original rewrite, the July/August testset-by-testset parity audit, and the September late-v2
catch-up are all closed. There is no known v2 subsystem or post-audit v2 capability that still
needs to be ported.

The integrated `v3` tree after the September catch-up tranche was validated as one combined tree,
not only as isolated PR heads. The permanent CI matrix was green for core, certification, JET,
formatting, benchmark tracking, TTFX tracking, and the full TTFX inventory.

This does **not** mean that the branch is release-frozen. Remaining work is release hardening and
v3-specific cleanup, not v2 feature parity.

## Capability summary

At or beyond v2 parity:

- total-degree and polyhedral solving, including projective and overdetermined systems;
- parameter homotopies, fixed parameters, symbolic/custom homotopies, system composition,
  subspace solves and many-target sweeps;
- predictor/corrector tracking, DoubleF64 escalation, singular endgame, valuation and path
  diagnostics;
- `Serial`, `Threaded` and multi-process `DistributedExecutor` tracking;
- monodromy with group actions, linear-subspace parameters, permutations, trace tests and
  certified duplicate admission;
- witness sets, membership, intersections, regeneration and numerical irreducible decomposition,
  including affine/projective, zero-dimensional, parametric and rational cases;
- lossless unresolved NID output: unresolved witness orbits are retained with
  `Irreducibility.UNKNOWN` instead of being discarded;
- Krawczyk certification with Arb fallback and low-memory `ResultIterator` certification in the
  separate `HomotopyContinuationNextCertification` subpackage;
- polynomial, rational and transcendental expression input, including principal-branch `log` and
  the complete interpreter/Taylor/certification lowering used by those expressions;
- multi-homogeneous variable groups;
- SemialgebraicSets integration and solution/parameter file I/O;
- result clustering, multiplicity tracking, and opt-in group-action orbit clustering.

Areas where v3 deliberately exceeds v2 include distributed execution, the monomorphic evaluator
firewall, pure-Julia symbolic construction, lossless unresolved NID results, broader
transcendental certification, and the split lightweight-core/heavy-certification package design.

## September 2026 parity catch-up

The final parity tranche closed the late upstream changes and several gaps found by re-auditing
v2.22.x. The merged implementation includes:

- predictor trust-region fallback at zero crossings and rejection of non-finite/non-positive
  trust radii;
- `LinearSubspace` endpoint dimensions (`dim == 0/n`, `codim == 0/n`), safe in-place redraws,
  and coefficient-type promotion on translation;
- certification from one successful representative per `Result` cluster, including successful
  endpoints numerically labelled singular;
- principal-branch `log` through the expression IR, interpreter/codegen, Taylor recurrences,
  DoubleF64 and interval/Arb certification, with branch-cut rejection;
- exact ordered ambient-coordinate compatibility for witness-set intersections;
- regeneration ordering/tolerance propagation and the invariant that internal monodromy may use
  group actions to discover connectivity but may not quotient witness cardinality;
- preservation and public reporting of unresolved NID witness sets;
- hardened monodromy endpoint admission, including base-problem revalidation of genuinely new
  heuristic endpoints and zero-padded incomplete permutation histories;
- `FixedParameterHomotopy` plus the public `AbstractHomotopy` interface needed by custom
  parameterized homotopies;
- permanent development CI for tests, certification, JET, formatting, benchmarks and TTFX.

See `09_v2_parity_freeze.md` for the PR-by-PR record and the upstream mapping.

## Important semantics fixed during the final tranche

### Regeneration owns witness cardinality

Regeneration and decomposition use monodromy internally for orbit discovery, but the outer
algorithm owns the physical witness-point identity. Internal monodromy therefore runs with
`equivalence_classes = false`; its endpoint tolerances are set by the outer algorithm; and the
post-monodromy `unique_points` pass intentionally does **not** receive `group_actions`.

This prevents symmetry orbits from being mistaken for one witness point.

### Decomposition is lossless under an iteration budget

`Decomposition` keeps a bounded `max_iters` policy (currently 50 by default), rather than copying
v2's very large retry budget. The semantic difference is safe because exhausting the budget does
not drop points: unresolved connected orbits are returned as `WitnessSet`s with
`Irreducibility.UNKNOWN`, and `unresolved_witness_sets` / `unresolved_degree` expose them.

Changing the default iteration budget is therefore a policy/performance decision, not a parity
bug.

### Successful certification candidates are not filtered by numerical singularity labels

`certify(F, result)` takes one successful representative from each `Result` cluster. A numerical
singularity flag is a tracker classification, not proof that Krawczyk certification is
inapplicable. Failed paths remain excluded and raw path duplicates are still not re-certified.

## Performance status

The last documented v2 comparison showed total-degree solving about 1.8–3.6x faster than v2 and
polyhedral solving within noise (roughly 0.95–1.01x). Cold load + construction + first solve was
about 10.6 s versus roughly 45 s for v2's default compiled path. These numbers are historical
measurements, not release guarantees; use the benchmark and TTFX commands above for current data.

The performance architecture remains the central v3 distinction: system-specific compiled code is
hidden behind a monomorphic `SystemEvaluator` / `HomotopyEvaluator` firewall, so a new polynomial
system does not create a new tracker/endgame/solver type chain.

## Remaining work before a release freeze

These are **not v2 feature gaps**.

1. **Public API contract.** Decide the supported downstream surface explicitly. The current
   implementation exports a broad usable API, but `export` alone still conflates convenience
   exports with names that are stable public extension points. The existing proposal is to use a
   public-interface marker (for example SciMLPublic / `Base.ispublic`) and then test that surface.

2. **Release-facing documentation.** The implementation docs are now current after the September
   parity update, but user documentation still needs a release pass for the final public API,
   migration examples, certification split, and v2→v3 differences.

3. **v3-specific executor consistency.** `DistributedExecutor` exceeds v2, but it does not yet
   penetrate every internal witness move, u-homotopy intersection and membership operation.
   This is architecture debt, not a v2 parity blocker.

4. **Direct compile-mode characterization.** A clean v3 `COMPILED_ALL` versus v2 `:all` benchmark
   and fresh-session first-solve comparison would complete the performance record. Benchmark CI
   itself is now implemented.

5. **General cleanup.** Consolidate duplicated path-map loops/progress idioms where worthwhile,
   formalize the wrapper-system interface, and resolve the small items in `TODO.md` without
   changing solver semantics.

## Feature-parity verdict

The project should no longer track work under a "v2 parity" umbrella.

The phase transition is:

```
rewrite
  -> v2 capability parity
  -> late-v2 catch-up
  -> feature-complete freeze
  -> API stabilization / release preparation
```

As of 2026-09-14 the first three stages are complete. New work should be classified as a bug,
release/API hardening, documentation, performance work, or a genuinely new v3 feature.