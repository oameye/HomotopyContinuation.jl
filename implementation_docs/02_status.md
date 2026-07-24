# Status

Last updated: 2026-07-24.

**Reproduce:**
- `make benchmark` — steady-state timings
- `make compare` — v3/v2 ratios
- `julia --project=benchmark benchmark/compare/tracking.jl` — end-to-end solve comparison
- `make test` runs the core suite (40 files, parallel via ParallelTestRunner) then the certification subpackage; `make test-cert` runs only the latter

## Summary

v3 is a credible replacement for v2's core solve pipeline. Total-degree solving is 1.8–3.6x faster
than v2. Polyhedral solving matches v2 within noise (0.95–1.01x). Cold load + construction +
first solve is now ~10.6s without a precompile workload, versus v2's 45s. Endgame is at full v2
result parity using v2's default parameters. Threading,
overdetermined systems, parameter homotopies, and monodromy (full v2 parity including group
actions, linear subspaces, and the trace test) are done. Certification (Krawczyk interval
method with arbitrary-precision Arb fallback) is at v2 parity and lives in a **separate
`lib/HomotopyContinuationNextCertification` subpackage**, which keeps the heavy Arblib
dependency out of core (core load dropped from about 1.6s to about 0.77s). Witness sets and
numerical irreducible decomposition are implemented (`witness_set`, `trace_test`, `membership`,
`regeneration`, `decompose`, `nid`), validated at v2 parity, including projective witness sets
and membership, zero-dimensional varieties, parametric (`target_parameters`) witness sets, and
threaded membership and intersection. The main remaining gaps are a distributed executor and
two input-layer features (rational expression input, system composition; see Not Done).

## Feature Checklist

### Done

- [x] `solve(F)` with CommonSolve.jl `init`/`solve!`
- [x] Total-degree and polyhedral start systems
- [x] Parameter homotopy (CoefficientHomotopy with linear parameter interpolation)
- [x] `System{P,V,M,S}` type (compile mode `M` and square/overdetermined shape `S`
  live in the type domain; caches interpreters for all eval modes and stores original MP polys)
- [x] `SystemEvaluator` / `HomotopyEvaluator` type firewall (FunctionWrapper, 10 wrappers each,
  including a DF64-output evaluate for extended-precision residual combining)
- [x] StraightLineHomotopy, CoefficientHomotopy, ToricHomotopy
- [x] Cauchy product Taylor convolution for parametric homotopies
- [x] Two-stage toric reparameterization (weight renormalization when max_weight ≥ 10)
- [x] Predictor-corrector tracker (Padé 2,1 + adaptive step + s-plane Hermite for winding > 1)
- [x] Multi-round iterative refinement in predictor (weighted-norm orders 2–3, inf-norm order 1)
- [x] Newton corrector (α-theory) with DoubleF64 extended precision
- [x] Step control: ω extrapolation, convergence-rate rejection, near-target scaling
- [x] Endgame tracker — Puiseux valuation, winding number estimation, singular Cauchy endgame
  (geometric stepping λ=0.25), at-infinity/at-zero detection, cubic Hermite endpoint prediction,
  jump-to-zero gating for m=1 paths
- [x] Solution deduplication (union-find clustering, sort-and-window candidate sweep
  scaling to thousands of paths), multiplicity tracking
- [x] Binomial system solver (HNF with BigInt/BigFloat overflow fallback and result
  validation, v2 parity), weighted norms, custom LU with Skeel scaling
- [x] Tape interpreter for eval, jacobian, Taylor 1–3, DF64
- [x] CSE optimizer, Moshi ADTs for SExpr/ExecInstruction
- [x] RGF compiled eval+jac backend (`CompileMode.COMPILED`, 3–6x kernel speedup)
- [x] RGF compiled Taylor backend (`CompileMode.COMPILED_ALL`, 1.3–1.7x Taylor kernel speedup, 1.2–1.4x end-to-end vs `INTERPRETED`)
- [x] Automatic coefficient normalization (scales polynomials with O(10^8+) coefficients to O(1))
- [x] AllocCheck zero-allocation enforcement on all hot paths
- [x] Integration tests from v2 with exact result parity
- [x] Threading via OhMyThreads.jl — `Serial`/`Threaded` executor types, builder/worker-state
  pattern for thread-safe evaluator cloning, `@tasks`/`@local` work distribution
- [x] Overdetermined systems: `RandomizedSystem` square-up (identity block plus random fold of
  the lowest-degree equations, permutation keeps degrees exact), wired into total-degree
  (squared-up evaluator) and polyhedral (merged support/coefficients), excess-solution
  filtering post-pass (Newton on the original system for nonsingular endpoints, residual
  comparison for singular ones), `PATH_EXCESS_SOLUTION` result code and `nexcess_solutions`
- [x] Standalone `newton(F, x0)` at v2 parity (`extended_precision` defaults to `false`; `norm` arg
  omitted since v3 is inf-norm only; underdetermined m < n supported via column-pivoted QR)
- [x] Invalid-start classification: singular start Jacobian reported as
  `TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN` (v2 parity)
- [x] Tracker option presets `DEFAULT/FAST/CONSERVATIVE_TRACKER_OPTIONS` (v2's
  TrackerParameters presets)
- [x] Path diagnostics at full v2 accessor parity on `PathResult` (incl. `path_number`, `start_solution`,
  `valuation`, `multiplicity`, `cond`, `is_failed`/`is_finite`) and `Result` (`seed`, `ntracked`,
  `failed`, `at_infinity`, `nonsingular`, `singular`, `nfailed`, `statistics`)
- [x] Progress bars via ProgressMeter.jl (`show_progress` kwarg on `solve`/`init`, threaded, with v2's
  live `showvalues` counts and `delay=0.3` suppression)
- [x] `ParameterHomotopy` (linear parameter interpolation, retargetable via
  `start_parameters!`/`target_parameters!`) and `solve(F, starts; start_parameters, target_parameters)`
- [x] Tracker warm start (`track!` reusing ω/μ from a previous path; `ω`/`μ` accessors on `PathResult`)
- [x] `GroupActions`/`SymmetricGroup` (orbit generation, composed actions)
- [x] `VoronoiTree` and `UniquePoints` (group-action-aware nearest-point dedup),
  `multiplicities`, `unique_points`
- [x] `LinearSubspace`: intrinsic/extrinsic descriptions, `rand_subspace`, `translate`,
  `geodesic`/`geodesic_distance` on the Grassmannian, `coord_change`
- [x] Subspace homotopies (`IntrinsicSubspaceHomotopy`, `ExtrinsicSubspaceHomotopy`,
  `linear_subspace_homotopy`) and affine charts (`on_affine_chart`, `AffineChartSystem`/`Homotopy`)
- [x] `monodromy_solve` at full v2 parity: `find_start_pair`, serial and threaded
  (Channel job queue) execution, `MonodromyOptions` (~27 explicit kwargs, no splatting),
  `reuse_loops` (`:all`/`:random`/`:none`), heuristic stop, `target_solutions_count`,
  equivalence classes via group actions, `LinearSubspace` parameters, permutations, trace
- [x] `verify_solution_completeness` (trace test with augmented system, auxiliary monodromy,
  singular-value trace check)
- [x] **Certification** (`certify`) at full v2 parity, in the **separate `lib/HomotopyContinuationNextCertification`
  subpackage** (keeps Arblib out of core, so core load is ~0.77s; certification is the only Arblib
  consumer). Krawczyk operator with ε-inflation over interval arithmetic (`Interval`/`IComplexF64`,
  generic tape interpreter reused), arbitrary-precision Arb fallback (`AcbInterpreter` with in-place
  ops + `setprecision!`, escalates 128→256 bits), `SolutionCertificate`/`ExtendedSolutionCertificate`,
  all accessors and counts (`ncertified`, `nreal_certified`, `ndistinct_*`,
  `is_real`/`is_complex`/`is_positive`), duplicate grouping via interval tree, `save`,
  `show_straight_line_program`, and every input form
  (`Result`/`PathResult`/`Vector`/single/`MonodromyResult`, positional and `target_parameters`).
  Includes the 3264-conics regression (steiner system, 3264 distinct real certified).
  Load with `using HomotopyContinuationNext, HomotopyContinuationNextCertification`
- [x] **`DistinctCertifiedSolutions`** streaming accumulator (`add_solution!`,
  `distinct_certified_solutions`/`!`, `certificates`, `solutions`): certifies and deduplicates
  on the fly, thread-safe via a per-task `CertificationCache` (each carries its own cloned
  `SystemEvaluator` so refinement newton never shares interpreter tapes), OhMyThreads `@tasks`/`@local`
- [x] **Witness sets and numerical irreducible decomposition (NID)** for affine systems, in core
  (`src/solving/{witness_set,regeneration,nid}.jl`). Provides `witness_set` (dim/codim, explicit
  subspace, move), `trace_test`, `membership`, `intersect(W, H)` / `intersect(W, f)`; `regeneration`
  (u-regeneration, Duff/Leykin/Rodriguez); `decompose` plus `NumericalIrreducibleDecomposition`
  (`nid` / `numerical_irreducible_decomposition`, `ncomponents`, `degrees`, `witness_sets`,
  hand-rolled degree table with no PrettyTables dep).
  Redesign: it stays in **ambient coordinates and appends the linear equations** `A x − b` (v2's
  `SlicedSystem`) instead of slicing into intrinsic coords, so witness init reuses the ordinary
  total-degree `solve`, moves reuse `ExtrinsicSubspaceHomotopy`, and the u-homotopy reuses
  `StraightLineHomotopy`; fill-up/decompose reuse `MonodromySolver` (intrinsic under the hood).
  `membership` (and regeneration's junk-removal containment test) moves witness points with a single
  concrete `IntrinsicSubspaceHomotopy`, well-conditioned in every dim/codim regime. `decompose`
  matches v2's control flow (one accumulating monodromy loop per iteration, `iter >= 5` singleton
  gate) with orbit connectivity tracked by point identity + union-find (a data-representation change,
  immune to index drift).
  Added the `weighted_normal` monodromy sampler (v2 had it only for regen/decompose; it preserves
  the zero-structure of the flag subspaces). Validated at v2 parity: two circles give 2 components
  of degree (2,2); the multi-dimensional example gives 11 components (dim-2 deg-2, dim-1 deg-4 twice,
  dim-0 deg-1 eight times), and the degree table matches v2 output. Tests:
  `test/{witness_set,nid}_test.jl`.
  Completed 2026-07-24 (the former Not-Done sub-items):
  **projective witness sets end to end**: `witness_set`/`trace_test` were already
  chart-consistent; `membership` now supports projective witness sets (the query subspace is
  linear through the ray of the query point via `_random_orthonormal_through`, the
  `IntrinsicSubspaceHomotopy` runs the system on a shared random affine chart
  (`AffineChartSystem`), and all point comparisons happen between chart representatives, so any
  projective scaling of a query point works).
  **Zero-dimensional varieties**: `witness_set` slices dim-0 varieties (affine and projective)
  with the codim-0 full space (`_full_subspace`, `A` is `0 × n`) instead of hitting
  `rand_subspace(codim = 0)`, so the sliced system is `F` itself plus (projectively) a chart
  row; `trace_test`, `membership`, and `decompose` (each point its own component) all work on
  the result, and a negative computed dimension throws a clear "V(F) is empty" error.
  **Parametric witness sets**: `witness_set(F; target_parameters)` and
  `witness_set(F, L; target_parameters)` substitute the parameter values into `F`
  (`_fix_parameters`, the analog of v2's `fix_parameters`, compile mode preserved) and store
  the parameter-free system, so moves/trace/membership/decompose need no parameter plumbing;
  missing, spurious, and wrong-length parameter values throw `ArgumentError`s, and
  `regeneration`/`nid` reject parametric systems loudly.
  **Threading**: the u-homotopy d-th-root tracking in regeneration/`intersect` is threaded
  (one task per (point, root) pair via `@tasks`/`@local`, per-task `EndgameTracker` built from
  `_clone_system_evaluator`d endpoint systems, same γ across tasks, endpoints collected in
  serial order), and `membership` queries points in parallel (per-task `MembershipState`
  bundling a cloned evaluator, homotopy, tracker, and buffers; `threading` kwarg at v2 parity).
  Post-review parity alignment (same day): `membership` defaults match v2
  (`EndgameOptions(max_endgame_steps = 100, max_endgame_extended_steps = 100, sing_cond = 1e12)`,
  `show_progress = true`; the capped endgame also speeds up regeneration junk removal
  noticeably), `witness_set` defaults to `show_progress = true` like v2's solve-forwarding,
  polynomial input forms (`witness_set(f)` / `witness_set([f, g])`, with or without an
  explicit subspace) mirror v2's Expression forms, and the `WitnessSet` constructor rejects
  parametric systems with a clear error instead of failing deep inside membership/moves.
  Unlike v2, the threaded intersection does not `deepcopy` trackers (unsafe with
  FunctionWrappers in v3; evaluators are cloned instead) and pushes endpoints in
  deterministic serial order. `membership` is fully deterministic across threading modes:
  all randomness (chart, gauge point, gamma, one pre-drawn matrix per query) is drawn from
  the global RNG in the driver, so `threading = true` and `false` give bit-identical
  results and leave the global RNG in the same state; per-query scratch buffers live in
  `MembershipState`, and only the `LinearSubspace` data (kept by reference by the homotopy)
  is allocated per query. Also unlike v2, the query subspace direction is genuinely random
  per query (v2's `MembershipCache` builds its `A` from `svd(zeros(...)).Vt`, a fixed
  axis-aligned frame).

### Not Done

- [ ] **Distributed executor**: extend `AbstractExecutor` with a `Distributed` type for multi-process path tracking (Distributed.jl / MPI)
- [ ] Direct polynomial compiler (`polynomial_compiler.jl` exists, deferred)
- [ ] Rational-input witness sets stay blocked by the polynomial-only input layer (below);
  everything else under witness sets / NID is done (see the Done bullet above)
- [ ] **Solve-level subspace / many-target API** (discovered 2026-07-24 while porting v2's
  sliced-solve tests): v2 exposes `solve(F; target_subspace)`, `solve(F, S; start_subspace,
  target_subspace, intrinsic)`, plural `target_subspaces` (a sweep over many subspaces
  returning per-target results), the analogous many-`target_parameters` sweep, and
  `iterator_only`. v3 covers the underlying semantics through `witness_set(F, L)` /
  `witness_set(W, L)` / `linear_subspace_homotopy` (the portable v2 tests are ported as
  test/witness_set_test.jl "v2 sliced-solve parity"), but has no solve-level kwargs and no
  many-target sweep. v2 tests that stay unportable until then: solve_test.jl "solve (affine
  sliced)" (the `solver_startsolutions`/`slice` internals), solve_test.jl "Many parameters",
  result_test.jl "Target subspaces" (`iterator_only`), systems_test.jl "SlicedSystem"
  (parametric slicing; v3 slices after parameter substitution)
- [ ] **Non-polynomial (rational) expression input.** v2's ModelKit builds straight-line
  programs, so systems like `u₁/x² + u₂` or triangulation objectives with `y[1:2] ./ y[3]`
  work directly; v3's DynamicPolynomials input layer is polynomial-only. Blocks porting two
  v2 monodromy testsets ("Monodromy rational functions", the triangulation system) and v2's
  "certify uses approximate inverse of jacobian" certification testset (needs `log`/rational
  input plus Combinatorics; its `approx_inv!` code path is already covered by the Arb-fallback
  testset in `test/certification_test.jl`)
- [ ] **System composition** (v2 `CompositionSystem`, `L₂ ∘ f ∘ L₁`): no v3 equivalent.
  Blocks porting the v2 symmetroids monodromy test (305 solutions, custom `distance`;
  the custom-distance kwarg itself is already supported)
- [ ] Group-action symmetry in `Result` clustering (the `GroupActions` API and
  group-action-aware `UniquePoints` now exist and monodromy uses them, but
  `solve()`'s `Result` dedup still uses the plain union-find clustering; see
  open items and `03_v2_improvement_opportunities.md` item 2)
- [ ] Compile-mode benchmark, v2 side: v3 `COMPILED_ALL` vs v2 `:all`, plus
  fresh-session first-solve per v3 default candidate (see `04_compile_modes.md`
  TODO; the v3-only matrix is measured, `benchmark/compile_modes_e2e.jl`)
- [ ] Benchmark CI

## Test Suite

40 test files run in parallel via ParallelTestRunner (`make test`, default 10 workers):

| Category | Files | Purpose |
|----------|-------|---------|
| Quality gates | `aqua_test.jl`, `jet_test.jl`, `explicit_imports_test.jl` | Static analysis, type inference, import hygiene |
| Type safety | `concrete_structs_test.jl` | Verify all struct fields are concretely typed |
| Allocation | `alloc_check_test.jl` | Zero-allocation hot paths (norms, LA, predictor, Newton, tracker, endgame) |
| Primitives | `double_f64_test.jl`, `norms_test.jl`, `linear_algebra_test.jl`, `operations_test.jl` | DoubleF64, WeightedNorm, MatrixWorkspace, op_* functions |
| Model kit | `interpreter_test.jl`, `codegen_test.jl`, `instruction_count_test.jl`, `taylor_test.jl`, `polynomial_input_test.jl` | Tape execution, RGF codegen, instruction regression, Taylor series |
| Core | `core_test.jl`, `linear_subspace_test.jl`, `parameter_homotopy_test.jl`, `subspace_homotopy_test.jl` | System/Homotopy construction and evaluation, subspaces, parameter/subspace homotopies, affine charts |
| Tracking | `tracking_test.jl`, `endgame_test.jl`, `newton_test.jl`, `tracker_warmstart_test.jl` | Newton, predictor, path tracking, valuation, winding, warm start |
| Solving | `solve_test.jl`, `binomial_system_test.jl`, `polyhedral_regression_test.jl`, `overdetermined_test.jl`, `result_clustering_test.jl`, `path_diagnostics_test.jl`, `progress_test.jl` | End-to-end solving, executor dispatch, serial/threaded consistency, binomial HNF, polyhedral regression, square-up + excess-solution filtering, clustering, diagnostics |
| Monodromy | `monodromy_test.jl`, `voronoi_tree_test.jl`, `group_actions_test.jl` | monodromy_solve (serial + threaded), permutations, trace test, verify_solution_completeness, VoronoiTree, group actions |
| Witness/NID | `witness_set_test.jl`, `nid_test.jl` | witness_set (init, move, projective, zero-dim, parametric), trace test, membership (affine + projective, serial + threaded), regeneration (u-regeneration), intersect (serial + threaded), decompose, nid multi-dimensional v2 parity |
| v2 parity | `compare_v2_primitives_test.jl`, `compare_v2_solve_counts_test.jl`, `compare_v2_solve_match_test.jl`, `v2_parity_test.jl` | Primitive matching, solution counts, solution values, overall parity |
| Misc | `utils_test.jl` | SegmentStepper, stable_sort, etc. |

JET test filters known false positives: MP.variables dispatch (construction-time), Moshi `@match`/`@derive` generated code.

## Performance

Measured 2026-04-03, Julia 1.12.5, single-threaded.

### End-to-end solve vs v2

`CompileMode.COMPILED`, fixed seed `0x4567`, endgame enabled.

#### Total-degree (katsura, chain)

| System | v3/v2 ratio | v3 steps/path | v2 steps/path |
|--------|------------:|--------------:|--------------:|
| katsura-3 | **3.07x** | 36.2 (290 acc, 0 rej) | 87.0 (696 acc, 0 rej) |
| katsura-4 | **2.06x** | 47.3 (757 acc, 0 rej) | 78.8 (1260 acc, 0 rej) |
| katsura-5 | **2.01x** | 56.4 (1806 acc, 0 rej) | 98.5 (3137 acc, 15 rej) |
| chain-3 | **3.62x** | 23.0 (184 acc, 0 rej) | 46.2 (370 acc, 0 rej) |
| chain-4 | **2.35x** | 34.4 (550 acc, 0 rej) | 64.2 (1028 acc, 0 rej) |
| chain-5 | **1.82x** | 40.2 (1288 acc, 0 rej) | 65.0 (2081 acc, 0 rej) |

#### Polyhedral (cyclic, random sparse)

| System | v3/v2 ratio | v3 steps/path | v2 steps/path |
|--------|------------:|--------------:|--------------:|
| cyclic-4 | 0.96x | 77.1 (1188 acc, 46 rej) | 74.3 (1158 acc, 31 rej) |
| cyclic-5 | 1.00x | 50.8 (3556 acc, 0 rej) | 50.8 (3553 acc, 0 rej) |
| sparse-3x3 | 0.95x | 45.0 (1924 acc, 9 rej) | 45.0 (1924 acc, 9 rej) |
| sparse-4x4 | 1.01x | 75.1 (12575 acc, 47 rej) | 75.0 (12551 acc, 47 rej) |
| sparse-5x5 | 0.98x | 83.3 (34080 acc, 85 rej) | 83.4 (34121 acc, 90 rej) |

#### Compiled vs interpreted kernels

Raw `SystemEvaluator` kernels. These speedups do not carry to end-to-end
solves (see the next table): through the `StraightLineHomotopy` they shrink to
1.7–2.0x, and Taylor plus linear algebra dominate the tracker step.

| System | Eval speedup | Jac speedup |
|--------|-------------:|------------:|
| katsura-3 | 3.35x | 3.82x |
| katsura-5 | 3.65x | 5.04x |
| katsura-7 | 3.29x | 6.74x |

#### End-to-end solve per compile mode

Measured 2026-07-17, serial executor, speedup vs `INTERPRETED`
(`benchmark/compile_modes_e2e.jl`; analysis in `04_compile_modes.md`):

| System | INTERPRETED | COMPILED | COMPILED_ALL |
|--------|------------:|---------:|-------------:|
| katsura-3 | 0.91ms | 1.10x | 1.18x |
| katsura-5 | 8.79ms | 1.15x | 1.25x |
| katsura-7 | 64.98ms | 1.23x | 1.33x |
| katsura-9 | 482.94ms | 1.27x | 1.41x |

#### COMPILED_ALL (compiled Taylor) vs INTERPRETED

Scalar-parameter Taylor kernel (parameter-free systems, speedup = interp/all):

| System | Taylor 1 speedup | Taylor 2 speedup | Taylor 3 speedup | Solve speedup vs COMPILED | Build overhead |
|--------|------------------:|------------------:|------------------:|--------------:|---------------:|
| katsura-3 | 1.66x | 1.72x | 1.62x | 1.09x | 1.62x |
| katsura-5 | 1.73x | 1.59x | 1.61x | 1.09x | 1.39x |
| katsura-7 | 1.74x | 1.65x | 1.76x | 1.08x | 1.29x |

(The solve column is `COMPILED_ALL` vs `COMPILED`, not vs `INTERPRETED`;
katsura-7 value measured 2026-07-17. Re-measured 2026-07-17 at katsura 3/5/7/9
the ratio is 1.08–1.10x across all sizes.)

TaylorVector-parameter Taylor kernel (production path — CoefficientHomotopy/ToricHomotopy):

| System | Taylor 1 speedup | Taylor 2 speedup | Taylor 3 speedup |
|--------|------------------:|------------------:|------------------:|
| katsura-3 | 1.48x | 1.72x | 1.48x |
| katsura-5 | 1.51x | 1.65x | 1.25x |
| katsura-7 | 1.30x | 1.63x | 1.20x |

#### TTFX (fresh session)

Final v3 root-cause pass measured 2026-07-18; v2 numbers from 2026-04-03.

| Metric | Time |
|--------|------|
| v3 package load | 0.77s |
| v3 construction + init + first solve! | 9.80--9.83s |
| **v3 total (load + construction + solve)** | **10.58--10.61s** |
| v2 package load | 1.30s |
| v2 first solve() [:mixed] | 43.78s |
| **v2 total [:mixed]** | **45.08s** |
| v2 first solve() [:none] | 10.79s |
| v2 second solve() (different system) | 5.72s |

v3 is over 4x faster than v2[:mixed] on the cold workload. The root-cause pass reduced
build/init/solve from 14.44s to about 9.81s (32%) by isolating mutually exclusive compiler
branches, using adaptive direct polynomial lowering, and stabilizing constructor types.
See `05_ttfx_invalidations.md` for the full SnoopCompile, JET, Cthulhu, LLVM, and
invalidation report.

The 0.77s core load holds because certification (the sole consumer of the heavy Arblib
dependency) is a separate `lib/` subpackage. Adding Arblib to core raised load to about 1.6s
and roughly 50% more invalidation descendants (one Arblib `show` method alone accounted for
about 2700), which is what motivated the split. Loading
`HomotopyContinuationNextCertification` pays that Arblib cost only when certification is used.

### Endgame result parity

| System | v2 result | v3 result |
|--------|-----------|-----------|
| (x-10)^2 | nresults=1, nsingular=1 | same |
| at-infinity | 2 success + 2 at_infinity | same |
| winding family d=2,4,6 | d+1 success each | same |
| Hyperbolic 6,6 | nresults=2, nsingular=2 | same |
| singular multiplicity 3 | nresults=2, nsingular=1, nnonsingular=1 | same |

## Open Items

### Architecture debt

4. Two solution-dedup mechanisms: `Result` clustering (`_cluster_solutions`, union-find)
   and `UniquePoints`/`VoronoiTree` from the monodromy port. Consolidating `Result`
   clustering onto the VoronoiTree would remove the duplication, make `solve()` dedup
   group-action aware, and likely speed up large results, but changes solution-count
   semantics (transitive closure vs first-match), so it needs its own tests
   (see `01_decisions.md`, "Two solution-dedup mechanisms exist")
5. Two threading coordinators: OhMyThreads executor for `solve()`, Channel job queue for
   threaded monodromy. Justified by the dynamic monodromy workload; revisit only if a
   third dynamic consumer appears

### Infrastructure

8. No benchmark CI: regressions go unnoticed
9. TTFX gate (construction + first solve < 5s) remains unmet at about 9.81s after the
   root-cause pass. PrecompileTools remains deliberately disabled pending a final last-mile pass.
