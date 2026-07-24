# Status

Last updated: 2026-07-24.

**Reproduce:**
- `make benchmark` — steady-state timings
- `make compare` — v3/v2 ratios
- `julia --project=benchmark benchmark/compare/tracking.jl` — end-to-end solve comparison
- `make test` runs the core suite (41 files, parallel via ParallelTestRunner) then the certification subpackage; `make test-cert` runs only the latter

## Summary

Total-degree solving is 1.8–3.6x faster than v2; polyhedral matches within noise (0.95–1.01x).
Cold load + construction + first solve is ~10.6s versus v2's 45s, with no precompile workload.
Endgame is at v2 result parity on v2's default parameters.

At v2 parity: threading, overdetermined systems, parameter homotopies, monodromy (group actions,
linear subspaces, trace test), certification (Krawczyk with Arb fallback, in the separate
`lib/HomotopyContinuationNextCertification` subpackage), and witness sets / NID (`witness_set`,
`trace_test`, `membership`, `regeneration`, `decompose`, `nid`, including projective,
zero-dimensional, and parametric cases).

Remaining gaps: distributed executor, rational expression input, system composition.

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
- [x] Direct polynomial compiler (`polynomial_compiler.jl`): lowers MP input straight to
  tape instructions, skipping SExpr/CSE. Auto-selected at construction for ≤ 2
  variables + parameters and ≤ 8 terms; larger systems keep the symbolic compiler for
  its global CSE. Shares `MonomialCache` with the polyhedral support frontend
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
- [x] **Certification** (`certify`) at full v2 parity, in the separate
  `lib/HomotopyContinuationNextCertification` subpackage (the only Arblib consumer).
  Krawczyk operator with ε-inflation over interval arithmetic (`Interval`/`IComplexF64`, reusing
  the generic tape interpreter), arbitrary-precision Arb fallback (`AcbInterpreter`, in-place ops
  + `setprecision!`, escalates 128→256 bits), `SolutionCertificate`/`ExtendedSolutionCertificate`,
  all accessors and counts, duplicate grouping via interval tree, `save`,
  `show_straight_line_program`, and every input form
  (`Result`/`PathResult`/`Vector`/single/`MonodromyResult`, positional and `target_parameters`).
  Includes the 3264-conics regression. Load with
  `using HomotopyContinuationNext, HomotopyContinuationNextCertification`
- [x] **`DistinctCertifiedSolutions`** streaming accumulator (`add_solution!`,
  `distinct_certified_solutions`/`!`, `certificates`, `solutions`): certifies and deduplicates
  on the fly, thread-safe via a per-task `CertificationCache` (each carries its own cloned
  `SystemEvaluator` so refinement newton never shares interpreter tapes), OhMyThreads `@tasks`/`@local`
- [x] **Witness sets and numerical irreducible decomposition (NID)** for affine systems, in core
  (`src/solving/{witness_set,regeneration,nid}.jl`). Provides `witness_set` (dim/codim, explicit
  subspace, move), `trace_test`, `membership`, `intersect(W, H)` / `intersect(W, f)`; `regeneration`
  (u-regeneration, Duff/Leykin/Rodriguez); `decompose` plus `NumericalIrreducibleDecomposition`
  (`nid` / `numerical_irreducible_decomposition`, `ncomponents`, `degrees`, `witness_sets`,
  hand-rolled degree table with no PrettyTables dep). Covers projective witness sets,
  zero-dimensional varieties, parametric (`target_parameters`) witness sets, and threaded
  membership and intersection. Tests: `test/{witness_set,nid}_test.jl`.

  Key design choices, all deviations from v2:
  - Stays in **ambient coordinates and appends the linear equations** `A x − b` rather than
    slicing into intrinsic coords, so witness init reuses total-degree `solve`, moves reuse
    `ExtrinsicSubspaceHomotopy`, and the u-homotopy reuses `StraightLineHomotopy`.
    `membership` and regeneration's junk-removal test move points with a single concrete
    `IntrinsicSubspaceHomotopy`, well-conditioned in every dim/codim regime.
  - `decompose` follows v2's control flow but tracks orbit connectivity by point identity plus
    union-find, immune to index drift.
  - Parametric input substitutes values into `F` (`_fix_parameters`) and stores the
    parameter-free system, so no parameter plumbing reaches moves/trace/membership/decompose.
  - Threaded intersection clones evaluators instead of `deepcopy`ing trackers (unsafe with
    FunctionWrappers) and pushes endpoints in serial order.
  - `membership` is bit-identical across threading modes: all randomness is drawn from the
    global RNG in the driver, so both modes leave the global RNG in the same state. The query
    subspace direction is genuinely random per query, unlike v2's fixed axis-aligned frame.
  - Adds the `weighted_normal` monodromy sampler (v2 has it only for regen/decompose).

### Not Done

- [ ] **Distributed executor**: extend `AbstractExecutor` with a `Distributed` type for multi-process path tracking (Distributed.jl / MPI)
- [ ] Rational-input witness sets, blocked by the polynomial-only input layer (below)
- [ ] **Solve-level subspace / many-target API**: v2 exposes `solve(F; target_subspace)`,
  `solve(F, S; start_subspace, target_subspace, intrinsic)`, plural `target_subspaces`, the
  analogous many-`target_parameters` sweep, and `iterator_only`. v3 covers the semantics through
  `witness_set(F, L)` / `witness_set(W, L)` / `linear_subspace_homotopy` (portable v2 tests are
  in test/witness_set_test.jl "v2 sliced-solve parity") but has no solve-level kwargs or sweep.
  Unportable until then: solve_test.jl "solve (affine sliced)" and "Many parameters",
  result_test.jl "Target subspaces", systems_test.jl "SlicedSystem"
- [ ] **Non-polynomial (rational) expression input.** v2's ModelKit builds straight-line
  programs, so `u₁/x² + u₂` or `y[1:2] ./ y[3]` work directly; v3's DynamicPolynomials input
  layer is polynomial-only. Blocks two v2 monodromy testsets ("Monodromy rational functions",
  triangulation) and v2's "certify uses approximate inverse of jacobian" (its `approx_inv!`
  path is already covered by the Arb-fallback testset)
- [ ] **System composition** (v2 `CompositionSystem`, `L₂ ∘ f ∘ L₁`). Blocks the v2 symmetroids
  monodromy test (305 solutions; its custom-`distance` kwarg is already supported)
- [ ] Group-action symmetry in `Result` clustering: the `GroupActions` API and group-action-aware
  `UniquePoints` exist and monodromy uses them, but `solve()`'s `Result` dedup still uses plain
  union-find clustering (see `03_v2_improvement_opportunities.md` item 2)
- [ ] Compile-mode benchmark, v2 side: v3 `COMPILED_ALL` vs v2 `:all`, plus fresh-session
  first-solve per v3 default candidate (the v3-only matrix is measured; see `04_compile_modes.md`)
- [ ] Benchmark CI

## Test Suite

41 test files run in parallel via ParallelTestRunner (`make test`, default 10 workers):

| Category | Files | Notes |
|----------|-------|-------|
| Quality gates | `aqua_test.jl`, `jet_test.jl`, `explicit_imports_test.jl` | Static analysis, type inference, import hygiene |
| Type safety | `concrete_structs_test.jl` | All struct fields concretely typed |
| Allocation | `alloc_check_test.jl` | Zero-alloc norms, LA, predictor, Newton, tracker, endgame |
| Primitives | `double_f64_test.jl`, `norms_test.jl`, `linear_algebra_test.jl`, `operations_test.jl` | |
| Model kit | `interpreter_test.jl`, `codegen_test.jl`, `instruction_count_test.jl`, `taylor_test.jl`, `polynomial_input_test.jl` | Tape execution, RGF codegen, instruction-count regression |
| Core | `core_test.jl`, `linear_subspace_test.jl`, `parameter_homotopy_test.jl`, `subspace_homotopy_test.jl` | Also affine charts |
| Tracking | `tracking_test.jl`, `endgame_test.jl`, `newton_test.jl`, `tracker_warmstart_test.jl` | |
| Solving | `solve_test.jl`, `binomial_system_test.jl`, `polyhedral_regression_test.jl`, `overdetermined_test.jl`, `result_clustering_test.jl`, `path_diagnostics_test.jl`, `progress_test.jl` | Executor dispatch, serial/threaded consistency, excess-solution filtering |
| Monodromy | `monodromy_test.jl`, `voronoi_tree_test.jl`, `group_actions_test.jl` | Serial + threaded, permutations, trace, `verify_solution_completeness` |
| Witness/NID | `witness_set_test.jl`, `nid_test.jl` | Affine + projective, zero-dim, parametric, serial + threaded |
| v2 parity | `compare_v2_primitives_test.jl`, `compare_v2_solve_counts_test.jl`, `compare_v2_solve_match_test.jl`, `v2_parity_test.jl` | Primitives, counts, values |
| Misc | `utils_test.jl` | |

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

Raw `SystemEvaluator` kernels. These do not carry to end-to-end solves: through the
`StraightLineHomotopy` they shrink to 1.7–2.0x, and Taylor plus linear algebra dominate
the tracker step.

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

(The solve column is `COMPILED_ALL` vs `COMPILED`, not vs `INTERPRETED`; the ratio holds at
1.08–1.10x across katsura 3/5/7/9.)

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

The root-cause pass cut build/init/solve from 14.44s to ~9.81s by isolating mutually exclusive
compiler branches, using adaptive direct polynomial lowering, and stabilizing constructor types.
Full SnoopCompile/JET/Cthulhu/invalidation report: `05_ttfx_invalidations.md`.

Core load stays at 0.77s because certification is a separate `lib/` subpackage. Adding Arblib to
core raised load to ~1.6s and ~50% more invalidation descendants (one Arblib `show` method alone
accounted for ~2700), which is what motivated the split.

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

- Two solution-dedup mechanisms: `Result` clustering (`_cluster_solutions`, union-find) and
  `UniquePoints`/`VoronoiTree` from the monodromy port. Consolidating onto the VoronoiTree would
  make `solve()` dedup group-action aware and likely speed up large results, but changes
  solution-count semantics (transitive closure vs first-match), so it needs its own tests
  (see `01_decisions.md`, "Two solution-dedup mechanisms exist")
- Two threading coordinators: OhMyThreads executor for `solve()`, Channel job queue for threaded
  monodromy. Justified by the dynamic monodromy workload; revisit only if a third dynamic
  consumer appears

### Infrastructure

- No benchmark CI: regressions go unnoticed
- TTFX gate (construction + first solve < 5s) unmet at ~9.81s. PrecompileTools stays
  deliberately disabled pending a final last-mile pass.
