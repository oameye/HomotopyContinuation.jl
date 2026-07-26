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

Remaining gaps: distributed executor, non-polynomial expression input, system composition.

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
  `linear_subspace_homotopy`) and affine charts (`on_affine_chart`, `AffineChartSystem`/`Homotopy`),
  including the appended-row Taylor coefficient `c·x_K` for the chart row (see
  `01_decisions.md`); `AffineChartHomotopy` keeps `0` because a homotopy is only ever the
  outermost wrapper, where the predictor has already zeroed that row
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
- [x] **Solve-level subspace API and many-target sweeps**
  (`src/solving/{slice,subspace_solve,sweep,result_iterator}.jl`, `src/core/sliced_system.jl`).
  Everything is positional and typed rather than v2's kwargs, so each route is its own method and
  a session compiles only the routes it calls.
  - `slice(F, L)` → the polynomial system `[F; A x − b]` in ambient coordinates, preserving `F`'s
    variables, parameters and compile mode; `chart = c` appends `c·x − 1`.
  - `solve(F, L, alg, exec)` / `init`: `V(F) ∩ L` through the ordinary total-degree or polyhedral
    route, so `Result`, clustering and the excess checker apply unchanged. A homogeneous `F` with
    a linear `L` gets a seed-reproducible chart row; parametric `F` takes `target_parameters`,
    substituted before slicing. Solutions are ambient (v2's too, though it reaches them through
    an intrinsic sliced system).
  - `SlicedSystem` appends the linear rows by wrapping `F`'s **evaluator**, so the square
    total-degree route (and `_witness_init`, hence witness sets, NID and regeneration) never
    re-runs CSE or rebuilds tapes: `init` 0.02 ms versus 9.64 ms on a dense degree-4 system in 5
    variables, with identical step counts and `solve!` 2.39 ms versus 2.57 ms. Under- or
    overdetermined slices still rebuild the polynomial system, since the square-up machinery
    needs one.
  - `solve(F, starts, L_start, L_target, exec; intrinsic)`: intrinsic (`F(A(t)v + a(t))`) when
    `dim(L_start) <= codim(L_start)`, else extrinsic (`[F(x); A(t)x − a(t)]`), forceable either
    way. Start points and solutions are ambient in both regimes; the intrinsic regime converts in
    at `t = 1` and back out through `_to_ambient` at the `t` each path reported. Consequences:
    clustering compares ambient points (v2 clusters intrinsic ones), and per-path diagnostics
    (accuracy, residual, condition number, valuation) stay in tracking coordinates, as in v2.
  - `solve(F, starts, targets, exec; start_parameters, ...)` and
    `solve(F, starts, L_start, targets, exec; ...)`: one homotopy built and retargeted per target
    (`target_parameters!` through the concrete handle in the worker state), `transform_result`,
    `transform_parameters` and `flatten` matching v2's four return shapes. Threading runs over the
    (target, path) product, each task owning its worker state and retargeting it when it crosses a
    target boundary, so the speedup no longer caps at the number of targets: measured 4.31x versus
    1.36x on 12 threads for a one-target sweep of 125 paths, and unchanged at 24+ targets. Chunks
    are contiguous in target-major order, so this costs at most `n_targets + ntasks` retargets. The
    v2 kwarg spelling (`target_parameters = [p1, ...]`) is deliberately not accepted: it would make
    an existing route's return type value-dependent.
  - `result_iterator(...)` → `ResultIterator`, the typed replacement for v2's `iterator_only`
    kwarg: lazy per-path tracking for the total-degree, polyhedral, sliced, parameter and
    subspace routes (serial), `bitmask` / `bitmask_filter`, and `Result(ri)` for clustering and
    excess reclassification. A `ResultIterator` may be passed as the start solutions of another
    solve.
  - Tests: `test/{sliced_solve,subspace_solve,many_targets,result_iterator}_test.jl`
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
    slicing into intrinsic coords, so witness init reuses total-degree `solve` (through
    `_init_sliced_total_degree`, i.e. the `SlicedSystem` wrapper for a square slice), moves reuse
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
- [ ] **Non-polynomial (rational) expression input.** The expression frontend must cover both
  v2's rational straight-line programs (`u₁/x² + u₂`, `y[1:2] ./ y[3]`) and retained unary
  operations such as `sqrt(γ) * x₁ + x₂^2`, `sin(γ) * x₁`, and `cos(x₁) - x₂`.
  `OP_SQRT`, `OP_SIN`, and `OP_COS` remain in `ExecInstruction` even though the current
  DynamicPolynomials frontends do not emit them. This work blocks two v2 monodromy testsets
  ("Monodromy rational functions", triangulation) and v2's "certify uses approximate inverse
  of jacobian" (its `approx_inv!` path is already covered by the Arb-fallback testset).

  Implementation and validation checklist:
  - add expression-IR/frontend nodes for division, integer powers, `sqrt`, `sin`, and `cos`,
    with variable/parameter discovery and constant folding;
  - lower them to the existing `OP_DIV`/`OP_INV`/`OP_POW_INT` and
    `OP_SQRT`/`OP_SIN`/`OP_COS` instructions in every compile mode;
  - implement analytic Jacobian rules and verify Taylor orders 1--3 for scalar and
    Taylor-valued parameters;
  - complete the extended-precision surface: `sqrt` already works for `ComplexDF64`;
    `sin` and `cos` still need `DoubleF64`/`ComplexDF64` implementations;
  - add RGF codegen and interval/Acb certification coverage for every new expression node;
  - add end-to-end regression systems, including `sqrt(γ) * x₁ + x₂^2`, and validate
    interpreted/compiled parity, Jacobians, DF64 residuals, Taylor coefficients against a
    Cauchy-integral oracle, parameter tracking, monodromy, and certification;
  - keep low-level interpreter tests for all retained unary variants so an operation cannot
    disappear from an execution backend before the frontend begins emitting it.
- [ ] **System composition** (v2 `CompositionSystem`, `L₂ ∘ f ∘ L₁`). Blocks the v2 symmetroids
  monodromy test (305 solutions; its custom-`distance` kwarg is already supported)
- [ ] Group-action symmetry in `Result` clustering: the `GroupActions` API and group-action-aware
  `UniquePoints` exist and monodromy uses them, but `solve()`'s `Result` dedup still uses plain
  union-find clustering (see `03_v2_improvement_opportunities.md` item 2)
- [ ] Compile-mode benchmark, v2 side: v3 `COMPILED_ALL` vs v2 `:all`, plus fresh-session
  first-solve per v3 default candidate (the v3-only matrix is measured; see `04_compile_modes.md`)
- [ ] Benchmark CI

## Test Suite

51 test files run in parallel via ParallelTestRunner (`make test`, default 10 workers),
about 5070 passing assertions plus one `@test_skip` (`nid_test.jl`). The total is not
exactly reproducible: 25 assertion loops iterate over *discovered* solutions
(`for s in solutions(res)`), so a run that finds a different number of endpoints
reports a different number of assertions. Observed 5039 and 5086 on one commit:

| Category | Files | Notes |
|----------|-------|-------|
| Quality gates | `aqua_test.jl`, `jet_test.jl`, `explicit_imports_test.jl` | Static analysis, type inference, import hygiene |
| Type safety | `concrete_structs_test.jl` | All struct fields concretely typed |
| Allocation | `alloc_check_test.jl` | Zero-alloc norms, LA, predictor, Newton, tracker, endgame, and the three system wrappers whose appended rows run on the predictor's hot path (`RandomizedSystem`, `AffineChartSystem`, `SlicedSystem`) |
| Primitives | `double_f64_test.jl`, `norms_test.jl`, `linear_algebra_test.jl`, `operations_test.jl` | |
| Model kit | `interpreter_test.jl`, `codegen_test.jl`, `instruction_count_test.jl`, `taylor_test.jl`, `polynomial_input_test.jl` | Tape execution, RGF codegen, instruction-count regression; every `taylor_op_*` against a Cauchy-integral oracle |
| Evaluation sweep | `system_sweep_test.jl` (+ `test_systems.jl`) | 9 real systems (cyclic5/7, bacillus, cyclo, moments3, six_revolute, steiner, four_bar, tritangents) × 3 compile modes: eval, jacobian, DF64, Taylor 1–3 with constant and Taylor-valued parameters, plus the straight-line homotopy, all against exact symbolic ground truth |
| Core | `core_test.jl`, `linear_subspace_test.jl`, `parameter_homotopy_test.jl`, `subspace_homotopy_test.jl`, `affine_chart_test.jl` | `affine_chart_test.jl` checks the chart row's Taylor coefficient `c·x_K` for a nonzero and a zeroed top row |
| Tracking | `tracking_test.jl`, `endgame_test.jl`, `newton_test.jl`, `tracker_warmstart_test.jl`, `valuation_test.jl`, `tracker_regression_test.jl` | `valuation_test.jl` checks asymptotic valuations (finite, diverging, fractional); `tracker_regression_test.jl` covers the four-bar and Steiner near-singular paths |
| Solving | `solve_test.jl`, `binomial_system_test.jl`, `polyhedral_regression_test.jl`, `overdetermined_test.jl`, `result_clustering_test.jl`, `path_diagnostics_test.jl`, `progress_test.jl` | Executor dispatch, serial/threaded consistency, excess-solution filtering |
| Subspaces / sweeps | `sliced_solve_test.jl`, `subspace_solve_test.jl`, `many_targets_test.jl`, `result_iterator_test.jl` | `slice`, subspace→subspace, many-target sweeps, lazy iteration. `many_targets_test.jl` asserts threaded == serial for target counts on both sides of `ntasks`, since `(target, path)` threading splits one target's paths across tasks; run it under `-t N` (the parallel runner gives each worker one thread, where `Threaded()` degenerates to one task) |
| Monodromy | `monodromy_test.jl`, `voronoi_tree_test.jl`, `group_actions_test.jl` | Serial + threaded, permutations, trace, `verify_solution_completeness` |
| Witness/NID | `witness_set_test.jl`, `nid_test.jl` | Affine + projective, zero-dim, parametric, serial + threaded |
| v2 parity | `compare_v2_primitives_test.jl`, `compare_v2_solve_counts_test.jl`, `compare_v2_solve_match_test.jl`, `v2_parity_test.jl`, `monodromy_v2_parity_test.jl` | Primitives, counts, values |
| Misc | `utils_test.jl` | |

The certification subpackage adds 486 assertions (`make test-cert`): `interval_arithmetic_test.jl`,
`acb_interpreter_test.jl` (the Arb tape interpreter against Float64 ground truth, ball containment
and precision refinement), `certification_test.jl`, `export_surface_test.jl`, `quality_test.jl`.

JET test filters known false positives: MP.variables dispatch (construction-time), Moshi `@match`/`@derive` generated code.

The default `JOBS=10` can drive a worker to SIGTERM under memory pressure (observed on
`endgame_test`, which passes in 15s on its own and reports
`Malt.TerminatedWorkerException` when killed). `make test JOBS=6` is the reliable
setting. A `TerminatedWorkerException` is a resource symptom, not a test failure;
re-run the file alone before believing it.

### Not covered from v2's suite

Every v2 test file has a v3 counterpart except the ones whose feature is absent (see Not Done):
`semialgebraic_sets_test.jl` (no SemialgebraicSets integration), `model_kit/symbolic_test.jl` (v3's
input layer is DynamicPolynomials, not a symbolic `Expression` type), and the composition /
rational-input / `paths_to_track` / `mixed_volume` / `stop_early_cb` / start-target-`solve`
testsets, whose APIs v3 does not have.

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

Latest v3 pass measured 2026-07-25; v2 numbers from 2026-04-03.

The v3 rows predate the tape-executor precompilation of 2026-07-26, which cut the
common path by 1.28s in a paired comparison (`01_decisions.md`, "Tape executors are
precompiled by signature"). They are not restated here because the machine was
contended when the change landed and no clean absolute sweep was taken; re-run
`make ttfx` on a quiet machine before trusting the numbers below.

| Metric | Time |
|--------|------|
| v3 package load | 0.77s |
| v3 construction + init + first solve! | 8.44--8.77s |
| **v3 total (load + construction + solve)** | **9.21--9.54s** |
| v2 package load | 1.30s |
| v2 first solve() [:mixed] | 43.78s |
| **v2 total [:mixed]** | **45.08s** |
| v2 first solve() [:none] | 10.79s |
| v2 second solve() (different system) | 5.72s |

The root-cause pass cut build/init/solve from 14.44s to ~9.81s by isolating mutually exclusive
compiler branches, using adaptive direct polynomial lowering, and stabilizing constructor types.
A later pass took it to ~9.2--9.4s by not factorizing the `MatrixWorkspace` QR field at
construction and by putting shape dispatch behind an inference barrier. A third pass reached
~8.4--8.8s while testing four changes: temporarily dropping three non-polynomial unary
interpreter variants, installing the extended-precision wrappers on first use, extracting the
support on demand, and handing `skeel_row_scaling!` a matrix instead of a `MatrixWorkspace`
(see `01_decisions.md`). The unary-variant removal recovered about 0.26s but was rejected because
those operations belong to the planned expression frontend. The table below therefore records a
historical experiment, not the current absolute TTFX; remeasure with the variants retained.
Over 3 interleaved fresh-process pairs at `-t 4`, the experimental build had 21 of 21 paired
differences negative:

| workload | before | after | |
|---|---:|---:|---:|
| `total_degree_interpreted_serial` | 9.339s | 8.437s | −9.7% |
| `newton_standard` | 6.031s | 5.259s | −12.8% |
| `singular_endgame` | 9.333s | 8.505s | −8.9% |
| `overdetermined_total_degree` | 10.009s | 9.180s | −8.3% |
| `total_degree_compiled_all_serial` | 10.243s | 9.532s | −6.9% |
| `witness_set_build` | 16.046s | 14.984s | −6.6% |
| `polyhedral_interpreted_serial` | 13.745s | 13.170s | −4.2% |

Polyhedral gains least because it still extracts the support and is dominated by
MixedSubdivisions. Full SnoopCompile/JET/Cthulhu/invalidation report: `05_ttfx_invalidations.md`.

A fourth pass took the untaken `MatrixWorkspace` QR branch out of the square routes by
erasing both its entry points behind shape-chosen `FunctionWrapper` fields, which keeps the
tracker hot path free of runtime dispatch (`01_decisions.md`, "The QR path is erased behind a
shape-chosen FunctionWrapper"). Over 5 interleaved fresh-process pairs at `-t 4` on
`total_degree_interpreted_serial`, all 5 paired differences favour it: 8.427--9.062s before
against 8.312--8.426s after, ~0.15s median. A square-only session now compiles zero
specializations of `qr!`, `qr_ldiv!`, `reflector!` and `lmul_Q_adj!`; the first tall solve pays
~0.35s to compile the path on demand.

SnoopCompile v3.2.5 does not load on Julia 1.12 (`UndefVarError: Compiler.Params`), so that pass
used `SnoopCompileCore`'s `@snoop_inference` tree directly: on 1.12 the per-node costs live on the
`CodeInstance` as `time_infer_self` / `time_compile`, `UInt16` fields holding `Float16` seconds
(`reinterpret(Float16, ci.time_infer_self)`). Codegen outweighs inference on a first solve, 9.97s
against 6.20s for `polyhedral_interpreted_serial`, so a change that removes generated code counts
for more than one that only removes inference.

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
- The eight builder structs each repeat `tracker_options` / `endgame_options`. A shared
  `BuilderOptions` field would remove sixteen declarations and add an indirection at
  every use, so it was left alone; the construction *tail* they all shared is now
  `_endgame_tracker` (see `01_decisions.md`)
- Em dashes are used as sentence pauses in comments across the older `src/` files,
  against the repo's writing rule. New and touched code is clean; a global sweep
  would be pure churn on files nothing else is changing
- `Compiler.inferiterate_2arg` re-inference costs ~245ms of every first call. It
  survives a build that never loads OhMyThreads, so the InitialValues invalidations are
  not the cause and the trigger is still unattributed (`01_decisions.md`, "Dependency
  invalidations are largely inert")

### Infrastructure

- No benchmark CI: regressions go unnoticed
- TTFX gate (construction + first solve < 5s) still unmet; last clean measurement was
  ~8.6s, since reduced by 1.28s in a paired comparison but not re-measured absolutely.
  PrecompileTools stays deliberately disabled; `src/precompile_signatures.jl` is a
  separate mechanism (`01_decisions.md`, "Tape executors are precompiled by signature")
- The routes built on top of the plain total-degree stack cost 2.6s to 8.4s more on
  first call, each measured in a fresh process at `-t 4` (`make ttfx`), against
  8.59s for `total_degree_interpreted_serial` in the same run. The table predates
  the tape-executor precompilation, which took the common path down by 1.28s, and
  the routes below share that path, so every absolute number here should be read as
  an upper bound until the sweep is re-run on a quiet machine:

  | workload | first call (s) |
  |---|---:|
  | `slice_solve` | 11.16 |
  | `result_iterator_lazy` | 11.74 |
  | `parameter_sweep` | 11.93 |
  | `slice_solve_projective` | 14.10 |
  | `witness_set_build` | 15.27 |
  | `subspace_sweep_extrinsic` | 16.55 |
  | `subspace_sweep_intrinsic` | 17.04 |

  Each of these compiles a wrapper stack the plain route never builds
  (`SlicedSystem`, the chart row, the subspace homotopies, the retargeting worker
  states), so the cost is additive rather than a regression in the common path.
  The two subspace sweeps are the worst because a geodesic retarget and both
  regimes' coordinate conversions land in the same first call.
