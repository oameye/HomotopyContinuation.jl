# Status

Last updated: 2026-08-06.

**Reproduce:**
- `make benchmark` — steady-state timings
- `make compare` — v3/v2 ratios
- `julia --project=benchmark benchmark/compare/tracking.jl` — end-to-end solve comparison
- `make test` runs the core suite (61 files, parallel via ParallelTestRunner) then the certification subpackage; `make test-cert` runs only the latter

## Summary

Total-degree solving is 1.8–3.6x faster than v2; polyhedral matches within noise (0.95–1.01x).
Cold load + construction + first solve is ~10.6s versus v2's 45s, with no precompile workload.
Endgame is at v2 result parity on v2's default parameters.

At v2 parity: threading, overdetermined systems, parameter homotopies, monodromy (group actions,
linear subspaces, trace test), certification (Krawczyk with Arb fallback and the low-memory
route for a `ResultIterator`, in the separate `lib/HomotopyContinuationNextCertification`
subpackage), and witness sets / NID (`witness_set`,
`trace_test`, `membership`, `regeneration`, `decompose`, `nid`, including projective,
zero-dimensional, parametric, and rational cases).

No remaining gaps against v2 on the executor axis: `Serial`, `Threaded` and
`DistributedExecutor` cover single-task, multi-task and multi-process tracking.

A testset-by-testset comparison against v2's suite on 2026-07-29 found 11 v2 test files
that were not fully ported, all because the feature behind them did not exist yet. They are
listed under "v2 parity gaps" below, the closed ones marked as such.

As of 2026-07-31 every feature gap on that list is closed. What is left there is not a feature:
the public interface surface (declaring which names downstream code may depend on) and two
entries in the evaluation sweep's system collection, both recorded with their reasons.

## Feature Checklist

### Done

- [x] `solve(F)` with CommonSolve.jl `init`/`solve!`
- [x] Total-degree and polyhedral start systems
- [x] Parameter homotopy (CoefficientHomotopy with linear parameter interpolation)
- [x] `System{P,V,M,S}` type (compile mode `M` and square/overdetermined shape `S`
  live in the type domain; caches interpreters for all eval modes and stores original MP polys)
- [x] `SystemEvaluator` / `HomotopyEvaluator` type firewall (FunctionWrapper, 10 wrappers each,
  including a DF64-output evaluate for extended-precision residual combining)
- [x] **Every evaluator carries the thunk that rebuilds it** (`_clone`). Tapes are mutable, so
  a task needs its own evaluator, and nothing else survives the erasure. Each wrapper system
  implements `_clone_system`; `_clone_system_evaluator` collapsed to one method, homotopies
  clone, and the extension serializes an evaluator as its cloner. A caller's own type falls
  back to `deepcopy`, which the `deepcopy_internal` hook makes safe. TTFX unchanged
  (`01_decisions.md`).
- [x] StraightLineHomotopy, CoefficientHomotopy, ToricHomotopy
- [x] Cauchy product Taylor convolution for parametric homotopies
- [x] Two-stage toric reparameterization (weight renormalization when max_weight ≥ 10)
- [x] Predictor-corrector tracker (Padé 2,1 + adaptive step + s-plane Hermite for winding > 1)
- [x] Multi-round iterative refinement in predictor (weighted-norm orders 2–3, inf-norm order 1)
- [x] Newton corrector (α-theory) with DoubleF64 extended precision, escalated both from an
  accepted step (`update_precision!`) and from three consecutive rejected steps whose
  correction stopped contracting after reaching the prediction. The second route is what
  carries a path into a singular endpoint: a correction that stalls on the double-precision
  residual is not rescued by a smaller step, so without it the step size shrinks until it
  underflows. v2 has only the first route
- [x] Step control: ω extrapolation, convergence-rate rejection, near-target scaling
- [x] Endgame tracker — Puiseux valuation, winding number estimation, singular Cauchy endgame
  (geometric stepping λ=0.25), at-infinity/at-zero detection, cubic Hermite endpoint prediction,
  jump-to-zero gating for m=1 paths
- [x] Solution deduplication (union-find clustering, sort-and-window candidate sweep
  scaling to thousands of paths), multiplicity tracking, and opt-in group-action
  symmetry via `recluster(::Result; group_action, atol, rtol)`; `clusters(::Result)` /
  `cluster_of(::Result, i)` expose the partition (the orbits, after a `recluster`)
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
- [x] Automatic equation normalization (scales an equation with O(10^8+) coefficients to O(1)).
  Both front-ends go through it: the polynomial one takes the largest coefficient, the
  `Expression` one evaluates the tree with every variable set to 1 and every literal replaced
  by its absolute value (`expression_scale`), which is the ℓ1 coefficient norm of a polynomial
  and a ratio of the two for a rational function. v2 instead leaves the target alone and scales
  its total-degree start system per equation to match
- [x] Condition number of the row- and column-scaled Jacobian (`_scaled_cond`), read by the
  endgame's singularity, at-infinity and precision decisions. Higham's truncated 1-norm
  estimator, so a lower bound on the dense `cond(D_r A D_c, Inf)` (within a factor 3 over a
  12500-matrix sweep) and never above it, at every row scale and whether or not the LU factors
  already carry the row scaling. That second state is what a Newton solve leaves behind, and it
  covers four of the five call sites
- [x] AllocCheck zero-allocation enforcement on all hot paths
- [x] Integration tests from v2 with exact result parity
- [x] **Non-polynomial (rational/transcendental) expression input.** `Expression` is a
  `Number` subtype built on a Moshi `@data` ADT (`ENum`/`EVar`/`EAdd`/`EMul`/`EPow`/`EFn`),
  declared with `@var`/`@unique_var`. It canonicalizes on construction (flattening, constant
  folding, like-term and power collection), so `x - x`, `x*inv(x)` and `(a*b)^k` reduce
  structurally. Covers division, negative integer powers, `sqrt`, `sin` and `cos`.
  - `SExpr` gained an `SUnary(kind, arg)` variant; the tape compiler lowers it to
    `OP_SQRT`/`OP_SIN`/`OP_COS`, and division/negative powers reuse the `SPow`-with-negative-
    exponent path that already emitted `OP_DIV`/`OP_INV`/`OP_INVSQR`/`OP_POW_INT`.
  - Jacobians come from symbolic differentiation on `Expression` (product, power, quotient,
    chain rules for `sqrt`/`sin`/`cos`), not `MP.differentiate`.
  - `System(exprs; variables, parameters)` accepts `Expression`s, and `MP.RationalPoly` input
    (`u₁/x² + u₂`, `y[1:2] ./ y[3]`) converts to `Expression` automatically. All three compile
    modes work; `degrees` reports `-1` for a non-polynomial equation and `TotalDegree`/
    `Polyhedral` reject those systems with a targeted error, on the sliced routes as well as
    the plain ones.
  - The solve routes that rewrite equations rather than just evaluate them dispatch on the
    front-end: `fix_parameters` (used by sliced solves and witness sets) substitutes through
    `subs` instead of `MP.subs`, and `support_coefficients` recovers the polyhedral support by
    expanding the expression tree into `exponent vector => coefficient`, since the front-end
    keeps products and powers unexpanded. Both agree term for term with the same system built
    through DynamicPolynomials.
  - `DoubleF64` gained `exp`, `log`, `sin`, `cos`, `sincos`, `tan`, `asin`, `acos`, `atan`,
    `sinh`, `cosh`, `tanh` (~32 digits). `sin`/`cos` switch
    to angle addition over the two limbs past `|a| = 2⁵³`, where double-double reduction modulo
    `2π` drops below Float64 accuracy, and `sinh`/`cosh` take a separate branch past `|a| = 40`,
    where `e ± 1/e` would be NaN. `log` and `atan` are one Newton step on `exp` and on
    `sin`/`cos` from a Float64 seed; `asin`/`acos` reduce to `atan`. `log` splits off the binary
    exponent first, because `exp(-x)` for the Newton step underflows the double-double range
    (`floatmin(DoubleF64) = 2e-292`, not `Float64`'s `2e-308`) once `|a|` leaves it, which
    silently costs 7 digits. For `a` near 1 the correction cancels and `log` is accurate
    absolutely rather than relatively.
    `ComplexDF64` takes `exp`, `sqrt`, `sin`, `cos`, `sinh`, `cosh` from the generic `Base`
    complex methods; `log`, `tan`, `tanh`, `asin` and `acos` are written out, since Base routes
    those through `log1p`, `asinh` and `typemax` on the real part at `Float64`-tuned thresholds.
  - Rectangular `Interval`/`IComplex` arithmetic gained `sqrt`, `exp`, `log`, `sin`, `cos`,
    `tan`, `asin`, `acos`, `sinh`, `cosh`, `tanh` and a non-integer `^`, so every tape stays
    on the Float64 Krawczyk path and
    escalates to Arb only when the test genuinely fails. v2 has none of these, and its
    Float64 path is the only way into `certify`, so on 2.22.1 `certify` of a system carrying
    any of them throws a `MethodError` (`no method matching sin(::IComplexF64)`) rather than
    falling through to Arb: v2 cannot certify a transcendental system at all. The complex
    `sqrt` takes whichever of
    `|z| ± Re z` does not cancel and recovers the other root from `2uv = Im z`; the naive
    form inflates the enclosure of a box around a real solution to the square root of its
    width. A box meeting the branch cut returns empty, which falls through to Arb.
    The complex `log` reads its argument off whichever of `y/x` and `x/y` cannot straddle a
    pole, which is what keeps a box next to either axis tight; `asin`/`acos` and the
    non-integer `^` are built on it, so its rejection of the negative real axis is also
    theirs. `tan`/`tanh` divide by a real interval that straddles zero exactly on a pole,
    so the empty result is the pole check.
  - The Acb kernels reject a ball on a branch cut rather than trusting Arb, which for a
    narrow straddling ball returns a sound but discontinuous enclosure the Krawczyk
    hypotheses cannot use: `acb_asin`/`acb_acos` at `2.0 ± 1e-12` answer, and
    `acb_pow` on the negative real axis answers at every width. v2's Acb kernels are bare
    `Arblib` calls with no cut check anywhere, `sqrt` and `pow` included.
  - Front-end lowering (`_lower_input`) happens before the `@nospecialize` builder chain.
    Dispatching on the input representation inside it made inference walk both front-ends
    on every build, so a polynomial system paid ~0.6 s of TTFX to infer the expression
    pipeline it never runs.
  - `LinearAlgebra.det` on an `AbstractMatrix{Expression}` expands by cofactors: the generic
    `det` factors, which needs `abs` to pick a pivot and division to eliminate.
  - `conj` recurses and conjugates numeric literals, leaving variables alone, so `adjoint` on a
    matrix of expressions agrees with DynamicPolynomials. v2 conjugated neither.
  - Closes the two v2 monodromy testsets ("Monodromy rational functions", triangulation),
    v2's "certify uses approximate inverse of jacobian", the `small_rational`/`sqrt_parameters`
    entries of v2's `model_kit/e2e_test.jl` sweep (system, homotopy and Acb), and the
    Subs / Evaluation / Linear Algebra / Modeling / rational-functions testsets of v2's
    `model_kit/symbolic_test.jl`.
  - `regeneration`/`nid`/`intersect` take rational input, through `num_den` and the front-end
    dispatch described under witness sets/NID below. Closes v2's `nid_test.jl`
    "rational systems" testset.
- [x] Threading via OhMyThreads.jl: `Serial`/`Threaded` executor types, builder/worker-state
  pattern for thread-safe evaluator cloning, `@tasks`/`@local` work distribution
- [x] Multi-process tracking: `DistributedExecutor` in the `Distributed` package extension,
  covering every route (total degree, polyhedral, parameter homotopy, start to target,
  an explicit homotopy object, subspace moves in both regimes, both sweep kinds, and
  monodromy). Dynamic batching over a
  `RemoteChannel`, each process internally threaded, results stored by global path index so
  that at a fixed seed they are bit-identical to `Serial()` at any batch size.
  `Serialization` methods for `System`, `_SupportSystem`, `CompositionSystem`,
  `FixedParameterSystem` and `SystemEvaluator` ship builders as plain data
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
  `start_parameters!`/`target_parameters!`) and `solve(F, starts, p_start, p_target)`
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
- [x] `solve(F, [sols, p], Monodromy(), exec)` at full v2 parity: `find_start_pair`, serial and threaded
  (Channel job queue) execution, `MonodromyOptions` (~27 explicit kwargs, no splatting),
  `reuse_loops` (`:all`/`:random`/`:none`), heuristic stop, `target_solutions_count`,
  equivalence classes via group actions, `LinearSubspace` parameters, permutations, trace
  test. One deliberate signature deviation: `parameter_sampler` is called `sampler(rng, p)`
  rather than v2's `sampler(p)`.
- [x] `verify_solution_completeness` (trace test with augmented system, auxiliary monodromy,
  singular-value trace check)
- [x] **Certification** (`certify`) at full v2 parity, in the separate
  `lib/HomotopyContinuationNextCertification` subpackage (the only Arblib consumer).
  Krawczyk operator with ε-inflation over interval arithmetic (`Interval`/`IComplexF64`, reusing
  the generic tape interpreter), arbitrary-precision Arb fallback (`AcbInterpreter`, in-place ops
  + `setprecision!`, escalates 128→256 bits), `SolutionCertificate`/`ExtendedSolutionCertificate`,
  all accessors and counts, duplicate grouping via interval tree, `save`,
  `show_straight_line_program`, and every input form
  (`Result`/`PathResult`/`Vector`/single/`MonodromyResult`, with the parameter values positional).
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
    a linear `L` gets a seed-reproducible chart row; a parametric `F` is rejected, and
    `fix_parameters(F, p)` is what the route accepts. Solutions are ambient (v2's too, though it
    reaches them through an intrinsic sliced system).
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
  - `solve(F, starts, p_start, targets, Sweep(), exec)` and
    `solve(F, starts, L_start, targets, Sweep(), exec)`: one homotopy built and retargeted per target
    (`target_parameters!` through the concrete handle in the worker state), `transform_result`,
    `transform_parameters` and `flatten` matching v2's four return shapes. Threading runs over the
    (target, path) product, each task owning its worker state and retargeting it when it crosses a
    target boundary, so the speedup no longer caps at the number of targets: measured 4.31x versus
    1.36x on 12 threads for a one-target sweep of 125 paths, and unchanged at 24+ targets. Chunks
    are contiguous in target-major order, so this costs at most `n_targets + ntasks` retargets. The
    many-target route is its own verb because the single-vs-many distinction cannot be carried by a
    positional slot's type: numeric metadata targets (`1:20` plus `transform_parameters`) are
    indistinguishable from one multi-value target, and v2's runtime `isa(…, Number)` branch would
    make the return type value-dependent.
  - `result_iterator(...)` → `ResultIterator`, the typed replacement for v2's `iterator_only`
    kwarg: lazy per-path tracking for the total-degree, polyhedral, sliced, parameter, subspace
    and explicit-homotopy routes (serial), and `Result(ri)` for clustering and excess
    reclassification. A `ResultIterator` may be passed as the start solutions of another solve.
    Selection is one index domain, over the start solutions: `selection(ri)` is the current
    mask, `selection(f, ri)` tracks once and records `f` at the selected positions, and
    `restrict(ri, mask)` narrows (never widens) without tracking. `filter(f, ri)` keeps the
    results instead of a mask, `Iterators.filter` keeps that lazy, and `ri[k]` tracks the `k`-th
    selected path alone. Iteration goes through the cache's single tracker, so one pass at a
    time.
  - Tests: `test/{sliced_solve,subspace_solve,many_targets,result_iterator}_test.jl`
- [x] **Witness sets and numerical irreducible decomposition (NID)** for affine systems, in core
  (`src/solving/{witness_set,regeneration,nid}.jl`). Provides `Witness` (dim/codim, explicit
  subspace, move), `trace_test`, `membership`, `intersect(W, H)` / `intersect(W, f)`; `Regeneration`
  (u-regeneration, Duff/Leykin/Rodriguez); `Decomposition` plus `NumericalIrreducibleDecomposition`
  (`ncomponents`, `degrees`, `witness_sets`,
  hand-rolled degree table with no PrettyTables dep). Covers projective witness sets,
  zero-dimensional varieties, witness sets of a `fix_parameters` system, rational input to
  `Regeneration`/`Decomposition`/`intersect`, and threaded membership and intersection.
  Tests: `test/{witness_set,nid}_test.jl`.

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
    parameter-free system, so no parameter plumbing reaches moves/trace/membership/decomposition.
  - Threaded intersection clones evaluators instead of `deepcopy`ing trackers (unsafe with
    FunctionWrappers) and pushes endpoints in serial order.
  - `membership` is bit-identical across threading modes: all randomness is drawn in the
    driver, off the stream its `seed` builds, before any task starts. The query subspace
    direction is genuinely random per query, unlike v2's fixed axis-aligned frame.
  - Adds the `weighted_normal` monodromy sampler (v2 has it only for regen/decompose).
    Both samplers take the random number generator as their first argument, `sampler(rng, p)`,
    where v2's take only `p`; that is what lets a route's `seed` determine its loops.
  - Both input front-ends reach `regeneration` / `nid` / `intersect`, the routes that rebuild
    equations rather than only evaluating them. `_regeneration_equations`, `_u_degree`,
    `_u_start_equation`, `_numerator_system` and `_rename_variables` dispatch on the equation
    type; `num_den(::Expression)` (v2's `get_num_den`) splits a rational equation `f = p/q`.
    The witness set of the hypersurface `f = 0` is computed from `p` and the zeros of `p` that
    also kill `q` are dropped by `_drop_poles`, instead of v2's re-track through an L→L homotopy.
    The test is the first-order distance from the point to `V(q)`, `|q(r)| / ‖∇q(r)‖`, relative
    to `‖r‖`. Value and gradient scale together with `q`, so unlike a residual threshold on `f`
    the decision survives rescaling `f` or `q`; and being a distance rather than a magnitude it
    separates a pole from a zero that merely sits near one, which is the case `(x − ε)/x` at
    `x = ε`. `POLE_DISTANCE_TOL = 1e-10` sets the boundary: a zero closer than that to `V(q)` is
    dropped, and a pole is kept if the point that reached it is less accurate than that.
    The u-homotopy deforms `(u^d − 1)/q` into
    `p/q`, so both endpoint systems are singular on the same set. Equations whose numerator or
    denominator is not polynomial in the variables (`sqrt`/`sin`/`cos` of a variable, including
    `1/sqrt(x)` and `x/(1 + sqrt(x))`) are rejected up front,
    and `intersect` rejects input that mixes the two front-ends. `witness_set` on a rational
    system keeps asking for cleared denominators (its total-degree slice would be over the
    numerators, and `V(p₁, …, pₘ)` is larger than `V(F)`); equation-by-equation regeneration is
    what separates the two, as it does in v2.

- [x] **System composition** (`src/core/composition_system.jl`): `compose(G, F)` and the infix
  `G ∘ F` build `G(F(x; p); p)`, accepted by every route that only evaluates the system
  (`solve` from start solutions, `solve` by total degree, `Monodromy`, `newton`,
  `find_start_pair`, all typed on `SystemLike = Union{System, CompositionSystem}`). Routes
  that need the composed monomials or equations (polyhedral, witness sets,
  `verify_solution_completeness`) reach a composition through `System(C)`, matching how v2
  reaches them through `System(F::AbstractSystem)`.
  All three v2 composition tests are ported: the evaluate/Jacobian/Taylor sweep and the
  parameter-list assertions (v2 `systems_test.jl`) and `solve(e ∘ f ∘ g)` by total degree and
  by polyhedral (v2 `solve_test.jl`) live in `test/composition_test.jl`; the symmetroids
  monodromy test (`L₂ ∘ f ∘ L₁`, 305 solutions, custom `distance`) is in
  `test/monodromy_v2_parity_test.jl`.

  Key design choices:
  - Composition happens at the **evaluator** level, as with `RandomizedSystem` and
    `SlicedSystem`: nested `_ComposedSystem <: AbstractSystem` wrappers, each behind a
    `SystemEvaluator`. Substituting `F` into `G` symbolically keeps the tree unexpanded, but
    differentiating and running CSE over the substituted tree costs 83s for the symmetroid
    composition, against ~0.1s to wrap the evaluators; the wrapper does the same chain-rule
    product numerically, one `mul!` per Jacobian.
  - A composition of any depth is **one concrete type**: stages are stored innermost-first as
    `FunctionWrapper{SystemEvaluator, Tuple{}}` clone thunks, so `_clone_system_evaluator`
    rebuilds the whole chain per worker and threading works unchanged.
  - Taylor: the order-K coefficient of `G ∘ F` needs the series of `F` to order K, and a
    `SystemEvaluator` returns one order per call, so the inner tape runs K + 1 times per
    composed order (v2 fills all orders in one run through a `TaylorVector` output). Both the
    constant-parameter and the `TaylorVector`-parameter variants are implemented, the latter
    being what parameter homotopies call. Lower-order views of a series share the backing
    matrix, so no series is copied.
  - `System` divides an equation whose coefficients are far above unit scale by that scale.
    That leaves `V(F)` alone but changes `F` as a map, so `System` now records the factors
    (`equation_scales`) and the fold multiplies each inner stage's factors back into its
    output. Without it, `G ∘ F` would silently solve `G(F(x)/s)`.
  - Parameters: both stages see the same vector (v2's rule), so stage parameter lists must
    agree unless one is empty, and a parameter-free stage simply ignores the vector.
  - `System(C::CompositionSystem)` rebuilds the composed equations by substitution, undoing
    every inner stage's scaling symbolically. It agrees with `C` up to a constant factor per
    equation, since `System` renormalizes what it is handed. Each stage keeps a second thunk
    (`StageEquations`) that converts its equations to `Expression`s only when asked, so
    `compose` stays cheap. This is the escape hatch v2 spells `System(F::AbstractSystem)` and
    uses for `certify`, `witness_set` and `is_homogeneous` on a composition.
  - Degrees and homogeneity are folded from the stages by weighting: `deg(gⱼ ∘ F)` is the
    degree of `gⱼ` in the weights `deg(fᵢ)`, and `gⱼ ∘ F` is homogeneous when `gⱼ` is
    homogeneous in those weights and every `fᵢ` is. `MonodromySolver` reads `is_homogeneous`
    to decide whether to put the problem on an affine chart, so a wrong `false` there tracks
    a projective problem in ambient coordinates; weighting is what makes a composition such
    as `[a·u − v, w² − b·u, u + v] ∘ [x², y², z]` come out homogeneous, which the plain rule
    `deg(gⱼ) · d` cannot see because the inner degrees `[2, 2, 1]` are not uniform. Where they
    are uniform the two rules agree and the fold takes the cheap one, which reads no stage
    equations. Degrees are upper bounds and homogeneity is structural, as for a `System`.
    The folded degrees are also what lets total degree take a composition without rebuilding
    anything: `_init_total_degree` needs the degrees, the evaluator and the clone, all of
    which a composition has. A stage that is not polynomial in its variables gives a degree
    of `-1` and is rejected with a message naming the composition rather than the
    non-polynomial-equation message a `System` gets.
  - `find_start_pair` on a composition runs Newton in `(x, p)` jointly on `_StartPairSystem`
    (`src/core/start_pair_system.jl`), a wrapper turning an evaluator into a system in
    `[x; p]`; its parameter Jacobian block is the order-1 Taylor coefficient along
    `p + eⱼ t`, exact where v2 uses FiniteDiff. A `System` keeps its symbolic strategies,
    which get the parameter derivatives from one tape.

- [x] **Grassmannian geodesic memo** (`src/core/subspace_homotopies.jl`): each subspace homotopy
  owns a `GeodesicCache`, a round-robin ring of up to 8 geodesics keyed on the Stiefel frame pair
  the geodesic was built from (`intrinsic(L).X` or `extrinsic(L).A`), so a retarget that revisits
  a `(start, target)` pair costs a linear scan instead of a Grassmannian SVD. The key is the
  frames rather than the `LinearSubspace`es because the geodesic is independent of the offsets,
  and it is compared by value rather than by identity because `set_subspaces!` rebuilds an
  equal-but-distinct start subspace on every retarget whenever the γ perturbation is on. Frames
  are stored as copies since `copy!(::LinearSubspace, ::LinearSubspace)` can overwrite a subspace
  the caller passed in. Per-homotopy rather than v2's module-global LRU of size 128: worker
  states hold one homotopy each, so the memo needs no lock, and the working set of a monodromy
  loop or a subspace sweep is a handful of pairs.

- [x] **Distributed monodromy** (`ext/.../monodromy.jl`): `solve(F, [sols, p], Monodromy(),
  DistributedExecutor())`. The shared state never leaves the calling process: it keeps the job
  queue, the `UniquePoints` set, the trace matrix, the statistics and the loop list, and hands
  out single loops as `MonodromyJob`s (loop, start point, its `ω`/`μ`/precision flag) that come
  back as `MonodromyJobResult`s (the `PathResult` plus, when the trace test is on, that loop's
  three trace columns). The queue stays on the driver rather than in the `RemoteChannel` so only
  `batch_size` jobs per task are ever in flight, which is what lets a timeout or a reached
  `target_solutions_count` drop the rest of a generation instead of waiting it out. Deduplication,
  permutation recording and the trace fold then run in one thread on one process, and the
  dispatch order is the serial one, so nothing about the algorithm changes. `track_loop!` gained
  a trace-sink argument (the solver itself, or a `TraceColumns` for a process with no solver at
  hand) and the subspace worker builder became a named `SubspaceMonodromyBuilder` so it ships as
  data like the other seven builders. On one machine this is the slower option and `Threaded()`
  remains the default: on Steiner over six cores, threads reach ~5x serial and three processes
  1.2x to 2.2x, the gap being a ~5 ms channel handoff per job against ~6 ms of tracking (measured
  in `01_decisions.md`). It is the multi-machine route, and it closes the gap on its own once a
  loop costs much more than the handoff

### Not Done

- [ ] Compile-mode benchmark, v2 side: v3 `COMPILED_ALL` vs v2 `:all`, plus fresh-session
  first-solve per v3 default candidate (the v3-only matrix is measured; see `04_compile_modes.md`)
- [ ] Benchmark CI

### v2 parity gaps

Found by comparing every v2 `@testset` against the v3 suite on 2026-07-29, and re-run against
v2.22.1 on 2026-08-01 (see "Re-audit against v2.22.1" below). Each entry names the v2 test that
was unported until the feature landed. Ordered by consequence.

- [x] **Multi-homogeneous (variable-group) total degree.** `System(polys; variable_groups)`
  stores the groups as index vectors and `is_homogeneous` becomes per-group, which is strictly
  stronger and is what makes the count valid. The start system is a `System` built from
  `Expression`s (a product of powers of linear forms per equation, so the interpreter supplies
  its Taylor coefficients), and its solutions come from one small LU per assignment of
  equations to groups. `_multi_start_coefficients` puts an identity block in the leading
  columns, which is what makes a single group covering every variable reproduce the plain
  route exactly, paths and all. The M chart rows are one codim-M `LinearSubspace`, so the
  target is charted by the existing `SlicedSystem` rather than a multi-chart wrapper, and
  charting happens *before* the square-up with a zero fold block on the chart rows: they then
  stay chart rows, the excess check runs on a full-rank `[F; charts]`, and the folded degrees
  keep the per-group maxima instead of v2's `[D I]`, which inflates them (proj_ov: 5 paths
  against v2's 17). `_check_single_group` rides on `_affine_chart`, the one place a route draws
  a chart for all the variables at once, so a route that would leave a cone per remaining group
  rejects the input without having to remember the check. A grouped system that never reaches a
  chart draw (an affine slice, a group-inhomogeneous system) is accepted with the groups unused.
  Closes `solve_test.jl` "total degree (variable groups)" and `symbolic_test.jl`
  "System variables groups + homogeneous". Tests: `test/variable_groups_test.jl`.
- [x] **Projective total degree and polyhedral.** Both plain routes test `is_homogeneous` ahead
  of the shape and delegate to the sliced route with the whole ambient space as `L`
  (`_init_projective`), so the seeded chart draw keeps its single home in `_sliced_solve_setup`.
  Homogeneity before shape is the consequential part: it makes a *square* homogeneous system
  projective rather than a cone in ambient coordinates, which is v2's rule and the only reading
  under which it has finitely many solutions. `_check_projective_determined` counts the chart row,
  so `m ≥ n − 1` is determined, and path counts match v2. A homogeneous `CompositionSystem` or
  `FixedParameterSystem` goes through `_polynomial_system`, paying a rebuild the square case
  would not need if `SlicedStraightLineBuilder` took a `CloneableSystem`.
  The branch must stay behind `Base.inferencebarrier`: without it the plain route infers the
  sliced stack, worth 6.5s versus 3.5s on a first affine `solve(F, TotalDegree())`.
  Closes the `proj_square`, `proj_ov` and `proj_ov_reordering` cases of `solve_test.jl`
  "total degree (simple)" and the projective half of its "polyhedral" testset.
  Tests: `test/solve_test.jl` "projective: …".
- [x] **`AbstractResult` / `AbstractSolutionResult`**: `Result` and `MonodromyResult` share one
  set of accessor bodies (`solutions`, `real_solutions`, `nsolutions`, `nsingular`,
  `nnonsingular`, `nreal`, `results`, `nresults`), so `MonodromyResult` gained `Result`'s filter
  keywords and `path_results`. `WitnessSet` and `NumericalIrreducibleDecomposition` are
  `AbstractResult` but hold no path set. `is_success` and `seed` stay per type; see
  `01_decisions.md`.
- [x] **`fix_parameters(F, p)`**, and **`FixedParameterSystem`**. One public operation fixes a
  parametric system at one parameter value, and its result is what every route accepts:
  `solve(fix_parameters(F, p), TotalDegree())`, `solve(fix_parameters(F, p), L, Polyhedral())`,
  `solve(fix_parameters(F, p), Witness())`. No solve route takes parameter values any more, which
  removed 22 `target_parameters::Union{Nothing, …}` keywords across five files and gave
  `Decomposition` and `Regeneration` the capability for free; `start_parameters`/`target_parameters` survive
  only on the parameter-homotopy and sweep routes, where they name two different ends. A
  `System` has the values substituted into its equations; a `CompositionSystem` has no
  equations to substitute into and gets a `FixedParameterSystem`, which binds them at the
  evaluator level and pays a second FunctionWrapper hop of 11% to 19% per kernel call
  (measurements in `01_decisions.md`, "Fixing parameters"). Tests:
  `test/fixed_parameter_test.jl` (which representation each input gets, the bound evaluator
  against the substituted system on every interface method, per-worker cloning, both
  algorithms, square and overdetermined, all three compile modes, both executors, composition)
  plus the bound wrapper's hot-path entry in `test/alloc_check_test.jl`. Closes
  `systems_test.jl` "FixedParameterSystem".
- [x] **`solve(G, F, starts)`** between two parameter-free systems: tracks
  `γ·t·G(x) + (1 − t)·F(x)`, with `γ` and the projective chart drawn from `seed`. Both may be
  any `CloneableSystem`, so a parametric problem reaches it as
  `solve(fix_parameters(f, p₁), fix_parameters(f, p₀), starts)`, which replaces v2's
  `start_parameters`/`target_parameters` here and is the ported half of v2's testset. Threads
  and distributes like the other routes. A homogeneous pair takes any projective
  representatives; an overdetermined pair is tracked rectangular, as in v2, with no excess
  check. Closes the second half of `solve_test.jl` "solve (start target)".
- [x] **`solve(H::AbstractHomotopy, starts, exec)`** on all three executors: `Serial()` tracks
  the homotopy given, the others rebuild it per task, both bit-identical to serial (76.7 ms
  against 14.6 ms on 8 threads, 81 paths). A caller's homotopy with no `_clone_homotopy`
  method is rebuilt by `deepcopy`, so it threads too. Start points are in the
  homotopy's coordinates, except that an `AffineChartHomotopy` takes projective
  representatives. Closes `solve_test.jl` "solve (Homotopy)". Both routes in
  `src/solving/homotopy_solve.jl`; tests: `test/homotopy_solve_test.jl`.
- [x] **`early_stop_callback`** on `TotalDegree`, `Polyhedral`, `Continuation`, on all three
  executors (driver-side at `batch_size` granularity under `DistributedExecutor`). v2 spells it
  `stop_early_cb`. `tracked_paths` now means the number of paths that actually ran, so `nfailed`
  stays 0 when a callback stops a run early; see `01_decisions.md`.
- [x] **SemialgebraicSets.jl integration**: `SemialgebraicSetsHCSolver`, in the
  `HomotopyContinuationNextSemialgebraicSetsExt` package extension. The solver type subtypes
  `SemialgebraicSets.AbstractAlgebraicSolver`, so it can only be *defined* where that package
  is loaded, and an extension cannot export a name; core therefore declares the constructor
  `function SemialgebraicSetsHCSolver end` and exports that, and the extension adds the method
  returning its own concrete type. Nothing else needs the type by name (`@set` takes an
  instance, and the one type-dispatched method, `promote_for`, is defined inside the
  extension), so this stays an extension where certification needed a subpackage.
  The solver holds an `algorithm`/`executor` pair instead of v2's `options::Any` kwargs bag,
  with `compile` forwarded to `System`; `algorithm` is restricted to
  `Union{TotalDegree, Polyhedral}`, the algorithms that build their own start system.
  `real_atol`/`real_rtol` are applied in the extension
  (`abs(imag(z)) <= atol + rtol * abs(z)`), since `is_real(::PathResult)` takes one combined
  `tol`.

  Four departures from the v2 design:
  - `excess_residual_tol` is an option of `TotalDegree`/`Polyhedral`, not of the solver, so
    the readmission happens in the excess check that already runs before clustering. v2 (and
    the first v3 port) flipped return codes afterwards and rebuilt the `Result`, clustering
    every solution twice and re-evaluating the system through the allocating `evaluate`. The
    solver keyword remains, forwarded to the algorithm, and `0.0` rather than `NaN` means off.
  - Variables come from `MP.variables(equalities(V))`, so a point's coordinates are ordered as
    `SemialgebraicSets` itself indexes them, and a set with a variable in no equation is
    reported positive-dimensional instead of being solved in fewer unknowns.
  - `promote_for` is `Float64`, not `float(T)`: tracking is done in `Float64` whatever the
    coefficients are, so `eltype(V)` now states the type a point is actually returned in and
    no conversion happens on the way into `V.elements`.
  - `solve(V, alg, exec)` returns the full `Result` and `real_solutions(V, solver)` returns the
    points, both beyond v2, which exposed only `SemialgebraicSets.solve`.

  Closes `semialgebraic_sets_test.jl`. Tests: `test/semialgebraic_sets_test.jl`.
- [x] **Solution and parameter file I/O**: `write_solutions`, `read_solutions`,
  `write_parameters`, `read_parameters` in `src/utils.jl`, hand-rolled rather than through
  DelimitedFiles (the format is two floats per line). Bertini-compatible and byte-identical to
  v2's writer; the reader additionally checks the declared count and rejects a line with more
  than two fields. Closes `utils_test.jl` "writing and reading".
- [x] **`path_info`** and the **tracker path iterator**, both in `src/tracking/path_info.jl`.
  `path_info(tracker, x₀, t₁, t₀)` returns a `PathInfo <: AbstractVector{PathStep}`, one
  `PathStep` per attempted step (arc length and step size, `ω`, `μ`, accuracy, trust region,
  condition estimate, first Newton update, predictor local error, predicted-to-corrected
  distance, `‖x‖∞`, accepted, extended precision), so the per-step invariant is structural
  where v2 keeps 12 parallel vectors, and `filter`/`map`/`count` work on it. Three-argument
  `show` prints a hand-rolled unicode table (no PrettyTables dependency) and elides the middle
  rows when `io` limits its height; two-argument `show` stays one line, and `path_table` prints
  every row. The condition column is the one v2 collects and never displays.
  `iterator(tracker, x₀, t₁, t₀)` is the stateful iterator over accepted steps yielding
  `(x, t)`; `t`'s type comes from dispatch on `t₁`/`t₀` (`PathIterator{Float64}` for a real
  pair), so `eltype` is concrete instead of v2's `Union` from a runtime flag, and a failed path
  stops the iteration rather than repeating its last point. `ResultIterator` remains a
  different thing (lazy per-path at solve level).
  Closes `tracker_test.jl` "path info" and "iterator". Tests: `test/path_info_test.jl`.
- [x] **`mixed_volume`**, the `Polyhedral` method of `paths_to_track`, and the polyhedral
  `only_torus` option. `paths_to_track(F, TotalDegree())` counts `init`'s start solutions
  rather than deriving the number a second time, so it agrees with `Result.tracked_paths` on
  every route (grouped, projective, squared-up, sliced) whenever no `early_stop_callback`
  fires, `tracked_paths` being the number of paths that ran. It therefore materializes them,
  where v2 counts a lazy iterator; the fix if that ever costs, for a Bezout number large
  enough to matter, is lazy start solutions rather than a second derivation of the count. The
  `Polyhedral` method and `mixed_volume(F)` follow the same rule, `mixed_volume(F)` being
  exactly `paths_to_track(F, Polyhedral(; only_torus = true))`, which is v2's definition and
  is correct for projective and overdetermined input without duplicating the support logic.
  `only_torus` divides each equation by its lowest monomial (`A .- minimum(A; dims = 2)`)
  instead of padding every support with the zero exponent vector, so it finds only the
  solutions with no zero coordinate. v3 exposes the one knob; v2 spells it as the pair
  `only_torus` / `only_non_zero`, where the second defaults to the first and nothing else
  distinguishes them. Closes `polyhedral_test.jl` "only torus" (92 and 54 paths, matching v2).
  Tests: `test/solve_test.jl` "Polyhedral: only_torus".
- [x] **Symbolic `Homotopy` type** (`Homotopy(h, vars, t; parameters, compile)`) for
  user-defined homotopies, in `src/core/symbolic_homotopy.jl`, with
  `size`/`length`/`nvariables`/`nparameters`/`variables`/`parameters`/`expressions`/`==`/`show`,
  callable evaluation `H(x, t[, p])` and `evaluate`/`jacobian`. `h` may be `Expression`s or MP
  polynomials. `solve(H, starts, alg, exec)` accepts it on every executor.
  `Homotopy` is to `AbstractHomotopy` what `System` is to `AbstractSystem`: a concrete
  symbolic front-end holding one `SystemEvaluator`, with no type parameter, so nothing
  downstream specializes on the input's polynomial type or compile mode. The equations compile
  to a system in `x` with parameters `[t; parameters]`, and `fix_parameters(H, p) -> Homotopy`
  substitutes the values into them exactly as `fix_parameters(::System, p)` does; a parametric
  `Homotopy` is rejected until then, the rule every other route follows. The tracked tape
  therefore has `t` as its only parameter, and the private
  `_PathParameterHomotopy <: AbstractHomotopy` writes one value per call: `dt/dt = 1`, the
  vanishing higher orders of `t` and the constant rows of the order-1 `x` series are written
  once, at construction. All Taylor orders come from the interpreter's parameter convolution,
  exact for arbitrary dependence on `t`. Against routing the same equations through a
  `ParameterHomotopy` interpolating `[1; p]` to `[0; p]`, which is what this replaced: 10-16%
  per call on a 4-variable, 5-parameter homotopy (~8% from the parameter handling, the rest
  from the shorter substituted tape), and the fixed parameters no longer move with `t` in the
  last bit. Closes `symbolic_test.jl` "Homotopy"; the `show` layout matches, and term order
  within an equation is v3's canonical one.
  Tests: `test/symbolic_homotopy_test.jl`.
- [x] **Symbolic utilities on `Expression`** (`src/model_kit/symbolic_utils.jl`): `expand`,
  `to_dict`, `horner`, `monomials`, `dense_poly`, `rand_poly`, `coefficients(f, vars)`,
  `coeffs_as_dense_poly`, `exponents_coefficients`, `poly_from_exponents_coefficients`,
  `to_number`, `convert(T, ::Expression)` for any `T <: Number`, `evaluate(exprs, subs...)`
  and calling an `Expression` on substitutions. `multi_degrees(F::System)` already existed.
  Everything that needs one term per monomial goes through `_expr_terms`, which expands into
  `exponent vector in vars => coefficient expression`; variables outside `vars` land in the
  coefficient, which is what lets `to_dict` report a coefficient such as `a + 1`.
  - `coefficients` and `exponents_coefficients` return `Vector{ComplexF64}` and throw for a
    symbolic coefficient, pointing at `to_dict`. v2 returns whichever of the two the values
    happen to be, from one function; v3 keeps the return type fixed and puts the symbolic case
    behind its own name. No capability is lost: `to_dict` is what v2's own `horner` and
    `coeffs_as_dense_poly` use internally.
  - `evaluate` narrows to a real result when every value is real, which is the point of v2's
    issues #500 and #511: a real-coefficient system must evaluate to `Vector{Float64}`, not to
    complex numbers with vanishing imaginary parts. This is the one place in v3 where a return
    type is value-dependent, and it is confined to the API boundary; nothing internal calls it.
    Unlike v2 it does not fall back to returning the expression when a variable is left
    unsubstituted; that is an `ArgumentError` naming the leftover.
  - `rand_poly` additionally takes an `rng` first argument, so a caller can reproduce it.
  - Calling a system as a function: `F(x)`, `F(x, p)` and `evaluate`/`jacobian` on any system,
    in `src/core/system_evaluate.jl`. These build the `FSVec` buffers the in-place `evaluate!`
    needs, so the tracker's zero-allocation path is untouched. A `System` evaluates the
    equations `polynomials(F)` reports, i.e. the normalized ones (`equation_scales` records the
    factors); that keeps `F(x)` consistent with everything else v3 computes, where v2 never
    normalizes.
  Closes `symbolic_test.jl` "Expand", "to_dict", "Horner", "Rand / dense poly", "Polynomial to
  exponents_coefficients and back", "Convert", "evaluate - Issue #500" and
  "evaluate - Issue #511". Tests: `test/symbolic_utils_test.jl`.
- [x] **`is_real` on a system** (`src/core/system_evaluate.jl`): evaluates at one random real
  point and checks the imaginary parts, correct with probability one, as in v2. Extends the
  `is_real` already defined for a `PathResult`, since v3 is one flat module where v2 has a
  `ModelKit` submodule. Closes `systems_test.jl` "is_real".
- [ ] **Public interface surface.** 22 of v2's 53 exports resolve to existing v3 internals
  that are not exported: `variables`, `parameters`, `nvariables`, `nparameters`,
  `variable_groups`, `is_homogeneous`, `is_polynomial`, `degree`, `polynomials` (v2's
  `expressions`), `support_coefficients`, `evaluate!`, `evaluate_and_jacobian!`, `taylor!`,
  `AbstractSystem`, `AbstractHomotopy`, `TaylorVector`, `TruncatedTaylorSeries`,
  `Interpreter`. Implementing a custom `AbstractSystem` currently needs qualified access to
  the interface functions.
- [x] **Ported features whose v2 test was missing:** cyclic-7 (924 solutions on both total
  degree, 5040 paths, and polyhedral, 924 paths) in `test/endgame_test.jl`; the compression
  round trip in `test/result_iterator_test.jl`, for which `total_degree_start_solutions` is now
  exported and `result_iterator` gained a `(G, F, starts)` method. The compression test records its mask from the forward run rather
  than from a track back to the start system, as v2 does: v3's start-to-target homotopy draws
  its `γ` from the algorithm's seed, so `solve(F, G, R)` and `solve(G, F, S)` are not the same
  path family and the backward track does not invert the forward one. Recording the mask
  forwards makes the round trip exact, and is what a caller would do anyway.
- [x] **Sweep collection.** All 14 of v2's systems exist in v3. `TEST_SYSTEM_COLLECTION` holds
  10 of them, `minors` having been added; `NONPOLYNOMIAL_SYSTEM_COLLECTION` holds
  `small_rational`, `sqrt_parameters` and `rigid_multiview`, whose ground truth is a
  plain-Julia `ref` rather than the `MP.differentiate` / `MP.coefficient` one the polynomial
  sweep uses.
  - `rigid_multiview` is the gradient of the two-view reprojection error of a rigid point
    pair, so its `ref` spells the chain rule out in `reprojection_gradient`. That is what the
    Jacobian and Taylor checks compare against, on top of the central differences and the
    Cauchy-integral oracle they already run.
  - `fano_quintic` is built through the `Expression` front end exactly as in v2 (`dense_poly`,
    `subs`, `to_dict`, `horner`) and stays outside both collections: its six equations are the
    coefficients of a dense quintic restricted to a line, which no `ref` can state more
    directly than the builder already does. `test/fano_quintic_test.jl` checks its shape and
    that its equations agree with the quintic restricted to the line; the 15625-path solve
    confirming the count of 2875 lives in the extensive suite (below), since it takes ~4
    minutes in a single-threaded worker against ~3 minutes for the whole suite.
  - The sweep's tape roundtrip compares `expand` of both sides. Re-executing a tape over
    `Expression` values reorders commutative operands and can fold a constant factor into a
    sum, so the reconstruction is the same function but not the same tree.

### Re-audit against v2.22.1

The list above was written against `HomotopyContinuation/`, the vendored copy of v2 in this
repository, which is **2.18.0**, as is the pin in `test/Manifest.toml` that `compare_v2_*`
loads. The four releases since it were never compared. Re-running the testset-by-testset
comparison against **2.22.1** on 2026-08-01 found the following. Everything above stays closed;
these are new.

Closed by this audit:

- [x] **`trace(::ResultIterator)`**, the coordinate-wise sum of the selected solutions,
  accumulated one path at a time so they are never all held at once. v2 reaches it through
  `bitmask_filter`, whose v3 spelling is `restrict(ri, selection(f, ri))`. Diverging paths
  contribute nothing and an empty selection sums to the empty vector rather than throwing.
  Closes the `trace` half of `result_test.jl` "Basic functionality of ResultIterator".
- [x] **`EuclideanNorm`**, alongside `InfNorm`, as a distance marker for `UniquePoints`,
  `multiplicities`, `unique_points` and monodromy. Both are metrics, so `VoronoiTree` prunes
  with either; `‖·‖₂ ≤ √d‖·‖∞` means the two disagree on a point just inside the tolerance,
  which is what the test pins.
- [x] **Weakened parity assertions tightened to v2's exact counts** in `v2_parity_test.jl`:
  `(x-10)^6` (6 paths of winding number 6, not "at least 4"), the polyhedral affine/torus
  split (8 paths → 6 solutions, 3 → 3, matching `polyhedral_test.jl` "affine + torus
  solutions"), and `paths_to_track` / `mixed_volume` (16 / 8 / 3 / 3). v3 already matched v2
  on all of these; only the assertions were loose.
- [x] **`skeel_row_scaling!` stops scaling once the Jacobian is large.** Fixed: the factors now
  depend only on the ratios between the weighted row sums, and a row below the threshold is
  scaled by the threshold bound rather than left at raw magnitude. The condition numbers the
  endgame acts on moved to the column-scaled Jacobian in the same pass, because normalizing the
  rows destroys the divergence signal at-infinity detection and u-regeneration junk removal both
  read (`01_decisions.md`, "Row scaling serves the solve"). Counts over
  `TEST_SYSTEM_COLLECTION` plus mohab, ~9000 paths: two paths change classification, both
  between at-infinity and max-steps; solution counts, the 3264 instance and the Fano quintic are
  unchanged.
- [x] **Mohab reaches v2's 693.** Two independent causes, both fixed on 2026-08-01.
  `tracking_stopped!` rejected an endpoint on an absolute `‖H(x,0)‖ > 1e-3`, which on a system
  whose terms reach 10^44 at the endpoint threw away two converged solutions; the check is now
  relative to the row scale `Σⱼ|∂Hᵢ/∂xⱼ|·|xⱼ|`. And `_cluster_solutions` merged any two
  successful endpoints within `rtol`, collapsing eight regular roots separated by 6e-9 into
  four and relabelling them multiple; proximity merging is now restricted to endpoints the
  endgame flagged singular, as v2 does by clustering only `filter(is_singular, path_results)`.
  Now 900 tracked, 693 nonsingular, 0 singular, 207 at infinity. The first four are
  seed-independent; `nat_infinity` is not, so `v2_parity_test.jl` pins a seed for it (see the
  open item below).
- [x] **`(x-10)^d` winding numbers, d = 2 and 6.** The port was the defect: v2's test writes
  `@var x` and keeps the root factored, the port wrote `@polyvar x`, and DynamicPolynomials
  expands, so the degree-6 coefficients cancel from 10^6 down to 10^-9 near x = 10 and the
  Jacobian carries about three correct digits. v2 on the expanded form is worse than v3 (6/6
  winding numbers on 1 of 10 seeds, against v3's 52 of 60). With `@var` v3 already matched v2
  at d = 6. d = 2 also needed a fix: the singular endgame declines a prediction until its
  sample condition number crosses `min_cond`, hands the path back, and v3's tracker (unlike
  v2's) then converges at t = 0, so a 1e-16 prediction was discarded for a 1e-11 endpoint
  with no winding number. `EndgameState` now keeps the best prediction it handed back and
  `tracking_stopped!` prefers it when it beats the endpoint. 80 seeds at each degree give
  v2's exact answer: d winding numbers of d, one singular solution at 10.
- [x] **Projective regeneration**, which closes both the `Regeneration` / `Decomposition`
  entry and the `intersect` one: they were one missing mode, not two bugs. Homogeneous input
  now takes `projective = is_homogeneous(F)` through the regeneration state. The seed
  subspace is linear of one dimension more (`initialize_witness_sets(...; affine = false)`),
  which makes `get_flag` take `u = 0` and keeps the whole flag linear; `expected_max_codim`
  drops by one; the deformation equation is `u^d − ℓ^d` for a generic linear form `ℓ`, so it
  stays homogeneous, and each start point takes `u = ℓ(p)·ζ` over the `d`-th roots of unity;
  the u-homotopy and the containment test run on an affine chart, the latter drawing its query
  rows through the point (as the projective `membership` already did) rather than reusing the
  structured ones. `ℓ` has real coefficients so it types like the equations it joins; the locus
  it must avoid is a proper subvariety, which meets `ℝⁿ` in measure zero. `intersect` requires
  both witness sets to agree on `projective` and draws a linear slice for a hypersurface
  argument. Now `Dict(2 => [4], 1 => [6])` and `degree.(B) == [4, 6]`, matching v2, with
  `W.projective` set and linear subspaces throughout.

  Two core bugs surfaced on the way, both latent until a chart wrapper appeared over a
  straight-line homotopy:
  - `taylor_op_pow_int` divides by the base's constant term, so `x^r` at `x = 0` returned
    `Inf * 0 = NaN` instead of finite coefficients. The projective flag puts the endpoint at
    exactly `u = 0`, so every u-homotopy path hit it: the predictor's trust region collapsed
    and the tracker could not take a step. A vanishing constant term with `r > 0` now goes
    through division-free repeated squaring. (v2 never hits this because its u-homotopy runs
    on an intermediate generic subspace and only a third stage lands on `u = 0`.)
  - `StraightLineHomotopy`'s `evaluate!` / `evaluate_and_jacobian!` / `taylor!` looped over
    `eachindex(u)`, the caller's buffer, while indexing their own `m`-row scratch. Wrapped in
    an `AffineChartHomotopy` the buffer is one row longer, so they read the scratch out of
    bounds under `@inbounds`. They now loop over the scratch.
- [x] **Mohab's at-infinity count is seed-dependent.** A few divergent paths reached
  `max_steps` before the at-infinity criterion fired, so the 207 non-solutions split between
  `PATH_AT_INFINITY` and `PATH_TERMINATED_MAX_STEPS` differently from seed to seed: over 25
  runs, 207 on 19 and 195-205 on the rest. `tracked_paths = 900` and the 693 nonsingular
  solutions never moved. Now 207 on 30 of 30 seeds.

  The confirmation stage wants the coordinate to grow by a factor of 20 since it was marked,
  and with `val_x = -1` that is a demand for `t` to fall by 20. On these paths it does not:
  the endgame spends its whole 2000-step budget moving `t` from 9.7e-3 to 1.3e-3 while
  `val_x` sits at `-1.000` with `ε∞ ≈ 1.4e-3`, `|x|` climbs 57 → 435 and the scaled condition
  number climbs 5.7e9 → 3.3e11. How far a path gets before the budget runs out is what the
  seed moves. `check_at_infinity!` now takes a `relaxed` flag that drops the growth demand to
  "grew at all", keeping the gate (a standing candidate, so `val_x + ε∞ < -val_finite_tol`
  with `ε∞ < val_at_infinity_tol`) and the condition-growth requirement, and the four
  give-up sites consult it before reporting out-of-steps. It cannot reclassify a path that
  terminates any other way, and a singular finite endpoint has `val_x = 0`, so it is not a
  candidate at all. v2 has the same criterion and the same drift.
- [x] **The splitting stage's trace test on a projective witness set.** Splitting the
  projective quartic of `nid_test.jl` "Homogeneous systems" dropped the surface on 2 of 30
  seeds, reporting degree 1 or nothing and warning that the trace test failed. Both causes are
  the affine chart the projective regime tracks on, and both are fixed; the test no longer
  pins a seed, and the split is now clean over 60 seeds isolated and 40 through the full
  `Decomposition`, warnings included.
  - The chart normal was drawn at random, independent of the points it had to normalize.
    `on_chart!` divides by `v'x`, so a normal near-orthogonal to one witness point inflated
    that representative by `1/|v'x|` (observed: 123 against ~1 for its siblings) and the loop
    lost the path. The two failing seeds were exactly the two whose worst normalized alignment
    `|v'x| / (‖v‖‖x‖)` fell below 0.03. `MonodromySolver` now takes the start solutions and
    keeps the best of up to 16 draws, stopping at an alignment of 0.2.
  - The trace loop translates the base subspace by `‖v‖ = 5` twice, which is calibrated for an
    affine base with `‖b‖ ~ 1`. A linear base has no scale of its own: the points are held on
    the chart at `‖x‖ ~ 1` while `A x = t v` forces `‖x‖ ≳ ‖t v‖`, so the two constraints pull
    apart and the equations, of degree 6 and 7 here, lose the digits the `1.0e-10`
    `trace_test_tol` needs. `_trace_step` now takes the step from the median solution norm
    when the base is linear, which is what both ends want: the trace test has to resolve the
    three slices apart (separation `‖v‖ / ‖x‖`) and the tracker has to hold accuracy through
    the growth `(‖x‖ + ‖v‖)^deg`. Affinely the fixed wide step stands, as in v2. `add_loop!`
    carries the current results in to compute it.

  A failed path is what made this silent rather than loud: it contributes no trace column, so
  the trace test reads the remaining points as a non-complete witness set, which is a statement
  about the geometry rather than about the tracking. `MonodromySolver` now counts both what
  summed into the trace and what was asked to and lost a segment (`trace_paths`,
  `trace_dropped`, `trace_complete`), and `_decompose_with_monodromy` attributes its warning
  through them: a short trace reports the failed paths, and orbits skipped for that reason are
  reported at the end rather than dropped in silence. The counts do not gate the stopping rule;
  whether a short trace should refuse to declare `SUCCESS` is a separate behavioural change.

  v2 draws its chart and its translation the same way, so neither is a v2/v3 divergence.
- [x] **`EquationSorting.RANDOMIZED`** on `Regeneration` and `Decomposition`: sort by
  decreasing degree, then replace the equations by a random upper-triangular combination, which
  leaves every `V(eqs[i:end])` unchanged while giving each equation the top degree. Rejected for
  homogeneous input, where such a combination is not homogeneous. v2 spells the option
  `sorted = :randomized`; v3 has `sorted::EquationSorting.T` with `UNSORTED` / `BY_DEGREE` /
  `RANDOMIZED`, since the codebase uses scoped enums rather than Symbol keywords.
  Closes `nid_test.jl` "randomization".
- [x] **`rand_subspace!`**, the in-place redraw of a `LinearSubspace` into given `A`/`b`
  buffers, with and without a point to pass through, `affine = false` projecting the rows off
  the point so the whole ray is contained. The returned subspace holds copies, so redrawing
  does not disturb one handed out earlier.
- [x] **`DistinctCertifiedSolutions` incremental API.** `add_solution!` now returns
  `(added, status, representative, certified_solution)`, with `representative` the index
  carried by the certificate a duplicate was matched to and `certified_solution` the
  interval midpoint when `added`. Added `stats`, `nprocessed`, `ncertified_distinct`,
  `nduplicates`, `nnotcertified` (atomic counters, so the threaded route counts correctly)
  and `merge!`, which is restricted to two accumulators of the same certificate type since
  the interval tree is typed. The bulk routes call an internal entry point returning the
  matched certificate instead of its midpoint: `solution_approximation` allocates a vector
  per call and they discard it.

  Porting v2's collision case exposed a defect in v3's dedup:
  `DistinctSolutionCertificates` keyed one certificate per `squared_distance_interval`, so
  two distinct solutions equidistant from the reference point overwrote each other and the
  distinct count was silently short. The tree now buckets per key, as v2 does. A
  `reference_point` keyword on the constructor makes the collision reachable in a test
  without the mutable-struct reassignment v2 uses.
  Closes `certification_test.jl` "DistinctCertifiedSolutions incremental API".

- [x] **Transcendental operations and real powers.** `exp`, `tan`, `asin`, `acos`, `sinh`,
  `cosh`, `tanh` and a non-integer `^` (v2's `OP_POW`), landed as one piece: they reach the same
  stack and the last two v2 test files depend on both. Eight `OpType` entries, seven of them new
  `SUnaryKind`/`EFn` kinds that the existing lowering already carries, plus a node for the power.
  - The new ops are **appended to `OpType`, out of arity order**, because
    `execute_instructions!` emits its switch in declaration order and a rarely used op ahead of
    `OP_ADD`/`OP_MUL` lengthens the chain every hot tape walks (`interpreter.jl` records the
    387ns → 710ns cyclic-7 measurement behind that ordering). Confirmed against a worktree at
    the previous commit, alternating runs of a 200k-iteration loop: cyclic-7 eval 83 to 86 ns
    before against 83 to 88 after, Jacobian 411 to 422 ns against 396 to 451, inside the noise
    either way. `@benchmarkable` alone cannot see this, since its timer quantizes to 10 ns here
    and reported a spurious 700 → 750 ns.
  - `OP_POW`'s exponent is a **tape constant, not an instruction immediate**: an `Instruction`
    input is an `Int32` slot, which `OP_POW_INT`'s integer fits and a real one does not. That
    makes it a plain arity-2 op, needing no `should_use_index_not_reference` case and no codegen
    branch, and its Taylor rule reads the exponent off a series whose higher coefficients are
    zero by construction.
  - A non-integer power is its own node (`ERPow`/`SRPow`) rather than a widened `EPow`, whose
    `exp::Int` the degree, numerator and power-collection paths all read. `_erpow` folds an
    integer-valued exponent back to `EPow`, which is what lets `_emul` tally every exponent as a
    `ComplexF64` and still keep `x * x` an integer power. Without that tally `x^1.5 * x^-1` stayed
    two factors, and CSE saw two nodes where one belongs.
  - The Taylor recurrences are `exp`, the coupled `sinh`/`cosh` pair, `tan`/`tanh` through
    `t' = (1 ± t²)a'`, `asin` through `y'·√(1-a²) = a'` with `acos` its negation past order 0,
    and `OP_POW` reusing the `OP_POW_INT` logarithmic-differentiation recurrence, which is
    generic in the exponent: only the order-0 term needed to know it is not an integer.
  - `sincos(::Expression)` was missing, so the *existing* sin/cos Taylor rules could not run over
    symbolic coefficients. That is what had made v2's `operations_test.jl` unportable; with it,
    all 32 ops are now checked against symbolic differentiation of their scalar op at N = 1, 2, 4.
  - Closes `symbolic_test.jl` "trigonometric functions", `homotopies_test.jl` "Homotopy with
    trigonometric functions", `slp_test.jl` "fractional powers" and "Evaluation of Acb with
    fractional powers", and `operations_test.jl` in full. Tests: `operations_test.jl`,
    `taylor_test.jl`, `nonpolynomial_test.jl` (a `transcendental` entry in
    `NONPOLYNOMIAL_SYSTEM_COLLECTION` puts every new op through the sweep's central-difference,
    Cauchy-integral and tape-roundtrip oracles), and the certification subpackage.

- [x] **Certification of a `ResultIterator`**, the low-memory route (Breiding, Brysiewicz and
  Johnson, arXiv:2604.16623), in `lib/.../src/iterator_certification.jl`. `certify(F, ri, p,
  IteratorCertification(), exec)` streams the iterator, certifies each successful endpoint, and
  files the enclosure of one coordinate's real part into a binary partition of the real line;
  leaves are refined until each holds at most `leaf_size_bound` enclosures, and only then is a
  leaf certified jointly and deduplicated. Two enclosures in different leaves are separated in
  that coordinate, so they are never compared. What bounds the memory is that a pass keeps one
  interval per path and drops the certificate: certificates are held one leaf at a time.
  Returns an `IteratorCertificationResult` (`bsp`, `ntracked`, `nstart_solutions`,
  `ncandidates`, the certified and distinct counts, `nleaves`, `max_leaf_size`,
  `oversized_leaves`, `unsplittable_leaves`, `nleaf_splits`), whose `BSPPartition` is iterable:
  `collect` it for the leaves, each a `(lo, hi, nenclosures, unsplittable)` named tuple.
  Closes all four `certification_test.jl` "BSP certification" testsets. Tests:
  `lib/.../test/iterator_certification_test.jl`, plus the 3264-conic instance through this route
  in `test/extensive/iterator_certification_extensive_test.jl`, which is where the pass count is
  pinned against the solution count.

  Where it departs from v2, and why:
  - **v2's memory-bounded sampling is dead code in v2's own flow, and it is what its refinement
    is built around.** `process_leaf!` reaches its reservoir sample and its two "recollect the
    full leaf" passes only when `initial_entries === nothing`, and none of its three call sites
    ever passes that. So v2 always holds every enclosure of the leaf it is refining, while every
    step of that refinement is written as though it held one sampled enclosure: it proposes a cut
    from `entries[1]` alone, then re-tracks and re-certifies the whole leaf to find out whether
    the cut crosses anything and which child each solution lands in. v3 takes a leaf's entries as
    a required argument and does all of that arithmetically.
  - **Refinement therefore costs no tracking at all**, and the whole route is one pass over the
    iterator to place the solutions plus one pass per terminal leaf to certify it (`ntracked` is
    exactly `length(ri) + ncertified` when nothing is oversized, which the extensive test pins at
    27072 + 3264). Against v2: a cut derived from one enclosure peels that enclosure off, so
    reducing a leaf of N enclosures to `leaf_size_bound` takes O(N) splits, each paying a
    re-tracking pass over the leaf, and leaves ~N terminal leaves each paying another. v3's cut
    is the most balanced one the leaf admits, so the tree is O(log) deep and the terminal leaves
    number ~N / `leaf_size_bound`. Sorting the enclosures by upper endpoint makes the candidate
    cuts the gaps after each of them, and one suffix-minimum sweep says which gaps no later
    enclosure reaches into, so the balanced valid cut costs one O(N log N) sort per split.
  - **Leaves merge only while solutions are being placed.** After the assignment pass every
    enclosure is inside its leaf by construction, and a valid cut leaves that true of both
    children, so the "this leaf is stale, coarsen and start over" path cannot arise during
    refinement. v2 carries it through every pass (`stable_leaf_entries!`, the `:restart` status),
    which it has to, because its passes are what discover a straddling enclosure.
  - **The terminal pass checks the assumption the partition rests on.** Both v2 and v3 drop
    certificates and recompute them per leaf, which is only sound if certification is
    reproducible. v3 asserts it: a leaf that comes back with a different number of certified
    enclosures, or with one reaching outside the leaf it was filed into, errors instead of
    quietly reporting a solution as distinct that was never separated. v2 checks the count only.
  - **An oversized leaf that goes uncertified warns.** Its solutions are certified but never
    deduplicated, so they are missing from the distinct counts; `oversized_leaves` records it,
    but a caller reading `ndistinct_certified` alone would take an undercount for an answer.
  - **One pass primitive, two folds.** Deciding where an enclosure sits relative to a leaf is
    arithmetic on the projected interval, so it happens in a serial fold after the pass rather
    than inside the tasks. Both remaining passes are then the same function (certify, project,
    collect in iterator order), differing only in whether the fold keeps the interval or the
    certificate. v2 writes each of its four passes twice, once serial and once threaded.
  - **Threading goes through a core primitive, not `deepcopy`.** A `ResultIterator` tracks through
    its cache's single tracker, so a concurrent pass needs one worker state per task:
    `_foreach_path(f, make_state, ri, exec)` (`src/solving/result_iterator.jl`) builds them from
    the cache's builder, as the threaded solve routes do, and hands each task its own
    `CertificationCache` from a channel pool. v2 `deepcopy`s the tracker and the system, which in
    v3 is unsafe (FunctionWrappers cache an object pointer). Payloads are sorted back into
    iterator order, so every count is identical to `Serial()`; the test asserts all fourteen.
    A cache whose builder cannot hand out independent workers (`SharedHomotopyBuilder`, which
    `result_iterator(H, starts)` gets) tracks on one task whatever `exec` asks for, and both the
    cache pool and the worker pool are sized from that same `_replay_ntasks` rather than from
    `exec`; see `01_decisions.md`. The workers are built once for the run and handed to every pass
    (`_path_workers`, then `_foreach_path(…, workers)`): one costs a cloned system evaluator plus
    an endgame tracker, and this route makes a pass per terminal leaf, so rebuilding them per pass
    was the whole of a measured 6.1× allocation on a 400-solution system.
  - **One counter, not three.** An `IteratorCertificationStats` in the context carries every count
    the run accumulates and is what the result stores; the progress meter reads it. The leaf
    recursion therefore returns only the index the walk resumes at, and the display carries no
    mirror of the counts it shows.
  - **A leaf iterator is `restrict(ri, mask)`.** v3's selection is one index domain over the start
    solutions, so the parent's selected indices are collected once into the context and a leaf's
    mask is one write per entry. v2 carries an `indices` vector and a `bitmask` and composes them
    per leaf.
  - **The partition is its sorted cut points, not an interval tree.** Leaf `i` is
    `(cuts[i], cuts[i + 1])` with parallel `counts` and `unsplittable` vectors, so the leaves are
    a contiguous cover addressed by index: `_find_leaf` is a `searchsortedlast`, a split inserts a
    cut, a merge deletes a run of them, and the walk over the partition advances by the leaf count
    the subtree left behind. An `IntervalTrees.IntervalMap` keyed by leaf (plus a duplicate sorted
    vector of the same leaves, plus a `Set` of tuple-keyed unsplittable ones) bought nothing here:
    the leaves are disjoint and ordered, so no overlap query is ever asked. `IntervalTrees` stays a
    dependency for `DistinctSolutionCertificates`, where the boxes genuinely do overlap.
  - **Per-leaf dedup builds a fresh `DistinctSolutionCertificates`** rather than `empty!`ing a
    reused accumulator (v3's `DistinctCertifiedSolutions` is immutable, and an
    `IntervalTrees.IntervalMap` has no `empty!`). One reference point serves the whole run: two
    overlapping enclosures have intersecting squared-distance intervals to any point, so the
    reference point decides only how the tree buckets, never whether two certificates match.
  - Options are one `IteratorCertification` struct embedding `Certification` and re-exposing its
    keywords, as `TotalDegree` does with `CommonOptions`, where v2 passes twelve loose keywords
    through four call layers. Dropped: v2's `check_oversized_leaves`, which only zeroes the
    reported `oversized_leaves` count and changes nothing that was computed.
  - A lazily filtered iterator is accepted (`Iterators.filter(f, ri)`) by carrying the predicate,
    which is what v2's `result_predicate` does. The eager spelling,
    `restrict(ri, selection(f, ri))`, needs nothing: it arrives as a plain `ResultIterator`.
  - **Three accessors are spelled v3's way, not v2's**, which is a name-level parity break for a
    caller porting code: v2's `npaths` is `ntracked` (core already means path trackings by that,
    and every pass adds to it), `start_iterator_length` is `nstart_solutions` (a core accessor on
    `ResultIterator` too, since the number is the solve's, and the old name read like
    `length(ri)`, which it is not for a restricted iterator), and `target_iterator_length` is
    `ncandidates` (also added to `CertificationResult`, which counted the same thing with no
    accessor). `bsp` keeps v2's name. `BSPPartition` is iterable and has `length`, where v2
    exposes the type with no way to read it; iteration rather than a `leaves` accessor keeps a
    generic word out of the export surface, which core curates.
  - `nstart_solutions` differs for a chained iterator: v3 materializes an iterator's
    successful endpoints when it is used as another route's start solutions, so this is the number
    that got through, where v2's lazy start solutions report the length of the first iterator.
  - **`Certification` on this route errors instead of `MethodError`-ing**, naming
    `IteratorCertification` and `certify(F, collect(ri), ...)`, and only a filter that bottoms out
    at a `ResultIterator` is in reach of dispatch. Certifying every successful endpoint (singular
    included) rather than `solutions(result)` is what the extensive test's singular-Jacobian
    candidates come from, and the docstring says so.

  One bug in `certify` itself surfaced, found by the 3264-conic test and latent for every caller:
  `certify_solution` formed the approximate inverse with `inv!(lu!(J))`, which throws
  `SingularException` on an exact zero pivot. Krawczyk has no operator at a singular Jacobian, so
  such a candidate is now reported uncertified instead. Nothing reached it before because the
  eager routes are handed `solutions(result)`, which is nonsingular-only; this route certifies
  every successful endpoint, and a polyhedral solve of the 3264 instance ends on endpoints at
  condition numbers past `1/eps`.

  What is and is not verified: the counts, the pass count and the leaf count are pinned by tests,
  including on the 3264-conic instance. The bounded memory is structural rather than measured. A
  pass keeps one interval per path and holds one certificate per task, and only a terminal leaf's
  certificates are alive at once, but no test asserts a peak and none is claimed.

Open, ordered by consequence:

- [ ] **`monodromy` `duplicate_check = :certified`**, which dedups by Krawczyk certificate
  instead of by distance, and the `ncertified_distinct` / `ndiscarded_uncertified` accessors on
  `MonodromyResult`. Awkward in v3's layout: monodromy is in core and `certify` is in the
  certification subpackage, so core cannot call it. Blocks `monodromy_test.jl` "certified
  duplicate checks".

Deliberately not ported from v2.22: `model_kit/compiled_cache_test.jl`, which exercises the
locks around v2's global `TSYSTEM_TABLE` / `THOMOTOPY_TABLE`. v3 keeps no global compile table
(`compile` is a per-`System` setting), so there is nothing to make thread-safe.

### Extensive suite (`make test-extensive`)

`test/extensive/` holds the two large solves, ported from v2's `test/extensive/extensive_test.jl`
(which v2's own `runtests.jl` has commented out, so those expectations were never verified). It has
its own environment because it certifies, and a plain `runtests.jl` rather than ParallelTestRunner,
whose workers are single-threaded while these solves want every core. `test/runtests.jl` filters the
directory out of discovery, so `make test` does not run it. Takes ~4 minutes.

- **Lines on a quintic surface.** Total degree, 15625 paths, 2875 solutions, all 2875 distinct
  certified; polyhedral, 6725 paths, 2875. The parameter draw is pinned: one draw in five gives
  2874 across every `γ` and both start systems, and no 2875th root is recoverable by Newton from
  any discarded endpoint, so that count is a property of the instance rather than of the tracking.
- **3264 conics tangent to five conics.** Polyhedral at a generic target, 27072 paths, 3264
  solutions, all 3264 distinct certified. At five real conics both routes also reach all 3264, and
  certification is what shows it: tracking the generic solutions there gives 3264 successful
  endpoints in bijection with the known real solution set (no collision, nothing spurious), all
  3264 certified distinct and real. The direct polyhedral solve returns those 3264 plus 14
  endpoints at condition number 1e17-4.6e18 that are not solutions, and certification rejects
  exactly those 14.

  `nsolutions` reported 3259 until `sing_cond` was corrected. Five of the 3264 conics have a
  scaled Jacobian condition number between 1.03e14 and 1.51e14, which the old `1e14` default
  (v2's, still) put on the singular side, and `nsolutions` is nonsingular-only. All five are
  regular: multiplicity 1, no winding number estimated, accuracy ~1e-16, and Krawczyk certifies
  them. Both routes flag the same five roots along disjoint sets of paths, so the conditioning
  belongs to the instance rather than to the tracking. The default is now `inv(eps(Float64))`,
  derived from the working precision rather than fitted (see `01_decisions.md`), and both routes
  report 3264 with the 14 excess endpoints still excluded. This is a deliberate deviation from v2,
  which reports ~3259 here; the four `nsingular` parity assertions in
  `compare_v2_solve_counts_test.jl` are unaffected, because every multiple root in them is found
  by winding number at condition numbers orders of magnitude below either threshold.

Deliberately not ported, since they are v2 architecture rather than features: `MixedSystem`
and `MixedHomotopy`, `CompiledSystem`/`InterpretedSystem` as user-facing types,
`set_default_compile` (v3 takes `compile` per `System` and keeps no global mutable state),
`optimize` (CSE runs at construction), the `ModelKit` submodule, the TreeViews and Juno
`show` methods, and `rand_unitary_matrix` (`RandomizedSystem` uses an identity block plus a
random fold). v2's `get_num_den` is v3's exported `num_den`.


## Open Items

### Architecture debt

- Two solution-dedup mechanisms: `Result` clustering (`_cluster_solutions`, union-find) and
  `UniquePoints`/`VoronoiTree` from the monodromy port. The two now meet in `_orbit_merge!`,
  which indexes proximity-cluster representatives in a `VoronoiTree` and walks their orbit
  images itself to get group-action awareness. Consolidating the proximity sweep itself onto the VoronoiTree would likely speed up
  large results, but changes solution-count semantics (transitive closure vs first-match), so it
  needs its own tests (see `01_decisions.md`, "Two solution-dedup mechanisms exist")
- Two threading coordinators: OhMyThreads executor for `solve()`, Channel job queue for threaded
  monodromy. Justified by the dynamic monodromy workload; revisit only if a third dynamic
  consumer appears
- **Surviving `Union{Nothing, T}` struct fields, audited.** Eight remain, all of them a genuinely
  absent value rather than a flag state, so they are deliberate and an audit should not churn on
  them:
  - `MonodromyOptions.target_solutions_count`, `.timeout`, `.min_solutions`: the stopping
    heuristic was not requested, so there is no count, deadline or minimum to compare against.
  - `MonodromyOptions.unique_points_rtol`: the default is `uniqueness_rtol(res)`, which needs the
    endpoint, so it cannot be resolved at construction. Its sibling `unique_points_atol` could
    (the default was the constant `1.0e-14`) and is now a plain `Float64`.
  - `MonodromyJobResult.result`, `.trace`: a path that failed has no `PathResult`, and a loop that
    failed on a later segment contributed no trace columns. The two are independent.
  - `MonodromyResult.trace`: no trace test was run.
  - `TraceColumns.columns`: the loop never reached its halfway subspace.

  What was removed rather than kept, and the spelling that replaced it: a tri-state flag became a
  `Bool` with a computed keyword default (`intrinsic::Bool = _default_intrinsic(L_start)`,
  `triangle_inequality::Bool = satisfies_triangle_inequality(distance)`); an unknown-until-computed
  state became an enum (`WitnessSet.irreducibility::Irreducibility.T`, replacing a
  `Union{Nothing, Bool}` field whose accessor returned `Union{Symbol, Bool}`); a field absent only
  in one construction mode became an empty sentinel of the concrete type
  (`GrassmannianGeodesic.B_start::Matrix{ComplexF64}`, empty for an intrinsic geodesic); and a
  "use the default constant" sentinel became the constant. Where the absent case is fixed at
  construction, a type parameter keeps each instance concrete instead:
  `MonodromyOptions{D, GA <: Union{Nothing, GroupActions}, …}` with `group_actions::GA`.
  A nullable *seed* was the remaining case and was not in the legitimate class:
  `NumericalIrreducibleDecomposition.seed === nothing` meant "no seed was recorded", which is what
  made an NID result unreproducible. It is now a plain `UInt32`, and every route that consumes
  randomness now derives all of it from its `seed` (see `01_decisions.md`, "Task-local RNG for
  reproducibility").
- The eight builder structs each repeat `tracker_options` / `endgame_options`. A shared
  `BuilderOptions` field would remove sixteen declarations and add an indirection at
  every use, so it was left alone; the construction *tail* they all shared is now
  `_endgame_tracker` (see `01_decisions.md`)
- Em dashes are used as sentence pauses in comments across the older `src/` files,
  against the repo's writing rule. New and touched code is clean; a global sweep
  would be pure churn on files nothing else is changing
- The API surface is declared only by `export`, so there is no way to tell a documented
  entry point from an implementation detail that happens to be reachable. Names that are
  public but should not be dumped into the caller's namespace (option structs, result
  accessors, the `AbstractSystem`/`AbstractHomotopy` interface methods a user overloads)
  currently have to choose between being exported or looking private. Adopt SciMLPublic
  and mark every intended entry point with `@public`, keeping `export` for the small set
  users want unqualified (`solve`, `System`, `@polyvar`, `@var`, ...). That gives three
  explicit tiers, exported, public-not-exported, and private, lets Aqua and
  ExplicitImports check the boundary, and makes `Base.ispublic` the single source of
  truth for what downstream code may depend on. 39 `export` lines across the main module
  today, plus the certification subpackage.
- `Compiler.inferiterate_2arg` re-inference costs ~245ms of every first call. It
  survives a build that never loads OhMyThreads, so the InitialValues invalidations are
  not the cause and the trigger is still unattributed (`01_decisions.md`, "Dependency
  invalidations are largely inert")

Audited 2026-07-28 over `src/`, `ext/` and `lib/`. Findings below, most consequential first.

- **Six near-identical track-all-paths loops.** `_solve_total_degree_serial` /
  `_solve_total_degree_threaded` (`solve.jl:211,244`), `_track_all_serial` /
  `_track_all_threaded` (`subspace_solve.jl:58,74`) and `_solve_polyhedral_serial` /
  `_solve_polyhedral_threaded` (`polyhedral.jl:726,761`) are the same loop three times over.
  The two threaded ones differ on exactly one line, the last argument to `_finalize_result`
  (`cache.excess_checker` against `nothing`); everything else, including the
  `Threads.Atomic` counter, the `ReentrantLock` and the `@tasks`/`@local` preamble, is
  character-identical. The polyhedral pair differs only in the per-path call. The serial
  three differ further by accident rather than by intent: they `push!` onto a `sizehint!`ed
  `PathResult[]` where the threaded three preallocate `Vector{PathResult}(undef, n)`.
  The distributed extension already factored exactly this shape, into `_distributed_map`
  plus the `TrackWork`/`PolyhedralWork`/`SweepWork` callables
  (`ext/.../distributed_map.jl`, `ext/.../solve.jl:9,17,43`), so the abstraction exists and
  core is the side that did not adopt it. Cost: any change to progress accounting, result
  ordering or finalization has to land in three files and be checked in six places.
- **The progress/inference ritual is repeated six times and is undocumented.** Each
  `CommonSolve.solve!` picks between a `_with_progress` and a `_without_progress`
  `@noinline` trampoline, launders the choice through `Base.inferencebarrier`, and calls the
  shared `@nospecialize`d `_dispatch_solve_policy` (`solve.jl:182,190,231`,
  `subspace_solve.jl:99,115`, `polyhedral.jl:666,748`). That is 12 trampolines plus 6
  five-line `solve!` bodies. The device is sound (it splits the `Union{Nothing,
  ProgressMeter.Progress}` so each loop body specializes on one arm instead of branching on
  a union at runtime) but `01_decisions.md` documents every other `inferencebarrier` use and
  not this one, so a new route reproduces it by copying a neighbour rather than by reading a
  rationale, and nothing states what breaks if it is dropped.
- **`DistributedExecutor` reachability past the initial solve.** Every route now takes a
  positional `exec::AbstractExecutor` and `threading::Bool` is gone, so witness sets, NID and
  regeneration accept a `DistributedExecutor` and it reaches their initial solve and their
  monodromy stages. It does not yet reach the witness *moves*, the u-homotopy intersection or
  `membership`: `_move_witness_points` loops serially and builds its tracker inline,
  `_threaded_intersection!` builds its tracker inside `@tasks` with no builder type, and
  `membership` uses a bespoke `MembershipState`. None of the eleven builders in `builder.jl` is
  involved, so each needs a builder, a work unit in `ext/.../solve.jl` and serialization.
  `membership` additionally hard-codes `nt = Threads.nthreads()` rather than reading a task
  count off its executor, so `Threaded(2)` cannot bound it.
- **Four progress-bar idioms.** `progress.jl` offers `make_progress`/`update_progress!` and
  `make_many_progress`/`update_many_progress!`; monodromy has its own `ProgressUnknown`
  pair (`monodromy.jl:1457,1269`); `membership` reuses `make_progress` but drives it with
  bare `ProgressMeter.next!` under its own lock (`witness_set.jl:687-705`); and
  `regeneration.jl:279` constructs a `ProgressMeter.Progress` inline. The inline one skips
  the `progress.tlast += delay` suppression that `make_progress` applies, so a regeneration
  that finishes inside 0.3 s prints a bar where an equally fast `solve` stays silent. Four
  spellings of "count something and draw a bar", and the odd one out is a visible behaviour
  difference rather than a style difference.
- **Wrapper systems satisfy a ten-method contract that is nowhere declared.** Each of the
  six `AbstractSystem` wrappers (`TotalDegreeStartSystem`, `RandomizedSystem`,
  `_StartPairSystem`, `AffineChartSystem`, `SlicedSystem`, `_ComposedSystem`) implements
  three `evaluate!` variants (F64/F64, DF64-in/F64-out, DF64/DF64), one
  `evaluate_and_jacobian!` and six `taylor!` variants (orders 1 to 3, scalar and
  `TaylorVector` parameters). `abstract_types.jl` declares three bare `function ... end`
  stubs and documents one signature each, so the actual required set is discoverable only by
  reading an existing wrapper, and a wrapper that omits the DF64 `evaluate!` fails with a
  `MethodError` inside `extended_prec_refinement_step!` near a singular solution rather than
  at construction. The two spellings compound it: `RandomizedSystem`,
  `TotalDegreeStartSystem`, `_StartPairSystem` and `_ComposedSystem` write the six `taylor!`
  methods out one per `(K, N)` pair, while `SlicedSystem` and `AffineChartSystem` collapse
  them to two methods generic in `K`. Both are defensible (the explicit form pins the valid
  `(K, N)` pairs as the repo's signature rule prefers, the generic form removes 16
  declarations) but nothing records which is the house style, and the generic form silently
  accepts `(Val{7}, TaylorVector{2})`.
- **Two wrappers append the projective chart row.** `AffineChartSystem` is `SlicedSystem`
  with zero linear rows: both append `v'x - 1`, both get its Jacobian row and its order-K
  Taylor coefficient from the shared `evaluate_chart` / `_chart_taylor_row`
  (`affine_chart.jl:81,135`), and neither wraps the other. Which one a route picks is per
  call site and unexplained: `builder.jl:193` and `witness_set.jl:572` build an
  `AffineChartSystem`, `slice.jl:182` builds a `SlicedSystem` carrying a chart, and the plain
  projective routes reach the latter by slicing with `_full_subspace` rather than adding a
  third spelling. A new projective route still has to guess, and the chart-row Taylor pitfall
  documented in `01_decisions.md` ("Appended linear rows") has two places to get wrong.
- **The certification subpackage depends on 36 core names, 23 of them unexported.** Its
  `using HomotopyContinuationNext:` list
  (`lib/HomotopyContinuationNextCertification/src/HomotopyContinuationNextCertification.jl:26`)
  reaches into the tape compiler and interpreter internals: `_EXEC_INSTRUCTION_SPECS`,
  `nested_ifs`, `exec_instruction_storage`, `_compile_exec_instructions`,
  `should_use_index_not_reference`, `instruction_op`, `instruction_output`, `op_call`,
  `arity`, `ExecInstructionT`, plus `_newton`, `_clone_system_evaluator` and
  `make_progress`. Only `System`, `Expression`, `solution`, `Result`, `PathResult` and
  `MonodromyResult` are public. Because the subpackage is a path dependency, no compat bound
  can express any of this, so renaming an interpreter internal breaks it with no warning and
  only `make test-cert` notices. Either a named internal surface (an `AcbBackend` seam the
  Acb interpreter builds against) or an explicit note in `interpreter.jl` /
  `instruction_sequence.jl` that these names are load-bearing outside core.
- **The equation front-ends have no named interface.** Eight functions carry one
  `MP.AbstractPolynomialLike` method and one `Expression` method, scattered across three
  files: `_regeneration_equations`, `_check_regeneration_input`, `_u_degree`,
  `_u_start_equation`, `_numerator_system`, `_rename_variables`
  (`regeneration.jl:333-436`), `_fix_parameters` (`slice.jl:11,45`) and
  `support_coefficients` (`support.jl:13`), guarded by the two-method predicate
  `_is_expression_front_end`. Dispatching on the equation type is the right call and is
  recorded as such above, but nothing lists the set, so adding a third front-end means
  finding the members by grep and discovering an omission at runtime on whichever route
  rewrites equations rather than only evaluating them.
- **`solving/support.jl` is a core-layer file under the solving directory.** It is
  `include`d between `core/system_evaluator.jl` and `core/system.jl`
  (`HomotopyContinuationNext.jl:107`) because `System` construction needs
  `support_coefficients`, even though `Polyhedral` is its only consumer. The include order
  already documents the real layer; the path does not. Same shape for
  `TotalDegreeStartSystem`, an `AbstractSystem` living in `solving/total_degree.jl` and
  consumed by `builder.jl`.
- **`show` conventions differ across the result types.** `Result` (`result.jl:368`),
  `MonodromyResult`, `NumericalIrreducibleDecomposition` and `CertificationResult` define
  multi-line two-argument `show`, which is the method Julia uses inside containers and for
  `repr`. `PathResult` defines only a `MIME"text/plain"` method
  (`path_result.jl:193`), so it has no two-argument method at all. Both directions misprint,
  and both shapes are public return types: sweeps return `Vector{Result}`
  (`sweep.jl:77`) and `path_results(r)` returns `Vector{PathResult}`. Measured:
  `show(stdout, [r, r])` prints `Result[Result with 2 tracked paths\n • 2 non-singular
  solutions (2 real)\n, Result with 2 ...]`, and `show(stdout, path_results(r)[1:1])` prints
  the full 19-field struct dump. The fix is one method each: a one-line two-argument `show`
  for both, with the multi-line body moved to `MIME"text/plain"` on `Result` and friends.
- **`WitnessPoints` is the one `mutable struct` missing `const` on its fixed fields.**
  `L` and `Lᵤ` are never reassigned (only `R` is, at `regeneration.jl:725,829`) and the
  struct carries no note on why it is mutable (`regeneration.jl:26`). Two `const` keywords
  and a sentence. Every other mutable struct in the tree either consts its fixed fields or
  states why it cannot; `AcbCertCache` is the model, documenting that its buffers are
  reassigned wholesale by `set_arb_precision!`.
- **Two stale `kwargs...` docstrings and one real splat.** `multiplicities` and
  `unique_points` document "The remaining `kwargs` are passed to `UniquePoints`"
  (`unique_points.jl:113,170`) but their signatures enumerate every keyword explicitly, as
  the repo requires, so the sentence promises forwarding that does not happen.
  `DistinctSolutionCertificates(dim::Integer; kwargs...)` (`certification.jl:422`) is the
  one surviving splat, one hop forwarding one keyword.

Not found, checked: `Any`-typed struct fields (none; the five `::Any` occurrences are
deliberate dispatch tags on parameter-Taylor builders and unsupported-route stubs),
`FixedSizeVector`/`FixedSizeMatrix` in struct fields (none), package-global mutable state
(only the two `const Dict` subscript tables in `expression.jl`), `deepcopy` on a
FunctionWrapper path (none; the one mention is a comment forbidding it), and TODO/FIXME/HACK
markers (one, `polyhedral.jl:355`, an upstream request rather than local debt).

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
