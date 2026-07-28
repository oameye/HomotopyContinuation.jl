# Status

Last updated: 2026-07-26.

**Reproduce:**
- `make benchmark` — steady-state timings
- `make compare` — v3/v2 ratios
- `julia --project=benchmark benchmark/compare/tracking.jl` — end-to-end solve comparison
- `make test` runs the core suite (53 files, parallel via ParallelTestRunner) then the certification subpackage; `make test-cert` runs only the latter

## Summary

Total-degree solving is 1.8–3.6x faster than v2; polyhedral matches within noise (0.95–1.01x).
Cold load + construction + first solve is ~10.6s versus v2's 45s, with no precompile workload.
Endgame is at v2 result parity on v2's default parameters.

At v2 parity: threading, overdetermined systems, parameter homotopies, monodromy (group actions,
linear subspaces, trace test), certification (Krawczyk with Arb fallback, in the separate
`lib/HomotopyContinuationNextCertification` subpackage), and witness sets / NID (`witness_set`,
`trace_test`, `membership`, `regeneration`, `decompose`, `nid`, including projective,
zero-dimensional, parametric, and rational cases).

No remaining gaps against v2 on the executor axis: `Serial`, `Threaded` and
`DistributedExecutor` cover single-task, multi-task and multi-process tracking.

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
    front-end: `_fix_parameters` (used by sliced solves and witness sets) substitutes through
    `subs` instead of `MP.subs`, and `support_coefficients` recovers the polyhedral support by
    expanding the expression tree into `exponent vector => coefficient`, since the front-end
    keeps products and powers unexpanded. Both agree term for term with the same system built
    through DynamicPolynomials.
  - `DoubleF64` gained `exp`, `sin`, `cos`, `sincos`, `sinh`, `cosh` (~32 digits); the
    `ComplexDF64` versions follow from the generic `Base` complex methods. `sin`/`cos` switch
    to angle addition over the two limbs past `|a| = 2⁵³`, where double-double reduction modulo
    `2π` drops below Float64 accuracy, and `sinh`/`cosh` take a separate branch past `|a| = 40`,
    where `e ± 1/e` would be NaN.
  - Rectangular `Interval`/`IComplex` arithmetic gained `sqrt`, `sin`, `cos` (plus `sinh`
    and `cosh` as building blocks), so every tape stays on the Float64 Krawczyk path and
    escalates to Arb only when the test genuinely fails. v2 has none of these and routes
    such systems to Arb unconditionally. The complex `sqrt` takes whichever of
    `|z| ± Re z` does not cancel and recovers the other root from `2uv = Im z`; the naive
    form inflates the enclosure of a box around a real solution to the square root of its
    width. A box meeting the branch cut returns empty, which falls through to Arb.
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
  covering every route except monodromy (total degree, polyhedral, parameter homotopy,
  subspace moves in both regimes, and both sweep kinds). Dynamic batching over a
  `RemoteChannel`, each process internally threaded, results stored by global path index so
  that at a fixed seed they are bit-identical to `Serial()` at any batch size.
  `Serialization` methods for `System`, `_SupportSystem` and `CompositionSystem` ship builders
  as plain data
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
  zero-dimensional varieties, parametric (`target_parameters`) witness sets, rational input to
  `regeneration`/`nid`/`intersect`, and threaded membership and intersection.
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
    parameter-free system, so no parameter plumbing reaches moves/trace/membership/decompose.
  - Threaded intersection clones evaluators instead of `deepcopy`ing trackers (unsafe with
    FunctionWrappers) and pushes endpoints in serial order.
  - `membership` is bit-identical across threading modes: all randomness is drawn from the
    global RNG in the driver, so both modes leave the global RNG in the same state. The query
    subspace direction is genuinely random per query, unlike v2's fixed axis-aligned frame.
  - Adds the `weighted_normal` monodromy sampler (v2 has it only for regen/decompose).
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
  (`solve` from start solutions, `solve` by total degree, `monodromy_solve`, `newton`,
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

### Not Done

- [ ] **Distributed monodromy**: `DistributedExecutor` covers every route except monodromy,
  whose shared `UniquePoints` dedup set and trace matrix would have to become cross-process
- [ ] **Cache the Grassmannian geodesic across retargets**: `_set_subspaces!` rebuilds
  `GrassmannianGeodesic` on every retarget, so a subspace sweep pays one SVD per target and a
  sweep that revisits a target pays it again. The geodesic depends only on `(start, target)`, so
  memoizing on that key would make the repeat free. v2 keys a module-global LRU of size 128 on
  the pair; a per-homotopy cache avoids the shared mutable global that v2 has to lock
- [ ] Compile-mode benchmark, v2 side: v3 `COMPILED_ALL` vs v2 `:all`, plus fresh-session
  first-solve per v3 default candidate (the v3-only matrix is measured; see `04_compile_modes.md`)
- [ ] Benchmark CI


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
