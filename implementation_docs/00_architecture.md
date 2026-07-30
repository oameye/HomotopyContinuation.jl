# Architecture

## Pipeline

```
@polyvar x y; F = System([x^2+y, x*y-1])   @var x a; F = System([sqrt(a)*x^2-1])
         │                                          │
         ▼                                          ▼
   poly_to_sexpr() + MP.differentiate()     expression_to_sexpr() + differentiate()
         │                                          │
         └──────────────────┬───────────────────────┘
                            ▼
   SExpr trees (@data SExpr — single concrete type SExprT)
         │  cse() = opt_cse() + tree_cse()
         ▼
   CSE output: (replacements, reduced_exprs)
         │  compile_to_instructions()
         ▼
   InstructionSequence (tape layout: constants|params|vars|scratch|assignments)
         │  Interpreter(V, seq) + compile to ExecInstruction variants
         ▼
   execute!(u, I, x) / execute!(u, U, I, x) / execute_taylor!(...)
         │                                         │
         ├── CompileMode.INTERPRETED ──────────────┤
         │   (all via interpreter)                  │
         │                                          │
         ├── CompileMode.COMPILED ─────────────────┤
         │   (RGF eval+jac, interpreter Taylor)     │
         │                                          │
         └── CompileMode.COMPILED_ALL ─────────────┘
             (RGF for eval, jac, and Taylor)
         │
         ▼
   FunctionWrapper → SystemEvaluator (concrete, no type params)
         │
         ▼
   Homotopy (StraightLine / Coefficient / Toric) → HomotopyEvaluator (concrete)
         │
         ▼
   Tracker (monomorphic) → Predictor · Newton · Jacobian · WeightedNorm
         │
         ▼
   EndgameTracker → Valuation · singular endgame · at-infinity detection
         │
         ▼
   solve(F, alg, exec) → init(F, alg, exec) → SolveCache{E,B} / PolyhedralSolveCache{E,B,S}
         │
         ├── Serial()   → solve!(cache::Cache{Serial})   single-task loop
         └── Threaded(n) → solve!(cache::Cache{Threaded}) @tasks/@local via OhMyThreads
                           Builder() → WorkerState (fresh evaluator + tracker per task)
         │
         ▼
   Result{Vector{PathResult}} + clustering

   Parameter families (optional):
   monodromy_solve(F; ...) → MonodromyLoop (ParameterHomotopy round trips)
         │  serial loop, Channel-based threaded coordinator, or driver-held
         │  queue over RemoteChannels (DistributedExecutor)
         ▼
   UniquePoints (VoronoiTree + GroupActions) dedup → MonodromyResult
         │
         ▼
   verify_solution_completeness (trace test) / permutations / trace
```

## Package Layout

```
src/                                         ~17,450 lines total
├── HomotopyContinuationNext.jl      (120)   Main module, exports, type aliases
├── utils.jl                         (189)   SegmentStepper, _stable_sort!, fast_abs
├── primitives/
│   ├── double_f64.jl                (795)   DoubleF64, ComplexDF64, exp/sin/cos/sinh/cosh
│   ├── norms.jl                     (201)   WeightedNorm (infinity norm only)
│   └── linear_algebra.jl            (1116)  MatrixWorkspace, LU, QR, condition est.
├── model_kit/
│   ├── operations.jl                (157)   OpType enum (25 ops), op_* scalar functions
│   ├── sexpr.jl                     (429)   Moshi @data SExpr ADT, canonicalization, poly_to_sexpr
│   ├── cse.jl                       (591)   SymEngine CSE port: opt_cse + tree_cse
│   ├── tape_compiler.jl             (691)   SExpr → InstructionSequence, fusion, register alloc
│   ├── instruction_sequence.jl      (299)   Instruction, DAG reorder, linear-scan register alloc
│   ├── interpreter.jl               (465)   ExecInstruction variants, execute!, execute_taylor!
│   ├── codegen.jl                   (300)   RuntimeGeneratedFunctions for COMPILED/COMPILED_ALL
│   ├── taylor.jl                    (508)   TruncatedTaylorSeries, TaylorVector, taylor_op_*
│   ├── expression.jl                (947)   Expression ADT, @var, arithmetic, differentiate, subs, det, conj
│   ├── expression_compiler.jl       (51)    Expression lowering path (expr→SExpr→CSE→tape)
│   ├── symbolic_polynomial_compiler.jl (62) Active MP lowering path (poly→SExpr→CSE→tape)
│   ├── polynomial_compiler.jl       (140)   Experimental direct path (NOT default, gated by TODO)
│   └── polynomial_input.jl          (72)    Variable discovery, System construction orchestrator
├── core/
│   ├── abstract_types.jl            (76)    AbstractSystem, AbstractHomotopy interfaces
│   ├── system.jl                    (267)   System type (caches compiled interpreters)
│   ├── system_evaluator.jl          (184)   FunctionWrapper wrapper for AbstractSystem
│   ├── homotopy_evaluator.jl        (162)   FunctionWrapper wrapper for AbstractHomotopy
│   ├── straight_line_homotopy.jl    (191)   γ·t·G(x) + (1-t)·F(x)
│   ├── coefficient_homotopy.jl      (180)   Coefficient interpolation (polyhedral phase 2)
│   ├── parameter_homotopy.jl        (219)   H(x,t) = F(x; t·p₁ + (1-t)·p₀), retargetable
│   ├── toric_homotopy.jl            (400)   Toric deformation (polyhedral phase 1)
│   ├── linear_subspace.jl           (562)   LinearSubspace, intrinsic/extrinsic descriptions, geodesics
│   ├── subspace_homotopies.jl       (747)   Intrinsic/ExtrinsicSubspaceHomotopy, Grassmannian geodesic
│   ├── affine_chart.jl              (274)   AffineChartSystem/Homotopy, on_affine_chart
│   ├── randomized_system.jl         (201)   Square-up for overdetermined systems
│   ├── composition_system.jl        (469)   `G ∘ F` wrapper and user-facing CompositionSystem
│   └── start_pair_system.jl         (196)   `F(x; p)` as a system in the joint unknown `[x; p]`
├── tracking/
│   ├── tracker.jl                   (620)   Path tracker, adaptive step control, warm start
│   ├── predictor.jl                 (351)   Pade (2,1), Taylor coefficients, trust region
│   ├── newton_corrector.jl          (424)   Alpha-theory Newton, DoubleF64 refinement
│   ├── newton.jl                    (249)   Standalone newton(F, x0) API
│   ├── valuation.jl                 (224)   Puiseux series valuation for endgame detection
│   └── endgame_tracker.jl           (960)   Endgame state machine, singular endpoint handling
└── solving/
    ├── executor.jl                  (140)   AbstractExecutor, Serial, Threaded, DistributedExecutor
    ├── worker_state.jl              (117)   TrackingWorkerState, PolyhedralWorkerState, _clone_system_evaluator
    ├── builder.jl                   (266)   StraightLineBuilder, ParameterBuilder, subspace builders, PolyhedralBuilder
    ├── solve.jl                     (206)   solve() API, CommonSolve integration, serial/threaded dispatch
    ├── total_degree.jl              (167)   Bezout start system
    ├── polyhedral.jl                (513)   Two-phase: toric + coefficient, MixedSubdivisions
    ├── binomial_system.jl           (544)   HNF binomial solver
    ├── excess_solution.jl           (156)   Excess-solution filtering for overdetermined square-up
    ├── path_result.jl               (395)   PathResult (immutable, enum codes)
    ├── result.jl                    (420)   Result, clustering, solutions(), real_solutions()
    ├── progress.jl                  (55)    ProgressMeter integration
    ├── voronoi_tree.jl              (275)   VoronoiTree nearest-point search structure
    ├── unique_points.jl             (205)   UniquePoints, multiplicities, unique_points
    ├── group_actions.jl             (116)   GroupActions, SymmetricGroup
    ├── monodromy.jl                 (2335)  monodromy_solve, trace test, verify_solution_completeness
    └── support.jl                   (191)   Extract support/coefficients from MP or Expression
```

### Certification subpackage

Solution certification lives in a **separate package** under `lib/`, not in core:

```
lib/HomotopyContinuationNextCertification/
├── src/
│   ├── interval_arithmetic.jl      Interval / IComplex / IComplexF64, inf_norm_bound, sqr,
│   │                               sqrt / sin / cos / sinh / cosh enclosures
│   ├── interval_arblib.jl          Acb ↔ Interval bridge (Arblib)
│   ├── acb_interpreter.jl          AcbInterpreter: arbitrary-precision in-place tape interpreter
│   ├── certification.jl            certify(), Krawczyk operator, certificate types, accumulator
│   └── certification_arb.jl        Arb (extended-precision) Krawczyk fallback
└── test/                           certification, interval, export-surface, and quality suites
```

It depends on core plus **Arblib** (a heavy binary dependency) and IntervalTrees.
Certification is the *only* consumer of Arblib, so keeping it in a subpackage is
what keeps core free of Arblib and its load-time and invalidation cost (core
load dropped from about 1.6s to about 0.77s). A package extension cannot be used
here because every certificate type embeds an `AcbMatrix` field, and Julia
extensions cannot define or export new types into the parent module. To certify:

```julia
using HomotopyContinuationNext, HomotopyContinuationNextCertification
certify(F, solutions)
```

The subpackage reaches into core internals (the tape interpreter, `System`,
`_newton`, `_clone_system_evaluator`, the `ExecInstruction` ADT) via explicit
qualified imports; it extends core `is_real`/`solutions` with certificate methods.

## Key Types

### CompileMode

```julia
@enumx CompileMode::Int8 begin
    INTERPRETED      # Tape-based interpreter for all operations (default)
    COMPILED         # RGF eval+jac, interpreter Taylor (3–6x kernel speedup)
    COMPILED_ALL     # RGF for eval, jac, and Taylor (additional ~1.3x Taylor speedup)
end
```

### Type Firewall

Every `AbstractSystem`/`AbstractHomotopy` is wrapped via `FunctionWrapper` into concrete evaluators. The tracker is monomorphic — compiled once, reused for all systems.

```julia
SystemEvaluator    # wraps eval/jac/Taylor FunctionWrappers plus a lazy DF64 pair,
                   #   taylor 1/2/3 (scalar param), taylor 1/2/3 (TaylorVector param)
                   #   + size, nparameters
HomotopyEvaluator  # wraps 10 FunctionWrappers: eval, eval_df64, eval_jac,
                   #   taylor 1/2/3, set_solution, get_solution,
                   #   start_parameters, target_parameters + size
```

All FW signatures use `FSVec{T}`/`FSMat{T}` (concrete FixedSizeArray aliases). The abstract interface uses `AbstractVector`/`AbstractMatrix` — Julia dispatch resolves automatically.

### System

```julia
struct System{P, V, M, S}
    polys::FSVec{P}                    # original MP polynomials
    parameters::FSVec{V}               # parameter variables
    variables::FSVec{V}                # decision variables
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    equation_scales::Vector{Float64}   # factor each input equation was divided by
    nvars::Int; nparams::Int
    variable_groups::Vector{Vector{Int}}
    group_degrees::Matrix{Int}         # degree per group per equation; empty when ungrouped
    is_homogeneous::Bool               # per group when grouped
    _support_coefficients::Base.RefValue{SupportCoefficients} # filled on first use
    # GC roots — interpreters must stay alive for FunctionWrapper closures
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_jac::Interpreter{Vector{ComplexF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}}
    compile_mode::CompileMode.T        # needed by _clone_system_evaluator for threading
end
```

### CompositionSystem

`compose(G, F)` (infix `G ∘ F`) builds the system `x ↦ G(F(x; p); p)` without
rebuilding equations. The user-facing `CompositionSystem` stores one thunk per
stage (`FunctionWrapper{SystemEvaluator, Tuple{}}` around
`_clone_system_evaluator`), innermost first, so a chain of any depth is one
concrete type and every worker clones the whole chain. Folding the thunks
produces nested `_ComposedSystem <: AbstractSystem` wrappers, each erased by a
`SystemEvaluator`: evaluation runs the stages inside out, the Jacobian is one
`mul!` of the two stage Jacobians, and the order-K Taylor coefficient comes from
running `G` on the series of `F` filled from order 0 up (K + 1 runs of the inner
tape, since a `SystemEvaluator` returns one order per call).

Stages are composed after `System`'s per-equation normalization, which changes
an inner stage as a map, so the fold multiplies each inner stage's
`equation_scales` back into its output.

`SystemLike = Union{System, CompositionSystem}` marks the routes that never
rebuild equations: parameter homotopies, `monodromy_solve`, `newton`, and total
degree, which needs only the folded degrees, the evaluator and the clone.
Polyhedral and witness sets need the composed monomials and reach them through
`System(C::CompositionSystem)`, which substitutes the stages into each other
(undoing every inner stage's scaling symbolically) and pays the differentiation
and CSE cost the composition exists to avoid. Each stage therefore keeps a
second thunk, `StageEquations`, that converts its equations to `Expression`s on
demand.

Degrees and homogeneity are folded from the stages rather than the composed
equations: `deg(gⱼ ∘ F)` is the degree of `gⱼ` in the weights `deg(fᵢ)`, and
`gⱼ ∘ F` is homogeneous when `gⱼ` is homogeneous in those weights and every
`fᵢ` is. This matters because `MonodromySolver` picks an affine chart off
`is_homogeneous`, and a false negative there tracks a projective problem in
ambient coordinates with a rank-deficient Jacobian. Where every equation of `F`
shares one degree `d` the rule collapses to `deg(gⱼ) · d` with `gⱼ` homogeneous,
which the stage already records; that case is kept separate because it reads no
equations, while the weighted rule calls `StageEquations`. Degrees are upper
bounds and homogeneity is structural, as they are for a `System`.

`find_start_pair` on a composition runs Newton in `(x, p)` jointly on
`_StartPairSystem` (`core/start_pair_system.jl`), which wraps an evaluator as a
system in `[x; p]`. The parameter block of its Jacobian is the order-1
coefficient of `F(x; p + eⱼ t)`, exact and reusing the parameter-series Taylor
path. A `System` keeps its symbolic strategies (linear-in-parameters solve, then
a joint `System` in the promoted variables), which differentiate in one tape.

### Interpreter

```julia
@data ExecInstruction begin ... end  # 25 variants, one per OpType
const ExecInstructionT = typeof(ExecInstruction.Stop(Int32(0)))

struct Interpreter{T}
    sequence::InstructionSequence
    tape::FSVec{T}
    instructions::Vector{ExecInstructionT}  # compiled from Instruction at construction
end
```

Parameterized by tape type: `Vector{ComplexF64}`, `Vector{ComplexDF64}`, or `Vector{TTS{N,ComplexF64}}`.

### SExpr (Symbolic IR)

```julia
@data SExpr begin
    SConst(val::ComplexF64)
    SVar(idx::Int)
    SParam(idx::Int)
    STmp(id::Int)
    SAdd(args::Vector{SExpr})
    SMul(args::Vector{SExpr})
    SPow(base::SExpr, exp::Int)          # exp may be negative (division)
    SNeg(arg::SExpr)
    SUnary(kind::SUnaryKind.T, arg::SExpr)   # sqrt, sin, cos
    SFuncSym(kind::SFuncKind.T, args::Vector{SExpr})
end
const SExprT = typeof(SExpr.SConst(zero(ComplexF64)))  # single concrete type
```

All variants share one concrete type. Access via `variant_storage(expr)` for pattern dispatch on storage types (`SConstStorage`, `SVarStorage`, etc.).

`SPow` with a negative exponent is how division reaches the tape: `a / b` lowers to
`SMul([a, SPow(b, -1)])`, and the tape compiler splits products into numerator and
denominator, emitting `OP_DIV`/`OP_INV`/`OP_INVSQR`/`OP_POW_INT`. `SUnary` lowers to
`OP_SQRT`/`OP_SIN`/`OP_COS`.

### Expression (user-facing symbolic front-end)

```julia
@data SymExpr <: Number begin
    ENum(val::ComplexF64)
    EVar(name::Symbol)
    EAdd(args::Vector{SymExpr})
    EMul(args::Vector{SymExpr})
    EPow(base::SymExpr, exp::Int)        # exp is never 0 or 1, may be negative
    EFn(kind::SUnaryKind.T, arg::SymExpr)
end
const Expression = typeof(SymExpr.ENum(zero(ComplexF64)))
```

`Expression` is the input layer for systems that are not polynomial: division, negative
integer powers, `sqrt`, `sin` and `cos`. Because it subtypes `Number`, ordinary Julia
arithmetic, `sum`, broadcasting and matrix products build trees without extra machinery.

Every constructor canonicalizes: `EAdd`/`EMul` flatten nested nodes, fold numeric literals,
and collect like terms (by base) and like powers (by base); `EPow` folds constant bases,
composes nested powers, and distributes over products; `EFn` folds constant arguments. Two
structurally equal expressions are therefore `==` and hash equal, which is what CSE relies
on downstream.

Variables come from `@var` / `@unique_var` and are keyed by `Symbol`, so `@polyvar` names
round-trip through `Expression(::MP.AbstractPolynomialLike)` and `Expression(::MP.RationalPoly)`.
Jacobians come from `differentiate` on the tree (product, power, quotient and chain rules),
not `MP.differentiate`.

`System` records a degree of `-1` for an equation that is not polynomial in the variables;
`TotalDegree` and `Polyhedral` reject such systems, while parameter homotopies, `monodromy_solve`
and `certify` accept them.

### Tracker Stack

```julia
Tracker
  ├── homotopy::HomotopyEvaluator     # concrete, wraps any AbstractHomotopy
  ├── predictor::Predictor             # mutable: trust_region, taylor coeffs
  ├── corrector::NewtonCorrector       # immutable: scratch buffers
  ├── state::TrackerState              # mutable: x, accuracy, omega, step counts, code
  └── options::TrackerOptions          # max_steps, step_size bounds, etc.

EndgameTracker
  ├── tracker::Tracker                 # inner path tracker
  ├── state::EndgameState              # endgame-specific state (code, samples, predictions)
  ├── val::Valuation                   # per-coordinate Puiseux series valuations
  └── options::EndgameOptions          # endgame_start, max_winding_number, tolerances
```

The solve pipeline creates `EndgameTracker` wrapping `Tracker`. The inner tracker handles
predictor-corrector stepping; the endgame layer monitors valuations and switches to singular
endpoint extrapolation when winding number > 1 is detected. The predictor automatically
switches from Padé (2,1) in t-space to cubic Hermite in s-space (s = t^{1/m}) when
`winding_number > 1`, matching v2's approach for singular paths.

### Result Types

```julia
@enumx PathResultCode::Int8 begin
    PATH_SUCCESS
    PATH_AT_INFINITY
    PATH_AT_ZERO
    PATH_TERMINATED_ACCURACY
    PATH_TERMINATED_ILL_CONDITIONED
    PATH_TERMINATED_MAX_STEPS
    PATH_TERMINATED_STEP_SIZE
    PATH_TERMINATED_INVALID_START
end

struct PathResult                      # immutable
    return_code::PathResultCode.T
    solution::Vector{ComplexF64}
    t, accuracy, condition_jacobian::Float64
    winding_number::Int; singular::Bool
    accepted_steps, rejected_steps, steps_eg::Int
    extended_precision_used::Bool
    last_path_point::Vector{ComplexF64}; last_path_t::Float64
end

struct Result
    path_results::Vector{PathResult}
    tracked_paths::Int; seed::UInt32
    clusters::Vector{Vector{Int}}      # groups of paths converging to same solution
    multiplicity::Vector{Int}          # per-path cluster size (0 for non-success)
end
```

`Result` automatically deduplicates solutions at construction time via O(k²) proximity
clustering with union-find. `nsolutions`/`nresults` return the deduplicated count.
`solutions()` and `results()` return one representative per cluster. `multiplicity(r, i)`
gives the cluster size for path `i`.

### Executor & Threading

```julia
abstract type AbstractExecutor end
struct Serial <: AbstractExecutor end
struct Threaded <: AbstractExecutor
    ntasks::Int  # default Threads.nthreads(), validated ≤ nthreads()
end
struct DistributedExecutor <: AbstractExecutor
    pids::Vector{Int}       # empty ⇒ Distributed.workers(), resolved at solve time
    tasks_per_process::Int  # 0 ⇒ Threads.nthreads() on each process
    batch_size::Int         # 0 ⇒ derived from path/process/task counts
end
```

`solve(F, alg, exec)` dispatches on executor type via `SolveCache{E,B,C}`,
`PolyhedralSolveCache{E,B,S,C}` and `WorkerSolveCache{E,W,B}`.

**Builder pattern.** Each builder stores immutable reconstruction data and produces fresh
worker state per task via `builder()`. Ten live in `solving/builder.jl`, plus two in
`solving/monodromy.jl`:

```julia
StraightLineBuilder             → TrackingWorkerState   (TotalDegree)
RandomizedStraightLineBuilder   → TrackingWorkerState   (squared-up overdetermined)
MultiHomogeneousBuilder         → TrackingWorkerState   (variable-group total degree)
ParameterBuilder                → TrackingWorkerState   (parameter homotopy)
SlicedStraightLineBuilder       → TrackingWorkerState   (total degree against a slice)
ParameterRetargetBuilder        → AmbientWorkerState    (retargeted parameter homotopy)
ExtrinsicSubspaceBuilder        → AmbientWorkerState    (subspace move, ambient)
ChartExtrinsicSubspaceBuilder   → AmbientWorkerState    (projective subspace move)
IntrinsicSubspaceBuilder        → IntrinsicWorkerState  (subspace move, intrinsic)
PolyhedralBuilder               → PolyhedralWorkerState (two-phase polyhedral)
```

A builder is the only place that knows how its homotopy is stacked. Every `init` takes the
cache's own tracker from `builder()` too, so the serial tracker and the worker trackers cannot
be built two different ways: `_solve_cache` for the `SolveCache` routes, `builder()` directly
for `WorkerSolveCache` and `PolyhedralSolveCache`.

Thread safety: `_clone_system_evaluator(sys)` creates a fresh `SystemEvaluator` from the
system's `InstructionSequence`s (immutable, shared) with independent interpreter tapes
(mutable, per-worker). It preserves `CompileMode`: INTERPRETED rebuilds interpreters,
COMPILED/COMPILED_ALL re-generates `@RuntimeGeneratedFunction`s.

OhMyThreads `@tasks`/`@local` handles work distribution. `@local` creates one worker state
per task (amortized), not per path.

**Distributed** (`ext/HomotopyContinuationNextDistributedExt.jl`, weakdeps `Distributed` and
`Serialization`). The same flat index space as the threaded loops, split across processes:

- Core owns the executor type and the `_distributed_solve!` / `_distributed_sweep_entries`
  hooks. The extension fills them in. The untyped fallbacks in `executor.jl` throw an
  actionable error, and being less specific than the extension's methods, both can be
  precompiled (a method the extension had to overwrite could not be).
- The driver queues index batches on a `RemoteChannel` and issues one long-lived
  `remotecall_wait` per process, skipping any surplus process the queue is too short to
  reach. Each process loops: take a batch, claim indices from it with an atomic counter
  shared by its tasks, put back the batch's `PathResult`s. Only the loop task touches the
  channels, so socket traffic stays on one thread per process. Worker states persist across
  batches but are built only as tasks come to need them, so a small solve never pays for a
  full pool on every process.
- Every result is written at its global path index, so at a fixed seed a `DistributedExecutor`
  result is bit-identical to `Serial()` on every route, sweeps included, at any batch size.
- Builders ship as plain data: `Serialization` methods for `System`, `_SupportSystem` and
  `CompositionSystem` write the `InstructionSequence`s and rebuild the evaluator on the far
  side instead of shipping `FunctionWrapper` closures.
- Monodromy is the one route that is not a flat index space, so it gets its own scheduler
  (`monodromy.jl` in the extension). The driver keeps the job queue, the `UniquePoints` set and
  the trace matrix and hands out `MonodromyJob` batches, each carrying its loop and start point;
  a `MonodromyJobResult` brings back the `PathResult` and that loop's trace columns. Only
  `track_loop!` runs remotely, so dedup stays single-writer and the dispatch order is the serial
  one. `Threaded()` is faster on one machine (see `01_decisions.md`).

### Monodromy Stack

Ported from v2 (`HomotopyContinuation/src/monodromy.jl`) at full parity. See
`docs/superpowers/specs/2026-07-17-monodromy-port-design.md` for the port spec
and documented divergences.

```julia
monodromy_solve(F; parameter_sampler, group_actions, ...)  # or (F, sols, p₀)
  ├── find_start_pair(F)          # Newton from a random point if no seed pair given
  ├── MonodromyLoop               # p₀ → p₁ → p₂ → p₀ round trip
  │     └── ParameterHomotopy     # reused across legs via start/target_parameters!
  ├── UniquePoints{Vector{ComplexF64}, InfNorm, GA}
  │     ├── VoronoiTree           # O(log n) nearest-point search
  │     └── GroupActions          # orbit-aware dedup (SymmetricGroup, custom actions)
  └── MonodromyResult             # solutions, permutations, trace, statistics
```

Serial execution runs loops in a plain while loop. Threaded execution
(`threading = true`, default when `Threads.nthreads() > 1`) uses a
Channel-based job queue rather than the OhMyThreads executor because the
workload is dynamic: finished loops enqueue new loops and workers share
statistics mid-flight (see `01_decisions.md`). A trailing executor argument
(`monodromy_solve(F, sols, p, DistributedExecutor())`) overrides `threading` and
selects the multi-process scheduler instead, which keeps every shared structure
on the calling process.

`verify_solution_completeness` implements the trace test (del Campo/Rodriguez
2017, Leykin/Rodriguez/Sottile 2018): it builds the augmented system
`[F(x, p + λv); (Σaᵢxᵢ - 1)λ + t]`, runs an auxiliary monodromy with a
zero-first-parameter sampler, and checks the numerical rank of the trace
matrix via singular values.

`LinearSubspace` (intrinsic + extrinsic descriptions, Grassmannian geodesics,
`rand_subspace`, `geodesic_distance`) supports monodromy on positive-dimensional
solution sets: `monodromy_solve` accepts a `LinearSubspace` in place of the
parameter vector and moves it via `linear_subspace_homotopy`
(IntrinsicSubspaceHomotopy by default).

## Interface Contracts

### AbstractSystem — must implement:

```julia
Base.size(F)::Tuple{Int,Int}
evaluate!(u, F, x, p)
evaluate_and_jacobian!(u, U, F, x, p)
taylor!(u, ::Val{K}, F, tx, p) where K  # K = 1, 2, 3
```

Optional: `nparameters(F)::Int` (default 0).

### AbstractHomotopy — must implement:

```julia
Base.size(H)::Tuple{Int,Int}
evaluate!(u, H, x, t)
evaluate_and_jacobian!(u, U, H, x, t)
taylor!(u, ::Val{1}, H, x, t)                           # order 1: plain vector
taylor!(u, ::Val{K}, H, tx, t, incremental=false) where K  # order >= 2
```

Optional: `set_solution!(x, y, t)`, `get_solution!(out, x, t)`, `start_parameters!(H, p)`, `target_parameters!(H, p)`.

## Homotopy Types

| Type | Formula | Used by |
|------|---------|---------|
| StraightLineHomotopy | H(x,t) = γ·t·G(x) + (1-t)·F(x) | TotalDegree |
| CoefficientHomotopy | H(x,t) = F(x; t·start + (1-t)·target) | Polyhedral phase 2 |
| ParameterHomotopy | H(x,t) = F(x; t·p₁ + (1-t)·p₀), retargetable via start/target_parameters! | Parameter solve, `solve_targets`, monodromy loops |
| ToricHomotopy | H(x,t) = F(x; c_j·t^{w_j}) | Polyhedral phase 1 |
| IntrinsicSubspaceHomotopy | F restricted to a moving subspace, intrinsic coords (Grassmannian geodesic) | linear_subspace_homotopy (default) |
| ExtrinsicSubspaceHomotopy | [F; interpolated extrinsic equations] | linear_subspace_homotopy (fallback) |
| AffineChartHomotopy | H on a random affine chart of projective space | on_affine_chart |

## OpType Reference

| Arity | Operations |
|------:|------------|
| 0 | `STOP` |
| 1 | `CB`, `COS`, `IDENTITY`, `INV`, `INV_NOT_ZERO`, `INVSQR`, `NEG`, `SIN`, `SQR`, `SQRT` |
| 2 | `ADD`, `DIV`, `MUL`, `SUB`, `POW_INT` (2nd arg is literal int) |
| 3 | `ADD3`, `MUL3`, `MULADD`(a*b+c), `MULSUB`(a*b-c), `SUBMUL`(c-a*b) |
| 4 | `ADD4`, `MUL4`, `MULMULADD`(a*b+c*d), `MULMULSUB`(a*b-c*d) |
