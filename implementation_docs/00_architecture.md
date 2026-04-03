# Architecture

## Pipeline

```
@polyvar x y; F = System([x^2+y, x*y-1])
         │
         ▼
   poly_to_sexpr() + MP.differentiate()
         │
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
```

## Package Layout

```
src/                                         ~10,900 lines total
├── HomotopyContinuationNext.jl      (84)    Main module, exports, type aliases
├── utils.jl                         (175)   SegmentStepper, _stable_sort!, fast_abs
├── primitives/
│   ├── double_f64.jl                (655)   DoubleF64, ComplexDF64
│   ├── norms.jl                     (201)   WeightedNorm (infinity norm only)
│   └── linear_algebra.jl            (1094)  MatrixWorkspace, LU, QR, condition est.
├── model_kit/
│   ├── operations.jl                (157)   OpType enum (25 ops), op_* scalar functions
│   ├── sexpr.jl                     (380)   Moshi @data SExpr ADT, canonicalization, poly_to_sexpr
│   ├── cse.jl                       (588)   SymEngine CSE port: opt_cse + tree_cse
│   ├── tape_compiler.jl             (670)   SExpr → InstructionSequence, fusion, register alloc
│   ├── instruction_sequence.jl      (299)   Instruction, DAG reorder, linear-scan register alloc
│   ├── interpreter.jl               (464)   ExecInstruction variants, execute!, execute_taylor!
│   ├── codegen.jl                   (299)   RuntimeGeneratedFunctions for COMPILED/COMPILED_ALL
│   ├── taylor.jl                    (508)   TruncatedTaylorSeries, TaylorVector, taylor_op_*
│   ├── symbolic_polynomial_compiler.jl (62) Active MP lowering path (poly→SExpr→CSE→tape)
│   ├── polynomial_compiler.jl       (140)   Experimental direct path (NOT default, gated by TODO)
│   └── polynomial_input.jl          (83)    Variable discovery, System construction orchestrator
├── core/
│   ├── abstract_types.jl            (76)    AbstractSystem, AbstractHomotopy interfaces
│   ├── system.jl                    (259)   System type (caches compiled interpreters)
│   ├── system_evaluator.jl          (166)   FunctionWrapper wrapper for AbstractSystem
│   ├── homotopy_evaluator.jl        (162)   FunctionWrapper wrapper for AbstractHomotopy
│   ├── straight_line_homotopy.jl    (188)   γ·t·G(x) + (1-t)·F(x)
│   ├── coefficient_homotopy.jl      (174)   Coefficient interpolation (parameter + polyhedral phase 2)
│   └── toric_homotopy.jl            (400)   Toric deformation (polyhedral phase 1)
├── tracking/
│   ├── tracker.jl                   (558)   Path tracker, adaptive step control
│   ├── predictor.jl                 (351)   Pade (2,1), Taylor coefficients, trust region
│   ├── newton_corrector.jl          (425)   Alpha-theory Newton, DoubleF64 refinement
│   ├── valuation.jl                 (224)   Puiseux series valuation for endgame detection
│   └── endgame_tracker.jl           (950)   Endgame state machine, singular endpoint handling
└── solving/
    ├── executor.jl                  (44)    AbstractExecutor, Serial, Threaded
    ├── worker_state.jl              (74)    TrackingWorkerState, PolyhedralWorkerState, _clone_system_evaluator
    ├── builder.jl                   (86)    StraightLineBuilder, CoefficientBuilder, PolyhedralBuilder
    ├── solve.jl                     (148)   solve() API, CommonSolve integration, serial/threaded dispatch
    ├── total_degree.jl              (157)   Bezout start system
    ├── polyhedral.jl                (428)   Two-phase: toric + coefficient, MixedSubdivisions
    ├── binomial_system.jl           (328)   HNF binomial solver
    ├── path_result.jl               (154)   PathResult (immutable, enum codes)
    ├── result.jl                    (264)   Result, clustering, solutions(), real_solutions()
    └── support.jl                   (93)    Extract support/coefficients from MP
```

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
SystemEvaluator    # wraps 9 FunctionWrappers: eval, eval_df64, eval_jac,
                   #   taylor 1/2/3 (scalar param), taylor 1/2/3 (TaylorVector param)
                   #   + size, nparameters
HomotopyEvaluator  # wraps 10 FunctionWrappers: eval, eval_df64, eval_jac,
                   #   taylor 1/2/3, set_solution, get_solution,
                   #   start_parameters, target_parameters + size
```

All FW signatures use `FSVec{T}`/`FSMat{T}` (concrete FixedSizeArray aliases). The abstract interface uses `AbstractVector`/`AbstractMatrix` — Julia dispatch resolves automatically.

### System

```julia
struct System{P, V}
    polys::FSVec{P}                    # original MP polynomials
    parameters::FSVec{V}               # parameter variables
    variables::FSVec{V}                # decision variables
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    nvars::Int; nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    support::Vector{Matrix{Int32}}
    coefficients::Vector{Vector{ComplexF64}}
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
    SPow(base::SExpr, exp::Int)
    SNeg(arg::SExpr)
    SFuncSym(kind::SFuncKind.T, args::Vector{SExpr})
end
const SExprT = typeof(SExpr.SConst(zero(ComplexF64)))  # single concrete type
```

All variants share one concrete type. Access via `variant_storage(expr)` for pattern dispatch on storage types (`SConstStorage`, `SVarStorage`, etc.).

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
```

`solve(F, alg, exec)` dispatches on executor type via `SolveCache{E,B}` / `PolyhedralSolveCache{E,B,S}`.

**Builder pattern** — each builder stores immutable reconstruction data and produces fresh
worker state per task via `builder()`:

```julia
StraightLineBuilder  → TrackingWorkerState    (TotalDegree)
CoefficientBuilder   → TrackingWorkerState    (parameter homotopy)
PolyhedralBuilder    → PolyhedralWorkerState  (two-phase polyhedral)
```

Thread safety: `_clone_system_evaluator(sys)` creates a fresh `SystemEvaluator` from the
system's `InstructionSequence`s (immutable, shared) with independent interpreter tapes
(mutable, per-worker). Preserves `CompileMode` — INTERPRETED rebuilds interpreters,
COMPILED/COMPILED_ALL re-generates `@RuntimeGeneratedFunction`s.

OhMyThreads `@tasks`/`@local` handles work distribution — `@local` creates one worker state
per task (amortized), not per path.

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
| CoefficientHomotopy | H(x,t) = F(x; t·start + (1-t)·target) | Polyhedral phase 2, parameter homotopy |
| ToricHomotopy | H(x,t) = F(x; c_j·t^{w_j}) | Polyhedral phase 1 |

## OpType Reference

| Arity | Operations |
|------:|------------|
| 0 | `STOP` |
| 1 | `CB`, `COS`, `IDENTITY`, `INV`, `INV_NOT_ZERO`, `INVSQR`, `NEG`, `SIN`, `SQR`, `SQRT` |
| 2 | `ADD`, `DIV`, `MUL`, `SUB`, `POW_INT` (2nd arg is literal int) |
| 3 | `ADD3`, `MUL3`, `MULADD`(a*b+c), `MULSUB`(a*b-c), `SUBMUL`(c-a*b) |
| 4 | `ADD4`, `MUL4`, `MULMULADD`(a*b+c*d), `MULMULSUB`(a*b-c*d) |
