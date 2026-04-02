# Architecture

## Pipeline

```
@polyvar x y; F = [x^2+y, x*y-1]
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
   solve() orchestration → Result{Vector{PathResult}}
```

## Package Layout

```
src/
├── HomotopyContinuationNext.jl      Main module, exports
├── utils.jl                         SegmentStepper, _stable_sort!, fast_abs
├── primitives/
│   ├── double_f64.jl                DoubleF64, ComplexDF64
│   ├── norms.jl                     WeightedNorm (infinity norm only)
│   └── linear_algebra.jl            MatrixWorkspace, LU, QR, condition est.
├── model_kit/
│   ├── operations.jl                OpType enum (25 ops), op_* scalar functions
│   ├── sexpr.jl                     Moshi @data SExpr ADT, canonicalization, poly_to_sexpr
│   ├── cse.jl                       SymEngine CSE port: opt_cse + tree_cse
│   ├── tape_compiler.jl             SExpr → InstructionSequence, fusion, register alloc
│   ├── instruction_sequence.jl      Instruction, DAG reorder, linear-scan register alloc
│   ├── interpreter.jl               ExecInstruction variants, execute!, execute_taylor!
│   ├── taylor.jl                    TruncatedTaylorSeries, TaylorVector, taylor_op_*
│   ├── symbolic_polynomial_compiler.jl  Active MP lowering path (poly→SExpr→CSE→tape)
│   ├── polynomial_compiler.jl       Experimental direct path (NOT default, gated by TODO)
│   └── polynomial_input.jl          Variable discovery, System construction orchestrator
├── core/
│   ├── abstract_types.jl            AbstractSystem, AbstractHomotopy interfaces
│   ├── system.jl                    System type (caches compiled interpreters)
│   ├── system_evaluator.jl          FunctionWrapper wrapper for AbstractSystem
│   ├── homotopy_evaluator.jl        FunctionWrapper wrapper for AbstractHomotopy
│   ├── straight_line_homotopy.jl    γ·t·G(x) + (1-t)·F(x)
│   ├── coefficient_homotopy.jl      Coefficient interpolation (parameter + polyhedral phase 2)
│   └── toric_homotopy.jl            Toric deformation (polyhedral phase 1)
├── tracking/
│   ├── tracker.jl                   Path tracker, adaptive step control
│   ├── predictor.jl                 Pade (2,1), Taylor coefficients, trust region
│   ├── newton_corrector.jl          Alpha-theory Newton, DoubleF64 refinement
│   ├── valuation.jl                 Puiseux series valuation for endgame detection
│   └── endgame_tracker.jl           Endgame state machine, singular endpoint handling
└── solving/
    ├── solve.jl                     solve() API, CommonSolve integration
    ├── total_degree.jl              Bezout start system
    ├── polyhedral.jl                Two-phase: toric + coefficient, MixedSubdivisions
    ├── binomial_system.jl           HNF binomial solver
    ├── path_result.jl               PathResult (immutable, enum codes)
    ├── result.jl                    Result, solutions(), real_solutions(), nsolutions()
    └── support.jl                   Extract support/coefficients from MP
```

## Key Types

### Type Firewall

Every `AbstractSystem`/`AbstractHomotopy` is wrapped via `FunctionWrapper` into concrete evaluators. The tracker is monomorphic — compiled once, reused for all systems.

```julia
SystemEvaluator    # wraps 6 FunctionWrappers: eval, eval_df64, eval_jac, taylor 1/2/3
HomotopyEvaluator  # wraps 10 FunctionWrappers: eval, eval_df64, eval_jac, taylor 1/2/3,
                   #   set_solution, get_solution, start_parameters, target_parameters
```

All FW signatures use `FSVec{T}`/`FSMat{T}` (concrete FixedSizeArray aliases). The abstract interface uses `AbstractVector`/`AbstractMatrix` — Julia dispatch resolves automatically.

### System

```julia
struct System
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    nvars::Int; nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    support::Vector{Matrix{Int32}}
    coefficients::Vector{Vector{ComplexF64}}
    # GC roots — interpreters must stay alive for FunctionWrapper closures
    _interp_f64, _interp_df64, _interp_jac, _interp_t1, _interp_t2, _interp_t3
end
```

### Interpreter

```julia
@data ExecInstruction begin ... end  # 25 variants, one per OpType
const ExecInstructionT = typeof(ExecInstruction.Stop(Int32(0)))

struct Interpreter{V<:AbstractVector}
    sequence::InstructionSequence
    instructions::Vector{ExecInstructionT}  # compiled from Instruction at construction
    tape::V                                  # contents mutated, reference fixed
end
```

Parameterized by tape type: `Vector{ComplexF64}`, `Vector{ComplexDF64}`, or `Vector{TTS{N,ComplexF64}}`.

### SExpr (Symbolic IR)

```julia
@data SExpr begin
    SConst; SVar; SParam; STmp; SAdd; SMul; SPow; SNeg; SFuncSym
end
const SExprT = typeof(SExpr.SConst(zero(ComplexF64)))  # single concrete type
```

All variants share one concrete type. Access via `sexpr_storage(expr)` for pattern dispatch on storage types (`SConstStorage`, `SVarStorage`, etc.).

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
endpoint extrapolation when winding number > 1 is detected.

### Result Types

```julia
struct PathResult                      # immutable
    return_code::PathResultCode        # enumx: PATH_SUCCESS, PATH_AT_INFINITY, etc.
    solution::Vector{ComplexF64}
    t, accuracy, condition_jacobian::Float64
    winding_number::Int; singular::Bool
    accepted_steps, rejected_steps, steps_eg::Int
    extended_precision_used::Bool
    last_path_point::Vector{ComplexF64}; last_path_t::Float64
end

struct Result
    path_results::Vector{PathResult}
    tracked_paths::Int; seed::UInt32; start_system::Symbol
end
```

## Interface Contracts

### AbstractSystem — must implement:

```julia
Base.size(F)::Tuple{Int,Int}
evaluate!(u, F, x, p)
evaluate_and_jacobian!(u, U, F, x, p)
taylor!(u, ::Val{K}, F, tx::TaylorVector{K+1}, p) where K
```

### AbstractHomotopy — must implement:

```julia
Base.size(H)::Tuple{Int,Int}
evaluate!(u, H, x, t)
evaluate_and_jacobian!(u, U, H, x, t)
taylor!(u, ::Val{1}, H, x, t)                           # order 1: plain vector
taylor!(u, ::Val{K}, H, tx, t, incremental=false) where K  # order >= 2
```

Optional: `set_solution!`, `get_solution!`, `start_parameters!`, `target_parameters!`.

## OpType Reference

| Arity | Operations |
|------:|------------|
| 0 | `STOP` |
| 1 | `CB`, `COS`, `IDENTITY`, `INV`, `INV_NOT_ZERO`, `INVSQR`, `NEG`, `SIN`, `SQR`, `SQRT` |
| 2 | `ADD`, `DIV`, `MUL`, `SUB`, `POW_INT` (2nd arg is literal int) |
| 3 | `ADD3`, `MUL3`, `MULADD`(a*b+c), `MULSUB`(a*b-c), `SUBMUL`(c-a*b) |
| 4 | `ADD4`, `MUL4`, `MULMULADD`(a*b+c*d), `MULMULSUB`(a*b-c*d) |
