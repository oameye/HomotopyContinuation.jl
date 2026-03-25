# HomotopyContinuationNext.jl — Design Document

**Date:** 2026-03-25
**Status:** Draft — Architecture Phase
**Scope:** Minimal core — `solve()` with total degree and polyhedral start systems

---

## 1. Design Principles

1. **Concrete by default.** The tracker, Newton corrector, predictor — all hot-path types are fully concrete with zero abstract-typed fields.
2. **Extensible by contract.** `AbstractSystem` and `AbstractHomotopy` define the user-facing extension points. User types are wrapped into concrete evaluators via `FunctionWrapper` at the solve boundary.
3. **No type parameter propagation.** The system/homotopy identity never appears in the tracker's type. Every new polynomial system reuses the same compiled tracker code.
4. **Interpreter-first.** A single tape-based interpreter handles all evaluation modes (eval, jacobian, Taylor, DF64) through Julia's generic dispatch on the tape element type. Compiled mode is a future optimization, not a structural dependency.
5. **Lean on the ecosystem.** DynamicPolynomials.jl for user-facing input, MultivariatePolynomials.jl for symbolic differentiation and polynomial manipulation, FunctionWrappers.jl for type erasure. No Symbolics.jl in core.

---

## 2. User API

### Primary: DynamicPolynomials

```julia
using HomotopyContinuationNext
using DynamicPolynomials

# Basic solve
@polyvar x y
result = solve([x^2 + y - 1, x*y - 2])

# Access solutions
solutions(result)
real_solutions(result)
nsolutions(result)

# With parameters
@polyvar x y a b
result = solve(
    [x^2 + a*y, x*y - b];
    parameters = [a, b],
    target_parameters = [1.0, 2.0],
)

# Variable groups (multi-homogeneous)
@polyvar x y z
result = solve([x*y - z^2, x + y + z]; variable_groups = [[x], [y, z]])

# Choose start system
result = solve(F; start_system = :total_degree)   # Bézout bound
result = solve(F; start_system = :polyhedral)      # BKK bound (default)
```

### Secondary: Custom systems via AbstractSystem

```julia
struct MySystem <: AbstractSystem
    # user fields...
end
Base.size(F::MySystem) = (2, 2)  # (nequations, nvariables)

# Must implement:
evaluate!(u, F::MySystem, x, p)
evaluate_and_jacobian!(u, U, F::MySystem, x, p)
taylor!(u, ::Val{K}, F::MySystem, tx, p) where K

result = solve(MySystem(...))
```

### Future extension: Symbolics.jl (non-polynomial)

```julia
# Via package extension — not in core
using Symbolics
@variables x y
result = solve([x^2 + sin(y), cos(x) - y])  # uses Symbolics.build_function internally
```

---

## 3. Architecture

```
                        ┌───────────────────────────┐
                        │      User Input            │
                        │  @polyvar x y              │
                        │  F = [x^2+y, x*y-1]       │
                        │  solve(F)                  │
                        └─────────┬─────────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │   MP.exponents, MP.coefficients,       │
              │   MP.differentiate → Jacobian polys    │
              │   MP.maxdegree → degrees               │
              └───────────────────┬───────────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │        Instruction Sequence             │
              │   Shared monomial table (F + J)         │
              │   DAG reorder, register allocation      │
              └───────────────────┬───────────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │   Interpreter{V}                       │
              │   One tape, multiple element types:     │
              │   • V = Vector{ComplexF64}   (eval)    │
              │   • V = Vector{ComplexDF64}  (DF64)    │
              │   • V = Vector{TTS{N,CF64}}  (Taylor)  │
              └───────────────────┬───────────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │   FunctionWrapper{...}                  │
              │   → SystemEvaluator (concrete, no type params)│
              └───────────────────┬───────────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │   Homotopy (StraightLine, Toric, etc.)  │
              │   → HomotopyEvaluator (concrete)             │
              └───────────────────┬───────────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │   Tracker (monomorphic)                 │
              │     Predictor · NewtonCorrector          │
              │     Jacobian · WeightedNorm              │
              │   EndgameTracker                         │
              │     Valuation · Singular/∞ detection     │
              └───────────────────┬───────────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │   solve() orchestration                 │
              │     TotalDegree / Polyhedral             │
              │     Serial + threaded tracking           │
              └───────────────────┬───────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Result{Vector{PathResult}} │
                    └─────────────────────────────┘
```

### The Key Ideas

**1. FunctionWrapper as the type firewall.** Every `AbstractSystem` and `AbstractHomotopy` is wrapped into `SystemEvaluator` / `HomotopyEvaluator` using `FunctionWrapper`. The tracker only sees concrete types. New polynomial systems reuse the same compiled tracker code.

**2. Interpreter-first.** One tape-based interpreter handles all evaluation modes — `ComplexF64`, `ComplexDF64`, and `TruncatedTaylorSeries{N}` — through Julia's generic dispatch on the tape element type. The same `InstructionSequence` drives all modes. This eliminates the dual-track (compiled vs. interpreted) complexity.

**3. MP as the symbolic engine.** MultivariatePolynomials provides differentiation (`MP.differentiate`), exponent/coefficient extraction, degree computation, and homogeneity checks. No SymEngine, no Symbolics.jl in core.

---

## 4. Feature Inventory (Minimal Core)

### Must-have for v1

| Category | Features |
|----------|----------|
| **Input** | Accept `Vector{<:MP.AbstractPolynomialLike}`, `AbstractSystem` |
| **Extraction** | MP-native: `exponents`, `coefficients`, `differentiate`, `maxdegree`, `effective_variables`, `ishomogeneous` |
| **Evaluation** | Tape interpreter for eval, jacobian, Taylor orders 1-3, DF64 |
| **Start systems** | Total degree, polyhedral (BKK mixed volume) |
| **Path tracking** | Padé (2,1) predictor, α-theory Newton corrector, adaptive step size |
| **Endgame** | Singular endpoint detection, at-infinity detection, Hermite endgame |
| **Precision** | Float64 + DoubleF64 mixed-precision iterative refinement |
| **Linear algebra** | Custom LU with Skeel scaling, condition estimation |
| **solve()** | Serial + threaded solving, progress reporting |
| **Results** | `PathResult`, `Result`, filtering (solutions, real_solutions, etc.) |
| **Utilities** | UniquePoints, VoronoiTree, WeightedNorm, LinearSubspace |

### Deferred to later

| Feature | Reason |
|---------|--------|
| Compiled mode (`:mixed`, `:all`) | Optimization via Symbolics.jl package extension — interpreter is within 4% |
| Monodromy | Not needed for basic solve |
| Certification (Krawczyk) | Requires interval arithmetic |
| Witness sets / NID | Requires monodromy |
| Parameter homotopy (user API) | Add after core works |
| Subspace homotopies | Add after core works |
| SemialgebraicSets | Package extension |
| Symbolics.jl input | Package extension for non-polynomial systems |

---

## 5. Module Structure

```
src/
├── HomotopyContinuationNext.jl
│
├── primitives/
│   ├── double_f64.jl               # DoubleF64, ComplexDF64
│   ├── norms.jl                    # InfNorm, WeightedNorm
│   ├── linear_algebra.jl           # MatrixWorkspace, LU, QR, condition est.
│   ├── voronoi_tree.jl             # Proximity search
│   ├── unique_points.jl            # Deduplication + group actions
│   └── linear_subspaces.jl         # LinearSubspace, geodesics
│
├── model_kit/
│   ├── operations.jl               # OpType enum, op_* scalar functions
│   ├── instruction_sequence.jl     # IR builder, tape layout, DAG reorder, register alloc
│   ├── interpreter.jl              # Tape-based evaluator (generic over element type)
│   ├── taylor.jl                   # TruncatedTaylorSeries, TaylorVector, taylor_op_*
│   └── polynomial_input.jl         # MP polynomials → InstructionSequence
│
├── core/
│   ├── abstract_types.jl           # AbstractSystem, AbstractHomotopy
│   ├── system_eval.jl              # SystemEvaluator (FunctionWrapper-based)
│   ├── homotopy_eval.jl            # HomotopyEvaluator (FunctionWrapper-based)
│   ├── straight_line_homotopy.jl   # γt·G + (1-t)·F
│   ├── affine_chart.jl             # Projective normalization
│   ├── coefficient_homotopy.jl     # For polyhedral 2nd phase
│   ├── toric_homotopy.jl           # Toric degeneration for polyhedral
│   ├── fixed_parameters.jl         # Bind parameter values
│   └── randomized_system.jl        # Square-up for overdetermined
│
├── tracking/
│   ├── predictor.jl                # Padé (2,1), Hermite
│   ├── newton_corrector.jl         # α-theory Newton
│   ├── tracker.jl                  # Core path tracker
│   ├── valuation.jl                # Puiseux valuation
│   ├── endgame.jl                  # Endgame tracker
│   └── path_result.jl             # PathResult (immutable, enum codes)
│
├── solving/
│   ├── total_degree.jl             # Bézout start system
│   ├── polyhedral.jl               # BKK mixed volume, PolyhedralTracker
│   ├── binomial.jl                 # HNF binomial solver
│   ├── overdetermined.jl           # OverdeterminedTracker
│   ├── solve.jl                    # Top-level solve()
│   └── result.jl                   # Result, statistics, filtering
│
└── utils.jl                        # SegmentStepper, fast_abs, etc.
```

---

## 6. Type Architecture

### 6.1 Abstract Interface (user-facing)

```julia
abstract type AbstractSystem end
abstract type AbstractHomotopy end

# === Required interface for AbstractSystem ===
Base.size(F::AbstractSystem)::Tuple{Int,Int}   # (nequations, nvariables)
nparameters(F::AbstractSystem)::Int             # default: 0

evaluate!(u::AbstractVector, F::AbstractSystem,
          x::AbstractVector, p::AbstractVector)
evaluate_and_jacobian!(u::AbstractVector, U::AbstractMatrix,
                       F::AbstractSystem, x::AbstractVector, p::AbstractVector)

# Taylor: Val{K} means compute order-K derivative. tx has K+1 slots (orders 0..K).
taylor!(u::AbstractVector, ::Val{K}, F::AbstractSystem,
        tx::TaylorVector{K+1}, p::AbstractVector) where K

# === Required interface for AbstractHomotopy ===
Base.size(H::AbstractHomotopy)::Tuple{Int,Int}
evaluate!(u::AbstractVector, H::AbstractHomotopy,
          x::AbstractVector, t::Number)
evaluate_and_jacobian!(u::AbstractVector, U::AbstractMatrix,
                       H::AbstractHomotopy, x::AbstractVector, t::Number)

# Taylor for homotopies:
# - Val{1} with plain Vector x: compute dx/dt from H(x,t)=0
# - Val{K} with TaylorVector{K+1}: compute order-K coefficient
# - incremental::Bool: if true, lower-order coefficients are already cached
taylor!(u::AbstractVector, ::Val{1}, H::AbstractHomotopy,
        x::AbstractVector, t::Number)
taylor!(u::AbstractVector, ::Val{K}, H::AbstractHomotopy,
        tx::TaylorVector{K+1}, t::Number, incremental::Bool=false) where K

# === Optional (with defaults) ===
set_solution!(x, H::AbstractHomotopy, y, t) = (x .= y)
get_solution!(out, H::AbstractHomotopy, x, t) = copyto!(out, x)
start_parameters!(H::AbstractHomotopy, p) = H
target_parameters!(H::AbstractHomotopy, p) = H
```

### 6.2 SystemEvaluator — The Type Firewall

All FW signatures use `FSVec`/`FSMat` from FixedSizeArrays.jl (see doc 01 for full definitions).
The abstract interface (section 6.1) uses `AbstractVector`/`AbstractMatrix` — Julia dispatch
resolves `FSVec <: AbstractVector` automatically, so user implementations work without importing
FixedSizeArrays.

```julia
using FunctionWrappers: FunctionWrapper
using FixedSizeArrays: FixedSizeArray
# CRITICAL: FixedSizeVector{T} is NOT concrete (Mem param is free).
# Must pin Memory{T} for struct fields to be type-stable.
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

# System FunctionWrapper signatures (see doc 01 for full list)
const SysEvalFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, FSVec{ComplexF64}}}
const SysEvalDF64FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexDF64}, FSVec{ComplexF64}}}
const SysEvalJacFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSMat{ComplexF64},
    FSVec{ComplexF64}, FSVec{ComplexF64}}}
const SysTaylor1FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{2,ComplexF64}, FSVec{ComplexF64}}}
const SysTaylor2FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{3,ComplexF64}, FSVec{ComplexF64}}}
const SysTaylor3FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{4,ComplexF64}, FSVec{ComplexF64}}}

struct SystemEvaluator
    _evaluate!::SysEvalFW
    _evaluate_df64!::SysEvalDF64FW
    _evaluate_and_jacobian!::SysEvalJacFW
    _taylor_1!::SysTaylor1FW        # order 1: TaylorVector{2}
    _taylor_2!::SysTaylor2FW        # order 2: TaylorVector{3}
    _taylor_3!::SysTaylor3FW        # order 3: TaylorVector{4}
    _size::Tuple{Int,Int}
    _nparameters::Int
end
```

**For user-defined AbstractSystem:**
```julia
function SystemEvaluator(F::AbstractSystem)
    SystemEvaluator(
        SysEvalFW((u,x,p) -> (evaluate!(u, F, x, p); nothing)),
        SysEvalDF64FW((u,x,p) -> (evaluate!(u, F, x, p); nothing)),
        SysEvalJacFW((u,U,x,p) -> (evaluate_and_jacobian!(u, U, F, x, p); nothing)),
        SysTaylor1FW((u,tx,p) -> (taylor!(u, Val(1), F, tx, p); nothing)),
        SysTaylor2FW((u,tx,p) -> (taylor!(u, Val(2), F, tx, p); nothing)),
        SysTaylor3FW((u,tx,p) -> (taylor!(u, Val(3), F, tx, p); nothing)),
        size(F), nparameters(F),
    )
end
```

**For polynomial input (produces SystemEvaluator directly, no intermediate type):**
```julia
function system_eval(
    polys::AbstractVector{<:MP.AbstractPolynomialLike};
    variables = MP.effective_variables(polys),
    parameters = eltype(variables)[],
)::Tuple{PolynomialSystemInfo, SystemEvaluator}
    vars = setdiff(variables, parameters)
    n, m = length(vars), length(polys)

    # 1. Extract supports + coefficients using MP
    supports, coeffs = extract_supports(polys, vars, parameters)

    # 2. Symbolic Jacobian via MP.differentiate (returns MP polynomials)
    J_polys = MP.differentiate(polys, vars)
    j_supports, j_coeffs = extract_supports(vec(J_polys), vars, parameters)

    # 3. Build shared instruction sequence (F + J share monomials = CSE)
    seq = build_instruction_sequence(supports, coeffs, j_supports, j_coeffs, vars, parameters)

    # 4. Build interpreters (same tape, different element types)
    interp_f64  = Interpreter(Vector{ComplexF64}, seq)
    interp_df64 = Interpreter(Vector{ComplexDF64}, seq)
    interp_t1   = Interpreter(Vector{TruncatedTaylorSeries{2,ComplexF64}}, seq)
    interp_t2   = Interpreter(Vector{TruncatedTaylorSeries{3,ComplexF64}}, seq)
    interp_t3   = Interpreter(Vector{TruncatedTaylorSeries{4,ComplexF64}}, seq)

    # 5. Wrap in FunctionWrappers
    degrees = [MP.maxdegree(p) for p in polys]
    info = PolynomialSystemInfo(degrees, n, length(parameters), ...)

    eval = SystemEvaluator(
        SysEvalFW((u,x,p)     -> (execute!(u, interp_f64, x, p); nothing)),
        SysEvalDF64FW((u,x,p) -> (execute!(u, interp_df64, x, p); nothing)),
        SysEvalJacFW((u,U,x,p)-> (execute_jac!(u, U, interp_f64, x, p); nothing)),
        SysTaylor1FW((u,tx,p) -> (execute_taylor!(u, Val(1), interp_t1, tx, p); nothing)),
        SysTaylor2FW((u,tx,p) -> (execute_taylor!(u, Val(2), interp_t2, tx, p); nothing)),
        SysTaylor3FW((u,tx,p) -> (execute_taylor!(u, Val(3), interp_t3, tx, p); nothing)),
        (m, n), length(parameters),
    )
    return info, eval
end
```

### 6.3 HomotopyEvaluator

```julia
# Homotopy FunctionWrapper signatures (FSVec/FSMat, matching doc 01)
const HomEvalFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
const HomEvalDF64FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexDF64}, ComplexF64}}
const HomEvalJacFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSMat{ComplexF64},
    FSVec{ComplexF64}, ComplexF64}}
const HomTaylor1FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
const HomTaylor2FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{3,ComplexF64}, ComplexF64, Bool}}
const HomTaylor3FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{4,ComplexF64}, ComplexF64, Bool}}
const HomSetSolFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
const HomGetSolFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
const HomParamsFW = FunctionWrapper{Nothing, Tuple{FSVec{ComplexF64}}}

struct HomotopyEvaluator
    _evaluate!::HomEvalFW
    _evaluate_df64!::HomEvalDF64FW
    _evaluate_and_jacobian!::HomEvalJacFW
    _taylor_1!::HomTaylor1FW
    _taylor_2!::HomTaylor2FW
    _taylor_3!::HomTaylor3FW
    _set_solution!::HomSetSolFW
    _get_solution!::HomGetSolFW
    _start_parameters!::HomParamsFW
    _target_parameters!::HomParamsFW
    _size::Tuple{Int,Int}
end

function HomotopyEvaluator(H::AbstractHomotopy)
    HomotopyEvaluator(
        HomEvalFW((u,x,t) -> (evaluate!(u, H, x, t); nothing)),
        HomEvalDF64FW((u,x,t) -> (evaluate!(u, H, x, t); nothing)),
        HomEvalJacFW((u,U,x,t) -> (evaluate_and_jacobian!(u, U, H, x, t); nothing)),
        HomTaylor1FW((u,x,t) -> (taylor!(u, Val(1), H, x, t); nothing)),
        HomTaylor2FW((u,tx,t,inc) -> (taylor!(u, Val(2), H, tx, t, inc); nothing)),
        HomTaylor3FW((u,tx,t,inc) -> (taylor!(u, Val(3), H, tx, t, inc); nothing)),
        HomSetSolFW((x,y,t) -> (set_solution!(x, H, y, t); nothing)),
        HomGetSolFW((out,x,t) -> (get_solution!(out, H, x, t); nothing)),
        HomParamsFW((p) -> (start_parameters!(H, p); nothing)),
        HomParamsFW((p) -> (target_parameters!(H, p); nothing)),
        size(H),
    )
end
```

### 6.4 Polynomial System Metadata

```julia
struct PolynomialSystemInfo
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Union{Nothing, Vector{Vector{Int}}}
    is_homogeneous::Bool
    # Keep interpreters alive (FW closures capture them by reference)
    _seq::InstructionSequence
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2,ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3,ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4,ComplexF64}}}
end
```

### 6.5 Homotopy Types

Built-in homotopies store `SystemEvaluator` (already type-erased) and pre-allocated scratch:

```julia
struct StraightLineHomotopy <: AbstractHomotopy
    start::SystemEvaluator
    target::SystemEvaluator
    γ::ComplexF64
    u_start::FSVec{ComplexF64}
    u_target::FSVec{ComplexF64}
    ū_start::FSVec{ComplexDF64}
    ū_target::FSVec{ComplexDF64}
    U_start::FSMat{ComplexF64}
    U_target::FSMat{ComplexF64}
    dv_start::TaylorVector{4,ComplexF64}
    dv_target::TaylorVector{4,ComplexF64}
end

struct CoefficientHomotopy <: AbstractHomotopy
    system::SystemEvaluator
    start_coeffs::FSVec{ComplexF64}
    target_coeffs::FSVec{ComplexF64}
    t_cache::Base.RefValue{ComplexF64}        # last cached t (NaN = invalid)
    t_taylor_cache::Base.RefValue{ComplexF64} # last cached t for Taylor
    coeffs::FSVec{ComplexF64}                 # mutated contents
    dt_coeffs::FSVec{ComplexF64}
    taylor_coeffs::TaylorVector{2,ComplexF64}
end

struct ToricHomotopy <: AbstractHomotopy
    system::SystemEvaluator
    system_coeffs::FSVec{ComplexF64}
    weights::FSVec{Float64}
    t_weights::FSVec{Float64}
    complex_t_weights::FSVec{ComplexF64}
    coeffs::FSVec{ComplexF64}                 # mutated contents
    dt_coeffs::FSVec{ComplexF64}
    x::FSVec{ComplexF64}                      # scratch for toric-rescaled point
    t_coeffs::Base.RefValue{ComplexF64}       # last cached t (NaN = invalid)
    t_taylor_coeffs::Base.RefValue{ComplexF64}
    taylor_coeffs::TaylorVector{5,ComplexF64}
    tc3::TaylorVector{4,ComplexF64}
    tc2::TaylorVector{3,ComplexF64}
end

struct AffineChartHomotopy <: AbstractHomotopy
    homotopy::HomotopyEvaluator     # inner homotopy, already wrapped
    chart::FSVec{ComplexF64}
    ndims::Int
end
```

### 6.6 Path Tracking (Fully Concrete)

```julia
struct Tracker
    homotopy::HomotopyEvaluator
    predictor::Predictor
    corrector::NewtonCorrector
    state::TrackerState
    options::TrackerOptions
end

struct EndgameTracker
    tracker::Tracker
    state::EndgameState
    options::EndgameOptions
end
```

All types in the tracking pipeline are concrete. No type parameters. The tracker compiles once during precompilation and is reused for every solve.

### 6.7 Result Types

```julia
@enum PathResultCode::Int8 begin
    PATH_SUCCESS
    PATH_AT_INFINITY
    PATH_AT_ZERO
    PATH_EXCESS_SOLUTION
    PATH_TERMINATED_ACCURACY
    PATH_TERMINATED_ILL_CONDITIONED
    PATH_TERMINATED_MAX_STEPS
    PATH_TERMINATED_STEP_SIZE
    PATH_TERMINATED_INVALID_START
end

struct PathResult                       # immutable
    return_code::PathResultCode
    solution::Vector{ComplexF64}
    t::Float64
    accuracy::Float64
    residual::Float64
    condition_jacobian::Float64
    winding_number::Int                # 0 = not computed
    multiplicity::Int
    singular::Bool
    ω::Float64
    μ::Float64
    accepted_steps::Int
    rejected_steps::Int
end

struct Result
    path_results::Vector{PathResult}
    tracked_paths::Int
    seed::UInt32
    start_system::Symbol
end
```

### 6.8 Thread Safety

Homotopy types have mutable scratch buffers. For threaded solving, each thread gets a deep copy:

```julia
function threaded_solve(tracker::Tracker, starts; kwargs...)
    trackers = [i == 1 ? tracker : deepcopy(tracker) for i in 1:Threads.nthreads()]
    # Each thread uses trackers[Threads.threadid()]
end
```

`deepcopy(::Tracker)` should copy the entire tree including FunctionWrapper closures and their captured homotopy scratch buffers. The concrete type is unchanged.

**VERIFICATION REQUIRED (Phase 6):** `FunctionWrapper` stores closures via an internal representation that may use raw pointers. It is NOT guaranteed that `deepcopy` correctly deep-copies the closure environment. Before relying on threaded solving, we must verify with a test that `deepcopy(tracker)` produces fully independent scratch buffers. If it does not, the fallback is to reconstruct `HomotopyEvaluator` from the original homotopy object per thread rather than deep-copying.

**Precompilation:** FunctionWrapper objects are not serializable, so `SystemEvaluator`/`HomotopyEvaluator` are constructed at runtime in `solve()`. But the *code that uses them* — `step!`, `newton!`, `predict!` — compiles against concrete types and is precompilable via `@compile_workload`.

---

## 7. Compilation Pipeline

### 7.1 From MP Polynomials to Interpreter

```
Input: Vector{<:MP.AbstractPolynomialLike}
  │
  ▼
Step 1: MP extraction
  MP.effective_variables(polys)   → variables (only those that appear)
  MP.maxdegree(p)                 → degrees (for total degree start system)
  MP.ishomogeneous(p)             → homogeneity check (DynamicPolynomials)
  MP.terms(p), MP.exponents(t),
  MP.coefficient(t)               → exponent matrices + coefficient vectors
  │
  ▼
Step 2: Jacobian via MP
  MP.differentiate(polys, vars)   → m×n matrix of MP polynomials
  → extract supports/coefficients from Jacobian polynomials too
  │
  ▼
Step 3: Build shared monomial table
  Collect all distinct monomials across F and ∂F/∂x
  Key: exponent vector, Value: monomial ID
  → CSE by construction: shared monomials computed once
  │
  ▼
Step 4: Build InstructionSequence
  For each monomial: power decomposition + multiply tree
    (OP_SQR, OP_CB, OP_POW_INT, OP_MUL, OP_MUL3)
  For each polynomial: coeff × monomial, greedy sum fusion
    (OP_MULADD, OP_ADD3, OP_MULMULADD)
  DAG-based instruction reorder (data locality)
  Liveness-based register allocation (minimize tape size)
  │
  ▼
Step 5: Build Interpreters
  Interpreter{Vector{ComplexF64}}                      (eval, jacobian)
  Interpreter{Vector{ComplexDF64}}                     (extended precision)
  Interpreter{Vector{TruncatedTaylorSeries{2,CF64}}}  (Taylor order 1)
  Interpreter{Vector{TruncatedTaylorSeries{3,CF64}}}  (Taylor order 2)
  Interpreter{Vector{TruncatedTaylorSeries{4,CF64}}}  (Taylor order 3)
  │
  ▼
Step 6: Wrap in FunctionWrapper → SystemEvaluator
```

### 7.2 Why Interpreter-Only

The current codebase's `InterpretedSystem` benchmarks show:
- **TTFX:** 1.1s (vs 28s for compiled mode)
- **Runtime:** within 4% of compiled
- **No unique types:** the interpreter has no type parameters that vary per system

The 4% runtime gap comes from the interpreter's dispatch loop overhead (one branch per instruction). For systems with ≥5 variables, the O(n²) Jacobian evaluation + O(n³) LU factorization dominate, making the interpreter overhead negligible.

### 7.3 Future: Compiled Mode via Package Extension

When the 4% matters, add a Symbolics.jl extension:

```julia
# ext/SymbolicsExt.jl — optional, not loaded by default
using Symbolics

function compile_system_eval(polys, vars, params)
    sym_exprs = dp_to_symbolics(polys, vars)
    _, f! = Symbolics.build_function(sym_exprs, ...; expression=Val{false}, cse=true)
    J = Symbolics.jacobian(sym_exprs, ...)
    _, j! = Symbolics.build_function(J, ...; expression=Val{false}, cse=true)
    # Return FunctionWrappers wrapping the compiled functions
    # Taylor and DF64 still go through interpreter
end
```

This keeps Symbolics.jl out of the core dependency tree and its load time out of `using HomotopyContinuationNext`.

---

## 8. Solve Orchestration

### 8.1 solve() dispatch

```julia
function solve(
    F::AbstractVector{<:MP.AbstractPolynomialLike};
    start_system = :polyhedral,
    parameters = eltype(MP.variables(F))[],
    target_parameters = ComplexF64[],
    variable_groups = nothing,
    threading = Threads.nthreads() > 1,
    seed = rand(UInt32),
    show_progress = true,
    kwargs...
)
    # 1. Build SystemEvaluator from polynomials
    info, sys_eval = system_eval(F; variables=MP.effective_variables(F), parameters)

    # 2. Dispatch on start system
    if start_system == :polyhedral
        tracker, starts = polyhedral_start(sys_eval, info; kwargs...)
    elseif start_system == :total_degree
        tracker, starts = total_degree_start(sys_eval, info; kwargs...)
    end

    # 3. Track all paths
    if threading
        threaded_solve(tracker, starts; seed, show_progress)
    else
        serial_solve(tracker, starts; seed, show_progress)
    end
end

# Also accept AbstractSystem directly
function solve(F::AbstractSystem; kwargs...)
    sys_eval = SystemEvaluator(F)
    ...
end
```

### 8.2 Total Degree

Start system: `gᵢ(x) = xᵢ^{dᵢ} - 1`, start solutions: roots of unity, paths: `∏ dᵢ`.

```julia
function total_degree_start(sys_eval, info; kwargs...)
    start_eval = build_total_degree_evaluator(info.degrees, info.nvars)
    γ = cis(2π * rand())
    homotopy = StraightLineHomotopy(start_eval, sys_eval, γ)
    hom_eval = HomotopyEvaluator(homotopy)
    tracker = Tracker(hom_eval; kwargs...)
    starts = TotalDegreeIterator(info.degrees)
    return tracker, starts
end
```

### 8.3 Polyhedral (BKK)

Two-phase: toric degeneration (binomial start → generic coefficients) then coefficient homotopy (generic → target coefficients).

---

## 9. Dependency Map

```
HomotopyContinuationNext.jl
├── MultivariatePolynomials.jl      # Abstract interface, differentiation, exponent access
├── DynamicPolynomials.jl           # @polyvar, concrete polynomial types, ishomogeneous
├── FunctionWrappers.jl             # Type-stable function erasure
├── FixedSizeArrays.jl              # Non-resizable vectors/matrices (size not in type)
├── MixedSubdivisions.jl            # BKK mixed volume computation
├── LinearAlgebra                   # stdlib
├── Random                          # stdlib
└── ProgressMeter.jl                # Progress bars
```

**What each provides:**

| Package | Functions we use |
|---------|-----------------|
| MultivariatePolynomials | `variables`, `effective_variables`, `terms`, `exponents`, `coefficient`, `coefficients`, `differentiate`, `maxdegree`, `nvariables`, `monomials`, `polynomial` |
| DynamicPolynomials | `@polyvar`, `ishomogeneous`, `homogenize`, `subs` |
| FunctionWrappers | `FunctionWrapper` for SystemEvaluator / HomotopyEvaluator |
| FixedSizeArrays | `FixedSizeArray{T,N,Memory{T}}` via `FSVec{T}`/`FSMat{T}` aliases — all pre-allocated scratch buffers and FW argument types. Size is runtime, not a type parameter. Note: `FixedSizeVector{T}` is NOT concrete. |
| MixedSubdivisions | `mixed_volume`, `fine_mixed_cells` |

**Not in core (future extensions):**
- `Symbolics.jl` — compiled mode, non-polynomial systems
- `Arblib` — certification
- `SemialgebraicSets` — algebraic set interface

---

## 10. Implementation Order

### Phase 1: Primitives
- DoubleF64 / ComplexDF64
- WeightedNorm (infinity norm, adaptive weights)
- MatrixWorkspace, custom LU, Skeel scaling, condition estimation, iterative refinement
- SegmentStepper, utility functions

### Phase 2: Interpreter Pipeline
- OpType enum, scalar operations (`op_add`, `op_mul`, `op_muladd`, etc.)
- MP extraction: `extract_supports()` using `MP.terms`, `MP.exponents`, `MP.coefficient`
- Jacobian: `MP.differentiate(polys, vars)` → extract supports from Jacobian polynomials
- Shared monomial table, IR builder, DAG reorder, register allocation
- `Interpreter{V}`: tape evaluator with `@generated` dispatch
- TruncatedTaylorSeries, TaylorVector, `taylor_op_*` for orders 1-3
- DF64 interpreter (`Interpreter{Vector{ComplexDF64}}`)
- `system_eval()` constructor: MP polynomials → SystemEvaluator
- Test: verify evaluation correctness, Taylor matches finite differences, DF64 precision

### Phase 3: Core Types
- AbstractSystem, AbstractHomotopy interfaces
- SystemEvaluator, HomotopyEvaluator (FunctionWrapper wrappers)
- `SystemEvaluator(F::AbstractSystem)` for user-defined systems
- StraightLineHomotopy, AffineChartHomotopy
- CoefficientHomotopy, ToricHomotopy

### Phase 4: Path Tracking
- NewtonCorrector (α-theory certified Newton)
- Predictor (Padé 2,1 + cubic Hermite)
- Tracker (monomorphic, concrete)
- Valuation (Puiseux series estimation)
- EndgameTracker (singular/infinity detection, Hermite endpoint prediction)
- PathResult (immutable, enum codes)

### Phase 5: Solve
- TotalDegree start system + start solution iterator
- BinomialSystemSolver (HNF-based)
- Polyhedral start system + PolyhedralTracker (two-phase)
- Overdetermined handling (RandomizedSystem, excess solution check)
- VoronoiTree + UniquePoints (deduplication)
- `solve()` orchestration (serial + threaded, progress bars)
- Result types + filtering

### Phase 6: Testing & Validation
- Port test cases from existing codebase
- Benchmark against current HomotopyContinuation.jl (runtime + TTFX)
- JET `@report_opt` on hot paths — zero runtime dispatches
- TTFX target: first `solve(F)` < 5s
- `@compile_workload` for tracker hot path precompilation

Each phase is independently testable. Phase 2 tests evaluation. Phase 4 tests tracking on known paths. Phase 5 integrates everything.
