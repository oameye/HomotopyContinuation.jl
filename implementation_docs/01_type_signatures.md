# Concrete Type Signatures — HomotopyContinuationNext.jl

Implementation reference. All types listed here are concrete (no abstract-typed fields on hot paths).

**Convention:** Pre-allocated buffers with fixed size (known at construction, never resized) use
`FixedSizeArray` from FixedSizeArrays.jl. Size is NOT a type parameter —
`FSVec{ComplexF64}` is the same concrete type regardless of length.

**CRITICAL:** `FixedSizeVector{T}` is NOT a concrete type — its `Mem` type parameter is free.
Using it as a struct field type causes type instability and ~30x performance loss.
Always use the fully concrete aliases below:

```julia
using FixedSizeArrays: FixedSizeArray
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}   # concrete vector type
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}   # concrete matrix type
```

---

## 1. Primitives

### DoubleDouble

```julia
struct DoubleF64 <: AbstractFloat
    hi::Float64
    lo::Float64
end
const ComplexDF64 = Complex{DoubleF64}
```

### Norms

```julia
struct InfNorm end

@kwdef struct WeightedNormOptions
    scale_min::Float64 = 1e-4
    scale_abs_min::Float64 = 1e-6
    scale_max::Float64 = exp2(511)
end

struct WeightedNorm
    weights::FSVec{Float64}
    options::WeightedNormOptions
end
# Always infinity norm. No type parameter needed — only one norm used in practice.
```

### Linear Algebra

```julia
# MUTABLE: lu and qr fields must be reassigned after lu!/qr! (which return new objects).
# factorized/scaled are flags for lazy evaluation.
mutable struct MatrixWorkspace <: AbstractMatrix{ComplexF64}
    const A::FSMat{ComplexF64}           # shared with lu.factors (aliased)
    factorized::Bool
    lu::LA.LU{ComplexF64, FSMat{ComplexF64}, Vector{LinearAlgebra.BlasInt}}
    qr::LA.QR{ComplexF64, FSMat{ComplexF64}, Vector{ComplexF64}}
    const row_scaling::FSVec{Float64}
    scaled::Bool
    const x̄::FSVec{ComplexDF64}          # extended precision workspace
    const r::FSVec{ComplexF64}            # residual
    const r̄::FSVec{ComplexDF64}          # extended precision residual
    const δx::FSVec{ComplexF64}           # correction
    const inf_norm_est_work::FSVec{ComplexF64}
    const inf_norm_est_rwork::FSVec{Float64}
end

struct Jacobian
    workspace::MatrixWorkspace
    factorizations::Base.RefValue{Int}
    ldivs::Base.RefValue{Int}
end
```

### Utils

```julia
# MUTABLE: s, s′ advanced every tracker step
mutable struct SegmentStepper
    const start::ComplexF64
    const target::ComplexF64
    const abs_Δ::Float64
    const forward::Bool
    s::Float64
    s′::Float64
end
```

### VoronoiTree

```julia
# MUTABLE: nentries grows, children lazily assigned
mutable struct VoronoiTreeNode{T,Id}
    nentries::Int
    const values::Matrix{T}
    const ids::Vector{Id}
    const children::Vector{VoronoiTreeNode{T,Id}}
    const distances::Vector{Tuple{Float64,Int}}
end

# MUTABLE: nentries incremented on insertion
mutable struct VoronoiTree{T,Id,M}
    const root::VoronoiTreeNode{T,Id}
    nentries::Int
    const distance::M
    const triangle_inequality::Bool
end
```

### UniquePoints

```julia
struct GroupActions{T<:Tuple}
    actions::T
end

struct UniquePoints{T, Id, M, GA}
    tree::VoronoiTree{T, Id, M}
    group_actions::GA              # Nothing or GroupActions{...}
    zero_vec::FSVec{T}
end
```

### LinearSubspace

```julia
struct ExtrinsicDescription{T}
    A::FSMat{T}
    b::FSVec{T}
end

struct IntrinsicDescription{T}
    A::FSMat{T}
    b::FSVec{T}
    X::FSMat{T}
    Y::FSMat{T}
end

struct LinearSubspace{T}
    extrinsic::ExtrinsicDescription{T}
    intrinsic::IntrinsicDescription{T}
end
```

---

## 2. Model Kit

### Operations

```julia
@enumx OpType::Int8 begin
    OP_STOP
    OP_CB; OP_COS; OP_INV; OP_INV_NOT_ZERO; OP_INVSQR
    OP_NEG; OP_SIN; OP_SQR; OP_SQRT; OP_IDENTITY
    OP_ADD; OP_DIV; OP_MUL; OP_SUB; OP_POW_INT
    OP_ADD3; OP_MUL3; OP_MULADD; OP_MULSUB; OP_SUBMUL
    OP_ADD4; OP_MUL4; OP_MULMULADD; OP_MULMULSUB
end
```

### Instruction Sequence

No separate IR — SExpr trees compile directly to `Instruction` values via `TapeCompiler`.

```julia
struct Instruction
    input::NTuple{4, Int32}
    op::OpType.T
    output::Int32
end

struct InstructionSequence
    instructions::Vector{Instruction}
    constants::Vector{ComplexF64}
    constants_range::UnitRange{Int}
    parameters_range::UnitRange{Int}
    variables_range::UnitRange{Int}
    output_dim::Int
    tape_space_needed::Int
    u_assignments::Vector{Tuple{Int, Int}}
    U_assignments::Vector{Tuple{Int, Int}}
    all_u_assigned::Bool
    all_U_assigned::Bool
end
```

### Taylor

```julia
struct TruncatedTaylorSeries{N,T}
    val::NTuple{N,T}
end

struct TaylorVector{N,T} <: AbstractVector{TruncatedTaylorSeries{N,T}}
    data::FSMat{T}             # N × n matrix; data[k,i] = k-th coefficient of i-th element
end
```

### Interpreter

```julia
struct Interpreter{V<:AbstractVector}
    sequence::InstructionSequence
    tape::V                        # contents mutated via tape[i] = val; reference is fixed
    variables::Vector{Symbol}
    parameters::Vector{Symbol}
end
```

---

## 3. FunctionWrapper Protocol

Convention: `Val{K}` means order K. `TaylorVector{K+1}` stores K+1 coefficients (orders 0..K).

All buffer arguments use `FSVec`/`FSMat` (FixedSizeArrays). These are `DenseArray` subtypes,
so user-defined `AbstractSystem` implementations can index and mutate them normally —
the only difference from `Vector` is that they cannot be resized.

```julia
using FunctionWrappers: FunctionWrapper

# === System FunctionWrapper signatures ===

# evaluate!(u, x, p) → nothing
const SysEvalFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, FSVec{ComplexF64}}}
# evaluate!(u, x_df64, p) → nothing  (extended precision input)
const SysEvalDF64FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexDF64}, FSVec{ComplexF64}}}
# evaluate_and_jacobian!(u, U, x, p) → nothing
const SysEvalJacFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSMat{ComplexF64},
    FSVec{ComplexF64}, FSVec{ComplexF64}}}
# taylor!(u, tx, p) → nothing  (order K: tx is TaylorVector{K+1})
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

# === Homotopy FunctionWrapper signatures ===

# evaluate!(u, x, t) → nothing
const HomEvalFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
# evaluate!(u, x_df64, t) → nothing
const HomEvalDF64FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexDF64}, ComplexF64}}
# evaluate_and_jacobian!(u, U, x, t) → nothing
const HomEvalJacFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSMat{ComplexF64},
    FSVec{ComplexF64}, ComplexF64}}
# taylor! order 1: (u, x, t) — x is plain FSVec (current point only)
const HomTaylor1FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
# taylor! order 2: (u, tx, t, incremental) — tx is TaylorVector{3}
const HomTaylor2FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{3,ComplexF64}, ComplexF64, Bool}}
# taylor! order 3: (u, tx, t, incremental) — tx is TaylorVector{4}
const HomTaylor3FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{4,ComplexF64}, ComplexF64, Bool}}
# set_solution!(x, y, t) → nothing
const HomSetSolFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
# get_solution!(out, x, t) → nothing (writes into out)
const HomGetSolFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
# start_parameters!(p) / target_parameters!(p)
const HomParamsFW = FunctionWrapper{Nothing, Tuple{FSVec{ComplexF64}}}

struct HomotopyEvaluator
    _evaluate!::HomEvalFW
    _evaluate_df64!::HomEvalDF64FW
    _evaluate_and_jacobian!::HomEvalJacFW
    _taylor_1!::HomTaylor1FW        # order 1: plain FSVec x
    _taylor_2!::HomTaylor2FW        # order 2: TaylorVector{3} + incremental
    _taylor_3!::HomTaylor3FW        # order 3: TaylorVector{4} + incremental
    _set_solution!::HomSetSolFW
    _get_solution!::HomGetSolFW
    _start_parameters!::HomParamsFW
    _target_parameters!::HomParamsFW
    _size::Tuple{Int,Int}
end
```

---

## 4. Polynomial System Metadata

No `PolynomialSystem <: AbstractSystem` type. Polynomial systems produce a `SystemEvaluator`
directly via `system_eval()`. Metadata is stored separately:

```julia
struct PolynomialSystemInfo
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Union{Nothing, Vector{Vector{Int}}}
    is_homogeneous::Bool
    # Prevent GC of objects captured by FW closures
    _seq::InstructionSequence
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2,ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3,ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4,ComplexF64}}}
end
```

### Construction from MP polynomials

`system_eval()` produces both `PolynomialSystemInfo` and `SystemEvaluator` directly from
DynamicPolynomials input. No intermediate `AbstractSystem` subtype — no double-wrapping.

MP functions used:
- `MP.effective_variables(polys)` — variables that actually appear
- `MP.terms(p)`, `MP.exponents(t)`, `MP.coefficient(t)` — support extraction
- `MP.differentiate(polys, vars)` — symbolic Jacobian (returns MP polynomials)
- `MP.maxdegree(p)` — degrees for total degree start system
- `MP.ishomogeneous(p)` — homogeneity check (DynamicPolynomials)

---

## 5. Homotopy Types

```julia
struct StraightLineHomotopy <: AbstractHomotopy
    start::SystemEvaluator
    target::SystemEvaluator
    γ::ComplexF64
    # Scratch (all fixed-size, allocated once at construction)
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
    coeffs::FSVec{ComplexF64}                 # mutated contents, not the vector itself
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
    x::FSVec{ComplexF64}
    t_coeffs::Base.RefValue{ComplexF64}       # last cached t (NaN = invalid)
    t_taylor_coeffs::Base.RefValue{ComplexF64}
    taylor_coeffs::TaylorVector{5,ComplexF64}
    tc3::TaylorVector{4,ComplexF64}
    tc2::TaylorVector{3,ComplexF64}
end

struct AffineChartHomotopy <: AbstractHomotopy
    homotopy::HomotopyEvaluator
    chart::FSVec{ComplexF64}
    ndims::Int
end
```

---

## 6. Path Tracking

```julia
@enumx PredictionMethod::Int8 begin
    PADE21
    HERMITE
end

# MUTABLE: ~12 scalar fields (trust_region, t, winding_number, etc.) updated every step.
# Captured by reference in Tracker — cannot use Accessors (closure wouldn't see new copy).
mutable struct Predictor
    method::PredictionMethod
    const order::Int
    use_hermite::Bool
    trust_region::Float64
    local_error::Float64
    cond_H_ẋ::Float64
    const tx⁰::TaylorVector{1,ComplexF64}   # aliased into tx³
    const tx¹::TaylorVector{2,ComplexF64}    # aliased into tx³
    const tx²::TaylorVector{3,ComplexF64}    # aliased into tx³
    const tx³::TaylorVector{4,ComplexF64}
    t::ComplexF64
    tx_norm::NTuple{4,Float64}               # always 4 elements, stack-allocated
    const xtemp::FSVec{ComplexF64}
    const u::FSVec{ComplexF64}
    const u₁::FSVec{ComplexF64}
    const u₂::FSVec{ComplexF64}
    const prev_tx¹::TaylorVector{2,ComplexF64}
    prev_t::ComplexF64
    winding_number::Int
    s::ComplexF64
    prev_s::ComplexF64
    const ty¹::TaylorVector{2,ComplexF64}
    const prev_ty¹::TaylorVector{2,ComplexF64}
end

@enumx NewtonCode::Int8 begin
    NEWT_CONVERGED
    NEWT_TERMINATED
    NEWT_MAX_ITERS
    NEWT_SINGULARITY
end

struct NewtonCorrectorResult
    return_code::NewtonCode
    accuracy::Float64
    iters::Int
    ω::Float64
    θ::Float64
    μ_low::Float64
    norm_Δx₀::Float64
end

struct NewtonCorrector
    a::Float64
    h_a::Float64
    Δx::FSVec{ComplexF64}
    r::FSVec{ComplexF64}
    x_ext::FSVec{ComplexDF64}
end

@enumx TrackerCode::Int8 begin
    TRACKING
    TRACKER_SUCCESS
    TERMINATED_MAX_STEPS
    TERMINATED_ACCURACY_LIMIT
    TERMINATED_ILL_CONDITIONED
    TERMINATED_INVALID_STARTVALUE
    TERMINATED_STEP_SIZE_TOO_SMALL
end

# MUTABLE: ~20 scalar fields (accuracy, ω, μ, τ, counters, flags) all mutated every step.
# Core state machine of the tracker.
mutable struct TrackerState
    const x::FSVec{ComplexF64}       # contents mutated, reference fixed
    const x̂::FSVec{ComplexF64}
    const x̄::FSVec{ComplexF64}
    segment::SegmentStepper          # replaced per segment
    Δs_prev::Float64
    accuracy::Float64
    ω::Float64
    ω_prev::Float64
    μ::Float64
    τ::Float64
    norm_Δx₀::Float64
    extended_prec::Bool
    used_extended_prec::Bool
    keep_extended_prec::Bool
    const norm::WeightedNorm         # weights contents mutated, struct reference fixed
    use_strict_β_τ::Bool
    const jacobian::Jacobian         # workspace contents mutated, reference fixed
    cond_J_ẋ::Float64
    code::TrackerCode
    accepted_steps::Int
    rejected_steps::Int
    last_steps_failed::Int
end

struct Tracker
    homotopy::HomotopyEvaluator
    predictor::Predictor
    corrector::NewtonCorrector
    state::TrackerState
    options::TrackerOptions
end

@kwdef struct TrackerOptions
    max_steps::Int = 10_000
    max_step_size::Float64 = Inf
    max_initial_step_size::Float64 = 0.1
    extended_precision::Bool = true
    min_step_size::Float64 = 0.0
    terminate_cond::Float64 = 1e14
end
```

### Endgame

```julia
@enumx EndgameCode::Int8 begin
    EG_TRACKING
    EG_SUCCESS
    EG_AT_INFINITY
    EG_AT_ZERO
    EG_TERMINATED_ACCURACY
    EG_TERMINATED_ILL_CONDITIONED
    EG_TERMINATED_MAX_STEPS
    EG_TERMINATED_STEP_SIZE
    EG_TERMINATED_INVALID_START
    EG_TERMINATED_MAX_WINDING
    EG_EXCESS_SOLUTION
end

# MUTABLE: logt_data (NTuple{2,Float64}) requires struct-level mutation.
# Vector fields have contents mutated but references are fixed.
mutable struct Valuation
    const val_x::FSVec{Float64}              # contents mutated
    const val_tẋ::FSVec{Float64}
    const Δval_x::FSVec{Float64}
    const Δval_tẋ::FSVec{Float64}
    const val_x_data::NTuple{2, FSVec{Float64}}   # inner contents mutated
    const val_ẋ_data::NTuple{2, FSVec{Float64}}
    const logx_data::NTuple{2, FSVec{Float64}}
    const logẋ_data::NTuple{2, FSVec{Float64}}
    logt_data::NTuple{2, Float64}            # the only field that needs struct mutation
end

# MUTABLE: code, winding_number, accuracy, cond, flags all updated during endgame.
mutable struct EndgameState
    code::EndgameCode
    singular_endgame::Bool
    const val::Valuation
    winding_number::Int            # 0 = unknown
    const solution::FSVec{ComplexF64}    # contents mutated
    accuracy::Float64
    cond::Float64
    singular::Bool
    steps_eg::Int
    # Infinity detection (contents mutated, references fixed)
    const at_infinity_starts::FSVec{Float64}
    const at_infinity_tols::FSVec{Float64}
    const at_infinity_abs_coords::FSVec{Float64}
    # Singular endgame
    const samples::Vector{TaylorVector{2,ComplexF64}}   # grows up to 3
    const sample_times::Vector{Float64}                  # grows up to 3
    const prediction::FSVec{ComplexF64}       # contents mutated
    const prev_prediction::FSVec{ComplexF64}  # contents mutated
end

@kwdef struct EndgameOptions
    endgame_start::Float64 = 0.1
    max_endgame_steps::Int = 2000
    max_endgame_extended_steps::Int = 400
    min_cond::Float64 = 1e6
    min_cond_growth::Float64 = 1e4
    min_coord_growth::Float64 = 100.0
    at_infinity_check::Bool = true
    only_nonsingular::Bool = false
    singular_min_accuracy::Float64 = 1e-6
    max_winding_number::Int = 6
    val_finite_tol::Float64 = 0.05
    val_at_infinity_tol::Float64 = 0.01
    sing_cond::Float64 = 1e14
    sing_accuracy::Float64 = 1e-12
    refine_steps::Int = 3
end

struct EndgameTracker
    tracker::Tracker
    state::EndgameState
    options::EndgameOptions
end
```

### PathResult

```julia
@enumx PathResultCode::Int8 begin
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

struct PathResult
    return_code::PathResultCode
    solution::Vector{ComplexF64}
    t::Float64
    accuracy::Float64
    residual::Float64
    condition_jacobian::Float64
    winding_number::Int
    multiplicity::Int
    singular::Bool
    ω::Float64
    μ::Float64
    accepted_steps::Int
    rejected_steps::Int
end
```

---

## 7. Solve Layer

```julia
struct Result
    path_results::Vector{PathResult}
    tracked_paths::Int
    seed::UInt32
    start_system::Symbol
end

struct TotalDegreeIterator
    degrees::Vector{Int}
    # Iterates over ∏ dᵢ start solutions
end

struct BinomialSolver
    A::Matrix{Int32}
    b::Vector{ComplexF64}
    X::Matrix{ComplexF64}
    H::Matrix{Int64}
    U::Matrix{Int64}
    H_big::Matrix{BigInt}
    U_big::Matrix{BigInt}
end

struct PolyhedralTracker
    toric_tracker::Tracker
    generic_tracker::EndgameTracker
    support::Vector{Matrix{Int32}}
    lifting::Vector{Vector{Int32}}
end

struct OverdeterminedTracker
    tracker::EndgameTracker
    original_system::SystemEvaluator
    newton_cache::NewtonCache
end

struct NewtonCache
    x::FSVec{ComplexF64}
    Δx::FSVec{ComplexF64}
    x_ext::FSVec{ComplexDF64}
    J::MatrixWorkspace
    r::FSVec{ComplexF64}
end
```

---

## 8. Interface Contracts

### AbstractSystem

User-facing interface uses `AbstractVector`/`AbstractMatrix`. Internally the solver passes
`FSVec`/`FSMat`, which are subtypes of `AbstractVector`/`AbstractMatrix` — so user methods
dispatch correctly without importing FixedSizeArrays.

```julia
# Must implement:
Base.size(F::MySystem) -> Tuple{Int,Int}
evaluate!(u::AbstractVector, F::MySystem, x::AbstractVector, p::AbstractVector)
evaluate_and_jacobian!(u::AbstractVector, U::AbstractMatrix,
                       F::MySystem, x::AbstractVector, p::AbstractVector)
# Taylor: Val{K} = order K, tx has K+1 slots (orders 0..K)
taylor!(u::AbstractVector, ::Val{K}, F::MySystem,
        tx::TaylorVector{K+1,ComplexF64}, p::AbstractVector) where K

# Optional (have defaults):
nparameters(F::MySystem) = 0
```

### AbstractHomotopy

```julia
# Must implement:
Base.size(H::MyHomotopy) -> Tuple{Int,Int}
evaluate!(u::AbstractVector, H::MyHomotopy, x::AbstractVector, t::ComplexF64)
evaluate_and_jacobian!(u::AbstractVector, U::AbstractMatrix,
                       H::MyHomotopy, x::AbstractVector, t::ComplexF64)
# Taylor order 1: x is a plain vector (current point)
taylor!(u::AbstractVector, ::Val{1}, H::MyHomotopy,
        x::AbstractVector, t::ComplexF64)
# Taylor order K >= 2: tx is TaylorVector{K+1}, incremental flag
taylor!(u::AbstractVector, ::Val{K}, H::MyHomotopy,
        tx::TaylorVector{K+1,ComplexF64}, t::ComplexF64,
        incremental::Bool=false) where K

# Optional (have defaults):
set_solution!(x, H::MyHomotopy, y, t) = (x .= y)
get_solution!(out, H::MyHomotopy, x, t) = (copyto!(out, x))  # in-place, no allocation
start_parameters!(H::MyHomotopy, p) = H
target_parameters!(H::MyHomotopy, p) = H
```
