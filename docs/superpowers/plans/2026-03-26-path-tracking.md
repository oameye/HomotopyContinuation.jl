# Phase 4 Stage 1: Core Path Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working path tracker that follows homotopy paths from t=1 to t=0 for regular (non-singular) endpoints, using Padé (2,1) prediction and α-theory Newton correction.

**Architecture:** Three components with strict dependency ordering: Newton corrector (standalone) → Predictor (uses Taylor coefficients + Jacobian ldiv) → Tracker (orchestrates predictor + corrector with adaptive step control). All hot-path types are fully concrete with zero allocations. The tracker is monomorphic — it sees only `HomotopyEvaluator` (FunctionWrapper-erased), `Jacobian`, `WeightedNorm`, and `SegmentStepper`.

**Tech Stack:** EnumX.jl (return codes), FixedSizeArrays (FSVec/FSMat buffers), existing primitives (MatrixWorkspace, Jacobian, WeightedNorm, SegmentStepper, DoubleF64)

**Reference:** Algorithms from Timme (2020) "Mixed Precision Path Tracking for Polynomial Homotopy Continuation" (arXiv:1902.02968), ported from HomotopyContinuation.jl v2.

---

## Scope

This plan covers **Stage 1 only**: Newton corrector + Predictor + Tracker. This produces a working `track()` function that can follow paths for non-singular endpoints.

**Deferred to Stage 2:** Valuation, EndgameTracker, PathResult, singular/infinity detection, Hermite endgame.

## File Structure

| File | Responsibility |
|------|---------------|
| `src/tracking/newton_corrector.jl` | `NewtonCode` enum, `NewtonCorrectorResult`, `NewtonCorrector` struct, `newton!` algorithm, `init_newton!` |
| `src/tracking/predictor.jl` | `PredictionMethod` enum, `Predictor` struct, Taylor coefficient computation (`update!`), `predict!` (Padé 2,1) |
| `src/tracking/tracker.jl` | `TrackerCode` enum, `TrackerOptions`, `TrackerState`, `Tracker` struct, `step!`, `track!`, step size control |
| `test/tracking_test.jl` | Tests for all tracking components |
| `benchmark/tracking.jl` | Benchmarks for Newton step, predictor, and full tracking |

---

### Task 1: Newton Corrector — Enums, Structs, Result Type

**Files:**
- Create: `src/tracking/newton_corrector.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)

- [ ] **Step 1: Create the file with enums and struct definitions**

Create `src/tracking/newton_corrector.jl`:

```julia
## Newton corrector — α-theory certified Newton iteration for path tracking.
#
# Based on: Timme (2020), "Mixed Precision Path Tracking for Polynomial
# Homotopy Continuation" (arXiv:1902.02968).

@enumx NewtonCode::Int8 begin
    NEWT_CONVERGED
    NEWT_TERMINATED
    NEWT_MAX_ITERS
    NEWT_SINGULARITY
end

struct NewtonCorrectorResult
    return_code::NewtonCode.T
    accuracy::Float64
    iters::Int
    ω::Float64
    θ::Float64
    μ_low::Float64
    norm_Δx₀::Float64
end

"""
    NewtonCorrector

Pre-allocated workspace for α-theory Newton correction.

- `a`: convergence parameter (typically 0.125)
- `h_a`: derived bound `2a(√(4a²+1) - 2a)`
- `Δx`: Newton step vector (mutated in-place)
- `r`: residual vector (mutated in-place)
- `x_ext`: extended precision workspace (mutated in-place)
"""
struct NewtonCorrector
    a::Float64
    h_a::Float64
    Δx::FSVec{ComplexF64}
    r::FSVec{ComplexF64}
    x_ext::FSVec{ComplexDF64}
end

function NewtonCorrector(a::Float64, n::Int, m::Int)
    h_a = 2a * (sqrt(4a^2 + 1) - 2a)
    return NewtonCorrector(
        a, h_a,
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, n)),
    )
end
```

- [ ] **Step 2: Add include to main module**

In `src/HomotopyContinuationNext.jl`, after the core includes, add:

```julia
include("tracking/newton_corrector.jl")
```

Create the `src/tracking/` directory first.

- [ ] **Step 3: Run format and verify module loads**

```bash
make format
julia --project -e 'using HomotopyContinuationNext; println("OK")'
```

- [ ] **Step 4: Run quality gates**

```bash
make test
```

---

### Task 2: Newton Corrector — The `newton!` Algorithm

**Files:**
- Modify: `src/tracking/newton_corrector.jl`

This is the core numerical algorithm. It performs up to 11 Newton iterations with α-theory convergence monitoring.

- [ ] **Step 1: Implement the `newton!` function**

Append to `src/tracking/newton_corrector.jl`:

```julia
const _NEWTON_MAX_ITERS = 11

"""
    newton!(x̄, NC, H, x₀, t, J, norm; ω, μ, extended_precision, first_correction) → NewtonCorrectorResult

Correct predicted point `x₀` using Newton's method with α-theory convergence guarantees.

Writes the corrected point into `x̄`. The Jacobian `J` workspace is used for
factorization and linear solves. All scratch buffers are in `NC`.

# Arguments
- `x̄::FSVec{ComplexF64}`: output corrected point
- `NC::NewtonCorrector`: workspace (Δx, r, x_ext)
- `H::HomotopyEvaluator`: the homotopy H(x,t)
- `x₀::FSVec{ComplexF64}`: predicted point (input)
- `t::ComplexF64`: current parameter value
- `J::Jacobian`: Jacobian workspace with factorization
- `norm::WeightedNorm`: weighted infinity norm
- `ω::Float64`: Lipschitz constant estimate
- `μ::Float64`: accuracy target / step size limit
- `extended_precision::Bool`: whether to use DF64 refinement
- `first_correction::Bool`: whether this is the very first correction at the start
"""
function newton!(
        x̄::FSVec{ComplexF64},
        NC::NewtonCorrector,
        H::HomotopyEvaluator,
        x₀::FSVec{ComplexF64},
        t::ComplexF64,
        J::Jacobian,
        norm::WeightedNorm;
        ω::Float64,
        μ::Float64,
        extended_precision::Bool = false,
        first_correction::Bool = false,
    )::NewtonCorrectorResult
    a = NC.a
    h_a = NC.h_a
    Δx = NC.Δx
    r = NC.r

    # Initialize: x̄ ← x₀
    copyto!(x̄, x₀)

    ā = a
    norm_Δx_prev = 0.0
    norm_Δx₀ = 0.0
    θ = NaN
    μ_low = NaN

    for i in 0:(_NEWTON_MAX_ITERS - 1)
        # ── Evaluate H and Jacobian at current x̄ ──
        evaluate_and_jacobian!(r, J.workspace.A, H, x̄, t)
        updated!(J)

        # ── Solve J * Δx = r ──
        LA.ldiv!(Δx, J, r, norm)

        norm_Δx = weighted_norm(Δx, norm)

        # ── Singularity check ──
        if isnan(norm_Δx)
            return NewtonCorrectorResult(
                NewtonCode.NEWT_SINGULARITY, μ, i, ω, θ, μ_low, norm_Δx₀,
            )
        end

        # ── Update x̄ ← x̄ - Δx ──
        @inbounds for k in eachindex(x̄)
            x̄[k] -= Δx[k]
        end

        if i == 0
            norm_Δx₀ = norm_Δx
            # Initial check: 0.125 * ‖Δx₀‖ * ω > h_a ⟹ not in convergence basin
            if !first_correction && 0.125 * norm_Δx₀ * ω > h_a
                return NewtonCorrectorResult(
                    NewtonCode.NEWT_TERMINATED, μ, 1, ω, θ, μ_low, norm_Δx₀,
                )
            end
        else
            # ── Update ω and θ ──
            ω = 2 * norm_Δx / (norm_Δx_prev^2)
            θ = norm_Δx / norm_Δx_prev

            # ── Contraction check ──
            if θ > ā
                return NewtonCorrectorResult(
                    NewtonCode.NEWT_TERMINATED, μ, i + 1, ω, θ, μ_low, norm_Δx₀,
                )
            end
        end

        # ── Main convergence criterion ──
        # ω * ‖Δxᵢ‖² < 2μ * √(1 - 2h_a)
        if ω * norm_Δx^2 < 2 * μ * sqrt(1 - 2 * h_a)
            # Converged — do one more evaluation + solve for accuracy estimate
            evaluate_and_jacobian!(r, J.workspace.A, H, x̄, t)
            updated!(J)
            LA.ldiv!(Δx, J, r, norm)

            norm_Δx_next = weighted_norm(Δx, norm)

            if isnan(norm_Δx_next)
                return NewtonCorrectorResult(
                    NewtonCode.NEWT_SINGULARITY, μ, i + 2, ω, θ, μ_low, norm_Δx₀,
                )
            end

            # Check that step decreased
            if norm_Δx_next > sqrt(norm_Δx)
                θ = norm_Δx_next / norm_Δx
                return NewtonCorrectorResult(
                    NewtonCode.NEWT_TERMINATED, μ, i + 2, ω, θ, μ_low, norm_Δx₀,
                )
            end

            # Apply final correction
            @inbounds for k in eachindex(x̄)
                x̄[k] -= Δx[k]
            end

            # Update μ (accuracy estimate)
            if norm_Δx_next > 2μ
                # Recompute from fresh residual
                evaluate!(r, H, x̄, t)
                LA.ldiv!(Δx, J, r)
                μ = weighted_norm(Δx, norm)
            else
                μ = norm_Δx_next
            end

            # Refine ω estimate at first iteration
            if i == 0
                ω_new = 2 * norm_Δx / (norm_Δx_next^2)
                if ω_new < ω
                    ω = ω_new
                else
                    ω = 0.25 * ω
                end
            end

            return NewtonCorrectorResult(
                NewtonCode.NEWT_CONVERGED, μ, i + 2, ω, θ, μ_low, norm_Δx₀,
            )
        end

        norm_Δx_prev = norm_Δx
        if i >= 1
            ā = ā^2  # Quadratic tightening of contraction requirement
        end
    end

    # Exhausted all iterations
    return NewtonCorrectorResult(
        NewtonCode.NEWT_MAX_ITERS, μ, _NEWTON_MAX_ITERS, ω, θ, μ_low, norm_Δx₀,
    )
end
```

- [ ] **Step 2: Implement `init_newton!`**

This function computes initial ω and μ estimates by taking Newton steps from the start point.

Append to `src/tracking/newton_corrector.jl`:

```julia
"""
    init_newton!(x̄, NC, H, x₀, t, J, norm) → (valid::Bool, ω::Float64, μ::Float64)

Compute initial Lipschitz estimate ω and accuracy μ for the start point.
Returns `(false, ω, μ)` if the start point is not a valid solution of H(x,t)≈0.
"""
function init_newton!(
        x̄::FSVec{ComplexF64},
        NC::NewtonCorrector,
        H::HomotopyEvaluator,
        x₀::FSVec{ComplexF64},
        t::ComplexF64,
        J::Jacobian,
        norm::WeightedNorm,
    )::Tuple{Bool, Float64, Float64}
    a = NC.a
    Δx = NC.Δx
    r = NC.r

    # First Newton step from x₀
    evaluate_and_jacobian!(r, J.workspace.A, H, x₀, t)
    updated!(J)
    LA.ldiv!(Δx, J, r, norm)

    norm_Δx₀ = weighted_norm(Δx, norm)
    if isnan(norm_Δx₀)
        return (false, 1.0, eps())
    end

    # Apply step: x̄ = x₀ - Δx
    @inbounds for k in eachindex(x̄)
        x̄[k] = x₀[k] - Δx[k]
    end

    # Second Newton step
    evaluate_and_jacobian!(r, J.workspace.A, H, x̄, t)
    updated!(J)
    LA.ldiv!(Δx, J, r, norm)

    norm_Δx₁ = weighted_norm(Δx, norm)
    if isnan(norm_Δx₁) || norm_Δx₁ >= norm_Δx₀
        return (false, 1.0, eps())
    end

    # Apply step: x̄ = x̄ - Δx
    @inbounds for k in eachindex(x̄)
        x̄[k] -= Δx[k]
    end

    # Compute initial estimates
    ω = 2 * norm_Δx₁ / (norm_Δx₀^2)
    μ = norm_Δx₁

    # Validate: ω * μ should be moderate
    if ω * μ > a^5
        # Try tighter correction
        result = newton!(
            x̄, NC, H, x̄, t, J, norm;
            ω = ω, μ = a^7 / ω, extended_precision = false, first_correction = true,
        )
        if result.return_code == NewtonCode.NEWT_CONVERGED
            ω = result.ω
            μ = result.accuracy
        else
            return (false, ω, μ)
        end
    end

    return (true, max(ω, 0.1), max(μ, eps()))
end
```

- [ ] **Step 3: Format and verify**

```bash
make format
julia --project -e 'using HomotopyContinuationNext; println("OK")'
```

- [ ] **Step 4: Run quality gates**

```bash
make test
```

---

### Task 3: Predictor — Struct and Taylor Coefficient Computation

**Files:**
- Create: `src/tracking/predictor.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)

- [ ] **Step 1: Create predictor.jl with struct and Taylor update**

Create `src/tracking/predictor.jl`:

```julia
## Predictor — Padé (2,1) and Hermite prediction for path tracking.
#
# Computes Taylor coefficients of the solution path x(t) up to order 3
# using implicit differentiation of H(x(t), t) = 0, then forms a
# Padé (2,1) rational approximant for prediction.

@enumx PredictionMethod::Int8 begin
    PADE21
    HERMITE
end

"""
    Predictor

Mutable state for the path predictor. Stores Taylor coefficients and
prediction metadata.

**Mutable justification:** ~12 scalar fields (trust_region, t, winding_number,
etc.) are updated every tracker step. Buffer fields are `const`.
"""
mutable struct Predictor
    method::PredictionMethod.T
    const order::Int
    use_hermite::Bool
    trust_region::Float64
    local_error::Float64
    cond_H_ẋ::Float64
    const tx⁰::TaylorVector{1, ComplexF64}  # aliased view into tx³
    const tx¹::TaylorVector{2, ComplexF64}   # aliased view into tx³
    const tx²::TaylorVector{3, ComplexF64}   # aliased view into tx³
    const tx³::TaylorVector{4, ComplexF64}
    t::ComplexF64
    tx_norm::NTuple{4, Float64}
    const xtemp::FSVec{ComplexF64}
    const u::FSVec{ComplexF64}
    const u₁::FSVec{ComplexF64}
    const u₂::FSVec{ComplexF64}
    const prev_tx¹::TaylorVector{2, ComplexF64}
    prev_t::ComplexF64
    winding_number::Int
    s::ComplexF64
    prev_s::ComplexF64
    const ty¹::TaylorVector{2, ComplexF64}
    const prev_ty¹::TaylorVector{2, ComplexF64}
end

function Predictor(m::Int, n::Int)
    # tx³ is the master storage; tx⁰, tx¹, tx² are aliased views
    tx³ = TaylorVector{4, ComplexF64}(n)
    # Create aliased TaylorVectors sharing the same underlying data
    # tx⁰ sees row 1, tx¹ sees rows 1:2, tx² sees rows 1:3
    data = tx³.data
    tx⁰ = TaylorVector{1, ComplexF64}(FSMat{ComplexF64}(view(data, 1:1, :) |> collect |> FSMat{ComplexF64}))
    tx¹ = TaylorVector{2, ComplexF64}(FSMat{ComplexF64}(view(data, 1:2, :) |> collect |> FSMat{ComplexF64}))
    tx² = TaylorVector{3, ComplexF64}(FSMat{ComplexF64}(view(data, 1:3, :) |> collect |> FSMat{ComplexF64}))

    return Predictor(
        PredictionMethod.PADE21,
        4,          # order
        true,       # use_hermite
        Inf,        # trust_region
        NaN,        # local_error
        NaN,        # cond_H_ẋ
        tx⁰, tx¹, tx², tx³,
        complex(NaN),                               # t
        (NaN, NaN, NaN, NaN),                       # tx_norm
        FSVec{ComplexF64}(zeros(ComplexF64, n)),     # xtemp
        FSVec{ComplexF64}(zeros(ComplexF64, m)),     # u
        FSVec{ComplexF64}(zeros(ComplexF64, m)),     # u₁
        FSVec{ComplexF64}(zeros(ComplexF64, m)),     # u₂
        TaylorVector{2, ComplexF64}(n),              # prev_tx¹
        complex(NaN),                               # prev_t
        1,                                           # winding_number
        complex(NaN),                               # s
        complex(NaN),                               # prev_s
        TaylorVector{2, ComplexF64}(n),              # ty¹
        TaylorVector{2, ComplexF64}(n),              # prev_ty¹
    )
end
```

**Important note on aliasing:** The TaylorVector aliasing is an optimization from v2 — tx⁰, tx¹, tx² share memory with tx³ so that writing x[k] coefficients into tx³ automatically populates the smaller views. However, constructing FSMat from views requires care. The implementer should check whether simple independent allocations work first (with manual copying), and optimize aliasing later if needed for performance. For a correct initial implementation, use **independent TaylorVectors** and copy values in `update!`.

- [ ] **Step 2: Implement `update!` — compute Taylor coefficients**

The key algorithm: differentiate H(x(t),t)=0 implicitly to get dx/dt, d²x/dt², d³x/dt³.

Append to `src/tracking/predictor.jl`:

```julia
"""
    update!(pred, H, x, t, J, norm) → nothing

Compute Taylor coefficients of the path x(t) at the current point.
Stores x⁰=x, x¹=ẋ, x²=ẍ/2, x³=x⃛/6 into the predictor's TaylorVectors.

Uses the implicit function theorem: H(x(t),t)=0 ⟹
  H_x·ẋ + H_t = 0 ⟹ ẋ = -H_x⁻¹·H_t

Higher derivatives obtained by differentiating again and using the
homotopy's `taylor!` method.
"""
function update!(
        pred::Predictor,
        H::HomotopyEvaluator,
        x::FSVec{ComplexF64},
        t::ComplexF64,
        J::Jacobian,
        norm::WeightedNorm,
    )::Nothing
    u = pred.u
    xtemp = pred.xtemp
    n = length(x)

    # Save previous values for Hermite
    if !isnan(pred.t)
        # Copy current tx¹ → prev_tx¹
        @inbounds for i in 1:n
            pred.prev_tx¹[i] = pred.tx¹[i]
        end
        pred.prev_t = pred.t
        pred.prev_s = pred.s
    end
    pred.t = t

    # ── Order 0: store x ──
    @inbounds for i in 1:n
        pred.tx³.data[1, i] = x[i]
    end
    pred.tx_norm = (weighted_norm(x, norm), pred.tx_norm[2], pred.tx_norm[3], pred.tx_norm[4])

    # ── Order 1: ẋ = -J⁻¹ · taylor!(u, Val(1), H, x, t) ──
    # taylor!(u, Val(1), H, x, t) computes ∂H/∂t
    taylor!(u, Val(1), H, x, t)
    @inbounds for i in eachindex(u)
        u[i] = -u[i]
    end
    LA.ldiv!(xtemp, J, u)
    @inbounds for i in 1:n
        pred.tx³.data[2, i] = xtemp[i]
    end
    norm_x1 = weighted_norm(xtemp, norm)
    pred.tx_norm = (pred.tx_norm[1], norm_x1, pred.tx_norm[3], pred.tx_norm[4])

    # Condition number
    pred.cond_H_ẋ = LA.cond(J)

    # ── Order 2: x² via taylor!(u, Val(2), H, tx¹, t) ──
    # Need tx¹ populated with [x, ẋ] — copy from tx³ rows 1:2
    @inbounds for i in 1:n
        pred.tx¹.data[1, i] = pred.tx³.data[1, i]
        pred.tx¹.data[2, i] = pred.tx³.data[2, i]
    end
    taylor!(u, Val(2), H, pred.tx¹, t)
    @inbounds for i in eachindex(u)
        u[i] = -u[i]
    end
    LA.ldiv!(xtemp, J, u)
    @inbounds for i in 1:n
        pred.tx³.data[3, i] = xtemp[i]
    end
    norm_x2 = weighted_norm(xtemp, norm)
    pred.tx_norm = (pred.tx_norm[1], pred.tx_norm[2], norm_x2, pred.tx_norm[4])

    # ── Order 3: x³ via taylor!(u, Val(3), H, tx², t) ──
    @inbounds for i in 1:n
        pred.tx².data[1, i] = pred.tx³.data[1, i]
        pred.tx².data[2, i] = pred.tx³.data[2, i]
        pred.tx².data[3, i] = pred.tx³.data[3, i]
    end
    taylor!(u, Val(3), H, pred.tx², t)
    @inbounds for i in eachindex(u)
        u[i] = -u[i]
    end
    LA.ldiv!(xtemp, J, u)
    @inbounds for i in 1:n
        pred.tx³.data[4, i] = xtemp[i]
    end
    norm_x3 = weighted_norm(xtemp, norm)
    pred.tx_norm = (pred.tx_norm[1], pred.tx_norm[2], pred.tx_norm[3], norm_x3)

    # ── Compute trust region ──
    _compute_trust_region!(pred)

    return nothing
end
```

- [ ] **Step 3: Implement trust region and local error computation**

Append to `src/tracking/predictor.jl`:

```julia
"""
    _compute_trust_region!(pred) → nothing

Compute the trust region radius τ from the Taylor coefficient norms.
τ ≈ min_i |x²[i] / x³[i]| — the radius where the 3rd-order term becomes significant.
"""
function _compute_trust_region!(pred::Predictor)::Nothing
    n = size(pred.tx³.data, 2)
    τ = Inf
    tol = 1.0e-14

    @inbounds for i in 1:n
        c1 = abs(pred.tx³.data[2, i])  # |x¹|
        c2 = abs(pred.tx³.data[3, i])  # |x²|
        c3 = abs(pred.tx³.data[4, i])  # |x³|

        λ = max(1.0e-6, c1)
        c1n = c1 / λ
        c2n = c2 / λ^2
        c3n = c3 / λ^3

        thresh = tol * max(c1n, c2n, c3n)
        if c1n <= thresh && c2n <= thresh && c3n <= thresh
            continue
        end
        if c2n <= thresh
            continue
        end
        if c3n > thresh
            τ_i = c2n / c3n / λ
            τ = min(τ, τ_i)
        end
    end

    # Fallback
    if !isfinite(τ)
        τ = pred.tx_norm[3] > 0 && pred.tx_norm[4] > 0 ?
            pred.tx_norm[3] / pred.tx_norm[4] :
            pred.tx_norm[1] / max(pred.tx_norm[1], pred.tx_norm[2], pred.tx_norm[3], pred.tx_norm[4])
    end

    pred.trust_region = isfinite(τ) ? τ : 1.0
    return nothing
end

"""
    _compute_local_error!(pred, x̂, norm, Δs) → nothing

Compute the local prediction error from the difference between prediction and correction.
"""
function _compute_local_error!(
        pred::Predictor,
        x̂::FSVec{ComplexF64},
        x::FSVec{ComplexF64},
        norm::WeightedNorm,
        Δs::Float64,
    )::Nothing
    if isnan(Δs) || Δs == 0.0
        pred.local_error = NaN
        return nothing
    end
    err = weighted_distance(x, x̂, norm)
    pred.local_error = err / Δs^pred.order
    return nothing
end
```

- [ ] **Step 4: Implement `predict!` — Padé (2,1) prediction**

Append to `src/tracking/predictor.jl`:

```julia
"""
    predict!(x̂, pred, Δt) → nothing

Compute predicted point x̂ = x(t + Δt) using Padé (2,1) rational approximation.

For each component i:
  δᵢ = 1 - Δt · x³[i] / x²[i]
  x̂[i] = x[i] + Δt · (x¹[i] + Δt · x²[i] / δᵢ)

Falls back to quadratic Taylor when x³ or x² is negligible.
"""
function predict!(
        x̂::FSVec{ComplexF64},
        pred::Predictor,
        Δt::ComplexF64,
    )::Nothing
    n = length(x̂)
    data = pred.tx³.data
    tol = 1.0e-14

    @inbounds for i in 1:n
        x0 = data[1, i]
        x1 = data[2, i]
        x2 = data[3, i]
        x3 = data[4, i]

        c2 = abs(x2)
        c3 = abs(x3)

        if c3 < tol || c2 < tol
            # Quadratic Taylor: x̂ = x + Δt*x¹ + Δt²*x²
            x̂[i] = x0 + Δt * (x1 + Δt * x2)
        else
            # Padé (2,1): x̂ = x + Δt*(x¹ + Δt*x²/(1 - Δt*x³/x²))
            δ = 1 - Δt * x3 / x2
            if abs(δ) < tol
                x̂[i] = x0 + Δt * (x1 + Δt * x2)
            else
                x̂[i] = x0 + Δt * (x1 + Δt * x2 / δ)
            end
        end
    end

    return nothing
end
```

- [ ] **Step 5: Add include and format**

In `src/HomotopyContinuationNext.jl`, after the Newton include:

```julia
include("tracking/predictor.jl")
```

```bash
make format
julia --project -e 'using HomotopyContinuationNext; println("OK")'
make test
```

---

### Task 4: Tracker — Enums, Options, State, Struct

**Files:**
- Create: `src/tracking/tracker.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)

- [ ] **Step 1: Create tracker.jl with enums, options, state**

Create `src/tracking/tracker.jl`:

```julia
## Tracker — core path tracking state machine.
#
# Orchestrates predictor-corrector steps with adaptive step size control.
# All types are fully concrete — the tracker is monomorphic over any homotopy.

@enumx TrackerCode::Int8 begin
    TRACKING
    TRACKER_SUCCESS
    TERMINATED_MAX_STEPS
    TERMINATED_ACCURACY_LIMIT
    TERMINATED_ILL_CONDITIONED
    TERMINATED_INVALID_STARTVALUE
    TERMINATED_STEP_SIZE_TOO_SMALL
end

@kwdef struct TrackerOptions
    max_steps::Int = 10_000
    max_step_size::Float64 = Inf
    max_initial_step_size::Float64 = 0.1
    extended_precision::Bool = true
    min_step_size::Float64 = 0.0
    terminate_cond::Float64 = 1.0e14
    a::Float64 = 0.125
    β_ω::Float64 = 3.0
    β_τ::Float64 = 0.4
    strict_β_τ::Float64 = 0.3
end

"""
    TrackerState

Core mutable state of the path tracker. All buffer fields are `const`;
scalar fields are updated every step.

**Mutable justification:** ~20 scalar fields (accuracy, ω, μ, etc.) are
updated on every tracker step. Immutable would require reconstructing the
entire struct each step.
"""
mutable struct TrackerState
    const x::FSVec{ComplexF64}
    const x̂::FSVec{ComplexF64}
    const x̄::FSVec{ComplexF64}
    segment::SegmentStepper
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
    const norm::WeightedNorm
    use_strict_β_τ::Bool
    const jacobian::Jacobian
    cond_J_ẋ::Float64
    code::TrackerCode.T
    accepted_steps::Int
    rejected_steps::Int
    last_steps_failed::Int
end

function TrackerState(m::Int, n::Int, segment::SegmentStepper)
    return TrackerState(
        FSVec{ComplexF64}(zeros(ComplexF64, n)),     # x
        FSVec{ComplexF64}(zeros(ComplexF64, n)),     # x̂
        FSVec{ComplexF64}(zeros(ComplexF64, n)),     # x̄
        segment,
        0.0,           # Δs_prev
        eps(),         # accuracy
        1.0,           # ω
        1.0,           # ω_prev
        eps(),         # μ
        Inf,           # τ
        NaN,           # norm_Δx₀
        false,         # extended_prec
        false,         # used_extended_prec
        false,         # keep_extended_prec
        WeightedNorm(n),
        false,         # use_strict_β_τ
        Jacobian(MatrixWorkspace(m, n)),
        NaN,           # cond_J_ẋ
        TrackerCode.TRACKING,
        0, 0, 0,       # accepted, rejected, last_steps_failed
    )
end

struct Tracker
    homotopy::HomotopyEvaluator
    predictor::Predictor
    corrector::NewtonCorrector
    state::TrackerState
    options::TrackerOptions
end

function Tracker(
        H::HomotopyEvaluator;
        start::ComplexF64 = complex(1.0),
        target::ComplexF64 = complex(0.0),
        options::TrackerOptions = TrackerOptions(),
    )
    m, n = size(H)
    segment = SegmentStepper(start, target)
    return Tracker(
        H,
        Predictor(m, n),
        NewtonCorrector(options.a, n, m),
        TrackerState(m, n, segment),
        options,
    )
end
```

- [ ] **Step 2: Add include**

```julia
include("tracking/tracker.jl")
```

- [ ] **Step 3: Format and verify**

```bash
make format && make test
```

---

### Task 5: Tracker — Step Size Control and `step!`

**Files:**
- Modify: `src/tracking/tracker.jl`

- [ ] **Step 1: Implement step size computation**

Append to `src/tracking/tracker.jl`:

```julia
## ── Step size control ────────────────────────────────────────────────────────

function _h(a::Float64)::Float64
    return 2a * (sqrt(4a^2 + 1) - 2a)
end

"""
    _compute_initial_stepsize(state, pred, opts) → Float64

Compute the first step size from ω, trust region, and local error.
"""
function _compute_initial_stepsize(
        state::TrackerState, pred::Predictor, opts::TrackerOptions,
    )::Float64
    a = opts.a
    p = pred.order
    ω = state.ω
    e = pred.local_error
    τ = pred.trust_region

    # ω-based step size
    if isfinite(e) && e > 0 && isfinite(ω)
        Δs₁ = nthroot((sqrt(1 + 2 * _h(a)) - 1) / (ω * e), p) / opts.β_ω
    else
        Δs₁ = Inf
    end

    # Trust-region-based step size
    Δs₂ = opts.β_τ * τ

    Δs = nanmin(Δs₁, Δs₂)
    Δs = min(Δs, opts.max_step_size, opts.max_initial_step_size)
    return max(Δs, opts.min_step_size)
end

"""
    _update_stepsize!(state, result, pred, opts) → nothing

Adapt step size after a predictor-corrector step.
"""
function _update_stepsize!(
        state::TrackerState,
        result::NewtonCorrectorResult,
        pred::Predictor,
        opts::TrackerOptions,
    )::Nothing
    a = opts.a
    p = pred.order

    if result.return_code == NewtonCode.NEWT_CONVERGED
        ω = state.ω
        e = pred.local_error
        τ = pred.trust_region

        # ω-based
        if isfinite(e) && e > 0 && isfinite(ω)
            Δs₁ = nthroot((sqrt(1 + 2 * _h(a)) - 1) / (ω * e), p) / opts.β_ω
        else
            Δs₁ = Inf
        end

        # Trust-region based
        β_τ = state.use_strict_β_τ || dist_to_target(state.segment) < opts.β_τ * τ ?
            opts.strict_β_τ : opts.β_τ
        Δs₂ = β_τ * τ

        Δs = min(nanmin(Δs₁, Δs₂), opts.max_step_size)

        # Limit growth to 10x
        if state.Δs_prev > 0
            Δs = min(Δs, 10 * state.Δs_prev)
        end

        # After failure, don't increase
        if state.last_steps_failed > 0
            Δs = min(Δs, state.Δs_prev)
        end
    else
        # Step rejected — reduce
        Δs = 0.25 * abs(state.segment.Δs)
    end

    propose_step!(state.segment, max(Δs, opts.min_step_size))
    return nothing
end
```

- [ ] **Step 2: Implement `_check_terminated!`**

Append to `src/tracking/tracker.jl`:

```julia
"""
    _check_terminated!(state, opts) → nothing

Check all termination conditions and update `state.code` if any are met.
"""
function _check_terminated!(state::TrackerState, opts::TrackerOptions)::Nothing
    if is_done(state.segment)
        state.code = TrackerCode.TRACKER_SUCCESS
    elseif state.accepted_steps + state.rejected_steps >= opts.max_steps
        state.code = TrackerCode.TERMINATED_MAX_STEPS
    elseif state.ω * state.μ > _h(opts.a)
        state.code = TrackerCode.TERMINATED_ACCURACY_LIMIT
    end
    return nothing
end
```

- [ ] **Step 3: Implement `step!`**

Append to `src/tracking/tracker.jl`:

```julia
"""
    step!(tracker) → Bool

Perform one predictor-corrector step. Returns `true` if the step was accepted.
"""
function step!(tracker::Tracker)::Bool
    state = tracker.state
    pred = tracker.predictor
    H = tracker.homotopy
    opts = tracker.options

    t = state.segment.t
    t′ = state.segment.t′
    Δt = state.segment.Δt

    # ── Predict ──
    predict!(state.x̂, pred, Δt)
    update!(state.norm, state.x̂)

    # ── Newton correct ──
    result = newton!(
        state.x̄, tracker.corrector, H, state.x̂, t′,
        state.jacobian, state.norm;
        ω = state.ω,
        μ = state.μ,
        extended_precision = state.extended_prec,
        first_correction = state.accepted_steps == 0,
    )

    accepted = result.return_code == NewtonCode.NEWT_CONVERGED

    if accepted
        # ── Accept step ──
        state.Δs_prev = abs(state.segment.Δs)
        step_success!(state.segment)

        # Update state
        copyto!(state.x, state.x̄)
        state.accuracy = result.accuracy
        state.μ = max(result.accuracy, eps())
        state.ω_prev = state.ω
        state.ω = max(result.ω, 0.5 * state.ω, 0.1)
        state.norm_Δx₀ = result.norm_Δx₀
        state.cond_J_ẋ = pred.cond_H_ẋ

        # Compute local error from prediction vs correction
        _compute_local_error!(pred, state.x̂, state.x, state.norm, state.Δs_prev)

        # Update predictor for next step (compute Taylor coefficients at new point)
        update!(pred, H, state.x, state.segment.t, state.jacobian, state.norm)

        state.τ = pred.trust_region
        state.accepted_steps += 1
        state.last_steps_failed = 0
    else
        # ── Reject step ──
        state.rejected_steps += 1
        state.last_steps_failed += 1
    end

    # ── Adapt step size ──
    _update_stepsize!(state, result, pred, opts)

    # ── Check termination ──
    _check_terminated!(state, opts)

    return accepted
end
```

- [ ] **Step 4: Format and verify**

```bash
make format && make test
```

---

### Task 6: Tracker — `init!` and `track!`

**Files:**
- Modify: `src/tracking/tracker.jl`

- [ ] **Step 1: Implement `init!`**

Append to `src/tracking/tracker.jl`:

```julia
"""
    init!(tracker, x₀, t₁, t₀) → TrackerCode.T

Initialize the tracker at start point `x₀` with path from `t₁` to `t₀`.
Computes initial ω, μ, step size, and Taylor coefficients.
Returns the initial tracker code (TRACKING or an error code).
"""
function init!(
        tracker::Tracker,
        x₀::AbstractVector{ComplexF64},
        t₁::ComplexF64 = complex(1.0),
        t₀::ComplexF64 = complex(0.0),
    )::TrackerCode.T
    state = tracker.state
    pred = tracker.predictor
    opts = tracker.options

    m, n = size(tracker.homotopy)

    # Reset state
    state.segment = SegmentStepper(t₁, t₀)
    copyto!(state.x, x₀)
    state.Δs_prev = 0.0
    state.accuracy = eps()
    state.ω = 1.0
    state.ω_prev = 1.0
    state.μ = eps()
    state.τ = Inf
    state.extended_prec = false
    state.used_extended_prec = false
    state.keep_extended_prec = false
    state.use_strict_β_τ = false
    state.code = TrackerCode.TRACKING
    state.accepted_steps = 0
    state.rejected_steps = 0
    state.last_steps_failed = 0

    # Reset predictor
    pred.t = complex(NaN)
    pred.prev_t = complex(NaN)
    pred.winding_number = 1
    pred.local_error = NaN

    # Initialize norm weights
    init!(state.norm, state.x)
    init!(state.jacobian)

    # Compute initial Newton estimates
    valid, ω, μ = init_newton!(
        state.x̄, tracker.corrector, tracker.homotopy, state.x, t₁,
        state.jacobian, state.norm,
    )

    if !valid
        state.code = TrackerCode.TERMINATED_INVALID_STARTVALUE
        return state.code
    end

    # Accept the corrected start point
    copyto!(state.x, state.x̄)
    state.ω = ω
    state.ω_prev = ω
    state.μ = μ

    # Compute initial Taylor coefficients
    update!(pred, tracker.homotopy, state.x, t₁, state.jacobian, state.norm)
    state.τ = pred.trust_region
    state.cond_J_ẋ = pred.cond_H_ẋ

    # Compute initial step size
    Δs = _compute_initial_stepsize(state, pred, opts)
    propose_step!(state.segment, Δs)

    return state.code
end
```

- [ ] **Step 2: Implement `track!`**

Append to `src/tracking/tracker.jl`:

```julia
"""
    track!(tracker, x₀; t₁=1.0, t₀=0.0) → TrackerCode.T

Track the path starting at `x₀` from `t₁` to `t₀`. The solution is
stored in `tracker.state.x`. Returns the final tracker code.
"""
function track!(
        tracker::Tracker,
        x₀::AbstractVector{ComplexF64};
        t₁::ComplexF64 = complex(1.0),
        t₀::ComplexF64 = complex(0.0),
    )::TrackerCode.T
    code = init!(tracker, x₀, t₁, t₀)
    if code != TrackerCode.TRACKING
        return code
    end

    while tracker.state.code == TrackerCode.TRACKING
        step!(tracker)
    end

    return tracker.state.code
end
```

- [ ] **Step 3: Format and verify**

```bash
make format && make test
```

---

### Task 7: Integration Tests

**Files:**
- Create: `test/tracking_test.jl`

- [ ] **Step 1: Write the test file**

Create `test/tracking_test.jl`:

```julia
using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: system_eval, StraightLineHomotopy, HomotopyEvaluator,
    evaluate!, evaluate_and_jacobian!, taylor!,
    NewtonCorrector, NewtonCode, NewtonCorrectorResult, newton!, init_newton!,
    Predictor, PredictionMethod, predict!, update! as predictor_update!,
    Tracker, TrackerCode, TrackerOptions, TrackerState, track!, step!, init!,
    Jacobian, MatrixWorkspace, WeightedNorm, SegmentStepper,
    TaylorVector, TruncatedTaylorSeries, weighted_norm
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using LinearAlgebra: LinearAlgebra as LA

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

@testset "Path Tracking" begin

    # ── Newton Corrector ──────────────────────────────────────────────────

    @testset "NewtonCorrector: converges on known solution" begin
        @polyvar x y
        # System: x^2-1, y^2-1. Solution at (1,1).
        G = [x - 1, y - 1]
        F = [x^2 - 1, y^2 - 1]
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)

        m, n = size(heval)
        NC = NewtonCorrector(0.125, n, m)
        J = Jacobian(MatrixWorkspace(m, n))
        norm = WeightedNorm(n)

        # At t=0, H = F. Solution of F at (1,1).
        x₀ = FSVec{ComplexF64}(ComplexF64[1.01, 0.99])  # Near solution
        x̄ = FSVec{ComplexF64}(zeros(ComplexF64, n))
        t = ComplexF64(0.0)

        init!(norm, x₀)

        # Evaluate Jacobian at x₀ first
        r = FSVec{ComplexF64}(zeros(ComplexF64, m))
        evaluate_and_jacobian!(r, J.workspace.A, heval, x₀, t)
        HC.updated!(J)

        result = newton!(
            x̄, NC, heval, x₀, t, J, norm;
            ω = 1.0, μ = 0.1, first_correction = true,
        )

        @test result.return_code == NewtonCode.NEWT_CONVERGED
        @test abs(x̄[1] - 1.0) < 1.0e-10
        @test abs(x̄[2] - 1.0) < 1.0e-10
    end

    @testset "NewtonCorrector: zero allocations" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x^2 - 1, y^2 - 1]
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)
        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)

        m, n = 2, 2
        NC = NewtonCorrector(0.125, n, m)
        J = Jacobian(MatrixWorkspace(m, n))
        norm = WeightedNorm(n)
        x₀ = FSVec{ComplexF64}(ComplexF64[1.01, 0.99])
        x̄ = FSVec{ComplexF64}(zeros(ComplexF64, n))
        t = ComplexF64(0.0)
        init!(norm, x₀)

        # Warmup
        newton!(x̄, NC, heval, x₀, t, J, norm; ω = 1.0, μ = 0.1, first_correction = true)

        allocs = @allocated newton!(x̄, NC, heval, x₀, t, J, norm; ω = 1.0, μ = 0.1, first_correction = true)
        @test allocs == 0
    end

    # ── Tracker: end-to-end ───────────────────────────────────────────────

    @testset "track!: linear system (trivial path)" begin
        @polyvar x y
        G = [x - 1, y - 1]       # Start: solution at (1,1)
        F = [x - 2, y - 3]       # Target: solution at (2,3)
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)
        tracker = Tracker(heval)

        x₀ = ComplexF64[1.0, 1.0]
        code = track!(tracker, x₀)

        @test code == TrackerCode.TRACKER_SUCCESS
        @test abs(tracker.state.x[1] - 2.0) < 1.0e-8
        @test abs(tracker.state.x[2] - 3.0) < 1.0e-8
    end

    @testset "track!: quadratic system" begin
        @polyvar x y
        F = [x^2 + y - 1, x * y - 0.5]  # Target system
        G = [x^2 - 1, y^2 - 1]           # Start system (total degree)
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)
        tracker = Tracker(heval)

        # Start at a solution of G: (1, 1)
        x₀ = ComplexF64[1.0, 1.0]
        code = track!(tracker, x₀)

        if code == TrackerCode.TRACKER_SUCCESS
            sol = tracker.state.x
            # Verify it's a solution of F
            u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            evaluate!(u, eval_F, sol, FSVec{ComplexF64}(ComplexF64[]))
            @test abs(u[1]) < 1.0e-6
            @test abs(u[2]) < 1.0e-6
        end

        # The path should either succeed or terminate — not infinite loop
        @test tracker.state.accepted_steps + tracker.state.rejected_steps ≤ 10_000
    end

    @testset "track!: invalid start value detected" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x^2 - 1, y^2 - 1]
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)
        tracker = Tracker(heval)

        # Start at a point that is NOT a solution of G at t=1
        x_bad = ComplexF64[100.0, 100.0]
        code = track!(tracker, x_bad)

        # Should detect invalid start or terminate early
        @test code != TrackerCode.TRACKING
    end
end
```

- [ ] **Step 2: Run the tests**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/tracking_test.jl")'
```

Expect: at minimum the linear system test should pass. The quadratic test validates that we can find a genuine solution. Iterate on failures — the algorithm may need tuning of tolerances or the `evaluate!` calls in Newton may need the 2-arg homotopy form.

- [ ] **Step 3: Debug and fix until tests pass**

Common issues to watch for:
- `evaluate!` on `HomotopyEvaluator` takes `(u, H, x, t)` not `(u, H, x, t, p)` — no parameter arg
- `taylor!` on `HomotopyEvaluator` Val(1) takes `(u, Val(1), H, x, t)` with plain `FSVec` x, not TaylorVector
- `taylor!` on `SystemEvaluator` Val(K) takes `(u, Val(K), S, tx, p)` with TaylorVector and params
- The predictor `update!` calls `taylor!(u, Val(K), H, tx, t)` on the **homotopy** evaluator (not system evaluator)
- Ensure the predictor Taylor calls match the homotopy's `taylor!` signatures from `homotopy_evaluator.jl`

- [ ] **Step 4: Run full test suite**

```bash
make format && make test
```

---

### Task 8: Benchmarks

**Files:**
- Create: `benchmark/tracking.jl`
- Modify: `benchmark/runbenchmarks.jl`

- [ ] **Step 1: Write tracking benchmarks**

Create `benchmark/tracking.jl`:

```julia
using BenchmarkTools
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using HomotopyContinuationNext:
    system_eval, StraightLineHomotopy, HomotopyEvaluator,
    Tracker, TrackerOptions, track!

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

function benchmark_tracking!(SUITE::BenchmarkGroup)
    SUITE["tracking"] = BenchmarkGroup()

    # ── Katsura-3: 4 equations, 4 variables, 81 paths ────────────────────
    @polyvar x0 x1 x2 x3
    F_katsura = [
        x0 + 2x1 + 2x2 + 2x3 - 1,
        x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
        2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
        x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
    ]

    # Total degree start system
    @polyvar y0 y1 y2 y3
    G_td = [y0 - 1, y1^2 - 1, y2^2 - 1, y3^2 - 1]
    _, eval_G = system_eval(G_td)
    _, eval_F = system_eval(F_katsura)

    H = StraightLineHomotopy(eval_G, eval_F)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval)

    # One path from (1,1,1,1) — solution of G
    x₀ = ComplexF64[1.0, 1.0, 1.0, 1.0]

    # Warmup
    track!(tracker, x₀)

    SUITE["tracking"]["track_one_path_katsura3"] =
        @benchmarkable track!($tracker, $x₀)

    return SUITE
end
```

- [ ] **Step 2: Wire into runbenchmarks.jl**

Add after the core benchmark include:

```julia
include("tracking.jl")
benchmark_tracking!(SUITE)
```

- [ ] **Step 3: Run benchmarks**

```bash
make benchmark
```

- [ ] **Step 4: Run full quality gates**

```bash
make format && make test
```

---

## Implementation Notes

### Key API Signatures Reference

**Existing primitives used by the tracker:**
```julia
# Jacobian (from linear_algebra.jl)
updated!(J::Jacobian)                                    # Mark matrix as changed
init!(J::Jacobian)                                       # Reset counters
LA.ldiv!(x, J::Jacobian, b::AbstractVector{ComplexF64})  # Solve without scaling
LA.ldiv!(x, J::Jacobian, b, w::WeightedNorm)             # Solve with Skeel scaling
LA.cond(J::Jacobian)                                     # Condition estimate

# WeightedNorm (from norms.jl)
weighted_norm(x, w::WeightedNorm) → Float64
weighted_distance(x, y, w::WeightedNorm) → Float64
init!(w::WeightedNorm, x)                                # Initialize weights from x
update!(w::WeightedNorm, x)                              # Interpolate weights toward x

# SegmentStepper (from utils.jl)
is_done(S)::Bool
step_success!(S)
propose_step!(S, Δs)
dist_to_target(S)::Float64
S.t, S.t′, S.Δs, S.Δt                                   # Virtual properties

# HomotopyEvaluator (from core/)
evaluate!(u, H, x, t)                                    # u = H(x,t)
evaluate_and_jacobian!(u, U, H, x, t)                    # u = H(x,t), U = ∂H/∂x
taylor!(u, Val(1), H, x, t)                              # ∂H/∂t (x is FSVec)
taylor!(u, Val(2), H, tx, t; incremental=false)           # tx is TaylorVector{3}
taylor!(u, Val(3), H, tx, t; incremental=false)           # tx is TaylorVector{4}
```

### Potential Issues

1. **TaylorVector aliasing**: The predictor ideally shares memory between tx⁰/tx¹/tx²/tx³. If this is too complex, use independent allocations and copy. The performance difference is small for n ≤ 20.

2. **Predictor calls homotopy taylor!, not system taylor!**: The `update!` in the predictor differentiates the homotopy H(x(t),t)=0, not the individual systems. This is critical — the taylor! on HomotopyEvaluator has different signatures than on SystemEvaluator.

3. **Newton's evaluate_and_jacobian! writes to J.workspace.A**: The Jacobian workspace's A matrix is shared with the LU factors. Calling `updated!(J)` marks it as unfactored. Then `LA.ldiv!` triggers factorization.

4. **Step size must be positive**: `propose_step!` takes a positive `Δs` regardless of direction. The `SegmentStepper` handles forward/backward internally.
