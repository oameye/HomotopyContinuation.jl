# Newton corrector for homotopy continuation path tracking.
# Implements alpha-theory based convergence monitoring.

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
    norm_Δx₀::Float64
end

"""
    NewtonCorrector

Pre-allocated workspace for α-theory Newton correction.

Fields:
- `a`, `h_a`, `sqrt_1m2ha`: convergence parameters (constant for a given `a`)
- `Δx`, `r`, `x_ext`: scratch buffers (contents mutated, references fixed)
"""
struct NewtonCorrector
    a::Float64
    h_a::Float64
    sqrt_1m2ha::Float64
    Δx::FSVec{ComplexF64}
    r::FSVec{ComplexF64}
    x_ext::FSVec{ComplexDF64}
end

function NewtonCorrector(a::Float64, n::Int, m::Int)
    h_a = 2a * (sqrt(4a^2 + 1) - 2a)
    sqrt_1m2ha = sqrt(1.0 - 2.0 * h_a)
    return NewtonCorrector(
        a, h_a, sqrt_1m2ha,
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, n)),
    )
end

const NEWTON_MAX_ITERS = 11

"""
    newton!(x̄, NC, H, x₀, t, J, norm, ω, μ, first_correction) → NewtonCorrectorResult

Perform Newton correction on predicted point `x₀`. Writes corrected point into `x̄`.
All arguments are positional for hot-path performance (no kwargs overhead).

- `ω`: Lipschitz constant estimate
- `μ`: accuracy target
- `first_correction`: true on the very first correction from init (skips early exit)
"""
function newton!(
        x̄::FSVec{ComplexF64},
        NC::NewtonCorrector,
        H::HomotopyEvaluator,
        x₀::FSVec{ComplexF64},
        t::ComplexF64,
        J::Jacobian,
        norm::WeightedNorm,
        ω::Float64,
        μ::Float64,
        first_correction::Bool,
    )::NewtonCorrectorResult
    a = NC.a
    h_a = NC.h_a
    sqrt_1m2ha = NC.sqrt_1m2ha
    Δx = NC.Δx
    r = NC.r

    # Copy starting point
    @inbounds for i in eachindex(x̄, x₀)
        x̄[i] = x₀[i]
    end

    norm_Δx₀ = 0.0
    norm_Δx_prev = 0.0
    θ = 0.0
    ā = a

    for k in 0:(NEWTON_MAX_ITERS - 1)
        # Evaluate H(x̄, t) and Jacobian
        evaluate_and_jacobian!(r, J.workspace.A, H, x̄, t)
        updated!(J)

        # Solve J * Δx = r
        LA.ldiv!(Δx, J, r, norm)

        norm_Δx = weighted_norm(Δx, norm)

        # Check for singularity
        if isnan(norm_Δx)
            return NewtonCorrectorResult(
                NewtonCode.NEWT_SINGULARITY, Inf, k, ω, θ, norm_Δx₀,
            )
        end

        # Apply correction: x̄ -= Δx
        @inbounds for i in eachindex(x̄, Δx)
            x̄[i] -= Δx[i]
        end

        if k == 0
            norm_Δx₀ = norm_Δx
            # Early termination check for non-first corrections
            if !first_correction && 0.125 * norm_Δx₀ * ω > h_a
                return NewtonCorrectorResult(
                    NewtonCode.NEWT_TERMINATED, Inf, k + 1, ω, θ, norm_Δx₀,
                )
            end
        else
            # Update omega and theta estimates
            if norm_Δx_prev > eps()
                norm_Δx_prev_sq = norm_Δx_prev * norm_Δx_prev
                ω = 2.0 * norm_Δx / norm_Δx_prev_sq
                θ = norm_Δx / norm_Δx_prev
            else
                θ = 0.0
            end

            if θ > ā
                return NewtonCorrectorResult(
                    NewtonCode.NEWT_TERMINATED, Inf, k + 1, ω, θ, norm_Δx₀,
                )
            end
        end

        # Convergence check: ω * ‖Δx‖² < 2μ * √(1 - 2h_a)
        norm_Δx_sq = norm_Δx * norm_Δx
        if ω * norm_Δx_sq < 2.0 * μ * sqrt_1m2ha
            # One more eval+solve to get accuracy estimate
            evaluate_and_jacobian!(r, J.workspace.A, H, x̄, t)
            updated!(J)
            LA.ldiv!(Δx, J, r, norm)
            norm_Δx_next = weighted_norm(Δx, norm)

            if isnan(norm_Δx_next)
                return NewtonCorrectorResult(
                    NewtonCode.NEWT_SINGULARITY, Inf, k + 2, ω, θ, norm_Δx₀,
                )
            end

            # Check for divergence
            if norm_Δx_next > sqrt(norm_Δx)
                return NewtonCorrectorResult(
                    NewtonCode.NEWT_TERMINATED, norm_Δx_next, k + 2, ω, θ, norm_Δx₀,
                )
            end

            # Apply final correction
            @inbounds for i in eachindex(x̄, Δx)
                x̄[i] -= Δx[i]
            end

            # Update μ
            if norm_Δx_next > 2.0 * μ
                # Recompute from fresh residual
                evaluate!(r, H, x̄, t)
                LA.ldiv!(Δx, J, r, norm)
                μ = weighted_norm(Δx, norm)
            else
                μ = norm_Δx_next
            end

            # Refine ω estimate at first iteration
            if k == 0 && norm_Δx_sq > eps()^2
                ω_new = 2.0 * norm_Δx_next / norm_Δx_sq
                if ω_new < ω
                    ω = ω_new
                else
                    ω = max(0.25 * ω, 0.1)
                end
            end

            return NewtonCorrectorResult(
                NewtonCode.NEWT_CONVERGED, max(μ, eps()), k + 2, ω, θ, norm_Δx₀,
            )
        end

        norm_Δx_prev = norm_Δx
        if k >= 1
            ā = ā * ā
        end
    end

    return NewtonCorrectorResult(
        NewtonCode.NEWT_MAX_ITERS, Inf, NEWTON_MAX_ITERS, ω, θ, norm_Δx₀,
    )
end

"""
    init_newton!(x̄, NC, H, x₀, t, J, norm) → (success, ω, μ)

Compute initial `ω` and `μ` estimates by taking Newton steps from `x₀`.
Uses perturbation strategy for exact/near-exact start points.
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

    # First Newton step from x₀ to get initial residual size
    evaluate_and_jacobian!(r, J.workspace.A, H, x₀, t)
    updated!(J)
    LA.ldiv!(Δx, J, r, norm)
    v = weighted_norm(Δx, norm) + eps()

    # Apply: x̄ = x₀ - Δx
    @inbounds for i in eachindex(x̄, x₀, Δx)
        x̄[i] = x₀[i] - Δx[i]
    end

    # Perturbation strategy: try up to 3 times with decreasing perturbation
    ε = sqrt(v)
    ω = NaN
    μ = NaN
    valid = false

    for _attempt in 1:3
        # Perturb: x̄ = x̄ + ε * weights
        @inbounds for i in eachindex(x̄)
            x̄[i] = x̄[i] + ε * norm.weights[i]
        end

        # First step from perturbed point
        evaluate_and_jacobian!(r, J.workspace.A, H, x̄, t)
        updated!(J)
        LA.ldiv!(Δx, J, r, norm)
        norm_Δx₀ = weighted_norm(Δx, norm)

        if isnan(norm_Δx₀)
            ε = ε * sqrt(ε)
            @inbounds for i in eachindex(x̄, x₀)
                x̄[i] = x₀[i]
            end
            continue
        end

        @inbounds for i in eachindex(x̄)
            x̄[i] -= Δx[i]
        end

        # Second step
        evaluate_and_jacobian!(r, J.workspace.A, H, x̄, t)
        updated!(J)
        LA.ldiv!(Δx, J, r, norm)
        norm_Δx₁ = weighted_norm(Δx, norm)

        if isnan(norm_Δx₁) || norm_Δx₁ >= a * norm_Δx₀ || norm_Δx₀ < eps()
            ε = ε * sqrt(ε)
            # Reset x̄ from x₀
            evaluate_and_jacobian!(r, J.workspace.A, H, x₀, t)
            updated!(J)
            LA.ldiv!(Δx, J, r, norm)
            @inbounds for i in eachindex(x̄, x₀, Δx)
                x̄[i] = x₀[i] - Δx[i]
            end
            continue
        end

        @inbounds for i in eachindex(x̄)
            x̄[i] -= Δx[i]
        end

        # Compute estimates
        ω = 2.0 * norm_Δx₁ / (norm_Δx₀^2)
        μ = norm_Δx₁

        # If ω*μ is too large, refine
        if ω * μ > a^7
            result = newton!(
                x̄, NC, H, x̄, t, J, norm, ω, a^7 / ω, true,
            )
            if result.return_code == NewtonCode.NEWT_CONVERGED
                ω = result.ω
                μ = result.accuracy
                valid = true
                break
            end
        else
            valid = true
            break
        end

        ε = ε * sqrt(ε)
        @inbounds for i in eachindex(x̄, x₀)
            x̄[i] = x₀[i]
        end
    end

    if !valid
        copyto!(x̄, x₀)
        return (true, 1.0, eps())
    end

    return (true, max(ω, 0.1), max(μ, eps()))
end
