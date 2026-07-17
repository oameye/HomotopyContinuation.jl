# Standalone Newton's method on a `System`, decoupled from path tracking.
#
# This is a user-facing convenience: refine a candidate solution of `F` (or the
# least-squares zero of an overdetermined `F`) starting from `x₀`. Convergence
# uses the contraction-factor criterion — an update is accepted only once the
# Newton corrections shrink fast enough.

@enumx NewtonReturnCode::Int8 begin
    NEWTON_SUCCESS
    NEWTON_REJECTED
    NEWTON_MAX_ITERS
end

"""
    NewtonResult

Result of [`newton`](@ref).

## Fields
- `return_code::NewtonReturnCode.T`: `NEWTON_SUCCESS`, `NEWTON_REJECTED`, or `NEWTON_MAX_ITERS`.
- `x::Vector{ComplexF64}`: the last iterate.
- `accuracy::Float64`: norm of the final Newton update, an estimate of the distance to a zero.
- `residual::Float64`: infinity norm of `F(x)` at the returned point.
- `iters::Int`: number of iterations performed.
- `contraction_ratio::Float64`: `‖Δxᵢ‖ / ‖Δxᵢ₋₁‖` at the last step.
"""
struct NewtonResult
    return_code::NewtonReturnCode.T
    x::Vector{ComplexF64}
    accuracy::Float64
    residual::Float64
    iters::Int
    contraction_ratio::Float64
end

"""
    is_success(r::NewtonResult) -> Bool

`true` if [`newton`](@ref) converged.
"""
is_success(r::NewtonResult)::Bool = r.return_code == NewtonReturnCode.NEWTON_SUCCESS

"""
    solution(r::NewtonResult) -> Vector{ComplexF64}

The last iterate stored in `r` (the approximate zero on success).
"""
solution(r::NewtonResult)::Vector{ComplexF64} = r.x

function Base.show(io::IO, r::NewtonResult)
    print(io, "NewtonResult: ", r.return_code)
    print(io, " (iters = ", r.iters)
    print(io, ", accuracy = ", r.accuracy)
    print(io, ", residual = ", r.residual, ")")
    return
end

"""
    NewtonCache(F::System)

Pre-allocate the scratch buffers and matrix workspace used by [`newton`](@ref).
Pass the same cache to repeated `newton` calls on `F` to avoid re-allocating.
Square and overdetermined systems (`m ≥ n`) use the LU-based `MatrixWorkspace`;
underdetermined systems (`m < n`) store the Jacobian in `J_wide` and solve each
Newton step with an allocating column-pivoted QR.
Empty-size sentinels mark the unused branch: `J_wide` is `0×0` when `m ≥ n`,
and `workspace` is a `1×1` placeholder when `m < n`.
"""
struct NewtonCache
    x::FSVec{ComplexF64}
    Δx::FSVec{ComplexF64}
    x_ext::FSVec{ComplexDF64}
    r::FSVec{ComplexF64}
    workspace::MatrixWorkspace
    J_wide::FSMat{ComplexF64}
end

function NewtonCache(F::System)::NewtonCache
    m, n = size(F)
    workspace = m >= n ? MatrixWorkspace(m, n) : MatrixWorkspace(1, 1)
    J_wide = if m < n
        FSMat{ComplexF64}(zeros(ComplexF64, m, n))
    else
        FSMat{ComplexF64}(zeros(ComplexF64, 0, 0))
    end
    return NewtonCache(
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        workspace,
        J_wide,
    )
end

# Least-squares Newton step for the underdetermined case via column-pivoted QR.
# Cold path: allocates the factorization freely.
function _solve_wide!(
        Δx::FSVec{ComplexF64}, J::FSMat{ComplexF64}, r::FSVec{ComplexF64},
    )::Nothing
    M = Matrix{ComplexF64}(J)
    b = Vector{ComplexF64}(r)
    copyto!(Δx, LA.qr!(M, LA.ColumnNorm()) \ b)
    return nothing
end

"""
    newton(F::System, x₀::AbstractVector; options...) -> NewtonResult

Run Newton's method on the polynomial system `F` starting from `x₀`. For an
overdetermined `F` (m > n) the Newton step solves the linear least-squares
problem, so the method converges to a common zero when one exists. For an
underdetermined `F` (m < n) each step solves the rank-revealing least-squares
problem via column-pivoted QR and converges to a nearby point on the solution
variety. Computations
are in `Complex{Float64}`; `extended_precision` optionally evaluates the residual
in double-double precision to push the achievable accuracy toward machine eps.

## Options
- `p::AbstractVector = ComplexF64[]`: parameter values for a parametric system
  (length `nparameters(F)`); leave empty for a parameter-free system.
- `atol::Float64 = 1e-8`: converged when `‖Δxᵢ‖ < atol`.
- `rtol::Float64 = atol`: converged when `‖Δxᵢ‖ < max(atol, rtol · ‖x₀‖)`.
- `max_iters::Int = 20`: maximum number of iterations.
- `extended_precision::Bool = false`: evaluate `F(x)` in extended precision.
- `contraction_factor::Float64 = 1.0`: accept only if `‖Δxᵢ‖ < a · ‖Δxᵢ₋₁‖` (`a` squares each accepted step).
- `min_contraction_iters::Int = typemax(Int)`: accept anyway after this many iterations.
- `max_abs_norm_first_update::Float64 = Inf`: reject `x₀` if `‖Δx₁‖ > this`.
- `max_rel_norm_first_update::Float64 = max_abs_norm_first_update`: reject `x₀` if `‖Δx₁‖ > this · ‖x₀‖`.
- `cache::NewtonCache = NewtonCache(F)`: pre-allocated workspace for repeated calls.
"""
function newton(
        F::System, x₀::AbstractVector;
        p::AbstractVector = ComplexF64[],
        atol::Float64 = 1.0e-8,
        rtol::Float64 = atol,
        max_iters::Int = 20,
        extended_precision::Bool = false,
        contraction_factor::Float64 = 1.0,
        min_contraction_iters::Int = typemax(Int),
        max_abs_norm_first_update::Float64 = Inf,
        max_rel_norm_first_update::Float64 = max_abs_norm_first_update,
        cache::NewtonCache = NewtonCache(F),
    )::NewtonResult
    m, n = size(F)
    length(x₀) == n || throw(
        DimensionMismatch("x₀ has length $(length(x₀)), expected $n"),
    )
    np = nparameters(F)
    params = if isempty(p)
        np == 0 || throw(ArgumentError("system has $np parameters, pass `p`"))
        FSVec{ComplexF64}(ComplexF64[])
    else
        length(p) == np || throw(
            DimensionMismatch("p has length $(length(p)), expected $np"),
        )
        FSVec{ComplexF64}(ComplexF64.(p))
    end
    return _newton(
        F.evaluator, cache, x₀, params, atol, rtol, max_iters, extended_precision,
        contraction_factor, min_contraction_iters, max_abs_norm_first_update,
        max_rel_norm_first_update,
    )
end

function _newton(
        S::SystemEvaluator,
        cache::NewtonCache,
        x₀::AbstractVector,
        p::FSVec{ComplexF64},
        atol::Float64,
        rtol::Float64,
        max_iters::Int,
        extended_precision::Bool,
        contraction_factor::Float64,
        min_contraction_iters::Int,
        max_abs_norm_first_update::Float64,
        max_rel_norm_first_update::Float64,
    )::NewtonResult
    x = cache.x
    Δx = cache.Δx
    x_ext = cache.x_ext
    r = cache.r
    WS = cache.workspace
    wide = !isempty(cache.J_wide)

    copyto!(x, x₀)
    a = contraction_factor
    norm_x = inf_norm(x)
    norm_Δxᵢ = NaN
    norm_Δxᵢ₋₁ = NaN
    res = NaN

    for i in 1:max_iters
        if wide
            evaluate_and_jacobian!(r, cache.J_wide, S, x, p)
        else
            evaluate_and_jacobian!(r, WS.A, S, x, p)
            updated!(WS)
        end
        if extended_precision
            _copy_df64!(x_ext, x)
            evaluate!(r, S, x_ext, p)
        end
        if wide
            _solve_wide!(Δx, cache.J_wide, r)
        else
            LA.ldiv!(Δx, WS, r)
        end
        norm_Δxᵢ = inf_norm(Δx)
        res = inf_norm(r)

        @inbounds for k in eachindex(x, Δx)
            x[k] -= Δx[k]
        end

        if i > 1
            if norm_Δxᵢ < a * norm_Δxᵢ₋₁
                norm_Δxᵢ₋₁ = norm_Δxᵢ
                a *= a
            elseif i > min_contraction_iters || norm_Δxᵢ < max(atol, rtol * norm_x)
                return NewtonResult(
                    NewtonReturnCode.NEWTON_SUCCESS, Vector{ComplexF64}(x),
                    norm_Δxᵢ, res, i, norm_Δxᵢ / norm_Δxᵢ₋₁,
                )
            else
                return NewtonResult(
                    NewtonReturnCode.NEWTON_REJECTED, Vector{ComplexF64}(x),
                    norm_Δxᵢ, res, i, norm_Δxᵢ / norm_Δxᵢ₋₁,
                )
            end
        elseif i == 1 && (
                norm_Δxᵢ > max_rel_norm_first_update * norm_x ||
                    norm_Δxᵢ > max_abs_norm_first_update
            )
            return NewtonResult(
                NewtonReturnCode.NEWTON_REJECTED, Vector{ComplexF64}(x),
                norm_Δxᵢ, res, i, NaN,
            )
        else
            norm_Δxᵢ₋₁ = norm_Δxᵢ
        end
    end

    return NewtonResult(
        NewtonReturnCode.NEWTON_MAX_ITERS, Vector{ComplexF64}(x),
        norm_Δxᵢ, res, max_iters, norm_Δxᵢ / norm_Δxᵢ₋₁,
    )
end
