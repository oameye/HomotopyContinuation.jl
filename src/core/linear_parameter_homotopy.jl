## LinearParameterHomotopy — H(x, t) = F(x; p(t)) with p(t) = t·p₁ + (1 - t)·p₀,
## a linear interpolation of the parameter/coefficient vector from `start` (at
## t = 1) to `target` (at t = 0).
##
## The `Shortcut::Bool` type parameter selects ONLY the order-1 Taylor path:
##
##   Shortcut = true  (`CoefficientHomotopy`): dH/dt = F(x; p₁ - p₀). This is
##       valid ONLY when F is linear AND homogeneous in the parameters (the
##       polyhedral coefficient case). Skips the parameter-Taylor convolution.
##
##   Shortcut = false (`ParameterHomotopy`): order 1 goes through the full
##       parameter-Taylor convolution, exact for arbitrary parameter dependence.
##
## Both instantiations are distinct concrete, monomorphic types; the tracker
## only ever sees them through the HomotopyEvaluator FunctionWrapper firewall,
## so the extra type parameter costs nothing on the hot path. Every method below
## except the order-1 `taylor!` is shared between the two.

struct LinearParameterHomotopy{Shortcut} <: AbstractHomotopy
    system::SystemEvaluator
    start_p::FSVec{ComplexF64}
    target_p::FSVec{ComplexF64}
    p_t::FSVec{ComplexF64}              # p(t) cache, contents mutated
    dp::FSVec{ComplexF64}               # start - target (constant dp/dt)
    t_cache::Base.RefValue{ComplexF64}
    tx1::TaylorVector{2, ComplexF64}    # [x, 0] packing for the Val(1) convolution
    tp1::TaylorVector{2, ComplexF64}    # [p(t), dp]
    tp2::TaylorVector{3, ComplexF64}    # [p(t), dp, 0]
    tp3::TaylorVector{4, ComplexF64}    # [p(t), dp, 0, 0]
end

"""
    CoefficientHomotopy(system::SystemEvaluator, start_coeffs, target_coeffs)

The homotopy `H(x, t) = F(x; p(t))` with `p(t) = t·start + (1 - t)·target`, for
a system that is **linear and homogeneous in the parameters** (the polyhedral
coefficient case). Only under that precondition is the order-1 shortcut
`dH/dt = F(x; start - target)` valid. For general parameter dependence use
[`ParameterHomotopy`](@ref).
"""
const CoefficientHomotopy = LinearParameterHomotopy{true}

"""
    ParameterHomotopy(F::System, start_parameters, target_parameters)
    ParameterHomotopy(system::SystemEvaluator, start_parameters, target_parameters)

The homotopy `H(x, t) = F(x; p(t))` where `p(t) = t·p₁ + (1 - t)·p₀` moves
linearly from the `start_parameters` `p₁` (at `t = 1`) to the
`target_parameters` `p₀` (at `t = 0`).

Unlike [`CoefficientHomotopy`](@ref) (valid only for systems linear and
homogeneous in the parameters), all Taylor orders including `Val(1)` go through
the interpreter's parameter-Taylor convolution, which is exact for arbitrary
parameter dependence.
"""
const ParameterHomotopy = LinearParameterHomotopy{false}

function LinearParameterHomotopy{S}(
        system::SystemEvaluator,
        start_parameters::AbstractVector{<:Number},
        target_parameters::AbstractVector{<:Number},
    ) where {S}
    np = nparameters(system)
    length(start_parameters) == np || throw(
        ArgumentError(
            "start parameters have length $(length(start_parameters)), but the system " *
                "has $np parameter(s).",
        ),
    )
    length(target_parameters) == np || throw(
        ArgumentError(
            "target parameters have length $(length(target_parameters)), but the " *
                "system has $np parameter(s).",
        ),
    )
    n = size(system)[2]

    sp = FSVec{ComplexF64}(collect(ComplexF64, start_parameters))
    tp = FSVec{ComplexF64}(collect(ComplexF64, target_parameters))
    p_t = FSVec{ComplexF64}(zeros(ComplexF64, np))
    dp = FSVec{ComplexF64}(zeros(ComplexF64, np))
    @inbounds for i in eachindex(dp)
        dp[i] = sp[i] - tp[i]
    end

    return LinearParameterHomotopy{S}(
        system, sp, tp, p_t, dp, Ref(complex(NaN)),
        TaylorVector{2, ComplexF64}(n),
        TaylorVector{2, ComplexF64}(np),
        TaylorVector{3, ComplexF64}(np),
        TaylorVector{4, ComplexF64}(np),
    )
end

ParameterHomotopy(
    F::System,
    start_parameters::AbstractVector{<:Number},
    target_parameters::AbstractVector{<:Number},
) = ParameterHomotopy(F.evaluator, start_parameters, target_parameters)

Base.size(H::LinearParameterHomotopy) = size(H.system)

_clone_homotopy(H::LinearParameterHomotopy{S}) where {S} =
    LinearParameterHomotopy{S}(
    _clone_system_evaluator(H.system), H.start_p, H.target_p,
)

## Retargeting (cold path, e.g. 4 calls per monodromy loop). Recomputes dp and
## invalidates the p(t) cache so the next evaluate!/taylor! rebuilds p(t).

function parameters!(
        H::LinearParameterHomotopy, p::AbstractVector{<:Number}, q::AbstractVector{<:Number},
    )::Nothing
    copyto!(H.start_p, p)
    copyto!(H.target_p, q)
    @inbounds for i in eachindex(H.dp)
        H.dp[i] = H.start_p[i] - H.target_p[i]
    end
    H.t_cache[] = complex(NaN)  # NaN never == t, so the p(t) cache invalidates
    return nothing
end

function start_parameters!(H::LinearParameterHomotopy, p::AbstractVector{<:Number})::Nothing
    copyto!(H.start_p, p)
    @inbounds for i in eachindex(H.dp)
        H.dp[i] = H.start_p[i] - H.target_p[i]
    end
    H.t_cache[] = complex(NaN)
    return nothing
end

function target_parameters!(H::LinearParameterHomotopy, q::AbstractVector{<:Number})::Nothing
    copyto!(H.target_p, q)
    @inbounds for i in eachindex(H.dp)
        H.dp[i] = H.start_p[i] - H.target_p[i]
    end
    H.t_cache[] = complex(NaN)
    return nothing
end

## p(t) interpolation cache

@inline function _update_p!(H::LinearParameterHomotopy, t::ComplexF64)::Nothing
    H.t_cache[] == t && return nothing
    # Use real arithmetic when t is real to avoid complex-multiply roundoff.
    if isreal(t)
        s = real(t)
        s1 = 1.0 - s
        @inbounds for i in eachindex(H.p_t)
            H.p_t[i] = s * H.start_p[i] + s1 * H.target_p[i]
        end
    else
        t1 = one(ComplexF64) - t
        @inbounds for i in eachindex(H.p_t)
            H.p_t[i] = t * H.start_p[i] + t1 * H.target_p[i]
        end
    end
    H.t_cache[] = t
    return nothing
end

## Parameter TaylorVector packing: p₀ = p(t), p₁ = dp, higher orders zero
## (p(t) is linear in t).

@inline function _pack_param_taylor!(
        tp::TaylorVector{N, ComplexF64}, H::LinearParameterHomotopy, t::ComplexF64,
    )::Nothing where {N}
    _update_p!(H, t)
    np = length(H.p_t)
    @inbounds for i in 1:np
        tp.data[1, i] = H.p_t[i]
        tp.data[2, i] = H.dp[i]
        for k in 3:N
            tp.data[k, i] = zero(ComplexF64)
        end
    end
    return nothing
end

## evaluate!

function evaluate!(
        u::FSVec{ComplexF64}, H::LinearParameterHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    _update_p!(H, t)
    evaluate!(u, H.system, x, H.p_t)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, H::LinearParameterHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    _update_p!(H, t)
    evaluate!(u, H.system, x, H.p_t)
    return nothing
end

## evaluate_and_jacobian!

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        H::LinearParameterHomotopy, x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    _update_p!(H, t)
    evaluate_and_jacobian!(u, U, H.system, x, H.p_t)
    return nothing
end

## taylor!
#
# Order 1 is the only method that depends on `Shortcut`:
#   Shortcut = true : dH/dt = F(x; dp) — valid only for linear+homogeneous
#       parameter dependence (CoefficientHomotopy).
#   Shortcut = false: full parameter-Taylor convolution with [x, 0] and
#       [p(t), dp], exact for arbitrary parameter dependence (ParameterHomotopy).
# Orders 2 and 3 always use the Cauchy-product convolution and are shared.

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::LinearParameterHomotopy{true},
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    evaluate!(u, H.system, x, H.dp)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::LinearParameterHomotopy{false},
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    _pack_param_taylor!(H.tp1, H, t)
    n = length(x)
    @inbounds for i in 1:n
        H.tx1.data[1, i] = x[i]
        H.tx1.data[2, i] = zero(ComplexF64)
    end
    taylor!(u, Val(1), H.system, H.tx1, H.tp1)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::LinearParameterHomotopy,
        tx::TaylorVector{3, ComplexF64}, t::ComplexF64,
    )::Nothing
    _pack_param_taylor!(H.tp2, H, t)
    taylor!(u, Val(2), H.system, tx, H.tp2)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::LinearParameterHomotopy,
        tx::TaylorVector{4, ComplexF64}, t::ComplexF64,
    )::Nothing
    _pack_param_taylor!(H.tp3, H, t)
    taylor!(u, Val(3), H.system, tx, H.tp3)
    return nothing
end

## set/get solution: identity

function set_solution!(
        x::FSVec{ComplexF64}, ::LinearParameterHomotopy,
        y::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(x, y)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, ::LinearParameterHomotopy,
        x::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(out, x)
    return nothing
end
