## CoefficientHomotopy — linear interpolation of coefficients for polyhedral homotopy.
#
# H(x, t) = F(x; p(t)) where p(t) = t·start + (1-t)·target
# At t=1: p = start (generic system). At t=0: p = target (user's system).
#
# Taylor computation uses Cauchy product convolution: parameter Taylor coefficients
# are packed into a TaylorVector and passed through the SystemEvaluator FunctionWrapper,
# so the interpreter's taylor_op_mul handles the convolution automatically (v2 parity).

struct CoefficientHomotopy <: AbstractHomotopy
    system::SystemEvaluator
    start_coeffs::FSVec{ComplexF64}
    target_coeffs::FSVec{ComplexF64}
    coeffs::FSVec{ComplexF64}           # current p₀(t) — contents mutated
    dt_coeffs::FSVec{ComplexF64}        # p₁ = start - target (constant dp/dt)
    t_cache::Base.RefValue{ComplexF64}
    # Parameter TaylorVectors for Cauchy product convolution
    tp2::TaylorVector{3, ComplexF64}    # [p₀, p₁, 0] for Val(2) calls
    tp3::TaylorVector{4, ComplexF64}    # [p₀, p₁, 0, 0] for Val(3) calls
end

function CoefficientHomotopy(
        system::SystemEvaluator,
        start_coeffs::AbstractVector{ComplexF64},
        target_coeffs::AbstractVector{ComplexF64},
    )
    np = nparameters(system)
    @assert length(start_coeffs) == np "start_coeffs length must match nparameters"
    @assert length(target_coeffs) == np "target_coeffs length must match nparameters"

    sc = FSVec{ComplexF64}(collect(ComplexF64, start_coeffs))
    tc = FSVec{ComplexF64}(collect(ComplexF64, target_coeffs))
    coeffs = FSVec{ComplexF64}(zeros(ComplexF64, np))
    dt = FSVec{ComplexF64}(zeros(ComplexF64, np))
    @inbounds for i in eachindex(dt)
        dt[i] = sc[i] - tc[i]
    end

    return CoefficientHomotopy(
        system, sc, tc, coeffs, dt, Ref(complex(NaN)),
        TaylorVector{3, ComplexF64}(np),
        TaylorVector{4, ComplexF64}(np),
    )
end

Base.size(H::CoefficientHomotopy) = size(H.system)

## ── Coefficient interpolation ────────────────────────────────────────────

@inline function _update_coeffs!(H::CoefficientHomotopy, t::ComplexF64)::Nothing
    H.t_cache[] == t && return nothing
    # Use real arithmetic when t is real to avoid complex multiply roundoff (v2 parity)
    if isreal(t)
        s = real(t)
        s1 = 1.0 - s
        @inbounds for i in eachindex(H.coeffs)
            H.coeffs[i] = s * H.start_coeffs[i] + s1 * H.target_coeffs[i]
        end
    else
        t1 = one(ComplexF64) - t
        @inbounds for i in eachindex(H.coeffs)
            H.coeffs[i] = t * H.start_coeffs[i] + t1 * H.target_coeffs[i]
        end
    end
    H.t_cache[] = t
    return nothing
end

## ── Parameter TaylorVector builder ──────────────────────────────────────
# p(t) is linear so p₀ = coeffs(t), p₁ = start − target, p_k≥2 = 0.

@inline function _pack_param_taylor!(
        tp::TaylorVector{N, ComplexF64}, H::CoefficientHomotopy, t::ComplexF64,
    )::Nothing where {N}
    _update_coeffs!(H, t)
    np = length(H.coeffs)
    @inbounds for i in 1:np
        tp.data[1, i] = H.coeffs[i]
        tp.data[2, i] = H.dt_coeffs[i]
        for k in 3:N
            tp.data[k, i] = zero(ComplexF64)
        end
    end
    return nothing
end

## ── evaluate! ────────────────────────────────────────────────────────────

function evaluate!(
        u::FSVec{ComplexF64}, H::CoefficientHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    _update_coeffs!(H, t)
    evaluate!(u, H.system, x, H.coeffs)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, H::CoefficientHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    _update_coeffs!(H, t)
    evaluate!(u, H.system, x, H.coeffs)
    return nothing
end

## ── evaluate_and_jacobian! ───────────────────────────────────────────────

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        H::CoefficientHomotopy, x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    _update_coeffs!(H, t)
    evaluate_and_jacobian!(u, U, H.system, x, H.coeffs)
    return nothing
end

## ── taylor! ──────────────────────────────────────────────────────────────

# Order 1: dH/dt at fixed x.
# Since p(t) is linear and F is linear in p: dH/dt = F(x; dp/dt) = F(x; start - target)
function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::CoefficientHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    evaluate!(u, H.system, x, H.dt_coeffs)
    return nothing
end

# Orders 2, 3: Single call with TaylorVector parameters.
# The Cauchy product inside the interpreter computes:
#   [H]_k = Σ_{j=0}^{k} [F(x)]_{k-j} · p_j
# For CoefficientHomotopy: p₀ = coeffs(t), p₁ = start - target, p₂ = p₃ = 0.

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::CoefficientHomotopy,
        tx::TaylorVector{3, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    _pack_param_taylor!(H.tp2, H, t)
    taylor!(u, Val(2), H.system, tx, H.tp2)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::CoefficientHomotopy,
        tx::TaylorVector{4, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    _pack_param_taylor!(H.tp3, H, t)
    taylor!(u, Val(3), H.system, tx, H.tp3)
    return nothing
end

## ── set/get solution, parameters — identity ──────────────────────────────

function set_solution!(
        x::FSVec{ComplexF64}, ::CoefficientHomotopy,
        y::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(x, y)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, ::CoefficientHomotopy,
        x::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(out, x)
    return nothing
end

start_parameters!(::CoefficientHomotopy, ::FSVec{ComplexF64})::Nothing = nothing
target_parameters!(::CoefficientHomotopy, ::FSVec{ComplexF64})::Nothing = nothing
