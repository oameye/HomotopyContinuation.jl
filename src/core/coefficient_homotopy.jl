## CoefficientHomotopy — linear interpolation of coefficients for polyhedral homotopy.
#
# H(x, t) = F(x; p(t)) where p(t) = t·start + (1-t)·target
# At t=1: p = start (generic system). At t=0: p = target (user's system).

struct CoefficientHomotopy <: AbstractHomotopy
    system::SystemEvaluator
    start_coeffs::FSVec{ComplexF64}
    target_coeffs::FSVec{ComplexF64}
    coeffs::FSVec{ComplexF64}           # current p(t) — contents mutated
    dt_coeffs::FSVec{ComplexF64}        # start - target (constant dp/dt)
    t_cache::Base.RefValue{ComplexF64}
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

    return CoefficientHomotopy(system, sc, tc, coeffs, dt, Ref(complex(NaN)))
end

Base.size(H::CoefficientHomotopy) = size(H.system)

## ── Coefficient interpolation ────────────────────────────────────────────

@inline function _update_coeffs!(H::CoefficientHomotopy, t::ComplexF64)::Nothing
    H.t_cache[] == t && return nothing
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(H.coeffs)
        H.coeffs[i] = t * H.start_coeffs[i] + t1 * H.target_coeffs[i]
    end
    H.t_cache[] = t
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

# Orders 2, 3: p is linear in t so pure time-derivatives of p vanish.
# The x(t)-contribution comes from the system's Taylor with current coefficients.
function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::CoefficientHomotopy,
        tx::TaylorVector{3, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    _update_coeffs!(H, t)
    taylor!(u, Val(2), H.system, tx, H.coeffs)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::CoefficientHomotopy,
        tx::TaylorVector{4, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    _update_coeffs!(H, t)
    taylor!(u, Val(3), H.system, tx, H.coeffs)
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
