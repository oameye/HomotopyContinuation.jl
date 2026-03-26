## StraightLineHomotopy: H(x,t) = γ·t·G(x) + (1-t)·F(x)
#
# At t=1: H = γ·G (start system)
# At t=0: H = F  (target system)

const _EMPTY_PARAMS = FSVec{ComplexF64}(ComplexF64[])

struct StraightLineHomotopy <: AbstractHomotopy
    start::SystemEvaluator
    target::SystemEvaluator
    γ::ComplexF64
    # Scratch buffers (contents mutated, references fixed)
    u_start::FSVec{ComplexF64}
    u_target::FSVec{ComplexF64}
    ū_start::FSVec{ComplexDF64}
    ū_target::FSVec{ComplexDF64}
    U_start::FSMat{ComplexF64}
    U_target::FSMat{ComplexF64}
    dv_start::TaylorVector{4, ComplexF64}
    dv_target::TaylorVector{4, ComplexF64}
end

function StraightLineHomotopy(
        start::SystemEvaluator, target::SystemEvaluator;
        γ::ComplexF64 = cis(2π * rand()),
    )
    m, n = size(target)
    @assert size(start) == (m, n) "Start and target systems must have the same size"
    return StraightLineHomotopy(
        start, target, γ,
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, m)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, m)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
        TaylorVector{4, ComplexF64}(m),
        TaylorVector{4, ComplexF64}(m),
    )
end

Base.size(H::StraightLineHomotopy) = size(H.target)

## evaluate! — H(x,t) = γ·t·G(x) + (1-t)·F(x)

function evaluate!(
        u::FSVec{ComplexF64}, H::StraightLineHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    evaluate!(H.u_start, H.start, x, _EMPTY_PARAMS)
    evaluate!(H.u_target, H.target, x, _EMPTY_PARAMS)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = γt * H.u_start[i] + t1 * H.u_target[i]
    end
    return nothing
end

## evaluate! DF64 variant

function evaluate!(
        u::FSVec{ComplexF64}, H::StraightLineHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    evaluate!(H.u_start, H.start, x, _EMPTY_PARAMS)
    evaluate!(H.u_target, H.target, x, _EMPTY_PARAMS)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = γt * H.u_start[i] + t1 * H.u_target[i]
    end
    return nothing
end

## evaluate_and_jacobian!

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        H::StraightLineHomotopy, x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    evaluate_and_jacobian!(H.u_start, H.U_start, H.start, x, _EMPTY_PARAMS)
    evaluate_and_jacobian!(H.u_target, H.U_target, H.target, x, _EMPTY_PARAMS)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = γt * H.u_start[i] + t1 * H.u_target[i]
    end
    @inbounds for j in axes(U, 2), i in axes(U, 1)
        U[i, j] = γt * H.U_start[i, j] + t1 * H.U_target[i, j]
    end
    return nothing
end

## taylor! order 1 — ∂H/∂t = γ·G(x) - F(x)

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::StraightLineHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    evaluate!(H.u_start, H.start, x, _EMPTY_PARAMS)
    evaluate!(H.u_target, H.target, x, _EMPTY_PARAMS)
    @inbounds for i in eachindex(u)
        u[i] = H.γ * H.u_start[i] - H.u_target[i]
    end
    return nothing
end

## taylor! order 2

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::StraightLineHomotopy,
        tx::TaylorVector{3, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    taylor!(H.u_start, Val(2), H.start, tx, _EMPTY_PARAMS)
    taylor!(H.u_target, Val(2), H.target, tx, _EMPTY_PARAMS)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = γt * H.u_start[i] + t1 * H.u_target[i]
    end
    return nothing
end

## taylor! order 3

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::StraightLineHomotopy,
        tx::TaylorVector{4, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    taylor!(H.u_start, Val(3), H.start, tx, _EMPTY_PARAMS)
    taylor!(H.u_target, Val(3), H.target, tx, _EMPTY_PARAMS)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = γt * H.u_start[i] + t1 * H.u_target[i]
    end
    return nothing
end

## set_solution! / get_solution! — default identity (just copy)

function set_solution!(
        x::FSVec{ComplexF64}, ::StraightLineHomotopy,
        y::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    copyto!(x, y)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, ::StraightLineHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    copyto!(out, x)
    return nothing
end

## start_parameters! / target_parameters! — no-op (no parameters)

start_parameters!(::StraightLineHomotopy, ::FSVec{ComplexF64})::Nothing = nothing
target_parameters!(::StraightLineHomotopy, ::FSVec{ComplexF64})::Nothing = nothing
