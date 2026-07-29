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
    u_cross_start::FSVec{ComplexF64}
    u_cross_target::FSVec{ComplexF64}
    ū_start::FSVec{ComplexDF64}
    ū_target::FSVec{ComplexDF64}
    U_start::FSMat{ComplexF64}
    U_target::FSMat{ComplexF64}
    tx1::TaylorVector{2, ComplexF64}
    tx2::TaylorVector{3, ComplexF64}
    dv_start::TaylorVector{4, ComplexF64}
    dv_target::TaylorVector{4, ComplexF64}
end

function StraightLineHomotopy(
        start::SystemEvaluator, target::SystemEvaluator;
        γ::ComplexF64 = cis(2π * rand()),
    )
    m, n = size(target)
    size(start) == (m, n) || throw(
        ArgumentError(
            "the start system has size $(size(start)), but the target system has " *
                "size $((m, n)); a homotopy needs both to agree.",
        ),
    )
    return StraightLineHomotopy(
        start, target, γ,
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, m)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, m)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
        TaylorVector{2, ComplexF64}(n),
        TaylorVector{3, ComplexF64}(n),
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
#
# The two terms cancel along the path (H(x,t) ≈ 0), so evaluate both systems
# with DF64 output and combine in extended precision, rounding only on store.

function evaluate!(
        u::FSVec{ComplexF64}, H::StraightLineHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    evaluate!(H.ū_start, H.start, x, _EMPTY_PARAMS)
    evaluate!(H.ū_target, H.target, x, _EMPTY_PARAMS)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = ComplexF64(γt * H.ū_start[i] + t1 * H.ū_target[i])
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

@inline function _copy_prefix!(
        dst::TaylorVector{K, T},
        src::TaylorVector{N, T},
    )::Nothing where {K, N, T}
    @inbounds for j in axes(src.data, 2), i in 1:K
        dst.data[i, j] = src.data[i, j]
    end
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::StraightLineHomotopy,
        tx::TaylorVector{3, ComplexF64}, t::ComplexF64,
    )::Nothing
    _copy_prefix!(H.tx1, tx)
    taylor!(H.u_cross_start, Val(1), H.start, H.tx1, _EMPTY_PARAMS)
    taylor!(H.u_cross_target, Val(1), H.target, H.tx1, _EMPTY_PARAMS)
    taylor!(H.u_start, Val(2), H.start, tx, _EMPTY_PARAMS)
    taylor!(H.u_target, Val(2), H.target, tx, _EMPTY_PARAMS)
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = H.γ * (H.u_cross_start[i] + t * H.u_start[i]) +
            t1 * H.u_target[i] - H.u_cross_target[i]
    end
    return nothing
end

## taylor! order 3

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::StraightLineHomotopy,
        tx::TaylorVector{4, ComplexF64}, t::ComplexF64,
    )::Nothing
    _copy_prefix!(H.tx2, tx)
    taylor!(H.u_cross_start, Val(2), H.start, H.tx2, _EMPTY_PARAMS)
    taylor!(H.u_cross_target, Val(2), H.target, H.tx2, _EMPTY_PARAMS)
    taylor!(H.u_start, Val(3), H.start, tx, _EMPTY_PARAMS)
    taylor!(H.u_target, Val(3), H.target, tx, _EMPTY_PARAMS)
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = H.γ * (H.u_cross_start[i] + t * H.u_start[i]) +
            t1 * H.u_target[i] - H.u_cross_target[i]
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
