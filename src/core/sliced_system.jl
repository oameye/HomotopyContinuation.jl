## SlicedSystem: `[F(x); A x − b]` (plus an optional affine chart row) built by
## wrapping an existing `SystemEvaluator` instead of rebuilding polynomials.
#
# The linear rows are evaluated in a plain loop, their Jacobian rows are `A`, and
# their order-K Taylor coefficient is `A x_K`, NOT zero (see
# `_sliced_taylor_rows!`). `A` is stored transposed so the inner loop runs over the
# contiguous index.

struct SlicedSystem <: AbstractSystem
    system::SystemEvaluator
    # transpose of the k × n matrix A of the extrinsic description
    Aᵗ::FSMat{ComplexF64}
    b::FSVec{ComplexF64}
    # empty unless the problem is projective
    chart::Vector{ComplexF64}
end

function SlicedSystem(
        system::SystemEvaluator, L::LinearSubspace,
        chart::Vector{ComplexF64} = ComplexF64[],
    )::SlicedSystem
    E = extrinsic(convert(LinearSubspace{ComplexF64}, L))
    n = size(system)[2]
    size(E.A, 2) == n || throw(
        ArgumentError(
            "The subspace lives in dimension $(size(E.A, 2)), but the system has " *
                "$n variables.",
        ),
    )
    isempty(chart) || length(chart) == n || throw(
        ArgumentError("The chart must have length $n, got $(length(chart))."),
    )
    return SlicedSystem(
        system,
        FSMat{ComplexF64}(collect(transpose(E.A))),
        FSVec{ComplexF64}(copy(E.b)),
        chart,
    )
end

# Canonical wrap: all consumers go through this so evaluator construction cannot
# drift between call sites.
_sliced_evaluator(
    inner::SystemEvaluator, L::LinearSubspace, chart::Vector{ComplexF64},
)::SystemEvaluator = SystemEvaluator(SlicedSystem(inner, L, chart))

_clone_system(S::SlicedSystem)::SlicedSystem = SlicedSystem(
    _clone_system_evaluator(S.system), copy(S.Aᵗ), copy(S.b), copy(S.chart),
)

_nlinear(S::SlicedSystem)::Int = size(S.Aᵗ, 2)

function Base.size(S::SlicedSystem)::Tuple{Int, Int}
    m, n = size(S.system)
    return (m + _nlinear(S) + (isempty(S.chart) ? 0 : 1), n)
end

nparameters(S::SlicedSystem)::Int = nparameters(S.system)

# `A x − b` into the rows after the wrapped system's, then the chart row. The
# accumulator follows the input precision; only the store rounds.
@inline function _sliced_rows!(
        u::FSVec{U}, S::SlicedSystem, x::FSVec{X}, m::Int,
    )::Nothing where {U <: Complex, X <: Complex}
    k = _nlinear(S)
    n = size(S.Aᵗ, 1)
    @inbounds for i in 1:k
        acc = -X(S.b[i])
        for j in 1:n
            acc = muladd(S.Aᵗ[j, i], x[j], acc)
        end
        u[m + i] = U(acc)
    end
    isempty(S.chart) || (u[m + k + 1] = U(evaluate_chart(S.chart, x)))
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, S::SlicedSystem,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(u, S.system, x, p)
    _sliced_rows!(u, S, x, size(S.system)[1])
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, S::SlicedSystem,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(u, S.system, x, p)
    _sliced_rows!(u, S, x, size(S.system)[1])
    return nothing
end

function evaluate!(
        u::FSVec{ComplexDF64}, S::SlicedSystem,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(u, S.system, x, p)
    _sliced_rows!(u, S, x, size(S.system)[1])
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, S::SlicedSystem,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    # The interpreter writes only the wrapped system's rows; the linear rows and
    # the chart row are written afterwards.
    evaluate_and_jacobian!(u, U, S.system, x, p)
    m = size(S.system)[1]
    _sliced_rows!(u, S, x, m)
    k = _nlinear(S)
    n = size(S.Aᵗ, 1)
    @inbounds for j in 1:n, i in 1:k
        U[m + i, j] = S.Aᵗ[j, i]
    end
    if !isempty(S.chart)
        @inbounds for j in 1:n
            U[m + k + 1, j] = S.chart[j]
        end
    end
    return nothing
end

# The appended rows are affine forms, so the order-K coefficient of
# `A x(t) − b` is `A x_K`: only the highest-order coefficient of `x` contributes.
# It is zero exactly when the caller zeroed that row (the predictor does), but a
# straight-line homotopy also asks for the order-(K-1) coefficient with a
# nonzero top row, and dropping the term there costs prediction accuracy.
@inline function _sliced_taylor_rows!(
        u::FSVec{ComplexF64}, S::SlicedSystem, tx::TaylorVector{N, ComplexF64},
        ::Val{K},
    )::Nothing where {N, K}
    x_K = vectors(tx)[K + 1]
    m = size(S.system)[1]
    k = _nlinear(S)
    n = size(S.Aᵗ, 1)
    @inbounds for i in 1:k
        acc = zero(ComplexF64)
        for j in 1:n
            acc = muladd(S.Aᵗ[j, i], x_K[j], acc)
        end
        u[m + i] = acc
    end
    isempty(S.chart) || (u[m + k + 1] = _chart_taylor_row(S.chart, tx, Val(K)))
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, v::Val{K}, S::SlicedSystem,
        tx::TaylorVector{N, ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing where {K, N}
    taylor!(u, v, S.system, tx, p)
    _sliced_taylor_rows!(u, S, tx, v)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, v::Val{K}, S::SlicedSystem,
        tx::TaylorVector{N, ComplexF64}, tp::TaylorVector{M, ComplexF64},
    )::Nothing where {K, N, M}
    taylor!(u, v, S.system, tx, tp)
    _sliced_taylor_rows!(u, S, tx, v)
    return nothing
end
