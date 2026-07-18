## Affine charts for projective (homogeneous) problems.
#
# A SINGLE projective chart only: the chart is one Vector{ComplexF64} and one
# appended normalization row v'x - 1; there is no PVector/multiprojective
# machinery.

"""
    AffineChartSystem(system::SystemEvaluator, chart::Vector{ComplexF64})

Given a homogeneous system ``F(x): ℙ^{n-1} → ℂ^m`` this creates a new affine
system operating on the affine chart defined by ``v`` and the augmented
condition ``v'x = 1``.
"""
struct AffineChartSystem <: AbstractSystem
    system::SystemEvaluator
    chart::Vector{ComplexF64}
end

Base.size(F::AffineChartSystem) = (size(F.system)[1] + 1, size(F.system)[2])
nparameters(F::AffineChartSystem)::Int = nparameters(F.system)

"""
    AffineChartHomotopy(homotopy::AbstractHomotopy, chart::Vector{ComplexF64})

Given a homotopy ``H(x, t): ℙ^{n-1} × ℂ → ℂ^m`` this creates a new affine
homotopy operating on the affine chart defined by ``v`` and the augmented
condition ``v'x = 1``.
"""
struct AffineChartHomotopy{H <: AbstractHomotopy} <: AbstractHomotopy
    homotopy::H
    chart::Vector{ComplexF64}
end

Base.size(H::AffineChartHomotopy) = (size(H.homotopy)[1] + 1, size(H.homotopy)[2])

"""
    on_affine_chart(F::System)
    on_affine_chart(H::AbstractHomotopy[, chart])

Construct an `AffineChartSystem` (resp. `AffineChartHomotopy`) on a randomly
generated chart `v`. Each entry is drawn independently from a normal
distribution.
"""
on_affine_chart(F::System) =
    AffineChartSystem(F.evaluator, randn(ComplexF64, size(F.evaluator)[2]))
on_affine_chart(
    H::AbstractHomotopy,
    chart::Vector{ComplexF64} = randn(ComplexF64, size(H)[2]),
) = AffineChartHomotopy(H, chart)

"""
    on_chart!(x, H::AffineChartHomotopy)

Rescale `x` in place so that the chart row is satisfied: `v'x = 1`.
"""
function on_chart!(x::AbstractVector{ComplexF64}, H::AffineChartHomotopy)::Nothing
    on_chart!(x, H.chart)
    return nothing
end

function on_chart!(x::AbstractVector{ComplexF64}, chart::Vector{ComplexF64})::Nothing
    λ = zero(ComplexF64)
    @inbounds for i in eachindex(chart)
        λ += chart[i] * x[i]
    end
    # A point orthogonal to the chart normal (v'x = 0) cannot be placed on the
    # chart; guard rather than silently scaling by inv(0) = Inf/NaN (the plain
    # `inv` avoids `@fastmath` swallowing the non-finite result).
    iszero(λ) && throw(
        ArgumentError(
            "point cannot be placed on the affine chart: it is orthogonal to the chart normal (v'x = 0)",
        ),
    )
    λ⁻¹ = inv(λ)
    @inbounds for i in eachindex(chart)
        x[i] *= λ⁻¹
    end
    return nothing
end

@inline function evaluate_chart(chart::Vector{ComplexF64}, x::AbstractVector)
    out = zero(promote_type(ComplexF64, eltype(x)))
    @inbounds for i in eachindex(chart)
        out += chart[i] * x[i]
    end
    return out - 1.0
end

## AffineChartSystem: AbstractSystem interface (consumed by SystemEvaluator)

function evaluate!(
        u::FSVec{ComplexF64}, F::AffineChartSystem,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(u, F.system, x, p)
    u[size(F.system)[1] + 1] = evaluate_chart(F.chart, x)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, F::AffineChartSystem,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(u, F.system, x, p)
    u[size(F.system)[1] + 1] = ComplexF64(evaluate_chart(F.chart, x))
    return nothing
end

function evaluate!(
        u::FSVec{ComplexDF64}, F::AffineChartSystem,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(u, F.system, x, p)
    u[size(F.system)[1] + 1] = evaluate_chart(F.chart, x)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, F::AffineChartSystem,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    # The interpreter writes only the first m rows (Cartesian indexing);
    # the chart row is written afterwards.
    evaluate_and_jacobian!(u, U, F.system, x, p)
    m = size(F.system)[1]
    u[m + 1] = evaluate_chart(F.chart, x)
    n = size(F.system)[2]
    @inbounds for j in 1:n
        U[m + 1, j] = F.chart[j]
    end
    return nothing
end

# The chart row's order-k Taylor remainder is zero: the row is an affine
# linear form and the J·x_k term is excluded by the call-site contract.
function taylor!(
        u::FSVec{ComplexF64}, v::Val{K}, F::AffineChartSystem,
        tx::TaylorVector{N, ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing where {K, N}
    taylor!(u, v, F.system, tx, p)
    u[size(F.system)[1] + 1] = zero(ComplexF64)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, v::Val{K}, F::AffineChartSystem,
        tx::TaylorVector{N, ComplexF64}, tp::TaylorVector{M, ComplexF64},
    )::Nothing where {K, N, M}
    taylor!(u, v, F.system, tx, tp)
    u[size(F.system)[1] + 1] = zero(ComplexF64)
    return nothing
end

## AffineChartHomotopy: AbstractHomotopy interface

function evaluate!(
        u::FSVec{ComplexF64}, H::AffineChartHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    evaluate!(u, H.homotopy, x, t)
    u[size(H.homotopy)[1] + 1] = evaluate_chart(H.chart, x)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, H::AffineChartHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    evaluate!(u, H.homotopy, x, t)
    u[size(H.homotopy)[1] + 1] = ComplexF64(evaluate_chart(H.chart, x))
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        H::AffineChartHomotopy, x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    # The wrapped homotopy writes its rows with explicit 2D indexing; the chart
    # row is written afterwards.
    evaluate_and_jacobian!(u, U, H.homotopy, x, t)
    m = size(H.homotopy)[1]
    u[m + 1] = evaluate_chart(H.chart, x)
    n = size(H.homotopy)[2]
    @inbounds for j in 1:n
        U[m + 1, j] = H.chart[j]
    end
    return nothing
end

# Chart row contributes zero to every Taylor coefficient (affine linear form
# independent of t; the J·x_k term is excluded by the call-site contract).
function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::AffineChartHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    taylor!(u, Val(1), H.homotopy, x, t)
    u[size(H.homotopy)[1] + 1] = zero(ComplexF64)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::AffineChartHomotopy,
        tx::TaylorVector{3, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    taylor!(u, Val(2), H.homotopy, tx, t; incremental = incremental)
    u[size(H.homotopy)[1] + 1] = zero(ComplexF64)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::AffineChartHomotopy,
        tx::TaylorVector{4, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    taylor!(u, Val(3), H.homotopy, tx, t; incremental = incremental)
    u[size(H.homotopy)[1] + 1] = zero(ComplexF64)
    return nothing
end

# Tracker-facing solution transfer is the identity (decision 8); the monodromy
# solver renormalizes via on_chart! through the concrete handle.
function set_solution!(
        x::FSVec{ComplexF64}, ::AffineChartHomotopy,
        y::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(x, y)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, ::AffineChartHomotopy,
        x::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(out, x)
    return nothing
end

# Retargeting passthroughs for subspace-homotopy wrapping. Without these,
# start_parameters!/target_parameters! would fall through to the AbstractHomotopy
# no-op default and silently leave the inner homotopy on stale subspaces.
set_subspaces!(
    H::AffineChartHomotopy{<:SubspaceHomotopy},
    start::LinearSubspace, target::LinearSubspace,
)::Nothing = set_subspaces!(H.homotopy, start, target)
start_parameters!(
    H::AffineChartHomotopy{<:SubspaceHomotopy}, p::LinearSubspace,
)::Nothing = start_parameters!(H.homotopy, p)
target_parameters!(
    H::AffineChartHomotopy{<:SubspaceHomotopy}, q::LinearSubspace,
)::Nothing = target_parameters!(H.homotopy, q)

# Retargeting passthroughs for parameter-homotopy wrapping (projective systems
# with parameters).
parameters!(
    H::AffineChartHomotopy{ParameterHomotopy},
    p::Vector{ComplexF64}, q::Vector{ComplexF64},
)::Nothing = parameters!(H.homotopy, p, q)
start_parameters!(
    H::AffineChartHomotopy{ParameterHomotopy}, p::AbstractVector{<:Number},
)::Nothing = start_parameters!(H.homotopy, p)
target_parameters!(
    H::AffineChartHomotopy{ParameterHomotopy}, q::AbstractVector{<:Number},
)::Nothing = target_parameters!(H.homotopy, q)

"""
    linear_subspace_homotopy(F::System, V::LinearSubspace, W::LinearSubspace;
                             intrinsic = nothing, gamma = cis(2π * rand()))

Constructs an [`IntrinsicSubspaceHomotopy`](@ref) (if `dim(V) <= codim(V)`, or
forced via `intrinsic = true`) or an [`ExtrinsicSubspaceHomotopy`](@ref)
(if `dim(V) > codim(V)`, or forced via `intrinsic = false`). For a homogeneous
system with linear subspaces, the problem is put on a random affine chart
(intrinsic: chart row appended to the system; extrinsic: homotopy wrapped in
[`AffineChartHomotopy`](@ref)).
"""
function linear_subspace_homotopy(
        F::System,
        V::LinearSubspace,
        W::LinearSubspace;
        intrinsic::Union{Nothing, Bool} = nothing,
        gamma::Union{Nothing, ComplexF64} = cis(2 * pi * rand()),
    )
    use_intrinsic = intrinsic === nothing ? dim(V) <= codim(V) : intrinsic
    projective = is_linear(V) && is_linear(W) && is_homogeneous(F)
    return if use_intrinsic
        if projective
            IntrinsicSubspaceHomotopy(
                SystemEvaluator(on_affine_chart(F)), V, W; gamma = gamma,
            )
        else
            IntrinsicSubspaceHomotopy(F.evaluator, V, W; gamma = gamma)
        end
    else
        H = ExtrinsicSubspaceHomotopy(F.evaluator, V, W; gamma = gamma)
        if projective
            on_affine_chart(H)
        else
            H
        end
    end
end
