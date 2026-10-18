# Norms — InfNorm, EuclideanNorm and WeightedNorm for homotopy continuation.
# Only the infinity norm is used internally. WeightedNorm is hardcoded to infinity norm.
# Weights are stored in FSVec{Float64} for a single concrete type regardless of size.

"""
    InfNorm()

The infinity norm ``\\max_i |x_i|``, as a value that can be passed wherever a
`distance` is taken (`UniquePoints`, `multiplicities`, `unique_points`,
`monodromy`). This is the default everywhere.
"""
struct InfNorm end

"""
    EuclideanNorm()

The Euclidean norm ``\\sqrt{\\sum_i |x_i|^2}``, as a value that can be passed
wherever a `distance` is taken (`UniquePoints`, `multiplicities`,
`unique_points`, `monodromy`).
"""
struct EuclideanNorm end

# Squared moduli overflow above about 1.3e154 and lose precision below about
# 1.5e-154. A sum of squares outside [floatmin, Inf) therefore triggers a second
# pass that divides by the largest modulus first, as `LinearAlgebra.norm` does.
# NaN fails both comparisons and propagates from the first pass.
@inline sumsq_out_of_range(s::Real) = s < floatmin(s) || isinf(s)

"""
    euclidean_norm(x::AbstractVector)

Compute the Euclidean norm ``\\sqrt{\\sum_i |x_i|^2}`` without overflow or
underflow in the squared moduli.
"""
@inline function euclidean_norm(x::AbstractVector)
    s = zero(float(real(eltype(x))))
    @inbounds @simd for i in eachindex(x)
        s += abs2(x[i])
    end
    sumsq_out_of_range(s) && return scaled_euclidean_norm(x)
    return sqrt(s)
end

@noinline function scaled_euclidean_norm(x::AbstractVector)
    m = zero(float(real(eltype(x))))
    @inbounds for i in eachindex(x)
        m = max(m, abs(x[i]))
    end
    (iszero(m) || !isfinite(m)) && return m
    s = zero(m)
    @inbounds for i in eachindex(x)
        s += abs2(x[i] / m)
    end
    return m * sqrt(s)
end

"""
    euclidean_distance(x::AbstractVector, y::AbstractVector)

Compute the Euclidean distance ``\\sqrt{\\sum_i |x_i - y_i|^2}`` without
overflow or underflow in the squared moduli.
"""
@inline function euclidean_distance(x::AbstractVector, y::AbstractVector)
    s = zero(float(real(promote_type(eltype(x), eltype(y)))))
    @inbounds @simd for i in eachindex(x, y)
        s += abs2(x[i] - y[i])
    end
    sumsq_out_of_range(s) && return scaled_euclidean_distance(x, y)
    return sqrt(s)
end

@noinline function scaled_euclidean_distance(x::AbstractVector, y::AbstractVector)
    m = zero(float(real(promote_type(eltype(x), eltype(y)))))
    @inbounds for i in eachindex(x, y)
        m = max(m, abs(x[i] - y[i]))
    end
    (iszero(m) || !isfinite(m)) && return m
    s = zero(m)
    @inbounds for i in eachindex(x, y)
        s += abs2((x[i] - y[i]) / m)
    end
    return m * sqrt(s)
end

Base.@kwdef struct WeightedNormOptions
    scale_min::Float64 = 1.0e-4
    scale_abs_min::Float64 = 1.0e-6
    scale_max::Float64 = exp2(511)
end

"""
    WeightedNorm(weights, options)

A weighted infinity norm ``||D⁻¹x||_∞`` where ``D`` is the diagonal matrix with
diagonal `weights`. Weights are updated dynamically via `init!` and `update!`
so that ``||D⁻¹x||_∞ ≈ 1``.
"""
struct WeightedNorm
    weights::FSVec{Float64}
    options::WeightedNormOptions
end

WeightedNorm(weights::FSVec{Float64}) = WeightedNorm(weights, WeightedNormOptions())
WeightedNorm(n::Integer) = WeightedNorm(FSVec{Float64}(ones(n)), WeightedNormOptions())

Base.length(w::WeightedNorm) = length(w.weights)

# ---------------------------------------------------------------------------
# inf_norm: max_i |x_i|, using abs2 for speed with an `abs` fallback when the
# largest squared modulus overflows or underflows.
# FSVec is always 1-based, so we initialize with x[1] and loop from 2.
# ---------------------------------------------------------------------------

"""
    inf_norm(x::AbstractVector)

Compute the infinity norm ``\\max_i |x_i|``.
Uses `abs2` for speed; falls back to `abs` when the largest squared modulus
overflows or underflows.
"""
@inline function inf_norm(x::AbstractVector)
    n = length(x)
    @inbounds dmax = abs2(x[1])
    for i in 2:n
        @inbounds dᵢ = abs2(x[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    if sumsq_out_of_range(dmax)
        @inbounds dmax = abs(x[1])
        for i in 2:n
            @inbounds dᵢ = abs(x[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    end
    return sqrt(dmax)
end

# ---------------------------------------------------------------------------
# inf_distance: max_i |x_i - y_i|
# ---------------------------------------------------------------------------

"""
    inf_distance(x::AbstractVector, y::AbstractVector)

Compute the infinity-norm distance ``\\max_i |x_i - y_i|``.
"""
@inline function inf_distance(x::AbstractVector, y::AbstractVector)
    n = length(x)
    @inbounds dmax = abs2(x[1] - y[1])
    for i in 2:n
        @inbounds dᵢ = abs2(x[i] - y[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    if sumsq_out_of_range(dmax)
        @inbounds dmax = abs(x[1] - y[1])
        for i in 2:n
            @inbounds dᵢ = abs(x[i] - y[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    end
    return sqrt(dmax)
end

# ---------------------------------------------------------------------------
# weighted_norm: max_i |x_i / w_i|
# ---------------------------------------------------------------------------

"""
    weighted_norm(x::AbstractVector, w::WeightedNorm)

Compute ``||D^{-1}x||_\\infty = \\max_i |x_i / w_i|``.
"""
@inline function weighted_norm(x::AbstractVector, w::WeightedNorm)
    n = length(x)
    weights = w.weights
    @inbounds dmax = abs2(x[1] / weights[1])
    for i in 2:n
        @inbounds dᵢ = abs2(x[i] / weights[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    if sumsq_out_of_range(dmax)
        @inbounds dmax = abs(x[1] / weights[1])
        for i in 2:n
            @inbounds dᵢ = abs(x[i] / weights[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    end
    return sqrt(dmax)
end

# ---------------------------------------------------------------------------
# weighted_distance: max_i |(x_i - y_i) / w_i|
# ---------------------------------------------------------------------------

"""
    weighted_distance(x::AbstractVector, y::AbstractVector, w::WeightedNorm)

Compute ``||D^{-1}(x-y)||_\\infty = \\max_i |(x_i - y_i) / w_i|``.
"""
@inline function weighted_distance(
        x::AbstractVector,
        y::AbstractVector,
        w::WeightedNorm,
    )
    n = length(x)
    weights = w.weights
    @inbounds dmax = abs2((x[1] - y[1]) / weights[1])
    for i in 2:n
        @inbounds dᵢ = abs2((x[i] - y[i]) / weights[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    if sumsq_out_of_range(dmax)
        @inbounds dmax = abs((x[1] - y[1]) / weights[1])
        for i in 2:n
            @inbounds dᵢ = abs((x[i] - y[i]) / weights[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    end
    return sqrt(dmax)
end

# ---------------------------------------------------------------------------
# init!: set weights from x so that ||D⁻¹x||_∞ ≈ 1
# ---------------------------------------------------------------------------

"""
    init!(w::WeightedNorm, x::AbstractVector)

Initialize the weights of `w` from `x` so that ``||D^{-1}x||_\\infty \\approx 1``.
"""
function init!(w::WeightedNorm, x::AbstractVector)
    opts = w.options
    point_norm = inf_norm(x)
    weights = w.weights
    for i in eachindex(x)
        @inbounds wᵢ = fast_abs(x[i])
        if wᵢ < opts.scale_min * point_norm
            wᵢ = opts.scale_min * point_norm
        elseif wᵢ > opts.scale_max * point_norm
            wᵢ = opts.scale_max * point_norm
        end
        @inbounds weights[i] = max(wᵢ, opts.scale_abs_min)
    end
    return w
end

# ---------------------------------------------------------------------------
# update!: interpolate weights toward new x
# ---------------------------------------------------------------------------

"""
    update!(w::WeightedNorm, x::AbstractVector)

Update the weights of `w` by interpolating between the previous weights and
the elementwise magnitudes of `x`.
"""
function update!(w::WeightedNorm, x::AbstractVector)
    opts = w.options
    norm_x = weighted_norm(x, w)
    weights = w.weights
    for i in eachindex(x)
        @inbounds wᵢ = (fast_abs(x[i]) + weights[i]) / 2
        if wᵢ < opts.scale_min * norm_x
            wᵢ = opts.scale_min * norm_x
        elseif wᵢ > opts.scale_max * norm_x
            wᵢ = opts.scale_max * norm_x
        end
        if isfinite(wᵢ)
            @inbounds weights[i] = max(wᵢ, opts.scale_abs_min)
        end
    end
    return w
end
