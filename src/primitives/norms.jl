# Norms — InfNorm and WeightedNorm for homotopy continuation.
# Only the infinity norm is used internally. WeightedNorm is hardcoded to infinity norm.
# Weights are stored in FSVec{Float64} for a single concrete type regardless of size.

struct InfNorm end

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
# inf_norm: max_i |x_i|, using abs2 for speed with isinf fallback for overflow.
# FSVec is always 1-based, so we initialize with x[1] and loop from 2.
# ---------------------------------------------------------------------------

"""
    inf_norm(x::AbstractVector)::Float64

Compute the infinity norm ``\\max_i |x_i|``.
Uses `abs2` for speed; falls back to `abs` if overflow is detected.
"""
function inf_norm(x::AbstractVector)::Float64
    n = length(x)
    @inbounds dmax = abs2(x[1])
    for i in 2:n
        @inbounds dᵢ = abs2(x[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    d = sqrt(dmax)
    if isinf(d)
        @inbounds dmax = abs(x[1])
        for i in 2:n
            @inbounds dᵢ = abs(x[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    end
    return d
end

# ---------------------------------------------------------------------------
# inf_distance: max_i |x_i - y_i|
# ---------------------------------------------------------------------------

"""
    inf_distance(x::AbstractVector, y::AbstractVector)::Float64

Compute the infinity-norm distance ``\\max_i |x_i - y_i|``.
"""
function inf_distance(x::AbstractVector, y::AbstractVector)::Float64
    n = length(x)
    @inbounds dmax = abs2(x[1] - y[1])
    for i in 2:n
        @inbounds dᵢ = abs2(x[i] - y[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    d = sqrt(dmax)
    if isinf(d)
        @inbounds dmax = abs(x[1] - y[1])
        for i in 2:n
            @inbounds dᵢ = abs(x[i] - y[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    end
    return d
end

# ---------------------------------------------------------------------------
# weighted_norm: max_i |x_i / w_i|
# ---------------------------------------------------------------------------

"""
    weighted_norm(x::AbstractVector, w::WeightedNorm)::Float64

Compute ``||D^{-1}x||_\\infty = \\max_i |x_i / w_i|``.
"""
function weighted_norm(x::AbstractVector, w::WeightedNorm)::Float64
    n = length(x)
    weights = w.weights
    @inbounds dmax = abs2(x[1] / weights[1])
    for i in 2:n
        @inbounds dᵢ = abs2(x[i] / weights[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    d = sqrt(dmax)
    if isinf(d)
        @inbounds dmax = abs(x[1] / weights[1])
        for i in 2:n
            @inbounds dᵢ = abs(x[i] / weights[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    end
    return d
end

# ---------------------------------------------------------------------------
# weighted_distance: max_i |(x_i - y_i) / w_i|
# ---------------------------------------------------------------------------

"""
    weighted_distance(x::AbstractVector, y::AbstractVector, w::WeightedNorm)::Float64

Compute ``||D^{-1}(x-y)||_\\infty = \\max_i |(x_i - y_i) / w_i|``.
"""
function weighted_distance(
        x::AbstractVector,
        y::AbstractVector,
        w::WeightedNorm,
    )::Float64
    n = length(x)
    weights = w.weights
    @inbounds dmax = abs2((x[1] - y[1]) / weights[1])
    for i in 2:n
        @inbounds dᵢ = abs2((x[i] - y[i]) / weights[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    d = sqrt(dmax)
    if isinf(d)
        @inbounds dmax = abs((x[1] - y[1]) / weights[1])
        for i in 2:n
            @inbounds dᵢ = abs((x[i] - y[i]) / weights[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    end
    return d
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
