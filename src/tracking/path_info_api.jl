"""
    iterator(H::AbstractHomotopy, x₀, t₁ = 1.0, t₀ = 0.0;
             tracker_options = TrackerOptions()) -> PathIterator

Iterate over the accepted steps of the path of `H` starting at `x₀` from `t₁` to
`t₀`, yielding the tuple `(x, t)` in each iteration. `t` is a `Float64` when `t₁`
and `t₀` are both real and a `ComplexF64` otherwise. The first iteration yields
the start point; iteration ends at the target or as soon as the path fails.
`tracker_options` configures the predictor-corrector path tracker.
"""
function iterator(
        H::AbstractHomotopy, x₀::AbstractVector{<:Number},
        t₁::Real = 1.0, t₀::Real = 0.0;
        tracker_options::TrackerOptions = TrackerOptions(),
    )::PathIterator{Float64}
    return _path_iterator(
        Tracker(HomotopyEvaluator(H); options = tracker_options),
        x₀, ComplexF64(t₁), ComplexF64(t₀), Float64,
    )
end

function iterator(
        H::AbstractHomotopy, x₀::AbstractVector{<:Number},
        t₁::Number, t₀::Number = 0.0;
        tracker_options::TrackerOptions = TrackerOptions(),
    )::PathIterator{ComplexF64}
    return _path_iterator(
        Tracker(HomotopyEvaluator(H); options = tracker_options),
        x₀, ComplexF64(t₁), ComplexF64(t₀), ComplexF64,
    )
end

function _path_iterator(
        tracker::Tracker, x₀::AbstractVector{<:Number},
        t₁::ComplexF64, t₀::ComplexF64, ::Type{T},
    )::PathIterator{T} where {T <: Union{Float64, ComplexF64}}
    init!(tracker, x₀, t₁, t₀)
    return PathIterator{T}(tracker)
end

"""
    path_info(H::AbstractHomotopy, x₀, t₁ = 1.0, t₀ = 0.0;
              tracker_options = TrackerOptions()) -> PathInfo

Track one path of `H` from `t₁` to `t₀` and record every attempted step.
`tracker_options` exposes the same path-tracking controls used by the
continuation algorithms.
"""
function path_info(
        H::AbstractHomotopy, x₀::AbstractVector{<:Number},
        t₁::Number = 1.0, t₀::Number = 0.0;
        tracker_options::TrackerOptions = TrackerOptions(),
    )::PathInfo
    return path_info(
        Tracker(HomotopyEvaluator(H); options = tracker_options),
        x₀, t₁, t₀,
    )
end

"""
    is_success(info::PathInfo) -> Bool

Whether single-path diagnostic tracking reached its requested target.
"""
is_success(info::PathInfo)::Bool = info.return_code == TrackerCode.TRACKER_SUCCESS
