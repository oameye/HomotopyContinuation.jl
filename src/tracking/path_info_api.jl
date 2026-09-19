"""
    iterator(H::AbstractHomotopy, x₀, t₁ = 1.0, t₀ = 0.0;
             tracker_options = TrackerOptions()) -> PathIterator

Public single-path iterator for an `AbstractHomotopy`. A private tracker is built
internally, so callers do not need to construct `Tracker` or `HomotopyEvaluator`.
`tracker_options` configures the predictor-corrector path tracker.
"""
function iterator(
        H::AbstractHomotopy, x₀::AbstractVector{<:Number},
        t₁::Real = 1.0, t₀::Real = 0.0;
        tracker_options::TrackerOptions = TrackerOptions(),
    )::PathIterator{Float64}
    return iterator(
        Tracker(HomotopyEvaluator(H); options = tracker_options),
        x₀, t₁, t₀,
    )
end

function iterator(
        H::AbstractHomotopy, x₀::AbstractVector{<:Number},
        t₁::Number, t₀::Number = 0.0;
        tracker_options::TrackerOptions = TrackerOptions(),
    )::PathIterator{ComplexF64}
    return iterator(
        Tracker(HomotopyEvaluator(H); options = tracker_options),
        x₀, ComplexF64(t₁), ComplexF64(t₀),
    )
end

"""
    path_info(H::AbstractHomotopy, x₀, t₁ = 1.0, t₀ = 0.0;
              tracker_options = TrackerOptions()) -> PathInfo

Track one path of `H` and record every attempted step. The tracker is constructed
internally; `tracker_options` exposes the same path-tracking controls used by the
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
