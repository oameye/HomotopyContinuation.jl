## PathResult — per-path outcome from homotopy continuation tracking.

@enumx PathResultCode::Int8 begin
    PATH_SUCCESS
    PATH_AT_INFINITY
    PATH_TERMINATED_ACCURACY
    PATH_TERMINATED_ILL_CONDITIONED
    PATH_TERMINATED_MAX_STEPS
    PATH_TERMINATED_STEP_SIZE
    PATH_TERMINATED_INVALID_START
end

struct PathResult
    return_code::PathResultCode.T
    solution::Vector{ComplexF64}
    t::Float64
    accuracy::Float64
    condition_jacobian::Float64
    winding_number::Int
    accepted_steps::Int
    rejected_steps::Int
end

is_success(r::PathResult)::Bool = r.return_code == PathResultCode.PATH_SUCCESS

function is_real(r::PathResult; tol::Float64 = DEFAULT_REAL_TOL)::Bool
    return is_success(r) && all(x -> abs(imag(x)) < tol * max(1.0, abs(x)), r.solution)
end

function _tracker_code_to_path_code(code::TrackerCode.T)::PathResultCode.T
    if code == TrackerCode.TRACKER_SUCCESS
        return PathResultCode.PATH_SUCCESS
    elseif code == TrackerCode.TERMINATED_MAX_STEPS
        return PathResultCode.PATH_TERMINATED_MAX_STEPS
    elseif code == TrackerCode.TERMINATED_ACCURACY_LIMIT
        return PathResultCode.PATH_TERMINATED_ACCURACY
    elseif code == TrackerCode.TERMINATED_ILL_CONDITIONED
        return PathResultCode.PATH_TERMINATED_ILL_CONDITIONED
    elseif code == TrackerCode.TERMINATED_INVALID_STARTVALUE
        return PathResultCode.PATH_TERMINATED_INVALID_START
    elseif code == TrackerCode.TERMINATED_STEP_SIZE_TOO_SMALL
        return PathResultCode.PATH_TERMINATED_STEP_SIZE
    else
        return PathResultCode.PATH_TERMINATED_MAX_STEPS
    end
end

function PathResult(tracker::Tracker)
    state = tracker.state
    return PathResult(
        _tracker_code_to_path_code(state.code),
        Vector{ComplexF64}(state.x),
        real(state.segment.t),
        state.accuracy,
        state.cond_J_ẋ,
        0,
        state.accepted_steps,
        state.rejected_steps,
    )
end
