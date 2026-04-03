## PathResult — per-path outcome from homotopy continuation tracking.

@enumx PathResultCode::Int8 begin
    PATH_SUCCESS
    PATH_AT_INFINITY
    PATH_AT_ZERO
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
    singular::Bool
    accepted_steps::Int
    rejected_steps::Int
    steps_eg::Int
    extended_precision_used::Bool
    last_path_point::Vector{ComplexF64}
    last_path_t::Float64
end

"""
    _add_steps(r, accepted, rejected)

Return a copy of `r` with additional accepted/rejected steps added (e.g. from a prior phase).
"""
function _add_steps(r::PathResult, accepted::Int, rejected::Int)::PathResult
    return PathResult(
        r.return_code, r.solution, r.t, r.accuracy, r.condition_jacobian,
        r.winding_number, r.singular,
        r.accepted_steps + accepted, r.rejected_steps + rejected,
        r.steps_eg, r.extended_precision_used, r.last_path_point, r.last_path_t,
    )
end

is_success(r::PathResult)::Bool = r.return_code == PathResultCode.PATH_SUCCESS
is_singular(r::PathResult)::Bool = is_success(r) && r.singular
is_nonsingular(r::PathResult)::Bool = is_success(r) && !r.singular
is_at_infinity(r::PathResult)::Bool = r.return_code == PathResultCode.PATH_AT_INFINITY

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

function _endgame_code_to_path_code(code::EndgameCode.T)::PathResultCode.T
    if code == EndgameCode.SUCCESS
        return PathResultCode.PATH_SUCCESS
    elseif code == EndgameCode.AT_INFINITY
        return PathResultCode.PATH_AT_INFINITY
    elseif code == EndgameCode.AT_ZERO
        return PathResultCode.PATH_AT_ZERO
    elseif code == EndgameCode.TERMINATED_MAX_STEPS ||
            code == EndgameCode.TERMINATED_MAX_EXTENDED_STEPS ||
            code == EndgameCode.TERMINATED_MAX_WINDING_NUMBER
        return PathResultCode.PATH_TERMINATED_MAX_STEPS
    elseif code == EndgameCode.TERMINATED_ACCURACY_LIMIT
        return PathResultCode.PATH_TERMINATED_ACCURACY
    elseif code == EndgameCode.TERMINATED_ILL_CONDITIONED
        return PathResultCode.PATH_TERMINATED_ILL_CONDITIONED
    elseif code == EndgameCode.TERMINATED_INVALID_STARTVALUE
        return PathResultCode.PATH_TERMINATED_INVALID_START
    elseif code == EndgameCode.TERMINATED_STEP_SIZE_TOO_SMALL
        return PathResultCode.PATH_TERMINATED_STEP_SIZE
    else
        return PathResultCode.PATH_TERMINATED_MAX_STEPS
    end
end

# Tracker-only PathResult — used for polyhedral toric phase failures (no endgame needed)
function PathResult(tracker::Tracker)
    state = tracker.state
    return PathResult(
        _tracker_code_to_path_code(state.code),
        Vector{ComplexF64}(state.x),
        real(state.segment.t),
        state.accuracy,
        state.cond_J_ẋ,
        0,
        false,
        state.accepted_steps,
        state.rejected_steps,
        0,
        state.used_extended_prec,
        Vector{ComplexF64}(state.x),
        real(state.segment.t),
    )
end

function PathResult(eg::EndgameTracker)
    state = eg.state
    ts = eg.tracker.state
    success = state.code == EndgameCode.SUCCESS

    # Only successful endgame paths report the extrapolated endpoint at t=0.
    # Failed or truncated paths report the actual last tracker point.
    solution = if success
        Vector{ComplexF64}(state.solution)
    else
        Vector{ComplexF64}(ts.x)
    end

    # Report t=0 for endgame success, otherwise the tracker's actual terminal t.
    t = if success
        0.0
    else
        real(ts.segment.t)
    end

    # Accuracy: use endgame state (populated by tracking_stopped! or singular path)
    # Fall back to tracker accuracy if endgame accuracy was never set
    accuracy = isnan(state.accuracy) ? ts.accuracy : state.accuracy

    return PathResult(
        _endgame_code_to_path_code(state.code),
        solution,
        t,
        accuracy,
        state.cond,
        state.winding_number,
        state.singular,
        ts.accepted_steps,
        ts.rejected_steps,
        state.steps_eg,
        ts.used_extended_prec,
        Vector{ComplexF64}(ts.x),  # last_path_point: tracker's actual position
        real(ts.segment.t),        # last_path_t: tracker's actual t
    )
end
