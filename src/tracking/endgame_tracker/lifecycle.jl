# ---------------------------------------------------------------------------
# State reset
# ---------------------------------------------------------------------------

function reset_state!(state::EndgameState)::Nothing
    state.code = EndgameCode.TRACKING
    state.in_endgame = false
    state.in_singular_endgame = false
    state.winding_number = 0
    fill!(state.solution, zero(ComplexF64))
    state.accuracy = NaN
    state.cond = 1.0
    state.singular = false
    state.steps_eg = 0
    state.ext_steps_eg_start = typemax(Int)
    state.jump_to_zero_attempted = (false, false)
    fill!(state.at_inf_starts, NaN)
    fill!(state.at_inf_abs_coords, NaN)
    fill!(state.at_inf_conds, NaN)
    fill!(state.at_inf_active, false)
    for s in state.samples
        fill!(s.data, zero(ComplexF64))
    end
    fill!(state.sample_times, 0.0)
    fill!(state.sample_conds, 0.0)
    state.singular_start = NaN
    state.singular_steps = 0
    fill!(state.prediction, zero(ComplexF64))
    fill!(state.prev_prediction, zero(ComplexF64))
    state.prev_accuracy = Inf
    fill!(state.best_singular, zero(ComplexF64))
    state.best_singular_accuracy = Inf
    state.best_singular_winding = 0
    fill!(state.col_scaling, 0.0)
    return nothing
end

# ---------------------------------------------------------------------------
# Code mapping
# ---------------------------------------------------------------------------

function tracker_code_to_endgame_code(code::TrackerCode.T)::EndgameCode.T
    if code == TrackerCode.TRACKER_SUCCESS
        return EndgameCode.SUCCESS
    elseif code == TrackerCode.TERMINATED_MAX_STEPS
        return EndgameCode.TERMINATED_MAX_STEPS
    elseif code == TrackerCode.TERMINATED_ACCURACY_LIMIT
        return EndgameCode.TERMINATED_ACCURACY_LIMIT
    elseif code == TrackerCode.TERMINATED_ILL_CONDITIONED
        return EndgameCode.TERMINATED_ILL_CONDITIONED
    elseif code == TrackerCode.TERMINATED_INVALID_STARTVALUE
        return EndgameCode.TERMINATED_INVALID_STARTVALUE
    elseif code == TrackerCode.TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN
        return EndgameCode.TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN
    elseif code == TrackerCode.TERMINATED_STEP_SIZE_TOO_SMALL
        return EndgameCode.TERMINATED_STEP_SIZE_TOO_SMALL
    else
        return EndgameCode.TERMINATED_MAX_STEPS
    end
end

# ---------------------------------------------------------------------------
# init!
# ---------------------------------------------------------------------------

function init!(
        eg::EndgameTracker,
        x₀::AbstractVector{<:Number},
        t₁::ComplexF64 = complex(1.0),
        t₀::ComplexF64 = complex(0.0);
        ω::Float64 = NaN,
        μ::Float64 = NaN,
        extended_precision::Bool = false,
        max_initial_step_size::Float64 = Inf,
        keep_steps::Bool = false,
    )::EndgameCode.T
    reset_state!(eg.state)
    init!(eg.val)

    tracker_code = init!(
        eg.tracker, x₀, t₁, t₀;
        ω = ω,
        μ = μ,
        extended_precision = extended_precision,
        max_initial_step_size = max_initial_step_size,
        keep_steps = keep_steps,
    )
    if tracker_code != TrackerCode.TRACKING
        copyto!(eg.state.solution, eg.tracker.state.x)
        eg.state.accuracy = eg.tracker.state.accuracy
        eg.state.cond = eg.tracker.state.cond_J_ẋ
        eg.state.code = tracker_code_to_endgame_code(tracker_code)
        return eg.state.code
    end

    return eg.state.code
end

# ---------------------------------------------------------------------------
# tracking_stopped! — inner tracker reached terminal state
# ---------------------------------------------------------------------------

function tracking_stopped!(eg::EndgameTracker)::Nothing
    state = eg.state
    ts = eg.tracker.state
    opts = eg.options

    state.accuracy = ts.accuracy

    if ts.code == TrackerCode.TRACKER_SUCCESS && state.accuracy > 1.0e-14
        state.accuracy = refine_current_solution!(
            eg.tracker; min_tol = 1.0e-14,
            nsteps = opts.refine_steps
        )
    end

    copyto!(state.solution, ts.x)

    # The singular endgame hands a path back whenever its acceptance test has not
    # fired yet, not because the prediction was bad. If that prediction beat the
    # endpoint the tracker went on to reach, it is the answer.
    if ts.code == TrackerCode.TRACKER_SUCCESS &&
            state.best_singular_accuracy < state.accuracy
        copyto!(state.solution, state.best_singular)
        state.accuracy = state.best_singular_accuracy
        state.winding_number = state.best_singular_winding
        state.singular = true
    end

    if ts.code == TrackerCode.TRACKER_SUCCESS
        @inbounds for i in eachindex(state.col_scaling)
            state.col_scaling[i] = ts.norm.weights[i]
        end
        # Evaluate Jacobian at (solution, t=0) to assess singularity at the target.
        # The tracker's workspace holds the Jacobian at its last position (t > 0),
        # which is not the right place for the singularity classification.
        ws = ts.jacobian.workspace
        evaluate_and_jacobian!(
            eg.tracker.corrector.r, ws.A, eg.tracker.homotopy,
            state.solution, complex(0.0),
        )

        # A spurious endpoint leaves ‖H(solution, 0)‖ far from zero; call it
        # at-infinity rather than a success. Relative to the row scale, so the
        # threshold means the same whether the terms of H are O(1) at the endpoint
        # or O(10^40).
        if max_relative_residual(eg.tracker.corrector.r, ws.A, state.col_scaling) >
                opts.max_residual
            state.code = EndgameCode.AT_INFINITY
            return nothing
        end

        updated!(ws)
        factorize!(ws)
        state.cond = scaled_cond(ws, state.unit_scaling, state.col_scaling)
        if state.cond > opts.sing_cond || state.accuracy > opts.sing_accuracy
            state.singular = true
        end
    end

    if ts.code == TrackerCode.TERMINATED_MAX_STEPS && check_at_infinity_at_giveup!(eg)
        return nothing
    end

    state.code = tracker_code_to_endgame_code(ts.code)
    return nothing
end
