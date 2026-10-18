# ---------------------------------------------------------------------------
# step!
# ---------------------------------------------------------------------------

function step!(eg::EndgameTracker)::Nothing
    state = eg.state
    tracker = eg.tracker
    opts = eg.options

    # Check endgame step limits
    if state.in_endgame && state.steps_eg >= opts.max_endgame_steps
        if !isnan(state.accuracy) && state.accuracy < opts.singular_min_accuracy
            state.code = EndgameCode.SUCCESS
        elseif !check_at_infinity_at_giveup!(eg)
            state.code = EndgameCode.TERMINATED_MAX_STEPS
        end
        return nothing
    end

    if state.in_endgame
        eg_ext_steps = ext_steps(tracker.state) - state.ext_steps_eg_start
        if eg_ext_steps >= opts.max_endgame_extended_steps
            if !isnan(state.accuracy) && state.accuracy < opts.singular_min_accuracy
                state.code = EndgameCode.SUCCESS
            elseif !check_at_infinity_at_giveup!(eg)
                state.code = EndgameCode.TERMINATED_MAX_EXTENDED_STEPS
            end
            return nothing
        end
    end

    # Singular endgame phase
    if state.in_singular_endgame
        singular_endgame_step!(eg)
        return nothing
    end

    # Track whether the proposed step was trying to reach t=0.
    # This gates singular endgame entry for m=1 paths: only enter the singular
    # endgame if a previous step already attempted to jump directly to zero.
    # For a backward segment (1→0), s′=0.0 means the step targets the endpoint.
    seg = tracker.state.segment
    is_jump_to_zero = !seg.forward && seg.s′ == 0.0

    # Regular tracker step
    accepted = step!(tracker)

    if tracker.state.code != TrackerCode.TRACKING
        tracking_stopped!(eg)
        return nothing
    end

    t = real(tracker.state.segment.t)

    # Update jump-to-zero history: shift (prev_prev, prev) window.
    state.jump_to_zero_attempted = (state.jump_to_zero_attempted[2], is_jump_to_zero)

    # Pre-endgame: just forward
    if t > opts.endgame_start
        return nothing
    end

    # Enter endgame phase
    if !state.in_endgame
        state.in_endgame = true
        state.ext_steps_eg_start = ext_steps(tracker.state)
    end
    state.steps_eg += 1

    # Only update valuation state after an accepted tracker step.
    # Rejected steps keep the same predictor data and t-value.
    if !accepted
        # If the valuation from previous accepted endgame steps already
        # certifies a singular endpoint, do not burn additional rejected
        # regular-tracking steps near t=0 before switching to the singular endgame.
        # Guard: only when enough valuation samples exist for a reliable winding estimate.
        if state.in_endgame && eg.val.samples >= 3 && check_finite!(eg)
            return nothing
        end
        return nothing
    end

    # Update valuation
    update!(eg.val, tracker.predictor, t)

    # Check for singular endpoint (need ≥2 valuation samples for reliable winding estimate).
    if !opts.only_nonsingular && eg.val.samples >= 2
        if check_finite!(eg)
            return nothing
        end
    end

    # Check for at-infinity (need ≥2 valuation samples for reliable divergence estimate).
    if opts.at_infinity_check && eg.val.samples >= 2
        if check_at_infinity!(eg)
            return nothing
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# track!
# ---------------------------------------------------------------------------

function track!(
        eg::EndgameTracker,
        x₀::AbstractVector{<:Number};
        t₁::ComplexF64 = complex(1.0),
        t₀::ComplexF64 = complex(0.0),
        ω::Float64 = NaN,
        μ::Float64 = NaN,
        extended_precision::Bool = false,
    )::EndgameCode.T
    code = init!(
        eg, x₀, t₁, t₀;
        ω = ω, μ = μ, extended_precision = extended_precision,
    )
    if code != EndgameCode.TRACKING
        return code
    end

    while eg.state.code == EndgameCode.TRACKING
        step!(eg)
    end

    return eg.state.code
end
