# ---------------------------------------------------------------------------
# Singular endgame helpers
# ---------------------------------------------------------------------------

function add_sample!(eg::EndgameTracker, idx::Int)::Nothing
    state = eg.state
    tracker = eg.tracker
    m = state.winding_number
    t = real(tracker.state.segment.t)
    s = nthroot(t, m)
    μ_s = m > 1 ? m * s^(m - 1) : 1.0
    n = length(state.solution)

    # Rolling buffer: fill slots 1-3, then shift
    if idx >= 3
        state.samples[1], state.samples[2], state.samples[3] =
            state.samples[2], state.samples[3], state.samples[1]
        state.sample_times[1] = state.sample_times[2]
        state.sample_times[2] = state.sample_times[3]
        state.sample_conds[1] = state.sample_conds[2]
        state.sample_conds[2] = state.sample_conds[3]
    end
    slot = min(idx + 1, 3)

    @inbounds for i in 1:n
        state.samples[slot].data[1, i] = tracker.state.x[i]
        state.samples[slot].data[2, i] = μ_s * tracker.predictor.tx3.data[2, i]
    end
    state.sample_times[slot] = s
    state.sample_conds[slot] = scaled_cond(
        tracker.state.jacobian.workspace,
        state.unit_scaling,
        state.col_scaling,
    )
    return nothing
end

function cubic_hermite!(
        x_hat::FSVec{ComplexF64},
        ty0::TaylorVector{2, ComplexF64}, s0::Float64,
        ty1::TaylorVector{2, ComplexF64}, s1::Float64,
        s_target::Float64,
    )::Nothing
    n = length(x_hat)
    s = (s_target - s0) / (s1 - s0)
    h00 = (1.0 + 2.0 * s) * (1.0 - s)^2
    h10 = (s_target - s0) * (1.0 - s)^2
    h01 = s^2 * (3.0 - 2.0 * s)
    h11 = (s_target - s0) * s * (s - 1.0)

    @inbounds for i in 1:n
        y0 = ty0.data[1, i]
        dy0 = ty0.data[2, i]
        y1 = ty1.data[1, i]
        dy1 = ty1.data[2, i]
        x_hat[i] = h00 * y0 + h10 * dy0 + h01 * y1 + h11 * dy1
    end
    return nothing
end

function predict_endpoint!(eg::EndgameTracker)::Float64
    state = eg.state
    state.singular_steps >= 2 || return Inf

    # Bootstrap: first call with 3 samples — compute initial prediction from [1] and [2]
    if state.singular_steps == 2
        cubic_hermite!(
            state.prediction,
            state.samples[1], state.sample_times[1],
            state.samples[2], state.sample_times[2],
            0.0,
        )
    end

    # Save current prediction before overwriting
    copyto!(state.prev_prediction, state.prediction)

    # Compute new prediction from samples[2] and samples[3]
    cubic_hermite!(
        state.prediction,
        state.samples[2], state.sample_times[2],
        state.samples[3], state.sample_times[3],
        0.0,
    )

    # Accuracy estimate (MSW92 eq. 7)
    p = state.sample_times[3] / state.sample_times[2]
    diff = inf_distance(state.prediction, state.prev_prediction)
    norm_pred = inf_norm(state.prediction)
    acc = diff / abs(p^4 - 1.0)
    if norm_pred > 1.0e-8
        acc /= norm_pred
    end
    return acc
end

function predict_and_finalize!(eg::EndgameTracker, max_steps::Bool)::Nothing
    state = eg.state
    tracker = eg.tracker
    val = eg.val
    opts = eg.options
    m = state.winding_number
    n = length(state.solution)

    κ_sample = state.sample_conds[min(state.singular_steps + 1, 3)]

    # Zero-clamp coordinates with small valuation
    zero_cond = 1.0 / (m + 1)
    clamped = 0.0
    @inbounds for i in 1:n
        if val.val_x[i] < zero_cond
            state.solution[i] = state.prediction[i]
        else
            state.solution[i] = zero(ComplexF64)
            clamped = max(clamped, fast_abs(state.prediction[i]))
        end
    end
    # Error of the clamped vector, bounded by the prediction error plus the
    # displacement clamping introduced. Relative, as `state.accuracy` is.
    norm_sol = inf_norm(state.solution)
    acc_clamped = state.accuracy + (norm_sol > 1.0e-8 ? clamped / norm_sol : clamped)

    # Compute condition number at t=0
    ws = tracker.state.jacobian.workspace
    evaluate_and_jacobian!(
        tracker.corrector.r, ws.A, tracker.homotopy,
        state.solution, complex(0.0),
    )
    updated!(ws)
    κ_0 = scaled_cond(ws, state.unit_scaling, state.col_scaling)
    J0_norm = row_scaled_inf_norm_matrix(ws, state.unit_scaling)

    # Acceptance criteria for singular endpoint prediction
    accepted = state.accuracy < opts.singular_min_accuracy && (
        (
            m > 1 && κ_sample > opts.min_cond &&
                nanmax(κ_0, inv(J0_norm)) > κ_sample
        ) ||
            (m == 1 && κ_0 > 1.0e12) ||
            max_steps ||
            (n == 1 && inv(J0_norm) < opts.min_cond)
    )

    if accepted
        state.cond = max(κ_0, inv(J0_norm))
        state.singular = true
        state.code = EndgameCode.SUCCESS
    elseif !max_steps
        latch_best_singular!(state, opts, acc_clamped)
        switch_to_regular!(eg)
    else
        state.code = EndgameCode.TERMINATED_MAX_STEPS
    end
    return nothing
end

# ---------------------------------------------------------------------------
# singular_endgame_step!
# ---------------------------------------------------------------------------

function singular_endgame_step!(eg::EndgameTracker)::Nothing
    state = eg.state
    tracker = eg.tracker
    opts = eg.options
    λ = opts.lambda
    t_current = real(tracker.state.segment.t)
    t_new = λ * t_current

    # Track inner tracker to next geometric point via lightweight segment reinit
    # (preserves x, counters, norm, Jacobian, last_steps_failed)
    reinit!(tracker.state.segment, complex(t_current), complex(t_new))
    tracker.state.code = TrackerCode.TRACKING
    tracker.state.Δs_prev = 0.0
    Δs = _compute_initial_stepsize(
        tracker.state, tracker.predictor, tracker.options, tracker.constants,
    )
    propose_step!(tracker.state.segment, Δs)

    while tracker.state.code == TrackerCode.TRACKING
        step!(tracker)
        state.steps_eg += 1
        max_steps = false
        if state.steps_eg >= opts.max_endgame_steps
            max_steps = true
        elseif ext_steps(tracker.state) - state.ext_steps_eg_start >=
                opts.max_endgame_extended_steps
            max_steps = true
        end
        if max_steps
            if !isnan(state.accuracy) && state.accuracy < opts.singular_min_accuracy
                state.singular = true
                state.code = EndgameCode.SUCCESS
            elseif state.singular_steps >= 2
                predict_and_finalize!(eg, true)
            elseif !check_at_infinity_at_giveup!(eg)
                state.code = EndgameCode.TERMINATED_MAX_STEPS
            end
            return nothing
        end
    end

    # Inner tracker failed to reach t_new — attempt finalization with existing
    # samples if we have enough, otherwise give up.
    if tracker.state.code != TrackerCode.TRACKER_SUCCESS
        if state.singular_steps >= 2
            predict_and_finalize!(eg, true)
        else
            state.accuracy = tracker.state.accuracy
            copyto!(state.solution, tracker.state.x)
            state.cond = tracker.state.cond_J_ẋ
            state.code = tracker_code_to_endgame_code(tracker.state.code)
        end
        return nothing
    end

    # Update valuation at new point
    update!(eg.val, tracker.predictor, t_new)

    # Check winding number consistency
    n = length(state.solution)
    m̂, m̂_err = estimate_winding_number(eg.val, n, opts.max_winding_number)
    if m̂_err > 0.1 || m̂ != state.winding_number
        latch_best_singular!(state, opts, state.accuracy)
        switch_to_regular!(eg)
        return nothing
    end

    # Add sample in s-plane (offset by 1: slot 1 is the initial sample from switch_to_singular!)
    add_sample!(eg, state.singular_steps + 1)
    state.singular_steps += 1

    # Need at least 2 completed geometric steps for prediction
    if state.singular_steps < 2
        return nothing
    end

    # Predict endpoint via cubic Hermite
    acc = predict_endpoint!(eg)

    if state.singular_steps == 2
        # Always store the first prediction. Convergence acceptance
        # happens in predict_and_finalize!, not here.
        state.accuracy = acc
        copyto!(state.solution, state.prediction)
        return nothing
    end

    if acc < state.accuracy && state.accuracy > 1.0e-12
        # Accuracy improved — update solution and continue
        state.accuracy = acc
        copyto!(state.solution, state.prediction)
    else
        predict_and_finalize!(eg, false)
    end

    return nothing
end
