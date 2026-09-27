# ---------------------------------------------------------------------------
# check_finite! — detect singular endpoint and switch to singular endgame
# ---------------------------------------------------------------------------

function check_finite!(eg::EndgameTracker)::Bool
    state = eg.state
    val = eg.val
    opts = eg.options
    n = length(state.solution)

    # Guard: only consider the singular endgame if all coordinates have finite
    # valuations. Without this gate, at-infinity paths (val_x ≈ −1) would
    # erroneously enter the singular endgame before check_at_infinity! runs.
    is_finite(
        val;
        finite_tol = opts.val_finite_tol,
        zero_is_finite = !opts.zero_is_at_infinity,
        max_winding_number = opts.max_winding_number,
    ) || return false

    m, m_err = estimate_winding_number(val, n, opts.max_winding_number)

    if m_err < opts.val_finite_tol
        # Winding number is reliable.
        # For m=1: only enter singular endgame if a previous step attempted to
        # jump directly to t=0. This prevents spurious singular endgame entry
        # for paths that are regular but have slightly noisy winding estimates.
        if m == 1 && !state.jump_to_zero_attempted[1]
            return false
        end
    else
        # Winding number not reliable yet
        return false
    end

    if m > opts.max_winding_number
        state.code = EndgameCode.TERMINATED_MAX_WINDING_NUMBER
        return true
    end

    # Switch to singular endgame
    state.winding_number = m
    t = real(eg.tracker.state.segment.t)
    switch_to_singular!(eg, t)
    return true
end

# ---------------------------------------------------------------------------
# switch_to_singular! / switch_to_regular!
# ---------------------------------------------------------------------------

function switch_to_singular!(eg::EndgameTracker, t::Float64)::Nothing
    state = eg.state
    tracker = eg.tracker

    state.in_singular_endgame = true
    state.singular_start = t
    state.singular_steps = 0
    fill!(state.sample_times, 0.0)
    fill!(state.sample_conds, 0.0)
    fill!(state.prediction, zero(ComplexF64))
    fill!(state.prev_prediction, zero(ComplexF64))
    state.prev_accuracy = Inf
    for s in state.samples
        fill!(s.data, zero(ComplexF64))
    end

    # Initialize row/col scaling if not yet done.
    ensure_endgame_scaling!(state, tracker)

    add_sample!(eg, 0)
    tracker.predictor.winding_number = state.winding_number
    # Condition baseline
    state.at_inf_conds[1] = state.sample_conds[1]
    # Latch extended precision
    tracker.state.keep_extended_prec = true
    return nothing
end

# `acc` must describe `state.solution` as it stands, not the Hermite prediction it
# came from: zero-clamping moves the vector away from the point the prediction
# error was measured on, and `tracking_stopped!` compares the latched accuracy
# against a regular endpoint's.
function latch_best_singular!(
        state::EndgameState, opts::EndgameOptions, acc::Float64,
    )::Nothing
    if acc < min(opts.singular_min_accuracy, state.best_singular_accuracy)
        copyto!(state.best_singular, state.solution)
        state.best_singular_accuracy = acc
        state.best_singular_winding = state.winding_number
    end
    return nothing
end

function switch_to_regular!(eg::EndgameTracker)::Nothing
    state = eg.state
    tracker = eg.tracker

    state.in_singular_endgame = false
    state.winding_number = 0
    tracker.predictor.winding_number = 1
    # Lightweight reinit: preserve x, counters, norm, Jacobian state
    resume_from!(tracker, complex(0.0))
    return nothing
end

# ---------------------------------------------------------------------------
# check_at_infinity!
# ---------------------------------------------------------------------------

function check_at_infinity!(eg::EndgameTracker, relaxed::Bool = false)::Bool
    state = eg.state
    val = eg.val
    opts = eg.options
    tracker = eg.tracker
    n = length(state.solution)
    t = real(tracker.state.segment.t)
    κ = NaN

    @inbounds for i in 1:n
        vx = val.val_x[i]
        vtx = val.val_tẋ[i]
        dvx = val.Δval_x[i]
        dvtx = val.Δval_tẋ[i]
        ε∞ = at_infinity_tol_gated(vx, vtx, dvx, dvtx, opts.val_finite_tol, opts.zero_is_at_infinity)

        if !state.at_inf_active[i]
            # Stage 1: mark candidates
            if ε∞ < opts.val_at_infinity_tol
                if all(!, state.at_inf_active)
                    ensure_endgame_scaling!(state, tracker)
                end
                if isnan(κ)
                    κ = scaled_cond(
                        tracker.state.jacobian.workspace,
                        state.unit_scaling,
                        state.col_scaling,
                    )
                end
                state.at_inf_active[i] = true
                state.at_inf_starts[i] = t
                state.at_inf_abs_coords[i] = fast_abs(tracker.state.x[i])
                state.at_inf_conds[i] = κ
            end
        else
            if !(ε∞ < opts.val_at_infinity_tol)
                clear_at_infinity_candidate!(state, i)
                continue
            end

            ensure_endgame_scaling!(state, tracker)
            if isnan(κ)
                κ = scaled_cond(
                    tracker.state.jacobian.workspace,
                    state.unit_scaling,
                    state.col_scaling,
                )
            end

            # Stage 2: confirm divergence
            v = vx
            x_now = fast_abs(tracker.state.x[i])

            coord_growth = if v < 0.0
                x_now / state.at_inf_abs_coords[i]  # growing → at infinity
            else
                state.at_inf_abs_coords[i] / x_now   # shrinking → at zero
            end

            cond_growth = κ / state.at_inf_conds[i]
            min_growth = relaxed ? 1.0 :
                clamp(0.25^(4.0 * abs(v)), 20.0, opts.min_coord_growth)

            if coord_growth > min_growth &&
                    (cond_growth > opts.min_cond_growth || κ > max(1.0e8, opts.min_cond))
                # Copy current tracker state before terminating
                copyto!(state.solution, tracker.state.x)
                state.accuracy = tracker.state.accuracy
                state.cond = κ
                if v > 0.0 && !opts.zero_is_at_infinity
                    state.code = EndgameCode.AT_ZERO
                else
                    state.code = EndgameCode.AT_INFINITY
                end
                return true
            elseif coord_growth < 1.0
                # Growth reversed — deactivate stale candidate
                clear_at_infinity_candidate!(state, i)
            end
        end
    end

    return false
end
