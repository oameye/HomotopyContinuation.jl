# EndgameTracker — wraps Tracker with endgame detection and singular endpoint handling.

using LinearAlgebra: LinearAlgebra as LA

@kwdef struct EndgameOptions
    endgame_start::Float64 = 0.1
    max_endgame_steps::Int = 2000
    max_endgame_extended_steps::Int = 400
    only_nonsingular::Bool = false
    at_infinity_check::Bool = true
    zero_is_at_infinity::Bool = false
    max_winding_number::Int = 6
    val_finite_tol::Float64 = 0.05
    val_at_infinity_tol::Float64 = 0.01
    singular_min_accuracy::Float64 = 1.0e-6
    min_cond::Float64 = 1.0e6
    min_cond_growth::Float64 = 1.0e4
    min_coord_growth::Float64 = 100.0
    sing_cond::Float64 = 1.0e14
    sing_accuracy::Float64 = 1.0e-12
    scaling_threshold::Float64 = -30.0
    max_residual::Float64 = 1.0e-3
    refine_steps::Int = 3
    lambda::Float64 = 0.25
end

@enumx EndgameCode::Int8 begin
    TRACKING
    SUCCESS
    AT_INFINITY
    AT_ZERO
    TERMINATED_MAX_STEPS
    TERMINATED_MAX_EXTENDED_STEPS
    TERMINATED_MAX_WINDING_NUMBER
    TERMINATED_ACCURACY_LIMIT
    TERMINATED_ILL_CONDITIONED
    TERMINATED_INVALID_STARTVALUE
    TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN
    TERMINATED_STEP_SIZE_TOO_SMALL
end

"""
    EndgameState

Mutable endgame-specific state. Wraps all per-path endgame tracking data.
"""
mutable struct EndgameState
    code::EndgameCode.T
    in_endgame::Bool
    in_singular_endgame::Bool
    winding_number::Int
    # Solution
    const solution::FSVec{ComplexF64}
    accuracy::Float64
    cond::Float64
    singular::Bool
    # Step counters
    steps_eg::Int
    ext_steps_eg_start::Int
    # Jump-to-zero tracking: (prev_prev, prev) history of whether the tracker
    # proposed a step reaching t=0. Used to gate singular endgame entry for
    # m=1 paths — prevents spurious singular endgame entry for regular paths.
    jump_to_zero_attempted::Tuple{Bool, Bool}
    # At-infinity per-coordinate tracking
    const at_inf_starts::FSVec{Float64}
    const at_inf_abs_coords::FSVec{Float64}
    const at_inf_conds::FSVec{Float64}
    const at_inf_active::Vector{Bool}
    # Singular endgame
    const samples::Vector{TaylorVector{2, ComplexF64}}
    const sample_times::FSVec{Float64}
    const sample_conds::FSVec{Float64}
    singular_start::Float64
    singular_steps::Int
    const prediction::FSVec{ComplexF64}
    const prev_prediction::FSVec{ComplexF64}
    prev_accuracy::Float64
    # Scaling for condition number
    const row_scaling::FSVec{Float64}
    const col_scaling::FSVec{Float64}
end

function EndgameState(n::Int)
    return EndgameState(
        EndgameCode.TRACKING,
        false, false, 0,                                # code, in_endgame, in_singular, winding
        FSVec{ComplexF64}(zeros(ComplexF64, n)),         # solution
        NaN, 1.0, false,                                 # accuracy, cond, singular
        0, typemax(Int),                                 # steps_eg, ext_steps_eg_start
        (false, false),                                  # jump_to_zero_attempted
        FSVec{Float64}(fill(NaN, n)),                    # at_inf_starts
        FSVec{Float64}(fill(NaN, n)),                    # at_inf_abs_coords
        FSVec{Float64}(fill(NaN, n)),                    # at_inf_conds
        fill(false, n),                                  # at_inf_active
        [TaylorVector{2, ComplexF64}(n) for _ in 1:3],   # samples
        FSVec{Float64}(zeros(3)),                         # sample_times
        FSVec{Float64}(zeros(3)),                         # sample_conds
        NaN, 0,                                           # singular_start, singular_steps
        FSVec{ComplexF64}(zeros(ComplexF64, n)),          # prediction
        FSVec{ComplexF64}(zeros(ComplexF64, n)),          # prev_prediction
        Inf,                                              # prev_accuracy
        FSVec{Float64}(ones(n)),                          # row_scaling
        FSVec{Float64}(zeros(n)),                         # col_scaling
    )
end

struct EndgameTracker
    tracker::Tracker
    state::EndgameState
    val::Valuation
    options::EndgameOptions
end

function EndgameTracker(
        tracker::Tracker,
        options::EndgameOptions = EndgameOptions(),
    )
    n = length(tracker.state.x)
    return EndgameTracker(tracker, EndgameState(n), Valuation(n), options)
end

function _endgame_tracker(
        H::AbstractHomotopy, tracker_options::TrackerOptions,
        endgame_options::EndgameOptions,
    )::EndgameTracker
    return EndgameTracker(
        Tracker(HomotopyEvaluator(H); options = tracker_options), endgame_options,
    )
end

# ---------------------------------------------------------------------------
# State reset
# ---------------------------------------------------------------------------

function _reset_state!(state::EndgameState)::Nothing
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
    fill!(state.row_scaling, 1.0)
    fill!(state.col_scaling, 0.0)
    return nothing
end

# ---------------------------------------------------------------------------
# Code mapping
# ---------------------------------------------------------------------------

function _tracker_code_to_endgame_code(code::TrackerCode.T)::EndgameCode.T
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
    _reset_state!(eg.state)
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
        eg.state.code = _tracker_code_to_endgame_code(tracker_code)
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

        # Residual sanity check: if ‖H(solution, 0)‖ is large, the path diverged
        # to a spurious point. Reclassify as at-infinity rather than success.
        residual = inf_norm(eg.tracker.corrector.r)
        if residual > opts.max_residual
            state.code = EndgameCode.AT_INFINITY
            return nothing
        end

        updated!(ws)
        skeel_row_scaling!(state.row_scaling, ws.A, state.col_scaling)
        factorize!(ws)
        state.cond = _scaled_cond(ws, state.row_scaling, state.col_scaling)
        if state.cond > opts.sing_cond || state.accuracy > opts.sing_accuracy
            state.singular = true
        end
    end

    state.code = _tracker_code_to_endgame_code(ts.code)
    return nothing
end

# ---------------------------------------------------------------------------
# Endgame linear algebra helpers
# ---------------------------------------------------------------------------

@inline function _scaled_inf_norm_matrix(
        ws::MatrixWorkspace,
        row_scaling::FSVec{Float64},
        col_scaling::FSVec{Float64},
    )::Float64
    A = ws.A
    m, n = size(A)
    norm_val = -Inf
    @inbounds for i in 1:m
        row_sum = 0.0
        for j in 1:n
            row_sum += fast_abs(A[i, j]) * col_scaling[j]
        end
        norm_val = @fastmath max(norm_val, row_sum * row_scaling[i])
    end
    return norm_val
end

# Row-scaled-only inf norm for J₀ (no col_scaling — used for singular endgame acceptance)
@inline function _row_scaled_inf_norm_matrix(
        ws::MatrixWorkspace,
        row_scaling::FSVec{Float64},
    )::Float64
    A = ws.A
    m, n = size(A)
    norm_val = -Inf
    @inbounds for i in 1:m
        row_sum = 0.0
        for j in 1:n
            row_sum += fast_abs(A[i, j])
        end
        norm_val = @fastmath max(norm_val, row_sum * row_scaling[i])
    end
    return norm_val
end

function _scaled_cond(
        ws::MatrixWorkspace,
        row_scaling::FSVec{Float64},
        col_scaling::FSVec{Float64},
    )::Float64
    m, n = size(ws)
    if m == n == 1
        return inv(row_scaling[1] * fast_abs(ws.A[1, 1]) * col_scaling[1])
    elseif m > n
        ws.factorized || factorize!(ws)
        rmax = -Inf
        rmin = Inf
        @inbounds for i in 1:n
            ri = fast_abs(ws.qr.factors[i, i]) * col_scaling[i]
            rmax = max(rmax, ri)
            rmin = min(rmin, ri)
        end
        return rmax / rmin
    else
        ws.factorized || factorize!(ws)
        return _scaled_inf_norm_matrix(ws, row_scaling, col_scaling) *
            _inverse_inf_norm_est(
            ws.lu, row_scaling, col_scaling, ws.row_scaling, ws.scaled,
            ws.inf_norm_est_work, ws.inf_norm_est_rwork,
        )
    end
end

# Raw tolerance: used by is_finite in valuation.jl
@inline function _at_infinity_tol(
        val_x::Float64,
        val_tẋ::Float64,
        Δval_x::Float64,
        Δval_tẋ::Float64,
    )::Float64
    if abs(val_x) < 1.0e-30 || abs(val_tẋ) < 1.0e-30
        return Inf
    end
    ε∞ = max(
        abs(1.0 - val_tẋ / val_x),
        abs(Δval_x / val_x),
        abs(Δval_tẋ / val_tẋ),
    )
    return isfinite(ε∞) ? ε∞ : Inf
end

# Gated version for check_at_infinity!: only returns finite ε∞ when the valuation
# actually indicates divergence (val_x < 0 → ∞) or convergence to zero (val_x > 0 → 0).
# Without this gate, regular coordinates with small ε∞ get spuriously marked.
@inline function _at_infinity_tol_gated(
        val_x::Float64,
        val_tẋ::Float64,
        Δval_x::Float64,
        Δval_tẋ::Float64,
        finite_tol::Float64,
        zero_is_at_infinity::Bool,
    )::Float64
    ε∞ = _at_infinity_tol(val_x, val_tẋ, Δval_x, Δval_tẋ)
    if !isfinite(ε∞)
        return Inf
    end
    if val_x + ε∞ < -finite_tol
        return ε∞
    elseif zero_is_at_infinity && val_x - ε∞ > finite_tol
        return ε∞
    else
        return Inf
    end
end

@inline function _clear_at_infinity_candidate!(state::EndgameState, i::Int)::Nothing
    state.at_inf_active[i] = false
    state.at_inf_starts[i] = NaN
    state.at_inf_abs_coords[i] = NaN
    state.at_inf_conds[i] = NaN
    return nothing
end

function _ensure_endgame_scaling!(state::EndgameState, tracker::Tracker)::Nothing
    if all(iszero, state.col_scaling)
        @inbounds for i in eachindex(state.col_scaling)
            state.col_scaling[i] = tracker.state.norm.weights[i]
        end
        skeel_row_scaling!(
            state.row_scaling,
            tracker.state.jacobian.workspace.A,
            state.col_scaling,
        )
    end
    return nothing
end

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
        else
            state.code = EndgameCode.TERMINATED_MAX_STEPS
        end
        return nothing
    end

    if state.in_endgame
        eg_ext_steps = ext_steps(tracker.state) - state.ext_steps_eg_start
        if eg_ext_steps >= opts.max_endgame_extended_steps
            if !isnan(state.accuracy) && state.accuracy < opts.singular_min_accuracy
                state.code = EndgameCode.SUCCESS
            else
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
    _ensure_endgame_scaling!(state, tracker)

    add_sample!(eg, 0)
    tracker.predictor.winding_number = state.winding_number
    # Condition baseline
    state.at_inf_conds[1] = state.sample_conds[1]
    # Latch extended precision
    tracker.state.keep_extended_prec = true
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

function check_at_infinity!(eg::EndgameTracker)::Bool
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
        ε∞ = _at_infinity_tol_gated(vx, vtx, dvx, dvtx, opts.val_finite_tol, opts.zero_is_at_infinity)

        if !state.at_inf_active[i]
            # Stage 1: mark candidates
            if ε∞ < opts.val_at_infinity_tol
                if all(!, state.at_inf_active)
                    _ensure_endgame_scaling!(state, tracker)
                end
                if isnan(κ)
                    κ = _scaled_cond(
                        tracker.state.jacobian.workspace,
                        state.row_scaling,
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
                _clear_at_infinity_candidate!(state, i)
                continue
            end

            _ensure_endgame_scaling!(state, tracker)
            if isnan(κ)
                κ = _scaled_cond(
                    tracker.state.jacobian.workspace,
                    state.row_scaling,
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
            min_growth = clamp(0.25^(4.0 * abs(v)), 20.0, opts.min_coord_growth)

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
                _clear_at_infinity_candidate!(state, i)
            end
        end
    end

    return false
end

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
    state.sample_conds[slot] = _scaled_cond(
        tracker.state.jacobian.workspace,
        state.row_scaling,
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

function _predict_and_finalize!(eg::EndgameTracker, max_steps::Bool)::Nothing
    state = eg.state
    tracker = eg.tracker
    val = eg.val
    opts = eg.options
    m = state.winding_number
    n = length(state.solution)

    κ_sample = state.sample_conds[min(state.singular_steps + 1, 3)]

    # Zero-clamp coordinates with small valuation
    zero_cond = 1.0 / (m + 1)
    @inbounds for i in 1:n
        state.solution[i] = val.val_x[i] < zero_cond ? state.prediction[i] : zero(ComplexF64)
    end

    # Compute condition number at t=0
    ws = tracker.state.jacobian.workspace
    evaluate_and_jacobian!(
        tracker.corrector.r, ws.A, tracker.homotopy,
        state.solution, complex(0.0),
    )
    updated!(ws)
    κ_0 = _scaled_cond(ws, state.row_scaling, state.col_scaling)
    J0_norm = _row_scaled_inf_norm_matrix(ws, state.row_scaling)

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
                _predict_and_finalize!(eg, true)
            else
                state.code = EndgameCode.TERMINATED_MAX_STEPS
            end
            return nothing
        end
    end

    # Inner tracker failed to reach t_new — attempt finalization with existing
    # samples if we have enough, otherwise give up.
    if tracker.state.code != TrackerCode.TRACKER_SUCCESS
        if state.singular_steps >= 2
            _predict_and_finalize!(eg, true)
        else
            state.accuracy = tracker.state.accuracy
            copyto!(state.solution, tracker.state.x)
            state.cond = tracker.state.cond_J_ẋ
            state.code = _tracker_code_to_endgame_code(tracker.state.code)
        end
        return nothing
    end

    # Update valuation at new point
    update!(eg.val, tracker.predictor, t_new)

    # Check winding number consistency
    n = length(state.solution)
    m̂, m̂_err = estimate_winding_number(eg.val, n, opts.max_winding_number)
    if m̂_err > 0.1 || m̂ != state.winding_number
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
        # happens in _predict_and_finalize!, not here.
        state.accuracy = acc
        copyto!(state.solution, state.prediction)
        return nothing
    end

    if acc < state.accuracy && state.accuracy > 1.0e-12
        # Accuracy improved — update solution and continue
        state.accuracy = acc
        copyto!(state.solution, state.prediction)
    else
        _predict_and_finalize!(eg, false)
    end

    return nothing
end
