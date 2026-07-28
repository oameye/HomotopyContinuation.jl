# Tracker — main path tracker for homotopy continuation.
# Combines predictor-corrector steps with adaptive step size control.

@enumx TrackerCode::Int8 begin
    TRACKING
    TRACKER_SUCCESS
    TERMINATED_MAX_STEPS
    TERMINATED_ACCURACY_LIMIT
    TERMINATED_ILL_CONDITIONED
    TERMINATED_INVALID_STARTVALUE
    TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN
    TERMINATED_STEP_SIZE_TOO_SMALL
end

@kwdef struct TrackerOptions
    max_steps::Int = 10_000
    max_step_size::Float64 = Inf
    max_initial_step_size::Float64 = Inf
    extended_precision::Bool = true
    min_step_size::Float64 = 1.0e-48
    terminate_cond::Float64 = 1.0e13
    a::Float64 = 0.125
    β_a::Float64 = 1.0
    β_ω::Float64 = 3.0
    β_τ::Float64 = 0.4
    strict_β_τ::Float64 = 0.3
end

# Preset option sets; strict_β_τ = min(0.75 β_τ, 0.4).

"""
    DEFAULT_TRACKER_OPTIONS

[`TrackerOptions`](@ref) with a good balance between robustness and efficiency
(the same as `TrackerOptions()`).
"""
const DEFAULT_TRACKER_OPTIONS = TrackerOptions()

"""
    FAST_TRACKER_OPTIONS

[`TrackerOptions`](@ref) which trade speed against a higher chance of path
jumping.
"""
const FAST_TRACKER_OPTIONS = TrackerOptions(; β_ω = 2.0, β_τ = 0.75, strict_β_τ = 0.4)

"""
    CONSERVATIVE_TRACKER_OPTIONS

[`TrackerOptions`](@ref) which trade robustness against some speed.
"""
const CONSERVATIVE_TRACKER_OPTIONS =
    TrackerOptions(; β_ω = 4.0, β_τ = 0.25, strict_β_τ = 0.1875)

# Precomputed constants derived from TrackerOptions.a
struct TrackerConstants
    tol_acc::Float64   # a³ * h(a) — accuracy termination threshold
    h_a_step::Float64  # √(1 + 2*h(β_a*a)) - 1 — used in step size formula
end

function TrackerConstants(opts::TrackerOptions)
    a = opts.a
    h_a = 2a * (sqrt(4a^2 + 1) - 2a)
    step_a = opts.β_a * a
    h_step = 2step_a * (sqrt(4step_a^2 + 1) - 2step_a)
    return TrackerConstants(a^3 * h_a, sqrt(1 + 2 * h_step) - 1)
end

"""
    TrackerState

Mutable state for the path tracker.
"""
mutable struct TrackerState
    const x::FSVec{ComplexF64}
    const x̂::FSVec{ComplexF64}
    const x̄::FSVec{ComplexF64}
    segment::SegmentStepper
    Δs_prev::Float64
    accuracy::Float64
    ω::Float64
    ω_prev::Float64
    μ::Float64
    τ::Float64
    norm_Δx₀::Float64
    extended_prec::Bool
    used_extended_prec::Bool
    keep_extended_prec::Bool
    const norm::WeightedNorm
    use_strict_β_τ::Bool
    const jacobian::Jacobian
    cond_J_ẋ::Float64
    code::TrackerCode.T
    accepted_steps::Int
    rejected_steps::Int
    ext_accepted_steps::Int
    ext_rejected_steps::Int
    last_steps_failed::Int
end

function TrackerState(m::Int, n::Int, segment::SegmentStepper)
    return TrackerState(
        FSVec{ComplexF64}(zeros(ComplexF64, n)),     # x
        FSVec{ComplexF64}(zeros(ComplexF64, n)),     # x̂
        FSVec{ComplexF64}(zeros(ComplexF64, n)),     # x̄
        segment,
        0.0, eps(), 1.0, 1.0, eps(), Inf, NaN,      # Δs_prev..norm_Δx₀
        false, false, false,                          # extended_prec flags
        WeightedNorm(n),
        false,                                        # use_strict_β_τ
        Jacobian(MatrixWorkspace(m, n)),
        NaN,                                          # cond_J_ẋ
        TrackerCode.TRACKING,
        0, 0, 0, 0, 0,                                # counters: accepted, rejected, ext_accepted, ext_rejected, last_failed
    )
end

struct Tracker
    homotopy::HomotopyEvaluator
    predictor::Predictor
    corrector::NewtonCorrector
    state::TrackerState
    options::TrackerOptions
    constants::TrackerConstants
end

function Tracker(
        H::HomotopyEvaluator;
        start::ComplexF64 = complex(1.0),
        target::ComplexF64 = complex(0.0),
        options::TrackerOptions = TrackerOptions(),
    )
    m, n = size(H)
    segment = SegmentStepper(start, target)
    return Tracker(
        H,
        Predictor(m, n),
        NewtonCorrector(options.a, n, m),
        TrackerState(m, n, segment),
        options,
        TrackerConstants(options),
    )
end

# ---------------------------------------------------------------------------
# Convergence helper
# ---------------------------------------------------------------------------

@inline function _h(a::Float64)::Float64
    return 2a * (sqrt(4a^2 + 1) - 2a)
end

# ---------------------------------------------------------------------------
# Step size control
# ---------------------------------------------------------------------------

function _compute_initial_stepsize(
        state::TrackerState, pred::Predictor, opts::TrackerOptions,
        consts::TrackerConstants,
    )::Float64
    p = pred.order
    ω = state.ω
    e = pred.local_error
    τ = pred.trust_region

    # Fallback for infinite/NaN local error — use a conservative estimate
    if !isfinite(e) || e <= 0
        e = 1.0e5
    end
    Δs₁ = isfinite(ω) ? nthroot(consts.h_a_step / (ω * e), p) / opts.β_ω : Inf
    Δs₂ = opts.β_τ * τ

    Δs = nanmin(Δs₁, Δs₂)
    Δs = min(Δs, opts.max_step_size, opts.max_initial_step_size)
    return max(Δs, opts.min_step_size)
end

function _update_stepsize!(
        state::TrackerState,
        result::NewtonCorrectorResult,
        pred::Predictor,
        opts::TrackerOptions,
        consts::TrackerConstants,
    )::Nothing
    p = pred.order

    if result.return_code == NewtonCode.NEWT_CONVERGED
        # ω extrapolation: predict ω trend to take larger steps
        ω = clamp(state.ω + 2 * (state.ω - state.ω_prev), state.ω, 8 * state.ω)
        e = pred.local_error
        τ = state.τ

        if isfinite(e) && e > 0 && isfinite(ω)
            Δs₁ = nthroot(consts.h_a_step / (ω * e), p) / opts.β_ω
        else
            Δs₁ = Inf
        end

        β_τ = if state.use_strict_β_τ || dist_to_target(state.segment) < opts.β_τ * τ
            opts.strict_β_τ
        else
            opts.β_τ
        end
        Δs₂ = β_τ * τ

        Δs = min(nanmin(Δs₁, Δs₂), opts.max_step_size)

        # Near-target refinement: tighten step size when close to endpoint
        if state.use_strict_β_τ && dist_to_target(state.segment) < Δs
            Δs *= opts.strict_β_τ
        end

        # Limit step increase rate
        if state.Δs_prev > 0
            Δs = min(Δs, 10 * state.Δs_prev)
        end

        if state.last_steps_failed > 0
            Δs = min(Δs, state.Δs_prev)
        end
    else
        # Convergence-rate-based rejection: use Newton convergence rate θ to
        # estimate how much to reduce step size
        j = result.iters - 2
        Θ_j = j > 0 ? nthroot(result.θ, 1 << j) : result.θ
        h_Θ_j = _h(Θ_j)
        h_half_a = _h(0.5 * opts.β_a * opts.a)

        if isnan(Θ_j) ||
                result.return_code == NewtonCode.NEWT_SINGULARITY ||
                isnan(result.accuracy) ||
                result.iters <= 1 ||
                h_Θ_j < h_half_a
            # Fallback: fixed 0.25x reduction
            Δs = 0.25 * abs(state.segment.Δs)
        else
            # Proportional reduction based on convergence rate
            Δs = nthroot(
                (sqrt(1 + 2 * h_half_a) - 1) / (sqrt(1 + 2 * h_Θ_j) - 1), p,
            ) * abs(state.segment.Δs)
        end
    end

    propose_step!(state.segment, max(Δs, opts.min_step_size))
    return nothing
end

# ---------------------------------------------------------------------------
# Termination check
# ---------------------------------------------------------------------------

function _check_terminated!(
        state::TrackerState, opts::TrackerOptions, consts::TrackerConstants,
    )::Nothing
    if is_done(state.segment)
        state.code = TrackerCode.TRACKER_SUCCESS
    elseif state.accepted_steps + state.rejected_steps >= opts.max_steps
        state.code = TrackerCode.TERMINATED_MAX_STEPS
    else
        # Only terminate on accuracy if in extended precision or extended precision disabled.
        tol_acc = if state.extended_prec || !opts.extended_precision
            consts.tol_acc
        else
            Inf  # Can still switch to extended precision
        end
        if state.ω * state.μ > tol_acc
            state.code = TrackerCode.TERMINATED_ACCURACY_LIMIT
        elseif abs(state.segment.Δs) < opts.min_step_size ||
                fast_abs(state.segment.t′ - state.segment.t) <=
                2eps(fast_abs(state.segment.t))
            state.code = TrackerCode.TERMINATED_STEP_SIZE_TOO_SMALL
            # NOTE: the ill-conditioned termination below is intentionally
            # disabled. It would kill paths passing briefly through an
            # ill-conditioned region (e.g. monodromy loops between two nearby
            # positive-dimensional components), which should keep tracking under
            # step-size/accuracy control. Enabling it made `decompose` lose paths.
            # elseif state.last_steps_failed >= 3 && state.cond_J_ẋ > opts.terminate_cond
            #     state.code = TrackerCode.TERMINATED_ILL_CONDITIONED
        end
    end
    return nothing
end

function use_extended_precision!(tracker::Tracker)::Float64
    state = tracker.state
    opts = tracker.options
    opts.extended_precision || return state.μ
    state.extended_prec && return state.μ

    state.extended_prec = true
    state.used_extended_prec = true

    μ = state.μ
    for _ in 1:2
        μ = extended_prec_refinement_step!(
            state.x, tracker.corrector, tracker.homotopy, state.x,
            state.segment.t, state.jacobian, state.norm;
            simple_newton_step = false,
        )
    end
    state.μ = max(μ, eps())
    return state.μ
end

function update_precision!(tracker::Tracker, μ_low::Float64)::Bool
    state = tracker.state
    opts = tracker.options
    opts.extended_precision || return false

    a = opts.a
    if state.extended_prec && !state.keep_extended_prec && isfinite(μ_low) && μ_low > state.μ
        if μ_low * state.ω < a^7 * _h(a)
            state.extended_prec = false
            state.μ = μ_low
        end
    elseif state.μ * state.ω > a^5 * _h(a)
        use_extended_precision!(tracker)
    end

    return state.extended_prec
end

function refine_current_solution!(
        tracker::Tracker;
        min_tol::Float64 = 4 * eps(),
        nsteps::Int = 3,
    )::Float64
    state = tracker.state
    state.used_extended_prec = true

    μ = state.accuracy
    μ̄ = extended_prec_refinement_step!(
        state.x̄, tracker.corrector, tracker.homotopy, state.x,
        state.segment.t, state.jacobian, state.norm;
        simple_newton_step = false,
    )
    if μ̄ < μ
        copyto!(state.x, state.x̄)
        μ = μ̄
    end

    k = 1
    while μ > min_tol && k <= nsteps
        μ̄ = extended_prec_refinement_step!(
            state.x̄, tracker.corrector, tracker.homotopy, state.x,
            state.segment.t, state.jacobian, state.norm,
        )
        if μ̄ < μ
            copyto!(state.x, state.x̄)
            μ = μ̄
        end
        k += 1
    end

    return μ
end

# ---------------------------------------------------------------------------
# step! — single predictor-corrector step
# ---------------------------------------------------------------------------

"""
    step!(tracker::Tracker) -> Bool

Perform a single predictor-corrector step. Returns `true` if the step was
accepted (Newton corrector converged), `false` if rejected.
"""
function step!(tracker::Tracker)::Bool
    state = tracker.state
    pred = tracker.predictor
    H = tracker.homotopy
    opts = tracker.options
    consts = tracker.constants

    # Predict
    predict!(state.x̂, pred, state.segment.Δt)
    update!(state.norm, state.x̂)

    # Newton correct
    result = newton!(
        state.x̄, tracker.corrector, H, state.x̂, state.segment.t′,
        state.jacobian, state.norm,
        state.ω, state.μ, state.accepted_steps == 0, state.extended_prec,
    )

    accepted = result.return_code == NewtonCode.NEWT_CONVERGED

    if accepted
        state.Δs_prev = abs(state.segment.Δs)
        step_success!(state.segment)

        copyto!(state.x, state.x̄)
        state.accuracy = result.accuracy
        state.μ = max(result.accuracy, eps())
        state.ω_prev = state.ω
        state.ω = max(result.ω, 0.5 * state.ω, 0.1)
        update_precision!(tracker, result.μ_low)
        state.cond_J_ẋ = pred.cond_H_x

        if is_done(state.segment) && opts.extended_precision && state.accuracy > 1.0e-14
            state.accuracy = refine_current_solution!(tracker; min_tol = 1.0e-14)
            state.μ = max(state.accuracy, eps())
        end

        compute_local_error!(pred, state.x̂, state.x, state.norm, state.Δs_prev)

        update!(pred, H, state.x, state.segment.t, state.jacobian, state.norm)
        state.τ = pred.trust_region

        state.accepted_steps += 1
        state.ext_accepted_steps += Int(state.extended_prec)
        state.last_steps_failed = 0
    else
        state.rejected_steps += 1
        state.ext_rejected_steps += Int(state.extended_prec)
        state.last_steps_failed += 1
        # `iters > 1`: the correction reached the prediction and stopped contracting,
        # so it is precision-limited and a smaller step will not help. Escalating
        # refines `state.x` by about `μ`, leaving the predictor that stale until the
        # next accepted step.
        if state.last_steps_failed >= 3 && !state.extended_prec &&
                result.return_code == NewtonCode.NEWT_TERMINATED && result.iters > 1
            use_extended_precision!(tracker)
        end
    end

    state.norm_Δx₀ = result.norm_Δx₀
    _update_stepsize!(state, result, pred, opts, consts)
    _check_terminated!(state, opts, consts)

    return accepted
end

# ---------------------------------------------------------------------------
# init! — initialize tracker for a new path
# ---------------------------------------------------------------------------

"""
    _start_jacobian_corank(A) -> Int

Corank of the start-point Jacobian, used to distinguish a rank-deficient start
Jacobian from a generic invalid start value. Returns 0 (generic) when the
matrix contains non-finite entries and cannot be classified.
"""
function _start_jacobian_corank(A::FSMat{ComplexF64})::Int
    M = Matrix{ComplexF64}(A)
    all(isfinite, M) || return 0
    return size(M, 2) - LA.rank(M; rtol = 1.0e-14)
end

"""
    init!(tracker::Tracker, x₀, t₁, t₀) -> TrackerCode.T

Initialize the tracker for tracking a path from `t₁` to `t₀` starting at `x₀`.
Returns `TrackerCode.TRACKING` on success, or an error code if the start value
is invalid.

Warm-start kwargs: `ω` and `μ` seed the Newton certificates from a previous
track of the same point (skipping the initial Newton correction when both are
given), and `extended_precision = true` starts in extended precision.
"""
function init!(
        tracker::Tracker,
        x₀::AbstractVector{<:Number},
        t₁::ComplexF64 = complex(1.0),
        t₀::ComplexF64 = complex(0.0);
        ω::Float64 = NaN,
        μ::Float64 = NaN,
        extended_precision::Bool = false,
        max_initial_step_size::Float64 = Inf,
        keep_steps::Bool = false,
    )::TrackerCode.T
    state = tracker.state
    pred = tracker.predictor
    opts = tracker.options

    reinit!(state.segment, t₁, t₀)
    copyto!(state.x, x₀)
    state.Δs_prev = 0.0
    state.accuracy = NaN
    state.ω = 1.0
    state.ω_prev = 1.0
    state.μ = eps()
    state.τ = Inf
    state.norm_Δx₀ = NaN
    state.extended_prec = extended_precision
    state.used_extended_prec = extended_precision
    state.keep_extended_prec = false
    state.use_strict_β_τ = false
    state.cond_J_ẋ = NaN
    state.code = TrackerCode.TRACKING
    if !keep_steps
        state.accepted_steps = 0
        state.rejected_steps = 0
        state.ext_accepted_steps = 0
        state.ext_rejected_steps = 0
    end
    state.last_steps_failed = 0

    pred.t = complex(NaN)
    pred.prev_t = complex(NaN)
    pred.winding_number = 1
    pred.local_error = NaN

    init!(state.norm, state.x)
    init!(state.jacobian)

    # Compute any missing initialization data via Newton.
    # This lets callers preserve a toric-phase μ while still deriving ω at the handoff.
    if isnan(ω) || isnan(μ)
        valid, ω_init, μ_init = init_newton!(
            state.x̄, tracker.corrector, tracker.homotopy, state.x, t₁,
            state.jacobian, state.norm, false,
        )

        if !valid && opts.extended_precision
            valid, ω_init, μ_init = init_newton!(
                state.x̄, tracker.corrector, tracker.homotopy, state.x, t₁,
                state.jacobian, state.norm, true,
            )
            state.extended_prec = valid
            state.used_extended_prec = valid
        end

        if !valid
            # Classify the failure: re-evaluate the Jacobian at the original
            # start point and check its rank. Cold path, so the allocating
            # rank computation is fine.
            evaluate_and_jacobian!(
                tracker.corrector.r, state.jacobian.workspace.A,
                tracker.homotopy, state.x, t₁,
            )
            corank = _start_jacobian_corank(state.jacobian.workspace.A)
            state.code = if corank > 0
                TrackerCode.TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN
            else
                TrackerCode.TERMINATED_INVALID_STARTVALUE
            end
            return state.code
        end

        copyto!(state.x, state.x̄)
        state.ω = isnan(ω) ? ω_init : ω
        state.ω_prev = state.ω
        state.μ = max(isnan(μ) ? μ_init : μ, eps())
        state.accuracy = state.μ
    else
        # ω and μ provided — trust caller, skip Newton correction
        state.ω = ω
        state.ω_prev = ω
        state.μ = max(μ, eps())
        state.accuracy = max(μ, eps())
    end

    # Initialize predictor (evaluate Jacobian + Taylor coefficients)
    evaluate_and_jacobian!(
        tracker.corrector.r, state.jacobian.workspace.A,
        tracker.homotopy, state.x, t₁,
    )
    updated!(state.jacobian)
    update!(pred, tracker.homotopy, state.x, t₁, state.jacobian, state.norm)
    state.τ = pred.trust_region
    state.cond_J_ẋ = pred.cond_H_x

    Δs = _compute_initial_stepsize(state, pred, opts, tracker.constants)
    Δs = min(Δs, max_initial_step_size)
    propose_step!(state.segment, Δs)

    return state.code
end

ext_steps(state::TrackerState)::Int = state.ext_accepted_steps + state.ext_rejected_steps

"""
    resume_from!(tracker, t₀) -> TrackerCode.T

Lightweight reinit: reset segment from current position to `t₀`, recompute
initial step size, but preserve x, accuracy, ω, norm, Jacobian, and step
counters. Used by endgame fallback to continue tracking without losing state.
"""
function resume_from!(tracker::Tracker, t₀::ComplexF64)::TrackerCode.T
    state = tracker.state
    pred = tracker.predictor
    opts = tracker.options

    t_current = state.segment.t
    reinit!(state.segment, t_current, t₀)
    state.code = TrackerCode.TRACKING
    state.Δs_prev = 0.0

    Δs = _compute_initial_stepsize(state, pred, opts, tracker.constants)
    propose_step!(state.segment, Δs)

    return state.code
end

# ---------------------------------------------------------------------------
# track! — track a full path
# ---------------------------------------------------------------------------

"""
    track!(tracker::Tracker, x₀; t₁, t₀) -> TrackerCode.T

Track a path from `t₁` to `t₀` starting at `x₀`. Returns the final tracker code.
"""
function track!(
        tracker::Tracker,
        x₀::AbstractVector{ComplexF64};
        t₁::ComplexF64 = complex(1.0),
        t₀::ComplexF64 = complex(0.0),
        ω::Float64 = NaN,
        μ::Float64 = NaN,
        extended_precision::Bool = false,
    )::TrackerCode.T
    code = init!(
        tracker, x₀, t₁, t₀;
        ω = ω, μ = μ, extended_precision = extended_precision,
    )
    if code != TrackerCode.TRACKING
        return code
    end

    while tracker.state.code == TrackerCode.TRACKING
        step!(tracker)
    end

    return tracker.state.code
end
