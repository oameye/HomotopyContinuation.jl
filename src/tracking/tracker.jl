# Tracker — main path tracker for homotopy continuation.
# Combines predictor-corrector steps with adaptive step size control.

@enumx TrackerCode::Int8 begin
    TRACKING
    TRACKER_SUCCESS
    TERMINATED_MAX_STEPS
    TERMINATED_ACCURACY_LIMIT
    TERMINATED_ILL_CONDITIONED
    TERMINATED_INVALID_STARTVALUE
    TERMINATED_STEP_SIZE_TOO_SMALL
end

@kwdef struct TrackerOptions
    max_steps::Int = 10_000
    max_step_size::Float64 = Inf
    max_initial_step_size::Float64 = 0.1
    extended_precision::Bool = true
    min_step_size::Float64 = 0.0
    terminate_cond::Float64 = 1.0e14
    a::Float64 = 0.125
    β_ω::Float64 = 3.0
    β_τ::Float64 = 0.4
    strict_β_τ::Float64 = 0.3
end

# Precomputed constants derived from TrackerOptions.a
struct TrackerConstants
    h_a::Float64       # 2a(√(4a²+1) - 2a)
    tol_acc::Float64   # a³ * h_a — accuracy termination threshold
    h_a_step::Float64  # √(1 + 2*h_a) - 1 — used in step size formula
end

function TrackerConstants(opts::TrackerOptions)
    a = opts.a
    h_a = 2a * (sqrt(4a^2 + 1) - 2a)
    return TrackerConstants(h_a, a^3 * h_a, sqrt(1 + 2 * h_a) - 1)
end

"""
    TrackerState

Mutable state for the path tracker.

**Mutable justification:** Nearly all fields are updated every tracker step
(step size, accuracy, omega, counters, code, flags). Buffer fields (`x`, `x̂`,
`x̄`, `norm`, `jacobian`) are `const` because they are pre-allocated and never
reassigned. `segment` is reassigned in `init!` when a new path segment begins.
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
        0, 0, 0,                                      # counters
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

    if isfinite(e) && e > 0 && isfinite(ω)
        Δs₁ = nthroot(consts.h_a_step / (ω * e), p) / opts.β_ω
    else
        Δs₁ = Inf
    end
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
        ω = state.ω
        e = pred.local_error
        τ = pred.trust_region

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

        if state.Δs_prev > 0
            Δs = min(Δs, 10 * state.Δs_prev)
        end

        if state.last_steps_failed > 0
            Δs = min(Δs, state.Δs_prev)
        end
    else
        Δs = 0.25 * abs(state.segment.Δs)
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
        end
    end
    return nothing
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

    # Newton correct (positional args — no kwargs overhead)
    result = newton!(
        state.x̄, tracker.corrector, H, state.x̂, state.segment.t′,
        state.jacobian, state.norm,
        state.ω, state.μ, state.accepted_steps == 0,
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
        state.norm_Δx₀ = result.norm_Δx₀
        state.cond_J_ẋ = pred.cond_H_x

        compute_local_error!(pred, state.x̂, state.x, state.norm, state.Δs_prev)

        update!(pred, H, state.x, state.segment.t, state.jacobian, state.norm)
        state.τ = pred.trust_region

        state.accepted_steps += 1
        state.last_steps_failed = 0
    else
        state.rejected_steps += 1
        state.last_steps_failed += 1
    end

    _update_stepsize!(state, result, pred, opts, consts)
    _check_terminated!(state, opts, consts)

    return accepted
end

# ---------------------------------------------------------------------------
# init! — initialize tracker for a new path
# ---------------------------------------------------------------------------

"""
    init!(tracker::Tracker, x₀, t₁, t₀) -> TrackerCode.T

Initialize the tracker for tracking a path from `t₁` to `t₀` starting at `x₀`.
Returns `TrackerCode.TRACKING` on success, or an error code if the start value
is invalid.
"""
function init!(
        tracker::Tracker,
        x₀::AbstractVector{<:Number},
        t₁::ComplexF64 = complex(1.0),
        t₀::ComplexF64 = complex(0.0),
    )::TrackerCode.T
    state = tracker.state
    pred = tracker.predictor
    opts = tracker.options

    state.segment = SegmentStepper(t₁, t₀)
    copyto!(state.x, x₀)
    state.Δs_prev = 0.0
    state.accuracy = eps()
    state.ω = 1.0
    state.ω_prev = 1.0
    state.μ = eps()
    state.τ = Inf
    state.extended_prec = false
    state.used_extended_prec = false
    state.keep_extended_prec = false
    state.use_strict_β_τ = false
    state.code = TrackerCode.TRACKING
    state.accepted_steps = 0
    state.rejected_steps = 0
    state.last_steps_failed = 0

    pred.t = complex(NaN)
    pred.prev_t = complex(NaN)
    pred.winding_number = 1
    pred.local_error = NaN

    init!(state.norm, state.x)
    init!(state.jacobian)

    valid, ω, μ = init_newton!(
        state.x̄, tracker.corrector, tracker.homotopy, state.x, t₁,
        state.jacobian, state.norm,
    )

    if !valid
        state.code = TrackerCode.TERMINATED_INVALID_STARTVALUE
        return state.code
    end

    copyto!(state.x, state.x̄)
    state.ω = ω
    state.ω_prev = ω
    state.μ = μ

    update!(pred, tracker.homotopy, state.x, t₁, state.jacobian, state.norm)
    state.τ = pred.trust_region
    state.cond_J_ẋ = pred.cond_H_x

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
    )::TrackerCode.T
    code = init!(tracker, x₀, t₁, t₀)
    if code != TrackerCode.TRACKING
        return code
    end

    while tracker.state.code == TrackerCode.TRACKING
        step!(tracker)
    end

    return tracker.state.code
end
