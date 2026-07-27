## PathResult — per-path outcome from homotopy continuation tracking.

@enumx PathResultCode::Int8 begin
    PATH_SUCCESS
    PATH_AT_INFINITY
    PATH_AT_ZERO
    PATH_EXCESS_SOLUTION
    PATH_TERMINATED_ACCURACY
    PATH_TERMINATED_ILL_CONDITIONED
    PATH_TERMINATED_MAX_STEPS
    PATH_TERMINATED_STEP_SIZE
    PATH_TERMINATED_INVALID_START
    PATH_TERMINATED_INVALID_START_SINGULAR_JACOBIAN
end

struct PathResult
    return_code::PathResultCode.T
    solution::Vector{ComplexF64}
    t::Float64
    accuracy::Float64
    ω::Float64   # Newton contraction certificate at the endpoint (warm-start reuse)
    μ::Float64   # limit accuracy certificate at the endpoint (warm-start reuse)
    residual::Float64
    condition_jacobian::Float64
    winding_number::Int
    singular::Bool
    accepted_steps::Int
    rejected_steps::Int
    steps_eg::Int
    extended_precision_used::Bool
    last_path_point::Vector{ComplexF64}
    last_path_t::Float64
    path_number::Int
    start_solution::Vector{ComplexF64}
    valuation::Vector{Float64}
    multiplicity::Int
end

"""
    _add_steps(r, accepted, rejected)

Return a copy of `r` with additional accepted/rejected steps added (e.g. from a prior phase).
"""
function _add_steps(r::PathResult, accepted::Int, rejected::Int)::PathResult
    return PathResult(
        r.return_code, r.solution, r.t, r.accuracy, r.ω, r.μ, r.residual,
        r.condition_jacobian, r.winding_number, r.singular,
        r.accepted_steps + accepted, r.rejected_steps + rejected,
        r.steps_eg, r.extended_precision_used, r.last_path_point, r.last_path_t,
        r.path_number, r.start_solution, r.valuation, r.multiplicity,
    )
end

is_success(r::PathResult)::Bool = r.return_code == PathResultCode.PATH_SUCCESS
is_singular(r::PathResult)::Bool = is_success(r) && r.singular
is_nonsingular(r::PathResult)::Bool = is_success(r) && !r.singular

is_at_infinity(r::PathResult)::Bool =
    r.return_code == PathResultCode.PATH_AT_INFINITY ||
    r.return_code == PathResultCode.PATH_AT_ZERO
is_excess_solution(r::PathResult)::Bool = r.return_code == PathResultCode.PATH_EXCESS_SOLUTION

"""
    is_failed(r::PathResult)

`true` if the path neither succeeded, diverged to infinity/zero, nor was flagged
as an excess solution (i.e. tracking terminated for a numerical reason).
"""
is_failed(r::PathResult)::Bool =
    !(is_success(r) || is_at_infinity(r) || is_excess_solution(r))

"""
    is_finite(r::PathResult)

`true` if `r` is a finite solution; coincides with success.
"""
is_finite(r::PathResult)::Bool = is_success(r)
Base.isfinite(r::PathResult)::Bool = is_finite(r)

# ── Diagnostic accessors ────────────────────────────────────────────────────
# Per-path outcome inspection: the solution and its quality (accuracy, residual,
# conditioning) plus the tracking effort that produced it (step counts, winding).

"""
    solution(r::PathResult)

The solution vector stored in `r`.
"""
solution(r::PathResult)::Vector{ComplexF64} = r.solution

"""
    accuracy(r::PathResult)

Estimated accuracy of the solution (the final Newton update norm at the endpoint).
"""
accuracy(r::PathResult)::Float64 = r.accuracy

"""
    residual(r::PathResult)

Infinity norm `‖H(x, t)‖∞` of the homotopy at the reported endpoint.
"""
residual(r::PathResult)::Float64 = r.residual

"""
    accepted_steps(r::PathResult)

Number of accepted tracker steps along the path.
"""
accepted_steps(r::PathResult)::Int = r.accepted_steps

"""
    rejected_steps(r::PathResult)

Number of rejected tracker steps along the path.
"""
rejected_steps(r::PathResult)::Int = r.rejected_steps

"""
    steps(r::PathResult)

Total number of steps the path tracker performed (accepted + rejected).
Endgame steps are reported separately (see [`Base.show`](@ref)).
"""
steps(r::PathResult)::Int = accepted_steps(r) + rejected_steps(r)

"""
    winding_number(r::PathResult)

Estimated winding number of a singular endpoint (`0` when not estimated).
"""
winding_number(r::PathResult)::Int = r.winding_number

"""
    condition_jacobian(r::PathResult)

Estimated condition number of the Jacobian at the endpoint.
"""
condition_jacobian(r::PathResult)::Float64 = r.condition_jacobian

"""
    last_path_point(r::PathResult)

The tracker's last `(point, t)` on the path before the endgame extrapolation.
"""
last_path_point(r::PathResult)::Tuple{Vector{ComplexF64}, Float64} =
    (r.last_path_point, r.last_path_t)

"""
    cond(r::PathResult)

Estimated condition number of the Jacobian at the endpoint. Alias for
[`condition_jacobian`](@ref).
"""
LA.cond(r::PathResult)::Float64 = r.condition_jacobian

"""
    path_number(r::PathResult)

Index of the path (start solution) that produced `r`, or `0` if not recorded.
"""
path_number(r::PathResult)::Int = r.path_number

"""
    start_solution(r::PathResult)

The start solution the path was tracked from, or an empty vector if not recorded.
"""
start_solution(r::PathResult)::Vector{ComplexF64} = r.start_solution

"""
    valuation(r::PathResult)

The per-coordinate Puiseux valuation estimated during the endgame, or an empty
vector when the endgame did not sample a valuation for this path.
"""
valuation(r::PathResult)::Vector{Float64} = r.valuation

"""
    multiplicity(r::PathResult)

Multiplicity of the solution (the number of paths that converged to the same
point), filled in by the enclosing [`Result`](@ref). `0` for non-success paths or
an unclustered result.
"""
multiplicity(r::PathResult)::Int = r.multiplicity

# Positional / Base overloads for `is_real`.
is_real(r::PathResult, tol::Float64)::Bool = is_real(r; tol = tol)
Base.isreal(r::PathResult; tol::Float64 = DEFAULT_REAL_TOL)::Bool = is_real(r; tol = tol)
Base.isreal(r::PathResult, tol::Float64)::Bool = is_real(r; tol = tol)

function Base.show(io::IO, ::MIME"text/plain", r::PathResult)
    println(io, "PathResult:")
    println(io, " • return_code: ", r.return_code)
    println(io, " • solution: ", r.solution)
    println(io, " • accuracy: ", r.accuracy)
    println(io, " • residual: ", r.residual)
    println(io, " • condition_jacobian: ", r.condition_jacobian)
    r.winding_number > 0 && println(io, " • winding_number: ", r.winding_number)
    print(
        io, " • steps: ", steps(r), " (", r.accepted_steps, " accepted, ",
        r.rejected_steps, " rejected); ", r.steps_eg, " endgame",
    )
    return
end

"""
    _with_return_code(r, code)

Return a copy of `r` with the return code replaced (used to reclassify excess solutions).
"""
function _with_return_code(r::PathResult, code::PathResultCode.T)::PathResult
    return PathResult(
        code, r.solution, r.t, r.accuracy, r.ω, r.μ, r.residual,
        r.condition_jacobian,
        r.winding_number, r.singular, r.accepted_steps, r.rejected_steps,
        r.steps_eg, r.extended_precision_used, r.last_path_point, r.last_path_t,
        r.path_number, r.start_solution, r.valuation, r.multiplicity,
    )
end

"""
    _with_multiplicity(r, m)

Return a copy of `r` with its `multiplicity` field set to `m` (filled in by the
`Result` constructor once solutions have been clustered).
"""
function _with_multiplicity(r::PathResult, m::Int)::PathResult
    return PathResult(
        r.return_code, r.solution, r.t, r.accuracy, r.ω, r.μ, r.residual,
        r.condition_jacobian,
        r.winding_number, r.singular, r.accepted_steps, r.rejected_steps,
        r.steps_eg, r.extended_precision_used, r.last_path_point, r.last_path_t,
        r.path_number, r.start_solution, r.valuation, m,
    )
end

"""
    _to_ambient(r, H::IntrinsicSubspaceHomotopy)

Return a copy of `r` whose `solution` and `last_path_point` are converted from
the intrinsic coordinates of `H` to ambient coordinates. Each point is converted
at the `t` it was reported at, so failed paths (reported at their terminal `t`,
not at 0) convert on the subspace they actually stopped on. The diagnostics
(`accuracy`, `residual`, `condition_jacobian`) and `valuation` stay in intrinsic
coordinates; `start_solution` is already ambient.
"""
function _to_ambient(r::PathResult, H::IntrinsicSubspaceHomotopy)::PathResult
    n = length(H.x)
    solution = Vector{ComplexF64}(undef, n)
    ambient_coordinates!(solution, H, r.solution, complex(r.t))
    last_point = Vector{ComplexF64}(undef, n)
    ambient_coordinates!(last_point, H, r.last_path_point, complex(r.last_path_t))
    return PathResult(
        r.return_code, solution, r.t, r.accuracy, r.ω, r.μ, r.residual,
        r.condition_jacobian, r.winding_number, r.singular,
        r.accepted_steps, r.rejected_steps, r.steps_eg, r.extended_precision_used,
        last_point, r.last_path_t, r.path_number, r.start_solution,
        r.valuation, r.multiplicity,
    )
end

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
    elseif code == TrackerCode.TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN
        return PathResultCode.PATH_TERMINATED_INVALID_START_SINGULAR_JACOBIAN
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
    elseif code == EndgameCode.TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN
        return PathResultCode.PATH_TERMINATED_INVALID_START_SINGULAR_JACOBIAN
    elseif code == EndgameCode.TERMINATED_STEP_SIZE_TOO_SMALL
        return PathResultCode.PATH_TERMINATED_STEP_SIZE
    else
        return PathResultCode.PATH_TERMINATED_MAX_STEPS
    end
end

# Residual ‖H(x, t)‖∞ of the endpoint, evaluated in-place into `scratch`
# (reuses the corrector residual buffer — safe at end of path).
function _homotopy_residual!(
        scratch::FSVec{ComplexF64}, H::HomotopyEvaluator,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Float64
    evaluate!(scratch, H, x, t)
    return inf_norm(scratch)
end

# Tracker-only PathResult — used for polyhedral toric phase failures (no endgame needed)
function PathResult(
        tracker::Tracker;
        path_number::Int = 0,
        start_solution::Vector{ComplexF64} = ComplexF64[],
    )
    state = tracker.state
    t = real(state.segment.t)
    residual = _homotopy_residual!(
        tracker.corrector.r, tracker.homotopy, state.x, state.segment.t,
    )
    return PathResult(
        _tracker_code_to_path_code(state.code),
        Vector{ComplexF64}(state.x),
        t,
        state.accuracy,
        state.ω,
        state.μ,
        residual,
        state.cond_J_ẋ,
        0,
        false,
        state.accepted_steps,
        state.rejected_steps,
        0,
        state.used_extended_prec,
        Vector{ComplexF64}(state.x),
        t,
        path_number,
        start_solution,
        Float64[],   # no endgame valuation for a tracker-only result
        0,
    )
end

function PathResult(
        eg::EndgameTracker;
        path_number::Int = 0,
        start_solution::Vector{ComplexF64} = ComplexF64[],
    )
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

    # Residual of the reported endpoint: evaluate H at the extrapolated solution
    # (t = 0) for success, otherwise at the tracker's actual terminal point.
    residual = if success
        _homotopy_residual!(eg.tracker.corrector.r, eg.tracker.homotopy, state.solution, complex(0.0))
    else
        _homotopy_residual!(eg.tracker.corrector.r, eg.tracker.homotopy, ts.x, ts.segment.t)
    end

    # Per-coordinate Puiseux valuation, recorded only once the endgame has taken
    # enough samples to make `val_x` meaningful (empty otherwise).
    valuation = eg.val.samples > 0 ? Vector{Float64}(eg.val.val_x) : Float64[]

    return PathResult(
        _endgame_code_to_path_code(state.code),
        solution,
        t,
        accuracy,
        ts.ω,
        ts.μ,
        residual,
        state.cond,
        state.winding_number,
        state.singular,
        ts.accepted_steps,
        ts.rejected_steps,
        state.steps_eg,
        ts.used_extended_prec,
        Vector{ComplexF64}(ts.x),  # last_path_point: tracker's actual position
        real(ts.segment.t),        # last_path_t: tracker's actual t
        path_number,
        start_solution,
        valuation,
        0,
    )
end
