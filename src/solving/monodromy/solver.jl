######################
## Monodromy solver ##
######################

# Chart-normalizing wrapper implementing the 2-arg `actions(cb, s)` protocol
# for UniquePoints: every orbit image is renormalized onto the chart before
# comparison. Used when tracking on an affine chart (projective problems).
struct _ChartActions{GA <: GroupActions}
    chart::Vector{ComplexF64}
    actions::GA
end
function (CA::_ChartActions)(cb::CB, s) where {CB}
    return apply_actions(CA.actions, s) do v
        w = copy(v)
        on_chart!(w, CA.chart)
        cb(w)
    end
end

"""
    MonodromyWorkerState

Per-worker state for the monodromy solver: the concrete homotopy instance (the
EXACT instance captured in the tracker's evaluator closures, so retargeting it
retargets the tracker), the endgame tracker built around it, the base
parameters, and ambient/intrinsic scratch buffers.
"""
struct MonodromyWorkerState{H, P}
    homotopy::H
    tracker::EndgameTracker
    base::P                        # base parameters (Vector or LinearSubspace)
    x_buffer::Vector{ComplexF64}   # ambient scratch for subspace conversion
    u_buffer::Vector{ComplexF64}   # tracker-coordinate scratch
    # Chart vector for projective problems where the chart row lives INSIDE the
    # wrapped system (intrinsic subspace homotopy over an AffineChartSystem) and
    # is therefore not reachable through the homotopy handle. Empty otherwise.
    chart::Vector{ComplexF64}
end

MonodromyWorkerState(
    homotopy::H, tracker::EndgameTracker, base::P,
    x_buffer::Vector{ComplexF64}, u_buffer::Vector{ComplexF64},
) where {H, P} =
    MonodromyWorkerState{H, P}(homotopy, tracker, base, x_buffer, u_buffer, ComplexF64[])

## Retargeting through the concrete handle

function _retarget!(
        ws::MonodromyWorkerState{ParameterHomotopy}, p::Vector{ComplexF64},
        q::Vector{ComplexF64},
    )::Nothing
    parameters!(ws.homotopy, p, q)
    return nothing
end
function _retarget!(
        ws::MonodromyWorkerState{AffineChartHomotopy{ParameterHomotopy}},
        p::Vector{ComplexF64}, q::Vector{ComplexF64},
    )::Nothing
    parameters!(ws.homotopy, p, q)
    return nothing
end
function _retarget!(
        ws::MonodromyWorkerState, p::LinearSubspace{ComplexF64},
        q::LinearSubspace{ComplexF64},
    )::Nothing
    set_subspaces!(ws.homotopy, p, q)
    return nothing
end

parameters!(ws::MonodromyWorkerState{ParameterHomotopy}, p, q) = _retarget!(ws, p, q)

"""
    set_loop_segment!(ws, loop::MonodromyLoop, segment::Int)

Retarget the worker's homotopy to the given loop segment. Vector loops have 3
segments (`p → p₁ → p₂ → p`), subspace loops 4 (`p → p₀₁ → p₁ → p₂ → p`).
"""
function set_loop_segment!(
        ws::MonodromyWorkerState, loop::MonodromyLoop{Vector{ComplexF64}}, segment::Int,
    )::Nothing
    segment == 1 && return _retarget!(ws, loop.p, loop.p₁)
    segment == 2 && return _retarget!(ws, loop.p₁, loop.p₂)
    segment == 3 && return _retarget!(ws, loop.p₂, loop.p)
    throw(ArgumentError("Vector-parameter loops have 3 segments, got $segment."))
end
function set_loop_segment!(
        ws::MonodromyWorkerState, loop::MonodromyLoop{LinearSubspace{ComplexF64}},
        segment::Int,
    )::Nothing
    segment == 1 && return _retarget!(ws, loop.p, loop.p₀₁)
    segment == 2 && return _retarget!(ws, loop.p₀₁, loop.p₁)
    segment == 3 && return _retarget!(ws, loop.p₁, loop.p₂)
    segment == 4 && return _retarget!(ws, loop.p₂, loop.p)
    throw(ArgumentError("Subspace loops have 4 segments, got $segment."))
end

## Coordinate transfer between the solver's ambient state and the tracker

# Write the tracker-side start coordinates for the current segment (t = 1)
# from ws.x_buffer into ws.u_buffer and return it.
function _tracker_input!(ws::MonodromyWorkerState{IntrinsicSubspaceHomotopy})
    # Projective case: the wrapped system carries a chart row v'x = 1, so the
    # ambient representative must be rescaled onto the chart first (the
    # subspace is linear, so rescaling stays on it).
    isempty(ws.chart) || on_chart!(ws.x_buffer, ws.chart)
    intrinsic_coordinates!(ws.u_buffer, ws.homotopy, ws.x_buffer, complex(1.0))
    return ws.u_buffer
end
function _tracker_input!(ws::MonodromyWorkerState)
    copyto!(ws.u_buffer, ws.x_buffer)
    return ws.u_buffer
end
function _tracker_input!(ws::MonodromyWorkerState{<:AffineChartHomotopy})
    copyto!(ws.u_buffer, ws.x_buffer)
    # The chart row v'x = 1 must hold at the start point.
    on_chart!(ws.u_buffer, ws.homotopy)
    return ws.u_buffer
end

# Convert the tracker's endpoint (t = 0) back into ws.x_buffer (ambient).
function _tracker_output!(
        ws::MonodromyWorkerState{IntrinsicSubspaceHomotopy}, u::AbstractVector{ComplexF64},
    )::Nothing
    ambient_coordinates!(ws.x_buffer, ws.homotopy, u, complex(0.0))
    return nothing
end
function _tracker_output!(
        ws::MonodromyWorkerState, u::AbstractVector{ComplexF64},
    )::Nothing
    copyto!(ws.x_buffer, u)
    return nothing
end

# Renormalize a finished ambient solution for dedup (chart case only).
_normalize_solution!(::MonodromyWorkerState, x::AbstractVector{ComplexF64})::Nothing =
    nothing
function _normalize_solution!(
        ws::MonodromyWorkerState{<:AffineChartHomotopy}, x::AbstractVector{ComplexF64},
    )::Nothing
    on_chart!(x, ws.homotopy)
    return nothing
end

# Copy of `r` with its solution replaced by the ambient conversion.
function _with_solution(r::PathResult, sol::Vector{ComplexF64})::PathResult
    return PathResult(
        r.return_code, sol, r.t, r.accuracy, r.ω, r.μ, r.residual,
        r.condition_jacobian, r.winding_number, r.singular,
        r.accepted_steps, r.rejected_steps, r.steps_eg, r.extended_precision_used,
        r.last_path_point, r.last_path_t, r.path_number, r.start_solution,
        r.valuation, r.multiplicity,
    )
end

# Build the solver-facing PathResult from the endgame tracker: convert the
# endpoint to ambient coordinates and chart-normalize when required.
function _finalize_result(ws::MonodromyWorkerState, r::PathResult)::PathResult
    _tracker_output!(ws, r.solution)
    _normalize_solution!(ws, ws.x_buffer)
    return _with_solution(r, copy(ws.x_buffer))
end

# Track ws.x_buffer through the current (already retargeted) segment with the
# RAW inner tracker (no endgame), warm-started, updating ws.x_buffer in place.
function _track_middle_segment!(
        ws::MonodromyWorkerState, ω::Float64, μ::Float64, extended_precision::Bool,
    )::Bool
    tr = ws.tracker.tracker
    u = _tracker_input!(ws)
    code = track!(tr, u; ω = ω, μ = μ, extended_precision = extended_precision)
    code == TrackerCode.TRACKER_SUCCESS || return false
    _tracker_output!(ws, tr.state.x)
    return true
end

"""
    track_start!(ws, x)

Track `x` at the base parameters (trivial `p → p` retarget, full endgame) and
return the resulting [`PathResult`](@ref). The result carries the tracker's
return code, so a caller that wants only converged points filters on
[`is_success`](@ref). Used to validate and refine start solutions.
"""
function track_start!(
        ws::MonodromyWorkerState, x::AbstractVector{ComplexF64},
    )::PathResult
    _retarget!(ws, ws.base, ws.base)
    copyto!(ws.x_buffer, x)
    u = _tracker_input!(ws)
    track!(ws.tracker, u)
    return _finalize_result(ws, PathResult(ws.tracker))
end

"""
    MonodromySolver

Shared state of a monodromy computation: the worker states (index 1 is the
serial worker; threaded runs grow the vector lazily via `builder`), the loop
list, the deduplication structure with its lock, options, statistics and the
trace matrix for the trace test.
"""
mutable struct MonodromySolver{
        H, P, B, UP <: UniquePoints, MO <: MonodromyOptions,
        CS <: Union{Nothing, AbstractCertifiedSolutions},
    }
    # Mutable: `loops`, `statistics` and the two trace counters are reset
    # between solves; all other fields are const.
    const workers::Vector{MonodromyWorkerState{H, P}}
    const builder::B                                  # callable () -> MonodromyWorkerState{H, P}
    loops::Vector{MonodromyLoop{P}}
    const unique_points::UP
    const unique_points_lock::ReentrantLock
    # `nothing` under `DuplicateCheck.HEURISTIC`, else the certification
    # package's accumulator of certified-distinct solutions.
    const certified_solutions::CS
    const options::MO
    statistics::MonodromyStatistics
    # We save the sums of the solutions for three values of the loop parameter
    # to check whether the trace is colinear. The sums are augmented by 1 to
    # make this work for the one/two variable cases: an (n + 1) × 3 matrix.
    const trace::Matrix{ComplexF64}
    # Paths that reached the halfway subspace and summed into `trace`, and paths
    # that were asked to but lost a segment on the way. Both are written under
    # `trace_lock`. A nonzero `trace_dropped` makes the trace a statement about
    # the tracking rather than about the witness set.
    trace_paths::Int
    trace_dropped::Int
    const trace_lock::ReentrantLock
end

# The two duplicate-check policies as solver types: `HeuristicMonodromySolver`
# carries no accumulator, so its certified branch is statically dead. Together they
# cover the `CS` bound, which is what keeps the dedup path free of a half that has
# no method.
const HeuristicMonodromySolver = MonodromySolver{
    H, P, B, UP, MO, Nothing,
} where {H, P, B, UP <: UniquePoints, MO <: MonodromyOptions}
const CertifiedMonodromySolver = MonodromySolver{
    H, P, B, UP, MO, CS,
} where {
    H, P, B, UP <: UniquePoints, MO <: MonodromyOptions,
    CS <: AbstractCertifiedSolutions,
}

# Run `f` on the group actions the deduplication compares orbits with: none when
# the options ask for none or switch them off, and a chart-normalizing wrapper
# when the solutions are projective.
_with_chart_actions(
    f::Fn, ::MonodromyOptions{<:Any, Nothing}, ::Vector{ComplexF64}, ::Bool,
) where {Fn} = f(nothing)

function _with_chart_actions(
        f::Fn, options::MonodromyOptions, chart::Vector{ComplexF64}, use_chart::Bool,
    ) where {Fn}
    options.equivalence_classes || return f(nothing)
    actions = options.group_actions
    # The constructor clears `equivalence_classes` when there is no group action,
    # so this branch is unreachable; it is what tells inference so.
    actions === nothing && return f(nothing)
    return use_chart ? f(_ChartActions(chart, actions)) : f(actions)
end

function _with_monodromy_solver_from_builder(
        f::Fn, worker::MonodromyWorkerState{H, P}, builder::B, n::Int,
        options::MO, chart::Vector{ComplexF64}, use_chart::Bool,
        # Required, so a new route cannot silently downgrade to the heuristic check.
        certified_solutions::CS,
    ) where {
        Fn, H, P, B, MO <: MonodromyOptions,
        CS <: Union{Nothing, AbstractCertifiedSolutions},
    }
    return _with_chart_actions(options, chart, use_chart) do group_actions
        f(
            _monodromy_solver_from_builder(
                worker, builder, n, options, group_actions, certified_solutions,
            ),
        )
    end
end

function _monodromy_solver_from_builder(
        worker::MonodromyWorkerState{H, P}, builder::B, n::Int, options::MO,
        group_actions::GA, certified_solutions::CS,
    ) where {
        H, P, B, GA, MO <: MonodromyOptions,
        CS <: Union{Nothing, AbstractCertifiedSolutions},
    }
    unique_points = UniquePoints(
        n;
        distance = options.distance,
        group_actions = group_actions,
        triangle_inequality = options.triangle_inequality,
    )
    trace = zeros(ComplexF64, n + 1, 3)
    return MonodromySolver(
        [worker],
        builder,
        MonodromyLoop{P}[],
        unique_points,
        ReentrantLock(),
        certified_solutions,
        options,
        MonodromyStatistics(),
        trace,
        0,
        0,
        ReentrantLock(),
    )
end

function _parameter_monodromy_worker(
        sys_eval::SystemEvaluator, p::Vector{ComplexF64}, n::Int,
        tracker_options::TrackerOptions,
    )::MonodromyWorkerState{ParameterHomotopy, Vector{ComplexF64}}
    H = ParameterHomotopy(sys_eval, p, p)
    eg = endgame_tracker(
        H, tracker_options, EndgameOptions(; endgame_start = 0.0),
    )
    return MonodromyWorkerState(
        H, eg, copy(p), zeros(ComplexF64, n), zeros(ComplexF64, n),
    )
end

function _chart_parameter_monodromy_worker(
        sys_eval::SystemEvaluator, p::Vector{ComplexF64},
        chart::Vector{ComplexF64}, n::Int, tracker_options::TrackerOptions,
    )::MonodromyWorkerState{
        AffineChartHomotopy{ParameterHomotopy}, Vector{ComplexF64},
    }
    H = AffineChartHomotopy(ParameterHomotopy(sys_eval, p, p), chart)
    eg = endgame_tracker(
        H, tracker_options, EndgameOptions(; endgame_start = 0.0),
    )
    return MonodromyWorkerState(
        H, eg, copy(p), zeros(ComplexF64, n), zeros(ComplexF64, n),
    )
end

struct ParameterMonodromyBuilder{S <: SystemLike} <: AbstractPathBuilder
    system::S
    parameters::Vector{ComplexF64}
    nvariables::Int
    tracker_options::TrackerOptions
end

function (builder::ParameterMonodromyBuilder)()
    return _parameter_monodromy_worker(
        _clone_system_evaluator(builder.system), builder.parameters,
        builder.nvariables, builder.tracker_options,
    )
end

struct ChartParameterMonodromyBuilder{S <: SystemLike} <: AbstractPathBuilder
    system::S
    parameters::Vector{ComplexF64}
    chart::Vector{ComplexF64}
    nvariables::Int
    tracker_options::TrackerOptions
end

function (builder::ChartParameterMonodromyBuilder)()
    return _chart_parameter_monodromy_worker(
        _clone_system_evaluator(builder.system), builder.parameters,
        builder.chart, builder.nvariables, builder.tracker_options,
    )
end

# Smallest normalized alignment |v'x| / (‖v‖‖x‖) over the start solutions.
function _chart_alignment(
        v::Vector{ComplexF64}, S::AbstractVector{<:AbstractVector},
    )::Float64
    q = Inf
    nv = LA.norm(v)
    for x in S
        length(x) == length(v) || continue
        nx = LA.norm(x)
        iszero(nx) && continue
        λ = zero(ComplexF64)
        for i in eachindex(v)
            λ += v[i] * x[i]
        end
        q = min(q, abs(λ) / (nv * nx))
    end
    return q
end

# `on_chart!` divides by v'x, so a chart normal near-orthogonal to a start
# solution inflates that representative by 1/|v'x| and costs the tracker the
# digits the trace test needs. Keep the best of a few draws.
function _conditioned_chart(
        rng::Random.AbstractRNG, n::Int, S::AbstractVector{<:AbstractVector},
    )::Vector{ComplexF64}
    chart = randn(rng, ComplexF64, n)
    q = _chart_alignment(chart, S)
    for _ in 2:16
        q >= 0.2 && break
        v = randn(rng, ComplexF64, n)
        qv = _chart_alignment(v, S)
        if qv > q
            chart, q = v, qv
        end
    end
    return chart
end

@noinline _certification_needs_equations(F) = throw(
    ArgumentError(
        "`duplicate_check = DuplicateCheck.CERTIFIED` certifies the equations of a " *
            "`System`, which a $(typeof(F)) does not carry.",
    ),
)

# Runs `fn` on the square system the certified duplicate check certifies against, in
# the coordinates the run reports: a homogeneous system is charted, a subspace
# intersection is sliced. Slicing rebuilds over `ComplexF64` while an unsliced system
# keeps its own coefficient type.
with_certified_system(fn::F, G::System, chart::Vector{ComplexF64}) where {F} =
    isempty(chart) ? fn(G) :
    fn(slice(G, _full_subspace(nvariables(G)); chart = chart))
with_certified_system(
    fn::F, G::System, L::LinearSubspace{ComplexF64}, chart::Vector{ComplexF64},
) where {F} = fn(_rebuild_sliced(G, L, chart))
with_certified_system(::F, G::SystemLike, ::Vector{ComplexF64}) where {F} =
    _certification_needs_equations(G)
with_certified_system(
    ::F, G::SystemLike, ::LinearSubspace{ComplexF64}, ::Vector{ComplexF64},
) where {F} = _certification_needs_equations(G)

# Run `f` on the accumulator the certified duplicate check files into, or on
# `nothing` under `DuplicateCheck.HEURISTIC`. A parameter run passes the
# parameters it certifies at, a witness-set run the subspace it intersects with;
# `chart` is empty unless the solutions are projective.
function _with_certified_accumulator(
        f::Fn, F::SystemLike, options::MonodromyOptions, p::Vector{ComplexF64},
        chart::Vector{ComplexF64},
    ) where {Fn}
    options.duplicate_check == DuplicateCheck.CERTIFIED || return f(nothing)
    return with_certified_system(F, chart) do G
        f(
            monodromy_certified_solutions(
                G, p, options.certification_max_precision,
                options.certification_refine_solution,
            ),
        )
    end
end

function _with_certified_accumulator(
        f::Fn, F::SystemLike, options::MonodromyOptions,
        L::LinearSubspace{ComplexF64}, chart::Vector{ComplexF64},
    ) where {Fn}
    options.duplicate_check == DuplicateCheck.CERTIFIED || return f(nothing)
    # `monodromy_certified_solutions` is the certification package's entry point
    # and takes `nothing` for a parameter-free system.
    return with_certified_system(F, L, chart) do G
        f(
            monodromy_certified_solutions(
                G, nothing,
                options.certification_max_precision,
                options.certification_refine_solution,
            ),
        )
    end
end

"""
    with_monodromy_solver(f, F, p; options, tracker_options, rng, start_solutions)
    with_monodromy_solver(f, F, L; options, tracker_options, intrinsic, rng,
                          start_solutions)

Run `f` on the monodromy solver for `F` at the parameters `p`, or for the
parameter-free `F` intersected with the linear subspace `L`.

The solver is handed to `f` rather than returned: its homotopy, builder,
deduplication structure and duplicate-check accumulator are all chosen from the
system and the options, so each branch builds a differently parameterized solver.
"""
function with_monodromy_solver(
        f::Fn, F::SystemLike, p::Vector{ComplexF64};
        options::MonodromyOptions = MonodromyOptions(),
        tracker_options::TrackerOptions = TrackerOptions(),
        rng::Random.AbstractRNG = Random.default_rng(),
        start_solutions::AbstractVector{<:AbstractVector} = Vector{ComplexF64}[],
    ) where {Fn}
    return is_homogeneous(F) ?
        _with_chart_parameter_solver(
            f, F, p, options, tracker_options, rng, start_solutions,
        ) :
        _with_affine_parameter_solver(f, F, p, options, tracker_options)
end

# Homogeneous system: solutions are projective, put the problem on a random
# affine chart. All workers must share the SAME chart so deduplication is
# consistent.
function _with_chart_parameter_solver(
        f::Fn, F::SystemLike, p::Vector{ComplexF64}, options::MonodromyOptions,
        tracker_options::TrackerOptions, rng::Random.AbstractRNG,
        start_solutions::AbstractVector{<:AbstractVector},
    ) where {Fn}
    n = nvariables(F)
    chart = _conditioned_chart(rng, n, start_solutions)
    builder = ChartParameterMonodromyBuilder(F, p, chart, n, tracker_options)
    worker = _chart_parameter_monodromy_worker(
        F.evaluator, p, chart, n, tracker_options,
    )
    return _with_certified_accumulator(F, options, copy(p), chart) do cs
        _with_monodromy_solver_from_builder(
            f, worker, builder, n, options, chart, true, cs,
        )
    end
end

function _with_affine_parameter_solver(
        f::Fn, F::SystemLike, p::Vector{ComplexF64}, options::MonodromyOptions,
        tracker_options::TrackerOptions,
    ) where {Fn}
    n = nvariables(F)
    builder = ParameterMonodromyBuilder(F, p, n, tracker_options)
    worker = _parameter_monodromy_worker(F.evaluator, p, n, tracker_options)
    return _with_certified_accumulator(F, options, copy(p), ComplexF64[]) do cs
        _with_monodromy_solver_from_builder(
            f, worker, builder, n, options, ComplexF64[], false, cs,
        )
    end
end

function _subspace_monodromy_worker(
        H::AbstractHomotopy, tracker_options::TrackerOptions,
        L::LinearSubspace{ComplexF64}, n::Int, worker_chart::Vector{ComplexF64},
    )
    eg = endgame_tracker(
        H, tracker_options, EndgameOptions(; endgame_start = 0.0),
    )
    n_u = size(H)[2]
    return MonodromyWorkerState{typeof(H), LinearSubspace{ComplexF64}}(
        H, eg, copy(L), zeros(ComplexF64, n), zeros(ComplexF64, n_u),
        worker_chart,
    )
end

# `INTRINSIC` and `PROJECTIVE` select which of three homotopies every worker gets.
# They are decided once, when the solver is built.
struct SubspaceMonodromyBuilder{S <: SystemLike, INTRINSIC, PROJECTIVE} <:
    AbstractPathBuilder
    system::S
    subspace::LinearSubspace{ComplexF64}
    chart::Vector{ComplexF64}
    nvariables::Int
    tracker_options::TrackerOptions
    # Every worker tracks the SAME homotopy, so the perturbation is drawn once
    # here rather than per worker.
    gamma::ComplexF64
end

function (builder::SubspaceMonodromyBuilder{S, INTRINSIC, PROJECTIVE})() where {
        S, INTRINSIC, PROJECTIVE,
    }
    L = builder.subspace
    sys_eval = _clone_system_evaluator(builder.system)
    H = if INTRINSIC
        base_eval = PROJECTIVE ?
            SystemEvaluator(AffineChartSystem(sys_eval, builder.chart)) : sys_eval
        IntrinsicSubspaceHomotopy(base_eval, L, L; gamma = builder.gamma)
    else
        He = ExtrinsicSubspaceHomotopy(sys_eval, L, L; gamma = builder.gamma)
        PROJECTIVE ? AffineChartHomotopy(He, builder.chart) : He
    end
    # For the intrinsic projective case the chart row is buried inside the
    # wrapped AffineChartSystem, so the worker keeps its own reference for
    # normalizing start points (extrinsic reaches it via the homotopy).
    worker_chart = INTRINSIC && PROJECTIVE ? builder.chart : ComplexF64[]
    return _subspace_monodromy_worker(
        H, builder.tracker_options, L, builder.nvariables, worker_chart,
    )
end

function with_monodromy_solver(
        f::Fn, F::SystemLike, L::LinearSubspace{ComplexF64};
        options::MonodromyOptions = MonodromyOptions(),
        tracker_options::TrackerOptions = TrackerOptions(),
        intrinsic::Bool = _default_intrinsic(L),
        rng::Random.AbstractRNG = Random.default_rng(),
        start_solutions::AbstractVector{<:AbstractVector} = Vector{ComplexF64}[],
    ) where {Fn}
    n = nvariables(F)
    projective = is_linear(L) && is_homogeneous(F)
    # All workers must share the SAME chart so deduplication is consistent. It
    # is unused affinely, where the draw only keeps the random stream in step.
    chart = projective ? _conditioned_chart(rng, n, start_solutions) :
        randn(rng, ComplexF64, n)
    gamma = _random_gamma(rng)
    return _with_subspace_builder(
        F, L, chart, n, tracker_options, intrinsic, projective, gamma,
    ) do builder
        _with_certified_accumulator(
            F, options, L, projective ? chart : ComplexF64[],
        ) do cs
            _with_monodromy_solver_from_builder(
                f, builder(), builder, n, options, chart, projective, cs,
            )
        end
    end
end

# `intrinsic` and `projective` decide which of three homotopies every worker gets.
function _with_subspace_builder(
        f::Fn, F::SystemLike, L::LinearSubspace{ComplexF64},
        chart::Vector{ComplexF64}, n::Int, tracker_options::TrackerOptions,
        intrinsic::Bool, projective::Bool, gamma::ComplexF64,
    ) where {Fn}
    S = typeof(F)
    args = (F, L, chart, n, tracker_options, gamma)
    return intrinsic ?
        (
            projective ? f(SubspaceMonodromyBuilder{S, true, true}(args...)) :
            f(SubspaceMonodromyBuilder{S, true, false}(args...))
        ) :
        (
            projective ? f(SubspaceMonodromyBuilder{S, false, true}(args...)) :
            f(SubspaceMonodromyBuilder{S, false, false}(args...))
        )
end

function add_loop!(
        MS::MonodromySolver{H, P}, rng::Random.AbstractRNG,
        results::Vector{PathResult},
    ) where {H, P}
    base = MS.workers[1].base
    push!(
        MS.loops,
        MonodromyLoop(
            base, MS.options.parameter_sampler, rng, _trace_step(base, results),
        ),
    )
    Threads.atomic_add!(MS.statistics.generated_loops, 1)
    if MS.options.permutations
        push!(MS.statistics.permutations, zeros(Int, length(MS.unique_points)))
    end
    return MS
end
loop(MS::MonodromySolver, i::Int) = MS.loops[i]
nloops(MS::MonodromySolver)::Int = length(MS.loops)

_reset_certified!(::HeuristicMonodromySolver) = nothing
_reset_certified!(MS::CertifiedMonodromySolver) =
    (empty!(MS.certified_solutions); nothing)

_size_certified_caches!(::HeuristicMonodromySolver, ::Int) = nothing
_size_certified_caches!(MS::CertifiedMonodromySolver, ntasks::Int) =
    monodromy_size_caches!(MS.certified_solutions, ntasks)

function reset_loops!(MS::MonodromySolver)
    empty!(MS.loops)
    return MS
end

function reset_trace!(MS::MonodromySolver)::Nothing
    MS.trace .= 0
    MS.trace[end, :] .= 1
    MS.trace_paths = 0
    MS.trace_dropped = 0
    return nothing
end

# A trace summed over fewer paths than were sent around the loop says nothing
# about the witness set: it is short by whatever the lost paths would have
# contributed. Callers acting on `trace_colinearity` must attribute a failure
# through this first.
trace_complete(MS::MonodromySolver)::Bool = MS.trace_dropped == 0

# Whether the trace says anything about the witness set at all: it must have
# summed at least one path and lost none. `reset_trace!` leaves the augmentation
# row in place, so an untouched trace matrix has rank one and `trace_colinearity`
# reads it as perfectly colinear. Without the path count a trace that never ran
# is indistinguishable from one that passed.
trace_conclusive(MS::MonodromySolver)::Bool =
    MS.trace_paths > 0 && trace_complete(MS)

# Colinearity measure of the three accumulated trace columns: σ₃/σ₁ of the
# singular values. Near zero iff the columns are (affinely) colinear.
function trace_colinearity(MS::MonodromySolver)::Float64
    σ = LA.svdvals(MS.trace)
    return σ[3] / σ[1]
end

# The cap `sqrt(accuracy)` can fall below the `1e-14` floor for an extremely
# accurate endpoint; `max(..., 1e-14)` keeps the clamp bounds ordered (lo ≤ hi)
# so the floor wins instead of the clamp silently returning a value above the cap.
"""
    uniqueness_rtol(res::PathResult)

Relative tolerance for the uniqueness check of a monodromy endpoint, derived
from the endpoint's Newton certificates: within this radius Newton's method
contracts to the same solution.
"""
uniqueness_rtol(res::PathResult)::Float64 =
    clamp(0.25 * inv(res.ω)^2, 1.0e-14, max(1.0e-14, sqrt(res.accuracy)))

# Trace-test columns of one loop, for a consumer with no solver at hand.
# `nothing` unless the loop reached its halfway subspace.
# An empty matrix is the no-trace sentinel: a collected trace always has the
# three columns the test sums over, so `isempty` is unambiguous.
mutable struct TraceColumns
    columns::Matrix{ComplexF64}
end

TraceColumns() = TraceColumns(Matrix{ComplexF64}(undef, 0, 0))

function _accumulate_trace!(
        MS::MonodromySolver, x₀::Vector{ComplexF64}, x₀₁::Vector{ComplexF64},
        x₁::Vector{ComplexF64},
    )::Nothing
    Base.@lock MS.trace_lock begin
        for i in eachindex(x₀)
            MS.trace[i, 1] += x₀[i]
            MS.trace[i, 2] += x₀₁[i]
            MS.trace[i, 3] += x₁[i]
        end
        MS.trace_paths += 1
    end
    return nothing
end

# A path asked for trace columns that never reached the halfway subspace.
function _trace_dropped!(MS::MonodromySolver)::Nothing
    Base.@lock MS.trace_lock (MS.trace_dropped += 1)
    return nothing
end

# The driver counts for a `TraceColumns` sink: the worker returning it is a
# separate process and holds no solver.
_trace_dropped!(::TraceColumns)::Nothing = nothing

function _accumulate_trace!(
        sink::TraceColumns, x₀::Vector{ComplexF64}, x₀₁::Vector{ComplexF64},
        x₁::Vector{ComplexF64},
    )::Nothing
    sink.columns = [x₀ x₀₁ x₁]
    return nothing
end

# Fold columns collected elsewhere into the solver's trace matrix.
function _accumulate_trace!(MS::MonodromySolver, columns::Matrix{ComplexF64})::Nothing
    Base.@lock MS.trace_lock begin
        for j in 1:3, i in axes(columns, 1)
            MS.trace[i, j] += columns[i, j]
        end
        MS.trace_paths += 1
    end
    return nothing
end

"""
    track_loop!(ws, loop::MonodromyLoop, res::PathResult, collect_trace, MS)
    track_loop!(ws, loop, x, ω, μ, extended_precision, collect_trace, trace_sink)

Track the solution of `res` (or the point `x` with certificates `ω`, `μ`,
`extended_precision`) around the monodromy loop `loop`. Middle segments run the
raw inner tracker warm-started with those certificates; the final segment back
to the base parameters runs the full endgame tracker and produces the returned
[`PathResult`](@ref). Returns `nothing` if any segment fails. When
`collect_trace` (subspace loops only), the segment chain goes through the
halfway subspace `p₀₁` and the column sums go to `trace_sink`, either a
[`MonodromySolver`](@ref) or a [`TraceColumns`](@ref).
"""
function track_loop!(
        ws::MonodromyWorkerState, loop::MonodromyLoop{P}, res::PathResult,
        collect_trace::Bool, MS::MonodromySolver,
    )::PathResult where {P}
    return track_loop!(
        ws, loop, solution(res), res.ω, res.μ, res.extended_precision_used,
        collect_trace, MS,
    )
end

function track_loop!(
        ws::MonodromyWorkerState, loop::MonodromyLoop{P},
        x_start::Vector{ComplexF64}, ω::Float64, μ::Float64,
        extended_precision::Bool, collect_trace::Bool, trace_sink::TS,
    )::PathResult where {P, TS}
    tr = ws.tracker.tracker
    x = ws.x_buffer
    copyto!(x, x_start)

    # A segment failure yields a tracker-only `PathResult` carrying the failing
    # code, so every exit of this function has the same concrete type. Callers
    # filter on `is_success`.
    if P === LinearSubspace{ComplexF64} && collect_trace
        x₀ = copy(x)
        set_loop_segment!(ws, loop, 1)   # p → p₀₁
        if !_track_middle_segment!(ws, ω, μ, extended_precision)
            _trace_dropped!(trace_sink)
            return PathResult(tr; start_solution = x_start)
        end
        x₀₁ = copy(x)
        set_loop_segment!(ws, loop, 2)   # p₀₁ → p₁
        if !_track_middle_segment!(ws, tr.state.ω, tr.state.μ, tr.state.extended_prec)
            _trace_dropped!(trace_sink)
            return PathResult(tr; start_solution = x_start)
        end
        x₁ = copy(x)
        _accumulate_trace!(trace_sink, x₀, x₀₁, x₁)
    else
        # p → p₁ directly (the halfway point is skipped without trace).
        _retarget!(ws, loop.p, loop.p₁)
        _track_middle_segment!(ws, ω, μ, extended_precision) ||
            return PathResult(tr; start_solution = x_start)
    end

    _retarget!(ws, loop.p₁, loop.p₂)
    _track_middle_segment!(ws, tr.state.ω, tr.state.μ, tr.state.extended_prec) ||
        return PathResult(tr; start_solution = x_start)

    # Final segment back to base: full endgame tracker produces the PathResult.
    _retarget!(ws, loop.p₂, loop.p)
    u = _tracker_input!(ws)
    track!(
        ws.tracker, u;
        ω = tr.state.ω, μ = tr.state.μ, extended_precision = tr.state.extended_prec,
    )
    return _finalize_result(ws, PathResult(ws.tracker))
end
