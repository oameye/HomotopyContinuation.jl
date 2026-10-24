######################
## Monodromy solver ##
######################

# Chart-normalizing wrapper implementing the 2-arg `actions(cb, s)` protocol
# for UniquePoints: every orbit image is renormalized onto the chart before
# comparison. Used when tracking on an affine chart (projective problems).
struct ChartActions{GA <: GroupActions}
    chart::Vector{ComplexF64}
    actions::GA
end
function (CA::ChartActions)(cb::CB, s) where {CB}
    return apply_actions(CA.actions, s) do v
        w = copy(v)
        on_chart!(w, CA.chart)
        cb(w)
    end
end

"""
    MonodromyWorkerState

Per-worker state for the monodromy solver: the endgame tracker, the base
parameters, ambient/tracker scratch buffers and the homotopy the worker
retargets. That homotopy is the exact instance captured in the tracker's
evaluator closures, so retargeting it retargets the tracker. It sits in exactly
one of `parameter`, `intrinsic` and `extrinsic`, the route the builder chose;
`chart` is empty unless the solutions are projective.
"""
struct MonodromyWorkerState{P}
    tracker::EndgameTracker
    base::P
    x_buffer::Vector{ComplexF64}
    u_buffer::Vector{ComplexF64}
    chart::Vector{ComplexF64}
    parameter::Vector{ParameterHomotopy}
    intrinsic::Vector{IntrinsicSubspaceHomotopy}
    extrinsic::Vector{ExtrinsicSubspaceHomotopy}
    homotopy_size::Tuple{Int, Int}
end

## Retargeting through the concrete handle

function retarget!(
        ws::MonodromyWorkerState{Vector{ComplexF64}}, p::Vector{ComplexF64},
        q::Vector{ComplexF64},
    )::Nothing
    parameters!(ws.parameter[1], p, q)
    return nothing
end
function retarget!(
        ws::MonodromyWorkerState{LinearSubspace{ComplexF64}},
        p::LinearSubspace{ComplexF64}, q::LinearSubspace{ComplexF64},
    )::Nothing
    if isempty(ws.intrinsic)
        set_subspaces!(ws.extrinsic[1], p, q)
    else
        set_subspaces!(ws.intrinsic[1], p, q)
    end
    return nothing
end

"""
    set_loop_segment!(ws, loop::MonodromyLoop, segment::Int)

Retarget the worker's homotopy to the given loop segment. Vector loops have 3
segments (`p → p₁ → p₂ → p`), subspace loops 4 (`p → p₀₁ → p₁ → p₂ → p`).
"""
function set_loop_segment!(
        ws::MonodromyWorkerState, loop::MonodromyLoop{Vector{ComplexF64}}, segment::Int,
    )::Nothing
    segment == 1 && return retarget!(ws, loop.p, loop.p₁)
    segment == 2 && return retarget!(ws, loop.p₁, loop.p₂)
    segment == 3 && return retarget!(ws, loop.p₂, loop.p)
    throw(ArgumentError("Vector-parameter loops have 3 segments, got $segment."))
end
function set_loop_segment!(
        ws::MonodromyWorkerState, loop::MonodromyLoop{LinearSubspace{ComplexF64}},
        segment::Int,
    )::Nothing
    segment == 1 && return retarget!(ws, loop.p, loop.p₀₁)
    segment == 2 && return retarget!(ws, loop.p₀₁, loop.p₁)
    segment == 3 && return retarget!(ws, loop.p₁, loop.p₂)
    segment == 4 && return retarget!(ws, loop.p₂, loop.p)
    throw(ArgumentError("Subspace loops have 4 segments, got $segment."))
end

## Coordinate transfer between the solver's ambient state and the tracker

# Write the tracker-side start coordinates for the current segment (t = 1)
# from ws.x_buffer into ws.u_buffer and return it. A projective point is moved
# onto the chart first: the intrinsic homotopy carries the chart row inside its
# wrapped system, the others as the extra row of an affine chart homotopy.
function tracker_input!(ws::MonodromyWorkerState)::Vector{ComplexF64}
    if isempty(ws.intrinsic)
        copyto!(ws.u_buffer, ws.x_buffer)
        isempty(ws.chart) || on_chart!(ws.u_buffer, ws.chart)
    else
        isempty(ws.chart) || on_chart!(ws.x_buffer, ws.chart)
        intrinsic_coordinates!(ws.u_buffer, ws.intrinsic[1], ws.x_buffer, complex(1.0))
    end
    return ws.u_buffer
end

# Convert the tracker's endpoint (t = 0) back into ws.x_buffer (ambient).
function tracker_output!(
        ws::MonodromyWorkerState, u::AbstractVector{ComplexF64},
    )::Nothing
    if isempty(ws.intrinsic)
        copyto!(ws.x_buffer, u)
    else
        ambient_coordinates!(ws.x_buffer, ws.intrinsic[1], u, complex(0.0))
    end
    return nothing
end

# Renormalize a finished ambient solution for dedup. The intrinsic route's
# ambient points already satisfy the chart row of its wrapped system.
function normalize_solution!(
        ws::MonodromyWorkerState, x::AbstractVector{ComplexF64},
    )::Nothing
    (isempty(ws.chart) || !isempty(ws.intrinsic)) && return nothing
    on_chart!(x, ws.chart)
    return nothing
end

# Copy of `r` with its solution replaced by the ambient conversion.
function with_solution(r::PathResult, sol::Vector{ComplexF64})::PathResult
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
function finalize_result(ws::MonodromyWorkerState, r::PathResult)::PathResult
    tracker_output!(ws, r.solution)
    normalize_solution!(ws, ws.x_buffer)
    return with_solution(r, copy(ws.x_buffer))
end

# Track ws.x_buffer through the current (already retargeted) segment with the
# RAW inner tracker (no endgame), warm-started, updating ws.x_buffer in place.
function track_middle_segment!(
        ws::MonodromyWorkerState, ω::Float64, μ::Float64, extended_precision::Bool,
    )::Bool
    tr = ws.tracker.tracker
    u = tracker_input!(ws)
    code = track!(tr, u, complex(1.0), complex(0.0), ω, μ, extended_precision)
    code == TrackerCode.TRACKER_SUCCESS || return false
    tracker_output!(ws, tr.state.x)
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
    retarget!(ws, ws.base, ws.base)
    copyto!(ws.x_buffer, x)
    u = tracker_input!(ws)
    track!(ws.tracker, u)
    return finalize_result(ws, PathResult(ws.tracker))
end

"""
    MonodromySolver

Shared state of a monodromy computation: the worker states (index 1 is the
serial worker; threaded runs grow the vector lazily via `builder`), the loop
list, the deduplication structure with its lock, options, statistics and the
trace matrix for the trace test.
"""
mutable struct MonodromySolver{
        P, B, UP <: UniquePoints, MO <: MonodromyOptions,
        CS <: Union{Nothing, AbstractCertifiedSolutions},
    }
    # Mutable: `loops`, `statistics` and the two trace counters are reset
    # between solves; all other fields are const.
    const workers::Vector{MonodromyWorkerState{P}}
    const builder::B                                  # callable () -> MonodromyWorkerState{P}
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
    P, B, UP, MO, Nothing,
} where {P, B, UP <: UniquePoints, MO <: MonodromyOptions}
const CertifiedMonodromySolver = MonodromySolver{
    P, B, UP, MO, CS,
} where {
    P, B, UP <: UniquePoints, MO <: MonodromyOptions,
    CS <: AbstractCertifiedSolutions,
}

# Run `f` on the group actions the deduplication compares orbits with: none when
# the options ask for none or switch them off, and a chart-normalizing wrapper
# when the solutions are projective.
with_chart_actions(
    f::Fn, ::MonodromyOptions{<:Any, Nothing}, ::Vector{ComplexF64}, ::Bool,
) where {Fn} = f(nothing)

function with_chart_actions(
        f::Fn, options::MonodromyOptions, chart::Vector{ComplexF64}, use_chart::Bool,
    ) where {Fn}
    options.equivalence_classes || return f(nothing)
    actions = options.group_actions
    # The constructor clears `equivalence_classes` when there is no group action,
    # so this branch is unreachable; it is what tells inference so.
    actions === nothing && return f(nothing)
    return use_chart ? f(ChartActions(chart, actions)) : f(actions)
end

function with_monodromy_solver_from_builder(
        f::Fn, worker::MonodromyWorkerState{P}, builder::B, n::Int,
        options::MO, chart::Vector{ComplexF64}, use_chart::Bool,
        # Required, so a new route cannot silently downgrade to the heuristic check.
        certified_solutions::CS,
    ) where {
        Fn, P, B, MO <: MonodromyOptions,
        CS <: Union{Nothing, AbstractCertifiedSolutions},
    }
    return with_chart_actions(options, chart, use_chart) do group_actions
        f(
            monodromy_solver_from_builder(
                worker, builder, n, options, group_actions, certified_solutions,
            ),
        )
    end
end

function monodromy_solver_from_builder(
        worker::MonodromyWorkerState{P}, builder::B, n::Int, options::MO,
        group_actions::GA, certified_solutions::CS,
    ) where {
        P, B, GA, MO <: MonodromyOptions,
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

"""
    MonodromyBuilder

Data-only recipe for a monodromy worker: the system, the base parameters or
subspace, the chart (empty unless the solutions are projective), and for a
subspace whether to track in intrinsic coordinates. Every worker tracks the same
homotopy, so the perturbation `gamma` is drawn once here rather than per worker.
"""
struct MonodromyBuilder{S <: SystemLike, P} <: AbstractPathBuilder
    system::S
    base::P
    chart::Vector{ComplexF64}
    nvariables::Int
    tracker_options::TrackerOptions
    gamma::ComplexF64
    intrinsic::Bool
end

(builder::MonodromyBuilder)() =
    monodromy_worker(clone_system_evaluator(builder.system), builder)

function monodromy_worker(
        sys_eval::SystemEvaluator, b::MonodromyBuilder{S, Vector{ComplexF64}},
    )::MonodromyWorkerState{Vector{ComplexF64}} where {S}
    H = ParameterHomotopy(sys_eval, b.base, b.base)
    parameter = ParameterHomotopy[H]
    intrinsic = IntrinsicSubspaceHomotopy[]
    extrinsic = ExtrinsicSubspaceHomotopy[]
    isempty(b.chart) &&
        return monodromy_worker_state(H, b, parameter, intrinsic, extrinsic)
    return monodromy_worker_state(
        AffineChartHomotopy(H, b.chart), b, parameter, intrinsic, extrinsic,
    )
end

function monodromy_worker(
        sys_eval::SystemEvaluator, b::MonodromyBuilder{S, LinearSubspace{ComplexF64}},
    )::MonodromyWorkerState{LinearSubspace{ComplexF64}} where {S}
    L = b.base
    parameter = ParameterHomotopy[]
    if b.intrinsic
        base_eval = isempty(b.chart) ? sys_eval :
            SystemEvaluator(AffineChartSystem(sys_eval, b.chart))
        Hi = IntrinsicSubspaceHomotopy(base_eval, L, L; gamma = b.gamma)
        return monodromy_worker_state(
            Hi, b, parameter, IntrinsicSubspaceHomotopy[Hi], ExtrinsicSubspaceHomotopy[],
        )
    end
    He = ExtrinsicSubspaceHomotopy(sys_eval, L, L; gamma = b.gamma)
    intrinsic = IntrinsicSubspaceHomotopy[]
    extrinsic = ExtrinsicSubspaceHomotopy[He]
    isempty(b.chart) &&
        return monodromy_worker_state(He, b, parameter, intrinsic, extrinsic)
    return monodromy_worker_state(
        AffineChartHomotopy(He, b.chart), b, parameter, intrinsic, extrinsic,
    )
end

function monodromy_worker_state(
        H::AbstractHomotopy, b::MonodromyBuilder{S, P},
        parameter::Vector{ParameterHomotopy},
        intrinsic::Vector{IntrinsicSubspaceHomotopy},
        extrinsic::Vector{ExtrinsicSubspaceHomotopy},
    )::MonodromyWorkerState{P} where {S, P}
    eg = endgame_tracker(
        H, b.tracker_options, EndgameOptions(; endgame_start = 0.0),
    )
    m, n_u = size(H)
    return MonodromyWorkerState{P}(
        eg, copy(b.base), zeros(ComplexF64, b.nvariables), zeros(ComplexF64, n_u),
        b.chart, parameter, intrinsic, extrinsic, (m, n_u),
    )
end

# Smallest normalized alignment |v'x| / (‖v‖‖x‖) over the start solutions.
function chart_alignment(
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
function conditioned_chart(
        rng::Random.AbstractRNG, n::Int, S::AbstractVector{<:AbstractVector},
    )::Vector{ComplexF64}
    chart = randn(rng, ComplexF64, n)
    q = chart_alignment(chart, S)
    for _ in 2:16
        q >= 0.2 && break
        v = randn(rng, ComplexF64, n)
        qv = chart_alignment(v, S)
        if qv > q
            chart, q = v, qv
        end
    end
    return chart
end

@noinline certification_needs_equations(F) = throw(
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
    fn(slice(G, full_subspace(nvariables(G)); chart = chart))
with_certified_system(
    fn::F, G::System, L::LinearSubspace{ComplexF64}, chart::Vector{ComplexF64},
) where {F} = fn(rebuild_sliced(G, L, chart))
with_certified_system(::F, G::SystemLike, ::Vector{ComplexF64}) where {F} =
    certification_needs_equations(G)
with_certified_system(
    ::F, G::SystemLike, ::LinearSubspace{ComplexF64}, ::Vector{ComplexF64},
) where {F} = certification_needs_equations(G)

# Run `f` on the accumulator the certified duplicate check files into, or on
# `nothing` under `DuplicateCheck.HEURISTIC`. A parameter run passes the
# parameters it certifies at, a witness-set run the subspace it intersects with;
# `chart` is empty unless the solutions are projective.
function with_certified_accumulator(
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

function with_certified_accumulator(
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
    with_monodromy_solver(f, F, p, options, tracker_options, rng, start_solutions)
    with_monodromy_solver(f, F, L, options, tracker_options, intrinsic, rng,
                          start_solutions)

Run `f` on the monodromy solver for `F` at the parameters `p`, or for the
parameter-free `F` intersected with the linear subspace `L`.

The solver is handed to `f` rather than returned: its deduplication structure
and duplicate-check accumulator are chosen from the options, so each branch
builds a differently parameterized solver. A homogeneous system has projective
solutions and is tracked on a random affine chart, which all workers share so
that deduplication is consistent.
"""
function with_monodromy_solver(
        f::Fn, F::SystemLike, p::Vector{ComplexF64}, options::MonodromyOptions,
        tracker_options::TrackerOptions, rng::Random.AbstractRNG,
        start_solutions::AbstractVector{<:AbstractVector},
    ) where {Fn}
    n = nvariables(F)
    projective = is_homogeneous(F)
    chart = projective ? conditioned_chart(rng, n, start_solutions) : ComplexF64[]
    builder = MonodromyBuilder(
        F, copy(p), chart, n, tracker_options, complex(1.0), false,
    )
    worker = monodromy_worker(F.evaluator, builder)
    return with_certified_accumulator(F, options, copy(p), chart) do cs
        with_monodromy_solver_from_builder(
            f, worker, builder, n, options, chart, projective, cs,
        )
    end
end

function with_monodromy_solver(
        f::Fn, F::SystemLike, L::LinearSubspace{ComplexF64}, options::MonodromyOptions,
        tracker_options::TrackerOptions, intrinsic::Bool, rng::Random.AbstractRNG,
        start_solutions::AbstractVector{<:AbstractVector},
    ) where {Fn}
    n = nvariables(F)
    projective = is_linear(L) && is_homogeneous(F)
    # It is unused affinely, where the draw only keeps the random stream in step.
    drawn = projective ? conditioned_chart(rng, n, start_solutions) :
        randn(rng, ComplexF64, n)
    chart = projective ? drawn : ComplexF64[]
    gamma = random_gamma(rng)
    builder = MonodromyBuilder(
        F, copy(L), chart, n, tracker_options, gamma, intrinsic,
    )
    return with_certified_accumulator(F, options, L, chart) do cs
        with_monodromy_solver_from_builder(
            f, builder(), builder, n, options, drawn, projective, cs,
        )
    end
end

function add_loop!(
        MS::MonodromySolver, rng::Random.AbstractRNG,
        results::Vector{PathResult},
    )
    base = MS.workers[1].base
    push!(
        MS.loops,
        MonodromyLoop(
            base, MS.options.parameter_sampler, rng, trace_step(base, results),
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

reset_certified!(::HeuristicMonodromySolver) = nothing
reset_certified!(MS::CertifiedMonodromySolver) =
    (empty!(MS.certified_solutions); nothing)

size_certified_caches!(::HeuristicMonodromySolver, ::Int) = nothing
size_certified_caches!(MS::CertifiedMonodromySolver, ntasks::Int) =
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

function accumulate_trace!(
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
function count_trace_dropped!(MS::MonodromySolver)::Nothing
    Base.@lock MS.trace_lock (MS.trace_dropped += 1)
    return nothing
end

# The driver counts for a `TraceColumns` sink: the worker returning it is a
# separate process and holds no solver.
count_trace_dropped!(::TraceColumns)::Nothing = nothing

function accumulate_trace!(
        sink::TraceColumns, x₀::Vector{ComplexF64}, x₀₁::Vector{ComplexF64},
        x₁::Vector{ComplexF64},
    )::Nothing
    sink.columns = [x₀ x₀₁ x₁]
    return nothing
end

# Fold columns collected elsewhere into the solver's trace matrix.
function accumulate_trace!(MS::MonodromySolver, columns::Matrix{ComplexF64})::Nothing
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
        if !track_middle_segment!(ws, ω, μ, extended_precision)
            count_trace_dropped!(trace_sink)
            return PathResult(tr; start_solution = x_start)
        end
        x₀₁ = copy(x)
        set_loop_segment!(ws, loop, 2)   # p₀₁ → p₁
        if !track_middle_segment!(ws, tr.state.ω, tr.state.μ, tr.state.extended_prec)
            count_trace_dropped!(trace_sink)
            return PathResult(tr; start_solution = x_start)
        end
        x₁ = copy(x)
        accumulate_trace!(trace_sink, x₀, x₀₁, x₁)
    else
        # p → p₁ directly (the halfway point is skipped without trace).
        retarget!(ws, loop.p, loop.p₁)
        track_middle_segment!(ws, ω, μ, extended_precision) ||
            return PathResult(tr; start_solution = x_start)
    end

    retarget!(ws, loop.p₁, loop.p₂)
    track_middle_segment!(ws, tr.state.ω, tr.state.μ, tr.state.extended_prec) ||
        return PathResult(tr; start_solution = x_start)

    # Final segment back to base: full endgame tracker produces the PathResult.
    retarget!(ws, loop.p₂, loop.p)
    u = tracker_input!(ws)
    track!(
        ws.tracker, u, complex(1.0), complex(0.0), tr.state.ω, tr.state.μ,
        tr.state.extended_prec,
    )
    return finalize_result(ws, PathResult(ws.tracker))
end
