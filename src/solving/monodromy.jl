## Monodromy solver.

@enumx MonodromyCode::Int8 begin
    IN_PROGRESS
    SUCCESS
    HEURISTIC_STOP
    TIMEOUT
    TERMINATED_CALLBACK
    INVALID_STARTVALUE
    INTERRUPTED
end

@enumx ReuseLoops::Int8 begin
    ALL
    RANDOM
    NONE
end

function _symbol_to_reuse_loops(s::Symbol)::ReuseLoops.T
    s == :all && return ReuseLoops.ALL
    s == :random && return ReuseLoops.RANDOM
    s == :none && return ReuseLoops.NONE
    throw(ArgumentError("Unknown `reuse_loops` value :$s. Expected :all, :random or :none."))
end
_symbol_to_reuse_loops(r::ReuseLoops.T)::ReuseLoops.T = r

always_false(args...) = false

"""
    independent_normal(p::AbstractVector)

Sample a vector where each entry is drawn independently from the complex
normal distribution by calling `randn(ComplexF64)`.

    independent_normal(L::LinearSubspace)

Creates a random linear subspace by calling [`rand_subspace`](@ref).
"""
independent_normal(p::AbstractVector)::Vector{ComplexF64} = randn(ComplexF64, length(p))
independent_normal(L::LinearSubspace)::LinearSubspace{ComplexF64} =
    convert(LinearSubspace{ComplexF64}, rand_subspace(ambient_dim(L); dim = dim(L)))

#####################
# Monodromy Options #
#####################

"""
    MonodromyOptions(; options...)

Options for [`monodromy_solve`](@ref). `group_actions` accepts a `Function`,
`Tuple`, `AbstractVector` or [`GroupActions`](@ref); `reuse_loops` accepts
`:all`, `:random`, `:none` or a `ReuseLoops` enum value.
"""
struct MonodromyOptions{D, GA <: Union{Nothing, GroupActions}, CB, PS}
    check_startsolutions::Bool
    group_actions::GA
    # Callback and sampler are user closures on the cold path; the type
    # parameters keep the struct concrete.
    loop_finished_callback::CB
    parameter_sampler::PS
    equivalence_classes::Bool
    # stopping heuristics
    trace_test::Bool
    trace_test_tol::Float64
    target_solutions_count::Union{Nothing, Int}
    timeout::Union{Nothing, Float64}
    min_solutions::Union{Nothing, Int}
    max_loops_no_progress::Int
    reuse_loops::ReuseLoops.T
    permutations::Bool
    # unique points options
    distance::D
    triangle_inequality::Union{Nothing, Bool}
    unique_points_atol::Union{Nothing, Float64}
    unique_points_rtol::Union{Nothing, Float64}
    single_loop_per_start_solution::Bool
end

function MonodromyOptions(;
        check_startsolutions::Bool = true,
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        loop_finished_callback = always_false,
        parameter_sampler = independent_normal,
        equivalence_classes::Bool = group_actions !== nothing,
        trace_test::Bool = true,
        trace_test_tol::Float64 = 1.0e-6,
        target_solutions_count::Union{Nothing, Int} = nothing,
        timeout::Union{Nothing, Real} = nothing,
        min_solutions::Union{Nothing, Int} = nothing,
        max_loops_no_progress::Int = 5,
        reuse_loops::Union{Symbol, ReuseLoops.T} = ReuseLoops.ALL,
        permutations::Bool = false,
        distance = InfNorm(),
        triangle_inequality::Union{Nothing, Bool} = nothing,
        unique_points_atol::Union{Nothing, Float64} = nothing,
        unique_points_rtol::Union{Nothing, Float64} = nothing,
        single_loop_per_start_solution::Bool = false,
    )
    if group_actions isa Function || group_actions isa Tuple ||
            group_actions isa AbstractVector
        group_actions = GroupActions(group_actions)
    end
    # Equivalence classes only make sense with group actions.
    group_actions === nothing && (equivalence_classes = false)
    return MonodromyOptions(
        check_startsolutions,
        group_actions,
        loop_finished_callback,
        parameter_sampler,
        equivalence_classes,
        trace_test,
        trace_test_tol,
        target_solutions_count,
        timeout === nothing ? nothing : Float64(timeout),
        min_solutions,
        max_loops_no_progress,
        _symbol_to_reuse_loops(reuse_loops),
        permutations,
        distance,
        triangle_inequality,
        unique_points_atol,
        unique_points_rtol,
        single_loop_per_start_solution,
    )
end

#######################
# Loop data structure #
#######################

# A single unit of work: track the solution with index `id` around loop `loop_id`.
struct LoopTrackingJob
    id::Int
    loop_id::Int
end

struct MonodromyLoop{P <: Union{LinearSubspace{ComplexF64}, Vector{ComplexF64}}}
    # p -> p₁ -> p₂ -> p (vector case, 3 segments)
    # p -> p₀₁ -> p₁ -> p₂ -> p (subspace case, 4 segments)
    # Per-segment geodesics are recomputed on retarget (set_subspaces!), not
    # cached here.
    p::P
    p₀₁::P # halfway
    p₁::P
    p₂::P
end

function MonodromyLoop(base::AbstractVector, parameter_sampler::PS) where {PS}
    p = convert(Vector{ComplexF64}, base)
    p₁ = convert(Vector{ComplexF64}, parameter_sampler(p))
    p₂ = convert(Vector{ComplexF64}, parameter_sampler(p))

    # The stored halfway point is 0.5(p₁ - p), not p + 0.5(p₁ - p). It is
    # unused for vector parameters (the loop is a 3-segment chain).
    return MonodromyLoop(p, 0.5 .* (p₁ .- p), p₁, p₂)
end

function MonodromyLoop(base::LinearSubspace, parameter_sampler::PS) where {PS}
    L = convert(LinearSubspace{ComplexF64}, base)
    # The second linear space is just a translation in order to perform a
    # trace test. To still find new solutions quickly we translate the linear
    # space by a larger distance.
    # EQUAL SPACING of L, L₀₁, L₁ is load-bearing for the trace test
    # (prototype 11): L₀₁ - L == L₁ - L₀₁ == v.
    v = LA.rmul!(LA.normalize!(randn(ComplexF64, codim(L))), 5)
    L₀₁ = translate(L, v, Extrinsic)
    L₁ = translate(L₀₁, v, Extrinsic)
    L₂ = convert(LinearSubspace{ComplexF64}, parameter_sampler(L))

    return MonodromyLoop(L, L₀₁, L₁, L₂)
end

##########################
## Monodromy Statistics ##
##########################

Base.@kwdef mutable struct MonodromyStatistics
    tracked_loops::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    tracking_failures::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    generated_loops::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    solutions::Vector{Int} = Int[]                 # nsolutions after each finished loop generation
    permutations::Vector{Vector{Int}} = Vector{Int}[]
end

function Base.show(io::IO, S::MonodromyStatistics)
    println(io, "MonodromyStatistics")
    println(io, " • tracked_loops → ", S.tracked_loops[])
    println(io, " • tracking_failures → ", S.tracking_failures[])
    print(io, " • solutions → ", S.solutions)
    return
end

function loop_tracked!(stats::MonodromyStatistics)
    Threads.atomic_add!(stats.tracked_loops, 1)
    return stats
end
function loop_failed!(stats::MonodromyStatistics)
    Threads.atomic_add!(stats.tracking_failures, 1)
    return stats
end
function loop_finished!(stats::MonodromyStatistics, nsolutions::Int)
    push!(stats.solutions, nsolutions)
    return stats
end
# Record that solution `start_id` mapped to solution `end_id` under loop
# `loop_id` (0 marks a failed track). The per-loop permutation vector grows on
# demand.
function add_permutation!(
        stats::MonodromyStatistics, loop_id::Int, start_id::Int, end_id::Int,
    )
    perms = stats.permutations[loop_id]
    while length(perms) < start_id
        push!(perms, 0)
    end
    perms[start_id] = end_id
    return stats
end

function solutions_current_loop(stats::MonodromyStatistics, nsolutions::Int)
    return isempty(stats.solutions) ? nsolutions : nsolutions - stats.solutions[end]
end
function solutions_last_loop(stats::MonodromyStatistics)
    return length(stats.solutions) > 1 ? stats.solutions[end] - stats.solutions[end - 1] : 0
end

# Number of consecutive finished loop generations without solution growth.
function loops_no_change(stats::MonodromyStatistics, nsolutions::Int)
    k = 0
    for i in length(stats.solutions):-1:1
        stats.solutions[i] == nsolutions || break
        k += 1
    end
    return max(k - 1, 0)
end

@noinline function make_showvalues(
        stats::MonodromyStatistics; queued::Int, solutions::Int,
    )
    return [
        ("tracked loops (queued)", "$(stats.tracked_loops[]) ($queued)"),
        (
            "solutions in current (last) loop",
            "$(solutions_current_loop(stats, solutions)) ($(solutions_last_loop(stats)))",
        ),
        (
            "generated loops (no change)",
            "$(stats.generated_loops[]) ($(loops_no_change(stats, solutions)))",
        ),
    ]
end

############
## Result ##
############

"""
    MonodromyResult

Contains the result of a [`monodromy_solve`](@ref) computation.
"""
struct MonodromyResult{P, LP}
    returncode::MonodromyCode.T
    results::Vector{PathResult}
    parameters::P
    loops::Vector{MonodromyLoop{LP}}
    statistics::MonodromyStatistics
    equivalence_classes::Bool
    seed::UInt32
    trace::Union{Nothing, Float64}
end

function Base.show(io::IO, result::MonodromyResult)
    println(io, "MonodromyResult")
    println(io, "="^length("MonodromyResult"))
    println(io, "• return_code → ", result.returncode)
    if result.equivalence_classes
        println(io, "• $(nsolutions(result)) classes of solutions (modulo group action)")
    else
        println(io, "• $(nsolutions(result)) solutions")
    end
    println(io, "• $(result.statistics.tracked_loops[]) tracked loops")
    print(io, "• random_seed → ", sprint(show, result.seed))
    if result.trace !== nothing
        print(io, "\n• trace → ", sprint(show, result.trace))
    end
    return
end

"""
    is_success(result::MonodromyResult)

Returns true if the monodromy computation achieved its target solution count.
"""
is_success(result::MonodromyResult)::Bool = result.returncode == MonodromyCode.SUCCESS

"""
    is_heuristic_stop(result::MonodromyResult)

Returns true if the monodromy computation stopped due to the heuristic.
"""
is_heuristic_stop(result::MonodromyResult)::Bool =
    result.returncode == MonodromyCode.HEURISTIC_STOP

"""
    solutions(result::MonodromyResult)

Return all solutions.
"""
solutions(r::MonodromyResult)::Vector{Vector{ComplexF64}} =
    [solution(pr) for pr in r.results]

"""
    nsolutions(result::MonodromyResult)

Returns the number of solutions of the `result`.
"""
nsolutions(r::MonodromyResult)::Int = length(r.results)

"""
    results(result::MonodromyResult)

Returns the computed [`PathResult`](@ref)s.
"""
results(r::MonodromyResult)::Vector{PathResult} = r.results

"""
    nresults(result::MonodromyResult)

Returns the number of results computed.
"""
nresults(r::MonodromyResult)::Int = length(r.results)

"""
    parameters(result::MonodromyResult)

Return the parameters corresponding to the given result `r`.
"""
parameters(r::MonodromyResult) = r.parameters

"""
    seed(result::MonodromyResult)

Return the random seed used for the computations.
"""
seed(r::MonodromyResult)::UInt32 = r.seed

"""
    trace(result::MonodromyResult)

Return the result of the trace test computed during the monodromy.
"""
trace(r::MonodromyResult)::Union{Nothing, Float64} = r.trace

"""
    permutations(r::MonodromyResult; reduced = true)

Return the permutations of the solutions that are induced by tracking over the
loops, as a matrix whose columns are permutations. If `reduced = false`, all
recorded permutations are returned; otherwise repetitions are removed.
If a solution was not tracked in a loop, the corresponding entry is 0.
"""
function permutations(r::MonodromyResult; reduced::Bool = true)::Matrix{Int}
    π = r.statistics.permutations
    N = nresults(r)

    π = filter(πⱼ -> length(πⱼ) == N, π)
    if reduced
        π = unique(π)
    end

    A = zeros(Int, N, length(π))
    for (j, πⱼ) in enumerate(π), i in 1:N
        A[i, j] = πⱼ[i]
    end

    return A
end

## find_start_pair (spec 5.D)

"""
    find_start_pair(F::System; max_tries = 1_000, atol = 0.0, rtol = 1e-12)

Try to find a pair `(x, p)` for the system `F` such that `F(x, p) = 0` by
sampling a random `x` and solving the linear system in the parameters (when
`F` is linear in the parameters), or by a Newton solve of the joint system in
`(x, p)` otherwise. For a parameter-free system, returns `(x, nothing)` with
`F(x) = 0`. Returns `nothing` if no pair could be found in `max_tries` tries.
"""
function find_start_pair(
        F::System;
        max_tries::Int = 1_000,
        atol::Float64 = 0.0,
        rtol::Float64 = 1.0e-12,
    )::Union{Nothing, Tuple{Vector{ComplexF64}, Union{Nothing, Vector{ComplexF64}}}}
    refine_atol = atol > 0 ? atol : 1.0e-12
    strategy = nparameters(F) == 0 ?
        _parameter_free_start_pair : _parameterized_start_pair
    # Parameter count is construction-time policy stored as a runtime field.
    # Prevent inference from traversing both Newton-on-F and parameter-system
    # construction for every automatic monodromy start.
    strategy = Base.inferencebarrier(strategy)
    return _dispatch_start_pair_strategy(
        strategy, F, max_tries, refine_atol, rtol,
    )
end

@noinline function _dispatch_start_pair_strategy(
        strategy::Function, F::System, max_tries::Int,
        refine_atol::Float64, rtol::Float64,
    )::Union{Nothing, Tuple{Vector{ComplexF64}, Union{Nothing, Vector{ComplexF64}}}}
    Base.@nospecialize strategy F
    return strategy(F, max_tries, refine_atol, rtol)
end

@noinline function _parameter_free_start_pair(
        F::System, max_tries::Int, refine_atol::Float64, rtol::Float64,
    )::Union{Nothing, Tuple{Vector{ComplexF64}, Nothing}}
    nvars = nvariables(F)
    cache = NewtonCache(F)
    for _ in 1:max_tries
        x₀ = randn(ComplexF64, nvars)
        res = newton(F, x₀; atol = 1.0e-8, cache = cache)
        if res.return_code == NewtonReturnCode.NEWTON_SUCCESS
            refined = newton(
                F, res.x; atol = refine_atol, rtol = rtol, cache = cache,
            )
            if refined.return_code == NewtonReturnCode.NEWTON_SUCCESS
                return (refined.x, nothing)
            end
        end
    end
    return nothing
end

@noinline function _parameterized_start_pair(
        F::System, max_tries::Int, refine_atol::Float64, rtol::Float64,
    )::Union{Nothing, Tuple{Vector{ComplexF64}, Vector{ComplexF64}}}

    # 1. Linear-in-parameters fast path (prototype 1). Each attempt draws a
    # fresh random x₀ internally, so a `nothing` (bad draw) should retry, not
    # abandon the fast path.
    for _ in 1:3
        pair = _linear_in_params_start_pair(F)
        pair === nothing && continue
        return pair
    end

    # The joint-Newton fallback is rare and much wider than the linear path.
    # Cross a hard function barrier so successful linear starts do not compile
    # it speculatively.
    fallback = Base.inferencebarrier(_joint_newton_start_pair)
    return _dispatch_start_pair_strategy(
        fallback, F, max_tries, refine_atol, rtol,
    )
end

@noinline function _joint_newton_start_pair(
        F::System, max_tries::Int, refine_atol::Float64, rtol::Float64,
    )::Union{Nothing, Tuple{Vector{ComplexF64}, Vector{ComplexF64}}}
    nvars = nvariables(F)
    np = nparameters(F)
    G = System(
        collect(F.polys);
        variables = [collect(F.variables); collect(F.parameters)],
    )
    cache = NewtonCache(G)
    F_cache = NewtonCache(F)
    for _ in 1:max_tries
        xp₀ = randn(ComplexF64, nvars + np)
        res = newton(G, xp₀; atol = 1.0e-8, cache = cache)
        if res.return_code == NewtonReturnCode.NEWTON_SUCCESS
            x = res.x[1:nvars]
            p = res.x[(nvars + 1):end]
            refined = newton(
                F, x; p = p, atol = refine_atol, rtol = rtol, cache = F_cache,
            )
            if refined.return_code == NewtonReturnCode.NEWTON_SUCCESS
                return (refined.x, p)
            end
        end
    end
    return nothing
end

# Fast path: sample x₀, substitute it into every polynomial and check that the
# result is linear in the parameters (every term touches at most one parameter,
# with exponent at most 1, and every equation actually contains a parameter).
# Then solve the linear system A p = b exactly.
function _linear_in_params_start_pair(
        F::System,
    )::Union{Nothing, Tuple{Vector{ComplexF64}, Vector{ComplexF64}}}
    nvars = nvariables(F)
    np = nparameters(F)
    m = length(F.polys)
    m <= np || return nothing

    x₀ = randn(ComplexF64, nvars)
    vars = collect(F.variables)
    params = collect(F.parameters)
    pidx = Dict(p => j for (j, p) in enumerate(params))

    A = zeros(ComplexF64, m, np)
    b = zeros(ComplexF64, m)
    for (i, f) in enumerate(F.polys)
        g = MP.subs(f, vars => x₀)
        has_param_term = false
        for t in MP.terms(g)
            mon = MP.monomial(t)
            d = MP.degree(mon)
            if d == 0
                b[i] -= ComplexF64(MP.coefficient(t))
            elseif d == 1
                j = 0
                for (v, e) in MP.powers(mon)
                    if e > 0
                        j = get(pidx, v, 0)
                        break
                    end
                end
                j == 0 && return nothing
                A[i, j] += ComplexF64(MP.coefficient(t))
                has_param_term = true
            else
                # parameter-degree >= 2 or a term mixing several parameters
                return nothing
            end
        end
        # A parameter-free equation cannot be satisfied by choosing p.
        has_param_term || return nothing
    end

    p₀ = if iszero(b)
        # Only the trivial solution when the system is square; otherwise
        # sample from the nullspace.
        m == np && return nothing
        N = LA.nullspace(A)
        size(N, 2) == 0 && return nothing
        Vector{ComplexF64}(N * randn(ComplexF64, size(N, 2)))
    else
        Vector{ComplexF64}(LA.qr(A, LA.ColumnNorm()) \ b)
    end
    all(isfinite, p₀) || return nothing
    return (x₀, p₀)
end

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

Track `x` at the base parameters (trivial `p → p` retarget, full endgame).
Returns the resulting [`PathResult`](@ref) on success, `nothing` otherwise.
Used to validate and refine start solutions.
"""
function track_start!(
        ws::MonodromyWorkerState, x::AbstractVector{ComplexF64},
    )::Union{Nothing, PathResult}
    _retarget!(ws, ws.base, ws.base)
    copyto!(ws.x_buffer, x)
    u = _tracker_input!(ws)
    code = track!(ws.tracker, u)
    code == EndgameCode.SUCCESS || return nothing
    return _finalize_result(ws, PathResult(ws.tracker))
end

"""
    trust_start!(ws, x)

Refine `x` at the base parameters like [`track_start!`](@ref) but always return
a [`PathResult`](@ref) (never `nothing`), trusting the caller-supplied solution
instead of sorting out non-converged points. Used when `check_startsolutions`
is disabled.
"""
function trust_start!(
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
mutable struct MonodromySolver{H, P, B, UP <: UniquePoints, MO <: MonodromyOptions}
    # Mutable: `loops` and `statistics` are reset between solves; all other
    # fields are const.
    const workers::Vector{MonodromyWorkerState{H, P}}
    const builder::B                                  # callable () -> MonodromyWorkerState{H, P}
    loops::Vector{MonodromyLoop{P}}
    const unique_points::UP
    const unique_points_lock::ReentrantLock
    const options::MO
    statistics::MonodromyStatistics
    # We save the sums of the solutions for three values of the loop parameter
    # to check whether the trace is colinear. The sums are augmented by 1 to
    # make this work for the one/two variable cases: an (n + 1) × 3 matrix.
    const trace::Matrix{ComplexF64}
    const trace_lock::ReentrantLock
end

function _monodromy_solver_from_builder(
        builder::B, n::Int, ::Type{P}, options::MO, chart::Vector{ComplexF64},
        use_chart::Bool,
    ) where {B, P, MO <: MonodromyOptions}
    worker = builder()
    group_actions = options.equivalence_classes ? options.group_actions : nothing
    if group_actions !== nothing && use_chart
        group_actions = _ChartActions(chart, group_actions)
    end
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
        options,
        MonodromyStatistics(),
        trace,
        ReentrantLock(),
    )
end

function MonodromySolver(
        F::System, p::Vector{ComplexF64};
        options::MonodromyOptions = MonodromyOptions(),
        tracker_options::TrackerOptions = TrackerOptions(),
    )
    n = nvariables(F)
    if is_homogeneous(F)
        # Homogeneous system: solutions are projective, put the problem on a
        # random affine chart. All workers must share the SAME chart so
        # deduplication is consistent.
        chart = randn(ComplexF64, n)
        chart_builder = function ()
            sys_eval = _clone_system_evaluator(F)
            H = AffineChartHomotopy(ParameterHomotopy(sys_eval, p, p), chart)
            tracker = Tracker(HomotopyEvaluator(H); options = tracker_options)
            eg = EndgameTracker(tracker, EndgameOptions(; endgame_start = 0.0))
            return MonodromyWorkerState(
                H, eg, copy(p), zeros(ComplexF64, n), zeros(ComplexF64, n),
            )
        end
        return _monodromy_solver_from_builder(
            chart_builder, n, Vector{ComplexF64}, options, chart, true,
        )
    end
    builder = function ()
        sys_eval = _clone_system_evaluator(F)
        H = ParameterHomotopy(sys_eval, p, p)
        tracker = Tracker(HomotopyEvaluator(H); options = tracker_options)
        eg = EndgameTracker(tracker, EndgameOptions(; endgame_start = 0.0))
        return MonodromyWorkerState(
            H, eg, copy(p), zeros(ComplexF64, n), zeros(ComplexF64, n),
        )
    end
    return _monodromy_solver_from_builder(
        builder, n, Vector{ComplexF64}, options, ComplexF64[], false,
    )
end

function MonodromySolver(
        F::System, L::LinearSubspace{ComplexF64};
        options::MonodromyOptions = MonodromyOptions(),
        tracker_options::TrackerOptions = TrackerOptions(),
        intrinsic::Union{Nothing, Bool} = nothing,
    )
    n = nvariables(F)
    use_intrinsic = intrinsic === nothing ? dim(L) <= codim(L) : intrinsic
    projective = is_linear(L) && is_homogeneous(F)
    # All workers must share the SAME chart so deduplication is consistent.
    chart = randn(ComplexF64, n)
    builder = function ()
        sys_eval = _clone_system_evaluator(F)
        H = if use_intrinsic
            base_eval = projective ?
                SystemEvaluator(AffineChartSystem(sys_eval, chart)) : sys_eval
            IntrinsicSubspaceHomotopy(base_eval, L, L)
        else
            He = ExtrinsicSubspaceHomotopy(sys_eval, L, L)
            projective ? AffineChartHomotopy(He, chart) : He
        end
        tracker = Tracker(HomotopyEvaluator(H); options = tracker_options)
        eg = EndgameTracker(tracker, EndgameOptions(; endgame_start = 0.0))
        n_u = size(H)[2]
        # For the intrinsic projective case the chart row is buried inside the
        # wrapped AffineChartSystem, so the worker keeps its own reference for
        # normalizing start points (extrinsic reaches it via the homotopy).
        worker_chart = use_intrinsic && projective ? chart : ComplexF64[]
        return MonodromyWorkerState{typeof(H), LinearSubspace{ComplexF64}}(
            H, eg, copy(L), zeros(ComplexF64, n), zeros(ComplexF64, n_u),
            worker_chart,
        )
    end
    return _monodromy_solver_from_builder(
        builder, n, LinearSubspace{ComplexF64}, options, chart, projective,
    )
end

function add_loop!(MS::MonodromySolver{H, P}) where {H, P}
    base = MS.workers[1].base
    push!(MS.loops, MonodromyLoop(base, MS.options.parameter_sampler))
    Threads.atomic_add!(MS.statistics.generated_loops, 1)
    if MS.options.permutations
        push!(MS.statistics.permutations, zeros(Int, length(MS.unique_points)))
    end
    return MS
end
loop(MS::MonodromySolver, i::Int) = MS.loops[i]
nloops(MS::MonodromySolver)::Int = length(MS.loops)

function reset_loops!(MS::MonodromySolver)
    empty!(MS.loops)
    return MS
end

function reset_trace!(MS::MonodromySolver)::Nothing
    MS.trace .= 0
    MS.trace[end, :] .= 1
    return nothing
end

# Colinearity measure of the three accumulated trace columns: σ₃/σ₁ of the
# singular values. Near zero iff the columns are (affinely) colinear.
function trace_colinearity(MS::MonodromySolver)::Float64
    σ = LA.svdvals(MS.trace)
    return σ[3] / σ[1]
end

"""
    uniqueness_rtol(res::PathResult)

Relative tolerance for the uniqueness check of a monodromy endpoint, derived
from the endpoint's Newton certificates: within this radius Newton's method
contracts to the same solution.
"""
# The cap `sqrt(accuracy)` can fall below the `1e-14` floor for an extremely
# accurate endpoint; `max(..., 1e-14)` keeps the clamp bounds ordered (lo ≤ hi)
# so the floor wins instead of the clamp silently returning a value above the cap.
uniqueness_rtol(res::PathResult)::Float64 =
    clamp(0.25 * inv(res.ω)^2, 1.0e-14, max(1.0e-14, sqrt(res.accuracy)))

"""
    track_loop!(ws, loop::MonodromyLoop, res::PathResult, collect_trace, MS)

Track the solution of `res` around the monodromy loop `loop`. Middle segments
run the raw inner tracker warm-started with the endpoint certificates; the
final segment back to the base parameters runs the full endgame tracker and
produces the returned [`PathResult`](@ref). Returns `nothing` if any segment
fails. When `collect_trace` (subspace loops only), the segment chain goes
through the halfway subspace `p₀₁` and the column sums for the trace test are
accumulated into `MS.trace` under `MS.trace_lock`.
"""
function track_loop!(
        ws::MonodromyWorkerState, loop::MonodromyLoop{P}, res::PathResult,
        collect_trace::Bool, MS::MonodromySolver,
    )::Union{Nothing, PathResult} where {P}
    tr = ws.tracker.tracker
    x = ws.x_buffer
    copyto!(x, solution(res))

    if P === LinearSubspace{ComplexF64} && collect_trace
        x₀ = copy(x)
        set_loop_segment!(ws, loop, 1)   # p → p₀₁
        _track_middle_segment!(ws, res.ω, res.μ, res.extended_precision_used) ||
            return nothing
        x₀₁ = copy(x)
        set_loop_segment!(ws, loop, 2)   # p₀₁ → p₁
        _track_middle_segment!(ws, tr.state.ω, tr.state.μ, tr.state.extended_prec) ||
            return nothing
        x₁ = copy(x)
        Base.@lock MS.trace_lock begin
            for i in 1:length(x₀)
                MS.trace[i, 1] += x₀[i]
                MS.trace[i, 2] += x₀₁[i]
                MS.trace[i, 3] += x₁[i]
            end
        end
    else
        # p → p₁ directly (the halfway point is skipped without trace).
        _retarget!(ws, loop.p, loop.p₁)
        _track_middle_segment!(ws, res.ω, res.μ, res.extended_precision_used) ||
            return nothing
    end

    _retarget!(ws, loop.p₁, loop.p₂)
    _track_middle_segment!(ws, tr.state.ω, tr.state.μ, tr.state.extended_prec) ||
        return nothing

    # Final segment back to base: full endgame tracker produces the PathResult.
    _retarget!(ws, loop.p₂, loop.p)
    u = _tracker_input!(ws)
    code = track!(
        ws.tracker, u;
        ω = tr.state.ω, μ = tr.state.μ, extended_precision = tr.state.extended_prec,
    )
    code == EndgameCode.SUCCESS || return nothing
    return _finalize_result(ws, PathResult(ws.tracker))
end

##########################
## Serial solve loop    ##
##########################

# Deduplication tolerances for a finished PathResult under the solver's policy.
# Shared by the serial `add!(MS, …)` and the threaded worker so the two paths
# cannot drift apart.
function _dedup_tolerances(opts::MonodromyOptions, res::PathResult)::Tuple{Float64, Float64}
    rtol = if opts.unique_points_rtol === nothing
        uniqueness_rtol(res)
    else
        opts.unique_points_rtol::Float64
    end
    atol = if opts.unique_points_atol === nothing
        1.0e-14
    else
        opts.unique_points_atol::Float64
    end
    return atol, rtol
end

# Dedup-add a finished PathResult under the solver's tolerance policy.
function add!(MS::MonodromySolver, res::PathResult, id::Int)
    atol, rtol = _dedup_tolerances(MS.options, res)
    return add!(MS.unique_points, solution(res), id; atol = atol, rtol = rtol)
end

"""
    check_start_solutions!(MS, X)

Track every provided start solution at the base parameters (`track_start!`),
deduplicate, and return the resulting `PathResult`s.
"""
function check_start_solutions!(
        MS::MonodromySolver, X::AbstractVector{<:AbstractVector},
    )::Vector{PathResult}
    ws = MS.workers[1]
    results = PathResult[]
    check = MS.options.check_startsolutions
    for x in X
        res = if check
            track_start!(ws, ComplexF64.(x))
        else
            trust_start!(ws, ComplexF64.(x))
        end
        res === nothing && continue
        check && !is_success(res) && continue
        _, added = add!(MS, res, length(results) + 1)
        if added
            push!(results, res)
        end
    end
    return results
end

update_progress!(
    progress::Nothing, stats::MonodromyStatistics;
    queued::Int = 0, solutions::Int = 0, finish::Bool = false,
) = nothing
function update_progress!(
        progress::ProgressMeter.ProgressUnknown,
        stats::MonodromyStatistics;
        queued::Int,
        solutions::Int,
        finish::Bool = false,
    )
    if finish
        ProgressMeter.update!(progress, solutions)
        showvalues = make_showvalues(stats; queued = queued, solutions = solutions)
        ProgressMeter.finish!(progress; showvalues = showvalues)
    else
        # ProgressMeter exposes these forwarded properties as `Float64` at
        # runtime, but their inferred contracts are `Float64` and `Real`.
        # Narrow them here so this optional UI branch does not introduce
        # arithmetic dispatch into monodromy's compiler graph.
        tlast = progress.tlast::Float64
        dt = progress.dt::Float64
        time() > tlast + dt || return nothing
        showvalues = make_showvalues(stats; queued = queued, solutions = solutions)
        ProgressMeter.update!(progress, solutions, showvalues = showvalues)
    end
    return nothing
end

"""
    serial_monodromy_solve!(MS, results, seed, progress)

Run the monodromy loop generation and job-tracking cycle single-threaded.
"""
function serial_monodromy_solve!(
        MS::MonodromySolver,
        results::Vector{PathResult},
        seed::UInt32,
        progress::Union{Nothing, ProgressMeter.ProgressUnknown},
    )::MonodromyCode.T
    queue = LoopTrackingJob[]
    ws = MS.workers[1]
    t₀ = time()
    stats = MS.statistics
    opts = MS.options
    is_subspace = ws.base isa LinearSubspace

    retcode = MonodromyCode.IN_PROGRESS
    while retcode == MonodromyCode.IN_PROGRESS
        loop_finished!(stats, length(results))

        if opts.loop_finished_callback(results)
            retcode = MonodromyCode.TERMINATED_CALLBACK
            break
        end
        if is_subspace && nloops(MS) > 0 && opts.trace_test &&
                trace_colinearity(MS) < opts.trace_test_tol
            retcode = MonodromyCode.SUCCESS
            break
        end
        if opts.target_solutions_count === nothing &&
                length(results) >= something(opts.min_solutions, 0) &&
                loops_no_change(stats, length(results)) >= opts.max_loops_no_progress
            retcode = MonodromyCode.HEURISTIC_STOP
            break
        end
        if length(results) == something(opts.target_solutions_count, -1)
            retcode = MonodromyCode.SUCCESS
            break
        end
        if nloops(MS) > 0 && opts.single_loop_per_start_solution
            retcode = MonodromyCode.SUCCESS
            break
        end

        add_loop!(MS)
        reset_trace!(MS)
        # schedule all jobs on the fresh loop
        new_loop_id = nloops(MS)
        for i in 1:length(results)
            push!(queue, LoopTrackingJob(i, new_loop_id))
        end

        while !isempty(queue)
            job = popfirst!(queue)
            collect_trace = opts.trace_test && nloops(MS) == job.loop_id
            res = track_loop!(
                ws, loop(MS, job.loop_id), results[job.id], collect_trace, MS,
            )
            if res !== nothing
                loop_tracked!(stats)

                # 1) check whether the solution already exists
                id, got_added = add!(MS, res, length(results) + 1)

                if opts.permutations
                    add_permutation!(stats, job.loop_id, job.id, id)
                end

                if got_added
                    # 2) doesn't exist, so add to results
                    push!(results, res)

                    # 3) schedule on the same loop again
                    if !opts.single_loop_per_start_solution
                        push!(queue, LoopTrackingJob(id, job.loop_id))
                    end

                    # 4) schedule on other loops
                    if opts.reuse_loops == ReuseLoops.ALL
                        for k in 1:nloops(MS)
                            k != job.loop_id || continue
                            push!(queue, LoopTrackingJob(id, k))
                        end
                    elseif opts.reuse_loops == ReuseLoops.RANDOM && nloops(MS) >= 2
                        k = rand(2:nloops(MS))
                        if k <= job.loop_id
                            k -= 1
                        end
                        push!(queue, LoopTrackingJob(id, k))
                    end
                end
            else
                loop_failed!(stats)
                if opts.permutations
                    add_permutation!(stats, job.loop_id, job.id, 0)
                end
            end

            update_progress!(
                progress, stats;
                solutions = length(results), queued = length(queue),
            )

            if length(results) == something(opts.target_solutions_count, -1) &&
                    # only terminate after a completed loop to ensure that we
                    # collect proper permutation information
                    !opts.permutations
                retcode = MonodromyCode.SUCCESS
                break
            elseif opts.timeout !== nothing && time() - t₀ > (opts.timeout::Float64)
                retcode = MonodromyCode.TIMEOUT
                break
            end
        end
    end

    update_progress!(
        progress, stats;
        finish = true, solutions = length(results), queued = length(queue),
    )

    return retcode
end

################
## Entrypoint ##
################

function _monodromy_solve!(
        MS::MonodromySolver{H, P},
        X::AbstractVector{<:AbstractVector},
        p::P,
        seed::UInt32;
        show_progress::Bool,
        threading::Bool,
        catch_interrupt::Bool,
        warning::Bool,
    )::MonodromyResult{P, P} where {H, P}
    runner = if threading
        show_progress ?
            _monodromy_threaded_with_progress! :
            _monodromy_threaded_without_progress!
    else
        show_progress ?
            _monodromy_serial_with_progress! :
            _monodromy_serial_without_progress!
    end
    # Threading and progress are invocation policy. Keep all four bodies out of
    # one inferred union so a serial, quiet solve does not compile threaded
    # scheduling or ProgressMeter.
    runner = Base.inferencebarrier(runner)
    return _dispatch_monodromy_policy(
        runner, MS, X, p, seed, catch_interrupt, warning,
    )
end

@noinline function _dispatch_monodromy_policy(
        runner::Function, MS::MonodromySolver{H, P},
        X::AbstractVector{<:AbstractVector}, p::P, seed::UInt32,
        catch_interrupt::Bool, warning::Bool,
    )::MonodromyResult{P, P} where {H, P}
    Base.@nospecialize runner MS X p
    return runner(MS, X, p, seed, catch_interrupt, warning)
end


function _make_monodromy_progress(MS::MonodromySolver)::ProgressMeter.ProgressUnknown
    desc = if MS.options.equivalence_classes
        "Solutions (modulo group action) found:"
    else
        "Solutions found:"
    end
    progress = ProgressMeter.ProgressUnknown(; dt = 0.4, desc = desc, output = stdout)
    progress.tlast += 0.3
    return progress
end

@noinline function _monodromy_serial_without_progress!(
        MS::MonodromySolver{H, P}, X, p::P, seed::UInt32,
        catch_interrupt::Bool, warning::Bool,
    )::MonodromyResult{P, P} where {H, P}
    return _monodromy_solve_body!(
        MS, X, p, seed, nothing, Serial(), catch_interrupt, warning,
    )
end

@noinline function _monodromy_serial_with_progress!(
        MS::MonodromySolver{H, P}, X, p::P, seed::UInt32,
        catch_interrupt::Bool, warning::Bool,
    )::MonodromyResult{P, P} where {H, P}
    return _monodromy_solve_body!(
        MS, X, p, seed, _make_monodromy_progress(MS), Serial(),
        catch_interrupt, warning,
    )
end

@noinline function _monodromy_threaded_without_progress!(
        MS::MonodromySolver{H, P}, X, p::P, seed::UInt32,
        catch_interrupt::Bool, warning::Bool,
    )::MonodromyResult{P, P} where {H, P}
    return _monodromy_solve_body!(
        MS, X, p, seed, nothing, Threaded(), catch_interrupt, warning,
    )
end

@noinline function _monodromy_threaded_with_progress!(
        MS::MonodromySolver{H, P}, X, p::P, seed::UInt32,
        catch_interrupt::Bool, warning::Bool,
    )::MonodromyResult{P, P} where {H, P}
    return _monodromy_solve_body!(
        MS, X, p, seed, _make_monodromy_progress(MS), Threaded(),
        catch_interrupt, warning,
    )
end


function _run_monodromy_loop!(
        ::Serial, MS::MonodromySolver, results::Vector{PathResult},
        seed::UInt32, progress,
    )::MonodromyCode.T
    return serial_monodromy_solve!(MS, results, seed, progress)
end

function _run_monodromy_loop!(
        ::Threaded, MS::MonodromySolver, results::Vector{PathResult},
        seed::UInt32, progress,
    )::MonodromyCode.T
    return threaded_monodromy_solve!(MS, results, seed, progress)
end

function _monodromy_solve_body!(
        MS::MonodromySolver{H, P},
        X::AbstractVector{<:AbstractVector},
        p::P,
        seed::UInt32,
        progress,
        executor::AbstractExecutor,
        catch_interrupt::Bool,
        warning::Bool,
    )::MonodromyResult{P, P} where {H, P}
    MS.statistics = MonodromyStatistics()
    empty!(MS.unique_points)
    reset_trace!(MS)
    reset_loops!(MS)
    results = check_start_solutions!(MS, X)
    retcode = MonodromyCode.IN_PROGRESS
    if isempty(results)
        if warning
            @warn "None of the provided solutions is a valid start solution (Newton's method did not converge)."
        end
        retcode = MonodromyCode.INVALID_STARTVALUE
    else
        try
            retcode = _run_monodromy_loop!(executor, MS, results, seed, progress)
        catch e
            if !catch_interrupt || !(
                    isa(e, InterruptException) ||
                        (isa(e, TaskFailedException) && isa(e.task.exception, InterruptException))
                )
                rethrow(e)
            end
            retcode = MonodromyCode.INTERRUPTED
        end
    end

    return MonodromyResult(
        retcode,
        results,
        p,
        MS.loops,
        MS.statistics,
        MS.options.equivalence_classes,
        seed,
        p isa LinearSubspace ? trace_colinearity(MS) : nothing,
    )
end

"""
    monodromy_solve(F, [sols, p]; options...)

Solve a polynomial system `F(x; p)` with specified parameters and initial
solutions `sols` by monodromy techniques. This makes loops in the parameter
space of `F` to find new solutions. If the parameters occur only *linearly* in
`F`, a start pair `(x₀, p₀)` can be computed automatically; in this case `sols`
and `p` can be omitted and the generated parameters can be obtained with
[`parameters`](@ref) from the [`MonodromyResult`](@ref).

    monodromy_solve(F, [sols, L]; dim, codim, intrinsic = nothing, options...)

Solve the system `[F(x); L(x)] = 0` where `L` is a [`LinearSubspace`](@ref).
If `sols` and `L` are not provided it is necessary to provide `dim` or `codim`,
the expected (co)dimension of a component of `V(F)`. See also
[`linear_subspace_homotopy`](@ref) for the `intrinsic` option.

## Options

* `catch_interrupt = true`: If true catches interruptions (e.g. issued by
  pressing Ctrl-C) and returns the partial result.
* `check_startsolutions = true`: If `true`, track each entry of `sols` at the
  base parameters and sort out any that fail to converge. If `false`, the
  provided solutions are refined but trusted (non-converged points are kept).
* `distance = InfNorm()`: The distance function used for [`UniquePoints`](@ref).
* `loop_finished_callback = always_false`: A callback called with all current
  [`PathResult`](@ref)s after a loop is exhausted. Return `true` to stop.
* `equivalence_classes = true`: Only applies with group actions: consider two
  solutions equivalent when one maps to the other under the actions, and only
  track one solution per equivalence class.
* `group_action = nothing`: A function taking one solution and returning other
  solutions obtainable constructively (e.g. by symmetry).
* `group_actions = nothing`: Several group actions chained via
  [`GroupActions`](@ref).
* `max_loops_no_progress = 5`: Maximal number of loop generations without any
  progress.
* `min_solutions`: Minimal number of solutions before a stopping heuristic
  applies.
* `parameter_sampler = independent_normal`: A function taking the parameter `p`
  and returning a new random parameter `q`.
* `permutations = false`: Whether to keep track of the permutations induced by
  the loops.
* `reuse_loops = :all`: Strategy to reuse other loops for newly found
  solutions: `:all`, `:random` or `:none`.
* `seed`: Seed for the random number generator.
* `target_solutions_count`: Stop once this number of solutions is reached.
* `threading = Threads.nthreads() > 1`: Enable multithreaded path tracking.
* `timeout`: Maximal number of seconds the computation is allowed to run.
* `trace_test = true`: Perform a trace test to check completeness (only for
  linear-subspace monodromy).
* `trace_test_tol = 1e-6`: Tolerance for the trace test.
* `unique_points_atol` / `unique_points_rtol`: tolerances for the solution
  deduplication.
"""
function monodromy_solve(
        F::System,
        args...;
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        catch_interrupt::Bool = true,
        dim::Union{Nothing, Int} = nothing,
        codim::Union{Nothing, Int} = nothing,
        intrinsic::Union{Nothing, Bool} = nothing,
        warning::Bool = true,
        # monodromy options
        check_startsolutions::Bool = true,
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        loop_finished_callback = always_false,
        parameter_sampler = independent_normal,
        equivalence_classes::Union{Nothing, Bool} = nothing,
        trace_test::Bool = true,
        trace_test_tol::Float64 = 1.0e-6,
        target_solutions_count::Union{Nothing, Int} = nothing,
        timeout::Union{Nothing, Real} = nothing,
        min_solutions::Union{Nothing, Int} = nothing,
        max_loops_no_progress::Int = 5,
        reuse_loops::Union{Symbol, ReuseLoops.T} = ReuseLoops.ALL,
        permutations::Bool = false,
        distance = InfNorm(),
        triangle_inequality::Union{Nothing, Bool} = nothing,
        unique_points_atol::Union{Nothing, Float64} = nothing,
        unique_points_rtol::Union{Nothing, Float64} = nothing,
        single_loop_per_start_solution::Bool = false,
    )::MonodromyResult
    if group_actions !== nothing && !(group_actions isa GroupActions)
        group_actions = GroupActions(group_actions)
    end
    options = MonodromyOptions(;
        check_startsolutions = check_startsolutions,
        group_actions = group_actions,
        loop_finished_callback = loop_finished_callback,
        parameter_sampler = parameter_sampler,
        equivalence_classes = something(equivalence_classes, group_actions !== nothing),
        trace_test = trace_test,
        trace_test_tol = trace_test_tol,
        target_solutions_count = target_solutions_count,
        timeout = timeout,
        min_solutions = min_solutions,
        max_loops_no_progress = max_loops_no_progress,
        reuse_loops = reuse_loops,
        permutations = permutations,
        distance = distance,
        triangle_inequality = triangle_inequality,
        unique_points_atol = unique_points_atol,
        unique_points_rtol = unique_points_rtol,
        single_loop_per_start_solution = single_loop_per_start_solution,
    )

    Random.seed!(seed)

    local S, p
    if length(args) == 0
        start_pair = find_start_pair(F)
        if start_pair === nothing
            error(
                "Cannot compute a start pair (x, p) using `find_start_pair(F)`." *
                    " You need to explicitly pass a start pair.",
            )
        end
        x, p0 = start_pair
        S = [x]
        if p0 === nothing
            # No parameters: intersect with a linear subspace. We need the
            # intended (co)dimension; we don't guess `dim` to catch the case
            # that the user just forgot to pass parameters.
            if dim === nothing && codim === nothing
                error(
                    "Given system doesn't have any parameters. If you intended to intersect " *
                        "with a linear subspace it is necessary to provide a " *
                        "dimension (`dim`) or codimension (`codim`) of the component of interest.",
                )
            end
            projective = is_homogeneous(F)
            codim_c = codim === nothing ? nothing : codim + Int(projective)
            # NOTE the swap: the `dim`/`codim` kwargs are COMPONENT dimensions;
            # the subspace has complementary dimensions.
            p = rand_subspace(x; dim = codim_c, codim = dim, affine = !projective)
        else
            p = p0
        end
    elseif length(args) == 2
        sols, p_arg = args
        S = sols isa AbstractVector{<:Number} ? [sols] : sols
        p = p_arg
    else
        throw(ArgumentError("Expected `monodromy_solve(F)` or `monodromy_solve(F, sols, p)`."))
    end

    if p isa LinearSubspace
        cp = convert(LinearSubspace{ComplexF64}, p)
        MS = MonodromySolver(
            F, cp;
            options = options, tracker_options = tracker_options,
            intrinsic = intrinsic,
        )
        mH, nH = size(MS.workers[1].homotopy)
        if mH < nH
            throw(
                ArgumentError(
                    "The homotopy for the subspace intersection is underdetermined " *
                        "($mH equations for $nH unknowns). The provided component dimension " *
                        "(dim = $dim, codim = $codim) is likely overstated for this system.",
                ),
            )
        end
        return _monodromy_solve!(
            MS, S, cp, seed;
            show_progress = show_progress, threading = threading,
            catch_interrupt = catch_interrupt, warning = warning,
        )
    else
        cp = convert(Vector{ComplexF64}, p)
        MS = MonodromySolver(
            F, cp;
            options = options, tracker_options = tracker_options,
        )
        return _monodromy_solve!(
            MS, S, cp, seed;
            show_progress = show_progress, threading = threading,
            catch_interrupt = catch_interrupt, warning = warning,
        )
    end
end

function monodromy_solve(
        F::AbstractVector{<:MP.AbstractPolynomialLike},
        args...;
        parameters = nothing,
        variables = nothing,
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        catch_interrupt::Bool = true,
        dim::Union{Nothing, Int} = nothing,
        codim::Union{Nothing, Int} = nothing,
        intrinsic::Union{Nothing, Bool} = nothing,
        warning::Bool = true,
        check_startsolutions::Bool = true,
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        loop_finished_callback = always_false,
        parameter_sampler = independent_normal,
        equivalence_classes::Union{Nothing, Bool} = nothing,
        trace_test::Bool = true,
        trace_test_tol::Float64 = 1.0e-6,
        target_solutions_count::Union{Nothing, Int} = nothing,
        timeout::Union{Nothing, Real} = nothing,
        min_solutions::Union{Nothing, Int} = nothing,
        max_loops_no_progress::Int = 5,
        reuse_loops::Union{Symbol, ReuseLoops.T} = ReuseLoops.ALL,
        permutations::Bool = false,
        distance = InfNorm(),
        triangle_inequality::Union{Nothing, Bool} = nothing,
        unique_points_atol::Union{Nothing, Float64} = nothing,
        unique_points_rtol::Union{Nothing, Float64} = nothing,
        single_loop_per_start_solution::Bool = false,
    )::MonodromyResult
    sys = System(F; parameters = parameters, variables = variables)
    return monodromy_solve(
        sys, args...;
        seed = seed,
        tracker_options = tracker_options,
        show_progress = show_progress,
        threading = threading,
        catch_interrupt = catch_interrupt,
        dim = dim,
        codim = codim,
        intrinsic = intrinsic,
        warning = warning,
        check_startsolutions = check_startsolutions,
        group_actions = group_actions,
        loop_finished_callback = loop_finished_callback,
        parameter_sampler = parameter_sampler,
        equivalence_classes = equivalence_classes,
        trace_test = trace_test,
        trace_test_tol = trace_test_tol,
        target_solutions_count = target_solutions_count,
        timeout = timeout,
        min_solutions = min_solutions,
        max_loops_no_progress = max_loops_no_progress,
        reuse_loops = reuse_loops,
        permutations = permutations,
        distance = distance,
        triangle_inequality = triangle_inequality,
        unique_points_atol = unique_points_atol,
        unique_points_rtol = unique_points_rtol,
        single_loop_per_start_solution = single_loop_per_start_solution,
    )
end

"""
    threaded_monodromy_solve!(MS, results, seed, progress)

Multithreaded variant of [`serial_monodromy_solve!`](@ref): one long-lived
worker task per thread consuming a job channel, plus a coordinator task that
generates loop generations and waits for quiescence (all workers idle,
channel empty, no job in flight).
"""
function threaded_monodromy_solve!(
        MS::MonodromySolver,
        results::Vector{PathResult},
        seed::UInt32,
        progress::Union{Nothing, ProgressMeter.ProgressUnknown},
    )::MonodromyCode.T
    queue = Channel{LoopTrackingJob}(Inf)

    # Grow the worker states to one per thread via the builder (never deepcopy).
    nthr = Threads.nthreads()
    while length(MS.workers) < nthr
        push!(MS.workers, MS.builder())
    end

    data_lock = MS.unique_points_lock
    t0 = time()
    retcode = Ref(MonodromyCode.IN_PROGRESS)
    stats = MS.statistics
    opts = MS.options
    is_subspace = MS.workers[1].base isa LinearSubspace
    notify_lock = ReentrantLock()
    cond_queue_emptied = Threads.Condition(notify_lock)
    workers_idle = fill(true, nthr)
    interrupted = Ref(false)
    queued = Ref(0)
    inflight_count = Threads.Atomic{Int}(0)  # in-flight counter to check termination
    # Number of jobs sitting in the channel, tracked manually because the
    # channel's own count (Base.n_avail) is not public API. Incremented before
    # every push!, decremented when a worker receives a job.
    queued_count = Threads.Atomic{Int}(0)
    # Torn-read-free mirror of length(results). `results` is mutated only under
    # data_lock, but the progress and heuristic checks below read the count
    # without holding it; reading this atomic instead avoids racing with push!.
    n_results = Threads.Atomic{Int}(length(results))
    enqueue! = job -> begin
        Threads.atomic_add!(queued_count, 1)
        return push!(queue, job)
    end

    # Quiescence is decided by two checks: all(workers_idle) and
    # inflight_count[] == 0. One is probably enough; it is safe to have both.
    try
        for tid in 1:nthr
            let ws = MS.workers[tid], tid = tid
                Threads.@spawn begin
                    for job in queue
                        Threads.atomic_add!(queued_count, -1)
                        stop_queue = false
                        Base.@lock notify_lock begin
                            if interrupted[]
                                workers_idle[tid] = true
                                if all(workers_idle)
                                    notify(cond_queue_emptied)
                                end
                                stop_queue = true
                            else
                                workers_idle[tid] = false
                            end
                        end
                        if stop_queue
                            break
                        end
                        Threads.atomic_add!(inflight_count, 1)

                        try
                            start_res = Base.@lock data_lock results[job.id]
                            collect_trace = opts.trace_test && nloops(MS) == job.loop_id
                            res = track_loop!(
                                ws, loop(MS, job.loop_id), start_res, collect_trace, MS,
                            )

                            if res !== nothing
                                loop_tracked!(stats)

                                # 1) check whether the solution already exists
                                lock(data_lock)
                                got_added = false
                                id = 0
                                if length(results) <
                                        something(opts.target_solutions_count, typemax(Int))
                                    atol, rtol = _dedup_tolerances(opts, res)
                                    id, got_added = add!(
                                        MS.unique_points, solution(res),
                                        length(results) + 1;
                                        atol = atol, rtol = rtol,
                                    )
                                    if opts.permutations
                                        add_permutation!(stats, job.loop_id, job.id, id)
                                    end
                                end
                                if got_added
                                    # 2) doesn't exist, so add to results
                                    push!(results, res)
                                    Threads.atomic_add!(n_results, 1)
                                    unlock(data_lock)

                                    # 3) schedule on the same loop again
                                    if !opts.single_loop_per_start_solution
                                        enqueue!(LoopTrackingJob(id, job.loop_id))
                                    end
                                    # 4) schedule on other loops
                                    if opts.reuse_loops == ReuseLoops.ALL
                                        for k in 1:nloops(MS)
                                            k != job.loop_id || continue
                                            enqueue!(LoopTrackingJob(id, k))
                                        end
                                    elseif opts.reuse_loops == ReuseLoops.RANDOM &&
                                            nloops(MS) >= 2
                                        k = rand(2:nloops(MS))
                                        if k <= job.loop_id
                                            k -= 1
                                        end
                                        enqueue!(LoopTrackingJob(id, k))
                                    end
                                else
                                    unlock(data_lock)
                                end
                            else
                                loop_failed!(stats)
                                if opts.permutations
                                    Base.@lock data_lock add_permutation!(
                                        stats, job.loop_id, job.id, 0,
                                    )
                                end
                            end

                            update_progress!(
                                progress, stats;
                                solutions = n_results[],
                                queued = max(queued_count[], 0),
                            )

                            # mark worker as idle
                            Base.@lock notify_lock begin
                                workers_idle[tid] = true
                                # if the queue is empty, check whether all
                                # others are also waiting
                                if !isready(queue) && all(workers_idle)
                                    notify(cond_queue_emptied)
                                end
                            end

                            if n_results[] >=
                                    something(opts.target_solutions_count, typemax(Int)) &&
                                    # only terminate after a completed loop to ensure
                                    # that we collect proper permutation information
                                    !opts.permutations
                                retcode[] = MonodromyCode.SUCCESS
                                Base.@lock notify_lock begin
                                    interrupted[] = true
                                end
                            elseif opts.timeout !== nothing &&
                                    time() - t0 > (opts.timeout::Float64)
                                retcode[] = MonodromyCode.TIMEOUT
                                Base.@lock notify_lock begin
                                    interrupted[] = true
                                end
                            end
                        finally
                            Threads.atomic_add!(inflight_count, -1)
                            Base.@lock notify_lock begin
                                if !isready(queue) && inflight_count[] == 0
                                    notify(cond_queue_emptied)
                                end
                            end
                        end
                    end
                end
            end
        end

        t = Threads.@spawn begin
            Base.@lock notify_lock begin
                while true
                    if interrupted[]
                        break
                    end
                    loop_finished!(stats, n_results[])

                    if opts.loop_finished_callback(results)
                        retcode[] = MonodromyCode.TERMINATED_CALLBACK
                        break
                    end

                    if opts.target_solutions_count === nothing &&
                            n_results[] >= something(opts.min_solutions, 0) &&
                            loops_no_change(stats, n_results[]) >=
                            opts.max_loops_no_progress
                        retcode[] = MonodromyCode.HEURISTIC_STOP
                        break
                    end

                    if n_results[] >=
                            something(opts.target_solutions_count, typemax(Int))
                        retcode[] = MonodromyCode.SUCCESS
                        break
                    end

                    if is_subspace && nloops(MS) > 0 && opts.trace_test &&
                            trace_colinearity(MS) < opts.trace_test_tol
                        retcode[] = MonodromyCode.SUCCESS
                        break
                    end

                    add_loop!(MS)
                    reset_trace!(MS)
                    # schedule all jobs
                    new_loop_id = nloops(MS)
                    for i in 1:length(results)
                        enqueue!(LoopTrackingJob(i, new_loop_id))
                    end

                    wait(cond_queue_emptied)
                    if opts.single_loop_per_start_solution
                        retcode[] = MonodromyCode.SUCCESS
                        break
                    end
                    retcode[] == MonodromyCode.IN_PROGRESS || break
                end
            end
        end

        wait(t)
    catch e
        close(queue)
        interrupted[] = true
        rethrow(e)
    finally
        queued[] = max(queued_count[], 0)
        close(queue)
    end

    update_progress!(
        progress, stats;
        finish = true, solutions = length(results), queued = queued[],
    )

    return retcode[]
end

"""
    solve(F::System, R::MonodromyResult; target_parameters, options...)

Track the solutions of the monodromy result `R` from its parameters to
`target_parameters` via a parameter homotopy.
"""
function solve(
        F::System,
        R::MonodromyResult,
        exec::AbstractExecutor = Threaded();
        target_parameters::AbstractVector{<:Number},
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::Result
    return solve(
        F, solutions(R), exec;
        start_parameters = parameters(R),
        target_parameters = target_parameters,
        seed = seed,
        tracker_options = tracker_options,
        endgame_options = endgame_options,
        show_progress = show_progress,
    )
end

## ── verify_solution_completeness ─────────────────────────────────────────────
# Trace test (del Campo & Rodriguez 2017; Leykin, Rodriguez & Sottile 2018).
# Cold diagnostic path.

# Parameter sampler for the auxiliary monodromy computation: keeps the first
# parameter (the trace variable t) fixed at zero. The LinearSubspace method
# exists only to keep the sampler total over both MonodromyLoop branches; the
# auxiliary system always has vector parameters, so it is unreachable.
function _zero_first_parameter_sampler(pp::AbstractVector)::Vector{ComplexF64}
    return [0.0 + 0.0im; randn(ComplexF64, length(pp) - 1)]
end
function _zero_first_parameter_sampler(::LinearSubspace)
    throw(ArgumentError("the completeness verification sampler only supports vector parameters"))
end

"""
    verify_solution_completeness(F::System, R::MonodromyResult; kwargs...)
    verify_solution_completeness(F::System, sols, p; trace_tol = 1e-14, kwargs...)

Verify that a monodromy computation found all solutions of the polynomial
system `F(x; p) = 0` on the fiber over the parameters `p` using the trace
test. The correctness of this verification procedure requires that the
parametrized family is irreducible and that the given solutions are correct.

The algorithm constructs the augmented system
`[F(x, p + λv); (Σᵢ aᵢxᵢ - 1)λ + t]` in the variables `[x; λ]` with
parameters `[t; p; v; a]`, computes additional witnesses on the `λ ≠ 0`
component via monodromy (with the first parameter `t` fixed to zero), and
then performs two parameter homotopies moving `t` along a random complex
direction. The combined witness set is complete if and only if the traces of
the three witness sets are colinear; the deviation from colinearity is
measured by the relative third singular value of the trace matrix and
compared against `trace_tol`.

Returns `true` (complete), `false` (incomplete), or `nothing` when a
parameter homotopy lost solutions so no verdict is possible.

## Options

* `trace_tol = 1e-14`: tolerance for the trace colinearity test.
* `show_progress = true`: print progress information.
* `seed`: seed for the auxiliary monodromy computation.
* `threading`: use multiple threads for the auxiliary monodromy computation.
* `tracker_options`, `endgame_options`: forwarded to the auxiliary solves.
* `max_loops_no_progress = 5`: forwarded to the auxiliary monodromy solve.
"""
function verify_solution_completeness(
        F::System,
        mres::MonodromyResult;
        trace_tol::Float64 = 1.0e-14,
        show_progress::Bool = true,
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        max_loops_no_progress::Int = 5,
    )::Union{Nothing, Bool}
    return verify_solution_completeness(
        F, solutions(mres), Vector(parameters(mres));
        trace_tol = trace_tol,
        show_progress = show_progress,
        seed = seed,
        threading = threading,
        tracker_options = tracker_options,
        endgame_options = endgame_options,
        max_loops_no_progress = max_loops_no_progress,
    )
end

function verify_solution_completeness(
        F::System,
        sols::AbstractVector{<:AbstractVector},
        q::AbstractVector;
        trace_tol::Float64 = 1.0e-14,
        show_progress::Bool = true,
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        max_loops_no_progress::Int = 5,
    )::Union{Nothing, Bool}
    n = nvariables(F)
    m = nparameters(F)
    # DynamicPolynomials variables are identity distinct, so these fresh
    # variables cannot collide with user variables of the same name.
    @polyvar t v[1:m] a[1:n] λ

    x = collect(F.variables)
    p = collect(F.parameters)

    verify_system = System(
        [
            [MP.subs(f, p => p .+ λ .* v) for f in F.polys];
            (sum(a .* x) - 1) * λ + t
        ];
        variables = [x; λ],
        parameters = [t; p; v; a],
    )

    # Monodromy computation for the additional witnesses: use verify_system
    # but enforce t = 0 and start with λ ≠ 0 so we stay on a different
    # irreducible component.
    if show_progress
        @info "Compute additional witnesses for completeness check..."
    end

    Random.seed!(seed)
    q0 = convert(Vector{ComplexF64}, q)

    # Start solutions: sample random parameters qq to set v = qq - q. More
    # than one start solution is good; construct up to n by a parameter
    # homotopy to qq, then compute an `a` such that those solutions lie on
    # the linear space a⋅x - 1 = 0.
    qq = randn(ComplexF64, m)
    qq_res = solve(
        F, sols[1:min(n, length(sols))], Serial();
        start_parameters = q0,
        target_parameters = qq,
        seed = seed,
        tracker_options = tracker_options,
        endgame_options = endgame_options,
        show_progress = show_progress,
    )
    a0 = reduce(vcat, transpose.(solutions(qq_res))) \ ones(nsolutions(qq_res))
    Y = map(s -> [s; 1], solutions(qq_res))
    base_params = [q0; qq .- q0; a0]

    additional_mres = monodromy_solve(
        verify_system, Y, [0.0; base_params];
        parameter_sampler = _zero_first_parameter_sampler,
        seed = seed,
        threading = threading,
        show_progress = show_progress,
        tracker_options = tracker_options,
        max_loops_no_progress = max_loops_no_progress,
    )
    additional_sols = solutions(additional_mres)
    if show_progress
        @info additional_mres
        @info "Computed $(length(additional_sols)) additional witnesses"
        @info "Compute trace using two parameter homotopies..."
    end

    # Parameter homotopies for the trace: move t along a random direction γ.
    S = [map(s -> [s; 0], sols); additional_sols]
    γ = randn(ComplexF64)
    res1 = solve(
        verify_system, S, Serial();
        start_parameters = [0.0; base_params],
        target_parameters = [0.5 * γ; base_params],
        seed = seed,
        tracker_options = tracker_options,
        endgame_options = endgame_options,
        show_progress = show_progress,
    )
    S1 = solutions(res1)
    if length(S1) != length(S)
        if show_progress
            @warn "Lost solution during parameter homotopy. Abort."
        end
        return nothing
    end

    res2 = solve(
        verify_system, S1, Serial();
        start_parameters = [0.5 * γ; base_params],
        target_parameters = [1.0 * γ; base_params],
        seed = seed,
        tracker_options = tracker_options,
        endgame_options = endgame_options,
        show_progress = show_progress,
    )
    S2 = solutions(res2)
    if length(S2) != length(S)
        if show_progress
            @warn "Lost solution during parameter homotopy. Abort."
        end
        return nothing
    end

    T = sum(S)
    T1 = sum(S1)
    T2 = sum(S2)

    M = [T T1 T2; 1 1 1]
    singvals = LA.svdvals(M)
    trace_norm = singvals[3] / singvals[1]

    if show_progress
        @info "Norm of trace: $trace_norm"
    end

    return trace_norm < trace_tol
end
