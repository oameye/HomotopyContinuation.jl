## solve — CommonSolve.jl integration for polynomial system solving.
#
# Pattern: solve(F, alg, exec) = solve!(init(F, alg, exec))

struct SolveCache{E <: AbstractExecutor, B, C}
    executor::E
    builder::B
    tracker::EndgameTracker
    start_solutions::Vector{Vector{ComplexF64}}
    seed::UInt32
    # ExcessSolutionChecker for overdetermined systems, Nothing for square ones
    excess_checker::C
    show_progress::Bool
end

"""
    _check_square_or_overdetermined(F)

Throw for underdetermined input (positive-dimensional solution set).
"""
function _check_square_or_overdetermined(F::System)::Nothing
    return _check_square_or_overdetermined(system_shape(F), F)
end

_check_square_or_overdetermined(::SquareShape, ::System)::Nothing = nothing
_check_square_or_overdetermined(::OverdeterminedShape, ::System)::Nothing = nothing
function _check_square_or_overdetermined(::UnderdeterminedShape, F::System)::Nothing
    m, n = size(F.evaluator)
    throw(
        ArgumentError(
            "The system has $m equation(s) in $n variables. The solution set is " *
                "positive-dimensional; only square or overdetermined systems with " *
                "finitely many solutions are supported.",
        ),
    )
end

# A composition carries no shape parameter, so the same check reads its size.
function _check_square_or_overdetermined(C::CompositionSystem)::Nothing
    m, n = size(C)
    m >= n || throw(
        ArgumentError(
            "The composition has $m equation(s) in $n variables. The solution set is " *
                "positive-dimensional; only square or overdetermined systems with " *
                "finitely many solutions are supported.",
        ),
    )
    return nothing
end

"""
    _check_parameter_free(F, route)

Throw when `F` still has parameters. `route` names the algorithm in the message.
"""
function _check_parameter_free(F::SystemLike, route::String)::Nothing
    np = nparameters(F)
    np == 0 || throw(
        ArgumentError(
            "$route requires a parameter-free system, but the system has $np " *
                "parameter(s). Substitute their values first, or track from known " *
                "start solutions with a parameter homotopy " *
                "(`solve(F, starts; start_parameters, target_parameters)`).",
        ),
    )
    return nothing
end

"""
    _check_polynomial(F, route)

Throw when `F` has an equation that is not polynomial in its variables. `System`
records those with a degree of `-1`.
"""
function _check_polynomial(F::System, route::String)::Nothing
    any(<(0), F.degrees) || return nothing
    throw(
        ArgumentError(
            "$route requires a system that is polynomial in its variables, but at " *
                "least one equation uses division by a variable, a negative power, " *
                "or a unary function of a variable. Clear denominators first, or " *
                "track from known start solutions with a parameter homotopy or " *
                "`monodromy_solve`.",
        ),
    )
end

# A composed degree of `-1` comes from a stage, so the message names one.
function _check_polynomial(C::CompositionSystem, route::String)::Nothing
    any(<(0), degrees(C)) || return nothing
    throw(
        ArgumentError(
            "$route requires a composition whose stages are polynomial in their " *
                "variables, but at least one stage equation uses division by a " *
                "variable, a negative power, or a unary function of a variable. " *
                "Clear denominators first, or track from known start solutions " *
                "with a parameter homotopy or `monodromy_solve`.",
        ),
    )
end

# ── CommonSolve.init: System + TotalDegree ────────────────────────────────

function _total_degree_solve_cache(
        exec::E,
        builder::B,
        target_evaluator::SystemEvaluator,
        degrees::Vector{Int},
        seed::UInt32,
        excess_checker::C,
        show_progress::Bool,
        tracker_options::TrackerOptions,
        endgame_options::EndgameOptions,
        γ::ComplexF64,
    )::SolveCache{E, B, C} where {E <: AbstractExecutor, B, C}
    start_evaluator = _total_degree_startevaluator(degrees)
    starts = _total_degree_solutions(degrees)

    H = StraightLineHomotopy(start_evaluator, target_evaluator; γ = γ)
    eg = _endgame_tracker(H, tracker_options, endgame_options)

    return SolveCache(exec, builder, eg, starts, seed, excess_checker, show_progress)
end

function CommonSolve.init(
        F::SystemLike, alg::TotalDegree,
        exec::AbstractExecutor = Threaded();
        show_progress::Bool = true,
    )::SolveCache
    seed = alg.seed
    _check_square_or_overdetermined(F)
    _check_parameter_free(F, "`TotalDegree`")
    _check_polynomial(F, "`TotalDegree`")

    rng = Random.MersenneTwister(seed)
    γ = _random_gamma(rng)
    # Dynamic call: specializes the body on the concrete `System` so that
    # `system_shape(F)` resolves statically instead of union-splitting.
    initializer = Base.inferencebarrier(_init_total_degree_shaped)
    return initializer(F, alg, exec, rng, γ, show_progress)
end

_init_total_degree_shaped(
    F::SystemLike, alg::TotalDegree, exec::AbstractExecutor,
    rng::Random.MersenneTwister, γ::ComplexF64, show_progress::Bool,
) = _init_total_degree(system_shape(F), F, alg, exec, rng, γ, show_progress)

function _init_total_degree(
        ::SquareShape, F::SystemLike, alg::TotalDegree, exec::AbstractExecutor,
        ::Random.MersenneTwister, γ::ComplexF64, show_progress::Bool,
    )
    degrees = F.degrees
    builder = StraightLineBuilder(
        degrees, F, γ, alg.tracker_options, alg.endgame_options,
    )
    return _total_degree_solve_cache(
        exec, builder, F.evaluator, degrees, alg.seed, nothing,
        show_progress, alg.tracker_options, alg.endgame_options, γ,
    )
end

function _init_total_degree(
        ::OverdeterminedShape, F::SystemLike, alg::TotalDegree,
        exec::AbstractExecutor, rng::Random.MersenneTwister,
        γ::ComplexF64, show_progress::Bool,
    )
    n = nvariables(F)
    A, perm, excess_checker = _square_up(rng, F)
    target_evaluator = _randomized_evaluator(F.evaluator, A, perm)
    degrees = F.degrees[perm[1:n]]
    builder = RandomizedStraightLineBuilder(
        degrees, F, A, perm, γ, alg.tracker_options, alg.endgame_options,
    )
    return _total_degree_solve_cache(
        exec, builder, target_evaluator, degrees, alg.seed, excess_checker,
        show_progress, alg.tracker_options, alg.endgame_options, γ,
    )
end

# ── CommonSolve.solve!: serial ─────────────────────────────────────────────

function CommonSolve.solve!(cache::SolveCache{Serial})::Result
    solver = cache.show_progress ?
        _solve_total_degree_serial_with_progress :
        _solve_total_degree_serial_without_progress
    solver = Base.inferencebarrier(solver)
    return _dispatch_solve_policy(solver, cache)
end

@noinline function _dispatch_solve_policy(solver::Function, cache)::Result
    Base.@nospecialize solver cache
    return solver(cache)
end

@noinline _solve_total_degree_serial_without_progress(cache::SolveCache{Serial}) =
    _solve_total_degree_serial(cache, nothing)
@noinline _solve_total_degree_serial_with_progress(cache::SolveCache{Serial}) =
    _solve_total_degree_serial(cache, make_progress(length(cache.start_solutions), true))

# `start_solution` is copied: the caller's `starts` vector outlives the result.
function _track_path!(
        eg::EndgameTracker, x₀::Vector{ComplexF64}, k::Int,
    )::PathResult
    track!(eg, x₀)
    return PathResult(eg; path_number = k, start_solution = copy(x₀))
end

_track_path!(ws::TrackingWorkerState, x₀::Vector{ComplexF64}, k::Int)::PathResult =
    _track_path!(ws.tracker, x₀, k)

function _solve_total_degree_serial(cache::SolveCache{Serial}, progress)::Result
    eg = cache.tracker
    n_paths = length(cache.start_solutions)
    path_results = PathResult[]
    sizehint!(path_results, n_paths)

    stats = ProgressStats()
    for (k, x₀) in enumerate(cache.start_solutions)
        pr = _track_path!(eg, x₀, k)
        push!(path_results, pr)
        update_progress!(progress, k, stats, pr)
    end

    return _finalize_result(
        path_results, n_paths, cache.seed, cache.excess_checker,
    )
end

# ── CommonSolve.solve!: threaded ───────────────────────────────────────────

function CommonSolve.solve!(cache::SolveCache{Threaded})::Result
    solver = cache.show_progress ?
        _solve_total_degree_threaded_with_progress :
        _solve_total_degree_threaded_without_progress
    solver = Base.inferencebarrier(solver)
    return _dispatch_solve_policy(solver, cache)
end

@noinline _solve_total_degree_threaded_without_progress(cache::SolveCache{Threaded}) =
    _solve_total_degree_threaded(cache, nothing)
@noinline _solve_total_degree_threaded_with_progress(cache::SolveCache{Threaded}) =
    _solve_total_degree_threaded(cache, make_progress(length(cache.start_solutions), true))

function _solve_total_degree_threaded(cache::SolveCache{Threaded}, progress)::Result
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    n_paths = length(starts)
    results = Vector{PathResult}(undef, n_paths)

    stats = ProgressStats()
    counter = Threads.Atomic{Int}(0)
    plock = ReentrantLock()

    @tasks for i in eachindex(starts)
        @set ntasks = nt
        @local ws = cache.builder()
        results[i] = _track_path!(ws, starts[i], i)
        if progress !== nothing
            k = Threads.atomic_add!(counter, 1) + 1
            @lock plock update_progress!(progress, k, stats, results[i])
        end
    end

    return _finalize_result(results, n_paths, cache.seed, cache.excess_checker)
end

# ── CommonSolve.solve!: distributed (extension) ────────────────────────────

CommonSolve.solve!(cache::SolveCache{DistributedExecutor})::Result =
    _distributed_solve!(cache)

# ── Convenience: solve(F, alg, exec) ─────────────────────────────────────

"""
    solve(F::System, alg=TotalDegree(), exec=Threaded())

Solve a polynomial system using homotopy continuation.

A [`CompositionSystem`](@ref) is accepted too: the total-degree start system
needs only the composed degrees, which are folded from the stages, so the
equations are never rebuilt.
"""
function solve(
        F::SystemLike,
        alg::TotalDegree = TotalDegree(),
        exec::AbstractExecutor = Threaded();
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg, exec; show_progress = show_progress))
end

function solve(F::SystemLike, exec::AbstractExecutor; show_progress::Bool = true)::Result
    return solve(F, TotalDegree(), exec; show_progress = show_progress)
end

function solve(
        F::System,
        alg::Polyhedral,
        exec::AbstractExecutor = Threaded();
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg, exec; show_progress = show_progress))
end

# The polyhedral start system is built from the composed monomials, which only
# the substituted equations carry.
function solve(
        C::CompositionSystem,
        alg::Polyhedral,
        exec::AbstractExecutor = Threaded();
        show_progress::Bool = true,
    )::Result
    return solve(System(C), alg, exec; show_progress = show_progress)
end

CommonSolve.init(
    C::CompositionSystem, alg::Polyhedral,
    exec::AbstractExecutor = Threaded();
    show_progress::Bool = true,
) = CommonSolve.init(System(C), alg, exec; show_progress = show_progress)

# ── Parameter homotopy ─────────────────────────────────────────────────────

function solve(
        F::SystemLike,
        starts::AbstractVector{<:AbstractVector{<:Number}},
        exec::AbstractExecutor = Threaded();
        start_parameters::AbstractVector{<:Number},
        target_parameters::AbstractVector{<:Number},
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            F, starts, exec;
            start_parameters = start_parameters,
            target_parameters = target_parameters,
            seed = seed,
            tracker_options = tracker_options,
            endgame_options = endgame_options,
            show_progress = show_progress,
        ),
    )
end

function CommonSolve.init(
        F::SystemLike,
        starts::AbstractVector{<:AbstractVector{<:Number}},
        exec::AbstractExecutor = Threaded();
        start_parameters::AbstractVector{<:Number},
        target_parameters::AbstractVector{<:Number},
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )
    _check_square_or_overdetermined(F)
    @assert nparameters(F) > 0 "System must have parameters for parameter homotopy"
    @assert length(start_parameters) == nparameters(F) "start_parameters length must match nparameters"
    @assert length(target_parameters) == nparameters(F) "target_parameters length must match nparameters"

    sp = ComplexF64.(start_parameters)
    tp = ComplexF64.(target_parameters)
    H = ParameterHomotopy(F.evaluator, sp, tp)
    eg = _endgame_tracker(H, tracker_options, endgame_options)

    builder = ParameterBuilder(F, sp, tp, tracker_options, endgame_options)

    start_sols = [Vector{ComplexF64}(ComplexF64.(s)) for s in starts]

    return SolveCache(exec, builder, eg, start_sols, seed, nothing, show_progress)
end
