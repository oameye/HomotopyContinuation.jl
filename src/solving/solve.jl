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
    m, n = size(F.evaluator)
    m >= n || throw(
        ArgumentError(
            "The system has $m equation(s) in $n variables. The solution set is " *
                "positive-dimensional; only square or overdetermined systems with " *
                "finitely many solutions are supported.",
        ),
    )
    return nothing
end

# ── CommonSolve.init: System + TotalDegree ────────────────────────────────

function CommonSolve.init(
        F::System, alg::TotalDegree,
        exec::AbstractExecutor = Threaded();
        show_progress::Bool = true,
    )::SolveCache
    seed = alg.seed
    _check_square_or_overdetermined(F)
    m, n = size(F.evaluator)

    rng = Random.MersenneTwister(seed)
    γ = cis(2π * rand(rng))

    if m > n
        A, perm, excess_checker = _square_up(rng, F)
        target_evaluator = _randomized_evaluator(F.evaluator, A, perm)
        degrees = F.degrees[perm[1:n]]
        builder = RandomizedStraightLineBuilder(
            degrees, F, A, perm, γ, alg.tracker_options, alg.endgame_options,
        )
    else
        target_evaluator = F.evaluator
        degrees = F.degrees
        excess_checker = nothing
        builder = StraightLineBuilder(
            F.degrees, F, γ, alg.tracker_options, alg.endgame_options,
        )
    end

    start_evaluator = _total_degree_startevaluator(degrees)
    starts = _total_degree_solutions(degrees)

    H = StraightLineHomotopy(start_evaluator, target_evaluator; γ = γ)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = alg.tracker_options)
    eg = EndgameTracker(tracker, alg.endgame_options)

    return SolveCache(exec, builder, eg, starts, seed, excess_checker, show_progress)
end

# ── CommonSolve.solve!: serial ─────────────────────────────────────────────

function CommonSolve.solve!(cache::SolveCache{Serial})::Result
    eg = cache.tracker
    n_paths = length(cache.start_solutions)
    path_results = PathResult[]
    sizehint!(path_results, n_paths)

    progress = make_progress(n_paths, cache.show_progress)
    stats = ProgressStats()
    for (k, x₀) in enumerate(cache.start_solutions)
        track!(eg, x₀)
        pr = PathResult(eg; path_number = k, start_solution = Vector{ComplexF64}(x₀))
        push!(path_results, pr)
        update_progress!(progress, k, stats, pr)
    end

    return _finalize_result(
        path_results, n_paths, cache.seed, cache.excess_checker,
    )
end

# ── CommonSolve.solve!: threaded ───────────────────────────────────────────

function CommonSolve.solve!(cache::SolveCache{Threaded})::Result
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    n_paths = length(starts)
    results = Vector{PathResult}(undef, n_paths)

    progress = make_progress(n_paths, cache.show_progress)
    stats = ProgressStats()
    counter = Threads.Atomic{Int}(0)
    plock = ReentrantLock()

    @tasks for i in eachindex(starts)
        @set ntasks = nt
        @local ws = cache.builder()
        track!(ws.tracker, starts[i])
        results[i] = PathResult(ws.tracker; path_number = i, start_solution = Vector{ComplexF64}(starts[i]))
        if progress !== nothing
            k = Threads.atomic_add!(counter, 1) + 1
            @lock plock update_progress!(progress, k, stats, results[i])
        end
    end

    return _finalize_result(results, n_paths, cache.seed, cache.excess_checker)
end

# ── Convenience: solve(F, alg, exec) ─────────────────────────────────────

"""
    solve(F::System, alg=TotalDegree(), exec=Threaded())

Solve a polynomial system using homotopy continuation.
"""
function solve(
        F::System,
        alg::TotalDegree = TotalDegree(),
        exec::AbstractExecutor = Threaded();
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg, exec; show_progress = show_progress))
end

function solve(F::System, exec::AbstractExecutor; show_progress::Bool = true)::Result
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

# ── Parameter homotopy ─────────────────────────────────────────────────────

function solve(
        F::System,
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
        F::System,
        starts::AbstractVector{<:AbstractVector{<:Number}},
        exec::AbstractExecutor = Threaded();
        start_parameters::AbstractVector{<:Number},
        target_parameters::AbstractVector{<:Number},
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::SolveCache
    _check_square_or_overdetermined(F)
    @assert nparameters(F) > 0 "System must have parameters for parameter homotopy"
    @assert length(start_parameters) == nparameters(F) "start_parameters length must match nparameters"
    @assert length(target_parameters) == nparameters(F) "target_parameters length must match nparameters"

    sp = ComplexF64.(start_parameters)
    tp = ComplexF64.(target_parameters)
    H = CoefficientHomotopy(F.evaluator, sp, tp)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = tracker_options)
    eg = EndgameTracker(tracker, endgame_options)

    builder = CoefficientBuilder(F, sp, tp, tracker_options, endgame_options)

    start_sols = [Vector{ComplexF64}(ComplexF64.(s)) for s in starts]

    return SolveCache(exec, builder, eg, start_sols, seed, nothing, show_progress)
end
