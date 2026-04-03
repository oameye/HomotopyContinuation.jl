## solve — CommonSolve.jl integration for polynomial system solving.
#
# Pattern: solve(F, alg, exec) = solve!(init(F, alg, exec))

struct SolveCache{E <: AbstractExecutor, B}
    executor::E
    builder::B
    tracker::EndgameTracker
    start_solutions::Vector{Vector{ComplexF64}}
    seed::UInt32
end

# ── CommonSolve.init: System + TotalDegree ────────────────────────────────

function CommonSolve.init(
        F::System, alg::TotalDegree,
        exec::AbstractExecutor = Threaded(),
    )::SolveCache
    seed = alg.seed

    start_evaluator = _total_degree_startevaluator(F.degrees)
    starts = _total_degree_solutions(F.degrees)

    rng = Random.MersenneTwister(seed)
    γ = cis(2π * rand(rng))
    H = StraightLineHomotopy(start_evaluator, F.evaluator; γ = γ)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = alg.tracker_options)
    eg = EndgameTracker(tracker, alg.endgame_options)

    builder = StraightLineBuilder(
        F.degrees, F, γ, alg.tracker_options, alg.endgame_options,
    )

    return SolveCache(exec, builder, eg, starts, seed)
end

# ── CommonSolve.solve!: serial ─────────────────────────────────────────────

function CommonSolve.solve!(cache::SolveCache{Serial})::Result
    eg = cache.tracker
    path_results = PathResult[]
    sizehint!(path_results, length(cache.start_solutions))

    for x₀ in cache.start_solutions
        track!(eg, x₀)
        push!(path_results, PathResult(eg))
    end

    return Result(path_results, length(cache.start_solutions), cache.seed)
end

# ── CommonSolve.solve!: threaded ───────────────────────────────────────────

function CommonSolve.solve!(cache::SolveCache{Threaded})::Result
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    n_paths = length(starts)
    results = Vector{PathResult}(undef, n_paths)

    @tasks for i in eachindex(starts)
        @set ntasks = nt
        @local ws = cache.builder()
        track!(ws.tracker, starts[i])
        results[i] = PathResult(ws.tracker)
    end

    return Result(results, n_paths, cache.seed)
end

# ── Convenience: solve(F, alg, exec) ─────────────────────────────────────

"""
    solve(F::System, alg=TotalDegree(), exec=Threaded())

Solve a polynomial system using homotopy continuation.
"""
function solve(
        F::System,
        alg::TotalDegree = TotalDegree(),
        exec::AbstractExecutor = Threaded(),
    )::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg, exec))
end

function solve(F::System, exec::AbstractExecutor)::Result
    return solve(F, TotalDegree(), exec)
end

function solve(
        F::System,
        alg::Polyhedral,
        exec::AbstractExecutor = Threaded(),
    )::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg, exec))
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
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            F, starts, exec;
            start_parameters = start_parameters,
            target_parameters = target_parameters,
            seed = seed,
            tracker_options = tracker_options,
            endgame_options = endgame_options,
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
    )::SolveCache
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

    return SolveCache(exec, builder, eg, start_sols, seed)
end
