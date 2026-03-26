## solve — CommonSolve.jl integration for polynomial system solving.
#
# Pattern: solve(F, alg) = solve!(init(F, alg))
# F is a System (compiled polynomial system). The algorithm struct carries config.

"""
    SolveCache

Holds a pre-built tracker and start solutions, ready for `solve!` to track all paths.
Created by `CommonSolve.init`.
"""
struct SolveCache
    tracker::Tracker
    start_solutions::Vector{Vector{ComplexF64}}
    seed::UInt32
end

# ── CommonSolve.init: System + TotalDegree ────────────────────────────────

function CommonSolve.init(F::System, alg::TotalDegree)::SolveCache
    seed = alg.seed

    sys_G = _total_degree_startsystem(F.degrees)
    starts = _total_degree_solutions(F.degrees)

    rng = Random.MersenneTwister(seed)
    γ = cis(2π * rand(rng))
    H = StraightLineHomotopy(sys_G.evaluator, F.evaluator; γ = γ)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = alg.tracker_options)

    return SolveCache(tracker, starts, seed)
end

# ── CommonSolve.solve!: track all paths ──────────────────────────────────

function CommonSolve.solve!(cache::SolveCache)::Result
    tracker = cache.tracker
    path_results = PathResult[]
    sizehint!(path_results, length(cache.start_solutions))

    for x₀ in cache.start_solutions
        track!(tracker, x₀)
        push!(path_results, PathResult(tracker))
    end

    return Result(path_results, length(cache.start_solutions), cache.seed)
end

# ── Convenience: solve(F) and solve(F, alg) ──────────────────────────────

"""
    solve(F::System, alg=TotalDegree())

Solve a polynomial system using homotopy continuation.

The default algorithm is `TotalDegree()` which tracks `prod(degrees)` paths.

# Examples
```julia
@polyvar x y
F = System([x^2 + y - 1, x*y - 2])
result = solve(F)
solutions(result)
real_solutions(result)

# With explicit algorithm and options
result = solve(F, TotalDegree(; seed=UInt32(42)))
```
"""
function solve(F::System, alg::TotalDegree = TotalDegree())::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg))
end

"""
    solve(F::System, alg::Polyhedral)

Solve a polynomial system using polyhedral homotopy continuation.
Tracks `mixed_volume` paths (BKK bound), which is at most the Bezout bound.

# Examples
```julia
@polyvar x y
F = System([x^2 + y - 1, x*y - 2])
result = solve(F, Polyhedral())
solutions(result)
```
"""
function solve(F::System, alg::Polyhedral)::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg))
end

# ── Parameter homotopy: solve(F, starts; start_parameters, target_parameters) ─

"""
    solve(F::System, starts; start_parameters, target_parameters, seed, tracker_options)

Track solutions from `start_parameters` to `target_parameters` using parameter homotopy.

The system `F` must have been constructed with `parameters` — the homotopy interpolates
`p(t) = t·start_parameters + (1-t)·target_parameters` from t=1 to t=0.

# Examples
```julia
@polyvar x y a b
F = System([x^2 + a*y - 1, x*y - b]; parameters=[a, b])

# Find solutions at parameters [1, 2]
result₁ = solve(F, TotalDegree(); start_parameters=[1.0, 2.0])

# Track those solutions to new parameters [3, 4]
result₂ = solve(F, solutions(result₁);
    start_parameters=[1.0, 2.0],
    target_parameters=[3.0, 4.0],
)
```
"""
function solve(
        F::System,
        starts::AbstractVector{<:AbstractVector{<:Number}};
        start_parameters::AbstractVector{<:Number},
        target_parameters::AbstractVector{<:Number},
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            F, starts;
            start_parameters = start_parameters,
            target_parameters = target_parameters,
            seed = seed,
            tracker_options = tracker_options,
        ),
    )
end

function CommonSolve.init(
        F::System,
        starts::AbstractVector{<:AbstractVector{<:Number}};
        start_parameters::AbstractVector{<:Number},
        target_parameters::AbstractVector{<:Number},
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
    )::SolveCache
    @assert nparameters(F) > 0 "System must have parameters for parameter homotopy"
    @assert length(start_parameters) == nparameters(F) "start_parameters length must match nparameters"
    @assert length(target_parameters) == nparameters(F) "target_parameters length must match nparameters"

    sp = ComplexF64.(start_parameters)
    tp = ComplexF64.(target_parameters)
    H = CoefficientHomotopy(F.evaluator, sp, tp)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = tracker_options)

    start_sols = [Vector{ComplexF64}(ComplexF64.(s)) for s in starts]

    return SolveCache(tracker, start_sols, seed)
end
