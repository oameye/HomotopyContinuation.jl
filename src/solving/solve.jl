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
