## solve — CommonSolve.jl integration for polynomial system solving.
#
# Pattern: solve(polys, alg) = solve!(init(polys, alg))
# The polynomial vector IS the problem. The algorithm struct carries config.

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

# ── CommonSolve.init: polys + TotalDegree ────────────────────────────────

function CommonSolve.init(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        alg::TotalDegree;
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )::SolveCache
    seed = alg.seed === nothing ? rand(Random.RandomDevice(), UInt32) : alg.seed

    info, eval_F = system_eval(polys; parameters = parameters, variables = variables)
    eval_G = _total_degree_startsystem(info.degrees, variables)
    starts = _total_degree_solutions(info.degrees)

    rng = Random.MersenneTwister(seed)
    γ = cis(2π * rand(rng))
    H = StraightLineHomotopy(eval_G, eval_F; γ = γ)
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

# ── Convenience: solve(polys) and solve(polys, alg) ──────────────────────

"""
    solve(polynomials; parameters=[], variables=..., kwargs...)
    solve(polynomials, algorithm; parameters=[], variables=...)

Solve a polynomial system using homotopy continuation.

The default algorithm is `TotalDegree()` which tracks `prod(degrees)` paths.

# Examples
```julia
@polyvar x y
result = solve([x^2 + y - 1, x*y - 2])
solutions(result)
real_solutions(result)

# With explicit algorithm and options
result = solve(F, TotalDegree(; seed=UInt32(42)))
```
"""
function solve(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        alg::TotalDegree = TotalDegree();
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(polys, alg; parameters = parameters, variables = variables),
    )
end
