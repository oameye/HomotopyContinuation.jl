module Workloads

# One workload per public feature, written through the exported and `@public`
# API only, with one file per category. Each workload builds its own problem, so
# a cold run measures the whole first execution: problem construction plus the
# first solve. The package runs every workload while it precompiles, so the code
# they reach is cached native code; included from elsewhere, the file needs
# `HomotopyContinuation` bound in the including module.

using ..HomotopyContinuation
using Random: MersenneTwister, Random

const SEED = UInt32(0x1234)

function square_system(mode)
    @polyvar x y
    return System(
        [x^2 + y - 1, x * y - 0.5];
        variables = [x, y],
        compile = mode,
    )
end

function solve_total_degree(mode, executor; progress = false)
    return solve(
        square_system(mode),
        TotalDegree(; seed = SEED, show_progress = progress),
        executor,
    )
end

function solve_polyhedral(mode, executor)
    return solve(
        square_system(mode),
        Polyhedral(; seed = SEED, show_progress = false),
        executor,
    )
end

function conic_and_line()
    Random.seed!(5)
    @polyvar x y
    F = System([x^2 + y^2 - 5]; variables = [x, y])
    return F, rand_subspace(2; codim = 1)
end

include("solve.jl")
include("continuation.jl")
include("geometry.jl")
include("monodromy.jl")
include("results.jl")
include("evaluation.jl")

const CATEGORIES = (
    solve = SOLVE,
    continuation = CONTINUATION,
    geometry = GEOMETRY,
    monodromy = MONODROMY,
    results = RESULTS,
    evaluation = EVALUATION,
)

workloads() = Tuple(Iterators.flatten(values(CATEGORIES)))

run(name::Symbol) = run(Val(name))

end
