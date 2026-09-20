############
## Result ##
############

"""
    MonodromyResult

Contains the result of a [`Monodromy`](@ref) computation.
"""
struct MonodromyResult{P, LP} <: AbstractSolutionResult
    returncode::MonodromyCode.T
    results::Vector{PathResult}
    parameters::P
    loops::Vector{MonodromyLoop{LP}}
    statistics::MonodromyStatistics
    equivalence_classes::Bool
    duplicate_check::DuplicateCheck.T
    seed::UInt32
    # `NaN` when no trace test ran: the test only applies to subspace monodromy.
    trace::Float64
end

"""
    ParameterMonodromyResult
    SubspaceMonodromyResult

The two shapes a [`MonodromyResult`](@ref) comes in: loops based at a parameter
vector, and loops based at the linear subspace a parameter-free system was
intersected with.
"""
const ParameterMonodromyResult =
    MonodromyResult{Vector{ComplexF64}, Vector{ComplexF64}}

const SubspaceMonodromyResult =
    MonodromyResult{LinearSubspace{ComplexF64}, LinearSubspace{ComplexF64}}

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
    if !isnan(result.trace)
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
    ncertified_distinct(result::MonodromyResult)

Return the number of solutions that certified as pairwise distinct, which is `0`
unless the run used `duplicate_check = DuplicateCheck.CERTIFIED`. With
`equivalence_classes = true` these are certified distinct points, one per orbit
found, but whether two of them lie in the same orbit is still decided by distance.
"""
ncertified_distinct(r::MonodromyResult)::Int =
    r.duplicate_check == DuplicateCheck.CERTIFIED ? nresults(r) : 0

"""
    ndiscarded_uncertified(result::MonodromyResult)

Return the number of tracked endpoints that were discarded because they failed
certification, which is `0` unless the run used `duplicate_check = DuplicateCheck.CERTIFIED`.
"""
ndiscarded_uncertified(r::MonodromyResult)::Int = r.statistics.uncertified_discards[]

"""
    path_results(result::MonodromyResult)

Returns the computed [`PathResult`](@ref)s, one per distinct solution.
"""
path_results(r::MonodromyResult)::Vector{PathResult} = r.results

# Monodromy dedups during the run through `UniquePoints`, so `results` already
# holds one entry per distinct solution and every index is a representative.
_solution_indices(r::MonodromyResult) = eachindex(r.results)

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

Return the result of the trace test computed during the monodromy, or `NaN`
when no trace test ran (it applies to subspace monodromy only).
"""
trace(r::MonodromyResult)::Float64 = r.trace

"""
    permutations(r::MonodromyResult; reduced = true)

Return the permutations of the solutions that are induced by tracking over the
loops, as a matrix whose columns are permutations. If `reduced = false`, all
recorded permutations are returned; otherwise repetitions are removed.
If a solution was not tracked in a loop, the corresponding entry is 0.
"""
function permutations(r::MonodromyResult; reduced::Bool = true)::Matrix{Int}
    π = reduced ? unique(r.statistics.permutations) : r.statistics.permutations
    N = nresults(r)
    for πⱼ in π
        N = max(N, length(πⱼ))
        isempty(πⱼ) || (N = max(N, maximum(πⱼ)))
    end

    A = zeros(Int, N, length(π))
    for (j, πⱼ) in enumerate(π), i in eachindex(πⱼ)
        A[i, j] = πⱼ[i]
    end
    return A
end
