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
    p::P
    p₀₁::P # halfway
    p₁::P
    p₂::P
end

function MonodromyLoop(
        base::AbstractVector, parameter_sampler::PS, rng::Random.AbstractRNG,
        ::Float64 = 0.0,
    ) where {PS}
    p = convert(Vector{ComplexF64}, base)
    p₁ = convert(Vector{ComplexF64}, parameter_sampler(rng, p))
    p₂ = convert(Vector{ComplexF64}, parameter_sampler(rng, p))

    # The stored halfway point is 0.5(p₁ - p), not p + 0.5(p₁ - p). It is
    # unused for vector parameters (the loop is a 3-segment chain).
    return MonodromyLoop(p, 0.5 .* (p₁ .- p), p₁, p₂)
end

function MonodromyLoop(
        base::LinearSubspace, parameter_sampler::PS, rng::Random.AbstractRNG,
        step::Float64 = _trace_step(base, PathResult[]),
    ) where {PS}
    L = convert(LinearSubspace{ComplexF64}, base)
    # The second linear space is just a translation in order to perform a
    # trace test. EQUAL SPACING of L, L₀₁, L₁ is load-bearing for it:
    # L₀₁ - L == L₁ - L₀₁ == v.
    v = LA.rmul!(LA.normalize!(randn(rng, ComplexF64, codim(L))), step)
    L₀₁ = translate(L, v, Extrinsic)
    L₁ = translate(L₀₁, v, Extrinsic)
    L₂ = convert(LinearSubspace{ComplexF64}, parameter_sampler(rng, L))

    return MonodromyLoop(L, L₀₁, L₁, L₂)
end

# Step between the three trace slices, in the extrinsic coordinates whose rows
# are normalized. Affinely the base carries the ambient scale, and a wide step
# also finds new solutions quickly. A linear base is the projective regime: the
# solutions are chart representatives and `A x = v` forces ‖x‖ ≳ ‖v‖, so a step
# wider than the points inflates them until the equations lose accuracy at their
# degree, while a narrower one stops separating the three slices. Both ends want
# the step at the scale of the points themselves, which is what the median is.
_trace_step(::Vector{ComplexF64}, ::Vector{PathResult})::Float64 = 0.0

function _trace_step(L::LinearSubspace, results::Vector{PathResult})::Float64
    is_linear(L) || return 5.0
    norms = Float64[]
    for r in results
        ν = LA.norm(solution(r))
        (isfinite(ν) && ν > 0) && push!(norms, ν)
    end
    isempty(norms) && return 1.0
    sort!(norms)
    n = length(norms)
    return isodd(n) ? norms[(n + 1) ÷ 2] : 0.5 * (norms[n ÷ 2] + norms[n ÷ 2 + 1])
end

# A `LoopTrackingJob` made self-contained: everything a consumer needs beyond its
# own `MonodromyWorkerState`. `id` and `loop_id` are echoed back in the result.
struct MonodromyJob{P}
    id::Int
    loop_id::Int
    loop::MonodromyLoop{P}
    x::Vector{ComplexF64}
    ω::Float64
    μ::Float64
    extended_precision::Bool
    collect_trace::Bool
end

function MonodromyJob(
        job::LoopTrackingJob, loop::MonodromyLoop{P}, res::PathResult,
        collect_trace::Bool,
    ) where {P}
    return MonodromyJob{P}(
        job.id, job.loop_id, loop, solution(res), res.ω, res.μ,
        res.extended_precision_used, collect_trace,
    )
end

struct MonodromyJobResult
    id::Int
    loop_id::Int
    result::PathResult
    trace::Matrix{ComplexF64}
end

##########################
## Monodromy Statistics ##
##########################

Base.@kwdef mutable struct MonodromyStatistics
    tracked_loops::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    tracking_failures::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    generated_loops::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    # All three stay 0 unless `duplicate_check = DuplicateCheck.CERTIFIED`.
    certification_attempts::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    certified_duplicates::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    uncertified_discards::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    solutions::Vector{Int} = Int[]                 # nsolutions after each finished loop generation
    permutations::Vector{Vector{Int}} = Vector{Int}[]
end

function Base.show(io::IO, S::MonodromyStatistics)
    println(io, "MonodromyStatistics")
    println(io, " • tracked_loops → ", S.tracked_loops[])
    println(io, " • tracking_failures → ", S.tracking_failures[])
    if S.certification_attempts[] > 0
        println(io, " • certification_attempts → ", S.certification_attempts[])
        println(io, " • certified_duplicates → ", S.certified_duplicates[])
        println(io, " • uncertified_discards → ", S.uncertified_discards[])
    end
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
