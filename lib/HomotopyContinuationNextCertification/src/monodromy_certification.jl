## The certified duplicate check of `Monodromy`.
#
# `duplicate_check = DuplicateCheck.CERTIFIED` accepts a monodromy endpoint only if it
# certifies as a solution distinct from every one found so far. The accumulator
# below is what core's `monodromy_certified_solutions` hands back, and it holds
# one certification cache per task index, since every task certifies its own
# candidates.

mutable struct MonodromyCertifiedSolutions{
        P <: Union{Nothing, Vector{ComplexF64}},
        D <: DistinctCertifiedSolutions,
    } <: AbstractCertifiedSolutions
    # Kept for `empty!`, which rebuilds `distinct` wholesale: its interval tree of
    # certificates has no `empty!`. The system it is built for lives on `distinct`.
    const parameters::P
    const max_precision::Int
    const refine_solution::Bool
    distinct::D
    # The parameter values in the shape the diagnostics' Newton step takes them.
    const newton_params::FSVec{ComplexF64}
    # One cache per task index, `nothing` until that task certifies its first
    # candidate. Sized to the task count before a threaded solve hands them out, so a
    # task only ever reads and writes its own slot and needs no lock for it;
    # `caches_lock` covers the sizing itself.
    const caches::Vector{Union{Nothing, CertificationCache}}
    const caches_lock::ReentrantLock
end

_distinct_accumulator(
    F::System, p::Union{Nothing, Vector{ComplexF64}}, max_precision::Int,
) = DistinctCertifiedSolutions(F, p; max_precision = max_precision)

function HomotopyContinuationNext.monodromy_certified_solutions(
        F::System, p::Union{Nothing, Vector{ComplexF64}}, max_precision::Int,
        refine_solution::Bool,
    )
    distinct = _distinct_accumulator(F, p, max_precision)
    return MonodromyCertifiedSolutions(
        p, max_precision, refine_solution, distinct,
        FSVec{ComplexF64}(isnothing(p) ? ComplexF64[] : p),
        # The accumulator's own cache serves the first task, so a serial run
        # allocates none of its own. A cache is scratch for a fixed system, so
        # the caches survive `empty!`.
        _initial_caches(distinct.cache), ReentrantLock(),
    )
end

function Base.empty!(d::MonodromyCertifiedSolutions)
    d.distinct = _distinct_accumulator(
        d.distinct.system, d.parameters, d.max_precision,
    )
    return d
end

Base.show(io::IO, d::MonodromyCertifiedSolutions) = print(
    io, "MonodromyCertifiedSolutions with ", length(d.distinct), " distinct solutions",
)

function _initial_caches(first::CertificationCache)
    caches = Vector{Union{Nothing, CertificationCache}}(nothing, Threads.nthreads())
    caches[1] = first
    return caches
end

function HomotopyContinuationNext.monodromy_size_caches!(
        d::MonodromyCertifiedSolutions, ntasks::Int,
    )
    Base.@lock d.caches_lock begin
        while length(d.caches) < ntasks
            push!(d.caches, nothing)
        end
    end
    return nothing
end

function _task_cache(d::MonodromyCertifiedSolutions, tid::Int)::CertificationCache
    tid <= length(d.caches) ||
        HomotopyContinuationNext.monodromy_size_caches!(d, tid)
    cache = d.caches[tid]
    cache === nothing || return cache
    fresh = CertificationCache(d.distinct.system)
    d.caches[tid] = fresh
    return fresh
end

# The diagnostics of the certified midpoint, measured at that point rather than
# carried over from the tracked endpoint they replace: one Newton update supplies
# the update norm and the residual, and the workspace it factorized supplies the
# condition estimate, all three the quantities core's own accessors name. The
# update is discarded, so the point stays the midpoint of the proven enclosure.
function _certified_diagnostics(
        d::MonodromyCertifiedSolutions, x::Vector{ComplexF64}, cache::CertificationCache,
    )::CertifiedEndpoint
    step = _newton(
        cache.system_evaluator, cache.newton_cache, x, d.newton_params,
        0.0, 0.0, 1, false, 1.0, typemax(Int), Inf, Inf,
    )
    return (
        solution = x,
        accuracy = step.accuracy,
        residual = step.residual,
        condition = LinearAlgebra.cond(cache.newton_cache.workspace),
    )
end

# A candidate that has been certified but not yet filed: the verdict, the index of
# the solution a guaranteed duplicate was matched to, and the certificate together
# with the cache that produced it, which is the one the diagnostics of its midpoint
# are measured on. The certificate is stored exactly when the verdict is
# `CERTIFIED_DISTINCT`, so filing dispatches on the verdict.
struct MonodromyCandidate{C <: AbstractSolutionCertificate} <: AbstractCertifiedCandidate
    status::AddSolutionCode.T
    representative::Int
    certificate::Union{Nothing, C}
    cache::CertificationCache
end

function HomotopyContinuationNext.monodromy_certify_candidate(
        d::MonodromyCertifiedSolutions{P, D}, sol::Vector{ComplexF64}, tid::Int,
    ) where {P, C, D <: DistinctCertifiedSolutions{<:System, <:Any, C}}
    cache = _task_cache(d, tid)
    status, representative, cert = _certify_candidate!(
        d.distinct, sol, cache, 0, d.max_precision, d.refine_solution,
    )
    return MonodromyCandidate{C}(status, representative, cert, cache)
end

# The diagnostics are measured here rather than when the candidate was certified:
# only a candidate that files as distinct is ever reported, and the others outnumber
# it once the run saturates.
function HomotopyContinuationNext.monodromy_file_certified!(
        d::MonodromyCertifiedSolutions{P, D}, candidate::MonodromyCandidate{C},
        index::Int,
    ) where {P, C, D <: DistinctCertifiedSolutions{<:System, <:Any, C}}
    cert = candidate.certificate
    cert === nothing && return (candidate.status, candidate.representative, nothing)
    status, representative, stored = _file_certificate!(d.distinct, cert, index)
    stored === nothing && return (status, representative, nothing)
    midpoint = solution_approximation(stored)::Vector{ComplexF64}
    return (status, representative, _certified_diagnostics(d, midpoint, candidate.cache))
end
