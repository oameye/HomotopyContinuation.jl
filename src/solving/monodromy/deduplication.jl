##########################
## Serial solve loop    ##
##########################

# Deduplication tolerances for a finished PathResult under the solver's policy.
# Shared by the serial `add!(MS, …)` and the threaded worker so the two paths
# cannot drift apart.
function _dedup_tolerances(opts::MonodromyOptions, res::PathResult)::Tuple{Float64, Float64}
    rtol = isnan(opts.unique_points_rtol) ? uniqueness_rtol(res) :
        opts.unique_points_rtol
    return opts.unique_points_atol, rtol
end

# The id of a stored solution `res` is an orbit image of, and `nothing` when there
# is none or the run keeps no equivalence classes. Call with the lock guarding the
# stored solutions held.
# Returns the id of the stored orbit duplicate, or `0` when there is none.
function _orbit_duplicate(MS::MonodromySolver, res::PathResult)::Int
    MS.options.equivalence_classes || return 0
    atol, rtol = _dedup_tolerances(MS.options, res)
    x = solution(res)
    UP = MS.unique_points
    return search_in_radius(UP, x, tolerance_radius(UP, x, atol, rtol))
end

# Certify a finished PathResult, which is everything the certified duplicate check
# can do before the caller takes the lock guarding the stored solutions. `nothing`
# under `DuplicateCheck.HEURISTIC`, which has nothing to do here; `tid` selects the
# calling task's certification cache.
certify_candidate(::HeuristicMonodromySolver, ::PathResult, ::Int = 1) = NoCandidate()

function certify_candidate(MS::CertifiedMonodromySolver, res::PathResult, tid::Int = 1)
    # Certifying an orbit image of a stored solution is wasted work. `add!` repeats
    # the search under the lock it files in, which is what settles the race with a
    # task that stored the image in between.
    if MS.options.equivalence_classes
        orbit = Base.@lock MS.unique_points_lock _orbit_duplicate(MS, res)
        iszero(orbit) || return NoCandidate()
    end
    Threads.atomic_add!(MS.statistics.certification_attempts, 1)
    return monodromy_certify_candidate(MS.certified_solutions, solution(res), tid)
end

# Certify and file in one call, for a caller that takes no lock of its own.
add!(MS::HeuristicMonodromySolver, res::PathResult, id::Int) =
    add!(MS, res, id, NoCandidate())
add!(MS::CertifiedMonodromySolver, res::PathResult, id::Int) =
    add!(MS, res, id, certify_candidate(MS, res))

# Dedup-add a finished PathResult under the solver's policy, given the candidate
# `certify_candidate` produced for it. Returns the id of the solution it represents,
# whether it was added, and the endpoint to store.
function add!(
        MS::HeuristicMonodromySolver, res::PathResult, id::Int, ::NoCandidate,
    )
    atol, rtol = _dedup_tolerances(MS.options, res)
    found, added = add!(MS.unique_points, solution(res), id; atol = atol, rtol = rtol)
    return (found, added, res)
end

# Under `DuplicateCheck.CERTIFIED` a candidate is kept only if it certifies as a
# solution distinct from every stored one, and what gets stored is the midpoint
# of its certified interval rather than the tracked endpoint.
function add!(
        MS::CertifiedMonodromySolver, res::PathResult, id::Int,
        candidate::AbstractCertifiedCandidate,
    )
    orbit = _orbit_duplicate(MS, res)
    iszero(orbit) || return (orbit, false, res)
    # Nothing was certified for this endpoint: it is an orbit image, or the target
    # count was already met when it finished.
    candidate isa NoCandidate && return (0, false, res)
    stats = MS.statistics
    status, representative, certified = monodromy_file_certified!(
        MS.certified_solutions, candidate, id,
    )
    if status == AddSolutionCode.DUPLICATE
        Threads.atomic_add!(stats.certified_duplicates, 1)
        return (representative, false, res)
    elseif status == AddSolutionCode.NOT_CERTIFIED
        Threads.atomic_add!(stats.uncertified_discards, 1)
        return (0, false, res)
    end
    accepted = _certified_endpoint(res, certified::CertifiedEndpoint)
    atol, rtol = _dedup_tolerances(MS.options, res)
    add!(MS.unique_points, solution(accepted), id; atol = atol, rtol = rtol)
    return (id, true, accepted)
end

function add_tracked_result!(
        MS::HeuristicMonodromySolver, res::PathResult, id::Int, ::NoCandidate,
        tid::Int = 1,
    )
    atol, rtol = _dedup_tolerances(MS.options, res)
    x = solution(res)
    UP = MS.unique_points
    existing = search_in_radius(UP, x, tolerance_radius(UP, x, atol, rtol))
    iszero(existing) || return (existing, false, res)

    validated = track_start!(MS.workers[tid], x)
    if !is_success(validated) || validated.singular
        return (0, false, res)
    end
    return add!(MS, validated, id, NoCandidate())
end

function add_tracked_result!(
        MS::CertifiedMonodromySolver, res::PathResult, id::Int,
        candidate::AbstractCertifiedCandidate, ::Int = 1,
    )
    return add!(MS, res, id, candidate)
end

"""
    check_start_solutions!(MS, X)

Track every provided start solution at the base parameters (`track_start!`),
deduplicate, and return the resulting `PathResult`s.
"""
function check_start_solutions!(
        MS::MonodromySolver, X::AbstractVector{<:AbstractVector},
    )::Vector{PathResult}
    ws = MS.workers[1]
    results = PathResult[]
    check = MS.options.check_startsolutions
    for x in X
        res = track_start!(ws, ComplexF64.(x))
        check && !is_success(res) && continue
        _, added, accepted = add!(MS, res, length(results) + 1)
        if added
            push!(results, accepted)
        end
    end
    return results
end
