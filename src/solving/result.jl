## Result — aggregate result from solve().

# ---------------------------------------------------------------------------
# Solution clustering — group successful paths converging to the same endpoint
# ---------------------------------------------------------------------------

"""
    _cluster_solutions(path_results, atol, rtol, group_actions) -> (clusters, multiplicity)

Group successful path results by solution proximity. Returns:
- `clusters::Vector{Vector{Int}}` — groups of indices into `path_results`
- `multiplicity::Vector{Int}` — per-path multiplicity (group size, 0 for non-success)

Two solutions are considered identical if their infinity-norm distance is
≤ `max(atol, rtol * max(norm(s1), norm(s2)))`. Uses union-find to compute
connected components over the tolerance graph, so the result is transitive
and order-independent: if A≈B and B≈C then A, B, C are always in one cluster.

Candidate pairs are found with a sort-and-window sweep instead of a full O(k²)
scan: paths are sorted by the 1-Lipschitz-bounded projection
`Re(x₁) + Im(x₁)`, and a pair at inf-distance `d` differs by at most `2d` in
that key, so only pairs within a `2W` key window (`W` = the largest possible
pair tolerance) need the exact distance check. Typical cost is O(k log k);
the worst case (all keys within one window) degrades gracefully to an
O(k²) pairwise scan.

With `group_actions`, `_orbit_merge!` additionally merges clusters sharing an orbit,
so `clusters` holds one entry per orbit. `multiplicity` is read off the proximity
clusters before that merge, so it reports the multiplicity of the root rather than
the size of its orbit.
"""
function _cluster_solutions(
        path_results::Vector{PathResult}, atol::Float64, rtol::Float64, group_actions::GA,
    )::Tuple{Vector{Vector{Int}}, Vector{Int}} where {GA}
    n = length(path_results)
    multiplicity = zeros(Int, n)
    clusters = Vector{Int}[]

    # Collect successful indices
    success_idx = Int[i for i in 1:n if is_success(path_results[i])]
    isempty(success_idx) && return (clusters, multiplicity)

    k = length(success_idx)

    # Union-find over successful paths (indexed 1:k, mapping to success_idx)
    parent = collect(1:k)
    rank = zeros(Int, k)

    function _find(x::Int)::Int
        while parent[x] != x
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        end
        return x
    end

    function _union!(a::Int, b::Int)::Nothing
        ra = _find(a)
        rb = _find(b)
        ra == rb && return nothing
        if rank[ra] < rank[rb]
            parent[ra] = rb
        elseif rank[ra] > rank[rb]
            parent[rb] = ra
        else
            parent[rb] = ra
            rank[ra] += 1
        end
        return nothing
    end

    # Precompute norms and projection keys. The key is 1-Lipschitz in each of
    # Re(x₁) and Im(x₁), so inf_distance(s1, s2) ≤ tol implies
    # |key1 - key2| ≤ 2 tol.
    norms = Vector{Float64}(undef, k)
    keys = Vector{Float64}(undef, k)
    max_norm = 0.0
    for j in 1:k
        sol = path_results[success_idx[j]].solution
        norms[j] = inf_norm(sol)
        keys[j] = isempty(sol) ? 0.0 : real(sol[1]) + imag(sol[1])
        max_norm = max(max_norm, norms[j])
    end
    # Largest tolerance any pair can have; window in key space is twice that.
    window = 2.0 * max(atol, rtol * max_norm)
    order = sortperm(keys)

    # Build edges: sorted sweep, comparing only pairs within the key window
    for a in 1:k
        j = order[a]
        sol_j = path_results[success_idx[j]].solution
        norm_j = norms[j]
        key_j = keys[j]
        for b in (a + 1):k
            l = order[b]
            keys[l] - key_j > window && break
            d = inf_distance(sol_j, path_results[success_idx[l]].solution)
            tol = max(atol, rtol * max(norm_j, norms[l]))
            if d <= tol
                _union!(j, l)
            end
        end
    end

    # Before any orbit merge, so the component sizes are the multiplicities.
    prox_root = Vector{Int}(undef, k)
    prox_size = zeros(Int, k)
    for j in 1:k
        root = _find(j)
        prox_root[j] = root
        prox_size[root] += 1
    end
    for j in 1:k
        multiplicity[success_idx[j]] = prox_size[prox_root[j]]
    end

    if group_actions !== nothing
        _orbit_merge!(
            _union!, path_results, success_idx, prox_root, norms,
            atol, rtol, group_actions,
        )
    end

    # Extract connected components
    comp = Dict{Int, Vector{Int}}()
    for j in 1:k
        root = _find(j)
        idx = success_idx[j]
        if haskey(comp, root)
            push!(comp[root], idx)
        else
            comp[root] = Int[idx]
        end
    end

    for cluster in values(comp)
        push!(clusters, cluster)
    end

    return (clusters, multiplicity)
end

"""
    _orbit_merge!(do_union!, path_results, success_idx, prox_root, norms, atol, rtol, actions)

Merge proximity clusters that lie in a common orbit of `actions`, by calling
`do_union!(a, b)` for every pair of distinct cluster representatives found
equivalent. Only representatives are indexed.

Every representative is indexed before any is queried, and each is unioned with
every representative its images land on, so `actions` need only be a generating
set of the symmetry group rather than an enumeration of whole orbits.
"""
function _orbit_merge!(
        do_union!::F, path_results::Vector{PathResult}, success_idx::Vector{Int},
        prox_root::Vector{Int}, norms::Vector{Float64},
        atol::Float64, rtol::Float64, actions::GA,
    )::Nothing where {F, GA}
    k = length(success_idx)
    d = length(path_results[success_idx[1]].solution)
    d == 0 && return nothing    # no coordinates, so the sweep already collapsed them
    tree = VoronoiTree{ComplexF64}(d; distance = InfNorm())
    reps = Int[j for j in 1:k if prox_root[j] == j]
    for j in reps
        insert!(tree, path_results[success_idx[j]].solution, j)
    end
    acts = _as_group_actions(actions)
    for j in reps
        tol = max(atol, rtol * norms[j])
        apply_actions(acts, path_results[success_idx[j]].solution) do w
            l = search_in_radius(tree, w, tol)
            if l !== nothing && l != j
                do_union!(j, l)
            end
            return false
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

struct Result
    path_results::Vector{PathResult}
    tracked_paths::Int
    seed::UInt32
    # clusters[i] = path indices of one solution, or of one orbit after `recluster`
    clusters::Vector{Vector{Int}}
    # Paths converging to the same point, independent of any orbit merge; 0 for non-success
    multiplicity::Vector{Int}
end

function Result(path_results::Vector{PathResult}, tracked_paths::Int, seed::UInt32)
    clusters, multiplicity = _cluster_solutions(
        path_results, DEFAULT_CLUSTER_ATOL, DEFAULT_CLUSTER_RTOL, nothing,
    )
    # Stamp each path with its multiplicity so `multiplicity(::PathResult)`
    # reports the cluster size without a back-reference to the `Result`.
    prs = PathResult[_with_multiplicity(pr, multiplicity[i]) for (i, pr) in enumerate(path_results)]
    return Result(prs, tracked_paths, seed, clusters, multiplicity)
end

"""
    recluster(r::Result; group_action = nothing, group_actions = nothing,
              atol = DEFAULT_CLUSTER_ATOL, rtol = DEFAULT_CLUSTER_RTOL)

Redo the solution clustering of `r` and return the reclustered [`Result`](@ref).

With `group_action` (one function) or `group_actions` (a chain of them, see
[`GroupActions`](@ref)), solutions in a common orbit are collapsed into one
cluster, so `nsolutions`, `results` and `solutions` count and return orbits rather
than individual points, represented by their lowest-numbered path.
`multiplicity` is unaffected by the collapse. The actions need only generate the
symmetry group: an orbit is collapsed whole even when each action returns a single
image, as long as the orbit's points are all present among the solutions.

Two solutions are treated as one when their infinity-norm distance is at most
`max(atol, rtol * norm(solution))`.

## Example
```julia
julia> @polyvar x y;

julia> r = solve(System([x^2 + y^2 - 5, x * y - 2]));

julia> nsolutions(r)
4

julia> nsolutions(recluster(r; group_action = s -> ([s[2], s[1]],)))
2
```
"""
function recluster(
        r::Result;
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        atol::Float64 = DEFAULT_CLUSTER_ATOL,
        rtol::Float64 = DEFAULT_CLUSTER_RTOL,
    )::Result
    clusters, multiplicity = _cluster_solutions(
        r.path_results, atol, rtol, _as_group_actions(group_actions),
    )
    prs = PathResult[
        _with_multiplicity(pr, multiplicity[i]) for (i, pr) in enumerate(r.path_results)
    ]
    return Result(prs, r.tracked_paths, r.seed, clusters, multiplicity)
end

"""
    path_results(r::Result)

The full vector of per-path [`PathResult`](@ref)s, including failures and
excess solutions. Use this for path-level diagnostics; `results(r)` returns
only the deduplicated successful solutions.
"""
path_results(r::Result)::Vector{PathResult} = r.path_results

"""
    seed(r::Result)

The random seed used to generate the start system, for reproducing the solve.
"""
seed(r::Result)::UInt32 = r.seed

"""
    ntracked(r::Result)

Total number of paths that were tracked.
"""
ntracked(r::Result)::Int = r.tracked_paths

"""
    failed(r::Result)

The path results whose tracking failed (see [`is_failed`](@ref)).
"""
failed(r::Result)::Vector{PathResult} = filter(is_failed, r.path_results)

"""
    at_infinity(r::Result)

The path results that diverged to infinity or zero (see [`is_at_infinity`](@ref)).
"""
at_infinity(r::Result)::Vector{PathResult} = filter(is_at_infinity, r.path_results)

"""
    nonsingular(r::Result; only_real = false)

The unique non-singular solutions as [`PathResult`](@ref)s.
"""
nonsingular(r::Result; only_real::Bool = false, real_tol::Float64 = DEFAULT_REAL_TOL)::Vector{PathResult} =
    results(r; only_nonsingular = true, only_real = only_real, real_tol = real_tol)

"""
    singular(r::Result; only_real = false)

The unique singular solutions as [`PathResult`](@ref)s.
"""
singular(r::Result; only_real::Bool = false, real_tol::Float64 = DEFAULT_REAL_TOL)::Vector{PathResult} =
    results(r; only_singular = true, only_real = only_real, real_tol = real_tol)

"""
    nfailed(r::Result) -> Int

Number of paths whose tracking failed.
"""
nfailed(r::Result)::Int = count(is_failed, r.path_results)

"""
    ResultStatistics

Summary counts for a [`Result`](@ref), produced by [`statistics`](@ref).
Every field is an `Int`.
"""
struct ResultStatistics
    total::Int
    nonsingular::Int
    singular::Int
    real::Int
    real_nonsingular::Int
    real_singular::Int
    at_infinity::Int
    excess_solution::Int
    failed::Int
end

"""
    statistics(r::Result; real_tol = DEFAULT_REAL_TOL) -> ResultStatistics

Aggregate solution counts for `r` (deduplicated unique solutions, plus raw
per-path failure / at-infinity / excess counts).
"""
function statistics(r::Result; real_tol::Float64 = DEFAULT_REAL_TOL)::ResultStatistics
    n_nonsingular = nnonsingular(r)
    n_singular = nsingular(r)
    n_real_nonsingular = nreal(r; tol = real_tol)
    n_real_singular = nresults(r; only_real = true, only_singular = true, real_tol = real_tol)
    return ResultStatistics(
        n_nonsingular + n_singular,
        n_nonsingular,
        n_singular,
        n_real_nonsingular + n_real_singular,
        n_real_nonsingular,
        n_real_singular,
        nat_infinity(r),
        nexcess_solutions(r),
        nfailed(r),
    )
end

"""
    _finalize_result(path_results, tracked_paths, seed, excess_checker) -> Result

Shared tail of every `solve!` method: reclassify excess solutions of an
overdetermined solve (no-op when the checker is `nothing`), then assemble
the `Result`.
"""
function _finalize_result(
        path_results::Vector{PathResult}, tracked_paths::Int, seed::UInt32,
        excess_checker::Union{ExcessSolutionChecker, Nothing},
    )::Result
    _check_excess_solutions!(path_results, excess_checker)
    return Result(path_results, tracked_paths, seed)
end

function Base.show(io::IO, r::Result)
    n_nonsing = nnonsingular(r)
    n_real = nreal(r)
    n_sing = nsingular(r)
    print(io, "Result with ", r.tracked_paths, " tracked paths\n")
    print(io, " • ", n_nonsing, " non-singular solutions (", n_real, " real)\n")
    if n_sing > 0
        print(io, " • ", n_sing, " singular solutions\n")
    end
    n_excess = nexcess_solutions(r)
    if n_excess > 0
        print(io, " • ", n_excess, " excess solutions\n")
    end
    n_at_inf = nat_infinity(r)
    n_failed = r.tracked_paths - count(is_success, r.path_results) - n_excess - n_at_inf
    if n_failed > 0
        print(io, " • ", n_failed, " paths failed\n")
    end
    if n_at_inf > 0
        print(io, " • ", n_at_inf, " paths at infinity\n")
    end
    return nothing
end

"""
    nresults(r; only_real, only_nonsingular, only_singular)

Number of unique solutions (deduplicated by proximity). This is the primary
solution count.
"""
function nresults(
        r::Result;
        only_real::Bool = false,
        only_nonsingular::Bool = false,
        only_singular::Bool = false,
        real_tol::Float64 = DEFAULT_REAL_TOL,
    )::Int
    n = 0
    for cluster in r.clusters
        rep = r.path_results[first(cluster)]
        if only_nonsingular && rep.singular
            continue
        end
        if only_singular && !rep.singular
            continue
        end
        if only_real && !is_real(rep; tol = real_tol)
            continue
        end
        n += 1
    end
    return n
end

"""
    multiplicity(r, i) -> Int

Multiplicity of the i-th path result: the number of paths that converged to the
same point. Returns 0 for non-success paths. Unaffected by an orbit merge, so it
reports the multiplicity of the solution even after [`recluster`](@ref) with a
group action.
"""
multiplicity(r::Result, i::Int)::Int = r.multiplicity[i]

"""
    clusters(r::Result) -> Vector{Vector{PathResult}}

The deduplication partition of the successful paths: one group of
[`PathResult`](@ref)s per unique solution, each led by its representative and in
the order an unfiltered [`results`](@ref) returns them. After [`recluster`](@ref)
with a group action each group is one orbit.

[`results`](@ref) returns only the representatives, `results(r;
multiple_results = true)` every member but ungrouped.
"""
clusters(r::Result)::Vector{Vector{PathResult}} =
    Vector{PathResult}[PathResult[r.path_results[i] for i in cl] for cl in r.clusters]

"""
    cluster_of(r::Result, i::Int) -> Vector{PathResult}

The group of [`PathResult`](@ref)s that path `i` was deduplicated into, `i`
included, representative first. Empty when path `i` did not succeed.

After [`recluster`](@ref) with a group action this is the orbit of solution `i`,
so `cluster_of(r, path_number(pr))` gets the symmetric partners of `pr`.
"""
function cluster_of(r::Result, i::Int)::Vector{PathResult}
    checkbounds(r.path_results, i)
    for cl in r.clusters
        if i in cl
            return PathResult[r.path_results[j] for j in cl]
        end
    end
    return PathResult[]
end

"""
    results(r; only_real, only_nonsingular, only_singular, multiple_results)

Return path results, optionally filtered. By default returns one representative per
unique solution cluster (`multiple_results=false`).
"""
function results(
        r::Result;
        only_real::Bool = false,
        only_nonsingular::Bool = false,
        only_singular::Bool = false,
        multiple_results::Bool = false,
        real_tol::Float64 = DEFAULT_REAL_TOL,
    )::Vector{PathResult}
    out = PathResult[]
    if multiple_results
        for pr in r.path_results
            is_success(pr) || continue
            only_nonsingular && pr.singular && continue
            only_singular && !pr.singular && continue
            only_real && !is_real(pr; tol = real_tol) && continue
            push!(out, pr)
        end
    else
        for cluster in r.clusters
            rep = r.path_results[first(cluster)]
            only_nonsingular && rep.singular && continue
            only_singular && !rep.singular && continue
            only_real && !is_real(rep; tol = real_tol) && continue
            push!(out, rep)
        end
    end
    return out
end

"""
    solutions(r; only_real)

Return nonsingular solutions (deduplicated). Singular solutions are excluded by
default — use `results(r; only_singular=true)` to access them.
"""
function solutions(r::Result; only_real::Bool = false, real_tol::Float64 = DEFAULT_REAL_TOL)::Vector{Vector{ComplexF64}}
    out = Vector{ComplexF64}[]
    for cluster in r.clusters
        rep = r.path_results[first(cluster)]
        rep.singular && continue
        if only_real && !is_real(rep; tol = real_tol)
            continue
        end
        push!(out, rep.solution)
    end
    return out
end

"""
    real_solutions(r; tol)

Return real nonsingular solutions (deduplicated).
"""
function real_solutions(r::Result; tol::Float64 = DEFAULT_REAL_TOL)::Vector{Vector{Float64}}
    out = Vector{Float64}[]
    for cluster in r.clusters
        rep = r.path_results[first(cluster)]
        rep.singular && continue
        if is_real(rep; tol = tol)
            push!(out, Float64.(real.(rep.solution)))
        end
    end
    return out
end

# Unique solution counts (deduplicated)
# nsolutions: nonsingular only. Use nresults for all (singular + nonsingular).
nsolutions(r::Result)::Int = nnonsingular(r)

function nsingular(r::Result)::Int
    return count(c -> r.path_results[first(c)].singular, r.clusters)
end

function nnonsingular(r::Result)::Int
    return count(c -> !r.path_results[first(c)].singular, r.clusters)
end

nat_infinity(r::Result)::Int = count(is_at_infinity, r.path_results)

"""
    nexcess_solutions(r) -> Int

Number of paths whose endpoint solves the squared-up randomized system but not
the original overdetermined system. Always 0 for square systems.
"""
nexcess_solutions(r::Result)::Int = count(is_excess_solution, r.path_results)

# nreal: nonsingular real solutions only
nreal(r::Result; tol::Float64 = DEFAULT_REAL_TOL)::Int =
    count(c -> !r.path_results[first(c)].singular && is_real(r.path_results[first(c)]; tol = tol), r.clusters)

# Points are owned copies, never aliases.
_start_points(
    starts::AbstractVector{<:AbstractVector{<:Number}},
)::Vector{Vector{ComplexF64}} = [Vector{ComplexF64}(ComplexF64.(s)) for s in starts]
_start_points(r::Result)::Vector{Vector{ComplexF64}} = [copy(s) for s in solutions(r)]
_start_points(x) = throw(
    ArgumentError(
        "start solutions must be a vector of solution vectors, a `Result`, or a " *
            "`ResultIterator`, got $(typeof(x)).",
    ),
)
