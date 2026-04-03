## Result — aggregate result from solve().

# ---------------------------------------------------------------------------
# Solution clustering — group successful paths converging to the same endpoint
# ---------------------------------------------------------------------------

"""
    _cluster_solutions(path_results; atol, rtol) -> (clusters, multiplicity)

Group successful path results by solution proximity. Returns:
- `clusters::Vector{Vector{Int}}` — groups of indices into `path_results`
- `multiplicity::Vector{Int}` — per-path multiplicity (group size, 0 for non-success)

Two solutions are considered identical if their infinity-norm distance is
≤ `max(atol, rtol * max(norm(s1), norm(s2)))`. Uses union-find to compute
connected components over the tolerance graph, so the result is transitive
and order-independent: if A≈B and B≈C then A, B, C are always in one cluster.
"""
function _cluster_solutions(
        path_results::Vector{PathResult};
        atol::Float64 = 1.0e-6,
        rtol::Float64 = 1.0e-3,
    )::Tuple{Vector{Vector{Int}}, Vector{Int}}
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

    # Build edges: O(k²) pairwise distance check
    for j in 1:k
        i1 = success_idx[j]
        sol_j = path_results[i1].solution
        norm_j = inf_norm(sol_j)
        for l in (j + 1):k
            i2 = success_idx[l]
            sol_l = path_results[i2].solution
            norm_l = inf_norm(sol_l)
            d = inf_distance(sol_j, sol_l)
            tol = max(atol, rtol * max(norm_j, norm_l))
            if d <= tol
                _union!(j, l)
            end
        end
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
        m = length(cluster)
        for idx in cluster
            multiplicity[idx] = m
        end
    end

    return (clusters, multiplicity)
end

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

struct Result
    path_results::Vector{PathResult}
    tracked_paths::Int
    seed::UInt32
    # Deduplication: clusters[i] = indices of paths converging to same solution
    clusters::Vector{Vector{Int}}
    # Per-path multiplicity (cluster size); 0 for non-success paths
    multiplicity::Vector{Int}
end

function Result(path_results::Vector{PathResult}, tracked_paths::Int, seed::UInt32)
    clusters, multiplicity = _cluster_solutions(path_results)
    return Result(path_results, tracked_paths, seed, clusters, multiplicity)
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
    n_failed = r.tracked_paths - count(is_success, r.path_results)
    if n_failed > 0
        print(io, " • ", n_failed, " paths failed\n")
    end
    n_at_inf = nat_infinity(r)
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

Multiplicity of the i-th path result (size of its cluster). Returns 0 for non-success paths.
"""
multiplicity(r::Result, i::Int)::Int = r.multiplicity[i]

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

# nreal: nonsingular real solutions only
nreal(r::Result; tol::Float64 = DEFAULT_REAL_TOL)::Int =
    count(c -> !r.path_results[first(c)].singular && is_real(r.path_results[first(c)]; tol = tol), r.clusters)
