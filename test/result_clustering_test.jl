using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: PathResult, PathResultCode, _cluster_solutions,
    _orbit_merge!, GroupActions, SymmetricGroup, Result, recluster, multiplicity,
    clusters, cluster_of
using Random: MersenneTwister

# The tolerances `Result` clusters with; the references below track them.
const ATOL = HC.DEFAULT_CLUSTER_ATOL
const RTOL = HC.DEFAULT_CLUSTER_RTOL

# Minimal synthetic PathResult: only return_code and solution matter for clustering.
function fake_path_result(sol::Vector{ComplexF64}; success::Bool = true)
    code = success ? PathResultCode.PATH_SUCCESS : PathResultCode.PATH_TERMINATED_MAX_STEPS
    return PathResult(
        code, sol, 0.0, 1.0e-12, 1.0, 1.0e-12, 1.0e-12, 1.0, 1, false, 10, 0, 0, false,
        copy(sol), 0.0, 0, ComplexF64[], Float64[], 0,
    )
end

# Reference O(k²) clustering used to cross-check the production implementation
# on random inputs (same tolerance rule, union-find replaced by BFS).
function brute_force_clusters(
        prs::Vector{PathResult}; atol::Float64 = ATOL, rtol::Float64 = RTOL,
    )
    idx = [i for i in eachindex(prs) if HC.is_success(prs[i])]
    k = length(idx)
    adj = [Int[] for _ in 1:k]
    for j in 1:k, l in (j + 1):k
        s1, s2 = prs[idx[j]].solution, prs[idx[l]].solution
        d = HC.inf_distance(s1, s2)
        tol = max(atol, rtol * max(HC.inf_norm(s1), HC.inf_norm(s2)))
        if d <= tol
            push!(adj[j], l)
            push!(adj[l], j)
        end
    end
    return connected_components(adj, idx)
end

# As above, plus an edge whenever an orbit image lands within tolerance of another
# solution. Closure is taken pairwise, stricter than a spatial index can be. Both
# directions are checked, since an action need not be an involution.
function brute_force_orbit_clusters(
        prs::Vector{PathResult}, action; atol::Float64 = ATOL, rtol::Float64 = RTOL,
    )
    idx = [i for i in eachindex(prs) if HC.is_success(prs[i])]
    k = length(idx)
    adj = [Int[] for _ in 1:k]
    orbits = [GroupActions(action)(prs[i].solution) for i in idx]
    near(w, s) = HC.inf_distance(w, s) <=
        max(atol, rtol * max(HC.inf_norm(w), HC.inf_norm(s)))
    for j in 1:k, l in (j + 1):k
        s1, s2 = prs[idx[j]].solution, prs[idx[l]].solution
        equivalent = any(w -> near(w, s2), orbits[j]) ||
            any(w -> near(w, s1), orbits[l])
        if equivalent
            push!(adj[j], l)
            push!(adj[l], j)
        end
    end
    return connected_components(adj, idx)
end

function connected_components(adj::Vector{Vector{Int}}, idx::Vector{Int})
    k = length(adj)
    seen = falses(k)
    comps = Vector{Vector{Int}}()
    for j in 1:k
        seen[j] && continue
        comp = Int[]
        stack = [j]
        seen[j] = true
        while !isempty(stack)
            v = pop!(stack)
            push!(comp, idx[v])
            for w in adj[v]
                if !seen[w]
                    seen[w] = true
                    push!(stack, w)
                end
            end
        end
        push!(comps, sort(comp))
    end
    return sort(comps)
end

normalize_clusters(clusters) = sort([sort(c) for c in clusters])

@testset "Solution clustering" begin
    @testset "empty and all-failed inputs" begin
        @test _cluster_solutions(PathResult[], ATOL, RTOL, nothing) == (Vector{Int}[], Int[])
        prs = [fake_path_result(ComplexF64[1.0]; success = false)]
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, nothing)
        @test isempty(clusters)
        @test mult == [0]
    end

    @testset "exact duplicates cluster, distinct points do not" begin
        prs = [
            fake_path_result(ComplexF64[1.0 + 1.0im, 2.0]),
            fake_path_result(ComplexF64[1.0 + 1.0im, 2.0]),
            fake_path_result(ComplexF64[-3.0, 0.5im]),
        ]
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, nothing)
        @test normalize_clusters(clusters) == [[1, 2], [3]]
        @test mult == [2, 2, 1]
    end

    @testset "failed paths are skipped and keep multiplicity 0" begin
        prs = [
            fake_path_result(ComplexF64[1.0]),
            fake_path_result(ComplexF64[1.0]; success = false),
            fake_path_result(ComplexF64[1.0]),
        ]
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, nothing)
        @test normalize_clusters(clusters) == [[1, 3]]
        @test mult == [2, 0, 2]
    end

    @testset "transitivity: chain A≈B≈C merges even when A is far from C" begin
        # atol = 1e-6; spacing 0.8e-6 chains into one component although
        # |A - C| = 1.6e-6 > atol.
        a = ComplexF64[0.0]
        b = ComplexF64[0.8e-6]
        c = ComplexF64[1.6e-6]
        prs = fake_path_result.([a, b, c])
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, nothing)
        @test length(clusters) == 1
        @test mult == [3, 3, 3]
    end

    @testset "rtol scales the tolerance for large-norm solutions" begin
        # ‖s‖ = 1e6, rtol = 1e-3 → pair tolerance 1e3; distance 1 clusters.
        prs = [
            fake_path_result(ComplexF64[1.0e6, 0.0]),
            fake_path_result(ComplexF64[1.0e6 + 1.0, 0.0]),
            fake_path_result(ComplexF64[2.0e6, 0.0]),
        ]
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, nothing)
        @test normalize_clusters(clusters) == [[1, 2], [3]]
        @test mult == [2, 2, 1]
    end

    @testset "identical projection keys with distinct solutions stay separate" begin
        # All share coordinate 1 (degenerate sort key); differ in coordinate 2.
        prs = [
            fake_path_result(ComplexF64[5.0, k * 1.0]) for k in 1:20
        ]
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, nothing)
        @test length(clusters) == 20
        @test all(==(1), mult)
    end

    @testset "random inputs match the brute-force reference" begin
        rng = MersenneTwister(1234)
        for trial in 1:20
            k = rand(rng, 1:80)
            nvars = rand(rng, 1:4)
            # Draw from a small set of centers plus tiny jitter → many true clusters
            centers = [randn(rng, ComplexF64, nvars) for _ in 1:max(1, k ÷ 4)]
            prs = PathResult[]
            for _ in 1:k
                c = centers[rand(rng, 1:length(centers))]
                jitter = 1.0e-8 * randn(rng, ComplexF64, nvars)
                push!(prs, fake_path_result(c .+ jitter; success = rand(rng) < 0.9))
            end
            clusters, mult = _cluster_solutions(prs, ATOL, RTOL, nothing)
            @test normalize_clusters(clusters) == brute_force_clusters(prs)
            for cl in clusters, i in cl
                @test mult[i] == length(cl)
            end
        end
    end

    @testset "large input smoke test (2000 paths)" begin
        rng = MersenneTwister(99)
        prs = PathResult[]
        for i in 1:1000
            s = randn(rng, ComplexF64, 6)
            push!(prs, fake_path_result(s))
            push!(prs, fake_path_result(s .+ 1.0e-9 .* randn(rng, ComplexF64, 6)))
        end
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, nothing)
        @test length(clusters) == 1000
        @test all(==(2), mult)
    end
end

@testset "Group-action clustering" begin
    sign_flip(s) = (-s,)

    @testset "an orbit becomes one cluster, multiplicity stays proximity-based" begin
        s = ComplexF64[1.0, 2.0]
        prs = [
            fake_path_result(copy(s)), fake_path_result(s .+ 1.0e-12),
            fake_path_result(-s), fake_path_result(-s .- 1.0e-12),
        ]
        # Without the action: two solutions, each of multiplicity 2.
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, nothing)
        @test normalize_clusters(clusters) == [[1, 2], [3, 4]]
        @test mult == [2, 2, 2, 2]
        # With the action: one orbit, and the multiplicities are unchanged.
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, sign_flip)
        @test normalize_clusters(clusters) == [[1, 2, 3, 4]]
        @test mult == [2, 2, 2, 2]
        @test first(only(clusters)) == 1
    end

    @testset "failed paths stay out of the orbit merge" begin
        s = ComplexF64[1.0, 2.0]
        prs = [
            fake_path_result(copy(s)), fake_path_result(-s; success = false),
            fake_path_result(-s),
        ]
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, sign_flip)
        @test normalize_clusters(clusters) == [[1, 3]]
        @test mult == [1, 0, 1]
    end

    @testset "chained actions collapse the full orbit" begin
        s = ComplexF64[1.0, 2.0]
        swap(v) = ([v[2], v[1]],)
        prs = [
            fake_path_result(copy(s)), fake_path_result(-s),
            fake_path_result(ComplexF64[2.0, 1.0]), fake_path_result(ComplexF64[-2.0, -1.0]),
        ]
        # Either generator alone splits the four points into two orbits.
        @test length(first(_cluster_solutions(prs, ATOL, RTOL, sign_flip))) == 2
        @test length(first(_cluster_solutions(prs, ATOL, RTOL, swap))) == 2
        # Together they generate the full orbit.
        clusters, mult = _cluster_solutions(prs, ATOL, RTOL, (sign_flip, swap))
        @test normalize_clusters(clusters) == [[1, 2, 3, 4]]
        @test all(==(1), mult)
    end

    @testset "one generator collapses the orbit it generates" begin
        # `rot` generates the order-4 cyclic group but returns a single image, so
        # the orbit only connects through the chain rot(s) ≈ s′, rot(s′) ≈ s″, ….
        rot(v) = (im .* v,)
        s = ComplexF64[1.0, 2.0]
        orbit = [s, im .* s, -s, -im .* s]
        for order in ([1, 2, 3, 4], [3, 1, 4, 2], [4, 3, 2, 1])
            prs = PathResult[fake_path_result(copy(orbit[k])) for k in order]
            clusters, mult = _cluster_solutions(prs, ATOL, RTOL, rot)
            @test normalize_clusters(clusters) == [[1, 2, 3, 4]]
            @test all(==(1), mult)
        end

        # Random full orbits of the same group, against the brute-force reference.
        rng = MersenneTwister(8765)
        for _ in 1:20
            nvars = rand(rng, 1:4)
            prs = PathResult[]
            for _ in 1:rand(rng, 1:6)
                c = randn(rng, ComplexF64, nvars)
                for s in (c, im .* c, -c, -im .* c)
                    jitter = 1.0e-9 * randn(rng, ComplexF64, nvars)
                    push!(prs, fake_path_result(s .+ jitter; success = rand(rng) < 0.9))
                end
            end
            clusters, _ = _cluster_solutions(prs, ATOL, RTOL, rot)
            @test normalize_clusters(clusters) == brute_force_orbit_clusters(prs, rot)
        end
    end

    @testset "tolerances are honored" begin
        s = ComplexF64[1.0, 2.0]
        prs = [fake_path_result(copy(s)), fake_path_result(-s .+ 1.0e-5)]
        # Loose enough to see through the 1e-5 offset, then too tight for it.
        @test length(first(_cluster_solutions(prs, 1.0e-4, RTOL, sign_flip))) == 1
        @test length(first(_cluster_solutions(prs, 1.0e-8, 1.0e-8, sign_flip))) == 2
    end

    @testset "random inputs match the brute-force reference" begin
        rng = MersenneTwister(4321)
        for _ in 1:20
            nvars = rand(rng, 1:4)
            centers = [randn(rng, ComplexF64, nvars) for _ in 1:rand(rng, 1:8)]
            prs = PathResult[]
            for c in centers, s in (c, -c)
                # Each orbit member appears once or twice, with tiny jitter.
                for _ in 1:rand(rng, 1:2)
                    jitter = 1.0e-9 * randn(rng, ComplexF64, nvars)
                    push!(prs, fake_path_result(s .+ jitter; success = rand(rng) < 0.9))
                end
            end
            clusters, _ = _cluster_solutions(prs, ATOL, RTOL, sign_flip)
            @test normalize_clusters(clusters) == brute_force_orbit_clusters(prs, sign_flip)
        end
    end
end

@testset "_orbit_merge!" begin
    sign_flip(s) = (-s,)
    # Paths 1-2 cluster at s, 3-4 at -s, 5 elsewhere; only reps 1, 3, 5 are queried.
    s = ComplexF64[1.0, 2.0]
    prs = [
        fake_path_result(copy(s)), fake_path_result(copy(s)),
        fake_path_result(-s), fake_path_result(-s),
        fake_path_result(ComplexF64[7.0, 9.0]),
    ]
    success_idx = collect(1:5)
    prox_root = [1, 1, 3, 3, 5]
    norms = [HC.inf_norm(pr.solution) for pr in prs]

    merged = Tuple{Int, Int}[]
    _orbit_merge!(
        (a, b) -> push!(merged, (a, b)), prs, success_idx, prox_root, norms,
        ATOL, RTOL, GroupActions(sign_flip),
    )
    # Both sides of the pair, and never a representative with itself.
    @test merged == [(1, 3), (3, 1)]

    # Without an action in play nothing is merged, whatever the layout.
    empty!(merged)
    _orbit_merge!(
        (a, b) -> push!(merged, (a, b)), prs, success_idx, prox_root, norms,
        ATOL, RTOL, GroupActions(v -> (v,)),
    )
    @test isempty(merged)

    # A 0-dimensional solution bails out instead of building a dimension-0 tree.
    empty_prs = [fake_path_result(ComplexF64[]), fake_path_result(ComplexF64[])]
    _orbit_merge!(
        (a, b) -> push!(merged, (a, b)), empty_prs, [1, 2], [1, 1], [0.0, 0.0],
        ATOL, RTOL, GroupActions(sign_flip),
    )
    @test isempty(merged)
end

@testset "recluster" begin
    sign_flip(s) = (-s,)
    swap(v) = ([v[2], v[1]],)
    # (1,2) and (2,1) are one swap orbit, (-1,-2) and (-2,-1) the other; the sign
    # flip fuses them. Path 5 duplicates path 1, so multiplicity is exercised too.
    prs = [
        fake_path_result(ComplexF64[1.0, 2.0]),
        fake_path_result(ComplexF64[2.0, 1.0]),
        fake_path_result(ComplexF64[-1.0, -2.0]),
        fake_path_result(ComplexF64[-2.0, -1.0]),
        fake_path_result(ComplexF64[1.0, 2.0] .+ 1.0e-12),
    ]
    r = Result(prs, 5, UInt32(0x1234))
    @test normalize_clusters(r.clusters) == [[1, 5], [2], [3], [4]]
    @test r.multiplicity == [2, 1, 1, 1, 2]

    r_swap = recluster(r; group_action = swap)
    @test normalize_clusters(r_swap.clusters) == [[1, 2, 5], [3, 4]]
    # Collapsing an orbit must not inflate multiplicities.
    @test r_swap.multiplicity == [2, 1, 1, 1, 2]
    @test [multiplicity(r_swap, i) for i in 1:5] == [2, 1, 1, 1, 2]
    @test [multiplicity(pr) for pr in r_swap.path_results] == [2, 1, 1, 1, 2]
    # Each representative is the lowest-numbered path of its orbit.
    @test all(cl -> first(cl) == minimum(cl), r_swap.clusters)
    # Metadata carries over.
    @test HC.seed(r_swap) == UInt32(0x1234) && HC.ntracked(r_swap) == 5

    # Chained actions, `group_actions` spelling.
    @test normalize_clusters(recluster(r; group_actions = (swap, sign_flip)).clusters) ==
        [[1, 2, 3, 4, 5]]
    # A SymmetricGroup-driven relabeling is the same action as `swap`.
    relabeling = let S₂ = SymmetricGroup(2)
        v -> map(p -> v[p], S₂)
    end
    @test normalize_clusters(recluster(r; group_action = relabeling).clusters) ==
        normalize_clusters(r_swap.clusters)

    # No action: the clustering of the original result is reproduced exactly.
    r_plain = recluster(r)
    @test normalize_clusters(r_plain.clusters) == normalize_clusters(r.clusters)
    @test r_plain.multiplicity == r.multiplicity

    # Reclustering an already reclustered result is a no-op.
    r_twice = recluster(r_swap; group_action = swap)
    @test normalize_clusters(r_twice.clusters) == normalize_clusters(r_swap.clusters)
    @test r_twice.multiplicity == r_swap.multiplicity

    # Loose: one cluster. Tighter than the 1e-12 offset: path 5 splits off path 1,
    # while the swap orbits still merge.
    @test length(recluster(r; atol = 10.0, rtol = 10.0).clusters) == 1
    @test normalize_clusters(
        recluster(r; group_action = swap, atol = 1.0e-14, rtol = 1.0e-14).clusters,
    ) == [[1, 2], [3, 4], [5]]
end

@testset "clusters and cluster_of" begin
    swap(v) = ([v[2], v[1]],)
    prs = [
        fake_path_result(ComplexF64[1.0, 2.0]),
        fake_path_result(ComplexF64[2.0, 1.0]),
        fake_path_result(ComplexF64[1.0, 2.0]; success = false),
        fake_path_result(ComplexF64[1.0, 2.0] .+ 1.0e-12),
        fake_path_result(ComplexF64[7.0, 9.0]),
    ]
    r = Result(prs, 5, UInt32(0x1234))
    @test normalize_clusters(r.clusters) == [[1, 4], [2], [5]]

    # Every successful path lands in exactly one group; the failed path in none.
    groups = clusters(r)
    @test sum(length, groups) == count(HC.is_success, HC.path_results(r))
    @test length(groups) == length(r.clusters)
    # The groups are the clusters resolved to `PathResult`s, led by the `results` rep.
    @test [[pr.solution for pr in g] for g in groups] ==
        [[r.path_results[i].solution for i in cl] for cl in r.clusters]
    @test map(first, groups) == HC.results(r)

    # The group of a path contains its duplicates.
    @test [pr.solution for pr in cluster_of(r, 1)] ==
        [prs[1].solution, prs[4].solution]
    @test cluster_of(r, 4) == cluster_of(r, 1)
    @test length(cluster_of(r, 2)) == 1
    # A failed path belongs to no group.
    @test isempty(cluster_of(r, 3))
    @test_throws BoundsError cluster_of(r, 6)
    @test_throws BoundsError cluster_of(r, 0)

    # After a recluster the groups are orbits.
    r_swap = recluster(r; group_action = swap)
    @test sort(map(length, clusters(r_swap))) == [1, 3]
    @test [pr.solution for pr in cluster_of(r_swap, 1)] ==
        [prs[1].solution, prs[2].solution, prs[4].solution]
    @test cluster_of(r_swap, 2) == cluster_of(r_swap, 1)
end
