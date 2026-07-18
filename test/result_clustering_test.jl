using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: PathResult, PathResultCode, _cluster_solutions
using Random: MersenneTwister

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
        prs::Vector{PathResult}; atol::Float64 = 1.0e-6, rtol::Float64 = 1.0e-3,
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
        @test _cluster_solutions(PathResult[]) == (Vector{Int}[], Int[])
        prs = [fake_path_result(ComplexF64[1.0]; success = false)]
        clusters, mult = _cluster_solutions(prs)
        @test isempty(clusters)
        @test mult == [0]
    end

    @testset "exact duplicates cluster, distinct points do not" begin
        prs = [
            fake_path_result(ComplexF64[1.0 + 1.0im, 2.0]),
            fake_path_result(ComplexF64[1.0 + 1.0im, 2.0]),
            fake_path_result(ComplexF64[-3.0, 0.5im]),
        ]
        clusters, mult = _cluster_solutions(prs)
        @test normalize_clusters(clusters) == [[1, 2], [3]]
        @test mult == [2, 2, 1]
    end

    @testset "failed paths are skipped and keep multiplicity 0" begin
        prs = [
            fake_path_result(ComplexF64[1.0]),
            fake_path_result(ComplexF64[1.0]; success = false),
            fake_path_result(ComplexF64[1.0]),
        ]
        clusters, mult = _cluster_solutions(prs)
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
        clusters, mult = _cluster_solutions(prs)
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
        clusters, mult = _cluster_solutions(prs)
        @test normalize_clusters(clusters) == [[1, 2], [3]]
        @test mult == [2, 2, 1]
    end

    @testset "identical projection keys with distinct solutions stay separate" begin
        # All share coordinate 1 (degenerate sort key); differ in coordinate 2.
        prs = [
            fake_path_result(ComplexF64[5.0, k * 1.0]) for k in 1:20
        ]
        clusters, mult = _cluster_solutions(prs)
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
            clusters, mult = _cluster_solutions(prs)
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
        clusters, mult = _cluster_solutions(prs)
        @test length(clusters) == 1000
        @test all(==(2), mult)
    end
end
