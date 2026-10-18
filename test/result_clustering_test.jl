using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using Random: MersenneTwister

inf_dist(u, v) = maximum(abs.(u .- v))
inf_nrm(u) = maximum(abs.(u))

# Reference O(k²) clustering of successful endpoints: singular endpoints within
# max(atol, rtol·max‖·‖∞) are joined, and so are two endpoints related by one of
# the given actions under the same tolerance. Components by depth-first search.
function brute_force_clusters(r, atol, rtol; actions = ())
    prs = path_results(r)
    # With zero tolerance no two distinct endpoints merge, so is_singular reads
    # each endpoint's own flag rather than a multiplicity from an earlier clustering.
    flagged = is_singular.(path_results(recluster(r; atol = 0.0, rtol = 0.0)))
    idx = findall(is_success, prs)
    sols = [solution(prs[i]) for i in idx]
    near(u, v) = inf_dist(u, v) <= max(atol, rtol * max(inf_nrm(u), inf_nrm(v)))
    k = length(idx)
    adj = [Int[] for _ in 1:k]
    for j in 1:k, l in (j + 1):k
        geometric = (flagged[idx[j]] || flagged[idx[l]]) && near(sols[j], sols[l])
        orbit = any(
            act -> any(w -> near(w, sols[l]), act(sols[j])) ||
                any(w -> near(w, sols[j]), act(sols[l])),
            actions,
        )
        if geometric || orbit
            push!(adj[j], l)
            push!(adj[l], j)
        end
    end
    seen = falses(k)
    comps = Vector{Vector{Int}}()
    for j in 1:k
        seen[j] && continue
        comp, stack = Int[], [j]
        seen[j] = true
        while !isempty(stack)
            v = pop!(stack)
            push!(comp, idx[v])
            for w in adj[v]
                seen[w] || (seen[w] = true; push!(stack, w))
            end
        end
        push!(comps, sort(comp))
    end
    return sort(comps)
end

# Clusters of `r` as sorted lists of path indices.
cluster_indices(r) = sort([sort([path_number(p) for p in g]) for g in clusters(r)])

@testset "public solution clustering" begin
    @testset "nearby regular roots stay separate under loose reclustering tolerance" begin
        @polyvar x
        F = System([(x - 1) * (x - 101 // 100)]; variables = [x])
        raw = solve(
            F,
            TotalDegree(; seed = UInt32(0x51a7), show_progress = false),
            Serial(),
        )

        @test nsolutions(raw) == 2
        @test count(is_success, path_results(raw)) == 2
        @test all(is_nonsingular, path_results(raw))
        @test sort([real(solution(pr)[1]) for pr in path_results(raw)]) ≈ [1.0, 1.01] atol = 1.0e-8

        # The roots are much closer than this tolerance, but regular endpoints
        # are distinct solutions and must not be merged by geometric proximity.
        r = recluster(raw; atol = 0.1, rtol = 0.0)
        @test nsolutions(r) == 2
        @test sort(length.(clusters(r))) == [1, 1]
        @test all(==(1), multiplicity.(path_results(r)))
    end

    @testset "recluster groups a symmetry orbit without changing root multiplicity" begin
        @polyvar x y
        F = System([x^2 - 1, y^2 - 1]; variables = [x, y])
        r = solve(
            F,
            TotalDegree(; seed = UInt32(0x2c71), show_progress = false),
            Serial(),
        )
        @test nsolutions(r) == 4
        @test sort(length.(clusters(r))) == [1, 1, 1, 1]

        swap(v) = ([v[2], v[1]],)
        rs = recluster(r; group_action = swap)
        @test sort(length.(clusters(rs))) == [1, 1, 2]
        @test all(==(1), multiplicity.(path_results(rs)))

        paths = path_results(rs)
        i = findfirst(pr -> isapprox(solution(pr), ComplexF64[1, -1]; atol = 1.0e-8), paths)
        j = findfirst(pr -> isapprox(solution(pr), ComplexF64[-1, 1]; atol = 1.0e-8), paths)
        @test i !== nothing
        @test j !== nothing
        @test cluster_of(rs, i) == cluster_of(rs, j)
        @test length(cluster_of(rs, i)) == 2

        # Fixed points of the swap remain singleton clusters.
        k = findfirst(pr -> isapprox(solution(pr), ComplexF64[1, 1]; atol = 1.0e-8), paths)
        @test k !== nothing
        @test length(cluster_of(rs, k)) == 1

        # Reapplying the same public action is idempotent.
        twice = recluster(rs; group_action = swap)
        @test sort(length.(clusters(twice))) == [1, 1, 2]
        @test all(==(1), multiplicity.(path_results(twice)))
    end

    # Exact Gaussian-integer roots of multiplicity mₖ·nₗ. Whatever the tracker
    # makes of them, the clusters over its endpoints must equal the brute-force
    # components at every tolerance.
    @testset "random multiple roots: reclustering matches brute force" begin
        @polyvar x y
        rng = MersenneTwister(0x1234)
        for trial in 1:6
            cs = [complex(3k, rand(rng, -2:2)) for k in 1:3]
            ds = [complex(rand(rng, -2:2), 3l) for l in 1:2]
            ms = rand(rng, 1:2, 3)
            ns = rand(rng, 1:2, 2)
            F = System(
                [
                    prod((x - cs[k])^ms[k] for k in 1:3),
                    prod((y - ds[l])^ns[l] for l in 1:2),
                ]
            )
            r = solve(
                F, Polyhedral(; seed = UInt32(trial), show_progress = false), Serial(),
            )
            prs = path_results(r)
            @test ntracked(r) == sum(ms) * sum(ns)
            @test count(is_success, prs) > 0
            for (atol, rtol) in (
                    (1.0e-14, sqrt(eps())), (1.0e-6, 0.0), (1.0e-10, 1.0e-8),
                    (1.0, 0.0), (4.0, 0.0), (0.0, 0.5),
                )
                rc = recluster(r; atol = atol, rtol = rtol)
                @test cluster_indices(rc) == brute_force_clusters(r, atol, rtol)
                for g in clusters(rc)
                    @test all(p -> multiplicity(p) == length(g), g)
                end
            end
        end
    end

    @testset "a regular path onto a singular root joins its cluster" begin
        @polyvar x y
        # (1, 1) is both a total-degree start point and the double root, so one
        # path never moves and is flagged singular while the other arrives
        # through the regular tracker without the flag.
        r = solve(System([(x - 1)^2, y - 1]), TotalDegree(; seed = UInt32(1), show_progress = false))
        @test count(is_success, path_results(r)) == 2
        @test sort(length.(clusters(r))) == [2]
        @test all(is_singular, path_results(r))
        @test nsingular(r) == 1
        @test nsolutions(r) == 0
    end

    @testset "double root at x = 2 forms one cluster" begin
        @polyvar x y
        F = System([(x - 2)^2, y - 1])
        r = solve(F, TotalDegree(; seed = UInt32(1), show_progress = false), Serial())
        @test all(p -> inf_dist(solution(p), [2, 1]) < 1.0e-8, path_results(r))
        @test count(is_success, path_results(r)) == 2
        @test sort(length.(clusters(r))) == [2]
    end

    @testset "orbit clustering matches the brute-force reference" begin
        @polyvar x y
        # Double roots at the fourth roots of unity and a simple root at 3.
        rot(v) = (im .* v,)
        F = System([(x^4 - 1)^2 * (x - 3)]; variables = [x])
        r = solve(F, TotalDegree(; seed = UInt32(0x0b17), show_progress = false), Serial())
        @test nresults(r) == 5
        ro = recluster(r; group_action = rot)
        @test sort(length.(clusters(ro))) == [1, 8]
        # Grouping an orbit leaves each root's multiplicity untouched.
        @test sort(multiplicity.(path_results(ro))) == [1; fill(2, 8)]
        @test cluster_indices(ro) == brute_force_clusters(
            r, 1.0e-14, sqrt(eps()); actions = (rot,),
        )

        # [p(x), p(y)] is swap-symmetric; its roots are all pairs of roots of p.
        swap(v) = ([v[2], v[1]],)
        neg(v) = (-v,)
        G = System([(x^2 - 1)^2 * (x - 2), (y^2 - 1)^2 * (y - 2)])
        rg = solve(G, TotalDegree(; seed = UInt32(0x5a), show_progress = false), Serial())
        @test ntracked(rg) == 25
        @test nresults(rg) == 9
        for actions in ((swap,), (neg,), (swap, neg))
            rr = recluster(rg; group_actions = actions)
            @test cluster_indices(rr) == brute_force_clusters(
                    rg, 1.0e-6, 0.0; actions = actions,
                ) == cluster_indices(recluster(rg; group_actions = actions, atol = 1.0e-6, rtol = 0.0))
        end
        # Swap orbits of {±1, 2}²: 3 diagonal points and 3 off-diagonal pairs.
        @test length(clusters(recluster(rg; group_action = swap))) == 6
    end

    @testset "many endpoints, singular and regular" begin
        @polyvar z[1:5]
        # 4·3·3·3·2 = 216 paths onto 108 roots of multiplicity 2 each.
        F = System([(z[1]^2 - 1)^2, z[2]^3 - 1, z[3]^3 - 1, z[4]^3 - 1, z[5]^2 - 1])
        r = solve(F, TotalDegree(; seed = UInt32(0x99), show_progress = false))
        @test ntracked(r) == 216
        @test all(is_success, path_results(r))
        @test nresults(r) == 108
        @test all(==(2), length.(clusters(r)))
        @test all(==(2), multiplicity.(path_results(r)))
        ω = cis(2π / 3)
        exact = vec(
            [
                ComplexF64[a, b, c, d, e] for a in (-1, 1), b in (1, ω, ω^2),
                    c in (1, ω, ω^2), d in (1, ω, ω^2), e in (-1, 1)
            ],
        )
        reps = solution.(results(r))
        # Each exact root is represented exactly once.
        @test all(z -> count(v -> inf_dist(v, z) < 1.0e-6, reps) == 1, exact)
    end

    @testset "clusters and cluster_of partition the successful paths" begin
        @polyvar x
        F = System([(x - 1)^3 * (x + 2)]; variables = [x])
        r = solve(F, TotalDegree(; seed = UInt32(0x1111), show_progress = false), Serial())
        groups = clusters(r)
        @test sum(length, groups) == count(is_success, path_results(r))
        @test sort(length.(groups)) == [1, 3]
        @test map(first, groups) == results(r)
        for i in 1:ntracked(r)
            g = cluster_of(r, i)
            @test i in path_number.(g)
            @test all(j -> cluster_of(r, j) == g, path_number.(g))
        end
        @test_throws BoundsError cluster_of(r, ntracked(r) + 1)
        @test_throws BoundsError cluster_of(r, 0)
    end
end
