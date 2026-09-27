using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using Random: MersenneTwister

# Mixed volume of two lattice polygons from its definition,
# MV(P, Q) = area(P + Q) - area(P) - area(Q), with a monotone-chain hull.
function hull_area(points::Vector{Tuple{Int, Int}})
    pts = sort(unique(points))
    cross(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
    chain(seq) = foldl(seq; init = Tuple{Int, Int}[]) do h, p
        while length(h) >= 2 && cross(h[end - 1], h[end], p) <= 0
            pop!(h)
        end
        push!(h, p)
    end
    h = [chain(pts)[1:(end - 1)]; chain(reverse(pts))[1:(end - 1)]]
    twice = sum(h[i][1] * h[mod1(i - 1, end)][2] - h[mod1(i - 1, end)][1] * h[i][2] for i in eachindex(h))
    return abs(twice) // 2
end

function mixed_area(P::Vector{Tuple{Int, Int}}, Q::Vector{Tuple{Int, Int}})
    PQ = [(p[1] + q[1], p[2] + q[2]) for p in P for q in Q]
    return hull_area(PQ) - hull_area(P) - hull_area(Q)
end

cyclic(z) = let n = length(z)
    [
        [sum(prod(z[mod1(i + k, n)] for k in 0:(j - 1)) for i in 1:n) for j in 1:(n - 1)];
        prod(z) - 1
    ]
end

max_residual(F, x) = maximum(abs.(evaluate(F, x)))

@testset "Polyhedral public regressions" begin
    @testset "cyclic-4 high-weight cells mostly succeed" begin
        @polyvar c1 c2 c3 c4
        F = System(
            [
                c1 + c2 + c3 + c4,
                c1 * c2 + c2 * c3 + c3 * c4 + c4 * c1,
                c1 * c2 * c3 + c2 * c3 * c4 + c3 * c4 * c1 + c4 * c1 * c2,
                c1 * c2 * c3 * c4 - 1,
            ]
        )
        r = solve(F, Polyhedral(; seed = UInt32(42), show_progress = false))
        @test ntracked(r) == 16
        @test count(is_success, path_results(r)) >= 14
        @test nresults(r) > 0
        for p in path_results(r)
            is_success(p) || continue
            @test maximum(abs.(evaluate(F, solution(p)))) < 1.0e-6
        end
    end

    @testset "combined toric and coefficient phase steps" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2])
        r = solve(F, Polyhedral(; seed = UInt32(123), show_progress = false))

        @test nsolutions(r) > 0
        @test sum(steps, path_results(r)) > 0
        for p in path_results(r)
            is_success(p) || continue
            @test accepted_steps(p) >= 2
            @test maximum(abs.(evaluate(F, solution(p)))) < 1.0e-6
        end

        r2 = solve(F, Polyhedral(; seed = UInt32(123), show_progress = false))
        @test nsolutions(r2) == nsolutions(r)
        @test [accepted_steps(p) for p in path_results(r2)] ==
            [accepted_steps(p) for p in path_results(r)]
    end

    @testset "toric phase contributes materially to reported steps" begin
        @polyvar qx qy
        F = System([qx^2 + qy - 1, qx * qy - 2])
        r = solve(F, Polyhedral(; seed = UInt32(456), show_progress = false))
        for p in path_results(r)
            is_success(p) || continue
            @test accepted_steps(p) >= 10
        end
    end

    # Cyclic-n root counts are known: 16 finite roots of cyclic-4 in the torus
    # bound (mixed volume 16), and 70 isolated regular roots of cyclic-5.
    @testset "cyclic-5 attains its 70 regular roots" begin
        @polyvar z[1:5]
        F = System(cyclic(z))
        @test mixed_volume(F) == 70
        @test paths_to_track(F, Polyhedral()) == 70
        r = solve(F, Polyhedral(; seed = UInt32(42), show_progress = false))
        @test ntracked(r) == 70
        @test nsolutions(r) == 70
        @test nnonsingular(r) == 70
        @test nfailed(r) == 0
        for sol in solutions(r)
            @test max_residual(F, sol) < 1.0e-10
            # z ↦ ζz with ζ⁵ = 1 permutes the roots; every root has ∏zᵢ = 1.
            @test prod(sol) ≈ 1 atol = 1.0e-10
        end
    end

    @testset "generic sparse system: mixed volume from polygon areas" begin
        @polyvar x y
        P = [(0, 0), (3, 1), (1, 2), (0, 1)]
        Q = [(0, 0), (2, 0), (1, 3), (0, 2)]
        rng = MersenneTwister(3)
        c = randn(rng, ComplexF64, 8)
        F = System(
            [
                c[1] + c[2] * x^3 * y + c[3] * x * y^2 + c[4] * y,
                c[5] + c[6] * x^2 + c[7] * x * y^3 + c[8] * y^2,
            ]
        )
        bkk = mixed_area(P, Q)
        @test bkk == 11
        @test mixed_volume(F) == bkk
        @test paths_to_track(F, Polyhedral()) == bkk
        @test paths_to_track(F, TotalDegree()) == 16

        # Generic coefficients attain the BKK bound, and the total-degree solve,
        # which tracks 16 paths through a different start system, finds the same set.
        reference = solutions(
            solve(F, TotalDegree(; seed = UInt32(0x7d), show_progress = false), Serial()),
        )
        @test length(reference) == bkk
        for s in UInt32.(1:3)
            r = solve(F, Polyhedral(; seed = s, show_progress = false), Serial())
            @test ntracked(r) == bkk
            @test nsolutions(r) == bkk
            @test nnonsingular(r) == bkk
            for sol in solutions(r)
                @test max_residual(F, sol) < 1.0e-10
                @test minimum(ref -> maximum(abs.(sol .- ref)), reference) < 1.0e-8
            end
            for ref in reference
                @test minimum(sol -> maximum(abs.(sol .- ref)), solutions(r)) < 1.0e-8
            end
        end
    end

    @testset "same seed reproduces the solve bit for bit" begin
        @polyvar c1 c2 c3 c4
        F = System(cyclic([c1, c2, c3, c4]))
        alg = Polyhedral(; seed = UInt32(0xbeef), show_progress = false)
        r1, r2 = solve(F, alg, Serial()), solve(F, alg, Serial())
        @test isequal(solution.(path_results(r1)), solution.(path_results(r2)))
        @test [return_code(p) for p in path_results(r1)] ==
            [return_code(p) for p in path_results(r2)]
        @test steps.(path_results(r1)) == steps.(path_results(r2))
    end
end
