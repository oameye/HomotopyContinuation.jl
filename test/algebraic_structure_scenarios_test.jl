using Test
using HomotopyContinuation

@testset "algebraic structure scenarios" begin
    @testset "sparse binomial system realizes its BKK count" begin
        @polyvar x y
        F = System([x^3 * y - 1, x * y^2 - 1])

        @test mixed_volume(F) == 5
        result = solve(
            F,
            Polyhedral(; seed = UInt32(0x6301), show_progress = false),
            Serial(),
        )

        @test nfailed(result) == 0
        @test nsolutions(result) == 5
        @test ntracked(result) == 5
        for sol in solutions(result)
            @test maximum(abs.(evaluate(F, sol))) < 1.0e-8
        end
    end

    @testset "double root is detected as one singular solution" begin
        @polyvar x
        F = System([(x - 1)^2])
        result = solve(
            F,
            TotalDegree(; seed = UInt32(0x6302), show_progress = false),
            Serial(),
        )

        @test nfailed(result) == 0
        @test ntracked(result) == 2
        @test nsolutions(result) == 1
        @test nsingular(result) == 1
        root = only(singular(result))
        @test abs(solution(root)[1] - 1) < 1.0e-8
        @test residual(root) < 1.0e-8
    end

    @testset "swap symmetry collapses four roots to two solution orbits" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5, x * y - 2])
        result = solve(
            F,
            TotalDegree(; seed = UInt32(0x6303), show_progress = false),
            Serial(),
        )

        @test nfailed(result) == 0
        @test nsolutions(result) == 4
        orbit_result = recluster(result; group_action = s -> ([s[2], s[1]],))
        @test nsolutions(orbit_result) == 2
        @test all(sol -> maximum(abs.(evaluate(F, sol))) < 1.0e-8, solutions(orbit_result))
    end
end
