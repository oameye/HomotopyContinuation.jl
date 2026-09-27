using Test
using HomotopyContinuation

include("minors_polys.jl")

@testset "Overdetermined systems" begin
    # x² = 1, y² = 1, xy = 1 has exactly the two solutions (1,1) and (-1,-1).
    # Squaring the system up tracks four paths; the two extra endpoints must not
    # be returned as solutions of the original system.
    @testset "TotalDegree separates genuine and excess roots ($exec)" for exec in
        (Serial(), Threaded())
        @polyvar x y
        F = System([x^2 - 1, y^2 - 1, x * y - 1])
        result = solve(F, TotalDegree(; seed = UInt32(0x42), show_progress = false), exec)

        @test ntracked(result) == 4
        @test nsolutions(result) == 2
        @test count(is_success, path_results(result)) == 2
        @test nexcess_solutions(result) + nat_infinity(result) + nfailed(result) == 2

        for s in solutions(result)
            @test maximum(abs.(evaluate(F, s))) < 1.0e-8
        end
        rsols = sort(real_solutions(result); by = first)
        @test length(rsols) == 2
        @test rsols[1] ≈ [-1.0, -1.0] atol = 1.0e-8
        @test rsols[2] ≈ [1.0, 1.0] atol = 1.0e-8
    end

    @testset "Polyhedral separates genuine and excess roots ($exec)" for exec in
        (Serial(), Threaded())
        @polyvar x y
        F = System([x^2 - 1, y^2 - 1, x * y - 1])
        result = solve(F, Polyhedral(; seed = UInt32(0x42), show_progress = false), exec)

        @test nsolutions(result) == 2
        @test count(is_success, path_results(result)) == 2
        @test all(s -> maximum(abs.(evaluate(F, s))) < 1.0e-8, solutions(result))
    end

    @testset "randomization is reproducible at a fixed seed" begin
        @polyvar x y
        F = System([x^2 - 1, y^2 - 1, x * y - 1])
        r1 = solve(F, TotalDegree(; seed = UInt32(7), show_progress = false), Serial())
        r2 = solve(F, TotalDegree(; seed = UInt32(7), show_progress = false), Serial())

        @test sort(map(first, solutions(r1)); by = real) ≈
            sort(map(first, solutions(r2)); by = real)
        @test nexcess_solutions(r1) == nexcess_solutions(r2)
    end

    @testset "singular solution of an overdetermined system is retained" begin
        @polyvar x y
        F = System([(x - 1)^2, y - 1, (x - 1) * y])
        result = solve(F, TotalDegree(; seed = UInt32(3), show_progress = false), Serial())

        @test nresults(result) >= 1
        @test any(results(result)) do r
            s = solution(r)
            abs(s[1] - 1) < 1.0e-4 && abs(s[2] - 1) < 1.0e-4 &&
                maximum(abs.(evaluate(F, s))) < 1.0e-6
        end
    end

    # 10 degree-six equations in three variables. Squaring up gives 6³ = 216
    # paths: 80 genuine roots and 136 excess roots of the randomized square system.
    @testset "3 by 5 minors classify all 216 paths" begin
        F = System(minors_polys())
        @test size(F) == (10, 3)
        result = solve(
            F,
            TotalDegree(; seed = UInt32(0x1234), show_progress = false),
            Threaded(),
        )

        @test ntracked(result) == 216
        @test count(is_success, path_results(result)) == 80
        @test nsolutions(result) == 80
        @test nexcess_solutions(result) == 136
        @test nfailed(result) == 0
    end

    @testset "underdetermined affine and projective input is rejected" begin
        @polyvar x y z
        affine = System([2.3 * x^2 + 1.2 * y^2 + 3x - 2y + 3])
        @test_throws ArgumentError solve(
            affine, TotalDegree(; seed = UInt32(2), show_progress = false), Serial(),
        )
        @test_throws ArgumentError solve(
            affine, Polyhedral(; seed = UInt32(2), show_progress = false), Serial(),
        )

        projective = System([2.3 * x^2 + 1.2 * y^2 + 3x * z])
        @test_throws "affine chart" solve(
            projective, TotalDegree(; seed = UInt32(2), show_progress = false), Serial(),
        )
        @test_throws "affine chart" solve(
            projective, Polyhedral(; seed = UInt32(2), show_progress = false), Serial(),
        )
    end

    @testset "underdetermined parameter continuation is rejected" begin
        @polyvar x y z a b
        F = System([x^2 - a]; variables = [x, y], parameters = [a, b])
        @test_throws ArgumentError solve(
            F,
            [[1.0, 1.0]],
            [1, 0],
            [2, 4],
            Continuation(; show_progress = false),
            Serial(),
        )

        F_projective = System(
            [x * y + (b - a) * z^2];
            variables = [x, y, z], parameters = [a, b],
        )
        @test_throws ArgumentError solve(
            F_projective,
            [[1.0, 1.0, 1.0]],
            [1, 0],
            [2, 4],
            Continuation(; show_progress = false),
            Serial(),
        )
    end

    @testset "square systems are unaffected" begin
        @polyvar x y
        F = System([x^2 - 1, y^2 - 4])
        result = solve(F, TotalDegree(; seed = UInt32(11), show_progress = false), Serial())
        @test nsolutions(result) == 4
        @test nexcess_solutions(result) == 0
        @test nfailed(result) == 0
    end
end
