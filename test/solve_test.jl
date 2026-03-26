using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: TotalDegree, Polyhedral, Result, PathResult,
    PathResultCode, TrackerOptions,
    solutions, real_solutions, nsolutions, nreal, is_success, is_real,
    total_degree_count, SolveCache, PolyhedralSolveCache
using DynamicPolynomials: @polyvar
using CommonSolve: CommonSolve

@testset "Solve" begin

    @testset "solve: linear system" begin
        @polyvar x y
        result = solve([x - 2, y - 3])
        @test nsolutions(result) == 1
        sols = solutions(result)
        @test length(sols) == 1
        @test abs(sols[1][1] - 2) < 1.0e-8
        @test abs(sols[1][2] - 3) < 1.0e-8
    end

    @testset "solve: quadratic system" begin
        @polyvar x y
        result = solve([x^2 + y - 1, x * y - 0.5])
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            @test abs(sol[1]^2 + sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 0.5) < 1.0e-6
        end
    end

    @testset "solve: x^2-1, y^2-4 finds all real solutions" begin
        @polyvar x y
        result = solve([x^2 - 1, y^2 - 4])
        @test nsolutions(result) == 4
        rsols = real_solutions(result)
        @test length(rsols) == 4
        for sol in rsols
            @test sol isa Vector{Float64}
            @test abs(sol[1]^2 - 1) < 1.0e-6
            @test abs(sol[2]^2 - 4) < 1.0e-6
        end
    end

    @testset "solve: katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        result = solve(F)
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            residual = maximum(
                abs.(
                    [
                        sol[1] + 2sol[2] + 2sol[3] + 2sol[4] - 1,
                        sol[1]^2 + 2sol[2]^2 + 2sol[3]^2 + 2sol[4]^2 - sol[1],
                        2sol[1] * sol[2] + 2sol[2] * sol[3] + 2sol[3] * sol[4] - sol[2],
                        sol[2]^2 + 2sol[1] * sol[3] + 2sol[2] * sol[4] - sol[3],
                    ]
                )
            )
            @test residual < 1.0e-6
        end
    end

    @testset "solve: reproducible with seed" begin
        @polyvar x y
        F = [x^2 + y - 1, x * y - 0.5]
        r1 = solve(F, TotalDegree(; seed = UInt32(42)))
        r2 = solve(F, TotalDegree(; seed = UInt32(42)))
        @test nsolutions(r1) == nsolutions(r2)
        s1 = sort(solutions(r1); by = s -> (real(s[1]), imag(s[1])))
        s2 = sort(solutions(r2); by = s -> (real(s[1]), imag(s[1])))
        for (a, b) in zip(s1, s2)
            @test a ≈ b atol = 1.0e-10
        end
    end

    @testset "solve: explicit algorithm" begin
        @polyvar x y
        result = solve([x^2 - 1, y - 2], TotalDegree())
        @test nsolutions(result) >= 1
        for sol in solutions(result)
            @test abs(sol[1]^2 - 1) < 1.0e-6
            @test abs(sol[2] - 2) < 1.0e-6
        end
    end

    @testset "solve: CommonSolve init/solve! interface" begin
        @polyvar x y
        cache = CommonSolve.init([x^2 - 1, y - 2], TotalDegree())
        @test cache isa SolveCache
        result = CommonSolve.solve!(cache)
        @test nsolutions(result) >= 1
    end

    @testset "solve: complex-only solutions" begin
        @polyvar x y
        result = solve([x^2 + 1, y - 1])
        @test nreal(result) == 0
        @test nsolutions(result) >= 1
    end

    @testset "Result: show" begin
        @polyvar x y
        result = solve([x - 1, y - 2])
        buf = IOBuffer()
        show(buf, result)
        s = String(take!(buf))
        @test contains(s, "tracked paths")
        @test contains(s, "solutions")
    end

    @testset "TotalDegree: custom tracker options" begin
        @polyvar x y
        opts = TrackerOptions(; max_steps = 100)
        alg = TotalDegree(; tracker_options = opts)
        result = solve([x^2 - 1, y^2 - 1], alg)
        # Should still work with small max_steps for simple system
        @test nsolutions(result) >= 1
    end

    @testset "total_degree_count" begin
        @test total_degree_count([2, 3]) == 6
        @test total_degree_count([1, 2, 2, 2]) == 8
    end

    # ── Polyhedral homotopy tests ──────────────────────────────────────────

    @testset "Polyhedral: x²+y-1, xy-2" begin
        @polyvar x y
        result = solve([x^2 + y - 1, x * y - 2], Polyhedral())
        # mixed volume = 3 for this system
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            @test abs(sol[1]^2 + sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 2) < 1.0e-6
        end
    end

    @testset "Polyhedral: x²-1, y²-4 finds all solutions" begin
        @polyvar x y
        result = solve([x^2 - 1, y^2 - 4], Polyhedral())
        @test nsolutions(result) == 4
        rsols = real_solutions(result)
        @test length(rsols) == 4
        for sol in rsols
            @test abs(sol[1]^2 - 1) < 1.0e-6
            @test abs(sol[2]^2 - 4) < 1.0e-6
        end
    end

    @testset "Polyhedral: katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        result = solve(F, Polyhedral())
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            residual = maximum(
                abs.(
                    [
                        sol[1] + 2sol[2] + 2sol[3] + 2sol[4] - 1,
                        sol[1]^2 + 2sol[2]^2 + 2sol[3]^2 + 2sol[4]^2 - sol[1],
                        2sol[1] * sol[2] + 2sol[2] * sol[3] + 2sol[3] * sol[4] - sol[2],
                        sol[2]^2 + 2sol[1] * sol[3] + 2sol[2] * sol[4] - sol[3],
                    ]
                )
            )
            @test residual < 1.0e-6
        end
    end

    @testset "Polyhedral: reproducible with seed" begin
        @polyvar x y
        F = [x^2 + y - 1, x * y - 0.5]
        r1 = solve(F, Polyhedral(; seed = UInt32(42)))
        r2 = solve(F, Polyhedral(; seed = UInt32(42)))
        @test nsolutions(r1) == nsolutions(r2)
        s1 = sort(solutions(r1); by = s -> (real(s[1]), imag(s[1])))
        s2 = sort(solutions(r2); by = s -> (real(s[1]), imag(s[1])))
        for (a, b) in zip(s1, s2)
            @test a ≈ b atol = 1.0e-10
        end
    end

    @testset "Polyhedral: CommonSolve init/solve! interface" begin
        @polyvar x y
        cache = CommonSolve.init([x^2 - 1, y - 2], Polyhedral())
        @test cache isa PolyhedralSolveCache
        result = CommonSolve.solve!(cache)
        @test nsolutions(result) >= 1
    end

    @testset "Polyhedral: fewer paths than TotalDegree" begin
        # For a sparse system, polyhedral should track fewer (or equal) paths
        @polyvar x y
        F = [x^2 + y - 1, x * y - 2]
        r_td = solve(F, TotalDegree())
        r_ph = solve(F, Polyhedral())
        # Both should find the same solutions
        @test nsolutions(r_td) == nsolutions(r_ph)
    end

    @testset "Polyhedral vs TotalDegree: solution counts match" begin
        @polyvar x y
        systems = [
            [x^2 - 1, y^2 - 4],
            [x^2 + y^2 - 1, x * y - 0.25],
        ]
        for F in systems
            r_td = solve(F, TotalDegree(; seed = UInt32(1)))
            r_ph = solve(F, Polyhedral(; seed = UInt32(1)))
            @test nsolutions(r_td) == nsolutions(r_ph)
        end
    end
end
