using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar

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
        @test r.tracked_paths == 16
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
end
