using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar

@testset "Path tracking through public solve" begin
    @testset "max_step_size changes path resolution without changing the root" begin
        @polyvar x y
        F = System([x - 2, y - 3])
        seed = UInt32(0x1234)

        baseline = solve(
            F,
            TotalDegree(; seed, show_progress = false),
            Serial(),
        )
        constrained = solve(
            F,
            TotalDegree(;
                seed,
                tracker_options = TrackerOptions(; max_step_size = 0.01),
                show_progress = false,
            ),
            Serial(),
        )

        @test nfailed(baseline) == nfailed(constrained) == 0
        @test nsolutions(baseline) == nsolutions(constrained) == 1
        @test only(solutions(baseline)) ≈ only(solutions(constrained)) atol = 1.0e-10

        baseline_path = only(path_results(baseline))
        constrained_path = only(path_results(constrained))
        @test is_success(baseline_path)
        @test is_success(constrained_path)
        @test accepted_steps(constrained_path) > accepted_steps(baseline_path)
        @test steps(constrained_path) ==
            accepted_steps(constrained_path) + rejected_steps(constrained_path)
    end
end
