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

    @testset "a start point with a singular Jacobian fails without a solution" begin
        @polyvar x y
        G = System([x^2 - 1, y^2 - 1])
        F = System([x^2 - 4, y^2 - 9])
        # G's Jacobian diag(2x, 2y) vanishes at the origin, which is also not a
        # root of G. (1, 1) is a regular root and continues to (2, 3).
        result = solve(
            G, F, [ComplexF64[0, 0], ComplexF64[1, 1]],
            Continuation(; seed = UInt32(0x51), show_progress = false),
            Serial(),
        )
        invalid, valid = path_results(result)

        @test is_failed(invalid)
        @test !is_success(invalid)
        @test return_code(invalid) ==
            PathResultCode.PATH_TERMINATED_INVALID_START_SINGULAR_JACOBIAN
        @test return_code(valid) == PathResultCode.PATH_SUCCESS
        @test start_solution(invalid) == ComplexF64[0, 0]
        @test is_success(valid)
        @test solution(valid) ≈ ComplexF64[2, 3] atol = 1.0e-10
        @test nfailed(result) == 1
        @test nsolutions(result) == 1
        @test only(solutions(result)) ≈ ComplexF64[2, 3] atol = 1.0e-10
    end

    @testset "a non-finite start is invalid without a singular-Jacobian verdict" begin
        @polyvar x y
        G = System([x^2 - 1, y^2 - 1])
        F = System([x^2 - 4, y^2 - 9])
        result = solve(
            G, F, [ComplexF64[NaN, 1], ComplexF64[1, 0]],
            Continuation(; seed = UInt32(0x53), show_progress = false),
            Serial(),
        )
        nonfinite, singular_start = path_results(result)
        @test is_failed(nonfinite)
        @test return_code(nonfinite) == PathResultCode.PATH_TERMINATED_INVALID_START
        # G's Jacobian diag(2, 0) at (1, 0) has corank 1 and G(1, 0) ≠ 0.
        @test return_code(singular_start) ==
            PathResultCode.PATH_TERMINATED_INVALID_START_SINGULAR_JACOBIAN
        @test nsolutions(result) == 0
    end

    @testset "a path linear in t is tracked exactly in a few steps" begin
        # y(t) = t + 9(1 - t): every Taylor coefficient beyond the first vanishes.
        @polyvar y q
        F = System([y - q]; variables = [y], parameters = [q])
        result = solve(
            F, [[1.0 + 0.0im]], [1.0 + 0im], [9.0 + 0im],
            Continuation(; seed = UInt32(0x54), show_progress = false),
            Serial(),
        )
        path = only(path_results(result))
        @test is_success(path)
        @test solution(path)[1] ≈ 9 atol = 1.0e-12
        @test steps(path) < 10
    end
end
